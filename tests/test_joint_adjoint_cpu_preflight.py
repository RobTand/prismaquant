"""The same capture CLI can audit CPU startup without capture or price output."""
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant import joint_adjoint_capture as entry, joint_cost_stage_a as stage_a
from prismaquant.calibration_data import load_calibration_input
from prismaquant.layer_streaming import _build_weight_map
from prismaquant.model_profiles import detect_profile
from prismaquant.staged_tier_policy import TierPolicyRefused
from test_stage_a_head_skip import _calibration
from test_streamed_metadata_staged_reads import _activate, _deny_pool_opens
from test_strict_reader_tier_enforcement import MANIFEST, _forget_state  # noqa: F401
from test_stagea_readset_source_coverage import campaign  # noqa: F401

pytestmark = pytest.mark.own_process


def _case(tmp_path, monkeypatch, *, missing=(), extra_config=None):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    root = tmp_path / 'source'
    root.mkdir()
    config = {'model_type': 'llama', 'architectures': ['LlamaForCausalLM'],
              'hidden_size': 8, 'num_hidden_layers': 1, 'num_attention_heads': 1,
              'num_key_value_heads': 1, 'intermediate_size': 16, 'vocab_size': 8}
    config.update(extra_config or {})
    (root / 'config.json').write_text(json.dumps(config))
    (root / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {'model.layers.0.proj.weight': 'weights.safetensors'}}))
    from test_stagea_readset_source_coverage import _write_shard
    spans = _write_shard(root / 'weights.safetensors', [
        ('model.layers.0.proj.weight', 16)])
    calibration = _calibration(tmp_path)
    prepared = tmp_path / 'prepared.json'
    prepared.write_text('{}')
    paths = [prepared, calibration, root / 'config.json', root / 'model.safetensors.index.json']
    manifest = {'entries': [{'path': str(path), 'offset': 0,
                             'bytes': path.stat().st_size,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                            for path in paths]}
    start, stop = spans['model.layers.0.proj.weight']
    manifest['entries'].append({
        'path': str(root / 'weights.safetensors'), 'offset': start,
        'bytes': stop - start, 'sha256': None})
    manifest['read_plan'] = {'phases': [
        {'name': 'head', 'entry_indices': list(range(len(paths)))},
        {'name': 'forward-000', 'entry_indices': [len(paths)]}]}
    _activate(tmp_path, monkeypatch, manifest, skip={str(root / name) for name in missing})
    # Only the claim identity boundary is doubled; source selection and
    # staged byte reads are real. The CAS decoder has separate byte tests.
    monkeypatch.setattr('prismaquant.staged_lease.load_sealed_manifest',
                        lambda _digest: manifest)
    _deny_pool_opens(monkeypatch, root)
    draw = {'path': str(calibration), 'sha256': manifest['entries'][1]['sha256']}
    plan = {'model': str(root), 'calibration_input': draw,
            'execution': {'n_calib_samples': 5, 'calib_seqlen': 4}}
    # Plan admission has independent tests; here the real CLI receives these
    # already-admitted startup values and reads the actual staged inputs.
    monkeypatch.setattr('prismaquant.tessera_joint_aura.load_joint_anchor_plan',
                        lambda *_args, **_kwargs: plan)
    output = tmp_path / 'must-not-create-capture'
    args = ['--plan', str(tmp_path / 'plan.json'), '--plan-sha256', '1' * 64,
            '--prepared', str(prepared), '--prepared-sha256', manifest['entries'][0]['sha256'],
            '--data-manifest-sha256', MANIFEST, '--output-root', str(output),
            '--cpu-input-preflight']
    return plan, args, output


def test_same_cli_cpu_preflight_reports_no_capture_or_price(tmp_path, monkeypatch, capsys):
    plan, args, output = _case(tmp_path, monkeypatch)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('CPU preflight must never run the CUDA capture wrapper')
    monkeypatch.setattr(stage_a, 'run_adjoint_capture', forbidden)
    assert entry.main(args) == 0
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert report['command'] == 'cpu-input-preflight'
    assert report['capture_executed'] is False and report['price_measured'] is False
    assert report['calibration_shape'] == [5, 4]
    assert not output.exists()


@pytest.mark.parametrize('missing', ['config.json', 'model.safetensors.index.json'])
def test_cpu_preflight_catches_missing_staged_startup_metadata(
        tmp_path, monkeypatch, missing):
    _plan, args, output = _case(tmp_path, monkeypatch, missing={missing})
    with pytest.raises(TierPolicyRefused, match='metadata-readset-not-staged'):
        entry.main(args)
    assert not output.exists()


@pytest.mark.parametrize('extra_config', [{}, {'future_optional_field': 'accepted'},
                                         {'vocab_size': 1}])
def test_preflight_adds_no_metadata_or_token_refusal_to_gpu_startup_owners(
        tmp_path, monkeypatch, extra_config):
    plan, args, _output = _case(tmp_path, monkeypatch, extra_config=extra_config)
    # These are the real GPU startup owners, on real staged files. In
    # particular, do not add the separately deferred vocabulary-size guard.
    ids, _ = load_calibration_input(plan['calibration_input']['path'],
        expected_sha256=plan['calibration_input']['sha256'], n_samples=5, seqlen=4)
    assert list(ids.shape) == [5, 4]
    profile = detect_profile(plan['model'])
    assert _build_weight_map(plan['model'], profile=profile)[0]
    assert entry.main(args) == 0


def test_default_cli_still_invokes_the_gpu_wrapper(tmp_path, monkeypatch):
    _plan, args, _output = _case(tmp_path, monkeypatch)
    args.remove('--cpu-input-preflight')
    seen = []
    def wrapper(*_args, **_kwargs):
        seen.append(True)
        return {'command': 'adjoint-capture', 'passed': True}
    monkeypatch.setattr(stage_a, 'run_adjoint_capture', wrapper)
    assert entry.main(args) == 0
    assert seen == [True]


@pytest.mark.parametrize('phase_name', ['forward-000', 'chain-000'])
@pytest.mark.parametrize('defect', ['missing', 'short', 'wrong-phase'])
def test_cpu_preflight_checks_each_selected_tensor_in_its_consumption_phase(
        tmp_path, monkeypatch, phase_name, defect):
    _plan, args, output = _case(tmp_path, monkeypatch)
    from prismaquant.staged_lease import load_sealed_manifest
    manifest = load_sealed_manifest(MANIFEST)
    phase = manifest['read_plan']['phases'][1]
    phase['name'] = phase_name
    source_index = phase['entry_indices'][0]
    if defect == 'missing':
        phase['entry_indices'].clear()
    elif defect == 'short':
        manifest['entries'][source_index]['bytes'] -= 1
    else:
        phase['entry_indices'].clear()
        manifest['read_plan']['phases'][0]['entry_indices'].append(source_index)
    with pytest.raises(stage_a.AdjointIdentityRefused, match=phase_name):
        entry.main(args)
    assert not output.exists()


@pytest.mark.parametrize('completed', [False, True])
def test_production_gate_checks_the_shard_tail_in_forward_and_reverse_phases(
        campaign, completed):
    from test_stagea_readset_source_coverage import _build, _spans
    manifest = _build(campaign, _spans(campaign) if completed else None)
    config = {'model': campaign['model']}
    if completed:
        report = stage_a.audit_stage_a_source_spans(config, manifest)
        assert report['uncovered_spans'] == 0
        assert report['loader_selected_spans_checked'] == 10
        assert report['source_phases_checked'] == 5
    else:
        with pytest.raises(stage_a.AdjointIdentityRefused, match='forward-001.*b.safetensors'):
            stage_a.audit_stage_a_source_spans(config, manifest)
