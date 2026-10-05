"""The original encoding draw and diagnostic joint draw have different owners."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.tessera_joint_eval_panel import (
    DRAW_SCHEMA, SCHEMA, STATUS, evaluation_execution, evaluation_formats,
    load_eval_draw, make_panel, observation_status, select_evaluation, select_panel,
    validate_eval_draw_descriptor)


def test_legacy_full_draw_and_nested_deterministic_prefix():
    ids = torch.arange(512 * 4, dtype=torch.int64).reshape(512, 4)
    calibration = {'artifact_sha256': 'a' * 64, 'shape': [512, 4]}
    legacy, selected = select_panel(ids, calibration, None)
    assert legacy is ids and selected is None
    small = make_panel(ids, artifact_sha256=calibration['artifact_sha256'], seed=237, size=16)
    large = make_panel(ids, artifact_sha256=calibration['artifact_sha256'], seed=237, size=32)
    assert large['selection']['indices'][:16] == small['selection']['indices']
    assert small['selection']['indices'] != list(range(16))
    evaluated, replayed = select_panel(ids, calibration, small)
    assert evaluated.shape == (16, 4) and replayed == small
    assert calibration['shape'] == [512, 4] and torch.equal(ids, legacy)
    assert small['eval_ids_sha256'] != make_panel(ids,
        artifact_sha256=calibration['artifact_sha256'], seed=238, size=16)['eval_ids_sha256']


@pytest.mark.parametrize("field", ["indices", "eval_ids_sha256", "shape", "artifact", "seed"])
def test_panel_refuses_changed_selection_and_token_identity(field, monkeypatch, capsys):
    ids = torch.arange(48, dtype=torch.int64).reshape(12, 4)
    calibration = {'artifact_sha256': 'a'*64}
    panel = make_panel(ids, artifact_sha256=calibration['artifact_sha256'], seed=7, size=4)
    changed = copy.deepcopy(panel)
    if field == 'indices': changed['selection']['indices'][0] ^= 1
    elif field == 'eval_ids_sha256': changed[field] = 'b'*64
    elif field == 'shape': changed[field] = [5, 4]
    elif field == 'artifact': changed['calibration_input_sha256'] = 'b'*64
    else: changed['selection']['seed'] += 1
    if field == "artifact":
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
        selected, recorded = select_panel(ids, calibration, changed)
        assert torch.equal(selected, ids[changed["selection"]["indices"]])
        assert recorded is changed and "[DEV-MODE]" in capsys.readouterr().out
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
        with pytest.raises(RuntimeError, match="panel encoding calibration binding"):
            select_panel(ids, calibration, changed)
    else:
        with pytest.raises(ValueError, match="joint evaluation"):
            select_panel(ids, calibration, changed)
    ids[panel['selection']['indices'][0], 0] += 1
    with pytest.raises(ValueError, match='joint evaluation'):
        select_panel(ids, calibration, panel)


def test_pilot_handoff_and_allocator_refuse_even_observed_zero():
    from prismaquant.cost_currency import CostCurrencyError, require_run_currency
    from prismaquant.tessera_joint_allocation import bind_allocation_payload
    from test_tessera_joint_allocation import fixture
    joint, data, prepared, metadata, kwargs = fixture()
    joint['provenance']['joint_eval'] = {'status': 'diagnostic_pilot'}
    joint['provenance']['tessera_joint_anchors']['joint_eval'] = {'status': 'diagnostic_pilot'}
    name = next(iter(joint['costs']))
    joint['stats'][name]['joint_eval_observations'] = {'tokens': 0, 'calls': 0}
    joint['stats'][name]['joint_eval_status'] = 'unknown_unobserved'
    assert observation_status({'tokens': 0, 'calls': 0}) == 'unknown_unobserved'
    assert observation_status({'tokens': 0, 'calls': 1}) == 'unknown_unobserved'
    assert observation_status({'tokens': 3, 'calls': 1}) == 'observed'
    with pytest.raises(CostCurrencyError, match='separate sampled-proposal path'):
        require_run_currency(joint)
    with pytest.raises(ValueError, match='separate sampled-proposal path'):
        bind_allocation_payload(joint, data, prepared, metadata, **kwargs)


@pytest.mark.parametrize('alias', ['dotdot', 'symlink'])
def test_plan_generator_refuses_alias_of_original_output_root(tmp_path, monkeypatch, alias):
    import json
    from prismaquant import tessera_joint_eval_panel as panel

    baseline = tmp_path / 'original-output'
    baseline.mkdir()
    base_plan = tmp_path / 'base-plan.json'
    base_plan.write_text(json.dumps({
        'schema': 'prismaquant.tessera_joint_aura.plan.v1',
        'output_root': str(baseline),
        'execution': {'n_calib_samples': 4, 'calib_seqlen': 2,
                      'boundary_storage': {'directory': str(baseline / 'exact-boundaries')}},
        'calibration_input': {'path': '/fixture/tokens.safetensors', 'sha256': 'a'*64},
    }))
    monkeypatch.setattr(panel, 'load_calibration_input', lambda *_args, **_kwargs:
                        (torch.arange(8, dtype=torch.int64).reshape(4, 2),
                         {'artifact_sha256': 'a'*64}))
    if alias == 'dotdot':
        requested = baseline / '..' / baseline.name
    else:
        requested = tmp_path / 'alias'
        requested.symlink_to(baseline, target_is_directory=True)
    assert str(requested) != str(baseline)
    assert requested.resolve() == baseline.resolve()
    output = tmp_path / 'pilot-plan.json'
    with pytest.raises(ValueError, match='distinct output root'):
        panel.main(['--base-plan', str(base_plan), '--output', str(output),
                    '--output-root', str(requested), '--seed', '7', '--size', '2'])
    assert not output.exists()


def test_observer_distinguishes_invoked_exact_zero_from_unobserved():
    from prismaquant.joint_aura import JointOperatorStatisticsLease
    from prismaquant.format_registry import get_format
    a = torch.nn.Linear(2, 2, bias=False).eval()
    b = torch.nn.Linear(2, 2, bias=False).eval()
    modules = {'invoked': a, 'unobserved': b}
    specs = {name: {'FP8_E4M3': get_format('FP8_E4M3')} for name in modules}
    with JointOperatorStatisticsLease(modules, specs,
            max_statistics_bytes=2048, max_candidate_bytes=2048) as lease:
        lease.begin_probe()
        x = torch.zeros(3, 2, requires_grad=True)
        a(x).sum().backward()
        lease.finish_observations()
        diagnostics = lease.operator_diagnostics(collect_col_energy=False)
        assert diagnostics['invoked']['observed_calls'] == 1
        assert diagnostics['invoked']['observed_tokens'] == 3
        assert diagnostics['invoked']['g_trace'] == 0
        assert diagnostics['unobserved'] == {'g_trace': 0., 'observed_tokens': 0, 'observed_calls': 0}


def test_pilot_counts_are_per_probe_and_survive_resume(tmp_path, monkeypatch):
    from prismaquant import aura_cost as aura
    from test_joint_aura_streamed import _fixture, _run
    from test_joint_operator_windows import policy
    monkeypatch.setattr(aura, '_checkpoint_git_commit', lambda: '1'*40)
    panel = {'schema': 'prismaquant.tessera_joint_eval_panel.v1', 'status': 'diagnostic_pilot'}
    def run(resume):
        _, _, runner, cache = _fixture()
        return _run(runner, cache, operator_windows=policy(), checkpoint_dir=tmp_path,
                    checkpoint_identity_extra={'joint_eval': panel}, resume=resume)
    first = run(False)
    resumed = run(True)
    assert first['costs'] == resumed['costs']
    for name, stat in first['stats'].items():
        observations = stat['joint_eval_observations']
        assert observations['count_scope'] == 'summed_over_probes'
        assert observations['n_probes'] == 3
        assert len(observations['per_probe']) == 3
        assert observations['tokens'] == sum(row['tokens'] for row in observations['per_probe'])
        assert observations['calls'] == sum(row['calls'] for row in observations['per_probe'])
        assert stat['joint_eval_status'] == 'observed'
        assert resumed['stats'][name] == stat


def _calibration_artifact(tmp_path, name, *, rows, seqlen, seed, first_token=0,
                          provenance_overrides=None):
    """One genuine safetensors draw with the production provenance roster."""
    ids = (torch.arange(rows * seqlen, dtype=torch.int64)
           + first_token).reshape(rows, seqlen)
    provenance = {'source': 'wikitext-2-raw-v1/train', 'split_role': 'calibration',
                  'model': '/models/tiny', 'seed': seed,
                  'text_sha256': hashlib.sha256(b'fixture-corpus').hexdigest(),
                  'nsamples': rows, 'seqlen': seqlen, 'fit_tokens': rows * seqlen,
                  'fit_ids_sha256': hashlib.sha256(
                      ids.to(torch.int32).numpy().tobytes()).hexdigest()}
    provenance.update(provenance_overrides or {})
    path = tmp_path / name
    save_file({'calibration_ids': ids}, str(path),
              metadata={'calibration_provenance': json.dumps(provenance)})
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), ids, provenance


def _encoding_calibration(tmp_path):
    from prismaquant.calibration_data import load_calibration_input
    path, sha, ids, _ = _calibration_artifact(tmp_path, 'encoding.safetensors',
                                              rows=4, seqlen=8, seed=3)
    _, calibration = load_calibration_input(path, expected_sha256=sha,
                                            n_samples=4, seqlen=8)
    return calibration, ids


def _fresh_draw(tmp_path, encoding, *, provenance_overrides=None):
    """A genuine fresh draw with the delivered 2048-token context (2x2048)."""
    path, sha, ids, _ = _calibration_artifact(
        tmp_path, 'fisher-draw.safetensors', rows=2, seqlen=2048, seed=5,
        first_token=4096, provenance_overrides=provenance_overrides)
    draw = {'schema': DRAW_SCHEMA, 'status': STATUS,
            'encoding_calibration_input_sha256': encoding['artifact_sha256'],
            'calibration_input': {'path': str(path), 'sha256': sha},
            'shape': list(ids.shape),
            'calibration_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest()}
    return draw, ids


def _plan_config(tmp_path, encoding):
    return {'schema': 'prismaquant.tessera_joint_aura.plan.v1',
            'calibration_input': {'path': str(tmp_path / 'encoding.safetensors'),
                                  'sha256': encoding['artifact_sha256']},
            'execution': {'n_calib_samples': 4, 'calib_seqlen': 8,
                          'boundary_storage': {'directory': '/boundaries'}}}


def test_fresh_draw_descriptor_and_load_round_trip(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('PRISMAQUANT_DEV_MODE', raising=False)
    encoding, encoding_ids = _encoding_calibration(tmp_path)
    draw, fresh_ids = _fresh_draw(tmp_path, encoding)
    assert (DRAW_SCHEMA, STATUS) == ('prismaquant.tessera_joint_eval_draw.v1',
                                     'diagnostic_pilot')
    assert validate_eval_draw_descriptor(
        draw, encoding_artifact_sha256=encoding['artifact_sha256']) is draw
    ids, calibration, returned = load_eval_draw(draw, encoding_calibration=encoding)
    assert capsys.readouterr().out.count('[DEV-MODE]') == 0
    assert returned is draw
    assert ids.shape == (2, 2048) and ids.dtype == torch.int64 and ids.shape[1] == 2048
    assert torch.equal(ids, fresh_ids)
    # A different sample count than the encoding draw and the delivered
    # 2048-token context -- read from its own file bytes, not a subset,
    # reshape or concat of the old draw.
    assert draw['shape'] == [2, 2048] and list(encoding_ids.shape) == [4, 8]
    assert int(ids.min()) > int(encoding_ids.max())
    assert calibration['schema'] == 'prismaquant.calibration_input.v1'
    assert calibration['artifact_sha256'] == draw['calibration_input']['sha256']
    assert calibration['shape'] == [2, 2048] and calibration['dtype'] == 'torch.int64'
    assert calibration['calibration_sha256'] == draw['calibration_sha256']
    assert calibration['provenance']['fit_ids_sha256'] != calibration['calibration_sha256']
    assert calibration['provenance'] == {
        'source': 'wikitext-2-raw-v1/train', 'split_role': 'calibration',
        'model': '/models/tiny', 'seed': 5,
        'text_sha256': hashlib.sha256(b'fixture-corpus').hexdigest(),
        'nsamples': 2, 'seqlen': 2048, 'fit_tokens': 4096,
        'fit_ids_sha256': hashlib.sha256(
            fresh_ids.to(torch.int32).numpy().tobytes()).hexdigest()}


def test_evaluation_execution_tightens_only_a_bound_evaluation(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('PRISMAQUANT_DEV_MODE', raising=False)
    encoding, ids = _encoding_calibration(tmp_path)
    draw, _ = _fresh_draw(tmp_path, encoding)
    config = _plan_config(tmp_path, encoding)
    assert evaluation_execution(config) is config['execution']
    config['joint_eval_draw'] = draw
    bounded = evaluation_execution(config)
    assert bounded['n_calib_samples'] == 2 and bounded['calib_seqlen'] == 2048
    assert bounded['boundary_storage'] == {'directory': '/boundaries'}
    assert config['execution']['n_calib_samples'] == 4
    del config['joint_eval_draw']
    config['joint_eval'] = make_panel(ids, artifact_sha256=encoding['artifact_sha256'],
                                      seed=7, size=2)
    pilot = evaluation_execution(config)
    assert pilot['n_calib_samples'] == 2 and pilot['calib_seqlen'] == 8
    assert config['execution']['n_calib_samples'] == 4
    tampered = copy.deepcopy(config['joint_eval'])
    tampered['calibration_input_sha256'] = 'b' * 64
    config['joint_eval'] = tampered
    changed_execution = evaluation_execution(config)
    assert changed_execution["n_calib_samples"] == 2
    assert "[DEV-MODE]" in capsys.readouterr().out
    config['joint_eval'] = make_panel(ids, artifact_sha256=encoding['artifact_sha256'],
                                      seed=7, size=2)
    config['joint_eval_draw'] = draw
    with pytest.raises(ValueError, match='mutually exclusive'):
        evaluation_execution(config)
    del config['joint_eval']
    draw['encoding_calibration_input_sha256'] = 'b' * 64
    # D32: a binding mismatch is a seal, not a refusal. Dev mode stamps one
    # [DEV-MODE] line and continues with the stored descriptor; certified0
    # is the only mode that stops.
    bounded = evaluation_execution(config)
    assert bounded['n_calib_samples'] == 2 and bounded['calib_seqlen'] == 2048
    assert ('[DEV-MODE] seal old encoding calibration binding differs'
            in capsys.readouterr().out)
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', '0')
    with pytest.raises(RuntimeError, match='old encoding calibration binding differs'):
        evaluation_execution(config)


def test_select_evaluation_returns_one_triple_per_owner(tmp_path):
    encoding, ids = _encoding_calibration(tmp_path)
    draw, fresh_ids = _fresh_draw(tmp_path, encoding)
    config = _plan_config(tmp_path, encoding)
    legacy_ids, legacy_calibration, descriptor = select_evaluation(ids, encoding, config)
    assert legacy_ids is ids and legacy_calibration is encoding and descriptor is None
    config['joint_eval'] = make_panel(ids, artifact_sha256=encoding['artifact_sha256'],
                                      seed=7, size=2)
    subset, panel_calibration, panel = select_evaluation(ids, encoding, config)
    assert subset.shape == (2, 8) and panel_calibration is encoding
    assert torch.equal(subset, ids[panel['selection']['indices']])
    config['joint_eval_draw'] = draw
    with pytest.raises(ValueError, match='mutually exclusive'):
        select_evaluation(ids, encoding, config)
    del config['joint_eval']
    fresh, fresh_calibration, returned = select_evaluation(ids, encoding, config)
    assert torch.equal(fresh, fresh_ids) and fresh_calibration is not encoding
    assert returned is draw
    assert fresh_calibration['provenance']['nsamples'] == 2
    assert fresh_calibration['provenance']['seqlen'] == 2048


def _mutated_descriptor(draw, field):
    if field == 'extra_key':
        draw['note'] = 'x'
    elif field == 'missing_field':
        del draw['status']
    elif field == 'panel_schema':
        draw['schema'] = SCHEMA
    elif field == 'bad_status':
        draw['status'] = 'observed'
    elif field == 'not_a_dict':
        return 'draw'
    elif field == 'relative_path':
        draw['calibration_input']['path'] = 'fisher-draw.safetensors'
    elif field == 'dotdot_path':
        draw['calibration_input']['path'] = '/data/sub/../fisher-draw.safetensors'
    elif field == 'short_sha':
        draw['calibration_sha256'] = 'a' * 63
    elif field == 'uppercase_sha':
        draw['calibration_sha256'] = 'A' * 64
    elif field == 'string_dim':
        draw['shape'] = ['2', 2048]
    elif field == 'zero_dim':
        draw['shape'] = [0, 2048]
    elif field == 'bool_dim':
        draw['shape'] = [True, 2048]
    elif field == 'wide_shape':
        draw['shape'] = [2, 2048, 1]
    elif field == 'extra_input_key':
        draw['calibration_input']['seed'] = 5
    elif field == 'old_binding_format':
        draw['encoding_calibration_input_sha256'] = 'z' * 63
    return draw


@pytest.mark.parametrize('field', [
    'extra_key', 'missing_field', 'panel_schema', 'bad_status', 'not_a_dict',
    'relative_path', 'dotdot_path', 'short_sha', 'uppercase_sha', 'string_dim',
    'zero_dim', 'bool_dim', 'wide_shape', 'extra_input_key', 'old_binding_format'])
def test_draw_descriptor_refuses_malformed_metadata(tmp_path, field):
    encoding, _ = _encoding_calibration(tmp_path)
    draw, _ = _fresh_draw(tmp_path, encoding)
    draw = _mutated_descriptor(draw, field)
    with pytest.raises(ValueError, match='joint evaluation draw'):
        validate_eval_draw_descriptor(draw,
                                      encoding_artifact_sha256=encoding['artifact_sha256'])


#: Provenance fields whose mismatch is a seal (D32), not a refusal.
_SOFT_PROVENANCE = [
    {'source': 'pile-00/train'},
    {'model': '/models/other'},
    {'text_sha256': hashlib.sha256(b'other-corpus').hexdigest()}]


@pytest.mark.parametrize('overrides', _SOFT_PROVENANCE)
def test_dev_mode_stamps_and_continues_foreign_draw_provenance(tmp_path, monkeypatch,
                                                               capsys, overrides):
    monkeypatch.delenv('PRISMAQUANT_DEV_MODE', raising=False)
    encoding, _ = _encoding_calibration(tmp_path)
    draw, _ = _fresh_draw(tmp_path, encoding, provenance_overrides=overrides)
    ids, calibration, returned = load_eval_draw(draw, encoding_calibration=encoding)
    assert returned is draw
    assert all(calibration['provenance'][field] == value
               for field, value in overrides.items())
    assert '[DEV-MODE] seal draw provenance' in capsys.readouterr().out


@pytest.mark.parametrize('overrides', _SOFT_PROVENANCE)
def test_certified_mode_refuses_foreign_draw_provenance(tmp_path, monkeypatch, overrides):
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', '0')
    encoding, _ = _encoding_calibration(tmp_path)
    draw, _ = _fresh_draw(tmp_path, encoding, provenance_overrides=overrides)
    with pytest.raises(RuntimeError, match='draw provenance'):
        load_eval_draw(draw, encoding_calibration=encoding)


@pytest.mark.parametrize('dev_mode', ['dev', 'certified'])
@pytest.mark.parametrize('role', ['validation', 'test'])
def test_final_benchmark_split_refuses_in_both_modes(tmp_path, monkeypatch, role, dev_mode):
    if dev_mode == 'certified':
        monkeypatch.setenv('PRISMAQUANT_DEV_MODE', '0')
    else:
        monkeypatch.delenv('PRISMAQUANT_DEV_MODE', raising=False)
    encoding, _ = _encoding_calibration(tmp_path)
    draw, _ = _fresh_draw(tmp_path, encoding, provenance_overrides={'split_role': role})
    with pytest.raises(ValueError, match='sealed final panel'):
        load_eval_draw(draw, encoding_calibration=encoding)


def test_load_refuses_token_digest_drift_and_misdeclared_shape(tmp_path):
    encoding, _ = _encoding_calibration(tmp_path)
    draw, fresh_ids = _fresh_draw(tmp_path, encoding)
    drifted = copy.deepcopy(draw)
    drifted['calibration_sha256'] = hashlib.sha256(
        (fresh_ids + 1).numpy().tobytes()).hexdigest()
    with pytest.raises(ValueError, match='token identity'):
        load_eval_draw(drifted, encoding_calibration=encoding)
    for shape in ([1, 2048], [2, 1024]):  # a different row count or a narrower width
        misdeclared = copy.deepcopy(draw)
        misdeclared['shape'] = shape
        with pytest.raises(ValueError, match='differs from requested draw'):
            load_eval_draw(misdeclared, encoding_calibration=encoding)


@pytest.mark.parametrize('dev_mode', ['dev', 'certified'])
def test_load_refuses_a_mutated_draw_file(tmp_path, monkeypatch, dev_mode):
    if dev_mode == 'certified':
        monkeypatch.setenv('PRISMAQUANT_DEV_MODE', '0')
    else:
        monkeypatch.delenv('PRISMAQUANT_DEV_MODE', raising=False)
    encoding, _ = _encoding_calibration(tmp_path)
    draw, _ = _fresh_draw(tmp_path, encoding)
    assert validate_eval_draw_descriptor(
        draw, encoding_artifact_sha256=encoding['artifact_sha256']) is draw
    pinned = Path(draw['calibration_input']['path'])
    pinned.write_bytes(pinned.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        load_eval_draw(draw, encoding_calibration=encoding)


def test_evaluation_formats_selects_exact_targets_in_caller_order():
    available = {'a.weight': ['FP8_E4M3', 'NVFP4', 'INT8'], 'b.weight': ['NVFP4']}
    assert evaluation_formats({'execution': {}}, available) is available
    config = {'joint_eval_draw': {},
              'joint_eval_targets': {'b.weight': ['NVFP4'], 'a.weight': ['INT8']}}
    selected = evaluation_formats(config, available)
    assert list(selected) == ['a.weight', 'b.weight']  # caller order, not plan order
    assert selected['a.weight'] == ['INT8'] and selected['b.weight'] == ['NVFP4']
    assert available['a.weight'] == ['FP8_E4M3', 'NVFP4', 'INT8']  # nothing mutated
    panel_config = {'joint_eval': {},
                    'joint_eval_targets': {'a.weight': ['NVFP4', 'FP8_E4M3']}}
    # the list follows the available entry's order, not the plan's
    assert evaluation_formats(panel_config, available)['a.weight'] == ['FP8_E4M3', 'NVFP4']
    specs = {'a.weight': {'FP8_E4M3': 'spec-a', 'NVFP4': 'spec-b'}}
    picked = evaluation_formats({'joint_eval_draw': {},
                                 'joint_eval_targets': {'a.weight': ['NVFP4']}}, specs)
    assert picked == {'a.weight': {'NVFP4': 'spec-b'}}


@pytest.mark.parametrize('field', [
    'no_owner', 'not_a_dict', 'empty', 'non_string_qname', 'empty_qname',
    'unknown_qname', 'not_a_list', 'empty_list', 'non_string_format',
    'empty_format', 'duplicate_format', 'unknown_format'])
def test_evaluation_formats_refuse_malformed_or_unknown_targets(field):
    available = {'a.weight': ['FP8_E4M3', 'NVFP4'], 'b.weight': ['NVFP4']}
    owner = {'joint_eval_draw': {}}
    if field == 'no_owner':
        config = {'joint_eval_targets': {'a.weight': ['NVFP4']}}
    elif field == 'not_a_dict':
        config = {**owner, 'joint_eval_targets': ['a.weight']}
    elif field == 'empty':
        config = {**owner, 'joint_eval_targets': {}}
    elif field == 'non_string_qname':
        config = {**owner, 'joint_eval_targets': {1: ['NVFP4']}}
    elif field == 'empty_qname':
        config = {**owner, 'joint_eval_targets': {'': ['NVFP4']}}
    elif field == 'unknown_qname':
        config = {**owner, 'joint_eval_targets': {'c.weight': ['NVFP4']}}
    elif field == 'not_a_list':
        config = {**owner, 'joint_eval_targets': {'a.weight': 'NVFP4'}}
    elif field == 'empty_list':
        config = {**owner, 'joint_eval_targets': {'a.weight': []}}
    elif field == 'non_string_format':
        config = {**owner, 'joint_eval_targets': {'a.weight': ['NVFP4', 4]}}
    elif field == 'empty_format':
        config = {**owner, 'joint_eval_targets': {'a.weight': ['']}}
    elif field == 'duplicate_format':
        config = {**owner, 'joint_eval_targets': {'a.weight': ['NVFP4', 'NVFP4']}}
    else:
        config = {**owner, 'joint_eval_targets': {'a.weight': ['MXFP8']}}
    with pytest.raises(ValueError, match='joint evaluation'):
        evaluation_formats(config, available)
