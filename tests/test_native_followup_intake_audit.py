"""Native intake order and producer reuse on real tiny checkpoint bytes.

CUDA admission, tokenizer and calibration sampling use the shared CPU doubles.
These checks make no served quality, full-model or performance claim.
"""
import json
import os
import pickle
import subprocess
from pathlib import Path
import re
import sys

import pytest
import torch

from prismaquant import prepriced_cost as intake
from prismaquant import streamed_model_identity as identity
from tests.native_source_contract_fixtures import (
    producer_cli_inputs, producer_cpu_guard, producer_source,
)
from tests.test_native_retained_cost_source_admission_audit import _pipeline_reusable


def _payload(binding):
    return {'costs': {'unit': {
        'FP8_E4M3': {'predicted_dloss': .125, 'source_binding': binding},
        'BF16': {'predicted_dloss': 0., 'source_binding': binding},
    }}}


def test_all_cells_refuse_before_bulk_source_hash(tmp_path, monkeypatch):
    source, _ = producer_source(tmp_path)
    binding = identity.build_source_model_binding(source, dtype=torch.bfloat16)
    payload = _payload(binding)
    payload['costs']['unit']['BF16'].pop('source_binding')

    def no_hash(*_args, **_kwargs):
        pytest.fail('invalid later cell must refuse before full-shard hashing')

    monkeypatch.setattr(identity, 'build_source_model_binding', no_hash)
    with pytest.raises(ValueError, match='unit/BF16: native source binding is unqualified'):
        intake.validate_native_cost_source_for_model(payload, model=str(source), where='late-cell')


def test_native_intake_resolves_only_local_source(tmp_path, monkeypatch):
    source, _ = producer_source(tmp_path)
    binding = identity.build_source_model_binding(source, dtype=torch.bfloat16)
    actual = identity.resolve_source_checkpoint_path
    requests = []

    def resolve(model, *, local_files_only=False):
        requests.append(local_files_only)
        assert local_files_only, 'read-only reuse must not download source bytes'
        return actual(model, local_files_only=local_files_only)

    monkeypatch.setattr(identity, 'resolve_source_checkpoint_path', resolve)
    intake.validate_native_cost_source_for_model(_payload(binding), model=str(source), where='local-only')
    assert requests and all(requests)


def test_shared_binding_is_canonicalized_once_per_intake(tmp_path, monkeypatch):
    source, _ = producer_source(tmp_path)
    binding = identity.build_source_model_binding(source, dtype=torch.bfloat16)
    actual = identity.validate_source_model_binding
    calls = []

    def validate(value, **kwargs):
        calls.append(value)
        return actual(value, **kwargs)

    monkeypatch.setattr(identity, 'validate_source_model_binding', validate)
    intake.require_native_cost_source_binding(_payload(binding), binding, where='shared pickle object')
    assert len(calls) == 2, 'validate the expected binding and the shared cell object once each'


@pytest.mark.parametrize('joint', [False, True], ids=['ordinary', 'joint'])
def test_native_binding_refuses_current_bf16_source(tmp_path, joint):
    original = tmp_path / 'original'
    original.mkdir()
    native, _ = producer_source(original)
    binding = identity.build_source_model_binding(native, dtype=torch.bfloat16)
    source, _ = producer_source(tmp_path, native=False)
    payload = _payload(binding)
    if joint:
        from tests.test_joint_aura_streamed import _fixture, _run

        _, _, runner, cache = _fixture()
        result = _run(runner, cache)
        name = next(iter(result['costs']))
        row = result['costs'][name]['FP8_E4M3']
        row['source_binding'] = binding
        payload = {'costs': {name: {'FP8_E4M3': row}}}
    with pytest.raises(ValueError, match='native source binding.*current source is not native'):
        intake.validate_native_cost_source_for_model(payload, model=str(source), where='replaced-source')


@pytest.mark.parametrize('change', ['same', 'qname', 'format'])
def test_joint_rows_keep_their_own_lineage(tmp_path, change):
    from tests.test_joint_aura_streamed import _fixture, _run

    source, _ = producer_source(tmp_path)
    _, _, runner, cache = _fixture()
    result = _run(runner, cache)
    name = next(iter(result['costs']))
    row = result['costs'][name]['FP8_E4M3']
    fmt = 'NVFP4' if change == 'format' else 'FP8_E4M3'
    key = 'another.unit' if change == 'qname' else name
    payload = {'costs': {key: {fmt: row}}}
    if change == 'same':
        intake.validate_native_cost_source_for_model(payload, model=str(source), where='joint control')
    else:
        with pytest.raises(ValueError, match='joint source identity names another cell'):
            intake.validate_native_cost_source_for_model(payload, model=str(source), where='joint control')


def test_real_producer_rows_pass_pipeline_and_offline_stock_intake(tmp_path, monkeypatch):
    from prismaquant import aura_cost as aura, expert_empirical_cost as expert
    from prismaquant import incremental_measure_quant_cost as baseline
    from prismaquant import memory_management
    from prismaquant.model_profiles import detect_profile
    from prismaquant.production_render_cost import synthesize_production_render_cost_payload
    from prismaquant.production_weight_cache import fill_production_weight_cache
    from prismaquant.streaming_model import load_resident_source_model

    source, expected = producer_source(tmp_path)
    tokens = producer_cli_inputs(monkeypatch)
    guard = producer_cpu_guard(tmp_path, monkeypatch)
    monkeypatch.setattr(memory_management, 'CaptureMemoryGuard', lambda *_args: guard)
    binding = identity.build_source_model_binding(source, dtype=torch.bfloat16)
    names = sorted(name.removesuffix('.weight') for name in expected if name.endswith('.self_attn.o_proj.weight'))
    backfill_name = names[0].replace('o_proj', 'q_proj')
    probe_names = [*names, backfill_name]
    probe = tmp_path / 'probe.pkl'
    stats = {name: {'h_trace': 1., 'n_params': expected[name + '.weight'].numel()} for name in probe_names}
    probe.write_bytes(pickle.dumps({'stats': stats, 'meta': {}}))
    activations = tmp_path / 'activations'
    activations.mkdir()
    for name in probe_names:
        width = expected[name + '.weight'].shape[-1]
        torch.save({'inputs': torch.ones(4, width), 'name': name},
                   activations / (baseline.ActivationIndex._FNAME_SUB.sub('__', name) + '.pt'))
    base_path = tmp_path / 'baseline.pkl'
    monkeypatch.setattr(sys, 'argv', ['incremental_measure_quant_cost', '--model', str(source),
        '--probe', str(probe), '--activation-cache-dir', str(activations),
        '--work-dir', str(tmp_path / 'baseline-work'), '--output', str(base_path),
        '--device', 'cpu', '--dtype', 'bf16', '--mode', 'unbatched',
        '--formats', 'FP8_E4M3,BF16', '--cost-mode', 'local', '--threads', '1',
        '--no-include-mtp', '--no-include-visual', '--no-include-lm-head', '--no-watchdog'])
    baseline.main()
    base = pickle.loads(base_path.read_bytes())
    before = base_path.read_bytes()
    assert set(probe_names) <= base['costs'].keys()
    model = load_resident_source_model(str(source), device=torch.device('cpu'), dtype=torch.bfloat16,
        requires_grad=True, offload_folder=str(tmp_path / 'resident'), resource_check=guard.check,
        attn_implementation='eager')
    aura_payload = aura.compute_aura_cost(model, tokens, ['FP8_E4M3', 'BF16'],
        n_probes=1, n_linear_chunks=1, min_free_gib=0,
        unit_filter=re.escape(names[0]) + '$', allow_packed_expert_omission=True,
        profile=detect_profile(str(source)), checkpoint_identity_extra={'source_binding': binding})
    cache = fill_production_weight_cache(model, tokens, [names[1]], formats=['FP8_E4M3'],
        levers={'gptq': False, 'scale_sweep': False}, source_binding=binding,
        max_act_rows=4, progress=False)
    render_base = {**base, 'costs': {names[1]: base['costs'][names[1]]}}
    rendered = synthesize_production_render_cost_payload(cache, render_base, score_field='weight_mse')
    dense = {**aura_payload, 'costs': {**aura_payload['costs'], **rendered['costs']}}
    dense['provenance'].update(model=str(source), cost_mode='aura')
    dense_path = tmp_path / 'dense.pkl'
    dense_path.write_bytes(pickle.dumps(dense))
    monkeypatch.setenv('PRISMAQUANT_EXPERT_EVAL_DRIVER', 'forked')
    output = tmp_path / 'hybrid.pkl'
    assert expert.main(['--model', str(source), '--output', str(output), '--device', 'cpu',
        '--dataset', 'synthetic-token-draw', '--formats', 'FP8_E4M3,BF16', '--max-units', '1',
        '--streaming', '--checkpoint-dir', str(tmp_path / 'expert-journal'),
        '--merge-base', str(dense_path), '--backfill-base', str(base_path), '--cost-mode', 'aura']) == 0
    payload = pickle.loads(output.read_bytes())
    assert payload['costs'][names[0]] == aura_payload['costs'][names[0]]
    assert payload['costs'][names[1]] == rendered['costs'][names[1]]
    assert payload['costs'][backfill_name] == base['costs'][backfill_name]
    assert len(payload['costs']) > len(probe_names), 'include real empirical expert rows'
    result = _pipeline_reusable(output, source)
    assert result.returncode == 0, result.stdout + result.stderr
    intake.require_native_cost_source_binding(payload, binding, where='offline stock expert input')
    assert base_path.read_bytes() == before
    (source / 'foreign_policy.py').write_text('POLICY = 2\n')
    foreign = identity.build_source_model_binding(source, dtype=torch.bfloat16)
    (source / 'foreign_policy.py').unlink()
    payload['costs'][backfill_name]['FP8_E4M3']['source_binding'] = foreign
    output.write_bytes(pickle.dumps(payload))
    result = _pipeline_reusable(output, source)
    assert result.returncode != 0 and 'differs from the current checkpoint/decoder' in result.stderr
    with pytest.raises(ValueError, match='differs from the harvest'):
        intake.require_native_cost_source_binding(payload, binding, where='offline stock expert input')


@pytest.mark.parametrize("layers_per_shard", ["1", "auto"])
def test_native_baseline_loads_the_snapshot_it_bound(tmp_path, monkeypatch, layers_per_shard):
    """Use real metadata, source delivery and measurement after Hub resolution."""
    import huggingface_hub
    from prismaquant import incremental_measure_quant_cost as baseline

    source, expected = producer_source(tmp_path)
    producer_cli_inputs(monkeypatch)
    names = sorted(name.removesuffix('.weight') for name in expected
                   if name.endswith('.self_attn.o_proj.weight')) + ['lm_head']
    probe = tmp_path / 'hub-probe.pkl'
    probe.write_bytes(pickle.dumps({'stats': {
        name: {'h_trace': 1., 'n_params': expected[name + '.weight'].numel()}
        for name in names}, 'meta': {}}))
    activations = tmp_path / 'hub-activations'
    activations.mkdir()
    for name in names:
        torch.save({'inputs': torch.ones(4, expected[name + '.weight'].shape[-1]), 'name': name},
                   activations / (baseline.ActivationIndex._FNAME_SUB.sub('__', name) + '.pt'))
    hub_id = 'contract-fixture/native-baseline'
    snapshots = []

    def snapshot(**kwargs):
        assert not snapshots, 'native baseline must load the bound snapshot without resolving the Hub reference again'
        snapshots.append(kwargs['repo_id'])
        return str(source)

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    output = tmp_path / 'hub-baseline.pkl'
    monkeypatch.setattr(sys, 'argv', ['incremental_measure_quant_cost', '--model', hub_id,
        '--probe', str(probe), '--activation-cache-dir', str(activations),
        '--work-dir', str(tmp_path / 'hub-work'), '--output', str(output),
        '--layers-per-shard', layers_per_shard,
        '--device', 'cpu', '--dtype', 'bf16', '--mode', 'unbatched', '--formats', 'FP8_E4M3',
        '--cost-mode', 'local', '--threads', '1', '--no-include-mtp', '--no-include-visual',
        '--no-watchdog'])
    baseline.main()
    assert snapshots == [hub_id]
    payload = pickle.loads(output.read_bytes())
    binding = identity.build_source_model_binding(source, dtype=torch.bfloat16)
    assert set(payload['costs']) == set(names)
    for rows in payload['costs'].values():
        for row in rows.values():
            assert row['source_binding'] == binding


@pytest.mark.parametrize("mode", ["local", "aura", "production-render-score"])
@pytest.mark.parametrize("state", ["unbound", "current", "foreign", "bf16-unbound"])
def test_baseline_reuse_checks_source_in_every_cost_mode(tmp_path, mode, state):
    source, _ = producer_source(tmp_path, native=state != "bf16-unbound")
    binding = identity.build_source_model_binding(source, dtype=torch.bfloat16)
    if state == "foreign":
        (source / "foreign.py").write_text("REVISION = 1\n")
        binding = identity.build_source_model_binding(source, dtype=torch.bfloat16)
        (source / "foreign.py").unlink()
    payload = _payload(binding)
    payload['provenance'] = {'cost_mode': 'local'}
    if state in {"unbound", "bf16-unbound"}:
        for row in payload['costs']['unit'].values():
            row.pop('source_binding')
    baseline = tmp_path / 'cost_baseline.pkl'
    baseline.write_bytes(pickle.dumps(payload))
    driver = (Path(__file__).resolve().parents[1] / 'prismaquant/run-pipeline.sh').read_text()
    functions = 'cost_table_cost_mode() {' + driver.split('cost_table_cost_mode() {', 1)[1].split(
        '\nrequire_stage_settings() {', 1)[0]
    decision = 'BASE_COST_REUSABLE=1' + driver.split('BASE_COST_REUSABLE=1', 1)[1].split(
        '\nif [[ "$BASE_COST_REUSABLE" == "0" ]]', 1)[0]
    environment = dict(os.environ, COST_MODE=mode, MODEL_PATH=str(source),
        BASE_COST_PATH=str(baseline), COST_PATH=str(baseline if mode == 'local' else tmp_path / 'cost.pkl'))
    environment['PATH'] = str(Path(sys.executable).parent) + os.pathsep + environment.get('PATH', '')
    result = subprocess.run(['bash', '-euo', 'pipefail', '-c', functions + '\n' + decision +
        '\nprintf "BASE_REUSABLE=%s\\n" "$BASE_COST_REUSABLE"'],
        env=environment, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = int(state in {'current', 'bf16-unbound'})
    assert f'BASE_REUSABLE={expected}' in result.stdout, result.stdout + result.stderr


def test_native_baseline_sidecars_carry_only_checkpoint_evidence(tmp_path, monkeypatch):
    """Sidecar measurements are doubled. Schedule and evidence publication are real."""
    from safetensors.torch import load_file, save_file
    from prismaquant import incremental_measure_quant_cost as baseline

    source, expected = producer_source(tmp_path)
    producer_cli_inputs(monkeypatch)
    names = [next(name.removesuffix('.weight') for name in expected
                  if name.endswith('.self_attn.o_proj.weight')),
             'mtp.layers.0.proj', 'model.visual.blocks.0.proj']
    raw = load_file(source / 'model.safetensors')
    raw.update({name + '.weight': torch.ones(4, 4, dtype=torch.bfloat16) for name in names[1:]})
    save_file(raw, source / 'model.safetensors')
    (source / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {name: 'model.safetensors' for name in raw}}))
    config = json.loads((source / 'config.json').read_text())
    config.update(num_nextn_predict_layers=1, vision_config={'depth': 1})
    (source / 'config.json').write_text(json.dumps(config))
    probe = tmp_path / 'sidecar-probe.pkl'
    probe.write_bytes(pickle.dumps({'stats': {name: {'h_trace': 1., 'n_params': 16}
        for name in names}, 'meta': {}}))
    activations = tmp_path / 'sidecar-act'
    activations.mkdir()
    for name in names:
        width = raw[name + '.weight'].shape[-1]
        torch.save({'inputs': torch.ones(4, width), 'name': name},
                   activations / (baseline.ActivationIndex._FNAME_SUB.sub('__', name) + '.pt'))

    def sidecar(*_args, **kwargs):
        selected = {name: {'BF16': {'weight_mse': 0.}}
                    for name in names if re.search(kwargs['linear_include'], name)}
        Path(kwargs['output_path']).write_bytes(pickle.dumps({'costs': selected,
            'formats': ['BF16'], 'meta': {}, 'provenance': {'cost_mode': 'local'}}))
        return None

    monkeypatch.setattr(baseline, '_run_mtp_cost_shard', sidecar)
    monkeypatch.setattr(baseline, '_run_visual_cost_shard', sidecar)
    output = tmp_path / 'sidecar-cost.pkl'
    monkeypatch.setattr(sys, 'argv', ['incremental_measure_quant_cost', '--model', str(source),
        '--probe', str(probe), '--activation-cache-dir', str(activations),
        '--work-dir', str(tmp_path / 'sidecar-work'), '--output', str(output),
        '--device', 'cpu', '--dtype', 'bf16', '--mode', 'unbatched', '--formats', 'BF16',
        '--cost-mode', 'local', '--threads', '1', '--no-include-lm-head', '--no-watchdog'])
    baseline.main()
    payload = pickle.loads(output.read_bytes())
    assert set(payload['costs']) == set(names)
    full = identity.build_source_model_binding(source, dtype=torch.bfloat16)
    checkpoint_only = {key: value for key, value in full.items() if key != 'source_decode'}
    checkpoint_only['schema'] = identity.SOURCE_MODEL_BINDING_SCHEMA
    assert payload['costs'][names[0]]['BF16']['source_binding'] == full
    for name in names[1:]:
        assert payload['costs'][name]['BF16']['source_binding'] == checkpoint_only
    intake.validate_native_cost_source_for_model(payload, model=str(source), where='current sidecars')
    for name in names[1:]:
        row = payload['costs'][name]['BF16']
        row['source_binding'] = {**checkpoint_only, 'checkpoint': {**checkpoint_only['checkpoint'],
            'content_sha256': '0' * 64}}
        with pytest.raises(ValueError, match='checkpoint content identity does not match'):
            intake.validate_native_cost_source_for_model(payload, model=str(source), where='invalid sidecars')
        row['source_binding'] = checkpoint_only
    (source / 'foreign.py').write_text('REVISION = 2\n')
    foreign = identity.build_source_model_binding(source, dtype=torch.bfloat16)
    (source / 'foreign.py').unlink()
    foreign = {key: value for key, value in foreign.items() if key != 'source_decode'}
    foreign['schema'] = identity.SOURCE_MODEL_BINDING_SCHEMA
    for name in names[1:]:
        row = payload['costs'][name]['BF16']
        row['source_binding'] = foreign
        with pytest.raises(ValueError, match='differs from the current checkpoint/decoder'):
            intake.validate_native_cost_source_for_model(payload, model=str(source), where='foreign sidecars')
        row['source_binding'] = full
        with pytest.raises(ValueError, match='sidecar binding must exclude the text decoder claim'):
            intake.validate_native_cost_source_for_model(payload, model=str(source), where='false decode claim')
        row['source_binding'] = checkpoint_only
    payload['costs'][names[0]]['BF16']['source_binding'] = checkpoint_only
    with pytest.raises(ValueError, match='legacy decoder identity cannot qualify native costs'):
        intake.validate_native_cost_source_for_model(payload, model=str(source), where='text control')
