"""Actual native resident delivery, cache fill, scores and cost synthesis."""
from copy import deepcopy
import json
import pickle

import pytest
import torch

from prismaquant.prepriced_cost import validate_native_cost_source_for_model
from prismaquant.production_render_cost import synthesize_production_render_cost_payload
from prismaquant.production_weight_cache import fill_production_weight_cache
from prismaquant.streaming_model import load_resident_source_model
from prismaquant.streamed_model_identity import build_source_model_binding
from tests.native_source_contract_fixtures import producer_cpu_guard, producer_source


@pytest.mark.parametrize('score_field', ['score_sum', 'output_mse', 'weight_mse'])
@pytest.mark.parametrize('fallback_binding', [False, True], ids=['unbound-fallback', 'foreign-fallback'])
def test_actual_render_synthesis_binds_only_scored_cells(
        tmp_path, monkeypatch, score_field, fallback_binding):
    source, expected = producer_source(tmp_path)
    guard = producer_cpu_guard(tmp_path, monkeypatch)
    model = load_resident_source_model(
        str(source), device=torch.device('cpu'), dtype=torch.bfloat16,
        offload_folder=str(tmp_path / 'resident'), resource_check=guard.check,
        attn_implementation='eager')
    names = sorted(name.removesuffix('.weight') for name in expected
                   if name.endswith('.self_attn.o_proj.weight'))
    binding = build_source_model_binding(source, dtype=torch.bfloat16)
    fmt = 'FP8_E4M3'
    cache = fill_production_weight_cache(
        model, torch.tensor([[1, 2, 3, 4]]), names,
        formats=[fmt], render_qnames=names[:1],
        levers={'gptq': False, 'scale_sweep': False},
        source_binding=binding, max_act_rows=4, progress=False)
    assert cache.metadata['render_scores']['records']
    baseline = {'meta': {'model': str(source)}, 'formats': [fmt, 'BF16'], 'costs': {
        name: {fmt: {'predicted_dloss': 0.375, 'cost_source': 'fallback-evidence'},
               'BF16': {'predicted_dloss': 0., 'cost_source': 'old-passthrough'}}
        for name in names}}
    if fallback_binding:
        (source / 'source_policy.py').write_text('FOREIGN_SOURCE = True\n')
        foreign = build_source_model_binding(source, dtype=torch.bfloat16)
        (source / 'source_policy.py').unlink()
        for row in baseline['costs'][names[1]].values():
            row['source_binding'] = foreign
    before = pickle.dumps(baseline)
    fallback = deepcopy(baseline['costs'][names[1]])
    result = synthesize_production_render_cost_payload(cache, baseline, score_field=score_field)
    assert pickle.dumps(baseline) == before
    assert result['costs'][names[1]][fmt] == fallback[fmt]
    assert result['costs'][names[1]]['BF16'].get('source_binding') == fallback['BF16'].get('source_binding')
    for row in result['costs'][names[0]].values():
        assert row.get('source_binding') == binding
    with pytest.raises(ValueError, match='source binding'):
        validate_native_cost_source_for_model(result, model=str(source), where='mixed render/fallback')
    result['costs'] = {names[0]: result['costs'][names[0]]}
    validate_native_cost_source_for_model(result, model=str(source), where='actual scored cells')


@pytest.mark.parametrize('foreign_region', ['all', 'mtp', 'visual'])
def test_native_render_sidecars_reuse_their_own_production_score_table(
        tmp_path, monkeypatch, foreign_region):
    """Render real body, MTP and visual matrices. Check source evidence only."""
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig
    from prismaquant import production_render_cost as render
    from prismaquant.measure_quant_cost import ActivationIndex
    from prismaquant.model_profiles import detect_profile
    from prismaquant.mtp_production_cache import fill_profile_mtp_production_cache
    from prismaquant.streaming_production_cache import _render_dense_layer
    from tests.native_source_contract_fixtures import producer_cli_inputs
    from tests.test_native_retained_cost_source_admission_audit import _pipeline_reusable

    source, expected = producer_source(tmp_path)
    producer_cli_inputs(monkeypatch)
    profile = detect_profile(str(source))
    mtp = profile.build_mtp_module(AutoConfig.from_pretrained(str(source), local_files_only=True))
    raw = load_file(source / 'model.safetensors')
    raw.update({'mtp.' + name: torch.full_like(value, 0.125).to(torch.bfloat16)
                for name, value in mtp.state_dict().items()})
    visual_name = 'model.visual.blocks.0.proj'
    raw[visual_name + '.weight'] = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4) / 32
    save_file(raw, source / 'model.safetensors')
    (source / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {name: 'model.safetensors' for name in raw}}))
    config = json.loads((source / 'config.json').read_text())
    config.update(num_nextn_predict_layers=1, vision_config={'depth': 1})
    (source / 'config.json').write_text(json.dumps(config))
    binding = build_source_model_binding(source, dtype=torch.bfloat16)
    checkpoint_only = {key: value for key, value in binding.items() if key != 'source_decode'}
    checkpoint_only['schema'] = 'prismaquant.source_model_binding.v1'
    body_name = next(name.removesuffix('.weight') for name in expected
                     if name.endswith('.self_attn.o_proj.weight'))
    mtp_name = 'mtp.layers.0.self_attn.o_proj'
    guard = producer_cpu_guard(tmp_path, monkeypatch)
    model = load_resident_source_model(str(source), device=torch.device('cpu'), dtype=torch.bfloat16,
        offload_folder=str(tmp_path / 'resident'), resource_check=guard.check, attn_implementation='eager')
    cache = fill_production_weight_cache(model, torch.tensor([[1, 2, 3, 4]]), [body_name, 'lm_head'],
        formats=['FP8_E4M3'], levers={'gptq': False, 'scale_sweep': False},
        source_binding=binding, max_act_rows=4, progress=False)
    activations = tmp_path / 'sidecar-act'
    activations.mkdir()
    for name in [mtp_name, visual_name]:
        torch.save({'inputs': torch.ones(4, raw[name + '.weight'].shape[-1]), 'name': name},
            activations / (ActivationIndex._FNAME_SUB.sub('__', name) + '.pt'))
    assert fill_profile_mtp_production_cache(cache, str(source), profile=profile,
        activation_cache_dir=activations, formats=['FP8_E4M3'], include_qnames=[mtp_name],
        device='cpu', dtype=torch.bfloat16, max_act_rows=4, progress=False) == 1
    visual = torch.nn.Module()
    visual.blocks = torch.nn.ModuleList([torch.nn.Module()])
    visual.blocks[0].proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    visual.blocks[0].proj.weight.data.copy_(raw[visual_name + '.weight'])
    model.model.add_module('visual', visual)
    records = cache.metadata['render_scores']['records']
    assert _render_dense_layer(model, {visual_name: visual.blocks[0].proj},
        render_formats_by_qname={visual_name: ['FP8_E4M3']},
        act_index=ActivationIndex(activations, [visual_name]), cache=cache, levers=cache.levers,
        cache_dir_path=None, profile=profile, device=torch.device('cpu'), fisher_rows=None,
        render_score_records=records, col_weights=None, retain_rendered=True,
        consume_render=None, max_act_rows=4, progress=False) == 1
    names = [body_name, 'lm_head', mtp_name, visual_name]
    assert all(name + '|FP8_E4M3' in records for name in names)
    baseline = {'formats': ['FP8_E4M3', 'BF16'], 'costs': {
        name: {fmt: {'predicted_dloss': 99., 'source_binding':
                     checkpoint_only if name in [mtp_name, visual_name] else binding}
               for fmt in ['FP8_E4M3', 'BF16']} for name in names},
        'provenance': {'cost_mode': 'local'}, 'meta': {'shards': [{'model': str(source)}]}}
    cache_path, baseline_path, output = [tmp_path / name for name in ['cache.pkl', 'baseline.pkl', 'cost.pkl']]
    cache_path.write_bytes(pickle.dumps(cache))
    baseline_path.write_bytes(pickle.dumps(baseline))
    before = (cache_path.read_bytes(), baseline_path.read_bytes())
    arguments = ['--production-cache', str(cache_path), '--baseline-cost', str(baseline_path),
                 '--output', str(output), '--cost-mode', 'production-render-score',
                 '--score-field', 'score_sum', '--require-render-scores']
    assert render.main(arguments) == 0
    payload = pickle.loads(output.read_bytes())
    validate_native_cost_source_for_model(payload, model=str(source), where='real render sidecars')
    reused = _pipeline_reusable(output, source, cost_mode='production-render-score')
    assert reused.returncode == 0, reused.stdout + reused.stderr
    for name in names:
        for row in payload['costs'][name].values():
            assert row['source_binding'] == (checkpoint_only if name in [mtp_name, visual_name] else binding)
    original_output = output.read_bytes()
    assert render.main(arguments) == 0
    assert output.read_bytes() == original_output
    assert before == (cache_path.read_bytes(), baseline_path.read_bytes())
    (source / 'foreign_policy.py').write_text('POLICY = 2\n')
    cache.metadata['source_binding'] = build_source_model_binding(source, dtype=torch.bfloat16)
    (source / 'foreign_policy.py').unlink()
    cache_path.write_bytes(pickle.dumps(cache))
    assert render.main(arguments) == 0
    foreign = pickle.loads(output.read_bytes())
    if foreign_region != 'all':
        foreign_name = mtp_name if foreign_region == 'mtp' else visual_name
        current = pickle.loads(original_output)
        for name in names:
            if name != foreign_name:
                foreign['costs'][name] = deepcopy(current['costs'][name])
        for name in names:
            for fmt, row in foreign['costs'][name].items():
                if name == foreign_name:
                    assert row['source_binding']['checkpoint'] != binding['checkpoint']
                else:
                    assert row['source_binding'] == current['costs'][name][fmt]['source_binding']
        output.write_bytes(pickle.dumps(foreign))
    with pytest.raises(ValueError, match='differs from the current checkpoint/decoder'):
        validate_native_cost_source_for_model(foreign, model=str(source), where='foreign render sidecars')
    reused = _pipeline_reusable(output, source, cost_mode='production-render-score')
    assert reused.returncode != 0
    assert 'differs from the current checkpoint/decoder' in reused.stderr
