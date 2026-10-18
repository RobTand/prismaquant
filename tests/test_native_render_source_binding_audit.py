"""Actual native resident delivery, cache fill, scores and cost synthesis."""
from copy import deepcopy
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
    baseline = {'formats': [fmt, 'BF16'], 'costs': {
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
