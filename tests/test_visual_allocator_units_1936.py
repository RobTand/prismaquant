"""Visual/merger rows in the probe take the uniform --visual-format (#1936).

Synthetic CPU candidate/contract evidence only; no production vision serving
qualification is inferred.
"""
from __future__ import annotations

import copy
import json
import pickle
import sys

from prismaquant import allocator
from prismaquant.layer_config import load_assignment

VISION = 'model.visual.blocks.0.attn.qkv'
MERGER = 'model.visual.merger.mlp.0'
BODY = {
    'model.layers.0.self_attn.o_proj': dict(router_path=None, expert_id=None),
    'model.layers.0.mlp.experts.0.down_proj': dict(router_path='model.layers.0.mlp.gate',
                                                   expert_id='0'),
    'model.layers.0.mlp.shared_expert.down_proj': dict(router_path=None, expert_id=None),
}
FORMATS = ('NVFP4', 'FP8_DYNAMIC')


def _row(**extra):
    return dict(n_params=64 * 64, in_features=64, out_features=64, **extra)


def _cost(nvfp4, fp8):
    return {'NVFP4': {'predicted_dloss': nvfp4, 'weight_mse': nvfp4 / 2},
            'FP8_DYNAMIC': {'predicted_dloss': fp8, 'weight_mse': fp8 / 2}}


def _inputs(tmp_path, monkeypatch):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text(json.dumps({
        'model_type': 'qwen3_moe', 'architectures': ['Qwen3MoeForCausalLM']}))
    probe = {'stats': {name: _row(**row) for name, row in BODY.items()},
             'meta': {'model': str(model)}}
    costs = {name: _cost(4e-4, 1e-5) for name in BODY}
    for name, role in ((VISION, 'vis_attn_qkv'), (MERGER, 'merger_fc1')):
        probe['stats'][name] = _row(router_path=None, expert_id=None, type=role)
        # Nonzero AURA prices that disagree per row: an allocation would split
        # the two visual rows, so a uniform result comes from --visual-format.
        costs[name] = _cost(1e-5, 4e-4) if name == VISION else _cost(8e-4, 2e-5)
    probe_path = tmp_path / 'probe.pkl'
    cost_path = tmp_path / 'cost.pkl'
    probe_path.write_bytes(pickle.dumps(probe))
    cost_path.write_bytes(pickle.dumps({'costs': costs, 'formats': list(FORMATS),
                                        'provenance': {'cost_mode': 'aura'}}))
    # Source-header census is synthetic too: exercise the late
    # uniform-restamping path without real weights.
    source_stats = {name: {**copy.deepcopy(probe['stats'][name]), 'source_dtype': 'bf16'}
                    for name in (VISION, MERGER)}
    monkeypatch.setattr(allocator, 'discover_visual_linear_stats_from_source',
                        lambda _model, **_kwargs: copy.deepcopy(source_stats))
    monkeypatch.setattr(sys, 'argv', [
        'allocator', '--probe', str(probe_path), '--costs', str(cost_path),
        '--formats', ','.join(FORMATS), '--target-bits', '16', '--pareto-targets', '16',
        '--bit-precision', '0.1', '--layer-config', str(tmp_path / 'layer.json'),
        '--pareto-csv', str(tmp_path / 'pareto.csv'),
        '--no-fused-aggregation', '--no-packed-aggregation', '--visual-format', 'BF16',
        '--bit-attribution-json', str(tmp_path / 'attribution.json'),
        '--bit-attribution-csv', str(tmp_path / 'attribution.csv'),
    ])
    return tmp_path / 'layer.json', probe


def test_explicit_uniform_mode_preserves_the_source_precision_control(tmp_path, monkeypatch):
    output, _probe = _inputs(tmp_path, monkeypatch)
    allocator.main()
    assignment = load_assignment(output)
    assert assignment[VISION] == assignment[MERGER] == 'BF16'
    attribution = json.loads((tmp_path / 'attribution.json').read_text())
    assert attribution['n_body_linears'] == 3
