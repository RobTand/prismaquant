"""Visual/merger rows in the probe take the uniform --visual-format (#1936).

Synthetic CPU candidate/contract evidence only; no production vision serving
qualification is inferred.
"""
from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path

from prismaquant import allocator, format_registry as fr
from prismaquant.layer_config import load_assignment
from test_tessera_scope_endpoints import DENSE, _allocator_inputs, _cli_scope, _v5_contract

VISION = 'model.visual.blocks.0.attn.qkv'
MERGER = 'model.visual.merger.mlp.0'
DENSE_RUNG = 'TESSERA_E4M3_K1_R1024'
VALUE_RUNG = 'TESSERA_E2M1_K2_R896'


def _inputs(tmp_path, monkeypatch):
    _v5_contract(monkeypatch)
    argv = _allocator_inputs(tmp_path, DENSE_RUNG)
    probe_path = Path(argv[argv.index('--probe') + 1])
    cost_path = Path(argv[argv.index('--costs') + 1])
    probe = pickle.loads(probe_path.read_bytes())
    cost = pickle.loads(cost_path.read_bytes())
    for name, role in ((VISION, 'vis_attn_qkv'), (MERGER, 'merger_fc1')):
        probe['stats'][name] = {**copy.deepcopy(probe['stats'][DENSE]), 'type': role}
        row = copy.deepcopy(cost['costs'][DENSE][DENSE_RUNG])
        cost['costs'][name] = {DENSE_RUNG: row}
        # Keep a nonzero, measured cost: this is not a fabricated identity.
        row['output_mse'] = 1e-5 if name == VISION else 8e-4
        row['weight_mse'] = row['output_mse'] / 2
        alternate = {**copy.deepcopy(row), 'output_mse': 4e-4 if name == VISION else 2e-5}
        alternate['weight_mse'] = alternate['output_mse'] / 2
        cost['costs'][name][VALUE_RUNG] = alternate
    probe_path.write_bytes(pickle.dumps(probe))
    cost_path.write_bytes(pickle.dumps(cost))
    # Source-header census is synthetic too: exercise the late
    # uniform-restamping path without real weights.
    source_stats = {name: {**copy.deepcopy(probe['stats'][name]), 'source_dtype': 'bf16'}
                    for name in (VISION, MERGER)}
    monkeypatch.setattr(allocator, 'discover_visual_linear_stats_from_source',
                        lambda _model, **_kwargs: copy.deepcopy(source_stats))
    argv[argv.index('--formats') + 1] = f'{DENSE_RUNG},{VALUE_RUNG}'
    monkeypatch.setattr(sys, 'argv', [
        'allocator', *argv, '--no-fused-aggregation', '--no-packed-aggregation',
        '--visual-format', 'BF16',
        '--bit-attribution-json', str(tmp_path / 'attribution.json'),
        '--bit-attribution-csv', str(tmp_path / 'attribution.csv'), *_cli_scope(),
    ])
    return tmp_path / 'layer.json', probe


def test_explicit_uniform_mode_preserves_the_source_precision_control(tmp_path, monkeypatch):
    output, _probe = _inputs(tmp_path, monkeypatch)
    allocator.main()
    assignment = load_assignment(output)
    assert assignment[VISION] == assignment[MERGER] == 'BF16'
    attribution = json.loads((tmp_path / 'attribution.json').read_text())
    assert attribution['n_body_linears'] == 3
