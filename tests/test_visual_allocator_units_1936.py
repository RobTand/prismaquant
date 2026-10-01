"""Measured visual/merger units must survive the real allocator (#1936).

Synthetic CPU candidate/contract evidence only; no production vision serving
qualification is inferred. The external producer's non-body refusal stays put.
"""
from __future__ import annotations

import copy
import csv
import json
import pickle
import sys
from pathlib import Path

import pytest

from prismaquant import allocator, format_registry as fr
from prismaquant.layer_config import load_assignment
from test_tessera_scope_endpoints import DENSE, _allocator_inputs, _cli_scope, _v5_contract

VISION = 'model.visual.blocks.0.attn.qkv'
MERGER = 'model.visual.merger.mlp.0'
DENSE_RUNG = 'TESSERA_E4M3_K1_R1024'
VALUE_RUNG = 'TESSERA_E2M1_K2_R896'


def _inputs(tmp_path, monkeypatch, *, missing_cost=None, sensitivity='fisher',
            source_extra=False, source_shape_mismatch=False):
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
    if missing_cost:
        del cost['costs'][missing_cost]
    probe_path.write_bytes(pickle.dumps(probe))
    cost_path.write_bytes(pickle.dumps(cost))
    # Source-header census is synthetic too: exercise both complete roster
    # reconciliation and the late uniform-restamping path without real weights.
    source_stats = {name: {**copy.deepcopy(probe['stats'][name]), 'source_dtype': 'bf16'}
                    for name in (VISION, MERGER)}
    if source_extra:
        source_stats['model.visual.merger.mlp.2'] = copy.deepcopy(source_stats[MERGER])
    if source_shape_mismatch:
        source_stats[MERGER]['in_features'] += 1
    monkeypatch.setattr(allocator, 'discover_visual_linear_stats_from_source',
                        lambda _model, **_kwargs: copy.deepcopy(source_stats))
    argv[argv.index('--formats') + 1] = f'{DENSE_RUNG},{VALUE_RUNG}'
    monkeypatch.setattr(sys, 'argv', [
        'allocator', *argv, '--no-fused-aggregation', '--no-packed-aggregation',
        '--visual-sensitivity', sensitivity, '--visual-format', 'BF16',
        '--bit-attribution-json', str(tmp_path / 'attribution.json'),
        '--bit-attribution-csv', str(tmp_path / 'attribution.csv'), *_cli_scope(),
    ])
    return tmp_path / 'layer.json', probe


def test_real_main_keeps_independent_measured_vision_and_merger_choices(tmp_path, monkeypatch):
    output, probe = _inputs(tmp_path, monkeypatch)
    allocator.main()
    assignment = load_assignment(output)
    assert assignment[VISION] == DENSE_RUNG
    assert assignment[MERGER] == VALUE_RUNG
    metadata = json.loads(output.read_text())['__prismaquant__']
    by_unit = metadata['serving_lane_provenance']['by_unit']
    for name in (VISION, MERGER):
        assert by_unit[name]['format'] == assignment[name]
        assert by_unit[name]['serving_context']['structure'] == 'dense'
    attribution = json.loads((tmp_path / 'attribution.json').read_text())
    assert attribution['n_body_linears'] == len(probe['stats'])
    assert attribution['body_quantizable_params'] == sum(
        entry['n_params'] for entry in probe['stats'].values())
    with (tmp_path / 'attribution.csv').open() as stream:
        by_name = {row['qname']: row for row in csv.DictReader(stream)}
    assert by_name[VISION]['format'] == DENSE_RUNG
    assert by_name[MERGER]['format'] == VALUE_RUNG
    assert float(by_name[VISION]['predicted_dloss']) > 0
    assert float(by_name[MERGER]['predicted_dloss']) > 0


def test_partial_visual_cost_population_is_refused_before_selection(tmp_path, monkeypatch):
    output, _probe = _inputs(tmp_path, monkeypatch, missing_cost=MERGER)
    with pytest.raises(SystemExit, match='visual Fisher.*incomplete.*merger'):
        allocator.main()
    assert not output.exists()


def test_explicit_uniform_mode_preserves_the_source_precision_control(tmp_path, monkeypatch):
    output, _probe = _inputs(tmp_path, monkeypatch, sensitivity='uniform')
    allocator.main()
    assignment = load_assignment(output)
    assert assignment[VISION] == assignment[MERGER] == 'BF16'
    attribution = json.loads((tmp_path / 'attribution.json').read_text())
    assert attribution['n_body_linears'] == 3


@pytest.mark.parametrize('source_extra,source_shape_mismatch', [(True, False), (False, True)])
def test_incomplete_or_mismatched_visual_source_roster_is_refused(
        tmp_path, monkeypatch, source_extra, source_shape_mismatch):
    output, _probe = _inputs(tmp_path, monkeypatch, source_extra=source_extra,
                            source_shape_mismatch=source_shape_mismatch)
    with pytest.raises(SystemExit, match='visual Fisher source roster.*incomplete.*merger'):
        allocator.main()
    assert not output.exists()
