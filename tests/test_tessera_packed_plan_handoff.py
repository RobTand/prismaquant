"""Packed decisions reach the pinned producer translator as source units."""
import hashlib
import json
from pathlib import Path

from test_tessera_export_projection import (
    case, _cli, _isolate_other_gates, _units, DENSE, N, ROUTER, STACK,
)
from test_tessera_packed_export_scope import _pack


def test_preflight_packed_census_reaches_real_producer_translator(case, tmp_path, monkeypatch):
    _isolate_other_gates(monkeypatch)
    parameters = _pack(case)
    original = case.assignment.read_bytes()
    assert _cli(case, tmp_path) == 0
    build = json.loads((tmp_path / 'build.json').read_text())
    translated = Path(build.get('plan_assignment', str(case.assignment)))
    from prismaquant.tessera_plan_writer import build_serving_plan, tessera_surface
    config = json.loads(translated.read_text())
    plan, provenance = build_serving_plan(config,
        {**{n + '.weight': (N, N) for n in (DENSE, *_units())},
         ROUTER + '.weight': (2, N)},
        cover='as-allocated', allow_disagreement=False, surface=tessera_surface(), with_control=False)
    assert all(plan[n + '.weight'] == {'grid': 'E4M3', 'q256': 1024} for n in _units())
    assert not any(name in config for name in parameters)
    assert case.assignment.read_bytes() == original
    assert build['layer_config'] == str(case.assignment)
    assert build['layer_config_sha'] == hashlib.sha256(original).hexdigest()
    assert translated != case.assignment
    assert build['plan_assignment_sha256'] == hashlib.sha256(translated.read_bytes()).hexdigest()
    assert config['__prismaquant__']['tessera_export_assignment']['source_sha256'] == build['layer_config_sha']
