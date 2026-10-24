"""Consumer checks against tiny pre-routing expressions, not new owners."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from prismaquant import tessera_export_lane as export
from prismaquant import tessera_materialization as materialization

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/digest_1301/serving-evidence'
spec = importlib.util.spec_from_file_location('plans_original', FIXTURES / 'plans_original.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def report(build):
    return dict(build=build, serving_target={'z': '雪', 'a': (1, -0.0)},
                tessera={'release': 'test'}, executes=['test'], structure='test',
                pinned_version='test', pinned_commit='abc123', shipcard_slots=[],
                unrecorded_gates=[], unsupported_producer_tools=[],
                producer_tools={'available': ['test'], 'missing': []},
                platform_executes_derived=False, admission={}, structure_check=None,
                priced_inputs=None)


def test_export_main_build_bytes_and_digest(tmp_path, monkeypatch, capsys):
    # Includes nonfinite numbers: strict file helper would be an invalid route.
    for value in ({'z': '雪', 'a': (True, -0.0, None)},
                  {'nan': float('nan'), 'inf': float('inf')},
                  {2: 'two', 1: 'one'}):
        data = report(value)
        monkeypatch.setattr(export, 'preflight', lambda *a, **kw: data)
        output = tmp_path / 'nested/build.json'
        assert export.main(['--model', str(tmp_path), '--assignment', 'unused',
                            '--write-build-json', str(output), '--print-build-sha256']) == 0
        captured = capsys.readouterr()
        expected = original.export_build(data)
        assert output.read_bytes() == expected
        assert captured.out == hashlib.sha256(expected).hexdigest() + '\n'
        assert '[preflight] explicit serving target: ' + original.export_target(data) + '\n' in captured.err


def test_export_main_refusal_does_not_publish(tmp_path, monkeypatch, capsys):
    data = report({'bad': object()})
    monkeypatch.setattr(export, 'preflight', lambda *a, **kw: data)
    output = tmp_path / 'build.json'
    output.write_bytes(b'existing publication')
    with pytest.raises(TypeError):
        original.export_build(data)
    with pytest.raises(TypeError):
        export.main(['--model', str(tmp_path), '--assignment', 'unused',
                     '--write-build-json', str(output)])
    assert output.read_bytes() == b'existing publication'
    assert not capsys.readouterr().out


def test_materialization_atomic_json_bytes(tmp_path):
    destination = tmp_path / 'request.json'
    value = {'z': '雪', 'a': (1, -0.0, None), 'nested': {'b': 2, 'a': 1}}
    materialization._json(destination, value)
    assert destination.read_bytes() == original.materialization_file(value)
    assert materialization._sha(destination) == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert list(tmp_path.iterdir()) == [destination]


def test_materialization_strict_refusal_preserves_publication(tmp_path):
    destination = tmp_path / 'request.json'
    for value in ({'nan': float('nan')}, {'inf': float('inf')}, {'bad': object()},
                  {'mixed': {1: 1, 'a': 2}}):
        destination.write_bytes(b'old')
        with pytest.raises((ValueError, TypeError)) as expected:
            original.materialization_file(value)
        with pytest.raises(type(expected.value)):
            materialization._json(destination, value)
        assert destination.read_bytes() == b'old'
        assert list(tmp_path.iterdir()) == [destination]


def test_plans_ledger_nodes_and_accounting():
    ledger = json.loads((FIXTURES / 'ledger_plans.json').read_text())
    for row in ledger['replacements']:
        for nodeid in row['test_node_ids']:
            filename, name = nodeid.split('::')
            assert '[' not in name
            tree = ast.parse((ROOT / filename).read_text())
            assert any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
                       for n in ast.walk(tree))
        for fixture in row['fixtures']:
            assert (ROOT / fixture).is_file()
    # Catch new unaccounted calls including unsorted supplemental recipes.
    for filename, entries in ledger['whole_file_accounting'].items():
        tree = ast.parse((ROOT / filename).read_text())
        current = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and ((isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                         and n.func.value.id == 'json' and n.func.attr in ('dumps', 'dump', 'JSONEncoder'))
                        or (isinstance(n.func, ast.Attribute) and n.func.attr in ('text', 'encoded')
                            and isinstance(n.func.value, ast.Name) and n.func.value.id.startswith('DIRECT_'))
                        or (isinstance(n.func, ast.Name) and n.func.id == 'indent2_json_file_bytes'))]
        assert sorted(current) == sorted(e['expression'] for e in entries)
