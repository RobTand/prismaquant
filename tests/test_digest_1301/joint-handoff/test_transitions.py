"""Compatibility evidence from frozen original consumers, not owner-derived goldens."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIX = ROOT / 'tests/fixtures/digest_1301/joint-handoff'
LEDGER = json.loads((FIX / 'ledger_transitions.json').read_text())
OUTCOMES = json.loads((FIX / 'outcomes.json').read_text())


def function(source, name):
    return next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def test_frozen_original_provenance():
    registry = json.loads((ROOT / 'tests/fixtures/digest_parent_completion_1301.json').read_text())
    calls = {c['id']: c for c in registry['calls']}
    rows = LEDGER['replacements'] + LEDGER['unresolved']
    for row in rows:
        original = calls[row['original_id']]
        raw = (ROOT / row['fixtures'][0]).read_bytes()
        excerpt = LEDGER['fixture_excerpts'][row['fixtures'][0]]
        assert hashlib.sha256(raw).hexdigest() == excerpt['sha256']
        assert len(raw.splitlines()) < 80
        assert excerpt['source_blob'] == row['source_blob'] == original['source_blob']
        assert row['source_commit'] == registry['source_commit']
        for node_id in row['tests']:
            path, name = node_id.split('::')
            function((ROOT / path).read_text(), name)
    files = {'tools/prepare_extended_joint_quanta.py', 'tools/regenerate_joint_quanta.py',
             'tools/stage_b_preparation_submission.py', 'tools/generate_transition_rewrites.py'}
    scoped = {c['id'] for c in registry['calls'] if c['file'] in files}
    assert {r['original_id'] for r in rows} == scoped


def test_original_digest_outcomes(tmp_path):
    from prismaquant.digests import bytes_sha256hex
    wire = bytes.fromhex(OUTCOMES['wire_hex'])
    path = tmp_path / 'wire'
    path.write_bytes(wire)
    env = dict(bytes_sha256hex=bytes_sha256hex, hashlib=hashlib, raw=wire,
               Path=Path, path=path, extension_path=path, manifest_path=path,
               slice_path=str(path), binding={'path': str(path)},
               manifest_bytes=wire, template_bytes=wire,
               moved={'slice_manifests': {'q': {}}}, qid='q',
               seal_manifest_bytes=lambda value: wire)
    for filename in sorted({r['original_id'].split(':')[1] for r in LEDGER['replacements']}):
        current = (ROOT / filename).read_text()
        rows = [r for r in LEDGER['replacements'] if r['original_id'].split(':')[1] == filename and 'LengthFramed' not in r['owner_route']]
        for scope in sorted({r['original_id'].split('::')[1].split('@')[0] for r in rows}):
            scoped_rows = sorted([r for r in rows if r['original_id'].split('::')[1].split('@')[0] == scope], key=lambda r: tuple(map(int, r['original_id'].split('@')[1].split(':'))))
            nodes = sorted([n for n in ast.walk(function(current, scope)) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'bytes_sha256hex'], key=lambda n: (n.lineno, n.col_offset))
            assert len(nodes) == len(scoped_rows)
            for row, node in zip(scoped_rows, nodes):
                case = OUTCOMES['original_expressions'][row['original_id']]
                assert eval(case['expression'], env) == case['expected']
                assert eval(compile(ast.Expression(node), '<current>', 'eval'), env) == case['expected']
    from tools.prepare_extended_joint_quanta import bind
    assert bind(path) == {'path': str(path.resolve()), 'sha256': bytes_sha256hex(wire)}
    with pytest.raises(FileNotFoundError):
        bind(tmp_path / 'absent')


def test_package_order_and_selection(tmp_path):
    from tools.generate_transition_rewrites import _package_digest, _read_tree
    case = OUTCOMES['package']
    files = {k: bytes.fromhex(v) for k, v in case['files'].items()}
    assert _package_digest(files) == case['expected']
    assert _package_digest(dict(reversed(list(files.items())))) == case['expected']
    for name, raw in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    for name in ('ignored.pyc', 'ignored.pyo', '__pycache__/x.py', 'sub/__pycache__/x'):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'ignored')
    assert _read_tree(tmp_path) == files
    assert _package_digest(_read_tree(tmp_path)) == case['expected']


def test_excerpted_functions_unchanged():
    class RestoreHash(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node.func) == 'bytes_sha256hex':
                return ast.Call(func=ast.Attribute(value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id='hashlib', ctx=ast.Load()),
                                       attr='sha256', ctx=ast.Load()),
                    args=node.args, keywords=[]), attr='hexdigest', ctx=ast.Load()),
                    args=[], keywords=[])
            return node
    for stem in ('prepare_extended_joint_quanta', 'regenerate_joint_quanta',
                 'stage_b_preparation_submission', 'generate_transition_rewrites'):
        old = ast.parse((FIX / (stem + '.original.txt')).read_text())
        new = RestoreHash().visit(ast.parse((ROOT / 'tools' / (stem + '.py')).read_text()))
        old_functions = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
        new_functions = {n.name: n for n in new.body if isinstance(n, ast.FunctionDef)}
        assert old_functions.keys() <= new_functions.keys()
        for name in old_functions:
            if name != '_package_digest':
                assert ast.dump(old_functions[name]) == ast.dump(new_functions[name]), (stem, name)


def test_unresolved_serialization():
    assert LEDGER['current_retentions'] == []
    assert len(LEDGER['unresolved']) == 4
    assert all(r['status'] == 'API-gap unresolved' for r in LEDGER['unresolved'])
    for row in LEDGER['unresolved']:
        filename = row['original_id'].split(':')[1]
        scope = row['original_id'].split('::')[1].split('@')[0]
        old = (ROOT / row['fixtures'][0]).read_text()
        current = (ROOT / filename).read_text()
        def dumps(node):
            return [ast.dump(n, include_attributes=False) for n in ast.walk(node)
                    if isinstance(n, ast.Call) and ast.unparse(n.func) == 'json.dumps']
        assert dumps(ast.parse(old)) == dumps(function(current, scope))
    from tools.regenerate_joint_quanta import _pretty
    ns = {'json': json}
    exec(compile(ast.Module(body=[function((FIX / 'regenerate_joint_quanta.original.txt').read_text(), '_pretty')], type_ignores=[]), '<original>', 'exec'), ns)
    assert _pretty({'z': 'é', 'a': [1, None]}) == ns['_pretty']({'z': 'é', 'a': [1, None]})
    for pretty in (_pretty, ns['_pretty']):
        with pytest.raises(ValueError, match='Out of range float'):
            pretty({'nan': float('nan')})


def test_owner_profiles_are_not_exact_substitutes():
    from prismaquant.digests import (DIRECT_ASCII_INDENT1_LAX_DEFAULT_STR,
                                     DIRECT_ASCII_SPACED_LAX)
    unsupported = object()
    # The closest indent=1 profile silently stringifies unsupported values.
    assert DIRECT_ASCII_INDENT1_LAX_DEFAULT_STR.text(unsupported)
    with pytest.raises(TypeError):
        json.dumps(unsupported, sort_keys=True, indent=1)
    assert 'NaN' in DIRECT_ASCII_INDENT1_LAX_DEFAULT_STR.text(float('nan'))
    # The stdout site preserves insertion order; all finished profiles sort.
    value = {'z': 1, 'a': 2}
    assert DIRECT_ASCII_SPACED_LAX.text(value) != json.dumps(value)
