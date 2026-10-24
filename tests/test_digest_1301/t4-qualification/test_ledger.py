"""Piece-level accounting, not a substitute for consumer compatibility tests."""
import ast
import hashlib
import json
from pathlib import Path

from tools import duplication_inventory as inventory

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/digest_1301/t4-qualification'


def test_completion_provenance_and_executable_nodes():
    ledger = json.loads((FIXTURES / 'ledger.json').read_text())
    registry = json.loads((ROOT / ledger['registry']).read_text())
    original = {r['id']: r for r in registry['calls']}
    completed = ledger['completed_calls']
    assert len(completed) == len({r['original_id'] for r in completed}) == 26
    for row in completed:
        base = original[row['original_id']]
        assert row['source_commit'] == registry['source_commit']
        assert row['source_blob'] == base['source_blob']
        assert row['owner_route'] and row['fixtures'] and row['tests']
        for fixture in row['fixtures']:
            assert fixture.startswith('tests/fixtures/digest_1301/t4-qualification/')
            assert (ROOT / fixture).is_file()
        for node in row['tests']:
            filename, function = node.split('::')
            assert '[' not in function and function.startswith('test_')
            assert filename.startswith('tests/test_digest_1301/t4-qualification/test_')
            tree = ast.parse((ROOT / filename).read_text())
            assert any(isinstance(n, ast.FunctionDef) and n.name == function for n in tree.body)


def test_residuals_are_explicit_gaps_or_independent_refusals():
    rows = json.loads((FIXTURES / 'residuals.json').read_text())['calls']
    expected = {r['id']: r for r in rows}
    assert len(expected) == len(rows) == 24
    live = {}
    for filename in {r['file'] for r in rows}:
        raw = (ROOT / filename).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        for call, scope, aliases in inventory._iter_scoped_calls(ast.parse(raw)):
            callee = inventory._resolved_callee(call.func, aliases)
            if callee not in ('json.dumps', 'json.dump', 'json.JSONEncoder') and not (callee or '').startswith('hashlib.'):
                continue
            sorted_keys = any(k.arg == 'sort_keys' and isinstance(k.value, ast.Constant)
                              and k.value.value is True for k in call.keywords)
            family = 'hashlib' if callee.startswith('hashlib.') else 'sorted-json' if sorted_keys else 'supplemental-json'
            identity = f'{family}:{filename}::{scope}@{call.lineno}:{call.col_offset}'
            live[identity] = blob
    assert set(live) == set(expected)
    for identity, row in expected.items():
        assert row['source_blob'] == live[identity]
        if row['disposition'] == 'retain':
            assert not row['eligible'] and row['retention_evidence']
            assert row['file'] == 'tools/test_qualify_t4_overlay.py'
            assert row['scope'] == 'test_partial_checkpoint_corruption_is_not_adopted'
        else:
            assert row['eligible'] and row['disposition'] == 'blocked-exact-owner-gap'
            assert row['retention_evidence'] is None
    assert sum(r['disposition'] == 'retain' for r in rows) == 3
    # The independent malformed-checkpoint construction itself is unchanged.
    registry = json.loads((ROOT / 'tests/fixtures/digest_parent_completion_1301.json').read_text())
    old_blobs = {r['source_blob'] for r in registry['calls'] if r['file'] == 'tools/test_qualify_t4_overlay.py'}
    assert {r['source_blob'] for r in rows if r['disposition'] == 'retain'} == old_blobs
