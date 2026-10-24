"""Leaf accounting is provenance/coverage, never a compatibility oracle."""
import ast
import hashlib
import json
from pathlib import Path

from tools import duplication_inventory as inventory

ROOT = Path(__file__).resolve().parents[3]
LEDGER = Path(__file__).with_name('ledger.json')


def test_leaf_registry_and_current_residuals():
    ledger = json.loads(LEDGER.read_text())
    registry = json.loads((ROOT / 'tests/fixtures/digest_parent_completion_1301.json').read_text())
    originals = {row['id']: row for row in registry['calls']
                 if row['file'] in ledger['assigned_files']}
    rows = ledger['completed_calls'] + ledger['retained_calls'] + ledger['blocked_calls']
    assert len(rows) == len({row['original_id'] for row in rows})
    assert {row['original_id'] for row in rows} == set(originals)
    for row in rows:
        base = originals[row['original_id']]
        assert row['source_commit'] == registry['source_commit']
        assert row['source_blob'] == base['source_blob']
        for fixture in row['fixtures']:
            assert (ROOT / fixture).is_file()
    for row in ledger['completed_calls'] + ledger['retained_calls']:
        assert row['tests']
        for node in row['tests']:
            file, test = node.split('::')
            assert file.startswith('tests/test_') and '[' not in test
            functions = {n.name for n in ast.parse((ROOT / file).read_text()).body
                         if isinstance(n, ast.FunctionDef)}
            assert test in functions
    for row in ledger['completed_calls']:
        assert row['owner_route'].startswith(('prismaquant.digests.', 'prismaquant.tensor_digests.'))
    expected = {row['current_id']: row for row in ledger['retained_calls'] + ledger['blocked_calls']}
    assert len(expected) == len(ledger['retained_calls']) + len(ledger['blocked_calls'])
    actual = {}
    for file in ledger['assigned_files']:
        raw = (ROOT / file).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        for call, scope, aliases in inventory._iter_scoped_calls(ast.parse(raw.decode())):
            callee = inventory._resolved_callee(call.func, aliases)
            if callee and callee.startswith('hashlib.'):
                family = 'hashlib'
            elif callee in ('json.dumps', 'json.JSONEncoder') and any(
                    keyword.arg == 'sort_keys' and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True for keyword in call.keywords):
                family = 'sorted-json'
            elif callee in {'json.dumps', 'json.dump', 'json.JSONEncoder'}:
                family = 'supplemental-json'
            else:
                continue
            ident = f'{family}:{file}::{scope}@{call.lineno}:{call.col_offset}'
            actual[ident] = blob
    assert set(actual) == set(expected)
    assert all(expected[ident]['current_source_blob'] == blob for ident, blob in actual.items())
