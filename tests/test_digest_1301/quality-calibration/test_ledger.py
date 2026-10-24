"""Local accounting only: compatibility is proved by each referenced consumer test."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

from tools import duplication_inventory as inventory

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / 'tests/fixtures/digest_1301/quality-calibration/ledger.json'


def calls(files):
    rows = []
    for name in files:
        raw = (ROOT / name).read_bytes()
        source = raw.decode('utf-8')
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        for call, scope, aliases in inventory._iter_scoped_calls(ast.parse(source)):
            callee = inventory._resolved_callee(call.func, aliases)
            if callee and callee.startswith('hashlib.'):
                family = 'hashlib'
            elif callee in {'json.dumps', 'json.dump', 'json.JSONEncoder'}:
                sorted_keys = any(k.arg == 'sort_keys' and isinstance(k.value, ast.Constant)
                                  and k.value.value is True for k in call.keywords)
                family = 'sorted-json' if sorted_keys and callee != 'json.dump' else 'supplemental-json'
            else:
                continue
            rows.append(dict(id=f'{family}:{name}::{scope}@{call.lineno}:{call.col_offset}',
                             file=name, scope=scope, callee=callee, source_blob=blob,
                             expression=ast.get_source_segment(source, call),
                             syntax_sha256=hashlib.sha256(ast.dump(call, include_attributes=False).encode()).hexdigest()))
    return rows


def identity(row):
    return row['file'], row['scope'], row['callee'], row['syntax_sha256']


def test_local_call_accounting_and_provenance():
    ledger = json.loads(LEDGER.read_text())
    original = json.loads((ROOT / 'tests/fixtures/digest_parent_completion_1301.json').read_text())
    live = calls(ledger['assigned_files'])
    remaining = Counter(identity(row) for row in live)
    replaced = []
    for row in original['calls']:
        if row['file'] not in ledger['assigned_files']:
            continue
        key = identity(row)
        if remaining[key]:
            remaining[key] -= 1
        else:
            replaced.append(row)
    completed = {row['original_id']: row for row in ledger['completed_calls']}
    assert len(completed) == len(ledger['completed_calls'])
    assert set(completed) == {row['id'] for row in replaced}
    for row in replaced:
        evidence = completed[row['id']]
        assert evidence['source_commit'] == original['source_commit']
        assert evidence['source_blob'] == row['source_blob']
        assert evidence['owner_route'] and evidence['fixtures'] and evidence['tests']
        for fixture in evidence['fixtures']:
            assert fixture.startswith('tests/fixtures/digest_1301/quality-calibration/')
            assert (ROOT / fixture).is_file()
        for node in evidence['tests']:
            path, *names = node.split('::')
            assert path.startswith('tests/test_digest_1301/quality-calibration/')
            body = ast.parse((ROOT / path).read_text()).body
            for name in names:
                matches = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
                assert len(matches) == 1, node
                body = matches[0].body
            assert names and names[-1].startswith('test_')
    recorded = {row['id']: row for row in ledger['unresolved_calls']}
    assert set(recorded) == {row['id'] for row in live}
    for row in live:
        assert recorded[row['id']]['source_blob'] == row['source_blob']
        assert recorded[row['id']]['expression'] == row['expression']
        assert recorded[row['id']]['status'] == 'unresolved-owner-recipe-gap'
    assert not ledger['retentions'], 'Recipe gaps are unfinished work, not retentions'
