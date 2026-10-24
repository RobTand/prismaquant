"""Local integration fragment membership and research-base provenance checks."""
import ast
import hashlib
import json
from pathlib import Path

from tools import duplication_inventory as inventory

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / 'tests/fixtures/digest_1301/serving-evidence/ledger.json'


def test_fragment_covers_current_whole_files():
    ledger = json.loads(LEDGER.read_text())
    observed = {}
    for filename, identity in ledger['source_blobs'].items():
        raw = (ROOT / filename).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        assert identity == {'git_blob': blob, 'sha256': hashlib.sha256(raw).hexdigest()}
        source = raw.decode()
        for call, scope, aliases in inventory._iter_scoped_calls(ast.parse(source)):
            callee = inventory._resolved_callee(call.func, aliases)
            if callee and callee.startswith('hashlib.'):
                family = 'hashlib'
            elif callee in {'json.dumps', 'json.dump', 'json.JSONEncoder'}:
                family = 'sorted-json' if callee != 'json.dump' and any(
                    k.arg == 'sort_keys' and isinstance(k.value, ast.Constant)
                    and k.value.value is True for k in call.keywords) else 'supplemental-json'
            else:
                continue
            key = f'{family}:{filename}::{scope}@{call.lineno}:{call.col_offset}'
            observed[key] = (blob, ast.get_source_segment(source, call))
    rows = ledger['calls'] + ledger['supplemental_calls']
    assert len(rows) == len({r['id'] for r in rows})
    assert {r['id']: (r['source_blob'], r['expression']) for r in rows} == observed
    for row in rows:
        assert (ROOT / row['contract_evidence']).is_file()
        if row['disposition'] == 'blocked-eligible':
            assert row['blocked_reason']
        else:
            assert row['retention_reason']


def test_completed_calls_use_frozen_registry_and_executable_consumer_nodes():
    ledger = json.loads(LEDGER.read_text())
    registry = json.loads((ROOT / 'tests/fixtures/digest_parent_completion_1301.json').read_text())
    frozen = {r['id']: r for r in registry['calls']}
    assert len(ledger['completed_calls']) == 5
    for row in ledger['completed_calls']:
        assert row['source_commit'] == registry['source_commit']
        assert row['source_blob'] == frozen[row['original_id']]['source_blob']
        assert row['owner_route'].startswith('prismaquant.digests.')
        for fixture in row['fixtures']:
            assert (ROOT / fixture).is_file()
        for node in row['tests']:
            filename, function = node.split('::')
            assert filename.startswith('tests/test_digest_1301/serving-evidence/')
            assert function.startswith('test_') and '[' not in function
            tree = ast.parse((ROOT / filename).read_text())
            assert any(isinstance(n, ast.FunctionDef) and n.name == function for n in tree.body)
