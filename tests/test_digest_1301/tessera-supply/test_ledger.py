"""Local supply-chain accounting; not a substitute for consumer compatibility."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from tools import duplication_inventory as inventory

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / 'tests/fixtures/digest_1301/tessera-supply/ledger.json'
REGISTRY = ROOT / 'tests/fixtures/digest_parent_completion_1301.json'


def test_supply_ledger_exact_registry_and_current_membership():
    ledger = json.loads(LEDGER.read_text())
    original = json.loads(REGISTRY.read_text())
    registry = {r['id']: r for r in original['calls'] if r['file'] in ledger['scope']}
    rows = ledger['completed_calls'] + ledger['remaining_calls']
    assert len(rows) == len(registry)
    assert {r['original_id'] for r in rows} == registry.keys()
    for row in rows:
        assert row['source_commit'] == original['source_commit']
        assert row['source_blob'] == registry[row['original_id']]['source_blob']
    live = []
    for rel in ledger['scope']:
        raw = (ROOT / rel).read_bytes()
        source = raw.decode()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        for call, scope, aliases in inventory._iter_scoped_calls(ast.parse(source)):
            callee = inventory._resolved_callee(call.func, aliases)
            if callee and callee.startswith('hashlib.'):
                family = 'hashlib'
            elif callee in {'json.dumps', 'json.dump', 'json.JSONEncoder'}:
                sorted_keys = callee != 'json.dump' and any(
                    k.arg == 'sort_keys' and isinstance(k.value, ast.Constant)
                    and k.value.value is True for k in call.keywords)
                family = 'sorted-json' if sorted_keys else 'supplemental-json'
            else:
                continue
            live.append({'id': f'{family}:{rel}::{scope}@{call.lineno}:{call.col_offset}',
                         'source_blob': blob,
                         'expression': ast.get_source_segment(source, call)})
    assert sorted(live, key=lambda r: r['id']) == ledger['current_calls']
    # Missing exact recipes are explicitly unfinished, never disguised as retentions.
    retained = [r for r in ledger['remaining_calls'] if r['status'] == 'retained']
    assert {r['scope'] for r in retained} == {'_scan_file', '_replay_frozen_digests'}
    assert all('descriptor' in r['reason'].lower() for r in retained)
    assert any(r['scope'] == 'tree_digest' and r['status'] == 'unresolved'
               for r in ledger['remaining_calls'])


def test_completed_supply_references_resolve_without_history():
    ledger = json.loads(LEDGER.read_text())
    for row in ledger['completed_calls']:
        assert row['owner_route'].startswith('prismaquant.digests.')
        assert row['fixtures'] and row['tests']
        for name in row['fixtures']:
            assert not Path(name).is_absolute()
            assert (ROOT / name).is_file(), name
        for reference in row['tests']:
            file, *names = reference.split('::')
            assert file.startswith('tests/test_') and file.endswith('.py')
            assert names and '[' not in reference
            nodes = ast.parse((ROOT / file).read_text()).body
            for name in names:
                matches = [n for n in nodes if isinstance(
                    n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name]
                assert len(matches) == 1, reference
                nodes = matches[0].body


def test_progress_report_checkout_layout_stays_stdlib_only(tmp_path):
    import subprocess
    import sys

    # Stage only the tool and its stdlib owner; no package initializer/site packages.
    (tmp_path / 'tools').mkdir()
    (tmp_path / 'prismaquant').mkdir()
    for name in ('tools/derive_tessera_progress_grace.py', 'prismaquant/digests.py'):
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    script = '''import runpy, sys
from pathlib import Path
ns = runpy.run_path(sys.argv[1])
p = Path(sys.argv[2])
p.write_bytes(b"\\x00\\xfforiginal bytes")
assert ns['sha256'](p) == '2711d3ffbc6a74075ffbeb1f023bd8ee37a1c3afadb5063351c7647c01188f5b'
assert 'prismaquant' not in sys.modules and 'torch' not in sys.modules
'''
    # Literal digest independently frozen from the original hashlib expression.
    result = subprocess.run([sys.executable, '-I', '-S', '-c', script,
                             str(tmp_path / 'tools/derive_tessera_progress_grace.py'),
                             str(tmp_path / 'input.bin')],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
