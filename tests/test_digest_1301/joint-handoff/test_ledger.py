"""Local handoff ledger completeness, without changing the shared census/ratchet."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/digest_1301/joint-handoff'
FILES = {
    'tools/build_stagea_forward_recovery_package.py',
    'tools/build_stagea_seed_package.py',
    'tools/build_stagea_split_package.py',
    'tools/check_readset_coverage.py',
    'tools/dispatch_joint_quanta.py',
    'tools/dispatch_stage_a_split.py',
    'tools/prepare_extended_joint_quanta.py',
    'tools/regenerate_joint_quanta.py',
    'tools/stage_b_preparation_submission.py',
    'tools/generate_transition_rewrites.py',
}


def test_local_ledger_accounts_for_frozen_calls():
    registry = json.loads((ROOT / 'tests/fixtures/digest_parent_completion_1301.json').read_text())
    ledger = json.loads((FIXTURES / 'ledger.json').read_text())
    expected = {r['id']: r for r in registry['calls'] if r['file'] in FILES}
    rows = ledger['completed_calls'] + ledger['unresolved_calls']
    assert len(rows) == len({r['original_id'] for r in rows})
    assert {r['original_id'] for r in rows} == set(expected)
    for row in rows:
        assert row['source_commit'] == registry['source_commit']
        assert row['source_blob'] == expected[row['original_id']]['source_blob']
    for row in ledger['completed_calls']:
        assert row['owner_route'].startswith('prismaquant.digests.')
        assert row['fixtures'] and row['tests']
        for fixture in row['fixtures']:
            assert (ROOT / fixture).is_file()
        for node in row['tests']:
            path, name = node.split('::')
            assert '[' not in name
            tree = ast.parse((ROOT / path).read_text())
            assert any(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body)
    assert len(ledger['completed_calls']) == 41
    assert len(ledger['unresolved_calls']) == 15
    assert len(ledger['supplemental_embedded_retentions']) == 1
