"""Local integration ledger completeness, without modifying the shared census."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / 'tests/fixtures/digest_1301/orchestration-runtime/ledger.json'
FILES = {
    f'prismaquant/{name}.py' for name in (
        'pipeline', 'cluster_campaign', 'artifact_registry', 'dev_mode',
        'io_engine', 'io_spans', 'autoscale', 'day0_model_intake',
        'streaming_initialization', 'streaming_model', 'ship_gates',
        'release_receipts', 'container_runtime_identity', 'prismabuild_progress',
    )
}


def test_local_ledger_covers_frozen_calls_and_executable_evidence():
    ledger = json.loads(LEDGER.read_text())
    frozen = json.loads((ROOT / 'tests/fixtures/digest_parent_completion_1301.json').read_text())
    expected = {row['id']: row for row in frozen['calls'] if row['file'] in FILES}
    rows = [row for key in ('completed_calls', 'blocked_calls', 'retained_calls')
            for row in ledger[key]]
    assert len(rows) == len({row['original_id'] for row in rows})
    assert {row['original_id'] for row in rows} == expected.keys()
    assert not ledger['parent_closure']  # Missing exact owners are not retentions.
    for row in rows:
        original = expected[row['original_id']]
        assert row['source_commit'] == '6d8288adc94b3c4b624c143ed372fb6513f32784'
        assert row['source_blob'] == original['source_blob']
        source = (ROOT / row['file']).read_bytes()
        assert row['current_blob'] == hashlib.sha1(
            b'blob ' + str(len(source)).encode() + b'\0' + source).hexdigest()
        assert row['fixtures'] and row['tests']
        for fixture in row['fixtures']:
            assert hashlib.sha256((ROOT / fixture).read_bytes()).hexdigest() == row['fixture_sha256s'][fixture]
        for node in row['tests']:
            path, function = node.split('::')
            assert path.startswith('tests/test_digest_1301/orchestration-runtime/test_')
            assert '[' not in function
            tree = ast.parse((ROOT / path).read_text())
            assert any(isinstance(n, ast.FunctionDef) and n.name == function for n in tree.body)
    for row in ledger['completed_calls']:
        assert row['owner_route'].startswith('prismaquant.digests.')
