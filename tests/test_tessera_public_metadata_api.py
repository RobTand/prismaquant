"""The producer metadata API supplies facts without importing serving code."""
import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest
from tessera import metadata

from prismaquant import lane_eligibility as lane
from prismaquant.tessera_runtime_contract import installed_runtime_snapshot, RuntimeContractSnapshot


@pytest.fixture
def fresh_snapshot():
    return RuntimeContractSnapshot(installed_runtime_snapshot().raw, 'fresh-api-boundary')


def test_missing_api_version_refuses_the_table(monkeypatch, fresh_snapshot):
    monkeypatch.setattr(metadata, 'METADATA_API_VERSION', 'unknown')
    with pytest.raises(lane.LaneEligibilityError, match='incompatible Tessera metadata API'):
        fresh_snapshot.eligibility


def test_missing_public_capability_refuses_the_table(monkeypatch, fresh_snapshot):
    monkeypatch.delattr(metadata, 'parse_smoke_record')
    with pytest.raises(lane.LaneEligibilityError, match='callable.*parse_smoke_record'):
        fresh_snapshot.eligibility


def test_missing_public_module_refuses_without_a_serving_fallback(monkeypatch, fresh_snapshot):
    monkeypatch.setitem(sys.modules, 'tessera.metadata', None)
    with pytest.raises(lane.LaneEligibilityError, match='public metadata API'):
        fresh_snapshot.eligibility


def test_the_public_grammar_returns_caller_owned_record_rows():
    record = {'instrument': 'tools/smoke.py', 'rule': 'owned observed scoring rule',
              'reference': None, 'rows': [{'prompt': 'p', 'form': 'pure_greedy',
              'interface': 'raw_completion', 'status': 'recorded', 'reference_status': None}]}
    before = copy.deepcopy(record)
    result = metadata.parse_smoke_record(record, control=None, where='owned.record')
    result['rows'][0]['status'] = 'repetitive'
    assert record == before
    assert metadata.derive_smoke_status({'status': 'recorded', 'record': record}) == 'recorded'


def test_pinned_metadata_parses_in_a_fresh_serving_free_process(tmp_path):
    root = Path(__file__).resolve().parents[1]
    child = '''
import sys
class ForbiddenRuntime:
    def find_spec(self, name, path=None, target=None):
        if name == 'tessera.serving' or name.startswith('tessera.serving.') or name.split('.')[0] in {'torch', 'transformers', 'triton', 'vllm'}:
            raise AssertionError('forbidden runtime import: ' + name)
sys.meta_path.insert(0, ForbiddenRuntime())
from prismaquant.tessera_runtime_contract import installed_runtime_snapshot
snapshot = installed_runtime_snapshot()
assert len(snapshot.eligibility.cells) == 22
assert snapshot.sha256 == 'ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e'
assert not any(name == 'tessera.serving' or name.startswith('tessera.serving.') for name in sys.modules)
print('PUBLIC-METADATA-OK')
'''
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join([str(Path(metadata.__file__).resolve().parents[1]), str(root)])
    result = subprocess.run([sys.executable, '-c', child], cwd=tmp_path,
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'PUBLIC-METADATA-OK'


def test_metadata_readers_keep_the_serving_runtime_outside_the_dependency_boundary():
    """The migrated admission boundary cannot reacquire serving dependencies."""
    import ast

    root = Path(__file__).resolve().parents[1]
    violations = []
    for name in ('lane_eligibility.py', 'tessera_runtime_contract.py'):
        path = root / 'prismaquant' / name
        for node in ast.walk(ast.parse(path.read_text())):
            imported = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or '']
                if node.module == 'tessera':
                    imported.extend('tessera.' + alias.name for alias in node.names)
            elif isinstance(node, ast.Call):
                function = ast.unparse(node.func)
                if function.endswith(('.files', '.import_module')) or function == '__import__':
                    imported = [arg.value for arg in node.args
                                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
            if any(name == 'tessera.serving' or name.startswith('tessera.serving.')
                   for name in imported):
                violations.append(f'{path.relative_to(root)}:{node.lineno}')
    assert not violations, violations
