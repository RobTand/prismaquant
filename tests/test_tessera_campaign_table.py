"""Campaign metadata decisions work without model or serving runtime imports."""
from pathlib import Path
import json
import subprocess
import sys

import pytest


@pytest.fixture(scope='module')
def measured_joint_row():
    from test_joint_aura_streamed import _fixture, _run

    _, _, runner, cache = _fixture()
    payload = _run(runner, cache)
    name = next(iter(payload['costs']))
    return name, payload['costs'][name]['FP8_E4M3']


@pytest.mark.parametrize('case', ['ordinary', 'joint', 'malformed', 'rekeyed'])
def test_cold_table_coverage_validates_actual_joint_rows(tmp_path, measured_joint_row, case):
    """The cold consumer admits real measured rows and retains its refusals."""
    root = Path(__file__).resolve().parents[1]
    name, row = measured_joint_row
    child = '''
import json
import sys
sys.path.insert(0, ROOT)
name, row, case = json.load(sys.stdin)
class ForbiddenRuntime:
    def find_spec(self, name, path=None, target=None):
        if name in {'prismaquant.joint_aura', 'prismaquant.allocator_candidates'} or name.split('.')[0] in {'torch', 'numpy', 'transformers', 'triton', 'vllm', 'tools'} or name == 'tessera.serving' or name.startswith('tessera.serving.'):
            raise AssertionError('numerical dependency in table coverage: ' + name)
sys.meta_path.insert(0, ForbiddenRuntime())
from prismaquant.tessera_campaign_table import bound_joint_cells, ScopeRefused
if case == 'ordinary':
    assert bound_joint_cells({name: {'BF16': {}}}, label='cold table') == set()
elif case == 'joint':
    assert bound_joint_cells({name: {'FP8_E4M3': row}}, label='cold table') == {(name, 'FP8_E4M3')}
elif case == 'malformed':
    row['predicted_dloss'] += 1
    try:
        bound_joint_cells({name: {'FP8_E4M3': row}}, label='cold table')
    except ScopeRefused as error:
        assert 'predicted loss differs' in str(error)
    else:
        raise AssertionError('malformed samples admitted')
else:
    try:
        bound_joint_cells({name + '.donated': {'FP8_E4M3': row}}, label='cold table')
    except ScopeRefused as error:
        assert 'produced for' in str(error)
    else:
        raise AssertionError('rekeyed joint row admitted')
print('COLD-COVERAGE-OK')
'''.replace('ROOT', repr(str(root)), 1)
    result = subprocess.run([sys.executable, '-c', child], cwd=tmp_path,
                            input=json.dumps([name, row, case]), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'COLD-COVERAGE-OK'
