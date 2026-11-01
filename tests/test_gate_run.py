"""Test outcomes are final; an absent outcome alone permits an infrastructure retry."""
from __future__ import annotations

import importlib.util
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gate_run', ROOT / 'tools/ci/gate_run.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def fixture_cost_policy(tmp_path):
    target = tmp_path / 'tools/ci/gate_budget.json'
    target.parent.mkdir(parents=True)
    target.write_text('{"seconds":20,"slow":{}}')


def fixture_junit(tmp_path):
    (tmp_path / 'junit.xml').write_text(
        '<testsuites><testcase file="tests/test_one.py" classname="tests.test_one" '
        'name="test_one" time="0.01"/></testsuites>')


@pytest.mark.parametrize('code,verdict', [(0, 'pass'), (1, 'fail')])
def test_completed_pytest_writes_a_final_verdict(tmp_path, code, verdict):
    calls = []
    def execute(command, **kwargs):
        calls.append((command, kwargs))
        fixture_junit(tmp_path)
        return SimpleNamespace(returncode=code)
    result = runner.run_plan({'selected': [{'path': 'tests/test_one.py'}],
                             'deselect': ['tests/test_one.py::test_bad']},
                            root=tmp_path, output=tmp_path / 'output', junit=tmp_path / 'junit.xml',
                            execute=execute, affinity={11, 12, 13, 14, 15})
    assert result == (0 if code == 0 else 1)
    assert json.loads((tmp_path / 'output').read_text())['verdict'] == verdict
    command, kwargs = calls[0]
    assert command[command.index('-n') + 1] == '5'
    assert '--deselect=tests/test_one.py::test_bad' in command
    assert json.loads(kwargs['env']['GATE_ASSIGNED_FILES']) == ['tests/test_one.py']
    assert command[-1] == str(tmp_path)
    assert kwargs['env']['PRISMABUILD_TEST_TIMEOUT_S'] == '300'
    assert kwargs['env']['OMP_NUM_THREADS'] == kwargs['env']['OPENBLAS_NUM_THREADS'] == '1'


@pytest.mark.parametrize('code', [2, 3, 4, 5])
def test_pytest_without_complete_test_verdict_is_infrastructure(tmp_path, code):
    result = runner.run_plan({'selected': [{'path': 'tests/test_one.py'}], 'deselect': []},
        root=tmp_path, output=tmp_path / 'output', junit=tmp_path / 'junit.xml',
        execute=lambda *a, **k: SimpleNamespace(returncode=code), affinity={1})
    assert result == 75
    assert json.loads((tmp_path / 'output').read_text())['verdict'] is None


def test_launch_failure_leaves_no_verdict(tmp_path):
    def unavailable(*args, **kwargs):
        raise OSError('interpreter unavailable')
    result = runner.run_plan({'selected': [{'path': 'tests/test_one.py'}], 'deselect': []},
                        root=tmp_path, output=tmp_path / 'output', junit=tmp_path / 'junit.xml',
                        execute=unavailable, affinity={1})
    assert result == 75
    assert json.loads((tmp_path / 'output').read_text())['verdict'] is None


def test_killed_pytest_leaves_no_verdict(tmp_path):
    result = runner.run_plan({'selected': [{'path': 'tests/test_one.py'}], 'deselect': []},
                        root=tmp_path, output=tmp_path / 'output', junit=tmp_path / 'junit.xml',
                        execute=lambda *a, **k: SimpleNamespace(returncode=-9), affinity={1})
    assert result == 75
    assert json.loads((tmp_path / 'output').read_text())['verdict'] is None


def test_empty_selection_cannot_default_to_running_the_entire_suite(tmp_path):
    result = runner.run_plan({'selected': [], 'deselect': []}, root=tmp_path,
                        output=tmp_path / 'output', junit=tmp_path / 'junit.xml',
                        execute=lambda *a, **k: pytest.fail('must not start pytest'), affinity={1})
    assert result == 75


def test_runner_uses_assigned_files_and_job_cpus_with_portable_junit(tmp_path):
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'pytest.ini').write_text('[pytest]\n')
    (tmp_path / 'tests/test_sample.py').write_text(
        'import math\ndef test_math(): assert math.isqrt(81) == 8\n')
    result = runner.run_plan({'selected': [{'path': 'tests/test_sample.py'},
                                         {'path': 'tests/test_unassigned.py'}], 'deselect': []},
        files=['tests/test_sample.py'], workers=1, root=tmp_path,
        output=tmp_path / 'result.json', junit=tmp_path / 'junit.xml')
    assert result == 1
    record = json.loads((tmp_path / 'result.json').read_text())
    assert record['files'] == ['tests/test_sample.py'] and record['verdict'] == 'fail'
    cases = list(ET.parse(tmp_path / 'junit.xml').iter('testcase'))
    assert len(cases) == 1 and cases[0].get('file') == 'tests/test_sample.py'
    assert cases[0].find('failure') is not None


def test_runner_rejects_files_outside_the_plan_before_launch(tmp_path):
    result = runner.run_plan({'selected': [{'path': 'tests/test_one.py'}], 'deselect': []},
        files=['tests/test_other.py'], root=tmp_path, output=tmp_path / 'output',
        junit=tmp_path / 'junit.xml', execute=lambda *a, **k: pytest.fail('must not launch'))
    assert result == 75


def test_newline_assignment_preserves_a_path_containing_spaces(tmp_path):
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'pytest.ini').write_text('[pytest]\n')
    filename = 'tests/test with spaces.py'
    (tmp_path / filename).write_text('def test_value(): assert 7 * 6 == 42\n')
    (tmp_path / 'plan.json').write_text(json.dumps({'selected': [{'path': filename}], 'deselect': []}))
    (tmp_path / 'assigned.txt').write_text(filename + '\n')
    result = runner.main(['--plan', str(tmp_path / 'plan.json'), '--root', str(tmp_path),
        '--files-from', str(tmp_path / 'assigned.txt'), '--workers', '1',
        '--result', str(tmp_path / 'result.json'), '--junitxml', str(tmp_path / 'junit.xml')])
    assert result == 0
    assert json.loads((tmp_path / 'result.json').read_text())['files'] == [filename]


def test_ci_python_loader_environment_does_not_leak_into_proof_capture(tmp_path, monkeypatch):
    monkeypatch.setenv('LD_LIBRARY_PATH', '/hosted/python/lib')
    monkeypatch.setenv('LD_PRELOAD', '/hosted/injected.so')
    monkeypatch.setenv('GATE_KEEP', 'retained')
    observed = []
    def execute(command, **kwargs):
        observed.append(kwargs['env'])
        fixture_junit(tmp_path)
        return SimpleNamespace(returncode=0)
    code = runner.run_plan({'selected': [{'path': 'tests/test_one.py'}], 'deselect': []},
        root=tmp_path, output=tmp_path / 'output', junit=tmp_path / 'junit.xml',
        execute=execute, affinity={1})
    assert code == 0
    assert 'LD_LIBRARY_PATH' not in observed[0] and 'LD_PRELOAD' not in observed[0]
    assert observed[0]['GATE_KEEP'] == 'retained'


@pytest.mark.parametrize('code', [0, 2, 3, -9])
@pytest.mark.parametrize('outcome', ['failure', 'error'])
def test_recorded_junit_failure_is_final_despite_pytest_exit(tmp_path, code, outcome):
    def execute(command, **kwargs):
        (tmp_path / 'junit.xml').write_text(
            '<testsuites><testcase file="tests/test_one.py" classname="tests.test_one" '
            f'name="test_one"><{outcome}>recorded defect</{outcome}></testcase></testsuites>')
        return SimpleNamespace(returncode=code)
    result = runner.run_plan({'selected': [{'path': 'tests/test_one.py'}], 'deselect': []},
        root=tmp_path, output=tmp_path / 'output', junit=tmp_path / 'junit.xml',
        execute=execute, affinity={1})
    record = json.loads((tmp_path / 'output').read_text())
    assert (result, record['verdict']) == (1, 'fail')


@pytest.mark.parametrize('code', [2, 3])
def test_infrastructure_exit_with_only_passing_junit_can_retry(tmp_path, code):
    def execute(command, **kwargs):
        fixture_junit(tmp_path)
        return SimpleNamespace(returncode=code)
    result = runner.run_plan({'selected': [{'path': 'tests/test_one.py'}], 'deselect': []},
        root=tmp_path, output=tmp_path / 'output', junit=tmp_path / 'junit.xml',
        execute=execute, affinity={1})
    assert result == 75
    assert json.loads((tmp_path / 'output').read_text())['verdict'] is None


def test_long_files_start_first_without_changing_assigned_coverage(tmp_path):
    files = ['tests/test_a_short.py', 'tests/test_b_medium.py', 'tests/test_z_long.py']
    observed = []
    def execute(command, **kwargs):
        observed.extend(json.loads(kwargs['env']['GATE_ASSIGNED_FILES']))
        (tmp_path / 'junit.xml').write_text('<testsuites>' + ''.join(
            f'<testcase file="{path}" classname="{path[:-3].replace("/", ".")}" '
            'name="test_value" time="1"/>' for path in files) + '</testsuites>')
        return SimpleNamespace(returncode=0)
    code = runner.run_plan({'selected': [
        {'path': files[0], 'estimated_seconds': 1},
        {'path': files[1], 'estimated_seconds': 5},
        {'path': files[2], 'estimated_seconds': 90}], 'deselect': []},
        files=files, root=tmp_path, output=tmp_path / 'output', junit=tmp_path / 'junit.xml',
        execute=execute, affinity={1, 2})
    assert code == 0
    assert observed == ['tests/test_z_long.py', 'tests/test_b_medium.py', 'tests/test_a_short.py']


def test_a_file_stays_on_one_worker_to_preserve_shared_fixtures(tmp_path):
    """Shared build and audit fixtures must not be duplicated across workers."""
    import json
    root = tmp_path / 'project'
    (root / 'tests').mkdir(parents=True)
    (root / 'tools/ci').mkdir(parents=True)
    (root / 'tools/ci/gate_budget.json').write_text(json.dumps({'seconds': 20, 'slow': {}}))
    (root / 'pyproject.toml').write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    path = 'tests/test_parallel_file.py'
    (root / path).write_text('''from pathlib import Path
import pytest

@pytest.mark.parametrize('case', range(4))
def test_case(case, worker_id):
    Path(__file__).with_name('worker-' + str(case)).write_text(worker_id)
''')
    plan = {'selected': [{'path': path, 'estimated_seconds': 100}], 'deselect': []}
    code = runner.run_plan(plan, root=root, output=root / 'result.json',
                           junit=root / 'junit.xml', workers=2, affinity={0, 1})
    assert code == 0
    assert len({(root / 'tests' / ('worker-' + str(case))).read_text() for case in range(4)}) == 1


def test_collection_scans_a_directory_once_and_imports_only_assigned_files(tmp_path):
    root = tmp_path
    (root / 'tests').mkdir()
    (root / 'pyproject.toml').write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    (root / 'conftest.py').write_text('''from pathlib import Path

def pytest_collectstart(collector):
    if collector.path.name == 'tests':
        with Path(__file__).with_name('directory-visits').open('a') as output:
            output.write('visit\\n')
''')
    for name in ['long', 'short']:
        (root / f'tests/test_{name}.py').write_text(f'''from pathlib import Path

def test_case():
    with Path(__file__).parents[1].joinpath('order').open('a') as output:
        output.write('{name}\\n')
''')
    (root / 'tests/test_unassigned.py').write_text('raise AssertionError("unassigned file was imported")\n')
    files = ['tests/test_short.py', 'tests/test_long.py']
    plan = {'selected': [{'path': files[0], 'estimated_seconds': 1},
                         {'path': files[1], 'estimated_seconds': 20}], 'deselect': []}
    code = runner.run_plan(plan, root=root, output=root / 'result.json',
                           junit=root / 'junit.xml', workers=1, affinity={0})
    assert code == 0
    assert (root / 'directory-visits').read_text().splitlines() == ['visit']
    assert (root / 'order').read_text().splitlines() == ['long', 'short']
    assert {case.get('file') for case in ET.parse(root / 'junit.xml').iter('testcase')} == set(files)


def test_assigned_collection_preserves_imported_dependency_bytecode(tmp_path):
    root = tmp_path
    (root / 'tests').mkdir()
    (root / 'pytest.ini').write_text('[pytest]\n')
    (root / 'gate_fixture_dependency.py').write_text('VALUE = 42\n')
    filename = 'tests/test_assigned.py'
    (root / filename).write_text('''import importlib.util
from pathlib import Path
import sys


def test_regular_dependency_bytecode():
    sys.dont_write_bytecode = False
    sys.pycache_prefix = None
    import gate_fixture_dependency
    assert gate_fixture_dependency.VALUE == 42
    assert Path(importlib.util.cache_from_source(gate_fixture_dependency.__file__)).is_file()
''')
    code = runner.run_plan({'selected': [{'path': filename}], 'deselect': []},
                           root=root, output=root / 'result.json', junit=root / 'junit.xml',
                           workers=1, affinity={0})
    assert code == 0


def test_assigned_collection_node_ids_match_plain_pytest(tmp_path):
    import subprocess
    import sys

    sources = {
        'pytest.ini': '[pytest]\n',
        'conftest.py': """import json
import os
from pathlib import Path
import pytest

@pytest.fixture(params=['root-a', 'root-b'])
def root_value(request):
    return request.param


def pytest_collection_finish(session):
    Path(os.environ['COLLECTION_REPORT']).write_text(
        json.dumps(sorted(item.nodeid for item in session.items)))
""",
        'tests/nested/conftest.py': """import pytest

@pytest.fixture(params=['nested-a', 'nested-b'])
def nested_value(request, root_value):
    return root_value, request.param


def pytest_generate_tests(metafunc):
    if 'generated' in metafunc.fixturenames:
        metafunc.parametrize('generated', [3, 7], ids=['three', 'seven'])
""",
        'tests/nested/test_generated.py':
            'def test_generated(nested_value, generated): assert generated > 0\n',
        'tests/pkg/__init__.py': 'VALUE = 42\n',
        'tests/pkg/sub/__init__.py': '',
        'tests/pkg/sub/test_package.py':
            'from .. import VALUE\ndef test_package(root_value): assert VALUE == 42\n',
        'tests/test_with spaces.py':
            'def test_spaces(root_value): assert root_value\n',
        'tests/test_é.py':
            'def test_unicode(root_value): assert root_value\n',
    }
    for filename, source in sources.items():
        target = tmp_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding='utf-8')
    files = ['tests/nested/test_generated.py', 'tests/pkg/sub/test_package.py',
             'tests/test_with spaces.py', 'tests/test_é.py']
    environment = {**os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
                   'PYTHONPATH': str(ROOT / 'tools/ci')}
    # An inherited -k or plugin would change both collections the same way and hide a difference.
    for name in ('GATE_ASSIGNED_FILES', 'PYTEST_ADDOPTS', 'PYTEST_PLUGINS'):
        environment.pop(name, None)
    collected = []
    for mode in ['plain', 'assigned']:
        report = tmp_path / (mode + '.json')
        env = {**environment, 'COLLECTION_REPORT': str(report)}
        arguments = [str(tmp_path)]
        if mode == 'assigned':
            env['GATE_ASSIGNED_FILES'] = json.dumps(files)
            arguments = ['-p', 'gate_collect', *arguments]
        completed = subprocess.run(
            [sys.executable, '-m', 'pytest', '--collect-only', '-q',
             '-c', str(tmp_path / 'pytest.ini'), '--rootdir', str(tmp_path), *arguments],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        collected.append(json.loads(report.read_text()))
    plain, assigned = collected
    assert len(plain) == 14
    assert assigned == plain
