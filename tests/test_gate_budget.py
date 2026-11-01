"""CI costs may shrink; complete test outcomes cannot hide missing cost evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gate_run', ROOT / 'tools/ci/gate_run.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def isolate_job_summary(monkeypatch):
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)


def policy(root, seconds=20, slow=None):
    target = root / 'tools/ci/gate_budget.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'seconds': seconds, 'slow': slow or {}}))


def budget_run(tmp_path, xml, *, base=None, pytest_code=0):
    plan = {'selected': [{'path': 'tests/test_sample.py'}], 'deselect': [], 'base_sha': base}
    def execute(*args, **kwargs):
        (tmp_path / 'junit.xml').write_text(xml)
        return SimpleNamespace(returncode=pytest_code)
    code = runner.run_plan(plan, root=tmp_path, output=tmp_path / 'result.json',
                           junit=tmp_path / 'junit.xml', execute=execute, affinity={1})
    return code, json.loads((tmp_path / 'result.json').read_text())


def case(seconds='21', *, classname='tests.test_sample', name='test_cost', time_attr=True):
    timing = f' time="{seconds}"' if time_attr else ''
    return (f'<testsuites><testsuite><testcase file="tests/test_sample.py" '
            f'classname="{classname}" name="{name}"{timing}/></testsuite></testsuites>')


def commit_target(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True).strip()
    git('init', '-q')
    git('add', '.')
    git('-c', 'user.name=Gate fixture', '-c', 'user.email=gate@example.invalid', 'commit', '-qm', 'Target')
    return git('rev-parse', 'HEAD')


def test_overrun_is_a_final_failure_even_when_pytest_passed(tmp_path, capsys):
    policy(tmp_path)
    code, result = budget_run(tmp_path, case('30.001'))
    assert code == 1
    assert result['verdict'] == 'fail' and result['pytest_exit_code'] == 0
    assert result['budget']['status'] == 'fail'
    assert result['budget']['overruns'] == [
        {'nodeid': 'tests/test_sample.py::test_cost', 'seconds': 30.001, 'budget_seconds': 20}]
    assert 'tests/test_sample.py::test_cost' in capsys.readouterr().out


@pytest.mark.parametrize('seconds', ['20.527', '30'])
def test_within_tolerance_passes_with_visible_warning(tmp_path, monkeypatch, capsys, seconds):
    policy(tmp_path)
    summary = tmp_path / 'summary.md'
    summary.write_text('Existing job summary\n')
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))
    xml = case(seconds).replace('</testsuite>', '<system-out>existing output</system-out></testsuite>')
    code, result = budget_run(tmp_path, xml)
    assert code == 0 and result['verdict'] == 'pass'
    assert result['budget']['overruns'] == []
    assert result['budget']['warnings'] == [
        {'nodeid': 'tests/test_sample.py::test_cost', 'seconds': float(seconds), 'budget_seconds': 20}]
    warning = 'tests/test_sample.py::test_cost'
    assert warning in capsys.readouterr().out
    assert summary.read_text().startswith('Existing job summary\n')
    assert warning in summary.read_text() and 'warning' in summary.read_text().lower()
    tree = ET.parse(tmp_path / 'junit.xml')
    output = '\n'.join(node.text or '' for node in tree.iter('system-out'))
    assert 'existing output' in output and warning in output and 'BUDGET_WARNING' in output
    testcase = next(tree.iter('testcase'))
    assert testcase.get('file') == 'tests/test_sample.py' and testcase.get('time') == seconds
    assert testcase.find('failure') is None and testcase.find('error') is None


@pytest.mark.parametrize('seconds,expected', [('63', 0), ('63.001', 1)])
def test_tolerance_uses_the_precise_slow_exception(tmp_path, seconds, expected):
    nodeid = 'tests/test_sample.py::TestCosts::test_cost[a.b]'
    policy(tmp_path, slow={nodeid: 42})
    code, result = budget_run(tmp_path, case(seconds, classname='tests.test_sample.TestCosts',
                                          name='test_cost[a.b]'))
    assert code == expected
    rows = result['budget']['warnings' if expected == 0 else 'overruns']
    assert rows == [{'nodeid': nodeid, 'seconds': float(seconds), 'budget_seconds': 42}]


def test_at_policy_limit_needs_no_warning(tmp_path):
    policy(tmp_path)
    code, result = budget_run(tmp_path, case('20'))
    assert code == 0 and result['budget']['warnings'] == []
    assert not list(ET.parse(tmp_path / 'junit.xml').iter('system-out'))


@pytest.mark.parametrize('seconds,slow', [
    (21, {}), (20, {'tests/test_sample.py::test_cost': 40}),
    (20, {'tests/test_sample.py::test_known': 41}),
])
def test_growth_is_checked_against_the_actual_target_commit(tmp_path, seconds, slow):
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40})
    base = commit_target(tmp_path)
    policy(tmp_path, seconds, slow)
    code, result = budget_run(tmp_path, case('0'), base=base)
    assert code == 1
    assert result['budget']['status'] == 'fail'
    assert 'grow' in result['budget']['reason']
    assert result['pytest_exit_code'] is None


def test_shrink_and_removed_exceptions_are_allowed(tmp_path):
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40, 'removed': 60})
    base = commit_target(tmp_path)
    policy(tmp_path, 10, {'tests/test_sample.py::test_known': 30})
    code, result = budget_run(tmp_path, case('10'), base=base)
    assert code == 0
    assert result['budget']['status'] == 'pass'
    assert result['budget']['baseline_sha'] == base


def test_absent_target_policy_bootstraps_but_missing_target_object_does_not(tmp_path):
    (tmp_path / 'tracked').write_text('baseline predates the policy')
    base = commit_target(tmp_path)
    policy(tmp_path)
    assert budget_run(tmp_path, case('1'), base=base)[0] == 0
    code, result = budget_run(tmp_path, case('1'), base='f' * 40)
    assert code == 75 and result['verdict'] is None
    assert 'target' in result['infrastructure_reason']


@pytest.mark.parametrize('xml', [
    case(time_attr=False), case('nan'), case('-1'), case('inf'),
    '<testsuites/>', '<not-xml',
    '<testsuites><testcase name="test_cost" time="1"/></testsuites>',
])
def test_missing_or_invalid_timing_is_no_verdict(tmp_path, xml):
    policy(tmp_path)
    code, result = budget_run(tmp_path, xml)
    assert code == 75 and result['verdict'] is None
    assert 'infrastructure_reason' in result


def test_known_test_failure_is_never_retried_even_if_junit_is_missing(tmp_path):
    policy(tmp_path)
    code, result = budget_run(tmp_path, '<not-xml', pytest_code=1)
    assert code == 1 and result['verdict'] == 'fail'
    assert result['pytest_exit_code'] == 1
    assert result['budget']['status'] == 'unavailable'


def test_class_and_parameter_id_use_the_precise_exception(tmp_path):
    policy(tmp_path, slow={'tests/test_sample.py::TestCosts::test_cost[a.b]': 42})
    code, result = budget_run(tmp_path, case('41', classname='tests.test_sample.TestCosts',
                                          name='test_cost[a.b]'))
    assert code == 0 and result['budget']['status'] == 'pass'


@pytest.mark.parametrize('seconds', [True, 0, -1, float('nan'), float('inf')])
def test_invalid_policy_limits_fail_closed(tmp_path, seconds):
    policy(tmp_path, seconds)
    code, result = budget_run(tmp_path, case('0'))
    assert code == 1 and result['budget']['status'] == 'fail'


@pytest.mark.parametrize('pytest_code,overrun,expected_code,expected_verdict', [
    (0, False, 75, None), (1, False, 1, 'fail'), (0, True, 1, 'fail'),
])
def test_warning_write_failure_preserves_definite_failure(tmp_path, monkeypatch,
                                                        pytest_code, overrun,
                                                        expected_code, expected_verdict):
    policy(tmp_path)
    xml = case('21')
    if overrun:
        xml = xml.replace('</testsuite>',
                          '<testcase file="tests/test_sample.py" classname="tests.test_sample" '
                          'name="test_overrun" time="31"/></testsuite>')
    if pytest_code == 1:
        xml = xml.replace('/></testsuite>', '><failure message="regression"/></testcase></testsuite>')
    def unavailable(*args):
        raise OSError('injected warning artifact write failure')
    monkeypatch.setattr(runner, '_write_budget_warnings', unavailable)
    code, result = budget_run(tmp_path, xml, pytest_code=pytest_code)
    assert code == expected_code and result['verdict'] == expected_verdict
    assert result['budget']['status'] == 'unavailable'
    assert result['budget']['reason'] == 'injected warning artifact write failure'


def test_warning_summary_has_a_bounded_preview_but_junit_keeps_every_warning(tmp_path, monkeypatch):
    summary = tmp_path / 'summary.md'
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))
    warnings = [{'nodeid': f'tests/test_cost.py::test_{i:04d}_' + 'x' * 1000,
                 'seconds': 21, 'budget_seconds': 20} for i in range(2000)]
    junit = tmp_path / 'junit.xml'
    junit.write_text('<testsuites><testsuite/></testsuites>')
    runner._write_budget_warnings(junit, warnings)
    text = summary.read_text()
    assert len(text.encode()) < 32 * 1024
    assert '2000' in text and '1975 more' in text
    assert 'test_0000_' in text and 'test_1999_' not in text
    assert 'junit.xml' in text
    assert 'test_1999_' in junit.read_text()
