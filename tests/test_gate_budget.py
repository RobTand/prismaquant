"""CI costs may shrink; complete test outcomes cannot hide missing cost evidence."""
import importlib.util
import json
import math
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


def policy(root, seconds=20, slow=None, file_seconds=None):
    target = root / 'tools/ci/gate_budget.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'seconds': seconds, 'slow': slow or {}, 'file_seconds': file_seconds or {}}))


def budget_run(tmp_path, xml, *, base=None, pytest_code=0):
    plan = {'selected': [{'path': 'tests/test_sample.py'}], 'deselect': [], 'base_sha': base}
    def execute(command, **kwargs):
        report = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--junitxml=')))
        report.write_text(xml)
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


@pytest.mark.parametrize('seconds', ['20.527', '30', '30.001', '120'])
def test_overrun_passes_with_visible_warning(tmp_path, monkeypatch, capsys, seconds):
    policy(tmp_path)
    summary = tmp_path / 'summary.md'
    summary.write_text('Existing job summary\n')
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))
    xml = case(seconds).replace('</testsuite>', '<system-out>existing output</system-out></testsuite>')
    code, result = budget_run(tmp_path, xml)
    assert code == 0 and result['verdict'] == 'pass'
    assert result['budget_warnings'] == [
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


@pytest.mark.parametrize('seconds,expected', [('63', 0), ('63.001', 0)])
def test_warning_uses_the_precise_slow_exception(tmp_path, seconds, expected):
    nodeid = 'tests/test_sample.py::TestCosts::test_cost[a.b]'
    policy(tmp_path, slow={nodeid: 42})
    code, result = budget_run(tmp_path, case(seconds, classname='tests.test_sample.TestCosts',
                                          name='test_cost[a.b]'))
    assert code == expected
    rows = result['budget_warnings']
    assert rows == [{'nodeid': nodeid, 'seconds': float(seconds), 'budget_seconds': 42}]


def test_at_policy_limit_needs_no_warning(tmp_path):
    policy(tmp_path)
    code, result = budget_run(tmp_path, case('20'))
    assert code == 0 and result['budget_warnings'] == []
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
    (0, False, 75, None), (1, False, 1, 'fail'), (0, True, 75, None),
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


def test_enabled_enforcement_cannot_be_disabled_against_base(tmp_path):
    policy(tmp_path)
    base = commit_target(tmp_path)
    path = tmp_path / 'tools/ci/gate_budget.json'
    payload = json.loads(path.read_text())
    payload['enforced'] = False
    path.write_text(json.dumps(payload))
    code, result = budget_run(tmp_path, case('1'), base=base)
    assert code == 1
    assert result['verdict'] == 'fail'


def test_imported_test_budget_uses_collected_node_not_definition(tmp_path):
    policy(tmp_path, slow={'tests/test_sample.py::test_cost': 42})
    xml = case('43').replace('file="tests/test_sample.py"', 'file="tests/helpers.py"')
    xml = xml.replace('/></testsuite>', '><properties><property name="gate_nodeid" '
        'value="tests/test_sample.py::test_cost"/></properties></testcase></testsuite>')
    code, result = budget_run(tmp_path, xml)
    assert code == 0
    assert result['budget_warnings'][0]['nodeid'] == 'tests/test_sample.py::test_cost'


@pytest.mark.parametrize('identity', ['tests/foreign.py::test_cost',
                                      'tests/test_sample.py::other', ''])
def test_inconsistent_collected_identity_cannot_supply_a_verdict(tmp_path, identity):
    policy(tmp_path)
    xml = case('1').replace('/></testsuite>',
        '><properties><property name="gate_nodeid" value="' + identity +
        '"/></properties></testcase></testsuite>')
    code, result = budget_run(tmp_path, xml)
    assert code == 75 and result['verdict'] is None


@pytest.mark.parametrize('first,second,expected', [(55, 1, 0), (1, 55, 0), (46, 45, 0)])
def test_shared_fixture_file_cost_is_independent_of_the_first_test(tmp_path, first, second, expected):
    policy(tmp_path, file_seconds={'tests/test_sample.py': 60})
    xml = case(str(first)).replace('</testsuite>',
        '<testcase file="tests/test_sample.py" classname="tests.test_sample" '
        f'name="test_other" time="{second}"/></testsuite>')
    code, result = budget_run(tmp_path, xml)
    assert code == expected
    assert result['budget_warnings'] == ([
        {'nodeid': 'tests/test_sample.py', 'seconds': 91, 'budget_seconds': 60}]
        if first + second == 91 else [])


def test_shared_fixture_file_warning_retains_the_complete_cost(tmp_path):
    policy(tmp_path, file_seconds={'tests/test_sample.py': 40})
    code, result = budget_run(tmp_path, case('55'))
    assert code == 0
    assert result['budget_warnings'] == [
        {'nodeid': 'tests/test_sample.py', 'seconds': 55, 'budget_seconds': 40}]


@pytest.mark.parametrize('baseline,current', [({}, {'tests/test_sample.py': 40}),
                                            ({'tests/test_sample.py': 40}, {'tests/test_sample.py': 41})])
def test_file_budget_additions_and_growth_are_rejected_against_base(tmp_path, baseline, current):
    policy(tmp_path, file_seconds=baseline)
    base = commit_target(tmp_path)
    policy(tmp_path, file_seconds=current)
    code, result = budget_run(tmp_path, case('1'), base=base)
    assert code == 1 and result['budget']['status'] == 'fail'


def test_file_budget_can_shrink_or_be_removed(tmp_path):
    policy(tmp_path, file_seconds={'tests/test_sample.py': 40})
    base = commit_target(tmp_path)
    policy(tmp_path, file_seconds={'tests/test_sample.py': 30})
    assert budget_run(tmp_path, case('25'), base=base)[0] == 0
    policy(tmp_path)
    assert budget_run(tmp_path, case('10'), base=base)[0] == 0


@pytest.mark.parametrize('file_seconds', [{'tests/test_sample.py': 0},
    {'tests/test_sample.py': True}, {'tests/test_sample.py': float('nan')},
    {'../outside.py': 40}, {'tests/test_sample.py::test_cost': 40}])
def test_invalid_file_budget_is_a_policy_failure(tmp_path, file_seconds):
    policy(tmp_path, file_seconds=file_seconds)
    code, result = budget_run(tmp_path, case('1'))
    assert code == 1 and result['budget']['status'] == 'fail'


@pytest.mark.parametrize('file_budget', [False, True])
@pytest.mark.parametrize('failure', [None, 'failure', 'error'])
def test_budget_warning_never_starts_a_second_execution(tmp_path, monkeypatch, file_budget, failure):
    policy(tmp_path, file_seconds={'tests/test_sample.py': 20} if file_budget else None)
    summary = tmp_path / 'summary.md'
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        report = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--junitxml=')))
        xml = case('120')
        if failure:
            xml = xml.replace('/></testsuite>',
                f'><{failure} message="assertion or pytest TIMEOUT"/></testcase></testsuite>')
        report.write_text(xml)
        return SimpleNamespace(returncode=1 if failure else 0)
    code = runner.run_plan({'selected': [{'path': 'tests/test_sample.py'}], 'deselect': []},
        root=tmp_path, output=tmp_path / 'result.json', junit=tmp_path / 'junit.xml',
        execute=execute, affinity={1})
    result = json.loads((tmp_path / 'result.json').read_text())
    assert code == (1 if failure else 0)
    assert len(calls) == 1
    node = 'tests/test_sample.py' + ('' if file_budget else '::test_cost')
    assert result['budget_warnings'] == [{'nodeid': node, 'seconds': 120, 'budget_seconds': 20}]
    assert 'budget_warnings' in summary.read_text()
    assert node in summary.read_text() and '120.000s > 20.000s' in summary.read_text()
    assert 'budget_reruns' not in result
    assert not list(tmp_path.glob('junit.budget-rerun-*'))


def test_hard_process_timeout_fails_without_a_second_execution(tmp_path):
    policy(tmp_path)
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 600)
    code = runner.run_plan({'selected': [{'path': 'tests/test_sample.py'}], 'deselect': []},
        root=tmp_path, output=tmp_path / 'result.json', junit=tmp_path / 'junit.xml',
        execute=execute, affinity={1})
    result = json.loads((tmp_path / 'result.json').read_text())
    assert code == 1 and result['verdict'] == 'fail'
    assert result['pytest_exit_code'] == 124
    assert result['timeout_seconds'] == 600
    assert len(calls) == 1


NODE = 'tests/test_sample.py::test_cost'
MIGRATIONS = 'tools/ci/gate_budget_migrations.json'
X64, ARM = 'ubuntu-24.04', 'ubuntu-24.04-arm'


def evidence(x64=40.2, arm=31.5):
    return [{'run': 101, 'attempt': 1, 'runner': X64, 'seconds': x64},
            {'run': 102, 'attempt': 1, 'runner': ARM, 'seconds': arm}]


def base_policy_blob(root):
    return subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD:tools/ci/gate_budget.json'], text=True).strip()


def record(root, blob, nodes, *, reason='dependency change: PrismaBuild SDK installed, #3223'):
    (root / MIGRATIONS).write_text(json.dumps({
        'schema': 'prismaquant.gate_budget_migration.v1',
        'migrations': [{'base_policy': blob, 'base_sha': 'a' * 40, 'reason': reason,
                        'slow': nodes}]}))


def entry(seconds=None, rows=None):
    rows = evidence() if rows is None else rows
    limit = math.ceil(2 * max(row['seconds'] for row in rows)) if seconds is None else seconds
    return {'seconds': limit, 'evidence': rows}


def based(tmp_path):
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40})
    base = commit_target(tmp_path)
    return base, base_policy_blob(tmp_path)


def test_a_listed_node_may_be_added_at_its_measured_limit(tmp_path):
    base, blob = based(tmp_path)
    record(tmp_path, blob, {NODE: entry()})
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40, NODE: 81})
    code, result = budget_run(tmp_path, case('45'), base=base)
    assert code == 0, result
    assert result['budget']['status'] == 'pass'


def test_without_a_record_the_same_addition_is_refused(tmp_path):
    base, _ = based(tmp_path)
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40, NODE: 81})
    code, result = budget_run(tmp_path, case('45'), base=base)
    assert code == 1 and 'grow' in result['budget']['reason']


def test_a_record_for_another_base_policy_grants_nothing(tmp_path):
    base, _ = based(tmp_path)
    record(tmp_path, 'b' * 40, {NODE: entry()})
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40, NODE: 81})
    code, result = budget_run(tmp_path, case('45'), base=base)
    assert code == 1 and 'grow' in result['budget']['reason']


@pytest.mark.parametrize('slow', [
    {'tests/test_sample.py::test_known': 40, NODE: 82},            # above its limit
    {'tests/test_sample.py::test_known': 41, NODE: 81},            # an unlisted node grows
    {'tests/test_sample.py::test_known': 40, NODE: 81, 'tests/test_sample.py::test_other': 30},
])
def test_growth_beyond_the_record_is_refused(tmp_path, slow):
    base, blob = based(tmp_path)
    record(tmp_path, blob, {NODE: entry()})
    policy(tmp_path, slow=slow)
    code, result = budget_run(tmp_path, case('0'), base=base)
    assert code == 1 and 'grow' in result['budget']['reason']


def test_a_record_cannot_raise_the_default_or_file_budgets(tmp_path):
    base, blob = based(tmp_path)
    record(tmp_path, blob, {NODE: entry()})
    policy(tmp_path, seconds=21, slow={'tests/test_sample.py::test_known': 40, NODE: 81})
    code, result = budget_run(tmp_path, case('0'), base=base)
    assert code == 1 and 'grow' in result['budget']['reason']
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40, NODE: 81},
           file_seconds={'tests/test_other.py': 50})
    code, result = budget_run(tmp_path, case('0'), base=base)
    assert code == 1 and 'grow' in result['budget']['reason']


@pytest.mark.parametrize('nodes', [
    {NODE: entry(seconds=90)},                                      # not ceil(2 * max)
    {NODE: entry(rows=evidence()[:1])},                             # one architecture only
    {NODE: entry(81, [{**evidence()[0], 'seconds': float('nan')}, evidence()[1]])},
    {NODE: entry(81, [{**evidence()[0], 'run': 'x'}, evidence()[1]])},
    {NODE: {'seconds': 81}},                                        # no evidence
])
def test_a_record_without_two_architecture_evidence_fails_closed(tmp_path, nodes):
    base, blob = based(tmp_path)
    record(tmp_path, blob, nodes)
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40, NODE: 81})
    code, result = budget_run(tmp_path, case('0'), base=base)
    assert code == 1 and result['budget']['status'] == 'fail'
    assert 'migration' in result['budget']['reason']


@pytest.mark.parametrize('reason', ['', None])
def test_a_record_must_state_its_reason(tmp_path, reason):
    base, blob = based(tmp_path)
    record(tmp_path, blob, {NODE: entry()}, reason=reason)
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40, NODE: 81})
    code, result = budget_run(tmp_path, case('0'), base=base)
    assert code == 1 and 'migration' in result['budget']['reason']


@pytest.mark.parametrize('seconds', [120, 123])
def test_budget_warning_uses_the_reviewed_migration_limit_without_rerunning(tmp_path, seconds):
    base, blob = based(tmp_path)
    record(tmp_path, blob, {NODE: entry()})
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40, NODE: 81})
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        report = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--junitxml=')))
        report.write_text(case(str(seconds)))
        return SimpleNamespace(returncode=0)

    code = runner.run_plan({'selected': [{'path': 'tests/test_sample.py'}],
                            'deselect': [], 'base_sha': base},
                           root=tmp_path, output=tmp_path / 'result.json',
                           junit=tmp_path / 'junit.xml', execute=execute, affinity={1})
    result = json.loads((tmp_path / 'result.json').read_text())
    assert code == 0 and result['verdict'] == 'pass'
    assert len(calls) == 1
    assert result['budget_warnings'] == [
        {'nodeid': NODE, 'seconds': float(seconds), 'budget_seconds': 81}]
    assert result['budget']['status'] == 'pass'
    assert result['budget']['baseline_sha'] == base
    assert 'budget_reruns' not in result
    assert not list(tmp_path.glob('junit.budget-rerun-*'))
