"""Qualification summaries must use effective verdicts and measured scope memory."""
import json
from pathlib import Path

from tools.ci import gate_evidence as evidence


def test_full_gate_uses_effective_retry_artifact_once(tmp_path):
    (tmp_path / 'run.json').write_text(json.dumps({
        'id': 123, 'run_attempt': 2, 'head_sha': 'a' * 40, 'conclusion': 'success'}))
    (tmp_path / 'jobs.json').write_text('{"jobs": []}')
    plan = tmp_path / 'gate-plan-2'
    plan.mkdir()
    (plan / 'job-matrix.json').write_text(json.dumps({
        'include': [{'index': 0, 'runner': 'ubuntu-24.04-arm'}]}))
    for dirname, seconds, outcome in [
            ('gate-result-2-a1-0', 100, '<failure/>'),
            ('gate-final-2/effective/0', 21, '')]:
        target = tmp_path / dirname
        target.mkdir(parents=True)
        (target / 'junit.xml').write_text(
            '<testsuites><testsuite><testcase file="tests/test_sample.py" '
            'classname="tests.test_sample" name="test_cost" time="' + str(seconds) + '">'
            + outcome + '</testcase></testsuite></testsuites>')
    previous = tmp_path / 'gate-final-1/effective/0'
    previous.mkdir(parents=True)
    (previous / 'junit.xml').write_text(
        '<testsuites><testsuite><testcase file="tests/test_sample.py" '
        'classname="tests.test_sample" name="test_cost" time="9999"/>'
        '</testsuite></testsuites>')
    report = evidence.hosted([tmp_path], 20)
    node = 'tests/test_sample.py::test_cost'
    assert report['durations'] == {'tests/test_sample.py': 21}
    assert report['node_durations'] == {'tests/test_sample.py::test_cost': 21}
    assert report['failures'] == {}
    assert report['budget']['slow'] == {node: 42}
    assert report['budget']['evidence'][node]['runner'] == 'ubuntu-24.04-arm'


def test_hosted_wall_percentiles_use_completed_successful_full_gates(tmp_path):
    from datetime import datetime, timedelta, timezone
    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    directories = []
    for index, (seconds, conclusion) in enumerate([(100, 'success'), (200, 'success'),
                                                  (300, 'success'), (900, 'failure')]):
        directory = tmp_path / str(index)
        directory.mkdir()
        stamp = lambda offset: (start + timedelta(seconds=offset)).isoformat()
        (directory / 'run.json').write_text(json.dumps({
            'id': index, 'run_attempt': 1, 'head_sha': 'a' * 40,
            'conclusion': conclusion, 'run_started_at': stamp(0)}))
        (directory / 'jobs.json').write_text(json.dumps({'jobs': [
            {'name': 'shard-0-a1', 'started_at': stamp(10), 'completed_at': stamp(30),
             'conclusion': 'success', 'labels': ['ubuntu-24.04-arm'], 'steps': [
                 {'name': 'Install CPU test environment', 'started_at': stamp(12), 'completed_at': stamp(17)},
                 {'name': 'Run assigned gate tests', 'started_at': stamp(20), 'completed_at': stamp(29)}]},
            {'name': 'gate', 'started_at': stamp(seconds - 5), 'completed_at': stamp(seconds),
             'conclusion': conclusion}]}))
        directories.append(directory)
    report = evidence.hosted(directories, 20)
    assert report['full_gate_wall_seconds'] == {'samples': 3, 'p50': 200, 'p90': 300}
    assert [run['wall_seconds'] for run in report['runs']] == [100, 200, 300, 900]
    shard = report['runs'][0]['jobs'][0]
    assert shard['setup_seconds'] == 10
    assert shard['test_seconds'] == 9
    assert shard['install_seconds'] == 5


def test_hosted_imported_test_costs_follow_collection_identity(tmp_path):
    (tmp_path / 'run.json').write_text(json.dumps({
        'id': 123, 'run_attempt': 1, 'head_sha': 'a' * 40, 'conclusion': 'success'}))
    (tmp_path / 'jobs.json').write_text('{"jobs": []}')
    target = tmp_path / 'gate-result-1-a1-0'
    target.mkdir()
    (target / 'junit.xml').write_text(
        '<testsuites><testcase file="tests/helpers.py" classname="tests.test_sample" '
        'name="test_cost" time="21"><properties><property name="gate_nodeid" '
        'value="tests/test_sample.py::test_cost"/></properties></testcase></testsuites>')
    report = evidence.hosted([tmp_path], 20)
    assert report['durations'] == {'tests/test_sample.py': 21}
    assert report['budget']['slow'] == {'tests/test_sample.py::test_cost': 42}


def test_canary_wall_starts_at_creation_or_explicit_rerun_request(tmp_path):
    for attempt, requested, expected in [(1, None, 120), (2, '2026-11-01T00:00:15Z', 105)]:
        (tmp_path / 'run.json').write_text(json.dumps({
            'id': 123, 'run_attempt': attempt, 'head_sha': 'a' * 40, 'conclusion': 'success',
            'created_at': '2026-11-01T00:00:00Z', 'run_started_at': '2026-11-01T00:00:30Z',
            **({'requested_at': requested} if requested else {})}))
        (tmp_path / 'jobs.json').write_text(json.dumps({'jobs': [
            {'name': 'gate', 'conclusion': 'success', 'completed_at': '2026-11-01T00:02:00Z'}]}))
        report = evidence.hosted([tmp_path], 20)
        assert report['runs'][0]['wall_seconds'] == expected


def test_file_budget_calibration_uses_full_cost_when_fixture_charge_moves(tmp_path):
    directories = []
    for run, times, runner in [(1, (55, 1), 'ubuntu-24.04-arm'),
                               (2, (1, 59), 'ubuntu-24.04')]:
        directory = tmp_path / str(run)
        directory.mkdir()
        (directory / 'run.json').write_text(json.dumps({
            'id': run, 'run_attempt': 1, 'head_sha': 'a' * 40, 'conclusion': 'success'}))
        (directory / 'jobs.json').write_text('{"jobs": []}')
        plan = directory / 'gate-plan-1'
        plan.mkdir()
        (plan / 'job-matrix.json').write_text(json.dumps({'include': [{'index': 0, 'runner': runner}]}))
        result = directory / 'gate-result-1-a1-0'
        result.mkdir()
        (result / 'junit.xml').write_text('<testsuites>' + ''.join(
            f'<testcase file="tests/test_sample.py" classname="tests.test_sample" '
            f'name="test_{index}" time="{seconds}"/>' for index, seconds in enumerate(times)) + '</testsuites>')
        directories.append(directory)
    report = evidence.hosted(directories, 20, file_budgets=['tests/test_sample.py'])
    assert report['budget']['slow'] == {}
    assert report['budget']['file_seconds'] == {'tests/test_sample.py': 120}
    assert report['budget']['file_evidence']['tests/test_sample.py'] == {
        'seconds': 60, 'run': 2, 'attempt': 1, 'runner': 'ubuntu-24.04',
        'artifact': 'gate-result-1-a1-0/junit.xml'}


def test_split_file_duration_is_summed_per_run_before_architecture_medians(tmp_path):
    (tmp_path / 'run.json').write_text(json.dumps({'id': 123, 'run_attempt': 1,
        'head_sha': 'a' * 40, 'conclusion': 'success'}))
    (tmp_path / 'jobs.json').write_text('{"jobs": []}')
    plan = tmp_path / 'gate-plan-1'; plan.mkdir()
    (plan / 'job-matrix.json').write_text(json.dumps({'include': [
        {'index': 0, 'runner': 'ubuntu-24.04-arm'}, {'index': 1, 'runner': 'ubuntu-24.04-arm'}]}))
    for index, seconds in enumerate([30, 70]):
        target = tmp_path / f'gate-final-1/effective/{index}'; target.mkdir(parents=True)
        (target / 'junit.xml').write_text('<testsuites>' +
            f'<testcase file="tests/test_sample.py" classname="tests.test_sample" '
            f'name="test_{index}" time="{seconds}"/></testsuites>')
    report = evidence.hosted([tmp_path], 20)
    assert report['durations'] == {'tests/test_sample.py': 100}
    assert report['node_durations'] == {'tests/test_sample.py::test_0': 30, 'tests/test_sample.py::test_1': 70}
