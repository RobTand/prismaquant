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
    report = evidence.hosted([tmp_path], 20)
    node = 'tests/test_sample.py::test_cost'
    assert report['durations'] == {'tests/test_sample.py': 21}
    assert report['failures'] == {}
    assert report['budget']['slow'] == {node: 42}
    assert report['budget']['evidence'][node]['runner'] == 'ubuntu-24.04-arm'


def test_inventory_reads_aggregate_scope_peak_and_repeat_outcomes(tmp_path):
    action_evidence = {}
    reports = []
    for repeat, outcome in enumerate(('failed', 'passed', 'failed')):
        key = str(repeat) * 64
        action_evidence[key] = {
            'complete': True, 'timed_out': [], 'state': 'done',
            'outcome': {'finished_host': 'worker'},
            'resource_profile': {'scope': {'memory_peak_bytes': 321}}}
        outcomes = {'file_durations': {'tests/test_sample.py': repeat + 1},
                    'reports': [['tests/test_sample.py::test_cost', 'call', outcome]]}
        path = tmp_path / ('repeat-' + str(repeat) + '.json')
        path.write_text(json.dumps([{'action_key': key, 'returncode': 0,
            'output': 'pbtest-outcomes: ' + json.dumps(outcomes), 'verdict': {}}]))
        reports.append(path)
    report = evidence.inventory(reports, action_evidence=action_evidence)
    assert {action['memory_peak_bytes'] for action in report['actions']} == {321}
    assert report['always_failing'] == []
    assert report['changed_outcomes'] == {
        'tests/test_sample.py::test_cost': {1: 'fail', 2: 'pass', 3: 'fail'}}
    assert report['files'] == {'tests/test_sample.py': 3}


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
