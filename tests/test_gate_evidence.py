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
    queue = tmp_path / 'queue'
    (queue / 'done').mkdir(parents=True)
    reports = []
    for repeat, outcome in enumerate(('failed', 'passed', 'failed')):
        key = str(repeat) * 64
        (queue / 'done' / (key + '.json')).write_text(json.dumps({
            'finished_host': 'worker', 'detail': {'resource_profile': {
                'scope': {'memory_peak_bytes': 321}}}}))
        outcomes = {'file_durations': {'tests/test_sample.py': repeat + 1},
                    'reports': [['tests/test_sample.py::test_cost', 'call', outcome]]}
        path = tmp_path / ('repeat-' + str(repeat) + '.json')
        path.write_text(json.dumps([{'action_key': key, 'returncode': 0,
            'output': 'pbtest-outcomes: ' + json.dumps(outcomes), 'verdict': {}}]))
        reports.append(path)
    report = evidence.inventory(reports, queue_root=queue)
    assert {action['memory_peak_bytes'] for action in report['actions']} == {321}
    assert report['always_failing'] == []
    assert report['changed_outcomes'] == {
        'tests/test_sample.py::test_cost': {1: 'fail', 2: 'pass', 3: 'fail'}}
    assert report['files'] == {'tests/test_sample.py': 3}
