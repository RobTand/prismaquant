"""Bazel must run exactly the hosted gate's population, and say how to fix drift."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('bazel_parity', ROOT / 'tools/ci/bazel_parity.py')
bazel_parity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bazel_parity)

PLAN = {'selected': [{'path': 'tests/test_a.py'}, {'path': 'tests/test_b.py'}],
        'deselect': ['tests/test_a.py::test_flaky'],
        'deferred': [{'path': 'tests/test_root.py', 'lane': 'root'},
                     {'path': 'tests/test_a.py::test_flaky', 'issue': 2000}]}


def rule(path, *, args=(), tags=(), name=None):
    return json.dumps({'type': 'RULE', 'rule': {
        'name': name or '//tests:' + Path(path).stem, 'ruleClass': 'py_test',
        'attribute': [{'name': 'main', 'stringValue': bazel_parity.RUNNER},
                      {'name': 'args', 'stringListValue': [path, *args]},
                      {'name': 'tags', 'stringListValue': list(tags)}]}})


MATCHING = [rule('tests/test_a.py', args=['--deselect=tests/test_a.py::test_flaky']),
            rule('tests/test_b.py'), rule('tests/test_root.py', tags=['manual', 'gate-lane-root'])]


def test_matching_bazel_targets_pass_static_parity():
    summary = bazel_parity.compare_static(PLAN, bazel_parity.bazel_tests(MATCHING))
    assert summary == {'population': 2, 'manual': 1, 'deselected_nodes': 1}


@pytest.mark.parametrize('lines, message', [
    (MATCHING[:1] + MATCHING[2:], 'Bazel does not run tests/test_b.py'),
    (MATCHING + [rule('tests/test_new.py')], 'Bazel runs tests/test_new.py, which the hosted gate does not'),
    ([rule('tests/test_a.py', args=['--deselect=tests/test_a.py::test_flaky'], tags=['manual'])] + MATCHING[1:],
     'Bazel does not run tests/test_a.py'),
    ([rule('tests/test_a.py')] + MATCHING[1:], 'tests/test_a.py deselects'),
    (MATCHING[:2], 'tests/test_root.py has no Bazel target'),
    (MATCHING + [rule('tests/test_b.py', name='//tests:twice')], 'two Bazel targets run tests/test_b.py'),
])
def test_drift_fails_naming_the_file_and_the_fix(lines, message):
    with pytest.raises(RuntimeError) as failed:
        bazel_parity.compare_static(PLAN, bazel_parity.bazel_tests(lines))
    assert message in str(failed.value)
    assert 'Fix: tools/bazel/generate.sh' in str(failed.value)


def test_hosted_lane_targets_count_as_run_and_plain_manual_ones_do_not():
    hosted = [MATCHING[0], rule('tests/test_b.py', tags=['manual', 'gate-lane-hosted']), MATCHING[2]]
    assert bazel_parity.compare_static(PLAN, bazel_parity.bazel_tests(hosted))['population'] == 2
    with pytest.raises(RuntimeError, match='Bazel does not run tests/test_b.py'):
        bazel_parity.compare_static(PLAN, bazel_parity.bazel_tests(
            [MATCHING[0], rule('tests/test_b.py', tags=['manual']), MATCHING[2]]))


def test_an_empty_population_is_refused():
    with pytest.raises(RuntimeError, match='empty'):
        bazel_parity.compare_static({'selected': [], 'deselect': [], 'deferred': []}, {})


def test_the_population_comes_from_the_gate_selector_without_reading_imports():
    plan = bazel_parity.population(ROOT)
    paths = {row['path'] for row in plan['selected']}
    assert 'tests/test_bazel_parity.py' in paths
    assert 'tests/g3job/test_g3.py' not in paths  # quarantined whole file (#3181)


def junit(path, cases):
    body = ''.join(f'<testcase file="{file}" classname="{file[:-3].replace("/", ".")}" name="{name}" time="0.1">'
                   f'{"<skipped/>" if outcome == "skipped" else ""}</testcase>'
                   for file, name, outcome in cases)
    path.write_text(f'<testsuites><testsuite>{body}</testsuite></testsuites>')
    return path


def test_receipts_match_hosted_nodes_and_outcomes(tmp_path):
    hosted = junit(tmp_path / 'junit.xml', [('tests/test_a.py', 'test_one', 'passed'),
                                            ('tests/test_a.py', 'test_two', 'skipped')])
    outputs = tmp_path / 'testlogs/tests/test_a/test.outputs'
    outputs.mkdir(parents=True)
    (outputs / 'inventory.json').write_text(json.dumps({'file': 'tests/test_a.py', 'nodes': {
        'tests/test_a.py::test_one': 'passed', 'tests/test_a.py::test_two': 'skipped'}}))
    expected, actual = bazel_parity.hosted_nodes([hosted]), bazel_parity.bazel_nodes(tmp_path / 'testlogs')
    assert bazel_parity.compare_receipts(expected, actual) == {'nodes': 2, 'files': 1}
    actual['tests/test_a.py::test_two'] = 'passed'
    with pytest.raises(RuntimeError, match='test_two: hosted skipped, Bazel passed'):
        bazel_parity.compare_receipts(expected, actual)
    del actual['tests/test_a.py::test_two']
    with pytest.raises(RuntimeError, match='missing from Bazel: tests/test_a.py::test_two'):
        bazel_parity.compare_receipts(expected, actual)


def test_quoted_args_are_read_as_bazel_tokenizes_them():
    quoted = rule('tests/test_a.py', args=["'--deselect=tests/test_a.py::test_flaky'"])
    assert bazel_parity.bazel_tests([quoted])['tests/test_a.py'][0][2] == ['tests/test_a.py::test_flaky']


def test_missing_imported_execution_fails_node_parity(tmp_path):
    definition = 'tests/test_band_serial_batched_regime.py'
    collecting = 'tests/test_band_serial_scratch_plane.py'
    name = 'test_the_control_at_batch_one_still_holds'
    report = tmp_path / 'imported.xml'
    cases = ''.join(
        f'<testcase file="{definition}" classname="{path[:-3].replace("/", ".")}" '
        f'name="{name}" time="0.1"><properties>'
        f'<property name="gate_nodeid" value="{path}::{name}"/>'
        '</properties></testcase>' for path in (definition, collecting))
    report.write_text('<testsuites><testsuite>' + cases + '</testsuite></testsuites>')
    outputs = tmp_path / 'testlogs/tests/definition/test.outputs'
    outputs.mkdir(parents=True)
    (outputs / 'inventory.json').write_text(json.dumps({
        'file': definition, 'nodes': {definition + '::' + name: 'passed'}}))
    with pytest.raises(RuntimeError, match='missing from Bazel: ' + collecting):
        bazel_parity.compare_receipts(bazel_parity.hosted_nodes([report]),
                                     bazel_parity.bazel_nodes(tmp_path / 'testlogs'))
    # The collecting module's execution is distinct even though JUnit names
    # the same definition file. Restoring exactly that execution repairs parity.
    imported = tmp_path / 'testlogs/tests/imported/test.outputs'
    imported.mkdir(parents=True)
    (imported / 'inventory.json').write_text(json.dumps({
        'file': collecting, 'nodes': {collecting + '::' + name: 'passed'}}))
    assert bazel_parity.compare_receipts(
        bazel_parity.hosted_nodes([report]),
        bazel_parity.bazel_nodes(tmp_path / 'testlogs')) == {'nodes': 2, 'files': 2}
