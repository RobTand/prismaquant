"""The cutover records failures and refuses empty architecture proofs."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('pq_bazel_execution', ROOT / 'tools/ci/bazel_gate.py')
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
LABEL = '//tests:test_consumer'


def events(*rows):
    return '\n'.join(json.dumps(row) for row in rows)


def summary(status='PASSED'):
    return {'id': {'testSummary': {'label': LABEL}}, 'testSummary': {'overallStatus': status}}


def result(*, local=False, remote=False, shard=0):
    return {'id': {'testResult': {'label': LABEL, 'run': 1, 'shard': shard, 'attempt': 1}},
            'testResult': {'status': 'PASSED', 'cachedLocally': local,
                           'executionInfo': {'cachedRemotely': remote}}}


def test_budget_warnings_preserve_a_passing_execution():
    run = gate.bep_run(events(result(), summary()), returncode=0)
    warnings = [{'nodeid': 'tests/test_consumer.py::test_one', 'seconds': 21, 'budget_seconds': 20}]
    found = gate.architecture_result({LABEL}, run, budget_warnings=warnings, require_execution=True)
    assert found['verdict'] == 'pass'
    assert found['budget_warnings'] == warnings
    assert found['executed_test_actions'] == 1
    assert found['failed'] == []
    assert 'reruns' not in found


@pytest.mark.parametrize('status', ['FAILED', 'TIMEOUT', 'INCOMPLETE', 'NO_STATUS'])
def test_hard_failures_never_turn_into_warnings(status):
    run = gate.bep_run(events(result(), summary(status)), returncode=3)
    found = gate.architecture_result({LABEL}, run)
    assert found['verdict'] == 'fail'
    assert found['failed'] == [LABEL]


def test_missing_selected_status_fails_even_if_bazel_exits_zero():
    run = gate.bep_run(events(result()), returncode=0)
    assert gate.architecture_result({LABEL}, run)['failed'] == [LABEL]


def test_build_failure_cannot_hide_behind_passing_tests():
    run = gate.bep_run(events(result(), summary()), returncode=1)
    assert gate.architecture_result({LABEL}, run)['verdict'] == 'fail'


@pytest.mark.parametrize('rows', [(), (summary(),), (result(local=True), summary()),
                                  (result(remote=True), summary())])
def test_postmerge_requires_executed_tests_on_each_architecture(rows):
    run = gate.bep_run(events(*rows), returncode=0)
    found = gate.architecture_result({LABEL}, run, require_execution=True)
    assert found['verdict'] == 'fail'
    assert found['executed_test_actions'] == 0


def test_normal_selection_can_reuse_successful_test_cache():
    run = gate.bep_run(events(result(local=True), summary()), returncode=0)
    assert gate.architecture_result({LABEL}, run)['verdict'] == 'pass'


def test_empty_docs_selection_is_explicitly_skipped():
    assert gate.architecture_result(set(), gate.bep_run('', returncode=0))['verdict'] == 'skipped'


def test_shards_count_once_and_terminal_failure_cannot_be_overwritten():
    run = gate.bep_run(events(result(shard=0), result(shard=0), result(shard=1),
                              summary('TIMEOUT'), summary()), returncode=0)
    assert run['executed_test_actions'] == 2
    assert gate.architecture_result({LABEL}, run)['failed'] == [LABEL]


def test_malformed_bep_is_an_evidence_failure():
    with pytest.raises(gate.GateError, match='build events'):
        gate.bep_run('{', returncode=0)


def shard_record(index, labels, *, status='COMPLETE_INVOCATION_STATUS', cached=False):
    run = {'statuses': {label: 'PASSED' for label in labels}, 'invocation': f'child-{index}',
           'returncode': 0, 'executed_by_label': {label: 0 if cached else 1 for label in labels}}
    return {'index': index, 'invocation_status': status, 'run': run, 'budget_warnings': []}


def test_full_partition_is_deterministic_disjoint_and_covers_the_query():
    labels = [f'//tests:test_{number:04}' for number in range(1376)]
    shards = gate.partition_population(labels, shard_count=8)
    assert shards == gate.partition_population(reversed(labels), shard_count=8)
    assert len(shards) == 8
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    assert gate.check_partition(labels, shards) is None
    assert sorted(label for shard in shards for label in shard) == labels


@pytest.mark.parametrize('shards', [[], [[LABEL], [LABEL]], [['//tests:test_foreign']]])
def test_partition_refuses_missing_duplicate_or_foreign_targets(shards):
    with pytest.raises(gate.GateError, match='partition'):
        gate.check_partition([LABEL], shards)


@pytest.mark.parametrize('count', [0, -1])
def test_full_partition_requires_a_nonempty_query_and_valid_shard_count(count):
    with pytest.raises(gate.GateError, match='partition'):
        gate.partition_population([LABEL], shard_count=count)
    with pytest.raises(gate.GateError, match='population'):
        gate.partition_population([], shard_count=8)


def test_complete_shards_aggregate_budget_warnings_without_rerunning():
    labels = [LABEL, '//tests:test_second']
    shards = gate.partition_population(labels, shard_count=2)
    records = [shard_record(index, shard) for index, shard in enumerate(shards)]
    warning = {'nodeid': 'tests/test_second.py::test_one', 'seconds': 21, 'budget_seconds': 20}
    records[1]['budget_warnings'] = [warning]
    found = gate.aggregate_shards(labels, shards, records, require_execution=True)
    assert found['verdict'] == 'pass'
    assert found['executed_test_actions'] == 2
    assert found['budget_warnings'] == [warning]
    assert len(found['shards']) == 2


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'disconnected', 'wrong_population'])
def test_sharded_full_fails_closed_on_missing_or_incomplete_evidence(fault):
    labels = [LABEL, '//tests:test_second']
    shards = gate.partition_population(labels, shard_count=2)
    records = [shard_record(index, shard) for index, shard in enumerate(shards)]
    if fault == 'missing':
        records.pop()
    elif fault == 'duplicate':
        records[1]['index'] = 0
    elif fault == 'disconnected':
        records[1]['invocation_status'] = 'DISCONNECTED_INVOCATION_STATUS'
    else:
        records[1]['run']['statuses'] = {'//tests:test_foreign': 'PASSED'}
    with pytest.raises(gate.GateError):
        gate.aggregate_shards(labels, shards, records)


@pytest.mark.parametrize('fault', ['timeout', 'cache_only'])
def test_full_shards_preserve_hard_failures_and_execution_requirement(fault):
    shards = [[LABEL]]
    records = [shard_record(0, shards[0], cached=fault == 'cache_only')]
    if fault == 'timeout':
        records[0]['run']['statuses'][LABEL] = 'TIMEOUT'
    assert gate.aggregate_shards([LABEL], shards, records, require_execution=True)['verdict'] == 'fail'
