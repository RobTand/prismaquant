"""The cutover records failures and refuses empty architecture proofs."""
import importlib.util
import io
import json
from pathlib import Path
import zipfile

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


def test_full_partition_distributes_tests_and_includes_non_test_build_targets():
    tests = {f'//tests:test_{number:04}' for number in range(16)}
    libraries = {label + '_lib' for label in tests}
    population = tests | libraries
    shards = gate.partition_population(population, tests=tests, shard_count=4)
    assert sorted(label for shard in shards for label in shard) == sorted(population)
    assert [len(set(shard) & tests) for shard in shards] == [4] * 4
    assert shards == gate.partition_population(reversed(sorted(population)),
                                               tests=tests, shard_count=4)


def test_test_population_must_belong_to_full_build_population():
    with pytest.raises(gate.GateError, match='population'):
        gate.partition_population([LABEL], tests={LABEL, '//tests:foreign'})


def test_full_shards_verify_only_test_statuses_but_build_failures_still_fail():
    population = [LABEL, '//prismaquant:library']
    shards = [[LABEL], ['//prismaquant:library']]
    records = [shard_record(0, [LABEL]), shard_record(1, [])]
    passed = gate.aggregate_shards(population, shards, records, test_population={LABEL},
                                   require_execution=True)
    assert passed['verdict'] == 'pass'
    assert passed['executed_test_actions'] == 1
    records[1]['run']['returncode'] = 1
    assert gate.aggregate_shards(population, shards, records, test_population={LABEL},
                                 require_execution=True)['verdict'] == 'fail'


def test_full_build_population_cannot_hide_a_missing_test_status():
    population = [LABEL, '//prismaquant:library']
    records = [shard_record(0, [])]
    with pytest.raises(gate.GateError, match='population'):
        gate.aggregate_shards(population, [population], records, test_population={LABEL})


@pytest.mark.parametrize('status,code,verdict', [('PASSED', 0, 'pass'), ('TIMEOUT', 3, 'fail')])
def test_remote_shard_executes_once_with_private_bep_and_preserves_timeout(
        tmp_path, monkeypatch, status, code, verdict):
    monkeypatch.setenv('MERGIFY_TOKEN', 'must-not-reach-bazel')
    monkeypatch.setenv('UNRELATED_SECRET', 'must-not-reach-bazel')
    manifest = {'mode': 'main', 'base': 'b' * 40, 'head': 'a' * 40,
                'branch': 'main', 'architecture': 'arm64', 'index': 0,
                'targets': [LABEL, '//prismaquant:library'], 'tests': [LABEL],
                'require_execution': True}
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert '--config=remote-arm64' in command and '--jobs=16' in command
        assert '--nocache_test_results' in command and '--noremote_accept_cached' in command
        assert '--flaky_test_attempts=1' in command
        assert 'MERGIFY_TOKEN' not in kwargs['env'] and 'UNRELATED_SECRET' not in kwargs['env']
        path = Path(next(arg.split('=', 1)[1] for arg in command
                         if arg.startswith('--build_event_json_file=')))
        assert not path.is_relative_to(tmp_path / 'upload')
        path.write_text(events(result(), summary(status)))
        from types import SimpleNamespace
        return SimpleNamespace(returncode=code)
    found = gate.execute_shard(manifest, root=tmp_path, runner=run)
    assert len(calls) == 1
    assert found['verdict'] == verdict
    assert found['targets'] == manifest['targets']
    assert found['architecture'] == 'arm64'
    assert found['run']['statuses'] == {LABEL: status}
    assert found['budget_warnings'] == []


def inventory_zip(payload):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('inventory.json', json.dumps(payload))
    return buffer.getvalue()


def output_event(uri='bytestream://remote.buildbuddy.io/own-output'):
    return {'id': {'testResult': {'label': LABEL}}, 'testResult': {
        'testActionOutput': [{'name': 'test.outputs__outputs.zip', 'uri': uri}]}}


def test_inventory_warnings_and_nodes_are_collected_once_from_own_bep_outputs():
    warning = {'nodeid': 'tests/test_consumer.py::test_one', 'seconds': 21, 'budget_seconds': 20}
    calls = []
    def fetch(uri):
        calls.append(uri)
        return inventory_zip({'file': 'tests/test_consumer.py',
                              'nodes': {warning['nodeid']: 'passed'}, 'budget_warnings': [warning]})
    found = gate.inventory_evidence(events(output_event(), output_event()), fetch)
    assert found['budget_warnings'] == [warning]
    assert found['nodes'] == {warning['nodeid']: 'passed'}
    assert calls == ['bytestream://remote.buildbuddy.io/own-output']


@pytest.mark.parametrize('raw', [b'not a zip', inventory_zip({'budget_warnings': 'invalid'})])
def test_malformed_inventory_is_an_evidence_failure(raw):
    with pytest.raises(gate.GateError, match='inventory'):
        gate.inventory_evidence(events(output_event()), lambda uri: raw)


def test_native_tests_without_python_inventory_have_no_budget_warnings():
    assert gate.inventory_evidence(events(result(), summary()), lambda uri: b'') == {
        'nodes': {}, 'budget_warnings': []}
