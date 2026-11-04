"""The cutover records failures and refuses empty architecture proofs."""
import importlib.util
import hashlib
import io
import json
import errno
import socket
import ssl
from pathlib import Path
import urllib.request
import urllib.error
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('pq_bazel_execution', ROOT / 'tools/ci/bazel_gate.py')
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
LABEL = '//tests:test_consumer'


@pytest.mark.parametrize('architecture,mode,wanted', [
    ('amd64', 'pr', ['tests/test_owned.py']),
    ('amd64', 'main', ['tests/test_owned.py']),
    ('arm64', 'main', ['tests/test_disk.py']),
])
def test_hosted_assignment_preserves_the_existing_plan_and_exact_lane(architecture, mode, wanted):
    plan = {'schema': 'prismaquant.gate_plan.v1', 'mode': 'full',
        'base_sha': 'b' * 40, 'head_sha': 'a' * 40,
        'population': {'version': 'full-population'},
        'selected': [{'path': 'tests/test_owned.py', 'estimated_seconds': 2},
                     {'path': 'tests/test_disk.py', 'estimated_seconds': 3},
                     {'path': 'tests/test_other.py', 'estimated_seconds': 4}],
        'deselect': ['tests/test_owned.py::test_known', 'tests/test_other.py::test_known'],
        'deferred': []}
    result = {'schema': gate.SCHEMA, 'mode': mode, 'base': 'b' * 40, 'head': 'a' * 40,
        'verdict': 'pass', 'hosted': ['//tests:test_owned'],
        'hosted_arm64': ['//tests:test_disk'] if mode == 'main' else []}
    found = gate.hosted_assignment(plan, result, architecture)
    assert [row['path'] for row in found['selected']] == wanted
    assert found['population'] == {'version': 'full-population'}
    assert found['deselect'] == (['tests/test_owned.py::test_known'] if architecture == 'amd64' else [])
    assert len(plan['selected']) == 3 and len(plan['deselect']) == 2
    assert found['base_sha'] == 'b' * 40 and found['head_sha'] == 'a' * 40


@pytest.mark.parametrize('fault', ['base', 'head', 'duplicate', 'unknown', 'escape', 'failed', 'architecture'])
def test_hosted_assignment_refuses_missing_or_inconsistent_lane_evidence(fault):
    plan = {'schema': 'prismaquant.gate_plan.v1', 'base_sha': 'b' * 40,
        'head_sha': 'a' * 40, 'selected': [{'path': 'tests/test_owned.py'}], 'deselect': []}
    result = {'schema': gate.SCHEMA, 'mode': 'pr', 'base': 'b' * 40, 'head': 'a' * 40,
        'verdict': 'pass', 'hosted': ['//tests:test_owned'], 'hosted_arm64': []}
    if fault in ('base', 'head'):
        result[fault] = 'c' * 40
    elif fault == 'duplicate':
        result['hosted'] *= 2
    elif fault == 'unknown':
        result['hosted'] = ['//tests:test_missing']
    elif fault == 'escape':
        result['hosted'] = ['//tests:../test_owned']
    elif fault == 'failed':
        result['verdict'] = 'fail'
    with pytest.raises(gate.GateError):
        gate.hosted_assignment(plan, result, 'other' if fault == 'architecture' else 'amd64')


@pytest.mark.parametrize('architecture,fault', [
    ('amd64', None), ('arm64', None), ('amd64', 'head'),
    ('amd64', 'malformed'), ('amd64', 'missing'), ('amd64', 'occupied'),
])
def test_hosted_cli_materializes_only_the_source_verified_assignment(tmp_path, architecture, fault):
    plan = {'schema': 'prismaquant.gate_plan.v1', 'mode': 'full',
        'base_sha': 'b' * 40, 'head_sha': 'a' * 40,
        'population': {'version': 'existing-policy'},
        'selected': [{'path': 'tests/test_owned.py', 'estimated_seconds': 2},
                     {'path': 'tests/test_disk.py', 'estimated_seconds': 3},
                     {'path': 'tests/test_other.py', 'estimated_seconds': 4}],
        'deselect': ['tests/test_owned.py::test_known', 'tests/test_other.py::test_known'],
        'deferred': ['existing-policy-exclusion']}
    result = {'schema': gate.SCHEMA, 'mode': 'main', 'base': 'b' * 40,
        'head': 'c' * 40 if fault == 'head' else 'a' * 40, 'verdict': 'pass',
        'hosted': ['//tests:test_owned'], 'hosted_arm64': ['//tests:test_disk']}
    plan_path, result_path, output = (tmp_path / name for name in ('plan.json', 'remote.json', 'assigned.json'))
    plan_path.write_text(json.dumps(plan))
    result_path.write_text('{' if fault == 'malformed' else json.dumps(result))
    if fault == 'missing':
        result_path.unlink()
    if fault == 'occupied':
        output.write_text('preexisting private evidence')
    code = gate.main(['hosted', '--architecture', architecture, '--plan', str(plan_path),
                      '--result', str(result_path), '--output', str(output)])
    if fault:
        assert code == 75
        if fault == 'occupied':
            assert output.read_text() == 'preexisting private evidence'
        else:
            assert not output.exists()
    else:
        assert code == 0
        assigned = json.loads(output.read_text())
        assert assigned == {**plan,
            'selected': [plan['selected'][0 if architecture == 'amd64' else 1]],
            'deselect': ['tests/test_owned.py::test_known'] if architecture == 'amd64' else []}
    assert json.loads(plan_path.read_text()) == plan


def test_shard_output_reader_does_not_load_the_submission_controller(monkeypatch):
    real_spec = importlib.util.spec_from_file_location
    requested = []

    def load_reader(name, path, *args, **kwargs):
        assert Path(path).name != 'remote_gate.py', 'a shard reader cannot depend on its controller'
        return real_spec(name, path, *args, **kwargs)

    def read_output(request, *, timeout):
        requested.append(request)
        return io.BytesIO(b'own-inventory-output')

    monkeypatch.setenv('BUILDBUDDY_API_KEY', 'reader-test-sentinel')
    monkeypatch.setattr(importlib.util, 'spec_from_file_location', load_reader)
    monkeypatch.setattr(urllib.request, 'urlopen', read_output)
    assert gate.fetch_artifact('bytestream://remote.buildbuddy.io/own-output') == b'own-inventory-output'
    assert len(requested) == 1
    assert json.loads(requested[0].data) == {'uri': 'bytestream://remote.buildbuddy.io/own-output'}


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


def output_event(uri='bytestream://remote.buildbuddy.io/own-output', name='test.outputs__outputs.zip'):
    return {'id': {'testResult': {'label': LABEL}}, 'testResult': {
        'testActionOutput': [{'name': name, 'uri': uri}]}}


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


@pytest.mark.parametrize('name', ['test.outputs/outputs.zip', 'test.outputs/inventory.json'])
def test_inventory_accepts_bazel_zip_and_direct_output_names(name):
    receipt = {'nodes': {'tests/test_one.py::test_one': 'passed'}, 'budget_warnings': []}
    raw = json.dumps(receipt).encode() if name.endswith('inventory.json') else inventory_zip(receipt)
    assert gate.inventory_evidence(events(output_event(name=name)), lambda uri: raw) == receipt


def test_remote_shard_includes_warnings_and_node_receipts_without_retry(tmp_path):
    warning = {'nodeid': 'tests/test_consumer.py::test_one', 'seconds': 21, 'budget_seconds': 20}
    payload = inventory_zip({'nodes': {warning['nodeid']: 'passed'}, 'budget_warnings': [warning]})
    manifest = {'mode': 'pr', 'base': 'b' * 40, 'head': 'a' * 40, 'branch': 'bbrun/cutover',
                'architecture': 'amd64', 'index': 0, 'targets': [LABEL], 'tests': [LABEL],
                'require_execution': False}
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        path = Path(next(arg.split('=', 1)[1] for arg in command
                         if arg.startswith('--build_event_json_file=')))
        path.write_text(events(result(), summary(), output_event()))
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0)
    found = gate.execute_shard(manifest, root=tmp_path, runner=run, fetch=lambda uri: payload)
    assert found['verdict'] == 'pass'
    assert found['budget_warnings'] == [warning]
    assert found['nodes'] == {warning['nodeid']: 'passed'}
    assert len(calls) == 1


@pytest.mark.parametrize('fault', [None, 'digest', 'head', 'union', 'partition'])
def test_immutable_manifest_loading_checks_digest_source_union_and_partition(fault):
    document = {'schema': 'prismaquant.bazel-shards.v1', 'mode': 'main',
                'base': 'b' * 40, 'head': 'a' * 40,
                'architectures': {'amd64': {'population': [LABEL, '//prismaquant:library'],
                    'tests': [LABEL], 'shards': [[LABEL], ['//prismaquant:library']]}}}
    if fault == 'head':
        document['head'] = 'd' * 40
    if fault == 'union':
        document['architectures']['amd64']['shards'][1] = [LABEL]
    raw = json.dumps(document).encode()
    reference = {'mode': 'main', 'base': 'b' * 40, 'head': 'a' * 40, 'branch': 'main',
                 'architecture': 'amd64', 'index': 0, 'require_execution': True,
                 'manifest_uri': 'bytestream://remote.buildbuddy.io/own-manifest',
                 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                 'partition_sha256': hashlib.sha256((LABEL + '\n').encode()).hexdigest()}
    if fault == 'digest':
        reference['manifest_sha256'] = 'e' * 64
    if fault == 'partition':
        reference['partition_sha256'] = 'f' * 64
    if fault:
        with pytest.raises(gate.GateError, match='manifest|partition'):
            gate.load_shard_manifest(reference, lambda uri: raw)
    else:
        found = gate.load_shard_manifest(reference, lambda uri: raw)
        assert found['targets'] == [LABEL]
        assert found['tests'] == [LABEL]
        assert found['require_execution'] is True
        assert found['manifest_sha256'] == reference['manifest_sha256']


@pytest.mark.parametrize('scan_fault', [None, 'missing_key', 'leak', 'unscannable'])
def test_shard_cli_uploads_only_its_scanned_result(tmp_path, monkeypatch, scan_fault):
    monkeypatch.setattr(gate.artifacts, 'require_scan_python', lambda: None)
    monkeypatch.setattr(gate.artifacts, 'scan_runtime_available', lambda: True)
    document = {'schema': 'prismaquant.bazel-shards.v1', 'mode': 'main',
        'base': 'b' * 40, 'head': 'a' * 40, 'architectures': {'amd64': {
            'population': [LABEL], 'tests': [LABEL], 'shards': [[LABEL]]}}}
    raw = json.dumps(document).encode()
    reference = {'mode': 'main', 'base': 'b' * 40, 'head': 'a' * 40, 'branch': 'main',
        'architecture': 'amd64', 'index': 0, 'require_execution': True,
        'manifest_uri': 'bytestream://remote.buildbuddy.io/own-manifest',
        'manifest_sha256': hashlib.sha256(raw).hexdigest(),
        'partition_sha256': hashlib.sha256((LABEL + '\n').encode()).hexdigest()}
    monkeypatch.setenv('PQ_BAZEL_SHARD', json.dumps(reference))
    monkeypatch.setenv('BUILDBUDDY_ARTIFACTS_DIRECTORY', str(tmp_path / 'upload'))
    monkeypatch.setenv('BUILDBUDDY_API_KEY', 'pq-scan-controller-sentinel')
    if scan_fault == 'missing_key':
        monkeypatch.delenv('BUILDBUDDY_API_KEY')
    monkeypatch.setattr(gate, 'fetch_artifact', lambda uri: raw, raising=False)
    calls = []
    def execute(manifest):
        calls.append(manifest)
        if scan_fault == 'unscannable':
            (tmp_path / 'upload' / 'execution_log.binpb.zst').write_bytes(b'\x28\xb5\x2f\xfd')
        return {'schema': gate.SCHEMA, 'mode': 'main', 'head': 'a' * 40,
                'verdict': 'pass', 'targets': [LABEL], 'budget_warnings': [],
                'error': 'pq-scan-controller-sentinel' if scan_fault == 'leak' else None}
    monkeypatch.setattr(gate, 'execute_shard', execute)
    monkeypatch.setattr(gate.artifacts, 'prepare_history', lambda head: calls.append(('history', head)))
    original_write_bytes = Path.write_bytes
    def stage_receipt(path, content):
        if path.name == 'gate-result.json':
            assert not path.is_relative_to(tmp_path / 'upload')
            assert not list((tmp_path / 'upload').iterdir())
        return original_write_bytes(path, content)
    monkeypatch.setattr(Path, 'write_bytes', stage_receipt)
    code = gate.main(['shard'])
    assert calls[0] == ('history', 'a' * 40)
    assert len(calls) == 2
    if scan_fault:
        assert code == 75
        assert list((tmp_path / 'upload').iterdir()) == []
    else:
        assert code == 0
        files = list((tmp_path / 'upload').iterdir())
        assert [path.name for path in files] == ['gate-result.json']
        found = json.loads(files[0].read_text())
        assert found['manifest_sha256'] == reference['manifest_sha256']
        assert found['partition_sha256'] == reference['partition_sha256']


@pytest.mark.parametrize('mode,include_test,build_exit,passes', [
    ('pr', False, 0, True), ('pr', True, 0, True),
    ('main', True, 0, True), ('main', False, 0, False),
    ('pr', True, 1, False),
])
def test_build_only_partitions_survive_the_cli_and_controller_handoff(
        tmp_path, monkeypatch, mode, include_test, build_exit, passes):
    monkeypatch.setattr(gate.artifacts, 'require_scan_python', lambda: None)
    monkeypatch.setattr(gate.artifacts, 'scan_runtime_available', lambda: True)
    from argparse import Namespace
    from types import SimpleNamespace
    from tools.ci import remote_gate
    library = '//prismaquant:library'
    population = [library] + ([LABEL] if include_test else [])
    row = {'population': population, 'tests': [LABEL] if include_test else [],
           'shards': [[label] for label in population]}
    document = {'schema': 'prismaquant.bazel-shards.v1', 'mode': mode,
        'base': 'b' * 40, 'head': 'a' * 40, 'selection': 'impacted',
        'scopes': ['//prismaquant'], 'all_scopes': False, 'reason': [],
        'hosted': [], 'hosted_arm64': [],
        'architectures': {arch: row for arch in (
            ('amd64', 'arm64') if mode == 'main' else ('amd64',))}}
    raw = json.dumps(document).encode()
    monkeypatch.setenv('BUILDBUDDY_API_KEY', 'build-only-upload-sentinel')
    monkeypatch.setattr(gate.artifacts, 'prepare_history', lambda head: None)
    calls, records = [], {}
    execute = gate.execute_shard
    def run_partition(manifest):
        def bazel(command, **kwargs):
            calls.append((list(manifest['tests']), command[1]))
            report = Path(next(word.split('=', 1)[1] for word in command
                               if word.startswith('--build_event_json_file=')))
            if manifest['tests']:
                report.write_text(events(summary(), result()))
                return SimpleNamespace(returncode=0)
            report.write_text(events({'id': {'started': {}}, 'started': {'uuid': 'build-only'}}))
            return SimpleNamespace(returncode=build_exit if command[1] == 'build' else 4)
        return execute(manifest, root=tmp_path, runner=bazel)
    monkeypatch.setattr(gate, 'execute_shard', run_partition)
    class NativeBoundaries:
        def call(self, method, body, **kwargs):
            if method == 'Run':
                identity = 'runner-' + str(len(records))
                upload = tmp_path / identity
                with monkeypatch.context() as scoped:
                    scoped.setenv('PQ_BAZEL_SHARD', body['env']['PQ_BAZEL_SHARD'])
                    scoped.setenv('BUILDBUDDY_ARTIFACTS_DIRECTORY', str(upload))
                    scoped.setattr(gate, 'fetch_artifact', lambda uri: raw)
                    code = gate.main(['shard'])
                records[identity] = {'code': code, 'result': (upload / 'gate-result.json').read_bytes()}
                return {'invocation_id': identity}
            if method == 'GetInvocation':
                identity = body['selector']['invocation_id']
                return {'invocation': [{'invocationId': identity,
                    'invocationStatus': 'COMPLETE_INVOCATION_STATUS',
                    'success': records[identity]['code'] == 0,
                    'artifacts': [{'name': 'gate-result.json', 'uri': 'bytestream://' + identity}]}]}
            if method == 'GetFile':
                return records[body['uri'].removeprefix('bytestream://')]['result']
            raise AssertionError(method)
    args = Namespace(mode=mode, base='b' * 40, head='a' * 40, branch='probe',
                     repo='https://github.com/RobTand/prismaquant', timeout=20)
    try:
        found = remote_gate.run_manifest(NativeBoundaries(), args,
            'bytestream://manifest', raw, clock=lambda: 1)
    except remote_gate.RemoteError:
        assert not passes
    else:
        assert (found['verdict'] == 'pass') is passes
    if passes:
        assert any(not tests and operation == 'build' for tests, operation in calls)
        assert all(operation == 'test' for tests, operation in calls if tests)
        assert all(record['code'] == 0 for record in records.values())


@pytest.mark.parametrize('fault,phase,category', [
    ('native', 'native-execution', 'runtime'),
    ('missing_bep', 'bep-read', 'missing-file'),
    ('invalid_bep', 'bep-parse', 'invalid-json'),
    ('timeout', 'inventory-fetch', 'timeout'),
    ('http', 'inventory-fetch', 'http-429'),
    ('local', 'inventory-local-read', 'io-directory'),
    ('local_late', 'inventory-local-read', 'io-directory'),
    ('reset', 'inventory-remote-fetch', 'network-reset'),
    ('ssl', 'inventory-remote-fetch', 'ssl-certificate'),
    ('invalid_zip', 'inventory-parse', 'invalid-zip'),
    ('invalid_receipt', 'inventory-validate', 'invalid-evidence'),
])
def test_shard_refusal_names_fixed_phase_and_category_without_private_values(
        tmp_path, monkeypatch, capsys, fault, phase, category):
    from types import SimpleNamespace
    sentinel = 'private-value-must-never-be-logged'
    upload = tmp_path / 'upload'
    manifest = {'mode': 'pr', 'base': 'b' * 40, 'head': 'a' * 40,
                'branch': 'probe', 'architecture': 'amd64', 'index': 0,
                'targets': [LABEL], 'tests': [LABEL], 'require_execution': False}
    monkeypatch.setenv('BUILDBUDDY_ARTIFACTS_DIRECTORY', str(upload))
    monkeypatch.setenv('PQ_BAZEL_SHARD', '{}')
    monkeypatch.setenv('BUILDBUDDY_API_KEY', sentinel)
    monkeypatch.setattr(gate.artifacts, 'require_scan_python', lambda: None)
    monkeypatch.setattr(gate.artifacts, 'prepare_history', lambda head: None)
    monkeypatch.setattr(gate, 'load_shard_manifest', lambda reference, fetch: manifest)
    native_calls = []
    local_output = tmp_path / sentinel
    local_output.mkdir()
    output_uri = local_output.as_uri() if fault.startswith('local') else 'bytestream://remote.buildbuddy.io/own-output'
    first_output = tmp_path / 'good-output.zip'
    first_output.write_bytes(inventory_zip({'nodes': {}}))
    output_rows = ([output_event(first_output.as_uri()), output_event(first_output.as_uri())]
                   if fault == 'local_late' else [])
    output_rows.append(output_event(output_uri))
    output_rows.append(output_event('bytestream://remote.buildbuddy.io/later-output'))

    def native(command, **kwargs):
        native_calls.append(command)
        if fault == 'native':
            raise RuntimeError(sentinel)
        path = Path(next(word.split('=', 1)[1] for word in command
                         if word.startswith('--build_event_json_file=')))
        if fault != 'missing_bep':
            path.write_text('{' + sentinel if fault == 'invalid_bep' else
                            events(result(remote=True), summary(), *output_rows))
        return SimpleNamespace(returncode=0)

    def fetch(uri):
        if fault == 'timeout':
            raise TimeoutError(sentinel)
        if fault in ('http', 'reset', 'ssl'):
            # The real client hides network exception text. Its category must
            # survive that boundary without exposing the private URL or value.
            def unavailable(request, **kwargs):
                if fault == 'http':
                    raise urllib.error.HTTPError(sentinel, 429, sentinel, None, None)
                reason = (ConnectionResetError(errno.ECONNRESET, sentinel) if fault == 'reset'
                          else ssl.SSLCertVerificationError(1, sentinel))
                raise urllib.error.URLError(reason) from None
            monkeypatch.setattr(urllib.request, 'urlopen', unavailable)
            return gate.fetch_artifact(uri)
        if fault.startswith('local'):
            return gate.fetch_artifact(uri)
        if fault == 'invalid_zip':
            return sentinel.encode()
        return inventory_zip({'nodes': {sentinel: 'unknown'}})

    execute = gate.execute_shard
    monkeypatch.setattr(gate, 'execute_shard', lambda document: execute(
        document, root=tmp_path, runner=native, fetch=fetch))
    assert gate.main(['shard']) == 75
    captured = capsys.readouterr()
    if fault == 'http':
        phase = 'inventory-remote-fetch'
    expected = f'shard refused at {phase}; category={category}; upload refused\n'
    if phase.startswith('inventory-'):
        local = 2 if fault == 'local_late' else int(fault == 'local')
        remote = 1 if fault.startswith('local') else 2
        index = 2 if fault == 'local_late' else 1
        expected += (f'inventory sources: local={local} remote={remote}; '
                     f'failed at index {index} of {local + remote}\n')
    assert captured.err == expected
    assert sentinel not in captured.out + captured.err
    assert len(native_calls) == 1
    assert list(upload.iterdir()) == []


@pytest.mark.parametrize('error,category', [
    (OSError(errno.EIO, 'private'), 'io-device'),
    (OSError(errno.ELOOP, 'private'), 'io-symlink-loop'),
    (OSError(errno.ENOSPC, 'private'), 'io-no-space'),
    (OSError(errno.EMFILE, 'private'), 'io-fd-limit'),
    (OSError(errno.ENFILE, 'private'), 'io-fd-limit'),
    (OSError(errno.ESTALE, 'private'), 'io-stale'),
    (OSError(errno.ENOMEM, 'private'), 'io-resource-limit'),
    (OSError(errno.EINVAL, 'private'), 'io'),
    (socket.gaierror(socket.EAI_AGAIN, 'private'), 'network-dns'),
    (ConnectionRefusedError(errno.ECONNREFUSED, 'private'), 'network-refused'),
    (ConnectionAbortedError(errno.ECONNABORTED, 'private'), 'network-aborted'),
    (BrokenPipeError(errno.EPIPE, 'private'), 'network-broken-pipe'),
    (OSError(errno.ENETUNREACH, 'private'), 'network-unreachable'),
    (OSError(errno.EHOSTUNREACH, 'private'), 'network-unreachable'),
    (ssl.SSLEOFError(1, 'private'), 'ssl-eof'),
    (ssl.SSLError(1, 'private'), 'ssl-protocol'),
])
def test_inventory_error_categories_are_allowlisted_and_survive_masked_transport(error, category):
    assert gate.refusal_category(error) == category
    try:
        raise urllib.error.URLError(error) from None
    except urllib.error.URLError:
        try:
            raise RuntimeError('masked private transport') from None
        except RuntimeError as masked:
            assert gate.refusal_category(masked) == category


def test_inventory_progress_does_not_change_successful_duplicate_or_mixed_source_reads(tmp_path):
    first = tmp_path / 'inventory.zip'
    first.write_bytes(inventory_zip({'nodes': {'first': 'passed'}}))
    text = events(output_event(first.as_uri()), output_event(first.as_uri()), output_event())
    calls = []

    def fetch(uri):
        calls.append(uri)
        return (gate.fetch_artifact(uri) if uri.startswith('file://')
                else inventory_zip({'nodes': {'second': 'skipped'}}))

    assert gate.inventory_evidence(text, fetch) == {
        'nodes': {'first': 'passed', 'second': 'skipped'}, 'budget_warnings': []}
    assert len(calls) == 2


def test_manifest_transport_refusal_keeps_manifest_phase(tmp_path, monkeypatch, capsys):
    upload = tmp_path / 'upload'
    sentinel = 'private-manifest-transport'
    monkeypatch.setenv('BUILDBUDDY_ARTIFACTS_DIRECTORY', str(upload))
    monkeypatch.setenv('PQ_BAZEL_SHARD', json.dumps({'manifest_uri': 'bytestream://private-manifest'}))
    monkeypatch.setenv('BUILDBUDDY_API_KEY', sentinel)
    monkeypatch.setattr(gate.artifacts, 'require_scan_python', lambda: None)

    def unavailable(request, **kwargs):
        raise urllib.error.URLError(ConnectionResetError(errno.ECONNRESET, sentinel)) from None

    monkeypatch.setattr(urllib.request, 'urlopen', unavailable)
    assert gate.main(['shard']) == 75
    captured = capsys.readouterr()
    assert captured.err == 'shard refused at manifest; category=network-reset; upload refused\n'
    assert sentinel not in captured.out + captured.err
    assert list(upload.iterdir()) == []
