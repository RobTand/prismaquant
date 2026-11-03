"""The cutover Remote Bazel controller fails closed on its own run evidence."""
from __future__ import annotations

from argparse import Namespace
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('pq_remote_gate', ROOT / 'tools/ci/remote_gate.py')
remote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote)
HEAD = 'a' * 40
BASE = 'b' * 40


def arguments(mode='pr'):
    return Namespace(mode=mode, base=BASE, head=HEAD, branch='bbrun/cutover',
                     repo='https://github.com/RobTand/prismaquant', timeout=90)


@pytest.mark.parametrize('mode', ['pr', 'main'])
def test_runner_keeps_its_cloud_placement_and_four_cpu_size(mode, monkeypatch):
    monkeypatch.setenv('BUILDBUDDY_API_KEY', 'remote-controller-sentinel')
    request = remote.run_request(arguments(mode))
    assert request['commit_sha'] == HEAD
    assert request['platform_properties'] == {
        'EstimatedCPU': '4', 'EstimatedMemory': '8GB', 'EstimatedFreeDiskBytes': '50GB',
        'container-image': remote.IMAGE, 'workload-isolation-type': 'firecracker',
        'recycle-runner': 'true'}
    assert 'remote-controller-sentinel' not in json.dumps(request)
    assert request['steps'][0]['run'].startswith('python3 tools/ci/bazel_gate.py ' + mode)
    assert request['wait_until'] == 'STARTED'


def test_empty_refs_cannot_start_a_remote_gate():
    args = arguments()
    args.base = ''
    with pytest.raises(remote.RemoteError, match='base'):
        remote.run_request(args)


@pytest.mark.parametrize('status,success,code', [
    ('DISCONNECTED_INVOCATION_STATUS', None, None),
    ('COMPLETE_INVOCATION_STATUS', False, 3),
    ('COMPLETE_INVOCATION_STATUS', False, 36),
])
def test_failed_or_disconnected_runs_are_not_retried(status, success, code):
    class Client:
        def __init__(self):
            self.methods = []
        def call(self, method, body):
            self.methods.append(method)
            return {'invocation': [{'invocationId': 'own-run', 'invocationStatus': status,
                                    'success': success, 'bazelExitCode': code}]}
    client = Client()
    with pytest.raises(remote.RemoteError):
        remote.await_invocation(client, 'own-run', 100, clock=lambda: 1)
    assert client.methods == ['GetInvocation']


def test_a_submission_acknowledgement_is_not_a_pass():
    class Client:
        def call(self, method, body):
            return {'invocation': []}
    with pytest.raises(remote.RemoteError, match='finish'):
        remote.await_invocation(Client(), 'ack-only', 0, clock=lambda: 1)


def test_completed_run_returns_only_after_its_own_success():
    record = {'invocationId': 'own-run', 'invocationStatus': 'COMPLETE_INVOCATION_STATUS',
              'success': True, 'bazelExitCode': 0}
    class Client:
        def call(self, method, body):
            assert body['selector']['invocation_id'] == 'own-run'
            return {'invocation': [record]}
    assert remote.await_invocation(Client(), 'own-run', 100, clock=lambda: 1) == record


def test_public_api_nested_invocation_id_is_accepted():
    record = {'id': {'invocationId': 'own-run'},
              'invocationStatus': 'COMPLETE_INVOCATION_STATUS',
              'success': True, 'bazelExitCode': 0}
    class Client:
        def call(self, method, body):
            return {'invocation': [record]}
    assert remote.await_invocation(Client(), 'own-run', 100, clock=lambda: 1) == record


def test_public_api_nested_invocation_id_cannot_return_another_run():
    class Client:
        def call(self, method, body):
            return {'invocation': [{'id': {'invocationId': 'foreign-run'},
                'invocationStatus': 'COMPLETE_INVOCATION_STATUS',
                'success': True, 'bazelExitCode': 0}]}
    with pytest.raises(remote.RemoteError, match='wrong invocation'):
        remote.await_invocation(Client(), 'own-run', 100, clock=lambda: 1)


@pytest.mark.parametrize('raw', [None, b'not JSON', b'{}',
    json.dumps({'schema': 'prismaquant.bazel-gate.v1', 'head': BASE, 'verdict': 'pass'}).encode()])
def test_missing_or_wrong_head_results_fail(raw):
    with pytest.raises(remote.RemoteError):
        remote.parse_result(raw, HEAD)


def test_budget_warnings_survive_a_passing_result():
    result = {'schema': 'prismaquant.bazel-gate.v1', 'head': HEAD, 'mode': 'pr',
              'verdict': 'pass', 'budget_warnings': [
                  {'nodeid': 'tests/test_one.py::test_one', 'seconds': 2, 'budget_seconds': 1}]}
    assert remote.parse_result(json.dumps(result).encode(), HEAD) == result


def shard_result(index=0):
    return {'schema': remote.SCHEMA, 'head': HEAD, 'base': BASE, 'mode': 'pr', 'verdict': 'pass',
            'architecture': 'amd64', 'index': index, 'targets': [f'//tests:test_{index}'],
            'manifest_sha256': 'c' * 64,
            'partition_sha256': hashlib.sha256(f'//tests:test_{index}\n'.encode()).hexdigest(),
            'budget_warnings': [],
            'run': {'statuses': {f'//tests:test_{index}': 'PASSED'}, 'returncode': 0,
                    'invocation': f'child-{index}',
                    'executed_by_label': {f'//tests:test_{index}': 1}}}


def verified_shard(index):
    return {**shard_result(index), 'runner_invocation': f'parent-{index}',
            'invocation_status': 'COMPLETE_INVOCATION_STATUS'}


class ShardClient:
    def __init__(self, *, status='COMPLETE_INVOCATION_STATUS', missing=False, wrong_head=False):
        self.calls = []
        self.status, self.missing, self.wrong_head = status, missing, wrong_head

    def call(self, method, body, **kwargs):
        self.calls.append(method)
        index = self.calls.count('Run') - 1
        if method == 'Run':
            return {'invocation_id': f'parent-{index}'}
        if method == 'GetInvocation':
            assert body['selector']['invocation_id'] == f'parent-{index}'
            return {'invocation': [{'invocationId': f'parent-{index}',
                'invocationStatus': self.status, 'success': True, 'bazelExitCode': 0,
                'artifacts': [] if self.missing else [
                    {'name': remote.RESULT, 'uri': f'bytestream://own-result-{index}'}]}]}
        if method == 'GetFile':
            assert body['uri'] == f'bytestream://own-result-{index}'
            row = shard_result(index)
            if self.wrong_head:
                row['head'] = BASE
            return json.dumps(row).encode()
        raise AssertionError(method)


def shard_requests():
    return [remote.shard_request(arguments(), 'amd64', index,
                                 'bytestream://remote.buildbuddy.io/own-selection-manifest',
                                 'c' * 64, shard_result(index)['partition_sha256'])
            for index in range(2)]


def test_full_runs_use_serial_subhour_requests_and_own_result_artifacts():
    client = ShardClient()
    requests = shard_requests()
    assert all(request['timeout'] == '55m' for request in requests)
    assert all(request['platform_properties'] == remote.run_request(arguments())[
        'platform_properties'] for request in requests)
    assert all('//tests:test_' not in json.dumps(request) for request in requests)
    assert all(len(json.dumps(request)) < 2000 for request in requests)
    found = remote.submit_shards(client, requests, HEAD, clock=lambda: 1)
    assert found == [verified_shard(0), verified_shard(1)]
    assert client.calls == ['Run', 'GetInvocation', 'GetFile'] * 2


@pytest.mark.parametrize('fault', ['missing', 'disconnected', 'wrong_head'])
def test_incomplete_shard_stops_without_submitting_another_or_retrying(fault):
    client = ShardClient(missing=fault == 'missing', wrong_head=fault == 'wrong_head',
        status='DISCONNECTED_INVOCATION_STATUS' if fault == 'disconnected'
               else 'COMPLETE_INVOCATION_STATUS')
    with pytest.raises(remote.RemoteError):
        remote.submit_shards(client, shard_requests(), HEAD, clock=lambda: 1)
    assert client.calls.count('Run') == 1


@pytest.mark.parametrize('field,value', [('index', 1), ('architecture', 'arm64'),
                                        ('targets', ['//tests:foreign'])])
def test_shard_result_must_match_the_exact_requested_partition(field, value):
    class WrongPartition(ShardClient):
        def call(self, method, body, **kwargs):
            result = super().call(method, body, **kwargs)
            if method == 'GetFile':
                row = json.loads(result)
                row[field] = value
                return json.dumps(row).encode()
            return result
    client = WrongPartition()
    with pytest.raises(remote.RemoteError, match='partition'):
        remote.submit_shards(client, shard_requests(), HEAD, clock=lambda: 1)
    assert client.calls.count('Run') == 1


class BoundedFallbackClient:
    """Own admitted IDs remain active until their own terminal evidence is read."""
    def __init__(self, fail_first=False):
        self.active = set()
        self.events = []
        self.fail_first = fail_first

    def call(self, method, body, **kwargs):
        if method == 'Run':
            index = json.loads(body['env']['PQ_BAZEL_SHARD'])['index']
            self.active.add(index)
            self.events.append(('submit', index))
            assert len(self.active) <= 2, 'more than two admitted fallback shards'
            return {'invocation_id': f'parent-{index}'}
        if method == 'GetInvocation':
            index = int(body['selector']['invocation_id'].removeprefix('parent-'))
            self.events.append(('terminal', index))
            return {'invocation': [{'invocationId': f'parent-{index}',
                'invocationStatus': ('DISCONNECTED_INVOCATION_STATUS'
                    if self.fail_first and index == 0 else 'COMPLETE_INVOCATION_STATUS'),
                'success': True, 'bazelExitCode': 0,
                'artifacts': [{'name': remote.RESULT, 'uri': f'bytestream://result-{index}'}]}]}
        if method == 'GetFile':
            index = int(body['uri'].removeprefix('bytestream://result-'))
            self.active.remove(index)
            self.events.append(('evidence', index))
            return json.dumps(shard_result(index)).encode()
        raise AssertionError(method)


def fallback_requests(count=3, *, mode='pr', selection='fallback'):
    return [remote.shard_request(arguments(mode), 'amd64', index,
        'bytestream://remote.buildbuddy.io/own-selection-manifest', 'c' * 64,
        shard_result(index)['partition_sha256'], selection=selection) for index in range(count)]


def test_fallback_may_admit_two_shards_and_checks_both_before_the_next_batch():
    client = BoundedFallbackClient()
    found = remote.submit_shards(client, fallback_requests(), HEAD,
                                 concurrency=2, clock=lambda: 1)
    assert found == [verified_shard(0), verified_shard(1), verified_shard(2)]
    assert client.events == [('submit', 0), ('submit', 1), ('terminal', 0),
        ('evidence', 0), ('terminal', 1), ('evidence', 1),
        ('submit', 2), ('terminal', 2), ('evidence', 2)]
    assert not client.active


@pytest.mark.parametrize('concurrency,mode,selection', [
    (0, 'pr', 'fallback'), (3, 'pr', 'fallback'),
    (2, 'pr', 'impacted'), (2, 'pr', 'irrelevant'), (2, 'main', 'fallback'),
])
def test_concurrency_above_one_is_bounded_and_exclusive_to_pr_fallback(
        concurrency, mode, selection):
    client = BoundedFallbackClient()
    with pytest.raises(remote.RemoteError, match='concurrency|fallback'):
        remote.submit_shards(client, fallback_requests(mode=mode, selection=selection),
                             HEAD, concurrency=concurrency, clock=lambda: 1)
    assert client.events == []


def test_fallback_parallel_failure_still_reads_the_other_admitted_shard_and_stops():
    client = BoundedFallbackClient(fail_first=True)
    with pytest.raises(remote.RemoteError):
        remote.submit_shards(client, fallback_requests(), HEAD,
                             concurrency=2, clock=lambda: 1)
    assert client.events == [('submit', 0), ('submit', 1), ('terminal', 0),
                             ('terminal', 1), ('evidence', 1)]


def test_fallback_remains_serial_unless_the_controller_explicitly_selects_two():
    client = BoundedFallbackClient()
    remote.submit_shards(client, fallback_requests(count=2), HEAD, clock=lambda: 1)
    assert client.events == [('submit', 0), ('terminal', 0), ('evidence', 0),
                             ('submit', 1), ('terminal', 1), ('evidence', 1)]


def test_shard_enters_the_gate_before_its_python314_bootstrap():
    request = remote.shard_request(arguments(), 'amd64', 0,
        'bytestream://remote.buildbuddy.io/manifest', 'c' * 64, 'd' * 64)
    assert request['steps'] == [{'run': 'python3 tools/ci/bazel_gate.py shard'}]


def test_verified_shard_records_feed_the_exact_union_aggregator():
    from tools.ci.bazel_gate import aggregate_shards

    records = remote.submit_shards(ShardClient(), shard_requests(), HEAD, clock=lambda: 1)
    labels = [f'//tests:test_{index}' for index in range(2)]
    assert [record['invocation_status'] for record in records] == [
        'COMPLETE_INVOCATION_STATUS', 'COMPLETE_INVOCATION_STATUS']
    assert [record['runner_invocation'] for record in records] == ['parent-0', 'parent-1']
    summary = aggregate_shards(labels, [[label] for label in labels], records,
                               require_execution=True)
    assert summary['verdict'] == 'pass'
    assert summary['executed_test_actions'] == 2


def test_same_head_receipt_from_another_base_is_refused():
    class OtherBase(ShardClient):
        def call(self, method, body, **kwargs):
            result = super().call(method, body, **kwargs)
            if method == 'GetFile':
                row = json.loads(result)
                row['base'] = 'd' * 40
                return json.dumps(row).encode()
            return result

    client = OtherBase()
    with pytest.raises(remote.RemoteError, match='base'):
        remote.submit_shards(client, shard_requests(), HEAD, clock=lambda: 1)
    assert client.calls.count('Run') == 1


def controller_manifest(mode='pr', selection='fallback'):
    labels = [f'//tests:test_{index}' for index in range(4)]
    architectures = {}
    for architecture in ('amd64', 'arm64') if mode == 'main' else ('amd64',):
        count = 2 if architecture == 'amd64' else 4
        population = [] if selection == 'irrelevant' else labels
        architectures[architecture] = {'population': population, 'tests': population,
            'shards': [population[index::count] for index in range(count)] if population else []}
    return {'schema': 'prismaquant.bazel-shards.v1', 'mode': mode, 'base': BASE,
        'head': HEAD, 'selection': selection, 'scopes': ['//tests'],
        'all_scopes': selection == 'fallback' or mode == 'main', 'reason': [],
        'hosted': [], 'hosted_arm64': [], 'architectures': architectures}


class ManifestClient:
    def __init__(self, document, *, cached_arm64=False):
        self.document, self.cached_arm64 = document, cached_arm64
        self.references, self.active, self.events = {}, set(), []
        self.max_active = 0

    def call(self, method, body, **kwargs):
        if method == 'Run':
            reference = json.loads(body['env']['PQ_BAZEL_SHARD'])
            identity = f"parent-{reference['architecture']}-{reference['index']}"
            self.references[identity] = reference
            self.active.add(identity)
            self.max_active = max(self.max_active, len(self.active))
            self.events.append(('submit', identity))
            return {'invocation_id': identity}
        if method == 'GetInvocation':
            identity = body['selector']['invocation_id']
            assert identity in self.active
            self.events.append(('terminal', identity))
            return {'invocation': [{'invocationId': identity,
                'invocationStatus': 'COMPLETE_INVOCATION_STATUS', 'success': True,
                'bazelExitCode': 0, 'artifacts': [
                    {'name': remote.RESULT, 'uri': 'bytestream://' + identity}]}]}
        if method == 'GetFile':
            identity = body['uri'].removeprefix('bytestream://')
            reference = self.references[identity]
            self.active.remove(identity)
            self.events.append(('evidence', identity))
            architecture, index = reference['architecture'], reference['index']
            row = self.document['architectures'][architecture]
            labels = row['shards'][index]
            tests = set(labels) & set(row['tests'])
            executed = 0 if self.cached_arm64 and architecture == 'arm64' else 1
            return json.dumps({'schema': remote.SCHEMA, 'mode': reference['mode'],
                'base': BASE, 'head': HEAD, 'architecture': architecture, 'index': index,
                'manifest_sha256': reference['manifest_sha256'],
                'partition_sha256': reference['partition_sha256'],
                'targets': labels, 'verdict': 'pass', 'budget_warnings': [],
                'run': {'statuses': {label: 'PASSED' for label in tests},
                    'invocation': 'child-' + identity, 'returncode': 0,
                    'executed_by_label': {label: executed for label in tests}}}).encode()
        raise AssertionError(method)


@pytest.mark.parametrize('mode,selection,maximum', [
    ('pr', 'fallback', 2), ('pr', 'impacted', 1), ('main', 'fallback', 1),
])
def test_manifest_controller_runs_exact_partitions_with_only_fallback_overlap(mode, selection, maximum):
    document = controller_manifest(mode, selection)
    raw = json.dumps(document).encode()
    client = ManifestClient(document)
    found = remote.run_manifest(client, arguments(mode),
        'bytestream://remote.buildbuddy.io/manifest', raw, clock=lambda: 1)
    assert found['verdict'] == 'pass'
    assert found['scopes'] == document['scopes']
    assert found['all_scopes'] == document['all_scopes']
    assert client.max_active == maximum
    assert not client.active
    assert set(found['architectures']) == set(document['architectures'])
    for reference in client.references.values():
        assert reference['manifest_sha256'] == hashlib.sha256(raw).hexdigest()
        assert reference['require_execution'] is (mode == 'main')


def test_manifest_controller_makes_unread_docs_an_explicit_empty_selection():
    document = controller_manifest(selection='irrelevant')
    client = ManifestClient(document)
    found = remote.run_manifest(client, arguments(), 'bytestream://remote.buildbuddy.io/manifest',
                                json.dumps(document).encode(), clock=lambda: 1)
    assert found['verdict'] == 'skipped'
    assert client.events == []


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'wrong_head', 'missing_arm64'])
def test_manifest_controller_validates_the_whole_union_before_any_submission(fault):
    document = controller_manifest(mode='main')
    row = document['architectures']['arm64']
    if fault == 'duplicate':
        row['shards'][1] = row['shards'][0]
    elif fault == 'missing':
        row['shards'].pop()
    elif fault == 'wrong_head':
        document['head'] = 'd' * 40
    else:
        del document['architectures']['arm64']
    client = ManifestClient(document)
    with pytest.raises(remote.RemoteError):
        remote.run_manifest(client, arguments('main'), 'bytestream://remote.buildbuddy.io/manifest',
                            json.dumps(document).encode(), clock=lambda: 1)
    assert client.events == []


def test_manifest_controller_refuses_a_cache_only_postmerge_architecture():
    document = controller_manifest(mode='main')
    client = ManifestClient(document, cached_arm64=True)
    found = remote.run_manifest(client, arguments('main'),
        'bytestream://remote.buildbuddy.io/manifest', json.dumps(document).encode(), clock=lambda: 1)
    assert found['verdict'] == 'fail'
    assert found['architectures']['arm64']['error'] == 'architecture ran no test actions'
