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
    return {'schema': remote.SCHEMA, 'head': HEAD, 'mode': 'pr', 'verdict': 'pass',
            'architecture': 'amd64', 'index': index, 'targets': [f'//tests:test_{index}'],
            'manifest_sha256': 'c' * 64,
            'partition_sha256': hashlib.sha256(f'//tests:test_{index}\n'.encode()).hexdigest(),
            'run': {'statuses': {f'//tests:test_{index}': 'PASSED'}, 'returncode': 0}}


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
    assert found == [shard_result(0), shard_result(1)]
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
