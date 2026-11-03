"""The cutover Remote Bazel controller fails closed on its own run evidence."""
from __future__ import annotations

from argparse import Namespace
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
