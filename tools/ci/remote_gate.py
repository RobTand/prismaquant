#!/usr/bin/env python3
"""Remote Bazel requests and exact-run evidence for the PrismaQuant cutover."""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
import urllib.error
import urllib.request

API = 'https://app.buildbuddy.io/api/v1/'
IMAGE = ('docker://gcr.io/flame-public/rbe-ubuntu24-04@sha256:'
         '5f0c01e5f6d79e58d7a996330de33aa38ce599bf8431eaf27521e7602daadf50')
SCHEMA = 'prismaquant.bazel-gate.v1'
RESULT = 'gate-result.json'


class RemoteError(RuntimeError):
    """Remote evidence is unavailable or does not establish success."""


class BuildBuddy:
    def __init__(self, key, api=API):
        if not key:
            raise RemoteError('BUILDBUDDY_API_KEY is unavailable')
        self.key, self.api = key, api

    def call(self, method, body, *, raw=False, timeout=120):
        request = urllib.request.Request(
            self.api + method, data=json.dumps(body).encode(),
            headers={'x-buildbuddy-api-key': self.key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read() if raw else json.load(response)
        except (urllib.error.URLError, TimeoutError, ValueError):
            raise RemoteError(f'BuildBuddy {method} request failed') from None


def run_request(args):
    if not args.base or not args.head:
        raise RemoteError('empty base or head: queue-aware refs are unavailable')
    command = ['tools/ci/bazel_gate.py', args.mode, '--base', args.base, '--head', args.head,
               '--branch', args.branch]
    return {'repo': args.repo, 'branch': args.branch, 'commit_sha': args.head,
            'steps': [{'run': 'python3 ' + shlex.join(command)}],
            'platform_properties': {
                'EstimatedCPU': '4', 'EstimatedMemory': '8GB', 'EstimatedFreeDiskBytes': '50GB',
                'container-image': IMAGE, 'workload-isolation-type': 'firecracker',
                'recycle-runner': 'true'},
            'timeout': f'{args.timeout}m', 'wait_until': 'STARTED'}


def shard_request(args, architecture, index, manifest_uri, manifest_sha256,
                  partition_sha256, *, require_execution=False):
    """One bounded runner drives self-hosted actions for exactly one partition."""
    if architecture not in ('amd64', 'arm64') or index < 0:
        raise RemoteError('invalid shard partition')
    if (not manifest_uri.startswith('bytestream://')
            or any(not re.fullmatch('[0-9a-f]{64}', value)
                   for value in (manifest_sha256, partition_sha256))):
        raise RemoteError('invalid immutable shard manifest reference')
    request = run_request(args)
    manifest = {'mode': args.mode, 'base': args.base, 'head': args.head,
                'branch': args.branch, 'architecture': architecture, 'index': index,
                'manifest_uri': manifest_uri, 'manifest_sha256': manifest_sha256,
                'partition_sha256': partition_sha256,
                'require_execution': require_execution}
    request['steps'] = [{'run': 'python3 tools/ci/bazel_gate.py shard'}]
    request['env'] = {'PQ_BAZEL_SHARD': json.dumps(manifest, separators=(',', ':'))}
    request['timeout'] = '55m'
    return request


def submit_shards(client, requests, head, *, clock=time.monotonic):
    """Resolve each own result before admitting another runner; never retry a shard."""
    results = []
    for request in requests:
        expected = json.loads(request['env']['PQ_BAZEL_SHARD'])
        response = client.call('Run', request)
        invocation_id = response.get('invocationId', response.get('invocation_id'))
        if not invocation_id:
            raise RemoteError('shard submission returned no acknowledgement ID')
        print('Remote Bazel shard: ' + invocation_id, flush=True)
        record = await_invocation(client, invocation_id, clock() + 58 * 60, clock=clock)
        artifacts = [row for row in record.get('artifacts', []) if row.get('name') == RESULT]
        if len(artifacts) != 1:
            raise RemoteError('shard has no unique own result artifact')
        result = parse_result(client.call('GetFile', {'uri': artifacts[0]['uri']}, raw=True), head)
        if any(result.get(key) != expected[key] for key in (
                'architecture', 'index', 'manifest_sha256', 'partition_sha256')):
            raise RemoteError('shard result has the wrong partition')
        targets = result.get('targets')
        if (not isinstance(targets, list) or not targets
                or any(not isinstance(label, str) for label in targets)
                or hashlib.sha256(('\n'.join(targets) + '\n').encode()).hexdigest()
                   != expected['partition_sha256']):
            raise RemoteError('shard result has the wrong partition population')
        if result['mode'] != expected['mode'] or result.get('error'):
            raise RemoteError('shard result does not establish its requested mode')
        results.append(result)
    return results


def succeeded(record):
    code = record.get('bazelExitCode', record.get('bazel_exit_code'))
    return (record.get('invocationStatus') == 'COMPLETE_INVOCATION_STATUS'
            and record.get('success') is True and code in (None, 0, 'OK', 'SUCCESS'))


def await_invocation(client, invocation_id, deadline, *, interval=15,
                     clock=time.monotonic, sleep=time.sleep):
    while True:
        records = client.call('GetInvocation', {'selector': {'invocation_id': invocation_id},
                                               'include_artifacts': True}).get('invocation', [])
        if len(records) > 1:
            raise RemoteError('remote run evidence is ambiguous')
        record = records[0] if records else {}
        identity = record.get('invocationId') or record.get('id', {}).get('invocationId')
        if record and identity != invocation_id:
            raise RemoteError('remote run evidence has the wrong invocation ID')
        status = record.get('invocationStatus')
        if status == 'COMPLETE_INVOCATION_STATUS':
            if not succeeded(record):
                raise RemoteError(f'remote run {invocation_id} failed')
            return record
        if status == 'DISCONNECTED_INVOCATION_STATUS':
            raise RemoteError(f'remote run {invocation_id} disconnected')
        if clock() >= deadline:
            raise RemoteError(f'remote run {invocation_id} did not finish before its deadline')
        sleep(min(interval, max(0, deadline - clock())))


def parse_result(raw, head):
    if raw is None:
        raise RemoteError(f'remote run left no {RESULT}')
    try:
        result = json.loads(raw)
    except (ValueError, TypeError):
        raise RemoteError(f'{RESULT} is not valid JSON') from None
    if not isinstance(result, dict) or result.get('schema') != SCHEMA or result.get('head') != head:
        raise RemoteError(f'{RESULT} has the wrong schema or head')
    if result.get('mode') not in ('pr', 'main'):
        raise RemoteError(f'{RESULT} has no valid mode')
    if not result.get('error') and result['mode'] == 'pr' and result.get('verdict') not in (
            'pass', 'fail', 'skipped'):
        raise RemoteError(f'{RESULT} has no valid test verdict')
    return result
