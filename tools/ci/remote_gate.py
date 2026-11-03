#!/usr/bin/env python3
"""Remote Bazel requests and exact-run evidence for the PrismaQuant cutover."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prismaquant.digests import DIRECT_ASCII_SPACED_LAX, bytes_sha256hex, text_sha256hex
from tools.ci import bazel_artifacts as artifacts
from tools.ci.bazel_artifacts import BuildBuddy, RemoteError

IMAGE = ('docker://gcr.io/flame-public/rbe-ubuntu24-04@sha256:'
         '5f0c01e5f6d79e58d7a996330de33aa38ce599bf8431eaf27521e7602daadf50')
SCHEMA = 'prismaquant.bazel-gate.v1'
RESULT = 'gate-result.json'


@contextlib.contextmanager
def _evidence_phase(phase):
    try:
        yield
    except (RemoteError, OSError, ValueError, KeyError, TypeError, RuntimeError):
        print('Remote evidence refused at ' + phase, file=sys.stderr)
        raise


def run_request(args):
    if not args.base or not args.head:
        raise RemoteError('empty base or head: queue-aware refs are unavailable')
    command = ['tools/ci/bazel_selection.py', 'plan', '--mode', args.mode,
               '--base', args.base, '--head', args.head,
               '--branch', args.branch]
    return {'repo': args.repo, 'branch': args.branch, 'commit_sha': args.head,
            'steps': [{'run': 'python3 ' + shlex.join(command)}],
            'platform_properties': {
                'EstimatedCPU': '4', 'EstimatedMemory': '8GB', 'EstimatedFreeDiskBytes': '50GB',
                'container-image': IMAGE, 'workload-isolation-type': 'firecracker',
                'recycle-runner': 'true'},
            'timeout': f'{args.timeout}m', 'wait_until': 'STARTED'}


def shard_request(args, architecture, index, manifest_uri, manifest_sha256,
                  partition_sha256, *, require_execution=False, selection='impacted'):
    """One bounded runner drives self-hosted actions for exactly one partition."""
    if (architecture not in ('amd64', 'arm64') or index < 0
            or selection not in ('impacted', 'irrelevant', 'fallback')):
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
                'require_execution': require_execution, 'selection': selection}
    request['steps'] = [{'run': 'python3 tools/ci/bazel_gate.py shard'}]
    request['env'] = {'PQ_BAZEL_SHARD': json.dumps(manifest, separators=(',', ':'))}
    request['timeout'] = '55m'
    return request


def submit_shards(client, requests, head, *, concurrency=1, clock=time.monotonic):
    """Only PR fallback may admit two runs; drain each batch without retries."""
    if type(concurrency) is not int or not 1 <= concurrency <= 2:
        raise RemoteError('shard concurrency must be one or two')
    planned = [(request, json.loads(request['env']['PQ_BAZEL_SHARD']))
               for request in requests]
    if concurrency == 2 and (not planned or any(
            expected['mode'] != 'pr' or expected.get('selection') != 'fallback'
            for _, expected in planned)):
        raise RemoteError('concurrency above one is allowed only for PR fallback')
    results = []
    for start in range(0, len(planned), concurrency):
        pending, error = [], None
        for request, expected in planned[start:start + concurrency]:
            try:
                response = client.call('Run', request)
                invocation_id = response.get('invocationId', response.get('invocation_id'))
                if not invocation_id:
                    raise RemoteError('shard submission returned no acknowledgement ID')
                print('Remote Bazel shard: ' + invocation_id, flush=True)
                pending.append((invocation_id, expected, clock() + 58 * 60))
            except RemoteError as failure:
                error = failure
                break
        # A failed sibling cannot abandon an already admitted run or permit
        # a new batch. Read each known run's own terminal evidence first.
        for invocation_id, expected, deadline in pending:
            try:
                results.append(verify_shard_result(client, invocation_id, expected,
                                                   head, deadline, clock=clock))
            except RemoteError as failure:
                error = error or failure
        if error:
            raise error
    return results


def run_manifest(client, args, manifest_uri, raw_manifest, *,
                 clock=time.monotonic, fallback_concurrency=2):
    """Validate every partition before submitting any run, then join own results."""
    from tools.ci.bazel_gate import GateError, aggregate_shards, check_partition

    try:
        document = json.loads(raw_manifest)
        if (document['schema'] != 'prismaquant.bazel-shards.v1'
                or args.mode not in ('pr', 'main')
                or any(document[key] != getattr(args, key)
                       for key in ('mode', 'base', 'head'))
                or document['selection'] not in ('impacted', 'fallback', 'irrelevant')):
            raise RemoteError('manifest has the wrong source, mode or schema')
        architectures = ('amd64', 'arm64') if args.mode == 'main' else ('amd64',)
        if set(document['architectures']) != set(architectures):
            raise RemoteError('manifest has the wrong architecture population')
        if (not isinstance(document['scopes'], list)
                or any(not isinstance(scope, str) for scope in document['scopes'])
                or type(document['all_scopes']) is not bool
                or not isinstance(document['reason'], list)):
            raise RemoteError('manifest has invalid scope evidence')
        for row in document['architectures'].values():
            population, tests, shards = (row[key] for key in ('population', 'tests', 'shards'))
            if (not isinstance(population, list) or not isinstance(tests, list)
                    or not isinstance(shards, list)
                    or any(not isinstance(label, str) for label in population + tests)
                    or len(population) != len(set(population))
                    or len(tests) != len(set(tests)) or not set(tests) <= set(population)):
                raise RemoteError('manifest has an invalid build or test population')
            if population:
                check_partition(population, shards)
            elif shards or tests or args.mode != 'pr' or document['selection'] == 'fallback':
                raise RemoteError('manifest has an invalid empty population')
        if type(fallback_concurrency) is not int or not 1 <= fallback_concurrency <= 2:
            raise RemoteError('fallback concurrency must be one or two')
    except (ValueError, TypeError, AttributeError, KeyError, GateError) as error:
        raise RemoteError('invalid shard manifest') from error

    manifest_sha256 = bytes_sha256hex(raw_manifest)
    results = {}
    concurrency = (fallback_concurrency if args.mode == 'pr'
                   and document['selection'] == 'fallback' else 1)
    for architecture in architectures:
        row = document['architectures'][architecture]
        if not row['population']:
            results[architecture] = {'verdict': 'skipped', 'shards': [],
                'executed_test_actions': 0, 'budget_warnings': [], 'error': None}
            continue
        requests = [shard_request(args, architecture, index, manifest_uri,
            manifest_sha256, text_sha256hex('\n'.join(labels) + '\n'),
            require_execution=args.mode == 'main', selection=document['selection'])
            for index, labels in enumerate(row['shards'])]
        records = submit_shards(client, requests, args.head,
                                concurrency=concurrency, clock=clock)
        try:
            results[architecture] = aggregate_shards(row['population'], row['shards'],
                records, require_execution=args.mode == 'main', test_population=row['tests'])
        except (GateError, KeyError, TypeError) as error:
            raise RemoteError('shard evidence does not cover its manifest') from error
    verdicts = {result['verdict'] for result in results.values()}
    return {'schema': SCHEMA, 'mode': args.mode, 'base': args.base, 'head': args.head,
            'selection': document['selection'], 'scopes': document['scopes'],
            'all_scopes': document['all_scopes'], 'reason': document['reason'],
            'manifest_sha256': manifest_sha256, 'architectures': results,
            'verdict': 'fail' if 'fail' in verdicts else 'pass' if 'pass' in verdicts else 'skipped',
            'budget_warnings': [warning for result in results.values()
                                for warning in result['budget_warnings']]}


def run_gate(client, args, *, clock=time.monotonic):
    """Read this planner's complete, source-bound manifest before any test Run."""
    response = client.call('Run', run_request(args))
    invocation = response.get('invocationId', response.get('invocation_id'))
    if not invocation:
        raise RemoteError('plan submission returned no acknowledgement ID')
    print('Remote Bazel plan: ' + invocation, flush=True)
    with _evidence_phase('plan-await'):
        record = await_invocation(client, invocation, clock() + args.timeout * 60, clock=clock)
    with _evidence_phase('plan-validate'):
        if record.get('commitSha') != args.head:
            raise RemoteError('plan runner has the wrong source head')
        artifacts = {}
        for name in (RESULT, 'shard-manifest.json'):
            rows = [row for row in record.get('artifacts', []) if row.get('name') == name]
            if (len(rows) != 1 or not isinstance(rows[0].get('uri'), str)
                    or not rows[0]['uri'].startswith('bytestream://')):
                raise RemoteError('plan has no unique own ' + name + ' artifact')
            artifacts[name] = rows[0]['uri']
    with _evidence_phase('plan-fetch'):
        receipt_raw = client.call('GetFile', {'uri': artifacts[RESULT]}, raw=True)
        raw = client.call('GetFile', {'uri': artifacts['shard-manifest.json']}, raw=True)
    with _evidence_phase('plan-validate'):
        receipt = parse_result(receipt_raw, args.head)
        try:
            manifest = json.loads(raw)
            if (receipt.get('operation') != 'plan' or receipt.get('error')
                    or receipt.get('verdict') != 'pass'
                    or receipt.get('mode') != args.mode or receipt.get('base') != args.base
                    or receipt.get('manifest_sha256') != bytes_sha256hex(raw)
                    or any(receipt.get(key) != manifest[key] for key in (
                        'mode', 'base', 'head', 'selection', 'scopes', 'all_scopes',
                        'reason', 'hosted', 'hosted_arm64'))):
                raise RemoteError('plan receipt does not match its source and manifest')
        except (ValueError, TypeError, KeyError):
            raise RemoteError('plan manifest is unavailable or malformed') from None
    result = run_manifest(client, args, artifacts['shard-manifest.json'], raw, clock=clock)
    result.update({'plan_invocation': invocation, 'hosted': manifest['hosted'],
                   'hosted_arm64': manifest['hosted_arm64']})
    return result


def verify_shard_result(client, invocation_id, expected, head, deadline, *, clock):
    with _evidence_phase('shard-await'):
        record = await_invocation(client, invocation_id, deadline, clock=clock)
    with _evidence_phase('shard-validate'):
        artifacts = [row for row in record.get('artifacts', []) if row.get('name') == RESULT]
        if len(artifacts) != 1:
            raise RemoteError('shard has no unique own result artifact')
    with _evidence_phase('shard-fetch'):
        raw = client.call('GetFile', {'uri': artifacts[0]['uri']}, raw=True)
    with _evidence_phase('shard-validate'):
        result = parse_result(raw, head)
        if any(result.get(key) != expected[key] for key in (
                'architecture', 'index', 'manifest_sha256', 'partition_sha256')):
            raise RemoteError('shard result has the wrong partition')
        targets = result.get('targets')
        if (not isinstance(targets, list) or not targets
                or any(not isinstance(label, str) for label in targets)
                or text_sha256hex('\n'.join(targets) + '\n')
                   != expected['partition_sha256']):
            raise RemoteError('shard result has the wrong partition population')
        if result.get('base') != expected['base']:
            raise RemoteError('shard result has the wrong source base')
        if result['mode'] != expected['mode'] or result.get('error'):
            raise RemoteError('shard result does not establish its requested mode')
        return {**result, 'runner_invocation': invocation_id,
                'invocation_status': record['invocationStatus']}


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['pr', 'main'])
    parser.add_argument('--base', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--branch', required=True)
    parser.add_argument('--repo', default='https://github.com/RobTand/prismaquant')
    parser.add_argument('--timeout', type=int, default=55, choices=range(1, 56))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    upload = args.output
    phase = 'upload-precondition'
    try:
        if any(not re.fullmatch('[0-9a-f]{40}', revision) for revision in (args.base, args.head)):
            raise RemoteError('queue-aware source revisions are invalid')
        upload.mkdir(parents=True, exist_ok=True)
        if upload.is_symlink() or list(upload.iterdir()):
            raise RemoteError('controller upload directory is not empty')
        phase = 'scanner-runtime'
        artifacts.require_scan_python()
        phase = 'remote-evidence'
        result = run_gate(BuildBuddy(os.environ.get('BUILDBUDDY_API_KEY')), args)
        keys = ['BUILDBUDDY_API_KEY'] + [name for name in (
            'GITHUB_TOKEN', 'MERGIFY_TOKEN') if os.environ.get(name)]
        phase = 'publication'
        artifacts.publish_json(upload, {
            'result.json': DIRECT_ASCII_SPACED_LAX.encoded(result) + b'\n'}, keys)
        if output := os.environ.get('GITHUB_OUTPUT'):
            with Path(output).open('a') as stream:
                stream.write('scopes=' + json.dumps(result['scopes'], separators=(',', ':')) + '\n')
                stream.write('all_scopes=' + str(result['all_scopes']).lower() + '\n')
                stream.write('scopes_ready=true\n')
        if summary := os.environ.get('GITHUB_STEP_SUMMARY'):
            with Path(summary).open('a') as stream:
                stream.write('Bazel verdict: ' + result['verdict'] + '\n\n')
                stream.write('Budget warnings: ' + str(len(result['budget_warnings'])) + '\n')
        return 0 if result['verdict'] in ('pass', 'skipped') else 1
    except (RemoteError, RuntimeError, OSError, ValueError, KeyError, TypeError):
        if upload.is_symlink():
            upload.unlink()
        elif upload.exists():
            shutil.rmtree(upload)
        upload.mkdir(parents=True, exist_ok=True)
        print('Remote gate refused at ' + phase + '; upload refused', file=sys.stderr)
        return 75


if __name__ == '__main__':
    raise SystemExit(main())
