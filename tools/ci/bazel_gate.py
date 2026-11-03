"""Read private Bazel evidence for the cutover's per-architecture records."""
from __future__ import annotations

from collections import Counter
import argparse
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.ci import bazel_artifacts as artifacts
from prismaquant.digests import DIRECT_ASCII_SPACED_LAX, bytes_sha256hex, text_sha256hex

SCHEMA = 'prismaquant.bazel-gate.v1'


class GateError(RuntimeError):
    """The gate cannot establish complete build or test evidence."""


def hosted_assignment(plan, result, architecture):
    """Narrow the existing hosted plan to this verified result's exact lane."""
    try:
        if (architecture not in ('amd64', 'arm64')
                or plan['schema'] != 'prismaquant.gate_plan.v1'
                or result['schema'] != SCHEMA or result['mode'] not in ('pr', 'main')
                or result['verdict'] not in ('pass', 'skipped')
                or architecture == 'arm64' and result['mode'] != 'main'
                or any(not re.fullmatch('[0-9a-f]{40}', result[key])
                       or plan[key + '_sha'] != result[key] for key in ('base', 'head'))):
            raise GateError('hosted assignment has inconsistent source or lane evidence')
        labels = result['hosted' if architecture == 'amd64' else 'hosted_arm64']
        if (not isinstance(labels, list) or any(not isinstance(label, str) for label in labels)
                or len(labels) != len(set(labels))):
            raise GateError('hosted assignment has duplicate or invalid labels')
        paths = set()
        for label in labels:
            if not re.fullmatch(r'//tests(?:/[^:\s]+)?:[^:\s]+', label):
                raise GateError('hosted label is not a test target')
            raw = label[2:].replace(':', '/') + '.py'
            path = PurePosixPath(raw)
            if path.as_posix() != raw or any(part in ('.', '..') for part in path.parts):
                raise GateError('hosted label has a noncanonical path')
            paths.add(raw)
        rows = {row['path']: row for row in plan['selected']}
        if len(rows) != len(plan['selected']) or not paths <= rows.keys():
            raise GateError('hosted label is outside the existing test population')
        return {**plan, 'selected': [rows[path] for path in sorted(paths)],
                'deselect': [node for node in plan['deselect'] if node.split('::', 1)[0] in paths]}
    except (KeyError, TypeError, ValueError, AttributeError):
        raise GateError('hosted assignment evidence is malformed') from None


def partition_population(population, *, shard_count=8, tests=None):
    """Sorted striding keeps full runs bounded without changing their population."""
    labels = sorted(set(population))
    if not labels:
        raise GateError('full population is empty')
    if shard_count < 1:
        raise GateError('partition needs a positive shard count')
    count = min(shard_count, len(labels))
    if tests is None:
        shards = [labels[index::count] for index in range(count)]
    else:
        tests = set(tests)
        if not tests <= set(labels):
            raise GateError('test population is outside the queried build population')
        # Tests and their libraries often alternate in label order. Striding
        # that order as one list can put all tests into a single shard.
        shards = [[] for _ in range(count)]
        for index, label in enumerate(sorted(tests) + sorted(set(labels) - tests)):
            shards[index % count].append(label)
        shards = [sorted(shard) for shard in shards]
    check_partition(labels, shards)
    return shards


def check_partition(population, shards):
    counts = Counter(label for shard in shards for label in shard)
    if (not shards or any(not shard for shard in shards)
            or counts != Counter(set(population))):
        raise GateError('partition does not cover the queried population exactly once')


def aggregate_shards(population, shards, records, *, require_execution=False,
                     test_population=None):
    """Every planned shard needs its own complete invocation and exact test evidence."""
    check_partition(population, shards)
    tests = set(population if test_population is None else test_population)
    if not tests <= set(population):
        raise GateError('test population is outside the queried build population')
    if Counter(record['index'] for record in records) != Counter(range(len(shards))):
        raise GateError('missing or duplicate shard records')
    ordered = sorted(records, key=lambda record: record['index'])
    results = []
    for labels, record in zip(shards, ordered):
        if record['invocation_status'] != 'COMPLETE_INVOCATION_STATUS':
            raise GateError('shard invocation is incomplete or disconnected')
        run = record['run']
        selected_tests = set(labels) & tests
        if set(run['statuses']) != selected_tests:
            raise GateError('shard evidence has a different test population')
        result = architecture_result(selected_tests, run,
                                     budget_warnings=record['budget_warnings'])
        results.append(result)
    executed = sum(result['executed_test_actions'] for result in results)
    missing_execution = require_execution and executed == 0
    return {'verdict': 'fail' if missing_execution or any(
                result['verdict'] == 'fail' for result in results) else 'pass',
            'shards': results,
            'executed_test_actions': executed,
            'error': 'architecture ran no test actions' if missing_execution else None,
            'budget_warnings': [warning for result in results for warning in result['budget_warnings']]}


def bep_run(text, *, returncode):
    statuses, executed = {}, set()
    invocation = None
    try:
        for line in text.splitlines():
            event = json.loads(line)
            identity = event.get('id', {})
            if 'started' in identity:
                invocation = event.get('started', {}).get('uuid')
            summary = identity.get('testSummary')
            if summary:
                label = summary['label']
                status = event.get('testSummary', {}).get('overallStatus', 'NO_STATUS')
                # Conflicting terminal records cannot erase an observed failure.
                if statuses.get(label, 'PASSED') == 'PASSED':
                    statuses[label] = status
            result = identity.get('testResult')
            if result:
                evidence = event.get('testResult', {})
                if (evidence.get('status') and not evidence.get('cachedLocally', False)
                        and not evidence.get('executionInfo', {}).get('cachedRemotely', False)):
                    executed.add((result['label'], result.get('run', 1),
                                  result.get('shard', 0), result.get('attempt', 1)))
    except (ValueError, TypeError, AttributeError, KeyError):
        raise GateError('Bazel build events are malformed or incomplete') from None
    counts = dict(Counter(row[0] for row in executed))
    return {'statuses': statuses, 'invocation': invocation, 'returncode': returncode,
            'executed_test_actions': len(executed), 'executed_by_label': counts}


def architecture_result(population, run, *, budget_warnings=(), require_execution=False):
    failed = sorted(label for label in population if run['statuses'].get(label) != 'PASSED')
    executed = sum(run['executed_by_label'].get(label, 0) for label in population)
    missing_execution = require_execution and executed == 0
    if failed or run['returncode'] or missing_execution:
        verdict = 'fail'
    else:
        verdict = 'pass' if population else 'skipped'
    return {'verdict': verdict, 'failed': failed, 'invocation': run['invocation'],
            'bazel_exit_code': run['returncode'], 'executed_test_actions': executed,
            'budget_warnings': list(budget_warnings),
            'error': 'architecture ran no test actions' if missing_execution else None}


def load_shard_manifest(reference, fetch):
    """Resolve a compact immutable reference before admitting any Bazel command."""
    try:
        raw = fetch(reference['manifest_uri'])
        if bytes_sha256hex(raw) != reference['manifest_sha256']:
            raise GateError('shard manifest digest does not match its reference')
        document = json.loads(raw)
        if (document['schema'] != 'prismaquant.bazel-shards.v1'
                or any(document[key] != reference[key] for key in ('mode', 'base', 'head'))):
            raise GateError('shard manifest has the wrong source or schema')
        architecture = document['architectures'][reference['architecture']]
        population, tests, shards = (architecture[key] for key in ('population', 'tests', 'shards'))
        if (len(population) != len(set(population)) or not set(tests) <= set(population)):
            raise GateError('shard manifest has an invalid build or test population')
        check_partition(population, shards)
        index = reference['index']
        if type(index) is not int or not 0 <= index < len(shards):
            raise GateError('shard manifest has no requested partition')
        targets = shards[index]
        digest = text_sha256hex('\n'.join(targets) + '\n')
        if digest != reference['partition_sha256']:
            raise GateError('shard manifest partition digest does not match')
        return {**reference, 'targets': targets, 'tests': sorted(set(targets) & set(tests))}
    except (ValueError, KeyError, TypeError, AttributeError, OSError):
        raise GateError('shard manifest is malformed or unavailable') from None


def inventory_evidence(text, fetch):
    """Read Python node receipts from only the output URIs in this run's BEP."""
    seen, nodes, warnings = set(), {}, []
    try:
        for line in text.splitlines():
            event = json.loads(line)
            for output in event.get('testResult', {}).get('testActionOutput', []):
                name = output.get('name')
                if name not in ('test.outputs__outputs.zip', 'test.outputs/outputs.zip',
                                 'test.outputs/inventory.json'):
                    continue
                uri = output['uri']
                if uri in seen:
                    continue
                seen.add(uri)
                if name.endswith('inventory.json'):
                    receipts = [json.loads(fetch(uri))]
                else:
                    with zipfile.ZipFile(io.BytesIO(fetch(uri))) as archive:
                        receipts = [json.loads(archive.read(member)) for member in archive.namelist()
                                    if Path(member).name == 'inventory.json']
                for receipt in receipts:
                    found_nodes, found_warnings = receipt.get('nodes', {}), receipt.get('budget_warnings', [])
                    if not isinstance(found_nodes, dict) or not isinstance(found_warnings, list):
                        raise ValueError('invalid inventory structure')
                    for node, status in found_nodes.items():
                        if not isinstance(node, str) or status not in ('passed', 'skipped', 'failed'):
                            raise ValueError('invalid inventory node')
                        if nodes.get(node) != 'failed':
                            nodes[node] = status
                    for row in found_warnings:
                        if (not isinstance(row, dict) or not isinstance(row.get('nodeid'), str)
                                or any(type(row.get(key)) not in (int, float)
                                       for key in ('seconds', 'budget_seconds'))):
                            raise ValueError('invalid budget warning')
                        warnings.append({key: row[key] for key in ('nodeid', 'seconds', 'budget_seconds')})
    except (ValueError, TypeError, AttributeError, KeyError, OSError, zipfile.BadZipFile):
        raise GateError('test output inventory is malformed or unavailable') from None
    return {'nodes': nodes, 'budget_warnings': warnings}


def execute_shard(manifest, *, root=None, runner=subprocess.run, fetch=None):
    """Execute one immutable build partition, keeping diagnostics out of uploads."""
    root = Path(root or Path(__file__).resolve().parents[2])
    targets, tests = manifest['targets'], manifest['tests']
    if (manifest['architecture'] not in ('amd64', 'arm64') or not targets
            or len(set(targets)) != len(targets) or not set(tests) <= set(targets)
            or any(not re.fullmatch(r'//[^\s]*:[^\s]+', label) for label in targets)
            or not re.fullmatch('[0-9a-f]{40}', manifest['head'])):
        raise GateError('invalid immutable shard manifest')
    spec = importlib.util.spec_from_file_location(
        'pq_shard_artifacts', Path(__file__).with_name('bazel_artifacts.py'))
    artifacts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(artifacts)
    environment = artifacts.child_environment()
    # Only Bazel's client receives its remote-auth key. Neither this key nor
    # any other controller credential is forwarded to the test action env.
    if os.environ.get('BUILDBUDDY_API_KEY'):
        environment['BUILDBUDDY_API_KEY'] = os.environ['BUILDBUDDY_API_KEY']
    private = Path(tempfile.mkdtemp(prefix='pq-bazel-shard-'))
    patterns, bep = private / 'targets.txt', private / 'events.jsonl'
    patterns.write_text('\n'.join(targets) + '\n')
    config = 'remote' if manifest['architecture'] == 'amd64' else 'remote-arm64'
    command = ['bazel', 'test' if tests else 'build', '--config=' + config, '--config=ci', '--jobs=16',
               '--spawn_strategy=remote,local', '--remote_local_fallback=false',
               '--keep_going',
               '--build_event_json_file=' + str(bep),
               '--target_pattern_file=' + str(patterns),
               '--build_metadata=COMMIT_SHA=' + manifest['head'],
               '--build_metadata=BRANCH_NAME=' + manifest['branch']]
    if tests:
        command += ['--flaky_test_attempts=1', '--zip_undeclared_test_outputs']
    if manifest['require_execution']:
        command += ['--noremote_accept_cached', '--nocache_test_results']
    with (private / 'bazel.log').open('w') as diagnostic:
        outcome = runner(command, cwd=root, env=environment, stdout=diagnostic,
                         stderr=subprocess.STDOUT, check=False)
    try:
        text = bep.read_text()
        run = bep_run(text, returncode=outcome.returncode)
    except OSError:
        raise GateError('shard left no private build-event evidence') from None
    inventory = inventory_evidence(text, fetch or fetch_artifact)
    result = architecture_result(tests, run, budget_warnings=inventory['budget_warnings'],
                                 require_execution=manifest['require_execution'] and bool(tests))
    if not tests and result['verdict'] == 'skipped':
        result['verdict'] = 'pass'
    return {'schema': SCHEMA, 'mode': manifest['mode'], 'head': manifest['head'],
            'base': manifest['base'], 'architecture': manifest['architecture'],
            'index': manifest['index'], 'targets': targets, 'run': run,
            'nodes': inventory['nodes'], **result}


def fetch_artifact(uri):
    if uri.startswith('file://'):
        return Path(unquote(urlsplit(uri).path)).read_bytes()
    if not uri.startswith('bytestream://'):
        raise GateError('artifact has an unsupported URI')
    spec = importlib.util.spec_from_file_location(
        'pq_shard_output_client', Path(__file__).with_name('bazel_artifacts.py'))
    artifacts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(artifacts)
    return artifacts.BuildBuddy(os.environ.get('BUILDBUDDY_API_KEY')).call(
        'GetFile', {'uri': uri}, raw=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    commands.add_parser('shard')
    hosted = commands.add_parser('hosted')
    hosted.add_argument('--architecture', choices=['amd64', 'arm64'], required=True)
    hosted.add_argument('--plan', type=Path, required=True)
    hosted.add_argument('--result', type=Path, required=True)
    hosted.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == 'hosted':
        try:
            assignment = hosted_assignment(json.loads(args.plan.read_text()),
                                           json.loads(args.result.read_text()), args.architecture)
            with args.output.open('xb') as stream:
                stream.write(DIRECT_ASCII_SPACED_LAX.encoded(assignment) + b'\n')
        except (GateError, ValueError, TypeError, OSError):
            print('hosted assignment evidence unavailable or invalid', file=sys.stderr)
            return 75
        return 0
    directory = os.environ.get('BUILDBUDDY_ARTIFACTS_DIRECTORY')
    if not directory:
        print('shard upload directory unavailable', file=sys.stderr)
        return 75
    upload = Path(directory)
    phase = 'upload-precondition'
    try:
        upload.mkdir(parents=True, exist_ok=True)
        if upload.is_symlink() or list(upload.iterdir()):
            raise GateError('shard upload directory is not empty')
        phase = 'scanner-runtime'
        artifacts.require_scan_python()
        phase = 'manifest'
        reference = json.loads(os.environ['PQ_BAZEL_SHARD'])
        manifest = load_shard_manifest(reference, fetch_artifact)
        phase = 'history'
        artifacts.prepare_history(manifest['head'])
        phase = 'execution'
        result = execute_shard(manifest)
        result.update({key: reference[key] for key in ('manifest_sha256', 'partition_sha256')})
        keys = ['BUILDBUDDY_API_KEY'] + [name for name in ('GITHUB_TOKEN', 'MERGIFY_TOKEN')
                                       if os.environ.get(name)]
        phase = 'publication'
        artifacts.publish_json(upload, {
            'gate-result.json': DIRECT_ASCII_SPACED_LAX.encoded(result) + b'\n'}, keys)
        return 0 if result['verdict'] == 'pass' else 1
    except (GateError, ValueError, TypeError, KeyError, OSError, RuntimeError):
        if upload.is_symlink():
            upload.unlink()
        elif upload.exists():
            shutil.rmtree(upload)
        upload.mkdir(parents=True, exist_ok=True)
        print('shard refused at ' + phase + '; upload refused', file=sys.stderr)
        return 75


if __name__ == '__main__':
    raise SystemExit(main())
