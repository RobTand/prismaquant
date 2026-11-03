"""Read private Bazel evidence for the cutover's per-architecture records."""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

SCHEMA = 'prismaquant.bazel-gate.v1'


class GateError(RuntimeError):
    """The gate cannot establish complete build or test evidence."""


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


def execute_shard(manifest, *, root=None, runner=subprocess.run):
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
    command = ['bazel', 'test', '--config=' + config, '--config=ci', '--jobs=16',
               '--spawn_strategy=remote,local', '--remote_local_fallback=false',
               '--keep_going', '--flaky_test_attempts=1',
               '--build_event_json_file=' + str(bep),
               '--target_pattern_file=' + str(patterns),
               '--build_metadata=COMMIT_SHA=' + manifest['head'],
               '--build_metadata=BRANCH_NAME=' + manifest['branch']]
    if manifest['require_execution']:
        command += ['--noremote_accept_cached', '--nocache_test_results']
    with (private / 'bazel.log').open('w') as diagnostic:
        outcome = runner(command, cwd=root, env=environment, stdout=diagnostic,
                         stderr=subprocess.STDOUT, check=False)
    try:
        run = bep_run(bep.read_text(), returncode=outcome.returncode)
    except OSError:
        raise GateError('shard left no private build-event evidence') from None
    result = architecture_result(tests, run, require_execution=manifest['require_execution'])
    return {'schema': SCHEMA, 'mode': manifest['mode'], 'head': manifest['head'],
            'base': manifest['base'], 'architecture': manifest['architecture'],
            'index': manifest['index'], 'targets': targets, 'run': run, **result}
