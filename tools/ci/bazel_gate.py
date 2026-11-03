"""Read private Bazel evidence for the cutover's per-architecture records."""
from __future__ import annotations

from collections import Counter
import json


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
