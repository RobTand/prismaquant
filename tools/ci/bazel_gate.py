"""Read private Bazel evidence for the cutover's per-architecture records."""
from __future__ import annotations

from collections import Counter
import json


class GateError(RuntimeError):
    """The gate cannot establish complete build or test evidence."""


def partition_population(population, *, shard_count=8):
    """Sorted striding keeps full runs bounded without changing their population."""
    labels = sorted(set(population))
    if not labels:
        raise GateError('full population is empty')
    if shard_count < 1:
        raise GateError('partition needs a positive shard count')
    shards = [labels[index::min(shard_count, len(labels))]
              for index in range(min(shard_count, len(labels)))]
    check_partition(labels, shards)
    return shards


def check_partition(population, shards):
    counts = Counter(label for shard in shards for label in shard)
    if (not shards or any(not shard for shard in shards)
            or counts != Counter(set(population))):
        raise GateError('partition does not cover the queried population exactly once')


def aggregate_shards(population, shards, records, *, require_execution=False):
    """Every planned shard needs its own complete invocation and exact test evidence."""
    check_partition(population, shards)
    if Counter(record['index'] for record in records) != Counter(range(len(shards))):
        raise GateError('missing or duplicate shard records')
    ordered = sorted(records, key=lambda record: record['index'])
    results = []
    for labels, record in zip(shards, ordered):
        if record['invocation_status'] != 'COMPLETE_INVOCATION_STATUS':
            raise GateError('shard invocation is incomplete or disconnected')
        run = record['run']
        if set(run['statuses']) != set(labels):
            raise GateError('shard evidence has a different test population')
        result = architecture_result(labels, run, budget_warnings=record['budget_warnings'],
                                     require_execution=require_execution)
        results.append(result)
    return {'verdict': 'pass' if all(result['verdict'] == 'pass' for result in results) else 'fail',
            'shards': results,
            'executed_test_actions': sum(result['executed_test_actions'] for result in results),
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
