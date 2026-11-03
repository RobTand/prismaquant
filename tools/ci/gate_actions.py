#!/usr/bin/env python3
"""Evaluate hosted gate attempt records without retrying test failures."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_partition import assigned_nodes, validate_node_partition


def _node_coverage(partition, record, excluded):
    if record.get('node_partition') != partition:
        return False
    coverage = record.get('node_coverage', {})
    if not isinstance(coverage, dict):
        return False
    complete = record.get('execution_coverage', {})
    if not isinstance(complete, dict):
        return False
    selected, ran = complete.get('selected'), complete.get('executed')
    if (not isinstance(selected, list) or not isinstance(ran, list) or
            any(not isinstance(node, str) or node.split('::', 1)[0] not in record['files']
                for node in selected + ran) or len(selected) != len(set(selected)) or
            sorted(selected) != sorted(ran)):
        return False
    collected, executed = coverage.get('collected'), coverage.get('executed')
    if not isinstance(collected, dict) or set(collected) != set(partition['owners']):
        return False
    for path, nodes in collected.items():
        if (not isinstance(nodes, list) or any(not isinstance(node, str) or
                not node.startswith(path + '::') for node in nodes) or len(nodes) != len(set(nodes))):
            return False
    return (isinstance(executed, list) and all(isinstance(node, str) for node in executed) and
            sorted(executed) == assigned_nodes(collected, partition, excluded) and
            sorted(executed) == sorted(node for node in ran if node.split('::', 1)[0] in partition['owners']))

def _verdict(plan, files, record, partition=None):
    if not isinstance(record, dict):
        return None
    if any(record.get(field) != plan.get(field) for field in ('base_sha', 'head_sha', 'mode')):
        return None
    if record.get('population_version') != plan['population']['version']:
        return None
    if (not isinstance(record.get('files'), list) or
            any(not isinstance(path, str) for path in record['files']) or
            sorted(record['files']) != sorted(files)):
        return None
    code = record.get('pytest_exit_code')
    budget = record.get('budget')
    if (record.get('verdict') == 'fail' and isinstance(budget, dict) and
            budget.get('status') == 'fail' and (code is None or type(code) is int and code in {0, 1})):
        return 'fail'
    if type(code) is not int:
        return None
    verdict = record.get('verdict')
    if not isinstance(verdict, str):
        return None
    if verdict == 'fail':
        return verdict
    if (verdict, code) != ('pass', 0):
        return None
    if partition and (not isinstance(budget, dict) or budget.get('enforced') is not True or
                      budget.get('status') != 'pass' or
                      not _node_coverage(partition, record, plan.get('deselect', []))):
        return None
    return verdict


def assess_attempts(plan, assignments, attempts):
    """Return retry indices and the final verdict, preserving any test failure."""
    indices = [row['index'] for row in assignments]
    partitions = [row.get('node_partition') for row in assignments]
    split_files = set()
    if any(partitions):
        for row, partition in zip(assignments, partitions):
            validate_node_partition(partition, row['files'])
            if (partition['count'] != len(assignments) or partition['index'] != row['index'] or
                    partition['owners'] != partitions[0]['owners']):
                raise ValueError('shards must recompute an identical node partition')
        split_files = set(partitions[0]['owners'])
    assigned = Counter(path for row in assignments for path in row['files'] if path not in split_files)
    selected = [row['path'] for row in plan['selected']]
    if (len(indices) != len(set(indices)) or len(selected) != len(set(selected)) or
            any(len(row['files']) != len(set(row['files'])) for row in assignments) or
            set(assigned) | split_files != set(selected) or any(count != 1 for count in assigned.values())):
        raise ValueError('assignments must cover every selected file exactly once, with unique indices')
    retry, failed, missing, effective = [], [], [], {}
    for row in assignments:
        index, files = row['index'], row['files']
        partition = row.get('node_partition')
        first = _verdict(plan, files, attempts.get(1, {}).get(index), partition)
        second = _verdict(plan, files, attempts.get(2, {}).get(index), partition)
        if first is None:
            retry.append(index)
        if 'fail' in (first, second):
            verdict, attempt = 'fail', 1 if first == 'fail' else 2
        elif first is None:
            verdict, attempt = second, 2
        else:
            verdict, attempt = first, 1
        if verdict is None:
            missing.append(index)
        else:
            effective[index] = attempt
            if verdict == 'fail':
                failed.append(index)
    coverage = None
    warnings = [warning for index, attempt in effective.items()
                for warning in attempts[attempt][index].get('budget_warnings', [])
                if warning['nodeid'] not in split_files]
    if split_files and not missing and not failed:
        records = {index: attempts[attempt][index] for index, attempt in effective.items()}
        reference = next(iter(records.values()))['node_coverage']['collected']
        drift = [index for index, record in records.items()
                 if record['node_coverage']['collected'] != reference]
        if drift:
            missing.extend(drift)
            retry.extend(index for index in drift if effective[index] == 1 and index not in retry)
        else:
            executed = Counter(node for record in records.values() for node in record['node_coverage']['executed'])
            expected = {node for nodes in reference.values() for node in nodes} - set(plan.get('deselect', []))
            if set(executed) != expected or any(count != 1 for count in executed.values()):
                missing.extend(indices)
            else:
                coverage = {'split_files': len(split_files), 'nodes': len(expected), 'exactly_once': True}
                limits = next(iter(records.values()))['budget'].get('file_limits', {})
                if any(record['budget'].get('file_limits', {}) != limits for record in records.values()):
                    missing.extend(indices)
                for path, limit in limits.items():
                    totals = [record['budget'].get('file_totals', {}).get(path) for record in records.values()]
                    if (path not in split_files or type(limit) not in (float, int) or
                            not math.isfinite(limit) or limit <= 0 or
                            any(type(value) not in (float, int) or not math.isfinite(value) or value < 0
                                for value in totals)):
                        missing.extend(indices)
                        continue
                    seconds = sum(totals)
                    if seconds > limit:
                        warnings.append({'nodeid': path, 'seconds': seconds, 'budget_seconds': limit})
    return {'schema': 'prismaquant.gate_aggregate.v1', 'retry_indices': sorted(retry),
            'failed_indices': sorted(set(failed)), 'missing_indices': sorted(set(missing)),
            'budget_warnings': warnings,
            'effective_attempts': effective, **({'node_coverage': coverage} if coverage else {}),
            'exit_code': 1 if failed else 75 if missing else 0}


def _read_result(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['retries', 'finish'])
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--durations', type=Path, default=Path('tools/ci/gate_durations.json'))
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--prefix', default='gate-result')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--effective-output', type=Path)
    parser.add_argument('--summary', type=Path)
    args = parser.parse_args(argv)
    try:
        spec = importlib.util.spec_from_file_location('gate_partition', Path(__file__).with_name('gate_partition.py'))
        partition = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(partition)
        plan = json.loads(args.plan.read_text())
        matrix = json.loads(args.matrix.read_text())
        recorded = json.loads(args.durations.read_text())
        costs = recorded['files']
        assignments = partition.shards([row['path'] for row in plan['selected']], costs,
                              matrix['shards'], unknown_seconds=partition.default_unknown_seconds(costs),
                              node_durations=recorded.get('nodes'), runner_factors=matrix.get('runner_cost_factors'),
                              excluded=plan.get('deselect', []))
        if (len(matrix['include']) != len(assignments) or
                sorted(row['index'] for row in matrix['include']) != list(range(len(assignments)))):
            raise ValueError('matrix indices do not match the complete partition')
        attempts = {attempt: {row['index']: _read_result(args.artifacts /
                    f"{args.prefix}-a{attempt}-{row['index']}" / 'result.json') for row in assignments}
                    for attempt in (1, 2)}
        result = assess_attempts(plan, assignments, attempts)
        if args.command == 'retries':
            include = [row for row in matrix['include'] if row['index'] in result['retry_indices']]
            # Keep a valid nonempty expression even when the job's `if` skips the matrix.
            output = {'include': include or matrix['include'][:1]}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output) + '\n')
            print('retry=' + str(bool(include)).lower())
            print('matrix=' + json.dumps(output, separators=(',', ':')))
            return 0
        if args.effective_output:
            for index, attempt in result['effective_attempts'].items():
                source = args.artifacts / f'{args.prefix}-a{attempt}-{index}'
                target = args.effective_output / str(index)
                target.mkdir(parents=True, exist_ok=True)
                for name in ('result.json', 'junit.xml'):
                    if (source / name).is_file():
                        shutil.copyfile(source / name, target / name)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        marker = {0: 'GATE_PASS', 1: 'GATE_TEST_FAILURE', 75: 'NO_VERDICT'}[result['exit_code']]
        message = f"{marker}: failed shards {result['failed_indices']}; missing verdicts {result['missing_indices']}"
        print(message)
        if args.summary:
            with args.summary.open('a') as handle:
                handle.write('## Gate verdict\n\n' + message + '\n')
                warnings = result['budget_warnings']
                if warnings:
                    handle.write('\n## Gate budget_warnings\n\n' +
                                 f'{len(warnings)} warnings; complete list in final.json.\n\n')
                    for row in warnings[:25]:
                        handle.write(f"- {row['nodeid'][:1000]}: {row['seconds']:.3f}s > "
                                     f"{row['budget_seconds']:.3f}s\n")
                    if len(warnings) > 25:
                        handle.write(f'{len(warnings) - 25} more warnings in final.json.\n')
        return result['exit_code']
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('NO_VERDICT: ' + str(error), file=sys.stderr)
        return 75


if __name__ == '__main__':
    raise SystemExit(main())
