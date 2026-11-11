#!/usr/bin/env python3
"""Summarize retained execution inventories and downloaded hosted gate artifacts."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prismaquant.digests import file_sha256hex
from tools.ci.gate_run import gate_junit_identity


def hosted(directories, default, *, file_budgets=()):
    file_budgets = set(file_budgets)
    tests = defaultdict(list)
    file_tests = defaultdict(list)
    files = defaultdict(lambda: defaultdict(list))
    runs = []
    failures = defaultdict(set)
    def gate_elapsed_seconds(start, end):
        if not start or not end:
            return None
        seconds = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
        if seconds < 0:
            raise ValueError('hosted timestamps run backwards')
        return seconds
    for directory in directories:
        metadata = json.loads((directory / 'run.json').read_text())
        jobs = json.loads((directory / 'jobs.json').read_text())['jobs']
        final = next((job for job in jobs if job['name'] == 'gate' and job.get('conclusion') != 'skipped'), {})
        job_times = []
        for job in jobs:
            steps = {step['name']: step for step in job.get('steps', [])}
            testing = steps.get('Run assigned gate tests', {})
            install = steps.get('Install CPU test environment', {})
            job_times.append({**{key: job.get(key) for key in
                ('name', 'started_at', 'completed_at', 'conclusion', 'labels', 'steps')},
                'setup_seconds': gate_elapsed_seconds(job.get('started_at'), testing.get('started_at')),
                'test_seconds': gate_elapsed_seconds(testing.get('started_at'), testing.get('completed_at')),
                'install_seconds': gate_elapsed_seconds(install.get('started_at'), install.get('completed_at'))})
        requested = metadata.get('requested_at') or (metadata.get('created_at')
            if metadata['run_attempt'] == 1 else None) or metadata.get('run_started_at')
        runs.append({'id': metadata['id'], 'attempt': metadata['run_attempt'],
                     'head_sha': metadata['head_sha'], 'conclusion': metadata['conclusion'],
                     'final_check': final.get('name'), 'final_conclusion': final.get('conclusion'),
                     'requested_at': requested,
                     'wall_seconds': gate_elapsed_seconds(requested, final.get('completed_at')),
                     'jobs': job_times})
        # The final artifact exports only effective attempts. Do not count retries twice.
        attempt = metadata['run_attempt']
        effective = list(directory.glob(f'gate-final-{attempt}/effective/*/junit.xml'))
        effective += list(directory.glob(f'gate-pr-final-{attempt}/effective/*/junit.xml'))
        if not effective:
            effective = list(directory.glob(f'gate-result-{attempt}-a1-*/junit.xml'))
        matrices = list(directory.glob(f'gate-plan-{attempt}/job-matrix.json'))
        matrix = json.loads(matrices[0].read_text()) if matrices else {'include': []}
        runners = {row['index']: row['runner'] for row in matrix['include']}
        run_files = defaultdict(lambda: {'seconds': 0.0, 'runners': set(), 'artifacts': []})
        for path in effective:
            index = int(path.parent.name) if path.parent.name.isdigit() else int(path.parent.name.rsplit('-', 1)[1])
            runner = runners.get(index, 'unknown')
            sums = defaultdict(float)
            for case in ET.parse(path).iter('testcase'):
                try:
                    filename, node = gate_junit_identity(case)
                except ValueError:
                    if any(prop.get('name') == 'gate_nodeid'
                           for prop in case.findall('properties/property')):
                        raise
                    # Older reports have definition locations for imported tests.
                    filename, node = case.get('file', ''), None
                if not filename.endswith('.py'):
                    continue
                elapsed = float(case.get('time', '0'))
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError('invalid JUnit timing')
                sums[filename] += elapsed
                if node is not None:
                    tests[node].append({'seconds': elapsed, 'run': metadata['id'],
                                        'attempt': metadata['run_attempt'], 'runner': runner, 'artifact': str(path.relative_to(directory))})
                    if case.find('failure') is not None or case.find('error') is not None:
                        failures[node].add(metadata['id'])
            for filename, seconds in sums.items():
                run_files[filename]['seconds'] += seconds
                run_files[filename]['runners'].add(runner)
                run_files[filename]['artifacts'].append(str(path.relative_to(directory)))
        for filename, row in run_files.items():
            runner = next(iter(row['runners'])) if len(row['runners']) == 1 else 'mixed'
            files[filename][runner].append(row['seconds'])
            file_tests[filename].append({'seconds': row['seconds'], 'run': metadata['id'],
                'attempt': metadata['run_attempt'], 'runner': runner,
                'artifact': ','.join(sorted(row['artifacts']))})
    slow = {}
    evidence = {}
    for node, values in sorted(tests.items()):
        largest = max(values, key=lambda value: value['seconds'])
        if largest['seconds'] > default and node.split('::', 1)[0] not in file_budgets:
            slow[node] = math.ceil(2 * largest['seconds'])
            evidence[node] = largest
    successful = sorted(run['wall_seconds'] for run in runs
                        if run['final_check'] == 'gate' and run['final_conclusion'] == 'success'
                        and run['wall_seconds'] is not None)
    budget = {'seconds': default, 'slow': slow, 'evidence': evidence,
              'hosted_runs': sorted({run['id'] for run in runs})}
    if file_budgets:
        if file_budgets - file_tests.keys():
            raise ValueError('file budget has no hosted timing evidence')
        file_evidence = {name: max(file_tests[name], key=lambda value: value['seconds'])
                         for name in sorted(file_budgets)}
        budget.update(file_seconds={name: math.ceil(2 * row['seconds'])
                                    for name, row in file_evidence.items()},
                      file_evidence=file_evidence)
    return {'schema': 'prismaquant.gate_hosted_evidence.v1', 'runs': runs,
        'full_gate_wall_seconds': {'samples': len(successful),
            'p50': statistics.median(successful) if successful else None,
            'p90': successful[math.ceil(.9 * len(successful)) - 1] if successful else None},
        'failures': {node: sorted(ids) for node, ids in sorted(failures.items())},
        'durations': {name: max(statistics.median(samples) for samples in values.values())
                      for name, values in sorted(files.items())},
        'node_durations': {node: max(statistics.median(
            row['seconds'] for row in values if row['runner'] == runner)
            for runner in {row['runner'] for row in values}) for node, values in sorted(tests.items())},
        'architecture_durations': {name: {runner: statistics.median(samples) for runner, samples in sorted(values.items())}
                                   for name, values in sorted(files.items())},
        'budget': budget}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['hosted'])
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--default-seconds', type=float, default=20)
    parser.add_argument('--file-budget', action='append', default=[],
                        help='test file with shared fixture cost; calibrate its complete JUnit total')
    args = parser.parse_args()
    result = hosted(args.inputs, args.default_seconds, file_budgets=args.file_budget)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
