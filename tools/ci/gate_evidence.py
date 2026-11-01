#!/usr/bin/env python3
"""Summarize PrismaBuild inventories and downloaded hosted gate artifacts."""
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


def inventory(reports, *, action_evidence=None):
    """Reduce pbtest reports with optional exported public pb_action responses."""
    action_evidence = {} if action_evidence is None else action_evidence
    outcomes = defaultdict(dict)
    timings = defaultdict(list)
    actions = []
    incomplete = []
    for repeat, path in enumerate(reports, 1):
        for row in json.loads(path.read_text()):
            key = row['action_key']
            record = action_evidence.get(key, {})
            receipt = Path(row['receipt_path']) if row.get('receipt_path') else None
            complete = record.get('complete') is True and not record.get('timed_out') and not record.get('unavailable')
            actions.append({'repeat': repeat, 'key': key, 'exit': row['returncode'],
                'summary': row.get('summary'), 'terminal_state': record.get('state') if complete else None,
                'evidence_complete': complete,
                'host': record.get('outcome', {}).get('finished_host') if complete else None,
                'receipt': str(receipt) if receipt else None,
                'receipt_exists': bool(receipt and receipt.is_file()),
                'memory_peak_bytes': record.get('resource_profile', {}).get('scope', {}).get('memory_peak_bytes')
                    if complete and record.get('resource_profile') else None})
            parsed = None
            for line in row.get('output', '').splitlines():
                if line.startswith('pbtest-outcomes: '):
                    parsed = json.loads(line.removeprefix('pbtest-outcomes: '))
            if parsed is None:
                incomplete.append({'repeat': repeat, 'key': key, 'reason': 'missing outcome record'})
                continue
            for filename, seconds in parsed.get('file_durations', {}).items():
                timings[filename].append(seconds)
            nodes = defaultdict(list)
            for report in parsed.get('reports', []):
                nodes[report[0]].append(report[1:3])
            for node, phases in nodes.items():
                labels = {outcome for phase, outcome in phases}
                outcome = 'fail' if labels & {'failed', 'error'} else 'skip' if 'skipped' in labels else 'pass'
                outcomes[node][repeat] = outcome
            missing = row.get('verdict', {}).get('unobserved_files', [])
            if missing:
                incomplete.append({'repeat': repeat, 'key': key, 'files': missing})
    return {'schema': 'prismaquant.gate_inventory.v1',
        'reports': [{'path': str(path), 'sha256': file_sha256hex(path)} for path in reports],
        'actions': actions, 'incomplete': incomplete,
        'always_failing': sorted(node for node, values in outcomes.items()
                                 if len(values) == len(reports) and set(values.values()) == {'fail'}),
        'changed_outcomes': {node: values for node, values in sorted(outcomes.items())
                             if len(set(values.values())) > 1},
        'files': {name: max(values) for name, values in sorted(timings.items())},
        'observed_nodes': len(outcomes)}


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
        final = next((job for job in jobs if job['name'] == 'gate' and job.get('conclusion') != 'skipped'), None)
        if final is None:
            final = next((job for job in jobs if job['name'] == 'pr-gate' and job.get('conclusion') != 'skipped'), {})
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
                files[filename][runner].append(seconds)
                file_tests[filename].append({'seconds': seconds, 'run': metadata['id'],
                    'attempt': metadata['run_attempt'], 'runner': runner,
                    'artifact': str(path.relative_to(directory))})
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
        'architecture_durations': {name: {runner: statistics.median(samples) for runner, samples in sorted(values.items())}
                                   for name, values in sorted(files.items())},
        'budget': budget}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['inventory', 'hosted'])
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--default-seconds', type=float, default=20)
    parser.add_argument('--file-budget', action='append', default=[],
                        help='test file with shared fixture cost; calibrate its complete JUnit total')
    parser.add_argument('--action-evidence', type=Path, help='mapping of action keys to exported public pb_action responses')
    args = parser.parse_args()
    actions = json.loads(args.action_evidence.read_text()) if args.action_evidence else None
    result = inventory(args.inputs, action_evidence=actions) if args.kind == 'inventory' else hosted(
        args.inputs, args.default_seconds, file_budgets=args.file_budget)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
