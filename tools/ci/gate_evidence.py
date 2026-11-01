#!/usr/bin/env python3
"""Summarize PrismaBuild inventories and downloaded hosted gate artifacts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(reports):
    outcomes = defaultdict(dict)
    timings = defaultdict(list)
    actions = []
    incomplete = []
    for repeat, path in enumerate(reports, 1):
        for row in json.loads(path.read_text()):
            key = row['action_key']
            terminal = next((candidate for state in ('done', 'failed')
                if (candidate := Path('/mnt/shared/prismabuild-fleet/pb-queue') / state / (key + '.json')).is_file()), None)
            record = json.loads(terminal.read_text()) if terminal else {}
            receipt = Path(row['receipt_path']) if row.get('receipt_path') else None
            actions.append({'repeat': repeat, 'key': key, 'exit': row['returncode'],
                'summary': row.get('summary'), 'terminal': str(terminal) if terminal else None,
                'host': record.get('finished_host'), 'receipt': str(receipt) if receipt else None,
                'receipt_exists': bool(receipt and receipt.is_file()),
                'memory_peak_bytes': record.get('detail', {}).get('resource_telemetry', {}).get('memory_peak_bytes')})
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
        'reports': [{'path': str(path), 'sha256': digest(path)} for path in reports],
        'actions': actions, 'incomplete': incomplete,
        'always_failing': sorted(node for node, values in outcomes.items()
                                 if len(values) == len(reports) and set(values.values()) == {'fail'}),
        'changed_outcomes': {node: values for node, values in sorted(outcomes.items())
                             if len(set(values.values())) > 1},
        'files': {name: max(values) for name, values in sorted(timings.items())},
        'observed_nodes': len(outcomes)}


def hosted(directories, default):
    tests = defaultdict(list)
    files = defaultdict(list)
    runs = []
    failures = defaultdict(set)
    for directory in directories:
        metadata = json.loads((directory / 'run.json').read_text())
        jobs = json.loads((directory / 'jobs.json').read_text())['jobs']
        runs.append({'id': metadata['id'], 'attempt': metadata['run_attempt'],
                     'head_sha': metadata['head_sha'], 'conclusion': metadata['conclusion'],
                     'jobs': [{key: job.get(key) for key in
                         ('name', 'started_at', 'completed_at', 'conclusion', 'labels', 'steps')}
                         for job in jobs]})
        # The final artifact exports only effective attempts. Do not count retries twice.
        effective = list(directory.glob('gate-*-final-*/effective/*/junit.xml'))
        if not effective:
            effective = list(directory.glob('gate-result-*-a1-*/junit.xml'))
        for path in effective:
            sums = defaultdict(float)
            for case in ET.parse(path).iter('testcase'):
                filename = case.get('file', '')
                if not filename.endswith('.py'):
                    continue
                elapsed = float(case.get('time', '0'))
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError('invalid JUnit timing')
                sums[filename] += elapsed
                classname = case.get('classname', '')
                module = filename[:-3].replace('/', '.')
                if classname == module or classname.startswith(module + '.'):
                    classes = classname[len(module):].lstrip('.').replace('.', '::')
                    node = filename + '::' + (classes + '::' if classes else '') + case.get('name', '')
                    tests[node].append({'seconds': elapsed, 'run': metadata['id'],
                                        'attempt': metadata['run_attempt'], 'artifact': str(path.relative_to(directory))})
                    if case.find('failure') is not None or case.find('error') is not None:
                        failures[node].add(metadata['id'])
            for filename, seconds in sums.items():
                files[filename].append(seconds)
    slow = {}
    evidence = {}
    for node, values in sorted(tests.items()):
        largest = max(values, key=lambda value: value['seconds'])
        if largest['seconds'] > default:
            slow[node] = math.ceil(2 * largest['seconds'])
            evidence[node] = largest
    return {'schema': 'prismaquant.gate_hosted_evidence.v1', 'runs': runs,
        'failures': {node: sorted(ids) for node, ids in sorted(failures.items())},
        'durations': {name: statistics.median(values) for name, values in sorted(files.items())},
        'budget': {'seconds': default, 'slow': slow, 'evidence': evidence,
                   'hosted_runs': sorted({run['id'] for run in runs})}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['inventory', 'hosted'])
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--default-seconds', type=float, default=20)
    args = parser.parse_args()
    result = inventory(args.inputs) if args.kind == 'inventory' else hosted(args.inputs, args.default_seconds)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
