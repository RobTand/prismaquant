#!/usr/bin/env python3
"""Prove Bazel runs exactly the hosted gate's population.

static compares the tests `bazel query` finds with the population
tools/ci/gate_select.py plans in full mode for the checked-out tree: the same
files run (on Bazel, or in the hosted lane for targets tagged
gate-lane-hosted), every other test file has a manual target, and each file
keeps its node-level deselects. It runs on every Bazel gate.

receipts compares the nodes Bazel's runner recorded (test.outputs/
inventory.json) with the JUnit reports of a hosted gate run on the same commit.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
FIX = 'tools/bazel/generate.sh'
RUNNER = '//tools/bazel:python_runner.py'
HOSTED = 'gate-lane-hosted'


def _module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def population(root=ROOT):
    """The full-mode plan. Names are all it needs, so no test is parsed."""
    gate_select = _module_at('gate_select_for_parity', root / 'tools/ci/gate_select.py')
    tests = {path.relative_to(root).as_posix(): (set(), set()) for path in sorted((root / 'tests').rglob('*.py'))}
    read = lambda name: json.loads((root / 'tools/ci' / name).read_text())  # noqa: E731
    return gate_select.create_plan(root, [], durations=read('gate_durations.json'),
                                   quarantine=read('gate_quarantine.json'), index=tests, mode='full')


def bazel_tests(query_lines):
    """{test file: [(target, manual, deselects)]} from `bazel query --output=streamed_jsonproto`."""
    found = {}
    for line in query_lines:
        rule = json.loads(line).get('rule')
        if not rule:
            continue
        attributes = {attribute['name']: attribute for attribute in rule.get('attribute', [])}
        if attributes.get('main', {}).get('stringValue') != RUNNER:
            continue
        # Bazel tokenizes args like a shell; the generator quotes node ids that need it.
        args = [token for arg in attributes.get('args', {}).get('stringListValue', []) for token in shlex.split(arg)]
        tags = attributes.get('tags', {}).get('stringListValue', [])
        deselects = sorted(arg.removeprefix('--deselect=') for arg in args[1:] if arg.startswith('--deselect='))
        # The hosted lane runs gate-lane-hosted targets; they count as run.
        manual = 'manual' in tags and HOSTED not in tags
        found.setdefault(args[0], []).append((rule['name'], manual, deselects))
    return found


def compare_static(plan, bazel):
    expected = {row['path'] for row in plan['selected']}
    deferred = {row['path'] for row in plan['deferred'] if '::' not in row['path']}
    problems = []
    for path, targets in sorted(bazel.items()):
        if len(targets) > 1:
            problems.append(f'two Bazel targets run {path}: ' + ', '.join(name for name, _, _ in targets))
    running = {path for path, targets in bazel.items() if any(not manual for _, manual, _ in targets)}
    problems += [f'Bazel does not run {path}.' for path in sorted(expected) if path not in running]
    problems += [f'Bazel runs {path}, which the hosted gate does not.'
                 for path in sorted(running) if path not in expected]
    problems += [f'{path} has no Bazel target; it must exist, tagged manual.'
                 for path in sorted(deferred) if path not in bazel]
    for path in sorted(path for path in expected if path in running):
        wanted = sorted(node for node in plan['deselect'] if node.split('::', 1)[0] == path)
        if bazel[path][0][2] != wanted:
            problems.append(f'{path} deselects {bazel[path][0][2]}; the hosted gate deselects {wanted}.')
    if not expected:
        problems.append('The hosted gate population is empty; refusing an empty inventory.')
    if problems:
        raise RuntimeError('Test parity drift:\n' + '\n'.join(problems) + f'\nFix: {FIX}')
    manual = sum(1 for path in bazel if path not in running)
    return {'population': len(expected), 'manual': manual, 'deselected_nodes': len(plan['deselect'])}


def junit_nodes(report):
    """Node ID to passed, skipped or failed, as the Bazel runner reads xunit1 reports."""
    runner = _module_at('python_runner_for_parity', ROOT / 'tools/bazel/python_runner.py')
    return runner.outcomes(report)


def hosted_nodes(reports):
    nodes = {}
    for report in reports:
        nodes.update(junit_nodes(report))
    return nodes


def read_receipt(outputs):
    """A test's inventory.json from its test.outputs directory, zipped or not."""
    plain, archive = outputs / 'inventory.json', outputs / 'outputs.zip'
    if plain.is_file():
        return json.loads(plain.read_text())
    if archive.is_file():
        with zipfile.ZipFile(archive) as zipped:
            if 'inventory.json' in zipped.namelist():
                return json.loads(zipped.read('inventory.json'))
    return None


def bazel_nodes(testlogs):
    nodes = {}
    for outputs in sorted(Path(testlogs).rglob('test.outputs')):
        receipt = read_receipt(outputs)
        if receipt:
            nodes.update(receipt['nodes'])
    return nodes


def compare_receipts(expected, actual):
    problems = [f'missing from Bazel: {node}' for node in sorted(expected) if node not in actual]
    problems += [f'not run by the hosted gate: {node}' for node in sorted(actual) if node not in expected]
    problems += [f'{node}: hosted {expected[node]}, Bazel {actual[node]}'
                 for node in sorted(expected) if node in actual and expected[node] != actual[node]]
    if problems:
        raise RuntimeError('Node parity drift:\n' + '\n'.join(problems))
    if not expected:
        raise RuntimeError('empty node inventory')
    return {'nodes': len(expected), 'files': len({node.split('::', 1)[0] for node in expected})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    static = commands.add_parser('static')
    static.add_argument('--bazel', default='bazel', help='bazel command, split on spaces')
    receipts = commands.add_parser('receipts')
    receipts.add_argument('--hosted-junit', type=Path, nargs='+', required=True)
    receipts.add_argument('--bazel-testlogs', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'static':
            lines = subprocess.check_output([*args.bazel.split(), 'query', 'tests(//...)',
                                             '--output=streamed_jsonproto'], cwd=ROOT, text=True).splitlines()
            summary = compare_static(population(), bazel_tests(lines))
        else:
            summary = compare_receipts(hosted_nodes(args.hosted_junit), bazel_nodes(args.bazel_testlogs))
    except (RuntimeError, OSError, ET.ParseError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    sys.exit(main())
