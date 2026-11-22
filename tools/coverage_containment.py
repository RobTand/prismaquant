#!/usr/bin/env python3
"""Per-test covered lines of prismaquant, and which tests contain a test's lines (PQ #2806).

  coverage_containment.py build DATA_FILE RANGES [--outcomes TSV]
  coverage_containment.py query RANGES TEST... [--lines]

build reads a coverage.py database from pytest --cov=prismaquant --cov-context=test and writes RANGES, a gzip TSV.
A test's lines are the union of its contexts "<nodeid>|setup", "|run" and "|teardown", so lines that a module- or
session-scope fixture runs count for the first test that uses the fixture. --outcomes names a TSV of
"<nodeid>\t<outcome>" rows; without it, each test in DATA_FILE counts as passed. RANGES has one
"<nodeid>\t<file>\t<line ranges such as 3-5,9>" row for each test and file it covers, and one
"<nodeid>\tNOT MEASURED\t<outcome>" row for each test that did not pass or covered no line. Coverage does not
measure child processes, so a test whose work runs in a child covers no line. Such a test is never in a
containment answer.

query prints, for each named test (a node id, or a path that names all its tests): its covered-line count, its lines
with --lines, and each measured test that covers all of its lines.
"""
import argparse
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import coverage

PHASES = {'setup', 'run', 'teardown'}
NOT_MEASURED = 'NOT MEASURED'


def runs(lines):
    """Sorted line numbers as maximal (first, last) runs of consecutive numbers."""
    spans = []
    for line in lines:
        if spans and spans[-1][1] == line - 1:
            spans[-1][1] = line
        else:
            spans.append([line, line])
    return [tuple(span) for span in spans]


def text(spans):
    return ','.join(f'{first}' if first == last else f'{first}-{last}' for first, last in spans)


def parse_spans(spans):
    return [(int(first), int(last or first)) for first, _, last in (span.partition('-') for span in spans.split(','))]


def build_line_ranges(data_file, out, outcomes_file=None):
    data = coverage.CoverageData(basename=str(data_file))
    data.read()
    covered = defaultdict(lambda: defaultdict(list))  # test -> file -> lines
    for path in data.measured_files():
        for line, contexts in data.contexts_by_lineno(path).items():
            for test in {test for test, _, phase in (context.rpartition('|') for context in contexts)
                         if test and phase in PHASES}:
                covered[test][path].append(line)
    outcomes = dict.fromkeys(covered, 'passed')
    if outcomes_file:
        outcomes |= dict(csv.reader(Path(outcomes_file).open(), delimiter='\t'))
    with gzip.open(out, 'wt', compresslevel=9) as rows:
        for test, outcome in sorted(outcomes.items()):
            if outcome != 'passed' or test not in covered:
                rows.write(f'{test}\t{NOT_MEASURED}\t{outcome}\n')
                continue
            for path, lines in sorted(covered[test].items()):
                rows.write(f'{test}\t{path}\t{text(runs(sorted(lines)))}\n')


def load_line_ranges(ranges):
    """{test: {file: [(first, last), ...]}} for measured tests, and {test: outcome} for the others."""
    measured, unmeasured = defaultdict(dict), {}
    with gzip.open(ranges, 'rt') as rows:
        for test, path, spans in csv.reader(rows, delimiter='\t'):
            if path == NOT_MEASURED:
                unmeasured[test] = spans
            else:
                measured[test][path] = parse_spans(spans)
    return measured, unmeasured


def count(files):
    return sum(last - first + 1 for spans in files.values() for first, last in spans)


def spans_within(small, large):
    """True when each run of small lies in one run of large (both sorted and maximal)."""
    position = 0
    for first, last in small:
        while position < len(large) and large[position][1] < first:
            position += 1
        if position == len(large) or large[position][0] > first or large[position][1] < last:
            return False
    return True


def query(ranges, names, show_lines=False, out=sys.stdout):
    measured, unmeasured = load_line_ranges(ranges)
    by_file = defaultdict(set)
    for test, files in measured.items():
        for path in files:
            by_file[path].add(test)
    print(f'not-measured-count\t{len(unmeasured)}', file=out)
    known = sorted({*measured, *unmeasured})
    for name in names:
        tests = [test for test in known if test == name or test.startswith(name + '::')]
        if not tests:
            print(f'unknown\t{name}', file=out)
        for test in tests:
            if test in unmeasured:
                print(f'not-measured\t{test}\t{unmeasured[test]}', file=out)
                continue
            files = measured[test]
            print(f'lines\t{test}\t{count(files)}', file=out)
            if show_lines:
                for path, spans in sorted(files.items()):
                    print(f'range\t{test}\t{path}\t{text(spans)}', file=out)
            candidates = set.intersection(*(by_file[path] for path in sorted(files, key=lambda p: len(by_file[p]))))
            for other in sorted(candidates - {test}):
                if all(spans_within(spans, measured[other][path]) for path, spans in files.items()):
                    print(f'contained-by\t{test}\t{other}\t{count(measured[other])}', file=out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    builder = commands.add_parser('build')
    builder.add_argument('data_file', type=Path)
    builder.add_argument('ranges', type=Path)
    builder.add_argument('--outcomes', type=Path)
    asker = commands.add_parser('query')
    asker.add_argument('ranges', type=Path)
    asker.add_argument('tests', nargs='+')
    asker.add_argument('--lines', action='store_true')
    arguments = parser.parse_args()
    if arguments.command == 'build':
        build_line_ranges(arguments.data_file, arguments.ranges, arguments.outcomes)
    else:
        query(arguments.ranges, arguments.tests, arguments.lines)
