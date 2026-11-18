#!/usr/bin/env python3
"""Record file times from amd64 Bazel post-merge logs. Refuse a record in which a test did not pass."""

import argparse
import json
from pathlib import Path
import re

ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
RESULT = re.compile(r'^(//tests:\S+)[ \t]+(.+?)[ \t]*$'
                    r'(?:\n[ \t]+Stats over (\d+) runs: max = [\d.]+s, min = [\d.]+s, avg = ([\d.]+)s, dev = [\d.]+s)?', re.M)


def record_durations(table, logs):
    files = {}
    for log in logs:
        for label, result, runs, average in RESULT.findall(ANSI.sub('', log.read_text())):
            if not (passed := re.fullmatch(r'PASSED in ([\d.]+)s', result)):
                raise SystemExit(f'{log}: {label} {result}; record only a record in which every test passed')
            path = label[2:].replace(':', '/') + '.py'
            files[path] = max(files.get(path, 0), round(float(average) * int(runs) if runs else float(passed[1]), 3))
    data = json.loads(table.read_text())
    data['files'] = dict(sorted(files.items()))
    invocations = ', '.join(sorted(log.stem for log in logs))
    data['source'] = (f'Bazel amd64 post-merge record invocations {invocations}. '
                      'Sum executor seconds across test shards. Keep the largest total for each file.')
    table.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', type=Path, nargs='+', metavar='LOG')
    record_durations(Path(__file__).with_name('gate_durations.json'), parser.parse_args().logs)
