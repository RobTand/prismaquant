#!/usr/bin/env python3
"""Fail CI when a derived Bazel input is stale, naming the files and the one fix."""
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
FIX = 'tools/bazel/generate.sh'


def commands():
    """Every derived-input check. The gate runs these on each head tree."""
    return [
        [sys.executable, 'tools/bazel/requirements.py', '--check'],
        [sys.executable, 'tools/bazel/build_files.py', '--check'],
    ]


def check(checks=None, root=ROOT):
    failures = []
    for command in commands() if checks is None else checks:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        if result.returncode:
            failures.append(f'$ {shlex.join(map(str, command))}\n{result.stdout}{result.stderr}')
    if failures:
        raise RuntimeError('Generated Bazel inputs are stale:\n' + '\n'.join(failures)
                           + f'\nFix them all with: {FIX}')


if __name__ == '__main__':
    try:
        check()
    except RuntimeError as error:
        sys.exit(str(error))
