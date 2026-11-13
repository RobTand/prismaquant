#!/usr/bin/env python3
"""Fail CI when a derived Bazel input is stale, naming the files and the one fix."""
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
FIX = 'tools/bazel/generate.sh'


def commands():
    """Every derived-input check. The hosted gate lane runs them on each head tree."""
    return [
        [sys.executable, 'tools/bazel/requirements.py', '--check'],
        [sys.executable, 'tools/bazel/build_files.py', '--check'],
    ]


def check_generated():
    failures = []
    for command in commands():
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            failures.append(f'$ {shlex.join(map(str, command))}\n{result.stdout}{result.stderr}')
    if failures:
        raise RuntimeError('Generated Bazel inputs are stale:\n' + '\n'.join(failures)
                           + f'\nFix them all with: {FIX}')


if __name__ == '__main__':
    try:
        check_generated()
    except RuntimeError as error:
        sys.exit(str(error))
