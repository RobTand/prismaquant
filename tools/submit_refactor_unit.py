#!/usr/bin/env python3
"""Validate source paths, freeze a checkout, and submit its checks to PrismaBuild."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from freeze_refactor_checkout import freeze_refactor_checkout

PBRUN = Path('/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cwd', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--compile', nargs='+', required=True, dest='modules')
    parser.add_argument('--tests', nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--memory-gb', type=int, default=16)
    parser.add_argument('--timeout-s', type=int, default=1800)
    parser.add_argument('--env', action='append', default=[])
    args = parser.parse_args()
    root = args.cwd.resolve()
    for value in [*args.modules, *args.tests, 'tools/verify_refactor_unit.py']:
        path = (root / value.split('::', 1)[0]).resolve()
        if not path.is_relative_to(root) or not path.exists():
            parser.error(f'check source path does not exist inside the checkout: {value}')
    frozen = freeze_refactor_checkout(root, args.output)
    command = ['python3', str(PBRUN), '--cwd', frozen['checkout'],
               '--cpus', str(args.workers), '--demand', f'mem_gb={args.memory_gb}',
               '--timeout-s', str(args.timeout_s), '--detach']
    for value in ['OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', *args.env]:
        command.extend(['--env', value])
    command.extend(['--', args.python, 'tools/verify_refactor_unit.py',
                    '--compile', *args.modules, '--tests', *args.tests,
                    '--workers', str(args.workers)])
    result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
    receipt = json.loads(result.stdout)
    receipt['frozen_checkout'] = frozen
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
