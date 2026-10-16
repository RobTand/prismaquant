#!/usr/bin/env python3
"""Move the existing corpus sampler into the calibration artifact owner."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess

import refactor_probe_statistics as migration

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    original = subprocess.check_output(
        ['git', 'show', f'{args.base}:prismaquant/sensitivity_probe.py'], cwd=ROOT, text=True)
    original_node = next(node for node in ast.parse(original).body
                         if isinstance(node, ast.FunctionDef) and node.name == 'load_calibration')
    target = ROOT / 'prismaquant/calibration_data.py'
    if args.write:
        if not any(isinstance(node, ast.FunctionDef) and node.name == 'load_calibration'
                   for node in ast.parse(target.read_text()).body):
            source_path = ROOT / 'prismaquant/sensitivity_probe.py'
            source = source_path.read_text(); lines = source.splitlines(keepends=True)
            node = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'load_calibration')
            block = ''.join(lines[node.lineno-1:node.end_lineno])
            del lines[node.lineno-1:node.end_lineno]
            source_path.write_text(''.join(lines))
            target.write_text(target.read_text().replace('import re\n', 'import re\nimport random\n',1)
                              + '\n\n' + block)
        migration.ROOT = ROOT
        migration.TARGET = 'prismaquant.calibration_data'
        migration.IMPORT_MOVES = {'prismaquant.sensitivity_probe': {'load_calibration':'load_calibration'}}
        for folder in ('prismaquant','tools','tests','experiments'):
            for path in (ROOT/folder).rglob('*.py'):
                if path.resolve() != Path(__file__).resolve() and migration.migrate_statistics_imports(path):
                    print(path.relative_to(ROOT))
    else:
        actual = next(node for node in ast.parse(target.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == 'load_calibration')
        if ast.dump(original_node, include_attributes=False) != ast.dump(actual, include_attributes=False):
            raise SystemExit('Corpus sampler body changed')
        print('Corpus sampler retains its original body.')


if __name__ == '__main__':
    main()
