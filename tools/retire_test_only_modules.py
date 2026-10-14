#!/usr/bin/env python3
"""Retire named modules only when no runtime import or CLI reaches them."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETIREMENTS = (
    'proposal_validation', 'tessera_full_domain_ledger', 'speed_quality_frontier',
    'shard_layout', 'shard_commit_progress', 'tessera_reduced_schedule_repair',
    'sensitivity_coverage',
)


def retirement_references(path, names):
    source = path.read_text()
    if not any(name in source for name in names):
        return []
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            imports = [module, *(module + '.' + alias.name for alias in node.names)]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            imports = [f'prismaquant.{name}' for name in names
                       if f'prismaquant.{name}' in node.value]
        else:
            imports = []
        hits.extend((node.lineno, name) for name in names
                    if name in imports or f'prismaquant.{name}' in imports)
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    removed = {ROOT / 'prismaquant' / f'{name}.py' for name in RETIREMENTS}
    removed.update(ROOT / 'tests' / f'test_{name}.py' for name in RETIREMENTS)
    blockers = []
    for folder in ('prismaquant', 'tools', 'tests', 'experiments'):
        for path in (ROOT / folder).rglob('*.py'):
            if path in removed or path.resolve() == Path(__file__).resolve():
                continue
            blockers.extend(f'{path.relative_to(ROOT)}:{line}: {name}'
                            for line, name in retirement_references(path, RETIREMENTS))
    for name in RETIREMENTS:
        path = ROOT / 'prismaquant' / f'{name}.py'
        if path.exists() and any(isinstance(node, ast.If)
                                and '__main__' in ast.unparse(node.test)
                                for node in ast.parse(path.read_text()).body):
            blockers.append(f'{path.relative_to(ROOT)}: CLI entry point')
    if blockers:
        raise SystemExit('Retirement has callers:\n' + '\n'.join(blockers))
    for path in sorted(removed):
        if path.exists():
            print(path.relative_to(ROOT))
            if args.write:
                path.unlink()


if __name__ == '__main__':
    main()
