#!/usr/bin/env python3
"""Move packed-expert inventory queries to their shared owner."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import subprocess

from refactor_probe_statistics import migrate_statistics_imports
import refactor_probe_statistics as migration

ROOT = Path(__file__).resolve().parents[1]
NAMES = {name: name.lstrip('_') for name in (
    '_packed_expert_param_name_set', '_packed_expert_parent_for_projection',
    '_is_packed_experts_module', '_packed_experts_param_names',
    '_packed_expert_projection_candidate_names')}


def inventory_symbol_names(source):
    return re.sub(r'\b(' + '|'.join(NAMES) + r')\b',
                  lambda match: NAMES[match.group()], source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    original = subprocess.check_output(
        ['git', 'show', f'{args.base}:prismaquant/sensitivity_probe.py'],
        cwd=ROOT, text=True)
    source_path = ROOT / 'prismaquant/sensitivity_probe.py'
    target = ROOT / 'prismaquant/probe_inventory.py'
    nodes = [node for node in ast.parse(original).body
             if isinstance(node, ast.FunctionDef) and node.name in NAMES]
    if args.write:
        if not target.exists():
            lines = original.splitlines(keepends=True)
            blocks = [inventory_symbol_names(''.join(lines[node.lineno-1:node.end_lineno]))
                      for node in nodes]
            target.write_text('"""Profile-declared probe unit recognition and inventory policy."""\n'
                              'from __future__ import annotations\n\nimport torch.nn as nn\n\n\n'
                              + '\n\n'.join(block.rstrip() for block in blocks) + '\n')
            source = source_path.read_text(); lines = source.splitlines(keepends=True)
            for node in reversed([node for node in ast.parse(source).body
                                  if isinstance(node, ast.FunctionDef) and node.name in NAMES]):
                del lines[node.lineno-1:node.end_lineno]
            source_path.write_text(''.join(lines))
        migration.ROOT = ROOT
        migration.TARGET = 'prismaquant.probe_inventory'
        migration.IMPORT_MOVES = {'prismaquant.sensitivity_probe': NAMES}
        for folder in ('prismaquant', 'tools', 'tests', 'experiments'):
            for path in (ROOT / folder).rglob('*.py'):
                if path.resolve() == Path(__file__).resolve():
                    continue
                if migrate_statistics_imports(path):
                    print(path.relative_to(ROOT))
        source = source_path.read_text()
        source = re.sub(r'\b(' + '|'.join(NAMES) + r')\b',
                        lambda match: 'probe_inventory.' + NAMES[match.group()], source)
        source = source.replace('from . import probe_inventory\n', '')
        source = source.replace('from __future__ import annotations\n',
                                'from __future__ import annotations\n\nfrom . import probe_inventory\n',1)
        source_path.write_text(source)
    else:
        actual = {node.name: node for node in ast.parse(target.read_text()).body
                  if isinstance(node, ast.FunctionDef)}
        for node in nodes:
            expected = ast.parse(inventory_symbol_names(ast.unparse(node))).body[0]
            if ast.dump(expected, include_attributes=False) != ast.dump(
                    actual[NAMES[node.name]], include_attributes=False):
                raise SystemExit(f'changed inventory definition: {node.name}')
        print('All five packed-expert inventory definitions retain their original bodies.')


if __name__ == '__main__':
    main()
