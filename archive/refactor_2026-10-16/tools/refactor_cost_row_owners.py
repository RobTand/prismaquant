#!/usr/bin/env python3
"""Move native source contracts and cost interpretation out of candidate assembly."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess

import refactor_probe_statistics as migration

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'prismaquant.allocator_candidates'
PRICE_FUNCTIONS = frozenset({
    '_stats_indicates_packed_expert', '_has_measured_output_mse',
    '_has_interpolated_output_mse', '_prices_from_output_mse',
    'cost_entry_is_bit_exact', 'cost_entry_is_source_passthrough',
    'cost_entry_is_band_interpolated', 'cost_entry_is_exact_by_construction',
    'synthesized_source_passthrough_cost_entry', 'cost_entry_uses_measured_output_mse',
    'cost_entry_source', '_fisher_output_mse_allocator_enabled',
    'cost_entry_measured_activation_dloss', 'cost_entry_weight_only_dloss',
    'cost_entry_is_joint_aura', 'cost_entry_is_anchored_aura_supersurrogate',
    'cost_entry_activation_pricing_branch', '_format_act_quant_changes_input',
    '_activation_penalty', 'cost_entry_predicted_dloss', 'cost_entry_act_dloss',
    'cost_entry_prices_unmeasured_activation_at_zero', '_cost_ucb_z', '_resolve_cost_entry',
})
RENAMES = {'_resolve_cost_entry': 'resolve_cost_row'}
PRICE_CONSTANTS = frozenset({'SOURCE_PASSTHROUGH_COST_SOURCE', 'ANCHORED_AURA_COST_CURRENCY',
                           'ANCHORED_AURA_COST_SOURCE', 'ANCHORED_AURA_BRANCH',
                           'TESSERA_INTERPOLATED_COST_SOURCE', 'ACT_DLOSS_KEY'})


def name(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    target = node.target if isinstance(node, ast.AnnAssign) else (
        node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
    return target.id if isinstance(target, ast.Name) else None


def migrate_cost_row_imports(target, names):
    migration.ROOT = ROOT
    migration.TARGET = 'prismaquant.' + target
    migration.IMPORT_MOVES = {SOURCE: {value: RENAMES.get(value, value) for value in names}}
    for directory in ('prismaquant', 'tools', 'tests', 'experiments'):
        for path in (ROOT / directory).rglob('*.py'):
            if path.resolve() != Path(__file__).resolve():
                migration.migrate_statistics_imports(path)
                text = path.read_text()
                for value in names:
                    text = text.replace('allocator_candidates.' + value, target + '.' + RENAMES.get(value, value))
                path.write_text(text)
    # The original owner consumes the new module explicitly, without reexports.
    migration.qualify_statistics_owner(SOURCE)
    owner = ROOT / 'prismaquant/allocator_candidates.py'
    text = owner.read_text()
    for value in names:
        text = text.replace('probe_statistics.' + RENAMES.get(value, value), target + '.' + RENAMES.get(value, value))
    text = text.replace('from . import probe_statistics', 'from . import ' + target)
    owner.write_text(text)
    for directory in ('prismaquant', 'tools', 'tests', 'experiments'):
        for path in (ROOT / directory).rglob('*.py'):
            if path.resolve() == Path(__file__).resolve():
                continue
            migrate_module_aliases(path, target, names)


def migrate_module_aliases(path, target, names):
    text = path.read_text()
    tree = ast.parse(text)
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module in ('prismaquant', None) and (node.level or node.module == 'prismaquant'):
                aliases.update(alias.asname or alias.name for alias in node.names if alias.name == 'allocator_candidates')
        if isinstance(node, ast.Import):
            aliases.update(alias.asname for alias in node.names if alias.name == SOURCE and alias.asname)
    if not aliases:
        return
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    def replace(node, value=target):
        begin = offsets[node.lineno - 1] + len(lines[node.lineno - 1].encode()[:node.col_offset].decode())
        end = offsets[node.end_lineno - 1] + len(lines[node.end_lineno - 1].encode()[:node.end_col_offset].decode())
        edits.append((begin, end, value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases and node.attr in names:
            replace(node, target + '.' + RENAMES.get(node.attr, node.attr))
        if isinstance(node, ast.Call) and len(node.args) >= 2:
            subject, attribute = node.args[:2]
            if isinstance(subject, ast.Name) and subject.id in aliases and isinstance(attribute, ast.Constant) and attribute.value in names:
                replace(subject)
                if attribute.value in RENAMES:
                    replace(attribute, repr(RENAMES[attribute.value]))
    if not edits:
        return
    for begin, end, value in sorted(set(edits), reverse=True):
        text = text[:begin] + value + text[end:]
    tree = ast.parse(text)
    insertion = next((node.end_lineno for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == '__future__'),
                     tree.body[0].end_lineno if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str) else 0)
    lines = text.splitlines(keepends=True)
    lines.insert(insertion, 'from prismaquant import ' + target + '\n')
    path.write_text(''.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    original = subprocess.check_output(['git', 'show', args.base + ':prismaquant/allocator_candidates.py'],
                                       cwd=ROOT, text=True)
    tree = ast.parse(original)
    nodes = tree.body
    start = next(node.lineno for node in nodes if name(node) == 'ROUTE_DELEGATED_NATIVE')
    stop = next(node.end_lineno for node in nodes if name(node) == 'passthrough_serving_notes')
    source_nodes = [node for node in nodes if start <= node.lineno <= stop]
    source_names = {name(node) for node in source_nodes if name(node)}
    price_names = PRICE_FUNCTIONS | PRICE_CONSTANTS
    price_nodes = [node for node in nodes if name(node) in price_names]
    if args.write:
        owner = ROOT / 'prismaquant/allocator_candidates.py'
        source_target = ROOT / 'prismaquant/source_format_contracts.py'
        price_target = ROOT / 'prismaquant/priced_cell.py'
        if not source_target.exists():
            lines = original.splitlines(keepends=True)
            source_target.write_text('"""Native source storage, footprint, wire and measured serving contracts."""\n'
                'from __future__ import annotations\n\nfrom collections import Counter\n'
                'from dataclasses import dataclass\nfrom . import format_registry as fr\n'
                'from .footprint import plain_source_dtype_tensor_payload_breakdown\n\n\n'
                + ''.join(lines[start - 1:stop]))
            blocks = []
            for node in price_nodes:
                begin = min([node.lineno, *[d.lineno for d in getattr(node, 'decorator_list', [])]])
                block = ''.join(lines[begin - 1:node.end_lineno])
                if name(node) in RENAMES:
                    block = block.replace('def ' + name(node) + '(', 'def ' + RENAMES[name(node)] + '(', 1)
                blocks.append(block)
            price_target.write_text('"""Cost-row evidence, objective and activation-price interpretation."""\n'
                'from __future__ import annotations\n\nimport math\nimport operator\nimport os\n'
                'from . import format_registry as fr\nfrom .source_format_contracts import SOURCE_PASSTHROUGH_FORMATS\n'
                'from .allocator_solver import predicted_dloss\n'
                'from .schemas import refuse_retired_ladder_cost_source\n'
                'from .activation_fair_pricing import (APPLIED_MARKER_KEY, ActivationFairPricing,\n'
                '    BRANCH_ACTIVATION_IDENTITY, BRANCH_BIT_EXACT, BRANCH_CALIBRATED,\n'
                '    BRANCH_INTERPOLATED_OUTPUT, BRANCH_MEASURED, BRANCH_SOURCE_PASSTHROUGH, BRANCH_UNCALIBRATED)\n\n\n'
                + '\n\n'.join(block.rstrip() for block in blocks) + '\n')
            spans = [(start, stop)] + [(min([node.lineno, *[d.lineno for d in getattr(node, 'decorator_list', [])]]), node.end_lineno) for node in price_nodes]
            current = owner.read_text().splitlines(keepends=True)
            for begin, end in sorted(spans, reverse=True):
                del current[begin - 1:end]
            remaining = ''.join(current)
            for target, names in [('source_format_contracts', source_names), ('priced_cell', price_names)]:
                imports = ', '.join(RENAMES.get(value, value) + (' as ' + value if value in RENAMES else '') for value in sorted(names))
                remaining = remaining.replace('from __future__ import annotations\n',
                    'from __future__ import annotations\nfrom prismaquant.' + target + ' import ' + imports + '\n', 1)
            owner.write_text(remaining)
        migrate_cost_row_imports('source_format_contracts', source_names)
        migrate_cost_row_imports('priced_cell', price_names)
    else:
        actual = {}
        for target in ('source_format_contracts', 'priced_cell'):
            actual.update({name(node): node for node in ast.parse((ROOT / 'prismaquant' / (target + '.py')).read_text()).body if name(node)})
        for node in [*source_nodes, *price_nodes]:
            key = name(node)
            if key in RENAMES:
                node.name = RENAMES[key]
            if key is not None and ast.dump(node, include_attributes=False) != ast.dump(actual[RENAMES.get(key, key)], include_attributes=False):
                raise SystemExit('changed cost-row definition: ' + key)
        expected_guards = [node for node in source_nodes if isinstance(node, ast.If)]
        actual_guards = [node for node in ast.parse((ROOT / 'prismaquant/source_format_contracts.py').read_text()).body if isinstance(node, ast.If)]
        if ast.dump(ast.Module(body=expected_guards, type_ignores=[]), include_attributes=False) != ast.dump(ast.Module(body=actual_guards, type_ignores=[]), include_attributes=False):
            raise SystemExit('changed source contract initialization guard')
        print(f'{len(source_names) + len(price_names)} source/price definitions and source initialization guards retain their original bodies.')


if __name__ == '__main__':
    main()
