#!/usr/bin/env python3
"""Rerun the scalar joint-row ownership move and its direct caller migration."""
from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess

ROW_NODES = {
    'JOINT_CURRENCY', 'JOINT_AURA_COST_CURRENCY', 'JOINT_AURA_COST_SOURCE',
    'PROBE_UNCERTAINTY_SCOPE', 'SEQUENCE_ATTRIBUTION_SCHEMA',
    'SEQUENCE_ATTRIBUTION_GATE_METHOD', 'SEQUENCE_LEAVEOUT_SCOPE',
    '_ROW_UNCERTAINTY_SCOPES', '_ValidatedProbeIdentity',
    'prepare_joint_aura_identities', 'validated_identity_memo',
    'release_joint_aura_identities', 'validated_probe_identity', 'identity_sha256',
    'validate_joint_aura_entry', '_validate_sequence_attribution', 'squared_signed',
    '_attribution_reconciliation_residuals', '_attribution_reconciliation_norms',
    '_attribution_value', '_leaveout_value', 'sequence_attribution_sidecar',
    'sequence_cohort_summary', 'make_joint_aura_entry', '_require_probe_alignment',
}
BACKEND_NODES = {'SCHEMA', 'FUSED_NAME', 'QUALIFICATION_PATH', 'REFERENCE_IDENTITY',
                 '_qualification', 'validate_projection_backend_identity'}
MOVES = {
    'prismaquant.joint_aura': ('joint_aura_rows', ROW_NODES),
    'prismaquant.priced_cell': ('joint_aura_rows', {'cost_entry_is_joint_aura'}),
    'prismaquant.allocator_candidates': ('joint_aura_rows',
        {'JointCellCoordinateError', 'joint_row_binds_cell'}),
    'prismaquant.joint_projection_backend': ('joint_projection_contract', BACKEND_NODES),
}
RENAMES = {'_require_probe_alignment': 'require_probe_alignment',
           '_qualification': 'projection_qualification',
           'identity_sha256': 'joint_identity_sha256'}


def _row_migration_name(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    target = node.target if isinstance(node, ast.AnnAssign) else (
        node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
    return target.id if isinstance(target, ast.Name) else None


def _row_migration_span(text, node):
    lines = text.split('\n')
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)
    return (offsets[node.lineno - 1] + len(lines[node.lineno - 1].encode()[:node.col_offset].decode()),
            offsets[node.end_lineno - 1] + len(lines[node.end_lineno - 1].encode()[:node.end_col_offset].decode()))


def _row_migration_apply(text, edits):
    for start, end, replacement in sorted(set(edits), reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def _row_migration_lf_lines(text):
    lines = text.split('\n')
    return [line + '\n' for line in lines[:-1]] + [lines[-1]]


def _row_migration_extract(root, source, names):
    path = root / source
    text = path.read_text()
    lines = _row_migration_lf_lines(text)
    blocks, edits = [], []
    for node in ast.parse(text).body:
        if _row_migration_name(node) not in names:
            continue
        first = min([node.lineno, *[item.lineno for item in getattr(node, 'decorator_list', [])]])
        begin = sum(map(len, lines[:first - 1]))
        end = sum(map(len, lines[:node.end_lineno]))
        block = text[begin:end].rstrip()
        old = _row_migration_name(node)
        if old in RENAMES:
            block = block.replace('def ' + old + '(', 'def ' + RENAMES[old] + '(', 1)
        blocks.append(block)
        edits.append((begin, end, ''))
    path.write_text(_row_migration_apply(text, edits))
    return blocks


def _row_migration_import(root, path):
    text = path.read_text()
    tree = ast.parse(text)
    aliases, edits, inserted, renamed_names = {}, [], set(), {}
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    selected_imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = ('prismaquant.' + node.module if node.level == 1 and node.module else node.module)
            if module in MOVES:
                target, names = MOVES[module]
                moved = [item for item in node.names if item.name in names]
                if moved:
                    kept = [item for item in node.names if item.name not in names]
                    clauses = []
                    if kept:
                        clauses.append(ast.unparse(ast.ImportFrom(module=node.module, names=kept, level=node.level)))
                    for item in moved:
                        new = RENAMES.get(item.name, item.name)
                        if item.asname is None and new != item.name:
                            renamed_names[item.name] = new
                            selected_imports.setdefault(_row_migration_scope(node, parents), set()).add(item.name)
                    clauses.append('from prismaquant.' + target + ' import ' + ', '.join(
                        RENAMES.get(item.name, item.name) + (' as ' + item.asname if item.asname else '') for item in moved))
                    start, end = _row_migration_span(text, node)
                    edits.append((start, end, ('\n' + ' ' * node.col_offset).join(clauses)))
            if module == 'prismaquant' or (node.level == 1 and node.module is None):
                for item in node.names:
                    if 'prismaquant.' + item.name in MOVES:
                        aliases[item.asname or item.name] = 'prismaquant.' + item.name
        elif isinstance(node, ast.Import):
            for item in node.names:
                if item.name in MOVES:
                    aliases[item.asname or item.name] = item.name
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            module = aliases.get(ast.unparse(node.value))
            if module:
                target, names = MOVES[module]
                if node.attr in names:
                    start, end = _row_migration_span(text, node)
                    edits.append((start, end, target + '.' + RENAMES.get(node.attr, node.attr)))
                    inserted.add(target)
        elif isinstance(node, ast.Call) and len(node.args) >= 2:
            subject, attribute = node.args[:2]
            module = aliases.get(ast.unparse(subject))
            if module and isinstance(attribute, ast.Constant):
                target, names = MOVES[module]
                if attribute.value in names:
                    edits.append((*_row_migration_span(text, subject), target))
                    edits.append((*_row_migration_span(text, attribute), repr(RENAMES.get(attribute.value, attribute.value))))
                    inserted.add(target)
        elif (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
              and node.id in renamed_names
              and not _row_migration_shadows(node, node.id, parents, selected_imports)):
            edits.append((*_row_migration_span(text, node), renamed_names[node.id]))
    if not edits:
        return
    text = _row_migration_apply(text, edits)
    for target in sorted(inserted):
        text = _row_migration_add_module(text, target)
    path.write_text(text)


def _row_migration_scope(node, parents):
    node = parents.get(node)
    while node is not None and not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
        node = parents.get(node)
    return node


def _row_migration_shadows(node, name, parents, selected_imports):
    scope = _row_migration_scope(node, parents)
    while scope is not None:
        if name in selected_imports.get(scope, ()):
            return False
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = scope.args
            args = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                    *([arguments.vararg] if arguments.vararg else []),
                    *([arguments.kwarg] if arguments.kwarg else [])]
            if any(arg.arg == name for arg in args):
                return True
            for child in ast.walk(scope):
                if _row_migration_scope(child, parents) is not scope:
                    continue
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store) and child.id == name:
                    return True
                if isinstance(child, (ast.ImportFrom, ast.Import)):
                    if any((item.asname or item.name) == name for item in child.names):
                        return True
        scope = _row_migration_scope(scope, parents)
    return False


def _row_migration_export_paths(root):
    """Follow actual module imports so accidental reexport callers migrate too."""
    sources = []
    for directory in ('prismaquant', 'tools', 'tests', 'experiments'):
        for path in (root / directory).rglob('*.py'):
            if path.name == Path(__file__).name:
                continue
            module = '.'.join(path.relative_to(root).with_suffix('').parts)
            sources.append((module, path, ast.parse(path.read_text())))
    changed = True
    while changed:
        changed = False
        for module, path, tree in sources:
            for node in tree.body:
                if not isinstance(node, ast.ImportFrom):
                    continue
                source = 'prismaquant.' + node.module if node.level == 1 and node.module else node.module
                if source not in MOVES:
                    continue
                target, names = MOVES[source]
                exported = {item.asname or item.name for item in node.names if item.name in names}
                if not exported:
                    continue
                for key in (module, *([path.stem] if path.parent.name in ('tests', 'tools') else [])):
                    if key in MOVES:
                        if MOVES[key][0] != target:
                            continue
                        before = len(MOVES[key][1])
                        MOVES[key][1].update(exported)
                        changed |= len(MOVES[key][1]) != before
                    else:
                        MOVES[key] = (target, set(exported))
                        changed = True


def _row_migration_add_module(text, target):
    statement = 'from prismaquant import ' + target
    if statement in text:
        return text
    tree = ast.parse(text)
    after = next((node.end_lineno for node in tree.body if isinstance(node, ast.ImportFrom)
                  and node.module == '__future__'),
                 tree.body[0].end_lineno if tree.body and isinstance(tree.body[0], ast.Expr)
                 and isinstance(tree.body[0].value, ast.Constant)
                 and isinstance(tree.body[0].value.value, str) else 0)
    lines = _row_migration_lf_lines(text)
    lines.insert(after, statement + '\n')
    return ''.join(lines)


def _row_migration_qualify(root, source, names, target):
    path = root / source
    text = path.read_text()
    edits = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names:
            edits.append((*_row_migration_span(text, node), target + '.' + RENAMES.get(node.id, node.id)))
    if edits:
        text = _row_migration_add_module(_row_migration_apply(text, edits), target)
    path.write_text(text)


def _row_migration_finish(root):
    replacements = {
        'prismaquant/joint_aura.py': [
            ('import re\n', ''),
            ('# The #1962 descriptive per-sequence/per-block sidecar. Its arithmetic is a\n'
             '# separate reconstruction scope: the authoritative whole-draw fields stay\n'
             '# bitwise authoritative, and the sidecar\'s residual against them is reported,\n'
             '# gated and never distributed onto the price.\n', ''),
        ],
        'prismaquant/joint_projection_backend.py': [
            ('from functools import lru_cache\n', ''), ('import json\n', ''),
            ('from .digests import bytes_sha256hex, file_sha256hex',
             'from .digests import file_sha256hex'),
        ],
        'prismaquant/tessera_campaign_table.py': [
            ('prismaquant.allocator_candidates.joint_row_binds_cell',
             'prismaquant.joint_aura_rows.joint_row_binds_cell'),
        ],
        'prismaquant/glm_mtp_quantum.py': [
            ('``joint_aura.make_joint_aura_entry``', '``joint_aura_rows.make_joint_aura_entry``'),
        ],
        'tests/test_digest_profiles_1301.py': [
            ('("prismaquant.joint_aura", "identity_sha256", {})',
             '("prismaquant.joint_aura_rows", "joint_identity_sha256", {})'),
            ("module.replace('aura_checkpoint_journal', 'aura_cost')",
             "module.replace('aura_checkpoint_journal', 'aura_cost').replace('joint_aura_rows', 'joint_aura')"),
            ("name.replace('_checkpoint_json_sha256', '_canonical_json_sha256')",
             "name.replace('_checkpoint_json_sha256', '_canonical_json_sha256').replace('joint_identity_sha256', 'identity_sha256')"),
        ],
        'tests/test_probe_identity_routing_1688.py': [
            ('monkeypatch.setattr(ja, "DIRECT_ASCII_STRICT"',
             'monkeypatch.setattr(joint_aura_rows, "DIRECT_ASCII_STRICT"'),
            ('monkeypatch.setattr(ja, "bytes_sha256hex"',
             'monkeypatch.setattr(joint_aura_rows, "bytes_sha256hex"'),
        ],
        'tests/test_no_new_seals.py': [
            ('    "prismaquant/joint_projection_backend.py",\n',
             '    "prismaquant/joint_projection_backend.py",\n    "prismaquant/joint_projection_contract.py",\n'),
            ('("prismaquant/joint_projection_backend.py", "validate_projection_backend_identity")',
             '("prismaquant/joint_projection_contract.py", "validate_projection_backend_identity")'),
            ("            '_require_probe_alignment': 1,  # same mathematical probe coordinates",
             "        },\n        'prismaquant/joint_aura_rows.py': {\n"
             "            'require_probe_alignment': 1,  # same mathematical probe coordinates"),
        ],
        'tests/fixtures/duplication_baseline.json': [
            ('   "prismaquant/joint_aura.py",\n', ''),
            ('  "_qualification": [\n   "prismaquant/glm_kda_capture_kernel.py",\n'
             '   "prismaquant/joint_projection_backend.py"\n  ],\n', ''),
        ],
    }
    for relative, changes in replacements.items():
        path = root / relative
        text = path.read_text()
        for old, new in changes:
            text = text.replace(old, new)
        path.write_text(text)
    # Collapse only empty top-level gaps left by extraction, never string data.
    for relative in ('joint_aura.py', 'joint_projection_backend.py',
                     'allocator_candidates.py', 'priced_cell.py'):
        path = root / 'prismaquant' / relative
        text = path.read_text()
        nodes, edits = ast.parse(text).body, []
        for left, right in zip(nodes, nodes[1:]):
            end = _row_migration_span(text, left)[1]
            begin = _row_migration_span(text, right)[0]
            gap = text[end:begin]
            if not gap.strip() and gap.count('\n') > 3:
                edits.append((end, begin, '\n\n\n'))
        path.write_text(_row_migration_apply(text, edits))


def _row_migration_verify(root, base):
    """Compare retained scalar/tensor operations; actual behavior is tested separately."""
    inverse = {value: key for key, value in RENAMES.items()}

    class Normalize(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None

        def visit_Import(self, node):
            return None

        def visit_Attribute(self, node):
            node = self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id in {
                    'joint_aura_rows', 'joint_projection_contract', 'priced_cell'}:
                if node.value.id != 'priced_cell' or node.attr == 'cost_entry_is_joint_aura':
                    return ast.Name(id=inverse.get(node.attr, node.attr), ctx=node.ctx)
            return node

        def visit_Name(self, node):
            node.id = inverse.get(node.id, node.id)
            return node

        def visit_FunctionDef(self, node):
            node.name = inverse.get(node.name, node.name)
            return self.generic_visit(node)

    checked = []
    for module, (target, moved) in MOVES.items():
        relative = module.replace('.', '/') + '.py'
        original = subprocess.check_output(['git', 'show', base + ':' + relative], cwd=root, text=True)
        old_nodes = {_row_migration_name(node): node for node in ast.parse(original).body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        current = {_row_migration_name(node): node for node in ast.parse((root/relative).read_text()).body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        new_nodes = {_row_migration_name(node): node for node in ast.parse(
            (root/'prismaquant'/str(target + '.py')).read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for name, before in old_nodes.items():
            owner = new_nodes if name in moved else current
            after = owner[RENAMES.get(name, name) if name in moved else name]
            normalize = lambda node: ast.dump(Normalize().visit(copy.deepcopy(node)), include_attributes=False)
            if normalize(before) != normalize(after):
                raise SystemExit('operation drift at ' + module + '.' + name)
            if name in moved and name in current:
                raise SystemExit('old owner retained ' + module + '.' + name)
            checked.append(module + '.' + name)
    legacy = []
    for directory in ('prismaquant', 'tools', 'tests', 'experiments'):
        for path in (root / directory).rglob('*.py'):
            if path.name == Path(__file__).name:
                continue
            tree, aliases = ast.parse(path.read_text()), {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = 'prismaquant.' + node.module if node.level == 1 and node.module else node.module
                    if module in MOVES:
                        legacy.extend(f'{path.relative_to(root)}:{node.lineno}:{item.name}'
                                      for item in node.names if item.name in MOVES[module][1])
                    if module == 'prismaquant' or (node.level == 1 and node.module is None):
                        for item in node.names:
                            if 'prismaquant.' + item.name in MOVES:
                                aliases[item.asname or item.name] = 'prismaquant.' + item.name
                elif isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name in MOVES:
                            aliases[item.asname or item.name] = item.name
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    module = aliases.get(ast.unparse(node.value))
                    if module and node.attr in MOVES[module][1]:
                        legacy.append(f'{path.relative_to(root)}:{node.lineno}:{node.attr}')
    if legacy:
        raise SystemExit('legacy row/contract API reaches: ' + ', '.join(legacy))
    print(json.dumps({'base': base, 'normalized_operation_bodies': len(checked),
                      'legacy_api_reaches': legacy, 'preserved': checked}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--verify-base', help='Verify operation preservation instead of editing; run through PB.')
    args = parser.parse_args()
    root = args.root
    if args.verify_base:
        _row_migration_verify(root, args.verify_base)
        return
    rows = root / 'prismaquant/joint_aura_rows.py'
    if rows.exists():
        raise SystemExit('joint_aura_rows already exists; use a fresh pre-migration checkout')
    _row_migration_export_paths(root)
    blocks = _row_migration_extract(root, 'prismaquant/joint_aura.py', ROW_NODES)
    blocks += _row_migration_extract(root, 'prismaquant/priced_cell.py', {'cost_entry_is_joint_aura'})
    blocks += _row_migration_extract(root, 'prismaquant/allocator_candidates.py',
        {'JointCellCoordinateError', 'joint_row_binds_cell'})
    rows.write_text('"""Joint AURA scalar rows, sample identity custody and coordinate validation.\n\n'
        'This owner validates recorded scientific data without importing a tensor\n'
        'runtime, loading a model or allocating a cache.\n"""\n'
        'from __future__ import annotations\n\nfrom collections.abc import Mapping\n'
        'from dataclasses import dataclass\nimport json\nimport math\nimport re\n'
        'from .digests import DIRECT_ASCII_STRICT, bytes_sha256hex\nfrom .dev_mode import seal_check\n\n\n'
        + '\n\n\n'.join(blocks) + '\n')
    backend = _row_migration_extract(root, 'prismaquant/joint_projection_backend.py', BACKEND_NODES)
    (root / 'prismaquant/joint_projection_contract.py').write_text(
        '"""Packaged qualification and serialized joint-projection arithmetic contract."""\n'
        'from __future__ import annotations\n\nfrom functools import lru_cache\nimport json\n'
        'from pathlib import Path\nfrom .digests import bytes_sha256hex\n\n\n'
        + '\n\n\n'.join(backend) + '\n')
    for module in ('prismaquant.joint_aura', 'prismaquant.priced_cell',
                   'prismaquant.allocator_candidates', 'prismaquant.joint_projection_backend'):
        target, names = MOVES[module]
        _row_migration_qualify(root, module.replace('.', '/') + '.py', names, target)
    for directory in ('prismaquant', 'tools', 'tests', 'experiments'):
        for path in (root / directory).rglob('*.py'):
            if path.name in ('test_no_new_seals.py', Path(__file__).name):
                continue
            _row_migration_import(root, path)
    text = rows.read_text().replace('priced_cell.cost_entry_is_joint_aura(cost_entry)',
                                    'cost_entry_is_joint_aura(cost_entry)')
    text = text.replace('    from prismaquant.joint_aura_rows import validate_joint_aura_entry\n', '')
    rows.write_text(text)
    contract = root / 'prismaquant/joint_projection_contract.py'
    contract.write_text(contract.read_text().replace('= _qualification()', '= projection_qualification()'))
    for path in (rows, contract):
        text = path.read_text()
        edits = [(*_row_migration_span(text, node), RENAMES[node.id])
                 for node in ast.walk(ast.parse(text))
                 if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                 and node.id in RENAMES]
        path.write_text(_row_migration_apply(text, edits))
    _row_migration_finish(root)
    print(json.dumps({'owner': str(rows), 'moved_row_nodes': sorted(ROW_NODES),
                      'moved_backend_nodes': sorted(BACKEND_NODES)}))


if __name__ == '__main__':
    main()
