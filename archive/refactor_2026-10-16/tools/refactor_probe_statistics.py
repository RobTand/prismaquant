#!/usr/bin/env python3
"""Move Fisher statistics owners while preserving their normalization bodies."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path
import textwrap
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
TARGET = "prismaquant.probe_statistics"
MOVES = {
    "prismaquant.sensitivity_probe": {
        "h_detail_blob": "h_detail_blob",
        "finalize_fisher_stats": "finalize_fisher_stats",
        "_DENSE_FISHER_MARGINAL_KEYS": "FISHER_MARGINAL_KEYS",
    },
    "prismaquant.incremental_probe": {
        "merge_marginals": "merge_marginals",
        "_MARGINAL_MAX_KEYS": "FISHER_MAX_KEYS",
        "_PACKED_MARGINAL_KEYS": "PACKED_FISHER_MARGINAL_KEYS",
    },
    "prismaquant.allocator": {"renormalize_probe_fisher": "renormalize_probe_fisher"},
    "prismaquant.measure_quant_cost": {"h_detail_expected_norm_tokens": "h_detail_expected_norm_tokens"},
}
IMPORT_MOVES = {module: dict(names) for module, names in MOVES.items()}
IMPORT_MOVES["prismaquant.incremental_probe"].update(
    finalize_fisher_stats="finalize_fisher_stats", h_detail_blob="h_detail_blob",
    _MARGINAL_KEYS="FISHER_MARGINAL_KEYS")


def statistics_definition_name(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def statistics_source(module, base):
    return subprocess.check_output(
        ["git", "show", f"{base}:{module.replace('.', '/')}.py"], cwd=ROOT, text=True)


def statistics_symbols(text, names):
    replacements = dict(names)
    replacements.update(_MARGINAL_KEYS="FISHER_MARGINAL_KEYS", _MARGINAL_MAX_KEYS="FISHER_MAX_KEYS")
    return re.sub(r"\b(" + "|".join(re.escape(name) for name in replacements) + r")\b",
                  lambda match: replacements[match.group()], text)


def move_definitions(base):
    blocks = {}
    for module, names in MOVES.items():
        source = statistics_source(module, base)
        lines = source.splitlines(keepends=True)
        nodes = [node for node in ast.parse(source).body if statistics_definition_name(node) in names]
        for node in reversed(nodes):
            text = "".join(lines[node.lineno - 1:node.end_lineno])
            del lines[node.lineno - 1:node.end_lineno]
            text = statistics_symbols(text, names)
            blocks[names[statistics_definition_name(node)]] = text
        source = "".join(lines)
        imports = ", ".join(new + (f" as {old}" if old != new else "") for old, new in names.items())
        source = source.replace("from __future__ import annotations\n",
                                f"from __future__ import annotations\n\nfrom .probe_statistics import {imports}\n", 1)
        (ROOT / (module.replace(".", "/") + ".py")).write_text(source)
    order = ["FISHER_MARGINAL_KEYS", "FISHER_MAX_KEYS", "PACKED_FISHER_MARGINAL_KEYS",
             "merge_marginals", "h_detail_blob", "finalize_fisher_stats",
             "renormalize_probe_fisher", "h_detail_expected_norm_tokens"]
    (ROOT / "prismaquant/probe_statistics.py").write_text(
        '"""Shared Fisher rows, marginal reduction, normalization and detail contracts."""\n'
        'from __future__ import annotations\n\nimport numpy as np\nimport torch\n\n\n'
        + "\n\n".join(blocks[name].rstrip() for name in order) + "\n")


def statistics_import_statement(module, names):
    prefix = f"from {module} import "
    values = ", ".join(names)
    if len(prefix + values) <= 88:
        return prefix + values
    return prefix + "(\n" + "".join(f"    {name},\n" for name in names) + ")"


def migrate_statistics_imports(path):
    source = path.read_text()
    if not any(module.rsplit(".", 1)[1] in source for module in IMPORT_MOVES):
        return False
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in sorted(ast.walk(tree), key=lambda node: getattr(node, "lineno", 0), reverse=True):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = ("prismaquant." if node.level else "") + (node.module or "")
        moved = IMPORT_MOVES.get(module, {})
        if not any(alias.name in moved for alias in node.names):
            continue
        groups = defaultdict(list)
        for alias in node.names:
            if alias.name in moved:
                name = moved[alias.name]
                local = alias.asname or alias.name
                groups[TARGET].append(name + (f" as {local}" if local != name else ""))
            else:
                groups[module].append(alias.name + (f" as {alias.asname}" if alias.asname else ""))
        statements = [statistics_import_statement(module, names) for module, names in groups.items()]
        lines[node.lineno - 1:node.end_lineno] = [textwrap.indent("\n".join(statements), " " * node.col_offset) + "\n"]
    updated = "".join(lines)
    for module, names in IMPORT_MOVES.items():
        for old, new in names.items():
            updated = updated.replace(module + "." + old, TARGET + "." + new)
    if updated != source:
        path.write_text(updated)
        return True
    return False


def qualify_statistics_owner(module):
    path = ROOT / (module.replace(".", "/") + ".py")
    source = path.read_text()
    tree = ast.parse(source)
    imported = {}
    lines = source.splitlines(keepends=True)
    changes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in (TARGET, "probe_statistics"):
            imported.update({alias.asname or alias.name: alias.name for alias in node.names})
            changes.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, ""))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in imported:
            changes.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset,
                            "probe_statistics." + imported[node.id]))
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    for lineno, col, end_lineno, end_col, text in sorted(changes, reverse=True):
        begin = offsets[lineno - 1] + len(lines[lineno - 1].encode()[:col].decode())
        end = offsets[end_lineno - 1] + len(lines[end_lineno - 1].encode()[:end_col].decode())
        source = source[:begin] + text + source[end:]
    if changes:
        source = source.replace("from __future__ import annotations\n",
                                "from __future__ import annotations\n\nfrom . import probe_statistics\n", 1)
        path.write_text(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.write:
        if args.rebuild or not (ROOT / "prismaquant/probe_statistics.py").exists():
            move_definitions(args.base)
        for directory in ("prismaquant", "tools", "tests", "experiments"):
            for path in (ROOT / directory).rglob("*.py"):
                if path.resolve() != Path(__file__).resolve() and migrate_statistics_imports(path):
                    print(path.relative_to(ROOT))
        for module in MOVES:
            qualify_statistics_owner(module)
    else:
        actual = {statistics_definition_name(node): node for node in ast.parse(
            (ROOT / "prismaquant/probe_statistics.py").read_text()).body}
        for module, names in MOVES.items():
            original = ast.parse(statistics_source(module, args.base))
            for node in original.body:
                old = statistics_definition_name(node)
                if old not in names:
                    continue
                text = ast.unparse(node)
                text = statistics_symbols(text, names)
                expected = ast.parse(text).body[0]
                if ast.dump(expected, include_attributes=False) != ast.dump(actual[names[old]], include_attributes=False):
                    raise SystemExit(f"changed statistics definition: {module}.{old}")
        print("All Fisher statistics definitions retain their original bodies.")


if __name__ == "__main__":
    main()
