#!/usr/bin/env python3
"""Move streamed boundary and model-identity owners and migrate their callers."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "prismaquant/cost_streaming.py"
BOUNDARIES = "streamed_boundary_artifacts"
IDENTITY = "streamed_model_identity"
IDENTITY_CONSTANTS = {
    "STREAMED_MODEL_IDENTITY_SCHEMA", "STREAMED_MODEL_IDENTITY_CACHE_SCHEMA",
    "STREAMED_MODEL_PORTABLE_CONTENT_SCHEMA", "_STREAMED_MODEL_CONFIG_PROVENANCE_FIELDS",
}


def names(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return [target.id for target in targets if isinstance(target, ast.Name)]
    return []


def start_line(node):
    return min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", ()))])


def python_source_lines(source):
    parts = source.split("\n")
    return [part + "\n" for part in parts[:-1]] + [parts[-1]]


def ownership(source):
    tree = ast.parse(source)
    forward = next(start_line(n) for n in tree.body if names(n) == ["SelectedSource"])
    identity = next(n.lineno for n in tree.body if names(n) == ["_SOURCE_IDENTITY_BLOCK_BYTES"])
    owners = {}
    for node in tree.body:
        for name in names(node):
            if node.lineno < forward:
                owners[name] = IDENTITY if name in IDENTITY_CONSTANTS else BOUNDARIES
            elif node.lineno >= identity:
                owners[name] = IDENTITY
    return owners, forward, identity


def split(source, owners, forward, identity):
    tree = ast.parse(source)
    lines = python_source_lines(source)
    first = next(start_line(n) for n in tree.body if names(n))
    prefix = "".join(lines[:first - 1])
    boundary_nodes = [n for n in tree.body if any(owners.get(x) == BOUNDARIES for x in names(n))]
    boundary_start = min(start_line(n) for n in boundary_nodes)
    boundary_text = "".join(lines[boundary_start - 1:forward - 1])
    identity_prefix = "\n\n".join(
        "".join(lines[start_line(n) - 1:n.end_lineno]).rstrip()
        for n in tree.body if any(x in IDENTITY_CONSTANTS for x in names(n))
    )
    forward_text = "".join(lines[forward - 1:identity - 1])
    boundary_dependencies = sorted({
        node.id for node in ast.walk(ast.parse(forward_text))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        and owners.get(node.id) == BOUNDARIES
    })
    blocks = {
        BOUNDARIES: prefix + boundary_text,
        IDENTITY: prefix + identity_prefix + "\n\n" + "".join(lines[identity - 1:]),
        "cost_streaming": prefix + f"from .{BOUNDARIES} import {', '.join(boundary_dependencies)}\n\n" + forward_text,
    }
    for module, body in blocks.items():
        doc = {
            BOUNDARIES: "Exact activation boundaries and their produced-output lifecycle.",
            IDENTITY: "Checkpoint and streamed-model identity, verification, and reuse.",
            "cost_streaming": "Streamed causal-LM forwards over the existing model residency owner.",
        }[module]
        old_doc = ast.parse(body).body[0]
        body_lines = python_source_lines(body)
        body = f'"""{doc}"""\n' + "".join(body_lines[old_doc.end_lineno:])
        (ROOT / "prismaquant" / f"{module}.py").write_text(prune_imports(body).rstrip() + "\n")


def prune_imports(source):
    tree = ast.parse(source)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    lines = python_source_lines(source)
    for node in reversed(tree.body):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        aliases = [a for a in node.names if (a.asname or a.name.split('.')[0]) in used]
        if aliases == node.names:
            continue
        if aliases:
            node.names = aliases
            text = ast.unparse(node) + "\n"
        else:
            text = ""
        lines[node.lineno - 1:node.end_lineno] = [text]
    return "".join(lines)


def offsets(source):
    result = [0]
    for line in source.split("\n"):
        result.append(result[-1] + len(line) + 1)
    return result


def moved_reference(value, owners):
    for name, module in owners.items():
        value = value.replace(f"prismaquant.cost_streaming.{name}", f"prismaquant.{module}.{name}")
        value = value.replace(f"{SOURCE}::{name}", f"prismaquant/{module}.py::{name}")
    return value


def migrate(path, owners):
    source = path.read_text()
    original_source = source
    if "cost_streaming" not in source:
        return False
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    points = offsets(source)
    source_lines = source.split("\n")
    changes = []
    aliases = {}
    import_nodes = defaultdict(list)
    extra = defaultdict(set)

    def position(lineno, byte_column):
        prefix = source_lines[lineno - 1].encode("utf-8")[:byte_column].decode("utf-8")
        return points[lineno - 1] + len(prefix)

    def replace(node, text):
        changes.append((position(node.lineno, node.col_offset),
                        position(node.end_lineno, node.end_col_offset), text))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("cost_streaming"):
            groups = defaultdict(list)
            for alias in node.names:
                groups[owners.get(alias.name, "cost_streaming")].append(alias)
            prefix = ("." * node.level) + (node.module or "").removesuffix("cost_streaming")
            statements = []
            for module, imports in groups.items():
                values = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in imports)
                statements.append(f"from {prefix}{module} import {values}")
            replace(node, ("\n" + " " * node.col_offset).join(statements))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "prismaquant.cost_streaming":
                    name = alias.asname or alias.name
                    aliases[name] = alias.asname or "cost_streaming"
                    import_nodes[name].append(node)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "cost_streaming" and node.module in (None, "prismaquant"):
                    name = alias.asname or alias.name
                    aliases[name] = name
                    import_nodes[name].append(node)

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            name = ast.unparse(node.value)
            if name in aliases and node.attr in owners:
                module = owners[node.attr]
                new_alias = aliases[name] + ("_boundaries" if module == BOUNDARIES else "_identity")
                extra[name].add((module, new_alias))
                replace(node, f"{new_alias}.{node.attr}")
        elif isinstance(node, ast.Call) and len(node.args) >= 2:
            name = ast.unparse(node.args[0])
            key = node.args[1]
            if (name in aliases and isinstance(key, ast.Constant)
                    and isinstance(key.value, str) and key.value in owners):
                module = owners[key.value]
                new_alias = aliases[name] + ("_boundaries" if module == BOUNDARIES else "_identity")
                extra[name].add((module, new_alias))
                replace(node.args[0], new_alias)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            updated = moved_reference(node.value, owners)
            parent = parents.get(node)
            if node.value in (SOURCE, "prismaquant.cost_streaming") and isinstance(parent, (ast.Tuple, ast.List)):
                if len(parent.elts) >= 2 and parent.elts[0] is node:
                    symbol = parent.elts[1]
                    if isinstance(symbol, ast.Constant) and isinstance(symbol.value, str):
                        owner = owners.get(symbol.value.split(".")[0])
                        if owner:
                            updated = (f"prismaquant/{owner}.py" if node.value == SOURCE
                                       else f"prismaquant.{owner}")
                if path.name == "test_no_new_seals.py" and len(parent.elts) > 10:
                    existing = {value.value for value in parent.elts if isinstance(value, ast.Constant)}
                    missing = [f"prismaquant/{module}.py" for module in (BOUNDARIES, IDENTITY)
                               if f"prismaquant/{module}.py" not in existing]
                    if missing:
                        replace(node, repr(SOURCE) + "".join(",\n    " + repr(value) for value in missing))
                        continue
            if updated != node.value:
                replace(node, repr(updated))

    for loop in ast.walk(tree):
        if not (isinstance(loop, ast.For) and isinstance(loop.target, ast.Name)
                and isinstance(loop.iter, (ast.Tuple, ast.List))):
            continue
        values = [v.value for v in loop.iter.elts if isinstance(v, ast.Constant)]
        modules = {owners.get(value) for value in values}
        if len(values) != len(loop.iter.elts) or len(modules) != 1 or None in modules:
            continue
        module = next(iter(modules))
        for call in ast.walk(loop):
            if not isinstance(call, ast.Call) or len(call.args) < 2:
                continue
            name = ast.unparse(call.args[0])
            if name not in aliases or ast.unparse(call.args[1]) != loop.target.id:
                continue
            new_alias = aliases[name] + ("_boundaries" if module == BOUNDARIES else "_identity")
            extra[name].add((module, new_alias))
            replace(call.args[0], new_alias)

    for name, additions in extra.items():
        for node in import_nodes[name]:
            end = position(node.end_lineno, node.end_col_offset)
            prefix = "\n" + " " * node.col_offset
            imports = prefix.join(f"from prismaquant import {module} as {alias}" for module, alias in sorted(additions))
            changes.append((end, end, prefix + imports))
    for begin, end, value in sorted(set(changes), reverse=True):
        source = source[:begin] + value + source[end:]
    tree = ast.parse(source)
    loaded = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    lines = python_source_lines(source)
    for node in sorted(ast.walk(tree), key=lambda n: getattr(n, "lineno", 0), reverse=True):
        if isinstance(node, ast.ImportFrom) and node.module in (None, "prismaquant"):
            kept = [a for a in node.names if a.name != "cost_streaming" or (a.asname or a.name) in loaded]
        elif isinstance(node, ast.Import):
            kept = [a for a in node.names if a.name != "prismaquant.cost_streaming" or (a.asname or "prismaquant") in loaded]
        else:
            continue
        if kept == node.names:
            continue
        node.names = kept
        replacement = " " * node.col_offset + ast.unparse(node) + "\n" if kept else ""
        parent = parents.get(node)
        sole_statement = parent is not None and any(
            isinstance(value, list) and value == [node] for value in vars(parent).values()
        )
        if not kept and sole_statement:
            replacement = " " * node.col_offset + "pass\n"
        lines[node.lineno - 1:node.end_lineno] = [replacement]
    source = "".join(lines)
    changed = source != original_source
    if changed:
        path.write_text(source)
    return changed


def migrate_fixtures(owners):
    for filename in ("duplication_baseline.json", "digest_profiles_1301.json"):
        path = ROOT / "tests" / "fixtures" / filename
        source = path.read_text()
        updated = moved_reference(source, owners)
        if updated != source:
            path.write_text(updated)


def verify_definition_ownership(base, owners):
    original = ast.parse(base)
    actual = {}
    for module in (BOUNDARIES, IDENTITY, "cost_streaming"):
        for node in ast.parse((ROOT / "prismaquant" / f"{module}.py").read_text()).body:
            for name in names(node):
                actual[name] = (module, ast.dump(node, include_attributes=False))
    failures = []
    for node in original.body:
        for name in names(node):
            wanted = owners.get(name, "cost_streaming")
            if actual.get(name) != (wanted, ast.dump(node, include_attributes=False)):
                failures.append(name)
    if failures:
        raise SystemExit("moved definitions differ: " + ", ".join(failures))
    print(f"Verified {len(actual)} unchanged definitions and their module owners.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="pre-refactor Git commit")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="rebuild modules from the named base")
    args = parser.parse_args()
    source = subprocess.check_output(["git", "show", f"{args.base}:{SOURCE}"], cwd=ROOT, text=True)
    owners, forward, identity = ownership(source)
    if args.write:
        if args.rebuild or not (ROOT / "prismaquant" / f"{BOUNDARIES}.py").exists():
            split(source, owners, forward, identity)
        changed = []
        for root in ("prismaquant", "tools", "tests", "experiments"):
            for path in (ROOT / root).rglob("*.py"):
                if path.resolve() == Path(__file__).resolve():
                    continue
                if path.name == "test_cost_streaming_refactor.py":
                    continue
                if migrate(path, owners):
                    changed.append(str(path.relative_to(ROOT)))
        for module in (BOUNDARIES, IDENTITY, "cost_streaming"):
            path = ROOT / "prismaquant" / f"{module}.py"
            path.write_text(prune_imports(path.read_text()))
        migrate_fixtures(owners)
        print("\n".join(changed))
    else:
        verify_definition_ownership(source, owners)


if __name__ == "__main__":
    main()
