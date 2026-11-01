#!/usr/bin/env python3
"""Static repository paths and a rerunnable inventory of path-loading idioms.

References use repository paths, directory/** for an unresolved descendant, or
**/suffix when a literal tail has an unresolved base. No test code is executed.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from fnmatch import fnmatchcase
import json
import posixpath
from pathlib import Path

REPOSITORY = '/__pb_gate_repository__'
PATH_TYPES = {'Path', 'PurePath', 'PurePosixPath', 'PosixPath'}


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + '.' + node.attr
    return ''


def join_reference(base, tail):
    if base and (base.endswith('/**') or base == '**'):
        return base
    if tail is None:
        return (base.rstrip('/') + '/**') if base else None
    if base is None:
        return '**/' + tail.lstrip('/') if not tail.startswith('/') else tail
    return posixpath.normpath(posixpath.join(base, tail))


def resolve(node, bindings, filename):
    """Return symbolic paths; empty means unknown, never an empty directory."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Name):
        return {filename} if node.id == '__file__' else bindings.get(node.id, set())
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        # Formatting can enclose a Path division because / and % share precedence.
        # Keep the known containing directory even when the filename is computed.
        prefixes = {value.split('%', 1)[0] for value in resolve(node.left, bindings, filename)}
        return {posixpath.dirname(value).rstrip('/') + '/**' for value in prefixes if '/' in value}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, right = resolve(node.left, bindings, filename), resolve(node.right, bindings, filename)
        return {value for base in left or {None} for tail in right or {None}
                if (value := join_reference(base, tail)) is not None}
    if isinstance(node, ast.Attribute) and node.attr == 'parent':
        return {value if value.endswith('/**') else posixpath.dirname(value)
                for value in resolve(node.value, bindings, filename)}
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == 'parents':
        if isinstance(node.slice, ast.Constant) and type(node.slice.value) is int and node.slice.value >= 0:
            values = resolve(node.value.value, bindings, filename)
            # A source-relative path cannot have more useful ancestors than its parts.
            for _ in range(min(node.slice.value + 1, filename.count('/') + 1)):
                values = {posixpath.dirname(value) for value in values}
            return values
    if isinstance(node, ast.JoinedStr):
        prefix = ''
        for value in node.values:
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                directory = prefix.rsplit('/', 1)[0] if '/' in prefix else ''
                return {directory + '/**'} if directory else set()
            prefix += value.value
        return {prefix}
    if not isinstance(node, ast.Call):
        return set()
    name = dotted(node.func)
    leaf = name.rsplit('.', 1)[-1]
    if leaf in PATH_TYPES | {'str', 'fspath', 'abspath', 'realpath'} and node.args:
        values = resolve(node.args[0], bindings, filename)
        for arg in node.args[1:] if leaf in PATH_TYPES else []:
            tails = resolve(arg, bindings, filename)
            values = {value for base in values or {None} for tail in tails or {None}
                      if (value := join_reference(base, tail)) is not None}
        return values
    if leaf == 'dirname' and node.args:
        return {posixpath.dirname(value) for value in resolve(node.args[0], bindings, filename)}
    if leaf == 'join' and (name.endswith('.path.join') or name in {'posixpath.join', 'join'}):
        values = resolve(node.args[0], bindings, filename) if node.args else set()
        for arg in node.args[1:]:
            tails = resolve(arg, bindings, filename)
            values = {value for base in values or {None} for tail in tails or {None}
                      if (value := join_reference(base, tail)) is not None}
        return values
    if isinstance(node.func, ast.Attribute):
        bases = resolve(node.func.value, bindings, filename)
        if leaf in {'resolve', 'absolute', 'expanduser'}:
            return bases
        if leaf in {'joinpath', 'with_name', 'with_suffix', 'glob', 'rglob'}:
            if leaf in {'with_name', 'with_suffix'}:
                bases = {posixpath.dirname(value) for value in bases}
            if leaf in {'glob', 'rglob'}:
                return {value.rstrip('/') + '/**' for value in bases}
            for arg in node.args:
                tails = resolve(arg, bindings, filename)
                if leaf == 'with_suffix':
                    # Filename transformations cannot narrow below the containing directory.
                    tails = set()
                bases = {value for base in bases or {None} for tail in tails or {None}
                         if (value := join_reference(base, tail)) is not None}
            return bases
    return set()


def references(tree: ast.AST, path: str) -> set[str]:
    filename = REPOSITORY + '/' + path
    found = set()

    def record(values):
        for value in values:
            if not value or '\n' in value or '\0' in value:
                continue
            if value == REPOSITORY + '/**':
                found.add('**')
            elif value.startswith(REPOSITORY + '/'):
                found.add(value[len(REPOSITORY) + 1:])
            elif value.startswith('**/'):
                found.add(value)
            elif value != '**' and not value.startswith('/'):
                found.add(posixpath.normpath(value))
                found.add(posixpath.normpath(posixpath.join(posixpath.dirname(path), value)))

    def visit(node, bindings):
        record(resolve(node, bindings, filename))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            local = bindings.copy()
            # Function-local assignments/arguments must not inherit a global directory binding.
            def mask(child):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    local.pop(child.id, None)
                if isinstance(child, ast.arg):
                    local.pop(child.arg, None)
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    for nested in ast.iter_child_nodes(child):
                        mask(nested)
            for child in ast.iter_child_nodes(node):
                mask(child)
            for child in ast.iter_child_nodes(node):
                visit(child, local)
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is not None:
                visit(node.value, bindings)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = resolve(node.value, bindings, filename) if node.value else set()
            return
        if isinstance(node, ast.If):
            visit(node.test, bindings)
            branches = []
            for body in (node.body, node.orelse):
                branch = bindings.copy()
                for child in body:
                    visit(child, branch)
                branches.append(branch)
            for key in branches[0].keys() | branches[1].keys():
                bindings[key] = branches[0].get(key, set()) | branches[1].get(key, set())
            return
        for child in ast.iter_child_nodes(node):
            visit(child, bindings)

    visit(tree, {})
    return found


def gate_path_matches(path: str, references: set[str]) -> bool:
    references = {reference for reference in references if reference.strip('/*?.')}
    if path in references:
        return True
    return any(fnmatchcase(path, reference) or
               (reference.startswith('**/') and fnmatchcase(path, reference[3:]))
               for reference in references if '**' in reference)



def survey_path_idioms(root: Path) -> dict:
    """Count syntactic idioms, including temporary/generated paths, without claiming coverage."""
    counts = Counter()
    files = defaultdict(set)
    examples = defaultdict(list)
    for path in sorted((root / 'tests').rglob('*.py')):
        relative = path.relative_to(root).as_posix()
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            idiom = None
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                idiom = 'slash_join'
            elif isinstance(node, ast.JoinedStr):
                idiom = 'interpolated_string'
            elif isinstance(node, ast.Call):
                name = dotted(node.func)
                leaf = name.rsplit('.', 1)[-1]
                if leaf in PATH_TYPES:
                    idiom = 'path_constructor'
                elif leaf in {'spec_from_file_location', 'SourceFileLoader', 'run_path', 'joinpath',
                              'with_name', 'with_suffix', 'glob', 'rglob', 'read_text', 'read_bytes', 'open'}:
                    idiom = leaf
                elif name in {'os.path.join', 'posixpath.join', 'os.path.dirname', 'os.path.abspath', 'os.path.realpath'}:
                    idiom = name
            if idiom:
                counts[idiom] += 1
                files[idiom].add(relative)
                if len(examples[idiom]) < 5:
                    examples[idiom].append({'path': relative, 'line': node.lineno,
                                            'expression': ast.unparse(node)[:180]})
    return {'schema': 'prismaquant.gate_path_survey.v1',
            'method': 'AST occurrence and distinct-file counts across tests/**/*.py; overlapping idioms; '
                      'includes temporary paths and fixture generation, not a claim of dependency completeness',
            'idioms': {name: {'occurrences': counts[name], 'files': len(files[name]), 'examples': examples[name]}
                       for name in sorted(counts)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.write_text(json.dumps(survey_path_idioms(args.root), indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
