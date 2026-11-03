#!/usr/bin/env python3
"""Read PrismaQuant's dependency graph for Bazel from its own evidence (#3216).

Gazelle can't express PrismaQuant's imports: test modules import other test
modules (a py_test can't be a dependency), import cycles exist, conftests put
directories such as tools/g3job on sys.path at run time, modules launch other
modules by path or with `python -m`, and prismaquant/__init__.py exports names
lazily. The merge gate's static selector already reads most of this, so the
graph comes from the same evidence (tools/ci/gate_select.py,
tools/ci/gate_paths.py), with the rules PrismaBuild's generator settled on
(PB #2613) and the PrismaQuant cases above:

- library edges: import statements anywhere in a file, imports in embedded
  `python -c` snippets, literal import_module and __import__ names,
  pytest_plugins entries, exact loader paths, lazy package exports and, in
  tests, exact paths to non-test modules and `-m` module launches;
- data: every file a module names by path (tests), or reads for every caller
  (production: exact files and with_name siblings);
- a test's own inputs, which never reach other tests: modules it names in
  strings when it imports by name, test files it runs in a child pytest, the
  tree reads of production modules it imports directly, and the programs its
  closure launches (`.py` paths that reach subprocess, os.exec*, runpy or a
  file loader, and `-m` modules), file by file;
- the uncertain list: modules whose launches, sys.path entries or dynamic
  imports can't be followed. A launch that can't be followed makes every exact
  path in the module count; the others are reported, not guessed.

This milestone writes the graph (--graph-json) and the uncertain list. BUILD
files, rules_python and pip come next; the population tags they need are
already here (test_attributes).
"""
from __future__ import annotations

import argparse
import ast
import builtins
from collections import defaultdict
import importlib.util
import json
import math
import os
import posixpath
import re
from pathlib import Path, PurePosixPath
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]
# Bazel packages that will hold modules. Every other file belongs to the root
# package. tools/bazel is written by hand.
PACKAGES = ('prismaquant', 'tools', 'tests')
HANDWRITTEN = ('tools/bazel/',)
IMPORT_LINE = re.compile(r'^\s*(import\s+[A-Za-z_][\w.]*|from\s+[A-Za-z_.][\w.]*\s+import\s+[\w*(])')
# Calls that import a module named at run time.
NAME_IMPORTERS = frozenset({'import_module', '__import__', 'spec_from_file_location', 'SourceFileLoader',
                            'run_path', 'run_module'})
# A string naming a module, as gate_select reads one.
MODULE_NAME = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*')
# `python -m module` inside a command string.
DASH_M = re.compile(r'(?:^|\s)-m\s+([A-Za-z_][\w.]*)')
# Directories that are never inputs: VCS metadata and Bazel's own output links.
SKIPPED = ('.git/', 'bazel-')
STDLIB = frozenset(sys.stdlib_module_names)
# Files longer than this on hosted runners are split across Bazel test shards.
SHARD_SECONDS = 40.0
MAX_SHARDS = 8

LAUNCH_NOT_FOLLOWED = 'launch arguments not followed'
SYS_PATH_NOT_FOLLOWED = 'sys.path entry not followed'
DYNAMIC_IMPORT_NOT_FOLLOWED = 'dynamic import not followed'
NOT_PARSED = 'not parseable as Python'


def import_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The merge gate's selector from this checkout: module names, path references
# and the population all come from it.
GATE_SELECT = import_from_path('gate_select_for_bazel', ROOT / 'tools/ci/gate_select.py')
gate_paths = GATE_SELECT.gate_paths


def repository_files(root):
    """Tracked files in the Git index, including staged but uncommitted additions."""
    output = subprocess.run(['git', 'ls-files', '-z', '--cached'],
                            cwd=root, capture_output=True, check=True).stdout.decode()
    # BUILD files define the graph; no test reads them.
    return sorted(path for path in set(output.split('\0')) if path and not path.startswith(SKIPPED)
                  and PurePosixPath(path).name != 'BUILD.bazel'
                  and (root / path).is_file() and not (root / path).is_symlink())


def package_of(path):
    if path.startswith(HANDWRITTEN):
        return HANDWRITTEN[0].rstrip('/')
    for package in PACKAGES:
        if path.startswith(package + '/'):
            return package
    return ''


def is_module(path):
    """Any Python file outside the handwritten Bazel tree: tests import experiments/,
    docs/ and scripts/ modules too, so no directory is left out."""
    return path.endswith('.py') and not path.startswith(HANDWRITTEN)


def is_test(path):
    return path.startswith('tests/') and PurePosixPath(path).name.startswith('test_') and path.endswith('.py')


def is_conftest(path):
    return PurePosixPath(path).name == 'conftest.py'


def is_test_side(path):
    return path.startswith('tests/') or is_conftest(path)


def conftest_chain(test, known):
    """The conftests pytest loads for one test file: a root one, and each one from
    the top directory down to the test's own. A conftest in another directory
    is not an input of this test."""
    chain = ['conftest.py'] if 'conftest.py' in known else []
    parts = PurePosixPath(test).parent.parts
    for end in range(1, len(parts) + 1):
        candidate = '/'.join(parts[:end]) + '/conftest.py'
        if candidate in known:
            chain.append(candidate)
    return chain


def strongly_connected(nodes, edges):
    """Tarjan's algorithm, iterative. Returns components in a stable order."""
    index, low, on_stack, stack, components = {}, {}, set(), [], []
    counter = 0
    for start in sorted(nodes):
        if start in index:
            continue
        work = [(start, iter(sorted(edges.get(start, ()))))]
        index[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(edges.get(child, ())))))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(sorted(component))
    return sorted(components)


def parse(text):
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError):
        return None


_WALKED = {}


def walk(tree):
    """ast.walk of a whole parsed file, computed once: every rule reads each file's nodes."""
    key = id(tree)
    if key not in _WALKED or _WALKED[key][0] is not tree:
        _WALKED[key] = (tree, list(ast.walk(tree)))
    return _WALKED[key][1]


def leaf(func):
    """The last name of a called function: `resolve` in `Path(x).resolve()`."""
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ''


class Bindings:
    """A module's import aliases and every assignment to each name, in any scope.

    Following an argument through them is how gate_select follows loaders: it
    keeps every value a name was ever given, so it errs towards more paths.
    """

    def __init__(self, tree):
        self.aliases, self.assignments = defaultdict(set), defaultdict(list)
        self.aliases['__import__'].add('__import__')
        for node in walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.aliases[alias.asname or alias.name.split('.')[0]].add(
                        alias.name if alias.asname else alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for alias in node.names:
                    self.aliases[alias.asname or alias.name].add(f'{node.module}.{alias.name}')
            evidence = []
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                evidence = [node]
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                evidence = [ast.copy_location(ast.Assign(targets=[node.target], value=node.iter), node.target)]
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                evidence = [ast.copy_location(ast.Assign(targets=[item.optional_vars], value=item.context_expr),
                                              item.optional_vars)
                            for item in node.items if item.optional_vars is not None]
            for assignment in evidence:
                for target in assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                            self.assignments[name.id].append(assignment)

    def meanings(self, func):
        """Every alias-resolved dotted name a call's function may be."""
        first, _, rest = gate_paths.dotted(func).partition('.')
        return {meaning + ('.' + rest if rest else '') for meaning in self.aliases.get(first, ())}

    def follow(self, arguments):
        """(assignments the arguments read, transitively; whether any value comes
        from a parameter or from a call other than a path or string conversion)."""
        pending = [name.id for argument in arguments for name in ast.walk(argument) if isinstance(name, ast.Name)]
        opaque = any(isinstance(part, ast.Call) and leaf(part.func) not in CONVERSIONS
                     for argument in arguments for part in ast.walk(argument))
        seen, bindings = set(), []
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            if name not in self.assignments and name not in self.aliases and name not in BUILTIN_NAMES:
                opaque = True  # A parameter: its value comes from a caller.
            for assignment in self.assignments.get(name, ()):
                bindings.append(assignment)
                pending.extend(child.id for child in ast.walk(assignment.value) if isinstance(child, ast.Name))
        return sorted({id(item): item for item in bindings}.values(), key=lambda item: item.lineno), opaque, seen

    def values(self, argument, path, gate_paths):
        """(the symbolic paths one argument may be, as gate_paths resolves them;
        whether it can't be followed). Each name keeps every value it is given,
        and a loop over a literal list or tuple takes each element."""
        filename = gate_paths.REPOSITORY + '/' + path
        followed, opaque, _ = self.follow([argument])
        names = {}

        def resolve(value):
            if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                return set().union(*(resolve(item) for item in value.elts))
            return gate_paths.resolve(value, names, filename)

        for assignment in followed:
            for target in assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]:
                if isinstance(target, ast.Name):
                    names[target.id] = names.get(target.id, set()) | resolve(assignment.value)
        return resolve(argument), opaque

    def references(self, arguments, path, gate_paths):
        """(repository paths the arguments may be, whether they can't be followed)."""
        bindings, opaque, _ = self.follow(arguments)
        body = [*bindings, *(ast.Expr(value=argument) for argument in arguments)]
        return gate_paths.references(ast.Module(body=body, type_ignores=[]), path), opaque


# Calls whose result is still the path or text they were given.
CONVERSIONS = frozenset({'str', 'Path', 'PurePath', 'PosixPath', 'PurePosixPath', 'fspath', 'resolve', 'absolute',
                         'list', 'tuple', 'join', 'joinpath', 'format', 'sorted', 'quote', 'abspath', 'realpath',
                         'normpath', 'expanduser', 'parent'})
BUILTIN_NAMES = frozenset(dir(builtins)) | {'__file__', '__name__'}
# Calls that start a program or load a file as code, by alias-resolved name.
LAUNCHERS = frozenset({
    'subprocess.run', 'subprocess.Popen', 'subprocess.call', 'subprocess.check_call', 'subprocess.check_output',
    'os.execv', 'os.execve', 'os.execvp', 'os.execvpe', 'os.execl', 'os.execle', 'os.execlp', 'os.execlpe',
    'os.spawnv', 'os.spawnve', 'os.spawnvp', 'os.spawnvpe', 'os.posix_spawn', 'os.posix_spawnp', 'os.system',
    'runpy.run_path', 'runpy.run_module', 'importlib.util.spec_from_file_location',
    'importlib.machinery.SourceFileLoader', 'asyncio.create_subprocess_exec', 'asyncio.create_subprocess_shell'})
LOADER_SLOTS = {'importlib.util.spec_from_file_location': (1, 'location'),
                'importlib.machinery.SourceFileLoader': (1, 'path'),
                'importlib.machinery.SourcelessFileLoader': (1, 'path'),
                'runpy.run_path': (0, 'path_name')}
SYS_PATH_CALLS = frozenset({'sys.path.insert', 'sys.path.append', 'site.addsitedir'})


def loader_evidence(tree, path, gate_paths, bindings=None):
    """(module names imported by a literal import_module or __import__, loader paths,
    whether a dynamic import's name is not a literal).

    PrismaBuild's gate_select.production_loads, which also follows names bound
    by for and with, and keeps every meaning of a reused alias.
    """
    bindings = bindings or Bindings(tree)
    modules, paths, dynamic = set(), set(), []
    for node in walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for name in sorted(bindings.meanings(node.func)):
            if name in {'importlib.import_module', '__import__'}:
                target = next((item.value for item in node.keywords if item.arg == 'name'),
                              node.args[0] if node.args else None)
                if isinstance(target, ast.Constant) and isinstance(target.value, str):
                    module = target.value
                    if module.startswith('.') and name == 'importlib.import_module':
                        package = next((item.value for item in node.keywords if item.arg == 'package'),
                                       node.args[1] if len(node.args) > 1 else None)
                        if not isinstance(package, ast.Constant) or not isinstance(package.value, str):
                            dynamic.append(target)
                            continue
                        parts = package.value.split('.')
                        level = len(module) - len(module.lstrip('.'))
                        module = '.'.join(parts[:len(parts) - level + 1] + [module.lstrip('.')]).rstrip('.')
                    modules.add(module)
                elif target is not None:
                    dynamic.append(target)
            if name not in LOADER_SLOTS:
                continue
            position, keyword = LOADER_SLOTS[name]
            target = next((item.value for item in node.keywords if item.arg == keyword),
                          node.args[position] if len(node.args) > position else None)
            if target is None:
                continue
            followed, _, _ = bindings.follow([target])
            references = gate_paths.references(
                ast.Module(body=[*followed, ast.Expr(value=target)], type_ignores=[]), path)
            if any(not isinstance(binding, ast.Name) for assignment in followed
                   for binding in (assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target])):
                # Compound bindings can split directory and filename literals. Keep
                # descendants and filename tails rather than lose their combinations.
                references.update(reference.rstrip('/') + '/**' for reference in tuple(references)
                                  if '**' not in reference and not reference.endswith('.py'))
                references.update('**/' + PurePosixPath(reference).name
                                  for reference in tuple(references) if reference.endswith('.py'))
            paths.update(references)
    return modules, paths, dynamic


def launched_references(tree, path, gate_paths, bindings=None):
    """(path references that reach a launch in production module `path`,
    whether a launch's arguments couldn't be followed).

    Each launch call's arguments are followed through the module's assignments
    to the names they use, and resolved with gate_paths. An argument built from
    a parameter or from a call other than a path or string conversion can't be
    followed; then every exact path in the module counts (Evidence.read).
    """
    bindings = bindings or Bindings(tree)
    found, uncertain = set(), False
    for node in walk(tree):
        if not isinstance(node, ast.Call) or not bindings.meanings(node.func) & LAUNCHERS:
            continue
        arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
        references, opaque = bindings.references(arguments, path, gate_paths)
        found |= references
        uncertain = uncertain or opaque
    return found, uncertain


PATH_BUILDERS = frozenset({'joinpath', 'join', 'open', 'Path', 'PurePath', 'with_name', 'copy', 'copy2',
                           'copyfile', 'read_text', 'read_bytes'})


def path_built_names(tree):
    """Bare .py file names this module builds paths from, as opposed to names it
    only matches or logs. A name counts when it, a table holding it, or a loop
    variable over that table appears in a '/' join or a path-building call.
    """
    def bare(value):
        return isinstance(value, str) and value.endswith('.py') and '/' not in value

    tables = defaultdict(set)
    loops = defaultdict(set)
    for node in walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            values = {item.value for item in ast.walk(node.value) if isinstance(item, ast.Constant) and bare(item.value)}
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Name) and values:
                    tables[target.id] |= values
        if isinstance(node, (ast.For, ast.comprehension)):
            iterated = {name.id for name in ast.walk(node.iter) if isinstance(name, ast.Name)}
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    loops[target.id] |= iterated
    found = set()

    def collect(expression):
        for item in ast.walk(expression):
            if isinstance(item, ast.Constant) and bare(item.value):
                found.add(item.value)
            elif isinstance(item, ast.Name):
                found.update(tables.get(item.id, ()))
                for table in loops.get(item.id, ()):
                    found.update(tables.get(table, ()))

    for node in walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            collect(node.right)
        elif isinstance(node, ast.Call) and ast.unparse(node.func).rsplit('.', 1)[-1] in PATH_BUILDERS:
            for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
                collect(argument)
    return found


def module_launches(tree):
    """(modules launched with `-m`, whether the code runs a child pytest).

    `-m` followed by a module name in a list or tuple is an argv; `-m name`
    inside a string is a command line or an f-string piece of one.
    """
    found = set()
    for node in walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            items = node.elts
            for first, second in zip(items, items[1:]):
                if (isinstance(first, ast.Constant) and first.value == '-m' and isinstance(second, ast.Constant)
                        and isinstance(second.value, str) and MODULE_NAME.fullmatch(second.value)):
                    found.add(second.value)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and '-m' in node.value:
            found.update(DASH_M.findall(node.value))
    child_pytest = 'pytest' in found or any(
        isinstance(node, ast.Call) and gate_paths.dotted(node.func) == 'pytest.main' for node in walk(tree))
    return found, child_pytest


def is_script(root, path):
    """An extension-less Python program under tools/ or scripts/."""
    if not path.startswith(('tools/', 'scripts/')) or '.' in PurePosixPath(path).name or path.startswith(HANDWRITTEN):
        return False
    with open(root / path, 'rb') as handle:
        first = handle.readline(200)
    return first.startswith(b'#!') and b'python' in first


def wildcard(reference):
    return '*' in reference.removeprefix('**/') or '?' in reference


def parse_snippets(text):
    """Python embedded in a string: a `-c` argument or a script a test writes.

    The whole text when it parses after dedenting; otherwise each import line
    alone, which also covers the literal pieces of an f-string.
    """
    try:
        return [ast.parse(textwrap.dedent(text))]
    except (SyntaxError, ValueError):
        pass
    found = []
    for line in text.splitlines():
        if IMPORT_LINE.match(line):
            try:
                # The statement only: an f-string piece may stop mid-call after ';'.
                found.append(ast.parse(line.split(';', 1)[0].strip()))
            except (SyntaxError, ValueError):
                continue
    return found


def statement_imports(tree, path):
    """Every module an import statement or pytest_plugins entry names, as gate_select resolves them."""
    names = set()
    for node in walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = node.module or ''
            if node.level:
                package = list(PurePosixPath(path).parent.parts)
                package = package[:len(package) - node.level + 1]
                prefix = '.'.join([*package, *([prefix] if prefix else [])])
            if prefix:
                names.add(prefix)
            names.update(f'{prefix}.{alias.name}' if prefix else alias.name for alias in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == 'pytest_plugins' for target in targets):
                names.update(value.value for value in ast.walk(node.value)
                             if isinstance(value, ast.Constant) and isinstance(value.value, str))
    return names


def dotted_parts(path):
    parts = path.removesuffix('.py').split('/')
    return parts[:-1] if parts[-1] == '__init__' else parts


def lazy_exports(tree):
    """(table name, {attribute: relative module}) of a package that exports
    names through a module-level __getattr__ and a literal table, or None."""
    if not any(isinstance(node, ast.FunctionDef) and node.name == '__getattr__' for node in tree.body):
        return None
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            pairs = list(zip(node.value.keys, node.value.values))
            if (len(targets) == 1 and isinstance(targets[0], ast.Name) and pairs and all(
                    isinstance(key, ast.Constant) and isinstance(key.value, str) and isinstance(value, ast.Constant)
                    and isinstance(value.value, str) and value.value.startswith('.') for key, value in pairs)):
                return targets[0].id, {key.value: value.value for key, value in pairs}
    return None


class Facts:
    """What one file's source says, before any name is resolved."""

    def __init__(self, names=(), loaders=(), references=(), siblings=(), launched=(), launched_modules=(),
                 child_pytest=False):
        self.names, self.loaders, self.references = set(names), set(loaders), set(references)
        self.siblings, self.launched, self.launched_modules = set(siblings), set(launched), set(launched_modules)
        self.child_pytest = child_pytest


class Evidence:
    """Each module's library edges and the files it reads, from its source.

    `launch_rule=False` counts every exact path production code names as a
    launch, as PB measured before its launch rule; it exists to measure that
    rule, not to build with.
    """

    def __init__(self, root, files, *, launch_rule=True, gate_select=None):
        self.root, self.files, self.file_set = Path(root), list(files), set(files)
        self.launch_rule = launch_rule
        self.gate_select = gate_select or GATE_SELECT
        self.gate_paths = self.gate_select.gate_paths
        self.modules = sorted(path for path in self.files if is_module(path))
        self.scripts = sorted(path for path in self.files if is_script(self.root, path))
        self.directories = {str(parent) for path in self.files for parent in PurePosixPath(path).parents}
        self.uncertain = defaultdict(set)
        self.trees = {}
        for path in self.modules + self.scripts:
            self.trees[path] = parse((self.root / path).read_text(errors='replace'))
            if self.trees[path] is None:
                self.uncertain[path].add(NOT_PARSED)
        self.bindings = {path: Bindings(tree) for path, tree in self.trees.items() if tree is not None}
        self.sys_path_roots = set()
        for path in self.modules:
            self.sys_path_roots |= self.sys_path_entries(path)
        self.aliases = defaultdict(set)
        for path in self.modules:
            for alias in self.module_aliases(path):
                self.aliases[alias].add(path)
        self.lazy, self.lazy_tables = {}, {}
        for path in self.modules:
            if PurePosixPath(path).name == '__init__.py' and self.trees[path] is not None:
                table = lazy_exports(self.trees[path])
                if table:
                    self.lazy_tables[path] = table[0]
                    package = '.'.join(dotted_parts(path))
                    self.lazy[package] = {name: self.imported(self.relative_module(package, module))
                                          for name, module in table[1].items()}
        self.facts = {path: self.read(path) for path in self.modules + self.scripts}
        self._runs, self._tree_reads = {}, {}

    @staticmethod
    def relative_module(package, module):
        level = len(module) - len(module.lstrip('.'))
        parts = package.split('.')
        return '.'.join(parts[:len(parts) - level + 1] + [module.lstrip('.')])

    def sys_path_entries(self, path):
        """Repository directories this module puts on sys.path.

        An entry that resolves to no repository directory (a parameter,
        sys.argv, a path outside the checkout) is uncertain.
        """
        tree, found = self.trees.get(path), set()
        if tree is None:
            return found
        bindings = self.bindings[path]
        for node in walk(tree):
            if not isinstance(node, ast.Call) or not node.args or gate_paths.dotted(node.func) not in SYS_PATH_CALLS:
                continue
            values, opaque = bindings.values(node.args[-1], path, self.gate_paths)
            directories = set()
            for value in values:
                if value == self.gate_paths.REPOSITORY:
                    directories.add('')
                elif value.startswith(self.gate_paths.REPOSITORY + '/'):
                    directories.add(posixpath.normpath(value[len(self.gate_paths.REPOSITORY) + 1:]))
                elif not value.startswith(('/', '*')) and posixpath.normpath(value) in self.directories:
                    directories.add(posixpath.normpath(value))  # relative to the working directory
            directories = {directory for directory in directories if directory in self.directories or not directory}
            if opaque or not directories:
                self.uncertain[path].add(SYS_PATH_NOT_FOLLOWED)
            found |= directories - {''}
        return found

    def module_aliases(self, path):
        """The names a module imports as: gate_select's names, its dotted path from
        the repository root (pytest.ini's pythonpath), every dotted suffix under
        tools/ and tests/ (conftests and tests put their directories on sys.path),
        and its path below any directory a module puts on sys.path. A suffix that
        would shadow a standard library module is left out."""
        parts = dotted_parts(path)
        suffixes = set(self.gate_select.module_names(path))
        if path.startswith(('tools/', 'tests/')):
            suffixes.update('.'.join(parts[start:]) for start in range(1, len(parts)))
        for directory in self.sys_path_roots:
            if path.startswith(directory + '/'):
                suffixes.add('.'.join(dotted_parts(path[len(directory) + 1:])))
        names = {name for name in suffixes if name and name.split('.')[0] not in STDLIB
                 and all(part.isidentifier() for part in name.split('.'))}
        if parts and all(part.isidentifier() for part in parts):
            names.add('.'.join(parts))
        return names

    def read(self, path):
        tree = self.trees[path]
        if tree is None:
            return Facts()
        trees = [tree] + [snippet for node in walk(tree)
                          if isinstance(node, ast.Constant) and isinstance(node.value, str) and 'import ' in node.value
                          for snippet in parse_snippets(node.value)]
        names, loaders, dynamic = set(), set(), []
        for index, each in enumerate(trees):
            names |= statement_imports(each, path)
            found, loaded, unnamed = loader_evidence(each, path, self.gate_paths, self.bindings[path] if index == 0 else None)
            names |= found
            loaders |= loaded
            if index == 0:
                dynamic += unnamed
        if dynamic and not is_test(path) and not self.lazy_followed(path, dynamic):
            # A test file's own name strings cover it (test_modules); others can't be followed.
            self.uncertain[path].add(DYNAMIC_IMPORT_NOT_FOLLOWED)
        references = set().union(*(self.gate_paths.references(each, path) for each in trees))
        launched_modules, child_pytest = set(), False
        for each in trees:
            found, child = module_launches(each)
            launched_modules |= found
            child_pytest = child_pytest or child
        launched = set()
        if not is_test_side(path):
            launched, uncertain = launched_references(tree, path, self.gate_paths, self.bindings[path])
            if uncertain or not self.launch_rule:
                # Prefer sound: a launch we can't follow may run any path named here.
                if uncertain:
                    self.uncertain[path].add(LAUNCH_NOT_FOLLOWED)
                launched |= {reference for reference in references if not wildcard(reference)}
            # A bare file name a path is built from: the repository module with that name.
            for name in path_built_names(tree):
                launched |= {module for module in self.modules if module.endswith('/' + name)}
        return Facts(names, loaders, references, sibling_names(tree), launched, launched_modules, child_pytest)

    def lazy_followed(self, path, targets):
        """Whether every non-literal dynamic import in a package reads its own lazy-export table."""
        table = self.lazy_tables.get(path)
        if table is None:
            return False
        bindings = self.bindings[path]
        return all(table in bindings.follow([target])[2] for target in targets)

    def imported(self, name):
        parts = name.split('.')
        found = set()
        for end in range(1, len(parts) + 1):
            prefix = '.'.join(parts[:end])
            found.update(self.aliases.get(prefix, ()))
            if end < len(parts) and parts[end] in self.lazy.get(prefix, {}):
                found.update(self.lazy[prefix][parts[end]])
        return found

    def referenced(self, references):
        """(files named exactly, files a pattern matches), from the repository root.

        A bare directory names nothing, as in gate_select. '**/X' is X under a
        base the analyzer couldn't resolve, such as a root passed in as a
        parameter; it is read as the repository path X, not matched anywhere,
        so tmp_path tails like '**/cas' name nothing. '**/**' and '**/' carry
        no path at all. A test that reads a repository file it didn't declare
        finds it missing on the remote executor.
        """
        anchored = set()
        for reference in references:
            if reference.startswith('**/'):
                reference = reference[3:]
                if reference in ('', '**'):
                    continue
            anchored.add(reference)
        if '**' in anchored:
            return set(), set(self.files)
        exact = {reference for reference in anchored if reference in self.file_set}
        patterns = {reference for reference in anchored if '**' in reference}
        matched = ({path for path in self.files if self.gate_paths.gate_path_matches(path, patterns)}
                   if patterns else set())
        return exact, matched - exact

    def of(self, path):
        """(modules this module needs as libraries, files it reads)."""
        facts = self.facts[path]
        modules = set().union(*(self.imported(name) for name in facts.names)) if facts.names else set()
        if is_test_side(path):
            exact, matched = self.referenced(facts.references)
            modules |= {file for file in exact if is_module(file) and not is_test_side(file)}
            # A module a test launches with -m needs its imports, as one it imports does.
            for name in facts.launched_modules:
                modules |= self.imported(name)
            # A conftest's patterns name the collected test files, which each
            # test declares; only its exact paths are inputs.
            data = exact if is_conftest(path) else exact | matched
        else:
            # Exact loader paths only: a loader's argument is bound module-wide,
            # so its patterns would make every sibling an edge.
            exact, _ = self.referenced({loader for loader in facts.loaders if not wildcard(loader)})
            modules |= {file for file in exact if is_module(file) and not is_test_side(file)}
            data = self.production_reads(path, facts.references, facts.siblings)
        modules.discard(path)
        return modules, data

    def production_reads(self, path, references, siblings):
        """The files production module `path` reads for every caller: exact files
        and siblings named with Path(__file__).with_name. Its tree reads are
        inputs only of tests that import the module themselves (tree_reads)."""
        exact = self.referenced({reference for reference in references if not wildcard(reference)})[0]
        named = {posixpath.join(posixpath.dirname(path), name) for name in siblings}
        trees = set().union(*(self.referenced({name + '/**'})[1] for name in named)) if named else set()
        return exact | (named & self.file_set) | trees

    def tree_reads(self, path):
        """Files that production module `path` reads through patterns."""
        if path not in self._tree_reads:
            references = self.facts[path].references
            self._tree_reads[path] = frozenset().union(
                *self.referenced({reference for reference in references if wildcard(reference)}))
        return self._tree_reads[path]

    def test_modules(self, path):
        """A test file's own extra libraries, which never propagate.

        Modules it names in a string, when it imports by name itself
        (import_module, __import__, a file loader, runpy), as with a
        parametrized list of modules; a bare string elsewhere is not an import.
        Test files it names by path, when it runs a child pytest: the child
        imports them.
        """
        tree, facts = self.trees.get(path), self.facts[path]
        if tree is None:
            return set()
        found = set()
        if any(isinstance(node, ast.Call) and leaf(node.func) in NAME_IMPORTERS
               for node in walk(tree)):
            for node in walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and MODULE_NAME.fullmatch(node.value):
                    found |= self.imported(node.value)
        if facts.child_pytest:
            exact, matched = self.referenced(facts.references)
            found |= {file for file in exact | matched if is_test(file)}
        found.discard(path)
        return found

    def run_by_path(self, closure):
        """Modules the code in `closure` runs: .py paths production code launches,
        `-m` modules it launches, and the imports of Python scripts it launches
        or a test names. A .py name in a process-name table or a log line is a
        name, not an input. On a library these edges would close cycles across
        packages; on the test target they never propagate."""
        found = set()
        for module in closure:
            if module in self.facts:
                found |= self.runs(module)
        return found - set(closure)

    def runs(self, module):
        """What one module runs, as run_by_path counts it."""
        if module not in self._runs:
            facts, found = self.facts[module], set()
            # Production code counts only paths that reach a launch; test code
            # names its scripts directly.
            references = facts.references if is_test_side(module) else facts.launched
            exact = self.referenced({reference for reference in references if not wildcard(reference)})[0]
            if not is_test_side(module):
                found |= {file for file in exact if is_module(file) and not is_test_side(file)}
                for name in facts.launched_modules:
                    found |= self.imported(name)
            for script in exact & set(self.scripts):
                found.add(script)
                names = self.facts[script].names
                found |= set().union(*(self.imported(name) for name in names)) if names else set()
            self._runs[module] = frozenset(found)
        return self._runs[module]

    def test_data(self, path, modules):
        """A test file's own extra inputs: the tree reads of the production
        modules it imports directly, so the test that exercises a scanner has
        what it scans."""
        return set().union(*(self.tree_reads(module) for module in modules if not is_test_side(module))) \
            if modules else set()


def sibling_names(tree):
    """Literal file names passed to .with_name(...); gate_paths keeps only their directory."""
    return {node.args[0].value for node in walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'with_name' and node.args
            and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)}


class Graph:
    """Library edges and data for every module, and each test file's inputs."""

    def __init__(self, evidence):
        self.evidence = evidence
        self.edges, self.data = {}, {}
        for path in evidence.modules:
            self.edges[path], self.data[path] = evidence.of(path)
        self._reach = {}

    def reach(self, start):
        """Every module `start` imports, transitively, itself included."""
        if start not in self._reach:
            seen, pending = {start}, [start]
            while pending:
                for module in self.edges.get(pending.pop(), ()):
                    if module not in seen:
                        seen.add(module)
                        pending.append(module)
            self._reach[start] = frozenset(seen)
        return self._reach[start]

    def closure(self, starts):
        found = set()
        for start in starts:
            found |= self.reach(start)
        return found

    def launched(self, closure):
        """Modules launched from `closure`, and from what they launch, in turn."""
        found, reached = set(), set(closure)
        while True:
            new = self.evidence.run_by_path(reached) - found
            if not new:
                return found
            found |= new
            reached |= self.closure(new)

    def programs(self, closure):
        """The files of programs launched from `closure`: each launched module's
        own import closure and data, file by file."""
        found = set()
        for module in self.launched(closure):
            for member in self.reach(module):
                found.add(member)
                found |= self.data.get(member, set())
        return found

    def conftests_for(self, test):
        return conftest_chain(test, self.edges)

    def test_inputs(self, test):
        """Everything one test target holds, by role."""
        conftests = self.conftests_for(test)
        deps = self.evidence.test_modules(test)
        library = self.closure([test, *conftests])
        modules = library | self.closure(deps)
        data = set().union(*(self.data[module] for module in modules))
        return {'conftests': conftests, 'deps': deps, 'modules': modules, 'data': data,
                'own_data': self.evidence.test_data(test, self.edges[test]), 'programs': self.programs(modules)}

    def test_files(self, test):
        """Every repository file a test target needs: a change to any of them selects it."""
        inputs = self.test_inputs(test)
        return inputs['modules'] | inputs['data'] | inputs['own_data'] | inputs['programs']

    def cycles(self):
        return [component for component in strongly_connected(self.evidence.modules, self.edges)
                if len(component) > 1]


def graph_record(graph, population):
    """The graph as JSON: library edges and data per module, each test's own
    inputs, import cycles and the uncertain list."""
    population = set(population)
    tests = {}
    for path in graph.evidence.modules:
        if is_test(path):
            inputs = graph.test_inputs(path)
            tests[path] = {'conftests': inputs['conftests'], 'deps': sorted(inputs['deps']),
                           'own_data': sorted(inputs['own_data']), 'programs': sorted(inputs['programs']),
                           'in_population': path in population}
    return {
        'schema': 'prismaquant.bazel_graph.v1',
        'modules': graph.evidence.modules,
        'scripts': graph.evidence.scripts,
        'sys_path_roots': sorted(graph.evidence.sys_path_roots),
        'edges': {path: sorted(graph.edges[path]) for path in graph.evidence.modules},
        'data': {path: sorted(files) for path, files in graph.data.items() if files},
        'cycles': graph.cycles(),
        'tests': tests,
        'uncertain': {path: sorted(reasons) for path, reasons in sorted(graph.evidence.uncertain.items())},
    }


class GenerationError(RuntimeError):
    """The evidence can't become a valid Bazel graph."""


def relative(path, package):
    return path[len(package) + 1:] if package else path


def library_name(path):
    stem = relative(path, package_of(path)).removesuffix('.py')
    # The test target takes the file's name; its importable module needs another.
    return stem + '_lib' if is_test(path) else stem


def hosted_plan(root, files, gate_select=None):
    """The hosted gate's full-mode plan: who runs, who doesn't and why."""
    gate_select = gate_select or GATE_SELECT
    root = Path(root)
    index = {path: (set(), set()) for path in files if path.startswith('tests/') and path.endswith('.py')}
    durations = json.loads((root / 'tools/ci/gate_durations.json').read_text())
    plan = gate_select.create_plan(root, [], durations=durations,
                                   quarantine=json.loads((root / 'tools/ci/gate_quarantine.json').read_text()),
                                   index=index, mode='full')
    return plan, durations['files']


def test_attributes(path, plan, durations):
    """args, tags, shard_count and timeout for one test file."""
    selected = {row['path'] for row in plan['selected']}
    attributes = {}
    if path in selected:
        attributes['args'] = ['--deselect=' + node for node in plan['deselect']
                              if node.split('::', 1)[0] == path]
    else:
        reasons = [row for row in plan['deferred'] if row['path'] == path]
        if not reasons:
            raise GenerationError(f'{path} is neither in the gate population nor deferred')
        tags = ['manual']
        for row in reasons:
            tags.append(f"gate-lane-{row['lane']}" if row.get('lane') else f"gate-issue-{row['issue']}")
        attributes['tags'] = sorted(set(tags))
    seconds = durations.get(path)
    if seconds is not None and seconds > SHARD_SECONDS:
        shards = min(MAX_SHARDS, math.ceil(seconds / SHARD_SECONDS))
        attributes['shard_count'] = shards
        seconds /= shards
    attributes['timeout'] = 'long' if seconds is not None and seconds * 3 > 300 else 'moderate'
    return attributes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', 1)[0])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--graph-json', type=Path, help='write the dependency graph here')
    parser.add_argument('--uncertain', action='store_true',
                        help='list modules whose launches, sys.path entries or dynamic imports can\'t be followed')
    args = parser.parse_args(argv)
    files = repository_files(args.root)
    graph = Graph(Evidence(args.root, files))
    if args.uncertain:
        for path, reasons in sorted(graph.evidence.uncertain.items()):
            print(f"{path}\t{'; '.join(sorted(reasons))}")
    if args.graph_json:
        plan, _ = hosted_plan(args.root, files)
        record = graph_record(graph, [row['path'] for row in plan['selected']])
        args.graph_json.write_text(json.dumps(record, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
