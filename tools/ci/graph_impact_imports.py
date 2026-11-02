#!/usr/bin/env python3
"""Legacy import evidence retained by graph_impact_pq's unified baseline.

The full check and CLI live in graph_impact_pq; this parser preserves the
previous metric's meaning while the full Bazel graph adds reach, U and SCCs.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = ('prismaquant/', 'tools/')
SCANNED_ROOTS = ('tests', 'prismaquant', 'tools', 'experiments', 'scripts', 'pqtel')

#: How far a summary figure may move without a baseline change. Adding an
#: ordinary test shifts the fractions by about 0.1 percentage points, so this
#: absorbs routine work while a structural regression still fails.
PERCENT_TOLERANCE = 1.0
MODULE_TOLERANCE = 2

# The Bazel graph generator owns the conftest rule; it also loads the gate's selector.
GRAPH_SPEC = importlib.util.spec_from_file_location('bazel_build_files_for_fanout',
                                                    ROOT / 'tools/bazel/build_files.py')
build_files = importlib.util.module_from_spec(GRAPH_SPEC)
GRAPH_SPEC.loader.exec_module(build_files)
gate_select = build_files.GATE_SELECT
is_conftest = build_files.is_conftest

# The start of an import statement on its own line. The statement is then
# read to its logical end and parsed, so parentheses, continuations and
# comments are Python's own business.
STATEMENT_START = re.compile(r'^[ \t]*((?:from[ \t]+\.*[\w.]*[ \t]+import|import[ \t]+[A-Za-z_]))', re.M)
# An import inside a one-line snippet, such as ``"import sys; import g3_lib"``
# handed to ``python -c``: it follows a quote, a semicolon or an escaped
# newline rather than starting a line.
SNIPPET_IMPORT = re.compile(r'''(?:['";]|\\n)[ \t]*(?:from[ \t]+([A-Za-z_][\w.]*)[ \t]+import[ \t]+([\w., \t]+)'''
                            r'''|import[ \t]+([\w., \t]+))''')
# ``pytest_plugins`` names modules pytest imports for every test under the conftest.
PLUGINS = re.compile(r'^pytest_plugins[ \t]*=', re.M)
DYNAMIC_IMPORT = re.compile(r'''\b(?:import_module|__import__)\(\s*(?:name\s*=\s*)?['"]([A-Za-z_][\w.]*)['"]''')
PY_LITERAL = re.compile(r'''['"]((?:[\w.-]+/)*[\w-][\w.-]*\.py)['"]''')


def is_source(path: str) -> bool:
    return path.startswith(SOURCE_ROOTS)


def is_test_module(path: str) -> bool:
    return path.startswith('tests/') and PurePosixPath(path).name.startswith('test_')


def names_of(path: str) -> set[str]:
    """The gate selector's names for a file, plus its dotted path from the repository root."""
    names = set(gate_select.module_names(path))
    parts = path.removesuffix('.py').split('/')
    if parts[-1] == '__init__':
        parts = parts[:-1]
    if parts and all(part.isidentifier() for part in parts):
        names.add('.'.join(parts))
    return names


def python_files(root: Path) -> list[str]:
    """Every module the graph may name: the scanned trees and a root conftest."""
    found = {path.relative_to(root).as_posix()
             for top in SCANNED_ROOTS if (root / top).is_dir()
             for path in (root / top).rglob('*.py')
             if '__pycache__' not in path.relative_to(root).parts}
    if (root / 'conftest.py').is_file():
        found.add('conftest.py')
    return sorted(found)


def _logical_statement(source: str, start: int) -> str:
    """The text of one logical line from ``start``, as Python would join it."""
    depth, index = 0, start
    while index < len(source):
        char = source[index]
        if char == '#':
            newline = source.find('\n', index)
            index = len(source) if newline < 0 else newline
            continue
        if char == '\\' and source.startswith('\n', index + 1):
            index += 2
            continue
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        elif char == ';' or (char == '\n' and depth <= 0):
            break
        index += 1
    return source[start:index]


def _package_prefix(path: str, level: int) -> str:
    package = list(PurePosixPath(path).parent.parts)
    return '.'.join(package[:len(package) - level + 1])


def imported_names(source: str, path: str = '') -> set[str]:
    """Dotted names a file imports, by statement, snippet or literal call."""
    names = set()
    for match in STATEMENT_START.finditer(source):
        try:
            tree = ast.parse(_logical_statement(source, match.start(1)))
        except SyntaxError:
            continue  # prose that happens to start with "import"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ''
                if node.level:
                    prefix = '.'.join(part for part in (_package_prefix(path, node.level), prefix) if part)
                if prefix:
                    names.add(prefix)
                names.update(f'{prefix}.{alias.name}' if prefix else alias.name
                             for alias in node.names if alias.name != '*')
    for match in SNIPPET_IMPORT.finditer(source):
        module, members, plain = match.groups()
        listed = [part.split()[0] for part in (members or plain).split(',') if part.split()]
        if module:
            names.add(module)
            names.update(f'{module}.{member}' for member in listed if member.isidentifier())
        else:
            names.update(name for name in listed if re.fullmatch(r'[A-Za-z_][\w.]*', name))
    names.update(DYNAMIC_IMPORT.findall(source))
    for match in PLUGINS.finditer(source):
        try:
            tree = ast.parse(_logical_statement(source, match.start()))
        except SyntaxError:
            continue
        names.update(node.value for node in ast.walk(tree)
                     if isinstance(node, ast.Constant) and isinstance(node.value, str))
    return names


def named_files(source: str) -> set[str]:
    """``.py`` paths written as string literals: files read or run by path."""
    return set(PY_LITERAL.findall(source))


class ImportGraph:
    """Module edges, and the files each module names as leaves, for one tree."""

    def __init__(self, root: Path, evidence: str = 'imports'):
        if evidence not in {'imports', 'gate'}:
            raise ValueError('evidence must be imports or gate')
        self.root, self.evidence = Path(root), evidence
        self.files = python_files(self.root)
        self.known = set(self.files)
        self.aliases: dict[str, set[str]] = {}
        for path in self.files:
            for alias in names_of(path):
                self.aliases.setdefault(alias, set()).add(path)
        self.by_name: dict[str, set[str]] = {}
        for path in self.files:
            self.by_name.setdefault(PurePosixPath(path).name, set()).add(path)
        self.edges: dict[str, set[str]] = {}
        self.leaves: dict[str, set[str]] = {}
        for path in self.files:
            source = (self.root / path).read_text(encoding='utf-8', errors='replace')
            if evidence == 'imports':
                self.edges[path], self.leaves[path] = self._import_evidence(source, path)
            else:
                self.edges[path], self.leaves[path] = self._gate_evidence(source, path)

    def resolve(self, name: str) -> set[str]:
        """Every file a dotted name may load, the package ``__init__`` included."""
        parts, found = name.split('.'), set()
        for end in range(1, len(parts) + 1):
            found.update(self.aliases.get('.'.join(parts[:end]), ()))
        return found

    def _suffix_matches(self, literal: str) -> set[str]:
        tail = PurePosixPath(literal)
        return {path for path in self.by_name.get(tail.name, ())
                if path == literal or path.endswith('/' + literal.lstrip('./'))}

    def _import_evidence(self, source: str, path: str) -> tuple[set[str], set[str]]:
        edges = set().union(*(self.resolve(name) for name in imported_names(source, path)))
        named = set().union(*(self._suffix_matches(literal) for literal in named_files(source)))
        edges |= {file for file in named if is_source(file)}
        return edges - {path}, named - edges - {path}

    def _gate_evidence(self, source: str, path: str) -> tuple[set[str], set[str]]:
        test_side = path.startswith('tests/') or is_conftest(path)
        try:
            imports, references = gate_select.dependencies(source, path, production=not test_side)
            if not test_side:
                references = references | gate_select.gate_paths.references(ast.parse(source), path)
        except (SyntaxError, ValueError):
            return set(), set()
        edges = set().union(*(self.resolve(name) for name in imports))
        if '**' in references:
            return edges - {path}, set(self.files)
        # An exact path to a module is a library edge.
        edges.update(reference for reference in references if reference in self.known)
        patterns = {reference for reference in references if '**' in reference}
        leaves = ({file for file in self.files if gate_select.gate_paths.gate_path_matches(file, patterns)}
                  if patterns else set())
        if not test_side:
            # Production code that names a test file reads it; it does not import it.
            leaves |= {file for file in edges if file.startswith('tests/') or is_conftest(file)}
            edges = {file for file in edges if not (file.startswith('tests/') or is_conftest(file))}
        return edges - {path}, leaves - {path}

    def conftests_for(self, test: str) -> list[str]:
        return build_files.conftest_chain(test, self.edges)

    def source_closures(self) -> tuple[list[str], dict[str, int]]:
        """Each module's transitive source dependencies, as a bit mask over source modules.

        A fixed point over the edges: cycles need no special case, and a
        leaf contributes its own bit without its imports.
        """
        sources = [path for path in self.files if is_source(path)]
        bit = {path: 1 << index for index, path in enumerate(sources)}
        own = {path: bit.get(path, 0) | sum(bit.get(leaf, 0) for leaf in self.leaves[path])
               for path in self.files}
        masks = dict(own)
        changed = True
        while changed:
            changed = False
            for path in self.files:
                mask = own[path]
                for target in self.edges[path]:
                    mask |= masks[target]
                if mask != masks[path]:
                    masks[path], changed = mask, True
        return sources, masks


def gate_population(root: Path) -> list[str]:
    """The gate's full-mode test files, by the gate's own definition."""
    root = Path(root)
    names = [path for path in python_files(root) if path.startswith('tests/')]
    plan = gate_select.create_plan(
        root, [], durations=json.loads((root / 'tools/ci/gate_durations.json').read_text()),
        quarantine=json.loads((root / 'tools/ci/gate_quarantine.json').read_text()),
        index={name: (set(), set()) for name in names}, mode='full')
    return [row['path'] for row in plan['selected']]


def nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)] if ordered else 0.0


def measure_fanout(root: Path = ROOT, evidence: str = 'imports', population: list[str] | None = None) -> dict:
    """The fan-out of every source module over the gate population, and its summary."""
    graph = ImportGraph(root, evidence)
    population = gate_population(root) if population is None else population
    sources, masks = graph.source_closures()
    dependents = dict.fromkeys(sources, 0)
    for test in population:
        mask = masks[test]
        for conftest in graph.conftests_for(test):
            mask |= masks[conftest]
        index = 0
        while mask:
            if mask & 1:
                dependents[sources[index]] += 1
            mask >>= 1
            index += 1
    fractions = [count / len(population) for count in dependents.values()] if population else []
    conftests = [path for path in graph.files if is_conftest(path)]
    conftest_mask = 0
    for conftest in conftests:
        conftest_mask |= masks[conftest]
    return {
        'evidence': evidence,
        'tests': len(population),
        'source_modules': len(sources),
        'median_selected_pct': round(100 * nearest_rank(fractions, 0.5), 1),
        'p90_selected_pct': round(100 * nearest_rank(fractions, 0.9), 1),
        'modules_over_half': sum(fraction > 0.5 for fraction in fractions),
        'conftest_source_modules': bin(conftest_mask).count('1'),
        'conftest_source_imports': sorted(f'{conftest} -> {target}' for conftest in conftests
                                          for target in graph.edges[conftest] if is_source(target)),
        'test_to_test_imports': sorted(f'{path} -> {target}' for path in graph.files
                                       if path.startswith('tests/')
                                       for target in graph.edges[path] if is_test_module(target)),
        'dependents': dict(sorted(dependents.items(), key=lambda item: (-item[1], item[0]))),
    }
