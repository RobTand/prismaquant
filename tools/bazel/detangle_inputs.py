"""Map lock packages and gate configuration to graph inputs for replay."""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
import re
import tomllib


def wheel_name(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def packages(text):
    result = defaultdict(list)
    for package in tomllib.loads(text).get('package', []):
        result[wheel_name(package['name'])].append(package)
    return dict(result)


def changed_wheels(before, after):
    a, b = packages(before), packages(after)
    # Compare complete records, including wheel hashes, source and dependency
    # markers. Same-version rebuilds and removed wheels still matter.
    return {name for name in a.keys() | b.keys() if a.get(name) != b.get(name)}


def mapped_changes(merge, map_lock):
    changed = set(merge['changed'])
    if map_lock and 'uv.lock' in changed:
        if 'changed_wheels' not in merge:
            raise ValueError('uv.lock replay requires changed_wheels from its historical diff')
        changed.remove('uv.lock')
        changed.update('wheel:' + wheel_name(name) for name in merge['changed_wheels'])
    return changed


# Distribution names whose public imports differ from their normalized name.
IMPORT_NAMES = {'pillow': ('PIL',), 'pyyaml': ('yaml',), 'protobuf': ('google',),
                'scikit-learn': ('sklearn',), 'opencv-python': ('cv2',),
                'tessera-quant': ('tessera',), 'huggingface-hub': ('huggingface_hub',),
                'pytest-cov': ('pytest_cov',), 'pytest-timeout': ('pytest_timeout',),
                'pytest-xdist': ('xdist',), 'setuptools': ('setuptools', 'pkg_resources')}


def wheel_graph(root, files, lock_text):
    """Build distribution dependencies and source imports without executing code.

    Import roots are conservative. Lock dependencies include all versions and
    markers. Pytest plugins run for every test, regardless of direct imports.
    """
    locked = packages(lock_text)
    imports = defaultdict(set)
    edges = defaultdict(set)
    for name, records in locked.items():
        for module in IMPORT_NAMES.get(name, (name.replace('-', '_'),)):
            imports[module].add('wheel:' + name)
        for record in records:
            for dep in record.get('dependencies', []):
                edges['wheel:' + name].add('wheel:' + wheel_name(dep['name']))
    for path in sorted(files):
        if not path.endswith('.py') or not (Path(root) / path).is_file():
            continue
        try:
            tree = ast.parse((Path(root) / path).read_text(errors='replace'))
        except SyntaxError:
            continue
        trees = [tree]
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and 'import ' in node.value:
                try: trees.append(ast.parse(node.value))
                except SyntaxError: pass
        for tree in trees:
            modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import): modules.extend(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom) and not node.level and node.module: modules.append(node.module)
                elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                    func = ast.unparse(node.func)
                    if func in ('__import__', 'importlib.import_module', 'pytest.importorskip') and isinstance(node.args[0].value, str):
                        modules.append(node.args[0].value)
            for module in modules:
                edges[path].update(imports.get(module.split('.')[0], ()))
    return dict(edges), {'distribution_count': len(locked), 'import_aliases': IMPORT_NAMES,
                        'dependency_markers': 'union of all versions and markers',
                        'scope': 'AST imports including embedded Python and literal dynamic imports; observed Python coverage omits installed wheels'}


def add_gate_inputs(selection, executed, root, files, lock_text):
    """Attach wheel nodes to static reach and observed repository executions.

    Dynamic wheel selection uses imports of observed source files plus lock
    dependency closure. It is an estimate because the original coverage sweep
    retained only repository filenames, not installed distribution filenames.
    """
    from tools.bazel.detangle_plan import Selection
    wheels, metadata = wheel_graph(root, files, lock_text)
    edges = {source: set(targets) | wheels.get(source, set()) for source, targets in selection.edges.items()}
    for source, targets in wheels.items(): edges.setdefault(source, set()).update(targets)
    locked = packages(lock_text)
    gate_inputs = {'pytest.ini', 'pyproject.toml'}
    if (Path(root) / '.github/ci-constraints.txt').exists():
        gate_inputs.add('.github/ci-constraints.txt')
    for name in ('pytest', 'pytest-cov', 'pytest-timeout', 'pytest-xdist'):
        if wheel_name(name) in locked:
            gate_inputs.add('wheel:' + name)
    # pytest.ini configures every invocation. The gate's editable install reads
    # pyproject.toml and its constraints in every shard.
    extras = {t: set(selection.extras.get(t, ())) | gate_inputs for t in selection.tests}
    observed = {}
    for test, paths in executed.items():
        seen = set(paths) | gate_inputs
        visited = set()
        pending = [target for source in paths for target in wheels.get(source, ())]
        pending.extend(p for p in gate_inputs if p.startswith('wheel:'))
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            seen.add(node)
            pending.extend(wheels.get(node, ()))
        observed[test] = seen
    metadata['gate_input_consumers'] = {
        'pytest.ini': 'every pytest invocation',
        'pyproject.toml': 'editable install in every gate shard',
        '.github/ci-constraints.txt': 'pip constraint input in every gate shard'}
    return Selection(edges, selection.roots, selection.data, extras, selection.launches), observed, metadata
