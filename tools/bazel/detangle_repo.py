"""PrismaQuant graph and AST adapter for detangle_plan.

Only this module knows the repository's test population and graph generator.
Selection uses the generator's library edges, conftests, own data and launched
programs. Program edges are recomputed per module, so a cut also removes the
programs reachable only through that module. It never imports test modules.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

from tools.bazel import build_files as build


def module_name(path):
    return path.removesuffix('.py').removesuffix('/__init__').replace('/', '.')


def import_symbols(source, source_path, target_path):
    """Resolve relative imports, aliases, attribute uses, and owning functions."""
    tree = ast.parse(source)
    target = module_name(target_path)
    names, functions, import_lines, module_import = set(), set(), set(), False
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def function(node):
        names = []
        while node in parent:
            node = parent[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(node.name)
        return '.'.join(reversed(names)) or '<module>'

    def uses(binding, scope):
        return [node for node in ast.walk(scope) if isinstance(node, ast.Name) and node.id == binding
                and isinstance(node.ctx, ast.Load)]

    module_bindings, imported = {}, []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            prefix = node.module or ''
            if node.level:
                package = source_path.removesuffix('.py').split('/')[:-1]
                prefix = '.'.join(package[:len(package) - node.level + 1] + ([prefix] if prefix else []))
            for alias in node.names:
                if prefix == target:
                    names.add(alias.name)
                    imported.append((node, alias.asname or alias.name))
                elif prefix + '.' + alias.name == target:
                    module_bindings[alias.asname or alias.name] = (target, node)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == target:
                    module_bindings[alias.asname or alias.name.split('.')[0]] = (alias.name if alias.asname else '', node)
    for binding, (prefix, node) in module_bindings.items():
        module_import = True
        imported.append((node, binding))
        matched = False
        for attr in ast.walk(tree):
            if not isinstance(attr, ast.Attribute):
                continue
            value = ast.unparse(attr)
            qualified = prefix + value[len(binding):] if prefix else value
            if qualified.startswith(target + '.'):
                names.add(qualified[len(target) + 1:].split('.')[0])
                functions.add(function(attr))
                matched = True
        if not matched:
            names.add('*module*')
    for node, binding in imported:
        import_lines.update(range(node.lineno, node.end_lineno + 1))
        owner = function(node)
        if owner != '<module>':
            functions.add(owner)
        functions.update(function(use) for use in uses(binding, tree))
    return {'source': source_path, 'target': target_path, 'symbols': sorted(names),
            'functions': sorted(f for f in functions if f != '<module>'),
            'module_uses': '<module>' in functions,
            'has_module_import': module_import, 'import_lines': sorted(import_lines)}


def symbol_lines(tree, names):
    """Include top-level definitions referenced by the moved definitions."""
    definitions = {}
    if tree is None:
        return 0, []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        definitions[name.id] = node
    pending, seen, lines = list(names), set(), set()
    while pending:
        name = pending.pop()
        if name in seen or name not in definitions:
            continue
        seen.add(name)
        node = definitions[name]
        start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', ())])
        lines.update(range(start, node.end_lineno + 1))
        pending.extend(n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id not in seen)
    return len(lines), sorted(seen)


def adapt(root, files=None):
    from tools.bazel.detangle_plan import Candidate, Selection, feedback_cut
    root = Path(root)
    files = files if files is not None else build.repository_files(root)
    evidence = build.Evidence(root, files)
    graph = build.Graph(evidence)
    gate, hosted = build.hosted_plan(root, files)
    population = sorted(row['path'] for row in gate['selected'])
    roots, extras = {}, {}
    for test in population:
        roots[test] = {test, *graph.conftests_for(test), *evidence.test_modules(test)}
        extras[test] = evidence.test_data(test, graph.edges[test])
    launches = {p: set(evidence.runs(p)) for p in evidence.modules}
    selection = Selection(graph.edges, roots, graph.data, extras, launches)
    # Keep candidates in the selected reach. Data-only Python files do not grow
    # imports: the graph generator treats their source as data, too.
    reachable, pending = set().union(*roots.values()), list(set().union(*roots.values()))
    combined = selection.combined()
    while pending:
        for target in combined.get(pending.pop(), ()):
            if target not in reachable:
                reachable.add(target)
                pending.append(target)
    edge_info, edge_cost = {}, {}
    target_users = defaultdict(list)
    for source in sorted(reachable):
        tree = evidence.trees.get(source)
        if tree is None or build.is_test(source):
            continue
        text = (root / source).read_text(errors='replace')
        for target in sorted(graph.edges.get(source, ())):
            info = import_symbols(text, source, target)
            if not info['symbols']:
                info['symbols'] = ['<loader or package side effect>']
            size, closure = symbol_lines(evidence.trees.get(target), info['symbols'])
            info['moved_symbol_closure'] = closure
            info['definition_lines'] = size
            target_users[target].append(info)
            edge_info[source, target] = info
    for edge, info in edge_info.items():
        # Estimate edits to the definition and every known importer of these
        # symbols, not just the edge being removed.
        symbols = set(info['symbols'])
        importer_lines = sum(max(1, len(other['import_lines'])) for other in target_users[edge[1]]
                             if symbols & set(other['symbols']))
        edge_cost[edge] = max(2, info['definition_lines'] + importer_lines + len(info['import_lines']))
    candidates = []
    for (source, target), info in edge_info.items():
        if target.endswith('/__init__.py'):
            kind = 'split this re-export hub'
        elif len(info['functions']) == 1 and not info['module_uses']:
            kind = 'make this import local to the one function'
        else:
            kind = 'move these symbols down'
        candidates.append(Candidate(kind, ((source, target, 'import'),), edge_cost[source, target], (info,)))
    for source in sorted(reachable):
        if build.is_test(source):
            continue
        tree = evidence.trees.get(source)
        if tree is None:
            continue
        literals = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        for target in sorted(graph.data.get(source, ())):
            matching = [n for n in literals if target in n.value or n.value == Path(target).name]
            # A pattern expansion is not a literal edge that can be removed on
            # its own. Only offer exact literal evidence.
            if not matching:
                continue
            info = {'source': source, 'target': target, 'symbols': sorted({n.value for n in matching}),
                    'literal_lines': sorted({n.lineno for n in matching})}
            candidates.append(Candidate('drop this path literal', ((source, target, 'data'),),
                                        len(info['literal_lines']) + 1, (info,)))
    sccs = []
    for component in graph.cycles():
        nodes = set(component) & reachable
        if len(nodes) < 2:
            continue
        weights = {edge: edge_cost.get(edge, 10000) for edge in edge_info if edge[0] in nodes and edge[1] in nodes}
        back = feedback_cut(nodes, graph.edges, weights)
        cuts = set(back)
        if cuts:
            candidates.append(Candidate('SCC feedback cut', tuple((s, t, 'import') for s, t in sorted(cuts)),
                                        sum(edge_cost.get(e, 10000) for e in cuts),
                                        tuple(edge_info.get(e, {'source': e[0], 'target': e[1], 'symbols': ['<unresolved>']}) for e in sorted(cuts))))
        # ELS order plus deterministic graph traversals give partitions of each
        # SCC. Offer both directions around several large partitions. Unlike a
        # single-edge score these retain complementary cuts across a cycle.
        partitions = []
        orders = [sorted(nodes)]
        for start in sorted(nodes)[:8]:
            seen, stack, order = set(), [start], []
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                order.append(node)
                stack.extend(sorted((graph.edges.get(node, set()) & nodes) - seen, reverse=True))
            orders.append(order)
        for order in orders:
            for numerator in (1, 2, 3):
                part = set(order[:max(1, len(order) * numerator // 4)])
                if len(part) == len(nodes):
                    continue
                partitions.append(part)
        for part in partitions:
            for left, right in ((part, nodes - part), (nodes - part, part)):
                crossing = {(s, t) for s in left for t in graph.edges.get(s, ()) if t in right}
                if not crossing or not crossing <= edge_info.keys():
                    continue
                candidates.append(Candidate('SCC sub-cut', tuple((s, t, 'import') for s, t in sorted(crossing)),
                                            sum(edge_cost[e] for e in crossing),
                                            tuple(edge_info[e] for e in sorted(crossing))))
        sccs.append({'size': len(nodes), 'members': sorted(nodes), 'feedback_edges': len(back)})
    unique = {c.edges: c for c in candidates}
    return selection, list(unique.values()), hosted, {'population': len(population), 'sccs': sccs,
             'adapter': 'PrismaQuant Evidence/Graph', 'package_roots': list(build.PACKAGES), 'test_roots': ['tests/'],
             'local_import_note': 'Localizing alone does not remove a conservative Bazel edge; scores require a target/symbol boundary cut.'}
