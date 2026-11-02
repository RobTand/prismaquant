"""Shared, stdlib-only graph-impact core. Vendor this file byte-identically.

Input contract: ``tests: {test_id: set(source_file)}``, ``edges: set((from, to))``,
and optional ``loc: {source_file: lines}``. Names are opaque, including Bazel
labels and Rust crates. With loc, its keys also declare unreached source files;
without it, source nodes are edge endpoints minus test IDs, plus test reach.
Zero-line files count in U but contribute zero weight to LOC-weighted U.

``measure(tests, edges, loc=None)`` returns a JSON-serializable measurement.
``compare(base, head, baseline, changed_tests=(), allowlist=None)`` returns
failures. Base/head are measurements, base may be None. Adapters identify
changed test bodies and may add numeric figures and ``edge_groups`` to a
measurement. Baselines store figures, tolerances, the reach budget, and frozen
edge-group exceptions, never a full graph. ``baseline_record`` accepts extra
figure tolerances for migrating an existing check. ``check`` reads caller-given
baseline/allowlist paths, always prints the summary, and returns an exit code.
Adapters own graph discovery, git, source classification, and external blame.
``explain(test_id)`` may return bazel-diff explain's Mermaid output, which check
prints on failure. The built-in shortest path supports pre-Bazel bridges.

Exceptions use exact diagnostic keys: ``reach:<test>``, ``edge:<group>:<a> ->
<b>``, ``figure:<name>``, and ``scc:<comma-separated sorted members>``. Reasons
must be nonempty, and allowlist_errors rejects new entries or changed reasons.
Update graph_impact.sha256 only when this core changes in all consuming repos.
"""
from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import statistics
import sys

SCHEMA = 'graph-impact.v1'
DEFAULT_TOLERANCES = {'u_pct': 1.0, 'loc_u_pct': 1.0, 'scc_max': 0}


def components(nodes, edges):
    """Iterative Tarjan SCCs; stable results, including isolated nodes."""
    adjacency = {node: set() for node in nodes}
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set())
    index, low, active, stack, result = {}, {}, set(), [], []
    counter = 0
    for start in sorted(adjacency):
        if start in index:
            continue
        work = [(start, iter(sorted(adjacency[start])))]
        index[start] = low[start] = counter
        counter += 1
        stack.append(start)
        active.add(start)
        while work:
            node, children = work[-1]
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    active.add(child)
                    work.append((child, iter(sorted(adjacency[child]))))
                    break
                if child in active:
                    low[node] = min(low[node], index[child])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[node])
                if low[node] == index[node]:
                    component = []
                    while True:
                        member = stack.pop()
                        active.remove(member)
                        component.append(member)
                        if member == node:
                            break
                    result.append(sorted(component))
    return sorted(result)


def measure(tests, edges, loc=None):
    """Compute reach, fanout, U, weighted U, SCCs and reach distribution."""
    reach = {test: sorted(set(files)) for test, files in sorted(tests.items())}
    edges = set(map(tuple, edges))
    nodes = {node for edge in edges for node in edge} | set(reach)
    reached = set().union(*map(set, reach.values())) if reach else set()
    sources = (set(loc) if loc is not None else nodes - reach.keys()) | reached
    weights = {source: loc.get(source, 0) if loc is not None else 1 for source in sorted(sources)}
    if any(not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0 for n in weights.values()):
        raise ValueError('LOC must be finite nonnegative line counts')
    dependents = dict.fromkeys(sorted(sources), 0)
    for files in reach.values():
        for source in files:
            dependents[source] += 1
    count = len(reach)
    denominator = count * len(sources)
    total_loc = sum(weights.values())
    sccs = components(nodes | sources, edges)
    largest = max(sccs, key=lambda row: (len(row), row), default=[])
    sizes = sorted(map(len, reach.values()))
    return {'reach': reach, 'test_count': count, 'source_files': len(sources),
            'edges': [list(edge) for edge in sorted(edges)], 'loc': weights,
            'dependents': dependents,
            'u_pct': 100 * sum(dependents.values()) / denominator if denominator else 0.0,
            'loc_u_pct': (100 * sum(weights[p] * n for p, n in dependents.items()) / (count * total_loc)
                          if count and total_loc else 0.0),
            'scc_max': len(largest), 'scc_members': largest, 'components': sccs,
            'reach_median': statistics.median(sizes) if sizes else 0,
            'reach_p90': sizes[max(0, math.ceil(.9 * len(sizes)) - 1)] if sizes else 0,
            'reach_max': max(sizes, default=0),
            'edge_groups': {'test_to_test': sorted(f'{a} -> {b}' for a, b in edges if a in reach and b in reach)}}


def baseline_record(live, baseline=None, *, tolerances=None):
    """Record minimum figures and budget; reject losses before calling this."""
    limits = dict(baseline['tolerances'] if baseline else DEFAULT_TOLERANCES)
    limits.update(tolerances or {})
    figures = {key: live[key] for key in limits}
    groups = {key: sorted(set(rows)) for key, rows in live.get('edge_groups', {}).items()}
    budget = math.ceil(live['reach_p90'])
    if baseline:
        figures = {key: min(value, baseline['figures'].get(key, value)) for key, value in figures.items()}
        budget = min(budget, baseline['reach_budget'])
        groups = {key: sorted(set(groups.get(key, ())) & set(rows))
                  for key, rows in baseline.get('edge_groups', {}).items()} | {
                      key: rows for key, rows in groups.items() if key not in baseline.get('edge_groups', {})}
    return {'schema': SCHEMA, 'figures': figures, 'tolerances': limits,
            'reach_budget': budget, 'edge_groups': groups}


def regressions(live, baseline, allowlist=None):
    """Absolute figures and forbidden edges, also enforced without history."""
    allowlist = allowlist or {}
    errors = []
    for key, old in baseline['figures'].items():
        value = live[key]
        tolerance = baseline['tolerances'][key]
        if value > old + tolerance and f'figure:{key}' not in allowlist:
            errors.append(f'{key} rose from {old:g} to {value:g} (tolerance {tolerance:g})')
    for group, rows in live.get('edge_groups', {}).items():
        for edge in sorted(set(rows) - set(baseline.get('edge_groups', {}).get(group, ()))):
            if f'edge:{group}:{edge}' not in allowlist:
                errors.append(f'new {group} edge: {edge}')
    return errors


def stale_entries(live, baseline):
    """Require material gains to be recorded; routine fractional noise is tolerated."""
    errors = []
    for key, old in baseline['figures'].items():
        if live[key] < old - baseline['tolerances'][key]:
            errors.append(f'{key} fell from {old:g} to {live[key]:g}; record the gain')
    for group, rows in baseline.get('edge_groups', {}).items():
        for edge in sorted(set(rows) - set(live.get('edge_groups', {}).get(group, ()))):
            errors.append(f'{group} edge is gone: {edge}; record the gain')
    if math.ceil(live['reach_p90']) < baseline['reach_budget']:
        errors.append(f'reach p90 fell to {live["reach_p90"]}; record the reach budget gain')
    return errors


def baseline_errors(current, previous):
    """Prevent editing the baseline to hide a regression."""
    errors = []
    if current.get('schema') != SCHEMA:
        errors.append('unsupported baseline schema')
    if previous is None:
        return errors
    if current['tolerances'] != previous['tolerances']:
        errors.append('baseline tolerances changed')
    for key, value in previous['figures'].items():
        if key not in current['figures'] or current['figures'][key] > value:
            errors.append(f'baseline figure {key} may only shrink')
    if current['reach_budget'] > previous['reach_budget']:
        errors.append('reach budget may only shrink')
    for group, rows in current.get('edge_groups', {}).items():
        if set(rows) - set(previous.get('edge_groups', {}).get(group, ())):
            errors.append(f'baseline {group} exceptions may only shrink')
    return errors


def allowlist_errors(current, previous):
    errors = []
    for key, reason in sorted(current.items()):
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f'allowlist exception {key} needs a reason')
        if key not in previous or previous[key] != reason:
            errors.append(f'allowlist may only shrink: {key}')
    return errors


def shortest_hub_path(live, test):
    """Shortest path to the most distant reachable hub, with stable ties.

    A hub has maximum fanout among the reachable source files. Among these,
    choose the greatest shortest-path distance from the test. If the graph
    source has no connecting evidence, say so instead of inventing an edge.
    """
    adjacency = {}
    for source, target in live.get('path_edges', live['edges']):
        adjacency.setdefault(source, set()).add(target)
    parent, depth, pending = {test: None}, {test: 0}, deque([test])
    while pending:
        node = pending.popleft()
        for target in sorted(adjacency.get(node, ())):
            if target not in parent:
                parent[target], depth[target] = node, depth[node] + 1
                pending.append(target)
    reachable = set(live['reach'][test]) & depth.keys()
    if not reachable:
        return f'{test} (no connecting edge evidence available)'
    hub = max(reachable, key=lambda p: (live['dependents'][p], depth[p], p))
    path = [hub]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return ' → '.join(reversed(path))


def _attribution(base, head, added, *, members=None):
    if members is not None:
        selected = [(a, b) for a, b in added if a in members and b in members]
    else:
        # An added edge can grow fanout when a head test reaches its source and
        # now reaches its target. Discard edges unrelated to any growing reach.
        growing = {test: set(files) - set(base['reach'].get(test, ())) for test, files in head['reach'].items()}
        selected = [(a, b) for a, b in added if any(
            b in gained and (a == test or a in head['reach'][test] or a not in head['dependents'])
            for test, gained in growing.items())]
    edges = ', '.join(f'{a} -> {b}' for a, b in selected) or 'none (population or source-set change)'
    growth = sorted(((n - base['dependents'].get(p, 0), p) for p, n in head['dependents'].items()
                     if n > base['dependents'].get(p, 0)), key=lambda row: (-row[0], row[1]))[:5]
    files = ', '.join(f'{p} +{n}' for n, p in growth) or 'none'
    return f'; added edges: {edges}; largest fanout growth: {files}'


def compare(base, head, baseline, *, changed_tests=(), allowlist=None, record_gains=False):
    """All absolute and base-to-head failure rules. Does not perform IO."""
    allowlist = allowlist or {}
    errors = regressions(head, baseline, allowlist)
    if not record_gains:
        errors += stale_entries(head, baseline)
    changed = set(head['reach']) - (base['reach'].keys() if base else set())
    changed |= set(changed_tests)
    if base is None:
        changed = set(head['reach'])
    for test in sorted(changed & head['reach'].keys()):
        size = len(head['reach'][test])
        if size > baseline['reach_budget'] and f'reach:{test}' not in allowlist:
            errors.append(f'{test} reach {size} exceeds budget {baseline["reach_budget"]}: '
                          + shortest_hub_path(head, test))
    if base is None:
        return errors
    added = set(map(tuple, head['edges'])) - set(map(tuple, base['edges']))
    for key in ('u_pct', 'loc_u_pct'):
        if head[key] > base[key] + baseline['tolerances'].get(key, 0) and f'figure:{key}' not in allowlist:
            errors.append(f'{key} rose from {base[key]:g} to {head[key]:g}' + _attribution(base, head, added))
    # A new SCC must be wholly contained in a former component. Checking only
    # the largest size misses mergers and growth of smaller components.
    old_component = {node: frozenset(row) for row in base['components'] for node in row}
    for component in head['components']:
        members = set(component)
        if len(members) <= 1 or any(members <= old_component.get(node, frozenset()) for node in members):
            continue
        if 'scc:' + ','.join(component) not in allowlist:
            errors.append(f'SCC grew or merged ({len(component)} nodes): {", ".join(component)}'
                          + _attribution(base, head, added, members=members))
    for group, rows in head.get('edge_groups', {}).items():
        for edge in sorted(set(rows) - set(base.get('edge_groups', {}).get(group, ()))):
            message = f'new {group} edge: {edge}'
            if f'edge:{group}:{edge}' not in allowlist and message not in errors:
                errors.append(message)
    return errors


def summary(base, head, budget):
    tests = sorted(set(head['reach']) - (base['reach'].keys() if base else set()))
    sizes = ', '.join(str(len(head['reach'][test])) for test in tests) or 'none'
    relation = '≤' if all(len(head['reach'][test]) <= budget for test in tests) else '>'
    added = set(map(tuple, head['edges'])) - (set(map(tuple, base['edges'])) if base else set())
    hubs = {p for p, n in head['dependents'].items() if n > head['test_count'] / 2}
    hub_edges = sum(target in hubs for _, target in added)
    last = f'{hub_edges} to a hub' if hub_edges else 'none to a hub'
    prior_u = f'{base["u_pct"]:.1f}%' if base else 'no base available'
    prior_scc = str(base['scc_max']) if base else 'no base available'
    return (f'U {prior_u} → {head["u_pct"]:.1f}%, SCC max {prior_scc} → {head["scc_max"]}, '
            f'new tests {len(tests)} (reach {sizes} {relation} budget {budget}), new edges {len(added)}, {last}')


def check(base, head, baseline_path, allowlist_path=None, *, changed_tests=(),
          base_allowlist=None, previous_baseline=None, write_baseline=False, explain=None):
    """Run the check with caller-owned paths, printing a summary on pass or fail."""
    baseline_path = Path(baseline_path)
    baseline = json.loads(baseline_path.read_text())
    allowlist = json.loads(Path(allowlist_path).read_text()) if allowlist_path else {}
    print(summary(base, head, baseline['reach_budget']), flush=True)
    errors = allowlist_errors(allowlist, base_allowlist if base_allowlist is not None else allowlist)
    errors += baseline_errors(baseline, previous_baseline)
    errors += compare(base, head, baseline, changed_tests=changed_tests, allowlist=allowlist,
                      record_gains=write_baseline)
    if errors:
        if write_baseline:
            print('refusing to record a loss:', file=sys.stderr)
        print('\n'.join(errors), file=sys.stderr)
        if explain is not None:
            affected = (set(head['reach']) - (base['reach'].keys() if base else set())) | set(changed_tests)
            for test in sorted(affected & head['reach'].keys()):
                diagram = explain(test)
                if diagram:
                    print(f'```mermaid\n{diagram.rstrip()}\n```', file=sys.stderr)
        return 1
    if write_baseline:
        baseline_path.write_text(json.dumps(baseline_record(head, baseline), indent=2) + '\n')
    return 0


def _bazel_inputs(dependencies, test_targets, source_labels, manual_tests=(), loc=None):
    """Project a label graph onto source reach without losing its blame edges."""
    edges = {(source, target) for source, targets in dependencies.items() for target in targets}
    labels = set(dependencies) | {target for _, target in edges}
    sources = set(source_labels)
    tests = {}
    for test in sorted(set(test_targets) - set(manual_tests)):
        if test not in labels:
            raise ValueError(f'test target missing from dependency graph: {test}')
        seen, pending = {test}, [test]
        while pending:
            for target in dependencies.get(pending.pop(), ()):
                if target not in seen:
                    seen.add(target)
                    pending.append(target)
        tests[test] = seen & sources
    weights = {source: loc.get(source, 0) if loc is not None else 1 for source in sorted(sources)}
    return tests, edges, weights


def read_deps(path, test_targets, source_labels, *, manual_tests=(), loc=None):
    """Read bazel-diff v42 --depEdgesFile (label -> direct dependency labels).

    test_targets comes from ``tests(//...)``; remove manual-tagged targets with
    manual_tests, or filter them in the caller's query. source_labels comes
    from --includeTargetType's SourceFile hashes or a source-file query. Do
    not guess source types from extensions: generated files and rules can
    share those names. If deps.json omits source-file labels, use read_query
    on one streamed_jsonproto ``deps(tests(//...))`` query per revision.
    """
    dependencies = json.loads(Path(path).read_text())
    if not isinstance(dependencies, dict) or any(
            not isinstance(source, str) or not isinstance(targets, list)
            or any(not isinstance(target, str) for target in targets)
            for source, targets in dependencies.items()):
        raise ValueError('deps.json must map labels to lists of dependency labels')
    labels = set(dependencies) | {target for targets in dependencies.values() for target in targets}
    if source_labels and not (set(source_labels) & labels):
        raise ValueError('deps.json has no source-file labels; use the Bazel query fallback')
    return _bazel_inputs(dependencies, test_targets, source_labels, manual_tests, loc)


def read_query(path, *, test_targets=None, loc=None):
    """Fallback: Bazel query --output=streamed_jsonproto over deps(tests(//...)).

    One record per line. Rule inputs retain the paths through libraries;
    sourceFile records identify leaves; manual tests remain as dependencies
    but are excluded from the test population. The optional explicit test
    population supports custom rules whose class does not end in _test.
    """
    dependencies, sources, tests, manual = {}, set(), set(), set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if 'sourceFile' in record:
            name = record['sourceFile']['name']
            sources.add(name)
            dependencies.setdefault(name, [])
        elif 'rule' in record:
            rule = record['rule']
            name = rule['name']
            dependencies[name] = rule.get('ruleInput', [])
            if rule.get('ruleClass', '').endswith('_test'):
                tests.add(name)
            if any(attribute.get('name') == 'tags' and 'manual' in attribute.get('stringListValue', [])
                   for attribute in rule.get('attribute', [])):
                manual.add(name)
    return _bazel_inputs(dependencies, tests if test_targets is None else test_targets, sources, manual, loc)
