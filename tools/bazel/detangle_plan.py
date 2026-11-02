#!/usr/bin/env python3
"""Rank symbol cuts by expected selected test seconds over first-parent merges.

The pure Selection/cost/ranking functions accept any repository's graph. The
PrismaQuant adapter lives in detangle_repo.py. For Tessera, provide a graph
adapter exposing Selection, candidates, and per-file hosted durations, and a
Config with that repository's fallback policy. No test code executes here.

Coverage input is JSON (optionally gzip): {"tests": {"test/path.py":
{"files": ["source.py"], "exit": 0}}}. Missing files fail closed. Failed tests
make the observed dynamic cost provisional and disable the stopping claim.
Timings accept run-alone TSV (path, alone_seconds) or JSON (files: {path: s}).
Cuts estimate removal of static edges; they do not prove behavior-preserving
refactors. Local imports remain static edges in Bazel until the adapter's
explicit target boundary changes. Data reads absent from coverage make the
observed dynamic cost a lower bound, not evidence that those reads are unused.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
import csv
import gzip
import json
import math
from pathlib import Path, PurePosixPath
import random
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.bazel.build_files import strongly_connected
from tools.bazel.detangle_repo import import_symbols
from tools.bazel.detangle_inputs import changed_wheels, mapped_changes


@dataclass(frozen=True)
class Config:
    fallback_names: tuple[str, ...] = (
        'MODULE.bazel', 'MODULE.bazel.lock', 'WORKSPACE', 'WORKSPACE.bazel', '.bazelrc', '.bazelversion',
        '.bazelignore', 'BUILD', 'BUILD.bazel', 'uv.lock', 'pyproject.toml', 'pytest.ini', 'setup.cfg',
        'setup.py', 'conftest.py', '.mergify.yml')
    fallback_prefixes: tuple[str, ...] = ('.github/', 'tools/ci/', 'tools/bazel/')
    fallback_suffixes: tuple[str, ...] = ('.bzl',)
    constraint_fragments: tuple[str, ...] = ('requirements', 'constraints')

    fallback_paths: tuple[str, ...] = ()
    map_lock: bool = False

    @classmethod
    def narrowed(cls):
        return cls(
            fallback_names=('MODULE.bazel', 'MODULE.bazel.lock', 'WORKSPACE', 'WORKSPACE.bazel',
                            'BUILD', 'BUILD.bazel', '.bazelrc', '.bazelversion', '.bazelignore'),
            fallback_prefixes=('tools/bazel/',), fallback_suffixes=('.bzl',), constraint_fragments=(),
            fallback_paths=('.github/workflows/gate.yml', 'tools/resolve_tessera_dev_pin.py',
                            *('tools/ci/' + name for name in (
                                'gate_select.py', 'gate_run.py', 'gate_collect.py', 'gate_partition.py',
                                'gate_actions.py', 'gate_paths.py', 'gate_budget.json',
                                'gate_durations.json', 'gate_quarantine.json'))),
            map_lock=True)

    def __post_init__(self):
        for name in ('fallback_names', 'fallback_prefixes', 'fallback_suffixes', 'constraint_fragments', 'fallback_paths'):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def fallback(self, path):
        name = PurePosixPath(path).name
        return (path in self.fallback_paths or name in self.fallback_names or path.startswith(self.fallback_prefixes)
                or path.endswith(self.fallback_suffixes)
                or (any(part in name for part in self.constraint_fragments) and not name.endswith('.py')))


@dataclass
class Selection:
    edges: dict[str, set[str]]
    roots: dict[str, set[str]]
    data: dict[str, set[str]] = field(default_factory=dict)
    extras: dict[str, set[str]] = field(default_factory=dict)
    launches: dict[str, set[str]] = field(default_factory=dict)

    @property
    def tests(self):
        return sorted(self.roots)

    def without(self, cuts):
        removed = defaultdict(set)
        for source, target, kind in cuts:
            removed[source, kind].add(target)
        # Scoring treats adjacency sets as read-only. Reuse untouched sets; a
        # test may hold thousands of data files that an import cut cannot alter.
        edges = {s: targets - removed[s, 'import'] if (s, 'import') in removed else targets
                 for s, targets in self.edges.items()}
        data = {s: targets - removed[s, 'data'] if (s, 'data') in removed else targets
                for s, targets in self.data.items()}
        return Selection(edges, self.roots, data, self.extras, self.launches)

    def combined(self):
        return {s: set(self.edges.get(s, ())) | set(self.launches.get(s, ()))
                for s in self.edges.keys() | self.launches.keys()}

    def masks(self):
        """Propagate test bitsets across the SCC condensation, then add data."""
        edges = self.combined()
        nodes = set(edges) | {t for targets in edges.values() for t in targets}
        nodes |= {n for roots in self.roots.values() for n in roots}
        components = strongly_connected(nodes, edges)
        owner = {n: i for i, c in enumerate(components) for n in c}
        deps = [set() for _ in components]
        indegree = [0] * len(components)
        for s, targets in edges.items():
            for t in targets:
                a, b = owner[s], owner[t]
                if a != b and b not in deps[a]:
                    deps[a].add(b)
                    indegree[b] += 1
        bits = [0] * len(components)
        for i, test in enumerate(self.tests):
            for root in self.roots[test]:
                bits[owner[root]] |= 1 << i
        queue = deque(i for i, degree in enumerate(indegree) if not degree)
        while queue:
            i = queue.popleft()
            for child in deps[i]:
                bits[child] |= bits[i]
                indegree[child] -= 1
                if not indegree[child]:
                    queue.append(child)
        masks = defaultdict(int, {n: bits[owner[n]] for n in nodes})
        for source, paths in self.data.items():
            for path in paths:
                masks[path] |= bits[owner[source]] if source in owner else 0
        for i, test in enumerate(self.tests):
            masks[test] |= 1 << i
            for path in self.extras.get(test, ()):
                masks[path] |= 1 << i
        return dict(masks)

    def selected(self, changed):
        masks = self.masks()
        selected = 0
        for path in changed:
            selected |= masks.get(path, 0)
        return {test for i, test in enumerate(self.tests) if selected & (1 << i)}


def dynamic_masks(executed, population):
    masks = defaultdict(int)
    for i, test in enumerate(sorted(population)):
        for path in set(executed[test]) | {test}:
            masks[path] |= 1 << i
    return dict(masks)


def costs(masks, replay, weights, config):
    tests = sorted(weights)
    values = [weights[t] for t in tests]
    cache = {0: 0.0}
    rows = []
    everything = (1 << len(tests)) - 1
    for merge in replay:
        fallback = sorted(p for p in merge['changed'] if config.fallback(p))
        mask = everything if fallback else 0
        if not fallback:
            for path in mapped_changes(merge, config.map_lock):
                mask |= masks.get(path, 0)
        if mask not in cache:
            cache[mask] = sum(value for i, value in enumerate(values) if mask & (1 << i))
        rows.append({'sha': merge['sha'], 'seconds': cache[mask], 'selected': mask.bit_count(),
                     'fallback': fallback, 'mask': mask})
    n = len(rows)
    if not n:
        raise ValueError('replay must contain at least one merge')
    floor = sum(row['seconds'] for row in rows if row['fallback']) / n
    mean = sum(row['seconds'] for row in rows) / n
    graph = [r['seconds'] for r in rows if not r['fallback']]
    return {'mean_seconds': mean, 'fallback_seconds': floor,
            'graph_only_mean_seconds': sum(graph) / len(graph) if graph else 0,
            'floor_share': floor / mean if mean else 0,
            'fallback_merges': n - len(graph), 'merge_count': n, 'merges': rows}


@dataclass(frozen=True)
class Candidate:
    kind: str
    edges: tuple[tuple[str, str, str], ...]
    effort_lines: int
    symbols: tuple[dict, ...]
    runtime_after: tuple[tuple[str, float], ...] = ()


def projected_savings(selection, replay, weights, candidate, config):
    before = costs(selection.masks(), replay, weights, config)['mean_seconds']
    updated = dict(weights)
    updated.update(candidate.runtime_after)
    after = costs(selection.without(set(candidate.edges)).masks(), replay, updated, config)['mean_seconds']
    return before - after


class Scorer:
    """Reuse the SCC condensation and propagate only a cut's changed merge bits.

    An inter-component cut cannot change SCC membership. Internal cuts first
    check just that component. Only a split rebuilds the condensation. This
    preserves complementary cycle gains without rescanning the entire graph
    for the many edges whose removal changes no selected tests.
    """

    def __init__(self, selection, replay, weights, config, *, _node_seed=None, _fixed=None):
        self.selection, self.replay, self.weights, self.config = selection, replay, weights, config
        self.n = len(replay)
        if not self.n:
            raise ValueError('replay must contain at least one merge')
        if _fixed is None:
            self.changed = defaultdict(int)
            fallbacks = 0
            for i, merge in enumerate(replay):
                if any(config.fallback(p) for p in merge['changed']):
                    fallbacks += 1
                else:
                    for path in mapped_changes(merge, config.map_lock):
                        self.changed[path] |= 1 << i
            self.fallback_cost = fallbacks * sum(weights.values())
            self.extra_bits = {}
            for test in selection.roots:
                bits = self.changed[test]
                for path in selection.extras.get(test, ()):
                    bits |= self.changed[path]
                self.extra_bits[test] = bits
        else:
            # Cuts preserve replay, population, test roots and own inputs.
            self.changed = _fixed.changed
            self.fallback_cost = _fixed.fallback_cost
            self.extra_bits = _fixed.extra_bits
        self.edges = selection.combined()
        nodes = set(self.edges) | {t for ts in self.edges.values() for t in ts}
        nodes |= {p for roots in selection.roots.values() for p in roots}
        self.components = strongly_connected(nodes, self.edges)
        self.owner = {node: i for i, component in enumerate(self.components) for node in component}
        count = len(self.components)
        self.parents = [set() for _ in range(count)]
        self.deps = [set() for _ in range(count)]
        self.origins = defaultdict(set)
        self.seed = [0] * count
        self.node_seed = {}
        for node in nodes:
            i = self.owner[node]
            if _node_seed is None:
                seed = self.changed[node]
                for path in selection.data.get(node, ()):
                    seed |= self.changed[path]
            else:
                seed = _node_seed[node]
            self.node_seed[node] = seed
            self.seed[i] |= seed
            for target in self.edges.get(node, ()):
                j = self.owner[target]
                if i != j:
                    self.deps[i].add(j)
                    self.parents[j].add(i)
                    self.origins[i, j].add((node, target))
        degree = [len(deps) for deps in self.deps]
        self.bits = list(self.seed)
        queue = deque(i for i, value in enumerate(degree) if not value)
        while queue:
            i = queue.popleft()
            for parent in self.parents[i]:
                self.bits[parent] |= self.bits[i]
                degree[parent] -= 1
                if not degree[parent]:
                    queue.append(parent)
        self.test_roots = {test: {self.owner[root] for root in roots} for test, roots in selection.roots.items()}
        self.root_tests = defaultdict(set)
        self.test_bits = {}
        for test, roots in self.test_roots.items():
            bits = self.extra_bits[test]
            for root in roots:
                self.root_tests[root].add(test)
                bits |= self.bits[root]
            self.test_bits[test] = bits
        self.baseline = (self.fallback_cost + sum(
            bits.bit_count() * weights[test] for test, bits in self.test_bits.items())) / self.n

    def cost(self, cuts=frozenset()):
        if not cuts:
            return self.baseline
        removed = {(s, t) for s, t, kind in cuts if kind == 'import'
                   and t in self.selection.edges.get(s, ()) and t not in self.selection.launches.get(s, ())
                   and self.bits[self.owner[t]]}
        data_cuts = defaultdict(set)
        for source, target, kind in cuts:
            if kind == 'data' and source in self.owner and self.changed[target]:
                data_cuts[source].add(target)
        if not removed and not data_cuts:
            return self.baseline
        internal = {self.owner[s] for s, t in removed if self.owner[s] == self.owner[t]}
        for component in internal:
            nodes = self.components[component]
            node_set = set(nodes)
            subgraph = {s: {t for t in self.edges.get(s, ()) if t in node_set and (s, t) not in removed} for s in nodes}
            if len(strongly_connected(nodes, subgraph)) > 1:
                seeds = dict(self.node_seed) if data_cuts else self.node_seed
                for source, removed_data in data_cuts.items():
                    seed = self.changed[source]
                    for path in self.selection.data.get(source, set()) - removed_data:
                        seed |= self.changed[path]
                    seeds[source] = seed
                return Scorer(self.selection.without(cuts), self.replay, self.weights, self.config,
                              _node_seed=seeds, _fixed=self).baseline
        changed_deps, changed_seed = {}, {}
        pairs = {(self.owner[s], self.owner[t]) for s, t in removed if self.owner[s] != self.owner[t]}
        for source, target in pairs:
            if self.origins[source, target] <= removed:
                changed_deps.setdefault(source, set(self.deps[source])).discard(target)
        for source in data_cuts:
            component = self.owner[source]
            if component in changed_seed:
                continue
            bits = 0
            for node in self.components[component]:
                if node not in data_cuts:
                    bits |= self.node_seed[node]
                else:
                    bits |= self.changed[node]
                    for path in self.selection.data.get(node, set()) - data_cuts[node]:
                        bits |= self.changed[path]
            changed_seed[component] = bits
        updates = {}
        pending = deque(changed_deps.keys() | changed_seed.keys())
        queued = set(pending)
        while pending:
            component = pending.popleft()
            queued.remove(component)
            bits = changed_seed.get(component, self.seed[component])
            for child in changed_deps.get(component, self.deps[component]):
                bits |= updates.get(child, self.bits[child])
            if bits == updates.get(component, self.bits[component]):
                continue
            updates[component] = bits
            for parent in self.parents[component]:
                if parent not in queued:
                    pending.append(parent)
                    queued.add(parent)
        affected = set().union(*(self.root_tests[i] for i in updates)) if updates else set()
        saved = 0
        for test in affected:
            bits = self.extra_bits[test]
            for root in self.test_roots[test]:
                bits |= updates.get(root, self.bits[root])
            saved += (self.test_bits[test].bit_count() - bits.bit_count()) * self.weights[test]
        return self.baseline - saved / self.n


def cheaper_tests(cold, warm, *, effort=None, measured=None, estimates=None, default_effort=100):
    """Price a caller's runtime target, preferring measured fixes to estimates.

    Warm timings estimate a possible saving; they do not establish that a cold
    test can achieve it. Effort uses the same estimated-line unit as cuts.
    """
    effort, measured, estimates = effort or {}, measured or {}, estimates or {}
    unknown = (warm.keys() | measured.keys() | estimates.keys() | effort.keys()) - cold.keys()
    if unknown:
        raise ValueError('runtime inputs contain unknown tests: ' + ', '.join(sorted(unknown)))
    result = []
    for test, before in sorted(cold.items()):
        source = measured if test in measured else estimates if test in estimates else warm
        if test not in source:
            continue
        after = float(source[test])
        lines = effort.get(test, default_effort)
        if not math.isfinite(before) or not math.isfinite(after) or before < 0 or not 0 <= after <= before:
            raise ValueError('runtime targets must be finite and between zero and the current runtime: ' + test)
        if not isinstance(lines, int) or lines <= 0:
            raise ValueError('effort must be a positive estimated line count: ' + test)
        if after == before:
            continue
        info = {'source': test, 'target': test, 'symbols': [], 'before_seconds': before,
                'after_seconds': after, 'measurement': 'measured' if test in measured else
                    'estimate' if test in estimates else 'cold-minus-warm estimate'}
        result.append(Candidate('make test cheaper', (), lines, (info,), ((test, after),)))
    return result


def test_costs(static_masks, measured_masks, replay, weights, config):
    static = costs(static_masks, replay, weights, config)
    dynamic = costs(measured_masks, replay, weights, config)
    result = []
    for i, test in enumerate(sorted(weights)):
        a = sum(bool(row['mask'] & (1 << i)) for row in static['merges']) / len(replay)
        b = sum(bool(row['mask'] & (1 << i)) for row in dynamic['merges']) / len(replay)
        result.append({'path': test, 'runtime_seconds': weights[test],
                       'static_selection_frequency': a, 'dynamic_selection_frequency': b,
                       'static_seconds_per_merge': a * weights[test],
                       'dynamic_seconds_per_merge': b * weights[test]})
    return sorted(result, key=lambda row: (-row['static_seconds_per_merge'], row['path']))


def rank(selection, replay, weights, candidates, config, limit=20):
    cuts, result = set(), []
    remaining = list(candidates)
    current_weights = dict(weights)
    edge_effort = {c.edges[0]: c.effort_lines for c in candidates if len(c.edges) == 1}
    current = Scorer(selection, replay, current_weights, config).cost()
    fallback_count = sum(any(config.fallback(p) for p in m['changed']) for m in replay)
    # Rebuild after each pick so runtime changes also update fallback cost.
    for iteration in range(limit):
        scorer = Scorer(selection.without(cuts), replay, current_weights, config)
        best = None
        largest_gain = 0
        for candidate in remaining:
            new = set(candidate.edges) - cuts
            if candidate.runtime_after:
                saved = {t: current_weights[t] - min(current_weights[t], value)
                         for t, value in candidate.runtime_after}
                gain = sum((scorer.test_bits[t].bit_count() / scorer.n +
                            fallback_count / scorer.n)
                           * value for t, value in saved.items())
                after = current - gain
            elif new:
                after = scorer.cost(new)
                gain = current - after
            else:
                continue
            largest_gain = max(largest_gain, gain)
            already = set(candidate.edges) & cuts
            effort = max(1, math.ceil(candidate.effort_lines - sum(
                edge_effort.get(e, candidate.effort_lines / len(candidate.edges)) for e in already)))
            key = (gain / effort, gain, -effort, candidate.edges, candidate.runtime_after)
            if gain > 1e-9 and (best is None or key > best[0]):
                best = key, candidate, after, effort
        if best is None:
            break
        _, candidate, after, effort = best
        if iteration == 0:
            initial_best = largest_gain
        applied = set(candidate.edges) - cuts
        pairs = {(s, t) for s, t, _ in applied}
        symbols = list(candidate.symbols) if candidate.runtime_after else [info for info in candidate.symbols
                   if tuple(info.get('bazel_edge', [info['source'], info['target']])) in pairs]
        result.append({**asdict(candidate), 'edges': [list(e) for e in sorted(applied)],
                       'runtime_after': [list(pair) for pair in candidate.runtime_after],
                       'effort_lines': effort, 'symbols': symbols,
                       'delta_seconds': current - after,
                       'delta_fraction': (current - after) / current if current else 0,
                       'seconds_per_line': (current - after) / effort,
                       'remaining_seconds': after, 'best_initial_delta_seconds': initial_best})
        cuts.update(candidate.edges)
        for test, value in candidate.runtime_after:
            current_weights[test] = min(current_weights[test], value)
        remaining.remove(candidate)
        current = after
    return result


def feedback_cut(nodes, edges, weights):
    """Weighted Eades-Lin-Smyth, adapted from notes/graph/fas.py.

    Try deterministic seeds; restore redundant edges after choosing the best
    order. This is an inclusion-minimal heuristic, not an exact minimum FAS.
    """
    arcs = {(s, t): weights.get((s, t), 1) for s in nodes for t in edges.get(s, ()) if t in nodes and t != s}
    best = None
    for seed in range(8):
        rnd = random.Random(seed)
        out, incoming = defaultdict(dict), defaultdict(dict)
        for (s, t), weight in arcs.items():
            out[s][t] = weight
            incoming[t][s] = weight
        live, left, right = set(nodes), [], []

        def remove(node):
            live.remove(node)
            for target in list(out[node]):
                incoming[target].pop(node, None)
            for source in list(incoming[node]):
                out[source].pop(node, None)

        while live:
            sinks = [n for n in sorted(live) if not out[n]]
            sources = [n for n in sorted(live) if not incoming[n]]
            if sinks:
                for n in sinks:
                    right.append(n)
                    remove(n)
            elif sources:
                for n in sources:
                    left.append(n)
                    remove(n)
            else:
                n = max(sorted(live), key=lambda n: (sum(out[n].values()) - sum(incoming[n].values()), rnd.random()))
                left.append(n)
                remove(n)
        position = {n: i for i, n in enumerate(left + right[::-1])}
        back = {e for e in arcs if position[e[1]] < position[e[0]]}
        key = (sum(arcs[e] for e in back), len(back), sorted(back))
        if best is None or key < best[0]:
            best = key, back
    cut = best[1] if best else set()
    for s, t in sorted(cut, key=lambda e: (-arcs[e], e)):
        # Reachability below restores an edge only when it cannot close a cycle.
        seen, pending = {t}, [t]
        while pending:
            node = pending.pop()
            for child in edges.get(node, ()):
                if child in nodes and (node, child) not in cut and child not in seen:
                    seen.add(child)
                    pending.append(child)
        if s not in seen:
            cut.remove((s, t))
    return cut


def runtime_weights(population, timings, hosted):
    measured = {}
    if timings:
        if Path(timings).suffix == '.tsv':
            with Path(timings).open() as handle:
                measured = {r['path']: float(r['alone_seconds']) for r in csv.DictReader(handle, delimiter='\t')
                            if r.get('alone_seconds')}
        else:
            measured = json.loads(Path(timings).read_text())
            measured = measured.get('files', measured)
    missing = sorted(set(population) - measured.keys() - hosted.keys())
    known = sorted(float(v) for v in hosted.values())
    median = known[len(known) // 2] if known else 1.0
    weights = {p: float(measured.get(p, hosted.get(p, median))) for p in sorted(population)}
    if any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('runtimes must be finite and nonnegative')
    return weights, {'measured': len(set(population) & measured.keys()),
                     'hosted': len(set(population) & (hosted.keys() - measured.keys())),
                     'missing': missing, 'missing_imputed_seconds': median}


def load_dynamic(path, population):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as handle:
        record = json.load(handle)
    rows = record['tests']
    missing = sorted(set(population) - rows.keys())
    if missing:
        raise ValueError('coverage missing tests: ' + ', '.join(missing))
    for test in population:
        if rows[test].get('report_exit', 0) != 0:
            raise ValueError('coverage report failed for ' + test)
    return {p: rows[p]['files'] for p in population}, record


def stopping(static, dynamic, best, gate_seconds, floor_seconds):
    return {'gap_below_ten_percent': static - dynamic < .1 * static,
            'best_below_two_percent_and_gate_near_floor': bool(
                best < .02 * static and gate_seconds is not None and floor_seconds is not None
                and floor_seconds > 0 and gate_seconds <= 1.2 * floor_seconds),
            'gate_evidence_available': gate_seconds is not None and floor_seconds is not None}


def replay_merges(root, revision, count=100):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    rows = []
    for sha in git('rev-list', '--first-parent', '--merges', '-n', str(count), revision).split():
        rows.append({'sha': sha, 'title': git('log', '-1', '--format=%s', sha),
                     'changed': git('diff', '--name-only', '--no-renames', f'{sha}^1', sha).splitlines()})
        if 'uv.lock' in rows[-1]['changed']:
            def lock(ref):
                exists = subprocess.run(['git', 'cat-file', '-e', f'{ref}:uv.lock'], cwd=root,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return git('show', f'{ref}:uv.lock') if exists.returncode == 0 else ''
            rows[-1]['changed_wheels'] = sorted(changed_wheels(lock(f'{sha}^1'), lock(sha)))
    return rows



def main(argv=None):
    from tools.bazel import detangle_repo
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--history-root', type=Path)
    parser.add_argument('--revision', default='HEAD', help='First-parent replay revision')
    parser.add_argument('--source-revision', help='Identity of an exported or overlaid source tree')
    parser.add_argument('--files-from', type=Path)
    parser.add_argument('--population-from', type=Path, help='Fixed test-file population for comparing overlays')
    parser.add_argument('--deps-json', type=Path, help='bazel-diff v42 dependency edges')
    parser.add_argument('--bazel-tests', type=Path, help='JSON mapping test files to non-manual Bazel labels')
    parser.add_argument('--source-labels', type=Path, help='Bazel source-file labels, one per line')
    parser.add_argument('--graph-core', type=Path, help='Shared graph-impact module containing read_deps/read_query')
    parser.add_argument('--query-json', type=Path, help='Shared reader fallback: Bazel streamed_jsonproto')
    parser.add_argument('--config', type=Path, help='JSON overrides for Config fallback fields')
    parser.add_argument('--timings', type=Path)
    parser.add_argument('--warm-timings', type=Path, help='JSON test-to-seconds estimates from warm runs')
    parser.add_argument('--runtime-fixes', type=Path, help='JSON with effort, measured and/or estimates by test')
    parser.add_argument('--fallback-rule', choices=('old', 'narrowed'), default='narrowed')
    parser.add_argument('--dynamic', type=Path, required=True)
    parser.add_argument('--gate-timings', type=Path, help='JSON with fixed_overhead_seconds, gate_seconds, source')
    parser.add_argument('--merges', type=int, default=100)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--cheap-lines', type=int, default=100, help='Per-candidate effort ceiling for residual-cost classification')
    parser.add_argument('--out', type=Path, required=True, help='Output stem, without .json or .md')
    args = parser.parse_args(argv)
    config = Config(**json.loads(args.config.read_text())) if args.config else (
        Config.narrowed() if args.fallback_rule == 'narrowed' else Config())
    history = args.history_root or args.root
    revision = subprocess.check_output(['git', 'rev-parse', args.revision], cwd=history, text=True).strip()
    source_revision = args.source_revision or revision
    files = args.files_from.read_text().splitlines() if args.files_from else None
    if args.deps_json:
        if not args.bazel_tests or not args.source_labels:
            parser.error('--deps-json requires --bazel-tests and --source-labels')
        selection, candidates, hosted, metadata = detangle_repo.adapt_bazel(
            args.root, args.deps_json, json.loads(args.bazel_tests.read_text()),
            args.source_labels.read_text().splitlines(), args.graph_core, args.query_json)
    else:
        selection, candidates, hosted, metadata = detangle_repo.adapt(args.root, files)
    if args.population_from:
        population = set(args.population_from.read_text().splitlines())
        missing = population - selection.roots.keys()
        if missing:
            raise ValueError('population tests absent from graph: ' + ', '.join(sorted(missing)))
        selection = Selection(selection.edges, {t: selection.roots[t] for t in sorted(population)},
                              selection.data, selection.extras, selection.launches)
        metadata['population'] = len(population)
        metadata['fixed_population'] = str(args.population_from)
    replay = replay_merges(history, revision, args.merges)
    weights, weight_sources = runtime_weights(selection.tests, args.timings, hosted)
    executed, coverage = load_dynamic(args.dynamic, selection.tests)
    from tools.bazel.detangle_inputs import add_gate_inputs
    selection, executed, wheel_metadata = add_gate_inputs(
        selection, executed, args.root, files or subprocess.check_output(
            ['git', 'ls-files'], cwd=args.root, text=True).splitlines(),
        (args.root / 'uv.lock').read_text() if (args.root / 'uv.lock').exists() else '')
    metadata['gate_inputs'] = wheel_metadata
    fixes = json.loads(args.runtime_fixes.read_text()) if args.runtime_fixes else {}
    warm = json.loads(args.warm_timings.read_text()) if args.warm_timings else {}
    candidates += cheaper_tests(weights, warm.get('files', warm), **fixes)
    static_masks, measured_masks = selection.masks(), dynamic_masks(executed, selection.tests)
    comparison = {}
    for name, rule in (('old', Config()), ('narrowed', Config.narrowed())):
        a, b = costs(static_masks, replay, weights, rule), costs(measured_masks, replay, weights, rule)
        comparison[name] = {'static_seconds': a['mean_seconds'], 'dynamic_seconds': b['mean_seconds'],
                            'gap_seconds': a['mean_seconds'] - b['mean_seconds'],
                            'floor_seconds': a['fallback_seconds'], 'floor_share': a['floor_share'],
                            'fallback_merges': a['fallback_merges']}
    heavy = test_costs(static_masks, measured_masks, replay, weights, config)
    static = costs(static_masks, replay, weights, config)
    dynamic = costs(dynamic_masks(executed, selection.tests), replay, weights, config)
    print(f'Ranking {len(candidates)} candidates over {len(selection.tests)} tests...', file=sys.stderr, flush=True)
    ranked = rank(selection, replay, weights, candidates, config, args.limit)
    initial_scorer = Scorer(selection, replay, weights, config)
    cycle_cuts = [{**asdict(c), 'delta_seconds': static['mean_seconds'] - initial_scorer.cost(set(c.edges))}
                  for c in candidates if c.kind.startswith('SCC')]
    cheap_edges = {edge for c in candidates if c.effort_lines <= args.cheap_lines for edge in c.edges}
    after_cheap = initial_scorer.cost(cheap_edges)
    timing = json.loads(args.gate_timings.read_text()) if args.gate_timings else {}
    overhead = timing.get('fixed_overhead_seconds')
    longest = sum(max((weights[t] for i, t in enumerate(sorted(weights)) if row['mask'] & (1 << i)), default=0)
                  for row in dynamic['merges']) / len(replay)
    floor = overhead + longest if overhead is not None else None
    initial_best = ranked[0]['best_initial_delta_seconds'] if ranked else 0
    stop = stopping(static['mean_seconds'], dynamic['mean_seconds'], initial_best, timing.get('gate_seconds'), floor)
    failures = sorted(p for p in selection.tests if coverage['tests'][p].get('exit', 0) not in (0, 5))
    recovery = coverage.get('recovery')
    source_revisions = {test: coverage['tests'][test].get('sha', coverage.get('sha', source_revision))
                        for test in selection.tests}
    revision_matches = (coverage.get('sha', source_revision) == source_revision and
                        (not recovery or recovery.get('sha') == source_revision) and
                        all(sha == source_revision for sha in source_revisions.values()))
    coverage_valid = not failures and revision_matches
    estimated_wheels = any('uv.lock' in merge['changed'] for merge in replay)
    if not coverage_valid:
        stop['gap_below_ten_percent'] = False
        stop['best_below_two_percent_and_gate_near_floor'] = False
    stop['coverage_complete_and_passing'] = coverage_valid
    residual = ranked[-1]['remaining_seconds'] if ranked else static['mean_seconds']
    proposed_weights = dict(weights)
    for candidate in ranked:
        proposed_weights.update(candidate['runtime_after'])
    proposed_dynamic = costs(measured_masks, replay, proposed_weights, config)['mean_seconds']
    ranked_edges = {tuple(edge) for candidate in ranked for edge in candidate['edges']}
    graph_after_ranked = initial_scorer.cost(ranked_edges)
    static_masks, measured_masks = selection.masks(), dynamic_masks(executed, selection.tests)
    outside_static = sum((measured_masks.get(p, 0) & ~static_masks.get(p, 0)).bit_count() for p in measured_masks)
    result = {'schema': 'detangle.plan.v1', 'revision': source_revision, 'replay_revision': revision, 'metadata': metadata,
              'config': asdict(config), 'runtime_sources': weight_sources,
              'fallback_comparison': comparison, 'test_costs': heavy,
              'coverage': {'sha': coverage.get('sha'), 'data_reads': coverage.get('data_reads', False),
                           'subprocess_coverage': coverage.get('subprocess_coverage', False),
                           'failed_tests': failures, 'sdk_group_installed': coverage.get('sdk_group_installed', False),
                           'observed_pairs_outside_static': outside_static,
                           'revision_matches': revision_matches, 'recovery': recovery,
                           'source_revisions': source_revisions,
                           'provisional': not coverage_valid or bool(outside_static) or estimated_wheels},
              'static': static, 'dynamic': dynamic, 'gap_seconds': static['mean_seconds'] - dynamic['mean_seconds'],
              'floor_share': static['floor_share'], 'candidates': ranked, 'cycle_cuts': cycle_cuts, 'candidate_count': len(candidates),
              'stopping': stop, 'wall_floor': {'cores': 80, 'fixed_overhead_seconds': overhead,
                  'mean_longest_selected_seconds': longest, 'mean_floor_seconds': floor,
                  'gate_seconds': timing.get('gate_seconds'), 'source': timing.get('source', 'unavailable')},
              'causes': {'essential_seconds': dynamic['mean_seconds'] - dynamic['fallback_seconds'],
                         'floor_seconds': static['fallback_seconds'],
                         'accidental_after_ranked_cuts_seconds': max(0, residual - proposed_dynamic),
                         'ranked_cut_savings_seconds': static['mean_seconds'] - graph_after_ranked,
                         'ranked_runtime_savings_seconds': graph_after_ranked - residual,
                         'ranked_candidate_savings_seconds': static['mean_seconds'] - residual,
                         'cheap_effort_lines_per_candidate': args.cheap_lines,
                         'accidental_after_all_cheap_cuts_seconds': max(0, after_cheap - dynamic['mean_seconds']),
                         'note': 'Accidental cost is an estimate; cuts require semantic review. Data reads are ignored by the floor unless traced.'}}
    # Installed wheels were not retained by the original coverage sweep.
    result['coverage']['dynamic_wheels'] = 'estimated from imports of observed repository source files'
    if outside_static or estimated_wheels:
        result['stopping']['gap_below_ten_percent'] = False
        result['stopping']['best_below_two_percent_and_gate_near_floor'] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    args.out.with_suffix('.md').write_text(markdown(result))
    print(markdown(result))


def markdown(result):
    s, d = result['static'], result['dynamic']
    lines = [f"# Detangle plan at {result['revision'][:12]}", '',
             f"Static {s['mean_seconds']:.2f} s; dynamic {d['mean_seconds']:.2f} s; gap {result['gap_seconds']:.2f} s per merge.",
             f"Fallback floor {s['fallback_seconds']:.2f} s, {100 * result['floor_share']:.1f}% of static; {s['fallback_merges']}/{s['merge_count']} merges.",
             f"Graph-only static {s['graph_only_mean_seconds']:.2f} s; dynamic {d['graph_only_mean_seconds']:.2f} s.",
             f"Coverage provisional: {result['coverage']['provisional']}; failed files: {len(result['coverage']['failed_tests'])}; data reads traced: {result['coverage']['data_reads']}.",
             f"Stopping conditions: {json.dumps(result['stopping'])}", '',
             '| Rank | Cut | Delta seconds | Estimated lines | Symbols and files |',
             '| --- | --- | ---: | ---: | --- |']
    for i, cut in enumerate(result['candidates'], 1):
        symbols = '; '.join(f"{s['source']} → {s['target']}: {', '.join(s.get('symbols', []))}" for s in cut['symbols'])
        lines.append(f"| {i} | {cut['kind']} | {cut['delta_seconds']:.2f} | {cut['effort_lines']} | {symbols.replace('|', '/')} |")
    if 'fallback_comparison' in result:
        lines += ['', 'Fallback comparison, seconds per merge:', '',
                  '| Rule | Static | Dynamic | Gap | Floor | Floor share |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name, row in result['fallback_comparison'].items():
            lines.append(f"| {name} | {row['static_seconds']:.3f} | {row['dynamic_seconds']:.3f} | "
                         f"{row['gap_seconds']:.3f} | {row['floor_seconds']:.3f} | {100*row['floor_share']:.3f}% |")
        lines += ['', 'Test cost per merge:', '',
                  '| Test | Runtime | Static cost | Dynamic cost |', '| --- | ---: | ---: | ---: |']
        for row in result['test_costs']:
            lines.append(f"| {row['path']} | {row['runtime_seconds']:.3f} | "
                         f"{row['static_seconds_per_merge']:.3f} | {row['dynamic_seconds_per_merge']:.3f} |")
    lines += ['', 'Wall floor: ' + json.dumps(result['wall_floor']), '',
              'Remaining causes: ' + json.dumps(result['causes']), '',
              'Coverage source: ' + json.dumps(result['coverage']), '']
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
