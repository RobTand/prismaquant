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


@dataclass(frozen=True)
class Config:
    fallback_names: tuple[str, ...] = (
        'MODULE.bazel', 'MODULE.bazel.lock', 'WORKSPACE', 'WORKSPACE.bazel', '.bazelrc', '.bazelversion',
        '.bazelignore', 'BUILD', 'BUILD.bazel', 'uv.lock', 'pyproject.toml', 'pytest.ini', 'setup.cfg',
        'setup.py', 'conftest.py', '.mergify.yml')
    fallback_prefixes: tuple[str, ...] = ('.github/', 'tools/ci/', 'tools/bazel/')
    fallback_suffixes: tuple[str, ...] = ('.bzl',)
    constraint_fragments: tuple[str, ...] = ('requirements', 'constraints')

    def fallback(self, path):
        name = PurePosixPath(path).name
        return (name in self.fallback_names or path.startswith(self.fallback_prefixes)
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
        edges = {s: targets - removed[s, 'import'] for s, targets in self.edges.items()}
        data = {s: targets - removed[s, 'data'] for s, targets in self.data.items()}
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
            for path in merge['changed']:
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


def delta(selection, replay, weights, candidate, config):
    before = costs(selection.masks(), replay, weights, config)['mean_seconds']
    after = costs(selection.without(set(candidate.edges)).masks(), replay, weights, config)['mean_seconds']
    return before - after


class Scorer:
    """Propagate 100 merge bits instead of materializing every test's file set."""

    def __init__(self, selection, replay, weights, config):
        self.weights, self.selection = weights, selection
        self.n = len(replay)
        self.changed = defaultdict(int)
        self.fallbacks = 0
        for i, merge in enumerate(replay):
            if any(config.fallback(p) for p in merge['changed']):
                self.fallbacks += 1
            else:
                for path in merge['changed']:
                    self.changed[path] |= 1 << i

    def cost(self, cuts=frozenset()):
        selection = self.selection.without(cuts) if cuts else self.selection
        edges = selection.combined()
        nodes = set(edges) | {t for ts in edges.values() for t in ts}
        nodes |= {p for roots in selection.roots.values() for p in roots}
        components = strongly_connected(nodes, edges)
        owner = {node: i for i, component in enumerate(components) for node in component}
        parents = [set() for _ in components]
        degree = [0] * len(components)
        bits = [0] * len(components)
        for node in nodes:
            i = owner[node]
            bits[i] |= self.changed[node]
            for path in selection.data.get(node, ()):
                bits[i] |= self.changed[path]
            for target in edges.get(node, ()):
                j = owner[target]
                if i != j and i not in parents[j]:
                    parents[j].add(i)
                    degree[i] += 1
        queue = deque(i for i, value in enumerate(degree) if not value)
        while queue:
            i = queue.popleft()
            for parent in parents[i]:
                bits[parent] |= bits[i]
                degree[parent] -= 1
                if not degree[parent]:
                    queue.append(parent)
        total = self.fallbacks * sum(self.weights.values())
        for test, roots in selection.roots.items():
            selected = self.changed[test]
            for root in roots:
                selected |= bits[owner[root]]
            for path in selection.extras.get(test, ()):
                selected |= self.changed[path]
            total += selected.bit_count() * self.weights[test]
        return total / self.n


def rank(selection, replay, weights, candidates, config, limit=20):
    scorer = Scorer(selection, replay, weights, config)
    cuts, result = set(), []
    remaining = list(candidates)
    current = scorer.cost()
    # An upper bound on later marginal gains is NOT safe across SCC cuts.
    # Every candidate is therefore rescored after every pick.
    for _ in range(limit):
        best = None
        for candidate in remaining:
            new = set(candidate.edges) - cuts
            if not new:
                continue
            after = scorer.cost(cuts | new)
            gain = current - after
            effort = max(1, candidate.effort_lines)
            key = (gain / effort, gain, -effort, candidate.edges)
            if gain > 1e-9 and (best is None or key > best[0]):
                best = key, candidate, after
        if best is None:
            break
        _, candidate, after = best
        result.append({**asdict(candidate), 'edges': [list(e) for e in candidate.edges],
                       'delta_seconds': current - after,
                       'delta_fraction': (current - after) / current if current else 0,
                       'seconds_per_line': (current - after) / max(1, candidate.effort_lines),
                       'remaining_seconds': after})
        cuts.update(candidate.edges)
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
    return rows


# Re-export the symbol reader for callers using the original single-tool API.
def import_symbols(source, source_path, target_path):
    from tools.bazel.detangle_repo import import_symbols as read
    return read(source, source_path, target_path)


def main(argv=None):
    from tools.bazel import detangle_repo
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--history-root', type=Path)
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--files-from', type=Path)
    parser.add_argument('--config', type=Path, help='JSON overrides for Config fallback fields')
    parser.add_argument('--timings', type=Path)
    parser.add_argument('--dynamic', type=Path, required=True)
    parser.add_argument('--gate-timings', type=Path, help='JSON with fixed_overhead_seconds, gate_seconds, source')
    parser.add_argument('--merges', type=int, default=100)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--out', type=Path, required=True, help='Output stem, without .json or .md')
    args = parser.parse_args(argv)
    config = Config(**json.loads(args.config.read_text())) if args.config else Config()
    history = args.history_root or args.root
    revision = subprocess.check_output(['git', 'rev-parse', args.revision], cwd=history, text=True).strip()
    files = args.files_from.read_text().splitlines() if args.files_from else None
    selection, candidates, hosted, metadata = detangle_repo.adapt(args.root, files)
    replay = replay_merges(history, revision, args.merges)
    weights, weight_sources = runtime_weights(selection.tests, args.timings, hosted)
    executed, coverage = load_dynamic(args.dynamic, selection.tests)
    static = costs(selection.masks(), replay, weights, config)
    dynamic = costs(dynamic_masks(executed, selection.tests), replay, weights, config)
    print(f'Ranking {len(candidates)} candidates over {len(selection.tests)} tests...', file=sys.stderr, flush=True)
    ranked = rank(selection, replay, weights, candidates, config, args.limit)
    timing = json.loads(args.gate_timings.read_text()) if args.gate_timings else {}
    overhead = timing.get('fixed_overhead_seconds')
    longest = sum(max((weights[t] for i, t in enumerate(sorted(weights)) if row['mask'] & (1 << i)), default=0)
                  for row in dynamic['merges']) / len(replay)
    floor = overhead + longest if overhead is not None else None
    initial_best = max((delta(selection, replay, weights, c, config) for c in candidates), default=0)
    stop = stopping(static['mean_seconds'], dynamic['mean_seconds'], initial_best, timing.get('gate_seconds'), floor)
    failures = sorted(p for p in selection.tests if coverage['tests'][p].get('exit', 0) not in (0, 5))
    coverage_valid = not failures
    if not coverage_valid:
        stop['gap_below_ten_percent'] = False
        stop['best_below_two_percent_and_gate_near_floor'] = False
    stop['coverage_complete_and_passing'] = coverage_valid
    residual = ranked[-1]['remaining_seconds'] if ranked else static['mean_seconds']
    static_masks, measured_masks = selection.masks(), dynamic_masks(executed, selection.tests)
    outside_static = sum((measured_masks.get(p, 0) & ~static_masks.get(p, 0)).bit_count() for p in measured_masks)
    result = {'schema': 'detangle.plan.v1', 'revision': revision, 'metadata': metadata,
              'config': asdict(config), 'runtime_sources': weight_sources,
              'coverage': {'sha': coverage.get('sha'), 'data_reads': coverage.get('data_reads', False),
                           'subprocess_coverage': coverage.get('subprocess_coverage', False),
                           'failed_tests': failures, 'sdk_group_installed': coverage.get('sdk_group_installed', False),
                           'observed_pairs_outside_static': outside_static,
                           'provisional': not coverage_valid or bool(outside_static)},
              'static': static, 'dynamic': dynamic, 'gap_seconds': static['mean_seconds'] - dynamic['mean_seconds'],
              'floor_share': static['floor_share'], 'candidates': ranked, 'candidate_count': len(candidates),
              'stopping': stop, 'wall_floor': {'cores': 80, 'fixed_overhead_seconds': overhead,
                  'mean_longest_selected_seconds': longest, 'mean_floor_seconds': floor,
                  'gate_seconds': timing.get('gate_seconds'), 'source': timing.get('source', 'unavailable')},
              'causes': {'essential_seconds': dynamic['mean_seconds'] - dynamic['fallback_seconds'],
                         'floor_seconds': static['fallback_seconds'],
                         'accidental_after_ranked_cuts_seconds': max(0, residual - dynamic['mean_seconds']),
                         'ranked_cut_savings_seconds': static['mean_seconds'] - residual,
                         'note': 'Accidental cost is an estimate; cuts require semantic review. Data reads are ignored by the floor unless traced.'}}
    if outside_static:
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
             f"Stopping conditions: {json.dumps(result['stopping'], sort_keys=True)}", '',
             '| Rank | Cut | Delta seconds | Estimated lines | Symbols and files |',
             '| --- | --- | ---: | ---: | --- |']
    for i, cut in enumerate(result['candidates'], 1):
        symbols = '; '.join(f"{s['source']} → {s['target']}: {', '.join(s.get('symbols', []))}" for s in cut['symbols'])
        lines.append(f"| {i} | {cut['kind']} | {cut['delta_seconds']:.2f} | {cut['effort_lines']} | {symbols.replace('|', '/')} |")
    lines += ['', 'Wall floor: ' + json.dumps(result['wall_floor'], sort_keys=True), '',
              'Remaining causes: ' + json.dumps(result['causes'], sort_keys=True), '',
              'Coverage source: ' + json.dumps(result['coverage'], sort_keys=True), '']
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
