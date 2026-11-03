"""PrismaQuant's temporary Python-import bridge to the shared graph-impact core.

The preferred source is graph_impact_bazel: bazel-diff deps.json, or the single
streamed query needed when that file omits source edges. This bridge uses the
same Evidence/Graph, roots, own data and launches as detangle_repo.adapt.
Only this adapter knows package/test roots, conftest rules, the gate population
and git. Test IDs are gate file paths; sources are prismaquant/ and tools/.
The pre-existing import-fanout figures remain in this one baseline with their
original evidence, so migrating to the fuller generator cannot erase a gain.
"""
from __future__ import annotations

import argparse
from collections import deque
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prismaquant.digests import file_sha256hex
from tools.bazel import build_files as build
from tools.ci import graph_impact as core
from tools.ci import graph_impact_imports as legacy_fanout

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path('tests/fixtures/graph_impact_baseline.json')
ALLOWLIST = Path('tests/fixtures/graph_impact_allowlist.json')
SOURCE_ROOTS = ('prismaquant/', 'tools/')
LEGACY_TOLERANCES = {'median_selected_pct': 1.0, 'p90_selected_pct': 1.0,
                     'modules_over_half': 2, 'conftest_source_modules': 0}
PR_A_ANALYZERS = ('8e949f05f9cddf82f3353d459f6463920671a00c1dd20dee474350610a79a07f',
                 '9ad443c9fc2cd96d1c35c488c422742d09b746e91745d8011a8c74562e58a3c7')


def files_in(root):
    """Use tracked/unignored files, or the materialized tree without .git."""
    result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=root, capture_output=True, text=True)
    if result.returncode == 0 and Path(result.stdout.strip()).resolve() == Path(root).resolve():
        return build.repository_files(root)
    return sorted(p.relative_to(root).as_posix() for p in Path(root).rglob('*')
                  if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts
                  and not p.relative_to(root).as_posix().startswith(build.SKIPPED)
                  and p.name != 'BUILD.bazel')


def load_snapshot(root, *, files=None, population=None, source_roots=SOURCE_ROOTS, legacy=True):
    root = Path(root)
    files = files_in(root) if files is None else files
    evidence = build.Evidence(root, files)
    graph = build.Graph(evidence)
    if population is None:
        plan, _ = build.hosted_plan(root, files)
        population = [row['path'] for row in plan['selected']]
    sources = {path for path in evidence.modules if path.startswith(source_roots)}
    source_order = sorted(sources)
    bit = {path: 1 << index for index, path in enumerate(source_order)}
    launches = {path: set(evidence.runs(path)) for path in evidence.modules}
    combined = {path: graph.edges[path] | launches[path] for path in evidence.modules}
    # Source-only bitsets preserve data leaves, and avoid materializing every
    # test's whole-repository data set. This is the planner's roots/data/launch
    # interface projected onto source files, with no separate evidence rules.
    masks = {path: bit.get(path, 0) | sum(bit.get(target, 0) for target in graph.data[path])
             for path in evidence.modules}
    reverse = {}
    for source, targets in combined.items():
        for target in targets:
            reverse.setdefault(target, set()).add(source)
    pending, queued = deque(evidence.modules), set(evidence.modules)
    while pending:
        source = pending.popleft()
        queued.remove(source)
        mask = masks[source]
        for target in combined[source]:
            mask |= masks.get(target, bit.get(target, 0))
        if mask != masks[source]:
            masks[source] = mask
            for parent in reverse.get(source, set()) - queued:
                pending.append(parent)
                queued.add(parent)
    tests = {}
    for test in population:
        mask = 0
        for start_node in [test, *graph.conftests_for(test), *evidence.test_modules(test)]:
            mask |= masks[start_node]
        for leaf in evidence.test_data(test, graph.edges[test]):
            mask |= bit.get(leaf, 0)
        reached = set()
        while mask:
            lowest = mask & -mask
            reached.add(source_order[lowest.bit_length() - 1])
            mask ^= lowest
        tests[test] = reached
    edges = {(source, target) for source, targets in graph.edges.items() for target in targets}
    path_edges = {(source, target) for source, targets in combined.items() for target in targets}
    for test in population:
        path_edges.update((test, start_node) for start_node in [*graph.conftests_for(test), *evidence.test_modules(test)])
    loc = {path: len((root / path).read_text(errors='replace').splitlines()) for path in sources}
    live = core.measure(tests, edges, loc, test_packages={test: str(Path(test).parent) for test in tests})
    live['path_edges'] = [list(edge) for edge in sorted(path_edges)]
    live['contents'] = {test: file_sha256hex(root / test) for test in population}
    live['edge_groups'] = {
        'test_to_test_imports': sorted(f'{a} -> {b}' for a, b in edges
                                     if build.is_test_side(a) and build.is_test(b)),
        'conftest_source_imports': sorted(f'{a} -> {b}' for a, b in edges if build.is_conftest(a) and b in sources)}
    if legacy:
        old = legacy_fanout.measure_fanout(root, population=population)
        live.update({key: old[key] for key in LEGACY_TOLERANCES})
        # Both evidence sources describe this revision, never a stored list.
        for key in live['edge_groups']:
            live['edge_groups'][key] = sorted(set(live['edge_groups'][key]) | set(old[key]))
    return live


def numeric_baseline(value):
    return {key: item for key, item in value.items() if key != 'edge_groups'}


def record(live, previous=None):
    previous = numeric_baseline(previous) if previous is not None else None
    if previous is not None and 'test_to_test_count' not in previous:
        previous = {**previous, 'test_to_test_count': len(core.test_to_test_edges(live))}
    return {**numeric_baseline(core.baseline_record(live, previous, tolerances=LEGACY_TOLERANCES)),
            'analyzer_sha256': file_sha256hex(ROOT / 'tools/bazel/build_files.py')}


def baseline_history(before, current):
    """Recognize PR A's one-shot remeasurement, never a new-to-new increase."""
    previous = numeric_baseline(before['recorded_baseline']) if before and before.get('recorded_baseline') else None
    if (previous is not None and previous.get('analyzer_sha256') == PR_A_ANALYZERS[0]
            and before.get('recorded_analyzer_sha256') == PR_A_ANALYZERS[0]
            and current.get('analyzer_sha256') == PR_A_ANALYZERS[1]
            and file_sha256hex(ROOT / 'tools/bazel/build_files.py') == PR_A_ANALYZERS[1]
            and previous.get('test_to_test_count') == 875 and previous['figures']['scc_max'] == 168
            and len(core.test_to_test_edges(before)) == 889 and before['scc_max'] == 174
            and current.get('test_to_test_count') == 889 and current['figures']['scc_max'] == 174):
        # Only these two measurements changed; retain every historical limit.
        return {**previous, 'figures': {**previous['figures'], 'scc_max': 174},
                'test_to_test_count': 889, 'analyzer_sha256': PR_A_ANALYZERS[1]}
    return previous


def check(before, after, baseline_path, allowlist_path, *, changed_tests=(), write_baseline=False):
    baseline = numeric_baseline(json.loads(baseline_path.read_text()))
    allowlist = json.loads(allowlist_path.read_text())
    previous = baseline_history(before, baseline)
    base_allowlist = before.get('recorded_allowlist') if before else None
    mode = core.policy_mode(baseline)
    print(core.summary(before, after, baseline['reach_budget'], test_to_test=mode), flush=True)
    for reminder in core.policy_reminders(after, baseline):
        print(reminder, flush=True)
    errors = core.baseline_errors(baseline, previous)
    errors += core.allowlist_errors(allowlist, base_allowlist if base_allowlist is not None else allowlist)
    # Frozen core 75aff74a accepts edge groups in its policy input. Supply only
    # live history, and keep its stored-gain check restricted to numeric limits.
    policy = {**baseline, 'edge_groups': (before if before is not None else after)['edge_groups']}
    errors += core.compare(before, after, policy, changed_tests=changed_tests, allowlist=allowlist,
                           record_gains=True, test_packages=after['test_packages'],
                           base_package_max_reach=core.package_max_reach(before) if before else {},
                           edge_diff=core.diff_edges(before, after) if before else None)
    if not write_baseline:
        errors += core.stale_entries(after, baseline)
    if errors:
        if write_baseline:
            print('refusing to record a loss:', file=sys.stderr)
        print('\n'.join(errors), file=sys.stderr)
        return 1
    if write_baseline:
        baseline_path.write_text(json.dumps(record(after, baseline), indent=2) + '\n')
    return 0


def git(root, *args):
    return subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def base_revision(root, explicit=None, *, head='HEAD'):
    try:
        target = explicit or os.environ.get('GATE_BASE') or 'origin/main'
        if os.environ.get('GATE_GRAPH_BASE_MODE', 'pr') == 'batch':
            return git(root, 'rev-parse', target + '^{commit}')
        return git(root, 'merge-base', target, head)
    except subprocess.CalledProcessError as error:
        if explicit or os.environ.get('GATE_BASE'):
            probe = subprocess.run(['git', 'rev-parse', '--git-dir'], cwd=root, capture_output=True)
            if probe.returncode == 0:
                raise ValueError(f'cannot resolve requested base {target}') from error
        return None


def rename_paths(root, base, head='HEAD'):
    fields = subprocess.check_output(['git', 'diff', '-M', '--name-status', '-z', base, head], cwd=root).split(b'\0')
    pairs = []
    index = 0
    while index < len(fields) - 1:
        status, source = fields[index:index + 2]
        index += 2
        if status.startswith(b'R'):
            destination = fields[index]
            index += 1
            source, destination = os.fsdecode(source), os.fsdecode(destination)
            if Path(source).name == Path(destination).name:
                pairs.append((source, destination))
    sources, destinations = {}, {}
    for source, destination in pairs:
        sources[source] = sources.get(source, 0) + 1
        destinations[destination] = destinations.get(destination, 0) + 1
    return {source: destination for source, destination in pairs
            if sources[source] == destinations[destination] == 1}


def align_base(before, renames):
    if not renames:
        return before
    nodes = set(before['loc']) | set(before['reach']) | {node for edge in before['edges'] for node in edge}
    renames = {source: target for source, target in renames.items()
               if target not in nodes or target in renames}
    def path(value):
        return renames.get(value, value)
    result = core.measure({path(test): {path(source) for source in sources}
                           for test, sources in before['reach'].items()},
                          {(path(source), path(target)) for source, target in before['edges']},
                          {path(source): lines for source, lines in before['loc'].items()},
                          test_packages={path(test): str(Path(path(test)).parent) for test in before['reach']})
    result.update({key: before[key] for key in LEGACY_TOLERANCES if key in before})
    result['contents'] = {path(test): digest for test, digest in before['contents'].items()}
    result['path_edges'] = [[path(a), path(b)] for a, b in before.get('path_edges', before['edges'])]
    result['edge_groups'] = {group: sorted(f'{path(a)} -> {path(b)}' for a, b in
                                         (edge.split(' -> ', 1) for edge in edges))
                             for group, edges in before['edge_groups'].items()}
    for key in ('recorded_baseline', 'recorded_allowlist', 'recorded_analyzer_sha256'):
        result[key] = before.get(key)
    return result


def snapshot_at(root, revision, *, temp_dir=None, legacy=True):
    """Materialize tracked files via git archive; never run base revision code."""
    archive = subprocess.check_output(['git', 'archive', '--format=tar', revision], cwd=root)
    with tempfile.TemporaryDirectory(prefix='graph-impact-', dir=temp_dir) as directory:
        tree = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            # Git paths are repository-relative. Refuse links and traversals;
            # metadata files suffice for the same Evidence builder in-process.
            files = []
            for member in tar:
                path = Path(member.name)
                if not member.isfile() or path.is_absolute() or '..' in path.parts:
                    continue
                target = tree / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
                if target.name != 'BUILD.bazel' and not member.name.startswith(build.SKIPPED):
                    files.append(member.name)
        live = load_snapshot(tree, files=sorted(files), legacy=legacy)
        live['recorded_baseline'] = json.loads((tree / BASELINE).read_text()) if (tree / BASELINE).exists() else None
        analyzer = tree / 'tools/bazel/build_files.py'
        if analyzer.is_file():
            live['recorded_analyzer_sha256'] = file_sha256hex(analyzer)
            if live['recorded_baseline'] is not None:
                live['recorded_baseline'].setdefault('analyzer_sha256', live['recorded_analyzer_sha256'])
        live['recorded_allowlist'] = json.loads((tree / ALLOWLIST).read_text()) if (tree / ALLOWLIST).exists() else None
        return live


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--base')
    parser.add_argument('--head', help='an optional historical head revision, otherwise the working tree')
    parser.add_argument('--temp-dir', type=Path)
    parser.add_argument('--write-baseline', action='store_true')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    head_sha = args.head or 'HEAD'
    base_sha = base_revision(args.root, args.base, head=head_sha)
    before = snapshot_at(args.root, base_sha, temp_dir=args.temp_dir) if base_sha else None
    after = snapshot_at(args.root, args.head, temp_dir=args.temp_dir) if args.head else load_snapshot(args.root)
    if before is not None:
        before = align_base(before, rename_paths(args.root, base_sha, head_sha))
    changed = {test for test, digest in after['contents'].items()
               if before is None or before['contents'].get(test) != digest}
    if args.json:
        args.json.write_text(json.dumps(after, indent=2) + '\n')
    try:
        code = check(before, after, args.root / BASELINE, args.root / ALLOWLIST,
                     changed_tests=changed, write_baseline=args.write_baseline)
    finally:
        print(f'graph impact {time.perf_counter() - started:.3f}s; '
              f'LOC-weighted U {after["loc_u_pct"]:.1f}%; '
              f'reach median/p90/max {after["reach_median"]}/{after["reach_p90"]}/{after["reach_max"]}')
    return code


if __name__ == '__main__':
    sys.exit(main())
