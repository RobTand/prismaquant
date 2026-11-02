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
import hashlib
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

from tools.bazel import build_files as build
from tools.ci import graph_impact as core
from tools.ci import import_fanout as legacy_fanout

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path('tests/fixtures/graph_impact_baseline.json')
ALLOWLIST = Path('tests/fixtures/graph_impact_allowlist.json')
SOURCE_ROOTS = ('prismaquant/', 'tools/')
LEGACY_TOLERANCES = {'median_selected_pct': 1.0, 'p90_selected_pct': 1.0,
                     'modules_over_half': 2, 'conftest_source_modules': 0}


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
    live = core.measure(tests, edges, loc)
    live['path_edges'] = [list(edge) for edge in sorted(path_edges)]
    live['contents'] = {test: hashlib.sha256((root / test).read_bytes()).hexdigest() for test in population}
    live['edge_groups'] = {
        'test_to_test_imports': sorted(f'{a} -> {b}' for a, b in edges
                                     if build.is_test_side(a) and build.is_test(b)),
        'conftest_source_imports': sorted(f'{a} -> {b}' for a, b in edges if build.is_conftest(a) and b in sources)}
    if legacy:
        old = legacy_fanout.measure_fanout(root, population=population)
        live.update({key: old[key] for key in LEGACY_TOLERANCES})
        # Keep a single forbidden-edge set, with both the former and current
        # generator evidence. Existing exceptions can only be removed.
        for key in live['edge_groups']:
            live['edge_groups'][key] = sorted(set(live['edge_groups'][key]) | set(old[key]))
    return live


def record(live, previous=None):
    return core.baseline_record(live, previous, tolerances=LEGACY_TOLERANCES)


def migrate(live, old):
    """Preserve every previous numeric minimum and forbidden-edge exception."""
    result = record(live)
    for key in LEGACY_TOLERANCES:
        result['figures'][key] = min(old[key], live[key])
    for key in ('test_to_test_imports', 'conftest_source_imports'):
        additions = set(live['edge_groups'][key]) - set(old[key])
        if additions:
            # Full generator evidence can see pre-existing paths the cheap
            # import parser could not. Do not grandfather those as new imports
            # unless they were already present at the merge base.
            raise ValueError(f'migration found previously unrecorded {key}: {sorted(additions)}')
        result['edge_groups'][key] = sorted(set(live['edge_groups'][key]) & set(old[key]))
    return result


def git(root, *args):
    return subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def base_revision(root, explicit=None):
    try:
        target = explicit or os.environ.get('GATE_BASE') or 'origin/main'
        return git(root, 'merge-base', target, 'HEAD')
    except subprocess.CalledProcessError as error:
        # A no-history executor is supported, but an invalid explicit gate
        # base must not silently downgrade a git checkout's check.
        if explicit or os.environ.get('GATE_BASE'):
            probe = subprocess.run(['git', 'rev-parse', '--git-dir'], cwd=root, capture_output=True)
            if probe.returncode == 0:
                raise ValueError(f'cannot resolve requested base {target}') from error
        return None


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
    base_sha = base_revision(args.root, args.base)
    before = snapshot_at(args.root, base_sha, temp_dir=args.temp_dir) if base_sha else None
    after = snapshot_at(args.root, args.head, temp_dir=args.temp_dir) if args.head else load_snapshot(args.root)
    changed = {test for test, digest in after['contents'].items()
               if before is None or before['contents'].get(test) != digest}
    if args.json:
        args.json.write_text(json.dumps(after, indent=2) + '\n')
    try:
        code = core.check(before, after, args.root / BASELINE, args.root / ALLOWLIST,
                          changed_tests=changed,
                          base_allowlist=before['recorded_allowlist'] if before else None,
                          previous_baseline=before['recorded_baseline'] if before else None,
                          write_baseline=args.write_baseline)
    finally:
        print(f'graph impact {time.perf_counter() - started:.3f}s; '
              f'LOC-weighted U {after["loc_u_pct"]:.1f}%; '
              f'reach median/p90/max {after["reach_median"]}/{after["reach_p90"]}/{after["reach_max"]}')
    return code


if __name__ == '__main__':
    sys.exit(main())
