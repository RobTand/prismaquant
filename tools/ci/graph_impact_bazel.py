"""Preferred Bazel source for graph_impact, independent of repository/language.

Reuse bazel-diff v42 base/head hashes and --depEdgesFile outputs. Pass the
population from ``tests(//...) except attr(tags, "manual", tests(//...))``.
Source-file labels come from --includeTargetType hashes, or --sources. Real
v42 deps files omit source edges, so supply one fallback query per revision:
``bazel query 'deps(tests(//...))' --output=streamed_jsonproto``. It includes
source-file identities AND ruleInput edges; filtering query output down to
source files alone loses the mapping from each test to its source files.

Example: python -m tools.ci.graph_impact_bazel --deps head-deps.json
--hashes head.json --tests head-tests.txt --query head-query.json
--base-deps base-deps.json --base-hashes base.json --base-tests base-tests.txt
--base-query base-query.json --baseline baseline.json --allowlist exceptions.json
--bazel-diff 'java -jar /path/bazel-diff.jar'

Mermaid blame calls bazel-diff explain. Released v70.0.2 supports it; v42
reports an unavailable-command diagnostic. Query edges supplement the raw
export for blame without modifying the gate's selection graph.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.ci import graph_impact as core


def labels(path):
    text = Path(path).read_text()
    return set(json.loads(text) if text.lstrip().startswith('[') else text.splitlines())


def hash_values(path):
    value = json.loads(Path(path).read_text())
    return value.get('hashes', value)


def load_bazel_graph(deps, tests, *, hashes=None, sources=None, query=None, loc=None):
    population = labels(tests)
    source_files = labels(sources) if sources else {
        label for label, value in hash_values(hashes).items() if value.startswith('SourceFile#')}
    try:
        inputs = core.read_deps(deps, population, source_files, loc=loc)
        source = 'bazel-diff deps.json'
    except ValueError as error:
        if query is None or 'no source-file labels' not in str(error):
            raise
        inputs = core.read_query(query, test_targets=population, loc=loc)
        source = 'Bazel streamed_jsonproto query (deps.json omits source-file labels)'
    result = core.measure(*inputs, test_packages={test: test.rsplit(':', 1)[0] for test in population})
    result['graph_source'] = source
    return result


def explain(command, base_hashes, head_hashes, deps, target, *, edges=None):
    try:
        with tempfile.TemporaryDirectory(prefix='graph-impact-blame-') as directory:
            if edges is not None:
                dependencies = {label: set(targets) for label, targets in json.loads(Path(deps).read_text()).items()}
                for source, destination in edges:
                    dependencies.setdefault(source, set()).add(destination)
                deps = Path(directory) / 'deps.json'
                deps.write_text(json.dumps({label: sorted(targets) for label, targets in dependencies.items()}))
            result = subprocess.run([*shlex.split(command), 'explain', '-sh', str(base_hashes),
                                     '-fh', str(head_hashes), '-d', str(deps), '--target', target,
                                     '--format', 'mermaid'], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f'bazel-diff explain unavailable for {target}: {error}', file=sys.stderr)
        return ''
    if result.returncode:
        print(f'bazel-diff explain unavailable for {target} (exit {result.returncode}): '
              + (result.stderr or result.stdout).strip(), file=sys.stderr)
        return ''
    return result.stdout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    for name in ('deps', 'hashes', 'tests', 'sources', 'query'):
        parser.add_argument('--' + name, type=Path, required=name in {'deps', 'tests'})
        parser.add_argument('--base-' + name, type=Path)
    parser.add_argument('--loc', type=Path)
    parser.add_argument('--base-loc', type=Path)
    parser.add_argument('--changed-tests', type=Path)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--allowlist', type=Path)
    parser.add_argument('--base-allowlist', type=Path)
    parser.add_argument('--base-baseline', type=Path)
    parser.add_argument('--bazel-diff', default='bazel-diff')
    parser.add_argument('--write-baseline', action='store_true')
    args = parser.parse_args(argv)
    if not args.hashes and not args.sources:
        parser.error('provide --hashes with --includeTargetType, or --sources')
    def read(path):
        return json.loads(path.read_text()) if path else None
    after = load_bazel_graph(args.deps, args.tests, hashes=args.hashes, sources=args.sources,
                 query=args.query, loc=read(args.loc))
    before = None
    if args.base_deps:
        if not args.base_tests or not (args.base_hashes or args.base_sources):
            parser.error('base graph requires --base-tests and --base-hashes or --base-sources')
        before = load_bazel_graph(args.base_deps, args.base_tests, hashes=args.base_hashes, sources=args.base_sources,
                      query=args.base_query, loc=read(args.base_loc))
    changed = labels(args.changed_tests) if args.changed_tests else set()
    blame = (lambda target: explain(args.bazel_diff, args.base_hashes, args.hashes, args.deps, target,
                                    edges=after['edges'])) if (
        args.base_hashes and args.hashes) else None
    print('graph source: ' + after['graph_source'])
    return core.check(before, after, args.baseline, args.allowlist, changed_tests=changed,
                      base_allowlist=read(args.base_allowlist), previous_baseline=read(args.base_baseline),
                      write_baseline=args.write_baseline, explain=blame, test_packages=after['test_packages'],
                      base_package_max_reach=core.package_max_reach(before) if before else {},
                      edge_diff=core.diff_edges(before, after) if before else None)


if __name__ == '__main__':
    sys.exit(main())
