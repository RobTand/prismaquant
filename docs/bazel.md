# Dependency graph impact

`tests/test_graph_impact.py` runs on every hosted PR gate. It checks the graph
at the merge base against the PR head, without using run history. `GATE_BASE`
selects the gate base; local runs use `git merge-base origin/main HEAD`.

The check reports the expected fraction of tests selected by a uniformly
chosen source-file change, U, and a LOC-weighted U. It also reports source
fanout, the largest SCC and its members, and median/p90/maximum test reach.
A changed or new test must fit the recorded reach budget. New test-to-test
edges, new source imports in conftests, a material rise in U, and any growing
or merging SCC fail. The diagnostic attributes added edges and fanout growth.

The initial budget is the ceiling of the current p90 source reach. Existing
tests above that budget are listed with reasons in
`tests/fixtures/graph_impact_allowlist.json`. Those exceptions and their
reasons can only be removed. The graph check itself has a reasoned exception
because it inventories the whole graph. New tests do not inherit exceptions.

## Shared core

`tools/ci/graph_impact.py` is the stdlib-only core shared byte-identically by
PrismaQuant, PrismaBuild, Tessera and TrustLadder. Its inputs are opaque test
IDs mapped to sets of source files, a set of directed edges, and optional
source line counts. No Python naming convention or git operation lives in
this file. `measure` produces JSON-safe metrics; `compare` applies policy;
`check` reads caller-supplied baseline and allowlist paths and prints the
summary on both pass and failure. Adapters supply changed test IDs when a
body changes without changing reach. LOC keys declare the complete source
universe, including unreached files. Without LOC, non-test edge nodes and
reached files form the source universe, with unit weights.

The vendoring test checks `tools/ci/graph_impact.sha256`. Update the recorded
hash only when updating the shared core in all four repositories.

## Bazel source

The preferred entry point is `python -m tools.ci.graph_impact_bazel`. Reuse
base and head `bazel-diff generate-hashes --includeTargetType --depEdgesFile`
outputs. Pass test labels from
`tests(//...) except attr(tags, "manual", tests(//...))`.

Released bazel-diff v42 omits source-file edges from `deps.json`, even though
its hashes include `SourceFile` labels. This was verified on a real Bazel
workspace in BuildBuddy, rather than inferred from synthetic JSON. Supply
one fallback query per revision:

```sh
bazel query 'deps(tests(//...))' --output=streamed_jsonproto > query.json
```

`read_query` keeps `ruleInput` edges and source-file records and removes
manual-tagged tests from the population. Querying only source-file records
would discard the edges needed to associate sources with individual tests.
Use `--query` and `--base-query` alongside the corresponding deps/hash/test
manifests. Provide `--loc` and `--base-loc` JSON maps for measured LOC weights;
otherwise the Bazel source reports unit-weighted U.

On a failure, the adapter invokes
`bazel-diff explain -sh base.json -fh head.json -d deps.json --target LABEL --format mermaid`.
A supporting implementation prints its Mermaid output in the diagnostic.
The released v42 CLI does not implement `explain`; the adapter reports that
error explicitly. Native Mermaid blame requires a version that supplies it.

## Python bridge and baseline maintenance

Until generated BUILD files are available, run:

```sh
python -m tools.ci.graph_impact_pq
python -m tools.ci.graph_impact_pq --write-baseline
```

`graph_impact_pq` uses the generator's `Evidence` and `Graph` APIs, with the
same roots, data leaves and launched programs as the detangle planner's
`detangle_repo.adapt`. It keeps package roots, conftest rules, test population
and git materialization outside the shared core. Its bitsets project that
existing graph onto source files; they do not invent narrower import rules.
The bridge prints the shortest path through conftests to the farthest hub.

The old `import_fanout` CLI forwards to this check. Its original evidence
parser supplies the migrated figures in the same baseline, preserving each
recorded minimum. There is one baseline and one test file. Writing the
baseline refuses losses and keeps the minimum of every numeric figure.
Material gains must be recorded in the same PR. A committed baseline holds
only limits, the reach budget, and existing forbidden-edge exceptions, so
ordinary concurrent PRs do not update a full edge snapshot.

The full check belongs in Bazel's runner-VM lane because attribution requires
a git checkout and history. Without git, it reports `no base available` and
still checks absolute figures, forbidden edges and every test's reach budget.
The shrink-only exceptions preserve existing broad tests in that lane.
