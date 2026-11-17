# Dependency graph impact

`tools/ci/graph_impact_pq.py` runs in its own required
hosted job, in parallel with the gate shards. It checks the graph
at the merge base against the PR head, without using run history. `GATE_BASE`
is the event's base commit. Ordinary PRs compare from its merge base with the
head; authenticated Mergify batch drafts compare from that actual base commit.
`GATE_GRAPH_BASE_MODE` distinguishes those cases; a full canary is still a PR.
Local runs use `git merge-base origin/main HEAD`. Before diffing, Git's `-M`
rename records align base paths for unambiguous same-basename moves, including
moves with content edits. A basename change remains a deletion and addition.

The check reports the expected fraction of tests selected by a uniformly
chosen source-file change, U, and a LOC-weighted U. It also reports source
fanout, the largest SCC and its members, and median/p90/maximum test reach.
A new or changed test uses the maximum base reach in its package. An edited
test can retain its own base reach; a new package uses the recorded global
budget. New edges that expand existing-source fanout among existing tests,
or deepen existing-test reach, fail. Conftest source imports remain forbidden.
The shared core supports `test_to_test: strict|reach`. Strict mode refuses every
new test-to-test edge; PQ uses it since November 16, 2026 ([the CI gate guide](ci_gate.md)).
Every SCC is checked for growth or mergers, even when an unrelated larger cycle
hides the size change. The SCC maximum can only shrink.
U and LOC-weighted U are reported trends and never fail a PR. Baseline writes
keep their minimum. The diagnostic attributes structural growth to added edges.

The global fallback budget is the ceiling of the initial p90 source reach. Existing
tests above that budget are listed with reasons in
`tests/fixtures/graph_impact_allowlist.json`. Those exceptions and their
reasons can only be removed. The graph check itself has a reasoned exception
because it inventories the whole graph. New tests do not inherit exceptions.

## Shared core

`tools/ci/graph_impact.py` is the stdlib-only core shared byte-identically by
PrismaQuant, Tessera and TrustLadder. Its inputs are opaque test
IDs mapped to sets of source files, directed edges, optional source line
counts, and the exact adapter metadata `test_packages`,
`base_package_max_reach`, and `edge_diff`. The typed diff contains `added` and
`removed`, each split into `rule_to_rule` and `rule_to_source` pairs. No Python naming convention or git operation lives in
this file. `measure` produces JSON-safe metrics; `compare` applies policy;
`check` reads caller-supplied baseline and allowlist paths and prints the
summary on both pass and failure. Adapters supply changed test IDs when a
body changes without changing reach. LOC keys declare the complete source
universe, including unreached files. Without LOC, non-test edge nodes and
reached files form the source universe, with unit weights.

The vendoring test checks `tools/ci/graph_impact.sha256`. Update the recorded
hash only when updating the shared core in all four repositories.
The duplication inventory permits this exact file's independent helper names;
its structural and primitive-digest checks still apply. Adjacent adapters use
PQ owners and remain subject to all duplication checks.

## Bazel source

The preferred entry point is `python -m tools.ci.graph_impact_bazel`. Reuse
base and head `bazel-diff generate-hashes --includeTargetType --depEdgesFile`
outputs. Pass test labels from
`tests(//...) except attr(tags, "manual", tests(//...))`.

Released bazel-diff v42 and v70.0.2 omit source-file edges from `deps.json`, even though
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
Version v70.0.2 prints its Mermaid output in the diagnostic, verified on a
real source change in a tiny BuildBuddy workspace. The adapter supplements a
temporary copy of the raw dependency export with query edges for blame,
preserving the gate's selection graph. The released v42 CLI does not
implement `explain`; the adapter reports that error explicitly.

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

The head's `scc_max` must not exceed the live merge base or batch base value, or the
absolute baseline ceiling. The baseline's `analyzer_sha256` must match `build_files.py`.
A PR that changes the baseline, `build_files.py` or a graph_impact module also records
`scc_max` and `test_to_test_count` no higher than its base, measured by its analyzer.
Explicit baseline writes record gains and refuse structural losses. New-edge checks compare the aligned
base graph with the head graph, without stored edge lists. New source files
inherit a package when their existing-test consumers fit a peer's base consumers.
The separate allowlist holds reasoned exceptions.

Duplication gains need no fixture edit. The fixture is the absolute ceiling.
The adapter also compares live base and head inventories with the current scanner.
New pairs, same-name groups or members, and primitive digest sites fail.
`tools/duplication_inventory.py --write-baseline` can record gains but is optional.

The full check belongs in Bazel's runner-VM lane because attribution requires
a git checkout and history. Without a base, or without the baseline at the same
path in the base, it checks the absolute limits and reach budgets, and fails.
The shrink-only exceptions preserve existing broad tests in that lane.

## Remote gate execution

The cutover sends amd64 Bazel actions to `selfhosted-amd64` and arm64 actions
to BuildBuddy's hosted arm64 executors, with no fallback. Remote Bazel runners
keep their existing four-CPU cloud placement. Hosted pytest exception lanes stay
on GitHub-hosted runners.

Selected gates and post-merge records submit one Remote Bazel shard at a
time. The gate's fail-closed `//...` PR fallback uses six shards and may submit six
concurrent Remote Bazel runs per controller on the self-hosted amd64 pool, each with
`--jobs=16`. The controller rejects concurrency above six, and rejects
concurrency above one outside PR fallback. Post-merge and development runs
remain serial. The shard request cap remains 55 minutes and the controller's
per-shard await deadline remains 58 minutes. Mergify's `checks_timeout` and merge
conditions stay unchanged.

Fallback shards balance the measured file times in `tools/ci/gate_durations.json`,
mapped to Bazel test labels. These are hosted Python 3.12 JUnit totals, not remote
scheduling measurements. Longest tests are placed first on the least-loaded
shard, with the measured median used for untimed tests. Build-only targets are
distributed by count. Without matching timings, all targets are balanced by count.

Mergify's `max_parallel_checks: 1` and per-PR workflow concurrency
bound gate overlap. There is no shared GitHub job concurrency group, because
GitHub cancels an older pending job when a newer one queues, causing Mergify
to see a failed check and dequeue it. For Bazel, lock or workflow changes, one
batch-draft fallback controller with six shards can submit six Remote
Bazel runs per repo. Extra runs queue on the self-hosted executors
rather than fail; this appears as longer shards, not errors. This exception does
not permit concurrent agent or development runs.

Both modes use deterministic target-pattern partitions, verify their union
against the queried population exactly once, and fail closed on any missing
or DISCONNECTED shard. A PR or queue shard stops at its first failed test
(`--notest_keep_going`); the first shard that does not pass ends the gate red, and the
controller cancels its admitted siblings. A record keeps going and lists every failure.
A shard whose Bazel run ends with any exit code other than 0, 1 (build failed) or 3 (tests
failed) before its selected tests have results, and with no test that ran and failed, has no
verdict (exit 75). It names no failed test.
No shard is retried to obtain a pass.

TL3 keeps BuildBuddy priority if a fallback batch arrives while its queue is backed up; this cutover adds no scheduler.

### Find the merge that broke a record

A user timer on dl380g10 (`record-backstop.timer`, owned by the Fleet lead) starts the full record with
`workflow_dispatch` 3 minutes after each slot: 00:17, 06:17, 12:17 and 18:17 UTC. It starts one only when main moved since
the last finished record, green or red; it runs no test on the host. GitHub's own schedule came hours late or not at
all, so the repository has none (#3874). If a slot has no record 30 minutes later, the CI-speed watch reports it; a lead
then starts one with `gh workflow run ci.yml -R RobTand/prismaquant --ref main`. One red record can cover several
merges. Bisect only the targets that failed, not the whole suite:

1. Read the failed targets and their architecture from the red record's `failed test <label> on <arch>`
   annotations.
2. List the merges that the record covers:
   `git log --first-parent --merges --format='%H %s' <last green record head>..<red record head>`.
3. Bisect over that list. At each merge head, run only the failed targets with `bbrun.py`, the
   BuildBuddy client that holds the API key, in an admitted Flux job:
   `bazel test --config=remote --config=ci --noremote_accept_cached --nocache_test_results <targets>`.
   For arm64, use `--config=remote-arm64`. N merges take about log2(N) runs.
4. The first merge where a target fails is the cause. Tell the owner of that merge.
   If the target passes when you run it at the red record head, the failure is flaky, not a merge.
   Report it to the owner of the test. Do not rerun the record to get a green result.


Upload scanning runs in-process under Python 3.14 with stdlib `compression.zstd`.
The vendored scanner remains unchanged. The upload boundary refuses every
`not scanned` diagnostic, including scanner exit zero, and refuses a missing
Python 3.14 runtime or zstd decoder. Remote requests enter through the runner's
Python 3.12, then provision managed Python 3.14.0 with uv 0.9.5 before fetching a
manifest or running a shard. Bootstrap children receive an allowlisted environment
without credentials. The parent replaces its own interpreter before scanning.
Only valid UTF-8 JSON receipts are uploaded.
Raw execution logs, BEP, archives and other diagnostics stay private on the
runner. Renaming a binary artifact to `.json` does not make it eligible.

The test population retains Python 3.12. Scanner unit tests isolate runtime
admission; CLI tests isolate interpreter bootstrap. Each trusted upload step
runs tests/bazel_scan_runtime.py explicitly with Python 3.14 to prove native
zstd decoding and JSON publication. No runtime-dependent test is skipped.
Build-only partitions invoke bazel build and publish a successful shard record
on build success. Post-merge still rejects zero test executions across an
entire architecture, while a library-only partition may execute zero tests.
