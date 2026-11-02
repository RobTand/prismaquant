# Dependency graph impact

`tests/test_graph_impact.py::test_graph_impact_gate` runs in its own required
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
new test-to-test edge. PQ uses reach mode until November 16, 2026: an edge may
inherit the importer's base package maximum, but deeper reach fails. The total
edge count may rise by at most five above its recorded minimum, and baseline
writes retain the minimum. [The CI gate guide](ci_gate.md) describes PQ's
mandatory switch test and date-only trigger. Every SCC is checked for growth
or mergers, even when an unrelated larger cycle hides the size change. The
SCC maximum can only shrink.
U and LOC-weighted U are reported trends and never fail a PR. Baseline writes
keep their minimum. The diagnostic attributes structural growth to added edges.

The global fallback budget is the ceiling of the initial p90 source reach. Existing
tests above that budget are listed with reasons in
`tests/fixtures/graph_impact_allowlist.json`. Those exceptions and their
reasons can only be removed. The graph check itself has a reasoned exception
because it inventories the whole graph. New tests do not inherit exceptions.

## Shared core

`tools/ci/graph_impact.py` is the stdlib-only core shared byte-identically by
PrismaQuant, PrismaBuild, Tessera and TrustLadder. Its inputs are opaque test
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

The old `import_fanout` CLI forwards to this check. Its original evidence
parser supplies the migrated figures in the same baseline, preserving each
recorded minimum. There is one baseline and one live gate test. Writing the
baseline refuses structural losses and keeps every recorded numeric minimum.
SCC gains must be recorded in the same PR; U and edge-count gains are
recorded on explicit baseline writes. New-edge checks use only the aligned
base graph and the head graph. No stored edge list judges whether an edge is new. New source files inherit a package when
their existing-test consumers fit an existing peer's base consumers; that
ordinary package growth is reported in the summary instead of failed. A committed baseline holds
`schema`, `figures`, `tolerances`, `reach_budget`, `test_to_test`,
`test_to_test_count`, and `test_to_test_strict_by`. Reasoned exceptions remain
in the separate allowlist. Neither baseline reads nor writes retain edge identities;
`import_fanout_baseline.json` is removed. The frozen core receives transient
edge groups from the live base graph through the PQ adapter.

The full check belongs in Bazel's runner-VM lane because attribution requires
a git checkout and history. Without git, it reports `no base available` and
still checks absolute figures, the count limit and every test's reach budget.
The shrink-only exceptions preserve existing broad tests in that lane.

# Measure the remaining test-selection cost

`tools/bazel/detangle_plan.py` ranks potential dependency cuts by selected test
seconds saved per estimated changed line. It performs static analysis over a
stored per-file coverage map. It does not run tests or apply the cuts.

Use measured per-file timings and the same coverage population for both costs:

```sh
python tools/bazel/detangle_plan.py \
    --root . --revision origin/main \
    --timings /path/to/alone.tsv \
    --dynamic /path/to/coverage.json.gz \
    --gate-timings /path/to/gate-timings.json \
    --out /path/to/plan
```

The command writes `plan.json` and `plan.md`. JSON retains each merge's selection,
its fallback paths, all ranked symbol evidence, and input provenance. Keep large
coverage maps and measurement outputs outside the repository.

## Read the costs

For each of the last 100 first-parent merges, the planner uses
`git diff M^1 M --name-only --no-renames`. It sums the runtime of every selected
test file, then reports the mean over merges. Run-alone timings take precedence
over hosted `tools/ci/gate_durations.json`. Missing measurements use the hosted
median and appear explicitly in `runtime_sources.missing`.

Static selection follows the graph adapter's imports, conftests, data inputs,
and launched Python programs. Dynamic selection uses the repository source
files executed by each test. A changed test also selects itself. Both selectors
apply exactly the same fallback policy for build, lock, configuration, and
workflow paths. `floor_share` is the fallback contribution divided by total
static cost. Graph-only costs exclude fallback merges from their denominator.

Coverage input has this shape, optionally compressed with gzip:

```json
{"sha": "measured commit", "data_reads": false, "subprocess_coverage": true,
 "tests": {"tests/test_example.py": {"files": ["package/example.py"],
                                      "exit": 0, "report_exit": 0}}}
```

Collect each gate file in a fresh process with the hosted gate's environment,
quarantine, and marker exclusions. `coverage run --source=. -m pytest <file>`
records imports as execution. Coverage's subprocess patch can instrument Python
children too. Record failures, skips, and unavailable dependencies in the
measurement artifact. Missing coverage rejects the input. Test failures or
observed file/test pairs outside the static graph make the result provisional
and prevent a stopping claim. If data reads are not traced, the dynamic floor
ignores them. Reusing coverage after a refactor also requires a fresh sweep
before treating that revision's floor as measured.

## Read the candidates

The AST adapter reports imported symbols, aliases, attribute uses, owning
functions, and the target's top-level symbol closure. Effort estimates include
the definitions and their known importers. They are estimates of changed lines,
not implementation promises. Localizing an import alone does not change the
conservative Bazel graph. Its projected saving requires a corresponding target
boundary or symbol move. Path candidates identify exact literal evidence.

The cycle search uses the weighted Eades-Lin-Smyth feedback-arc heuristic from
the graph measurements. It tries deterministic orders and restores redundant
cuts. This yields an inclusion-minimal feedback cut, not an exact minimum.
Partition cuts split each SCC into large pieces. The planner scores these groups
as a unit, since individually useless edges can have a useful combined effect.
After each pick, it recomputes every remaining candidate's marginal gain with
bitsets. Results are ordered by seconds saved per estimated changed line.

The first stopping condition requires a gap below 10% of static cost. The
second requires both a best remaining cut below 2% and measured gate wall time
within 20% of its floor. Supply wall-time evidence as:

```json
{"fixed_overhead_seconds": 0.0, "gate_seconds": 0.0,
 "source": "measurement artifact or invocation URL"}
```

Use measured values in that file. If PrismaQuant lacks them, identify the
TL3 or PrismaBuild Bazel measurement used as a proxy in `source`. Without wall
evidence, the second condition remains false. The 80-core wall floor is fixed
overhead plus the mean longest dynamically selected test across the replay.
It is a lower bound, not a scheduling estimate. The report separates essential
execution cost, fallback cost, and accidental cost left after the ranked cuts.

## Use another repository or graph source

`Selection` is the graph contract: `edges` maps nodes to library dependencies,
`roots` maps test files to starting nodes, `data` maps nodes to non-propagating
file inputs, `extras` holds test-specific inputs, and `launches` maps programs to
child programs. File paths and test identifiers are repository-relative.

The cost, dynamic selector, SCC scoring, ranking, and stopping functions have no
PrismaQuant package assumptions. `detangle_repo.py` is the temporary Python graph
adapter. Tessera can replace that adapter and pass its fallback names, prefixes,
suffixes, and requirement-file patterns through `--config`. Config fields match
the `Config` dataclass. `--files-from` and `--history-root` support exported source
trees while replay history stays in the designated checkout.

The preferred next source is bazel-diff v42's `deps.json`, generated with
`generate-hashes --depEdgesFile deps.json`. Use the shared graph-impact reader
rather than adding a second parser. Feed its rule edges and source-file labels
through an adapter that preserves the test-file timing identifiers. Only Bazel
can establish which target boundary actually disappeared after a refactor.

To use the native source, add `--deps-json deps.json`,
`--bazel-tests tests.json`, and `--source-labels sources.txt`. `tests.json` maps
repository test files to the non-manual labels returned by `tests(//...)`.
`sources.txt` lists labels Bazel identifies as source files. Supply
`--graph-core /path/to/graph_impact.py` while that shared core is on its own
branch. `--query-json query.json` uses its streamed-query fallback when
`deps.json` lacks source-file labels. Both paths call that core's reader.

Native rule edges remain rule edges. Source-file labels become repository
paths for the replay. Symbol cuts are projected only across single-source
Python target boundaries. Multi-source targets appear in
`metadata.coarse_python_targets`; their internal cuts require a target split
before the planner can price them from a new Bazel graph.

`--source-revision` identifies an exported overlay independently of
`--revision`, which fixes the replay history. Coverage from another source
revision makes the dynamic result provisional. `--cheap-lines` sets the
per-candidate effort ceiling used for the estimate of accidental cost left
after all cuts below that ceiling. The default is 100 lines. The JSON report
also retains standalone SCC group scores, including groups outside the top 20.

Use `--population-from tests.txt` to hold the test-file population fixed when
comparing overlays. New tests on an overlay are then excluded from both costs.
Record that choice with the report. It avoids assigning invented coverage to
new files that were not part of the baseline sweep.

Replay requires full Git history. Fetch an initially shallow checkout with
`git fetch --unshallow` before running the planner.
