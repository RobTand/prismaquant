# Hosted merge gate

Every PR push runs `pr-gate`. Its required `Graph impact` job runs
`tests/test_graph_impact.py::test_graph_impact_gate` in parallel with the shards,
with full git history and the event's base commit. Ordinary PRs use the merge
base; authenticated Mergify drafts use the actual draft base. Same-basename
Git renames align base paths before edge comparison. The graph baseline stores
numeric limits and policy, with no edge lists. The job needs only pytest. A failed,
skipped, cancelled, or missing graph verdict fails the aggregate. Its JUnit must
contain exactly one passing graph test. The live test is excluded from shard
execution and duration placement; its synthetic unit tests remain in the normal
population. This job has no test-budget waiver.

PQ's baseline uses `"test_to_test": "reach"`, a minimum edge count, and
`"test_to_test_strict_by": "2026-11-16"`. It has no count-based switch trigger.
After that date in UTC, the live check fails until you set `"test_to_test": "strict"`
in `tests/fixtures/graph_impact_baseline.json` and fix or allowlist any new
test→test edges. Injected dates cover before, on, and after the deadline, and the
live check also asserts the real UTC date. The shared core remains unchanged.

TODO after PR A: switch the graph job from the Python-import bridge to Bazel's
`deps.json` plus the source-label query fallback described in [bazel.md](bazel.md).

The selector follows reverse imports, relative imports,
test helpers, and literal repository paths. A change it cannot explain selects the
full population. Changes to gate tooling, workflows, conftest files, pytest configuration,
constraints, dependency declarations, or the Tessera pin select the full population.
Non-Python runtime files under `prismaquant/` or `tools/`, and `run-pipeline.sh`,
also select the full population. Missing or deleted non-documentation paths select
the full population even when a test names a same-named temporary fixture.
Wildcard-only path references explain nothing.
Documentation changes can select no tests. The plan records every selected file and reason.

A same-repository Mergify batch draft runs `gate` on its integrated head SHA.
Its branch starts `mergify/merge-queue/`, and its author must be `mergify[bot]`
with numeric ID 37929162. A same-repository PR with `gate-canary` also runs the
full gate. Labels take effect on the next push or reopen. Label events do not
trigger this workflow, so Mergify's queue labels cannot restart `pr-gate`.

`GATE_SHARDS` in `.github/workflows/gate.yml` is the full-population shard count.
Both gates use the same duration-balanced matrix. Small selections can use one
job; a large selection fans out. Measured nodes count toward the useful shard
limit, so a PR selecting one heavy file can still use several shards. The partitioner places the longest file or node on the shard with the earliest
projected finish on its assigned runner. Unknown files cost the nearest-rank p90 of recorded
file times, or one second before the first inventory. Every selected node executes
exactly once. Initial placements start on ARM in both modes; full workflow reruns
swap the architecture assigned to every shard.

The runners are GitHub-hosted `ubuntu-24.04` and `ubuntu-24.04-arm`. CI uses no
fleet admission or fleet runner. Python 3.12, CPU torch, the resolved Tessera
commit, and `pip install -c .github/ci-constraints.txt -e '.[test]'`
preserve the existing CPU installation. The retired execution system's client is
no longer a dependency: no dependency group installs it and tests that drove it were
ported to the native stack or deleted.
Pip caches include dependency declarations, constraints, and the Tessera pin.
Native libraries get one thread each. Pytest workers are bounded by inherited
CPU affinity and two GiB per worker after reserving one GiB for the runner. Work
stealing shares long files across those workers, as in the post-merge suite.
Ordinary files belong to one hosted shard. A file above the per-shard target, total estimated time divided by shard count, splits by collected
pytest node ID. Recorded node durations drive longest-first assignment. Every
shard collects split files and executes only its assigned nodes. A new node uses
SHA-256 of its exact node ID modulo the shard count. Deleted nodes disappear at
collection. Module fixtures run once per worker on each shard that executes part
of the file.

The matrix, each shard, and the aggregator recompute the same assignment from
committed durations. Split-file collection must match across architectures.
JUnit must match each shard's selected nodes exactly once; the aggregate proves
unique execution across shards too. Collection differences or missing evidence
produce exit 75. A shared file budget still applies to the sum across shards.

Tests marked `own_cgroup` run after their shard's ordinary tests, without xdist,
in a transient systemd scope containing only that pytest process. Both phases
contribute JUnit evidence and the shard verdict. JUnit carries the collected
`gate_nodeid` separately from the defining source location of imported tests.
Coverage, exact budgets, and duration evidence use that collected identity. Optional-dependency module skips
and complete deselections are recorded separately from incomplete collection.

## Verdicts and retries

Exit 0 is a pass, 1 is a test or static budget-policy failure, and 75 means no verdict.
The retry planner retries only missing or invalid verdicts, once, in another job.
A test failure survives a later pass and is never retried. Lost runners and setup
failures can retry. Every record binds the base SHA, head SHA, population version,
mode, assigned files, and the node partition. The final check requires complete, unique coverage.
Artifacts are separated by workflow attempt, shard, and execution attempt. Use
"Re-run all jobs" to rerun a workflow. "Re-run failed jobs" cannot reuse the old
plan artifact under the new attempt number and therefore cannot produce a verdict.
Full reruns swap every shard's architecture. Pytest failures, including xdist
worker crash reports, remain final failures. There is no automatic test retry.

## Time budgets and quarantine

`tools/ci/gate_budget.json` defines the default and exact-node exceptions.
Every measured overrun is a warning, regardless of its size. It never fails a
passing test or starts a second pytest execution. Both shard `result.json` and
the final aggregate record expose `budget_warnings`, a list of objects with
`nodeid`, measured `seconds`, and the configured `budget_seconds` limit.
For a file budget, `nodeid` is the file path. Warnings also appear in JUnit
`system-out`, shard job summaries, and the final job summary. Summaries show a
bounded preview; the JSON records retain the complete list.

Assertion failures and hard pytest or process timeouts still fail, with no
budget rerun. Missing verdicts retain the infrastructure retry policy above.
Rob changed time overruns to warnings on 2026-11-03 because rerunning slow tests
adds load to the shared executors. The budget-growth policy below still applies.
Four files have measured `file_seconds` exceptions: the duplication baseline,
`tessera_legal_domain`, `quality_prefill_dry_run`, and `tessera_campaign`. Their module fixtures or
cached derivations charge setup to whichever test runs first on a worker. Each
file limit covers the sum of all JUnit setup, call, and teardown durations across
all shards. The aggregate warns when that sum exceeds the file limit. The hosted
maximum and source attempt remain in `file_evidence`. File exceptions have no overlapping node exceptions.
All other tests retain per-test limits.
The default, node/file exception lists, and each exception can only decrease against the PR
base. An absent base policy permits initial setup; an unavailable base commit
produces no verdict. Exceptions are `ceil(2 * largest hosted time)` and retain
hosted run IDs. Development-run times do not set hosted limits.

The initial policy uses fourteen calibration attempts: 41275735452/1,
41277228478/1, 41278311643/1, 41278869921/1 and /2, 41280260494/2,
41281908452/1, 41282567022/1, 41284744088/1, 41285636094/1 and /2, 41287142037/1, 41292593909/1, and 41293449843/1. Both passing and failed attempts contribute
to 90 node exceptions and four file exceptions. The last two attempts exposed
moving legal-domain and campaign setup charges after node splitting. This remains initial
bootstrap calibration because the PR base has no gate budget policy. Once
installed, the same shrink-only checks prevent adding or enlarging exceptions.
The native-import predecessor regression measured 36.245 seconds on x64 and keeps an exact 73-second exception. It retains total JUnit setup, call, and teardown time. Fixture setup can move
between the first tests on xdist workers, so both architectures and swapped
placements contribute maxima rather than successful runs alone. Slow tests keep
measured exceptions. [Issue #3200](https://github.com/RobTand/prismaquant/issues/3200)
owns reducing these costs; a higher measured limit does not make a test faster.
The separate hang timeout remains 600 seconds per test. The budget file records
its exact run evidence. Scheduling costs in `gate_durations.json` use 41285636094 attempts 1 and 2,
and 41290209398 attempt 2. Each file and node uses its largest architecture
median. The matrix applies cost factors of 1.0 for x64 and 0.625 for ARM. The paired
hosted median x64/ARM ratio across 135 files above five seconds is 1.716;
1.6 is a conservative scheduling ratio. Each longest-first placement minimizes
projected finish time on the actual runner. The recorded matrix carries the
factors into assignment, retry planning and aggregation. A full rerun swaps
factors with the runner order; every shard recomputes the same owners within
that attempt. New unmeasured nodes still use the stable hash owner. These
scheduling estimates do not set test budgets.

After bootstrap, a budget cannot be raised by editing the JSON, adding a label,
or rerunning a failed job. First fix the expensive test or fixture and remeasure.
If a verified runner or dependency change invalidates the baseline, the readiness
lead must review an explicit policy migration in tooling plus a new hosted
baseline and final-head qualification. The current runner offers no override.
`tools/ci/gate_budget_migrations.json` is that migration record. Each migration
names the Git blob of the base `gate_budget.json` it was measured against
(`base_policy`), the base commit, the reason, and per node hosted evidence from
both `ubuntu-24.04` and `ubuntu-24.04-arm`. Its limit must equal
`ceil(2 * largest hosted time)`. `gate_run` accepts a new or larger node exception
only for a listed node, only up to that limit, and only while the PR base carries
that exact policy blob, so a merged migration grants nothing to a later PR. The
default, file budgets and every unlisted node stay shrink-only. A malformed record
fails closed. The first migration (#3223) recorded the dependency change of a retired client install.

`tools/ci/gate_quarantine.json` replaces `.github/hosted-known-failures.txt`.
The legacy file at main `8d72e303e2f070925077ce9e072b254df3856909` had 54 active
entries, not 72. The initial policy carries those entries plus the observed ARM
exit-ownership race owned by #3197. As with budgets, a base predating this policy
permits initial setup. Once present, quarantine can only shrink against the PR
base during an ordinary code change. A whole-file exclusion may narrow to an
exact node owned by the same issue. New targets, broader targets, or owner changes
require a PR touching only the quarantine file and optional documentation.
A fix can always remove an entry alongside its code changes.

Planning checks each distinct owner using GitHub's public REST API. Owners must
be open issues, not PR numbers. Closed or missing owners fail. An unavailable,
rate-limited, malformed, or timed-out API returns exit 75, never a pass.
Whole-file exclusions happen before import. Node exclusions use exact collected
IDs, so excluding `test_a` never excludes `test_a_more` or an unlisted parameter.
The post-merge full suite uses the same exact exclusion hook. Cgroup membership
comes from collected pytest markers, including aliases and hooks, rather than
source substrings. Quarantine owners repair their failures and remove entries.

## Secrets

`gate_run.py` strips `BUILDBUDDY_API_KEY` and `MERGIFY_TOKEN` from pytest's
child environment. Keep this explicit denylist in step with workflow secrets.
Each upload in `gate.yml` requires a successful preceding
`tools/ci/check_no_secret_leak.py --env BUILDBUDDY_API_KEY --env MERGIFY_TOKEN`
scan, with values confined to that step's `env`. `ci.yml` has no uploads;
the contract test covers both workflows. The shared scanner detects raw,
base64 and URL-encoded values, including gzip members, and never prints values.
It uses standard-library `compression.zstd` when available; otherwise zstd is
reported as "not scanned". Unset/empty secrets do not fail fork PRs. Exit 1
means a detected value; exit 75 means unreadable or incomplete input.
Vendor the core and SHA-256 pin in step across PrismaQuant, TL3
and Tessera. Static tests reject secret command-line flags and expressions
outside step-level `env`. Authentication belongs in the external
`~/.config/buildbuddy/bazelrc` or a Python API client's environment-fed HTTP header.

## Reproduce a shard

Plan and inspect the files without executing tests:

```sh
python tools/ci/gate_select.py --base "$BASE_SHA" --head "$HEAD_SHA" --mode full \
  --output plan.json --files-output files.txt
python tools/ci/gate_partition.py matrix --files-from files.txt \
  --durations tools/ci/gate_durations.json --shards 12 \
  --runners ubuntu-24.04-arm,ubuntu-24.04 --output matrix.json
python tools/ci/gate_partition.py shard --files-from files.txt \
  --durations tools/ci/gate_durations.json --shards 12 --matrix matrix.json --index 0 \
  --output-files assigned.txt --output-assignment assignment.json
```

Agents submit development tests through admitted Flux jobs from a committed, detached
worktree. The interpreter must match `tools/resolve_tessera_dev_pin.py`.
Submit the gate contracts as admitted Flux jobs (see the flux skill); the file list is:

```text
tests/test_gate_actions.py tests/test_gate_budget.py tests/test_gate_partition.py
tests/test_gate_run.py tests/test_gate_select.py tests/test_gate_workflow.py
tests/test_gate_node_partition.py tests/test_gate_evidence.py
```

Inside an admitted job or the hosted job, run the assignment directly:

```sh
python tools/ci/gate_run.py --plan plan.json --assignment assignment.json \
  --workers 4 --result result.json --junitxml junit.xml
```

Check the terminal record, logs, and CAS receipt. Failed jobs normally have
terminal evidence without a result publication. A submission acknowledgement is not
a pass. Six fleet-data files require a declared manifest and are recorded as
excluded from the undeclared-data development inventory; CUDA bodies remain skipped.

## Reduce qualification evidence

`gate_evidence.py hosted` reads downloaded REST run metadata, job metadata and
gate artifacts. The final aggregate's effective shard artifacts take precedence
over individual attempts. Duration costs use the largest architecture median;
exception candidates use twice the largest observed test duration, rounded up.
Both retain their hosted run evidence.

## Qualification measurements

The fixed-main inventory at `8d72e303e2f070925077ce9e072b254df3856909`
completed three 32-shard repeats. It observed 22,509 nodes and measured 1,312
files, with no incomplete outcome records. All 96 effective actions have public
terminal evidence across x64 and ARM workers. The 37 passing
actions have CAS receipts; the other 59 actions contain final test failures.

There are 76 nodes that failed all three repeats, 228 that changed between pass
and fail, and 39 that changed between pass and skip. Worker architecture, filesystems, and
installed dependency availability vary across repeats. These are observed
outcome changes, not evidence that every affected test fails randomly.
[The inventory evidence comment](https://github.com/RobTand/prismaquant/issues/3180#issuecomment-6554113772)
records the action and exclusion evidence.

Each action reserved one CPU and one native thread, at priority -10, with no GPU
and portable placement. The first 2 GiB shard-0 attempt hit its memory cap.
The same files completed with a 4 GiB reservation, which the later repeats also
used. The maximum measured aggregate cgroup peak was 3,058,761,728 bytes.
The six manifest-dependent exclusions and skipped CUDA bodies remain unqualified.
Fleet Python 3.14.4 and torch 2.11 differ from hosted Python 3.12 and torch 2.15.

The captured 20-PR replay is rerun after selector changes. The PR body records
the exact replay head and its selection sizes from a 1,328-file population. These
counts measure work selected, not selection accuracy. Shadow selections are uploaded but are not yet
compared automatically against full-gate failures. String-based production module
or subprocess references can remain outside the import graph, so that comparison
is still needed before making stronger accuracy claims.

Calibration history includes successful and failed runs across different heads
and layouts. Those runs set initial budgets and are not final-head qualification.
The first repaired 16-shard trial, run 41287142037/1, took 511 seconds and
failed only the shared-fixture budget attribution described above. The matching
12-shard attempt took 514 seconds; both were limited by one legal-domain file
with the same four workers. That is no demonstrated speed advantage for 16,
so this configuration retains 12 shards and leaves more capacity for other gates.
The earlier successful 16-shard sample was 523 seconds and a 12-shard sample was
506 seconds, both measured from workflow start. A swapped 12-shard sample took
822 seconds and failed its budget verdict. These samples do not establish a
latency guarantee or a shard-count speed advantage. Issue #3200 owns the long
legal-domain derivation tail. No CPU profile was captured on that slow hosted VM,
so its cause remains unproven.

Final qualification requires at least three passing full `gate` attempts on one
unchanged head, including a passing swapped rerun, with budget warnings recorded. Record
run ID, full head, attempt, architecture layout, and wall time from run creation
or the timestamp immediately before the REST rerun request to the final `gate`
verdict. Include failing attempts when assessing reliability. The target is p90
at most ten minutes, with five minutes preferred. Measure 14 or 16 shards if the
12-shard layout cannot meet that reliably. The PR body carries the final-head
table, so recording a completed canary does not itself change the qualified head.

Before node splitting, run 41290209398 at
`f5c1cb933b83d6cc7ffe5f1aa3921afc903825a7` passed twice. The lead measured
526 seconds on attempt 1 and about 628 seconds on swapped attempt 2. Shard 0
held the legal-domain file and took 458 seconds on ARM and 576 seconds on x64.
The retained swapped JUnit sums to 1,561.842 seconds for that file. Increasing
the shard count could not divide it. The first split-file attempt, 41292593909/1 at
`b762e75702a2baccdecd7d9be0d5b210fe504e5e`, ended after 409 seconds. Its test
tail was 269 seconds. It failed the primitive-digest inventory and four budget
attributions, so it does not qualify. The digest repair uses the shared owner.
Moving legal-domain setup charges use the existing complete-file budget
mechanism. Calibration retains the larger historical complete-file maxima:
2,463.380 seconds for legal-domain, 222.950 seconds for quality-prefill, and
146.434 seconds for campaign, all from 41293449843/1. Their initial limits
are 4,927, 446, and 293 seconds, respectively. Campaign uses a cached menu
build whose charge moves among its first consumers. The second split-file
candidate, `1440db9983e609eb075d3a7a3794036045057f10`, failed run
41293449843/1 after 512 seconds because of that charge. Its test tail was
367 seconds on x64 versus 262 seconds on ARM. The weighted trial at
`0a6e9f19828210eaabeb8ca21a8f94aeb0e46e2d`, run 41294639469/1, passed
in 481 seconds with budgets enforced and 22,670 unique executed nodes. Its
test tail was 351 seconds. That modest observed improvement supports retaining
the runner factors; one trial does not establish a latency guarantee. Splitting removes the whole-file
scheduling constraint; issue #3200 still owns reducing these tests themselves.

Final-head node-split qualification is recorded in PR #3192 with exact heads,
attempts, budgets, and node coverage. Elapsed intervals use run creation for
attempt 1 and a UTC timestamp immediately before each full rerun request.
Actions occasionally reports job timestamps one hour off across an hour
boundary. The evidence reducer uses elapsed seconds modulo 3,600 and checks
them against actual request and completion-observation times. It retains raw
REST timestamps and never counts an incomplete attempt as a pass.

Of the PB inventory's 228 pass/fail flips, 223 were in 19 files that passed all
observed hosted executions. Likewise 75 of its 76 consistently failing nodes
remain in the hosted population. Fleet Python 3.14 and torch 2.11 differ from the
hosted environment. These observations justify keeping those tests covered;
they do not prove an absence of flakes. No dedicated five-repeat per-architecture
hosted experiment was performed.

Ordinary draft PRs run `pr-gate` too. Mergify drafts need the full gate and the
current workflow deliberately keeps feedback on other drafts. This can consume
up to the full shard matrix; account capacity is shared. The staged FP8 DSv4 oracle now uses the existing `own_process` boundary. A real
native-import predecessor regression verifies its staged-read assertions and
retains the parent process's native-override refusal guard.

The shell-gate fixture in `tests/test_run_pipeline_gates_execute.py` owns a unique
scratch directory per worker. Finishing one worker cannot delete another worker's
active model. Its controlled subprocess environment also bounds native threads.
Both regressions failed in admitted jobs before the fixture repair.
Existing PR heads that
predate the gate tooling fail with a message to merge `main` and push again.

The queue configuration lands in the same PR as this gate. Landing order:

1. Wait until PQ's queue is empty.
2. The lead changes classic branch protection to require `linked issue` only.
3. The lead queues this PR with `@mergifyio queue` using the old in-place queue,
   which then needs only `linked issue`.
4. Once merged, the new `.mergify.yml` queues by label with `pr-gate` and `gate`.

The post-merge tests job keeps its name and runs on main pushes and manual
dispatches. PR import and boundary checks remain in `ci.yml`. Only the lead changes
branch protection or queues this PR. See [the merge queue guide](merge_queue.md).
