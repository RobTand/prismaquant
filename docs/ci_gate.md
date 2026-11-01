# Hosted merge gate

Every PR push runs `pr-gate`. The selector follows reverse imports, relative imports,
test helpers, and literal repository paths. A change it cannot explain selects the
full population. Changes to gate tooling, workflows, conftest files, pytest configuration,
constraints, dependency declarations, or the Tessera pin select the full population.
Documentation changes can select no tests. The plan records every selected file and reason.

A same-repository Mergify batch draft runs `gate` on its integrated head SHA.
Its branch starts `mergify/merge-queue/`, and its author must be `mergify[bot]`
with numeric ID 37929162. A same-repository PR with `gate-canary` also runs the
full gate. Labels take effect on the next push or reopen. Label events do not
trigger this workflow, so Mergify's queue labels cannot restart `pr-gate`.

`GATE_SHARDS` in `.github/workflows/gate.yml` is the full-population shard count.
Both gates use the same duration-balanced matrix. Small selections can use one
job; a large selection fans out. The partitioner assigns the longest files to the
least-loaded shard first. Unknown files cost the nearest-rank p90 of recorded
file times, or one second before the first inventory. Every selected file occurs
exactly once. Workflow reruns swap the architecture assigned to each shard.

The runners are GitHub-hosted `ubuntu-24.04` and `ubuntu-24.04-arm`. CI uses no
PrismaBuild admission or fleet runner. Python 3.12, CPU torch, the resolved Tessera
commit, and `pip install -c .github/ci-constraints.txt -e '.[test]'` preserve the
existing CPU installation. Pip caches include dependency declarations, constraints,
and the Tessera pin. Native libraries get one thread each. Pytest workers are
bounded by inherited CPU affinity and two GiB per worker after reserving one GiB
for the runner. Work stealing shares long files across those workers, as in the
post-merge suite; each file still belongs to exactly one hosted shard.

Tests marked `own_cgroup` run after their shard's ordinary tests, without xdist,
in a transient systemd scope containing only that pytest process. Both phases
contribute JUnit evidence and the shard verdict. JUnit carries the collected
`gate_nodeid` separately from the defining source location of imported tests.
Coverage, exact budgets, and duration evidence use that collected identity. Optional-dependency module skips
and complete deselections are recorded separately from incomplete collection.

## Verdicts and retries

Exit 0 is a pass, 1 is a test or budget failure, and 75 means no verdict.
The retry planner retries only missing or invalid verdicts, once, in another job.
A test failure survives a later pass and is never retried. Lost runners and setup
failures can retry. Every record binds the base SHA, head SHA, population version,
mode, and assigned files. The final check requires complete, unique coverage.
Artifacts are separated by workflow attempt, shard, and execution attempt.

## Time budgets and quarantine

`tools/ci/gate_budget.json` defines the default and exact-node exceptions.
A test above its limit warns through 1.5 times that limit and fails above it.
The default, exception list, and each exception can only decrease against the PR
base. An absent base policy permits initial setup; an unavailable base commit
produces no verdict. Exceptions are `ceil(2 * largest hosted time)` and retain
hosted run IDs. PrismaBuild inventory times do not set hosted limits.

Hosted trials 41275735452 and 41277228478 measured Python 3.12 CPU tests
on x64 and arm64. The 66 bootstrap exceptions are twice the largest observed
test time, rounded up. Enforcement remains enabled. Work stealing can move
shared-fixture setup between testcases, so the exception evidence includes both
schedulers. Swapped reruns extend architecture coverage before handoff.
The separate hang timeout remains 600 seconds per test.

`tools/ci/gate_quarantine.json` replaces `.github/hosted-known-failures.txt`.
At main `8d72e303e2f070925077ce9e072b254df3856909`, the legacy list has 54 active
entries. Each now names an open owning issue and its inherited failure reason.
The gate ignores whole-file entries before import and deselects node entries.
The ARM exit-ownership failure found in canary 41277228478 adds one exact-node
entry owned by #3197. Its unchanged production code passed an earlier hosted
sample, then classified its own exiting child as a foreign PID. Quarantine owners
reproduce and repair the failures, then remove their entries.
The post-merge full suite uses the same quarantine.

## Reproduce a shard

Plan and inspect the files without executing tests:

```sh
python tools/ci/gate_select.py --base "$BASE_SHA" --head "$HEAD_SHA" --mode full \
  --output plan.json --files-output files.txt
python tools/ci/gate_partition.py matrix --files-from files.txt \
  --durations tools/ci/gate_durations.json --shards 12 \
  --runners ubuntu-24.04,ubuntu-24.04-arm --output matrix.json
python tools/ci/gate_partition.py shard --files-from files.txt \
  --durations tools/ci/gate_durations.json --shards 12 --index 0 \
  --output-files assigned.txt
```

Agents submit development tests through PrismaBuild from a committed, detached
worktree, at priority -10. The interpreter must match `tools/resolve_tessera_dev_pin.py`.
For example, submit the gate contracts with the published client:

```sh
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbtest.py \
  --checkout "$CLEAN_WORKTREE" --python "$PINNED_PYTHON" \
  --shards 6 --workers-per-shard 1 --threads-per-shard 1 \
  --cpus-per-shard 1 --mem-gb 2 --priority -10 \
  --json "$REPORT" tests/test_gate_actions.py tests/test_gate_budget.py \
  tests/test_gate_partition.py tests/test_gate_run.py tests/test_gate_select.py \
  tests/test_gate_workflow.py
```

Inside an admitted action or the hosted job, run the assignment directly:

```sh
python tools/ci/gate_run.py --plan plan.json --files-from assigned.txt \
  --workers 4 --result result.json --junitxml junit.xml
```

Check the terminal record, logs, and CAS receipt. Failed actions normally have
terminal evidence without a success receipt. A submission acknowledgement is not
a pass. Six fleet-data files require a declared manifest and are recorded as
excluded from the undeclared-data development inventory; CUDA bodies remain skipped.

## Reduce qualification evidence

`gate_evidence.py inventory` reads the three `pbtest --json` reports. Export
public `pb_action` MCP responses as a JSON object keyed by action key, then pass
that file with `--action-evidence`. The reducer records aggregate cgroup memory
peaks only from complete responses. It does not read PrismaBuild's internal
queue or receipt schemas. Missing telemetry remains unknown.

`gate_evidence.py hosted` reads downloaded REST run metadata, job metadata and
gate artifacts. The final aggregate's effective shard artifacts take precedence
over individual attempts. Duration costs use the largest architecture median;
exception candidates use twice the largest observed test duration, rounded up.
Both retain their hosted run evidence.

## Qualification measurements

The fixed-main PrismaBuild inventory and hosted 12-versus-16-shard trials are in
progress. There is no p50, p90, architecture qualification, or budget-enforcement
claim until the observed runs are recorded here.

After the gate merges and its checks pass, the readiness lead removes
`tests (py3.12, cpu)` from classic branch protection, leaving `linked issue`.
The tests job retains its name and runs on main pushes and manual dispatches.
The Mergify queue requires `pr-gate` to queue and the integrated `gate` to merge.
The `import surface` and `boundary allowlists only shrink` jobs remain in CI.
