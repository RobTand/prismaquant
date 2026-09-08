# Whole-target joint-statistics planner — 2026-09-08

Issue #385. `plan_joint_statistics_target_windows` uses the same extracted
module validation, activation grouping and FP32 matrix requirements as both
joint projection leases. It emits immutable target-name windows under an
explicit statistics cap. It neither installs observers nor executes replay.
Source/cotangent/graph/PWC/transient admission remains a separate obligation.
No probe arithmetic, format gate or production default changes here.

Grouping retains the lease's original format insertion order. Equal activation
receipts with different dynamic callable objects remain different groups;
static served QDQ ignores the unused dynamic callable. Plans persist ordered
format rosters and activation receipts with group indexes, never Python IDs,
callables, FormatSpecs, modules or tensors. Target names sort lexicographically
for deterministic greedy whole-target windows. Invalid coverage, module/spec
types, aliases, static scale/geometry, backend, cap and oversized targets refuse
before hooks or statistics allocation.

The CPU geometry test uses meta Linear modules with the actual largest GLM
layer's 289 `[4096,2048]` and 578 `[2048,4096]` projections. Static A4 plus
dynamic A8 and identity candidate groups charge 96 MiB per target:
87,275,077,632 bytes total (81.28125 GiB). A 32 GiB cap yields windows of
341, 341 and 185 names. The widest first-three-layer dense target
`[4096,12288]` requires 576 MiB with those groups and refuses a one-byte-smaller
cap. These are geometry/contract derivations, not actual GPU allocation or a
full-model fit measurement.

PrismaBuild CPU validation on dl380g10 passed 146 tests, with two explicit
native-only skips, across four independently admitted shards (two workers,
six GiB per shard, native threads one). This includes planner policy, source
retention, packed projections, existing operator and signed leases, streamed
and microbatch arithmetic, projection backend, architecture and staleness:

- `ecb9f254c210`: 43 passed.
- `b7b27d72a8a7`: 44 passed, 2 skipped.
- `d0754432b1ff`: 29 passed.
- `853c0aa04442`: 30 passed.
- Compile action `bdae96797f33520bd8a64170aee3cc8cc0ea885ab621341bdd233f14f5c7687c`
  compiled the three changed Python modules successfully.

The first run `0d476f86d1a1` passed 90 tests and skipped two native-only tests,
with one new test incorrectly expecting ValueError for the existing
ActivationScaleContractError. The test was corrected to the established
exception; implementation behavior did not change for that failure. An initial
pbtest client command rejected unsupported `-q` report forwarding before any
submission; the accepted fanout used its ordinary report options.

All validation artifacts are retained under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/joint-statistics-target-windows-01/`.
`cpu-final.json` gives file populations and per-shard output; `audit.json` binds
actual terminal state, source snapshot, CAS payload bytes and cleanup. No GPU
benchmark, latency, energy or throughput claim follows from this planner.
