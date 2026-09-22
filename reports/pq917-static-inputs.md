# PQ #917 static prepared-input route — executor report

Branch: `fix/pq-917-static-prepared-inputs-20260922`, HEAD `34f8e0ccbf`
(implementation commit; this report follows as a second commit).
Base: fresh current main carrying accepted #910 (`2fabd11d7b`) ancestry
via `88fcdd4ce8` plus the adapted #909 source-only filter (`0dcec1a991`),
without #914's dynamic writer/gate ancestors. Trees/branches for #914
and #871 untouched (`muse-pq870-20260921` still clean at `c0e627d2d2`);
root handles PR disposition. Closes #917; attributes #900/#909; does not
close #870 and does not promote #914/#871.

## What was built

Static prepared-render read phases as ordinary immutable input staging
(per `reports/pq914-architecture-review.md`, static route items 1–7):

- `prismaquant/joint_layer_quanta.py`: new `PREPARED_INPUT_SCHEMA`
  (`...prepared_input.v1`), `executable_render_phase_name`, a
  `render_phases` flag on `quantum_executable_phase_names` (legacy order
  byte-identical without it), `check_prepared_input_windows` (schema /
  digest / window-coverage / member / entry validation) and
  `check_prepared_windows_against_resolved` (sealed membership vs live
  geometry). `build/bind/emit_quantum_executable_manifest*` accept
  `prepared_inputs`: roster digests must equal the sealed render
  prerequisite, the prepared digest must equal the sealed campaign,
  entries deduplicate through the existing v2 index owner, one
  `render-{w:02d}` phase seals immediately before that window's replays,
  and `annotations.prepared_input` carries the bound membership. The
  binder copies it into the bound `executable_readset` block.
- `tools/dispatch_joint_quanta.py`: production `quantum_argv` accepts an
  executable row only for the complete contract (wire digest, schema,
  prerequisite/campaign digest binding, window coverage, render phase
  with exactly the window's entries immediately before its replays,
  receipt binding, block–manifest agreement). Legacy sequencing-only
  rows keep the exact `ExecutableBindingUnsupported` refusal; a
  mover-reference-shaped annotation is not a read capability and
  refuses. Submitted rows stage the executable manifest through the
  existing `pbrun --data-manifest` path with manifest-order progress.
- `prismaquant/joint_cost_quantum.py`: `before_window` enters the
  render phase (executable rows only) before
  `observe_and_project_retained_windows` opens the PWC retained window,
  so the phase precedes every retained file open; skipped/resumed
  windows still enter it. After `resolve_quantum_windows`, sealed
  prepared membership is compared with the recomputed geometry and
  refuses (`QuantumIdentityRefused`, exit 3) before GPU work on
  mismatch. Readiness/integrity reuse the existing API: window-boundary
  `plan_retained_window` preflight (no deserialization),
  `require_file_load_sha256` digest binding, and the strict
  resolver/lease/tier path as the single read mechanism.
- `tests/test_stageb_prepared_render_inputs.py` (new, 15 tests):
  builder phases/ordering/annotations, legacy byte stability, binder
  block carriage, membership-vs-resolved matrix, normal-CLI dispatch
  accept, legacy refusal, foreign-roster and mover-impostor refusals,
  deployed PB phase-planner acceptance of render phases, strict-PWC
  two-window serve with phase-before-open ordering /
  `bytes_from_pool == 0` / pin IDs / pin release, corrupt-payload and
  undeclared-render refusals, and read-phase reporting without pricing.

No `residency_map` / `residency_shard_reader` / `staged_lease` /
`layer_streaming` / `streaming_model` edits (R5 lane). No PB runtime
change: the deployed generation `d1c640e74d97-1790008584-459339399864`
is untouched; the `joint_cost_quantum` change is PQ-side progress
reporting plus a pre-work membership comparison, no new waiter, planner,
dispatcher, sharding, re-render, reprepare, or production-record
mutation. Full 512-sample / 360-window / parent / slice / chunk /
plan / prepared-payload surface preserved (diff touches 3 source files
plus the new test).

## RED (pre-fix, actual PB)

- `8c7c53b534d1`: 2 failed — manifest declares no prepared-render
  phases; production main refuses the complete contract.
- `5d8dd89f6ca7`: 2 failed — same gates after fixture correction
  (receipt digests, layout), failing for the missing route only.
- Full logs: `/home/rob/tmp/muse-pq870-scratch-20260921/red917{,b}.pbtest.json`.

## GREEN (post-fix, actual PB, priority -10, portable `gb10`)

- `8e1c117abeae`: `tests/test_stageb_prepared_render_inputs.py`,
  15 passed, rc 0, CAS receipt `ae1f759d…` / result `719dc611…`.
- Regression, all rc 0 with CAS receipts present:
  `ace33720554e` (19), `40c74c2d25d3` (33), `e8428b7e7fd3` (8),
  `f7e66406c073` (13), `fe6818a983e6` (10), `aaddcc729028` (27) —
  executable-readset, dispatch, #909 render-exclusion, source-coverage,
  metadata-generation suites; `219dffae65c9` (30), `2f6fcdc5cdb9`
  (34), `35aa26b70bef` (8) — quantum runtime, boundary readset,
  launch-contract suites. Full logs: `green917d` / `reg917{,b}.pbtest.json`
  under `/home/rob/tmp/muse-pq870-scratch-20260921/`.
- Skips: none. The PB planner test runs (not skipped) against the
  deployed generation receipt.

## Preservation

`git status` shows only the 3 source files plus the new test; no plan,
prepared, calibration, or production-record file changed. Campaign
payloads were never read beyond tiny synthetic fixtures (no 197,990
payload rehash). Before/after: legacy builder output asserted
byte-identical in-test (`test_builder_keeps_legacy_bytes…`).

## Known limits

- Sealed prepared membership is explicit input at seal time (validated,
  not conjured); deriving the roster from live PWC/census geometry
  inside the builder is future work tied to a real prepared pickle.
- Runtime membership comparison runs where the record block travels
  (quantum run); mover-reference validation stays PB's mover path —
  dispatch never consults it as a capability.
- CPU/meta kernel stand-ins in fixtures are explicitly scoped; no GPU
  performance claim.
