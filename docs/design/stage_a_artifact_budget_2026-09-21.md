# Stage A durable artifact budget declaration + preflight (2026-09-21)

Issue #882. Branch `fix/stagea-artifact-budget-20260921`.

## Finding

Stage A (`prismaquant.joint_adjoint_capture` -> `joint_cost_stage_a`
`execute` / `run_adjoint_capture_core`) retains, for the layer quanta:

- 46 forward boundary groups written (input boundary-0 + transformer
  1..45); the tail retires the FINAL boundary-45 per batch
  (`joint_cost_stage_a.py:425`), so **45 INPUT groups 0..44 stay live**
  on origin through the reverse and after the run (doc `:280`, receipt
  `:526`).
- One live cotangent plane set: `n_probes` probe planes x batches entries
  (four probe planes of 512 entries each on the 4-probe production panel,
  2048 entries total) -- roll replaces the previous via
  `storage.write(previous=...)` (`:491`).
- One strided checkpoint copy of that plane set per boundary in
  `derive_checkpoint_boundaries(num_layers, stride)` (45 layers, S=8 ->
  `{45,40,32,24,16,8}` = 6), each with shared state + manifest, retained
  for quanta and never read by the same Stage A.

Production panel
(`complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02.plan.json`,
sha `0b2cc0066bb612e32af6d0c8c809912d325b2975583297eedeee97851ee545da`):
`n_probes=4`, `probe_microbatch=1`, `seed_base=7000`; calibration 512 rows
x 512 ids (`calibration_tokens.safetensors` shape `[512,512]`); model 45
layers / hidden 4096 / `hc_mult` 4 / BF16. Per-boundary tensor raw
`1*512*4*4096*2 = 16777216` B (16 MiB); writer ceiling `nbytes + 65536`;
observed generation `cfe17deb6d18452da2d4e90545af49e3` 512 files x
16779369 B = 8591036928 B (~8.001 GiB) per boundary group
(`written_tensor_bytes` 8589934592, 2153 B mean header).

Hard lower bound at the first (tail) checkpoint, before headers / shared /
manifest: `45*8 + 4*8 + 4*8 = 424 GiB`. Six-checkpoint peak before overhead:
`360 + 32 + 192 = 584 GiB`. The plan declares `446676598784 B = 416 GiB`.
It cannot fit. The existing reserve-before-write guard
(`cost_streaming.StreamedBoundaryArtifacts`) is correct and unchanged; the
defect is the missing invocation budget derivation, not the guard. This is
a proactive source/geometry finding (current failure `8ca` was the earlier
strict readback), not an observed full-run failure.

Calibration note: the plan's `calib_seqlen` is 512; the 16 MiB tensor is
`512 tokens x 4 streams x 4096 x 2 B`, not 2048 tokens. Prior prose naming
2048 tokens is corrected here; the code (`model_profiles/glm5_next.py`
`expand_hidden_for_layers`, `cost_streaming.py:_prepare`) is authoritative.

## Seam (durable-side #809, analogous to prefetch override #819)

- `joint_cost_stage_a --artifact-budget-bytes BYTES` /
  `PRISMAQUANT_STAGE_A_ARTIFACT_BUDGET_BYTES=BYTES`: strict positive-integer
  bytes (bools, floats, non-decimals refuse with units named). Two explicit
  sources that disagree refuse. Absent: the plan's sealed
  `boundary_storage.max_artifact_bytes` threads verbatim. No default is
  raised and nothing silent happens.
- Resolved by `resolve_artifact_budget_override` beside the sealed plan;
  the deviation stamp (`artifact_budget.v1`: source, unit bytes,
  `plan_sealed_bytes`, `run_used_bytes`) rides `results.json`,
  `counters.json` and the adjoint receipt's `artifact_budget_override`.
  The sealed plan is unchanged; the run's storage policy uses `run_used`.
- Dispatcher `tools/dispatch_joint_quanta.py --stage-a-artifact-budget-bytes`
  threads the payload's `--artifact-budget-bytes` (the PB channel; the
  container forwards no ambient env) and records it in campaign state.
  Absent: argv unchanged.

## Preflight (early, before any expensive forward)

`estimate_stage_a_artifact_demand` bounds demand from geometry alone and
allocates nothing: retained groups `num_layers`, live plane set
`n_probes` probe planes x batches, checkpoint copies per
`derive_checkpoint_boundaries`. Batches are counted exactly -- full
batches plus the remainder batch at its own true smaller size (both
derived through the profile's own expansion on `meta` tensors): a
`ceil`-of-full-size count would be an UPPER estimate, not a floor.

It returns two numbers with different standing. `lower_bound_bytes` is
the HARD floor of true raw tensor bytes (full groups + true remainder,
before headers / shared / manifest) -- the mandatory early-refusal
threshold. `planning_estimate_bytes` is an explicitly named planning
allowance, NOT a proven upper bound: per-file header envelope (`nbytes +
65536`, the writer's admitted envelope) plus the caller-supplied
per-checkpoint shared and manifest allowances. The shared allowance
reuses the plan's `max_auxiliary_bytes` as a stated assumption --
deduplicated live storages can serialize into separate retained files, so
it does not prove a ceiling -- and the manifest rides the existing
`_checkpoint_manifest_envelope_bytes` estimator on the real synthetic
file plan. An underivable geometry or manifest estimate refuses named
rather than guessing (no silent fallback size).

`preflight_stage_a_artifact_budget` refuses a below-floor ceiling with
the concrete floor / planning / declared values and the named remedy
(`--artifact-budget-bytes` / env), retaining the plan sealed value as
provenance. `run_adjoint_capture` runs it via `_run_artifact_preflight`
from live geometry (embedding width off the live model, stream expansion
through the runner's own profile on `meta` tensors -- full and remainder
each at true size) after the runner + calibration load and before the
core. The runtime reserve / write / commit guards stay authoritative for
serialized bytes and unpredicted overhead -- a ceiling between the floor
and the planning allowance still passes the preflight and is admitted or
refused at write time, exactly as the scaled regression demonstrates.

## Proposed invocation budget

File-envelope peak `37376 * 16842752 + 16842752 = 629531541504 B
(~586.3 GiB)`; raw floor `37376 * 16777216 = 627065225216 B (584 GiB
exact)`. Planning allowances: shared cap `6 * 2147483648 =
12884901888 B (12 GiB, stated assumption from the plan auxiliary bound --
not a proven ceiling)` + manifests (`6 * 4194304`, estimator-bounded;
the preflight refuses named if the estimator cannot bound this
invocation's geometry instead of guessing) + temp put the planning
estimate at `642441609216 B (~598.3 GiB)`. **Proposed run budget
`687194767360 B = 640 GiB`** gives ~44 GiB (~7%) headroom over the
planning estimate for unpredicted shared/manifest growth. Dataset
`storage_pool/shared` as mounted at `/mnt/shared`: `27593443115008 B`
available (`df -B1`, PB `abe1e66988ef`, sparky 2026-09-21 -- the dataset
mount view, not raw pool free; no `zfs` CLI on fleet workers, PB
`88b08ae5c382`, so a fileserver-side `zfs get available` read remains an
operator-side confirmation, not a number this proposal depends on);
640 GiB is ~2.5% of that available. The PQ adapter declares the
origin-class maximum from this same invocation budget (root integrates);
the sealed 416 GiB plan stays as provenance.

Exact dispatcher / CLI change in the artifact:

```
tools/dispatch_joint_quanta.py --stage-a-artifact-budget-bytes 687194767360
  -> payload: python3 -m prismaquant.joint_adjoint_capture ... --artifact-budget-bytes 687194767360
```

Do not edit the live sealed plan, prepared, read manifest or calibration;
do not launch GPU from this branch.

## Gates

`tests/test_stage_a_artifact_budget.py`: resolve strictness + provenance,
production retention math (416 < 584 floor, 640 covers the planning
estimate), remainder-at-true-size on the actual tiny fixture (below the
naive ceil), early floor refusal naming floor/planning/declared/remedy
with geometry unchanged, scaled three-band exhibit (below-floor early
refusal with no files, floor-passing late guard refusal, adequate
override proceeds) on the real `write_adjoint_checkpoint` path, core
stamp + ceiling threading, CLI + dispatcher threading. Existing
checkpoint accounting (`test_checkpoint_artifact_budget.py`) and Stage A
contract suites run unchanged.
