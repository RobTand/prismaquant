# Full GLM routed-stack admission — 2026-09-08

**Current metadata admission accepts all 132 groups and all 36,423 units**,
without sampling. The maximum is 105,271,281,496 bytes (98.041521 GiB), rounded
to a 99 GiB request under the declared 104 GiB box budget.

The 124.783 GiB rejection belongs to the **old v1 Hessian export phase**.
Merged PR #379 changed that ownership before main `9ff3b97f7ba569d18e6bcecc69539abad8909591`:
the selected-anchor v2 plan prices a bounded, verified tensor-record file-page
window. No new memory mechanism or statistical approximation is required to
remove that historical planning rejection. Native full-stack execution and fit
are still separate gates.

This audit recomputes current source-header/census arithmetic through PrismaBuild.
It does not run `cmd_plan`, generate a complete capture, read tensor payloads,
start a probe or authorize a native action. The earlier draft is consulted only
for existing resource settings: max activation rows 512, streaming slots 2,
prefetch workers 1, headroom 24 GiB, anchor batch size 1 and box budget 104 GiB.
Its stale container/menu/timeout fields are not an executable specification.

## Current all-group result

The table names the widest complete row in each class. Its phase columns belong
to that row; they are not independent maxima assembled from different rows.

| Group class | Rows | Widest group | Source preparation GiB | Export GiB | Resident anchors GiB | Requested GiB |
|---|---:|---|---:|---:|---:|---:|
| Routed stack | 42 | layer 4 experts | 68.220791 | 84.368271 | 98.041521 | 99 |
| Dense fused | 45 | layer 10 shared gate/up | 54.752041 | 24.915146 | 38.588396 | 55 |
| Dense singleton | 45 | layer 10 shared down | 54.736416 | 24.637802 | 38.115724 | 55 |

The earlier 28.140/30.141 GiB dense examples described layer 0, not the maxima
across all 45 layers. A small shared-expert selection still prepares its
containing routed layer. The routed maximum retains 40.5 GiB H, 5.625 GiB X and
13.5 GiB selected weights, plus the existing source-validation, memo, scratch
and 24 GiB headroom terms. No H alias credit or smaller headroom was introduced.
`resource-summary.json` binds the complete 4 MiB inspection output and records
all group counts, request histograms and maximum-row phase terms.

## Exact execution path

Use the existing dispatcher with `--groups-per-row 1 --rows-per-box 1`, without
`--stack-sample`, once the **original complete** capture and reviewed campaign
specification exist. `cmd_plan` (`tools/dispatch_tessera_campaign.py:465-605`)
creates one row per original anchor group and binds the same canonical capture.
Its `_streamed_resource_plan` calls the existing `selected_anchor_resources`;
`partition_rows_by_fit` only decides which rows are submitted. PrismaBuild owns
placement, concurrency and retry. A fit multiplier of one does not manually
pin a worker or bypass external load admission.

There are 132 original groups: 45 dense fused groups, 45 dense singletons, and
42 routed stacks of 864 units each. Their disjoint union is all 36,423 original
units. The group is the shared rate-placement decision: every member measures
the same group grid, and the group's worst eligible leave-one-out error drives
refinement (`tessera_campaign.py:2529,4595-4675`). Retain this contract.

`--anchor-batch-size 1` controls compatible encoder work and its factor memo
inside the row (`tessera_campaign.py:581-596,4231,4676`). It does **not** split a
stack's decision or release all but one member's resident H/X/source weights.
Larger batches require recomputing their own scratch/memo allowance; this audit
does not choose a throughput batch size. The existing group partition and PB
fanout already express the exact roster; no manually divided experts/files or
second dispatcher is necessary.

After all original rows complete, `merge_payloads` checks every whole group
exactly once and refuses gaps or overlap (`dispatch_tessera_campaign.py:736-750`).
`load_measured_anchor_input` (`tessera_joint_aura.py:105-177`) then requires the
full census unit/group roster, full journal roster, exact measured per-unit
cost/wire cells and `cost_source=tessera_campaign_measured`. The same resulting
candidate union feeds the one joint probe and all six recipe solves. A partial
90-row result never closes these completion gates.

## Why H alias subtraction would be wrong

`declared_shared_capture_groups` (`routed_experts.py:306`) describes checked
**forward accumulator** sharing for original packed gate/up inputs. The capture
collector explicitly creates independent CPU sibling outputs
(`tessera_campaign.py:2094-2130`), and canonical input files remain per unit.
`prefetch_capture` (`tessera_calibration_cache.py:776-858`) validates, loads and
transfers each selected unit's H/X independently, retaining all selected device
tensors before encoding. Equal values or common input origins are not a shared
resident allocation. The v2 sum of per-unit H/X bytes matches this owner.

The separate export-H reference handoff under development changes exported
mapping/merge ownership, not selected source/H/X prefetch. Its author confirmed
that it neither introduces resident aliases nor lowers those retained owners.
This audit does not edit that worktree or subtract its prospective savings.

## Sampling is a different path

`--stack-sample N` persists a probe-weighted expert draw, full-frame inclusion
probabilities, roles and an independent audit subset (`dispatch_tessera_campaign.py:399-459`).
The campaign replays those records from original packed probe statistics and
seed before accepting them (`tessera_campaign.py:2618-2678`). This prevents
different roles/rungs from silently using unrelated draws, but it does not make
a subset an exact census.

The stack price uses Horvitz–Thompson weighting; its reported standard error is
an approximation rather than a generally exact design-unbiased variance
(`tessera_campaign.py:865-912`). The current allocator explicitly does not
consume that error (`:1037-1048`). The audit's extra rung measures interpolation
error within the sampled set, not unsampled experts' exact quality. With a full
census and probability one, sampling error is zero because every expert was
measured; this gives no memory saving.

A sampled run carries packed `tessera_campaign_measured_stack_sample` costs.
It cannot substitute for this task's full per-unit measured candidate union or
pass the joint intake merely by listing complete frame membership. Adopting an
approximation would need explicit review of estimator/quality acceptance and
materialization coverage. Nothing in this audit selects such a policy.

## Provenance and limits

The original v1 estimate is retained as dated history in
`docs/measurements/selected-source-anchors-2026-09-08.md`. The superseding writer
measurement and representative v2 derivation are in
`docs/measurements/bounded-hessian-writer-2026-09-08.md` (PR #379 / issue #377).
That measured a 1 GiB sidecar with exact byte/content-seal parity and lower
file-page ownership; it did not qualify a full GLM stack. The present all-group
metadata derivation adds coverage of the current planner, not a native fit claim.

The canonical complete capture, actual source-preparation/anchor residency,
encoder compatibility and quality/wire gates still bind. Page advice is checked
and guarded but does not guarantee physical reclamation. Runtime refusal and
PrismaBuild's fresh external-load admission remain authoritative. No model,
format menu, budget, production default or native qualification is changed.

## PrismaBuild receipt audit

- Metadata action `b597d4bfad1fd360ef215a4452e4f00d86d7d724803e0df70095271b44fe1546`:
  actual exit 0, all 132/132 admissible, zero declined, complete disjoint 36,423
  unit coverage. CPU 1 / 4 GiB, x86 dl380g10, no GPU, priority -10, native
  OMP/MKL/OpenBLAS threads each one. The inspection output SHA-256 is
  `d66555b4895bd63da5c87eca104d38da34c48903fc10de5bb5018edd6e22b307`.
- Compile/runtime inventory action
  `7f7ca771153ac269681e8447ecab2ccdf4e446517bc0a54f4543e02be20c79c3`:
  actual exit 0; helper compiles, Torch `2.10.0+cpu`, Transformers `5.16.1`,
  CUDA uninitialized. CPU 1 / 2 GiB, same worker class and native thread bounds.
- An initial compile submission was rejected before queue publication because
  the checkout changed during snapshotting. It executed no action; the clean
  resubmission above completed. No test or native result is claimed for it.

`pb-audit.json` records actual log hashes, source-bundle comparisons, terminal
status and native CAS receipt/result verification. Both actions had no live
processes in final telemetry. Full sanitized audits and the original inspection
remain at `/mnt/shared/tessera-measurements/glm-canonical-census-20260908/full-stack-admission-inspection-01/`.
No production code changed, so no new behavioral regression was needed; the
helper ran through the real current planner and its compile check passed.
