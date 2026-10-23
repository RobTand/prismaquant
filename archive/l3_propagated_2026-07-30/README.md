# L3 propagated end-KL + the cost cascade — archived 2026-07-30

**Kill order.** Architecture re-vet 2026-07-30, proposal **R4** (Lens 6 P1),
accepted by Robert as part of the wave-2 mandate: *"implement ALL proposals."*
This is the wall the spine rewrite points at: `CLAUDE.md` §3,
`docs/ARCHITECTURE.md` §2.2 / §4.4 / §11 no longer describe PrismaQuant as a
three-level cascade.

## What was retired

Not a lever — a **framing**. PrismaQuant was described, top-billed in every
document a new reader met first, as a three-level cost cascade:

- **L1** — additive diagonal-Fisher `½·H_trace·MSE`, solved by multi-choice
  knapsack DP.
- **L2** — the perturbed-X fixed point: re-measure each Linear's output MSE
  under the activation distribution the *current* assignment induces, re-solve,
  iterate to convergence.
- **L3** — propagated end-KL for a bounded neighborhood of uncertain Linears,
  measured paired against a frozen L2 baseline.

**Neither L2 nor L3 was executed by the shipping pipeline.** No `COST_MODE`
ran the L2 fixed point (`perturbed_x_cache.py` survives, and stays live, purely
as an activation/model-loading utility). L3 was opt-in and OFF —
`ALLOC_PROPAGATED_SENSITIVITY_REPORT` defaulted empty, and
`kl_sensitivity_probe` had **zero** references in `run-pipeline.sh`. Its
downstream consumer (coordinate-descent / DP polish) was already archived under
`archive/polish_2026-05-15/`, and D19 walled ten dead launcher targets that
drove it.

## Durable lesson

**Cross-layer interaction turned out to be a quantity with nothing in it to
model — and the way to find that out was to measure it, not to build a level
for it.**

Three independent measurements retired the cascade:

1. **The cascade head-to-head** (`aura_cascade_headtohead`): the L2 fixed point
   beat additive L1 by **−1.5%**. Over the same baseline, the AURA KL-adjoint
   cost — a *better single cost*, not another level — beat L1 by **−38.5%**.
   Cross-layer modelling bought roughly nothing; a faithful unary cost bought
   25× more.
2. **The cross-layer sensitivity sweep** (`xlayer_sensitivity_2026_06_09`): the
   fp32 pairwise residual is **+5–12% and diffuse**, with **3 of 1180 pairs**
   significant. There is no sparse pairwise structure to capture. The companion
   finding (`cross_layer_additivity_fp32`) is sharper still: per-Linear KLs
   *do* add in fp32 — the observed non-additivity was a **bf16 artifact**. That
   is also why CLADO/QUBO never paid.
3. **L3-polish-of-many non-additivity** (§11, `prismaclade_l3_non_additivity`):
   per-Linear L3 costs measured under an L2 context do not sum to true end-KL
   when many flip at once — so the one thing L3's expensive measurement was for
   could not be composed anyway. Measured one-at-a-time coordinate descent was
   the safe alternative, and it lives (archived separately) as polish.

The **method** survives intact and unchanged: *surrogates generate, real KL
selects.* What died is the claim that the surrogate needs three levels. It
needs **one faithful cost** (AURA: KL-Fisher adjoint × production-rendered dW,
plus empirical unit-KL for packed experts, which the smooth cost is
route-flip-blind for) and **real held-out KL** to select among the candidates it
proposes. Levels 2 and 3 were an answer to a question the measurements closed.

**The meta-lesson is about docs, not code.** ~15k lines and the *first section
every reader saw* described machinery with no executor, no consumer, and a null
result. An architecture document that leads with an unexecuted abstraction
teaches every new contributor the wrong system. Retiring the framing was the
larger part of R4; the `git mv` was the smaller.

## Contents

**Modules** (`prismaquant/`)
- `kl_sensitivity_probe.py` (3,678 L) — the L3 driver: production-cache
  integration, L3 neighborhood construction, and the frontier solver, all in
  one CLI.
- `propagated_sensitivity_costs.py` (293 L) —
  `apply_propagated_sensitivity_penalty`: folds a measured propagated-KL report
  into the allocator's cost table, with the three format-extrapolation rules
  (`local_mse_ratio`, `current_only`, `bits_interp`). *`current_only` is itself
  a graveyard entry — it won the hook screen and lost full-vocab KL.*
- `sensitivity_response.py` (416 L) — the response-curve report builder.
- `kl_measurement_l3.py` (4,629 L) — **the L3 half of `kl_measurement.py`**,
  split out here: 97 top-level symbols including `select_l3_neighborhood`,
  `build_global_l3_neighborhood`, `solve_frozen_l3_neighborhood`,
  `build_l3_candidates`, `measure_propagated_costs`,
  `measure_lane_batched_kl_deltas`, `measure_override_set_kl`,
  `measure_override_paired_kl_deltas`, `QuantWeightCache` +
  `build_quant_weight_cache`, `tail_forward_from_layer`, the `_TailCudaGraph*`
  cache, `_DepthGroupTargetHooks` / `_OverrideSetTargetHooks` / `_LaneOutputMSE`,
  and every `_l3_*` memory-budget helper. **It imports and runs against the
  live tree** — the 22 helpers shared with the live KL path
  (`CUDAGraphRegistry`, `_replay_lane_kl_totals`, `_clone_static_tree`,
  `assignment_bit_total`, `l2_cost_value`, …) are re-imported from
  `prismaquant.kl_measurement`, which still exports them. Verified importable
  at wall time. That stopped holding in the 2026-10 refactor; see
  *Resurrection status (2026-10-16)* below.

**Tools** (`tools/`) — `sensitivity_propagated_group_report.py`,
`build_propagated_sensitivity_cost_sweep.py`,
`summarize_kl_sensitivity_response.py`.

**Tests** (`tests/`) — `test_kl_sensitivity_probe.py`,
`test_propagated_sensitivity_costs.py`, `test_sensitivity_response.py`,
`test_kl_measurement_override_cache.py`. Records; not wired into the live
suite.

**Excised surfaces, verbatim** — `allocator_propagated_surface.py`: the five
`--propagated-sensitivity-*` CLI arguments, the cost-folding body, and the two
`propagated_sensitivity_costs` metadata keys removed from
`prismaquant/allocator.py`.

**Launcher** — `examples/launchers/proto-production-prismascout.sh`. The
2026-07-30 launcher wall deliberately left it live because it still invoked
`kl_sensitivity_probe` alongside removed modules (`iterate_block_clado`,
`prismascout`). R4 is that per-stage decision: with the probe walled, every
non-trivial stage it drives is gone.

## Note: the D5 damp-provenance fix rode along

`kl_sensitivity_probe._normalized_production_cache_levers` was rewritten
earlier on 2026-07-30 to delegate to
`production_weight_cache._resolve_production_render_levers`, closing D5 (two
readers of `PRISMAQUANT_GPTQ_DAMP_SWEEP` with opposite defaults). That fix
comes to the wall with its file — it is historical now, since the forked copy
it repaired no longer exists in the live tree. **The contract it pinned is
still tested**: `tests/test_production_weight_cache.py` carries the delegation
as a local shim and keeps every assertion (sweep OFF by default since
2026-06-12; sweep-off renders must record their fixed damp).

## Live-tree state after the wall

- `run-pipeline.sh`: `ALLOC_PROPAGATED_SENSITIVITY_REPORT` non-empty now
  **`exit 2`**s pointing here; the four companion `ALLOC_PROPAGATED_*`
  defaults and the allocator pass-through are gone.
- `prismaquant/allocator.py`: no `propagated` references remain.
- `prismaquant/kl_measurement.py`: **1,206 lines**, down from 5,731. It keeps
  whole-assignment KL (`measure_assignment_kl`), the per-sequence tail
  machinery added the same day (`sequence_token_nll`,
  `summarize_per_sequence_kl`, `return_per_sequence`), `assignment_bit_total` /
  `assignment_hash`, `l2_cost_value`, and the `CUDAGraphRegistry`.
  `validate_assignments_kl` and `validation_harness` are unchanged.
- `tools/smoke_graph_memory.py` stays live with its L3 phase removed — it still
  covers the two graph-capturing paths the shipping pipeline runs (cost pass,
  whole-assignment KL).
- `tests/test_format_menu_expansion.py` keeps the format-menu expansion rule as
  a local helper (the rule is still what `run-pipeline.sh` computes for
  `CACHE_FORMATS`), no longer importing it from the probe.

## To resurrect

Move `prismaquant/*.py` back to `prismaquant/` (renaming
`kl_measurement_l3.py` or importing it as-is), make the import edits and
restores listed under *Resurrection status (2026-10-16)*, restore the
allocator surface from `allocator_propagated_surface.py`, restore the five
`ALLOC_PROPAGATED_*` defaults and the pass-through in `run-pipeline.sh`, and
delete the `exit 2` gate. **Before doing so, read §11**: the three measurements
above are the reason it left, and reviving it wants a *new* measurement, not a
new argument.

## Resurrection status (2026-10-16)

The statement above that `kl_measurement_l3.py` "needs no edits" was true at
wall time and at the refactor base `138069b799`. It is false after the 2026-10
refactor. A static AST check of every `prismaquant` import in this wall, run
at the commit that added this section, found these gaps.

Introduced by the 2026-10 refactor:

- `prismaquant.layer_state_cache` (`prismaquant/kl_measurement_l3.py:57`,
  `prismaquant/kl_sensitivity_probe.py:31`) and `prismaquant.weight_session`
  (`prismaquant/kl_sensitivity_probe.py:2764`, weight-session mode only) were
  deleted on 2026-10-14. Restore both from
  `archive/state_prototypes_2026-10-14/`.
- `prismaquant.allocator_candidates` no longer defines
  `_stats_indicates_packed_expert`, `cost_entry_predicted_dloss`,
  `cost_entry_is_bit_exact`, `cost_entry_uses_measured_output_mse` or
  `cost_entry_source`. They are in `prismaquant/priced_cell.py`. Edit the
  imports in `prismaquant/kl_measurement_l3.py:46`,
  `prismaquant/propagated_sensitivity_costs.py:9`,
  `prismaquant/sensitivity_response.py:24` and
  `tests/test_propagated_sensitivity_costs.py:27`.
- `prismaquant.sensitivity_probe.load_calibration` is now
  `prismaquant.calibration_data.load_calibration`. Edit
  `prismaquant/kl_sensitivity_probe.py:39` and
  `tools/sensitivity_propagated_group_report.py:25`.

Present since this wall was built (`08612a4a46`):

- `prismaquant/kl_sensitivity_probe.py:40`,
  `tests/test_kl_measurement_override_cache.py:6`,
  `tests/test_kl_sensitivity_probe.py:33` and
  `tools/sensitivity_propagated_group_report.py:20` import L3 names from
  `prismaquant.kl_measurement`. Those names are defined in this wall's
  `kl_measurement_l3.py`, which is what the rename note above refers to.
- `prismaquant/propagated_sensitivity_costs.py:14` and
  `tools/sensitivity_propagated_group_report.py:23` import
  `prismaquant.mse_promotion`, archived separately at
  `archive/mse_promotion_2026-07-30/`.

All other `prismaquant` imports here resolve in the live tree. Resolution is
not behavior: the moved pricing functions are the refactor's rewrite, so a
resurrection still needs its own tests and measurements. The static check and
its output are recorded with Claude review finding R3B-04.
