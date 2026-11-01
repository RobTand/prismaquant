# Constrained Pareto allocation: serving SLOs as a second selection axis

*Ultraplan P5c. Original policy, now archived:
[`format-speed-policy.md`](../../archive/gridbook_lane_2026-09-02/docs/lanes/nvfp4-cb/format-speed-policy.md)
§1. Motivating audit: gridbook
`docs/audits/ultraplan_perf_2026-08-01.md` §6, item P5c.*

The serving-constraint owner remains active. Gridbook was retired on
2026-09-02. Its measurements and examples below explain the original design;
they do not describe a supported container or a current serving qualification.
Sections 6–8 retain that historical example, harness and work list.

## 1. The problem, and why it needed a second axis

Policy §1 has always specified production selection as

```text
minimize    predicted_quality_loss(a)

subject to  exact_whole_artifact_bytes(a) <= B
            p95_TTFT(a, workload)          <= SLO_prefill
            p95_ITL(a, workload)           <= SLO_decode_itl
            p05_TPS(a, workload)           >= SLO_decode_tps
            resident + KV + peak_scratch   <= device_budget
            backend, shape, TP, fallback and serving-unit coupling are legal
```

and, until this work, implemented only the byte line. The audit states the
consequence in one sentence: *"until P1/P2 land, choosing FP8-CB over vanilla
NVFP4 at ~4.5 bpw buys quality at a measured 1.44× dense-prefill cost, and the
allocator should see that trade rather than discover it at the release gate."*

The 1.44× is not a tie-break. It is the difference between an artifact that
meets a deployment's prefill SLO and one that does not, on a decision the
allocator was making blind.

## 2. What was deliberately NOT done

**No λ.** Latency does not enter the objective — not as a weighted term, not
as a phase-blended `serve_ms`, not as a soft penalty. Policy §1 forbids it and
NATIVE-PARITY forbids it. The reason is not stylistic: a λ makes "1.44× slower"
tradeable against "0.3% worse predicted Δloss" at an exchange rate nobody
measured, and it silently converts a deployment constraint into a preference.
An assignment that misses an SLO is **infeasible**, and infeasible is not a
score.

**No default workload mix.** Policy §1: "no default workload mix hidden in the
allocator". A mix is a claim about what the deployment runs; the allocator does
not get to assume one. With no serving constraints supplied, the axis is inert.
An explicit latency or device-memory constraint requires both the dispatch
table and mix; incomplete context refuses instead of ignoring the limit.
The separate measured-runtime-table path keeps its own admission contract.

**No change to `solve_allocation`.** The bits-DP's semantics for the
unconstrained case are byte-identical and pinned by
`tests/test_allocator_solver_bins.py` plus the end-to-end identity test in
`tests/test_allocator_serve_constraints.py`.

**No composition of incommensurable measurements.** Published serving numbers
are ratios against different denominators: the 27B dense-prefill 1.44× is
against a native compressed-tensors artifact; the fused mid-M 1.04×/1.26×/1.45×
are against FP8-CB's *own* expand+GEMM route. Multiplying the two to get "fused
mid-M vs native" would produce a number that looks measured and is not. The
schema makes it impossible: each `(phase, M-regime)` arena carries exactly one
reference route.

## 3. The three pieces

### 3.1 `prismaquant/serve_dispatch_table.py` — the declarative input

Schema `prismaquant.serve_dispatch_table.v1`. Torch-free, stdlib only.

* An **arena** is one `(phase, m_regime)` cell: the reference route every row
  in it is measured against, the metric (`ttft_ms`, `itl_ms`, `decode_tok_s`,
  `operator_ms`), the reference's absolute value when one is published, the
  `statistic` it is (`p95`, `median_of_repeated_samples`,
  `single_seed_point_measurement`, `ratio_only_no_absolute`), and the `m` it
  was measured at.
* A **row** is one `(format_family, phase, m_regime, lane)` relative cost
  against that arena's reference.
* **Provenance is mandatory on every row and arena**: `source`, `date`, `gpu`,
  `measured_quantity`, `units`, `derivation`. A missing or blank field is a
  load error. `derivation` is the field that keeps this honest — published
  numbers are quoted as speedups, slowdowns, throughputs or wall times, and the
  transform into a slowdown ratio is a modelling step that belongs in the
  artifact next to the number it produced.
* `operator_ms` arenas and arenas with no absolute reference are loaded and
  kept but marked **not SLO-eligible**. They are real evidence (they are the
  only published numbers for some lanes) but policy §5 is binding: "Raw
  standalone kernel timing is never served evidence."

The `lane` axis is what P5b made answerable. `FP8_CB_K36` and `FP8_CB_K37` are
the same format family and the same bpw class; Gridbook 0.7.0 instantiates the
fused mid-M kernel for one and not the other, permanently (K1.2 resolved to a
`k % 4 == 0` format+TMA law). They must not be priced identically, and here
they are not.

### 3.2 `prismaquant/serve_constraints.py` — the aggregation and the verdict

**Aggregation model** (stamped as
`additive_layer_time__param_share_weighted__table_driven_proposal`): for one
phase and one arena,

```text
predicted_phase_ms = reference_ms(arena) * SUM_u  share_u * relative_cost(u)
```

summed across the workload's arenas with the mix weights, where `share_u` is
unit `u`'s share of allocated parameters and `relative_cost(u)` is the row for
`(dispatch family of u's format, phase, regime, u's resolved serving lane)`.

**The eight named assumptions**, all stamped into every artifact:

| | Assumption |
|---|---|
| A1 | **Additivity.** Whole-phase time is the sum of per-unit times: no overlap, no scheduler batching, no CUDA-graph effect, no attention/KV/router or other non-Linear work. |
| A2 | **Parameter-share weighting.** A unit's share of the reference phase time equals its share of allocated parameters. Sound only while the phase is traffic-bound. |
| A3 | **Route locality.** A unit's relative cost depends only on `(family, phase, regime, lane)`. |
| A4 | **Regime uniformity.** One workload mix applies to every unit. |
| A5 | **Baseline transfer.** The arena reference was measured on another model / shape / runtime. |
| A6 | **Resident bytes.** Resident weight bytes are the exact serialized tensor payload; runtime scratch is the operator's `peak_scratch` input. |
| A7 | **Single-stream identity.** `p05_TPS = 1000 / p95_ITL_ms`. Exact for batch-1 single-stream, not served throughput under concurrency. |
| A8 | **Statistic transfer.** Where the SLO is a percentile and the arena reference is a point measurement, the mismatch is recorded as a caveat on the check. |

A1–A8 are why the output is proposal data and why the served protocol remains
the gate. They are listed in the artifact rather than in a docstring nobody
reads at review time.

**Fail-closed rules.** A phase is UNPRICED — and therefore infeasible, never
"passed" — when any unit has no dispatch row, when the arena has no absolute
reference, or when the arena is an isolated-operator microbenchmark. "We could
not price it" is not "it met the SLO".

**Determinism.** Fixed constraint order, sorted iteration, no wall-clock, no
RNG. The binding constraint is the first violated in canonical order when
infeasible, and the satisfied check with the least relative slack when
feasible.

### 3.3 Where it is enforced, and why there

In `allocator.py`'s byte-budget ratchet: `_artifact_for_target` evaluates the
verdict on the same exact expanded assignment it just byte-priced, and `_fits`
becomes `_fits_bytes(cand) and _serve_ok(cand)`.

Three reasons for that point rather than a second DP dimension:

1. **The DP must not change for the unconstrained case.** A second bin axis
   would change `solve_allocation` for every run, constrained or not.
2. **The aggregation needs the object that ships.** It is a parameter-share
   sum over the EXPANDED, promoted assignment — super-item expansion and
   serving-unit promotion included. The DP does not see that object; the exact
   filter does. Certifying the pre-promotion proposal would certify an
   assignment nobody ships.
3. **Honesty of the claim.** The byte axis already stamps
   `additive_candidate_proposal_then_exact_assignment_filter` with
   `global_optimality_claimed: false`. A latency axis inside the DP would
   invite a global-optimality claim the outer exact loop cannot support. The
   assignment-level filter makes the weaker, true claim: every assignment the
   ratchet ACCEPTS is feasible on both axes; the ratchet probes a bounded set
   and does not enumerate the feasible set.

Candidate-level pruning was considered and rejected. A per-unit bound on a
parameter-share-weighted sum is only provable when one unit alone busts the
budget, which cannot happen for a unit whose share is small — i.e. never on a
real model. A prune that fires only on degenerate inputs is cost without
benefit, and it would require the DP to carry a second price.

## 4. Relative-tax reporting

Policy §1's reference rule travels with every verdict and every harness report:

> the reference is the fastest **globally feasible assignment under the same
> whole-artifact byte budget**, memory limits, legality rules, and
> serving-unit coupling, defined **independently for prefill and decode**.
> Summing each unit's independently fastest format is invalid.

The allocator records a `fastest_feasible_reference` per phase — but scoped
honestly, because the ratchet probed a bounded set (grid rungs, the
near-lossless cap, bisection midpoints) rather than enumerating feasible
assignments. The scope note says so in the same object as the number. Anything
narrower than the rule's denominator must declare itself narrower.

## 5. CLI and pipeline surface

Allocator flags (all optional; none supplied ⇒ pre-P5c behaviour):

```
--serve-dispatch-table PATH
--serve-workload-mix   'prefill:dense_prefill_1400=1.0,decode:decode_batch1=1.0'
--slo-prefill-p95-ttft-ms FLOAT
--slo-decode-p95-itl-ms   FLOAT
--slo-decode-p05-tps      FLOAT
--serve-device-budget-bytes INT
--serve-kv-bytes            INT
--serve-peak-scratch-bytes  INT
```

`run-pipeline.sh` exposes them as `SERVE_DISPATCH_TABLE`,
`SERVE_WORKLOAD_MIX`, `SLO_PREFILL_P95_TTFT_MS`, `SLO_DECODE_P95_ITL_MS`,
`SLO_DECODE_P05_TPS`, `SERVE_DEVICE_BUDGET_BYTES`, `SERVE_KV_BYTES`,
`SERVE_PEAK_SCRATCH_BYTES`, all recorded in `STAGE_SETTINGS_ENV`. Setting a
latency SLO without a dispatch table is refused by name.

## 6. The archived example table

`archive/gridbook_lane_2026-09-02/prismaquant/serve_dispatch_tables/gridbook_gb10_2026-08-01.example.json`.
**Example / proposal data only**, populated exclusively from measurements
already published in the Gridbook repository, each row citing its source:

| family | phase | arena | lane | cost | source |
|---|---|---|---|---:|---|
| NVFP4 | prefill | `dense_prefill_1400` | native | 1.000 | BENCHMARKS 27B speed table (0.746 s TTFT(1400), the reference arm) |
| FP8_E4M3 | prefill | `dense_prefill_1400` | native | 1.000 | same row — the published "native" artifact is mixed NVFP4/FP8 |
| **FP8_CB** | prefill | `dense_prefill_1400` | fallback | **1.44** | BENCHMARKS 27B: 1.075 s vs 0.746 s — the audit's headline tax |
| BF16 | prefill | `dense_prefill_1400` | native | 1.701 | BENCHMARKS 27B: 1.269 s / 0.746 s |
| NVFP4 | decode | `decode_batch1` | native | 1.000 | BENCHMARKS 27B: 10.26 tok/s, the reference arm |
| FP8_E4M3 | decode | `decode_batch1` | native | 1.000 | same row |
| FP8_CB | decode | `decode_batch1` | fallback | 0.999 | BENCHMARKS 27B: 10.26 / 10.27, the slowest end of the published 10.27–10.30 band |
| BF16 | decode | `decode_batch1` | native | 2.235 | BENCHMARKS 27B: 10.26 / 4.59 |
| FP8_CB | prefill | `dense_mid_m_{32,64,128}` | fallback | 1.000 | KERNELS — the transient expand+GEMM denominator |
| FP8_CB | prefill | `dense_mid_m_{32,64,128}` | fused_mid_m | 0.9615 / 0.7937 / 0.6897 | KERNELS: reciprocals of the published 1.04×/1.26×/1.45× |
| FP8_CB | prefill | `dense_large_m_1400_gemm` | fallback / fused_mid_m | 1.000 / 4.545 | KERNELS: reciprocal of ≈0.22× — the structural ceiling |
| NVFP4_CB | prefill | `moe_grouped_w13_bf16` | fallback | 1.2104 | BENCHMARKS 2026-08-01 DSV4 microbenchmark: 6.471 / 5.346 ms warm (the published 0.826×, inverted) |
| NVFP4_CB | prefill | `moe_grouped_w2_bf16` | fallback | 1.0868 | same section, w2: 2.792 / 2.569 ms warm (0.920×, inverted) |

Two absences are deliberate and stated in the table's own notes:

* **NVFP4_CB has no whole-model row.** No published measurement gives the
  default fp4-CB BF16-bridge quality path a whole-artifact TTFT or decode
  number. Its only published timings are the two MoE grouped-GEMM
  microbenchmarks, which are `operator_ms` and therefore never SLO-eligible.
  Consequence: an assignment containing NVFP4_CB cannot be certified against a
  prefill or decode SLO from this table, and the evaluator refuses instead of
  interpolating. That is the honest state of the record.
* **MoE expand ≈35% of MoE layer time at Laguna scale** is a *share*, not a
  ratio against a named reference route, so it is a note rather than a row.

## 7. P5d: the archived D0.3 exact-rate harness

`prismaquant/d03_exact_rate.py` and `scripts/run_d03_exact_rate.sh`, both under
`archive/gridbook_lane_2026-09-02/`, ran the two experiments the Gridbook
ROADMAP D0.3 named, with the P5a pricing and these
stamps: (i) `FP8_CB_K36` vs vanilla `NVFP4` on dense units at matched exact
whole-artifact bytes; (ii) below 4.5 bpw, byte-neutral sweeps whose NVFP4
promotions are funded by cheaper CB rungs elsewhere.

It **prepares** release-gate evidence; it does not constitute it. Its two
refusals are the point:

* **No cross-family verdict** when P5a's band check failed — suppressing that
  verdict is the check's entire purpose, and printing it "with a caveat" would
  defeat it.
* **No quality verdict** when the two arms miss the ≤0.1% whole-artifact
  byte-match target policy §5 already names (the threshold the published 0.6B
  endpoint pair missed at +0.154%).

Packed-expert vanilla NVFP4 is **excluded** from the contest and the exclusion
is recorded in every report: the producer profile denies stock NVFP4/FP8 on
packed expert stacks because no stock-compressed-tensors packed-expert emit
path exists, and building one is out of scope under the one-payload /
no-new-packer rule. Unlocking it is gridbook **D0.2**.

## 8. Historical work list at the time of this design

The retired Gridbook-specific items are retained as history. They are not a
request to restore that lane. Current supported lanes require their own
measured dispatch inputs and serving qualification.

1. **Real measured tables.** The shipped example is a worked demonstration on
   published 27B/DSV4 numbers. A deployment needs its own table on its own
   hardware, model and workload — A5 (baseline transfer) is otherwise doing all
   the work.
2. **The NVFP4_CB whole-model hole.** Until the fp4-CB quality path has a
   published whole-artifact TTFT/decode measurement, the constraint axis cannot
   certify any assignment containing it. Closing that hole is a measurement
   task, not a modelling one.
3. **Policy §7 item 4**, now the gating question: do per-layer timing tables
   predict end-to-end ranks? The constraint axis is only as predictive as the
   answer.
4. **The served protocol.** Nothing here promotes anything.
