# First measured prefill x accuracy curve — Qwen3-0.6B, Tessera layer 0

Status: **measurement record.** 2026-09-13, branch
`claude/first-prefill-frontier-qwen3-0.6b`, PQ #237.

**Amended 2026-09-13, branch `claude/price-the-fp4-route-class`, PQ #559.** The
curve in §3 measured two of the three route classes the pinned contract
publishes on sm_121. §8 adds the third — `TESSERA_NVFP4` fp4 x fp4, format
`TESSERA_E2M1_K2_R896` — and reports what it priced, what it refused, and what
§3.3 has to be narrowed to now that a third class exists. §1 through §7 are
unchanged except where a sentence is marked as amended.

This is the first time a `prismaquant.measured_runtime_prices.v2` table has been
built from real Tessera native operator receipts, the first time
`admit_native_rows` has admitted rows that came off a GPU rather than out of a
test fixture, and the first prefill-price-versus-accuracy curve over real served
PrismaQuant artifacts.

It is also a record of what the same evidence **refuses**. The table does not
reach the allocator. Three independent causes are named in §4, each with the
exact refusal string and the code line that emits it. Nothing in §3 depends on
§4 being fixed, and nothing in §4 is worked around.

Receipts root (`$R` throughout):

```
/mnt/shared/tessera-runs/receipts/frontier-qwen3-0.6b-20260913
```

---

## 1. Scope — what this measures, and what it does not

**Measured.** The CUDA-event time of one `(unit, format)` operator apply, for
the 7 layer-0 Linears of Qwen3-0.6B, at every format the seven served artifacts
assign them: 49 cells.

**Amended (PQ #559) — which route classes these 49 cells cover.** Two, not
three. 42 cells are `TESSERA_E4M3_K1_R*`, which route to `torch._scaled_mm` with
activation contract `fp8_per_token_dynamic`; 7 cells are `TESSERA_BF16_K1_R1792`,
which route to `torch.mm`. The third class the pinned contract publishes on
sm_121 — `TESSERA_NVFP4`, fp4 x fp4 through the same `torch._scaled_mm` symbol
with activation contract `e2m1_group16_ue4m3_static` — is **not** in these 49
cells. It is measured in §8. Prefill is a `[512, 1024] x [1024, N]` apply; decode is
the same operator at one token. 32 samples per phase per cell, 8 warmup
iterations, `timing_scope = cuda_events_after_resource_collector_stop`. The
table row's price is `statistics.median(samples_ms)`
(`native_receipt_table.py:186-187`), so an artifact's price is a **sum of
medians** over its seven units.

**Environment, and the scope every number below inherits.** sparklina
(`GPU-b1eceeea-fec7-371e-2cf3-cd10f2e7b705`), sm_121, TP1, batch 1,
`graph_mode: eager`, 512 prompt tokens, `TESSERA_SERVE_MODE=resident`, image
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`.
A different batch size, sequence length, graph mode or serve mode is a
different measurement; none of the findings below are claimed outside this one.

**Not measured, and not estimated.** The whole-model fixed charge.
`admit_fixed_resources` refuses every v2 table today (§4), so the curve carries
**no** fixed term rather than declaring one zero. Every price in §3 is an
operator sum over the seven priced units alone and says so in its own document
(`curve/prefill-accuracy-qwen3-0.6b.json`, field `scope`).

**Stated scope on the priced bytes.** The rendered bytes these cells timed are
*not* the served artifacts' layer-0 wires — those were rendered by the campaign
under its own calibration. These were rendered fresh, same shape, same rung,
under the calibration in `$R/inputs/`. Timing does not read weight values, so a
same-shape same-rung operator prices the artifact's operator; the rate axis,
which is what changes between arms, is carried in the format name and therefore
in the wire. This caveat is stated in
`experiments/pq_frontier_native_cells.py`'s own docstring and is repeated here
rather than left in the source.

**The accuracy axis is not measured here.** It is a served whole-artifact
KL-vs-BF16 read from `accuracy/accuracy.md` and copied through with its metric
identity attached (§3.1). It is not a per-unit quantity and is not re-derived
by any tool in this record.

---

## 2. Measured — the evidence chain

### 2.1 Production runs

| Stage | PB key | Host | Cells | Wall | Outcome |
|---|---|---|---|---|---|
| calibration draw | `c74e127b2dd1` | sparky | — | — | `inputs/calibration-qwen-train-n4-s512.safetensors` |
| G1 prepare (full2) | `980009bbf2c5` | sparky | 49 | 120.8 s | render + joint AURA + native inputs + freeze |
| G2 timing r1 | `61ad724cf763` | sparklina | 2 | — | 2/2 `timing_admissible` |
| G2 timing r2 | `87ad54bac591` | sparklina | 24 | 772 s | 24/24 `timing_admissible` |
| G2 timing r3 | `565eb1e1b7ef` | sparklina | 23 | 711 s | 23/23 `timing_admissible` |

All GPU work went through PrismaBuild. Every G2 action is on sparklina — the
relation requires one GPU identity across every run it binds — and every action
is well under the 30-minute ceiling. r2/r3 `checkout_snapshot.commit`:
`84f5e4adbac6a7e3c8b3a0d764a58cf085a8b456` and
`8e9a73d897ac9e60c986530abfd146e14c1983cb`.

**Per-cell status was read from the 49 durable `receipt.json` files, not from
the PB logs.** PB truncates an action's stdout at roughly 10 KB, so only 4 of
r2's 24 cells appear in its log; counting `timing_admissible` in the log would
have reported 8 of 49 and been wrong.

### 2.2 The runtime identity holds across all three actions

All 49 receipts carry one byte-identical native runtime record, identity
`3a09e05d9266acd7a8e48ca7809c1b5c240f4b7426044060ac101dca0ca78fed`.
`native_receipt_table.derive_context` requires exactly that, which is why 49
cells measured in three separate containers can sit in one table.

Post-core audit on every cell: `stock_files_unchanged = 4956`,
`native_returncode = 0`.

### 2.3 Triton JIT cache — zero `.so` generated

Each G2 action seeded `TRITON_CACHE_DIR` from the #399 engine run's own cache.
The runs added generated `_unpack_body_kernel` artifacts
(`.json/.cubin/.llir/.ptx/.source/.ttgir/.ttir`): r1 8, r2 24, r3 24. **No label
added a `.so`.** Only `.so` files enter `native_libraries`, so no native
production dependency exists that the full-engine run never loaded — the #323
failure mode, checked rather than assumed.

One honesty note: the r1 launcher crashed at its own cache-verification line
(`g2_launch.py:129`, a `list | set` TypeError, fixed in `fb7afc4`) *before*
writing `triton-cache-verification.json`. The r1 digests were recomputed
afterwards from the durable `/out` cache and the engine cache the run was seeded
from, and the file recording them is deliberately named
`native-r1/triton-cache-verification.recomputed.json` and carries a
`recomputed_after_the_fact` field saying so. r2 and r3 wrote theirs in-run.

### 2.4 The relation

`$R/relation/relation-all.json` · sha256
`40d2a7b5051c26b7b98fe006bacbc53aa9f125f10340b1a0d52b8e8f37e4d890`

* 50 runs: the #399 full-engine run `engine-a5` plus one per measured cell (one
  run entry per cell, not per container — loaded-package identity and the
  post-run core audit are facts about the process that wrote that receipt).
* 8 869 `production_dependencies` (49 x 181), every one bound to full-engine
  bytes.
* 40 `full_engine_extra_libraries`.
* **0 unbound native libraries.**
* `load_runtime_relation` verdict: **loaded**.

Nothing in the tree wrote a relation document before this one; the loader only
ever read them, which is why the first real table had no input to be judged on.

### 2.5 The table, and its reproducibility

`$R/table/qwen3-0.6b-layer0-all.json` · sha256
`9bbb283a2ea4dae23cc1f18dd688ec7dd86ce54877fd82359d6123c12b27d3c9`
· 49 rows, 7 units, `table_id` `qwen3-0.6b-layer0-native-all`.

Re-running the emitter from the same receipts, relation, cost payload and
`--now` reproduced the table **byte for byte** (same sha256). Command and exit
record: `$R/stage/control/` + `$R/stage/evidence/emitter-rerun/`. The emitter's
exit code is **2** — it emits the table and reports the fixed-resource refusal
rather than pretending admission.

It was re-emitted a **third** time after the branch's own regression fix
(`1e8ed594`, which moved the full-engine report's recomputation out of the
emitter and into `runtime_provenance.recompute_fixed_resources`, restoring
`admit_fixed_resources` as that report's only reader). Same three digests:
table `9bbb283a…`, fixed-resource receipt `58f0d3d7…`, and an emission report
matching field for field. `$R/stage/evidence/emitter-postfix/`. The two
emission reports differ in exactly one field, `table_path`, which is the `--out`
they were each written to. So no number in this document depends on which side
of that fix produced it.

### 2.6 Admission gates, reported separately

`admit_runtime_provenance` raises on the first gate that refuses, so a single
verdict cannot say whether `admit_native_rows` admitted. The three gates were
called in order and each reported
(`$R/stage/evidence/admit-all.json`, sha256
`cfed505aa6f36ff411728d214577bc6df8cc91e03338552c605e0d9b9b2f9857`):

| Gate | Verdict | Detail |
|---|---|---|
| `load_runtime_relation` | **loaded** | 50 runs |
| `admit_native_rows` | **ADMITTED** | **49 rows, refusal `None`** |
| `admit_fixed_resources` | refused | 143 distinct refusal shapes, 21 468 parts |

The middle row is deliverable 3: the first admission of rows derived from real
Tessera receipts. (A synthetic-fixture admission already existed in
`tests/test_runtime_provenance.py`; what is new is that these 49 rows came off
a GPU.)

### 2.7 Parity diagnostics, read before the curve

Two checks on the render that produced the priced bytes. They belong here, not
in a footnote.

`$R/full2/prep/streamed-body-parity.json` — **gate**:

```
policy                  exact per-Linear source-weight equality,
                        resident renderer vs streamed producer
units_compared          196      units_declared 197
mismatched_units        []
outside_streamed_body   ["lm_head"]
priced_units_outside_body []
```

`$R/full2/prep/teacher-parity.json` — **diagnostic, explicitly not a gate**:

```
bit_exact                       false
max_absolute_logit_difference   0.626953125   (max_reference_logit_magnitude 39.25)
mean_absolute_logit_difference  0.005009195301681757
top1_agreement                  0.998046875   over 2048 positions
scope   "DIAGNOSTIC, not a gate. A resident whole-model forward and a streamed
         layer-major forward select different matmul kernels, so their BF16
         reduction orders differ and their logits cannot be bit-equal. The
         refusal is the exact per-Linear weight equality in streamed_body_parity."
```

The reader should take the first as the check that binds and the second as a
recorded non-bit-exactness with its cause named. Neither was relaxed.

---

## 3. Measured — the curve

`$R/curve/prefill-accuracy-qwen3-0.6b.json` · sha256
`92b28f9e1127dcb777ef1d46aa89f7c08ad0f01cf7f2f61085711f550a6da21d`
· schema `prismaquant.prefill_accuracy_curve.v1` · 7 points, 0 refused artifacts.

**Who produced this, stated before the numbers.** Not `prismaquant.prefill_frontier`.
The shipping consumer was run against this table, unmodified, and **refused**
(§4) — it never reached a DP, so it proposed no allocation and emitted no
frontier. The curve below is `experiments/pq_prefill_accuracy_curve.py` reading
the 49 rows that `admit_native_rows` **did** admit, and summing them the way the
table's own `composition` field says to (`sequential_operator_sum`, a sum of
per-row medians). It is a measured prefill-price-versus-accuracy curve over real
receipts; it is **not** the allocator's output, and no SLO-constrained frontier
exists on this evidence. Read §4 before quoting §3.

### 3.1 Accuracy metric identity

```
metric     KL-vs-BF16, top-1024 support, teacher-student-intersection partition
bound      lower bound (data-processing inequality)
regime     prefill, 4088 positions
corpus     076d33efc447 (4096 tok)   tokenizer 76f13c8e6e55
teacher    qwen_teacher_bf16_lina (BF16 /home/rob/models/Qwen3-0.6B,
           host gx10-6b77, 2026-09-02T17:09:06Z)
tool       kl_tool.py compare        source accuracy/accuracy.md
scope      served whole-artifact KL; NOT a per-unit quantity, not re-measured here
```

### 3.2 The curve

Prefill and decode are operator sums over the 7 priced layer-0 units, in
milliseconds. Brackets are the **paired bootstrap interval over each row's own
CUDA-event samples** — every row resampled with replacement from its own 32
samples, re-reduced by the same median, re-summed; 10 000 draws, seed 237. The
interval describes only the dispersion the measurement itself carries. No
threshold is applied and no verdict is derived from it; it is published so a
reader cannot mistake an unresolved ordering for a measured one.

| artifact | prefill ms [2.5 %, 97.5 %] | decode ms [2.5 %, 97.5 %] | serialized B | resident B | bpw | KL | cKL | top-1 |
|---|---|---|---|---|---|---|---|---|
| `uniform-R1262` | 0.268544 [0.267872, 0.269952] | 0.260272 [0.259472, 0.261008] | 9 836 258 | 15 777 792 | 4.9297 | 0.056389 | 0.038868 | 0.8640 |
| `uniform-R750` | 0.269392 [0.268672, 0.270528] | 0.260736 [0.260368, 0.261152] | 5 904 082 | 15 777 792 | 2.9297 | 0.969951 | 0.841033 | 0.5083 |
| `alloc-5.0` | 0.269488 [0.268960, 0.270608] | 0.260720 [0.260192, 0.261600] | 9 835 748 | 15 777 792 | 4.9294 | 0.162200 | 0.116254 | 0.7938 |
| `alloc-3.0` | 0.269632 [0.268816, 0.270928] | 0.260448 [0.260144, 0.261392] | 5 903 570 | 15 777 792 | 2.9294 | 2.257361 | 2.483408 | 0.2833 |
| `alloc-4.0` | 0.270048 [0.269584, 0.271056] | 0.260112 [0.259632, 0.260832] | 7 869 663 | 15 777 792 | 3.9294 | 0.348474 | 0.263438 | 0.6835 |
| `uniform-R1006` | 0.270784 [0.269344, 0.271584] | 0.261984 [0.261392, 0.262416] | 7 870 169 | 15 777 792 | 3.9297 | 0.174557 | 0.114561 | 0.7747 |
| `bf16-k1-r7-t460` | 0.483104 [0.481600, 0.484208] | 0.205584 [0.205168, 0.206208] | 14 021 376 | 31 506 432 | 7.0000 | 0.004923 | 0.003311 | 0.9582 |

Artifact names are `qwen3-0.6b-<arm>`. Operator routes: all six E4M3 arms route
all seven units to `torch._scaled_mm`; the BF16 arm routes all seven to
`torch.mm`.

**Amended (PQ #559) — the route name in this table is the GEMM symbol alone,
and that is not enough to name a route class.** `native_receipt_table.py:179`
derives each row's `binding.operator_route` (`:184`) as
`panel["phases"]["prefill"]["expected_route"]["symbol"]`, so both `TESSERA_FP8`
and `TESSERA_NVFP4` record the identical string `torch._scaled_mm` while
executing different kernels on differently packed operands. The distinguishing
field is in the receipt but not in the table row: the route's `policy`
(`TESSERA_FP8:resident` against `TESSERA_NVFP4:resident`), `decoder`
(`torch_window` against `native_span2`) and `contract` (`fp8_per_token_dynamic`
against `e2m1_group16_ue4m3_static`). A route class is the pair (route policy,
activation contract); the GEMM symbol alone collapses two of them. Recorded as a
schema observation, not changed here.
### 3.3 Finding 1 — prefill price at layer 0 is a route-class step

The six E4M3 arms' prefill intervals **all overlap pairwise**; their union is
[0.267872, 0.271584] ms. The BF16 arm's interval [0.481600, 0.484208] ms is
disjoint from all of them: **1.77x to 1.81x** (interval-derived range) every
E4M3 arm.

**This confirms a design guarantee rather than discovering a surprise, and the
guarantee is the reason it is worth publishing.** In `TESSERA_SERVE_MODE=resident`
the trellis wire is unpacked once at load; the timed apply — `sample_unit:
single_apply` in every receipt — is the same `[512,1024] x [1024,N]` GEMM for
R750 and R1262 alike. The rate cannot move the apply time. The measurement
therefore prices the **route class**, `torch._scaled_mm` against `torch.mm`, and
it prices it on the hardware rather than asserting it.

The bytes confirm the mechanism directly: `resident_bytes_sum` is **identical**
(15 777 792 B) across all six E4M3 arms while `serialized_bytes_sum` moves
5 903 570 → 9 836 258 B with the rate. Rate changes what is stored; it does not
change what the operator applies.

**Amended (PQ #559) — narrowed, not retracted.** The finding holds exactly as
measured, for the two route classes it measured: `torch._scaled_mm` fp8 x fp8
against `torch.mm`, a 1.77x to 1.81x step with disjoint intervals. What it does
**not** support is the general form "prefill price is a step from one route class
to the next". §8 adds the third class on the same box, same session, same units:
`TESSERA_NVFP4` lands within about 4 % of `TESSERA_FP8` on prefill — disjoint
intervals, so a resolved ordering, but not a step — while `torch.mm` is still
2.30x away. On this evidence the large prefill separation is between the
`_scaled_mm` family and `torch.mm`, and the two `_scaled_mm` classes are close to
each other. See §8.4 and §8.8.

### 3.4 Finding 2 — prefill and decode are anti-monotone across that boundary

BF16 decode [0.205168, 0.206208] ms is disjoint from, and faster than, every
E4M3 arm's decode; E4M3 pays **1.26x to 1.28x** the BF16 decode. The direction
inverts across the same route boundary that Finding 1 prices.

Concretely, at layer 0 under this scope: an allocator minimising prefill picks
E4M3 and pays ~26-28 % more decode; one minimising decode picks BF16 and pays
~77-81 % more prefill. The obvious reading is per-call overhead of
`torch._scaled_mm` dominating at M = 1 against its throughput advantage at
M = 512, but this record does not measure the kernel internals and does not
claim that mechanism — only the two disjoint orderings.

This is the fact that makes a prefill-only frontier incomplete. It is the reason
the curve tool bootstraps both phases rather than only the one the deliverable
asked for.

### 3.5 Non-findings, published as unresolved

* **No ordering among the six E4M3 arms on prefill is resolved.** Every pair
  overlaps. The rate axis (R750 … R1262) buys a factor of ~40 in KL — 0.969951
  down to 0.056389 — at **no prefill cost this measurement can resolve**.
* **On decode, four pairs do separate, and all four involve `uniform-R1006`**
  (against `uniform-R1262`, `uniform-R750`, `alloc-3.0`, `alloc-4.0`). The
  ordering is **not monotone in rate** — R1006 is slower than both R750 below it
  and R1262 above it — so this is not a rate effect, and this record does not
  identify what it is. It is reported, not explained.

---

## 4. Refused — three independent causes, none worked around

The shipping consumer was run against the real table, unmodified:

```
$R/stage/control/run_frontier.sh        # durable, citable invocation
$R/frontier/command.txt  stdout.txt  stderr.txt  exit.txt  refusal-shapes.json
```

`python3 -m prismaquant.prefill_frontier --output … --slo-grid 5 -- --probe …
--costs … --measured-runtime-table $R/table/qwen3-0.6b-layer0-all.json
--measured-runtime-context $R/stage/evidence/context-all.json`

**exit 1**, stderr beginning `[alloc] ERROR: measured runtime: no qualified
recomputable full-engine resource partition: …`. Summary:
`$R/frontier/refusal-shapes.json` (sha256
`0e9daad75203b1209d4126aab451d3526a481855735a9ff30b10281f8b82d76f`) — 21 468
refusal parts, 143 distinct shapes.

**`--probe` is a placeholder on this path and was never read.** `allocator.py`
loads the measured runtime table at `:2262` and first opens `args.probe` at
`:2319`; the refusal happens at `:2262`. Verified empirically: re-running the
same command with `--probe /nonexistent/probe-that-does-not-exist.pkl` produces
the identical refusal and the identical exit code. No DP ran, and no allocation
was proposed. (The `selected row (…, 'TESSERA_FP8')` refusals below are about
the #399 report's own recorded workload selection, not about any allocator
output.)

### 4.1 Cause A — the transient charge boundary is not versioned (design debt)

`runtime_provenance.py:718-719` appends, **unconditionally**:

> `the native-row and full-engine transient charge boundary is not versioned, so
> no candidate activation or scratch term may be compared to a priced row`

Consequence, traced: `admit_fixed_resources` can never pass for *any* v2 table →
`load_measured_runtime_table` always refuses → `producer_admitted` is never set →
`build_runtime_resources` refuses at `measured_runtime_prices.py:477`. **A v2
table structurally cannot reach the allocator today**, no matter what it
contains. `FIXED_TERM_FIELDS`' own docstring records this as owed design work
("Set native/full-engine charge boundary").

This is the deliverable-6 disposition: it is **reported, not patched**. Versioning
that boundary is a contract decision about what a native row and a full-engine
run each own; inventing one here to make a gate pass would be exactly the
band-aid principle 1 forbids. Filed as debt D37 in `docs/ARCHITECTURE.md` §12,
with this table as the artifact that demonstrates it.

### 4.2 Cause B — the #399 a5 ledger is not closed

21 328 of the 21 468 parts are the a5 report's own allocation ledger:

| count | refusal shape |
|---|---|
| 20 734 | `allocation <id> is unclassified (no single supported owner category), which nulls every term` |
| 370 | `allocation <id> is unclassified (shared or unknown ownership supplies neither classification nor invariance), which nulls every term` |
| 224 | `allocation <id> is classified (candidate, resident) but no term charges it` |
| 1 | `the ledger carries unresolved issues` |

Plus six unclosed domains, one each: `worker_startup`, `history_join`,
`external_closure`, `provenance_admission`, `cache_capacity`,
`timing_partition` — "… is not closed, so every term depending on it stays
null".

And, consequently, every fixed term refuses for want of evidence rather than
being defaulted: `no fixed_resident is recomputable, so this table's fixed
resident_bytes (0) has no evidence` and the matching lines for
`fixed_activation`, `fixed_scratch`, `fixed_kv`; `no off-step transient peak is
recomputable`; `no candidate_resident is recomputable`; `no scalar device budget
is expressible from this report`; `no placement obligation is recomputable from
this report`; and six `the capture observes no <kv_observations |
observer_qualification | owner_views | runtime_provenance_relation |
timing_captures | worker_startup_records>` lines.

This is a **producer gap on the Tessera side** (RobTand/tessera#399), not a
consumer defect. The a5 report is exactly what #399 says it is: "full-engine
resource observation of one served artifact; no timing, fixed-resource or
release admission".

### 4.3 Cause C — the a5 report measured other bytes than this table's

```
stale model_sha256: the report was measured on other bytes
stale runtime_manifest_sha256: the report was measured on other bytes
the workload names another calibration than this table's
```

The a5 report was measured on `/mnt/shared/tessera-runs/allocated/qwen3-0.6b-uniform-R1024-tpstamp`
(uniform `TESSERA_FP8` E4M3 q256=1024, 112 modules / 196 units;
`model_sha256 787f4c9adc4d…`, `runtime_manifest_sha256 de396fd9ce0f…`,
workload calibration `3e933e10d856…`). This table's context names calibration
`5f92bfdd5154…`.

The point is not the mismatch itself but what a "fixed charge" has to be. A
fixed whole-model charge priced from one artifact is only admissible against
another artifact if it is *invariant* under the difference between them; the
gate says so directly, twice:

* `the table prices more than one format for [the 7 layer-0 units], and one
  measured assignment establishes no invariant fixed charge under the others`
* `the canonical census names units [112 fused g:/l: units] where this table
  prices [7 leaf units]`, followed by 112 `selected row (<g:/l: unit>,
  'TESSERA_FP8') is not a row this table prices`

Both of these are constraints on **`admit_fixed_resources` only**
(`_fixed_resource_refusals`, `runtime_provenance.py:564-728`). They do **not**
constrain `admit_native_rows` (`:731`), which is why a 49-row multi-format table
was admitted at §2.6. An earlier working note in this branch attributed them to
`admit_native_rows`; that was wrong, and the empirical admission settles it.

---

## 5. Null — what is absent rather than refused

* **No whole-model prefill price.** Only the operator sum over 7 of 196 units.
  This is not a latency figure and not a TTFT figure; `prefill_frontier`'s own
  banner says whole-operator sums cannot certify p95 or end-to-end SLOs.
* **No SLO-constrained frontier.** `--slo-prefill-p95-ttft-ms` was exercised
  through `--slo-grid 5`; the run refused before any grid point was evaluated,
  so no SLO result exists to report either way.
* **No layers other than 0.** 7 units; the model has 196 in its streamed body.
* **No fused-unit prices.** The allocator fuses q/k/v and gate/up by default
  (`allocator.py:3504-3509`) and refuses when serving promotion changes a
  measured proposal (`:3950-3963`). These are **leaf**-unit receipts, so only
  units with no fused siblings (`o_proj`, `down_proj`) could be priced under the
  allocator's own unit convention without whole-group measurements. The other
  five are priced here as leaves and are, for allocator purposes, a gap.
* **No second model, no second box, no repeat session.** One GPU, one day.

---

## 6. Reproducing this

Driver tooling, staged with digests at `$R/stage/control/` (`SHA256SUMS.txt`):
`g1_prepare.sh`, `g2_launch.py`, `g2_driver.sh`, `g2_bench.sh`,
`full_engine_plugin_install.py`, `pq_frontier_native_runner.py`,
`make_relation_plan.py`, `admit_report.py`, `run_frontier.sh`,
`draw_calibration.{py,sh}`, `context-all.json`,
`arm-assignments-layer0.json`, `accuracy-column.json`.

In-repo modules: `prismaquant/native_receipt_table.py` (emitter + CLI),
`experiments/pq_frontier_native_cells.py` (prepare / freeze / manifest),
`experiments/pq_runtime_relation.py` (relation builder),
`experiments/pq_prefill_accuracy_curve.py` (the curve).

`--measured-runtime-context` takes a **standalone context document**, not the
table: passing the whole table gives
`expected exactly fields ['batch_size','calibration_sha256','gpu_identity',
'graph_mode','operator_routes','prompt_tokens','runtime_sha256','schema',
'serving_context','source_sha256','tensor_parallel']`. `context-all.json` is
`table["context"]` alone.

## 7. Digest fact sheet

| Object | sha256 |
|---|---|
| table `qwen3-0.6b-layer0-all.json` | `9bbb283a2ea4dae23cc1f18dd688ec7dd86ce54877fd82359d6123c12b27d3c9` |
| relation `relation-all.json` | `40d2a7b5051c26b7b98fe006bacbc53aa9f125f10340b1a0d52b8e8f37e4d890` |
| curve `prefill-accuracy-qwen3-0.6b.json` | `92b28f9e1127dcb777ef1d46aa89f7c08ad0f01cf7f2f61085711f550a6da21d` |
| cost payload `full2/prep/joint.pkl` | `7dbfe9456635a33f824888010f68287aafc887480d52adb06dfe1d717a7cb391` |
| calibration safetensors | `20b2cc8f717bdc4b9efdd49fb2567cbb64f019c2e2c9abdb4b47c7eb3e2928f7` |
| image manifest | `0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c` |
| Tessera source tar (78 members) | `13127ff7da5e7ec53208cb974fda8ad070a6993fa523995ea863a6928363976c` |
| #399 a5 report | `219585f45d5e5d4c92624ed3e327200f8dd87785635ca27e345dac5ee29b904f` |
| frontier refusal summary | `0e9daad75203b1209d4126aab451d3526a481855735a9ff30b10281f8b82d76f` |
| admit report `$R/stage/evidence/admit-all.json` | `cfed505aa6f36ff411728d214577bc6df8cc91e03338552c605e0d9b9b2f9857` |
| evidence manifest `$R/stage/evidence/SHA256SUMS.txt` | `8f59cfad4438149fdb33635b854c2637fe9301bc5947a74ef299cb3a3795a5a7` |

Context digests: `runtime_sha256`
`243010b3aa57c1b321d372a3d4ec0970bf67d2092853d66001491ced7c5fbde2`,
`source_sha256`
`665160a182a42a16cf5e6b33ed6a405e525823194daa431ced57699dea5bde49`,
`calibration_sha256`
`5f92bfdd51549668f8ba576137c9597707bc13c52bf4f6fca99af9f4ef3a9b36`.
Source seals: producer tree `0833671b…` (76 files), installed subset
`d9067c76…` (71 files), installer identity `ea51743a…` (78 tar members).

---

## 8. Amendment — the fp4 route class (PQ #559)

2026-09-13, branch `claude/price-the-fp4-route-class`, PQ #559. Receipts root
(`$NR` throughout):

```
/mnt/shared/tessera-runs/receipts/frontier-fp4-qwen3-0.6b-20260913
```

`$R` above is frozen and was not modified. `$NR` is a separate root built from
the same driver set, the same producer tree
(`producer-source-d403cc5a31`, `contract_version 22`), the same image
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`,
the same calibration `5f92bfdd5154…` and the same box, so every number in §8 is
comparable to §3 in everything except the units and the route class.

### 8.0 Outcome, first

1. The fp4 route is **native** and proven so from the receipt: `TESSERA_NVFP4`,
   `torch._scaled_mm`, decoder `native_span2`, contract
   `e2m1_group16_ue4m3_static`, `state: served`, `reason: null`. Not a fallback
   (§8.2).
2. **fp4 is not faster than fp8 at prefill at this scope.** Same session, same
   three MLP units, byte-matched arms: fp4 0.133104 ms against fp8 0.127616 ms,
   disjoint intervals, about 1.04x. At decode fp4 is the slowest of the three
   (§8.4). Rob's expectation is not supported by this measurement.
3. A 7-unit fp4 arm **cannot be built** on this evidence: 3 of the 7 layer-0
   units are `timing_admissible`, 2 refuse the activation-side numeric gate, 1
   (`q_proj`) holds no receipt but the diagnostic probe shows it would refuse for
   the same reason `v_proj` did, and 1 (`k_proj`) was never run (§8.3).
4. Even the 3 admissible fp4 rows **cannot enter a v2 table or the curve**, for
   two structural reasons that have nothing to do with the numbers: a v2 table
   refuses to hold fp4 rows beside fp8/bf16 rows at all, and the fp4 rows have no
   full-engine bytes to bind to (§8.5).
5. The accuracy coordinate for a uniform `TESSERA_E2M1_K2_R896` arm was **not
   attempted** and is carried as an explicit `null`, not estimated (§8.7).

### 8.1 Production runs

| Stage | PB key | Host | Outcome |
|---|---|---|---|
| G1 prepare (`prepA`) | `167efb54e93f` | sparklina | 56 cells prepared, 185.2 s, done |
| G2 `native-t1` | `cb9f2af2ed8e` | sparklina | rc 2, `v_proj` fp4 `numerical_refused` |
| G2 `native-f1` (7 fp4 units) | `3824abd5aac8` | sparklina | rc 2, 3 admissible then `o_proj` refused |
| G2 `native-f2` (9 cells, 3 families) | `28afcdc891db` | sparklina | 9/9 `timing_admissible`, 323.7 s, done |
| diagnostic qdq probe | `4419403135f9` | sparklina | done, 50.4 s, `$NR/diagnostics/d4/report.json` |

Three earlier probe submissions failed on harness faults and are recorded rather
than hidden: `328f7473b1f9` (rc 1, `PYTHONPATH` missing `/tessera/src`),
`7960d88fa839` (rc 1, the tensor-parallel group was not initialized — fixed by
running inside Tessera's own `native_runtime_context()`), `4b3e7438f430` (rc 1,
`float(None)` on the fp8 control cell, whose `input_global_scale` is `null`).
Every status above was read from `pb-queue/{done,failed}/<key>.json`
`detail.action_returncode`, not from a wrapper message. All GPU work went
through PrismaBuild at `--priority -10`; every action is far under the
30-minute ceiling.

**Power, read against the envelope.** Netdata `nvidia_smi.gpu_power_draw` on
sparklina over the `28afcdc891db` window (2026-09-13T17:20:33Z to 17:25:57Z):
mean **8.15 W**, min 4 W, peak 12 W against the 140 W envelope — a mean envelope
fraction of **0.058**. On GB10 `gpu_utilization` is non-diagnostic, and power
says plainly that these layer-0 cells do not load the device. Read every §8.4
number as a launch-and-activation-quantize overhead price, not an arithmetic-rate
price.

### 8.2 The route is native fp4 — deliverable 2

Every fp4 cell's `receipt.json` `phases.{prefill,decode}.route` records:

```
policy     TESSERA_NVFP4:resident
symbol     torch._scaled_mm
decoder    native_span2
contract   e2m1_group16_ue4m3_static
state      served          reason  null          tile_m  0
```

Compare the same field on the other two families in the same session:
`TESSERA_FP8:resident` / `torch._scaled_mm` / `torch_window` /
`fp8_per_token_dynamic`, and `TESSERA_BF16:resident` / `torch.mm` /
`torch_window` / `bf16_unquantized`.

The fallback would have been visible. `scheme.py:466-477` declares
exactly one launch for `TESSERA_NVFP4` — decoder `native_span2` — and its own
comment says the stock-materialize decoder "is what runs when the extension is
ABSENT … and is not a launch an attested cell may name". `ext.py:292-295` is
where that fallback is published: resident mode substitutes decoder
`torch_materialize_stock`, streamed mode refuses. A substituted decoder is a
different `decoder` string, which the panel comparison refuses. It did not run: the JIT extension was built and loaded, and
its bytes are named in every fp4 receipt's runtime record:

```
/out/cache/extensions/tessera_nvfp4_84439e84…/tessera_nvfp4_84439e84….so
sha256 7bfe7714dba8f16b03724772d2cad0449684f62082accb7c5c36a484156cc9d2
```

### 8.3 The activation-side numeric gate — 3 admissible, 2 refused, 1 refused by the probe, 1 not run

Both numeric gates run at one declared tolerance, `atol = rtol = 0.015625`
(2^-6). Nothing below was relaxed, skipped or widened.

| unit | fp4 status | prefill `numerics` | prefill `qdq_numerics` |
|---|---|---|---|
| `mlp.down_proj` | `timing_admissible` | passed, 0.015625 | passed, **0.0** |
| `mlp.gate_proj` | `timing_admissible` | passed, 0.015625 | passed, **0.0** |
| `mlp.up_proj` | `timing_admissible` | passed, 0.015625 | passed, **0.0** |
| `self_attn.o_proj` | `numerical_refused` | passed, 0.0166015625 | **failed, 0.095703125** |
| `self_attn.v_proj` | `numerical_refused` | passed, 0.013671875 | **failed, 0.1435546875** |
| `self_attn.k_proj` | no receipt — not run | — | — |
| `self_attn.q_proj` | no receipt — probe says it would refuse | — | (probe: 0.1435546875) |

The driver exits on the first refusal, so `k_proj` and `q_proj` hold **no
receipt**. They are absent, not passing. The two are not in the same position,
though: the d4 diagnostic probe (below) *did* run `q_proj`, and it records the
identical disagreement `v_proj` records — same input tensor, same `G`, same
17 978 / 0.1435546875. `q_proj` would have refused the gate for the same reason.
`k_proj` was never executed by anything and no claim is made about it.

`measure_prepared_operator` takes **no** timing samples unless both gates pass,
so a refused cell has no price at all, and `native_operator_panel.py:190`
refuses any receipt whose status is not `timing_admissible`. The gate is doing
its job in both places.

**The same units in the same session on fp8 are exactly 0.0.** The
`TESSERA_E4M3_K1_R1006` control cells record `qdq_numerics.max_abs_error = 0.0`
at both phases. The refusal is specific to the fp4 activation path, and it is
the activation side: the output `numerics` gate passed on every refused cell.

**What the disagreement is, decomposed** (`$NR/diagnostics/d4/report.json`, one
process, one GPU, Tessera's `represented_native_input` against PrismaQuant's
`nvfp4_activation_qdq_served`):

* The global scale `G` is the same object on both sides —
  `bench_native_operator.py:458` refuses the cell outright if the operator's
  `input_global_scale` differs from the joint activation record's.
* **Block scales agree bit for bit on every group of every cell**:
  `groups_differing 0`, `max_abs_scale_diff 0.0`, `max_relative_scale_diff 0.0`.
* **The scale-plane swizzle is not the cause.**
  `swizzle_roundtrip_covers_every_group: true`, and
  `swizzle_plane_used == swizzle_plane_elements == 32768` — the whole plane is
  used. The 2026-08-17 NVFP4_CB failure mode, where `torch._scaled_mm` accepts
  an unswizzled scale plane and silently miscomputes at 67–70 % error, is **not**
  firing here; this is a 3.43 % element disagreement, not a wrong answer.
* **PrismaQuant's frozen `reference_qdq` is bit-equal to
  `nvfp4_activation_qdq_served(x, G)`** (`reference_qdq_equals_pq_oracle: true`),
  so the reference is PrismaQuant's own oracle and nothing in between.
* The residual is **isolated single E2M1 code flips**. The worst element on
  `v_proj` prefill: `abs_error 0.1435546875`, `smallest_e2m1_step
  0.1432291716337204`, `abs_error_in_steps 1.0022727` — one step of the code
  lattice, with `native_stored_scale == pq_stored_scale == 0.5`. 17 978 of
  524 288 elements, 3.43 %.
* It is **input-tensor-specific, not shape-specific**. `q_proj` and `v_proj`
  share one input tensor and one `G` (1.7454545) and record the identical
  17 978 / 0.1435546875. `o_proj` takes a different input and records
  0.095703125. `gate_proj`, `up_proj` and `down_proj` are **bit-identical, 0
  differing elements, max 0.0**, at both phases. The fp8 control is 0.0.

One field in `d4/report.json` must not be read as evidence:
`e2m1_level.elements_differing` compares the two sides in their **own** storage
orderings, which are not aligned, so it reports large counts (937 722 on
`down_proj`) on cells whose actual `native_vs_reference` difference is exactly
0. The aligned quantities are `native_vs_pq`, `native_vs_reference`,
`block_scale` and `worst_element`; those are what is cited above.

PrismaQuant's own oracle says where to look: the docstring at
`prismaquant/nvfp4_activation_contract.py:1123-1125` records that installed
kernels use `rcp.approx.ftz.f32` in `outputScale`, so "arbitrary random Torch
results are a numerical oracle, not a packed-byte equivalence claim". A one-code
tie-break difference at a rounding boundary is consistent with that. **This
record stops there.** Deciding which side is authoritative, and whether the fp4
render and the fp4 execution are the same object under principle 8, is a
separate question from #559 and is filed separately. Nothing here was fixed,
and no tolerance was moved.

### 8.4 Measured — the same-session three-family comparison

`$NR/native-f2/cells/` · PB `28afcdc891db` · one container, one session, one
GPU. Nine cells: the three layer-0 MLP units at each of the three route classes.
32 CUDA-event samples per phase per cell, 8 warmup iterations, median per row,
`sequential_operator_sum` over the three units. Brackets are the same paired
bootstrap §3.2 uses — each row resampled with replacement from its own 32
samples, re-reduced by median, re-summed, 10 000 draws, seed 237
(`$NR/stage/evidence/bootstrap-f2.json`).

**These are 3-unit sums. §3's sums are over 7 units. The two are not
comparable as absolute numbers; only the ratios within each table are.**

| route class | format | prefill ms [2.5 %, 97.5 %] | decode ms [2.5 %, 97.5 %] | serialized B |
|---|---|---|---|---|
| `TESSERA_FP8` / `_scaled_mm` | `E4M3_K1_R1006` | **0.127616** [0.127120, 0.128432] | **0.112608** [0.112288, 0.112928] | 4 701 232 |
| `TESSERA_NVFP4` / `_scaled_mm` | `E2M1_K2_R896` | **0.133104** [0.132528, 0.133840] | **0.131584** [0.130944, 0.132288] | 4 722 278 |
| `TESSERA_BF16` / `torch.mm` | `BF16_K1_R1792` | **0.293440** [0.292832, 0.294080] | **0.088704** [0.088448, 0.088976] | 8 372 288 |

The fp4 and fp8 arms are **byte-matched to 0.45 %** (4 722 278 B against
4 701 232 B), so this is a route-class comparison at one rate, not a rate
comparison.

**Finding 3 — fp4 is slower than fp8 at prefill, at this scope.** The intervals
are disjoint: fp4 costs about **1.04x** fp8. The expectation going into #559 was
that fp4 would beat fp8 at prefill. It does not, here.

**Finding 4 — fp4 is the slowest of the three at decode.** 1.17x fp8 and 1.48x
bf16, all intervals disjoint. The §3.4 anti-monotone pattern still holds between
`_scaled_mm` and `torch.mm`; fp4 does not sit on either end of it.

**Finding 5 — the fp4 arm's prefill is its own decode.** Taking each arm's
3-unit sum, prefill ÷ decode is **1.01x for fp4** (0.133104 / 0.131584), 1.13x
for fp8 (0.127616 / 0.112608) and 3.31x for bf16 (0.293440 / 0.088704), across a
512x change in M. Per cell the fp4 pattern is uniform, not an artifact of one
unit: `up_proj` 0.044368 → 0.044384, `gate_proj` 0.043856 → 0.043072,
`down_proj` 0.044880 → 0.044128. At 8 W of a 140 W envelope, the fp4 cell is
bound by something that does not scale with M. Both `_scaled_mm` routes
pre-decode the weight once and then quantize the **activation on every apply**:
`nvfp4_route.py:226` calls `native_ops.native_fp4_quant(x2, gs)` and
`fp8_route.py:444` calls `native_ops.native_fp8_quant(x2)`. So "fp4 quantizes
the activation and fp8 does not" would be wrong — they both do. What differs is
the shape of that work: fp8 produces one fp32 scale per token, fp4 produces
group-16 UE4M3 block scales into a swizzled plane. bf16 (`torch.mm`) quantizes
nothing and is the only arm whose prefill/decode ratio tracks M at all. This
record measures the timing and names the code paths; it does not profile the
kernels and does not claim the mechanism.

**Labelling these numbers.** All nine f2 cells are `timing_admissible` — the
three fp4 rows passed **both** numeric gates, with `qdq_numerics` at exactly 0.0.
They are excluded from the v2 table and from the curve by §8.5's two structural
gates, not by a numeric refusal. The two numerically refused units (`o_proj`,
`v_proj`) carry no timing at all.

### 8.5 Run the gates unmodified — deliverable 3

Three tables were emitted from the same nine receipts, and every gate was run
unchanged. `admit_report.py` calls `load_runtime_relation`, `admit_native_rows`
and `admit_fixed_resources` in order and reports each, because
`admit_runtime_provenance` raises on the first refusal and a single verdict
cannot say whether the middle gate admitted.

| rows | relation | emitter | `load_runtime_relation` | `admit_native_rows` | `admit_fixed_resources` |
|---|---|---|---|---|---|
| **9** (fp4 + fp8 + bf16) | exit 2 | exit 1, **no table** | not reached | **not reached** | not reached |
| **3** (fp4 only) | exit 2 | exit 2, table emitted | **refused** | **not reached** | not reached |
| **6** (fp8 + bf16) | exit 0 | exit 2, table emitted | loaded, 7 runs | **admitted, 6 rows** | refused, 143 shapes / 21 468 parts |

Verbatim verdicts, in order of how early they bite:

**A. A v2 table cannot hold fp4 rows beside fp8 or bf16 rows.**
`prismaquant/native_receipt_table.py:209`, `derive_context`:

> `native receipts were produced on more than one runtime`

`derive_context` requires `identity_sha256(panel["runtime"])` to be equal across
every bound panel. Diffing the nine same-session panels field by field, the
**only** field that differs is `runtime.native_libraries`: the three fp4 panels
carry the JIT-built `tessera_nvfp4_84439e84….so`
(`7bfe7714dba8f16b…`) and the six fp8/bf16 panels do not. Image, GPU, execution
mode, arithmetic, versions, source and resource collector are all equal. So, as
runtime identity is currently recorded, loading a route's kernel extension makes
that route's rows a different runtime from every route that does not load it, and
one table cannot hold both. That blocks a **mixed-format** prefill table, not
only #559's fp4 arm. Reported; not changed here.

**B. The fp4 rows have no full-engine bytes to bind to.**
`experiments/pq_runtime_relation.py` on the fp4 runs:

> `{"status": "incomplete", "error": "3 native production libraries have no full-engine bytes"}`

and the consuming gate, `runtime_provenance.py:431` via `:39`:

> `complete native production dependency coverage: evidence mismatch`

`runtime_provenance.py:418-435` requires every native run's production library
to exist in the full-engine run with the same sha256. The unbound library is the
same `tessera_nvfp4_84439e84….so`, for each of the three fp4 runs. The #399
full-engine run `engine-a5` served a `TESSERA_FP8` artifact and never loaded the
nvfp4 extension, so those bytes have no counterpart. This is the #323 failure
mode §2.3 checked for and found **absent** for fp8 and bf16 — and present for
fp4, because fp4 is the route class that JIT-builds a `.so`.

The fp8/bf16 control in the same session behaves exactly as §2.4 did:
`relation-f2ctl.json`, verdict **loaded**, 1 086 production dependencies, 40
full-engine extra libraries, 0 unbound.

**C. `admit_native_rows` admits the control, and the fixed-charge debt is
unchanged.** On the 6-row fp8/bf16 table, `admit_native_rows` **admitted 6 rows,
refusal `None`** — §2.6 reproduces on new receipts. `admit_fixed_resources`
refuses with the same debt D37 text as §4.1:

> `no qualified recomputable full-engine resource partition: stale model_sha256:
> the report was measured on other bytes; …`

So of the three gates, **two refuse the fp4 arm for reasons the E4M3 rows never
hit**, and the third refuses everything equally and is not a #559 finding. This
is the deliverable-3 outcome the brief anticipated: the new rows refuse for a
reason the E4M3 rows did not, reported and not patched.

Records: `$NR/stage/evidence/{emit,relation,admit,curve}-{f2,f2fp4,f2ctl}.*`.

### 8.6 Re-emit the curve — deliverable 5

`experiments/pq_prefill_accuracy_curve.py`, unmodified, same bootstrap (10 000
draws, seed 237), over the three MLP units with three arms declared:
`uniform-R1006` (fp8), `bf16-k1-r7-t460` (bf16), and `uniform-E2M1-R896` (fp4).

**On a table containing the fp4 rows the tool refuses and emits nothing** — it
calls `load_runtime_relation` before it prices anything, so it exits 1 with
refusal B above (`$NR/stage/evidence/curve-f2fp4.{exit,stderr}.txt`).

**On the fp8/bf16 table it emits, and the fp4 arm lands in
`refused_artifacts`** — `$NR/curve/prefill-accuracy-qwen3-0.6b-f2ctl.json`,
sha256 `01425b9b3c395c2747ae684822fd9f2ec278f131e2e108579c1db59035eeeb54`,
2 points, 1 refused artifact:

```
qwen3-0.6b-uniform-R1006     prefill_sum_ms 0.127616 [0.127120, 0.128432]   decode 0.112608
qwen3-0.6b-bf16-k1-r7-t460   prefill_sum_ms 0.293440 [0.292832, 0.294080]   decode 0.088704

refused_artifacts:
  qwen3-0.6b-uniform-E2M1-R896
    unpriced_cells: down_proj@TESSERA_E2M1_K2_R896, gate_proj@…, up_proj@…
    accuracy: null
```

Both of the arm's defects are named in the document itself: the cells are
unpriced because the table cannot hold them (§8.5), and the accuracy coordinate
is absent (§8.7). The 2.30x step between `torch._scaled_mm` and `torch.mm`
reproduces on the 3-unit scope, in a new session, on new receipts.

### 8.7 The accuracy coordinate — deliverable 4, not attempted

No KL was measured for a uniform `TESSERA_E2M1_K2_R896` arm, and none is
estimated. Two reasons, stated separately:

1. **A KL alone is not a curve point.** The prefill coordinate for this arm does
   not exist (§8.3, §8.5), so a measured KL would have nothing to pair with.
2. **No such artifact exists to serve.** There is no uniform
   `TESSERA_E2M1_K2_R896` export under `/mnt/shared/tessera-runs/allocated/`, and
   `experiments/export_tessera_serving.py` — whose defaults are exactly
   `--grid E2M1x2 --q256 896` — requires `--input-scales` from a stock NVFP4
   export that has not been produced. Building one is an export campaign, not a
   serve, and is out of scope for this issue.

The arm is carried as an explicit `null` in
`$NR/stage/control/accuracy-column-fp4.json`, which is the mechanism the curve
tool already has for a missing coordinate: a `null` accuracy puts the artifact in
`refused_artifacts` rather than silently dropping it. No vLLM serve was started
for #559, so none of the serve rules were exercised.

### 8.8 What §3.3 becomes

§3.3 measured two route classes and found a 1.77x–1.81x step between them. With
a third class measured on the same box, the same session and the same units, the
finding narrows:

* The large prefill separation is between the **`_scaled_mm` family and
  `torch.mm`** — 2.30x on the 3-unit scope here, 1.77x–1.81x on the 7-unit scope
  in §3.
* The **two `_scaled_mm` classes are close to each other**: fp4 costs 1.04x fp8
  at prefill. The ordering is resolved (disjoint intervals) but it is not a step.
* So "prefill price at layer 0 is a route-class step" is **true of the pair §3
  measured and false as a general rule**. Prefill price is not a function of
  route class alone at this scope; two classes sharing a GEMM symbol can land 4 %
  apart while a third is 2.3x away.
* And, at 0.058 of the power envelope, none of these three numbers is an
  arithmetic-rate measurement. What separates them is per-apply overhead, not
  peak FLOPs. A scope where the device is loaded may order them differently, and
  this record makes no claim there.

The §3.2 note applies to the table schema as well as to the prose: a row's
`binding.operator_route` is the GEMM symbol alone, so an fp4 row and an fp8 row
are indistinguishable in that field. Anything that groups rows by route class
should read (route policy, activation contract) from the receipt.

### 8.9 Reproducing §8, and digests

Driver set staged with digests at `$NR/stage/control/` (`SHA256SUMS.txt`,
sha256 `ec5838e2e566a2f331262824c7f5bcf97dd560a9e5995883cbba1d4d4752f31e`):
`g1_prepare.sh`, `g2_bench.sh`, `g2_launch.py`, `g2_driver.sh`,
`pq_frontier_native_runner.py`, `full_engine_plugin_install.py`,
`make_relation_plan.py`, `emit_and_gate.sh`, `admit_report.py`, `run_curve.sh`,
`fp4_qdq_probe.{py,sh}`, `cells-fp4.json`,
`arm-assignments-layer0-mlp.json`, `accuracy-column-fp4.json`. `g1_prepare.sh`
and `g2_bench.sh` differ from `$R/stage/control/`'s only in the receipts root,
the staged PrismaQuant tree and the cells plan.

The `f2fp4` and `f2ctl` tables were emitted through symlink subsets at
`$NR/views/`; the receipts themselves are the canonical
`$NR/native-f2/cells/<unit>__<format>/` directories. No emitted table, curve or
digest records a path under `$NR/views/` — the views are a re-derivable
convenience for re-running the emitter, not evidence.

| Object | sha256 |
|---|---|
| fp4 arm summary `$NR/stage/evidence/fp4-arm-summary.json` | `0f149eb81b9d4b31989b784924a088f4ea356f5766e7bafca000e8309a124d08` |
| relation `relation-f2.json` (9 cells, incomplete) | `1df655837523a0f8865871122e7a81a74e83084c012a6223a5307ec0f08864fa` |
| curve `prefill-accuracy-qwen3-0.6b-f2ctl.json` | `01425b9b3c395c2747ae684822fd9f2ec278f131e2e108579c1db59035eeeb54` |
| calibration safetensors (copied from `$R/inputs/`) | `20b2cc8f717bdc4b9efdd49fb2567cbb64f019c2e2c9abdb4b47c7eb3e2928f7` |
| cells plan `cells-fp4.json` (56 cells) | `f5c61432a23a4adb2c37b3d03f35df53b75bf0f0174e5e5b4dbfc354fbdb405f` |
| nvfp4 JIT extension `tessera_nvfp4_84439e84….so` | `7bfe7714dba8f16b03724772d2cad0449684f62082accb7c5c36a484156cc9d2` |
| driver manifest `$NR/stage/control/SHA256SUMS.txt` | `ec5838e2e566a2f331262824c7f5bcf97dd560a9e5995883cbba1d4d4752f31e` |
| evidence manifest `$NR/stage/evidence/SHA256SUMS.txt` (39 entries, all verify) | `00ea5993e797ad91054c50374a43361ccc98fd0dd8547e39ae00fc6f485ac8c9` |

**Method note.** A `fable-high` consultation was attempted on the framing of the
activation-side disagreement in §8.3 and returned HTTP 429, "You've reached your
Fable limit." No consultant answer informed any conclusion above.
