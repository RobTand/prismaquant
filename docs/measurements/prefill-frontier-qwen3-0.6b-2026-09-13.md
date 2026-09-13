# First measured prefill x accuracy curve — Qwen3-0.6B, Tessera layer 0

Status: **measurement record.** 2026-09-13, branch
`claude/first-prefill-frontier-qwen3-0.6b`, PQ #237.

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
assign them: 49 cells. Prefill is a `[512, 1024] x [1024, N]` apply; decode is
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
