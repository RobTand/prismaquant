# PACT transient charge boundary: a versioned model so a real runtime table can reach the allocator

Status: design proposal for debt D37, 2026-09-18. **No gate, default, stage,
format, lane, pin or ship gate changes in this document or its PR.** It
records three decisions Rob can accept or amend, the code delta that follows
from each, and the tests that pin it. Every line reference was re-read on
2026-09-18 against `origin/main` at `d6c336d346`.

Related: debt D37 and D39 in `docs/ARCHITECTURE.md` §12;
RobTand/prismaquant#570 (D39 leg b), #559, #716; RobTand/tessera#399 and
#400; `docs/design/runtime_fixed_resource_admission.md` (the 2026-09-08
producer contract this document extends, not replaces).

## 0. One page for Rob

The allocator cannot draw the PACT curve for GLM-5.3-Flash because
`runtime_provenance._fixed_resource_refusals` refuses every v2 runtime table
twice: once unconditionally, because nobody has written down what a native
operator row and a full-engine capture each own
(`prismaquant/runtime_provenance.py:922-923`), and once whenever a unit prices
more than one format (`:799-805`), which a PACT table does by definition. Both
refusals are correct as written. This document proposes what replaces them.

**Decision 1 — version the boundary as a pairing of two producer-declared
rules, and define "equal" as ownership, not arithmetic.** A boundary version
(`prismaquant.transient_charge_boundary.v1`) is a PrismaQuant registry entry
that names the native bound's own `composition` string
(`sum_of_independent_peaks_including_output`) and the full-engine partition
schema (`tessera.full_engine_resource_partition.v1`), and maps each transient
term to one owner. The table's context and the report both carry the name.
"Equal" under v1 is a membership identity on the one measured assignment:
every allocation made inside a unit's interval belongs to that unit, every
allocation that belongs to a unit was made inside its interval, a unit's
escaping bytes are exactly its returned output, and the engine's simultaneous
peak for that unit never exceeds the row's bound. It never compares a native
peak to a partition term by value, because the two are differently bounded on
four counts (§1) and a numeric equality would refuse every real report or agree
by coincidence.

**Decision 2 — bytes belong to whoever allocated them, inside or outside a
unit interval.** The native row owns its invocation-local scratch and its
returned output, both of which its `peak_scratch_bytes` already covers. The
full-engine side owns the unit's input (allocated by the non-Linear op before
it), every fixed activation and scratch buffer, KV, and the off-step peak. The
consequence the allocator has to honor: a row's `activation_bytes` is the
logical input the row read, which the fixed terms already hold, so under v1 the
DP charges rows for `resident_bytes` and `peak_scratch_bytes` only and reads
`activation_bytes` as a shape witness. Lazily allocated runtime workspace is
the one class the rule exposes rather than hides: it surfaces as a
candidate-owned resident allocation no row prices, and refuses by name until
the scalar row carries the workspace identity the per-rank rows already carry.

**Decision 3 — one fixed charge, scoped to the route-class set one
mixed-artifact run exercised, with the invariance witnessed rather than
assumed.** Fixed ops allocate by shape, not by a neighbour's format. The real
non-invariance is KV: vLLM sizes the cache from free memory after load unless
the configuration pins it, so `fixed_kv` depends on the assignment. Under v1 a
configuration that does not pin KV capacity refuses by name; with it pinned,
the fixed charge measured on the one joint full-engine run that #570's option
A already requires (a mixed artifact spanning every attested route class) is
admitted for every assignment whose formats lie in the route classes that run
exercised. The fixed charge stays one constant added once, so
`allocator_solver.solve_runtime_frontier` stays a multi-choice knapsack. The
`:799-805` refusal becomes a route-class coverage check. The honest solver for
the rejected per-route-class alternative is written down in §4 so nobody
rebuilds it by accident.

**What it costs.** One full-engine run per (model, configuration) on a mixed
artifact, in both `resources` and `timings` observation modes, plus the #399
derivation layer in Tessera (§5). No new GPU measurement is claimed here.

**What it refuses.** A table or report with no boundary, or with different
boundaries; a report whose partition classifies a fixed-owned allocation inside
a unit interval or a candidate-owned one outside; a unit whose escaping bytes
are not its returned output; a route class the table prices that the
full-engine run never executed; a configuration whose KV capacity is derived
from free memory; a lazily allocated workspace no row prices; any timing or
serialized term the report does not observe.

**The one question only you can answer.** On unified memory the box holds the
caching allocator's *reserved* segments; the engine needs its *allocated* peak
plus fragmentation it cannot reclaim, and the allocator releases cached
segments only on its own failed `cudaMalloc` retry. Does
`--serve-device-budget-bytes` mean "this serve fits" (charge allocated) or
"this serve coexists with a neighbour on the same pool" (charge reserved)?
v1 as written observes and publishes reservation slack and charges neither
answer, because choosing one silently is the default principle 1 forbids.
Everything else in this document has a derivation; this one has a policy.

## 1. The quantities, exactly

Two producers measure transient bytes, over different intervals, with
different bounds. This section says what each number is, cited to the code
that produces it.

### 1.1 A native operator row

One row prices one `(unit, format)` from a Tessera native receipt
(`prismaquant/native_receipt_table.py:386-407`). Its transient fields:

| Field | What it is | Interval | Bound |
|---|---|---|---|
| `peak_scratch_bytes` | `max` over the prefill and decode phases of the receipt's `peak_scratch_bytes` (`:391, :407`). Each phase value is `external_native_peak_bytes + torch_peak_increment_bytes` and the receipt's bound must declare `composition == "sum_of_independent_peaks_including_output"` (`prismaquant/native_operator_panel.py:498-507`). | One `apply` call, after warmup. | An upper bound: the sum of two peaks that need not coincide. |
| `activation_bytes` | `max` over phases of the receipt's `input_bytes`, which is the logical byte size of the input tensor (`native_operator_panel.py:412`, `native_receipt_table.py:392`). | Not measured. A shape. | Exact for the input the row read; says nothing about what else was live. |
| `resident_bytes` | The observation's `resident_bytes` (`native_receipt_table.py:406`), the prepared weights under `resident` serve mode. | Load time. | Exact. |
| `kv_bytes` | `0` (`:407`). | — | A native row sees no cache. |

The torch increment is `torch.cuda.max_memory_allocated() - before` after
`reset_peak_memory_stats()` around the apply (Tessera pinned tree
`experiments/bench_native_operator.py:494-498`). `max_memory_allocated`
returns the `allocated_bytes.all.peak` statistic (installed torch 2.11.0
`torch/cuda/memory.py:548`), which PyTorch documents against the separate
`requested_bytes` statistic that excludes "allocation rounding" (`:297-299`).
So a native row's scratch charge counts **allocator block bytes, including
rounding**, and it counts the **returned output**, which the apply allocates
and which is live when the apply returns.

### 1.2 The full-engine partition

One capture prices one complete assignment (one row per unit, `_fixed_resource_refusals`
`runtime_provenance.py:811-838`) under one configuration. The consumer
`prismaquant/full_engine_resource_report.py` recomputes seven terms from the
partition's membership rows (`:132-133`): `fixed_resident`, `candidate_resident`,
`fixed_activation`, `candidate_activation`, `fixed_scratch`, `candidate_scratch`,
`fixed_kv`, plus the off-step peak `non_step_transient_peak_bytes` (`:190`).
Each allocation gets one `(owner_class, lifetime_class, unit)` from `_classify`
(`:827-870`):

- Lifetime first. Never freed within the capture: `resident`. Freed inside its
  own unit interval (`lifetime_scope == "inside_unit"`): `scratch`. Allocated
  inside a unit interval and freed after it (`escapes_unit`): `activation`.
  Freed outside every unit interval: decided by liveness against the declared
  engine steps, and `non_step` when live during none of them.
- Ownership second. One supported class from `("fixed", "candidate", "kv")`
  (`:164`); `shared` and `unknown` classify nothing (`:167`).

The producer derives `lifetime_scope` from the unit intervals it recorded
(Tessera `experiments/full_engine_resources.py:985-990`). Its per-allocation
`bytes` is the requested size from the allocator history (`:941-950`), and
the ledger says so about itself: `torch_observed_live_peak_scope =
"requested_allocation_bytes_excluding_allocator_rounding"` (`:1003`). The
rounded block size is recorded beside it per allocation as
`allocator_block_bytes_observed` (`:539-544`), so rounding is observable, not
charged. Every transient term is a **simultaneous** maximum over its interval
(`docs/design/runtime_fixed_resource_admission.md:291-295`, rule 7).

### 1.3 Where the two are differently bounded

The `FIXED_TERM_FIELDS` docstring (`runtime_provenance.py:461-466`) says the
two sides are "differently bounded". These are the terms:

1. **The returned output.** The native row counts it inside `peak_scratch_bytes`.
   The partition allocates it inside the unit interval and frees it after the
   consumer reads it, so it is `escapes_unit` and lands in
   `candidate_activation[unit]`, not `candidate_scratch[unit]`.
2. **Allocator rounding.** Inside the native charge (`allocated_bytes`),
   outside the full-engine charge (requested bytes, `:1003`).
3. **The bound itself.** The native row is a sum of two independent peaks; the
   partition is one simultaneous maximum. The native number is never smaller
   than what the engine did for that unit; it may be larger.
4. **The word "activation".** The row's `activation_bytes` is the logical
   *input* the row read. The partition's `candidate_activation[unit]` is the
   *output* that escaped. For unit `u` the two name different tensors, and they
   agree only when the input and output happen to have the same byte size.

A numeric equality on any of these four would refuse every honest report
(terms 1, 2, 3) or pass on a coincidence (term 4). That is the sentence the
docstring compresses, and it is why the boundary is a model, not a check.

## 2. Decision 1: the versioned boundary model

### 2.1 Where a version can live under principle 14

PrismaQuant may not assert how the serving runtime classified its bytes.
Both producers already freeze their own rule as data: the native bound's
`composition` string (`native_operator_panel.py:500`, a producer field the
consumer compares verbatim) and the partition's schema name
(`full_engine_resource_report.py:39`, `tessera.full_engine_resource_partition.v1`).
A boundary version is therefore a PrismaQuant registry entry that **pairs**
those two producer-declared rules with an ownership map. PrismaQuant asserts
nothing about the runtime; it asserts which pair of published rules its map is
defined over, and refuses any table or report that names a different pair.

### 2.2 Schema

Name: `prismaquant.transient_charge_boundary.v1`. The `prismaquant.` prefix
follows `prismaquant.measured_runtime_prices.v2`,
`prismaquant.runtime_rank_totals.v1` and `prismaquant.native_dense_observation.v1`;
the noun is the docstring's own phrase.

```json
{
  "schema": "prismaquant.transient_charge_boundary.v1",
  "native_row": {
    "bound_composition": "sum_of_independent_peaks_including_output",
    "byte_extent": "allocated_bytes_including_allocator_rounding",
    "interval": "single_apply",
    "activation_bytes_is": "logical_input_shape_witness"
  },
  "full_engine": {
    "partition_schema": "tessera.full_engine_resource_partition.v1",
    "byte_extent": "requested_allocation_bytes_excluding_allocator_rounding",
    "returned_output_lifetime_class": "activation",
    "step_coverage_required": "complete"
  },
  "ownership": {
    "unit_scratch": "native_row",
    "returned_output": "native_row",
    "unit_input": "full_engine",
    "fixed_activation": "full_engine",
    "fixed_scratch": "full_engine",
    "kv": "full_engine",
    "non_step_peak": "full_engine",
    "runtime_workspace": "row_identity_once_per_rank",
    "reservation_slack": "observed_not_charged"
  },
  "row_terms_charged": ["resident_bytes", "peak_scratch_bytes"],
  "row_terms_witnessed": ["activation_bytes"],
  "fixed_terms_charged": ["fixed_resident", "fixed_activation", "fixed_scratch",
                          "fixed_kv", "non_step_transient_peak_bytes"],
  "invariance": "route_class_set_of_one_full_engine_run"
}
```

This object is code, not a file: a frozen registry in a new module
`prismaquant/transient_charge_boundary.py` (§6). The table's `RuntimeContext`
carries the name as `transient_charge_boundary`; the report carries the same
name at `reference.transient_charge_boundary`, stamped by the #399 derivation
layer (§5). Both are strings a gate compares verbatim.

### 2.3 What "equal" means under v1

Equality is an identity of ownership on the one measured assignment, checked
from the partition's membership rows and the native observation, never from a
`derived` block:

1. **Containment.** Every allocation whose `allocate_index` lies inside unit
   `u`'s interval has `owner_class == "candidate"` and `unit == u`. A
   fixed-owned or KV-owned allocation inside a unit interval refuses, naming
   the unit and the allocation id.
2. **Attribution.** Every allocation with `owner_class == "candidate"` and
   `unit == u` was allocated inside `u`'s interval. A candidate-owned
   allocation outside its unit's interval refuses by name.
3. **Escape.** The `escapes_unit` set for `u` is exactly one allocation whose
   requested `bytes` equals the observation's `output_bytes`
   (`native_operator_panel.py:413`). Two escaping allocations, or one of another
   size, refuse naming the unit and the bytes.
4. **Cover.** The simultaneous maximum over `u`'s interval of candidate-`u`
   allocations, taken over rounded `allocator_block_bytes_observed` so both
   sides count the same extent, does not exceed the selected row's
   `peak_scratch_bytes`. A row that the engine exceeded refuses; a row that
   exceeds the engine is disclosed conservatism, which rule 7 already accepts.
5. **Residency.** Every candidate-`u` allocation with lifetime `resident`
   equals the row's `resident_bytes` after the partition already charges it as
   `candidate_resident` (`runtime_provenance.py:900-921`, unchanged). A second
   resident candidate allocation the row does not price is the workspace case
   (§3.4) and refuses by name.

Checks 1 and 2 are the witness that the ownership map is true of this
runtime; 3 and 4 are the witness that the native bound covers what the engine
did; 5 is already there. None of them compares a native peak to a partition
term by value.

### 2.4 Alternatives rejected

- **Numeric equality of terms** (`candidate_scratch[u] == row.peak_scratch_bytes`,
  `candidate_activation[u] == row.activation_bytes`). Refuses every real report
  on terms 1 to 3 of §1.3 and passes term 4 by coincidence. This is the
  alternative the docstring rejects.
- **Subtract the output from the native peak.** Forbidden by the existing
  producer contract: "the output need not be live at the peak"
  (`runtime_fixed_resource_admission.md:365-367`).
- **Re-measure every row inside the engine and drop the native table.** One
  full-engine run observes one assignment. A PACT table prices three route
  classes for each of `N` units; the engine would need `3^N` runs.
- **A tolerance band on any term.** Principle 2: the only admissible constants
  are dtype precisions, and no dtype relates a sum of peaks to a simultaneous
  maximum.

## 3. Decision 2: ownership per transient term

The rule is one sentence: **a byte belongs to the operator that allocated it,
and a native row is the operator for everything allocated inside its apply
interval.** Everything else follows from what the engine holds during the unit
interval.

### 3.1 Shared activations (the unit's input)

During `u`'s interval the engine holds `u`'s input. In every architecture
Tessera serves, that input is produced by a non-Linear op: layer norm for
`q/k/v_proj` and `gate/up_proj`, attention for `o_proj`, the gated product for
`down_proj`, the router dispatch for a routed expert. That op is fixed-owned,
its output is allocated outside any unit interval, and its byte size does not
change with `u`'s format. So the input is **full-engine-owned**, and the
partition already charges it: as `fixed_scratch` when it is freed inside the
step, as `fixed_activation` when it crosses a step boundary. The row's
`activation_bytes` records the same tensor's logical size; charging it again
from the row would count the same physical extent twice, which rule 7 forbids
in the other direction and which is equally wrong in this one. Under v1 the DP
reads `activation_bytes` as a witness and adds nothing for it (§6.3).

A future architecture that feeds one Linear straight into another does not
break the rule: the first unit allocated the buffer, the first row's
`peak_scratch_bytes` includes it as returned output, and the second row reads
it as input without charging it. Check 3 in §2.3 sees the buffer escape the
first unit; check 1 sees no allocation inside the second unit's interval for
it. Nothing is assumed about the architecture; the identity checks decide.

### 3.2 Returned outputs

Allocated inside the apply, live when the apply returns, consumed by the next
fixed op. **Native-row-owned**, and already inside `peak_scratch_bytes` by the
bound's own composition. The partition classifies the same bytes as
`candidate_activation[u]`; under v1 that term is a witness (check 3), not a
charge. This is the one place the two producers put one tensor under two
names, and the boundary resolves it by owner, not by renaming either side.

### 3.3 Scratch

Allocated and freed inside the apply: the fp8 route's quantized input copy,
the fp4 route's epilogue temporary (D40), a GEMM workspace the route requests
per call. **Native-row-owned**, inside `peak_scratch_bytes`. The partition's
`candidate_scratch[u]` is the cover witness (check 4).

### 3.4 Runtime workspace

A process-global allocation made lazily on first use: the `vllm.WorkspaceManager`
buffer, or device state a JIT-built route extension allocates on its first
apply. In the full-engine capture it is allocated inside the first unit's
interval that touched it and never freed, so check 1 attributes it to that
unit and check 5 finds a resident candidate allocation the row does not
price. **That refusal is correct**: the scalar row has no field for it. The
per-rank rows already solve this with `workspace_resident_bytes` and
`workspace_sha256`, charged once per identity per rank
(`measured_runtime_prices.py` `RANK_WORKSPACE_RULE`,
`allocator_solver.py:168-180`). v1 owns it as `row_identity_once_per_rank`
and the scalar row gains the same two fields when the receipt carries them
(§6.2). Until then the refusal names the allocation and the unit, which is the
signal that the receipt is owed a field, not that the gate is wrong.

### 3.5 Allocator rounding

Inside every allocation's block. Owned by **whoever owns the allocation**.
The native side already counts it; the full-engine side observes it per
allocation (`allocator_block_bytes_observed`). Under v1 the cover check (§2.3
item 4) is taken over rounded extents on both sides, and the fixed terms are
recomputed over rounded extents for the same reason: the box holds the block,
not the request. That is a v1 requirement on the derivation layer (§5.2), not a
change to the partition schema's `bytes` field.

### 3.6 KV

**Full-engine-owned**, as today (`fixed_kv`, domain `cache_capacity`). What v1
adds is the invariance condition in §4.

### 3.7 Reservation slack

Segments the caching allocator holds with no live block. Owned by nobody in
the composition. v1 requires the report to observe it (the segment events at
Tessera `full_engine_resources.py:931` already exist) and publish it as
`reservation_slack_peak_bytes`, and charges it to neither the fit nor the
coexistence obligation until Rob answers the question in §0. A report that
does not observe it refuses by name, so the number is always there for the
day it is charged.

### 3.8 What the allocator charges under v1

For a candidate assignment `f` over units `U`, with the admitted fixed charge
`F` from the report:

```
step_bytes(f)   = F.fixed_resident + F.fixed_activation + F.fixed_scratch + F.fixed_kv
                + sum_u resident_bytes[u, f_u]
                + max_u peak_scratch_bytes[u, f_u]
                + workspace charged once per identity per rank
placement(f)    = max(step_bytes(f), F.non_step_transient_peak_bytes)
```

This is `allocator_solver._placement_bytes` (`:274-286`) with the activation
axis removed from the row sum, because the input is in `F`. It is an upper
bound on what the engine holds at any instant inside a unit interval: the
fixed terms are step-wide maxima of fixed-owned bytes and so cover the input
at any instant; the row's bound covers everything the unit allocated. Outside
unit intervals the escaped output is still inside `max_u peak_scratch_bytes`.
Conservative where the maxima do not coincide, and disclosed as such.

## 4. Decision 3: the fixed charge under a multi-format menu

### 4.1 What is and is not invariant

The `:799-805` refusal says one measured assignment establishes no invariant
fixed charge under the others. Term by term:

- `fixed_resident`: engine weights outside the candidate roster, the loaded
  libraries' device state, KV metadata. Invariant across assignments whose
  route classes were all loaded in the run; not invariant if a route class
  the run never loaded brings its own extension (D39 leg b, #570).
- `fixed_activation`, `fixed_scratch`: allocated by fixed ops, sized by
  `(M, hidden, heads, vocab)` and the kernels those ops choose. A neighbour's
  weight format does not enter those shapes. Witnessed, not assumed: check 1
  of §2.3 proves no fixed-owned allocation happened inside any unit interval
  in the run, and the D39 residual (every priced family present in the served
  manifest) proves every route class ran in the same process.
- `fixed_kv`: **not invariant by default.** vLLM sizes the KV pool from the
  memory left after the model loads and the profile run peaks, so a smaller
  assignment gets a larger cache. It becomes invariant exactly when the
  configuration pins capacity (`num_gpu_blocks_override` or
  `kv_cache_memory_bytes`), which the report's `kv_observations` can be
  checked against (`num_blocks`, `group_page_size_bytes`,
  `full_engine_resource_report.py:126-128`). Rule 5 of the producer contract
  already requires the capacity policy to be bound; v1 makes the unpinned case
  a named refusal.
- `non_step_transient_peak_bytes`: startup and between-step peaks of fixed
  ops. Invariant under the same witness as `fixed_scratch`.

### 4.2 Options

**(a) Per-format fixed charges, one full-engine run per route class.** Three
uniform artifacts, three runs, three fixed charges. A mixed assignment is
still unmeasured, and the fixed charge under a mixed run is not the maximum
of the uniform ones without an assumption about the runtime. Measuring every
subset needs `2^3 - 1 = 7` runs. And a per-route-class fixed charge makes the
fixed term a function of the assignment, which changes the solver (§4.4).

**(b) A fixed charge scoped to the selected assignment, re-evaluated inside
the DP.** The DP explores `3^N` assignments and the fixed charge is known for
the measured ones only, so "re-evaluated inside the DP" means the DP proposes
and a full-engine run per frontier point decides. That is the KL selection
spine applied to bytes, and it is the right *ship gate* for the chosen point;
it cannot price the search, because the search has no number to prune on.

**(c) Bound plus residual.** A measured fixed charge plus a margin for what
other assignments might add. There is no published table to derive the margin
from, so it is an invented constant. Rejected on principle 2.

**(d) One fixed charge scoped to the route-class set of one mixed-artifact
run, with a witnessed invariance rule.** The run #570's option A already
requires (one engine process, one configuration, a served artifact whose
priced units span every attested route class) gives a fixed charge measured
with every route class loaded and executing. Checks 1 and 2 of §2.3 witness
that fixed ops allocated nothing inside unit intervals under that mix, the
D39 residual witnesses every family ran, and a pinned KV capacity makes
`fixed_kv` a configuration constant. The fixed charge is then admitted for
every assignment over the same roster whose formats lie in that route-class
set.

### 4.3 Recommendation

**(d)**, with (b) kept as the ship gate it already is (the full-engine run of
the exported point is what `validate_native_export` needs anyway). The
invariance rule, stated so a gate can read it:

> The fixed charge measured on assignment `A` under configuration `C` is
> admitted for assignment `B` when: `B` prices the same unit roster; every
> `(unit, format)` in `B` has a `binding.operator_route` (the #565 route-class
> spelling) that some row of `A` also has; `C` pins KV capacity; and the
> report's identity checks 1 and 2 passed.

What it refuses: a format whose route class `A` never exercised (named with
its units), an unpinned KV policy, and any report where a fixed op allocated
inside a unit interval. Nothing here is a constant.

### 4.4 What the DP has to do

Under (d) the fixed charge is one number added once, so
`solve_runtime_frontier` stays a multi-choice knapsack over `(unit, format)`
with additive `resident_bytes` and a maximum over `peak_scratch_bytes`; the
change is the dropped activation axis (§3.8), not the solver's shape.

Under (a), for the record: the fixed charge becomes `F(S)` where `S` is the
set of route classes the assignment uses. That breaks additivity, and the
honest solver is the same DP with its state augmented by `S`, which has
`2^3 = 8` values for three route classes, so the table grows eightfold and the
fixed charge is added at the end per `S`. Cheap to run, and useless until
`F(S)` is measured for the mixed subsets, which is what (d) measures once for
the full set. Nobody should build it before that measurement exists.

## 5. The #399 derivation layer, as a plan

Tessera #400 merged the raw observer; #399 owns turning its capture into a
report a gate can read. Today the analysis hard-sets the answer:
`analyze_engine_resource_ledger` returns `status: incomplete`,
`admission: not_implemented`, `fixed_resources: None`, `timings: None`,
`full_model_fixed_resources_complete: False` (Tessera worktree
`experiments/full_engine_resources.py:789-797`), and the timing observer
emits `"timings": None, "admission": "not_implemented"`
(`experiments/full_engine_timings.py:287`). The report consumer on the
PrismaQuant side already refuses `prefill_ms`, `decode_ms` and
`serialized_bytes` by name because no schema version carries them
(`runtime_provenance.py:476, :924-931`).

### 5.1 The report the observer must emit

`tessera.full_engine_resource_report.v1`, extended in place (same schema
name; the consumer's `_OBSERVATION_FIELDS` and `_PARTITION_FIELDS` grow, so
an older report refuses on its missing fields by name rather than being read
as a newer one):

- `reference.transient_charge_boundary`: the boundary name the partition was
  classified against (§2.2).
- `partition.byte_extent`: `"allocator_block"` for every transient term, per
  §3.5, with requested bytes kept per membership row.
- `observations.timings`: per repeated sample and per phase, the ordered
  native apply intervals and the adjacent fixed gaps from the same CUDA-event
  chain the `timings` observation mode already records
  (`full_engine_timings.py:196-245`), with the stream-coverage checks that
  already refuse unjoined streams and overlapping copy-stream work
  (`:110-121`). The consumer recomputes `fixed_prefill_ms` and
  `fixed_decode_ms` as the median across complete samples of each sample's
  summed fixed gaps, per the producer contract
  (`runtime_fixed_resource_admission.md:379-401`).
- `partition.serialized`: exact file and tensor extents from the export proof
  #400 already binds (source census, member wires, checkpoint roster), so
  `fixed_serialized_bytes` is the non-candidate remainder of the checkpoint
  and `candidate + fixed + excluded == total` is recomputable (rule 6).
- `observations.reservation_slack`: the segment-level peak from the segment
  events (`full_engine_resources.py:931`), published, not charged (§3.7).

`FIXED_TERM_FIELDS` then gains `fixed_prefill_ms -> prefill_ms`,
`fixed_decode_ms -> decode_ms`, `fixed_serialized -> serialized_bytes`, and
`UNOBSERVED_FIXED_FIELDS` empties. Each still refuses by name when its
observation is null.

### 5.2 How the owner rule closes the a5 gaps

The #399 a5 run left 20,734 + 370 allocations unclassified, 224 classified but
uncharged, and six domains open (D37 row), and #400 records why: "the supplied
source-BF16 FlashInfer owner rule requires a library absent from this native
run". Ownership was being assigned per library, by observers that know one
library each, and the rows no observer claimed stayed `shared` or `unknown`.

Under v1 ownership is positional. An allocation made inside a unit interval is
the unit's; one made outside is the engine's; a KV backing is `kv` by the
capacity assertion the capture already checks. The per-library observers
(the BLAS workspace observer at `full_engine_resources.py:223-240`, the
FlashInfer rule) stop being classifiers and become witnesses: they confirm
that a fixed-owned allocation is the workspace it claims to be, and a
witness for a library the run never loaded is a null witness, not a gap.
The unclassified rows are, by the D37 row's own description, allocations
outside unit intervals with no owner category; positional ownership gives
them `fixed`, and lifetime gives them `scratch`, `activation`, `resident` or
`non_step` from the step intervals `_classify` already reads. `shared` stops
being a label the partition can emit. What positional ownership cannot do is
decide a row whose interval evidence is missing, and those still refuse.

### 5.3 From hard-sets to an admission path

Replace the constants at `full_engine_resources.py:789-797` with the
derivation: domain states from the ledger (the `derived` spelling the consumer
insists on, `full_engine_resource_report.py:198-199`), `fixed_resources` as
the simultaneous maxima over classified lifetimes, `admission` as the list of
refusals the producer found in its own capture. Replace `timings: None` at
`full_engine_timings.py:287` with the per-sample gap partition. Keep the rule
that a producer number is a claim the consumer recomputes and never reads.

### 5.4 Who owns which piece

| Piece | Repo |
|---|---|
| Observer, capture, ledger, positional classification, timing gaps, serialized extents, boundary stamp, report emission | Tessera (#399) |
| Boundary registry, identity checks 1 to 5, gate delta, route-class coverage, solver composition flag, context field, emitter exit 0 | PrismaQuant (this design) |
| The mixed artifact, its configuration document, the 21-cell re-freeze | #570 option A, unchanged |

## 6. The code delta that follows

Nothing below is in this PR. It is what the decisions imply, named so the
implementing PR can be reviewed against this document.

### 6.1 New module: `prismaquant/transient_charge_boundary.py`

- `BOUNDARY_V1 = "prismaquant.transient_charge_boundary.v1"`.
- `BOUNDARIES: Mapping[str, BoundarySpec]`, a frozen registry holding the
  object in §2.2.
- `require_boundary(context, report) -> BoundarySpec`: raises
  `RuntimePriceError` naming whichever of the four conditions failed — the
  table declares none, the report declares none, the two differ, or the name
  is not registered.
- `boundary_identity_refusals(spec, report, observation, selected_rows) ->
  list[str]`: checks 1 to 5 of §2.3, each refusal naming the unit and the
  allocation or bytes it is about.
- `route_class_coverage_refusals(spec, table, selected_rows) -> list[str]`:
  §4.3, naming each unpriced route class and its units, and the KV policy
  when the configuration does not pin capacity.

### 6.2 Table and row schema

- `RuntimeContext` gains `transient_charge_boundary: str | None = None`
  (`measured_runtime_prices.py:177`). It appears in `as_dict` only when set,
  the pattern `RuntimeResources.as_dict` uses for `OFF_STEP_FIELD`
  (`:292-297`), so every table emitted before the field re-emits
  byte-identically and keeps its digest.
- `native_receipt_table.derive_context` stamps `BOUNDARY_V1` when every bound
  receipt's `composition` is the one v1 names, and refuses otherwise.
- The scalar `RuntimeResources` gains optional `workspace_resident_bytes` and
  `workspace_sha256` on the same appear-only-when-set pattern, read from the
  receipt when Tessera records them, so §3.4 has a field to land in.

### 6.3 Gate delta in `runtime_provenance._fixed_resource_refusals`

- `:922-923` becomes `refusals.extend(boundary_identity_refusals(...))` after
  `spec = require_boundary(context, report)`; a missing boundary is one named
  refusal instead of the unconditional sentence.
- `:799-805` becomes `refusals.extend(route_class_coverage_refusals(...))`.
  A table with no boundary still refuses on the multi-format menu, through
  `require_boundary`, so no path admits a multi-format table without v1.
- `:924-931` stays as written; the three terms pass once the report carries
  them (§5.1) and `FIXED_TERM_FIELDS` names them.
- `admitted_fixed_resources` (`measured_runtime_prices.py:806`) returns the
  spec beside the resources so the caller can hand it to the solver.

### 6.4 Solver

`solve_runtime_frontier` (`allocator_solver.py:289`) takes `boundary:
BoundarySpec | None`. When `fixed_device_bytes` or `fixed_non_step_peak_bytes`
is supplied and `boundary` is `None`, it raises: a fixed charge with no
boundary has no composition. Under v1 the row axis for `activation_bytes` is
not added into `_placement_bytes` (`:274-286`); the axis stays in the frontier
vector and in `dimensions` so the diagnostics still show it. No default value
for `boundary` makes anything pass.

### 6.5 Emitter

`native_receipt_table` exit `0` becomes reachable; the docstring at `:26-32`
that says it is unreachable while D37 stands is updated in the same commit.

### 6.6 Tests, mutating the driver

New, in `tests/test_transient_charge_boundary.py` and additions to
`tests/test_runtime_fixed_resource_admission.py`:

1. A table declaring v1 and a report declaring v1 whose partition satisfies
   checks 1 to 5 produces no boundary refusal; the old unconditional sentence
   is absent from the refusal list.
2. The same table with `transient_charge_boundary` set to another string
   refuses naming both the table's and the report's value.
3. The same pair with the report's `observations.timings` set to `null`
   refuses with `prefill_ms` and `decode_ms` named, and with
   `partition.serialized` absent refuses naming `serialized_bytes`.
4. A PACT table pricing three route classes for every unit, with a report
   whose `selected_rows` cover all three classes and a configuration that
   pins KV capacity, is admitted; the same table with a report covering two
   classes refuses naming the third class and its units; the same table with
   no boundary refuses at `require_boundary`.
5. A report with one fixed-owned allocation inside a unit interval refuses
   naming the unit and allocation id; a report whose escaping set for a unit
   is two allocations, or one of the wrong size, refuses naming the bytes;
   a resident candidate allocation the row does not price refuses naming
   the workspace.
6. `solve_runtime_frontier` with a fixed charge and no boundary raises;
   under v1 the activation axis does not enter `device_bytes`, checked on a
   two-unit fixture where the two compositions differ.

Existing tests that must keep passing: `tests/test_runtime_provenance.py`,
`tests/test_full_engine_resource_report.py`,
`tests/test_measured_runtime_prices.py`, `tests/test_native_receipt_table.py`,
`tests/test_native_receipt_table_routed.py`,
`tests/test_pq267_native_receipt_qualification.py`,
`tests/test_allocator_runtime_frontier.py`,
`tests/test_allocator_measured_runtime_cli.py`, `tests/test_allocator_solver_bins.py`.

Two existing assertions pin the exact refusal strings this delta replaces and
change with it, deliberately: `tests/test_runtime_fixed_resource_admission.py:262`
(the multi-format sentence) and `:391` (the boundary sentence), and
`tests/test_prefill_frontier_shape_only_scope.py:43` (the same boundary
sentence). The implementing PR updates them to the named refusals and says so.

## 7. What this does not do

- It does not price the fp4 route class. The 2026-09-18 `TESSERA_E2M1_K2`
  cells are held by #715 (the panel stamp lacks the attested image scope) and
  #717 (three of seven cells refuse on a one-code flip).
- It does not stamp the executing image. #716 closed the binder for the
  native-cell driver; the production paths still owe
  `image_content_sha256`.
- It does not touch the sealed GLM-5.3-Flash campaign, its pin, its
  configuration digest or any PrismaBuild action.
- It does not weaken the Tessera menu gate. #572's residual stands: the
  pinned contract attests one rung per family, so the DP's second axis is the
  three route classes, not a rate sweep.
- It does not change `admit_fixed_resources`, `build_runtime_resources`,
  `solve_runtime_frontier` or any test. The delta in §6 is a plan.
- It does not decide the reservation-slack question in §0.
