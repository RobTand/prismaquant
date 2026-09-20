# The distributed joint-AURA cost campaign: per-layer quanta on both Sparks (#778)

2026-09-19. Design and contracts, not an implementation. Rob's committed
direction this night: the complete-512 cost run's work sharded across **both**
DGX Sparks as PrismaBuild quanta, with the tier machinery feeding each box,
replacing the single-consumer sealed-run architecture that kept one GPU dark
all night. Success metrics, Rob's words: *"the utilization of the ramdisk and
the utilization of the GPU."*

Status: design; no code in this change. Builders implement against the
contracts in §3–§9; §10 names what is deliberately deferred and which pieces
belong to PrismaBuild's repository, not ours. This note extends the
verification-quanta pattern of #765 (named and specified in #768's
`joint_head_walk_quanta`) from the head walk to capture/cost work. Nothing
here builds an application-side dispatcher: PrismaBuild owns sharding,
placement, admission and retry (AGENTS.md granularity policy; PB #192); the
new tools only *publish* rows and *join* receipts.

## 1. Why: tonight's arithmetic

Every number in this section is read off live records on 2026-09-19.

**One consumer cannot fill one GPU, let alone two.** The live run (PB action
`e28b8c64b435dec6ae771bc8b031d9fcf43f32165900c4b9d47a0206db7b0437`, claimed
on `sparky` at 03:26:26Z) reports accepted progress `layer-9` at 04:09:42Z:
its read frontier had entered layer 9 of 46 phases, so it had read at least
head (10.15 GiB) plus layers 0–8 (3.07 GiB × 3, 133.72 GiB, 122.25 GiB,
81.74 GiB × 4) — ≥ 521 GiB of the sealed 3,960 GiB read set in 43 minutes,
**≈213 MB/s sustained** (217 MB/s at that interval's floor, at most 250 MB/s
if layer 9 was nearly done). The
Sparks' link to the storage host is 100 Gbps RDMA (12.5 GB/s). One consumer
used ~1.7% of the wire and 50% of the fleet's GPUs; `sparklina` (3 worker
loops, same GB10 class) had no campaign work at all.

**The read set is renders, and renders are per-layer.** The sealed run data
manifest (`sha256 4100221caa…`, 46 phases, 234,591 entries) decomposes as:
**renders 3,100.4 GiB** over 197,990 files, source extents 574.7 GiB over 161
coalesced ranges, head inputs 7.4 GiB, resume journals 0.36 GiB. Per-layer
phases: layer-3 133.72 GiB, layer-4 122.25 GiB, layers 5–44 81.74 GiB each,
layers 0–2 3.07 GiB each. A layer's phase is that layer's work and nobody
else's — the per-Linear candidate set (36,423 units; 867 per layer for
layers 3–44 — the layer's projections across its routed experts — and 3
each for layers 0–2; 132 campaign groups; 360 retained windows at the sealed
`hostcap32` plan) partitions exactly along the same boundaries.

**Phase-granular promotion cannot stream a layer.** PB #671 (verified live
2026-09-19): the tier loop's own `ram-window-stalled` events showed
`runahead_budget_gib=0` with blocked phase `layer-3` (134 GiB) against
`free_gib` 112 — `runahead_budget_gib` is `max(0, capacity − step)` and the
step exceeds the window when a phase is larger than it. PB #672 raised the ram
window to 160 GiB: one whole 134-GiB phase plus margin, inside the #645/#665
worker-demand guard (160.5). That is a floor, not a fix: at phase granularity
the window's run-ahead budget is `160 − 134 = 26 GiB` once the consumer rolls,
which is less than one more phase — **run-ahead is zero phases for the rest of
the campaign**. Sub-phase promotion granularity is therefore a hard
requirement of this design (§5.3).

**The single consumer is also what kept tripping the fill ledger.** PB #654: a
mover that *adopts* resident ranges barely touches the pool, and the fill fold
read its pacing as a pool ceiling — deadlock by construction. PB #669 (act
three, tonight): a non-reader record rebuilt fill best from nothing, the stage
minted 1 MB/s of fill, nothing promoted, the tmpfs stayed empty, and *"the
campaign's GPU starved between bursts while the RDMA wire sat idle."* Both
fences have landed in PB; the demand pattern that made them load-bearing —
one consumer, 45 promotions of 80–134 GiB each, adoption-heavy on resubmit —
is what this design replaces with many small consumers whose promotions are
chunk-sized and whose receipts are honest movers.

**What the night costs.** At 213 MB/s the read set alone is 5.2 serial hours on
one GPU while the second GPU, the second RDMA path, the stage tier's free
capacity (65.9 GiB announced at 04:10Z — the rest held by the single
campaign's staged phases) and the ram tier's 239 GiB all sit idle. The work
is independent; the architecture serialized it.

## 2. Campaign shape and the independence proof

### 2.1 What the run actually does

`prismaquant.tessera_joint_aura run` (the live action's payload path,
`compute_aura_cost_streamed` in `prismaquant/aura_cost.py`) executes three
movements over one calibration contract (512 samples × 512 tokens, 4 probes,
seed base 7000 — all sealed in the plan):

1. **Forward boundary capture** (layer-ascending, sequential): streams source
   weights layer by layer and writes, per (batch partition, layer), the exact
   input activation of that layer — `boundary-{batch}-{layer}` entries under
   the run's `exact-boundaries/<generation>/` (v2 layer-major boundary
   storage; measured tonight: 85 GiB at layer-9 ≈ 8.5 GiB/layer, budget
   446.7 GB in the sealed plan). Sequential by construction: layer L's input
   is layer L−1's output.
2. **Tail cotangents**: from the final boundary, per probe × batch, the
   adjoint seed — `cotangent-{probe}-{batch}` entries at boundary
   `num_layers`.
3. **Reverse walk** (layer-descending 44→0): for each layer, install the
   source layer, replay that layer's **retained windows' renders** (this is
   the 3.1-TB read), recompute the layer for every probe from its stored
   input boundary, and project the weight gradient onto each candidate's
   rendered delta — producing that layer's per-unit cost rows. The only state
   crossing a layer boundary in this walk is the **rolling cotangent**: the
   walk reads `(boundary_cpu, incoming_cpu)` per batch (`prefetched_boundary_batches`),
   and writes the rolled cotangent for the next layer down, retiring the
   previous (`write(…, boundary_index=layer, probe_index, previous=…)`).

So the reverse walk's layer step is already layer-local in code. The one
cross-layer dependency is the incoming cotangent at boundary L+1.

### 2.2 The decomposition

The distributed campaign splits the run into three stages with three kinds of
PB work. Nothing else changes: same plan, same prepared completion, same
calibration, same seeds, same renders, same kernels.

**Stage A — adjoint capture (one sequential consumer).** The forward boundary
capture (1) and tail cotangents (2), plus a render-free cotangent chain: walk
the source-model backward 44→0 *without the window replay*, writing the
incoming cotangent at **strided** boundaries only (a checkpoint every S
layers, §3.4), with the full shared-pass adjoint state serialized beside the
activation cotangent. Reads: source extents only (~575 GiB over the whole
stage, tier-fed like today). Writes: boundary entries for all 45 layers
(~383 GiB), strided adjoint checkpoints, and a published receipt
(`adjoint-capture.json`, §3.3). GPU-bound: recompute + backward, no render
I/O. This stage is exactly tonight's run with the render replay deleted and
the cotangent retirement rule changed — a scoped mode, not a new engine.

**Stage B — per-layer cost quanta (45 independent consumers, both boxes).**
Quantum `layer-NNN`: rebuild the incoming cotangent at boundary L+1 by
chaining ≤ S−1 render-free source backwards from the nearest strided
checkpoint above L (the same chain stage A runs, so the arithmetic is
identical), then run the existing per-layer reverse step for layer L:
replay the layer's sealed retained windows' renders, recompute per probe,
project per candidate, and commit that layer's 867 units' cost rows to a
per-layer checkpoint space. Every quantum is one PB action with its own
data-manifest slice and its own residency plan (§5). Quanta may run in any
order, on either Spark, concurrently with each other.

**Stage C — join (deterministic merge).** Merge the 45 per-layer cost payloads
into the campaign's `joint-cost.pkl` shape (the allocation stage's pareto
input) under a coverage proof; report gaps, never invent them (§7).

### 2.3 Independence proof (what may run concurrently)

Two quanta `layer-L` and `layer-M` (L ≠ M) share **no mutable state**:

- **Read sets are disjoint.** Renders and source-extent entries partition by
  layer in the sealed manifest (each entry belongs to exactly one layer
  phase; the producer's coverage proof checks the tiling at entry level,
  §4). Boundary entries are keyed `(batch, layer)`; distinct layers read
  distinct entries. The strided adjoint checkpoints are read-only once
  stage A publishes them.
- **Write spaces are disjoint by key.** Each quantum writes only
  `<output_root>/layer-quanta/layer-NNN/` (§6.4). Checkpoint shards are
  keyed by `sha256(unit_qname)`; a qname embeds its layer index
  (`model.language_model.layers.<L>.…`), so two layers cannot name one
  shard — the same property the head-walk journals rely on (#768).
- **Shared mutable surfaces are PB's, not ours.** Stage/ram occupancy tokens,
  the fill ledger and the residency maps are arbitrated by PB's tier loop;
  quanta interact with them only as competing consumers of a shared tier,
  which is the condition the tokens exist to price. On `/mnt/shared` the
  quanta write only inside their own output root.
- **The results join is a disjoint union.** The merged payload's `costs` map
  takes each qname's rows from exactly one quantum; no row is ever combined
  across quanta. Addition being commutative here means: *completion order
  cannot change the merged bytes.* The ordering that IS required, and why:
  (a) **within a layer**, floating-point accumulation over windows and probe
  partitions must follow the sealed window index order and the batch
  partition order — the quantum replays windows in sealed index order, exactly
  as the single run does, because FP addition is not associative and the
  single-run equality gate (§9.3) depends on it; (b) **within the joiner**,
  serialization is canonical (sorted keys, §7.2), so the file bytes are
  deterministic whatever order the receipts arrive in; (c) **across quanta**
  no ordering is required at all — there is no cross-layer arithmetic in the
  join.
- **Hardware state is not shared mutable state.** Both Sparks are GB10s with
  identical kernels; a quantum's outputs depend on its inputs and the sealed
  plan, never on which box ran it or on co-tenants (PB measurement isolation
  applies if a quantum is submitted `--measurement`; not required here).

A builder implements stage B as N independent PB actions with the CLI of §6
and no inter-quantum edges; stage C as a pure function of receipts.

## 3. The layer-quantum record (contract)

Owned by the producer module `prismaquant/joint_layer_quanta.py` (pure data,
stdlib only, torch-free like `joint_prewarm_phases`). One JSON record per
layer; `LAYER_QUANTUM_SCHEMA = "prismaquant.joint_layer_quanta.v1"`.

### 3.1 Fields

```json
{
  "schema": "prismaquant.joint_layer_quanta.v1",
  "quantum_id": "layer-013",
  "layer": 13,
  "campaign": {
    "plan_path": "…/complete-512-seed237….hostcap32.plan.json",
    "plan_sha256": "eff2f7fb…",
    "prepared_path": "…/prepare/prepared.json",
    "prepared_sha256": "962207a3…",
    "read_manifest_sha256": "4100221c…",
    "campaign_scope": { …the sealed scope block, copied verbatim… },
    "unit_roster_sha256": "…sha256 of the sorted 36,423-qname roster…"
  },
  "read_set": {
    "manifest_path": "…/layer-quanta/manifests/layer-013.data-manifest.json.gz",
    "manifest_sha256": "…digest of the sealed slice bytes…",
    "entry_count": 5211, "total_bytes": 87712291850,
    "source_phase": {"name": "layer-13", "start_bytes": 61539435410,
                     "end_bytes": 149251727260}
  },
  "chunks": [
    {"name": "layer-013-chunk-000", "start_bytes": 0, "end_bytes": 29237430617},
    {"name": "layer-013-chunk-001", "start_bytes": 29237430617, "end_bytes": 58474861234},
    {"name": "layer-013-chunk-002", "start_bytes": 58474861234, "end_bytes": 87712291850}
  ],
  "windows": [
    {"window_index": 0, "names": ["model.language_model.layers.13.mlp.gate_proj", …],
     "statistics_bytes": …, "render_file_upper_bound_bytes": …,
     "candidate_count": …}
  ],
  "adjoint": {
    "checkpoint_boundary": 20,
    "chain_layers": [19, 18, 17, 16, 15, 14],
    "boundary_artifacts": "…/layer-quanta/adjoint/…",
    "receipt_sha256": "…"
  },
  "output_space": {
    "root": "…/complete-512-seed237….encoder-reuse-02/layer-quanta/layer-013",
    "cost_payload": "…/layer-quanta/layer-013/cost.pkl",
    "results": "…/layer-quanta/layer-013/results.json",
    "counters": "…/layer-quanta/layer-013/counters.json",
    "checkpoint_dir": "…/layer-quanta/layer-013/checkpoints"
  },
  "identity_sha256": "…canonical_json_sha256 of the whole record…"
}
```

Rules a builder implements without asking:

- `quantum_id` is `f"layer-{layer:03d}"`; `layer` ∈ the plan's source layers
  (0…44 at this plan). Names are unique, matching the #768 quantum naming.
- `campaign` binds the record to one sealed campaign: plan digest, prepared
  digest, the **full run's** read-manifest digest, the sealed campaign scope
  block, and the roster digest (same construction as #768's `roster_digest`:
  sha256 of the sorted qname roster, one per line). A record from another
  plan, prepared revision, or scope refuses everywhere it is checked.
- `read_set` names the layer's **slice manifest** (§4.3): a standalone
  `prismaquant.prismabuild.data_manifest.v1` document whose entries are
  exactly the source manifest's entries in the layer phase's byte range, in
  read order, with a fresh phase table of the chunk names. `source_phase`
  cites the parent phase by name and byte range so the tiling proof can be
  replayed against the parent manifest alone.
- `chunks` tile `[0, total_bytes)` of the slice half-open, contiguously, each
  boundary on an entry boundary (the same rule `residency_plan.validate_plan`
  enforces; a cut through an entry would hand a mover more bytes than its
  tokens reserved). Derivation in §5.3.
- `windows` is the layer's slice of the sealed retained-window partition
  (`windows_by_layer`), copied verbatim with the window order preserved —
  the quantum replays windows in this order and no other. The partition is
  sealed at derivation from one of exactly two sources: the plan's
  `retained_window_budget_derivation` block (its original home), or — when
  the run plan deliberately carries none because it is the single-run plan
  the prepared completion binds — an explicit `window_partition` derivation
  input carrying the same sealed record. Two sealed sources that disagree
  refuse; neither source refuses; there is no default and no silent
  override. The derivation envelope (v2) records which source was used and
  the partition's own canonical digest, and each record's `windows` field
  remains identity-sealed regardless of source.
- `adjoint` names the nearest strided checkpoint boundary at or above L+1,
  the chain layers it must walk (descending, exclusive of the checkpoint,
  down to and including L+1's producing layer set — §6.2), the stage-A
  receipt, and its digest. For layer 44 the checkpoint is the tail set and
  `chain_layers` is empty.
- `identity_sha256` is `cost_stage_checkpoint.canonical_json_sha256(record)`;
  it is what the joiner's custody check and the PB action identity (via the
  record file as a CAS input) both carry. Any field change is a new identity;
  there are no in-place edits, ever.

### 3.2 Coverage proofs

Two pure functions, both refusing rather than repairing (the #768 pattern):

- `verify_quanta_coverage(records, parent_manifest)` — the parent run
  manifest's entries are tiled exactly once across all records'
  `source_phase` ranges (contiguous, gapless, entry-aligned), every record
  shares one `campaign` block, `quantum_id`s are unique, and the union of
  `windows` equals the plan's whole window partition. A lost or duplicated
  layer is a refusal naming the gap, never a silent hole.
- `check_quantum_for_campaign(record, campaign)` — one record against the
  live plan/prepared/manifest digests and scope; refuses a record built for
  any other campaign revision.

### 3.3 Stage-A receipt

`<output_root>/layer-quanta/adjoint/adjoint-capture.json`,
schema `prismaquant.joint_adjoint_capture.v1`: the run identity (plan,
prepared, scope, implementation digest), the boundary-storage generation id
and its policy, per-layer boundary entry digests, the strided checkpoint list
(`[{boundary, activation_entries: […], shared_state_entries: […],
cotangent_sha256}]`), the retention budget actually consumed, and the
telemetry block of §8. Sealed atomically (`atomic_write_bytes`) once, first
writer wins, exactly like a residency plan freeze; quanta bind it by digest
and refuse a moved or mismatched receipt. **The generation id makes boundary
paths non-deterministic across stage-A attempts; that is why boundary
artifacts are receipt-addressed rather than sealed into slice manifests**
(§10 records the sealed-identity change that would lift this).

### 3.4 Stride derivation (S)

`S = 8` at this plan, derived, not chosen: the artifact budget for retained
adjoint state is `ceil(45 / S) × checkpoint_bytes(boundary)` where
`checkpoint_bytes` is 4 probes × the measured per-layer boundary bytes
(≈8.5 GiB → ≈34 GiB per checkpoint; the exact number comes from stage A's own
telemetry on the first run and is sealed into the receipt). S=8 retains
⌈45/8⌉ = 6 checkpoints ≈ 204 GiB alongside ≈383 GiB of boundaries — inside
the plan's 446.7-GB boundary-artifact budget class, which the distributed
plan rederives as `boundaries_all + checkpoints + tail + margin`. The per-
quantum price is ≤ S−1 = 7 render-free layer backwards. A builder reads S
from the plan's new `distributed_campaign` block (§5.1) and derives the same
checkpoints; S is a plan knob, not a CLI guess, and the plan records the
arithmetic beside it the way `retained_window_budget_derivation` already
does.

## 4. The producer (contract)

`prismaquant/joint_layer_quanta.py`, stdlib only, no GPU image needed (a CPU
checkout builds the campaign — the same property `glm_data_manifests.py`
keeps). Pure, deterministic, testable: same inputs → byte-identical records.

### 4.1 Functions

- `layer_quanta(plan, prepared, parent_manifest, *, chunk_target_bytes,
  stride, output_root)` → `{"records": […45 records…],
  "slice_manifests": {quantum_id: manifest_dict},
  "adjoint_manifest": <the stage-A read manifest, v2 `read_plan` in true
  consumption order -- `head`, per-layer `forward-{L:03d}` ascending,
  per-layer `chain-{L:03d}` descending (no tail phase; tail work commits
  under forward-last) -- with `entry_indices` into one
  entries list (no duplication for the repeated reads) and
  `entry_point: "prismaquant.joint_adjoint_capture"`>,
  "coverage": <the proof object>}`. Reads the plan, the prepared
  completion, the sealed run manifest and the campaign scope; cuts per §3;
  writes nothing (callers persist). Determinism: layers ascending; every list
  sorted or in sealed order; canonical JSON only.
- `verify_quanta_coverage`, `check_quantum_for_campaign` (§3.2).
- `join_layer_quanta(receipts, dest, campaign, roster)` — the joiner entry
  point (§7), colocated because it shares the custody/coverage machinery.
  *(Resolved 2026-09-19, #787 decision D5: the joiner is NOT colocated. #783
  merged ``prismaquant/joint_quanta_join.py`` first and that module won --
  the producer's parallel joiner was deleted, not adapted. The joiner
  imports this module's pure constructions -- ``roster_digest``,
  ``phase_ranges``, ``quantum_id``, ``qname_layer`` -- so both sides of the
  wire share one spelling of every digest and id.)*
- `slice_layer_manifest(parent_manifest, layer)` — the slice builder (§4.3).

### 4.2 What the producer must NOT do

No threads, no placement, no submission, no execution, no sizing heuristics
beyond the two derivations this note pins (chunks §5.3, stride §3.4). It is
data: a reviewer can diff two campaigns' records byte for byte.

### 4.3 The slice manifest

A standalone v1 data manifest: `mount_prefix` copied; `entries` = the parent
manifest's entries with cumulative offsets in `[phase.start_bytes,
phase.end_bytes)`, re-based to start at 0, order preserved (consumption
order is never re-sorted); `annotations.phases` = `head` (the quantum's
shared prefix: the layer's source extents for the chain and install, the
head inputs the CLI reads — §6.2 — and the layer's boundary artifacts are
deliberately NOT manifest entries, §3.3) followed by the chunk names; plus
`annotations` binding: `entry_point: "prismaquant.joint_cost_quantum"`,
`quantum_id`, `parent_manifest_sha256`, `plan_sha256`, `prepared_sha256`,
`campaign_scope`, `windows` (copied), `argv` (the exact sealed CLI argv).
The gzip member is sealed deterministically (`gzip.compress(mtime=0)`) as
`dispatch_tessera_campaign` already does, and its digest is the record's
`read_set.manifest_sha256`. The slice is what `pbrun --data-manifest` seals
into the quantum's action key — so a different slice is a different action,
and no receipt can answer for bytes another action read.

## 5. The dispatcher (contract)

`tools/dispatch_joint_layer_campaign.py` — a submitter, not a scheduler. It
publishes rows PB owns; it never claims, places, retries or reorders work,
and it holds no long-running state.

### 5.1 The plan block

The distributed campaign is declared in the plan (a new optional
`distributed_campaign` block beside `source_prefetch`; absent = today's
single-consumer behavior, byte-identical):

```json
"distributed_campaign": {
  "schema": "prismaquant.joint_layer_quanta.plan.v1",
  "enabled": true,
  "consumer_tags": ["gb10"],    /* conjoined tags PB matches; class, not host */
  "max_resident_consumers": 2,
  "ram_window_gib": 160,
  "chunk_target_bytes": null,   /* derived when null: §5.3 */
  "cotangent_checkpoint_stride": 8,
  "submission_priority": -5,
  "adjoint": {"entry": "prismaquant.joint_adjoint_capture",
              "tag": "sparky"},
  "consumer": {"entry": "prismaquant.joint_cost_quantum"}
}
```

`consumer_tags` names the placement tags PB **conjoins** when it matches a
worker (every listed tag, never any one of them): the default is the shared
`gb10` class tag that both Sparks offer, and the dispatcher refuses an empty
or ill-typed list rather than publishing an unconstrained or unplaceable
row. `max_resident_consumers` is the declared concurrency the ram window is
double-buffered for, not a scheduler: it sizes chunks (§5.3) the way
`prefetch_lookahead` sizes the streaming context.

### 5.2 Submission shape and order

The tool is idempotent and receipt-driven — each invocation derives the same
rows (action keys are content hashes; a resubmission is a CAS hit, never a
repartition, the same freeze semantics `residency_stage_rows` already keeps):

1. Seal (or verify) all 46 artifacts: 45 records + slice manifests, coverage
   proof green. Nothing is submitted if the proof refuses.
2. **Stage A first:** submit the adjoint action — `pbrun --tag <adjoint.tag>
   --data-manifest <adjoint read set> --residency stage
   --progress-phase head=<head_grace> --progress-phase <phase>=900 … (one per
   manifest read phase, in manifest order) --container-image
   <spec image> --detach -- python3 -m tools.tessera_campaign_container --spec
   <spec> -- python3 -m prismaquant.joint_adjoint_capture …` — and record its
   action key in `<output_root>/layer-quanta/campaign-state.json` (the
   campaign's own machine-readable state; atomic append of submission events,
   never edits). The payload carries `--data-manifest-sha256` (the submitted
   manifest bytes) and `--read-manifest-sha256` (the annotated parent
   read-set digest): a pass that seals no read schedule is told its manifest
   by the submitter, and a pass told nothing binds nothing and gets no
   redirect (PQ #835).
3. **Then quanta, when their inputs exist:** a layer-L quantum is publishable
   once stage A's terminal record says `executed` AND
   `adjoint-capture.json` validates (digests match the state file). The tool
   publishes every publishable quantum not yet submitted, then exits. Re-run
   it (cron, a shell loop, or a human) as stage A completes; publication
   order is descending layer id — deterministic, and the order that fronts
   the cutover gate, because the single run's reverse walk also completes
   high layers first, so both campaigns hold layers 44, 43, 42 earliest
   (§9.3). Within a run every publishable row is published — the tool never
   waits on a worker.

Per quantum:

```
pbrun --tag gb10 \
      --data-manifest …/manifests/layer-013.data-manifest.json.gz \
      --residency stage --residency-ram auto \
      --container-image <spec image> \
      --progress-phase head=<head_grace> \
      --progress-phase layer-013-chunk-000=900 … (one per chunk) \
      --priority -5 --env PRISMAQUANT_DEV_MODE=1 --detach -- \
      python3 -m tools.tessera_campaign_container --spec <spec> -- \
      python3 -m prismaquant.joint_cost_quantum \
        --quantum …/layer-quanta/records/layer-013.json \
        --quantum-sha256 <record file wire digest> \
        --plan <plan> --plan-sha256 <plan digest> \
        --prepared <prepared> --prepared-sha256 <prepared digest> \
        --adjoint <adjoint-capture.json> --adjoint-sha256 <receipt wire digest> \
        --data-manifest-sha256 <slice manifest bytes digest> \
        --resume \
        --output-root …/complete-512-seed237….encoder-reuse-02
```

The slice digest is the row's own read-set digest (`read_set.manifest_sha256`),
verified against the slice file at dispatch: the quantum binds the bytes pbrun
stages for it, never the campaign parent it also carries (PQ #835). The
record digest is the record file's wire bytes (the consumer checks raw bytes
first; its canonical body check inside stays), and the receipt digest is the
receipt file's wire bytes (PQ #838: earlier rows bound the canonical digests
and died in argparse or at the first gate).

Wire and document identity stay distinct end to end. The producer seals the
canonical digest of the decoded receipt (`bind_adjoint_receipt`); the writer
persists pretty JSON plus a newline (`write_adjoint_receipt`). The dispatcher
receipt gate and the consumer's record-vs-argv check therefore compare the
record's canonical digest against the canonical digest of the decoded file --
never raw bytes against the seal, which valid writer output fails. The CLI
flags bind wire on both files, and the wire checks stay where they were.

Quantum records are produced, never edited (producer D3). The reviewed
regeneration path is `tools/regenerate_joint_quanta.py`: it replays the
`layer_quanta()` producer call from digest-verified plan/prepared/parent
inputs (gzip-transparent, digest over wire bytes) with the authoritative
output root (never the tool's own directory), writes record files into a
reviewed directory and slice manifests at the producer-named absolute
paths the records bind (verified to resolve after writing), reproduces
receipt-less at the original root against on-disk records under
`--expect-existing`/`--original-root` (Gate 1a), validates that a root move
touches only the authorized path fields (Gate 1b), and re-seals against
`--adjoint-receipt` (Gate 2). Old records and history stay where they are;
boundary payloads are never copied. The adjoint manifest itself is the
phase worker's file and is never written here.

- `PRISMAQUANT_DEV_MODE=1` (PR #776) is the interim lane: submissions run
  from any checkout with no campaign branch, no transition receipts, and the
  provenance gates stamp (`dev_uncertified: true` + the executing tree's
  digest) instead of walling. The design does not depend on any other
  dev-mode behavior; certified mode stays the production resubmission path.
- The quantum record itself is passed by path+digest and sealed into the
  action key via the slice manifest's `argv` annotation; the CLI re-verifies
  both digests before doing anything (fail closed, exit 3, §6.4).
- `--container-image <spec image>` is the campaign container declared to
  PrismaBuild *before* it claims the row (PB #714, §14). It is read from the
  same parsed spec that is serialized into `--spec`, so the image the action
  runs is the image PB admitted it against.

### 5.3 Placement policy, chunk derivation, and failure modes

**Placement: one shared `gb10` class tag on every quantum row; PB owns which
box claims.** A row tagged `gb10` is claimable by whichever Spark's worker
loops reach it first; PB's ready-order (−priority, −passes, age), the boxes'
own loop counts (5 vs 3) and the tier tokens do the balancing. The tool
never reads capacity to choose a box. The tag list is a **conjunction**, not
a menu of boxes: PB admits a worker only when it offers *every* listed tag
(`wanted.issubset(offer.tags)`, `src/prismabuild/pool.py:2847`), and each
live Spark offers `gb10` plus its own host name — so the host pair this
design originally shipped (`sparky`+`sparklina`, PQ #831) admitted neither
box and every quantum row was unplaceable. The dispatcher defaults to the
single class tag; a plan may declare `consumer_tags` for a genuine
conjunction (for example a required capability tag), and the rows carry
exactly that list. Failure modes, stated: (a) *straggler* — one box finishes
its queue share early; PB assigns the remaining rows to it, which is the
intended behavior and the reason not to pre-split layers by host;
(b) *the class tag refuses* (both Sparks offline) — their loops claim
nothing and the campaign waits; one Spark offline is the ordinary case and
the other box drains the rows, slower, with no action from anyone;
(c) *a claimed quantum dies* — PB's retry policy applies per action
(movers' `retry_safe` already true by construction; consumers declare
`retry_safe` because the checkpoint journals re-verify), and a re-claim may
land on the other box because the outputs are keyed per layer under the
shared output root, not per host; (d) *the failure mode we refuse to have*:
an agent or tool watching utilization and steering boxes at runtime. If the
static policy starves a box, that is a PB placement capability gap to file,
not a knob to turn here.

**Chunk derivation.** `chunk_target_bytes = floor(ram_window_gib × 2³⁰ /
(2 × max_resident_consumers))` — a double-buffered window split across the
declared consumers: with W = 160 GiB and B = 2, chunks target 40 GiB; then
bounded (clamp to [8 GiB, 64 GiB], refusing the plan if the window cannot
support 8-GiB chunks — a window smaller than 32 GiB with two consumers is a
plan error, not a smaller chunk, because per-chunk movement-node overhead
then dominates). Per layer, `n = ceil(layer_bytes / chunk_target_bytes)` and
boundaries are the **first entry boundary ≥ k × layer_bytes / n** for
k = 1…n−1 (ceiling cut, deterministic, entry-aligned; layers 0–2 at 3.07 GiB
are single-chunk). Layer-3: 133.72 GiB → 4 chunks ≈ 33.4 GiB; a typical
81.74-GiB layer → 3 chunks ≈ 27.2 GiB. Why 2×B: the run-ahead budget a
rolling consumer earns is `capacity − step` (PB #633/#671 arithmetic), and
the phase being read is not run-ahead — so a window must hold B × (current +
next) chunks before any consumer stalls. At W=160, B=2, C=40: 4×40 = 160 —
the window is exactly the working set, and a consumer that reports no
progress is still bounded by `step` = one chunk, not one layer.

**Promotion node shape per chunk.** None of this is new machinery: `pbrun
--residency stage` seals, from the slice manifest's own phase table, one
`mover_row` (pool→stage) and one `ram_mover_row`/`ram_egress_row` pair
(stage→tmpfs, `/ram/prewarm` on `ram:dl380g10`, gated on the chunk's stage
range having landed) **per phase** — the chunk phases *are* the sub-phase
promotion granularity; the residency plan freezes them all at submission and
the tier loop publishes/evicts them on the quantum's accepted progress, one
chunk at a time (PB #601/#633/#640/#672 semantics, unchanged). For chunk
`layer-013-chunk-001` the sealed rows are the existing shapes, verbatim:

```json
{"phase": "layer-013-chunk-001", "start_bytes": 29237430617,
 "end_bytes": 58474861234, "stage_gib": 28,
 "mover_row": {
   "action_key": "…hash of the row body…",
   "resources": {"cpu": 1, "mem_gb": 4,
                 "stage_gib@prismabuild-stage:dl380g10": 28,
                 "fill_mb_s@prismabuild-stage:dl380g10": "<priced from mover receipts>"},
   "residency": {"schema": "prismabuild.residency.v1",
                 "tier_id": "prismabuild-stage:dl380g10",
                 "manifest_sha256": "<the slice manifest's digest>",
                 "range_start_bytes": 29237430617, "range_end_bytes": 58474861234}},
 "ram_mover_row": {
   "action_key": "…hash…",
   "resources": {"ram_gib@ram:dl380g10": 28},
   "residency": {"schema": "prismabuild.residency.v1", "tier_id": "ram:dl380g10",
                 "manifest_sha256": "<the slice manifest's digest>",
                 "range_start_bytes": 29237430617, "range_end_bytes": 58474861234}},
 "ram_egress_row": {"action_key": "…hash…", "resources": {"cpu": 1}}}
```

(The exact resource spelling is what `residency_stage_rows` seals today; the
figure pins the shape — one movement node per chunk leg, tokens at the
chunk's ceiling, pins naming the slice manifest and the chunk's range — so a
builder checks the slice manifest's phase table and needs nothing else.) The
submitter's only obligations are the chunk table (§4.3) and the progress
phase names matching it, which is what `residency_plan.remaining` matches
against. A 134-GiB layer therefore streams through the 160-GiB window as
four 33-GiB promotions with ≤ 2 resident per consumer, instead of one
134-GiB promotion with zero run-ahead.

**Failure modes of the feed, stated:** (a) a chunk promotion lands partially
— PB #644's release semantics and the map's fail-closed ram epoch fence
apply per chunk; the consumer reads the stage copy, never a hole; (b) the
fill ledger under-prices the tier — post-#654/#669 fences require an honest
mover receipt to rebuild best, and every chunk mover is an honest mover (it
copies pool bytes); (c) stage capacity — the stage is 720 GiB shared with
the running single-consumer campaign; every chunk reserves `stage_gib@tier`
at its sealed size, so admission, not the design, arbitrates (interim
coexistence in §9.1).

## 6. The per-box consumer runtime (contract)

New module `prismaquant/joint_cost_quantum.py`; CLI
`python3 -m prismaquant.joint_cost_quantum`. It reuses the existing seams —
`compute_aura_cost_streamed`'s per-layer reverse step, `prefetched_boundary_batches`,
`cost_stage_checkpoint` journals, PWC verified loads, the residency-map
reader — scoped to one layer (the `unit_filter` seam already exists). No new
cache, no second preload path (AGENTS.md principle 3): renders flow through
`ProductionWeightCache` exactly as the single run loads them, boundary
artifacts through the existing `StreamedBoundaryArtifacts` reader.

### 6.1 Arguments

`--quantum PATH --quantum-sha256 HEX` (the record; digest re-verified),
`--plan PATH --plan-sha256 HEX`, `--prepared PATH --prepared-sha256 HEX`,
`--adjoint PATH --adjoint-sha256 HEX` (stage-A receipt),
`--output-root PATH` (the campaign output root; the quantum writes only
under its `output_space`), `--device cuda`, `--profile-tool cprofile`
(optional, default off — the single run's profile tool applies to the
campaign of record, not every quantum).

### 6.2 What it does, in order

1. Verify all four digests and the campaign binding (`check_quantum_for_campaign`);
   refuse on any mismatch (exit 3, `quantum_identity_refused`, nothing written).
2. Load head inputs (the slice's `head` phase: plan, prepared completion, PWC
   pickle, calibration, source identity cache — the same 7.4-GiB set, via the
   same verified loaders; dev mode stamps what certified mode walls).
3. Rebuild the incoming cotangent at boundary L+1: for each chain layer
   (descending), install the source layer from its extents, recompute from
   the stored boundary, backward with the carried cotangent (activation +
   deserialized shared state), exactly the arithmetic stage A's chain runs —
   identical kernels, identical order, so the §9.3 equality gate holds.
   Source-extent reads for chain layers are in the slice's `head` phase and
   therefore tier-fed.
4. For each window of `record.windows` **in sealed index order**: load the
   window's renders through PWC (residency-map redirected; per-file SHA
   verified), recompute the layer per probe from the stored boundary with
   the incoming cotangent, project each candidate's `dW`, accumulate the
   window's statistics in sealed order, commit finished units to the
   quantum's checkpoint space as they complete (per-unit shards, the
   existing journal grammar; resume re-verifies banked bytes before trusting
   them).
5. Report accepted progress per chunk through the declared phase names
   (`PRISMABUILD_ACTION_PROGRESS_PATH`), cumulative across phases, continuing
   from what the head committed — the same currency the single run reports,
   which is what advances the promotion window.
6. Write the outputs (§6.4) and the counters (§8.1); exit 0 only when every
   unit of the layer has a committed, re-verified cost row.

### 6.3 What it never does

No whole-model load (one source layer installed at a time; the plan's
retained/operator-window budgets apply unchanged per quantum — they were
derived from the worst layer, so every quantum fits them by construction).
No render synthesis, no wire reads, no head walk (the prepared completion is
an input, not a task). No writes outside `output_space`. No cross-layer
state: the process ends with the layer.

### 6.4 What it writes

Under `<output_root>/layer-quanta/layer-NNN/`:

- `cost.pkl` — the layer's payload: `{"costs": {qname: {fmt: row}}},
  "provenance": {…the campaign binding, the quantum identity, the adjoint
  receipt digest, per-window telemetry…}`, rows validated by
  `validate_joint_aura_entry` before write; pickled with the run's pinned
  protocol; atomic.
- `results.json` — the per-quantum report (the single run's `results.json`
  shape restricted to this layer: identity blocks, env, io counters,
  residency report, phases, peak GPU bytes against the plan budget).
- `counters.json` — §8.1.
- `checkpoints/` — the quantum's `cost_stage_checkpoint` journal (manifest +
  unit shards), identity extras carrying the quantum identity digest.
- `status.json` — terminal receipt: `{"schema":
  "prismaquant.joint_layer_quantum.status.v1", "quantum_id", "identity_sha256",
  "status": "complete"|"gapped", "units": [n_done, n_total], "unix"}`,
  written atomically as the last act.

A retried quantum resumes from its own journal (idempotent; banked units are
re-verified, not recomputed); a quantum whose record digest changes writes
to a different `identity_sha256` and refuses to touch the old space.

## 7. The joiner (contract)

`prismaquant/joint_quanta_join.py` (module + CLI `main`; #783, and the
survivor of #787's two-joiner decision — the §4.1 sketch of a colocated
`join_layer_quanta` and a `tools/join_joint_layer_costs.py` CLI is
superseded). Deterministic merge of per-layer payloads into the campaign's
results shape — the pareto input — with gaps reported, never hidden.

Wire pins recorded by #787 (2026-09-19), each verified against the sealed
takeover records and the §6 runtime's writer before landing:

- The roster digest is #768's construction (§3.1): sha256 of the **sorted**
  roster, one per line, **no trailing newline** — imported from the
  producer's `roster_digest`, never recomputed.
- Expected quantum ids are **constructed** (`layer-{layer:03d}`) from the
  parent manifest's declared layer set (`annotations.layers`, else its
  `layer-N` phase names); the sealed manifest's phase rows carry unpadded
  names and no `quantum_id` field, and non-layer phases (`head`) are not
  quanta. The tiling proof replays through the producer's `phase_ranges`
  (cumulative marks, entry-aligned). The CLI reads the gzip-sealed run
  manifest member; digests stay over the sealed file bytes.
- Records seal windows index-only (D2, derivation v2); a gap's units are
  named from the caller's roster by the record's layer (`qname_layer`), so
  gaps carry true unit counts and an absent record still accounts for its
  units — a complete quantum that drops rows refuses.
- The payload provenance grammar (§6.4) is what the runtime seals:
  `campaign_binding` (plan/prepared/read-manifest digests, scope, roster
  digest), `distributed_quantum` (quantum id, record identity, adjoint
  receipt digest, checkpoint boundary, chain layers, window count, chunk
  names), and the top-level `adjoint_receipt_sha256`. The joiner checks
  both blocks and refuses unbound (pre-A) records; no implementation digest
  is promised in the payload — it is bound through `prepared_sha256`.

### 7.1 Inputs and checks

One receipt per quantum (`{quantum_id, cost_path, cost_sha256,
identity_sha256, status_path}`) plus the campaign binding (plan, prepared,
parent manifest, scope, roster) **supplied by the caller, never derived from
the surviving shards** — a lost quantum fails the coverage proof instead of
shrinking the layer set to fit (the #768 rule). Checks, in order:

1. **Custody:** every receipt's `identity_sha256` matches its record; every
   payload's provenance equals the campaign binding and answers for its
   record (the §7 wire pins above — including the adjoint receipt digest);
   only per-layer content may differ.
2. **Coverage:** replay `verify_quanta_coverage` over the receipt set against
   the parent manifest; then check the *unit* tiling — the union of payload
   `costs` keys must be exactly the roster (36,423 qnames), and each qname's
   candidate set must equal the prepared `formats_by_qname[qname]`.
3. **Rows:** every row passes `validate_joint_aura_entry`; a row that does
   not fails the join with the qname named (this is a defect, not a gap).

### 7.2 Outputs and merge semantics

- `<output_root>/layer-quanta/joined/joint-cost.pkl` — the merged payload:
  `costs` = the disjoint union (no arithmetic, no reordering of rows),
  `provenance` = the campaign binding + per-quantum receipts + the join's
  own coverage proof. Serialized deterministically: keys sorted, pickle
  protocol pinned — completion order cannot change the bytes.
- `<output_root>/layer-quanta/joined/results.json` — the campaign
  results.json shape (the pareto-facing report) with `distributed:
  {per_layer: […], gaps: […], joined_unix, coverage_sha256}` beside the
  blocks the single run's report already carries. The allocation stage
  reads costs from the joined `joint-cost.pkl` path it is given; the single
  run's own outputs are left untouched (§9.1).
- **Failure semantics:** a missing or `gapped` quantum does not fail the
   join. The joiner completes the other layers, writes the merged payload
   with `status: "gapped"`, lists the gaps by quantum id and unit count, and
   exits 0 — because retry is free (resubmit the quantum's sealed action
   key; PB re-executes; the joiner re-runs idempotently and the same output
   path is atomically replaced with the complete merge). What fails closed
   is *consumption*: the allocation stage refuses a gapped joined payload
   (its coverage block names the gaps), so a partial campaign can never be
   read as a score. A campaign whose quantum exhausted PB retries reports
   the gap in the handover; repair is a new submission or a plan revision,
   never an edit.

## 8. Observability (contract)

Rob's two metrics, defined so a number answers each.

### 8.1 Per-quantum counters (Rob's bytes_from_ram metric)

`counters.json`, schema `prismaquant.joint_layer_quantum.counters.v1`:

- `bytes_from_ram`, `bytes_from_stage`, `bytes_from_pool` — per layer, from
  the existing residency-map accounting (`residency_report()` already counts
  hits and bytes per tier; this contract requires it **per chunk phase** and
  per window, not only run-total), plus `residency_refusals` by reason (a
  refused entry falls back and is recorded — never silent).
- `gpu_joules`, `gpu_power_w_{p50,p95,max}` — 1 Hz `nvidia-smi
  --query-gpu=power.draw` sampling in the action, reported against the
  ~140 W GB10 envelope. **On GB10, `nvidia_smi.gpu_utilization` is
  non-diagnostic** (AGENTS.md principle 13: it reads 96% for a memory-stalled
  kernel exactly as for a saturated one); GPU utilization is therefore
  reported as `kernel_active_s / wall_s` from the action's profiler (CUDA
  kernel-time sum) plus work per joule (units completed per kWh), and
  utilization percentages are never used to diagnose the hot path.
- `chain`: chain-layer backwards count, wall, joules (the stride's price,
  visible per quantum).
- `windows`: per window — render bytes read, per-tier split, kernel_active_s.
- `phases`: accepted-progress timestamps per chunk (the currency the window
  advanced on).

### 8.2 Fleet-level (ramdisk and GPU per box)

- **Ramdisk utilization, per box:** (a) the existing PB gauge
  `prismabuild_tier_tokens{tier="ram:dl380g10", resource="ram_gib",
  state=capacity|available|held}` is the tier's utilization; (b) per box,
  the campaign report rolls up `bytes_from_ram` by the box that ran each
  quantum (from PB terminal records' `claimed_host`) — the ramdisk's
  *utilization by consumer* is bytes served per box per hour and the ram
  share of that box's total reads. A PB-side gauge family
  (`prismabuild_tier_served_bytes{tier, host}`) would publish this without
  a campaign report; that is a PB capability request this design files (see
  §10), not something this repo builds.
- **GPU utilization, per box:** the existing PB box-window peaks already
  carry GPU power against the device's published reference; the campaign
  report adds the energy-based utilization above per box. The success
  criterion the design commits to measuring: **both boxes run quanta
  concurrently the same night** (the dark-GPU elimination), with per-box
  work-per-joule reported, and no `ram-window-stalled` event with
  `runahead_budget_gib=0` at chunk granularity for the rest of the campaign
  (the #671 arithmetic must not recur one level down).
- **Campaign summary** (in the joined results.json): totals per tier, per
  box, per layer; bytes_from_ram share of the 3.1-TB render read; the
  promotion pipeline's achieved rate per box (bytes_from_ram+bytes_from_stage
  per wall second) against the 213 MB/s single-consumer baseline of §1 —
  the before/after the design exists for.

## 9. Migration and cutover (contract)

### 9.1 Coexistence with tonight's single-consumer run

The distributed campaign **must not** withdraw, reprioritize or rewrite the
running consumer (PB #608's withdrawal hazards; the run is the campaign of
record until §9.3 says otherwise). Rules a builder implements without
asking:

- **Separate output root namespace:** everything the distributed campaign
  writes lives under `<output_root>/layer-quanta/` (records, manifests,
  per-quantum spaces, adjoint stage, and the joined outputs of §7.2); the
  single run's `run/`, `checkpoints/`, `exact-boundaries/`,
  `prepare/` and root-level `joint-cost.pkl` are read-only to it.
- **Shared corpus, read-only:** renders, source shards, prepared completion
  and plan are read by both campaigns concurrently — they are immutable
  inputs (the CAS/sealed-manifest identity guarantees this).
- **Tier competition is intended and bounded:** the quanta's chunk movers
  and promotions share `prismabuild-stage:dl380g10` and `ram:dl380g10` with
  the single run's phase-granular movers. Tokens arbitrate; the quanta
  submit at `submission_priority` (default −5) **below** the single run's
  priority 0, so the running campaign's feed wins contention and the
  distributed campaign eats idle capacity — which is the entire point of the
  second GPU. If the stage fills, admission defers quanta movers (a
  reported denial, not a deadlock: PB #623/#632's lessons are already
  fenced).
- **No plan mutation:** the single run's frozen residency plan, sealed
  manifests and journals are untouched; the distributed campaign carries its
  own slice manifests and its own frozen plans. A resubmitted single
  consumer keeps its frozen window (the existing first-writer rule).

### 9.2 Rollback

The distributed campaign is abandoned by publishing nothing further and
leaving its outputs in place (append-only history). The single run is the
campaign of record throughout; its numbers never depend on anything the
distributed campaign wrote.

### 9.3 Cutover criterion (the gate)

The distributed run's **first three completed layer quanta** (the first
three `status.json` receipts reading `complete`, whatever their layer ids —
publication is descending, §5.2, so these are expected to be the highest
layers, which is also where the single run's reverse walk lands first) must
match the single run's numbers for those layers **exactly**:

- **What is compared:** per-unit cost rows. The single run commits each
  unit's measured envelope to its `cost_stage_checkpoint` journal as the
  reverse walk passes it; the quantum commits the same envelope grammar to
  its own journal. The gate is `sha256` equality of the unit envelopes for
  every unit of the three layers, identity extras aside (the journals are
  identity-bound; the comparison is on the measurement envelopes, both sides
  canonicalized). Bitwise equality is the requirement, not a tolerance: the
  quantum replays the same kernels in the same order from the same exact
  bytes (boundary artifacts are digest-checked; renders are file-SHA
  checked; the chain arithmetic is stage A's own), so any difference is a
  defect to find, never noise to threshold away.
- **When it resolves:** a layer's gate arm resolves when *both* campaigns
  hold that layer's units (the single run reaches layer L in its reverse
  walk; the quantum completes). The comparison tool
  (`tools/compare_joint_layer_gate.py`, builder-owned per this contract)
  reads both journals and prints per-layer verdicts; it is re-runnable and
  owns no state.
- **Who decides:** the gate's verdict (three green layers, both campaigns'
  counters attached) goes to Rob; flipping the campaign of record — the
  allocation stage reading the joined payload as the score, and the single
  consumer retiring — is Rob's call, recorded in the plan (`campaign_of_record`),
  not an agent's. Until the flip, both campaigns may keep running and the
  joiner's output is advisory.

## 10. Deliberately deferred, and the PB-side requests

- **Boundary artifacts in the residency manifest.** Stage A's generation id
  is a uuid, so artifact paths are not sealable at submission. Quanta read
  them plain over the shared mount (they are ARC-warm by construction —
  written through the file server's ARC). Sealing them wants the same
  sealed-identity change #688's report describes for prepare journals
  (needs-design, needs-decision); this design does not sneak it in.
- **A `pbcampaign` residency field.** Quanta submit via `pbrun --residency`
  because that is the path that seals movement nodes today; a campaign-level
  residency field is PB's to add if 46 `--detach` submissions annoy anyone.
  Filed as a PB capability request, not built here.
- **`prismabuild_tier_served_bytes{tier, host}`** (§8.2) — PB-side gauge;
  the campaign report carries the same numbers until PB serves them.
- **Stage A distribution.** The forward capture is sequential by
  mathematics (layer L's input is L−1's output) and reads ~575 GiB total;
  splitting it would buy nothing and cost a boundary handoff protocol. Not
  designed.
- **Strided-checkpoint recomputation tuning (S).** S=8 is derived from the
  artifact budget; measuring the chain's true cost per stride and re-deriving
  S is a builder's follow-up once §8.1's `chain` counters exist on a real
  run. The knob is in the plan, and the plan records the arithmetic.
- **Certified-mode submission of quanta.** The interim lane is dev mode
  (PR #776). A certified distributed submission needs the transition/receipt
  grammar extended to N consumers — designed only if the cutover gate passes
  and Rob retires the single consumer.
- **MoE/other models.** This design is for the GLM-5.3-Flash complete-512
  joint panel's shape (45 decoder target layers whose 36,423 units include
  routed experts, one shared prepared completion). Generalizing is a later
  note, not an assumption smuggled into this one.

## 11. Test obligations for the builders

Each component lands with its gates (dev-mode stamps do not waive these):

- Producer: determinism (two builds byte-equal), coverage refusals (gap,
  overlap, entry-misalignment, mixed campaign), chunk derivation arithmetic
  pinned (W=160/B=2 → 40 GiB; layer-3 → 4 chunks; 3.07-GiB layer → 1),
  stride derivation pinned, slice manifests validate as standalone v1
  documents with the parent-tiling proof green.
- Quantum CLI: identity refusals (every digest mismatch, exit 3), one-layer
  scope (no whole-model load, by memory accounting on a tiny fixture),
  per-window order pinned, counters schema, resume idempotence, and — the
  load-bearing one — **equality with the single-run path on a small
  campaign**: the same tiny plan run both ways, unit envelopes bitwise
  equal.
- Joiner: custody refusals (retargeted receipt, foreign provenance),
  coverage gap → `status: gapped` with exit 0 and the gap named, complete →
  deterministic bytes under receipt permutation, allocation-side refusal on
  a gapped payload.
- Dispatcher: dry-run prints the exact argv (the seam `dispatch_tessera_campaign`
  already has), idempotence (second run publishes nothing new), stage-A
  precondition refusal when the receipt is absent or stale.

## 12. Provenance

- Live evidence: PB action `e28b8c64…` (claimed sparky 2026-09-19T03:26:26Z,
  progress layer-9 04:09:42Z), sealed run manifest `4100221caa…`, plan
  `eff2f7fb…` at
  `…/joint-panel/complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02.hostcap32.plan.json`;
  tier records `arc|ram|prismabuild-stage:dl380g10` read from
  `pb-queue/tiers/` the same night; boundary artifact volume measured at
  85 GB for 10 captured layers.
- PrismaBuild: #583 (movement nodes), #601 (residency window), #632 (46
  phases / 3.60 TB / 744-GB stage incident), #633 (run-ahead bound), #640
  (RAM tier), #654 and #669 (fill-ledger poisoning and its fences), #671 +
  #672 (the 112-GiB zero-run-ahead arithmetic and the 160-GiB window), #192
  (tiny quanta through PB), #567 (cohort design, consulted for placement
  language).
- PrismaQuant: #765/#768 (verification quanta, naming, coverage/custody
  join pattern), #754/#763 (resumable parallel head walk), #776
  (`PRISMAQUANT_DEV_MODE`), #607 (resumed read-order sealing — the slice
  manifests inherit its rules).

## 13. Addendum (2026-09-19, later that night): the stage-A prefetch override (#819)

The first live stage A (PB action `74f12044f774…`) confirmed this note's
IO diagnosis from the other side: at layer ~24/45 the capture showed a few
GPU batches and then minutes of halt, repeatedly — the plan's sealed
`source_prefetch` budget is `{prefetch_workers: 1, max_cache_slots: 2,
prefetch_lookahead: 1}` (#737's single-worker pin), one worker is
latency-bound on many-file collections over NFS-RDMA from the dl380-hosted
tiers, and the GB10s stage no local copy. The plan is frozen for this
campaign (the prepared binds its digest; re-sealing costs a 7+h
re-prepare), so the §5.2 stage-A submission gains the IO-side #809 seam
instead of a new seal:

- `prismaquant.joint_adjoint_capture` accepts `--prefetch-override
  <path>` (or `PRISMAQUANT_STAGE_A_PREFETCH_OVERRIDE` for direct
  invocations; two explicit sources that disagree refuse). The document
  carries a non-empty `reason` and a `source_prefetch` block that passes
  the plan's own completeness check (`_source_prefetch`: the same six
  fields, the same rules, prefetched residency still required).
- Given an override, the capture's model build threads the override's
  budget instead of the plan's, **for that run only**. Plan bytes, the
  plan/prepared digests, and the adjoint receipt are untouched — the
  quanta's receipt bindings do not move. No override given: the plan's
  block verbatim, byte-identical behavior.
- The deviation is stamped into the run's provenance, never silent:
  `results.json` and `counters.json` carry `prefetch_override`
  (`{plan_sealed, run_used, reason, path, sha256, source}`, or `null`),
  and the attempt log prints the same block at startup.
- `tools/dispatch_joint_quanta.py --stage-a-prefetch-override <path>`
  threads the payload's `--prefetch-override` (the payload argv is the
  channel that crosses the container boundary — the launcher forwards no
  ambient action environment) and records the path in the campaign
  state's `stage-a-submitted` event.

This is a recorded per-run deviation, not a new default: the next plan
seal adopts measured numbers through `recommend_source_prefetch` (#737),
and the override retires with the campaign that needed it.

## 14. Addendum (2026-09-20): the container image is declared to PrismaBuild before claim admission

RobTand/prismaquant#825, paired with RobTand/prismabuild#714. A GB10-class
action (`dd23c05a3a4b…`) was claimed by sparklina, which did not hold its
pinned image (`sha256:c0e532d28a78…`, installed on sparky), and died inside
`tools.tessera_campaign_container.inspect_or_load` after the attempt was
spent: the image lived only inside the `--spec` JSON, so PB had no placement
declaration to enforce before claiming. The paired PB change adds
`pbrun --container-image REF` (repeatable, immutable refs only) and the
campaign row field `container_images: [REF]`, requiring the
`container-image-v1` worker capability: a box that cannot positively show the
reference leaves the item `ready` for a box that can, and dispatches refuse
when no recorded eligible worker reports it. PB still neither pulls nor
transfers an image.

The PrismaQuant caller half, in this addendum:

- `tools/dispatch_joint_quanta.py` (stage A and quanta) adds
  `--container-image <ref>` to the pbrun envelope, before the payload
  separator. The reference comes from the same parsed spec that is
  serialized into `--spec`, so the sealed spec and the admission declaration
  cannot disagree (a second read could race a spec rewrite).
- `tools/dispatch_tessera_campaign.py` declares `container_images: [ref]` on
  every generated manifest row, from the row's *resolved* container class —
  a class override declares its own image, and `_pbrun_argv` (submit-joint /
  submit-aqua / submit-allocation / submit-export) adds the same
  `--container-image` flag from the same parsed spec it seals.
- **Archive-backed specs declare nothing.** A spec whose `container.archive`
  passes `validate_container` (canonical path, SHA-256, and the
  `content_sha256` seal) has its image established *inside* the action by
  the launcher's digest-verifying `inspect_or_load` on whichever worker
  claims it; a local-presence prerequisite would refuse the claim before the
  loader ran. The loader remains the responsible party and its
  bytes-then-content checks are unchanged. A malformed archive refuses at
  submission rather than silently skipping the declaration.
- No-container rows are byte-identical to before and declare nothing. Rows
  declared with `container_images` (and commands with `--container-image`)
  are **new actions**: the reference is sealed into the action key, so an
  image-pinned submission is a different action from its undeclared twin.
  Existing sealed requests are untouched and are not re-sealed by this
  change; a declaration is only carried by newly built rows and commands.
- A mutable tag is declared exactly as the spec spells it and is refused by
  PB's immutable-only admission: pin the spec's image to `sha256:<64 hex>` or
  `repository@sha256:<64 hex>`, or bind an archive.

**Deployment gate.** The published PB client must carry the flag and the row
field before any new image-declaring submission can succeed. Until it does,
submissions fail closed by construction: `pbrun` exits 2 on the unrecognized
`--container-image` before sealing anything, and `pbcampaign` refuses an
unknown `container_images` row field at manifest load (verified against the
2026-09-20 published client, recorded in the #825 result). There is no
compatibility branch that drops the declaration and no silent fallback;
rollout waits on the paired PB runtime.

Gates: `tests/test_container_image_admission.py` (same-parse identity, direct
payload boundary, manifest field, class override, archive-backed omission and
malformed-archive refusal, non-container rows unchanged) plus the existing
dispatcher, launcher and image-content suites. The red run on
`164db9148f` (PB `cd5d723dec5a…`, 14 failed / 6 passed) is the original
omission; the green runs (PB `afe712fb79b5…`, 20 passed; caller-regression
batch `cec95d11c972…` 5/5 shards, submission-path batch `5d5b9f3e189a…` 7/7
shards) are the corrected direct and manifest paths.
