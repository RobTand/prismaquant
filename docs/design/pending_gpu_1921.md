# Pending MTP/vision qualification evidence index (#2810 / #1921)

**Status: all seven rows pending; no qualification is established by this index.**
This PR submits, schedules and runs no GPU row and grants no policy/gap permit.
It changes no implementation or architectural behavior and requires no architecture
stamp edit. Filing, publication and closure transitions belong to TrustLadder.

## Provenance and ownership

The prospective roster and requirements come from [#2810](https://github.com/RobTand/prismaquant/issues/2810),
which supplies the 2026-10-01 handoff. That handoff is a source of requested
measurements, not proof that any measurement passed. Its historical campaign
`tmp/PENDING-GPU.md` is superseded by this durable index and reportedly unavailable;
neither it nor the historical `tmp/mtp-vision-sol/FINAL-CPU-HANDOFF.md` was inspected
for this record. Those campaign-relative paths are provenance, not current
authorities or portable runtime dependencies.

Rows 1–3 retain #1271/#2772 MTP implementation/campaign acceptance ownership;
their presence here is a shared evidence index, not new execution authorization.
See [the MTP protocol](glm_mtp_option_b_protocol_2771.md), lines 183–233, for
exact-identity reuse boundaries and existing ownership. Selected-cache/export
verification remains with #2809/#2575; policy decisions remain with #2811/#2578;
outstanding full-gamut qualification remains with #2812. This document neither
duplicates those efforts nor reconciles or closes their acceptance.

## Shared proof and identity rules

Existing behavior is bounded by [ARCHITECTURE.md](../ARCHITECTURE.md), lines
3304–3349 (visual discovery/shard plumbing, not complete capture, wire output or
serving qualification) and 3388–3406 (complete measured populations and matching
source shapes required; allocation is separate from export/plugin admission).
The [visual allocator CPU record](../measurements/visual-allocator-roots-2026-10-02.md),
lines 71–81, likewise supplies no empirical visual prices or non-body wire/kernel
admission. These are existing-behavior references, not GPU acceptance receipts.

Source/profile discovery and CPU fixtures do not establish complete multimodal
capture, shape legality, emitted-byte equivalence or served admission. Missing
units, incomplete required populations, mismatched source shapes and unsupported
shapes must continue to refuse. Preserve independent producer, cache, exporter,
plugin and runtime gates, including unsupported non-body low-bit refusal where
applicable. Do not infer identical support for later exporter/profile combinations:
each exact combination needs its own evidence or a valid scoped reuse decision.
Source-native BF16 vision remains the accepted A8S control matching EXL3; separate
full-gamut PACT acceptance remains open and does not block that control.

For every row, retain an immutable manifest linking calibration content/order/split,
source checkpoint revision and tensor digests, processor/tokenizer revisions and
content digests, full configuration digest, implementation revisions, and workload.
Bind applicable selected recipe/assignment, cache, runtime/image, packaged contract,
source-shape roster and original-wire digests to the same manifest. Record hardware,
commands, seeds, observation windows, raw outputs and receipt digests. Reuse evidence
only for the exact identities and scope it demonstrates; changed identities require
owner disposition, not an assumption of equivalence. A field that is genuinely
inapplicable needs an explicit scoped reason, not an omitted identity.

**Actual identities, numerical tolerances, measurement receipts and approvals are
pending for every row.** Before eligible measurements, the existing owners must
record the applicable evaluation protocol, numerical tolerances and decision
provenance; this index supplies none. Historical or CPU receipts cannot be promoted
to current GPU acceptance. A structural document test proves neither the prose nor
any row's qualification.

Every speed, residency or cost claim in any row requires **before/after in-process
profiler evidence plus aligned Netdata from both boxes**, bound to matched workload,
identities and observation windows. Retain raw profiles/series, clock alignment,
window boundaries and completed-work counts so comparisons can be reproduced.
Evaluate GB10 efficiency as completed work per joule against its recorded power
envelope, never utilization alone; retain energy/power evidence for those windows.
Missing profiles, either box's telemetry, tolerances or receipts leave the claim
pending. This document claims no unmeasured improvement.

## 1. current selected MTP whole-artifact/cache/wire/byte qualification

### Inputs

The current selected MTP recipe, complete checkpoint and sidecar roster, production
cache entries, original encoded wire and full exported artifact are required.
Supply a whole-artifact accounting report joining selected units to cache keys,
wire payloads and actual serialized bytes, including immutable regions separately.
Rows 1–3 are evidence requests under #1271/#2772, not a new campaign.

### Identities

Bind the shared calibration/source/processor/configuration manifest to the exact
selected recipe and assignment digest, MTP tensor/shape roster, cache-key and payload
digests, exporter revision, full-artifact and original-wire digests. Include the
applicable runtime pin and packaged-contract digest. Actual bindings are pending.

### Pass bar

Require complete selected-population closure from recipe through production cache
to unchanged original wire and full-artifact byte accounting, with no missing units,
substituted renders or unexplained bytes. Preserve unsupported-shape and independent
export/runtime refusals; selected allocation or CPU accounting alone is not wire
qualification. Numerical tolerances and receipts remain pending. Any cost/residency
claim also needs the shared matched profiles and both-box telemetry.

## 2. MTP served-load/speculative correctness

### Inputs

Use the qualified selected artifact from row 1, its actual serving loader/plugin,
and representative served requests with a matched non-speculative control.
Retain load diagnostics, draft/verify traces and outputs covering acceptance,
rejection, fallback and request/cache lifecycle behavior under served load.

### Identities

Bind calibration/source/processor/configuration identities and the selected recipe,
artifact/original-wire digests to the exact serving image/runtime and contract,
speculation settings, supported TP, request corpus/order, seeds and source shapes.
Record the control's identities separately. Actual pins and receipts are pending.

### Pass bar

Demonstrate actual artifact loading and correct speculative verification, rejection
and fallback against the matched non-speculative control under the owner-defined
numerical protocol. No silent substitution or unsupported-shape admission is
acceptable. Wire closure alone does not prove served correctness. Tolerances and
served receipts remain pending under #1271/#2772. Any throughput, residency or cost
comparison needs before/after profiles and aligned Netdata from both boxes.

## 3. matched MTP BF16/EXL3 acceptance and speed/quality A/B

### Inputs

Supply the selected MTP candidate and matched BF16 and EXL3 references, held-out
quality requests and a fixed served workload. Retain speculative acceptance counts
with denominators, draft/verify timing, end-to-end speed and quality outputs for
each arm, not just a local MTP-head or self-KL surrogate.

### Identities

Bind all arms to immutable calibration and held-out corpus manifests, common source,
processor/tokenizer and evaluation configuration. Record each arm's recipe,
artifact/original-wire digests, runtime/contract, TP, shapes and speculation settings;
record any unavoidable differences explicitly for owner disposition. Identities,
comparison tolerances and measurement receipts are pending.

### Pass bar

Require matched BF16/EXL3 acceptance, end-to-end speed and held-out quality evidence
under the prospective #1271/#2772 protocol; neither a selector curve nor a CPU
fixture establishes acceptance. Preserve independent served admission and refuse
unsupported cells. Speed/residency/cost conclusions require before/after in-process
profiles, aligned both-box Netdata and GB10 work-per-joule/power-envelope evidence.
No improvement or parent acceptance is asserted here.

## 4. complete real multimodal visual/merger Fisher/Hessian/activation capture

### Inputs

Supply real multimodal calibration examples and processor outputs that exercise
the complete required visual tower and merger population, including required
source shapes. Retain activation captures, Fisher/Hessian statistics, sample counts
and a coverage reconciliation against the discovered unit and source-shape roster.
Synthetic fixtures or discovered names alone are insufficient.

### Identities

Bind immutable calibration media/text/order and processor configuration to source
tensor digests, model/profile and capture-code revisions, capture settings and
complete unit/shape roster digests. Bind activation and Fisher/Hessian output
digests and applicable recipe/runtime/contract identities. Actual identities and
coverage receipts remain pending.

### Pass bar

Require complete real multimodal activation/Fisher/Hessian coverage for every
required visual/merger unit and source shape, with valid statistics under the
owner's pending numerical protocol. Missing units, incomplete populations and
mismatched shapes must refuse rather than receive uniform or invented statistics.
Discovery/shard CPU proofs do not discharge this row, and complete capture alone
does not admit low-bit export or serving. Capture cost/residency claims require the
shared before/after profiler and aligned both-box telemetry evidence.

## 5. exact-pin admitted per-shape Tessera rung quality/cost/byte/runtime panel

### Inputs

Use row 4's complete measured population, the proposed rung menu and actual
producer outputs for each required visual/merger source shape. Supply per-cell
quality, cost, serialized-byte and runtime measurements plus an explicit admission
matrix; distinguish measured, refused and still-pending cells.

### Identities

Bind calibration/source/processor/configuration and selected recipe digests to each
unit, shape, rung and original-wire digest. Record the exact Tessera commit and
SHA-256 of its packaged runtime contract, producer/runtime images, hardware and
per-cell scope. The actual pin, contract hash and panel receipt identities remain
pending, not inferred from a release label or a historical panel.

### Pass bar

Each admitted shape/rung cell needs quality within the pending agreed tolerance,
measured cost, actual serialized bytes and runtime evidence at that exact pin and
contract scope. Refuse unsupported shapes/cells; do not extrapolate across shapes,
rungs or later profiles/exporters. Allocation legality does not bypass exporter or
plugin gates. Require matched before/after profiles and both-box Netdata for cost,
speed or residency claims, with GB10 completed work per joule and its power envelope.
Full-gamut PACT acceptance remains pending under #2812.

## 6. selected original-wire cache/export/full-artifact witnesses

### Inputs

Supply the selected visual/merger assignment and complete production cache,
original-wire entries, export output and full-artifact manifest. Retain witnesses
joining each selected unit/shape/rung to cache identity, emitted payload and total
artifact accounting, including preserved source-native/immutable regions.
Verification remains within #2809/#2575 ownership.

### Identities

Bind calibration/source/processor/configuration manifests and selected recipe to
cache keys, original-wire payload digests, full-artifact digest, source-shape roster,
producer/exporter revisions and target runtime/contract identities. Record exact
profile/export configuration, not an assumed family-wide support statement.
Concrete identities, byte witnesses and receipts are pending.

### Pass bar

Prove complete selected-cache coverage and unchanged original-wire emission into
the full artifact, with reconciled actual byte totals and no silent re-encoding,
substitution or dropped units. Missing required population and unsupported non-body
low-bit/shape assignments must still refuse under the applicable exporter gates.
CPU cache/export fixtures and row 5 pricing are not emitted-byte equivalence or
served admission. Tolerances and witnesses remain pending; any export speed,
cost or residency claim needs the shared matched profiles and both-box telemetry.

## 7. plugin recognition/TP/mixed-rung output and BF16/EXL3 serving A/B

### Inputs

Supply row 6's actual-wire full artifact, exact plugin/runtime and contract, the
supported TP configurations, and mixed-rung visual/merger outputs. Retain loader
recognition/routing diagnostics and generated outputs, then matched real multimodal
BF16/EXL3 serving A/B requests and quality/performance observations.

### Identities

Bind calibration/source/processor/configuration, selected recipe, shape/rung roster,
original-wire and artifact digests to the plugin/runtime image, exact Tessera commit
and packaged-contract SHA-256, hardware/TP topology and request/observation manifests.
Record BF16/EXL3 control identities and comparison settings independently. All
actual pins, tolerances, telemetry and serving receipt identities remain pending.

### Pass bar

Require actual-wire plugin recognition and correct mixed-rung outputs on explicitly
supported TP, plus matched BF16/EXL3 served quality and speed evidence under the
owner's prospective protocol. Reject unsupported TP/shapes and missing units;
producer success, allocation and CPU fixtures cannot waive plugin/runtime gates.
Preserve source-native BF16 as the accepted A8S control, not proof of full-gamut
PACT admission. Every speed/residency/cost claim needs before/after in-process
profiles and aligned Netdata from both boxes; GB10 efficiency requires completed
work per joule against the recorded power envelope. #2812 qualification and
#2811/#2578 policy decisions remain open, with no permit granted by this index.
