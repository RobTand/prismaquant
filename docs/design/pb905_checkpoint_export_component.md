# Genuine checkpoint export component (PQ #2111, PrismaBuild #905)

`experiments/pb905_checkpoint_export.py` prepares and authenticates an opt-in,
CPU-only offline checkpoint handoff. Its export mode uses the existing writer
and produced-output lifecycle with a frozen per-group pacing override. This is
component preparation; completion cannot qualify PB #905 or make held PB
PRs #1325/#1326 eligible. There is no production default or pipeline change.

The reviewed input proposal selects the first 768 entries in checkpoint name
order, in twelve distinct groups of 64. Every group contains 1,073,879,616
original serialized bytes; the complete output bound is 12,886,555,392 bytes.
Ordered arms are `paced, unpaced` repeated six times. The full original
checkpoint remains the roster anchor. Changed order, repeated entries, changed
metadata, unequal groups, inadequate budgets and overlapping input/output
paths refuse. No tensor regeneration, serialization, synthetic payload or
replacement checkpoint is allowed.

The genuine source is the boundary-040 checkpoint under
`/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/ws-sa-997-a4/run/layer-quanta/adjoint/`.
Its `checkpoint.json` is 2,384,916 bytes, SHA-256
`a0b60cd5cbb18c4d7600e6868479942e7bc0e55f1c7ca7abe3ecbc52abdc0c3e`.
Selected BF16 cotangents have shape `[1,512,4,4096]` and original session
generation `8f4246cf7bae4c8ebbc200c2a981f76c`. Metadata selection alone does not
authenticate their bodies.

Preparation uses `load_pinned_checkpoint` and PB's retained SDK3 helper. It
publishes fresh packet controls, one complete data-manifest-v2 readset, twelve
cohort projections and the existing write-only boundary template. The packet
is capped at 4 MiB. Authentication and export activate the existing staged
tier policy, bind the complete readset digest, compare PB's admitted read
order and authenticate the staged original checkpoint metadata. Each body
then passes the exact-entry owner, including original archive hash, metadata,
session, tensor/storage shape and file fences. A serial `EntryReadScratch`
provides those same authenticated bytes within its live verified window;
the adapter neither rereads the body nor reserializes the tensor.

Serialized groups use `CaptureMemoryGuard` and the existing allocation check.
Tensor residency remains charged by `StreamedBoundaryArtifacts.reserve_resident`.
Authentication retains only one serialized body at a time; export retains one
group. All original sources remain read-only. Each mode publishes fresh
evidence outside the output prefix and refuses overwrite/reuse of that path.

Only `FrozenArmBackend.submit_group` adds the fixed `paced` argument. The
existing owners still claim canonical prewrites, reserve the local spool,
write bytes, seal PB exports, retry those sealed exports, wait for checked
acknowledgements, commit origins and release spool groups. There is no new
scheduler. Authentication cohorts go through `pbcampaign`; paired export is
a single non-retry-safe action. Progress phases are `input-check`, the selected
ordered `group-NN` phases, then `finalize`. Units advance only after a durable
component record, and export group records follow acknowledged origin commit.
Primary failures survive failures in final evidence publication.

The concrete packet is at
`/mnt/shared/astra-resume-20261002/pb905-phase0/packet-component-v1/packet-files.json`.
Packet SHA-256 is
`5bd5f00c368672feda331fe5fc21229636e009c9e15c0ad95d6cb303b372d50e`;
full readset SHA-256 is
`34b3e128823f633aba59b03ef0a09a89b801e58810088d77676cea2f4f5a8721`.
The preparation action is `0def6d2011ad04ea5931933f3524dd9a6a6e4b451670542714ecf595fa5b47bb`.
Its sealed helper generation is `d028dfee920b-1790960385-1d815cff72d1`.
This does not assert that newer PB main/SDK4 is deployed.

Before a benchmark GO, review the exact source/packet/readset/template hashes,
all 768 authenticated body records, eligible interpreter and host spool,
canonical tier fill price/family, profiling, hard deadline, progress and
cleanup disposition. Proposed parent demand is CPU2/mem8 GiB/GPU0/native1,
priority -10, spool3 GiB, hard deadline1800s, phases `input-check=120`, twelve
`group-NN=300`, `finalize=180`. PB derives the host `spool_gb` charge. No host
capacity, tags, default or gates may be invented to admit it. DL380's qualified
CPU interpreter currently has no host spool offer; a spool-capable Spark needs
an actually qualified interpreter before the paired export can be placed.
The published client requires an `x86` dependency tag for the pinned CPU
interpreter when it is absent on the submitting box; this tag is not a claim
that every worker has that interpreter.

Use one profiled parent, with arm-labelled write/read scopes and local monotonic
durations, plus attributable export/mover phase evidence. A parent sample
profile cannot observe separately admitted mover processes. Retain raw Netdata
CPU/IO/pressure/disk-member series on DL380 and both Sparks. PB #1440 clock
alignment remains unqualified; no aligned cross-host energy/work-per-joule
claim follows from those timestamps. Authentication timing is no pacing A/B.

Only existing useful application movers may provide contention. No dummy
reader or repeated output fill is permitted. Without their bound readsets,
attributed copy-read evidence and a representative displaced baseline, the
outcome is `INCONCLUSIVE` and both defaults remain held. Component results
always carry `pb905_gate_qualified=False`; root evaluates the separate gate.

Export commits origins with explicit `retain` lifetime. Checked origin commit
and spool release do **not** delete canonical output. PB's `origin_retirement_tick`
never deletes a retained origin, and `reclaim_origin` proves already-absent
files rather than unlinking them. The root-owned, evidence-only namespace is
therefore bounded retained evidence (12,886,555,392 payload bytes, plus the
existing template's separately reserved temporary class), pending a reviewed
producer-side disposal plan. This driver does not manufacture consumers or
hand-delete originals/exports to get automatic retirement. A finite physical
cleanup claim is unavailable; the cleanup disposition must be accepted before
GO. Failed prewrites, live exports, uncertain containment or charged debt
remain explicit PB-owned state. The hard deadline is no cleanup proof.

CPU controls cover original-byte custody, input tampering, session/reader
refusal, allocation bounds, frozen arms, full roster/readset checks and evidence
overwrite refusal. A private admitted real-queue fixture executes two actual
exports and checks origin commitments and spool release; its tiny synthetic
files only validate the lifecycle and cannot qualify PB #905's I/O gate.
Current control validation: 21 collected, 21 ran, 21 passed, no skips, through
PB actions `8ceebc5ffe59e8c5eec49896aba2e9f0233e3221b2ef08c48126d663e03aeb6e`
and `fb555bcc07f6b0893af930b83647894e280ef1f64f581d198a2f5ea6de4594dc`
on DL380, Python3.14.4/Torch2.11.0+cpu, reviewed PB/Tessera installs verified
by `pbtest`. No throughput improvement, KL/bpp, serving, GPU or default
qualification is claimed.

All twelve genuine authentication cohorts subsequently completed through PB.
Their durable records cover exactly 768 unique pinned sources and
12,886,555,392 bytes, with matching original hashes and file signatures; no
source bodies were reread by the receipt audit. Every selected body passed the
existing staged exact-entry owner. The maximum final cgroup memory peak across
these twelve actions was 515,936,256 bytes. This is an observed authentication
peak, not an export memory estimate. Seventeen endings (twelve cohorts, two
control shards, two documentation shards and metadata preparation) passed the
existing full action/result verifier, source-bundle/member checks and empty,
released resource-scope checks. The verifier ran read-only from PB's current
SDK4 source; action execution still used the published SDK3 generation.

Operator evidence lives under
`/home/rob/tmp/astra-resume-20261002/pb_p1/`:
`pq905-body-record-audit.json` fixes each cohort's action and durable record
digest; `pq905-verified-endings.json` carries publication/attempt identities,
receipts, source input hashes and containment evidence. Raw cohort records
remain under the shared `authentication-v1/cohort-NN/` namespace. These bounded
records are retained evidence. The paired 12 GiB export has **not** run; input
authentication does not supply its performance, organic contention, eligible
spool/interpreter or physical cleanup proof.
