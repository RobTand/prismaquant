# Selected-source authentication A/B preparation — 2026-09-08

The source-only pilot harness and its CPU contracts are prepared. No native A/B,
original GLM payload hashing, capture forward, activation/Hessian load, encoding,
or serving run was performed for this preparation. Native execution still needs
the original **complete** capture manifest, its SHA256, and root review/freeze of
the exact invocation. The draft cannot pass the entry point's frozen-plan check.

The implementation extends the reviewed selected-source descriptor owner through
existing source construction, prefetch, snapshot and projection mechanisms. It
imports the screen's shared bounded Netdata evidence owner, integrated from
`dd1ecf780e2951b823bbf803e19a16273e8d3786`. Source runtime and architecture remain
byte-identical to `c75f259fefb56433d0c7882795d09536a0a9d807`; this experiment changes
no production default, source contract, lane, format menu or ship gate.

## Fixed experiment

One admitted action runs **full → selected**, once, with no automatic follow-up.
The four source units are layer 0 MLP down/gate and layer 6 expert 0 down/gate
from `/mnt/shared/models/GLM-5.3-Flash-BF16`. The source builder retains its full
selected-layer dependencies and early nonbody tensors. It uses two cache slots,
one prefetch worker, four existing source reader threads, one-layer lookahead,
24 GiB headroom and required prefetched residency.

Both arms use the original census SHA256, original 512 × 512 seed-zero token
artifact, full canonical capture identity and initialization witness, attention
implementation, producer container content digest, and selected unit shapes.
The original complete capture is accepted through the public factory before any
CUDA initialization or source payload access. Its historical initialization
witness stays provenance: this source-only experiment performs no new forward.

The full arm preserves legacy ordering: source construction, fresh full
`capture_identity`, selected snapshot, source teardown, then source projection
checks. Its full authentication must hash all **120** original safetensors
shards, totaling **642,652,070,880 bytes**. The selected arm authenticates metadata
through the public complete-capture owner before source construction and freshly
hashes each consumed shard once through its held descriptor. The source tensor
reads are observations of the existing safetensors seams; the harness adds no
source weight cache or application dispatcher.

Before either timed arm, a 234,881,024-byte BF16 GPU reference is allocated and
zero-filled. Each arm obtains its own source snapshot. After the full arm's
source work, its weights are copied into the reference outside primary source
phases; selected-arm weights are compared through uint8 views for exact bit
parity. Per-weight content identities and projected-source identities must also
agree. Runner teardown and owned descriptor closure are checked. Source file
stat signatures and the complete capture bytes must remain unchanged across
both arms. This reference is bounded measurement state only.

## Conditional admission

[`resource-plan.json`](resource-plan.json) was produced through PB using the
existing `selected_anchor_resources.v2` owner, original source headers and file
lengths. Source payload SHA256 values are copied from the frozen census producer
roster; this resource calculation did not freshly read original payload bytes.
The final resource file SHA256 is
`272da75e558c3696f541cc754f8744fc9e929ea798494aa93b17a90e2a49c23a`.

| Quantity | Bytes |
|---|---:|
| Maximum derived physical phase | 66,264,248,880 |
| Maximum derived GPU subset | 42,625,151,536 |
| Conservative cgroup + CUDA + margin envelope | 111,036,884,064 |
| Full-hash envelope if every source page stays charged | 689,122,948,064 |

The draft requests **104 GiB total physical**, **40 GiB GPU**, seven CPUs with
native thread counts fixed at one, and a single GB10 measurement action through
PB. CPU roles are main, four source readers, prefetch and telemetry. Physical
and CUDA charges can overlap; the guard deliberately adds them. The two-GiB
instrumentation reserve remains conservative after removal of Torch event
capture. Memory and profiler allocation are not a measured fit claim.

The unconditional full-page-residue envelope does **not** qualify for 104 GiB.
The conditional pilot checks absolute cgroup `memory.stat file` charge at the
existing 16 MiB hash-read boundaries, including before and after each block.
During legacy full authentication it must remain ≤ 8 GiB; after that hash and
before the snapshot/between arms it must be ≤ 4 GiB. The existing page advice is
the only reclamation mechanism. Failure to meet either bound refuses the pilot;
there is no global cache drop, added reclaim loop, retry or subset baseline.
The aggregate CaptureMemoryGuard and CUDA reservation cap are also enforced.

Full and selected arm deadlines are 1,800 and 900 seconds. PB and the harness
have a 3,600-second overall limit. These are refusal deadlines, not predictions
that source preparation will finish in that time.

## Evidence and interpretation

Each main phase emits cProfile, process-wide `/proc/self/io` before/after values,
wall interval, CUDA allocation/reservation observations, and guard state. Hashes
on existing background reader threads receive their own cProfile files. Source
events identify header opens, tensor/slice consumption, source file, logical
bytes and whether the read used a held descriptor alias. Hash records contain
fresh digest, logical file length and descriptor provenance. Final output seals
all produced profiler, source-event and Netdata files.

**Per-hash `/proc/self/io` intervals overlap across concurrent worker hashes and
cover the entire process. Do not sum them as exclusive per-file physical I/O.**
Main per-phase wall and process-I/O intervals are the A/B observations; logical
hash byte totals are a separate quantity. Observational wrappers and profiling
are present in both arms.

The shared screen evidence owner samples both Sparky and Sparklina continuously
and requires fresh, bracketed observations without excessive gaps for every
phase. CPU, pressure, memory, NFS/mount activity, GPU power and clocks are retained.
An endpoint error, stale chart or missing host coverage prevents success. GPU
utilization percentage is not used as saturation evidence. There is no Torch
event capture or kernel-attribution claim.

A single ordered pair cannot establish order-independent speed, cache warmth
independence, full-anchor throughput, or model quality/bpp/runtime tradeoffs.
The native result may establish only its measured source authentication and
source preparation deltas under the recorded environment. No such native delta
has been measured yet.

## CPU validation and audit

All actions ran CPU-only on PB's x86 worker at priority -10, with one CPU and
bounded native threads. Fixture actions reserved 8 GiB, metadata actions 4 GiB,
and compile 1 GiB. No CUDA operation or actual GLM source payload read occurred.

- `b5c3ed08a9bd`: 40 passed, A/B harness plus reviewed source-owner contracts.
- `f3943c9859b8`: 30 passed, final shared-owner harness plus screen evidence tests
  (14 A/B tests and 16 screen tests), no skips.
- `dac10ac21aa7`: compile passed for both A/B modules and the A/B test module.
- `2909e1917790`: final metadata resource plan passed; same conditional arithmetic
  as the initial `61500b06ffda` receipt, with instruments made explicit.
- `b79c60d7ce65`: earlier 12-test harness run passed.
- `69297a2aac17`: retained failed prototype, six tests passed and one fixture
  failed because its projected unit omitted `rows`/`cols`; the fixture was fixed.

The tests use a complete tiny capture and the actual source descriptor,
materialization, layer reader, snapshot input and source projection mechanisms;
only model construction is a small CPU fixture. They check unchanged identity
and manifest, all-shard versus consumed-shard hashes, held descriptors, byte
parity, teardown on refusal, exact tokens, missing/partial capture refusal before
CUDA, guard/deadline bounds, source event limits, and both-host telemetry gaps.
Final compile additionally includes explicit overlapping-I/O labels and final
artifact seals added after the 30-test run; these are reporting fields.

The external evidence directory is
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/selected-source-authentication-ab-01`.
Its `pb-audit.json` independently verifies seven terminal records, cleanup,
source CAS bytes, six successful result CAS blobs, canonical receipt hashes and
actual action commands. The failed fixture's logs are retained. Broker resource
credentials are excluded. `source-audit.json` compares compiled/tested snapshot
bytes to the review tree and confirms unchanged runtime. `files.json` seals the
retained bundle. [`native-plan.draft.json`](native-plan.draft.json) and
[`native-invocation.draft.json`](native-invocation.draft.json) carry the concrete
pending capture/freeze fields; neither has been submitted.
