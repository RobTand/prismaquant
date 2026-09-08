# Original GLM prefix graph gate: implementation and CPU preflight

Date: 2026-09-08. Helper source: `62e7f09d5dc661c175e57fe5ad7a1c94983bdfd5`.
The native invocation is frozen for root review and **has not been submitted**.
No native graph, full statistics, PWC, cost, quality, serving, throughput, or
full-model fit result is established by these CPU checks.

The helper extends the existing streamed layer visitor, exact boundary owner,
isolated-layer call, shared cotangent fork, source cache and prefetch worker.
It does not replace model kernels or create another source cache. The original
GLM config and all 45 layers remain intact; a dedicated completion exception
stops the visitor after source layers 0 through 4 have processed original sealed
token rows 0 and 511. Layers 0, 3 and 4 each run four deterministic boundary
stimuli through isolated baseline, disposable fork and final original-owner
arms: 72 backwards. These stimuli are BF16 plus/minus 1/256, generated from
local CPU generators with seeds 7000 through 7003; they are not downstream
Fisher adjoints.

The gate compares complete replay output bytes against each original prefix
output and complete input-cotangent bytes against the isolated baseline. It
checks input, module parameter pointers/versions/sampled values, buffer state,
source metadata, empty GLM pass state and Torch RNG preservation. Hooks on the
observed arms record both HC4 sites, the dense width-12288 path, the router's
actual 512-by-8 IDs and 4096 assignments, and the 512-token shared MLP. They do
not require all 288 experts to receive a token. The baseline has no activity
hooks. Original decorated KDA/DSA/conv/expert functions remain in place.

Before that graph run, the helper independently executes the existing
`_prepare` on all 512 original token rows, releases each expanded hidden
immediately, retains only ordinary metadata/pass state under the existing
2-GiB auxiliary cap with K=4 accounting, and verifies metadata-owner release.
This is a metadata-residency check, not 512 original graph forwards.

## Authenticated source scope

The admitted CPU preflight authenticated `config.json` (5,939 bytes) and
`model.safetensors.index.json` (4,083,163 bytes), plus the small sealed token
file and lead manifest. The existing profile's checkpoint mapping derives
**3,103 consumed tensor keys in 14 payload shards, totaling 73,880,076,968
bytes**, covering fixed tensors and layers 0 through 5. Layer 5 is the existing
forward lookahead. The earlier 21-file list also included five irrelevant
tokenizer/generation files; those are excluded. CPU preflight hashes no
original payload shard.

Native preflight will hash these 14 payloads exactly once against their
expected manifest digests, using sequential buffers of at most 8 MiB and page
release. The same O_RDONLY descriptors remain held through source consumption
and prefetch shutdown. Both safetensors and ordinary `open`/`os.open` readers
bind to those authenticated descriptors. Repeated path/fstat checks detect
replacement or modification; those checks are freshness evidence, not content
authentication. Unexpected files, writable opens and out-of-roster tensor
reads fail and leave a sticky violation.

The existing all-layer source-capacity estimator still inspects headers across
the full index. Unselected shards expose only bounded header metadata through
an adapter that cannot return tensor payloads. Their observed header digests
and descriptor identities are recorded separately; they are not represented
as authenticated original payloads.

## CPU validation

Both final actions used PB on dl380g10, two CPUs and 4 GiB each, with native
math threads bounded to one and GPU visibility disabled. The committed helper
and tests were frozen before submission. Compile checks and **15 tests passed
with no skips or xfails**; 14 warnings are the existing Torch JIT deprecation.
The second action successfully preflighted the actual original source and
sealed calibration file. Both actions returned zero, completed scope cleanup
without OOM, and passed payload hashing, CAS claim and source-closure checks.

| Action | PB key | Result |
| --- | --- | --- |
| Compile and behavioral tests | `91b071990de0bfd45110debcb6ee8114af7b193860c96bb9c645a6deedce3e4e` | 15 passed in 4.75 seconds |
| Real-input CPU preflight | `7d7fe77ee939e4ec06ae9f5195853e480c323a84219c83dc6da0d2fda5ee07c2` | Correct roster, small-input hashes, descriptors closed |

Behavioral tests exercise a real existing `_read_layer_to_device` read from
an authenticated tiny safetensors file; descriptor identity, bounded hashing
and hash reuse; CPU roster binding without payload reads; same-size mutation,
path replacement and wrong digest refusals; sticky writable/unauthenticated
read refusals; header-only payload denial; profile-based roster selection;
actual CPU backward under outer `no_grad`; inference-mode, RNG and module
mutation rejection; transparent return identity and recursive call routing;
failure propagation, exact completion scheduling and source cleanup order.

The first real-input preflight failed because the helper used HF's live config
alias `num_local_experts` when inspecting raw JSON, whose authenticated field
is `n_routed_experts`. The helper was corrected to the actual raw field and
the preflight passed. This was an experiment-helper defect, not a production
runtime change. An initial receipt audit encountered transient CAS path
visibility; subsequent explicit audits verified both actual receipts and
payloads successfully. The earlier failed run is retained under PB key
`b367322264ffa4a53808c97db1f852fa9141cde49c551b4d62bc90c498e35f32`.

## Frozen native invocation and limits

`native-invocation-01.json` records the exact argv and 13 consumed source-file
hashes. The request is portable GB10 placement with six CPUs, four source
reader threads, one native math thread, 104-GiB physical admission, 92-GiB GPU
subset, measurement isolation and an 1800-second deadline. It uses
`prismaquant-glm-producer:content-qualified-20260908` with content digest
`eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd`, not a
host-specific image ID. The original model mount is read-only. Source cache
slots/workers/lookahead are 2/1/1 with 24-GiB source headroom/free floor. The
existing physical guard also checks before actual source reads and reserves
16 GiB of graph workspace after the forward lookahead settles.

Nine raw Torch profiles are planned, one per arm on the first row/stimulus of
each measured layer. A 100-ms bounded memory ring, CUDA peaks and required
one-second Netdata samples from both GB10 hosts accompany them. These profiles
qualify graph/resource behavior; cold compilation and instrumentation differ
between phases, so they do not establish a throughput or energy ranking.

Source futures are joined while descriptor binding remains active, then the
existing source cache is cleared, observer owners are released and source
storage weak references are checked. Runtime failures propagate; only the
dedicated completed-schedule exception is treated as normal prefix termination.

The remaining graph-plus-full-statistics envelope in the design remains
conditional: this invocation allocates no 32-GiB statistics window or PWC
candidate. Empty original GLM pass state also does not qualify nonempty shared
KV state. No default, serving lane or production architecture contract changes.

Evidence files in this directory include the final CPU campaign, two audited
PB receipts, compact actual-input summary and frozen invocation. Full inputs,
output and audit logs live under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/joint-original-graph-implementation-01/`.

## Update at 14:59 UTC: first native attempt failed; helper corrected

Following root review, the exact first invocation ran as PB action
`5dac548ddf0844d62873aefb9201bbc42e72a85a07a9615e3b50cd609a30c17a` on
Sparklina. It returned 1 after a 160.9-second action window. The failure
happened before the first original layer forward, with zero backwards and
no replay profiler traces. PB completed scope cleanup with zero OOM kills.
This is negative harness evidence, not original-model graph qualification.

The attempt authenticated all 14 selected payloads and the two metadata
files against their content digests: 73,884,166,070 total bytes. All descriptor
owners closed with no source-read violation. The actual 512-row metadata gate
completed, retaining a peak 272,891,904 auxiliary bytes, reserving zero shared
cotangent bytes for the empty original GLM state, and releasing its metadata
owners. Both-host Netdata contains 148 samples per host. The original helper's
poisoned-CUDA cleanup suppressed the in-process memory ring and final CUDA
peaks; those measurements are unavailable for this attempt.

The defect was in experiment instrumentation: `torch.linspace(...).long()`
created sampling indices in FP32. For the actual dense weight's 50,331,648
elements, endpoint 50,331,647 rounded up to 50,331,648. CUDA rejected the
out-of-range gather. A CPU regression using the actual 12288-by-4096 BF16
storage geometry reproduced precisely the same out-of-bounds index before
the fix in PB action
`68d1d4cece92a626b0747c17d4ed397918574171644e6bc83c153142bb80f51d`.

Commit `dd910b9e1c` constructs the 64 coordinates with exact integer
arithmetic and transfers them as int64. Separate commit `1be1d9eec5` preserves
the original exception through cleanup failures, releases observer references
in a finally block, attempts source cleanup once, always stops/joins telemetry,
and records unavailable CUDA measurements as separate cleanup errors. A
successful qualification still refuses any cleanup error or retained source
owner. Source-authentication, metadata and per-layer progress records are now
written outside the measured replay profiles. Exact-byte checks and resource
limits remain unchanged.

The revised committed code passed compile checks and **18 CPU tests** in
5.92 seconds, without skips or xfails, under PB action
`125d4019ab850a9d857fb49267f13aa2dc74b949020340047d4545b9e26a5b09`.
The actual-input CPU preflight also passed again under
`17b50b2e0e857d67edc2608b0f2edbcfc4574deb4a60b586472e833171d8b9c6`.
Both revised actions have independently checked zero exit status, completed
scope cleanup, CAS payload/claims and source-closure evidence. The added tests
cover original-size integer sampling, observer release when cache cleanup
raises, and preservation of the first error plus host observations when CUDA
cleanup fails.

`native-invocation-02.json` freezes source
`1be1d9eec5` with the same admission, image, source descriptors, geometry,
stimuli and comparisons; only the corrected helper and new output directory
change. At this update it awaits root review and has not been submitted.
`native-negative-audit-01.json` records the first attempt's source parent
(`2bdec238606d63d659662898c1df46954af60770`) and actual PB snapshot
(`e3c40be969bac23bf18b4e045c9c4f819b0216c9`), all 13 original frozen
file-hash checks, terminal evidence, complete metadata result and Netdata
identities. The raw failed result and log remain under the shared run directory.

## Update: second native attempt reached backward and failed its cotangent gate

After revised review, PB action
`d3afc4eea473d32e7451a425f041ac1b2373399bdb75006054d6279db3c1f32e`
executed native invocation 02 on Sparklina and returned 1 after 87.7 seconds.
The original layer-0 primary forward and first isolated baseline
forward/backward ran. The combined `absent/nonfinite/zero` input-cotangent
predicate refused that backward, so zero backwards were accepted. That
predicate did not record which category failed. Output checksums alone had
not established output finiteness. Neither the full prefix nor its source
ownership gate passed.

The revised cleanup preserved the original failure, the memory ring and final
CUDA statistics, without reporting cleanup errors. Peak CUDA allocated and
reserved bytes were 11,080,916,992 and 11,704,205,312 respectively. Source weak
owners remained live in the failed exception stack; that is **not** successful
in-process ownership qualification. PB independently completed process/scope
cleanup without OOM kills. All 16 source files authenticated and descriptors
closed without violations. The 512-row metadata result remained
272,891,904 bytes with expired metadata owners. Both-host Netdata has 76
samples per host.

The first-baseline raw trace is 10,797,411 bytes with SHA256
`69a0cf6fe1ec8f79c1d036f497008fd0c077cdb0cc45e5745e041207a596be4f`.
Admitted CPU trace analysis (`e799e86a8524`) counted 31,804 events, including
3,256 kernels with 427.139 ms of summed duration. A separately audited full
name inventory (`f9bc428fd357`) found 88 kernel names, none matching
`kda`, `delta`, `chunk`, `fla`, `causal` or `triton`. CPU events include 24
`ExpBackward0`, nine `MaskedFillBackward0` and one `CumsumBackward0` calls.
This is dispatch evidence consistent with the stock PyTorch fallback
arithmetic; kernel names do not identify failing numerical values.

A source-based hypothesis is that the stock chunk KDA fallback exponentiates
the full cumulative-gate difference matrix before causal masking. Large
upper-triangular differences could overflow while subsequent masking keeps
the forward finite, then `0 * Inf` could contaminate backward. The observations
above do not yet establish that this happened in the original run. No kernel
substitution, reference implementation or weaker stimulus has been used to
turn the failure into a pass.

## Frozen one-backward diagnostic

Commit `aa46075ce824ef8629824e5d07a1e580a9654e84` adds scalar presence,
finite/nonfinite/NaN/positive-Inf/negative-Inf/zero counts and finite extrema
for primary output, replay output and leaf cotangent before refusal. Primary
and replay output identities are retained, and their exact equality is
checked before backward. Optional diagnostic observers attach transparent
tensor gradient hooks to original attention, HC and MLP inputs/outputs during
the same actual backward. Module forward observers and tensor hooks both
return None; there are no full-module backward hooks or replacement views.

The explicit diagnostic mode selects only original layer 0, row 0, seed 7000
and the isolated baseline arm. It skips the already measured 512-row metadata
phase. The original 45-layer config is unchanged, and the existing cache still
prefetches layer 1. Re-derived consumption is **402 keys in two payload files,
9,531,639,384 bytes**, plus the authenticated config and index. Payloads will
again be authenticated inside held descriptors on the native worker; CPU
preflight reads none of their payload bytes. The 104/92-GiB admission,
16-GiB graph-workspace reserve, image, source reader mechanism and BF16
plus/minus-1/256 stimulus remain unchanged.

Compile checks and **25 CPU tests passed** in 5.27 seconds under PB action
`d7096b059aaff26d2b84875db6bf768716b87edfc7e233854014c8b632791c80`, with
no skips or xfails. Actual diagnostic-input preflight passed under
`b2739f6322304663fdd1b5479d901aa0ce55f8aa4ae225b69a4478b7b6d552a6`.
Both have independently audited zero status, completed scope cleanup,
canonical CAS/source receipts and closure-only snapshot changes. The new
tests distinguish all three leaf-failure categories, reject nonfinite primary
output before replay, verify one-case scheduling, and check that transparent
tensor observers preserve actual CPU output and gradient bytes.

`native-invocation-diagnostic-01.json` is frozen for root review and has not
been submitted at this update. A successful diagnostic would still be labeled
diagnostic-only, not full original graph qualification. The second failed
attempt's source parent is `f7b617024f15727edb4b3c7308f757f0b4bc365b` and its
actual PB snapshot is `dddf8c771a8acdd272c8c25703b062375485df79`;
`native-negative-audit-02.json` verifies all 13 frozen source hashes and records
the retained negative evidence.

The frozen diagnostic invocation also records bounds derived from the pinned
layer-0 shapes: the largest observed boundary is 8,388,608 BF16 elements
([1,512,4,4096], 16 MiB); the FP32 forget-gate output has 4,194,304 elements
([1,512,64,128], also 16 MiB). No selected branch tensor is larger. Each
boolean mask is at most 8 MiB and each same-dtype `where` result at most
16 MiB; a full identity's CPU transfer plus byte digest buffer is at most
32 MiB. These are per-tensor bounds, not a measured combined temporary peak.
Operator-internal temporaries remain charged within the existing 16-GiB
workspace reserve. Only scalar summaries and hashes persist after observation.
