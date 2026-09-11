# GLM TR3 final-panel full-vocabulary experiment — 2026-09-08

This is a separate experimental benchmark on the first sealed upstream final
panel: 25 windows of 2,048 tokens, 2,047 causal prediction rows each, across
four documents. It never resamples inputs, changes fitting capture, or feeds
final-panel measurements into the allocator. The existing gold v1/v2 tools
retain their distinct 8×512 contracts. No pricing package bytes change.

Input handoff SHA256: `35f0c5c973be614f29db757e9bd4bce407ea218b974a8407ec7e64c571aad72b`.
Dataset `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits`, revision
`95f4fdd94bf29989db2e0d1054e4931f55edb6aa`. All 25 token-file hashes and
NPY layouts were inspected against the handoff. Teacher logit files named in
the upstream manifest are absent from that published revision. This tool
emits its own explicitly attributed reference; it does not claim to recover
or reproduce unpublished upstream teacher bytes.

The reference is `zai-org/GLM-5.3-Flash-BF16` revision
`a6c167b62691b2bac901344b65cb651a70f53e43`, tokenizer.json SHA256
`19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`.
Root's independent upstream audit is bound by SHA256
`23370e25f6b42e316f72d6cbc8f3f5747a65705b388c910da93545f487edc0fa`.
It compares the accepted capture's 120 shard hashes with upstream LFS digests
and hashes six small immutable-revision downloads. It does not prove the
current local shard bodies are unchanged. A separate explicit CPU PB pass
uses the existing `build_source_checkpoint_identity` to authenticate local
bytes and produce its native mutation-sensitive digest cache. Later teacher
and candidate intake refuse incomplete/stale caches instead of silently
rehashing the checkpoint. Cache path spelling, machine, device and inode
must remain valid; historical hashes never receive invented fresh stats.

`build_glm_tr3_teacher.py` visits every ordered B=1 window once per resident
source layer through `visit_layer_batches`, retaining existing independent
per-window pass states. Existing StreamingContext owns two source cache slots
and one-layer lookahead, with prefetched residency required. The output
consumer emits raw FP32 logits `[2047,154880]` for each window. A final manifest
is written only after all windows and pre/post source, execution, tokenizer,
producer, derivative, input and upstream checks pass. Failed partial outputs
have no manifest. The bounded torch trace covers the first source layer over
all 25 windows; host telemetry must accompany an actual run.

The scorer matches upstream `token_kld_chunk`: cast raw logits to FP64,
normalize over the entire vocabulary in FP64, then sum
`exp(reference_logp) * (reference_logp - candidate_logp)` in FP64. Production
vocabulary work executes on co-resident CUDA tensors in bounded row tiles.
The copied NumPy calculation is a small CPU test oracle only.

`measure_glm_tr3_vllm.py` installs a non-mutating PyTorch forward hook through
stock vLLM's public `apply_model`. It does not patch vLLM core or transport
full vocabulary arrays through RPC. A single unchunked 2,048-token request
must yield one full 2,047-row prompt call and one sampled-token call. Partial
vocabulary, repeated/missing calls and ambiguous TP ownership refuse. Rank
zero alone preloads and scores the reference window; all TP ranks attest
call geometry. The hook's target-token log probabilities must match stock
vLLM prompt scores within 1e-4, proving causal row alignment. A one-window
native qualification binds candidate bytes, teacher, exact installed worker
source files, image, topology and experiment producer before whole-panel
replay is admitted. This native path has not yet run.

Outputs retain the 51,175 per-position KL values, per-window/domain/document
summaries and source/runtime provenance. Token positions are correlated within
four documents. There is no independent-token bootstrap, strong p-value or
broad generalization claim. A paired comparison still needs the same teacher
and input identities, matched measured bitrate, complete serving qualification
and declared performance workloads for both arms.

## Derived resource plan (not an observed fit or performance claim)

- Source shard files total 642,652,070,880 bytes. The explicit authentication
  pass reserves CPU1, 4 GiB host memory and no GPU on either eligible GB10;
  native threads are one. Its actual worker owns the reusable cache.
- The FP32 teacher arrays total 31,703,936,000 bytes (29.5266 GiB), plus small
  NPY headers and manifest. One array is 1,268,157,440 bytes. Source output is
  emitted one window at a time, never as 7.9 billion Python numbers.
- At B=1, the final raw logits are 634,388,480 bytes in BF16 or
  1,268,776,960 bytes in FP32 before omitting the final row. Current hidden
  states for all windows are 1,677,721,600 bytes at BF16 (twice that at FP32),
  using GLM hc_mult=4 and hidden_size=4096; pass-state/scratch are additional.
- Original safetensors headers show a maximum ordinary decoder layer payload
  of 14,825,277,272 bytes; layer 45 is an additional 14,865,185,408-byte MTP
  layer, while the text runtime visits 45 layers (0–44). Non-layer payloads
  are 3,664,816,128 bytes, including components the text profile may not load.
  The existing source cache stores device tensors and installation aliases them;
  it is not a second CPU cache or a duplicate installed layer. Two retained
  cache slots plus one in-flight prefetch conservatively allow a
  44,475,831,816-byte payload component before packing temporaries, persistent
  state, activations and allocator overhead. The runner writes its actual
  estimated layer size, cache limits, prefetch floor and pre-window allocated/
  reserved CUDA bytes before traversal. The existing source initialization
  audit must cover the complete streamed forward before the final manifest. Proposed teacher admission is
  CPU4, 96 GiB shared RAM and an 88 GiB GPU subset, subject to root review and
  current host telemetry. This is a budget, not proof of a native fit.
- Scoring holds one 1.268-GB teacher window on the TP owner, the engine's full
  prompt logits, and bounded FP64 tiles. Each 32×154880 FP64 tile is
  39,649,280 bytes. Autograd is disabled; normalization and reduction need
  several simultaneous tiles. Teacher-file authentication/preload is outside
  the inference request and temporarily holds bounded CPU file/array buffers.
  Candidate weights, KV cache and runtime scratch dominate total admission.

## CPU evidence

Initial PB action `ae34d52a6fc412146fbee0628b7d9c8a35bb8c967e21dd0720350e52c3376c99`
passed 39 tests, zero skips, 56 Torch deprecation warnings, in 6.90 seconds.
PB used dl380g10, four pytest workers with its default two native threads
(CPU8 total), 12 GiB reservation and 1,417,412,608 peak cgroup bytes. The
actual terminal, canonical CAS receipt, result bytes and snapshot file bytes
were independently verified in `initial-cpu-audit.json`. The first submission
attempt rejected unsupported `pbtest --pytest-args ["-q"]` before execution;
removing that display-only option produced the recorded run. Native kernel,
whole-model, source-authentication and GPU fit claims require their own receipts.

## Integration and launch evidence added 23:15 UTC

The broader CPU integration action `29761463f3fd2e606da38b90d814d8c0d221cddada8e6936156b8da67e191121`
compiled five files and passed 138 tests, zero skips, in 65.74 seconds on
dl380g10 with CPU4/native1 and 12 GiB. `integration-cpu-audit.json` records
the actual source bundle, result and canonical CAS receipt.

The later observer action `8da2e17659665c19c9e0ab4b4f7032c9607d9b4e976fdec698797e6002df0092`
first reproduced the missing worker-local KV observation against the earlier
source snapshot (one expected failure), then compiled the touched modules
and passed 57 current tests, zero skips, in 7.35 seconds. The scorer now uses
public `collective_rpc` to check every actual worker and model runner cache
configuration, in addition to the coordinator configuration. A silent worker
promotion refuses. Tests also verify that the teacher manifest is withheld
if the both-box sampler fails during shutdown. `observer-cpu-audit.json`
authenticates the CAS receipt, result and all five current source files.

Source authentication completed through PB: BF16 action
`d6642c3bd4714caaeef099275fce1e7fae70c4b3d9defdb9398b060c2814ac23`
read all 642,652,070,880 shard-file bytes on Sparklina and matched the complete
upstream roster; EXL3 action
`59e8df2a5d917ff3adcf749d4184a758420f1e70658f0b42690bcf4eb4c5e170`
read all 175,642,157,752 shard-file bytes on Sparky and matched all 120
SHA256SUMS entries. Each used CPU1, native1 and 4 GiB, no GPU. Their native
digest caches bind `/source` on Sparklina and `/model` on Sparky respectively.
Actual receipts and source/cache artifacts are retained under shared census
directories `tr3-source-auth-03` and `tr3-candidate-auth-01`. Earlier source
authentication launch attempts failed before hashing because a host-local
image ID was unavailable on the other node and then because the container
command needed `python3`; the working command uses the content-qualified
existing image and records its actual local ID.

The finalized teacher reservation is CPU6/native1, 96 GiB shared memory and
an 88 GiB GPU subset. Four layer-reader threads plus the main/observer work
explain the CPU change from the initial proposal. The conservative derived
GPU ledger is 70.959 GiB before allocator slack: three source layers,
9,663,676,416 bytes of gate/up packing destination, non-layer weights,
all 25 FP32 hidden states and 14 GiB of original eager KDA scratch allowance.
Two GiB of host observer allowance remains within the shared reservation.
`max_cache_slots=2` is passed to `LayerCache(max_entries=2)` by the existing
constructor; the larger byte limit does not permit extra retained entries.

The actual teacher action
`90a36b449ef47cf5fbeb967400a59a7468281d661b05ac5bc60f3dac77ce4d2e`
was admitted on Sparklina and was still running when this entry was written.
Output is `/mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-04`.
The exact reviewed invocation, explicit JSON-null source policy, byte-preserved
panel and retry records are in adjacent `tr3-teacher-inputs-01`. Three earlier
teacher attempts refused before a forward: missing container Git provenance,
the worker's own untracked stdout artifact, and an unwritable default staging
directory. The launcher now validates and excludes only the exact single
PB stdout artifact from Git status, rechecks a clean source snapshot and
passes its actual HEAD through the existing provenance overrides; staging is
explicitly `/out/tmp`. No source hashes receive fabricated filesystem stats.

The teacher reuses `CaptureObserver`: both-box Netdata every five seconds
with its existing 3-GiB disk cap, and one-second main-thread stack/IO samples.
Synchronous samples bracket source initialization and traversal. The final
teacher manifest is published only after observer shutdown succeeds, with
the hashes of the complete telemetry files. A first-layer torch trace and
live both-box samples exist; completed teacher bytes, native hook qualification
and any comparative quality/performance result remain unclaimed.

The resolved-KV follow-up action
`f6784b2e29c1fa0a227a12453d46a06ef51abe4d81660eff245bd1ccf5319a7a`
passed all 48 experiment tests, zero skips, in 8.19 seconds on dl380g10
(CPU4/native1, 12 GiB). The coordinator may retain the requested dtype when
promotion occurs inside workers; each worker must still show the explicitly
declared resolved dtype. Attention receipts retain actual allocated cache
tensor dtype/shape separately from the model runner's dtype assigned before
model construction. The exact pinned EXL3 image's `mla_attention.py` defines
the auto-to-fp8_ds_mla transition for FLASHINFER_MLA_SPARSE_SM120 at lines
349–359 and mutates the worker cache config at lines 468–474. This source-based
expectation does not replace the pending native qualification.
`resolved-kv-cpu-audit.json` verifies the actual CAS receipt/result and all
five current implementation/test files in snapshot `8426a765b2689de9c956fbd752694ea361d9ddfd`.

## Completed teacher evidence added 23:29 UTC

Teacher action `90a36b449ef47cf5fbeb967400a59a7468281d661b05ac5bc60f3dac77ce4d2e`
completed with return code zero and confirmed resource cleanup. Action wall
time was 986.594 seconds; the builder's traversal/emission timer was 970.654
seconds. The original BF16 source initialization contract completed all 45
layers and 1,262 persistent text-source tensors, with zero derived buffers.
The final manifest is
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-04/artifact/teacher.json`,
SHA256 `1cc798a32a3457f996e859f778fe61fd987561b91490fe2953b698457ea747ae`.
Its source checkpoint identity equals the independently authenticated original.

`teacher-completion-audit.json` independently verifies the terminal return
code/cleanup, canonical CAS receipt and actual log artifact, the source bundle,
all nine gold-tool closure files, all five experiment/observer files and all
231 PrismaQuant package files. The full package hash is
`5a507741716b7a7a7f50dd8f650852e2f5b910d245e9325244b9ccf736c64a53`.
The producing snapshot is `7882eda3a87fd6a45a1142de140aa17c49a8021e`.
The teacher's helper/scorer file predates subsequent worker-KV observer
hardening; the builder and core panel/logits code are unchanged.

A separate portable CPU1/2-GiB PB action
`ec8d349dc358d06f6dd56f716352c4b1539f5764653f117fa7272c6076a3d0ad`
ran on dl380g10 and reread all 25 array bodies in 101.486 seconds. Every
SHA256 and NPY header matched: little-endian FP32, C order, `[2047,154880]`.
Total file bytes are 31,703,939,200, including 25 128-byte headers. The
independent action's canonical CAS receipt and actual output are verified
in `teacher-array-cas-audit.json`; this was an artifact read, not another
model forward.

The existing host observer completed with zero errors and retained 194
Netdata samples per host plus 945 main-thread stack/IO samples. Both hosts'
maximum sampling gap was below 6.883 seconds. On the source host, the 99
distinct power updates span 980 seconds: 24.031 W time-weighted gross mean,
44 W peak and approximately 23,550 J by trapezoidal integration. These are
whole-device readings and do not establish a GPU-bound teacher path. Sparky
carried separate qualification work during this interval. The GB10 Netdata
framebuffer chart has empty dimensions, so it supplies no GPU memory estimate.
`teacher-host-telemetry-summary.json` retains those limitations and values.

The first-layer trace is 136,280,505 bytes and parses into 377,938 events,
including 36,491 CUDA kernels and 1,227 GPU copy events. Actual FP32
elementwise multiply, exponential and reduction kernels carry the largest
accumulated durations. `teacher-first-layer-profile-summary.json` binds the
trace hash and its top event totals. There is no before/after speed claim.

The verified teacher is ready for the native EXL3 hook qualification. A clean,
self-contained scorer checkout is retained at adjacent `tr3-score-source-01`,
commit `38ef485559a70d0d3e15f5fce620ae6976321f6b`. The sealed adapter and
argument hashes are in `tr3-exl3-offline-adapter-01/handoff.json`. No native
EXL3 hook or complete paired quality score has run as of this entry.

## Native V2 logits layout repair — 2026-09-09

Native attempt 05 reached model execution after the runtime Git and callback
serialization repairs, then the strict legacy hook refused its first 1024-row
prompt-logit chunk. The actual pinned V2 runner samples one token first, calls
`gpu/sample/prompt_logprob.py::compute_prompt_logprobs_with_chunking` over all
2048 prompt hidden states in two fixed 1024-row chunks, and omits the final
prompt score from the returned prompt logprobs. Scheduler chunked prefill was
observed disabled. Logits tiling and scheduler prefill chunking are separate.

The experimental scorer now exposes explicit
`--logits-layout vllm_v2_chunk1024`; the legacy single-prompt-call default is
retained. The new path requires exactly the ordered sample `[1,V]`, prompt
`[1024,V]`, prompt `[1024,V]` calls. It scores rows 0 through 2046, excludes the
extra final prompt row, retains the complete resident teacher on rank zero,
and checks native target log probabilities at all 2047 causal positions.
Unknown layouts, partial vocabulary/chunks, missing/extra calls and TP layout
mismatches refuse. The native V2 runner and prompt-worker classes and source
hashes join the qualification binding. This changes only the experimental
instrument; teacher bytes, fitting capture and the pricing package are fixed.

The before/after CPU driver replay uses the exact native chunk function body
(SHA256 of its original file:
`4cf22d390e7bf59e44c458cff7180268db63e95d06af2f1e14d720a2d77c9663`),
with a CPU top-k stand-in, in the pinned serving image. The old hook refuses
at `[1024,11]`; the repaired hook observes `[1,11], [1024,11], [1024,11]`,
returns exactly 2047 positions, and matches independent FP64 full-vocabulary
KL and target-logprob oracles. This replay does not measure native GPU logits
or certify hook qualification. Evidence remains at
`/home/rob/tmp/tr3-native-layout-01/{probe.py,prompt_logprob.py,before.log,after.log}`.
The actual native repaired-layout qualification and full-panel score are
still pending at this checkpoint.

### Initialized runtime observations

A separate content-addressed `*.runtime-<sha256>.json` now records the actual
worker/runtime binding before the first scored window or qualification replay
comparison. It retains raw allocated KV tensor shapes without normalizing them.
Its schema marks `initialized_before_scoring` with zero scored windows and has
no success flag; it is diagnostic evidence, never a hook qualification or KL
result. This preserves the exact initialized state when a native forward or
strict replay binding subsequently refuses. Distinct runtime bindings produce
distinct filenames, so a later attempt does not erase earlier observations.
The qualification comparison remains exact; no KV capacity exception is added.


### Native qualification and restart-capacity repair — 2026-09-09

Attempt 07 completed the native hook qualification on frozen source
`7d9992b85783d5ae0a490f3f1be2a520b4f56606`. The actual head exited 0 without
OOM at 01:00:29Z. Its 300,416-byte result has SHA256
`04a602b9e98d83b9f9bb89c4912e5cfdd70d1b2ae73474b1654913627f03c35f`.
Both TP ranks observed `[1,154880], [1024,154880], [1024,154880]`; all 2047
causal positions aligned with native prompt scores to 2.3839675122871995e-6,
below 1e-4. EXL3 grouped prefill calls increased 42 to 84 on each rank.
The one-window mean full-vocabulary KL was 0.047159120783769146. This is
hook qualification evidence, not a complete-panel baseline or a paired win.
The sealed result's interpretation still says four documents; qualification
actually covers one window from one document. The development branch's
separate prose correction says “within the reported documents” instead,
without rewriting any measured artifact.

Attempt 08 reloaded the same source, teacher, candidate and configuration for
the complete panel. It exited 1 without OOM at 01:10:05Z before scoring because
the exact qualification comparison included cache capacity. Comparing the
raw initialized observations finds exactly 66 changes, all in
`allocated_kv_cache.shape[0]`: 12441→12532, 957→964 or 49764→50128, 22 each
across both ranks. No other runtime-binding field changed. Byte-identical
observations are now regression fixtures in `tests/fixtures/glm_tr3_runtime`.

The comparison now ignores only that positive integer block-count dimension
for the three observed backends whose pinned `get_kv_cache_shape` source
explicitly places `num_blocks` first: `DeepseekV32IndexerBackend`,
`KpoolTailBackend` and `FlashInferMLASparseSM120Backend`. Every remaining
field compares exactly, including all source/teacher/candidate identities,
backend/module, cache dtype/device and invariant tensor dimensions. Invalid
geometry refuses and unknown backend layouts receive no exception. Raw
observations and result bindings retain every original allocated dimension;
only copied comparison data replaces the capacity axis. New scorer source
requires a fresh native qualification; attempt 07 cannot qualify changed code.

The real-observation regression failed with the previous equality under PB
`1f751f7fdfdab62bad90f57952b06cbb53545b1b77b5fd50dbc1f2ec9a67b51c`.
After repair, PB
`4f7c6b4b95fdf68fec16ec840aee286e672eaa91825bcf425923abd5a7f4d25d`
passed 79 CPU tests in 8.64s, including 19 negative geometry/provenance cases,
then compiled both scorer modules. Actual terminal records, cleanup and CAS
payload hashes were checked; `native-runtime-capacity-cpu-audit.json` records
the evidence. Both runs used the scoped x86 CPU environment on DL380 with
native threads bounded to 1. Fresh native qualification and full 25-window scoring
remain pending at this checkpoint. No teacher or production package bytes
changed, and no speed claim follows from this comparison repair.

### Completion evidence and PR review — 2026-09-09

The preceding pending checkpoint is superseded by retained native results on
frozen source `ddc9aac80cda44c519d947b8468a0577a3ec17cb`. Qualification09 and
the subsequent full25 launch both report head exit 0. The full result contains
25 windows and all 51,175 finite causal KL values, with mean
**0.02446462473923544**. Both TP ranks report the expected 1+1024+1024 call
geometry on every window. Maximum alignment error against native prompt
logprobs is 5.9569720178842545e-6, below 1e-4. Qualification/full runtime
bindings differ in exactly 66 positive KV block capacities and nothing else.
`native-completion-review.json` records the artifact/log hashes, recomputed
summary, listed source-file hash verification against ddc9aac8 and provenance.
The logs retain forced engine shutdown and leaked IPC warnings; successful
result production does not establish graceful runtime teardown.

This is a measured full-panel result, **not yet a stable quality comparator**.
[Issue #468](https://github.com/RobTand/prismaquant/issues/468) tracks the
unresolved first-window divergence. A later diagnostic on source
`632e2324349bb122cbf25514ee6ecdee8126c91e` repeats identical final-0000 four
times in one engine and obtains means 0.0462136121, 0.0472286403,
0.0573906394 and 0.0569008298. Native prompt alignment still passes. This
rules out treating alignment alone as evidence of repeatability; no root cause
or measured noise floor follows. The prepared repeatability04 run has only a
successful preflight ending “nothing was launched,” with no result. The
research instrument can be retained while #468 remains open; these results
do not establish a paired Tessera win or production admission.

The reviewer-requested failure diagnostics now report up to eight deterministic
JSON paths after the existing capacity normalization, plus explicit envelope
and invalid-shape errors. Known-capacity acceptance and all other comparisons
are preserved. The four new diagnostic regressions failed on the prior source;
the repaired scorer passed all **83 CPU tests**, without skips, and compiled
under PB `cffb2862274320f5f085614e24514bb14e28206d8831b7e1bb3109821554757f`.
The action used the existing x86 Python 3.12 environment, four CPUs, 6 GiB,
native threads 1 and priority -10. Terminal cleanup, CAS receipt and payload,
and exact tested scorer/test bytes were verified; see
`native-runtime-diagnostics-cpu-audit.json`. The first generic CPU environment
attempt could not collect tests because `compressed_tensors` was absent.

Review of the teacher builder and panel loader found no additional defect:
they preserve sealed panel/array identity, authenticated reference inputs,
resident streaming, complete-manifest publication and explicit source policy.
No new native execution was performed for this diagnostic edit. The changed
source revision and integration with main require fresh qualification before
subsequent scoring; the historical native receipts qualify only ddc9aac8.
