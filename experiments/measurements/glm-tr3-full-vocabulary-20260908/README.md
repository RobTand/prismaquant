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
