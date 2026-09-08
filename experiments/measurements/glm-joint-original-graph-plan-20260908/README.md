# Original GLM prefix graph gate — proposed, 2026-09-08

**Design only; no helper implemented and no native run authorized by this
artifact.** The next smallest useful gate is a source prefix through layers
0–4 on **original calibration row indices 0 and 511**, each B1 × 512 tokens.
Measure original layers **0, 3 and 4**, with four deterministic incoming
boundary stimuli and exact isolated-replay parity. This covers the widest
original dense MLP, DSA plus routed/shared experts, KDA plus routed/shared
experts, and the original HC4 graph. A separate metadata-only phase constructs
the actual preparation metadata for all 512 original rows, without running
512 source graphs or collecting calibration X/H.

This gate complements the passed planned-size statistics fixture. It does
not repeat that fixture, allocate 32 GiB of statistics, read PWC donors, run
anchors, or produce costs. A pass qualifies only these original prefix graphs
and the measured metadata preparation. It cannot certify maxima across all
45 layers or all 512 graph inputs, coupled statistics/graph memory, original
downstream Fisher adjoints, source quality, or serving performance.

## Inspected implementation and immutable inputs

Runtime contracts were read at `34388d2bfeea0e76830645aab92d15132f766bcf`:

- `cost_streaming.py:538` prepares original embeddings, masks, position data
  and HC expansion; `:555` calls the installed layer; `:623` implements
  `isolated_layer`; `:698` implements the existing layer-major visitor.
  That visitor uses `torch.no_grad()`, not inference mode, so a narrowly
  nested `torch.enable_grad()` is legal. The new helper must reject
  `torch.is_inference_mode_enabled()` and inference tensors before replay.
- `layer_streaming.py:2274` is the common layer call. At `:2353` it passes
  `past_key_values=None` and `use_cache=False` in this runner. The helper
  observes these actual arguments rather than merely trusting config.
- `streaming_model.py:855` installs the cache alias; `:912` settles the exact
  prefetch roster; `:941` describes installed/cache/future ownership without
  retaining tensors. The exact visitor suppresses adaptive top-up and schedules
  one explicit forward successor. All graph replay stays within that current
  installed source; no source transition is allowed inside an arm.
- `aura_cost.py:3003` defines production replay's `fork_for_replay`, isolated
  pass state, `graft`, produced roots, backward and `harvest` ordering. The
  helper exercises those existing objects in that order. It does not claim to
  execute the enclosing full cost runner or its candidate projection path.
- `ModelProfile` defaults at `model_profiles/base.py:1334` return `{}`, `None`,
  `{}` for new/captured/isolated pass state. `Glm5NextProfile` does not override
  them. GLM's shared experts are an MLP branch, not a shared KV container.
  The actual config has only `indexer_types="full"`; no previous DSA indexer
  output must be supplied. Assert these original contracts at runtime.

The original modeling source was copied read-only from an **unstarted**
disposable container made from the pinned producer image, then the container
was removed. No process or GPU code ran. The 102,403-byte file's SHA-256 is
`2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b`.
It is retained under shared `joint-original-graph-plan-01/` as
`modeling_glm5_next.pinned-image.py`; it is inspection evidence, not vendored
runtime code. In that file:

- `:628–734`: KDA convolution/recurrent cache updates require a non-null
  `cache_params`; B1 T512 with no cache takes full convolution and chunk KDA,
  with `initial_state=None` and `output_final_state=False`.
- `:1178` and `:807`: DSA/its indexer update caches only with non-null
  `past_key_values`. The original all-full indexer schedule avoids the
  `prev_topk_indices` dependency of shared-indexer configurations.
- `:1280–1329`: both hyperconnection mixers and the full original MLP execute.
- `:120–140`, `:158–185`, `:200–206`: packed expert dispatch, top-8 router and
  the shared MLP branch execute. Their decorators may select an installed
  backend, so native receipts must identify the *actual* bound implementations
  and kernels. No `.__wrapped__` substitution, test fixture kernel patch,
  expert replacement, dimension reduction or config truncation is allowed.

The immutable model is `/mnt/shared/models/GLM-5.3-Flash-BF16`.
Config SHA-256 is `33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943`;
index SHA-256 is `e6007bd58fb7e07f9fe69544257ee2713f252ef5855bbf685b48c991d524ef0f`.
Existing prefix source file manifest
`/mnt/shared/tessera-measurements/glm-streaming-source-20260907/workspace-prefix-inputs-03/source-files.json`
has SHA-256 `f05c8d55f091e2fa4a2663f8b983f3137e94d0f153648b35c336b8e4ecea082c`:
21 files, 73,904,396,020 bytes, covering fixed tensors and prefix/lookahead
shards. Every stored inode/length/mtime/ctime matched when inspected for this
design. CPU preflight must rederive the exact consumed shard roster from the
index/profile, including lookahead layer 5 and fixed tensors, check coverage,
and refuse changed fingerprints. It must not silently trust an old manifest
or reread all model shards when the existing content binding remains valid.

Use `calibration_data.load_calibration_input` on
`exact-calibration-input-01/calibration_tokens.safetensors` under the canonical
measurement root. Its complete-file SHA-256 is
`9cd1fa129f249abd80d22efaeb8bc7e8b2d3b4252f173a8c6f2b2e496a4f8329`;
it contains int64 `[512,512]` with original int32 token identity
`6b6a0c4283de3aae633fd2bf00f74ac80928a8c389c42aa6d5101b765115532e`.
Load and verify the full artifact before selecting `[0,511]`; retain the row
indices and exact row hashes. Do not invoke `_calibration_tokens`, tokenize a
corpus, or draw new samples. Existing canonical per-Linear X/H files lack the
HC boundary plus masks/positions/pass-state relationship and are unsuitable
as graph inputs. Neither the active capture's files nor process is touched.

## Concrete execution design

Implement one experiment helper and bounded CPU tests, using the existing
producer container content
`eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd`.
The machine-readable [plan.json](plan.json) is a proposed binding, not a CLI
execution manifest. Implementation must first freeze its own source commit and
resolved dependency/source-file hashes, then produce a PB invocation.

1. Verify frozen bindings, actual 104 GiB cgroup/92 GiB GPU subset, eval mode,
   BF16, eager attention, HC4, 45-layer original config, all-full indexers and
   no enabled gradient checkpointing. Use `build_streamed_causal_lm` with
   2 slots, 1 prefetch worker, lookahead 1, 24 GiB source headroom/minimum
   prefetch availability, and mandatory prefetched residency. Freeze every
   original parameter, including packed parents and fixed tensors. Record the
   installed model class, transformers/package versions, modeling file hash,
   callable wrappers/modules/files and loaded native libraries/kernel names.
2. Metadata-only phase: for each of the **512 original B1 rows**, call the
   existing `runner._prepare` under no-grad, construct the normal
   `StreamedForwardBoundaries` metadata object, and immediately discard the
   16 MiB expanded hidden tensor. Keep actual per-row ids, positions, masks,
   position embeddings and original pass-state containers; never expand a
   single sample's metadata by multiplication. Bind the existing
   `StreamedBoundaryArtifacts` owner with `n_probes=4`, call `check_auxiliary`
   throughout, record alias-aware CPU/GPU bytes and the conservative shared
   adjoint reservation. Assert GLM's state is empty, then close the owner and
   prove these metadata owners expired before the graph phase. This tests
   full population preparation, not later graph-state growth.
3. Run `visit_layer_batches` on the two selected rows with its existing exact
   `StreamedBoundaryArtifacts` v2 owner. Use a transparent `_call` observer
   to access each original layer's actual prefix-derived hidden, batch kwargs
   and pass state. Delegate the primary call unchanged, retaining its original
   no-grad output for return. Layers 1 and 2 only advance the prefix. At layers
   0, 3 and 4, settle the explicitly scheduled successor `[layer+1]`, then
   measure the graph replays below while that source pair remains stationary.
   Future 5 is included during layer 4; no unaccounted loader may coexist with
   graph admission. Returning the original output prevents the observer's
   replay from becoming the next layer's source input.
4. For each selected layer/row and each seed 7000–7003, supply one BF16
   `[1,512,4,4096]` cotangent of exact `+/-1/256`, generated with a local CPU
   generator outside the guarded source call. These **four deterministic
   boundary qualification stimuli are not original downstream/Fisher
   adjoints**. K4 is execution/owner coverage only. One incoming cotangent and
   one graph are live at a time; reference results become CPU byte hashes.
5. Three arms use identical original input/kwargs and the same stimulus:
   an unobserved isolated baseline, a nonfinal replay with a fork of a fresh
   quiescent `SharedStateCotangents`, and a final replay with its original
   owner. Each uses `profile.isolated_layer_pass_state`, `graft`,
   `runner.isolated_layer`, `produced_roots`, one `torch.autograd.backward`,
   and `harvest`. Assert full output bytes match the preserved no-grad source
   output, and complete outgoing input-gradient bytes match the baseline
   exactly (`rtol=atol=0`, not sampled checks). Record finite/nonzero gradients,
   owner counters and empty pending keys. GLM's observed shared state must
   remain empty; do not manufacture nonempty Gemma-style state to satisfy K4.
   Total: **72 backward calls**, 3 layers × 2 original rows × 4 stimuli × 3 arms.
6. The observer must use a narrow phase flag so isolated replay delegates to
   the original callable without recursive observation. No hooks rewrite
   outputs or gradients. Around each arm record source parameter/buffer
   pointers, versions, dtypes, storage aliases, sampled value hashes, module
   tensor/container state, CPU/CUDA RNG bytes, pass-state structure and actual
   cache kwargs. Any mutation is a refusal, not a reset-and-continue. Full
   source-file bindings are checked again after the action. These sampled
   source-value checks must not be described as a full value rehash.
7. In the visitor, raise a dedicated `PrefixGraphQualificationComplete` only
   after all two source rows at layer 4 and the exact 72-record roster passed.
   Catch only that signal inside the artifact-owner scope; exceptions from
   loading, graphs, parity, guards or telemetry propagate. Existing visitor
   `finally` unloads the current layer; explicit source settlement/drain and
   existing context reset/shutdown release the lookahead/cache. No broad
   `RuntimeError` catch, fake successful full traversal, or complete 45-layer
   initialization receipt. Write an explicitly partial prefix result.

## Owners, guards and proposed caps

PB submission: portable `gb10`, measurement isolation, 6 CPUs, **104 GiB shared
physical / 92 GiB GPU subset**, 1,800-second ceiling. Preserve PB affinity,
4 source reader threads and 1 native math thread. Keep existing source page
release, `MIMALLOC_PURGE_DELAY=0` and expandable CUDA segments. PB alone chooses
placement; the running original capture and other admitted work are external
load. No native execution before review of the CPU gate and frozen invocation.

The full `RESOURCE-PLAN.v2.md` maxima remain conservative inputs:
31.027723 GiB settled source; 13.500366 GiB additional CUDA source loader/packer
transient; 16 GiB graph/workspace; 2 GiB auxiliary; 2.125 GiB exact-boundary
resident cap; 256 MiB replay fork; 8 GiB inactive CUDA allowance; 4 GiB host
runtime/metadata allowance; 2 GiB physical guard margin; 8 GiB host free floor.
Use the actual observed source storage, not an allocated remainder proxy.

The helper's exact boundary working disk cap is **1 GiB**: six prefix planes
× two original B1 rows × 16 MiB = 192 MiB raw payload, plus an atomic writer
and envelopes. The configured resident cap remains 2.125 GiB, but the two-row
prefix cannot prove its full B64 behavior. No X/H, PWC or all-target matrix
owner is created. Stimulus, input clone, preserved source output, autograd
saved tensors, gradients and observer transfer scratch all belong to the
16 GiB workspace allowance; reference tensors may not escape an arm.

| Conditional envelope | Physical including 2 GiB margin | GPU subset |
| --- | ---: | ---: |
| Source transition with loader/packer | 78.653089 GiB | 70.528089 GiB |
| Settled source graph, no matrix/fork plane | 65.152723 GiB | 57.027723 GiB |
| Add future full statistics + replay fork | 97.402723 GiB | 89.277723 GiB |

These are planning sums, not measured fit. The last line remains conditional
because that statistics plane is not allocated. No realized source-graph
success upgrades it to a coupled full observation claim.

Use the existing `CaptureMemoryGuard` plus
`check_operator_allocation` (allocator release before a future reservation),
with actual `cgroup.current + cuda.reserved + future <= 104 GiB - 2 GiB` and
`host_available >= 8 GiB + future`. Never discount cgroup/CUDA overlap.

Required phase records: before skeleton/fixed-source construction; before each
source prefetch/load with its full transient reservation; after install plus
settled successor; before/after metadata preparation and release; before each
baseline/replay forward; immediately before backward; after backward; after
all graph owners are dropped; after prefix/source teardown. Source transition
and graph reservations must not be confused. While a graph is active, no
source load/install/top-up is permitted. Before backward, future reserve may
conservatively charge a full 16 GiB despite existing graph allocations; no
hand subtraction is used to relax the authoritative guard.

Reset CUDA peak counters only at documented arm boundaries after source
settlement. Measure graph/workspace high water against the **unique stationary
source and measured auxiliary GPU owners**, including every other device
allocation in the residual. It must stay within 16 GiB. Also record raw
allocator peaks, physical guards and host cgroup current/peak; the residual is
not a replacement for them. At phase admission, unused reservation must stay
within the ledger's 8 GiB allowance after release. Report cgroup runtime/page
charges and known CPU tensor owners separately; the proposed 4 GiB host
allowance is not proved by calling all non-CUDA bytes metadata. If the
allowance cannot be attributed/bounded, that full-ledger gate remains open.

## CPU gate, instrumentation and acceptance

Proposed new tests (PB CPU, portable eligible x86 Torch environment, bounded
threads) must demonstrate:

- Manifest/config/token/source-roster validation refuses digest, selected-row,
  dtype, original graph-kind, HC, cache-argument or source-coverage mismatches;
  no calibration sampler is invoked and no inference tensor enters backward.
- A small deterministic streaming test proves the transparent observer returns
  the original no-grad output, runs the exact 72-arm schedule, avoids recursive
  callbacks, preserves batch order, and catches only the dedicated completion
  signal after full expected coverage. Inject a parity/guard/telemetry failure
  and prove it propagates with owner cleanup and failure evidence.
- The graph path uses existing state graft/harvest/fork APIs; nonfinal mutation
  cannot modify the original owner. Use existing nonempty state tests for the
  API contract, while the GLM-specific expected native state remains empty.
- Source settlement precedes graph admission; unexpected pending futures or a
  source install during an arm refuse. Derived memory sums, working-artifact
  cap, matrix absence and weak-reference cleanup are checked without claiming
  CPU fixture graph parity proves the native original kernels.

Use in-process `torch.profiler` on baseline/nonfinal/final for original row 0,
probe 0, at each measured layer. Run remaining calls unprofiled with the same
checks. Avoid `record_shapes`/`profile_memory` so profiling does not retain the
graph. Export/close each bounded trace before advancing layers; record trace
bytes/hash and loaded callable/native-library identities. Original KDA/DSA,
router, shared MLP and both HC mixers must show actual execution. Transparent
forward hooks record only integer counts, shapes and scalar/hash diagnostics:
router top-k `[512,8]`, finite weights, in-range expert IDs, observed per-expert
counts summing to 4,096 assignments, shared MLP 512 rows, dense down projection
512 rows with input width 12,288, and HC input `[1,512,4,4096]`. Do not require
all 288 experts to be visited by two rows. Route indices, weights and outputs
must match between the primary call and each replay for the same row.

Netdata from **both** hosts runs throughout at 1 Hz; cgroup/allocator/process-I/O
samples run at 100 ms; CUDA peak counters catch intra-sample device peaks.
Capture power against the 140 W envelope, residency, CPU and actual I/O.
Attribute startup/source loading separately from graph intervals. Trace
analysis uses the checked-in bounded parser through PB and verifies CAS and
raw hashes. This is not a speed comparison: no work-per-joule ranking is
claimed from correctness replays with different observer scopes.

A native pass requires every exact numerical/routing check, the original
implementation identities, all phase guards, <=16 GiB measured workspace,
<=2 GiB actual prepared auxiliary state, <=256 MiB fork, unchanged source and
RNG/state, zero source loads inside replay, zero leaked graph/metadata owners,
completed artifact cleanup, no OOM, complete telemetry, and independently
verified PB terminal/CAS/source receipts. Missing tools, unqualified backend
selection, graph nondeterminism or any failed cap is a retained negative
result. Do not substitute kernels, enlarge tolerances/caps, resample rows or
repeat the passed statistics fixture to turn a failure into a pass.
