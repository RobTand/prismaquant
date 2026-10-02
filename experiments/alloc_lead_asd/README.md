# Issue #1962 surrogate diagnostic

This research diagnostic compares the existing signed joint-AURA arithmetic
with realized activation and RTN weight perturbations on a dense Qwen3 model,
and independently checks staged cotangent/row contraction on a tiny GLM fixture.
It does not correct, rescale, or replace the production estimator.

The retained sequence-major diagnostic and its CPU oracle originate at
Claude scratch commit `7a18208575795ae1b1bbbc4b06bac0c2a3e8c060`.
The original fp32/bf16 pretrained pair was withdrawn and never completed.
The old two-sequence/two-probe smoke is not evidence for H9.

## Bounded dtype screen

`run_pair.py` executes sequential FP32 and BF16 legs on exactly four
`fit_s42` sequences of length 512 with eight global-row Rademacher probes
(seeds 7000..7007), global token count 2048, all token positions, temperature 1,
and FP32 logit-delta storage. It retains the production lease crosscheck,
all/attention/MLP A and W arms, the no-position-zero A arm and W+A. It omits
single-unit and per-layer arms in this initial screen.

The runtime uses the existing campaign container adapter, which authenticates
the declared scientific image content and source import before launching the
immutable local image ID. The recipe declares a separate PrismaBuild portable
image reference; those two hashes have different schemas. Tokens and models
are already on the shared mount. Their complete files are declared in one
startup phase and staged by PrismaBuild. The existing whole-file reader opens
each file through a lifetime-pinned window, RAM first when offered and SSD
otherwise; pool reads refuse. Its owned bytes supply digest verification and
safetensors decoding. The exact token-prefix digest is recorded inside the
action. The pair records effective
Transformers forward source files, runtime selectors/configuration and a
bitwise repeated-clean-forward null gate. It exports a short pricing Torch
trace per dtype and hashes every durable output on completion.

Submit from dl380g10 through published PrismaBuild; the launcher is only a
child of that admitted action. CPU-only preflight uses `--anywhere`, CPU 2 and
aggregate memory 8 GiB. The GPU numerical screen uses `--anywhere --exclusive
--gpu --cpus 4 --demand mem_gb=32 --gpu-memory-gb 32`; on GB10 that last 32 GiB
is a subset cap within the shared physical reservation. The container
CPU-accounted cap is 8 GiB. Native threads are 1. Astra approved this bounded
screen after fresh fleet telemetry. Class-scoped measurement submission from
x86 dl380 refuses because the submitter has no NVIDIA platform evidence;
ordinary exclusive admission preserves numerical scope without spoofing those
facts. Profiles and aligned both-box telemetry supply contextual evidence;
there is no timing or throughput acceptance claim.
Neither a declared budget nor the CPU preflight establishes measured GPU peak
memory. Increase scope only after reading this action's actual result/profile.

Interpret `P_add/P_joint` as the cross-unit omission, `P_joint/S_real` as
network linearization, `S_real/Q_real` as probe sampling and `Q_real/KL_true`
as the logit second-order approximation. Keep whole-draw prices separate from
the per-sequence diagonal estimate; eight probes and four sequences yield a
screen, not calibration confirmation or a general estimator fix.

An elevated BF16 price changes the primal path, QDQ and kernel selection as
well as backward precision. It needs a targeted backward-only VJP or finite
difference control with fixed operands before H9 can be called causal.
GLM RMSNorm and mHC already perform their inner arithmetic in FP32; FP8's
maximum element maps exactly to448. The original stronger H9 explanation is
therefore unsupported. If the pair is calibrated without a precision effect,
compare direct and stored tiny-model cotangents with explicit row coordinates;
reproducing the existing stored GLM chain alone cannot exclude H10 alignment.

## Qualified CPU container evidence

Action `11eda169ce8bd68b1e453e0b66ae7285d6a9562f8a077b13d9974d0cd5d5c872`
ran the imported tiny random Qwen3 oracle on sparky with the GPU unattached,
source snapshot `880ec37fe5b8b0e24eaca43dd7105e640357efb4`, parent
`a08ceb680eb2236ce7b4977cdddff8db8d5510b3`. Terminal exit0, cleanup complete,
all CPU gates ran, no skips. The CPU-only profiler printed a CUDA device-query
warning(code35); this is not GPU validation.

Measured: lease-v2 components agreed within 9.5e-7 relative-to-RMS; all sixteen
tiny arm KL comparisons within 2.4e-13 relative, quadratic comparisons within
6.7e-10, realized probe components within 1.1e-5. Scope peak accounted memory
was 1,463,259,136 bytes. These are arithmetic/container gates on a random tiny
model, not pretrained H9 results or GLM quality evidence.

The CAS payload was read and its SHA256 verified as
`371f03a441a4d53b2abccf3e581e5e38bdd4f7dd285c9ff77c7e5c1096defd39`;
receipt `88b81f27d76e4d05415ffbe4628ee16295ab3d1626ad16470946304aebe9919a`.
Terminal and CAS records live under `/mnt/shared/prismabuild-fleet`, keyed by
the full action above. Outputs are
`/mnt/shared/tessera-measurements/pq1962-sol-20261002/container-qualify/`.

## Loader regression and correction

The first pretrained action
`b66654186cca50c0ed5f4f91b1fbc4c9a845c8685af2c9d324b1f77b3f627300`
failed before pricing, exit 1 and cleanup complete, on sparklina. The new
meta-device constructor left Qwen3's nonpersistent rotary buffers on meta:
they are not checkpoint state, so assigning source parameters did not
materialize them. This was a diagnostic loader defect, not H9 evidence.
Its durable `pair1/inputs.ready.json` records four real lease/pin IDs,
1,505,923,749 bytes served from SSD staging, zero misses/fallbacks and zero
pool bytes. RAM was not offered at those opens; permitted SSD served.

The correction uses Transformers' existing `no_init_weights` context with
ordinary construction, copies the checkpoint into the requested parameter
dtype and moves devices without casting the constructor's FP32 rotary
buffers. CPU action
`063384986b5e067bec5f344884734fda5ea59964382b78f796b395f7fc92d5e0`
qualified exact checkpoint values, tied embeddings, nonpersistent buffer
values/dtypes and logits in both FP32/BF16, plus refusal of a missing
checkpoint parameter. Exit 0 and cleanup complete. Source parent
`a9684ac44d8e5e0c76ff204b918f4175c18f8941`; CAS receipt
`376c4302977d7f7f5d573d7a40596338990a40abf1f58efb02c6f2d03d9b7d47`.

## Qualified small-model result, 2026-10-02

The recovered v2 diagnostic failed its required BF16 lease crosscheck in
action `a2ad67617ebbe6679157a20ba8c2a9311188f5b179cbc4251a9afde472d9e6de`:
maximum difference/RMS was 8.58% for A and 5.10% for W. Its FP32 leg completed
and was retained, not rerun. A same-graph, same-probe control then separated
cotangent production from contraction arithmetic:

- Ordinary backward: 190/196 units changed their cotangents even between
  two `backward()` calls on the same graph/scalar. Every hook ran exactly
  once; clones excluded retained alias mutation. Fixed-g FP32 operator and
  output-space contractions agreed within 3.80e-6 relative RMS. Trace events
  identified actual BF16 FlashAttention backward kernels.
- Strict deterministic backward: every cotangent became bitwise identical
  across `grad`, `backward`, and repeated `backward`, for both probes.
  The resulting API projection difference was exactly zero; fixed-g
  contractions agreed within 4.74e-6. The trace still used FlashAttention.

Controls were PB `773eebb6edfcc8b6193781dc09562754c9957dad238e8522caed2d6084d92f91`
and `30f0312ae001d194e0f3ffa6b6c2fc1edeae2ffa21988f28eb59c72963792498`,
both exit 0 and cleanup complete. Their CAS receipts are respectively
`78f1717a52be97a41e7eb3d5b6a8eb960e99e3fb30953c0b9a409856d90b7404`
and `c67f315b97dcd99c4a20f5088f0ff47f6d84499ffd33bd162913e1c11116b307`.
This localizes the diagnostic discrepancy to nondeterministic backward;
it does not establish GLM's factor-seven cause.

The correction is opt-in and diagnostic-only: `backward_policy` enables
strict deterministic algorithms around pricing autograd calls and restores
the prior enabled/warn-only flags in `finally`. Forward/probe construction
and the production estimator remain unchanged. The BF16-only completion
reused the banked FP32 result with this explicit backward-policy difference:

| Term | FP32 full price | BF16 strict-backward full price | Paired sequence-diagonal BF16/FP32 ratio, screen 95% CI |
| --- | ---: | ---: | --- |
| A all | 0.0112950816 | 0.0107347344 | 0.9662 [0.9233, 1.0179] |
| A without position zero | 0.0110305633 | 0.0104136215 | 0.9644 [0.9237, 1.0129] |
| W4 all | 0.2101098709 | 0.2094051954 | 0.9949 [0.9873, 1.0035] |

The FP32 A-all KL is 0.0104800690; KL/full-price is 0.9278. The FP32
KL/sequence-diagonal-price ratio is 0.9635 [0.8968, 1.0379]. BF16's own
forward-differenced KL is not substituted for that reference. Four sequences
and eight probes are a mechanism screen; these 4,000 paired bootstrap draws
describe only the observed sequence set with fixed probes, not domain-wide
uncertainty. There was no net positive BF16 excess, so a top-five positive
excess concentration share is not applicable. H9's generic BF16-inflation
mechanism is negative on this screen; GLM H7/H10 remain open.

BF16 completion `cec83e225733784fd80235019c01fd3d6b1ec4ce2c6c7662b13c82918aeb1400`
ran source `1127ff021eb243856d41fe5918f24f1fb62d4db3`, exit 0, cleanup
complete, CAS receipt
`7d803aa1efe1d3e0c597d49eb8f91e0dde73db586ced0b5b4272c86cd5b6c5e8`.
Its lease-v2 crosscheck is within 4.54e-6 and its repeated-clean-forward
null gate is bitwise equal. CPU comparison plus exceptional flag restoration
passed action `5ee8292b34821a3955362d6bccd26b39ac1cce648d28824436a14e07dcbc9b96`,
CAS receipt `d6335611f6fe6afea1db2c69a12308adddca323cf5ed62f7ea530f2ccd912c27`.
Compilation of nine touched/diagnostic modules passed action
`f7be838f9ad8155e5af2da8d54d7deabe573426fa2c7bd60541cf122b335d9b3`.

All payloads/terminal records were read; source parents, return codes and
cleanup were checked. Outputs and Torch traces are under the shared root
above in `pair2`, `control1`, `control2-deterministic`, `bf16-complete3`.
Aligned both-box Netdata windows are in `netdata-*-aligned`. These runs used
ordinary exclusive GPU admission and ambient CPU load; profiles/telemetry
are contextual evidence, with no timing or throughput acceptance.

## Tiny GLM row and chain control, 2026-10-02

The existing two-layer GLM5-Next fixture supplies KDA, DSA/MLA, mHC, one
dense MLP and a four-expert/top-two routed MLP with shared experts. This is
a seeded synthetic model, hidden size 64, vocabulary 128, four sequences of
length 32, two Rademacher probes 7000..7001 and global token count 128.
It uses the exact corrected derivative image/modeling binding already
recorded in the original GLM L05 source. Source execution is eager attention,
grouped_mm experts, highest FP32 matmul precision, TF32 off and BF16 reduced
precision reduction off. No strict deterministic-backward override is used
for this GLM fixture.

The immutable model safetensors SHA256 is
`333d34e4a1f75a6be9608bf0f014c35d5d240304bf201f985fd7f23983a8d65e`;
the raw token tensor SHA256 is
`2074dda59d19f7be08493b69be15fbf3093d55d8933d4b31206fbf614708c8aa`.
They contain no original GLM weights. `qualify_tiny_glm.py` records the full
configuration, source identity and derivative callable binding. The CPU
source qualification passed PB `38523ef3ba8976e03ad07a9ab5c17a2a078b8622e995e1402aa537aa40dbdfb1`.

`tiny_glm_control.py` compares a direct full graph with the existing
`run_adjoint_capture_core`: boundary microbatch 1, layer-roll batch 4 and probe
fusion enabled. The existing packed observer and gradient selector retain
explicit `(sequence,position,routing-slot)` coordinates for each activation
error/cotangent pair. `JointOperatorStatisticsLease` contracts those same
captured operands. BF16 additionally feeds them through the existing
`StageBReplaySpill` recording/replay seam with operator_gemm with chunk_rows 65536.
This is not the full layer-quantum capture driver. FP32 cannot enter that
spill's production 16-bit operand contract; it uses the statistics owner
directly. Model weights are fully resident in this tiny fixture and the
source-provider callback asserts their device residency; this does not
qualify production source loading or prefetch.

The full FP32 entrypoint passed CPU PB
`f24f5c738f4cc0c48b4cbe033d5f6f8029dfd15f43c06227a265db4f46e9b2bd`
(exit 0, cleanup complete). Its library Torch short-convolution substitution
is CPU-preflight-only. On CPU all four boundary cotangents and all 36 signed
unit/probe components were exact. The GPU profiler identifies actual
`aten::conv1d`, `aten::_grouped_mm` and `GroupedMmBackward0` events; this
control makes no fused-KDA kernel or full-size projector qualification claim.

| Comparison | FP32 GPU | BF16 GPU |
| --- | ---: | ---: |
| Boundary cotangent maximum relative RMS difference | 1.31e-7 | exact |
| Direct vs captured signed A components, max abs/RMS | 3.17e-7 | exact |
| Fixed-g output dot vs operator contraction, max abs/RMS | 6.95e-7 | 3.61e-7 |
| Captured operator vs spill replay signed A | outside 16-bit spill contract | exact |

All 36 unit/probe records have identical direct/captured coordinate order,
with exactly 256 routed `(sequence,position,slot)` pairs per probe/role and
all 128 dense/shared rows per unit. FP32 x and dx match exactly; its per-unit
g differs at the small cotangent scale above. All BF16 x, dx and g are
bitwise identical. Swapping two nonidentical spill input rows while retaining
the original metadata/checksum refuses with `Stage B spill checksum mismatch`.
This confirms checksum-sensitive delivery at that existing seam, not immunity
to an upstream producer assigning incorrect coordinates to self-consistent bytes.

FP32 results are banked under `tiny-glm-control4`: action
`7668acbf4020878f30ae2c0cd1beef59778542c35d3bc3bbf0107f45f6e52a7f`,
source parent `c1abf626f6b74816bd3a900878eeeeb1b5903981`.
The completed FP32 leg reused its direct graph from action `51bb661a...`;
that graph had been saved before a source-residency refusal. The containing
pair subsequently exited 1 when BF16 reused the same immutable produced-output
group IDs. Its FP32 result remains attributable; the wrapper is not a pair pass.
BF16's completed direct graph was saved before that refusal, then reused
through authenticated staged inputs by the BF16-only completion.

BF16 completion is `tiny-glm-control6`, action
`5ddafd588f8b6e470f84ab46a89385549bbb96e39e5a8853bcc56042ae217241`,
source parent `04ea01bc7c60106915e6071ef430c0e46a8c3db4`, exit 0 and cleanup
complete. CAS receipt
`939a0e473dbf858948913139034d579b027d57dc155a865dd436f43e5ccf581d`,
payload `dc177936b20d457fb87f06414259535306f8d1a2fb65a32a37e878da8b5b98b1`
was read and hash-verified. Both GPU quanta used CPU 4/native 1, aggregate
memory 16 GiB with GPU demand 16 GiB as a shared-memory subset, portable exclusive
ordinary admission, a 900 s hard cap and semantic startup/control/publish phases.
BF16's seven source/banked-result inputs served 6,091,966 bytes through pinned
SSD staging, with zero pool bytes, misses or fallbacks. Stage A's produced
boundaries/cotangents used its existing owned local spool; the diagnostic did
not add a cache or dispatcher.

CPU output audit `f6ff5bae49c617969b4b4c5fbfab7a4323d2354f73640db21b34edaa34f27dac`
verified the numerical/coordinate statements above, exit 0 and cleanup complete,
CAS receipt `dea90287d73311c1a09fc26db3829671c395e926175afe663f55e04395f93443`.
Its machine-readable result is `tiny-glm-audit.json`. Each completed leg has
its Torch trace. Both-Spark raw Netdata windows aligned to claim/finish with
120 s margins are in `netdata-tiny-glm-{fp32,bf16}-aligned`. CPU load is ambient;
there is no timing, throughput or useful-work-per-joule acceptance claim.

Setup refusals were retained rather than recertified: missing import/dependency,
two-window rather than required three-window produced funding, source-residency
declaration, CAS ingestion mount, unintended vision units in the roster, and
the reused produced-output identity. Action `b38b3a41...` then refused a malformed
manifest digest before staged reads/model execution; its empty result directory
is retained as startup evidence. CPU full-entrypoint qualification preceded
the successful text-only GPU chain. The known-good derivative image was not rebuilt;
missing xxhash 3.7.0 uses the locked ARM64 wheel SHA256
`f3e7b689c3bce16699efcf736066f5c6cc4472c3840fe4b22bd8279daf4abdac`
in a scoped dependency directory whose files are verified before use.

This is a negative H10 mechanism screen at the tiny fixture dimensions. It
does not establish original GLM L05 boundary, cotangent, rendered assignment,
full-size projection, or per-sequence calibration correctness. Issue #1962 and
the full GLM validation gate remain open.

## Historical GLM comparison contract

`cpu_glm_provenance.py` audited existing L05 cost/adjoint/handoff metadata in
PB `8db7e41d2dd1b3011faa18cf862130877e4653aa817b3c38defefb8200113e3b`.
All 867 format rows share one K4/global-row probe identity: 512×512 calibration,
all 512 output-logit positions, global token count 262144, seeds 7000..7003,
temperature 1. Boundary 5 and cotangent 6 coordinates cover 512 sequences and
512×4 probes exactly. Metadata consistency does not independently validate
the retained tensor pairings. L05 additive A is 0.0026182497805902175;
expert 240 down alone contributes 0.0021725103132911663.

CPU audit `c43848557fab7e0ffab3cf5692665299da797aff32c6e91215171b515010a4a3`
reconstructed the banked g3cal first 511-position comparison and authenticated
its calibration tokens, window hashes, source identities and self-teacher
bindings (exit0, cleanup complete; CAS receipt
`63fe105b44605287bc7935d985fe8346a15b958c050a4d1b116383c3d9a455c9`).
That pass concatenated four source 512-token rows per 2048-token window.
Its historical 0..510 band has the original source prefix only for rows
`0,4,8,...,96`: 25 rows, not all 100 rows in the joined windows. The remaining
75 rows have foreign preceding context. On the first 511 band, saved finite
decoded-weight arms give WA−W 0.00410335096 ± 0.00266368979 (paired-window SE).
The whole 512-row surrogate A aggregate is 0.01443150679; its within-unit
weight/cross correction predicts WA−W 0.01466578715. These are different
calibration/position scopes and different loss objects: the A quadratic at
the clean source is not standalone finite WA−W on rendered weights, and the
allocator's additive-unit sum omits cross-unit network terms.

The saved cost row has only draw-summed signed per-probe components. The
inspected historical replay directory contains only `L05.dry-inputs.json`,
not measured per-sequence/token pricing. The exact 25-row prediction cannot
be recovered algebraically from those sums, and their calibration variance
cannot be bounded from this evidence. Removing activation position 511 would
not remove that output position's Fisher contribution to earlier activations.
Because distribution KL uses no next-token target, an alternative first 512
logit band 0..511 can match the surrogate's all-logit position scope, subject
to independent source-prefix/causality qualification. Position 511 must not be
discarded merely because the next observed token belongs to a joined row.
That alternative still needs a 25-row projection decomposition; tiny controls
and metadata do not establish the actual factor-seven cause.

The first 512-logit prefix audit passed PB `a994f9927e8e2124dc09c9fa25f788081b48fbd7c2e8fd9a0abbb4da532ced6e`, exit 0 and cleanup complete. Its saved first 512-position WA−W is 0.00411478698 ± 0.00265704925; the matched-row price remains unavailable. Machine-readable output is `glm-comparison-alllogits-audit.json`. Final compile of the research directory and incidental staged-reader docstring passed PB `e3ba1ddf8529ae76759357efa5afc8cff907ef4bf297232fdabff73a36159104`, source parent `0a11e08d818`, exit 0 and cleanup complete.
