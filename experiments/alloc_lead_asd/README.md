# Issue1962 surrogate diagnostic

This research diagnostic compares the existing signed joint-AURA arithmetic
with realized activation and RTN weight perturbations on a dense Qwen3 model.
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
