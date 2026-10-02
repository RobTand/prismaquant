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
`fit_s42` sequences of length512 with eight global-row Rademacher probes
(seeds7000..7007), global token count2048, all token positions, temperature1,
and FP32 logit-delta storage. It retains the production lease crosscheck,
all/attention/MLP A and W arms, the no-position-zero A arm and W+A. It omits
single-unit and per-layer arms in this initial screen.

The runtime uses the existing campaign container adapter, which authenticates
the declared scientific image content and source import before launching the
immutable local image ID. The recipe declares a separate PrismaBuild portable
image reference; those two hashes have different schemas. Tokens and models
are already on the shared mount. Their file digests and the exact token-prefix
digest are checked/recorded inside the action. The pair records effective
Transformers forward source files, runtime selectors/configuration and a
bitwise repeated-clean-forward null gate. It exports a short pricing Torch
trace per dtype and hashes every durable output on completion.

Submit from dl380g10 through published PrismaBuild; the launcher is only a
child of that admitted action. CPU-only preflight uses `--anywhere`, CPU2 and
aggregate memory8GiB. The proposed GPU action uses `--measurement
--host-class gb10 --gpu --cpus4 --demand mem_gb=32
--gpu-memory-gb32`; on GB10 that last32GiB is a subset cap within the shared
physical reservation. The container CPU-accounted cap is8GiB. Native threads
are1. GPU admission awaits Astra's current fleet telemetry and capacity check.
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

Measured: lease-v2 components agreed within9.5e-7 relative-to-RMS; all sixteen
tiny arm KL comparisons within2.4e-13 relative, quadratic comparisons within
6.7e-10, realized probe components within1.1e-5. Scope peak accounted memory
was1,463,259,136 bytes. These are arithmetic/container gates on a random tiny
model, not pretrained H9 results or GLM quality evidence.

The CAS payload was read and its SHA256 verified as
`371f03a441a4d53b2abccf3e581e5e38bdd4f7dd285c9ff77c7e5c1096defd39`;
receipt `88b81f27d76e4d05415ffbe4628ee16295ab3d1626ad16470946304aebe9919a`.
Terminal and CAS records live under `/mnt/shared/prismabuild-fleet`, keyed by
the full action above. Outputs are
`/mnt/shared/tessera-measurements/pq1962-sol-20261002/container-qualify/`.
