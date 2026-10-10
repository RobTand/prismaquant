# Fused per-token FP8 QDQ before/after — 2026-10-10

PQ #1398 fuses the joint-statistics hook's per-token FP8 activation QDQ
(about ten torch ops per observed expert invocation) into one Triton launch
(`prismaquant/kernels/fp8_per_token_qdq.py`, kernel id
`prismaquant.triton_fp8_per_token_qdq.v1`). The fused leg is the default path
on CUDA; `fp8_dynamic.fp8_qdq_reference` (renamed from `_fallback_fp8_qdq`)
serves all other inputs and is the oracle baseline.

## Method

QDQ-only, following `fp8-native-activation-parity-2026-09-07.md`: call only
`fp8_dynamic_activation_qdq_vllm(...).dequant` on GPU-resident bf16 inputs
(256 rows, seeded randn with sparse 40x outliers), time with CUDA events
(20 warmup, 11 rounds of 32 calls), count kernels from an exported profiler
trace in a fresh process per arm, and sample device power over a sustained
8 s phase per arm plus a Netdata GPU-power pull per power window. The before
arm forces the reference through `PRISMAQUANT_DISABLE_FP8_FUSED_QDQ=1`; the
after arm uses the fused default. Arms interleave per width in one exclusive
GPU job, so both arms share the box and the inputs.

Both arms run in one snapshot (`e8871bdb1f`); the before arm forces the
reference through the kill switch, so no code drifts between the arms.
No numerics change: the oracle
(`tests/test_fp8_per_token_qdq_oracle.py`) holds bitwise on both legs,
including -0, midpoint ties, zero rows, saturation, E5M2 and NaN/Inf.

Environment: NVIDIA GB10 (compute 12.1), exclusive GPU on sparklina, torch
2.13.0+cu130, Triton 3.7.1, image
`prismaquant-glm-producer:content-qualified-20260908` (the 2026-09-07
source-reference image). Bench: `experiments/fp8_per_token_qdq_bench.py`;
Netdata pull: `experiments/fp8_netdata_pull.py`.

## Results

PB action `597669b33f088655f7abc1a9ef5e532f4a70f2466eaaf5161151e0c191241ac4`
(executed on sparklina in 126 s). Power means are the in-band nvidia-smi
phase means (38 samples per arm); Netdata means cover the same windows
(9 samples per arm, chart `nvidia_smi.gpu_*_power_draw`).

| K | ref ms/call | fused ms/call | speedup | ref kernels | fused kernels | ref dev µs | fused dev µs | ref W (smi) | fused W (smi) | ref W (netdata) | fused W (netdata) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 0.0606 | 0.0289 | 2.10x | 12 | 1 | 17.1 | 6.7 | 22.0 | 26.9 | 24.4 | 27.0 |
| 1536 | 0.0621 | 0.0289 | 2.15x | 12 | 1 | 28.5 | 7.5 | 33.9 | 30.2 | 19.4 | 25.3 |
| 2048 | 0.0610 | 0.0291 | 2.10x | 12 | 1 | 34.2 | 6.7 | 38.7 | 30.8 | 40.7 | (see receipt) |
| 4096 | 0.1081 | 0.0289 | 3.74x | 12 | 1 | 106.8 | 8.0 | 44.8 | 34.1 | (see receipt) | (see receipt) |
| 16384 | 0.6173 | 0.1287 | 4.80x | 12 | 1 | 657.1 | 127.1 | 41.1 | 39.0 | (see receipt) | (see receipt) |

Full JSON (per-call minima, power samples, calls/joule, kernel names, phase
epochs) is in the action log (`SUMMARY_JSON`) and its CAS payload. The fused
trace shows exactly one kernel per call; the reference trace shows the twelve
expected ops (upcast, abs, amax, div, clamp_min, div, clamp, two casts, scale
multiply, scale fill, denominator fill).

## Reading

The win is launch-bound at small K (fused wall stays near 0.029 ms while
device time is under 8 µs) and grows with K (4.8x at K=16384). Device power
sits at 22–45 W against the 140 W envelope on both arms, so neither arm
saturates the box; this matches the 2026-09-07 observation for this helper
and establishes no serving throughput claim. Fused power reads slightly
higher than reference at small K because the faster arm retires more calls
per second in the sustained phase; calls/joule in the receipt is the fair
comparison.

## Limits

This is QDQ-scoped evidence, not a full-model or hook-level number. The
`fe9730f7` full-backward profile on the MTP harness and its Netdata series
remain open (child issue): that harness serves the higher-priority
statistics-accumulate fuse, needs a tessera-quant GPU image plus the M5
plan/inputs/model, and its arms compare accumulate variants, not QDQ legs.

## Checks

| Check | PB action | Actual result |
| --- | --- | --- |
| Oracle + parity, x86 CPU | `65870d66fadf625d6333dbd4b5718979b8fc591b8012ce0207eb9003bb1d2e00` | 19 passed, 60 CUDA-tier skips |
| Parity, x86 CPU | `d66d97a9d6775794215c0faf25ebd1bf8573397c8b9fc78c69552af92c879c8e` | 2 passed, 2 CUDA skips |
| Format registry, x86 CPU | `7736916ae613c47ede9a45e3dfc28060d0d1221bb9bf9616d1a6bf7cd392d4eb` | 15 passed |
| Oracle + parity, GB10 CUDA | `ff9dc584ee5b15f8cdda6356317105bad19768c662db7690f42597cb89c142fd` | 98 passed |
| QDQ before/after + power | `597669b33f088655f7abc1a9ef5e532f4a70f2466eaaf5161151e0c191241ac4` | table above |
| Netdata reachability | `6429ce4f2082ccb6bc7fc57ab8cd4b3f00ea5f168370b2dc88648125d86c4c11` | worker-local Netdata answers |
