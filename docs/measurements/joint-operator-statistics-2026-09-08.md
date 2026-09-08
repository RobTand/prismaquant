# Deferred joint operator contractions — 2026-09-08

Issue #374; implementation `6a2cf9f91399f4a0e2d3ed1d10f795c918f2817b`.
This is a synthetic primitive qualification at GLM expert dimensions. It does
not qualify a complete calibration probe, ProductionWeightCache fill, model
residency, KL, bpp, export, or serving behavior. Pipeline defaults are unchanged.

## Contract

`JointOperatorStatisticsLease` extends the existing joint-AURA source observer.
For one probe it accumulates FP32 `sum(G.T @ X)` and, per distinct nonidentity
activation quantizer, `sum(G.T @ dX)`. The observation seal contracts activation
terms against unchanged source weights, removes hooks, and releases its source
references. The caller then supplies prefetched candidate deltas in bounded
quanta. The lease returns signed weight, activation, mixed and total terms and
never owns candidate deltas beyond a projection call. The finite candidate
budget charges whole backing storages, including storage hidden behind views.

This changes FP32 contraction order and therefore has a separate arithmetic
identity. It is not bit-identical to the original per-invocation contractions.
The original lease remains available. Full integration needs phase admission
for source weights, statistics, render buffers, cache transfers and physical
memory; these two tensor budgets alone cannot certify a whole process.

## Matched native measurement

PrismaBuild action
`438397d5a6c876645a08c6eef7e87c76707fbda9f03c6a2caf83cfa3261525d9`
ran on Sparklina, NVIDIA GB10, Torch `2.13.0+cu130`, CUDA 13.0, CPUs 5 and 6,
with two reserved CPUs, 12 GiB physical memory and an 8 GiB GPU subset cap.
Native threads were bounded to one. The content-qualified producer image has
content seal `eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd`.
Sparky concurrently ran the independent common capture; both hosts were sampled.

Each shape uses A/B/B/A, ten seconds per arm, with a separate Torch trace after
each timing arm. Every invocation includes eight synthetic candidate deltas,
four backwards of 32 tokens each, BF16 source weights/inputs and FP32 delta
planes. Deltas are constant `(candidate_index+1)/1024`; QDQ rounds `X*8` and
divides by eight in source dtype. TF32 is disabled. Both arms include delta
construction. The new lease admits 64 MiB statistics and one 32 MiB delta.

| Projection shape | Legacy latency | Deferred latency | Ratio | Incremental CUDA peak, legacy → deferred | Work/board-joule ratio |
|---|---:|---:|---:|---:|---:|
| 2048 × 4096 | 41.881 ms | 16.782 ms | 2.496× | 404,363,776 → 134,219,776 B | 2.365× |
| 4096 × 2048 | 41.691 ms | 16.756 ms | 2.488× | 404,101,632 → 134,219,776 B | 2.421× |

Latencies are means of the two arm medians, not pooled invocation means.
All eight arms return to their pre-arm CUDA allocation after profiling.
The attempt cgroup peaked at 2,754,871,296 bytes, consumed 119.681 CPU seconds,
exited zero, completed cleanup and left no owned container running.

The profiles explain the gain: for 2048 × 4096, the first legacy/deferred
traces reduce `aten::mul` from 72 calls/26.908 ms to 21 calls/7.239 ms and
`aten::sum` from 68 calls/10.482 ms to 17 calls/2.665 ms. Deferred accumulation
adds six `aten::add_` calls/2.568 ms. Matrix multiplies remain 16 calls, around
1.94–2.01 ms. Candidate projection now scans each summed operator once rather
than scanning every candidate during every backward. Times are profiler
self-device totals and should not be summed with child kernel events.

Existing pqteld 2 Hz samples give 20 power observations per arm. Legacy board
power spans 22.399–25.736 W; deferred spans 24.656–26.115 W, far below GB10's
approximately 140 W envelope. There is no saturation claim. Work/joule uses
completed invocations divided by elapsed time and mean board power, then the
ratio of paired-arm means; no idle subtraction or whole-system energy is claimed.
Netdata host CPU means were 7.49–8.22% on Sparklina. Both-host raw Netdata and
pqteld windows are retained. The automatic PB box window lacked its pqteld CSV;
the existing host CSVs were copied for these exact intervals and read with the
published PB window reader. No power value is inferred from GPU utilization.

## Numerical and lifecycle validation

The initial missing-API regression recorded nine failures in PB
`424b26ef368a132f91eb3fd0ff0eb0802cf37a09c8ff786d1344828115e09d0b`.
Final CPU action
`8fe0afac8f716f8ac24403c78198de3f8285aa6581bf06cc7162ea0c842b2b27`
passed 57 tests, skipped the two CUDA-only shape cases, and compiled both touched
Python modules. The native action above passed all 46 selected tests with zero
skips, including independent FP64 output-residual oracles at both full expert
shapes. Coverage includes signed cancellation, repeated/shared QDQ, real packed
source slices, never-routed experts, source mutation, missing/duplicate backwards,
source-owner release, atomic refusal of nonfinite candidate quanta, and complete
backing-storage accounting. The synthetic timing fixture's maximum component
change from legacy is 0.000030517578125; this is an arithmetic screen, not a
model-quality threshold. One intermediate test edit failed collection with a
syntax error; that failure and its corrected successful run are retained.

## Reproduction and evidence

The committed harness is `experiments/joint_operator_statistics_profile.py`.
The complete PB command, image/environment declaration and source identity are
in `native-invocation-01.json` under:

`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/joint-operator-statistics-01/`

That directory contains `cpu-root-audit-01.json`, `native-root-audit-01.json`,
`native-submission-01.log` and `native-01/{result.json,summary.json,box-windows.json,
netdata.jsonl,pqteld/,*.trace.json}`. The root audits independently verified
terminal status, canonical CAS receipt/payload hashes and complete source
snapshots against the implementation commit; only generated PB closure files
differ. All eight trace hashes were checked. Native result SHA256:
`3798d4008715690bbd0885f650e7d0e6e62073ae0e04df0b47cb169105c6d7d0`.

## Review corrections and final native qualification

Review found that a QDQ exception after an operator update could be caught and
retried, double-counting the earlier update. It also found that a retained
output kept completed observer input/source captures alive. Three regression
cases failed on the original implementation in PB `ede589437c71`. Commit
`6ffbf2e5d` makes any failed collection poison the lease, clears pending captures
on failure/exit, and clears consumed captures immediately while preserving the
duplicate-backward refusal. PB `34c97d65248f` then passed 66 CPU cases with two
CUDA skips and compiled the touched modules. An earlier mistargeted submission
`ca3c77d863bf` selected no tests; it provides no validation evidence.

Final native PB `93e6a3ae8fdb3d4022fe27974beb7c39ea4b4aa0a9e2ec301c0908ed7a4e489d`
ran the corrected commit on Sparky under the same content-qualified image and
12/8 GiB physical/GPU budgets: 49 passed, zero skipped, exit zero and completed
cleanup. Its new A/B/B/A pairs are self-contained on that host; no cross-host
latency delta is inferred from the earlier Sparklina run. For 2048×4096 the
means of arm medians are 41.905→16.458 ms (2.546×); for 4096×2048 they are
41.750→16.637 ms (2.509×). Approximate work/board-joule ratios from matched
2 Hz host samples are 2.413× and 2.431×. The synthetic workload and scope limits
above still apply. New traces, both-host telemetry and derived summaries are
in `native-02/`, with invocation and independent canonical-CAS/full-source
verification in `native-invocation-02.json` and `native-root-audit-02.json`.
