# Planned-size native allocator turnover proxy, 2026-09-08

The planned-size CUDA allocator proxy passed on GB10 with the actual existing
`CaptureMemoryGuard` and `check_operator_allocation` helper. This establishes
retired allocator-block turnover at the tested size. It does not allocate the
next statistics/workspace, run an actual operator window, load GLM source, or
qualify full-model fit or quality.

The live uint8 proxy has exactly 33,315,763,888 bytes, taken from
`terms_bytes.source_settled` in the planning ledger. That is geometry-derived
current-plus-lookahead source and fixed nonbody size, not actual source tensors.
The frozen ledger SHA256 is
`dfc21a3a9988ec89db74db0f0e3502279477abfcfe97ea9a26fb45ae5a989e0c`;
its `BLOCKED_NOT_RUNNABLE` status is preserved. An additional actual 32 GiB CUDA
statistics proxy was allocated and filled. Dropping its last owner left the
reservation inactive; weak backing-storage observation confirmed no real tensor
owner remained. The requested next 32 GiB statistics plus 16 GiB workspace stayed
a **48 GiB future reservation**, without being allocated.

| Observed bytes | Old guard with retired blocks | Fresh guard after existing helper |
| --- | ---: | ---: |
| Actual cgroup charge | 702,423,040 | 725,213,184 |
| Actual CUDA reservation | 67,698,163,712 | 33,323,745,280 |
| Future statistics plus workspace | 51,539,607,552 | 51,539,607,552 |
| Conservative total including future | 119,940,194,304 | 85,588,566,016 |
| Physical refusal threshold | 109,521,666,048 | 109,521,666,048 |
| Guard outcome | refused | passed |

The old state also lacked the future-aware host floor: available memory was
51.17 GiB against 48 GiB future demand plus the existing 8 GiB host floor.
After retirement, 83.71 GiB was available. No guard cap, margin or failure state
was weakened. The old guard remained sticky after cleanup; the corrected call
used a fresh guard.

The helper released 34,374,418,432 reserved bytes (32.013671875 GiB). Live
allocator bytes stayed exactly 33,315,764,224 before and after; the slight
rounding above storage size belongs to allocator bookkeeping. The live storage
pointer, exact storage byte count, dtype, shape and all 64 evenly spaced scalar
samples stayed identical. Every sampled value remained 17. No full-sized clone
or boolean comparison tensor was created. Final owner cleanup returned actual
CUDA allocation to zero.

Raw before/after CPU+CUDA profiler traces were retained and hashed. The first
trace contains allocation/touch and old-guard refusal: 49 actual kernels with
348.00 ms summed kernel durations, plus CUDA memory-create calls. The second
contains existing-helper retirement and sampled identity verification: one
7.36 µs sampling kernel, with `cuMemUnmap` and `cuMemRelease` calls. These are
different phases; their durations are not an algorithm throughput comparison.
Netdata has ten samples from each Spark. PB also retained the worker's power,
unified-memory and CPU series. Whole-action minimum available unified memory
was 55,001,505,792 bytes and observed unified used peak was 75,594,485,760 bytes.
The cgroup-only peak of 783,994,880 bytes is not presented as the GPU footprint:
GB10 CUDA allocations can remain outside that charge, which is why the guard
adds the complete CUDA reservation.

PB action `b95221b1984bddbae27475fe0b788b2d596ca191d6c526b43ef7ee74654e3cb2`
ran on Sparklina with four CPUs, exactly 104 GiB physical admission and a 92 GiB
GPU subset. Actual exit was zero, cleanup completed, and OOM/local OOM were zero.
The unchanged physical guard used its 2 GiB margin and 8 GiB host floor.
Source is `0447071be47205c6a924cbc6a9c20bd60b54a5cd`, with runtime
`34388d2bfeea0e76830645aab92d15132f766bcf` and the same content-pinned producer
container/PyTorch 2.13.0+cu130 used for the final tiny GLM ABBA. The snapshot
adds only PB's closure declaration. Eleven CPU geometry, hash and bounded-sample
checks passed with zero skips before the native launch. Native result/trace
hashes, ledger identity, terminal outcome, CAS payload and claim binding were
checked, including a second independent coordinator audit.

Evidence: [result](result.json), [exact invocation](invocation.json),
[CPU audit](cpu-audit.json), [native audit](native-audit.json),
[raw-profile observations](profile-observation.json),
[independent PB audit](independent-pb-audit.json) and
[independent artifact audit](independent-artifact-audit.json).
Raw traces and Netdata remain under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/joint-allocator-turnover-01/native-01/`.
The separate [three-layer ABBA](../glm-joint-operator-windows-20260908/final-runtime.md)
is numerical/owner evidence at tiny scale, not a substitute for native operator
matrices at the planned 32 GiB statistics scale.
