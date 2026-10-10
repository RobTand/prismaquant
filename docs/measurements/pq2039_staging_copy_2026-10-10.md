# PQ #2039 paired staging-copy measure, 2026-10-10

One admitted paired GPU action compares the legacy private staging copy
with pinned-direct adoption. Order is ABBA. Both arms use the same 32
BF16 units (`[1024, 2048]`), the same live CUDA bytes, and the same
prepare, launch, and settle path. The recovery L20 baseline stays the
context for the full pass. This action proves the mechanism on a small
shape. It makes no full-campaign claim.

## Bindings

- Base: `origin/main` at `30b6231df0`. Head: this branch at push time.
- CPU PB: `33408f12bf4ac4bfd796de26b3104130231dba935bdc70bf40e6267da1c597aa`
  (staging file, 2 passed, 4 CUDA skips) and
  `e53bf20ee14c94bc95dd3188e26fe9fc45a871e1753b46cd49df0408f760482f`
  (preparation file, 33 passed, 6 CUDA skips). Interpreter:
  `/home/rob/venvs/pq-task-suite-layer-sdk5-20261009/bin/python`, tag `x86`.
- GPU PB action:
  `71668966e4242996cf3858bb08eb85ec89af9dfbc7244859f353c3641ff1645d`.
  Worker: `sparky`, GB10, `sm121`, driver `595.99.02`. Ordinary pool action
  (not a sealed `--measurement`). Receipt and CAS payload verified.
- Reservations: 4 CPUs, 16 GiB host, 8 GiB GPU subset. Threads:
  `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`.
- Driver: `tools/measure_projected_copy_2039.py`. Schema:
  `prismaquant.pq2039_paired_copy_measure.v1`. Passes: 100 per arm.
  Units per pass: 32. Total per paired side: 6400 unit checks.
- Netdata: worker `http://localhost:19999/api/v1/info` returns 200 with
  16482 bytes. Gap: this action ran on one worker host, so only the worker
  series exists. Both-host Netdata (criterion 4) stays open. No second host
  takes part. Clock alignment stays on HOLD.

## Result

| Arm | Staging `copy_` calls | Wall mean per 32-unit pass (s) | `aten::copy_` self CPU (s) | `aten::copy_` count | Mean W |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before (arm 0) | 3200 | 0.008049 | 0.623849 | 6500 | 6.67 |
| After (arm 1) | 0 | 0.003794 | 0.006052 | 3300 | 12.96 |
| After (arm 2) | 0 | 0.003799 | 0.006096 | 3300 | 13.75 |
| Before (arm 3) | 3200 | 0.007672 | 0.619587 | 6500 | 14.42 |
| Paired before | 6400 | 0.007860 | 1.243436 | — | — |
| Paired after | 0 | 0.003796 | 0.012148 | — | — |

Wall time falls 51.7%. Staging copies fall 6400 to 0. Profiler
`aten::copy_` self CPU falls 1.243 s to 0.012 s (99.0%).

## Attribution

The Python-level spy counts only the prepare-site `pinned.copy_(weight)`.
Before arms record 3200 calls per 100 passes. After arms record 0. The
profiler difference matches: 6500 minus 3300 equals 3200. The common 3300
covers the H2D path (`aten::_to_copy` 3300, `cudaMemcpyAsync` 3300). The
device compare (`aten::ne` 3200, `aten::any` 3200) is equal on both arms.
The removed cost is the prepare-site copy. No second cache exists. The
staged reader fills the pinned buffer. The check adopts it.

## Power and HOLDs

Watts are descriptive samples against the 140 W envelope. Arm 0 shows 6.67
W mean (4 samples, cold first pass 0.033 s). Arm 3 shows 14.42 W mean
(4 samples). After arms show 12.96 W (2 samples) and 13.75 W (2 samples).
Energy integration stays on HOLD. Work per joule stays on HOLD. Clock
alignment stays on HOLD. No saturation, export, serving, KL, or bpp claim.
