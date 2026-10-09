# PQ #2463: container-cache peak on a Stage B compile row

Part of PQ #1091. The cache ROOT/MAX pair declares an operator-chosen
reservation. Two GPU rows ran the Stage B compile workload through
PrismaBuild with cache writes observed. Peak bytes are recorded with
immutable runtime evidence. The declared ceiling derives from the peak.

## Workload and scope

`tools.measure_container_cache_row` runs in the known-good campaign
container (image `sha256:c0e532d28a78b3bf425bbbc0d862e2840ba624249162aedfadd09748a6c68c37`,
content `d0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026`;
torch 2.13.0+cu130, triton 3.7.1, transformers 5.16.1, NVIDIA GB10,
capability 12.1). The row compiles the served NVFP4 RTN activation
quantiser (`torch.compile`, fixed tensor 64x256, seed 2463) and probes the
six KDA capture kernels (`triton.jit` launches, probe digest
`844e6ad2f98c481dcb6de63099428b7d49fa6b464da2f89dc4edf2afd6235f45`).
Caches route under `PRISMAQUANT_CONTAINER_CACHE_ROOT` (hf, triton,
inductor, xdg). `PRISMAQUANT_TMPDIR` sits in a separate charged workspace
(spill `row-tmp`). Each row starts from a fresh empty cache root. The
sampler walks the root every 250 ms and keeps the largest observed sample.

## Measured result

| Row | PB action | Peak bytes | Samples | Valid | Receipt digest |
|---|---|---:|---:|---|---|
| 1 | `16f1927200e31631266851629bd6a88a64548980d69c6a94aae51e4494f19741` | 3596288 | 21 | true | `4acc1cfe81cba1a0c4c9dc84101ffe84ad1c278461da60643390c074c4d001b0` |
| 2 (repeat) | `3459dca86f771632db6f67a7cdc8c0ce3142fcdb299b7146f4b2d2878df1faa0` | 3596288 | 22 | true | `fee88bd132453221dde9911e607aef30f79eccf51fe7c8d0ae89a211108051e7` |

Both peaks are identical. Row 1 grew 12288 to 3596288 bytes over 5.37 s
(triton 2846720, inductor 745472, 88 files, 23 dirs). No scan errors, no
gaps, no incomplete scans. Both rows ran on sparky (GB10 class evidence
from `a471256b52679ed681014b25b66eea51be4890bbd51857fa363e19878f152395`).
Sealed demand: gpu=1, mem_gb=32, gpu-mem 20 GiB, 4 CPUs, 1500 s timeout.
Row 1 charged spool_gb 14 (cot 4 + spill 8 + cache 2). The repeat declared
the 1 GiB ceiling and charged spool_gb 13 (4+8+1); its peak fits.

Receipts: `container_cache_peak_row1_2463.json`,
`container_cache_peak_row2_2463.json`. Both-Spark Netdata for the row 1
window: `container_cache_row1_netdata_2463.json` (sparky power mean 7.27 W,
max 12 W; sparklina idle 4 W). In-process cProfile tops ride in each
receipt (`profile_top`; full text on the worker). PB endings carry memory
peak 3.0 GB, GPU power peak 13.23 W, IO read 1.16 GB / write 94.7 MB.

## Derivation

`C = G * ceil((P + H) / G)` with `G = 1073741824`, `P = 3596288`,
`H = 2605056`. `H` is four times the largest single 250 ms interval growth
(651264 bytes): one more unseen jump past the final sample (the peak sits
at the last sample), one concurrent compile burst, scan-gap margin.
`P + H = 6201344`, so `C = 1073741824` (1 GiB). Fixture:
`tests/fixtures/container_cache_ceiling_2463.py` binds both digests, both
peaks, the inputs and the ceiling.

## Limits

This ceiling covers this workload, runtime, concurrency and initial cache
state only. It proves no quota enforcement, cleanup or crash recovery.
Those stay with PQ #1091 and PB #1360.
