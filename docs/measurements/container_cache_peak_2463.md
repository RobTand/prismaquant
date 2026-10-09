# PQ #2463: container-cache peak on a complete Stage B quantum row

Part of PQ #1091. The cache ROOT/MAX pair declares an operator-chosen
reservation. Two GPU rows ran a complete Stage B quantum through
PrismaBuild with cache writes observed. Peak bytes are recorded with
immutable runtime evidence. The declared ceiling derives from the peak.

## Workload and scope

`tools.measure_container_cache_quantum_row` runs in the known-good
campaign container (image `sha256:c0e532d28a78b3bf425bbbc0d862e2840ba624249162aedfadd09748a6c68c37`,
content `d0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026`;
torch 2.13.0+cu130, triton 3.7.1, transformers 5.16.1, NVIDIA GB10,
capability 12.1). The row runs Stage A adjoint capture on a tiny
two-layer bf16 fixture model, then one Stage B layer quantum for that
capture's slice (unit `model.layers.1.proj`, formats FP8_E4M3/NVFP4A16/
BF16, two probes, seed 7000), then the served NVFP4 RTN activation
quantiser compile (`torch.compile`, fixed tensor 64x256, seed 2463) and
the six KDA capture kernels (`triton.jit` launches, probe digest
`844e6ad2f98c481dcb6de63099428b7d49fa6b464da2f89dc4edf2afd6235f45`).
The quantum never quantizes a fixture input, so the compile probe runs
the served path the fixture leaves cold; both write under the routed
cache root. Caches route under `PRISMAQUANT_CONTAINER_CACHE_ROOT` (hf,
triton, inductor, xdg). `PRISMAQUANT_TMPDIR` sits in a separate charged
cotangent workspace. Each row starts from a cleared empty cache root.
The sampler forks a child with its own GIL, walks the root every
250 ms, and keeps the largest observed sample.

## Measured result

| Row | PB action | Peak bytes | Samples | Valid | Receipt digest |
|---|---|---:|---:|---|---|
| 1 | `628ceda204fb2e1b5e126c66926f9da49d8da3945006d540d058ba0880bac078` | 3596288 | 27 | true | `074ba9750361c0288f463b3fecb66b26365431023181942ef104e78a14f7134c` |
| 2 (repeat) | `02da0e85bb7472fc6f2ba0a5df2940fd046449475cbc5977585b8dc72cc94a4e` | 3596288 | 27 | true | `a52a2c5e341020f0876ec6bd81691a7fdb173502aa66fcdd143484adc5e08c61` |

Both peaks are identical. Each row grew 4096 to 3596288 bytes over
6.54 s (88 files, 23 dirs at peak; the peak sits at the last sample).
Initial state is the cleared root (4096 bytes, 1 dir, 0 files), captured
before the sampler starts. No scan errors, no gaps, no incomplete scans.
The repeat declared the 1 GiB ceiling and charged spool_gb 10 (cache 2
+ cotangent 8 on row 1; cache 1 + cotangent 8 on row 2); both peaks fit.
Sealed demand: gpu=1, mem_gb=32, gpu-mem 20 GiB, 4 CPUs, 1500 s timeout,
progress phases row/head/layer-001-chunk-000 at 600 s.

Receipts: `container_cache_quantum_row1_2463.json`,
`container_cache_quantum_row2_2463.json`. Both-Spark Netdata for the row
1 window: `container_cache_quantum_row1_netdata_2463.json` (sparky power
mean 10.6 W, max 12 W; sparklina idle 4 W). In-process cProfile tops ride
in each receipt (`profile_top`; full text on the worker). Row 1 in-process
power: 76.36 GPU joules, peak 13.25 W.

## Derivation

`C = G * ceil((P + H) / G)` with `G = 1073741824`, `P = 3596288`,
`H = 4341760`. `H` is four times the largest single 250 ms interval growth
(1085440 bytes): one more unseen jump past the final sample (the peak sits
at the last sample), one concurrent compile burst, scan-gap margin.
`P + H = 7938048`, so `C = 1073741824` (1 GiB). Fixture:
`tests/fixtures/container_cache_ceiling_2463.py` binds both digests, both
peaks, the inputs and the ceiling; its tests rehash both receipts and
check the headroom against the recorded samples.

## Limits

This ceiling covers this workload, runtime, concurrency and initial cache
state only. It proves no quota enforcement, cleanup or crash recovery.
Those stay with PQ #1091 and PB #1360.
