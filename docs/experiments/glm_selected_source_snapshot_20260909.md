# GLM selected source snapshot measurement — 2026-09-09

The opt-in `selected-tensors-v1` source policy reduced complete-group wall time
from 265.02 s to 236.40 s (1.121× throughput, 10.80% less time) and device energy
from 6757.74 J to 6386.74 J (5.49% less energy, 1.058× anchors/J). All four arms
matched the existing production group's 14 anchors, measured error, bit costs,
and authenticated wire files exactly. This is one shared-expert gate/up group,
not a routed-expert or end-to-end model-quality result.

## Workload and environment

The workload is GLM-5.3-Flash-BF16 layer 4 shared-expert gate/up: two Linears,
three dense families (BF16 K1, E2M1 K2, E4M3 K1), rate band 832–1088, three
initial anchors, adaptive anchor budget 12, and 512 retained scoring rows.
It reuses the original 512 × 512 calibration capture, manifest SHA256
`f4bcbf408d3aa81b04c1fabd1d2d7457176a95dcccdd5ed368800de5c37e277c`.
No new source-model forwards or calibration draw were performed.

PB selected Sparky, NVIDIA GB10, driver 595.84, Linux 6.17.0-1032-nvidia.
The qualified producer image content digest is
`eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd`;
PyTorch 2.13.0+cu130, CUDA 13.0; producer source and encoder settings are bound
in the measurement plan. The action reserved six CPUs and 61 GiB total/shared
GPU memory, including the control's 57 GiB demand and 4 GiB profiler allowance.
PB preserved affinity and isolated the GPU measurement. Sparklina continued
its existing expert-pricing action and has Netdata coverage for every arm.

Each A/B/B/A arm ran in a fresh Python process with separate Triton/Inductor
caches. The host page cache was shared. Both policies used identical profiling:
main-thread stacks and `/proc/self/io` sampled each second, Netdata on both
boxes every five seconds, and two bounded CUDA activity windows. Reported wall
times include that instrumentation; they are not unprofiled production timings.

| Arm | Source policy | Wall seconds | Device joules | Pre-anchor seconds |
|---|---|---:|---:|---:|
| A0 | Whole layer | 265.580 | 6658.49 | 64.43 |
| B1 | Selected tensors | 235.768 | 6357.15 | 35.21 |
| B2 | Selected tensors | 237.033 | 6416.33 | 35.68 |
| A3 | Whole layer | 264.452 | 6856.99 | 63.56 |

## What changed and what remains

The shared streaming reader/cache now loads only the authenticated dependency
closure needed for the selected weight snapshots. Head, embedding and vision
weights remain meta; an expert view still requires its complete packed parent.
Source-file authentication remains mandatory. Admission for this group falls
from 57 to 42 GiB; the complete source-validation allowance remains charged.
Actual concurrent throughput under the smaller reservation was not measured.

The profiles place nearly all the improvement before the first anchor:
63.99 → 35.44 s on average. The child's sampled `read_bytes` delta falls from
47.64/47.71 GB to 6.162/6.163 GB, and `rchar` from 32.83/32.91 GB to
7.276/7.279 GB. These are kernel process-I/O counters, not network-wire bytes.
Main-thread samples in `wait` fall from 24 in the first control to 5/6 in the
selected arms. The encoding phase is essentially unchanged.

Device power averages 25.50 → 27.02 W and peaks around 40 W, well below the
approximately 140 W envelope. The optimized steady-state BF16 CUDA window
attributes 92.86% of recorded device time to `_step_best` (341,849 calls,
4.721 µs average). This remains a candidate for further batching/kernel work;
this source change does not establish saturated encoding. CUDA/CPU event
correlation is not used because the activity-toggle warning makes it unreliable.
GPU energy integrates the 0.5-second `pqteld` device-power samples over each
arm with full temporal coverage and no gap above two seconds. It excludes
whole-host electrical power.

## Evidence and validation

Artifacts live under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/selected-source-snapshot-optimization-03/`:
`plan.json`, `native/result.json`, `native-analysis.json`,
`native-power-window.json`, per-arm profiles and parity records, and
`native-cas-source-audit.json`. The analysis script used is retained beside this
evidence as `analyze-selected-source-ab.py`.

Native PB action:
`782b0a7b31cce8a79a4e9d38a70b6fa523b4231001c71f874c764091b0480615`.
Its terminal exit is zero, scope cleanup complete, CAS receipt/payload verified,
and source snapshot identical to `8f1cc63705` except PB closure metadata.
Later commits tighten pre-I/O guards and seed-score identity; CPU regressions
cover those changes. They do not change the measured encoder or source-copy path.

The initial attempt in `selected-source-snapshot-optimization-02` failed because
one steady-state trace exceeded its 256 MiB export cap. Its completed anchors
were retained as negative evidence; it was not accepted as an A/B result.
The successful run raised only that bound to 512 MiB. The bound checks exported
bytes and is not a live profiler-memory limit.

The source change's CPU suite passed 215 tests with one CUDA-only skip, followed
by eight focused snapshot tests. Review regressions reproduced three failures;
the guard fix passed 31 tests with one CUDA-only skip (PB `59d2c85f2425`).
Seed-score validation reproduced adoption across changed scoring activations
(PB `063e64c1d28c`), then passed 113 tests with two skips (`554c9024efc5`), plus
an explicit unchanged-input adoption test (`d7a152352d31`). Workspace reuse
passed six tests (`039b8e22935c`); integrated planner/docs checks passed 58
(`451865c80e10`). Counts overlap and must not be summed. Actual source/receipt
audits are stored with the evidence. No skipped check is represented as a pass.

An independent finding, PrismaBuild #451, showed that the deployed terminal
process-I/O aggregate can omit exited descendants. This report therefore uses
the child's own sampled counters; the old aggregate is not treated as zero I/O.
The GB10 fleet limit remains 104 GiB. Its reactive GPU guard cannot establish
the user's requested guarantee for a higher hard aggregate cap (PB #450).
