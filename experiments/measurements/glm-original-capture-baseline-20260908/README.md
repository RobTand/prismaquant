# Partial original GLM capture: saved trace observations, 2026-09-08

Six completed profiler windows from the still-running original 512 × 512
calibration capture were analyzed without replaying GPU work. This is a
single-run baseline observation, not a before/after performance comparison or
a completed-capture claim. The capture's existing journal and calibration draw
were retained. Native source is frozen at `e22a0286a820b23f2aaaa6a9232912abf49394c5`,
PB action `8740a0b3456bb6cb334ae80b0e35fc3da31c62918abdda8c41ad226020c4e88a`,
with six CPUs, 104 GiB physical admission and 92 GiB GPU subset on Sparky.

Each listed window contains two original batch forwards. Window 00 observes
batches 0–1 and window 01 observes batches 31–32. Actual CUDA kernel events
are counted below; profiler annotations and overhead events are excluded.
The largest kernel family is FP32 elementwise multiplication for collection 0
and `CUDAFunctor_add<float>` for collections 3 and 4.

| Trace | Actual kernels | Summed kernel ms | Largest family ms | Family share |
| --- | ---: | ---: | ---: | ---: |
| collection-00-window-00 | 1,939 | 196.076 | 53.283 | 27.17% |
| collection-00-window-01 | 1,936 | 201.237 | 52.474 | 26.08% |
| collection-03-window-00 | 18,327 | 866.374 | 305.057 | 35.21% |
| collection-03-window-01 | 17,970 | 1135.572 | 602.049 | 53.02% |
| collection-04-window-00 | 19,331 | 992.583 | 305.585 | 30.79% |
| collection-04-window-01 | 18,948 | 1290.824 | 612.955 | 47.49% |

The raw Torch summary for collection 4, window 01 attributes 612.346 ms of
CUDA time to `aten::add_` (1,150 calls) and 487.548 ms to `aten::mm`
(2,906 calls). Its CPU self-time table includes 531.745 ms labelled
`Command Buffer Full`. Kernel family totals and framework operator totals are
different views and must not be added together. Nested CPU event totals in the
JSON summaries overlap and do not represent CPU self time.

The existing Hessian accumulator at
`prismaquant/tessera_campaign.py` computes `gram = f32.t() @ f32`, then adds
that matrix to the existing Hessian. This is a candidate site for future
profiling and an explicit arithmetic experiment. These traces do not establish
that replacing it with fused accumulation is bit-exact or faster, and no such
change was made to this capture or its calibration artifact.

## Complete collector intervals, rather than tiny profiler windows

The table below integrates the GPU power recorder over each complete
instrumented collector call, including its 512 original batches and collector
output work. Linear interpolation at the interval endpoints and trapezoidal
integration were used; the maximum recorded sample gap was 0.501 seconds.
There is no idle subtraction or full-system energy claim. The layers perform
different work, so these rows are not competing implementations.

| Collection | Wall seconds | GPU joules | Mean GPU W | Batches/GPU joule |
| --- | ---: | ---: | ---: | ---: |
| 0 | 54.581 | 2,657.50 | 48.689 | 0.192662 |
| 3 | 322.199 | 13,123.10 | 40.730 | 0.039015 |
| 4 | 358.921 | 14,449.37 | 40.258 | 0.035434 |

The MoE collector means are about 29% of the roughly 140 W GB10 envelope.
This power observation alone does not identify the limiting resource.
Both-host Netdata was retained over the same intervals. Sparky's mean CPU
busy fractions were 21.68%, 13.70% and 11.52% for collections 0, 3 and 4;
mean CPU iowait stayed below 0.55%. Sparklina's mean CPU busy fractions were
2.01%, 2.40% and 1.60%, and its coarse maximum GPU gauges were 4, 12 and 12 W.
These are whole-host observations; no claim is made that all CPU activity
belongs to the capture. GPU utilization percentage was not used as a
saturation measure.

## Reproduction and evidence

`campaign.json` pins the existing streaming JSON analyzer by SHA256 and
submits each of the six independent immutable trace files through PrismaBuild:

```
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbcampaign.py \
  /mnt/shared/tessera-measurements/glm-canonical-census-20260908/full-capture-trace-analysis-01/campaign.json \
  --wait-s 1200
```

All six CPU actions completed with exit 0 on dl380g10, each with one CPU and
1 GiB memory. Source snapshots, canonical CAS receipts, payload hashes and
scope cleanup were independently checked. Each saved summary is exactly equal
to its CAS-backed JSON output. The parser checks the complete raw trace hash
before and after bounded streaming analysis.

`root-pb-audit.json` and `root-artifact-audit.json` carry these checks.
`power-observation.json` names the retained recorder CSV and exact interval
boundaries. `netdata-observation.json` names the immutable both-host slice and
its digest. The continuously growing source telemetry was not hashed as if it
were an immutable complete run. Raw traces remain in `full-capture-profile-03`.
All paths above are under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/`.
