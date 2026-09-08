# Native operator-window probe qualification, 2026-09-08

The initial end-to-end ABBA on runtime `1f7b135ee9` passed 33 complete probe
calls, with 33 genuine tiny GLM decoder targets, including 12 packed expert
projections. Every call produces 99 candidate rows for actual registry RTN
FP8_E4M3 and NVFP4A16 PWC donors plus BF16 passthrough. Source weights use the
original GLM checkpoint layout, with genuine random initialization seed
20260826. This is a synthetic calibration diagnostic, not a full GLM fit,
production throughput, quality, bpp or serving gate.

Both paths use exactly the same 3×17 tokens, complete-sequence B1 partitions,
three Rademacher probes with seeds 7000–7002, original source and BF16 PWC bytes.
Sixty-nine source/PWC files were hashed before/after each arm and independently
rechecked after the run. Existing dynamic activation QDQ uses no static clipping
override. The existing tiny GLM test's upstream reference KDA/causal-convolution
implementations are named in the result; no source routing or forward math is
replaced by the operator-window observer.

All 1,188 signed weight/activation/mixed/total components per call satisfy the
fixed relative tolerance `3e-5`, absolute tolerance `2e-9`. The windowed path's
maximum absolute error is `9.313225746154785e-10`; maximum relative error for
reference values above the absolute tolerance is `2.7188278830904012e-5`.
Dense and packed targets both have nonzero weight, activation and mixed terms.
The comparator preserves source/render/probe identities while allowing the
runtime's explicitly different operator accumulation arithmetic identity.
This is parity against the unchanged legacy probe, not a new native FP64 oracle.

Every windowed invocation has 108 exact outgoing replay cotangents matching the
18 baseline layer/batch/probe cotangents. All nine tail cotangents also match
exactly. Both source decoder layers are read once per complete call, with no
reads per target window. Visual-tower initialization is counted separately.
There are three distinct windows containing packed targets. Each window arm
asserts every source layer parameter has `requires_grad=False` and `grad=None`
before replay, and no layer unload retains parameter gradients.

| Observation | Legacy A1/A2 | Window B1/B2 |
| --- | ---: | ---: |
| Calls, including each arm's profiled call | 13 / 12 | 4 / 4 |
| Peak observed PWC backing bytes | 180,224 | 16,384 |
| Retained dW backing bytes | 360,448 | 32,768 candidate quantum |
| Retained statistics bytes | none | 65,536 |
| Peak CUDA allocation | 76,274,176 | 75,761,152 |
| Peak CUDA reservation | 94,371,840 | 94,371,840 |
| Median instrumented whole-probe time | 0.4751 / 0.4767 s | 3.6397 / 3.6530 s |
| Actual CUDA kernel events per profiled call | 39,547 | 158,254 |
| Sum of kernel durations | 52.99 / 52.28 ms | 202.88 / 202.67 ms |

Backing-storage observations use weak storage references, including aliases.
Every statistics matrix expires at projection completion; no earlier matrix or
candidate survives into the next lease, and no tracked backing storage remains
at return. PWC, dW and statistics peaks describe their separately observed
lifetimes, not a summed whole-process peak. The real CUDA allocation delta is
513,024 bytes on this tiny fixture; reserved device memory is unchanged.

The timing regression is explicit. Fresh source setup and ownership/process
observation remain in these diagnostic calls; they are not production throughput
measurements. The whole-probe cProfile attributes about 0.879 s of B1's 5.165 s
profiled call to 36 observer physical snapshots, and 2.187 s cumulatively to
`replay_backward`. Raw CPU/CUDA traces show the extra replay work. Nested CPU
durations overlap; the trace summary does not label them as self time or count
GPU annotation intervals as kernels.

Both Sparks have 76 Netdata samples. During the measured arms, Sparklina CPU busy
averages 5.49–5.88% of the box. Only two distinct, lagged GPU power observations
occur per arm, at 11–13 W, so work-per-joule ranking is not established. PB's
whole-action power mean/peak is 11.32/12.9 W against the recorded 140 W envelope.
This tiny probe is not GPU saturated.

PB action `f7a7f5cf766a8b27fc74fbb2c06c1a75d1d77e5419c563891ddf7a6cd0577598`
ran on Sparklina with four CPUs, 8 GiB physical DRAM and a 4 GiB GPU subset.
Actual cgroup peak was 5,307,764,736 bytes, exit zero, no OOM, cleanup complete.
The 8 GiB guard, 2 GiB margin and host free-memory floor remain enabled.
Runtime is `1f7b135ee97e6e0bb48e66c5d3cd5764c1a14e9e`; measured harness source is
`6a786a7120c40298188cf07a77958bfcd180f343`. The pinned producer container uses
PyTorch 2.13.0+cu130/CUDA 13.0 on GB10. Native math and source reads use one
thread, source prefetch one worker/two slots, and PWC prefetch two workers.
The snapshot differs from the named source only by PB's closure declaration.
CAS payload, local claim and cleanup evidence were independently checked.

Thirteen CPU instrumentation tests passed through PB, zero skips; the final
pre-export persistence code also passed PB compilation before native execution.
Four independent CPU trace-analysis quanta ran through PB on DL380G10 with one
CPU/1 GiB each. They stream JSON events with a bounded buffer, verify trace
hashes, and check actual kernel/CPU event coverage. All four exited zero and
their CAS payloads were checked. Large traces were never parsed in the GPU job.

Negative attempts are retained, not counted as qualification:

- `d0d5f5b545c5`: the fixture's generic Linear collector included visual targets;
  the decoder-prefix gate refused before probing. The harness now restricts
  donors to actual decoder targets, with a regression test.
- `af6dd5ca885b`: full shape/allocation recording followed by in-process
  `key_averages()` hit the 8 GiB cgroup limit after the first window trace was
  exported. The action exited 137 with one local OOM kill. No numerical gate
  success is inferred from those earlier log lines.
- `16fc435b8357`: removing `key_averages()` preserved the first window's completed
  numerical/ownership gate, but retained trace recorder heap/page cache caused
  the next call's physical guard to refuse. Raw traces plus cProfile remain;
  shape/allocation-event recording is disabled equally in both successful
  arms, with direct CUDA/process counters and weak storage measurements kept.

Each profiled arm persists its checked numerical and ownership result before
trace export. Those files are diagnostic checkpoints; complete ABBA and PB
terminal success remain separate requirements.

The [summary](initial-summary.json), [native audit](initial-native-audit.json),
[invocation](initial-invocation.json), [CPU audit](helper-cpu-audit.json) and
[trace analysis audit](initial-trace-analysis-audit.json) bind these claims.
Raw artifacts are at
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/joint-operator-windows-01/`;
the successful result and four traces are under `native-04/`. Earlier attempts
remain under `native-01/` through `native-03/`.

This initial runtime measurement does not exercise a full-scale 32 GiB operator
reservation or certify future allocator/boundary policy changes.
