# Final native operator-window runtime gate, 2026-09-08

Runtime `34388d2bfe` passed the complete native ABBA: 21 whole-probe calls
(A1/B1/B2/A2: 8/3/3/7) on a genuine three-layer original-layout GLM fixture,
with two source slots and lookahead one. Both arms use exact boundary schema
`prismaquant.aura.boundary_storage.v2`, `capture_order=layer_major`.
The 44 decoder targets include 12 packed expert projections. Actual registry
RTN FP8_E4M3 and NVFP4A16 donors plus BF16 passthrough produce 132 candidate
rows per call. This remains a synthetic diagnostic, not full GLM fit, quality,
bpp, serving, production throughput, or a 32 GiB operator qualification.

The original 3×17 tokens, complete-sequence B1 partitions, K=3 seeds 7000–7002,
dynamic activation QDQ, genuine initialization seed 20260826, and reference
KDA/convolution fixture kernels are unchanged. Ninety-one original source/PWC
files were hashed before and after every arm and rechecked after execution.
The failed 8 GiB diagnostic and successful 12 GiB retry have identical relative
source/PWC file inventories. Runtime, profile settings and numerical tolerances
also stayed unchanged across that retry.

Every call passed 1,584 signed weight/activation/mixed/total comparisons using
`math.isclose(rel_tol=3e-5, abs_tol=2e-9)`. Windowed maximum absolute error is
`1.862645149230957e-9`. Its largest relative error for reference magnitudes above
`2e-9` is `6.427148274310688e-5`; that comparison passes the unchanged absolute
criterion. Dense and packed targets both have nonzero weight, activation and
mixed components. All 162 outgoing window replay cotangents match the 27 unique
baseline layer/batch/probe cotangents exactly; all nine tail cotangents match.

Each call reads all three decoder layers once in forward capture, then only the
evicted first layer once in reverse: four original-layout reads total. Target
windows do not reread source layers. Every completed boundary generation has
zero hot-read misses, zero live artifact bytes and zero resident tensor bytes.
The windowed boundary peak is 26,112 tensor bytes, with 2,601 auxiliary bytes.
All source parameters remain frozen without gradients in operator replay.
Weak backing-storage checks verify statistics expiration at projection finish,
no previous candidate/statistics owner at the next lease, and no tracked owners
at return.

| Observation | Legacy A1/A2 | Window B1/B2 |
| --- | ---: | ---: |
| Peak observed PWC backing bytes | 180,224 | 16,384 |
| Retained dW / candidate backing bytes | 360,448 | 32,768 |
| Retained statistics backing bytes | none | 65,536 |
| Peak CUDA allocation | 76,279,808 | 75,761,152 |
| Peak CUDA reservation | 94,371,840 | 94,371,840 |
| Median instrumented whole-probe time | 1.210 / 1.260 s | 6.985 / 7.001 s |
| Actual CUDA kernel events per profile | 64,143 | 274,162 |
| Sum of kernel event durations | 84.56 / 83.82 ms | 349.05 / 349.11 ms |
| Repeated complete calls per GPU joule | 0.05003 / 0.04666 | 0.01093 / 0.01075 |

Backing-storage peaks describe separately observed owner lifetimes, not a
summed process peak. The observed CUDA allocation reduction is 518,656 bytes.
The instrumented throughput and energy regressions are explicit. This fixture
runs upstream reference attention, fresh source setup and ownership observation;
B1 cProfile attributes 1.415 seconds cumulatively to 54 physical snapshots and
0.219 seconds to 452 guarded allocation checks. Raw traces preserve CPU/CUDA
execution with allocation-event and shape recording disabled equally in both
arms. Nested CPU durations are not treated as self time.

Netdata contains 93 samples from each Spark. Sparklina whole-box CPU busy during
repeat windows averages 6.01–6.30%. The independently retained pqteld interval
has approximately 0.5-second power sampling; trapezoidal integration uses samples
bracketing each repeat interval, with maximum gap 0.501 seconds. GPU energy is
139.92/182.92/186.13/128.60 joules for 7/2/2/6 repeated complete calls. This
includes source setup and observer work, subtracts no idle power, and is not
whole-system energy. Legacy achieves about 4.47 times as many complete diagnostic
calls per GPU joule across the combined repeat windows. Whole-action GPU power
mean/peak is 11.51/12.83 W against the recorded 140 W envelope, establishing that
this tiny diagnostic does not saturate the GPU.

A separate bounded native allocation transition inside the same action allocated
and touched a 256 MiB statistics proxy, dropped its owner, then used the actual
`check_operator_allocation` and `CaptureMemoryGuard`. Inactive CUDA reservation
fell from 274,726,912 to 2,097,152 bytes. A live 1,048,576-byte tensor retained its
allocation and sampled values. This validates real allocator retirement at that
size; it does not establish the later planned-size allocator gate.

PB action `7118eea155202f51d15ea6a7c7764f696670d27e2a5ead6fa87e03c0519e494d`
ran on Sparklina with four CPUs, 12 GiB physical memory and a 4 GiB GPU subset.
It exited zero, cleanup completed, OOM/local OOM were zero, and cgroup peak was
7,481,188,352 bytes. The actual guard retains its 2 GiB margin and 8 GiB host
floor. Harness source is `fbdeadf954c9685d94525ea37a8be3d8084987c7`; snapshot
changes are limited to PB's closure declaration. The container remains pinned
by content hash `eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd`,
PyTorch 2.13.0+cu130 / CUDA 13.0, with native math/source reads limited to one
thread and PWC prefetch to two workers.

The preceding action `b4bd478c7d943b28a84bab40ae6cb7cce0ddd9b8d76a0c7f9a7cd55908c2e9c8`
is a retained profiler-budget negative. Both profiled window calls and two B1
repeats passed, but the next B2 repeat refused at entry: cgroup 6,941,057,024 plus
CUDA reservation 85,983,232 exceeded 8 GiB minus the 2 GiB margin. At refusal,
anonymous memory was 4,529,737,728 bytes and file cache 2,361,040,896 bytes; the
two window traces were about 914/916 MB. Actual exit was one, cleanup completed,
OOM/local OOM stayed zero, and peak was 7,386,832,896 bytes. The successful retry
reserved the larger diagnostic footprint without changing runtime or inputs.

Fifteen fixture/schema/instrumentation CPU tests passed through PB with zero
skips. Four independent trace-analysis actions and one input/boundary/energy
summary action also exited zero through PB; their source snapshots, payload
hashes and local claim bindings were verified. These audits do not claim an
independent worker-attestation signature verification.

Evidence: [summary](final-summary.json), [native audit](final-native-audit.json),
[exact invocation](final-invocation.json), [CPU gate](final-helper-cpu-audit.json),
[trace audits](final-trace-analysis-audit.json), [summary audit](final-summary-audit.json),
[negative attempt](three-layer-profiler-budget-negative.json) and
[retry input identity](retry-input-identity.json). Raw traces, profiles, power
CSV, Netdata, generations and signed rows remain under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/joint-operator-windows-01/native-06/`.
The [initial two-layer result](README.md) remains bounded prior evidence.


## Root integration verification

The complete CPU population covered 422 test files: 6,758 passed, 187 skipped,
and three expected failures. The fifteen profiling-harness checks cover the
additional test file separately, with zero skips. The CPU environment used
Torch 2.10.0 and Python 3.12 on dl380g10; native evidence above uses the pinned
CUDA container. CUDA-only skips are not native passes.

The first PB fanout reserved 5 GiB for each two-worker shard. Six shards hit
that cgroup limit and exited 137 with confirmed local OOM events and complete
cleanup. Only their 159 incomplete files were resubmitted through PB fanout,
using 12 GiB per two-worker shard. All six completed. The ten successful original
shards were retained. Snapshot, CAS payload, terminal and exact file-population
checks cover all sixteen successful quanta; no interrupted shard is counted.

The root independently checked all 21 native call records, 91 immutable input
files, four trace hashes and trace-analysis receipts, source/render/probe
identities, signed-component tolerances, cotangent hashes and owner expiration.
Independent integration of the retained power CSV reproduces the four GPU
energy values above. The actual native snapshot matches `fbdeadf954` except
for PB's closure declaration. The allocator and exact-lookahead regressions
both failed before their fixes and passed afterward.

Root records remain under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/`:
`joint-operator-combined-full-cpu-root-audit-01.json`,
`joint-operator-combined-full-cpu-oom-evidence-01.json`,
`joint-operator-allocator-root-audit-01.json`, and
`joint-operator-lookahead-root-audit-01.json`. Native records are in the
`joint-operator-windows-01/` subdirectory:
`root-native-06-pb-audit.json`, `root-native-06-artifact-audit.json`, and
`root-native-06-energy-audit.json`.
