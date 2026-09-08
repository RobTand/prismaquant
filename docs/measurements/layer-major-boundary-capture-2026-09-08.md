# Exact layer-major boundary capture — 2026-09-08

Issue [#394](https://github.com/RobTand/prismaquant/issues/394),
PR [#400](https://github.com/RobTand/prismaquant/pull/400).
Implementation `cf1271fc3d`; native token-device fix and qualified source
`02ef6a5630`. Shared-cotangent replay forks are the separate `0e8c501cfa`
commit. All test and GPU execution below used PrismaBuild at priority -10.

The explicit v2 boundary policy extends `StreamedCausalLM.visit_layer_batches`
with the existing exact activation artifact owner. Each source layer installs
once and processes original complete batches in order, reading checked input
windows and immediately writing exact output boundaries. Per-batch source
kwargs and profile state remain separate. V1/default capture and the reverse
probe/projection traversal are unchanged. The new policy requires evaluation
mode, prefetched source residency and bounded auxiliary storage, and refuses
observed Torch CPU or runner-device CUDA RNG consumption. This does not certify
arbitrary mutable custom models or other RNG domains.

Evidence root:
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/`.
`layer-major-boundary-implementation-01/` holds terminal/log audits, verified CAS
receipt/result hashes and `native-evidence-audit.json`. Exact native command and
container environment are in `layer-major-boundary-native-invocation-02.json`.
Raw traces, cProfile files, cost panels, tensor digests and both-host telemetry
are retained in `layer-major-boundary-native-02/`.

| Gate | PB action prefix | Actual result |
|---|---|---|
| Before-change source-install regression | `9bf0eecaada7` | Failed: 10 installs, expected 2 |
| Initial implementation fixtures | `a11184dbf764` | 52 passed, 11 failed: fake context lacked the existing `prefetch_following` argument; fixture corrected |
| Boundary/state/replay implementation | `e0a3e4bd90c0` | 64 passed in 17.73 s |
| Broader CPU and four module compile checks | `e81c87c46a25` | 122 passed, no skips, 274.28 s |
| First native experiment | `14a3ee3087f8` | Baseline arm passed; layer-major refused CPU/CUDA token comparison, fixed in `02ef6a5630` |
| Native paired qualification | `63dffca2d164` | Exit 0, all four arms passed |
| Final targeted CPU and independent saved-cost comparison | `b8a94df84c6a` | 45 passed, one CUDA-only skip, 16.59 s; four saved panels matched |

CPU tests used `/home/rob/venvs/pq-cpu312/bin/python` on DL380, with four or
six physical CPU workers and native threads bounded to one. The broader gate
included existing collector-source-release and actual GLM campaign/visitor
tests, plus architecture and staleness checks. The final CPU skip is
`test_original_cpu_tokens_compare_with_prepared_cuda_tokens`: the CPU worker
has no CUDA. The genuine CUDA native rerun exercised the same original-CPU /
prepared-CUDA token path after its recorded regression. Actual Gemma4 shared
state, probe adjoints, budget refusal, cleanup and quiescent replay forks passed
in CPU fixtures; the native GLM fixture has no nonempty shared-KV state.

The native experiment used Sparklina GB10, image
`prismaquant-glm-producer:content-qualified-20260908`, observed image ID
`sha256:9f9b9f05b17531399ba66dc6415b054cf5d68c82270626d0e9150e75c808435f`,
content seal `eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd`,
Torch 2.13.0/CUDA 13.0. PB reserved four physical CPUs, 8 GiB total UMA and a
4 GiB GPU subset with a 600 s ceiling. Actual scope peak was 3,069,296,640
bytes; terminal exit, scope cleanup and an empty container list were checked.

The genuine original-layout tiny GLM has three layers (one KDA, two DSA), two
source-cache slots, H=64, mHC=4 and four routed experts. It executes five
original B1 sequences of length 17 and four probes with IDs 7000–7003 in BF16.
The upstream Torch reference KDA/conv functions run on CUDA explicitly; this
is not a fused-kernel qualification. Synthetic source+0.03125 candidate tensors
cover 33 profile-unpinned dense/packed targets with the same format assignments
in every arm. No full GLM checkpoint or serving artifact was produced.

| ABBA arm | Capture order | Actual source reads / installs | Materialized source bytes | Capture seconds | Full instrumented call seconds |
|---|---|---|---|---|---|
| 0 | batch-major v1 | 15 / 15 | 1,755,300 | 1.218376 | 4.185343 |
| 1 | layer-major v2 | 3 / 3 | 351,060 | 0.362056 | 3.169947 |
| 2 | layer-major v2 | 3 / 3 | 351,060 | 0.369237 | 3.145303 |
| 3 | batch-major v1 | 15 / 15 | 1,755,300 | 0.579450 | 4.739085 |

Source read counts and materialized storage come from actual reader calls,
not estimated checkpoint traffic or disk-byte counters. V2 reduced these
capture calls by 80%. All costs and h-trace/Fisher columns matched exactly,
including an independent PB read of the four saved `.cost.pt` panels. All 80
propagated cotangent hashes and each batch's input/output/position hashes
matched; reverse global call order also matched. Initial global forward order
deliberately changes from batch/layer to layer/batch.

All arms capped owned exact boundary residency at 43,520 bytes, with auxiliary
peak 4,335 bytes. V2 adds 130,560 logical boundary input bytes during capture:
whole-call exact reads were 1,218,560 versus 1,088,000 bytes. All arms wrote
870,400 tensor bytes in 100 entries, retired all 100, ended with zero live
artifact bytes and recorded zero hot-read misses. This trades explicit bounded
activation reads for amortized source loading.

Capture-only Torch CPU/CUDA profiles are retained for every arm. Self CPU
totals were 1.132 s, 316.424 ms, 323.200 ms and 474.330 ms; self CUDA totals
were 12.736, 11.756, 11.755 and 12.559 ms. Full-call cProfile and process I/O
snapshots are also retained. Instrumentation, initialization and durable file
operations dominate this small fixture, so the timing table is not a
production throughput claim. CUDA allocation peaks were 76,018,688 bytes in
the cold first arm and 76,449,280 thereafter; reserved peak was 94,371,840
bytes throughout. No physical-memory reduction is claimed.

Both-host Netdata recorded six samples per host at five-second cadence:
Sparklina GPU power 4–12 W, CPU nonidle 5.02–7.58%; Sparky 13 W and
6.13–6.62%. The short capture windows and coarse power cadence cannot support
work-per-joule ranking. The fixture does not saturate GB10; GPU utilization is
not used as a saturation diagnostic. Source, candidate, gradient and scratch
budgets, full-scale physical memory admission and production qualification
remain independent. The new mode stays default-off.

Successful native CAS receipt:
`220df31f692a73fe16f497d6c58eaa37e48c00a2031d39dca4aab6c12209392e`;
result `8730acb55a33231e40014ee54fda44f038a5ffbd48d1cc092452f570b3ddabb7`
(7,297 bytes). Canonical receipt hash, result length/hash and retained evidence
file hashes were independently checked. Failed native output remains bounded
evidence in `layer-major-boundary-native-01/`; it is superseded by the complete
paired qualification and is not counted as a passing gate.
