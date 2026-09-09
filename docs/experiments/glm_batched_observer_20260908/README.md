# Batched selected-anchor observation — 2026-09-08

The existing experiment wrapper refused `--anchor-batch-size 8` even though
PrismaQuant already supports compatible expert batches. It wrapped only
`_measure_anchor`, so removing that argument guard alone would omit the real
batched calls. The wrapper now observes both existing entry points in one
invocation sequence, records copied member names and batch size, preserves
argument/output ownership and restores both functions after failure.

`--anchor-cuda-only` opts into CUDA activities while retaining the independent
main-thread Python sampler and both Sparks’ Netdata series. Default activity
collection remains CPU+CUDA. Native events are still required: a profiler
without CUDA events is failed evidence, never a native pass. Observation
failure still allows an already successful original call to be journaled once.

This does not alter encoding, calibration, pricing-package source, scheduling
or runtime gates. The old 21ae pricing checkout stays unchanged. The selected
call indices are bounded; the explicit byte cap applies after trace export,
not to live profiler memory. CUDA-only collection is not a time/memory bound.

## Verification

All executions used PrismaBuild. Source snapshots, terminal cleanup, canonical
CAS receipt hashes and actual output bytes were checked. Source differences
from the named commits contain only generated PB closure files.

| Action | Source | Result |
| --- | --- | --- |
| `9f6b24b22d74b58155abc9bb8aece8029171afaf61e3ea3a1de9459a0b32288f` | `943a86b417` | CPU regression: 4 failed, 19 passed |
| `c4dff0d424b8c5aa06146b7610d92682f0cd9f5ea31ca1c40ba6bc2555f1371d` | `0f2400fe4b` | CPU observer: 23 passed, no skips |
| `167c16d35a184b51feecbae5b8f917c60017913d1feee1467b8c2cb1a1ed4aa1` | `0f2400fe4b` | Capture/Netdata: 11 passed; native CUDA capture test skipped on CPU |
| `01f4a1bcd825f48aae500961a333c482caf238726287d31046ce7829d63ae497` | `f4f6470e2f` | Native combined suite: 36 passed, no skips, 10.05 seconds |

CPU checks used the scoped x86 pq-cpu312 environment, CPU2/6 GiB, native threads
one. The native action ran on Sparklina/SM121 with CPU2/8 GiB shared RAM,
4 GiB GPU subset and native threads one. Its known producer image has content
SHA256 `eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd`,
Torch 2.13.0+cu130 and CUDA 13.0. It reused the frozen Tessera source and the
campaign’s existing dependency paths. No model weights or encoder ran.

The native CUDA-only smoke preserved one eight-member 64×64 BF16 batched
matrix-multiply result. Its 25,794-byte trace contains one actual CUDA kernel
event and no CPU-op category, SHA256
`3157c2df1d3cbcf50891f69ad68afce3751beb64534ecac0bccedd8b7af3d980`.
Python samples and both-host Netdata are present. This qualifies instrumentation;
it is not a GLM encoder trace-size, peak-memory or throughput measurement.

Three native submission attempts failed before tests: `a5386c86cb42` lacked the
module import path, `0ae9cde46f4b` supplied a filename where the launcher requires
JSON, and `1538974280b0` lacked Tessera on PYTHONPATH during collection. The final
invocation uses `python3 -m tools.tessera_campaign_container --spec JSON` with
the complete known producer dependency paths. None of those attempts is a pass.

## Retained negative evidence and next gate

The withdrawn broad-band scalar qualification produced rejected CPU+CUDA
traces of 1,297,221,859 bytes (expert) and 7,059,541,982 bytes (dense), exceeding
its 512 MiB exported cap. Those were different, expensive BF16-family work,
not an A/B baseline for this small smoke. Their disposition is preserved in
`full-anchor-preparation-03/root-broad-qualification-final-disposition.json`.

The first narrow-band GLM pricing qualification must still check actual
native trace size/events, physical memory, resident H/X and both-host telemetry.
Width eight retains the planner’s separate 4 GiB observer allowance; this is an
admission plan, not observed encoder fit. No performance improvement is claimed.

Evidence root:
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/batched-observer-01/`.
It contains exact invocation JSON, submission logs, root CAS/source audits and
`native-pytest-v4/test_native_cuda_only_batch_tr0/attempt-822b24f9def84f9cb707b7918eb6d6a1/`.
