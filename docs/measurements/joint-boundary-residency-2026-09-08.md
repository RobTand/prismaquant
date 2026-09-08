# Exact streamed AURA boundary residency — 2026-09-08

Issue #373 adds an explicit, default-off exact artifact policy for source
boundaries and rolling cotangents. The existing activation artifact owner
writes the original tensor bytes and verifies bounded, fully resident input
windows before layer execution. Generation metadata owns references, not a
second tensor cache. Source traversal, signed scalar projection, production
weight-cache behavior and `dW` arithmetic retain their existing implementation.

The initial CPU regression retained 3,840 boundary bytes against a 1,024-byte
fixture cap and failed (`c2f15a278645`, exit 1). This was the actual legacy
full-draw behavior before implementation. The bounded owner passes that case.
Its explicit cap covers leased exact CPU tensors plus one compact CPU write
copy. A separate auxiliary cap charges full underlying tensor storage in
input IDs, positional state, attention masks and shared pass state, plus the
larger of actual shared cotangents and a conservative per-probe reservation.
Opaque tensor owners refuse. These caps do not replace total process or GPU
admission, source prefetch, candidate or gradient budgets.

Twenty focused owner tests cover full tensor/dtype preservation, GLM-style
rank-four streams, nonempty Gemma4 shared state, exact propagated cotangents,
source call order, seeds 7000–7003, uneven prefetch windows, and missing,
corrupt, stale or oversized entries. A noncontiguous-tensor regression uses
the CPU profiler's self allocation accounting to require one compact copy.
Root review added explicit mapping-key inspection: tensor keys are charged and
opaque keys refuse; all 20 focused cases passed (`62bf4b27e4ec`, 11.24 s).
Failed and interrupted runs clean their owned tensors and artifacts; resume
recaptures a fresh generation and can reuse completed cost shards. A complete
cost checkpoint returns without creating an artifact generation. Hot window
lookups have no disk-loading branch.

The final CPU gate (`0ad52f784868`, DL380, six physical CPU workers, 18 GiB
aggregate reservation, one native thread each) passed 145 tests in 19.84 s and
compiled all seven touched Python modules. One existing Gemma4 sweep-order
case skipped: the installed Transformers build marks layer 6 KV-shared but
does not expose `kv_shared_layer_index`. Synthetic shared-state tests using
the actual Gemma4 profile passed. Earlier gates passed 40, 88 and 107 tests;
the latter two had the same skip. An initial unsuitable Python 3.14 venv
lacked `compressed_tensors` (`c924e121d726`), and the first implementation
snapshot had a syntax error (`61f439f40f1d`); both failures are retained.

The native qualification harness is
`experiments/joint_boundary_profile.py`. It interleaves legacy/exact/exact/
legacy over a genuine random tiny GLM original-layout checkpoint: BF16,
two KDA/DSA layers, hidden size 64, four mHC streams, four routed experts,
five complete 17-token sequences at B1, four probes, and identical synthetic
candidate tensors. These candidates are fixtures, not serving artifacts.
The KDA/short-convolution functions use the upstream Torch reference backend
on CUDA; this qualifies boundary lifetime and arithmetic parity, not optional
fused kernels. Each arm records a CPU/CUDA Torch trace, cProfile, process I/O,
CUDA allocator state, source call order and every cotangent's byte digest.
The existing observer records Netdata from both Sparks and Python stacks.

Native PB measurement `f8f2ab675f09` completed on Sparklina with exit 0 and
verified scope/container cleanup. All four arms passed exact cost/statistic,
source-order and cotangent parity: 18 profile-unpinned dense/packed targets,
50 source-layer calls and 60 cotangents per arm. The fixture passes its exact
per-target format plan through the production plan interface. Two preliminary
attempts stopped before source forwards because a uniform target list included
vision units (`6e02875b73cf`) and then pinned attention units with no fixture
render (`45204e7e83e1`). Their failure logs are retained; neither required a
production-code change.

| Arm | Profiled call seconds | Retained captured boundary bytes | Exact window + writer peak bytes |
|---|---:|---:|---:|
| Legacy 0 | 5.477 | 130,560 | — |
| Exact 1 | 4.930 | 0 | 43,520 |
| Exact 2 | 5.657 | 0 | 43,520 |
| Legacy 3 | 4.269 | 130,560 | — |

The columns describe different ownership phases, not a total process-memory
ratio. Both exact arms retained only references between windows, observed
4,335 auxiliary bytes and zero shared-adjoint reservation for this GLM fixture,
wrote/retired all 75 exact entries, and closed all 27 prefetch windows with
zero hot misses and zero remaining artifact bytes. Each wrote 652,800 tensor
bytes and read 739,840; peak working artifact bytes were 336,567. CPU user
annotations locate the write barriers at 442/1,062 ms and input prefetch at
125/124 ms in the two exact arms. These I/O phases are outside resident lookup.

CUDA allocated/reserved peaks were identical in all arms: 76,011,008 and
94,371,840 bytes. Torch profiling and export retain substantially more CPU
memory than this tiny source; these data do not establish a reduction in
whole-process physical memory. Twenty Netdata samples from each host cover
the run: Sparklina GPU power 4–13 W and CPU busy 6.1–7.6%, Sparky power 42–47 W
and CPU busy 6.5–17.0% with the independent full capture still active. At this
fixture size the CUDA kernels total roughly 0.1 s and instrumentation/CPU work
dominate. Timing differences are not a throughput claim, and five-second host
sampling is too coarse to rank such short calls by work per joule. No
full-model fit, saturation or serving-kernel claim follows.

The full GLM census still requires a separately qualified source traversal
that amortizes source reads, bounded candidate/projection lifetimes and an
aggregate admission model. This change intentionally preserves batch-major
capture, whose repeated source-layer reads are a remaining limitation; no
source-read duration or overlap claim has been measured here. Exact artifact
writes and prefetch are explicit I/O phases outside the checked resident
windows. Full 512-sequence execution remains subject to its existing gates.

Evidence root:
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/`.
`joint-boundary-implementation-01/` contains commands, logs and canonical
CAS receipt/result audits; `native-audit.json` independently compares recorded
cotangent hashes and source order, checks trace markers, and hashes all arm
artifacts. `joint-boundary-native-03/` contains the successful observations,
four CPU/CUDA traces (1.58 GB total), cProfile files and both-host Netdata.
`joint-boundary-native-invocation-04.json` is the successful launch; earlier
numbered invocations preserve failed/deferred attempts. These files record
the native producer image content seal, runtime environment and PB demands.
An early measurement submission (`d6cc0c85238d`) was withdrawn from the ready
queue with zero tokens to preserve the existing direct-vLLM experiment's
isolation during its CPU-only container startup gaps.
