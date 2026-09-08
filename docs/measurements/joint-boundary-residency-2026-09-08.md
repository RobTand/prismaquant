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

Nineteen focused owner tests cover full tensor/dtype preservation, GLM-style
rank-four streams, nonempty Gemma4 shared state, exact propagated cotangents,
source call order, seeds 7000–7003, uneven prefetch windows, and missing,
corrupt, stale or oversized entries. A noncontiguous-tensor regression uses
the CPU profiler's self allocation accounting to require one compact copy.
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

Native results are recorded in the evidence directory after qualification.
No full-model fit or throughput claim follows from this tiny fixture.

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
CAS receipt/result audits. `joint-boundary-native-invocation-*.json` records
the native producer image content seal, runtime environment and PB demands.
An early measurement submission (`d6cc0c85238d`) was withdrawn from the ready
queue with zero tokens to preserve the existing direct-vLLM experiment's
isolation during its CPU-only container startup gaps.
