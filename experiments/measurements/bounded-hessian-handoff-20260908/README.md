# Bounded canonical Hessian handoff — CPU contract evidence

2026-09-08. PrismaQuant source `11cdcc823325a56b14a53692a3db2565bc5d16ce`
(base `00a7b5d35b82e7b78caeb5b461e16b2de7a5c5a9`) and Tessera source
`615e1c0302693447a9cd713bb1f9dddc21fbd378` (base
`98aa317a06e55731c36c53f509adc04260668cc1`, implementation `ad828f6782`,
separate ordering correction `615e1c0302`). Author worktrees were isolated;
no runtime release pin, format admission, row resource estimator or native
experiment was changed by this work. Cross-repository integration and producer
development-pin updates belong to the coordinator's later integration.

The opt-in selected campaign argument `--export-hessian-reference-policy`
accepts a closed `tessera.hessian_reference_load.v1` policy with positive byte
caps `max_metadata_bytes`, `max_file_bytes`, `max_hessian_bytes`. Metadata has a
128 MiB bootstrap ceiling and the declared metadata cap cannot exceed it;
H cannot exceed the declared file cap. The row writer emits metadata references
to the existing complete canonical capture, its exact census, full-census
counts and selected-unit H commitments. The old `tessera.hessian_capture.v1`
content seal remains exact, including unsorted reference JSON. Row merge binds
each row seal/provenance/selection and unions only metadata. Allocation carries
the canonical manifest/census binding separately; producer intake uses the
closed `tessera.priced_export_inputs.v2` expectation.

Metadata acceptance is not verification of unconsumed H. Every actual H lookup,
including cached-wire input identity, hashes the bounded original canonical
file and actual H through a held descriptor before returning a detached copy.
The reader keeps JSON metadata and descriptors, never an H/X cache. One lookup
can own one capped source mapping, one H copy and existing H-sized hash bytes;
caller-owned H and encoder workspaces are additional. The tests assert no
retained returned H and that source changes during mapping load are refused.
Legacy `.pt` input remains eager. Sampled selected-wire materialization refuses
references at request intake because its existing completion union is eager;
complete priced wires are the supported reference scope. No fallback copies
the complete H corpus.

## Verified runs

All execution used PrismaBuild, one CPU, native thread libraries bounded to
one, one attempt, priority -10 and x86 tooling placement (`dl380g10`). Producer
used `/home/rob/venvs/pb-cpu/bin/python` (Torch 2.11.0+cpu), 3 GiB reservation;
PQ used `/home/rob/venvs/pq-cpu312/bin/python`, 4 GiB reservation. These are
correctness runs, not timing comparisons or GPU qualification.

| Action | Result | Meaning |
|---|---|---|
| `8326b6e31f23e4e7ad563c47a902003a698cbcee83b4fea502a46c19ca6c1070` | 1 expected failure, 1.70 s | Baseline public reader sends reference JSON through eager `torch.load`; new bounded behavior absent. |
| `4915c95804a917c5674af1f4db747511b323648247f51deef4e79a710b05b8f7` | 1 expected failure, 21 deselected, 1.23 s | Review regression: unsorted valid reference metadata changed the old seal in initial implementation. |
| `740c213d5bd5fd890cf2496d421363d8a8fd23cd696b5548663504066bfa40b2` | 71 passed, 0 skipped, 0 missing modules, 17.15 s | Final producer reader, priced intake, legacy capture role and cached producer tests. No CUDA allocation. |
| `e77e2f2fe875f2fc16b030d4b642d684cb7b277ae036e32419e9165081b47a9f` | 200 passed, 2 skipped, 14 Torch deprecation warnings, 111.54 s | Final PQ compile, exact-producer handoff, canonical capture, priced export, materialization, dispatch/merge, allocator, bounded legacy writer and documentation checks. |

PQ's two skips are explicit: the old producer without a capture-seal API is
inapplicable to this producer, and the native writer measurement requires its
separate explicit opt-in. No new reference test was skipped. The successful
public selected-CLI test preserves its canonical manifest and source identity,
reads only required source shards, forbids extra calibration forwards, and
publishes references without an H `.pt` sidecar. It uses the existing small
source-orchestration fixture and an empty research menu; it is not a native
model pricing or serving run. Other tests cover row digest equivalence,
metadata-only merge, exact selection and binding refusals, deferred payload
reporting, policy applicability, mixed allocation binding and sampled intake
refusal. Producer cases cover source/metadata replacement, byte corruption,
limits, forged commitments, detached lifetime, `for_unit`, cached-wire intake,
v1/v2 separation and legacy eager loading.

Final PQ execution ran `experiments/hessian_reference_contract_check.py`. It
hashes and extracts the producer archive inside the admitted process, checks
the imported module path, compiles all six changed runtime/CLI modules, then
runs the eleven named test files. Archive:
`/mnt/shared/prismaquant-experiments/hessian-reference-handoff-20260908/tessera-615e1c0302693447a9cd713bb1f9dddc21fbd378.tar`,
SHA-256 `c6a6e849f7826ab18053004d4d625b2f26d9131923fda32b20fc6ca6a458aba7`.

The checked JSON receipts contain action resource requests, terminal outcomes,
log locations and byte hashes, exact snapshot inputs, CAS receipt/result
identities, cleanup and measured process memory. Actual logs and source-bundle
bytes/hashes were checked; canonical receipt hashes were recomputed and actual
CAS result bytes/hashes verified. Final producer/PQ source snapshots differ
from the named final commits only in PB closure metadata. Producer peak scope
memory was 378,736,640 bytes; PQ was 918,777,856 bytes. Both finished with no OOM,
no live process and complete successful scope release. These peaks describe
small CPU fixtures, not full-census native resource demand.

The GLM canonical capture was not consumed or rewritten by these tests. No
native H handoff, CUDA encode, full-model export, serving correctness,
throughput, physical page eviction or GPU residency claim follows from them.
The changed producer package receives a new encoder source hash; original
capture reuse remains compatible through its original model/census identity,
but old encoded-wire identities cannot be reused under the new producer hash.

`invocations.json` records the exact final command arguments and resource
declarations. `preparation-attempts.json` gives the separate failed fixture and
tooling attempts and their disposition; they are not counted as coverage.
The initial local-placement submission was withdrawn from the ready queue
before any execution and replaced by the explicit x86 tooling class.

Checked-in stdout excerpts omit the trailing worker receipt JSON and trim
line-end spaces. The receipt JSONs retain hashes and locations of the exact
original logs whose bytes were independently verified.
