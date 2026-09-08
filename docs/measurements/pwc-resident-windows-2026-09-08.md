# Production weight-cache resident windows — 2026-09-08

Issue #381; implementation `d2a9218f08`. This is an opt-in cache API and CPU
lifecycle qualification. It changes no pipeline default, source traversal,
quantization arithmetic, cache artifact, format menu or serving gate. No GPU
throughput, whole-process memory fit, KL or bpp claim is made.

## API and ownership

`ProductionWeightCache.plan_resident_windows(keys, *, max_resident_bytes,
max_workers)` resolves aliases, removes duplicates in first-occurrence order,
and returns finite tuples of concrete cache keys. Each quantum has at most
`max_workers` entries. This bounds the existing `prefetch` pool's submitted
futures and loaded results without adding another cache or dispatcher.

`resident_window(keys, *, max_resident_bytes, max_workers,
max_load_buffer_bytes=None, release_file_pages=False)` admits exactly one
nonempty quantum, prefetches it, checks residency and yields a key/scalar-only
receipt. All existing PWC tensor backing storages count against the resident
cap, including unrelated entries and storage hidden behind small views.
Storage aliases count once. Incoming files use the existing conservative file
estimate and uncompressed Torch archive storage-record preflight. Compressed,
opaque and symlink inputs refuse; sparse/meta tensor storages refuse. Loaded
storage is checked against the archive bound and complete cache storage is
checked before exposing a window. The LRU must have room for the incoming
storage without evicting another resident entry. Nested windows refuse.

Serialized buffers have a separate aggregate cap, defaulting to the resident
cap. The context reports both caps and `load_buffer_capacity_bytes`. An active
window uses the existing exact-byte file-load receipt path, checking the file
stat identity before/during the read and recording SHA-256 of the actual bytes.
Optional page advice runs through `release_activation_cache_file_pages` after
the checked load; the helper rechecks file identity. Advice is not evidence of
physical reclaim.

`get_resident(name, fmt)` resolves the existing aliases, preserves CB producer
identity and loaded-tensor validation, and checks an enabled file-load receipt
against its exact tensor/file lifetime. A missing or evicted key raises instead
of invoking a disk load. A failed receipt check cannot be bypassed by repeating
the lookup during an active window.

On context completion or failure, `release_resident_tensors(keys)` drops the
selected disk-backed owners and their receipts through the existing cache
release mechanism. It preserves unrelated entries and in-memory values with
no recorded load path. A same-named file alone is not adopted as proof that an
in-memory value is safely releasable. The existing no-argument whole-cache
release/compaction behavior remains unchanged.

Callers must drop borrowed tensor references at the boundary and separately
admit device copies, candidate deltas, source/statistics buffers, allocator
overhead and consumer workspaces. These cache caps do not certify a complete
candidate projection phase or physical memory fit.

## Validation and evidence

All execution used PrismaBuild, CPU-only on DL380G10, with the existing
`/home/rob/venvs/pq-cpu312/bin/python` interpreter. Native math threads were
bounded to one. No tests or GPU probes ran outside admission.

| Action | Result | Reservation |
|---|---|---|
| `036495496825` | Seven expected missing-API failures before implementation | 2 CPUs, 3 GiB |
| `6f3f7c225ca8` | 16 window/file-receipt tests passed | 2 CPUs, 3 GiB |
| `b23326e12a2f` | 85 expanded window and existing-cache tests passed | 4 CPUs, 6 GiB |
| `70db9cbf929a` | Final 108 tests passed, zero skips | 4 CPUs, 6 GiB |
| `f6a1a7322df1` | Both touched Python files compiled successfully | 1 CPU, 1 GiB |

The final suite used two xdist workers, each permitted at most two loader
threads. It covers full backing storages and aliases, unrelated resident
owners, LRU refusal before load, strict missing/evicted lookup, existing CB
validators, optional receipts, repeated lookup after receipt invalidation,
file drift, bounded serialized buffers, malformed input refusal, checked page
advice, selected release, normal/exception cleanup and legacy cache behavior.
It also ran the architecture and documentation-staleness checks.

The final test command, following `pbrun.py --tag x86 --cpus 4 --demand mem_gb=6
--priority -10` and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, was:

```text
/home/rob/venvs/pq-cpu312/bin/python -m pytest -q -n 2
  tests/test_pwc_resident_windows.py tests/test_pwc_file_load_receipts.py
  tests/test_production_weight_cache.py tests/test_docs_staleness.py
  tests/test_architecture_doc.py
```

Complete commands, raw logs, the read-only audit script and `pb-audit.json` are
under `/mnt/shared/tessera-measurements/glm-canonical-census-20260908/pwc-resident-windows-01/`.
The audit checked all five terminal exit statuses and scope cleanup, four
canonical CAS receipt hashes and their actual payload hashes, and all source
snapshot bundle hashes. The final test and compilation source trees match the
implementation commit across every tracked path; their only extra files are
generated PrismaBuild closure records. No successful receipt is inferred from
the submission acknowledgement.
