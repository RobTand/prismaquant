# Consumed source-page coalescing — 2026-09-08

Issue #380. This fixes the opt-in CUDA source-reader page-advice range, without
changing checkpoint bytes, selected tensors, copy/map lifetime fences, source
identity, admission thresholds or the canonical calibration contract.

The common GLM capture stopped at its physical memory guard before layer 11.
Its retired cgroup retained 18,513,117,184 file-cache bytes, including
18,111,004,672 `file_thp` bytes, with anonymous memory zero. The old helper
aligned every tensor individually, leaving a partial page at every adjacent
consumed tensor boundary. Large file-cache folios containing those pages
remained resident. The corrected helper validates all selected spans, unions
adjacent/overlapping consumed bytes, then aligns each union inward. Headers,
unread gaps and outer partial pages remain protected. There is no global cache
drop, separate source cache or new allowance to pass the physical guard.

A failing pre-fix CPU regression (`f8baf61182b3`) reproduced two advice ranges
where one contiguous consumed range was required. The corrected focused suite
passed 24 tests (`1de1d672fc9f`); source-page, architecture and staleness checks
passed 43 tests with no skips (`0c3924531a25`). These tests include unread gaps,
invalid later spans refusing before any advice, source mutation, nonregular
files, CPU mapping ownership and mocked CUDA copy/map lifecycle.

The native measurement is PB action
`2f7d0b024d3c39b632c005b0a5e2411b7702d4f7b07731eaf362ba884769cc2d`:
`--measurement --host-class gb10`, placed on Sparky, 2 CPUs, 12 GiB physical
memory and a 6 GiB GPU subset. Native library threads and the existing source
read pool each used one thread. The content-qualified producer image seal is
`eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd`,
Torch 2.13.0+cu130, CUDA 13.0, NVIDIA driver 595.84.

`experiments/source_page_release_profile.py` reads the same 64 contiguous BF16
GLM layer-20 expert tensors from the actual checkpoint, totaling 1 GiB, in one
admitted ABBA experiment. The old-control arm invokes the shared helper once
per key, reproducing its original page ranges. The corrected arm calls it once
per completed reader chunk. Both are profiled with CPU/CUDA `torch.profiler`,
memory events and shape recording. A 20 ms sampler records cgroup memory,
`file_thp` and CUDA reservations. `mincore` reads residency without faulting in
the selected payload. A completed warm read supplies exact GPU reference
values; each arm must compare equal for every tensor. Between arms only the
already consumed byte union receives advice, so every initial payload residency
is the same 2,093,056 bytes. No source or unrelated cache bytes are modified.

| Arm | Payload pages retained | Cgroup `file_thp` after read | Profiled wall time |
| --- | ---: | ---: | ---: |
| Old per-key 1 | 134,213,632 B | 136,314,880 B | 0.5704 s |
| Coalesced 1 | 2,093,056 B | 4,194,304 B | 0.3270 s |
| Coalesced 2 | 2,093,056 B | 4,194,304 B | 0.3988 s |
| Old per-key 2 | 134,213,632 B | 136,314,880 B | 0.4257 s |

The measured memory reduction is 126 MiB of retained payload/cache-folio
storage per 64 adjacent expert tensors in this workload. All four GPU results
are byte-exact. The protected outer edge still retains roughly 2 MiB; advice
is best effort and cannot authorize a general zero-residency assumption.
The first arm also allocates its second GPU output plane; later arms reuse that
allocator reservation. The control reparses the small header for each key.
These short timings therefore do not establish a production speedup or work
per joule. Each trace records 64 HtoD copies. This experiment qualifies memory
release, and does not establish GPU-bound source loading or a full-capture fit.

Both-host Netdata CPU, RAM, available-memory and GPU-power charts have ten
samples spanning the experiment plus three seconds of context on either side.
Their resolution is too coarse for per-arm energy comparison. GPU utilization
is not used as a saturation claim. PB ended with exit 0, no OOM, complete
scope cleanup and independently empty local Docker process listing.

Attributable artifacts are under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/coalesced-source-pages-01/`:

- `native-invocation.json` records the complete command and sealed container.
- `native-abba-01/measurement.json` records every arm, source stat, exact keys,
  span, samples, advice boundaries and exact GPU equality.
- `native-abba-01/{0-per_key,1-coalesced,2-coalesced,3-per_key}.trace.json`
  and `profile-summary.json` provide before/after in-process profiles.
- `native-abba-01/netdata-*.json` preserves both boxes' eight chart series.
- `audit.json` checks actual terminal status, cleanup, source snapshot and CAS
  payload bytes; `artifact-seals.json` binds measurement files by SHA-256.
- Prefix-named logs preserve the failed regression and passing checks.

The previous full capture remains a failed, partial artifact. Only a separately
completed canonical capture can establish full-model progress after this fix.
