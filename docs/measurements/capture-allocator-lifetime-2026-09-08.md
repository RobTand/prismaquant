# Bounded capture CPU allocator lifetime — 2026-09-08

The full canonical GLM capture completed and durably wrote layers 0–3, then
refused before layer 4 at its physical memory guard. The retained prefix is
876 entries, 52,007,061,660 file bytes. Every journal envelope and entry SHA256
was independently verified before retry; no complete capture manifest exists
for that failed attempt. PB action `cbd010f53bd2` exited one with complete scope
cleanup and no OOM. It is not a successful capture.

At the refusal, the cgroup charge was 59,080,269,824 bytes, CUDA reservation
56,637,784,064 bytes, and host available memory 7,659,896,832 bytes. The guard
correctly refused its conservative 115,718,053,888-byte sum and the host floor.
The completed layer’s CPU tensor objects had been deleted; object lifetime
alone did not establish physical release.

## Reproduction and cause

A bounded diagnostic in the exact producer image creates 32 representative
shared-input groups: 4096-column gate/up Hessians, 2048-column down Hessians,
and 512 retained rows per unit. It materializes independent CPU siblings,
uses the existing durable activation writer and completed-file page advice,
then releases all output owners. Storage weak references, cgroup memory.stat,
process smaps, CUDA/pinned allocator counters, cProfile and both Sparks’ raw
Netdata series record the boundary. No model forward or calibration change is
involved in this diagnostic.

All 192 output storages expire. The original configuration nevertheless
retains about 5.4 GiB of anonymous host pages. Python GC, libc malloc_trim,
Torch pinned-host emptyCache and accelerator empty_host_cache leave that charge
in place; the outputs are not pinned and the pinned allocator counters are zero.
These negative attempts rule out those release mechanisms for this image.

Independent disassembly of the image’s libc10.so shows `c10::alloc_cpu` calling
its statically linked `mi_malloc_aligned`, and `c10::free_cpu` calling `mi_free`.
The allocator is mimalloc, even though no separate mimalloc library or
LD_PRELOAD entry appears in process maps. This agrees with Torch’s
[allocator source at the image’s build commit](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/c10/core/impl/alloc_cpu.cpp).

The final before/after pair differs only in the explicit
`MIMALLOC_PURGE_DELAY=0` child environment and evidence directory. Full source
snapshots compare identically apart from PB closure metadata. The setting makes
mimalloc purge freed pages immediately, as described in its
[environment options](https://github.com/microsoft/mimalloc#environment-options).

| Physical checkpoint | Original configuration | Immediate purge |
| --- | ---: | ---: |
| Initial cgroup bytes | 541,822,976 | 541,433,856 |
| After durable writes | 6,441,553,920 | 6,441,926,656 |
| After output deletion | 6,441,537,536 | 632,369,152 |
| Live output storages after deletion | 0 | 0 |

This establishes physical release after a completed layer. It does not reduce
the live output footprint, discount admission, establish full-model fit, or
claim faster encoding/serving. Each short diagnostic has cProfile, process
memory snapshots and two samples from each Spark’s Netdata at the configured
five-second interval. No GPU saturation or work-per-joule improvement is claimed.

## Contract and validation

The existing bounded-capture dispatcher seals immediate purge and source-page
release into the PB environment and its inner container specification before
Python starts. Explicit incompatible settings refuse. Direct bounded CUDA
capture requires the same environment. The physical guard additionally samples
after deleting completed outputs, before installing the next layer. Existing
budget, source prefetch, tensor bytes and numerical identities remain unchanged.
The legacy and prior shared-input policies retain their environments.

Seven new launch-contract regressions failed before implementation, while the
legacy case passed (`3e5f626dd5d5`). After implementation, 47 CPU cases passed
(`6e477a80af44`), followed by 50 native cases with no skips (`b4b544f94479`),
including tiny GLM streamed-capture tensor parity and actual CUDA profiler
coverage. The CPU run used four workers, Torch 2.11 CPU and pytest 9.1.1 on
DL380. Native validation used the exact Torch 2.13/CUDA 13 producer image on
Sparky, two xdist workers, four reserved CPU cores, native threads one, and a
24 GiB physical budget. The first native launch failed before collection because
xdist was absent (`61bb25eb028a`); scoped pytest-xdist 3.8.0 and execnet 2.1.2
were installed and the corrected run supplied them explicitly. Read-only pytest
cache warnings do not hide missing tests.

A separate observer fix publishes final status to progress.json as well as
result.json. The regression demonstrated that progress retained `running`
after failure (`ec0f61ac138b`); the corrected case passed (`f8af63d37005`).
Terminal PB state and actual result files remain the completion authority.

The full retry, action `63992d4c42ad`, uses source `0ec032210c`, the same census,
source checkpoint, producer, image, 512×512 B1 token draw, output journal,
104 GiB physical reservation, 92 GiB GPU cap and 24-hour deadline. It changes
the allocator environment and adds the checked release boundary. It must replay
the completed prefix to recover hidden states and requires exact equality to
all preserved entries. Its outcome is recorded separately; submission alone
establishes no full-capture success.

## Evidence

Root: `/mnt/shared/tessera-measurements/glm-canonical-census-20260908/`.

- `full-capture-failure-root-audit-01.json`: failed terminal, complete source,
  all 876 journal/file checksums, original profiler artifacts and memory refusal.
- `capture-allocator-lifetime-root-audit-01.json`: all four diagnostic CAS
  receipts/payloads, full source snapshots, isolated pair and artifact hashes.
- `capture-lifetime-01/` through `capture-lifetime-04/`: original diagnostic
  outputs; the final pair is 03/04 (`e041d1c5669c`, `2c4bea45a57f`).
- `capture-libc10-allocator-disassembly.txt`: actual allocator call targets;
  the binary hash accompanies the audit.
- `capture-allocator-tests-root-audit-01.json`: complete CPU/native source,
  canonical success CAS receipts and actual test output.
- `full-capture-retry-invocation-02.json` and `full-capture-profile-02/`:
  retry declaration and live profiler/host evidence.
