# Selected-source authentication CPU qualification — 2026-09-08

Issue #388: selected anchor preparation previously rehashed every source shard
through `capture_identity`, including payload that the selected row never read.
The complete-capture identity is unchanged. A selected invocation now binds the
complete manifest and census before model construction, authenticates metadata,
and gives the existing source readers one shared owner of source descriptors.
The first tensor read hashes that held shard; subsequent readers reuse that
invocation's authenticated descriptor. There is no persistent digest cache.

The complete tiny-capture campaign regression retains actual nonbody and layer
readers, replacing only model construction with the existing orchestration
fixture. Before the fix it hashed **2,130,208 payload bytes**; the corrected path
hashes exactly the **32,960 bytes** in the consumed head and selected shards.
Both paths assert selected tensor equality and preservation of their original
complete canonical identity and manifest bytes. This is a byte-count regression,
not a throughput measurement or an actual GLM selected-row A/B.

The final focused PrismaBuild action
`1c95e20c822cfd6ee904f3da3450146de8fbeb7c8005782f9e8a5c479491638f`
passed **89 tests, with one CUDA-only encoder-memo skip**. It covers original
identity preservation, unused payload deferral, metadata and selected payload
tampering with restored mtime, census/manifest/source-roster refusal, shared
threaded reads, reader draining on error, projected source tensors, paired
scales, descriptor replacement, FIFO refusal, HF-style symlinks, context failure
cleanup, metadata leases, and escaped slice refusal. The FIFO regression was
first reproduced in a two-second bounded child in action `ff1a912a9be1…`, then
fixed with a nonblocking open followed by regular-file validation. The final
source and test files were compared byte for byte with the passing CAS snapshot.

Broader validation covered 15 files: **232 passed, one CUDA-only skip** before
the four final FIFO/context/symlink cases were added and rerun in the focused
suite above. Two initial mixed shards reached their declared 4 GiB ceiling;
their five files were resubmitted through PB fanout at 8 GiB per shard and all
passed. Completed green shards were retained. The wider run includes the actual
tiny GLM source/campaign tests and existing FP8/MXFP4, streaming, projection,
resume and architecture tests. A separate PB compile action `8d10a8ec6cb5…`
passed for all six changed runtime modules and both changed test modules.

All runs were CPU-only on the eligible x86 fleet using Python 3.12 and the
scoped `pq-cpu312` environment (Torch 2.10.0+cpu, Transformers 5.16.1). Native
threads were bounded to one; test actions used priority -10 and explicit CPU
and memory admission. Deprecation warnings include Torch JIT and the bounded
FIFO test's POSIX fork. The earlier prototype's obsolete orchestration test,
the two expected red regressions, and both memory-limit failures are retained.

The audit index is
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/selected-source-authentication-01/pb-audit.json`.
It checks all 17 terminal records, cleanup, source CAS hashes and successful
result CAS/receipt hashes: 12 successful actions and five retained failed
attempts. `source-audit.json`, `cpu-suite.json`, `cpu-suite-memory-retry.json`
and the action logs are in the same directory.

Ordinary source files must remain stable for their descriptor leases. Stat
fences reject mutation or pathname replacement; they do not authenticate bytes
or make a concurrently mutable filesystem immutable. The public selected path
still requires the complete canonical manifest. No original GLM capture, full
checkpoint hash, GPU run, menu, serving gate, or calibration draw changed.
Native selected-row I/O, memory and timing qualification waits for the original
complete manifest and a matched before/after run with in-process profiling and
both hosts' telemetry. No full-model speed or fit result is claimed here.

## Root integration qualification

Integration with the verified capture loader at `3a94d7520b67` passed **355
CPU tests across 23 files, with one CUDA-only encoder-memo skip and two passing
subtests**. The root audit verified all 23 successful terminal records, result
receipts and source CAS snapshots; each source tree differs from that integration
commit only by its PB closure manifest. The invocation and consolidated coverage
are recorded beside the earlier evidence in
`root-integrated-cpu-invocation-01.json`,
`root-integrated-cpu-consolidated.json`, and
`root-integrated-positive-cas-audit.json`.

One initial shard reserved one CPU while its qualification-window fixture
explicitly requested two cache workers. The affinity guard correctly rejected
that fixture. Only that file was retried with two CPUs and passed nine tests
in action `568bfcd59e2238c4cd06e8220af80a5edec31efc1716cd3de4bb6db85926d771`.
The original failure and reservation diagnosis are retained in
`root-integrated-cpu-affinity-negative.json`; no runtime change was needed.
Native GLM measurements required by #388 remain outstanding.

The first full CI run (`34251207739`) found nine failures in the older
prefetch-scheduling fixture: it bypasses `StreamingContext.__init__` and omitted
the new optional authentication field. PB action `4e244dd5b4b0…` independently
reproduced all nine failures. Initializing that fixture field to `None` restores
the ordinary unauthenticated scheduling tests without changing runtime code.
PB action `2057d00854c6e2c24b962054f7bd5595ff4d6b0f5a9ac2f54cc97c7280d5e361`
then passed all **32 tests** in the scheduling and streamed-admission files,
with five admitted CPUs for the main thread and four fixture readers, one native
thread per reader, and 4 GiB. The original CI result (6,870 passed, nine failed,
201 skipped, three xfailed, 192 passing subtests) is retained as a failed run.
The root evidence is `root-prefetch-fixture-negative.json` and
`root-prefetch-fixture-positive-cas-audit.json` beside the earlier receipts.
