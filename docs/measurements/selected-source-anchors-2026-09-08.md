# Selected-source Tessera anchors — 2026-09-08

A complete canonical capture could previously be reused only after loading the
whole model: `--streaming --units` exited before source preparation. Issue #370
adds selected source preparation through the existing streaming layer cache.
It verifies the complete census geometry, draw, source hashes, runtime and
capture manifest, then snapshots only selected dense or declared logical-expert
weights. Required layers arrive through resident prefetch; cloned expert views
retain no packed parent storage. Source preparation performs zero forwards and
releases the source model before selected uncapped H and retained X are loaded.
The recorded initialization witness belongs to the completed canonical capture;
it does not claim a new full-source initialization audit.

The existing encoder memo retains at most the compatible anchor batch width
for this path. Its deterministic eviction changes ownership, not producer
inputs. Resident-source execution retains its prior policy. Admission derives
source, export-input and encoder phase peaks from source headers and the shared
loader dtype policy, including strict FP32 parameters. The existing physical
cgroup guard checks selected CUDA execution; source page advice and immediate
host allocator purge are required. Completed sidecar and PWC files use the
existing verified page-release helper. The whole-file writer remains unchanged.

## Qualification and evidence

All tests ran through PrismaBuild. The evidence root is
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/selected-source-anchors-01/`.
`regression-before.json` records the original CLI regression failing at the
streaming selection guard (action `d1f5a9da1405`). A separate strict-dtype test
failed before correction (`38ae4c6fbd4d`): the meta skeleton incorrectly charged
8 bytes for a strict FP32 plus ordinary BF16 pair requiring 6 bytes. The corrected
shared loader snapshot preserves the original FP32 value exactly and passes the
6-byte budget (`05c4e8d9ac25`, 7 passed). Its temporary regression worktree was
retired after preserving `dtype-before-regression.patch` and the PB snapshot.

The first native CUDA qualification on Sparklina used the immutable producer
image `sha256:9f9b9f05b17531399ba66dc6415b054cf5d68c82270626d0e9150e75c808435f`,
Torch 2.13/CUDA 13.0 and Transformers 5.16.1, reserving four CPUs, 24 GiB physical
memory and a 16 GiB GPU subset. Action `40dba33a6fe0` passed 52 tests with no
skips in 44.75 seconds. It covers a real tiny GLM original-layout checkpoint,
full canonical capture, selected source reuse, a native Tessera anchor and
resume identity, plus native wire/price equality for three units at two rungs.
This preceded the final dtype-budget and export-page accounting additions.

The memo comparison retained 786,432 bytes of owned factor tensors with three
unbounded entries versus 262,144 bytes with capacity one, with identical native
wire bytes and quality/cost rows. `native-bounded-01/memo-parity.json` records the
comparison. Both `native-control-01/` and `native-bounded-01/` contain Torch
CPU/CUDA memory traces, hash-bearing summaries, source receipts, and Netdata CPU,
RAM, available-memory and power series from both Sparks. The parent canonical
capture was external load on Sparky. These tiny cases establish ownership and
numerical parity; their timings establish no speed or work-per-joule claim.

The control attempt had one synthetic fixture failure caused by mocking its
resource plan while allowing CUDA detection; real GLM tests passed. The fixture
was explicitly made CPU-only before the 52-test native pass. An earlier CPU
collection environment lacked compressed-tensors; the qualified pq-cpu312
installation was used thereafter. The retained failure logs distinguish these
setup errors from regressions.

After the export-phase changes, five independent CPU PB shards passed 66 tests
with two native-only skips (`cpu-export-phase.json`). The complete capture suite,
including exact legacy/selected Hessian sidecar byte equality, passed 32 tests
with no skips (`ced399e32b0e`). Earlier resume/fanout checks passed 61 tests;
profile/header and packed-expert checks are recorded in `cpu-audit.json`.
`final-audit.json` independently verifies terminal records, checkout snapshots,
exit status and CAS result bytes for the final focused checks and the native
qualification. Pytest used one native thread per process on DL380 with CPU Torch
2.10 and Transformers 5.16.1; CUDA-only comparisons were covered on Sparklina.

## Full GLM admission remains constrained

The canonical census has 36,423 units. The existing planner with
`--groups-per-row 1` yields 132 independent rows: 45 fused dense groups,
45 dense singletons and 42 routed stacks. Each routed stack has 864 logical
units and remains indivisible under the exact full-group contract. Statistical
expert sampling is a different estimator and cannot stand in for exact coverage.

Using the canonical census SHA-256
`b63f7bf6c4320714b4ceb38fbd6996e032e0f0c9b82ac2a30a8337d846e358fd`,
source headers and current dtype policy, action `8449e462362c` derived these
phase maxima (GiB):

| Representative group | Source preparation | Export inputs | Resident anchors | Admission |
| --- | ---: | ---: | ---: | ---: |
| Layer 0 fused gate/up | 28.140 | 25.134 | 26.258 | 28.140 |
| Layer 0 down | 28.047 | 26.923 | 30.141 | 30.141 |
| Layer 10 routed stack | 68.221 | 124.783 | 98.019 | 124.783 |

`resource-inspection-invocation.json` and `full-glm-derived-resources.json`
record the exact derivation. These are conservative bounds, not a full GLM
execution or fit measurement. The earlier 98.019 GiB stack estimate omitted the
Hessian writer's whole-file page owner and is superseded. The current writer
can retain a full 40.5 GiB sidecar while H/X/weights remain resident, so this row
must refuse a 104 GiB worker. Within-file bounded serialization and/or a supported
exact smaller quantum require separate work. No partial capture becomes
canonical, no joint scope is narrowed, and no serving format, menu or pin changes.
