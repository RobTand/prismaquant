# Deterministic head-walk tests and digest pruning (Refs #1929)

This test-only slice replaces two scheduling sleeps with real Event/Barrier
completion gates. The first roster unit is now proven to finish last rather
than assumed slower; durable prefix order, serial/parallel equivalence and
error-prefix cases still exercise the real driver. Five-second waits are
failure bounds, not success-path delays. Two cadence cases advance the existing
injected clock after real journal hashing and at overlay progress boundaries;
production timing, concurrency, publication and timeout policy are unchanged.

Digest tests retain all six golden inputs. Removed: the implementation/source
spelling-only test; the standalone empty-blob assertion already exercised by
bytes_sha256hex:selected_cached_units_manifest; and two intentionally skipped
indent2 rows already checked by the two indent2 cases. Head-walk and progress
node IDs are unchanged. Digest collection falls from 11 to 7 nodes, giving
30 collected / 28 passed / 2 intentionally skipped before and 26 collected /
26 passed / zero skipped after. Digest row indices shift from row2..row5 to
row0..row3; the input bytes and expected digests do not change. No CUDA case is
counted as coverage; these are CPU orchestration and byte-output checks only.

## Frozen source and observed verification

Base: 736fd56aa6853c3ee06655eda6015b18bef763bd.
Test commit: 2b65105925006b07507129d21caee7a43330f53a.
Both runs used published pbtest.py on explicit dl380g10, one PB-owned shard,
one pytest worker, CPU2 / mem4 GiB, native threads1, GPU visibility disabled,
600-second action deadline. Python3.14.4, Torch2.11.0+cpu,
Transformers5.16.1, PB SDK95a59051 and Tessera b40c93cb were verified by the
existing install/pin preflight in both logs. Test inventory:

- tests/test_joint_head_walk_754.py
- tests/test_head_resume_progress_822.py
- tests/test_digest_export_lane_1599.py

Before action: d2cefb78b0a4f6887c3d572df6bb3ff43a08cf7ff784fb94e03ae9d75a9a8176.
Receipt: b59ad51caeac51b44c2880af26551388497379396e5f5fd3822900d203899097.
Snapshot parent is the exact base; snapshot commit7ce34b5931866345764585fb99ec846c2e63aa8e,
bundle0e8f7bbd096cd5a0b4991630c22e4f41cac6fb6a0b9918bf3483d27e2b2b7a21.
After action: c0cc2cb29531ad24406832199ac088904c961d2d31958a47afff816327cd8bc5.
Receipt: 54ab3a50ce5cfd07be382fa7613e94acf352400b338cd61541e7547c0c4da2e0.
Both reconciliations record zero never-ran, missing-file, missing-collection,
double-collected or outcome problems; both action returns are zero.

| Observation | Before | After |
|---|---:|---:|
| Pytest wall | 7.71s | 7.58s |
| Cgroup CPU | 12.390s | 12.799s |
| Cgroup peak memory | 474730496 bytes | 473272320 bytes |
| PB contained wall | 14.747s | 14.752s |
| py-spy100Hz samples | 986 | 1036 |

Both allocations use preferred cores0-1. Before sampled module import is5.78s
inclusive, versus0.21s fixture and0.46s interrupted-walk test; the removed
spelling test itself is0.02s sampled. Inclusive samples overlap, are not wall
wait measurements, and must not be added. Profiles:

- Before d8115f6019c770e2eb8c83d85e8d48ed33ff8d9858a4baf8a409168d3b9bd67b.
- After e18390f3624e63f35e678c704e0386f879fdb16436445520804fce88119eaae4.

Both profiles are produced successful py-spy0.4.2 CAS blobs; exact commands,
package provenance, inventory, scope telemetry and exit propagation are in
PB terminal/log/receipt records. The local complete evidence checkpoint is
/home/rob/tmp/pq1929-head-walk-evidence.json, with all six raw Netdata series.

## Telemetry and confounds

CPU busy excludes idle/iowait, one-second average buckets covering each PB
window (before1790997309..1790997325; after1790997541..1790997557):

| Host | Before busy | After busy |
|---|---:|---:|
| dl380g10 | 12.704% | 13.899% |
| sparky | 7.482% | 3.082% |
| sparklina | 1.711% | 1.700% |

PB's own slightly different window averages are12.807% /14.188%; both retain
explicit missing-pqteld diagnostics. These are sequential single observations
on a non-isolated box; the routed baseline overlaps the after arm. The0.13s
pytest difference is not a reliable speedup estimate; contained wall is flat
and CPU slightly higher. This slice removes at least0.475s of literal
success-path waits (two ordering checks, two journal units, three overlay
cells; the hash wrapper can run additional times) and fragile scheduler
assumptions, not a full-suite performance claim.
The larger previously accepted #2095 and #2103 profiles are retained evidence,
not remeasured or attributed to this change. #1929 remains OPEN; union suite,
line-coverage comparison and under-five-minute acceptance belong to the
single compatible integration qualification, not this focused run.

A first pbtest invocation rejected unsupported -q before publishing work;
a commit attempt failed for unset Git identity before candidate submission.
Neither is a test result. The unrelated routed-profile client later disconnected
for the authorized Sparky reboot; its DL action was not cancelled.
