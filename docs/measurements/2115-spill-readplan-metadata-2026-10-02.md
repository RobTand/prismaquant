# Stage B compact read-plan metadata — 2026-10-02

PQ #2115, PR #2120; partial work toward #1086. Implementation source
`8a7cef99b7a635243c7ba21c1625338b728ea171`; additional reclaim qualification
`9d59c67f3a4`. Parent #1086 remains open.

`StageBReplaySpill(packed_read_plan=True)` is a constructor-only research
option, default false. It enables the existing packed firing table, uses
the same bounded UInt64Rows storage abstraction, and stores chunk headers, record positions,
input/gradient offsets and last-use positions in compact arrays. Published
chunk queries expose frozen views. Per-probe reader callbacks retain an
index and resolve those views on the existing reader. The existing IO engine
still owns every descriptor and scheduling state.

The conservative spill geometry caps each array section. A probe-0 firing
beyond the declared total refuses before another input entry or payload
write. The legacy first-use chunk order, offset calculation, per-record
gradient association, checksum gate and last-use reclamation semantics are
preserved. Pipeline callers and sealed phases do not select this option.
There are no changes to IO-engine scheduling, source-layer lifetime,
produced-output/spool writers, numerics, formats or serving gates.

## Measured memory and CPU tradeoff

Workload: 867 synthetic targets, 512 invocations, four probes, 443,904 packed
firing records held equally before tracing. Both modes construct the real
spill plan and the existing IO-engine descriptors, using a 1 MiB read budget,
varied tensor row counts/residues and paired targets sharing inputs. This
isolates legacy versus compact **read-plan** metadata; it does not compare
list firing storage against packed firing storage. Declared conservative
maximum: 2,219,520 parts. No scratch payload is allocated or read; the real
reader readiness gate remains false.

Four fresh subprocesses run serially in legacy/packed/legacy/packed order,
under one admitted, isolated x86 CPU measurement on dl380g10. The parent and
children preserve PB affinity [0], native threads one, CPU reservation one,
memory reservation 6 GiB, and a 600-second action deadline. cProfile and
tracemalloc cover plan construction and replay-stream construction.
The ordered-plan digest and telemetry checks run after those instruments stop.

Values below are decimal bytes and observed arm ranges. cProfile times
include profiler/tracemalloc overhead; they are not production throughput.

| Metric | Legacy read plan, two arms | Compact read plan, two arms |
| --- | ---: | ---: |
| Retained plan metadata | 120,813,520 | 17,948,061–17,948,112 |
| Plan construction traced peak | 124,665,885 | 21,574,026–21,574,077 |
| Retained plan plus engine descriptors/state | 202,802,449–202,803,066 | 96,955,635–96,955,681 |
| Plan plus engine traced peak | 204,110,871 | 98,264,204–98,264,255 |
| Child process maximum RSS | 1,161,666,560–1,162,362,880 | 828,108,800–828,604,416 |
| Profiled construction seconds | 32.764–33.248 | 43.571–44.865 |
| cProfile calls | 13,770,837 | 26,443,967 |
| Full-chunk callback bindings | 74,580 | 0 |

Retained plan memory decreased 85.1%; retained plan plus engine memory
decreased 52.2%. Compact construction took about 34% longer in these profiled
arms. The profiles attribute the added work to checked array appends in
`PackedReadPlan.append`/ `UInt64Rows.append`: planner cumulative time
27.990/28.353 seconds becomes 39.251/38.034 seconds. This is a measured
memory/CPU tradeoff, so the option remains research-only.

All four arms produced 18,645 chunks and 74,580 descriptors and the same
ordered chunk/offset/last-use SHA-256:

```text
97de9210515722907427d826b6f3a0ad9ba29a9d42e1d642e5483645ebb2c8a9
```

The legacy callback-binding count denotes references to shared chunk tuples,
not 74,580 independent deep chunk copies. The original before-run JSON used
the earlier field name `retained_reader_chunk_objects`; the paired reports
use the corrected `retained_reader_chunk_bindings` label.

The PB action resource profile reports 201.530 seconds wall time, 197.664
scope CPU seconds, and 1,393,045,504 bytes scope peak memory across all arms.
The action averaged 0.98 CPU cores over its window. PB's executing
host Netdata window has 204 CPU samples, mean box CPU busy 10.04%, peak 12.31%,
maximum CPU some-pressure avg10 1.26. The profile records the absent dl380g10
pqteld CSV; the Netdata CPU series is present. The combined scope profile is
not a per-arm peak.

The experiment also records 98 complete Netdata snapshots on each of
sparky.lan and sparklina.lan over about 193 seconds, with at least 48 metrics
per snapshot and no collection errors. These are background host views;
dl380g10 executes this CPU workload. They are not served GPU measurements.
No GPU power, useful work per joule, GLM completion time, model peak memory,
KL or bpp delta was measured.

## Validation and attribution

Genuine pre-change regression:
`70f21a997c36da73a4cf9d826fce44004aaba41e4915ad7c1544caf556ac0212`
failed two assertions against the actual main spill core: declared geometry
did not refuse before read-plan growth, and readers retained full chunk
bindings. Failure did not rely on an unavailable constructor keyword or
missing import. Its sealed spill source matches main
`693a38f3ae3467ac3234cd9a7d756e41b69c3242`.

Final selection: **75 passed, zero failed/skipped, 75 collected**, with no
missing collection or uncounted outcomes:

| Selection | Passed | PB action |
| --- | ---: | --- |
| New metadata, operand/integrity, geometry, freeze, close, reclaimed-chunk reread | 20 | `03e3a8265275a8c01a9ae49e0ffae3b69b2425f5ddc3a6007a89b86dbc2ad159` |
| Existing packed firing metadata | 21 | `2e0c21ac265e0c0da482a561798550f855b0a188bd36c50563647a109cc269b3` |
| Existing spill integrity | 15 | `a8a40f137e82d2471e3012ee24ec7769feae1c334d1c2762d814d6adf56596d8` |
| Architecture doc | 13 | `20e2943c4ef1d9be2509ef8fce43be61746ee008bdec05843bfdb6c8bc815331` |
| Docs staleness | 6 | `80aa9892ae99be653cb1d73ad2e2a60efa51814cdc4368dc3a3c4916006b06f6` |

Four final changed Python modules compile under the dependency pin guard:
`1819e9025aeaf2a3a8b95aa3132aefaa8bdad6512f3b0aa0b0167320f5062d1d`.
Tests use Python 3.14.4, torch 2.11.0+cpu, transformers 5.16.1 on dl380g10,
two pytest workers per shard, two reserved CPUs, native threads one, 6 GiB
per shard, 600-second shard/90-second per-test bounds. The pin guard verifies
PrismaBuild `95a59051d48cda82eea7927f31870c6c862d7174` (44 RECORD files)
and Tessera `b40c93cb73745097e57a1ba4cf5b9eee166c759a` (102).
Existing PyTorch Python-3.14 deprecation warnings remain.

Original before-only measurement:
`02e361f244e0dbb942ef2d86d864a928b42fd6cdc29cb3aee1a3c372152a11aa`.
Interleaved paired measurement:
`c482f415ab4ca7d52bb7c00e867b5a3dbb47bf5a73e322a346e02c013f52698b`.
Both complete unambiguously with return code zero. Paired CAS result:
`afe610d0de4649a2525f85b1f8bcf3edd4ae9f2ba606723a1316f30249e499ac`,
5,513 bytes; receipt
`2a5bd0a0a212a80ce8653cb73782b6ed5574dc7485e93b7ccd2947841fd7104b`.
Paired sealed source bundle
`1fc0aa40d0a90aeb9ad1405db87997d69aaa5b63fa17326545980c6e10d47d29`,
20,124,160 bytes, commit `93b4153a15be084b1f075ab1765ce114ad863916`.
Its production, harness and architecture blobs match the delivered
implementation; it predates the additional reclaimed-chunk test.
Final controls/compile contain that test. They precede the final one-line
architecture clarification that these arrays share a storage abstraction,
rather than a single raw array. The final architecture action qualifies that
wording and all four final Python blobs. The audit verifies the exact prose
replacement when reusing the earlier controls; it does not relabel the
measured implementation as a later source revision.

Published fleet runtime:
`d028dfee920b-1790960385-1d815cff72d1`.
Artifact paths:

- `/mnt/shared/astra-pq-readplan-2115-before-20261002/`: original JSON/profile
  and both-host Netdata.
- `/mnt/shared/astra-pq-readplan-2115-paired-20261002/`: four JSON reports,
  four cProfile pstats and both-host Netdata.
- `/home/rob/tmp/astra-resume-20261002/pq_readplan_1086/`:
  `EVIDENCE.json`, `FINAL-ACTIONS.json`, `VERIFIED.json`,
  `audit_receipts.py` and reconciled PB test reports.

The audit hashes actual CAS result bytes and source bundles, unbundles the
sealed sources and compares qualified file blobs, and verifies every paired
artifact hash/size. It checks terminal state and return codes independently
of the submission wrapper.

Reproduction uses the published `tools/pbtest.py` with the parameters above,
and, from dl380g10, `tools/pbrun.py --cwd
/mnt/shared/astra-pq-readplan-2115-source-20261002 --measurement --cpus 1
--demand mem_gb=6 --timeout-s 600 --env PYTHONPATH=. --env CUDA_VISIBLE_DEVICES=
--env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --
/home/rob/venvs/pq-pb95a59051-tessera-b40c93cb/bin/python -m
experiments.spill_readplan_metadata --paired --out NEW_EMPTY_ARTIFACT_DIR`.
The archive is a self-contained Git snapshot; each run needs a fresh output
directory because Netdata creation is exclusive.

## Remaining #1086 acceptance

Arrays remain O(firing records/input entries/chunks), bounded by declared
geometry. IO-engine descriptors and scheduling maps remain O(chunks ×
probes); capture entry/checksum owners and transient per-chunk replay/fill
structures remain. Empty chunks can still contain many records. This
increment does not establish constant total memory.

Parent #1086's representative GLM/proxy completion, depth progression,
whole-quantum peak memory and production integration remain unmeasured.
Historical e62aee44 results were not rerun. The shared pipeline quantum,
source capture, produced output and spool writers belong to other work.
