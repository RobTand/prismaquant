# Compact read-plan section construction proposal

Refs #1086/#2115. This is a separate proposal on
`sol/pq-readplan-construction-cpu`, based on PR #2120's frozen
`5a931f3ba4fc386ef417afee5851dc63909d4058`. The owner agreed to a separate
worktree; their branch and PR #2123 remain unchanged. Root owns review and
integration. The implementation commit is
`e720a4f73e649df927429a753ae2c4e1503bdb72`.

## Finding and change

The existing paired profiles of the canonical synthetic workload showed
1,128,661 checked `UInt64Rows.append` calls, taking 8.05–8.33 cumulative
seconds within the compact plan. `PackedReadPlan.append` already admits
all section capacities before appending each row separately. Per-row frozen,
capacity and method-dispatch work repeats that admitted section's checks.

The proposal extends records, input offsets and gradient offsets once per
section through the existing UInt64Rows array owner. Exact integer type,
uint64 range and row widths are validated before extension; capacity and
frozen guards remain fail-closed. A header still uses ordinary append.
No second packed buffer/cache is added. Last-use storage, frozen views,
chunk/offset order, IO-engine ownership and constructor-only research
selection remain unchanged. `packed_read_plan=False` remains the default.
The existing harness adds `--packed-only` for two fresh compact arms,
reusing qualified original legacy/before profiles.

## Actual before/after profiles

Workload: 867 synthetic targets × 512 invocations × four probes, 443,904
packed firing rows, 2,219,520 declared maximum parts, 1 MiB read budget,
18,645 chunks and 74,580 engine descriptors. Firing storage is held equally
before tracing. No payload is allocated/read; reader readiness stays false.
Both action parents and fresh children retained PB affinity `[0]` on
`dl380g10`, CPU1/memory6 GiB, native threads1, Python3.14.4,
torch2.11.0+cpu, transformers5.16.1, pinned PB95a/Tesserab40.

The before evidence is the two original compact arms in PB action
`c482f415ab4ca7d52bb7c00e867b5a3dbb47bf5a73e322a346e02c013f52698b`.
After action
`d3050bb86cf3bce7e686b86f92cff93a18e49928be236effb0e6105b959e6afb`
completed unambiguously, exit0. The old profile/source/artifact evidence was
rechecked rather than rerunning completed controls. Both intervals use
cProfile and tracemalloc around `_plan` plus replay-stream construction;
digest verification and stream close run outside those instruments.

| Metric | Original compact arms | Proposed compact arms |
|---|---:|---:|
| Profiled construction seconds | 44.865 / 43.571 | 30.364 / 36.265 |
| Mean profiled construction seconds | 44.218 | 33.314 |
| `_plan` cumulative seconds | 39.251 / 38.034 | 25.447 / 30.815 |
| `PackedReadPlan.append` cumulative seconds | 10.704 / 10.328 | 2.767 / 3.367 |
| `UInt64Rows.append` calls | 1,128,661 | 18,645 |
| Row-section extension calls | 0 | 37,290 |
| Scalar-section extension calls | 0 | 18,645 |
| Total cProfile calls | 26,443,967 | 20,658,654 |
| Retained plan bytes | 17,948,061–17,948,112 | 17,950,324–17,950,382 |
| Retained plan plus engine bytes | 96,955,635–96,955,681 | 96,957,503–96,958,350 |
| Child process maximum RSS bytes | 828,108,800–828,604,416 | 827,211,776–827,740,160 |

The observed mean is **24.66% lower for profiled synthetic construction**.
The after arms vary substantially, and cProfile taxes Python calls while
tracemalloc taxes allocation: this is not an unprofiled throughput or whole
quantum improvement. The attribution is the eliminated per-row append work;
common planner loop time also varies between arms. Plan memory changes by
about 2.3 KiB; the original compact storage benefit is preserved.

All six before/after arms, including the unchanged original legacy controls,
have the same ordered chunk/offset/last-use digest:
`97de9210515722907427d826b6f3a0ad9ba29a9d42e1d642e5483645ebb2c8a9`.
No GPU/model run, payload IO throughput, KL/bpp, or production improvement is
claimed. Representative #1086 acceptance remains open.

## Host evidence, qualification and source

After scope: wall90.316s, CPU89.599s (0.992 cores), peak1,058,201,600 bytes.
Executing-host PB Netdata: 96 CPU samples, mean box busy7.57%, peak14.87%,
CPU some-pressure avg10 maximum0.24. The original scope had mean box busy10.04%
and pressure maximum1.26. The after run has 44 complete snapshots per Spark
host, at least48 metrics each and no collector errors; before has98 per host.
Those GPU-box views describe external background load, not this CPU workload.
Both executing-host profiles retain the missing pqteld CSV notice while their
Netdata CPU series is present. Four-versus-two-arm scope totals are not a
per-arm CPU or memory comparison.

Targeted PB checks: **96 passed, zero failed/skipped**. They cover strict
section integer/width/geometry/freeze behavior and storage identity14,
existing compact plan/real operand/integrity/reclaim behavior20,
existing packed firing records21, spill integrity15, architecture/staleness19,
and duplication guard7. Pin-guarded compile action
`69d5695436c26aeca2d10a5f714b2e4543fa3b1094990ba4f5bc3c8965a5c9a3`
completed exit0. Every shard checks noneditable Git/RECORD PB
`95a59051d48cda82eea7927f31870c6c862d7174` and Tessera
`b40c93cb73745097e57a1ba4cf5b9eee166c759a`; compile rechecked both after the
profile. They are unchanged scoped environments, not a repin or deployment.

Read-only inspection independently hashed CAS results/receipts and attempt
logs, fetched exact source bundles, compared qualified code blobs and verified
all14 profile/report/Netdata artifacts across both actions. After result
SHA-256 `f6970647576fa67f76903d90f608eedec79b4e8954a1c512bcda836f640df2a8`,
2,884 bytes; receipt SHA-256
`233a828c2295a3a73468dc82449e0e1fb71d1c6471a7c52ab1e74b0ee88d0acb`.
After PB snapshot commit `1fe14c5d21b97efdca89814bb7a4d7c9d9cdf7bf`,
source bundle `f967fd4bf57c9e5b036bc1609a68cc734b2a5c8603900d66dbfe6c212b5c937b`,
40,914,906 bytes, parent `e720a4f73e649df927429a753ae2c4e1503bdb72`.
Qualified PrismaQuant package digest
`3613e41ad6fe640fa9c276d58ac349564be34f4fdf9896ac2accc317c27175d1`.
Earlier 20-case controls predate the max-uint64 test/prose/harness-only edits;
the production arrays match. The final14-case shard qualifies the delivered
max-uint64 test, and the final harness is the one profiled/compiled. This
inspection does not independently attest the worker.

## Reproduction and bounded artifacts

Submission from the original CPU measurement host uses published pbrun:

```sh
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /mnt/shared/astra-pq-readplan-section-source-20261002 \
  --measurement --cpus 1 --demand mem_gb=6 --priority -10 --timeout-s 600 \
  --env PYTHONPATH=. --env CUDA_VISIBLE_DEVICES= \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  -- /home/rob/venvs/pq-pb95a59051-tessera-b40c93cb/bin/python \
  -m experiments.spill_readplan_metadata --packed-only --out NEW_EMPTY_DIRECTORY
```

The first attempt was rejected before publication because measurement mode
added the submitting Spark tag alongside x86. Submission from dl380g10 then
refused a shallow source clone. The source was unshallowed and submitted
normally; neither refusal was bypassed or counted as execution.

Source/artifact directory `/mnt/shared/astra-pq-readplan-section-source-20261002`
is retained at the measured commit; after artifacts are
`/mnt/shared/astra-pq-readplan-section-after-20261002/`.
The owned audit root
`/home/rob/tmp/astra-resume-20261002/pq_spill/readplan-cpu/` contains
`BEFORE.json`, `BEFORE-AFTER.json`, `PROFILE-ARTIFACTS.json`,
`readplan-source-evidence.json`, 11 typed action records, reconciled test
reports, logs and exact profile/compile commands. No other worker's branch,
cache or scratch was edited. The proposal remains subject to root acceptance.
