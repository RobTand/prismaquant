# PQ #2097: CPU qualification of two-bank chain-roll host reuse

Source: `ee93f9655e6ddff8cbc6ed6bb33dc65fe08ef883`, based on main
`693a38f3ae3467ac3234cd9a7d756e41b69c3242`. PR #2105 closes the scoped CPU
child #2097 and references parent #1250. Public callers keep the allocating
path. Parent #1366's executable incoming-plane integration is separate.

## Measured result and its limits

On dl380g10, four interleaved arms of the same CPU allocation corpus gave:

| Arm | Groups / rows | Host allocation calls | Aggregate allocated bytes |
|---|---:|---:|---:|
| Allocating, first and second repeats | 32 / 128 | 128 | 134,217,728 (128 MiB) |
| Reuse, first and second repeats | 32 / 128 | 8 | 8,388,608 (8 MiB) |

Each group has four individually compact 1 MiB rows. The two held banks
allocate once each; subsequent groups allocate no host row tensors.
The explicit allocation counter and torch profiler's `aten::empty` events
agree exactly. Every arm's ordered output digest is
`8a69f06c4895e874b8abe9285c12fbceb710afe7e82a7c7b7d3142b905fdc8b2`.

These are real CPU allocations and copies through `_RollPipeline`, with
pageable replacements for pinned allocation and simulated CUDA events.
They establish allocation traffic, not live memory, pinned residency,
CUDA stream safety, GPU speed, energy or end-to-end campaign gain. No GPU
work ran. The cProfile and torch traces are retained per arm. Sixteen
complete before/after Netdata snapshots cover both Sparks, eight per host,
without missing required charts; the corpus spans about 1.7 seconds, so
these snapshots do not establish independent per-arm power/clock means.
PB's box window records the executing x86 host separately.

## PrismaBuild evidence

The published PB generation was `d028dfee920b-1790960385-1d815cff72d1`.
All runs were CPU-only, priority -10, with explicit 600-second deadlines,
1 native thread per worker and PB affinity preserved. The interpreter was
`/home/rob/venvs/pq-pb95a59051-tessera-b40c93cb/bin/python`:
Python 3.14.4, torch 2.11.0+cpu, transformers 5.16.1. Pin guards verified
44 PB files at `95a59051d48cda82eea7927f31870c6c862d7174` and 102 Tessera
files at `b40c93cb73745097e57a1ba4cf5b9eee166c759a`.

| Check | Full action key | Result |
|---|---|---|
| Original allocating core, constructor signature adapter only | `db44d90ab084f3c9a084ed4340584cccdd8be0a89f379cee839b1d3abe9d14ff` | Genuine RED: 15 allocations, oracle requires 4; rc 1 |
| Final lifetime/alias/bytes regressions | `17dfbc6ab441d8c7c1dfc314961895e769cdbc8571629b0536430c0791cc2207` | 20 passed, no skips |
| Final existing chain regressions | `c7e4cf2eb08d4fa8fb601448d6dcfd7eeff11bc6b93c611fce3240739ef633b5` | 31 passed, 6 CUDA skips |
| Checkpoint incoming adapter | `3d15364cc68054de094deb9fc83d12ed6b1011f542dfc19ffb80f08ebfb3d85a` | 17 passed |
| Referenced checkpoint | `a1b928760cd5b98af5797da1295bc616d4294c89c3e2d8cd1c390dddd05f008e` | 9 passed |
| Architecture mechanics | `ea762822fd5c0f1fa05833c1f74287acb0b168ac595ae511dc72951bb58e8959` | 13 passed |
| Documentation staleness | `a38ed1e8ace42e05dc554eb38613eae72cc2ad1e84dc0ab1b494df14ce5c415d` | 6 passed |
| Final compile and exact-pin guard | `d1e1f83ccca3cf0e236f897edd3fb33adb11fa1b8120082d68800d6c2f0c8d3d` | 4 modules compiled; rc 0 |
| Final paired CPU allocation profiles | `cc772c08144b6e59d965de3282241c3097c3c6eedddd10610f3bf3c83d0d6476` | rc 0; both repeats agree |

The selected coverage is 96 passed / 6 skipped / 0 failed, 102 collected,
with no missing collection or reconciliation problems. The final head's
changed lifetime code and new test were rerun with the existing chain suite;
the 45 disjoint checkpoint/documentation checks reuse their earlier receipts.
All Python outside `_RollPipeline` has an identical AST between those
checkpoint qualification snapshots and the final delivered module. The
extra architecture sentence describes the final callback fence only.
All 20 new cases pass without a CUDA skip. The six existing skips are the
pinned serialization case, four pinned-row regimes, and CUDA default/pre-997
bitwise equivalence. They supply no device qualification.

Every successful terminal record was checked: done / rc 0, complete,
untimed-out and unambiguous, with actual CAS result bytes hash/size checked.
Final lifetime, chain, compile and profile source bundles match all five
qualified delivered files. The allocation profile used an exact archived
source checkout, submitted natively on dl380g10 under `--measurement`,
CPU 1 / memory 4 GiB. Tests use PB-owned fanout, CPU 2 / memory 6 GiB per
shard, two pytest workers, native threads one.

Final profile receipt:
`ae46a2f910ee8076015dd61f15d7ddc72702f074ab0446db25eb4de4cb256134`;
CAS result:
`a5efaadff9bfae7e75d1aacba6d70a229b4e9e6fb8db341fbffea0c2d3516bc3`.
Artifact directory: `/mnt/shared/astra-pq-roll-host-2097-final-20261002/`,
four `.pstats`, four Chrome `.trace.json`, per-arm reports and `netdata.jsonl`.
All 13 artifacts were independently hash/size checked against that result.
Netdata SHA-256:
`25133f9b35175b29ce6bbe22b984f9960c95c607b827aef364b0ab796034b32a`.

Local detailed receipts and attribution:
`/home/rob/tmp/astra-resume-20261002/pq_stageb_io/EVIDENCE.json`,
`VERIFIED.json`, `profile-manifest.json`, `final-tests.json` and
`final-lifetime-tests.json`. The first selected run (95 passed / 6 skips)
and earlier profile at `7e3885b652d7` are superseded for changed lifetime
code by the final runs above, retained as bounded development evidence.
The original-core RED worktree is retained at `pq_stageb_io/red-wt`.

Two non-qualification failures are retained: the initial unprovisioned
`pb-cpu` interpreter missed compressed-tensors (no behavioral test ran),
and the first cross-class measurement refused the submitting GB10's
accelerator identity on x86 before workload execution. Current PB source
explicitly binds submitter platform/device facts; that refusal was correct.
The measurement was resubmitted from its actual x86 class with isolation
preserved. A lifetime submission also refused a concurrent commit during
snapshotting before creating an action; its frozen-head resubmission passed.

## Reproduction

CPU allocation evidence, submitted from the x86 measurement class against an
exact source archive, with a fresh output directory:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /mnt/shared/astra-pq-roll-host-2097-source-20261002 \
  --measurement --cpus 1 --demand mem_gb=4 --priority -10 --timeout-s 600 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=. --env CUDA_VISIBLE_DEVICES= \
  -- /home/rob/venvs/pq-pb95a59051-tessera-b40c93cb/bin/python \
  -m experiments.roll_host_reuse_profile --out /path/to/fresh/owned/evidence
```

Final changed-code regression gate:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbtest.py \
  --checkout /home/rob/tmp/astra-resume-20261002/pq_stageb_io/wt \
  --python /home/rob/venvs/pq-pb95a59051-tessera-b40c93cb/bin/python --tag x86 \
  --workers-per-shard 2 --threads-per-shard 1 --cpus-per-shard 2 --mem-gb 6 \
  --priority -10 --timeout-s 600 --test-timeout-s 90 \
  tests/test_roll_host_reuse_2097.py tests/test_chain_roll_overlap.py
```

## Remaining production acceptance

Astra owns critical review, any GPU window and production selection. Before
integration, fund both banks in the existing owner/guard and PB host-memory
reservation; do not add another residency owner. Capacity is exactly twice
the first group's sum of compact row bytes. At a four-row BF16 group with
2048 tokens and width 4096, this is 128 MiB; this sizing example is not a
measurement of a campaign's current geometry or spare memory. Banks remain
held across window drains, so admission must include that residency while
boundary inputs and write buffers are live.

First qualify real pinned copies/events and exact serialized output at the
same batch/fusion regime and shape, including short tails and failure paths,
in the known-good container through PB. Then profile the existing real
chain-roll workload with allocating/reuse arms interleaved, the same source,
calibration, grouping, output ownership and whole-row phase boundaries.
Use in-process traces and both-Spark Netdata; report allocation/page-fault
counts and measured peak memory. GPU speed/energy claims require qualified
power/clock evidence and useful work per joule. No GPU command is authorized
or launched by this CPU prerequisite.
