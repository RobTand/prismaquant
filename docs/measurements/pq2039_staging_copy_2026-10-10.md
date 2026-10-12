# PQ #2039 staged-copy measure, 2026-10-10

One admitted GPU action ran the device projected-unit check in ABBA order. Both arms read every
unit through the real staged read: `source_unit_weight` over a bound residency map whose staged
ranges hold the bytes. The arms differ in one request, the page-locked buffer. This report gives
the paired timing, the split of `aten::copy_` by call site, the Netdata of both Sparks, and what
stays on HOLD. It replaces the earlier version of this file, whose measure stubbed the read and
so did not show a saving on the real read path.

## What the change does, and where

Before the change the check read each unit into pageable memory. It then made one private copy,
`pinned.copy_(weight)`, into a page-locked buffer and enqueued the host-to-device copy from that
buffer (`tessera_campaign._prepare_device_projected_check`). The staged reader now fills a
page-locked buffer itself when the check asks for one (`pinned_host`,
`StagedShardReader._read_payload`). The check adopts that buffer.

The saving applies when the residency map stages the read. A pool read, an unmapped read and the
qualified-original owner's read take the other branch. `source_unit_weight` leaves the pin flag
out for them and pins after the read, so they keep their one private copy. The scope control
below measures that branch. This report does not measure the owner's sealed-material read itself.

Unchanged by the change, and covered by tests: first-mismatch order and text, dtype and equality
semantics, source-page retirement, private-buffer ownership until the host-to-device copy ends,
and cancellation. The authenticated source checks do not change, and no second cache exists.

## Bindings

- Base: `origin/main` at `30b6231df0`. Reviewed head: `d1c51a298b8f`. **Measured head:
  `27a6385dbd23`**, pushed to `prismaquant-2039`. Later commits change only files under `docs/`.
- Final paired action: `b54f384bd46ae0bf31651b0f34825fa8a00cc5c0262db2af82570c6dba7b347c`,
  exit 0, 141 s, `sparky`, GB10 `sm121`, driver `595.99.02`. Interpreter
  `/home/rob/venvs/pq-pbdc4803-tessera-b40c93cb/bin/python` (Python 3.12.3, torch
  2.11.0+cu130). The measured tree carried no uncommitted edit: the snapshot parent is the
  measured head and its `dirty_sha256` is the SHA-256 of empty input.
- Result record: `/mnt/shared/tessera-measurements/pq2039-staged-pin-20261010/final-v4.json`,
  118,899 bytes, SHA-256
  `792f3abf2af14f767b312adcfc3809dd9c04d689fc39db52ccbc8e5e03a99e91`. The eight profiled-pass
  Chrome traces sit in `final-v4-traces/` beside it; the record holds each trace's SHA-256.
- Ordinary pool action, not a sealed `--measurement` (D26 submits those from a Spark). The ABBA
  order and both Sparks' Netdata give the load evidence a sealed admission would give by rule.
- Command:

```
pbrun --cwd <checkout> --tag gb10 --demand gpu=1,mem_gb=8 --cpus 2 --gpu-memory-gb 2 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --priority 0 \
  --timeout-s 1800 --detach -- \
  /home/rob/venvs/pq-pbdc4803-tessera-b40c93cb/bin/python -m tools.measure_projected_copy_2039 \
  --sdk-root /mnt/shared/prismabuild-fleet/qualification/pq-pb-sdk5-20261009/027103d9a8417e06c7f13356e58779a313cd7088 \
  --units 32 --rows 2048 --cols 4096 --arm-seconds 30 --min-passes 20 --profile-passes 3 \
  --control-passes 30 --declared-cpus 2 --declared-mem-gb 8 --declared-gpu-memory-gb 2 \
  --trace-dir .../final-v4-traces --out .../final-v4.json
```

- The map is validated by PrismaBuild's own validator. PrismaQuant pins SDK 5 (`027103d9`). The
  live fleet generation serves SDK 6, which PrismaQuant refuses by version, and no GB10
  interpreter carries SDK 5 installed. The action therefore binds the sealed source bundle at
  `027103d9` (`SDK_VERSION` 5) with `staged_lease.set_lease_helper_root`, as the repo's
  `reader_sdk_bound` fixture does. The record names the bound file. Probe action
  `a835c07427dc5c545acc22ca12724abde2c6e838e265e149a73736841e0cd21d` shows the refusal an
  ordinary admitted action meets today: `lease-helper-unsupported: prismabuild.client SDK_VERSION
  6, this package needs 5`.

## Method

- Fixture: one declared shard of 32 BF16 tensors `[2048, 4096]`, 16 MiB each (the GLM-5.3-Flash
  expert projection). Each tensor has one staged range file on local NVMe. A residency map names
  them. The staged bytes equal the declared bytes. Live CUDA tensors equal the source.
- A pass is the production serial check: `_start_projected_unit_check` for each of 32 units,
  then one `_settle_projected_unit_checks` and a CUDA synchronize. One pass reads 512 MiB.
  Source-page retirement is on in both arms.
- `before` withholds the page-locked request from the real read, which is the call the check made
  before the change. `after` is the shipped code. Order is before, after, after, before, with 426
  passes per arm. Four warm-up passes per kind size the arms from the median of the last three, so the slower arm lasts 30 s.
- Timing and profiling never share a pass. The timed passes run with no profiler. Three profiled
  passes follow each arm. Torch 2.11 keeps no Python stack on its events. The profile reads the
  Python frames from the exported Chrome trace. It charges each `aten::copy_` to the innermost
  frame outside torch on its thread. The frame line is the function's first line, so the split is
  per function. Profile microseconds include the profiler's own cost: they locate the work, and
  the timed passes measure it.
- Every arm must read all its bytes from the stage. The run fails if `bytes_from_stage` differs
  from the expected total, or if any pool byte or fallback appears. A mismatch pass flips one
  element in two live tensors. Both arms must name the same two units in the same words.

## Result

Timed passes, per arm (CPU is the process's user plus system seconds):

| Arm | Passes | Wall median (ms/pass) | CPU (ms/pass) | of it user / system | Minor faults/pass | Mean GPU board W (samples) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 before | 426 | 71.86 | 71.45 | 39.2 / 32.2 | 32.0 | 15.45 (31) |
| 2 after | 426 | 49.87 | 49.49 | 17.5 / 31.9 | 32.0 | 17.18 (21) |
| 3 after | 426 | 49.92 | 49.69 | 17.9 / 31.8 | 32.0 | 17.33 (21) |
| 4 before | 426 | 74.46 | 74.31 | 41.7 / 32.6 | 32.0 | 16.09 (32) |
| **Pooled before** |  | **73.16** | 72.88 |  |  |  |
| **Pooled after** |  | **49.90** | 49.59 |  |  |  |

Wall time falls **31.8%** (23.26 ms per 32-unit pass, 0.727 ms per 16 MiB unit). CPU seconds per pass fall 32.0%. The two arms of one kind differ by 3.55% (before) and 0.08% (after). Private staging copies counted over the timed passes: 27264 before, 0 after.

Where each pass spends its time (ms per pass, timed passes):

| Arm | Read | Prepare | of it outside the read | Launch | Settle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 before | 46.69 | 67.60 | 20.91 | 3.19 | 0.48 |
| 2 after | 42.81 | 45.17 | 2.36 | 3.71 | 0.47 |
| 3 after | 42.86 | 45.28 | 2.42 | 3.65 | 0.48 |
| 4 before | 48.25 | 69.69 | 21.44 | 3.95 | 0.47 |

The removed work is user-space work. User CPU per pass falls from 40.5 to 17.7 ms. System CPU
stays at about 32 ms: it is the kernel's copy out of the staged file, which both arms still make.
Of the 23.3 ms saved per pass, 18.8 ms is prepare work outside the read: the private copy and its
second page-locked allocation. 4.6 ms is a cheaper read: the pageable read allocated and
zero-filled a 16 MiB buffer for each unit, and the page-locked read takes a cached buffer. The
measure does not isolate that further. Minor faults per pass are equal in both arms. The launch
stage differs by 0.1 ms.

Every arm read all of its bytes from the stage (230,854,492,160 bytes per arm, 0 from the pool, 0
fallbacks). The mismatch pass named `w1` and `w9` in both arms with identical text.

## Attribution of `aten::copy_` by call site

Profiled passes: 3 passes of 32 units after each arm.

| Arm | Call site (function) | Copies | Self CPU (ms) | Self CPU per copy (µs) | Nested `cudaMemcpyAsync` (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 before | `tessera_campaign.py:5422 _prepare_device_projected_check` | 96 | 51.33 | 534.7 | 0.00 |
| 1 before | `tessera_campaign.py:5462 _launch_prepared_projected_check` | 96 | 0.97 | 10.1 | 2.17 |
| 1 before | `tessera_campaign.py:5685 _settle_projected_unit_checks` | 3 | 0.01 | 3.8 | 0.77 |
|  | **all `aten::copy_` (`FunctionEvent` self CPU)** | 195 | **52.32** | 268.3 |  |
| 2 after | `tessera_campaign.py:5462 _launch_prepared_projected_check` | 96 | 1.41 | 14.7 | 2.13 |
| 2 after | `tessera_campaign.py:5685 _settle_projected_unit_checks` | 3 | 0.01 | 3.8 | 0.76 |
|  | **all `aten::copy_` (`FunctionEvent` self CPU)** | 99 | **1.42** | 14.4 |  |
| 3 after | `tessera_campaign.py:5462 _launch_prepared_projected_check` | 96 | 1.31 | 13.6 | 2.05 |
| 3 after | `tessera_campaign.py:5685 _settle_projected_unit_checks` | 3 | 0.01 | 3.9 | 0.77 |
|  | **all `aten::copy_` (`FunctionEvent` self CPU)** | 99 | **1.32** | 13.3 |  |
| 4 before | `tessera_campaign.py:5422 _prepare_device_projected_check` | 96 | 51.29 | 534.3 | 0.00 |
| 4 before | `tessera_campaign.py:5462 _launch_prepared_projected_check` | 96 | 0.96 | 10.0 | 2.33 |
| 4 before | `tessera_campaign.py:5685 _settle_projected_unit_checks` | 3 | 0.01 | 4.2 | 0.71 |
|  | **all `aten::copy_` (`FunctionEvent` self CPU)** | 195 | **52.27** | 268.0 |  |

In the `before` arm 195 `aten::copy_` calls ran over 96 units: 96 staging copies, 96
host-to-device copies and 3 verdict reads. The staging copy at `_prepare_device_projected_check`
holds 51.3 of the 52.3 ms of `aten::copy_` self CPU (98%), at 535 µs per 16 MiB unit. The
host-to-device copy costs 10 µs of self CPU per unit plus 23 µs inside `cudaMemcpyAsync`. In the
`after` arm the staging site is gone. The remaining 99 calls cost 1.4 ms in total. The issue's
profile (`aten::copy_` 1.04 s over 1,729 copies, `cudaMemcpyAsync` 0.03 s over 865 calls) has the
same shape: the aggregate is the staging memcpy, and the enqueue is small.

## Scope control: no residency map

Same ABBA, 30 passes per arm, no map bound, pages resident.

| Arm | Pages | Wall median (ms/pass) | CPU (ms/pass) | Read (ms) | Prepare (ms) | Copy site (self CPU ms / copies) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| unmapped-before | resident | 47.30 | 47.33 | 6.11 | 45.40 | `_prepare_device_projected_check` 84.8 / 96 |
| unmapped-after | resident | 47.45 | 47.58 | 42.17 | 43.12 | `source_unit_weight` 83.1 / 96 |
| unmapped-after | resident | 48.03 | 47.72 | 42.34 | 43.29 | `source_unit_weight` 81.6 / 96 |
| unmapped-before | resident | 47.75 | 47.50 | 7.99 | 42.80 | `_prepare_device_projected_check` 84.7 / 96 |

Pooled wall median 47.52 ms before, 47.74 ms after (+0.46%). Pooled CPU 47.41 ms before, 47.65 ms after (+0.49%).

With no map the pool serves every unit and `source_unit_weight` pins after the read. The copy
moves from the check into the read (`_prepare_device_projected_check` before,
`source_unit_weight` after). The cost stays within noise. This pass is faster than the staged
`before` pass (47.5 against 73.2 ms). The pool read is one copy from mapped resident
pages. The staged read made two passes over the bytes. The change brings the staged read
to within 2.4 ms of it (49.9 ms). The control keeps its pages resident. In an earlier run the control
retired its pages after each read. Every pass then re-read the mapped file from disk at 2.5 GB/s,
and a CPU change could not have shown.

## Both Sparks' Netdata, load per arm

Mean over each arm's own seconds. Cells: CPU busy % / GPU board W / RAM used GiB.

| Arm | Seconds | sparklina CPU busy % / GPU W / RAM used GiB | sparky CPU busy % / GPU W / RAM used GiB |
| --- | ---: | --- | --- |
| before | 31 | 10.0 / 4.3 / 7.6 | 11.3 / 14.6 / 9.5 |
| after | 21 | 9.4 / 4.0 / 8.3 | 11.6 / 16.9 / 9.3 |
| after | 21 | 9.5 / 4.2 / 6.6 | 10.4 / 16.8 / 9.2 |
| before | 32 | 9.3 / 4.2 / 7.4 | 11.0 / 16.0 / 9.9 |
| unmapped-before | 2 | 9.5 / 4.0 / 8.2 | 9.2 / 15.1 / 10.1 |
| unmapped-after | 1 | 7.1 / 4.0 / 5.5 | 6.6 / 14.5 / 10.1 |
| unmapped-after | 1 | 7.1 / 4.0 / 6.5 | 10.9 / 13.9 / 10.1 |
| unmapped-before | 1 | 8.8 / 4.0 / 7.0 | 11.2 / 13.4 / 9.8 |

The control arms last 1 to 2 s, so their means rest on one or two Netdata rows.

Clock probes (peer minus local, by `ssh date`): sparklina: offset +0.003 to +0.004 s, bound ±0.003 to ±0.004 s over 3 probes.

The arms ran on `sparky`. Its CPU busy share stayed between 10.4% and 11.6% over the four paired
arms, and `sparklina` between 9.3% and 10.0%. Both series were complete. Every required chart on
both hosts returned fresh, finite samples for the whole window (140 rows per chart over 139 s).
Each chart passed `validate_netdata_window`. Netdata hides its `idle` dimension, so busy is the sum of
the other dimensions without `iowait`. The raw series are in the result record.

## Reservations

Declared by the submitting command, and observed:

| Quantity | Declared | Observed |
| --- | ---: | ---: |
| CPUs | 2 | affinity [5, 6]; cgroup CPU 127 s over 143 s wall (0.88 cores mean) |
| Memory (GB10 aggregate) | 8 GiB | cgroup peak 2.91 GiB; process max RSS 2.17 GiB |
| GPU subset | 2 GiB | CUDA allocated peak 0.52 GiB, reserved 0.54 GiB |
| Box window (PrismaBuild) |  | CPU busy mean 11.1% peak 33.8%; GPU 15.0 W mean, 17.8 W peak; unified used peak 12.7 GiB of 121.6 |

The declared memory follows D30: the observed peak plus a margin of about 3 GiB, rounded up.

## Repeats

Two earlier paired actions ran the same paired code at earlier heads. They differ only in the
control and the record fields added since.

| Action | Head | Host | Passes | Wall before to after (ms/pass) | Wall | CPU | Staging copies |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `61d409e95619` | `edb045c57c72` | sparklina | 421 | 72.14 to 50.18 | -30.5% | -30.4% | 26,944 to 0 |
| `b7f1902feb62` | `72fea5dbc926` | sparklina | 417 | 73.06 to 49.80 | -31.8% | -31.8% | 26,688 to 0 |
| `b54f384bd46a` | `27a6385dbd23` | sparky | 426 | 73.16 to 49.90 | -31.8% | -32.0% | 27,264 to 0 |

Records: `final.json` (SHA-256 `6a3dd79be9de4cd8ae2435abe20970fbe507c1a9cc9149539c09683a6a4fc40e`),
`final-v2.json` (`1717594007623a914429ec59ff7d19cac43719f8a3966b81bc982f00c6f3d730`), both beside
`final-v4.json`.

## Tests

All tests ran through PrismaBuild. CPU runs used `--tag x86` and the SDK 5 interpreter
`/home/rob/venvs/pq-task-suite-layer-sdk5-20261009/bin/python`.

- Red, at the reviewed head `d1c51a298b8f`, GB10, the staging tests of that head:
  `ccc8e75ab05bf84d57484f89dbe946ba35734c77001be9c8301245d1b6be02e4`: 2 passed, 9 errors. The
  staged tests could not bind an SDK on a GB10 (`SDK 5 needed, dc4803da installed`), so no GPU
  action had ever run them.
- Green, at the measured head, GB10 (`pq-pbdc4803-tessera-b40c93cb`):
  `c3a1308ec58980bc658b147bb32493c74746692df101b68b114e6f578954651a`: 112 passed, 0 skipped.
  The files are `test_projected_staging_copy_2039`, `test_measure_projected_copy_2039`,
  `test_collect_row_netdata_2039`, `test_projected_preparation_2039` and
  `test_projected_unit_check_1935`. The CUDA tests ran for the first time. They found a bug in the
  test helper that counts copies, now fixed.
- CPU sweep at the measured head: 27 files that name a changed module or guard the tree.
  529 tests, 4 shards, 0 failed. The 10 skips need CUDA.
  `579cdf9d883da92313881d2865eb157ffea9263d54f879b7def3d7775458f41b`,
  `94fb60ddc50cd034617a9baece6758b62d1780493f65d1071a9f74f7e6b4ae4d`,
  `782642076c45c33ba749384959d931bcb388d5e1b1145e7861a302674981a194`,
  `3bf87afc3946da580c6e5d0ac57d72d5e1731baceacc90b2913132f1c5504215`. The sweep found two
  guard failures in an earlier revision of the new measure (a same-name helper, raw `hashlib`
  calls). Both are fixed.
- Red, at the reviewed head `d1c51a298b8f`, x86, the repo guards
  (`test_io_site_freeze`, `test_duplication_baseline`, `test_tool_imports_1304`,
  `test_prismabuild_boundary`): `71f8c925fe6eee4a023a76a4acd5ce52405f3cf1208edc8030550d7013213a4a`:
  1 failed, 171 passed. `test_io_site_freeze` refused the old measure's new thread
  (`tools/measure_projected_copy_2039.py::main.run_arm`). Green at the measured head in the sweep
  above: the new measure uses the shared `PeriodicSampler`.
- CPU dry run of the measure's entry point (D38), `--cpu-dry-run` with the sealed SDK 5 bundle, at
  the final head: `023d3ecd7cf0823d5329314f8515c75d63f406da274ed1991079c568a61902d6`. Four
  units read through the stage, 0 pool bytes, 0 fallbacks, SDK version 5.
- Documentation and tmpfs-sensitive tests at the final head, on default scratch `/home/rob/tmp`.
  The Stage B spill and cotangent scratch guards refuse tmpfs, so these run off `/tmp`:
  `44213282a56713a08ed37bd0756041deb118a5d9c65c4bc6e0d5b9637d0db023`: 132 passed, 1 skipped. The
  skip needs a DIO-capable worker and has nothing to do with this change.

## HOLD and limits

- **Energy and work per joule: HOLD.** The only power reading is the GPU board's, by
  `nvidia-smi`. It reads 15.5 to 16.1 W in the `before` arms and 17.2 to 17.3 W in the `after`
  arms. That is about 11 to 12% of the 140 W envelope. It does not see the CPU and memory side, where the removed
  copy ran, so joules per pass from it would not describe the change. This run read no sensor
  that closes that gap.
- **Clock alignment: HOLD.** The probe from `sparky` to `sparklina` gave an offset of +0.003 to
  +0.004 s with a bound of 3 to 4 ms. The probes of the other runs gave bounds of 3 ms to 150 ms.
  The bound is not stable, so no cross-host alignment is claimed.
- No saturation claim. The pass is host-bound: 43 of 50 ms is the staged read. GPU utilization is
  not diagnostic on GB10, and board power stays near 17 W.
- The fixture's stage is local NVMe with a warm page cache. A production stage serves from NFS,
  SSD or RAM. The removed work is a memcpy and an allocation. It should cost the same per
  unit on any stage (0.73 ms per 16 MiB here). That is expected, not measured. The share of a
  pass changes with the read.
- The chunked read of an NFS stage fills the same pinned buffer in disjoint windows. The tests
  cover that path with a four-stream mount table, and no NFS stage was measured.
- **Live reachability.** The saving needs the residency map to validate. An ordinary admitted
  action today resolves the live SDK 6 generation and refuses the map (the probe above). The
  staged read, and this saving with it, is reachable in a live campaign row only when the SDK pin
  and the live generation agree. This measure binds the sealed SDK 5 bundle explicitly to show the
  read path. It does not show that a live row binds that bundle. The follow-up issue filed with
  this change names the open decision: which side moves.
- No full-campaign, export, serving, KL or bpp claim. The L20 original-source control that
  motivated the issue reads through the qualified-original owner. That owner's read keeps its one
  private copy under this change.

## Reproduce

```
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py --cwd <checkout at 27a6385dbd23> ...   # the command above
```

The tables are the record's own fields: `arms[].wall_median_s`, `.cpu_s_per_pass`, `.phase_s_per_pass`,
`.profile.aten_copy_sites`, `unmapped_control.arms[]`, `netdata.arm_means[]` and `reservations`. The
measure builds its fixture under `$TMPDIR` and removes it. Its CPU entry point is
`python -m tools.measure_projected_copy_2039 --cpu-dry-run --sdk-root <bundle>`.
