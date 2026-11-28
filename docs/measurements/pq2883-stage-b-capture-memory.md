# PQ #2883: Stage B capture memory attribution

## Baseline run

W1 refers to [PQ #2881](pq2881-stage-b-allocation-snapshot.md). This run uses its `test_stage_b_allocation_snapshot_record` harness. `PassProfileSession` in `prismaquant/stage_b_pass_profile.py` records the allocator history. The test replays that history and identifies the cuBLAS workspace blocks.

The figures below come from `pq2881/attribution.json` in the Flux check of `077ad32df8cb`. The GB10 host was `sparklina`. The run used CUDA 13, Torch 2.14.1, cuBLAS 13.1.1 and the native CUDA allocator. It recorded 339 allocator events and 211 CUDA kernel launches.

The snapshot covers layer 1, probe 0, group 0 of the two-layer BF16 routed-expert fixture. The capture batch is 2. The group contains stored batches 0 and 1. The input and incoming gradient each have shape `[2, 8, 16]` and stride `[128, 16, 1]`.

Set `TMPDIR` to the Flux job's local scratch directory before you run the test.

```sh
PQ2881_OUT="$TL3B_OUT/pq2881" python -m pytest -qs --tb=short --basetemp="$TMPDIR/pytest" \
  tests/test_stageb_one_pass_spill.py::test_stage_b_allocation_snapshot_record
```

The output directory contains `layer-001-p0-capture.memory_snapshot.pickle`, `layer-001-p0-capture.timing.json` and `layer-001-p0-capture.trace.json.gz`. `phases.pickle` holds the phase samples. `attribution.json` holds the replay totals, peak allocation stacks and workspace block sizes.

## Allocation owner

The cuBLAS workspace cache held 67,108,864 of 69,343,408 live bytes at the replay peak (about 97%). The replay counted 91,648 more bytes from blocks already live at entry. Two 33,554,432-byte blocks were live at group entry. Neither block had a free event during capture. Both blocks remained live until the harness cleared the cuBLAS workspace cache after the quantum. The clear identifies the owner. It does not identify the calls that allocated the blocks or show that capture can release either block safely.

Peak replay reservation was 71,303,168 bytes. Device-wide memory use rose from 95,821,135,872 to 95,874,330,624 bytes during the group (53,194,752 bytes). The allocator replay cannot attribute that device-wide rise to cuBLAS or any other owner.

## Limits

The historical W1 stacks attribute its two largest new live allocations to the spill input digest in `prismaquant/joint_replay_spill.py`. Those allocations are not the dominant owner. This baseline does not show a limiting overlap of the capture output and cotangent at staging.

The fixture has no GLM KDA layer and bypasses the production capture guard. The recorder resets the peak counters at group entry. These measurements do not establish full-action admission under 16 GiB or the full-workload ceilings. This record has no candidate comparison. It makes no claim about memory reduction, speed, payload equality or serving.
