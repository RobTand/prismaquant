# PQ #2881: Stage B allocation snapshot

## Run decision

Use one Flux GPU job on the GB10 with CUDA 13. Budget at most five GPU minutes. Expect less than one GPU minute. Measure group 0 of probe 0 of layer 1 in the two-layer BF16 routed-expert fixture. Use two stored batches per group. Keep the existing capture pass and disable the windowed shadow. This case can exercise grouped GEMM, autograd saves and spill hooks in the same backward pass. CPU cannot record CUDA allocator events or identify the launched CUDA kernel.

This fixture has no GLM KDA layer. It cannot confirm the KDA mode or reproduce the historical 512-token workspace result. This run does not cover the full original shape. The fixture bypasses the production capture memory guard. This run does not qualify serving, numerical results or spill scratch.

The first attempt stopped at the spill qualification fixture's scratch prerequisite. It used 0 GPU seconds. The experiment uses private job scratch and keeps the runtime's scratch checks.

For the review follow-up, use the same fixture and group in a fresh Flux process. Expect less than one GPU minute. Match entry blocks to the blocks that PyTorch releases when it clears the cuBLAS workspace cache after the quantum. Read the operand fields from the CPU spans for the CUDA launches. A CPU run cannot identify the CUDA workspaces or correlate the CUDA launches.

Profile spec: `capture=0,windowed=none,wait=0,warmup=0,active=1,memory_group=0,memory_entries=100000`.

```sh
export PQ2881_OUT="$TL3B_OUT/pq2881-review-record"
python -m pytest -qs --tb=short tests/test_stageb_one_pass_spill.py::test_stage_b_allocation_snapshot_record > "$TL3B_OUT/pq2881-review-record.log" 2>&1
```

## Result

The review follow-up test passed in 8.61 seconds. The Flux box reported 0.0 GPU seconds at its display precision. The earlier check used 0.2 GPU seconds for the candidate. The trace recorded 211 CUDA kernel launches. The review follow-up ran in a fresh pytest process with no CPU-only profile. The environment used Python 3.12.3, Torch 2.14.1+cu130, CUDA 13.0, cuBLAS 13.1.1, Transformers 5.16.1 and Tessera Quant 0.1.0. The device was an NVIDIA GB10 (sm_121). Torch used the native allocator. The tables below use the review follow-up results.

The snapshot covers quantum `layer-001`, probe 0, group 0 and stored batches 0 and 1. The BF16 input and incoming gradient each had shape `[2, 8, 16]` and stride `[128, 16, 1]`. The recorder marked the history complete. It retained 339 events against a bound of 100,000.

## Allocation attribution

Seed the replay with the live blocks at group entry. Add `alloc` events and subtract allocations at `free_requested`. Track reservation changes separately with the segment events. Match storage addresses at forward end. Match allocation generations at the replay peak so later reuse of an address does not inherit its previous owner.

| Live allocation category | Forward end (bytes) | Replay peak (bytes) |
| --- | ---: | ---: |
| cuBLAS workspace cache | 67,108,864 | 67,108,864 |
| Other blocks live before group entry | 91,648 | 91,648 |
| Autograd save only | 27,472 | 27,472 |
| Spill input only | 0 | 0 |
| Autograd save and spill input | 4,608 | 4,608 |
| Other allocations in the group | 832 | 2,110,816 |

The cuBLAS workspace cache accounts for about 97% of the replay peak. Two 33,554,432-byte blocks were live at entry and had no allocation stacks. Neither had a free event during the group. Both matched live blocks by address and block metadata just before the cache clear. Neither remained allocated after `torch._C._cuda_clearCublasWorkspaces()`. The cache clear identifies the owner, not the calls that first allocated the blocks. These bytes were already live before the group.

The largest new live blocks at the peak are two 1,048,576-byte allocations. Their Python stacks include `_input_digest`, `_record` and `spill` in `joint_replay_spill.py`. These allocations occur on the spill backward path. They are not the inputs that the hooks retain from forward.

The autograd pack hook tags the storage of saved tensors. The spill observer tags the input storage that the backward callback holds. The callback clears its held references in `finally`. After the quantum, the test found no live references to the detached tensors from the pack hook. The table counts the 4,608 shared bytes once. The role tags identify the storages observed at forward end. At the peak they identify still-live allocation generations, not a count of outstanding callback references.

In the confirmation run, the spill observer tagged BF16 views of widths 16 and 32 with 5, 6, 8, 13 or 16 rows. The spill tags cover four distinct storage addresses. The autograd pack calls recorded 22 distinct storage addresses over the run. Those counts do not give the number of live tensors at the peak.

## Counter reconciliation

| Bytes | Replay peak | Counter peak | Peak residual | Replay end | Counter end |
| --- | ---: | ---: | ---: | ---: | ---: |
| Allocated | 69,343,408 | 69,349,888 | 6,480 | 69,329,504 | 69,336,064 |
| Reserved | 71,303,168 | 71,303,168 | 0 | 71,303,168 | 71,303,168 |

The allocated replay starts with allocator block sizes. New events can give sizes below the rounded block sizes. At exit the replay is short by 6,560 bytes. The block snapshots show rounding, but they do not assign every byte of the peak residual to a specific allocation. The allocated peak is not an exact replay match.

| Block bytes | Entry | Forward end | Exit |
| --- | ---: | ---: | ---: |
| Active allocated | 67,216,384 | 67,236,352 | 69,336,064 |
| Rounding inside active blocks | 9,152 | 12,080 | 15,712 |
| Inactive | 1,989,632 | 1,969,664 | 1,967,104 |
| Inactive within segments with active blocks | 1,989,632 | 1,969,664 | 1,967,104 |

Rounding is part of the active bytes. Inactive bytes in segments with active blocks are not independent releasable segments. Do not add either row to the allocator totals a second time.

In the review follow-up, device-wide `total - free` rose by 55,468,032 bytes from entry to exit. Torch reservation rose by 2,097,152 bytes. The 53,370,880-byte difference is outside this allocator replay. In the earlier check, the device-wide rise was 48,193,536 bytes. The device-wide counter does not identify the source of the difference. It can include library or driver activity and other device users. It is not a measurement of a non-Torch workspace for this group.

## Dispatch examination

The trace recorded 30 CUTLASS launches with `cutlass_80_wmma_tensorop_bf16` symbols. One example is the `s161616gemm_bf16_16x16_128x1_tn_align8` variant. The test matches kernel events to `cuda_driver` `cuLaunchKernel` events by correlation ID. It finds the CPU spans that contain each driver launch on the same thread.

The table counts launches by shape and stride in the parent CPU span. Each row has two BF16 operands. Each grouped row also has a cumulative offset operand of type INT32, shape `[4]` and stride `[1]`. Group rows use the `aten::_grouped_mm` span. The other rows use the `aten::mm` span. These layouts describe operator inputs, not the expert slice of each kernel launch.

| Launch context | Launches | Left shape / stride | Right shape / stride |
| --- | ---: | --- | --- |
| Grouped forward | 4 | `[32,16]` / `[16,1]` | `[4,16,64]` / `[1024,1,16]` |
| Grouped forward | 4 | `[32,32]` / `[32,1]` | `[4,32,16]` / `[512,1,32]` |
| `GroupedMmBackward0` | 4 | `[16,32]` / `[1,16]` | `[32,32]` / `[32,1]` |
| `GroupedMmBackward0` | 4 | `[32,16]` / `[16,1]` | `[4,16,32]` / `[512,32,1]` |
| `GroupedMmBackward0` | 4 | `[32,64]` / `[64,1]` | `[4,64,16]` / `[1024,16,1]` |
| `MmBackward0` | 1 | `[16,16]` / `[1,16]` | `[16,32]` / `[32,1]` |
| `MmBackward0` | 1 | `[16,16]` / `[16,1]` | `[16,32]` / `[32,1]` |
| `MmBackward0` | 2 | `[16,32]` / `[32,1]` | `[32,16]` / `[16,1]` |
| `MmBackward0` | 1 | `[16,4]` / `[4,1]` | `[4,16]` / `[16,1]` |
| `MmBackward0` | 1 | `[4,16]` / `[1,4]` | `[16,16]` / `[16,1]` |
| `aten::linear` | 2 | `[16,16]` / `[16,1]` | `[16,32]` / `[1,16]` |
| `aten::linear` | 1 | `[16,16]` / `[16,1]` | `[16,4]` / `[1,16]` |
| `aten::linear` | 1 | `[16,32]` / `[32,1]` | `[32,16]` / `[1,32]` |

The test queries the loaded cuBLAS library for its version. The version describes the environment. It does not show that cuBLAS launched the CUTLASS kernels. The symbol does not identify a standalone CUTLASS package version.

### Note for issue #2881

The bounded routed-expert capture observed `cutlass_80` symbols on GB10. Two cached cuBLAS blocks accounted for 67,108,864 bytes at group entry and at the replay peak. A symbol is not evidence of an incorrect kernel selection on sm_121. This run made no dependency comparison and found no basis for a dependency-only remedy. It gives no reason to change serving code.

## Artifacts

The review follow-up retained the outputs under `$TL3B_OUT/pq2881-review-record/` on the GB10 Flux box. Its log is `$TL3B_OUT/pq2881-review-record.log`. The earlier check kept its outputs under `$TL3B_OUT/pq2881/` in the candidate step. The confirmation run kept its outputs under `$TL3B_OUT/pq2881-confirm/`.

Use `layer-001-p0-capture.memory_snapshot.pickle` for allocator history, `layer-001-p0-capture.timing.json` for group counters and `layer-001-p0-capture.trace.json.gz` for dispatch. Use `phases.pickle` for the phase samples and `roles.json` for the storage tags. The follow-up `attribution.json` contains the replay, the workspace block sizes and the launch correlations with operand fields.

The shared measurement store was read-only. The jobs did not publish these files. Use a fresh directory in the issue's measurement store for publication.
