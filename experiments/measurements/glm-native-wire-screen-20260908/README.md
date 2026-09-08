# GLM v3 original-wire screen preparation — 2026-09-08

CPU preparation only. **No native run or fit qualification has been performed.**
The experimental entry point refuses execution without a frozen envelope, the
reviewed selected-source authentication API, and a hash-bound complete canonical
capture. The live partial capture journal is not accepted or relabeled.

## Scope and sources

[`glm_native_wire_screen_plan.py`](../../glm_native_wire_screen_plan.py) checks
the four selected original GLM units and all 28 cells against the sealed v3
**complete-group initial rung grids**. It ignores the superseded per-shape
screen suggestion retained in the group inventory. The complete v3 union
proposal, native proposal, census and group contract are content-bound in
[`resource-plan.json`](resource-plan.json). Its SHA256 is
`ee7ec9a8178caf5e4713bf3b3a5202af7e056c813be3f22c2a2efe0b886baa75`.

The selected shapes are L0 down `[4096,12288]`, L0 gate `[12288,4096]`,
L6 expert 0 down `[4096,2048]`, and L6 expert 0 gate `[2048,4096]`.
The original 512 × 512 seed-zero calibration is unchanged. Scoring retains
512 original rows per selected unit; H uses the full original census count:
262,144 for each selected dense unit and 8,132 for each selected expert unit.
Original BF16 weights remain the comparison reference; `TESSERA_BF16_K1` is
a lossy weight family, not the unquantized control.

The screen qualifies only these 28 original-wire cells. It cannot publish a
partial fused-group cost table, qualify the 864-member L6 expert group, change
the readable/production menu, attest serving, or establish full-model KL.

## Three different memory quantities

The CPU probe extends `selected_anchor_resources.v2` using actual source
headers, cache slots 2, one prefetch worker, an encoder memo capacity of 1,
and 24 GiB total runtime/workspace headroom. The existing source preparation
loads complete selected layers; the four independent BF16 copies total only
234,881,024 B. Their resident H/X total 801,112,064 B.

| Phase | Physical bound, B | Conservative GPU subset, B | Guard envelope, B |
|---|---:|---:|---:|
| Source preparation | 59,570,139,696 | 42,390,270,512 | 104,107,893,856 |
| Resident encoding and qualification | 57,664,516,656 | 13,971,324,928 | 73,783,325,232 |
| Capture prefetch | 33,499,906,048 | 10,255,073,280 | 45,902,462,976 |

The nominal physical maximum rounds to **56 GiB**, and the separately capped
GPU subset rounds to **40 GiB**. These are derived bounds, not measured peaks.
The GPU source bound conservatively charges the whole loader transient to CUDA
and includes 8 GiB of the total 24 GiB headroom.

`CaptureMemoryGuard` checks `cgroup.current + cuda.reserved + future <= cap -
2 GiB`. CUDA may already be included in the cgroup charge. The separate guard
envelope therefore adds the full phase GPU bound to physical demand plus its
2 GiB margin. Its maximum rounds to a **97 GiB requested PB memory cap**.
That extra reservation is a conservative admission consequence, not a claim
that this screen physically owns 97 GiB. Neither 56 nor 97 GiB certifies native
fit; observations and the unchanged guard can still refuse the run.

Verified capture loading reserves a 1 GiB private serialized-file cap, up to
four such files' cumulative kernel pages, decoded CPU storage, a transfer
destination and 64 MiB validation scratch. Completed wire/render pages are
also charged cumulatively: per-file advice is best effort, never admission
evidence. Existing PWC disk entries and a one-render LRU own rendered weights;
the harness creates no parallel residency system.

The proposed CPU reservation is 6: one main thread, four source-read workers,
and one prefetch worker, with native library threads bounded to 1. The
**3,600-second deadline is a hard termination cap**, not a completion-time
estimate. There is one attempt. These proposed values still require the root's
frozen review before submission.

## Prepared native entry point

[`glm_native_wire_screen.py`](../../glm_native_wire_screen.py) follows:

1. Verify the frozen source proposal, resource plan, complete capture, original
   draw hashes, activation environment, encoder-source digest and packaged
   reader contract. Missing selected-source API propagation fails before CUDA
   initialization; no whole-checkpoint-hash fallback exists.
2. Use the existing streamed selected-weight snapshot and descriptor-bound
   source reader. Drain and release the runner before prefetched H/X. Verify
   projected expert weights through the existing producer byte comparison.
3. Prefetch all four verified capture entries, retaining original uncapped H.
   Derive static scales from the **full census**, including unselected siblings.
4. Use the shared plane-keyed encoder memo, `_measure_anchor`, ordinary wire
   records, existing PWC entries, and `verify_anchor_render`. Require all 28
   finite scores, exact payload prices, unchanged source/H/X tensors, bound
   encoder identities and exact decoded BF16 equality. Failures and unfinished
   candidates have explicit states; none is silently promoted.
5. Retain source/H/X tensor identities, original capture entry hashes, source
   file authentication, original wires, renders, scale tables and per-cell
   receipts. Original source weights remain in their hash-bound checkpoint;
   no new weight donor artifact substitutes for that checkpoint.

All phases have cProfile and CUDA peak records. First and last cells for each
shape additionally have Torch traces and operator tables for encode and
qualification. Both boxes' Netdata is sampled; incomplete required telemetry
prevents an overall pass. No per-operation energy interpolation or GPU
utilization-based saturation inference is made. Actual native kernels still
need inspection in the resulting profiles.

This is one proposed admitted action because the four selected snapshots and
H/X are held under common source/residency dependencies. The script does not
dispatch work or assign hosts. PB retains placement, grouping/fanout, admission
and termination ownership; there is no automatic follow-on full-group run.

## CPU validation and limitations

The final eight CPU checks pass on dl380g10 through PB action
`9e32e78c8800926e13cd48a594910d108190970454c515d3699906127a1d79b5`:
complete-group rate authority, shape/member/duplicate/missing-row refusal,
sealed-file replacement refusal, separate loader/physical/GPU/guard accounting,
and no unauthenticated native API fallback. Both experimental modules are
imported and compiled by these checks. The native numerical body has not run.

Final CPU resource derivation is PB action
`8f264a222b6c53a9e76fd980999f079dc2c964255b1eb1a1772350594f6fce40`.
Its CAS stdout seals the resource JSON's actual hash. Terminal exit codes,
CAS payload lengths/hashes, canonical receipt digests, zero OOMs, cleanup and
zero live descendants are independently checked in
[`receipts.json`](receipts.json). Neither CPU action requested a GPU.

Two preparation failures were corrected: the first negative membership fixture
shared its mutable member list with the proposal, testing the wrong refusal;
the first resource invocation used a script path without repository import
resolution. The fixture now owns separate lists and the entry point runs with
`python -m experiments.glm_native_wire_screen_plan`. Intermediate resource
plans omitted cumulative pages and/or the separate conservative guard envelope;
they are superseded by the checked-in final plan.
Their bounded failure evidence is retained in
[`negative-preparation-attempts.json`](negative-preparation-attempts.json).

Remaining gates are review/integration of source authentication, a complete
canonical capture, exact producer source pinning in the frozen envelope, and
root review of the harness/resources/invocation. The draft envelope is
deliberately rejected by the native entry point. No production defaults or
architecture contracts change here.

The reviewable, unsubmitted inputs are
[`native-plan.draft.json`](native-plan.draft.json) and
[`native-invocation.draft.json`](native-invocation.draft.json). Pending seals and
the draft schema must be replaced only after those prerequisites are satisfied.
