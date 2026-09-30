# Fixed-rate routed expert partitions

This opt-in CPU delivery implements #1796, a slice of #1314. It does not
qualify a GPU run, promote a format or change default campaign layouts.

## Plan a new layout

Use a fresh workspace, a producer-bound census and a fixed rate:

```bash
python tools/dispatch_tessera_campaign.py plan \
  --spec SPEC.json --workspace NEW_WORKSPACE \
  --groups-per-row 1 --experts-per-row 8
```

The shared campaign argv must contain `--rate-band r,r --max-rounds 1`.
`--experts-per-row` is absent/zero by default. Dense groups keep their original
rows. Routed rows gain deterministic `-pNNNN` suffixes and contain at most the
specified number of experts, including every projection of each expert.
PrismaBuild places, balances and retries the resulting independent rows.
No worker is selected by this planner.

Refused combinations: adaptive/unset bands, multiple rounds, stack sampling,
seed checkpoints or seed workspaces, research exact-member selection,
work-based packing, and `--groups-per-row` other than 1. Producer records must
have uniform per-expert projection geometry. This bounds the metadata menu
work across chunks; heterogeneous geometry is not silently substituted.
A row-class override must preserve the fixed-rate contract.

## Selection and runtime contract

A partition uses `prismaquant.tessera_campaign_units.v3`. Its group retains
`key` and the full original `members` roster. The closed `partition` descriptor
uses `prismaquant.tessera_campaign_expert_partition.v1` and contains:

- `experts_per_row`: positive integer chunk size.
- `index` and `count`: zero-based chunk index and derived number of chunks.
- `rate_q256`: the pinned rate.
- `members`: the complete, sorted projection names of the selected experts.

The shared selection owner derives chunks from sorted producer expert IDs;
there is no sampled estimate or inclusion probability. A v1/v2 selection may
not hide this descriptor. The runtime refuses altered, overlapping or
role-incomplete chunks and prices only the declared subset.

The legal rate grid is still the intersection over the *full* group's menus,
not the potentially wider local chunk menu. The existing shape/family menu
cache supplies metadata-only menus for the full roster. No parallel rendered
weight cache or residency mechanism is added. Existing own-row demand and
PrismaBuild data-manifest owners consume the actual priced subset; the same
memory-fit guard applies before submission. Streaming proof requirements,
phase plans, capture bindings and checkpoint/wire validation remain in force.

## Merge and restart

The plan stores every partition selection. Merge derives the expected roster
from those declarations, never from whichever outputs survive. It requires
complete expert/index coverage, disjoint chunks, exact selection provenance,
exact per-member cost coverage and matching anchor-group membership. Missing,
changed, duplicated, seeded or undeclared partition rows refuse.

Successful merge unions existing per-unit prices and anchor membership, then
uses the existing whole-group selection, population and journal reconciliation.
It does not invent a partial-stack estimator or rewrite the Tessera wire.
Each row restarts its own identity-bound checkpoint; external seed adoption is
not supported by this first delivery. Replanning in a submitted workspace can
overwrite selection files, so use a new workspace for a different layout.

## Remaining #1314 acceptance

CPU fixtures establish selection, planner demand, readset publication and
plan-derived merge behavior. They do not establish GPU numerics or peak
residency. Before production use, compare one unsampled routed group against
all of its fixed-rate chunks under the same immutable producer, calibration,
family policy, rate, anchor count, cache settings and source proof. Inspect
terminal/CAS receipts, exact wire bytes, all non-timing cost/anchor fields and
measured peak memory; deleting one chunk must refuse merge.

GPU execution needs the coordinator's explicit approval in `PENDING-GPU.md`.
Use PrismaBuild, priority -10, explicit timeouts, no host pin, bounded native
threads and the U4 window guard. A speed claim additionally needs before/after
in-process profiles and Netdata series on both boxes; on GB10 compare power
against the ~140 W envelope and work per joule, not GPU utilization.
