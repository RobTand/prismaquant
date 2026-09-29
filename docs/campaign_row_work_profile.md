# Opt-in campaign row work profiles

`tools/dispatch_tessera_campaign.py plan --row-work-profile PATH` packs whole
non-routed anchor groups using supplied timing predictions. The default remains
count-based `--groups-per-row` packing. Work-based packing requires
`--groups-per-row 1`. The planner writes a new layout; it does not resize PB
actions. It does not detect whether a workspace has already been submitted.
Use a new workspace for a different layout so replanning does not overwrite
selection files named by published rows. An existing row journal can seed only
an identical selection, as before.

## Input contract

Provide a JSON object with schema `prismaquant.campaign_row_work.v1`:

```json
{
  "schema": "prismaquant.campaign_row_work.v1",
  "census_sha256": "SHA256_OF_EXACT_CENSUS_BYTES",
  "campaign_argv": ["EXACT_SHARED_CAMPAIGN_ARGUMENTS_FROM_SPEC"],
  "startup_seconds": 20.0,
  "max_startup_fraction": 0.4,
  "gpu_slots": 2,
  "group_pricing_seconds": {
    "u:layer0.down_proj": 10.0,
    "g:layer0.gate_up": 20.0
  },
  "evidence": {
    "startup": "IMMUTABLE_STARTUP_MEASUREMENT_REFERENCE",
    "pricing": "IMMUTABLE_GROUP_PRICING_MEASUREMENT_REFERENCE"
  }
}
```

The numbers above are an illustrative, synthetic example, not measurements.
`group_pricing_seconds` must cover every non-`s:` census group exactly once;
do not include routed stacks. Each positive, finite value predicts the entire
group's pricing under the exact shared campaign arguments, including its
anchors and adaptive rounds. `startup_seconds` is positive and finite.
`max_startup_fraction` is an explicit objective strictly between zero and one.
`gpu_slots` is a positive integer describing the available measurement envelope,
not a placement instruction. Evidence references must be nonempty strings;
the planner records them, but does not verify or qualify measurements.

Bind predictions to representative runs on the same source, calibration,
sequence length, rate/family settings, cache policy, runtime and hardware.
Use an in-process profiler and Netdata series on both boxes before and after.
Record immutable receipts/profiles in `evidence`. Do not treat a routed-row
startup measurement as a measured dense-row prediction. Refresh predictions
when startup or pricing behavior changes. The planner checks census and argv
identity; those checks alone do not establish measurement quality or freshness.

## Packing and admission

The minimum predicted pricing per dense row is
`startup_seconds * (1 - max_startup_fraction) / max_startup_fraction`.
Start with the number of rows the total dense work can support at this bound.
When this allows at least one full wave, round down to a multiple of the
supplied GPU slots. Assign whole groups longest-first to the least-loaded row,
using group keys and bin indices to break ties. If a row misses the bound,
reduce the row count and repeat. Refuse if even a single dense row cannot meet
the objective. This is a deterministic packing heuristic, not an optimal
scheduler or a measured speedup.

Routed stacks remain single-group rows. Row IDs use the first group's index in
the original sorted census, so routed row IDs, selection bytes, action fields
and data-manifest bytes do not change. Dense selections sort their group keys;
fused members remain together. Plans record the profile's exact SHA-256 and
validated contents, plus each packed row's predicted pricing and startup share.

`_row_memory_gb` still derives the packed row's full demand, and
`partition_rows_by_fit` still declines rows that do not fit. Packing never
lowers a demand to force co-residency and does not split a declined bundle.
PrismaBuild alone admits, places, retries and balances the published rows.

## Validation boundary

CPU tests establish deterministic packing, complete whole-group coverage,
predicted-share bounds, routed-byte preservation and fail-closed inputs.
They use synthetic timing inputs. They do **not** establish speed, GPU load,
wire identity or identical measured cost rows. Before using this option for a
production campaign, complete #1750's approved dense-set A/B: identical
calibration/settings, before/after profiles and Netdata, GPU power against the
140 W GB10 envelope, work per joule, whole-set wall time, and exact wire plus
non-timing cost/anchor equivalence. The CPU slice (#1754) does not close #1750.
