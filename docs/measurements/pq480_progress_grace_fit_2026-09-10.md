# PQ-480 progress-grace fit — 2026-09-10

This record explains the watchdog allowances in
`tools/dispatch_tessera_campaign.py` for the completed GLM pricing workload.
It is a measurement of that workload, rather than an assertion that its
cadence applies to another campaign.

## Input and selection

The source is the retained root-action record
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-anchor-preparation-05/root-action-records-current.json`,
whose SHA-256 is
`03d6bf9f988ca7f514b37bea7d0e6001b45c44831fa7900b87162c06f5232387`.
For every `done` action it names, the extractor reads the corresponding
`/mnt/shared/prismabuild-fleet/pb-queue/done/<action-key>.json` terminal
record. It selects records whose final campaign line says `864 units, 2592
priced rungs`, counts their durable `[campaign] ... batch=` lines, and takes
`detail.elapsed_s` as elapsed wall time. The selection has 23 rows.

The committed [JSON artifact](artifacts/pq480_progress_grace_fit_2026-09-10.json)
contains every selected row ID, action key, terminal-record SHA-256, batch
count, and elapsed time. It is the exact stdout from:

```bash
python tools/derive_tessera_progress_grace.py > docs/measurements/artifacts/pq480_progress_grace_fit_2026-09-10.json
```

The command was admitted by PrismaBuild as action
`eca3426e3a1eccc5a4d8a26c3fb7964da19b439648328c5b7fc1eaf97d630878` on
`dl380g10`, returned 0, and published CAS receipt
`8f99bc54a962267d3e94cfa160ebafbf5a4dd12eb621f71e556d1faf224df8c1`.

## Fit and allowance

Ordinary least squares of elapsed seconds on committed batches gives:

| Quantity | Seconds |
| --- | ---: |
| Per committed batch | 18.672808072814316 |
| Fitted intercept | 836.1036216758985 |
| Largest absolute residual | 160.45635778753422 |

`pricing=900` therefore allows 48.20 fitted batch intervals. The fitted
intercept plus its largest observed residual is 996.56 seconds;
both `startup=3600` and `finalize=1800` exceed it. Together the phase quiet
limits are 6,300 seconds, below the 14,400-second blanket deadline that
previously terminated rows despite their committed-anchor progress.

The intercept estimates time outside pricing; it is not a direct measurement
of the longest startup, pricing or finalization gap. The allowances are
chosen margins informed by this fit, not measured worst-case bounds.

The extractor deliberately fails if retained data no longer selects exactly
23 rows, or if those rows do not vary in committed batch count. Re-run it when
the campaign changes instead of treating these values as a generic timeout.
