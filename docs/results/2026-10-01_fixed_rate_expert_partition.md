# Fixed-rate expert partition qualification

This record closes #1314's fixed-rate planner/merge acceptance. It qualifies
one complete unsampled routed group, not a complete model campaign, a serving
lane promotion or a performance improvement. Adaptive groups remain indivisible.

## Implementation and CPU controls

PR #1808 (merge `11f884da9b693ee889a11b5ba6ac1be45cc323e8`) added opt-in
`plan --experts-per-row N` with producer-complete expert chunks, full-group menu
identity, own-subset demand/readsets and plan-derived coverage. Its 223/223 CPU
tests passed with zero skips, including positive whole-group table equivalence,
adaptive-band refusal and demand rechecking. PR #1872 (merge
`36091f95f31cdfb59016d086df8ff390cab639b5`) repaired the discovered typed
provenance reader; 147/147 tests passed with zero skips (RED: four expected
failures and 14 controls). CLI values and numeric rates did not change.

## Immutable workload

- GPU source: `11f884da9b693ee889a11b5ba6ac1be45cc323e8`.
- Producer: Tessera `b40c93cb73745097e57a1ba4cf5b9eee166c759a`, unchanged
  contract SHA `0869f326543374dbd26b75e1d736befed378280d9a5724c4f170bf398aefdbaa`.
- Model: `/mnt/shared/models/GLM-5.3-Flash-BF16`.
- Group: `s:model.language_model.layers.10.mlp.experts`, 288 experts and
  864 gate/up/down projections; pinned `--rate-band 896,896 --max-rounds 1`,
  family E4M3_K1, unchanged calibration, anchors and cache policies.
- Capture SHA: `f4bcbf408d3aa81b04c1fabd1d2d7457176a95dcccdd5ed368800de5c37e277c`.
- Source proof SHA: `543afb5077e531adef1c72d6eeaf0b0d66dbd3d4f828a1b9bea0a0657d6a1f08`.
- Owned non-release input/output root:
  `/mnt/shared/tessera-measurements/pq-rows-qualification-20260930`.

The whole-group row was compared with all 36 `--experts-per-row 8` chunks,
24 projections each. PB controlled portable GB10 placement. No live release
input or immutable GPU source was edited, and no completed GPU row was repeated.
The existing scope still declares 132 groups: this artifact prices one and
reports the other 131 as unpriced.

## Terminal and memory evidence

The whole row's PB action was
`516abece7840cd576778dbbbb0472689d298ce51dc0efb777b112e6116d8f1d5`.
Its terminal/log/CAS receipt passed inspection; actual action-cgroup peak was
27.22 GiB against the declared 69 GiB reservation. All 36 chunks had one attempt,
rc0 endings, authenticated CAS payloads and authenticated logs with 24 completed
units. Their maximum action-cgroup peak was 8,397,398,016 bytes (7.82 GiB), below
their declared 56 GiB reservations; minimum observed host availability was
55,611,908,096 bytes (51.79 GiB), above the 16 GiB floor. These are action-cgroup
telemetry, not a claim about staged weight residency or kernel utilization.
Both-box Netdata history was collected; no speed/work-per-joule claim is made.

Audit and row-to-action ledger:
`/home/rob/tmp/claude-campaign-20260926/pi/pq-rows/partition-terminal-audit.json`
and
`/home/rob/tmp/claude-campaign-20260926/pi/pq-rows/partition-action-ledger.json`.

## Canonical comparison

A raw prefix check correctly refused row-specific Hessian capture digests.
The existing authenticated H-reference union owner—not blind field removal—
then produced identical complete capture commitments:
`31051b9aa953c8067650356a7e89c28cb550c36987c9b35c46a62b15dea5977b`.
All 864 canonical non-timing prices, journal identities and non-timing anchor
fields matched. The positive completed before missing and duplicated actual
chunk controls, both of which refused.

PB CPU action:
`64a63d8ec771cc41512ee64a3cefc6b7b1e1ca1a74a06ff00ea3e66ebc2810c2`,
rc0, CAS payload 4,683 bytes,
SHA `f544a58a4bf0603ee88236bf3911b82902dcc5d85426dd12f75cefbd43fc22b5`.
Only timing fields were excluded: price `encode_seconds`,
`encode_seconds_accounting`, `encoding_batch_size`; anchor `seconds`,
`encoding_batch_size`. No score, rate, cost, identity, Hessian or anchor-value
field was excluded.

## Wire comparison

The producer's `cached_unit.verify_cached_unit` verified every actual whole
and chunk blob against matching identity/byte receipts. All 864 blobs were
byte-identical, totaling 3,190,412,160 bytes on either side. Input commitments
matched. A positive complete comparison preceded an in-memory damaged-blob
control, which the producer refused without modifying any original artifact.

PB CPU action:
`6323d10d2c6ee880239c26c7859b4f2216fc3643583e30fb5c45b1d86248237b`,
rc0, CAS payload 3,894 bytes,
SHA `80c61c4602a21e530cb10f14b67dc98a49ad6ab2fc658cc9a9e8863485ee3cfe`.
Canonical endings, CAS payload lengths/hashes and result records were inspected
independently. `full_campaign_complete` and `performance_qualified` remain false.

## Reproduction and limits

GPU plans are `plan-partition-whole` and `plan-partition-p8` under the owned root.
CPU comparison sources are snapshotted from the separate validation checkout;
its only runtime patch is the independently tested typed-band reader, preserving
the GPU source/artifacts. Commands use pin-matching x86 Python, CPU1/mem8 GiB,
native threads1, priority -10, timeout1800s and declared readsets:

```text
pbrun.py --cwd VALIDATION --tag x86 --cpus 1 --demand mem_gb=8 \
  --priority -10 --timeout-s 1800 --wait-s 2400 --data-manifest READSET \
  --env PYTHONPATH=. --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 -- PIN_PYTHON -u -m tools.pq_rows_qualification.MODULE
```

`MODULE` is `compare_canonical_union` or `compare_actual_wire`; readsets are
`canonical-union.data-manifest.json` and `actual-wire.data-manifest.json`.
The latter explicitly adds 1,728 actual wire files (controlled, nonrecursive
known-row enumeration). Logs are
`/home/rob/tmp/claude-campaign-20260926/pi/pq-rows/canonical-union-v2-comparison.log`
and `/home/rob/tmp/claude-campaign-20260926/pi/pq-rows/actual-wire-comparison.log`.

Sampling, adaptive partitions, external seeds and heterogeneous expert geometry
remain unsupported/refused by this first implementation. This evidence does
not qualify another family/rate, all 132 model groups, a shipping default,
serving performance, or #1750's separate dense-packing throughput claim.
