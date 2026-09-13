# Native receipt consumption on a real producer receipt — 2026-09-13 (#267)

`prismaquant/native_operator_panel.py` has had a synthetic boundary suite since
2026-09-07. `docs/research/native_panel_267_2026-09-07.md` ends by recording
that fresh GPU qualification was still pending. This is that qualification,
consumer side.

## What ran

Tessera produced one `tessera.native_dense_operator_receipt.v1` receipt on
sparky (GB10 sm121) from retained artifact bytes: LFM2.5-8B-A1B
`model.layers.0.feed_forward.w1`, 7,168 x 2,048, `TESSERA_BF16_K1_R1792`,
eager/resident/TP1, both phases numerically gated before any timing. Producer
side: RobTand/tessera#473, PB action
`b8adfd50548a25dc24bf4a6617d49c40a6c6024d6d55351e43c9d979b3a2df50`.

`experiments/pq267_native_receipt_qualification.py` feeds that receipt, the
panel the producer froze and the raw CUPTI trace to the existing
`consume_native_receipt`. It adds no gate. It writes the observation the
consumer returns, plus the obligations a
`prismaquant.measured_runtime_prices.v2` native row still owes beyond it.

Retained inputs, by SHA256 of the file bytes, under
`/mnt/shared/tessera-measurements/pq267-native-20260913/lfm-r1792-01/`:

- `native-receipt.json` — `e1d5376da439d82195885d36623a8a3397cc9603098fe91e736d660d122b1c5a`
- `native-receipt.json.panel.json` — `87f86bfa5f27d5d1448e89c78e8e91aa17fc4773e5b0b49d084ebda837c546d9`
- `native-receipt.json.memory.json` — `f980a2c34ff78601f68efe572bd1d5884b03779f600af66eda2594541992dd1c`

## What the consumer admitted

`prismaquant.native_dense_observation.v1`, `status = operator_evidence`,
`runtime_table_admissible = false`, `unknown =
["fixed_and_full_model_resources"]`. The `native_operator_scratch` complete
path was taken for both phases, so `peak_scratch_bytes` is non-null
throughout — the incomplete-ledger branch was not exercised by this receipt.

| Phase | median single-apply (ms) | peak scratch (B) | input (B) | output (B) |
|---|---|---|---|---|
| prefill (m=512) | 0.3910079896450043 | 36,700,160 | 2,097,152 | 7,340,032 |
| decode (m=1) | 0.17902400344610214 | 71,680 | 4,096 | 14,336 |

`resident_bytes` 29,388,800; `serialized_unit_bytes` 12,892,880.

Report:
`/mnt/shared/tessera-measurements/pq267-native-20260913/lfm-r1792-01/pq267-observation.v2.json`.

## What a v2 table row still owes

Read against the "Frontier sweep" obligations in
`docs/design/joint_aura_runtime_allocation.md` and enforced by
`runtime_provenance.admit_native_rows`:

Settled by this receipt:

- both phases measured, so the row cannot refuse as "lacks a complete measured
  phase";
- a non-null `peak_scratch_bytes` per phase, so the row cannot refuse as "has
  an incomplete resource ledger".

Values a table must then match, reported rather than assumed: `prompt_tokens`
512, `batch_size` 1, `source_sha256`
`b10839290c8232ce4c5bd075b93df94f032a4075de885d43cbe0ee15afd8caac`,
`calibration_sha256`
`c6d55c9789fd6486e09f77ad8e915f2ce49ae76fdfcf0809210f4a27961b732a`,
`runtime_sha256`
`9011cdd6011d2edb4b6259e64b5e95338ec45f2c34450d69223ab8146f6843a3`.

Still owed, and each is someone else's half:

1. **Joint currency.** The panel's `cost_sha256` is a producer-declared
   fixture, not a `prismaquant.joint_aura.operator.v1` row. Tessera froze this
   panel because no joint row exists for the artifact: the joint capture for it
   failed forward parity, and its replacement needs the regenerated canonical
   captures this repository's own correction note requires
   (`native_panel_267_2026-09-07.md`). Until a joint row exists, no v2 row may
   cite this panel. The consumer is not the blocker; the cost row is.
2. **Fixed resources.** `admit_fixed_resources` runs immediately after
   `admit_native_rows` inside `admit_runtime_provenance`, so a table load
   refuses whenever the fixed-resource half is missing, however clean the
   native rows are. At `tessera.full_engine_resource_report.v1` the report has
   no timing partition at all, so a table declaring a nonzero fixed
   `prefill_ms` is refused by name. That is #420's producer half, tracked with
   #237.

So the precise state of the end-to-end goal: **native rows are admissible from
a real receipt once a joint cost row exists; a v2 table still will not load,
and the refusal is the fixed-resource one, not a native-row one.**

## Receipts

| Evidence | PB action | Result |
|---|---|---|
| Consume the real receipt, report obligations | `bf54fea78b06873574b6a66150a0e025aabebb999c7776d5564c067e4b4ee770` | exit 0, `operator_evidence`, `runtime_table_admissible` false |
| First consumption, before the obligations block | `d7fb0af316037883ad32c8f2bea8205a2b9c66f0f1034e784e072499f08c28e4` | exit 0, same observation |

Done records are at
`/mnt/shared/prismabuild-fleet/pb-queue/done/<action>.json`. Both ran CPU-only
on dl380g10 with `/home/rob/venvs/pq-cpu312/bin/python`; no GPU, no model, no
serving runtime.

## Boundaries

No pin, default, format menu entry or serving gate changed. The timings belong
to one dense operator in isolation and are not a served latency. The reference
tensors this receipt was measured against were rendered before the
no-init/rotary correction was understood, so their agreement is a parity fact
about fixed bytes and carries no render-quality claim.
