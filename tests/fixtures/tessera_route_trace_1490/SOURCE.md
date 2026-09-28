# GLM-5.3 route-trace fixtures (PQ #1490)

These fixtures test the `route.trace` gate
(`prismaquant/tessera_route_trace_gate.py`) on the module names a real GLM-5.3
serve records. `generate.py` writes every file here, and `sources.json`
records each source path and its sha256.

## Sources

| file | serve | source |
|---|---|---|
| `measured/mtp-rank{0,1}.json` | U4 BAL, MTP k=1, TP2 (`BAL-2c-r5`) | `/mnt/shared/tessera-measurements/glm-pact-u4-20260927/results/BAL-2c-r5/run/2c-evidence/route-trace-rank{0,1}.json` |
| `measured/tr3-rank0.json` | U4 BAL, no spec decode, TP2 (TR3) | `/home/rob/tmp/claude-campaign-20260926/pact/u4/runs/u4-BAL-20260928T0540Z/head/route/tr3-rank0.json` |
| `measured/tr3-rank1.json` | same serve | `/mnt/shared/tessera-measurements/glm-pact-u4-20260927/results/BAL/run/route/tr3-rank1.json` |
| `config.json` | the priced side | `/mnt/shared/tessera-runs/moe/glm53-pact-balanced-20260928/body-mtp/exported/config.json` |

Both serves ran on `prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0…` (tag
`a5424378-mtpmap1`, see `prismaquant/serving_runtime_patches/glm53_mtp_mapper`)
with Tessera `3c01b0ef`, 2026-09-28.

## What is measured and what is synthetic

- **`measured/` is measured, trimmed.** The header fields the gate does not
  read (`note`, `pid`, `started_utc`, `flushed_utc`, `flushes`) are dropped.
  Every entry and every identity header field is exactly as Tessera wrote it.
  In every entry of both serves, the 29 NVFP4 routed expert stacks are
  unnamed (`module_names: []`, `unnamed_modules: 29`). Tessera before #680
  never bound their prefix.
- **`named/` is SYNTHETIC.** It holds the same traces with those 29 stacks
  named as Tessera #680 (merged 2026-09-28, `78287f73`) names them. The names
  are the BAL config's priced `TESSERA_NVFP4` `routed_moe` targets, mapped
  through the `glm5_next` profile (`served_module_name`), less any name the
  same M already carries. `modules` and `launches` keep their measured
  values, and `unnamed_modules` and `dispatches_without_prefix` go to zero.
  No other field changes, and
  `test_the_named_fixture_differs_from_the_measured_serve_only_in_the_filled_names`
  checks that.
- **`config.json` is trimmed** to what the gate and the profile read:
  `model_type`, `architectures`, `text_config` (`model_type`,
  `num_hidden_layers`, `num_nextn_predict_layers`) and the whole
  `quantization_config`.

The synthetic names come from the profile under test, so they cannot test the
map on their own. The 104 names the serve DID write (the body, and the draft
as `model.layers.45.mlp.experts`) can, and
`test_every_name_the_serve_wrote_is_a_priced_target_in_the_profiles_namespace`
checks them against the map. Replace `named/` with a measured post-#680 serve
when one exists.

## Regenerate

From the repository root, with the PQ test venv:

```
python tests/fixtures/tessera_route_trace_1490/generate.py \
  --config /mnt/shared/tessera-runs/moe/glm53-pact-balanced-20260928/body-mtp/exported/config.json \
  --mtp-rank0 /mnt/shared/tessera-measurements/glm-pact-u4-20260927/results/BAL-2c-r5/run/2c-evidence/route-trace-rank0.json \
  --mtp-rank1 /mnt/shared/tessera-measurements/glm-pact-u4-20260927/results/BAL-2c-r5/run/2c-evidence/route-trace-rank1.json \
  --tr3-rank0 /home/rob/tmp/claude-campaign-20260926/pact/u4/runs/u4-BAL-20260928T0540Z/head/route/tr3-rank0.json \
  --tr3-rank1 /mnt/shared/tessera-measurements/glm-pact-u4-20260927/results/BAL/run/route/tr3-rank1.json \
  --out-dir tests/fixtures/tessera_route_trace_1490
```
