# GLM-5.3 route-trace fixtures (PQ #1490)

These fixtures test the `route.trace` gate
(`prismaquant/tessera_route_trace_gate.py`) on the module names real GLM-5.3
serves recorded. `generate.py` writes every file here, and `sources.json`
records each source path and its sha256.

## Sources

All three serves priced the same artifact, the BAL export
(`/mnt/shared/tessera-runs/moe/glm53-pact-balanced-20260928/body-mtp/exported/config.json`),
on `prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`
(tag `a5424378-mtpmap1`, see `prismaquant/serving_runtime_patches/glm53_mtp_mapper`), TP2, 2026-09-28.

| file | serve | source | sha256 |
|---|---|---|---|
| `measured/mtp-r6-rank0.json` | MTP k=1, run `u4-BAL-20260928T0540Z-2c-r6-2c`, Tessera `f18f08b5` (`3c01b0ef` + #682 + #680) | `/mnt/shared/tessera-measurements/glm-pact-u4-20260927/results/BAL-2c-r6/run/2c-evidence/route-trace-rank0.json` | `1fdd254e88d069fbd2613670617faf36bb8a900616de4a388a180b7577f1c8f3` |
| `measured/mtp-r6-rank1.json` | same serve | `…/BAL-2c-r6/run/2c-evidence/route-trace-rank1.json` | `565018197d5ef3bfda4af583936d9fa6b44107772650494887bcb50caa6f06a2` |
| `measured/mtp-r5-rank{0,1}.json` | MTP k=1, run `u4-BAL-20260928T0540Z-2c-r5-2c`, Tessera `3c01b0ef` (before #680) | `…/BAL-2c-r5/run/2c-evidence/route-trace-rank{0,1}.json` | `sources.json` |
| `measured/tr3-rank0.json` | no spec decode (TR3), before #680 | `/home/rob/tmp/claude-campaign-20260926/pact/u4/runs/u4-BAL-20260928T0540Z/head/route/tr3-rank0.json` | `sources.json` |
| `measured/tr3-rank1.json` | same serve | `/mnt/shared/tessera-measurements/glm-pact-u4-20260927/results/BAL/run/route/tr3-rank1.json` | `sources.json` |

The r6 census receipt is VALID: `receipt-validation.json` in the same directory
reports 98 checks and 0 failed, and the census reports verdict `served` with no
problems. The r5 census was REFUSED, so r5 is used here only as the
unnamed-stack case.

## What is measured and what is synthetic

- **`measured/` is measured, trimmed.** The header fields the gate does not
  read (`note`, `pid`, `started_utc`, `flushed_utc`, `flushes`) are dropped.
  Every entry and every identity header field is exactly as Tessera wrote it.
  - r6 names all 133 priced modules at every M on both ranks, with 0 unnamed.
    The draft experts trace as the bare `model.layers.45.mlp.experts`.
  - r5 and TR3 predate Tessera #680: the 29 NVFP4 routed expert stacks are
    unnamed (`module_names: []`, `unnamed_modules: 29`) in every M.
- **`named/tr3-rank{0,1}.json` is SYNTHETIC.** It is the TR3 serve with those
  29 stacks named: the BAL config's priced `TESSERA_NVFP4` `routed_moe`
  targets, mapped through the `glm5_next` profile (`served_module_name`).
  `modules` and `launches` keep their measured values, and `unnamed_modules`
  and `dispatches_without_prefix` go to zero. It is kept only because no
  post-#680 non-speculative serve has been traced, and it is the one input
  for the non-spec body-scope criterion. r6 shows the rule it applies is the
  one the serve uses: every r6 NVFP4 entry names exactly those 29 modules
  (`test_r6_names_the_nvfp4_stacks_exactly_as_the_synthetic_tr3_rule_does`).
  Replace it with a measured post-#680 TR3 trace when one exists.
- **`config.json` is trimmed** to what the gate and the profile read:
  `model_type`, `architectures`, `text_config` (`model_type`,
  `num_hidden_layers`, `num_nextn_predict_layers`) and the whole
  `quantization_config`.

## Regenerate

From the repository root, with the PQ test venv:

```
R=/mnt/shared/tessera-measurements/glm-pact-u4-20260927/results
python tests/fixtures/tessera_route_trace_1490/generate.py \
  --config /mnt/shared/tessera-runs/moe/glm53-pact-balanced-20260928/body-mtp/exported/config.json \
  --mtp-r6-rank0 $R/BAL-2c-r6/run/2c-evidence/route-trace-rank0.json \
  --mtp-r6-rank1 $R/BAL-2c-r6/run/2c-evidence/route-trace-rank1.json \
  --mtp-r5-rank0 $R/BAL-2c-r5/run/2c-evidence/route-trace-rank0.json \
  --mtp-r5-rank1 $R/BAL-2c-r5/run/2c-evidence/route-trace-rank1.json \
  --tr3-rank0 /home/rob/tmp/claude-campaign-20260926/pact/u4/runs/u4-BAL-20260928T0540Z/head/route/tr3-rank0.json \
  --tr3-rank1 $R/BAL/run/route/tr3-rank1.json \
  --out-dir tests/fixtures/tessera_route_trace_1490
```
