# tessera_route_trace_union fixtures

Cut by `generate.py` from the A8 window (GLM-5.3 uniform T-8, MTP artifact
`a8/body-mtp-v39/exported-r2`, run `u4-A8-20260928T1809Z`, 2026-09-28).

| File | Origin |
|---|---|
| `config.json` | The real A8 `config.json`, config groups cut to 5 real targets plus the MTP draft layer. |
| `nonspec-rank{0,1}.json` | REAL A8 `tr3` traces (tp2, sm_121, identity_version 1), entries cut to those modules and to M1 / M2048. Shapes, headers and module-name spellings are unchanged. |
| `spec-rank{0,1}.json` | `nonspec` plus **synthesized** entries (one per token count) for layer 45 `mlp.experts` on `TESSERA_BF16:resident` / `bf16_unquantized`, served under the draft namespace name `model.layers.45.mlp.experts` (what the pinned `glm5_next` profile's `served_module_name` returns for the MTP layer; the checkpoint target `model.language_model.layers.45.mlp.experts` is the priced spelling). Their shapes are copied from the real FP8 routed-MoE entries. No real speculative-serve trace with the draft layer dispatched exists yet: A8's 2c serve wrote none. |
| `empty-rank{0,1}.json` | The REAL header-only 2c trace A8 wrote (`rank null`, `world_size null`, `platform ""`, `rank_source "unavailable"`, `entries []`). |

The layer-45 entries are the only invented bytes. The tests that depend on
them state the assumption: the draft layer is priced BF16 and, once served,
dispatches on the BF16 contract at every token count.
