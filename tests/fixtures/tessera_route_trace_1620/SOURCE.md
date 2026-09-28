# GLM-5.3 A8 route-trace fixture (PQ #1620)

These fixtures test the `route.trace` gate
(`prismaquant/tessera_route_trace_gate.py`) on the module names and key shapes
a real GLM-5.3 serve recorded. `generate.py` writes every file here, and
`sources.json` records each source path and its sha256.

## Sources

The serve is the U4 A8 arm: the uniform Tessera-8 artifact (body E4M3, policy
`TESSERA_FP8:resident`, decoder `fp8_per_token_dynamic`) on Tessera v39
(`TS_PIN_A8=4c4ff1c2`), TP2, run `u4-A8-20260928T1809Z`, no speculative decode
(the TR3 serve).

| file | source | sha256 |
|---|---|---|
| `config.json` | `/mnt/shared/tessera-runs/moe/glm53-pact-uniform-arms-20260927/a8/body-mtp-v39/exported-r2/config.json` | `a0e206dbb207f418ae95d36b84c0f95a44f29cfcba622dd989dae8bd6eee44ad` |
| `tr3-rank0.json` | `/home/rob/tmp/claude-campaign-20260926/pact/u4/runs/u4-A8-20260928T1809Z/head/route/tr3-rank0.json` | `56b8f57be7eb8145b49342837db79849ccb97513eac911baf771949761aff7a6` |
| `tr3-rank1.json` | `/home/rob/tmp/claude-campaign-20260926/pact/u4/runs/u4-A8-20260928T1809Z/route/tr3-rank1.json` | `3f20c35232f6d2f9b532e301d4c8f9b4b5cf95d6ac5e47105b28af2f41fe7806` |

## What is measured and what is cut

Nothing here is synthetic. Every key, every module name and every contract
string is the serve's. Two things are cut, because the real traces name 132
modules on each rank and a test cannot read that:

- **Each trace entry keeps its first four module names** (in the trace's own
  sorted order). An entry that names four or fewer modules is kept whole.
  `modules` becomes the number kept. `launches` becomes the number kept times
  the entry's measured launches per module, which is constant per entry. The
  cut keeps 5 entries per token count at M=1, 2, 2048 and 2049 (20 per rank),
  18 modules per M. The cut can leave an M=2048 entry with 75 or 100 launches
  where the serve recorded a different multiple; the gate reads `launches`
  only as a positive integer.
- **The header drops** `note`, `pid`, `started_utc`, `flushed_utc` and
  `flushes`, which the gate never reads. `schema`, `identity_version`, `rank`,
  `world_size`, `rank_source`, `rank_conflict` and `platform` are the serve's.
- **`config.json` is trimmed** to what the gate and the profile read:
  `architectures`, `model_type`, the `text_config` layer counts, and
  `quantization_config` with its `config_groups` cut to the groups whose
  targets are all served by a kept module. Its `ignore` list is dropped (the
  gate does not read it). Of the 133 priced groups, 18 are kept and 115
  dropped.
- **The MTP draft group is dropped.** The A8 export prices one
  `TESSERA_BF16` `routed_moe` group at layer 45 (the MTP draft layer). This
  serve is not speculative, so the draft layer is never dispatched, and the
  full price refuses on it by design (fail-closed, PQ #1490). The fixture
  prices what the serve dispatches so its baseline agrees, and the test file
  builds every refusal from that baseline.

The serve records no `None` contract and no unnamed module
(`unnamed_modules` is 0 on every entry). What the real refusal text called
"served None" is the count line the gate itself appended to a long list; see
`tests/test_tessera_route_trace_gate_sentinel.py`.

## Regenerate

From the repository root, with the PQ test venv:

```
R=/home/rob/tmp/claude-campaign-20260926/pact/u4/runs/u4-A8-20260928T1809Z
python tests/fixtures/tessera_route_trace_1620/generate.py \
  --config /mnt/shared/tessera-runs/moe/glm53-pact-uniform-arms-20260927/a8/body-mtp-v39/exported-r2/config.json \
  --tr3-rank0 $R/head/route/tr3-rank0.json \
  --tr3-rank1 $R/route/tr3-rank1.json \
  --out-dir tests/fixtures/tessera_route_trace_1620
```
