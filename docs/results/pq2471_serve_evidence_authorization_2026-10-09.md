# GLM T-8 serve evidence and GPU authorization for PQ #2459

Refs #2459. Closes #2471. This record is the coordinator evidence and
written authorization that #2459 holds GPU qualification on. No GPU
action for #2459 may start before this record lands. Tessera #1095
supplies the separate producer-evidence prerequisite for #2459.

## Serve evidence

The GLM-5.3 T8R release serve runs Tessera
`83460680ed84e33c82eb62b31345381cc151aa58` (Tessera master merge of
#725, the loader fix for the T8R TP2 load). The pin was code-only:
contract v45, digest `0869f326…`, pin schema v2. The serve passes
`--compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}'`
with `VLLM_USE_BREAKABLE_CUDAGRAPH=0` (tessera#774). It runs at TP 2
on image
`localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`
with vLLM `0.30.1rc1.dev336+gaf5b4857e.d20260929` and Torch
`2.13.0+cu130`. Measured: L8192 c1 prefill +6.4%, eager TR3 panel
bit-identical (`0.027885896312391557`, 25 windows). Limits: decode
under FULL graphs past `max_model_len` 2048 is not eager-equivalent,
and the lane TP 1 reference script cannot serve the GLM-5.3 artifact.
The A8 window run `u4-A8-20260928T1809Z` (2026-09-28, GLM-5.3 uniform
T-8, MTP artifact `a8/body-mtp-v39/exported-r2`) wrote the
`tests/fixtures/tessera_route_trace_union/` traces (tp2, sm_121).

## Live pin (unchanged by this record)

This authorization preserves the live v2 pin and admission behavior.
It moves no pin, constant, reviewed answer, or frozen legal-domain state.

- Tessera commit: `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb`
- Contract: v60, lane schema v11
- Contract SHA-256:
  `ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e`
- Pin schema: `prismaquant.tessera_serving_runtime_pin.v2`
- Producer and serving commits remain equal. The serving-source
  constant remains `None`. The release label remains advisory.
- The independent export and serve gates stay in force. The
  `cell_evidence_admits` status-only rule stays in force.

## Authorized cell scope

Every target cell is `device_qualified`, requires plugin `tessera`,
and runs eager only. Each names its permitted stock vLLM runtime
image and fixture scope. The closed-world TP ceiling is 2 for every
family (`TESSERA_BF16_K1`, `TESSERA_E2M1_K2`, `TESSERA_E4M3_K1`),
citing the served TP2 receipt `glm53_a4_stub_tp2_sm121`.

### Image A: f8dbe1a0 (12 cells)

`localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`,
vLLM `0.28.1rc1.dev397+gfd4a15126.d20260904`, Torch `2.13.0+cu130`,
execution mode eager, residency resident.

- tessera_bf16_k1_dense_sm121_batch_resident
- tessera_bf16_k1_dense_sm121_decode_resident
- tessera_bf16_k1_routed_moe_sm121_batch_resident
- tessera_bf16_k1_routed_moe_sm121_decode_resident
- tessera_e2m1_k2_dense_sm121_batch_resident
- tessera_e2m1_k2_dense_sm121_decode_resident
- tessera_e2m1_k2_routed_moe_sm121_batch_resident
- tessera_e2m1_k2_routed_moe_sm121_decode_resident
- tessera_e4m3_k1_dense_sm121_batch_resident
- tessera_e4m3_k1_dense_sm121_decode_resident
- tessera_e4m3_k1_routed_moe_sm121_batch_resident
- tessera_e4m3_k1_routed_moe_sm121_decode_resident

### Image B: 5be13705 (8 cells, T-8 serve image)

`localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`,
vLLM `0.30.1rc1.dev336+gaf5b4857e.d20260929`, Torch `2.13.0+cu130`,
execution mode eager, residency resident.

- tessera_bf16_k1_dense_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e
- tessera_bf16_k1_dense_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e
- tessera_bf16_k1_routed_moe_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e
- tessera_bf16_k1_routed_moe_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e
- tessera_e4m3_k1_dense_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e
- tessera_e4m3_k1_dense_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e
- tessera_e4m3_k1_routed_moe_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e
- tessera_e4m3_k1_routed_moe_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e

### Image C: vanilla vLLM (2 cells)

`vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`,
vLLM `0.28.0`, Torch `2.13.0+cu130`, execution mode eager, residency
resident and streamed (`TESSERA_SERVE_MODE=resident|streamed`).

- tessera_e4m3_k1_dense_sm121_batch
- tessera_e4m3_k1_dense_sm121_decode

## Excluded scopes

The authorization covers none of these. Cells with absent or
incompatible evidence stay outside it.

- Platforms gfx1151 (Strix Halo) and gfx1201 (RDNA4): no sm_121
  qualification transfers to them. gfx1151 ships no cell. The two
  gfx1201 BF16 dense cells from contract v24 are not in the live
  v60 reviewed answer and are not authorized.
- Execution mode compiled: all 22 live cells run eager only.
- Residency streamed: authorized only on the two vanilla dense E4M3
  cells that publish it. The other 20 cells are resident only.
- Families, structures, or rungs outside the live reviewed answer.
- CPU fixtures: no cell qualifies from CPU fixtures.
- Any Tessera commit or contract digest other than the live pin.

## Coordinator authorization

The coordinator authorizes GPU qualification for #2459 in writing,
subject to every limit below. This authorization does not qualify
any cell or artifact. It records permission to run, not a result.

- Execute through PrismaBuild only. Use priority 0.
- Bound each GPU action to 30 minutes or less.
- Require a D38 CPU dry run for every new or changed GPU entry
  point before its GPU action. D38: the GLM-5.3 serve runs a
  patched runtime (`prismaquant/serving_runtime_patches/`,
  `serving_runtime_patch_set.py`); a patched runtime is not an
  attested one, so each entry point proves its path on CPU first.
- Require `runtime.tessera_commit` and
  `runtime.serving_source_sha256` on every target cell the packet
  qualifies. Compute the digest with
  `tessera.serving.source_identity.serving_source_sha256` under
  `tessera.package_source.v1`, not the provisioner checksum.
- Require the same qualified source digest in every trace rank.
- Publish the cell coverage matrix, raw traces, logs, action keys,
  and verified CAS receipts. Record failures and excluded scopes.
- Tessera #1095 remains a separate producer-evidence prerequisite.
  Resume the parent plan only after its applicable prerequisites pass.
- Equivalence: compare the live v2 admission results, route
  verdicts, and legal-domain projection before and after the
  evidence PR. Require identical results.
