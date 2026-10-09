# GLM T-8 serve evidence and GPU authorization for PQ #2459

Refs #2459. Closes #2471. This record is the coordinator evidence and
written authorization that #2459 holds GPU qualification on. No GPU
action for #2459 may start before this record lands. The hold stays
until this record merges. Tessera #1095 supplies the separate
producer-evidence prerequisite for #2459.

## Serve evidence

The measured run is window `u4-R1-20261001T0058Z`, arm A8SESHMN. It
executed 2026-10-01 (client start 02:21:36 UTC). It serves the A8S
artifact `/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported`
(config.json sha256
`3f5c2c7381aae1c02d486c645ec6015cd1a60eb41faa5686541a15f523d79898`,
index sha256
`2990e8c051fac609e913d8619a1dea3bfda49fe69bfaf474a8ca802bdf8b9c84`).
It runs Tessera `bb088715` with `TESSERA_FUSED_E4M3_MMA=e4m3` on image
`localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`
(vLLM `0.30.1rc1.dev336+gaf5b4857e.d20260929`, Torch `2.13.0+cu130`).
Topology is TP 2 over RoCE. Serve settings are `max_num_seqs 4`,
`max_model_len 8448`, `max_num_batched_tokens 2048`, MTP off. The serve
passes `--compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}'`
with `VLLM_USE_BREAKABLE_CUDAGRAPH=0` (tessera#774). The control arm is
A8SESH, window `u4-A8SESH-20260930T2132Z` (2026-09-30), without
`"mode":"NONE"`. Nothing else changes between the arms.

Measured against the control arm: L8192 c1 prefill 1558.5 to 1657.8
tok/s (+6.4%), TTFT 5256.3 to 4941.6 ms (-6.0%). The full panel is
L512, L2048 and L8192 at c1 and c4. L512 c4 TTFT rose while its
prefill rate rose 18.7%; that cell stays unresolved. The TR3 panel is
unchanged: full-vocabulary KL `0.027885896312391557` over 25 windows
(51,175 positions), with identical domain means in both arms. The TR3
serve is eager in both arms, so the panel shows the scorer did not
move. It does not score the graph serve.

Receipts for this run: runtime binding file
`full-vocabulary-kl.runtime-5fbf983a927983c15973e67ae0b9745802322af9d2c7c454be708172b1051d70.json`
(sha256 `5fbf983a927983c15973e67ae0b9745802322af9d2c7c454be708172b1051d70`,
serve image bound inside); route traces `tr3-rank0.json` and
`tr3-rank1.json` (`torch.distributed`, world size 2, `sm_121`);
`route/verdict-tr3.json` (exact-module qualified: 56 modules, 14 dense
plus 42 routed MoE, on `TESSERA_FP8` contracts; the layer-45 draft
target is removed as diagnostic); speed files `speed/host-L512.json`,
`speed/host-L2048.json`, `speed/host-L8192.json` with
`speed/summary.json`; measurement note
`docs/measurements/2026-10-01-glm-release-serve-mode-none.md`
(2026-10-01) in the Tessera checkout; tessera#774. Raw data paths are
`/mnt/shared/tessera-measurements/glm-pact-u4-20260927/results/A8SESHMN-nightly-20260930/run`
and `.../A8SESH-nightly-20260930/run`.

Limits of this evidence: decode under FULL graphs past `max_model_len`
2048 is not eager-equivalent (tessera#702 cause 2); the release serve
runs 8448. The lane TP 1 reference script cannot serve the GLM-5.3
artifact. The MTP drafter under graphs is open (tessera#695). The
measured Tessera commit `bb088715` predates the live pin below; the
recipe evidence binds the image, flags, topology and artifact class,
not the serving commit. Each qualification packet therefore records
its own serving commit and code digest.

## Live pin (unchanged by this record)

This authorization preserves the live v2 pin and admission behavior.
It moves no pin, constant, reviewed answer, or frozen legal-domain
state. The identities below are historical evidence, not a gate
against future runtime identities.

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

The coordinator authorizes 8 cells. Each is `device_qualified`,
requires plugin `tessera`, runs eager only, and names its permitted
stock vLLM runtime image as `repository@sha256`. Each row gives the
regime, the covered q256 rungs, and the permitted TP sizes. TP 2 rests
on the served TP2 receipt `glm53_a4_stub_tp2_sm121` and the R1 run at
world size 2. TP above 2 is refused by the closed-world ceiling.

Common runtime for all 8 authorized cells:
`localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`,
vLLM `0.30.1rc1.dev336+gaf5b4857e.d20260929`, Torch `2.13.0+cu130`,
execution mode eager, residency resident
(`TESSERA_SERVE_MODE=resident`), platform `sm_121`.

- `tessera_bf16_k1_dense_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e`
  | batch | rungs q256 832, 880, 960, 1024, 1088 | TP 1, 2
- `tessera_bf16_k1_dense_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e`
  | decode | rungs q256 832, 880, 960, 1024, 1088 | TP 1, 2
- `tessera_bf16_k1_routed_moe_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e`
  | batch | rungs q256 1024 | TP 1, 2
- `tessera_bf16_k1_routed_moe_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e`
  | decode | rungs q256 1024 | TP 1, 2
- `tessera_e4m3_k1_dense_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e`
  | batch | rungs q256 832, 960, 1024, 1088 | TP 1, 2
- `tessera_e4m3_k1_dense_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e`
  | decode | rungs q256 832, 960, 1024, 1088 | TP 1, 2
- `tessera_e4m3_k1_routed_moe_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e`
  | batch | rungs q256 896, 928, 1024, 1088 | TP 1, 2
- `tessera_e4m3_k1_routed_moe_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e`
  | decode | rungs q256 896, 928, 1024, 1088 | TP 1, 2

Fixture scope for each authorized cell: fresh served traces on the
named image, in the cell regime and residency, at the listed rungs, at
TP 1 or TP 2. The contract publishes covered rungs, not tensor shapes;
shapes come from the served artifact at run time, and the packet
records them per trace. All 8 cells carry evidence grade `route_only`
with smoke `not_recorded`, which the status-only gate admits. No
pre-existing per-cell served fixture exists; the qualification runs
produce the fixtures. Input artifact for qualification is the A8S
release export named above, or a newer reviewed export named in the
packet.

## Excluded scopes

The authorization covers none of these. Cells with absent or
incompatible evidence stay outside it.

- Image `f8dbe1a0`
  (`localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`):
  no measured serve runs on this image. Excluded cells:
  `tessera_bf16_k1_dense_sm121_batch_resident`,
  `tessera_bf16_k1_dense_sm121_decode_resident`,
  `tessera_bf16_k1_routed_moe_sm121_batch_resident`,
  `tessera_bf16_k1_routed_moe_sm121_decode_resident`,
  `tessera_e2m1_k2_dense_sm121_batch_resident`,
  `tessera_e2m1_k2_dense_sm121_decode_resident`,
  `tessera_e2m1_k2_routed_moe_sm121_batch_resident`,
  `tessera_e2m1_k2_routed_moe_sm121_decode_resident`,
  `tessera_e4m3_k1_dense_sm121_batch_resident`,
  `tessera_e4m3_k1_dense_sm121_decode_resident`,
  `tessera_e4m3_k1_routed_moe_sm121_batch_resident`,
  `tessera_e4m3_k1_routed_moe_sm121_decode_resident`.
  The whole `TESSERA_E2M1_K2` family is therefore outside this
  authorization.
- Vanilla vLLM image
  (`vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`):
  no measured serve runs on this image. Excluded cells:
  `tessera_e4m3_k1_dense_sm121_batch`,
  `tessera_e4m3_k1_dense_sm121_decode`.
- Platforms gfx1151 (Strix Halo) and gfx1201 (RDNA4): no sm_121
  qualification transfers to them. gfx1151 ships no cell. The two
  gfx1201 BF16 dense cells from contract v24 are not in the live
  v60 reviewed answer and are not authorized.
- Execution mode compiled: all live cells run eager only.
- Residency streamed: no authorized cell publishes it. The two
  vanilla dense E4M3 cells publish streamed, but they are excluded
  above for lack of serve evidence.
- TP above 2: refused by the closed-world ceiling.
- CPU fixtures: no cell qualifies from CPU fixtures.
- Synthetic fixtures: `tests/fixtures/tessera_route_trace_union/`
  carries synthesized speculative entries for layer 45 and empty real
  speculative traces (see its `PROVENANCE.md`). Those files support
  CPU unit tests only and supply no qualification evidence for any
  cell. The A8 window run `u4-A8-20260928T1809Z` (2026-09-28, GLM-5.3
  uniform T-8, MTP artifact `a8/body-mtp-v39/exported-r2`) is a
  separate fixture source, not the release serve.
- Families, structures, or rungs outside the live reviewed answer.
- Any Tessera commit or contract digest other than the live pin.

## Coordinator authorization

The coordinator authorizes GPU qualification for #2459 in writing,
for the 8 cells named above only, subject to every limit below. This
authorization does not qualify any cell or artifact. It records
permission to run, not a result.

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
