# Routed A4 executed activation-scale grouping (PQ #624)

**Status: DESIGN.** The narrow export guard in the section "What the tree does
today" shipped with #626. Everything else here is design: nothing in the tree
rescores a routed cell, and nothing claims a qualified grouped artifact.

## What is measured, and its scope

The campaign prices one static NVFP4 `input_global_scale` G per unit. On the
Tessera routed NVFP4 (`TESSERA_E2M1_K2`) wire, that means one G per expert
projection. The routed stage does not execute one G per expert:

- Tessera #507 `src/tessera/serving/nvfp4_moe_route.py:498-499` inverts each
  expert's scalar into `w13_input_scale` / `w2_input_scale`, and `:516` passes
  them to `convert_to_nvfp4_moe_kernel_format`.
- The clamped-auto probe on the Spark image selects `FLASHINFER_CUTLASS` /
  `FlashInferExperts` (receipt
  `experiments/results/nvfp4_moe_oracle_probe_spark_a5424378.json`, recorded
  by Tessera commit `b06a4923d`;
  vLLM `0.28.1rc1.dev397+gfd4a15126`, torch 2.13.0+cu130, capability 12.1).
- vLLM `utils/flashinfer_fp4_moe.py:371-380`
  (`prepare_nvfp4_moe_layer_for_fi_or_cutlass`) takes the shared-scale branch.
  `utils/quant_utils.py:82` (`amax_for_moe_activation_quant`) reduces it to one
  `a_scale.max()` per `(module, stage)`. The EMULATION branch
  (`fused_moe/oracle/nvfp4.py:479-494`) does the same.

The executed G is therefore `min_e G_e` for w13 (gate and up of every expert in
the layer) and, separately, for w2 (down). Row-0045 layer-10 `down_proj` has
288 scales with 185 distinct values; the widest group spread in the census is
114x (layer 23 `down_proj`). Evidence:
`/home/rob/dq-runs/glm-campaign-takeover-20260913/codex-takeover/a4-scales/EVIDENCE-a4-scales.md`.

Scope: this is one backend on one image, read from source plus one probe. It is
not a runtime-published table, so no gate reads it (principle 14). The direction
and size of the dloss change are not measured.

## What the tree does today (#626)

- `nvfp4_activation_contract.routed_static_scale_grouping` is the one producer
  of the declaration. It returns `per_unit.v1` for a static-scale set that
  contains a per-expert routed projection (`<parent>.experts.<e>.<leaf>`), and
  `None` for dense or native packed units. It raises for a per-expert name that
  `routed_moe_stage` cannot resolve to a `(module, stage)` group.
- `tessera_menu.priced_static_scales` (allocator and sampled proposal) and
  `tessera_materialization._static_scale_grouping` (selected-wire completion,
  scoped to static-contract selections) stamp through it.
- `tessera_export_lane.require_priced_export_inputs` refuses an ungroupable
  per-expert static-contract unit, and an absent, malformed or non-`per_unit.v1`
  declaration.
- An accepted export writes `tessera_activation_scale_grouping`
  (`qualified: false`) on the build anchor, beside the closed `priced_inputs`
  block that Tessera's exporter reads.

A routed A4 export can proceed today. Its card says the A-side prices are per
unit and not qualified.

## The real fix: rescore routed cells under the executed grouping

DESIGN, not implemented. Line numbers are against `origin/main` `e27fb88e75`.

### Precondition: the route and its attestation arrive together

The collapse is a fact about Tessera #507, which is not in the pin. At pin
`7dbbacbd` there is no `nvfp4_moe_route.py`, and
`src/tessera/serving/runtime_contract.json` lists only
`tessera_e2m1_k2_dense_sm121_{decode,batch}` cells: there is no routed E2M1_K2
cell. A rescore therefore prices "as #507 would execute". No export gate may
accept grouped rows until a pin bump carries both the routed route and a
machine-readable grouping table (principle 14).

### Inputs

- **Per-unit calibration amax.** `census["max_abs"]` through `census_max_abs`
  (`tessera_campaign.py:4035`), which refuses disagreement with observed
  maxima. The GLM-5.3 census has 36,288 per-expert entries.
- **Group membership.** `routed_expert_scale_group` maps each per-expert name
  to `(group_key, module, stage)`, with w13 = gate+up and w2 = down. Stack
  uniformity (whole routed stack on one format) makes membership the entire
  stack, independent of the assignment.
- **Group value.** Take the maximum member amax, then call
  `input_global_scale_from_max_abs` once. That F32 value is `min_e G_e`, the
  same object the #507 loader inverts. Gate and up of one expert read the
  same routed rows, so their amax is identical, and the min over 2E w13 slots
  equals the min over experts. Fused-sibling unification composes exactly.
- **Verified wire journal.** The row `cost.pkl`, `cache/` and
  `cost.anchors.json`. The wire files are `cache/wire/<qname>__<format>.tessera`,
  and each has a journal record `{file, blob_sha256, blob_bytes, identity}`.
  `encode_tessera_units` (`tessera_render.py:1368`) never reads G, so wires
  are reused, not re-encoded.
- **Resident calibration activations** under the calibration cache's
  `inputs/`, with the same `max_act_rows` prefix the original score used.

### Where it plugs in

The seam is `_static_input_scales` (`tessera_campaign.py:3281`), not the
scoring lambda. Three consumers compare a row's G to `static_scales[qname]` by
exact equality: `_require_resumable_anchor` (`:4521`),
`_campaign_checkpoint_identity` (`:1907`) and `write_export_inputs` (`:4639`).
Swapping only the scalar in `_prepare_anchor` would produce rows the journal
refuses and a scale file that still ships per-unit values.

1. After `unify_fused_sibling_max_abs`, add a group-minimum step keyed by
   `routed_expert_scale_group`, selected by an explicit grouping argument.
   `per_unit.v1` stays the default, and the grouping is recorded beside the
   policy.
2. For each routed cell, verify its wire against the journal receipt first
   (`verify_expert_wire_record`, then `api.verify_cached_unit`), and refuse on
   mismatch.
3. Call `_prepare_anchor` (`:488`) with the group G. G enters only in
   `activation_qdq` (`:532`). Skip encoding and take the render from the
   production-cache bf16 entry, or decode the verified wire, then score it
   through `_finish_anchor` (`:562`) unchanged.

### What it must record so priced equals executed

- **Every rescored row:** `input_global_scale` = the group value, plus an
  `input_global_scale_grouping` stamp: schema, grouping id, `group_key`,
  reduction (`layer_experts_max_amax`), a digest over the full membership, the
  effective value and the policy. `CampaignAnchor` has no field for this yet.
- **Checkpoint identity:** the grouping, bound next to
  `input_global_scale_policy`, so a union of per-unit and grouped rows refuses.
- **`input_scales.safetensors`:** every member carries the group value, and
  the metadata names the grouping. The provenance `activation_static_scales.source`
  must no longer say `campaign_calibration_amax_fused_unified` for grouped rows.
- **Producer and gate:** `routed_static_scale_grouping` reads the row stamps
  (all rows in one allocation must agree). The export gate accepts the grouped
  id only when the pinned `runtime_contract.json` publishes a routed grouping
  entry: platform, family, structure `routed_moe`, backend, kernel, reduction.
- **Experts with no calibration rows:** today they get no scale, and stack
  uniformity forces the layer to BF16. The executed value for such an expert
  would still be the group G. The rescore must state this disposition, not
  inherit it silently.

### Numerics: the unary cost survives

Fable consultation (tier `fable-high`, 2026-09-14). Question: does rescoring
each expert under the group scalar add a cross-expert dependency that the
additive per-unit cost cannot carry? Outcome: no. The answer was checked
against the code cited above.

- **Exact.** The group value is a pure function of census maxima fixed before
  allocation. Membership is the whole stack, and stack uniformity means every
  expert in an A4 layer is A4, so the value does not depend on the assignment.
  The served A-side oracle is per row and per 16-element block with no
  cross-row statistic, so scoring expert e on its own routed rows under the
  group G is the executed arithmetic for those rows.
- **Approximate (already true for dense A4).** Calibration uses BF16-upstream
  activations and the `max_act_rows` prefix, and the kernel's reciprocal
  differs from F32 division.
- **Needs measurement.** Direction and size. `nvfp4_group_stored_scale` clamps
  only from above (`clamp(max=FP8_E4M3_MAX)`), so a group G up to 114x smaller
  than an expert's own drives that expert's low-amax block scales toward E4M3
  subnormal or zero. Worse dloss for those experts is plausible and is not
  asserted.

### Minimal validation

Rescore a cell whose own G already equals the group value (layer 23
`down_proj` expert 205) on the same device and row prefix, and require its
existing dloss bit-exactly. Then rescore a far cell (expert 78, ratio 0.0088),
and confirm that the checkpoint identity and `_require_resumable_anchor` refuse
a mix of old and new rows.
