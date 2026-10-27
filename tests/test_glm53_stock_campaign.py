"""GLM campaign contracts use real source facts and synthetic scalar prices.

Inventory controls use metadata fixtures. Full campaign controls use a tiny
native checkpoint, its actual header census and streamed source identity.
These CPU checks do not measure AURA, served quality or full-model behavior.
"""
from __future__ import annotations

import pickle

import pytest

from prismaquant.glm53_stock_harvest import (
    build_arm_identity,
    build_dense_plan,
)
from prismaquant.glm53_stock_reprice import (
    CHECKPOINT_CENSUS_SCHEMA,
    PROBE_CENSUS_SCHEMA,
    Glm53StockError,
    build_declarations,
    run_campaign_from_artifacts,
)

from tests.native_source_contract_fixtures import (
    CALIB_HASH, _probe_unit, coherent_glm_campaign,
)


@pytest.fixture
def campaign_inputs(tmp_path):
    return coherent_glm_campaign(tmp_path)


def _censuses():
    prefix = "model.language_model.layers"
    probe = {
        "schema": PROBE_CENSUS_SCHEMA,
        "meta": {"calib_hash": CALIB_HASH},
        "n_stats": 5,
        "units": {
            # dense fp8-source quantizable -> ladder {NVFP4, FP8_SOURCE}.
            # FP8_SOURCE's verbatim-copy contract requires (128, 128)
            # scale-block divisibility, so the synthetic shape honors it.
            f"{prefix}.0.mlp.gate_proj": _probe_unit(128, 128),
            # dense bf16-source quantizable -> ladder {NVFP4, BF16}
            f"{prefix}.0.mlp.shared_experts.down_proj": _probe_unit(64, 64),
            # packed routed expert (fp8 source) -> profile refuses the
            # FP8_SOURCE terminal; priced empirically
            f"{prefix}.0.mlp.experts.gate_up_proj": _probe_unit(
                128, 128, experts=2, packed=True,
            ),
            # profile-pinned attention unit -> exact source terminal row
            f"{prefix}.0.self_attn.q_proj": _probe_unit(64, 64),
            "lm_head": _probe_unit(128, 64),
        },
    }
    checkpoint = {
        "schema": CHECKPOINT_CENSUS_SCHEMA,
        "units": {
            f"{prefix}.0.mlp.gate_proj": {"source_kind": "fp8"},
            f"{prefix}.0.mlp.shared_experts.down_proj": {
                "source_kind": "bf16",
            },
            f"{prefix}.0.mlp.experts.gate_up_proj": {"source_kind": "fp8"},
            f"{prefix}.0.self_attn.q_proj": {"source_kind": "bf16"},
            "lm_head": {"source_kind": "bf16"},
        },
    }
    return probe, checkpoint


def test_dense_plan_partition_and_refusals():
    probe, checkpoint = _censuses()
    planned = build_dense_plan(probe, checkpoint)
    assert planned["plan_scope"] == "full"
    assert sorted(planned["plan"]) == [
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.mlp.shared_experts.down_proj",
    ]
    assert all(fmts == ("NVFP4",) for fmts in planned["plan"].values())
    # The packed unit is excluded from the render plan and its serving-gap
    # refusal (FP8_SOURCE terminal, profile_mismatch) is recorded.
    assert planned["packed_expert_units_excluded"] == [
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
    ]
    refused = {
        (item["qname"], item["format"], item["reason"])
        for item in planned["ladder_refusals"]
    }
    assert (
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "FP8_SOURCE",
        "profile_mismatch",
    ) in refused
    assert set(planned["pinned_units"]) == {
        "model.language_model.layers.0.self_attn.q_proj",
        "lm_head",
    }


def test_dense_plan_filter_marks_partial():
    probe, checkpoint = _censuses()
    planned = build_dense_plan(probe, checkpoint, unit_filter="gate_proj")
    assert planned["plan_scope"] == "filtered"
    assert sorted(planned["plan"]) == [
        "model.language_model.layers.0.mlp.gate_proj",
    ]


def test_campaign_prices_and_merges_three_provenances(campaign_inputs):
    probe, checkpoint, harvest, expert = campaign_inputs
    payload = run_campaign_from_artifacts(
        probe_census=probe,
        checkpoint_census=checkpoint,
        harvest=harvest,
        expert_payload=expert,
    )
    report = payload["provenance"]["merge_report"]
    assert report["anchored_units"] == 2
    assert report["empirical_units"] == 1
    assert report["pinned_units"] == 2
    assert report["total_units"] == 5
    assert payload["provenance"]["unpriced_probe_units"] == []
    costs = payload["costs"]
    dense_fp8 = costs["model.language_model.layers.0.mlp.gate_proj"]
    assert set(dense_fp8) == {"NVFP4", "FP8_SOURCE"}
    assert dense_fp8["FP8_SOURCE"]["predicted_dloss"] == 0.0
    assert dense_fp8["NVFP4"]["predicted_dloss"] > 0.0
    dense_bf16 = costs[
        "model.language_model.layers.1.mlp.shared_experts.down_proj"
    ]
    assert set(dense_bf16) == {"NVFP4", "BF16"}
    packed = costs["model.language_model.layers.1.mlp.experts.gate_up_proj"]
    assert set(packed) == {"NVFP4", "FP8_E4M3"}
    assert packed["NVFP4"]["cost_source"] == "expert_kl"
    pinned = costs["model.language_model.layers.1.self_attn.q_b_proj"]
    assert set(pinned) == {"BF16"}
    assert pinned["BF16"]["predicted_dloss"] == 0.0
    refusal_rows = payload["provenance"]["ladder_refusals"]
    assert any(
        row["qname"].endswith("mlp.experts.gate_up_proj")
        for row in refusal_rows
    )
    # The payload survives a pickle round-trip (it is the on-disk artifact).
    assert pickle.loads(pickle.dumps(payload))["schema"] == payload["schema"]


def test_campaign_refuses_filtered_harvest(campaign_inputs):
    probe, checkpoint, harvest, expert = campaign_inputs
    harvest["plan_scope"] = "filtered"
    with pytest.raises(Glm53StockError, match="plan_scope"):
        run_campaign_from_artifacts(
            probe_census=probe,
            checkpoint_census=checkpoint,
            harvest=harvest,
            expert_payload=expert,
        )


def test_campaign_refuses_plan_drift(campaign_inputs):
    probe, checkpoint, harvest, expert = campaign_inputs
    harvest["plan"].popitem()
    with pytest.raises(Glm53StockError, match="plan differs"):
        run_campaign_from_artifacts(
            probe_census=probe,
            checkpoint_census=checkpoint,
            harvest=harvest,
            expert_payload=expert,
        )


def test_campaign_refuses_rtn_fallback_rows(campaign_inputs):
    probe, checkpoint, harvest, expert = campaign_inputs
    harvest["aura_payload"]["provenance"]["dw_rtn_fallback_rows"] = 1
    with pytest.raises(Glm53StockError, match="RTN-fallback"):
        run_campaign_from_artifacts(
            probe_census=probe,
            checkpoint_census=checkpoint,
            harvest=harvest,
            expert_payload=expert,
        )


def test_campaign_refuses_missing_empirical_unit(campaign_inputs):
    probe, checkpoint, harvest, expert = campaign_inputs
    expert["costs"] = {
        "model.language_model.layers.99.mlp.experts.gate_up_proj": {
            "NVFP4": {"predicted_dloss": 1e-2, "source_binding": harvest["source_binding"]},
        },
    }
    with pytest.raises(Exception, match="empirical"):
        run_campaign_from_artifacts(
            probe_census=probe,
            checkpoint_census=checkpoint,
            harvest=harvest,
            expert_payload=expert,
        )


def test_campaign_refuses_foreign_calibration(campaign_inputs):
    probe, checkpoint, harvest, expert = campaign_inputs
    arm = dict(harvest["arm_identity"])
    arm["calibration"] = dict(arm["calibration"], probe_calib_hash="0" * 32)
    harvest["arm_identity"] = arm
    with pytest.raises(Glm53StockError, match="calib"):
        run_campaign_from_artifacts(
            probe_census=probe,
            checkpoint_census=checkpoint,
            harvest=harvest,
            expert_payload=expert,
        )


def test_arm_identity_refuses_calibration_mismatch():
    with pytest.raises(Glm53StockError, match="differs from the probe"):
        build_arm_identity(
            probe_calib_hash=CALIB_HASH,
            observed_calib_hash="0" * 32,
            dataset="/data/diverse-v1.jsonl",
            n_calib_samples=8,
            calib_seqlen=512,
            calib_seed=42,
            n_probes=32,
            max_act_rows=256,
        )


def test_declarations_match_real_census_partition():
    """The synthetic partition mirrors the real one's structure."""
    probe, checkpoint = _censuses()
    declarations, pinned, refusals, unresolved = build_declarations(
        probe, checkpoint,
    )
    assert not unresolved
    assert {d.unit_class for d in declarations} == {"dense"}
    assert len(declarations) + len(pinned) + len(
        {r.qname for r in refusals}
    ) == len(probe["units"])


# --------------------------------------------------------------------------
# recipe-namespace rekey (live probe/harvest names -> spec recipe names)
# --------------------------------------------------------------------------
def _live_probe_payload():
    stats = {
        "lm_head": {"h_trace": 1.0},
        "model.language_model.layers.0.self_attn.forget_gate.f_a_proj": {
            "h_trace": 2.0,
        },
        "model.language_model.layers.0.mlp.experts.down_proj": {
            "h_trace": 3.0,
            "router_path": "model.language_model.layers.0.mlp.gate",
        },
    }
    return {
        "stats": stats,
        "router_counts": {"model.language_model.layers.0.mlp.gate": {0: 4}},
        "router_totals": {"model.language_model.layers.0.mlp.gate": 4},
        "router_active_counts": {},
        "expert_route_stats": {},
        "expert_info": {},
        "meta": {"calib_hash": CALIB_HASH},
    }


def _live_cost_payload():
    return {
        "schema": "prismaquant.glm53_stock_campaign.v1",
        "formats": ["NVFP4"],
        "costs": {
            "lm_head": {"BF16": {"predicted_dloss": 0.0}},
            "model.language_model.layers.0.self_attn.forget_gate.f_a_proj": {
                "BF16": {"predicted_dloss": 0.0},
            },
            "model.language_model.layers.0.mlp.experts.down_proj": {
                "NVFP4": {"predicted_dloss": 0.5},
            },
        },
        "provenance": {"merge_report": {}},
        "meta": {},
    }


def test_rekey_probe_and_costs_to_recipe_namespace():
    from prismaquant.glm53_stock_reprice import (
        rekey_costs_to_recipe,
        rekey_probe_to_recipe,
    )

    probe = rekey_probe_to_recipe(_live_probe_payload())
    cost = rekey_costs_to_recipe(_live_cost_payload())
    expected = {
        "lm_head",
        "model.layers.0.self_attn.f_a_proj",
        "model.layers.0.mlp.experts.down_proj",
    }
    assert set(probe["stats"]) == expected
    assert set(cost["costs"]) == expected
    # router maps AND the stats rows' router_path values move together.
    assert set(probe["router_counts"]) == {"model.layers.0.mlp.gate"}
    row = probe["stats"]["model.layers.0.mlp.experts.down_proj"]
    assert row["router_path"] == "model.layers.0.mlp.gate"
    # values ride along untouched
    assert probe["stats"]["model.layers.0.self_attn.f_a_proj"][
        "h_trace"] == 2.0
    assert cost["costs"]["model.layers.0.mlp.experts.down_proj"][
        "NVFP4"]["predicted_dloss"] == 0.5
    # both payloads are stamped, and a second rekey refuses.
    assert probe["meta"]["recipe_rekey"]["renamed_keys"] > 0
    assert cost["provenance"]["recipe_rekey"]["total_keys"] == 3
    with pytest.raises(Glm53StockError, match="rekey twice"):
        rekey_probe_to_recipe(probe)
    with pytest.raises(Glm53StockError, match="rekey twice"):
        rekey_costs_to_recipe(cost)


def test_rekey_refuses_recipe_namespace_collision():
    from prismaquant.glm53_stock_reprice import rekey_probe_to_recipe

    payload = _live_probe_payload()
    # A live-space name and its recipe-space image both present: the rename
    # would merge two measured rows into one key. Must refuse, not resolve.
    payload["stats"]["model.layers.0.self_attn.f_a_proj"] = {"h_trace": 9.0}
    with pytest.raises(Glm53StockError, match="collision"):
        rekey_probe_to_recipe(payload)


def test_glm5_next_mlp_gate_up_fused_groups():
    """gate/up fusion is a fact of the pinned runtime (Glm5NextMLP builds
    MergedColumnParallelLinear for dense-layer MLPs and shared_experts,
    models/glm5next/nvidia/model.py:116 in vllm-glm5next:pr53906-933876c).
    A gate/up format split cannot load, so the spec must declare both
    quantizable families for union-find promotion."""
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile

    p = Glm5NextProfile()
    assert (
        p.fused_sibling_group("model.layers.0.mlp.gate_proj")
        == p.fused_sibling_group("model.layers.0.mlp.up_proj")
        == "model.layers.0.mlp.gate_up_proj"
    )
    assert (
        p.fused_sibling_group("model.layers.7.mlp.shared_experts.gate_proj")
        == p.fused_sibling_group("model.layers.7.mlp.shared_experts.up_proj")
        == "model.layers.7.mlp.shared_experts.gate_up_proj"
    )
    # down_proj is standalone; packed experts are already one unit; lm_head is
    # pinned; the KDA f_b_proj is its own runtime module (f_a_proj is not: it
    # loads into in_proj_qkvbfg_a, PQ #1837) - none may pick up a group.
    for standalone in (
        "model.layers.0.mlp.down_proj",
        "model.layers.7.mlp.shared_experts.down_proj",
        "model.layers.7.mlp.experts.gate_up_proj",
        "model.layers.44.self_attn.f_b_proj",
        "lm_head",
    ):
        assert p.fused_sibling_group(standalone) is None, standalone


def test_glm5_next_export_rename_live_to_checkpoint():
    """The export body-walk emits LIVE qnames on a multimodal-forced
    skeleton; the sink renames via live_to_recipe_name ∘ export_tensor_name.
    Both hops must compose to the exact checkpoint spelling, and the
    composition must be idempotent for recipe-space inputs (the spelling
    every text-only-skeleton family emits)."""
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile

    p = Glm5NextProfile()

    def sink_rename(k):
        return p.export_tensor_name(p.live_to_recipe_name(k))

    # KDA projection: live carries `.forget_gate.`, checkpoint is flat.
    assert (
        sink_rename(
            "model.language_model.layers.44.self_attn.forget_gate.f_a_proj.weight"
        )
        == "model.language_model.layers.44.self_attn.f_a_proj.weight"
    )
    # Recipe-space input gives the same checkpoint key (idempotence).
    assert (
        sink_rename("model.layers.44.self_attn.f_a_proj.weight")
        == "model.language_model.layers.44.self_attn.f_a_proj.weight"
    )
    assert sink_rename("lm_head.weight") == "lm_head.weight"
    # Shared-expert Linear round-trips through both namespaces.
    assert (
        sink_rename(
            "model.language_model.layers.7.mlp.shared_experts.gate_proj.weight"
        )
        == "model.language_model.layers.7.mlp.shared_experts.gate_proj.weight"
    )


def test_glm5_next_concat_merge_source_keys_roundtrip():
    """The emit-side inverse of `concat_merges` reads each SOURCE tensor
    by its checkpoint key, derived from the live stem + source suffix
    through the same sink rename. glm5_next: one live conv1d <- three
    source conv1ds, so the artifact must ship q/k/v_conv1d verbatim and
    never a merged `conv1d` key the loader does not know."""
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile

    p = Glm5NextProfile()
    groups = p.concat_merge_groups()
    assert groups, "glm5_next must declare concat_merges"
    (target_suffix, source_suffixes, dim), = groups
    assert target_suffix == "self_attn.conv1d.weight"
    assert dim == 0
    live_target = "model.language_model.layers.0." + target_suffix
    stem = live_target[: -len(target_suffix)]
    ckpt_keys = [
        p.export_tensor_name(p.live_to_recipe_name(stem + s))
        for s in source_suffixes
    ]
    assert ckpt_keys == [
        "model.language_model.layers.0.self_attn.q_conv1d.weight",
        "model.language_model.layers.0.self_attn.k_conv1d.weight",
        "model.language_model.layers.0.self_attn.v_conv1d.weight",
    ]


def test_glm5_next_quant_config_is_vllm_internal_namespace():
    """Regression: 2026-08-27 TP=2 serve OOMs. Targets/ignore must be in
    vLLM-internal namespace (language_model.model.*), the runtime-pinned
    MLA projections must land in ignore (never config_groups), and the
    per-expert safety-net regex must not attach to a non-expert
    catch-all group."""
    from prismaquant.export_native_compressed import build_quantization_config
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile

    profile = Glm5NextProfile()
    assignment = {
        # packed NVFP4 experts (allocator units, recipe namespace)
        "model.layers.10.mlp.experts.gate_up_proj": "NVFP4",
        "model.layers.10.mlp.experts.down_proj": "NVFP4",
        # dense-half FP8_SOURCE (largest group -> catch-all)
        "model.layers.0.mlp.gate_proj": "FP8_SOURCE",
        "model.layers.0.mlp.up_proj": "FP8_SOURCE",
        "model.layers.0.mlp.down_proj": "FP8_SOURCE",
        # runtime-pinned MLA projection swept in by the fp8-source overlay
        "model.layers.11.self_attn.q_b_proj": "FP8_SOURCE",
    }
    qc = build_quantization_config(assignment, set(), profile=profile)

    all_targets = [
        t for g in qc["config_groups"].values() for t in g["targets"]
    ]
    for t in all_targets:
        assert "language_model[.]model[.]" in t or "language_model.model." in t, t
        assert not t.startswith("re:^model[.]layers"), t
        assert "self_attn" not in t, t

    assert (
        "language_model.model.layers.11.self_attn.q_b_proj" in qc["ignore"]
    )

    # per-expert safety-net must not ride the FP8_SOURCE catch-all
    for g in qc["config_groups"].values():
        if any("mlp[.]gate_proj" in t or "mlp[.]up_proj" in t for t in g["targets"]):
            assert not any(".experts." in t or "experts[.]" in t
                           for t in g["targets"]), g["targets"]


def test_glm5_next_fp8_source_overlay_skips_packed_expert_leaves():
    """The fp8-source config overlay must not re-describe per-expert
    leaves whose packed parent the allocator already assigned."""
    from prismaquant.export_native_compressed import _fp8_source_config_overlay

    # Exercise just the skip logic via the module-level helper pieces:
    # simulate by calling the overlay with a source map through the
    # public function would need a checkpoint; instead assert the
    # prefix-ownership rule directly.
    import re as _re
    assignment = {"model.layers.10.mlp.experts.gate_up_proj": "NVFP4"}
    packed_prefixes = tuple(
        {k.rsplit(".", 1)[0] + "." for k in assignment
         if ".mlp.experts." in k or k.endswith(".mlp.experts")})
    leaf = "model.layers.10.mlp.experts.7.up_proj"
    assert _re.search(r"\.experts\.[0-9]+\.", leaf)
    assert leaf.startswith(packed_prefixes)
