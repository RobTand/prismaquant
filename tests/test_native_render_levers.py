"""Explicit native render options bind arithmetic and stored fingerprints."""
from dataclasses import FrozenInstanceError

import pytest
import torch

import prismaquant.native_render_codec as codec
import prismaquant.export_native_compressed as exporter
import prismaquant.production_weight_cache as pwc
from prismaquant.native_render_levers import NativeRenderLevers


def test_explicit_all_false_preserves_rtn_when_render_options_enable_gptq(monkeypatch):
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(16, 16, generator=generator)
    acts = torch.randn(64, 16, generator=generator)
    monkeypatch.setenv("PRISMAQUANT_DO_NO_HARM", "0")
    options = NativeRenderLevers.resolve({"gptq": True})
    wp, ws, wg = codec.quantize_dequantize_nvfp4(weight)
    expected = {"weight_packed": wp, "weight_scale": ws, "weight_global_scale": wg}
    actual = codec._quantize_2d(
        weight, "NVFP4", cached_activations=acts, levers=options,
        gptq_enabled=False, scale_sweep_enabled=False,
        static_act_order_enabled=False, joint_scale_opt_enabled=False,
        compute_only=False)
    for key, tensor in expected.items():
        assert torch.equal(actual[key], tensor)


def test_resolved_options_freeze_levels_and_damp_roles(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS", "6,3,1")
    monkeypatch.setenv("PRISMAQUANT_GPTQ_DAMP_ROLES", "gate_up=0.3,down=3")
    options = NativeRenderLevers.resolve({"nvfp4_scale_rule": "joint_mse"})
    monkeypatch.setenv("PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS", "6,4")
    monkeypatch.setenv("PRISMAQUANT_GPTQ_DAMP_ROLES", "gate_up=1,down=1")
    groups = torch.randn(3, 4, 16, generator=torch.Generator().manual_seed(3))
    expected = codec._nvfp4_best_max_to_level_scale(groups, (6.0, 3.0, 1.0))
    actual = codec._select_nvfp4_group_scales(groups, levers=options)
    assert torch.equal(actual, expected)
    assert codec._resolve_gptq_damp_for_role("model.layers.0.mlp.gate_proj", levers=options) == 0.3
    assert codec._resolve_gptq_damp_for_role("model.layers.0.mlp.down_proj", levers=options) == 3.0
    with pytest.raises(FrozenInstanceError):
        options.gptq = True


def test_every_native_fingerprint_carries_complete_render_options(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS", "6,3,1")
    monkeypatch.setenv("PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_GRID", "9")
    monkeypatch.setenv("PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_SPAN_LO", "0.9")
    monkeypatch.setenv("PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_SPAN_HI", "1.1")
    monkeypatch.setenv("PRISMAQUANT_GPTQ_DAMP_ROLES", "gate_up=0.3")
    options = NativeRenderLevers.resolve(gptq_default=True, optional_flags_from_env=True)
    record = pwc._resolve_production_render_levers(None)
    assert record["nvfp4_joint_scale_levels"] == [6.0, 3.0, 1.0]
    assert record["nvfp4_joint_scale_global_grid"] == 9
    assert record["nvfp4_joint_scale_global_span_lo"] == 0.9
    assert record["nvfp4_joint_scale_global_span_hi"] == 1.1
    assert record["gptq_damp_roles"] == {"gate_up": 0.3}
    cache = pwc.ProductionWeightCache(
        {("a", "NVFP4"): torch.zeros(16, 16, dtype=torch.bfloat16)}, levers=record)
    monkeypatch.setattr(exporter, "_PRODUCTION_WEIGHT_CACHE", cache)
    assert exporter._production_cache_fingerprint(cache, [("a", "NVFP4")])["levers"] == options.record()
    assert exporter._render_lever_provenance()["native_render_levers"] == options.record()


def test_frozen_recipes_survive_interleaved_rendering(monkeypatch):
    generator = torch.Generator().manual_seed(2)
    weight = torch.randn(8, 32, generator=generator)
    static = NativeRenderLevers.resolve({"nvfp4_scale_rule": "static_6"})
    joint = NativeRenderLevers.resolve({"nvfp4_scale_rule": "joint_mse"})
    first = codec.quantize_dequantize_nvfp4(weight, levers=static)
    second = codec.quantize_dequantize_nvfp4(weight, levers=joint)
    assert not torch.equal(first[1], second[1])
    monkeypatch.setenv("PRISMAQUANT_NVFP4_SCALE_RULE", "four_over_six_mse")
    monkeypatch.setenv("PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS", "6,3,1")
    for options, expected in ((static, first), (joint, second), (static, first)):
        actual = codec.quantize_dequantize_nvfp4(weight, levers=options)
        for got, wanted in zip(actual, expected):
            assert torch.equal(got, wanted)


def test_record_roundtrip_is_environment_independent(monkeypatch):
    options = NativeRenderLevers.resolve({"gptq": True, "joint_scale_opt": True})
    record = options.record()
    monkeypatch.setenv("PRISMAQUANT_NVFP4_SCALE_RULE", "invalid-after-resolution")
    assert NativeRenderLevers.from_record(record) == options
    assert NativeRenderLevers.resolve(record) == options
    with pytest.raises(ValueError, match="incomplete"):
        NativeRenderLevers.from_record({"gptq": True})


def test_export_match_opt_out_still_refuses_unproven_historical_options(monkeypatch):
    cache = pwc.ProductionWeightCache({}, levers={"gptq": True})
    monkeypatch.setattr(exporter, "_RENDER_LEVERS", NativeRenderLevers(nvfp4_export_match_render_scale=False))
    with pytest.raises(ValueError, match="incomplete.*re-render"):
        exporter._export_render_options(cache)


@pytest.mark.parametrize("field,value", [
    ("nvfp4_joint_scale_levels", "6,4"),
    ("nvfp4_joint_scale_levels", []),
    ("nvfp4_joint_scale_levels", [6.0, float("nan")]),
    ("fp8_scale_sweep_factors", [1.0, float("inf")]),
    ("mxfp8_joint_scale_shifts", [True, 0]),
    ("gptq_damp_roles", [["down", 3.0]]),
    ("gptq_damp_roles", {"down": "3.0"}),
    ("gptq_damp_roles", {"down": float("inf")}),
    ("gptq_fixed_damp", float("nan")),
    ("fisher_row_weight_clip", float("inf")),
    ("act_clip_quantile", True),
    ("act_clip_quantile", 2.0),
    ("nvfp4_joint_scale_global_span_hi", 0.25),
    ("nvfp4_joint_scale_global_grid", 0),
    ("gptq_block_size", 0),
    ("nvfp4_scale_rule", "unknown"),
    ("nvfp4_input_global_scale_policy", "unknown"),
])
def test_completed_recipe_refuses_corrupt_controls(field, value):
    record = NativeRenderLevers().record()
    record[field] = value
    with pytest.raises(ValueError, match="native render option"):
        NativeRenderLevers.from_record(record)


def test_completed_recipe_preserves_supported_disabled_controls():
    options = NativeRenderLevers(
        act_clip_quantile=None, fisher_row_weight_clip=0.0,
        gptq_damp_analytical_c=0.0, gptq_damp_roles=(("unmatched", 0.25),))
    parsed = NativeRenderLevers.from_record(options.record())
    assert parsed == options
    acts = torch.tensor([[100.0, -5.0], [1.0, 2.0]])
    assert torch.equal(codec._activation_matrix_for_gptq(acts, 2, levers=parsed), acts)


def test_explicit_activation_rows_preserve_existing_weight_name_alias():
    generator = torch.Generator().manual_seed(51)
    weight = torch.randn(16, 16, generator=generator)
    acts = torch.randn(64, 16, generator=generator)
    record = NativeRenderLevers.resolve({
        "gptq": True, "render_progressive_gates": False,
        "do_no_harm": False}).record()
    expected = pwc.render_production_weight(
        weight, "NVFP4", qname="linear", activations={"linear": acts}, levers=record)
    actual = pwc.render_production_weight(
        weight, "NVFP4", qname="linear.weight", activations={"linear": acts}, levers=record)
    assert not torch.equal(expected, weight)
    assert torch.equal(actual, expected)
