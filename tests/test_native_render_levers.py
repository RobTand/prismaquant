"""Explicit native render options bind arithmetic and stored fingerprints."""
from dataclasses import FrozenInstanceError

import pytest
import torch

import prismaquant.native_render_codec as codec
import prismaquant.export_native_compressed as exporter
import prismaquant.production_weight_cache as pwc
from prismaquant.native_render_levers import NativeRenderLevers




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
