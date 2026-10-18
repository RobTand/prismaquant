"""Packed QDQ must preserve row-local arithmetic and grouped dispatch."""
from dataclasses import replace
import os

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from prismaquant import format_registry as fr
from prismaquant import perturbed_x_cache as owner


FORMATS = ["NVFP4", "MXFP4", "MXFP8_E4M3", "MXFP8_E5M2", "MXFP8_UE8M0_G32", "FP8_E4M3", "FP8_E5M2"]


def experts(family, implementation, *, device="cpu", hidden=64, intermediate=32, count=4):
    if family == "qwen3_moe":
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts as Experts
        from transformers import Qwen3MoeConfig as Config
    else:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts as Experts
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig as Config
    config = Config(num_experts=count, hidden_size=hidden, moe_intermediate_size=intermediate, hidden_act="silu")
    config._experts_implementation = implementation
    module = Experts(config).to(device)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, .1)
    return module


@pytest.mark.parametrize("name", FORMATS)
@pytest.mark.parametrize("width", [32, 64, 96])
def test_grouped_activation_rows_match_expert_reference(name, width):
    spec = fr.get_format(name)
    x = torch.randn(13, width)
    x[0].zero_()
    x[1].mul_(100)
    scales = {"projection": 2.0}
    grouped = owner._activation_qdq(x, spec, scales, "projection")
    reference = torch.cat([owner._activation_qdq(part, spec, scales, "projection")
                           for part in x.split([3, 1, 9])])
    assert torch.equal(grouped, reference)
    assert owner._packed_activation_qdq_is_row_local(spec)


@pytest.mark.parametrize("family", ["qwen3_moe", "qwen3_5_moe"])
@pytest.mark.parametrize("implementation", ["batched_mm", "grouped_mm"])
def test_real_hf_row_local_qdq_keeps_grouped_boundary(tmp_path, monkeypatch, family, implementation):
    monkeypatch.setenv("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", "1")
    model = nn.Module()
    model.experts = experts(family, implementation)
    cache = owner.PerturbedActivationCache(model, {"experts.gate_up_proj": "NVFP4", "experts.down_proj": "NVFP4"},
                                           tmp_path, cal_hash="grouped", capture_inputs=False)
    grouped = F.grouped_mm
    calls = []

    def observe(x, weight, **kwargs):
        calls.append(tuple(x.shape))
        return grouped(x, weight, **kwargs)

    monkeypatch.setattr(F, "grouped_mm", observe)
    x = torch.randn(7, 64)
    ids = torch.tensor([[0, 2], [2, 1], [3, 0], [1, 3], [3, 2], [2, 0], [0, 1]])
    weights = torch.full((7, 2), .5)
    with torch.no_grad():
        cache.install()
        try:
            actual = model.experts(x, ids, weights)
            assert len(calls) == 2, calls
            model.experts.config._experts_implementation = "eager"
            reference = model.experts(x, ids, weights)
        finally:
            cache.remove()
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-7)


def test_batch_dependent_qdq_keeps_eager_reference(tmp_path, monkeypatch):
    model = nn.Module()
    model.experts = experts("qwen3_moe", "grouped_mm")
    seen = []

    def quantize(x):
        seen.append(tuple(x.shape))
        return x / x.abs().amax().clamp_min(1e-8)

    spec = replace(fr.get_format("NVFP4"), name="AUDIT_BATCH_SCALE", static_activation_contract=None,
                   quantize_dequantize=lambda x: x, activation_quantize_dequantize=quantize)
    monkeypatch.setitem(fr.REGISTRY, spec.name, spec)
    monkeypatch.setenv("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", "1")
    cache = owner.PerturbedActivationCache(model, {"experts.gate_up_proj": spec.name, "experts.down_proj": spec.name},
                                           tmp_path, cal_hash="nonlocal", capture_inputs=False)
    with torch.no_grad():
        cache.install()
        try:
            model.experts(torch.randn(3, 64), torch.tensor([[0, 1]] * 3), torch.full((3, 2), .5))
        finally:
            cache.remove()
    assert len(seen) == 4
    assert model.experts.config._experts_implementation == "grouped_mm"


@pytest.mark.parametrize("implementation", ["batched_mm", "grouped_mm"])
def test_exception_restores_global_ops_and_real_config(tmp_path, monkeypatch, implementation):
    monkeypatch.setenv("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", "1")
    model = nn.Module()
    model.experts = experts("qwen3_moe", implementation)
    cache = owner.PerturbedActivationCache(model, {"experts.gate_up_proj": "NVFP4", "experts.down_proj": "NVFP4"},
                                           tmp_path, cal_hash="raise", capture_inputs=False)
    original_linear, original_grouped = F.linear, F.grouped_mm

    def broken(*args, **kwargs):
        raise RuntimeError("expert failed")

    model.experts.forward = broken
    with torch.no_grad():
        cache.install()
        try:
            with pytest.raises(RuntimeError, match="expert failed"):
                model.experts(torch.randn(2, 64), torch.zeros(2, 2, dtype=torch.int64), torch.ones(2, 2))
        finally:
            cache.remove()
    assert F.linear is original_linear
    assert F.grouped_mm is original_grouped
    assert model.experts.config._experts_implementation == implementation


@pytest.mark.parametrize("family", ["qwen3_moe", "qwen3_5_moe"])
@pytest.mark.parametrize("initial,replacement", [
    ("AUDIT_ROW_LOCAL", "AUDIT_BATCH_SCALE"),
    ("AUDIT_BATCH_SCALE", "AUDIT_ROW_LOCAL"),
    ("BF16", "AUDIT_ROW_LOCAL"),
    ("AUDIT_ROW_LOCAL", "BF16"),
])
def test_public_format_override_refreshes_packed_qdq(tmp_path, monkeypatch, family, initial, replacement):
    monkeypatch.setenv("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", "1")
    row = replace(fr.get_format("NVFP4"), name="AUDIT_ROW_LOCAL", static_activation_contract=None,
                  quantize_dequantize=lambda x: x.clone())
    batch = replace(row, name="AUDIT_BATCH_SCALE", activation_quantize_dequantize=(
        lambda x: x / x.abs().amax().clamp_min(1e-8)))
    monkeypatch.setitem(fr.REGISTRY, row.name, row)
    monkeypatch.setitem(fr.REGISTRY, batch.name, batch)
    model = nn.Module()
    model.experts = experts(family, "grouped_mm")
    names = ["experts.gate_up_proj", "experts.down_proj"]
    cache = owner.PerturbedActivationCache(model, dict.fromkeys(names, initial),
                                           tmp_path, cal_hash="override", capture_inputs=False)
    grouped, quantize = F.grouped_mm, owner._activation_qdq
    boundaries, formats = [], []

    def observe_grouped(x, weight, **kwargs):
        boundaries.append(tuple(x.shape))
        return grouped(x, weight, **kwargs)

    def observe_quantize(x, spec, *args, **kwargs):
        formats.append(spec.name)
        return quantize(x, spec, *args, **kwargs)

    monkeypatch.setattr(F, "grouped_mm", observe_grouped)
    monkeypatch.setattr(owner, "_activation_qdq", observe_quantize)
    x = torch.randn(3, 64)
    ids = torch.tensor([[0, 1]] * 3)
    weights = torch.full((3, 2), .5)

    def run(fmt):
        boundaries.clear()
        formats.clear()
        model.experts(x, ids, weights)
        assert len(boundaries) == (0 if fmt == batch.name else 2)
        assert formats == ([fmt] * (4 if fmt == batch.name else 2) if fmt != "BF16" else [])
        assert model.experts.config._experts_implementation == "grouped_mm"

    with torch.no_grad(), cache.frozen_weight_cache():
        cache.install()
        try:
            run(initial)
            with cache.override(dict.fromkeys(names, replacement)):
                run(replacement)
            run(initial)
        finally:
            cache.remove()


@pytest.mark.skipif(os.environ.get("PRISMAQUANT_PACKED_QDQ_CUDA") != "1",
                    reason="GB10 qualification needs an admitted CUDA action")
@pytest.mark.parametrize("family,intermediate,count", [
    ("qwen3_moe", 768, 128), ("qwen3_5_moe", 512, 256),
])
def test_gb10_qdq_rows_and_grouped_graph(tmp_path, monkeypatch, family, intermediate, count):
    assert torch.cuda.is_available(), "CUDA qualification cannot fall back to CPU"
    assert torch.cuda.get_device_capability() == (12, 1), "qualification requires GB10"
    monkeypatch.setenv("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", "1")
    for width in (2048, intermediate):
        rows = torch.randn(13, width, device="cuda", dtype=torch.bfloat16)
        rows[0].zero_()
        rows[1].mul_(100)
        for name in FORMATS:
            spec = fr.get_format(name)
            actual = owner._activation_qdq(rows, spec, {}, "projection")
            reference = torch.cat([
                owner._activation_qdq(part, spec, {}, "projection")
                for part in rows.split([3, 1, 9])])
            assert actual.is_cuda and reference.is_cuda
            assert torch.equal(actual, reference), (family, width, name)

    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = nn.Module()
        model.experts = experts(family, "grouped_mm", device="cuda", hidden=2048,
                                intermediate=intermediate, count=count)
    finally:
        torch.set_default_dtype(dtype)
    cache = owner.PerturbedActivationCache(model,
        {"experts.gate_up_proj": "NVFP4", "experts.down_proj": "NVFP4"},
        tmp_path, cal_hash="gb10", capture_inputs=False)
    inputs = torch.randn(32, 2048, device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor([[0, 1]] * 32, device="cuda")
    weights = torch.full((32, 2), .5, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        cache.install()
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    model.experts(inputs, indices, weights)
            torch.cuda.current_stream().wait_stream(stream)
            uncaptured = model.experts(inputs, indices, weights)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = model.experts(inputs, indices, weights)
            graph.replay()
            torch.cuda.synchronize()
            assert captured.is_cuda and torch.isfinite(captured).all()
            assert torch.equal(captured, uncaptured)
        finally:
            cache.remove()
