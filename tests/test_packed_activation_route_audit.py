"""The real HF dispatch must apply QDQ to both packed projections."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prismaquant import format_registry as fr
from prismaquant.perturbed_x_cache import PerturbedActivationCache


@pytest.mark.parametrize("family", ["qwen3_moe", "qwen3_5_moe"])
@pytest.mark.parametrize("implementation", ["eager", "batched_mm"])
def test_real_hf_packed_dispatch_quantizes_both_inputs(tmp_path, monkeypatch, family, implementation):
    if family == "qwen3_moe":
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts as Experts
    else:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts as Experts
    config = SimpleNamespace(num_experts=2, hidden_size=32, moe_intermediate_size=16,
                             hidden_act="silu", _experts_implementation=implementation)
    model = nn.Module()
    model.experts = Experts(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(.1)
    seen = []

    def qdq(x):
        seen.append(x.shape[-1])
        return torch.zeros_like(x)

    spec = fr.FormatSpec(
        name="AUDIT_ZERO_ACT", weight_bits=8, group_size=0, scale_bits=0,
        scale_dtype_name="none", weight_element_dtype="test", act_bits=4,
        quantize_dequantize=lambda w: w.clone(), activation_quantize_dequantize=qdq)
    monkeypatch.setitem(fr.REGISTRY, spec.name, spec)
    cache = PerturbedActivationCache(model,
        {"experts.gate_up_proj": spec.name, "experts.down_proj": spec.name},
        tmp_path, cal_hash="audit", capture_inputs=False)
    assert not cache.missing
    inputs = torch.ones(6, 32)
    indices = torch.tensor([[0, 1]] * 6)
    weights = torch.full((6, 2), .5)
    with torch.no_grad():
        baseline = model.experts(inputs, indices, weights)
        assert baseline.abs().sum() > 0
        cache.install()
        try:
            actual = model.experts(inputs, indices, weights)
        finally:
            cache.remove()
        assert config._experts_implementation == implementation
        restored = model.experts(inputs, indices, weights)
    torch.testing.assert_close(restored, baseline)
    assert set(seen) == {16, 32}, seen
    assert torch.count_nonzero(actual) == 0
