"""A grouped expert call must quantize every targeted projection exactly once."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers.integrations import moe

from prismaquant import perturbed_x_cache as owner
from test_packed_grouped_qdq_audit import experts


@pytest.mark.parametrize("family", ["qwen3_moe", "qwen3_5_moe"])
@pytest.mark.parametrize("implementation", ["batched_mm", "grouped_mm"])
@pytest.mark.parametrize("bypass", [
    "all", "gate_up_proj", "down_proj", "copied_gate_up_proj", "duplicate_gate_up_skips_down",
])
def test_grouped_dispatch_refuses_missing_or_repeated_qdq(tmp_path, monkeypatch, family, implementation, bypass):
    monkeypatch.setenv("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", "1")
    model = nn.Module()
    model.experts = experts(family, implementation)
    linear, grouped = F.linear, F.grouped_mm

    def dispatch(x, weight, offs):
        gate_up = weight.shape[-2] == 64
        if bypass == "duplicate_gate_up_skips_down" and gate_up:
            F.grouped_mm(x, weight, offs=offs)
        if (bypass == "all" or bypass == "gate_up_proj" and gate_up
                or bypass in ("down_proj", "duplicate_gate_up_skips_down") and not gate_up):
            return torch._grouped_mm(x, weight, offs=offs)
        if bypass == "copied_gate_up_proj" and gate_up:
            weight = weight.clone()
        return F.grouped_mm(x, weight, offs=offs)

    monkeypatch.setattr(moe, "_grouped_mm", dispatch)
    cache = owner.PerturbedActivationCache(model,
        {"experts.gate_up_proj": "NVFP4", "experts.down_proj": "NVFP4"},
        tmp_path, cal_hash="dispatch", capture_inputs=False)
    x = torch.randn(7, 64)
    ids = torch.tensor([[0, 1]] * 7)
    weights = torch.full((7, 2), .5)
    with torch.no_grad():
        cache.install()
        try:
            with pytest.raises(RuntimeError, match="packed activation QDQ grouped dispatch"):
                model.experts(x, ids, weights)
        finally:
            cache.remove()
    assert F.linear is linear and F.grouped_mm is grouped
    assert model.experts.config._experts_implementation == implementation


@pytest.mark.parametrize("family", ["qwen3_moe", "qwen3_5_moe"])
@pytest.mark.parametrize("implementation", ["batched_mm", "grouped_mm"])
def test_grouped_dispatch_only_requires_active_qdq_targets(tmp_path, monkeypatch, family, implementation):
    monkeypatch.setenv("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", "1")
    model = nn.Module()
    model.experts = experts(family, implementation)

    def dispatch(x, weight, offs):
        operation = F.grouped_mm if weight.shape[-2] == 64 else torch._grouped_mm
        return operation(x, weight, offs=offs)

    monkeypatch.setattr(moe, "_grouped_mm", dispatch)
    cache = owner.PerturbedActivationCache(model,
        {"experts.gate_up_proj": "NVFP4", "experts.down_proj": "BF16"},
        tmp_path, cal_hash="active-target", capture_inputs=False)
    with torch.no_grad():
        cache.install()
        try:
            actual = model.experts(torch.randn(7, 64), torch.tensor([[0, 1]] * 7), torch.full((7, 2), .5))
        finally:
            cache.remove()
    assert torch.isfinite(actual).all()
