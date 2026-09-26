"""Empirical packed-expert unit-KL cost (the AURA-MoE hybrid's expert leg).

Pins: unit KL measured per serving unit and split across members by
n_params; FP8 stays in the menu (measured, not banned); BF16 rows are
passthrough-zero; weights restored after measurement; hybrid merge refuses
double-costed names; backfill only adds missing rows.
"""
from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from prismaquant.expert_empirical_cost import (
    backfill_missing_from_base,
    measure_expert_unit_costs,
    merge_cost_payloads,
)

from test_packed_expert_cross_domain_gate import TinyLM


class TinyCausal(nn.Module):
    """TinyLM + head so the KL harness sees ``.logits``."""

    def __init__(self, vocab: int = 32, hidden: int = 16):
        super().__init__()
        self.inner = TinyLM(vocab=vocab, hidden=hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.vocab = vocab

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        h = self.inner(input_ids)
        logits = self.head(h).reshape(input_ids.shape[0], -1, self.vocab)
        return types.SimpleNamespace(logits=logits)


EXPERT_NAMES = {
    "inner.mlp.experts.gate_up_proj",
    "inner.mlp.experts.down_proj",
}


def test_measure_unit_costs_on_tiny_packed_moe():
    torch.manual_seed(7)
    model = TinyCausal().eval()
    calib = torch.randint(0, 32, (2, 24))
    before = {
        n: getattr(model.inner.mlp.experts, a).detach().clone()
        for n, a in (("gate_up", "gate_up_proj"), ("down", "down_proj"))
    }

    stats, costs, unit_kls = measure_expert_unit_costs(
        model, None, calib, ["NVFP4", "FP8_DYNAMIC", "BF16"],
        expert_chunk=1, progress=False)

    assert set(stats) == EXPERT_NAMES
    assert set(costs) == EXPERT_NAMES
    (unit,) = unit_kls.values()
    assert unit["NVFP4"] > 0.0
    assert unit["FP8_E4M3"] >= 0.0
    # FP8 error should be well below NVFP4 on the same unit.
    assert unit["FP8_E4M3"] < unit["NVFP4"]
    for name in EXPERT_NAMES:
        row = costs[name]
        assert set(row) == {"NVFP4", "FP8_E4M3", "BF16"}
        assert row["BF16"]["predicted_dloss"] == 0.0
        assert row["NVFP4"]["cost_source"] == "empirical_unit_kl"
        assert stats[name]["h_trace"] == 0.0
    # Member shares re-assemble exactly one unit KL per format.
    for fmt in ("NVFP4", "FP8_E4M3"):
        total = sum(costs[n][fmt]["predicted_dloss"] for n in EXPERT_NAMES)
        assert total == pytest.approx(unit[fmt], rel=1e-6)
    # In-place quantize/restore left the model untouched.
    assert torch.equal(
        model.inner.mlp.experts.gate_up_proj.detach(), before["gate_up"])
    assert torch.equal(
        model.inner.mlp.experts.down_proj.detach(), before["down"])


class _Dsv4PerExpert(nn.Module):
    def __init__(self, hidden: int = 16, intermediate: int = 32):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class _Dsv4PerExpertMlp(nn.Module):
    def __init__(self, hidden: int = 16, num_experts: int = 2):
        super().__init__()
        self.experts = nn.ModuleList(
            [_Dsv4PerExpert(hidden) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        route_flat = routes.reshape(-1).remainder(len(self.experts))
        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            selected = route_flat == expert_id
            if bool(selected.any()):
                out[selected] = expert(flat[selected])
        return out.reshape_as(x)


class _Dsv4PerExpertLayer(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.mlp = _Dsv4PerExpertMlp(hidden)

    def forward(self, x: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x, routes)


class _Dsv4PerExpertCausal(nn.Module):
    """Functional DSv4-shaped tree with per-expert ``nn.Linear`` rows."""

    def __init__(self, vocab: int = 32, hidden: int = 16):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([_Dsv4PerExpertLayer(hidden)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden, input_ids)
        return types.SimpleNamespace(logits=self.lm_head(hidden))


def test_measure_unit_costs_on_dsv4_unpacked_expert_linears():
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile

    torch.manual_seed(19)
    model = _Dsv4PerExpertCausal().eval()
    calib = torch.randint(0, 32, (2, 12))
    expert_names = {
        f"model.layers.0.mlp.experts.{expert_id}.{projection}"
        for expert_id in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    before = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if name in expert_names
    }

    stats, costs, unit_kls = measure_expert_unit_costs(
        model,
        DeepseekV4Profile(),
        calib,
        ["NVFP4", "BF16"],
        progress=False,
    )

    assert set(stats) == expert_names
    assert set(costs) == expert_names
    assert set(unit_kls) == {"model.layers.0.mlp.experts"}
    unit = unit_kls["model.layers.0.mlp.experts"]
    assert unit["NVFP4"] > 0.0
    assert sum(
        costs[name]["NVFP4"]["predicted_dloss"] for name in expert_names
    ) == pytest.approx(unit["NVFP4"], rel=1e-6)
    for name in expert_names:
        assert stats[name]["_unpacked_expert_unit"] == (
            "model.layers.0.mlp.experts"
        )
        assert "num_experts" not in stats[name]
        assert stats[name]["in_features"] == before[name].shape[1]
        assert stats[name]["out_features"] == before[name].shape[0]
    for name, module in model.named_modules():
        if name in expert_names:
            assert torch.equal(module.weight.detach(), before[name])


def test_dsv4_unpacked_expert_cost_refuses_a_retired_codebook_rung():
    # A stale menu naming a retired codebook rung (archived 2026-09-25,
    # #1304) refuses before any unit is measured.
    from prismaquant import format_registry as fr
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile

    torch.manual_seed(23)
    model = _Dsv4PerExpertCausal().eval()
    calib = torch.randint(0, 32, (1, 8))

    with pytest.raises(fr.RetiredFormatError, match="gridbook_lane"):
        measure_expert_unit_costs(
            model,
            DeepseekV4Profile(),
            calib,
            ["FP8_CB_K28"],
            progress=False,
            col_weights={},
        )


def test_merge_refuses_double_costed_names():
    base = {"stats": {"a": {}}, "costs": {"a": {"NVFP4": {}}}}
    with pytest.raises(RuntimeError, match="collision"):
        merge_cost_payloads(
            base, {"a": {}}, {"a": {"NVFP4": {}}}, formats=["NVFP4", "BF16"])


def test_merge_and_backfill():
    base = {
        "stats": {"lin": {"h_trace": 1.0}},
        "costs": {"lin": {"NVFP4": {"predicted_dloss": 0.5}}},
    }
    merged = merge_cost_payloads(
        base,
        {"experts.down_proj": {"h_trace": 0.0}},
        {"experts.down_proj": {"NVFP4": {"predicted_dloss": 0.1}}},
        formats=["NVFP4", "FP8_DYNAMIC", "BF16"],
    )
    assert set(merged["costs"]) == {"lin", "experts.down_proj"}
    assert merged["formats"] == ["NVFP4", "FP8_E4M3", "BF16"]

    base_cost = {
        "stats": {"mtp.fc": {"h_trace": 2.0}},
        "costs": {
            "mtp.fc": {"NVFP4": {"predicted_dloss": 9.0}},
            "lin": {"NVFP4": {"predicted_dloss": 777.0}},  # must NOT override
        },
    }
    added = backfill_missing_from_base(merged, base_cost)
    assert added == ["mtp.fc"]
    assert merged["costs"]["lin"]["NVFP4"]["predicted_dloss"] == 0.5
    assert merged["costs"]["mtp.fc"]["NVFP4"]["predicted_dloss"] == 9.0
    assert merged["stats"]["mtp.fc"]["h_trace"] == 2.0


# --------------------------------------------------------------------------
# Audit 2026-07-02 §3.5: NV formats derive one per-TENSOR global scale from
# the slice they are given, so chunk-batched expert quantization shared one
# global across the chunk and made the measured unit KL depend on the
# --expert-chunk knob. Export ships per-expert globals; the measurement must
# quantize per expert slice.
# --------------------------------------------------------------------------
def _spread_expert_model(num_experts: int = 16):
    """TinyCausal with ``num_experts`` experts and a 4x per-expert magnitude
    spread (a chunk-shared NVFP4 global visibly distorts the small ones)."""
    from test_packed_expert_cross_domain_gate import (
        TinyPackedExperts,
        TinyRouter,
    )

    model = TinyCausal().eval()
    model.inner.mlp.gate = TinyRouter(hidden_size=16, num_experts=num_experts)
    model.inner.mlp.experts = TinyPackedExperts(num_experts=num_experts)
    with torch.no_grad():
        for e in range(num_experts):
            scale = 1.0 + 3.0 * e / (num_experts - 1)
            model.inner.mlp.experts.gate_up_proj[e] *= scale
            model.inner.mlp.experts.down_proj[e] *= scale
    return model


def test_nvfp4_unit_kl_is_expert_chunk_invariant():
    from prismaquant import format_registry as fr
    from prismaquant.expert_empirical_cost import (
        _baseline_logprobs,
        _unit_kl,
    )

    torch.manual_seed(3)
    model = _spread_expert_model(16)
    mod = model.inner.mlp.experts
    pnames = ["gate_up_proj", "down_proj"]

    # Premise: on this tensor the chunk-shared global genuinely differs from
    # per-expert globals (otherwise the test could not discriminate).
    spec = fr.get_format("NVFP4")
    w = mod.gate_up_proj.detach().float()
    per_expert = torch.stack(
        [spec.quantize_dequantize(w[e].clone()) for e in range(16)])
    shared = spec.quantize_dequantize(w.clone())
    assert not torch.equal(per_expert, shared)

    calib = torch.randint(0, 32, (2, 24))
    baseline = _baseline_logprobs(model, calib)
    kls = {
        chunk: _unit_kl(model, calib, baseline, mod, pnames, "NVFP4",
                        expert_chunk=chunk)
        for chunk in (1, 4, 16)
    }
    # Per-expert quantization: the chunk knob cannot change the measurement.
    assert kls[4] == kls[1]
    assert kls[16] == kls[1]
    assert kls[1] > 0.0


def test_fp8_weight_qdq_is_chunk_invariant_on_packed_tensors():
    """FP8_E4M3 weight qdq reshapes to (-1, in) with an independent scale per
    output row, so chunk-batching experts is exact — the verified basis for
    keeping the batched path for non-NV formats in ``_unit_kl``."""
    from prismaquant import format_registry as fr
    from prismaquant.expert_empirical_cost import (
        _baseline_logprobs,
        _unit_kl,
    )

    torch.manual_seed(4)
    spec = fr.get_format("FP8_E4M3")
    w = torch.randn(16, 8, 16)
    w *= torch.linspace(1.0, 4.0, 16).view(-1, 1, 1)
    full = spec.quantize_dequantize(w.clone())
    per = torch.stack(
        [spec.quantize_dequantize(w[e].clone()) for e in range(16)])
    assert torch.equal(full, per)

    model = _spread_expert_model(16)
    mod = model.inner.mlp.experts
    calib = torch.randint(0, 32, (2, 24))
    baseline = _baseline_logprobs(model, calib)
    kls = {
        chunk: _unit_kl(model, calib, baseline, mod,
                        ["gate_up_proj", "down_proj"], "FP8_E4M3",
                        expert_chunk=chunk)
        for chunk in (4, 16)
    }
    assert kls[4] == kls[16]


# --------------------------------------------------------------------------- #
# Expert sampling + ladder generalization (encode-speed workstream 2026-07-19)
# --------------------------------------------------------------------------- #
def _wide_moe(num_experts: int = 8, seed: int = 11) -> "TinyCausal":
    from test_packed_expert_cross_domain_gate import (
        TinyPackedExperts,
        TinyRouter,
    )
    torch.manual_seed(seed)
    model = TinyCausal().eval()
    model.inner.mlp.gate = TinyRouter(hidden_size=16, num_experts=num_experts)
    model.inner.mlp.experts = TinyPackedExperts(num_experts=num_experts)
    return model


def test_expert_sampling_restores_and_annotates():
    model = _wide_moe()
    calib = torch.randint(0, 32, (2, 24))
    before = {
        n: getattr(model.inner.mlp.experts, a).detach().clone()
        for n, a in (("gate_up", "gate_up_proj"), ("down", "down_proj"))
    }
    stats, costs, unit_kls = measure_expert_unit_costs(
        model, None, calib, ["NVFP4", "FP8_DYNAMIC", "BF16"],
        expert_chunk=1, progress=False, expert_sample=3)
    (unit,) = unit_kls.values()
    samp = unit["_sampling"]
    assert samp["num_experts"] == 8 and samp["sampled"] == 3
    assert samp["scale"] == pytest.approx(8.0 / 3.0, rel=1e-3)
    assert unit["NVFP4"] > 0.0
    # Restoration exact even under slice-wise clone/restore.
    assert torch.equal(
        model.inner.mlp.experts.gate_up_proj.detach(), before["gate_up"])
    assert torch.equal(
        model.inner.mlp.experts.down_proj.detach(), before["down"])
    # Member shares still re-assemble the (scaled) unit KL.
    total = sum(costs[n]["NVFP4"]["predicted_dloss"] for n in EXPERT_NAMES)
    assert total == pytest.approx(unit["NVFP4"], rel=1e-6)


def test_expert_sampling_estimates_full_unit_kl():
    """Sampling all-but-none vs full: at S == E the sample path must equal the
    full path exactly (scale 1, same experts); at S < E the estimate should
    land within a loose factor of the full measurement on the tiny fixture
    (cross-expert additivity is exact only in fp32; this is a sanity band,
    the real gate is the 35B validation)."""
    model = _wide_moe(seed=13)
    calib = torch.randint(0, 32, (2, 24))
    _, _, full = measure_expert_unit_costs(
        model, None, calib, ["NVFP4", "BF16"], expert_chunk=1,
        progress=False)
    _, _, samp = measure_expert_unit_costs(
        model, None, calib, ["NVFP4", "BF16"], expert_chunk=1,
        progress=False, expert_sample=4)
    (kf,) = full.values()
    (ks,) = samp.values()
    assert 0.2 * kf["NVFP4"] < ks["NVFP4"] < 5.0 * kf["NVFP4"]


def test_max_units_and_filter():
    model = _wide_moe()
    calib = torch.randint(0, 32, (2, 24))
    _, _, unit_kls = measure_expert_unit_costs(
        model, None, calib, ["NVFP4", "BF16"], expert_chunk=1,
        progress=False, unit_filter="no-such-unit")
    assert unit_kls == {}
    _, _, unit_kls = measure_expert_unit_costs(
        model, None, calib, ["NVFP4", "BF16"], expert_chunk=1,
        progress=False, max_units=1)
    assert len(unit_kls) == 1
