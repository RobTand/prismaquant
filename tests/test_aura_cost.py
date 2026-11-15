"""Check AURA format and routed-expert policies."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from prismaquant.aura_cost import _delta_w


def test_aura_delta_refuses_a_retired_codebook_rung():
    # A stale menu naming a retired codebook rung (archived 2026-09-25, #1304)
    # refuses; it is never priced from an RTN fallback.
    from prismaquant import format_registry as fr

    with pytest.raises(fr.RetiredFormatError, match="gridbook_lane"):
        _delta_w(
            "layer.q_proj",
            "NVFP4_CB_K16",
            torch.randn(2, 256),
            cache=None,
        )


def test_cost_ucb_z_charges_stderr(monkeypatch):
    from prismaquant.priced_cell import cost_entry_predicted_dloss

    stats = {"h_trace": 1.0}
    row = {"predicted_dloss": 0.010, "predicted_dloss_stderr": 0.002,
           "output_mse_measured": False}
    assert cost_entry_predicted_dloss(stats, row) == 0.010  # default: identical
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", "2")
    assert abs(cost_entry_predicted_dloss(stats, row) - 0.014) < 1e-15
    # rows without stderr (old payloads) are unaffected even with z set
    old = {"predicted_dloss": 0.010, "output_mse_measured": False}
    assert cost_entry_predicted_dloss(stats, old) == 0.010


from prismaquant.aura_cost import (
    _guard_packed_expert_coverage,
    _target_linears,
)
class _PackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 32, 16))
        self.down_proj = nn.Parameter(torch.zeros(2, 32, 32))


class _PackedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.experts = _PackedExperts()


class _Dsv4UnpackedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(32, 32, bias=False)
        self.up_proj = nn.Linear(32, 32, bias=False)
        self.down_proj = nn.Linear(32, 32, bias=False)


class _Dsv4UnpackedModel(nn.Module):
    """DSv4 live/probe shape: routed experts are per-expert Linears."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(32, 32, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.experts = nn.ModuleList(
            [_Dsv4UnpackedExpert(), _Dsv4UnpackedExpert()]
        )


def test_aura_guard_rejects_packed_experts_by_default():
    with pytest.raises(RuntimeError, match="packed-MoE expert costs"):
        _guard_packed_expert_coverage(_PackedModel())


def test_aura_guard_requires_explicit_omission_for_packed_experts():
    omitted = _guard_packed_expert_coverage(
        _PackedModel(),
        allow_omission=True,
    )

    assert omitted == [
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.mlp.experts.gate_up_proj",
    ]


def test_aura_guard_allows_dense_only_models():
    model = nn.Sequential(nn.Linear(16, 16, bias=False))

    assert _guard_packed_expert_coverage(model) == []


def test_aura_guard_rejects_profile_declared_dsv4_unpacked_experts():
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile

    with pytest.raises(RuntimeError, match="expert costs"):
        _guard_packed_expert_coverage(
            _Dsv4UnpackedModel(),
            DeepseekV4Profile(),
        )


def test_aura_smooth_targets_exclude_dsv4_unpacked_experts():
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile

    model = _Dsv4UnpackedModel()
    profile = DeepseekV4Profile()
    omitted = _guard_packed_expert_coverage(
        model,
        profile,
        allow_omission=True,
    )

    assert omitted == [
        f"model.layers.0.mlp.experts.{expert_id}.{projection}"
        for expert_id in range(2)
        for projection in ("down_proj", "gate_proj", "up_proj")
    ]
    assert list(_target_linears(model, profile=profile)) == [
        "model.layers.0.self_attn.q_proj"
    ]


def test_aura_guard_fails_closed_when_profile_cannot_classify_experts():
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile

    class BrokenProfile(DeepseekV4Profile):
        def packed_expert_format_group(self, qname: str):
            raise LookupError(f"classification unavailable for {qname}")

    with pytest.raises(RuntimeError, match="could not determine routed-expert"):
        _guard_packed_expert_coverage(_Dsv4UnpackedModel(), BrokenProfile())


def test_source_digest_mismatch_names_the_package_inputs():
    """A ``producer_source_sha256`` refusal must name files, not just digests (#2218)."""
    from pathlib import Path

    from prismaquant.aura_checkpoint_journal import _checkpoint_identity_mismatch

    from prismaquant import production_weight_cache as cache

    error = _checkpoint_identity_mismatch(
        field="producer_source_sha256", stored="0" * 64, expected="1" * 64)
    text = str(error)
    assert "AURA checkpoint identity mismatch at producer_source_sha256" in text
    assert "refusing reuse or recompute" in text
    assert f"hashed files under {Path(cache.__file__).resolve().parent}" in text
    assert "newest:" in text

    # Only the source digest is opaque enough to need the listing: every
    # other field's refusal message stays exactly as it was.
    untouched = _checkpoint_identity_mismatch(
        field="manifest.schema", stored="old", expected="new")
    assert "hashed files under" not in str(untouched)
