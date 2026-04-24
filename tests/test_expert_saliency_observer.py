"""Unit tests for `prismaquant.observers.expert_saliency`.

Builds a tiny hand-made MoE module that mirrors the HF nested layout
(`parent.gate` Linear + `parent.experts` nn.ModuleList of expert
submodules) and verifies:

1. The REAP saliency formula `S_j = mean[g_j · ||f_j||_2]` matches a
   pure-Python reference within fp64 accumulation tolerance.
2. Both `reduction="mean"` and `reduction="max"` return sensible,
   monotone-in-their-definition values.
3. The helper `saliency_from_moe_structure` correctly groups expert
   linears by (router, experts_parent), ignoring unrelated numeric
   segments in the qname path (the bug caught in initial dev).
4. Dead experts (never in top-k during calibration) emit S_j = 0.
5. `remove_hooks` is idempotent.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant.observers import (
    ExpertSaliencyTracker,
    saliency_from_moe_structure,
)


# ---------------------------------------------------------------------------
# Dummy MoE layer matching HF-style nested layout
# ---------------------------------------------------------------------------
class _DummyExpert(nn.Module):
    """One SwiGLU-ish expert: gate / up / down Linears, silu gate."""
    def __init__(self, dim: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter, bias=False)
        self.up_proj = nn.Linear(dim, inter, bias=False)
        self.down_proj = nn.Linear(inter, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _DummyMoeLayer(nn.Module):
    """Router + ModuleList-of-experts MoE layer.

    forward(x): for each token, top_k experts are selected by the router,
    each expert runs on its token subset, outputs are weighted by
    softmaxed gate probs and summed.
    """
    def __init__(self, dim: int, inter: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [_DummyExpert(dim, inter) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [tokens, dim]
        logits = self.gate(x)
        topk_v, topk_i = logits.topk(self.top_k, dim=-1)
        probs = F.softmax(topk_v, dim=-1)
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask_e = (topk_i == e).any(dim=-1)
            if not mask_e.any():
                continue
            # Routing weight for expert e per token (0 where not selected)
            gate_e = torch.zeros(x.size(0), device=x.device, dtype=probs.dtype)
            for k in range(self.top_k):
                gate_e = torch.where(topk_i[:, k] == e, probs[:, k], gate_e)
            tokens_for_e = x[mask_e]
            f_e = self.experts[e](tokens_for_e)
            out[mask_e] += gate_e[mask_e].unsqueeze(-1) * f_e
        return out


def _compute_reference_saliency(
    layer: _DummyMoeLayer,
    x: torch.Tensor,
) -> tuple[dict[int, float], dict[int, float], dict[int, int]]:
    """Pure-Python reference: mean and max of `g_j · ||f_j||` per expert.

    Returns (mean_saliency, max_saliency, active_count).
    """
    logits = layer.gate(x)
    topk_v, topk_i = logits.topk(layer.top_k, dim=-1)
    probs = F.softmax(topk_v, dim=-1)
    means: dict[int, float] = {}
    maxes: dict[int, float] = {}
    counts: dict[int, int] = {}
    for e in range(layer.num_experts):
        gate_e = torch.zeros(x.size(0), dtype=probs.dtype)
        for k in range(layer.top_k):
            gate_e = torch.where(topk_i[:, k] == e, probs[:, k], gate_e)
        active = gate_e > 0
        if not active.any():
            means[e] = 0.0
            maxes[e] = 0.0
            counts[e] = 0
            continue
        f_e = layer.experts[e](x[active])
        norms = f_e.to(torch.float64).norm(dim=-1)
        g_active = gate_e[active].to(torch.float64)
        contrib = g_active * norms
        means[e] = float(contrib.mean().item())
        maxes[e] = float(contrib.max().item())
        counts[e] = int(active.sum().item())
    return means, maxes, counts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_saliency_mean_matches_reference():
    torch.manual_seed(0)
    dim, inter, n_experts, top_k = 32, 64, 8, 2
    layer = _DummyMoeLayer(dim, inter, n_experts, top_k)
    # Wrap in a parent container so we can use nested qnames.
    parent = nn.Module()
    parent.mlp = layer  # type: ignore[attr-defined]

    tracker = ExpertSaliencyTracker(
        parent,
        routers_and_experts=[("mlp.gate", "mlp.experts", list(range(n_experts)))],
        top_k=top_k,
    )
    x = torch.randn(50, dim)
    with torch.no_grad():
        _ = layer(x)

    got_mean = tracker.saliency(reduction="mean")["mlp.gate"]
    got_max = tracker.saliency(reduction="max")["mlp.gate"]

    ref_mean, ref_max, ref_count = _compute_reference_saliency(layer, x)
    for e in range(n_experts):
        # Mean should match the reference to fp64 precision (the
        # observer accumulates in fp64, reference computes in fp64).
        assert got_mean[e] == pytest.approx(ref_mean[e], abs=1e-10, rel=1e-8), (
            e, got_mean[e], ref_mean[e]
        )
        # Max likewise should be exact (both are max of identical
        # floating-point sequences over the same set of tokens).
        assert got_max[e] == pytest.approx(ref_max[e], abs=1e-10, rel=1e-8), (
            e, got_max[e], ref_max[e]
        )
    tracker.remove_hooks()


def test_saliency_geomean_blend():
    torch.manual_seed(1)
    dim, inter, n_experts, top_k = 32, 64, 6, 2
    layer = _DummyMoeLayer(dim, inter, n_experts, top_k)
    parent = nn.Module()
    parent.mlp = layer  # type: ignore[attr-defined]

    tracker = ExpertSaliencyTracker(
        parent,
        routers_and_experts=[("mlp.gate", "mlp.experts", list(range(n_experts)))],
        top_k=top_k,
    )
    x = torch.randn(40, dim)
    with torch.no_grad():
        _ = layer(x)

    mean = tracker.saliency(reduction="mean")["mlp.gate"]
    maxv = tracker.saliency(reduction="max")["mlp.gate"]
    geo = tracker.saliency(reduction="max_mean_geomean")["mlp.gate"]
    for e in range(n_experts):
        expected = math.sqrt(mean[e] * maxv[e])
        assert geo[e] == pytest.approx(expected, abs=1e-9, rel=1e-8), (e, geo[e], expected)


def test_dead_experts_get_zero_saliency():
    # Force all top_k routing weight onto expert 0 by rigging the gate
    # matrix. Experts 1..N-1 never activate → all reductions should be 0.
    torch.manual_seed(2)
    dim, inter, n_experts, top_k = 16, 32, 4, 1
    layer = _DummyMoeLayer(dim, inter, n_experts, top_k)
    with torch.no_grad():
        # Make expert 0 dominate: large positive weight on expert 0's
        # logit, random for others
        layer.gate.weight.zero_()
        layer.gate.weight[0, :] = 1.0  # expert-0 logit = sum of x; biggest.
    parent = nn.Module()
    parent.mlp = layer  # type: ignore[attr-defined]

    tracker = ExpertSaliencyTracker(
        parent,
        routers_and_experts=[("mlp.gate", "mlp.experts", list(range(n_experts)))],
        top_k=top_k,
    )
    # Use positive inputs so expert-0 logit is always the winner
    x = torch.randn(30, dim).abs() + 0.1
    with torch.no_grad():
        _ = layer(x)

    sal = tracker.saliency(reduction="mean")["mlp.gate"]
    assert sal[0] > 0.0, sal
    for e in range(1, n_experts):
        assert sal[e] == 0.0, (e, sal[e])
    tracker.remove_hooks()


def test_helper_skips_projection_list_layout():
    # projection-list layout puts the eid AFTER the proj name
    # (experts.gate_up_proj.<eid>) — no per-eid hookable module there,
    # so the helper should skip it silently.
    expert_info = {
        "model.layers.0.mlp.experts.0.gate_proj": ("model.layers.0.mlp.gate", "0"),
        "model.layers.0.mlp.experts.1.up_proj": ("model.layers.0.mlp.gate", "1"),
        # projection-list layout — no per-eid submodule
        "model.layers.2.mlp.experts.gate_up_proj.3": ("model.layers.2.mlp.gate", "3"),
    }
    got = saliency_from_moe_structure(expert_info)
    by_router = {rq: (parent, ids) for rq, parent, ids in got}
    assert "model.layers.0.mlp.gate" in by_router
    assert by_router["model.layers.0.mlp.gate"] == (
        "model.layers.0.mlp.experts", [0, 1]
    )
    assert "model.layers.2.mlp.gate" not in by_router


def test_remove_hooks_idempotent():
    dim, inter, n_experts, top_k = 16, 32, 4, 2
    layer = _DummyMoeLayer(dim, inter, n_experts, top_k)
    parent = nn.Module()
    parent.mlp = layer  # type: ignore[attr-defined]
    tracker = ExpertSaliencyTracker(
        parent,
        routers_and_experts=[("mlp.gate", "mlp.experts", list(range(n_experts)))],
        top_k=top_k,
    )
    tracker.remove_hooks()
    tracker.remove_hooks()   # must not error


def test_helper_with_numeric_layer_number():
    # Leaf qname has the layer number (which is ALSO an integer, and
    # matches the eid regex) as well as the actual expert id. The helper
    # must only latch onto the numeric segment that immediately follows
    # an `experts` marker — not the layer number.
    expert_info = {
        # layer 0, expert 0 — both are "0"; the helper must pick the
        # SECOND "0" (the one following "experts")
        "model.layers.0.mlp.experts.0.gate_proj": ("model.layers.0.mlp.gate", "0"),
    }
    got = saliency_from_moe_structure(expert_info)
    assert got == [("model.layers.0.mlp.gate", "model.layers.0.mlp.experts", [0])]
