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
    saliency_from_packed_moe,
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


# ---------------------------------------------------------------------------
# Packed-3D path (Qwen3.5/3.6 Qwen3_5MoeExperts layout)
# ---------------------------------------------------------------------------
class _FakeTopKRouter(nn.Module):
    """Qwen3_5MoeTopKRouter-shaped stub: `weight` is [num_experts, D]
    nn.Parameter (not an nn.Linear)."""
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_dim))

    def forward(self, x):
        x_flat = x.reshape(-1, self.hidden_dim)
        logits = F.linear(x_flat, self.weight)
        scores = F.softmax(logits, dtype=torch.float, dim=-1)
        topv, topi = torch.topk(scores, self.top_k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)
        return logits, topv.to(logits.dtype), topi


class _FakePackedExperts(nn.Module):
    """Minimal Qwen3_5MoeExperts-shaped stub carrying the attribute
    names (`gate_up_proj`, `down_proj`, `act_fn`, `num_experts`) and the
    same forward signature the patched forward expects."""
    def __init__(self, hidden_dim: int, inter_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * inter_dim, hidden_dim) * 0.02,
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_dim, inter_dim) * 0.02,
        )
        self.act_fn = F.silu

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            em = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(em.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit:
            e = e[0]
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(em[e])
            cur = hidden_states[token_idx]
            gate, up = F.linear(cur, self.gate_up_proj[e]).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            eo = F.linear(h, self.down_proj[e])
            routed = eo * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, routed.to(final.dtype))
        return final


class _FakeSparseMoeBlock(nn.Module):
    """Qwen3_5MoeSparseMoeBlock-shaped parent — has `gate` (TopKRouter)
    and `experts` (packed-3D). Router forward is invoked manually here
    to keep the test self-contained."""
    def __init__(self, hidden_dim: int, inter_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.gate = _FakeTopKRouter(hidden_dim, num_experts, top_k)
        self.experts = _FakePackedExperts(hidden_dim, inter_dim, num_experts)

    def forward(self, x):
        _, topv, topi = self.gate(x.reshape(-1, x.shape[-1]))
        return self.experts(x.reshape(-1, x.shape[-1]), topi, topv)


def _pure_python_reap_saliency(block: _FakeSparseMoeBlock, x: torch.Tensor):
    """Reference saliency via one eager re-run of each expert's forward;
    independent of the tracker so a mismatch here is a real bug."""
    x_flat = x.reshape(-1, x.shape[-1])
    _, topv, topi = block.gate(x_flat)
    em = F.one_hot(topi, num_classes=block.experts.num_experts).permute(2, 1, 0)
    means = {}
    maxes = {}
    counts = {}
    for e in range(block.experts.num_experts):
        top_k_pos, token_idx = torch.where(em[e])
        if token_idx.numel() == 0:
            means[e] = 0.0
            maxes[e] = 0.0
            counts[e] = 0
            continue
        cur = x_flat[token_idx]
        gate, up = F.linear(cur, block.experts.gate_up_proj[e]).chunk(2, dim=-1)
        h = block.experts.act_fn(gate) * up
        eo = F.linear(h, block.experts.down_proj[e])
        gate_vals = topv[token_idx, top_k_pos].to(torch.float64)
        norms = eo.to(torch.float64).norm(dim=-1)
        contrib = gate_vals * norms
        means[e] = float(contrib.mean().item())
        maxes[e] = float(contrib.max().item())
        counts[e] = int(contrib.numel())
    return means, maxes, counts


def test_saliency_from_packed_moe_discovers_block():
    torch.manual_seed(0)
    dim, inter, n_experts, top_k = 16, 24, 4, 2
    parent = nn.Module()
    parent.mlp = _FakeSparseMoeBlock(dim, inter, n_experts, top_k)
    # Intentionally throw in a non-MoE child with `gate`/`experts`
    # attributes that aren't 3D — discovery must ignore it.
    class _Decoy(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(dim, 8, bias=False)  # wrong shape
            self.experts = nn.Linear(dim, dim, bias=False)  # not packed
    parent.decoy = _Decoy()

    got = saliency_from_packed_moe(parent)
    assert len(got) == 1
    entry = got[0]
    assert entry["router_qname"] == "mlp.gate"
    assert entry["experts_qname"] == "mlp.experts"
    assert entry["num_experts"] == n_experts


def test_packed_saliency_patched_forward_matches_reference():
    torch.manual_seed(0)
    dim, inter, n_experts, top_k = 16, 24, 4, 2
    block = _FakeSparseMoeBlock(dim, inter, n_experts, top_k)
    parent = nn.Module()
    parent.mlp = block  # qnames become `mlp.gate`, `mlp.experts`
    x = torch.randn(50, dim)

    # Reference first — before any patching.
    ref_mean, ref_max, ref_count = _pure_python_reap_saliency(block, x)
    with torch.no_grad():
        ref_out = block(x)

    # Install tracker + patch + run.
    entries = saliency_from_packed_moe(parent)
    tracker = ExpertSaliencyTracker(
        parent, routers_and_experts=[], top_k=top_k,
        packed_moe_blocks=entries,
    )
    # Sanity: the patch marks the experts module.
    assert hasattr(block.experts, "_pq_saliency_patched")
    with torch.no_grad():
        got_out = block(x)

    # Forward output must be identical (patch is semantics-preserving).
    assert torch.allclose(got_out, ref_out, atol=1e-6), (
        (got_out - ref_out).abs().max()
    )

    # Saliency values must match the pure-Python reference to fp64
    # precision (we accumulate in fp64 in both).
    got_mean = tracker.saliency(reduction="mean")["mlp.gate"]
    got_max = tracker.saliency(reduction="max")["mlp.gate"]
    for e in range(n_experts):
        assert got_mean[e] == pytest.approx(ref_mean[e], abs=1e-10, rel=1e-8), (
            e, got_mean[e], ref_mean[e]
        )
        assert got_max[e] == pytest.approx(ref_max[e], abs=1e-10, rel=1e-8), (
            e, got_max[e], ref_max[e]
        )

    # remove_hooks() must revert the monkey-patch so the experts module
    # runs its own forward again afterwards.
    tracker.remove_hooks()
    assert not hasattr(block.experts, "_pq_saliency_patched")
    with torch.no_grad():
        after_out = block(x)
    assert torch.allclose(after_out, ref_out, atol=1e-6)


def test_packed_saliency_dead_experts_zero():
    """Rig the router so only expert 0 ever wins — experts 1..N-1
    must come out with S=0."""
    torch.manual_seed(1)
    dim, inter, n_experts, top_k = 8, 12, 4, 1
    block = _FakeSparseMoeBlock(dim, inter, n_experts, top_k)
    with torch.no_grad():
        block.gate.weight.zero_()
        block.gate.weight[0, :] = 1.0
    parent = nn.Module()
    parent.mlp = block
    x = torch.randn(30, dim).abs() + 0.1  # all-positive → expert 0 wins

    entries = saliency_from_packed_moe(parent)
    tracker = ExpertSaliencyTracker(
        parent, routers_and_experts=[], top_k=top_k,
        packed_moe_blocks=entries,
    )
    with torch.no_grad():
        _ = block(x)
    sal = tracker.saliency(reduction="mean")["mlp.gate"]
    assert sal[0] > 0.0, sal
    for e in range(1, n_experts):
        assert sal[e] == 0.0, (e, sal[e])
    tracker.remove_hooks()
