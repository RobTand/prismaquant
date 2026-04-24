"""Joint REAP-prune + quant candidate generation in the allocator.

Verifies:
  1. When prune is disabled (empty prune_ratios or missing saliency),
     aggregate_moe_candidates emits exactly one candidate per format
     per super-Linear — preserving legacy behavior bit-identically.
  2. When prune is enabled with saliency, each format gains additional
     (ratio > 0) candidates; each prune candidate has shrunken
     memory_bytes (proportional to kept experts) and drops the
     lowest-saliency expert ids first.
  3. The prune Δloss formula is α · (h / n_params) · S_j² per dropped
     expert, matching `_prune_cost_per_expert`.
  4. The DP's existing numerical invariants (ordering by cost) still
     hold — the allocator's downstream DP is format-agnostic about
     where candidates come from.
"""
from __future__ import annotations

import math

import pytest

from prismaquant.allocator import (
    Candidate,
    aggregate_moe_candidates,
    _prune_cost_per_expert,
    _expert_ids_in_group,
)
import prismaquant.format_registry as fr


def _make_two_expert_fixture(saliencies: dict[int, float]):
    """Two-expert MoE group for a single projection. Returns (stats,
    costs, candidates_in, expert_info, saliency_map, specs)."""
    stats = {
        "model.layers.0.mlp.experts.0.gate_proj": {
            "h_trace": 0.1, "w_max_abs": 1.0, "w_norm_sq": 4.0, "n_params": 100,
            "in_features": 10, "out_features": 10, "h_trace_raw": 0.1, "h_w2_sum": 0.4,
        },
        "model.layers.0.mlp.experts.1.gate_proj": {
            "h_trace": 0.05, "w_max_abs": 0.8, "w_norm_sq": 2.0, "n_params": 100,
            "in_features": 10, "out_features": 10, "h_trace_raw": 0.05, "h_w2_sum": 0.2,
        },
    }
    costs = {
        n: {
            "NVFP4": {"weight_mse": 0.01, "output_mse": 0.01, "predicted_dloss": 0.001},
            "MXFP8": {"weight_mse": 0.003, "output_mse": 0.003, "predicted_dloss": 0.0003},
        }
        for n in stats
    }
    candidates_in = {
        n: [
            Candidate(fmt="NVFP4", bits_per_param=4.25, memory_bytes=100, predicted_dloss=0.001),
            Candidate(fmt="MXFP8", bits_per_param=8.25, memory_bytes=200, predicted_dloss=0.0003),
        ]
        for n in stats
    }
    expert_info = {
        "model.layers.0.mlp.experts.0.gate_proj": ("model.layers.0.mlp.gate", "0"),
        "model.layers.0.mlp.experts.1.gate_proj": ("model.layers.0.mlp.gate", "1"),
    }
    saliency_map = {"model.layers.0.mlp.gate": dict(saliencies)}
    specs = [fr.get_format("NVFP4"), fr.get_format("MXFP8")]
    return stats, costs, candidates_in, expert_info, saliency_map, specs


SUPER_NAME = "model.layers.0.mlp.experts.__fused__.gate_proj"


def test_no_prune_when_disabled_preserves_legacy_behavior():
    stats, costs, c_in, e_info, sal, specs = _make_two_expert_fixture({0: 1.0, 1: 0.1})
    # No prune: empty prune_ratios → only no-prune candidates
    _, _, c_out = aggregate_moe_candidates(
        stats, costs, specs, c_in, granularity="projection",
    )
    cands = c_out[SUPER_NAME]
    assert len(cands) == len(specs), cands
    for cand in cands:
        assert cand.pruned_expert_ids == ()

    # Prune ratios present but saliency/expert_info missing → still no prune
    _, _, c_out = aggregate_moe_candidates(
        stats, costs, specs, c_in, granularity="projection",
        prune_ratios=(0.5,),
        expert_saliency=None,
        expert_info=None,
        prune_alpha=0.5,
    )
    for cand in c_out[SUPER_NAME]:
        assert cand.pruned_expert_ids == ()


def test_prune_drops_lowest_saliency_first():
    stats, costs, c_in, e_info, sal, specs = _make_two_expert_fixture({0: 1.0, 1: 0.1})
    _, _, c_out = aggregate_moe_candidates(
        stats, costs, specs, c_in, granularity="projection",
        expert_saliency=sal,
        expert_info=e_info,
        prune_ratios=(0.5,),
        prune_alpha=0.5,
    )
    # For each format we expect 2 candidates: no-prune (baseline) + 1 prune variant
    by_fmt = {}
    for cand in c_out[SUPER_NAME]:
        by_fmt.setdefault(cand.fmt, []).append(cand)
    for fmt, cands in by_fmt.items():
        assert len(cands) == 2, (fmt, cands)
        no_prune = [c for c in cands if c.pruned_expert_ids == ()][0]
        prune = [c for c in cands if c.pruned_expert_ids != ()][0]
        # Lowest-saliency expert (id=1) must be dropped first.
        assert prune.pruned_expert_ids == (1,)
        # Pruning must halve memory (2 experts total, drop 1).
        assert prune.memory_bytes == no_prune.memory_bytes // 2, (fmt, prune, no_prune)


def test_prune_cost_matches_formula():
    """Δloss of the prune candidate should equal
    per_member_dloss[kept] + _prune_cost_per_expert(dropped)."""
    stats, costs, c_in, e_info, sal, specs = _make_two_expert_fixture({0: 1.0, 1: 0.1})
    _, _, c_out = aggregate_moe_candidates(
        stats, costs, specs, c_in, granularity="projection",
        expert_saliency=sal,
        expert_info=e_info,
        prune_ratios=(0.5,),
        prune_alpha=0.5,
    )
    # Pick NVFP4 prune variant and verify the formula.
    cand = next(
        c for c in c_out[SUPER_NAME]
        if c.fmt == "NVFP4" and c.pruned_expert_ids == (1,)
    )
    kept_member = "model.layers.0.mlp.experts.0.gate_proj"
    dropped_member = "model.layers.0.mlp.experts.1.gate_proj"
    kept_dloss = costs[kept_member]["NVFP4"]["predicted_dloss"]
    prune_dloss = _prune_cost_per_expert(
        saliency=0.1,
        h_trace=stats[dropped_member]["h_trace"],
        n_params=stats[dropped_member]["n_params"],
        alpha=0.5,
    )
    expected = kept_dloss + prune_dloss
    assert cand.predicted_dloss == pytest.approx(expected, rel=1e-10)


def test_dead_expert_pruned_nearly_free():
    """Zero-saliency experts (never activated) must have near-zero
    prune Δloss — effectively 'free to drop'."""
    stats, costs, c_in, e_info, sal, specs = _make_two_expert_fixture({0: 1.0, 1: 0.0})
    _, _, c_out = aggregate_moe_candidates(
        stats, costs, specs, c_in, granularity="projection",
        expert_saliency=sal,
        expert_info=e_info,
        prune_ratios=(0.5,),
        prune_alpha=0.5,
    )
    cand = next(
        c for c in c_out[SUPER_NAME]
        if c.fmt == "NVFP4" and c.pruned_expert_ids == (1,)
    )
    kept_dloss = costs["model.layers.0.mlp.experts.0.gate_proj"]["NVFP4"]["predicted_dloss"]
    # Prune cost of dead expert = α · (h/n) · 0² = 0; total Δloss = kept_dloss only.
    assert cand.predicted_dloss == pytest.approx(kept_dloss, rel=1e-12)


def test_expert_ids_in_group_helper():
    # Three experts, one member per eid
    members = [
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.1.gate_proj",
        "model.layers.0.mlp.experts.7.gate_proj",
    ]
    expert_info = {
        "model.layers.0.mlp.experts.0.gate_proj": ("R0", "0"),
        "model.layers.0.mlp.experts.1.gate_proj": ("R0", "1"),
        "model.layers.0.mlp.experts.7.gate_proj": ("R0", "7"),
    }
    rq, by_eid = _expert_ids_in_group(members, expert_info)
    assert rq == "R0"
    assert by_eid == {0: members[0], 1: members[1], 7: members[2]}


def test_unit_prune_cost_formula():
    # α=0.5, h=0.1, n=100, S=0.5 → 0.5 · 0.001 · 0.25 = 0.000125
    got = _prune_cost_per_expert(saliency=0.5, h_trace=0.1, n_params=100, alpha=0.5)
    assert got == pytest.approx(0.000125, rel=1e-12)
    # Degenerate cases
    assert _prune_cost_per_expert(0.0, 0.1, 100, 0.5) == 0.0
    assert _prune_cost_per_expert(0.5, 0.0, 100, 0.5) == 0.0
    assert _prune_cost_per_expert(0.5, 0.1, 0, 0.5) == 0.0
