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
    add_packed_prune_candidates,
    aggregate_moe_candidates,
    apply_consensus_prune,
    build_prune_manifest,
    expand_moe_assignment,
    _packed_entry_router_qname,
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


# ---------------------------------------------------------------------------
# Manifest + expansion (the handoff from allocator to exporter)
# ---------------------------------------------------------------------------
def _four_expert_fixture():
    """4-expert MoE, 3 projections each (gate/up/down). Same router."""
    members_gate = [f"model.layers.0.mlp.experts.{i}.gate_proj" for i in range(4)]
    members_up = [f"model.layers.0.mlp.experts.{i}.up_proj" for i in range(4)]
    members_down = [f"model.layers.0.mlp.experts.{i}.down_proj" for i in range(4)]
    stats_ext = {
        "model.layers.0.mlp.experts.__fused__.gate_proj": {"_fused_members": members_gate},
        "model.layers.0.mlp.experts.__fused__.up_proj": {"_fused_members": members_up},
        "model.layers.0.mlp.experts.__fused__.down_proj": {"_fused_members": members_down},
    }
    expert_info = {}
    for proj_members in (members_gate, members_up, members_down):
        for i, m in enumerate(proj_members):
            expert_info[m] = ("model.layers.0.mlp.gate", str(i))
    return stats_ext, expert_info, members_gate, members_up, members_down


def test_manifest_single_projection_layer_granularity():
    """At layer granularity there's one super-Linear per router — the
    manifest is trivially the DP's decision."""
    stats_ext, einfo, mg, mu, md = _four_expert_fixture()
    # Fuse everything into one super-Linear (layer-granularity).
    stats_ext = {
        "model.layers.0.mlp.experts.__fused__.__all__": {
            "_fused_members": mg + mu + md
        }
    }
    pruned_map = {
        "model.layers.0.mlp.experts.__fused__.__all__": (1, 3),
    }
    manifest, warnings = build_prune_manifest(pruned_map, stats_ext, einfo)
    assert warnings == []
    entry = manifest["model.layers.0.mlp.gate"]
    assert entry["num_experts_orig"] == 4
    assert entry["num_experts_kept"] == 2
    assert entry["pruned_expert_ids"] == [1, 3]
    assert entry["kept_expert_ids"] == [0, 2]
    assert entry["orig_to_new_eid"] == {"0": 0, "2": 1}


def test_manifest_projection_granularity_intersection():
    """With 3 super-Linears per router disagreeing on the drop set, the
    consensus is their intersection; a warning lists the disagreement."""
    stats_ext, einfo, *_ = _four_expert_fixture()
    # gate wants {1, 2}, up wants {1, 3}, down wants {1} → intersection = {1}
    pruned_map = {
        "model.layers.0.mlp.experts.__fused__.gate_proj": (1, 2),
        "model.layers.0.mlp.experts.__fused__.up_proj": (1, 3),
        "model.layers.0.mlp.experts.__fused__.down_proj": (1,),
    }
    manifest, warnings = build_prune_manifest(pruned_map, stats_ext, einfo)
    entry = manifest["model.layers.0.mlp.gate"]
    assert entry["pruned_expert_ids"] == [1]
    assert entry["kept_expert_ids"] == [0, 2, 3]
    assert entry["orig_to_new_eid"] == {"0": 0, "2": 1, "3": 2}
    assert len(warnings) == 1
    assert "additional-wanted=[2, 3]" in warnings[0]


def test_apply_consensus_prune_rewrites_drops():
    """After manifest build, each super-Linear's pruned_expert_ids must
    be coerced to the consensus so expansion + sidecar match."""
    stats_ext, einfo, *_ = _four_expert_fixture()
    pruned_map = {
        "model.layers.0.mlp.experts.__fused__.gate_proj": (1, 2),
        "model.layers.0.mlp.experts.__fused__.up_proj": (1, 3),
        "model.layers.0.mlp.experts.__fused__.down_proj": (1,),
    }
    manifest, _ = build_prune_manifest(pruned_map, stats_ext, einfo)
    coerced = apply_consensus_prune(pruned_map, manifest, stats_ext, einfo)
    for super_name, dropped in coerced.items():
        assert dropped == (1,), (super_name, dropped)


def test_expand_moe_assignment_drops_pruned_leaves():
    """expand_moe_assignment must omit per-expert leaves whose eid is
    in the super-Linear's pruned_expert_ids — those weights shouldn't
    appear in layer_config.json at all."""
    stats_ext, einfo, mg, mu, md = _four_expert_fixture()
    assignment = {
        "model.layers.0.mlp.experts.__fused__.gate_proj": "NVFP4",
        "model.layers.0.mlp.experts.__fused__.up_proj": "NVFP4",
        "model.layers.0.mlp.experts.__fused__.down_proj": "NVFP4",
        "model.layers.0.self_attn.q_proj": "MXFP8",  # non-MoE — untouched
    }
    pruned_map = {
        "model.layers.0.mlp.experts.__fused__.gate_proj": (1, 3),
        "model.layers.0.mlp.experts.__fused__.up_proj": (1, 3),
        "model.layers.0.mlp.experts.__fused__.down_proj": (1, 3),
    }
    expanded = expand_moe_assignment(
        assignment, stats_ext, pruned_map=pruned_map, expert_info=einfo,
    )
    # Non-MoE entry survives unchanged.
    assert expanded["model.layers.0.self_attn.q_proj"] == "MXFP8"
    # Dropped expert leaves are absent from the expanded dict.
    for proj_members in (mg, mu, md):
        assert proj_members[0] in expanded
        assert proj_members[2] in expanded
        assert proj_members[1] not in expanded, expanded
        assert proj_members[3] not in expanded, expanded


def test_empty_prune_map_is_noop():
    """With no pruned_map, expand behaves exactly as before — all
    members retained under the super-Linear's format."""
    stats_ext, einfo, mg, _mu, _md = _four_expert_fixture()
    stats_ext_single = {
        "model.layers.0.mlp.experts.__fused__.gate_proj": {"_fused_members": mg},
    }
    assignment = {"model.layers.0.mlp.experts.__fused__.gate_proj": "NVFP4"}
    expanded = expand_moe_assignment(assignment, stats_ext_single)
    assert set(expanded) == set(mg)
    assert all(expanded[m] == "NVFP4" for m in mg)


# ---------------------------------------------------------------------------
# Packed-3D prune (Qwen3.5-style single packed entry per projection)
# ---------------------------------------------------------------------------
def test_packed_entry_router_qname():
    assert (
        _packed_entry_router_qname("model.layers.0.mlp.experts.gate_up_proj")
        == "model.layers.0.mlp.gate"
    )
    assert (
        _packed_entry_router_qname("model.layers.3.mlp.experts.down_proj")
        == "model.layers.3.mlp.gate"
    )
    # Not a packed pattern:
    assert _packed_entry_router_qname(
        "model.layers.0.mlp.experts.0.gate_proj"
    ) is None
    assert _packed_entry_router_qname("lm_head.weight") is None


def _packed_stat_fixture():
    """One packed entry with 4 experts, same params/Fisher as real Qwen3.5."""
    name = "model.layers.0.mlp.experts.gate_up_proj"
    stats = {
        name: {
            "h_trace": 4.0,          # per-expert after split: 1.0
            "n_params": 400,         # per-expert: 100
            "num_experts": 4,
            "in_features": 8,
            "out_features": 10,
            "w_max_abs": 1.0,
            "w_norm_sq": 4.0,
        }
    }
    router = "model.layers.0.mlp.gate"
    saliencies = {router: {0: 1.0, 1: 0.1, 2: 0.5, 3: 0.0}}
    return name, stats, router, saliencies


def test_add_packed_prune_candidates_emits_variants():
    name, stats, router, sal = _packed_stat_fixture()
    # Single baseline candidate per format.
    base_nvfp4 = Candidate(
        fmt="NVFP4", bits_per_param=4.25, memory_bytes=1000,
        predicted_dloss=0.010,
    )
    candidates = {name: [base_nvfp4]}
    n_ext = add_packed_prune_candidates(
        candidates, stats,
        expert_saliency=sal, prune_ratios=(0.25, 0.5), prune_alpha=0.5,
    )
    assert n_ext == 1
    cands = candidates[name]
    # baseline + 2 prune-variants.
    assert len(cands) == 3
    no_prune = [c for c in cands if c.pruned_expert_ids == ()]
    prune_vars = [c for c in cands if c.pruned_expert_ids != ()]
    assert len(no_prune) == 1 and no_prune[0] is base_nvfp4
    assert len(prune_vars) == 2
    # Drop-order: lowest-saliency first. Sal = {0:1.0, 1:0.1, 2:0.5, 3:0.0}
    # → order by prune-cost ascending = [3 (S=0), 1 (S=0.1), 2 (S=0.5), 0 (S=1.0)]
    # 25% of 4 = 1 drop → {3}.
    # 50% of 4 = 2 drops → {3, 1} (sorted).
    ratios_found = {c.pruned_expert_ids: c for c in prune_vars}
    assert (3,) in ratios_found
    assert (1, 3) in ratios_found
    # Memory scales with kept_frac.
    assert ratios_found[(3,)].memory_bytes == 750, ratios_found[(3,)]
    assert ratios_found[(1, 3)].memory_bytes == 500, ratios_found[(1, 3)]


def test_add_packed_prune_noop_without_saliency():
    name, stats, _router, _sal = _packed_stat_fixture()
    base = Candidate(fmt="NVFP4", bits_per_param=4.25, memory_bytes=1000,
                     predicted_dloss=0.01)
    candidates = {name: [base]}
    # No saliency → no extension.
    n_ext = add_packed_prune_candidates(
        candidates, stats, expert_saliency={},
        prune_ratios=(0.25,), prune_alpha=0.5,
    )
    assert n_ext == 0
    assert candidates[name] == [base]


def test_add_packed_prune_ignores_non_packed_entries():
    # Mix in a dense-Linear and a super-Linear alongside a packed entry.
    packed = "model.layers.0.mlp.experts.gate_up_proj"
    dense = "model.layers.0.self_attn.q_proj"
    super_name = "model.layers.0.mlp.experts.__fused__.gate_proj"
    stats = {
        packed: {"h_trace": 4.0, "n_params": 400, "num_experts": 4,
                 "in_features": 8, "out_features": 10,
                 "w_max_abs": 1.0, "w_norm_sq": 4.0},
        dense:  {"h_trace": 1.0, "n_params": 100, "in_features": 10,
                 "out_features": 10, "w_max_abs": 1.0, "w_norm_sq": 1.0},
        super_name: {"h_trace": 2.0, "n_params": 200,
                     "_fused_members": ["a", "b"]},
    }
    candidates = {
        packed: [Candidate("NVFP4", 4.25, 1000, 0.01)],
        dense:  [Candidate("NVFP4", 4.25, 100, 0.001)],
        super_name: [Candidate("NVFP4", 4.25, 200, 0.002)],
    }
    sal = {"model.layers.0.mlp.gate": {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}}
    n_ext = add_packed_prune_candidates(
        candidates, stats, expert_saliency=sal,
        prune_ratios=(0.25,), prune_alpha=0.5,
    )
    assert n_ext == 1  # only the packed entry
    assert len(candidates[packed]) == 2
    assert len(candidates[dense]) == 1  # untouched
    assert len(candidates[super_name]) == 1  # untouched


def test_add_packed_prune_uses_per_expert_fisher_when_available():
    """When the probe decomposed h_trace per expert, the prune-cost
    computation must use it (not the uniform h_total/E fallback).
    Verify via a skewed per-expert Fisher that reverses the drop order
    that uniform-Fisher would pick on the same saliencies."""
    name = "model.layers.0.mlp.experts.gate_up_proj"
    router = "model.layers.0.mlp.gate"
    # Equal saliencies → uniform-Fisher path ties on eid order; skewed
    # per-expert Fisher should break the tie by picking the largest
    # per-expert h (cost-expensive to drop) LAST. Equivalently, smallest
    # per-expert h gets dropped first.
    stats = {
        name: {
            "h_trace": 4.0,
            "n_params": 400,
            "num_experts": 4,
            "in_features": 8,
            "out_features": 10,
            # Per-expert Fisher: expert 2 has the smallest trace → it
            # should be the cheapest to drop. All saliencies equal so
            # Fisher is the tiebreaker.
            "h_trace_per_expert": [1.0, 1.0, 0.1, 1.0],
        }
    }
    sal = {router: {e: 1.0 for e in range(4)}}
    base = Candidate("NVFP4", 4.25, 1000, 0.01)
    candidates = {name: [base]}
    add_packed_prune_candidates(
        candidates, stats, expert_saliency=sal,
        prune_ratios=(0.25,), prune_alpha=0.5,
    )
    # One prune candidate (25% = drop 1 of 4). The dropped eid must be
    # 2 (smallest per-expert h, so cheapest prune Δloss).
    prune_v = [c for c in candidates[name] if c.pruned_expert_ids]
    assert len(prune_v) == 1
    assert prune_v[0].pruned_expert_ids == (2,), prune_v[0]


def test_build_prune_manifest_resolves_packed_entries():
    """Packed-entry pruned_map entries (no `_fused_members`) must
    still produce a router-keyed manifest by stripping the qname."""
    stats_ext = {
        "model.layers.0.mlp.experts.gate_up_proj": {"num_experts": 4},
        "model.layers.0.mlp.experts.down_proj":    {"num_experts": 4},
    }
    pruned_map = {
        "model.layers.0.mlp.experts.gate_up_proj": (1, 3),
        "model.layers.0.mlp.experts.down_proj":    (1, 3),
    }
    manifest, warnings = build_prune_manifest(pruned_map, stats_ext, expert_info={})
    assert warnings == []
    entry = manifest["model.layers.0.mlp.gate"]
    assert entry["num_experts_orig"] == 4
    assert entry["pruned_expert_ids"] == [1, 3]
    assert entry["kept_expert_ids"] == [0, 2]
