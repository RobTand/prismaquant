"""Joint REAP-prune + quant candidate generation in the allocator.

Verifies:
  1. When prune is disabled (empty prune_ratios or missing saliency),
     aggregate_moe_candidates emits exactly one candidate per format
     per super-Linear — preserving legacy behavior bit-identically.
  2. When prune is enabled with saliency, each format gains additional
     (ratio > 0) candidates; each prune candidate has shrunken
     memory_bytes (proportional to kept experts) and drops the
     lowest-saliency expert ids first.
  3. The prune Δloss formula is α · S_j per dropped expert, where S_j
     is the probe's REAP dropout-loss estimate, matching
     `_prune_cost_per_expert`.
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
    apply_consensus_prune,
    apply_global_prune_ratio,
    apply_nested_global_prune_ratio,
    build_prune_manifest,
    compute_achieved,
    compute_assignment_predicted_dloss,
    compute_max_prune_ratio,
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


def test_nested_global_prune_ratio_filters_superlinear_candidates():
    stats, costs, c_in, e_info, sal, specs = _make_two_expert_fixture({0: 1.0, 1: 0.1})
    stats_ext, _, c_out = aggregate_moe_candidates(
        stats, costs, specs, c_in, granularity="projection",
        expert_saliency=sal,
        expert_info=e_info,
        prune_ratios=(0.5,),
        prune_alpha=0.5,
    )

    no_prune, warnings = apply_nested_global_prune_ratio(c_out, stats_ext, 0.0)
    assert warnings == []
    assert no_prune is not None
    assert all(c.pruned_expert_ids == () for c in no_prune[SUPER_NAME])

    half_prune, warnings = apply_nested_global_prune_ratio(c_out, stats_ext, 0.5)
    assert warnings == []
    assert half_prune is not None
    assert {c.pruned_expert_ids for c in half_prune[SUPER_NAME]} == {(1,)}

    infeasible, warnings = apply_nested_global_prune_ratio(c_out, stats_ext, 1.0)
    assert infeasible is None
    assert warnings


def test_prune_ratio_uses_floor_not_round():
    """Ratios are user caps. R=0.375 over four experts should drop one
    expert, not round 1.5 up to two."""
    stats, costs, c_in, e_info, sal, specs = _four_expert_projection_cost_fixture()
    _, _, c_out = aggregate_moe_candidates(
        stats, costs, specs, c_in, granularity="projection",
        expert_saliency=sal,
        expert_info=e_info,
        prune_ratios=(0.375,),
        prune_alpha=1.0,
    )
    prune = next(
        c for c in c_out["model.layers.0.mlp.experts.__fused__.gate_proj"]
        if c.fmt == "NVFP4" and c.pruned_expert_ids
    )
    assert prune.pruned_expert_ids == (3,)


def test_partial_saliency_disables_projection_prune_variants():
    """Missing saliency means observer coverage is incomplete, not that
    the missing expert is zero-cost to drop."""
    stats, costs, c_in, e_info, _sal, specs = _make_two_expert_fixture({0: 1.0})
    partial = {"model.layers.0.mlp.gate": {0: 1.0}}  # expert 1 missing
    _, _, c_out = aggregate_moe_candidates(
        stats, costs, specs, c_in, granularity="projection",
        expert_saliency=partial,
        expert_info=e_info,
        prune_ratios=(0.5,),
        prune_alpha=0.5,
    )
    assert all(c.pruned_expert_ids == () for c in c_out[SUPER_NAME])


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
    # Prune cost of dead expert = α · 0 = 0; total Δloss = kept_dloss only.
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
    # Saliency is already REAP dropout loss units, so α=0.5, S=0.5 → 0.25.
    got = _prune_cost_per_expert(saliency=0.5, h_trace=0.1, n_params=100, alpha=0.5)
    assert got == pytest.approx(0.25, rel=1e-12)
    # Degenerate cases
    assert _prune_cost_per_expert(0.0, 0.1, 100, 0.5) == 0.0
    # h_trace and n_params are intentionally ignored by the current formula.
    assert _prune_cost_per_expert(0.5, 0.0, 100, 0.5) == pytest.approx(0.25)
    assert _prune_cost_per_expert(0.5, 0.1, 0, 0.5) == pytest.approx(0.25)


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


def _four_expert_projection_cost_fixture():
    members = [f"model.layers.0.mlp.experts.{i}.gate_proj" for i in range(4)]
    stats = {
        m: {
            "h_trace": 0.1,
            "w_max_abs": 1.0,
            "w_norm_sq": 1.0,
            "n_params": 100,
            "in_features": 10,
            "out_features": 10,
            "h_trace_raw": 0.1,
            "h_w2_sum": 0.1,
        }
        for m in members
    }
    costs = {
        m: {
            "NVFP4": {"weight_mse": 0.01, "output_mse": 0.01, "predicted_dloss": 0.001},
        }
        for m in members
    }
    candidates = {
        m: [Candidate("NVFP4", 4.25, 100, 0.001)]
        for m in members
    }
    expert_info = {
        m: ("model.layers.0.mlp.gate", str(i))
        for i, m in enumerate(members)
    }
    saliency = {"model.layers.0.mlp.gate": {0: 10.0, 1: 5.0, 2: 1.0, 3: 0.0}}
    return stats, costs, candidates, expert_info, saliency, [fr.get_format("NVFP4")]


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


def test_apply_global_prune_ratio_replaces_candidates():
    """At the constrained single-ratio path every packed entry's
    candidate list is REPLACED (not appended) — vLLM demands uniform
    num_experts_kept, so the DP must be unable to pick 'no prune'."""
    name, stats, router, sal = _packed_stat_fixture()
    # Saliencies: 0→1.0, 1→0.1, 2→0.5, 3→0.0
    # Ascending-cost drop order: 3, 1, 2, 0
    base_nvfp4 = Candidate(
        fmt="NVFP4", bits_per_param=4.25, memory_bytes=1000,
        predicted_dloss=0.010,
    )
    base_mxfp8 = Candidate(
        fmt="MXFP8", bits_per_param=8.25, memory_bytes=2000,
        predicted_dloss=0.003,
    )
    candidates = {name: [base_nvfp4, base_mxfp8]}
    n = apply_global_prune_ratio(
        candidates, stats, sal, global_ratio=0.5, prune_alpha=1.0,
    )
    assert n == 1
    # No baseline left — just 2 prune variants (one per format).
    assert len(candidates[name]) == 2
    for c in candidates[name]:
        assert c.pruned_expert_ids == (1, 3), c
    # Memory is computed from first principles:
    #   pruned_mem = bits_per_param × n_params_total × kept_frac / 8
    # n_params_total=400 (fixture), kept_frac=0.5
    # NVFP4: 4.25 * 400 * 0.5 / 8 = 106.25 → int(106)
    # MXFP8: 8.25 * 400 * 0.5 / 8 = 206.25 → int(206)
    # This replaces the (wrong) per-expert memory_bytes from
    # build_candidates for packed entries. bpp field still shrinks
    # linearly with kept_frac.
    nv = next(c for c in candidates[name] if c.fmt == "NVFP4")
    mx = next(c for c in candidates[name] if c.fmt == "MXFP8")
    assert nv.memory_bytes == 106, nv
    assert nv.bits_per_param == pytest.approx(2.125)
    assert mx.memory_bytes == 206, mx
    assert mx.bits_per_param == pytest.approx(4.125)


def test_apply_global_prune_ratio_uses_floor_not_round():
    name, stats, _router, sal = _packed_stat_fixture()
    candidates = {
        name: [Candidate("NVFP4", 4.25, 1000, 0.010)]
    }
    n = apply_global_prune_ratio(
        candidates, stats, sal, global_ratio=0.375, prune_alpha=1.0,
    )
    assert n == 1
    # floor(4 * 0.375) = 1, so only the lowest-saliency expert is dropped.
    assert candidates[name][0].pruned_expert_ids == (3,)


def test_apply_global_prune_ratio_requires_complete_saliency():
    """A partial saliency map is an observer/cache gap, not evidence that
    missing experts are dead. The packed global-prune path must skip
    such entries instead of treating missing saliency as zero-cost."""
    name, stats, router, sal = _packed_stat_fixture()
    sal[router] = {0: 1.0, 1: 0.1}  # experts 2 and 3 missing
    base = Candidate(
        fmt="NVFP4", bits_per_param=4.25, memory_bytes=1000,
        predicted_dloss=0.010,
    )
    candidates = {name: [base]}
    n = apply_global_prune_ratio(
        candidates, stats, sal, global_ratio=0.5, prune_alpha=1.0,
    )
    assert n == 0
    assert candidates[name] == [base]


def test_compute_achieved_uses_exact_pruned_candidate_memory():
    """Regression: achieved bits for a prune-aware assignment must use
    the chosen Candidate's shrunken memory, not the no-prune format map."""
    name = "model.layers.0.mlp.experts.gate_up_proj"
    stats = {
        name: {
            "n_params": 400,
            "in_features": 8,
            "out_features": 10,
            "_memory_bytes_by_format": {"NVFP4": 213},  # no-prune footprint
        }
    }
    candidates = {
        name: [
            Candidate("NVFP4", 4.25, 213, 0.010, ()),
            Candidate("NVFP4", 2.125, 106, 0.020, (1, 3)),
        ]
    }
    achieved, _ = compute_achieved(
        stats,
        {name: "NVFP4"},
        {"NVFP4": fr.get_format("NVFP4")},
        candidates=candidates,
        pruned_map={name: (1, 3)},
    )
    assert achieved == pytest.approx(8.0 * 106 / 400)


def test_apply_global_prune_ratio_noop_without_saliency():
    name, stats, _router, _sal = _packed_stat_fixture()
    base = Candidate(fmt="NVFP4", bits_per_param=4.25, memory_bytes=1000,
                     predicted_dloss=0.01)
    candidates = {name: [base]}
    n = apply_global_prune_ratio(
        candidates, stats, expert_saliency={},
        global_ratio=0.25, prune_alpha=1.0,
    )
    assert n == 0
    assert candidates[name] == [base]


def test_apply_global_prune_ratio_zero_is_noop():
    name, stats, _router, sal = _packed_stat_fixture()
    base = Candidate(fmt="NVFP4", bits_per_param=4.25, memory_bytes=1000,
                     predicted_dloss=0.01)
    candidates = {name: [base]}
    n = apply_global_prune_ratio(
        candidates, stats, sal, global_ratio=0.0, prune_alpha=1.0,
    )
    assert n == 0
    assert candidates[name] == [base]


def test_apply_global_prune_ratio_ignores_non_packed_entries():
    packed = "model.layers.0.mlp.experts.gate_up_proj"
    dense = "model.layers.0.self_attn.q_proj"
    stats = {
        packed: {"h_trace": 4.0, "n_params": 400, "num_experts": 4,
                 "in_features": 8, "out_features": 10,
                 "w_max_abs": 1.0, "w_norm_sq": 4.0},
        dense:  {"h_trace": 1.0, "n_params": 100, "in_features": 10,
                 "out_features": 10, "w_max_abs": 1.0, "w_norm_sq": 1.0},
    }
    sal = {"model.layers.0.mlp.gate": {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}}
    candidates = {
        packed: [Candidate("NVFP4", 4.25, 1000, 0.01)],
        dense:  [Candidate("NVFP4", 4.25, 100, 0.001)],
    }
    n = apply_global_prune_ratio(
        candidates, stats, sal, global_ratio=0.25, prune_alpha=1.0,
    )
    assert n == 1  # only packed
    # Dense untouched (same Candidate object identity).
    assert candidates[dense][0] is candidates[dense][0]
    assert len(candidates[dense]) == 1
    # Packed now has prune variants, no baseline.
    for c in candidates[packed]:
        assert c.pruned_expert_ids != ()


def test_compute_max_prune_ratio():
    stats = {
        "model.layers.0.mlp.experts.gate_up_proj": {"num_experts": 256},
        "model.layers.1.mlp.experts.gate_up_proj": {"num_experts": 256},
        # Dense Linears ignored.
        "model.layers.0.self_attn.q_proj": {"n_params": 1000},
    }
    # top_k=8, E=256 → max_r = (256-8)/256 = 0.96875
    assert compute_max_prune_ratio(stats, top_k=8) == pytest.approx(0.96875)
    # Tighter top_k squeezes max_r.
    assert compute_max_prune_ratio(stats, top_k=200) == pytest.approx((256 - 200) / 256)


def test_compute_max_prune_ratio_rejects_top_k_over_experts():
    stats = {"model.layers.0.mlp.experts.gate_up_proj": {"num_experts": 4}}
    with pytest.raises(ValueError, match="top_k=8"):
        compute_max_prune_ratio(stats, top_k=8)


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


def test_build_prune_manifest_uniform_kept_is_opt_in():
    """Default manifest construction must not add unscored drops for
    routers the DP did not prune. The packed serving path may request
    uniform_kept=True, but that behavior is explicit and auditable."""
    stats_ext = {
        "model.layers.0.mlp.experts.gate_up_proj": {"num_experts": 4},
        "model.layers.1.mlp.experts.gate_up_proj": {"num_experts": 4},
    }
    sal = {
        "model.layers.0.mlp.gate": {0: 1.0, 1: 0.1, 2: 0.5, 3: 0.0},
        "model.layers.1.mlp.gate": {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6},
    }
    pruned_map = {"model.layers.0.mlp.experts.gate_up_proj": (1, 3)}

    manifest, warnings = build_prune_manifest(
        pruned_map, stats_ext, expert_info={}, expert_saliency=sal,
    )
    assert warnings == []
    assert set(manifest) == {"model.layers.0.mlp.gate"}

    manifest_u, warnings_u = build_prune_manifest(
        pruned_map, stats_ext, expert_info={}, expert_saliency=sal,
        uniform_kept=True,
    )
    assert set(manifest_u) == {
        "model.layers.0.mlp.gate",
        "model.layers.1.mlp.gate",
    }
    assert any("DP chose no prune" in w for w in warnings_u)


def test_compute_assignment_predicted_dloss_uses_exact_pruned_candidate():
    name = "model.layers.0.mlp.experts.gate_up_proj"
    candidates = {
        name: [
            Candidate("NVFP4", 4.25, 213, 1.0, ()),
            Candidate("NVFP4", 2.125, 106, 3.5, (1, 3)),
        ]
    }
    got = compute_assignment_predicted_dloss(
        {name: "NVFP4"}, candidates, {name: (1, 3)},
    )
    assert got == pytest.approx(3.5)


def test_pruned_assignment_without_exact_candidate_is_invariant_failure():
    name = "model.layers.0.mlp.experts.gate_up_proj"
    candidates = {
        name: [Candidate("NVFP4", 4.25, 213, 1.0, ())],
    }
    with pytest.raises(AssertionError, match="no exact candidate"):
        compute_assignment_predicted_dloss(
            {name: "NVFP4"}, candidates, {name: (1, 3)},
        )
    with pytest.raises(AssertionError, match="no exact candidate"):
        compute_achieved(
            {name: {"n_params": 400, "in_features": 8, "out_features": 10}},
            {name: "NVFP4"},
            {"NVFP4": fr.get_format("NVFP4")},
            candidates=candidates,
            pruned_map={name: (1, 3)},
        )
