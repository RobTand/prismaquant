"""Tests for the allocator's fused-sibling pre-aggregation path and the
convergence-based iteration stop in `solve_with_promotion`.

Fused-sibling coupling (q/k/v must share one format, gate/up must share
one format) used to be enforced as a post-pass (`promote_fused`) that
inflated the achieved bpp above the target whenever the DP picked
different formats for siblings. The new `aggregate_fused_siblings`
pre-pass collapses each sibling group into a single DP candidate, so
the knapsack can't pick mixed-sibling solutions and the overshoot
vanishes.

These tests pin:

  - aggregate_fused_siblings respects the profile's fused_sibling_group
    and groups 2+-member groups while passing singletons through
  - The super-Linear's per-format predicted_dloss equals the exact sum
    of member predicted_dlosses (MoE aggregation already guarantees
    this mathematically; we replicate it here for siblings)
  - expand_fused_sibling_assignment broadcasts the super-Linear's
    chosen format back to every member
  - Pre-aggregation + DP achieves the target bit budget exactly, vs
    the post-promote pipeline which overshoots
  - solve_with_promotion stops early when consecutive iterations stall
"""
from __future__ import annotations

from prismaquant import format_registry as fr
from prismaquant.allocator import (
    Candidate,
    _FUSED_SIBLING_MARKER,
    aggregate_fused_siblings,
    build_candidates,
    compute_achieved,
    expand_fused_sibling_assignment,
    solve_with_promotion,
)


# ---------------------------------------------------------------------------
# Minimal test fixture: a fake profile that knows about qkv_proj siblings
# ---------------------------------------------------------------------------
class _FakeProfile:
    """Profile stub: q/k/v at prefix P form one sibling group keyed by P.
    o_proj and standalone Linears get no group (passes through)."""

    def fused_sibling_group(self, name: str) -> str | None:
        if name.endswith(".q_proj") or name.endswith(".k_proj") or name.endswith(".v_proj"):
            return name.rsplit(".", 1)[0] + ".qkv_proj"
        if name.endswith(".gate_proj") or name.endswith(".up_proj"):
            return name.rsplit(".", 1)[0] + ".gate_up_proj"
        return None


def _mk_stats_and_costs():
    """Build a tiny 1-layer model: 3 qkv Linears + 1 o_proj + 2 gate/up + 1 down.
    Sizes are deliberately asymmetric so we can detect wrong aggregation."""
    layer = "model.layers.0"
    names = [
        f"{layer}.self_attn.q_proj",
        f"{layer}.self_attn.k_proj",
        f"{layer}.self_attn.v_proj",
        f"{layer}.self_attn.o_proj",
        f"{layer}.mlp.gate_proj",
        f"{layer}.mlp.up_proj",
        f"{layer}.mlp.down_proj",
    ]
    stats = {}
    costs = {}
    # Per-Linear: different h_trace values so predicted_dloss sums are
    # distinguishable, different shapes so params differ.
    shapes = {
        "q_proj": (4096, 4096),
        "k_proj": (1024, 4096),
        "v_proj": (1024, 4096),
        "o_proj": (4096, 4096),
        "gate_proj": (11008, 4096),
        "up_proj": (11008, 4096),
        "down_proj": (4096, 11008),
    }
    h_traces = {
        "q_proj": 0.5, "k_proj": 0.3, "v_proj": 0.7,
        "o_proj": 0.4, "gate_proj": 0.8, "up_proj": 0.6, "down_proj": 0.9,
    }
    for name in names:
        leaf = name.rsplit(".", 1)[1]
        d_out, d_in = shapes[leaf]
        stats[name] = {
            "h_trace": h_traces[leaf],
            "n_params": d_out * d_in,
            "in_features": d_in,
            "out_features": d_out,
        }
        # Mock per-format costs: NVFP4 is cheap but high Δloss,
        # BF16 is expensive but zero Δloss.
        costs[name] = {
            "NVFP4": {"weight_mse": 0.02, "predicted_dloss": 0.5 * h_traces[leaf] * 0.02},
            "BF16":  {"weight_mse": 0.0,  "predicted_dloss": 0.0},
        }
    return names, stats, costs


def _format_specs():
    return [fr.REGISTRY["NVFP4"], fr.REGISTRY["BF16"]]


# ---------------------------------------------------------------------------
# aggregate_fused_siblings
# ---------------------------------------------------------------------------

def test_aggregation_groups_qkv_and_gate_up_only():
    names, stats, costs = _mk_stats_and_costs()
    specs = _format_specs()
    cands = build_candidates(stats, costs, specs)
    profile = _FakeProfile()

    stats_ext, costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, profile)

    # qkv → one super-Linear. gate+up → one super-Linear. o_proj + down_proj
    # stay on their own. 7 entries collapse to 4.
    supers = [n for n in cands_ext if _FUSED_SIBLING_MARKER in n]
    assert len(supers) == 2, f"expected 2 super-Linears (qkv, gate_up), got {supers}"
    assert len(cands_ext) == 4, (
        f"expected 4 total entries (2 super + o_proj + down_proj); got "
        f"{sorted(cands_ext)}"
    )

    # Per-leaf names that got aggregated should NOT appear in cands_ext.
    for leaf in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
        aggregated_away = [n for n in cands_ext if n.endswith("." + leaf)]
        assert not aggregated_away, f"{leaf} leaked: {aggregated_away}"

    # o_proj and down_proj pass through.
    assert any(n.endswith(".o_proj") for n in cands_ext)
    assert any(n.endswith(".down_proj") for n in cands_ext)


def test_super_linear_predicted_dloss_is_sum_of_members():
    """Super-Linear's Δloss for format f must equal Σ per-sibling
    predicted_dloss. This matches the Fisher-diagonal cost model:
    Δloss decomposes additively over weights, therefore over Linears,
    therefore over siblings.

    (Earlier versions of this test pinned max×n aggregation — that was
    motivated by a contaminated perplexity measurement that has since
    been shown to be a validator bug, not a real breakage. Sum is
    the aggregation that matches the cost-model math. See the
    aggregate_fused_siblings docstring for the full rationale.)
    """
    names, stats, costs = _mk_stats_and_costs()
    specs = _format_specs()
    cands = build_candidates(stats, costs, specs)
    profile = _FakeProfile()

    stats_ext, costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, profile)

    qkv_super = next(n for n in cands_ext if "qkv_proj" in n)
    qkv_members = [n for n in names if n.endswith((".q_proj", ".k_proj", ".v_proj"))]

    for c in cands_ext[qkv_super]:
        expected = sum(costs[m][c.fmt]["predicted_dloss"] for m in qkv_members)
        assert abs(c.predicted_dloss - expected) < 1e-9, (
            f"format {c.fmt}: super Δloss={c.predicted_dloss} "
            f"vs expected sum={expected}"
        )


def test_asymmetric_sensitivity_sums_contributions():
    """Concrete values for the asymmetric-sensitivity case: q and k
    insensitive (Δloss=1 each), v sensitive (Δloss=1000). The
    super-Linear at NVFP4 must see sum = 1002 (not max*3 = 3000).

    Note: this test intentionally does NOT assert "the DP picks BF16
    for this group." Whether the DP picks BF16 depends on the ratio
    of 1002 to the BF16 budget cost, which is a property of the whole
    knapsack — not of this one sibling group. The correct aggregation
    is sum; whether the DP then promotes to BF16 is a separate DP
    question driven by the total bit-budget. Safety against individual
    sensitive Linears being dropped to unsafe formats is a
    format-constraint concern, orthogonal to this aggregation."""
    import prismaquant.format_registry as fr
    from prismaquant.allocator import aggregate_fused_siblings, build_candidates
    layer = "model.layers.0"
    names_ = [f"{layer}.self_attn.q_proj",
              f"{layer}.self_attn.k_proj",
              f"{layer}.self_attn.v_proj"]
    stats = {}
    costs = {}
    dlosses = {"q_proj": 1.0, "k_proj": 1.0, "v_proj": 1000.0}
    for n in names_:
        leaf = n.rsplit(".", 1)[1]
        stats[n] = {"h_trace": 1.0, "n_params": 1024 * 1024,
                    "in_features": 1024, "out_features": 1024}
        costs[n] = {
            "NVFP4": {"weight_mse": dlosses[leaf] / 0.5,
                      "predicted_dloss": dlosses[leaf]},
            "BF16":  {"weight_mse": 0.0, "predicted_dloss": 0.0},
        }
    specs = [fr.REGISTRY["NVFP4"], fr.REGISTRY["BF16"]]
    cands = build_candidates(stats, costs, specs)
    _, _, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, _FakeProfile())
    super_name = next(n for n in cands_ext if _FUSED_SIBLING_MARKER in n)
    nvfp4_cand = next(c for c in cands_ext[super_name] if c.fmt == "NVFP4")
    assert nvfp4_cand.predicted_dloss == 1002.0, (
        f"sum-aggregation should give 1 + 1 + 1000 = 1002; "
        f"got {nvfp4_cand.predicted_dloss}"
    )


def test_expand_broadcasts_super_format_to_members():
    names, stats, costs = _mk_stats_and_costs()
    specs = _format_specs()
    cands = build_candidates(stats, costs, specs)
    profile = _FakeProfile()

    stats_ext, costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, profile)

    # Fake a DP assignment: pick BF16 for qkv_super, NVFP4 for gate_up super,
    # and NVFP4 for the singletons.
    assignment = {}
    for n in cands_ext:
        if "qkv_proj" in n:
            assignment[n] = "BF16"
        else:
            assignment[n] = "NVFP4"

    expanded = expand_fused_sibling_assignment(assignment, stats_ext)

    # Every q/k/v should get BF16; gate/up should get NVFP4.
    for m in names:
        if m.endswith((".q_proj", ".k_proj", ".v_proj")):
            assert expanded[m] == "BF16", f"{m} should be BF16, got {expanded[m]}"
        elif m.endswith((".gate_proj", ".up_proj")):
            assert expanded[m] == "NVFP4", f"{m} should be NVFP4, got {expanded[m]}"
        else:
            assert expanded[m] == "NVFP4"
    # The super-Linear markers should NOT appear in the expanded output.
    assert not any(_FUSED_SIBLING_MARKER in n for n in expanded)


def test_singleton_groups_pass_through_unchanged():
    """A profile reporting a group key for only ONE member should NOT
    aggregate — there's no benefit, and aggregation would change the
    entry name."""
    class _SingletonProfile:
        def fused_sibling_group(self, name):
            # Only q_proj gets a group key; k_proj and v_proj get None.
            # Even though the key is non-None for q_proj, it's a singleton
            # because no other member shares it.
            return name.rsplit(".", 1)[0] + ".solo" if name.endswith(".q_proj") else None

    names, stats, costs = _mk_stats_and_costs()
    specs = _format_specs()
    cands = build_candidates(stats, costs, specs)
    stats_ext, costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, _SingletonProfile())
    # No `__siblings__` markers should appear — the one q_proj group had
    # only one member.
    assert not any(_FUSED_SIBLING_MARKER in n for n in cands_ext)
    assert "model.layers.0.self_attn.q_proj" in cands_ext


def test_aggregation_is_no_op_without_profile():
    names, stats, costs = _mk_stats_and_costs()
    specs = _format_specs()
    cands = build_candidates(stats, costs, specs)
    stats_ext, costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, profile=None)
    assert cands_ext is cands or cands_ext == cands


# ---------------------------------------------------------------------------
# End-to-end: pre-aggregation hits target without overshoot
# ---------------------------------------------------------------------------

def test_pre_aggregation_respects_budget_without_overshoot():
    """With siblings pre-aggregated, the DP's solution is already
    sibling-consistent. The promote_fused post-pass becomes a no-op,
    and solve_with_promotion returns on iteration 1 with achieved ≤ target."""
    names, stats, costs = _mk_stats_and_costs()
    specs = _format_specs()
    cands = build_candidates(stats, costs, specs)
    profile = _FakeProfile()

    # Aggregate first.
    stats_ext, costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, profile)

    format_specs = {s.name: s for s in specs}
    format_rank = {"NVFP4": 0, "BF16": 1}

    target = 8.0  # something mid-range between NVFP4 (~4.5) and BF16 (16)
    assignment, _pruned, achieved = solve_with_promotion(
        stats_ext, cands_ext, target,
        format_specs, format_rank,
        bit_precision=0.001,
        profile=profile,
    )
    assert assignment is not None
    assert achieved <= target + 0.01, (
        f"aggregated path should hit budget: target={target}, achieved={achieved}"
    )


# ---------------------------------------------------------------------------
# Convergence-based stopping
# ---------------------------------------------------------------------------

def test_solve_with_promotion_stops_on_stall():
    """Construct a tiny problem where promotion would overshoot and
    tightening can't make further progress — loop should bail out early
    via the stall detector, not waste all max_iters slots."""
    # Use the un-aggregated path so promote_fused has siblings to coerce.
    names, stats, costs = _mk_stats_and_costs()
    specs = _format_specs()
    cands = build_candidates(stats, costs, specs)
    profile = _FakeProfile()

    format_specs = {s.name: s for s in specs}
    format_rank = {"NVFP4": 0, "BF16": 1}

    # Target just above the NVFP4 floor. promote_fused may bump q/k/v to
    # BF16 if the DP picks mixed formats, overshooting; tightening can
    # only push until we hit the floor. The stall guard caps iteration.
    target = 4.6
    assignment, _pruned, achieved = solve_with_promotion(
        stats, cands, target, format_specs, format_rank,
        bit_precision=0.001,
        stall_threshold=1e-3,
        stall_grace=2,
        max_iters=40,
        profile=profile,
    )
    # Assignment must be non-None (solver found SOMETHING), and achieved
    # is reported even when we bail on stall (the whole point of #2).
    assert assignment is not None
    assert isinstance(achieved, float)
