#!/usr/bin/env python3
"""allocator.py — multi-choice knapsack mixed-precision assignment.

Given:
  - per-Linear empirical Fisher diagonal trace (from sensitivity_probe.py)
  - per-(Linear, format) measured quantization cost (from measure_quant_cost.py)
  - a bit budget (target average bits per parameter)
  - a format registry (any subset of registered formats)

Solve for a per-Linear format assignment that minimizes total predicted
loss increase subject to the bit budget.

Derivation of the per-(layer, format) predicted loss term
---------------------------------------------------------
Let L be the per-token loss (negative log-likelihood). Quantizing layer
ℓ's weight tensor W by ΔW = W_q - W produces a perturbed loss whose
expectation under the calibration distribution admits the standard
second-order expansion:

    E[ΔL] ≈ 0.5 · ΔW · F · ΔWᵀ                         (1)

where F is the Fisher information matrix of L w.r.t. W. Replacing F by
its diagonal (the standard HAWQ-V1 simplification) and approximating
F_ww by the empirical Fisher diagonal F̂_ww = E_token[(∂L/∂W_w)²]:

    E[ΔL] ≈ 0.5 · Σ_w F̂_ww · (ΔW_w)²                   (2)

Under the further assumption that the per-weight quantization error
(ΔW_w)² and the per-weight Fisher diagonal F̂_ww are uncorrelated across
w (which is the same assumption HAWQ already makes when it summarizes a
layer by a single scalar), this collapses to the product of two
per-layer scalars:

    E[ΔL] ≈ 0.5 · H_trace · MSE_W                       (3)

where
    H_trace = Σ_w F̂_ww            (per-token Fisher diagonal trace)
    MSE_W   = (1/n_w) · Σ_w (ΔW_w)²

Both quantities are produced by upstream stages:
    H_trace ← sensitivity_probe.py / FisherAccumulator (`h_trace`)
    MSE_W   ← measure_quant_cost.py (per-(layer, format) `weight_mse`)

So we use eq. (3) directly. There is no `* d_out` factor; the previous
implementation carried one but it does not appear in the derivation —
it was a holdover from an earlier output-side formulation that mixed
units and was off by a per-layer multiplicative constant that varies
with d_out.

For MoE experts an additional route-probability normalization is folded
into H_trace inside the probe so that sparsely-routed experts' Fisher
contributions are on the same per-token footing as dense layers'.

Solver:
  Multi-choice knapsack via DP with bit-budget discretization (we round
  bit costs to 0.001-bit bins). For 35B with ~300 Linears × 8 formats ×
  ~5000 budget bins, runtime is under 1s.

Fused-projection siblings (q/k/v/o, gate/up, ...) are post-processed:
  all siblings promoted to the highest format chosen for any of them,
  to match vLLM's fused-tensor loader constraints. Since promotion can
  push achieved bits past the requested budget, the DP is re-run with a
  tightened target until achieved is within tolerance.

Optional empirical calibration:
  If `--calibration` points at a JSON produced by calibrate_allocator.py
  containing `calibrated_gains[fmt] = α_fmt`, the predicted Δloss for
  format f is multiplied by α_f before the DP runs. This corrects for
  systematic over- or under-prediction per format observed against
  measured KL on the bake-off frontier.

Auto-Pareto knee via Kneedle (Satopää et al.). Reports the knee target
plus a few flanking points so you can eyeball.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import format_registry as fr


# ---------------------------------------------------------------------------
# Fused-projection sibling detection (general)
# ---------------------------------------------------------------------------
_FUSED_PATTERNS = [
    # Standard HF self-attention
    (r"^(?P<pre>.+)\.self_attn\.(?P<sib>q_proj|k_proj|v_proj|o_proj)$",
     ("q_proj", "k_proj", "v_proj"), "self_attn"),
    # Standard HF MLP
    (r"^(?P<pre>.+)\.mlp\.(?P<sib>gate_proj|up_proj)$",
     ("gate_proj", "up_proj"), "mlp"),
    # Legacy Mixtral/Mistral
    (r"^(?P<pre>.+)\.(?P<sib>w1|w3)$",
     ("w1", "w3"), "w1w3"),
    # Legacy PaLM-style
    (r"^(?P<pre>.+)\.(?P<sib>gate|up)_proj$",
     ("gate_proj", "up_proj"), "gate_up_alt"),
    # Qwen3.6 hybrid linear-attention: vLLM fuses in_proj_qkv + in_proj_z
    # into a single `in_proj_qkvz` parameter, so they must share format.
    (r"^(?P<pre>.+)\.linear_attn\.(?P<sib>in_proj_qkv|in_proj_z)$",
     ("in_proj_qkv", "in_proj_z"), "linear_attn_qkvz"),
    # Qwen3.6 also fuses in_proj_a + in_proj_b into `in_proj_ba`.
    (r"^(?P<pre>.+)\.linear_attn\.(?P<sib>in_proj_a|in_proj_b)$",
     ("in_proj_a", "in_proj_b"), "linear_attn_ba"),
]


def fused_siblings(name: str) -> tuple[tuple[str, ...], str] | None:
    for pat, members, kind in _FUSED_PATTERNS:
        m = re.search(pat, name)
        if m:
            pre = m.group("pre")
            parent = name[: name.rindex(m.group("sib"))]
            # Reconstruct all sibling names that should be promoted together
            sibs = tuple(f"{parent}{s}" for s in members)
            return sibs, kind
    return None


def promote_fused(assignment: dict[str, str],
                  format_rank: dict[str, int]) -> dict[str, str]:
    """After per-Linear selection, bump each fused group's siblings to the
    highest-rank format picked for any group member."""
    out = dict(assignment)
    groups: dict[tuple, list[str]] = defaultdict(list)
    for name in assignment:
        sib = fused_siblings(name)
        if sib is not None:
            sibs, kind = sib
            groups[sibs].append(name)
    for sibs, members_present in groups.items():
        # Only consider siblings we actually have allocations for
        ranks = [format_rank[out[m]] for m in members_present]
        best = max(ranks)
        best_fmt = next(k for k, v in format_rank.items() if v == best)
        for m in members_present:
            if format_rank[out[m]] < best:
                out[m] = best_fmt
    return out


def solve_with_promotion(
    stats: dict,
    candidates: dict[str, list[Candidate]],
    target_bits: float,
    format_specs: dict[str, fr.FormatSpec],
    format_rank: dict[str, int],
    bit_precision: float,
    *,
    no_fused_promote: bool = False,
    overshoot_tolerance: float = 0.01,
    max_iters: int = 6,
) -> tuple[dict[str, str] | None, float]:
    """Solve the allocation, promote fused siblings, and re-solve with a
    tightened target if promotion blew past the budget.

    Promotion is allowed to inflate the achieved bits because vLLM's
    fused tensor loader requires a single format per fused group. The
    natural fix — already employed implicitly by the previous version —
    is to reserve some headroom in the DP for promotion. We make this
    explicit and adaptive: if promotion overshoots the requested target
    by more than `overshoot_tolerance` bits/param, halve the overshoot
    by tightening the next solve, and repeat up to `max_iters` times.

    Returns (assignment, achieved_bits). Assignment is None if even the
    untightened solve was infeasible.
    """
    tightened = float(target_bits)
    last_assign: dict[str, str] | None = None
    last_achieved = float("nan")
    for _ in range(max_iters):
        assign = solve_allocation(stats, candidates, tightened, bit_precision)
        if assign is None:
            return last_assign, last_achieved
        if not no_fused_promote:
            assign = promote_fused(assign, format_rank)
        achieved, _ = compute_achieved(stats, assign, format_specs)
        last_assign = assign
        last_achieved = achieved
        overshoot = achieved - target_bits
        if overshoot <= overshoot_tolerance:
            return assign, achieved
        # Tighten by half the overshoot. This converges geometrically:
        # if the first solve overshot by 0.1b, second by ~0.05b, etc.
        tightened -= overshoot / 2.0
        if tightened <= 0:
            break
    return last_assign, last_achieved


# ---------------------------------------------------------------------------
# Kneedle knee detection
# ---------------------------------------------------------------------------
def kneedle(x: list[float], y: list[float]) -> int:
    """Return index of the knee in a convex-decreasing curve."""
    if len(x) < 3:
        return 0
    xs = [xi for xi in x]
    ys = [yi for yi in y]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin or ymax == ymin:
        return 0
    x_norm = [(xi - xmin) / (xmax - xmin) for xi in xs]
    y_norm = [(yi - ymin) / (ymax - ymin) for yi in ys]
    # For a convex-decreasing curve, the knee is the point with max
    # distance below the chord from (0,1) to (1,0).
    diffs = [yn - (1.0 - xn) for xn, yn in zip(x_norm, y_norm)]
    # Convex-decreasing, so we want the most-negative diff (max dip).
    return min(range(len(diffs)), key=lambda i: diffs[i])


# ---------------------------------------------------------------------------
# Multi-choice knapsack DP
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    fmt: str
    bits_per_param: float
    memory_bytes: int
    predicted_dloss: float


def _shape_from_stats(entry: dict) -> tuple[int, ...]:
    out_features = int(entry.get("out_features", 0) or 0)
    in_features = int(entry.get("in_features", 0) or 0)
    if out_features > 0 and in_features > 0:
        return (out_features, in_features)
    n_params = int(entry.get("n_params", 0) or 0)
    return (n_params,)


def predicted_dloss(h_trace: float, weight_mse: float,
                    gain: float = 1.0) -> float:
    """Per-(layer, format) predicted ΔL under the diagonal-Fisher model.

    Δloss ≈ 0.5 · H_trace · MSE_W · α    (see module docstring eq. (3)).

    `gain` is the optional per-format calibration scalar α_f. Default
    1.0 leaves predictions uncalibrated.
    """
    return 0.5 * float(h_trace) * float(weight_mse) * float(gain)


def build_candidates(stats: dict, costs: dict, formats: list[fr.FormatSpec],
                     calibrated_gains: dict[str, float] | None = None
                     ) -> dict[str, list[Candidate]]:
    """For each Linear, build its candidate list (one per format).

    Per-(layer, format) predicted Δloss uses the closed-form
    diagonal-Fisher term `0.5 · h_trace · weight_mse` (see module
    docstring), optionally scaled by per-format calibrated_gains[fmt].
    """
    gains = calibrated_gains or {}
    out: dict[str, list[Candidate]] = {}
    for name, s in stats.items():
        if name not in costs:
            continue
        h_trace = s["h_trace"]
        shape = _shape_from_stats(s)
        cands = []
        for spec in formats:
            entry = costs[name].get(spec.name)
            if entry is None or "error" in entry:
                continue
            weight_mse = float(entry.get("weight_mse", 0.0))
            gain = float(gains.get(spec.name, 1.0))
            predicted = predicted_dloss(h_trace, weight_mse, gain=gain)
            cands.append(Candidate(
                fmt=spec.name,
                bits_per_param=spec.effective_bits_for_shape(shape),
                memory_bytes=spec.memory_bytes_for_shape(shape),
                predicted_dloss=max(predicted, 0.0),
            ))
        if cands:
            out[name] = cands
    return out


def _moe_group_and_projection(name: str) -> tuple[str, str] | None:
    """Return `(experts_group_path, projection_suffix)` for expert leaves.

    Supports both common layouts:
      - `<prefix>.experts.<eid>.<projection>`
      - `<prefix>.experts.<projection>.<eid>` (Qwen3.5/3.6 packed experts)
    """
    m = re.search(r"^(.+\.experts)\.\d+\.(.+)$", name)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"^(.+\.experts)\.(gate_up_proj|down_proj)\.\d+$", name)
    if m:
        return m.group(1), m.group(2)
    return None


def _aggregate_candidate_memory_bits(
    members: list[str],
    spec: fr.FormatSpec,
    stats: dict,
) -> tuple[int, float]:
    total_params = sum(stats[m]["n_params"] for m in members)
    total_bytes = 0
    for m in members:
        shape = _shape_from_stats(stats[m])
        total_bytes += spec.memory_bytes_for_shape(shape)
    bits_per_param = 8.0 * total_bytes / max(total_params, 1)
    return total_bytes, bits_per_param


def aggregate_moe_candidates(
    stats: dict, costs: dict, formats: list[fr.FormatSpec],
    candidates: dict[str, list[Candidate]],
    granularity: str = "projection",
    calibrated_gains: dict[str, float] | None = None,
) -> tuple[dict, dict, dict]:
    """Aggregate per-expert Linears into per-layer MoE super-candidates.

    vLLM's FusedMoE kernel requires a single format per layer's fused
    expert tensor. Per-expert mixing is only possible via slow unfused
    serving paths. Statistically, per-expert Fisher is also noise-
    dominated at typical calibration budgets, so aggregation gives
    cleaner signal too — both correctness arguments point the same way.

    This function:
      1. Groups Linears by (expert_group_path, projection_suffix), e.g.
         `model.layers.5.mlp.experts.*.gate_proj` becomes one group.
      2. Builds a synthetic "super-Linear" per group in returned stats_ext
         and costs_ext, with aggregated params/sensitivity/RTN errors.
      3. The super-Linear's `n_params` = Σ_i n_params_i (so MSE_W terms
         can be aggregated as a parameter-weighted mean).

    The aggregated predicted Δloss for the super-Linear at format f is
    the sum of per-expert predicted Δlosses, which under the closed-form
    formula 0.5 · h_i · mse_i,f decomposes cleanly:

        sum_pred(f) = Σ_i 0.5 · h_i · weight_mse_{i,f}

    To make the super-Linear behave identically under
    `predicted_dloss(h, mse, ...)` (which uses one h and one mse), we
    pick the natural representatives:

        h_super  = Σ_i h_i                (sum of per-expert Fisher trace)
        mse_super(f) = sum_pred(f) / (0.5 · h_super) if h_super > 0 else 0

    With that, `0.5 · h_super · mse_super(f)` reproduces sum_pred(f)
    exactly. The super-Linear's `out_features` is preserved as the
    expert's true `out_features` so downstream code that consults it
    (e.g. the bake-off summary) sees a real shape, not a sentinel.

    Returns (stats_ext, costs_ext, candidates_ext) where non-expert
    Linears are unchanged and each MoE expert-group becomes one synthetic
    entry keyed by `<group>.__fused__.<projection>`.
    """
    gains = calibrated_gains or {}
    expert_leaves: dict[tuple[str, str], list[str]] = {}
    non_expert_names: list[str] = []
    for name in stats:
        grp_proj = _moe_group_and_projection(name)
        if grp_proj is None:
            non_expert_names.append(name)
            continue
        grp, projection = grp_proj
        if granularity == "layer":
            expert_leaves.setdefault((grp, "__all__"), []).append(name)
        else:
            expert_leaves.setdefault((grp, projection), []).append(name)

    stats_ext = {n: stats[n] for n in non_expert_names}
    costs_ext = {n: costs.get(n, {}) for n in non_expert_names}
    candidates_ext = {n: candidates[n] for n in non_expert_names
                      if n in candidates}

    for (grp, projection), members in expert_leaves.items():
        n_params = sum(stats[m_]["n_params"] for m_ in members)
        # Preserve a real out_features for the super-Linear: pick the
        # representative expert's value (uniform across experts in a
        # well-formed MoE). This avoids the previous out_features=1
        # sentinel which forced downstream callers to special-case.
        d_out = int(stats[members[0]]["out_features"])
        d_in = int(stats[members[0]]["in_features"])
        sum_h = sum(stats[m_]["h_trace"] for m_ in members)
        super_name = f"{grp}.__fused__.{projection}"

        stats_ext[super_name] = {
            "h_trace": sum_h,
            "h_trace_raw": sum(stats[m_].get("h_trace_raw", 0.0) for m_ in members),
            "h_w2_sum": sum(stats[m_].get("h_w2_sum", 0.0) for m_ in members),
            "w_max_abs": max(stats[m_]["w_max_abs"] for m_ in members),
            "w_norm_sq": sum(stats[m_]["w_norm_sq"] for m_ in members),
            "n_params": n_params,
            "in_features": d_in,
            "out_features": d_out,
            "n_tokens_seen": sum(stats[m_].get("n_tokens_seen", 0) for m_ in members),
            "route_prob": None,  # aggregation washes out per-expert route prob
            "router_path": None,
            "expert_id": None,
            "_fused_members": members,
            "_memory_bytes_by_format": {},
        }

        # Per-format aggregation. The true summed Δloss across experts is
        #     sum_pred(f) = Σ_i 0.5 · h_i · weight_mse_{i,f} · α_f
        # Setting mse_super(f) = sum_pred(f) / (0.5 · sum_h · α_f) lets
        # the super-Linear use the same closed-form predicted_dloss as
        # any other Linear. The α_f gain is canceled in the inversion so
        # build_candidates re-applies it cleanly.
        super_cost = {}
        for spec in formats:
            available_members = [
                m_ for m_ in members
                if spec.name in costs.get(m_, {})
                and "error" not in costs.get(m_, {}).get(spec.name, {})
            ]
            if not available_members:
                super_cost[spec.name] = {"error": "partial"}
                continue
            # Parameter-weighted mean weight_mse (correct expert-level summary
            # because Σ_w (ΔW_w)² over the fused tensor equals the param-
            # weighted average of per-expert mean (ΔW_w)²).
            sum_weight_mse_x_params = 0.0
            sum_params_avail = 0
            for m_ in available_members:
                p_i = stats[m_]["n_params"]
                sum_weight_mse_x_params += costs[m_][spec.name]["weight_mse"] * p_i
                sum_params_avail += p_i
            mean_weight_mse = sum_weight_mse_x_params / max(sum_params_avail, 1)
            mean_output_mse = sum(
                costs[m_][spec.name]["output_mse"] for m_ in available_members
            ) / len(available_members)

            # True summed Δloss across all members at format f, using the
            # closed-form per-(layer, format) term.
            sum_pred = 0.0
            for m_ in members:
                c = costs.get(m_, {}).get(spec.name)
                if c is None or "error" in c:
                    c = {"weight_mse": mean_weight_mse,
                         "output_mse": mean_output_mse}
                h_i = stats[m_]["h_trace"]
                sum_pred += 0.5 * h_i * float(c["weight_mse"])

            # Invert to an effective per-element MSE so build_candidates'
            # formula 0.5 · sum_h · effective_mse · α_f reproduces sum_pred.
            if sum_h > 0:
                effective_mse = sum_pred / (0.5 * sum_h)
            else:
                effective_mse = 0.0

            super_cost[spec.name] = {
                "weight_mse": effective_mse,
                "output_mse": mean_output_mse,    # diagnostic only
                "rel_output_mse": mean_output_mse,
            }
        costs_ext[super_name] = super_cost

        cands = []
        for spec in formats:
            entry = super_cost.get(spec.name)
            if entry is None or "error" in entry:
                continue
            gain = float(gains.get(spec.name, 1.0))
            predicted = predicted_dloss(sum_h, entry["weight_mse"], gain=gain)
            memory_bytes, bits_per_param = _aggregate_candidate_memory_bits(
                members, spec, stats
            )
            stats_ext[super_name]["_memory_bytes_by_format"][spec.name] = memory_bytes
            cands.append(Candidate(
                fmt=spec.name,
                bits_per_param=bits_per_param,
                memory_bytes=memory_bytes,
                predicted_dloss=max(predicted, 0.0),
            ))
        if cands:
            candidates_ext[super_name] = cands

    return stats_ext, costs_ext, candidates_ext


def expand_moe_assignment(assignment: dict[str, str],
                          stats_ext: dict) -> dict[str, str]:
    """Replace `.__fused__.` super-Linear assignments with the per-expert
    assignments needed by AutoRound's layer_config (one entry per
    individual expert Linear, all sharing the super-Linear's format)."""
    out = {}
    for name, fmt in assignment.items():
        if ".__fused__." in name:
            members = stats_ext[name].get("_fused_members", [])
            for m_ in members:
                out[m_] = fmt
        else:
            out[name] = fmt
    return out


def solve_allocation(stats: dict, candidates: dict[str, list[Candidate]],
                     target_bits: float, bit_precision: float = 0.001
                     ) -> dict[str, str] | None:
    """Solve multi-choice knapsack via DP, working in avg-bits-per-param units.

    The budget is expressed as an average bits-per-parameter target; we
    discretize (target - baseline) into bins of `bit_precision`. Each
    layer's cost is its contribution to the weighted average, which for
    a layer with fraction f = params/total of the total is
        Δavg = (c.bits_per_param - baseline.bits_per_param) · f.
    Total DP budget ~= (target - baseline) / bit_precision, typically
    under 10 000 bins regardless of model size.

    Returns {linear_name: chosen_format_name}, or None if infeasible.
    """
    import numpy as np

    names = list(candidates.keys())
    total_params = sum(stats[n]["n_params"] for n in names)
    if total_params == 0:
        return {}

    baselines = {n: min(cs, key=lambda c: c.bits_per_param)
                 for n, cs in candidates.items()}
    min_bits = sum(baselines[n].bits_per_param * stats[n]["n_params"]
                   for n in names) / total_params

    if target_bits < min_bits - 1e-6:
        return None

    # Budget in bits-per-param units, so the bin count is independent of
    # model size. For a 35B model with 0.001 bit precision this gives
    # ~5000 bins at a 5.0-bit target, trivially small.
    excess = target_bits - min_bits
    n_bins = int(round(excess / bit_precision)) + 2

    # Per-layer: pre-compute (dbins, dgain, cand_idx) option list.
    # dbins is layer's contribution to the avg-bits-per-param budget,
    # scaled into integer bins.
    INF_NEG = -1e30
    dp = np.full(n_bins, INF_NEG, dtype=np.float64)
    dp[0] = 0.0
    choice: list[np.ndarray] = []

    for name in names:
        baseline = baselines[name]
        cs = candidates[name]
        params = stats[name]["n_params"]
        fraction = params / total_params
        baseline_loss = baseline.predicted_dloss
        options = []
        for idx, c in enumerate(cs):
            d_avg_bits = (c.bits_per_param - baseline.bits_per_param) * fraction
            dbins = int(round(d_avg_bits / bit_precision))
            if dbins < 0 or dbins >= n_bins:
                continue
            dgain = baseline_loss - c.predicted_dloss
            options.append((dbins, dgain, idx))
        if not options:
            options = [(0, 0.0, cs.index(baseline))]

        # Convert to arrays for fast inner loop
        opt_dbins = np.asarray([o[0] for o in options], dtype=np.int32)
        opt_dgain = np.asarray([o[1] for o in options], dtype=np.float64)
        opt_idx = np.asarray([o[2] for o in options], dtype=np.int32)

        new_dp = np.full(n_bins, INF_NEG, dtype=np.float64)
        new_choice = np.full(n_bins, -1, dtype=np.int32)

        # Vectorized update: for each option, add (dbins, dgain) to dp
        for db, dg, idx in zip(opt_dbins, opt_dgain, opt_idx):
            if db == 0:
                candidate_vals = dp + dg
                target_slice = new_dp
                mask = candidate_vals > target_slice
                new_dp = np.where(mask, candidate_vals, new_dp)
                new_choice = np.where(mask, idx, new_choice)
            else:
                candidate_vals = dp[:-db] + dg
                target_slice = new_dp[db:]
                mask = candidate_vals > target_slice
                target_slice[:] = np.where(mask, candidate_vals, target_slice)
                new_choice[db:] = np.where(mask, idx, new_choice[db:])
        dp = new_dp
        choice.append(new_choice)

    if not np.isfinite(dp).any() or dp.max() == INF_NEG:
        return None
    best_b = int(np.argmax(dp))

    # Backtrack
    assignment = {}
    cur = best_b
    for layer_idx in range(len(names) - 1, -1, -1):
        idx_chosen = int(choice[layer_idx][cur])
        name = names[layer_idx]
        cs = candidates[name]
        if idx_chosen < 0:
            idx_chosen = 0
        assignment[name] = cs[idx_chosen].fmt
        baseline = baselines[name]
        params = stats[name]["n_params"]
        fraction = params / total_params
        d_avg_bits = (cs[idx_chosen].bits_per_param
                      - baseline.bits_per_param) * fraction
        cur -= int(round(d_avg_bits / bit_precision))
        if cur < 0:
            cur = 0
    return assignment


def compute_achieved(stats: dict, assignment: dict[str, str],
                     format_specs: dict[str, fr.FormatSpec]) -> tuple[float, float]:
    """Return (avg_bits, total_predicted_dloss)."""
    total_params = sum(stats[n]["n_params"] for n in assignment)
    total_bits = 0.0
    for n in assignment:
        memory_map = stats[n].get("_memory_bytes_by_format")
        if memory_map is not None and assignment[n] in memory_map:
            total_bits += 8.0 * memory_map[assignment[n]]
        else:
            shape = _shape_from_stats(stats[n])
            total_bits += (
                format_specs[assignment[n]].effective_bits_for_shape(shape)
                * stats[n]["n_params"]
            )
    return total_bits / max(total_params, 1), 0.0  # dloss recomputed separately


def _allowed_format(target_profile: str, name: str, fmt: str) -> bool:
    if target_profile == "research":
        return True
    if target_profile == "vllm_qwen3_5_packed_moe":
        if ".mlp.experts" in name:
            return fmt in {"NVFP4", "FP8_E4M3", "FP8_E5M2", "BF16", "MXFP4"}
        return True
    raise ValueError(f"Unknown target profile: {target_profile}")


def filter_candidates_for_profile(
    candidates: dict[str, list[Candidate]],
    target_profile: str,
) -> dict[str, list[Candidate]]:
    out = {}
    for name, cands in candidates.items():
        kept = [c for c in cands if _allowed_format(target_profile, name, c.fmt)]
        if kept:
            out[name] = kept
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="sensitivity_probe pickle")
    ap.add_argument("--costs", required=True, help="measure_quant_cost pickle")
    ap.add_argument("--target-bits", type=float, default=4.75)
    ap.add_argument("--formats", default="",
                    help="Comma-separated format names to consider; empty=all")
    ap.add_argument("--pareto-targets",
                    default="4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25",
                    help="Comma-separated budgets to sweep for Pareto curve")
    ap.add_argument("--layer-config", required=True,
                    help="Output AutoRound layer_config JSON")
    ap.add_argument("--pareto-csv", required=True, help="Output Pareto CSV")
    ap.add_argument("--no-fused-promote", action="store_true",
                    help="Skip fused-projection sibling promotion")
    ap.add_argument("--enforce-family-coherence", action="store_true",
                    help="Error (instead of warn) if the format set contains "
                         "multiple candidates for the same bit tier (e.g. "
                         "NVFP4 and MXFP4 both at 4 bits)")
    ap.add_argument("--bit-precision", type=float, default=0.001,
                    help="Knapsack bit-bin granularity in avg-bits/param "
                         "(smaller = slower; default 0.001 → ~5000 bins)")
    ap.add_argument("--threads", type=int, default=0,
                    help="OMP/numpy threads for DP (0 = default)")
    ap.add_argument("--expert-granularity", choices=["layer", "expert"],
                    default="layer",
                    help="MoE experts allocation granularity. 'layer' (default) "
                         "assigns one format to all experts in a layer's fused "
                         "tensor — required for full-speed fused-MoE serving "
                         "on every major stack (vLLM FlashInfer/Marlin, SGLang, "
                         "TensorRT-LLM). 'expert' allows per-expert mixing but "
                         "forces slower sequential serving and is noise-floor "
                         "limited at typical calibration budgets.")
    ap.add_argument("--target-profile",
                    choices=["research", "vllm_qwen3_5_packed_moe"],
                    default="research",
                    help="Serving/backend constraint profile. "
                         "'vllm_qwen3_5_packed_moe' collapses Qwen3.5/3.6 MoE "
                         "to legal packed serving units and restricts MoE "
                         "formats to the existing vLLM path.")
    ap.add_argument("--calibration", default=None,
                    help="Optional path to a calibrate_allocator.py JSON "
                         "containing 'calibrated_gains[fmt] = α_fmt'. When "
                         "present, the per-(layer, format) predicted Δloss "
                         "is multiplied by α_fmt before the DP runs.")
    ap.add_argument("--overshoot-tolerance", type=float, default=0.01,
                    help="Maximum allowed overshoot (bits/param) of the "
                         "achieved budget over the requested target after "
                         "fused-sibling promotion. The DP is re-run with a "
                         "tightened target until overshoot is within tol.")
    args = ap.parse_args()

    if args.threads > 0:
        import os
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    with open(args.probe, "rb") as f:
        probe = pickle.load(f)
    with open(args.costs, "rb") as f:
        cost_data = pickle.load(f)
    stats = probe["stats"]
    costs = cost_data["costs"]
    print(f"[alloc] stats: {len(stats)} Linears, costs: {len(costs)} Linears")

    if args.formats:
        fmt_names = [s.strip() for s in args.formats.split(",") if s.strip()]
    else:
        fmt_names = cost_data["formats"]
    specs = [fr.get_format(n) for n in fmt_names]
    specs_sorted = sorted(specs, key=lambda s: s.effective_bits)

    # --- Format-family coherence check -----------------------------------
    # A sensible format ladder has at most ONE format per bit tier. Having
    # both NVFP4 and MXFP4 (or MXFP6_E3M2 and MXFP6_E2M3) means the allocator
    # picks between them based on tiny measurement noise per-layer, which
    # produces a serving mess: two separate kernel paths for the same tier.
    #
    # We bucket formats by effective_bits rounded to 0.25 and warn when a
    # bucket has more than one member. If --enforce-family-coherence is
    # set we error instead.
    from collections import Counter as _Counter
    buckets: dict[float, list[str]] = {}
    for s in specs_sorted:
        key = round(s.effective_bits * 4) / 4
        buckets.setdefault(key, []).append(s.name)
    collisions = {k: v for k, v in buckets.items() if len(v) > 1}
    if collisions:
        msg = ("format set has multiple candidates at the same bit tier; "
               "the allocator will pick among them based on per-layer RTN "
               "noise, which is usually not what you want:\n"
               + "\n".join(f"  {k} bits: {v}" for k, v in collisions.items())
               + "\nRecommended bundles (vLLM serving, today):\n"
               "  Ship-ready     : NVFP4,MXFP8       (validated)\n"
               "  MX-pure        : MXFP4,MXFP8\n"
               "  Experimental   : NVFP4,MXFP6_E3M2,MXFP8   "
               "(MXFP6 hardware-supported on Blackwell, vLLM kernels not yet landed)")
        if args.enforce_family_coherence:
            raise SystemExit(f"[alloc] ERROR: {msg}")
        else:
            print(f"[alloc] WARNING: {msg}", flush=True)
    format_rank = {s.name: i for i, s in enumerate(specs_sorted)}
    format_specs = {s.name: s for s in specs}
    print(f"[alloc] formats (low→high bits): "
          f"{[f'{s.name}({s.effective_bits:.2f}b)' for s in specs_sorted]}")

    # Optional empirical calibration: per-format scalar gain α_f produced
    # by calibrate_allocator.py. When absent, all gains default to 1.0.
    calibrated_gains: dict[str, float] = {}
    if args.calibration:
        with open(args.calibration) as f:
            cal_payload = json.load(f)
        cal_raw = cal_payload.get("calibrated_gains") or {}
        for fmt_name, gain_val in cal_raw.items():
            try:
                calibrated_gains[fmt_name] = float(gain_val)
            except (TypeError, ValueError):
                continue
        if calibrated_gains:
            print(f"[alloc] calibration loaded from {args.calibration}: "
                  f"{ {k: round(v, 4) for k, v in calibrated_gains.items()} }",
                  flush=True)
        else:
            print(f"[alloc] WARNING: {args.calibration} has no usable "
                  f"calibrated_gains; running uncalibrated", flush=True)

    candidates = build_candidates(stats, costs, specs_sorted, calibrated_gains)
    print(f"[alloc] candidates built for {len(candidates)} Linears")

    if args.target_profile == "vllm_qwen3_5_packed_moe":
        stats, costs, candidates = aggregate_moe_candidates(
            stats, costs, specs_sorted, candidates, granularity="layer",
            calibrated_gains=calibrated_gains)
        moe_groups = sum(1 for n in candidates if ".__fused__." in n)
        print(f"[alloc] packed-MoE serving aggregation: {moe_groups} fused MoE blocks")
    elif args.expert_granularity == "layer":
        stats, costs, candidates = aggregate_moe_candidates(
            stats, costs, specs_sorted, candidates, granularity="projection",
            calibrated_gains=calibrated_gains)
        moe_groups = sum(1 for n in candidates if ".__fused__." in n)
        print(f"[alloc] MoE aggregation: {moe_groups} fused-expert super-Linears")

    candidates = filter_candidates_for_profile(candidates, args.target_profile)

    # Pareto sweep
    targets = [float(x) for x in args.pareto_targets.split(",")]
    curve = []
    for t in targets:
        assignment, achieved = solve_with_promotion(
            stats, candidates, t, format_specs, format_rank,
            args.bit_precision,
            no_fused_promote=args.no_fused_promote,
            overshoot_tolerance=args.overshoot_tolerance,
        )
        if assignment is None:
            curve.append({"target_bits": t, "feasible": False})
            continue
        total_dloss = 0.0
        format_counts = defaultdict(int)
        format_params = defaultdict(int)
        for name, fmt in assignment.items():
            entry = costs[name].get(fmt, {})
            gain = float(calibrated_gains.get(fmt, 1.0))
            total_dloss += predicted_dloss(
                stats[name]["h_trace"], entry.get("weight_mse", 0.0), gain=gain,
            )
            format_counts[fmt] += 1
            format_params[fmt] += stats[name]["n_params"]
        curve.append({
            "target_bits": t,
            "feasible": True,
            "achieved_bits": achieved,
            "predicted_dloss": total_dloss,
            **{f"layers_{k}": v for k, v in format_counts.items()},
            **{f"params_{k}": v for k, v in format_params.items()},
        })

    # Output Pareto CSV
    keys = sorted({k for row in curve for k in row.keys()})
    with open(args.pareto_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in curve:
            w.writerow(row)
    print(f"[alloc] Pareto curve → {args.pareto_csv}")

    # Kneedle
    feasible = [row for row in curve if row.get("feasible")]
    if len(feasible) >= 3:
        kidx = kneedle([r["achieved_bits"] for r in feasible],
                       [r["predicted_dloss"] for r in feasible])
        knee = feasible[kidx]
        print(f"[alloc] suggested knee: target={knee['target_bits']}, "
              f"achieved={knee['achieved_bits']:.3f}, "
              f"Δloss={knee['predicted_dloss']:.3e}")

    # Print table
    print("\n  target  achieved     Δloss (pred)   " + "   ".join(
        f"{s.name[:11]:>11}" for s in specs_sorted))
    for row in curve:
        if not row.get("feasible"):
            print(f"  {row['target_bits']:>6.3f}  INFEASIBLE")
            continue
        fmt_str = "   ".join(
            f"{row.get(f'layers_{s.name}', 0):>11,}" for s in specs_sorted)
        print(f"  {row['target_bits']:>6.3f}  {row['achieved_bits']:>7.3f}  "
              f"{row['predicted_dloss']:>14.4e}   {fmt_str}")

    # Emit chosen layer_config for target_bits
    assignment, achieved = solve_with_promotion(
        stats, candidates, args.target_bits, format_specs, format_rank,
        args.bit_precision,
        no_fused_promote=args.no_fused_promote,
        overshoot_tolerance=args.overshoot_tolerance,
    )
    if assignment is None:
        raise SystemExit(
            f"Infeasible at target_bits={args.target_bits}. "
            "Consider raising the target or widening the format set.")

    # Expand MoE super-Linears back to per-expert entries before writing
    # the AutoRound layer_config (which expects one entry per individual
    # nn.Linear module name).
    if args.expert_granularity == "layer":
        assignment_expanded = expand_moe_assignment(assignment, stats)
    else:
        assignment_expanded = assignment

    layer_cfg = {}
    for name, fmt in assignment_expanded.items():
        layer_cfg[name] = format_specs[fmt].autoround_config()

    out = Path(args.layer_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(layer_cfg, f, indent=2)

    counts = defaultdict(int)
    for fmt in assignment.values():
        counts[fmt] += 1
    print(f"\n[alloc] target={args.target_bits} achieved={achieved:.3f}")
    for fmt, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {fmt:>14}: {n:>5} layers")
    print(f"\nLayer config → {out}")
    print(f"Feed to AutoRound via --layer_config {out}")


if __name__ == "__main__":
    main()
