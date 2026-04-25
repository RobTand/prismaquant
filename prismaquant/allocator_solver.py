"""Solver primitives for PrismaQuant allocation.

This module owns the multi-choice knapsack, candidate scoring, and
serve-time coupling promotions.  ``allocator.py`` keeps the CLI and
re-exports these symbols for backwards compatibility.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import format_registry as fr


@dataclass
class Candidate:
    fmt: str
    bits_per_param: float
    memory_bytes: int
    predicted_dloss: float
    pruned_expert_ids: tuple[int, ...] = ()


def _shape_from_stats(entry: dict) -> tuple[int, ...]:
    out_features = int(entry.get("out_features", 0) or 0)
    in_features = int(entry.get("in_features", 0) or 0)
    if out_features > 0 and in_features > 0:
        return (out_features, in_features)
    n_params = int(entry.get("n_params", 0) or 0)
    return (n_params,)


def predicted_dloss(h_trace: float, weight_mse: float,
                    gain: float = 1.0) -> float:
    """Per-(layer, format) predicted loss under the diagonal-Fisher model."""
    return 0.5 * float(h_trace) * float(weight_mse) * float(gain)


def _group_by_profile(names, profile) -> dict[str, list[str]]:
    """Group Linear names by the profile's fused-sibling key."""
    groups: dict[str, list[str]] = {}
    for name in names:
        key = profile.fused_sibling_group(name) if profile is not None else None
        if key is None:
            continue
        groups.setdefault(key, []).append(name)
    return groups


def fused_siblings(name: str, profile=None) -> tuple[tuple[str, ...], str] | None:
    """Legacy scalar sibling lookup kept for backward compatibility."""
    if profile is None:
        from .model_profiles import DefaultProfile
        profile = DefaultProfile()
    key = profile.fused_sibling_group(name)
    if key is None:
        return None
    return (name,), key


def promote_moe_pair(assignment: dict[str, str],
                     format_rank: dict[str, int]) -> dict[str, str]:
    """Promote MoE expert projections that must share one serving format."""
    out = dict(assignment)
    groups: dict[tuple[str, str], list[str]] = {}

    post_fused_re = re.compile(r"^(.+\.experts)\.(gate_up_proj|down_proj)$")
    per_expert_re = re.compile(
        r"^(.+\.experts)\.(\d+)\.(w1|w2|w3|gate_proj|up_proj|down_proj)$"
    )

    for name in assignment:
        m = post_fused_re.match(name)
        if m:
            groups.setdefault((m.group(1), "__post__"), []).append(name)
            continue
        m = per_expert_re.match(name)
        if m:
            groups.setdefault((m.group(1), m.group(2)), []).append(name)

    for members in groups.values():
        if len(members) < 2:
            continue
        ranks = [format_rank[out[m]] for m in members]
        best = max(ranks)
        best_fmt = next(k for k, v in format_rank.items() if v == best)
        for m in members:
            if format_rank[out[m]] < best:
                out[m] = best_fmt
    return out


def promote_fused(assignment: dict[str, str],
                  format_rank: dict[str, int],
                  profile=None) -> dict[str, str]:
    """Promote each fused-sibling group to one shared serving format."""
    if profile is None:
        from .model_profiles import DefaultProfile
        profile = DefaultProfile()
    out = dict(assignment)
    groups = _group_by_profile(assignment.keys(), profile)
    for members_present in groups.values():
        if len(members_present) < 2:
            continue
        ranks = [format_rank[out[m]] for m in members_present]
        best = max(ranks)
        best_fmt = next(k for k, v in format_rank.items() if v == best)
        for m in members_present:
            if format_rank[out[m]] < best:
                out[m] = best_fmt

    for group_key, members in groups.items():
        if len(members) < 2:
            continue
        fmts = {out[m] for m in members}
        if len(fmts) > 1:
            detail = ", ".join(f"{m}={out[m]}" for m in members)
            raise AssertionError(
                f"promote_fused post-check failed for group {group_key!r}: "
                f"siblings have mixed formats after promotion - {detail}. "
                "This produces an unservable artifact."
            )
    return out


def solve_allocation(stats: dict, candidates: dict[str, list[Candidate]],
                     target_bits: float, bit_precision: float = 0.001
                     ) -> tuple[dict[str, str], dict[str, Candidate]] | None:
    """Solve multi-choice knapsack in average-bits-per-parameter units."""
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

    excess = target_bits - min_bits
    n_bins = int(round(excess / bit_precision)) + 2

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

        opt_dbins = np.asarray([o[0] for o in options], dtype=np.int32)
        opt_dgain = np.asarray([o[1] for o in options], dtype=np.float64)
        opt_idx = np.asarray([o[2] for o in options], dtype=np.int32)

        new_dp = np.full(n_bins, INF_NEG, dtype=np.float64)
        new_choice = np.full(n_bins, -1, dtype=np.int32)

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

    assignment: dict[str, str] = {}
    chosen_cands: dict[str, Candidate] = {}
    cur = best_b
    for layer_idx in range(len(names) - 1, -1, -1):
        idx_chosen = int(choice[layer_idx][cur])
        name = names[layer_idx]
        cs = candidates[name]
        if idx_chosen < 0:
            idx_chosen = 0
        chosen = cs[idx_chosen]
        assignment[name] = chosen.fmt
        chosen_cands[name] = chosen
        baseline = baselines[name]
        params = stats[name]["n_params"]
        fraction = params / total_params
        d_avg_bits = (chosen.bits_per_param
                      - baseline.bits_per_param) * fraction
        cur -= int(round(d_avg_bits / bit_precision))
        if cur < 0:
            cur = 0
    return assignment, chosen_cands


def _candidate_for_assignment(
    name: str,
    fmt: str,
    candidates: dict[str, list[Candidate]],
    pruned_map: dict[str, tuple[int, ...]] | None = None,
) -> Candidate | None:
    """Resolve the scored candidate for one assignment entry."""
    cands_for_name = candidates.get(name, [])
    target_drops = tuple(sorted((pruned_map or {}).get(name, ())))
    for cand in cands_for_name:
        if cand.fmt == fmt and tuple(sorted(cand.pruned_expert_ids)) == target_drops:
            return cand
    if target_drops:
        available = [
            (c.fmt, tuple(sorted(c.pruned_expert_ids)))
            for c in cands_for_name
        ]
        raise AssertionError(
            f"assignment {name!r} picked fmt={fmt!r} with pruned_expert_ids="
            f"{target_drops}, but no exact candidate exists. Available "
            f"candidates: {available[:8]}"
        )
    for cand in cands_for_name:
        if cand.fmt == fmt:
            return cand
    return None


def compute_achieved(stats: dict, assignment: dict[str, str],
                     format_specs: dict[str, fr.FormatSpec],
                     candidates: dict[str, list[Candidate]] | None = None,
                     pruned_map: dict[str, tuple[int, ...]] | None = None,
                     ) -> tuple[float, float]:
    """Return ``(avg_bits, total_predicted_dloss)`` for an assignment."""
    total_params = sum(stats[n]["n_params"] for n in assignment)
    total_bits = 0.0
    pm = pruned_map or {}
    cs = candidates or {}
    for n in assignment:
        fmt = assignment[n]
        chosen_cand = _candidate_for_assignment(n, fmt, cs, pm)
        if chosen_cand is not None:
            total_bits += 8.0 * chosen_cand.memory_bytes
            continue
        memory_map = stats[n].get("_memory_bytes_by_format")
        if memory_map is not None and fmt in memory_map:
            total_bits += 8.0 * memory_map[fmt]
        else:
            shape = _shape_from_stats(stats[n])
            total_bits += (
                format_specs[fmt].effective_bits_for_shape(shape)
                * stats[n]["n_params"]
            )
    return total_bits / max(total_params, 1), 0.0


def compute_assignment_predicted_dloss(
    assignment: dict[str, str],
    candidates: dict[str, list[Candidate]],
    pruned_map: dict[str, tuple[int, ...]] | None = None,
) -> float:
    """Sum predicted loss for a concrete assignment."""
    total = 0.0
    pm = pruned_map or {}
    for name, fmt in assignment.items():
        chosen = _candidate_for_assignment(name, fmt, candidates, pm)
        if chosen is None:
            raise AssertionError(
                f"assignment {name!r} picked fmt={fmt!r}, but no candidate "
                "exists to price its predicted loss"
            )
        total += chosen.predicted_dloss
    return total


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
    max_iters: int = 40,
    stall_threshold: float = 1e-4,
    stall_grace: int = 3,
    profile=None,
) -> tuple[dict[str, str] | None, dict[str, tuple[int, ...]], float]:
    """Solve, promote coupled tensors, and retry if promotion exceeds budget."""
    tightened = float(target_bits)
    last_assign: dict[str, str] | None = None
    last_pruned: dict[str, tuple[int, ...]] = {}
    last_achieved = float("nan")
    prev_overshoot = float("inf")
    stall_count = 0
    for _iteration in range(max_iters):
        result = solve_allocation(stats, candidates, tightened, bit_precision)
        if result is None:
            return last_assign, last_pruned, last_achieved
        assign, chosen_cands = result
        pruned_map = {
            n: c.pruned_expert_ids
            for n, c in chosen_cands.items()
            if c.pruned_expert_ids
        }
        if not no_fused_promote:
            assign = promote_fused(assign, format_rank, profile=profile)
        assign = promote_moe_pair(assign, format_rank)
        achieved, _ = compute_achieved(
            stats, assign, format_specs,
            candidates=candidates, pruned_map=pruned_map,
        )
        last_assign = assign
        last_pruned = pruned_map
        last_achieved = achieved
        overshoot = achieved - target_bits
        if overshoot <= overshoot_tolerance:
            return assign, pruned_map, achieved

        if abs(prev_overshoot - overshoot) < stall_threshold:
            stall_count += 1
            if stall_count >= stall_grace:
                return assign, pruned_map, achieved
        else:
            stall_count = 0
        prev_overshoot = overshoot

        tightened -= overshoot / 2.0
        if tightened <= 0:
            break
    return last_assign, last_pruned, last_achieved
