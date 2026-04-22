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
# Fused-projection sibling detection
# ---------------------------------------------------------------------------
# Profile-driven: we group Linears by the key returned by
# `profile.fused_sibling_group(qname)`. Profiles default-derive that
# from the matching vLLM class's `packed_modules_mapping` attribute,
# so adding new architectures doesn't require new allocator code.

def _group_by_profile(names, profile) -> dict[str, list[str]]:
    """Group Linear names by the profile's fused-sibling key. Names
    that don't belong to any fused group are returned with their own
    unique key so they pass through the promotion logic untouched."""
    groups: dict[str, list[str]] = {}
    for name in names:
        key = profile.fused_sibling_group(name) if profile is not None else None
        if key is None:
            continue
        groups.setdefault(key, []).append(name)
    return groups


def fused_siblings(name: str, profile=None) -> tuple[tuple[str, ...], str] | None:
    """Legacy scalar sibling lookup — resolves a single Linear to its
    fused group + the group's "kind" label. Kept for backward
    compatibility; `_group_by_profile` is the path new code should use."""
    if profile is None:
        # Fall back to the default profile — still auto-derives from
        # the vLLM class if one is available.
        from .model_profiles import DefaultProfile
        profile = DefaultProfile()
    key = profile.fused_sibling_group(name)
    if key is None:
        return None
    # Return a single-element sibling tuple (the allocator's promote_fused
    # fills in the full group from all tracked names that share `key`
    # via _group_by_profile; this legacy path only needs enough info
    # for a single-element grouping).
    return (name,), key


def promote_moe_pair(assignment: dict[str, str],
                     format_rank: dict[str, int]) -> dict[str, str]:
    """Couple MoE expert projections within a layer to share a format.

    vLLM's FusedMoE layer requires that all projections it fuses share
    one quantization scheme, else `get_moe_method` raises 'All MoE
    projections need to have same quantization scheme but found
    multiple' at load time. Two naming conventions show up in
    practice and both need promotion:

    1. Post-fusion form (Qwen3 and similar): `experts.gate_up_proj` +
       `experts.down_proj` — a single pair per layer.
    2. Per-expert pre-fusion form (MiniMax, Mixtral-style): each
       expert keeps its own `experts.N.w1`, `experts.N.w2`,
       `experts.N.w3` (or `gate_proj` / `up_proj` / `down_proj`) — one
       triple per (layer, expert) pair, thousands per model.

    Pattern (1) is detected via a projection-name match at the leaf
    (`gate_up_proj|down_proj`). Pattern (2) is detected via the
    `\\.experts\\.\\d+\\.(w[123]|gate_proj|up_proj|down_proj)$` shape,
    which groups by `(layer_prefix, expert_idx)`. Both patterns are
    promoted to the highest-rank format in each group. Idempotent;
    safe to stack with `promote_fused`."""
    out = dict(assignment)
    groups: dict[tuple[str, str], list[str]] = {}

    # Pattern 1: post-fusion (.experts.gate_up_proj / .experts.down_proj)
    post_fused_re = re.compile(r"^(.+\.experts)\.(gate_up_proj|down_proj)$")
    # Pattern 2: per-expert pre-fusion (.experts.N.<leaf>)
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
            # One group per (experts-prefix, expert_idx). Collapsing
            # across expert indices would over-promote; each expert's
            # triple is an independent fused unit.
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
    """After per-Linear selection, bump each fused group's siblings to
    the highest-rank format picked for any group member.

    Uses `profile.fused_sibling_group(qname)` to decide which Linears
    belong to the same fused group. The profile default-derives its
    groups from vLLM's `packed_modules_mapping`, so arch-specific
    knowledge about fused tensors lives in one place (vLLM's model
    class) rather than in PrismaQuant's allocator.

    Raises `AssertionError` if the post-promotion assignment still has
    a fused group with inconsistent formats — that's a bug (either in
    promotion logic or profile sibling-detection) that silently
    produces unservable quantized artifacts (vLLM can't dispatch a
    single scheme to a fused Linear whose siblings disagree). We trade
    a loud crash at allocation time for a quiet failure at serving
    time."""
    if profile is None:
        from .model_profiles import DefaultProfile
        profile = DefaultProfile()
    out = dict(assignment)
    groups = _group_by_profile(assignment.keys(), profile)
    for members_present in groups.values():
        if len(members_present) < 2:
            # A group of 1 is a singleton — nothing to promote to.
            continue
        ranks = [format_rank[out[m]] for m in members_present]
        best = max(ranks)
        best_fmt = next(k for k, v in format_rank.items() if v == best)
        for m in members_present:
            if format_rank[out[m]] < best:
                out[m] = best_fmt

    # Post-condition verification: no fused group may have mixed fmts
    # after promotion. Any violation is an upstream bug (e.g. profile
    # fused_sibling_group returning inconsistent keys for siblings).
    for group_key, members in groups.items():
        if len(members) < 2:
            continue
        fmts = {out[m] for m in members}
        if len(fmts) > 1:
            detail = ", ".join(f"{m}={out[m]}" for m in members)
            raise AssertionError(
                f"promote_fused post-check failed for group {group_key!r}: "
                f"siblings have mixed formats after promotion — {detail}. "
                f"This produces an unservable artifact (vLLM dispatches "
                f"one scheme per fused Linear).")
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
    max_iters: int = 40,
    stall_threshold: float = 1e-4,
    stall_grace: int = 3,
    profile=None,
) -> tuple[dict[str, str] | None, float]:
    """Solve the allocation, promote fused siblings, and re-solve with a
    tightened target if promotion blew past the budget.

    Promotion is allowed to inflate the achieved bits because vLLM's
    fused tensor loader requires a single format per fused group. The
    natural fix — already employed implicitly by the previous version —
    is to reserve some headroom in the DP for promotion. We make this
    explicit and adaptive: if promotion overshoots the requested target
    by more than `overshoot_tolerance` bits/param, halve the overshoot
    by tightening the next solve, and repeat.

    The loop exits when ANY of:
      * overshoot ≤ `overshoot_tolerance` (success — budget respected)
      * `stall_grace` consecutive iterations with |Δovershoot| < `stall_threshold`
        (no meaningful progress — further iteration can't help)
      * `max_iters` hard cap
      * DP returns infeasible (target tightened below achievable floor)

    Stall detection lets us iterate "as long as each step is buying
    something" rather than giving up after a fixed count. On dense
    models the naive halving converges quickly (2-5 iters); on
    highly-coupled MoE models it sometimes oscillates and needs
    patience. The stall guard prevents infinite loops when the
    coupling structure makes the target unreachable.

    Returns (assignment, achieved_bits). Assignment is None if even the
    untightened solve was infeasible.
    """
    tightened = float(target_bits)
    last_assign: dict[str, str] | None = None
    last_achieved = float("nan")
    prev_overshoot = float("inf")
    stall_count = 0
    for iteration in range(max_iters):
        assign = solve_allocation(stats, candidates, tightened, bit_precision)
        if assign is None:
            return last_assign, last_achieved
        if not no_fused_promote:
            assign = promote_fused(assign, format_rank, profile=profile)
        # MoE pair coupling: gate_up_proj + down_proj within the same
        # layer must share a format (vLLM FusedMoE requirement). Run
        # unconditionally — there's no "skip promote" flag for this
        # because it's a hard correctness constraint, not an
        # optimization; the unservable artifact is silent otherwise.
        assign = promote_moe_pair(assign, format_rank)
        achieved, _ = compute_achieved(stats, assign, format_specs)
        last_assign = assign
        last_achieved = achieved
        overshoot = achieved - target_bits
        if overshoot <= overshoot_tolerance:
            return assign, achieved

        # Stall detection: if two consecutive iterations make
        # <stall_threshold bpp of progress on overshoot, we've hit the
        # structural floor (usually promotion coupling). Further
        # tightening just churns.
        if abs(prev_overshoot - overshoot) < stall_threshold:
            stall_count += 1
            if stall_count >= stall_grace:
                return assign, achieved
        else:
            stall_count = 0
        prev_overshoot = overshoot

        # Tighten by half the overshoot. Converges geometrically on
        # well-behaved problems.
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


def _format_kernel_supports_shape(fmt_name: str, in_features: int,
                                  out_features: int) -> bool:
    """Return True iff the runtime GEMM kernel for `fmt_name` can handle
    a Linear with (in_features, out_features).

    Two-layer dispatch:

    1. **Prefer the kernel's own check function** when exposed (e.g.
       FlashInfer's `_check_mm_mxfp8_problem_size`). Running the real
       kernel's validator keeps us in sync with kernel updates
       automatically — when FlashInfer relaxes or tightens the rules,
       we inherit the change for free.

    2. **Hand-curated fallback** when no check function is exposed, or
       when flashinfer isn't installed (allocator runs on CPU-only
       hosts without CUDA). Tile-alignment rules for modern FP8/FP4
       tensor-core kernels cluster around 128 (TMA tile) × 32 (MX
       block) × 16 (NVFP group). These are the documented minimums.

    Unknown formats default to True — allocator-time should not
    silently drop experimental formats that haven't been profiled.

    Both layers must accept before a shape passes. The flashinfer
    common check covers generic dtype/alignment rules; the hand-
    curated table captures backend-specific constraints that the
    generic check doesn't see (e.g., CUTLASS SM121 MXFP8 requires
    N % 128 while the common check only asks for N >= 128).
    """
    # Layer 1: try the kernel's authoritative validator. If it
    # explicitly rejects, we're done — no backend will accept.
    flashinfer_verdict = _flashinfer_kernel_accepts(
        fmt_name, in_features, out_features)
    if flashinfer_verdict is False:
        return False

    # Layer 2: hand-curated tile-alignment fallback.
    # Even if flashinfer's common check passed, backend-specific
    # rules may still reject (e.g., cutlass SM121 N % 128 tile).
    if fmt_name.startswith("MXFP8"):
        # CUTLASS mm_mxfp8 (SM100/121 Blackwell):
        #   - N ≥ 128, K ≥ 128 (generic check)
        #   - K % 32 (MX block size)
        #   - N % 128 (TMA tile alignment). Example miss: (1152, 4304)
        #     where 4304 % 128 = 80 — caught empirically on DGX Spark
        #     SM121 with vLLM's cutlass-MXFP8 kernel.
        if out_features < 128 or in_features < 128:
            return False
        if in_features % 32 != 0:
            return False
        if out_features % 128 != 0:
            return False
        return True
    if fmt_name.startswith("NVFP4"):
        # CUTLASS mm_nvfp4: K must be a multiple of the 16-wide group.
        return in_features % 16 == 0
    # BF16 and unknown: pass through.
    return True


def _flashinfer_kernel_accepts(fmt_name: str, in_features: int,
                               out_features: int) -> bool | None:
    """Ask FlashInfer's own problem-size check for `fmt_name`.

    Returns True/False if the kernel exposes a check function we can
    invoke with zero-element tensor stubs; None if no validator is
    available (caller falls back to hand-curated rules).

    This is cheap — FlashInfer's checks only inspect shapes/dtypes,
    they don't allocate CUDA memory or compile. Import is lazy so the
    allocator still works on CPU-only hosts where flashinfer is absent.
    """
    try:
        if fmt_name.startswith("MXFP8"):
            from flashinfer.gemm.gemm_base import _check_mm_mxfp8_problem_size
            import torch
            # Zero-element fp8 stubs with the right shape metadata.
            a = torch.empty((1, in_features), dtype=torch.float8_e4m3fn)
            b = torch.empty((in_features, out_features),
                            dtype=torch.float8_e4m3fn)
            # Match the swizzled-1d scale layout the runtime uses.
            from flashinfer.gemm.gemm_base import _mxfp8_swizzled_scale_len
            from flashinfer.gemm.gemm_base import SfLayout
            a_desc_len = _mxfp8_swizzled_scale_len(
                a.shape[0], a.shape[1], SfLayout.layout_8x4)
            b_desc_len = _mxfp8_swizzled_scale_len(
                b.shape[1], b.shape[0], SfLayout.layout_8x4)
            a_desc = torch.empty((a_desc_len,), dtype=torch.uint8)
            b_desc = torch.empty((b_desc_len,), dtype=torch.uint8)
            try:
                return _check_mm_mxfp8_problem_size(a, b, a_desc, b_desc) is True
            except Exception:
                return False
        # NVFP4 / BF16 / others: no flashinfer check exposed today —
        # caller will use hand-curated rules.
        return None
    except Exception:
        # flashinfer not installed, or an internal import failed — the
        # hand-curated fallback is authoritative in that case.
        return None


def build_candidates(stats: dict, costs: dict, formats: list[fr.FormatSpec],
                     calibrated_gains: dict[str, float] | None = None
                     ) -> dict[str, list[Candidate]]:
    """For each Linear, build its candidate list (one per format).

    Per-(layer, format) predicted Δloss uses the closed-form
    diagonal-Fisher term `0.5 · h_trace · weight_mse` (see module
    docstring), optionally scaled by per-format calibrated_gains[fmt].

    Format candidates are gated on kernel shape support: a Linear
    whose (in_features, out_features) violates the runtime GEMM's
    alignment rules (e.g., CUTLASS MXFP8's N ≥ 128) drops that format
    from its candidate list — the knapsack can't pick an unservable
    option. Dropping these at candidate-build time is cleaner than
    waiting for serve-time kernel dispatch failures.
    """
    gains = calibrated_gains or {}
    out: dict[str, list[Candidate]] = {}
    masked_by_shape: dict[str, list[str]] = {}
    for name, s in stats.items():
        if name not in costs:
            continue
        h_trace = s["h_trace"]
        shape = _shape_from_stats(s)
        in_features = int(s.get("in_features", 0) or 0)
        out_features = int(s.get("out_features", 0) or 0)
        cands = []
        for spec in formats:
            entry = costs[name].get(spec.name)
            if entry is None or "error" in entry:
                continue
            # Kernel shape check — skip formats whose runtime GEMM
            # can't handle this Linear's (in, out).
            if in_features and out_features and not _format_kernel_supports_shape(
                spec.name, in_features, out_features
            ):
                masked_by_shape.setdefault(spec.name, []).append(name)
                continue
            gain = float(gains.get(spec.name, 1.0))
            # Prefer the full per-weight Δloss `0.5 · <H_full, MSE_W_full>`
            # emitted by measure_quant_cost when h_detail was available.
            # Falls back to the scalar proxy `0.5 · h_trace · weight_mse`
            # for legacy cost pickles. Both are scalars with units of Δloss,
            # so the knapsack DP treats them interchangeably — the full
            # form is just a sharper estimator.
            if "predicted_dloss" in entry:
                predicted = float(entry["predicted_dloss"]) * gain
            else:
                weight_mse = float(entry.get("weight_mse", 0.0))
                predicted = predicted_dloss(h_trace, weight_mse, gain=gain)
            cands.append(Candidate(
                fmt=spec.name,
                bits_per_param=spec.effective_bits_for_shape(shape),
                memory_bytes=spec.memory_bytes_for_shape(shape),
                predicted_dloss=max(predicted, 0.0),
            ))
        if cands:
            out[name] = cands
    if masked_by_shape:
        for fmt, names in masked_by_shape.items():
            print(f"[alloc] kernel shape-mask: {len(names)} Linear(s) "
                  f"dropped {fmt} (sample: {names[:3]})", flush=True)
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

            # True summed Δloss across all members at format f. Uses the
            # full per-weight Fisher `predicted_dloss` from cost.pkl when
            # available (sharper), falls back to the scalar-proxy
            # `0.5 · h_trace · weight_mse` for legacy cost pickles.
            sum_pred = 0.0
            for m_ in members:
                c = costs.get(m_, {}).get(spec.name)
                if c is None or "error" in c:
                    c = {"weight_mse": mean_weight_mse,
                         "output_mse": mean_output_mse}
                if "predicted_dloss" in c:
                    sum_pred += float(c["predicted_dloss"])
                else:
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
                "predicted_dloss": sum_pred,       # exact summed Δloss
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


_FUSED_SIBLING_MARKER = ".__siblings__."


def aggregate_fused_siblings(
    stats: dict,
    costs: dict,
    formats: list[fr.FormatSpec],
    candidates: dict[str, list[Candidate]],
    profile,
    calibrated_gains: dict[str, float] | None = None,
) -> tuple[dict, dict, dict]:
    """Pre-aggregate fused-sibling groups (qkv_proj, gate_up_proj, etc.)
    into single DP items so the knapsack sees the coupling constraint
    directly.

    vLLM's fused-tensor loader requires q/k/v (and gate/up) to share one
    quantization scheme. Historically PrismaQuant solved per-Linear then
    ran `promote_fused` as a post-pass — which inflates achieved bits
    whenever the DP picked different formats for siblings, and needed a
    tightening retry loop to chase the budget. Aggregating siblings at
    build time eliminates the inflation entirely: the DP just can't pick
    mixed-sibling solutions because the option doesn't exist.

    Mirrors `aggregate_moe_candidates` numerically — the super-Linear's
    summed Δloss at format f equals Σ_i predicted_dloss(h_i, mse_{i,f}).
    Inverting via `effective_mse = sum_pred / (0.5 · sum_h)` lets
    build_candidates' closed-form formula reproduce sum_pred exactly.

    Single-member groups (Linears that don't fuse with anything — o_proj,
    down_proj, norms, lm_head) pass through unchanged.

    Returns (stats_ext, costs_ext, candidates_ext). MoE expert aggregation
    is expected to have run FIRST so this only sees non-expert candidates.
    """
    if profile is None:
        return stats, costs, candidates

    gains = calibrated_gains or {}

    # Partition candidate keys by their fused-sibling group key.
    grouped: dict[str, list[str]] = {}
    ungrouped: list[str] = []
    for name in candidates:
        # Already-aggregated MoE super-Linears use `.__fused__.` and must
        # NOT be re-grouped. Pass them through.
        if ".__fused__." in name:
            ungrouped.append(name)
            continue
        try:
            key = profile.fused_sibling_group(name)
        except Exception:
            key = None
        if key is None:
            ungrouped.append(name)
            continue
        grouped.setdefault(key, []).append(name)

    # Single-member "groups" also pass through — aggregating one Linear
    # into a super-Linear buys nothing.
    for key in list(grouped.keys()):
        if len(grouped[key]) < 2:
            ungrouped.extend(grouped.pop(key))

    if not grouped:
        return stats, costs, candidates

    stats_ext = {n: stats[n] for n in ungrouped}
    costs_ext = {n: costs.get(n, {}) for n in ungrouped}
    candidates_ext = {n: candidates[n] for n in ungrouped}

    for key, members in grouped.items():
        members = sorted(members)
        # `key` is a string like "model.layers.3.self_attn.qkv_proj" or
        # "mtp.layers.0.mlp.gate_up_proj" — use it as the super-Linear name
        # with a disambiguator so we can recognize it at expansion time.
        safe_key = key.replace(".", "__")
        super_name = f"{members[0].rsplit('.', 1)[0]}{_FUSED_SIBLING_MARKER}{safe_key}"

        # Summed statistics.
        n_params = sum(stats[m]["n_params"] for m in members)
        sum_h = sum(stats[m]["h_trace"] for m in members)
        d_out = int(stats[members[0]].get("out_features", 0) or 0)
        d_in = int(stats[members[0]].get("in_features", 0) or 0)

        stats_ext[super_name] = {
            "h_trace": sum_h,
            "h_trace_raw": sum(stats[m].get("h_trace_raw", 0.0) for m in members),
            "h_w2_sum": sum(stats[m].get("h_w2_sum", 0.0) for m in members),
            "w_max_abs": max(stats[m].get("w_max_abs", 0.0) for m in members),
            "w_norm_sq": sum(stats[m].get("w_norm_sq", 0.0) for m in members),
            "n_params": n_params,
            "in_features": d_in,
            "out_features": d_out,
            "n_tokens_seen": sum(stats[m].get("n_tokens_seen", 0) for m in members),
            "_fused_siblings": members,
            "_memory_bytes_by_format": {},
        }

        # Per-format: group_pred = SUM of per-sibling predicted_dloss.
        #
        # This matches the cost model's underlying math. Under the Fisher-
        # diagonal approximation:
        #
        #     Δloss ≈ 0.5 · Σᵢ hᵢ · (ΔWᵢ)²
        #
        # Δloss is additive over parameters, therefore additive over
        # Linears, therefore additive over siblings. A sibling's quantization
        # contributes its own per-Linear Δloss to the total, independent of
        # whether it's grouped with other siblings at serve time.
        #
        # Alternatives considered and rejected:
        #   max(dloss_i):        "sensitive sibling sets the floor" — this
        #                        conflates a safety argument (reality of
        #                        attention's multiplicative interaction)
        #                        with the cost-model question. Safety is
        #                        better expressed as a format-constraint
        #                        rule ("this Linear must be ≥ MXFP8"),
        #                        not by biasing the cost aggregation.
        #   max(dloss_i) × n:    no principled basis; preserves format
        #                        ranking (all formats scale uniformly) so
        #                        equivalent to max in DP effect, but
        #                        semantically a no-op dressed up as a fix.
        #
        # Empirical note (2026-04-22): an earlier version of this code used
        # max × n, motivated by a 35B-A3B perplexity measurement of 10 000.
        # That measurement was contaminated by vLLM's spec-decode + echo
        # logprobs path returning draft-model NLL. Re-measured with the
        # fixed validator, sum-aggregation and max × n produce artifacts
        # with indistinguishable perplexity (~4.15 vs ~4.01 baseline,
        # within noise). Sum is correct on principle, so we use it.
        super_cost = {}
        for spec in formats:
            missing = [m for m in members
                       if spec.name not in costs.get(m, {})
                       or "error" in costs.get(m, {}).get(spec.name, {})]
            if missing:
                super_cost[spec.name] = {"error": "partial"}
                continue
            sum_pred = 0.0
            for m in members:
                c = costs[m][spec.name]
                if "predicted_dloss" in c:
                    sum_pred += float(c["predicted_dloss"])
                else:
                    h_i = stats[m]["h_trace"]
                    sum_pred += 0.5 * h_i * float(c.get("weight_mse", 0.0))
            effective_mse = sum_pred / (0.5 * sum_h) if sum_h > 0 else 0.0
            super_cost[spec.name] = {
                "weight_mse": effective_mse,
                "predicted_dloss": sum_pred,
            }
        costs_ext[super_name] = super_cost

        # Super-Linear can only use a format that EVERY member supported.
        # If a member's shape was masked out of MXFP8 (DeltaNet
        # in_proj_a with out_features=48, for example), the fused
        # group as a whole inherits that restriction — the super's
        # shape equals each member's (siblings share both in and out).
        member_format_sets = [
            {c.fmt for c in candidates.get(m, [])}
            for m in members
        ]
        if member_format_sets:
            member_format_intersection = set.intersection(*member_format_sets)
        else:
            member_format_intersection = set()

        # Candidate list for the super-Linear.
        cands = []
        for spec in formats:
            if spec.name not in member_format_intersection:
                # At least one sibling had this format shape-masked —
                # the fused group can't use it either.
                continue
            entry = super_cost.get(spec.name)
            if entry is None or "error" in entry:
                continue
            total_bytes = 0
            for m in members:
                shape = _shape_from_stats(stats[m])
                total_bytes += spec.memory_bytes_for_shape(shape)
            bits_per_param = 8.0 * total_bytes / max(n_params, 1)
            stats_ext[super_name]["_memory_bytes_by_format"][spec.name] = total_bytes
            gain = float(gains.get(spec.name, 1.0))
            predicted = entry["predicted_dloss"] * gain
            cands.append(Candidate(
                fmt=spec.name,
                bits_per_param=bits_per_param,
                memory_bytes=total_bytes,
                predicted_dloss=max(predicted, 0.0),
            ))
        if cands:
            candidates_ext[super_name] = cands

    return stats_ext, costs_ext, candidates_ext


def expand_fused_sibling_assignment(assignment: dict[str, str],
                                    stats_ext: dict) -> dict[str, str]:
    """Replace `.__siblings__.` super-Linear assignments with per-member
    assignments (all members sharing the super-Linear's format)."""
    out = {}
    for name, fmt in assignment.items():
        if _FUSED_SIBLING_MARKER in name:
            members = stats_ext[name].get("_fused_siblings", [])
            for m in members:
                out[m] = fmt
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
# Visual encoder override
# ---------------------------------------------------------------------------
# Phase 1 visual-encoder support: the probe's text-only calibration does not
# exercise the visual tower, so per-Linear Fisher gradients for
# `model.visual.blocks.*` Linears are zero — the knapsack DP has no
# sensitivity signal to allocate on. Rather than let every visual Linear
# default to the cheapest format or go through stale passthrough, we accept
# a single uniform target format (`BF16`, `NVFP4`, or `MXFP8`) and assign
# every visual Linear to it. BF16 (the default) reproduces the previous
# passthrough behavior; NVFP4/MXFP8 shrink the tower to quantized storage
# using the same RTN math the body gets.
#
# Phase 2 (tracked separately) will replace this override with a real
# multimodal Fisher: load images + text, run full forward through the
# visual encoder → projector → body LM, capture per-Linear empirical Fisher
# gradients, and feed those into the allocator's closed-form Δloss. That
# requires a multimodal dataset loader, multimodal tokenizer wiring, and a
# probe path that doesn't strip the visual tower — none of which ship in
# Phase 1.
_VISUAL_PREFIX_RE = re.compile(r"^(?:model\.)?visual\.")


def _is_visual_linear(name: str) -> bool:
    """True when `name` refers to a Linear inside the visual encoder.

    Matches both the raw HF checkpoint form (`model.visual.blocks.*`) and
    the post-remap form (`visual.blocks.*`) so the override behaves the
    same regardless of which side of `profile.live_to_recipe_name` the
    allocator's stats dictionary landed on.
    """
    return bool(_VISUAL_PREFIX_RE.match(name))


def apply_visual_format_override(
    assignment: dict[str, str],
    visual_format: str,
) -> dict[str, str]:
    """Force every visual-encoder Linear in `assignment` to `visual_format`.

    Called after the knapsack DP + fused-sibling promotion so the override
    wins even if the solver would have picked a different format per
    per-Linear sensitivity noise (which is meaningless for visual Linears
    under text-only calibration — see module comment above).

    `visual_format="BF16"` is a no-op when a visual Linear already has no
    allocator entry (the export's existing passthrough keeps it at BF16);
    we still write `BF16` into the returned assignment so the layer_config
    round-trip is explicit and downstream tooling (export, validate) has a
    uniform record of the decision.
    """
    out = dict(assignment)
    for name in list(out.keys()):
        if _is_visual_linear(name):
            out[name] = visual_format
    return out


def discover_visual_linears_from_source(model_path: str) -> list[str]:
    """Scan the source safetensors index for `model.visual.blocks.*.weight`
    entries with rank-2 shapes — these are the Linear modules the visual
    encoder exposes.

    Returned names are the basename (`.weight` stripped) so they slot
    directly into the allocator's assignment dictionary and the exporter's
    quantize-by-recipe dispatch.

    The probe's text-only staging strips the visual tower, so visual
    Linears never appear in the probe or cost pickles. This helper lets
    the allocator emit a layer_config entry for them anyway when
    `--visual-format` is non-BF16 — the exporter can then quantize each
    of them uniformly under the requested format. Without this scan, the
    allocator has no way to enumerate visual Linear names (there is no
    in-memory visual module at allocation time).
    """
    src = Path(model_path)
    idx_path = src / "model.safetensors.index.json"
    candidates: list[tuple[str, tuple[int, ...]]] = []
    if idx_path.exists():
        with open(idx_path) as f:
            wm = json.load(f).get("weight_map", {})
        # Index file carries only names, not shapes. We need to open each
        # referenced shard once to read rank.
        from collections import defaultdict as _dd
        by_shard: dict[str, list[str]] = _dd(list)
        for key, shard in wm.items():
            if not key.endswith(".weight"):
                continue
            if not _VISUAL_PREFIX_RE.match(key):
                continue
            by_shard[shard].append(key)
        try:
            from safetensors import safe_open
        except ImportError:
            return []
        for shard, keys in by_shard.items():
            shard_path = src / shard
            if not shard_path.exists():
                continue
            with safe_open(str(shard_path), framework="pt") as sf:
                for k in keys:
                    try:
                        shape = tuple(sf.get_slice(k).get_shape())
                    except Exception:
                        continue
                    candidates.append((k, shape))
    else:
        # No index file — scan every safetensors shard directly. Used for
        # small, single-file checkpoints.
        try:
            from safetensors import safe_open
        except ImportError:
            return []
        import os as _os
        if not src.exists():
            return []
        for f in sorted(_os.listdir(src)):
            if not f.endswith(".safetensors"):
                continue
            with safe_open(str(src / f), framework="pt") as sf:
                for k in sf.keys():
                    if not k.endswith(".weight"):
                        continue
                    if not _VISUAL_PREFIX_RE.match(k):
                        continue
                    try:
                        shape = tuple(sf.get_slice(k).get_shape())
                    except Exception:
                        continue
                    candidates.append((k, shape))

    # Only rank-2 weights are Linear-like; conv1d / norms / biases are
    # kept at BF16 passthrough regardless of --visual-format.
    # Additionally, blacklist known rank-2 tensors that live in
    # `nn.Parameter` / `nn.Embedding` modules (NOT `nn.Linear`), which
    # the compressed-tensors loader in vLLM cannot consume. Example:
    # `model.visual.pos_embed.weight` is an Embedding-like learned
    # parameter with shape (num_pos, hidden) — rank-2 but NOT a Linear.
    # Quantizing it produces `pos_embed.input_global_scale` etc. which
    # vLLM's VL runtime rejects with `KeyError: pos_embed.input_global_scale`
    # because its `model.visual.pos_embed` is a bare Parameter, not a
    # quantizable Linear module.
    _NON_LINEAR_RE = re.compile(
        r"(?:^|\.)("
        r"pos_embed"            # positional embedding (nn.Parameter/Embedding)
        r"|rotary_emb"          # rotary pos embed cache
        r")(?:\.|$)"
    )
    out: list[str] = []
    for name, shape in candidates:
        if len(shape) != 2:
            continue
        if _NON_LINEAR_RE.search(name):
            continue
        out.append(name[:-len(".weight")] if name.endswith(".weight") else name)
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="sensitivity_probe pickle")
    ap.add_argument("--costs", required=True, help="measure_quant_cost pickle")
    ap.add_argument("--model-override", default=None,
                    help="Override the model path stored in probe.pkl's meta. "
                         "Useful when re-running allocator against a probe "
                         "whose container-side paths no longer exist (e.g., "
                         "the original source was at /src/qwen36 in a prior "
                         "container run but is now only accessible via a "
                         "different mount). Overrides both profile detection "
                         "and visual-Linear source discovery.")
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
    ap.add_argument("--no-fused-aggregation", action="store_true",
                    help="Disable pre-DP aggregation of fused siblings "
                         "(qkv_proj / gate_up_proj). Falls back to the "
                         "legacy promote_fused post-pass with tightening "
                         "retries. Pre-aggregation is strictly better "
                         "for hitting the target bit budget exactly on "
                         "dense models; use this flag only for "
                         "back-compat experiments.")
    ap.add_argument("--enforce-family-coherence", action="store_true",
                    help="Error (instead of warn) if the format set contains "
                         "multiple candidates for the same bit tier (e.g. "
                         "NVFP4 and MXFP4 both at 4 bits)")
    ap.add_argument("--bit-precision", type=float, default=0.0001,
                    help="Knapsack bit-bin granularity in avg-bits/param "
                         "(smaller = slower; default 0.0001 → ~50000 bins). "
                         "Measured on MiniMax-M2.7: going from 0.001 to 0.0001 "
                         "cuts predicted Δloss ~10% at the same bit budget. "
                         "Coarser values (0.01) leave 40% on the table.")
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
    ap.add_argument("--visual-format",
                    choices=["BF16", "NVFP4", "MXFP8"],
                    default="BF16",
                    help="Uniform format for all visual-encoder Linears "
                         "(`model.visual.blocks.*`). Phase 1 fallback: "
                         "assigned to every visual Linear when "
                         "--visual-sensitivity=uniform OR when --visual-"
                         "sensitivity=fisher but the probe / cost pickles "
                         "don't carry real visual Fisher data. BF16 (default) "
                         "reproduces passthrough behavior; NVFP4 / MXFP8 "
                         "shrink the tower to quantized storage via the "
                         "existing RTN math at export time.")
    ap.add_argument("--visual-sensitivity",
                    choices=["fisher", "uniform"],
                    default="fisher",
                    help="How visual-encoder Linears enter the allocator. "
                         "'fisher' (default) treats them as regular DP "
                         "candidates when the probe pickle carries real "
                         "multimodal Fisher stats (produced by "
                         "`incremental_probe --calibration-modality="
                         "multimodal`). If those stats are missing, falls "
                         "back to uniform --visual-format. 'uniform' forces "
                         "the Phase 1 path: every visual Linear gets "
                         "--visual-format regardless of what's in the probe.")
    args = ap.parse_args()

    if args.threads > 0:
        import os
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    # Detect the model profile from the probe's metadata. The probe
    # writes `meta.model` when it runs, so we can look up the HF
    # config at that path and map it to a registered ModelProfile.
    # Profile governs fused-sibling promotion (allocator's
    # `promote_fused`) and the vLLM-internal name remap
    # (`build_quantization_config` via export_native_compressed).
    from .model_profiles import detect_profile, DefaultProfile
    model_profile = DefaultProfile()
    with open(args.probe, "rb") as f:
        _probe_peek = pickle.load(f)
    probe_model_path = _probe_peek.get("meta", {}).get("model")
    del _probe_peek
    if args.model_override:
        probe_model_path = args.model_override
        print(f"[alloc] model-override: {probe_model_path}", flush=True)
    if probe_model_path:
        model_profile = detect_profile(probe_model_path)
        print(f"[alloc] model profile: {model_profile.name} "
              f"(derived from {probe_model_path})", flush=True)

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

    # Pre-aggregate fused siblings (qkv_proj, gate_up_proj, ...) into
    # single DP items. The DP can't pick mixed-sibling solutions because
    # there's only one item per group — so promote_fused becomes a no-op
    # on aggregated items and the overshoot-tightening loop collapses to
    # a single pass on well-behaved models. Must run AFTER the MoE
    # aggregation (it skips `.__fused__.` entries explicitly).
    if not args.no_fused_aggregation:
        stats, costs, candidates = aggregate_fused_siblings(
            stats, costs, specs_sorted, candidates, profile=model_profile,
            calibrated_gains=calibrated_gains)
        sib_groups = sum(1 for n in candidates if _FUSED_SIBLING_MARKER in n)
        print(f"[alloc] fused-sibling aggregation: {sib_groups} groups "
              f"(qkv_proj / gate_up_proj / ...)")

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
            profile=model_profile,
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
        profile=model_profile,
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

    # Expand fused-sibling super-Linears (qkv_proj / gate_up_proj).
    # Complementary to the MoE expansion above — the MoE path handles
    # `.__fused__.` markers; the sibling path handles `.__siblings__.`
    # markers. Running both is idempotent on any assignment that doesn't
    # contain the relevant marker.
    if not args.no_fused_aggregation:
        assignment_expanded = expand_fused_sibling_assignment(
            assignment_expanded, stats)

    # Post-expansion MoE unity promotion. The super-Linear solver picks
    # a format per (layer, projection); expansion propagates that to
    # all per-expert leaves with the same projection. But vLLM's
    # FusedMoE requires ALL projections (gate+up+down) of the SAME
    # expert to share one scheme — so when the solver picks e.g.
    # w1=MXFP8 and w2=NVFP4 for layer L, every (L, expert_i) triple
    # would disagree and the serve-time dispatch raises 'All MoE
    # projections need same scheme'. This pass groups expanded leaves
    # by (experts-prefix, expert-idx) and bumps the lower-rank
    # projections to match the highest. Costs a little bpp on layers
    # where gate/up/down really wanted different formats, but that
    # cost is the ticket for serveability.
    assignment_expanded = promote_moe_pair(assignment_expanded, format_rank)

    # Visual-encoder Linear handling. Two paths:
    #
    # 1. --visual-sensitivity=fisher (default) + probe/cost have real
    #    visual entries → visual Linears already participated in the
    #    knapsack DP above with their own per-Linear Fisher + per-format
    #    RTN cost. No override needed; just make sure every discoverable
    #    visual Linear has an assignment entry (fall back to --visual-
    #    format for any that the probe missed, e.g. patch_embed Linears
    #    that the probe's regex didn't hit).
    #
    # 2. --visual-sensitivity=uniform OR Fisher missing → Phase 1 path:
    #    scan source checkpoint for visual Linears and stamp them all
    #    with --visual-format.
    visual_format = args.visual_format
    visual_sensitivity = args.visual_sensitivity

    def _visual_fisher_available(stats_d: dict, costs_d: dict) -> bool:
        """True when both the probe and cost pickles carry real visual
        entries — the signal a multimodal calibration pass ran."""
        any_visual_stats = any(_is_visual_linear(n) for n in stats_d)
        any_visual_costs = any(_is_visual_linear(n) for n in costs_d)
        return any_visual_stats and any_visual_costs

    fisher_visual_ok = (visual_sensitivity == "fisher"
                        and _visual_fisher_available(stats, costs))
    if visual_sensitivity == "fisher" and not fisher_visual_ok:
        print("[alloc] --visual-sensitivity=fisher requested but probe / "
              "cost pickles have no visual Linear entries; falling back "
              f"to --visual-format={visual_format} (Phase 1 uniform).",
              flush=True)

    if probe_model_path:
        visual_names_src = discover_visual_linears_from_source(probe_model_path)
    else:
        visual_names_src = []

    if fisher_visual_ok:
        # Fisher path: DP already placed visual Linears. Fill in any
        # discoverable visual Linear that the DP missed (e.g. the probe
        # regex matched only `visual.blocks.*` but the source has
        # `visual.merger.*` or `visual.patch_embed.*` too) with the
        # uniform --visual-format as a safety net.
        dp_visual_count = sum(1 for n in assignment_expanded
                              if _is_visual_linear(n))
        filled = 0
        for vname in visual_names_src:
            if vname not in assignment_expanded:
                assignment_expanded[vname] = visual_format
                filled += 1
        print(f"[alloc] --visual-sensitivity=fisher: DP placed "
              f"{dp_visual_count} visual Linears via per-Linear Fisher; "
              f"{filled} additional visual Linears (un-probed) stamped "
              f"with --visual-format={visual_format}.", flush=True)
    else:
        # Uniform path (Phase 1): stamp every discoverable visual Linear.
        if visual_names_src:
            for vname in visual_names_src:
                assignment_expanded[vname] = visual_format
            print(f"[alloc] --visual-format={visual_format}: assigned "
                  f"{len(visual_names_src)} visual Linears uniformly "
                  f"(source={probe_model_path})", flush=True)
        elif visual_format != "BF16":
            print(f"[alloc] --visual-format={visual_format}: no visual "
                  f"Linears found in source checkpoint — override is a "
                  f"no-op", flush=True)

    layer_cfg = {}
    for name, fmt in assignment_expanded.items():
        if fmt in format_specs:
            layer_cfg[name] = format_specs[fmt].autoround_config()
        else:
            # Visual format outside the body's format set (e.g., user
            # passed --formats NVFP4,BF16 plus --visual-format MXFP8).
            # Resolve from the global registry.
            layer_cfg[name] = fr.get_format(fmt).autoround_config()

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
