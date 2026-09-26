"""Bit-rate (ship-bpp) selection for AURA — two principled selectors, no kneedle.

The 2026-06 Aura work showed the surrogate rate-distortion curve is a clean
exponential (KL ~ 10^(-a*bpp), straight in log-KL), so there is **no intrinsic
knee**: the classic kneedle picks the point of maximum curvature, which on a
log-linear curve is purely an artifact of how you scale the axes (KL vs log-KL,
bpp vs bytes — the 27B knee moved 7.5 -> 12 bpp across axis choices). The ship
bpp is therefore not a curvature to find; it is set by one of two real anchors:

  * CONSTRAINED ("fit the card") — ``select_under_byte_budget``: you target a
    GPU SKU (24/32 GB). Feasibility is exact and needs no KL measurement: an
    allocation ships only if its *exact* exported footprint
    (``prismaquant.footprint``) fits the budget. Among the rungs that fit,
    ``allocator.main`` ships the one with minimum predicted Δloss (ties to the
    larger footprint) — NOT simply the densest, because the RD curve is not
    monotone in bytes (§5: 5.5 bpp beat 6.0 bpp on measured 4B PPL). This is
    the real ship selector.

  * UNCONSTRAINED ("where do bits stop paying") — ``find_saturation_bpp``: with
    no byte budget, pick the *saturation* point on the **measured** distortion:
    the lowest bpp whose distortion is statistically indistinguishable (within
    z * combined stderr) from the high-bpp asymptote. The only non-arbitrary
    stopping anchor is the measurement noise floor — you cannot ship a quality
    gain you cannot distinguish from sampling scatter. One knob (z, a
    significance level — not a curvature unit), unit-free, anchored to a real
    quantity. It scans the candidate grid rather than assuming monotonic noisy
    measurements; when called from validated-frontier selection the grid has
    already been measured. NB: the band is z * combined stderr, so the measured
    frontier must carry a real per-bpp stderr (calib-repeats >= 4, or a
    per-position bootstrap); a single-rep stderr of 0 collapses the band so that
    only the asymptote is within-noise of itself, degenerating B* to the
    highest-bpp asymptote (the densest / safest allocation, i.e. ship the most
    bits) — not lowest-bpp.

Both are selectors over Pareto candidates (never a post-allocator rewrite), in
the spirit of the kneedle they replace.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence


def log_error_values(values: Sequence[float]) -> list[float]:
    """Map measured distortion (KL or predicted dloss) to log10 for a kneedle.

    Non-positive values are floored at the smallest positive value itself. A
    measured distortion <= 0 (fp32 round-off on a near-passthrough assignment,
    or an all-passthrough rung on an FP8-native source whose total dloss is
    exactly 0) means "at the floor of what this run can resolve", not "orders
    of magnitude better than every real point". Flooring at min_positive puts
    such points exactly 0 decades below the smallest real point; any lower
    floor (the old ``min_positive * 1e-6``) fabricates a multi-decade cliff in
    normalized log-space that compresses the real curve and drags the kneedle
    to the curve start, the worst point on the ship path. The allocator's
    surrogate knee and the validated frontier's measured knee share this one
    axis (#1394).
    """
    finite_positive = [
        float(value) for value in values
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    if not finite_positive:
        return [0.0 for _ in values]
    floor = min(finite_positive)
    return [math.log10(max(float(value), floor)) for value in values]


def bootstrap_mean_ci(
    samples: Sequence[float], *, n_boot: int = 1000, seed: int = 0,
) -> tuple[float, float]:
    """Return (mean, stderr_of_mean) via bootstrap over per-position values.

    Deterministic given seed. stderr is the std of bootstrap-resampled means,
    which is the noise floor on the mean distortion at one bpp."""
    n = len(samples)
    if n == 0:
        return 0.0, 0.0
    mean = sum(samples) / n
    if n == 1:
        return mean, float("inf")
    # Deterministic LCG bootstrap (no numpy dependency required at call site).
    state = (seed * 2654435761 + 12345) & 0xFFFFFFFF
    boot_means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            s += samples[state % n]
        boot_means.append(s / n)
    bm = sum(boot_means) / n_boot
    var = sum((m - bm) ** 2 for m in boot_means) / max(1, n_boot - 1)
    return mean, math.sqrt(var)


def find_saturation_bpp(
    grid: Sequence[float],
    measure_fn: Callable[[float], tuple[float, float]],
    *,
    z: float = 2.0,
    scan: str = "dense",
) -> dict:
    """Find B* = lowest bpp in `grid` whose distortion is within the noise band
    of the asymptote (highest bpp).

    Args:
      grid: sorted-ascending bpp candidates (allocations precomputed at each).
      measure_fn: bpp -> (distortion_mean, distortion_stderr). Memoize externally
        if you like; this calls each requested bpp once.
      z: significance multiplier on the combined stderr (2.0 ~= 95%).

    Returns dict with B* ('bpp'), the measured points, and the decision trace.
    Every grid point is checked against the asymptote band, so marginal
    non-monotone measurement noise cannot make an early bisection decision hide
    a lower saturated point.
    """
    g = sorted(grid)
    measured: dict[float, tuple[float, float]] = {}
    trace: list[dict] = []

    def m(bpp):
        if bpp not in measured:
            measured[bpp] = measure_fn(bpp)
        return measured[bpp]

    kl_hi, se_hi = m(g[-1])

    def _check(idx):
        bpp = g[idx]
        kl_m, se_m = m(bpp)
        band = z * math.hypot(se_m, se_hi)
        within = (kl_m - kl_hi) <= band
        trace.append({
            "bpp": bpp, "kl": kl_m, "se": se_m,
            "kl_asymptote": kl_hi, "band": band, "within_noise": within,
        })
        return within

    # scan modes (QC on review-batch): the dense scan is robust to
    # non-monotone noise but turns the documented O(log n) measurement
    # contract into O(n) — unacceptable when measure_fn is a LIVE
    # GPU KL measurement. 'auto' bisects first (O(log n) live calls),
    # then densifies only the already-measured-free region below the
    # bisection answer when every grid point is memoized externally.
    # 'dense' preserves the fully-robust behavior for precomputed grids.
    if scan not in ("auto", "dense", "bisect"):
        raise ValueError(f"unknown scan mode {scan!r}")
    best_i: int | None = None
    if scan == "dense":
        for idx in range(len(g)):
            if _check(idx) and best_i is None:
                best_i = idx
    else:
        lo, hi = 0, len(g) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if _check(mid):
                hi = mid
            else:
                lo = mid + 1
        best_i = lo
        if scan == "auto":
            # noise-robustness pass at zero extra measurement cost:
            # re-examine any grid point ALREADY measured during bisection
            # that sits below the bisection answer and is within the band.
            for idx in range(best_i):
                if g[idx] in measured and _check(idx):
                    best_i = idx
                    break
    if best_i is None or not _check(best_i):
        best_i = len(g) - 1
    # Slope view between adjacent measured points (transparency / sanity).
    pts = sorted(measured.items())
    slopes = []
    for (b0, (k0, _)), (b1, (k1, _)) in zip(pts, pts[1:]):
        if b1 > b0:
            slopes.append({"from": b0, "to": b1, "dKL_per_bit": (k1 - k0) / (b1 - b0)})
    return {
        "bpp": g[best_i],
        "kl_at_bstar": measured[g[best_i]][0],
        "kl_asymptote": kl_hi,
        "asymptote_bpp": g[-1],
        "z": z,
        "n_measurements": len(measured),
        "measured": {b: {"kl": v[0], "se": v[1]} for b, v in measured.items()},
        "slopes": slopes,
        "trace": trace,
    }


def select_under_byte_budget(
    candidates: Sequence[Mapping],
    budget_bytes: float,
    *,
    bytes_key: str = "disk_bytes",
    bpp_key: str = "bpp",
    loss_key: str = "dloss",
) -> dict:
    """Fit-the-card selection: the largest allocation that fits ``budget_bytes``.

    ``candidates`` is the Pareto set, each a mapping carrying at least the
    exported on-disk size (``bytes_key``); ``bpp_key`` / ``loss_key`` are used
    only for tie-breaking and reporting (a missing loss is treated as +inf).
    The chosen point is the one with the **largest footprint that still fits**
    the budget — the most bits you can afford. Ties on bytes break to lower
    loss, then higher bpp.

    IMPORTANT — this function is the FEASIBILITY GATE, not the ship objective.
    "Fill the card" would be the quality-optimal feasible point only if the RD
    curve were monotone in bytes, and it is not: CLAUDE.md §5 records 5.5 bpp
    beating 6.0 bpp on measured Qwen3-4B PPL, and since the solver began
    ratcheting on minimum predicted Δloss among feasible iterates, a denser
    rung can carry a strictly worse Δloss than a sparser one that also fits.
    The shipping selector in ``allocator.main`` therefore uses this function's
    ``feasible`` / ``below_floor`` verdict and its candidate set, but picks by
    **minimum predicted Δloss among the rungs that fit** (ties to the larger
    footprint) and records that objective in ``selection.json``
    (``ratchet_objective``). ``chosen`` is still returned, and the allocator
    records it as ``max_bytes_pick_*`` so the two objectives stay auditable
    side by side.

    Returns a dict with the decision and full context:
      ``chosen``           selected candidate (None if none fit),
      ``feasible``         True iff at least one candidate fits,
      ``below_floor``      True iff even the smallest candidate exceeds budget
                           (the card cannot hold the cheapest allocation),
      ``has_slack``        True iff every candidate fits (budget exceeds the
                           densest allocation; ``slack_bytes`` is the unused room),
      ``headroom_bytes``   budget − chosen footprint (>= 0; unused card space),
      ``rejected_next``    the cheapest candidate that did NOT fit (the next rung
                           up you could not afford), or None,
      ``budget_bytes``, ``n_candidates``.
    """
    cands = [c for c in candidates if c.get(bytes_key) is not None]
    if not cands:
        return {
            "chosen": None, "feasible": False, "below_floor": True,
            "has_slack": False, "headroom_bytes": None, "rejected_next": None,
            "budget_bytes": float(budget_bytes), "n_candidates": 0,
        }

    def _loss(c):
        v = c.get(loss_key)
        return float(v) if v is not None else math.inf

    def _bpp(c):
        v = c.get(bpp_key)
        return float(v) if v is not None else -math.inf

    fitting = [c for c in cands if float(c[bytes_key]) <= float(budget_bytes)]
    too_big = [c for c in cands if float(c[bytes_key]) > float(budget_bytes)]
    rejected_next = (
        min(too_big, key=lambda c: float(c[bytes_key])) if too_big else None
    )
    if not fitting:
        # Even the cheapest allocation overflows the card.
        cheapest = min(cands, key=lambda c: float(c[bytes_key]))
        return {
            "chosen": None, "feasible": False, "below_floor": True,
            "has_slack": False, "headroom_bytes": None,
            "rejected_next": cheapest,
            "budget_bytes": float(budget_bytes), "n_candidates": len(cands),
        }
    # Largest footprint that fits; ties -> lower loss, then higher bpp.
    chosen = max(
        fitting,
        key=lambda c: (float(c[bytes_key]), -_loss(c), _bpp(c)),
    )
    return {
        "chosen": chosen,
        "feasible": True,
        "below_floor": False,
        "has_slack": not too_big,
        "headroom_bytes": float(budget_bytes) - float(chosen[bytes_key]),
        "rejected_next": rejected_next,
        "budget_bytes": float(budget_bytes),
        "n_candidates": len(cands),
    }
