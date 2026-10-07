"""Grouped FIT price shrinkage and deterministic multi-choice body-byte allocator.

Research-only solver slice for the fine-grained per-block feasibility trial
(eng-pq-fine-grained, RobTand/prismaquant#2329).  Exactly two pure numpy
routines; no Tessera, capture, GPU or production-allocator imports.

``shrink_group_prices``
    Per-price-point shrinkage of FIT loss deltas toward the block-group mean:
    ``group_mean + shrinkage * (block_delta - group_mean)``.  Input deltas are
    measured relative to the baseline choice, whose column is zero by input
    convention, so the baseline column stays exactly zero after shrinkage.
    FIT data only; the routine never sees held-out anything.

``allocate_body_budget``
    One choice per block under an integer body-byte cap.  Lagrangian
    ``argmin(price + lambda * bytes)`` with lambda bisection to the byte
    boundary, then most-beneficial integer upgrades (strictly negative price
    deltas only) while under cap. Zero-byte-delta alternatives are canonicalised to
    the lowest loss first, so free improvements cannot be missed and exact
    duplicate (bytes, price) options resolve to the lowest option index.
    Per-block candidate chains keep only the Pareto frontier (fewer bytes and
    lower-or-equal loss dominates), but the chain itself may be non-convex and
    the argmin handles that.

    The multi-choice knapsack is NOT solved to global optimality: non-convex
    frontier pockets and the single-move greedy repair can miss a better mixed
    plan -- the documented reason the production allocator uses a DP.  So
    ``predicted_loss`` is proxy bookkeeping, never a quality claim.  A
    positive-price option is bought only when the cap forces bytes below it;
    unused capacity is reported as-is and padding is never purchased; a mixed
    plan that comes out worse than the feasible all-baseline plan is returned
    exactly as computed and flagged (``mixed_plan_worse_than_baseline``),
    never silently swapped for a baseline result.

Deterministic: no randomness, first-occurrence argmin ties on chains sorted
by (bytes asc, price asc, option index asc), totally ordered heap keys.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

__all__ = ["shrink_group_prices", "allocate_body_budget"]


def _finite_float_matrix(values, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == bool or not (
        np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)
    ):
        raise ValueError(f"{name} must be a numeric array, got dtype {arr.dtype}")
    arr = arr.astype(np.float64, copy=True)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D [B,P], got shape {arr.shape}")
    if arr.size and not np.isfinite(arr).all():
        first = tuple(int(v) for v in np.argwhere(~np.isfinite(arr))[0])
        raise ValueError(f"{name} must be finite; first non-finite entry at {first}")
    return arr


def _int_scalar(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def shrink_group_prices(fit_loss_deltas, groups, shrinkage=0.25) -> np.ndarray:
    """Shrink per-block FIT loss deltas toward their group mean.

    Args:
        fit_loss_deltas: finite [B,P] array; deltas relative to the baseline
            choice, whose column is zero by input convention.
        groups: integer [B] group id per block (arbitrary integer values).
        shrinkage: weight on the block's own residual in [0, 1].

    Returns:
        New float64 [B,P] array: ``group_mean[p] + shrinkage * (delta[b,p] -
        group_mean[p])`` per price point.  With the zero-baseline convention
        the baseline column stays exactly zero.
    """
    deltas = _finite_float_matrix(fit_loss_deltas, "fit_loss_deltas")
    blocks, points = deltas.shape

    s = float(shrinkage)
    if not math.isfinite(s) or not 0.0 <= s <= 1.0:
        raise ValueError(f"shrinkage must lie in [0, 1], got {shrinkage!r}")

    group_ids = np.asarray(groups)
    if group_ids.dtype == bool or not np.issubdtype(group_ids.dtype, np.integer):
        raise ValueError(f"groups must be an integer array, got dtype {group_ids.dtype}")
    group_ids = group_ids.astype(np.int64, copy=False).reshape(-1)
    if group_ids.shape[0] != blocks:
        raise ValueError(
            f"groups has {group_ids.shape[0]} entries for {blocks} blocks"
        )
    if blocks == 0:
        return deltas

    _, inverse = np.unique(group_ids, return_inverse=True)
    inverse = inverse.reshape(-1)
    means = np.zeros((int(inverse.max()) + 1, points), dtype=np.float64)
    np.add.at(means, inverse, deltas)
    counts = np.bincount(inverse).astype(np.float64)
    means /= counts[:, None]
    return means[inverse] + s * (deltas - means[inverse])


def allocate_body_budget(prices, body_bytes, cap_bytes, baseline_index) -> dict:
    """Pick one option per block under an integer body-byte cap.

    Args:
        prices: finite float [B,P] loss deltas; the baseline column is zero by
            input convention (negative = improvement).
        body_bytes: integer nonnegative [B,P] exact body byte counts.
        cap_bytes: integer total body-byte cap.
        baseline_index: option index that is the do-nothing choice.

    Returns:
        Dict with ``selection`` (int64 [B]), ``used_body_bytes``,
        ``unused_body_bytes``, ``predicted_loss`` and honest ``algorithm`` /
        ``limits`` bookkeeping.  Raises ``ValueError`` if even the minimum
        body exceeds the cap.
    """
    price = _finite_float_matrix(prices, "prices")
    blocks, points = price.shape

    raw_bytes = np.asarray(body_bytes)
    if raw_bytes.dtype == bool or not np.issubdtype(raw_bytes.dtype, np.integer):
        raise ValueError(f"body_bytes must be an integer array, got dtype {raw_bytes.dtype}")
    if raw_bytes.shape != price.shape:
        raise ValueError(
            f"body_bytes shape {raw_bytes.shape} does not match prices {price.shape}"
        )
    byte = raw_bytes.astype(np.int64)
    if byte.size and int(byte.min()) < 0:
        first = tuple(int(v) for v in np.argwhere(byte < 0)[0])
        raise ValueError(f"body_bytes must be nonnegative; negative entry at {first}")

    cap = _int_scalar(cap_bytes, "cap_bytes")
    if cap < 0:
        raise ValueError(f"cap_bytes must be nonnegative, got {cap}")
    baseline = _int_scalar(baseline_index, "baseline_index")
    if not 0 <= baseline < points:
        raise ValueError(f"baseline_index {baseline} outside [0, {points})")

    empty = {
        "selection": np.zeros(0, dtype=np.int64),
        "used_body_bytes": 0,
        "unused_body_bytes": cap,
        "predicted_loss": 0.0,
        "algorithm": {
            "name": "lagrangian-bisection-greedy-repair",
            "bisection_iterations": 0,
            "lambda_final": 0.0,
            "lambda_searched_high": 0.0,
            "fill_upgrade_moves": 0,
            "tie_order": (
                "argmin first occurrence on chains sorted by "
                "(bytes asc, price asc, option index asc); "
                "repair keys (ratio, -bytes, block)"
            ),
            "optimality": (
                "none: Lagrangian candidate plus greedy repair; the multi-choice "
                "knapsack is not solved to global optimality"
            ),
        },
        "limits": {
            "cap_bytes": cap,
            "min_body_bytes": 0,
            "baseline_body_bytes": 0,
            "baseline_plan_feasible": bool(0 <= cap),
            "mixed_plan_worse_than_baseline": False,
            "held_out_used": False,
        },
    }
    if blocks == 0:
        return empty

    # Canonical per-block chains: sort by (bytes, price, index), collapse
    # identical byte counts to the cheapest option (zero-byte-delta support),
    # then keep the Pareto frontier (strictly cheaper as bytes grow).
    grid = np.tile(np.arange(points, dtype=np.int64), (blocks, 1))
    order = np.lexsort((grid, price, byte), axis=-1)
    s_byte = np.take_along_axis(byte, order, axis=-1)
    s_price = np.take_along_axis(price, order, axis=-1)
    first_of_byte = np.ones((blocks, points), dtype=bool)
    if points > 1:
        first_of_byte[:, 1:] = s_byte[:, 1:] != s_byte[:, :-1]
    running = np.minimum.accumulate(np.where(first_of_byte, s_price, np.inf), axis=-1)
    prev = np.empty_like(running)
    prev[:, 0] = np.inf
    if points > 1:
        prev[:, 1:] = running[:, :-1]
    keep = first_of_byte & (s_price < prev)

    width = int(keep.sum(axis=-1).max())
    rows = np.arange(blocks)
    flat_pos = (np.cumsum(keep, axis=-1) - 1)[keep]
    flat_rows = np.repeat(rows, keep.sum(axis=-1))
    flat = flat_rows * width + flat_pos
    chain_byte = np.zeros((blocks, width), dtype=np.int64)
    chain_price = np.full((blocks, width), np.inf, dtype=np.float64)
    chain_idx = np.full((blocks, width), -1, dtype=np.int64)
    chain_byte.reshape(-1)[flat] = s_byte[keep]
    chain_price.reshape(-1)[flat] = s_price[keep]
    chain_idx.reshape(-1)[flat] = order[keep]

    min_total = int(chain_byte[:, 0].sum())
    if min_total > cap:
        raise ValueError(
            f"minimum body {min_total} bytes exceeds cap {cap} bytes; "
            "no feasible selection exists"
        )
    baseline_total = int(byte[:, baseline].sum())

    def gather(pos: np.ndarray) -> tuple[int, float]:
        used = int(chain_byte[rows, pos].sum())
        loss = float(chain_price[rows, pos].sum())
        return used, loss

    def lagrangian_config(lam: float) -> np.ndarray:
        score = chain_price + float(lam) * chain_byte
        return np.argmin(score, axis=-1)  # first occurrence: smallest bytes on ties

    lam_hi = 2.0 * float(np.abs(price).max()) + 1.0
    iterations = 0
    pos = lagrangian_config(0.0)
    used, _ = gather(pos)
    lo, hi = 0.0, lam_hi
    if used > cap:
        pos = lagrangian_config(lam_hi)  # min-bytes choice per block: feasible
        used, _ = gather(pos)
        while iterations < 128 and hi - lo > 1e-12 * max(1.0, lam_hi):
            mid = 0.5 * (lo + hi)
            candidate = lagrangian_config(mid)
            if gather(candidate)[0] <= cap:
                hi, pos = mid, candidate
            else:
                lo = mid
            iterations += 1
    cur = pos.astype(np.int64, copy=True)
    used, _ = gather(cur)

    version = np.zeros(blocks, dtype=np.int64)

    def best_upgrade(block: int, remaining: int, banned: set):
        c = int(cur[block])
        ks = [k for k in range(c + 1, width) if k not in banned]
        if not ks or remaining <= 0:
            return None
        ks_arr = np.asarray(ks, dtype=np.int64)
        gained = chain_byte[block, ks_arr] - chain_byte[block, c]
        cost = chain_price[block, ks_arr] - chain_price[block, c]
        ok = cost < 0.0  # beneficial only; never buy to fill bytes
        if not ok.any():
            return None
        ks_arr, gained, cost = ks_arr[ok], gained[ok], cost[ok]
        fits = gained <= remaining
        if not fits.any():
            return None
        ks_arr, gained, cost = ks_arr[fits], gained[fits], cost[fits]
        ratio = cost / gained.astype(np.float64)
        rank = np.lexsort((ks_arr, -gained, ratio))
        k = int(ks_arr[rank[0]])
        return (
            float(cost[rank[0]] / float(gained[rank[0]])),
            -float(gained[rank[0]]),
            block,
            k,
            int(version[block]),
        )

    if used > cap:
        raise RuntimeError("Lagrangian selection exceeded its feasible byte cap")
    # Fill: while capacity remains, take only strictly beneficial upgrades.
    fill_moves = 0
    banned = [set() for _ in range(blocks)]
    heap = []
    remaining = cap - used
    for b in range(blocks):
        entry = best_upgrade(b, remaining, banned[b])
        if entry is not None:
            heapq.heappush(heap, entry)
    while remaining > 0 and heap:
        _ratio, neg_gained, b, k, ver = heapq.heappop(heap)
        if ver != int(version[b]) or k <= int(cur[b]):
            continue
        if int(-neg_gained) > remaining:
            banned[b].add(k)  # capacity only shrinks; this k can never fit again
            entry = best_upgrade(b, remaining, banned[b])
            if entry is not None:
                heapq.heappush(heap, entry)
            continue
        cur[b] = k
        version[b] += 1
        fill_moves += 1
        remaining -= int(-neg_gained)
        if remaining > 0:
            entry = best_upgrade(b, remaining, banned[b])
            if entry is not None:
                heapq.heappush(heap, entry)

    selection = chain_idx[rows, cur].astype(np.int64, copy=True)
    used = int(chain_byte[rows, cur].sum())
    predicted = float(price[rows, selection].sum())
    return {
        "selection": selection,
        "used_body_bytes": used,
        "unused_body_bytes": cap - used,
        "predicted_loss": predicted,
        "algorithm": {
            "name": "lagrangian-bisection-greedy-repair",
            "bisection_iterations": int(iterations),
            "lambda_final": float(hi if iterations else 0.0),
            "lambda_searched_high": float(lam_hi),
            "fill_upgrade_moves": int(fill_moves),
            "tie_order": (
                "argmin first occurrence on chains sorted by "
                "(bytes asc, price asc, option index asc); "
                "repair keys (ratio, -bytes, block)"
            ),
            "optimality": (
                "none: Lagrangian candidate plus greedy repair; the multi-choice "
                "knapsack is not solved to global optimality"
            ),
        },
        "limits": {
            "cap_bytes": cap,
            "min_body_bytes": min_total,
            "baseline_body_bytes": baseline_total,
            "baseline_plan_feasible": bool(baseline_total <= cap),
            "mixed_plan_worse_than_baseline": bool(
                baseline_total <= cap and predicted > 0.0
            ),
            "held_out_used": False,
        },
    }

