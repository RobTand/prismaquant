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
    boundary, then greedy integer repair: cheapest loss-per-byte downgrades
    while over cap, most-beneficial upgrades (strictly negative price deltas
    only) while under cap.  Zero-byte-delta alternatives are canonicalised to
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
``--self-check`` runs the boundary, zero-cost, tie, cap-refusal and
exhaustive-oracle algorithm cases (algorithm correctness only, not a
quantization-quality result).
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import sys

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
            "trim_downgrade_moves": 0,
            "fill_upgrade_moves": 0,
            "zero_byte_delta_alternatives_dropped": 0,
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
    dropped = int(price.size - keep.sum())

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

    def best_downgrade(block: int):
        c = int(cur[block])
        if c <= 0:
            return None
        seg_byte = chain_byte[block, :c]
        seg_price = chain_price[block, :c]
        saved = float(chain_byte[block, c]) - seg_byte.astype(np.float64)
        cost = float(chain_price[block, c]) - seg_price
        ratio = cost / saved  # saved > 0, cost > 0 on a strict Pareto chain
        rank = np.lexsort((np.arange(c), -saved, ratio))
        j = int(rank[0])
        return (float(ratio[j]), -float(saved[j]), block, -j, int(version[block]))

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

    # Trim: while over cap, apply the cheapest loss-per-byte downgrade.  Each
    # applied move strictly decreases cur, so the loop terminates.
    trim_moves = 0
    heap = []
    for b in range(blocks):
        entry = best_downgrade(b)
        if entry is not None:
            heapq.heappush(heap, entry)
    overflow = used - cap
    while overflow > 0 and heap:
        _ratio, neg_saved, b, neg_j, ver = heapq.heappop(heap)
        if ver != int(version[b]) or -neg_j >= int(cur[b]):
            continue
        cur[b] = -neg_j
        version[b] += 1
        trim_moves += 1
        overflow -= int(-neg_saved)
        if overflow > 0:
            entry = best_downgrade(b)
            if entry is not None:
                heapq.heappush(heap, entry)
    used, _ = gather(cur)
    if used > cap:
        raise RuntimeError(
            f"trim repair left {used} bytes over cap {cap}; internal solver error"
        )

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
            "trim_downgrade_moves": int(trim_moves),
            "fill_upgrade_moves": int(fill_moves),
            "zero_byte_delta_alternatives_dropped": dropped,
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


# ---------------------------------------------------------------------------
# Self-check: algorithm correctness cases with an exhaustive small oracle.
# Not a quantization-quality result.
# ---------------------------------------------------------------------------


def _case_shrink(report: list) -> None:
    deltas = np.array([[-1.0, 0.0], [-3.0, 0.0], [0.0, 0.0]])
    out = shrink_group_prices(deltas, [0, 0, 1], shrinkage=0.25)
    expect = np.array([[-1.75, 0.0], [-2.25, 0.0], [0.0, 0.0]])
    assert np.array_equal(out, expect), f"shrink values {out}"
    assert out.dtype == np.float64
    assert np.array_equal(
        shrink_group_prices(deltas, [0, 0, 1], shrinkage=0.0),
        np.array([[-2.0, 0.0], [-2.0, 0.0], [0.0, 0.0]]),
    ), "shrinkage=0 must give pure group means"
    assert np.array_equal(
        shrink_group_prices(deltas, [0, 0, 1], shrinkage=1.0), deltas
    ), "shrinkage=1 must be the identity"
    for bad in (-0.01, 1.01, float("nan")):
        try:
            shrink_group_prices(deltas, [0, 0, 1], shrinkage=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"shrinkage={bad} accepted")
    try:
        shrink_group_prices(np.array([[np.nan, 0.0]]), [0], 0.25)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite delta accepted")
    try:
        shrink_group_prices(deltas, np.array([0.0, 0.0, 1.0]), 0.25)
    except ValueError:
        pass
    else:
        raise AssertionError("float groups accepted")
    report.append({"case": "shrink_group_prices", "ok": True})


def _case_boundary(report: list) -> None:
    out = allocate_body_budget([[0.0]], [[3]], 3, 0)
    assert out["selection"].tolist() == [0] and out["used_body_bytes"] == 3
    assert out["unused_body_bytes"] == 0
    try:
        allocate_body_budget([[0.0]], [[3]], 2, 0)
    except ValueError as exc:
        assert "exceeds" in str(exc), str(exc)
    else:
        raise AssertionError("cap refusal missing")
    empty = allocate_body_budget(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), 5, 0)
    assert empty["selection"].shape == (0,) and empty["used_body_bytes"] == 0
    assert empty["unused_body_bytes"] == 5
    report.append({"case": "boundary and cap refusal", "ok": True})


def _case_zero_byte_delta(report: list) -> None:
    # Same bytes, different losses: the cheapest must win and duplicates drop.
    prices = np.array([[0.0, -0.2, -0.5]])
    out = allocate_body_budget(prices, [[10, 10, 10]], 10, 0)
    assert out["selection"].tolist() == [2], out["selection"]
    assert out["predicted_loss"] == -0.5
    assert out["algorithm"]["zero_byte_delta_alternatives_dropped"] == 2
    report.append({"case": "zero-byte-delta improvement", "ok": True})


def _case_deterministic_ties(report: list) -> None:
    # Exact duplicate (bytes, price): canonical keeps the lowest option index.
    prices = np.array([[0.0, -1.0, -1.0], [0.0, -1.0, -1.0]])
    out = allocate_body_budget(prices, [[5, 5, 5], [7, 7, 7]], 12, 0)
    assert out["selection"].tolist() == [1, 1], out["selection"]
    again = allocate_body_budget(prices, [[5, 5, 5], [7, 7, 7]], 12, 0)
    assert json.dumps(out, sort_keys=True, default=float) == json.dumps(
        again, sort_keys=True, default=float
    ), "tie order not deterministic"
    report.append({"case": "deterministic ties", "ok": True})


def _case_useful_allocation(report: list) -> None:
    # Beneficial upgrades bought within cap; harmful ones never; unused reported.
    prices = np.array([[0.0, -2.0], [0.0, -1.0, 0.5]])
    byte = np.array([[8, 12], [8, 10, 1]])
    out = allocate_body_budget(prices, byte, 22, 0)
    assert out["selection"].tolist() == [1, 1], out["selection"]
    assert out["used_body_bytes"] == 22 and out["unused_body_bytes"] == 0
    assert abs(out["predicted_loss"] - (-3.0)) < 1e-12
    # One byte short: bisection lands just above the shared crossover ratio
    # 0.5, then the fill tie (equal ratio, larger byte gain first) buys block
    # 0's upgrade and block 1's no longer fits.
    out = allocate_body_budget(prices, byte, 21, 0)
    assert out["selection"].tolist() == [1, 0], out["selection"]
    assert out["used_body_bytes"] == 20 and out["unused_body_bytes"] == 1
    assert abs(out["predicted_loss"] - (-2.0)) < 1e-12
    # Cap that fits nothing but baselines plus a useless remainder: the harmful
    # 1-byte option is never bought to pad.
    out = allocate_body_budget(prices, byte, 17, 0)
    assert out["selection"].tolist() == [0, 0], out["selection"]
    assert out["used_body_bytes"] == 16 and out["unused_body_bytes"] == 1
    assert out["predicted_loss"] == 0.0
    assert out["limits"]["baseline_plan_feasible"] is True
    report.append({"case": "useful allocation", "ok": True})


def _case_forced_below_baseline(report: list) -> None:
    prices = np.array([[0.0, 0.5]])
    byte = np.array([[20, 5]])
    out = allocate_body_budget(prices, byte, 10, 0)
    assert out["selection"].tolist() == [1]
    assert out["used_body_bytes"] == 5 and out["unused_body_bytes"] == 5
    assert out["predicted_loss"] == 0.5  # honest positive loss, no fake fallback
    assert out["limits"]["baseline_plan_feasible"] is False
    assert out["limits"]["mixed_plan_worse_than_baseline"] is False
    report.append({"case": "forced below baseline", "ok": True})


def _case_nonmonotone(report: list) -> None:
    # Chain 0 -> -0.5 -> -2.0 -> -2.5 over bytes 10..13 has slopes 0.5, 1.5,
    # 0.5: bytes 12 sits in a non-convex pocket (inside the hull chord), so
    # only a narrow lambda window selects it and byte-11 never does.
    prices = np.array([[0.0, -0.5, -2.0, -2.5]])
    byte = np.array([[10, 11, 12, 13]])
    out = allocate_body_budget(prices, byte, 13, 0)
    assert out["selection"].tolist() == [3] and out["predicted_loss"] == -2.5
    out = allocate_body_budget(prices, byte, 12, 0)
    assert out["selection"].tolist() == [2] and out["predicted_loss"] == -2.0
    assert out["used_body_bytes"] == 12 and out["unused_body_bytes"] == 0
    # Cap 11 sits inside the pocket: bytes(lambda) jumps 12 -> 10, the 11-byte
    # rung is never on the lambda envelope, so the plan honestly falls back to
    # the baseline rung and reports the one byte it leaves unused.
    out = allocate_body_budget(prices, byte, 11, 0)
    assert out["selection"].tolist() == [0] and out["predicted_loss"] == 0.0
    assert out["used_body_bytes"] == 10 and out["unused_body_bytes"] == 1
    report.append({"case": "nonmonotone price curve", "ok": True})


def _case_exhaustive_oracle(report: list) -> None:
    rng = np.random.default_rng(20261006)
    gaps, violations, feasible = [], 0, 0
    for _ in range(64):
        n_blocks = int(rng.integers(1, 6))
        n_points = int(rng.integers(2, 5))
        byte = rng.integers(0, 13, size=(n_blocks, n_points)).astype(np.int64)
        prices = rng.integers(-4, 5, size=(n_blocks, n_points)).astype(np.float64)
        baseline = int(rng.integers(0, n_points))
        prices[:, baseline] = 0.0
        if n_points > 2 and rng.random() < 0.5:  # force zero-byte-delta twins
            prices[:, 1] = prices[:, 0]
        min_total = int(byte.min(axis=1).sum())
        base_total = int(byte[:, baseline].sum())
        cap = int(min_total + rng.integers(0, max(1, base_total + 2)))
        out = allocate_body_budget(prices, byte, cap, baseline)
        sel = out["selection"]
        assert sel.dtype == np.int64 and sel.shape == (n_blocks,)
        assert sel.min() >= 0 and sel.max() < n_points
        used = sum(int(byte[i, sel[i]]) for i in range(n_blocks))
        loss = sum(float(prices[i, sel[i]]) for i in range(n_blocks))
        assert used == out["used_body_bytes"], "used bytes bookkeeping mismatch"
        assert used <= cap, "selection violates the byte cap"
        assert out["unused_body_bytes"] == cap - used
        assert abs(loss - out["predicted_loss"]) < 1e-9, "loss bookkeeping mismatch"
        best = None
        for combo in itertools.product(range(n_points), repeat=n_blocks):
            combo_used = sum(int(byte[i, c]) for i, c in enumerate(combo))
            if combo_used <= cap:
                combo_loss = sum(float(prices[i, c]) for i, c in enumerate(combo))
                best = combo_loss if best is None else min(best, combo_loss)
        assert best is not None
        gap = float(out["predicted_loss"] - best)
        assert gap >= -1e-9, f"beat the exhaustive oracle by {gap}"
        gaps.append(gap)
        if base_total <= cap:
            feasible += 1
            if out["predicted_loss"] > 1e-9:
                violations += 1
    report.append(
        {
            "case": "exhaustive oracle (64 seeded instances)",
            "ok": True,
            "max_oracle_gap": max(gaps),
            "mean_oracle_gap": sum(gaps) / len(gaps),
            "baseline_feasible_instances": feasible,
            "greedy_worse_than_feasible_baseline": violations,
            "note": (
                "gap is informational: the greedy repair claims no global "
                "optimality, only feasibility and honest bookkeeping"
            ),
        }
    )


def _self_check() -> list:
    report: list = []
    _case_shrink(report)
    _case_boundary(report)
    _case_zero_byte_delta(report)
    _case_deterministic_ties(report)
    _case_useful_allocation(report)
    _case_forced_below_baseline(report)
    _case_nonmonotone(report)
    _case_exhaustive_oracle(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run the algorithm-correctness self-check and print one JSON report",
    )
    args = parser.parse_args()
    if not args.self_check:
        parser.error("nothing to do: pass --self-check")
    report = _self_check()
    ok = all(case.get("ok") for case in report)
    print(json.dumps({"status": "ok" if ok else "failed", "cases": report}, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
