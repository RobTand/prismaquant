"""Offline, surrogate-only replay of an already production-qualified menu.

This module does not admit formats, qualify W2, or validate written wires.
The explicit-input driver is responsible for production admission and provenance.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
import math

from .allocator_candidates import (
    aggregate_packed_serving_groups, aggregate_fused_siblings,
    expand_fused_sibling_assignment, expand_packed_group_assignment,
)
from .allocator_solver import solve_allocation
from .footprint import assignment_artifact_bytes
from .priced_cell import CostPricingPolicy, cost_entry_is_band_interpolated

TARGETS = ("3.5", "4.5", "6.0")


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def replay_three_arms(*, stats, candidates, costs, formats, profile, currency,
                      source_total_bytes, source_manifest, overhead_reserve_bytes,
                      refused_cells, measured_only_units, targets=TARGETS,
                      bit_precision=0.001, measured_overhead_bytes=0):
    """Replay full/meas/fullm with an integer all-in ceiling, never a bpp roundtrip.

    DP bins conservatively round *up* integer byte deltas from the exact group
    floor. Bin width follows the requested precision (at least one byte); this
    is a conservative projection, not an assertion of exact knapsack optimality.
    Expanded selections are independently accounted before returning any pick.
    """
    if tuple(str(t) for t in targets) != TARGETS:
        raise ValueError("immutable nominal targets are 3.5, 4.5 and 6.0")
    if not currency or not math.isfinite(bit_precision) or bit_precision <= 0:
        raise ValueError("declared currency and positive finite precision required")
    for value in (source_total_bytes, overhead_reserve_bytes, measured_overhead_bytes):
        if type(value) is not int or value < 0:
            raise ValueError("accounting inputs must be nonnegative integer bytes")
    if source_manifest is None:
        raise ValueError("source-bound byte manifest required")
    if not candidates or set(candidates) != set(stats):
        raise ValueError("mutable stats and candidate scope must match")
    if set(measured_only_units) - set(stats):
        raise ValueError("measured-only roster contains unknown units")
    nparams = sum(int(s["n_params"]) for s in stats.values())
    if nparams <= 0:
        raise ValueError("positive frozen mutable parameter denominator required")
    reference = {}
    for name, cs in candidates.items():
        reference[name] = {c.fmt: c for c in cs}
        if len(reference[name]) != len(cs):
            raise ValueError(f"duplicate candidate format: {name}")
        for c in cs:
            if not math.isfinite(c.predicted_dloss):
                raise ValueError("nonfinite reference price")
            row_currency = costs[name][c.fmt].get("cost_currency", currency)
            if row_currency != currency:
                raise ValueError("mixed reference currencies")

    def interpolated(name, c):
        return (c.activation_pricing == "interpolated_output_mse" or
                cost_entry_is_band_interpolated(costs[name][c.fmt]))

    menus = {}
    for arm in ("full", "meas"):
        menus[arm] = {
            n: sorted((copy.deepcopy(c) for c in cs
                       if (n, c.fmt) not in refused_cells
                       and not (interpolated(n, c) and
                                (arm == "meas" or n in measured_only_units))),
                      key=lambda c: c.fmt)
            for n, cs in sorted(candidates.items())}
        if any(not cs for cs in menus[arm].values()):
            raise ValueError(f"empty eligible {arm} menu")
    count = sum(interpolated(n, c) for n, cs in menus["full"].items() for c in cs)
    if not count:
        raise ValueError("no eligible interpolated candidates: nondegenerate replay required")
    overhead = measured_overhead_bytes + overhead_reserve_bytes

    def account(assignment, *, complete=True):
        if complete and set(assignment) != set(stats):
            raise ValueError("group expansion did not cover the frozen leaf scope")
        result = assignment_artifact_bytes(
            assignment, stats, source_total_bytes=source_total_bytes,
            source_manifest=source_manifest)
        if result["missing_stats_names"]:
            raise ValueError("missing byte-accounting stats")
        return result["artifact_payload_bytes"] + overhead, result

    def score(assignment):
        return sum(reference[n][fmt].predicted_dloss for n, fmt in sorted(assignment.items()))

    prepared = {}
    serving_groups = {"packed": [], "fused": []}
    for arm, menu in menus.items():
        gs, gc, gm = copy.deepcopy(stats), copy.deepcopy(costs), copy.deepcopy(menu)
        policy = CostPricingPolicy(ucb_z=0.0, fisher_output_mse=False)
        gs, gc, gm = aggregate_packed_serving_groups(
            gs, gc, formats, gm, profile=profile, pricing_policy=policy,
            preserve_runtime_frontier=True)
        gs, gc, gm = aggregate_fused_siblings(
            gs, gc, formats, gm, profile=profile, pricing_policy=policy,
            preserve_runtime_frontier=True)

        def expand(a):
            return expand_packed_group_assignment(
                expand_fused_sibling_assignment(a, gs), gs)

        # Price each legal atomic option using the same source-bound calculator.
        # Additivity is checked on the final complete assignment below.
        option_values = {}
        for n, cs in sorted(gm.items()):
            option_values[n] = {}
            for c in cs:
                a = expand({n: c.fmt})
                option_values[n][c.fmt] = (account(a, complete=False)[0], score(a))
        floor_pick = {n: min(cs, key=lambda c: (option_values[n][c.fmt][0], c.fmt)).fmt
                      for n, cs in gm.items()}
        floor_bytes, _ = account(expand(floor_pick))
        # Partial accounting retains every other source span. Those constants
        # cancel in each atomic option's delta; no full-model walk per option.
        # Reference scores likewise come only from the expanded immutable leaves.
        options = {}
        for n, cs in sorted(gm.items()):
            floor_size, floor_loss = option_values[n][floor_pick[n]]
            options[n] = [(c, option_values[n][c.fmt][0] - floor_size,
                           option_values[n][c.fmt][1] - floor_loss)
                          for c in sorted(cs, key=lambda c: c.fmt)]
        # Admission alone is not availability: atomic grouping can remove a
        # leaf rung. Report/hash only options that actually reach this DP.
        surviving = set()
        for n, cs in gm.items():
            for c in cs:
                surviving.update(expand({n: c.fmt}).items())
        menus[arm] = {n: [c for c in cs if (n, c.fmt) in surviving]
                      for n, cs in menu.items()}
        if arm == "full":
            count = sum(interpolated(n, reference[n][f]) for n, f in surviving)
            if not count:
                raise ValueError("no eligible interpolated candidates survive production grouping")
            for entry in gs.values():
                for kind, key in (("packed", "_packed_group_members"),
                                  ("fused", "_fused_siblings")):
                    if entry.get(key):
                        serving_groups[kind].append(sorted(entry[key]))
        prepared[arm] = (gs, options, floor_pick, floor_bytes)

    def select(arm, budget):
        gs, options, floor_pick, floor_bytes = prepared[arm]
        menu = menus[arm]
        out = {"budget_bytes": budget, "floor_bytes": floor_bytes,
               "floor_gap_bytes": max(0, floor_bytes - budget),
               "candidate_formats": {n: [c.fmt for c in cs] for n, cs in menu.items()},
               "candidate_sha256": _sha({n: [{"fmt": c.fmt,
                   "reference_dloss": c.predicted_dloss,
                   "memory_bytes": c.memory_bytes,
                   "serialized_identity": c.serialized_identity,
                   "activation_pricing": c.activation_pricing,
                   "reference_row": costs[n][c.fmt]} for c in cs]
                   for n, cs in menu.items()})}
        if floor_bytes > budget:
            return dict(out, status="infeasible", assignment=None)
        quantum = max(1, math.ceil(bit_precision * nparams / 8))
        capacity = (budget - floor_bytes) // quantum
        dp_menu = {}
        for n, opts in options.items():
            dp_menu[n] = [replace(c,
                bits_per_param=math.ceil(delta / quantum) * quantum * 8 / gs[n]["n_params"],
                predicted_dloss=loss) for c, delta, loss in opts
                if math.ceil(delta / quantum) <= capacity]
        precision = quantum * 8 / nparams
        # solve_allocation permits one extra bin. Subtract it explicitly;
        # zero capacity has only zero-delta options in the menu already.
        result = solve_allocation(gs, dp_menu,
                                  max(0, capacity - 1) * precision, precision)
        if result is None:
            raise ValueError("production DP refused an exactly feasible menu floor")
        selected, _ = result
        assignment = expand_packed_group_assignment(
            expand_fused_sibling_assignment(selected, gs), gs)
        achieved, accounting = account(assignment)
        expected = floor_bytes + sum(next(d for c, d, _ in options[n]
                                        if c.fmt == f) for n, f in selected.items())
        if achieved != expected or achieved > budget:
            raise ValueError("independent integer-byte feasibility/additivity check failed")
        return dict(out, status="ok", assignment=dict(sorted(assignment.items())),
                    assignment_sha256=_sha(assignment), achieved_bytes=achieved,
                    residual_bytes=budget-achieved, predicted_dloss=score(assignment),
                    accounting=accounting, dp_byte_quantum=quantum)

    def ratio(numerator, denominator):
        return {"numerator": numerator, "denominator": denominator,
                "currency": currency, "ratio": (numerator / denominator
                                                  if denominator != 0 else None),
                "zero_denominator": "undefined_null"}

    rows = []
    for target in TARGETS:
        budget = int(Decimal(target) * nparams / 8)
        full, meas = select("full", budget), select("meas", budget)
        fullm = (select("full", meas["achieved_bytes"]) if meas["status"] == "ok"
                 else {"status": "unavailable", "reason": "meas infeasible"})
        reach = (dict(ratio(meas["predicted_dloss"], full["predicted_dloss"]),
                      numerator_arm="meas", denominator_arm="full",
                      byte_gap=meas["achieved_bytes"] - full["achieved_bytes"],
                      equal_size=meas["achieved_bytes"] == full["achieved_bytes"],
                      interpretation="common_nominal_target_reach")
                 if full["status"] == meas["status"] == "ok" else None)
        ranking = None
        if fullm["status"] == "ok":
            gap = meas["achieved_bytes"] - fullm["achieved_bytes"]
            ranking = dict(ratio(meas["predicted_dloss"], fullm["predicted_dloss"]),
                           numerator_arm="meas", denominator_arm="fullm",
                           byte_gap=gap, equal_size=gap == 0,
                           interpretation="common_budget_surrogate_comparison")
        rows.append({"target_bpp": target, "target_bytes": budget,
                     "arms": {"full": full, "meas": meas, "fullm": fullm},
                     "reach": reach, "ranking": ranking})
    return {"status": "surrogate_only_pending_w4", "currency": currency,
            "frozen_mutable_n_params": nparams,
            "accounting_convention": "floor(Decimal(target_bpp)*N/8); all-in tensor spans + measured overhead + reserve",
            "measured_overhead_bytes": measured_overhead_bytes,
            "overhead_reserve_bytes": overhead_reserve_bytes,
            "physical_equality_validated": False,
            "eligible_interpolated_count": count,
            "serving_groups": serving_groups, "targets": rows,
            "solver": {"entry_point": "solve_allocation", "bit_precision": bit_precision,
                       "projection": "conservative ceil of exact group byte deltas"}}
