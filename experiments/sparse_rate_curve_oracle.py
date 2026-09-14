#!/usr/bin/env python3
"""Offline all-truth lower bound for sparse-rate interpolation anchors.

This oracle may run only after a complete measured curve has been published.
It uses every measured rung to find the fewest anchors that reproduce that
curve within a declared tolerance.  It is not an acquisition policy, provides
no deployable stopping rule, and cannot establish measurement savings.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.sparse_rate_adaptive import CURRENCY, TOLERANCES, validate_curve
from prismaquant.adaptive_anchored_shape import _predict_between


SCHEMA = "prismaquant.sparse_rate_curve_oracle.v1"
MODES = ("value", "log2")
MAX_ROSTER_LENGTH = 2048
TIE_BREAK = "lowest_predecessor_index"


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_inputs(rates, values, mode, max_relative_error, require_strict_decrease):
    """Validate a finite positive roster without reading any curve artifact."""
    try:
        rates, values = tuple(rates), tuple(values)
    except TypeError as exc:
        raise ValueError("rates and values must be finite sequences") from exc
    if not 2 <= len(rates) <= MAX_ROSTER_LENGTH or len(values) != len(rates):
        raise ValueError(f"rates and values need matching lengths from 2 through {MAX_ROSTER_LENGTH}")
    if (any(type(rate) is not int or rate <= 0 for rate in rates)
            or any(left >= right for left, right in zip(rates, rates[1:]))):
        raise ValueError("rates must be a strictly increasing roster of positive integers")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("values must be positive finite measurements")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if (isinstance(max_relative_error, bool) or not isinstance(max_relative_error, (int, float))
            or not math.isfinite(max_relative_error) or max_relative_error < 0):
        raise ValueError("max_relative_error must be a finite nonnegative number")
    if type(require_strict_decrease) is not bool:
        raise ValueError("require_strict_decrease must be a boolean")
    return rates, tuple(map(float, values)), float(max_relative_error), require_strict_decrease


def _edge_error(rates, values, mode, left, right):
    """Return the exact maximum interior relative error for one candidate edge."""
    if right - left <= 1:
        return 0.0
    left_rate, right_rate = rates[left], rates[right]
    fraction = ((np.asarray(rates[left + 1:right], dtype=np.float64) - left_rate)
                / (right_rate - left_rate))
    if mode == "value":
        predicted = values[left] + fraction * (values[right] - values[left])
    else:
        predicted = np.exp2(np.log2(values[left]) + fraction * (np.log2(values[right]) - np.log2(values[left])))
    error = np.abs(predicted / np.asarray(values[left + 1:right]) - 1.0)
    return float(error.max())


def minimum_anchors(
    rates, values, *, mode: str, max_relative_error: float,
    require_strict_decrease: bool = True,
):
    """Find an exact all-truth minimum-anchor path over a finite roster.

    Every measured interior rung on a legal edge has relative interpolation
    error at most the requested bound.  By default, edge endpoints must also
    strictly decrease; research callers can opt out of that shape constraint.
    Among equal-cardinality paths, the lowest predecessor index is selected at
    every dynamic-programming state.
    """
    rates, values, max_relative_error, require_strict_decrease = _validate_inputs(
        rates, values, mode, max_relative_error, require_strict_decrease,
    )
    count = len(rates)
    predecessor = [-1] * count
    anchor_counts = [math.inf] * count
    edge_errors = [None] * count
    anchor_counts[0] = 1
    eligible_edges = 0

    for right in range(1, count):
        for left in range(right):
            if (require_strict_decrease and not values[left] > values[right]) or not math.isfinite(anchor_counts[left]):
                continue
            actual_max = _edge_error(rates, values, mode, left, right)
            if actual_max > max_relative_error:
                continue
            eligible_edges += 1
            candidate_count = anchor_counts[left] + 1
            if (candidate_count < anchor_counts[right]
                    or (candidate_count == anchor_counts[right]
                        and (predecessor[right] == -1 or left < predecessor[right]))):
                anchor_counts[right] = candidate_count
                predecessor[right] = left
                edge_errors[right] = actual_max

    if not math.isfinite(anchor_counts[-1]):
        reason = ("no feasible strictly decreasing anchor path" if require_strict_decrease
                  else "no feasible anchor path")
        return {"status": "refused_shape", "reason": reason,
                "mode": mode, "max_relative_error": max_relative_error,
                "require_strict_decrease": require_strict_decrease,
                "optimal_anchor_count": None, "anchor_indices": [], "anchor_rates": [],
                "actual_max_relative_error": None, "eligible_edge_count": eligible_edges,
                "tie_break": TIE_BREAK}

    path = []
    current = count - 1
    while current != -1:
        path.append(current)
        current = predecessor[current]
    path.reverse()
    selected_errors = [edge_errors[index] for index in path[1:]]
    return {"status": "optimal", "mode": mode, "max_relative_error": max_relative_error,
            "require_strict_decrease": require_strict_decrease,
            "optimal_anchor_count": int(anchor_counts[-1]), "anchor_indices": path,
            "anchor_rates": [rates[index] for index in path],
            "rate_choices": [rates[index] for index in path],
            "actual_max_relative_error": max(selected_errors, default=0.0),
            "edge_actual_max_relative_errors": selected_errors,
            "eligible_edge_count": eligible_edges, "tie_break": TIE_BREAK}


def evaluate_curve(
    curve, *, mode: str, tolerance: float, require_strict_decrease: bool = True,
):
    """Validate a published complete curve, then run the all-truth oracle."""
    rates, values = validate_curve(curve)
    return minimum_anchors(
        rates, values, mode=mode, max_relative_error=tolerance,
        require_strict_decrease=require_strict_decrease,
    )


def write_json(path: Path, value) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", required=True, help="published complete measured curve JSON")
    parser.add_argument("--out", required=True, help="new directory for the oracle report")
    parser.add_argument("--profile-out", help="optional cProfile output path; not a timing claim")
    parser.add_argument("--allow-nonmonotone", dest="require_strict_decrease", action="store_false",
                        help="research-only: allow flat or increasing anchor edges")
    args = parser.parse_args()

    curve_path, out = Path(args.curve), Path(args.out)
    curve = json.loads(curve_path.read_text())
    # Validation is intentionally before any truth values enter the oracle.
    validate_curve(curve)
    out.mkdir(parents=True, exist_ok=False)
    profiler = cProfile.Profile() if args.profile_out else None
    invoke = (lambda: {
        f"{mode}:{tolerance:g}": evaluate_curve(
            curve, mode=mode, tolerance=tolerance,
            require_strict_decrease=args.require_strict_decrease,
        )
        for mode in MODES for tolerance in TOLERANCES
    })
    results = profiler.runcall(invoke) if profiler else invoke()
    profile_sha256 = None
    if profiler:
        profile_path = Path(args.profile_out)
        profiler.dump_stats(profile_path)
        profile_sha256 = digest(profile_path)
    write_json(out / "report.json", {
        "schema": SCHEMA, "research_only": True,
        "offline_all_truth_lower_bound": True,
        "not_a_deployable_adaptive_policy": True,
        "does_not_establish_measurement_savings": True,
        "require_strict_decrease": args.require_strict_decrease,
        "input_curve_sha256": digest(curve_path), "script_sha256": digest(__file__),
        "profile_sha256": profile_sha256,
        "currency": CURRENCY, "curve_id": curve["curve_id"], "qname": curve["qname"],
        "family": curve["family"], "activation_contract": curve["activation_contract"],
        "source_identity": curve["source_identity"], "calibration_identity": curve["calibration_identity"],
        "recipe_identity": curve["recipe_identity"], "family_activation_aggregation": "none",
        "results": results,
    })
    print(json.dumps({"out": str(out), "result_count": len(results), "statuses":
                      {key: value["status"] for key, value in results.items()}}), flush=True)


if __name__ == "__main__":
    main()
