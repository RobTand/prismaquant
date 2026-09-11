#!/usr/bin/env python3
"""Replay adaptive acquisition on complete measured curves, with hidden truth.

Families and activation contracts are independent. Only never-revealed points
enter the accuracy audit; observed probe residuals remain training diagnostics.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from prismaquant.adaptive_anchored_shape import AdaptiveAnchoredCurve, _predict_between
from prismaquant.anchored_shape import AnchoredShapeError

SCHEMA = "prismaquant.complete_curve_adaptive_study.v1"
CURVE_SCHEMA = "prismaquant.complete_rate_curve.v1"
MEASUREMENT_PLAN_SCHEMA = "prismaquant.complete_rate_measurement_plan.v1"
CURRENCY = "output_mse_under_route_activation_contract"
CAPS = (2, 3, 5, 9, 17, 33, 65)
TOLERANCES = (.001, .005, .01)
MODES = ("value", "log2")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def policy_caps(*, require_strict_decrease=True, max_measurements=65):
    if type(require_strict_decrease) is not bool:
        raise ValueError("require_strict_decrease must be a bool")
    if type(max_measurements) is not int or not 2 <= max_measurements <= 257:
        raise ValueError("max_measurements must be an integer in [2, 257]")
    return tuple(sorted({cap for cap in (*CAPS, 129, 257) if cap <= max_measurements}
                        | {max_measurements}))


def study_plan(*, require_strict_decrease=True, max_measurements=65):
    """Freeze this matrix before opening the complete measured curves."""
    caps = policy_caps(require_strict_decrease=require_strict_decrease,
                       max_measurements=max_measurements)
    return {"schema": SCHEMA, "currency": CURRENCY, "research_only": True,
        "caps": list(caps), "modes": list(MODES), "tolerances": list(TOLERANCES),
        "require_strict_decrease": require_strict_decrease,
        "max_measurements": max_measurements,
        "measurement_transformation": "none; raw positive finite measurements are retained",
        "checks_per_interval": [1, 2],
        "primary_policy": {"mode": "value", "relative_tolerance": .005, "checks_per_interval": 2},
        "primary_policy_reason": "value interpolation led earlier studies; two sentinels guard a midpoint-only blind spot; 0.5% leaves margin below the existing 1% p99 screen",
        "comparators": ["fixed_widest_gap_same_count", "endpoint_only_same_unrevealed"],
        "audit": "all legal measured rungs never revealed to the sampler",
        "qualification_screen": {"p99_relative": .01, "max_relative": .05},
        "families_independent": True, "family_wide_qualification": False,
        "continuous_error_bound": False, "joint_aura_qualified": False,
        "deployment_measurement_saving_qualified": False,
        "source_sha256": digest(__file__),
        "adaptive_core_sha256": digest(Path(__file__).resolve().parents[1] / "prismaquant/adaptive_anchored_shape.py"),
        "oracle_source_sha256": digest(Path(__file__).with_name("sparse_rate_curve_oracle.py"))}


def validate_curve(curve, *, min_points=3):
    """Validate a receipted measured curve.

    Complete-curve replay retains its historical three-point minimum.  A
    prospective protocol may deliberately acquire an endpoint as its own
    one-point curve, so it opts in explicitly with ``min_points=1``.
    """
    if type(min_points) is not int or min_points < 1:
        raise ValueError("min_points must be a positive integer")
    if (curve.get("schema") != CURVE_SCHEMA or curve.get("currency") != CURRENCY
            or curve.get("measurement_kind") != "measured"):
        raise ValueError("complete-curve measurement contract is unsupported")
    for key in ("curve_id", "qname", "family", "activation_contract"):
        if not isinstance(curve.get(key), str) or not curve[key]:
            raise ValueError(f"curve lacks {key}")
    for key in ("source_identity", "calibration_identity", "recipe_identity"):
        if not isinstance(curve.get(key), dict) or not curve[key]:
            raise ValueError(f"curve lacks {key}")
    source = curve["source_identity"]
    if not all(source.get(key) for key in ("model", "weight_identity", "producer")):
        raise ValueError("source identity lacks model, weight or producer binding")
    plan_ref = curve.get("measurement_plan")
    if (not isinstance(plan_ref, dict) or not isinstance(plan_ref.get("path"), str)
            or digest(plan_ref["path"]) != plan_ref.get("sha256")):
        raise ValueError("measurement plan reference does not match its bytes")
    plan = json.loads(Path(plan_ref["path"]).read_text())
    if plan.get("schema") != MEASUREMENT_PLAN_SCHEMA:
        raise ValueError("measurement plan schema is unsupported")
    for key in ("curve_id", "qname", "family", "activation_contract", "source_identity",
                "calibration_identity", "recipe_identity", "legal_rates", "audit_regions"):
        if plan.get(key) != curve.get(key):
            raise ValueError(f"curve differs from its pre-measurement plan: {key}")
    rates, values = curve.get("rates"), curve.get("values")
    if (not isinstance(rates, list) or len(rates) < min_points
            or any(type(rate) is not int for rate in rates)
            or any(left >= right for left, right in zip(rates, rates[1:]))):
        raise ValueError("curve rates must be a strictly increasing legal roster")
    if (not isinstance(values, list) or len(values) != len(rates)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value <= 0 for value in values)):
        raise ValueError("curve values must be complete positive finite measurements")
    if curve.get("legal_rates") != rates:
        raise ValueError("measured rates do not equal the declared legal roster")
    regions = curve.get("audit_regions", {})
    if (not isinstance(regions, dict) or any(not isinstance(region, list)
            or len(set(region)) != len(region) or not set(region) <= set(rates)
            for region in regions.values())):
        raise ValueError("audit regions must be declared subsets of the legal roster")
    receipts = curve.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != len(rates):
        raise ValueError("every measured rate needs a receipt")
    for rate, value, receipt in zip(rates, values, receipts):
        if (not isinstance(receipt, dict) or receipt.get("rate") != rate
                or receipt.get("value") != value or receipt.get("kind") != "measured"
                or not isinstance(receipt.get("sha256"), str)
                or len(receipt["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in receipt["sha256"])):
            raise ValueError("receipt does not bind the corresponding measurement")
        if not isinstance(receipt.get("path"), str) or digest(receipt["path"]) != receipt["sha256"]:
            raise ValueError("measurement receipt reference differs from its bytes")
        record = json.loads(Path(receipt["path"]).read_text())
        expected = {"curve_id": curve["curve_id"], "measurement_plan_sha256": plan_ref["sha256"],
                    "rate": rate, "value": value, "kind": "measured"}
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("measurement receipt bytes differ from curve/plan")
    return tuple(rates), tuple(map(float, values))


def error_metrics(predictions, truth, tolerance):
    actual = np.asarray(truth)
    if not len(actual):
        return {"count": 0, "empty_audit": True, "screen_pass": None}
    errors = np.abs(np.asarray(predictions) / actual - 1)
    result = {"count": len(actual), "mean_relative": float(errors.mean()),
        **{f"{name}_relative": float(np.quantile(errors, q)) for name, q in
           (("p50", .5), ("p95", .95), ("p99", .99), ("max", 1))},
        "above_requested_tolerance": int((errors > tolerance).sum()),
        "fraction_above_requested_tolerance": float((errors > tolerance).mean())}
    result["screen_pass"] = result["p99_relative"] <= .01 and result["max_relative"] <= .05
    return result


def predict_measured(measured, rate, mode):
    if rate in measured:
        return measured[rate]
    ordered = sorted(measured)
    upper = next(i for i, coordinate in enumerate(ordered) if coordinate > rate)
    left, right = ordered[upper - 1], ordered[upper]
    return _predict_between(mode, left, measured[left], right, measured[right], rate)


def fixed_schedule(rates):
    """Geometry-only widest-gap acquisition; no measured value enters."""
    measured = {rates[0], rates[-1]}
    order = [rates[0], rates[-1]]
    while len(measured) < len(rates):
        intervals = [(left, right) for left, right in zip(sorted(measured), sorted(measured)[1:])
                     if any(left < rate < right for rate in rates)]
        left, right = min(intervals, key=lambda pair: (-(pair[1] - pair[0]), pair[0]))
        interior = [rate for rate in rates if left < rate < right]
        midpoint = min(interior, key=lambda rate: (abs(2 * rate - left - right), rate))
        measured.add(midpoint)
        order.append(midpoint)
    return tuple(order)


def state_audit(state, truth, tolerance, fixed_order, regions):
    """Evaluator-only; these results never flow back into the sampler."""
    hidden = [rate for rate in truth if rate not in state.measurements]
    endpoints = {rate: truth[rate] for rate in (state.candidate_coordinates[0], state.candidate_coordinates[-1])}
    fixed = {rate: truth[rate] for rate in fixed_order[:state.measurement_count]}
    fixed_hidden = [rate for rate in truth if rate not in fixed]
    common = [rate for rate in hidden if rate not in fixed]
    def score(predict, selected):
        return error_metrics([predict(rate) for rate in selected], [truth[rate] for rate in selected], tolerance)
    return {"measurement_count": state.measurement_count, "measured_rates": sorted(state.measurements),
        "never_revealed_metrics": score(state.predict, hidden),
        "endpoint_same_unrevealed_metrics": score(lambda r: predict_measured(endpoints, r, state.mode), hidden),
        "fixed_same_count_metrics": score(lambda r: predict_measured(fixed, r, state.mode), fixed_hidden),
        "common_unrevealed_adaptive": score(state.predict, common),
        "common_unrevealed_fixed": score(lambda r: predict_measured(fixed, r, state.mode), common),
        "region_audits": {label: {"legal_candidate_count": len(region),
            "metrics": score(state.predict, [r for r in hidden if r in region])}
            for label, region in regions.items()},
        "fixed_measured_rates": sorted(fixed),
        "accepted_intervals": [asdict(interval) for interval in state.accepted_intervals],
        "unverified_intervals": [asdict(interval) for interval in state.unverified_intervals],
        "all_candidates_measured": state.all_candidates_measured}


def evaluate_curve(curve, *, mode, tolerance, checks_per_interval,
                   require_strict_decrease=True, max_measurements=65):
    caps = policy_caps(require_strict_decrease=require_strict_decrease,
                       max_measurements=max_measurements)
    rates, values = validate_curve(curve)
    truth, order = dict(zip(rates, values)), fixed_schedule(rates)
    regions = curve.get("audit_regions", {})
    try:
        state = AdaptiveAnchoredCurve.start(rates, {rates[0]: values[0], rates[-1]: values[-1]},
            mode=mode, relative_tolerance=tolerance, max_measurements=min(max_measurements, len(rates)),
            checks_per_interval=checks_per_interval, require_strict_decrease=require_strict_decrease)
    except AnchoredShapeError as exc:
        return {"status": "invalid_endpoints", "reason": str(exc), "snapshots": [],
                "actual_measurements": 2, "required_fallback": "measure every legal rung; interpolation refused"}
    snapshots, probes = [], []
    while True:
        if state.measurement_count in caps:
            snapshots.append(state_audit(state, truth, tolerance, order, regions))
        pending_state, probe = state.request_next()
        if probe is None:
            if not snapshots or snapshots[-1]["measurement_count"] != state.measurement_count:
                snapshots.append(state_audit(state, truth, tolerance, order, regions))
            by_count = {snapshot["measurement_count"]: i for i, snapshot in enumerate(snapshots)}
            cap_results = {str(cap): {"snapshot_index": by_count.get(cap, len(snapshots) - 1),
                "stopped_before_cap": state.measurement_count < cap}
                for cap in caps if cap <= len(rates)}
            return {"status": "budget_exhausted" if state.unverified_intervals else "empirically_checked",
                "mode": mode, "tolerance": tolerance, "checks_per_interval": checks_per_interval,
                "snapshots": snapshots, "cap_results": cap_results,
                "excluded_caps_above_roster_length": [cap for cap in caps if cap > len(rates)],
                "require_strict_decrease": require_strict_decrease,
                "max_measurements": max_measurements,
                "effective_max_measurements": min(max_measurements, len(rates)),
                "probes": probes, "actual_measurements": state.measurement_count,
                "required_fallback": "unverified intervals require further measurements" if state.unverified_intervals else None}
        # Commit the acquisition decision before revealing its measured truth.
        actual = truth[probe.coordinate]
        probes.append({"rate": probe.coordinate, "prediction_before_reveal": probe.prediction,
                       "measured_value": actual, "prequential_relative_error": abs(probe.prediction / actual - 1)})
        try:
            state = pending_state.record_measurement(probe.coordinate, actual)
        except AnchoredShapeError as exc:
            return {"status": "invalid_measurement", "reason": str(exc), "mode": mode,
                "tolerance": tolerance, "checks_per_interval": checks_per_interval,
                "snapshots": snapshots, "probes": probes, "actual_measurements": state.measurement_count + 1,
                "required_fallback": "measure every legal rung; interpolation refused"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-plan")
    parser.add_argument("--plan")
    parser.add_argument("--curve")
    parser.add_argument("--out")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--tolerance", type=float, choices=TOLERANCES)
    parser.add_argument("--checks-per-interval", type=int, choices=(1, 2))
    parser.add_argument("--allow-nonmonotone", action="store_true",
                        help="retain raw positive measurements even when they rise or tie")
    parser.add_argument("--max-measurements", type=int, default=65)
    args = parser.parse_args()
    policy = {"require_strict_decrease": not args.allow_nonmonotone,
              "max_measurements": args.max_measurements}
    if args.freeze_plan:
        write_json(args.freeze_plan, study_plan(**policy))
        return
    if any(v is None for v in (args.plan, args.curve, args.out, args.mode, args.tolerance, args.checks_per_interval)):
        parser.error("replay needs plan, curve, out, mode, tolerance and checks-per-interval")
    if json.loads(Path(args.plan).read_text()) != study_plan(**policy):
        raise ValueError("frozen study plan differs from this implementation")
    curve = json.loads(Path(args.curve).read_text())
    validate_curve(curve)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started, profiler = time.time(), cProfile.Profile()
    result = profiler.runcall(evaluate_curve, curve, mode=args.mode,
        tolerance=args.tolerance, checks_per_interval=args.checks_per_interval, **policy)
    profiler.dump_stats(out / "profile.prof")
    write_json(out / "report.json", {"schema": SCHEMA, "research_only": True,
        "plan_sha256": digest(args.plan), "curve_sha256": digest(args.curve),
        "profile_sha256": digest(out / "profile.prof"), "qname": curve["qname"],
        "family": curve["family"], "currency": CURRENCY, "curve_id": curve["curve_id"],
        "activation_contract": curve["activation_contract"], "source_identity": curve["source_identity"],
        "measurement_plan": curve["measurement_plan"],
        "policy": policy,
        "calibration_identity": curve["calibration_identity"], "recipe_identity": curve["recipe_identity"],
        "started_unix": started, "finished_unix": time.time(), "result": result})
    print(json.dumps({"out": str(out), "status": result["status"],
                      "measurements": result["actual_measurements"]}), flush=True)


if __name__ == "__main__":
    main()
