#!/usr/bin/env python3
"""bakeoff.py — decide whether a PrismQuant change is worth keeping.

This script compares a candidate run against:
  - the additive baseline calibration
  - an optional interaction-refined recipe
  - an optional oracle local search on the same tiny problem

The goal is not just to print metrics, but to answer:
  "did the new method buy enough quality to justify keeping it?"
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass
class Point:
    label: str
    bits: float
    kl: float


def _load_calibration_point(path: str, selector: str) -> Point:
    with open(path) as f:
        data = json.load(f)
    if selector == "baseline":
        row = min(data["results"], key=lambda r: r["achieved_bits"])
    elif selector == "high":
        row = max(data["results"], key=lambda r: r["achieved_bits"])
    elif selector == "knee":
        rows = sorted(data["results"], key=lambda r: r["achieved_bits"])
        row = rows[1] if len(rows) >= 2 else rows[0]
    else:
        raise ValueError(selector)
    return Point(selector, float(row["achieved_bits"]), float(row["actual_last_token_kl"]))


def _load_refined_point(path: str, calibrated_kl: float) -> Point:
    with open(path) as f:
        data = json.load(f)
    estimate = data.get("calibrated_last_token_kl_estimate")
    if estimate is None:
        estimate = calibrated_kl + data["refined_delta_kl_estimate"]
    return Point(
        "refined",
        float(data["bits_per_param"]),
        float(estimate),
    )


def _load_refined_actual_kl(path: str) -> float | None:
    """Pull the measured KL of the refined recipe out of a calibration
    JSON, if one was produced after re-running calibrate_allocator.py
    against the refined assignment.

    Accepts either:
      - a calibrate_allocator.py output containing `results: [...]` with
        a single entry tagged for the refined recipe, or
      - a small JSON `{ "actual_last_token_kl": <float> }` written by
        an ad-hoc measurement script.

    Returns None if no usable value is found.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "actual_last_token_kl" in data:
        try:
            return float(data["actual_last_token_kl"])
        except (TypeError, ValueError):
            return None
    results = data.get("results") if isinstance(data, dict) else None
    if isinstance(results, list) and results:
        try:
            return float(results[0]["actual_last_token_kl"])
        except (TypeError, ValueError, KeyError):
            return None
    return None


def _load_oracle_best(path: str) -> Point:
    with open(path) as f:
        data = json.load(f)
    best = data["best"]
    return Point("oracle", float(best["bits_per_param"]), float(best["actual_last_token_kl"]))


def _summarize(candidate: Point, baseline: Point, oracle: Point | None,
               candidate_actual_kl: float | None = None):
    out = {
        "candidate": candidate.__dict__,
        "baseline": baseline.__dict__,
        "delta_kl_vs_baseline": candidate.kl - baseline.kl,
        "delta_bits_vs_baseline": candidate.bits - baseline.bits,
    }
    if oracle is not None:
        out["oracle"] = oracle.__dict__
        out["oracle_gap_signed"] = candidate.kl - oracle.kl
        out["oracle_gap_abs"] = abs(candidate.kl - oracle.kl)
        out["oracle_gap_rel"] = abs(candidate.kl - oracle.kl) / max(abs(oracle.kl), 1e-12)
    if candidate_actual_kl is not None:
        # The candidate.kl coming from a refined recipe is the unary +
        # pairwise interaction-model PREDICTION. If we also have the
        # actually measured KL of that recipe, the residual quantifies
        # how much the sparse-pairwise model failed to capture (triples
        # and out-of-knee interactions).
        residual = float(candidate_actual_kl) - float(candidate.kl)
        out["candidate_actual_kl"] = float(candidate_actual_kl)
        out["interaction_model_residual_kl"] = residual
        out["interaction_model_residual_kl_rel"] = (
            residual / max(abs(candidate.kl - baseline.kl), 1e-12)
        )
    return out


def _decision(summary: dict, max_kl_regression: float, min_kl_gain: float, max_oracle_gap: float | None):
    delta = summary["delta_kl_vs_baseline"]
    if delta > max_kl_regression:
        return "reject", f"KL regressed by {delta:.4e} (> {max_kl_regression:.4e})"
    if delta < -min_kl_gain:
        if max_oracle_gap is not None and "oracle_gap_abs" in summary and summary["oracle_gap_abs"] > max_oracle_gap:
            return "investigate", (
                f"improved vs baseline ({delta:.4e}) but still {summary['oracle_gap_abs']:.4e} "
                f"away from oracle (> {max_oracle_gap:.4e})"
            )
        return "keep", f"improved KL by {-delta:.4e}"
    if max_oracle_gap is not None and "oracle_gap_abs" in summary and summary["oracle_gap_abs"] <= max_oracle_gap:
        return "keep", f"near oracle gap ({summary['oracle_gap_abs']:.4e})"
    return "investigate", "change is neutral; needs broader justification"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--candidate", choices=["baseline", "knee", "high", "refined"], default="knee")
    ap.add_argument("--refined", help="quadratic_refine_allocator output when --candidate refined")
    ap.add_argument("--oracle", help="oracle_search output")
    ap.add_argument("--refined-actual-kl",
                    help="Optional path to a JSON containing the actually "
                         "measured KL of the refined recipe (e.g. a second "
                         "calibrate_allocator.py run). Enables an "
                         "interaction-model residual diagnostic: "
                         "measured_kl - predicted_kl tells us how much the "
                         "sparse-pairwise interaction model under- or "
                         "over-estimates the truth.")
    ap.add_argument("--refined-actual-kl-value", type=float, default=None,
                    help="Same as --refined-actual-kl but pass the float "
                         "directly. Useful in scripted bake-offs.")
    ap.add_argument("--max-kl-regression", type=float, default=1e-3)
    ap.add_argument("--min-kl-gain", type=float, default=1e-3)
    ap.add_argument("--max-oracle-gap", type=float, default=5e-3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    baseline = _load_calibration_point(args.calibration, "baseline")
    knee = _load_calibration_point(args.calibration, "knee")
    if args.candidate == "refined":
        if not args.refined:
            raise SystemExit("--refined is required for candidate=refined")
        candidate = _load_refined_point(args.refined, knee.kl)
    else:
        candidate = _load_calibration_point(args.calibration, args.candidate)
    oracle = _load_oracle_best(args.oracle) if args.oracle else None

    candidate_actual_kl: float | None = None
    if args.refined_actual_kl_value is not None:
        candidate_actual_kl = float(args.refined_actual_kl_value)
    elif args.refined_actual_kl:
        candidate_actual_kl = _load_refined_actual_kl(args.refined_actual_kl)

    summary = _summarize(candidate, baseline, oracle,
                         candidate_actual_kl=candidate_actual_kl)
    decision, reason = _decision(
        summary,
        max_kl_regression=args.max_kl_regression,
        min_kl_gain=args.min_kl_gain,
        max_oracle_gap=args.max_oracle_gap if args.oracle else None,
    )
    summary["decision"] = decision
    summary["reason"] = reason

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
