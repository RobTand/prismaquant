#!/usr/bin/env python3
"""quadratic_refine_allocator.py — sparse interaction-aware knee refinement."""
from __future__ import annotations

import argparse
import json

from .interaction_refine import RefinementUnit, UnitOption, expand_unit_assignment, sparse_local_refine


def _load_units(payload: dict):
    units = []
    allowed = {}
    for row in payload["selected_units"]:
        options = tuple(
            UnitOption(
                fmt=opt["fmt"],
                bits_total=float(opt["bits_total"]),
                predicted_dloss=float(opt["predicted_dloss"]),
            )
            for opt in row["options"]
        )
        base_member_fmts = tuple(tuple(x) for x in row.get("base_member_fmts", []))
        if not base_member_fmts:
            base_member_fmts = tuple((member, row["base_fmt"]) for member in row["members"])
        unit = RefinementUnit(
            key=row["key"],
            members=tuple(row["members"]),
            base_fmt=row["base_fmt"],
            base_member_fmts=base_member_fmts,
            options=options,
        )
        units.append(unit)
        allowed[unit.key] = tuple(opt for opt, raw in zip(options, row["options"]) if raw.get("allowed", True))
    return units, allowed


def _fit_piecewise_monotone(points: list[tuple[float, float]]):
    pts = sorted((float(x), float(y)) for x, y in points)
    dedup = []
    for x, y in pts:
        if dedup and abs(dedup[-1][0] - x) < 1e-12:
            dedup[-1] = (x, min(dedup[-1][1], y))
        else:
            dedup.append((x, y))
    if not dedup:
        return None
    xs = [x for x, _ in dedup]
    ys = []
    running = float("-inf")
    for _x, y in dedup:
        running = max(running, y)
        ys.append(running)

    def _interp(x: float) -> float:
        if len(xs) == 1:
            return ys[0]
        if x <= xs[0]:
            x0, y0 = xs[0], ys[0]
            x1, y1 = xs[1], ys[1]
        elif x >= xs[-1]:
            x0, y0 = xs[-2], ys[-2]
            x1, y1 = xs[-1], ys[-1]
        else:
            for i in range(1, len(xs)):
                if x <= xs[i]:
                    x0, y0 = xs[i - 1], ys[i - 1]
                    x1, y1 = xs[i], ys[i]
                    break
        if abs(x1 - x0) < 1e-12:
            return y0
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    return _interp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--calibration", help="calibrate_allocator output for KL mapping")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-passes", type=int, default=8)
    args = ap.parse_args()

    with open(args.interactions) as f:
        payload = json.load(f)

    units, allowed = _load_units(payload)
    pairwise = {
        (
            row["left_unit"],
            row["left_fmt"],
            row["right_unit"],
            row["right_fmt"],
        ): float(row["interaction_delta"])
        for row in payload["pairwise"]
    }
    unary = {
        unit_key: {fmt: float(delta) for fmt, delta in fmts.items()}
        for unit_key, fmts in payload["unary"].items()
    }

    result = sparse_local_refine(
        units=units,
        unary=unary,
        pairwise=pairwise,
        target_total_bits=max(float(payload["target_total_bits"]), float(payload["base_total_bits"])),
        fixed_bits_total=float(payload["fixed_bits_total"]),
        allowed=allowed,
        max_passes=args.max_passes,
    )
    result["bits_per_param"] = result["bits_total"] / max(float(payload["total_params"]), 1.0)
    refined_assignment = dict(payload["base_assignment"])
    refined_assignment.update(expand_unit_assignment(units, result["choices"]))
    unit_map = {unit.key: unit for unit in units}
    refined_predicted_dloss = float(payload.get("fixed_predicted_dloss_total", 0.0))
    for unit in units:
        chosen_fmt = result["choices"].get(unit.key, unit.base_fmt)
        refined_predicted_dloss += unit_map[unit.key].option_map[chosen_fmt].predicted_dloss

    calibrated_last_token_kl_estimate = None
    if args.calibration:
        with open(args.calibration) as f:
            calib = json.load(f)
        mapper = _fit_piecewise_monotone(
            [
                (row["predicted_dloss"], row["actual_last_token_kl"])
                for row in calib["results"]
            ]
        )
        if mapper is not None:
            calibrated_last_token_kl_estimate = float(mapper(refined_predicted_dloss))

    out = {
        "source": args.interactions,
        "calibration": args.calibration,
        "base_last_token_kl": payload["base_last_token_kl"],
        "base_predicted_dloss": payload.get("base_predicted_dloss"),
        "refined_predicted_dloss": refined_predicted_dloss,
        "refined_delta_kl_estimate": result["objective_delta"],
        "refined_last_token_kl_estimate": payload["base_last_token_kl"] + result["objective_delta"],
        "calibrated_last_token_kl_estimate": calibrated_last_token_kl_estimate,
        "bits_total": result["bits_total"],
        "bits_per_param": result["bits_per_param"],
        "selected_choices": result["choices"],
        "refined_assignment": refined_assignment,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[refine] wrote {args.output}")


if __name__ == "__main__":
    main()
