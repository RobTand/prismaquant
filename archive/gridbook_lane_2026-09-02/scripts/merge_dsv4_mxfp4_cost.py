#!/usr/bin/env python3
"""Merge the ≤4.25-bpw ladder cost run with the shipped v2 K14/K15/K36 table,
and validate the interpolator against the columns both runs share.

WHY A MERGE AT ALL. The v2 production run measured exactly three CB rungs
(NVFP4_CB_K14, NVFP4_CB_K15, FP8_CB_K36) over all 43 layers. The mxfp4 run
widens the expert menu to every rung at or under 4.25 bpw and prices it with
the RD-ladder interpolator, which measures anchors + a holdout per family and
FITS the rest. Those two tables overlap on K14/K15 — measured in one, fitted
in the other — and that overlap is not a redundancy to discard. It is the only
out-of-sample check on the interpolator that uses THIS model's real production
rows rather than a synthetic fixture, so this script reports it before it
merges, and merging always PREFERS the measured value.

FP8_CB_K36 is kept from v2 even though 4.508 bpw is above the menu cap: it is
dominated for experts (MXFP4_SOURCE is fewer bytes at exactly zero error) but
it remains the body's measured option, and dropping a measured column because
a different unit class cannot use it would be throwing away evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import statistics
from collections import defaultdict
from pathlib import Path

BAND_INTERPOLATED = "band_interpolated"


def _load(path: Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _family(fmt: str) -> str:
    if fmt.startswith("NVFP4_CB_"):
        return "nvfp4_cb"
    if fmt.startswith("FP8_CB_"):
        return "fp8_cb"
    return "other"


def _dloss_like(entry: dict) -> float | None:
    """The scalar the DP would price this row on, before any P5a gain."""
    for key in ("predicted_dloss", "output_mse", "weight_mse"):
        if key in entry:
            try:
                value = float(entry[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return None


def validate_overlap(base: dict, new: dict) -> dict:
    """Per-family relative error of FITTED vs MEASURED on shared columns.

    This is the interpolator's error band, measured rather than assumed. The
    comparison is restricted to rows the new run FITTED (``cost_source ==
    band_interpolated``); a row the ladder's per-tensor holdout gate rejected
    was re-measured, and comparing two measurements would flatter the fit.
    """
    base_costs = base["costs"]
    new_costs = new["costs"]
    per_family: dict[str, list[float]] = defaultdict(list)
    per_format: dict[str, list[float]] = defaultdict(list)
    n_fitted = 0
    n_regated = 0
    for name, new_row in new_costs.items():
        base_row = base_costs.get(name)
        if not base_row:
            continue
        for fmt, new_entry in new_row.items():
            base_entry = base_row.get(fmt)
            if not isinstance(base_entry, dict) or not isinstance(new_entry, dict):
                continue
            if new_entry.get("cost_source") != BAND_INTERPOLATED:
                n_regated += 1
                continue
            n_fitted += 1
            fitted = _dloss_like(new_entry)
            measured = _dloss_like(base_entry)
            if fitted is None or measured is None or measured <= 0.0:
                continue
            rel = (fitted - measured) / abs(measured)
            if math.isfinite(rel):
                per_family[_family(fmt)].append(rel)
                per_format[fmt].append(rel)

    def _summary(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        absolute = [abs(v) for v in values]
        return {
            "n": len(values),
            "mean_signed_rel_err": statistics.fmean(values),
            "median_abs_rel_err": statistics.median(absolute),
            "p90_abs_rel_err": sorted(absolute)[int(0.9 * (len(absolute) - 1))],
            "max_abs_rel_err": max(absolute),
        }

    return {
        "schema": "prismaquant.ladder_overlap_validation.v1",
        "comparison": (
            "new-run FITTED value vs v2-run MEASURED value, on the columns "
            "both runs contain (NVFP4_CB_K14/K15); relative to the measured "
            "value"
        ),
        "rows_fitted_and_comparable": n_fitted,
        "rows_skipped_not_fitted": n_regated,
        "by_family": {fam: _summary(v) for fam, v in sorted(per_family.items())},
        "by_format": {f: _summary(v) for f, v in sorted(per_format.items())},
    }


def merge(base: dict, new: dict) -> tuple[dict, dict]:
    """Union the two cost tables, preferring a MEASURED value over a fitted one."""
    merged_costs: dict[str, dict] = {}
    stats = {"measured_kept": 0, "fitted_kept": 0, "fitted_overridden": 0}
    names = set(base["costs"]) | set(new["costs"])
    for name in names:
        row: dict[str, dict] = {}
        row.update(base["costs"].get(name, {}))
        for fmt, entry in new["costs"].get(name, {}).items():
            existing = row.get(fmt)
            if isinstance(existing, dict) and isinstance(entry, dict):
                # Both runs priced this column. A measured value always wins:
                # the fit exists to cover columns nobody measured, never to
                # overwrite a measurement that already exists.
                if entry.get("cost_source") == BAND_INTERPOLATED:
                    stats["fitted_overridden"] += 1
                    continue
            row[fmt] = entry
            if isinstance(entry, dict) and entry.get("cost_source") == BAND_INTERPOLATED:
                stats["fitted_kept"] += 1
            else:
                stats["measured_kept"] += 1
        merged_costs[name] = row
    formats = sorted(set(base.get("formats", [])) | set(new.get("formats", [])))
    merged = {
        "costs": merged_costs,
        "formats": formats,
        "provenance": {
            **(base.get("provenance") or {}),
            "merged_from": ["v2_measured_k14_k15_k36", "mxfp4_ladder_run"],
        },
        "meta": {
            **(base.get("meta") or {}),
            "merge": stats,
        },
    }
    return merged, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="v2 measured cost pickle")
    ap.add_argument("--new", required=True, help="ladder-run cost pickle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    base = _load(Path(args.base))
    new = _load(Path(args.new))
    validation = validate_overlap(base, new)
    merged, stats = merge(base, new)

    with open(args.out, "wb") as fh:
        pickle.dump(merged, fh)
    report = {
        "validation": validation,
        "merge": stats,
        "formats": merged["formats"],
        "n_rows": len(merged["costs"]),
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
