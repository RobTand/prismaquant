#!/usr/bin/env python3
"""Development-only transfer test of Gridbook-style sparse-rate interpolation.

This bounded scalar replay compares a Gridbook-inspired shared log shape with
the established log chord.  It cannot qualify an allocator, wire, runtime, or
activation-floor mechanism.  The Gridbook ``K % 4`` codebook term is purposely
absent: Tessera rates are q256 column-rate coordinates, not codebook rungs.
"""
from __future__ import annotations

import argparse
import cProfile
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from experiments.sparse_rate_models import (
    CURRENCY, RANCHOR, RHI, RLO, empirical_gate, group_keys, metrics,
    validate_dataset,
)
from prismaquant.anchored_shape import (
    AnchoredShapeError, LogShapeObservation, fit_anchor_correction,
    fit_centered_log_shape, predict_anchored,
)

SCHEMA = "prismaquant.sparse_rate_gridbook_transfer.v1"
MODELS = (
    "log_chord", "gridbook_shared_logshape_two_anchor",
    "gridbook_shared_logshape_one_anchor960", "rate_only_logshape_two_anchor",
    "floor_mixture_beta1", "floor_mixture_beta2", "floor_mixture_beta3",
    "floor_mixture_beta4",
)
FLOOR_MODELS = tuple(model for model in MODELS if model.startswith("floor_mixture_"))


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    with path.open("x") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def rate_features(rates: np.ndarray, *, gridbook: bool) -> dict[str, tuple[float, ...]]:
    """Coordinates for the transferable Gridbook law, with no modulo feature."""
    result = {}
    for rate in rates:
        k = float(rate) / 256.0
        result[str(int(rate))] = (k, max(0.0, k - 4.0)) if gridbook else (k,)
    return result


def pilot_ids(data: dict[str, np.ndarray], train: np.ndarray, *, limit: int) -> np.ndarray:
    """Choose by name before any interior-cost access; endpoints establish eligibility."""
    rates, values = data["rates"], data["values"]
    lo, hi = (int(np.flatnonzero(rates == rate)[0]) for rate in (RLO, RHI))
    eligible = np.flatnonzero(train & np.isfinite(values[:, lo]) & np.isfinite(values[:, hi]))
    ordered = sorted(eligible, key=lambda index: hashlib.sha256(
        str(data["qnames"][index]).encode()).digest())
    return np.asarray(ordered[:limit], dtype=np.int64)


def fit_shape(data: dict[str, np.ndarray], ids: np.ndarray, *, gridbook: bool):
    features = rate_features(data["rates"], gridbook=gridbook)
    observations = []
    for index in ids:
        for column, rate in enumerate(data["rates"]):
            value = data["values"][index, column]
            if np.isfinite(value) and value > 0:
                observations.append(LogShapeObservation(str(int(index)), str(int(rate)), float(value)))
    return fit_centered_log_shape(observations, features), features


def anchored_prediction(shape, features, values: np.ndarray, rates: np.ndarray,
                        target: np.ndarray, anchors: tuple[int, ...]) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    anchor_columns = [int(np.flatnonzero(rates == rate)[0]) for rate in anchors]
    for index in np.flatnonzero(target):
        if not all(np.isfinite(values[index, column]) and values[index, column] > 0
                   for column in anchor_columns):
            continue
        correction = fit_anchor_correction(
            shape, {str(rate): float(values[index, column])
                    for rate, column in zip(anchors, anchor_columns)},
            {key: float(rate) for key, rate in ((str(int(rate)), int(rate)) for rate in rates)},
        )
        for column, rate in enumerate(rates):
            result[index, column] = predict_anchored(correction, str(int(rate)))
    return result


def log_chord(values: np.ndarray, rates: np.ndarray, target: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    lo, hi = (int(np.flatnonzero(rates == rate)[0]) for rate in (RLO, RHI))
    for index in np.flatnonzero(target):
        left, right = values[index, lo], values[index, hi]
        if not (np.isfinite(left) and left > 0 and np.isfinite(right) and right > 0):
            continue
        for column, rate in enumerate(rates):
            if rate < RLO or rate > RHI:
                continue
            if column == lo:
                result[index, column] = left
            elif column == hi:
                result[index, column] = right
            else:
                t = (float(rate) - RLO) / (RHI - RLO)
                result[index, column] = 2.0 ** ((1.0 - t) * math.log2(left) + t * math.log2(right))
    return result


def floor_mixture(values: np.ndarray, rates: np.ndarray, target: np.ndarray, beta: int,
                  allowed: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray]:
    """Fit D=F+C*q to exact endpoints; retain positive non-strict predictions.

    ``F < 0`` is reported, never clamped.  Consequently the strict nonnegative
    floor interpretation is explicitly invalid for those units while the
    endpoint-preserving numerical comparator remains auditable.
    """
    result = np.full(values.shape, np.nan)
    strict_valid = np.zeros(len(values), dtype=bool)
    lo, hi = (int(np.flatnonzero(rates == rate)[0]) for rate in (RLO, RHI))
    floors, negative, nonpositive = [], 0, 0
    for index in np.flatnonzero(target & allowed):
        left, right = values[index, lo], values[index, hi]
        if not (np.isfinite(left) and left > 0 and np.isfinite(right) and right > 0):
            continue
        def q(rate):
            k, fraction = divmod(float(rate) / 256.0, 1.0)
            return ((1.0 - fraction) * 2.0 ** (-beta * math.floor(k))
                    + fraction * 2.0 ** (-beta * (math.floor(k) + 1)))
        qlo, qhi = q(RLO), q(RHI)
        coefficient = (left - right) / (qlo - qhi)
        floor = left - coefficient * qlo
        floors.append(float(floor))
        negative += int(floor < 0.0)
        strict_valid[index] = floor >= 0.0
        for column, rate in enumerate(rates):
            if rate < RLO or rate > RHI:
                continue
            predicted = left if column == lo else right if column == hi else floor + coefficient * q(int(rate))
            if predicted > 0 and math.isfinite(predicted):
                result[index, column] = predicted
            else:
                nonpositive += 1
    return result, {
        "beta": beta, "eligible_units": int(np.count_nonzero(target & allowed)),
        "fitted_units": len(floors), "negative_floor_count": negative,
        "strict_nonnegative_floor_valid": negative == 0,
        "strict_nonnegative_floor_note": "invalid when any fitted F is negative; anchors were never clamped",
        "nonpositive_interpolations_omitted": nonpositive,
    }, strict_valid


def masked_truth(values: np.ndarray, rates: np.ndarray, target: np.ndarray,
                 anchors: tuple[int, ...], allowed: np.ndarray | None = None) -> np.ndarray:
    truth = np.where(target[:, None], values, np.nan).copy()
    if allowed is not None:
        truth[~allowed] = np.nan
    for rate in anchors:
        truth[:, rates == rate] = np.nan
    truth[:, (rates < RLO) | (rates > RHI)] = np.nan
    return truth


def per_group(predictions, truth, groups):
    return {str(key): metrics(predictions[groups == key], truth[groups == key])
            for key in sorted(set(groups))}


def evaluate_model(model, data, development, groups, pilot_limit):
    rates, values, layers = data["rates"], data["values"], data["layers"]
    predictions = np.full(values.shape, np.nan)
    panels, model_details = [], {}
    strict_valid = None
    started = time.perf_counter()
    for group_key in sorted(set(groups)):
        group = groups == group_key
        for fold in range(4):
            train = development & group & (layers % 4 != fold)
            target = development & group & (layers % 4 == fold)
            if not target.any() or len(set(layers[train])) < 3:
                continue
            ids = pilot_ids(data, train, limit=pilot_limit)
            panel = {"segment": str(group_key), "fold": fold,
                     "target_layers": sorted(map(int, set(layers[target]))),
                     "pilot_units": int(len(ids)),
                     "pilot_roster_sha256": hashlib.sha256("\n".join(sorted(
                         map(str, data["qnames"][ids]))).encode()).hexdigest()}
            panels.append(panel)
            local = np.flatnonzero(group)
            local_target = target[local]
            try:
                if model == "log_chord":
                    p = log_chord(values[local], rates, local_target)
                elif model in FLOOR_MODELS:
                    beta = int(model.rsplit("beta", 1)[1])
                    allowed = np.isin(data["families"][local], ("TESSERA_E4M3_K1", "TESSERA_BF16_K1"))
                    p, detail, local_strict_valid = floor_mixture(values[local], rates, local_target, beta, allowed)
                    if strict_valid is None:
                        strict_valid = np.zeros(len(values), dtype=bool)
                    strict_valid[local] |= local_strict_valid
                    model_details[f"{group_key}/fold{fold}"] = detail
                else:
                    shape, features = fit_shape(data, ids, gridbook=(model != "rate_only_logshape_two_anchor"))
                    anchors = (RANCHOR,) if model.endswith("one_anchor960") else (RLO, RHI)
                    p = anchored_prediction(shape, features, values[local], rates, local_target, anchors)
                predictions[local[target[local]]] = p[target[local]]
            except AnchoredShapeError as exc:
                model_details[f"{group_key}/fold{fold}"] = {"fit_error": str(exc)}
    anchors = (RANCHOR,) if model.endswith("one_anchor960") else (RLO, RHI)
    anchor_columns = [int(np.flatnonzero(rates == rate)[0]) for rate in anchors]
    eligible = np.all(np.isfinite(values[:, anchor_columns]) & (values[:, anchor_columns] > 0), axis=1)
    metric_target = development & eligible
    allowed = None
    unsupported = 0
    if model in FLOOR_MODELS:
        allowed = np.isin(data["families"], ("TESSERA_E4M3_K1", "TESSERA_BF16_K1"))
        unsupported = int(np.count_nonzero(development & ~allowed))
    truth = masked_truth(values, rates, metric_target, anchors, allowed)
    report = {"model": model, "anchors": list(anchors), "fit_predict_seconds": time.perf_counter() - started,
              "pilot_units_per_segment": pilot_limit, "pilot_panels": panels,
              "units_without_required_anchors": int(np.count_nonzero(development & ~eligible)),
              "unsupported_geometry_units": unsupported, "details": model_details,
              "metrics": metrics(predictions, truth), "groups": per_group(predictions, truth, groups)}
    report["rates"] = {str(int(rate)): metrics(predictions[:, column], truth[:, column])
                       for column, rate in enumerate(rates)}
    report["segment_empirical_gate"] = {key: {"passed": empirical_gate(value), "production_qualified": False}
                                        for key, value in report["groups"].items()}
    if model in FLOOR_MODELS:
        strict_truth = masked_truth(values, rates, metric_target & allowed & strict_valid, anchors)
        report["strict_nonnegative_floor_metrics"] = metrics(predictions, strict_truth)
        report["strict_nonnegative_floor_groups"] = per_group(predictions, strict_truth, groups)
    return report, predictions, truth


def profile_model(arguments):
    model, data, development, groups, pilot_limit, out = arguments
    profile = cProfile.Profile()
    result = profile.runcall(evaluate_model, model, data, development, groups, pilot_limit)
    profile.dump_stats(str(Path(out) / f"{model}.prof"))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot-units-per-segment", type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.pilot_units_per_segment < 1:
        parser.error("workers must be 1..4 and pilot size positive")
    root, out = Path(args.dataset), Path(args.out)
    if out.exists():
        raise ValueError("output exists; preserve prior experiment")
    with np.load(root / "sparse_rate_dataset.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    manifest = json.loads((root / "manifest.json").read_text())
    validate_dataset(data, manifest)
    if manifest.get("npz_sha256") != digest(root / "sparse_rate_dataset.npz"):
        raise ValueError("dataset digest differs from manifest")
    if not {RLO, RANCHOR, RHI} <= set(map(int, data["rates"])):
        raise ValueError("dataset lacks declared anchors")
    out.mkdir(parents=True)
    development = data["layers"] % 5 != 3
    identity = {"dataset_npz_sha256": digest(root / "sparse_rate_dataset.npz"),
                "dataset_manifest_sha256": digest(root / "manifest.json"),
                "script_sha256": digest(__file__),
                "numerical_core_sha256": digest(Path("prismaquant/anchored_shape.py")),
                "gridbook_source_sha256": digest("/mnt/shared/tessera-measurements/glm-canonical-census-20260908/sparse-rate-20260911/measurement-source-01/tools/dsv4_afast_campaign.py"),
                "grammar_source_sha256": digest("/mnt/shared/tessera-measurements/glm-canonical-census-20260908/identity-reseal-20260911/producer-source-d403cc5a31/src/tessera/grammar.py")}
    plan = {"schema": SCHEMA, "currency": CURRENCY, "identity": identity,
            "research_only": True, "production_qualified": False,
            "final_untouched": False, "final_untouched_note": "historically inspected final evidence exists, but this analysis only fits and scores development layers",
            "development_rule": "layer % 5 != 3", "development_cv": "layer % 4 within development and exact group_keys segments",
            "pilot_policy": "smallest SHA256(qname) among endpoint-eligible fitting units before interior-cost reads",
            "pilot_units_per_segment": args.pilot_units_per_segment,
            "models": list(MODELS),
            "gridbook_transfer": {"source": "tools/dsv4_afast_campaign.py:89-133", "retained": "shared log rate law plus per-unit anchor correction", "excluded": "K % 4 codebook feature", "log_base": "anchored_shape base10"},
            "floor_geometry_hypothesis": "Tessera schedules neighboring integer column rates; activation flooring is not known and LDLQ/output terms can break linear mixture behavior",
            "profile_note": "cProfile records CPU call attribution per comparator; it is not a speed or performance claim"}
    write_json(out / "plan.json", plan)
    groups = group_keys(data)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(profile_model, [(model, data, development, groups,
                                                 args.pilot_units_per_segment, str(out))
                                                for model in MODELS]))
    reports = [result[0] for result in results]
    write_json(out / "report.json", {**plan, "models": reports})
    for report, predictions, truth in results:
        np.savez_compressed(out / f"{report['model']}.npz", predictions=predictions, truth=truth)
        print(json.dumps({"model": report["model"], **report["metrics"],
                          "fit_predict_seconds": report["fit_predict_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
