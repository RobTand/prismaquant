#!/usr/bin/env python3
"""Bounded, layer-held-out sparse rate model development on saved measurements.

This experiment predicts the dataset's scalar currency. It emits no allocator
prices, invented measurements, wires, or serving qualification. Final layers
are opened only by the separately invoked final stage after choice.json exists.
"""
from __future__ import annotations

import argparse
import cProfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from experiments.sparse_rate_context import CONTEXT_MODELS, predict_context_surface
from prismaquant.anchored_shape import (
    AnchoredShapeError, LogShapeObservation, fit_endpoint_curvature,
)

SCHEMA = "prismaquant.sparse_rate_model_study.v1"
TWO_MODELS = ("log_chord", "value_chord", "log_constant", "value_constant",
              "log_ridge1", "log_ridge2", "value_ridge1", "value_ridge2",
              "log_surface1", "log_surface2", "value_surface1", "value_surface2",
              "exp1", "exp2", "exp3", "exp4") + CONTEXT_MODELS
ONE_MODELS = ("one_constant", "one_ridge1", "one_ridge2")
RLO, RHI, RANCHOR = 832, 1088, 960
CURRENCY = "output_mse_under_route_activation_contract"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    with Path(path).open("x") as f:
        json.dump(payload, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def validate_dataset(data, manifest):
    if (manifest.get("schema") != "prismaquant.sparse_rate_dataset.v1"
            or manifest.get("currency") != CURRENCY
            or manifest.get("research_only") is not True
            or manifest.get("not_joint_aura") is not True
            or manifest.get("not_serving_admission") is not True):
        raise ValueError("dataset contract is unsupported")
    required = {"values", "rates", "qnames", "families", "activation_contracts",
                "layers", "roles", "structures", "rows", "cols", "counts",
                "wire_bytes", "encode_seconds"}
    if set(data) != required or data["values"].ndim != 2:
        raise ValueError("dataset array inventory or dimensions are invalid")
    n, m = data["values"].shape
    for name, array in data.items():
        shape = (n, m) if name in {"values", "wire_bytes", "encode_seconds"} else ((m,) if name == "rates" else (n,))
        if array.shape != shape:
            raise ValueError(f"invalid dataset array shape: {name}")
        if manifest.get("arrays", {}).get(name) != {"dtype": str(array.dtype), "shape": list(shape)}:
            raise ValueError(f"array differs from manifest: {name}")
    for name in ("rates", "layers", "rows", "cols", "counts", "wire_bytes"):
        if data[name].dtype.kind not in "iu" or np.any(data[name] < 0):
            raise ValueError(f"invalid nonnegative integer array: {name}")
    if np.any(np.diff(data["rates"]) <= 0):
        raise ValueError("rate coordinates must be strictly increasing")
    if data["values"].dtype.kind != "f" or np.any(np.isinf(data["values"])) or np.any(data["values"] <= 0):
        raise ValueError("costs must be positive finite measurements or missing NaNs")


def validate_choice(choice, identity):
    if (choice.get("schema") != SCHEMA or choice.get("identity") != identity
            or choice.get("two_anchor") not in TWO_MODELS
            or choice.get("one_anchor") not in ONE_MODELS):
        raise ValueError("frozen choice does not match this implementation and dataset")


def metrics(pred, truth):
    present = np.isfinite(truth) & (truth > 0)
    valid = present & np.isfinite(pred) & (pred > 0)
    error = np.abs(np.log2(pred[valid]) - np.log2(truth[valid]))
    relative = np.abs(pred[valid] / truth[valid] - 1)
    out = {"truth_count": int(present.sum()), "predicted_count": int(valid.sum()),
           "missing_predictions": int((present & ~valid).sum())}
    if error.size:
        for label, values in (("abs_log2", error), ("relative", relative)):
            out[label] = {"mean": float(values.mean()),
                          **{q: float(np.quantile(values, p)) for q, p in
                             (("p50", .5), ("p95", .95), ("p99", .99), ("max", 1.0))}}
    return out


def empirical_gate(result):
    """A declared held-out scalar screen, never a production qualification."""
    relative = result.get("relative", {})
    return (result["truth_count"] >= 100 and result["missing_predictions"] == 0
            and relative.get("p99", math.inf) <= .01
            and relative.get("max", math.inf) <= .05)


def robust_fit(x, y, degree):
    """Small ridge/Huber fit; standardization uses only fitting observations."""
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    z = (x - mean) / scale
    design = basis(z, degree)
    ridge = np.eye(design.shape[1]) * (len(y) * 1e-5)
    ridge[0, 0] = 0
    weights = np.ones(len(y))
    coefficients = np.zeros(design.shape[1])
    for _ in range(5):
        weighted = design * weights[:, None]
        coefficients = np.linalg.solve(design.T @ weighted + ridge, weighted.T @ y)
        residual = y - design @ coefficients
        mad = np.median(np.abs(residual - np.median(residual)))
        if mad == 0:
            break
        ratio = np.abs(residual) / (1.5 * mad)
        weights = np.ones(len(y))
        np.divide(1, ratio, out=weights, where=ratio > 1)
    return mean, scale, coefficients


def basis(z, degree):
    columns = [np.ones(len(z))]
    columns.extend(z[:, j] for j in range(z.shape[1]))
    if degree == 2:
        columns.extend(z[:, j] * z[:, k] for j in range(z.shape[1])
                       for k in range(j, z.shape[1]))
    return np.column_stack(columns)


def reg_predict(fit, features, degree):
    mean, scale, coef = fit
    return basis((features - mean) / scale, degree) @ coef


def two_predict(model, values, rates, train, target, *, fitted_models=None):
    lo, hi = (int(np.flatnonzero(rates == r)[0]) for r in (RLO, RHI))
    fit_valid = train & np.isfinite(values[:, lo]) & np.isfinite(values[:, hi])
    target_valid = target & np.isfinite(values[:, lo]) & np.isfinite(values[:, hi])
    out = np.full(values.shape, np.nan)
    ids = np.flatnonzero(target_valid)
    if not len(ids):
        return out
    left, right = values[ids, lo], values[ids, hi]
    logleft, logright = np.log2(left), np.log2(right)
    train_left, train_right = values[:, lo], values[:, hi]
    degree = 2 if model.endswith("2") else 1
    surface_fit = None
    if "surface" in model:
        observations = [LogShapeObservation(f"{i:012d}", str(int(rate)), values[i, j])
                        for j, rate in enumerate(rates) if RLO <= rate <= RHI
                        for i in np.flatnonzero(fit_valid & np.isfinite(values[:, j]))]
        try:
            shape = fit_endpoint_curvature(observations,
                {str(int(r)): int(r) for r in rates if RLO <= r <= RHI},
                mode="log2" if model.startswith("log") else "value", degree=degree)
            surface_fit = (np.asarray(shape.means), np.asarray(shape.scales),
                           np.asarray(shape.coefficients))
            if fitted_models is not None:
                fitted_models.append(asdict(shape))
        except AnchoredShapeError:
            # An unidentifiable pilot must remain a visible coverage gap.
            surface_fit = None
    for j, rate in enumerate(rates):
        if rate < RLO or rate > RHI:
            continue
        t = (int(rate) - RLO) / (RHI - RLO)
        if j in (lo, hi):
            out[ids, j] = values[ids, j]
            continue
        log_chord = (1-t) * logleft + t * logright
        if model == "log_chord":
            out[ids, j] = np.exp2(log_chord)
            continue
        if model == "value_chord":
            out[ids, j] = (1-t) * left + t * right
            continue
        if model.startswith("exp"):
            beta = int(model[-1])
            # An affine shared exponential template can represent an additive
            # activation floor; its two measured endpoints remain exact.
            fraction = np.expm1(-beta * math.log(2) * (1-t)) / np.expm1(-beta * math.log(2))
            fraction *= np.exp2(-beta * t)
            out[ids, j] = right + (left-right) * fraction
            continue
        if "surface" in model:
            if surface_fit is None:
                continue
            features = np.column_stack((logleft, logright-logleft, np.full(len(ids), t)))
            correction = reg_predict(surface_fit, features, degree)
            if model.startswith("log"):
                out[ids, j] = np.exp2(log_chord + t*(1-t)*correction)
            else:
                out[ids, j] = (1-t)*left+t*right + left*t*(1-t)*correction
            continue
        measured = fit_valid & np.isfinite(values[:, j])
        fit_ids = np.flatnonzero(measured)
        if len(fit_ids) < 12:
            continue
        a, b, y = train_left[fit_ids], train_right[fit_ids], values[fit_ids, j]
        if model.startswith("log"):
            response = (np.log2(y) - ((1-t)*np.log2(a)+t*np.log2(b))) / (t*(1-t))
        else:
            # Normalize by positive endpoint level, not their difference: a
            # nearly flat pair must not amplify roundoff into a giant target.
            response = (y - ((1-t)*a+t*b)) / a / (t*(1-t))
        if model.endswith("constant"):
            correction = np.full(len(ids), np.median(response))
        else:
            features = np.column_stack((np.log2(a), np.log2(b)-np.log2(a)))
            target_features = np.column_stack((logleft, logright-logleft))
            correction = reg_predict(robust_fit(features, response, degree), target_features, degree)
        if model.startswith("log"):
            out[ids, j] = np.exp2(log_chord + t*(1-t)*correction)
        else:
            out[ids, j] = (1-t)*left+t*right + left*t*(1-t)*correction
    return out


def one_predict(model, values, rates, train, target, counts):
    anchor = int(np.flatnonzero(rates == RANCHOR)[0])
    out = np.full(values.shape, np.nan)
    ids = np.flatnonzero(target & np.isfinite(values[:, anchor]))
    degree = 2 if model.endswith("2") else 1
    for j, rate in enumerate(rates):
        if not RLO <= rate <= RHI:
            continue
        if j == anchor:
            out[ids, j] = values[ids, j]
            continue
        fit_ids = np.flatnonzero(train & np.isfinite(values[:, anchor]) & np.isfinite(values[:, j]))
        if len(fit_ids) < 12:
            continue
        base = np.log2(values[fit_ids, anchor])
        response = np.log2(values[fit_ids, j]) - base
        target_base = np.log2(values[ids, anchor])
        if model == "one_constant":
            correction = np.full(len(ids), np.median(response))
        else:
            # Routed-token support is known before encoding. log1p handles a
            # zero-support entry without manufacturing a positive cost floor.
            features = np.column_stack((base, np.log1p(counts[fit_ids])))
            target_features = np.column_stack((target_base, np.log1p(counts[ids])))
            correction = reg_predict(robust_fit(features, response, degree), target_features, degree)
        out[ids, j] = np.exp2(target_base + correction)
    return out


def group_keys(data):
    # Exact current recorded family/activation/shape/structure boundaries. The
    # dataset manifest additionally binds the source, calibration and recipe.
    return np.array(["|".join(map(str, fields)) for fields in zip(
        data["families"], data["activation_contracts"], data["roles"],
        data["structures"], data["rows"], data["cols"])])


def masked_truth(values, rates, target, anchors):
    truth = np.where(target[:, None], values, np.nan).copy()
    for rate in anchors:
        truth[:, rates == rate] = np.nan
    truth[:, (rates < RLO) | (rates > RHI)] = np.nan
    return truth


def evaluate_model(model, data, development, final, groups, *, final_stage=False,
                   pilot_limit=256):
    values, rates, layers = data["values"], data["rates"], data["layers"]
    predictions = np.full(values.shape, np.nan)
    pilot_panels = []
    fitted_models = []
    begin = time.perf_counter()
    for key in sorted(set(groups)):
        group = groups == key
        if final_stage:
            splits = [(development & group, final & group)]
        else:
            splits = [(development & group & (layers % 4 != fold),
                       development & group & (layers % 4 == fold)) for fold in range(4)]
        for train, target in splits:
            if not target.any() or len(set(layers[train])) < 3:
                continue
            # Select the shared pilot from names only, before reading interior
            # costs. The cap counts units across fitting layers in this segment.
            # Zero means an explicitly labelled full-data diagnostic comparator.
            ids = np.flatnonzero(train)
            if pilot_limit and len(ids) > pilot_limit:
                ids = np.array(sorted(ids, key=lambda i: hashlib.sha256(
                    str(data["qnames"][i]).encode()).digest())[:pilot_limit])
                train = np.zeros(len(values), dtype=bool)
                train[ids] = True
            pilot_panels.append({"segment": str(key),
                                 "target_layers": sorted(map(int, set(layers[target]))),
                                 "pilot_units": int(len(ids)),
                                 "pilot_measurements": int(np.isfinite(values[ids]).sum()),
                                 "pilot_roster_sha256": hashlib.sha256("\n".join(sorted(
                                     map(str, data["qnames"][ids]))).encode()).hexdigest()})
            # Work only on the local segment; this also prevents accidental
            # transfer between dense and routed units with matching shapes.
            local = np.flatnonzero(group)
            if model in CONTEXT_MODELS:
                p = predict_context_surface(model, data, rates, train, target, group)
                predictions[target] = p[target]
                continue
            if model in ONE_MODELS:
                p = one_predict(model, values[local], rates, train[local], target[local], data["counts"][local])
            else:
                panel_fits = []
                p = two_predict(model, values[local], rates, train[local], target[local],
                                fitted_models=panel_fits)
                fitted_models.extend({"segment": str(key),
                                      "target_layers": sorted(map(int, set(layers[target]))),
                                      "model": fit} for fit in panel_fits)
            selected = local[target[local]]
            predictions[selected] = p[target[local]]
    elapsed = time.perf_counter() - begin
    anchors = [RANCHOR] if model in ONE_MODELS else [RLO, RHI]
    target = final if final_stage else development
    eligibility = np.ones(len(values), dtype=bool)
    for anchor in anchors:
        eligibility &= np.isfinite(values[:, int(np.flatnonzero(rates == anchor)[0])])
    unsupported_units = int((target & ~eligibility).sum())
    target = target & eligibility
    truth = masked_truth(values, rates, target, anchors)
    report = {"model": model, "anchors": anchors, "fit_predict_seconds": elapsed,
              "pilot_units_per_segment": pilot_limit, "pilot_panels": pilot_panels,
              "fitted_models": fitted_models,
              "units_without_required_anchors": unsupported_units,
              "metrics": metrics(predictions, truth), "groups": {}}
    report["rates"] = {str(int(r)): metrics(predictions[:, j], truth[:, j])
                       for j, r in enumerate(rates)}
    valid = np.isfinite(truth) & np.isfinite(predictions) & (predictions > 0)
    row, col = np.nonzero(valid)
    errors = np.abs(predictions[row, col] / truth[row, col] - 1)
    worst = np.argsort(errors)[-20:][::-1]
    report["worst_cases"] = [{"qname": str(data["qnames"][row[k]]),
                              "family": str(data["families"][row[k]]),
                              "layer": int(layers[row[k]]), "rate": int(rates[col[k]]),
                              "counts": int(data["counts"][row[k]]),
                              "relative_error": float(errors[k]),
                              "predicted": float(predictions[row[k], col[k]]),
                              "measured": float(truth[row[k], col[k]])} for k in worst]
    for key in sorted(set(groups)):
        mask = groups == key
        report["groups"][key] = metrics(predictions[mask], truth[mask])
    report["segment_empirical_gate"] = {
        key: {"passed": empirical_gate(value), "production_qualified": False}
        for key, value in report["groups"].items()}
    return report, predictions, truth


def score(report):
    m = report["metrics"]
    # Coverage first, then tail accuracy; averages cannot hide a bad segment.
    return (m["missing_predictions"], m.get("abs_log2", {}).get("p99", math.inf),
            m.get("abs_log2", {}).get("max", math.inf))


def profile_model(arguments):
    model, data, development, final, groups, final_stage, out, pilot_limit = arguments
    profile = cProfile.Profile()
    result = profile.runcall(evaluate_model, model, data, development, final, groups,
                            final_stage=final_stage, pilot_limit=pilot_limit)
    profile.dump_stats(str(Path(out) / f"{model}.prof"))
    return result


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("development", "final"), default="development")
    ap.add_argument("--choice")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pilot-units-per-segment", type=int, default=256,
                    help="shared pilot cap; 0 is a full-data diagnostic, not a sparse plan")
    ap.add_argument("--models", help="comma-separated development comparator subset")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("workers must be positive")
    if args.pilot_units_per_segment < 0:
        ap.error("pilot cap must be nonnegative")
    root, out = Path(args.dataset), Path(args.out)
    if out.exists():
        raise ValueError("output exists; preserve the previous experiment")
    out.mkdir(parents=True)
    with np.load(root / "sparse_rate_dataset.npz", allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    manifest = json.loads((root / "manifest.json").read_text())
    identity = {"dataset_npz_sha256": digest(root / "sparse_rate_dataset.npz"),
                "dataset_manifest_sha256": digest(root / "manifest.json"),
                "script_sha256": digest(__file__),
                "context_script_sha256": digest(Path(__file__).with_name("sparse_rate_context.py")),
                "numerical_core_sha256": digest(Path(__file__).resolve().parents[1] / "prismaquant/anchored_shape.py"),
                "pilot_units_per_segment": args.pilot_units_per_segment}
    if manifest.get("npz_sha256") != identity["dataset_npz_sha256"]:
        raise ValueError("dataset NPZ digest differs from its manifest")
    validate_dataset(data, manifest)
    values, layers = data["values"], data["layers"]
    required = set((RLO, RHI, RANCHOR))
    if not required <= set(map(int, data["rates"])):
        raise ValueError("dataset lacks the declared base anchors")
    final = layers % 5 == 3
    development = ~final
    groups = group_keys(data)
    plan = {"schema": SCHEMA, "stage": args.stage, "identity": identity,
            "currency": CURRENCY, "research_only": True,
            "declared_span_q256": [RLO, RHI], "final_layer_rule": "layer % 5 == 3",
            "development_cv": "layer % 4, within development layers",
            "pilot_policy": "smallest SHA256(qname) in each fitting segment, independent of measured values",
            "pilot_units_per_segment": args.pilot_units_per_segment,
            "final_layers": sorted(map(int, set(layers[final]))),
            "development_layers": sorted(map(int, set(layers[development]))),
            "qualification_targets": {"p99_relative": .01, "max_relative": .05,
                                      "minimum_holdout_observations_per_segment": 100},
            "measured_rate_only_audit": True, "full_span_qualified": False,
            "joint_aura_qualified": False, "runtime_qualified": False}
    if args.stage == "development":
        models = TWO_MODELS + ONE_MODELS if args.models is None else tuple(args.models.split(","))
        if (len(set(models)) != len(models) or not set(models) <= set(TWO_MODELS + ONE_MODELS)
                or not set(models) & set(TWO_MODELS) or not set(models) & set(ONE_MODELS)):
            ap.error("development needs distinct supported comparators including one and two anchor models")
    else:
        if args.models:
            ap.error("final comparator roster comes only from the frozen choice")
        if not args.choice:
            ap.error("final requires the frozen development choice")
        choice = json.loads(Path(args.choice).read_text())
        validate_choice(choice, identity)
        models = tuple(dict.fromkeys(("log_chord", choice["two_anchor"], choice["one_anchor"])))
        plan["choice_sha256"] = digest(args.choice)
    write_json(out / "plan.json", plan)
    # Each process owns its profiler. CPython 3.12 cannot allocate the same
    # cProfile monitoring slot concurrently to independent thread profilers.
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(profile_model, [
            (model, data, development, final, groups, args.stage == "final", str(out),
             args.pilot_units_per_segment)
            for model in models]))
    reports = [result[0] for result in results]
    write_json(out / "report.json", {**plan, "models": reports,
                                    "note": "No full-band, final quality, or wall-time-saving claim follows from scalar replay."})
    for report, predictions, truth in results:
        np.savez_compressed(out / f"{report['model']}.npz", predictions=predictions, truth=truth)
        print(json.dumps({"model": report["model"], **report["metrics"],
                          "fit_predict_seconds": report["fit_predict_seconds"]}), flush=True)
    if args.stage == "development":
        two = min((r for r in reports if r["model"] in TWO_MODELS), key=score)
        one = min((r for r in reports if r["model"] in ONE_MODELS), key=score)
        write_json(out / "choice.json", {"schema": SCHEMA, "identity": identity,
                    "two_anchor": two["model"], "one_anchor": one["model"],
                    "selection_rule": "minimum missing predictions, p99 abs log2 error, maximum error",
                    "development_report_sha256": digest(out / "report.json")})


if __name__ == "__main__":
    main()
