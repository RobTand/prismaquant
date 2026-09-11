#!/usr/bin/env python3
"""Benchmark endpoint-polynomial evaluation on the measured sparse snapshot.

This is a bounded, CPU-only numerical benchmark.  The endpoint arrays and
unit identities come from a measured ``sparse_rate_dataset.v1`` snapshot;
the degree-two coefficients are deterministic synthetic coefficients used to
exercise the evaluator.  It measures direct standardized feature evaluation
at every rate against binding those features once and evaluating the resulting
Horner curve.  It does not measure prediction accuracy, GPU work, serving, or
end-to-end campaign performance.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import math
import platform
from pathlib import Path
import pstats
import sys
import time
from typing import Any

import numpy as np

from prismaquant.anchored_shape import EndpointCurvatureModel
from experiments.sparse_rate_models import validate_dataset


DATASET_SCHEMA = "prismaquant.sparse_rate_dataset.v1"
DATASET_NAME = "sparse_rate_dataset.npz"
MANIFEST_NAME = "manifest.json"
RATE_LO, RATE_HI, RATE_STEP = 832, 1088, 16
RATES = np.arange(RATE_LO, RATE_HI + RATE_STEP, RATE_STEP, dtype=np.int64)
BENCH_SCHEMA = "prismaquant.sparse_rate_inference_bench.v1"

# These are deliberately fixed research inputs, not fitted or measured model
# parameters.  The feature order is documented by EndpointCurvatureModel.
SYNTHETIC_MEANS = (0.0, 0.0, 0.5)
SYNTHETIC_SCALES = (8.0, 1.0, 0.5)
SYNTHETIC_COEFFICIENTS = (
    0.012, -0.003, 0.005, 0.009,
    0.0010, -0.0007, 0.0004, 0.0013, -0.0005, 0.0008,
)


class BenchmarkError(ValueError):
    """The requested benchmark cannot be run against the supplied snapshot."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"invalid dataset manifest: {path}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError("dataset manifest must be a JSON object")
    return value


def _load_dataset(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    npz_path, manifest_path = root / DATASET_NAME, root / MANIFEST_NAME
    if not npz_path.is_file() or not manifest_path.is_file():
        raise BenchmarkError(f"dataset must contain {DATASET_NAME} and {MANIFEST_NAME}")
    manifest = _json(manifest_path)
    if manifest.get("schema") != DATASET_SCHEMA:
        raise BenchmarkError("dataset contract schema is unsupported")
    if manifest.get("npz_file") != DATASET_NAME:
        raise BenchmarkError("dataset manifest names a different NPZ file")
    actual_hash = _sha256(npz_path)
    if manifest.get("npz_sha256") != actual_hash:
        raise BenchmarkError("dataset NPZ hash differs from its manifest")
    try:
        with np.load(npz_path, allow_pickle=False) as loaded:
            data = {name: loaded[name].copy() for name in loaded.files}
    except (OSError, ValueError) as exc:
        raise BenchmarkError("dataset NPZ cannot be read") from exc
    try:
        validate_dataset(data, manifest)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"dataset contract is invalid: {exc}") from exc
    if not np.array_equal(data["rates"], np.sort(data["rates"])):
        raise BenchmarkError("dataset rate coordinates are not sorted")
    required_rates = {RATE_LO, RATE_HI}
    if not required_rates.issubset(set(map(int, data["rates"].tolist()))):
        raise BenchmarkError("dataset lacks the required endpoint rates 832 and 1088")
    return data, manifest, actual_hash


def _direct_predict(left: np.ndarray, right: np.ndarray, rates: np.ndarray,
                    model: EndpointCurvatureModel) -> np.ndarray:
    """Evaluate standardized features and the degree-two basis per rate/unit."""
    output = np.empty((left.size, rates.size), dtype=np.float64)
    coefficients = model.coefficients
    for unit, (left_value, right_value) in enumerate(zip(left, right)):
        log_left = math.log2(float(left_value))
        raw_static = (log_left, math.log2(float(right_value)) - log_left)
        for column, rate in enumerate(rates):
            if int(rate) == RATE_LO:
                output[unit, column] = left_value
                continue
            if int(rate) == RATE_HI:
                output[unit, column] = right_value
                continue
            t = (int(rate) - model.rate_lo) / (model.rate_hi - model.rate_lo)
            raw = (raw_static[0], raw_static[1], t)
            z = tuple((value - mean) / scale
                      for value, mean, scale in zip(raw, model.means, model.scales))
            features = [1.0, *z]
            features.extend(z[index] * z[other]
                            for index in range(3) for other in range(index, 3))
            correction = sum(coefficient * feature
                             for coefficient, feature in zip(coefficients, features))
            log_value = (log_left + t * (math.log2(float(right_value)) - log_left)
                         + t * (1.0 - t) * correction)
            output[unit, column] = 2.0 ** log_value
    return output


def _bound_predict(left: np.ndarray, right: np.ndarray, rates: np.ndarray,
                   model: EndpointCurvatureModel) -> tuple[float, float, np.ndarray]:
    start_bind = time.perf_counter()
    curves = [model.bind(float(a), float(b)) for a, b in zip(left, right)]
    bind_seconds = time.perf_counter() - start_bind
    start_predict = time.perf_counter()
    output = np.empty((left.size, rates.size), dtype=np.float64)
    for unit, curve in enumerate(curves):
        for column, rate in enumerate(rates):
            output[unit, column] = curve.predict(int(rate))
    predict_seconds = time.perf_counter() - start_predict
    return bind_seconds, predict_seconds, output


def _profile(function, path: Path) -> dict[str, Any]:
    profiler = cProfile.Profile()
    started = time.perf_counter()
    result = profiler.runcall(function)
    elapsed = time.perf_counter() - started
    profiler.dump_stats(str(path))
    stats_path = path.with_suffix(".pstats.txt")
    with stats_path.open("x") as handle:
        stats = pstats.Stats(profiler, stream=handle)
        stats.strip_dirs().sort_stats("cumulative").print_stats(25)
    return {"wall_seconds": elapsed, "profile": path.name, "pstats": stats_path.name,
            "result_shape": list(result.shape)}


def _checksum(array: np.ndarray) -> str:
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise BenchmarkError("benchmark produced non-finite or non-positive predictions")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(arguments.dataset).resolve()
    out = Path(arguments.out).resolve()
    if out.exists():
        raise BenchmarkError(f"output directory already exists: {out}")
    out.mkdir(parents=True)
    start_marker = out / "benchmark_start.timestamp"
    end_marker = out / "benchmark_end.timestamp"
    start_marker.write_text(f"{time.time_ns()}\n")
    try:
        data, manifest, npz_hash = _load_dataset(dataset_root)
        lo_index = int(np.flatnonzero(data["rates"] == RATE_LO)[0])
        hi_index = int(np.flatnonzero(data["rates"] == RATE_HI)[0])
        development = (data["layers"] % 5) != 3
        has_endpoints = np.isfinite(data["values"][:, lo_index]) & np.isfinite(data["values"][:, hi_index])
        eligible = np.flatnonzero(development & has_endpoints)
        requested_units = int(arguments.units)
        if requested_units < 1:
            raise BenchmarkError("--units must be positive")
        selected = eligible[:min(requested_units, eligible.size)]
        if not selected.size:
            raise BenchmarkError("dataset has no eligible development layers")
        left = data["values"][selected, lo_index]
        right = data["values"][selected, hi_index]
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise BenchmarkError("selected units have missing or invalid endpoint measurements")
        if np.any(left <= 0) or np.any(right <= 0):
            raise BenchmarkError("selected endpoint measurements must be positive")

        model = EndpointCurvatureModel(
            mode="log2", rate_lo=RATE_LO, rate_hi=RATE_HI, degree=2,
            means=SYNTHETIC_MEANS, scales=SYNTHETIC_SCALES,
            coefficients=SYNTHETIC_COEFFICIENTS)
        timings: list[dict[str, Any]] = []
        checksums: dict[str, str] = {}
        for repeat in range(1, 4):
            order = ("direct", "bound") if repeat % 2 else ("bound", "direct")
            for arm in order:
                started_unix = time.time()
                started = time.perf_counter()
                if arm == "direct":
                    output = _direct_predict(left, right, RATES, model)
                    total = time.perf_counter() - started
                    record = {"repeat": repeat, "arm": arm, "total_seconds": total,
                              "hot_inference_seconds": total}
                else:
                    bind_seconds, predict_seconds, output = _bound_predict(left, right, RATES, model)
                    total = time.perf_counter() - started
                    record = {"repeat": repeat, "arm": arm, "total_seconds": total,
                              "binding_seconds": bind_seconds,
                              "hot_inference_seconds": predict_seconds}
                checksum = _checksum(output)
                checksums.setdefault(arm, checksum)
                if checksums[arm] != checksum:
                    raise BenchmarkError(f"{arm} checksum changed between repeats")
                record["checksum"] = checksum
                record["start_unix"] = started_unix
                record["end_unix"] = time.time()
                timings.append(record)

        direct = _direct_predict(left, right, RATES, model)
        _, _, bound = _bound_predict(left, right, RATES, model)
        relative = np.abs(direct - bound) / np.abs(bound)
        max_relative = float(np.max(relative))
        if max_relative > 1e-12:
            raise BenchmarkError(f"direct/bound relative mismatch {max_relative:.3e} exceeds 1e-12")
        np.testing.assert_array_equal(direct[:, 0], left)
        np.testing.assert_array_equal(direct[:, -1], right)
        profiles = {
            "direct": _profile(lambda: _direct_predict(left, right, RATES, model), out / "direct.prof"),
            "bound": _profile(lambda: _bound_predict(left, right, RATES, model)[2], out / "bound.prof"),
        }
        receipt = {
            "schema": BENCH_SCHEMA,
            "claim_scope": "CPU numerical evaluation only; not prediction accuracy, GPU, serving, or end-to-end campaign",
            "dataset": {"path": str(dataset_root), "schema": manifest["schema"], "npz_sha256": npz_hash,
                        "manifest_npz_sha256": manifest["npz_sha256"]},
            "workload": {"units_requested": requested_units, "units_eligible": int(eligible.size),
                         "development_units_without_two_endpoints": int((development & ~has_endpoints).sum()),
                         "units_measured": int(selected.size), "development_filter": "layer % 5 != 3",
                         "rates": [int(rate) for rate in RATES], "endpoints": [RATE_LO, RATE_HI],
                         "coefficients": {"source": "deterministic synthetic", "degree": 2,
                                          "mode": "log2", "means": list(SYNTHETIC_MEANS),
                                          "scales": list(SYNTHETIC_SCALES),
                                          "values": list(SYNTHETIC_COEFFICIENTS)}},
            "equivalence": {"max_relative_error": max_relative, "bound_checksum": checksums["bound"],
                            "direct_checksum": checksums["direct"], "endpoint_exact": True,
                            "finite_positive": True, "threshold": 1e-12},
            "timings": timings, "profiles": profiles,
            "environment": {"python": sys.version, "platform": platform.platform(),
                            "implementation": platform.python_implementation(),
                            "numpy": np.__version__},
        }
        with (out / "benchmark.json").open("x") as handle:
            json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        return receipt
    finally:
        end_marker.write_text(f"{time.time_ns()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="validated sparse-rate dataset directory")
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--units", type=int, default=36600, help="maximum eligible development units")
    arguments = parser.parse_args()
    try:
        receipt = run(arguments)
    except (BenchmarkError, OSError, ValueError) as exc:
        parser.error(str(exc))
    else:
        print(json.dumps({"out": str(Path(arguments.out).resolve()),
                          "units": receipt["workload"]["units_measured"],
                          "max_relative_error": receipt["equivalence"]["max_relative_error"]},
                         sort_keys=True))


if __name__ == "__main__":
    main()
