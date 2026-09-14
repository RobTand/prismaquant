#!/usr/bin/env python3
"""Measure synthetic scalar adaptive-curve queries on one CPU host.

This benchmark deliberately has no model, GPU, calibration, or accuracy input.
Its monotone synthetic curve exists only to force deterministic adaptive
refinement to fixed anchor counts.  In log2 mode it compares the new shared
curve with ``TesseraRateSurface`` over identical anchors and query coordinates;
the values must agree bit-for-bit before timing is reported.  Value mode is
reported as an unpaired implementation measurement because Tessera's prior
surface is log2-only.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import pstats
import statistics
import struct
import sys
import time
from typing import Callable

from prismaquant.adaptive_anchored_shape import AdaptiveAnchoredCurve
from prismaquant.tessera_rate_surface import TesseraRateSurface


SCHEMA = "prismaquant.adaptive_query_benchmark.v1"
RATE_LO = 832
RATE_HI = 1088
ROSTER = tuple(range(RATE_LO, RATE_HI + 1))
ANCHOR_COUNTS = (2, 5, 17, 65)
WALL_REPEATS = 7
# Seven interleaved samples at this size separate scalar query timing from
# startup on dl380g10; construction and profiling are measured separately.
QUERY_BATCH_REPEATS = 2_800
PROFILE_BATCH_REPEATS = 30
CONSTRUCTION_REPEATS = 25


class BenchmarkError(RuntimeError):
    """The synthetic benchmark's fixed comparison contract was violated."""


def synthetic_value(rate: int) -> float:
    """A positive, strictly decreasing, log-curved deterministic oracle."""
    if rate not in ROSTER:
        raise BenchmarkError("synthetic rate is outside the declared roster")
    offset = (rate - RATE_LO) / (RATE_HI - RATE_LO)
    return math.exp2(30.0 - 4.0 * offset - offset * offset)


def _adaptive(anchor_count: int, mode: str) -> AdaptiveAnchoredCurve:
    curve = AdaptiveAnchoredCurve.start(
        ROSTER,
        {RATE_LO: synthetic_value(RATE_LO), RATE_HI: synthetic_value(RATE_HI)},
        mode=mode,
        relative_tolerance=0.0,
        max_measurements=anchor_count,
    )
    while curve.measurement_count < anchor_count:
        curve, probe = curve.request_next()
        if probe is None:
            raise BenchmarkError(
                f"synthetic acquisition stopped at {curve.measurement_count}, "
                f"not requested {anchor_count} anchors"
            )
        curve = curve.record_measurement(probe.coordinate, synthetic_value(probe.coordinate))
    if curve.measurement_count != anchor_count:
        raise BenchmarkError("adaptive acquisition did not count exact unique anchors")
    return curve


def _tessera(anchor_coordinates: tuple[int, ...]) -> TesseraRateSurface:
    values = tuple(synthetic_value(rate) for rate in anchor_coordinates)
    return TesseraRateSurface(
        unit_name="synthetic.cpu.query",
        family="TESSERA_E2M1_K1",
        layout="tight",
        currency="synthetic_scalar_only",
        anchor_q256=anchor_coordinates,
        anchor_dloss=values,
        anchor_stderr=(0.0,) * len(values),
    )


def _predictions(predict: Callable[[int], float]) -> tuple[float, ...]:
    values = tuple(predict(rate) for rate in ROSTER)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise BenchmarkError("query output was not positive and finite")
    return values


def _checksum(values: tuple[float, ...]) -> str:
    return hashlib.sha256(b"".join(struct.pack("!d", value) for value in values)).hexdigest()


def _query_work(predict: Callable[[int], float], repeats: int) -> float:
    """Do fixed useful scalar query work and retain a dependency on every answer."""
    total = 0.0
    for _ in range(repeats):
        for rate in ROSTER:
            total += predict(rate)
    if not math.isfinite(total) or total <= 0.0:
        raise BenchmarkError("query work did not consume valid predictions")
    return total


def _profile(
    predict: Callable[[int], float],
    path: Path,
) -> dict[str, object]:
    profiler = cProfile.Profile()
    elapsed_start = time.perf_counter_ns()
    total = profiler.runcall(_query_work, predict, PROFILE_BATCH_REPEATS)
    elapsed_ns = time.perf_counter_ns() - elapsed_start
    profiler.dump_stats(str(path))
    text_path = path.with_suffix(".pstats.txt")
    with text_path.open("x") as handle:
        pstats.Stats(profiler, stream=handle).strip_dirs().sort_stats("cumulative").print_stats(30)
    return {
        "profile": path.name,
        "pstats": text_path.name,
        "queries": PROFILE_BATCH_REPEATS * len(ROSTER),
        "wall_seconds": elapsed_ns / 1e9,
        "checksum_dependency": total,
    }


def _quantiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
    }


def _measure_construction(anchor_count: int, mode: str) -> dict[str, object]:
    starts_ns: list[int] = []
    acquisition_ns: list[int] = []
    tessera_ns: list[int] = []
    anchor_digest = None
    for _ in range(CONSTRUCTION_REPEATS):
        start = time.perf_counter_ns()
        curve = AdaptiveAnchoredCurve.start(
            ROSTER,
            {RATE_LO: synthetic_value(RATE_LO), RATE_HI: synthetic_value(RATE_HI)},
            mode=mode,
            relative_tolerance=0.0,
            max_measurements=anchor_count,
        )
        starts_ns.append(time.perf_counter_ns() - start)
        start = time.perf_counter_ns()
        while curve.measurement_count < anchor_count:
            curve, probe = curve.request_next()
            if probe is None:
                raise BenchmarkError("synthetic construction acquisition stopped early")
            curve = curve.record_measurement(probe.coordinate, synthetic_value(probe.coordinate))
        acquisition_ns.append(time.perf_counter_ns() - start)
        anchors = tuple(sorted(curve.measurements))
        digest = hashlib.sha256(",".join(map(str, anchors)).encode()).hexdigest()
        if anchor_digest is None:
            anchor_digest = digest
        elif anchor_digest != digest:
            raise BenchmarkError("adaptive anchor selection was not deterministic")
        if mode == "log2":
            start = time.perf_counter_ns()
            _tessera(anchors)
            tessera_ns.append(time.perf_counter_ns() - start)
    return {
        "samples": CONSTRUCTION_REPEATS,
        "anchor_roster_sha256": anchor_digest,
        "adaptive_start_us": {key: value / 1e3 for key, value in _quantiles(starts_ns).items()},
        "adaptive_acquisition_us": {
            key: value / 1e3 for key, value in _quantiles(acquisition_ns).items()
        },
        "tessera_construction_us": (
            {key: value / 1e3 for key, value in _quantiles(tessera_ns).items()}
            if tessera_ns else None
        ),
    }


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(output: Path) -> dict[str, object]:
    if output.exists():
        raise BenchmarkError(f"output already exists: {output}")
    output.mkdir(parents=True)
    profiles = output / "profiles"
    profiles.mkdir()
    host = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )},
        "package_versions": {
            "tessera": _version("tessera"),
            "torch": _version("torch"),
        },
    }
    report: dict[str, object] = {
        "schema": SCHEMA,
        "scope": (
            "Synthetic scalar CPU query/acquisition measurement only; it makes no "
            "GLM, accuracy, GPU, serving, cache, or production-performance claim."
        ),
        "roster": list(ROSTER),
        "anchor_counts": list(ANCHOR_COUNTS),
        "wall_repeats": WALL_REPEATS,
        "query_batch_repeats": QUERY_BATCH_REPEATS,
        "queries_per_wall_sample": QUERY_BATCH_REPEATS * len(ROSTER),
        "construction_repeats": CONSTRUCTION_REPEATS,
        "profile_batch_repeats": PROFILE_BATCH_REPEATS,
        "host": host,
        "arms": [],
    }
    for anchor_count in ANCHOR_COUNTS:
        curves = {mode: _adaptive(anchor_count, mode) for mode in ("log2", "value")}
        anchors = tuple(sorted(curves["log2"].measurements))
        tessera = _tessera(anchors)
        adaptive_log_values = _predictions(curves["log2"].predict)
        tessera_values = _predictions(tessera.predict)
        if adaptive_log_values != tessera_values:
            raise BenchmarkError("log2 adaptive and Tessera predictions differ on identical anchors")
        log_checksum = _checksum(adaptive_log_values)
        if log_checksum != _checksum(tessera_values):
            raise BenchmarkError("log2 checksum mismatch")
        value_checksum = _checksum(_predictions(curves["value"].predict))
        predict_arms: list[tuple[str, Callable[[int], float], str, bool]] = [
            ("adaptive_log2", curves["log2"].predict, log_checksum, True),
            ("tessera_log2", tessera.predict, log_checksum, True),
            ("adaptive_value", curves["value"].predict, value_checksum, False),
        ]
        samples: dict[str, list[float]] = {name: [] for name, _, _, _ in predict_arms}
        for repeat in range(WALL_REPEATS):
            offset = repeat % len(predict_arms)
            for name, predict, checksum, paired in predict_arms[offset:] + predict_arms[:offset]:
                started = time.perf_counter_ns()
                total = _query_work(predict, QUERY_BATCH_REPEATS)
                elapsed = (time.perf_counter_ns() - started) / 1e9
                if not math.isfinite(total):
                    raise BenchmarkError("timed query work failed")
                samples[name].append(elapsed)
        construction = {
            "adaptive_log2": _measure_construction(anchor_count, "log2"),
            "adaptive_value": _measure_construction(anchor_count, "value"),
        }
        arm_records = []
        for name, predict, checksum, paired in predict_arms:
            profile = _profile(predict, profiles / f"{anchor_count}-{name}.pstats")
            seconds = _quantiles(samples[name])
            arm_records.append({
                "name": name,
                "paired_log2_comparator": paired,
                "checksum": checksum,
                "wall_seconds": seconds,
                "microseconds_per_query": {
                    key: value * 1e6 / (QUERY_BATCH_REPEATS * len(ROSTER))
                    for key, value in seconds.items()
                },
                "profile": profile,
            })
        report["arms"].append({
            "anchor_count": anchor_count,
            "anchor_coordinates": list(anchors),
            "log2_prediction_checksum": log_checksum,
            "value_prediction_checksum": value_checksum,
            "construction": construction,
            "wall": arm_records,
            "comparison_note": (
                "Only adaptive_log2 and tessera_log2 are a matched comparator pair. "
                "adaptive_value has no Tessera value-PWL baseline."
            ),
        })
    report_path = output / "report.json"
    with report_path.open("x") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(arguments.output.resolve())
    print(json.dumps({
        "output": str(arguments.output),
        "schema": report["schema"],
        "anchor_counts": report["anchor_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
