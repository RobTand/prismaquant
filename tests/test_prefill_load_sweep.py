"""The sweep driver's CPU-only halves: its intervals, its window and its table.

The measurement itself needs a GB10 and the pinned image; these are the parts a
reader has to trust when reading the table, so they are checked here.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load("pq_prefill_load_sweep", "experiments/pq_prefill_load_sweep.py")
runner = _load("pq_frontier_native_runner", "experiments/pq_frontier_native_runner.py")


def test_bootstrap_interval_covers_the_median_and_is_reproducible():
    samples = [0.1276, 0.1271, 0.1284, 0.1279, 0.1273, 0.1288, 0.1275, 0.1281]
    first = sweep.bootstrap_median(samples)
    assert first == sweep.bootstrap_median(samples)
    assert first["low_ms"] <= first["median_ms"] <= first["high_ms"]
    assert first["samples"] == len(samples)
    assert first["min_ms"] == min(samples) and first["max_ms"] == max(samples)


def test_bootstrap_refuses_a_row_that_cannot_carry_an_interval():
    with pytest.raises(ValueError):
        sweep.bootstrap_median([1.0, 2.0])


def test_window_discards_the_meter_lag_and_reports_the_envelope_fraction():
    series = sweep.PowerSeries()
    series.samples = [{"t": float(i) / 10.0, "watts": 4.0 if i < 15 else 70.0,
                       "sm_mhz": 1500, "temperature_c": 60} for i in range(100)]
    window = series.window(0.0, 10.0, settle_s=1.5)
    assert window["status"] == "observed"
    # Every retained sample is past the settle boundary, so the idle prefix is
    # not averaged into the loaded window.
    assert window["mean_w"] == pytest.approx(70.0)
    assert window["envelope_fraction"] == pytest.approx(70.0 / sweep.ENVELOPE_W)
    assert window["envelope_w"] == 140.0


def test_window_reports_absence_rather_than_inventing_a_power_number():
    assert sweep.PowerSeries().window(0.0, 1.0, settle_s=0.1)["status"] == "no_samples"


def test_table_rows_are_grouped_by_m_and_carry_the_power_fraction():
    report = {
        "rows": [512, 1024],
        "cells": {
            "a": {"unit": "model.layers.0.mlp.gate_proj", "format": "TESSERA_E2M1_K2_R896",
                  "points": {"512": {"timing": {"median_ms": 0.13, "low_ms": 0.12, "high_ms": 0.14},
                                     "sustained": {"mean_w": 70.0, "envelope_fraction": 0.5,
                                                   "wall_ms_per_apply": 0.15,
                                                   "achieved_tflop_s": 24.0, "gflop_per_joule": 342.0}}}},
            "b": {"unit": "model.layers.0.mlp.gate_proj", "format": "TESSERA_E4M3_K1_R1006",
                  "points": {"512": {"timing": {"median_ms": 0.12, "low_ms": 0.11, "high_ms": 0.13},
                                     "sustained": {"status": "no_samples"}}}},
        },
    }
    table = sweep.render_table(report)
    lines = table.splitlines()
    assert lines[0].startswith("| M | unit | format |")
    assert len(lines) == 4                       # header, rule, two rows; M=1024 has no points
    assert "TESSERA_E2M1_K2_R896" in lines[2] and "0.500" in lines[2] and "342.00" in lines[2]
    # A row with no power samples prints a gap, never a fabricated watt figure.
    assert lines[3].count("| - |") >= 3


def test_runner_defaults_to_tessera_harness_and_accepts_a_consumer_module():
    assert runner.DEFAULT_BENCH_MODULE == "experiments.bench_native_operator"
    source = (ROOT / "experiments/pq_frontier_native_runner.py").read_text()
    assert "runpy.run_module(args.module" in source
    assert '"--module", args.module' in source


def test_sweep_refuses_row_counts_that_are_not_whole_tiles():
    with pytest.raises(SystemExit):
        sweep.main(["--cells-root", ".", "--cells", "x", "--rows", "768", "--out", "/dev/null"])
    with pytest.raises(SystemExit):
        sweep.main(["--cells-root", ".", "--cells", "x", "--rows", "256", "--out", "/dev/null"])
