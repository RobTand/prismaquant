"""CPU-only tests for the tier-3 within-Linear row-dispersion pilot."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from prismaquant.row_dispersion import (
    assert_error_decomposition,
    per_row_error,
    split_prize_curve,
    tail_metrics,
)


@pytest.mark.parametrize(
    ("samples", "rows", "columns"),
    [(1, 3, 2), (5, 7, 11), (13, 9, 17), (4, 5, 7)],
)
def test_per_row_error_exactly_decomposes_full_output_error(
    samples, rows, columns
):
    generator = torch.Generator().manual_seed(samples * rows * columns)
    X = torch.randn(samples, columns, generator=generator, dtype=torch.float64)
    W = torch.randn(rows, columns, generator=generator, dtype=torch.float64)
    W_hat = W + 0.07 * torch.randn(
        rows, columns, generator=generator, dtype=torch.float64
    )

    errors = per_row_error(X, W, W_hat)
    full = (X @ (W - W_hat).T).square().sum()

    assert errors.shape == (rows,)
    assert errors.sum().item() == pytest.approx(full.item(), rel=1e-13, abs=1e-14)
    assert_error_decomposition(X, W, W_hat, errors, rtol=1e-13, atol=1e-14)


def test_tail_metrics_uniform_distribution_has_no_dispersion():
    metrics = tail_metrics(torch.full((16,), 3.0))
    assert metrics["p50"] == 3.0
    assert metrics["p90"] == 3.0
    assert metrics["p99"] == 3.0
    assert metrics["ratio_p99_over_p50"] == 1.0
    assert metrics["coefficient_of_variation"] == 0.0
    assert metrics["gini"] == pytest.approx(0.0, abs=1e-15)


def test_tail_metrics_single_spike_has_known_gini():
    values = torch.tensor([0.0] * 7 + [8.0])
    metrics = tail_metrics(values)
    assert metrics["mean"] == 1.0
    assert metrics["max"] == 8.0
    assert metrics["gini"] == pytest.approx(7 / 8, abs=1e-15)
    assert metrics["coefficient_of_variation"] == pytest.approx(7 ** 0.5)
    assert metrics["ratio_p99_over_p50"] == float("inf")


def test_tail_metrics_hand_computable_percentile_ratio():
    # Linear interpolation: p50=1 and p99 lies 91% of the way from 1 to 11.
    values = [1.0] * 9 + [11.0]
    metrics = tail_metrics(values)
    assert metrics["p50"] == 1.0
    assert metrics["p90"] == pytest.approx(2.0)
    assert metrics["p99"] == pytest.approx(10.1)
    assert metrics["ratio_p99_over_p50"] == pytest.approx(10.1)


def test_prize_curve_is_monotone_and_accounts_for_bytes_and_endpoints():
    cheap = torch.tensor([10.0, 8.0, 6.0, 4.0])
    expensive = torch.tensor([5.0, 4.0, 3.0, 2.0])
    curve = split_prize_curve(cheap, expensive, 2.0, 5.0, [0, 0.25, 0.5, 1])

    assert curve["rowwise_better"] is True
    assert curve["monotone_non_increasing"] is True
    assert curve["violations"] == []
    assert [point["total_error"] for point in curve["points"]] == [28, 23, 19, 14]
    assert [point["total_bytes"] for point in curve["points"]] == [8, 11, 14, 20]

    references = curve["uniform_references"]
    assert references["cheap_everywhere"]["total_error"] == float(cheap.sum())
    assert references["cheap_everywhere"]["total_bytes"] == 8.0
    assert references["expensive_everywhere"]["total_error"] == float(
        expensive.sum()
    )
    assert references["expensive_everywhere"]["total_bytes"] == 20.0


def test_prize_curve_surfaces_rows_where_expensive_is_worse():
    curve = split_prize_curve(
        [3.0, 2.0, 1.0], [2.0, 4.0, 0.5], 10.0, 20.0, [0, 1 / 3, 1]
    )
    assert curve["rowwise_better"] is False
    assert curve["violations"] == [{
        "row": 1,
        "cheap_error": 2.0,
        "expensive_error": 4.0,
        "excess_error": 2.0,
    }]
    # The endpoint is the measured expensive total, including the regression;
    # no clamp silently replaces it with the cheap value.
    assert curve["uniform_references"]["expensive_everywhere"]["total_error"] == 6.5


def test_cli_smoke_writes_vectors_and_summary(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "row-dispersion"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root))
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "tier3_row_dispersion_pilot.py"),
            "--weights", "synthetic:7x5",
            "--acts", "synthetic:11x5",
            "--quantized",
            "cheap=synthetic:7x5:0.20",
            "expensive=synthetic:7x5:0.05",
            "--bytes-per-row", "cheap=10",
            "--bytes-per-row", "expensive=20",
            "--fractions", "0,0.25,1",
            "--out", str(out),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "format" in completed.stdout
    assert "prize curves: 1" in completed.stdout

    cheap = np.load(out / "cheap_per_row_error.npy")
    expensive = np.load(out / "expensive_per_row_error.npy")
    assert cheap.shape == expensive.shape == (7,)
    assert np.all(expensive <= cheap)

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "prismaquant.row_dispersion.v1"
    assert summary["inputs"]["weights_shape"] == [7, 5]
    assert set(summary["formats"]) == {"cheap", "expensive"}
    assert summary["prize_curves"][0]["curve"]["rowwise_better"] is True
    assert summary["prize_curves"][0]["curve"]["violations"] == []
