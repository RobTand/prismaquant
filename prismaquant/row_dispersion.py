"""Per-output-row quantization-error dispersion measurements.

This module is research measurement tooling for the tier-3 row-splitting
pilot.  It deliberately has no cache, model-loading, or serving dependency:
callers provide one activation matrix and two versions of one Linear weight.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch


def _validated_inputs(
    X: torch.Tensor,
    W: torch.Tensor,
    W_hat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.as_tensor(X)
    W = torch.as_tensor(W)
    W_hat = torch.as_tensor(W_hat)
    if X.ndim != 2 or W.ndim != 2 or W_hat.ndim != 2:
        raise ValueError(
            "X, W, and W_hat must be 2-D; got "
            f"{tuple(X.shape)}, {tuple(W.shape)}, {tuple(W_hat.shape)}"
        )
    if W.shape != W_hat.shape:
        raise ValueError(
            f"W and W_hat shapes differ: {tuple(W.shape)} != "
            f"{tuple(W_hat.shape)}"
        )
    if X.shape[1] != W.shape[1]:
        raise ValueError(
            f"activation width {X.shape[1]} != weight width {W.shape[1]}"
        )
    if not (X.device == W.device == W_hat.device):
        raise ValueError(
            "X, W, and W_hat must be on the same device; got "
            f"{X.device}, {W.device}, {W_hat.device}"
        )
    dtype = torch.promote_types(X.dtype, W.dtype)
    dtype = torch.promote_types(dtype, W_hat.dtype)
    if not dtype.is_floating_point:
        raise TypeError("row-dispersion inputs must have floating-point dtype")
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    return X.to(dtype=dtype), W.to(dtype=dtype), W_hat.to(dtype=dtype)


def per_row_error(
    X: torch.Tensor,
    W: torch.Tensor,
    W_hat: torch.Tensor,
) -> torch.Tensor:
    """Measure activation-output SSE independently for every output row.

    For ``X`` shaped ``[samples, in_features]`` and weights shaped
    ``[out_features, in_features]``, the returned vector is

    ``e[r] = ||X @ (W[r] - W_hat[r])||_2^2``.

    Accumulation is float64 even when the matmul is float32.  Consequently,
    summing the vector is the exact row decomposition (up to floating-point
    reduction order) of ``||X @ (W - W_hat).T||_F^2``.
    """
    X, W, W_hat = _validated_inputs(X, W, W_hat)
    output_error = X @ (W - W_hat).T
    return output_error.square().sum(dim=0, dtype=torch.float64)


def assert_error_decomposition(
    X: torch.Tensor,
    W: torch.Tensor,
    W_hat: torch.Tensor,
    e: torch.Tensor | None = None,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    """Assert that per-row errors sum to the full-matrix output error."""
    X, W, W_hat = _validated_inputs(X, W, W_hat)
    if e is None:
        e = per_row_error(X, W, W_hat)
    else:
        e = torch.as_tensor(e, device=X.device, dtype=torch.float64)
    if e.ndim != 1 or e.numel() != W.shape[0]:
        raise ValueError(
            f"e must have shape ({W.shape[0]},), got {tuple(e.shape)}"
        )
    full = (X @ (W - W_hat).T).square().sum(dtype=torch.float64)
    torch.testing.assert_close(e.sum(), full, rtol=rtol, atol=atol)


def _error_vector(e: torch.Tensor | Iterable[float], name: str) -> torch.Tensor:
    values = torch.as_tensor(e, dtype=torch.float64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError(f"{name} must be a non-empty 1-D vector")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values")
    if bool((values < 0).any()):
        raise ValueError(f"{name} must be non-negative")
    return values


def tail_metrics(e: torch.Tensor | Iterable[float]) -> dict[str, float]:
    """Summarize the location and unevenness of a non-negative error vector.

    Percentiles use linear interpolation.  The coefficient of variation is
    population standard deviation divided by the mean.  For sorted values
    ``x_(i)`` with one-based rank ``i``, Gini is computed by the cumulative
    closed form

    ``G = 2 * sum(i * x_(i)) / (n * sum(x)) - (n + 1) / n``.

    The all-zero vector is defined to have CV and Gini zero and p99/p50 one.
    If p50 is zero but p99 is positive, the percentile ratio is infinite.
    """
    values = _error_vector(e, "e")
    quantiles = torch.quantile(
        values, torch.tensor([0.50, 0.90, 0.99], dtype=torch.float64)
    )
    p50, p90, p99 = (float(value) for value in quantiles)
    mean = float(values.mean())
    maximum = float(values.max())
    if p50 == 0.0:
        ratio = 1.0 if p99 == 0.0 else float("inf")
    else:
        ratio = p99 / p50
    coefficient_of_variation = (
        0.0 if mean == 0.0 else float(values.std(unbiased=False)) / mean
    )

    total = float(values.sum())
    if total == 0.0:
        gini = 0.0
    else:
        sorted_values = values.sort().values
        ranks = torch.arange(
            1, values.numel() + 1, dtype=torch.float64, device=values.device
        )
        n = values.numel()
        gini = float(
            2.0 * (ranks * sorted_values).sum() / (n * sorted_values.sum())
            - (n + 1.0) / n
        )
        # Roundoff can put a mathematically bounded statistic a few ulps out.
        gini = min(1.0, max(0.0, gini))

    return {
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "max": maximum,
        "mean": mean,
        "ratio_p99_over_p50": ratio,
        "coefficient_of_variation": coefficient_of_variation,
        "gini": gini,
    }


def _split_point(
    *,
    fraction: float,
    routed_rows: int,
    row_count: int,
    total_error: float,
    byte_cheap_per_row: float,
    byte_expensive_per_row: float,
) -> dict[str, float | int]:
    return {
        "fraction": float(fraction),
        "routed_rows": int(routed_rows),
        "realized_fraction": float(routed_rows / row_count),
        "total_error": float(total_error),
        "total_bytes": float(
            (row_count - routed_rows) * byte_cheap_per_row
            + routed_rows * byte_expensive_per_row
        ),
    }


def split_prize_curve(
    e_cheap: torch.Tensor | Iterable[float],
    e_expensive: torch.Tensor | Iterable[float],
    byte_cheap_per_row: float,
    byte_expensive_per_row: float,
    fractions: Iterable[float],
) -> dict[str, Any]:
    """Compute the analytic error/byte curve for a two-format row split.

    For each requested fraction, ``ceil(f * n_rows)`` rows with the largest
    cheap-format errors are routed to the expensive format.  This convention
    makes a non-zero pilot fraction inspect at least one row.  The selected
    sets are nested as the fraction grows.

    Rows where ``e_expensive > e_cheap`` are returned under ``violations``
    with both measured values and their excess.  They are never clamped or
    hidden.  When there are no such rows, an internal assertion guards the
    mathematical invariant that total error cannot increase with routed-row
    count; a failed assertion then indicates an implementation/numeric defect.
    """
    cheap = _error_vector(e_cheap, "e_cheap")
    expensive = _error_vector(e_expensive, "e_expensive")
    if cheap.shape != expensive.shape:
        raise ValueError(
            f"error-vector shapes differ: {tuple(cheap.shape)} != "
            f"{tuple(expensive.shape)}"
        )
    byte_cheap_per_row = float(byte_cheap_per_row)
    byte_expensive_per_row = float(byte_expensive_per_row)
    for name, value in (
        ("byte_cheap_per_row", byte_cheap_per_row),
        ("byte_expensive_per_row", byte_expensive_per_row),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    requested_fractions = [float(fraction) for fraction in fractions]
    for fraction in requested_fractions:
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fractions must lie in [0, 1], got {fraction}")

    row_count = cheap.numel()
    order = torch.argsort(cheap, descending=True, stable=True)
    cheap_total = float(cheap.sum())
    expensive_total = float(expensive.sum())

    def total_for_count(routed_rows: int) -> float:
        if routed_rows == 0:
            return cheap_total
        if routed_rows == row_count:
            return expensive_total
        top = order[:routed_rows]
        bottom = order[routed_rows:]
        return float(expensive[top].sum() + cheap[bottom].sum())

    points: list[dict[str, float | int]] = []
    for fraction in requested_fractions:
        routed_rows = min(row_count, int(math.ceil(fraction * row_count)))
        points.append(_split_point(
            fraction=fraction,
            routed_rows=routed_rows,
            row_count=row_count,
            total_error=total_for_count(routed_rows),
            byte_cheap_per_row=byte_cheap_per_row,
            byte_expensive_per_row=byte_expensive_per_row,
        ))

    uniform_cheap = _split_point(
        fraction=0.0,
        routed_rows=0,
        row_count=row_count,
        total_error=cheap_total,
        byte_cheap_per_row=byte_cheap_per_row,
        byte_expensive_per_row=byte_expensive_per_row,
    )
    uniform_expensive = _split_point(
        fraction=1.0,
        routed_rows=row_count,
        row_count=row_count,
        total_error=expensive_total,
        byte_cheap_per_row=byte_cheap_per_row,
        byte_expensive_per_row=byte_expensive_per_row,
    )

    violation_indices = torch.nonzero(expensive > cheap).flatten().tolist()
    violations = [
        {
            "row": int(index),
            "cheap_error": float(cheap[index]),
            "expensive_error": float(expensive[index]),
            "excess_error": float(expensive[index] - cheap[index]),
        }
        for index in violation_indices
    ]

    counts = sorted({0, row_count, *(int(p["routed_rows"]) for p in points)})
    totals = [total_for_count(count) for count in counts]
    scale = max(1.0, *(abs(total) for total in totals))
    tolerance = 1e-12 * scale
    monotone_non_increasing = all(
        right <= left + tolerance for left, right in zip(totals, totals[1:])
    )
    if not violations:
        assert monotone_non_increasing, (
            "row-wise-better expensive errors produced a non-monotone split "
            "curve"
        )

    return {
        "row_count": row_count,
        "selection": "descending_e_cheap",
        "rounding": "ceil_fraction_times_rows",
        "points": points,
        "uniform_references": {
            "cheap_everywhere": uniform_cheap,
            "expensive_everywhere": uniform_expensive,
        },
        "violations": violations,
        "rowwise_better": not violations,
        "monotone_non_increasing": monotone_non_increasing,
    }

