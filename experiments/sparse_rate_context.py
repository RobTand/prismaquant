#!/usr/bin/env python3
"""Cheap contextual two-anchor surfaces for sparse-rate model development.

This helper operates only in the saved dataset's scalar output-MSE currency.
It neither identifies activation/mixed components nor emits allocator prices.
Callers own the full-layer development/final split.  Only target endpoints are
read; target interior measurements never enter features or fitting.

The fitted response is the log2 residual from the endpoint chord.  Every
design column is multiplied by ``t * (1 - t)``, so predictions preserve both
real anchors exactly while one pooled fit supports arbitrary interior rates.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


RLO, RHI = 832, 1088
CONTEXT_MODELS = (
    "context_self",
    "context_siblings",
    "context_siblings_tail",
)
_STRUCTURES = ("gate_proj", "up_proj", "down_proj")


def _as_array(data: Mapping[str, np.ndarray], name: str, length: int) -> np.ndarray:
    if name not in data:
        raise ValueError(f"context model requires dataset field {name!r}")
    value = np.asarray(data[name])
    if value.ndim != 1 or len(value) != length:
        raise ValueError(f"dataset field {name!r} has the wrong shape")
    return value


def _endpoint_indices(rates: np.ndarray) -> tuple[int, int]:
    rates = np.asarray(rates)
    if rates.ndim != 1 or len(set(map(int, rates))) != len(rates):
        raise ValueError("rates must be a one-dimensional unique grid")
    found = []
    for rate in (RLO, RHI):
        indices = np.flatnonzero(rates == rate)
        if len(indices) != 1:
            raise ValueError(f"rate grid lacks unique endpoint {rate}")
        found.append(int(indices[0]))
    return found[0], found[1]


def _segment_is_homogeneous(data: Mapping[str, np.ndarray], active: np.ndarray) -> None:
    """Refuse accidental transfer across the existing exact segment keys."""
    for field in ("families", "activation_contracts", "roles", "structures",
                  "rows", "cols"):
        values = _as_array(data, field, len(active))[active]
        if len(set(values.tolist())) > 1:
            raise ValueError(f"context segment crosses dataset field {field!r}")


def _qname_stem(qname: str) -> str | None:
    if not isinstance(qname, str) or "." not in qname:
        return None
    stem, structure = qname.rsplit(".", 1)
    return stem if structure in _STRUCTURES else None


def _sibling_features(
    data: Mapping[str, np.ndarray],
    values: np.ndarray,
    lo: int,
    hi: int,
) -> np.ndarray:
    """Return gate/up endpoint summaries without reading any interior value."""
    n = len(values)
    qnames = _as_array(data, "qnames", n)
    families = _as_array(data, "families", n)
    contracts = _as_array(data, "activation_contracts", n)
    roles = _as_array(data, "roles", n)
    structures = _as_array(data, "structures", n)
    lookup: dict[tuple[object, ...], int] = {}
    for index in range(n):
        stem = _qname_stem(str(qnames[index]))
        key = (stem, families[index], contracts[index], roles[index], structures[index])
        if stem is None:
            continue
        if key in lookup:
            raise ValueError("duplicate contextual sibling identity")
        lookup[key] = index

    result = np.full((n, 4), np.nan, dtype=np.float64)
    for index in range(n):
        stem = _qname_stem(str(qnames[index]))
        if stem is None:
            continue
        common = (stem, families[index], contracts[index], roles[index])
        sibling_rows = [lookup.get((*common, structure))
                        for structure in ("gate_proj", "up_proj")]
        if any(row is None for row in sibling_rows):
            continue
        endpoints = values[np.asarray(sibling_rows, dtype=np.int64)][:, (lo, hi)]
        if not np.isfinite(endpoints).all() or not (endpoints > 0).all():
            continue
        logs = np.log2(endpoints)
        levels = logs.mean(axis=1)
        slopes = logs[:, 1] - logs[:, 0]
        result[index] = (
            levels.mean(), slopes.mean(), levels[0] - levels[1],
            slopes[0] - slopes[1],
        )
    return result


def _context_features(
    model: str,
    data: Mapping[str, np.ndarray],
    values: np.ndarray,
    lo: int,
    hi: int,
) -> tuple[np.ndarray, int, int | None]:
    n = len(values)
    left, right = values[:, lo], values[:, hi]
    valid = np.isfinite(left) & np.isfinite(right) & (left > 0) & (right > 0)
    logleft = np.full(n, np.nan)
    logright = np.full(n, np.nan)
    logleft[valid], logright[valid] = np.log2(left[valid]), np.log2(right[valid])
    layers = _as_array(data, "layers", n).astype(np.float64)
    counts = _as_array(data, "counts", n).astype(np.float64)
    if (counts < 0).any():
        raise ValueError("token support must be nonnegative")
    # Endpoint level/slope, known token support, and a smooth layer-position
    # trend.  The quadratic term is still a fixed low-degree atlas feature.
    columns = [
        (logleft + logright) / 2,
        logright - logleft,
        np.log1p(counts),
        layers,
        layers * layers,
    ]
    slope_column = 1
    sibling_slope_column = None
    if model != "context_self":
        siblings = _sibling_features(data, values, lo, hi)
        columns.extend(siblings[:, index] for index in range(siblings.shape[1]))
        # A missing sibling is explicit and will be imputed from training only.
        columns.extend((~np.isfinite(siblings[:, index])).astype(np.float64)
                       for index in range(siblings.shape[1]))
        sibling_slope_column = 6
    return np.column_stack(columns), slope_column, sibling_slope_column


def _prepare_context(
    features: np.ndarray,
    fitting_units: np.ndarray,
) -> np.ndarray:
    if not fitting_units.any():
        raise ValueError("context fit has no endpoint-complete training units")
    train = features[fitting_units]
    medians = np.zeros(features.shape[1], dtype=np.float64)
    for column in range(features.shape[1]):
        finite = train[np.isfinite(train[:, column]), column]
        medians[column] = np.median(finite) if len(finite) else 0.0
    filled = np.where(np.isfinite(features), features, medians)
    train = filled[fitting_units]
    means, scales = train.mean(axis=0), train.std(axis=0)
    scales[scales == 0] = 1.0
    standardized = (filled - means) / scales
    # Context extrapolation is not evidence.  Bound target coordinates to the
    # fitting envelope while still allowing interpolation along the rate axis.
    lower = standardized[fitting_units].min(axis=0)
    upper = standardized[fitting_units].max(axis=0)
    return np.clip(standardized, lower, upper)


def _surface_design(
    context: np.ndarray,
    t: float,
    *,
    slope_column: int,
    sibling_slope_column: int | None,
) -> np.ndarray:
    envelope = t * (1.0 - t)
    u = 2.0 * t - 1.0
    columns = [np.ones(len(context)), *[context[:, j] for j in range(context.shape[1])],
               np.full(len(context), u), u * context[:, slope_column]]
    if sibling_slope_column is not None:
        columns.append(u * context[:, sibling_slope_column])
    return envelope * np.column_stack(columns)


def _fit_ridge(
    design: np.ndarray,
    response: np.ndarray,
    *,
    tail: bool,
) -> np.ndarray | None:
    if len(response) < max(24, 2 * design.shape[1]):
        return None
    penalty = np.eye(design.shape[1]) * (len(response) * 1e-5)
    weights = np.ones(len(response), dtype=np.float64)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    iterations = 4 if tail else 1
    for _ in range(iterations):
        weighted = design * weights[:, None]
        coefficients = np.linalg.solve(design.T @ weighted + penalty,
                                       weighted.T @ response)
        if not tail:
            break
        residual = response - design @ coefficients
        center = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - center))
        if not np.isfinite(scale) or scale <= 0:
            break
        # Approximate an L4 tail objective, but cap influence so one corrupt
        # observation cannot determine the pooled surface.
        weights = 1.0 + np.minimum((np.abs(residual) / (3.0 * scale)) ** 2, 15.0)
    return coefficients


def predict_context_surface(
    model: str,
    data: Mapping[str, np.ndarray],
    rates: np.ndarray,
    train: np.ndarray,
    target: np.ndarray,
    segment: np.ndarray,
) -> np.ndarray:
    """Fit on ``train & segment`` and predict ``target & segment``.

    ``data`` is the complete sparse-rate dataset mapping.  Passing it whole is
    required because a down projection's gate/up siblings are separate rows.
    ``train``, ``target`` and ``segment`` are one-dimensional boolean masks.
    The returned array has the same shape as ``data['values']`` and is NaN
    outside eligible target rows and the closed endpoint interval.
    """
    if model not in CONTEXT_MODELS:
        raise ValueError(f"unknown context model {model!r}")
    values = np.asarray(data.get("values"), dtype=np.float64)
    rates = np.asarray(rates)
    if values.ndim != 2 or values.shape[1] != len(rates):
        raise ValueError("values/rates shape mismatch")
    n = len(values)
    masks = []
    for name, raw in (("train", train), ("target", target), ("segment", segment)):
        mask = np.asarray(raw)
        if mask.dtype != np.bool_ or mask.shape != (n,):
            raise ValueError(f"{name} must be a boolean unit mask")
        masks.append(mask)
    train, target, segment = masks
    if (train & target).any():
        raise ValueError("context train and target masks overlap")
    active = (train | target) & segment
    if not active.any():
        return np.full_like(values, np.nan)
    _segment_is_homogeneous(data, active)
    lo, hi = _endpoint_indices(rates)
    endpoints = (np.isfinite(values[:, lo]) & np.isfinite(values[:, hi])
                 & (values[:, lo] > 0) & (values[:, hi] > 0))
    fitting_units = train & segment & endpoints
    target_units = target & segment & endpoints
    output = np.full_like(values, np.nan)
    if not target_units.any():
        return output
    output[target_units, lo] = values[target_units, lo]
    output[target_units, hi] = values[target_units, hi]

    features, slope_column, sibling_slope_column = _context_features(
        model, data, values, lo, hi,
    )
    context = _prepare_context(features, fitting_units)
    designs, responses, curvatures = [], [], []
    for column, rate in enumerate(rates):
        t = (int(rate) - RLO) / (RHI - RLO)
        if not 0.0 < t < 1.0:
            continue
        measured = fitting_units & np.isfinite(values[:, column]) & (values[:, column] > 0)
        if not measured.any():
            continue
        ids = np.flatnonzero(measured)
        logleft, logright = np.log2(values[ids, lo]), np.log2(values[ids, hi])
        residual = np.log2(values[ids, column]) - ((1.0 - t) * logleft + t * logright)
        designs.append(_surface_design(context[ids], t, slope_column=slope_column,
                                       sibling_slope_column=sibling_slope_column))
        responses.append(residual)
        curvatures.append(residual / (t * (1.0 - t)))
    if not designs:
        return output
    design, response = np.concatenate(designs), np.concatenate(responses)
    coefficients = _fit_ridge(design, response, tail=model.endswith("_tail"))
    if coefficients is None:
        return output
    # Prevent feature extrapolation from inventing curvature outside the
    # actual training-layer envelope.  These are extrema, not target-tuned
    # quantiles, so rare development behavior remains represented.
    curvature = np.concatenate(curvatures)
    curvature_low, curvature_high = float(curvature.min()), float(curvature.max())
    ids = np.flatnonzero(target_units)
    logleft, logright = np.log2(values[ids, lo]), np.log2(values[ids, hi])
    for column, rate in enumerate(rates):
        t = (int(rate) - RLO) / (RHI - RLO)
        if not 0.0 < t < 1.0:
            continue
        design_at_rate = _surface_design(
            context[ids], t, slope_column=slope_column,
            sibling_slope_column=sibling_slope_column,
        )
        residual = design_at_rate @ coefficients
        q = np.clip(residual / (t * (1.0 - t)), curvature_low, curvature_high)
        predicted_log2 = (1.0 - t) * logleft + t * logright + t * (1.0 - t) * q
        prediction = np.exp2(predicted_log2)
        valid = np.isfinite(prediction) & (prediction > 0)
        output[ids[valid], column] = prediction[valid]
    return output


__all__ = ["CONTEXT_MODELS", "RLO", "RHI", "predict_context_surface"]
