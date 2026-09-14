"""Numerical contracts for the development-only Gridbook transfer replay."""
from __future__ import annotations

import numpy as np

from experiments.sparse_rate_gridbook_transfer import (
    RANCHOR, RHI, RLO, anchored_prediction, floor_mixture, log_chord,
    rate_features,
)
from prismaquant.anchored_shape import (
    AnchoredShapeError, fit_anchor_correction, fit_centered_log_shape,
    LogShapeObservation, predict_anchored,
)


RATES = np.asarray([RLO, 864, 896, RANCHOR, 1024, RHI], dtype=np.int64)


def _curves(n=12):
    values = np.empty((n, len(RATES)))
    for unit in range(n):
        left, right = 2.0 ** (-6 - unit / 10), 2.0 ** (-9 - unit / 12)
        for column, rate in enumerate(RATES):
            t = (rate - RLO) / (RHI - RLO)
            values[unit, column] = 2.0 ** ((1 - t) * np.log2(left) + t * np.log2(right))
    return values


def test_anchored_gridbook_shape_keeps_exact_endpoints():
    values = _curves()
    observations = [LogShapeObservation(str(unit), str(int(rate)), float(values[unit, column]))
                    for unit in range(10) for column, rate in enumerate(RATES)]
    shape = fit_centered_log_shape(observations, rate_features(RATES, gridbook=True))
    predicted = anchored_prediction(shape, rate_features(RATES, gridbook=True), values, RATES,
                                    np.arange(len(values)) >= 10, (RLO, RHI))
    np.testing.assert_array_equal(predicted[10:, 0], values[10:, 0])
    np.testing.assert_array_equal(predicted[10:, -1], values[10:, -1])


def test_rate_only_two_anchor_shape_equals_log_chord():
    values = _curves()
    observations = [LogShapeObservation(str(unit), str(int(rate)), float(values[unit, column]))
                    for unit in range(10) for column, rate in enumerate(RATES)]
    shape = fit_centered_log_shape(observations, rate_features(RATES, gridbook=False))
    target = np.arange(len(values)) >= 10
    anchored = anchored_prediction(shape, rate_features(RATES, gridbook=False), values, RATES,
                                  target, (RLO, RHI))
    np.testing.assert_allclose(anchored[target], log_chord(values, RATES, target)[target], rtol=2e-14, atol=0)


def test_floor_mixture_uses_integer_neighbor_coordinate_and_exact_anchors():
    rates = np.asarray([832, 896, 960, 1024, 1088])
    beta, floor, coefficient = 2, .01, 7.0
    def q(rate):
        k, f = divmod(rate / 256, 1)
        return (1 - f) * 2 ** (-beta * int(k)) + f * 2 ** (-beta * (int(k) + 1))
    values = np.asarray([[floor + coefficient * q(rate) for rate in rates]])
    predicted, detail, strict_valid = floor_mixture(values, rates, np.asarray([True]), beta, np.asarray([True]))
    np.testing.assert_allclose(predicted, values, rtol=2e-15, atol=0)
    assert detail["negative_floor_count"] == 0
    assert strict_valid[0]


def test_floor_mixture_reports_negative_floor_without_clamping_anchor():
    values = np.asarray([[.3, .2, .15, .1, .05]])
    rates = np.asarray([832, 896, 960, 1024, 1088])
    predicted, detail, strict_valid = floor_mixture(values, rates, np.asarray([True]), 1, np.asarray([True]))
    assert detail["negative_floor_count"] == 1
    assert not detail["strict_nonnegative_floor_valid"]
    assert not strict_valid[0]
    assert predicted[0, 0] == values[0, 0] and predicted[0, -1] == values[0, -1]


def test_anchored_shape_refuses_prediction_outside_declared_rate_domain():
    values = _curves()
    observations = [LogShapeObservation(str(unit), str(int(rate)), float(values[unit, column]))
                    for unit in range(10) for column, rate in enumerate(RATES)]
    features = rate_features(RATES, gridbook=True)
    shape = fit_centered_log_shape(observations, features)
    correction = fit_anchor_correction(shape, {str(RLO): values[10, 0], str(RHI): values[10, -1]},
                                       {str(int(rate)): float(rate) for rate in RATES})
    with np.testing.assert_raises(AnchoredShapeError):
        predict_anchored(correction, "outside_declared_domain")
