"""Pure numerical contracts for the immutable endpoint-curvature abstraction."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import sys

import numpy as np
import pytest

from experiments.sparse_rate_models import robust_fit
from prismaquant.anchored_shape import (
    AnchoredShapeError, EndpointCurvatureModel, LogShapeObservation,
    fit_endpoint_curvature,
)


def _basis(features, degree):
    columns = [1.0, *features]
    if degree == 2:
        columns.extend(features[left] * features[right]
                       for left in range(len(features))
                       for right in range(left, len(features)))
    return columns


def _linear_model(*, mode="value", coefficients=(.4, 0.0, 0.0, 0.0)):
    return EndpointCurvatureModel(mode, 832, 1088, 1, (0.0, 0.0, 0.0),
                                  (1.0, 1.0, 1.0), coefficients)


def test_value_curve_matches_known_positive_floor_curve_and_keeps_anchors_exact():
    curve = _linear_model().bind(3.0, 1.5)  # Both endpoints retain a positive floor.
    for rate in (832, 896, 960, 1024, 1088):
        t = (rate - 832) / (1088 - 832)
        expected = (1 - t) * 3.0 + t * 1.5 + 3.0 * t * (1 - t) * .4
        assert curve.predict(rate) == pytest.approx(expected, rel=2e-15)
    assert curve.predict(832) == 3.0
    assert curve.predict(1088) == 1.5


def test_log2_curve_equals_direct_standardized_feature_polynomial():
    # width 4: endpoint level, endpoint log slope, t, and one caller context.
    means = [2.0, -.7, .4, 3.0]
    scales = [1.5, .5, .2, 4.0]
    coefficients = tuple((index - 6) / 40 for index in range(15))
    model = EndpointCurvatureModel("log2", 832, 1088, 2, means, scales, coefficients)
    left, right, context = 5.0, 1.25, [7.0]
    curve = model.bind(left, right, context)
    for rate in (864, 901, 960, 1037, 1056):
        t = (rate - 832) / 256
        raw = [math.log2(left), math.log2(right) - math.log2(left), t, context[0]]
        features = [(value - mean) / scale for value, mean, scale in zip(raw, means, scales)]
        curvature = sum(coefficient * feature for coefficient, feature in
                        zip(coefficients, _basis(features, 2)))
        expected = 2.0 ** (math.log2(left) + t * (math.log2(right) - math.log2(left))
                           + t * (1 - t) * curvature)
        assert curve.predict(rate) == pytest.approx(expected, rel=3e-14)


def test_model_and_bound_curve_freeze_caller_inputs():
    means, scales, coefficients, context = [0.0, 0.0, 0.0, 2.0], [1.0] * 4, [0.1] * 5, [4.0]
    model = EndpointCurvatureModel("value", 1, 3, 1, means, scales, coefficients)
    curve = model.bind(2.0, 1.0, context)
    before = curve.predict(2)
    means[0], scales[2], coefficients[0], context[0] = 99.0, 99.0, 99.0, 99.0
    assert model.means == (0.0, 0.0, 0.0, 2.0)
    assert model.scales == (1.0, 1.0, 1.0, 1.0)
    assert model.coefficients == (0.1,) * 5
    assert curve.predict(2) == before
    with pytest.raises(FrozenInstanceError):
        curve.c0 = 1.0


@pytest.mark.parametrize("kwargs", [
    {"mode": "base10"},
    {"rate_lo": 2, "rate_hi": 2},
    {"rate_lo": math.nan},
    {"degree": 3},
    {"means": (0.0, 0.0)},
    {"scales": (1.0, 0.0, 1.0)},
    {"coefficients": (0.0,)},
])
def test_invalid_model_parameters_refuse(kwargs):
    baseline = {"mode": "value", "rate_lo": 1, "rate_hi": 3, "degree": 1,
                "means": (0.0, 0.0, 0.0), "scales": (1.0, 1.0, 1.0),
                "coefficients": (0.0, 0.0, 0.0, 0.0)}
    baseline.update(kwargs)
    with pytest.raises(AnchoredShapeError):
        EndpointCurvatureModel(**baseline)


def test_invalid_binding_predictions_and_extrapolation_refuse_without_clamping():
    model = _linear_model()
    with pytest.raises(AnchoredShapeError, match="left"):
        model.bind(0.0, 1.0)
    with pytest.raises(AnchoredShapeError, match="context width"):
        model.bind(1.0, 1.0, (3.0,))
    curve = model.bind(1.0, 1.0)
    for rate in (831, 1089, math.nan):
        with pytest.raises(AnchoredShapeError):
            curve.predict(rate)
    negative = _linear_model(coefficients=(-10.0, 0.0, 0.0, 0.0)).bind(1.0, 1.0)
    with pytest.raises(AnchoredShapeError, match="positive finite"):
        negative.predict(960)


def _pilot_panel(n_units=6):
    coordinates = {"lo": 832.0, "a": 896.0, "b": 960.0, "hi": 1088.0}
    observations = []
    for unit in range(n_units):
        left, right = 2.0 ** (-7.0 - unit / 8), 2.0 ** (-10.0 - unit / 11)
        log_left, slope = math.log2(left), math.log2(right) - math.log2(left)
        for key, rate in coordinates.items():
            t = (rate - coordinates["lo"]) / (coordinates["hi"] - coordinates["lo"])
            curvature = .08 + .013 * log_left - .021 * slope + .034 * t
            value = 2.0 ** (log_left + t * slope + t * (1 - t) * curvature)
            observations.append(LogShapeObservation(f"u{unit:02d}", key, value))
    return observations, coordinates


def test_fitter_matches_the_study_standardize_ridge_huber_arithmetic():
    observations, coordinates = _pilot_panel()
    fitted = fit_endpoint_curvature(observations, coordinates, mode="log2", degree=1)
    by_unit = {}
    for observation in observations:
        by_unit.setdefault(observation.unit, {})[observation.key] = observation.value
    features, responses = [], []
    for key in ("a", "b"):  # Same rate-major/unit-major order as the shared fitter.
        rate = coordinates[key]
        t = (rate - coordinates["lo"]) / (coordinates["hi"] - coordinates["lo"])
        for unit in sorted(by_unit):
            left, right, value = by_unit[unit]["lo"], by_unit[unit]["hi"], by_unit[unit][key]
            log_left = math.log2(left)
            slope = math.log2(right) - log_left
            features.append((log_left, slope, t))
            responses.append((math.log2(value) - (log_left + t * slope)) / (t * (1 - t)))
    mean, scale, coefficients = robust_fit(np.asarray(features), np.asarray(responses), 1)
    assert fitted.means == pytest.approx(tuple(mean), rel=0, abs=0)
    assert fitted.scales == pytest.approx(tuple(scale), rel=0, abs=0)
    assert fitted.coefficients == pytest.approx(tuple(coefficients), rel=0, abs=0)
    held = by_unit["u05"]
    assert fitted.bind(held["lo"], held["hi"]).predict(960) == pytest.approx(held["b"], rel=3e-4)


def test_fitter_refuses_incomplete_or_ambiguous_pilots_and_unsupported_degree():
    observations, coordinates = _pilot_panel()
    with pytest.raises(AnchoredShapeError, match="endpoint"):
        fit_endpoint_curvature([row for row in observations if not (row.unit == "u00" and row.key == "hi")], coordinates)
    with pytest.raises(AnchoredShapeError, match="duplicate"):
        fit_endpoint_curvature([*observations, observations[0]], coordinates)
    with pytest.raises(AnchoredShapeError, match="unknown"):
        fit_endpoint_curvature([*observations, LogShapeObservation("u00", "other", 1.0)], coordinates)
    with pytest.raises(AnchoredShapeError, match="degree"):
        fit_endpoint_curvature(observations, coordinates, degree=3)
    with pytest.raises(AnchoredShapeError, match="insufficient"):
        fit_endpoint_curvature(_pilot_panel(5)[0], coordinates)


def test_overflowing_rate_span_refuses_before_an_interior_can_be_treated_as_an_anchor():
    with pytest.raises(AnchoredShapeError, match="span"):
        EndpointCurvatureModel("value", -sys.float_info.max, sys.float_info.max, 1,
                               (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0,) * 4)
