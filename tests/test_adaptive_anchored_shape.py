"""Focused state-machine tests for format-neutral adaptive PWL interpolation."""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from prismaquant.adaptive_anchored_shape import AdaptiveAnchoredCurve
from prismaquant.anchored_shape import AnchoredShapeError


def _start(*, checks_per_interval=1, max_measurements=9, mode="value"):
    return AdaptiveAnchoredCurve.start(
        tuple(range(9)), {0: 100.0, 8: 1.0}, mode=mode,
        relative_tolerance=0.10, max_measurements=max_measurements,
        checks_per_interval=checks_per_interval,
    )


def _request(curve):
    next_curve, probe = curve.request_next()
    assert probe is not None
    return next_curve, probe


def test_curvature_failure_refines_then_keeps_real_anchors_exact():
    curve, first = _request(_start())
    assert (first.coordinate, first.prediction) == (4, 50.5)
    curve = curve.record_measurement(4, 40.0)
    assert curve.measurement_count == 3
    assert not curve.accepted_intervals
    assert [(item.left_coordinate, item.right_coordinate) for item in curve.unverified_intervals] == [
        (0, 4), (4, 8),
    ]

    curve, second = _request(curve)
    assert second.coordinate == 2
    assert second.prediction == 70.0
    curve = curve.record_measurement(2, 75.0)
    assert curve.predict(2) == 75.0
    assert [(item.left_coordinate, item.right_coordinate) for item in curve.accepted_intervals] == [
        (0, 4),
    ]

    curve, third = _request(curve)
    assert third.coordinate == 6
    assert math.isclose(third.prediction, 20.5)
    curve = curve.record_measurement(6, 12.0)
    assert curve.predict(0) == 100.0
    assert curve.predict(4) == 40.0
    assert curve.predict(6) == 12.0
    assert curve.predict(8) == 1.0
    assert curve.measurement_count == 5
    assert [(item.left_coordinate, item.right_coordinate) for item in curve.unverified_intervals] == [
        (4, 6), (6, 8),
    ]


def test_midpoint_pass_is_empirical_evidence_not_whole_domain_validation():
    curve, probe = _request(_start())
    curve = curve.record_measurement(probe.coordinate, probe.prediction)
    assert len(curve.accepted_intervals) == 1
    evidence = curve.accepted_intervals[0]
    assert (evidence.left_coordinate, evidence.right_coordinate) == (0, 8)
    assert curve.unverified_intervals == ()
    # An omitted rate can still be adversarially far from the accepted chord.
    adversarial_unmeasured_value = 95.0
    assert curve.predict(2) == 75.25
    assert abs(curve.predict(2) - adversarial_unmeasured_value) / adversarial_unmeasured_value > 0.10
    assert not curve.all_candidates_measured
    assert not hasattr(curve, "domain_validated")


def test_two_sentinels_catch_a_shape_midpoint_would_miss():
    curve, first = _request(_start(checks_per_interval=2))
    assert first.coordinate == 3
    curve = curve.record_measurement(3, first.prediction)
    curve, second = _request(curve)
    assert second.coordinate == 6
    # The second sentinel is evaluated against the original endpoints, not
    # the new PWL segment introduced by the first passing sentinel.
    assert math.isclose(second.prediction, 25.75)
    curve = curve.record_measurement(6, 10.0)
    assert not curve.accepted_intervals
    assert [(item.left_coordinate, item.right_coordinate) for item in curve.unverified_intervals] == [
        (0, 3), (3, 6), (6, 8),
    ]


def test_pending_probe_cannot_be_rerevealed_or_replaced():
    curve, probe = _request(_start())
    again, no_probe = curve.request_next()
    assert again is curve
    assert no_probe is None
    curve = curve.record_measurement(probe.coordinate, probe.prediction)
    with pytest.raises(AnchoredShapeError, match="pending"):
        curve.record_measurement(probe.coordinate, probe.prediction)


def test_measurement_budget_counts_actual_unique_reveals_and_leaves_brackets_unverified():
    curve, probe = _request(_start(max_measurements=3))
    curve = curve.record_measurement(probe.coordinate, 40.0)
    assert curve.measurement_count == 3
    assert curve.budget_exhausted
    unchanged, next_probe = curve.request_next()
    assert unchanged is curve
    assert next_probe is None
    assert [(item.left_coordinate, item.right_coordinate) for item in curve.unverified_intervals] == [
        (0, 4), (4, 8),
    ]


def test_first_of_two_checks_at_cap_remains_unverified():
    curve, probe = _request(_start(checks_per_interval=2, max_measurements=3))
    curve = curve.record_measurement(probe.coordinate, probe.prediction)
    assert curve.budget_exhausted
    assert not curve.accepted_intervals
    assert [(item.left_coordinate, item.right_coordinate) for item in curve.unverified_intervals] == [
        (0, 8),
    ]
    unchanged, next_probe = curve.request_next()
    assert unchanged is curve
    assert next_probe is None


def test_public_dataclass_rejects_forged_pending_or_accepted_evidence():
    pending_state, probe = _request(_start())
    with pytest.raises(AnchoredShapeError, match="frozen parent"):
        replace(pending_state, pending_checks=(replace(probe, prediction=1.0),))

    accepted = pending_state.record_measurement(probe.coordinate, 48.0)
    evidence = accepted.accepted_intervals[0]
    forged_check = replace(evidence.checks[0], relative_residual=0.0)
    with pytest.raises(AnchoredShapeError, match="empirical acceptance"):
        replace(accepted, accepted_intervals=(replace(evidence, checks=(forged_check,)),))


def test_direct_state_cannot_drop_unmeasured_coverage_but_exact_steps_are_terminal():
    unfinished = _start()
    with pytest.raises(AnchoredShapeError, match="cover the candidate domain"):
        replace(unfinished, _active_intervals=())

    terminal = AdaptiveAnchoredCurve.start(
        (0, 1, 2), {0: 16.0, 2: 1.0}, mode="value",
        relative_tolerance=0.0, max_measurements=3,
    )
    terminal, probe = _request(terminal)
    terminal = terminal.record_measurement(probe.coordinate, 8.0)
    assert terminal.all_candidates_measured
    assert not terminal.unverified_intervals


@pytest.mark.parametrize("coordinates,endpoints,kwargs", [
    ((0, 0, 8), {0: 100.0, 8: 1.0}, {}),
    ((0, 4, 8), {0: 1.0, 8: 100.0}, {}),
    ((0, 4, 8), {0: 100.0, 8: 1.0}, {"mode": "bad"}),
    ((0, 4, 8), {0: 100.0, 8: 1.0}, {"relative_tolerance": -0.1}),
    ((0, 4, 8), {0: 100.0, 8: 1.0}, {"max_measurements": 1}),
    ((0, 4, 8), {0: 100.0, 8: 1.0}, {"checks_per_interval": 3}),
])
def test_invalid_start_inputs_are_refused(coordinates, endpoints, kwargs):
    options = {
        "mode": "value", "relative_tolerance": 0.1,
        "max_measurements": 3, "checks_per_interval": 1,
    }
    options.update(kwargs)
    with pytest.raises(AnchoredShapeError):
        AdaptiveAnchoredCurve.start(coordinates, endpoints, **options)


def test_invalid_measurement_and_extrapolation_are_refused():
    curve, probe = _request(_start())
    with pytest.raises(AnchoredShapeError, match="nonmonotone"):
        curve.record_measurement(probe.coordinate, 200.0)
    with pytest.raises(AnchoredShapeError, match="outside"):
        curve.predict(9)
    with pytest.raises(AnchoredShapeError, match="pending coordinate"):
        curve.record_measurement(3, 80.0)


def test_modes_exact_anchors_and_request_order_are_deterministic():
    log_curve = AdaptiveAnchoredCurve.start(
        (0, 2, 4), {4: 1.0, 0: 16.0}, mode="log2",
        relative_tolerance=0.0, max_measurements=3,
    )
    value_curve = AdaptiveAnchoredCurve.start(
        (0, 2, 4), {0: 16.0, 4: 1.0}, mode="value",
        relative_tolerance=0.0, max_measurements=3,
    )
    assert log_curve.predict(2) == 4.0
    assert value_curve.predict(2) == 8.5
    a_state, a_probe = log_curve.request_next()
    b_state, b_probe = log_curve.request_next()
    assert a_probe == b_probe
    assert a_state == b_state
    assert a_probe is not None
    completed = a_state.record_measurement(a_probe.coordinate, 4.0)
    assert completed.predict(0) == 16.0
    assert completed.predict(2) == 4.0
    assert completed.predict(4) == 1.0
