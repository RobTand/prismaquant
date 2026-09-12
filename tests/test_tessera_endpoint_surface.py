"""Endpoint-curvature replay contracts; all measurements are synthetic."""
from __future__ import annotations

import math

import pytest

from prismaquant.tessera_anchored_surface import ReplayError, replay_measurements


COORDINATES = {"R832": 832, "R960": 960, "R1088": 1088}
DESCRIPTOR = {"family": "TESSERA_SYNTHETIC", "activation_contract": "fp8",
              "geometry": [8, 8], "wire_recipe": {"grid": "synthetic"},
              "hessian_applied": False, "role": "declared"}


def _curve(left, right, rate, *, mode="value", degree=1):
    t = (rate - 832) / 256
    curvature = .12 + .01 * math.log2(left) - .02 * (math.log2(right) - math.log2(left)) + .03 * t
    if degree == 2:
        curvature += .004 * t * t
    if mode == "value":
        return (1 - t) * left + t * right + left * t * (1 - t) * curvature
    return 2 ** (math.log2(left) + t * (math.log2(right) - math.log2(left))
                 + t * (1 - t) * curvature)


def _fixture(*, heldout_mid_scale=1.0, missing=(), mode="value", degree=1):
    # The shared fitter requires twelve independent interior observations.
    units = {f"p{index}": (4.0 + index, 1.0 + index / 13) for index in range(12)}
    units["h"] = (7.0, 1.4)
    measurements = {}
    for unit, (left, right) in units.items():
        for key, rate in COORDINATES.items():
            if (unit, key) in missing:
                continue
            value = _curve(left, right, rate, mode=mode, degree=degree)
            if unit == "h" and key == "R960":
                value *= heldout_mid_scale
            measurements[(unit, key)] = {"value": value, "currency": "output_mse_under_route_activation_contract",
                                          "coordinate": rate, **DESCRIPTOR}
    pilots = {unit: list(COORDINATES) for unit in units if unit.startswith("p")}
    plan = {"schema": "prismaquant.tessera_anchored_replay.plan.v1",
            "currency": "output_mse_under_route_activation_contract",
            "segments": [{"id": "synthetic", "descriptor": DESCRIPTOR,
                          "coordinates": COORDINATES, "pilot": pilots,
                          "heldout": {"h": {"anchors": ["R832", "R1088"], "audit": ["R960"]}},
                          "shape_model": {"kind": "endpoint_curvature", "mode": mode, "degree": degree},
                          "max_absolute_log10_error": 1e-4, "refit_after_audit": False}]}
    identity = {"recorded_menus": {unit: list(COORDINATES) for unit in (*units, "h_sibling")}}
    return measurements, plan, identity


def _unit(report):
    return report["segments"][0]["units"]["h"]


def test_endpoint_curvature_replay_keeps_endpoints_exact_and_reports_frozen_fit():
    measurements, plan, identity = _fixture()
    report = replay_measurements(measurements, plan, input_identity=identity,
                                 groups={"g": ["h", "h_sibling"]})
    unit = _unit(report)
    assert unit["status"] == "audit_accepted_research_only"
    assert unit["predictions"]["R832"]["source"] == "measured"
    assert unit["predictions"]["R1088"]["source"] == "measured"
    assert unit["predictions"]["R960"]["value"] == pytest.approx(measurements["h", "R960"]["value"])
    assert unit["audit"][0]["held_out_axes"] == ["unit"]
    fit = report["segments"][0]["pilot_fit"]
    assert fit["kind"] == "endpoint_curvature"
    assert set(fit) == {"kind", "mode", "degree", "rate_lo", "rate_hi", "means", "scales", "coefficients"}
    assert report["measurement_requests"] == []


def test_endpoint_curvature_replay_accepts_log2_quadratic_shape():
    measurements, plan, identity = _fixture(mode="log2", degree=2)
    report = replay_measurements(measurements, plan, input_identity=identity)
    assert _unit(report)["status"] == "audit_accepted_research_only"
    assert report["segments"][0]["pilot_fit"]["mode"] == "log2"
    assert report["segments"][0]["pilot_fit"]["degree"] == 2


def _sparse_rate_fixture(*, heldout_scale=1.0):
    coordinates = {"R832": 832, "R864": 864, "R928": 928, "R960": 960,
                   "R1056": 1056, "R1088": 1088}

    def curve(left, right, rate):
        t = (rate - 832) / 256
        curvature = .08 + .01 * math.log2(left) + .02 * t + .003 * t * t
        return (1 - t) * left + t * right + left * t * (1 - t) * curvature

    units = {f"p{index}": (4.0 + index, 1.0 + index / 13) for index in range(12)}
    units["h"] = (7.0, 1.4)
    measurements = {}
    for unit, (left, right) in units.items():
        for key, rate in coordinates.items():
            value = curve(left, right, rate)
            if unit == "h" and key == "R928":
                value *= heldout_scale
            measurements[(unit, key)] = {"value": value,
                                          "currency": "output_mse_under_route_activation_contract",
                                          "coordinate": rate, **DESCRIPTOR}
    pilot_keys = ["R832", "R864", "R960", "R1056", "R1088"]
    plan = {"schema": "prismaquant.tessera_anchored_replay.plan.v1",
            "currency": "output_mse_under_route_activation_contract",
            "segments": [{"id": "sparse", "descriptor": DESCRIPTOR,
                          "coordinates": coordinates,
                          "pilot": {unit: pilot_keys for unit in units if unit.startswith("p")},
                          "heldout": {"h": {"anchors": ["R832", "R1088"], "audit": ["R928"]}},
                          "shape_model": {"kind": "endpoint_curvature", "mode": "value", "degree": 2},
                          "max_absolute_log10_error": .01, "refit_after_audit": False}]}
    identity = {"recorded_menus": {unit: list(coordinates) for unit in units}}
    return measurements, plan, identity


def test_sparse_arbitrary_rate_audit_is_unseen_and_frozen():
    measurements, plan, identity = _sparse_rate_fixture()
    base = replay_measurements(measurements, plan, input_identity=identity)
    changed_measurements, changed_plan, changed_identity = _sparse_rate_fixture(heldout_scale=1.001)
    changed = replay_measurements(changed_measurements, changed_plan, input_identity=changed_identity)
    base_unit, changed_unit = _unit(base), _unit(changed)
    assert base_unit["status"] == changed_unit["status"] == "audit_accepted_research_only"
    assert base_unit["audit"][0]["held_out_axes"] == ["unit", "rung"]
    assert base_unit["audit"][0]["predicted"] == pytest.approx(changed_unit["audit"][0]["predicted"])
    assert base_unit["audit"][0]["measured"] != changed_unit["audit"][0]["measured"]


def test_failed_audit_uses_existing_group_measurement_requests():
    measurements, plan, identity = _fixture(heldout_mid_scale=1.1)
    report = replay_measurements(measurements, plan, input_identity=identity,
                                 groups={"g": ["h", "h_sibling"]})
    assert _unit(report)["reason"] == "audit_or_monotonicity_failed"
    requests = {(row["unit"], row["key"]) for row in report["measurement_requests"]}
    assert requests == {(unit, key) for unit in ("h", "h_sibling") for key in COORDINATES}


def test_missing_required_pilot_anchor_requests_it_without_a_prediction():
    measurements, plan, identity = _fixture(missing={("p0", "R832")})
    report = replay_measurements(measurements, plan, input_identity=identity)
    assert report["segments"][0]["pilot_fit_error"] == "missing_or_invalid_pilot_measurements"
    assert _unit(report)["predictions"] == {}
    assert {row["unit"] for row in report["measurement_requests"]} >= {"p0", "h"}


@pytest.mark.parametrize("model", [
    {"kind": "endpoint_curvature", "mode": "base10", "degree": 1},
    {"kind": "endpoint_curvature", "mode": "value", "degree": 3},
])
def test_endpoint_policy_and_anchor_requirements_fail_closed(model):
    measurements, plan, identity = _fixture()
    plan["segments"][0]["shape_model"] = model
    with pytest.raises(ReplayError, match="endpoint-curvature"):
        replay_measurements(measurements, plan, input_identity=identity)

    measurements, plan, identity = _fixture()
    plan["segments"][0]["heldout"]["h"]["anchors"] = ["R832", "R960"]
    with pytest.raises(ReplayError, match="exact domain endpoints"):
        replay_measurements(measurements, plan, input_identity=identity)
