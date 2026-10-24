"""Synthetic #2774 regressions for exact decisions and consumption coordinates.

These reuse the intake fixtures; they are not authentic authorization or
scientific qualification evidence. Protected acceptance files are unchanged.
"""
from __future__ import annotations

import copy

import pytest

from prismaquant.joint_aura_rows import joint_identity_sha256 as digest
from prismaquant.tessera_expert_projection import select_priced_unit_upgrades
from test_interim_depth_record_admission import _interim_intake_fixture, _interim_policy
from test_routed_depth_price_consumption import HIGH, LOW, _consumption_depth_fixture
from test_routed_depth_pricing import EARLY, LATE, _payload, _policy, _record


@pytest.mark.parametrize("value, rehash_parameters", [(True, False), (True, True), (1, False)])
def test_decision_cannot_use_python_numeric_equality_instead_of_exact_binding(value, rehash_parameters):
    payload, _, _ = _payload()
    profile, decision = _interim_intake_fixture(payload)
    decision["parameters"]["late"]["loss_multiplier"] = value
    if rehash_parameters:
        decision["parameters_sha256"] = digest(decision["parameters"])
    profile["authorization_sha256"] = digest(decision)
    before = copy.deepcopy((payload, profile, decision))
    with pytest.raises(ValueError):
        _interim_policy(payload, profile, decision)
    assert (payload, profile, decision) == before


@pytest.mark.parametrize("state", ["interim_flat", "research_calibrated", "qualified"])
@pytest.mark.parametrize("mutation", ["non_joint_baseline", "non_joint_candidate", "relocated_unit", "relocated_format"])
@pytest.mark.parametrize("member", [EARLY[0], LATE[0]])
def test_selector_refuses_changed_or_relocated_admitted_coordinate(state, mutation, member):
    payload = _consumption_depth_fixture()
    if state == "interim_flat":
        profile, decision = _interim_intake_fixture(payload, multipliers=(4.0, 1.0))
        policy = _interim_policy(payload, profile, decision)
    else:
        policy = _policy(payload, _record(payload, qualified=state == "qualified"),
                         production=state == "qualified")
    costs = payload["costs"]
    assignment = {name: LOW for name in costs}
    # Establish the legitimate single-unit choice before changing coordinates.
    assert select_priced_unit_upgrades(costs, assignment, byte_budget=10,
        pricing_policy=policy)[0] == {EARLY[0]: HIGH}
    if mutation.startswith("non_joint"):
        fmt = LOW if mutation == "non_joint_baseline" else HIGH
        costs[member][fmt] = {"predicted_dloss": 12345.0, "wire_bytes": 100 if fmt == LOW else 110}
    elif mutation == "relocated_unit":
        other = LATE[0] if member == EARLY[0] else EARLY[0]
        costs[member], costs[other] = costs[other], costs[member]
    else:
        costs[member][LOW], costs[member][HIGH] = costs[member][HIGH], costs[member][LOW]
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError):
        select_priced_unit_upgrades(costs, assignment, byte_budget=10, pricing_policy=policy)
    assert payload == before


def test_coordinate_authentication_preserves_unrelated_non_joint_unit_prices_and_bytes():
    payload = _consumption_depth_fixture()
    policy = _policy(payload, _record(payload, qualified=True), production=True)
    unrelated = "model.layers.0.self_attn.q_proj"
    costs = {unrelated: {LOW: {"predicted_dloss": 4.5, "wire_bytes": 100},
                         HIGH: {"predicted_dloss": 0.5, "wire_bytes": 110}}}
    before = copy.deepcopy(costs)
    assignment = {unrelated: LOW}
    legacy = select_priced_unit_upgrades(costs, assignment, byte_budget=10)
    assert legacy[0] == {unrelated: HIGH}
    assert select_priced_unit_upgrades(costs, assignment, byte_budget=10,
                                      pricing_policy=policy) == legacy
    assert costs == before
