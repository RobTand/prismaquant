"""#2774 consumption regressions from the prior #2535 delivery review.

All records are synthetic CPU fixtures, not a numeric interim authorization
or scientific qualification. Source bindings must survive admission and be
enforced by the actual consumers, not just compared at JSON intake.
"""
from __future__ import annotations

import copy

import pytest

from prismaquant import allocator_candidates as ac, joint_aura_rows, priced_cell
from prismaquant.model_profiles import DefaultProfile
from prismaquant.rung_allowability import complete_scientific_price
from prismaquant.tessera_expert_projection import select_priced_unit_upgrades
from canonical_quality_fixtures import complete_probe, scope_for
from test_glm_mtp_selection import _row as complete_row
from test_joint_aura_assignment_diagnostics import _row
from test_routed_depth_pricing import EARLY, LATE, _payload, _policy, _record

LOW = "TESSERA_E4M3_K1_R1024"
HIGH = "TESSERA_E4M3_K1_R1088"


def _consumption_depth_fixture(*, scientific=False):
    payload, _, _ = _payload()
    costs = {}
    for name in EARLY + LATE:
        costs[name] = {}
        for fmt, signed, wire in ((LOW, [1, 3, 1, 3], 100),
                                   (HIGH, [1, 1, 1, 1], 110)):
            row = (complete_row(name, fmt, signed, complete_probe()) if scientific
                   else _row(name, signed, fmt=fmt))
            row["wire_bytes"] = wire
            if scientific:
                row["quality_scope"] = scope_for(row)
                # Establish validity before testing the new policy: neither a
                # malformed scope nor a missing producer API explains failure.
                assert complete_scientific_price(row, format_name=fmt) == row["predicted_dloss"]
            assert joint_aura_rows.validate_joint_aura_entry(row)
            costs[name][fmt] = row
    payload["costs"] = costs
    return payload


@pytest.mark.parametrize("z", [0.0, 2.0])
def test_complete_scientific_joint_price_receives_band_transform_exactly_once(z):
    payload = _consumption_depth_fixture(scientific=True)
    original = copy.deepcopy(payload)
    policy = _policy(payload, _record(payload, qualified=True), production=True, z=z)
    for name, factor in ((EARLY[0], 4.0), (LATE[0], 1.0)):
        row = payload["costs"][name][LOW]
        legacy = priced_cell.price_cell({}, row, format_name=LOW,
            policy=priced_cell.CostPricingPolicy(ucb_z=z, fisher_output_mse=False))
        assert legacy.predicted_dloss == 2.5
        effective = priced_cell.price_cell({"h_trace": 1e12}, row, format_name=LOW, policy=policy)
        # Preserve the complete scientific quantity's existing hedge semantics;
        # this policy transforms its loss currency, not Fisher or its identity.
        assert effective.predicted_dloss == legacy.predicted_dloss * factor
        assert priced_cell.cost_entry_predicted_dloss({}, row, format_name=LOW,
            policy=policy) == effective.predicted_dloss
    assert payload == original
    # A different fixed-price currency must not inherit a routed multiplier
    # merely because the qualified quantity names an in-scope unit.
    row = copy.deepcopy(payload["costs"][EARLY[0]][LOW])
    scope = row["quality_scope"]
    scope["currency"] = scope["objective"]["currency"] = "served_kl"
    served = {"predicted_dloss": 0.75, "quality_scope": scope}
    assert complete_scientific_price(served, format_name=LOW) == 0.75
    assert priced_cell.price_cell({}, served, format_name=LOW, policy=policy).predicted_dloss == 0.75


@pytest.mark.parametrize("consumer", ["cell", "paired", "selector"])
@pytest.mark.parametrize("scientific", [False, True], ids=["raw_joint", "complete_joint"])
def test_qualified_depth_receipts_cannot_be_replayed_against_replacement_measurements(consumer, scientific):
    payload = _consumption_depth_fixture(scientific=scientific)
    record = _record(payload, qualified=True)
    policy = _policy(payload, record, production=True)
    # This replacement has self-consistent scientific identities and summary,
    # same qname/format, but different signed measurements. Intake receipts
    # belong to the original table, not to every future row at this coordinate.
    replacement = (complete_row(EARLY[0], LOW, [9, 9, 9, 9], complete_probe()) if scientific
                   else _row(EARLY[0], [9, 9, 9, 9], fmt=LOW))
    replacement["wire_bytes"] = 100
    if scientific:
        replacement["quality_scope"] = scope_for(replacement)
        assert complete_scientific_price(replacement, format_name=LOW) == 40.5
    assert joint_aura_rows.validate_joint_aura_entry(replacement)
    changed = copy.deepcopy(payload)
    changed["costs"][EARLY[0]][LOW] = replacement
    before = copy.deepcopy(changed)
    with pytest.raises(ValueError):
        if consumer == "cell":
            priced_cell.price_cell({}, replacement, format_name=LOW, policy=policy)
        elif consumer == "paired":
            ac.price_paired_rate_trade(changed["costs"],
                {name: LOW for name in EARLY + LATE}, {name: HIGH for name in EARLY + LATE},
                profile=DefaultProfile(), ucb_z=0, pricing_policy=policy)
        else:
            select_priced_unit_upgrades(changed["costs"],
                {name: LOW for name in EARLY + LATE}, byte_budget=10, pricing_policy=policy)
    assert changed == before


@pytest.mark.parametrize("consumer", ["cell", "paired", "selector"])
def test_finite_band_parameter_cannot_admit_infinite_effective_loss(consumer):
    payload = _consumption_depth_fixture()
    # All raw samples, raw point prices and the parameter are finite. Their
    # product is not. Refuse at admission or consumption, never silently send
    # inf/NaN to candidate ranking, covariance or the allocation solver.
    record = _record(payload, multipliers=(1e308, 1.0))
    factory = getattr(priced_cell.CostPricingPolicy, "from_depth_profile", None)
    assert callable(factory), "#2774: no band-price admission to enforce finite effective prices"
    original = copy.deepcopy(payload)
    with pytest.raises(ValueError):
        policy = factory(payload, profile_record=record, production=False, ucb_z=0)
        if consumer == "cell":
            priced_cell.price_cell({}, payload["costs"][EARLY[0]][LOW], format_name=LOW, policy=policy)
        elif consumer == "paired":
            ac.price_paired_rate_trade(payload["costs"],
                {name: LOW for name in EARLY + LATE}, {name: HIGH for name in EARLY + LATE},
                profile=DefaultProfile(), ucb_z=0, pricing_policy=policy)
        else:
            select_priced_unit_upgrades(payload["costs"],
                {name: LOW for name in EARLY + LATE}, byte_budget=10, pricing_policy=policy)
    assert payload == original


def test_admitted_policy_owns_bindings_and_parameters_not_mutable_intake_aliases():
    payload = _consumption_depth_fixture()
    record = _record(payload)
    policy = _policy(payload, record)
    pristine = copy.deepcopy(payload["costs"][EARLY[0]][LOW])
    record["parameters"]["early"]["loss_multiplier"] = 999.0
    exposed_metadata = policy.depth_profile
    exposed_metadata["parameters"]["early"]["loss_multiplier"] = 999.0
    payload["costs"][EARLY[0]][LOW] = _row(EARLY[0], [9, 9, 9, 9], fmt=LOW)
    # An admitted source snapshot must still recognize its own scientific row.
    assert priced_cell.price_cell({}, pristine, format_name=LOW, policy=policy).predicted_dloss == 10.0
    assert policy.depth_profile["parameters"]["early"]["loss_multiplier"] == 4.0
