"""Synthetic regressions for complete scientific coordinates and one quantity."""
from __future__ import annotations

import copy
import math

import pytest

from prismaquant import allocator_candidates as candidates, format_registry as registry
from prismaquant import joint_aura as joint
from prismaquant.activation_fair_pricing import ActivationFairPricing, FamilyCalibration, _family_of
from prismaquant.rung_allowability import RungAllowabilityError, qualified_cost_scope
from test_canonical_consumer_join import FAMILY, _cost_owner, _mutated
from test_glm_mtp_selection import _row
from canonical_quality_fixtures import complete_probe, scope_for



def anchor_rows(*, currency=joint.JOINT_CURRENCY, producer="d", upper_seed=7000):
    rows = {}
    for rate, value, source, seed in ((768, 2.0, "d", 7000), (1024, 1.0, producer, upper_seed)):
        name = f"{FAMILY}_R{rate}"
        row = _row("u", name, [value] * 4, complete_probe(producer=source, seed=seed))
        row["quality_scope"] = scope_for(row)
        if currency == "served_kl":
            scope = copy.deepcopy(row["quality_scope"])
            scope["currency"] = currency
            scope["objective"]["currency"] = currency
            row = {"predicted_dloss": 1.0 if rate == 768 else 0.5, "quality_scope": scope}
        rows[name] = row
    return rows


def owner_for_rows(monkeypatch):
    owner = _cost_owner(monkeypatch)
    def change(table):
        for shape in table["scope"]["shapes"]:
            shape["rows"], shape["columns"] = 64, 128
        for row in table["rungs"]:
            for measurement in row["measurements"]:
                measurement["evidence"]["rows"], measurement["evidence"]["columns"] = 64, 128
    return _mutated(owner, change)


def nontrivial_calibration(fmt):
    family = _family_of(fmt)
    fit = FamilyCalibration(family, 8, 13.0, math.log2(13.0), 0.0, 0.0, 0.0, 0.0,
                            (fmt,), ((fmt, math.log2(13.0), 8),), 0.0, (), "f" * 64)
    return ActivationFairPricing(True, "synthetic calibration", {family: fit},
                                 {family: 8}, {family: 8}, ())


@pytest.mark.parametrize("currency,expected", [(joint.JOINT_CURRENCY, 1.25), ("served_kl", 0.75)])
def test_body_chord_keeps_its_quantity_without_another_transfer(monkeypatch, currency, expected):
    owner = owner_for_rows(monkeypatch)
    fmt = f"{FAMILY}_R896"
    stats = {"u": {"out_features": 64, "in_features": 128, "n_params": 8192,
                   "h_trace": 9.0, "unit_structure": "dense"}}
    built = candidates.build_candidates(stats, {"u": anchor_rows(currency=currency)},
        [registry.get_format(fmt)], target_profile="research", rung_allowability={FAMILY: owner},
        allowability_m=8, calibrated_gains={fmt: 9.0}, activation_pricing=nontrivial_calibration(fmt))
    assert built["u"][0].predicted_dloss == expected
    assert built["u"][0].activation_pricing == "canonical_qualified_chord"


@pytest.mark.parametrize("missing", ["family", "format", "shape", "teacher", "window", "objective",
                                    "source_weight", "activation_contract", "probe_identity", "probe_ids"])
def test_joint_anchor_requires_complete_scope(monkeypatch, missing):
    owner = _cost_owner(monkeypatch)
    rows = anchor_rows()
    rows[f"{FAMILY}_R768"]["quality_scope"].pop(missing)
    with pytest.raises(ValueError):
        owner.chord_cost(f"{FAMILY}_R896", unit="u", costs=rows)


def test_optional_scope_cannot_bypass_joint_sample_validation(monkeypatch):
    owner = _cost_owner(monkeypatch)
    rows = anchor_rows()
    rows[f"{FAMILY}_R768"]["predicted_dloss"] = 99.0
    with pytest.raises(ValueError, match="aligned samples"):
        owner.chord_cost(f"{FAMILY}_R896", unit="u", costs=rows)


def test_joint_anchor_binds_its_actual_format_key(monkeypatch):
    owner = _cost_owner(monkeypatch)
    rows = anchor_rows()
    wrong = _row("u", f"{FAMILY}_R1024", [2.0] * 4, complete_probe())
    wrong["quality_scope"] = scope_for(wrong)
    wrong["quality_scope"]["format"] = f"{FAMILY}_R768"
    rows[f"{FAMILY}_R768"] = wrong
    with pytest.raises(ValueError, match="produced|format"):
        owner.chord_cost(f"{FAMILY}_R896", unit="u", costs=rows)


def test_producer_only_probe_difference_stamps_and_keeps_the_chord(monkeypatch, capsys):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    owner = _cost_owner(monkeypatch)
    rows = anchor_rows(producer="e")
    result = owner.chord_cost(f"{FAMILY}_R896", unit="u", costs=rows)
    assert result["predicted_dloss"] == 1.25
    assert "[DEV-MODE]" in capsys.readouterr().out


def test_mathematical_probe_difference_refuses_in_development_mode(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    owner = _cost_owner(monkeypatch)
    with pytest.raises(ValueError, match="probe|sample|coordinate"):
        owner.chord_cost(f"{FAMILY}_R896", unit="u", costs=anchor_rows(upper_seed=8000))


def test_corrupt_own_probe_digest_still_refuses_with_optional_scope(monkeypatch):
    owner = _cost_owner(monkeypatch)
    rows = anchor_rows()
    rows[f"{FAMILY}_R768"]["probe_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="probe identity"):
        owner.chord_cost(f"{FAMILY}_R896", unit="u", costs=rows)


def test_fused_quantity_is_not_scaled_after_member_prices(monkeypatch):
    from types import SimpleNamespace
    from prismaquant.allocator_solver import Candidate
    fmt = f"{FAMILY}_R896"
    spec = registry.get_format(fmt)
    units = ("blk.gate_proj", "blk.up_proj")
    stats, costs, supplied = {}, {}, {}
    for unit in units:
        scope = copy.deepcopy(anchor_rows(currency="served_kl")[f"{FAMILY}_R768"]["quality_scope"])
        scope.update(unit=unit, format=fmt)
        row = {"predicted_dloss": 0.75, "quality_scope": scope}
        stats[unit] = {"out_features": 64, "in_features": 128, "n_params": 8192,
                       "h_trace": 9.0, "unit_structure": "dense"}
        costs[unit] = {fmt: row}
        memory, identity, sidecar = candidates.serialized_candidate_payload(spec, (64, 128), qname=unit)
        supplied[unit] = [Candidate(fmt, 8 * memory / 8192, memory, 0.75,
            serialized_identity=identity, serialized_sidecar_identity=sidecar,
            activation_pricing="qualified_served_kl")]
    profile = SimpleNamespace(fused_sibling_group=lambda name: "pair",
        packed_expert_format_group=lambda name: None)
    grouped_stats, grouped_costs, grouped = candidates.aggregate_fused_siblings(
        stats, costs, [spec], supplied, profile, calibrated_gains={fmt: 9.0},
        activation_pricing=nontrivial_calibration(fmt))
    option = next(c for options in grouped.values() for c in options if c.fmt == fmt)
    assert option.predicted_dloss == 1.5
    group = next(name for name in grouped if name not in units)
    assert candidates.cost_entry_predicted_dloss(grouped_stats[group], grouped_costs[group][fmt],
        gain=9.0, format_name=fmt, activation_pricing=nontrivial_calibration(fmt)) == 1.5



def test_direct_joint_chord_requires_actual_sample_evidence(monkeypatch):
    owner = _cost_owner(monkeypatch)
    rows = anchor_rows()
    lower = rows[f"{FAMILY}_R768"]["quality_scope"]
    upper = rows[f"{FAMILY}_R1024"]["quality_scope"]
    lower.pop("joint_anchor")
    with pytest.raises(ValueError, match="joint_anchor"):
        owner.chord_quality(896, lower_rung=768, upper_rung=1024, lower_value=2.0, upper_value=0.5,
                            lower_scope=lower, upper_scope=upper)


def test_joint_anchor_cannot_relabel_its_actual_currency(monkeypatch):
    owner = _cost_owner(monkeypatch)
    rows = anchor_rows()
    for row in rows.values():
        row["quality_scope"]["currency"] = "served_kl"
        row["quality_scope"]["objective"]["currency"] = "served_kl"
    with pytest.raises(ValueError, match="currency"):
        owner.chord_cost(f"{FAMILY}_R896", unit="u", costs=rows)


def test_recorded_mtp_chord_keeps_producer_only_drift_but_refuses_changed_windows(monkeypatch, capsys):
    from prismaquant.glm_mtp_selection import _recompute_recorded_quality
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    owner = _cost_owner(monkeypatch)
    rows = anchor_rows()
    expected = owner.chord_cost(f"{FAMILY}_R896", unit="u", costs=rows)["canonical_quality"]
    revised = {name: _row("u", name, [value] * 4, complete_probe(producer="e"))
               for name, value in ((f"{FAMILY}_R768", 2.0), (f"{FAMILY}_R1024", 1.0))}
    for row in revised.values():
        row["quality_scope"] = scope_for(row)
    result = _recompute_recorded_quality({"costs": {"u": revised}}, {"u": {f"{FAMILY}_R896": expected}})
    assert result["u"][f"{FAMILY}_R896"]["predicted_dloss"] == 1.25
    assert "[DEV-MODE]" in capsys.readouterr().out
    for name, value in ((f"{FAMILY}_R768", 2.0), (f"{FAMILY}_R1024", 1.0)):
        probe = complete_probe(producer="e")
        probe["token_scope"] = "last"
        row = _row("u", name, [value] * 4, probe)
        row["quality_scope"] = scope_for(row)
        revised[name] = row
    with pytest.raises(ValueError, match="window|coordinate"):
        _recompute_recorded_quality({"costs": {"u": revised}}, {"u": {f"{FAMILY}_R896": expected}})



def test_missing_actual_probe_objective_cannot_be_hidden_by_scope(monkeypatch):
    owner = _cost_owner(monkeypatch)
    rows = {}
    for rate, value in ((768, 2.0), (1024, 1.0)):
        probe = complete_probe()
        probe.pop("objective")
        name = f"{FAMILY}_R{rate}"
        row = _row("u", name, [value] * 4, probe)
        row["quality_scope"] = scope_for(row)
        rows[name] = row
    with pytest.raises(ValueError, match="objective"):
        owner.chord_cost(f"{FAMILY}_R896", unit="u", costs=rows)

