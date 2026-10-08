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

@pytest.mark.parametrize("currency,expected", [(joint.JOINT_CURRENCY, 1.75), ("served_kl", 1.25)])
def test_fused_mixed_measured_and_chord_members_keep_complete_prices(monkeypatch, currency, expected):
    from types import SimpleNamespace
    from prismaquant import allocator_solver
    from test_allocator_sibling_aggregation import _installed_fused_licence
    monkeypatch.setattr(allocator_solver, "lane_fused_module_licence", _installed_fused_licence)
    owner = owner_for_rows(monkeypatch)
    formats = [f"{FAMILY}_R{rate}" for rate in (896, 1024)]
    specs = [registry.get_format(name) for name in formats]
    units = ("blk.gate_proj", "blk.up_proj")
    stats, costs = {}, {}
    for unit in units:
        stats[unit] = {"out_features": 64, "in_features": 128, "n_params": 8192,
                       "h_trace": 9.0, "unit_structure": "dense"}
        costs[unit] = {}
        for rate, value in ((768, 2.0), (1024, 1.0)):
            name = f"{FAMILY}_R{rate}"
            row = _row(unit, name, [value] * 4, complete_probe())
            row["quality_scope"] = scope_for(row)
            if currency == "served_kl":
                scope = copy.deepcopy(row["quality_scope"])
                scope["currency"] = scope["objective"]["currency"] = currency
                row = {"predicted_dloss": 1.0 if rate == 768 else 0.5, "quality_scope": scope}
            costs[unit][name] = row
    supplied = candidates.build_candidates(stats, costs, specs, target_profile="research",
        rung_allowability={FAMILY: owner}, allowability_m=8)
    supplied = {unit: [option for option in supplied[unit] if option.fmt == fmt]
                for unit, fmt in zip(units, formats)}
    profile = SimpleNamespace(fused_sibling_group=lambda name: "pair",
        packed_expert_format_group=lambda name: None)
    grouped_stats, grouped_costs, grouped = candidates.aggregate_fused_siblings(
        stats, costs, specs, supplied, profile)
    group, options = next((name, options) for name, options in grouped.items() if name not in units)
    mixed = next(option for option in options if option.member_formats == dict(zip(units, formats)))
    assert mixed.predicted_dloss == expected
    complete = grouped_costs[group][mixed.fmt]
    assert candidates.cost_entry_predicted_dloss(grouped_stats[group], complete,
        gain=9.0, format_name=mixed.fmt, activation_pricing=nontrivial_calibration(formats[0])) == expected
    assert {member["unit"]: member["format"] for member in complete["canonical_members"]} == dict(zip(units, formats))


@pytest.mark.parametrize("anchor_wires", [True, False])
def test_mtp_anchor_only_quality_builds_and_replays_fractional_wire(monkeypatch, anchor_wires):
    from prismaquant.glm_mtp_selection import select_mtp_rungs, _recompute_recorded_quality, _unit_rows
    from test_glm_mtp_selection import _probe, ROUTED, PARAMS, CONSTANTS
    unit = ROUTED[0]
    owner = owner_for_rows(monkeypatch)
    probe = _probe()
    probe.update(calibration_shape=[512, 512], token_scope="all")
    rows = {f"{FAMILY}_R{rate}": _row(unit, f"{FAMILY}_R{rate}", [value] * 4, probe)
            for rate, value in ((768, 2.0), (1024, 1.0))}
    for row in rows.values():
        row["quality_scope"] = scope_for(row)
    half = f"{FAMILY}_R896"
    wire = {half: 500}
    if anchor_wires:
        wire.update({f"{FAMILY}_R768": 400, f"{FAMILY}_R1024": 600})
    payload = {"schema": "prismaquant.glm_mtp_cost.v1", "mtp_layer": 45,
        "groups": {"g": [unit]}, "params": {unit: PARAMS}, "source_dtype": {unit: "bfloat16"},
        "costs": {unit: rows}, "wire_bytes": {unit: wire}}
    original = copy.deepcopy(payload)
    record = select_mtp_rungs(payload, byte_budget=501, constants=CONSTANTS,
        formats=[half], rung_allowability={FAMILY: owner},
        stats={unit: {"unit_structure": "dense"}}, allowability_m=8)
    assert record["assignment"] == {unit: half}
    assert (record["E"], record["resident_bytes"]) == (1.25, 500)
    replayed = _recompute_recorded_quality(payload, record["canonical_quality"])
    export_rows, _ = _unit_rows(payload, quality_prices=replayed)
    assert export_rows[unit][half] == (1.25, 500)
    assert payload == original
    assert half not in payload["costs"][unit]


def test_canonical_densify_does_not_publish_anchor_stderr_as_fractional_uncertainty(monkeypatch):
    from prismaquant.tessera_rate_surface import TesseraRateSurface, densify_rate_surface
    from prismaquant.tessera_allocator import tessera_pareto_frontier, tessera_solver_candidate_menu
    from test_canonical_consumer_join import _owner, _anchor
    owner = _cost_owner(monkeypatch)
    def actual_shape(table):
        for shape in table["scope"]["shapes"]:
            shape["rows"], shape["columns"] = 64, 256
        for row in table["rungs"]:
            for measurement in row["measurements"]:
                measurement["evidence"]["rows"], measurement["evidence"]["columns"] = 64, 256
    owner = _mutated(owner, actual_shape)
    surface = TesseraRateSurface("u", FAMILY, "tight", "served_kl", (768, 1024),
        (1.0, 0.5), (0.1, 0.2), allowability=owner,
        anchor_scopes={768: _anchor(shape=(64, 256)), 1024: _anchor(rate=1024, shape=(64, 256))})
    built = densify_rate_surface(surface, (64, 256), q256_values=[896, 1024],
        allowability_scope={"kernel_kind": "dense", "m": 8})
    half = built[0]
    assert half.predicted_dloss_stderr is None
    assert half.quality_provenance["canonical_quality"]["anchors"] == (768, 1024)
    assert half.quality_provenance["anchor_diagnostics"]["stderr"] == (0.1, 0.2)
    assert tessera_solver_candidate_menu(built)["u"][0].predicted_dloss == 0.75
    assert tessera_pareto_frontier(built, uncertainty_z=0).candidates == tuple(built)
    with pytest.raises(ValueError, match="uncertainty.*unavailable|stderr.*unavailable"):
        tessera_pareto_frontier(built, uncertainty_z=1.96)


@pytest.fixture(params=[joint.JOINT_CURRENCY, "served_kl"])
def canonical_chord_candidate(monkeypatch, request):
    from prismaquant.tessera_rate_surface import TesseraRateSurface, densify_rate_surface
    base_owner = owner_for_rows(monkeypatch)
    def widen(table):
        for shape in table["scope"]["shapes"]:
            shape["rows"], shape["columns"] = 64, 256
        for row in table["rungs"]:
            for measurement in row["measurements"]:
                measurement["evidence"]["rows"], measurement["evidence"]["columns"] = 64, 256
    owner = _mutated(base_owner, widen)
    rows = anchor_rows(currency=request.param)
    for row in rows.values():
        scope = row.get("quality_scope")
        if scope is None:
            continue
        if "joint_operator_identity" in row:
            operator = row["joint_operator_identity"]
            operator["source_weight"]["shape"] = [64, 256]
            operator["source_weight"]["logical_bytes"] = 2 * 64 * 256
            operator["rendered_weight"]["shape"] = [64, 256]
            operator["rendered_weight"]["logical_bytes"] = 4 * 64 * 256
            row["joint_operator_identity_sha256"] = joint.identity_sha256(operator)
            from canonical_quality_fixtures import scope_for as rebuild_scope
            row["quality_scope"] = rebuild_scope(row)
        else:
            scope["shape"] = [64, 256]
            scope["source_weight"]["shape"] = [64, 256]
            scope["source_weight"]["logical_bytes"] = 2 * 64 * 256
    formats = [f"{FAMILY}_R{rate}" for rate in (768, 1024)]
    surface = TesseraRateSurface("u", FAMILY, "tight", request.param, (768, 1024),
        tuple(rows[fmt]["predicted_dloss"] for fmt in formats), (0.1, 0.2),
        allowability=owner, anchor_scopes={rate: rows[fmt]["quality_scope"]
            for rate, fmt in zip((768, 1024), formats)})
    return densify_rate_surface(surface, (64, 256), q256_values=[896],
        allowability_scope={"kernel_kind": "dense", "m": 8})[0]


def _candidate_constructor_fields(candidate):
    record = candidate.as_dict()
    return {"unit_name": candidate.unit_name, "family": candidate.family,
        "body_rate_q256": candidate.body_rate_q256, "layout": candidate.layout,
        "variant_label": candidate.variant_label, "footprint": record["footprint"],
        "predicted_dloss_mean": candidate.predicted_dloss_mean,
        "predicted_dloss_stderr": candidate.predicted_dloss_stderr,
        "servability": candidate.servability, "quality_provenance": record["quality_provenance"]}


@pytest.mark.parametrize("stderr", [0.0, 0.1, 0.2])
def test_canonical_chord_constructor_refuses_unmeasured_stderr(canonical_chord_candidate, stderr):
    from prismaquant.tessera_allocator import TesseraAllocatorCandidate
    from prismaquant.tessera_formats import TesseraFormatError
    fields = _candidate_constructor_fields(canonical_chord_candidate)
    fields["predicted_dloss_stderr"] = stderr
    with pytest.raises(TesseraFormatError, match="canonical chord.*uncertainty.*unavailable"):
        TesseraAllocatorCandidate(**fields)


def test_canonical_chord_point_menu_does_not_claim_a_measured_interval(canonical_chord_candidate):
    from prismaquant.tessera_allocator import tessera_pareto_frontier, tessera_solver_candidate_menu
    from prismaquant.tessera_formats import TesseraFormatError
    candidate = canonical_chord_candidate
    assert candidate.predicted_dloss_stderr is None
    assert candidate.as_dict()["predicted_dloss_stderr"] is None
    assert tessera_solver_candidate_menu([candidate])["u"][0].predicted_dloss == candidate.predicted_dloss_mean
    assert tessera_pareto_frontier([candidate], uncertainty_z=0).candidates == (candidate,)
    with pytest.raises(TesseraFormatError, match="uncertainty.*unavailable"):
        tessera_pareto_frontier([candidate], uncertainty_z=1.96)


@pytest.mark.parametrize("round_trip", ["constructor", "replace"])
def test_canonical_chord_frozen_provenance_round_trip(canonical_chord_candidate, round_trip):
    from dataclasses import replace
    from prismaquant.tessera_allocator import TesseraAllocatorCandidate
    candidate = canonical_chord_candidate
    original = candidate.as_dict()
    if round_trip == "constructor":
        fields = _candidate_constructor_fields(candidate)
        fields["quality_provenance"] = candidate.quality_provenance
        rebuilt = TesseraAllocatorCandidate(**fields)
    else:
        rebuilt = replace(candidate)
    assert rebuilt.as_dict() == original
    assert rebuilt.predicted_dloss_stderr is None
    assert rebuilt.quality_provenance["canonical_quality"]["anchors"] == (768, 1024)
    assert rebuilt.quality_provenance["quality_scope"]["currency"] == candidate.quality_provenance["quality_scope"]["currency"]
    assert rebuilt.quality_provenance["canonical_anchors"] == candidate.quality_provenance["canonical_anchors"]
    with pytest.raises(TypeError):
        rebuilt.quality_provenance["quality_scope"]["source_weight"]["shape"][0] = 1
    serialized = rebuilt.as_dict()
    serialized["quality_provenance"]["quality_scope"]["source_weight"]["shape"][0] = 1
    assert rebuilt.as_dict() == original
    assert candidate.as_dict() == original


