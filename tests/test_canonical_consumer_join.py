"""Canonical consumer mechanics with a declared pure metadata dependency.

Synthetic tables prove source integration only. They qualify no speed,
weight error, served loss or serving frontier.
"""
from __future__ import annotations

import copy
import json

from pathlib import Path
from types import MappingProxyType

import pytest

from prismaquant.rung_allowability import RungAllowability, RungAllowabilityError

V3FIX = Path(__file__).parent / "fixtures" / "rung_allowability_v3"
FAMILY = "TESSERA_E4M3_K1"
BUILD_ID = "fixture-v3"
DENSE_SHAPE = {"kernel_kind": "dense", "rows": 4, "columns": 8}


def _needs_producer(monkeypatch):
    from prismaquant import rung_allowability
    from _rung_allowability_producer import ExternalProducer
    producer = ExternalProducer()
    monkeypatch.setattr(rung_allowability, "_producer_api", lambda: producer)
    monkeypatch.setenv("PRISMAQUANT_TESSERA_MENU", "research")
    return producer


def _owner(name, monkeypatch, *, format=FAMILY, rule=None):
    producer = _needs_producer(monkeypatch)
    table = json.loads((V3FIX / name).read_text())
    if rule is None:
        rule = {row["rung"]: True for row in table["rungs"]}
    return RungAllowability(
        format, MappingProxyType(dict(table["kernel_build"])),
        table["table_version"], str(V3FIX / name), table, producer,
        MappingProxyType(dict(rule)),
        frozenset(row["rung"] for row in table["rungs"]))


def _mutated(owner, change):
    table = copy.deepcopy(owner._table)
    change(table)
    table["geometry_classes"] = owner._producer.measured_geometry_classes(table)
    return RungAllowability(
        owner.format, owner.kernel_build, owner.table_version,
        owner.source_path, table, owner._producer, owner._rule, owner._rungs)


def test_whole_and_half_rungs_admit_unscoped_and_scoped(monkeypatch):
    whole = _owner("v3-whole.json", monkeypatch)
    assert whole.allows(1024)
    assert whole.allows(1024, scope=dict(DENSE_SHAPE))
    half = _owner("v3-half.json", monkeypatch)
    assert half.allows(896)
    assert half.allows(896, scope=dict(DENSE_SHAPE))


def test_unlisted_rungs_wait_or_follow_the_menu(monkeypatch):
    whole = _owner("v3-whole.json", monkeypatch)
    assert "missing_actual_measurement" in whole.refusal(768)
    # Outside the owning menu is excluded, never an option; R1280 stays a
    # pricing anchor only.
    assert "outside_performant_menu" in whole.refusal(1280)
    assert not whole.allows(1280, scope=dict(DENSE_SHAPE))


def test_listed_fractional_outside_menu_is_excluded(monkeypatch):
    menu = _owner("v3-menu.json", monkeypatch)
    assert menu.allows(1024)
    assert "outside_performant_menu" in menu.refusal(1152)
    assert "outside_performant_menu" in menu.refusal(1152, scope=dict(DENSE_SHAPE))


def test_wrong_shape_and_unresolved_explicit_scope_wait(monkeypatch):
    whole = _owner("v3-whole.json", monkeypatch)
    wrong = {"kernel_kind": "dense", "rows": 7, "columns": 7, "m": 8}
    assert "unmeasured_shape_or_M_scope" in whole.refusal(1024, scope=wrong)
    assert not whole.allows(1024, scope={"rows": 4, "columns": 8})
    assert whole.allows(1024, scope={"kernel_kind": "dense"})
    assert whole.cells_for(kernel_kind="dense", rows=4, columns=8, m=8) == ("dense:c0:M8",)
    assert whole.cells_for(kernel_kind="dense", rows=7, columns=7, m=8) == ()


def test_routed_scope_never_borrows_dense_cells(monkeypatch):
    whole = _owner("v3-whole.json", monkeypatch)
    assert whole.cells_for(kernel_kind="routed", rows=4, columns=8, m=8) == ("routed:c0:M8",)
    assert whole.allows(1024, scope={"kernel_kind": "routed", "rows": 4,
                                     "columns": 8, "m": 8})


def test_v11_rule_still_intersects_the_owner(monkeypatch):
    whole = _owner("v3-whole.json", monkeypatch, rule={})
    assert whole.refusal(1024) == "excluded by the published allowable_rungs rule"


def test_scope_spelling_refuses(monkeypatch):
    whole = _owner("v3-whole.json", monkeypatch)
    with pytest.raises(RungAllowabilityError, match="kernel_kind|structure"):
        whole.refusal(1024, scope={"kernel_kind": "tensor"})
    with pytest.raises(RungAllowabilityError, match="positive integer"):
        whole.refusal(1024, scope={"kernel_kind": "dense", "rows": 0,
                                   "columns": 8, "m": 8})
    with pytest.raises(RungAllowabilityError, match="unknown.*axis"):
        whole.refusal(1024, scope={"kernel_kind": "dense", "qubits": 4})
    with pytest.raises(RungAllowabilityError, match="mapping"):
        whole.refusal(1024, scope=["dense"])


def _first_row(table, rung):
    return next(row for row in table["rungs"] if row["rung"] == rung)


def test_holds_and_source_refusals_survive_scoping(monkeypatch):
    whole = _owner("v3-whole.json", monkeypatch)

    def hold(table):
        row = _first_row(table, 1024)
        row["anomaly_flags"] = ["quality_failure"]
        row["quality"]["anomaly_flags"] = ["quality_failure"]
    assert "recorded_correctness_hold" in _mutated(whole, hold).refusal(
        1024, scope=dict(DENSE_SHAPE))

    def unsupported(table):
        row = _first_row(table, 1024)
        row["supported"] = False
        row["measurement_status"] = "unsupported"
    assert "recorded_source_refusal" in _mutated(whole, unsupported).refusal(1024)

    def pending(table):
        row = _first_row(table, 1024)
        row["measurement_status"] = "pending"
        row["measurements"] = []
    assert "missing_actual_measurement" in _mutated(whole, pending).refusal(1024)


def test_canonical_time_measured_inherited_and_wait(monkeypatch):
    timing = _owner("v3-timing.json", monkeypatch)
    measured = timing.canonical_time(896, cell_id="dense:c0:M8")
    assert measured["status"] == "measured"
    assert measured["measurement"]["kernel_time_us"] == 10.0
    assert measured["provenance"]["table_version"] == 1
    assert measured["numerical_qualification_inherited"] is False
    assert measured["serving_qualification_inherited"] is False
    dense_measurement = next(m for m in _first_row(timing._table, 896)["measurements"]
                             if m["cell_id"] == "dense:c0:M8")
    identity = timing.class_identity(896, dense_measurement)
    derived = timing.canonical_time(898, cell_id="dense:c0:M8", class_identity=identity)
    assert derived["status"] == "inherited"
    assert derived["kernel_time_us"] == 12.0
    assert derived["anchors"] == [896, 897]
    assert derived["menu_admitted"] is False
    waiting = timing.canonical_time(898, cell_id="dense:c0:M8")
    assert waiting["status"] == "wait"


def test_prev3_tables_wait_for_canonical_timing_but_keep_row_admission(monkeypatch):
    producer = _needs_producer(monkeypatch)
    path = Path(__file__).parent / "fixtures" / "rung_allowability" / FAMILY / "fixture-t8" / "v0001.json"
    table = json.loads(path.read_text())
    owner = RungAllowability(
        FAMILY, MappingProxyType(dict(table["kernel_build"])), 1, str(path),
        table, producer, MappingProxyType({1024: True}), frozenset((1024,)))
    assert owner.canonical_time(1024, cell_id="dense-1")["reason"] == "canonical_timing_requires_v3_table"
    assert owner.cells_for(kernel_kind="dense", rows=1, columns=1, m=1) == ()
    assert owner.allows(1024)
    assert not owner.allows(1024, scope=dict(DENSE_SHAPE))


def _anchor(unit="u", currency="served_kl", validated=True, **extra):
    return {"unit": unit, "currency": currency, "validated": validated,
            "calibration": "fixture-calibration", **extra}


def test_chord_is_exact_between_qualified_anchors(monkeypatch):
    half = _owner("v3-half.json", monkeypatch)
    result = half.chord_quality(896, lower_rung=768, upper_rung=1024,
                                lower_value=1.0, upper_value=0.5,
                                lower_scope=_anchor(), upper_scope=_anchor())
    assert result["value"] == 0.75
    assert result["fraction"] == 0.5
    assert result["anchors"] == [768, 1024]
    assert result["provenance"] == {"format": FAMILY, "unit": "u", "currency": "served_kl"}


@pytest.mark.parametrize("lower,upper,match", [
    (_anchor(validated=False), _anchor(), "not validated"),
    (_anchor(currency="weight_sse"), _anchor(), "currency"),
    (_anchor(unit="v"), _anchor(), "units"),
    (_anchor(calibration="a"), _anchor(calibration="b"), "calibrations"),
    (_anchor(), {**_anchor(), "family": "TESSERA_BF16_K1"}, "another format"),
])
def test_chord_withholds_unqualified_anchors(monkeypatch, lower, upper, match):
    half = _owner("v3-half.json", monkeypatch)
    with pytest.raises(RungAllowabilityError, match=match):
        half.chord_quality(1024, lower_rung=896, upper_rung=1152,
                           lower_value=1.0, upper_value=0.5,
                           lower_scope=lower, upper_scope=upper)


def test_chord_withholds_bad_values_and_intervals(monkeypatch):
    half = _owner("v3-half.json", monkeypatch)
    with pytest.raises(RungAllowabilityError, match="number"):
        half.chord_quality(1024, lower_rung=896, upper_rung=1152,
                           lower_value=True, upper_value=0.5,
                           lower_scope=_anchor(), upper_scope=_anchor())
    with pytest.raises(RungAllowabilityError, match="interval"):
        half.chord_quality(897, lower_rung=896, upper_rung=897,
                           lower_value=1.0, upper_value=0.5,
                           lower_scope=_anchor(), upper_scope=_anchor())


def test_canonical_times_keep_raw_kernel_provenance(monkeypatch):
    from prismaquant import shape_runtime_prices as prices
    whole = _owner("v3-whole.json", monkeypatch)
    canonical = prices.canonical_rung_times(whole, 1024, kernel_kind="dense",
                                            rows=4, columns=8, m=8)
    assert canonical["cells"] == ["dense:c0:M8"]
    assert canonical["times"][0]["status"] == "measured"
    assert canonical["times"][0]["measurement"]["kernel_time_us"] == 10.0
    assert canonical["times"][0]["numerical_qualification_inherited"] is False


def test_candidate_seam_threads_scope(monkeypatch):
    from prismaquant import allocator_candidates as candidates
    whole = _owner("v3-whole.json", monkeypatch)
    rung_allowability = {FAMILY: whole}
    name = f"{FAMILY}_R1024"
    assert candidates.candidate_rung_admission(
        name, target_profile="research", rung_allowability=rung_allowability,
        allowability_scope=dict(DENSE_SHAPE)).admits("research")
    assert not candidates.candidate_rung_admission(
        name, target_profile="research", rung_allowability=rung_allowability,
        allowability_scope={"kernel_kind": "dense", "rows": 7, "columns": 7,
                            "m": 8}).admits("research")
    assert candidates.candidate_rung_admission(
        name, target_profile="research",
        rung_allowability=rung_allowability).admits("research")


def test_final_check_threads_per_unit_scope(monkeypatch):
    from prismaquant import allocator_candidates as candidates
    whole = _owner("v3-whole.json", monkeypatch)
    assignment = {"u": f"{FAMILY}_R1024"}
    candidates.require_assignment_rung_allowability(
        assignment, target_profile="research",
        rung_allowability={FAMILY: whole},
        allowability_scope_by_unit={"u": dict(DENSE_SHAPE)})
    with pytest.raises(ValueError, match="not allocation-eligible"):
        candidates.require_assignment_rung_allowability(
            assignment, target_profile="research",
            rung_allowability={FAMILY: whole},
            allowability_scope_by_unit={"u": {"kernel_kind": "dense", "rows": 7,
                                              "columns": 7, "m": 8}})


def test_mtp_uses_the_same_scoped_input(monkeypatch):
    from prismaquant import allocator, format_registry as registry
    from types import SimpleNamespace
    whole = _owner("v3-whole.json", monkeypatch)
    monkeypatch.setattr(registry, "format_is_producer_eligible", lambda *a, **k: True)
    target = SimpleNamespace(context=lambda structure: _serving_context(structure))
    profile = SimpleNamespace(structure_spec=lambda: {}, packed_expert_format_group=lambda name: None,
        per_expert_moe_regex=lambda: None, per_expert_mtp_regex=lambda: None)
    costs = {"mtp.unit": {f"{FAMILY}_R1024": {"joint_operator_identity":
        {"source_weight": {"shape": [4, 8]}}}}}
    eligible = allocator._mtp_rung_attestation(target, profile,
        rung_allowability={FAMILY: whole}, target_profile="research", costs=costs, m=8)
    assert eligible("mtp.unit", f"{FAMILY}_R1024")
    assert not eligible("mtp.unit", f"{FAMILY}_R1152")
    wrong = allocator._mtp_rung_attestation(target, profile,
        rung_allowability={FAMILY: whole}, target_profile="research", costs=costs, m=9)
    assert not wrong("mtp.unit", f"{FAMILY}_R1024")


def _serving_context(structure):
    from prismaquant.lane_eligibility import ServingContext
    return ServingContext(platform="sm_121", structure=structure, residency="resident",
                          runtime_image="localhost/test@sha256:" + "0" * 64,
                          execution_mode="eager")


def test_body_builder_scopes_per_unit_shape(monkeypatch):
    from prismaquant import allocator_candidates as candidates, format_registry as registry
    whole = _owner("v3-whole.json", monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_TESSERA_MENU", "research")
    monkeypatch.setattr(candidates, "check_stats_format_applicability",
                        lambda *a, **k: candidates.FormatApplicability(True))
    spec = registry.get_format(f"{FAMILY}_R1024")
    stats = {"good": {"in_features": 8, "out_features": 4, "n_params": 32},
             "bad": {"in_features": 7, "out_features": 7, "n_params": 49}}
    costs = {name: {spec.name: {"predicted_dloss": 0.1}} for name in stats}
    contexts = {name: _serving_context("dense") for name in stats}
    built = candidates.build_candidates({"good": stats["good"]}, costs, [spec], target_profile="research",
                                        context_by_unit=contexts,
                                        rung_allowability={FAMILY: whole})
    assert set(built) == {"good"}
    with pytest.raises(ValueError, match="no serving-eligible"):
        candidates.build_candidates(stats, costs, [spec], target_profile="research",
            context_by_unit=contexts, rung_allowability={FAMILY: whole})


def test_t4_family_waits_without_invented_admission(monkeypatch):
    t4 = _owner("v3-t4wait.json", monkeypatch, format="TESSERA_E2M1_K2")
    assert t4.refusal(128) != ""
    assert not t4.allows(128, scope={"kernel_kind": "dense", "rows": 1,
                                     "columns": 1, "m": 8})


@pytest.mark.parametrize("scope", [
    {"kernel_kind": "dense", "rows": 4},
    {"kernel_kind": "dense", "rows": 4, "columns": 8, "routing": "none"},
    {"activation_contract": "wrong"},
    {"recipe": {"body": "wrong"}},
])
def test_explicit_unresolved_scope_never_broadens(monkeypatch, scope):
    from prismaquant.allocator_candidates import candidate_rung_admission
    whole = _owner("v3-whole.json", monkeypatch)
    admission = candidate_rung_admission(f"{FAMILY}_R1024", target_profile="research",
        rung_allowability={FAMILY: whole}, allowability_scope=scope)
    assert not admission.admits("research")


def test_legacy_producer_needs_only_original_api(tmp_path, monkeypatch):
    import shutil
    import sys
    import types
    from prismaquant import rung_allowability as consumer
    from prismaquant.lane_eligibility import load_published_formats
    from prismaquant.tessera_runtime_contract import contract_path
    from _rung_allowability_producer import ExternalProducer
    original = ExternalProducer()
    legacy = types.SimpleNamespace(**{name: getattr(original, name) for name in
        ("validate_index", "validate_table", "admit_rung")})
    import tessera
    monkeypatch.setattr(tessera, "rung_allowability", legacy, raising=False)
    monkeypatch.setitem(sys.modules, "tessera.rung_allowability", legacy)
    root = tmp_path / "publication"
    shutil.copytree(Path(__file__).parent / "fixtures" / "rung_allowability", root)
    table = json.loads((root / FAMILY / "fixture-t8" / "v0001.json").read_text())
    owner = consumer.load_rung_allowability(root,
        format_entry=load_published_formats(contract_path=contract_path())[FAMILY],
        expected_kernel_build=table["kernel_build"])
    assert owner.allows(1024)
    assert not owner.allows(1026)


def _cost_owner(monkeypatch):
    owner = _owner("v3-timing.json", monkeypatch)
    whole = _owner("v3-whole.json", monkeypatch)
    table = copy.deepcopy(owner._table)
    table["scope"]["rung_max"] = 1024
    pending = _first_row(table, 898)
    table["rungs"].extend({**copy.deepcopy(pending), "rung": rung} for rung in range(899, 1024))
    table["rungs"].append(copy.deepcopy(_first_row(whole._table, 1024)))
    table["geometry_classes"] = owner._producer.measured_geometry_classes(table)
    return RungAllowability(owner.format, owner.kernel_build, 1, "synthetic cost mechanics",
        table, owner._producer, MappingProxyType({row["rung"]: True for row in table["rungs"]}),
        frozenset(row["rung"] for row in table["rungs"]))


def _body_costs(unit="u"):
    return {unit: {
        f"{FAMILY}_R768": {"predicted_dloss": 1.0, "quality_scope": _anchor(unit=unit)},
        f"{FAMILY}_R1024": {"predicted_dloss": 0.5, "quality_scope": _anchor(unit=unit)},
        f"{FAMILY}_R896": {"predicted_dloss": 999.0}}}


def test_body_prices_the_qualified_chord_and_exact_fractional_bytes(monkeypatch):
    from prismaquant import allocator_candidates as candidates, format_registry as registry
    owner = _cost_owner(monkeypatch)
    monkeypatch.setattr(candidates, "check_stats_format_applicability",
                        lambda *a, **k: candidates.FormatApplicability(True))
    stats = {"u": {"in_features": 8, "out_features": 4, "n_params": 32, "unit_structure": "dense"}}
    specs = [registry.get_format(f"{FAMILY}_R{rung}") for rung in (896, 1024)]
    built = candidates.build_candidates(stats, _body_costs(), specs, target_profile="research",
        rung_allowability={FAMILY: owner}, allowability_m=8)
    by_format = {item.fmt: item for item in built["u"]}
    half = by_format[f"{FAMILY}_R896"]
    whole = by_format[f"{FAMILY}_R1024"]
    assert half.predicted_dloss == 0.75
    assert type(half.memory_bytes) is int and half.memory_bytes < whole.memory_bytes
    assert half.memory_bytes == candidates.serialized_candidate_payload(specs[0], (4, 8), qname="u")[0]
    scope = stats["u"]["_rung_allowability_scope_by_format"][half.fmt]
    assert (scope["rows"], scope["columns"], scope["m"]) == (4, 8, 8)
    assert scope["activation_contract"] == owner.kernel_build["activation_contract"]
    assert scope["recipe"] == _first_row(owner._table, 896)["measurements"][0]["geometry"]["recipe"]
    assert stats["u"]["_canonical_quality_by_format"][half.fmt]["numerical_qualification_inherited"] is False
    with pytest.raises(ValueError, match="no serving-eligible"):
        candidates.build_candidates(stats, _body_costs(), specs, target_profile="research",
            rung_allowability={FAMILY: owner}, allowability_m=9)


def test_final_scope_retains_rank_local_shape_m_activation_and_recipe(monkeypatch):
    from prismaquant import allocator, allocator_candidates as candidates
    owner = _owner("v3-whole.json", monkeypatch)
    unit = "expert.gate_proj"
    assignment = {unit: f"{FAMILY}_R1024"}
    stats = {unit: {"out_features": 8, "in_features": 8, "n_params": 64, "unit_structure": "dense"}}
    scopes = allocator._final_allowability_scopes(assignment, None, stats, {FAMILY: owner},
        m=8, tensor_parallel=2)
    assert (scopes[unit]["rows"], scopes[unit]["columns"], scopes[unit]["m"]) == (4, 8, 8)
    candidates.require_assignment_rung_allowability(assignment, target_profile="research",
        rung_allowability={FAMILY: owner}, allowability_scope_by_unit=scopes)
    scopes[unit]["activation_contract"] = "wrong"
    with pytest.raises(ValueError, match="activation"):
        candidates.require_assignment_rung_allowability(assignment, target_profile="research",
            rung_allowability={FAMILY: owner}, allowability_scope_by_unit=scopes)


def _shape_price_fixture():
    from dataclasses import replace
    from conftest import set_cell_census
    from prismaquant import lane_eligibility as lane, shape_runtime_prices as prices
    from test_shape_runtime_prices import (_payload, _doc, _row, DENSE_E4M3, SCOPE, COMMIT, SHA)
    payload = _payload()
    for cell in payload["lane_eligibility"]["cells"]:
        if cell["family"] == FAMILY and cell["structure"] == "dense":
            set_cell_census(payload, cell, [896, 897, 1024])
    eligibility = lane._parse_table(payload["lane_eligibility"], payload["formats"], "", COMMIT, SHA,
        native_extensions=payload["native_extensions"])
    doc = _doc([_row("dense", "4x8", FAMILY, 897, 8, DENSE_E4M3, (0.023, 0.024, 0.025)),
                _row("dense", "4x8", FAMILY, 1024, 8, DENSE_E4M3, (0.049, 0.05, 0.051))],
               regimes=(8,), tensor_parallel=1)
    table = prices.admit_shape_table(prices.parse_shape_table(doc),
        scope=replace(SCOPE, tensor_parallel=1), eligibility=eligibility)
    return table, {row["family"]: row for row in payload["formats"]}


def test_actual_pact_resources_use_reconciled_class_costs_and_exact_budget(monkeypatch):
    from prismaquant import allocator_candidates as candidates, format_registry as registry
    from prismaquant import shape_runtime_prices as prices, pact_hull
    owner = _cost_owner(monkeypatch)
    monkeypatch.setattr(candidates, "check_stats_format_applicability",
                        lambda *a, **k: candidates.FormatApplicability(True))
    stats = {"u": {"in_features": 8, "out_features": 4, "n_params": 32, "unit_structure": "dense"}}
    specs = [registry.get_format(f"{FAMILY}_R{rung}") for rung in (896, 1024)]
    built = candidates.build_candidates(stats, _body_costs(), specs, target_profile="research",
        rung_allowability={FAMILY: owner}, allowability_m=8)
    table, formats = _shape_price_fixture()
    members = {("u", option.fmt): {"u": option.fmt} for option in built["u"]}
    pricing = prices.build_shape_runtime_resources(table, built, option_members=members,
        member_shapes={"u": (4, 8)}, member_structure={"u": "dense"}, regime_m=8,
        published_formats=formats, rung_allowability={FAMILY: owner})
    half_key = ("u", f"{FAMILY}_R896")
    assert pricing.resources[half_key].prefill_ms == pytest.approx(0.02)
    assert pricing.prefill[half_key].source == "canonical_class"
    assert pricing.prefill[half_key].canonical["serving_qualification_inherited"] is False
    assert pricing.prefill[half_key].canonical["canonical_target"][0]["kernel_time_us"] == 10.0
    half_bytes = next(c.memory_bytes for c in built["u"] if c.fmt == half_key[1])
    budget = half_bytes + 1
    times = {key: value.prefill_ms for key, value in pricing.resources.items()}
    menu = pricing.time_candidates(built)
    hull = pact_hull.dichotomic_lower_hull(menu, times, max_memory_bytes=budget)
    assert all(dict(point.assignment) == {"u": half_key[1]} for point in hull.vertices)
    assert all(point.memory_bytes == half_bytes for point in hull.vertices)
    assert budget - hull.vertices[0].memory_bytes == 1
    assert pricing.gap_report()["canonical_prices"]
    def donor_hold(table):
        row = _first_row(table, 897)
        row["anomaly_flags"] = row["quality"]["anomaly_flags"] = ["quality_failure"]
    held = _mutated(owner, donor_hold)
    held_table = prices.build_shape_runtime_resources(table, built, option_members=members,
        member_shapes={"u": (4, 8)}, member_structure={"u": "dense"}, regime_m=8,
        published_formats=formats, rung_allowability={FAMILY: held})
    assert half_key not in held_table.resources


def test_real_surface_prediction_uses_the_qualified_neighbour_chord(monkeypatch):
    from prismaquant.tessera_rate_surface import TesseraRateSurface
    owner = _owner("v3-half.json", monkeypatch)
    surface = TesseraRateSurface("u", FAMILY, "tight_offset", "served_kl", (768, 1024),
        (1.0, 0.5), (0.1, 0.1), allowability=owner,
        anchor_scopes={768: _anchor(), 1024: _anchor()})
    assert surface.predict(896) == 0.75
    assert surface.predict(768) == 1.0
    with pytest.raises(RungAllowabilityError, match="screen"):
        owner.chord_cost(f"{FAMILY}_R896", unit="u", costs={
            f"{FAMILY}_R768": {"predicted_dloss": 1.0, "quality_scope": _anchor(currency="weight_sse")},
            f"{FAMILY}_R1024": {"predicted_dloss": 0.5, "quality_scope": _anchor(currency="weight_sse")}})



def test_independent_mtp_selector_uses_chord_prices_and_bound_wire_bytes(monkeypatch):
    from prismaquant.glm_mtp_selection import select_mtp_rungs, _recompute_recorded_quality
    from test_glm_mtp_selection import _row, _probe, ROUTED, PARAMS, CONSTANTS
    owner = _cost_owner(monkeypatch)
    unit = ROUTED[0]
    probe = _probe()
    rows = {f"{FAMILY}_R{rate}": _row(unit, f"{FAMILY}_R{rate}", [value] * 4, probe)
            for rate, value in ((768, 2.0), (896, 10.0), (1024, 1.0))}
    wire = {f"{FAMILY}_R{rate}": value for rate, value in ((768, 400), (896, 500), (1024, 600))}
    payload = {"schema": "prismaquant.glm_mtp_cost.v1", "mtp_layer": 45,
        "groups": {"g": [unit]}, "params": {unit: PARAMS}, "source_dtype": {unit: "bfloat16"},
        "costs": {unit: rows}, "wire_bytes": {unit: wire}}
    record = select_mtp_rungs(payload, byte_budget=501, constants=CONSTANTS,
        formats=[f"{FAMILY}_R896", f"{FAMILY}_R1024"], rung_allowability={FAMILY: owner})
    assert record["assignment"] == {unit: f"{FAMILY}_R896"}
    assert record["E"] == 1.25
    assert record["resident_bytes"] == 500
    assert record["byte_budget"] - record["resident_bytes"] == 1
    quality = record["canonical_quality"]
    assert _recompute_recorded_quality(payload, quality)[unit][f"{FAMILY}_R896"]["predicted_dloss"] == 1.25
    quality[unit][f"{FAMILY}_R896"]["value"] = 2.0
    with pytest.raises(ValueError, match="actual bound anchors"):
        _recompute_recorded_quality(payload, quality)


def test_canonical_densify_keeps_fractional_bytes_and_withholds_unadmitted_rates(monkeypatch):
    from prismaquant.tessera_rate_surface import TesseraRateSurface, densify_rate_surface
    owner = _owner("v3-half.json", monkeypatch)
    def actual_shape(table):
        for shape in table["scope"]["shapes"]:
            shape["rows"], shape["columns"] = 64, 256
        for row in table["rungs"]:
            for measurement in row["measurements"]:
                measurement["evidence"]["rows"], measurement["evidence"]["columns"] = 64, 256
    owner = _mutated(owner, actual_shape)
    surface = TesseraRateSurface("u", FAMILY, "tight", "served_kl", (768, 1024),
        (1.0, 0.5), (0.1, 0.1), allowability=owner,
        anchor_scopes={768: _anchor(), 1024: _anchor()})
    built = densify_rate_surface(surface, (64, 256), q256_values=[896, 1024, 1280],
        allowability_scope={"kernel_kind": "dense", "m": 8})
    assert [candidate.body_rate_q256 for candidate in built] == [896]
    assert built[0].predicted_dloss_mean == 0.75
    assert type(built[0].memory_bytes) is int



def test_missing_qualified_chord_never_drops_the_actual_unit(monkeypatch):
    from prismaquant import allocator_candidates as candidates, format_registry as registry
    owner = _cost_owner(monkeypatch)
    monkeypatch.setattr(candidates, "check_stats_format_applicability",
                        lambda *a, **k: candidates.FormatApplicability(True))
    stats = {"u": {"in_features": 8, "out_features": 4, "n_params": 32, "unit_structure": "dense"}}
    fmt = f"{FAMILY}_R896"
    with pytest.raises(ValueError, match="canonical quality waits"):
        candidates.build_candidates(stats, {"u": {fmt: {"predicted_dloss": 5.0}}},
            [registry.get_format(fmt)], target_profile="research", rung_allowability={FAMILY: owner})


def test_explicit_candidate_index_uses_actual_schema_without_active_index_changes(tmp_path, monkeypatch):
    import shutil
    from prismaquant.rung_allowability import load_rung_allowability
    from test_rung_allowability import FIXTURE, _formats
    _needs_producer(monkeypatch)
    root = tmp_path / "publication"
    shutil.copytree(FIXTURE, root)
    active = (root / "index.json").read_bytes()
    table = json.loads((V3FIX / "v3-timing.json").read_text())
    (root / "v3-timing.json").write_text(json.dumps(table))
    candidate = root / "index.v3-candidate.json"
    candidate.write_text(json.dumps({"schema": "fleet.rung_allowability.index.v2", "formats": {
        FAMILY: {"kernel_builds": {table["kernel_build"]["id"]: {"current_version": 1,
            "versions": {"1": {"path": "v3-timing.json", "table_schema": table["schema"],
                                 "table_status": table["table_status"]}}}}}}}))
    owner = load_rung_allowability(candidate, format_entry=_formats()[FAMILY],
        expected_kernel_build=table["kernel_build"])
    assert owner.scoped
    assert owner.allows(896, scope={"kernel_kind": "dense", "rows": 4, "columns": 8, "m": 8})
    assert owner.provenance()["index_source_path"] == str(candidate)
    assert (root / "index.json").read_bytes() == active



def test_unknown_parallel_cut_waits_and_invalid_known_cut_refuses(monkeypatch):
    owner = _owner("v3-whole.json", monkeypatch)
    scope = owner.scope_for_unit(f"{FAMILY}_R1024", unit="attention.q_proj",
        shape=(4, 8), structure="dense", m=8, tensor_parallel=2)
    assert not owner.allows(1024, scope=scope)
    with pytest.raises(ValueError, match="not divisible"):
        owner.scope_for_unit(f"{FAMILY}_R1024", unit="expert.gate_proj",
            shape=(5, 8), structure="dense", m=8, tensor_parallel=2)

