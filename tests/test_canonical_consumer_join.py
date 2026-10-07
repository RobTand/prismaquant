"""Canonical v3 consumer-join mechanics through the staged owning producer.

Synthetic v3 tables prove the join only: scoped admission, canonical timing,
qualified chords and seam threading for body, MTP and final assignments.
Nothing here is measured qualification. The staged producer file arrives
through ``TESSERA_RUNG_ALLOWABILITY_MODULE`` and its SHA pin and is never
vendored; without it these tests skip.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import MappingProxyType

import pytest

from prismaquant.rung_allowability import RungAllowability, RungAllowabilityError

V3FIX = Path(__file__).parent / "fixtures" / "rung_allowability_v3"
FAMILY = "TESSERA_E4M3_K1"
BUILD_ID = "fixture-v3"
DENSE_SHAPE = {"kernel_kind": "dense", "rows": 4, "columns": 8}


def _needs_producer(monkeypatch):
    if not os.environ.get("TESSERA_RUNG_ALLOWABILITY_MODULE"):
        pytest.skip("staged canonical producer is absent")
    if not os.environ.get("TESSERA_RUNG_ALLOWABILITY_MODULE_SHA256"):
        pytest.skip("staged canonical producer pin is absent")
    from prismaquant import rung_allowability
    from _rung_allowability_producer import ExternalProducer
    producer = ExternalProducer()
    monkeypatch.setattr(rung_allowability, "_producer_api", lambda: producer)
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


def test_wrong_shape_waits_and_missing_structure_stays_unscoped(monkeypatch):
    whole = _owner("v3-whole.json", monkeypatch)
    wrong = {"kernel_kind": "dense", "rows": 7, "columns": 7, "m": 8}
    assert "unmeasured_shape_or_M_scope" in whole.refusal(1024, scope=wrong)
    # No structure evidence: the historical whole-table verdict stands.
    assert whole.allows(1024, scope={"rows": 4, "columns": 8})
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
        _first_row(table, 1024)["supported"] = False
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
        table, producer, MappingProxyType({1024: True}), frozenset(1024,))
    assert owner.canonical_time(1024, cell_id="dense-1")["reason"] == "canonical_timing_requires_v3_table"
    assert owner.cells_for(kernel_kind="dense", rows=1, columns=1, m=1) == ()
    assert owner.allows(1024, scope=dict(DENSE_SHAPE))


def _anchor(unit="u", currency="served_kl", validated=True, **extra):
    return {"unit": unit, "currency": currency, "validated": validated, **extra}


def test_chord_is_exact_between_qualified_anchors(monkeypatch):
    half = _owner("v3-half.json", monkeypatch)
    result = half.chord_quality(1024, lower_rung=896, upper_rung=1152,
                                lower_value=1.0, upper_value=0.5,
                                lower_scope=_anchor(), upper_scope=_anchor())
    assert result["value"] == 0.75
    assert result["fraction"] == 0.5
    assert result["anchors"] == [896, 1152]
    assert result["provenance"] == {"format": FAMILY, "unit": "u", "currency": "served_kl"}


@pytest.mark.parametrize("lower,upper,match", [
    (_anchor(validated=False), _anchor(), "not validated"),
    (_anchor(currency="weight_sse"), _anchor(), "currencies"),
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


def test_canonical_times_and_reconcile_stay_advisory(monkeypatch):
    from prismaquant import shape_runtime_prices as prices
    whole = _owner("v3-whole.json", monkeypatch)
    canonical = prices.canonical_rung_times(whole, 1024, kernel_kind="dense",
                                            rows=4, columns=8, m=8)
    assert canonical["cells"] == ["dense:c0:M8"]
    assert canonical["times"][0]["status"] == "measured"
    report = prices.reconcile_canonical_time(canonical, 0.011)
    assert report["compared"][0]["verdict"] == "compared"
    assert report["compared"][0]["rel_diff"] == pytest.approx(0.1)
    unpriced = prices.reconcile_canonical_time(canonical, None)
    assert unpriced["compared"][0]["verdict"] == "unpriced_scope"
    timing = _owner("v3-timing.json", monkeypatch)
    dense_measurement = next(m for m in _first_row(timing._table, 896)["measurements"]
                             if m["cell_id"] == "dense:c0:M8")
    derived = timing.canonical_time(898, cell_id="dense:c0:M8",
                                    class_identity=timing.class_identity(896, dense_measurement))
    manual = {"format": FAMILY, "rung": 898,
              "scope": {"kernel_kind": "dense", "rows": 4, "columns": 8,
                        "m": 8, "routing": None},
              "cells": ["dense:c0:M8"], "times": [derived]}
    assert prices.reconcile_canonical_time(manual, 0.012)["compared"][0]["verdict"] == "class_derived"


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
    whole = _owner("v3-whole.json", monkeypatch)
    monkeypatch.setattr(registry, "format_is_producer_eligible", lambda *a, **k: True)
    eligible = allocator._mtp_rung_attestation(
        None, None, rung_allowability={FAMILY: whole}, target_profile="research")
    assert eligible("mtp.unit", f"{FAMILY}_R1024")
    assert not eligible("mtp.unit", f"{FAMILY}_R1152")


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
    built = candidates.build_candidates(stats, costs, [spec], target_profile="research",
                                        context_by_unit=contexts,
                                        rung_allowability={FAMILY: whole})
    assert set(built) == {"good"}


def test_t4_family_waits_without_invented_admission(monkeypatch):
    t4 = _owner("v3-t4wait.json", monkeypatch, format="TESSERA_E2M1_K2")
    assert t4.refusal(128) != ""
    assert not t4.allows(128, scope={"kernel_kind": "dense", "rows": 1,
                                     "columns": 1, "m": 8})
