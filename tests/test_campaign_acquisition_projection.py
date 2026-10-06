"""Whole authenticated requests project onto explicit row units, not new rates."""
from __future__ import annotations

import copy
import pickle

import pytest

from prismaquant import tessera_campaign as campaign
from prismaquant.cost_currency import CostCurrencyError
from prismaquant.joint_aura import make_joint_aura_entry
from prismaquant.model_profiles import DefaultProfile
from prismaquant.tessera_full_domain_acquisition import (
    joint_acquisition_from_cost_data,
    load_joint_campaign_acquisition,
    project_joint_campaign_acquisition,
)
from prismaquant.tessera_legal_domain import live_pins, tessera_source_state
from test_campaign_acquisition_intake import bound_document, two_unit_payload
from test_campaign_acquisition_scheduler import domain, packet
from test_joint_full_domain_acquisition import FAMILY, FORMATS

DEFERRED = "TESSERA_BF16_K1"


@pytest.fixture(scope="module")
def full_document(two_unit_payload):
    payload = pickle.loads(pickle.dumps(two_unit_payload))
    shapes = {
        name: rows[FORMATS[0]]["joint_operator_identity"]["source_weight"]["shape"]
        for name, rows in payload["costs"].items()
    }
    actual = joint_acquisition_from_cost_data(payload, shapes, [FAMILY, DEFERRED], max_new_points=2)
    # This family is genuinely unmeasured. Its full legal domain remains in
    # the same original document while these row requests defer its work.
    deferred = joint_acquisition_from_cost_data(
        payload, shapes, [DEFERRED], max_new_points=2, boundary_policy="defer")
    reports = {(r["unit_name"], r["family"]): r for r in actual["reports"]}
    reports.update({(r["unit_name"], r["family"]): r for r in deferred["reports"]})
    return {
        "schema": "prismaquant.tessera_full_domain_campaign_acquisition.v1",
        "journal_bindings": {},
        "active_encoder_source_sha256": None,
        "domain_pins": live_pins().as_dict(),
        "producer_source_state": tessera_source_state(),
        "atomic_serving_group_expansion_required": True,
        "allocator_payload": False,
        "production_qualified": False,
        **actual,
        "reports": list(reports.values()),
        "total_requested_quality_measurements": sum(len(r["proposed_q256"]) for r in reports.values()),
    }


def test_two_actual_cohorts_keep_global_origin_and_exact_sources(
        two_unit_payload, full_document, tmp_path):
    binding, document = bound_document(tmp_path, full_document, two_unit_payload)
    request_bytes = (tmp_path / "request.json").read_bytes()
    cost_bytes = (tmp_path / "joint.pkl").read_bytes()
    complete = load_joint_campaign_acquisition(binding)
    assert load_joint_campaign_acquisition(binding, units=None) == complete
    names = sorted(complete["requests"])
    groups = campaign.resolve_anchor_groups(names, profile=DefaultProfile(), expert_members={})
    assert len(groups) == 2
    for members in groups.values():
        scoped = load_joint_campaign_acquisition(binding, units=members)
        assert scoped == project_joint_campaign_acquisition(complete, units=members)
        assert set(scoped["requests"]) == set(members)
        assert set(scoped["source_weights"]) == set(members)
        assert scoped["identity"] == complete["identity"]
        campaign._campaign_acquisition_scope(scoped, groups, selected=set(members))
        for name in members:
            assert scoped["requests"][name] == complete["requests"][name]
            assert scoped["requests"][name][DEFERRED] == []
            assert scoped["source_weights"][name] == two_unit_payload["costs"][name][FORMATS[0]][
                "joint_operator_identity"]["source_weight"]
    assert (tmp_path / "request.json").read_bytes() == request_bytes
    assert (tmp_path / "joint.pkl").read_bytes() == cost_bytes
    assert document["reports"] == full_document["reports"]


def test_projection_is_order_independent_and_does_not_mutate_intake(
        two_unit_payload, full_document, tmp_path):
    binding, _ = bound_document(tmp_path, full_document, two_unit_payload)
    complete = load_joint_campaign_acquisition(binding)
    before = pickle.dumps(complete)
    names = sorted(complete["requests"])
    projected = project_joint_campaign_acquisition(complete, units=list(reversed(names)))
    assert list(projected["requests"]) == names
    assert projected == project_joint_campaign_acquisition(complete, units=tuple(names))
    assert pickle.dumps(complete) == before
    projected["requests"][names[0]][FAMILY].append(-1)
    projected["source_weights"][names[0]]["shape"][0] = -1
    projected["identity"]["request_sha256"] = "0" * 64
    assert pickle.dumps(complete) == before


@pytest.mark.parametrize("units", [[], ["unknown.unit"], [""], [True], [1], "unit", b"unit", {}, None])
def test_pure_projection_rejects_invalid_explicit_selection(
        two_unit_payload, full_document, tmp_path, units):
    binding, _ = bound_document(tmp_path, full_document, two_unit_payload)
    complete = load_joint_campaign_acquisition(binding)
    with pytest.raises(ValueError, match="selected|scope"):
        project_joint_campaign_acquisition(complete, units=units)


def test_loader_rejects_duplicate_and_unknown_selected_names(
        two_unit_payload, full_document, tmp_path):
    binding, _ = bound_document(tmp_path, full_document, two_unit_payload)
    name = full_document["reports"][0]["unit_name"]
    for units in ([name, name], [name, "unknown.unit"], []):
        with pytest.raises(ValueError, match="selected|scope"):
            load_joint_campaign_acquisition(binding, units=units)


def test_all_deferred_selected_row_refuses_but_empty_member_is_retained(
        two_unit_payload, full_document, tmp_path):
    document = copy.deepcopy(full_document)
    names = sorted({r["unit_name"] for r in document["reports"]})
    for report in document["reports"]:
        if report["unit_name"] == names[1]:
            report["proposed_q256"] = []
            report["proposal_reasons"] = {}
    document["total_requested_quality_measurements"] = sum(len(r["proposed_q256"]) for r in document["reports"])
    binding, _ = bound_document(tmp_path, document, two_unit_payload)
    with pytest.raises(ValueError, match="no requested measurement work"):
        load_joint_campaign_acquisition(binding, units=[names[1]])
    combined = load_joint_campaign_acquisition(binding, units=names)
    assert combined["requests"][names[1]] == {FAMILY: [], DEFERRED: []}
    assert combined["source_weights"][names[1]] == two_unit_payload["costs"][names[1]][FORMATS[0]][
        "joint_operator_identity"]["source_weight"]
    assert combined["requests"][names[0]][FAMILY]


@pytest.mark.parametrize("change", ["source", "probe", "raw", "shape", "legal"])
def test_unselected_report_is_validated_before_projection(
        two_unit_payload, full_document, tmp_path, change):
    document = copy.deepcopy(full_document)
    names = sorted({r["unit_name"] for r in document["reports"]})
    report = next(r for r in document["reports"] if r["unit_name"] == names[1] and r["family"] == FAMILY)
    if change == "source":
        report["joint_source_weight"]["content_sha256"] = "f" * 64
    elif change == "probe":
        report["probe_identity_sha256"] = "f" * 64
    elif change == "raw":
        report["joint_measurement_records"][FORMATS[0]]["signed_components_per_probe"][0]["weight"] += 1
    elif change == "shape":
        report["shape"][0] += 1
    else:
        report["legal_q256"].pop()
        report["legal_rate_count"] -= 1
    binding, _ = bound_document(tmp_path, document, two_unit_payload)
    with pytest.raises(ValueError):
        load_joint_campaign_acquisition(binding, units=[names[0]])


@pytest.mark.parametrize("change", ["source", "probe", "raw", "scalar", "diagnostic"])
def test_unselected_cost_unit_cannot_escape_original_raw_validation(
        two_unit_payload, full_document, tmp_path, change):
    payload = pickle.loads(pickle.dumps(two_unit_payload))
    names = sorted(payload["costs"])
    row = payload["costs"][names[1]][FORMATS[0]]
    if change == "source":
        operator = copy.deepcopy(row["joint_operator_identity"])
        operator["source_weight"]["content_sha256"] = "f" * 64
        payload["costs"][names[1]][FORMATS[0]] = make_joint_aura_entry(
            operator_identity=operator, probe_identity=row["probe_identity"],
            signed_components=row["signed_components_per_probe"])
    elif change == "probe":
        row["probe_identity_sha256"] = "f" * 64
    elif change == "raw":
        del row["signed_components_per_probe"]
    elif change == "scalar":
        payload["costs"][names[1]][FORMATS[0]] = {"predicted_dloss": 0.0}
    else:
        payload["provenance"]["joint_eval"] = {"status": "diagnostic"}
    binding, _ = bound_document(tmp_path, full_document, payload)
    with pytest.raises((ValueError, CostCurrencyError)):
        load_joint_campaign_acquisition(binding, units=[names[0]])


def test_atomic_partial_selection_is_runtime_refusal_not_library_group_inference(domain):
    # Pure projection of the existing scheduler fixture deliberately does not
    # authenticate it or guess groups from names. Actual group ownership stays
    # with the scheduler, which refuses this incomplete attention cohort.
    complete = packet(domain)
    scoped = project_joint_campaign_acquisition(complete, units=[domain.names[0]])
    assert set(scoped["requests"]) == {domain.names[0]}
    with pytest.raises(ValueError, match="member|scope"):
        campaign._campaign_acquisition_scope(scoped, domain.groups, selected={domain.names[0]})
    valid = project_joint_campaign_acquisition(complete, units=domain.names[:3])
    _, members = campaign._campaign_acquisition_scope(valid, domain.groups, selected=set(domain.names[:3]))
    assert members == set(domain.names[:3])
    assert valid["requests"][domain.names[2]][FAMILY] == []
