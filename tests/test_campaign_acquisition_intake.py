"""Authenticated files from real streamed joint rows, without mock currency."""
from __future__ import annotations

import copy
import hashlib
import json
import pickle

import pytest

from prismaquant.cost_currency import CostCurrencyError
from prismaquant.joint_aura import identity_sha256, make_joint_aura_entry
from prismaquant.tessera_full_domain_acquisition import (
    joint_acquisition_from_cost_data,
    load_joint_campaign_acquisition,
)
from prismaquant.tessera_legal_domain import live_pins, tessera_source_state
from test_joint_full_domain_acquisition import FAMILY, FORMATS, joint_payload, selection


@pytest.fixture(scope="module")
def request_document(joint_payload):
    # Persisted rows restore ordinary JSON probe identities, just as the CLI's
    # bound pickle does. The underlying measurements are the streamed fixture.
    payload = pickle.loads(pickle.dumps(joint_payload))
    name, shape = selection(payload)
    actual = joint_acquisition_from_cost_data(payload, {name: shape}, [FAMILY], max_new_points=2)
    document = {
        "schema": "prismaquant.tessera_full_domain_campaign_acquisition.v1",
        "source_tensor_inventory_sha256": hashlib.sha256(json.dumps(
            [{"name": name + ".weight", "shape": shape}]).encode()).hexdigest(),
        "journal_bindings": {},
        "active_encoder_source_sha256": None,
        "domain_pins": live_pins().as_dict(),
        "producer_source_state": tessera_source_state(),
        "atomic_serving_group_expansion_required": True,
        "allocator_payload": False,
        "production_qualified": False,
        **actual,
    }
    document["total_requested_quality_measurements"] = sum(
        len(report["proposed_q256"]) for report in document["reports"])
    return document


def bound_bytes(path, raw):
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def bound_document(tmp_path, document, payload):
    document = copy.deepcopy(document)
    cost = bound_bytes(tmp_path / "joint.pkl", pickle.dumps(payload))
    document.update(cost_path=cost["path"], cost_sha256=cost["sha256"])
    binding = bound_bytes(tmp_path / "request.json", json.dumps(document, allow_nan=False).encode())
    return binding, document


def changed_document(request_document):
    return copy.deepcopy(request_document)


def test_authenticated_actual_rows_return_only_requested_units_and_bound_identities(
        joint_payload, request_document, tmp_path):
    binding, document = bound_document(tmp_path, request_document, joint_payload)
    before_request = (tmp_path / "request.json").read_bytes()
    before_cost = (tmp_path / "joint.pkl").read_bytes()
    result = load_joint_campaign_acquisition(binding)
    name, _ = selection(joint_payload)
    report = document["reports"][0]
    assert result["requests"] == {name: {FAMILY: report["proposed_q256"]}}
    assert result["source_weights"] == {name: report["joint_source_weight"]}
    assert set(result["source_weights"]) == {report["unit_name"]}
    assert result["identity"] == {
        "request_sha256": binding["sha256"],
        "cost_sha256": document["cost_sha256"],
        "joint_aura_identity_sha256": report["joint_aura_identity_sha256"],
        "probe_identity_sha256": report["probe_identity_sha256"],
    }
    assert (tmp_path / "request.json").read_bytes() == before_request
    assert (tmp_path / "joint.pkl").read_bytes() == before_cost
    assert report["missing_legal_rate_count"] > len(report["proposed_q256"])
    assert report["all_legal_rates_retained"] is True


def test_existing_cli_artifact_is_consumable(joint_payload, tmp_path):
    from experiments.tessera_full_domain_acquisition import main

    name, shape = selection(joint_payload)
    cost = tmp_path / "producer-cost.pkl"
    cost.write_bytes(pickle.dumps(joint_payload))
    inventory = tmp_path / "source.json"
    inventory.write_text(json.dumps([{"name": name + ".weight", "shape": shape}]))
    out = tmp_path / "producer-request.json"
    assert main(["--costs", str(cost), "--source-tensors", str(inventory),
                 "--unit", name, "--family", FAMILY, "--cost-currency", "joint-aura",
                 "--max-new-points", "2", "--out", str(out)]) == 0
    raw = out.read_bytes()
    result = load_joint_campaign_acquisition(
        {"path": str(out), "sha256": hashlib.sha256(raw).hexdigest()})
    assert result["requests"][name][FAMILY] == json.loads(raw)["reports"][0]["proposed_q256"]


def test_empty_deferred_family_is_retained_with_positive_work_elsewhere(
        joint_payload, request_document, tmp_path):
    document = changed_document(request_document)
    name, shape = selection(joint_payload)
    deferred = joint_acquisition_from_cost_data(
        joint_payload, {name: shape}, ["TESSERA_BF16_K1"], max_new_points=2,
        boundary_policy="defer")["reports"][0]
    assert deferred["proposed_q256"] == []
    document["reports"].append(deferred)
    binding, _ = bound_document(tmp_path, document, joint_payload)
    result = load_joint_campaign_acquisition(binding)
    assert result["requests"][name]["TESSERA_BF16_K1"] == []
    assert result["requests"][name][FAMILY]


@pytest.mark.parametrize("target", ["request", "cost"])
def test_actual_file_digests_are_checked(joint_payload, request_document, tmp_path, target):
    binding, _ = bound_document(tmp_path, request_document, joint_payload)
    if target == "request":
        binding["sha256"] = "0" * 64
    else:
        # Authentication remains tied to original bytes, not the filename.
        (tmp_path / "joint.pkl").write_bytes(pickle.dumps({"costs": {}}))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_joint_campaign_acquisition(binding)


@pytest.mark.parametrize("raw", [
    b'{"schema":"first","schema":"second"}',
    b'{"reports":[],"reports":[]}',
    b'{"value":NaN}',
    b'{"value":Infinity}',
    b'[]',
    b'{',
])
def test_authenticated_json_is_strict(tmp_path, raw):
    binding = bound_bytes(tmp_path / "request.json", raw)
    with pytest.raises(ValueError):
        load_joint_campaign_acquisition(binding)


@pytest.mark.parametrize("change", ["boolean", "duplicate", "offgrid", "refused_hole", "measured", "cap"])
def test_illegal_proposals_fail_closed(joint_payload, request_document, tmp_path, change):
    document = changed_document(request_document)
    report = document["reports"][0]
    q = report["proposed_q256"][0]
    if change == "boolean":
        proposed = [True]
    elif change == "duplicate":
        proposed = [q, q]
    elif change == "offgrid":
        proposed = [255]
    elif change == "refused_hole":
        proposed = [int(next(iter(report["producer_refused_q256"])))]
    elif change == "measured":
        proposed = [report["measured_q256"][0]]
    else:
        proposed = report["proposed_q256"]
        report["max_new_points"] = 0
    report["proposed_q256"] = proposed
    report["proposal_reasons"] = {str(rate): "decision_focused_interior" for rate in proposed}
    document["total_requested_quality_measurements"] = len(proposed)
    binding, _ = bound_document(tmp_path, document, joint_payload)
    with pytest.raises(ValueError):
        load_joint_campaign_acquisition(binding)


@pytest.mark.parametrize("change", [
    "duplicate_report", "shape", "boolean_shape", "source", "rawrecords", "probe_digest", "run_digest",
    "legal_roster", "measured_roster", "domain_pins", "producer_source", "cost_currency",
    "total", "boolean_total", "empty", "scalar_bindings", "cap_boolean", "cap_negative",
])
def test_authenticated_report_cannot_replace_actual_evidence(
        joint_payload, request_document, tmp_path, change):
    document = changed_document(request_document)
    report = document["reports"][0]
    if change == "duplicate_report":
        document["reports"].append(copy.deepcopy(report))
        document["total_requested_quality_measurements"] *= 2
    elif change == "shape":
        report["shape"][0] += 1
    elif change == "boolean_shape":
        report["shape"][0] = True
    elif change == "source":
        report["joint_source_weight"]["content_sha256"] = "f" * 64
    elif change == "rawrecords":
        report["joint_measurement_records"][FORMATS[0]]["signed_components_per_probe"][0]["weight"] += 1
    elif change == "probe_digest":
        report["probe_identity_sha256"] = "f" * 64
    elif change == "run_digest":
        report["joint_aura_identity_sha256"] = "f" * 64
    elif change == "legal_roster":
        report["legal_q256"].pop()
        report["legal_rate_count"] -= 1
        report["missing_legal_rate_count"] -= 1
    elif change == "measured_roster":
        report["measured_q256"] = []
    elif change == "domain_pins":
        document["domain_pins"] = {}
    elif change == "producer_source":
        document["producer_source_state"]["export_sha256"] = "f" * 64
    elif change == "cost_currency":
        document["cost_currency"] = {"cost_currency": "joint_aura_predicted_dloss"}
    elif change == "total":
        document["total_requested_quality_measurements"] += 1
    elif change == "boolean_total":
        document["total_requested_quality_measurements"] = True
    elif change == "empty":
        report["proposed_q256"] = []
        report["proposal_reasons"] = {}
        document["total_requested_quality_measurements"] = 0
    elif change == "scalar_bindings":
        document["journal_bindings"] = {report["unit_name"]: {"path": "scalar.pkl"}}
    elif change == "cap_boolean":
        report["max_new_points"] = True
    else:
        report["max_new_points"] = -1
    binding, _ = bound_document(tmp_path, document, joint_payload)
    with pytest.raises(ValueError):
        load_joint_campaign_acquisition(binding)


@pytest.mark.parametrize("scope", ["document", "report"])
@pytest.mark.parametrize("field,value", [
    ("prices", {"256": 0.0}),
    ("interpolation_error_bound", 0.0),
    ("interpolation_qualified", True),
    ("selected_assignment_confirmed", True),
    ("allocator_payload", True),
    ("production_qualified", True),
])
def test_request_does_not_authorize_prices_or_production(
        joint_payload, request_document, tmp_path, scope, field, value):
    document = changed_document(request_document)
    target = document if scope == "document" else document["reports"][0]
    target[field] = value
    binding, _ = bound_document(tmp_path, document, joint_payload)
    with pytest.raises(ValueError, match="claimed"):
        load_joint_campaign_acquisition(binding)


@pytest.mark.parametrize("change", [
    "mixed", "scalar", "diagnostic", "single_probe", "missing_raw", "source", "run", "run_probe",
])
def test_bound_pickle_is_revalidated_not_trusted_from_request(
        joint_payload, request_document, tmp_path, change):
    payload = pickle.loads(pickle.dumps(joint_payload))
    name, _ = selection(payload)
    row = payload["costs"][name][FORMATS[0]]
    if change == "mixed":
        payload["costs"][name]["BF16"] = {"predicted_dloss": 0.0}
    elif change == "scalar":
        payload = {"costs": {name: {FORMATS[0]: {
            "output_mse": 1.0, "output_mse_measured": True,
            "cost_source": "tessera_campaign_measured",
            "currency": "output_mse_under_route_activation_contract"}}},
            "provenance": {"cost_mode": "production-render-score"}}
    elif change == "diagnostic":
        payload["provenance"]["joint_eval"] = {"status": "diagnostic"}
    elif change == "single_probe":
        row["probe_identity"]["n_probes"] = 1
    elif change == "missing_raw":
        del row["signed_components_per_probe"]
    elif change == "source":
        operator = copy.deepcopy(row["joint_operator_identity"])
        operator["source_weight"]["content_sha256"] = "f" * 64
        payload["costs"][name][FORMATS[0]] = make_joint_aura_entry(
            operator_identity=operator, probe_identity=row["probe_identity"],
            signed_components=row["signed_components_per_probe"])
    elif change == "run":
        run = payload["provenance"]["joint_aura_identity"]
        run["cached_rendered_weights"][name][FORMATS[0]] = {}
        payload["provenance"]["joint_aura_identity_sha256"] = identity_sha256(run)
    else:
        run = payload["provenance"]["joint_aura_identity"]
        run["probe_identity"] = copy.deepcopy(run["probe_identity"])
        run["probe_identity"]["calibration_sha256"] = "e" * 64
        payload["provenance"]["joint_aura_identity_sha256"] = identity_sha256(run)
    binding, _ = bound_document(tmp_path, request_document, payload)
    with pytest.raises((ValueError, CostCurrencyError)):
        load_joint_campaign_acquisition(binding)


@pytest.fixture(scope="module")
def two_unit_payload():
    import torch
    import prismaquant.aura_cost as aura
    from prismaquant.production_weight_cache import ProductionWeightCache
    from test_joint_aura_streamed import _fixture
    from test_streamed_cost_checkpoints import _model_identity

    model, _, runner, _ = _fixture(layers=2)
    names = [name for name, _ in model.named_modules() if name.endswith(".proj")]
    cache = ProductionWeightCache(
        weights={(name, fmt): model.get_submodule(name).weight.detach().clone() + delta
                 for name in names for fmt, delta in zip(FORMATS, (0.25, 0.03125))},
        levers={}, activation_max_abs={name: 1.0 for name in names})
    return aura.compute_aura_cost_streamed(
        runner, torch.tensor([[1, 2, 3, 4]]), [*FORMATS, "BF16"], n_probes=3,
        min_free_gib=0, production_cache=cache, joint_activation=True,
        model_identity=_model_identity("joint-acquisition-two-unit-source"))


def test_source_projection_includes_empty_report_units_but_not_unreported_cost_units(
        two_unit_payload, request_document, tmp_path):
    payload = pickle.loads(pickle.dumps(two_unit_payload))
    names = sorted(payload["costs"])
    assert len(names) == 2
    shapes = {name: payload["costs"][name][FORMATS[0]]["joint_operator_identity"]["source_weight"]["shape"]
              for name in names}
    actual = joint_acquisition_from_cost_data(payload, shapes, [FAMILY], max_new_points=2)
    document = changed_document(request_document)
    document.update(actual)
    deferred = joint_acquisition_from_cost_data(
        payload, {names[1]: shapes[names[1]]}, [FAMILY], max_new_points=0)["reports"][0]
    document["reports"] = [actual["reports"][0], deferred]
    document["total_requested_quality_measurements"] = sum(
        len(report["proposed_q256"]) for report in document["reports"])
    binding, _ = bound_document(tmp_path, document, payload)
    result = load_joint_campaign_acquisition(binding)
    assert result["requests"][names[1]][FAMILY] == []
    assert set(result["source_weights"]) == set(names)
    document["reports"] = [actual["reports"][0]]
    binding, _ = bound_document(tmp_path, document, payload)
    result = load_joint_campaign_acquisition(binding)
    assert set(result["source_weights"]) == {names[0]}
    assert set(result["requests"]) == {names[0]}
