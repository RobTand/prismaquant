"""The paired-family transfer evaluator must stay prospective and receipted."""
import hashlib
import json
from pathlib import Path

import pytest

from experiments.sparse_rate_adaptive import CURRENCY, CURVE_SCHEMA, MEASUREMENT_PLAN_SCHEMA
from experiments.sparse_rate_family_transfer import (
    FamilyTransferError, audit_predictions, freeze_protocol, seal_predictions,
)


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return str(path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plan(tmp_path, role, family, rates, *, terminal=False):
    activation = "fp8_per_token_dynamic" if family == "TESSERA_E4M3_K1" else (
        "bf16_unquantized" if family == "TESSERA_BF16_K1" else "e2m1_group16_ue4m3_static")
    value = {"schema": MEASUREMENT_PLAN_SCHEMA, "status": "frozen_before_measurement",
             "measurement_kind": "measured", "currency": CURRENCY, "curve_id": role,
             "qname": "model.layers.20.down_proj", "family": family,
             "activation_contract": activation, "legal_rates": rates, "audit_regions": {},
             "source_identity": {"model": "fixture", "weight_identity": {"x": 1}, "producer": "fixture"},
             "calibration_identity": {"capture": "fixture"},
             "recipe_identity": {"family": family, "terminal": terminal}}
    path = tmp_path / f"{role}.plan.json"
    _write(path, value)
    return path, value


def _curve(tmp_path, plan_path, plan, values):
    receipts = []
    for rate, value in zip(plan["legal_rates"], values):
        receipt = tmp_path / f"{plan['curve_id']}-{rate}.receipt.json"
        _write(receipt, {"curve_id": plan["curve_id"], "measurement_plan_sha256": _sha(plan_path),
                         "rate": rate, "value": value, "kind": "measured"})
        receipts.append({"rate": rate, "value": value, "kind": "measured", "path": str(receipt), "sha256": _sha(receipt)})
    curve = {key: plan[key] for key in ("currency", "curve_id", "qname", "family", "activation_contract",
             "source_identity", "calibration_identity", "recipe_identity", "legal_rates", "audit_regions")}
    curve.update(schema=CURVE_SCHEMA, measurement_kind="measured", rates=plan["legal_rates"],
                 values=values, receipts=receipts,
                 measurement_plan={"path": str(plan_path), "sha256": _sha(plan_path)})
    path = tmp_path / f"{plan['curve_id']}.curve.json"
    _write(path, curve)
    return path


@pytest.fixture
def fixture(tmp_path):
    specs = {
        "bf_source": ("TESSERA_BF16_K1", [832, 833, 834], False, [10.0, 8.0, 6.0]),
        "e4_left": ("TESSERA_E4M3_K1", [832], False, [14.0]),
        "e4_right": ("TESSERA_E4M3_K1", [834], False, [10.0]),
        "e4_interior": ("TESSERA_E4M3_K1", [833], False, [12.0]),
        "e2_left": ("TESSERA_E2M1_K2", [832], False, [20.0]),
        "e2_right": ("TESSERA_E2M1_K2", [895], False, [11.0]),
        "e2_interior": ("TESSERA_E2M1_K2", [833, 894], False, [19.9, 11.1]),
        "e2_terminal": ("TESSERA_E2M1_K2", [896], True, [99.0]),
    }
    plans, curves = {}, {}
    for role, (family, rates, terminal, values) in specs.items():
        plan_path, plan = _plan(tmp_path, role, family, rates, terminal=terminal)
        plans[role] = str(plan_path); curves[role] = str(_curve(tmp_path, plan_path, plan, values))
    protocol = tmp_path / "protocol.json"
    freeze_protocol(plans, protocol)
    return protocol, curves


def test_sealer_accepts_only_endpoint_inputs_and_uses_fixed_e4_forms(fixture, tmp_path):
    protocol, curves = fixture
    out = tmp_path / "e4.seal.json"
    with pytest.raises(FamilyTransferError, match="exactly its endpoint"):
        seal_predictions(protocol, "e4", {"bf_source": curves["bf_source"],
                                            "e4_left": curves["e4_left"],
                                            "e4_right": curves["e4_right"],
                                            "e4_interior": curves["e4_interior"]}, out)
    seal = seal_predictions(protocol, "e4", {key: curves[key] for key in ("bf_source", "e4_left", "e4_right")}, out)
    middle = seal["predictions"][1]
    assert middle == {"rate": 833, "affine_bf_from_paired_endpoints": 12.0,
                      "rate_linear_delta_from_bf": 12.0}
    assert seal["inputs"].keys() == {"bf_source", "e4_left", "e4_right"}
    audit = audit_predictions(protocol, out,
                              {key: curves[key] for key in ("bf_source", "e4_left", "e4_right")},
                              {"e4_interior": curves["e4_interior"]}, tmp_path / "e4.audit.json")
    assert set(audit["metrics"]) == {"affine_bf_from_paired_endpoints", "rate_linear_delta_from_bf"}


def test_audit_rechecks_input_bytes_and_never_mixes_the_e2_terminal(fixture, tmp_path):
    protocol, curves = fixture
    seal_path = tmp_path / "e2.seal.json"
    seal = seal_predictions(protocol, "e2", {key: curves[key] for key in ("e2_left", "e2_right")}, seal_path)
    assert [row["rate"] for row in seal["predictions"]] == [832, 833, 894, 895]
    audit = audit_predictions(protocol, seal_path,
                              {key: curves[key] for key in ("e2_left", "e2_right")},
                              {key: curves[key] for key in ("e2_interior", "e2_terminal")},
                              tmp_path / "e2.audit.json")
    assert audit["unseen_only"] is True
    assert audit["terminal"]["prediction"] is None
    with open(curves["e2_left"], "a") as handle:
        handle.write(" ")
    with pytest.raises(FamilyTransferError, match="sealed source curve bytes changed"):
        audit_predictions(protocol, seal_path,
                          {key: curves[key] for key in ("e2_left", "e2_right")},
                          {key: curves[key] for key in ("e2_interior", "e2_terminal")},
                          tmp_path / "tampered.audit.json")


def test_resealed_prediction_roster_tamper_and_wrong_preselection_are_refused(fixture, tmp_path):
    protocol, curves = fixture
    seal_path = tmp_path / "e2.seal.json"
    seal_predictions(protocol, "e2", {key: curves[key] for key in ("e2_left", "e2_right")}, seal_path)
    seal = json.loads(seal_path.read_text())
    seal["predictions"].pop()
    seal.pop("seal_sha256")
    seal["seal_sha256"] = hashlib.sha256(json.dumps(
        seal, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    _write(seal_path, seal)
    with pytest.raises(FamilyTransferError, match="prediction roster"):
        audit_predictions(protocol, seal_path,
                          {key: curves[key] for key in ("e2_left", "e2_right")},
                          {key: curves[key] for key in ("e2_interior", "e2_terminal")},
                          tmp_path / "bad-roster.audit.json")

    root = Path(curves["bf_source"]).parent
    plans = {role: str(root / f"{role}.plan.json") for role in (
        "bf_source", "e4_left", "e4_right", "e4_interior", "e2_left", "e2_right", "e2_interior", "e2_terminal")}
    selection = {"schema": "prismaquant.family_transfer_preselection.v1",
                 "status": "frozen_before_new_dense_measurements", "research_only": True,
                 "qname": "wrong.linear", "measurement_plans": {
                     role: {"path": path, "sha256": _sha(path)} for role, path in plans.items()}}
    selection_path = root / "bad-preselection.json"; _write(selection_path, selection)
    with pytest.raises(FamilyTransferError, match="preselection"):
        freeze_protocol(plans, root / "bad-protocol.json", preselection_path=selection_path)
