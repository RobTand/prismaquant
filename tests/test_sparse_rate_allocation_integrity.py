"""Regression setup for allocation-audit evidence binding."""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pytest

from experiments import sparse_rate_allocation_audit as audit
from experiments.sparse_rate_study_seal import StudySealError, create_seal


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_unsealed_inputs(tmp_path, *, currency="wrong_currency", altered_anchor=True):
    dataset, study = tmp_path / "dataset", tmp_path / "study"
    dataset.mkdir(); study.mkdir()
    qnames = np.asarray([f"model.layers.1.mlp.experts.0.{p}"
                         for p in ("gate_proj", "up_proj", "down_proj")])
    values = np.asarray([[4., 2., 1.], [3., 1.5, .75], [8., 4., 2.]])
    np.savez(dataset / "sparse_rate_dataset.npz", qnames=qnames,
             families=np.asarray([audit.FAMILY] * 3), activation_contracts=np.asarray(["fp8"] * 3),
             roles=np.asarray(["expert"] * 3),
             structures=np.asarray(["gate_proj", "up_proj", "down_proj"]),
             layers=np.asarray([1, 1, 1]), rows=np.asarray([32] * 3), cols=np.asarray([64] * 3),
             counts=np.asarray([7, 7, 11]), rates=np.asarray([832, 960, 1088]),
             values=values, wire_bytes=np.asarray([[832, 960, 1088]] * 3))
    npz = dataset / "sparse_rate_dataset.npz"
    manifest = {"schema": audit.DATASET_SCHEMA, "currency": currency, "npz_sha256": _sha(npz)}
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    plan = {"schema": audit.STUDY_SCHEMA, "stage": "development", "currency": currency,
            "identity": {"dataset_npz_sha256": _sha(npz),
                         "dataset_manifest_sha256": _sha(dataset / "manifest.json")}}
    (study / "plan.json").write_text(json.dumps(plan))
    predictions = values.copy()
    if altered_anchor:
        predictions[:, 0] *= 100
    truth = np.full_like(values, np.nan); truth[:, 1] = values[:, 1]
    np.savez(study / "model.npz", predictions=predictions, truth=truth)
    (study / "report.json").write_text(json.dumps({**plan, "models": [
        {"model": "model", "anchors": [832, 1088]}]}))
    return dataset, study


def test_good_sealed_artifact_is_audited_with_its_measured_currency(tmp_path, monkeypatch):
    dataset, study = _write_unsealed_inputs(tmp_path, currency=audit.CURRENCY, altered_anchor=False)
    seal = tmp_path / "seal.json"
    create_seal(dataset, study, seal)
    out = tmp_path / "audit"
    monkeypatch.setattr(sys, "argv", ["audit", "--dataset", str(dataset), "--study", str(study),
                                        "--seal", str(seal), "--out", str(out)])
    audit.main()
    payload = json.loads((out / "allocation_audit.json").read_text())
    assert payload["currency"] == audit.CURRENCY
    assert payload["objective_label"] == "scalar_output_mse_allocation_regret_proxy"


def test_study_seal_refuses_changed_endpoint_before_sealing(tmp_path):
    dataset, study = _write_unsealed_inputs(tmp_path, currency=audit.CURRENCY)
    with pytest.raises(StudySealError, match="anchor prediction"):
        create_seal(dataset, study, tmp_path / "seal.json")


def test_study_seal_refuses_wrong_currency(tmp_path):
    dataset, study = _write_unsealed_inputs(tmp_path, altered_anchor=False)
    with pytest.raises(StudySealError, match="currency"):
        create_seal(dataset, study, tmp_path / "seal.json")


def test_audit_refuses_prediction_change_after_sealing(tmp_path, monkeypatch):
    dataset, study = _write_unsealed_inputs(tmp_path, currency=audit.CURRENCY, altered_anchor=False)
    seal = tmp_path / "seal.json"
    create_seal(dataset, study, seal)
    with np.load(study / "model.npz", allow_pickle=False) as archive:
        predictions, truth = archive["predictions"], archive["truth"]
    predictions[0, 0] *= 2
    np.savez(study / "model.npz", predictions=predictions, truth=truth)
    monkeypatch.setattr(sys, "argv", ["audit", "--dataset", str(dataset), "--study", str(study),
                                        "--seal", str(seal), "--out", str(tmp_path / "audit")])
    with pytest.raises(audit.AllocationAuditError, match="seal study binding"):
        audit.main()


def test_audit_refuses_stale_model_roster_after_sealing(tmp_path, monkeypatch):
    dataset, study = _write_unsealed_inputs(tmp_path, currency=audit.CURRENCY, altered_anchor=False)
    seal = tmp_path / "seal.json"
    create_seal(dataset, study, seal)
    report = json.loads((study / "report.json").read_text())
    report["models"].append({"model": "other", "anchors": [832, 1088]})
    (study / "report.json").write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["audit", "--dataset", str(dataset), "--study", str(study),
                                        "--seal", str(seal), "--out", str(tmp_path / "audit")])
    with pytest.raises(audit.AllocationAuditError, match="study model artifact missing"):
        audit.main()


def test_final_seal_refuses_choice_identity_mismatch(tmp_path):
    dataset, development = _write_unsealed_inputs(
        tmp_path, currency=audit.CURRENCY, altered_anchor=False)
    (development / "log_chord.npz").write_bytes((development / "model.npz").read_bytes())
    development_plan = json.loads((development / "plan.json").read_text())
    development_report = {**development_plan, "models": [
        {"model": "log_chord", "anchors": [832, 1088]}, {"model": "model", "anchors": [832, 1088]}]}
    (development / "report.json").write_text(json.dumps(development_report))
    choice = {"schema": audit.STUDY_SCHEMA, "identity": {"tampered": True},
              "two_anchor": "model", "one_anchor": "model",
              "development_report_sha256": _sha(development / "report.json")}
    choice_path = tmp_path / "choice.json"; choice_path.write_text(json.dumps(choice))
    final = tmp_path / "final"; final.mkdir()
    final_plan = {**development_plan, "stage": "final", "choice_sha256": _sha(choice_path)}
    (final / "plan.json").write_text(json.dumps(final_plan))
    (final / "report.json").write_text(json.dumps({**final_plan, "models": development_report["models"]}))
    for name in ("model", "log_chord"):
        (final / f"{name}.npz").write_bytes((development / f"{name}.npz").read_bytes())
    with pytest.raises(StudySealError, match="final choice"):
        create_seal(dataset, final, tmp_path / "seal.json", development_study=development, choice=choice_path)
