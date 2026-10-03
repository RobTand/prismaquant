"""Actual CPU streamed joint rows drive research requests, never qualification."""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
import subprocess
import sys

import pytest
import torch

from prismaquant.cost_currency import CostCurrencyError
from prismaquant.joint_aura import identity_sha256, make_joint_aura_entry
from prismaquant.tessera_full_domain_acquisition import joint_acquisition_from_cost_data
from test_joint_aura_streamed import _fixture
from test_streamed_cost_checkpoints import _model_identity

FAMILY = "TESSERA_E4M3_K1"
FORMATS = (FAMILY + "_R256", FAMILY + "_R2048")


@pytest.fixture(scope="module")
def joint_payload():
    import prismaquant.aura_cost as aura
    from prismaquant.production_weight_cache import ProductionWeightCache

    model, _, runner, _ = _fixture(layers=1)
    names = [name for name, _ in model.named_modules() if name.endswith(".proj")]
    weights = {(name, fmt): model.get_submodule(name).weight.detach().clone() + delta
               for name in names for fmt, delta in zip(FORMATS, (0.25, 0.03125))}
    cache = ProductionWeightCache(weights=weights, levers={},
                                  activation_max_abs={name: 1.0 for name in names})
    return aura.compute_aura_cost_streamed(
        runner, torch.tensor([[1, 2, 3, 4]]), [*FORMATS, "BF16"],
        n_probes=3, min_free_gib=0, production_cache=cache,
        joint_activation=True, model_identity=_model_identity("joint-acquisition-source"))


def selection(payload):
    name = next(iter(payload["costs"]))
    shape = payload["costs"][name][FORMATS[0]]["joint_operator_identity"]["source_weight"]["shape"]
    return name, shape


def acquire(payload, **kwargs):
    name, shape = selection(payload)
    return joint_acquisition_from_cost_data(payload, {name: shape}, [FAMILY],
                                            max_new_points=2, **kwargs)


def test_actual_streamed_joint_rows_retain_raw_cross_terms_and_run_identity(joint_payload):
    before = pickle.dumps(joint_payload)
    result = acquire(joint_payload)
    report = result["reports"][0]
    name, shape = selection(joint_payload)
    assert report["currency"] == "joint_aura_predicted_dloss"
    assert report["shape"] == shape
    assert report["producer_refused_q256"]  # Tiny columns have real quota holes.
    for fmt in FORMATS:
        assert report["joint_measurement_records"][fmt] == joint_payload["costs"][name][fmt]
        assert report["joint_measurement_records"][fmt]["signed_components_per_probe"]
    assert pickle.dumps(joint_payload) == before
    assert report["prices"] is None
    assert report["allocator_payload"] is False
    assert report["production_qualified"] is False
    assert report["interpolation_qualified"] is False
    assert report["adaptive_converged"] is None
    assert all(q in report["legal_q256"] for q in report["proposed_q256"])
    assert report["joint_aura_identity_sha256"] == joint_payload["provenance"]["joint_aura_identity_sha256"]


def test_wholly_unmeasured_family_is_a_request_not_a_dropped_domain(joint_payload):
    name, shape = selection(joint_payload)
    report = joint_acquisition_from_cost_data(joint_payload, {name: shape},
        ["TESSERA_BF16_K1"], max_new_points=2)["reports"][0]
    assert report["measured_q256"] == []
    assert report["joint_measurement_records"] == {}
    assert report["proposed_q256"] == [256, 4096]
    assert report["prices"] is None


@pytest.mark.parametrize("field", ["joint_aura_identity", "joint_aura_identity_sha256", "probe_identity"])
def test_joint_run_provenance_cannot_be_removed(joint_payload, field):
    payload = copy.deepcopy(joint_payload)
    del payload["provenance"][field]
    with pytest.raises(ValueError, match="run|probe"):
        acquire(payload)


@pytest.mark.parametrize("field", ["signed_components_per_probe", "x2_per_probe", "joint_operator_identity_sha256"])
def test_scalar_summary_cannot_replace_required_raw_joint_evidence(joint_payload, field):
    payload = copy.deepcopy(joint_payload)
    name, _ = selection(payload)
    del payload["costs"][name][FORMATS[0]][field]
    with pytest.raises((CostCurrencyError, ValueError)):
        acquire(payload)


def test_mixed_scalar_and_joint_rows_refuse(joint_payload):
    payload = copy.deepcopy(joint_payload)
    name, _ = selection(payload)
    payload["costs"][name]["BF16"] = {"predicted_dloss": 0.0}
    with pytest.raises(CostCurrencyError, match="mixes joint"):
        acquire(payload)


def test_scalar_only_table_refuses_joint_path(joint_payload):
    name, _ = selection(joint_payload)
    payload = {"costs": {name: {FORMATS[0]: {"output_mse": 1.0,
        "output_mse_measured": True, "cost_source": "tessera_campaign_measured",
        "currency": "output_mse_under_route_activation_contract"}}},
        "provenance": {"cost_mode": "production-render-score"}}
    with pytest.raises(CostCurrencyError, match="joint acquisition"):
        joint_acquisition_from_cost_data(payload, {name: (4, 4)}, [FAMILY], max_new_points=1)


@pytest.mark.parametrize("part", ["cached_rendered_weights", "activation_contracts"])
def test_valid_row_does_not_rebind_to_a_different_run(joint_payload, part):
    payload = copy.deepcopy(joint_payload)
    name, _ = selection(payload)
    run = payload["provenance"]["joint_aura_identity"]
    run[part][name][FORMATS[0]] = {}
    payload["provenance"]["joint_aura_identity_sha256"] = identity_sha256(run)
    with pytest.raises(ValueError, match="bound run"):
        acquire(payload)


def test_valid_joint_row_with_changed_source_refuses_across_candidates(joint_payload):
    payload = copy.deepcopy(joint_payload)
    name, _ = selection(payload)
    old = payload["costs"][name][FORMATS[0]]
    operator = copy.deepcopy(old["joint_operator_identity"])
    operator["source_weight"]["content_sha256"] = "f" * 64
    payload["costs"][name][FORMATS[0]] = make_joint_aura_entry(
        operator_identity=operator, probe_identity=old["probe_identity"],
        signed_components=old["signed_components_per_probe"])
    with pytest.raises(ValueError, match="mixes source"):
        acquire(payload)


def test_shape_inventory_cannot_rebind_operator(joint_payload):
    name, _ = selection(joint_payload)
    with pytest.raises(ValueError, match="source shape"):
        joint_acquisition_from_cost_data(joint_payload, {name: (256, 256)},
                                        [FAMILY], max_new_points=1)


def test_family_rate_metadata_cannot_rebind_raw_row(joint_payload):
    payload = copy.deepcopy(joint_payload)
    name, _ = selection(payload)
    payload["costs"][name][FORMATS[0]]["tessera_body_rate_q256"] = 257
    with pytest.raises(ValueError, match="format/family/rate"):
        acquire(payload)


def cli(tmp_path, payload, *extra):
    name, shape = selection(payload)
    cost = tmp_path / "joint.pkl"
    cost.write_bytes(pickle.dumps(payload))
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{"name": name + ".weight", "shape": shape}]))
    out = tmp_path / "requests.json"
    args = [sys.executable, "-m", "experiments.tessera_full_domain_acquisition",
            "--costs", str(cost), "--source-tensors", str(source),
            "--unit", name, "--family", FAMILY, "--max-new-points", "2",
            "--out", str(out), *extra]
    return subprocess.run(args, capture_output=True, text=True), out, cost


def test_actual_joint_cli_smoke_preserves_evidence_and_request_only(joint_payload, tmp_path):
    done, out, cost = cli(tmp_path, joint_payload, "--cost-currency", "joint-aura")
    assert done.returncode == 0, done.stderr
    result = json.loads(out.read_text())
    assert result["cost_sha256"] == hashlib.sha256(cost.read_bytes()).hexdigest()
    assert result["allocator_payload"] is False
    assert result["production_qualified"] is False
    assert result["reports"][0]["prices"] is None
    assert result["reports"][0]["joint_measurement_records"]
    assert result["active_encoder_source_sha256"] is None


def test_cli_scalar_journal_does_not_authorize_joint_mode(joint_payload, tmp_path):
    done, out, _ = cli(tmp_path, joint_payload, "--cost-currency", "joint-aura",
                       "--anchor-parts", str(tmp_path))
    assert done.returncode != 0
    assert "not a joint run identity" in done.stderr
    assert not out.exists()


def test_cli_default_does_not_silently_select_joint_currency(joint_payload, tmp_path):
    done, out, _ = cli(tmp_path, joint_payload, "--anchor-parts", str(tmp_path))
    assert done.returncode != 0
    assert "requires measured scalar render-score" in done.stderr
    assert not out.exists()
