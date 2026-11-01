"""Fixture contracts for the measured sparse-rate research snapshot."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.sparse_rate_dataset import (
    CURRENCY, DATASET_NAME, MANIFEST_NAME, SparseRateDatasetError, build_snapshot,
)


def _cell(family, rate, value, wire, seconds, *, measured=True):
    return {
        "output_mse": value,
        "output_mse_measured": measured,
        "cost_source": "tessera_campaign_measured" if measured else "tessera_campaign_interpolated",
        "currency": CURRENCY,
        "tessera_provenance": "measured" if measured else "interpolated",
        "tessera_family": family,
        "tessera_body_rate_q256": rate,
        "activation_contract": "w8a8-dynamic-e4m3-channel",
        **({"wire_bytes": wire, "encode_seconds": seconds} if measured else {}),
    }


def _payload(qname, cells, *, model="fixture-model", commit="fixture-commit"):
    return {
        "schema": "prismaquant.tessera_campaign_cost.v1", "currency": CURRENCY,
        "costs": {qname: cells},
        "provenance": {
            "model": model, "tessera_commit": commit,
            "identity_migration": [{"new_pins": {"prismaquant_source_sha256": "a" * 64,
                                                     "encoder_source_sha256": "b" * 64}}],
            "calibration_cache": {"sha256": "c" * 64, "path": "/fixture/capture.json"},
            "hessian": {"reference_binding": {"capture_sha256": "d" * 64}},
            "cost_mode": "production-render-score", "nsamples": 2, "seqlen": 4,
            "max_act_rows": 4, "layer_stride": 1, "tp_degree": 1,
            "family_restriction": {"policy": "fixture"}, "rate_band": [1, 9],
            "menu_mode": "fixture",
        },
    }


def _write_fixture(tmp_path, rows, *, model="fixture-model"):
    workspace = tmp_path / "workspace"
    plan_rows, counts, shapes = [], {}, {}
    for index, (qname, cells) in enumerate(rows):
        row_dir = workspace / "rows" / f"row-{index:04d}"
        row_dir.mkdir(parents=True)
        (row_dir / "cost.pkl").write_bytes(pickle.dumps(_payload(qname, cells, model=model)))
        plan_rows.append({"row_id": f"row-{index:04d}", "dir": str(row_dir), "members": [qname]})
        counts[qname] = 11 + index
        shapes[qname] = [16, 8]
    census = {
        "counts": counts, "unit_shapes": shapes,
        "dense_targets": [qname for qname in counts if ".experts." not in qname],
        "expert_targets": [qname for qname in counts if ".experts." in qname],
    }
    plan_path, census_path = workspace / "plan.json", workspace / "census.json"
    plan = {"schema": "prismaquant.tessera_campaign_plan.v1", "census": str(census_path), "rows": plan_rows}
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    census_path.write_text(json.dumps(census, sort_keys=True))
    return plan_path, census_path


def test_snapshot_keeps_only_measured_observations_and_all_measured_rates(tmp_path):
    qname = "model.layers.3.mlp.experts.7.gate_proj"
    plan, census = _write_fixture(tmp_path, [(qname, {
        "F_R832": _cell("F", 832, 0.4, 100, 1.5),
        "F_R833": _cell("F", 833, 0.3, 101, 0.0),
        "F_R834": _cell("F", 834, 0.2, 102, 2.5, measured=False),
        "G_R832": _cell("G", 832, 0.1, 111, 3.5),
    })])

    result = build_snapshot(plan, census, tmp_path / "out")

    assert result["status"] == "written"
    with np.load(tmp_path / "out" / DATASET_NAME, allow_pickle=False) as snapshot:
        assert snapshot["rates"].tolist() == [832, 833]
        assert snapshot["qnames"].tolist() == [qname, qname]
        assert snapshot["families"].tolist() == ["F", "G"]
        assert snapshot["roles"].tolist() == ["expert", "expert"]
        assert snapshot["structures"].tolist() == ["gate_proj", "gate_proj"]
        assert snapshot["values"][0].tolist() == [0.4, 0.3]
        assert np.isnan(snapshot["values"][1, 1])
        assert snapshot["wire_bytes"].tolist() == [[100, 101], [111, 0]]
        assert snapshot["encode_seconds"][0].tolist() == [1.5, 0.0]
        assert np.isnan(snapshot["encode_seconds"][1, 1])
    manifest = json.loads((tmp_path / "out" / MANIFEST_NAME).read_text())
    assert manifest["research_only"] is True
    assert manifest["not_joint_aura"] is True
    assert manifest["not_serving_admission"] is True
    assert manifest["coverage"]["measured_observations"] == 3
    assert manifest["coverage"]["ignored_unmeasured_or_interpolated"] == 1


def test_snapshot_refuses_duplicate_observations_and_source_discrepancies(tmp_path):
    qname = "model.layers.3.mlp.down_proj"
    cell = {"F_R832": _cell("F", 832, 0.4, 100, 1.5)}
    plan, census = _write_fixture(tmp_path / "duplicate", [(qname, cell), (qname, cell)])
    with pytest.raises(SparseRateDatasetError, match="duplicate measured rate"):
        build_snapshot(plan, census, tmp_path / "duplicate" / "out")

    left, right = "model.layers.3.mlp.down_proj", "model.layers.4.mlp.down_proj"
    plan, census = _write_fixture(tmp_path / "source", [(left, cell), (right, cell)], model="other-model")
    first = json.loads(plan.read_text())
    # Make just the second payload differ from the exact source identity.
    second_cost = tmp_path / "source" / "workspace" / "rows" / "row-0001" / "cost.pkl"
    second_cost.write_bytes(pickle.dumps(_payload(right, cell, model="third-model")))
    with pytest.raises(SparseRateDatasetError, match="source/calibration identity mismatch"):
        build_snapshot(plan, census, tmp_path / "source" / "out")
    assert first["rows"][1]["dir"] == str(second_cost.parent)


def test_snapshot_hashes_exact_parsed_bytes_and_is_reproducible(tmp_path):
    qname = "model.layers.3.mlp.down_proj"
    plan, census = _write_fixture(tmp_path, [(qname, {"F_R832": _cell("F", 832, 0.4, 100, 1.5)})])
    first = build_snapshot(plan, census, tmp_path / "one")
    repeated = build_snapshot(plan, census, tmp_path / "one")
    second = build_snapshot(plan, census, tmp_path / "two")

    manifest = json.loads((tmp_path / "one" / MANIFEST_NAME).read_text())
    assert repeated["status"] == "verified_existing"
    assert first["npz_sha256"] == second["npz_sha256"]
    assert manifest["input_files"]["plan"]["sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()
    assert manifest["input_files"]["census"]["sha256"] == hashlib.sha256(census.read_bytes()).hexdigest()
    cost = tmp_path / "workspace" / "rows" / "row-0000" / "cost.pkl"
    assert manifest["input_files"]["rows/row-0000"]["sha256"] == hashlib.sha256(cost.read_bytes()).hexdigest()

    plan.write_text(plan.read_text() + "\n")
    with pytest.raises(SparseRateDatasetError, match="identity differs"):
        build_snapshot(plan, census, tmp_path / "one")


@pytest.mark.parametrize("entrypoint", ["script", "module"])
def test_snapshot_dotted_layer_owner_preserves_parsed_results(tmp_path, monkeypatch, entrypoint):
    """#2779 / #1303: exercise outputs and delegation, not regex cache identity."""
    from prismaquant import qnames

    owner = qnames.DOTTED_LAYER_QNAME
    searched = []

    def search(name):
        searched.append(name)
        return owner.search(name)

    # Patch before a fresh import so both import styles can use the owner.
    monkeypatch.setattr(qnames, "DOTTED_LAYER_QNAME", SimpleNamespace(search=search))
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "sparse_rate_dataset.py"
    spec = importlib.util.spec_from_file_location("_sparse_rate_qname_2779", script)
    consumer = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, consumer)
    spec.loader.exec_module(consumer)
    searched.clear()

    cases = [
        ("model.layers.7.mlp.down_proj", 7, "dense", "down_proj"),
        ("model.layers.7.mlp.experts.2.gate_proj", 7, "expert", "gate_proj"),
        ("layers.7.mlp.down_proj", None, None, None),
        ("model.layers.7", None, None, None),
        ("model.layers.3.layers.19.mlp.down_proj", 3, "dense", "down_proj"),
        ("model.layers.bad.layers.19.mlp.down_proj", 19, "dense", "down_proj"),
        ("model.layers.٧.mlp.experts.2.up_proj", 7, "expert", "up_proj"),
        ("model.embed_tokens", None, None, None),
        # Keep this consumer's substring-based expert classification.
        ("model.layers.7.mlp.experts.shared.down_proj", 7, "expert", "down_proj"),
    ]
    for index, (qname, layer, role, structure) in enumerate(cases):
        case_dir = tmp_path / str(index)
        plan, census = _write_fixture(case_dir, [
            (qname, {"F_R832": _cell("F", 832, 0.4, 100, 1.5)}),
        ])
        out = case_dir / "out"
        if layer is None:
            with pytest.raises(consumer.SparseRateDatasetError, match="qname has no layer") as exc:
                consumer.build_snapshot(plan, census, out)
            assert qname in str(exc.value)
            assert not (out / consumer.DATASET_NAME).exists()
        else:
            result = consumer.build_snapshot(plan, census, out)
            assert result["status"] == "written"
            with np.load(out / consumer.DATASET_NAME, allow_pickle=False) as snapshot:
                assert snapshot["qnames"].tolist() == [qname]
                assert snapshot["layers"].tolist() == [layer]
                assert snapshot["roles"].tolist() == [role]
                assert snapshot["structures"].tolist() == [structure]
                assert snapshot["values"].tolist() == [[0.4]]

    # -E prevents a runner's PYTHONPATH from masking direct-script import bugs.
    args = [str(script)] if entrypoint == "script" else ["-m", "experiments.sparse_rate_dataset"]
    help_result = subprocess.run(
        [sys.executable, "-E", *args, "--help"], cwd=root,
        capture_output=True, text=True, timeout=30,
    )
    assert help_result.returncode == 0, help_result.stdout + help_result.stderr
    assert "--plan" in help_result.stdout
    assert "--census" in help_result.stdout
    # Parsed behavior alone already passes with the duplicate local grammar.
    assert searched == [case[0] for case in cases], "snapshot bypassed the dotted-layer owner"
