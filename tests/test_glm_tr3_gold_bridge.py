"""New measured TR3 records interoperate; retained raw output is never upgraded."""
from __future__ import annotations

import sys
import copy
import json

import pytest
import numpy as np
from pathlib import Path

from experiments import measure_glm_tr3_vllm as served
from test_glm_tr3_dual_teacher import BASE_ARGS
from prismaquant import release_receipts, shipcard, shipcard_cli
from tools.gold_measurement_fidelity import tr3_kl_fidelity
from test_release_receipts import _inputs
from test_glm_tr3_full_vocab import fake_measure, _FakeLLM


def _completed_result():
    fidelity = tr3_kl_fidelity(vocab_size=served.VOCAB_SIZE, n_windows=served.WINDOW_COUNT,
                              seqlen=served.CONTEXT_LENGTH)
    calibration = {"schema": "prismaquant.glm_tr3_gold_calibration.v1",
                   "panel_sha256": served.PANEL_SHA256, "teacher_sha256": "a" * 64}
    return {"schema": "prismaquant.glm_tr3_full_vocabulary_kl/1", "passed": True,
            "measurement_fidelity": fidelity, "calibration_contract": calibration,
            "calibration_contract_sha256": served.canonical_sha256(calibration),
            "teacher_source_execution": {"fixture": True},
            "runtime_binding": {"teacher_sha256": "a" * 64,
                                "producer_identity": {"gold_source": {"tools": {
                                    "git_commit": "b" * 40, "git_dirty": False}}}},
            "serve_manifest": {"schema": "prismaquant.serve_manifest/1",
                               "serve_fingerprint": "c" * 64},
            "summary": {"mean": 0.125, "windows": [
                {"window_id": f"final-{index:04d}", "positions": served.CONTEXT_LENGTH - 1}
                for index in range(served.WINDOW_COUNT)]}}


def test_supported_tr3_cli_can_request_a_producer_gold_record(monkeypatch):
    seen = {}
    monkeypatch.setattr(served, "measure", lambda args: seen.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["measure", *BASE_ARGS,
                                     "--gold-record-out", "records/gold.kl.json"])
    served.main()
    assert seen["gold_record_out"] == "records/gold.kl.json"


def test_qualification_only_cli_refuses_a_gold_output_before_measure(monkeypatch):
    monkeypatch.setattr(served, "measure", lambda args: pytest.fail("launched qualification"))
    monkeypatch.setattr(sys, "argv", ["measure", *BASE_ARGS, "--qualify-hook",
                                     "--gold-record-out", "records/gold.kl.json"])
    with pytest.raises(SystemExit):
        served.main()


def test_new_producer_record_interoperates_without_closing_other_slots(tmp_path):
    model, card, records, window = _inputs(tmp_path)
    record = served._tr3_gold_record(
        _completed_result(), model_sha=shipcard.compute_model_sha(model), model=model,
        spec_decode_detected=False)
    (records / "gold.kl.json").write_text(json.dumps(record))
    before = card.read_bytes()
    report = release_receipts.ingest_receipts(card, window=window, records_dir=records)
    assert report["preflight_problems"] == []
    assert report["prepared_slots"] == ["gold.kl"]
    assert card.read_bytes() == before
    report = release_receipts.ingest_receipts(card, window=window, records_dir=records, apply=True)
    assert report["applied_slots"] == ["gold.kl"]
    assert shipcard.load_shipcard(card)["slots"]["gold.kl"] == record
    assert report["problems"]  # Other independent release gates remain missing.
    assert not (model / "serve_manifest.json").exists()


@pytest.mark.parametrize("damage", ["legacy", "qualification", "partial", "positions", "dirty"])
def test_no_legacy_or_incomplete_measurement_can_mint_the_new_record(damage):
    result = _completed_result()
    if damage == "legacy":
        result.pop("measurement_fidelity")
    elif damage == "qualification":
        result["schema"] = "prismaquant.glm_tr3_hook_qualification/1"
    elif damage == "partial":
        result["summary"]["windows"].pop()
    elif damage == "positions":
        result["summary"]["windows"][0]["positions"] -= 1
    else:
        result["runtime_binding"]["producer_identity"]["gold_source"]["tools"]["git_dirty"] = True
    with pytest.raises(ValueError):
        served._tr3_gold_record(result, model_sha="d" * 64, model="fixture",
                                spec_decode_detected=False)


@pytest.mark.parametrize("spec", [None, True, 0])
def test_gold_record_does_not_infer_disabled_speculation(spec):
    with pytest.raises(ValueError, match="actually observed"):
        served._tr3_gold_record(_completed_result(), model_sha="d" * 64,
                                model="fixture", spec_decode_detected=spec)


def test_uniform_control_cli_preserves_the_producer_gold_record(tmp_path, monkeypatch):
    model, card, _, _ = _inputs(tmp_path)
    from test_shipcard import _artifact
    control = _artifact(tmp_path, name="control", weight_bytes=b"other weights")
    arm = served._tr3_gold_record(
        _completed_result(), model_sha=shipcard.compute_model_sha(control), model=control,
        spec_decode_detected=False)
    arm_path = tmp_path / "control-gold.json"
    arm_path.write_text(json.dumps(arm))
    block_path = tmp_path / "block.json"
    block_path.write_text(json.dumps({"verdict": {"measured": True}}))
    captured = {}

    def construct(**kwargs):
        captured.update(copy.deepcopy(kwargs))
        return shipcard.make_record(slot="uniform_control", tool="fixture", passed=False,
                                    model_sha=shipcard.compute_model_sha(model))

    monkeypatch.setattr(shipcard_cli, "make_uniform_control_record", construct)
    monkeypatch.setattr(shipcard_cli, "uniform_control_summary",
                        lambda *args, **kwargs: {"detail": "fixture"})
    shipcard_cli.main(["fill-control", str(card), "--control-block", str(block_path),
                       "--control-record", str(arm_path), "--control-model-dir", str(control)])
    assert captured["control_arm"] == arm


@pytest.mark.parametrize("damage", ["schema", "slot", "missing_schema", "identity", "spec", "count"])
def test_control_producer_record_refuses_drift_without_writing(tmp_path, damage):
    model, card, _, _ = _inputs(tmp_path)
    result = served._tr3_gold_record(
        _completed_result(), model_sha=shipcard.compute_model_sha(model), model=model,
        spec_decode_detected=False)
    if damage == "schema":
        result["measurement_schema"] = "forged"
    elif damage == "slot":
        result["slot"] = "gold.ppl"
    elif damage == "missing_schema":
        result.pop("measurement_schema")
    elif damage == "identity":
        result["model_sha"] = "0" * 64
    elif damage == "spec":
        result["spec_decode_detected"] = None
    else:
        result["metrics"].pop("n_positions")
        result["metrics"].pop("n_samples")
    arm = tmp_path / "control.json"
    arm.write_text(json.dumps(result))
    block = tmp_path / "block.json"
    block.write_text(json.dumps({"verdict": {"measured": True}}))
    before = card.read_bytes()
    assert shipcard_cli.main(["fill-control", str(card), "--control-block", str(block),
                              "--control-record", str(arm), "--control-model-dir", str(model)]) == 2
    assert card.read_bytes() == before


@pytest.mark.parametrize("alias", ["result", "qualification", "existing"])
def test_gold_output_cannot_overwrite_prior_evidence_or_other_outputs(tmp_path, alias):
    from types import SimpleNamespace
    output, qualification, gold = (tmp_path / name for name in ("result", "qualification", "gold"))
    if alias == "result":
        gold = output
    elif alias == "qualification":
        gold = qualification
    else:
        gold.write_bytes(b"prior evidence")
    args = SimpleNamespace(gold_record_out=str(gold), output=str(output), qualify_hook=False,
                           qualify_then_score=str(qualification))
    with pytest.raises(ValueError, match="output|distinct"):
        served.measure(args)


@pytest.mark.parametrize("damage", [None, "artifact_drift", "spec_drift"])
def test_actual_scorer_path_emits_a_record_only_after_its_owned_fences(
        fake_measure, monkeypatch, tmp_path, damage):
    """CPU engine/identity stand-ins, but the real scorer and ingestion owners."""
    from experiments import glm_tr3_full_vocab as domain
    from test_shipcard import _open_card

    args, _ = fake_measure
    model = Path(args.model)
    (model / "config.json").write_text('{"model_type":"fixture"}')
    (model / "model.safetensors").write_bytes(b"fixture weights")
    card = _open_card(tmp_path, model)
    records = tmp_path / "records"
    records.mkdir()
    args.gold_record_out = str(records / "gold.kl.json")
    old_panel = served.load_panel(args.panel)[0]
    panel = copy.deepcopy(old_panel)
    panel["windows"] = [{**old_panel["windows"][0], "window_id": f"final-{index:04d}"}
                        for index in range(served.WINDOW_COUNT)]
    tokens = [(np.arange(served.CONTEXT_LENGTH),)] * served.WINDOW_COUNT
    teacher = served.load_teacher(args.teacher, args.teacher_sha256, panel)
    teacher["windows"] = [{"window_id": window["window_id"]} for window in panel["windows"]]
    monkeypatch.setattr(served, "load_panel", lambda *a, **k: (panel, tokens))
    monkeypatch.setattr(served, "summarize_panel", domain.summarize_panel)
    monkeypatch.setattr(served, "producer_identity", lambda: {
        "gold_source": {"tools": {"git_commit": "b" * 40, "git_dirty": False}}})
    observations = []

    def manifest(**kwargs):
        observations.append(kwargs)
        assert kwargs["require_engine_descendant"] is True and "artifact_dir" not in kwargs
        assert kwargs["extra"]["runtime_binding"]["shipcard_model_sha"] == shipcard.compute_model_sha(model)
        return {"schema": "prismaquant.serve_manifest/1", "serve_fingerprint": "c" * 64}

    monkeypatch.setattr(served, "self_manifest", manifest)
    if damage == "artifact_drift":
        real_sha = shipcard.compute_model_sha
        checks = []

        def changing_sha(path):
            checks.append(path)
            return real_sha(path) if len(checks) == 1 else "0" * 64

        monkeypatch.setattr(shipcard, "compute_model_sha", changing_sha)
    elif damage == "spec_drift":
        checks = []
        monkeypatch.setattr(served, "refuse_if_spec_decode", lambda **kwargs:
                            checks.append(kwargs) or (False if len(checks) == 1 else True))
    if damage is not None:
        with pytest.raises(ValueError, match="changed while scoring"):
            served.measure(args)
        assert not Path(args.gold_record_out).exists() and not Path(args.output).exists()
        assert _FakeLLM.instances[0].removed
        return
    result = served.measure(args)
    record = json.loads(Path(args.gold_record_out).read_bytes())
    assert result["measurement_fidelity"]["vocabulary"] == "full"
    assert result["measurement_fidelity"]["accumulation_dtype"] == "float64"
    assert record["metrics"]["n_positions"] == 51175 and record["metrics"]["n_samples"] == 25
    assert record["runtime_binding"] == result["runtime_binding"]
    assert observations and _FakeLLM.instances[0].generated == 25 and _FakeLLM.instances[0].removed
    report = release_receipts.ingest_receipts(card, records_dir=records, apply=True)
    assert report["applied_slots"] == ["gold.kl"]
    assert shipcard.load_shipcard(card)["slots"]["gold.kl"] == record
    assert report["problems"] and not (model / "serve_manifest.json").exists()
