"""Offline receipt ingestion must preserve the existing shipcard refusals."""
from __future__ import annotations

import json
import pytest

import prismaquant.release_receipts as ingest
from prismaquant import shipcard
from test_shipcard import _artifact, _open_card, _native_record
from test_tessera_route_trace_gate import (
    _artifact as _trace_artifact, _trace_files, _fixture_traces, contract,
)


def _inputs(tmp_path):
    model = _artifact(tmp_path)
    card = _open_card(tmp_path, model)
    records = tmp_path / "records"
    records.mkdir()
    window = tmp_path / "window.json"
    window.write_text(json.dumps({"schema": "prismaquant.pact_u4.arm/1",
                                 "artifact": str(model), "phases": {}, "kl": {}}))
    return model, card, records, window


def _save_record(card, records):
    record = _native_record(
        "native_export.eager", shipcard.load_shipcard(card)["model_sha"],
        {"arm": "eager", "enforce_eager": True, "generated_chars": 128, "max_new_tokens": 16},
    )
    (records / "native_export.eager.json").write_text(json.dumps(record))
    return record


@pytest.mark.parametrize("damage", [None, "spec_unknown", "fingerprint", "nonfinite", "missing_count"])
def test_full_producer_receipt_set_uses_unchanged_verifier(tmp_path, damage):
    from test_publish_artifact import _close_all_slots
    model, card, records, _ = _inputs(tmp_path)
    _close_all_slots(model)
    complete = shipcard.load_shipcard(card)
    assert shipcard.verify(complete, model_dir=model) == []
    for slot, record in complete["slots"].items():
        if record is not None:
            (records / f"{slot}.json").write_text(json.dumps(record))
    gold = json.loads((records / "gold.kl.json").read_text())
    if damage == "spec_unknown":
        gold["spec_decode_detected"] = None
    elif damage == "fingerprint":
        gold["serve_fingerprint"] = "bad"
    elif damage == "nonfinite":
        gold["metrics"]["kl_mean"] = float("nan")
    elif damage == "missing_count":
        gold["metrics"].pop("n_positions", None)
        gold["metrics"].pop("n_samples", None)
    (records / "gold.kl.json").write_text(json.dumps(gold))
    complete["slots"] = dict.fromkeys(complete["slots"])
    shipcard.write_shipcard(card, complete)
    before = card.read_bytes()
    if damage is None:
        assert ingest.main([str(card), "--records-dir", str(records)]) == 0
        assert card.read_bytes() == before
        assert ingest.main([str(card), "--records-dir", str(records), "--apply"]) == 0
        assert shipcard.verify(shipcard.load_shipcard(card), model_dir=model) == []
    else:
        assert ingest.main([str(card), "--records-dir", str(records), "--apply"]) != 0
        assert card.read_bytes() == before


def test_dry_default_then_partial_apply_preserves_record_and_refusal(tmp_path):
    model, card, records, window = _inputs(tmp_path)
    record = _save_record(card, records)
    before = card.read_bytes()
    report = ingest.ingest_receipts(card, window=window, records_dir=records)
    assert card.read_bytes() == before
    assert report["prepared_slots"] == ["native_export.eager"]
    assert report["applied_slots"] == []
    assert report["problems"]  # Missing measurements never turn into a pass.
    report = ingest.ingest_receipts(card, window=window, records_dir=records, apply=True)
    assert report["applied_slots"] == ["native_export.eager"]
    assert shipcard.load_shipcard(card)["slots"]["native_export.eager"] == record
    assert report["problems"] == shipcard.verify(shipcard.load_shipcard(card), model_dir=model)


@pytest.mark.parametrize("field,value", [
    ("model_sha", "f" * 64), ("model_sha", None), ("slot", "native_export.graph"),
    ("passed", False), ("metrics", {"arm": "graph", "enforce_eager": False}),
])
def test_mutated_producer_record_refuses_without_writes(tmp_path, field, value):
    _, card, records, window = _inputs(tmp_path)
    record = _save_record(card, records)
    record[field] = value
    (records / "native_export.eager.json").write_text(json.dumps(record))
    before = card.read_bytes()
    report = ingest.ingest_receipts(card, window=window, records_dir=records, apply=True)
    assert report["input_problems"] or report["preflight_problems"]
    assert report["applied_slots"] == []
    assert card.read_bytes() == before


def test_changed_artifact_is_not_rebound(tmp_path):
    model, card, records, window = _inputs(tmp_path)
    _save_record(card, records)
    (model / "config.json").write_text('{"model_type":"changed"}')
    before = card.read_bytes()
    report = ingest.ingest_receipts(card, window=window, records_dir=records, apply=True)
    assert report["preflight_problems"]
    assert card.read_bytes() == before


@pytest.mark.parametrize("mutation", ["wrong_artifact", "wrong_schema", "duplicate", "nonfinite", "unknown_slot"])
def test_malformed_inputs_fail_closed(tmp_path, mutation):
    _, card, records, window = _inputs(tmp_path)
    if mutation == "wrong_artifact":
        window.write_text(json.dumps({"schema": "prismaquant.pact_u4.arm/1", "artifact": str(tmp_path)}))
    elif mutation == "wrong_schema":
        window.write_text('{"schema":"unknown"}')
    elif mutation == "duplicate":
        (records / "gold.kl.json").write_text('{"slot":"gold.kl","slot":"gold.ppl"}')
    elif mutation == "nonfinite":
        (records / "gold.kl.json").write_text('{"slot":"gold.kl","metric":NaN}')
    else:
        (records / "invented.json").write_text('{}')
    before = card.read_bytes()
    assert ingest.main([str(card), "--window", str(window), "--records-dir", str(records), "--apply"]) == 2
    assert card.read_bytes() == before


def test_current_tr3_is_reported_not_forged_into_gold(tmp_path):
    _, card, records, window = _inputs(tmp_path)
    result = tmp_path / "kl.json"
    result.write_text(json.dumps({"schema": "prismaquant.glm_tr3_full_vocabulary_kl/1",
                                  "passed": True, "serve_manifest": {"fingerprint": "a" * 64}}))
    payload = json.loads(window.read_text())
    payload["kl"] = {"full": {"result": str(result)}}
    window.write_text(json.dumps(payload))
    report = ingest.ingest_receipts(card, window=window, records_dir=records, apply=True)
    assert report["unsupported_outputs"]
    assert shipcard.load_shipcard(card)["slots"]["gold.kl"] is None
    assert not (card.parent / "serve_manifest.json").exists()


def test_body_only_route_success_is_never_promoted(tmp_path):
    _, card, records, window = _inputs(tmp_path)
    payload = json.loads(window.read_text())
    payload["phases"] = {"tr3": {"route_trace": {"verdict": {
        "body_scope": {"verdict": {"status": "AGREE"}}, "gate": {"status": "DISAGREE"}}}}}
    window.write_text(json.dumps(payload))
    report = ingest.ingest_receipts(card, window=window, records_dir=records)
    assert "route.trace" not in report["prepared_slots"]


@pytest.mark.parametrize("damage", [None, "missing_rank", "same_path", "wrong_contract", "body_only"])
def test_window_replays_raw_full_config_trace(tmp_path, contract, damage):
    model, card = _trace_artifact(tmp_path)
    traces = _fixture_traces()
    if damage == "wrong_contract":
        # Mutate carried bytes, not the wrapper's claimed verdict.
        traces[0][1]["schema"] = "invalid"
    paths = _trace_files(tmp_path, traces)
    if damage == "missing_rank":
        from pathlib import Path
        Path(paths[1]).unlink()
    if damage == "same_path":
        paths[1] = paths[0]
    verdict = {"schema": "prismaquant.pact_u4.route_trace/1", "platform": "sm_121",
               "traces": dict(zip(("rank0", "rank1"), paths, strict=True)),
               "gate": {"status": "AGREE"}}
    payload = {"schema": "prismaquant.pact_u4.arm/1", "artifact": str(model),
               "phases": {"2c": {"route_trace": {"verdict": verdict}}},
               "route_trace_full_config": {"source_phase": "2c", "verdict": verdict}}
    if damage == "body_only":
        payload.pop("route_trace_full_config")
        verdict["body_scope"] = {"verdict": {"status": "AGREE"}}
    window = tmp_path / "window.json"
    window.write_text(json.dumps(payload))
    before = card.read_bytes()
    result = ingest.ingest_receipts(card, window=window)
    assert card.read_bytes() == before
    if damage is None:
        assert result["prepared_slots"] == ["route.trace"]
        assert not result["input_problems"]
        assert not any(p.startswith("route.trace:") for p in result["problems"])
    else:
        assert "route.trace" not in result["prepared_slots"]
        assert result["problems"]


def test_lane_declares_existing_uniform_producer_and_fill_path():
    from prismaquant.lane_spec import load_lane_spec
    gate = load_lane_spec("tessera").gate("uniform_control")
    assert gate is not None
    assert gate.shipcard_slot == "uniform_control"
    assert "-m tessera.uniform_control verify" in gate.runner
    assert "fill-control" in gate.runner
    assert "--max-relative-slack" in gate.runner
    assert " && " in gate.runner  # Never fill from a stale report after producer failure.
