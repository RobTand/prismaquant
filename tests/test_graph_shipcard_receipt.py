"""Graph release evidence must exceed a successful generation smoke."""
import json
from pathlib import Path

import pytest

from prismaquant.shipcard import open_shipcard, verify
from prismaquant.shipcard_slots import make_record
from tests.graph_receipt_fixtures import graph_fixture


def test_graph_smoke_without_equality_receipt_is_refused(tmp_path):
    model = tmp_path / "artifact"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}')
    (model / "model.safetensors").write_bytes(b"owned tiny weights")
    card = open_shipcard(model, build={"achieved_bpp": {"value": 4.75}})
    slot = "native_export.graph"
    card["slots"][slot] = make_record(
        slot=slot, tool="validate_native_export.py", passed=True,
        model_sha=card["model_sha"], git_commit="a" * 40,
        metrics={"arm": "graph", "enforce_eager": False,
                 "generated_chars": 128, "max_new_tokens": 16},
    )
    problems = verify(card, model_dir=model, required=[slot])
    assert any(p.startswith(slot + ":") and "receipt" in p for p in problems), problems


def graph_card(tmp_path, *, owner=None):
    model = tmp_path / "artifact"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}')
    (model / "model.safetensors").write_bytes(b"owned tiny weights")
    card = open_shipcard(model, build={"achieved_bpp": {"value": 4.75}})
    record, receipt = graph_fixture(model, owner=owner)
    card["slots"]["native_export.graph"] = record
    return model, card, record, receipt


@pytest.fixture
def boundary_verifier(monkeypatch):
    """Isolate record/byte binding from the separately tested numerical owner."""
    from prismaquant import tessera_lane

    calls = []
    monkeypatch.setattr(tessera_lane, "graph_equality_verify",
                        lambda receipt, scope: calls.append((receipt, scope)))
    return calls


def test_producer_record_reaches_lane_verifier(tmp_path, boundary_verifier):
    model, card, record, receipt = graph_card(tmp_path)
    assert verify(card, model_dir=model, required=["native_export.graph"]) == []
    assert boundary_verifier == [(receipt, {
        **record["metrics"]["serve_scope"], "arm": "fixture-graph"})]


@pytest.mark.parametrize("damage", [
    "changed_bytes", "malformed", "duplicate_keys", "missing_file",
    "observation", "model_before", "model_after", "receipt_link",
    "resolved", "compilation", "rank_missing", "rank_duplicate",
    "rank_transport", "rank_image", "generation", "action", "arm",
    "commit", "tool", "speculation", "config_bytes",
])
def test_producer_binding_refuses_before_numerical_rule(tmp_path, boundary_verifier, damage):
    model, card, record, receipt = graph_card(tmp_path)
    metrics = record["metrics"]
    observed = metrics["graph_observation"]
    path = Path(metrics["graph_receipt_path"])
    if damage == "changed_bytes":
        path.write_bytes(path.read_bytes() + b" ")
    elif damage in {"malformed", "duplicate_keys"}:
        from prismaquant.digests import bytes_sha256hex

        raw = b"{" if damage == "malformed" else b'{"arms":[],"arms":[]}'
        path.write_bytes(raw)
        metrics["graph_receipt_sha256"] = bytes_sha256hex(raw)
        observed["graph_receipt_sha256"] = metrics["graph_receipt_sha256"]
    elif damage == "missing_file":
        path.unlink()
    elif damage == "observation":
        del metrics["graph_observation"]
    elif damage in {"model_before", "model_after"}:
        observed["model_sha_" + damage.split("_")[1]] = "f" * 64
    elif damage == "receipt_link":
        observed["graph_receipt_sha256"] = "f" * 64
    elif damage == "resolved":
        del observed["resolved"]["compilation_config"]
    elif damage == "compilation":
        observed["compilation_config"] = {"mode": "FULL_AND_PIECEWISE"}
    elif damage == "rank_missing":
        observed["ranks"].pop()
    elif damage == "rank_duplicate":
        observed["ranks"][1]["rank"] = 0
    elif damage in {"rank_transport", "rank_image"}:
        observed["ranks"][1]["fabric" if damage == "rank_transport" else "image"] = "another"
    elif damage == "generation":
        observed["generation"]["outputs"] = ["short"]
    elif damage in {"action", "arm"}:
        observed["equality_" + damage] = "another"
    elif damage in {"commit", "tool"}:
        record["git_commit" if damage == "commit" else "tool"] = None
    elif damage == "speculation":
        record["spec_decode_detected"] = None
    elif damage == "config_bytes":
        (model / "config.json").write_text('{"model_type":"changed"}')
    problems = verify(card, model_dir=model, required=["native_export.graph"])
    assert any(problem.startswith("native_export.graph:") for problem in problems), problems
    assert boundary_verifier == []


def test_same_config_other_weights_cannot_rebind_producer_record(tmp_path, boundary_verifier):
    model, card, record, _ = graph_card(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_bytes((model / "config.json").read_bytes())
    (other / "model.safetensors").write_bytes(b"other tiny weights")
    other_card = open_shipcard(other, build={"achieved_bpp": {"value": 4.75}})
    record["model_sha"] = other_card["model_sha"]
    other_card["slots"]["native_export.graph"] = record
    problems = verify(other_card, model_dir=other, required=["native_export.graph"])
    assert any("model_sha_before" in problem for problem in problems), problems
    assert boundary_verifier == []


def test_partial_graph_record_ingests_without_qualifying_missing_slots(tmp_path, boundary_verifier):
    from prismaquant.release_receipts import ingest_receipts
    from prismaquant.shipcard import load_shipcard, write_shipcard

    model, card, record, _ = graph_card(tmp_path)
    card["slots"]["native_export.graph"] = None
    path = model / "shipcard.json"
    write_shipcard(path, card)
    records = tmp_path / "records"
    records.mkdir()
    (records / "native_export.graph.json").write_text(json.dumps(record))
    report = ingest_receipts(path, records_dir=records, apply=True)
    assert report["preflight_problems"] == []
    assert report["applied_slots"] == ["native_export.graph"]
    assert load_shipcard(path)["slots"]["native_export.graph"] == record
    assert report["status"] == "refused"
    assert "native_export.eager: UNFILLED" in report["problems"]


def test_missing_compatible_verifier_refuses(tmp_path, monkeypatch):
    from prismaquant import tessera_lane
    import sys
    import types

    model, card, record, receipt = graph_card(tmp_path)
    package = types.ModuleType("tessera")
    package.graph_receipt = types.SimpleNamespace(SCHEMA=receipt["schema"], verify=lambda *args: None)
    monkeypatch.setitem(sys.modules, "tessera", package)
    problems = verify(card, model_dir=model, required=["native_export.graph"])
    assert any("missing shared population_refusal" in problem for problem in problems), problems


def test_missing_lane_hook_refuses(tmp_path, monkeypatch):
    from prismaquant import lane_spec

    model, card, _, _ = graph_card(tmp_path)
    original = lane_spec.single_lane_hook
    monkeypatch.setattr(lane_spec, "single_lane_hook", lambda name:
                        None if name == "graph_equality_verify" else original(name))
    problems = verify(card, model_dir=model, required=["native_export.graph"])
    assert any("no lane provides graph equality verification" in problem for problem in problems)


@pytest.mark.parametrize("field,value", [
    ("fabric", []), ("image", {}), ("speculative_tokens", True),
    ("compilation_config", []), ("model_config_sha256", None),
])
def test_malformed_scope_returns_refusal(tmp_path, boundary_verifier, field, value):
    model, card, record, _ = graph_card(tmp_path)
    record["metrics"]["serve_scope"][field] = value
    problems = verify(card, model_dir=model, required=["native_export.graph"])
    assert any(field in problem for problem in problems), problems
    assert boundary_verifier == []


@pytest.mark.parametrize("damage", ["nonfinite_compilation", "nonfinite_operators", "null_operators", "unresolved_input"])
def test_malformed_observed_settings_return_refusal(tmp_path, boundary_verifier, damage):
    model, card, record, _ = graph_card(tmp_path)
    observed = record["metrics"]["graph_observation"]
    if damage == "nonfinite_compilation":
        observed["compilation_config"]["mode"] = float("nan")
    elif damage == "nonfinite_operators":
        observed["resolved"]["ir_op_priority"] = {"operation": float("nan")}
    elif damage == "null_operators":
        observed["resolved"]["custom_ops"] = None
    else:
        observed["resolved"]["compilation_config"] = {"mode": "NONE"}
    problems = verify(card, model_dir=model, required=["native_export.graph"])
    assert any("compilation" in problem or "operator" in problem for problem in problems), problems
    assert boundary_verifier == []


@pytest.mark.parametrize("part", ["compilation_config", "resolved", "operators"])
def test_observed_compilation_preserves_json_types(tmp_path, boundary_verifier, part):
    from tests.graph_receipt_fixtures import replace_receipt

    model, card, record, receipt = graph_card(tmp_path)
    observed = record["metrics"]["graph_observation"]
    record["metrics"]["serve_scope"]["compilation_config"]["mode"] = 1
    observed["compilation_config"]["mode"] = 1
    observed["resolved"]["compilation_config"]["mode"] = 1
    observed["resolved"]["ir_op_priority"] = {"operation": 1}
    receipt["arms"][0]["resolved"]["ir_op_priority"] = {"operation": 1}
    replace_receipt(record, receipt)
    if part == "operators":
        observed["resolved"]["ir_op_priority"]["operation"] = True
    else:
        target = observed["compilation_config"] if part == "compilation_config" else observed["resolved"]["compilation_config"]
        target["mode"] = True
    problems = verify(card, model_dir=model, required=["native_export.graph"])
    assert any("compilation" in problem or "operator" in problem for problem in problems), problems
    assert boundary_verifier == []
