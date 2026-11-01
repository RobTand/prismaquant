"""Synthetic receipts checked by the external numerical owner, never GPU evidence."""
import copy

import pytest

from prismaquant.shipcard import verify
from tests.graph_receipt_fixtures import replace_receipt
from test_graph_shipcard_receipt import graph_card


@pytest.fixture
def numerical_owner():
    owner = pytest.importorskip("tessera.graph_receipt")
    if not callable(getattr(owner, "population_refusal", None)):
        pytest.skip("installed graph verifier lacks the required shared population rule")
    return owner


def test_real_shared_rule_accepts_complete_synthetic_record(tmp_path, numerical_owner):
    model, card, _, _ = graph_card(tmp_path, owner=numerical_owner)
    assert verify(card, model_dir=model, required=["native_export.graph"]) == []


@pytest.mark.parametrize("mode", ["0", "1"])
@pytest.mark.parametrize("field,value", [
    ("image", "registry/other@sha256:" + "2" * 64),
    ("model_config_sha256", "2" * 64),
    ("compilation_config", {"mode": "FULL_AND_PIECEWISE"}),
    ("speculative_tokens", 1), ("max_model_len", 9000),
    ("max_num_seqs", 8), ("tensor_parallel_size", 1), ("fabric", "roce"),
])
def test_scope_mismatches_refuse_in_both_modes(tmp_path, monkeypatch, numerical_owner, mode, field, value):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    model, card, record, receipt = graph_card(tmp_path, owner=numerical_owner)
    # Change the receipt's measured tuple so producer observations stay coherent.
    if field in {"image", "fabric"}:
        receipt["runtime"][field] = value
    elif field == "model_config_sha256":
        receipt["model"]["config_sha256"] = value
    else:
        receipt["arms"][0][field] = value
    numerical_owner.finish(receipt)
    replace_receipt(record, receipt)
    problems = verify(card, model_dir=model, required=["native_export.graph"])
    assert any(field in problem for problem in problems), problems


@pytest.mark.parametrize("mode,refused", [("0", True), ("1", False)])
def test_only_source_provenance_uses_seal_policy(tmp_path, monkeypatch, capsys, numerical_owner, mode, refused):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    model, card, record, _ = graph_card(tmp_path, owner=numerical_owner)
    record["metrics"]["serve_scope"]["tessera_src_sha256"] = "4" * 64
    problems = verify(card, model_dir=model, required=["native_export.graph"])
    assert bool(problems) is refused, problems
    if not refused:
        assert "[DEV-MODE]" in capsys.readouterr().out
    assert record["metrics"]["serve_scope"]["tessera_src_sha256"] == "4" * 64


@pytest.mark.parametrize("damage", [
    "v1", "forged_equal", "incomplete_population", "missing_capture",
    "missing_replay", "missing_class_replay",
])
def test_shared_rule_rederives_numerical_and_replay_refusals(tmp_path, numerical_owner, damage):
    model, card, record, receipt = graph_card(tmp_path, owner=numerical_owner)
    arm = receipt["arms"][0]
    if damage == "v1":
        receipt["schema"] = numerical_owner.SCHEMA_V1
    elif damage == "forged_equal":
        arm["passes"][1]["members"] -= 1
    elif damage == "incomplete_population":
        arm["passes"].pop()
    elif damage == "missing_capture":
        arm["graph"]["managers"] = {}
    elif damage == "missing_replay":
        arm["graph"]["managers"]["full"]["replayed_sizes"] = {}
    elif damage == "missing_class_replay":
        arm["graph"]["classes"]["replays"] = {}
    receipt["verdict"] = "equal"
    replace_receipt(record, receipt)
    problems = verify(card, model_dir=model, required=["native_export.graph"])
    assert any("graph" in problem and ("schema" in problem or "refused" in problem)
               for problem in problems), problems
