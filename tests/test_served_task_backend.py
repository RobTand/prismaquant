"""Port PR #2470 binding outcomes using the real, archived public oracle.

Every case uses tiny synthetic CPU bytes, not W1 launch or D50 evidence.
"""
import importlib
import importlib.util

import pytest

from _served_task_fixture import case, resign, write_json
from prismaquant import task_suite

pytestmark = pytest.mark.task_suite


def backend():
    name = "prismaquant.served_task_backend"
    assert importlib.util.find_spec(name) is not None, (
        "PR #2470 served binding implementation has not been reconciled onto main")
    return importlib.import_module(name)


def test_complete_binding_uses_public_authority(tmp_path):
    config, witness, expected = case(tmp_path)
    record = task_suite.preflight_tasks(config)["binding"]
    assert record["verdict"] == "pass"
    assert record["witness_fingerprint"] == witness["fingerprint"]
    assert record["public_verifier"]["verdict"] == "valid"
    for key in ("endpoint", "served_alias", "attempt_id", "ranks"):
        assert record[key] == expected[key]
    facts = record.get("verified_facts", record)
    assert facts["artifacts"] == expected["artifacts"]
    assert facts["tokenizer"] == expected["tokenizer"]


def test_backend_binding_requires_public_process_too(tmp_path):
    config, _, _ = case(tmp_path)
    module = backend()
    record = module.bind_served_task(config)
    assert record["public_verifier"]["verdict"] == "valid"
    del config["backend"]["serving_runtime"]["verifier"]
    with pytest.raises(ValueError, match="verifier"):
        module.bind_served_task(config)


@pytest.mark.parametrize("field", ["schema", "listener", "launch", "lifetime",
                                   "artifacts", "tokenizer", "byte_coverage",
                                   "qualification_scope", "fingerprint"])
def test_missing_observed_fact_refuses(tmp_path, field):
    config, witness, _ = case(tmp_path)
    del witness[field]
    write_json(config_path(config, "witness"), witness)
    with pytest.raises(ValueError, match="witness|public verifier|receipt|runtime"):
        task_suite.preflight_tasks(config)


def config_path(config, key):
    from pathlib import Path
    return Path(config["backend"]["serving_runtime"][key])


@pytest.mark.parametrize("field,value", [
    ("endpoint", "http://127.0.0.1:8143"),
    ("served_alias", "other"),
    ("attempt_id", "fixture/fixture-boot/10/11"),
    ("ranks", [0]),
    ("ranks", [1, 0]),
    ("ranks", [0, 0]),
])
def test_independent_expected_deployment_mismatch_refuses(tmp_path, field, value):
    config, _, expected = case(tmp_path)
    # Config and independent expectation agree; the producer must still refuse
    # their disagreement with actual recorded runtime facts.
    config["backend"]["serving_runtime"][field] = value
    expected[field] = value
    write_json(config_path(config, "expected"), expected)
    with pytest.raises(ValueError, match="rank|endpoint|alias|attempt|refus"):
        task_suite.preflight_tasks(config)


@pytest.mark.parametrize("damage", ["alias-only", "size-only", "artifact-digest",
                                    "tokenizer-digest", "vocab", "special-id"])
def test_independent_expected_bytes_and_loaded_tokenizer_refuse(tmp_path, damage):
    config, _, expected = case(tmp_path)
    assert task_suite.validate_config(config) is config
    if damage == "alias-only":
        expected["artifacts"] = {}
        expected["tokenizer"]["files"] = {}
    elif damage == "size-only":
        del expected["artifacts"]["model.safetensors"]["sha256"]
    elif damage == "artifact-digest":
        expected["artifacts"]["model.safetensors"]["sha256"] = "0" * 64
    elif damage == "tokenizer-digest":
        expected["tokenizer"]["files"]["tokenizer.json"]["sha256"] = "0" * 64
    elif damage == "vocab":
        expected["tokenizer"]["vocab"] = {"<s>": 1, "a": 0}
    else:
        expected["tokenizer"]["special_ids"]["bos"] = 1
    write_json(config_path(config, "expected"), expected)
    with pytest.raises(ValueError, match="artifact|tokenizer|refus"):
        task_suite.preflight_tasks(config)


@pytest.mark.parametrize("damage", ["wrong-schema", "incomplete-ranks", "rank-disagreement",
                                    "coverage", "fingerprint", "scope", "tokenizer-backend"])
def test_inconsistent_observed_evidence_refuses(tmp_path, damage):
    config, witness, _ = case(tmp_path)
    if damage == "wrong-schema":
        witness["schema"] = "unknown.v1"
    elif damage == "incomplete-ranks":
        witness["artifacts"].pop()
    elif damage == "rank-disagreement":
        witness["artifacts"][1]["models"][0]["files"]["model.safetensors"]["sha256"] = "1" * 64
    elif damage == "coverage":
        witness["byte_coverage"]["tensor_payload_bytes"] = 8
    elif damage == "scope":
        witness["qualification_scope"] = "served_task_quality"
    elif damage == "tokenizer-backend":
        witness["tokenizer"]["backend"]["model"]["vocab"] = {"<s>": 1, "a": 0}
    resign(witness)
    if damage == "fingerprint":
        witness["fingerprint"] = "0" * 64
    write_json(config_path(config, "witness"), witness)
    with pytest.raises(ValueError, match="witness|public verifier|receipt|runtime"):
        task_suite.preflight_tasks(config)


@pytest.mark.parametrize("filename,dir_key", [("model.safetensors", "served_dir"),
                                              ("tokenizer.json", "tokenizer_dir")])
@pytest.mark.parametrize("dev", ["0", "1"])
def test_same_size_file_substitution_refuses_in_every_mode(tmp_path, monkeypatch, filename, dir_key, dev):
    from pathlib import Path
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev)
    config, _, _ = case(tmp_path)
    path = Path(config["backend"]["serving_runtime"][dir_key]) / filename
    original = path.read_bytes()
    replacement = original[:-1] + bytes([original[-1] ^ 1])
    path.write_bytes(replacement)
    assert len(replacement) == len(original)
    with pytest.raises(ValueError, match="public verifier|refus|bytes|digest"):
        task_suite.preflight_tasks(config)


@pytest.mark.parametrize("text,reason", [('{' + '"schema":"a","schema":"b"}', "duplicate"),
                                         ('{"schema":NaN}', "strict|finite|constant")])
def test_strict_reader_preserved(tmp_path, text, reason):
    module = backend()
    path = tmp_path / "witness.json"
    path.write_text(text)
    with pytest.raises(ValueError, match=reason):
        module.read_witness(path)
