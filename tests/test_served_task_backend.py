"""Consume the Tessera public witness: join it, refuse gaps, bind tasks."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from prismaquant.served_task_backend import (
    bind_served_task,
    is_served_config,
    read_witness,
    verify,
)

PAYLOAD = b"\x01\x02\x03\x04"
TENSORS = {"weight": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}
BACKEND = {"version": "1.0", "model": {"type": "WordLevel",
           "vocab": {"<s>": 0, "a": 1}, "unk_token": "<s>"},
           "added_tokens": [], "truncation": None, "padding": None,
           "normalizer": None, "pre_tokenizer": None, "post_processor": None,
           "decoder": None}


def _owner(pid):
    return {"host": "fixture", "boot_id": "fixture-boot", "pid": pid,
            "start_ticks": 10, "started_unix": 100.0}


def _attempt(owner):
    return (f"{owner['host']}/{owner['boot_id']}/"
            f"{owner['pid']}/{owner['start_ticks']}")


def _source():
    header = json.dumps(TENSORS).encode()
    raw = len(header).to_bytes(8, "little") + header + PAYLOAD
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "data_start": 8 + len(header), "tensors": copy.deepcopy(TENSORS)}


def make_witness(*, world=2):
    loaded = hashlib.sha256(PAYLOAD).hexdigest()
    source = _source()
    model = {"model_path": "artifact",
             "load_started_unix": 200.0, "load_finished_unix": 300.0,
             "files": {"model.safetensors": source},
             "inputs": [{"file": "model.safetensors", "tensor": "weight",
                         "start": 0, "end": 4, "sha256": loaded,
                         "source_sha256": loaded, "target": "weight",
                         "loaded_unix": 250.0}],
             "resident": {"parameter:weight": {"sha256": loaded, "bytes": 4,
                                               "dtype": "torch.uint8",
                                               "shape": [4]}}}
    token_contents = {"tokenizer.json": copy.deepcopy(BACKEND),
                      "tokenizer_config.json": {"bos_token": "<s>"}}
    token_files = {}
    for name, content in token_contents.items():
        raw = json.dumps(content).encode()
        token_files[name] = {"sha256": hashlib.sha256(raw).hexdigest(),
                             "bytes": len(raw), "content": content}
    owner = _owner(10)
    witness = {"schema": "tessera.endpoint_runtime_witness.v1",
               "listener": {"endpoint": "http://127.0.0.1:8142",
                            "served_alias": "fixture", "owner": owner},
               "launch": {"attempt_id": _attempt(owner),
                          "ranks": list(range(world))},
               "lifetime": {"request_id": "fixture-request",
                            "started_unix": 400.0, "finished_unix": 600.0},
               "artifacts": [{"rank": rank, "world_size": world,
                              "owner": _owner(20 + rank),
                              "request_id": "fixture-request",
                              "observed_unix": 500.0,
                              "models": [copy.deepcopy(model)]}
                             for rank in range(world)],
               "tokenizer": {"path": "artifact",
                             "request_id": "fixture-request",
                             "observed_unix": 500.0, "files": token_files,
                             "backend": copy.deepcopy(BACKEND),
                             "vocab": {"<s>": 0, "a": 1},
                             "special_ids": {"bos": 0, "eos": None,
                                             "pad": None, "unk": None,
                                             "sep": None, "cls": None,
                                             "mask": None}},
               "byte_coverage": {"kind": "successful-loader-inputs-and-"
                                  "post-load-resident-state",
                                 "files": ["model.safetensors"],
                                 "tensor_payload_bytes": 4},
               "qualification_scope": "runtime_byte_binding"}
    witness["fingerprint"] = hashlib.sha256(json.dumps(
        {k: v for k, v in witness.items() if k != "fingerprint"},
        sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    return witness


def make_expected(witness):
    sources = witness["artifacts"][0]["models"][0]["files"]
    return {"endpoint": witness["listener"]["endpoint"],
            "served_alias": witness["listener"]["served_alias"],
            "artifacts": {name: {"sha256": fact["sha256"],
                                 "bytes": fact["bytes"]}
                          for name, fact in sources.items()},
            "tokenizer": {"files": {name: {"sha256": fact["sha256"],
                                           "bytes": fact["bytes"]}
                                    for name, fact in
                                    witness["tokenizer"]["files"].items()},
                          "backend": copy.deepcopy(
                              witness["tokenizer"]["backend"]),
                          "vocab": copy.deepcopy(
                              witness["tokenizer"]["vocab"]),
                          "special_ids": copy.deepcopy(
                              witness["tokenizer"]["special_ids"])},
            "attempt_id": witness["launch"]["attempt_id"],
            "ranks": list(witness["launch"]["ranks"])}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def served_config(witness_path, expected_path, witness):
    return {"backend": {"name": "served",
                        "serving_runtime": {
                            "witness": str(witness_path),
                            "expected": str(expected_path),
                            "endpoint":
                            witness["listener"]["endpoint"],
                            "served_alias":
                            witness["listener"]["served_alias"],
                            "attempt_id":
                            witness["launch"]["attempt_id"],
                            "ranks": list(
                                witness["launch"]["ranks"])}}}


def resign(witness):
    witness["fingerprint"] = hashlib.sha256(json.dumps(
        {k: v for k, v in witness.items() if k != "fingerprint"},
        sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    return witness


def test_complete_witness_passes_and_consumer_binds(tmp_path):
    witness = make_witness()
    witness_path = write_json(tmp_path / "witness.json", witness)
    expected_path = write_json(tmp_path / "expected.json",
                               make_expected(witness))
    verdict = verify(witness, make_expected(witness))
    assert verdict["verdict"] == "pass"
    assert verdict["proof_scope"] == "recorded_runtime_byte_binding"
    record = bind_served_task(
        served_config(witness_path, expected_path, witness))
    assert record["verdict"] == "pass"
    assert record["ranks"] == [0, 1]
    assert record["witness_fingerprint"] == witness["fingerprint"]


def test_verifier_cli_round_trip(tmp_path):
    witness = make_witness(world=1)
    witness_path = write_json(tmp_path / "witness.json", witness)
    expected_path = write_json(tmp_path / "expected.json",
                               make_expected(witness))
    out = tmp_path / "verdict.json"
    proc = subprocess.run(
        [sys.executable, "-m", "prismaquant.served_task_backend",
         "--witness", str(witness_path),
         "--endpoint", witness["listener"]["endpoint"],
         "--alias", witness["listener"]["served_alias"],
         "--attempt", witness["launch"]["attempt_id"],
         "--ranks", "0",
         "--expected", str(expected_path),
         "--out", str(out)],
        capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert json.loads(out.read_text())["verdict"] == "pass"


def test_verifier_cli_refuses_with_nonzero_status(tmp_path):
    witness = make_witness(world=1)
    witness["listener"]["served_alias"] = "other-alias"
    witness_path = write_json(tmp_path / "witness.json", witness)
    expected_path = write_json(tmp_path / "expected.json",
                               make_expected(make_witness(world=1)))
    proc = subprocess.run(
        [sys.executable, "-m", "prismaquant.served_task_backend",
         "--witness", str(witness_path),
         "--endpoint", "http://127.0.0.1:8142",
         "--alias", "fixture",
         "--attempt", make_witness(world=1)["launch"]["attempt_id"],
         "--ranks", "0",
         "--expected", str(expected_path)],
        capture_output=True, text=True, check=False)
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["verdict"] == "refuse"


def test_reader_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "witness.json"
    path.write_text('{"schema": "a", "schema": "b"}')
    with pytest.raises(ValueError, match="duplicate key"):
        read_witness(path)


def test_reader_rejects_non_strict_json(tmp_path):
    path = tmp_path / "witness.json"
    path.write_text('{"schema": NaN}')
    with pytest.raises(ValueError, match="strict JSON"):
        read_witness(path)


@pytest.mark.parametrize("field", ["schema", "listener", "launch", "lifetime",
                                   "artifacts", "tokenizer", "byte_coverage",
                                   "qualification_scope", "fingerprint"])
def test_missing_fact_refuses(field):
    witness = make_witness()
    del witness[field]
    verdict = verify(witness, make_expected(make_witness()))
    assert verdict["verdict"] == "refuse"


def test_wrong_schema_refuses():
    witness = make_witness()
    witness["schema"] = "prismaquant.serving_runtime_witness/1"
    verdict = verify(resign(witness), make_expected(make_witness()))
    assert verdict["verdict"] == "refuse"


def test_alias_only_evidence_refuses():
    witness = make_witness()
    witness["artifacts"] = []
    verdict = verify(witness, make_expected(make_witness()))
    assert verdict["verdict"] == "refuse"


def test_size_only_evidence_refuses():
    witness = make_witness()
    source = witness["artifacts"][0]["models"][0]["files"][
        "model.safetensors"]
    del source["sha256"]
    verdict = verify(resign(witness), make_expected(make_witness()))
    assert verdict["verdict"] == "refuse"


def test_incomplete_rank_coverage_refuses():
    witness = make_witness(world=2)
    witness["artifacts"] = witness["artifacts"][:1]
    verdict = verify(resign(witness), make_expected(make_witness()))
    assert verdict["verdict"] == "refuse"
    assert "incomplete" in verdict["reason"]


def test_rank_set_mismatch_refuses():
    witness = make_witness()
    expected = make_expected(witness)
    expected["ranks"] = [0]
    verdict = verify(witness, expected)
    assert verdict["verdict"] == "refuse"


@pytest.mark.parametrize("join", ["endpoint", "served_alias", "attempt_id"])
def test_join_mismatch_refuses(join):
    witness = make_witness()
    expected = make_expected(witness)
    if join == "endpoint":
        expected["endpoint"] = "http://127.0.0.1:8143"
    elif join == "served_alias":
        expected["served_alias"] = "other-alias"
    else:
        expected["attempt_id"] = "fixture/fixture-boot/10/11"
    verdict = verify(witness, expected)
    assert verdict["verdict"] == "refuse"


def test_artifact_digest_mismatch_refuses():
    witness = make_witness()
    expected = make_expected(witness)
    expected["artifacts"]["model.safetensors"]["sha256"] = "0" * 64
    verdict = verify(witness, expected)
    assert verdict["verdict"] == "refuse"


def test_tokenizer_digest_mismatch_refuses():
    witness = make_witness()
    expected = make_expected(witness)
    expected["tokenizer"]["files"]["tokenizer.json"]["sha256"] = "0" * 64
    verdict = verify(witness, expected)
    assert verdict["verdict"] == "refuse"


def test_rank_disagreement_refuses():
    witness = make_witness(world=2)
    other = _source()
    other["sha256"] = "1" * 64
    witness["artifacts"][1]["models"][0]["files"][
        "model.safetensors"] = other
    verdict = verify(resign(witness), make_expected(make_witness()))
    assert verdict["verdict"] == "refuse"


def test_coverage_tamper_refuses():
    witness = make_witness()
    witness["byte_coverage"]["tensor_payload_bytes"] = 8
    verdict = verify(resign(witness), make_expected(make_witness()))
    assert verdict["verdict"] == "refuse"


def test_fingerprint_tamper_refuses():
    witness = make_witness()
    witness["fingerprint"] = "0" * 64
    verdict = verify(witness, make_expected(make_witness()))
    assert verdict["verdict"] == "refuse"


def test_tokenizer_backend_mismatch_refuses():
    witness = make_witness()
    expected = make_expected(witness)
    expected["tokenizer"]["vocab"] = {"<s>": 0, "a": 2}
    verdict = verify(witness, expected)
    assert verdict["verdict"] == "refuse"


def test_tokenizer_special_id_mismatch_refuses():
    witness = make_witness()
    expected = make_expected(witness)
    expected["tokenizer"]["special_ids"]["bos"] = 1
    verdict = verify(witness, expected)
    assert verdict["verdict"] == "refuse"


def test_overstated_scope_refuses():
    witness = make_witness()
    witness["qualification_scope"] = "served_task_quality"
    verdict = verify(resign(witness), make_expected(make_witness()))
    assert verdict["verdict"] == "refuse"


def test_consumer_starts_no_rank_and_imports_no_runtime(tmp_path):
    import ast

    witness = make_witness()
    witness_path = write_json(tmp_path / "witness.json", witness)
    expected_path = write_json(tmp_path / "expected.json",
                               make_expected(witness))
    source = (Path(__file__).resolve().parents[1] / "prismaquant"
              / "served_task_backend.py").read_text()
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(part.name.split(".")[0] for part in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert "vllm" not in imports
    assert "tessera" not in imports
    assert "torch" not in imports
    assert "subprocess" not in imports
    assert "multiprocessing" not in imports
    assert "seal_check" not in source
    assert "Popen" not in source and "serve_once" not in source
    assert not is_served_config({"backend": {"name": "hf"}})
    record = bind_served_task(
        served_config(witness_path, expected_path, witness))
    assert record["verdict"] == "pass"


def test_consumer_refuses_without_verifier_pass(tmp_path):
    witness = make_witness()
    witness_path = write_json(tmp_path / "witness.json", witness)
    expected = make_expected(witness)
    expected["served_alias"] = "other-alias"
    expected_path = write_json(tmp_path / "expected.json", expected)
    with pytest.raises(ValueError, match="differ"):
        bind_served_task(served_config(witness_path, expected_path, witness))


def test_consumer_refuses_binding_mismatch_in_default_mode(tmp_path):
    witness = make_witness()
    witness_path = write_json(tmp_path / "witness.json", witness)
    expected_path = write_json(tmp_path / "expected.json",
                               make_expected(witness))
    config = served_config(witness_path, expected_path, witness)
    config["backend"]["serving_runtime"]["served_alias"] = "other-alias"
    with pytest.raises(ValueError, match="expectation file"):
        bind_served_task(config)


def test_consumer_refuses_incomplete_binding():
    with pytest.raises(ValueError, match="serving_runtime"):
        bind_served_task({"backend": {"name": "served"}})
