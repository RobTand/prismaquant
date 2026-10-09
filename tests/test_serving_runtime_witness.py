"""The producer witness joins endpoint, alias, attempt, ranks, bytes, tokenizer.

The standalone verifier accepts a complete witness and refuses gaps.
 The consumer binds a served task through that witness and verifier only.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant.serving_runtime_verifier import read_witness, verify
from prismaquant.serving_runtime_witness import WITNESS_SCHEMA, witness_problems
from prismaquant.served_task_backend import bind_served_task

pytestmark = pytest.mark.task_suite


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_witness(*, ranks=(0, 1), coverage="complete", alias="eager-model",
                 endpoint="http://127.0.0.1:8000", attempt="launch-7"):
    file_a, file_b = _sha("weight-a"), _sha("weight-b")
    tok_a = _sha("tokenizer-bytes")
    return {
        "schema": WITNESS_SCHEMA,
        "endpoint": {"base_url": endpoint},
        "served_alias": alias,
        "launch_attempt": {"attempt_id": attempt, "image": "vllm-node:test", "argv": ["serve"]},
        "ranks": [{"rank": rank,
                   "files": [{"path": "model.safetensors", "sha256": file_a, "bytes": 8},
                             {"path": "config.json", "sha256": file_b, "bytes": 4}],
                   "coverage": coverage, "representation_changes": []} for rank in ranks],
        "artifact": {"model_sha256": _sha("model"), "inventory_sha256": _sha("inventory"),
                     "artifact_bytes": 12},
        "tokenizer": {"source_files": {"tokenizer.json": {"sha256": tok_a, "bytes": 3}},
                      "content_sha256": _sha("tokenizer-content"),
                      "effective": {"add_special_tokens": False}},
    }


def make_expected(witness, *, ranks=(0, 1)):
    files = {row["path"]: row["sha256"] for row in witness["ranks"][0]["files"]}
    return {
        "endpoint": witness["endpoint"]["base_url"],
        "served_alias": witness["served_alias"],
        "attempt_id": witness["launch_attempt"]["attempt_id"],
        "ranks": list(ranks),
        "artifact_files": files,
        "artifact": dict(witness["artifact"]),
        "tokenizer_files": {name: row["sha256"]
                            for name, row in witness["tokenizer"]["source_files"].items()},
        "tokenizer_content_sha256": witness["tokenizer"]["content_sha256"],
    }


def write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def served_config(witness_path: Path, expected_path: Path, witness, *, ranks=(0, 1)):
    return {"schema": "prismaquant.task_suite/1",
            "backend": {"name": "served", "pretrained": "/artifact", "tokenizer": "/artifact",
                        "device": "cpu", "dtype": "float32", "batch_size": 1, "max_length": 128,
                        "trust_remote_code": False,
                        "serving_runtime": {
                            "witness": str(witness_path), "expected": str(expected_path),
                            "endpoint": witness["endpoint"]["base_url"],
                            "served_alias": witness["served_alias"],
                            "attempt_id": witness["launch_attempt"]["attempt_id"],
                            "ranks": list(ranks)}},
            "tasks": ["facts"],
            "sampling": {"limit": 2, "num_fewshot": 0, "random_seed": 0, "numpy_seed": 0,
                         "torch_seed": 0, "fewshot_seed": 0}, "criteria": []}


def test_complete_witness_passes_and_consumer_binds(tmp_path):
    witness = make_witness()
    assert witness_problems(witness) == []
    assert verify(witness, make_expected(witness))["verdict"] == "pass"
    witness_path = write_json(tmp_path / "witness.json", witness)
    expected_path = write_json(tmp_path / "expected.json", make_expected(witness))
    record = bind_served_task(served_config(witness_path, expected_path, witness))
    assert record["verdict"] == "pass"
    assert record["ranks"] == [0, 1]


def test_verifier_cli_round_trip(tmp_path):
    import subprocess
    import sys

    witness = make_witness()
    witness_path = write_json(tmp_path / "witness.json", witness)
    expected_path = write_json(tmp_path / "expected.json", make_expected(witness))
    out = tmp_path / "verdict.json"
    process = subprocess.run(
        [sys.executable, "-m", "prismaquant.serving_runtime_verifier",
         "--witness", str(witness_path), "--endpoint", witness["endpoint"]["base_url"],
         "--alias", witness["served_alias"],
         "--attempt", witness["launch_attempt"]["attempt_id"],
         "--ranks", "0,1", "--expected", str(expected_path),
         "--out", str(out)],
        capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stderr
    assert json.loads(out.read_text())["verdict"] == "pass"

def test_collector_digest_survives_write_and_read(tmp_path):
    from prismaquant.serving_runtime_witness import witness_sha256
    from prismaquant.serving_runtime_witness_collect import collect_witness, write_witness

    witness = make_witness()
    joined = collect_witness(
        endpoint=witness["endpoint"]["base_url"], served_alias=witness["served_alias"],
        attempt_id=witness["launch_attempt"]["attempt_id"], image="vllm-node:test",
        launch_argv=["serve"], ranks=witness["ranks"], artifact=witness["artifact"],
        tokenizer=witness["tokenizer"])
    assert joined["witness_sha256"] == witness_sha256(joined)
    path = write_witness(tmp_path / "witness.json", joined)
    reread = read_witness(path)
    assert witness_sha256(reread) == joined["witness_sha256"]
    assert verify(reread, make_expected(witness))["verdict"] == "pass"
    assert verify(reread, make_expected(witness))["witness_sha256"] == joined["witness_sha256"]



def test_verifier_reads_strict_witness_bytes(tmp_path):
    witness = make_witness()
    raw = json.dumps(witness, sort_keys=True) + "\n"
    path = tmp_path / "witness.json"
    path.write_text(raw)
    assert read_witness(path)["served_alias"] == witness["served_alias"]
    with pytest.raises(ValueError, match="duplicate"):
        path.write_text('{"a": 1, "a": 2}')
        read_witness(path)


@pytest.mark.parametrize("field", ["endpoint", "served_alias", "launch_attempt",
                                   "ranks", "artifact", "tokenizer"])
def test_missing_fact_refuses(field):
    witness = make_witness()
    del witness[field]
    assert witness_problems(witness)
    assert verify(witness, make_expected(make_witness()))["verdict"] == "refuse"


def test_alias_only_evidence_refuses():
    witness = make_witness()
    witness["ranks"] = []
    assert witness_problems(witness)
    assert verify(witness, make_expected(make_witness()))["verdict"] == "refuse"


def test_size_only_evidence_refuses():
    witness = make_witness()
    for row in witness["ranks"]:
        for entry in row["files"]:
            del entry["sha256"]
    assert witness_problems(witness)
    assert verify(witness, make_expected(make_witness()))["verdict"] == "refuse"


def test_incomplete_rank_coverage_refuses():
    witness = make_witness(coverage="partial")
    assert witness_problems(witness) == []
    assert "coverage" in verify(witness, make_expected(witness))["reason"]


def test_rank_set_mismatch_refuses():
    witness = make_witness(ranks=(0,))
    assert verify(witness, make_expected(witness, ranks=(0, 1)))["verdict"] == "refuse"


@pytest.mark.parametrize("join", ["endpoint", "served_alias", "attempt_id"])
def test_join_mismatch_refuses(join):
    witness = make_witness()
    expected = make_expected(witness)
    expected[join] = "other-value"
    assert verify(witness, expected)["verdict"] == "refuse"


def test_artifact_digest_mismatch_refuses():
    witness = make_witness()
    expected = make_expected(witness)
    expected["artifact"] = dict(expected["artifact"], model_sha256="0" * 64)
    assert "artifact" in verify(witness, expected)["reason"]


def test_tokenizer_digest_mismatch_refuses():
    witness = make_witness()
    expected = make_expected(witness)
    expected["tokenizer_content_sha256"] = "0" * 64
    assert "tokenizer" in verify(witness, expected)["reason"]


def test_rank_file_digest_mismatch_refuses():
    witness = make_witness()
    expected = make_expected(witness)
    expected["artifact_files"] = dict(expected["artifact_files"],
                                      **{"model.safetensors": "0" * 64})
    assert "digest differs" in verify(witness, expected)["reason"]


def test_consumer_starts_no_rank_and_imports_no_runtime(tmp_path):
    import ast

    witness = make_witness()
    witness_path = write_json(tmp_path / "witness.json", witness)
    expected_path = write_json(tmp_path / "expected.json", make_expected(witness))
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
    record = bind_served_task(served_config(witness_path, expected_path, witness))
    assert record["verdict"] == "pass"


def test_consumer_refuses_without_verifier_pass(tmp_path):
    witness = make_witness()
    witness_path = write_json(tmp_path / "witness.json", witness)
    expected = make_expected(witness)
    expected["served_alias"] = "other-alias"
    expected_path = write_json(tmp_path / "expected.json", expected)
    with pytest.raises(ValueError, match="served binding served_alias differs"):
        bind_served_task(served_config(witness_path, expected_path, witness))


def test_consumer_refuses_incomplete_binding():
    with pytest.raises(ValueError, match="serving_runtime"):
        bind_served_task({"backend": {"name": "served"}})
