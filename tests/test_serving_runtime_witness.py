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
    from prismaquant.serving_runtime_witness import tokenizer_content_sha256

    file_a, file_b = _sha("weight-a"), _sha("weight-b")
    tok_a = _sha("tokenizer-bytes")
    source_files = {"tokenizer.json": {"sha256": tok_a, "bytes": 3}}
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
        "tokenizer": {"source_files": source_files,
                      "content_sha256": tokenizer_content_sha256(source_files),
                      "effective": {"add_special_tokens": False}},
    }


def make_expected(witness, *, ranks=(0, 1)):
    files = {row["path"]: {"sha256": row["sha256"], "bytes": row["bytes"]}
             for row in witness["ranks"][0]["files"]}
    return {
        "endpoint": witness["endpoint"]["base_url"],
        "served_alias": witness["served_alias"],
        "attempt_id": witness["launch_attempt"]["attempt_id"],
        "ranks": list(ranks),
        "artifact_files": files,
        "artifact": dict(witness["artifact"]),
        "tokenizer_files": {name: dict(row)
                            for name, row in witness["tokenizer"]["source_files"].items()},
        "tokenizer_content_sha256": witness["tokenizer"]["content_sha256"],
        "tokenizer_effective": dict(witness["tokenizer"]["effective"]),
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
    with pytest.raises(ValueError, match="rank set"):
        verify(witness, make_expected(witness, ranks=(0, 1)),
               environ={"PRISMAQUANT_DEV_MODE": "0"})


@pytest.mark.parametrize("join", ["endpoint", "served_alias", "attempt_id"])
def test_join_mismatch_refuses(join):
    witness = make_witness()
    expected = make_expected(witness)
    expected[join] = "other-value"
    with pytest.raises(ValueError, match="differs"):
        verify(witness, expected, environ={"PRISMAQUANT_DEV_MODE": "0"})


def test_join_mismatch_stamps_in_dev_mode():
    witness = make_witness()
    expected = make_expected(witness)
    expected["served_alias"] = "other-value"
    verdict = verify(witness, expected, environ={"PRISMAQUANT_DEV_MODE": "1"})
    assert verdict["verdict"] == "refuse"
    assert "dev stamp" in verdict["reason"]


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
    expected["artifact_files"] = dict(
        expected["artifact_files"],
        **{"model.safetensors": {"sha256": "0" * 64, "bytes": 8}})
    assert "digest differs" in verify(witness, expected)["reason"]


def test_omitted_expected_file_refuses_despite_complete_coverage():
    witness = make_witness()
    for row in witness["ranks"]:
        row["files"] = [entry for entry in row["files"]
                        if entry["path"] != "config.json"]
    expected = make_expected(make_witness())
    verdict = verify(witness, expected)
    assert verdict["verdict"] == "refuse"
    assert "config.json" in verdict["reason"]


def test_extra_witness_file_refuses():
    witness = make_witness()
    extra = {"path": "extra.safetensors", "sha256": _sha("extra"), "bytes": 5}
    for row in witness["ranks"]:
        row["files"] = [*row["files"], dict(extra)]
    verdict = verify(witness, make_expected(make_witness()))
    assert verdict["verdict"] == "refuse"
    assert "extra.safetensors" in verdict["reason"]


def test_rank_file_byte_inconsistency_refuses():
    witness = make_witness()
    witness["ranks"][0]["files"][0]["bytes"] = 999
    verdict = verify(witness, make_expected(make_witness()))
    assert verdict["verdict"] == "refuse"
    assert "byte" in verdict["reason"].lower()


def test_rank_rows_must_cover_one_roster():
    witness = make_witness()
    witness["ranks"][1]["files"] = [dict(witness["ranks"][1]["files"][0])]
    verdict = verify(witness, make_expected(make_witness()))
    assert verdict["verdict"] == "refuse"
    assert "rank" in verdict["reason"].lower()


def test_artifact_byte_total_must_match_file_rows():
    witness = make_witness()
    witness["artifact"]["artifact_bytes"] = 13
    verdict = verify(witness, make_expected(make_witness()))
    assert verdict["verdict"] == "refuse"
    assert "byte" in verdict["reason"].lower()


def test_tokenizer_content_must_recompute_from_source_files():
    witness = make_witness()
    witness["tokenizer"]["content_sha256"] = "0" * 64
    expected = make_expected(make_witness())
    expected["tokenizer_content_sha256"] = "0" * 64
    verdict = verify(witness, expected)
    assert verdict["verdict"] == "refuse"
    assert "tokenizer" in verdict["reason"].lower()


def test_extra_tokenizer_source_file_refuses():
    from prismaquant.serving_runtime_witness import tokenizer_content_sha256

    witness = make_witness()
    witness["tokenizer"]["source_files"]["added_tokens.json"] = {
        "sha256": _sha("added"), "bytes": 2}
    witness["tokenizer"]["content_sha256"] = tokenizer_content_sha256(
        witness["tokenizer"]["source_files"])
    verdict = verify(witness, make_expected(make_witness()))
    assert verdict["verdict"] == "refuse"
    assert "added_tokens.json" in verdict["reason"]


def test_tokenizer_effective_settings_mismatch_refuses():
    witness = make_witness()
    witness["tokenizer"]["effective"] = {"add_special_tokens": True}
    verdict = verify(witness, make_expected(make_witness()))
    assert verdict["verdict"] == "refuse"
    assert "effective" in verdict["reason"].lower()


def test_collector_observes_rank_files_from_each_rank_root(tmp_path):
    from prismaquant.serving_runtime_witness_collect import (
        rank_byte_evidence_from_roots)

    roots = []
    for rank in (0, 1):
        root = tmp_path / f"rank{rank}"
        root.mkdir()
        (root / "model.safetensors").write_bytes(f"weight-{rank}".encode())
        (root / "config.json").write_bytes(b"{}")
        roots.append(root)
    with pytest.raises(ValueError, match="loaded bytes differ"):
        rank_byte_evidence_from_roots(roots)
    (roots[1] / "model.safetensors").write_bytes(b"weight-0")
    rows = rank_byte_evidence_from_roots(roots)
    assert [row["rank"] for row in rows] == [0, 1]
    assert [entry["path"] for entry in rows[0]["files"]] == [
        "config.json", "model.safetensors"]
    with pytest.raises(ValueError, match="must differ per rank"):
        rank_byte_evidence_from_roots([roots[0], roots[0]])


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
    assert "seal_check" in source
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
        bind_served_task(served_config(witness_path, expected_path, witness),
                         environ={"PRISMAQUANT_DEV_MODE": "0"})


def test_consumer_refuses_incomplete_binding():
    with pytest.raises(ValueError, match="serving_runtime"):
        bind_served_task({"backend": {"name": "served"}})
