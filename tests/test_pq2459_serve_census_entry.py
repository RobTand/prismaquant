"""PQ #2459 entry-point tests: one path, real fixtures, D32 seals.

These tests run on CPU through PrismaBuild. They check the actual
qualification entry point, not a metadata script. No test touches CUDA.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ENTRY = REPO / "tools/pq2459_serve_census.py"
MEMBER = REPO / "tools/pq2459_gang_member.sh"


def _run_entry(*argv: str, env: dict | None = None):
    import os
    base = dict(os.environ) if env is None else dict(env)
    cmd = [sys.executable, str(ENTRY), *argv]
    return subprocess.run(cmd, capture_output=True, text=True, env=base)


def _dev_env() -> dict:
    import os
    env = dict(os.environ)
    env["PRISMAQUANT_DEV_MODE"] = "1"
    return env


def test_dry_run_emits_the_exact_head_argv(tmp_path):
    out = tmp_path / "dry.json"
    proc = _run_entry("--mode", "dry-run", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads(out.read_text())
    assert manifest["schema"] == "prismaquant.pq2459_serve_census_dry_run.v2"
    argv = manifest["head_argv"]
    assert argv[1] == "tools/tessera_route_census.py"
    assert "--prompt-tokens" in argv
    assert argv[argv.index("--prompt-tokens") + 1] == "2048"
    assert argv[argv.index("--max-model-len") + 1] == "2049"
    assert argv[argv.index("--max-num-seqs") + 1] == "1"
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "2049"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert argv[argv.index("--distributed-executor-backend") + 1] == "ray"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.5"
    assert argv[argv.index("--kv-cache-memory-bytes") + 1] == "1073741824"
    assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8_ds_mla"
    assert argv[argv.index("--moe-backend") + 1] == "triton"
    assert "--trust-remote-code" in argv
    assert "--language-model-only" in argv
    assert manifest["engine_scope"]["prompt_tokens"] == 2048
    assert manifest["engine_scope"]["gpu_memory_utilization"] == 0.5
    assert manifest["head_trace"].endswith("trace-head.json")


def test_dry_run_covers_every_fixture_profile(tmp_path):
    out = tmp_path / "dry.json"
    proc = _run_entry("--mode", "dry-run", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads(out.read_text())
    assert sorted(manifest["profiles"]) == [
        "speed_batch", "speed_decode", "tr3_batch"]
    assert manifest["profiles"]["tr3_batch"]["token_rows"] == [2048, 2049]
    assert manifest["profiles"]["speed_batch"]["token_rows"] == [
        512, 1024, 1026, 1537, 2048]
    assert manifest["profiles"]["speed_decode"]["token_rows"] == [1, 2, 4]


def test_dry_run_binds_the_artifact_and_the_qualified_code(tmp_path):
    out = tmp_path / "dry.json"
    proc = _run_entry("--mode", "dry-run", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads(out.read_text())
    assert manifest["artifact"]["groups"] == 57
    assert manifest["artifact"]["config_sha256"].startswith("3f5c2c73")
    assert manifest["artifact"]["index_sha256"].startswith("2990e8c0")
    assert manifest["tessera_commit"] == (
        "9eef9fea6edce32f4e64abf87f0058b11dab2287")
    assert manifest["serving_source_sha256"] == (
        "a9b7bf32563ce874f45956dd4e5ff4b4c43de73459f9aa40dee1977b9b152330")
    assert manifest["runtime_image_sealed"] is True
    assert manifest["serving_commit_sealed"] is True


def test_other_commit_stamps_dev_mode_and_continues(tmp_path):
    out = tmp_path / "dry.json"
    proc = _run_entry("--mode", "dry-run", "--out", str(out),
                      "--tessera-commit", "0" * 40, env=_dev_env())
    assert proc.returncode == 0, proc.stderr
    assert "[DEV-MODE]" in proc.stdout
    manifest = json.loads(out.read_text())
    assert manifest["serving_commit_sealed"] is False


def test_other_image_stamps_dev_mode_and_continues(tmp_path):
    out = tmp_path / "dry.json"
    proc = _run_entry(
        "--mode", "dry-run", "--out", str(out),
        "--runtime-image", "localhost/example/img@sha256:" + "1" * 64,
        env=_dev_env())
    assert proc.returncode == 0, proc.stderr
    assert "[DEV-MODE]" in proc.stdout
    manifest = json.loads(out.read_text())
    assert manifest["runtime_image_sealed"] is False


def test_other_commit_refuses_in_certified_mode(tmp_path):
    import os

    out = tmp_path / "dry.json"
    env = dict(os.environ, PRISMAQUANT_DEV_MODE="0")
    proc = _run_entry("--mode", "dry-run", "--out", str(out),
                      "--tessera-commit", "0" * 40, env=env)
    assert proc.returncode != 0


def test_non_tp2_refuses_in_both_modes(tmp_path):
    import os

    out = tmp_path / "dry.json"
    proc = _run_entry("--mode", "dry-run", "--out", str(out),
                      "--tensor-parallel-size", "1")
    assert proc.returncode != 0
    env = dict(os.environ, PRISMAQUANT_DEV_MODE="0")
    proc = _run_entry("--mode", "dry-run", "--out", str(out),
                      "--tensor-parallel-size", "1", env=env)
    assert proc.returncode != 0


def test_streamed_mode_refuses_in_both_modes(tmp_path):
    import os

    out = tmp_path / "dry.json"
    env = dict(os.environ, TESSERA_SERVE_MODE="streamed")
    proc = _run_entry("--mode", "dry-run", "--out", str(out), env=env)
    assert proc.returncode != 0


def test_single_node_rendezvous_refuses(tmp_path):
    import os

    out = tmp_path / "dry.json"
    env = dict(os.environ, MASTER_ADDR="127.0.0.1")
    proc = _run_entry("--mode", "dry-run", "--out", str(out), env=env)
    assert proc.returncode != 0
    assert "MASTER_ADDR" in (proc.stderr + proc.stdout)


def test_member_launcher_has_no_ssh_and_names_both_roles():
    text = MEMBER.read_text()
    assert "ssh " not in text
    assert "ssh -" not in text
    assert '"head"' in text
    assert '"worker"' in text
    assert "pq2459_serve_census.py" in text
    assert "--mode head" in text
    assert "--mode worker" in text
    assert "NCCL_IB_DISABLE=1" in text
    assert "enp1s0f0np0" in text


def test_member_launcher_syntax_is_valid():
    proc = subprocess.run(["bash", "-n", str(MEMBER)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
