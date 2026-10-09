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

@pytest.fixture(autouse=True)
def local_model(tmp_path, monkeypatch):
    """Use isolated metadata; CPU argument tests need no fleet model."""
    model = tmp_path / "argument-model"
    model.mkdir()
    config = {
        "quantization_config": {
            "quant_method": "tessera",
            "config_groups": {f"group_{i}": {} for i in range(57)},
        },
    }
    (model / "config.json").write_text(json.dumps(config))
    (model / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    original = _run_entry

    def run(*argv, env=None):
        return original("--model", str(model), *argv, env=env)

    monkeypatch.setattr(sys.modules[__name__], "_run_entry", run)
    return model



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


def test_dry_run_binds_the_artifact_and_the_qualified_code(tmp_path, local_model):
    out = tmp_path / "dry.json"
    proc = _run_entry("--mode", "dry-run", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads(out.read_text())
    assert manifest["artifact"]["groups"] == 57
    import hashlib
    assert manifest["artifact"]["config_sha256"] == hashlib.sha256(
        (local_model / "config.json").read_bytes()).hexdigest()
    assert manifest["artifact"]["index_sha256"] == hashlib.sha256(
        (local_model / "model.safetensors.index.json").read_bytes()).hexdigest()
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


def test_member_launcher_syntax_is_valid():
    proc = subprocess.run(["bash", "-n", str(MEMBER)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.fixture
def census_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pq2459_entry", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("returncode", [0, 1])
def test_cleanup_retains_container_output_and_trace(
        tmp_path, monkeypatch, census_module, returncode):
    import shutil

    entry = census_module
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "quantization_config": {
            "quant_method": "tessera", "config_groups": {"group": {}}}}))
    (model / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    out = tmp_path / "host" / "receipt.json"
    out.parent.mkdir()
    container = tmp_path / "container"
    container.mkdir()
    raw = {"ranks": [{"rank": 0}, {"rank": 1}],
           "runtime": {"image": entry.RUNTIME_IMAGE},
           "verdict": "pass" if returncode == 0 else "fail"}
    observed_digest = "2" * 64
    trace = {"schema": "tessera.route_trace/1", "identity_version": 1,
             "rank": 0, "world_size": 2,
             "serving_source_sha256": observed_digest, "entries": []}
    (container / out.name).write_text(json.dumps(raw))
    (container / "trace-head.json").write_text(json.dumps(trace))
    (container / "runtime_contract.json").write_text('{"version": 57}')
    args = entry.build_parser().parse_args([
        "--mode", "head", "--out", str(out), "--model", str(model),
        "--container", "owned", "--head-addr", "127.0.0.1",
        "--runs-dir", str(out.parent)])
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    monkeypatch.setattr(entry, "_ray_alive", lambda *a: True)
    monkeypatch.setattr(entry, "_cluster_size", lambda *a: 2)

    def docker(cmd, **kwargs):
        if cmd[1] == "cp":
            source = container / Path(cmd[2].split(":", 1)[1]).name
            if not source.exists():
                return subprocess.CompletedProcess(cmd, 1)
            shutil.copyfile(source, cmd[3])
            return subprocess.CompletedProcess(cmd, 0)
        if cmd[1] == "rm":
            shutil.rmtree(container)
            return subprocess.CompletedProcess(cmd, 0)
        return subprocess.CompletedProcess(cmd, returncode)

    monkeypatch.setattr(entry, "_run", docker)
    if returncode:
        with pytest.raises(SystemExit):
            entry.run_head(args, out)
    else:
        assert entry.run_head(args, out) == 0
    assert not container.exists()
    assert json.loads(Path(str(out) + ".raw.json").read_text()) == raw
    assert json.loads(Path(str(out) + ".trace.json").read_text()) == trace
    if returncode:
        assert json.loads(Path(str(out) + ".refused.json").read_text()) == raw
    else:
        envelope = json.loads(out.read_text())
        assert envelope["observed_identity"]["serving_source_sha256"] == observed_digest
        assert envelope["qualified_cells"] == 0


def test_observed_identity_never_substitutes_expected_constants(tmp_path, census_module):
    import hashlib

    entry = census_module
    out = tmp_path / "receipt.json"
    raw = {"ranks": [{"rank": 0}, {"rank": 1}],
           "runtime": {"image": "example/observed@sha256:" + "3" * 64}}
    out.write_text(json.dumps(raw))
    trace_path = tmp_path / "trace.json"
    trace = {"schema": "tessera.route_trace/1", "identity_version": 1,
             "rank": 0, "world_size": 2, "serving_source_sha256": "4" * 64,
             "entries": []}
    trace_path.write_text(json.dumps(trace))
    contract_bytes = b'{"version": 57}\n'
    Path(str(out) + ".contract.json").write_bytes(contract_bytes)
    args = entry.build_parser().parse_args(["--mode", "head", "--out", str(out)])
    assert entry._wrap_receipt(args, out, str(trace_path)) == 0
    envelope = json.loads(out.read_text())
    observed = envelope["observed_identity"]
    assert observed["runtime_image"] == raw["runtime"]["image"]
    assert observed["serving_source_sha256"] == trace["serving_source_sha256"]
    assert observed["contract_sha256"] == hashlib.sha256(contract_bytes).hexdigest()
    assert observed["tessera_commit"] is None
    assert envelope["qualified_cells"] == 0


@pytest.mark.parametrize("dev_mode", ["0", "1"])
def test_launcher_image_mismatch_uses_the_actual_d32_path(tmp_path, dev_mode):
    import os

    staged = tmp_path / "tessera"
    (staged / "src/tessera/serving").mkdir(parents=True)
    (staged / "experiments").mkdir()
    (staged / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    # The former launcher called this hard refusal before its D32 entry point.
    (staged / "experiments/runtime_image.sh").write_text(
        "runtime_image_require() { return 2; }\n")
    (staged / "experiments/serve_lock.sh").write_text(
        "serve_lock_acquire() { :; }\nserve_lock_release() { :; }\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ["DOCKER_CALLS"]).open("a") as f:
    f.write(json.dumps(args) + "\\n")
if args[:2] == ["image", "inspect"]:
    print(json.dumps({"Id": "local-only", "RepoDigests": [
        "example/observed@sha256:" + "3" * 64]}))
elif args[0] == "cp":
    Path(args[2]).write_text('{"fixture": true}')
''')
    docker.chmod(0o755)
    calls_path = tmp_path / "docker-calls.jsonl"
    env = dict(
        os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        PQ2459_TS=str(staged), PQ2459_RUNS=str(tmp_path / "runs"),
        PQ2459_EXT=str(tmp_path / "ext"), PQ2459_PYTHON=sys.executable,
        PRISMAQUANT_DEV_MODE=dev_mode, DOCKER_CALLS=str(calls_path))
    out = tmp_path / "worker.json"
    proc = subprocess.run(["bash", str(MEMBER), "worker", str(out)],
                          capture_output=True, text=True, env=env, timeout=30)
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    launched = any(call[0] == "run" for call in calls)
    if dev_mode == "0":
        assert proc.returncode != 0
        assert not launched
    else:
        assert proc.returncode == 0, proc.stderr
        assert "[DEV-MODE]" in proc.stdout
        assert launched
        observed = json.loads(Path(str(out) + ".image.json").read_text())
        assert observed["resolved_reference"] == "example/observed@sha256:" + "3" * 64
        assert observed["identity_sealed"] is False
