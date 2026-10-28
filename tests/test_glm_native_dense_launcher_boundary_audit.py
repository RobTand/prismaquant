"""Exercise host launch admission without Docker or a Tessera installation."""
from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType

import pytest


IMAGE = "eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c"
ENCODER = Path("/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/native-pact-acquisition-20260922/tessera-ccc75b877691")
FORMAT = "TESSERA_E4M3_K1_R1006"
RUN = "re10-boundary-audit"


def _launcher(monkeypatch, *, forbid_serving=False, matching_digest=True):
    inspection = {
        "Id": "sha256:" + "a" * 64,
        "RepoDigests": [IMAGE] if matching_digest else ["other/image@sha256:" + "b" * 64],
        "Os": "linux",
        "Architecture": "arm64",
        "RootFS": {"Type": "layers", "Layers": ["sha256:" + "c" * 64]},
        "Config": {"Env": ["PATH=/usr/bin"], "Cmd": ["bash"]},
    }
    launches = []
    inspections = []

    def run(argv, *args, **kwargs):
        assert list(argv[:3]) == ["docker", "image", "inspect"], argv
        assert argv[3] == IMAGE
        inspections.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps([inspection]), stderr="")

    def check_output(argv, *args, **kwargs):
        result = run(argv, *args, **kwargs).stdout
        return result if kwargs.get("text") or kwargs.get("encoding") else result.encode()

    def call(argv, *args, **kwargs):
        assert list(argv[:2]) == ["docker", "run"], argv
        launches.append(list(argv))
        return 0

    def unexpected_process(*args, **kwargs):
        raise AssertionError(f"unexpected subprocess: {args!r}")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "check_output", check_output)
    monkeypatch.setattr(subprocess, "call", call)
    monkeypatch.setattr(subprocess, "Popen", unexpected_process)
    monkeypatch.setattr(sys, "argv", ["glm_native_dense_launch.py", "--format", FORMAT, "--run", RUN])
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {2, 7})

    legacy = ModuleType("tessera.serving.runtime_image")

    def resolve(requested):
        # Only the pre-fix command control supplies this external dependency.
        run(["docker", "image", "inspect", requested])
        return {
            "schema": "tessera.runtime_image/1",
            "requested": requested,
            "required": requested,
            "resolved_reference": requested,
            "repo_digests": inspection["RepoDigests"],
            "local_id": inspection["Id"],
            "present": True,
            "gated": True,
            "refused": requested not in inspection["RepoDigests"],
            "reason": "explicit_digest" if matching_digest else "image_digest_mismatch",
        }

    legacy.resolve = resolve
    legacy.container_env = lambda record: {
        "TESSERA_CENSUS_RUNTIME_IMAGE": record["resolved_reference"],
        "TESSERA_CENSUS_RUNTIME_IMAGE_DECLARATION": json.dumps(record, sort_keys=True),
    }
    for name in ("tessera", "tessera.serving"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, "tessera.serving.runtime_image", legacy)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if forbid_serving and (name == "tessera.serving" or name.startswith("tessera.serving.")):
            raise AssertionError("PrismaQuant host launcher imported Tessera serving runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    path = Path(__file__).resolve().parents[1] / "experiments/glm_native_dense_launch.py"
    return path, inspection, launches, inspections


def _run(path):
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(path), run_name="__main__")
    assert exited.value.code == 0


def test_native_dense_launcher_does_not_import_tessera_serving(monkeypatch):
    path, _, launches, _ = _launcher(monkeypatch, forbid_serving=True)
    _run(path)
    assert len(launches) == 1


def test_native_dense_launcher_preserves_image_and_downstream_declaration(monkeypatch):
    path, inspection, launches, inspections = _launcher(monkeypatch)
    _run(path)
    assert inspections
    assert len(launches) == 1
    argv = launches[0]
    assert argv.count(IMAGE) == 1
    image_index = argv.index(IMAGE)
    assert argv[image_index + 1:image_index + 3] == ["python3", "-c"]
    assert argv[argv.index("--memory") + 1] == "16g"
    assert argv[argv.index("--memory-swap") + 1] == "16g"
    assert argv[argv.index("--cpuset-cpus") + 1] == "2,7"
    mounts = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--mount"]
    assert f"type=bind,src={ENCODER},dst=/tessera,readonly" in mounts
    assert "type=bind,src=/mnt/shared,dst=/mnt/shared" in mounts
    assert any(mount.endswith(",dst=/pq,readonly") for mount in mounts)
    env = dict(argv[i + 1].split("=", 1) for i, arg in enumerate(argv) if arg == "--env")
    assert env["TESSERA_CENSUS_RUNTIME_IMAGE"] == IMAGE
    declaration = json.loads(env["TESSERA_CENSUS_RUNTIME_IMAGE_DECLARATION"])
    assert declaration["schema"] == "tessera.runtime_image/1"
    assert not declaration["refused"]
    assert declaration["present"] is declaration["gated"] is True
    assert declaration["requested"] == declaration["required"] == declaration["resolved_reference"] == IMAGE
    assert declaration["reason"] == "explicit_digest"
    assert IMAGE in declaration["repo_digests"]
    assert declaration["repo_digests"] == inspection["RepoDigests"]
    identity_owner = runpy.run_path(str(path.parents[1] / "prismaquant/container_runtime_identity.py"))
    assert env["PRISMAQUANT_CONTAINER_CONTENT_SHA256"] == identity_owner["image_content_sha256"](inspection)
    assert env["MAX_JOBS"] == "2"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS"):
        assert env[key] == "1"
    code = argv[image_index + 3]
    assert "experiments/glm_native_dense_acquire.py" in code
    assert f"FORMAT={FORMAT!r}" in code
    assert f"OUT={str(ENCODER.parent / RUN)!r}" in code


def test_native_dense_launcher_refuses_missing_requested_repo_digest(monkeypatch):
    path, _, launches, _ = _launcher(monkeypatch, matching_digest=False)
    with pytest.raises((RuntimeError, ValueError), match="(?i)(image|digest)"):
        runpy.run_path(str(path), run_name="__main__")
    assert launches == []
