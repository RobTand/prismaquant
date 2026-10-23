"""The LFM launcher must declare its admitted CPU set in Docker argv."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import lfm_mixed_native_host as host


IMAGE = "registry.invalid/tessera@sha256:" + "e" * 64


class _CapturedLaunch(Exception):
    pass


@pytest.fixture
def launch_input(tmp_path, monkeypatch):
    with tempfile.TemporaryDirectory(prefix="pq-lfm-affinity-audit-", dir="/mnt/shared") as root:
        root = Path(root)
        source, model = root / "source", root / "model"
        source.mkdir()
        model.mkdir()
        (source / "entry.py").write_text("value = 1\n")
        manifest = tmp_path / "source.json"
        manifest.write_text(json.dumps({
            "commit": "a" * 40,
            "files": {"entry.py": hashlib.sha256((source / "entry.py").read_bytes()).hexdigest()},
        }))
        output = tmp_path / "out"
        argv = [
            "lfm_mixed_native_host.py", "--out", str(output),
            "--source", str(source), "--source-manifest", str(manifest),
            "--source-sha256", hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "--image", IMAGE, "--model", str(model),
        ]
        calls = []

        def check_output(command, **kwargs):
            if command == ["docker", "image", "inspect", IMAGE]:
                return json.dumps([{"Id": "sha256:" + "f" * 64, "RepoDigests": [IMAGE]}]).encode()
            if command[:2] == ["git", "-C"] and command[-2:] == ["rev-parse", "HEAD"]:
                return "b" * 40 + "\n"
            raise AssertionError(f"unexpected process request: {command}")

        def capture(command, **kwargs):
            calls.append(command)
            raise _CapturedLaunch

        class NoThread:
            def __init__(self, **kwargs):
                pass

            def start(self):
                pass

            def join(self, **kwargs):
                pass

        local_out = tmp_path / "local"
        local_out.mkdir()
        monkeypatch.setattr(host.subprocess, "check_output", check_output)
        monkeypatch.setattr(host.subprocess, "run", capture)
        monkeypatch.setattr(host.threading, "Thread", NoThread)
        monkeypatch.setattr(host.tempfile, "mkdtemp", lambda **kwargs: str(local_out))
        monkeypatch.setattr(host.os, "sched_getaffinity", lambda pid: {2, 7, 11, 19})
        monkeypatch.setattr("sys.argv", argv)
        yield SimpleNamespace(source=source, manifest=manifest, output=output, calls=calls)


def test_lfm_launcher_declares_admitted_affinity(launch_input):
    with pytest.raises(_CapturedLaunch):
        host.main()
    command, = launch_input.calls
    flag = next((part for part in command if part.startswith("--cpuset-cpus=")), None)
    if flag is not None:
        cpuset = flag.split("=", 1)[1]
    else:
        assert "--cpuset-cpus" in command
        cpuset = command[command.index("--cpuset-cpus") + 1]
    assert cpuset == "2,7,11,19"
    assert "--cpus=4" in command
    assert "OMP_NUM_THREADS=1" in command
    assert "MKL_NUM_THREADS=1" in command
    assert "OPENBLAS_NUM_THREADS=1" in command
    assert IMAGE in command
    record = json.loads((launch_input.output / "host.json").read_text())
    assert record["cpu_affinity"] == [2, 7, 11, 19]
    assert record["argv"] == command


def test_lfm_launcher_refuses_source_manifest_drift(launch_input):
    (launch_input.source / "entry.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="Tessera source changed"):
        host.main()
    assert launch_input.calls == []


def test_lfm_launcher_refuses_unresolved_image_digest(launch_input, monkeypatch):
    monkeypatch.setattr(host.subprocess, "check_output", lambda *args, **kwargs: json.dumps([
        {"Id": "sha256:" + "f" * 64, "RepoDigests": ["registry.invalid/other@sha256:" + "d" * 64]},
    ]).encode())
    with pytest.raises(ValueError, match="declared serving digest"):
        host.main()
    assert launch_input.calls == []
