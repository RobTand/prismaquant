"""Real host CLI/bootstrap/Git; fake Docker only, with no model/GPU/container run."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "audit.invalid/vllm@sha256:" + "a" * 64

DOCKER = r'''
import json, os, sys
from pathlib import Path

path = Path(os.environ["AUDIT_DOCKER_STATE"])
state = json.loads(path.read_text())
argv = sys.argv[1:]
state["calls"].append(argv)
code = 0
if argv[:2] == ["image", "inspect"]:
    # A real host preflight refusal before any create/run or model capture.
    print(json.dumps([{"RepoDigests": []}]))
elif argv[:1] == ["stop"]:
    name = argv[-1]
    if name in state["containers"]:
        state["containers"][name]["running"] = False
        print(name)
    else:
        print("Error: No such container: " + name, file=sys.stderr)
        code = 1
elif argv[:2] == ["rm", "-f"]:
    name = argv[-1]
    if name in state["containers"]:
        del state["containers"][name]
        print(name)
    else:
        print("Error: No such container: " + name, file=sys.stderr)
        code = 1
elif argv[:1] == ["inspect"]:
    name = argv[-1]
    if name in state["containers"]:
        print(json.dumps([{"State": {"Running": state["containers"][name]["running"]}}]))
    else:
        print("Error: No such object: " + name, file=sys.stderr)
        code = 1
else:
    print("Unexpected Docker request: " + repr(argv), file=sys.stderr)
    code = 97
path.write_text(json.dumps(state))
raise SystemExit(code)
'''


def _real_host(tmp_path, containers):
    state = tmp_path / "docker-state.json"
    state.write_text(json.dumps({"calls": [], "containers": containers}))
    commands = tmp_path / "bin"
    commands.mkdir()
    docker = commands / "docker"
    docker.write_text("#!" + sys.executable + "\n" + textwrap.dedent(DOCKER))
    docker.chmod(0o755)
    output = tmp_path / "same-basename"
    env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"],
               AUDIT_DOCKER_STATE=str(state), CUDA_VISIBLE_DEVICES="",
               PQ_RUNTIME_PRISMAQUANT_ROOT=str(ROOT), PYTHONNOUSERSITE="1",
               PYTHONSAFEPATH="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-P", str(ROOT / "experiments/pq87_physical_ab.py"),
            "--image", IMAGE, "--out", str(output),
            "--model-dir", str(tmp_path / "never-read-model"), "--seconds", "20"]
    with subprocess.Popen(argv, cwd=ROOT, env=env, start_new_session=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=45)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate(timeout=10)
            pytest.fail(f"host fixture timed out: {stdout}\n{stderr}")
    assert process.returncode == 0, (stdout, stderr)
    report = json.loads((output / "campaign.json").read_bytes())
    observed = json.loads(state.read_bytes())
    assert report["status"] == "inconclusive"
    assert "immutable RepoDigest" in report["error"]
    assert re.fullmatch("[0-9a-f]{40}", report["source_snapshot"])
    assert not any(call[0] in {"run", "create"} for call in observed["calls"])
    return report, observed


def test_failed_preflight_cannot_stop_or_remove_a_borrowed_name(tmp_path):
    name = "pq87-same-basename"
    foreign = {name: {"owner": "another-action", "running": True}}
    _report, observed = _real_host(tmp_path, foreign)
    assert observed["containers"] == foreign, observed
    assert not any(call[0] in {"stop", "rm"} for call in observed["calls"]), observed


def test_failed_preflight_preserves_an_unrelated_named_container(tmp_path):
    foreign = {"pq87-another-basename": {"owner": "another-action", "running": True}}
    _report, observed = _real_host(tmp_path, foreign)
    assert observed["containers"] == foreign


def test_failed_preflight_without_a_container_remains_an_observation(tmp_path):
    report, observed = _real_host(tmp_path, {})
    assert observed["containers"] == {}
    assert report["status"] == "inconclusive"


LIFECYCLE_DOCKER = r'''
import json, os, sys
from pathlib import Path

path = Path(os.environ["AUDIT_DOCKER_STATE"])
state = json.loads(path.read_text())
argv = sys.argv[1:]
state["calls"].append(argv)
mode = state["mode"]
cid, foreign = "c" * 64, "f" * 64
code = 0

def find(target):
    return next((key for key, row in state["containers"].items()
                 if key == target or row["Name"] == "/" + target), None)

def borrowed(name):
    return {"Id": foreign, "Name": "/" + name, "State": {"Running": True},
            "Config": {"Labels": {"prismabuild.action": "another-action"}}}

if argv[:2] == ["image", "inspect"]:
    print(json.dumps([{"RepoDigests": [argv[-1]]}]))
elif argv[:1] == ["inspect"]:
    target = argv[-1]
    if mode == "occupied_name" and not state["containers"]:
        state["containers"][foreign] = borrowed(target)
    key = find(target)
    if key is None:
        print("Error: No such object: " + target, file=sys.stderr)
        code = 1
    else:
        print(json.dumps([state["containers"][key]]))
elif argv[:1] == ["run"]:
    name = argv[argv.index("--name") + 1]
    if mode == "reused_name":
        state["containers"][foreign] = borrowed(name)
        print("Error: name already in use", file=sys.stderr)
        code = 125
    elif mode == "failed_create":
        print("Error: create rejected before CID", file=sys.stderr)
        code = 125
    else:
        cidfile = Path(argv[argv.index("--cidfile") + 1])
        label = argv[argv.index("--label") + 1].split("=", 1)[1]
        labels = {"prismabuild.action": os.environ["PRISMABUILD_CONTAINER_OWNER"],
                  "prismaquant.pq87.campaign": label}
        row = {"Id": cid, "Name": "/" + name, "Config": {"Labels": labels},
               "State": {"Running": True}}
        if mode == "wrong_action":
            labels["prismabuild.action"] = "another-action"
        elif mode == "wrong_nonce":
            labels["prismaquant.pq87.campaign"] = "another-campaign"
        elif mode == "wrong_name":
            row["Name"] = "/renamed-container"
        state["containers"][cid] = row
        cidfile.write_text(cid + "\n")
        (cidfile.parent / "inside-status.json").write_text(json.dumps({"status": "completed"}))
        if mode == "failed_owned_create":
            print("Error: client failed after CID", file=sys.stderr)
            code = 125
        elif mode in {"auto_removed", "replaced_name"}:
            del state["containers"][cid]
            if mode == "replaced_name":
                state["containers"][foreign] = borrowed(name)
elif argv[:1] == ["stop"]:
    key = find(argv[-1])
    if key is not None:
        state["containers"][key]["State"]["Running"] = False
elif argv[:2] == ["rm", "-f"]:
    key = find(argv[-1])
    if key is not None:
        del state["containers"][key]
else:
    print("Unexpected Docker request: " + repr(argv), file=sys.stderr)
    code = 97
path.write_text(json.dumps(state))
raise SystemExit(code)
'''


def _lifecycle_host(tmp_path, mode):
    """Real host, source hashing and freeze; Docker/telemetry are external doubles."""
    import struct

    state = tmp_path / "docker-state.json"
    state.write_text(json.dumps({"calls": [], "containers": {}, "mode": mode}))
    commands = tmp_path / "bin"
    commands.mkdir()
    docker = commands / "docker"
    docker.write_text("#!" + sys.executable + "\n" + textwrap.dedent(LIFECYCLE_DOCKER))
    docker.chmod(0o755)
    smi = commands / "nvidia-smi"
    smi.write_text("#!" + sys.executable + "\nraise SystemExit(97)\n")
    smi.chmod(0o755)
    model = tmp_path / "source-model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"audit_toy"}')
    header = json.dumps({"proj.weight": {"dtype": "F32", "shape": [1, 1],
                                        "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    (model / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    output = tmp_path / "same-basename"
    env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"],
               AUDIT_DOCKER_STATE=str(state), CUDA_VISIBLE_DEVICES="",
               PQ_RUNTIME_PRISMAQUANT_ROOT=str(ROOT), PYTHONNOUSERSITE="1",
               PYTHONSAFEPATH="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    assert env.get("PRISMABUILD_CONTAINER_OWNER"), "fixture requires its real admitted PB owner"
    runner = '''import runpy, sys, urllib.request
def unavailable(*args, **kwargs):
    raise OSError("audit external telemetry unavailable")
urllib.request.urlopen = unavailable
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
    argv = [sys.executable, "-P", "-c", runner, str(ROOT / "experiments/pq87_physical_ab.py"),
            "--image", IMAGE, "--out", str(output), "--model-dir", str(model), "--seconds", "20"]
    with subprocess.Popen(argv, cwd=ROOT, env=env, start_new_session=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=45)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate(timeout=10)
            pytest.fail(f"host fixture timed out: {stdout}\n{stderr}")
    assert process.returncode == 0, (stdout, stderr)
    report = json.loads((output / "campaign.json").read_bytes())
    assert re.fullmatch("[0-9a-f]{40}", report["source_snapshot"])
    return report, json.loads(state.read_bytes())


def test_owned_creation_cleans_only_the_verified_cid(tmp_path):
    report, observed = _lifecycle_host(tmp_path, "owned")
    assert report["status"] == "completed", report
    assert report["cleanup"]["safe"] is True
    assert report["cleanup"]["container_id"] == "c" * 64
    assert observed["containers"] == {}
    assert [call[-1] for call in observed["calls"] if call[0] in {"stop", "rm"}] == ["c" * 64] * 2


@pytest.mark.parametrize("mode", ["failed_create", "failed_owned_create"])
def test_failed_creation_never_falls_back_to_name_cleanup(tmp_path, mode):
    report, observed = _lifecycle_host(tmp_path, mode)
    assert report["status"] == "inconclusive"
    assert report["cleanup"]["attempted"] is True
    assert observed["containers"] == {}
    targets = [call[-1] for call in observed["calls"] if call[0] in {"stop", "rm"}]
    if mode == "failed_create":
        assert targets == []
        assert report["cleanup"]["safe"] is False
    else:
        assert targets == ["c" * 64] * 2
        assert report["cleanup"]["safe"] is True


def test_existing_name_refuses_before_creation_and_remains_untouched(tmp_path):
    report, observed = _lifecycle_host(tmp_path, "occupied_name")
    assert report["status"] == "inconclusive"
    assert "already exists" in report["error"]
    assert report["cleanup"]["attempted"] is False
    assert observed["containers"]["f" * 64]["State"]["Running"] is True
    assert not any(call[0] in {"run", "stop", "rm"} for call in observed["calls"])


def test_name_reused_after_preflight_is_not_borrowed_on_failed_create(tmp_path):
    report, observed = _lifecycle_host(tmp_path, "reused_name")
    assert report["status"] == "inconclusive"
    assert report["cleanup"]["safe"] is False
    assert observed["containers"]["f" * 64]["State"]["Running"] is True
    assert not any(call[0] in {"stop", "rm"} for call in observed["calls"])


@pytest.mark.parametrize("mode", ["wrong_action", "wrong_nonce", "wrong_name"])
def test_cleanup_refuses_a_cid_with_changed_ownership(tmp_path, mode):
    report, observed = _lifecycle_host(tmp_path, mode)
    assert report["status"] == "inconclusive"
    assert report["cleanup"]["safe"] is False
    assert "ownership differs" in report["cleanup"]["error"]
    assert observed["containers"]["c" * 64]["State"]["Running"] is True
    assert not any(call[0] in {"stop", "rm"} for call in observed["calls"])


@pytest.mark.parametrize("mode", ["auto_removed", "replaced_name"])
def test_absent_owned_cid_never_cleans_a_successor_by_name(tmp_path, mode):
    report, observed = _lifecycle_host(tmp_path, mode)
    assert report["status"] == "completed", report
    assert report["cleanup"]["safe"] is True
    assert not any(call[0] in {"stop", "rm"} for call in observed["calls"])
    if mode == "replaced_name":
        assert observed["containers"]["f" * 64]["State"]["Running"] is True
    else:
        assert observed["containers"] == {}
