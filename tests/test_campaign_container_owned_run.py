"""The owned container lifetime: one captured CID, failed start, cancel, cleanup, evidence."""
import hashlib
import importlib
import json
import os
import signal
import threading
from pathlib import Path

import pytest

from container_admission import ADMISSION
from fake_podman import CID, calls, install

runner = importlib.import_module("tools.tessera_campaign_container")
IMAGE = "sha256:" + "b" * 64


def spec():
    return {"container": {"image": "qualified:fixed", "mounts": []}, "env": {}}


def launch(tmp_path, **kwargs):
    record_dir = tmp_path / "record"
    record_dir.mkdir(exist_ok=True)
    return runner.run_owned_container(
        spec(), ["python3", "-c", "pass"], cwd="/snapshot", image_id=IMAGE, content_sha256="c" * 64,
        admission={**ADMISSION, "record_dir": str(record_dir)}, record_dir=record_dir, environ={}, **kwargs), record_dir


def record_of(record_dir):
    return json.loads((record_dir / "container.json").read_text())


def test_positive_run_captures_the_cid_inspects_and_removes_it(monkeypatch, tmp_path, capsys):
    install(monkeypatch, runner, tmp_path)
    code, record_dir = launch(tmp_path)
    record = record_of(record_dir)
    assert code == 0 and record["cid"] == CID and record["removed"] is True and not record["cleanup_errors"]
    assert record["schema"] == runner.EXECUTION_SCHEMA and record["gpus"] == 0
    assert record["affinity"] == ADMISSION["affinity"] and record["memory_gib"] == 16
    assert record["inspection"] == {"Id": CID, "Image": "b" * 64, "CpusetCpus": "21,22,23,30", "Devices": []}
    assert set(record) == {"schema", "job_id", "launch_sha256", "source", "image", "cidfile", "cid", "affinity",
                           "memory_gib", "gpus", "creation_returncode", "inspection", "returncode",
                           "interrupted_signal", "cleanup_errors", "removed"}
    verbs = [c[0] for c in calls(tmp_path)]
    assert verbs[:3] == ["create", "inspect", "start"] and verbs[-1] == "container"
    out = capsys.readouterr().out.splitlines()[-1]
    path = record_dir / "container.json"
    assert out == f"campaign container evidence {path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}"


def test_the_record_with_its_cidfile_exists_before_create(monkeypatch, tmp_path):
    install(monkeypatch, runner, tmp_path)
    seen = {}
    real = runner.subprocess.run

    def spy(argv, **kwargs):
        if argv[1] == "create":
            seen["record"] = json.loads((tmp_path / "record" / "container.json").read_text())
        return real(argv, **kwargs)
    monkeypatch.setattr(runner.subprocess, "run", spy)
    launch(tmp_path)
    assert seen["record"]["cid"] is None and seen["record"]["cidfile"].endswith("container.cid")


def test_a_payload_failure_passes_through_and_the_container_is_still_removed(monkeypatch, tmp_path):
    install(monkeypatch, runner, tmp_path, start_rc=7)
    code, record_dir = launch(tmp_path)
    record = record_of(record_dir)
    assert code == 7 and record["returncode"] == 7 and record["removed"] is True


def test_failed_start_records_the_unknown_scope_and_exits_92(monkeypatch, tmp_path):
    install(monkeypatch, runner, tmp_path)
    (tmp_path / "create.fail").write_text("")
    code, record_dir = launch(tmp_path)
    record = record_of(record_dir)
    assert code == runner.EXIT_CREATE_FAILED and record["cid"] is None and record["creation_returncode"] == 125
    assert any("create failed" in error for error in record["cleanup_errors"])
    assert [c[0] for c in calls(tmp_path)] == ["create"]  # nothing was ever stopped or removed


def test_a_wrong_cpuset_or_a_device_is_refused_and_the_owned_cid_is_removed(monkeypatch, tmp_path):
    for case, kwargs in (("cpuset", dict(cpuset="1,2")), ("device", dict(devices=[{"PathOnHost": "/dev/nvidia0"}]))):
        directory = tmp_path / case
        directory.mkdir()
        install(monkeypatch, runner, directory, **kwargs)
        code, record_dir = launch(directory)
        record = record_of(record_dir)
        assert code == runner.EXIT_CREATE_FAILED, case
        assert record["cid"] == CID and record["removed"] is True, case
        assert "start" not in [c[0] for c in calls(directory)], case


def test_a_wrong_image_is_refused(monkeypatch, tmp_path):
    install(monkeypatch, runner, tmp_path, image="sha256:" + "d" * 64)
    code, record_dir = launch(tmp_path)
    assert code == runner.EXIT_CREATE_FAILED and record_of(record_dir)["removed"] is True


def test_a_clean_payload_with_failed_cleanup_is_not_a_success(monkeypatch, tmp_path):
    install(monkeypatch, runner, tmp_path)
    (tmp_path / "rm.fail").write_text("")
    code, record_dir = launch(tmp_path)
    record = record_of(record_dir)
    assert code == runner.EXIT_CLEANUP_FAILED and record["removed"] is False
    assert any(error.startswith("rm:") for error in record["cleanup_errors"])


def test_interruption_stops_and_removes_only_the_captured_cid(monkeypatch, tmp_path):
    install(monkeypatch, runner, tmp_path)
    (tmp_path / "start.sleep").write_text("")
    threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    code, record_dir = launch(tmp_path)
    record = record_of(record_dir)
    assert code == 128 + signal.SIGTERM and record["interrupted_signal"] == signal.SIGTERM
    assert record["removed"] is True and not record["cleanup_errors"]
    touched = {c[-1] for c in calls(tmp_path) if c[0] in ("stop", "rm", "container")}
    assert touched == {CID}


def test_an_existing_record_or_cidfile_refuses_before_any_podman_call(monkeypatch, tmp_path):
    install(monkeypatch, runner, tmp_path)
    record_dir = tmp_path / "record"
    record_dir.mkdir()
    (record_dir / "container.cid").write_text(CID)
    with pytest.raises(RuntimeError, match="already exists"):
        launch(tmp_path)
    assert calls(tmp_path) == []


def test_a_cidfile_replaced_by_a_symlink_is_not_followed(monkeypatch, tmp_path):
    install(monkeypatch, runner, tmp_path)
    cidfile = tmp_path / "c.cid"
    cidfile.symlink_to(tmp_path / "elsewhere")
    record = {"cidfile": str(cidfile), "cid": None, "cleanup_errors": [], "removed": False}
    runner._cleanup_container(record)
    assert record["removed"] is False and any("symlink" in e for e in record["cleanup_errors"])
    assert calls(tmp_path) == []
