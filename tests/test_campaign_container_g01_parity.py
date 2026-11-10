"""The adapter's create flags and cleanup keep the accepted Tessera G01 controls.

The controls are a committed extract of tools/_suite_container.py at the pinned
Tessera commit (read, never imported). Admitted proof jobs re-verify the extract
against the Tessera source blob; this unit test pins the adapter to the extract.
"""
import importlib
import json
from pathlib import Path

from container_admission import ADMISSION, CIDFILE

CONTROLS = json.loads((Path(__file__).parent / "fixtures" / "g01_suite_container_controls.json").read_text())
runner = importlib.import_module("tools.tessera_campaign_container")


def _argv(machine, monkeypatch):
    monkeypatch.setattr(runner.os, "uname", lambda: type("U", (), {"machine": machine})())
    return runner.podman_command({"container": {"image": "i", "mounts": []}, "env": {}}, ["x"], cwd="/s",
                                 admission=ADMISSION, cidfile=CIDFILE, image_id="sha256:" + "a" * 64)


def test_every_g01_create_flag_is_present_or_a_named_delta(monkeypatch):
    argv = _argv("x86_64", monkeypatch)
    for flag in CONTROLS["create_flags"]:
        if flag in CONTROLS["documented_deltas"]:
            assert flag not in argv
        else:
            assert flag in argv, flag


def test_user_mapping_matches_g01_per_architecture(monkeypatch):
    assert all(flag in _argv("x86_64", monkeypatch) for flag in CONTROLS["user_flags_x86_64"])
    arm = _argv("aarch64", monkeypatch)
    assert all(flag in arm for flag in CONTROLS["user_flags_aarch64"])
    assert "--user=0:0" not in arm


def test_the_g01_delta_flags_are_documented_not_silent():
    assert set(CONTROLS["documented_deltas"]) == {"--unsetenv-all", "--read-only"}
    assert all(CONTROLS["documented_deltas"].values())


def test_cleanup_and_start_sequence_matches_g01(monkeypatch, tmp_path):
    from fake_podman import install, calls
    install(monkeypatch, runner, tmp_path)
    record = {"cidfile": str(tmp_path / "c.cid"), "cid": None, "cleanup_errors": [], "removed": False}
    Path(record["cidfile"]).write_text("a" * 64)
    runner._cleanup_container(record)
    expected = [tuple(command) for command in CONTROLS["cleanup_commands"]]
    seen = calls(tmp_path)
    assert [call[:len(command)] for call, command in zip(seen, expected)] == expected
    assert len(seen) == len(expected) and all(call[-1] == "a" * 64 for call in seen)
    assert record["removed"] is True and not record["cleanup_errors"]
