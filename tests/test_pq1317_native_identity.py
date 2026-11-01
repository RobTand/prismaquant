"""CPU regression proof for the qualification observer, not GPU qualification.

Real mmap cases exercise the reviewer's pathname-replacement reproduction. The
map_files-denied cases also cover containers without CAP_CHECKPOINT_RESTORE.
"""
from __future__ import annotations

import copy
import errno
import importlib.util
import json
import mmap
import os
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "docs/measurements/pq1317-gpu-tests/harness"


def load(name):
    spec = importlib.util.spec_from_file_location(f"pq1317_{name}", HARNESS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe():
    return load("native_probe")


@pytest.fixture
def mapped(tmp_path, probe):
    path = tmp_path / "tessera_identity.so"
    path.write_bytes(b"original".ljust(mmap.PAGESIZE, b"x"))
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
        entry, = [e for e in probe._maps() if e["path"] == str(path)]
        yield path, mapping, entry


def deny_map_files(monkeypatch):
    original = Path.open

    def opened(path, *args, **kwargs):
        if str(path).startswith("/proc/self/map_files/"):
            raise PermissionError(errno.EPERM, "map_files denied", str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)


def test_maps_preserve_kernel_identity_including_hex_device(probe):
    entry, = probe.parse_maps([
        "1000-2000 r-xp 00001000 1a:2b 987 /build/"
        "tessera_identity_sm_121_tessera_guarded_v1/.nfs123 (deleted)"
    ])
    assert entry == {
        "module": "tessera_identity", "address": "1000-2000", "permissions": "r-xp",
        "offset": 4096, "device_major": 26, "device_minor": 43, "inode": 987,
        "path": "/build/tessera_identity_sm_121_tessera_guarded_v1/.nfs123",
        "nfs_renamed": True, "deleted": True,
    }


def test_fallback_records_reader_and_matches_open_descriptor(probe, mapped, monkeypatch):
    path, _, entry = mapped
    deny_map_files(monkeypatch)
    record = probe._digest(entry)
    assert record["identity_verified"] is True
    assert record["reader"] == "pathname"
    assert record["sha256"] == probe.hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["backing_file"]["inode"] == entry["inode"] == path.stat().st_ino
    assert record["attempts"][0]["reader"] == "map_files"
    assert "denied" in record["attempts"][0]["error"]
    assert record["attempts"][1]["opened_file"] == record["backing_file"]


def replace_path(path):
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(b"replacement".ljust(mmap.PAGESIZE, b"y"))
    replacement.replace(path)


def test_unmocked_replaced_path_never_attests_replacement_bytes(probe, mapped):
    path, mapping, _ = mapped
    original = probe.hashlib.sha256(mapping[:]).hexdigest()
    replace_path(path)
    assert mapping[:8] == b"original"
    info = probe.mapped_native_libraries()["tessera_identity"]
    assert info["sha256"] != probe.hashlib.sha256(path.read_bytes()).hexdigest()
    if info["identity_verified"]:
        assert info["sha256"] == original
        assert {m["reader"] for m in info["mappings"]} == {"map_files"}
        assert probe.native_identity_errors({"tessera_identity": info}) == []
    else:
        assert info["sha256"] is None
        assert probe.native_identity_errors({"tessera_identity": info})


def test_denied_map_files_replaced_path_fails_closed_even_after_a_good_read(probe, mapped, monkeypatch):
    path, _, entry = mapped
    deny_map_files(monkeypatch)
    assert probe._digest(entry)["identity_verified"]
    replace_path(path)
    record = probe._digest(entry)
    assert record["sha256"] is None and record["reader"] is None
    assert record["identity_verified"] is False
    assert "device/inode differs" in record["attempts"][1]["error"]


def test_fallback_accepts_nfs_style_rename_only_for_the_same_inode(probe, mapped, monkeypatch):
    path, _, entry = mapped
    directory = path.parent / "tessera_identity_sm_121_tessera_guarded_v1"
    directory.mkdir()
    renamed = directory / ".nfs123"
    path.rename(renamed)
    deny_map_files(monkeypatch)
    current, = [e for e in probe._maps() if e["path"] == str(renamed)]
    assert current["nfs_renamed"] is True and current["inode"] == entry["inode"]
    assert probe._digest(current)["identity_verified"] is True


def test_deleted_file_without_map_files_is_unqualified(probe, mapped, monkeypatch):
    path, _, _ = mapped
    path.unlink()
    deny_map_files(monkeypatch)
    info = probe.mapped_native_libraries()["tessera_identity"]
    assert info["deleted"] is True
    assert info["sha256"] is None and info["identity_verified"] is False


@pytest.mark.parametrize("field", ["device_major", "device_minor", "inode"])
def test_fallback_checks_every_identity_coordinate(probe, mapped, monkeypatch, field):
    _, _, entry = mapped
    deny_map_files(monkeypatch)
    altered = {**entry, field: entry[field] + 1}
    record = probe._digest(altered)
    assert record["sha256"] is None
    assert "device/inode differs" in record["attempts"][1]["error"]


def test_change_while_hashing_is_rejected(probe, mapped, monkeypatch):
    path, _, entry = mapped
    deny_map_files(monkeypatch)
    original = probe.hashlib.file_digest

    def changed(handle, algorithm):
        result = original(handle, algorithm)
        with path.open("ab") as write:
            write.write(b"changed")
        return result

    monkeypatch.setattr(probe.hashlib, "file_digest", changed)
    record = probe._digest(entry)
    assert record["sha256"] is None
    assert "changed while hashing" in record["attempts"][1]["error"]


def test_disappeared_mapping_is_rejected(probe, mapped, monkeypatch):
    _, _, entry = mapped
    deny_map_files(monkeypatch)
    monkeypatch.setattr(probe, "_maps", lambda: [])
    record = probe._digest(entry)
    assert record["sha256"] is None
    assert "mapping changed" in record["attempts"][1]["error"]


def test_unreadable_maps_cannot_look_like_an_empty_inventory(probe, monkeypatch):
    def unreadable():
        raise PermissionError("maps denied")

    monkeypatch.setattr(probe, "_maps", unreadable)
    info = probe.mapped_native_libraries()
    assert probe.native_identity_errors(info)
    assert "maps denied" in next(iter(info.values()))["error"]


def test_conflicting_segments_are_preserved_and_refused(probe, mapped, monkeypatch):
    _, _, entry = mapped
    deny_map_files(monkeypatch)
    good = probe._digest(entry)
    other = {**entry, "address": "1-2", "inode": entry["inode"] + 1}
    bad = {**other, "sha256": None, "identity_verified": False}
    monkeypatch.setattr(probe, "_maps", lambda: [entry, other])
    monkeypatch.setattr(probe, "_digest", lambda e: good if e is entry else bad)
    info = probe.mapped_native_libraries()["tessera_identity"]
    assert len(info["mappings"]) == 2
    assert info["sha256"] is None and info["identity_verified"] is False


@pytest.fixture
def valid_info(probe, mapped, monkeypatch):
    deny_map_files(monkeypatch)
    info = probe.mapped_native_libraries()["tessera_identity"]
    assert probe.native_identity_errors({"tessera_identity": info}) == []
    return info


@pytest.mark.parametrize("defect", ["legacy", "reader", "inode", "digest", "attempt"])
def test_qualification_rejects_missing_or_inconsistent_provenance(probe, valid_info, defect):
    info = copy.deepcopy(valid_info)
    if defect == "legacy":
        info.pop("mappings")
    elif defect == "reader":
        info["mappings"][0].pop("reader")
    elif defect == "inode":
        info["mappings"][0]["backing_file"]["inode"] += 1
    elif defect == "digest":
        info["mappings"][0]["sha256"] = "a" * 64
    else:
        info["mappings"][0]["attempts"] = []
    assert probe.native_identity_errors({"tessera_identity": info})


@pytest.fixture
def entrypoint(monkeypatch):
    monkeypatch.syspath_prepend(str(HARNESS))
    return load("container_entry")


def records_for(info):
    tests, sessions = [], []
    for worker in ("gw0", "gw1"):
        common = {"worker": worker, "pid": 1, "native_libraries_mapped": {"tessera_identity": info}}
        tests.append({**common, "kind": "test", "nodeid": f"test.py::{worker}", "when": "call",
                      "outcome": "passed", "assertions": [], "assertions_dropped": 0})
        sessions.append({**common, "kind": "session", "executes_tests": True, "exitstatus": 0})
    return [copy.deepcopy(record) for record in tests + sessions]


def merge(entrypoint, directory, records):
    probe_dir = directory / "probe"
    probe_dir.mkdir(exist_ok=True)
    (probe_dir / "probe.main.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return entrypoint.merge_probe(directory, ["test.py::gw0", "test.py::gw1"])


def test_merge_accepts_verified_records_and_retains_sessions(entrypoint, valid_info, tmp_path):
    summary = merge(entrypoint, tmp_path, records_for(valid_info))
    assert summary["qualification_ok"] is True and summary["errors"] == []
    assert len(json.loads((tmp_path / "probe-sessions.json").read_text())) == 2
    assert summary["native_library_sha256_seen"] == {"tessera_identity": [valid_info["sha256"]]}


@pytest.mark.parametrize("defect", ["legacy", "missing_session", "dropped", "duplicate", "skip", "observer"])
def test_merge_fails_closed_for_incomplete_evidence(entrypoint, valid_info, tmp_path, defect):
    records = records_for(valid_info)
    if defect == "legacy":
        records[0]["native_libraries_mapped"]["tessera_identity"].pop("mappings")
    elif defect == "missing_session":
        records.pop()
    elif defect == "dropped":
        records[0]["assertions_dropped"] = 1
    elif defect == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif defect == "skip":
        records[0]["outcome"] = "skipped"
    else:
        records.append({"kind": "error", "where": "observer", "error": "read failed"})
    summary = merge(entrypoint, tmp_path, records)
    assert summary["qualification_ok"] is False and summary["errors"]


def test_merge_never_overwrites_a_conflicting_binary(entrypoint, valid_info, tmp_path):
    records = records_for(valid_info)
    other = records[2]["native_libraries_mapped"]["tessera_identity"]
    other["sha256"] = "a" * 64
    for mapping in other["mappings"]:
        mapping["sha256"] = "a" * 64
    summary = merge(entrypoint, tmp_path, records)
    assert summary["qualification_ok"] is False
    assert any("changed during" in r["error"] for r in summary["errors"])
    assert set(summary["native_library_sha256_seen"]["tessera_identity"]) == {"a" * 64, valid_info["sha256"]}


def test_successful_pytest_cannot_override_failed_attestation(entrypoint, valid_info, tmp_path, monkeypatch):
    records = records_for(valid_info)
    records[0]["native_libraries_mapped"]["tessera_identity"].pop("mappings")
    merge(entrypoint, tmp_path, records)
    expected = tmp_path / "nodes.json"
    expected.write_text(json.dumps(["test.py::gw0", "test.py::gw1"]))
    monkeypatch.setattr(entrypoint, "install_test_tools", lambda out: {})
    monkeypatch.setattr(entrypoint, "bind_source", lambda *a: {
        "verified_at_candidate": True, "contract_matches_pin": True})
    monkeypatch.setattr(entrypoint, "runtime_facts", lambda: {})
    monkeypatch.setattr(entrypoint, "selftest_probe", lambda out: {"ok": True})
    monkeypatch.setattr(entrypoint, "collect_nodes", lambda *a: ["test.py::gw0", "test.py::gw1"])
    monkeypatch.setattr(entrypoint, "probe_loaded_in_collection", lambda out: True)
    monkeypatch.setattr(entrypoint, "run_suite", lambda *a: 0)
    code = entrypoint.main(["--suite", "dense", "--mode", "run", "--out", str(tmp_path),
                            "--source", str(tmp_path), "--candidate", "unused",
                            "--contract-sha256", "unused", "--nodes-json", str(expected)])
    assert code == 6
    assert json.loads((tmp_path / "container-report.json").read_text())["exit_code"] == 6


@pytest.mark.parametrize("defect", [None, "legacy", "built_digest", "unqualified", "missing"])
def test_final_build_inventory_join_fails_closed(entrypoint, valid_info, tmp_path, defect):
    runner = load("run_suite")
    summary = merge(entrypoint, tmp_path, records_for(valid_info))
    built = [{"path": "tessera_identity.so", "sha256": valid_info["sha256"]}]
    if defect == "legacy":
        summary["native_libraries_by_worker"]["gw0"]["tessera_identity"].pop("mappings")
    elif defect == "built_digest":
        built[0]["sha256"] = "a" * 64
    elif defect == "unqualified":
        summary["qualification_ok"] = False
    (tmp_path / "probe-summary.json").write_text(json.dumps(summary))
    if defect == "missing":
        (tmp_path / "probe-summary.json").unlink()
    errors = runner.native_inventory_errors(tmp_path, built)
    assert bool(errors) == (defect is not None)


def test_offline_tools_require_supplied_execution_versions(entrypoint, tmp_path, monkeypatch):
    versions = {pin.split("==")[0]: pin.split("==")[1]
                for pin in entrypoint.TEST_TOOLS.values()}
    versions["Pygments"] = "2.21.0"
    versions["packaging"] = "26.3"
    monkeypatch.setattr(entrypoint.importlib.util, "find_spec", lambda module: object())
    monkeypatch.setattr(entrypoint.importlib.metadata, "version", versions.__getitem__)
    result = entrypoint.install_test_tools(tmp_path)
    assert result["installed"] == [] and result["versions"] == versions
    versions["pytest"] = "9.0.0"
    with pytest.raises(RuntimeError, match="expected 8.4.2"):
        entrypoint.install_test_tools(tmp_path)


def test_offline_tools_never_install_a_missing_dependency(entrypoint, tmp_path, monkeypatch):
    monkeypatch.setattr(entrypoint.importlib.util, "find_spec", lambda module: None)
    with pytest.raises(RuntimeError, match="missing pre-provisioned test tool"):
        entrypoint.install_test_tools(tmp_path)
