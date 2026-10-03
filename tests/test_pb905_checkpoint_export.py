"""CPU component controls; no synthetic workload can qualify PB905's I/O gate."""
from __future__ import annotations

import hashlib
import copy
import json
from pathlib import Path

import pytest
import torch

from experiments import pb905_checkpoint_export as exp
from prismaquant.cost_streaming import BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts
from prismaquant.perturbed_x_cache import EntryReadScratch, write_exact_activation_cache_entry
from prismaquant.joint_adjoint_checkpoints import exact_entry_record


def _owner(tmp_path, *, budget=1 << 20):
    owner = StreamedBoundaryArtifacts({
        "schema": BOUNDARY_STORAGE_SCHEMA, "directory": str(tmp_path / "writer"),
        "max_resident_bytes": budget, "max_auxiliary_bytes": 1 << 20,
        "max_artifact_bytes": 1 << 20, "prefetch_batches": 2})
    owner.bind({"component": "PB905"}, n_probes=1)
    return owner


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_LAYER_READ_THREADS", "1")
    session = {"generation": "fixture", "run_identity_sha256": "a" * 64}
    ref = write_exact_activation_cache_entry(
        tmp_path / "original", "cotangent-0-0-at-40", torch.arange(8).reshape(2, 4),
        identity={"session": session, "slot": "cotangent-0-0", "kind": "cotangent",
                  "coordinates": {"batch": 0, "probe": 0, "boundary": 40}},
        max_tensor_bytes=64, max_file_bytes=65536)
    return exact_entry_record(ref), session


def test_original_serialized_bytes_come_from_the_verified_single_read(source, tmp_path, monkeypatch):
    entry, session = source
    path = Path(entry["path"])
    expected = path.read_bytes()
    before = path.stat()
    owner = _owner(tmp_path)
    scratch = EntryReadScratch()
    original = Path.open
    opens = []
    def opened(self, *args, **kwargs):
        if self == path:
            opens.append(args[0] if args else kwargs.get("mode"))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", opened)
    with owner:
        observed = exp.verified_bytes(entry, expected_session=session, scratch=scratch, owner=owner)
        assert observed == expected
        assert hashlib.sha256(observed).hexdigest() == entry["sha256"]
        assert opens == ["rb"]
        assert owner.telemetry["resident_tensor_bytes"] == 0
    scratch.release()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
        path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)


def test_body_change_refuses_before_writer_can_receive_bytes(source, tmp_path):
    entry, session = source
    path = Path(entry["path"])
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    owner = _owner(tmp_path)
    scratch = EntryReadScratch()
    with owner:
        with pytest.raises(RuntimeError, match="checksum"):
            exp.verified_bytes(entry, expected_session=session, scratch=scratch, owner=owner)
        assert owner.telemetry["resident_tensor_bytes"] == 0
    scratch.release()


def test_different_session_and_nonserial_reader_refuse(source, tmp_path, monkeypatch):
    entry, session = source
    owner = _owner(tmp_path)
    scratch = EntryReadScratch()
    with owner:
        with pytest.raises(RuntimeError, match="session"):
            exp.verified_bytes(entry, expected_session={}, scratch=scratch, owner=owner)
        monkeypatch.setenv("PRISMAQUANT_LAYER_READ_THREADS", "2")
        with pytest.raises(ValueError, match="serial"):
            exp.verified_bytes(entry, expected_session=session, scratch=scratch, owner=owner)
        assert owner.telemetry["resident_tensor_bytes"] == 0
    scratch.release()


def test_group_residency_and_exception_do_not_leak_a_charge(source, tmp_path):
    entry, session = source
    group = {"index": 0, "paced": True, "entries": [entry]}
    owner = _owner(tmp_path)
    scratch = EntryReadScratch()
    with owner:
        with pytest.raises(OSError, match="export fault"):
            with exp.verified_group(group, expected_session=session, scratch=scratch,
                                    owner=owner, keep_bytes=True) as (files, checked):
                assert len(files) == 1 and checked[0]["bytes"] == entry["file_bytes"]
                assert owner.telemetry["resident_tensor_bytes"] == 0
                raise OSError("export fault")
        assert owner.telemetry["resident_tensor_bytes"] == 0
    scratch.release()


def test_an_insufficient_group_budget_refuses_before_read(source, tmp_path, monkeypatch):
    entry, session = source
    owner = _owner(tmp_path, budget=64)
    def forbidden(*args, **kwargs):
        raise AssertionError("an unadmitted body was opened")
    monkeypatch.setattr(exp, "verified_bytes", forbidden)
    with owner:
        with pytest.raises(RuntimeError, match="residency budget"):
            with exp.verified_group({"entries": [entry]}, expected_session=session,
                                    scratch=EntryReadScratch(), owner=owner, keep_bytes=True):
                pass
        assert owner.telemetry["resident_tensor_bytes"] == 0


def test_frozen_adapter_changes_only_the_existing_per_group_hook():
    calls = []
    class Backend:
        max_bytes = 123
        def submit_group(self, batch_id, *, entries, paced=None):
            calls.append((batch_id, entries, paced))
            return {"ok": True, "export_key": "f" * 64}
        def release_group(self, batch_id):
            calls.append(("release", batch_id))
    arms = {"first": True, "second": False}
    backend = exp.FrozenArmBackend(Backend(), arms)
    arms["first"] = False
    entries = [{"original": "unchanged"}]
    assert backend.submit_group("first", entries=entries)["ok"]
    assert backend.submit_group("second", entries=entries)["ok"]
    assert calls == [("first", entries, True), ("second", entries, False)]
    assert backend.max_bytes == 123
    backend.release_group("second")
    assert calls[-1] == ("release", "second")
    with pytest.raises(ValueError, match="roster"):
        backend.submit_group("foreign", entries=entries)
    assert len(backend.observations) == 2
    assert [row["paced"] for row in backend.observations] == [True, False]


def test_adapter_preserves_a_primary_submission_failure():
    class Backend:
        def submit_group(self, *args, **kwargs):
            raise OSError("funding unavailable")
    backend = exp.FrozenArmBackend(Backend(), {"first": True})
    with pytest.raises(OSError, match="funding unavailable"):
        backend.submit_group("first", entries=[])
    assert backend.observations[0]["status"] == "failed"


def test_evidence_publication_does_not_adopt_existing_bytes(tmp_path):
    path = tmp_path / "record.json"
    exp._new_json(path, {"original": True})
    raw = path.read_bytes()
    with pytest.raises(FileExistsError):
        exp._new_json(path, {"replacement": True})
    assert path.read_bytes() == raw


def _packet(tmp_path, source):
    from prismaquant.joint_adjoint_checkpoints import (
        ADJOINT_CHECKPOINT_PACKED_SCHEMA, checkpoint_manifest_bytes, checkpoint_seal_sha256)
    entry, session = source
    entries = []
    for index in range(exp.GROUPS * exp.GROUP_SIZE):
        row = copy.deepcopy(entry)
        row["path"] = str(tmp_path / "original" / f"entry-{index:04d}.pt")
        row["name"] = f"entry-{index:04d}"
        entries.append(row)
    record = {"schema": ADJOINT_CHECKPOINT_PACKED_SCHEMA, "boundary": 40,
              "session": session, "activation_entries": entries, "shared_state_entries": []}
    record["cotangent_sha256"] = checkpoint_seal_sha256(record)
    checkpoint_raw = checkpoint_manifest_bytes(record)
    group_bytes = exp.GROUP_SIZE * entry["file_bytes"]
    prefix = str(tmp_path / "new-evidence")
    return {"schema": exp.SCHEMA, "scope": "offline-checkpoint-export-component",
            "checkpoint": {"path": str(tmp_path / "original" / "checkpoint.json"),
                           "bytes": len(checkpoint_raw), "sha256": exp.sha(checkpoint_raw)},
            "checkpoint_record": record,
            "groups": [{"index": index, "paced": index % 2 == 0,
                        "entries": entries[index * exp.GROUP_SIZE:(index + 1) * exp.GROUP_SIZE]}
                       for index in range(exp.GROUPS)],
            "output_prefix": prefix, "tier": "prismabuild-stage:fixture",
            "group_bytes": group_bytes, "total_bytes": exp.GROUPS * group_bytes,
            "storage": {"schema": BOUNDARY_STORAGE_SCHEMA, "directory": prefix + "/exact",
                        "max_resident_bytes": group_bytes + 4 * entry["file_bytes"] + entry["tensor_bytes"],
                        "max_auxiliary_bytes": 4 << 20, "max_artifact_bytes": exp.GROUPS * group_bytes,
                        "prefetch_batches": exp.GROUP_SIZE},
            "spool_max_bytes": 2 * group_bytes, "source_proposal_sha256": "a" * 64}


def test_complete_roster_remains_twelve_fixed_equal_byte_cohorts(source, tmp_path):
    packet = _packet(tmp_path, source)
    assert exp.check_packet(packet) is packet
    assert len({row["path"] for group in packet["groups"] for row in group["entries"]}) == 768
    assert [group["paced"] for group in packet["groups"]] == [True, False] * 6
    assert exp.selected_groups(packet, 7) == [packet["groups"][7]]
    with pytest.raises(ValueError, match="roster"):
        exp.selected_groups(packet, True)
    with pytest.raises(ValueError, match="roster"):
        exp.selected_groups(packet, 12)


@pytest.mark.parametrize("fault", ["repeated", "changed", "reordered", "checkpoint-bytes", "arm", "missing", "index-bool", "budget", "origin", "relative-origin"])
def test_packet_change_refuses_before_any_body_read(source, tmp_path, fault):
    packet = _packet(tmp_path, source)
    if fault == "repeated":
        packet["groups"][1]["entries"][0] = packet["groups"][0]["entries"][0]
    elif fault == "changed":
        # Group membership is independently anchored in the full checkpoint.
        packet["groups"][0]["entries"] = copy.deepcopy(packet["groups"][0]["entries"])
        packet["groups"][0]["entries"][0]["sha256"] = "f" * 64
    elif fault == "arm":
        packet["groups"][0]["paced"] = False
    elif fault == "reordered":
        packet["groups"][0]["entries"] = list(reversed(packet["groups"][0]["entries"]))
    elif fault == "checkpoint-bytes":
        packet["checkpoint"]["bytes"] += 1
    elif fault == "missing":
        packet["groups"].pop()
    elif fault == "index-bool":
        packet["groups"][0]["index"] = False
    elif fault == "budget":
        packet["spool_max_bytes"] -= 1
    elif fault == "relative-origin":
        packet["output_prefix"] = "relative-output"
    else:
        packet["output_prefix"] = str(tmp_path / "original")
        packet["storage"]["directory"] = packet["output_prefix"] + "/exact"
    with pytest.raises(ValueError):
        exp.check_packet(packet)


def test_authentication_scope_projects_exactly_one_logical_cohort(source, tmp_path, monkeypatch):
    packet = _packet(tmp_path, source)
    captured = []
    class Core:
        DATA_MANIFEST_SCHEMA_V2 = "fixture-data-v2"
        @staticmethod
        def validate_data_manifest(value):
            captured.append(value)
            return value
    import prismaquant.staged_lease as lease
    monkeypatch.setattr(lease, "sdk_submodule", lambda name: Core)
    result = exp.data_manifest(packet, tmp_path / "packet.json", b"packet", group_index=8)
    assert len(result["entries"]) == 66
    assert [phase["name"] for phase in result["read_plan"]["phases"]] == ["input-check", "group-08"]
    assert [entry["path"] for entry in result["entries"][2:]] == [
        row["path"] for row in packet["groups"][8]["entries"]]
    assert result["total_bytes"] == 6 + packet["checkpoint"]["bytes"] + packet["group_bytes"]
    assert captured == [result]
