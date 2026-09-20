"""Checkpoint artifact budget: count strided checkpoints in max_artifact_bytes.

`StreamedBoundaryArtifacts.write` enforces the artifact ceiling on ordinary
entries, but `write_adjoint_checkpoint` used to serialize beside the owner's
directory with no envelope reserved first. These regressions exercise the
actual checkpoint writer (tiny CPU tensors, never a stand-in): the legacy
blindness exhibit, pre-write refusal with zero files, mixed ordinary plus
checkpoint counting, rollback retain plus reclaim, idempotent commit rules,
published lifetime, unknown-metadata refusal, and the output descriptor.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from prismaquant.cost_streaming import StreamedBoundaryArtifacts
from prismaquant.joint_adjoint_checkpoints import (
    adjoint_space,
    write_adjoint_checkpoint,
)
from test_streamed_boundary_artifacts import _policy


def _owner(path, *, disk=1 << 24, published=False):
    owner = StreamedBoundaryArtifacts(_policy(path, disk=disk))
    owner.bind({"fixture": "checkpoint-budget"}, n_probes=2, published=published)
    return owner


def _session():
    return {"generation": "ab" * 16, "kind": "adjoint_checkpoint",
            "run_identity_sha256": "cd" * 32}


def _plane():
    return {(0, 0): torch.zeros(4, 4), (1, 0): torch.ones(2, 2)}


def _shared_adjoint():
    return {(0, 0): {"w": torch.zeros(2)}, (1, 0): {"w": torch.ones(3)}}


def _shared_pass():
    return {0: {"tag": "a"}}


def _write(owner, space, **overrides):
    kwargs = {"space": space, "boundary": 5, "session": _session(),
              "cotangents": _plane(), "shared_adjoint": _shared_adjoint(),
              "shared_pass": _shared_pass(), "owner": owner}
    kwargs.update(overrides)
    return write_adjoint_checkpoint(**kwargs)


def _space_files(space):
    return [path for path in Path(space).rglob("*") if path.is_file()]


def test_legacy_write_ignores_budget(tmp_path):
    """Exhibit: without an owner, checkpoints land outside every counter."""
    owner = _owner(tmp_path / "boundaries", disk=4096)
    space = adjoint_space(tmp_path / "out")
    record = write_adjoint_checkpoint(
        space, boundary=5, session=_session(), cotangents=_plane(),
        shared_adjoint=_shared_adjoint(), shared_pass=_shared_pass())
    assert record["cotangent_sha256"]
    assert owner.telemetry["live_checkpoint_bytes"] == 0
    assert owner.telemetry["checkpoint_reservations"] == 0
    assert _space_files(space)


def test_prewrite_refusal_creates_nothing(tmp_path):
    """An over-budget checkpoint refuses before its directory exists."""
    owner = _owner(tmp_path / "boundaries", disk=4096)
    space = adjoint_space(tmp_path / "out")
    with pytest.raises(RuntimeError, match="budget exceeded"):
        _write(owner, space)
    assert not (Path(space) / "checkpoints").exists()
    assert _space_files(space) == []
    assert owner.telemetry["checkpoint_refusals"] == 1
    assert owner.telemetry["live_checkpoint_bytes"] == 0
    assert owner._checkpoint_active is None


def test_mixed_ordinary_and_checkpoint_counting(tmp_path):
    """Ordinary live bytes plus checkpoint actuals share one ceiling."""
    owner = _owner(tmp_path / "boundaries")
    space = adjoint_space(tmp_path / "out")
    reference = owner.write(torch.zeros(8, 8), batch_index=0, boundary_index=0)
    ordinary = reference.file_bytes
    assert owner.telemetry["live_artifact_bytes"] == ordinary
    record = _write(owner, space)
    actual = (sum(row["file_bytes"] for row in record["activation_entries"])
              + sum(row["file_bytes"] for row in record["shared_state_entries"])
              + (Path(space) / "checkpoints" / "boundary-005"
                 / "checkpoint.json").stat().st_size)
    assert owner.telemetry["live_checkpoint_bytes"] == actual
    assert owner.telemetry["peak_checkpoint_bytes"] == actual
    remaining = owner.checkpoint_remaining_bytes()
    assert remaining == owner.config["max_artifact_bytes"] - ordinary - actual
    commitment = owner.checkpoint_commitment(record["cotangent_sha256"])
    assert commitment["actual_bytes"] == actual
    assert commitment["unused_bytes"] >= 0
    assert (commitment["envelope_bytes"] - commitment["actual_bytes"]
            == commitment["unused_bytes"])
    # A second copy no longer fits once the ceiling is nearly reached: leave
    # room for one ordinary entry envelope (256 tensor bytes + 65536 header)
    # plus the first checkpoint's actuals, which the ~330 KiB second
    # envelope still exceeds.
    owner2 = _owner(tmp_path / "b2",
                    disk=ordinary + actual + 70000, published=True)
    owner2.write(torch.zeros(8, 8), batch_index=0, boundary_index=0)
    with pytest.raises(RuntimeError, match="needs .* bytes, .* remain of"):
        write_adjoint_checkpoint(
            adjoint_space(tmp_path / "out2"), boundary=5, session=_session(),
            cotangents=_plane(), shared_adjoint=_shared_adjoint(),
            shared_pass=_shared_pass(), owner=owner2)


def test_rollback_retains_then_reclaims(tmp_path, monkeypatch):
    """Mid-write failure keeps bytes counted until an explicit reclaim."""
    import prismaquant.joint_adjoint_checkpoints as checkpoints

    owner = _owner(tmp_path / "boundaries")
    space = adjoint_space(tmp_path / "out")
    real_atomic = checkpoints.atomic_write_bytes
    calls = []

    def fail_first_pickle(path, payload):
        calls.append(str(path))
        if len(calls) == 1:
            raise RuntimeError("boom: simulated pickle failure")
        return real_atomic(path, payload)

    monkeypatch.setattr(checkpoints, "atomic_write_bytes", fail_first_pickle)
    with pytest.raises(RuntimeError, match="boom"):
        _write(owner, space)
    checkpoint_dir = Path(space) / "checkpoints" / "boundary-005"
    assert checkpoint_dir.is_dir()
    assert list((checkpoint_dir / "entries").glob("*.pt"))
    retained = [entry for entry in owner._checkpoint_reservations.values()
                if entry["state"] == "retained"]
    assert len(retained) == 1
    held = owner.checkpoint_remaining_bytes()
    assert held < owner.config["max_artifact_bytes"]
    result = owner.reclaim_checkpoint_artifact(space, 5)
    assert result["reclaimed"] is True
    assert not checkpoint_dir.exists()
    assert owner.checkpoint_remaining_bytes() == owner.config["max_artifact_bytes"]
    with pytest.raises(RuntimeError, match="no retained attempt"):
        owner.reclaim_checkpoint_artifact(space, 5)


def test_commit_is_idempotent_and_final(tmp_path):
    """Same receipt recommits cleanly; anything else refuses."""
    owner = _owner(tmp_path / "boundaries")
    space = adjoint_space(tmp_path / "out")
    record = _write(owner, space)
    commitment = owner.checkpoint_commitment(record["cotangent_sha256"])
    assert commitment is not None
    before = (owner.telemetry["live_checkpoint_bytes"],
              owner.telemetry["checkpoint_envelope_unused_bytes"])
    again = owner.commit_checkpoint_artifact(
        commitment["reservation"], record)
    assert again == commitment
    assert (owner.telemetry["live_checkpoint_bytes"],
            owner.telemetry["checkpoint_envelope_unused_bytes"]) == before
    tampered = dict(record, cotangent_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="different receipt"):
        owner.commit_checkpoint_artifact(commitment["reservation"], tampered)
    with pytest.raises(RuntimeError, match="unknown"):
        owner.commit_checkpoint_artifact(9999, record)
    assert (owner.telemetry["live_checkpoint_bytes"],
            owner.telemetry["checkpoint_envelope_unused_bytes"]) == before


def test_published_lifetime_survives_close(tmp_path):
    """Committed checkpoints (and published entries) outlive their owner."""
    policy = _policy(tmp_path / "boundaries")
    space = adjoint_space(tmp_path / "out")
    with StreamedBoundaryArtifacts(policy) as owner:
        owner.bind({"fixture": "checkpoint-budget"}, n_probes=2, published=True)
        ordinary = owner.write(torch.zeros(8, 8), batch_index=0, boundary_index=0)
        record = _write(owner, space)
        live = owner.telemetry["live_checkpoint_bytes"]
        assert live > 0
        receipt = owner.receipt()
        assert receipt["telemetry"]["live_checkpoint_bytes"] == live
        assert receipt["status"] == "running"
    checkpoint_dir = Path(space) / "checkpoints" / "boundary-005"
    assert (checkpoint_dir / "checkpoint.json").is_file()
    assert list((checkpoint_dir / "entries").glob("*.pt"))
    assert list((checkpoint_dir / "entries").glob("*.pkl"))
    assert Path(ordinary.path).is_file()


def test_unknown_shared_state_refuses_before_dumps(tmp_path, monkeypatch):
    """Opaque metadata refuses with zero files and zero serializations."""
    import pickle

    owner = _owner(tmp_path / "boundaries")
    space = adjoint_space(tmp_path / "out")
    dumps = []
    real_dumps = pickle.dumps

    def spy_dumps(state, *args, **kwargs):
        dumps.append(1)
        return real_dumps(state, *args, **kwargs)

    monkeypatch.setattr(pickle, "dumps", spy_dumps)
    with pytest.raises(RuntimeError, match="unaccountable shared state"):
        write_adjoint_checkpoint(
            space, boundary=5, session=_session(), cotangents=_plane(),
            shared_adjoint={(0, 0): object()}, shared_pass=_shared_pass(),
            owner=owner)
    assert dumps == []
    assert not (Path(space) / "checkpoints").exists()


def test_oversized_estimate_refuses_before_dumps(tmp_path, monkeypatch):
    """An estimate over remaining budget refuses before serializing."""
    import pickle

    owner = _owner(tmp_path / "boundaries", disk=4096)
    space = adjoint_space(tmp_path / "out")
    dumps = []
    real_dumps = pickle.dumps

    def spy_dumps(state, *args, **kwargs):
        dumps.append(1)
        return real_dumps(state, *args, **kwargs)

    monkeypatch.setattr(pickle, "dumps", spy_dumps)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        _write(owner, space)
    assert dumps == []
    assert not (Path(space) / "checkpoints").exists()


def test_output_descriptor_is_application_scope(tmp_path):
    """The PB732 hook reports counts and disclaims PB reservation."""
    owner = _owner(tmp_path / "boundaries")
    space = adjoint_space(tmp_path / "out")
    before = owner.checkpoint_output_descriptor()
    assert before["live_checkpoint_bytes"] == 0
    assert before["reserved_bytes"] == 0
    record = _write(owner, space)
    descriptor = owner.checkpoint_output_descriptor()
    assert descriptor["schema"] == "prismaquant.boundary_artifact_output.v1"
    assert descriptor["unit"] == "bytes"
    assert descriptor["ceiling_bytes"] == owner.config["max_artifact_bytes"]
    assert descriptor["live_checkpoint_bytes"] == owner.telemetry["live_checkpoint_bytes"]
    assert descriptor["live_ordinary_bytes"] == owner.telemetry["live_artifact_bytes"]
    assert "not a PrismaBuild" in descriptor["note"]
    assert owner.checkpoint_commitment(record["cotangent_sha256"]) is not None


def test_close_refuses_active_reservation(tmp_path):
    """Closing with a live reservation fails closed like a live window."""
    owner = _owner(tmp_path / "boundaries")
    plan = {"files": [{"name": "f", "envelope_bytes": 100}],
            "manifest_bytes": 100, "temp_overlap_bytes": 100,
            "envelope_bytes": 300}
    reservation = owner.reserve_checkpoint_artifact(
        label="fixture", envelope_bytes=300, file_plan=plan,
        checkpoint_dir=tmp_path / "checkpoints" / "boundary-005")
    with pytest.raises(RuntimeError, match="active checkpoint reservation"):
        owner.__exit__(None, None, None)
    owner.cancel_checkpoint_artifact(reservation)
    owner.__exit__(None, None, None)


def test_reentrant_reservation_refuses(tmp_path):
    """The single-owner writer never holds two envelopes at once."""
    owner = _owner(tmp_path / "boundaries")
    plan = {"files": [{"name": "f", "envelope_bytes": 100}],
            "manifest_bytes": 100, "temp_overlap_bytes": 100,
            "envelope_bytes": 300}
    first = owner.reserve_checkpoint_artifact(
        label="first", envelope_bytes=300, file_plan=plan,
        checkpoint_dir=tmp_path / "c1")
    with pytest.raises(RuntimeError, match="already active"):
        owner.reserve_checkpoint_artifact(
            label="second", envelope_bytes=300, file_plan=plan,
            checkpoint_dir=tmp_path / "c2")
    owner.cancel_checkpoint_artifact(first)
    second = owner.reserve_checkpoint_artifact(
        label="second", envelope_bytes=300, file_plan=plan,
        checkpoint_dir=tmp_path / "c2")
    owner.cancel_checkpoint_artifact(second)
    assert owner.checkpoint_remaining_bytes() == owner.config["max_artifact_bytes"]
