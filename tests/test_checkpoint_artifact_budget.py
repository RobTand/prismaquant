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


def _owner(path, *, disk=1 << 24, cap=1 << 22, published=False, check_memory=None):
    # Resident headroom covers per-entry transient serialization holds; the
    # artifact ceiling under test stays `disk`.
    owner = StreamedBoundaryArtifacts(
        _policy(path, cap=cap, disk=disk))
    owner.bind({"fixture": "checkpoint-budget"}, n_probes=2,
               check_memory=check_memory, published=published)
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

    def fail_manifest(path, payload):
        raise RuntimeError("boom: simulated manifest failure")

    monkeypatch.setattr(checkpoints, "atomic_write_bytes", fail_manifest)
    with pytest.raises(RuntimeError, match="boom"):
        _write(owner, space)
    checkpoint_dir = Path(space) / "checkpoints" / "boundary-005"
    assert checkpoint_dir.is_dir()
    # Every entry landed before the manifest failed: a true partial.
    assert list((checkpoint_dir / "entries").glob("*.pt"))
    assert list((checkpoint_dir / "entries").glob("*.pkl"))
    assert not (checkpoint_dir / "checkpoint.json").exists()
    retained = [entry for entry in owner._checkpoint_reservations.values()
                if entry["state"] == "retained"]
    assert len(retained) == 1
    held = owner.checkpoint_remaining_bytes()
    assert held < owner.config["max_artifact_bytes"]
    result = owner.reclaim_checkpoint_artifact(space, 5)
    assert result["reclaimed"] is True
    assert result["disposition"] == "deleted"
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
    real_dump = pickle.dump

    def spy_dump(obj, file, *args, **kwargs):
        dumps.append(1)
        return real_dump(obj, file, *args, **kwargs)

    monkeypatch.setattr(pickle, "dump", spy_dump)
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
    real_dump = pickle.dump

    def spy_dump(obj, file, *args, **kwargs):
        dumps.append(1)
        return real_dump(obj, file, *args, **kwargs)

    monkeypatch.setattr(pickle, "dump", spy_dump)
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
    directory = tmp_path / "checkpoints" / "boundary-005"
    plan = {"files": [{"name": "f", "path": str(directory / "f.pt"),
                       "envelope_bytes": 100}],
            "manifest_bytes": 100, "temp_overlap_bytes": 100,
            "envelope_bytes": 300}
    reservation = owner.reserve_checkpoint_artifact(
        label="fixture", envelope_bytes=300, file_plan=plan,
        checkpoint_dir=directory)
    with pytest.raises(RuntimeError, match="active checkpoint reservation"):
        owner.__exit__(None, None, None)
    owner.cancel_checkpoint_artifact(reservation)
    owner.__exit__(None, None, None)


def test_reentrant_reservation_refuses(tmp_path):
    """The single-owner writer never holds two envelopes at once."""
    owner = _owner(tmp_path / "boundaries")
    directory = tmp_path / "c1"
    plan = {"files": [{"name": "f", "path": str(directory / "f.pt"),
                       "envelope_bytes": 100}],
            "manifest_bytes": 100, "temp_overlap_bytes": 100,
            "envelope_bytes": 300}
    first = owner.reserve_checkpoint_artifact(
        label="first", envelope_bytes=300, file_plan=plan,
        checkpoint_dir=directory)
    plan2 = {"files": [{"name": "f", "path": str(tmp_path / "c2" / "f.pt"),
                        "envelope_bytes": 100}],
             "manifest_bytes": 100, "temp_overlap_bytes": 100,
             "envelope_bytes": 300}
    with pytest.raises(RuntimeError, match="already active"):
        owner.reserve_checkpoint_artifact(
            label="second", envelope_bytes=300, file_plan=plan2,
            checkpoint_dir=tmp_path / "c2")
    owner.cancel_checkpoint_artifact(first)
    second = owner.reserve_checkpoint_artifact(
        label="second", envelope_bytes=300, file_plan=plan2,
        checkpoint_dir=tmp_path / "c2")
    owner.cancel_checkpoint_artifact(second)
    assert owner.checkpoint_remaining_bytes() == owner.config["max_artifact_bytes"]


def _big_plane():
    return {(0, 0): torch.zeros(512, 512)}


def _checkpoint_actual(owner, space, record):
    return (sum(row["file_bytes"] for row in record["activation_entries"])
            + sum(row["file_bytes"] for row in record["shared_state_entries"])
            + (Path(space) / "checkpoints" / "boundary-005"
               / "checkpoint.json").stat().st_size)


def test_checkpoint_then_ordinary_share_ceiling(tmp_path):
    """A committed checkpoint narrows later ordinary writes (aggregate)."""
    owner = _owner(tmp_path / "boundaries", disk=2600000)
    space = adjoint_space(tmp_path / "out")
    record = write_adjoint_checkpoint(
        space, boundary=5, session=_session(), cotangents=_big_plane(),
        shared_adjoint=_shared_adjoint(), shared_pass=_shared_pass(),
        owner=owner)
    actual = _checkpoint_actual(owner, space, record)
    assert owner.telemetry["live_checkpoint_bytes"] == actual
    # 700x700 needs ~2.03 MiB envelope: fits live-only headroom, but the
    # committed checkpoint bytes push it over the shared ceiling.
    with pytest.raises(RuntimeError, match="budget exceeded"):
        owner.write(torch.zeros(700, 700), batch_index=0, boundary_index=0)
    assert owner.telemetry["live_artifact_bytes"] == 0
    small = owner.write(torch.zeros(64, 64), batch_index=0, boundary_index=0)
    assert owner.telemetry["live_artifact_bytes"] == small.file_bytes
    assert owner.checkpoint_remaining_bytes() == (
        owner.config["max_artifact_bytes"] - actual - small.file_bytes)


def test_retained_partial_blocks_ordinary_until_reclaimed(tmp_path, monkeypatch):
    """A retained envelope counts against ordinary writes until reclaimed."""
    import prismaquant.joint_adjoint_checkpoints as checkpoints

    owner = _owner(tmp_path / "boundaries", disk=2600000)
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
        write_adjoint_checkpoint(
            space, boundary=5, session=_session(), cotangents=_big_plane(),
            shared_adjoint=_shared_adjoint(), shared_pass=_shared_pass(),
            owner=owner)
    assert calls, "the failure never reached the pickle writer"
    # The 256 KiB ordinary envelope fits live-only headroom but not the
    # retained ~2.26 MiB checkpoint envelope beside it.
    with pytest.raises(RuntimeError, match="budget exceeded"):
        owner.write(torch.zeros(256, 256), batch_index=0, boundary_index=0)
    assert owner.telemetry["live_artifact_bytes"] == 0
    result = owner.reclaim_checkpoint_artifact(space, 5)
    assert result["disposition"] == "deleted"
    admitted = owner.write(torch.zeros(256, 256), batch_index=0, boundary_index=0)
    assert owner.telemetry["live_artifact_bytes"] == admitted.file_bytes


def test_cotangent_rollover_overlap_counts_checkpoints(tmp_path):
    """Rollover peak (old plus new) is admitted against the shared ceiling."""
    owner = _owner(tmp_path / "boundaries", disk=2600000)
    space = adjoint_space(tmp_path / "out")
    record = write_adjoint_checkpoint(
        space, boundary=5, session=_session(), cotangents=_big_plane(),
        shared_adjoint=_shared_adjoint(), shared_pass=_shared_pass(),
        owner=owner)
    actual = _checkpoint_actual(owner, space, record)
    old = owner.write(torch.zeros(256, 256), batch_index=0, boundary_index=1,
                      probe_index=0)
    # Same-shape rollover: old plus new plus checkpoint still fits.
    rolled = owner.write(torch.zeros(256, 256), batch_index=0, boundary_index=0,
                         probe_index=0, previous=old)
    assert owner.telemetry["live_artifact_bytes"] == rolled.file_bytes
    # A 700x700 rollover fits live-only headroom but overlaps the committed
    # checkpoint past the ceiling.
    old2 = owner.write(torch.zeros(256, 256), batch_index=1, boundary_index=1,
                       probe_index=0)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        owner.write(torch.zeros(700, 700), batch_index=1, boundary_index=0,
                    probe_index=0, previous=old2)
    assert owner.telemetry["live_artifact_bytes"] == (
        rolled.file_bytes + old2.file_bytes)
    assert actual == owner.telemetry["live_checkpoint_bytes"]


def test_baseline_legacy_bypass_both_directions(tmp_path):
    """Exhibit on the existing API: unwatched writes bypass every counter.

    Passes on unmodified main and on the fixed tree (the legacy path is
    preserved byte for byte); reported separately from missing-API RED.
    """
    owner = _owner(tmp_path / "boundaries", disk=100000)
    space = adjoint_space(tmp_path / "out")
    owner.write(torch.zeros(8, 8), batch_index=0, boundary_index=0)
    record = write_adjoint_checkpoint(
        space, boundary=5, session=_session(),
        cotangents={(0, 0): torch.zeros(256, 256)},
        shared_adjoint={}, shared_pass={})
    true_usage = (sum(path.stat().st_size for path in _space_files(space))
                  + sum(path.stat().st_size
                        for path in (Path(owner.directory) / "entries").glob("*")
                        if path.is_file()))
    assert true_usage > owner.config["max_artifact_bytes"]
    assert owner.telemetry.get("live_checkpoint_bytes", 0) == 0
    # ...and a later ordinary write is still admitted on live-only headroom
    # even though true on-disk usage already exceeds the ceiling.
    owner.write(torch.zeros(8, 8), batch_index=1, boundary_index=0)


def test_tensor_entry_receives_admitted_file_limit(tmp_path, monkeypatch):
    """The exact writer gets the reservation's per-file envelope."""
    import prismaquant.perturbed_x_cache as perturbed

    owner = _owner(tmp_path / "boundaries")
    space = adjoint_space(tmp_path / "out")
    seen = {}
    real_exact = perturbed.write_exact_activation_cache_entry

    def spy_exact(cache_dir, name, inputs, *, identity, max_tensor_bytes,
                  max_file_bytes, **kwargs):
        seen[name] = (max_tensor_bytes, max_file_bytes)
        return real_exact(cache_dir, name, inputs, identity=identity,
                          max_tensor_bytes=max_tensor_bytes,
                          max_file_bytes=max_file_bytes, **kwargs)

    monkeypatch.setattr(perturbed, "write_exact_activation_cache_entry", spy_exact)
    _write(owner, space)
    for (probe, batch), tensor in _plane().items():
        name = f"cotangent-{probe}-{batch}"
        nbytes = tensor.numel() * tensor.element_size()
        assert seen[name] == (nbytes, nbytes + 65536)


def test_oversized_shared_payload_refuses_pre_write(tmp_path, monkeypatch):
    """A payload over its admitted envelope never publishes its file."""
    import pickle

    owner = _owner(tmp_path / "boundaries")
    space = adjoint_space(tmp_path / "out")
    real_dump = pickle.dump

    def fat_dump(obj, file, *args, **kwargs):
        real_dump(obj, file, *args, **kwargs)
        if isinstance(obj, dict) and obj.get("tag") == "a":
            file.write(b"\x00" * (1 << 20))
            file.flush()

    monkeypatch.setattr(pickle, "dump", fat_dump)
    with pytest.raises(RuntimeError, match="admitted envelope"):
        _write(owner, space)
    entries = Path(space) / "checkpoints" / "boundary-005" / "entries"
    # The oversized temp is unlinked, never renamed: no trace publishes.
    assert [path.name for path in entries.glob("shared-pass-*")] == []
    assert list(entries.glob("shared-adjoint-*.pkl"))
    retained = [entry for entry in owner._checkpoint_reservations.values()
                if entry["state"] == "retained"]
    assert len(retained) == 1
    result = owner.reclaim_checkpoint_artifact(space, 5)
    assert result["reclaimed"] is True
    assert not (Path(space) / "checkpoints" / "boundary-005").exists()


def _reserve_for_record(owner, space, record):
    """Reserve the record's own actuals as its envelope for tamper tests."""
    directory = Path(space) / "checkpoints" / "boundary-005"
    files = [{"name": row["name"], "path": row["path"],
              "envelope_bytes": row["file_bytes"]}
             for row in record["activation_entries"] + record["shared_state_entries"]]
    manifest_bytes = (directory / "checkpoint.json").stat().st_size
    envelope = sum(row["envelope_bytes"] for row in files) + manifest_bytes
    return owner.reserve_checkpoint_artifact(
        label="tamper", envelope_bytes=envelope,
        file_plan={"files": files, "manifest_bytes": manifest_bytes,
                   "temp_overlap_bytes": 0, "envelope_bytes": envelope},
        checkpoint_dir=directory)


def test_commit_rejects_unplanned_receipt_rows(tmp_path):
    """Extra, missing, duplicate, and rerouted rows refuse before any stat."""
    owner = _owner(tmp_path / "boundaries")
    space = adjoint_space(tmp_path / "out")
    record = _write(owner, space)

    def attempt(mutated):
        reservation = _reserve_for_record(owner, space, record)
        with pytest.raises(RuntimeError, match="reserved plan|reserved path"):
            owner.commit_checkpoint_artifact(reservation, mutated)

    extra = json.loads(json.dumps(record))
    extra["activation_entries"] = extra["activation_entries"] + [
        dict(extra["activation_entries"][0], name="zzz-extra")]
    attempt(extra)
    missing = json.loads(json.dumps(record))
    missing["shared_state_entries"] = missing["shared_state_entries"][:-1]
    attempt(missing)
    duplicate = json.loads(json.dumps(record))
    duplicate["activation_entries"] = (
        duplicate["activation_entries"] + duplicate["activation_entries"][:1])
    attempt(duplicate)
    rerouted = json.loads(json.dumps(record))
    other = rerouted["activation_entries"][-1]["path"]
    rerouted["activation_entries"][0]["path"] = other
    attempt(rerouted)


def test_commit_rejects_per_file_over_envelope(tmp_path):
    """A receipt row over its admitted envelope refuses even with files present."""
    owner = _owner(tmp_path / "boundaries")
    space = adjoint_space(tmp_path / "out")
    record = _write(owner, space)
    inflated = json.loads(json.dumps(record))
    inflated["activation_entries"][0]["file_bytes"] += 1
    reservation = _reserve_for_record(owner, space, record)
    with pytest.raises(RuntimeError, match="reserved envelope"):
        owner.commit_checkpoint_artifact(reservation, inflated)


def test_hook_failure_after_commit_stays_truthful(tmp_path):
    """An error after the receipt commits propagates; state stays committed."""
    def hook(label):
        if "publication" in label:
            raise RuntimeError("hook boom: UMA floor")

    owner = _owner(tmp_path / "boundaries", check_memory=hook)
    space = adjoint_space(tmp_path / "out")
    with pytest.raises(RuntimeError, match="hook boom"):
        _write(owner, space)
    assert len(owner._checkpoint_reservations) == 1
    reservation = next(iter(owner._checkpoint_reservations))
    assert owner.checkpoint_reservation_state(reservation) == "committed"
    assert owner.telemetry["live_checkpoint_bytes"] > 0
    assert (Path(space) / "checkpoints" / "boundary-005" / "checkpoint.json").is_file()


def test_close_disposes_retained_attempt(tmp_path, monkeypatch):
    """Owner close reclaims never-receipted partials and keeps the error."""
    import prismaquant.joint_adjoint_checkpoints as checkpoints

    policy = _policy(tmp_path / "boundaries")
    space = adjoint_space(tmp_path / "out")
    real_atomic = checkpoints.atomic_write_bytes

    def fail_manifest(path, payload):
        raise RuntimeError("boom: simulated manifest failure")

    monkeypatch.setattr(checkpoints, "atomic_write_bytes", fail_manifest)
    owner = StreamedBoundaryArtifacts(policy)
    owner.bind({"fixture": "checkpoint-budget"}, n_probes=2)
    with pytest.raises(RuntimeError, match="boom"):
        with owner:
            _write(owner, space)
    assert not (Path(space) / "checkpoints" / "boundary-005").exists()


def test_pickle_serializes_each_view_backing(tmp_path):
    """Pin measured torch behavior: views do not share serialized backing.

    Two 2 MiB views sharing one 4 MiB backing dump ~8 MB together, the same
    as separately, so the estimator counts per-leaf backing ownership with
    no cross-leaf deduplication. The estimate must cover the real bytes.
    """
    import pickle

    from prismaquant.joint_adjoint_checkpoints import _shared_state_envelope_estimate

    base = torch.zeros(1_000_000)
    first, second = base[:500_000], base[500_000:]
    backing = base.untyped_storage().nbytes()
    both = len(pickle.dumps([first, second], protocol=pickle.HIGHEST_PROTOCOL))
    assert both > 1.5 * backing
    estimate = _shared_state_envelope_estimate({"a": first, "b": second})
    assert estimate >= both


def test_meta_tensor_refuses_pre_write(tmp_path):
    """Unsizable tensors refuse before reservation, with zero files."""
    owner = _owner(tmp_path / "boundaries")
    space = adjoint_space(tmp_path / "out")
    meta = torch.zeros(4, 4, device="meta")
    with pytest.raises(TypeError, match="materialized strided tensors"):
        write_adjoint_checkpoint(
            space, boundary=5, session=_session(),
            cotangents={(0, 0): meta},
            shared_adjoint=_shared_adjoint(), shared_pass=_shared_pass(),
            owner=owner)
    assert not (Path(space) / "checkpoints").exists()
    assert owner._checkpoint_reservations == {}
