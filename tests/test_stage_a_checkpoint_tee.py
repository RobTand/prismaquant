"""Stage A writes each checkpoint while it rolls the plane (RobTand/prismaquant#1002).

Before this, a checkpoint layer rolled its plane into the owner's rolling
entries and then read the whole plane back to copy it into the checkpoint.
On R12 that read-back was most of the checkpoint: 289 of 328 s at
checkpoint 40 went to waiting for the rolling groups to be staged again.
The read-back windows also replaced the read-ahead the roll had asked for
the next layer's first window, so that window opened cold.

Now the checkpoint is reserved before the pass, each cotangent is written
into it as the roll hands it to the owner, and the shared states and the
manifest follow the pass. These tests hold it to four claims:

* The checkpoint is the one the read-back writer publishes, byte for byte.
* Sealing it reads nothing and asks nothing to be staged, so the request
  the roll made for the next layer's first window is the one standing.
* It writes only what it planned, and shared states, known only after the
  pass, are admitted against the artifact budget before any is written.
* A run that dies with a checkpoint open retains its envelope, and only the
  owner's close disposes the directory.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest
import torch

from prismaquant import joint_cost_stage_a as stage_a
from prismaquant.cost_streaming import StreamedBoundaryArtifacts
from prismaquant.joint_adjoint_checkpoints import (
    checkpoint_directory,
    write_adjoint_checkpoint,
)

from test_checkpoint_artifact_budget import _owner, _session
from test_stage_a_chain_resume import _Interrupted, _at, _run


@pytest.fixture(autouse=True)
def _offline_tier_policy():
    from prismaquant.staged_tier_policy import deactivate_staged_tier_policy_for_tests
    deactivate_staged_tier_policy_for_tests()
    yield
    deactivate_staged_tier_policy_for_tests()


def _files(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _spec(tensor):
    return (tensor.numel() * tensor.element_size(), list(tensor.shape), str(tensor.dtype))


def _plane(probes=2, batches=5):
    return {(probe, batch): (torch.arange(1024, dtype=torch.float32) * (probe + 1)
                             + batch).reshape(4, 256)
            for probe in range(probes) for batch in range(batches)}


def _open(owner, space, tensors, shared_adjoint_keys, shared_pass_keys, boundary=3):
    from prismaquant.joint_adjoint_checkpoints import open_adjoint_checkpoint

    return open_adjoint_checkpoint(
        space, boundary=boundary, session=_session(),
        specs={key: _spec(tensor) for key, tensor in tensors.items()},
        shared_adjoint_keys=list(shared_adjoint_keys),
        shared_pass_keys=list(shared_pass_keys), owner=owner)


def test_a_teed_checkpoint_is_the_read_back_writers_checkpoint(tmp_path):
    """Same tensors, session and slots: the same files, record and bytes."""
    tensors = _plane()
    shared_adjoint = {key: {"w": [float(key[0]), float(key[1])], "n": 10 * key[0] + key[1]}
                      for key in tensors}
    shared_pass = {batch: {"tag": f"pass-{batch}"} for batch in range(5)}
    space = tmp_path / "space"

    source = _owner(tmp_path / "source", n_probes=2)
    references = {key: source.write(tensor, probe_index=key[0], batch_index=key[1],
                                    boundary_index=3)
                  for key, tensor in tensors.items()}
    copied = write_adjoint_checkpoint(
        space, boundary=3, session=_session(), cotangents=references,
        shared_adjoint=shared_adjoint, shared_pass=shared_pass, owner=source)
    first = _files(checkpoint_directory(space, 3))
    shutil.rmtree(checkpoint_directory(space, 3))

    owner = _owner(tmp_path / "teed", n_probes=2)
    reads = []
    owner.prefetch = lambda references: reads.append(references)
    attempt = _open(owner, space, tensors, shared_adjoint, shared_pass)
    # The roll's order is not the manifest's; the record sorts either way.
    for key in sorted(tensors, reverse=True):
        attempt.write_activation(*key, tensors[key])
    attempt.write_shared_states(shared_adjoint, shared_pass)
    teed = attempt.seal()

    assert teed == copied
    assert _files(checkpoint_directory(space, 3)) == first
    assert len(first) == 10 + 10 + 5 + 1
    assert reads == []
    assert (owner.checkpoint_commitment(teed["cotangent_sha256"])["actual_bytes"]
            == source.checkpoint_commitment(copied["cotangent_sha256"])["actual_bytes"])
    assert owner.telemetry["resident_tensor_bytes"] == 0


def test_the_tee_writes_only_what_it_planned(tmp_path):
    tensors = _plane(probes=2, batches=2)
    owner = _owner(tmp_path / "entries", n_probes=2)
    attempt = _open(owner, tmp_path / "space", tensors, tensors, range(2))
    tensor = tensors[0, 0]

    with pytest.raises(RuntimeError, match="planned no cotangent at probe 2, batch 0"):
        attempt.write_activation(2, 0, tensor)
    with pytest.raises(RuntimeError, match=r"is \[4, 255\] torch.float32, planned \[4, 256\]"):
        attempt.write_activation(0, 0, tensor[:, :255].contiguous())
    with pytest.raises(RuntimeError, match="torch.float64, planned"):
        attempt.write_activation(0, 0, tensor.double())
    for key in sorted(tensors):
        attempt.write_activation(*key, tensors[key])
    with pytest.raises(RuntimeError, match="has not written its shared states"):
        attempt.seal()
    with pytest.raises(RuntimeError, match="already wrote cotangent-0-0"):
        attempt.write_activation(0, 0, tensor)
    with pytest.raises(RuntimeError, match="differ from the set it planned"):
        attempt.write_shared_states({(0, 0): {}}, {0: {}, 1: {}})
    assert owner.checkpoint_reservation_state(attempt.reservation) == "active"
    attempt.abandon()
    assert owner.checkpoint_reservation_state(attempt.reservation) == "retained"
    attempt.abandon()


def test_a_seal_refuses_a_missing_cotangent(tmp_path):
    tensors = _plane(probes=1, batches=3)
    owner = _owner(tmp_path / "entries", n_probes=1)
    attempt = _open(owner, tmp_path / "space", tensors, tensors, range(3))
    attempt.write_activation(0, 1, tensors[0, 1])
    attempt.write_shared_states({key: {} for key in tensors}, {b: {} for b in range(3)})
    with pytest.raises(RuntimeError, match=r"missing 2 planned cotangents, first \[\(0, 0\), \(0, 2\)\]"):
        attempt.seal()
    assert not (checkpoint_directory(tmp_path / "space", 3) / "checkpoint.json").exists()


def test_shared_states_are_admitted_before_they_are_written(tmp_path):
    """The pass decides the shared states' sizes; the budget still comes first."""
    tensors = _plane(probes=1, batches=1)
    space = tmp_path / "space"
    owner = _owner(tmp_path / "entries", n_probes=1, disk=200_000)
    attempt = _open(owner, space, tensors, tensors, [0])
    envelope = owner._checkpoint_reservations[attempt.reservation]["envelope_bytes"]
    remaining = owner.checkpoint_remaining_bytes()
    attempt.write_activation(0, 0, tensors[0, 0])
    large = {(0, 0): {"w": "x" * remaining}}
    with pytest.raises(RuntimeError, match="needs [0-9]+ more bytes"):
        attempt.write_shared_states(large, {0: {}})
    entries = checkpoint_directory(space, 3) / "entries"
    assert sorted(path.name for path in entries.iterdir()) == ["cotangent-0-0.pt"]
    assert owner._checkpoint_reservations[attempt.reservation]["envelope_bytes"] == envelope
    assert owner.telemetry["checkpoint_refusals"] == 1
    too_large = {(0, 0): {"w": "x" * 200_001}}
    with pytest.raises(RuntimeError, match="the ceiling is 200000"):
        attempt.write_shared_states(too_large, {0: {}})
    attempt.abandon()

    owner = _owner(tmp_path / "entries-2", n_probes=1)
    attempt = _open(owner, tmp_path / "space-2", tensors, tensors, [0])
    envelope = owner._checkpoint_reservations[attempt.reservation]["envelope_bytes"]
    attempt.write_activation(0, 0, tensors[0, 0])
    state = {(0, 0): {"w": "y" * 5000}}
    attempt.write_shared_states(state, {0: {"tag": "a"}})
    reservation = owner._checkpoint_reservations[attempt.reservation]
    shared = [row for row in reservation["files"] if row["name"].startswith("shared-")]
    assert [row["name"] for row in shared] == ["shared-adjoint-0-0", "shared-pass-0"]
    assert reservation["envelope_bytes"] == envelope + sum(row["envelope_bytes"] for row in shared)
    record = attempt.seal()
    assert [row["name"] for row in record["shared_state_entries"]] == [
        "shared-adjoint-0-0", "shared-pass-0"]
    assert owner.checkpoint_reservation_state(attempt.reservation) == "committed"


def test_an_extension_is_admitted_like_a_reservation(tmp_path):
    owner = _owner(tmp_path / "entries", disk=10_000)
    directory = tmp_path / "space" / "checkpoints" / "boundary-003"
    plan = {"files": [{"name": "a", "path": str(directory / "entries" / "a.pt"),
                       "envelope_bytes": 1000}],
            "manifest_bytes": 500, "temp_overlap_bytes": 1000, "envelope_bytes": 2500}
    reservation = owner.reserve_checkpoint_artifact(
        label="fixture", envelope_bytes=2500, file_plan=plan, checkpoint_dir=directory)

    def row(name, envelope=100, path=None):
        return {"name": name, "path": path or str(directory / "entries" / f"{name}.pkl"),
                "envelope_bytes": envelope}

    for files, temp, reason in (
            ([row("a")], 1000, "repeats a planned file"),
            ([row("b"), row("b")], 1000, "repeats a planned file"),
            ([row("b", path=str(tmp_path / "b.pkl"))], 1000, "escapes"),
            ([row("b", envelope=0)], 1000, "malformed"),
            ([], 1000, "malformed"),
            ([row("b")], 999, "may only grow")):
        with pytest.raises(RuntimeError, match=reason):
            owner.extend_checkpoint_artifact(reservation, files=files,
                                             temp_overlap_bytes=temp)
    remaining = owner.checkpoint_remaining_bytes()
    with pytest.raises(RuntimeError, match=f"needs {remaining + 1} more bytes"):
        owner.extend_checkpoint_artifact(
            reservation, files=[row("b", envelope=remaining + 1)], temp_overlap_bytes=1000)
    assert owner.telemetry["checkpoint_refusals"] == 1
    assert owner.checkpoint_remaining_bytes() == remaining
    total = owner.extend_checkpoint_artifact(
        reservation, files=[row("b", envelope=300)], temp_overlap_bytes=1200)
    assert total == 2500 + 300 + 200
    assert owner.checkpoint_remaining_bytes() == remaining - 500
    owner.abandon_checkpoint_artifact(reservation)
    with pytest.raises(RuntimeError, match="not active"):
        owner.extend_checkpoint_artifact(reservation, files=[row("c")],
                                         temp_overlap_bytes=1200)


def test_a_checkpoint_reads_nothing_back_and_keeps_the_next_window_asked_for(
        tmp_path, monkeypatch):
    """The acceptance of #1002: the step after a checkpoint starts warm.

    Every read and every staging request is recorded against the roll and
    seal it happens in. Sealing a checkpoint reads nothing and asks for
    nothing, so the last request standing when the next layer's roll starts
    is the one the checkpoint layer's roll made for that roll's first
    window. The read-back writer failed both: it read the plane back in
    windows and each window replaced the request with its own.
    """
    events = []
    prefetch = StreamedBoundaryArtifacts.prefetch
    ahead = StreamedBoundaryArtifacts.stage_produced_reads_ahead
    seal = stage_a.write_checkpoint_with_snapshot
    roll = stage_a.render_free_layer_roll

    def reading(self, references):
        events.append(("read", tuple(sorted(ref.name for ref in references))))
        return prefetch(self, references)

    def asking(self, references):
        events.append(("ahead", tuple(sorted(ref.name for ref in references))))
        return ahead(self, references)

    def sealing(*args, **kwargs):
        events.append(("seal", kwargs["boundary"], "start"))
        try:
            return seal(*args, **kwargs)
        finally:
            events.append(("seal", kwargs["boundary"], "end"))

    def rolling(*args, **kwargs):
        events.append(("roll", kwargs["layer"], "start"))
        try:
            return roll(*args, **kwargs)
        finally:
            events.append(("roll", kwargs["layer"], "end"))

    monkeypatch.setattr(StreamedBoundaryArtifacts, "prefetch", reading)
    monkeypatch.setattr(StreamedBoundaryArtifacts, "stage_produced_reads_ahead", asking)
    monkeypatch.setattr(stage_a, "write_checkpoint_with_snapshot", sealing)
    monkeypatch.setattr(stage_a, "render_free_layer_roll", rolling)
    receipt = _run(tmp_path / "run", monkeypatch, stride=2)
    assert [c["boundary"] for c in receipt["checkpoints"]] == [5, 4, 2]
    window = 2  # the fixture policy's prefetch_batches

    for boundary in (5, 4, 2):
        start = events.index(("seal", boundary, "start"))
        end = events.index(("seal", boundary, "end"))
        assert [e for e in events[start:end] if e[0] in ("read", "ahead")] == [], (
            f"sealing checkpoint {boundary} read or staged")
    for boundary in (4, 2):
        rolled = events.index(("roll", boundary, "end"))
        following = events.index(("roll", boundary - 1, "start"))
        between = [e for e in events[rolled:following] if e[0] in ("read", "ahead")]
        assert between == [], f"between rolls {boundary} and {boundary - 1}: {between}"
        standing = [e for e in events[:rolled] if e[0] == "ahead"][-1]
        first = tuple(sorted(
            [f"boundary-{batch}-{boundary - 1}-at-{boundary - 1}" for batch in range(window)]
            + [f"cotangent-0-{batch}-at-{boundary}" for batch in range(window)]))
        assert standing == ("ahead", first)
        assert ("read", first) == next(e for e in events[following:] if e[0] == "read")


def test_a_run_that_dies_mid_roll_retains_its_open_checkpoint(tmp_path, monkeypatch):
    """Abandoned, never cancelled, and disposed only by the owner's close."""
    calls = []
    abandon = StreamedBoundaryArtifacts.abandon_checkpoint_artifact
    cancel = StreamedBoundaryArtifacts.cancel_checkpoint_artifact
    dispose = StreamedBoundaryArtifacts._dispose_retained_entry

    def abandoning(self, reservation):
        calls.append(("abandon", self._checkpoint_reservations[reservation]["dir"]))
        return abandon(self, reservation)

    def cancelling(self, reservation):
        calls.append(("cancel", reservation))
        return cancel(self, reservation)

    def disposing(self, reservation, entry):
        entries = Path(entry["dir"]) / "entries"
        calls.append(("dispose", entry["dir"],
                      sorted(path.name for path in entries.iterdir())))
        return dispose(self, reservation, entry)

    monkeypatch.setattr(StreamedBoundaryArtifacts, "abandon_checkpoint_artifact", abandoning)
    monkeypatch.setattr(StreamedBoundaryArtifacts, "cancel_checkpoint_artifact", cancelling)
    monkeypatch.setattr(StreamedBoundaryArtifacts, "_dispose_retained_entry", disposing)
    root = tmp_path / "run"
    with pytest.raises(_Interrupted):
        _run(root, monkeypatch, stride=2, interrupt=_at(2, 1, 2))
    directory = str(checkpoint_directory(root / "layer-quanta" / "adjoint", 2))
    # Probe 0's five cotangents and probe 1's batches 0 and 1: the owner
    # wrote (1, 2) and the run died before the checkpoint got it.
    written = sorted([f"cotangent-0-{b}.pt" for b in range(5)]
                     + ["cotangent-1-0.pt", "cotangent-1-1.pt"])
    assert calls == [("abandon", directory), ("dispose", directory, written)]
    assert not Path(directory).exists()
    sealed = root / "layer-quanta" / "adjoint" / "checkpoints"
    assert sorted(path.name for path in sealed.iterdir()) == ["boundary-004", "boundary-005"]
