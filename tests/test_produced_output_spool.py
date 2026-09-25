"""PQ's precommit adapter, with an explicitly controlled PB transport seam.

These tests qualify serializer bounds, canonical-reference carriage and delayed
progress/publication. The PB project's produced_spool tests qualify actual
sealing, source-host movement and durable acknowledgements independently.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import threading
import time

import pytest
import torch

from prismaquant.produced_output_spool import ProducedOutputSpool, ProducedOutputSpoolRefused
from prismaquant.perturbed_x_cache import write_exact_activation_cache_entry
from test_stage_a_produced_boundary_chain import _isolated_launch_context  # noqa: F401
import test_stage_a_produced_boundary_chain as chain

# PQ #1008: one pinned prismabuild per process (tests/conftest.py).
pytestmark = pytest.mark.own_process


class CapacityDeferred(RuntimeError):
    pass


class ControlledExport:
    """A transport double: acknowledgement is controlled explicitly by tests."""
    def __init__(self, root, *, capacity=1 << 30):
        self.root = root
        self.capacity = capacity
        self.groups = {}
        self.polls = 0

    def reserve_group(self, batch_id, *, ceiling_bytes):
        live = sum(g["ceiling"] for g in self.groups.values() if not g["released"])
        if live + ceiling_bytes > self.capacity:
            raise CapacityDeferred("quota full")
        directory = self.root / batch_id
        directory.mkdir(parents=True)
        self.groups[batch_id] = dict(ceiling=ceiling_bytes, directory=directory,
                                    entries=[], complete=False, released=False, failure=None)
        return directory

    def submit_group(self, batch_id, *, entries):
        self.groups[batch_id]["entries"] = entries
        return {"ok": True, "export_key": hashlib.sha256(batch_id.encode()).hexdigest()}

    def poll_group(self, batch_id):
        self.polls += 1
        group = self.groups[batch_id]
        if group["failure"]:
            return {"ok": False, "refusal": group["failure"]}
        return {"ok": True, "complete": group["complete"]}

    def acknowledge(self, batch_id):
        group = self.groups[batch_id]
        for row in group["entries"]:
            target = Path(row["destination_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(row["source_path"], target)
        group["complete"] = True

    def release_group(self, batch_id):
        group = self.groups[batch_id]
        assert group["complete"], "release without an acknowledgement"
        shutil.rmtree(group["directory"])
        group["released"] = True
        return {"ok": True}


def _owner(tmp_path, monkeypatch, *, n_batches=chain.GROUP_SIZE):
    owner, publication, queue, env, pb_repo = chain._bound_owner(
        tmp_path, n_batches=n_batches, staging_timeout_s=2)
    # The public bind path is tested separately; this replaces only PB's
    # asynchronous exporter while keeping the actual writer and publisher.
    owner._published = True
    backend = ControlledExport(tmp_path / "local")
    owner._local_output_spool = ProducedOutputSpool(
        backend, capacity_deferred=CapacityDeferred)
    return owner, publication, backend


def test_preallocation_uses_serialized_inode_and_preserves_exact_bytes(tmp_path, monkeypatch):
    calls = []
    original = os.posix_fallocate
    def allocate(fd, offset, length):
        original(fd, offset, length)
        calls.append((os.fstat(fd).st_ino, length))
    monkeypatch.setattr(os, "posix_fallocate", allocate)
    tensor = torch.arange(512, dtype=torch.float32).reshape(16, 32)
    kwargs = dict(identity={"session": "fixture"}, max_tensor_bytes=2048,
                  max_file_bytes=2048 + 65536)
    normal = write_exact_activation_cache_entry(tmp_path / "normal", "same", tensor, **kwargs)
    local = write_exact_activation_cache_entry(tmp_path / "local", "same", tensor,
                                             preallocate=True, **kwargs)
    assert calls == [(Path(local.path).stat().st_ino, 2048 + 65536)]
    assert Path(normal.path).read_bytes() == Path(local.path).read_bytes()
    assert normal.sha256 == local.sha256
    assert Path(local.path).stat().st_size == local.file_bytes < calls[0][1]


def test_serialization_cannot_overrun_preallocated_file(tmp_path):
    with pytest.raises(RuntimeError, match="preallocated ceiling|unexpected pos"):
        write_exact_activation_cache_entry(
            tmp_path, "too-large", torch.arange(8), identity={"session": "fixture"},
            max_tensor_bytes=64, max_file_bytes=128, preallocate=True)
    assert not list(tmp_path.glob("*.pt*"))


def test_local_write_does_not_publish_or_advance_before_durable_ack(tmp_path, monkeypatch):
    owner, publication, backend = _owner(tmp_path, monkeypatch)
    progress = []
    class Progress:
        def entry(self, **kwargs):
            progress.append(kwargs)
    owner.watch_progress(Progress())
    refs = chain._write_group(owner)
    group = next(iter(owner._produced_groups.values()))
    batch_id = group["batch_id"]
    assert backend.groups[batch_id]["entries"]
    assert all(not Path(ref.path).exists() for ref in refs)
    assert group["published"] is None and progress == []
    assert len(list((tmp_path / "local").rglob("*.pt"))) == chain.GROUP_SIZE
    # One look, and nothing advances on it: no progress, no publication.
    assert owner._local_output_spool.landed(batch_id) is False
    owner._commit_local_output_progress()
    assert group["published"] is None and progress == []
    backend.acknowledge(batch_id)
    owner.settle_local_output()
    assert len(progress) == chain.GROUP_SIZE
    assert backend.groups[batch_id]["released"]
    assert all(Path(ref.path).exists() for ref in refs)
    polls = backend.polls
    for _ in range(10):
        owner._commit_local_output_progress()
    assert len(progress) == chain.GROUP_SIZE and backend.polls == polls
    with owner._produced_lock.held():
        owner._produced_publish(next(iter(owner._produced_groups)), group,
                                deadline=time.monotonic() + 2)
    assert group["published"] is not None


def test_failed_export_retains_local_sources_and_canonical_prewrite(tmp_path, monkeypatch):
    owner, publication, backend = _owner(tmp_path, monkeypatch)
    refs = chain._write_group(owner)
    group = next(iter(owner._produced_groups.values()))
    batch_id = group["batch_id"]
    backend.groups[batch_id]["failure"] = "destination identity changed"
    with pytest.raises(ProducedOutputSpoolRefused, match="destination identity changed"):
        owner.settle_local_output()
    aborted = []
    monkeypatch.setattr(publication, "abort_prewrite", lambda **kwargs: aborted.append(kwargs))
    owner.__exit__(RuntimeError, RuntimeError("primary capture failure"), None)
    assert aborted == []
    assert not backend.groups[batch_id]["released"]
    assert len(list((tmp_path / "local").rglob("*.pt"))) == chain.GROUP_SIZE
    assert all(not Path(ref.path).exists() for ref in refs)
    assert owner.produced_output_report()["local_spool"]["pending_groups"] == 1


def test_incomplete_group_retains_reservation_on_failure(tmp_path, monkeypatch):
    owner, publication, backend = _owner(tmp_path, monkeypatch)
    owner.write(torch.arange(8), batch_index=0, boundary_index=0)
    group = next(iter(owner._produced_groups.values()))
    assert not backend.groups[group["batch_id"]]["entries"]
    aborted = []
    monkeypatch.setattr(publication, "abort_prewrite", lambda **kwargs: aborted.append(kwargs))
    owner.__exit__(RuntimeError, RuntimeError("capture failed mid-group"), None)
    assert aborted == []
    assert owner.produced_output_report()["local_spool"]["pending_groups"] == 1


def test_full_spool_waits_only_until_prior_export_is_released(tmp_path):
    backend = ControlledExport(tmp_path / "local", capacity=65536)
    adapter = ProducedOutputSpool(backend, capacity_deferred=CapacityDeferred)
    adapter.reserve("one", 65536)
    # A real tiny writer reference is enough to exercise reservation ownership.
    ref = write_exact_activation_cache_entry(
        adapter.directory("one"), "entry", torch.arange(8), identity={"session": "fixture"},
        max_tensor_bytes=64, max_file_bytes=65536)
    adapter.record("one", ref, tmp_path / "canonical")
    adapter.submit("one")
    finished = threading.Event()
    def acknowledge():
        time.sleep(0.15)
        backend.acknowledge("one")
        finished.set()
    thread = threading.Thread(target=acknowledge)
    thread.start()
    try:
        directory = adapter.reserve("two", 65536)
    finally:
        thread.join(timeout=3)
    assert finished.is_set() and backend.groups["one"]["released"]
    assert directory == tmp_path / "local" / "two"
    assert adapter.report()["retained_ceiling_bytes"] == 65536


def test_local_entry_refuses_group_geometry_overrun_before_allocation(tmp_path, monkeypatch):
    owner, _publication, backend = _owner(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="sealed group tensor ceiling"):
        owner.write(torch.zeros(5000), batch_index=0, boundary_index=0)
    assert backend.groups == {}
    assert not list((tmp_path / "local").rglob("*.pt*"))


def test_local_atomic_publication_never_overwrites_or_deletes_foreign_file(tmp_path, monkeypatch):
    real_link = os.link
    foreign = b"unrelated file appeared before publication"
    def raced_link(source, destination, **kwargs):
        Path(destination).write_bytes(foreign)
        return real_link(source, destination, **kwargs)
    monkeypatch.setattr(os, "link", raced_link)
    with pytest.raises(FileExistsError):
        write_exact_activation_cache_entry(
            tmp_path, "raced", torch.arange(8), identity={"session": "fixture"},
            max_tensor_bytes=64, max_file_bytes=65536, preallocate=True)
    assert (tmp_path / "raced.pt").read_bytes() == foreign
    assert not (tmp_path / "raced.pt.tmp").exists()


def test_a_checkpoint_entry_declares_its_class_and_a_payload_entry_stays_unchanged(tmp_path):
    backend = ControlledExport(tmp_path / "local")
    adapter = ProducedOutputSpool(backend, capacity_deferred=CapacityDeferred)
    adapter.reserve("g", 1 << 20)
    references = []
    for name in ("payload-entry", "checkpoint-entry"):
        references.append(write_exact_activation_cache_entry(
            adapter.directory("g"), name, torch.arange(8), identity={"session": "fixture"},
            max_tensor_bytes=64, max_file_bytes=65536))
    adapter.record("g", references[0], tmp_path / "canonical")
    adapter.record("g", references[1], tmp_path / "canonical", artifact_class="checkpoint")
    with pytest.raises(ProducedOutputSpoolRefused, match="artifact class"):
        adapter.record("g", references[1], tmp_path / "canonical", artifact_class="temp")
    adapter.submit("g")
    payload, checkpoint = backend.groups["g"]["entries"]
    assert "artifact_class" not in payload
    assert checkpoint["artifact_class"] == "checkpoint"
    backend.acknowledge("g")
    adapter.await_group("g")
    assert [ref.path for ref in adapter.durable_entries()] == [
        str(tmp_path / "canonical" / Path(ref.path).name) for ref in references]


def test_the_former_stage_a_import_names_the_same_client():
    from prismaquant import stage_a_local_spool as former
    assert former.BoundaryOutputSpool is ProducedOutputSpool
    assert former.BoundarySpoolRefused is ProducedOutputSpoolRefused


def _landed_read_back_group(adapter, backend, batch_id, tmp_path):
    """One acknowledged group whose entry a read on this box may follow."""
    adapter.reserve(batch_id, 65536)
    ref = write_exact_activation_cache_entry(
        adapter.directory(batch_id), f"{batch_id}-entry", torch.arange(8),
        identity={"session": "fixture"}, max_tensor_bytes=64, max_file_bytes=65536)
    canonical = adapter.record(batch_id, ref, tmp_path / "canonical", read_back=True)
    adapter.submit(batch_id)
    backend.acknowledge(batch_id)
    adapter.await_group(batch_id)
    return canonical


def test_a_full_window_releases_the_oldest_unread_landed_group_with_a_record(tmp_path):
    """PQ #1110: a landed copy kept for a later read goes when the window
    needs its room, oldest first, and the release is recorded."""
    backend = ControlledExport(tmp_path / "local", capacity=2 * 65536)
    adapter = ProducedOutputSpool(backend, capacity_deferred=CapacityDeferred)
    first = _landed_read_back_group(adapter, backend, "one", tmp_path)
    _landed_read_back_group(adapter, backend, "two", tmp_path)
    assert adapter.holds("one") and adapter.holds("two")
    adapter.reserve("three", 65536)
    assert not adapter.holds("one") and adapter.holds("two")
    assert [(e["batch_id"], e["where"]) for e in adapter.report()["evictions"]] == [
        ("one", "reserve three")]
    # Its entry is read through PrismaBuild now: the spool names no copy.
    with adapter.local_reads([first]) as local:
        assert local == {}


def test_a_group_released_for_room_keeps_its_retired_files_for_its_next_read(
        tmp_path, monkeypatch):
    """PQ #1236: the window releases a landed group for room while the roll
    has read only part of it. The entry the roll retires keeps its file,
    because the next window's read of a live entry is the group's first
    publication, and PrismaBuild stats every origin the group names. Before
    the fix the retirement unlinked the file at once and that publication
    refused ``descriptor-unstatable``. The held files go with the group's
    last live entry."""
    from prismaquant.stage_a_produced_output import (
        BoundaryProducedPublicationFailed)

    owner, publication, backend = _owner(tmp_path, monkeypatch)
    spool = owner._local_output_spool
    refs = chain._write_group(owner)
    key, group = owner._produced_group_for(refs[0])
    batch_id = group["batch_id"]
    backend.acknowledge(batch_id)
    assert spool.landed(batch_id) and spool.holds(batch_id)
    # The first window reads its entry from this box's copy.
    with owner.prefetch(refs[:1]) as window:
        owner.get(window, refs[0])
    assert owner.telemetry["produced_local_reads"] == 1
    # The next plane's first write finds the window full. The landed group
    # is not being read and has nothing retired, so it goes for room.
    backend.capacity = backend.groups[batch_id]["ceiling"]
    chain._write_group(owner, boundary_index=1, count=1)
    assert not spool.holds(batch_id)
    assert [e["batch_id"] for e in spool.report()["evictions"]] == [batch_id]
    # The roll retires the entry it read; three of the group's entries are
    # still live, so the file stays.
    owner._retire(refs[0])
    owner._produced_flush_deferred_unlinks()
    # The next window reads a live entry through PrismaBuild: the group's
    # first publication, over all four origins.
    try:
        with owner._produced_lock.held():
            owner._produced_fund_group_for_read(
                key, group, {key: group}, time.monotonic() + 2)
    except BoundaryProducedPublicationFailed as exc:
        pytest.fail(f"the read's first publication refused: {exc}")
    assert group["published"] is not None
    assert Path(refs[0].path).exists()
    assert owner.telemetry["produced_deferred_unlinks_done"] == 0
    # The roll retires the rest; the last one takes the held files with it.
    for ref in refs[1:-1]:
        owner._retire(ref)
    assert all(Path(ref.path).exists() for ref in refs)
    owner._retire(refs[-1])
    assert not any(Path(ref.path).exists() for ref in refs)
    assert owner.telemetry["produced_deferred_unlinks_done"] == len(refs) - 1
    assert owner.produced_output_report()["deferred_unlinks"] == {}


def test_a_window_nothing_can_free_refuses_at_once_with_a_record(tmp_path):
    """Every held copy is being read and no export is live: nothing will
    free room, so the writer's reservation refuses at once and says why; a
    claim ahead of the writer is declined and counted, not refused."""
    from prismaquant.produced_output_spool import ProducedWindowRefused
    backend = ControlledExport(tmp_path / "local", capacity=65536)
    adapter = ProducedOutputSpool(backend, capacity_deferred=CapacityDeferred)
    held = _landed_read_back_group(adapter, backend, "one", tmp_path)
    with adapter.local_reads([held]) as local:
        assert list(local) == [held]
        with pytest.raises(ProducedWindowRefused, match="the caller does not wait|no export is live"):
            adapter.reserve("ahead", 65536, wait=False)
        assert adapter.report()["refusals"] == []
        assert adapter.report()["window_declines"] == 1
        started = time.monotonic()
        with pytest.raises(ProducedWindowRefused, match="no export is live"):
            adapter.reserve("two", 65536)
        assert time.monotonic() - started < 1.0
    (record,) = adapter.report()["refusals"]
    assert (record["batch_id"], record["state"]) == ("two", "window-full-no-live-export")
    assert adapter.holds("one")


# --- PQ #1225: the Stage B handoff tail ------------------------------------
#
# Row 029 held its GPU 324 s in ``handoff-out``: about 50 s of per-file and
# per-directory fsync on spool entries PrismaBuild copies and hashes anyway,
# 204 s waiting on exports admitted one at a time, then 32 origin commits in
# a row after the last export. The export keys that name PrismaBuild's own
# records of those exports were nowhere in the row's output.

def _synced_paths(monkeypatch):
    """Record the path behind every descriptor ``os.fsync`` is given."""
    synced = []
    real = os.fsync

    def fsync(descriptor):
        synced.append(os.readlink(f"/proc/self/fd/{descriptor}"))
        return real(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)
    return synced


def test_a_spool_entry_is_written_without_fsync(tmp_path, monkeypatch):
    """The spool copy is never the only copy of committed work: PrismaBuild
    hashes every byte it exports, and a same-box read hashes every byte it
    reads. An fsync of it only holds the writer."""
    owner, _publication, _backend = _owner(tmp_path, monkeypatch)
    synced = _synced_paths(monkeypatch)
    chain._write_group(owner)
    local = os.path.realpath(tmp_path / "local")
    assert len(list(Path(local).rglob("*.pt"))) == chain.GROUP_SIZE
    assert [path for path in synced if path.startswith(local)] == []


def test_an_entry_outside_the_spool_is_still_fsynced(tmp_path, monkeypatch):
    """Without a spool the written file is the only copy: file and directory
    are both fsynced before the entry is returned."""
    synced = _synced_paths(monkeypatch)
    directory = tmp_path / "canonical"
    reference = write_exact_activation_cache_entry(
        directory, "entry", torch.arange(8), identity={"session": "fixture"},
        max_tensor_bytes=64, max_file_bytes=65536)
    # The file under its temporary name, then the directory that names it.
    assert os.path.realpath(reference.path) + ".tmp" in synced
    assert os.path.realpath(directory) in synced


def test_each_group_keeps_its_export_record(tmp_path):
    """The report names each group's export action, its bytes, when it was
    reserved, submitted, seen landed and released, and the drain's wait."""
    import json

    backend = ControlledExport(tmp_path / "local")
    adapter = ProducedOutputSpool(backend, capacity_deferred=CapacityDeferred)
    adapter.reserve("one", 65536)
    reference = write_exact_activation_cache_entry(
        adapter.directory("one"), "entry", torch.arange(8),
        identity={"session": "fixture"}, max_tensor_bytes=64, max_file_bytes=65536)
    adapter.record("one", reference, tmp_path / "canonical")
    adapter.submit("one")
    timer = threading.Timer(0.3, backend.acknowledge, args=("one",))
    timer.start()
    try:
        adapter.await_group("one", where="drain")
    finally:
        timer.join()
    adapter.drain(release=True)
    report = adapter.report()
    assert report["schema"] == "prismaquant.produced_output_spool.v3"
    assert "exports" in report, "the report keeps no export record"
    (record,) = report["exports"]
    assert record["export_key"] == hashlib.sha256(b"one").hexdigest()
    assert (record["batch_id"], record["entries"], record["bytes"]) == (
        "one", 1, reference.file_bytes)
    assert (record["reserved_unix"] <= record["submitted_unix"]
            <= record["landed_unix"] <= record["released_unix"])
    assert record["drain_wait_s"] >= 0.2
    assert record["export_wait_s"] == record["drain_wait_s"] == report["drain_wait_s"]
    json.dumps(report)


class _RecordingExport(ControlledExport):
    """The transport double, plus origin commits and a log of every call."""

    def __init__(self, root):
        super().__init__(root)
        self.calls = []
        self.committed = {}

    def poll_group(self, batch_id):
        self.calls.append(("poll", batch_id))
        return super().poll_group(batch_id)

    def release_group(self, batch_id):
        self.calls.append(("release", batch_id))
        return super().release_group(batch_id)

    def commit_origin_group(self, batch_id, descriptors, *, lifetime):
        self.calls.append(("commit", batch_id))
        self.committed.setdefault(batch_id, threading.Event()).set()
        return {"ok": True, "ref": {"batch_id": batch_id, "lifetime": lifetime}}


def _write_only_owner(adapter, groups):
    """An owner's settle state around a real spool: only what it reads."""
    from types import SimpleNamespace
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts

    owner = StreamedBoundaryArtifacts.__new__(StreamedBoundaryArtifacts)
    owner._local_output_spool = adapter
    owner._produced_plan = {"write_only": True, "origin_lifetime": "consumed"}
    owner._produced_groups = groups
    owner._produced = SimpleNamespace(
        descriptor_for=lambda reference, producer_generation: {"path": reference.path})
    owner._produced_origin_batches = []
    owner.telemetry = {"produced_groups_committed_at_origin": 0,
                       "produced_commit_origin_s": 0.0}
    owner._produced_flush_deferred_unlinks = lambda **_: None
    owner._commit_local_output_progress = lambda: None
    return owner


def test_a_write_only_group_commits_while_later_exports_still_copy(tmp_path):
    """Each group commits at its origin once its own export lands, not after
    the last one: row 029's 32 commits (9.8 s) followed its whole drain."""
    backend = _RecordingExport(tmp_path / "local")
    adapter = ProducedOutputSpool(backend, capacity_deferred=CapacityDeferred)
    groups = {}
    for batch_id in ("g0", "g1"):
        adapter.reserve(batch_id, 65536)
        reference = write_exact_activation_cache_entry(
            adapter.directory(batch_id), f"{batch_id}-entry", torch.arange(8),
            identity={"session": "fixture"}, max_tensor_bytes=64, max_file_bytes=65536)
        canonical = adapter.record(batch_id, reference, tmp_path / "canonical")
        adapter.submit(batch_id)
        groups[batch_id] = {"batch_id": batch_id, "references": [canonical],
                            "planned": [None, None]}
    backend.acknowledge("g0")
    first = backend.committed.setdefault("g0", threading.Event())
    seen = []

    def land_later():
        # g1's export lands once g0 committed, or after 5 s regardless.
        seen.append(first.wait(timeout=5))
        backend.acknowledge("g1")

    thread = threading.Thread(target=land_later)
    thread.start()
    try:
        _write_only_owner(adapter, groups).settle_local_output()
    finally:
        thread.join(timeout=10)
    assert seen == [True], "g0 committed only after g1's export landed"
    assert [call for call in backend.calls if call[0] == "commit"] == [
        ("commit", "g0"), ("commit", "g1")]
    assert all(backend.groups[batch_id]["released"] for batch_id in groups)
    assert [group["origin_ref"]["batch_id"] for group in groups.values()] == ["g0", "g1"]
