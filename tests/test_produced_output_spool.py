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
        backend, capacity_deferred=CapacityDeferred, timeout_s=2)
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
    with pytest.raises(TimeoutError, match="export has not landed"):
        with owner._produced_lock.held():
            owner._produced_publish(next(iter(owner._produced_groups)), group,
                                    deadline=time.monotonic() + 0.05)
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
    adapter = ProducedOutputSpool(backend, capacity_deferred=CapacityDeferred, timeout_s=2)
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
    adapter = ProducedOutputSpool(backend, capacity_deferred=CapacityDeferred, timeout_s=2)
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
