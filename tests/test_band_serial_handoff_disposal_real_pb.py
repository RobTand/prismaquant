"""A failed handoff's uncommitted groups are disposed of by their producer (PQ #1251).

PrismaBuild's ``abort_prewrite`` releases a group's durable claim only once
every planned path is absent, and leaves that disposal to the producer: it
never deletes a durable-origin file. A handoff names no entry until its
record exists, so a producer that fails before its record removes its own
uncommitted groups' files (exactly the planned paths of each of its own
prewrites) and then aborts each prewrite, which PrismaBuild accepts.

Before PQ #1251 the owner aborted without disposing: a group with a file on
disk kept its prewrite (``abort-files-present-retain``) and its files. The
two cases here run against the pinned PrismaBuild generation the handoff
suites share: a group written straight to its canonical paths and still
incomplete, and, through the local output spool, groups whose exports have
landed but which were never committed at their origin.

The module is marked ``own_process`` for the reason
``test_stage_a_produced_boundary_chain`` records (PQ #1008).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import test_stage_a_produced_boundary_chain as chain
from test_band_serial_handoff_produced import (
    ENTRY_GROUPS, N_BATCHES, N_PROBES, PREFETCH, _adjoint_slice, _Owner,
    band_campaign, pb_source, producer_owner)
from test_stage_a_produced_boundary_chain import _isolated_launch_context  # noqa: F401

# PQ #1008: one pinned prismabuild per process (tests/conftest.py).
pytestmark = pytest.mark.own_process


def _group_ids(publication, producer):
    """Every entry group's batch id: each probe's batches in ``PREFETCH`` groups."""
    return [publication.batch_id_for(kind="cotangent",
                                     boundary_index=int(producer["layer"]),
                                     probe_index=probe, group_index=group)
            for probe in range(N_PROBES)
            for group in range(-(-N_BATCHES // PREFETCH))]


def _prewrite(publication, batch_id):
    from prismabuild import produced_output as po
    return po._read_prewrite(po._prewrites_dir(publication.queue.root,
                                               publication.instance)
                             / f"{batch_id}.prewrite.json")


def _emitter(producer, storage, publication, emitters):
    from prismaquant.joint_quantum_handoff import HandoffEmitter
    emitter = HandoffEmitter(record=producer, adjoint_slice=_adjoint_slice(producer),
                             boundary_storage=storage, capture_batch=1,
                             publication=publication)
    emitters.append(emitter)
    return emitter


def _plane():
    return {(p, b): torch.full((2, 4), 10.0 * p + b)
            for p in range(N_PROBES) for b in range(N_BATCHES)}


def _owners():
    return [[_Owner({"scale": float(p + b)}) for b in range(N_BATCHES)]
            for p in range(N_PROBES)]


def _entries(producer):
    from prismaquant.joint_quantum_handoff import handoff_root
    root = handoff_root(producer["output_space"]["root"])
    return sorted(root.glob("*/entries/*")), sorted(root.glob("*/handoff.json"))


def test_a_failed_handoff_disposes_its_incomplete_group(tmp_path, monkeypatch):
    """Written straight to its canonical paths: the first group holds one of
    its two entries when the plane read fails. The abort then succeeds."""
    _src, pb_repo = pb_source(monkeypatch)
    producer, _consumer, storage = band_campaign(tmp_path)
    publication, _q, _env = producer_owner(tmp_path, pb_repo, producer, storage)
    source = _plane()

    class Plane(dict):
        def __getitem__(self, key):
            if key == (0, 1):
                raise RuntimeError("fixture: plane read failed")
            return source[key]

    emitters = []
    with pytest.raises(RuntimeError, match="plane read failed"):
        _emitter(producer, storage, publication, emitters).emit(
            grad_plane=Plane(), cotangent_owners=_owners(), n_probes=N_PROBES,
            n_batches=N_BATCHES, kda_capture_kernel=None)
    first = _group_ids(publication, producer)[0]
    files, records = _entries(producer)
    assert records == []
    assert files == [], "the incomplete group's entry is still on disk"
    assert _prewrite(publication, first) is None, "the prewrite was retained"
    # PrismaBuild agrees: nothing is left to abort.
    assert publication.abort_prewrite(batch_id=first) == {
        "ok": True, "batch_id": first, "aborted": False}
    report = emitters[0].export_report
    assert report["disposed_uncommitted"] == [
        {"batch_id": first, "files_removed": 1, "export_live": False,
         "abort_ok": True, "refusal": None}]


def test_a_failed_handoff_disposes_its_landed_uncommitted_groups(
        tmp_path, monkeypatch):
    """Through the spool: every entry group's export has landed and none is
    committed (the commits happen in the settle, which fails here). Each
    group's canonical files are removed and each prewrite aborts."""
    from fleet_sdk import require_prismabuild_sdk
    require_prismabuild_sdk()
    _src, pb_repo = pb_source(monkeypatch)
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts
    from prismaquant.produced_output_spool import MAX_ENV, ROOT_ENV

    producer, _consumer, storage = band_campaign(tmp_path)
    spool = tmp_path / "local-output"
    capacity = {"cpu": 4, "mem_gb": 4}
    publication, q, _env = producer_owner(
        tmp_path, pb_repo, producer, storage,
        producer_environment={ROOT_ENV: str(spool), MAX_ENV: str(1 << 20)},
        claim_capacity=capacity)

    def failing_settle(self):
        # Every export lands first, so each group is on disk, uncommitted.
        self._local_output_spool.drain()
        raise RuntimeError("fixture: the settle failed after every export landed")

    monkeypatch.setattr(StreamedBoundaryArtifacts, "settle_local_output",
                        failing_settle)
    emitters = []
    with chain._fleet(q, tmp_path, capacity=capacity):
        with pytest.raises(RuntimeError, match="the settle failed"):
            _emitter(producer, storage, publication, emitters).emit(
                grad_plane=_plane(), cotangent_owners=_owners(), n_probes=N_PROBES,
                n_batches=N_BATCHES, kda_capture_kernel=None)
    ids = _group_ids(publication, producer)
    assert len(ids) == ENTRY_GROUPS
    files, records = _entries(producer)
    assert records == []
    assert files == [], "a landed, uncommitted group's files are still on disk"
    for batch_id in ids:
        assert _prewrite(publication, batch_id) is None, batch_id
        assert publication.abort_prewrite(batch_id=batch_id) == {
            "ok": True, "batch_id": batch_id, "aborted": False}
    disposed = emitters[0].export_report["disposed_uncommitted"]
    assert sorted(row["batch_id"] for row in disposed) == sorted(ids)
    assert all(row["abort_ok"] and not row["export_live"] and row["files_removed"]
               and row["refusal"] is None for row in disposed), disposed
    assert not [path for path in spool.rglob("*")
                if path.is_file() and path.parent.name == "payload"], (
        "the spool still holds a group's payload")
