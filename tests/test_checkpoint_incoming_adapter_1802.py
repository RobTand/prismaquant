"""Real exact-entry checkpoint adapter contracts, explicitly CPU-only."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

import prismaquant.joint_adjoint_checkpoints as checkpoints
from test_referenced_adjoint_checkpoint import _referenced, _tensors
from test_stage_b_streamed_incoming_1143 import _Budget


def _incoming(tmp_path):
    space, owner, references, record = _referenced(tmp_path)
    incoming = checkpoints.CheckpointIncoming(record, n_probes=2, n_batches=1)
    return space, owner, references, record, incoming


def _raw(tensor):
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def test_stream_head_has_no_activation_reads_or_sink_writes(tmp_path, monkeypatch):
    space, owner, references, record, incoming = _incoming(tmp_path)
    reads, writes, slots = [], [], []
    original = checkpoints.read_exact_entry_tensors

    def read(rows, **kwargs):
        rows = list(rows)
        reads.extend(row["path"] for row in rows)
        return original(rows, **kwargs)

    class Sink(dict):
        def __setitem__(self, key, value):
            writes.append(key)
            super().__setitem__(key, value)

    def factory(rows):
        slots.extend(rows)
        return Sink()

    monkeypatch.setattr(checkpoints, "read_exact_entry_tensors", read)
    plane, shared, passes = checkpoints.load_adjoint_checkpoint(
        space, record, cotangent_factory=factory, stream_incoming=True)
    assert reads == writes == [] and dict(plane) == {}
    assert [row["name"] for row in slots] == ["cotangent-0-0", "cotangent-1-0"]
    assert [row["name"] for probe in range(2) for row in incoming.entries(probe)] == [
        checkpoints.checkpoint_cotangent_plane(record)[key]["name"]
        for key in [(0, 0), (1, 0)]]
    assert set(shared) == {(0, 0), (1, 0)} and passes[0] == {"tag": "a"}
    for probe in range(2):
        budget = _Budget(128)
        with incoming.open(probe, max_resident_bytes=budget.cap,
                           residency_check=budget) as stream:
            assert stream.layout((probe, 0)) == (tuple(_tensors()[probe, 0].shape), torch.float32)
            assert not reads
            (value,) = stream.take([(probe, 0)])
            assert _raw(value) == _raw(_tensors()[probe, 0])
            assert stream.telemetry["entries"] == 1
        assert budget.held == 0
        assert reads == [references[probe, 0].path]
        reads.clear()
    eager, eager_shared, eager_passes = checkpoints.load_adjoint_checkpoint(space, record)
    assert set(eager) == {(0, 0), (1, 0)}
    assert eager_passes == passes and set(eager_shared) == set(shared)
    assert all(Path(ref.path).is_file() for ref in references.values())
    owner.__exit__(None, None, None)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_loader_selection_is_strict_before_manifest_access(monkeypatch, value):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid option reached manifest reader")
    monkeypatch.setattr(checkpoints, "_verified_checkpoint_manifest", forbidden)
    with pytest.raises(ValueError, match="must be a bool"):
        checkpoints.load_adjoint_checkpoint("unused", {}, stream_incoming=value)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "bytes", "dtype", "shape", "boundary", "generation"])
def test_adapter_refuses_bad_grid_identity_and_layout_before_reads(tmp_path, monkeypatch, mutation):
    _space, owner, _references, record, _incoming_adapter = _incoming(tmp_path)
    altered = copy.deepcopy(record)
    row = altered["activation_entries"][0]
    if mutation == "missing":
        altered["activation_entries"].pop()
    elif mutation == "duplicate":
        altered["activation_entries"].append(copy.deepcopy(row))
    elif mutation == "bytes":
        row["tensor_bytes"] += 1
    elif mutation == "dtype":
        row["dtype"] = "not_a_dtype"
    elif mutation == "shape":
        row["shape"] = [True, 4]
    elif mutation == "boundary":
        row["metadata"]["identity"]["coordinates"]["boundary"] += 1
    else:
        row["metadata"]["identity"]["session"]["generation"] = "foreign"
    with pytest.raises(ValueError):
        checkpoints.CheckpointIncoming(altered, n_probes=2, n_batches=1)
    owner.__exit__(None, None, None)


@pytest.mark.parametrize("case", ["budget", "order", "unread", "pass_error"])
def test_stream_order_budget_and_primary_error_cleanup(tmp_path, case):
    _space, owner, _refs, _record, incoming = _incoming(tmp_path)
    budget = _Budget(63 if case == "budget" else 128)
    expected = KeyError if case == "pass_error" else RuntimeError
    match = {"budget": "resident budget", "order": "capture order",
             "unread": "unread", "pass_error": "original pass"}[case]
    with pytest.raises(expected, match=match):
        with incoming.open(0, max_resident_bytes=budget.cap, residency_check=budget) as stream:
            if case == "unread":
                pass
            elif case == "order":
                stream.take([(1, 0)])
            elif case == "pass_error":
                stream.take([(0, 0)])
                raise KeyError("original pass")
            else:
                stream.take([(0, 0)])
    assert budget.held == 0
    owner.__exit__(None, None, None)


def test_late_corruption_refuses_without_delivering_bad_tensor(tmp_path):
    _space, owner, refs, _record, incoming = _incoming(tmp_path)
    budget = _Budget(128)
    with incoming.open(0, max_resident_bytes=128, residency_check=budget) as stream:
        (first,) = stream.take([(0, 0)])
        assert _raw(first) == _raw(_tensors()[0, 0])
    assert budget.held == 0
    path = Path(refs[1, 0].path)
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    try:
        with pytest.raises((RuntimeError, ValueError)):
            with incoming.open(1, max_resident_bytes=128, residency_check=budget) as stream:
                stream.take([(1, 0)])
        assert stream.telemetry["entries"] == 0 and budget.held == 0
    finally:
        path.write_bytes(original)
        owner.__exit__(None, None, None)
