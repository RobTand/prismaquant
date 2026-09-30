"""Actual admission/ownership regressions found by the independent review."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

import prismaquant.joint_adjoint_checkpoints as cp
from test_referenced_adjoint_checkpoint import (
    _referenced, _tensors, _shared, BOUNDARY,
)


def _copied(tmp_path):
    space, owner, _refs, record = _referenced(tmp_path / "owner")
    destination = cp.adjoint_space(tmp_path / "copied")
    shared, passes = _shared()
    copied = cp.write_adjoint_checkpoint(
        destination, boundary=BOUNDARY, session=record["session"],
        cotangents=_tensors(), shared_adjoint=shared, shared_pass=passes)
    return destination, owner, copied


@pytest.mark.parametrize("field", ["shape", "dtype", "tensor_bytes", "schema"])
def test_metadata_mismatch_refuses_at_construction(tmp_path, field):
    _space, owner, _refs, original = _referenced(tmp_path)
    record = copy.deepcopy(original)
    metadata = record["activation_entries"][0]["metadata"]
    metadata[field] = {"shape": [1], "dtype": "torch.int8",
                       "tensor_bytes": 1, "schema": "unknown"}[field]
    try:
        with pytest.raises(ValueError):
            cp.CheckpointIncoming(record, n_probes=2, n_batches=1)
    finally:
        owner.__exit__(None, None, None)


def test_valid_copied_checkpoint_retains_canonical_slots_and_exact_operands(tmp_path):
    space, owner, record = _copied(tmp_path)
    incoming = cp.CheckpointIncoming(record, n_probes=2, n_batches=1)
    slots = []
    try:
        cp.load_adjoint_checkpoint(
            space, record, stream_incoming=True,
            cotangent_factory=lambda rows: slots.extend(rows) or {})
        assert [row["name"] for row in slots] == ["cotangent-0-0", "cotangent-1-0"]
        for probe in range(2):
            with incoming.open(probe, max_resident_bytes=128,
                               residency_check=lambda n: None) as stream:
                (value,) = stream.take([(probe, 0)])
                assert torch.equal(value, _tensors()[probe, 0])
    finally:
        owner.__exit__(None, None, None)


@pytest.mark.parametrize("field", ["slot", "kind", "coordinates"])
def test_copied_metadata_identity_refuses_before_payload(tmp_path, field):
    _space, owner, record = _copied(tmp_path)
    row = record["activation_entries"][0]
    row["metadata"]["identity"][field] = {
        "slot": "foreign", "kind": "foreign", "coordinates": {"probe": 1, "batch": 0},
    }[field]
    try:
        with pytest.raises(ValueError):
            cp.CheckpointIncoming(record, n_probes=2, n_batches=1)
    finally:
        owner.__exit__(None, None, None)


@pytest.mark.parametrize("exposure", ["record", "entries", "session"])
def test_validated_snapshot_cannot_be_redirected_after_construction(tmp_path, exposure):
    _space, owner, _refs, record = _referenced(tmp_path)
    incoming = cp.CheckpointIncoming(record, n_probes=2, n_batches=1)
    if exposure == "record":
        row = record["activation_entries"][0]
        row.clear()
        row.update(copy.deepcopy(record["activation_entries"][1]))
    elif exposure == "entries":
        row = incoming.entries(0)[0]
        row.clear()
        row.update(incoming.entries(1)[0])
    else:
        incoming.session["generation"] = "foreign"
    try:
        with incoming.open(0, max_resident_bytes=128, residency_check=lambda n: None) as stream:
            (value,) = stream.take([(0, 0)])
            assert torch.equal(value, _tensors()[0, 0]), "validated row redirected to another operand"
            assert stream.layout((0, 0)) == (tuple(_tensors()[0, 0].shape), _tensors()[0, 0].dtype)
    finally:
        owner.__exit__(None, None, None)


@pytest.mark.parametrize("dimension", ["float", "bool"])
@pytest.mark.parametrize("consumer", ["adapter", "loader"])
def test_metadata_dimensions_require_exact_positive_ints(tmp_path, dimension, consumer):
    space, owner, _refs, original = _referenced(tmp_path)
    # A valid control must construct first: a generic serializer failure must
    # not accidentally qualify the rejection cases below.
    cp.CheckpointIncoming(original, n_probes=2, n_batches=1)
    record = copy.deepcopy(original)
    if dimension == "float":
        row = record["activation_entries"][0]
        row["metadata"]["shape"] = [float(n) for n in row["shape"]]
    else:
        row = record["activation_entries"][1]
        assert row["tensor_bytes"] == 16
        row["shape"] = [1, 4]
        row["metadata"]["shape"] = [True, 4]
    calls = []
    def factory(rows):
        calls.append(list(rows))
        return {}
    try:
        with pytest.raises((ValueError, RuntimeError)):
            if consumer == "adapter":
                cp.CheckpointIncoming(record, n_probes=2, n_batches=1)
            else:
                manifest = cp.checkpoint_manifest_entry(record)
                Path(manifest["path"]).write_bytes(cp.checkpoint_manifest_bytes(record))
                cp.load_adjoint_checkpoint(space, record, stream_incoming=True,
                                           cotangent_factory=factory)
        assert calls == [], "non-integer metadata dimensions reached reservation"
    finally:
        owner.__exit__(None, None, None)


@pytest.mark.parametrize("field", ["tensor_bytes", "metadata_shape"])
def test_standalone_stream_loader_refuses_before_factory_with_matching_manifest(tmp_path, field):
    space, owner, _refs, original = _referenced(tmp_path)
    record = copy.deepcopy(original)
    row = record["activation_entries"][0]
    if field == "tensor_bytes":
        row["tensor_bytes"] += 1
    else:
        row["metadata"]["shape"] = [1]
    # Own fixture input only: manifest matches the altered receipt exactly.
    # This proves per-row validation, rather than a mismatched-manifest shortcut.
    manifest = cp.checkpoint_manifest_entry(record)
    Path(manifest["path"]).write_bytes(cp.checkpoint_manifest_bytes(record))
    calls = []
    def factory(rows):
        calls.append(list(rows))
        return {}
    try:
        with pytest.raises((ValueError, RuntimeError)):
            cp.load_adjoint_checkpoint(space, record, stream_incoming=True,
                                       cotangent_factory=factory)
        assert calls == [], "invalid layout reached destination reservation"
    finally:
        owner.__exit__(None, None, None)
