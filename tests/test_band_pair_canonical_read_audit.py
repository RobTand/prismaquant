"""Driver read accounting, with no live fleet or D32 admission claim.

The real consumer_role and shared whole-file refusal run. Original checkpoint
entries come from real checkpoint writers and a digest-checked bound Stage A
slice. Handoff authentication, reconstructed planes, progress, stage lookup and
final serving summary remain setup doubles. Both the old generation control and
the declared checkpoint target are watched for real canonical Path.read_bytes.
"""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from tools import band_serial_handoff_live_pair as pair_driver
from prismaquant import calibration_data, joint_quantum_handoff, prismabuild_progress
from prismaquant import residency_map, staged_whole_file
from prismaquant.staged_tier_policy import TierPolicyRefused
from prismaquant.joint_adjoint_slices import load_adjoint_slice
from test_quantum_executable_readset import (
    N_PROBES, N_BATCHES, _tiny_records, _tiny_receipt, _bind_slice,
)


@pytest.fixture
def observed_consumer(tmp_path, monkeypatch):
    head = tmp_path / "head.bin"
    head_bytes = b"independently bound setup head"
    head.write_bytes(head_bytes)
    handoff = tmp_path / "handoff.json"
    generation = tmp_path / "generation.json"
    generation.write_bytes(b'{"generation":"fixture-not-in-reader-manifest"}')
    records, _parent = _tiny_records(tmp_path)
    receipt = _tiny_receipt(tmp_path, records[0]["campaign"])
    consumer, _digest = _bind_slice(
        next(row for row in records if row["layer"] == 2), receipt,
        tmp_path / "adjoint-slices",
    )
    adjoint_slice = load_adjoint_slice(
        consumer["adjoint"]["slice_path"], consumer["adjoint"]["slice_sha256"],
        layer=2, checkpoint_boundary=4,
    )
    entry = adjoint_slice["checkpoint"]["activation_entries"][0]
    undeclared = Path(entry["path"])
    assert hashlib.sha256(undeclared.read_bytes()).hexdigest() == entry["sha256"]
    assert undeclared.stat().st_size == entry["file_bytes"]
    watched = {undeclared, generation}
    owners = tmp_path / "owners.pt"
    events = []
    read_bytes = Path.read_bytes

    def observe_read(path):
        if path in watched:
            events.append("canonical_payload_open")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", observe_read)

    class Resolver:
        def staged_read(self, path, **_kwargs):
            if Path(path) == head:
                return {"sha256": hashlib.sha256(head_bytes).hexdigest()}
            assert Path(path) in watched
            if Path(path) == undeclared:
                assert _kwargs["expected_sha256"] == entry["sha256"]
            events.append("strict_undeclared_lookup")
            return None

    monkeypatch.setattr(residency_map, "residency_resolver", lambda: Resolver())

    def head_entry(_resolver, path, _entry, *, label):
        assert Path(path) == head and label == "calibration"
        return head_bytes

    monkeypatch.setattr(staged_whole_file, "read_staged_entry", head_entry)
    monkeypatch.setattr(pair_driver, "_load", lambda _root: ({
        "head": {"path": str(head), "sha256": hashlib.sha256(head_bytes).hexdigest()},
        "n_probes": N_PROBES, "n_batches": N_BATCHES,
    }, {"consumer": consumer}))
    monkeypatch.setattr(pair_driver, "_slice", lambda _record: adjoint_slice)
    monkeypatch.setattr(pair_driver, "_bind_staged", lambda _digest: None)
    monkeypatch.setattr(joint_quantum_handoff, "load_quantum_handoff",
                        lambda *_args, **_kwargs: {"owner_states": {"path": str(owners)}})
    monkeypatch.setattr(joint_quantum_handoff, "require_band_serial_readset",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(joint_quantum_handoff, "load_handoff_inputs",
                        lambda *_args, **_kwargs: (
                            {(p, b): torch.full((2, 4), 10.0 * p + b)
                             for p in range(N_PROBES) for b in range(N_BATCHES)},
                            {(p, b): {"scale": float(p + b)}
                             for p in range(N_PROBES) for b in range(N_BATCHES)},
                            {b: {"mask": [0, 1]} for b in range(N_BATCHES)},
                        ))
    monkeypatch.setattr(joint_quantum_handoff, "handoff_read_entries",
                        lambda *_args: [{"path": str(owners)}])
    monkeypatch.setattr(prismabuild_progress, "report", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(pair_driver, "_serving", lambda paths: {
        "served": {str(path): ["stage"] for path in paths},
        "resolver": {"bytes_from_pool": 0}, "tier_bytes": 123, "read_bytes": 123,
    })
    return tmp_path, handoff, undeclared, entry["sha256"], events


def test_consumer_negative_control_never_opens_canonical_payload(
        observed_consumer, capsys):
    root, handoff, _undeclared, _digest, events = observed_consumer
    rc = pair_driver.consumer_role(root, "a" * 64, str(handoff), "b" * 64)
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert rc == 0 and payload["checks"]["undeclared_refused"] is True
    assert payload["checks"]["no_pool_bytes"] is True
    assert events.count("strict_undeclared_lookup") == 1
    assert "canonical_payload_open" not in events, events


def test_shared_reader_refuses_undeclared_path_before_canonical_open(observed_consumer):
    _root, _handoff, undeclared, digest, events = observed_consumer
    with pytest.raises(TierPolicyRefused, match="readset-not-staged"):
        calibration_data._read_calibration_payload(undeclared, expected_sha256=digest)
    assert events == ["strict_undeclared_lookup"]



_REAL_SLICE_READER = pair_driver._slice


@pytest.mark.parametrize("tampered", [False, True], ids=["authentic-slice", "altered-slice"])
def test_real_consumer_admits_only_the_bound_checkpoint_slice(
        observed_consumer, monkeypatch, capsys, tampered):
    from prismaquant.joint_adjoint_slices import AdjointSliceRefused

    root, handoff, _undeclared, _digest, events = observed_consumer
    consumer = pair_driver._load(root)[1]["consumer"]
    monkeypatch.setattr(pair_driver, "_slice", _REAL_SLICE_READER)
    if tampered:
        path = Path(consumer["adjoint"]["slice_path"])
        document = json.loads(path.read_bytes())
        entry = document["checkpoint"]["activation_entries"][0]
        original = entry["sha256"]
        entry["sha256"] = ("1" if original[0] == "0" else "0") + original[1:]
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        with pytest.raises(AdjointSliceRefused, match="digest"):
            pair_driver.consumer_role(root, "a" * 64, str(handoff), "b" * 64)
        assert events == []
    else:
        assert pair_driver.consumer_role(root, "a" * 64, str(handoff), "b" * 64) == 0
        report = json.loads(capsys.readouterr().out.splitlines()[-1])
        assert report["checks"]["undeclared_refused"] is True
        assert report["checks"]["no_pool_bytes"] is True


@pytest.mark.parametrize("tampered", [False, True], ids=["authentic-producer-slice", "altered-producer-slice"])
def test_real_producer_checks_slice_before_staged_work(tmp_path, monkeypatch, tampered):
    from prismaquant.joint_adjoint_slices import AdjointSliceRefused

    records, _parent = _tiny_records(tmp_path)
    receipt = _tiny_receipt(tmp_path, records[0]["campaign"])
    producer, _digest = _bind_slice(
        next(row for row in records if row["layer"] == 3), receipt,
        tmp_path / "adjoint-slices",
    )
    if tampered:
        path = Path(producer["adjoint"]["slice_path"])
        path.write_bytes(path.read_bytes() + b"\n")
    monkeypatch.setattr(pair_driver, "_load", lambda root: ({}, {"producer": producer}))
    reached = []

    class StagedWorkBoundary(Exception):
        pass

    def stop_before_staged_work(digest):
        reached.append(digest)
        raise StagedWorkBoundary

    monkeypatch.setattr(pair_driver, "_bind_staged", stop_before_staged_work)
    if tampered:
        with pytest.raises(AdjointSliceRefused, match="digest"):
            pair_driver.producer_role(tmp_path, "a" * 64)
        assert reached == []
    else:
        with pytest.raises(StagedWorkBoundary):
            pair_driver.producer_role(tmp_path, "a" * 64)
        assert reached == ["a" * 64]
