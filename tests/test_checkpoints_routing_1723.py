"""PQ #1723: checkpoint shared-state hashes delegate to digests owners.

Routing: _write_shared_state_payload + unpack_shared_states +
read_shared_state_pack + load_adjoint_receipt must call bytes_sha256hex;
write_adjoint_receipt + _checkpoint_manifest_envelope_bytes must call
indent2_json_file_bytes. _load_checkpoint_shared_states shares the same
one-line digest recipe but needs a staged checkpoint store to invoke; its
row is pinned by the baseline tripwire. Values: byte-identical to the
verbatim old spellings.
"""
import hashlib
import json
import struct

import pytest

import prismaquant.joint_adjoint_checkpoints as checkpoints
from prismaquant.digests import bytes_sha256hex, indent2_json_file_bytes


def _spies(monkeypatch):
    calls = {"dumps": [], "hash": []}
    real_dumps = getattr(checkpoints, "indent2_json_file_bytes", None)
    real_hash = getattr(checkpoints, "bytes_sha256hex", None)

    def stand_in_dumps(value):
        calls["dumps"].append(value)
        return real_dumps(value) if real_dumps is not None else None

    def stand_in_hash(value):
        calls["hash"].append(value)
        return real_hash(value) if real_hash is not None else None

    monkeypatch.setattr(checkpoints, "indent2_json_file_bytes", stand_in_dumps,
                        raising=False)
    monkeypatch.setattr(checkpoints, "bytes_sha256hex", stand_in_hash,
                        raising=False)
    return calls


def _pack(member: bytes):
    members = [{"name": "shared-adjoint-0-0", "offset": 0, "bytes": len(member),
                "sha256": hashlib.sha256(member).hexdigest()}]
    index = checkpoints._shared_state_pack_index(members)
    return member + index + struct.pack("<Q", len(index)) \
        + checkpoints._SHARED_STATE_PACK_MAGIC


def test_checkpoint_hashes_route_to_owners(monkeypatch, tmp_path):
    calls = _spies(monkeypatch)
    space = tmp_path / "space"
    (space / "entries").mkdir(parents=True)
    row = checkpoints._write_shared_state_payload(space, "s", b"payload")
    assert row["sha256"] == hashlib.sha256(b"payload").hexdigest()
    assert row["file_bytes"] == len(b"payload")
    payload = _pack(b"member-bytes")
    assert checkpoints.unpack_shared_states(payload) == [
        ("shared-adjoint-0-0", b"member-bytes")]
    blob = tmp_path / "entry.pkl"
    blob.write_bytes(b"shared-state bytes")
    entry = {"path": str(blob), "sha256": "0" * 64,
             "file_bytes": blob.stat().st_size, "name": "s"}
    with pytest.raises(RuntimeError):
        checkpoints.read_shared_state_pack(entry, deadline=None)
    with pytest.raises(RuntimeError):
        checkpoints.load_adjoint_receipt(blob, "0" * 64)
    assert checkpoints.write_adjoint_receipt(space, {"a": 1}) is True
    session = {"generation": 1, "run_identity_sha256": "0" * 64}
    plan = {"name": "n", "path": "p", "tensor_bytes": 8, "file_envelope": 8,
            "shape": [2], "dtype": "float32", "slot": "s", "batch_index": 0,
            "probe_index": 0}
    estimate = checkpoints._checkpoint_manifest_envelope_bytes(
        boundary=0, session=session, activation_plan=[plan], shared_plan=[plan])
    assert isinstance(estimate, int) and estimate > 0
    assert len(calls["hash"]) == 4
    assert len(calls["dumps"]) == 2


def test_checkpoint_values_match_verbatim_spellings():
    raw = b"\x00\x01checkpoint"
    assert bytes_sha256hex(raw) == hashlib.sha256(raw).hexdigest()
    document = {"b": [1, 2], "a": "x"}
    assert indent2_json_file_bytes(document) == (
        json.dumps(document, sort_keys=True, indent=2,
                   allow_nan=False) + "\n").encode()
