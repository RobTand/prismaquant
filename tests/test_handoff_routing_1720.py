"""PQ #1720: quantum-handoff manifest hashes delegate to digests owners.

Routing: handoff_record_bytes must call indent2_json_file_bytes;
load_quantum_handoff (driven to its pinned-digest mismatch) and
band_serial_manifest_bytes (driven to its sealed-readset mismatch) must call
bytes_sha256hex. HandoffStream._write + load_handoff_inputs.staged +
require_band_serial_readset share the same one-line recipe but need a live
stream, staged checkpoint entries, and a sealed derivation to invoke; their
rows are pinned by the baseline tripwire. Values: byte-identical to the
verbatim old spellings.
"""
import hashlib
import json

import pytest

import prismaquant.joint_quantum_handoff as handoff
from prismaquant.digests import bytes_sha256hex, indent2_json_file_bytes


def _record():
    return {"schema": "s", "boundary": 1, "producer": {"layer": 1},
            "source": "src", "session": "sess", "n_probes": 1, "n_batches": 1,
            "activation_entries": [], "owner_states": {},
            "handoff_sha256": "0" * 64}


def _spies(monkeypatch):
    calls = {"dumps": [], "hash": []}
    real_dumps = getattr(handoff, "indent2_json_file_bytes", None)
    real_hash = getattr(handoff, "bytes_sha256hex", None)

    def stand_in_dumps(value):
        calls["dumps"].append(value)
        return real_dumps(value) if real_dumps is not None else None

    def stand_in_hash(value):
        calls["hash"].append(value)
        return real_hash(value) if real_hash is not None else None

    monkeypatch.setattr(handoff, "indent2_json_file_bytes", stand_in_dumps,
                        raising=False)
    monkeypatch.setattr(handoff, "bytes_sha256hex", stand_in_hash, raising=False)
    return calls


def test_handoff_hashes_route_to_owners(monkeypatch, tmp_path):
    calls = _spies(monkeypatch)
    record = _record()
    assert handoff.handoff_record_bytes(record) == indent2_json_file_bytes(record)
    with pytest.raises(handoff.QuantumHandoffRefused):
        handoff.handoff_record_bytes({"schema": "s"})
    blob = tmp_path / "handoff.json"
    blob.write_bytes(b"not a sealed handoff")
    with pytest.raises(handoff.QuantumHandoffRefused):
        handoff.load_quantum_handoff(blob, "0" * 64, record={}, adjoint_slice={},
                                     kda_capture_kernel=None)
    manifest = tmp_path / "manifest.gz"
    manifest.write_bytes(b"not a sealed readset")
    record = {"executable_readset": {"manifest_path": str(manifest),
                                        "manifest_sha256": "0" * 64}}
    with pytest.raises(handoff.QuantumHandoffRefused):
        handoff.band_serial_manifest_bytes(record, {}, {}, output_root=tmp_path)
    assert len(calls["dumps"]) == 1
    assert len(calls["hash"]) == 2


def test_handoff_values_match_verbatim_spellings():
    record = _record()
    assert indent2_json_file_bytes(record) == (
        json.dumps(dict(record), sort_keys=True, indent=2,
                   allow_nan=False) + "\n").encode()
    raw = b"\x00\x01handoff"
    assert bytes_sha256hex(raw) == hashlib.sha256(raw).hexdigest()
