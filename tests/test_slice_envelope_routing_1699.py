"""PQ #1699: joint_adjoint_slices manifest envelope delegates to digests owners.

Routing: checkpoint_manifest_bytes must call indent2_json_file_bytes (exact
options incl. allow_nan=False); the five bytes-hash sites (manifest_entry,
write_band_receipt, load_stage_a_receipt_like, write_adjoint_slice,
load_adjoint_slice) must call bytes_sha256hex. Seal comparisons stay.
Values: the delegation is byte-identical to the verbatim old spellings.
"""
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant import joint_adjoint_slices as slices
from prismaquant.digests import bytes_sha256hex, indent2_json_file_bytes


def _record():
    return {"schema": "prismaquant.joint_adjoint_checkpoint.v1",
            "boundary": 3,
            "session": {"generation": 1},
            "activation_entries": [],
            "shared_state_entries": [],
            "cotangent_sha256": "0" * 64}


def _spy(monkeypatch, name):
    calls = []
    real = getattr(slices, name, None)
    def stand_in(value):
        calls.append(value)
        return real(value) if real is not None else None
    monkeypatch.setattr(slices, name, stand_in, raising=False)
    return calls


def test_manifest_bytes_routes_to_indent2_owner(monkeypatch):
    calls = _spy(monkeypatch, "indent2_json_file_bytes")
    out = slices.checkpoint_manifest_bytes(_record())
    assert len(calls) == 1
    assert out == indent2_json_file_bytes(_record())


def test_hash_sites_route_to_bytes_owner(monkeypatch, tmp_path):
    calls = _spy(monkeypatch, "bytes_sha256hex")
    monkeypatch.setattr(slices, "validate_band_receipt", lambda band: 3)
    monkeypatch.setattr(slices, "verify_adjoint_slice", lambda *a, **k: None)
    monkeypatch.setattr(slices, "_checkpoint_own_directory",
                        lambda record: tmp_path)
    monkeypatch.setattr(slices, "publish_new_bytes", lambda path, data: True)
    slices.checkpoint_manifest_entry(_record())
    slices.write_band_receipt(tmp_path / "band.json", {})
    slices.write_adjoint_slice(tmp_path / "slice.json", {}, layer=0)
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(b'{"kind": "x"}')
    with pytest.raises(RuntimeError):
        slices.load_stage_a_receipt_like(receipt, sha256="f" * 64)
    with pytest.raises(slices.AdjointSliceRefused):
        slices.load_adjoint_slice(receipt, sha256="f" * 64, layer=0)
    assert len(calls) == 5


def test_envelope_values_match_verbatim_spellings():
    record = _record()
    assert (slices.checkpoint_manifest_bytes(record)
            == (json.dumps(record, sort_keys=True, indent=2, allow_nan=False)
                + "\n").encode())
    raw = b'{"kind": "x"}'
    assert bytes_sha256hex(raw) == hashlib.sha256(raw).hexdigest()
