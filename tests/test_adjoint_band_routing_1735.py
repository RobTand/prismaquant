"""PQ #1735: adjoint-band bytes hashes delegate to digests owners.

Routing: _json_file must call bytes_sha256hex once for the pinned read.
read_sealed_checkpoint + stage_a_bind_identity + sealed_chain_state +
band_from_request need sealed stores / calibration files / chain-state dirs;
they share the same one-line bytes-hash recipe and their rows are pinned by
the baseline tripwire. Values: byte-identical to hashlib.
"""
import hashlib
import json

import pytest

import prismaquant.joint_adjoint_band as band


def _spies(monkeypatch):
    calls = {"hash": []}
    real_hash = getattr(band, "bytes_sha256hex", None)

    def stand_in_hash(value):
        calls["hash"].append(value)
        return real_hash(value) if real_hash is not None else None

    monkeypatch.setattr(band, "bytes_sha256hex", stand_in_hash, raising=False)
    return calls


def test_json_file_hash_routes_to_owner(monkeypatch, tmp_path):
    calls = _spies(monkeypatch)
    path = tmp_path / "plan.json"
    path.write_bytes(json.dumps({"a": 1}).encode())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert band._json_file(path, digest, where="plan") == {"a": 1}
    assert len(calls["hash"]) == 1
    with pytest.raises(band.BandRefused):
        band._json_file(path, "0" * 64, where="plan")


def test_band_values_match_verbatim_spelling():
    raw = b"\x00\x01adjoint-band"
    assert band.bytes_sha256hex(raw) == hashlib.sha256(raw).hexdigest()
