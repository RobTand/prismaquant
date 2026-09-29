"""PQ #1705: chain-seed strict envelope delegates to digests owners.

Routing: load_seed_spec + _sealed_checkpoint pinned checks and the
write_seed_marker/write_seed_receipt return hashes must call bytes_sha256hex;
both writers' payloads must call indent2_json_file_bytes. Seal comparisons
stay structurally identical (the lint counts ==/!= + raise, not hashlib
calls). Values: byte-identical to the verbatim old spellings.
"""
import hashlib
import json
import types

import pytest

from prismaquant import stage_a_chain_seed as seed
from prismaquant.digests import bytes_sha256hex, indent2_json_file_bytes


def _spy(monkeypatch, name):
    calls = []
    real = getattr(seed, name, None)

    def stand_in(value):
        calls.append(value)
        return real(value) if real is not None else None

    monkeypatch.setattr(seed, name, stand_in, raising=False)
    return calls


def test_envelope_routes_to_owners(monkeypatch, tmp_path):
    dumps_calls = _spy(monkeypatch, "indent2_json_file_bytes")
    hash_calls = _spy(monkeypatch, "bytes_sha256hex")
    plan = types.SimpleNamespace(binding={"b": 1})
    marker = seed.write_seed_marker(tmp_path, plan, run_identity={"r": 1})
    assert (tmp_path / seed.SEED_MARKER_NAME).read_bytes() == \
        indent2_json_file_bytes(dumps_calls[0])
    seed.write_seed_receipt(tmp_path, {"schema": seed.SEED_RECEIPT_SCHEMA})
    spec = tmp_path / "spec.json"
    spec.write_bytes(b'{"a": 1}')
    with pytest.raises(seed.ChainSeedRefused):
        seed.load_seed_spec(spec, "f" * 64)
    with pytest.raises(seed.ChainSeedRefused):
        seed._sealed_checkpoint({"path": str(spec), "sha256": "f" * 64}, "test")
    assert len(dumps_calls) == 2
    assert len(hash_calls) == 4
    assert marker["sha256"] == bytes_sha256hex(
        indent2_json_file_bytes({"schema": seed.SEED_MARKER_SCHEMA,
                                 "seed": {"b": 1}, "run_identity": {"r": 1}}))


def test_strict_indent2_recipe_golden():
    document = {"seed": {"b": [1, "é"]}, "schema": "s"}
    assert (indent2_json_file_bytes(document)
            == (json.dumps(document, sort_keys=True, indent=2, allow_nan=False)
                + "\n").encode())
    raw = indent2_json_file_bytes(document)
    assert bytes_sha256hex(raw) == hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        indent2_json_file_bytes({"nan": float("nan")})
