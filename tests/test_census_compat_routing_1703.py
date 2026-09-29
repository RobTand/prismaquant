"""PQ #1703: census/compat strict indent-2 records delegate to digests owners.

Routing: tessera_campaign._publish_capture_load_execution and
glm_capture_compatibility.create_capture_compatibility must call
indent2_json_file_bytes for the record and bytes_sha256hex for its digest.
Values: the delegation is byte-identical to the verbatim old spelling
(sort_keys, indent 2, allow_nan=False, one LF, UTF-8). _run_streamed_calibration
shares the same one-line recipe; its row is pinned by the baseline tripwire.
"""
import hashlib
import json
import types

import pytest

from prismaquant import tessera_campaign as tc
from prismaquant import glm_capture_compatibility as gcc
from prismaquant.digests import bytes_sha256hex, indent2_json_file_bytes


def _spy(monkeypatch, module, name):
    calls = []
    real = getattr(module, name, None)

    def stand_in(value):
        calls.append(value)
        return real(value) if real is not None else None

    monkeypatch.setattr(module, name, stand_in, raising=False)
    return calls


def test_publish_load_execution_routes(monkeypatch, tmp_path):
    dumps_calls = _spy(monkeypatch, tc, "indent2_json_file_bytes")
    hash_calls = _spy(monkeypatch, tc, "bytes_sha256hex")
    args = types.SimpleNamespace(cache_dir=str(tmp_path))
    out = tc._publish_capture_load_execution(
        args, capture={"c": 1}, execution={"e": 2}, resources={})
    assert len(dumps_calls) == 1
    assert len(hash_calls) == 1
    assert out["sha256"] == bytes_sha256hex(indent2_json_file_bytes(dumps_calls[0]))


def test_create_compat_routes(monkeypatch, tmp_path):
    dumps_calls = _spy(monkeypatch, gcc, "indent2_json_file_bytes")
    hash_calls = _spy(monkeypatch, gcc, "bytes_sha256hex")
    monkeypatch.setattr(gcc, "_verify", lambda *a, **k: None)
    monkeypatch.setattr(gcc, "source_derivative_identity", lambda model: {"r": 1})
    target = tmp_path / "compat.json"
    out = gcc.create_capture_compatibility(
        capture={"c": 1}, producer={"p": 2}, forward_equivalence={"f": 3},
        model="m", output=str(target))
    assert len(dumps_calls) == 1
    assert len(hash_calls) == 1
    assert target.read_bytes() == indent2_json_file_bytes(dumps_calls[0])
    assert out["sha256"] == bytes_sha256hex(hash_calls[0])


def test_strict_indent2_recipe_golden():
    payload = {"b": [1, 2.5, "x"], "a": {"n": None, "u": "é"}}
    assert (indent2_json_file_bytes(payload)
            == (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
                + "\n").encode())
    raw = indent2_json_file_bytes(payload)
    assert bytes_sha256hex(raw) == hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        indent2_json_file_bytes({"nan": float("nan")})
