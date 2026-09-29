"""PQ #1707: GLM source oracle hashes delegate to digests owners.

Routing: bound_json / corrected_source / original_source / _corrected_reach
must call bytes_sha256hex; sha256(path) must call file_sha256hex. All
_require comparisons stay with identical refusal type/text. _observe shares
the same one-line recipe (too environment-heavy to invoke: it reads the
installed transformers tree); its row is pinned by the baseline tripwire.
Values: byte-identical to the verbatim old spellings.
"""
import hashlib
from pathlib import Path

import pytest

from prismaquant import glm_source_derivative as gsl
from prismaquant.digests import bytes_sha256hex, file_sha256hex


def _spy(monkeypatch, name):
    calls = []
    real = getattr(gsl, name, None)

    def stand_in(value):
        calls.append(value)
        return real(value) if real is not None else None

    monkeypatch.setattr(gsl, name, stand_in, raising=False)
    return calls


def test_oracle_routes_to_owners(monkeypatch, tmp_path):
    hash_calls = _spy(monkeypatch, "bytes_sha256hex")
    file_calls = _spy(monkeypatch, "file_sha256hex")
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"\x00\x01glm-oracle")
    assert gsl.sha256(blob) == file_sha256hex(blob)
    with pytest.raises(ValueError):
        gsl.bound_json({"path": str(blob), "sha256": "0" * 64}, "test")
    with pytest.raises(ValueError):
        gsl.corrected_source(b"garbage")
    with pytest.raises(ValueError):
        gsl.original_source(b"garbage")
    with pytest.raises(ValueError):
        gsl._corrected_reach(blob)
    assert len(file_calls) == 1
    # bound_json, corrected_source, original_source, _corrected_reach key,
    # and the original_source check _corrected_reach falls into.
    assert len(hash_calls) == 5


def test_oracle_values_match_verbatim_spellings(tmp_path):
    raw = b"\x00\x01glm-oracle"
    assert bytes_sha256hex(raw) == hashlib.sha256(raw).hexdigest()
    blob = tmp_path / "blob.bin"
    blob.write_bytes(raw)
    with blob.open("rb") as stream:
        expected = hashlib.file_digest(stream, "sha256").hexdigest()
    assert file_sha256hex(blob) == expected
