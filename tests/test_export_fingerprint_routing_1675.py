"""Export compact-lax fingerprints route through DIRECT profiles (issue #1675).

RED: each test patches the DIRECT profile with a recorder and calls the
fingerprint helper. Before delegation the helpers dump inline, so the
recorder is never consulted and the test fails with the unrouted digest
observed. After delegation the recorder sees every call. The values test
holds on both sides (same options, including lax NaN and default=str).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY))

from prismaquant import digests as owners  # noqa: E402
from prismaquant import export_native_compressed as export  # noqa: E402


def _index(tmp_path):
    a = tmp_path / "a.pt"
    a.write_bytes(b"0" * 10)
    return SimpleNamespace(_paths={"a": a})


def _cache(tmp_path):
    w = tmp_path / "w.pt"
    w.write_bytes(b"1" * 10)
    return SimpleNamespace(
        weights={("m", "w"): str(w)},
        cache_dir=tmp_path,
        activation_max_abs={"m": float("nan")},
        metadata={"src": Path("/a")},
    )


def _stub(monkeypatch, name, seen, marker=b"routed"):
    import types
    stub = types.SimpleNamespace()
    stub.encoded = lambda v: seen.append((name, v)) or marker
    stub.sha256 = lambda v: hashlib.sha256(stub.encoded(v)).hexdigest()
    monkeypatch.setattr(owners, name, stub)


def test_activation_rows_route_through_owner(monkeypatch, tmp_path):
    seen = []
    _stub(monkeypatch, "DIRECT_ASCII_LAX", seen)
    out = export._activation_index_fingerprint(_index(tmp_path), tmp_path)
    assert out["hash"] == hashlib.sha256(b"routed").hexdigest()[:16]
    assert [n for n, _ in seen] == ["DIRECT_ASCII_LAX"]


def test_production_rows_route_through_owner(monkeypatch, tmp_path):
    seen = []
    _stub(monkeypatch, "DIRECT_ASCII_LAX", seen)
    _stub(monkeypatch, "DIRECT_ASCII_LAX_DEFAULT_STR", seen, marker=b"other")
    export._production_cache_fingerprint(_cache(tmp_path), [(("m", "w"))])
    assert [n for n, _ in seen].count("DIRECT_ASCII_LAX") == 2


def test_production_metadata_routes_through_default_str(monkeypatch, tmp_path):
    seen = []
    _stub(monkeypatch, "DIRECT_ASCII_LAX_DEFAULT_STR", seen)
    _stub(monkeypatch, "DIRECT_ASCII_LAX", seen, marker=b"other")
    export._production_cache_fingerprint(_cache(tmp_path), [(("m", "w"))])
    assert [n for n, _ in seen].count("DIRECT_ASCII_LAX_DEFAULT_STR") == 1


def test_values_match_inline(tmp_path):
    fp = export._activation_index_fingerprint(_index(tmp_path), tmp_path)
    rows = [["a", "a.pt", 10, (tmp_path / "a.pt").stat().st_mtime_ns]]
    assert fp["hash"] == hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    assert fp["n_files"] == 1
