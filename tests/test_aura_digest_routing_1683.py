"""Aura_cost bytes hashes route through the digests owner (issue #1683).

RED: the round-trip test patches digests.bytes_sha256hex with a recorder and
writes/loads a unit checkpoint. Before delegation the recorder is never
consulted and the test fails with the unrouted digest observed. After
delegation both the write and the load check route through it. The values
test holds on both sides (same algorithm on already-materialized bytes).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY))

from prismaquant import aura_cost as aura  # noqa: E402
from prismaquant import digests as owners  # noqa: E402


def test_checkpoint_roundtrip_routes_through_owner(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(aura, "bytes_sha256hex",
                        lambda data: seen.append(bytes(data)) or "0" * 64)
    aura._write_aura_unit_checkpoint(
        tmp_path, qname="q", identity_sha256="i" * 64,
        state={"a": 1})
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(written) == 1
    loaded = aura._load_aura_unit_checkpoint(
        written[0], qname="q", identity_sha256="i" * 64)
    assert loaded == {"a": 1}
    assert len(seen) >= 2


def test_values_match_hashlib():
    blob = b"abc" * 100
    assert owners.bytes_sha256hex(blob) == hashlib.sha256(blob).hexdigest()
