"""Bound-publication digest sites route through digests owners (pq #2639).

RED: each test patches the digests owner with a recorder and calls the
site. Before routing the site hashes inline, so the recorder sees
nothing and the test fails. After routing the recorder sees the call.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "tools"))
sys.path.insert(0, str(REPOSITORY))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUANTA = _load("route2639_quanta", "tools/dispatch_joint_quanta.py")

from prismaquant import digests as owners  # noqa: E402
from prismaquant import stage_b_prep_io as preparation  # noqa: E402


def test_file_range_recipe_matches_offset_read(tmp_path):
    blob = bytes(range(256)) * 40000
    target = tmp_path / "r.bin"
    target.write_bytes(blob)
    for offset, size in ((0, len(blob)), (1, 1000), (7, 8 << 20), (len(blob) - 3, 3), (5, 0)):
        assert owners.file_range_sha256hex(target, offset, size) == hashlib.sha256(
            blob[offset:offset + size]).hexdigest()
    with pytest.raises(ValueError):
        owners.file_range_sha256hex(target, 0, len(blob) + 1)


def test_hash_range_routes_through_owner(monkeypatch, tmp_path):
    target = tmp_path / "r.bin"
    target.write_bytes(b"0123456789")
    seen = []
    monkeypatch.setattr(preparation, "file_range_sha256hex",
                        lambda *a: seen.append(a) or "routed")
    assert preparation._hash_range(str(target), 2, 4) == "routed"
    assert seen == [(str(target), 2, 4)]


def test_routed_values_match_hashlib(tmp_path):
    blob = b"abc" * 1000
    target = tmp_path / "r.bin"
    target.write_bytes(blob)
    assert owners.file_range_sha256hex(target, 7, 2500) == hashlib.sha256(blob[7:2507]).hexdigest()
