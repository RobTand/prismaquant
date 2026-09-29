"""File-hash helpers route through the digests owners (issue #1672).

RED: each test patches the digests owner with a recorder and calls the site
helper. Before delegation the helpers hash inline, so the recorder is never
consulted and the test fails with the unrouted digest observed. After
delegation the recorder sees every call. Chunk sizes differ across sites
(1/8/16/64 MiB) but chunking never changes a hash value, so the values test
holds on both sides of the change.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "tools"))
sys.path.insert(0, str(REPOSITORY))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UNION = _load("route_union_1672", "tools/union_tessera_census_tables.py")
TEACHER = _load("route_teacher_1672", "tools/full_kl_teacher_payload.py")

from prismaquant import digests as owners  # noqa: E402
from prismaquant.quality_prefill_pb_adapter import (  # noqa: E402
    document_bytes,
    document_file_sha256,
)


def test_union_sha256_file_routes_through_owner(monkeypatch, tmp_path):
    target = tmp_path / "u.bin"
    target.write_bytes(b"abc" * 1000)
    seen = []
    monkeypatch.setattr(owners, "file_sha256hex",
                        lambda path, **k: seen.append(Path(path).name) or "routed")
    assert UNION._sha256_file(target) == "routed"
    assert seen == ["u.bin"]


def test_teacher_file_sha256_routes_through_owner(monkeypatch, tmp_path):
    target = tmp_path / "t.bin"
    target.write_bytes(b"abc" * 1000)
    seen = []
    monkeypatch.setattr(owners, "file_sha256hex",
                        lambda path, **k: seen.append(Path(path).name) or "routed")
    assert TEACHER.file_sha256(target) == "routed"
    assert seen == ["t.bin"]


def test_document_file_sha256_routes_through_owner(monkeypatch):
    import prismaquant.quality_prefill_pb_adapter as adapter

    seen = []
    monkeypatch.setattr(adapter, "bytes_sha256hex",
                        lambda data: seen.append(bytes(data)) or "routed")
    assert document_file_sha256({"a": 1}) == "routed"
    assert len(seen) == 1


def test_values_match_hashlib(tmp_path):
    blob = b"abc" * 1000
    target = tmp_path / "v.bin"
    target.write_bytes(blob)
    want = hashlib.sha256(blob).hexdigest()
    assert UNION._sha256_file(target) == want
    assert TEACHER.file_sha256(target) == want
    assert document_file_sha256({"a": 1}) == hashlib.sha256(
        document_bytes({"a": 1})).hexdigest()
