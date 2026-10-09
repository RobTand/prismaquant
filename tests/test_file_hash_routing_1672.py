"""File-hash helpers route through the digests owners (issue #1672).

The three recorder/sentinel wiring tests this file once carried (union,
teacher and document routes) only re-asserted which owner each helper
forwards to, so they are removed. What stays pins actual behavior: each
helper's digest matches hashlib over the same bytes, and the document
serialization recipe is checked over its bytes.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

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

from prismaquant.quality_prefill_pb_adapter import (  # noqa: E402
    document_bytes,
    document_file_sha256,
)


def test_values_match_hashlib(tmp_path):
    blob = b"abc" * 1000
    target = tmp_path / "v.bin"
    target.write_bytes(blob)
    want = hashlib.sha256(blob).hexdigest()
    assert UNION._sha256_file(target) == want
    assert TEACHER.file_sha256(target) == want
    assert document_file_sha256({"a": 1}) == hashlib.sha256(
        document_bytes({"a": 1})).hexdigest()
