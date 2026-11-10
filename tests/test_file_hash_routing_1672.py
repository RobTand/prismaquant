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


from prismaquant import tessera_census_union as UNION
TEACHER = _load("route_teacher_1672", "tools/full_kl_teacher_payload.py")

from prismaquant.digests import bytes_sha256hex, canonical_json_bytes  # noqa: E402


def test_values_match_hashlib(tmp_path):
    blob = b"abc" * 1000
    target = tmp_path / "v.bin"
    target.write_bytes(blob)
    want = hashlib.sha256(blob).hexdigest()
    assert UNION._sha256_file(target) == want
    assert TEACHER.file_sha256(target) == want
    encoded = canonical_json_bytes({'a': 1}, where='document') + b'\n'
    assert encoded == b'{"a":1}\n'
    assert bytes_sha256hex(encoded) == hashlib.sha256(encoded).hexdigest()
