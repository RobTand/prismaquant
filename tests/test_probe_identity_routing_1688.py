"""Joint_aura probe identity routes through DIRECT_ASCII_STRICT (issue #1688).

RED: the routing test swaps the profile for a recorder and builds an
identity (model validation stubbed at its lazy import seam). Before
delegation the recorder is never consulted and the test fails with the
unrouted digest observed. After delegation both the encoding and the hash
route through the owners. The values test holds on both sides, including
int keys (numeric sort) and non-ASCII (escaped).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import types

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY))

from prismaquant import cost_streaming as cs  # noqa: E402
from prismaquant import digests as owners  # noqa: E402
from prismaquant import joint_aura as ja  # noqa: E402


def test_probe_identity_routes_through_owner(monkeypatch):
    monkeypatch.setattr(cs, "validate_streamed_model_identity",
                        lambda *a, **k: {})
    seen = []
    stub = types.SimpleNamespace(
        encoded=lambda v: seen.append(("encode", v)) or b'{"b":1}')
    monkeypatch.setattr(ja, "DIRECT_ASCII_STRICT", stub)
    monkeypatch.setattr(ja, "bytes_sha256hex",
                        lambda data: seen.append(("hash", bytes(data))) or "routed")
    probe = {"source_model": {"x": 1}, "b": 1}
    ident = ja._ValidatedProbeIdentity(probe)
    assert ("encode", probe) in [(n, v) for n, v in seen if n == "encode"]
    assert ident._sha256 == "routed"


def test_values_match_inline():
    assert owners.DIRECT_ASCII_STRICT.encoded({"b": 1, "a": [1, 2]}) == b'{"a":[1,2],"b":1}'
    assert owners.DIRECT_ASCII_STRICT.encoded({10: "a", 9: "b"}) == b'{"9":"b","10":"a"}'
    assert owners.DIRECT_ASCII_STRICT.encoded({"u": "héllo"}) == b'{"u":"h\\u00e9llo"}'
    assert owners.bytes_sha256hex(b"abc") == hashlib.sha256(b"abc").hexdigest()
