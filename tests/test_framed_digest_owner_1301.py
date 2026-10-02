"""Framing and domains retain their existing identities through one owner."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismaquant import digests, tessera_joint_aura as aura


def _cells():
    document = json.loads((Path(__file__).parent / "fixtures" /
                           "digest_joint_aura_1593.json").read_text())
    return {tuple(key.split("|", 1)): value for key, value in document["inputs"]["cells"].items()}, document


def test_qualification_routes_exact_prefix_and_ordered_frames(monkeypatch):
    cells, document = _cells()
    owner = getattr(digests, "length_framed_bytes_sha256", None)
    seen = []

    def framed(frames, *, prefix):
        assert owner is not None
        assert iter(frames) is frames, "rows remain a lazy single-pass stream"
        rows = list(frames)
        seen.append((prefix, rows))
        return owner(rows, prefix=prefix)

    monkeypatch.setattr(aura, "length_framed_bytes_sha256", framed)
    assert aura._qualification_cells_sha256(cells) == document["old_rows"]["cells_seal_sha256"]
    assert seen == [(aura.QUALIFICATION_CELLS_SCHEMA.encode() + b"\n",
                     [bytes.fromhex(raw) for raw in document["old_rows"]["cell_row_hex"]])]


def test_source_v2_reuses_frame_owner_with_its_distinct_domain(monkeypatch):
    owner = getattr(digests, "length_framed_bytes_sha256", None)
    seen = []

    def framed(frames, *, prefix):
        assert owner is not None
        rows = list(frames)
        seen.append((prefix, rows))
        return owner(rows, prefix=prefix)

    monkeypatch.setattr(digests, "length_framed_bytes_sha256", framed)
    digests.source_tree_profiles([("é.py", b"\0\xff"), ("a.py", b"first")])
    assert seen == [(b"prismaquant.source_tree.v2\0",
                     [b"a.py", b"first", "é.py".encode(), b"\0\xff"])]


def test_frame_owner_propagates_stream_failure_without_consuming_more():
    observed = []

    def frames():
        observed.append("first")
        yield b"first"
        observed.append("failure")
        raise RuntimeError("fixture stream failed")

    with pytest.raises(RuntimeError, match="fixture stream failed"):
        digests.length_framed_bytes_sha256(frames(), prefix=b"domain\n")
    assert observed == ["first", "failure"]
