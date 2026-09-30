"""A negative retirement proof must read every declared binding document."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from prismaquant import stage_a_retirement as retirement


@pytest.mark.parametrize("kind", [
    "malformed-json", "malformed-gzip", "oversized", "inflated-gzip",
    "symlink-file", "symlink-directory",
])
def test_unreadable_binding_cannot_certify_no_live_consumer(tmp_path, monkeypatch, kind):
    space = tmp_path / "retired"
    space.mkdir()
    root = tmp_path / "bindings"
    root.mkdir()
    live = {"checkpoint": {"path": str(space / "checkpoints" / "boundary-2")}}
    document = root / "consumer.json"
    if kind == "malformed-json":
        document.write_text('{"checkpoint":')
    elif kind == "malformed-gzip":
        document = root / "consumer.json.gz"
        document.write_bytes(b"not a gzip document")
    elif kind == "oversized":
        document.write_text(json.dumps(live))
        monkeypatch.setattr(retirement, "BINDING_MAX_BYTES", 8)
    elif kind == "inflated-gzip":
        document = root / "consumer.json.gz"
        document.write_bytes(gzip.compress(json.dumps({"padding": "x" * 1000}).encode()))
        assert document.stat().st_size <= 64
        monkeypatch.setattr(retirement, "BINDING_MAX_BYTES", 64)
    else:
        target = tmp_path / "actual-consumers"
        target.mkdir()
        (target / "consumer.json").write_text(json.dumps(live))
        if kind == "symlink-file":
            document.symlink_to(target / "consumer.json")
        else:
            document = root / "consumer-directory"
            document.symlink_to(target, target_is_directory=True)

    with pytest.raises(retirement.RetirementRefused, match="binding") as refused:
        retirement.check_bindings([root], space=space, digests={})
    assert str(document) in str(refused.value)
    assert document.exists(), "the proof must not mutate the binding"
    assert not retirement.retirement_record_path(space).exists()


def test_unreadable_directory_cannot_certify_an_empty_binding_root(tmp_path, monkeypatch):
    root = tmp_path / "bindings"
    root.mkdir()

    def denied_walk(path, *, onerror=None):
        if onerror is not None:
            onerror(PermissionError(13, "Permission denied", str(path)))
        return iter(())

    monkeypatch.setattr(retirement.os, "walk", denied_walk)
    with pytest.raises(retirement.RetirementRefused, match="binding") as refused:
        retirement.check_bindings([root], space=tmp_path / "retired", digests={})
    assert str(root) in str(refused.value)


def test_complete_unrelated_scan_and_retired_subtree_exclusion_remain_valid(tmp_path):
    root = tmp_path / "bindings"
    root.mkdir()
    space = root / "retired"
    space.mkdir()
    (space / "old.json").write_text("old run's unfinished document")
    (root / "unrelated.json").write_text(json.dumps({"path": "/another/run/checkpoint"}))
    assert retirement.check_bindings([root], space=space, digests={}) == 1
