"""Shared render-bound metadata is read once per distinct path (#1247)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant.tessera_joint_aura import _prepare_file_read_bound


def test_shared_render_path_is_statted_once(tmp_path, monkeypatch):
    small = tmp_path / "small.pt"
    large = tmp_path / "large.pt"
    small.write_bytes(b"a" * 3)
    large.write_bytes(b"b" * 7)
    data = SimpleNamespace(cells={
        "a": {"render": str(small)}, "b": {"render": str(small)},
        "c": {"render": str(large)}, "d": {"render": str(large)},
    })
    calls = {small: 0, large: 0}
    original = Path.stat

    def count_stat(path, *args, **kwargs):
        if path in calls:
            calls[path] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", count_stat)
    assert _prepare_file_read_bound(data, max_render_bytes=7) == 7
    assert calls == {small: 1, large: 1}


def test_render_bound_rechecks_sizes_on_each_call(tmp_path):
    render = tmp_path / "render.pt"
    render.write_bytes(b"abc")
    data = SimpleNamespace(cells={"a": {"render": str(render)},
                                  "b": {"render": str(render)}})
    assert _prepare_file_read_bound(data, max_render_bytes=7) == 3
    render.write_bytes(b"1234567")
    assert _prepare_file_read_bound(data, max_render_bytes=7) == 7
    with pytest.raises(ValueError, match="original render shard exceeds"):
        _prepare_file_read_bound(data, max_render_bytes=6)


def test_missing_distinct_render_still_refuses(tmp_path):
    valid = tmp_path / "valid.pt"
    valid.write_bytes(b"abc")
    missing = tmp_path / "missing.pt"
    data = SimpleNamespace(cells={"a": {"render": str(valid)},
                                  "b": {"render": str(valid)},
                                  "c": {"render": str(missing)}})
    with pytest.raises(FileNotFoundError):
        _prepare_file_read_bound(data, max_render_bytes=7)


def test_zero_length_render_still_refuses(tmp_path):
    render = tmp_path / "empty.pt"
    render.write_bytes(b"")
    data = SimpleNamespace(cells={"a": {"render": str(render)}})
    with pytest.raises(ValueError, match="original render shard exceeds"):
        _prepare_file_read_bound(data, max_render_bytes=7)
