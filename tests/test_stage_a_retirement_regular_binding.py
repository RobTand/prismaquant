"""A binding proof must not open a special file and wait for an unbounded writer."""
from __future__ import annotations

import builtins
import os
from pathlib import Path

import pytest

from prismaquant import stage_a_retirement as retirement


def test_binding_scan_refuses_named_pipe_before_open(tmp_path, monkeypatch):
    root = tmp_path / "bindings"
    root.mkdir()
    pipe = root / "consumer.json"
    os.mkfifo(pipe)
    original_open = builtins.open

    def forbid_pipe(path, *args, **kwargs):
        if Path(path) == pipe:
            raise AssertionError("binding scanner opened a named pipe")
        return original_open(path, *args, **kwargs)

    # Trap the opening instead of blocking an admitted test on a pipe writer.
    monkeypatch.setattr(retirement, "open", forbid_pipe, raising=False)
    with pytest.raises(retirement.RetirementRefused, match="binding") as refused:
        retirement.check_bindings([root], space=tmp_path / "retired", digests={})
    assert str(pipe) in str(refused.value)
    assert pipe.exists()
