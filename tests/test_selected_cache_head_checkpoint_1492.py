"""Selected-cache readers can opt into the existing head-walk journal (#1492).

This pins CLI plumbing, not a storage-rate or GPU performance claim. The
reader still declares no PrismaBuild phase and never synthesizes renders.
"""
from __future__ import annotations

import pytest

import tools.build_tessera_selected_cache as tool
from test_selected_cache_plan_allowance_1488 import (
    INPUTS, _Reached, _argv, _write, captured,
)


@pytest.mark.parametrize("resume", [False, True])
def test_head_checkpoint_reaches_existing_loader(tmp_path, captured, resume):
    _plan, _digest, handoff, handoff_sha = _write(tmp_path, {"inputs": INPUTS})
    checkpoint = tmp_path / "selected-head"
    extra = ["--head-checkpoint", str(checkpoint)]
    if resume:
        extra.append("--head-resume")
    with pytest.raises(_Reached):
        tool.main(_argv(handoff, handoff_sha, *extra))
    assert captured["head_checkpoint"] == str(checkpoint)
    assert captured["head_resume"] is resume
    assert captured["progress_phase"] is None
    assert captured["require_existing_renders"] is True
    assert captured["verify_payloads"] is False


def test_head_resume_without_checkpoint_refuses_before_loader(tmp_path, captured):
    _plan, _digest, handoff, handoff_sha = _write(tmp_path, {"inputs": INPUTS})
    with pytest.raises(ValueError, match="head resume requires --head-checkpoint"):
        tool.main(_argv(handoff, handoff_sha, "--head-resume"))
    assert not captured


def test_without_checkpoint_flags_preserves_reader_defaults(tmp_path, captured):
    _plan, _digest, handoff, handoff_sha = _write(tmp_path, {"inputs": INPUTS})
    with pytest.raises(_Reached):
        tool.main(_argv(handoff, handoff_sha))
    assert captured.get("head_checkpoint") is None
    assert captured.get("head_resume", False) is False
    assert captured["progress_phase"] is None
    assert captured["require_existing_renders"] is True
