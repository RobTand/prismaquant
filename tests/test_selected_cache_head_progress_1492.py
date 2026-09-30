"""Explicit selected-cache progress uses the existing loader contract (#1492).

The submitting action, not this reader, declares a phase and stall allowance.
No flags means no progress; these tests do not measure storage or GPU speed.
"""
from __future__ import annotations

import pytest

import tools.build_tessera_selected_cache as tool
from test_selected_cache_plan_allowance_1488 import (
    INPUTS, _Reached, _argv, _write, captured,
)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_explicit_head_progress_reaches_existing_loader(tmp_path, captured, checkpoint):
    _plan, _digest, handoff, handoff_sha = _write(tmp_path, {"inputs": INPUTS})
    extra = ["--head-progress-phase", "selected-head",
             "--head-progress-allowance-s", "60.5"]
    if checkpoint:
        extra += ["--head-checkpoint", str(tmp_path / "head"), "--head-resume"]
    with pytest.raises(_Reached):
        tool.main(_argv(handoff, handoff_sha, *extra))
    assert captured["progress_phase"] == "selected-head"
    assert captured["progress_allowance_s"] == 60.5
    assert captured["head_resume"] is checkpoint
    assert captured["verify_payloads"] is False
    assert captured["require_existing_renders"] is True


@pytest.mark.parametrize("extra", [
    ["--head-progress-phase", "selected-head"],
    ["--head-progress-allowance-s", "60"],
])
def test_partial_progress_contract_refuses_before_loader(tmp_path, captured, extra):
    _plan, _digest, handoff, handoff_sha = _write(tmp_path, {"inputs": INPUTS})
    with pytest.raises(ValueError, match="phase and allowance must be supplied together"):
        tool.main(_argv(handoff, handoff_sha, *extra))
    assert not captured


@pytest.mark.parametrize("phase, allowance", [
    ("", "30"), ("  ", "30"), ("selected-head", "0"),
    ("selected-head", "-1"), ("selected-head", "nan"),
    ("selected-head", "inf"), ("selected-head", "-inf"),
])
def test_invalid_progress_contract_refuses_before_loader(tmp_path, captured, phase, allowance):
    _plan, _digest, handoff, handoff_sha = _write(tmp_path, {"inputs": INPUTS})
    with pytest.raises(ValueError, match="head progress"):
        tool.main(_argv(handoff, handoff_sha,
                        "--head-progress-phase", phase,
                        "--head-progress-allowance-s=" + allowance))
    assert not captured


def test_absent_progress_flags_preserve_no_phase_default(tmp_path, captured):
    _plan, _digest, handoff, handoff_sha = _write(tmp_path, {"inputs": INPUTS})
    with pytest.raises(_Reached):
        tool.main(_argv(handoff, handoff_sha))
    assert captured["progress_phase"] is None
    assert captured.get("progress_allowance_s") is None
