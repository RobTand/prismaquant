"""The selected-cache CLI carries the bound plan's encoder allowance (#1488).

``load_measured_anchor_input`` refuses a recorded encoder seal the installed
package cannot re-derive unless the plan that priced the checkpoint named it.
The allocation handoff passes that plan's ``historical_encoder_reuse``; the
selected-cache CLI dropped it, so a release pick priced under a historical
seal could not compose its manifest. These tests pin the plumbing: a plan
bound by digest, joined under by the handoff and carrying the handoff's own
inputs reaches the loader with its allowance; any other plan refuses before
the loader runs; and without ``--plan`` the loader stays strict.
"""
from __future__ import annotations

import hashlib
import json
import pickle

import pytest

import tools.build_tessera_selected_cache as tool


REUSE = {
    "schema": "prismaquant.tessera_joint_aura.historical_encoder_reuse.v1",
    "allowlist": [{"encoder_source_sha256": "08" * 32, "reason": "sealed reuse",
                   "evidence": "finding", "recorded_by": "test",
                   "recorded_unix": 1}],
}
INPUTS = {"merged_journal": {"path": "/journal", "sha256": "ab" * 32}}


class _Reached(Exception):
    """The loader was called; the test inspects what it was given."""


def _write(tmp_path, plan, *, joined_under=None, inputs=INPUTS):
    raw = json.dumps(plan).encode()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    handoff = {"provenance": {"tessera_joint_anchors": {
        "plan_sha256": joined_under or digest, "inputs": inputs}}}
    handoff_raw = pickle.dumps(handoff)
    handoff_path = tmp_path / "handoff.pkl"
    handoff_path.write_bytes(handoff_raw)
    return plan_path, digest, handoff_path, hashlib.sha256(handoff_raw).hexdigest()


@pytest.fixture
def captured(monkeypatch):
    seen = {}
    monkeypatch.setattr(tool, "_bind_selected_assignment", lambda path, digest: ({}, {}, {}))

    def loader(inputs, **kwargs):
        seen.update(kwargs, inputs=inputs)
        raise _Reached

    monkeypatch.setattr(tool, "load_measured_anchor_input", loader)
    return seen


def _argv(handoff, handoff_sha256, *extra):
    return ["--handoff", str(handoff), "--handoff-sha256", handoff_sha256,
            "--assignment", "layer_config.json", "--assignment-sha256", "0" * 64,
            "--out", "manifest.json", *extra]


def test_bound_plan_allowance_reaches_the_loader(tmp_path, captured):
    plan, digest, handoff, handoff_sha = _write(
        tmp_path, {"inputs": INPUTS, "historical_encoder_reuse": REUSE})
    with pytest.raises(_Reached):
        tool.main(_argv(handoff, handoff_sha, "--plan", str(plan), "--plan-sha256", digest))
    assert captured["historical_encoder_reuse"] == REUSE
    assert captured["inputs"] == INPUTS


def test_without_a_plan_the_loader_stays_strict(tmp_path, captured):
    _plan, _digest, handoff, handoff_sha = _write(
        tmp_path, {"inputs": INPUTS, "historical_encoder_reuse": REUSE})
    with pytest.raises(_Reached):
        tool.main(_argv(handoff, handoff_sha))
    assert captured.get("historical_encoder_reuse") is None


def test_wrong_plan_digest_refuses_before_the_loader(tmp_path, captured):
    plan, _digest, handoff, handoff_sha = _write(
        tmp_path, {"inputs": INPUTS, "historical_encoder_reuse": REUSE})
    with pytest.raises(ValueError, match="joint plan SHA-256"):
        tool.main(_argv(handoff, handoff_sha, "--plan", str(plan), "--plan-sha256", "1" * 64))
    assert not captured


def test_a_plan_the_handoff_was_not_joined_under_refuses(tmp_path, captured):
    plan, digest, handoff, handoff_sha = _write(
        tmp_path, {"inputs": INPUTS, "historical_encoder_reuse": REUSE},
        joined_under="2" * 64)
    with pytest.raises(ValueError, match="joined under plan 2{64}"):
        tool.main(_argv(handoff, handoff_sha, "--plan", str(plan), "--plan-sha256", digest))
    assert not captured


def test_plan_inputs_must_be_the_handoffs_inputs(tmp_path, captured):
    plan, digest, handoff, handoff_sha = _write(
        tmp_path, {"inputs": {"merged_journal": {"path": "/other", "sha256": "cd" * 32}},
                   "historical_encoder_reuse": REUSE})
    with pytest.raises(ValueError, match="original inputs differ"):
        tool.main(_argv(handoff, handoff_sha, "--plan", str(plan), "--plan-sha256", digest))
    assert not captured


def test_plan_path_and_digest_travel_together(tmp_path, captured):
    plan, _digest, handoff, handoff_sha = _write(tmp_path, {"inputs": INPUTS})
    with pytest.raises(ValueError, match="plan path and SHA-256"):
        tool.main(_argv(handoff, handoff_sha, "--plan", str(plan)))
    assert not captured


def test_research_proposal_keeps_its_own_pilot_plan(tmp_path, captured):
    plan, digest, handoff, handoff_sha = _write(tmp_path, {"inputs": INPUTS})
    with pytest.raises(ValueError, match="binds its own pilot plan"):
        tool.main(_argv(handoff, handoff_sha, "--plan", str(plan), "--plan-sha256", digest,
                        "--research-proposal", "proposal.json",
                        "--research-proposal-sha256", "3" * 64))
    assert not captured
