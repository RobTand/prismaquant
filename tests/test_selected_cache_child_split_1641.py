"""The selected-cache CLI splits off the units a bound child cache supplies (#1641).

The GLM-5.3 release pick selects MTP layer 45 jointly with the body, but the
body handoff's census roster has no MTP units and their wires live in a
separate cached-units child. The census rule refuses a non-roster Tessera
unit, correctly, so the CLI must build the handoff manifest from the
selection MINUS the units a byte-bound child names; the two are then composed
by ``tools/compose_tessera_cached_units.py``. These tests pin the split: the
child's units never reach the handoff manifest, every child unit must be a
selected non-BF16 unit, and without ``--child-manifest`` nothing changes.
"""
from __future__ import annotations

import hashlib
import json
import pickle

import pytest

import tools.build_tessera_selected_cache as tool


BODY = {"model.layers.0.mlp.experts.0.down_proj": "TESSERA_E4M3_K1_R1024",
        "model.layers.0.self_attn.o_proj": "BF16"}
MTP = {"model.layers.45.mlp.experts.0.down_proj": "TESSERA_BF16_K1_R1024",
       "model.layers.45.mlp.experts.0.up_proj": "TESSERA_BF16_K1_R1024"}
MTP_PASSTHROUGH = {"model.layers.45.eh_proj": "BF16"}
SELECTED = {**BODY, **MTP, **MTP_PASSTHROUGH}


class _Reached(Exception):
    """The manifest builder was called; the test inspects its assignment."""


def _child(tmp_path, units, *, schema="tessera.cached_units.v1"):
    raw = json.dumps({"schema": schema, "source": {"sha256": "ab" * 32},
                      "units": {name: {"file": f"{name}.tessera"} for name in units}}).encode()
    path = tmp_path / "child.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _handoff(tmp_path):
    raw = pickle.dumps({"provenance": {"tessera_joint_anchors": {
        "plan_sha256": "cd" * 32, "inputs": {"journal": "/j"}}}})
    path = tmp_path / "handoff.pkl"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


@pytest.fixture
def captured(monkeypatch):
    seen = {}
    monkeypatch.setattr(tool, "_bind_selected_assignment",
                        lambda path, digest: ({}, dict(SELECTED), {}))
    monkeypatch.setattr(tool, "load_measured_anchor_input", lambda inputs, **kw: object())

    def builder(assignment, metadata, handoff, data, **kwargs):
        seen["assignment"] = dict(assignment)
        raise _Reached

    monkeypatch.setattr(tool, "selected_cached_units_manifest", builder)
    return seen


def _argv(tmp_path, *extra):
    handoff, handoff_sha = _handoff(tmp_path)
    return ["--handoff", str(handoff), "--handoff-sha256", handoff_sha,
            "--assignment", "layer_config.json", "--assignment-sha256", "0" * 64,
            "--out", "manifest.json", *extra]


def test_child_units_never_reach_the_handoff_manifest(tmp_path, captured):
    child, digest = _child(tmp_path, MTP)
    with pytest.raises(_Reached):
        tool.main(_argv(tmp_path, "--child-manifest", str(child), "--child-manifest-sha256", digest))
    # BF16 passthrough outside the child stays with the handoff selection,
    # where the census rule admits it; only the child's own units leave.
    assert captured["assignment"] == {**BODY, **MTP_PASSTHROUGH}


def test_without_a_child_the_whole_selection_is_built(tmp_path, captured):
    with pytest.raises(_Reached):
        tool.main(_argv(tmp_path))
    assert captured["assignment"] == SELECTED


def test_wrong_child_digest_refuses_before_building(tmp_path, captured):
    child, _digest = _child(tmp_path, MTP)
    with pytest.raises(ValueError, match="child manifest SHA-256"):
        tool.main(_argv(tmp_path, "--child-manifest", str(child), "--child-manifest-sha256", "1" * 64))
    assert not captured


def test_a_child_unit_the_selection_does_not_name_refuses(tmp_path, captured):
    child, digest = _child(tmp_path, {**MTP, "model.layers.45.mlp.experts.9.down_proj": "x"})
    with pytest.raises(ValueError, match="not in the selected assignment.*experts.9"):
        tool.main(_argv(tmp_path, "--child-manifest", str(child), "--child-manifest-sha256", digest))
    assert not captured


def test_a_child_cannot_supply_a_bf16_passthrough(tmp_path, captured):
    child, digest = _child(tmp_path, {**MTP, **MTP_PASSTHROUGH})
    with pytest.raises(ValueError, match="BF16 passthrough.*eh_proj"):
        tool.main(_argv(tmp_path, "--child-manifest", str(child), "--child-manifest-sha256", digest))
    assert not captured


def test_a_child_with_no_units_refuses(tmp_path, captured):
    child, digest = _child(tmp_path, {})
    with pytest.raises(ValueError, match="names no units"):
        tool.main(_argv(tmp_path, "--child-manifest", str(child), "--child-manifest-sha256", digest))
    assert not captured


def test_child_path_and_digest_travel_together(tmp_path, captured):
    child, _digest = _child(tmp_path, MTP)
    with pytest.raises(ValueError, match="child_manifest path and SHA-256"):
        tool.main(_argv(tmp_path, "--child-manifest", str(child)))
    assert not captured
