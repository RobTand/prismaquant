"""The public joint-plan loader parses the exact authenticated JSON bytes.

This CPU intake fixture runs no model, projection backend, or GPU gate.
The observer replaces the path only after the real digest has completed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from prismaquant import stage_inputs
from prismaquant.tessera_joint_aura import load_joint_anchor_plan
from test_joint_projection_backend import _plan


def _write_plan(tmp_path):
    expected = _plan(tmp_path)
    raw = json.dumps(expected).encode("utf-8")
    path = tmp_path / "plan.json"
    path.write_bytes(raw)
    return path, expected, raw, hashlib.sha256(raw).hexdigest()


def test_joint_plan_load_does_not_accept_path_replacement_after_real_hash(tmp_path, monkeypatch):
    path, expected, original_raw, digest = _write_plan(tmp_path)
    changed = deepcopy(expected)
    changed["model"] = "another-source-checkpoint"
    replacement = tmp_path / "replacement.json"
    replacement.write_text(json.dumps(changed))
    swaps = []
    real_file_hash = stage_inputs.file_sha256hex
    real_bytes_hash = stage_inputs.bytes_sha256hex

    def replace_once():
        if not swaps:
            replacement.replace(path)
            swaps.append(path)

    def observe_file_hash(value):
        actual = real_file_hash(value)
        if Path(value) == path:
            replace_once()
        return actual

    def observe_owned_hash(raw):
        actual = real_bytes_hash(raw)
        if raw == original_raw:
            replace_once()
        return actual

    monkeypatch.setattr(stage_inputs, "file_sha256hex", observe_file_hash)
    monkeypatch.setattr(stage_inputs, "bytes_sha256hex", observe_owned_hash)

    try:
        loaded = load_joint_anchor_plan(path, digest, projection_runtime=False)
    except ValueError as refused:
        assert any(word in str(refused).lower()
                   for word in ("checksum", "identity mismatch", "owned bytes", "changed"))
    else:
        assert loaded == expected, "joint plan parsed bytes that its supplied digest did not authenticate"

    assert swaps == [path], "the post-digest replacement observer did not run"
    assert json.loads(path.read_text()) == changed


def test_joint_plan_load_keeps_unchanged_public_intake(tmp_path):
    path, expected, _raw, digest = _write_plan(tmp_path)

    assert load_joint_anchor_plan(path, digest, projection_runtime=False) == expected
