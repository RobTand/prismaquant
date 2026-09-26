"""``PRISMAQUANT_DEV_MODE=1`` skips the campaign checkpoint seal recompute.

The seal gate in ``load_measured_anchor_input`` (``tessera_joint_aura.py``)
recomputes ``canonical_json_sha256_normalized(identity)`` over the merged
campaign checkpoint and compares it to the manifest's declared
``identity_sha256``. On the real 7.2 GB checkpoint that single call measured
302.653 s of a 765.6 s profile (v15 action ``282c61140ba7``, 2026-09-20):
``run/profile.pstats``, ``cost_stage_checkpoint.py:126`` (the function has
since moved to ``digests.py:143``, PQ #1301).

Under the owner's dev-mode directive the seal is run-gate provenance: dev mode
skips the computation, requires the declared digest to be a full 64-hex
string, records a ``[DEV-MODE]`` line naming the digest it used, and hands
that digest to the existing downstream binding. These tests drive the real
loader on the two-unit campaign fixture of ``tests/test_tessera_joint_aura``
and spy on the seal function itself -- they do not mirror a helper.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismaquant import dev_mode
from prismaquant import tessera_joint_aura as bridge
from tests.test_tessera_joint_aura import bind, fixture


def _seal_calls(monkeypatch):
    """Record every seal computation the loader performs, by ``where``."""
    calls = []
    real = bridge.canonical_json_sha256_normalized

    def spy(value, *, where):
        calls.append(where)
        return real(value, where=where)

    monkeypatch.setattr(bridge, "canonical_json_sha256_normalized", spy)
    return calls


def _declare(config, value):
    """Rewrite the checkpoint's declared seal and rebind the file."""
    path = Path(config["merged_checkpoint"]["path"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["identity_sha256"] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    config["merged_checkpoint"] = bind(path)
    return config


def test_certified_intake_still_recomputes_and_verifies_the_seal(tmp_path, monkeypatch):
    config, names, *_ = fixture(tmp_path)
    calls = _seal_calls(monkeypatch)
    data = bridge.load_measured_anchor_input(config, verify_payloads=False)
    assert calls == ["joint anchor input"], "certified mode computes the seal exactly once"
    assert len(data.formats_by_qname) == len(names)


def test_dev_intake_skips_the_seal_computation(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(dev_mode.DEV_MODE_ENV, "1")
    config, names, *_ = fixture(tmp_path)
    calls = _seal_calls(monkeypatch)
    data = bridge.load_measured_anchor_input(config, verify_payloads=False)
    assert calls == [], "dev mode must not recompute the campaign checkpoint seal"
    assert len(data.formats_by_qname) == len(names), "the declared seal still binds the walk"
    warning = capsys.readouterr().out
    assert "[DEV-MODE]" in warning and "campaign checkpoint seal" in warning


def test_dev_intake_refuses_a_malformed_declared_seal(tmp_path, monkeypatch):
    monkeypatch.setenv(dev_mode.DEV_MODE_ENV, "1")
    config, *_ = fixture(tmp_path)
    _declare(config, "not-a-digest")
    with pytest.raises(ValueError, match="campaign checkpoint seal"):
        bridge.load_measured_anchor_input(config, verify_payloads=False)


def test_dev_intake_still_binds_each_unit_to_the_effective_seal(tmp_path, monkeypatch):
    """A well-formed but wrong declared seal is caught by the per-unit fence.

    Dev mode skips only the recompute; ``_load_unit`` still receives the
    effective seal, so envelopes banked under another identity refuse.
    """
    monkeypatch.setenv(dev_mode.DEV_MODE_ENV, "1")
    config, *_ = fixture(tmp_path)
    _declare(config, "0" * 64)
    with pytest.raises((ValueError, RuntimeError), match="identity mismatch"):
        bridge.load_measured_anchor_input(config, verify_payloads=False)
