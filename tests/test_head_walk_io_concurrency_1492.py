"""Explicit head I/O width is independent of the admitted CPU count (#1492)."""
from __future__ import annotations

import os

import pytest

from prismaquant import tessera_joint_aura as bridge
import tools.build_tessera_selected_cache as tool
from test_selected_cache_plan_allowance_1488 import (
    INPUTS, _Reached, _argv, _write, captured,
)


def test_explicit_io_width_does_not_require_extra_cpu_reservation(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    assert bridge._head_walk_worker_count(8) == 8
    assert bridge._head_walk_worker_count(
        environ={bridge.HEAD_WALK_WORKERS_ENV: "8"}) == 8
    assert bridge._head_walk_worker_count(4,
        environ={bridge.HEAD_WALK_WORKERS_ENV: "8"}) == 4
    assert bridge._head_walk_worker_count(environ={}) == 1


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4", 17])
def test_explicit_io_width_is_a_bounded_integer(monkeypatch, value):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(64)))
    with pytest.raises(ValueError, match="head walk workers"):
        bridge._head_walk_worker_count(value)


def test_selected_cache_forwards_explicit_io_width(tmp_path, captured):
    _plan, _digest, handoff, handoff_sha = _write(tmp_path, {"inputs": INPUTS})
    with pytest.raises(_Reached):
        tool.main(_argv(handoff, handoff_sha, "--head-walk-workers", "8"))
    assert captured["head_walk_workers"] == 8
    assert captured["require_existing_renders"] is True
    assert captured["verify_payloads"] is False


@pytest.mark.parametrize("value", ["0", "-1", "17"])
def test_selected_cache_refuses_invalid_width_before_loader(tmp_path, captured, value):
    _plan, _digest, handoff, handoff_sha = _write(tmp_path, {"inputs": INPUTS})
    with pytest.raises(ValueError, match="head walk workers"):
        tool.main(_argv(handoff, handoff_sha, "--head-walk-workers", value))
    assert not captured


def test_head_resume_io_policy_preserves_other_journal_cpu_limits(tmp_path, monkeypatch):
    from prismaquant.cost_stage_checkpoint import prepare_journal, write_unit

    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    kwargs = dict(stage="fixture", identity={"source": "same"}, qnames=["a", "b"])
    root, seal, _ = prepare_journal(tmp_path / "io", resume=False,
                                   unit_io_workers=8, **kwargs)
    for name in ("a", "b"):
        write_unit(root, stage="fixture", identity_sha256=seal, qname=name,
                   state={"value": name})
    _, _, states = prepare_journal(root, resume=True, unit_io_workers=8, **kwargs)
    assert list(states) == ["a", "b"]
    assert states == {name: {"value": name} for name in ("a", "b")}
    with pytest.raises(ValueError, match="CPU affinity"):
        prepare_journal(tmp_path / "cpu", resume=False, unit_workers=2, **kwargs)
    assert not (tmp_path / "cpu").exists()


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4", 17])
def test_journal_io_width_refuses_before_creating_outputs(tmp_path, value):
    from prismaquant.cost_stage_checkpoint import prepare_journal

    path = tmp_path / "refused"
    with pytest.raises(ValueError, match="unit_io_workers"):
        prepare_journal(path, stage="fixture", resume=False, identity={},
                        qnames=[], unit_io_workers=value)
    assert not path.exists()
