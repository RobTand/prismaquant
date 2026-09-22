"""Stage A startup and timeout hygiene (R10/R11 findings).

1. Adjoint checkpoints have fixed paths and ``mkdir(exist_ok=False)``. R10's
   tail left ``checkpoints/boundary-045``; R11 would have failed with
   FileExistsError about ten GPU-minutes in. Stage A now refuses at startup,
   before any GPU work, when a checkpoint path it writes is occupied.
2. ``PRISMAQUANT_STAGED_RANGE_WAIT_S`` must stay strictly below every phase
   grace the row declares, so a staging stall ends as the reader's staging
   refusal rather than as PrismaBuild's no-progress kill. R10's spec set
   900 s against a 900 s chain grace.

CPU-only.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import dispatch_joint_quanta as dispatch  # noqa: E402
from prismaquant import joint_cost_stage_a as stage_a  # noqa: E402
from prismaquant.joint_adjoint_checkpoints import adjoint_space  # noqa: E402

from test_dispatch_joint_quanta import _adjoint_manifest  # noqa: E402


# --------------------------------------------------- 1. occupied checkpoints

class _GpuWorkStarted(AssertionError):
    pass


class _PastStartupCheck(AssertionError):
    pass


def _start_stage_a(tmp_path, monkeypatch):
    """Run ``run_adjoint_capture`` up to its first input read.

    The CUDA guard is a no-op, and every GPU entry the startup could reach
    raises ``_GpuWorkStarted``. The prepared-anchor read (``_bound``, the
    first step after the startup checks) raises ``_PastStartupCheck``, so a
    run that passes the occupancy check stops there, before any GPU work.
    """
    from prismaquant import gpu_guard, joint_projection_backend, tessera_joint_aura
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *a, **kw: None)

    def gpu_work(*_args, **_kwargs):
        raise _GpuWorkStarted("GPU work started")

    monkeypatch.setattr(joint_projection_backend, "prewarm_projection_backend", gpu_work)
    monkeypatch.setattr(stage_a, "GpuPowerSampler", gpu_work)

    def past_check(*_args, **_kwargs):
        raise _PastStartupCheck("startup checks passed")

    monkeypatch.setattr(tessera_joint_aura, "_bound", past_check)
    # run_adjoint_capture sets this process-wide; register it for restore.
    monkeypatch.setenv(tessera_joint_aura.ACTIVATION_SCALE_ENV, "0")
    root = tmp_path / "campaign"
    config = {"execution": {"production_act_scales": "0"}, "output_root": str(root)}
    return lambda: stage_a.run_adjoint_capture(
        config, plan_sha256="a" * 64,
        prepared={"path": str(tmp_path / "prepared.json"), "sha256": "b" * 64},
        output_root=root, stride=8)


def test_an_occupied_checkpoint_refuses_before_gpu_work(tmp_path, monkeypatch):
    run = _start_stage_a(tmp_path, monkeypatch)
    occupied = adjoint_space(tmp_path / "campaign") / "checkpoints" / "boundary-045"
    occupied.mkdir(parents=True)
    with pytest.raises(stage_a.AdjointIdentityRefused, match="boundary-045"):
        run()


def test_a_renamed_checkpoint_does_not_block_startup(tmp_path, monkeypatch):
    run = _start_stage_a(tmp_path, monkeypatch)
    checkpoints = adjoint_space(tmp_path / "campaign") / "checkpoints"
    (checkpoints / "boundary-045.r10-superseded-20260922").mkdir(parents=True)
    (checkpoints / "boundary-45").mkdir()
    with pytest.raises(_PastStartupCheck):
        run()


def test_an_empty_space_passes_the_check(tmp_path, monkeypatch):
    run = _start_stage_a(tmp_path, monkeypatch)
    with pytest.raises(_PastStartupCheck):
        run()


# ------------------------------------------- 2. staged-range wait vs grace

def _spec(monkeypatch, tmp_path, env):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"container": {"image": "sha256:" + "0" * 64}, "env": env}))
    monkeypatch.setattr(dispatch, "SPEC_PATH", spec)


@pytest.fixture
def campaign(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"output_root": str(tmp_path / "campaign-root")}))
    return {"plan_sha256": "a" * 64, "prepared_sha256": "b" * 64,
            "manifest_sha256": "c" * 64, "scope": {"campaign": "fixture"},
            "read_manifest_sha256": "d" * 64, "plan_path": str(plan_path),
            "prepared_path": "/fixture/prepare/prepared.json",
            "roster_sha256": hashlib.sha256(b"roster\n").hexdigest()}


@pytest.mark.parametrize("wait", ["900", "901", "1800", "inf", "soon"])
def test_a_wait_at_or_above_the_chunk_grace_refuses(tmp_path, monkeypatch, campaign, wait):
    _spec(monkeypatch, tmp_path, {"PRISMAQUANT_STAGED_RANGE_WAIT_S": wait})
    manifest = _adjoint_manifest(tmp_path, campaign)
    with pytest.raises(dispatch.DispatchRefused, match="PRISMAQUANT_STAGED_RANGE_WAIT_S"):
        dispatch.stage_a_argv(manifest, campaign)


@pytest.mark.parametrize("env", [{}, {"PRISMAQUANT_STAGED_RANGE_WAIT_S": "600"},
                                 {"PRISMAQUANT_STAGED_RANGE_WAIT_S": "0"}])
def test_a_wait_below_every_grace_submits(tmp_path, monkeypatch, campaign, env):
    _spec(monkeypatch, tmp_path, env)
    manifest = _adjoint_manifest(tmp_path, campaign)
    argv = dispatch.stage_a_argv(manifest, campaign)
    assert "--progress-phase" in argv


def test_the_wait_bound_reads_the_sealed_spec_with_the_reader_rules():
    from prismaquant.residency_shard_reader import (
        STAGED_RANGE_WAIT_S, staged_range_wait_from_env)
    assert staged_range_wait_from_env({}) == STAGED_RANGE_WAIT_S
    assert staged_range_wait_from_env({"PRISMAQUANT_STAGED_RANGE_WAIT_S": "12.5"}) == 12.5
    with pytest.raises(ValueError):
        staged_range_wait_from_env({"PRISMAQUANT_STAGED_RANGE_WAIT_S": "inf"})
    progress = [("head", 1800), ("chain-000", 900)]
    dispatch.require_staged_wait_below_grace({"env": {}}, progress)
    with pytest.raises(dispatch.DispatchRefused, match="900"):
        dispatch.require_staged_wait_below_grace(
            {"env": {"PRISMAQUANT_STAGED_RANGE_WAIT_S": "900"}}, progress)


def test_both_row_builders_check_the_wait_on_the_parse_they_seal():
    """The quantum rows declare the same 900 s chunk grace as Stage A."""
    source = (Path(dispatch.__file__)).read_text()
    tree = ast.parse(source)
    for name in ("stage_a_argv", "quantum_argv"):
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == name)
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                 and getattr(node.func, "id", None) == "_container_wrap"]
        assert len(calls) == 1, name
        assert [kw.arg for kw in calls[0].keywords] == ["progress"], name


def test_the_default_wait_is_below_the_chunk_grace():
    from prismaquant.residency_shard_reader import STAGED_RANGE_WAIT_S
    assert STAGED_RANGE_WAIT_S < dispatch.CHUNK_PROGRESS_GRACE_S
