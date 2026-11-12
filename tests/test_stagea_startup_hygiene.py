"""Stage A refuses occupied checkpoint paths before GPU work.

R10 left checkpoints/boundary-045. R11 would have failed with FileExistsError
during capture. These tests check the startup refusal and its boundary.
"""
from __future__ import annotations

import pytest

from prismaquant import joint_cost_stage_a as stage_a
from prismaquant.joint_adjoint_checkpoints import adjoint_space


class _GpuWorkStarted(AssertionError):
    pass


class _PastStartupCheck(AssertionError):
    pass


def _start_stage_a(tmp_path, monkeypatch):
    """Run ``run_adjoint_capture`` up to its first input read.

    The CUDA guard is a no-op, and every GPU entry the startup could reach
    raises ``_GpuWorkStarted``. The prepared-anchor read (``stage_inputs.bound``, the
    first step after the startup checks) raises ``_PastStartupCheck``, so a
    run that passes the occupancy check stops there, before any GPU work.
    """
    from prismaquant import gpu_guard, joint_projection_backend, stage_inputs, tessera_joint_aura
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *a, **kw: None)

    def gpu_work(*_args, **_kwargs):
        raise _GpuWorkStarted("GPU work started")

    monkeypatch.setattr(joint_projection_backend, "prewarm_projection_backend", gpu_work)
    monkeypatch.setattr(stage_a, "GpuPowerSampler", gpu_work)

    def past_check(*_args, **_kwargs):
        raise _PastStartupCheck("startup checks passed")

    monkeypatch.setattr(stage_inputs, "bound", past_check)
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
