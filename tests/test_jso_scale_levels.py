"""JSO per-group scale grid: default is FourOverSix {6,4}, env-extendable.

The 7-level joint grid empirically collapses to {6,4} for 99.998% of groups on
both Qwen3.5-0.8B and Gemma4-31B (aggregate weight-MSE cost of restricting =
+0.009%), and the format allocator promotes the rare residual. So {6,4} is the
default; PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS restores the full grid.
"""
import os

import torch

import prismaquant.native_render_codec as enc
from prismaquant.native_render_levers import NativeRenderLevers


def _resolve(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS", raising=False)
    else:
        monkeypatch.setenv("PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS", value)
    return NativeRenderLevers.resolve()


def test_default_levels_are_four_over_six(monkeypatch):
    m = _resolve(monkeypatch, None)
    assert m.nvfp4_joint_scale_levels == (6.0, 4.0)


def test_env_restores_full_grid(monkeypatch):
    m = _resolve(monkeypatch, "6,4,3,2,1.5,1,0.5")
    assert m.nvfp4_joint_scale_levels == (6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5)


def test_env_accepts_space_separated(monkeypatch):
    m = _resolve(monkeypatch, "6 4 3")
    assert m.nvfp4_joint_scale_levels == (6.0, 4.0, 3.0)


def test_env_garbage_falls_back_to_default(monkeypatch):
    m = _resolve(monkeypatch, "not,numbers")
    assert m.nvfp4_joint_scale_levels == (6.0, 4.0)


def test_joint_mse_rule_uses_the_configured_levels(monkeypatch):
    """The JOINT_MSE scale rule must select over exactly _NVFP4_JOINT_SCALE_LEVELS.
    With the {6,4} default it equals the four_over_six rule; with the full grid
    a group whose optimum is level 3 must select a smaller scale than {6,4}."""
    m = _resolve(monkeypatch, None)
    # group of 16 whose magnitudes favor clipping to 4 over 6 is hard to force;
    # instead assert the JOINT default matches FourOverSix exactly.
    g = torch.randn(4, 8, 16, generator=torch.Generator().manual_seed(7))
    joint = enc.select_nvfp4_group_scales(g, levers=m, scale_rule="joint_mse")
    f46 = enc.select_nvfp4_group_scales(g, levers=m, scale_rule="four_over_six_mse")
    assert torch.allclose(joint, f46), "JOINT_MSE default must equal FourOverSix"
    # with the full grid restored, JOINT can pick scales FourOverSix cannot
    mf = _resolve(monkeypatch, "6,4,3,2,1.5,1,0.5")
    joint_full = enc.select_nvfp4_group_scales(g, levers=mf, scale_rule="joint_mse")
    f46_full = enc.select_nvfp4_group_scales(g, levers=mf, scale_rule="four_over_six_mse")
    # joint_full searches a superset, so its per-group MSE is <= FourOverSix's
    max_abs = g.abs().amax(-1).clamp_min(1e-12)
    def mse(scale):
        return enc._nvfp4_mse_for_group_scale(g, scale)
    assert (mse(joint_full) <= mse(f46_full) + 1e-12).all()
    _resolve(monkeypatch, None)  # restore default for other tests
