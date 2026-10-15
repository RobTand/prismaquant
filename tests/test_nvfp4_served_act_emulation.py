"""Serve-faithful NVFP4 activation emulation (audit M18-residual/C1 lever)."""
import os

import pytest
import torch

from prismaquant.format_registry import nvfp4_activation_qdq_served


def _reference(x: torch.Tensor, g: float) -> torch.Tensor:
    """Independent reference mirroring vLLM's ref_nvfp4_quant math:
    sf = fp8(amax/6 * G) stored, elements on the E2M1 grid at scale sf/G."""
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    x2 = x.reshape(-1, x.shape[-1] // 16, 16).double()
    out = torch.zeros_like(x2)
    for i in range(x2.shape[0]):
        for j in range(x2.shape[1]):
            blk = x2[i, j]
            sf = float(
                torch.tensor(min(float(blk.abs().max()) / 6.0 * g, 448.0))
                .to(torch.float8_e4m3fn).float())
            if sf == 0.0:
                continue
            s = sf / g
            q = (blk / s).clamp(-6.0, 6.0)
            d = (q.abs().unsqueeze(-1) - grid).abs()
            # ties toward zero: pick the SMALLER grid value on exact ties
            idx = torch.argmin(d + torch.arange(8) * 1e-12, dim=-1)
            out[i, j] = grid[idx] * torch.sign(q) * s
    return out.reshape(x.shape).float()


def test_matches_reference_convention_g():
    torch.manual_seed(0)
    x = torch.randn(8, 64) * 2.0
    g = 448.0 * 6.0 / float(x.abs().max())
    torch.testing.assert_close(
        nvfp4_activation_qdq_served(x, g), _reference(x, g),
        rtol=0, atol=1e-6)


def test_matches_reference_legacy_g():
    torch.manual_seed(1)
    x = torch.randn(8, 64) * 2.0
    g = 6.0 / float(x.abs().max())
    torch.testing.assert_close(
        nvfp4_activation_qdq_served(x, g), _reference(x, g),
        rtol=0, atol=1e-6)


def test_block_zeroing_under_legacy_g():
    # A block ~2000x below the calibration amax: legacy G puts its stored
    # scale below the FP8 subnormal floor -> the whole block dequants to 0.
    x = torch.zeros(1, 32)
    x[0, :16] = 100.0   # sets calibration-scale via max_abs
    x[0, 16:] = 0.05
    g = 6.0 / 100.0
    y = nvfp4_activation_qdq_served(x, g)
    assert torch.all(y[0, 16:] == 0.0)
    # convention G rescues the same block
    y2 = nvfp4_activation_qdq_served(x, 448.0 * 6.0 / 100.0)
    assert torch.any(y2[0, 16:] != 0.0)


def test_clipping_above_calib_amax_under_convention_g():
    # Serve block exceeding the calibration amax: convention G saturates
    # the stored scale at 448 -> values clip at calib_amax-grid ceiling.
    calib_amax = 10.0
    g = 448.0 * 6.0 / calib_amax
    x = torch.full((1, 16), 40.0)  # 4x above calibration amax
    y = nvfp4_activation_qdq_served(x, g)
    assert float(y.max()) <= calib_amax + 1e-4  # clipped
    # legacy G has 448x headroom: no clipping at 4x
    y2 = nvfp4_activation_qdq_served(x, 6.0 / calib_amax)
    assert float(y2.max()) > 30.0


def test_hook_lever_default_off_and_on(monkeypatch):
    from prismaquant import format_registry as fr
    from prismaquant.perturbed_x_cache import _activation_qdq
    spec = fr.get_format("NVFP4")
    torch.manual_seed(2)
    x = torch.randn(4, 64)
    scales = {"m.q_proj": float(x.abs().max())}
    monkeypatch.delenv("PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES",
                       raising=False)
    base = _activation_qdq(x, spec, scales, "m.q_proj")
    torch.testing.assert_close(
        base, spec.activation_quantize_dequantize(
            x.clamp(-scales["m.q_proj"], scales["m.q_proj"])))
    monkeypatch.setenv("PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES", "1")
    served = _activation_qdq(x, spec, scales, "m.q_proj")
    from prismaquant.native_render_codec import (
        _nvfp4_input_global_scale_from_max_abs,
    )
    g = _nvfp4_input_global_scale_from_max_abs(scales["m.q_proj"])
    torch.testing.assert_close(served, nvfp4_activation_qdq_served(x, g))
