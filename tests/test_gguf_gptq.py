"""GPTQ-into-k-quant: frozen scales, OBS error reduction, one math path."""

import numpy as np
import pytest
import torch

from prismaquant.gguf_formats import (
    compute_fields,
    gguf_pack_fields,
    reconstruct_fields,
)
from prismaquant.gguf_gptq import gptq_fields
from prismaquant.native_render_levers import NativeRenderLevers

RECIPE = NativeRenderLevers()

gguf = pytest.importorskip("gguf")


def _problem(seed=0, out_f=64, in_f=512, rows=256):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(rows, in_f // 8, generator=g)
    X = base.repeat_interleave(8, dim=1) + 0.3 * torch.randn(
        rows, in_f, generator=g)
    W = torch.randn(out_f, in_f, generator=g) * torch.rand(
        out_f, 1, generator=g).exp()
    qw = X.float().pow(2).mean(dim=0)
    return W, X, qw


@pytest.mark.parametrize("fmt", ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"])
def test_gptq_reduces_hessian_weighted_error_under_frozen_scales(fmt):
    W, X, qw = _problem()
    f_rtn = compute_fields(W, fmt, col_weights=qw)
    f_gptq = gptq_fields(W, fmt, X, col_weights=qw, levers=RECIPE)

    # Scales are FROZEN: only q may differ.
    for key in f_rtn:
        if key == "q":
            continue
        assert torch.equal(f_rtn[key], f_gptq[key]), (fmt, key)
    assert not torch.equal(f_rtn["q"], f_gptq["q"])

    rtn_w = reconstruct_fields(f_rtn, fmt).reshape(W.shape)
    gptq_w = reconstruct_fields(f_gptq, fmt).reshape(W.shape)
    e_rtn = ((X @ (W - rtn_w).T) ** 2).mean()
    e_gptq = ((X @ (W - gptq_w).T) ** 2).mean()
    assert e_gptq < 0.75 * e_rtn, (fmt, float(e_rtn), float(e_gptq))


@pytest.mark.parametrize("fmt", ["Q2_K", "Q6_K"])
def test_gptq_fields_pack_bit_exact(fmt):
    """GPTQ output lives on the frozen grid: gguf-py's dequantize of the
    packed bytes equals our reconstruction exactly."""
    W, X, qw = _problem(seed=1)
    fields = gptq_fields(W, fmt, X, col_weights=qw, levers=RECIPE)
    packed = gguf_pack_fields(fields, fmt, tuple(W.shape))
    decoded = gguf.quants.dequantize(
        packed, getattr(gguf.GGMLQuantizationType, fmt))
    ours = reconstruct_fields(fields, fmt).reshape(W.shape).numpy()
    np.testing.assert_array_equal(decoded, ours)


def test_gptq_dead_activation_channels_do_not_destroy_weights():
    """Serving-safe dead-channel convention (matches the NVFP4 lane):
    columns with no calibration activation quantize as plain RTN rather
    than being zeroed or corrupted by error propagation."""
    W, X, qw = _problem(seed=2)
    X[:, :64] = 0.0
    qw = X.float().pow(2).mean(dim=0)
    fields = gptq_fields(W, "Q3_K", X, col_weights=qw, levers=RECIPE)
    out = reconstruct_fields(fields, "Q3_K").reshape(W.shape)
    assert out[:, :64].abs().sum() > 0