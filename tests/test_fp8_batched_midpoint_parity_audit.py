"""Differential only; a CPU result makes no throughput or serving claim."""
from __future__ import annotations

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant.measure_quant_cost import _batched_quantize


@pytest.mark.parametrize("value", [1.1875, 1.25], ids=["midpoint", "exact-control"])
def test_plain_fp8_batched_cost_uses_registry_reconstruction(value):
    spec = fr.get_format("FP8_E4M3")
    weights = torch.zeros((1, 1, 32), dtype=torch.float32)
    weights[0, 0, :3] = torch.tensor([448.0, value, -value])
    activations = torch.ones((1, 32), dtype=torch.float32)
    activation_hat = spec.activation_quantize_dequantize(activations.clone())
    batched = _batched_quantize(spec, weights.clone())
    registry = torch.stack([
        spec.quantize_dequantize(weight.clone()) for weight in weights
    ])
    reference_output = activations @ weights[0].T
    batched_output = activation_hat @ batched[0].T
    registry_output = activation_hat @ registry[0].T
    batched_mse = (batched_output - reference_output).square().mean()
    registry_mse = (registry_output - reference_output).square().mean()
    assert batched_mse.item() == registry_mse.item(), {
        "batched_mse": batched_mse.item(), "registry_mse": registry_mse.item(),
    }
    assert torch.equal(batched, registry)


@pytest.mark.parametrize("format_name,fp8_dtype,midpoint", [
    ("FP8_E4M3", torch.float8_e4m3fn, 1.1875),
    ("FP8_E5M2", torch.float8_e5m2, 1.125),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_plain_fp8_batch_preserves_each_slice_and_row_scale(
    format_name, fp8_dtype, midpoint, dtype, device,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA codec parity requires an admitted GPU worker")
    spec = fr.get_format(format_name)
    weights = torch.arange(192, dtype=torch.float32, device=device).reshape(2, 3, 32) / 16 - 6
    weights[0, 0, :3] = torch.tensor([
        torch.finfo(fp8_dtype).max, midpoint, -midpoint,
    ], device=device)
    weights[0, 1] *= 0.375
    weights[0, 2].zero_()
    weights[1, 0] *= 1e-20
    weights[1, 1] *= 3
    weights[1, 2] *= 0.125
    weights = weights.to(dtype)
    actual = _batched_quantize(spec, weights.clone())
    expected = torch.stack([
        spec.quantize_dequantize(weight.clone()) for weight in weights
    ])
    assert actual.shape == weights.shape
    assert actual.dtype == dtype
    assert torch.equal(actual, expected)
