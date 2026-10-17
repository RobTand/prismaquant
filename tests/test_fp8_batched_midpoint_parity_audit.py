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
