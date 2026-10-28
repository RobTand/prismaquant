"""FP8 candidate prices must include every scale the native codec emits."""
import math

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import build_candidates
from prismaquant.footprint import format_tensor_payload_breakdown
from prismaquant.native_render_codec import (
    quantize_2d,
    quantize_dequantize_fp8_dynamic_packed,
)
from prismaquant.native_render_levers import NativeRenderLevers


def _weight(shape):
    values = torch.arange(math.prod(shape)).reshape(shape) % 47 - 23
    return values.to(torch.bfloat16) / 64


def _emitted_payload_bytes(weight):
    if weight.ndim == 2:
        emitted = quantize_2d(weight, "FP8_E4M3", levers=NativeRenderLevers())
        return sum(tensor.numel() * tensor.element_size()
                   for tensor in emitted.values())
    quant, scales = quantize_dequantize_fp8_dynamic_packed(weight)
    packed_bytes = sum(tensor.numel() * tensor.element_size()
                       for tensor in (quant, scales))
    # Native export splits the stack into expert Linears before writing them.
    assert packed_bytes == sum(_emitted_payload_bytes(expert) for expert in weight)
    return packed_bytes


@pytest.mark.parametrize("shape", [(32, 32), (1, 32, 32), (2, 32, 32), (3, 17, 64)])
def test_fp8_shape_price_matches_emitted_payload(shape):
    weight = _weight(shape)
    payload_bytes = _emitted_payload_bytes(weight)
    spec = fr.get_format("FP8_E4M3")
    assert spec.memory_bytes_for_shape(shape) == payload_bytes
    assert spec.bits_for_shape(shape) == payload_bytes * 8
    assert format_tensor_payload_breakdown(
        spec, shape, qname="model.layers.0.mlp.experts.down_proj",
    )["tensor_payload_bytes"] == payload_bytes


@pytest.mark.parametrize("shape", [(32, 32), (2, 32, 32)])
def test_allocator_candidate_prices_the_admitted_fp8_payload(shape):
    qname = ("model.layers.0.mlp.experts.down_proj" if len(shape) == 3
             else "model.layers.0.self_attn.o_proj")
    weight = _weight(shape)
    payload_bytes = _emitted_payload_bytes(weight)
    stats = {qname: {
        "h_trace": 1.0,
        "n_params": weight.numel(),
        "out_features": shape[-2],
        "in_features": shape[-1],
        **({"num_experts": shape[0]} if len(shape) == 3 else {}),
    }}
    costs = {qname: {"FP8_E4M3": {
        "weight_mse": 1.0e-4,
        "output_mse": 2.0e-4,
        "output_mse_measured": True,
    }}}
    masks = []
    built = build_candidates(
        stats, costs, [fr.get_format("FP8_E4M3")],
        source_manifest={qname: "bf16"},
        target_profile="vllm_packed_moe", mask_records=masks,
    )
    assert masks == []
    candidate, = built.candidates[qname]
    assert candidate.fmt == "FP8_E4M3"
    assert candidate.memory_bytes == payload_bytes
    assert candidate.bits_per_param == 8 * payload_bytes / weight.numel()
    assert stats[qname]["_memory_bytes_by_format"]["FP8_E4M3"] == payload_bytes


@pytest.mark.parametrize("format_name,shape,expected_scales", [
    pytest.param("FP8_E4M3", (), 0, id="per_channel_scalar"),
    pytest.param("FP8_E4M3", (7,), 7, id="per_channel_vector"),
    pytest.param("FP8_E4M3", (3, 17), 3, id="per_channel_dense"),
    pytest.param("FP8_E4M3", (2, 3, 5, 17), 30, id="per_channel_rank_four"),
    pytest.param("BF16", (2, 3, 17), 0, id="unscaled_packed"),
    pytest.param("MXFP8_E4M3", (), 0, id="grouped_scalar"),
    pytest.param("MXFP8_E4M3", (33,), 2, id="grouped_vector"),
    pytest.param("MXFP8_E4M3", (2, 33), 4, id="grouped_dense"),
    pytest.param("MXFP8_E4M3", (2, 3, 33), 12, id="grouped_packed"),
    pytest.param("FP8_SOURCE", (), 0, id="block_scalar"),
    pytest.param("FP8_SOURCE", (16385,), 2, id="block_vector"),
    pytest.param("FP8_SOURCE", (129, 129), 4, id="block_dense"),
    pytest.param("FP8_SOURCE", (2, 129, 129), 8, id="block_packed"),
])
def test_scale_geometry_preserves_other_tensor_layouts(
    format_name, shape, expected_scales,
):
    assert fr.get_format(format_name).scale_count_for_shape(shape) == expected_scales
