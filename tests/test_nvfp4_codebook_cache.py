"""The memoized NVFP4 codebook must be bit-identical to a freshly built one.

``_nvfp4_codebook`` is called once per column-quantize inside the GPTQ render,
and it was rebuilding a constant Python list into a CUDA tensor every time --
2.279 s of a 3.705 s single-Linear render. Caching it is a pure speed change,
which means the bar it has to clear is EXACTNESS, not accuracy: the production
render, the KL validation and the exported bytes must stay the same rendering
(principle 8), so a cached codebook that differed from a fresh one even in the
last bit would be a rendering confound rather than an optimization.

These tests pin three things: the tensor itself is identical, a whole rendered
Linear is identical, and the cache cannot serve a tensor from the wrong device.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prismaquant import export_native_compressed as enc  # noqa: E402


def _fresh(device, dtype=torch.float32):
    """What the function returned before it was memoized."""
    return torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                        device=device, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cached_codebook_is_bit_identical_to_fresh(dtype):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cached = enc._nvfp4_codebook(torch.device(dev), dtype=dtype)
    assert torch.equal(cached.view(torch.uint8), _fresh(dev, dtype).view(torch.uint8))
    assert cached.dtype == dtype
    # Second call must hand back the SAME object -- otherwise it is not caching
    # and the measured win silently disappears.
    assert enc._nvfp4_codebook(torch.device(dev), dtype=dtype) is cached

def test_registry_math_codebook_keeps_fifteen_exact_values():
    from prismaquant import format_registry as fr

    expected = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
                             0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    actual = fr._e2m1_codebook()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_all_packed_bytes_keep_signed_zero_and_nibble_order():
    from prismaquant.kernels.nvfp4_fused import (
        _pack_fp4_indices, nvfp4_dequantize_weight,
    )

    packed = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    indices = torch.stack(((packed & 15).long(), (packed >> 4).long()),
                          dim=-1).reshape(16, 32)
    expected = levels[indices]
    actual = nvfp4_dequantize_weight(
        packed, torch.ones(16, 2), torch.ones(1),
    )
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(enc._round_to_codebook(expected), indices)
    assert torch.equal(enc.pack_fp4_indices(indices, 32), packed)
    assert torch.equal(_pack_fp4_indices(indices, 32), packed)


def test_export_and_kernel_keep_ties_range_and_nonfinite_mapping():
    from prismaquant.kernels.nvfp4_fused import _indices_from_signed_e2m1_values

    values = torch.tensor([0.0, -0.0, 0.25, -0.25, 0.75, -0.75,
                           1.25, -1.25, 1.75, -1.75, 2.5, -2.5,
                           3.5, -3.5, 5.0, -5.0, 6.0, -6.0, 7.0, -7.0,
                           float("inf"), float("-inf"), float("nan")])
    expected = torch.tensor([0, 8, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12,
                             5, 13, 6, 14, 7, 15, 7, 15, 7, 15, 7])
    assert torch.equal(enc._round_to_codebook(values), expected)
    assert torch.equal(_indices_from_signed_e2m1_values(values), expected)


def test_cache_is_keyed_by_device():
    """A cached tensor from the wrong device is a correctness bug, not a slow path."""
    if not torch.cuda.is_available():
        pytest.skip("needs a second device to distinguish")
    cpu = enc._nvfp4_codebook(torch.device("cpu"))
    gpu = enc._nvfp4_codebook(torch.device("cuda:0"))
    assert cpu.device.type == "cpu"
    assert gpu.device.type == "cuda"
    assert torch.equal(cpu, gpu.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="the production render is a GPU path")
def test_full_render_is_unchanged_by_the_cache():
    """End-to-end: the same Linear renders identically with and without the cache.

    This is the assertion that actually matters. It renders through the real
    production path with the shipping levers (gptq + static_act_order + JSO),
    once with the cache primed and once with it cleared and monkeypatched back
    to the original uncached implementation, and requires bitwise equality.
    """
    from prismaquant.production_weight_cache import render_production_weight

    torch.manual_seed(0)
    qname = "probe.linear"
    w = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16) * 0.02
    acts = {qname: torch.randn(128, 1024, device="cuda", dtype=torch.float32)}
    levers = {"gptq": True, "static_act_order": True, "joint_scale_opt": True,
              "gptq_damp_sweep": False, "gptq_fixed_damp": 1.0,
              "nvfp4_scale_rule": "joint_mse", "scale_sweep": False}

    def render():
        return render_production_weight(
            w, "NVFP4", qname=qname, activations=acts, levers=levers)

    with_cache = render()

    original = enc._nvfp4_codebook
    try:
        enc._nvfp4_codebook = lambda device, dtype=torch.float32: _fresh(
            device, dtype)
        without_cache = render()
    finally:
        enc._nvfp4_codebook = original

    assert torch.equal(with_cache, without_cache), (
        "the codebook cache changed the rendered weight; that is a rendering "
        "confound, not a speedup")
