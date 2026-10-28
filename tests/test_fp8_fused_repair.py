"""Corrective regressions for fused FP8 decoding, dispatch and single-read rows."""
from __future__ import annotations

import re

import pytest
import torch

from prismaquant import fp8_dynamic as fp8
from test_fp8_per_token_qdq_oracle import (
    ELEMENTS,
    PUBLIC_ENTRIES,
    _assert_bit_identical,
    _assert_public_bits,
    _call_public,
    _CudaMetadataTensor,
    _forbidden,
    _fused,
    _devices,
    _real_activations,
    _reference,
    _stub_cuda_preparation,
    needs_cuda_triton,
)


@pytest.mark.parametrize("element", sorted(ELEMENTS))
@pytest.mark.parametrize(
    "capability, eligible",
    (((8, 0), False), ((8, 6), False), ((8, 8), False),
     ((8, 9), True), ((9, 0), True), ((12, 1), True)),
)
def test_fused_eligibility_checks_input_device_capability(monkeypatch, element, capability, eligible):
    queries = []
    rows = _CudaMetadataTensor((1, 4))
    input_device = torch.device("cuda:1")

    def get_capability(device=None):
        queries.append(device)
        return capability

    monkeypatch.setattr(_CudaMetadataTensor, "device", property(lambda self: input_device))
    monkeypatch.setattr(torch.cuda, "get_device_capability", get_capability)
    monkeypatch.setattr(torch.cuda, "current_device", _forbidden)
    assert _fused().fused_eligible(rows, element_dtype=ELEMENTS[element][0]) is eligible
    assert queries == [input_device]


@pytest.mark.parametrize("entry", PUBLIC_ENTRIES, ids=("qdq", "with_dx"))
@pytest.mark.parametrize("element", sorted(ELEMENTS))
@pytest.mark.parametrize("capability", ((8, 0), (8, 8)))
def test_public_unsupported_device_uses_reference_without_cuda(
    monkeypatch, entry, element, capability,
):
    element_dtype, element_max = ELEMENTS[element]
    cpu_values = torch.tensor([[0.0, -0.0, 1.0, -2.0], [448.0, 224.0, -448.0, 0.0]])
    want = _reference(cpu_values, element_dtype=element_dtype, element_max=element_max)
    values = _CudaMetadataTensor(tuple(cpu_values.shape))
    input_device = torch.device("cuda:1")
    queries = []
    reference_calls = []
    real_reference = fp8.fp8_qdq_reference

    def get_capability(device=None):
        queries.append(device)
        return capability

    def reference(*args, **kwargs):
        reference_calls.append(1)
        return real_reference(*args, **kwargs)

    monkeypatch.delenv(fp8.DISABLE_FUSED_ENV, raising=False)
    monkeypatch.setattr(_CudaMetadataTensor, "device", property(lambda self: input_device))
    monkeypatch.setattr(
        _CudaMetadataTensor, "to", lambda self, *args, **kwargs: cpu_values.to(*args, **kwargs),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", get_capability)
    monkeypatch.setattr(torch.cuda, "current_device", _forbidden)
    monkeypatch.setattr(fp8, "fp8_qdq_reference", reference)
    monkeypatch.setattr(_fused(), "fused_per_token_qdq", _forbidden)
    monkeypatch.setattr(_fused(), "_kernel", _forbidden)
    actual = entry(values, element_dtype=element_dtype, element_max=element_max)
    actual, dx = actual if entry is PUBLIC_ENTRIES[1] else (actual, None)
    assert queries == [input_device]
    assert reference_calls == [1]
    _assert_public_bits(entry, actual, dx, cpu_values, want)


@pytest.mark.parametrize("entry", PUBLIC_ENTRIES, ids=("qdq", "with_dx"))
@pytest.mark.parametrize("element", sorted(ELEMENTS))
@pytest.mark.parametrize("capability", ((8, 9), (9, 0), (12, 1)))
def test_public_supported_device_propagates_compiler_failure_without_cuda(
    monkeypatch, entry, element, capability,
):
    element_dtype, element_max = ELEMENTS[element]
    values = _CudaMetadataTensor((1, 4))
    state, _ = _stub_cuda_preparation(monkeypatch)
    error = RuntimeError("supported device compiler failure")

    def fail_kernel():
        raise error

    monkeypatch.delenv(fp8.DISABLE_FUSED_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(_CudaMetadataTensor, "to", _forbidden)
    monkeypatch.setattr(fp8, "fp8_qdq_reference", _forbidden)
    monkeypatch.setattr(_fused(), "_kernel", fail_kernel)
    with pytest.raises(RuntimeError, match="supported device compiler failure") as caught:
        entry(values, element_dtype=element_dtype, element_max=element_max)
    assert caught.value is error
    assert not state["inside"]


def test_register_tile_eligibility_boundaries(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    # No giant allocation: the offset bound must include power-of-two padding.
    element_dtype = torch.float8_e4m3fn
    assert _fused().fused_eligible(
        _CudaMetadataTensor((1, _fused().MAX_COLS)), element_dtype=element_dtype,
    )
    assert not _fused().fused_eligible(
        _CudaMetadataTensor((1, _fused().MAX_COLS + 1)), element_dtype=element_dtype,
    )
    cols = 9000
    rows = ((2**31 - 1) - _fused()._block_size(cols)) // cols + 1
    assert _fused().fused_eligible(_CudaMetadataTensor((rows, cols)), element_dtype=element_dtype)
    assert not _fused().fused_eligible(_CudaMetadataTensor((rows + 1, cols)), element_dtype=element_dtype)


def test_negative_view_is_ineligible_without_cuda():
    rows = torch._neg_view(_CudaMetadataTensor((2, 4)))
    assert rows.is_neg() and rows.is_contiguous()
    assert not _fused().fused_eligible(rows, element_dtype=torch.float8_e4m3fn)


@needs_cuda_triton
@pytest.mark.parametrize("element", sorted(ELEMENTS))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("width", [1, 2, 3, 4, 7, 8, 16, 48])
def test_all_finite_fp8_codes_decode_exactly_at_packing_boundaries(
    monkeypatch, element, dtype, width,
):
    monkeypatch.delenv(fp8.DISABLE_FUSED_ENV, raising=False)
    element_dtype, element_max = ELEMENTS[element]
    codes = torch.arange(256, device="cuda").to(torch.uint8).view(element_dtype)
    values = codes.float()
    values = values[torch.isfinite(values)]
    # Every finite code (including both zero signs and FP8 subnormals) appears
    # at every column position. A max-magnitude anchor gives scale=1 at K>1.
    rows = values[:, None].expand(-1, width).clone()
    if width > 1:
        rows[:, -1] = element_max
        rows = torch.cat([rows.roll(i, dims=1) for i in range(width)])
    rows = rows.to(dtype)
    want = _reference(rows, element_dtype=element_dtype, element_max=element_max)
    monkeypatch.setattr(fp8, "fp8_qdq_reference", _forbidden)
    actual, dx = fp8.fp8_dynamic_activation_qdq_vllm_with_dx(
        rows, element_dtype=element_dtype, element_max=element_max,
    )
    for field in ("quant", "scale", "dequant"):
        _assert_bit_identical(getattr(actual, field), getattr(want, field))
    _assert_bit_identical(dx, want.dequant - rows.float())


@needs_cuda_triton
@pytest.mark.parametrize("entry", PUBLIC_ENTRIES)
def test_public_negative_view_preserves_logical_sign(monkeypatch, entry):
    monkeypatch.delenv(fp8.DISABLE_FUSED_ENV, raising=False)
    values = torch.tensor([[1 + 2j]], device="cuda").conj().imag
    assert values.is_neg() and values.is_contiguous()
    want = _reference(values, element_dtype=torch.float8_e4m3fn, element_max=448.0)
    monkeypatch.setattr(_fused(), "_kernel", _forbidden)
    actual, dx = _call_public(entry, values)
    _assert_public_bits(entry, actual, dx, values, want)


@needs_cuda_triton
@pytest.mark.parametrize("entry", PUBLIC_ENTRIES)
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_public_forward_ad_retains_reference_tangents(monkeypatch, entry, grad_enabled):
    monkeypatch.delenv(fp8.DISABLE_FUSED_ENV, raising=False)
    primal = torch.tensor([[448.0, 224.0]], device="cuda")
    assert not primal.requires_grad
    monkeypatch.setattr(_fused(), "_kernel", _forbidden)
    with torch.set_grad_enabled(grad_enabled), torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(primal, primal.clone())
        with monkeypatch.context() as force:
            force.setenv(fp8.DISABLE_FUSED_ENV, "1")
            want, want_dx = _call_public(entry, dual)
        actual, dx = _call_public(entry, dual)
        for field in ("scale", "dequant"):
            got = torch.autograd.forward_ad.unpack_dual(getattr(actual, field))
            expected = torch.autograd.forward_ad.unpack_dual(getattr(want, field))
            _assert_bit_identical(got.primal, expected.primal)
            assert got.tangent is not None and expected.tangent is not None
            _assert_bit_identical(got.tangent, expected.tangent)
        if dx is not None:
            got = torch.autograd.forward_ad.unpack_dual(dx)
            expected = torch.autograd.forward_ad.unpack_dual(want_dx)
            _assert_bit_identical(got.primal, expected.primal)
            _assert_bit_identical(got.tangent, expected.tangent)


@needs_cuda_triton
@pytest.mark.parametrize("entry", PUBLIC_ENTRIES)
@pytest.mark.parametrize("element", sorted(ELEMENTS))
@pytest.mark.parametrize("element_max", [1e38, 1e6])
def test_public_noncanonical_maximum_preserves_reference(monkeypatch, entry, element, element_max):
    monkeypatch.delenv(fp8.DISABLE_FUSED_ENV, raising=False)
    element_dtype, _ = ELEMENTS[element]
    # A zero row exercises the subnormal scale floor; nonzero values exercise
    # torch's overflow/NaN cast semantics beyond the format's canonical max.
    values = torch.tensor([[0.0] * 4, [1.0, 0.0, -1.0, 2.0]], device="cuda")
    with monkeypatch.context() as force:
        force.setenv(fp8.DISABLE_FUSED_ENV, "1")
        want = entry(values, element_dtype=element_dtype, element_max=element_max)
    want, want_dx = want if isinstance(want, tuple) else (want, None)
    monkeypatch.setattr(_fused(), "_kernel", _forbidden)
    actual = entry(values, element_dtype=element_dtype, element_max=element_max)
    actual, dx = actual if isinstance(actual, tuple) else (actual, None)
    for field in ("quant", "scale", "dequant"):
        _assert_bit_identical(getattr(actual, field), getattr(want, field))
    if dx is not None:
        _assert_bit_identical(dx, want_dx)


@needs_cuda_triton
@pytest.mark.parametrize("element", sorted(ELEMENTS))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fused_special_values_and_dx_match_both_formats(element, dtype):
    element_dtype, element_max = ELEMENTS[element]
    rows = torch.tensor([
        [1.0, float("nan"), 2.0, -0.0],
        [float("inf"), 1.0, 2.0, 3.0],
        [float("-inf"), 1.0, 2.0, 3.0],
        [0.0, -0.0, 0.0, -0.0],
    ], device="cuda", dtype=dtype)
    want = _reference(rows, element_dtype=element_dtype, element_max=element_max)
    quant, scale, dequant, dx = _fused().fused_per_token_qdq(
        rows, element_dtype=element_dtype, element_max=element_max,
        want_dx=True, dequant_dtype=torch.float32,
    )
    for actual, expected in ((quant, want.quant), (scale, want.scale), (dequant, want.dequant)):
        _assert_bit_identical(actual, expected)
    _assert_bit_identical(dx, want.dequant - rows.float())


@needs_cuda_triton
@pytest.mark.parametrize("element", sorted(ELEMENTS))
@pytest.mark.parametrize("width", [4, 5000, 16384])
@pytest.mark.parametrize("want_dx", [False, True])
def test_compiled_kernel_reads_x_once_without_spills(monkeypatch, element, width, want_dx):
    element_dtype, element_max = ELEMENTS[element]
    rows = torch.linspace(-100, 100, width, device="cuda").to(torch.bfloat16)[None, :]
    kernel = _fused()._kernel()
    compiled = []

    class Capture:
        def __getitem__(self, grid):
            run = kernel[grid]

            def launch(*args, **kwargs):
                result = run(*args, **kwargs)
                compiled.append(result)
                return result

            return launch

    monkeypatch.setattr(_fused(), "_kernel", lambda: Capture())
    _fused().fused_per_token_qdq(
        rows, element_dtype=element_dtype, element_max=element_max,
        want_dx=want_dx, dequant_dtype=torch.float32,
    )
    assert len(compiled) == 1
    code = compiled[0]
    # The sole logical load is unrolled into per-thread vector loads in PTX.
    # There are no input reloads, loops or local-memory spills around stores.
    assert len(re.findall(r"\btt\.load\b", code.asm["ttir"])) == 1
    ptx = code.asm["ptx"]
    loads = list(re.finditer(r"\bld\.global\b", ptx))
    stores = list(re.finditer(r"\bst\.global\b", ptx))
    assert loads and stores and max(m.start() for m in loads) < stores[0].start()
    assert not re.search(r"\b(?:ld|st)\.local\b", ptx)
    assert code.n_spills == 0
    labels = {m.group(1): m.start() for m in re.finditer(r"(?m)^\s*([\w$]+):", ptx)}
    for branch in re.finditer(r"\bbra(?:\.uni)?\s+([\w$]+);", ptx):
        assert labels[branch.group(1)] > branch.start(), "unexpected input load loop"
    print(f"K={width} {element} dx={want_dx}: one tt.load, no PTX reload/loop/spill")


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("element", sorted(ELEMENTS))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["empty", "noncontiguous", "edge_rows"])
def test_public_layout_and_edge_dx_bits(device, element, dtype, layout):
    """Both public entries, including reference-derived dx on all edge rows."""
    element_dtype, element_max = ELEMENTS[element]
    if layout == "empty":
        values = torch.empty((0, 8), device=device, dtype=dtype)
    elif layout == "noncontiguous":
        # Seeded synthetic values, not captured model activations.
        values = _real_activations(3, 16, dtype, device)[:, ::2]
        assert not values.is_contiguous()
    else:
        midpoint = 11.5 if element == "e4m3" else 6.5
        values = torch.tensor([
            [element_max, midpoint, -midpoint, -0.0],
            [0.0, -0.0, 0.0, -0.0],
            [1e-8, -1e-8, 0.0, -0.0],
            [float("nan"), 1.0, -0.0, 2.0],
            [float("inf"), float("-inf"), 1.0, -0.0],
        ], device=device, dtype=dtype)
    want = _reference(values, element_dtype=element_dtype, element_max=element_max)
    for entry in PUBLIC_ENTRIES:
        result = entry(values, element_dtype=element_dtype, element_max=element_max)
        actual, dx = result if isinstance(result, tuple) else (result, None)
        _assert_public_bits(entry, actual, dx, values, want)


@needs_cuda_triton
@pytest.mark.parametrize("entry", PUBLIC_ENTRIES)
@pytest.mark.parametrize("element", sorted(ELEMENTS))
def test_public_rows_above_tile_bound_explicitly_use_reference(monkeypatch, entry, element):
    monkeypatch.delenv(fp8.DISABLE_FUSED_ENV, raising=False)
    element_dtype, element_max = ELEMENTS[element]
    values = _real_activations(2, 16385, torch.float32, "cuda")
    want = _reference(values, element_dtype=element_dtype, element_max=element_max)
    monkeypatch.setattr(_fused(), "_kernel", _forbidden)
    result = entry(values, element_dtype=element_dtype, element_max=element_max)
    actual, dx = result if isinstance(result, tuple) else (result, None)
    _assert_public_bits(entry, actual, dx, values, want)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("entry", PUBLIC_ENTRIES)
def test_public_reverse_ad_matches_reference_graph(monkeypatch, device, entry):
    monkeypatch.delenv(fp8.DISABLE_FUSED_ENV, raising=False)
    values = torch.tensor([[448.0, 224.0, -1.0]], device=device, requires_grad=True)

    def differentiated():
        result, dx = _call_public(entry, values)
        loss = result.scale.sum() + result.dequant.sum()
        if dx is not None:
            loss = loss + dx.sum()
        return result, dx, torch.autograd.grad(loss, values)[0]

    with monkeypatch.context() as force:
        force.setenv(fp8.DISABLE_FUSED_ENV, "1")
        want, want_dx, want_grad = differentiated()
    monkeypatch.setattr(_fused(), "_kernel", _forbidden)
    actual, dx, grad = differentiated()
    _assert_public_bits(entry, actual, dx, values, want)
    if dx is not None:
        _assert_bit_identical(dx, want_dx)
    _assert_bit_identical(grad, want_grad)
