"""Oracle: the fused per-token FP8 QDQ is the torch reference, bit for bit.

PQ #1398 fuses the joint-statistics hook's per-token FP8 activation QDQ
(about ten torch ops per observed expert invocation) into one Triton launch
that also writes the optional full-precision ``dx``. The fused leg is the
default path on CUDA; the tests below pin both legs to the same bits.

CPU tier (runs everywhere): the refactored default path keeps the native
constants, the scale floor, signed zero, ties, saturation, empty rows and
the profiler section, and the kill switch keeps the kernel off the call.
CUDA tier (needs CUDA and Triton): the fused launch matches the reference on
real-shaped activations across GLM widths, dtypes, both element types, -0,
ties, zero rows, saturation, NaN/Inf and the two-phase wide path.
"""

from __future__ import annotations

import pytest
import torch

from prismaquant import fp8_dynamic as fp8
from prismaquant.fp8_dynamic import (
    fp8_dynamic_activation_qdq_vllm,
    fp8_dynamic_activation_qdq_vllm_with_dx,
    fp8_qdq_reference,
)

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
ELEMENTS = {
    "e4m3": (E4M3, 448.0),
    "e5m2": (E5M2, 57344.0),
}

# GLM-5.3 activation widths the replay quantises, plus an awkward width and a
# width past the kernel's single-read tile (TILE = 4096).
WIDTHS = (256, 512, 1536, 2048, 4096, 16384)
AWKWARD_WIDTHS = (48, 80, 5000)


def _devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def _e5m2_usable(device: str) -> bool:
    if device != "cpu":
        return True
    try:
        torch.zeros(4, dtype=torch.float32).to(torch.float8_e5m2)
    except Exception:
        return False
    return True


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


needs_cuda_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and _triton_available()),
    reason="the fused per-token QDQ needs CUDA and Triton",
)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
        tensor.element_size()
    ]
    return tensor.contiguous().view(view)


def _assert_bit_identical(new: torch.Tensor, old: torch.Tensor) -> None:
    assert new.shape == old.shape and new.dtype == old.dtype
    assert torch.equal(_bits(new), _bits(old)), (
        f"{int((_bits(new) != _bits(old)).sum())} of {new.numel()} elements differ"
    )


def _reference(activation, *, element_dtype, element_max):
    act_f = activation.to(torch.float32)
    rows = act_f.reshape(-1, act_f.shape[-1])
    min_scale = 1.0 / (float(element_max) * 512.0)
    denominator = torch.full(
        (), float(element_max), dtype=torch.float32, device=rows.device
    )
    scale = (rows.abs().amax(dim=-1, keepdim=True) / denominator).clamp_min(min_scale)
    return fp8_qdq_reference(
        rows, scale, element_dtype=element_dtype, element_max=element_max
    )


def _real_activations(rows: int, cols: int, dtype: torch.dtype, device: str):
    generator = torch.Generator(device="cpu").manual_seed(1398 + rows + cols)
    values = torch.randn(
        (rows, cols), generator=generator, dtype=torch.float32, device="cpu"
    )
    # LLM-like rows: small bulk with sparse large outliers, plus exact zeros.
    outliers = torch.randint(
        0, cols, (rows, 4), generator=generator, device="cpu"
    )
    values.scatter_(1, outliers, values.gather(1, outliers) * 40.0)
    values[0, : cols // 16] = 0.0
    return values.to(dtype).to(device)


# --------------------------------------------------------------------------
# CPU tier: the refactored default path keeps the reference numerics.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", _devices())
def test_native_row_keeps_scale_codes_and_signed_zero(device):
    # Canonical LFM prefill row 53 and the row-40 signed zero (PQ #1398).
    values = torch.tensor(
        [[0.96875, -0.0908203125, 0.022705078125, 0.36328125, -0.0]],
        dtype=torch.bfloat16,
        device=device,
    )
    actual = fp8_dynamic_activation_qdq_vllm(values)
    assert torch.equal(
        actual.scale,
        torch.tensor([[0.0021623882930725813]], dtype=torch.float32, device=device),
    )
    assert torch.equal(
        actual.quant.view(torch.uint8),
        torch.tensor([[448.0, -44.0, 11.0, 176.0, -0.0]], dtype=E4M3, device=device).view(
            torch.uint8
        ),
    )


@pytest.mark.parametrize("device", _devices())
def test_zero_rows_keep_scale_floor_and_signed_zero(device):
    values = torch.tensor([[0.0, -0.0]], dtype=torch.bfloat16, device=device)
    actual = fp8_dynamic_activation_qdq_vllm(values)
    assert torch.equal(
        actual.scale, torch.full_like(actual.scale, 1 / (448 * 512))
    )
    assert actual.quant.view(torch.uint8).tolist() == [[0, 128]]


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("element", sorted(ELEMENTS))
def test_midpoint_ties_match_reference_bit_for_bit(device, element):
    if not _e5m2_usable(device) and element == "e5m2":
        pytest.skip("this torch build has no CPU E5M2 cast")
    # Unit scale rows put exact FP8 midpoints through the cast: 11.5 ties
    # 11|12 on E4M3, 6.5 ties 6|7 on E5M2. Both legs must pick the same side.
    element_dtype, element_max = ELEMENTS[element]
    if element == "e4m3":
        row = [448.0, 11.5, -11.5, 1.0, 0.0, -0.0]
    else:
        row = [57344.0, 6.5, -6.5, 1.0, 0.0, -0.0]
    values = torch.tensor([row], dtype=torch.bfloat16, device=device)
    actual = fp8_dynamic_activation_qdq_vllm(
        values, element_dtype=element_dtype, element_max=element_max
    )
    want = _reference(values, element_dtype=element_dtype, element_max=element_max)
    _assert_bit_identical(actual.quant.reshape(-1), want.quant.reshape(-1))
    _assert_bit_identical(actual.scale.reshape(-1), want.scale.reshape(-1))
    _assert_bit_identical(actual.dequant.reshape(-1), want.dequant.reshape(-1))


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("element", sorted(ELEMENTS))
def test_saturation_matches_reference_bit_for_bit(device, element):
    if not _e5m2_usable(device) and element == "e5m2":
        pytest.skip("this torch build has no CPU E5M2 cast")
    element_dtype, element_max = ELEMENTS[element]
    values = torch.tensor(
        [[1.0, 100.0, -100.0, 0.0], [0.5, -0.0, 1e-4, -1e-4]],
        dtype=torch.bfloat16,
        device=device,
    )
    actual = fp8_dynamic_activation_qdq_vllm(
        values, element_dtype=element_dtype, element_max=element_max
    )
    want = _reference(values, element_dtype=element_dtype, element_max=element_max)
    assert (actual.quant.float().abs().amax() <= element_max).all()
    _assert_bit_identical(actual.quant.reshape(-1), want.quant.reshape(-1))
    _assert_bit_identical(actual.dequant.reshape(-1), want.dequant.reshape(-1))


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("element", sorted(ELEMENTS))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
def test_real_rows_match_reference_bit_for_bit(device, element, dtype):
    if not _e5m2_usable(device) and element == "e5m2":
        pytest.skip("this torch build has no CPU E5M2 cast")
    element_dtype, element_max = ELEMENTS[element]
    values = _real_activations(37, 2048, dtype, device)
    actual = fp8_dynamic_activation_qdq_vllm(
        values, element_dtype=element_dtype, element_max=element_max
    )
    want = _reference(values, element_dtype=element_dtype, element_max=element_max)
    _assert_bit_identical(actual.quant.reshape(-1), want.quant.reshape(-1))
    _assert_bit_identical(actual.scale.reshape(-1), want.scale.reshape(-1))
    _assert_bit_identical(actual.dequant.reshape(-1), want.dequant.reshape(-1))


@pytest.mark.parametrize("device", _devices())
def test_rank3_and_noncontiguous_match_reference_bit_for_bit(device):
    base = _real_activations(8, 512, torch.bfloat16, device).reshape(2, 4, 512)
    assert not base.transpose(1, 2).is_contiguous()
    for values in (base, base.transpose(1, 2)):
        actual = fp8_dynamic_activation_qdq_vllm(values)
        want = _reference(values, element_dtype=E4M3, element_max=448.0)
        _assert_bit_identical(actual.quant.reshape(-1), want.quant.reshape(-1))
        _assert_bit_identical(actual.scale.reshape(-1), want.scale.reshape(-1))
        _assert_bit_identical(actual.dequant.reshape(-1), want.dequant.reshape(-1))


def test_cpu_input_never_reaches_the_fused_launch():
    from prismaquant.kernels import fp8_per_token_qdq as fused

    rows = _real_activations(4, 256, torch.bfloat16, "cpu").reshape(-1, 256)
    assert not fused.fused_eligible(rows, element_dtype=E4M3)
    with pytest.raises(RuntimeError, match="not eligible"):
        fused.fused_per_token_qdq(rows, element_dtype=E4M3, element_max=448.0)


@pytest.mark.parametrize("device", _devices())
def test_empty_rows_match_reference(device):
    values = torch.empty((0, 512), dtype=torch.bfloat16, device=device)
    actual = fp8_dynamic_activation_qdq_vllm(values)
    want = _reference(values, element_dtype=E4M3, element_max=448.0)
    assert actual.quant.shape == want.quant.shape == (0, 512)
    assert actual.scale.shape == want.scale.shape == (0, 1)
    assert actual.dequant.shape == want.dequant.shape == (0, 512)


@pytest.mark.parametrize("device", _devices())
def test_with_dx_matches_the_fp32_expression_bit_for_bit(device):
    values = _real_activations(17, 1024, torch.bfloat16, device)
    result, dx = fp8_dynamic_activation_qdq_vllm_with_dx(values)
    plain = fp8_dynamic_activation_qdq_vllm(values)
    _assert_bit_identical(result.quant.reshape(-1), plain.quant.reshape(-1))
    _assert_bit_identical(result.scale.reshape(-1), plain.scale.reshape(-1))
    _assert_bit_identical(result.dequant.reshape(-1), plain.dequant.reshape(-1))
    want_dx = result.dequant.reshape(-1, 1024) - values.float().reshape(-1, 1024)
    assert dx.dtype == torch.float32 and dx.shape == values.shape
    _assert_bit_identical(dx.reshape(-1), want_dx.reshape(-1))


def test_kill_switch_keeps_the_kernel_off_the_call(monkeypatch):
    import prismaquant.kernels.fp8_per_token_qdq as fused

    def _boom(*args, **kwargs):
        raise AssertionError("the fused kernel must not run under the kill switch")

    monkeypatch.setattr(fused, "fused_per_token_qdq", _boom)
    monkeypatch.setenv(fp8.DISABLE_FUSED_ENV, "1")
    values = _real_activations(4, 256, torch.bfloat16, "cpu")
    actual = fp8_dynamic_activation_qdq_vllm(values)
    want = _reference(values, element_dtype=E4M3, element_max=448.0)
    _assert_bit_identical(actual.dequant.reshape(-1), want.dequant.reshape(-1))


def test_record_function_isolates_the_qdq_call():
    values = _real_activations(4, 256, torch.bfloat16, "cpu")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        fp8_dynamic_activation_qdq_vllm(values)
    names = {event.key for event in prof.key_averages()}
    assert fp8.RECORD_FUNCTION in names


def test_cpu_input_never_reaches_the_fused_launch():
    from prismaquant.kernels import fp8_per_token_qdq as fused

    rows = _real_activations(4, 256, torch.bfloat16, "cpu").reshape(-1, 256)
    assert not fused.fused_eligible(rows, element_dtype=E4M3)
    with pytest.raises(RuntimeError, match="not eligible"):
        fused.fused_per_token_qdq(rows, element_dtype=E4M3, element_max=448.0)


def test_no_symbol_still_says_fallback():
    import pathlib

    tree = pathlib.Path(__file__).resolve().parent.parent / "prismaquant"
    hits = [
        path.name
        for path in tree.rglob("*.py")
        if "fallback_fp8" in path.read_text(encoding="utf-8")
    ]
    assert hits == []


# --------------------------------------------------------------------------
# CUDA tier: the fused launch is the reference, bit for bit.
# --------------------------------------------------------------------------


@needs_cuda_triton
@pytest.mark.parametrize("element", sorted(ELEMENTS))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("width", WIDTHS + AWKWARD_WIDTHS)
def test_fused_leg_matches_reference_across_widths(element, dtype, width):
    from prismaquant.kernels import fp8_per_token_qdq as fused

    element_dtype, element_max = ELEMENTS[element]
    rows = _real_activations(37, width, dtype, "cuda").reshape(-1, width)
    quant, scale, dequant, dx = fused.fused_per_token_qdq(
        rows,
        element_dtype=element_dtype,
        element_max=element_max,
        want_dx=True,
        dequant_dtype=torch.float32,
    )
    want = _reference(rows, element_dtype=element_dtype, element_max=element_max)
    assert dx is not None
    _assert_bit_identical(quant.reshape(-1), want.quant.reshape(-1))
    _assert_bit_identical(scale.reshape(-1), want.scale.reshape(-1))
    _assert_bit_identical(dequant.reshape(-1), want.dequant.reshape(-1))
    _assert_bit_identical(dx.reshape(-1), (want.dequant - rows.float()).reshape(-1))


@needs_cuda_triton
@pytest.mark.parametrize("rows_count", (1, 129, 1000))
def test_fused_leg_matches_reference_across_row_counts(rows_count):
    from prismaquant.kernels import fp8_per_token_qdq as fused

    rows = _real_activations(rows_count, 2048, torch.bfloat16, "cuda").reshape(
        -1, 2048
    )
    quant, scale, dequant, _ = fused.fused_per_token_qdq(
        rows, element_dtype=E4M3, element_max=448.0, dequant_dtype=torch.float32
    )
    want = _reference(rows, element_dtype=E4M3, element_max=448.0)
    _assert_bit_identical(quant.reshape(-1), want.quant.reshape(-1))
    _assert_bit_identical(scale.reshape(-1), want.scale.reshape(-1))
    _assert_bit_identical(dequant.reshape(-1), want.dequant.reshape(-1))


@needs_cuda_triton
def test_fused_leg_matches_reference_on_nan_and_inf():
    from prismaquant.kernels import fp8_per_token_qdq as fused

    rows = torch.tensor(
        [
            [1.0, float("nan"), 2.0, -0.0],
            [float("inf"), 1.0, 2.0, 3.0],
            [float("-inf"), 1.0, 2.0, 3.0],
            [0.0, -0.0, 0.0, -0.0],
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )
    quant, scale, dequant, _ = fused.fused_per_token_qdq(
        rows, element_dtype=E4M3, element_max=448.0, dequant_dtype=torch.float32
    )
    want = _reference(rows, element_dtype=E4M3, element_max=448.0)
    _assert_bit_identical(scale.reshape(-1), want.scale.reshape(-1))
    _assert_bit_identical(quant.reshape(-1), want.quant.reshape(-1))
    _assert_bit_identical(dequant.reshape(-1), want.dequant.reshape(-1))


@needs_cuda_triton
def test_fused_leg_serves_noncontiguous_and_empty_rows():
    from prismaquant.kernels import fp8_per_token_qdq as fused

    base = _real_activations(8, 512, torch.bfloat16, "cuda")
    wide = base.t()
    assert not wide.is_contiguous()
    quant, scale, dequant, _ = fused.fused_per_token_qdq(
        wide, element_dtype=E4M3, element_max=448.0, dequant_dtype=torch.float32
    )
    want = _reference(wide, element_dtype=E4M3, element_max=448.0)
    _assert_bit_identical(quant.reshape(-1), want.quant.reshape(-1))
    _assert_bit_identical(dequant.reshape(-1), want.dequant.reshape(-1))

    empty = torch.empty((0, 512), dtype=torch.bfloat16, device="cuda")
    quant, scale, dequant, dx = fused.fused_per_token_qdq(
        empty,
        element_dtype=E4M3,
        element_max=448.0,
        want_dx=True,
        dequant_dtype=torch.float32,
    )
    assert quant.shape == (0, 512) and scale.shape == (0, 1)
    assert dequant.shape == (0, 512) and dx is not None and dx.shape == (0, 512)


@needs_cuda_triton
def test_fused_leg_uses_one_launch_per_call(monkeypatch):
    # The profiler cannot count launches here: a CPU-only profiler session
    # earlier in the process stops later sessions from seeing CUDA launches.
    # Spy the launch grid instead: one application at exactly ``(M,)``.
    from prismaquant.kernels import fp8_per_token_qdq as fused

    real_kernel = fused._kernel()

    class _GridSpy:
        def __init__(self):
            self.grids = []

        def __getitem__(self, grid):
            self.grids.append(tuple(grid))
            return real_kernel[grid]

    spy = _GridSpy()
    monkeypatch.setattr(fused, "_kernel", lambda: spy)
    rows = _real_activations(64, 2048, torch.bfloat16, "cuda").reshape(-1, 2048)
    quant, scale, dequant, _ = fused.fused_per_token_qdq(
        rows, element_dtype=E4M3, element_max=448.0, dequant_dtype=torch.float32
    )
    assert spy.grids == [(64,)]
    want = _reference(rows, element_dtype=E4M3, element_max=448.0)
    _assert_bit_identical(quant.reshape(-1), want.quant.reshape(-1))
    _assert_bit_identical(dequant.reshape(-1), want.dequant.reshape(-1))
