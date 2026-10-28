"""Oracle: the fused per-token FP8 QDQ is the torch reference, bit for bit.

PQ #1398 fuses the joint-statistics hook's per-token FP8 activation QDQ
(about ten torch ops per observed expert invocation) into one Triton launch
that also writes the optional full-precision ``dx``. The fused leg is the
default path on CUDA; the tests below pin both legs to the same bits.

CPU tier (runs everywhere): the refactored default path keeps the native
constants, the scale floor, signed zero, ties, saturation, empty rows and
the profiler section, and the kill switch keeps the kernel off the call.
CUDA tier (needs CUDA and Triton): the fused launch matches the reference on
seeded synthetic activations across GLM widths, dtypes, both element types,
-0, ties, zero rows, saturation, NaN/Inf and whole-row tiles up to K=16384.

Reused from the #2666 corrected handoff ecaf0114 (research provenance only).
All numerical fixtures and formulas are in this checkout; no historical git
object or fleet file is loaded at test time. Feature lookup is deferred so an
unreconciled checkout fails with contract assertions, not collection errors.
"""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from prismaquant import fp8_dynamic as fp8
from prismaquant.fp8_dynamic import fp8_dynamic_activation_qdq_vllm


def fp8_dynamic_activation_qdq_vllm_with_dx(*args, **kwargs):
    entry = getattr(fp8, "fp8_dynamic_activation_qdq_vllm_with_dx", None)
    assert callable(entry), "the #2666 optional FP32 dx entry has not been reconciled"
    return entry(*args, **kwargs)


_REFERENCE = getattr(fp8, "fp8_qdq_reference", None)


def fp8_qdq_reference(*args, **kwargs):
    # Keep the real oracle callable even when a dispatch test spies on its name.
    entry = _REFERENCE
    assert callable(entry), "the retained torch reference must be named fp8_qdq_reference"
    return entry(*args, **kwargs)


def _fused():
    name = "prismaquant.kernels.fp8_per_token_qdq"
    assert importlib.util.find_spec(name) is not None, "the existing #2666 fused module is absent"
    return importlib.import_module(name)

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
ELEMENTS = {
    "e4m3": (E4M3, 448.0),
    "e5m2": (E5M2, 57344.0),
}

# Representative replay widths plus awkward widths. The repaired kernel
# keeps a whole-row power-of-two tile through K=16384; wider rows use reference.
WIDTHS = (256, 512, 1536, 2048, 4096, 16384)
AWKWARD_WIDTHS = (48, 80, 5000)
PUBLIC_ENTRIES = (
    fp8_dynamic_activation_qdq_vllm,
    fp8_dynamic_activation_qdq_vllm_with_dx,
)


def _call_public(entry, values):
    actual = entry(values)
    return actual if entry is fp8_dynamic_activation_qdq_vllm_with_dx else (actual, None)


def _assert_public_bits(entry, actual, dx, values, want):
    for field in ("quant", "scale", "dequant"):
        _assert_bit_identical(
            getattr(actual, field).reshape(getattr(want, field).shape),
            getattr(want, field),
        )
    if entry is fp8_dynamic_activation_qdq_vllm_with_dx:
        assert dx is not None and dx.shape == values.shape
        _assert_bit_identical(dx, want.dequant.reshape(values.shape) - values.float())
    else:
        assert dx is None


def _forbidden(*args, **kwargs):
    raise AssertionError("unexpected fused launch or reference dispatch")


class _LaunchSpy:
    """Observe construction, grid selection and launch; optionally delegate."""

    def __init__(self, kernel=None, *, observe=None, error=None):
        self.kernel = kernel
        self.observe = observe or (lambda stage: None)
        self.error = error
        self.grids = []
        self.calls = []

    def __call__(self):
        self.observe("kernel")
        return self

    def __getitem__(self, grid):
        self.observe("grid")
        self.grids.append(tuple(grid))
        run = self.kernel[grid] if self.kernel is not None else None

        def launch(*args, **kwargs):
            self.observe("launch")
            self.calls.append((args, kwargs))
            if self.error is not None:
                raise self.error
            if run is not None:
                return run(*args, **kwargs)

        return launch


class _CudaMetadataTensor(torch.Tensor):
    """Meta storage with CUDA metadata: no CUDA allocations or execution."""

    @staticmethod
    def __new__(cls, shape, *, requires_grad=False):
        base = torch.empty(shape, dtype=torch.float32, device="meta")
        return torch.Tensor._make_subclass(cls, base, requires_grad)

    @property
    def device(self):
        return torch.device("cuda:0")

    @property
    def is_cuda(self):
        return True


def _stub_cuda_preparation(monkeypatch):
    """CPU-only device/allocator doubles, NOT a GPU launch or profiler check."""
    state = {"inside": False, "events": []}
    real_empty = torch.empty

    def observe(stage):
        assert state["inside"], f"{stage} escaped the input-device guard"
        state["events"].append(stage)

    @contextmanager
    def device_guard(device):
        assert device == torch.device("cuda:0")
        assert not state["inside"]
        state["inside"] = True
        state["events"].append("enter")
        try:
            yield
        finally:
            state["inside"] = False
            state["events"].append("exit")

    def empty(*args, **kwargs):
        observe("allocate")
        assert torch.device(kwargs["device"]) == torch.device("cuda:0")
        return real_empty(*args, **{**kwargs, "device": "meta"})

    monkeypatch.setattr(torch.cuda, "device", device_guard)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    monkeypatch.setattr(torch, "empty", empty)
    return state, observe


def _refuse_optional_import(monkeypatch, optional, error):
    real_import = builtins.__import__
    attempts = []

    def import_spy(name, globals=None, locals=None, fromlist=(), level=0):
        if name == optional or name.startswith(optional + ".") or optional in (fromlist or ()):
            attempts.append(name)
            raise error
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_spy)
    return attempts


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
    not (
        torch.cuda.is_available()
        and _triton_available()
        and torch.cuda.get_device_capability() >= (8, 9)
    ),
    reason="the fused per-token QDQ needs sm89+ CUDA and Triton",
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
@pytest.mark.parametrize("element", sorted(ELEMENTS))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("entry", PUBLIC_ENTRIES, ids=("qdq", "with_dx"))
def test_native_row_keeps_scale_codes_and_signed_zero(device, element, dtype, entry):
    # Independently captured canonical LFM prefill row 53: amax and columns
    # 232,354,496; -0 is from row 40, column 1819. Provenance:
    # docs/measurements/fp8-native-activation-parity-2026-09-07.md,
    # diagnostic.safetensors SHA256
    # 8452ec28f9b325e52fcc6745ee4aa16d6828374e45650380b3e56ade770fffb2.
    # E4M3 expected codes below were captured natively. E5M2 and FP32 dx
    # expectations are reference-derived on these SAME captured constants;
    # they are not claims of independently captured E5M2 outputs.
    element_dtype, element_max = ELEMENTS[element]
    values = torch.tensor(
        [[0.96875, -0.0908203125, 0.022705078125, 0.36328125, -0.0]],
        dtype=dtype,
        device=device,
    )
    want = _reference(values, element_dtype=element_dtype, element_max=element_max)
    result = entry(values, element_dtype=element_dtype, element_max=element_max)
    actual, dx = result if entry is fp8_dynamic_activation_qdq_vllm_with_dx else (result, None)
    _assert_public_bits(entry, actual, dx, values, want)
    if element == "e4m3":
        _assert_bit_identical(
            actual.scale,
            torch.tensor([[0.0021623882930725813]], dtype=torch.float32, device=device),
        )
        _assert_bit_identical(
            actual.quant,
            torch.tensor([[448.0, -44.0, 11.0, 176.0, -0.0]], dtype=E4M3, device=device),
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


def test_direct_fused_helper_rejects_cpu_input():
    fused = _fused()

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
    fused = _fused()

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


@pytest.mark.parametrize("entry", PUBLIC_ENTRIES, ids=("qdq", "with_dx"))
def test_cpu_public_dispatch_uses_reference_without_fused_preparation(monkeypatch, entry):
    fused = _fused()

    values = _real_activations(8, 256, torch.bfloat16, "cpu").reshape(2, 4, 256)
    want = _reference(values, element_dtype=E4M3, element_max=448.0)
    reference_calls = []

    def reference(*args, **kwargs):
        reference_calls.append(1)
        return fp8_qdq_reference(*args, **kwargs)

    monkeypatch.delenv(fp8.DISABLE_FUSED_ENV, raising=False)
    monkeypatch.setattr(fp8, "fp8_qdq_reference", reference)
    for name in ("fused_eligible", "fused_per_token_qdq", "_kernel"):
        monkeypatch.setattr(fused, name, _forbidden)
    actual, dx = _call_public(entry, values)
    assert reference_calls == [1]
    _assert_public_bits(entry, actual, dx, values, want)


@pytest.mark.parametrize("entry", PUBLIC_ENTRIES, ids=("qdq", "with_dx"))
@pytest.mark.parametrize("optional", ("fp8_per_token_qdq", "triton"))
@pytest.mark.parametrize("error_type", (ImportError, RuntimeError))
def test_cpu_public_dispatch_never_imports_optional_fused_code(
    monkeypatch, entry, optional, error_type,
):
    values = _real_activations(4, 256, torch.bfloat16, "cpu")
    want = _reference(values, element_dtype=E4M3, element_max=448.0)
    monkeypatch.delenv(fp8.DISABLE_FUSED_ENV, raising=False)
    attempts = _refuse_optional_import(monkeypatch, optional, error_type("broken optional import"))
    actual, dx = _call_public(entry, values)
    assert attempts == []
    _assert_public_bits(entry, actual, dx, values, want)


@pytest.mark.parametrize("entry", PUBLIC_ENTRIES, ids=("qdq", "with_dx"))
@pytest.mark.parametrize("optional", ("fp8_per_token_qdq", "triton"))
@pytest.mark.parametrize("error_type", (ImportError, RuntimeError))
def test_public_cuda_metadata_dispatch_propagates_optional_import_failure(
    monkeypatch, entry, optional, error_type,
):
    # CPU-runnable control-flow test only. Even if import moves into _kernel,
    # meta allocations and a fake device guard prevent any real CUDA work.
    fused = _fused()

    values = _CudaMetadataTensor((4, 256))
    monkeypatch.delenv(fp8.DISABLE_FUSED_ENV, raising=False)
    monkeypatch.setattr(fp8, "fp8_qdq_reference", _forbidden)
    # Fail before any fallback scalar allocation could initialize CUDA.
    monkeypatch.setattr(_CudaMetadataTensor, "to", _forbidden)
    monkeypatch.setattr(fused, "_KERNEL", None)
    state, _ = _stub_cuda_preparation(monkeypatch)
    error = error_type("broken optional import")
    attempts = _refuse_optional_import(monkeypatch, optional, error)
    with pytest.raises(error_type, match="broken optional import") as caught:
        _call_public(entry, values)
    assert caught.value is error and attempts
    assert not state["inside"]


@pytest.mark.parametrize(
    "shape, eligible",
    (
        ((0, 512), True),
        ((1, 1), True),
        ((524287, 4096), True),  # largest whole-tile row count below INT_MAX
        ((524288, 4096), False),  # 2**31 elements, with no backing allocation
        ((1, 2**31), False),
        ((3, (2**31 - 1) // 3), False),  # masked tile offsets overflow int32
        ((4, 0), False),
    ),
)
def test_fused_eligibility_bounds_without_cuda_or_large_allocation(monkeypatch, shape, eligible):
    fused = _fused()

    rows = _CudaMetadataTensor(shape)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    monkeypatch.setattr(fused, "triton_available", lambda: True)
    monkeypatch.setattr(fused, "_kernel", _forbidden)
    assert fused.fused_eligible(rows, element_dtype=E4M3) is eligible
    if not eligible:
        with pytest.raises(RuntimeError, match="not eligible"):
            fused.fused_per_token_qdq(rows, element_dtype=E4M3, element_max=448.0)


@pytest.mark.parametrize("grad_enabled", (False, True))
@pytest.mark.parametrize("requires_grad", (False, True))
def test_fused_eligibility_preserves_autograd_contract(monkeypatch, grad_enabled, requires_grad):
    fused = _fused()

    rows = _CudaMetadataTensor((4, 256), requires_grad=requires_grad)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    monkeypatch.setattr(fused, "triton_available", lambda: True)
    with torch.set_grad_enabled(grad_enabled):
        assert fused.fused_eligible(rows, element_dtype=E4M3) is not (
            requires_grad and grad_enabled
        )


@pytest.mark.parametrize("want_dx", (False, True))
@pytest.mark.parametrize("fail_launch", (False, True))
def test_stubbed_launch_guards_preparation_and_uses_cpu_scalars(
    monkeypatch, want_dx, fail_launch,
):
    # This is a host control-flow stub, not numerical GPU or profiler evidence.
    fused = _fused()

    rows = _CudaMetadataTensor((4, 512))[:, ::2]
    assert not rows.is_contiguous()
    prepared_rows = _CudaMetadataTensor(tuple(rows.shape))
    real_full = torch.full
    scalars = []
    state, observe = _stub_cuda_preparation(monkeypatch)

    def contiguous(self, *args, **kwargs):
        observe("contiguous")
        # Model only the host copy boundary; real meta contiguous() internally
        # allocates on meta, not the CUDA placement modeled by this stub.
        return prepared_rows

    def scalar(*args, **kwargs):
        observe("scalar")
        assert kwargs.get("device") == "cpu", "host constants need explicit CPU placement"
        value = real_full(*args, **kwargs)
        assert value.device.type == "cpu" and value.dtype == torch.float32
        scalars.append(value.item())
        return value

    def to_tl(dtype):
        observe("dtype")
        return dtype  # opaque tokens suffice for the non-executing launch stub

    error = RuntimeError("stub launch failed") if fail_launch else None
    spy = _LaunchSpy(observe=observe, error=error)
    monkeypatch.setattr(_CudaMetadataTensor, "contiguous", contiguous)
    monkeypatch.setattr(fused, "fused_eligible", lambda *args, **kwargs: True)
    monkeypatch.setattr(fused, "_kernel", spy)
    monkeypatch.setattr(fused, "_torch_to_tl", to_tl)
    monkeypatch.setattr(torch, "full", scalar)

    def invoke():
        return fused.fused_per_token_qdq(
            rows, element_dtype=E4M3, element_max=448.0,
            want_dx=want_dx, dequant_dtype=torch.float32,
        )

    # An implicit torch.full would allocate on meta and .item() would fail.
    with torch.device("meta"):
        if fail_launch:
            with pytest.raises(RuntimeError, match="stub launch failed") as caught:
                invoke()
            assert caught.value is error
        else:
            quant, scale, dequant, dx = invoke()
            assert quant.shape == dequant.shape == (4, 256)
            assert scale.shape == (4, 1) and quant.dtype == E4M3
            assert dequant.dtype == torch.float32
            assert (dx is not None) is want_dx
            if dx is not None:
                assert dx.shape == rows.shape and dx.dtype == torch.float32
    assert not state["inside"]
    assert state["events"][0] == "enter" and state["events"][-1] == "exit"
    assert state["events"].count("contiguous") == 1
    assert state["events"].count("allocate") == 3 + int(want_dx)
    assert spy.grids == [(4,)] and len(spy.calls) == 1
    args, _ = spy.calls[0]
    assert args[0].is_contiguous() and args[11] is want_dx
    assert scalars == [
        448.0,
        real_full((), 1 / (448 * 512), dtype=torch.float32, device="cpu").item(),
    ]
    assert args[7:9] == tuple(scalars)


def test_no_symbol_still_says_fallback():
    tree = Path(__file__).resolve().parent.parent
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=tree)
    retired = "_".join(("fallback", "fp8"))
    hits = []
    # Tracked text includes tests/docs/scripts, not just package Python. Git
    # omits untracked generated output; never descend into .git or pycache.
    for name in tracked.decode("utf-8").split("\0"):
        if not name or {".git", "__pycache__", ".pytest_cache"}.intersection(Path(name).parts):
            continue
        path = tree / name
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if retired in text:
            hits.append(name)
    assert hits == []


# --------------------------------------------------------------------------
# CUDA tier: the fused launch is the reference, bit for bit.
# --------------------------------------------------------------------------


@needs_cuda_triton
@pytest.mark.parametrize("element", sorted(ELEMENTS))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("width", WIDTHS + AWKWARD_WIDTHS)
def test_fused_leg_matches_reference_across_widths(element, dtype, width):
    fused = _fused()

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
    fused = _fused()

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
    fused = _fused()

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
    fused = _fused()

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
@pytest.mark.parametrize("entry", PUBLIC_ENTRIES, ids=("qdq", "with_dx"))
def test_public_fused_dispatch_uses_one_launch_per_call(monkeypatch, entry):
    # Count real launch-grid applications, not profiler events (a prior CPU
    # profiler session can hide CUDA events). This spy delegates to Triton.
    fused = _fused()

    values = _real_activations(64, 2048, torch.bfloat16, "cuda").reshape(2, 32, 2048)
    want = _reference(values, element_dtype=E4M3, element_max=448.0)
    spy = _LaunchSpy(fused._kernel())
    monkeypatch.delenv(fp8.DISABLE_FUSED_ENV, raising=False)
    monkeypatch.setattr(fused, "_kernel", spy)
    monkeypatch.setattr(fp8, "fp8_qdq_reference", _forbidden)
    actual, dx = _call_public(entry, values)
    assert spy.grids == [(64,)] and len(spy.calls) == 1
    assert spy.calls[0][0][11] is (entry is fp8_dynamic_activation_qdq_vllm_with_dx)
    _assert_public_bits(entry, actual, dx, values, want)


@needs_cuda_triton
@pytest.mark.parametrize("entry", PUBLIC_ENTRIES, ids=("qdq", "with_dx"))
def test_public_cuda_default_device_allows_graph_capture(monkeypatch, entry):
    values = _real_activations(8, 512, torch.bfloat16, "cuda")
    want = _reference(values, element_dtype=E4M3, element_max=448.0)
    monkeypatch.delenv(fp8.DISABLE_FUSED_ENV, raising=False)
    monkeypatch.setattr(fp8, "fp8_qdq_reference", _forbidden)
    # Warm compilation/allocator state outside capture on a side stream.
    stream = torch.cuda.Stream(device=values.device)
    stream.wait_stream(torch.cuda.current_stream(values.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            _call_public(entry, values)
    torch.cuda.current_stream(values.device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.device(values.device):
        # Implicit CUDA scalar allocation followed by .item() is illegal here.
        with torch.cuda.graph(graph):
            actual, dx = _call_public(entry, values)
        graph.replay()
    torch.cuda.synchronize(values.device)
    _assert_public_bits(entry, actual, dx, values, want)


@needs_cuda_triton
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("entry", PUBLIC_ENTRIES, ids=("qdq", "with_dx"))
@pytest.mark.parametrize("fail_launch", (False, True))
def test_public_fused_launch_uses_input_device_stream_and_restores_caller(
    monkeypatch, entry, fail_launch,
):
    fused = _fused()

    caller = torch.cuda.current_device()
    input_device = (caller + 1) % torch.cuda.device_count()
    if torch.cuda.get_device_capability(input_device) < (8, 9):
        pytest.skip("the fused input device needs sm89+ CUDA")
    with torch.cuda.device(input_device):
        values = _real_activations(8, 256, torch.bfloat16, f"cuda:{input_device}")
        want = _reference(values, element_dtype=E4M3, element_max=448.0)
        stream = torch.cuda.Stream(device=input_device)
        stream.wait_stream(torch.cuda.current_stream(input_device))
        kernel = fused._kernel()
    observed = []

    def observe(stage):
        observed.append(stage)
        assert torch.cuda.current_device() == input_device
        assert torch.cuda.current_stream(input_device).cuda_stream == stream.cuda_stream

    error = RuntimeError("deliberate launch failure") if fail_launch else None
    spy = _LaunchSpy(kernel, observe=observe, error=error)
    monkeypatch.delenv(fp8.DISABLE_FUSED_ENV, raising=False)
    monkeypatch.setattr(fused, "_kernel", spy)
    monkeypatch.setattr(fp8, "fp8_qdq_reference", _forbidden)
    # Select a non-default stream on the input device, then deliberately make
    # the OTHER device current. The helper, not the caller, must switch back.
    with torch.cuda.stream(stream), torch.cuda.device(caller):
        assert torch.cuda.current_device() == caller != values.device.index
        if fail_launch:
            with pytest.raises(RuntimeError, match="deliberate launch failure") as caught:
                _call_public(entry, values)
            assert caught.value is error
        else:
            actual, dx = _call_public(entry, values)
        assert torch.cuda.current_device() == caller
    assert torch.cuda.current_device() == caller
    stream.synchronize()
    assert observed == ["kernel", "grid", "launch"]
    assert spy.grids == [(8,)] and len(spy.calls) == 1
    if not fail_launch:
        for tensor in (actual.quant, actual.scale, actual.dequant, dx):
            if tensor is not None:
                assert tensor.device == values.device
        _assert_public_bits(entry, actual, dx, values, want)
