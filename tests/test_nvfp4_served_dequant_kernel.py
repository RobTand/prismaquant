"""The served NVFP4 activation leg's fused dequantisation (RobTand/prismaquant#1211).

WHAT THIS HOLDS.

* The fused leg's output is BIT-identical to the Torch composition it replaced
  (``_nvfp4_activation_qdq_registered_op_unfused``), compared as raw bits, not
  only with ``torch.equal``: a signed zero or a NaN payload that moved would
  pass ``torch.equal`` and still be a different tensor.
* The group-maximum kernel under the shared scale rule is the priced plane
  (``_nvfp4_registered_stored_plane``), so the leg carries one scale rule.
* The kernel is declared: a registered-operator identity names it, a binding
  without it or naming another is refused, and a process that cannot load it
  refuses rather than pricing with another implementation.

The host-sync half of #1211 is ``tests/test_nvfp4_served_qdq_no_sync.py``.

WHERE EACH PART RUNS.  The identity and dispatch tests are CPU tests.  The
kernel tests need CUDA and Triton and use a stand-in for the registered
operator (PrismaQuant's own E2M1 code rule, packed the way the operator packs):
the property under test is the dequantisation of whatever codes arrive, which
does not depend on who chose them.  The ``real_operator`` tests need the vLLM
extension and run in the campaign image; they are the same comparisons with
the operator the serve runs.
"""
from __future__ import annotations

import types

import pytest
import torch

from nvfp4_served_qdq_fixtures import (
    G16, activation, cpu_kernels, real_operator_registered, stand_in_operator)
from prismaquant import nvfp4_activation_contract as owner

# GLM-5.3 activation widths the replay quantises: hidden 4096, q_lora 1536,
# kv_lora 512, MoE intermediate 2048, o_proj input 64 x 256.  Row counts cover
# one row, rows that are not a multiple of the operator's 128-row tile, and a
# chunk.
GLM53_WIDTHS = (512, 1536, 2048, 4096, 16384)
ROW_COUNTS = (1, 37, 128, 129, 1000)
# Awkward widths: one group, three groups (K/16 not a multiple of 4).
PADDING_WIDTHS = (16, 48, 80)
# G values: unit, the retained 84-group case, a campaign-shaped
# 448 * 6 / amax, a large power of two whose used scales are FP32 subnormal,
# and a small one.
G_VALUES = (1.0, 3.5720930099487305, 448.0 * 6.0 / 11.375, 2.0 ** 126, 0.015625)


def _registered_identity(**overrides) -> owner.ServedQuantizerIdentity:
    fields = {"backend": owner.SERVED_QUANTIZER_BACKEND_REGISTERED_OP,
              "op": owner.SERVED_QUANTIZER_OP, "platform": "sm_121",
              "torch": "2.13.0+cu130", "vllm": "0.1.dev20073",
              "image_content_sha256": "d" * 64,
              "dequant_kernel": owner.SERVED_QUANTIZER_DEQUANT_KERNEL}
    fields.update(overrides)
    return owner.ServedQuantizerIdentity(**fields)


# --------------------------------------------------------------------------
# CPU: the kernel is declared in the identity and fails closed.
# --------------------------------------------------------------------------


def test_a_registered_identity_names_the_dequant_kernel(monkeypatch):
    owner._reset_served_quantizer_identity_for_tests()
    monkeypatch.setattr(owner, "_register_served_quantizer_op", lambda: True)
    monkeypatch.setattr(owner, "_served_dequant_kernels", cpu_kernels)

    identity = owner.resolve_served_quantizer_identity(require=True, context="test")

    assert identity.dequant_kernel == owner.SERVED_QUANTIZER_DEQUANT_KERNEL
    assert identity.as_record()["dequant_kernel"] == owner.SERVED_QUANTIZER_DEQUANT_KERNEL
    # Recorded, not a reuse axis: the kernels are bit-identical to the Torch
    # composition, so a retained cost priced by either is the same number.
    assert "dequant_kernel" not in owner.SERVED_QUANTIZER_REUSE_AXES


def test_an_unloadable_kernel_refuses_the_registered_binding(monkeypatch):
    owner._reset_served_quantizer_identity_for_tests()
    monkeypatch.setattr(owner, "_register_served_quantizer_op", lambda: True)

    def _unloadable():
        raise owner.ServedQuantizerUnboundError("no triton here")

    monkeypatch.setattr(owner, "_served_dequant_kernels", _unloadable)

    resolved = owner.resolve_served_quantizer_identity(require=False, context="test")
    assert resolved.backend == owner.SERVED_QUANTIZER_BACKEND_REGISTERED_OP
    assert resolved.dequant_kernel is None
    with pytest.raises(owner.ServedQuantizerUnboundError, match="dequantisation kernel"):
        owner.resolve_served_quantizer_identity(require=True, context="test")
    with pytest.raises(owner.ServedQuantizerUnboundError, match="dequant_kernel"):
        owner.bind_served_quantizer_identity(require=True, context="test")


def test_a_binding_without_or_with_another_kernel_is_refused():
    owner._reset_served_quantizer_identity_for_tests()
    with pytest.raises(owner.ServedQuantizerUnboundError, match="dequant_kernel"):
        owner.bind_served_quantizer_identity(
            identity=_registered_identity(dequant_kernel=None), require=True,
            context="test")
    owner._reset_served_quantizer_identity_for_tests()
    with pytest.raises(owner.ServedQuantizerUnboundError, match="nothing else"):
        owner.bind_served_quantizer_identity(
            identity=_registered_identity(dequant_kernel="torch_elementwise"),
            require=True, context="test")


def test_a_contract_naming_another_kernel_is_refused_at_dispatch(monkeypatch):
    """An identity that reached a contract directly is checked again."""
    calls: list[tuple] = []
    monkeypatch.setattr(owner, "_nvfp4_activation_qdq_registered_op",
                        lambda x, g: calls.append((x, g)))
    contract = owner.StaticActivationContract(
        measured_as_served=True,
        served_quantizer=_registered_identity(dequant_kernel=None))

    with pytest.raises(owner.ServedQuantizerUnboundError, match="dequantisation kernel"):
        contract.quantize_dequantize(torch.zeros(1, G16), 1.0)
    assert calls == []


def test_a_kernel_module_under_another_name_is_refused(monkeypatch):
    import prismaquant.kernels as kernels_pkg

    fake = types.SimpleNamespace(KERNEL_ID="prismaquant.something_else.v9")
    monkeypatch.setattr(owner, "_SERVED_DEQUANT_KERNELS", None)
    monkeypatch.setitem(__import__("sys").modules,
                        "prismaquant.kernels.nvfp4_served_dequant", fake)
    monkeypatch.setattr(kernels_pkg, "nvfp4_served_dequant", fake, raising=False)

    with pytest.raises(owner.ServedQuantizerUnboundError, match="something_else"):
        owner._served_dequant_kernels()


def test_the_guards_still_run_first_and_in_order():
    with pytest.raises(ValueError, match="divisible by"):
        owner._nvfp4_activation_qdq_registered_op(torch.zeros(2, 30), 0.0)
    with pytest.raises(ValueError, match="finite and > 0"):
        owner._nvfp4_activation_qdq_registered_op(torch.zeros(2, 32), float("nan"))
    with pytest.raises(owner.ServedQuantizerUnboundError, match="(?i)cuda"):
        owner._nvfp4_activation_qdq_registered_op(torch.zeros(2, 32), 1.0)


# --------------------------------------------------------------------------
# GPU: the kernels against the Torch composition, bit for bit.
# --------------------------------------------------------------------------


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


needs_cuda_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and _triton_available()),
    reason="the fused dequantisation kernels need CUDA and Triton")


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
        tensor.element_size()]
    return tensor.contiguous().view(view)


def _assert_bit_identical(new: torch.Tensor, old: torch.Tensor) -> None:
    assert new.shape == old.shape and new.dtype == old.dtype
    assert new.device == old.device
    assert torch.equal(_bits(new), _bits(old)), (
        f"{int((_bits(new) != _bits(old)).sum())} of {new.numel()} elements differ")


@needs_cuda_triton
def test_group_abs_max_is_the_torch_maximum_bit_for_bit():
    from prismaquant.kernels import nvfp4_served_dequant as kernels

    special = torch.tensor(
        [-0.0, 0.0, float("inf"), -float("inf"), float("nan"), -float("nan"),
         2.0 ** -133, -(2.0 ** -126), 3.0e38, -1.0],
        dtype=torch.float32).to(torch.bfloat16)
    for rows, width in [(1, 16), (37, 48), (129, 4096), (1000, 16384)]:
        x = activation(rows, width, seed=rows + width)
        flat = x.view(-1)
        slots = flat[::17][: special.numel()]                # spread over groups
        flat[::17][: special.numel()] = special.cuda()[: slots.numel()]
        x[-1, :G16] = -0.0                                   # a negative-zero group
        if rows > 1:
            # A whole group of the smallest bf16 subnormal, so a subnormal
            # IS a group maximum: a flush to zero would show here.
            x[0, :G16] = 2.0 ** -133
        got = kernels.group_abs_max(x)
        want = x.float().reshape(rows, width // G16, G16).abs().amax(-1)
        if rows > 1:
            assert want[0, 0].item() == 2.0 ** -133            # the row is what it claims
        assert torch.equal(torch.isnan(got), torch.isnan(want))
        finite = ~torch.isnan(want)
        assert torch.equal(_bits(got[finite]), _bits(want[finite]))


@needs_cuda_triton
@pytest.mark.parametrize("g", G_VALUES)
def test_the_kernel_maximum_under_the_shared_rule_is_the_priced_plane(g):
    """``rule(group_abs_max(rows))`` is ``_nvfp4_registered_stored_plane``."""
    from prismaquant.kernels import nvfp4_served_dequant as kernels

    x = activation(129, 4096, seed=7)
    x[5, :G16] = float("nan")
    got = owner.nvfp4_stored_scale_from_amax(kernels.group_abs_max(x), g).float()
    want = owner._nvfp4_registered_stored_plane(x, g).reshape(got.shape)
    assert torch.equal(torch.isnan(got), torch.isnan(want))
    finite = ~torch.isnan(want)
    assert torch.equal(_bits(got[finite]), _bits(want[finite]))


@needs_cuda_triton
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("g", G_VALUES)
def test_dequantize_codes_is_the_torch_composition_bit_for_bit(dtype, g):
    """Every code (both zeros included) under real stored planes, zero groups
    included, written straight into the output dtype."""
    from prismaquant.kernels import nvfp4_served_dequant as kernels

    rows, width = 129, 2048
    generator = torch.Generator(device="cuda").manual_seed(11)
    packed = torch.randint(0, 256, (rows, width // 2), device="cuda",
                           dtype=torch.uint8, generator=generator)
    amax = torch.rand(rows, width // G16, device="cuda", generator=generator)
    amax = amax * torch.exp2(torch.randint(-20, 12, amax.shape, device="cuda",
                                           generator=generator).float())
    amax[:, 1] = 0.0                                          # zero-scale groups
    stored = owner.nvfp4_stored_scale_from_amax(amax, g).float()
    used = stored / g

    got = kernels.dequantize_codes(packed, stored, used,
                                   owner._e2m1_positive_table(packed.device), dtype)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(rows, width)
    want = owner._nvfp4_dequantize_registered_codes(codes, stored, g).to(dtype)
    _assert_bit_identical(got, want)
    assert torch.equal(got, want)


#: Edge rows for the code-to-value step, each an (amax, G) the shared scale
#: rule turns into a known stored and used scale (fable review D1, D2):
#: ``6 * 2**-135`` at ``G = 2**126`` gives stored ``2**-9`` (the smallest
#: e4m3 subnormal) and used ``2**-135``, so the used scale and every nonzero
#: product are FP32 subnormals; ``2**20`` at ``G = 1/64`` clamps to stored
#: 448 and used 28672, so code 7 gives 172032, past the fp16 maximum.
EDGE_ROWS = {
    "fp32_subnormal": (6.0 * 2.0 ** -135, 2.0 ** 126, 2.0 ** -9, 2.0 ** -135),
    "fp16_overflow": (2.0 ** 20, 1.0 / 64.0, 448.0, 28672.0),
}


@needs_cuda_triton
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("row", sorted(EDGE_ROWS))
def test_dequantize_codes_edge_rows_are_the_torch_composition_bit_for_bit(dtype, row):
    """Every code under a stored scale the rule puts at an FP32 edge."""
    from prismaquant.kernels import nvfp4_served_dequant as kernels

    amax_value, g, stored_value, used_value = EDGE_ROWS[row]
    rows, width = 3, 512
    # Every byte value, so every code appears in both nibbles.
    packed = (torch.arange(rows * width // 2, device="cuda") % 256).to(
        torch.uint8).reshape(rows, width // 2)
    amax = torch.full((rows, width // G16), amax_value, device="cuda")
    amax[1, ::3] = 0.0                                        # zero-scale groups
    stored = owner.nvfp4_stored_scale_from_amax(amax, g).float()
    used = stored / g
    live = amax != 0
    # The rule lands where the row claims, before any comparison.
    assert torch.all(stored[live] == stored_value)
    assert torch.all(used[live] == used_value)
    if row == "fp32_subnormal":
        tiny = torch.finfo(torch.float32).tiny
        assert 0.0 < used_value < tiny and 6.0 * used_value < tiny

    got = kernels.dequantize_codes(packed, stored, used,
                                   owner._e2m1_positive_table(packed.device), dtype)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(rows, width)
    want = owner._nvfp4_dequantize_registered_codes(codes, stored, g).to(dtype)
    _assert_bit_identical(got, want)
    if row == "fp16_overflow" and dtype == torch.float16:
        assert torch.isinf(want).any() and torch.isinf(got).any()
    if row == "fp32_subnormal" and dtype == torch.float32:
        nonzero = want != 0
        assert nonzero.any()
        assert torch.all(want[nonzero].abs() < torch.finfo(torch.float32).tiny)


@needs_cuda_triton
@pytest.mark.parametrize("width", GLM53_WIDTHS + PADDING_WIDTHS)
def test_the_fused_leg_is_the_torch_leg_bit_for_bit(monkeypatch, width):
    monkeypatch.setattr(torch, "ops", stand_in_operator())
    for rows in ROW_COUNTS:
        x = activation(rows, width, seed=rows * 31 + width)
        for g in G_VALUES:
            _assert_bit_identical(
                owner._nvfp4_activation_qdq_registered_op(x, g),
                owner._nvfp4_activation_qdq_registered_op_unfused(x, g))


@needs_cuda_triton
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
def test_the_fused_leg_keeps_the_input_dtype_and_shape(monkeypatch, dtype):
    monkeypatch.setattr(torch, "ops", stand_in_operator())
    x = activation(2 * 37, 1536, seed=3, dtype=dtype).reshape(2, 37, 1536)
    for g in G_VALUES:
        got = owner._nvfp4_activation_qdq_registered_op(x, g)
        assert got.shape == x.shape and got.dtype == dtype
        _assert_bit_identical(got, owner._nvfp4_activation_qdq_registered_op_unfused(x, g))


@needs_cuda_triton
def test_nan_and_inf_activations_match_bit_for_bit(monkeypatch):
    monkeypatch.setattr(torch, "ops", stand_in_operator())
    x = activation(37, 4096, seed=5)
    x[1, 7] = float("nan")
    x[2, 100] = -float("nan")
    x[4, 3000] = float("inf")
    x[6, 64] = -float("inf")
    for g in G_VALUES:
        _assert_bit_identical(owner._nvfp4_activation_qdq_registered_op(x, g),
                              owner._nvfp4_activation_qdq_registered_op_unfused(x, g))


@needs_cuda_triton
def test_an_empty_activation_is_allocated_by_both_legs(monkeypatch):
    monkeypatch.setattr(torch, "ops", stand_in_operator())
    for shape in ((0, 4096), (2, 0, 1536)):
        x = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        got = owner._nvfp4_activation_qdq_registered_op(x, 1.0)
        want = owner._nvfp4_activation_qdq_registered_op_unfused(x, 1.0)
        assert got.shape == want.shape == x.shape and got.dtype == want.dtype


# --------------------------------------------------------------------------
# The operator the serve runs (campaign image only).
# --------------------------------------------------------------------------


real_operator = pytest.mark.skipif(
    not (_triton_available() and real_operator_registered()),
    reason="needs the vLLM extension's registered scaled_fp4_quant on CUDA")


@real_operator
@pytest.mark.parametrize("width", GLM53_WIDTHS + PADDING_WIDTHS)
def test_real_operator_fused_leg_is_the_torch_leg_bit_for_bit(width):
    for rows in ROW_COUNTS:
        x = activation(rows, width, seed=rows * 17 + width)
        for g in G_VALUES:
            _assert_bit_identical(
                owner._nvfp4_activation_qdq_registered_op(x, g),
                owner._nvfp4_activation_qdq_registered_op_unfused(x, g))


@real_operator
def test_real_operator_edge_rows_are_bit_identical():
    """The real operator's codes on activations that reach the D1/D2 edges.

    A group of the smallest bf16 subnormal at ``G = 2**126`` has stored
    ``2**-9`` and used ``2**-135``, an FP32 subnormal; an fp16 group of
    65504 at ``G = 1/64`` rounds (as bf16) to 65536, a stored scale of 176
    and a used scale of 11264, so code 7 overflows fp16.
    """
    x = activation(4, 512, seed=4)
    x[1, :G16] = 2.0 ** -133
    x[2, 16:32] = -(2.0 ** -133)
    got = owner._nvfp4_activation_qdq_registered_op(x, 2.0 ** 126)
    want = owner._nvfp4_activation_qdq_registered_op_unfused(x, 2.0 ** 126)
    _assert_bit_identical(got, want)
    edge = want[1, :G16].float()                               # the row reached the edge
    assert (edge != 0).all() and (edge.abs() < torch.finfo(torch.float32).tiny).all()

    half = activation(4, 512, seed=5, dtype=torch.float16)
    half[1, :G16] = 65504.0
    half[2, 16:32] = -65504.0
    got = owner._nvfp4_activation_qdq_registered_op(half, 1.0 / 64.0)
    want = owner._nvfp4_activation_qdq_registered_op_unfused(half, 1.0 / 64.0)
    _assert_bit_identical(got, want)
    assert torch.isinf(want[1, :G16]).all() and torch.isinf(want[2, 16:32]).all()


@real_operator
def test_real_operator_chunk_is_bit_identical():
    """One replay chunk at the campaign's ``chunk_rows`` and hidden width."""
    x = activation(65536, 4096, seed=65536)
    g = 448.0 * 6.0 / float(x.float().abs().amax())
    _assert_bit_identical(owner._nvfp4_activation_qdq_registered_op(x, g),
                          owner._nvfp4_activation_qdq_registered_op_unfused(x, g))
