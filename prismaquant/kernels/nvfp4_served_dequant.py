"""The served NVFP4 activation leg's dequantisation, as two Triton kernels.

``nvfp4_activation_contract._nvfp4_activation_qdq_registered_op`` prices a
served W4A4 activation with the codes ``torch.ops._C.scaled_fp4_quant``
emits, dequantised by the contract's own rule: each 16-element group's stored
UE4M3 byte comes from :func:`~prismaquant.nvfp4_activation_contract.
nvfp4_stored_scale_from_amax`, and each element is
``e2m1(code) * (stored / G)``, zero where the stored byte is zero.  Before
RobTand/prismaquant#1211 that leg ran it as a chain of memory-bound Torch ops
(an FP32 copy of the rows for the group maximum, an int64 code tensor, an
``index_select``, two ``repeat_interleave`` and two ``where``), about 140 bytes
of DRAM traffic per element.  These kernels do the same arithmetic in two
passes.

Neither kernel decides anything the contract owns:

* :func:`group_abs_max` returns each group's ``max(|x|)`` in FP32.  A maximum
  is exact, and bf16 widens to FP32 exactly, so it equals
  ``rows.float().abs().amax(-1)`` bit for bit; a group holding a NaN is NaN,
  as Torch's reduction propagates it.  The scale RULE applied to that maximum
  stays in Torch, in the one shared function the attestation gate compares.
* :func:`dequantize_codes` reads the operator's packed nibbles, the stored
  scale plane and the used scale plane (``stored / G``, divided by the caller
  with the same Torch op as before), and writes ``+-e2m1 * used`` in the output
  dtype.  One FP32 multiply per element, with no add to contract into an FMA,
  then the same round-to-nearest-even narrowing ``Tensor.to`` performs.  The
  E2M1 magnitudes are read from the registry's device table, not a copy.

The rounding decision (which code) is the operator's, and nothing here
re-derives it.  Both kernels are declared in the execution contract as
``ServedQuantizerIdentity.dequant_kernel`` (:data:`KERNEL_ID`); a process that
cannot import this module cannot bind the registered-operator arithmetic.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

#: The implementation name a priced row carries in
#: ``ServedQuantizerIdentity.dequant_kernel``.  Bump the suffix whenever either
#: kernel's arithmetic or its read of the operator's packed layout changes.
KERNEL_ID = "prismaquant.triton_nvfp4_served_dequant.v1"

GROUP_SIZE = 16

#: Output dtypes the dequantiser writes directly.  Any other float dtype is
#: written as FP32 and narrowed by ``Tensor.to``, which is what the Torch
#: composition did for every dtype.
_DIRECT_OUTPUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

_ABS_MAX_GROUPS_PER_PROGRAM = 128
_DEQUANT_ELEMENTS_PER_PROGRAM = 2048


@triton.jit
def _group_abs_max_kernel(x_ptr, out_ptr, n_groups,
                          GROUPS: tl.constexpr, SIZE: tl.constexpr):
    group = tl.program_id(0).to(tl.int64) * GROUPS + tl.arange(0, GROUPS)
    live = group < n_groups
    offsets = group[:, None] * SIZE + tl.arange(0, SIZE)[None, :]
    values = tl.load(x_ptr + offsets, mask=live[:, None], other=0.0)
    magnitude = tl.abs(values.to(tl.float32))
    maximum = tl.max(magnitude, axis=1)
    # ``tl.max`` does not promise NaN propagation; Torch's ``amax`` does.
    has_nan = tl.max((magnitude != magnitude).to(tl.int32), axis=1) != 0
    maximum = tl.where(has_nan, float("nan"), maximum)
    tl.store(out_ptr + group, maximum, mask=live)


@triton.jit
def _dequantize_codes_kernel(packed_ptr, stored_ptr, used_ptr, table_ptr,
                             out_ptr, n_elements,
                             BLOCK: tl.constexpr, SIZE: tl.constexpr):
    element = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = element < n_elements
    # Element 2i is the LOW nibble of packed byte i, element 2i + 1 the high
    # one: the order the operator packs and the Torch leg unpacked
    # (``torch.stack((b & 0xF, b >> 4), -1)``).
    byte = tl.load(packed_ptr + element // 2, mask=live, other=0).to(tl.int32)
    code = (byte >> ((element % 2).to(tl.int32) * 4)) & 0xF
    magnitude = tl.load(table_ptr + (code & 0x7), mask=live, other=0.0)
    # The sign is the code's bit 3 moved onto the float's sign bit, so the
    # negative-zero code (8) yields -0.0 as ``-magnitude`` does in Torch.
    # Triton's unary minus is ``0 - x``, which turns -(+0.0) into +0.0.
    sign = ((code >> 3) & 0x1).to(tl.uint32) << 31
    signed = (magnitude.to(tl.uint32, bitcast=True) ^ sign).to(tl.float32, bitcast=True)
    group = element // SIZE
    stored = tl.load(stored_ptr + group, mask=live, other=0.0)
    used = tl.load(used_ptr + group, mask=live, other=0.0)
    value = tl.where(stored != 0, signed * used, 0.0)
    tl.store(out_ptr + element, value.to(out_ptr.dtype.element_ty), mask=live)


def _require_cuda_contiguous(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "cuda" or not tensor.is_contiguous():
        raise ValueError(
            f"{KERNEL_ID}: {name} must be a contiguous CUDA tensor, got "
            f"device={tensor.device} contiguous={tensor.is_contiguous()}")


def group_abs_max(rows: torch.Tensor) -> torch.Tensor:
    """``rows.float().abs().amax(-1)`` over 16-element groups, as ``[M, K/16]``.

    ``rows`` is the contiguous ``[M, K]`` bf16 tensor the operator quantises.
    """
    _require_cuda_contiguous(rows, "rows")
    if rows.dim() != 2 or rows.shape[-1] % GROUP_SIZE:
        raise ValueError(
            f"{KERNEL_ID}: rows must be [M, K] with K divisible by "
            f"{GROUP_SIZE}, got {tuple(rows.shape)}")
    out = torch.empty((rows.shape[0], rows.shape[1] // GROUP_SIZE),
                      dtype=torch.float32, device=rows.device)
    n_groups = out.numel()
    if n_groups:
        grid = (triton.cdiv(n_groups, _ABS_MAX_GROUPS_PER_PROGRAM),)
        _group_abs_max_kernel[grid](rows, out, n_groups,
                                    GROUPS=_ABS_MAX_GROUPS_PER_PROGRAM,
                                    SIZE=GROUP_SIZE, num_warps=4)
    return out


def dequantize_codes(packed: torch.Tensor, stored: torch.Tensor,
                     used: torch.Tensor, table: torch.Tensor,
                     dtype: torch.dtype) -> torch.Tensor:
    """``where(stored != 0, +-table[code & 7] * used, 0)`` as ``[M, K]`` ``dtype``.

    ``packed`` is the operator's ``[M, K/2]`` uint8 nibble plane; ``stored``
    and ``used`` are ``[M, K/16]`` FP32 planes; ``table`` holds the eight
    positive E2M1 magnitudes in FP32.
    """
    for tensor, name in ((packed, "packed"), (stored, "stored"),
                         (used, "used"), (table, "table")):
        _require_cuda_contiguous(tensor, name)
    if packed.dtype != torch.uint8 or packed.dim() != 2:
        raise ValueError(f"{KERNEL_ID}: packed must be [M, K/2] uint8, got "
                         f"{packed.dtype} {tuple(packed.shape)}")
    rows, columns = packed.shape[0], packed.shape[1] * 2
    planes = (rows, columns // GROUP_SIZE)
    if columns % GROUP_SIZE or tuple(stored.shape) != planes or tuple(used.shape) != planes:
        raise ValueError(
            f"{KERNEL_ID}: scale planes {tuple(stored.shape)}/{tuple(used.shape)} "
            f"do not cover packed codes {tuple(packed.shape)}")
    if stored.dtype != torch.float32 or used.dtype != torch.float32:
        raise ValueError(f"{KERNEL_ID}: scale planes must be float32")
    if table.dtype != torch.float32 or table.numel() != 8:
        raise ValueError(f"{KERNEL_ID}: the E2M1 table must be 8 float32 values")
    direct = dtype in _DIRECT_OUTPUT_DTYPES
    out = torch.empty((rows, columns), dtype=dtype if direct else torch.float32,
                      device=packed.device)
    n_elements = out.numel()
    if n_elements:
        grid = (triton.cdiv(n_elements, _DEQUANT_ELEMENTS_PER_PROGRAM),)
        _dequantize_codes_kernel[grid](packed, stored, used, table, out, n_elements,
                                       BLOCK=_DEQUANT_ELEMENTS_PER_PROGRAM,
                                       SIZE=GROUP_SIZE, num_warps=4)
    return out if direct else out.to(dtype)


__all__ = ["GROUP_SIZE", "KERNEL_ID", "dequantize_codes", "group_abs_max"]
