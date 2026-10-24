"""Fused per-token FP8 QDQ for the joint-statistics hook (PQ #1398).

The hook's activation QDQ ran as about ten separate torch ops per observed
expert invocation (abs, amax, div, clamp, casts). This module fuses the whole
per-token leg into one Triton launch per call: one program per row reads the
row, reduces its own FP32 scale into registers, then quantizes and writes the
dequantized row, the FP8 codes, the scale, and optionally the full-precision
``dx = dequant_fp32 - x_fp32`` the statistics accumulate consumes.

Arithmetic contract (bitwise against ``fp8_dynamic.fp8_qdq_reference`` on
finite inputs):
- The row scale divides in FP32 by a register denominator (``div.rn``), never
  by a reciprocal multiply, so no scale moves by one ULP across an FP8
  midpoint (the 2026-09-07 native-parity repair). Triton lowers ``/`` to an
  approximate divide, so both divides are ``div.rn.f32`` inline asm.
- The quantize cast is direct with no zero point, so -0 survives.
- The ``dx`` subtract rounds the product first and then subtracts
  (``sub.rn.f32`` asm), like the torch reference. The opaque barrier also
  stops FP fusion from contracting ``codes * scale - tile`` into one FMA,
  whose skipped rounding shows up amplified through the cancellation.
- Clamps use NaN-preserving select chains, and a row that holds a NaN keeps
  a NaN scale, matching torch's clamp/amax propagation. NaN rows are
  otherwise outside the hook's contract; the oracle pins the finite rows.
- Each supported row is loaded once into a power-of-two register tile,
  including wide rows. Wider/unsupported inputs use the torch reference.
- FP8 conversion keeps codes in full-width integer registers and decodes
  their bits for dequantization, avoiding corrupt packed-i8 lowering on
  narrow rows in the admitted toolchain.

``KERNEL_ID`` names this arithmetic. Bump the suffix whenever the kernel's
math or its read of the row layout changes.
"""

from __future__ import annotations

import torch

#: Bump when the kernel arithmetic or row layout changes.
KERNEL_ID = "prismaquant.triton_fp8_per_token_qdq.v2"

#: Minimum register tile; wider rows use a whole-row power-of-two tile.
TILE = 4096
#: Bound register usage of the single-read kernel to the qualified widths.
MAX_COLS = 16384


def _block_size(n_cols: int) -> int:
    return max(TILE, 1 << (n_cols - 1).bit_length())


def _torch_to_tl(dtype: torch.dtype):
    """Map a row dtype to its Triton counterpart, or None when unsupported."""
    import triton.language as tl

    # Triton 3.8 spells the FP8 types ``float8e4m3fn``/``float8e5m2``; 3.7
    # and older spell them ``float8e4nv``/``float8e5`` for the same formats.
    e4m3 = getattr(tl, "float8e4m3fn", getattr(tl, "float8e4nv", None))
    e5m2 = getattr(tl, "float8e5m2", getattr(tl, "float8e5", None))
    table = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }
    if e4m3 is not None:
        table[torch.float8_e4m3fn] = e4m3
    if e5m2 is not None:
        table[torch.float8_e5m2] = e5m2
    return table.get(dtype)


def triton_available() -> bool:
    """True when the Triton compiler imports on this host."""
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


def fused_eligible(
    rows: torch.Tensor,
    *,
    element_dtype: torch.dtype,
) -> bool:
    """Check tensor/layout support, not compiler availability.

    Eligible CUDA calls must surface compiler/import errors rather than silently
    substituting the multi-op reference. This kernel has no autograd backward.
    """
    if (
        not isinstance(rows, torch.Tensor)
        or rows.dim() != 2
        or rows.device.type != "cuda"
        or rows.shape[-1] == 0
    ):
        return False
    if rows.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if element_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return False
    if torch.is_grad_enabled() and rows.requires_grad:
        return False
    # Raw pointers cannot carry logical view bits or forward-AD tangents.
    if rows.is_neg() or torch.autograd.forward_ad.unpack_dual(rows).tangent is not None:
        return False
    # pid * n_cols and the masked tile offsets are signed int32 in Triton.
    # Bound the padded extent too: even masked lanes form pointer offsets.
    n_rows, n_cols = rows.shape
    if n_cols > MAX_COLS:
        return False
    padded_cols = _block_size(n_cols)
    extent = max(n_rows - 1, 0) * n_cols + padded_cols
    if rows.numel() > 2**31 - 1 or extent > 2**31 - 1:
        return False
    return True


_KERNEL = None


def _kernel():
    """Build the row kernel once and reuse it (PQ #1398).

    Rebuilding the ``@triton.jit`` wrapper per call reloads the compiled
    binary every invocation, which costs milliseconds per QDQ.
    """
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _div_rn(a, b):
        # Triton lowers ``/`` on FP32 to an approximate divide, which can move
        # a scale by one ULP and flip exact FP8 ties (PQ #1398). vLLM's native
        # kernel divides with ``div.rn`` (``common.cuh``); match it exactly.
        return tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
            dtype=tl.float32, is_pure=True, pack=1,
        )

    @triton.jit
    def _sub_rn(a, b):
        # ``dx`` must round the product first and then subtract, like the
        # torch reference. The opaque barrier also stops FP fusion from
        # contracting ``codes * scale - tile`` into one FMA (PQ #1398).
        return tl.inline_asm_elementwise(
            "sub.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
            dtype=tl.float32, is_pure=True, pack=1,
        )

    @triton.jit
    def _quantize_fp8(scaled, E4M3: tl.constexpr):
        # Keep the code in a full-width integer register. Native packed-i8
        # conversion corrupts in-register values on the admitted toolchain
        # when they feed dequant as well as a masked store. Duplicate the
        # scalar into the hardware pair, then use only its low byte; this is
        # still a direct FP32 -> FP8 RTNE cast.
        if E4M3:
            return tl.inline_asm_elementwise(
                "{ .reg .b16 q; cvt.rn.satfinite.e4m3x2.f32 q, $1, $1; "
                "cvt.u32.u16 $0, q; and.b32 $0, $0, 255; }",
                "=r,f", [scaled], dtype=tl.uint32, is_pure=True, pack=1,
            )
        else:
            return tl.inline_asm_elementwise(
                "{ .reg .b16 q; cvt.rn.satfinite.e5m2x2.f32 q, $1, $1; "
                "cvt.u32.u16 $0, q; and.b32 $0, $0, 255; }",
                "=r,f", [scaled], dtype=tl.uint32, is_pure=True, pack=1,
            )

    @triton.jit
    def _decode_fp8(bits, E4M3: tl.constexpr):
        # Decode full-width code bits without reintroducing native packed
        # FP8 casts. All finite FP8 values have exact FP32 representations.
        # Decode normal exponents directly; subnormals are small integers
        # times an exact power of two. Reattach the sign as bits to preserve
        # negative zero.
        sign = (bits & 0x80) << 24
        magnitude = bits & 0x7F
        if E4M3:
            mantissa = magnitude & 7
            exponent = magnitude >> 3
            normal = ((exponent + 120) << 23) | (mantissa << 20)
            subnormal = (mantissa.to(tl.float32) * 2.0**-9).to(tl.uint32, bitcast=True)
            decoded = tl.where(exponent == 0, subnormal, normal) | sign
            decoded = tl.where(magnitude == 0x7F, 0x7FFFFFFF, decoded)
        else:
            mantissa = magnitude & 3
            exponent = magnitude >> 2
            normal = ((exponent + 112) << 23) | (mantissa << 21)
            subnormal = (mantissa.to(tl.float32) * 2.0**-16).to(tl.uint32, bitcast=True)
            decoded = tl.where(exponent == 0, subnormal, normal) | sign
            special = tl.where(mantissa == 0, sign | 0x7F800000, 0x7FFFFFFF)
            decoded = tl.where(exponent == 31, special, decoded)
        return decoded.to(tl.float32, bitcast=True)

    @triton.jit
    def _fp8_per_token_qdq_kernel(
        x_ptr,
        quant_ptr,
        scale_ptr,
        dq_ptr,
        dx_ptr,
        n_rows,
        n_cols: tl.constexpr,
        denom,
        min_scale,
        clamp_lo,
        clamp_hi,
        HAS_DX: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        BLOCK: tl.constexpr,
        E4M3: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= n_rows:
            return
        base = pid * n_cols
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        tile = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(tile))
        nan_total = tl.sum((tile != tile).to(tl.int32))
        amax = tl.where(nan_total > 0, float("nan"), amax)
        scale = _div_rn(amax, denom)
        scale = tl.where(scale < min_scale, min_scale, scale)
        # fp8_qdq_reference also clamps the supplied scale, after the row floor.
        scale_floor = tl.full((), 2.0**-127, tl.float32)
        scale = tl.where(scale < scale_floor, scale_floor, scale)
        scaled = _div_rn(tile, tl.full([BLOCK], scale, tl.float32))
        scaled = tl.where(scaled > clamp_hi, clamp_hi, scaled)
        scaled = tl.where(scaled < clamp_lo, clamp_lo, scaled)
        codes = _quantize_fp8(scaled, E4M3)
        dequant = _decode_fp8(codes, E4M3) * scale
        quant_bytes = quant_ptr.to(tl.pointer_type(tl.uint8))
        tl.store(quant_bytes + base + offs, codes.to(tl.uint8), mask=mask)
        tl.store(scale_ptr + pid, scale)
        tl.store(dq_ptr + base + offs, dequant.to(OUT_DTYPE), mask=mask)
        if HAS_DX:
            tl.store(dx_ptr + base + offs, _sub_rn(dequant, tile), mask=mask)

    _KERNEL = _fp8_per_token_qdq_kernel
    return _fp8_per_token_qdq_kernel


def fused_per_token_qdq(
    rows: torch.Tensor,
    *,
    element_dtype: torch.dtype,
    element_max: float,
    want_dx: bool = False,
    dequant_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the fused per-token QDQ over contiguous 2D ``rows``.

    Returns ``(quant, scale, dequant, dx)`` where ``dequant`` has
    ``dequant_dtype`` (default the input dtype) and ``dx`` (FP32,
    ``dequant_fp32 - rows_fp32``) is None unless ``want_dx`` is set. ``dx``
    skips the BF16 round trip the hook's own subtract keeps, so it serves the
    statistics accumulate, not the hook's current ``dx`` line. Caller owns
    residency; the kernel allocates nothing but its outputs.
    """
    if not fused_eligible(rows, element_dtype=element_dtype):
        raise RuntimeError("fused per-token FP8 QDQ is not eligible for this input")
    # Triton uses the current CUDA device/stream, unlike tensor-dispatched
    # torch operations. Explicit output placement alone does not select it.
    # The guard also restores the caller's device if preparation/launch fails.
    with torch.cuda.device(rows.device):
        rows = rows.contiguous() if not rows.is_contiguous() else rows
        n_rows, n_cols = rows.shape
        device = rows.device
        quant = torch.empty((n_rows, n_cols), dtype=element_dtype, device=device)
        scale = torch.empty((n_rows, 1), dtype=torch.float32, device=device)
        dq_dtype = dequant_dtype or rows.dtype
        dequant = torch.empty((n_rows, n_cols), dtype=dq_dtype, device=device)
        dx = (
            torch.empty((n_rows, n_cols), dtype=torch.float32, device=device)
            if want_dx
            else None
        )
        if n_rows == 0:
            return quant, scale, dequant, dx
        kernel = _kernel()
        out_tl = _torch_to_tl(dq_dtype)
        if out_tl is None:  # pragma: no cover - eligible() guards this
            raise RuntimeError("fused per-token FP8 QDQ has no Triton dtype for this input")
        # Match torch's FP32 scalar rounding on the host regardless of the
        # default device. CUDA scalars here would add fills and .item() syncs
        # before the fused launch and would be illegal during graph capture.
        denom = torch.full(
            (), float(element_max), dtype=torch.float32, device="cpu",
        ).item()
        floor = torch.full(
            (), 1.0 / (float(element_max) * 512.0), dtype=torch.float32, device="cpu",
        ).item()
        kernel[(n_rows,)](
            rows,
            quant,
            scale,
            dequant,
            dx if dx is not None else dequant,
            n_rows,
            n_cols,
            denom,
            floor,
            -float(element_max),
            float(element_max),
            want_dx,
            out_tl,
            _block_size(n_cols),
            element_dtype == torch.float8_e4m3fn,
            num_warps=8 if n_cols > TILE else 4,
        )
    return quant, scale, dequant, dx


__all__ = [
    "KERNEL_ID",
    "TILE",
    "fused_eligible",
    "fused_per_token_qdq",
    "triton_available",
]
