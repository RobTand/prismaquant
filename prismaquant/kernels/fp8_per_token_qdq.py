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
  midpoint (the 2026-09-07 native-parity repair).
- The quantize cast is direct with no zero point, so -0 survives.
- Clamps use NaN-preserving select chains, and a row that holds a NaN keeps
  a NaN scale, matching torch's clamp/amax propagation. NaN rows are
  otherwise outside the hook's contract; the oracle pins the finite rows.
- Inputs wider than one 4096-element tile take the same two streaming passes
  inside the one launch; rows at or under the tile read once from DRAM.

``KERNEL_ID`` names this arithmetic. Bump the suffix whenever the kernel's
math or its read of the row layout changes.
"""

from __future__ import annotations

import torch

#: Bump when the kernel arithmetic or row layout changes.
KERNEL_ID = "prismaquant.triton_fp8_per_token_qdq.v1"

#: Rows at or under this width load once and quantize from registers.
TILE = 4096


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
    """True when the fused launch can serve 2D ``rows`` for ``element_dtype``."""
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
    return triton_available()


def _kernel():
    """Compile the row kernel on first use (Triton caches specializations)."""
    import triton
    import triton.language as tl

    def _div_rn(a, b):
        # Triton lowers ``/`` on FP32 to an approximate divide, which can move
        # a scale by one ULP and flip exact FP8 ties (PQ #1398). vLLM's native
        # kernel divides with ``div.rn`` (``common.cuh``); match it exactly.
        return tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
            dtype=tl.float32, is_pure=True, pack=1,
        )

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
        FP8_DTYPE: tl.constexpr,
        BLOCK: tl.constexpr,
        SINGLE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= n_rows:
            return
        base = pid * n_cols

        if SINGLE:
            offs = tl.arange(0, BLOCK)
            mask = offs < n_cols
            tile = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            amax = tl.max(tl.abs(tile))
            nan_total = tl.sum((tile != tile).to(tl.int32))
            amax = tl.where(nan_total > 0, float("nan"), amax)
            scale = _div_rn(amax, denom)
            scale = tl.where(scale < min_scale, min_scale, scale)
            scaled = _div_rn(tile, tl.full([BLOCK], scale, tl.float32))
            scaled = tl.where(scaled > clamp_hi, clamp_hi, scaled)
            scaled = tl.where(scaled < clamp_lo, clamp_lo, scaled)
            codes = scaled.to(FP8_DTYPE)
            dequant = codes.to(tl.float32) * scale
            tl.store(quant_ptr + base + offs, codes, mask=mask)
            tl.store(scale_ptr + pid, scale)
            tl.store(dq_ptr + base + offs, dequant.to(OUT_DTYPE), mask=mask)
            if HAS_DX:
                tl.store(dx_ptr + base + offs, dequant - tile, mask=mask)
        else:
            amax = tl.full((), 0.0, tl.float32)
            nan_total = tl.full((), 0, tl.int32)
            for k in range(0, n_cols, BLOCK):
                offs = k + tl.arange(0, BLOCK)
                mask = offs < n_cols
                tile = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
                amax = tl.maximum(amax, tl.max(tl.abs(tile)))
                nan_total = nan_total + tl.sum((tile != tile).to(tl.int32))
            amax = tl.where(nan_total > 0, float("nan"), amax)
            scale = _div_rn(amax, denom)
            scale = tl.where(scale < min_scale, min_scale, scale)
            tl.store(scale_ptr + pid, scale)
            scale_vec = tl.full([BLOCK], scale, tl.float32)
            for k in range(0, n_cols, BLOCK):
                offs = k + tl.arange(0, BLOCK)
                mask = offs < n_cols
                tile = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
                scaled = _div_rn(tile, scale_vec)
                scaled = tl.where(scaled > clamp_hi, clamp_hi, scaled)
                scaled = tl.where(scaled < clamp_lo, clamp_lo, scaled)
                codes = scaled.to(FP8_DTYPE)
                dequant = codes.to(tl.float32) * scale
                tl.store(quant_ptr + base + offs, codes, mask=mask)
                tl.store(dq_ptr + base + offs, dequant.to(OUT_DTYPE), mask=mask)
                if HAS_DX:
                    tl.store(dx_ptr + base + offs, dequant - tile, mask=mask)

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
    fp8_tl = _torch_to_tl(element_dtype)
    if out_tl is None or fp8_tl is None:  # pragma: no cover - eligible() guards this
        raise RuntimeError("fused per-token FP8 QDQ has no Triton dtype for this input")
    denom = torch.full((), float(element_max), dtype=torch.float32).item()
    floor = torch.full(
        (), 1.0 / (float(element_max) * 512.0), dtype=torch.float32,
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
        fp8_tl,
        TILE,
        n_cols <= TILE,
        num_warps=4,
    )
    return quant, scale, dequant, dx


__all__ = [
    "KERNEL_ID",
    "TILE",
    "fused_eligible",
    "fused_per_token_qdq",
    "triton_available",
]
