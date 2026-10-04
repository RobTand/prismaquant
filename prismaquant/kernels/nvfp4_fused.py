"""Fused NVFP4 activation-quant + matmul proof-of-concept.

The reference cost path models NVFP4 as E2M1 values with exact FP32
per-16-column scales. The pack helper below preserves that math and stores the
E2M1 codes in the same low-nibble/high-nibble layout used by the exporter.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from prismaquant import format_registry as fr
from prismaquant.nvfp4_activation_contract import _E2M1_POSITIVE
from prismaquant.memory_management import (
    enforce_gpu_memory_budget,
    env_flag_enabled,
)


_FP4_E2M1_MAX = 6.0
_NVFP4_GROUP_SIZE = 16
_NVFP4_FUSED_WARMUP_STATE = {
    "attempted": False,
    "compiled": False,
    "skipped_reason": None,
}
_NVFP4_FUSED_WARMUP_ACTIVE = False
_NVFP4_FUSED_COMPILED_SIGNATURES: set[tuple[int, int, int, int, int, int]] = set()


def _pack_fp4_indices(fp4_indices: torch.Tensor, last_dim: int) -> torch.Tensor:
    if last_dim % 2 != 0:
        raise ValueError("NVFP4 packing requires an even K dimension")
    pairs = fp4_indices.reshape(*fp4_indices.shape[:-1], last_dim // 2, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8).contiguous()


def _indices_from_signed_e2m1_values(values: torch.Tensor) -> torch.Tensor:
    """Map signed E2M1 values to packed FP4 code indices, nearest-neighbor.

    Bucketizes on the midpoint boundaries between adjacent codes (like the
    export codec's ``_round_to_codebook``; exact ties round toward zero),
    so a value ε above a code — e.g. from a bf16 round-trip — maps back to
    that code instead of jumping to the NEXT one (the old
    bucketize-on-codes behavior, a full-step error).
    """
    pos = torch.tensor(_E2M1_POSITIVE, device=values.device, dtype=torch.float32)
    midpoints = (pos[1:] + pos[:-1]) / 2.0
    abs_idx = torch.bucketize(
        values.abs().float().contiguous(), midpoints,
    ).clamp_max(7)
    sign_bit = torch.signbit(values).to(torch.long) << 3
    return (abs_idx.long() | sign_bit).to(torch.long)


def nvfp4_pack_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a BF16/FP32 ``[N, K]`` weight for ``nvfp4_fused_aw_matmul``.

    Returns ``(w_packed, w_scales, w_global_scale)`` where ``w_packed`` is
    uint8 ``[N, K // 2]``, ``w_scales`` is real-valued FP32 ``[N, K // 16]``,
    and ``w_global_scale`` is a scalar multiplier. Scales intentionally stay
    FP32 so dequantization matches ``format_registry``'s NVFP4 reference path.
    """
    if weight.dim() != 2:
        raise ValueError(f"NVFP4 fused weight must be 2D, got {tuple(weight.shape)}")
    rows, cols = weight.shape
    if cols % _NVFP4_GROUP_SIZE != 0:
        raise ValueError(f"NVFP4 group_size=16 does not divide K={cols}")
    if cols % 2 != 0:
        raise ValueError(f"NVFP4 packed K must be even, got K={cols}")

    # Export-codec-aligned packing (one rendering everywhere): the fused
    # fast path serves perturbed-X / resident W4A4 evaluation, which must
    # be byte-faithful to shipped NVFP4 (FP8-snapped group scales under a
    # per-tensor global). We bake the EFFECTIVE real scale per group into
    # w_scales and set the global multiplier to 1, so the Triton dequant
    # codes*scales reproduces the export dequant exactly.
    from prismaquant import export_native_compressed as enc

    w_float = weight.detach().float()
    grouped = w_float.reshape(rows, cols // _NVFP4_GROUP_SIZE, _NVFP4_GROUP_SIZE)
    scale_real, global_real = enc._select_nvfp4_pack_scales_and_global(grouped)
    codec = enc._nvfp4_quantize_grouped_codec(
        grouped,
        global_real=global_real,
        scale_real=scale_real,
    )
    eff_scale = enc._nvfp4_effective_scale_from_fp8(
        codec.scale, global_real,
    )
    fp4_indices = codec.indices.reshape(rows, cols)
    return (
        _pack_fp4_indices(fp4_indices, cols),
        eff_scale.to(torch.float32).contiguous(),
        torch.ones((1,), device=weight.device, dtype=torch.float32),
    )


def nvfp4_dequantize_weight(
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    w_global_scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Torch dequantizer for tests and fallback comparisons."""
    if w_packed.dim() != 2:
        raise ValueError("w_packed must have shape [N, K // 2]")
    rows, packed_cols = w_packed.shape
    cols = packed_cols * 2
    if w_scales.shape != (rows, cols // _NVFP4_GROUP_SIZE):
        raise ValueError(
            "w_scales must have shape "
            f"{(rows, cols // _NVFP4_GROUP_SIZE)}, got {tuple(w_scales.shape)}"
        )
    lo = (w_packed & 0xF).long()
    hi = ((w_packed >> 4) & 0xF).long()
    idx = torch.stack((lo, hi), dim=-1).reshape(rows, cols)
    pos = torch.tensor(_E2M1_POSITIVE, device=w_packed.device, dtype=torch.float32)
    abs_vals = pos[idx & 0x7]
    sign = torch.where((idx & 0x8) != 0, -1.0, 1.0)
    scale = (
        w_scales.float()
        .unsqueeze(-1)
        .expand(-1, -1, _NVFP4_GROUP_SIZE)
        .reshape(rows, cols)
    )
    global_scale = w_global_scale.float().reshape(-1)[0]
    return (sign * abs_vals * scale * global_scale).to(dtype)


@triton.jit
def _tl_e2m1_abs_from_index(abs_idx):
    v = tl.full(abs_idx.shape, 0.0, tl.float32)
    v = tl.where(abs_idx == 1, 0.5, v)
    v = tl.where(abs_idx == 2, 1.0, v)
    v = tl.where(abs_idx == 3, 1.5, v)
    v = tl.where(abs_idx == 4, 2.0, v)
    v = tl.where(abs_idx == 5, 3.0, v)
    v = tl.where(abs_idx == 6, 4.0, v)
    v = tl.where(abs_idx == 7, 6.0, v)
    return v


@triton.jit
def _tl_quantize_e2m1_dequant(x):
    # Nearest E2M1 code by midpoint thresholds; exact ties (|x| equal to a
    # midpoint) round half-toward-zero for BOTH signs, matching the export
    # codec's _round_to_codebook. (The old version used >= on the negative
    # branch — sign-asymmetric ties, negative half-ties rounded away from
    # zero.)
    ax = tl.abs(x)
    neg = x < 0.0
    idx = tl.full(x.shape, 0, tl.int32)
    idx = tl.where(ax > 0.25, 1, idx)
    idx = tl.where(ax > 0.75, 2, idx)
    idx = tl.where(ax > 1.25, 3, idx)
    idx = tl.where(ax > 1.75, 4, idx)
    idx = tl.where(ax > 2.5, 5, idx)
    idx = tl.where(ax > 3.5, 6, idx)
    idx = tl.where(ax > 5.0, 7, idx)
    q_abs = _tl_e2m1_abs_from_index(idx)
    return tl.where(neg, -q_abs, q_abs)


@triton.jit
def nvfp4_fused_aw_matmul_kernel(
    x_ptr,
    w_packed_ptr,
    w_scales_ptr,
    w_global_scale_ptr,
    out_ptr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sk,
    stride_om,
    stride_on,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused NVFP4 GEMM with inline activation quantization.

    This proof-of-concept dequantizes FP4 codes to BF16 inside the tile and
    uses ``tl.dot`` for BF16 x BF16 accumulation. It is still one kernel per
    Linear call, avoiding the reference path's many activation-quant launches.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    global_scale = tl.load(w_global_scale_ptr).to(tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        x_abs_grouped = tl.reshape(
            tl.abs(x),
            (BLOCK_M, BLOCK_K // 16, 16),
        )
        x_scale_g = tl.max(x_abs_grouped, axis=2) / 6.0
        x_scale_g = tl.maximum(x_scale_g, 1.0e-8 / 6.0)
        x_scale = tl.reshape(
            tl.broadcast_to(
                tl.expand_dims(x_scale_g, 2),
                (BLOCK_M, BLOCK_K // 16, 16),
            ),
            (BLOCK_M, BLOCK_K),
        )
        xq = _tl_quantize_e2m1_dequant(x / x_scale) * x_scale

        packed = tl.load(
            w_packed_ptr
            + offs_n[None, :] * stride_wn
            + (k[:, None] // 2) * stride_wk,
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        fp4_idx = tl.where((k[:, None] % 2) == 0, packed & 0xF, (packed >> 4) & 0xF)
        abs_idx = fp4_idx & 0x7
        sign = tl.where((fp4_idx & 0x8) != 0, -1.0, 1.0)
        w_vals = sign * _tl_e2m1_abs_from_index(abs_idx)
        w_scale = tl.load(
            w_scales_ptr
            + offs_n[None, :] * stride_sn
            + (k[:, None] // 16) * stride_sk,
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0.0,
        ).to(tl.float32)
        wq = w_vals * w_scale * global_scale
        acc += tl.dot(xq.to(tl.bfloat16), wq.to(tl.bfloat16), out_dtype=tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _validate_inputs(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    w_global_scale: torch.Tensor,
) -> tuple[int, int, int]:
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [M, K], got {tuple(x.shape)}")
    if w_packed.dim() != 2:
        raise ValueError(
            f"w_packed must be 2D [N, K // 2], got {tuple(w_packed.shape)}"
        )
    M, K = x.shape
    N, packed_k = w_packed.shape
    if K % _NVFP4_GROUP_SIZE != 0:
        raise ValueError(f"NVFP4 group_size=16 does not divide K={K}")
    if packed_k * 2 != K:
        raise ValueError(
            f"w_packed K mismatch: packed K={packed_k} implies {packed_k * 2}, "
            f"x has K={K}"
        )
    if tuple(w_scales.shape) != (N, K // _NVFP4_GROUP_SIZE):
        raise ValueError(
            "w_scales must have shape "
            f"{(N, K // _NVFP4_GROUP_SIZE)}, got {tuple(w_scales.shape)}"
        )
    if w_global_scale.numel() < 1:
        raise ValueError("w_global_scale must contain at least one scalar")
    if x.device != w_packed.device or x.device != w_scales.device:
        raise ValueError("x, w_packed, and w_scales must live on the same device")
    if x.device != w_global_scale.device:
        raise ValueError("w_global_scale must live on the same device as x")
    return M, N, K


def nvfp4_fused_aw_matmul(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    w_global_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``x @ dequantize(w_packed).T`` with inline NVFP4 activation RTN."""
    if (
        not _NVFP4_FUSED_WARMUP_ACTIVE
        and not _NVFP4_FUSED_WARMUP_STATE["attempted"]
    ):
        ensure_nvfp4_fused_warmup()
    M, N, K = _validate_inputs(x, w_packed, w_scales, w_global_scale)
    if not x.is_cuda:
        raise RuntimeError("nvfp4_fused_aw_matmul requires CUDA")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"x must be floating point, got {x.dtype}")
    if w_packed.dtype is not torch.uint8:
        raise TypeError(f"w_packed must be torch.uint8, got {w_packed.dtype}")
    if w_scales.dtype is torch.uint8:
        w_scales = w_scales.view(torch.float8_e4m3fn)
    if out is None:
        out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    elif tuple(out.shape) != (M, N):
        raise ValueError(f"out must have shape {(M, N)}, got {tuple(out.shape)}")

    block_m = 16 if M <= 16 else 32
    block_n = 32 if N <= 512 else 64
    block_k = 64 if K <= 64 else 128
    enforce_gpu_memory_budget(device=x.device, reason="NVFP4 fused matmul")
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    nvfp4_fused_aw_matmul_kernel[grid](
        x,
        w_packed,
        w_scales,
        w_global_scale,
        out,
        x.stride(0),
        x.stride(1),
        w_packed.stride(0),
        w_packed.stride(1),
        w_scales.stride(0),
        w_scales.stride(1),
        out.stride(0),
        out.stride(1),
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    _NVFP4_FUSED_COMPILED_SIGNATURES.add((M, N, K, block_m, block_n, block_k))
    return out


def nvfp4_fused_warmup_state() -> dict:
    return {
        **_NVFP4_FUSED_WARMUP_STATE,
        "compiled_signatures": sorted(_NVFP4_FUSED_COMPILED_SIGNATURES),
    }


def ensure_nvfp4_fused_warmup() -> bool:
    global _NVFP4_FUSED_WARMUP_ACTIVE
    if _NVFP4_FUSED_WARMUP_STATE["attempted"]:
        return bool(_NVFP4_FUSED_WARMUP_STATE["compiled"])
    _NVFP4_FUSED_WARMUP_STATE["attempted"] = True
    if not env_flag_enabled("PRISMAQUANT_NVFP4_FUSED_JIT_WARMUP", default=True):
        _NVFP4_FUSED_WARMUP_STATE["skipped_reason"] = "disabled"
        return False
    if not torch.cuda.is_available():
        _NVFP4_FUSED_WARMUP_STATE["skipped_reason"] = "cuda_unavailable"
        return False
    device = torch.device("cuda")
    _NVFP4_FUSED_WARMUP_ACTIVE = True
    try:
        with torch.no_grad():
            x = torch.zeros((8, 64), device=device, dtype=torch.bfloat16)
            weight = torch.zeros((8, 64), device=device, dtype=torch.bfloat16)
            w_packed, w_scales, w_global_scale = nvfp4_pack_weight(weight)
            out = torch.empty((8, 8), device=device, dtype=torch.bfloat16)
            nvfp4_fused_aw_matmul(x, w_packed, w_scales, w_global_scale, out=out)
            torch.cuda.synchronize(device)
        _NVFP4_FUSED_WARMUP_STATE["compiled"] = True
        _NVFP4_FUSED_WARMUP_STATE["skipped_reason"] = None
        return True
    except Exception as exc:
        _NVFP4_FUSED_WARMUP_STATE["skipped_reason"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return False
    finally:
        _NVFP4_FUSED_WARMUP_ACTIVE = False


ensure_nvfp4_fused_warmup()
