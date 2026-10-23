"""Shared FP8_DYNAMIC quantization adapters.

Use compressed-tensors primitives for FP8 cast/dequant behavior and qparam
generation where that library is the exporter/load-time authority. Activation
QDQ follows vLLM's served Linear path: flatten to rows and compute one dynamic
FP8 scale per row/token. On CUDA the activation leg runs fused
(``kernels.fp8_per_token_qdq``, one launch per call); elsewhere it runs the
torch reference below. Both legs divide in FP32 and preserve -0.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC
from compressed_tensors.quantization.utils.helpers import calculate_qparams

#: Profiler section that isolates the per-token FP8 QDQ (PQ #1398).
RECORD_FUNCTION = "prismaquant.fp8_qdq"

#: Set to 1/true/yes/on to force the torch reference on CUDA (debug only).
DISABLE_FUSED_ENV = "PRISMAQUANT_DISABLE_FP8_FUSED_QDQ"


@dataclass(frozen=True)
class FP8DynamicResult:
    quant: torch.Tensor
    scale: torch.Tensor
    dequant: torch.Tensor


def fp8_qdq_reference(
    values: torch.Tensor,
    scale: torch.Tensor,
    *,
    element_dtype: torch.dtype,
    element_max: float,
) -> FP8DynamicResult:
    """Torch reference for per-row FP8 QDQ: FP32 divide, clamp, direct cast.

    This is the deliberate emulation of vLLM's native per-token FP8 kernel,
    not a fallback for a missing op. It also serves inputs the fused kernel
    cannot take (CPU, other dtypes) and the oracle baseline for the fused leg.
    """
    values_f = values.to(torch.float32)
    scale_f = scale.to(device=values_f.device, dtype=torch.float32).clamp_min(
        2.0 ** -127,
    )
    quant = (
        values_f / scale_f
    ).clamp(-float(element_max), float(element_max)).to(element_dtype)
    dequant = quant.to(torch.float32) * scale_f
    return FP8DynamicResult(quant=quant, scale=scale_f, dequant=dequant)


def _fused_activation_result(
    activation: torch.Tensor,
    *,
    element_dtype: torch.dtype,
    element_max: float,
    want_dx: bool,
) -> tuple[FP8DynamicResult, torch.Tensor | None] | None:
    """Run the fused leg over ``activation``, or return None to use the reference.

    Returns the result with FP32 dequant plus the full-precision ``dx`` when
    ``want_dx`` is set, else None for ``dx``. ``dx`` equals
    ``dequant_fp32 - x_fp32`` without the BF16 round trip the hook's own
    subtract keeps, so it serves the statistics accumulate, not that line.
    """
    if os.environ.get(DISABLE_FUSED_ENV, "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        return None
    rows_in = activation.reshape(-1, activation.shape[-1])
    try:
        from .kernels import fp8_per_token_qdq as fused
    except Exception:
        return None
    if not fused.fused_eligible(rows_in, element_dtype=element_dtype):
        return None
    quant, scale, dequant, dx = fused.fused_per_token_qdq(
        rows_in,
        element_dtype=element_dtype,
        element_max=element_max,
        want_dx=want_dx,
        dequant_dtype=torch.float32,
    )
    return FP8DynamicResult(quant=quant, scale=scale, dequant=dequant), dx


def fp8_dynamic_weight_qdq(
    weight: torch.Tensor,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = 448.0,
) -> FP8DynamicResult:
    """Compressed-tensors FP8_DYNAMIC static per-channel weight QDQ."""
    weight_f = weight.to(torch.float32)
    rows = weight_f.reshape(-1, weight_f.shape[-1])

    if element_dtype != torch.float8_e4m3fn:
        scale = (
            rows.abs().amax(dim=-1, keepdim=True).clamp_min(2.0 ** -127)
            / float(element_max)
        )
        result = fp8_qdq_reference(
            rows,
            scale,
            element_dtype=element_dtype,
            element_max=element_max,
        )
        return FP8DynamicResult(
            quant=result.quant.reshape(weight.shape),
            scale=result.scale.reshape(*weight.shape[:-1], 1),
            dequant=result.dequant.reshape(weight.shape),
        )

    args = FP8_DYNAMIC["weights"]
    scale, zero_point = calculate_qparams(
        rows.amin(dim=-1),
        rows.amax(dim=-1),
        args,
    )
    scale = scale.reshape(-1, 1)
    zero_point = zero_point.reshape(-1, 1)
    quant = quantize(rows, scale, zero_point, args, dtype=element_dtype)
    dequant = quant.to(torch.float32) * scale
    return FP8DynamicResult(
        quant=quant.reshape(weight.shape),
        scale=scale.reshape(*weight.shape[:-1], 1),
        dequant=dequant.reshape(weight.shape),
    )


def fp8_dynamic_activation_qdq_vllm(
    activation: torch.Tensor,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = 448.0,
) -> FP8DynamicResult:
    """vLLM dynamic-token FP8 activation QDQ for served Linear inputs.

    The fused kernel is the default path on CUDA; the torch reference serves
    all other inputs. Set ``PRISMAQUANT_DISABLE_FP8_FUSED_QDQ`` to force the
    reference on CUDA. The profiler section ``prismaquant.fp8_qdq`` isolates
    this call (PQ #1398).
    """
    from torch.profiler import record_function

    with record_function(RECORD_FUNCTION):
        fused = _fused_activation_result(
            activation,
            element_dtype=element_dtype,
            element_max=element_max,
            want_dx=False,
        )
        if fused is not None:
            result, _ = fused
        else:
            act_f = activation.to(torch.float32)
            rows = act_f.reshape(-1, act_f.shape[-1])
            min_scale = 1.0 / (float(element_max) * 512.0)
            # CUDA scalar division may multiply by a rounded reciprocal instead.
            # That moves some scales by one FP32 ULP and can cross an FP8
            # midpoint. vLLM's native per-token kernel divides in FP32; keep
            # the denominator on-device so TensorIterator retains the same
            # operation rather than folding it.
            denominator = torch.full(
                (), float(element_max), dtype=torch.float32, device=rows.device,
            )
            scale = (
                rows.abs().amax(dim=-1, keepdim=True) / denominator
            ).clamp_min(min_scale)

            # The native activation kernel directly divides, clamps and casts.
            # Adding even a zero zero-point (the CT weight/export path) erases
            # negative zero.
            result = fp8_qdq_reference(
                rows, scale, element_dtype=element_dtype, element_max=element_max,
            )

    return FP8DynamicResult(
        quant=result.quant.reshape(activation.shape),
        scale=result.scale.reshape(*activation.shape[:-1], 1),
        dequant=result.dequant.reshape(activation.shape),
    )


def fp8_dynamic_activation_qdq_vllm_with_dx(
    activation: torch.Tensor,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = 448.0,
) -> tuple[FP8DynamicResult, torch.Tensor]:
    """Pair the activation QDQ with its full-precision ``dx``.

    Returns the result plus ``dx = dequant_fp32 - x_fp32`` over the input
    shape. The fused leg writes ``dx`` in the same launch; the reference
    subtracts once. ``dx`` skips the BF16 round trip the hook's own subtract
    keeps, so it serves the statistics accumulate, not that line.
    """
    from torch.profiler import record_function

    with record_function(RECORD_FUNCTION):
        fused = _fused_activation_result(
            activation,
            element_dtype=element_dtype,
            element_max=element_max,
            want_dx=True,
        )
        if fused is not None:
            result, dx = fused
            assert dx is not None
        else:
            act_f = activation.to(torch.float32)
            rows = act_f.reshape(-1, act_f.shape[-1])
            min_scale = 1.0 / (float(element_max) * 512.0)
            denominator = torch.full(
                (), float(element_max), dtype=torch.float32, device=rows.device,
            )
            scale = (
                rows.abs().amax(dim=-1, keepdim=True) / denominator
            ).clamp_min(min_scale)
            flat = fp8_qdq_reference(
                rows, scale, element_dtype=element_dtype, element_max=element_max,
            )
            result = FP8DynamicResult(
                quant=flat.quant,
                scale=flat.scale,
                dequant=flat.dequant,
            )
            dx = flat.dequant - rows

    return (
        FP8DynamicResult(
            quant=result.quant.reshape(activation.shape),
            scale=result.scale.reshape(*activation.shape[:-1], 1),
            dequant=result.dequant.reshape(activation.shape),
        ),
        dx.reshape(activation.shape),
    )
