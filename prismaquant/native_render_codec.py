"""Native weight arithmetic and wire codecs shared by render and export.

Artifact naming, serving policy, shard IO and CLI orchestration belong to the
exporter. This owner implements the existing native numerical recipes.
"""
from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from compressed_tensors.quantization.utils.mxfp_utils import generate_mx_scales
from . import nvfp4_activation_contract as _nvfp4_activation_contract
from .format_registry import _mx_rounded_amax_power2
from .fp8_dynamic import fp8_dynamic_weight_qdq
from .mx_formats import e8m0_to_scale, mxfp8_e4m3_qdq
from .layer_config import canonicalize_format
from .native_render_levers import NativeRenderLevers
from .render_score import normalize_clipped_fisher_row_weights, resolve_fisher_row_weight_clip

# `compressed_tensors.compressors.nvfp4.helpers.pack_fp4_to_uint8` and is
# verified byte-identical in tests. We pack indices directly rather than call
# that helper because (a) it takes a FLOAT tensor and runs its own argmin
# codebook assignment + clamping, whereas our scale-rule / JSO / four-over-six
# `_round_to_codebook` path already produces the indices; and (b) importing the
# library's package __init__ pulls in transformers internals that are not
# stable across the transformers 4.x -> 5.x break.
# ---------------------------------------------------------------------------
NVFP4_MAX = _nvfp4_activation_contract.FP4_E2M1_MAX
FP8_E4M3_MAX = _nvfp4_activation_contract.FP8_E4M3_MAX
NVFP4_SCALE_RULE_ENV = "PRISMAQUANT_NVFP4_SCALE_RULE"
NVFP4_SCALE_RULE_STATIC_6 = "static_6"
NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE = "four_over_six_mse"
NVFP4_SCALE_RULE_JOINT_MSE = "joint_mse"
_NVFP4_SCALE_RULE_ALIASES = {
    "": NVFP4_SCALE_RULE_STATIC_6,
    "default": NVFP4_SCALE_RULE_STATIC_6,
    "static": NVFP4_SCALE_RULE_STATIC_6,
    "static_6": NVFP4_SCALE_RULE_STATIC_6,
    "six": NVFP4_SCALE_RULE_STATIC_6,
    "6": NVFP4_SCALE_RULE_STATIC_6,
    "4/6": NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
    "4over6": NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
    "four_over_six": NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
    "four_over_six_mse": NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
    "mse": NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
    "joint": NVFP4_SCALE_RULE_JOINT_MSE,
    "joint_mse": NVFP4_SCALE_RULE_JOINT_MSE,
    "joint_scale": NVFP4_SCALE_RULE_JOINT_MSE,
    "joint_scale_opt": NVFP4_SCALE_RULE_JOINT_MSE,
    "joint_scale_optimization": NVFP4_SCALE_RULE_JOINT_MSE,
    "codebook_mse": NVFP4_SCALE_RULE_JOINT_MSE,
}
_DO_NO_HARM_STATS: Counter = Counter()


def _record_do_no_harm_revert(fmt: str) -> None:
    _DO_NO_HARM_STATS[f"{fmt}_reverts"] += 1


def _record_do_no_harm_failure(fmt: str, linear_name: str | None, exc: Exception) -> None:
    _DO_NO_HARM_STATS[f"{fmt}_failures"] += 1
    name = linear_name or "<unknown>"
    print(
        f"[do-no-harm] WARN {name} {fmt} gate failed: {exc!r}",
        flush=True,
    )

# The NVFP4 codebook is a CONSTANT -- the eight positive E2M1 levels -- but it
# was being rebuilt from a Python list on every call, and the GPTQ render calls
# it once per column-quantize. Measured on one 5120x5120 Linear with
# gptq+static_act_order+joint_scale_opt: 11,562 calls, 2.279 s of a 3.705 s
# render. Building a CUDA tensor from a Python list is ~200 us (Python-side
# element walk, then an H2D copy), so 62% of the render was materializing 8
# floats over and over while the GPU sat at 33 W.
#
# Memoizing is safe because the value never varies and both call sites are
# strictly read-only (`cb[idx]`, `torch.bucketize(x, cb)`, `cb.numel()`); no
# caller mutates or keeps an escaping alias. Keyed on (device, dtype) so a
# mixed-device run cannot be handed a tensor from the wrong device, which would
# be a silent correctness bug rather than a slow path.
#
# The cached tensor must be BIT-IDENTICAL to the freshly built one -- this runs
# inside the production render, and a render that differs from the exported
# bytes is the rendering confound principle 8 exists to prevent. Pinned by
# tests/test_nvfp4_codebook_cache.py, which compares cached against fresh and
# renders a whole Linear both ways.
_NVFP4_CODEBOOK_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}


def _nvfp4_codebook(device, dtype=torch.float32) -> torch.Tensor:
    key = (str(device), dtype)
    cb = _NVFP4_CODEBOOK_CACHE.get(key)
    if cb is None:
        cb = torch.tensor(_nvfp4_activation_contract._E2M1_POSITIVE, device=device, dtype=dtype)
        _NVFP4_CODEBOOK_CACHE[key] = cb
    return cb


def resolve_nvfp4_scale_rule(raw: str | None = None) -> str:
    """Canonicalize the NVFP4 block-scale rule.

    ``static_6`` is the compressed-tensors/llm-compressor default: every
    16-value block maps its maximum magnitude to FP4 code ±6.  FourOverSix
    evaluates max-to-6 and max-to-4 and keeps the lower block-MSE scale while
    preserving the same NVFP4 on-disk schema and vLLM runtime kernel.
    ``joint_mse`` extends that packer-compatible candidate set to every
    positive NVFP4 codebook level, making FourOverSix a strict subset.
    """
    if raw is None:
        raw = os.environ.get(NVFP4_SCALE_RULE_ENV, NVFP4_SCALE_RULE_STATIC_6)
    key = str(raw).strip().lower().replace("-", "_")
    try:
        return _NVFP4_SCALE_RULE_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted({
            NVFP4_SCALE_RULE_STATIC_6,
            NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
            NVFP4_SCALE_RULE_JOINT_MSE,
        }))
        raise ValueError(
            f"unsupported {NVFP4_SCALE_RULE_ENV}={raw!r}; "
            f"expected one of: {allowed}"
        ) from exc


def _nvfp4_scale_rule_from_env(
    *,
    levers: NativeRenderLevers | None = None,
) -> str:
    if levers is None:
        levers = NativeRenderLevers.resolve()
    return levers.nvfp4_scale_rule


def _decode_nvfp4_indices(
    fp4_idx: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return _decode_nvfp4_indices_with_eff_scale(fp4_idx, scale.unsqueeze(-1))


def _decode_nvfp4_indices_with_eff_scale(
    fp4_idx: torch.Tensor,
    eff_scale: torch.Tensor,
) -> torch.Tensor:
    cb = _nvfp4_codebook(fp4_idx.device, dtype=torch.float32)
    abs_idx = fp4_idx & 0x7
    sign = -((fp4_idx >> 3).to(torch.float32) * 2 - 1)
    return sign * cb[abs_idx] * eff_scale


def _nvfp4_mse_for_group_scale(
    grouped: torch.Tensor,
    scale: torch.Tensor,
    *,
    global_real: torch.Tensor | None = None,
) -> torch.Tensor:
    if global_real is not None:
        eff_scale = _nvfp4_effective_scale_from_real(
            scale,
            global_real,
            quantize_fp8=True,
        ).unsqueeze(-1)
    else:
        eff_scale = scale.unsqueeze(-1)
    _idx, dq = _nvfp4_quantize_dequantize_with_eff_scale(
        grouped,
        eff_scale,
    )
    return (grouped - dq).pow(2).sum(dim=-1)


def _nvfp4_best_max_to_level_scale(
    grouped: torch.Tensor,
    levels: Sequence[float],
    *,
    global_real: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pick the best max-to-codebook-level scale for each NVFP4 group.

    ``four_over_six_mse`` is the two-level set ``levels=(6, 4)``. The joint
    scale rule uses the bound ``nvfp4_joint_scale_levels`` (also ``(6, 4)`` by default;
    extendable via ``PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS``) and stays
    final-pack compatible because every chosen scale is
    ``max_abs / codebook_level``.
    """

    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)
    best_scale: torch.Tensor | None = None
    best_mse: torch.Tensor | None = None
    for level in levels:
        scale = max_abs / float(level)
        mse = _nvfp4_mse_for_group_scale(
            grouped,
            scale,
            global_real=global_real,
        )
        if best_mse is None:
            best_mse = mse
            best_scale = scale
            continue
        take = mse < best_mse
        best_mse = torch.where(take, mse, best_mse)
        assert best_scale is not None
        best_scale = torch.where(take, scale, best_scale)
    assert best_scale is not None
    return best_scale







def _select_nvfp4_group_scales(
    grouped: torch.Tensor,
    *,
    scale_rule: str | None = None,
    global_real: torch.Tensor | None = None,
    joint_scale_levels: tuple[float, ...] | None = None,

    levers: NativeRenderLevers | None = None,
) -> torch.Tensor:
    """Return per-block real NVFP4 scales for ``grouped[..., group_size]``.

    The returned tensor has shape ``grouped.shape[:-1]``.  This function is
    the single scale-selection point used by RTN, GPTQ block quantization,
    scale-sweep initialization, packed experts, and final export packing.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    rule = (
        _nvfp4_scale_rule_from_env( levers=levers)
        if scale_rule is None
        else resolve_nvfp4_scale_rule(scale_rule)
    )
    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)
    scale_6 = max_abs / NVFP4_MAX
    if rule == NVFP4_SCALE_RULE_STATIC_6:
        return scale_6
    if rule == NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE:
        return _nvfp4_best_max_to_level_scale(
            grouped,
            (6.0, 4.0),
            global_real=global_real,
        )
    if rule == NVFP4_SCALE_RULE_JOINT_MSE:
        return _nvfp4_best_max_to_level_scale(
            grouped,
            (
                levers.nvfp4_joint_scale_levels
                if joint_scale_levels is None
                else joint_scale_levels
            ),
            global_real=global_real,
        )
    raise AssertionError(f"unhandled NVFP4 scale rule: {rule!r}")


def _nvfp4_snapped_scale_scoring_enabled(
    *,
    levers: NativeRenderLevers | None = None,
) -> bool:
    """Score NVFP4 scale candidates under the FP8-SNAPPED effective scale.

    Research lever (default OFF). Scoring under the snapped scale the
    served kernel actually uses is more faithful in principle, but it
    changes the shipped NVFP4 bytes for the joint_mse (JSO, a production
    default lever) and four_over_six rules and has not cleared a served
    gold-metric A/B (QC finding on review-batch M21). Promote per the
    ladder: served KL+PPL at matched bpp on 4B, then a 27B confirmation.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    return levers.nvfp4_snapped_scale_scoring


def _select_nvfp4_pack_scales_and_global(
    grouped: torch.Tensor,
    *,
    global_real_override: torch.Tensor | None = None,
    scale_rule: str | None = None,
    snapped_scale_scoring: bool | None = None,
    joint_scale_levels: tuple[float, ...] | None = None,

    levers: NativeRenderLevers | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if levers is None:
        levers = NativeRenderLevers.resolve()
    score_snapped = (
        _nvfp4_snapped_scale_scoring_enabled( levers=levers)
        if snapped_scale_scoring is None
        else bool(snapped_scale_scoring)
    )
    if not score_snapped:
        # Pre-2026-06-12 behavior (byte-stable with shipped artifacts):
        # scales scored under the RAW real scale; the tensor global is
        # derived once from the chosen scales (or taken verbatim from the
        # fused-sibling override) without re-scoring.
        scale = _select_nvfp4_group_scales(
            grouped,
            scale_rule=scale_rule,
            joint_scale_levels=joint_scale_levels,
         levers=levers)
        if global_real_override is not None:
            global_real = global_real_override.to(
                grouped.device, dtype=torch.float32).clamp_min(1e-12)
        else:
            global_real = (scale.amax() / FP8_E4M3_MAX).clamp_min(1e-12)
        return scale, global_real
    scale = _select_nvfp4_group_scales(
        grouped,
        scale_rule=scale_rule,
        joint_scale_levels=joint_scale_levels,
     levers=levers)
    if global_real_override is not None:
        global_real = global_real_override.to(
            grouped.device,
            dtype=torch.float32,
        ).clamp_min(1e-12)
        scale = _select_nvfp4_group_scales(
            grouped,
            scale_rule=scale_rule,
            global_real=global_real,
            joint_scale_levels=joint_scale_levels,
         levers=levers)
        return scale, global_real

    global_real = (scale.amax() / FP8_E4M3_MAX).clamp_min(1e-12)
    for _ in range(3):
        snapped_scale = _select_nvfp4_group_scales(
            grouped,
            scale_rule=scale_rule,
            global_real=global_real,
            joint_scale_levels=joint_scale_levels,
         levers=levers)
        next_global = (
            snapped_scale.amax() / FP8_E4M3_MAX
        ).clamp_min(1e-12)
        scale = snapped_scale
        if torch.allclose(next_global, global_real, rtol=0.0, atol=1e-12):
            break
        global_real = next_global
    return scale, global_real





def _nvfp4_effective_scale_from_real(
    scale_real: torch.Tensor,
    global_real: torch.Tensor,
    *,
    quantize_fp8: bool,
) -> torch.Tensor:
    fp8_scale = _nvfp4_fp8_scale_from_real(
        scale_real,
        global_real,
        quantize_fp8=quantize_fp8,
    )
    return _nvfp4_effective_scale_from_fp8(fp8_scale, global_real)


def _nvfp4_fp8_scale_from_real(
    scale_real: torch.Tensor,
    global_real: torch.Tensor,
    *,
    quantize_fp8: bool = True,
) -> torch.Tensor:
    fp8_scale = (
        scale_real / global_real.to(scale_real.device, dtype=torch.float32)
    ).clamp(0, FP8_E4M3_MAX)
    if quantize_fp8:
        fp8_scale = fp8_scale.to(torch.float8_e4m3fn)
    return fp8_scale


def _nvfp4_effective_scale_from_fp8(
    fp8_scale: torch.Tensor,
    global_real: torch.Tensor,
) -> torch.Tensor:
    return (
        fp8_scale.to(torch.float32)
        * global_real.to(fp8_scale.device, dtype=torch.float32)
    ).clamp_min(1e-12)


def _nvfp4_quantize_dequantize_with_eff_scale(
    values: torch.Tensor,
    eff_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    in_grid = (values / eff_scale.clamp_min(1e-12)).clamp(
        -NVFP4_MAX,
        NVFP4_MAX,
    )
    fp4_idx = _round_to_codebook(in_grid)
    return fp4_idx, _decode_nvfp4_indices_with_eff_scale(fp4_idx, eff_scale)


def _nvfp4_quant_dequant_with_eff_scale(
    values: torch.Tensor,
    eff_scale: torch.Tensor,
) -> torch.Tensor:
    _idx, dequant = _nvfp4_quantize_dequantize_with_eff_scale(
        values,
        eff_scale,
    )
    return dequant


def _nvfp4_quantize_grouped_codec(
    grouped: torch.Tensor,
    *,
    global_real: torch.Tensor,
    scale_real: torch.Tensor | None = None,
    scale_rule: str | None = None,

    levers: NativeRenderLevers | None = None,
) -> _NVFP4CodecResult:
    if levers is None:
        levers = NativeRenderLevers.resolve()
    grouped_f = grouped.to(torch.float32)
    if scale_real is None:
        scale_real = _select_nvfp4_group_scales(
            grouped_f,
            scale_rule=scale_rule,
            global_real=global_real,
         levers=levers)
    scale_fp8 = _nvfp4_fp8_scale_from_real(
        scale_real.to(grouped_f.device, dtype=torch.float32),
        global_real,
        quantize_fp8=True,
    )
    eff_scale = _nvfp4_effective_scale_from_fp8(
        scale_fp8,
        global_real,
    ).unsqueeze(-1)
    fp4_idx, dequant = _nvfp4_quantize_dequantize_with_eff_scale(
        grouped_f,
        eff_scale,
    )
    return _NVFP4CodecResult(
        indices=fp4_idx,
        scale=scale_fp8,
        dequant=dequant,
    )


def _select_nvfp4_joint_gptq_eff_scale(
    grouped: torch.Tensor,
    global_real: torch.Tensor,
    *,
    col_importance: torch.Tensor | None = None,

    levers: NativeRenderLevers | None = None,
) -> torch.Tensor:
    """Return GPTQ-time effective scales for Lift-style joint scale search.

    Candidate group scales include max-to-6 and max-to-4, so FourOverSix is a
    strict subset. Additional max-to-codebook-level choices keep the output
    representable by the same NVFP4 compressed-tensors metadata when the
    final packer uses ``joint_mse``.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()

    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)
    weight = None
    if col_importance is not None:
        weight = col_importance.to(grouped.device, dtype=torch.float32)
        view_shape = (1,) * (grouped.dim() - 1) + (grouped.shape[-1],)
        weight = weight.reshape(view_shape)

    best_scale: torch.Tensor | None = None
    best_mse: torch.Tensor | None = None
    for level in levers.nvfp4_joint_scale_levels:
        scale = max_abs / float(level)
        eff = _nvfp4_effective_scale_from_real(
            scale,
            global_real,
            quantize_fp8=True,
        )
        dq = _nvfp4_quant_dequant_with_eff_scale(grouped, eff.unsqueeze(-1))
        err = (grouped - dq).pow(2)
        if weight is not None:
            err = err * weight
        mse = err.sum(dim=-1)
        if best_mse is None:
            best_mse = mse
            best_scale = eff
            continue
        take = mse < best_mse
        best_mse = torch.where(take, mse, best_mse)
        assert best_scale is not None
        best_scale = torch.where(take, eff, best_scale)
    assert best_scale is not None
    return best_scale.clamp_min(1e-12)


def _optimize_nvfp4_joint_global_real(
    weight: torch.Tensor,
    *,
    group_size: int,
    base_global_real: torch.Tensor,

    levers: NativeRenderLevers | None = None,
) -> torch.Tensor:
    """Choose a tensor global scale jointly with group-scale candidates.

    The search is deliberately small and opt-in. It scores candidate tensor
    globals after FP8 realization of group scales, chunking rows so large
    Linears do not materialize a candidate dimension over the full weight.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()

    grid = levers.nvfp4_joint_scale_global_grid
    if grid <= 1:
        return base_global_real.clamp_min(1e-12)
    span_lo = levers.nvfp4_joint_scale_global_span_lo
    span_hi = levers.nvfp4_joint_scale_global_span_hi
    W = weight.to(torch.float32)
    rows, cols = W.shape
    grouped = W.reshape(rows, cols // group_size, group_size)
    candidates = (
        base_global_real.to(W.device, dtype=torch.float32).reshape(())
        * torch.linspace(span_lo, span_hi, grid, device=W.device, dtype=torch.float32)
    ).clamp_min(1e-12)

    n_groups = cols // group_size
    bytes_per_row = max(1, n_groups * group_size * 4 * 4)
    row_chunk = min(rows, max(1, (512 * 1024 * 1024) // bytes_per_row))
    scores = torch.zeros((grid,), device=W.device, dtype=torch.float64)
    for idx, global_real in enumerate(candidates):
        total = torch.zeros((), device=W.device, dtype=torch.float64)
        for r0 in range(0, rows, row_chunk):
            r1 = min(r0 + row_chunk, rows)
            chunk = grouped[r0:r1]
            eff = _select_nvfp4_joint_gptq_eff_scale(chunk, global_real, levers=levers)
            dq = _nvfp4_quant_dequant_with_eff_scale(chunk, eff.unsqueeze(-1))
            total = total + (chunk - dq).pow(2).sum().to(torch.float64)
        scores[idx] = total
    best = int(scores.argmin().item())
    return candidates[best].reshape(()).clamp_min(1e-12)


def _round_to_codebook(values_in_grid: torch.Tensor) -> torch.Tensor:
    """Round per-element values (already scaled into the [-6, +6]
    NVFP4 grid) to the nearest codebook entry, using bucketize on the
    sorted absolute codebook. O(N log K) instead of O(N · K).

    Returns a Long tensor of 4-bit indices in [0, 15], where bit 3 is
    the sign bit and bits 0-2 are the abs-codebook index.
    """
    cb = _nvfp4_codebook(values_in_grid.device, dtype=torch.float32)
    abs_x = values_in_grid.abs().contiguous()
    idx = torch.bucketize(abs_x, cb)        # insertion: cb[idx-1] <= x < cb[idx]
    idx_lo = (idx - 1).clamp_min(0).clamp_max(cb.numel() - 1)
    idx_hi = idx.clamp_max(cb.numel() - 1)
    lo_v = cb[idx_lo]
    hi_v = cb[idx_hi]
    pick_hi = (hi_v - abs_x).abs() < (abs_x - lo_v).abs()
    abs_idx = torch.where(pick_hi, idx_hi, idx_lo).long()
    sign_bit = torch.signbit(values_in_grid).to(torch.long) << 3
    return abs_idx + sign_bit                # [..., shape]; values 0-15


MXFP8_LEGACY_ALIAS = "MXFP8"
MXFP8_EXPLICIT_FORMATS = {"MXFP8_E4M3", "MXFP8_E5M2"}


@dataclass(frozen=True)
class _NVFP4CodecResult:
    indices: torch.Tensor
    scale: torch.Tensor
    dequant: torch.Tensor


@dataclass(frozen=True)
class _FP8CodecResult:
    quant: torch.Tensor
    scale: torch.Tensor
    dequant: torch.Tensor


@dataclass(frozen=True)
class _MXFP8CodecResult:
    quant: torch.Tensor
    scale: torch.Tensor
    dequant: torch.Tensor


@dataclass(frozen=True)
class _MXFP4CodecResult:
    indices: torch.Tensor
    packed: torch.Tensor
    scale: torch.Tensor
    dequant: torch.Tensor


def _canonical_export_format(fmt: str) -> str:
    fmt_u = str(fmt).strip().upper()
    if fmt_u == MXFP8_LEGACY_ALIAS:
        return "MXFP8_E4M3"
    return fmt_u


def _resolve_act_clip_quantile(default: str = "0.999",
    *,
    levers: NativeRenderLevers | None = None,
) -> float | None:
    """Return the effective activation-clip quantile for GPTQ scoring."""
    if levers is None:
        levers = NativeRenderLevers.resolve()
    return levers.act_clip_quantile


def _normalize_act_clip_rescale(mode: str | None) -> str:
    if mode is None:
        mode = "none"
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"", "0", "false", "no", "off", "none"}:
        return "none"
    raise ValueError("activation clip rescaling is not supported")


def _rescale_clipped_activation_matrix(
    original: torch.Tensor,
    clipped: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Return clipped activations, rejecting retired row-rescale modes."""
    mode = _normalize_act_clip_rescale(mode)
    if mode == "none" or original.numel() == 0:
        return clipped
    raise ValueError("activation clip rescaling is not supported")


def _activation_matrix_for_gptq(
    activations: torch.Tensor,
    cols: int,
    *,
    device: torch.device | None = None,
    clip_threshold: float | None = None,
    clip_quantile: float | None = None,
    clip_rescale: str | None = None,
    row_weights: torch.Tensor | None = None,

    levers: NativeRenderLevers | None = None,
) -> torch.Tensor:
    """Flatten activations and apply the same optional clipping used by GPTQ.

    This is intentionally shared by the Hessian build, damping sweep
    evaluator, and do-no-harm gate. Mixing clipped optimization with
    unclipped local gates caused the full quality-win stack to undo part
    of the activation-clipping gain.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    X = activations.detach().to(torch.float32)
    if device is not None:
        X = X.to(device)
    X = X.reshape(-1, cols)
    if clip_threshold is not None and clip_threshold > 0.0 and X.numel() > 0:
        thresh = torch.tensor(
            float(clip_threshold), device=X.device, dtype=X.dtype)
        X_clipped = X.clamp(min=-thresh, max=thresh)
        X = _rescale_clipped_activation_matrix(
            X,
            X_clipped,
            mode=_normalize_act_clip_rescale(clip_rescale),
        )
    else:
        q = _resolve_act_clip_quantile( levers=levers) if clip_quantile is None else clip_quantile
        if q is not None and 0.0 < q < 1.0 and X.numel() > 0:
            thresh = X.abs().quantile(float(q), dim=1, keepdim=True)
            X = X.clamp(min=-thresh, max=thresh)
    if row_weights is not None and X.numel() > 0:
        rw = _normalize_fisher_row_weights(row_weights, X.shape[0], X.device, levers=levers)
        if rw is not None:
            X = X * rw.sqrt().unsqueeze(1).to(dtype=X.dtype)
    return X


def _normalize_fisher_row_weights(
    row_weights: torch.Tensor | None,
    n_rows: int,
    device: torch.device,

    *,
    levers: NativeRenderLevers | None = None,
) -> torch.Tensor | None:
    """Return non-negative Fisher row weights normalized to mean 1.

    The h-detail probe stores per-token gradient² weights.  For a local
    least-squares GPTQ objective, `X.T @ diag(g²) @ X` is equivalent to
    scaling each activation row by `sqrt(g²)`.  Normalizing the selected
    slice to mean 1 preserves the scale expected by the existing damping
    candidates and local error gates. The normalise-then-clip rule itself
    lives in render_score.normalize_clipped_fisher_row_weights so the cost
    paths cannot drift (issue #159).
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    if row_weights is None or n_rows <= 0:
        return None
    try:
        rw = row_weights.detach().reshape(-1).to(device=device, dtype=torch.float32)
    except Exception:
        return None
    if rw.numel() < n_rows:
        return None
    rw = rw[:n_rows]
    return normalize_clipped_fisher_row_weights(
        rw, levers.fisher_row_weight_clip, require_positive_mean=True
    )


def _activation_col_importance_for_gptq(
    activations: torch.Tensor,
    cols: int,
    *,
    device: torch.device | None = None,
    clip_threshold: float | None = None,
    clip_quantile: float | None = None,
    clip_rescale: str | None = None,
    row_weights: torch.Tensor | None = None,

    levers: NativeRenderLevers | None = None,
) -> torch.Tensor:
    if levers is None:
        levers = NativeRenderLevers.resolve()
    X = _activation_matrix_for_gptq(
        activations,
        cols,
        device=device,
        clip_threshold=clip_threshold,
        clip_quantile=clip_quantile,
        clip_rescale=clip_rescale,
        row_weights=row_weights,
     levers=levers)
    if X.numel() == 0:
        return torch.ones(cols, device=device, dtype=torch.float32)
    return X.pow(2).mean(dim=0).clamp_min(1e-12)


def _activation_weighted_weight_error(
    reference_weight: torch.Tensor,
    rendered_weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    row_weights: torch.Tensor | None = None,

    levers: NativeRenderLevers | None = None,
) -> float:
    """Score a rendered weight with the same cheap local gate as NVFP4.

    This is the diagonal form of output MSE: columns with larger calibration
    activation energy matter more. It is intentionally used only as a local
    do-no-harm gate; the production cache still uses the shared output scorer.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    ref = reference_weight.to(torch.float32)
    cand = rendered_weight.to(device=ref.device, dtype=torch.float32)
    a2 = _activation_col_importance_for_gptq(
        activations,
        ref.shape[1],
        device=ref.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=row_weights,
     levers=levers)
    return float((a2 * (ref - cand).pow(2).sum(dim=0)).sum())


def pack_fp4_indices(fp4_indices: torch.Tensor, last_dim: int) -> torch.Tensor:
    """Pack a tensor of 4-bit indices (final dim must be even) into
    uint8, two indices per byte. Preserves leading dimensions.
    """
    if last_dim % 2 != 0:
        raise ValueError("nvfp4 pack requires an even last dim")
    pairs = fp4_indices.reshape(*fp4_indices.shape[:-1], last_dim // 2, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)


DEFAULT_INPUT_GLOBAL_SCALE = (
    _nvfp4_activation_contract.UNCALIBRATED_INPUT_GLOBAL_SCALE
)  # uncalibrated fallback; matches
# compressed-tensors' 1.0 "no global scaling" fallback for uncalibrated or
# degenerate nonpositive tensors.

# Compatibility aliases retained for external callers/tests.  The versioned
# activation-contract module is the sole owner of their values and policy.
_FP4_E2M1_MAX = _nvfp4_activation_contract.FP4_E2M1_MAX
_FP8_E4M3_MAX = _nvfp4_activation_contract.FP8_E4M3_MAX


def _nvfp4_input_gscale_fp8_range_enabled(
    *,
    levers: NativeRenderLevers | None = None,
) -> bool:
    """Whether input_global_scale uses the compressed-tensors/vLLM
    convention ``FP8_MAX * FP4_MAX / amax`` (generate_gparam) instead of
    the legacy ``FP4_MAX / amax``.

    Default OFF (legacy bytes). The convention places serve-time
    FP8-stored activation block scales in (0, 448] instead of (0, 1] —
    rescuing blocks >64x below calibration amax from FP8 subnormals at
    the cost of CLIPPING any serve block whose amax exceeds calibration
    amax. Served A/Bs 2026-07-02 (weights byte-identical, only
    input_global_scale x448): Qwen3.6-35B-A3B MoE frontier -14.1% KL
    (WIN), LFM2.5 thin-calib smoke +5.8% (loss), Qwen3.6-27B regen dense
    +37.5% (LOSS). Strongly artifact-dependent, so the default stays
    backwards-compatible; set ``PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE=1``
    only behind a per-artifact served A/B (the scale is a free
    post-export knob: patch input_global_scale in place, re-measure).
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    return (
        levers.nvfp4_input_global_scale_policy
        == _nvfp4_activation_contract.FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY
    )


def _nvfp4_input_global_scale_from_max_abs(
    max_abs: float, *, policy: str | None = None,

    levers: NativeRenderLevers | None = None,
) -> float:
    """input_global_scale for a calibrated activation ``max_abs``.

    The shared policy owner selects the legacy ``6 / amax`` bytes by default
    or the explicit compressed-tensors ``448 * 6 / amax`` opt-in.  This
    compatibility wrapper never defines a second formula.

    ``policy`` is optional and resolves live when omitted, as every historical
    caller expects.  A caller that has already resolved ONE policy for a whole
    operation -- the production cache fill, whose render levers stamp it --
    passes it, so the G that reaches the renderer and the G its score records
    are priced at cannot come from two different resolutions (#227).
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    return _nvfp4_activation_contract.input_global_scale_from_max_abs(
        max_abs,
        policy=(
            _nvfp4_activation_contract.resolve_input_global_scale_policy(
                policy if policy is not None else levers.nvfp4_input_global_scale_policy)
        ),
        nonpositive_fallback=(
            _nvfp4_activation_contract.UNCALIBRATED_INPUT_GLOBAL_SCALE
        ),
    )


def compute_nvfp4_input_global_scale(activations: torch.Tensor,
    *,
    levers: NativeRenderLevers | None = None,
) -> float:
    """Per-tensor input_global_scale from cached activations.

    Activations can have any shape.  The shared activation-contract policy
    converts their maximum absolute value and preserves the legacy all-zero
    fallback required for byte-compatible native export.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    max_abs = float(activations.detach().abs().max().item())
    return _nvfp4_input_global_scale_from_max_abs(max_abs, levers=levers)


def _resolve_nvfp4_input_global_scale(
    override: float | None = None,
    *,
    target: str | None = None,
    calibrated_scales: Mapping[str, float] | None = None,
) -> float:
    """Legacy native-export compatibility delegate.

    The uncalibrated fallback preserves existing native artifact bytes.  Its
    presence is also why this exporter claims no versioned fused contract.
    """

    return _nvfp4_activation_contract.resolve_input_global_scale_value(
        override,
        target=target,
        calibrated_scales=calibrated_scales,
        allow_uncalibrated_fallback=True,
    )

# Module-level raw-activation cache populated by main() when
# --activation-cache-dir is provided AND any of the activation-aware
# passes (--gptq / --act-weighted-round / --scale-sweep) are enabled. Keyed
# by recipe name; values are 2D `[N, in_features]` float32 tensors
# (lazily upcast from the on-disk bfloat16 for numerical stability
# during Hessian + per-channel stats). None means "not loaded".
def _gptq_column_block_size(cols: int,
    *,
    levers: NativeRenderLevers | None = None,
) -> int:
    if levers is None:
        levers = NativeRenderLevers.resolve()
    return max(1, min(int(cols), levers.gptq_block_size))


def _gptq_columnwise_update(
    W: torch.Tensor,
    U: torch.Tensor,
    *,
    block_size: int,
    quantize_column: Callable[[torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """Run the FP-Quant/GPTQ column update with fixed quantizer params.

    This matches FP-Quant's block loop: quantize one column, propagate the
    OBS error through the remaining columns in the current GPTQ block, then
    apply the accumulated block error to later blocks.
    """
    _rows, cols = W.shape
    block_size = max(1, min(int(block_size), int(cols)))
    for block_start in range(0, cols, block_size):
        block_end = min(block_start + block_size, cols)
        ncols = block_end - block_start
        block = W[:, block_start:block_end].clone()
        errs = torch.zeros_like(block)
        U_block = U[block_start:block_end, block_start:block_end]
        for i in range(ncols):
            col = block[:, i]
            col_idx = block_start + i
            col_dq = quantize_column(col, col_idx).to(
                device=W.device,
                dtype=W.dtype,
            )
            W[:, col_idx] = col_dq
            denom = U_block[i, i].clamp_min(1e-12)
            err = (col - col_dq) / denom
            block[:, i:].addr_(err, U_block[i, i:], alpha=-1)
            errs[:, i] = err
        if block_end < cols:
            W[:, block_end:].addmm_(
                errs,
                U[block_start:block_end, block_end:],
                alpha=-1,
            )
    return W


def gptq_damp_sweep_enabled(
    *,
    levers: NativeRenderLevers | None = None,
) -> bool:
    """Whether GPTQ runs the legacy 5-candidate in-sample damp sweep.

    DEFAULT OFF as of 2026-06-12 (Robert: "hard code it for now to your
    empirical best finding"). The V1 served A/B found the sweep's
    in-sample evaluator picks inverted winners: fixed damp 0.3 beat the
    sweep on every gold-lane readout across two calibration draws
    (all-position KL −6.6/−11.5%, WikiText-test PPL −0.9/−1.4%, tail NLL
    improved) at ~4.4x less render time (docs/unified_render_theory.md
    §8 V1). Set PRISMAQUANT_GPTQ_DAMP_SWEEP=1 to reproduce historical
    sweep-rendered artifacts.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve(gptq_default=True)
    return levers.gptq_damp_sweep


def _resolve_gptq_fixed_damp(
    *,
    levers: NativeRenderLevers | None = None,
) -> float:
    """Fixed GPTQ damp used when the damp sweep is disabled.

    Default 1.0 = the V1 served A/B winner over THREE calibration draws:
    wins all-position KL and confident KL 3/3 draws vs damp 0.3, PPL 2/3
    and on average (26.80 vs 27.03); 0.3's only consistent edge was the
    max-of-16-chunks tail at shrinking margins (+0.3% on the last draw).
    ``PRISMAQUANT_GPTQ_DAMP`` overrides (0.01 reproduces vanilla GPTQ).
    Sweep paths pass explicit candidates and ignore this. Open research:
    derive the per-Linear optimum from weights/activations alone
    (docs/unified_render_theory.md §9.7 — two closed forms refuted so far).
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    return levers.gptq_fixed_damp


# Per-role GPTQ damp override (research lever, default-off). The unified-render
# V0b/V0c held-out basins are ROLE-structured (attention/o_proj -> 1.0,
# gate/up -> ~0.3, down_proj -> ~3.0); the optimal damp tracks the activation-
# conditioning role, NOT downstream sensitivity (measured Spearman(h_trace,
# opt_damp) ~= 0 on the 31 logged 4B Linears), while the *cost* of one global
# constant concentrates on the high-h_trace gate/up Linears (Spearman ~= +0.6).
# This lever lets a served A/B test the per-role table against the fixed-1.0
# default without reviving the in-sample sweep (docs/unified_render_theory.md
# §7-8). Spec, comma-separated role=damp pairs:
#   PRISMAQUANT_GPTQ_DAMP_ROLES="qkv=1.0,o_proj=1.0,gate_up=0.3,down=3.0"
# Roles: qkv, o_proj, gate_up, down, other. Unmatched roles and unparseable
# entries fall back to _resolve_gptq_fixed_damp(); unset => exact no-op.


def _gptq_role_of(qname: str) -> str:
    """Map a Linear name to a render role for per-role damp selection."""
    q = str(qname)
    if "down_proj" in q:
        return "down"
    if "gate_proj" in q or "up_proj" in q:
        return "gate_up"
    if "o_proj" in q:
        return "o_proj"
    if "q_proj" in q or "k_proj" in q or "v_proj" in q:
        return "qkv"
    return "other"





def _resolve_gptq_damp_for_role(qname: str,
    *,
    levers: NativeRenderLevers | None = None,
) -> float:
    """Per-role GPTQ damp (research lever); falls back to the fixed default.

    Returns ``_resolve_gptq_fixed_damp()`` unchanged when
    ``PRISMAQUANT_GPTQ_DAMP_ROLES`` is unset or the role is unlisted, so the
    production default (global fixed 1.0) is preserved bit-for-bit.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    return dict(levers.gptq_damp_roles).get(_gptq_role_of(qname), levers.gptq_fixed_damp)


def _gptq_obs_rounding_nvfp4(
    weight: torch.Tensor, activations: torch.Tensor,
    group_size: int = 16, damp: float | None = None,
    global_real_override: torch.Tensor | None = None,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    static_act_order: bool = False,
    joint_scale_opt: bool = False,

    *,
    levers: NativeRenderLevers | None = None,
) -> torch.Tensor:
    """GPTQ one-shot OBS rounding for NVFP4 weights.

    Standard GPTQ (Frantar et al. 2022): build the activation covariance
    `H = X^T X + λ·diag(H)`, invert via Cholesky, then round columns with
    fixed per-group scales. Error from each column's quant is propagated via
    `H_inv`, which is the closed-form OBS update for least-squares loss
    `||W - W_q||_H^2`.

    Returns the dequantized, error-propagated weight `[out, in]`
    (float32). The caller still runs NVFP4 packing on this tensor to
    produce on-disk storage — the bits end up the same as if we had
    quantized `weight` directly but with a smaller output-space error.

    `damp = 0.01` adds `0.01·mean(diag(H))` to `diag(H)` for Cholesky
    stability. `global_real_override` threads through for fused-sibling
    consistency (same semantics as `quantize_dequantize_nvfp4`).

    `static_act_order` applies Lift/MR-GPTQ style activation ordering without
    requiring a runtime column permutation: scales are selected in the original
    NVFP4 group layout, columns are processed in descending activation
    importance, then the result is unpermuted before packing.

    `joint_scale_opt` jointly searches the NVFP4 tensor global and per-group
    max-to-codebook-level scale choices used by GPTQ. Its candidate set
    contains max-to-6 and max-to-4, so FourOverSix is a strict subset.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    if damp is None:
        damp = _resolve_gptq_fixed_damp( levers=levers)
    W = weight.to(torch.float32).clone()
    rows, cols = W.shape
    if cols % group_size != 0:
        raise ValueError(f"GPTQ requires group_size={group_size} ∤ {cols}")

    # #42: per-token activation clipping to reduce Hessian condition
    # number. PRISMAQUANT_ACT_CLIP_QUANTILE in (0,1) clamps each token's
    # activations to ±|q-th percentile| of |x|. 0.999 removes ~4 extreme
    # outliers per 4k-dim row; condition number drops materially with
    # near-zero impact on bulk distribution. Set "0" or out-of-range to
    # disable. The same clipped matrix is used by local gates/sweeps so
    # those gates score the objective the candidate was optimized under.
    X = _activation_matrix_for_gptq(
        activations,
        cols,
        device=W.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=fisher_row_weights,
     levers=levers)
    # H = X^T X; guard against near-zero diagonal (dead channels).
    H = X.t() @ X                                         # [in, in]
    # Dead-channel handling: columns whose H diagonal is non-positive
    # (all-zero activation channel). Detect BEFORE damping — damping
    # lifts every diagonal above zero, which made this check
    # unreachable — and exclude dead entries from the damp reference
    # mean so a mass of dead channels can't deflate it. Dead diagonals
    # get an identity entry so the Cholesky succeeds. We do NOT zero
    # the weights of dead columns (deliberate serving-safe deviation
    # from reference GPTQ — a column unexercised by calibration must
    # not be destroyed for serving traffic); with an identity-like
    # row/col their OBS error propagation is a no-op and they quantize
    # as plain RTN.
    diag0 = torch.diagonal(H)
    dead = diag0 <= 0
    alive = ~dead
    diag_mean = (
        diag0[alive].mean() if bool(alive.any()) else diag0.new_ones(())
    ).clamp_min(1e-12)
    if dead.any():
        H[dead, dead] = 1.0
    H.diagonal().add_(damp * diag_mean)

    col_importance = torch.diagonal(H).detach().clone().clamp_min(1e-12)

    # Target NVFP4 grid. Pre-compute the per-tensor global_real so the
    # per-block quantization uses the same outer scale as the final
    # on-disk packing (otherwise error propagation would be under an
    # inconsistent scale). This mirrors quantize_dequantize_nvfp4.
    if global_real_override is not None:
        global_real = global_real_override.to(weight.device).clamp_min(1e-12).float()
    else:
        grouped_full = W.reshape(rows, cols // group_size, group_size)
        scale_rule = (
            NVFP4_SCALE_RULE_JOINT_MSE
            if joint_scale_opt
            else None
        )
        s_g_real_full = _select_nvfp4_group_scales(
            grouped_full,
            scale_rule=scale_rule,
         levers=levers)
        global_real = (s_g_real_full.amax() / FP8_E4M3_MAX).clamp_min(1e-12)
        if joint_scale_opt:
            global_real = _optimize_nvfp4_joint_global_real(
                W,
                group_size=group_size,
                base_global_real=global_real,
             levers=levers)

    scales_by_group = torch.empty(
        (rows, cols // group_size),
        dtype=torch.float32,
        device=W.device,
    )
    for group_idx, block_start in enumerate(range(0, cols, group_size)):
        block_end = block_start + group_size
        block = W[:, block_start:block_end]
        if joint_scale_opt:
            eff = _select_nvfp4_joint_gptq_eff_scale(
                block,
                global_real,
                col_importance=col_importance[block_start:block_end],
             levers=levers)
        else:
            s_g_real = _select_nvfp4_group_scales(block, levers=levers)
            eff = _nvfp4_effective_scale_from_real(
                s_g_real,
                global_real,
                quantize_fp8=True,
            )
        scales_by_group[:, group_idx] = eff
    scale_by_col = scales_by_group.repeat_interleave(group_size, dim=1)

    inverse_perm: torch.Tensor | None = None
    if static_act_order:
        perm = torch.argsort(col_importance, descending=True)
        inverse_perm = torch.empty_like(perm)
        inverse_perm[perm] = torch.arange(cols, device=W.device)
        W = W.index_select(1, perm).contiguous()
        H = H.index_select(0, perm).index_select(1, perm).contiguous()
        scale_by_col = scale_by_col.index_select(1, perm).contiguous()

    # Compute Cholesky + inverse. We follow the GPTQ paper's trick of
    # computing an upper-triangular inverse (`torch.cholesky_inverse`
    # then Cholesky again) so the column-wise update becomes a simple
    # multiplication by an upper-triangular factor.
    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        # Upper-triangular factor U such that U^T U = Hinv (GPTQ uses U
        # directly for the column updates).
        U = torch.linalg.cholesky(Hinv, upper=True)
    except Exception:
        # Fall back to RTN if the Cholesky numerically fails (rare:
        # extreme activation degeneracy).  Returning the original weight
        # here is not a valid NVFP4 render in compute_only/cache paths and
        # can make downstream local-MSE gates see an impossible zero error.
        # Pass `global_real` — the fused-sibling override when supplied,
        # else the per-tensor global computed above (which is the
        # JSO-optimized joint global under joint_scale_opt) — so the
        # fallback doesn't silently discard the joint-global pick.
        return _rtn_dequant_nvfp4(
            weight,
            group_size=group_size,
            global_real_override=global_real,
         levers=levers)

    def _quantize_nvfp4_col(col: torch.Tensor, col_idx: int) -> torch.Tensor:
        eff_scale = scale_by_col[:, col_idx:col_idx + 1].clamp_min(1e-12)
        _idx, col_dq = _nvfp4_quantize_dequantize_with_eff_scale(
            col.unsqueeze(1),
            eff_scale,
        )
        return col_dq.squeeze(1)

    W = _gptq_columnwise_update(
        W,
        U,
        block_size=_gptq_column_block_size(cols, levers=levers),
        quantize_column=_quantize_nvfp4_col,
    )
    if static_act_order:
        assert inverse_perm is not None
        return W.index_select(1, inverse_perm).contiguous()
    return W


def _gptq_obs_rounding_nvfp4_swept(
    weight: torch.Tensor, activations: torch.Tensor,
    group_size: int = 16,
    global_real_override: torch.Tensor | None = None,
    damp_candidates: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05, 0.1),
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    static_act_order: bool = False,
    joint_scale_opt: bool = False,
    linear_name: str | None = None,

    *,
    levers: NativeRenderLevers | None = None,
) -> torch.Tensor:
    """Per-Linear GPTQ damping sweep.

    For each candidate damping value, run the standard
    `_gptq_obs_rounding_nvfp4` and measure the Hessian-weighted
    reconstruction error `tr((W − W_q)^T H (W − W_q))`. Return the
    rounded weight from the candidate with the smallest error.

    Cost: ~|candidates|× the unswept call (Cholesky+propagation
    repeats per candidate). Memory: `H` is recomputed each pass; we
    keep only the best `W_q` so far. For the typical 5-candidate
    sweep on a 4k×4k Linear, total wallclock ≈ 5× single-damp.

    Quality: typically 0.02–0.05 PPL gain on Llama-class models
    because the optimal damping varies by Linear (attention out-proj
    likes higher damp; MLP gate/up like lower).

    Caller convention matches `_gptq_obs_rounding_nvfp4`. When the
    Cholesky fallback fires (degenerate H), we return the best
    successful pass; if all fail, we return the unswept fallback.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    W_orig = weight.to(torch.float32)
    X = _activation_matrix_for_gptq(
        activations,
        weight.shape[1],
        device=weight.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=fisher_row_weights,
     levers=levers)
    H_full = X.t() @ X  # [in, in], shared evaluator

    # Optional research instrumentation (#46-followup): log per-Linear
    # H spectrum + per-damp errors so we can fit an analytical damp
    # picker. Env-gated; cost is one eigvalsh per Linear (~O(n^3)
    # where n=in_features; tens of ms on 4k×4k).
    log_path = os.environ.get("PRISMAQUANT_DAMP_SWEEP_LOG")
    if log_path:
        try:
            eigvals = torch.linalg.eigvalsh(H_full.to(torch.float64)).to(torch.float32)
            lambda_max = float(eigvals[-1].item())
            positive = eigvals[eigvals > 1e-30]
            lambda_min = float(positive.min().item()) if positive.numel() > 0 else 0.0
            mean_diag = float(torch.diagonal(H_full).mean().item())
        except Exception:
            lambda_max = float("nan")
            lambda_min = float("nan")
            mean_diag = float("nan")

    # Optional analytical damp picker: skip the 5-candidate sweep and
    # pick damp = c * lambda_max(H) / mean(diag(H)) directly.
    # Equivalent to a kappa-target with K=10 (kappa target reduces to
    # this form when lambda_min ~= 0, which is true on nearly every
    # production Linear). c=1.784e-5 fitted on Qwen3-4B's 450 logged
    # damp-sweep winners (log-MSE 0.172 = typical prediction within
    # ~2.4x of the parabolic-interpolated continuous optimum, well
    # inside the 5x gap between sweep candidates).
    # Cost: 1 GPTQ pass + ~10-iter power iteration for lambda_max,
    # vs 5 GPTQ passes for the sweep. Net ~5x speedup.
    if levers.gptq_damp_analytical:
        try:
            c = levers.gptq_damp_analytical_c
            mean_diag_a = float(torch.diagonal(H_full).mean().item())
            if mean_diag_a > 0:
                # Power iteration for the dominant eigenvalue of H.
                n = H_full.shape[0]
                H_f = H_full.to(torch.float32)
                v = torch.randn(n, device=H_full.device, dtype=torch.float32)
                v = v / v.norm().clamp_min(1e-30)
                for _ in range(10):
                    v = H_f @ v
                    v = v / v.norm().clamp_min(1e-30)
                lambda_max_est = float((v @ (H_f @ v)).item())
                damp_pred = c * lambda_max_est / mean_diag_a
                damp_pred = min(max(damp_pred, 0.001), 0.1)
                w_q = _gptq_obs_rounding_nvfp4(
                    weight, activations, group_size=group_size,
                    damp=damp_pred, global_real_override=global_real_override,
                    clip_threshold=clip_threshold,
                    clip_rescale=clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                    static_act_order=static_act_order,
                    joint_scale_opt=joint_scale_opt,
                 levers=levers)
                diff = W_orig - w_q.to(torch.float32)
                err = float(torch.einsum("oi,ij,oj->", diff, H_full, diff))
                if math.isfinite(err) and err > 0:
                    if log_path:
                        import json as _json
                        entry = {
                            "linear_name": linear_name,
                            "shape": list(weight.shape),
                            "lambda_max_est": lambda_max_est,
                            "mean_diag": mean_diag_a,
                            "analytical_damp": damp_pred,
                            "analytical_err": err,
                        }
                        try:
                            with open(log_path, "a") as f:
                                f.write(_json.dumps(entry) + "\n")
                        except Exception:
                            pass
                    return w_q
        except Exception:
            pass  # fall through to the 5-candidate sweep

    best_w = None
    best_err = float("inf")
    best_damp: float | None = None
    per_damp_err: dict[float, float] = {}
    for damp in damp_candidates:
        try:
            w_q = _gptq_obs_rounding_nvfp4(
                weight, activations, group_size=group_size,
                damp=damp, global_real_override=global_real_override,
                clip_threshold=clip_threshold,
                clip_rescale=clip_rescale,
                fisher_row_weights=fisher_row_weights,
                static_act_order=static_act_order,
                joint_scale_opt=joint_scale_opt,
             levers=levers)
        except Exception:
            per_damp_err[damp] = float("inf")
            continue
        # Hessian-weighted reconstruction error (no damp injected here —
        # we want raw H for fair comparison across candidates).
        diff = W_orig - w_q.to(torch.float32)
        err = float(torch.einsum("oi,ij,oj->", diff, H_full, diff))
        per_damp_err[damp] = err
        if err < best_err:
            best_err = err
            best_w = w_q
            best_damp = damp
    if log_path:
        import json as _json
        entry = {
            "linear_name": linear_name,
            "shape": list(weight.shape),
            "lambda_max": lambda_max,
            "lambda_min": lambda_min,
            "mean_diag": mean_diag,
            "best_damp": best_damp,
            "best_err": best_err if best_err != float("inf") else None,
            "per_damp_err": {f"{k:.4g}": v for k, v in per_damp_err.items()},
        }
        try:
            with open(log_path, "a") as f:
                f.write(_json.dumps(entry) + "\n")
        except Exception:
            pass
    if best_w is None:
        return _rtn_dequant_nvfp4(
            W_orig,
            group_size=group_size,
            global_real_override=global_real_override,
         levers=levers)
    return best_w


def _scale_sweep_nvfp4(
    weight: torch.Tensor, activations: torch.Tensor,
    group_size: int = 16,
    global_real_override: torch.Tensor | None = None,
    grid: int = 32,
    span: tuple[float, float] = (0.5, 1.5),
    reference_weight: torch.Tensor | None = None,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,

    *,
    levers: NativeRenderLevers | None = None,
) -> torch.Tensor:
    """Per-group joint (scale, rounding) closed-form polish.

    For each NVFP4 group, sweep `grid` candidate scales spanning
    `[span[0]·s0, span[1]·s0]`, where s0 is the default max-abs scale
    derived from the pre-pass weight. For each candidate scale, run RTN
    on the NVFP4 codebook and compute the activation-weighted MSE
    `sum_j a_j²·(w_orig,j - w_q,j)²` against the ORIGINAL (pre-pass)
    weight. Keep the configuration minimizing MSE per group, with an
    improve-or-keep gate against whatever `weight` is coming in.

    `reference_weight`: the pre-pass (float32) weight used to measure
    MSE. Defaults to `weight` (the post-pass state) when not supplied
    — in that case the gate degenerates to "improve over no-op" which
    is not useful. Callers who want the gate to work should pass the
    original weight explicitly.

    Closed-form analog of AutoRound's SGD on per-weight V offsets:
    AutoRound searches a continuous relaxation; we enumerate the
    discrete scale dimension directly. Per-weight rounding at each
    scale is RTN (optimal conditional on the scale).

    Output is a dequantized tensor on valid NVFP4 grid points under the
    new per-group scales. The downstream packer re-derives fp8_scale
    through the active NVFP4 scale rule; FourOverSix gives the packer a
    second legal max-to-4 representation when max-to-6 would lose a swept
    scale choice.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    W_in = weight.to(torch.float32).contiguous()
    W_ref = (reference_weight if reference_weight is not None else W_in
             ).to(torch.float32).contiguous()
    if W_in.shape != W_ref.shape:
        raise ValueError(
            f"scale-sweep: weight shape {tuple(W_in.shape)} != "
            f"reference_weight shape {tuple(W_ref.shape)}")
    rows, cols = W_in.shape
    if cols % group_size != 0:
        raise ValueError(f"scale-sweep requires group_size={group_size} ∤ {cols}")

    if clip_threshold is not None and clip_threshold > 0.0:
        a = _activation_matrix_for_gptq(
            activations,
            cols,
            device=W_in.device,
            clip_threshold=clip_threshold,
            clip_rescale=clip_rescale,
            row_weights=fisher_row_weights,
         levers=levers)
    else:
        a = _activation_matrix_for_gptq(
            activations,
            cols,
            device=W_in.device,
            clip_quantile=0.0,
            row_weights=fisher_row_weights,
         levers=levers)
    col_importance = a.pow(2).mean(dim=0).clamp_min(1e-12)  # [in]

    # Use the REFERENCE weight to set the default per-group scale (s0)
    # and to measure MSE against.
    ref_grouped = W_ref.reshape(rows, cols // group_size, group_size)
    in_grouped = W_in.reshape(rows, cols // group_size, group_size)
    s_g_real = _select_nvfp4_group_scales(ref_grouped, levers=levers)
    if global_real_override is not None:
        global_real = global_real_override.to(W_in.device).clamp_min(1e-12).float()
    else:
        global_real = (s_g_real.amax() / FP8_E4M3_MAX).clamp_min(1e-12)
    eff_scale0 = _nvfp4_effective_scale_from_real(
        s_g_real,
        global_real,
        quantize_fp8=True,
    ).unsqueeze(-1)
    col_imp = col_importance.reshape(1, cols // group_size, group_size)  # [1, n_g, gs]

    # Incoming per-group MSE against reference.
    init_mse = (col_imp * (ref_grouped - in_grouped).pow(2)).sum(dim=-1)  # [rows, n_g]

    # Sweep scales. The full intermediate tensor
    # [rows, n_g, grid, gs, 15] would peak at >70 GB for a 12288-row
    # Linear × 192 groups × 32 scales × 16 weights × 15 codes × 4 B.
    # Chunk over rows so peak memory stays bounded regardless of size.
    mults = torch.linspace(span[0], span[1], grid,
                           device=W_in.device, dtype=torch.float32)  # [grid]

    # Target per-chunk intermediate budget: ~2 GB max on the biggest
    # tensor `d = [chunk, n_g, grid, gs, len(cb)]` (float32).
    n_g = cols // group_size
    bytes_per_row = n_g * grid * group_size * (2 * len(_nvfp4_activation_contract._E2M1_POSITIVE) - 1) * 4
    chunk_target = max(1, (2 * 1024 * 1024 * 1024) // max(1, bytes_per_row))
    row_chunk = min(rows, int(chunk_target))

    result_groups = torch.empty_like(ref_grouped)
    for r0 in range(0, rows, row_chunk):
        r1 = min(r0 + row_chunk, rows)
        scales_c = eff_scale0[r0:r1].squeeze(-1).unsqueeze(-1) * mults  # [c, n_g, grid]
        ref_c = ref_grouped[r0:r1]
        in_c = in_grouped[r0:r1]
        init_mse_c = init_mse[r0:r1]

        gexp = ref_c.unsqueeze(2)                   # [c, n_g, 1, gs]
        sexp = scales_c.unsqueeze(3)                # [c, n_g, grid, 1]
        _idx, Wq_cand = _nvfp4_quantize_dequantize_with_eff_scale(
            gexp,
            sexp,
        )                                           # [c, n_g, grid, gs]
        err = col_imp.unsqueeze(2) * (gexp - Wq_cand).pow(2)  # [c, n_g, grid, gs]
        mse = err.sum(dim=-1)                        # [c, n_g, grid]
        del err
        best = mse.argmin(dim=-1)                    # [c, n_g]
        bidx = best.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, group_size)
        chosen_Wq = Wq_cand.gather(2, bidx).squeeze(2)           # [c, n_g, gs]
        chosen_mse = mse.gather(2, best.unsqueeze(-1)).squeeze(-1)  # [c, n_g]
        del Wq_cand, mse, best, bidx

        use_new = chosen_mse < init_mse_c
        result_groups[r0:r1] = torch.where(
            use_new.unsqueeze(-1).expand(-1, -1, group_size),
            chosen_Wq,
            in_c,
        )
    return result_groups.reshape(rows, cols)


def compute_nvfp4_global_real(weight: torch.Tensor, group_size: int = 16
                              ,
    *,
    levers: NativeRenderLevers | None = None,
) -> torch.Tensor:
    """Return the per-tensor `global_real` that NVFP4 packing would
    pick for `weight` alone. Useful for fused-sibling pre-pass: caller
    takes the max across siblings and passes the joint value back into
    `quantize_dequantize_nvfp4(global_real_override=...)`."""
    if levers is None:
        levers = NativeRenderLevers.resolve()
    rows, cols = weight.shape
    grouped = weight.float().reshape(rows, cols // group_size, group_size)
    _s_g_real, global_real = _select_nvfp4_pack_scales_and_global(grouped, levers=levers)
    return global_real


def quantize_dequantize_nvfp4(
    weight: torch.Tensor, group_size: int = 16,
    global_real_override: torch.Tensor | None = None,

    *,
    levers: NativeRenderLevers | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply NVFP4 RTN to a 2D `[rows, cols]` weight and return the
    on-disk triple `(weight_packed, weight_scale, weight_global_scale)`
    in the **compressed-tensors NVFP4 convention**:

      - per-group dequant scale  s_g_real from active NVFP4 scale rule
        (`static_6` maps max-abs(group) to ±6; FourOverSix also tests ±4)
      - per-tensor outer scale   global   = max(s_g_real) / FP8_E4M3_MAX
        (so the fp8-stored per-group scale stays inside [0, 448])
      - on-disk weight_scale (fp8) = s_g_real / global  ∈ [0, 448]
      - on-disk weight_global_scale = 1 / global  (DIVISOR)
        vLLM inverts on load: `layer.weight_global_scale = 1/loaded`
        → recovers `global` and applies it as the per-tensor multiplier
        in the NVFP4 GEMM.

    Dequant in the kernel: `weight ≈ codebook[index] · weight_scale_fp8 · global`

    `global_real_override` lets a caller force a particular per-tensor
    scale — used for fused siblings (q/k/v, gate/up) that vLLM expects
    to share one global_scale slot. Pass the max across the sibling
    group's natural global_real values.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"NVFP4 group_size={group_size} ∤ {cols}")
    n_groups = cols // group_size
    grouped = weight.float().reshape(rows, n_groups, group_size)
    s_g_real, global_real = _select_nvfp4_pack_scales_and_global(
        grouped,
        global_real_override=global_real_override,
     levers=levers)
    codec = _nvfp4_quantize_grouped_codec(
        grouped,
        global_real=global_real,
        scale_real=s_g_real,
     levers=levers)
    fp4_idx = codec.indices.reshape(rows, cols)
    weight_packed = pack_fp4_indices(fp4_idx, cols)
    return (
        weight_packed,
        codec.scale,
        (1.0 / global_real).to(torch.float32).reshape(1),  # divisor convention
    )


def _rtn_dequant_nvfp4(
    weight: torch.Tensor, group_size: int = 16,
    global_real_override: torch.Tensor | None = None,

    *,
    levers: NativeRenderLevers | None = None,
) -> torch.Tensor:
    """RTN to NVFP4 grid, returning FP32 dequantized weights (no GPTQ
    error propagation, no scale sweep). Used by the do-no-harm gate
    (#do-no-harm) to compare against post-GPTQ/sweep state and revert
    if a Linear locally regressed."""
    if levers is None:
        levers = NativeRenderLevers.resolve()
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"NVFP4 group_size={group_size} ∤ {cols}")
    n_groups = cols // group_size
    W = weight.float()
    grouped = W.reshape(rows, n_groups, group_size)
    s_g_real, global_real = _select_nvfp4_pack_scales_and_global(
        grouped,
        global_real_override=global_real_override,
     levers=levers)
    codec = _nvfp4_quantize_grouped_codec(
        grouped,
        global_real=global_real,
        scale_real=s_g_real,
     levers=levers)
    return codec.dequant.reshape(rows, cols)


def render_nvfp4_dequant(
    weight: torch.Tensor,
    *,
    group_size: int = 16,
    global_real_override: torch.Tensor | None = None,
    scale_rule: str = NVFP4_SCALE_RULE_STATIC_6,
    snapped_scale_scoring: bool = False,
    joint_scale_levels: tuple[float, ...] = (6.0, 4.0),

    levers: NativeRenderLevers | None = None,
) -> torch.Tensor:
    """Render a 2-D weight through the exact production NVFP4 codec.

    Unlike the historical private RTN helper, every scale-plane choice is an
    explicit argument.  Offline selectors such as PrismaSnap must be
    reproducible from their receipt and therefore cannot inherit either
    ``PRISMAQUANT_NVFP4_SCALE_RULE`` or the research-only snapped-scale
    scoring switch from the process environment.  Existing exporter call
    sites continue to use their unchanged environment/default behavior.
    """
    if levers is None:
        levers = NativeRenderLevers()
    if weight.ndim != 2:
        raise ValueError(
            "production NVFP4 dequant render requires a rank-2 weight; "
            f"got shape={tuple(weight.shape)}"
        )
    rows, cols = weight.shape
    if cols % int(group_size) != 0:
        raise ValueError(f"NVFP4 group_size={group_size} ∤ {cols}")
    grouped = weight.float().reshape(
        rows, cols // int(group_size), int(group_size)
    )
    scale_real, global_real = _select_nvfp4_pack_scales_and_global(
        grouped,
        global_real_override=global_real_override,
        scale_rule=resolve_nvfp4_scale_rule(scale_rule),
        snapped_scale_scoring=snapped_scale_scoring,
        joint_scale_levels=joint_scale_levels,
     levers=levers)
    codec = _nvfp4_quantize_grouped_codec(
        grouped,
        global_real=global_real,
        scale_real=scale_real,
        scale_rule=resolve_nvfp4_scale_rule(scale_rule),
     levers=levers)
    return codec.dequant.reshape(rows, cols)


def nvfp4_global_real(
    weight: torch.Tensor,
    *,
    group_size: int = 16,
    scale_rule: str = NVFP4_SCALE_RULE_STATIC_6,
    snapped_scale_scoring: bool = False,
    joint_scale_levels: tuple[float, ...] = (6.0, 4.0),

    levers: NativeRenderLevers | None = None,
) -> torch.Tensor:
    """Return the explicit production-codec global multiplier for a weight.

    This is the companion to :func:`render_nvfp4_dequant` used when vLLM
    fuses sibling projections into one runtime parameter and therefore makes
    them share the maximum of their natural globals.
    """
    if levers is None:
        levers = NativeRenderLevers()
    if weight.ndim != 2:
        raise ValueError(
            "production NVFP4 global calculation requires a rank-2 weight; "
            f"got shape={tuple(weight.shape)}"
        )
    rows, cols = weight.shape
    if cols % int(group_size) != 0:
        raise ValueError(f"NVFP4 group_size={group_size} ∤ {cols}")
    grouped = weight.float().reshape(
        rows, cols // int(group_size), int(group_size)
    )
    _scale_real, global_real = _select_nvfp4_pack_scales_and_global(
        grouped,
        scale_rule=resolve_nvfp4_scale_rule(scale_rule),
        snapped_scale_scoring=snapped_scale_scoring,
        joint_scale_levels=joint_scale_levels,
     levers=levers)
    return global_real


def quantize_dequantize_nvfp4_packed(
    packed: torch.Tensor, group_size: int = 16,

    *,
    levers: NativeRenderLevers | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-expert NVFP4 packing for a 3D `[E, M, N]` packed tensor.
    Each expert gets its own `global_real` (so the weight_global_scale
    output has shape `[E]`); the on-disk values are divisors (1/scale)
    matching the compressed-tensors convention.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    E, M, N = packed.shape
    if N % group_size != 0:
        raise ValueError(f"NVFP4 group_size={group_size} ∤ {N}")
    g = N // group_size
    grouped = packed.float().reshape(E, M, g, group_size)
    s_g_real = _select_nvfp4_group_scales(grouped, levers=levers)                          # [E, M, g]
    global_real = (s_g_real.reshape(E, -1).amax(dim=-1) / FP8_E4M3_MAX).clamp_min(1e-12)  # [E]
    for _ in range(3):
        snapped = _select_nvfp4_group_scales(
            grouped,
            global_real=global_real.view(E, 1, 1),
         levers=levers)
        next_global = (
            snapped.reshape(E, -1).amax(dim=-1) / FP8_E4M3_MAX
        ).clamp_min(1e-12)
        s_g_real = snapped
        if torch.allclose(next_global, global_real, rtol=0.0, atol=1e-12):
            break
        global_real = next_global
    codec = _nvfp4_quantize_grouped_codec(
        grouped,
        global_real=global_real.view(E, 1, 1),
        scale_real=s_g_real,
     levers=levers)
    fp4_idx = codec.indices.reshape(E, M, N)
    weight_packed = pack_fp4_indices(fp4_idx, N)
    return (
        weight_packed,
        codec.scale,
        (1.0 / global_real).to(torch.float32),
    )


# ---------------------------------------------------------------------------
# MXFP8_E4M3/MXFP8_E5M2 packing (FP8 element format, E8M0 per-group scale).
# ---------------------------------------------------------------------------
MXFP8_E4M3_MAX = 448.0   # max representable in fp8_e4m3fn
MXFP4_E2M1_MAX = NVFP4_MAX


def _fp8_element_dtype_and_max(fmt: str) -> tuple[torch.dtype, float]:
    fmt_u = str(fmt).upper()
    if fmt_u.endswith("E5M2"):
        dtype = torch.float8_e5m2
    else:
        dtype = torch.float8_e4m3fn
    try:
        max_value = float(torch.finfo(dtype).max)
    except Exception:
        max_value = MXFP8_E4M3_MAX if dtype is torch.float8_e4m3fn else 57344.0
    return dtype, max_value


def _fp8_codec(
    values: torch.Tensor,
    *,
    scale: torch.Tensor,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = FP8_E4M3_MAX,
) -> _FP8CodecResult:
    values_f = values.to(torch.float32)
    scale_f = scale.to(
        device=values_f.device,
        dtype=torch.float32,
    ).clamp_min(2.0 ** -127)
    quant = (values_f / scale_f).clamp(
        -float(element_max),
        float(element_max),
    ).to(element_dtype)
    dequant = quant.to(torch.float32) * scale_f
    return _FP8CodecResult(
        quant=quant,
        scale=scale_f,
        dequant=dequant,
    )


def _fp8_dynamic_codec(
    values: torch.Tensor,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = FP8_E4M3_MAX,
) -> _FP8CodecResult:
    result = fp8_dynamic_weight_qdq(
        values,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return _FP8CodecResult(
        quant=result.quant,
        scale=result.scale,
        dequant=result.dequant,
    )


def _fp8_dequantize(
    quant: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return quant.to(torch.float32) * scale.to(
        device=quant.device,
        dtype=torch.float32,
    )


def _mx_base_exponent_from_amax(
    amax: torch.Tensor,
    *,
    element_max: float,
) -> torch.Tensor:
    if math.isclose(float(element_max), MXFP4_E2M1_MAX, rel_tol=0.0, abs_tol=1e-6):
        return generate_mx_scales(amax, num_bits=4).to(torch.float32) - 127.0
    if math.isclose(float(element_max), MXFP8_E4M3_MAX, rel_tol=0.0, abs_tol=1e-6):
        return generate_mx_scales(amax, num_bits=8).to(torch.float32) - 127.0

    # compressed-tensors only exposes MX scale generation for FP4 E2M1 and
    # FP8 E4M3. Keep the local fallback for research-only FP8_E5M2.
    rounded = _mx_rounded_amax_power2(amax)
    element_offset = int(math.floor(math.log2(float(element_max))))
    exponent = torch.floor(torch.log2(rounded)) - float(element_offset)
    return exponent.clamp(-127, 127)


def _mxfp8_base_exponent(
    grouped: torch.Tensor,
    *,
    element_max: float,
) -> torch.Tensor:
    return _mx_base_exponent_from_amax(
        grouped.abs().amax(dim=-1),
        element_max=element_max,
    )


def _mxfp8_grouped_codec(
    grouped: torch.Tensor,
    *,
    e8m0_unbiased: torch.Tensor | None = None,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = MXFP8_E4M3_MAX,
) -> _MXFP8CodecResult:
    if (
        e8m0_unbiased is None
        and element_dtype == torch.float8_e4m3fn
        and float(element_max) == float(MXFP8_E4M3_MAX)
    ):
        ungrouped = grouped.reshape(
            *grouped.shape[:-2],
            grouped.shape[-2] * grouped.shape[-1],
        )
        result = mxfp8_e4m3_qdq(ungrouped)
        return _MXFP8CodecResult(
            quant=result.quant.reshape_as(grouped),
            scale=result.scale,
            dequant=result.dequant.reshape_as(grouped),
        )

    grouped_f = grouped.to(torch.float32)
    if e8m0_unbiased is None:
        e8m0_unbiased = _mxfp8_base_exponent(
            grouped_f,
            element_max=element_max,
        )
    e = e8m0_unbiased.to(grouped_f.device, dtype=torch.float32).clamp(-127, 127)
    scale = torch.pow(
        torch.tensor(2.0, device=grouped_f.device, dtype=torch.float32),
        e,
    )
    fp8 = _fp8_codec(
        grouped_f,
        scale=scale.unsqueeze(-1),
        element_dtype=element_dtype,
        element_max=element_max,
    )
    e8m0_uint8 = (e + 127).to(torch.int32).clamp(0, 255).to(torch.uint8)
    return _MXFP8CodecResult(
        quant=fp8.quant,
        scale=e8m0_uint8,
        dequant=fp8.dequant,
    )


def _mxfp4_grouped_codec(grouped: torch.Tensor) -> _MXFP4CodecResult:
    """Return MXFP4 packed codes, E8M0 scale, and reconstructed values.

    ``grouped`` must have the 32-value MX block axis last. The same primitive
    is used by dense export, packed-expert export, and renderer-side checks so
    the E8M0 scale and FP4 codebook math cannot drift across call sites.
    """
    grouped_f = grouped.to(torch.float32)
    group_size = grouped_f.shape[-1]
    if group_size % 2 != 0:
        raise ValueError("MXFP4 packing requires an even group size")
    e8m0 = _mx_base_exponent_from_amax(
        grouped_f.abs().amax(dim=-1),
        element_max=MXFP4_E2M1_MAX,
    )
    scale = torch.pow(
        torch.tensor(2.0, device=grouped_f.device, dtype=torch.float32),
        e8m0,
    )
    indices, dequant = _nvfp4_quantize_dequantize_with_eff_scale(
        grouped_f,
        scale.unsqueeze(-1),
    )
    packed = pack_fp4_indices(indices, group_size)
    e8m0_uint8 = (e8m0 + 127).to(torch.int32).clamp(0, 255).to(torch.uint8)
    return _MXFP4CodecResult(
        indices=indices,
        packed=packed,
        scale=e8m0_uint8,
        dequant=dequant,
    )


def _mxfp4_quantize_grouped(grouped: torch.Tensor
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute MXFP4 packed E2M1 values + E8M0 scale for grouped weights.

    `grouped` must have the per-group axis in the final dimension
    (size 32 for the OCP MXFP4 layout). The returned packed tensor has
    two FP4 codes per byte along that final axis, and the scale tensor is
    uint8 E8M0 with the final group axis removed.
    """
    codec = _mxfp4_grouped_codec(grouped)
    return codec.packed, codec.scale


def quantize_dequantize_mxfp4(weight: torch.Tensor, group_size: int = 32
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MXFP4 RTN with E8M0 per-group scale to a 2D weight.

    On-disk schema (compressed-tensors `mxfp4-pack-quantized` format):
      - weight_packed: uint8, shape (rows, cols // 2)
      - weight_scale:  uint8 E8M0, shape (rows, cols // group_size)
    """
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"MXFP4 group_size={group_size} ∤ {cols}")
    grouped = weight.float().reshape(rows, cols // group_size, group_size)
    packed, e8m0_uint8 = _mxfp4_quantize_grouped(grouped)
    return packed.reshape(rows, cols // 2), e8m0_uint8


def _mxfp4_dequantize_2d(
    weight_packed: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor:
    rows = weight_packed.shape[0]
    cols = weight_packed.shape[1] * 2
    if cols % group_size != 0:
        raise ValueError(f"MXFP4 group_size={group_size} ∤ {cols}")
    lo = (weight_packed & 0xF).to(torch.long)
    hi = ((weight_packed >> 4) & 0xF).to(torch.long)
    fp4_idx = torch.stack([lo, hi], dim=-1).reshape(rows, cols)
    scale_by_col = e8m0_to_scale(
        scale,
        device=weight_packed.device,
    ).repeat_interleave(group_size, dim=1)
    return _decode_nvfp4_indices_with_eff_scale(fp4_idx, scale_by_col)


def quantize_dequantize_mxfp4_packed(packed: torch.Tensor, group_size: int = 32
                                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MXFP4 RTN to a 3D packed-experts tensor `[E, M, N]`."""
    E, M, N = packed.shape
    if N % group_size != 0:
        raise ValueError(f"MXFP4 group_size={group_size} ∤ {N}")
    grouped = packed.float().reshape(E, M, N // group_size, group_size)
    qpacked, e8m0_uint8 = _mxfp4_quantize_grouped(grouped)
    return qpacked.reshape(E, M, N // 2), e8m0_uint8


def _mxfp8_quantize_grouped(
    grouped: torch.Tensor,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = MXFP8_E4M3_MAX,
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute MXFP8_E4M3/MXFP8_E5M2 values + E8M0 scale for an arbitrary
    rank-N tensor whose LAST dim is the per-group axis (size group_size).

    Returns:
      - quant_fp8: same shape as `grouped`, dtype torch.float8_*
      - e8m0_uint8: same shape minus the last dim, uint8 (E8M0)

    Scale generation follows compressed-tensors' MX rule: round the block
    amax to the nearest representable power-of-two, subtract the element
    exponent offset, then store the biased E8M0 exponent. The resulting grid
    may place the block amax above the finite FP8 max; `_fp8_codec` clamps
    before casting so overflow cannot become NaN.
    """
    codec = _mxfp8_grouped_codec(
        grouped,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return codec.quant, codec.scale


def quantize_dequantize_mxfp8(
    weight: torch.Tensor,
    group_size: int = 32,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = MXFP8_E4M3_MAX,
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MXFP8_E4M3/MXFP8_E5M2 RTN with E8M0 per-group scale to a 2D weight.

    On-disk schema (compressed-tensors `mxfp8-quantized` format):
      - weight_packed: torch.float8_*, same shape as weight
      - weight_scale:  uint8 E8M0, shape (rows, cols // group_size)
    """
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"MXFP8_E4M3/MXFP8_E5M2 group_size={group_size} ∤ {cols}")
    grouped = weight.float().reshape(rows, cols // group_size, group_size)
    quant_fp8, e8m0_uint8 = _mxfp8_quantize_grouped(
        grouped,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return quant_fp8.reshape(rows, cols), e8m0_uint8


def _mxfp8_dequantize_grouped(
    quant_fp8: torch.Tensor,
    e8m0_uint8: torch.Tensor,
) -> torch.Tensor:
    scale = e8m0_to_scale(e8m0_uint8, device=quant_fp8.device)
    return quant_fp8.to(torch.float32) * scale.unsqueeze(-1)


def _mxfp8_dequantize_2d(
    quant_fp8: torch.Tensor,
    e8m0_uint8: torch.Tensor,
    *,
    group_size: int = 32,
) -> torch.Tensor:
    rows, cols = quant_fp8.shape
    grouped = quant_fp8.reshape(rows, cols // group_size, group_size)
    return _mxfp8_dequantize_grouped(grouped, e8m0_uint8).reshape(rows, cols)


def _mxfp8_scale_sweep_quantize(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    group_size: int = 32,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,

    levers: NativeRenderLevers | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Activation-weighted E8M0 scale search for MXFP8_E4M3.

    MXFP8_E4M3 has enough mantissa precision that GPTQ-style error propagation is
    usually not worth the cost. The scale, however, is exponent-only E8M0;
    max-abs/ceil is conservative and can waste resolution. Searching nearby
    exponents per row/group is cheap, vLLM-compatible, and preserves the exact
    compressed-tensors MXFP8_E4M3 representation.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"MXFP8_E4M3 scale sweep requires group_size={group_size} ∤ {cols}")
    W = weight.to(torch.float32)
    grouped = W.reshape(rows, cols // group_size, group_size)
    shifts = list(levers.mxfp8_scale_sweep_shifts)
    if shifts == [0]:
        codec = _mxfp8_grouped_codec(grouped)
        return (
            codec.quant.reshape(rows, cols),
            codec.scale,
            codec.dequant.reshape(rows, cols),
        )

    col_importance = _activation_col_importance_for_gptq(
        activations,
        cols,
        device=W.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=fisher_row_weights,
     levers=levers).reshape(1, cols // group_size, group_size)

    base_e = _mxfp8_base_exponent(grouped, element_max=MXFP8_E4M3_MAX)
    shift_t = torch.tensor(shifts, device=W.device, dtype=torch.float32)

    best_err: torch.Tensor | None = None
    best_e: torch.Tensor | None = None
    for shift in shift_t:
        e = (base_e + shift).clamp(-127, 127)
        codec = _mxfp8_grouped_codec(
            grouped,
            e8m0_unbiased=e,
        )
        err = ((grouped - codec.dequant).pow(2) * col_importance).sum(dim=-1)
        if best_err is None:
            best_err = err
            best_e = e
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_e = torch.where(take, e, best_e)

    assert best_e is not None
    codec = _mxfp8_grouped_codec(
        grouped,
        e8m0_unbiased=best_e,
    )
    return (
        codec.quant.reshape(rows, cols),
        codec.scale,
        codec.dequant.reshape(rows, cols),
    )


def quantize_dequantize_mxfp8_packed(
    packed: torch.Tensor,
    group_size: int = 32,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = MXFP8_E4M3_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MXFP8_E4M3/MXFP8_E5M2 RTN to a 3D packed-experts tensor `[E, M, N]`.

    Returns:
      - weight_packed: float8 `[E, M, N]`
      - weight_scale:  uint8 E8M0   `[E, M, N//group_size]`
    """
    E, M, N = packed.shape
    if N % group_size != 0:
        raise ValueError(f"MXFP8_E4M3/MXFP8_E5M2 group_size={group_size} ∤ {N}")
    grouped = packed.float().reshape(E, M, N // group_size, group_size)
    codec = _mxfp8_grouped_codec(
        grouped,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return codec.quant.reshape(E, M, N), codec.scale


def quantize_dequantize_fp8_dynamic(
    weight: torch.Tensor,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = FP8_E4M3_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 W8A8 dynamic per-channel weight quantization.

    Matches vLLM's CompressedTensorsW8A8Fp8 expectation:
      - weight: torch.float8_*, shape `[out, in]`
      - weight_scale: torch.float32, shape `[out, 1]` (per-channel)

    Per-channel scale = max-abs(row) / fp8_max. Dynamic-token activation
    quantization is handled at runtime by vLLM (no on-disk activation
    scale needed).
    """
    codec = _fp8_dynamic_codec(
        weight,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return codec.quant, codec.scale


def _dequantize_fp8_dynamic(
    quant: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return _fp8_dequantize(quant, scale)


def _rtn_dequant_fp8_dynamic(
    weight: torch.Tensor,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = FP8_E4M3_MAX,
) -> torch.Tensor:
    q, s = quantize_dequantize_fp8_dynamic(
        weight.to(torch.float32),
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return _dequantize_fp8_dynamic(q, s)


def _rtn_dequant_mxfp8(
    weight: torch.Tensor,
    *,
    group_size: int = 32,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = MXFP8_E4M3_MAX,
) -> torch.Tensor:
    q, s = quantize_dequantize_mxfp8(
        weight.to(torch.float32),
        group_size=group_size,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return _mxfp8_dequantize_2d(q, s, group_size=group_size)





def _mxfp8_joint_scale_shifts(
    *,
    levers: NativeRenderLevers | None = None,
) -> tuple[int, ...]:
    # {-1, 0}: ceil-log2 (the canonical NVIDIA recipe, never saturates) plus
    # ceil-1 (lets the block trade one ULP of saturation on the amax for
    # one bit more precision on the rest). The synthetic-LLM oracle search
    # picks -1 for ~5% of blocks and 0 for the rest; deeper negative shifts
    # over-saturate and never win on unweighted MSE.
    if levers is None:
        levers = NativeRenderLevers.resolve()
    return levers.mxfp8_joint_scale_shifts


def _fp8_quantize_dequantize_with_scale(
    values: torch.Tensor,
    scale: torch.Tensor,
    *,
    element_dtype: torch.dtype,
    element_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    codec = _fp8_codec(
        values,
        scale=scale,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return codec.quant, codec.dequant


def _mxfp8_quantize_dequantize_block(
    block: torch.Tensor,
    *,
    col_importance: torch.Tensor | None,
    joint_scale_opt: bool,
    element_dtype: torch.dtype,
    element_max: float,

    levers: NativeRenderLevers | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return MXFP8_E4M3/MXFP8_E5M2 q/scale/dequant for one `[rows, group_size]` block.

    The JSO path searches legal E8M0 block scales by unweighted block MSE.
    Activation-weighted MSE was tried and produced rankings that disagreed
    with end-task quality; unweighted block MSE aligns with the allocator's
    h_trace * weight_mse cost surrogate. `col_importance` is accepted for
    interface compatibility but no longer consulted.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    del col_importance  # unused after switch to unweighted block-MSE
    base_e = _mxfp8_base_exponent(block, element_max=element_max)
    if joint_scale_opt:
        best_err: torch.Tensor | None = None
        best_e: torch.Tensor | None = None
        for shift in _mxfp8_joint_scale_shifts( levers=levers):
            e = (base_e + float(shift)).clamp(-127, 127)
            codec = _mxfp8_grouped_codec(
                block,
                element_dtype=element_dtype,
                element_max=element_max,
                e8m0_unbiased=e,
            )
            score = (block - codec.dequant).pow(2).sum(dim=-1)
            if best_err is None:
                best_err = score
                best_e = e
                continue
            take = score < best_err
            best_err = torch.where(take, score, best_err)
            assert best_e is not None
            best_e = torch.where(take, e, best_e)
        assert best_e is not None
        e = best_e
    else:
        e = base_e
    codec = _mxfp8_grouped_codec(
        block,
        e8m0_unbiased=e,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return codec.quant, codec.scale, codec.dequant


def _gptq_obs_rounding_mxfp4(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    group_size: int = 32,
    damp: float | None = None,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    static_act_order: bool = False,

    levers: NativeRenderLevers | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ one-shot OBS rounding for MXFP4 weights.

    MXFP4 uses the same E2M1 FP4 codebook as NVFP4, but with an E8M0
    per-32-value block scale and no tensor-global FP8 scale. The returned
    tuple is directly exportable: packed FP4 bytes, E8M0 block scales, and
    the served dequantized weight.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    W = weight.to(torch.float32).clone()
    rows, cols = W.shape
    if cols % group_size != 0:
        raise ValueError(f"MXFP4 GPTQ requires group_size={group_size} ∤ {cols}")

    X = _activation_matrix_for_gptq(
        activations,
        cols,
        device=W.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=fisher_row_weights,
     levers=levers)
    if damp is None:
        damp = _resolve_gptq_fixed_damp( levers=levers)
    H = X.t() @ X
    # Dead-channel handling: detect diag(H) <= 0 BEFORE damping (see
    # _gptq_obs_rounding_nvfp4), exclude dead entries from the damp
    # reference mean, give dead diagonals an identity entry. We do NOT
    # zero the weights of dead columns (deliberate serving-safe
    # deviation from reference GPTQ — a column unexercised by
    # calibration must not be destroyed for serving traffic).
    diag0 = torch.diagonal(H)
    dead = diag0 <= 0
    alive = ~dead
    diag_mean = (
        diag0[alive].mean() if bool(alive.any()) else diag0.new_ones(())
    ).clamp_min(1e-12)
    if dead.any():
        H[dead, dead] = 1.0
    H.diagonal().add_(float(damp) * diag_mean)
    col_importance = torch.diagonal(H).detach().clone().clamp_min(1e-12)

    scale_out = torch.empty(
        (rows, cols // group_size),
        device=W.device,
        dtype=torch.uint8,
    )
    for group_idx, block_start in enumerate(range(0, cols, group_size)):
        block_end = block_start + group_size
        codec = _mxfp4_grouped_codec(W[:, block_start:block_end])
        scale_out[:, group_idx] = codec.scale
    scale_by_col = e8m0_to_scale(
        scale_out,
        device=W.device,
    ).repeat_interleave(group_size, dim=1)

    inverse_perm: torch.Tensor | None = None
    if static_act_order:
        perm = torch.argsort(col_importance, descending=True)
        inverse_perm = torch.empty_like(perm)
        inverse_perm[perm] = torch.arange(cols, device=W.device)
        W = W.index_select(1, perm).contiguous()
        H = H.index_select(0, perm).index_select(1, perm).contiguous()
        scale_by_col = scale_by_col.index_select(1, perm).contiguous()

    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        U = torch.linalg.cholesky(Hinv, upper=True)
    except Exception:
        q, scale = quantize_dequantize_mxfp4(
            weight.to(torch.float32),
            group_size=group_size,
        )
        return q, scale, _mxfp4_dequantize_2d(q, scale, group_size=group_size)

    idx_work = torch.empty((rows, cols), device=W.device, dtype=torch.uint8)

    def _quantize_mxfp4_col(col: torch.Tensor, col_idx: int) -> torch.Tensor:
        scale = scale_by_col[:, col_idx:col_idx + 1].clamp_min(1e-12)
        idx_col, col_dq = _nvfp4_quantize_dequantize_with_eff_scale(
            col.unsqueeze(1),
            scale,
        )
        idx_work[:, col_idx] = idx_col.squeeze(1)
        return col_dq.squeeze(1)

    _gptq_columnwise_update(
        W,
        U,
        block_size=_gptq_column_block_size(cols, levers=levers),
        quantize_column=_quantize_mxfp4_col,
    )

    idx_out = (
        idx_work.index_select(1, inverse_perm).contiguous()
        if inverse_perm is not None else
        idx_work
    )
    q_out = pack_fp4_indices(idx_out, cols)
    dequant = _mxfp4_dequantize_2d(q_out, scale_out, group_size=group_size)
    return q_out.contiguous(), scale_out.contiguous(), dequant.contiguous()


def _gptq_obs_rounding_mxfp4_swept(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    group_size: int = 32,
    damp_candidates: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05, 0.1),
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    static_act_order: bool = False,

    levers: NativeRenderLevers | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if levers is None:
        levers = NativeRenderLevers.resolve()
    W_orig = weight.to(torch.float32)
    X = _activation_matrix_for_gptq(
        activations,
        W_orig.shape[1],
        device=W_orig.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=fisher_row_weights,
     levers=levers)
    H_full = X.t() @ X
    best: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    best_err = float("inf")
    for damp in damp_candidates:
        try:
            candidate = _gptq_obs_rounding_mxfp4(
                W_orig,
                activations,
                group_size=group_size,
                damp=damp,
                clip_threshold=clip_threshold,
                clip_rescale=clip_rescale,
                fisher_row_weights=fisher_row_weights,
                static_act_order=static_act_order,
             levers=levers)
        except Exception:
            continue
        diff = W_orig - candidate[2].to(torch.float32)
        err = float(torch.einsum("oi,ij,oj->", diff, H_full, diff))
        if err < best_err:
            best_err = err
            best = candidate
    if best is not None:
        return best
    return _gptq_obs_rounding_mxfp4(
        W_orig,
        activations,
        group_size=group_size,
        damp=0.01,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        fisher_row_weights=fisher_row_weights,
        static_act_order=static_act_order,
     levers=levers)


def _gptq_obs_rounding_fp8_like(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    fmt: str,
    group_size: int = 32,
    damp: float | None = None,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    joint_scale_opt: bool = False,
    static_act_order: bool = False,

    levers: NativeRenderLevers | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ one-shot OBS rounding for FP8_E4M3/E5M2 and MXFP8_E4M3/E5M2 weights.

    Returns `(quant_weight, scale, dequant_weight)` in the exact representation
    the export path can serialize. Plain FP8 uses one fp32 scale per output
    row; MXFP8_E4M3/MXFP8_E5M2 use one uint8 E8M0 scale per row/group.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    fmt_u = _canonical_export_format(fmt)
    is_mx = fmt_u in MXFP8_EXPLICIT_FORMATS
    is_plain = fmt_u in {"FP8_E4M3", "FP8_E5M2"}
    if not (is_mx or is_plain):
        raise ValueError(f"unsupported FP8 GPTQ format: {fmt}")
    if is_mx:
        # MXFP8 uses the canonical E8M0 block scale. The historical
        # joint_scale_opt hook searched nearby legal exponents, but that
        # is not part of the production MXFP8 recipe.
        joint_scale_opt = False
    else:
        # Plain FP8 has a per-output-row dynamic scale, so static activation
        # ordering is not part of the current production recipe.
        static_act_order = False

    element_dtype, element_max = _fp8_element_dtype_and_max(fmt_u)
    W = weight.to(torch.float32).clone()
    rows, cols = W.shape
    if is_mx and cols % group_size != 0:
        raise ValueError(f"MXFP8_E4M3/MXFP8_E5M2 GPTQ requires group_size={group_size} ∤ {cols}")

    X = _activation_matrix_for_gptq(
        activations,
        cols,
        device=W.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=fisher_row_weights,
     levers=levers)
    if damp is None:
        damp = _resolve_gptq_fixed_damp( levers=levers)
    H = X.t() @ X
    # Dead-channel handling: detect diag(H) <= 0 BEFORE damping (see
    # _gptq_obs_rounding_nvfp4), exclude dead entries from the damp
    # reference mean, give dead diagonals an identity entry. We do NOT
    # zero the weights of dead columns (deliberate serving-safe
    # deviation from reference GPTQ — a column unexercised by
    # calibration must not be destroyed for serving traffic).
    diag0 = torch.diagonal(H)
    dead = diag0 <= 0
    alive = ~dead
    diag_mean = (
        diag0[alive].mean() if bool(alive.any()) else diag0.new_ones(())
    ).clamp_min(1e-12)
    if dead.any():
        H[dead, dead] = 1.0
    H.diagonal().add_(float(damp) * diag_mean)
    col_importance = torch.diagonal(H).detach().clone().clamp_min(1e-12)

    if is_plain:
        scale_out = _fp8_dynamic_codec(
            weight.to(torch.float32),
            element_dtype=element_dtype,
            element_max=element_max,
        ).scale
        scale_by_col = scale_out.expand(rows, cols)
    else:
        scale_out = torch.empty(
            (rows, cols // group_size),
            device=W.device,
            dtype=torch.uint8,
        )
        for group_idx, block_start in enumerate(range(0, cols, group_size)):
            block_end = block_start + group_size
            _q_block, scale_block, _block_dq = _mxfp8_quantize_dequantize_block(
                W[:, block_start:block_end],
                col_importance=col_importance[block_start:block_end],
                joint_scale_opt=joint_scale_opt,
                element_dtype=element_dtype,
                element_max=element_max,
             levers=levers)
            scale_out[:, group_idx] = scale_block
        scale_by_col = e8m0_to_scale(
            scale_out,
            device=W.device,
        ).repeat_interleave(group_size, dim=1)

    inverse_perm: torch.Tensor | None = None
    if static_act_order:
        perm = torch.argsort(col_importance, descending=True)
        inverse_perm = torch.empty_like(perm)
        inverse_perm[perm] = torch.arange(cols, device=W.device)
        W = W.index_select(1, perm).contiguous()
        H = H.index_select(0, perm).index_select(1, perm).contiguous()
        scale_by_col = scale_by_col.index_select(1, perm).contiguous()

    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        U = torch.linalg.cholesky(Hinv, upper=True)
    except Exception:
        if is_mx:
            q, scale = quantize_dequantize_mxfp8(
                weight.to(torch.float32),
                group_size=group_size,
                element_dtype=element_dtype,
                element_max=element_max,
            )
            return q, scale, _mxfp8_dequantize_2d(q, scale, group_size=group_size)
        q, scale = quantize_dequantize_fp8_dynamic(
            weight.to(torch.float32),
            element_dtype=element_dtype,
            element_max=element_max,
        )
        return q, scale, _dequantize_fp8_dynamic(q, scale)

    q_work = torch.empty((rows, cols), device=W.device, dtype=element_dtype)

    def _quantize_fp8_col(col: torch.Tensor, col_idx: int) -> torch.Tensor:
        scale = scale_by_col[:, col_idx:col_idx + 1].clamp_min(2.0 ** -127)
        q_col, col_dq = _fp8_quantize_dequantize_with_scale(
            col.unsqueeze(1),
            scale,
            element_dtype=element_dtype,
            element_max=element_max,
        )
        q_work[:, col_idx] = q_col.squeeze(1)
        return col_dq.squeeze(1)

    W = _gptq_columnwise_update(
        W,
        U,
        block_size=_gptq_column_block_size(cols, levers=levers),
        quantize_column=_quantize_fp8_col,
    )

    q_out = (
        q_work.index_select(1, inverse_perm).contiguous()
        if inverse_perm is not None else
        q_work
    )
    if is_mx:
        dequant = _mxfp8_dequantize_2d(q_out, scale_out, group_size=group_size)
    else:
        dequant = _dequantize_fp8_dynamic(q_out, scale_out)
    return q_out.contiguous(), scale_out.contiguous(), dequant.contiguous()


def _gptq_obs_rounding_fp8_like_swept(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    fmt: str,
    group_size: int = 32,
    damp_candidates: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05, 0.1),
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    joint_scale_opt: bool = False,
    static_act_order: bool = False,

    levers: NativeRenderLevers | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if levers is None:
        levers = NativeRenderLevers.resolve()
    W_orig = weight.to(torch.float32)
    X = _activation_matrix_for_gptq(
        activations,
        W_orig.shape[1],
        device=W_orig.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=fisher_row_weights,
     levers=levers)
    H_full = X.t() @ X
    best: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    best_err = float("inf")
    for damp in damp_candidates:
        try:
            candidate = _gptq_obs_rounding_fp8_like(
                W_orig,
                activations,
                fmt=fmt,
                group_size=group_size,
                damp=damp,
                clip_threshold=clip_threshold,
                clip_rescale=clip_rescale,
                fisher_row_weights=fisher_row_weights,
                joint_scale_opt=joint_scale_opt,
                static_act_order=static_act_order,
             levers=levers)
        except Exception:
            continue
        diff = W_orig - candidate[2].to(torch.float32)
        err = float(torch.einsum("oi,ij,oj->", diff, H_full, diff))
        if err < best_err:
            best_err = err
            best = candidate
    if best is not None:
        return best
    return _gptq_obs_rounding_fp8_like(
        W_orig,
        activations,
        fmt=fmt,
        group_size=group_size,
        damp=0.01,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        fisher_row_weights=fisher_row_weights,
        joint_scale_opt=joint_scale_opt,
        static_act_order=static_act_order,
     levers=levers)


def _fp8_scale_sweep_factors(
    *,
    levers: NativeRenderLevers | None = None,
) -> tuple[float, ...]:
    if levers is None:
        levers = NativeRenderLevers.resolve()
    return levers.fp8_scale_sweep_factors


def _fp8_dynamic_scale_sweep_quantize(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,

    levers: NativeRenderLevers | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Activation-weighted per-row scale search for vLLM FP8 E4M3."""
    if levers is None:
        levers = NativeRenderLevers.resolve()
    if weight.dim() != 2:
        raise ValueError("FP8 scale sweep expects a 2D Linear weight")
    rows, cols = weight.shape
    w_f = weight.detach().to(torch.float32)
    if activations.shape[-1] != cols:
        codec = _fp8_dynamic_codec(w_f)
        return codec.quant, codec.scale, codec.dequant
    col_importance = _activation_col_importance_for_gptq(
        activations,
        cols,
        device=w_f.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=fisher_row_weights,
     levers=levers).to(device=w_f.device, dtype=torch.float32)
    base = (
        w_f.abs().amax(dim=-1, keepdim=True).clamp_min(2.0 ** -127)
        / FP8_E4M3_MAX
    )
    best_score = torch.full((rows,), float("inf"), device=w_f.device)
    best_dequant = torch.empty_like(w_f)
    best_scale = torch.empty((rows, 1), device=w_f.device, dtype=torch.float32)
    for factor in _fp8_scale_sweep_factors( levers=levers):
        scale = base * float(factor)
        codec = _fp8_codec(
            w_f,
            scale=scale,
            element_dtype=torch.float8_e4m3fn,
            element_max=FP8_E4M3_MAX,
        )
        score = (
            (w_f - codec.dequant).pow(2) * col_importance.unsqueeze(0)
        ).sum(dim=1)
        take = score < best_score
        if bool(take.any().item()):
            best_score = torch.where(take, score, best_score)
            best_dequant = torch.where(
                take.unsqueeze(1),
                codec.dequant,
                best_dequant,
            )
            best_scale = torch.where(
                take.unsqueeze(1),
                codec.scale,
                best_scale,
            )
    best = _fp8_codec(
        best_dequant,
        scale=best_scale,
        element_dtype=torch.float8_e4m3fn,
        element_max=FP8_E4M3_MAX,
    )
    return (
        best.quant.contiguous(),
        best.scale.contiguous(),
        best.dequant.contiguous(),
    )


def quantize_dequantize_fp8_dynamic_packed(
    packed: torch.Tensor,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = FP8_E4M3_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert FP8 W8A8 dynamic per-channel for `[E, M, N]` packed.

    Returns weight `[E, M, N]` fp8 and scale `[E, M, 1]` fp32.
    """
    codec = _fp8_dynamic_codec(
        packed,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return codec.quant, codec.scale


def _quantize_2d(
    weight: torch.Tensor, fmt: str,
    nvfp4_global_real_override: torch.Tensor | None = None,
    input_global_scale_override: float | None = None,
    act_clip_threshold: float | None = None,
    act_clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    linear_name: str | None = None,
    gptq_enabled: bool | None = None,
    scale_sweep_enabled: bool | None = None,
    static_act_order_enabled: bool | None = None,
    joint_scale_opt_enabled: bool | None = None,
    cached_activations: torch.Tensor | None = None,
    compute_only: bool = False,

    *,
    levers: NativeRenderLevers | None = None,
) -> dict[str, torch.Tensor]:
    """Compress a 2D Linear weight under format `fmt`.

    Returns the dict of on-disk tensors keyed by the suffix
    (`weight_packed`, `weight_scale`, `weight_global_scale`, ...).

    `nvfp4_global_real_override`: when this Linear is one shard of a
    fused parameter (q/k/v/o, gate/up), pass the joint per-tensor
    scale shared across all siblings. vLLM warns when sibling scales
    differ and reports degraded accuracy; sharing avoids both.

    `input_global_scale_override`: explicit per-Linear activation scale.  If
    absent, the shared resolver checks the calibrated mapping and finally the
    legacy `DEFAULT_INPUT_GLOBAL_SCALE` (1.0).  That fallback preserves native
    artifact semantics but does not qualify the versioned fused contract.

    `gptq_enabled` and `scale_sweep_enabled` compose activation-aware passes
    on NVFP4, MXFP4, FP8_E4M3/FP8_E5M2, and MXFP8_E4M3/MXFP8_E5M2 paths. Each
    requires explicit `cached_activations` (aligned by the caller with
    `linear_name` when not supplied explicitly). For MXFP8_E4M3/MXFP8_E5M2,
    `joint_scale_opt_enabled` searches legal E8M0 block scales during GPTQ
    instead of reusing NVFP4's max-to-4/max-to-6 heuristic.

    `cached_activations`: optional `[N, in_features]` float tensor of
    probe-captured inputs for this Linear. If None and `linear_name`
    is set, the caller supplies that Linear's cached rows.

    `act_clip_threshold`: optional scalar clamp for the render-time
    activation-aware NVFP4/MXFP4/MXFP8_E4M3/MXFP8_E5M2 passes.  When None, legacy behavior is
    preserved: GPTQ/do-no-harm honor PRISMAQUANT_ACT_CLIP_QUANTILE,
    while scale_sweep uses raw cached activations.

    `fisher_row_weights`: optional per-token gradient² weights aligned to
    cached activation rows. When provided, GPTQ/scale-sweep local objectives
    become output/Fisher-weighted by scaling activation rows by sqrt(weight).

    `fmt = MXFP8_E4M3` and `fmt = MXFP8_E5M2` emit fp8 weights plus E8M0
    uint8 per-group scales (group_size=32). A bare `MXFP8` input is accepted
    only as a legacy alias for `MXFP8_E4M3`.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve({
            "gptq": bool(gptq_enabled), "scale_sweep": bool(scale_sweep_enabled),
            "static_act_order": bool(static_act_order_enabled),
            "joint_scale_opt": bool(joint_scale_opt_enabled)})
    fmt = _canonical_export_format(fmt)

    acts = cached_activations

    # Device fix: cached activations are stored on CPU (float32) to
    # amortize load cost across many quant calls; weights land on the
    # export device (typically CUDA). Move activations to the weight's
    # device here so every downstream op (GPTQ H matrix,
    # act-weighted rounding) runs on a consistent device. Repairs
    # `Expected all tensors to be on the same device, but found at
    # least two devices, cuda:0 and cpu!` in live Qwen3.6-35B export.
    if acts is not None and acts.device != weight.device:
        acts = acts.to(weight.device, non_blocking=True)

    gptq_enabled = levers.gptq if gptq_enabled is None else bool(gptq_enabled)
    scale_sweep_enabled = levers.scale_sweep if scale_sweep_enabled is None else bool(scale_sweep_enabled)
    static_act_order_enabled = levers.static_act_order if static_act_order_enabled is None else bool(static_act_order_enabled)
    joint_scale_opt_enabled = levers.joint_scale_opt if joint_scale_opt_enabled is None else bool(joint_scale_opt_enabled)
    static_act_order_enabled = bool(gptq_enabled and static_act_order_enabled)
    joint_scale_opt_enabled = bool(gptq_enabled and joint_scale_opt_enabled)

    if fmt == "NVFP4":
        w_work = weight.to(torch.float32)

        def _acts_for_error_passes() -> torch.Tensor | None:
            """Return cached activations aligned to this Linear."""
            if acts is None or acts.shape[-1] != w_work.shape[1]:
                return None
            return acts

        # Step 2: GPTQ one-shot OBS rounding (block-wise error prop).
        # Produces an already-dequantized tensor living on the NVFP4
        # grid; subsequent packing is lossless wrt this tensor.
        if gptq_enabled:
            acts_work = _acts_for_error_passes()
            if acts_work is not None:
                # Env-gated per-Linear damping sweep (#46). When set,
                # try multiple λ values for the Hessian regularizer and
                # pick the one with smallest output-space error. ~5×
                # GPTQ wallclock.
                # Default OFF since 2026-06-12 (see
                # gptq_damp_sweep_enabled(), :1845-1857): its evaluator
                # is in-sample, so its "winners" invert on held-out
                # basins (31/31) and it lost the V1 served A/B to a
                # fixed damp. Production uses the fixed damp from
                # _resolve_gptq_fixed_damp() (1.0, :1860).
                # PRISMAQUANT_GPTQ_DAMP_SWEEP=1 reproduces historical
                # sweep-rendered artifacts.
                if gptq_damp_sweep_enabled( levers=levers):
                    w_work = _gptq_obs_rounding_nvfp4_swept(
                        w_work, acts_work, group_size=16,
                        global_real_override=nvfp4_global_real_override,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=act_clip_rescale,
                        fisher_row_weights=fisher_row_weights,
                        static_act_order=static_act_order_enabled,
                        joint_scale_opt=joint_scale_opt_enabled,
                        linear_name=linear_name,
                     levers=levers)
                else:
                    w_work = _gptq_obs_rounding_nvfp4(
                        w_work, acts_work, group_size=16,
                        global_real_override=nvfp4_global_real_override,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=act_clip_rescale,
                        fisher_row_weights=fisher_row_weights,
                        static_act_order=static_act_order_enabled,
                        joint_scale_opt=joint_scale_opt_enabled,
                     levers=levers)

        # Step 3: closed-form per-group scale sweep. Joint (scale,
        # rounding-set) search on the NVFP4 codebook, activation-
        # weighted MSE against the ORIGINAL pre-pass weight, with an
        # improve-or-keep gate against the current w_work. Recovers
        # most of AutoRound's benefit without its 200-iter SGD.
        if scale_sweep_enabled:
            acts_work = _acts_for_error_passes()
            if acts_work is not None:
                w_work = _scale_sweep_nvfp4(
                    w_work, acts_work, group_size=16,
                    global_real_override=nvfp4_global_real_override,
                    reference_weight=weight.to(torch.float32),
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                 levers=levers)

        # Do-no-harm gate (codex review #3): if GPTQ ran and we have
        # cached activations, compute the activation-weighted
        # reconstruction MSE for both the post-pass weight (`w_work`)
        # and a pure-RTN baseline against the original. If RTN is
        # better, revert. Catches per-Linear cases where GPTQ + sweep
        # locally degraded reconstruction. Env-gated; default on
        # because the cost is one RTN dequant + two MSE sums (cheap).
        if (gptq_enabled and acts is not None
                and levers.do_no_harm):
            try:
                w_orig_f = weight.to(torch.float32)
                w_rtn = _rtn_dequant_nvfp4(
                    w_orig_f, group_size=16,
                    global_real_override=nvfp4_global_real_override,
                 levers=levers)
                a2 = _activation_col_importance_for_gptq(
                    acts,
                    w_orig_f.shape[1],
                    device=w_orig_f.device,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    row_weights=fisher_row_weights,
                 levers=levers)
                mse_rtn = float((a2 * (w_orig_f - w_rtn).pow(2)
                                 .sum(dim=0)).sum())
                mse_work = float((a2 * (w_orig_f - w_work).pow(2)
                                  .sum(dim=0)).sum())
                if mse_rtn < mse_work:
                    _record_do_no_harm_revert("NVFP4")
                    if os.environ.get(
                        "PRISMAQUANT_DO_NO_HARM_VERBOSE") == "1":
                        print(f"[do-no-harm] {linear_name}: "
                              f"reverted to RTN "
                              f"(mse {mse_work:.3e} → {mse_rtn:.3e})",
                              flush=True)
                    w_work = w_rtn
            except Exception as _e:
                _record_do_no_harm_failure("NVFP4", linear_name, _e)

        # Step 4: final NVFP4 pack. `w_work` is the post-GPTQ,
        # post-act-round, post-scale-sweep weight.
        input_scale = _resolve_nvfp4_input_global_scale(
            input_global_scale_override,
            target=linear_name,
        )

        # compute_only path: return the dequantized weight WITHOUT freezing
        # it into FP4 codes. Its only production consumer was block-output
        # match, walled 2026-07-30 (archive/block_output_match_2026-07-30/,
        # re-vet R25) along with the `_finalize_compute_only` packer that
        # closed the loop. Kept as a lever-threading introspection hook —
        # nothing in the export path sets it.
        if compute_only:
            return {
                "_compute_only": True,
                "_fmt": "NVFP4",
                "_w_dq": w_work,
                "_nvfp4_global_real": nvfp4_global_real_override,
                "_input_scale": float(input_scale),
            }

        wp, ws, wg = quantize_dequantize_nvfp4(
            w_work, group_size=16,
            global_real_override=nvfp4_global_real_override,
         levers=levers)
        return {
            "weight_packed": wp,
            "weight_scale": ws,
            "weight_global_scale": wg,
            # Required by vLLM's CompressedTensorsW4A4Nvfp4 process; see
            # compressed_tensors_w4a4_nvfp4.py:115. Without it vLLM
            # initializes input_global_scale to zeros and computes
            # 1/zero on activation quant → degenerate output.
            "input_global_scale": torch.tensor(
                [float(input_scale)], dtype=torch.float32,
            ),
        }
    if fmt in MXFP8_EXPLICIT_FORMATS:
        # MXFP8_E4M3/MXFP8_E5M2 GPTQ is export-faithful: it returns the FP8
        # codes and E8M0 scale tensor directly. MXFP8 deliberately does not
        # consume joint_scale_opt; it uses the canonical E8M0 scale rule.
        w_work = weight.to(torch.float32)
        acts_work = acts
        element_dtype, element_max = _fp8_element_dtype_and_max(fmt)
        has_acts = (
            acts_work is not None and acts_work.shape[-1] == w_work.shape[1]
        )

        def _mxfp8_rtn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q_rtn, s_rtn = quantize_dequantize_mxfp8(
                w_work,
                group_size=32,
                element_dtype=element_dtype,
                element_max=element_max,
            )
            return q_rtn, s_rtn, _mxfp8_dequantize_2d(q_rtn, s_rtn, group_size=32)

        if gptq_enabled and has_acts:
            assert acts_work is not None

            def _mxfp8_gptq_candidate(
                use_static_act_order: bool,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if gptq_damp_sweep_enabled( levers=levers):
                    return _gptq_obs_rounding_fp8_like_swept(
                        w_work,
                        acts_work,
                        fmt=fmt,
                        group_size=32,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=act_clip_rescale,
                        fisher_row_weights=fisher_row_weights,
                        joint_scale_opt=False,
                        static_act_order=use_static_act_order,
                     levers=levers)
                return _gptq_obs_rounding_fp8_like(
                    w_work,
                    acts_work,
                    fmt=fmt,
                    group_size=32,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                    joint_scale_opt=False,
                    static_act_order=use_static_act_order,
                 levers=levers)

            candidates = [_mxfp8_gptq_candidate(False)]
            if static_act_order_enabled:
                candidates.append(_mxfp8_gptq_candidate(True))
            w, ws, dq = min(
                candidates,
                key=lambda cand: _activation_weighted_weight_error(
                    w_work,
                    cand[2],
                    acts_work,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    row_weights=fisher_row_weights,
                 levers=levers),
            )
            if scale_sweep_enabled and fmt == "MXFP8_E4M3":
                w, ws, dq = _mxfp8_scale_sweep_quantize(
                    dq,
                    acts_work,
                    group_size=32,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                 levers=levers)
            if levers.do_no_harm:
                try:
                    q_rtn, s_rtn, dq_rtn = _mxfp8_rtn()
                    err_rtn = _activation_weighted_weight_error(
                        w_work,
                        dq_rtn,
                        acts_work,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=act_clip_rescale,
                        row_weights=fisher_row_weights,
                     levers=levers)
                    err_work = _activation_weighted_weight_error(
                        w_work,
                        dq,
                        acts_work,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=act_clip_rescale,
                        row_weights=fisher_row_weights,
                     levers=levers)
                    if err_rtn < err_work:
                        _record_do_no_harm_revert(fmt)
                        if os.environ.get(
                            "PRISMAQUANT_DO_NO_HARM_VERBOSE") == "1":
                            print(f"[do-no-harm] {linear_name}: "
                                  f"reverted MXFP8 to RTN "
                                  f"(mse {err_work:.3e} → {err_rtn:.3e})",
                                  flush=True)
                        w, ws, dq = q_rtn, s_rtn, dq_rtn
                except Exception as _e:
                    _record_do_no_harm_failure(fmt, linear_name, _e)
        elif scale_sweep_enabled and has_acts and fmt == "MXFP8_E4M3":
            assert acts_work is not None
            w, ws, _ = _mxfp8_scale_sweep_quantize(
                w_work,
                acts_work,
                group_size=32,
                clip_threshold=act_clip_threshold,
                clip_rescale=act_clip_rescale,
                fisher_row_weights=fisher_row_weights,
             levers=levers)
        else:
            w, ws, _ = _mxfp8_rtn()
        return {"weight": w, "weight_scale": ws}
    if fmt in {"FP8_E4M3", "FP8_E5M2"}:
        w_work = weight.to(torch.float32)
        acts_work = acts
        if (gptq_enabled and acts_work is not None
                and acts_work.shape[-1] == w_work.shape[1]):
            if gptq_damp_sweep_enabled( levers=levers):
                w, ws, dq = _gptq_obs_rounding_fp8_like_swept(
                    w_work,
                    acts_work,
                    fmt=fmt,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                    joint_scale_opt=False,
                 levers=levers)
            else:
                w, ws, dq = _gptq_obs_rounding_fp8_like(
                    w_work,
                    acts_work,
                    fmt=fmt,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                    joint_scale_opt=False,
                 levers=levers)
            if scale_sweep_enabled and fmt == "FP8_E4M3":
                w, ws, _ = _fp8_dynamic_scale_sweep_quantize(
                    dq,
                    acts_work,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                 levers=levers)
        elif (scale_sweep_enabled and acts_work is not None
              and acts_work.shape[-1] == w_work.shape[1]
              and fmt == "FP8_E4M3"):
            w, ws, _ = _fp8_dynamic_scale_sweep_quantize(
                w_work,
                acts_work,
                clip_threshold=act_clip_threshold,
                clip_rescale=act_clip_rescale,
                fisher_row_weights=fisher_row_weights,
             levers=levers)
        else:
            if fmt == "FP8_E5M2":
                raise ValueError("FP8_E5M2 export packing is research-only")
            w, ws = quantize_dequantize_fp8_dynamic(w_work)
        return {"weight": w, "weight_scale": ws}
    if fmt == "MXFP4":
        w_work = weight.to(torch.float32)
        acts_work = acts
        has_acts = (
            acts_work is not None and acts_work.shape[-1] == w_work.shape[1]
        )

        def _mxfp4_rtn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q_rtn, s_rtn = quantize_dequantize_mxfp4(w_work, group_size=32)
            return q_rtn, s_rtn, _mxfp4_dequantize_2d(q_rtn, s_rtn, group_size=32)

        if gptq_enabled and has_acts:
            assert acts_work is not None

            def _mxfp4_gptq_candidate(
                use_static_act_order: bool,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if gptq_damp_sweep_enabled( levers=levers):
                    return _gptq_obs_rounding_mxfp4_swept(
                        w_work,
                        acts_work,
                        group_size=32,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=act_clip_rescale,
                        fisher_row_weights=fisher_row_weights,
                        static_act_order=use_static_act_order,
                     levers=levers)
                return _gptq_obs_rounding_mxfp4(
                    w_work,
                    acts_work,
                    group_size=32,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                    static_act_order=use_static_act_order,
                 levers=levers)

            candidates = [_mxfp4_gptq_candidate(False)]
            if static_act_order_enabled:
                candidates.append(_mxfp4_gptq_candidate(True))
            wp, ws, dq = min(
                candidates,
                key=lambda cand: _activation_weighted_weight_error(
                    w_work,
                    cand[2],
                    acts_work,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    row_weights=fisher_row_weights,
                 levers=levers),
            )
            if levers.do_no_harm:
                try:
                    q_rtn, s_rtn, dq_rtn = _mxfp4_rtn()
                    err_rtn = _activation_weighted_weight_error(
                        w_work,
                        dq_rtn,
                        acts_work,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=act_clip_rescale,
                        row_weights=fisher_row_weights,
                     levers=levers)
                    err_work = _activation_weighted_weight_error(
                        w_work,
                        dq,
                        acts_work,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=act_clip_rescale,
                        row_weights=fisher_row_weights,
                     levers=levers)
                    if err_rtn < err_work:
                        _record_do_no_harm_revert("MXFP4")
                        if os.environ.get(
                            "PRISMAQUANT_DO_NO_HARM_VERBOSE") == "1":
                            print(f"[do-no-harm] {linear_name}: "
                                  f"reverted MXFP4 to RTN "
                                  f"(mse {err_work:.3e} → {err_rtn:.3e})",
                                  flush=True)
                        wp, ws, dq = q_rtn, s_rtn, dq_rtn
                except Exception as _e:
                    _record_do_no_harm_failure("MXFP4", linear_name, _e)
        else:
            wp, ws, _ = _mxfp4_rtn()
        return {"weight_packed": wp, "weight_scale": ws}
    if fmt == "BF16":
        return {"weight": weight.to(torch.bfloat16)}
    raise ValueError(f"unsupported format: {fmt}")


def _quantize_3d_packed(packed: torch.Tensor, fmt: str,
    *,
    levers: NativeRenderLevers | None = None,
) -> dict[str, torch.Tensor]:
    """Compress a 3D packed-expert tensor `[E, M, N]` as a single
    batched op (per-expert independent scales).

    Returns tensors with leading expert dim preserved, matching what
    vLLM's `compressed_tensors_moe_w4a4_nvfp4` allocates internally
    (uint8 packed weights, fp8/uint8 per-group scales, per-expert
    global scales for NVFP4).
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    fmt = _canonical_export_format(fmt)
    if fmt == "BF16":
        return {"weight": packed.to(torch.bfloat16)}
    if fmt == "NVFP4":
        wp, ws, wg = quantize_dequantize_nvfp4_packed(packed, group_size=16, levers=levers)
        return {
            "weight_packed": wp,
            "weight_scale": ws,
            "weight_global_scale": wg,
        }
    if fmt in MXFP8_EXPLICIT_FORMATS:
        element_dtype, element_max = _fp8_element_dtype_and_max(fmt)
        w, ws = quantize_dequantize_mxfp8_packed(
            packed,
            group_size=32,
            element_dtype=element_dtype,
            element_max=element_max,
        )
        return {"weight": w, "weight_scale": ws}
    if fmt in {"FP8_E4M3", "FP8_E5M2"}:
        element_dtype, element_max = _fp8_element_dtype_and_max(fmt)
        w, ws = quantize_dequantize_fp8_dynamic_packed(
            packed.to(torch.float32),
            element_dtype=element_dtype,
            element_max=element_max,
        )
        return {"weight": w, "weight_scale": ws}
    if fmt == "MXFP4":
        wp, ws = quantize_dequantize_mxfp4_packed(packed, group_size=32)
        return {"weight_packed": wp, "weight_scale": ws}
    raise ValueError(f"unsupported format for packed-MoE: {fmt}")


def _quantize_2d_group_same_shape(
    stacked_weights: torch.Tensor,
    fmt: str,
) -> dict[str, torch.Tensor]:
    """Compress a batch of same-shape 2D weights in one vectorized op.

    `stacked_weights` is `[B, out, in]`. Returned tensors keep the leading
    batch dimension so the caller can split them back to per-Linear keys.
    This is deliberately limited to RTN-only formats: activation-aware NVFP4
    remains scalar until its GPTQ/scale-sweep passes are vectorized too.
    """
    fmt = _canonical_export_format(fmt)
    if stacked_weights.dim() != 3:
        raise ValueError(
            "same-shape export grouping expects [B, out, in] weights; "
            f"got shape={tuple(stacked_weights.shape)}"
        )
    if fmt in MXFP8_EXPLICIT_FORMATS:
        element_dtype, element_max = _fp8_element_dtype_and_max(fmt)
        w, ws = quantize_dequantize_mxfp8_packed(
            stacked_weights.to(torch.float32),
            group_size=32,
            element_dtype=element_dtype,
            element_max=element_max,
        )
        return {"weight": w, "weight_scale": ws}
    raise ValueError(f"unsupported grouped 2D export format: {fmt}")


def _quantize_2d_nvfp4_group_batched(
    items: list,
    joint_globals: dict,
    device: torch.device,
    expert_chunk: int = 32,

    *,
    levers: NativeRenderLevers | None = None,
    activations: object | None = None,
    input_global_scales: Mapping[str, float] | None = None,
) -> list[dict]:
    """Batched NVFP4 quantization for a same-shape group of Linears
    when activation-aware passes (GPTQ / scale_sweep) are enabled.

    Replaces the per-Linear `_quantize_2d` flow's slow steps (GPTQ +
    scale_sweep) with the batched analogs in
    `prismaquant.export_batched_gptq`. The fast steps (final NVFP4
    pack, input-global-scale lookup) stay per-Linear since they are
    already cheap.

    Items: list of `(full, emit_full, recipe_key, mod)` tuples. All
    `mod.weight` must share `(out, in)` shape. The function returns a
    list of compressed dicts in the same order, ready to be merged
    into the export's `out` dict by the caller.

    The reference weight passed to scale_sweep is the same original weight
    used by the per-Linear path's `weight.to(float32)` argument.
    """
    if levers is None:
        levers = NativeRenderLevers.resolve()
    from .export_batched_gptq import (
        gptq_obs_rounding_nvfp4_batched,
        scale_sweep_nvfp4_batched,
    )

    n = len(items)
    if n == 0:
        return []

    # Stack weights into [E, out, in]. All shapes must match.
    weights = torch.stack(
        [it[3].weight.detach().to(torch.float32) for it in items], dim=0,
    ).to(device)
    reference_weights = weights.clone()  # pre-pass reference for scale_sweep

    # Per-Linear activation tensors (None where missing).
    acts_list: list = []
    for full, emit_full, recipe_key, mod in items:
        a = None
        if activations is not None:
            raw = activations.get(recipe_key)
            if raw is not None and raw.shape[-1] == mod.weight.shape[1]:
                a = raw.to(torch.float32).reshape(-1, raw.shape[-1])
        acts_list.append(a if a is not None else torch.zeros(
            0, weights.shape[2], dtype=torch.float32, device=device))

    # Per-Linear NVFP4 global_real overrides (from joint fused-sibling).
    # When recipe_key isn't in joint_globals, the batched path computes
    # per-Linear from the weights — pass `None` for that Linear. We
    # represent the override array as a [E] tensor with NaN for "no
    # override"; the batched function expects a single tensor of shape
    # [E], so we must split into "all overridden" or "none overridden"
    # groups within this function or fall back to per-Linear when mixed.
    overrides_list = [joint_globals.get(it[2]) for it in items]
    if all(v is not None for v in overrides_list):
        global_real_overrides = torch.stack(
            [v.to(device, dtype=torch.float32) for v in overrides_list]
        ).reshape(n)
    elif all(v is None for v in overrides_list):
        global_real_overrides = None
    else:
        # Mixed — split into homogeneous sub-groups and recurse.
        with_idx = [i for i, v in enumerate(overrides_list) if v is not None]
        without_idx = [i for i, v in enumerate(overrides_list) if v is None]
        results: list[dict] = [None] * n  # type: ignore[list-item]
        if with_idx:
            sub = [items[i] for i in with_idx]
            sub_results = _quantize_2d_nvfp4_group_batched(
                sub, joint_globals, device, expert_chunk=expert_chunk,
                activations=activations, input_global_scales=input_global_scales,
                levers=levers)
            for i, r in zip(with_idx, sub_results):
                results[i] = r
        if without_idx:
            sub = [items[i] for i in without_idx]
            sub_results = _quantize_2d_nvfp4_group_batched(
                sub, joint_globals, device, expert_chunk=expert_chunk,
                activations=activations, input_global_scales=input_global_scales,
                levers=levers)
            for i, r in zip(without_idx, sub_results):
                results[i] = r
        return results

    # Run the batched activation-aware passes. Match the per-Linear
    # `_quantize_2d` ordering: GPTQ → scale_sweep.
    # Codex review #46 batched extension: per-Linear damping sweep.
    # Run GPTQ at each candidate damp, measure activation-weighted
    # output MSE per Linear, keep the best per Linear. Cost is
    # n_candidates × the unswept GPTQ pass; gated by env so prod
    # default keeps the existing single-damp speed.
    if levers.gptq:
        # Default ON (validated on Qwen3-0.6B audit). =0 to disable.
        damp_sweep_on = (
            gptq_damp_sweep_enabled( levers=levers))
        if damp_sweep_on:
            damp_candidates = (0.001, 0.005, 0.01, 0.05, 0.1)
            best_w = None
            best_err = None  # [E] of activation-weighted MSE
            # Pre-compute per-Linear column importance for the gate.
            col_imp = torch.empty(
                (n, weights.shape[2]), device=device, dtype=torch.float32)
            for j, a in enumerate(acts_list):
                if a is None or a.numel() == 0:
                    col_imp[j] = 1.0
                else:
                    col_imp[j] = _activation_col_importance_for_gptq(
                        a, weights.shape[2], device=device, levers=levers)
            for damp in damp_candidates:
                cand_w = gptq_obs_rounding_nvfp4_batched(
                    weights, acts_list,
                    damp=damp,
                    global_real_overrides=global_real_overrides,
                    expert_chunk=expert_chunk,
                    static_act_order=bool(
                        levers.static_act_order
                    ),
                    joint_scale_opt=bool(
                        levers.joint_scale_opt
                    ),
                 levers=levers)
                # Per-Linear activation-weighted MSE vs reference.
                diff = reference_weights - cand_w
                err = (col_imp.unsqueeze(1) * diff.pow(2)).sum(dim=(1, 2))
                if best_w is None:
                    best_w = cand_w
                    best_err = err
                else:
                    take = err < best_err
                    if take.any():
                        idx = take.nonzero(as_tuple=True)[0]
                        best_w[idx] = cand_w[idx]
                        best_err[idx] = err[idx]
            weights = best_w
        else:
            weights = gptq_obs_rounding_nvfp4_batched(
                weights, acts_list,
                global_real_overrides=global_real_overrides,
                expert_chunk=expert_chunk,
                static_act_order=bool(
                    levers.static_act_order
                ),
                joint_scale_opt=bool(
                    levers.joint_scale_opt
                ),
             levers=levers)
    if levers.scale_sweep:
        weights = scale_sweep_nvfp4_batched(
            weights, acts_list,
            reference_weights=reference_weights,
            global_real_overrides=global_real_overrides,
            expert_chunk=expert_chunk,
         levers=levers)

    # Codex review #47 batched extension: per-Linear do-no-harm gate.
    # If the post-pass weight is worse on activation-weighted MSE than
    # a pure RTN of the original, swap that single Linear back to RTN.
    # Same default-on as the per-Linear path; PRISMAQUANT_DO_NO_HARM=0
    # disables. Cost: one RTN dequant + two MSE sums per Linear.
    if (levers.gptq
            and levers.do_no_harm):
        try:
            # Per-Linear activation column importance.
            col_imp = torch.empty(
                (n, weights.shape[2]), device=device, dtype=torch.float32)
            n_acts_avail = 0
            for j, a in enumerate(acts_list):
                if a is None or a.numel() == 0:
                    col_imp[j] = 1.0
                else:
                    col_imp[j] = _activation_col_importance_for_gptq(
                        a, weights.shape[2], device=device, levers=levers)
                    n_acts_avail += 1
            n_reverted = 0
            for i in range(n):
                if acts_list[i] is None or acts_list[i].numel() == 0:
                    continue  # no activations → can't gate; trust the pass
                override = overrides_list[i]
                w_rtn = _rtn_dequant_nvfp4(
                    reference_weights[i], group_size=16,
                    global_real_override=override,
                 levers=levers)
                ref_i = reference_weights[i]
                imp = col_imp[i]
                mse_pass = float(
                    (imp * (ref_i - weights[i]).pow(2).sum(dim=0)).sum())
                mse_rtn = float(
                    (imp * (ref_i - w_rtn).pow(2).sum(dim=0)).sum())
                if mse_rtn < mse_pass:
                    weights[i] = w_rtn
                    n_reverted += 1
            if n_reverted:
                _DO_NO_HARM_STATS["NVFP4_batched_reverts"] += int(n_reverted)
            if n_reverted and os.environ.get(
                    "PRISMAQUANT_DO_NO_HARM_VERBOSE") == "1":
                print(f"[do-no-harm batched] reverted {n_reverted}/{n} "
                      f"Linears to RTN", flush=True)
        except Exception as _e:
            _record_do_no_harm_failure("NVFP4_batched", None, _e)

    # Per-Linear final NVFP4 pack (cheap; reuses the existing function).
    out: list[dict] = []
    for i, (full, emit_full, recipe_key, mod) in enumerate(items):
        override = overrides_list[i]
        wp, ws, wg = quantize_dequantize_nvfp4(
            weights[i], group_size=16,
            global_real_override=override,
         levers=levers)
        input_scale = _resolve_nvfp4_input_global_scale(target=recipe_key, calibrated_scales=input_global_scales)
        out.append({
            "weight_packed": wp,
            "weight_scale": ws,
            "weight_global_scale": wg.reshape(1)
            if wg.dim() == 0 else wg,
            "input_global_scale": torch.tensor(
                [float(input_scale)], dtype=torch.float32),
        })
    return out

