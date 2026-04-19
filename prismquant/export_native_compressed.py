#!/usr/bin/env python3
"""export_native_compressed.py — materialize a PrismQuant recipe as a
standard `compressed-tensors` checkpoint that vLLM serves natively.

Reads the per-tensor format assignment produced by `allocator.py`
(layer_config.json) and emits a directory containing:

  - `model-*.safetensors` (sharded), with each Linear / packed-MoE
    tensor written under the standard compressed-tensors schema:
        <name>.weight_packed         (uint8, 4-bit packed for NVFP4)
        <name>.weight_scale          (fp8_e4m3fn for NVFP4 / e8m0 for MXFP8)
        <name>.weight_global_scale   (fp32, NVFP4 only)
        <name>.input_global_scale    (fp32, A4/A8 formats only)
    OR `<name>.weight` (passthrough bf16) for layers in the BF16 bucket.

  - `model.safetensors.index.json` matching the safetensors layout

  - `config.json` carrying a `quantization_config` with
    `format = mixed-precision` and one config_group per nominated
    format. Targets are explicit per-Linear regex anchors so vLLM's
    compressed-tensors dispatcher routes every parameter to the right
    scheme without ambiguity.

  - `mixed_native_manifest.json` summarizing the export (format
    histogram, ignore list, source recipe path) for traceability.

  - tokenizer / config files copied verbatim from the source.

Why this exists separately from llmcompressor's oneshot:
  - llmcompressor's QuantizationModifier matches nn.Linear modules. It
    does not handle 3D packed-expert tensors (Qwen3.5/3.6's
    `gate_up_proj` / `down_proj`), which silently fall back to dense
    bf16 in the standard pipeline.
  - llmcompressor pins transformers <5; transformers v5 is required to
    load Qwen3.6 (`qwen3_5_moe`). The two cannot coexist.

This exporter pins to transformers v5 for model load, uses the
compressed-tensors lib's `pack_fp4_to_uint8` reference (inlined to
avoid the lib's transformers-coupled `__init__`), and writes the
on-disk layout directly. vLLM's existing `compressed_tensors` and
`compressed_tensors_moe_w4a4_nvfp4` schemes load the result without
patches.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import shutil
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# NVFP4 packing (inlined from compressed-tensors fp4_quantized.py to avoid
# importing the library's __init__ which pulls in transformers internals
# that are not stable across the 4.x → 5.x break).
# ---------------------------------------------------------------------------
FLOAT_TO_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
NVFP4_MAX = 6.0     # max(|FLOAT_TO_E2M1|)
FP8_E4M3_MAX = 448.0  # max representable in torch.float8_e4m3fn


def _nvfp4_codebook(device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(FLOAT_TO_E2M1, device=device, dtype=dtype)


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


def pack_fp4_indices(fp4_indices: torch.Tensor, last_dim: int) -> torch.Tensor:
    """Pack a tensor of 4-bit indices (final dim must be even) into
    uint8, two indices per byte. Preserves leading dimensions.
    """
    if last_dim % 2 != 0:
        raise ValueError("nvfp4 pack requires an even last dim")
    pairs = fp4_indices.reshape(*fp4_indices.shape[:-1], last_dim // 2, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)


DEFAULT_INPUT_GLOBAL_SCALE = 1.0  # placeholder; overridden by calibration


def compute_nvfp4_global_real(weight: torch.Tensor, group_size: int = 16
                              ) -> torch.Tensor:
    """Return the per-tensor `global_real` that NVFP4 packing would
    pick for `weight` alone. Useful for fused-sibling pre-pass: caller
    takes the max across siblings and passes the joint value back into
    `quantize_dequantize_nvfp4(global_real_override=...)`."""
    rows, cols = weight.shape
    grouped = weight.float().reshape(rows, cols // group_size, group_size)
    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)
    s_g_real = max_abs / NVFP4_MAX
    return (s_g_real.amax() / FP8_E4M3_MAX).clamp_min(1e-12)


def quantize_dequantize_nvfp4(
    weight: torch.Tensor, group_size: int = 16,
    global_real_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply NVFP4 RTN to a 2D `[rows, cols]` weight and return the
    on-disk triple `(weight_packed, weight_scale, weight_global_scale)`
    in the **compressed-tensors NVFP4 convention**:

      - per-group dequant scale  s_g_real = max-abs(group) / NVFP4_MAX
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
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"NVFP4 group_size={group_size} ∤ {cols}")
    n_groups = cols // group_size
    grouped = weight.float().reshape(rows, n_groups, group_size)
    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)               # [rows, n_groups]
    s_g_real = max_abs / NVFP4_MAX                                       # the actual per-group scale
    if global_real_override is not None:
        global_real = global_real_override.to(weight.device).clamp_min(1e-12)
    else:
        global_real = (s_g_real.amax() / FP8_E4M3_MAX).clamp_min(1e-12)  # scalar
    fp8_scale_real = (s_g_real / global_real).clamp(0, FP8_E4M3_MAX)     # [rows, n_groups], in [0, 448]
    # Per-element grid mapping: weight / (fp8_scale_real * global_real) = weight / s_g_real
    in_grid = grouped / s_g_real.unsqueeze(-1).clamp_min(1e-12)          # [rows, n_groups, group_size]
    fp4_idx = _round_to_codebook(in_grid).reshape(rows, cols)
    weight_packed = pack_fp4_indices(fp4_idx, cols)
    return (
        weight_packed,
        fp8_scale_real.to(torch.float8_e4m3fn),
        (1.0 / global_real).to(torch.float32).reshape(1),  # divisor convention
    )


def quantize_dequantize_nvfp4_packed(
    packed: torch.Tensor, group_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-expert NVFP4 packing for a 3D `[E, M, N]` packed tensor.
    Each expert gets its own `global_real` (so the weight_global_scale
    output has shape `[E]`); the on-disk values are divisors (1/scale)
    matching the compressed-tensors convention.
    """
    E, M, N = packed.shape
    if N % group_size != 0:
        raise ValueError(f"NVFP4 group_size={group_size} ∤ {N}")
    g = N // group_size
    grouped = packed.float().reshape(E, M, g, group_size)
    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)
    s_g_real = max_abs / NVFP4_MAX                                          # [E, M, g]
    global_real = (s_g_real.reshape(E, -1).amax(dim=-1) / FP8_E4M3_MAX).clamp_min(1e-12)  # [E]
    fp8_scale_real = (s_g_real / global_real.view(E, 1, 1)).clamp(0, FP8_E4M3_MAX)
    in_grid = grouped / s_g_real.unsqueeze(-1).clamp_min(1e-12)
    fp4_idx = _round_to_codebook(in_grid).reshape(E, M, N)
    weight_packed = pack_fp4_indices(fp4_idx, N)
    return (
        weight_packed,
        fp8_scale_real.to(torch.float8_e4m3fn),
        (1.0 / global_real).to(torch.float32),
    )


# ---------------------------------------------------------------------------
# MXFP8 packing (E4M3 element format, E8M0 per-group scale).
# ---------------------------------------------------------------------------
MXFP8_E4M3_MAX = 448.0   # max representable in fp8_e4m3fn


def _mxfp8_quantize_grouped(grouped: torch.Tensor
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute MXFP8 quantized values + E8M0 scale for an arbitrary
    rank-N tensor whose LAST dim is the per-group axis (size group_size).

    Returns:
      - quant_fp8: same shape as `grouped`, dtype torch.float8_e4m3fn
      - e8m0_uint8: same shape minus the last dim, uint8 (E8M0)

    Care: with E8M0 round-to-nearest the per-group scale can be
    slightly smaller than max-abs/MXFP8_E4M3_MAX, which would push
    quant_grid past 448 (fp8_e4m3fn max) and produce NaN on cast.
    We use ceil() on log2 to guarantee s_g >= max-abs/MXFP8_E4M3_MAX,
    keeping all quant_grid values inside the representable range.
    """
    s_g_real = grouped.abs().amax(dim=-1).clamp_min(2.0 ** -127) / MXFP8_E4M3_MAX
    log2_s = torch.log2(s_g_real)
    e8m0 = torch.ceil(log2_s).clamp(-127, 127)
    s_g = torch.pow(2.0, e8m0)
    quant_grid = grouped / s_g.unsqueeze(-1).clamp_min(2.0 ** -127)
    # Defensive clamp against numerical edge cases at the saturation boundary.
    quant_grid = quant_grid.clamp(-MXFP8_E4M3_MAX, MXFP8_E4M3_MAX)
    quant_fp8 = quant_grid.to(torch.float8_e4m3fn)
    e8m0_uint8 = (e8m0 + 127).to(torch.int32).clamp(0, 255).to(torch.uint8)
    return quant_fp8, e8m0_uint8


def quantize_dequantize_mxfp8(weight: torch.Tensor, group_size: int = 32
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MXFP8 (E4M3) RTN with E8M0 per-group scale to a 2D weight.

    On-disk schema (compressed-tensors `mxfp8-quantized` format):
      - weight_packed: torch.float8_e4m3fn, same shape as weight
      - weight_scale:  uint8 E8M0, shape (rows, cols // group_size)
    """
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"MXFP8 group_size={group_size} ∤ {cols}")
    grouped = weight.float().reshape(rows, cols // group_size, group_size)
    quant_fp8, e8m0_uint8 = _mxfp8_quantize_grouped(grouped)
    return quant_fp8.reshape(rows, cols), e8m0_uint8


def quantize_dequantize_mxfp8_packed(packed: torch.Tensor, group_size: int = 32
                                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MXFP8 RTN to a 3D packed-experts tensor `[E, M, N]`.

    Returns:
      - weight_packed: float8_e4m3fn `[E, M, N]`
      - weight_scale:  uint8 E8M0   `[E, M, N//group_size]`
    """
    E, M, N = packed.shape
    if N % group_size != 0:
        raise ValueError(f"MXFP8 group_size={group_size} ∤ {N}")
    grouped = packed.float().reshape(E, M, N // group_size, group_size)
    quant_fp8, e8m0_uint8 = _mxfp8_quantize_grouped(grouped)
    return quant_fp8.reshape(E, M, N), e8m0_uint8


def quantize_dequantize_fp8_dynamic(weight: torch.Tensor
                                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 W8A8 dynamic per-channel weight quantization.

    Matches vLLM's CompressedTensorsW8A8Fp8 expectation:
      - weight: torch.float8_e4m3fn, shape `[out, in]`
      - weight_scale: torch.float32, shape `[out, 1]` (per-channel)

    Per-channel scale = max-abs(row) / fp8_max. Dynamic-token activation
    quantization is handled at runtime by vLLM (no on-disk activation
    scale needed).
    """
    rows, cols = weight.shape
    w_f = weight.float()
    s = w_f.abs().amax(dim=-1, keepdim=True).clamp_min(2.0 ** -127) / MXFP8_E4M3_MAX
    quant = (w_f / s).clamp(-MXFP8_E4M3_MAX, MXFP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return quant, s.to(torch.float32)


def quantize_dequantize_fp8_dynamic_packed(packed: torch.Tensor
                                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert FP8 W8A8 dynamic per-channel for `[E, M, N]` packed.

    Returns weight `[E, M, N]` fp8 and scale `[E, M, 1]` fp32.
    """
    E, M, N = packed.shape
    p_f = packed.float()
    s = p_f.abs().amax(dim=-1, keepdim=True).clamp_min(2.0 ** -127) / MXFP8_E4M3_MAX
    quant = (p_f / s).clamp(-MXFP8_E4M3_MAX, MXFP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return quant, s.to(torch.float32)


# ---------------------------------------------------------------------------
# Recipe parsing — mirrors export_mixed_native.canonicalize_format but
# accepts the allocator's exact AutoRound-shaped output.
# ---------------------------------------------------------------------------
def canonicalize_format(scheme_dict: dict | str | int) -> str:
    """Map a layer_config entry to one of {NVFP4, MXFP8, BF16}.

    Accepts the dicts emitted by allocator.py via FormatSpec.autoround_config()
    (data_type=nv_fp/mx_fp/float, bits=4/8/16) plus a few tolerant
    string aliases.
    """
    if isinstance(scheme_dict, dict):
        dt = scheme_dict.get("data_type")
        bits = int(scheme_dict.get("bits", 0))
        if dt == "nv_fp" and bits == 4:
            return "NVFP4"
        if dt == "mx_fp" and bits == 4:
            return "NVFP4"  # 4-bit floor — only NVFP4 has vLLM serving today
        if dt == "mx_fp" and bits == 8:
            return "MXFP8"
        if dt in ("float", "bfloat16") and bits in (16, 0):
            return "BF16"
        if dt == "fp8_e4m3" and bits == 8:
            return "MXFP8"  # collapse plain FP8 onto the MX bucket for now
        raise ValueError(f"unsupported scheme: {scheme_dict!r}")
    if isinstance(scheme_dict, str):
        s = scheme_dict.lower()
        if s in ("nvfp4", "fp4", "4"):
            return "NVFP4"
        if s in ("mxfp8", "fp8", "8"):
            return "MXFP8"
        if s in ("bf16", "bfloat16", "16"):
            return "BF16"
    if isinstance(scheme_dict, int):
        if scheme_dict <= 4:
            return "NVFP4"
        if scheme_dict <= 8:
            return "MXFP8"
        return "BF16"
    raise ValueError(f"unrecognized layer-config entry: {scheme_dict!r}")


def _strip_weight(name: str) -> str:
    return name[:-7] if name.endswith(".weight") else name


def _explicit_regex(name: str) -> str:
    """Anchor a Linear name as a compressed-tensors regex target."""
    return f"re:^{name.replace('.', '[.]')}$"


# ---------------------------------------------------------------------------
# Module / parameter discovery — mirrors what install_packed_expert_hooks
# detects, so the export sees the same units as the probe.
# ---------------------------------------------------------------------------
_PACKED_EXPERT_PARAM_NAMES = {
    "gate_up_proj", "down_proj", "w1", "w2", "w3",
    "gate_proj", "up_proj",
}


def _is_packed_experts_module(module: nn.Module) -> bool:
    cls_name = type(module).__name__.lower()
    if "expert" not in cls_name:
        return False
    for n, p in module.named_parameters(recurse=False):
        if (isinstance(p, nn.Parameter)
                and p.dim() == 3
                and n in _PACKED_EXPERT_PARAM_NAMES):
            return True
    return False


def _packed_experts_param_names(module: nn.Module) -> list[str]:
    return sorted(
        n for n, p in module.named_parameters(recurse=False)
        if (isinstance(p, nn.Parameter)
            and p.dim() == 3
            and n in _PACKED_EXPERT_PARAM_NAMES)
    )


# ---------------------------------------------------------------------------
# Fused-sibling joint global_scale (for dense Linears)
# ---------------------------------------------------------------------------
# vLLM's compressed_tensors_w4a4_nvfp4.process_weights_after_loading warns
# (and reduces accuracy) when q/k/v or gate/up have different
# weight_global_scale. We compute the max over each fused group's natural
# global_scale and force every sibling to use it.
#
# Patterns mirror vLLM's `packed_modules_mapping` for qwen3_5; if a new
# model family is added, mirror its packed_modules_mapping here.
_FUSED_DENSE_PATTERNS = [
    (re.compile(r"^(?P<pre>.+)\.self_attn\.(?P<sib>q_proj|k_proj|v_proj)$"),
     ("q_proj", "k_proj", "v_proj")),
    (re.compile(r"^(?P<pre>.+)\.mlp\.(?P<sib>gate_proj|up_proj)$"),
     ("gate_proj", "up_proj")),
    (re.compile(r"^(?P<pre>.+)\.mlp\.shared_expert\.(?P<sib>gate_proj|up_proj)$"),
     ("gate_proj", "up_proj")),
    (re.compile(r"^(?P<pre>.+)\.linear_attn\.(?P<sib>in_proj_qkv|in_proj_z)$"),
     ("in_proj_qkv", "in_proj_z")),
    (re.compile(r"^(?P<pre>.+)\.linear_attn\.(?P<sib>in_proj_a|in_proj_b)$"),
     ("in_proj_a", "in_proj_b")),
]


def _fused_dense_group(name: str) -> tuple[str, tuple[str, ...]] | None:
    """Return (group_key, sibling_member_names) if `name` is part of a
    known fused dense Linear group; else None. group_key is the parent
    prefix used to bucket siblings together."""
    for pat, members in _FUSED_DENSE_PATTERNS:
        m = pat.match(name)
        if m:
            return (m.group("pre"), members)
    return None


def _compute_nvfp4_joint_global(
    model: nn.Module, assignment: dict[str, str],
) -> dict[str, torch.Tensor]:
    """Pre-pass over the model: for each fused-sibling group whose
    members are all assigned to NVFP4, compute the joint global_real
    (max across siblings). Return a dict mapping each sibling's qname
    to the shared global_real tensor."""
    # Bucket siblings by (parent_prefix, kind). Missing siblings are
    # OK — vLLM's loader handles partial fusion fine.
    groups: dict[tuple[str, tuple[str, ...]], list[tuple[str, nn.Linear]]] = {}
    for qname, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if assignment.get(qname) != "NVFP4":
            continue
        g = _fused_dense_group(qname)
        if g is None:
            continue
        groups.setdefault(g, []).append((qname, mod))

    out: dict[str, torch.Tensor] = {}
    for (_pre, _members), siblings in groups.items():
        # Need every sibling to also be NVFP4 — otherwise vLLM allocates
        # the fused tensor under a different scheme and our joint scale
        # wouldn't apply consistently. The allocator's promote_fused
        # already enforces this; here we just verify and skip on partial
        # consistency (defensive — a mixed-format fused group is a bug
        # upstream of the export and would fail the load anyway).
        candidates = [
            compute_nvfp4_global_real(mod.weight.detach().float())
            for _, mod in siblings
        ]
        joint = torch.stack(candidates).max()
        for qname, _ in siblings:
            out[qname] = joint
    return out


# ---------------------------------------------------------------------------
# Quantization pipeline
# ---------------------------------------------------------------------------
def _quantize_2d(
    weight: torch.Tensor, fmt: str,
    nvfp4_global_real_override: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compress a 2D Linear weight under format `fmt`.

    Returns the dict of on-disk tensors keyed by the suffix
    (`weight_packed`, `weight_scale`, `weight_global_scale`, ...).

    `nvfp4_global_real_override`: when this Linear is one shard of a
    fused parameter (q/k/v/o, gate/up), pass the joint per-tensor
    scale shared across all siblings. vLLM warns when sibling scales
    differ and reports degraded accuracy; sharing avoids both.

    `fmt = MXFP8` is routed to vLLM's CompressedTensorsW8A8Fp8 path
    (per-channel FP8 + dynamic-token activations) — the closest 8-bit
    format that the version of compressed-tensors vLLM is pinned
    against actually serves natively.
    """
    if fmt == "NVFP4":
        wp, ws, wg = quantize_dequantize_nvfp4(
            weight, group_size=16,
            global_real_override=nvfp4_global_real_override,
        )
        return {
            "weight_packed": wp,
            "weight_scale": ws,
            "weight_global_scale": wg,
            # Required by vLLM's CompressedTensorsW4A4Nvfp4 process; see
            # compressed_tensors_w4a4_nvfp4.py:115. Without it vLLM
            # initializes input_global_scale to zeros and computes
            # 1/zero on activation quant → degenerate output. We supply
            # a sane default; calibrated values can be merged in later.
            "input_global_scale": torch.tensor(
                [DEFAULT_INPUT_GLOBAL_SCALE], dtype=torch.float32,
            ),
        }
    if fmt == "MXFP8":
        # FP8 W8A8 dynamic per-channel — naming `weight` (not `weight_packed`)
        # because vLLM's CompressedTensorsW8A8Fp8 looks for the standard
        # `weight` tensor in fp8 dtype rather than a packed-uint8 form.
        w, ws = quantize_dequantize_fp8_dynamic(weight)
        return {"weight": w, "weight_scale": ws}
    if fmt == "BF16":
        return {"weight": weight.to(torch.bfloat16)}
    raise ValueError(f"unsupported format: {fmt}")


def _quantize_3d_packed(packed: torch.Tensor, fmt: str) -> dict[str, torch.Tensor]:
    """Compress a 3D packed-expert tensor `[E, M, N]` as a single
    batched op (per-expert independent scales).

    Returns tensors with leading expert dim preserved, matching what
    vLLM's `compressed_tensors_moe_w4a4_nvfp4` allocates internally
    (uint8 packed weights, fp8/uint8 per-group scales, per-expert
    global scales for NVFP4).
    """
    if fmt == "BF16":
        return {"weight": packed.to(torch.bfloat16)}
    if fmt == "NVFP4":
        wp, ws, wg = quantize_dequantize_nvfp4_packed(packed, group_size=16)
        return {
            "weight_packed": wp,
            "weight_scale": ws,
            "weight_global_scale": wg,
        }
    if fmt == "MXFP8":
        # Per-expert FP8 W8A8 dynamic per-channel for packed MoE.
        w, ws = quantize_dequantize_fp8_dynamic_packed(packed)
        return {"weight": w, "weight_scale": ws}
    raise ValueError(f"unsupported format for packed-MoE: {fmt}")


def materialize_tensors(
    model: nn.Module,
    assignment: dict[str, str],
    *,
    bf16_passthrough: set[str],
) -> tuple[dict[str, torch.Tensor], dict]:
    """Walk the model and produce the dict of on-disk tensors plus a
    histogram of (kind, format) counts.

    `assignment` keys are stripped of any trailing `.weight`. They
    identify either:
      - a Linear module's qualified name (-> Linear.weight quantized)
      - a packed-experts parameter qualified name
        (e.g. `model.layers.0.mlp.experts.gate_up_proj`)

    Anything not in `assignment` is written verbatim as a passthrough
    tensor (norms, embeddings, lm_head, biases, conv1d weights, etc.).
    """
    out: dict[str, torch.Tensor] = {}
    hist = Counter()
    covered: set[str] = set()

    # Pre-pass: compute joint NVFP4 global_scale per fused-sibling group
    # so q/k/v (or gate/up, etc.) share one weight_global_scale slot.
    # vLLM warns + degrades accuracy when sibling scales disagree.
    nvfp4_joint_global = _compute_nvfp4_joint_global(model, assignment)

    # 1. Linear modules
    for qname, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        fmt_key = qname  # assignment uses module qname (no .weight)
        fmt = assignment.get(fmt_key)
        if fmt is None:
            continue
        if fmt == "BF16" or qname in bf16_passthrough:
            out[f"{qname}.weight"] = mod.weight.detach().to(torch.bfloat16).cpu()
            if mod.bias is not None:
                out[f"{qname}.bias"] = mod.bias.detach().to(torch.bfloat16).cpu()
            covered.add(qname)
            hist[("linear", "BF16")] += 1
            continue
        joint = nvfp4_joint_global.get(qname) if fmt == "NVFP4" else None
        compressed = _quantize_2d(
            mod.weight.detach().float(), fmt,
            nvfp4_global_real_override=joint,
        )
        for suffix, tensor in compressed.items():
            out[f"{qname}.{suffix}"] = tensor.cpu()
        if mod.bias is not None:
            out[f"{qname}.bias"] = mod.bias.detach().to(torch.bfloat16).cpu()
        covered.add(qname)
        hist[("linear", fmt)] += 1

    # 2. Packed-expert parameters — emit per-expert per-projection
    # tensors so vLLM's qwen3_5 loader can match each via its standard
    # `(gate_up_proj, gate_proj, 0)` and `(gate_up_proj, up_proj, 1)`
    # stacked_params_mapping path. The packed `gate_up_proj` is split
    # along its row dim (gate first, up second) into two separate
    # `gate_proj` / `up_proj` per-expert tensors.
    for qname, mod in model.named_modules():
        if not _is_packed_experts_module(mod):
            continue
        for pn in _packed_experts_param_names(mod):
            full_name = f"{qname}.{pn}" if qname else pn
            fmt = assignment.get(full_name)
            if fmt is None:
                continue
            packed_param = getattr(mod, pn).detach().float()  # [E, M, N]
            E, M, N = packed_param.shape
            if pn == "gate_up_proj":
                # Split rows: gate = packed[..., 0:M//2, :], up = [..., M//2:M, :]
                half = M // 2
                proj_split = [
                    ("gate_proj", packed_param[:, :half, :]),
                    ("up_proj",   packed_param[:, half:, :]),
                ]
            elif pn in ("down_proj", "w1", "w2", "w3", "gate_proj", "up_proj"):
                proj_split = [(pn, packed_param)]
            else:
                proj_split = [(pn, packed_param)]

            if fmt == "BF16" or full_name in bf16_passthrough:
                # BF16 passthrough preserves the original packed tensor
                # name + shape (vLLM handles fused tensors fine in bf16).
                out[full_name] = packed_param.to(torch.bfloat16).cpu()
                covered.add(full_name)
                hist[("packed_moe", "BF16")] += 1
                continue

            # When `gate_up_proj` was split into gate+up, the two
            # per-expert siblings need to share their global_scale so
            # vLLM's `w13_weight_global_scale[expert, w13_num_shards=2]`
            # holds two consistent values per expert. Pre-compute the
            # max across siblings per-expert, then pass into both calls.
            per_expert_joint: list[torch.Tensor | None] = [None] * E
            if fmt == "NVFP4" and len(proj_split) > 1:
                for e in range(E):
                    candidates = [
                        compute_nvfp4_global_real(sub_packed[e].float(),
                                                  group_size=16)
                        for _, sub_packed in proj_split
                    ]
                    per_expert_joint[e] = torch.stack(candidates).max()

            for proj_name, sub_packed in proj_split:
                # sub_packed shape [E, M_proj, N]
                E_p, Mp, Np = sub_packed.shape
                for e in range(E_p):
                    expert_2d = sub_packed[e]  # [Mp, N]
                    compressed = _quantize_2d(
                        expert_2d, fmt,
                        nvfp4_global_real_override=per_expert_joint[e],
                    )
                    base = f"{qname}.{e}.{proj_name}"
                    for suffix, tensor in compressed.items():
                        key = base if suffix == "weight" else f"{base}.{suffix}"
                        out[key] = tensor.cpu()
            covered.add(full_name)
            hist[("packed_moe_per_expert", fmt)] += 1

    # 3. Passthrough — everything else (norms, embeddings, biases on
    # non-quantized modules, conv1d, lm_head if not in assignment).
    for name, p in model.named_parameters():
        if any(name.startswith(c + ".") or name == c for c in covered):
            continue
        if name in out:
            continue
        out[name] = p.detach().to(torch.bfloat16).cpu()
        hist[("passthrough", "BF16")] += 1

    return out, dict(hist)


# ---------------------------------------------------------------------------
# Compressed-tensors quantization_config
# ---------------------------------------------------------------------------
NVFP4_SCHEME = {
    "format": "nvfp4-pack-quantized",
    "weights": {
        "num_bits": 4, "type": "float", "strategy": "tensor_group",
        "group_size": 16, "symmetric": True, "dynamic": False,
        "scale_dtype": "torch.float8_e4m3fn",
        "zp_dtype": "torch.float8_e4m3fn",
        "observer": "memoryless_minmax",
    },
    "input_activations": {
        "num_bits": 4, "type": "float", "strategy": "tensor_group",
        "group_size": 16, "symmetric": True,
        "dynamic": "local", "observer": "static_minmax",
        "scale_dtype": "torch.float8_e4m3fn",
        "zp_dtype": "torch.float8_e4m3fn",
    },
}
# FP8 dynamic per-channel — vLLM serves this via CompressedTensorsW8A8Fp8.
# Stand-in for MXFP8 (no compressed-tensors-MXFP8 in the version vLLM is
# pinned against). Same 8-bit bit-budget; per-channel scale instead of
# per-group E8M0; dynamic-token activation instead of dynamic-group.
FP8_DYNAMIC_SCHEME = {
    "format": "float-quantized",
    "weights": {
        "num_bits": 8, "type": "float", "strategy": "channel",
        "symmetric": True, "dynamic": False,
        "observer": "memoryless_minmax",
    },
    "input_activations": {
        "num_bits": 8, "type": "float", "strategy": "token",
        "symmetric": True, "dynamic": True,
    },
}
FORMAT_SCHEME = {
    "NVFP4": NVFP4_SCHEME,
    # Phase 1: route MXFP8 recipe entries through FP8_DYNAMIC. The
    # version of compressed-tensors that vLLM is pinned against does
    # not ship an MXFP8 dispatcher, so we fall back to the closest
    # vLLM-supported 8-bit format. Recipes that picked MXFP8 will
    # serve under the W8A8Fp8 path; quality is comparable for this
    # bucket.
    "MXFP8": FP8_DYNAMIC_SCHEME,
}


# Per-expert MoE pattern matching every per-expert per-projection
# Linear vLLM constructs from compressed MoE checkpoints. This single
# regex covers ~30k tensors that would otherwise need explicit entries.
# Constrained to `language_model.model.layers.X.mlp.experts.Y.{gate,up,down}_proj`
# so it doesn't accidentally match dense Linears or visual encoder
# blocks.
PER_EXPERT_MOE_REGEX = (
    r"re:^language_model[.]model[.]layers[.][0-9]+"
    r"[.]mlp[.]experts[.][0-9]+[.](gate|up|down)_proj$"
)

# Per-expert regex for MTP — vLLM constructs MTP experts as
# `mtp.layers.X.mlp.experts.Y.{gate,up,down}_proj` (the prefix is `mtp`
# because Qwen3_5MTP passes prefix="mtp" to its inner predictor, and
# that prefix is NOT remapped at scheme-dispatch time).
MTP_PER_EXPERT_REGEX = (
    r"re:^mtp[.]layers[.][0-9]+"
    r"[.]mlp[.]experts[.][0-9]+[.](gate|up|down)_proj$"
)


def _to_vllm_internal_name(name: str) -> str:
    """Apply vLLM's name mappers so module names line up with what
    `find_matched_target` compares against at scheme-dispatch time.

    Body (qwen3_5 multimodal class, qwen3_5.py):
      `model.visual.`        → `visual.`
      `lm_head.`             → `language_model.lm_head.`
      `model.language_model.`→ `language_model.model.`

    MTP: vLLM's `Qwen3_5MTP` passes `prefix="mtp"` when building the inner
    `Qwen3_5MultiTokenPredictor`, so every MTP Linear is constructed with
    a `mtp.*` prefix (e.g. `mtp.fc`, `mtp.layers.0.self_attn.q_proj`).
    That prefix is what `find_matched_target` compares against. The
    `mtp.→model.` string replacement in `qwen3_5_mtp.load_weights` is a
    *checkpoint-key* remap applied BEFORE weight loading, AFTER scheme
    dispatch — so it does NOT affect target matching. Leave MTP names
    as-is.

    We accept recipe keys in any of these upstream conventions:
      - body text-only probe form  `model.layers.X.*`
      - body HF multimodal source  `model.language_model.layers.X.*`
      - MTP probe / source form    `mtp.layers.X.*` (preserved)
      - visual source              `model.visual.blocks.X.*`
      - `lm_head`
    """
    # Body — multimodal source form.
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model."):]
    # Visual encoder.
    if name.startswith("model.visual."):
        return "visual." + name[len("model.visual."):]
    # MTP — preserve the `mtp.` prefix. See docstring.
    if name.startswith("mtp."):
        return name
    # Body — text-only recipe form.
    if (name.startswith("model.layers.")
            or name.startswith("model.embed_tokens")
            or name.startswith("model.norm")
            or name == "model"):
        return "language_model.model." + name[len("model."):]
    # lm_head — qwen3_5 multimodal class maps under language_model.
    if name == "lm_head" or name.startswith("lm_head."):
        return "language_model." + name
    return name


def build_quantization_config(
    assignment: dict[str, str],
    bf16_passthrough: set[str],
    extra_ignore: Iterable[str] = (),
) -> dict:
    """Emit a `quantization_config` dict with explicit per-name targets
    grouped by format. Targets and ignore are remapped to vLLM's
    internal naming so `find_matched_target` matches.

    `extra_ignore` is for module qnames that aren't in the recipe at
    all but should be excluded from any catch-all group (e.g. routers).
    The catch-all default group is the format with the most non-BF16
    members (typically NVFP4).
    """
    by_fmt: dict[str, list[str]] = {}
    ignore: list[str] = []
    for n in bf16_passthrough:
        ignore.append(_to_vllm_internal_name(n))
    for n in extra_ignore:
        ignore.append(_to_vllm_internal_name(n))
    for name, fmt in sorted(assignment.items()):
        vllm_name = _to_vllm_internal_name(name)
        if fmt == "BF16":
            ignore.append(vllm_name)
            continue
        by_fmt.setdefault(fmt, []).append(vllm_name)

    if not by_fmt:
        return {}

    sizes = {k: len(v) for k, v in by_fmt.items()}
    catchall = max(sizes, key=sizes.get) if sizes else None
    config_groups = {}
    idx = 0
    for fmt, names in by_fmt.items():
        if fmt == catchall:
            continue
        scheme = deepcopy(FORMAT_SCHEME[fmt])
        scheme["targets"] = [_explicit_regex(n) for n in sorted(names)]
        config_groups[f"group_{idx}"] = scheme
        idx += 1
    if catchall is not None:
        scheme = deepcopy(FORMAT_SCHEME[catchall])
        # Explicit per-name targets, NOT a class-name catch-all
        # ("Linear"). The class-name catch-all matches via a substring
        # check against module class (e.g. MergedColumnParallelLinear)
        # and short-circuits vLLM's fused-layer regex resolution, which
        # is needed to route the explicit per-component MXFP8 targets
        # to vLLM's fused parameter (in_proj_qkvz, qkv_proj, etc.).
        # We additionally add a single pattern regex for per-expert
        # MoE Linears so we don't have to enumerate ~30k entries.
        explicit = sorted(by_fmt[catchall])
        scheme["targets"] = (
            [_explicit_regex(n) for n in explicit]
            + [PER_EXPERT_MOE_REGEX, MTP_PER_EXPERT_REGEX]
        )
        config_groups[f"group_{idx}"] = scheme

    return {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": config_groups,
        "ignore": sorted(set(ignore)),
        "quantization_status": "compressed",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model dir (must be loadable by transformers v5)")
    ap.add_argument("--layer-config", required=True,
                    help="layer_config.json from allocator.py")
    ap.add_argument("--output", required=True,
                    help="Output directory for the compressed checkpoint")
    ap.add_argument("--shard-bytes", type=int, default=5 * 1024 ** 3,
                    help="Approximate per-shard size for safetensors split "
                         "(default: 5 GiB)")
    ap.add_argument("--device", default="cpu",
                    help="Device for quantization arithmetic. cpu is safest "
                         "for streaming a 35B model; cuda for speed.")
    ap.add_argument("--ignore", nargs="*", default=["lm_head"],
                    help="Module qnames to keep at bf16 even if assigned "
                         "elsewhere. Defaults to lm_head.")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] loading recipe from {args.layer_config}", flush=True)
    raw_recipe = json.load(open(args.layer_config))

    assignment: dict[str, str] = {}
    for raw_name, raw_value in raw_recipe.items():
        name = _strip_weight(raw_name)
        try:
            fmt = canonicalize_format(raw_value)
        except ValueError as e:
            print(f"[export] skip {name!r}: {e}", flush=True)
            continue
        assignment[name] = fmt
    print(f"[export] recipe: {len(assignment)} entries  "
          f"format mix: {dict(Counter(assignment.values()).most_common())}",
          flush=True)

    # The probe-side recipe keys use the text-only naming
    # `model.layers.X.*`. We load via AutoModelForCausalLM (text-only),
    # so live module names are also `model.layers.X.*` — no rewriting
    # needed for the recipe → live-module match.
    #
    # On the OUTPUT side, the on-disk safetensors must use the HF
    # multimodal convention `model.language_model.layers.X.*` because
    # vLLM's Qwen3_5MoeForConditionalGeneration loader was written for
    # that source naming (it then maps to `language_model.model.X.*`
    # internally). The output_name_remap dict applied at write time
    # adds the `language_model.` infix.

    bf16_passthrough = set(args.ignore)

    # Load the FULL model (no text-only staging). For multimodal
    # checkpoints (Qwen3.6 = Qwen3VLMoe class) vLLM expects parameter
    # names with the multimodal prefixes intact
    # (`model.language_model.layers.X.*`, `visual.blocks.X.*`, etc.).
    # Stripping those during staging produces a checkpoint vLLM can't
    # locate parameters in. The visual encoder + MTP heads we don't
    # quantize travel through as bf16 passthrough.
    from transformers import AutoModelForImageTextToText
    print(f"[export] loading model from {args.model}", flush=True)
    t0 = time.time()
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except (ValueError, KeyError):
        print("[export] AutoModelForCausalLM declined; using "
              "AutoModelForImageTextToText", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[export] model loaded in {time.time() - t0:.1f}s", flush=True)

    print("[export] materializing compressed tensors ...", flush=True)
    t0 = time.time()
    tensors, hist = materialize_tensors(
        model, assignment, bf16_passthrough=bf16_passthrough,
    )
    print(f"[export] materialized {len(tensors)} tensors in "
          f"{time.time() - t0:.1f}s", flush=True)
    print(f"[export] hist: {hist}", flush=True)

    # Free model — we now hold all weights in `tensors`.
    del model
    gc.collect()

    # Add `model.language_model.` prefix to language-model parameters
    # so the output matches the HF multimodal naming convention vLLM
    # expects. Top-level entries like `lm_head` are left alone.
    out_tensors: dict[str, torch.Tensor] = {}
    for k, v in tensors.items():
        if k.startswith("model.layers.") or k.startswith("model.embed_tokens") \
                or k.startswith("model.norm"):
            out_tensors["model.language_model." + k[len("model."):]] = v
        else:
            out_tensors[k] = v
    tensors = out_tensors

    # Quantize MTP heads per allocator decisions. Transformers drops
    # `mtp.*` on load (`_keys_to_ignore_on_load_unexpected`), so the main
    # materialize_tensors pass never sees them. We rebuild a standalone
    # MTP module, load source weights into it, and run the same quantize
    # pass for any `mtp.*` entries in the recipe.
    print("[export] materializing MTP tensors per allocator ...", flush=True)
    mtp_tensors = _materialize_mtp_tensors(args.model, assignment,
                                           bf16_passthrough=bf16_passthrough,
                                           hist=hist)
    print(f"[export] MTP: {len(mtp_tensors)} tensors materialized", flush=True)

    # Visual encoder is still source passthrough (we deferred real
    # calibration for it). MTP passthrough is now restricted to source
    # keys NOT covered by our MTP materialization (layernorms, mtp.fc
    # when allocator chose BF16, etc.).
    print("[export] merging visual + residual mtp passthrough from source ...",
          flush=True)
    src_extra = _load_source_passthrough(
        args.model,
        prefix_filters=("model.visual.", "mtp."),
    )
    # Drop source tensors whose target name was already materialized.
    # The materialize pass produces vLLM-native names (`mtp.fc.weight_packed`
    # etc.) — we strip any suffix back to the source base name to compare.
    materialized_bases: set[str] = set()
    for k in mtp_tensors:
        # k is like 'mtp.fc.weight_packed', 'mtp.fc.weight_scale',
        # 'mtp.fc.weight_global_scale', 'mtp.fc.weight' (if BF16), ...
        # Strip one suffix past the last dot.
        base = k
        for suf in (".weight_packed", ".weight_scale", ".weight_global_scale",
                    ".input_global_scale", ".weight"):
            if k.endswith(suf):
                base = k[:-len(suf)] + ".weight"
                break
        materialized_bases.add(base)
        # Also cover packed-expert per-expert outputs: 'mtp.layers.0.mlp.experts.E.gate_proj.weight_packed'
        # The source key is 'mtp.layers.0.mlp.experts.gate_up_proj' (shape [E, 2M, N]).
        import re as _re
        m = _re.match(r"^(mtp\.layers\.\d+\.mlp\.experts)\.\d+\.(gate|up|down)_proj\.", k)
        if m:
            if m.group(2) in ("gate", "up"):
                materialized_bases.add(f"{m.group(1)}.gate_up_proj")
            else:
                materialized_bases.add(f"{m.group(1)}.down_proj")
    src_extra = {k: v for k, v in src_extra.items() if k not in materialized_bases}
    overlap = set(tensors) & set(src_extra)
    if overlap:
        for k in overlap:
            del src_extra[k]
    overlap_mtp = set(mtp_tensors) & set(src_extra)
    if overlap_mtp:
        for k in overlap_mtp:
            del src_extra[k]
    tensors.update(mtp_tensors)
    tensors.update(src_extra)
    print(f"[export] merged {len(src_extra)} passthrough + "
          f"{len(mtp_tensors)} mtp-quantized tensors", flush=True)

    # Sharded safetensors save.
    print("[export] writing safetensors shards ...", flush=True)
    write_sharded_safetensors(tensors, out_dir, args.shard_bytes)

    # Enumerate Linears the recipe DOESN'T mention so we can add them
    # to the ignore list. Without this, any Linear not in `assignment`
    # would be silently caught by the catch-all group's regex and vLLM
    # would try to load a bf16 weight into an NVFP4 packed param.
    # Examples on Qwen3.6: routers (`mlp.gate`), `shared_expert_gate`,
    # `linear_attn.norm` (which is RMSNorm, not Linear, so excluded),
    # vision encoder Linears.
    extra_ignore: list[str] = []
    # Reload the model briefly via a no-op iteration would be expensive;
    # instead, scan the source safetensors for any 2D `.weight` keys
    # that aren't covered by `assignment`.
    seen_recipe = {n for n in assignment}
    src_dir = Path(args.model)
    if src_dir.exists():
        from safetensors.torch import safe_open
        import os as _os
        for f in sorted(_os.listdir(src_dir)):
            if not f.endswith(".safetensors"):
                continue
            with safe_open(str(src_dir / f), framework="pt") as sf:
                for k in sf.keys():
                    if not k.endswith(".weight"):
                        continue
                    base = k[:-7]   # strip .weight
                    # The recipe uses text-only naming `model.layers.X.*`;
                    # source uses multimodal `model.language_model.layers.X.*`.
                    # Convert source-name → recipe-name to compare.
                    if base.startswith("model.language_model."):
                        recipe_name = "model." + base[len("model.language_model."):]
                    else:
                        recipe_name = base
                    if recipe_name in seen_recipe:
                        continue
                    # Skip norm-like + embed-like + bias + 1D modules — only
                    # Linears need explicit ignore (catch-all targets Linear).
                    # We approximate "is this a Linear weight" by looking
                    # at the tensor's rank.
                    try:
                        meta = sf.get_slice(k)
                        shape = list(meta.get_shape())
                    except Exception:
                        shape = []
                    if len(shape) != 2:
                        continue
                    extra_ignore.append(base)

    print(f"[export] extra ignore (unmapped Linears): {len(extra_ignore)}",
          flush=True)

    # Write config.json with quantization_config.
    print("[export] writing config.json ...", flush=True)
    write_config_with_quantization(
        args.model, out_dir, assignment, bf16_passthrough,
        extra_ignore=extra_ignore,
    )

    # Tokenizer + auxiliary files.
    print("[export] copying tokenizer files ...", flush=True)
    _copy_tokenizer(args.model, out_dir)

    # Manifest for traceability.
    with open(out_dir / "mixed_native_manifest.json", "w") as f:
        json.dump({
            "source_model": args.model,
            "source_recipe": args.layer_config,
            "format_histogram": {f"{k[0]}/{k[1]}": v for k, v in hist.items()},
            "n_assignment_entries": len(assignment),
            "ignore": sorted(bf16_passthrough),
        }, f, indent=2)

    print(f"[export] done. Serve with:\n"
          f"  vllm serve {out_dir.resolve()} --quantization compressed-tensors",
          flush=True)


# ---------------------------------------------------------------------------
# Sharded safetensors writer (mirrors HF transformers' shard layout so
# the index file is the same one transformers + vLLM expect).
# ---------------------------------------------------------------------------
def write_sharded_safetensors(
    tensors: dict[str, torch.Tensor],
    out_dir: Path,
    shard_bytes: int,
) -> None:
    keys = sorted(tensors.keys())
    sizes = {k: tensors[k].numel() * tensors[k].element_size() for k in keys}
    total = sum(sizes.values())
    n_shards = max(1, math.ceil(total / shard_bytes))
    target = math.ceil(total / n_shards)

    shards: list[list[str]] = [[]]
    cur = 0
    for k in keys:
        if cur + sizes[k] > target and shards[-1]:
            shards.append([])
            cur = 0
        shards[-1].append(k)
        cur += sizes[k]

    if len(shards) == 1:
        path = out_dir / "model.safetensors"
        save_file(
            {k: tensors[k].contiguous() for k in shards[0]},
            str(path),
            metadata={"format": "pt"},
        )
        return

    weight_map: dict[str, str] = {}
    n = len(shards)
    for i, shard_keys in enumerate(shards):
        shard_name = f"model-{i+1:05d}-of-{n:05d}.safetensors"
        save_file(
            {k: tensors[k].contiguous() for k in shard_keys},
            str(out_dir / shard_name),
            metadata={"format": "pt"},
        )
        for k in shard_keys:
            weight_map[k] = shard_name

    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump({
            "metadata": {"total_size": total},
            "weight_map": weight_map,
        }, f, indent=2)


def write_config_with_quantization(
    src_model: str, out_dir: Path,
    assignment: dict[str, str],
    bf16_passthrough: set[str],
    extra_ignore: Iterable[str] = (),
) -> None:
    src_cfg_path = Path(src_model) / "config.json"
    cfg = json.load(open(src_cfg_path))
    qc = build_quantization_config(assignment, bf16_passthrough, extra_ignore)
    if qc:
        cfg["quantization_config"] = qc
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def _materialize_mtp_tensors(src_model: str,
                             assignment: dict[str, str],
                             *,
                             bf16_passthrough: set[str],
                             hist: dict) -> dict[str, torch.Tensor]:
    """Quantize MTP weights per the allocator recipe.

    Transformers v5 does not instantiate MTP modules when loading
    Qwen3.5/3.6 MoE checkpoints (see `_keys_to_ignore_on_load_unexpected`),
    so `materialize_tensors` — which walks `model.named_modules()` —
    never sees any `mtp.*` entry in `assignment`. We build a standalone
    MTP module, load the source `mtp.*` weights into it, wrap it in a
    parent module named `mtp` (so qualified names come out as
    `mtp.fc`, `mtp.layers.0.self_attn.q_proj`, ...), and run the same
    materialize pass.

    Output tensor names match the checkpoint convention (`mtp.fc.*`,
    `mtp.layers.0.<rest>`). vLLM's `qwen3_5_mtp.load_weights` remaps
    `mtp.→model.` at load time.
    """
    from .mtp_probe import MtpModule, _load_into_mtp, _load_mtp_state_dict
    from transformers import AutoConfig

    # Build an MTP wrapper with source weights.
    cfg = AutoConfig.from_pretrained(src_model, trust_remote_code=True)
    text_config = getattr(cfg, "text_config", cfg)
    inner = MtpModule(text_config)
    wrapper = nn.Module()
    wrapper.add_module("mtp", inner)
    wrapper.to(dtype=torch.bfloat16)
    raw = _load_mtp_state_dict(src_model)
    _load_into_mtp(inner, raw)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    # Filter assignment to just `mtp.*` entries.
    mtp_assignment = {k: v for k, v in assignment.items() if k.startswith("mtp.")}
    if not mtp_assignment:
        return {}

    out, sub_hist = materialize_tensors(
        wrapper, mtp_assignment, bf16_passthrough=bf16_passthrough,
    )
    # Merge MTP histogram into caller's.
    for k, v in sub_hist.items():
        hist[("mtp_" + k[0], k[1])] = hist.get(("mtp_" + k[0], k[1]), 0) + v
    return out


def _load_source_passthrough(src_model: str,
                             prefix_filters: tuple[str, ...]
                             ) -> dict[str, torch.Tensor]:
    """Pull tensors from the source safetensors whose key begins with
    any of `prefix_filters`. Returns the loaded tensors so they can be
    written back verbatim into the export. Used for visual encoder +
    MTP head weights that the recipe doesn't touch but vLLM expects to
    find at load time.
    """
    import os
    from safetensors.torch import safe_open
    src_dir = Path(src_model)
    out: dict[str, torch.Tensor] = {}
    for f in sorted(os.listdir(src_dir)):
        if not f.endswith(".safetensors"):
            continue
        with safe_open(str(src_dir / f), framework="pt") as sf:
            for k in sf.keys():
                if any(k.startswith(p) for p in prefix_filters):
                    out[k] = sf.get_tensor(k)
    return out


def _copy_tokenizer(src_model: str, out_dir: Path) -> None:
    src = Path(src_model)
    for name in (
        "tokenizer_config.json", "tokenizer.json", "chat_template.jinja",
        "special_tokens_map.json", "merges.txt", "vocab.json",
        "added_tokens.json", "generation_config.json", "configuration.json",
        # Multimodal preprocessor configs — vLLM's loader for
        # qwen3_vl_moe constructs the multimodal processor even for
        # text-only requests, so the preprocessor files must travel
        # with the checkpoint.
        "preprocessor_config.json", "video_preprocessor_config.json",
        "processor_config.json",
    ):
        p = src / name
        if p.exists():
            shutil.copy2(p, out_dir / name)


if __name__ == "__main__":
    main()
