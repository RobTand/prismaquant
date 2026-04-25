#!/usr/bin/env python3
"""export_native_compressed.py — materialize a PrismaQuant recipe as a
standard `compressed-tensors` checkpoint that vLLM serves natively.

This is the unified export path. Decoder layers are streamed from
safetensors one at a time: the model skeleton is built on meta via
`init_empty_weights`, head + embed + norm + lm_head + rotary stay
resident, and each decoder layer flows disk → quantize → emit → unload.
Small models pay the no-op cost of a LayerCache large enough to keep
everything resident; big models (Qwen3.5-122B at 244 GB BF16) fit
through the same path on a 121 GB host.

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
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors.torch import save_file

from .model_profiles.qwen3_5 import Qwen3_5Profile
from .schemas import validate_layer_config_payload, validate_prune_manifest_payload

# ---------------------------------------------------------------------------
# NVFP4 packing (inlined from compressed-tensors fp4_quantized.py to avoid
# importing the library's __init__ which pulls in transformers internals
# that are not stable across the 4.x → 5.x break).
# ---------------------------------------------------------------------------
FLOAT_TO_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
NVFP4_MAX = 6.0     # max(|FLOAT_TO_E2M1|)
FP8_E4M3_MAX = 448.0  # max representable in torch.float8_e4m3fn

# Back-compat exports for unit tests that validate the Qwen3.5 naming
# and per-expert catch-all contract via the historical helper symbols.
_COMPAT_QWEN_PROFILE = Qwen3_5Profile()
PER_EXPERT_MOE_REGEX = _COMPAT_QWEN_PROFILE.per_expert_moe_regex()


def _to_vllm_internal_name(checkpoint_name: str) -> str:
    """Compatibility helper kept for unit tests.

    The production path is profile-driven via `profile.to_vllm_internal_name`;
    this helper preserves the historical Qwen3.5/3.6 mapping semantics
    without depending on a local vLLM install.
    """
    name = checkpoint_name
    if name.startswith("mtp."):
        return name
    if name == "lm_head":
        return "language_model.lm_head"
    if name.startswith("model.visual."):
        return name[len("model."):]
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model."):]
    if (name.startswith("model.layers.")
            or name.startswith("model.embed_tokens")
            or name.startswith("model.norm")
            or name == "model"):
        return "language_model.model." + name[len("model."):]
    return name


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


INT2_MAX = 1   # symmetric: codes in {-2, -1, 0, 1} from (1<<(2-1))-1
INT3_MAX = 3   # symmetric: codes in {-4, ..., 3}


def _canonical_export_format(fmt: str) -> str:
    return {"NVINT2": "INT2", "NVINT3": "INT3"}.get(fmt, fmt)


def pack_int2(signed: torch.Tensor, last_dim: int) -> torch.Tensor:
    """Pack signed 2-bit ints ({-2,-1,0,1}) into uint8, 4 values per byte.

    Storage layout: LSB-first within each byte. Value i occupies bits
    [2i, 2i+1] of its byte. Value is stored as two's-complement-2bit
    (the sign bit is bit 1).
    """
    if last_dim % 4 != 0:
        raise ValueError(f"INT2 pack requires last dim divisible by 4; got {last_dim}")
    # Map signed -> unsigned 2-bit (two's complement: -2->0b10, -1->0b11, 0->0b00, 1->0b01).
    u = (signed.to(torch.int32) & 0b11).to(torch.uint8)
    q = u.reshape(*signed.shape[:-1], last_dim // 4, 4)
    return (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)).to(torch.uint8)


def pack_int3(signed: torch.Tensor, last_dim: int) -> torch.Tensor:
    """Pack signed 3-bit ints ({-4..3}) into uint8, 8 values per 3 bytes.

    Bitstream LSB-first: value i occupies bits [3i, 3i+2] of the
    flattened bitstream. Last dim must be divisible by 8 for clean
    byte-aligned packing (3 * 8 = 24 bits = 3 bytes per group of 8).
    """
    if last_dim % 8 != 0:
        raise ValueError(f"INT3 pack requires last dim divisible by 8; got {last_dim}")
    # 3-bit two's-complement (int range [-4, 3]; we store 0b000..0b111).
    u = (signed.to(torch.int32) & 0b111).to(torch.int32)
    leading = signed.shape[:-1]
    groups = u.reshape(*leading, last_dim // 8, 8).to(torch.int32)
    b0 = (groups[..., 0] | (groups[..., 1] << 3) | ((groups[..., 2] & 0b011) << 6)).to(torch.uint8)
    b1 = (((groups[..., 2] >> 2) & 0b001) | (groups[..., 3] << 1) | (groups[..., 4] << 4)
          | ((groups[..., 5] & 0b001) << 7)).to(torch.uint8)
    b2 = (((groups[..., 5] >> 1) & 0b011) | (groups[..., 6] << 2) | (groups[..., 7] << 5)).to(torch.uint8)
    # Stack the three bytes per group into the packed byte stream, last-dim first.
    packed = torch.stack([b0, b1, b2], dim=-1).reshape(*leading, last_dim * 3 // 8)
    return packed.to(torch.uint8)


def quantize_dequantize_int(
    weight: torch.Tensor, bits: int, group_size: int = 16,
    global_real_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric-int quantizer with the NVFP4-style scale envelope.

    Returns `(weight_packed, weight_scale_fp8, weight_global_scale)`:
      - weight_packed: uint8, bit-packed signed ints (2 or 3 bits each)
      - weight_scale_fp8: fp8_e4m3fn per-group, shape [rows, n_groups]
      - weight_global_scale: fp32 scalar (divisor convention, 1/global_real)

    vLLM loader reconstructs: w_fp32[g,i] ≈ code[g,i] * scale_fp8[g] * global_real.
    The decode-to-NVFP4 path then re-rounds those fp32 values to the
    NVFP4 E2M1 grid and runs the existing NVFP4 GEMM kernel.
    """
    if bits not in (2, 3):
        raise ValueError(f"INT quantization only supports bits=2 or 3; got {bits}")
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"INT{bits} group_size={group_size} ∤ {cols}")
    int_max = INT2_MAX if bits == 2 else INT3_MAX
    int_min = -(int_max + 1)
    n_groups = cols // group_size
    grouped = weight.float().reshape(rows, n_groups, group_size)
    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)
    s_g_real = max_abs / int_max
    if global_real_override is not None:
        global_real = global_real_override.to(weight.device).clamp_min(1e-12)
    else:
        global_real = (s_g_real.amax() / FP8_E4M3_MAX).clamp_min(1e-12)
    fp8_scale_real = (s_g_real / global_real).clamp(0, FP8_E4M3_MAX)
    q = torch.round(grouped / s_g_real.unsqueeze(-1).clamp_min(1e-12)).clamp(int_min, int_max)
    q = q.to(torch.int8)
    packer = pack_int2 if bits == 2 else pack_int3
    weight_packed = packer(q.reshape(rows, cols), cols)
    return (
        weight_packed,
        fp8_scale_real.to(torch.float8_e4m3fn),
        (1.0 / global_real).to(torch.float32).reshape(1),
    )


def pack_fp4_indices(fp4_indices: torch.Tensor, last_dim: int) -> torch.Tensor:
    """Pack a tensor of 4-bit indices (final dim must be even) into
    uint8, two indices per byte. Preserves leading dimensions.
    """
    if last_dim % 2 != 0:
        raise ValueError("nvfp4 pack requires an even last dim")
    pairs = fp4_indices.reshape(*fp4_indices.shape[:-1], last_dim // 2, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)


DEFAULT_INPUT_GLOBAL_SCALE = 1.0  # placeholder; overridden by calibration

# FP4 E2M1 maximum representable value. Used to rescale activations so
# they fit inside the FP4 grid after the per-tensor scale divide.
_FP4_E2M1_MAX = 6.0


def compute_nvfp4_input_global_scale(activations: torch.Tensor) -> float:
    """Per-tensor input_global_scale from cached activations.

    Returns `max(|activations|) / 6.0` so that `a / input_global_scale`
    lies in [-6, 6] — the representable range of FP4 E2M1 for per-group
    quant downstream. Activations can be any shape; we flatten for the
    max.
    """
    max_abs = float(activations.detach().abs().max().item())
    if max_abs <= 0.0:
        return float(DEFAULT_INPUT_GLOBAL_SCALE)
    # Use reciprocal convention matching vLLM's CompressedTensorsW4A4Nvfp4
    # which interprets input_global_scale as a *reciprocal* scale factor
    # applied when computing activation-quant group scales: a_q = a * s.
    # So s = FP4_MAX / max_abs means scaled_a ∈ [-FP4_MAX, +FP4_MAX].
    return _FP4_E2M1_MAX / max_abs


# Module-level cache populated by main() when --activation-cache-dir is
# provided. `_quantize_2d`'s NVFP4 branch consults it by recipe-name
# when no explicit override is passed in. Keyed by the recipe name
# (post-profile.live_to_recipe_name remap). None means "not computed".
_INPUT_GLOBAL_SCALES: dict[str, float] | None = None

# Module-level raw-activation cache populated by main() when
# --activation-cache-dir is provided AND any of the activation-aware
# passes (--awq / --gptq / --act-weighted-round) is enabled. Keyed
# by recipe name; values are 2D `[N, in_features]` float32 tensors
# (lazily upcast from the on-disk bfloat16 for numerical stability
# during Hessian + per-channel stats). None means "not loaded".
_CACHED_ACTIVATIONS: object | None = None


class _LazyActivationCache:
    """ActivationIndex-backed mapping with a dict-like `.get()`.

    Export only needs a Linear's calibration rows while quantizing that
    one Linear. Preloading every activation tensor as float32 keeps
    tens of GiB resident for the entire export and OOMs large MoE
    checkpoints before the sharded writer runs. Keep scale calibration
    eager, but make raw activation reads demand-driven.

    TODO(perf): this whole probe -> cost -> export activation flow needs
    a larger redesign. Thousands of tiny `.pt` activation files plus
    late whole-checkpoint materialization are avoidable; use per-layer
    activation bundles and streaming safetensors writes.
    """

    def __init__(self, index):
        self.index = index
        self.loads = 0

    def get(self, name: str):
        if name not in self.index:
            return None
        self.loads += 1
        return self.index.load(name).to(torch.float32)

# Module-level flag bundle that controls which activation-aware
# passes run when `_quantize_2d` is invoked from main()'s streaming
# loop. Kept as module-level state (mirroring _INPUT_GLOBAL_SCALES)
# so we don't have to thread 3 boolean kwargs through every call
# site — unit tests pass the flags directly via kwargs.
_ACT_AWARE_FLAGS: dict[str, bool] = {
    "awq": False,
    "gptq": False,
    "awq_round": False,
    "scale_sweep": False,
}

# Proper-AWQ fold scales: maps target Linear recipe name -> float32 1D
# tensor `s[in_features]` that was folded into the predecessor RMSNorm
# γ and simultaneously multiplied into the target's weight IN-PLACE by
# `_awq_fold_layer_predecessors`. Populated per-layer by the streaming
# loop. The entry is only used downstream to DIVIDE the cached
# activations for GPTQ and activation-weighted rounding — at runtime
# vLLM will feed `a/s` into the Linear because γ already has 1/s folded
# in, so for any error-minimization pass that references cached
# activations, we must divide by `s` to match the runtime distribution.
# The weight path does not consult this dict: weights have already been
# pre-scaled in-place by the fold pass.
_AWQ_PROPER_SCALES: dict[str, torch.Tensor] = {}

# Targets whose predecessor is a non-linearity (softmax, silu*up, linear-
# attn recurrent state, etc.). Proper AWQ cannot fold into these, so we
# fall back to PURE RTN.
_AWQ_SKIP_LEAF_NAMES = frozenset({
    "o_proj",          # attention V->softmax@V->o_proj path
    "down_proj",       # silu(gate) * up nonlinear product
    "out_proj",        # DeltaNet internal recurrent state
})


# ---------------------------------------------------------------------------
# Activation-aware quantization passes (closed-form, no iterative search).
#
# All three reuse the probe's already-cached activations; none of them
# perform gradient-based optimization. Composed in the NVFP4 path of
# `_quantize_2d`:  AWQ rescale → per-group RTN → GPTQ error prop →
# activation-weighted rounding polish.
# ---------------------------------------------------------------------------
def _awq_channel_scale(activations: torch.Tensor, eps: float = 1e-4,
                       clamp_ratio: float = 10.0,
                       ) -> torch.Tensor:
    """Compute AWQ per-input-channel scale `s[c] = mean|a[:, c]|^0.5`,
    normalized by the geometric mean of its max and min (AutoAWQ /
    LMQuant convention), and HARD-CLAMPED to a log-symmetric window
    `[1/clamp_ratio, clamp_ratio]` for bf16-runtime numerical safety.

    Why geomean not max: max-normalization pushes low-activation
    channels toward `eps`, making `γ/s` blow up by 1/eps at inference
    time. In bf16 runtime the cancellation `(W*s)·(γ/s) = W·γ` loses
    precision catastrophically when the ratio is extreme. Geomean
    normalization centers `s` around 1 in log space; the extra hard
    clamp at 10× caps bf16 error accumulation on real-world channel
    imbalance (some Qwen layers have max/min activation-mean ratios
    of ~1e4 which the geomean alone only tames to ~100×).

    Returns a float32 1D tensor of length `in_features`.
    """
    a = activations.detach().to(torch.float32).reshape(-1, activations.shape[-1])
    mean_abs = a.abs().mean(dim=0)                       # [in_features]
    s = mean_abs.clamp_min(eps).pow(0.5)                 # α = 0.5
    # Geomean normalization: s / sqrt(s_max * s_min) — centers around 1
    # in log space. See AutoAWQ `quantize/quantizer.py:406` and llm-awq
    # `auto_scale.py:130`.
    norm = (s.max() * s.min()).sqrt().clamp_min(eps)
    s = s / norm
    # Hard clamp on the ratio — bf16 mantissa is 8 bits, so per-product
    # error is ~0.4%. Keeping max(s)/min(s) ≤ clamp_ratio² bounds the
    # accumulated matmul error from the cancellation pattern `W*s · γ/s`.
    s = s.clamp(1.0 / clamp_ratio, clamp_ratio)
    # Defensive nan/inf guard — a constant-zero activation channel can
    # otherwise poison the entire scale vector.
    s = torch.nan_to_num(s, nan=1.0, posinf=1.0, neginf=1.0)
    return s


def _awq_rescale_weight(weight: torch.Tensor, activations: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """AWQ-style per-input-channel rescaling of a 2D `[out, in]` weight.

    APPROXIMATE AWQ: the true AWQ algorithm (Lin et al. 2023) folds the
    reciprocal per-channel scale `1/s[c]` into the PREVIOUS layer's
    output (usually a LayerNorm or a residual add), so the inference-
    time composition `Q(W*s) @ (x/s) ≈ Q(W*s) · (1/s) @ x = Q(W*s) / s @ x`
    recovers `W @ x` up to quant noise. We can't fold the reciprocal
    back through the network at export time without knowing the full
    graph.

    Instead: rescale `W * s` to bias the FP4 group-scale math toward
    high-activation channels (they get finer grid resolution because
    the per-group max-abs along the scaled input dim is dominated by
    the scaled-up channels), quantize in that space, then divide out
    `s` from the dequantized result before storage. Net effect: quant
    noise in the final stored weight is redistributed — high-activation
    channels get proportionally less noise per unit of activation
    energy, at the cost of more noise in low-activation channels
    (whose contribution to the output is dampened anyway).

    Returns `(W_scaled, s)` where `W_scaled = W * s[None, :]` is ready
    for group-quant and `s` is the per-input-channel scale the caller
    must divide out post-quant (`W_dq_final = W_dq_scaled / s`).
    """
    if weight.shape[1] != activations.shape[-1]:
        raise ValueError(
            f"AWQ rescale: weight.in={weight.shape[1]} ≠ "
            f"act.in={activations.shape[-1]}"
        )
    s = _awq_channel_scale(activations).to(weight.device)
    W_scaled = weight.to(torch.float32) * s.unsqueeze(0)
    return W_scaled, s


def _awq_joint_channel_scale(
    activations_list: list[torch.Tensor], eps: float = 1e-4,
    clamp_ratio: float = 10.0,
) -> torch.Tensor:
    """Compute a single AWQ per-input-channel scale from a list of
    cached activations that all feed through the SAME predecessor
    (e.g. q/k/v all read from the same input_layernorm output, so all
    three share identical activations at that tap — but callers still
    pass the list for defensive stacking in case only a subset is
    present).

    Applies the same geomean normalization + hard clamp as
    `_awq_channel_scale`. See that function's docstring for the
    bf16-numerical-safety rationale.
    """
    combined = torch.cat(
        [a.detach().to(torch.float32).reshape(-1, a.shape[-1])
         for a in activations_list],
        dim=0,
    )
    mean_abs = combined.abs().mean(dim=0)
    s = mean_abs.clamp_min(eps).pow(0.5)
    norm = (s.max() * s.min()).sqrt().clamp_min(eps)
    s = s / norm
    s = s.clamp(1.0 / clamp_ratio, clamp_ratio)
    s = torch.nan_to_num(s, nan=1.0, posinf=1.0, neginf=1.0)
    return s


# ---------------------------------------------------------------------------
# Proper AWQ: fold reciprocal into predecessor (RMSNorm γ) with weight
# pre-scaling of EVERY reader of that γ. This is the only way to preserve
# the math invariant across mixed-format readers and packed-expert tensors.
#
# Invariant (per γ we fold):
#   γ_new := γ / s
#   For every reader M of γ:   M.W_new[:, in] := M.W[:, in] * s[in]
# then at runtime:    M(γ_new · x) = M.W_new · (γ/s · x) = (M.W * s) · (γ/s · x)
#                   = M.W · γ · x  =  M_original(γ · x)           (identity)
#
# The scale `s` is computed from the NVFP4 readers' cached activations
# (those are the readers we want to minimize quant error for). But the
# fold applies to ALL readers — NVFP4, MXFP8, BF16, packed experts.
# Missing this for any reader breaks the identity: γ feeds `x/s` but the
# reader still uses `W`, producing `(γ/s·x) · W` ≠ `γ·x · W`.
# ---------------------------------------------------------------------------

# Maps a layer-relative submodule path (or packed-expert param name) to
# the name of its predecessor RMSNorm on the decoder layer. Readers
# whose predecessor is nonlinear ("skip") do NOT participate in AWQ —
# neither the γ nor the reader's weight are touched.
#
# The mapping is indexed by LEAF NAME (last dotted segment) because both
# dense Linears and packed-expert param names share a flat leaf-name
# space at their respective containers. Submodule-path prefixes are used
# to disambiguate (e.g. `self_attn.q_proj` vs a hypothetical top-level
# `q_proj`).
_AWQ_PREDECESSOR_KIND: dict[str, str] = {
    # Full-attention path.
    "q_proj": "input_layernorm",
    "k_proj": "input_layernorm",
    "v_proj": "input_layernorm",
    "o_proj": "skip",
    # Linear-attention (DeltaNet) path.
    "in_proj_qkv": "input_layernorm",
    "in_proj_z": "input_layernorm",
    "in_proj_a": "input_layernorm",
    "in_proj_b": "input_layernorm",
    "out_proj": "skip",
    # MLP path (dense, shared_expert, and packed-expert readers that
    # sit directly on `post_attention_layernorm(hidden)` — gate_proj /
    # up_proj / gate_up_proj / w1 / w3 all read the LN output).
    "gate_proj": "post_attention_layernorm",
    "up_proj": "post_attention_layernorm",
    "gate_up_proj": "post_attention_layernorm",
    "w1": "post_attention_layernorm",
    "w3": "post_attention_layernorm",
    # MoE router — also reads directly from post_attention_layernorm.
    # Qwen variants call it `gate`, Gemma/Mixtral call it `router`,
    # DeepSeek calls it `router.classifier`. We catch the common leaf
    # names here; `_awq_discover_layer_readers` adds a positional
    # check so other aliases still fold correctly.
    "gate": "post_attention_layernorm",
    "router": "post_attention_layernorm",
    # Nonlinear predecessors — do not fold.
    "down_proj": "skip",
    "w2": "skip",
}

# Packed-expert param names that read from `post_attention_layernorm`
# (i.e. their input dim is the LN output dim) vs those that don't.
_PACKED_READERS_OF_POST_LN = frozenset({
    "gate_proj", "up_proj", "gate_up_proj", "w1", "w3",
})


def _awq_discover_layer_readers(
    layer_mod: "nn.Module",
) -> dict["nn.Module", list[dict]]:
    """Enumerate every reader of every RMSNorm predecessor in the layer.

    Returns a dict mapping each predecessor module (γ-holder) to a list
    of reader records. Each reader record is one of:

      linear reader:
        {"kind": "linear", "sub_name": "self_attn.q_proj",
         "leaf": "q_proj", "mod": <nn.Linear>, "in_features": int}

      packed-expert reader:
        {"kind": "packed", "sub_name": "mlp.experts", "leaf": "gate_proj",
         "mod": <ExpertsModule>, "param_name": "gate_proj",
         "in_features": int}

    All modules that read the γ are included — regardless of their
    assigned format (NVFP4 / MXFP8 / BF16). The caller decides which
    readers contribute ACTIVATIONS (NVFP4 only) to compute the scale,
    but every reader is still weight-scaled by that scale.

    Predecessors whose kind is "skip" (post-nonlinearity readers like
    o_proj, down_proj) are excluded: those are not in the returned dict
    because there's no γ we can fold into on that path.
    """
    buckets: dict["nn.Module", list[dict]] = defaultdict(list)
    # First pass: nn.Linear readers.
    for sub_name, mod in layer_mod.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        leaf = sub_name.rsplit(".", 1)[-1]
        kind = _AWQ_PREDECESSOR_KIND.get(leaf)
        if kind is None or kind == "skip":
            continue
        try:
            pred_mod = layer_mod.get_submodule(kind)
        except AttributeError:
            continue
        if getattr(pred_mod, "weight", None) is None:
            continue
        buckets[pred_mod].append({
            "kind": "linear",
            "sub_name": sub_name,
            "leaf": leaf,
            "mod": mod,
            "in_features": int(mod.weight.shape[1]),
        })
    # Second pass: packed-experts readers. Params whose in-dim is the
    # post_attention_layernorm output participate; params after a
    # nonlinearity (down_proj / w2) are skipped.
    for sub_name, mod in layer_mod.named_modules():
        if not _is_packed_experts_module(mod):
            continue
        try:
            post_ln = layer_mod.get_submodule("post_attention_layernorm")
        except AttributeError:
            continue
        if getattr(post_ln, "weight", None) is None:
            continue
        for pn in _packed_experts_param_names(mod):
            if pn not in _PACKED_READERS_OF_POST_LN:
                continue
            p = getattr(mod, pn)
            if p.dim() != 3:
                continue
            in_features = int(p.shape[2])
            # Sanity: the γ we're about to fold into has dim matching the
            # reader's input dim. If not, skip this reader — folding would
            # corrupt. This guards against exotic layouts (e.g. gate_up_proj
            # shaped [E, 2*hidden, in] where in != hidden).
            if int(post_ln.weight.shape[-1]) != in_features:
                continue
            buckets[post_ln].append({
                "kind": "packed",
                "sub_name": sub_name,
                "leaf": pn,
                "mod": mod,
                "param_name": pn,
                "in_features": in_features,
            })
    return buckets


def _awq_fold_layer_predecessors(
    layer_mod: "nn.Module",
    layer_qname: str,
    assignment: dict[str, str],
    profile,
    activation_lookup: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Apply the proper-AWQ fold pass in-place on one resident decoder
    layer.

    Invariant established per predecessor γ:
        γ      ← γ / s
        M.W    ← M.W * s  (for every reader M of γ — Linear or packed)

    For each predecessor γ we:
      1. Enumerate every reader (all formats: NVFP4, MXFP8, BF16, packed
         experts). The `_awq_discover_layer_readers` helper finds them.
      2. Compute the joint scale `s` from cached activations of the
         NVFP4 readers ONLY. The scale represents "which channels need
         more FP4 quant grid budget" — an NVFP4-specific quantity. Non-
         NVFP4 readers don't contribute to the scale but ARE scaled by
         it so the identity holds.
      3. If no NVFP4 reader has cached activations (e.g. only BF16
         readers), skip this γ entirely — no fold.
      4. Fold `γ /= s` once, and multiply every reader's weight by `s`
         on the input dim in-place. For nn.Linear, scale `.weight.data`
         on columns (dim 1). For packed experts, scale the 3D param
         `.data` on dim 2 (`[E, out, in]`).

    Returns `{recipe_key -> s}` for every NVFP4 LINEAR reader (only —
    packed experts and non-NVFP4 readers don't appear). This dict is
    used downstream by `_quantize_2d` to divide cached activations by
    `s` when running GPTQ / activation-weighted rounding: at runtime
    the reader sees `a/s`, so the importance weighting and covariance
    must match.
    """
    readers_by_pred = _awq_discover_layer_readers(layer_mod)
    if not readers_by_pred:
        return {}

    per_target_scale: dict[str, torch.Tensor] = {}
    for pred_mod, readers in readers_by_pred.items():
        # Gather NVFP4-reader activations for scale computation. Non-
        # NVFP4 readers' activations are not loaded and the scale is
        # a weighting designed for 4-bit quant error — using non-NVFP4
        # readers' inputs would dilute the signal.
        nvfp4_readers: list[dict] = []
        acts_for_scale: list[torch.Tensor] = []
        for r in readers:
            # Build full qname. For Linear readers, sub_name already
            # ends in the leaf (e.g. `self_attn.q_proj`). For packed
            # experts, sub_name is the experts module (`mlp.experts`)
            # and the param_name is the suffix (`gate_proj`).
            if layer_qname:
                full = f"{layer_qname}.{r['sub_name']}"
            else:
                full = r["sub_name"]
            if r["kind"] == "packed":
                full = f"{full}.{r['param_name']}"
            recipe_key = profile.live_to_recipe_name(full)
            r["recipe_key"] = recipe_key
            r["full"] = full
            if assignment.get(recipe_key) != "NVFP4":
                continue
            if r["kind"] != "linear":
                # Packed-expert activations are keyed by the experts
                # module qname (not per-param), and the recipe key for
                # the param itself doesn't match that cache key. Skip
                # packed experts in the scale computation; they still
                # get scaled below.
                nvfp4_readers.append(r)
                continue
            acts = activation_lookup.get(recipe_key)
            if acts is None:
                nvfp4_readers.append(r)
                continue
            acts_for_scale.append(acts)
            nvfp4_readers.append(r)

        if not acts_for_scale:
            # No NVFP4 Linear with cached activations reads this γ —
            # nothing to fold. (Purely-BF16 or purely-packed buckets
            # land here; that's fine, fold is an optional optimization
            # for NVFP4 quant error.)
            continue

        # Sanity-check dim agreement across readers. `acts_for_scale`
        # all have `a.shape[-1] == in_features_nvfp4`. All other readers
        # must agree on in_features (because they all read the same γ).
        in_features = acts_for_scale[0].shape[-1]
        for r in readers:
            if r["in_features"] != in_features:
                raise RuntimeError(
                    f"[awq-fold] inconsistent in_features in layer "
                    f"{layer_qname!r}: γ at {type(pred_mod).__name__} "
                    f"feeds reader {r['sub_name']!r}.{r['leaf']} "
                    f"(in={r['in_features']}) but NVFP4 reader has "
                    f"in={in_features}. Aborting — fold would corrupt.")

        s = _awq_joint_channel_scale(acts_for_scale).to(device)
        s_safe = s.clamp_min(1e-12)

        # 1) Fold γ /= s (in-place on the layer-resident RMSNorm).
        gamma = pred_mod.weight
        g = gamma.detach().to(torch.float32).to(device)
        g_folded = g / s_safe
        gamma.data.copy_(g_folded.to(device=gamma.device, dtype=gamma.dtype))

        # 2) Scale every reader's weight on the input dim. In-place on
        # the resident weight storage. Both nn.Linear (2D) and packed
        # experts (3D [E, out, in]) receive the same logical update.
        for r in readers:
            if r["kind"] == "linear":
                lin_mod: nn.Linear = r["mod"]
                w = lin_mod.weight
                w_scaled = (w.detach().to(torch.float32).to(device)
                            * s.unsqueeze(0))
                w.data.copy_(w_scaled.to(device=w.device, dtype=w.dtype))
            elif r["kind"] == "packed":
                experts_mod = r["mod"]
                pn = r["param_name"]
                param = getattr(experts_mod, pn)
                # Scale on the in dim (index 2). Broadcast to [E, out, in].
                p_scaled = (param.detach().to(torch.float32).to(device)
                            * s.reshape(1, 1, -1))
                param.data.copy_(
                    p_scaled.to(device=param.device, dtype=param.dtype))
            else:
                raise RuntimeError(f"unknown reader kind: {r['kind']!r}")

        # 3) Report scale for each NVFP4 LINEAR reader so `_quantize_2d`
        # can divide cached activations by `s` for GPTQ / act-round.
        # Packed experts don't have per-param cached activations (they
        # share a single `experts`-module cache under a different key),
        # so emitting a scale for them would mislead `_quantize_2d`'s
        # lookup. The weight is already pre-scaled via the in-place
        # fold; when the packed path runs downstream it just quantizes
        # the scaled weights directly.
        for r in nvfp4_readers:
            if r["kind"] == "linear":
                per_target_scale[r["recipe_key"]] = s

    return per_target_scale


def _gptq_obs_rounding_nvfp4(
    weight: torch.Tensor, activations: torch.Tensor,
    group_size: int = 16, damp: float = 0.01,
    global_real_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """GPTQ one-shot OBS rounding for NVFP4 weights.

    Standard GPTQ (Frantar et al. 2022): build the activation covariance
    `H = X^T X + λ·diag(H)`, invert via Cholesky, then round columns in
    blocks (group_size=16 matching NVFP4's group structure). Error from
    each block's quant is propagated to the remaining columns via
    `H_inv`, which is the closed-form OBS update for least-squares loss
    `||W - W_q||_H^2`.

    Returns the dequantized, error-propagated weight `[out, in]`
    (float32). The caller still runs NVFP4 packing on this tensor to
    produce on-disk storage — the bits end up the same as if we had
    quantized `weight` directly but with a smaller output-space error.

    `damp = 0.01` adds `0.01·mean(diag(H))` to `diag(H)` for Cholesky
    stability. `global_real_override` threads through for fused-sibling
    consistency (same semantics as `quantize_dequantize_nvfp4`).
    """
    W = weight.to(torch.float32).clone()
    rows, cols = W.shape
    if cols % group_size != 0:
        raise ValueError(f"GPTQ requires group_size={group_size} ∤ {cols}")

    X = activations.detach().to(torch.float32).reshape(-1, cols)
    # H = X^T X; guard against near-zero diagonal (dead channels).
    H = X.t() @ X                                         # [in, in]
    diag_mean = torch.diagonal(H).mean().clamp_min(1e-12)
    H.diagonal().add_(damp * diag_mean)

    # Dead-channel handling (standard GPTQ trick): columns with zero
    # diagonal get set to identity-like so the Cholesky succeeds, and
    # we zero those weight columns.
    dead = torch.diagonal(H) <= 0
    if dead.any():
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

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
        # extreme activation degeneracy). Caller proceeds with vanilla.
        return W

    # Target NVFP4 grid. Pre-compute the per-tensor global_real so the
    # per-block quantization uses the same outer scale as the final
    # on-disk packing (otherwise error propagation would be under an
    # inconsistent scale). This mirrors quantize_dequantize_nvfp4.
    if global_real_override is not None:
        global_real = global_real_override.to(weight.device).clamp_min(1e-12).float()
    else:
        grouped_full = W.reshape(rows, cols // group_size, group_size)
        max_abs_full = grouped_full.abs().amax(dim=-1).clamp_min(1e-12)
        s_g_real_full = max_abs_full / NVFP4_MAX
        global_real = (s_g_real_full.amax() / FP8_E4M3_MAX).clamp_min(1e-12)

    cb = _nvfp4_codebook(W.device, dtype=torch.float32)   # [8] abs values
    # Build signed grid once: the 16 possible FP4 values.
    signed_grid = torch.cat([cb, -cb[1:]]).to(W.device)   # dedup 0

    for block_start in range(0, cols, group_size):
        block_end = min(block_start + group_size, cols)
        block = W[:, block_start:block_end]                # [rows, group_size]

        # Per-block RTN to NVFP4: per-row max within this block gives
        # the per-group scale (matching quantize_dequantize_nvfp4).
        block_max = block.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        s_g_real = block_max / NVFP4_MAX                  # [rows, 1]
        # fp8 per-group scale in the [0, 448] range after /global_real.
        fp8_scale_real = (s_g_real / global_real).clamp(0, FP8_E4M3_MAX)
        # Effective per-element scale = fp8_scale_real * global_real.
        eff_scale = (fp8_scale_real * global_real).clamp_min(1e-12)
        in_grid = block / eff_scale                        # scaled into [-6, 6]
        in_grid = in_grid.clamp(-NVFP4_MAX, NVFP4_MAX)
        fp4_idx = _round_to_codebook(in_grid)              # [rows, group_size]
        # Decode to value (signed codebook).
        abs_idx = fp4_idx & 0x7
        sign = -((fp4_idx >> 3).to(torch.float32) * 2 - 1)
        q_vals = sign * cb[abs_idx]                        # [rows, group_size]
        block_dq = q_vals * eff_scale                      # [rows, group_size]
        block_err = block - block_dq                       # [rows, group_size]

        # Propagate error to the remaining columns. Using the
        # upper-triangular factor U (Hinv = U^T U), the closed-form
        # update from GPTQ's paper (eq. 5) is:
        #   W[:, j+1:] -= (err / U[j,j]) · U[j, j+1:]
        # applied one column at a time within the block. Because we
        # quantize the whole block at once, the within-block error
        # propagation is skipped — the block's per-group scale is
        # already set, so per-column updates within the block would
        # re-trigger quantization.  The between-block propagation
        # handles inter-group error.
        if block_end < cols:
            # Treat each column's error as propagating with its own
            # diagonal divisor U[j,j], then dot with the row slice.
            # Batched: err_block / diag(U[block]) @ U[block, rest]
            U_block_diag = torch.diagonal(U)[block_start:block_end].clamp_min(1e-12)
            U_offdiag = U[block_start:block_end, block_end:]   # [gs, rest]
            prop = (block_err / U_block_diag.unsqueeze(0)) @ U_offdiag  # [rows, rest]
            W[:, block_end:] = W[:, block_end:] - prop

        W[:, block_start:block_end] = block_dq

    return W


def _activation_weighted_round_nvfp4(
    weight: torch.Tensor, activations: torch.Tensor,
    group_size: int = 16,
    global_real_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """For each weight, pick the NVFP4 grid neighbor (above or below)
    that minimizes per-column `|Δw|² · E[|a|²]`.

    Closed-form, no iteration: evaluate both rounding choices, keep
    the one with lower activation-weighted squared error per column.
    Returns dequantized weight `[out, in]` (float32) — caller still
    runs the NVFP4 packer on it, and because each weight lands on a
    valid grid point, the packed result matches this dequantized
    tensor bit-for-bit.
    """
    W = weight.to(torch.float32).contiguous()
    rows, cols = W.shape
    if cols % group_size != 0:
        raise ValueError(f"act-round requires group_size={group_size} ∤ {cols}")

    a = activations.detach().to(torch.float32).reshape(-1, cols)
    # Per-input-channel importance = E[a^2]. Clamp to avoid degenerate
    # channels (all-zero activations) making rounding indifferent.
    col_importance = a.pow(2).mean(dim=0).clamp_min(1e-12)     # [in]

    # Compute per-tensor outer scale consistently with
    # quantize_dequantize_nvfp4.
    grouped = W.reshape(rows, cols // group_size, group_size)
    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)       # [rows, n_g]
    s_g_real = max_abs / NVFP4_MAX
    if global_real_override is not None:
        global_real = global_real_override.to(W.device).clamp_min(1e-12).float()
    else:
        global_real = (s_g_real.amax() / FP8_E4M3_MAX).clamp_min(1e-12)
    fp8_scale_real = (s_g_real / global_real).clamp(0, FP8_E4M3_MAX)
    eff_scale = (fp8_scale_real * global_real).unsqueeze(-1).clamp_min(1e-12)
    # Scale into grid.
    in_grid = grouped / eff_scale                                # [rows, n_g, gs]

    cb = _nvfp4_codebook(W.device, dtype=torch.float32)          # [8]
    abs_x = in_grid.abs()
    idx = torch.bucketize(abs_x, cb)
    idx_lo = (idx - 1).clamp_min(0).clamp_max(cb.numel() - 1)
    idx_hi = idx.clamp_max(cb.numel() - 1)
    lo_v = cb[idx_lo]
    hi_v = cb[idx_hi]
    sign = torch.where(in_grid >= 0, 1.0, -1.0)
    neigh_lo = sign * lo_v
    neigh_hi = sign * hi_v
    # Deltas in grid space. Convert to weight space by multiplying
    # eff_scale. That preserves the per-column importance weighting
    # on real Δw² (what actually enters the output-space error).
    delta_lo = (neigh_lo - in_grid) * eff_scale                  # [rows, n_g, gs]
    delta_hi = (neigh_hi - in_grid) * eff_scale
    # col_importance broadcast: [cols] → [1, n_g, gs]
    col_imp = col_importance.reshape(1, cols // group_size, group_size)
    err_lo = delta_lo.pow(2) * col_imp
    err_hi = delta_hi.pow(2) * col_imp
    pick_hi = err_hi < err_lo
    chosen = torch.where(pick_hi, neigh_hi, neigh_lo)            # [rows, n_g, gs]

    W_dq = (chosen * eff_scale).reshape(rows, cols)
    return W_dq


def _scale_sweep_nvfp4(
    weight: torch.Tensor, activations: torch.Tensor,
    group_size: int = 16,
    global_real_override: torch.Tensor | None = None,
    grid: int = 32,
    span: tuple[float, float] = (0.5, 1.5),
    reference_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-group joint (scale, rounding) closed-form polish.

    For each NVFP4 group, sweep `grid` candidate scales spanning
    `[span[0]·s0, span[1]·s0]`, where s0 is the default max-abs scale
    derived from the pre-pass weight. For each candidate scale, run RTN
    on the NVFP4 codebook and compute the activation-weighted MSE
    `sum_j a_j²·(w_orig,j - w_q,j)²` against the ORIGINAL (pre-pass)
    weight. Keep the configuration minimizing MSE per group, with an
    improve-or-keep gate against whatever `weight` is coming in (which
    may already be post-GPTQ / post-awq_round).

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
    new per-group scales — the downstream packer re-derives fp8_scale
    from `max_abs(W_dq) / NVFP4_MAX` per group, which recovers the
    swept scales losslessly.
    """
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

    a = activations.detach().to(torch.float32).reshape(-1, cols)
    col_importance = a.pow(2).mean(dim=0).clamp_min(1e-12)  # [in]

    # Use the REFERENCE weight to set the default per-group scale (s0)
    # and to measure MSE against.
    ref_grouped = W_ref.reshape(rows, cols // group_size, group_size)
    in_grouped = W_in.reshape(rows, cols // group_size, group_size)
    max_abs = ref_grouped.abs().amax(dim=-1).clamp_min(1e-12)  # [rows, n_g]
    s_g_real = max_abs / NVFP4_MAX
    if global_real_override is not None:
        global_real = global_real_override.to(W_in.device).clamp_min(1e-12).float()
    else:
        global_real = (s_g_real.amax() / FP8_E4M3_MAX).clamp_min(1e-12)
    fp8_scale_real = (s_g_real / global_real).clamp(0, FP8_E4M3_MAX)
    eff_scale0 = (fp8_scale_real * global_real).unsqueeze(-1).clamp_min(1e-12)

    # Full NVFP4 symmetric codebook (15 levels; duplicated 0 is harmless).
    cb_pos = _nvfp4_codebook(W_in.device, dtype=torch.float32)  # [8] non-neg
    cb = torch.cat([-cb_pos.flip(0), cb_pos[1:]], dim=0)  # [15] signed
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
    bytes_per_row = n_g * grid * group_size * cb.numel() * 4
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
        v = gexp / sexp                             # [c, n_g, grid, gs]
        d = (v.unsqueeze(-1) - cb).abs()            # [c, n_g, grid, gs, 15]
        idx = d.argmin(dim=-1)                      # [c, n_g, grid, gs]
        del d, v
        Wq_cand = cb[idx] * sexp                    # [c, n_g, grid, gs]
        del idx
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
            # group_size disambiguates: FP8_SOURCE ships with group_size=128
            # (128×128 block-scaled, as stored in native-FP8 checkpoints
            # like MiniMax-M2/M2.7 and DeepSeek-V3); MXFP8 uses group_size=32
            # with E8M0 scales. Earlier versions collapsed both onto MXFP8
            # when FP8_SOURCE didn't exist yet — preserve that fallback for
            # unknown group sizes, but route recognizable FP8_SOURCE picks
            # to the passthrough branch in _quantize_2d so the 128-block
            # source tensors land on disk bit-identical to the input.
            if int(scheme_dict.get("group_size", 0)) == 128:
                return "FP8_SOURCE"
            return "MXFP8"
        # Low-bit INT / NVFP3 (load-time dequant to NVFP4 at serve time).
        # Emit the low-bit name here so downstream packer writes the
        # narrow payload; the vLLM loader expands to NVFP4 on load.
        if dt == "int" and bits == 2:
            return "INT2"
        if dt == "int" and bits == 3:
            return "INT3"
        if dt == "fp3_e2m0" and bits == 3:
            return "NVFP3"
        # MXFP6 variants (load-time dequant to MXFP8).
        if dt in ("mx_fp", "fp6_e3m2") and bits == 6:
            return "MXFP6_E3M2"
        if dt == "fp6_e2m3" and bits == 6:
            return "MXFP6_E2M3"
        raise ValueError(f"unsupported scheme: {scheme_dict!r}")
    if isinstance(scheme_dict, str):
        s = scheme_dict.lower()
        if s in ("nvfp4", "fp4", "4"):
            return "NVFP4"
        if s in ("mxfp8", "fp8", "8"):
            return "MXFP8"
        if s in ("int2", "nvint2", "2"):
            return "INT2"
        if s in ("int3", "nvint3", "3"):
            return "INT3"
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


# Matches a vLLM-internal per-expert Linear qname, e.g.
#   model.layers.10.mlp.experts.0.gate_proj
# (Qwen3.5 / MiniMax / Gemma4 layouts all normalize to this form via the
# profile's `to_vllm_internal_name`.)
_PER_EXPERT_LINEAR_RE = re.compile(
    r"^(?P<prefix>.*[.])layers[.](?P<L>\d+)[.](?P<inner>.*mlp)[.]"
    r"experts[.](?P<E>\d+)[.](?P<proj>gate|up|down)_proj$"
)


def _build_target_list(vllm_names: list[str]) -> list[str]:
    """Emit compressed-tensors regex targets with per-expert Linears
    collapsed from 1-per-expert enumerations to one compact regex per
    (layer-prefix, projection) pair.

    Why: without collapsing, a 256-expert / 62-layer MoE produces ~47k
    explicit regex targets in config_groups. vLLM's
    `find_matched_target` does an O(n²) per-Linear walk through this
    list with Python's built-in `re.match` LRU cache (bounded to ~512
    distinct patterns), so the cache thrashes and scheme dispatch
    takes hours. Collapsing shrinks that to ~(layers × projs × active
    formats) regexes — typically a few hundred — and scheme dispatch
    completes in seconds.

    Names that aren't per-expert Linears pass through as explicit
    `re:^...$` regexes (same output as before).

    Within a (layer, proj) bucket, if every expert index 0..N-1 is
    present we emit a `[0-9]+` regex; sparse subsets get an enumerated
    alternation.
    """
    from collections import defaultdict

    bucketed: dict[tuple[str, int, str, str], set[int]] = defaultdict(set)
    passthrough: list[str] = []
    # Pre-formed regex targets (e.g. the packed-MoE per-expert regex
    # build_quantization_config emits) must pass through verbatim.
    # Double-wrapping them via _explicit_regex would produce an
    # unmatchable `re:^re:^...$$`.
    preformed_regex: list[str] = []
    for n in vllm_names:
        if n.startswith("re:"):
            preformed_regex.append(n)
            continue
        m = _PER_EXPERT_LINEAR_RE.match(n)
        if not m:
            passthrough.append(n)
            continue
        prefix = m.group("prefix")
        L = int(m.group("L"))
        inner = m.group("inner")
        proj = m.group("proj")
        E = int(m.group("E"))
        bucketed[(prefix, L, inner, proj)].add(E)

    collapsed: list[str] = []
    for (prefix, L, inner, proj), experts in sorted(bucketed.items()):
        n = max(experts) + 1
        prefix_r = prefix.replace(".", "[.]")
        inner_r = inner.replace(".", "[.]")
        if experts == set(range(n)):
            expr = "[0-9]+"
        else:
            expr = "(" + "|".join(str(e) for e in sorted(experts)) + ")"
        collapsed.append(
            f"re:^{prefix_r}layers[.]{L}[.]{inner_r}[.]experts[.]{expr}"
            f"[.]{proj}_proj$"
        )

    out = (
        [_explicit_regex(n) for n in sorted(passthrough)]
        + sorted(preformed_regex)
        + sorted(collapsed)
    )
    return out


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
# Prune-manifest plumbing (REAP-style expert drop + reindex)
# ---------------------------------------------------------------------------
# Sidecar file written by the allocator next to `layer_config.json` as
# `<layer_config>.prune.json`. One entry per MoE layer that had experts
# dropped, keyed by router qname (e.g. `model.layers.0.mlp.gate`). Each
# entry carries:
#   num_experts_orig, num_experts_kept — the before/after counts
#   pruned_expert_ids : list[int]      — original eids to drop
#   kept_expert_ids   : list[int]      — original eids that survive
#   orig_to_new_eid   : {str(orig): new_dense_idx}  — reindex map
# The exporter uses this to (a) skip the pruned experts' tensors, (b)
# reindex kept experts to a contiguous 0..K-1 range in the output keys
# (vLLM's FusedMoE / ModuleList indexing requires dense), (c) shrink
# the router weight's out-dim to K rows in kept-order, and (d) update
# HF config num_experts fields.
_EXPERT_IDX_IN_QNAME_RE = re.compile(
    r"^(?P<parent>.+)\.experts\.(?P<eid>\d+)(?:\.(?P<rest>.+))?$"
)


def _load_prune_manifest(path: Path | str | None) -> dict[str, dict]:
    """Load a prune-sidecar JSON. Returns an empty dict when the file
    doesn't exist — non-prune exports are the default case."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    validate_prune_manifest_payload(data, str(p))
    return data


def _index_prune_by_parent(manifest: dict[str, dict]) -> dict[str, dict]:
    """Re-index the router-keyed manifest by the parent qname common to
    both the router and the experts module (e.g. `model.layers.0.mlp`
    for router `.mlp.gate` and experts container `.mlp.experts`).

    Return dicts carry the router_qname alongside the entry so callers
    can distinguish the router from its experts when they share a
    parent.
    """
    by_parent: dict[str, dict] = {}
    for router_qname, entry in manifest.items():
        parent = router_qname.rsplit(".", 1)[0]
        by_parent[parent] = {"router_qname": router_qname, **entry}
    return by_parent


def _resolve_linear_prune_action(
    full_qname: str,
    prune_by_parent: dict[str, dict],
) -> tuple[str, dict] | None:
    """Map a live nn.Linear's qname to a prune action.

    Returns one of:
      ("router", entry)  — this IS the gated router; shrink out-dim to
                           kept_expert_ids (in order).
      ("drop",   entry)  — this is a PRUNED per-expert Linear; skip it.
      ("reindex", entry) — this is a KEPT per-expert Linear; emit under
                           a reindexed qname. `entry["new_full"]` has
                           the rewritten qname.
      None               — not MoE-prune-relevant; fall through to the
                           default code path.
    """
    if not prune_by_parent:
        return None
    # Router check: exact qname match.
    parent = full_qname.rsplit(".", 1)[0]
    entry = prune_by_parent.get(parent)
    if entry is not None and entry["router_qname"] == full_qname:
        return "router", entry
    # Per-expert Linear: qname shape `<parent>.experts.<eid>.<rest>`.
    m = _EXPERT_IDX_IN_QNAME_RE.match(full_qname)
    if m is None:
        return None
    parent = m.group("parent")
    entry = prune_by_parent.get(parent)
    if entry is None:
        return None
    try:
        eid = int(m.group("eid"))
    except (TypeError, ValueError):
        return None
    if eid in set(entry["pruned_expert_ids"]):
        return "drop", entry
    new_eid = entry["orig_to_new_eid"].get(str(eid))
    if new_eid is None:
        return None
    rest = m.group("rest")
    new_full = (
        f"{parent}.experts.{new_eid}.{rest}" if rest
        else f"{parent}.experts.{new_eid}"
    )
    out_entry = dict(entry)
    out_entry["new_full"] = new_full
    out_entry["orig_eid"] = eid
    out_entry["new_eid"] = int(new_eid)
    return "reindex", out_entry


def _resolve_packed_experts_prune(
    experts_qname: str,
    prune_by_parent: dict[str, dict],
) -> dict | None:
    """For a packed-experts module at `experts_qname` (e.g.
    `model.layers.0.mlp.experts`), return the prune entry keyed by its
    parent (`model.layers.0.mlp`) if this layer is pruned. Otherwise
    None.
    """
    if not prune_by_parent:
        return None
    parent = experts_qname.rsplit(".", 1)[0]
    return prune_by_parent.get(parent)


def _shrink_router_weight(
    mod: nn.Linear,
    entry: dict,
) -> torch.Tensor:
    """Drop rows of the router's output dim, keeping `kept_expert_ids`
    in order. Validates against `num_experts_orig` up-front — a size
    mismatch means the manifest and the live router disagree and we
    would emit a silently-broken artifact otherwise.
    """
    w = mod.weight.detach()
    kept = entry["kept_expert_ids"]
    n_orig = int(entry["num_experts_orig"])
    if w.shape[0] != n_orig:
        raise RuntimeError(
            f"[export-stream] prune: router weight rows "
            f"({w.shape[0]}) != manifest num_experts_orig "
            f"({n_orig}). Manifest was built against a different "
            f"model — refusing to shrink."
        )
    idx = torch.as_tensor(kept, dtype=torch.long, device=w.device)
    return w.index_select(0, idx).contiguous()


# HF config field names that hold the MoE expert count. Different
# archs use different ones; we update whichever exist.
_MOE_EXPERT_COUNT_FIELDS = (
    "num_experts",
    "num_local_experts",
    "num_routed_experts",
    "n_routed_experts",
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


def _unify_input_global_scales_across_fused_siblings(
    scales: dict[str, float],
) -> dict[str, float]:
    """Post-process per-Linear input_global_scale values so fused-
    sibling groups share one scale.

    vLLM concatenates q/k/v (and gate/up) into a single fused Linear
    at load time and applies ONE input_global_scale to the forward
    pass. If the siblings' scales don't match, vLLM warns and reduces
    accuracy.

    Siblings receive the same upstream activation, so their max/6
    values are theoretically identical — but capture + subsampling
    order introduces float-precision drift in practice. Taking the
    max over the group picks the conservative value (every sibling's
    quantization range is at-or-above the computed max, so the fused
    Linear never truncates any sibling's activations). Siblings that
    weren't NVFP4-assigned pass through unchanged.
    """
    # Bucket siblings by fused group.
    groups: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for name in scales:
        g = _fused_dense_group(name)
        if g is None:
            continue
        groups.setdefault(g, []).append(name)

    out = dict(scales)
    n_unified = 0
    max_drift = 0.0
    for key, members in groups.items():
        members = [m for m in members if m in scales]
        if len(members) < 2:
            continue
        vals = [scales[m] for m in members]
        joint = max(vals)
        drift = max(abs(joint - v) for v in vals)
        max_drift = max(max_drift, drift)
        for m in members:
            out[m] = joint
        n_unified += 1
    if n_unified:
        print(f"[export-stream] unified input_global_scale across "
              f"{n_unified} fused-sibling groups "
              f"(max pre-unify drift: {max_drift:.3e})", flush=True)
    return out


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
    input_global_scale_override: float | None = None,
    linear_name: str | None = None,
    awq_enabled: bool = False,
    gptq_enabled: bool = False,
    awq_round_enabled: bool = False,
    scale_sweep_enabled: bool = False,
    cached_activations: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compress a 2D Linear weight under format `fmt`.

    Returns the dict of on-disk tensors keyed by the suffix
    (`weight_packed`, `weight_scale`, `weight_global_scale`, ...).

    `nvfp4_global_real_override`: when this Linear is one shard of a
    fused parameter (q/k/v/o, gate/up), pass the joint per-tensor
    scale shared across all siblings. vLLM warns when sibling scales
    differ and reports degraded accuracy; sharing avoids both.

    `input_global_scale_override`: per-Linear activation scale computed
    from calibration — `max_abs(cached_activations) / 6.0` so scaled
    activations fit in FP4 E2M1's ±6 range before per-group quant. If
    None, falls back to `DEFAULT_INPUT_GLOBAL_SCALE` (1.0). Calibrated
    values typically improve PPL noticeably on NVFP4 weights because
    otherwise vLLM's runtime activation quant uses an undersized
    dynamic range.

    `awq_enabled`, `gptq_enabled`, `awq_round_enabled`: activation-aware
    passes composed on the NVFP4 path, order = AWQ rescale → per-group
    RTN → GPTQ error prop → activation-weighted rounding polish. Each
    requires `cached_activations` (looked up from _CACHED_ACTIVATIONS
    by `linear_name` when not supplied explicitly). MXFP8 ignores
    gptq_enabled (8-bit quant noise is too small to justify the
    compute cost); AWQ + act-weighted rounding still run if enabled.

    `cached_activations`: optional `[N, in_features]` float tensor of
    probe-captured inputs for this Linear. If None and `linear_name`
    is set, `_CACHED_ACTIVATIONS[linear_name]` is used.

    `fmt = MXFP8` emits real MXFP8 tensors: fp8_e4m3fn weights plus
    E8M0 uint8 per-group scales (group_size=32).
    """
    fmt = _canonical_export_format(fmt)

    # Resolve activations from the module-level cache when not passed.
    acts = cached_activations
    if (acts is None and linear_name is not None
            and _CACHED_ACTIVATIONS is not None):
        acts = _CACHED_ACTIVATIONS.get(linear_name)

    # Device fix: cached activations are stored on CPU (float32) to
    # amortize load cost across many quant calls; weights land on the
    # export device (typically CUDA). Move activations to the weight's
    # device here so every downstream op (_awq_*, GPTQ H matrix,
    # act-weighted rounding) runs on a consistent device. Repairs
    # `Expected all tensors to be on the same device, but found at
    # least two devices, cuda:0 and cpu!` in live Qwen3.6-35B export.
    if acts is not None and acts.device != weight.device:
        acts = acts.to(weight.device, non_blocking=True)

    # Resolve act-aware flags from the module-level config when none
    # were explicitly enabled via kwargs — lets main() turn them on
    # once without threading through every call site. Kwargs still
    # win when any is set True (unit tests pass them explicitly).
    if not (awq_enabled or gptq_enabled or awq_round_enabled or scale_sweep_enabled):
        awq_enabled = bool(_ACT_AWARE_FLAGS.get("awq"))
        gptq_enabled = bool(_ACT_AWARE_FLAGS.get("gptq"))
        awq_round_enabled = bool(_ACT_AWARE_FLAGS.get("awq_round"))
        scale_sweep_enabled = bool(_ACT_AWARE_FLAGS.get("scale_sweep"))

    if fmt == "NVFP4":
        w_work = weight.to(torch.float32)
        # Proper AWQ contract: when the per-layer fold pass
        # `_awq_fold_layer_predecessors` has run, it has ALREADY
        # multiplied `W *= s` in-place on the caller's weight storage
        # (and divided the predecessor γ by the same `s`). The
        # `weight` argument we received here already carries that
        # scaling, so we do NOT re-scale inside `_quantize_2d`. The
        # `_AWQ_PROPER_SCALES[linear_name]` entry (if present) exists
        # solely to tell us "runtime activations for this module will
        # be `a/s` after γ-fold, so divide cached activations by `s`
        # when computing GPTQ covariance and activation-weighted
        # rounding importance."
        #
        # Test-path / inline callers that don't run the fold pass
        # simply leave `_AWQ_PROPER_SCALES` empty, in which case the
        # cached activations are used verbatim. AWQ by itself then
        # contributes nothing here — the rescaling IS the fold. GPTQ
        # and activation-weighted rounding still run unchanged.
        leaf_name = (linear_name or "").rsplit(".", 1)[-1]
        skip_awq = leaf_name in _AWQ_SKIP_LEAF_NAMES
        awq_s: torch.Tensor | None = None
        if (awq_enabled and not skip_awq and linear_name is not None
                and linear_name in _AWQ_PROPER_SCALES):
            s_cand = _AWQ_PROPER_SCALES[linear_name]
            if s_cand.numel() == w_work.shape[1]:
                awq_s = s_cand.to(device=w_work.device, dtype=torch.float32)

        def _acts_for_error_passes() -> torch.Tensor | None:
            """Return cached activations adjusted for the runtime
            distribution seen by this Linear. Under proper AWQ the
            predecessor now emits `a/s`, so GPTQ's H matrix and act-
            rounding's column importance must be computed from `a/s`.
            Without AWQ, use the raw cached activations directly."""
            if acts is None or acts.shape[-1] != w_work.shape[1]:
                return None
            if awq_s is None:
                return acts
            a2 = acts.to(torch.float32).reshape(-1, acts.shape[-1])
            return a2 / awq_s.clamp_min(1e-12).unsqueeze(0)

        # Step 2: GPTQ one-shot OBS rounding (block-wise error prop).
        # Produces an already-dequantized tensor living on the NVFP4
        # grid; subsequent packing is lossless wrt this tensor.
        if gptq_enabled and not skip_awq:
            acts_work = _acts_for_error_passes()
            if acts_work is not None:
                w_work = _gptq_obs_rounding_nvfp4(
                    w_work, acts_work, group_size=16,
                    global_real_override=nvfp4_global_real_override,
                )

        # Step 3: activation-weighted rounding polish. Measured to be
        # a no-op-at-best / GPTQ-undo-at-worst in the permutation
        # bake-off (see PrismaQuant repo notes). Off by default; leave
        # the code path here for A/B testing.
        if awq_round_enabled and not skip_awq:
            acts_work = _acts_for_error_passes()
            if acts_work is not None:
                w_work = _activation_weighted_round_nvfp4(
                    w_work, acts_work, group_size=16,
                    global_real_override=nvfp4_global_real_override,
                )

        # Step 3b: closed-form per-group scale sweep. Joint (scale,
        # rounding-set) search on the NVFP4 codebook, activation-
        # weighted MSE against the ORIGINAL pre-pass weight, with an
        # improve-or-keep gate against the current w_work. Recovers
        # most of AutoRound's benefit without its 200-iter SGD.
        if scale_sweep_enabled and not skip_awq:
            acts_work = _acts_for_error_passes()
            if acts_work is not None:
                w_work = _scale_sweep_nvfp4(
                    w_work, acts_work, group_size=16,
                    global_real_override=nvfp4_global_real_override,
                    reference_weight=weight.to(torch.float32),
                )

        # Step 4: final NVFP4 pack. `w_work` is the post-AWQ,
        # post-GPTQ, post-act-round, post-scale-sweep weight. Store it
        # as-is — the fold pass preserved the matmul identity
        # externally.
        wp, ws, wg = quantize_dequantize_nvfp4(
            w_work, group_size=16,
            global_real_override=nvfp4_global_real_override,
        )
        input_scale = input_global_scale_override
        if input_scale is None and linear_name is not None and _INPUT_GLOBAL_SCALES:
            input_scale = _INPUT_GLOBAL_SCALES.get(linear_name)
        if input_scale is None:
            input_scale = DEFAULT_INPUT_GLOBAL_SCALE
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
    if fmt == "MXFP8":
        # MXFP8 is pure RTN. AWQ/GPTQ/act-weighted-round are all
        # disabled on MXFP8: at 8-bit quant noise is already well
        # below 0.05 PPL and the quasi-AWQ cycle that previously
        # ran was mathematically equivalent to RTN at the stored
        # weight (rescale → dequant → divide out). Proper AWQ on
        # 8-bit would require the same predecessor-fold machinery
        # as NVFP4 but the marginal benefit doesn't justify it.
        w_work = weight.to(torch.float32)
        w, ws = quantize_dequantize_mxfp8(w_work, group_size=32)
        return {"weight": w, "weight_scale": ws}
    if fmt == "BF16":
        return {"weight": weight.to(torch.bfloat16)}
    if fmt in ("INT2", "INT3"):
        # Low-bit symmetric-int with NVFP4-style scale envelope.
        # Stored as bit-packed uint8 + fp8 per-group scale + fp32 global.
        # vLLM loader decodes to NVFP4 at process_weights_after_loading
        # and runs the NVFP4 GEMM kernel unchanged.
        w_work = weight.to(torch.float32)
        # Act-aware passes are noisy at 2/3-bit — keep RTN only.
        bits = 2 if fmt == "INT2" else 3
        wp, ws, wg = quantize_dequantize_int(
            w_work, bits=bits, group_size=16,
            global_real_override=nvfp4_global_real_override,
        )
        return {
            "weight_packed": wp,
            "weight_scale": ws,
            "weight_global_scale": wg,
        }
    raise ValueError(f"unsupported format: {fmt}")


def _quantize_int_packed(packed: torch.Tensor, fmt: str) -> dict[str, torch.Tensor]:
    """Per-expert INT2/INT3 packing for a 3D `[E, M, N]` tensor.

    Each expert gets its own global scale (so weight_global_scale has
    shape [E]). Mirrors `quantize_dequantize_nvfp4_packed` structure.
    """
    fmt = _canonical_export_format(fmt)
    bits = 2 if fmt == "INT2" else 3
    int_max = INT2_MAX if bits == 2 else INT3_MAX
    int_min = -(int_max + 1)
    group_size = 16
    E, M, N = packed.shape
    if N % group_size != 0:
        raise ValueError(f"{fmt} group_size={group_size} ∤ {N}")
    n_groups = N // group_size
    grouped = packed.float().reshape(E, M, n_groups, group_size)
    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)
    s_g_real = max_abs / int_max
    global_real = (s_g_real.amax(dim=(-2, -1)) / FP8_E4M3_MAX).clamp_min(1e-12)  # [E]
    fp8_scale_real = (s_g_real / global_real.view(E, 1, 1)).clamp(0, FP8_E4M3_MAX)
    q = torch.round(grouped / s_g_real.unsqueeze(-1).clamp_min(1e-12)).clamp(int_min, int_max)
    q = q.to(torch.int8).reshape(E, M, N)
    packer = pack_int2 if bits == 2 else pack_int3
    weight_packed = packer(q, N)  # [E, M, packed_cols]
    return {
        "weight_packed": weight_packed,
        "weight_scale": fp8_scale_real.to(torch.float8_e4m3fn),
        "weight_global_scale": (1.0 / global_real).to(torch.float32),
    }


def _quantize_3d_packed(packed: torch.Tensor, fmt: str) -> dict[str, torch.Tensor]:
    """Compress a 3D packed-expert tensor `[E, M, N]` as a single
    batched op (per-expert independent scales).

    Returns tensors with leading expert dim preserved, matching what
    vLLM's `compressed_tensors_moe_w4a4_nvfp4` allocates internally
    (uint8 packed weights, fp8/uint8 per-group scales, per-expert
    global scales for NVFP4).
    """
    fmt = _canonical_export_format(fmt)
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
        w, ws = quantize_dequantize_mxfp8_packed(packed, group_size=32)
        return {"weight": w, "weight_scale": ws}
    if fmt in ("INT2", "INT3"):
        return _quantize_int_packed(packed, fmt)
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
    if fmt in ("INT2", "INT3"):
        return _quantize_int_packed(stacked_weights.to(torch.float32), fmt)
    if fmt == "MXFP8":
        w, ws = quantize_dequantize_mxfp8_packed(
            stacked_weights.to(torch.float32), group_size=32,
        )
        return {"weight": w, "weight_scale": ws}
    raise ValueError(f"unsupported grouped 2D export format: {fmt}")


def _host_mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 1 << 30


def _export_vector_chunk_len(
    shape: tuple[int, int],
    max_items: int,
    device: torch.device,
) -> int:
    """Choose a conservative grouped-export chunk size.

    `PQ_EXPORT_VECTOR_CHUNK=<int>` pins the upper bound. The default `auto`
    keeps one path for all model sizes while scaling down when available
    memory is tight.
    """
    env = os.getenv("PQ_EXPORT_VECTOR_CHUNK", "auto").strip().lower()
    if env and env != "auto":
        try:
            cap = max(1, int(env))
        except ValueError:
            cap = 128
    else:
        cap = 128

    if device.type == "cuda":
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except RuntimeError:
            free_bytes = _host_mem_available_bytes()
    else:
        free_bytes = _host_mem_available_bytes()

    # Quantization creates grouped float32 views, integer code tensors, scale
    # tensors, and packed outputs. Budget for several live copies per item.
    per_item = max(1, int(math.prod(shape)) * 4)
    budget = max(16 << 20, min(int(free_bytes * 0.08), 2 << 30))
    by_mem = max(1, budget // max(per_item * 6, 1))
    return max(1, min(max_items, cap, by_mem))


# ---------------------------------------------------------------------------
# Fused-sibling joint NVFP4 scale (per-layer scope, used by the streaming
# materializer below). The whole-model variant `_compute_nvfp4_joint_global`
# lives above and is kept for the MTP path + unit tests.
# ---------------------------------------------------------------------------
_FUSED_SIBLINGS = {
    "q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv",
    "gate_proj": "gate_up", "up_proj": "gate_up",
    # Qwen3.5/3.6 DeltaNet linear-attention pairs. vLLM fuses
    # `in_proj_qkv + in_proj_z → in_proj_qkvz` and
    # `in_proj_b + in_proj_a → in_proj_ba` at load time; the fused
    # packed Linear needs ONE shared NVFP4 `weight_global_scale`.
    # Omitting these triggers vLLM's
    # `compressed_tensors_w4a4_nvfp4.py:97` warning about reduced
    # accuracy from mismatched parallel-layer scales.
    "in_proj_qkv": "qkvz", "in_proj_z": "qkvz",
    "in_proj_b": "ba", "in_proj_a": "ba",
}


def _compute_layer_joint_nvfp4(layer_mod: nn.Module,
                               layer_qname: str,
                               assignment: dict[str, str],
                               profile,
                               ) -> dict[str, torch.Tensor]:
    """Return {recipe_key -> joint global scale} for NVFP4 fused-sibling
    groups inside this decoder layer. Only keys assigned NVFP4 get an
    override entry; the rest compute per-Linear scales at quantize time.

    Under proper AWQ, fused siblings' weights have already been
    pre-scaled in-place by `_awq_fold_layer_predecessors` (q/k/v or
    gate/up share a γ, so they all receive the same `s`). Reading
    `mod.weight` here returns the already-scaled weight, so
    `compute_nvfp4_global_real` naturally produces the correct joint
    global for the post-AWQ stored weight.

    Semantically equivalent to a scoped `_compute_nvfp4_joint_global`
    across just this layer's modules."""
    groups: dict[tuple[str, str], list[tuple[str, nn.Linear]]] = defaultdict(list)
    for sub_name, mod in layer_mod.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        last = sub_name.rsplit(".", 1)[-1]
        fam = _FUSED_SIBLINGS.get(last)
        if fam is None:
            continue
        parent = sub_name.rsplit(".", 1)[0] if "." in sub_name else ""
        groups[(parent, fam)].append((sub_name, mod))

    out: dict[str, torch.Tensor] = {}
    for (_, _), members in groups.items():
        fqn_fmt = []
        for sub_name, mod in members:
            full = f"{layer_qname}.{sub_name}" if sub_name else layer_qname
            recipe_key = profile.live_to_recipe_name(full)
            fmt = assignment.get(recipe_key)
            fqn_fmt.append((full, recipe_key, fmt, mod))
        fmts = {f for _, _, f, _ in fqn_fmt}
        if fmts != {"NVFP4"}:
            continue
        candidates = [
            compute_nvfp4_global_real(mod.weight.detach().float(),
                                      group_size=16)
            for _, _, _, mod in fqn_fmt
        ]
        joint = torch.stack(candidates).max()
        for full, recipe_key, _, _ in fqn_fmt:
            out[recipe_key] = joint
    return out


def _init_rotary_inplace(base_model: nn.Module, device: torch.device,
                         dtype: torch.dtype) -> None:
    """After init_empty_weights, rotary modules exist but their
    `inv_freq` buffers are on meta. Re-run the module's own rope init
    (which is deterministic from config) so `inv_freq` lives on the
    exec device with correct values — matching what `from_pretrained`
    would have produced."""
    from .layer_streaming import _get_rotary
    rotary = _get_rotary(base_model)
    if rotary is None:
        return
    cfg = getattr(rotary, "config", None)
    if cfg is None:
        return
    try:
        rope_init_fn = rotary.compute_default_rope_parameters
    except AttributeError:
        return
    inv_freq, attention_scaling = rope_init_fn(cfg, device)
    rotary.register_buffer("inv_freq", inv_freq.to(dtype=torch.float32,
                                                   device=device),
                           persistent=False)
    if hasattr(rotary, "original_inv_freq"):
        rotary.register_buffer(
            "original_inv_freq",
            inv_freq.to(dtype=torch.float32, device=device).clone(),
            persistent=False)
    rotary.attention_scaling = attention_scaling


def _build_fp8_source_map(
    model_path: str, *, multimodal: bool = False,
) -> dict[str, tuple[str, str]]:
    """Scan the source safetensors index for native-FP8 block-scaled
    Linears and return `{live_base_name: (shard_path, ckpt_scale_inv_key)}`.

    A tensor qualifies as FP8-sourced when `<base>.weight` has a sibling
    `<base>.weight_scale_inv` in the index (the 128×128 block-scale
    convention MiniMax-M2, DeepSeek-V3, and NVIDIA FP8 checkpoints use).
    The returned keys are the LIVE-MODEL attribute paths (i.e., the same
    form as `full` in the per-layer loop), obtained by applying the same
    source → live name rewrite that `layer_streaming._build_weight_map`
    performs for the `.weight` tensors — so the exporter can look up
    directly by `live_base` without re-running the rewrite.

    `multimodal` must match what was passed to `_build_weight_map`:
    text-only path strips `model.language_model.` prefix; multimodal
    preserves it. (MiniMax-M2 is text-only; set False.)

    Returns `{}` when the source has no `.weight_scale_inv` sibling for
    any `.weight` — i.e., the source is not FP8-block quantized. In that
    case the FP8_SOURCE format is inert (allocator's passthrough-
    integrity filter drops it from every Linear's candidate set).
    """
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        single = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(single):
            return {}
        from safetensors import safe_open
        with safe_open(single, framework="pt") as f:
            raw = {k: single for k in f.keys()}
    else:
        with open(idx_path) as f:
            raw = json.load(f)["weight_map"]

    def _rename(k: str) -> str | None:
        # Mirror `layer_streaming._rename_text_only`, but WITHOUT the
        # `.weight_scale_inv` drop — we need those keys preserved.
        if not multimodal:
            if (k.startswith("model.visual.")
                    or k.startswith("model.audio_tower.")
                    or k.startswith("model.vision_tower.")
                    or k.startswith("model.embed_vision.")
                    or k.startswith("model.embed_audio.")
                    or k.startswith("mtp.")):
                return None
            if k.startswith("model.language_model."):
                return "model." + k[len("model.language_model."):]
            return k
        # multimodal umbrella
        if k.startswith("mtp."):
            return None
        return k

    # Group by `<live_base>`: the live-model qname without `.weight` /
    # `.weight_scale_inv` suffix.
    bases: dict[str, dict[str, tuple[str, str]]] = {}
    for ck_key, shard in raw.items():
        for suffix in (".weight_scale_inv", ".weight"):
            if ck_key.endswith(suffix):
                ck_base = ck_key[: -len(suffix)]
                live_base = _rename(ck_base)
                if live_base is None:
                    break
                bases.setdefault(live_base, {})[suffix[1:]] = (
                    os.path.join(model_path, shard), ck_key,
                )
                break

    out: dict[str, tuple[str, str]] = {}
    for live_base, kinds in bases.items():
        if "weight" in kinds and "weight_scale_inv" in kinds:
            # Only the scale_inv half is new information — the `.weight`
            # shard+ckpt_key is already in `weight_ckpt` from the main
            # loader. Callers combine the two.
            shard, ckpt_scale_inv_key = kinds["weight_scale_inv"]
            out[live_base] = (shard, ckpt_scale_inv_key)
    return out


def materialize_tensors_streaming(
    model_path: str,
    assignment: dict[str, str],
    *,
    profile,
    bf16_passthrough: set[str],
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
    offload_folder: str | None = None,
    prune_manifest: dict[str, dict] | None = None,
    tensor_sink: Callable[[dict[str, torch.Tensor]], None] | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Stream decoder layers through quantize → emit → unload. Never
    holds the full model in memory. Small models still exercise this
    path — the LayerCache just keeps everything resident, so load/
    unload degenerates to a no-op.

    Output: `(out_tensors, hist)` matching the shape the monolithic
    materialize used to return, ready for `write_sharded_safetensors`.
    When `tensor_sink` is supplied, each emitted head/layer batch is
    passed to the sink and cleared immediately; the returned tensor dict
    is then intentionally empty."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from .layer_streaming import (
        _build_fp8_scale_inv_map,
        _build_install_resolver,
        _build_weight_map,
        _fast_install,
        _get_layer_list,
        _head_prefixes,
        _materialize,
        _read_layer_to_device,
        _resolve_base_prefix,
        _unload,
    )
    from .sensitivity_probe import stage_text_only

    # ----- 1. Meta skeleton + manual head materialization -----
    # Pure `init_empty_weights` path — avoids accelerate's
    # `from_pretrained` which would write ~244 GB of offload files to
    # disk on Qwen3.5-122B before we ever read them. Instead we:
    #   (a) build the full skeleton on meta (0 bytes),
    #   (b) read head/embed/norm/lm_head tensors directly from the
    #       source safetensors and install on the exec device,
    #   (c) re-run rotary's init_fn to populate `inv_freq` (not in
    #       state_dict — computed from config),
    #   (d) leave decoder layers on meta until the per-layer loop
    #       streams them in.
    staged = stage_text_only(model_path)
    config = AutoConfig.from_pretrained(staged, trust_remote_code=True)
    # _init_weights is globally no-op'd by prismaquant.__init__'s
    # _polyfill_transformers (wasted work + transformers-5.x compat
    # landmine on remote modeling files).
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base_model, layers = _get_layer_list(model)
    base_prefix = _resolve_base_prefix(model, base_model)
    num_layers = len(layers)
    layers_prefix = f"{base_prefix}.layers." if base_prefix else "layers."

    weight_shard, weight_ckpt = _build_weight_map(model_path)
    # Native-FP8 dequant map, keyed by live weight-qname. Passed to
    # every `_read_layer_to_device` / `_materialize` call so fp8 source
    # weights land on the module as TRUE dequanted bf16 — not raw fp8
    # codes (range ±448) cast to bf16. Every downstream pass
    # (_quantize_2d for non-passthrough formats, probe Fisher, cost
    # RTN) then operates on the real weight values instead of scaled-
    # by-hidden-factor codes. Empty dict for BF16-native checkpoints.
    fp8_scale_inv_map = _build_fp8_scale_inv_map(model_path)
    if fp8_scale_inv_map:
        print(f"[export-stream] fp8 scale_inv map: "
              f"{len(fp8_scale_inv_map)} weights will be dequanted "
              f"inline at layer-load", flush=True)

    # FP8_SOURCE passthrough-emit map: keyed by live base name (no
    # `.weight` suffix), used by the `fmt == 'FP8_SOURCE'` emit branch
    # to copy source fp8 + scale_inv bytes verbatim into the output.
    # Distinct key format from the loader-side dequant map above.
    fp8_source_map = _build_fp8_source_map(model_path)
    prune_by_parent = _index_prune_by_parent(prune_manifest or {})
    if prune_by_parent:
        n_pruned_total = sum(
            e["num_experts_orig"] - e["num_experts_kept"]
            for e in prune_by_parent.values()
        )
        print(
            f"[export-stream] prune manifest: {len(prune_by_parent)} "
            f"MoE layers, {n_pruned_total} experts dropped total",
            flush=True,
        )
    if fp8_source_map:
        print(f"[export-stream] fp8 source-emit map: {len(fp8_source_map)} "
              f"Linears available for FP8_SOURCE passthrough", flush=True)

    # Materialize head (embed + norm + lm_head). These are in the
    # safetensors and get populated via `set_module_tensor_to_device`.
    print(f"[export-stream] base_prefix={base_prefix!r}  layers={num_layers}",
          flush=True)
    t0 = time.time()
    head_pfxs = _head_prefixes(None, base_prefix)
    loaded_n = _materialize(model, head_pfxs, weight_shard, weight_ckpt,
                            device, dtype,
                            fp8_scale_inv_map=fp8_scale_inv_map)

    # Rotary's `inv_freq` isn't in the state_dict — compute from config.
    _init_rotary_inplace(base_model, device, dtype)
    print(f"[export-stream] head materialized ({loaded_n} tensors, rotary "
          f"re-init) in {time.time()-t0:.1f}s", flush=True)

    out: dict[str, torch.Tensor] = {}
    hist: Counter = Counter()
    unmapped_keys: list[str] = []

    # ----- 2. Head / embed / norm / lm_head / rotary passthrough -----
    # These are resident on `device` already. Emit as BF16 passthrough
    # UNLESS `lm_head` (or similar) is explicitly in the assignment.
    t_head = time.time()

    def _emit_head_param(full_qname: str, param: nn.Parameter):
        recipe_key = profile.live_to_recipe_name(full_qname)
        # Recipe keys are module qnames (e.g. "lm_head"), not parameter
        # qnames ("lm_head.weight"). Strip the trailing `.weight` so the
        # assignment lookup hits — otherwise head params always fall
        # through to BF16 passthrough regardless of what the allocator
        # chose for them.
        if recipe_key.endswith(".weight"):
            recipe_key = recipe_key[:-len(".weight")]
        fmt = assignment.get(recipe_key)
        # Respect the passthrough set (e.g. `--ignore lm_head`) even if
        # the allocator assigned NVFP4/MXFP8 to this head module. See
        # the --ignore docstring for why lm_head is passthrough by
        # default despite vLLM rejecting quantized ParallelLMHead.
        if recipe_key in bf16_passthrough:
            fmt = "BF16"
        if fmt is not None:
            fmt = _canonical_export_format(fmt)
        if fmt == "FP8_SOURCE":
            raise NotImplementedError(
                f"[export-stream] FP8_SOURCE not wired for head params "
                f"(at {full_qname}). Native-FP8 checkpoints (MiniMax, "
                f"DeepSeek) keep lm_head/embed/norm in BF16 — the "
                f"allocator's passthrough-integrity filter should reject "
                f"FP8_SOURCE for these. If a future model ships FP8 "
                f"head weights, add the passthrough path here.")
        if fmt is not None and fmt != "BF16":
            joint = None
            compressed = _quantize_2d(
                param.detach().float(), fmt,
                nvfp4_global_real_override=joint,
                linear_name=recipe_key,
            )
            for suffix, t in compressed.items():
                base_name = (full_qname[:-len(".weight")]
                             if full_qname.endswith(".weight")
                             else full_qname)
                out_key = (base_name
                           if suffix == "weight"
                           else f"{base_name}.{suffix}")
                out[out_key] = t.cpu()
            hist[("head", fmt)] += 1
        else:
            out[full_qname] = param.detach().to(torch.bfloat16).cpu()
            hist[("head_passthrough", "BF16")] += 1

    for name, p in model.named_parameters():
        if p.is_meta:
            continue  # only head/embed/norm/lm_head resident here
        _emit_head_param(name, p)

    for mod_name, mod in model.named_modules():
        non_persistent = getattr(mod, "_non_persistent_buffers_set", set())
        for buf_name, buf in mod.named_buffers(recurse=False):
            if buf_name in non_persistent:
                continue
            if buf.is_meta:
                continue
            full = f"{mod_name}.{buf_name}" if mod_name else buf_name
            if full in out:
                continue
            out[full] = buf.detach().to(torch.bfloat16).cpu()
            hist[("head_buffer", "BF16")] += 1
    print(f"[export-stream] head+embed+norm+lm_head passthrough: "
          f"{time.time()-t_head:.1f}s  keys={len(out)}", flush=True)
    if tensor_sink is not None:
        tensor_sink(out)
        out = {}

    # ----- 3. Per-layer streaming quantize loop -----
    t_layers = time.time()
    for L in range(num_layers):
        if tensor_sink is not None:
            out = {}
        layer_t0 = time.time()
        layer_qname = f"{layers_prefix}{L}".rstrip(".")
        if layer_qname.endswith("."):
            layer_qname = layer_qname[:-1]

        # 3a. Load layer from safetensors (direct to device). When
        # `fp8_scale_inv_map` is non-empty, the loader applies the
        # 128x128 block dequant inline, so `mod.weight` receives the
        # true dequanted weight rather than raw fp8 codes cast to bf16.
        load_t0 = time.time()
        tensors = _read_layer_to_device(
            f"{layers_prefix}{L}.", weight_shard, weight_ckpt, dtype, device,
            fp8_scale_inv_map=fp8_scale_inv_map)
        resolver = _build_install_resolver(model, layer_qname)
        _fast_install(resolver, tensors, device, model=model)
        load_s = time.time() - load_t0

        layer_mod = model.get_submodule(layer_qname)

        # 3b. Proper-AWQ fold pass — modifies predecessor RMSNorm γ
        # AND every reader's weight (nn.Linear + packed experts)
        # IN-PLACE so the matmul identity `(W*s) @ (γ/s · x) = W·γ·x`
        # holds at runtime regardless of each reader's assigned format.
        # Must run BEFORE the fused-sibling joint NVFP4 pass and BEFORE
        # any `_quantize_2d` call so downstream passes see post-AWQ
        # weights. Returned dict maps NVFP4 Linear recipe_keys → `s`,
        # used only for dividing cached activations in GPTQ / act-
        # weighted rounding (runtime sees `a/s` after γ-fold, so the
        # error-minimization passes must too).
        global _AWQ_PROPER_SCALES
        if (_ACT_AWARE_FLAGS.get("awq")
                and _CACHED_ACTIVATIONS is not None):
            layer_scales = _awq_fold_layer_predecessors(
                layer_mod, layer_qname, assignment, profile,
                _CACHED_ACTIVATIONS, device,
            )
            _AWQ_PROPER_SCALES.update(layer_scales)
        else:
            layer_scales = {}

        # 3b'. Joint NVFP4 scales across fused siblings in this layer.
        # Proper-AWQ pre-scaling has already been applied in-place by
        # `_awq_fold_layer_predecessors`, so `mod.weight` is the post-
        # AWQ weight. `_compute_layer_joint_nvfp4` reads those
        # weights directly — no separate awq_scales kwarg needed.
        joint_globals = _compute_layer_joint_nvfp4(
            layer_mod, layer_qname, assignment, profile,
        )

        # 3c. Emit Linears.
        covered: set[str] = set()
        linear_count = 0
        grouped_linears: dict[
            tuple[str, tuple[int, int]],
            list[tuple[str, str, str, nn.Linear]]  # (full, emit_full, recipe_key, mod)
        ] = defaultdict(list)
        for sub_name, mod in layer_mod.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            linear_count += 1
            full = f"{layer_qname}.{sub_name}"

            # Prune-aware routing: resolve once per Linear. Actions:
            #   "router"  → shrink output dim to kept experts + BF16-emit
            #   "drop"    → pruned expert; do not emit this Linear at all
            #   "reindex" → kept expert; rewrite qname eid and continue
            #               through the normal emit path
            prune_action = _resolve_linear_prune_action(full, prune_by_parent)
            if prune_action is not None:
                kind, p_entry = prune_action
                if kind == "router":
                    if not mod.weight.is_meta:
                        w_shrunk = _shrink_router_weight(mod, p_entry)
                        out[f"{full}.weight"] = w_shrunk.to(torch.bfloat16).cpu()
                        if mod.bias is not None and not mod.bias.is_meta:
                            b_idx = torch.as_tensor(
                                p_entry["kept_expert_ids"], dtype=torch.long,
                                device=mod.bias.device,
                            )
                            out[f"{full}.bias"] = (
                                mod.bias.detach().index_select(0, b_idx)
                                .to(torch.bfloat16).cpu()
                            )
                        hist[("linear", "BF16_router_shrunk")] += 1
                        covered.add(full)
                    continue
                if kind == "drop":
                    # Skip pruned expert entirely. Mark covered so the
                    # residual-params loop doesn't re-emit its bias as
                    # a leftover buffer.
                    hist[("linear", "PRUNED")] += 1
                    covered.add(full)
                    continue
                # kind == "reindex": emit under the reindexed qname.
                emit_full = p_entry["new_full"]
            else:
                emit_full = full

            recipe_key = profile.live_to_recipe_name(full)
            fmt = assignment.get(recipe_key)
            if fmt is not None:
                fmt = _canonical_export_format(fmt)
            if fmt is None:
                # No assignment → BF16 passthrough.
                if not mod.weight.is_meta:
                    out[f"{emit_full}.weight"] = mod.weight.detach().to(torch.bfloat16).cpu()
                    if mod.bias is not None and not mod.bias.is_meta:
                        out[f"{emit_full}.bias"] = mod.bias.detach().to(torch.bfloat16).cpu()
                    hist[("linear", "BF16")] += 1
                    covered.add(full)
                continue

            if fmt == "BF16" or recipe_key in bf16_passthrough:
                out[f"{emit_full}.weight"] = mod.weight.detach().to(torch.bfloat16).cpu()
                if mod.bias is not None:
                    out[f"{emit_full}.bias"] = mod.bias.detach().to(torch.bfloat16).cpu()
                hist[("linear", "BF16")] += 1
                covered.add(full)
                continue

            if fmt == "FP8_SOURCE":
                # Passthrough: copy source `.weight` (fp8_e4m3fn) and
                # `.weight_scale_inv` (fp32, 128×128 block) verbatim.
                # The live model holds a BF16 dequant of the source
                # tensor — skip it and go back to the safetensors.
                scale_entry = fp8_source_map.get(full)
                weight_ckpt_key = weight_ckpt.get(f"{full}.weight")
                weight_shard_path = weight_shard.get(f"{full}.weight")
                if (scale_entry is None or weight_ckpt_key is None
                        or weight_shard_path is None):
                    raise RuntimeError(
                        f"[export-stream] FP8_SOURCE assigned to {full} "
                        f"but source is missing `.weight_scale_inv` "
                        f"(scale={scale_entry}, weight_shard="
                        f"{weight_shard_path}). The allocator's "
                        f"passthrough-integrity filter should have "
                        f"prevented this — source manifest is out of "
                        f"sync with the actual checkpoint.")
                scale_shard, scale_ckpt_key = scale_entry
                from safetensors import safe_open
                with safe_open(weight_shard_path, framework="pt") as sf:
                    w_fp8 = sf.get_tensor(weight_ckpt_key)
                    # Common case: scale lives in the same shard. Avoid
                    # a second `safe_open` when we can satisfy both
                    # reads from one file handle.
                    if scale_shard == weight_shard_path:
                        w_scale = sf.get_tensor(scale_ckpt_key)
                    else:
                        w_scale = None
                if w_scale is None:
                    with safe_open(scale_shard, framework="pt") as sf:
                        w_scale = sf.get_tensor(scale_ckpt_key)
                # Sanity check: source dtype must be fp8_e4m3fn; scale
                # must be fp32. Any deviation means the FP8_SOURCE
                # format is being misapplied.
                if w_fp8.dtype != torch.float8_e4m3fn:
                    raise RuntimeError(
                        f"[export-stream] FP8_SOURCE: expected "
                        f"fp8_e4m3fn at {weight_ckpt_key}, got "
                        f"{w_fp8.dtype}")
                out[f"{emit_full}.weight"] = w_fp8.cpu().contiguous()
                out[f"{emit_full}.weight_scale"] = w_scale.to(
                    torch.float32).cpu().contiguous()
                if mod.bias is not None and not mod.bias.is_meta:
                    out[f"{emit_full}.bias"] = mod.bias.detach().to(
                        torch.bfloat16).cpu()
                hist[("linear", "FP8_SOURCE")] += 1
                covered.add(full)
                continue

            if fmt in ("INT2", "INT3", "MXFP8") and mod.weight.dim() == 2:
                shape = (int(mod.weight.shape[0]), int(mod.weight.shape[1]))
                grouped_linears[(fmt, shape)].append((full, emit_full, recipe_key, mod))
                continue

            override = joint_globals.get(recipe_key) if fmt == "NVFP4" else None
            compressed = _quantize_2d(
                mod.weight.detach().float(), fmt,
                nvfp4_global_real_override=override,
                linear_name=recipe_key,
            )
            for suffix, t in compressed.items():
                out[f"{emit_full}.{suffix}"] = t.cpu()
            if mod.bias is not None:
                out[f"{emit_full}.bias"] = mod.bias.detach().to(torch.bfloat16).cpu()
            hist[("linear", fmt)] += 1
            covered.add(full)

        # RTN-only formats can be emitted in same-shape batches. MiniMax has
        # hundreds of expert Linears per layer with identical shapes; doing
        # those one at a time keeps the export CPU/Python-bound even though the
        # math itself is vectorized.
        export_dev = torch.device(device)
        for (fmt, shape), items in grouped_linears.items():
            chunk_len = _export_vector_chunk_len(shape, len(items), export_dev)
            for start in range(0, len(items), chunk_len):
                chunk = items[start:start + chunk_len]
                stacked = torch.stack(
                    [mod.weight.detach().to(torch.float32) for _, _, _, mod in chunk],
                    dim=0,
                )
                compressed_batch = _quantize_2d_group_same_shape(stacked, fmt)
                del stacked
                for i, (full, emit_full, _recipe_key, mod) in enumerate(chunk):
                    for suffix, tensor in compressed_batch.items():
                        piece = tensor[i]
                        if suffix == "weight_global_scale":
                            piece = piece.reshape(1)
                        out[f"{emit_full}.{suffix}"] = piece.cpu()
                    if mod.bias is not None:
                        out[f"{emit_full}.bias"] = mod.bias.detach().to(
                            torch.bfloat16).cpu()
                    hist[("linear", fmt)] += 1
                    covered.add(full)
                del compressed_batch

        # 3d. Emit packed MoE experts, scoped to this layer.
        packed_count = 0
        for sub_name, mod in layer_mod.named_modules():
            if not _is_packed_experts_module(mod):
                continue
            packed_count += 1
            for pn in _packed_experts_param_names(mod):
                experts_qname = (f"{layer_qname}.{sub_name}"
                                 if sub_name else layer_qname)
                full = f"{experts_qname}.{pn}"
                recipe_key = profile.live_to_recipe_name(full)
                fmt = assignment.get(recipe_key)
                if fmt is not None:
                    fmt = _canonical_export_format(fmt)
                if fmt is None:
                    unmapped_keys.append(full)
                    continue
                if fmt == "FP8_SOURCE":
                    raise NotImplementedError(
                        f"[export-stream] FP8_SOURCE not wired for "
                        f"packed-MoE tensors (at {full}). MiniMax-M2/M2.7 "
                        f"— the only natively-FP8 MoE today — uses "
                        f"per-expert `nn.Linear`s, so its experts go "
                        f"through the Linear emit path above, not here. "
                        f"If a new FP8-native MoE arch ships with a "
                        f"packed-expert live module, extend this branch "
                        f"to read per-expert `.weight` + "
                        f"`.weight_scale_inv` from source and emit the "
                        f"per-expert compressed-tensors pairs.")
                packed_param = getattr(mod, pn).detach().float()
                E, M, N = packed_param.shape
                if pn == "gate_up_proj":
                    half = M // 2
                    proj_split = [
                        ("gate_proj", packed_param[:, :half, :]),
                        ("up_proj",   packed_param[:, half:, :]),
                    ]
                else:
                    proj_split = [(pn, packed_param)]

                is_bf16 = fmt == "BF16" or full in bf16_passthrough
                disk_qname = profile.on_disk_expert_qname(experts_qname)
                should_split = profile.split_packed_experts_for_format(fmt)

                # Prune handling for this experts module. If the layer
                # is pruned, `iter_experts` enumerates (orig_eid, new_eid)
                # pairs — only kept experts appear. On the non-pruned
                # path it's `((e, e) for e in range(E))`, preserving
                # exact legacy behavior.
                prune_entry = _resolve_packed_experts_prune(
                    experts_qname, prune_by_parent,
                )
                if prune_entry is not None:
                    if E != int(prune_entry["num_experts_orig"]):
                        raise RuntimeError(
                            f"[export-stream] prune: packed experts at "
                            f"{experts_qname} have E={E} but manifest "
                            f"has num_experts_orig="
                            f"{prune_entry['num_experts_orig']}. "
                            f"Manifest was built against a different "
                            f"model — refusing to emit."
                        )
                    iter_experts = [
                        (int(orig_s), int(new)) for orig_s, new in
                        prune_entry["orig_to_new_eid"].items()
                    ]
                    # Sort by new_eid so the output tensor ordering is
                    # dense 0..K-1 in a predictable order.
                    iter_experts.sort(key=lambda x: x[1])
                else:
                    iter_experts = [(e, e) for e in range(E)]

                if not should_split:
                    if prune_entry is not None:
                        # Keep-packed path after prune: slice the 3D
                        # tensor on dim 0 to kept experts in new-id
                        # order, emit under the same unsliced name.
                        kept_idx = torch.as_tensor(
                            [o for o, _ in iter_experts],
                            dtype=torch.long, device=packed_param.device,
                        )
                        shrunk = packed_param.index_select(0, kept_idx)
                        out[f"{disk_qname}.{pn}"] = shrunk.to(torch.bfloat16).cpu()
                    else:
                        out[f"{disk_qname}.{pn}"] = packed_param.to(torch.bfloat16).cpu()
                    covered.add(full)
                    hist[("packed_moe", "BF16" if is_bf16 else fmt)] += 1
                    del packed_param
                    continue

                # Per-expert joint global scale when NVFP4 splits gate+up.
                per_expert_joint: list[torch.Tensor | None] = [None] * E
                if fmt == "NVFP4" and len(proj_split) > 1:
                    for orig_e, _ in iter_experts:
                        cands = [
                            compute_nvfp4_global_real(sp[orig_e].float(),
                                                      group_size=16)
                            for _, sp in proj_split
                        ]
                        per_expert_joint[orig_e] = torch.stack(cands).max()

                for proj_name, sub_packed in proj_split:
                    for orig_e, new_e in iter_experts:
                        expert_2d = sub_packed[orig_e]
                        base = f"{disk_qname}.{new_e}.{proj_name}"
                        if is_bf16:
                            out[f"{base}.weight"] = expert_2d.to(torch.bfloat16).cpu()
                        else:
                            compressed = _quantize_2d(
                                expert_2d, fmt,
                                nvfp4_global_real_override=per_expert_joint[orig_e],
                            )
                            for suffix, t in compressed.items():
                                key = (base
                                       if suffix == "weight"
                                       else f"{base}.{suffix}")
                                out[key] = t.cpu()
                covered.add(full)
                hist[("packed_moe_per_expert", "BF16" if is_bf16 else fmt)] += 1
                if prune_entry is not None:
                    hist[("packed_moe_pruned", "experts")] += (
                        E - prune_entry["num_experts_kept"]
                    )
                del packed_param, proj_split

        # 3e. Remaining layer-scoped params (norms, conv1d, biases on
        # passthrough-only modules) and persistent buffers.
        for sub_name, param in layer_mod.named_parameters():
            full = f"{layer_qname}.{sub_name}"
            if full in out:
                continue
            if any(full.startswith(c + ".") or full == c for c in covered):
                continue
            if param.is_meta:
                continue
            # Prune skip/reindex: pruned expert sub-params (e.g. a norm
            # inside a dropped expert) must not leak through; kept
            # experts' leftover params must be emitted under their new
            # eid. The Linear path already marked pruned Linears as
            # covered, but non-Linear params inside expert modules need
            # a separate resolve.
            leftover_action = _resolve_linear_prune_action(full, prune_by_parent)
            if leftover_action is not None:
                kind, p_entry = leftover_action
                if kind == "drop":
                    continue
                if kind == "reindex":
                    out[p_entry["new_full"]] = param.detach().to(torch.bfloat16).cpu()
                    hist[("layer_passthrough", "BF16")] += 1
                    continue
            # Router-weight shrink for non-Linear routers. Qwen3.5's
            # `Qwen3_5MoeTopKRouter` is a bare nn.Module with a `.weight`
            # Parameter — NOT an nn.Linear — so the Linear-loop's router
            # path never sees it. Detect by stripping `.weight` and
            # testing against prune_by_parent's router_qname.
            if prune_by_parent and full.endswith(".weight"):
                trimmed = full[: -len(".weight")]
                parent_r = trimmed.rsplit(".", 1)[0]
                entry_r = prune_by_parent.get(parent_r)
                if (entry_r is not None
                        and entry_r["router_qname"] == trimmed
                        and param.dim() >= 1
                        and int(param.shape[0]) == int(entry_r["num_experts_orig"])):
                    idx = torch.as_tensor(
                        entry_r["kept_expert_ids"], dtype=torch.long,
                        device=param.device,
                    )
                    shrunk = param.detach().index_select(0, idx).contiguous()
                    out[full] = shrunk.to(torch.bfloat16).cpu()
                    hist[("router_weight_shrunk", "BF16")] += 1
                    continue
            out[full] = param.detach().to(torch.bfloat16).cpu()
            hist[("layer_passthrough", "BF16")] += 1
        for mod_name, mod in layer_mod.named_modules():
            non_persistent = getattr(mod, "_non_persistent_buffers_set", set())
            for buf_name, buf in mod.named_buffers(recurse=False):
                if buf_name in non_persistent:
                    continue
                full_modpath = (f"{layer_qname}.{mod_name}"
                                if mod_name else layer_qname)
                full = f"{full_modpath}.{buf_name}"
                if full in out or buf.is_meta:
                    continue
                out[full] = buf.detach().to(torch.bfloat16).cpu()
                hist[("layer_buffer", "BF16")] += 1

        # 3f. Unload.
        _unload(model, [f"{layers_prefix}{L}."])
        del tensors, resolver, joint_globals
        # Aggressive GPU cleanup — we've already `.cpu()`'d every
        # quantized output into `out`, so the per-layer GPU working
        # set (fp32 weight copies, grouped/packed intermediates) can
        # be released immediately. Keeps per-layer peak bounded.
        if device.type == "cuda":
            torch.cuda.synchronize()  # ensure outputs are CPU-resident
            torch.cuda.empty_cache()
        if L % 4 == 0:
            gc.collect()
        if L % 4 == 0 or L == num_layers - 1:
            elapsed = time.time() - layer_t0
            print(f"[export-stream] layer {L:02d}  linears={linear_count} "
                  f"packed={packed_count}  load={load_s:.2f}s  "
                  f"total={elapsed:.2f}s  out_keys={len(out)}", flush=True)
        if tensor_sink is not None:
            tensor_sink(out)
            out = {}

    print(f"[export-stream] layer sweep: {time.time()-t_layers:.1f}s",
          flush=True)

    if unmapped_keys:
        print(f"[export-stream] WARN {len(unmapped_keys)} unmapped assignment "
              f"keys — first 5: {unmapped_keys[:5]}", flush=True)

    return out, dict(hist)


def _materialize_tensors_inmemory(
    model: nn.Module,
    assignment: dict[str, str],
    *,
    bf16_passthrough: set[str],
    profile: "ModelProfile | None" = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Whole-model quantizer used for small auxiliary modules (notably the
    MTP wrapper) that fit in RAM. The main decoder export path uses the
    streaming materializer above; this helper exists because MTP is
    built standalone from safetensors and its root module is orders of
    magnitude smaller than the decoder body."""
    from .model_profiles import DefaultProfile
    profile = profile or DefaultProfile()
    remap = profile.live_to_recipe_name

    out: dict[str, torch.Tensor] = {}
    hist = Counter()
    covered: set[str] = set()

    # Pre-pass: joint NVFP4 global_scale per fused-sibling group so
    # q/k/v (or gate/up, etc.) share one weight_global_scale slot.
    nvfp4_joint_global = _compute_nvfp4_joint_global(model, assignment)

    for qname, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        fmt_key = remap(qname)
        fmt = assignment.get(fmt_key)
        if fmt is not None:
            fmt = _canonical_export_format(fmt)
        if fmt is None:
            continue
        if fmt == "BF16" or fmt_key in bf16_passthrough:
            out[f"{qname}.weight"] = mod.weight.detach().to(torch.bfloat16).cpu()
            if mod.bias is not None:
                out[f"{qname}.bias"] = mod.bias.detach().to(torch.bfloat16).cpu()
            covered.add(qname)
            hist[("linear", "BF16")] += 1
            continue
        joint = nvfp4_joint_global.get(fmt_key) if fmt == "NVFP4" else None
        compressed = _quantize_2d(
            mod.weight.detach().float(), fmt,
            nvfp4_global_real_override=joint,
            linear_name=fmt_key,
        )
        for suffix, tensor in compressed.items():
            out[f"{qname}.{suffix}"] = tensor.cpu()
        if mod.bias is not None:
            out[f"{qname}.bias"] = mod.bias.detach().to(torch.bfloat16).cpu()
        covered.add(qname)
        hist[("linear", fmt)] += 1

    for qname, mod in model.named_modules():
        if not _is_packed_experts_module(mod):
            continue
        for pn in _packed_experts_param_names(mod):
            full_name = f"{qname}.{pn}" if qname else pn
            recipe_key = remap(full_name)
            fmt = assignment.get(recipe_key)
            if fmt is not None:
                fmt = _canonical_export_format(fmt)
            if fmt is None:
                continue
            packed_param = getattr(mod, pn).detach().float()
            E, M, N = packed_param.shape
            if pn == "gate_up_proj":
                half = M // 2
                proj_split = [
                    ("gate_proj", packed_param[:, :half, :]),
                    ("up_proj",   packed_param[:, half:, :]),
                ]
            elif pn in ("down_proj", "w1", "w2", "w3", "gate_proj", "up_proj"):
                proj_split = [(pn, packed_param)]
            else:
                proj_split = [(pn, packed_param)]

            is_bf16 = fmt == "BF16" or full_name in bf16_passthrough
            disk_qname = profile.on_disk_expert_qname(qname)
            should_split = profile.split_packed_experts_for_format(fmt)

            if not should_split:
                out[f"{disk_qname}.{pn}"] = packed_param.to(torch.bfloat16).cpu()
                covered.add(full_name)
                hist[("packed_moe", "BF16" if is_bf16 else fmt)] += 1
                continue

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
                E_p, Mp, Np = sub_packed.shape
                for e in range(E_p):
                    expert_2d = sub_packed[e]
                    base = f"{disk_qname}.{e}.{proj_name}"
                    if is_bf16:
                        out[f"{base}.weight"] = expert_2d.to(torch.bfloat16).cpu()
                    else:
                        compressed = _quantize_2d(
                            expert_2d, fmt,
                            nvfp4_global_real_override=per_expert_joint[e],
                        )
                        for suffix, tensor in compressed.items():
                            key = base if suffix == "weight" else f"{base}.{suffix}"
                            out[key] = tensor.cpu()
            covered.add(full_name)
            hist[("packed_moe_per_expert", "BF16" if is_bf16 else fmt)] += 1

    for name, p in model.named_parameters():
        if any(name.startswith(c + ".") or name == c for c in covered):
            continue
        if name in out:
            continue
        out[name] = p.detach().to(torch.bfloat16).cpu()
        hist[("passthrough", "BF16")] += 1

    for mod_name, mod in model.named_modules():
        non_persistent = getattr(mod, "_non_persistent_buffers_set", set())
        for buf_name, buf in mod.named_buffers(recurse=False):
            if buf_name in non_persistent:
                continue
            full = f"{mod_name}.{buf_name}" if mod_name else buf_name
            if any(full.startswith(c + ".") or full == c for c in covered):
                continue
            if full in out:
                continue
            out[full] = buf.detach().to(torch.bfloat16).cpu()
            hist[("passthrough_buffer", "BF16")] += 1

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
MXFP8_SCHEME = {
    "format": "mxfp8-quantized",
    "weights": {
        "num_bits": 8, "type": "float", "strategy": "group",
        "group_size": 32,
        "symmetric": True, "dynamic": False,
        "scale_dtype": "torch.uint8",
        "zp_dtype": "torch.uint8",
        "observer": "memoryless_minmax",
    },
    "input_activations": {
        "num_bits": 8, "type": "float", "strategy": "group",
        "group_size": 32,
        "symmetric": True, "dynamic": True,
        "scale_dtype": "torch.uint8",
        "zp_dtype": "torch.uint8",
    },
}
# INT2 / INT3: symmetric-int weights with the NVFP4 scale envelope
# (group=16, FP8 per-group scale, FP32 per-tensor global). vLLM's
# loader scheme decodes to NVFP4 at
# load time and serves through the existing NVFP4 GEMM kernel. No
# activation quantization (W{2,3}A16) — activations stay BF16.
# Use the generic "pack-quantized" format string so compressed-tensors'
# format whitelist accepts the group. The actual INT decode is picked
# by vLLM's dispatcher based on weight args (type=int, num_bits=2/3,
# strategy=tensor_group, scale_dtype=fp8_e4m3fn).
INT2_SCHEME = {
    "format": "pack-quantized",
    "weights": {
        "num_bits": 2, "type": "int", "strategy": "tensor_group",
        "group_size": 16, "symmetric": True, "dynamic": False,
        "scale_dtype": "torch.float8_e4m3fn",
        "zp_dtype": "torch.float8_e4m3fn",
        "observer": "memoryless_minmax",
    },
}
INT3_SCHEME = {
    "format": "pack-quantized",
    "weights": {
        "num_bits": 3, "type": "int", "strategy": "tensor_group",
        "group_size": 16, "symmetric": True, "dynamic": False,
        "scale_dtype": "torch.float8_e4m3fn",
        "zp_dtype": "torch.float8_e4m3fn",
        "observer": "memoryless_minmax",
    },
}
# Source-FP8 passthrough. Emitted for Linears whose source checkpoint
# already stores `.weight` as fp8_e4m3fn + `.weight_scale_inv` fp32 at
# 128×128 block granularity (MiniMax-M2/M2.7, DeepSeek V3, several
# NVIDIA FP8 releases). vLLM's compressed-tensors dispatcher routes
# this scheme to `_is_fp8_w8a8` which accepts BLOCK-strategy symmetric
# static FP8 weights with dynamic FP8 activations — matching the
# native MiniMax inference configuration.
#
# Compressed-tensors' `weight_scale` (forward-direction dequant scale:
# `w_bf16 = w_fp8 * weight_scale`) is semantically identical to
# MiniMax's `weight_scale_inv`; the tensor bytes are copied verbatim
# and only the suffix is renamed on export. No _quantize_2d pass runs.
FP8_SOURCE_SCHEME = {
    "format": "float-quantized",
    "weights": {
        "num_bits": 8, "type": "float", "strategy": "block",
        "block_structure": [128, 128],
        "symmetric": True, "dynamic": False,
        "observer": "memoryless_minmax",
    },
    # Per-tensor dynamic activation scaling (NOT per-token). vLLM's
    # FP8 MoE path `fp8_w8a8_moe_quant_config` asserts
    # `not per_act_token_quant` whenever weight `block_structure` is
    # set — block-scaled weight + per-token act isn't wired. This
    # matches MiniMax's native-serving `activation_scheme: dynamic`,
    # which is per-tensor dynamic in DeepSeek / MiniMax conventions.
    "input_activations": {
        "num_bits": 8, "type": "float", "strategy": "tensor",
        "symmetric": True, "dynamic": True,
        "observer": "memoryless_minmax",
    },
}
def _bf16_packed_expert_ignore_regex(
        recipe_key: str,
        profile,
) -> list[str]:
    """If `recipe_key` names a BF16 packed-MoE tensor
    (`...experts.gate_up_proj` or `...experts.down_proj`), return one or
    more regex strings that match the corresponding per-expert Linear
    qnames at scheme-dispatch time, so vLLM's `find_matched_target`
    routes them to `ignore` instead of a config_groups target.

    For `gate_up_proj` we emit two patterns (one for `gate_proj`, one
    for `up_proj`) because the packed tensor splits into both at
    materialize time. Returns `[]` if the recipe_key doesn't look
    like a packed-expert entry or the profile has no vLLM class to
    derive naming from."""
    import re as _re

    # Does this recipe key name a packed-expert tensor?
    m = _re.match(r"^(.*\.)(experts)\.(gate_up_proj|down_proj|w\d|gate_proj|up_proj)$",
                  recipe_key)
    if not m:
        return []
    parent = m.group(1)          # `model.layers.X.`  or `model.layers.X.moe.`
    pn = m.group(3)

    # Convert the recipe parent prefix to a live-model prefix by
    # asking the profile. `profile.live_to_recipe_name` is the
    # opposite direction, so we'd need its inverse — instead emit a
    # regex loose enough to match both live forms on both sides of
    # the remap (text-only-style `...layers.X.experts.Y.*` and
    # multimodal `language_model.model.layers.X.moe.experts.Y.*`).
    # The profile's `per_expert_moe_regex` already encodes the live
    # form; we narrow it to this specific layer by pinning the layer
    # index.
    # Distinguish MTP (`mtp.layers.N.*`) from body (`model.layers.N.*`)
    # — both can have layer index N but they're DIFFERENT layers, and
    # emitting a body-prefixed regex for a BF16 MTP assignment
    # accidentally ignores the body's NVFP4 experts at that layer idx.
    is_mtp = recipe_key.startswith("mtp.")
    layer_idx = None
    lm = _re.search(r"\.layers\.(\d+)\.", recipe_key)
    if lm:
        layer_idx = lm.group(1)
    # Build per-proj regex. `gate_up_proj` splits into `gate_proj`
    # and `up_proj` on disk; `down_proj` stays as `down_proj`.
    if pn == "gate_up_proj":
        proj_options = "gate_proj|up_proj"
    elif pn == "down_proj":
        proj_options = "down_proj"
    else:
        proj_options = _re.escape(pn)

    # Use the profile's own regex as the base; swap its `(gate|up|down)_proj`
    # group with the exact projections we emit, and constrain to this
    # layer.
    # MTP layers live under a `mtp.layers.N.*` prefix — separate
    # layer-index namespace from the body. Use the profile's dedicated
    # per_expert_mtp_regex (if any) instead of the body one.
    if is_mtp:
        mtp_base = profile.per_expert_mtp_regex() if profile else None
        if mtp_base and mtp_base.startswith("re:"):
            body = mtp_base[len("re:"):]
            pinned = _re.sub(
                r"layers\[\.\]\[0-9\]\+", f"layers[.]{layer_idx}",
                body, count=1,
            )
            pinned = pinned.replace(
                "(gate|up|down)_proj", f"({proj_options})",
            )
            return [f"re:{pinned}"]
        # Fallback: emit an `mtp.layers.N.*` regex directly.
        if layer_idx is None:
            return []
        return [
            rf"re:^mtp[.]layers[.]{layer_idx}[.]"
            rf"(?:moe[.])?experts[.][0-9]+[.]({proj_options})$"
        ]

    base = profile.per_expert_moe_regex() if profile else None
    if not base or not base.startswith("re:"):
        # No profile regex — emit a conservative default spanning
        # both common live-module conventions.
        patterns = []
        if layer_idx is None:
            return patterns
        # Try the multimodal (Gemma / Qwen3.6) layout first.
        patterns.append(
            rf"re:^language_model[.]model[.]layers[.]{layer_idx}[.]"
            rf"(?:moe[.])?experts[.][0-9]+[.]({proj_options})$"
        )
        # And the text-only / dense layout.
        patterns.append(
            rf"re:^model[.]layers[.]{layer_idx}[.]"
            rf"(?:moe[.])?experts[.][0-9]+[.]({proj_options})$"
        )
        return patterns

    # Profile-provided regex. Strip the `re:` prefix, pin to this
    # layer index, constrain to the emitted projections.
    body = base[len("re:"):]
    # Replace [0-9]+ between layers.X. and .experts. with the specific
    # layer index. Fall back to leaving as-is if the pattern doesn't
    # match our expectations.
    pinned = _re.sub(r"layers\[\.\]\[0-9\]\+", f"layers[.]{layer_idx}", body, count=1)
    # Replace `(gate|up|down)_proj` with only the split projections we
    # actually emitted (so we don't over-ignore).
    pinned = pinned.replace("(gate|up|down)_proj", f"({proj_options})")
    return [f"re:{pinned}"]


FORMAT_SCHEME = {
    "NVFP4": NVFP4_SCHEME,
    "MXFP8": MXFP8_SCHEME,
    "INT2": INT2_SCHEME,
    "INT3": INT3_SCHEME,
    "NVINT2": INT2_SCHEME,
    "NVINT3": INT3_SCHEME,
    "FP8_SOURCE": FP8_SOURCE_SCHEME,
}


def build_quantization_config(
    assignment: dict[str, str],
    bf16_passthrough: set[str],
    extra_ignore: Iterable[str] = (),
    *,
    profile: "ModelProfile | None" = None,
) -> dict:
    """Emit a `quantization_config` dict with explicit per-name targets
    grouped by format. Targets and ignore are remapped to vLLM's
    internal naming via the supplied `profile` so `find_matched_target`
    matches.

    `extra_ignore` is for module qnames that aren't in the recipe at
    all but should be excluded from any catch-all group (e.g. routers).
    The catch-all default group is the format with the most non-BF16
    members (typically NVFP4).

    `profile` controls the architecture-specific bits: name remap,
    per-expert MoE / MTP regexes. Defaults to `DefaultProfile()` (plain
    names, no catch-all regexes) when omitted.
    """
    from .model_profiles import DefaultProfile
    from .model_profiles.vllm_registry import (
        vllm_class_for_architecture, packed_modules_mapping_from_class,
    )
    profile = profile or DefaultProfile()

    by_fmt: dict[str, list[str]] = {}
    ignore: list[str] = []
    for n in bf16_passthrough:
        ignore.append(profile.to_vllm_internal_name(n))
    for n in extra_ignore:
        ignore.append(profile.to_vllm_internal_name(n))
    for name, fmt in sorted(assignment.items()):
        vllm_name = profile.to_vllm_internal_name(name)
        if fmt == "BF16":
            ignore.append(vllm_name)
            # Packed MoE tensors in BF16 are emitted as per-expert
            # per-projection splits (not as the 3D packed tensor). vLLM
            # scheme-dispatches against the per-expert Linear qnames
            # (e.g. `...experts.0.gate_proj`), not the packed parent —
            # so the `ignore` for a BF16 packed-expert recipe entry
            # must cover every per-expert per-projection for that layer.
            # We emit a narrow regex per layer rather than enumerating
            # hundreds of explicit names.
            regex_list = _bf16_packed_expert_ignore_regex(name, profile)
            for r in regex_list:
                ignore.append(r)
            continue
        by_fmt.setdefault(fmt, []).append(vllm_name)

    # Fill in fused-sibling members that exist in the live vLLM
    # model but weren't in the probe assignment — e.g. Gemma 4's
    # full_attention layers have no v_proj on disk, so the probe
    # never saw it, but vLLM's QKVParallelLinear still instantiates
    # a v_proj sub-module that gets k_proj's weights at load. Scheme
    # dispatch requires all fused siblings to have consistent
    # scheme. We infer missing siblings by walking the assignment for
    # fused groups that landed in `ignore` and filling in every
    # sibling from vLLM's `packed_modules_mapping` — including ones
    # we never saw weights for.
    vllm_cls = vllm_class_for_architecture(profile.vllm_architecture_class() or "")
    packed_mapping = packed_modules_mapping_from_class(vllm_cls)
    if packed_mapping:
        # Reverse map: sibling-leaf-name -> fused-name (e.g.
        # q_proj -> qkv_proj).
        leaf_to_fused: dict[str, str] = {}
        for fused_name, siblings in packed_mapping.items():
            for s in siblings:
                leaf_to_fused[s] = fused_name
        # Set of leaf suffixes we should have. We'll only fill in
        # siblings under names that match known fused patterns.
        bf16_name_set = set(ignore)
        for name, fmt in list(assignment.items()):
            if fmt != "BF16":
                continue
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in leaf_to_fused:
                continue
            fused = leaf_to_fused[leaf]
            expected_siblings = packed_mapping[fused]
            parent = name[: -(len(leaf))]
            for sib in expected_siblings:
                full = parent + sib
                vllm_name = profile.to_vllm_internal_name(full)
                if vllm_name not in bf16_name_set:
                    ignore.append(vllm_name)
                    bf16_name_set.add(vllm_name)

    # Packed-3D MoE target emission. vLLM's Qwen3_5/3_6 MoE loads as a
    # single FusedMoE module at qname `<block>.experts` that owns the
    # 3D packed expert tensors internally. Scheme dispatch
    # (`get_moe_method`) probes targets via THREE synthetic layer
    # names built off the FusedMoE prefix:
    #   `<block>.experts.0.gate_proj`
    #   `<block>.experts.0.up_proj`
    #   `<block>.experts.0.down_proj`
    # — this is the "Linear-before-fusion" naming convention, not the
    # packed-tensor qnames (`experts.gate_up_proj`, `experts.down_proj`)
    # we emit in the safetensors. Without matching targets on that
    # per-expert form, no scheme binds to FusedMoE, `w2_input_global_scale`
    # etc. are never registered, and load_weights KeyErrors on our
    # per-expert input scale keys.
    #
    # Fix: for each packed recipe entry under `by_fmt` or `ignore`,
    # replace it with a per-expert regex pinned to that layer index so
    # vLLM's scheme dispatch gets a match on expert 0's projection
    # names. One regex per layer covers all (expert, projection)
    # combinations. `promote_moe_pair` ensures gate_up_proj and
    # down_proj of a single layer share a scheme — we crash loud on
    # mismatch.
    _packed_moe_re = re.compile(r"^(.+\.experts)\.(gate_up_proj|down_proj)$")
    packed_fused_states: dict[str, set[str]] = {}
    for fmt, names in list(by_fmt.items()):
        kept = []
        for vname in names:
            m = _packed_moe_re.match(vname)
            if m:
                packed_fused_states.setdefault(m.group(1), set()).add(fmt)
            else:
                kept.append(vname)
        by_fmt[fmt] = kept
    ignore_kept = []
    for vname in ignore:
        # Preserve regex-prefixed ignores (our
        # _bf16_packed_expert_ignore_regex emits those); they already
        # cover the per-expert forms vLLM dispatches on.
        if vname.startswith("re:"):
            ignore_kept.append(vname)
            continue
        m = _packed_moe_re.match(vname)
        if m:
            packed_fused_states.setdefault(m.group(1), set()).add("IGNORE")
        else:
            ignore_kept.append(vname)
    ignore = ignore_kept

    def _per_expert_regex_for(fused_qname: str) -> str:
        """Regex matching any `<fused_qname>.<eid>.<proj>` where
        proj ∈ {gate_proj, up_proj, down_proj}. Uses `[.]` (not `\\.`)
        for literal-dot escapes, matching the rest of this file's
        regex-target style."""
        escaped = fused_qname.replace(".", "[.]")
        return (
            f"re:^{escaped}[.][0-9]+[.](gate_proj|up_proj|down_proj)$"
        )

    for fused_qname, states in packed_fused_states.items():
        if len(states) > 1:
            raise RuntimeError(
                f"[export-stream] FusedMoE at {fused_qname!r} has mixed "
                f"states across projections {states}; promote_moe_pair "
                f"should have forced gate_up_proj and down_proj to share "
                f"a scheme before this point."
            )
        state = next(iter(states))
        regex = _per_expert_regex_for(fused_qname)
        if state == "IGNORE":
            ignore.append(regex)
        else:
            by_fmt.setdefault(state, []).append(regex)

    # Fused-linear target emission. vLLM's model-loading time fuses
    # siblings from `packed_modules_mapping` into a single packed Linear
    # (e.g. Qwen3.5 DeltaNet's `in_proj_qkv + in_proj_z → in_proj_qkvz`,
    # standard `q_proj + k_proj + v_proj → qkv_proj`). Scheme dispatch
    # keys off the FUSED module's prefix, so our config must list that
    # fused name alongside the siblings. When all expected siblings
    # share one format, emit the fused name into that format's target
    # list; when all land in ignore, emit the fused name into ignore.
    # Mixed-format fused groups are blocked upstream by the allocator's
    # `fused_sibling_group` pre-pass — but we defensively skip emitting
    # a fused target in that case rather than guess.
    if packed_mapping:
        # Map leaf sibling → fused-name, using packed_mapping that vLLM
        # reads at load time.
        leaf_to_fused = {s: fused for fused, sibs in packed_mapping.items()
                         for s in sibs}

        # Build parent-path → {leaf: (fmt|IGNORE, vllm_name)} for every
        # live entry (assignment + extra_ignore + bf16_passthrough).
        def _parent_leaf(vname: str):
            parts = vname.rsplit(".", 1)
            if len(parts) != 2:
                return None, vname
            return parts[0], parts[1]

        # (parent, leaf) → (fmt or "IGNORE")
        leaf_state: dict[tuple[str, str], str] = {}
        for fmt, names in by_fmt.items():
            for vname in names:
                parent, leaf = _parent_leaf(vname)
                if parent is None:
                    continue
                leaf_state[(parent, leaf)] = fmt
        ignore_set = set(ignore)
        for vname in ignore_set:
            parent, leaf = _parent_leaf(vname)
            if parent is None:
                continue
            leaf_state.setdefault((parent, leaf), "IGNORE")

        # For each (parent, fused) pair where all siblings are present
        # and share a state, emit the fused-name target.
        fused_emitted: set[str] = set()
        parents = {p for (p, _) in leaf_state}
        for parent in parents:
            for fused_name, sibs in packed_mapping.items():
                # Skip degenerate fused definitions (single-sibling).
                if len(sibs) < 2:
                    continue
                states = [leaf_state.get((parent, s)) for s in sibs]
                if any(s is None for s in states):
                    continue  # not all siblings present → skip
                if len(set(states)) != 1:
                    continue  # mixed formats → caller's bug; don't emit
                state = states[0]
                fused_vllm_name = f"{parent}.{fused_name}"
                if fused_vllm_name in fused_emitted:
                    continue
                fused_emitted.add(fused_vllm_name)
                if state == "IGNORE":
                    ignore.append(fused_vllm_name)
                else:
                    by_fmt.setdefault(state, []).append(fused_vllm_name)

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
        scheme["targets"] = _build_target_list(names)
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
        # `_build_target_list` collapses per-expert enumerations into
        # compact regexes so a 256-expert / 62-layer MoE emits
        # a few hundred targets instead of ~47k. The profile's
        # per-expert regexes remain as a safety-net for any
        # per-expert Linear not captured by the collapse (e.g.
        # stray experts the recipe didn't enumerate).
        expert_regexes = []
        if (r := profile.per_expert_moe_regex()) is not None:
            expert_regexes.append(r)
        if (r := profile.per_expert_mtp_regex()) is not None:
            expert_regexes.append(r)
        scheme["targets"] = _build_target_list(by_fmt[catchall]) + expert_regexes
        config_groups[f"group_{idx}"] = scheme

    return {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": config_groups,
        "ignore": sorted(set(ignore)),
        "quantization_status": "compressed",
    }


# ---------------------------------------------------------------------------
# Recipe canonicalization + Main
# ---------------------------------------------------------------------------
def _canonicalize_assignment(raw: dict) -> dict[str, str]:
    """Accept either AutoRound-style dicts (`{key: {bits: 4, data_type: nv_fp,
    ...}}`) or shorthand (`{key: "NVFP4"}`). Return `{key: fmt_str}` with
    fmt in {"NVFP4", "MXFP8", "INT2", "INT3", "FP8_SOURCE", "BF16"}."""
    out: dict[str, str] = {}
    for k, v in raw.items():
        name = _strip_weight(k)
        out[name] = canonicalize_format(v)
    return out


# Per-expert siblings map to a fused packed parent at recipe level.
# If the parent IS quantized, the per-expert source keys are already
# covered and must NOT be added to `extra_ignore` — otherwise vLLM's
# compressed-tensors loader marks the FusedMoE layer as un-quantized
# and the NVFP4 scale params (w2_input_global_scale, ...) never get
# registered, crashing at weight-load.
_PER_EXPERT_RE = re.compile(
    r"^(?P<prefix>.+\.experts)\.\d+\.(?P<proj>gate|up|down)_proj$")


def _per_expert_parent(base: str) -> str | None:
    """Map a per-expert source tensor base like
    `model.layers.0.mlp.experts.3.gate_proj` to its packed parent
    `model.layers.0.mlp.experts.gate_up_proj` / `.down_proj`, or None
    if `base` is not a per-expert tensor."""
    m = _PER_EXPERT_RE.match(base)
    if not m:
        return None
    proj = m.group("proj")
    parent = "gate_up_proj" if proj in ("gate", "up") else "down_proj"
    return f"{m.group('prefix')}.{parent}"


def compute_extra_ignore(source_shape_iter, assignment: dict[str, str]
                         ) -> list[str]:
    """Return the list of 2D `.weight` basenames that must be added to
    the compressed-tensors `ignore` set because the recipe doesn't cover
    them.

    `source_shape_iter` yields `(ckpt_key, shape)` for every tensor in
    the source checkpoint (or None for shape when unknown — treated as
    non-2D and skipped). `assignment` maps recipe names to formats.

    Per-expert source keys (e.g. `...experts.3.gate_proj.weight`) are
    NOT added to `extra_ignore` when their packed parent is in the
    assignment — the parent's emitted compressed-tensors scheme already
    covers them at vLLM load time, and adding the per-expert name to
    `ignore` would mark the FusedMoE layer as un-quantized.
    """
    extra_ignore: list[str] = []
    seen_recipe = set(assignment)
    for ckpt_key, shape in source_shape_iter:
        if not ckpt_key.endswith(".weight"):
            continue
        base = ckpt_key[:-7]
        recipe_name = ("model." + base[len("model.language_model."):]
                       if base.startswith("model.language_model.")
                       else base)
        if recipe_name in seen_recipe:
            continue
        parent = _per_expert_parent(recipe_name)
        if parent is not None and parent in seen_recipe:
            continue
        if shape is None or len(shape) != 2:
            continue
        extra_ignore.append(base)
    return extra_ignore


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="HF model dir (source safetensors + config.json)")
    ap.add_argument("--layer-config", required=True,
                    help="layer_config.json from allocator.py")
    ap.add_argument("--prune-manifest", default=None,
                    help="Optional path to an expert-prune sidecar JSON "
                         "emitted by the allocator at "
                         "`<layer_config>.prune.json`. When omitted, the "
                         "exporter auto-detects that path and uses it if "
                         "it exists; pass an empty string to force a "
                         "non-prune export even when a sidecar is "
                         "present. A non-empty manifest drops pruned "
                         "experts' weights, reindexes kept experts to "
                         "dense 0..K-1, shrinks the router weight's "
                         "out-dim, and updates config.json's expert "
                         "count fields.")
    ap.add_argument("--output", required=True,
                    help="Output directory for the compressed checkpoint")
    ap.add_argument("--shard-bytes", type=int, default=5 * 1024**3,
                    help="Approx per-shard size in bytes (default 5 GiB)")
    ap.add_argument("--device", default="cuda",
                    help="Device for quantization arithmetic. Layer "
                         "weights are read into this device; "
                         "_quantize_2d / _quantize_3d_packed run here; "
                         "outputs are moved to CPU before storage.")
    ap.add_argument("--offload-folder", default=None,
                    help="Accelerate disk-offload folder (defaults to "
                         "sibling of output).")
    ap.add_argument("--ignore", nargs="*", default=["lm_head"],
                    help="Module qnames to keep at bf16 even if the "
                         "allocator assigned another format. "
                         "lm_head is ignored by default because vLLM's "
                         "ParallelLMHead module only accepts a single "
                         "`weight` parameter — it does not support the "
                         "compressed-tensors NVFP4/MXFP8 layout "
                         "(weight_packed + weight_scale + global_scales). "
                         "Quantizing lm_head here produces a valid recipe "
                         "but vLLM rejects it at load time with "
                         "'There is no module or parameter named "
                         "lm_head.input_global_scale in "
                         "<ForCausalLM>'. This is a RUNTIME limitation, "
                         "not an allocator choice — the probe + cost "
                         "stages measure lm_head's sensitivity correctly "
                         "and the allocator will happily place NVFP4 for "
                         "it (saving ~0.7 GB on 35B / ~1.1 GB on 122B). "
                         "Remove this default only if you're exporting "
                         "for a runtime that supports quantized "
                         "ParallelLMHead or patches vLLM's registration.")
    ap.add_argument("--activation-cache-dir", default=None,
                    help="Probe's activation cache directory. When "
                         "supplied, per-Linear input_global_scale is "
                         "computed from cached activations "
                         "(max_abs/6.0) instead of the 1.0 default. "
                         "Typically ~1-3% PPL improvement on NVFP4.")
    # Activation-aware passes.
    #
    # AWQ defaults to OFF — per-channel input scaling fights NVFP4's
    # 16-channel group_size: each FP4 group ends up with a mix of
    # scale-boosted (up to 10×) and scale-damped (down to 0.1×) input
    # channels, inflating per-group max-abs and DOUBLING quant error
    # instead of reducing it. Measured PPL on Qwen3.6-35B:
    #     baseline (no act-aware)  4.97
    #     AWQ only                 16.44   (+230%, much worse)
    #     GPTQ only                 4.84   (-2.7%)
    #     act-weighted-round only   4.88   (-1.8%)
    # AWQ was designed for W4A16 per-channel quant where no group
    # structure competes with its rescaling. For group-quant like
    # NVFP4 (or any 8/16-wide group), prefer GPTQ + act-weighted
    # rounding which ARE group-aware. Set --awq explicitly to opt in.
    #
    # GPTQ and --act-weighted-round remain tri-state "auto-on when
    # --activation-cache-dir is supplied" because they measurably help.
    ap.add_argument("--awq", dest="awq", default=None,
                    action=argparse.BooleanOptionalAction,
                    help="AWQ per-input-channel rescale + γ-fold. OFF "
                         "by default — incompatible with NVFP4 "
                         "group_size=16 (see source comment). Pass "
                         "--awq to opt in.")
    ap.add_argument("--gptq", dest="gptq", default=None,
                    action=argparse.BooleanOptionalAction,
                    help="GPTQ one-shot OBS rounding with block-wise "
                         "error propagation (NVFP4 only; skipped on "
                         "MXFP8). Auto-on when --activation-cache-dir "
                         "is supplied. Measured -2.7% PPL on Qwen3.6-35B.")
    ap.add_argument("--act-weighted-round", dest="awq_round", default=None,
                    action=argparse.BooleanOptionalAction,
                    help="Activation-weighted rounding polish on NVFP4 "
                         "(per-weight Δw²·E[a²] minimization at fixed "
                         "group scale). OFF by default — permutation "
                         "bake-off showed it undoes most of GPTQ's "
                         "benefit (geomean out_mse ratio: GPTQ=0.41, "
                         "GPTQ+act_round=0.99 ≈ RTN). Pass "
                         "--act-weighted-round to opt in.")
    ap.add_argument("--scale-sweep", dest="scale_sweep", default=None,
                    action=argparse.BooleanOptionalAction,
                    help="Per-group 1-D scale sweep with RTN rounding on "
                         "NVFP4 — closed-form analog of AutoRound's SGD. "
                         "Auto-on when --activation-cache-dir is supplied. "
                         "Measured best-in-bake-off when composed after "
                         "GPTQ: geomean out_mse ratio = 0.33 vs GPTQ-only "
                         "0.41 vs RTN 1.0, on Qwen3.6-35B visual+MTP "
                         "Linears.")
    args = ap.parse_args()

    from .model_profiles import detect_profile
    profile = detect_profile(args.model)
    print(f"[export-stream] model profile: {profile.name}", flush=True)

    # Resolve flag defaults.
    cache_supplied = bool(args.activation_cache_dir)
    # AWQ: OFF unless explicitly requested. Incompatible with NVFP4
    # group_size=16 (see long comment on the argparse definition).
    awq_enabled = bool(args.awq) if args.awq is not None else False
    # GPTQ + scale-sweep: ON iff activation cache supplied.
    gptq_enabled = args.gptq if args.gptq is not None else cache_supplied
    # act_round: OFF by default (bake-off showed it reverts GPTQ to RTN).
    awq_round_enabled = bool(args.awq_round) if args.awq_round is not None else False
    # scale_sweep: ON iff activation cache supplied.
    scale_sweep_enabled = (args.scale_sweep if args.scale_sweep is not None
                           else cache_supplied)
    act_passes_any = (awq_enabled or gptq_enabled or awq_round_enabled
                      or scale_sweep_enabled)
    # The activation-aware passes need the actual activations, not just
    # the scale summary. We only load raw activations when at least one
    # pass is enabled.
    if act_passes_any and not cache_supplied:
        print("[export-stream] WARN activation-aware passes requested "
              "but no --activation-cache-dir; disabling.", flush=True)
        awq_enabled = gptq_enabled = awq_round_enabled = False
        scale_sweep_enabled = False
        act_passes_any = False
    print(f"[export-stream] act-aware passes: awq={awq_enabled} "
          f"gptq={gptq_enabled} awq_round={awq_round_enabled} "
          f"scale_sweep={scale_sweep_enabled}", flush=True)
    # Publish to the module-level config so `_quantize_2d` picks them
    # up from every call site without needing the flags threaded
    # through `materialize_tensors_streaming` + MTP helpers.
    _ACT_AWARE_FLAGS["awq"] = awq_enabled
    _ACT_AWARE_FLAGS["gptq"] = gptq_enabled
    _ACT_AWARE_FLAGS["awq_round"] = awq_round_enabled
    _ACT_AWARE_FLAGS["scale_sweep"] = scale_sweep_enabled

    # Populate the module-level input-global-scale cache (used by
    # `_quantize_2d` for NVFP4 linears) from cached activations.
    # Same cache is reused to populate _CACHED_ACTIVATIONS when any
    # act-aware pass is enabled.
    if args.activation_cache_dir:
        from .measure_quant_cost import ActivationIndex
        global _INPUT_GLOBAL_SCALES, _CACHED_ACTIVATIONS
        cache_dir = Path(args.activation_cache_dir)
        if not cache_dir.exists():
            print(f"[export-stream] WARN activation cache dir {cache_dir} "
                  f"missing; input_global_scale falls back to "
                  f"{DEFAULT_INPUT_GLOBAL_SCALE}", flush=True)
        else:
            # Pull candidate names from the recipe — ActivationIndex
            # only loads for names that actually have a cached file.
            with open(args.layer_config) as _lc:
                _recipe_payload = json.load(_lc)
            validate_layer_config_payload(_recipe_payload, args.layer_config)
            _recipe_names = list(_recipe_payload.keys())
            idx = ActivationIndex(cache_dir, _recipe_names)
            scales: dict[str, float] = {}
            for name in idx.names():
                try:
                    acts = idx.load(name)
                    scales[name] = compute_nvfp4_input_global_scale(acts)
                except Exception as e:
                    print(f"[export-stream] WARN could not load "
                          f"activations for {name}: {e}", flush=True)
            # Unify input_global_scale across fused-sibling groups.
            # vLLM's fused Linear loader concatenates q/k/v (and gate/up)
            # into a single tensor and applies ONE input scale at
            # forward time. If q/k/v scales differ the warning
            #   "global scale for input or weight are different for
            #    parallel layers (e.g. q_proj, k_proj, v_proj). This
            #    will likely result in reduced accuracy."
            # fires at vLLM load. q/k/v siblings receive the same
            # upstream activation in principle, but captured per-
            # Linear from different shard subsamples, so the computed
            # max/6 values can drift by a float-precision tick. Take
            # the max over the group so vLLM runs on the conservative
            # (larger) scale for every sibling.
            scales = _unify_input_global_scales_across_fused_siblings(scales)
            _INPUT_GLOBAL_SCALES = scales
            if act_passes_any:
                _CACHED_ACTIVATIONS = _LazyActivationCache(idx)
                print(f"[export-stream] raw activations will be loaded "
                      f"lazily for AWQ/GPTQ/round passes "
                      f"({len(idx)}/{len(_recipe_names)} Linears indexed)",
                      flush=True)
            print(f"[export-stream] input_global_scale calibrated for "
                  f"{len(scales)}/{len(_recipe_names)} Linears from "
                  f"{cache_dir}", flush=True)

    with open(args.layer_config) as f:
        raw_recipe = json.load(f)
    validate_layer_config_payload(raw_recipe, args.layer_config)
    assignment = _canonicalize_assignment(raw_recipe)
    validate_mtp_assignment_coverage(args.model, assignment, profile)
    fmts = Counter(assignment.values())
    print(f"[export-stream] recipe: {len(assignment)} entries  mix={dict(fmts)}",
          flush=True)

    # Prune manifest: explicit path (empty string = opt-out), else
    # auto-discover sidecar next to layer_config.json.
    if args.prune_manifest is None:
        default_sidecar = Path(args.layer_config + ".prune.json")
        prune_manifest = _load_prune_manifest(
            default_sidecar if default_sidecar.exists() else None
        )
    elif args.prune_manifest == "":
        prune_manifest = {}
    else:
        prune_manifest = _load_prune_manifest(args.prune_manifest)

    dtype = torch.bfloat16
    device = torch.device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    bf16_passthrough = set(args.ignore)
    if args.offload_folder is None:
        args.offload_folder = str(out_dir / "_streaming_offload")

    # Rename body keys → `model.language_model.` on disk for multimodal-
    # umbrella arches (Qwen3.5/3.6 ConditionalGeneration, Gemma 4
    # ConditionalGeneration). Our streaming loop produces the text-only
    # `model.layers.X.*` form.
    body_infix = getattr(profile, "body_ondisk_infix", None)
    if callable(body_infix):
        infix = body_infix()
    else:
        # Default: Qwen3.5/3.6 pattern. Profiles for non-multimodal
        # archs can return "" and we'll skip the rename.
        infix = "language_model." if profile.name.startswith("qwen3_5") else ""

    def _rename_body_batch(
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not infix:
            return batch
        renamed: dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if (k.startswith("model.layers.")
                    or k.startswith("model.embed_tokens")
                    or k.startswith("model.norm")):
                renamed[f"model.{infix}{k[len('model.'):]}"] = v
            else:
                renamed[k] = v
        return renamed

    writer = IncrementalSafetensorsWriter(out_dir, args.shard_bytes)
    if infix:
        print(f"[export-stream] streaming body rename → model.{infix}...",
              flush=True)

    tensors, hist = materialize_tensors_streaming(
        args.model, assignment,
        profile=profile, bf16_passthrough=bf16_passthrough,
        dtype=dtype, device=device,
        offload_folder=args.offload_folder,
        prune_manifest=prune_manifest,
        tensor_sink=lambda batch: writer.add_tensors(_rename_body_batch(batch)),
    )
    print(f"[export-stream] streamed materialization complete "
          f"resident_tensors={len(tensors)}  hist={hist}",
          flush=True)

    # MTP materialization if the profile has heads. Uses the in-memory
    # helper — MTP heads are small enough that full-model residency
    # isn't a concern.
    mtp_tensors: dict[str, torch.Tensor] = {}
    if profile.has_mtp():
        print("[export-stream] materializing MTP tensors ...", flush=True)
        mtp_tensors = _materialize_mtp_tensors(
            args.model, assignment,
            bf16_passthrough=bf16_passthrough, hist=hist,
            device=device)
        print(f"[export-stream] MTP: {len(mtp_tensors)} tensors", flush=True)
    else:
        print(f"[export-stream] profile '{profile.name}' has no MTP — "
              "skipping", flush=True)

    # Merge source passthrough (visual/audio towers etc.) that aren't
    # part of our streaming pass. Drop entries that MTP materialize
    # already covered.
    passthrough_prefixes = tuple(profile.source_passthrough_prefixes())
    if passthrough_prefixes:
        src_extra = _load_source_passthrough(
            args.model, prefix_filters=passthrough_prefixes)
        materialized_bases: set[str] = set()
        for k in mtp_tensors:
            base = k
            for suf in (".weight_packed", ".weight_scale",
                        ".weight_global_scale", ".input_global_scale",
                        ".weight"):
                if k.endswith(suf):
                    base = k[:-len(suf)] + ".weight"
                    break
            materialized_bases.add(base)
            m = re.match(r"^(mtp\.layers\.\d+\.mlp\.experts)\.\d+\.(gate|up|down)_proj\.", k)
            if m:
                if m.group(2) in ("gate", "up"):
                    materialized_bases.add(f"{m.group(1)}.gate_up_proj")
                else:
                    materialized_bases.add(f"{m.group(1)}.down_proj")
        src_extra = {k: v for k, v in src_extra.items()
                     if k not in materialized_bases}
        for k in list(src_extra.keys()):
            if k in writer.seen_keys or k in mtp_tensors:
                del src_extra[k]

        # Phase 1 visual-encoder quant: when the allocator's recipe
        # assigns a non-BF16 format to a visual Linear, run its 2D
        # weight through `_quantize_2d` before emit. BF16 entries and
        # non-Linear tensors (norms, conv1d, biases, buffers) pass
        # through unchanged. See allocator's `--visual-format` docstring
        # for why this is a uniform override rather than a per-Linear
        # decision — text-only probe never exercises the visual tower.
        src_extra = _apply_visual_recipe_quant(
            src_extra, assignment, device=device)

        writer.add_tensors(mtp_tensors)
        writer.add_tensors(src_extra)
        print(f"[export-stream] merged {len(src_extra)} source-passthrough + "
              f"{len(mtp_tensors)} MTP tensors", flush=True)
    else:
        writer.add_tensors(mtp_tensors)

    print("[export-stream] finalizing safetensors shards ...", flush=True)
    t_write = time.time()
    writer.finalize()
    print(f"[export-stream] sharded write: {time.time()-t_write:.1f}s",
          flush=True)

    # Scan source safetensors for 2D `.weight` keys not covered by the
    # recipe — these are visual encoder / unmapped Linears that vLLM
    # instantiates during model-construction time. Without an explicit
    # ignore entry, compressed-tensors' `find_matched_target` raises
    # `ValueError: Unable to find matching target for visual.merger.*`.
    src_dir = Path(args.model)

    def _source_shape_iter():
        if not src_dir.exists():
            return
        from safetensors import safe_open
        import os as _os
        for f in sorted(_os.listdir(src_dir)):
            if not f.endswith(".safetensors"):
                continue
            with safe_open(str(src_dir / f), framework="pt") as sf:
                for k in sf.keys():
                    try:
                        shape = list(sf.get_slice(k).get_shape())
                    except Exception:
                        shape = None
                    yield k, shape

    extra_ignore = compute_extra_ignore(_source_shape_iter(), assignment)
    print(f"[export-stream] extra ignore (unmapped Linears): "
          f"{len(extra_ignore)}", flush=True)

    write_config_with_quantization(
        args.model, out_dir, assignment, bf16_passthrough,
        extra_ignore=extra_ignore,
        prune_manifest=prune_manifest)
    _copy_tokenizer(args.model, out_dir)

    prune_summary: dict | None = None
    if prune_manifest:
        prune_summary = {
            "n_layers_pruned": len(prune_manifest),
            "n_experts_orig_total": sum(
                int(e["num_experts_orig"]) for e in prune_manifest.values()
            ),
            "n_experts_kept_total": sum(
                int(e["num_experts_kept"]) for e in prune_manifest.values()
            ),
            "manifest_file": "prune_manifest.json",
        }
        # Also persist the manifest into the output dir for traceability
        # (the validator + any downstream re-export tooling can read it).
        with open(out_dir / "prune_manifest.json", "w") as f:
            json.dump(prune_manifest, f, indent=2, sort_keys=True)

    with open(out_dir / "mixed_native_manifest.json", "w") as f:
        json.dump({
            "source_model": args.model,
            "source_recipe": args.layer_config,
            "format_histogram": {f"{k[0]}/{k[1]}": v for k, v in hist.items()},
            "n_assignment_entries": len(assignment),
            "ignore": sorted(bf16_passthrough),
            "prune": prune_summary,
        }, f, indent=2)

    print(f"[export-stream] done. Serve with:\n"
          f"  vllm serve {out_dir.resolve()} --quantization compressed-tensors",
          flush=True)


# ---------------------------------------------------------------------------
# Sharded safetensors writer (mirrors HF transformers' shard layout so
# the index file is the same one transformers + vLLM expect).
# ---------------------------------------------------------------------------
def _clone_shared_storage_for_safetensors(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return a save-ready dict without same-file shared storage ties."""
    out = dict(tensors)
    seen_storage: dict[int, str] = {}
    for k, t in list(out.items()):
        try:
            sid = t.untyped_storage().data_ptr()
        except Exception:
            continue
        if sid in seen_storage:
            # This tensor shares storage with an earlier one. Deep-copy
            # so safetensors treats them independently.
            out[k] = t.detach().clone().contiguous()
        else:
            seen_storage[sid] = k
    return out


class IncrementalSafetensorsWriter:
    """Write HF-style safetensor shards while batches are produced.

    The legacy writer receives the entire tensor dict and therefore needs
    enough host RAM for the full compressed checkpoint. Large MoE exports
    can exceed that before the final write phase. This writer keeps only
    one output shard resident, writes temporary shard files as soon as
    they reach the byte budget, then renames them to the final
    `model-00001-of-000NN.safetensors` layout and writes the index once
    the final shard count is known.
    """

    def __init__(self, out_dir: Path, shard_bytes: int):
        self.out_dir = out_dir
        self.shard_bytes = int(shard_bytes)
        self.current: dict[str, torch.Tensor] = {}
        self.current_size = 0
        self.total_size = 0
        self.tmp_shards: list[tuple[Path, list[str]]] = []
        self.weight_map: dict[str, str] = {}
        self.seen_keys: set[str] = set()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _tensor_size(t: torch.Tensor) -> int:
        return int(t.numel() * t.element_size())

    def add_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        if not tensors:
            return
        for key in sorted(tensors):
            if key in self.seen_keys:
                raise RuntimeError(
                    f"duplicate tensor key emitted during export: {key}"
                )
            tensor = tensors[key].detach().cpu()
            size = self._tensor_size(tensor)
            if (self.current
                    and self.current_size + size > self.shard_bytes):
                self._flush_current()
            self.current[key] = tensor
            self.current_size += size
            self.total_size += size
            self.seen_keys.add(key)
            # A single tensor can exceed the target shard size. Flush it
            # immediately so the next shard starts cleanly.
            if self.current_size >= self.shard_bytes:
                self._flush_current()

    def _flush_current(self) -> None:
        if not self.current:
            return
        idx = len(self.tmp_shards) + 1
        tmp_path = self.out_dir / f".model-{idx:05d}.safetensors.tmp"
        save_file(
            {k: v.contiguous() for k, v in
             _clone_shared_storage_for_safetensors(self.current).items()},
            str(tmp_path),
            metadata={"format": "pt"},
        )
        self.tmp_shards.append((tmp_path, list(self.current.keys())))
        print(
            f"[export-stream] wrote temp shard {idx:05d} "
            f"keys={len(self.current)} bytes={self.current_size}",
            flush=True,
        )
        self.current = {}
        self.current_size = 0
        gc.collect()

    def finalize(self) -> None:
        self._flush_current()
        if not self.tmp_shards:
            raise RuntimeError("no tensors were written")

        if len(self.tmp_shards) == 1:
            tmp_path, keys = self.tmp_shards[0]
            final_name = "model.safetensors"
            os.replace(tmp_path, self.out_dir / final_name)
            for key in keys:
                self.weight_map[key] = final_name
            print("[export-stream] finalized single safetensors shard",
                  flush=True)
            return

        n = len(self.tmp_shards)
        for i, (tmp_path, keys) in enumerate(self.tmp_shards, start=1):
            final_name = f"model-{i:05d}-of-{n:05d}.safetensors"
            os.replace(tmp_path, self.out_dir / final_name)
            for key in keys:
                self.weight_map[key] = final_name

        with open(self.out_dir / "model.safetensors.index.json", "w") as f:
            json.dump({
                "metadata": {"total_size": self.total_size},
                "weight_map": self.weight_map,
            }, f, indent=2)
        print(f"[export-stream] finalized {n} safetensors shards",
              flush=True)


def write_sharded_safetensors(
    tensors: dict[str, torch.Tensor],
    out_dir: Path,
    shard_bytes: int,
) -> None:
    # Detach + clone any tensors that share underlying storage so
    # safetensors' dedup check doesn't raise. This covers tied
    # embeddings (Gemma 4: `lm_head.weight` ≡ `embed_tokens.weight`)
    # and any other view-ties produced by HF's
    # `_tied_weights_keys`. Cost: one extra copy of the embed matrix;
    # correctness: identical bytes on disk, no runtime semantic change.
    tensors = _clone_shared_storage_for_safetensors(tensors)

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
    prune_manifest: dict[str, dict] | None = None,
) -> None:
    from .model_profiles import detect_profile
    profile = detect_profile(src_model)
    src_cfg_path = Path(src_model) / "config.json"
    cfg = json.load(open(src_cfg_path))
    qc = build_quantization_config(assignment, bf16_passthrough,
                                   extra_ignore, profile=profile)
    if qc:
        cfg["quantization_config"] = qc

    # Prune: shrink MoE expert counts in the config so vLLM / HF
    # instantiate a ModuleList of the right size. We update every
    # common HF field name that exists in the source config, so the
    # loader finds the expected field regardless of arch convention.
    # All manifest entries must agree on num_experts_kept (same arch)
    # — mixing kept counts across layers isn't supported by the
    # shared-config convention.
    if prune_manifest:
        kept_counts = {int(e["num_experts_kept"]) for e in prune_manifest.values()}
        if len(kept_counts) != 1:
            raise RuntimeError(
                f"[export-stream] prune: manifest has inconsistent "
                f"num_experts_kept across layers ({sorted(kept_counts)}). "
                f"HF config carries a single scalar field — mixed "
                f"per-layer counts need a schema change."
            )
        new_k = next(iter(kept_counts))
        patched: list[tuple[str, int, int]] = []

        def _patch_scalar(d: dict) -> None:
            for field in _MOE_EXPERT_COUNT_FIELDS:
                if field in d and isinstance(d[field], int):
                    old = int(d[field])
                    if old != new_k:
                        d[field] = new_k
                        patched.append((field, old, new_k))

        _patch_scalar(cfg)
        # Some multimodal configs nest the text config (e.g. Qwen3.5/3.6
        # ConditionalGeneration). Patch there too if present.
        if isinstance(cfg.get("text_config"), dict):
            _patch_scalar(cfg["text_config"])
        for field, old, new in patched:
            print(f"[export-stream] config: {field} {old} → {new}", flush=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def _materialize_mtp_tensors(src_model: str,
                             assignment: dict[str, str],
                             *,
                             bf16_passthrough: set[str],
                             hist: dict,
                             device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Quantize MTP weights per the allocator recipe.

    Transformers v5 does not instantiate MTP modules when loading
    Qwen3.5/3.6 MoE checkpoints (see `_keys_to_ignore_on_load_unexpected`),
    so the streaming decoder-layer sweep never sees any `mtp.*` entry in
    `assignment`. We build a standalone MTP module, load the source
    `mtp.*` weights into it, wrap it in a parent module named `mtp` (so
    qualified names come out as `mtp.fc`, `mtp.layers.0.self_attn.q_proj`,
    ...), and run the in-memory materialize helper.

    Output tensor names match the checkpoint convention (`mtp.fc.*`,
    `mtp.layers.0.<rest>`). vLLM's `qwen3_5_mtp.load_weights` remaps
    `mtp.→model.` at load time.
    """
    from .mtp_module import MtpModule, _load_into_mtp, _load_mtp_state_dict
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
    # Move the whole MTP module to the export device so
    # _materialize_tensors_inmemory's per-linear quant runs on GPU when
    # EXPORT_DEVICE=cuda. Previously defaulted to CPU, costing ~10× on
    # MTP quant. The input weights (raw) are CPU, so we move after load.
    wrapper.to(device=device)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    # Filter assignment to just `mtp.*` entries.
    mtp_assignment = {k: v for k, v in assignment.items() if k.startswith("mtp.")}
    if not mtp_assignment:
        return {}

    out, sub_hist = _materialize_tensors_inmemory(
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


_VISUAL_KEY_RE = re.compile(r"^(?:model\.)?visual\.")


def _apply_visual_recipe_quant(
    src_extra: dict[str, torch.Tensor],
    assignment: dict[str, str],
    *,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Rewrite visual-encoder `.weight` entries in `src_extra` under the
    recipe's per-Linear format assignment.

    The allocator's `--visual-format` flag stamps every visual Linear
    with a uniform format (`BF16` | `NVFP4` | `MXFP8`). For BF16 we do
    nothing — the passthrough tensor is already in the right dtype
    (typically bf16 in the source). For NVFP4 / MXFP8 we route the
    rank-2 weight through `_quantize_2d` and replace the single
    `<name>.weight` key with the compressed-tensors tensor set
    (`<name>.weight_packed`, `<name>.weight_scale`,
    `<name>.weight_global_scale`, `<name>.input_global_scale` for NVFP4;
    `<name>.weight`, `<name>.weight_scale` for MXFP8).

    Non-Linear tensors (norms, conv1d, biases, buffers) and visual
    keys WITHOUT a recipe entry are passed through unchanged —
    consistent with the Phase 1 uniform-override contract: only
    Linears discovered by `discover_visual_linears_from_source` end up
    with a recipe entry, and that helper rejects anything that isn't
    rank-2.

    `device` is the compute device for quant arithmetic; outputs are
    moved to CPU before storage so they're ready for the sharded
    safetensors writer.
    """
    out: dict[str, torch.Tensor] = {}
    touched = 0
    for key, tensor in src_extra.items():
        if not key.endswith(".weight"):
            out[key] = tensor
            continue
        if not _VISUAL_KEY_RE.match(key):
            out[key] = tensor
            continue
        base = key[:-len(".weight")]
        fmt = assignment.get(base)
        if fmt is not None:
            fmt = _canonical_export_format(fmt)
        if fmt is None or fmt == "BF16":
            out[key] = tensor
            continue
        if tensor.ndim != 2:
            # Non-2D visual weights aren't Linear modules — skip them.
            out[key] = tensor
            continue
        weight = tensor.to(device=device, dtype=torch.float32)
        try:
            compressed = _quantize_2d(
                weight, fmt,
                nvfp4_global_real_override=None,
                linear_name=base,
            )
        except Exception as e:
            # Fail-safe: fall back to passthrough on any arithmetic
            # error. Better to land a BF16 visual Linear than crash
            # the whole export — the rest of the body/MTP are already
            # materialized.
            print(f"[export-stream] WARN visual quant failed for {base} "
                  f"({fmt}): {e}; falling back to BF16 passthrough",
                  flush=True)
            out[key] = tensor
            continue
        for suffix, t in compressed.items():
            out[f"{base}.{suffix}"] = t.cpu()
        touched += 1
    if touched:
        print(f"[export-stream] quantized {touched} visual Linear(s) "
              f"from recipe", flush=True)
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
    # Custom architecture modules (trust_remote_code). MiniMax-M2 ships
    # `configuration_minimax_m2.py` + `modeling_minimax_m2.py`;
    # DeepSeek-V3 and similar use the same pattern. vLLM's config loader
    # re-reads these via `get_class_from_dynamic_module` when the
    # exported config's `auto_map` still references them, so they must
    # travel with the checkpoint. Copy every `.py` at the source root
    # (there's only ever a handful — the custom modules and occasionally
    # a `modular_*.py` generator; the autogen header warns not to ship
    # both but copying is harmless).
    for py in src.glob("*.py"):
        shutil.copy2(py, out_dir / py.name)


def _source_has_prefixed_weights(src_model: str, prefix: str) -> bool:
    """Return True when the source safetensors index contains any key
    beginning with `prefix`.

    Export-time validation should use the index rather than a loaded HF
    model because transformers intentionally drops `mtp.*` on load for
    Qwen3.5/3.6, which would otherwise make missing recipe coverage look
    benign.
    """
    idx_path = Path(src_model) / "model.safetensors.index.json"
    if not idx_path.exists():
        return False
    with open(idx_path) as f:
        weight_map = json.load(f).get("weight_map", {})
    return any(k.startswith(prefix) for k in weight_map)


def validate_mtp_assignment_coverage(src_model: str,
                                     assignment: dict[str, str],
                                     profile) -> None:
    """Fail fast when an architecture with MTP source weights is being
    exported without any allocator coverage for `mtp.*`.

    Passing raw MTP weights through silently produces a checkpoint that
    looks complete but violates PrismaQuant's intended contract: MTP must
    participate in the same probe/cost/allocation loop as the body. This
    exact state was observed on Qwen3.5-122B where the body artifacts on
    disk were generated without merged MTP probe/cost results.
    """
    if not profile.has_mtp():
        return
    if not _source_has_prefixed_weights(src_model, "mtp."):
        return
    if any(k.startswith("mtp.") for k in assignment):
        return
    raise RuntimeError(
        "source checkpoint contains mtp.* weights but the allocator recipe "
        "contains no mtp.* entries. Re-run the incremental probe + cost "
        "with --include-mtp (the default) so mtp.* tensors are measured, "
        "then rerun allocator/export."
    )


if __name__ == "__main__":
    main()
