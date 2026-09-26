"""Canonical source-value contract for CB cost/cache/export renders."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch


def build_cb_source_fp8_scale_map(model_dir: str | Path):
    """Resolve every serialized FP8/MXFP4 scale through the model profile.

    The returned key space is the live model weight namespace.  Delegating to
    ``layer_streaming`` is intentional: it is the one producer-side contract
    that understands legacy ``.weight_scale_inv`` tensors, profile-provided
    pairs such as DeepSeek-v4's ``.scale`` siblings, nested text configs, and
    explicitly declared MXFP4 expert payloads.
    """
    from prismaquant.layer_streaming import (
        Fp8ScaleInvMap,
        _build_fp8_scale_inv_map,
    )

    text_map = _build_fp8_scale_inv_map(str(model_dir), multimodal=False)
    multimodal_map = _build_fp8_scale_inv_map(
        str(model_dir), multimodal=True
    )
    text_block = getattr(text_map, "block", None)
    multimodal_block = getattr(multimodal_map, "block", None)
    if (
        text_block is not None
        and multimodal_block is not None
        and tuple(text_block) != tuple(multimodal_block)
    ):
        raise RuntimeError(
            "text and multimodal FP8 scale maps disagree on the serialized "
            f"weight block: {text_block} != {multimodal_block}"
        )
    merged = dict(text_map)
    for name, entry in multimodal_map.items():
        if name in merged and merged[name] != entry:
            raise RuntimeError(
                f"FP8 scale mapping for {name!r} is ambiguous: "
                f"{merged[name]!r} != {entry!r}"
            )
        merged[name] = entry
    return Fp8ScaleInvMap(
        merged,
        text_block or multimodal_block,
        mxfp4_names=frozenset(
            getattr(text_map, "mxfp4_names", frozenset())
        ) | frozenset(
            getattr(multimodal_map, "mxfp4_names", frozenset())
        ),
    )


def checkpoint_weight_to_live_name(
    weight_key: str,
    *,
    profile,
) -> str:
    """Map a checkpoint weight key to the scale-map/live-model namespace."""
    if profile is not None:
        try:
            live = profile.checkpoint_to_live_name(weight_key)
        except Exception:
            live = None
        if live is not None:
            return str(live)
    return str(weight_key)


def cb_source_weight_bf16_value(
    weight: torch.Tensor,
    *,
    model_weight_name: str,
    fp8_scale_inv_map: Mapping[str, tuple[str, str]] | None,
) -> torch.Tensor:
    """Return the exact value a BF16 producer model exposes to CB rendering.

    Ordinary checkpoints are rounded to BF16, matching the production model
    load.  A mapped block-FP8 or declared MXFP4 source is decoded by the same
    profile-aware implementation used by layer streaming and then rounded to
    BF16.  A float8 tensor without a resolved scale is a hard error: casting
    its storage codes would silently render a different model.
    """
    from prismaquant.layer_streaming import (
        _apply_fp8_dequant_inplace,
        _require_fp8_scale,
    )

    source = torch.as_tensor(weight)
    scale_map = fp8_scale_inv_map or {}
    _require_fp8_scale(model_weight_name, source, scale_map)
    if model_weight_name in scale_map:
        decoded = {model_weight_name: source}
        _apply_fp8_dequant_inplace(
            decoded,
            scale_map,
            torch.device("cpu"),
        )
        source = decoded[model_weight_name]
    return source.to(torch.bfloat16).to(torch.float32).contiguous()
