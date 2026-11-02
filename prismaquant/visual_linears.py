"""Visual-Linear names and the source checkpoint's visual roster.

Moved from ``allocator`` (PQ #3215) so that a model profile reads its roster
(``ModelProfile.visual_linear_roster``) without importing the allocator, whose
imports reach most of the package. This module imports only the standard
library. ``allocator`` re-exports every name below and ``footprint`` re-exports
``NVFP4_WEIGHT_ONLY_STATS_KEY``, so their old import paths still work.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

# Stats identity used by allocator/validation accounting when the exporter
# deliberately emits stock NVFP4 as weight-only W4A16. This is an explicit
# producer contract, not a qname heuristic: callers set it from the resolved
# model profile's checkpoint-to-live mapping.
NVFP4_WEIGHT_ONLY_STATS_KEY = "_nvfp4_weight_only"

_VISUAL_PREFIX_RE = re.compile(r"^(?:model\.)?visual\.")

VISUAL_NON_LINEAR_NAME_RE = re.compile(
    r"(?:^|\.)("
    r"pos_embed|patch_embed|embed_tokens"  # embedding tables, not Linears
    r"|rotary_emb|lm_head"  # fixed parts: rotary cache, fixed head
    r"|router"  # router: never a priced Linear
    r"|downsample"  # strided conv block, not a Linear
    r"|[^.]*conv[^.]*"  # conv weights, not Linears
    r"|[^.]*norm[^.]*"  # norms, not Linears
    r")(?:\.|$)"
)


def _admitting_visual_root(name: str, profile=None) -> str | None:
    """Return the declared visual root that admits `name`, if any."""
    roots = profile.visual_root_prefixes() if profile is not None else ()
    if roots:
        for root in roots:
            if name.startswith(root + ".") or (
                root.startswith("model.") and name.startswith(root[6:] + ".")
            ):
                return root
        return None
    if _VISUAL_PREFIX_RE.match(name):
        return "model.visual" if name.startswith("model.") else "visual"
    return None


def _is_visual_linear(name: str, profile=None) -> bool:
    """Classify a visual Linear by the profile's declared root namespaces.

    Keep the existing source/recipe `model.` alias. Callers without declared
    roots retain the historical visual namespace control; a declared family
    cannot acquire unrelated body/audio namespaces through a loose prefix.
    Role tags such as `vis_*`/`merger_*` are not module namespaces and
    never admit.
    """
    return _admitting_visual_root(name, profile) is not None


def discover_visual_linear_stats_from_source(
    model_path: str,
    *,
    strict: bool = False,
    profile=None,
) -> dict[str, dict[str, object]]:
    """Scan source safetensors for visual-Linear names and exact shapes.

    Returned keys are the basename (``.weight`` stripped); values carry the
    same ``n_params``/``in_features``/``out_features`` fields as probe stats.
    Keeping shapes is required for exact Pareto byte pricing when the text-only
    probe did not instantiate the visual tower.

    Text-only staging omits visual probe/cost rows, so this census supplies
    exact auxiliary floors for the explicit source/uniform control. With
    measured multimodal inputs it independently checks complete roster and
    shape coverage before any visual per-Linear decision. It never admits
    a format that the downstream export/serving gates do not support.
    """
    src = Path(model_path)
    idx_path = src / "model.safetensors.index.json"
    candidates: list[tuple[str, tuple[int, ...], str]] = []
    source_tensor_names: set[str] = set()
    scan_errors: list[str] = []
    if idx_path.exists():
        with open(idx_path) as f:
            wm = json.load(f).get("weight_map", {})
        source_tensor_names.update(str(name) for name in wm)
        # Index file carries only names, not shapes. We need to open each
        # referenced shard once to read rank.
        from collections import defaultdict as _dd
        by_shard: dict[str, list[str]] = _dd(list)
        for key, shard in wm.items():
            if not key.endswith(".weight"):
                continue
            if not _is_visual_linear(key, profile):
                continue
            by_shard[shard].append(key)
        try:
            from safetensors import safe_open
        except ImportError as exc:
            if strict:
                raise RuntimeError(
                    "exact visual source discovery requires safetensors"
                ) from exc
            return {}
        for shard, keys in by_shard.items():
            shard_path = src / shard
            if not shard_path.exists():
                scan_errors.append(
                    f"missing indexed shard {shard!r} for {keys[:4]}"
                )
                continue
            with safe_open(str(shard_path), framework="pt") as sf:
                for k in keys:
                    try:
                        tensor_slice = sf.get_slice(k)
                        shape = tuple(tensor_slice.get_shape())
                        dtype = str(tensor_slice.get_dtype()).upper()
                    except Exception as exc:
                        scan_errors.append(f"{shard}:{k}: {exc}")
                        continue
                    candidates.append((k, shape, dtype))
    else:
        # No index file — scan every safetensors shard directly. Used for
        # small, single-file checkpoints.
        try:
            from safetensors import safe_open
        except ImportError as exc:
            if strict:
                raise RuntimeError(
                    "exact visual source discovery requires safetensors"
                ) from exc
            return {}
        import os as _os
        if not src.exists():
            if strict:
                raise FileNotFoundError(
                    f"visual source checkpoint does not exist: {src}"
                )
            return {}
        for f in sorted(_os.listdir(src)):
            if not f.endswith(".safetensors"):
                continue
            with safe_open(str(src / f), framework="pt") as sf:
                source_tensor_names.update(str(key) for key in sf.keys())
                for k in sf.keys():
                    if not k.endswith(".weight"):
                        continue
                    if not _is_visual_linear(k, profile):
                        continue
                    try:
                        tensor_slice = sf.get_slice(k)
                        shape = tuple(tensor_slice.get_shape())
                        dtype = str(tensor_slice.get_dtype()).upper()
                    except Exception as exc:
                        scan_errors.append(f"{f}:{k}: {exc}")
                        continue
                    candidates.append((k, shape, dtype))

    if strict and scan_errors:
        raise RuntimeError(
            "exact visual source discovery was incomplete: "
            + "; ".join(scan_errors[:8])
        )

    # Only rank-2 weights are Linear-like; conv / norms / biases are
    # kept at BF16 passthrough regardless of --visual-format.
    # Additionally, VISUAL_NON_LINEAR_NAME_RE excludes known non-Linear
    # modules by name across five classes (embedding, conv, norm, router,
    # fixed parts), including rank-2 tensors that live in `nn.Parameter` /
    # `nn.Embedding` modules (NOT `nn.Linear`) which the
    # compressed-tensors loader in vLLM cannot consume. Example:
    # `model.visual.pos_embed.weight` is an Embedding-like learned
    # parameter — rank-2 but NOT a Linear. Quantizing it produces
    # `pos_embed.input_global_scale` etc. which vLLM's VL runtime rejects
    # with `KeyError: pos_embed.input_global_scale` because its
    # `model.visual.pos_embed` is a bare Parameter, not a quantizable
    # Linear module.
    out: dict[str, dict[str, object]] = {}
    for name, shape, dtype in candidates:
        if len(shape) != 2:
            continue
        if VISUAL_NON_LINEAR_NAME_RE.search(name):
            continue
        qname = name[:-len(".weight")] if name.endswith(".weight") else name
        out_features, in_features = (int(shape[0]), int(shape[1]))
        entry = {
            "n_params": out_features * in_features,
            "out_features": out_features,
            "in_features": in_features,
            NVFP4_WEIGHT_ONLY_STATS_KEY: True,
            "source_dtype": (
                "bf16" if dtype == "BF16"
                else "fp8" if dtype.startswith("F8_")
                else "fp16" if dtype == "F16"
                else "fp32" if dtype == "F32"
                else dtype.lower()
            ),
            "has_fp8_scale": any(
                f"{qname}.{suffix}" in source_tensor_names
                for suffix in ("weight_scale_inv", "weight_scale")
            ),
        }
        previous = out.setdefault(qname, entry)
        if previous != entry:
            raise RuntimeError(
                f"visual Linear {qname!r} appears with conflicting source "
                f"shapes: {previous} versus {entry}"
            )
    return dict(sorted(out.items()))


def visual_linear_roster(
    model_path: str,
    *,
    profile=None,
    strict: bool = False,
) -> dict[str, dict[str, object]]:
    """Quantizable vision/merger roster with shape and name provenance.

    Keys match `discover_visual_linear_stats_from_source`. Each entry adds
    `shape` ([out, in]), `name_source` (`checkpoint_index` when names come
    from `model.safetensors.index.json`, else `source_header`), and
    `visual_root` (the declared root that admits the name). Out-of-scope
    classes (embedding, conv, norm, router, fixed parts) never appear:
    `VISUAL_NON_LINEAR_NAME_RE` excludes them by name and non-rank-2
    weights are not Linear-like. Read-only over the source checkpoint;
    allocator math, export, pins, kernels and gates are untouched.
    """
    stats = discover_visual_linear_stats_from_source(
        model_path, profile=profile, strict=strict
    )
    index_path = Path(model_path) / "model.safetensors.index.json"
    name_source = "checkpoint_index" if index_path.exists() else "source_header"
    roster: dict[str, dict[str, object]] = {}
    for qname, entry in stats.items():
        roster[qname] = {
            **entry,
            "shape": [entry["out_features"], entry["in_features"]],
            "name_source": name_source,
            "visual_root": _admitting_visual_root(qname, profile),
        }
    return roster
