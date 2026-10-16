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
        <name>.weight_scale          (fp8_e4m3fn for NVFP4 / e8m0 for MXFP8_E4M3/E5M2)
        <name>.weight_global_scale   (fp32, NVFP4 only)
        <name>.input_global_scale    (fp32, A4/A8 formats only)
    OR `<name>.weight` (passthrough in the source storage dtype) for
    layers in the uncompressed bucket.

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


import prismaquant.native_render_codec as _native_render_codec
from .native_render_levers import NativeRenderLevers

# Export-run inputs; numerical routines receive these explicitly.
_RENDER_LEVERS: NativeRenderLevers | None = None
_CACHED_ACTIVATIONS = None
_INPUT_GLOBAL_SCALES = None

import argparse
import gc
import resource
import json
import math
import os
import re
import shutil
import sys
import time
from contextlib import contextmanager
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Sequence

import torch
import torch.nn as nn
from .digests import DIRECT_ASCII_SPACED_LAX
try:
    from accelerate import init_empty_weights
except ModuleNotFoundError:
    @contextmanager
    def init_empty_weights():
        with torch.device("meta"):
            yield
from safetensors.torch import save_file

from . import io_spans
from . import nvfp4_activation_contract as _nvfp4_activation_contract
from prismaquant.source_format_contracts import PASSTHROUGH_SOURCE_REQUIREMENTS
from prismaquant.allocator_candidates import (
    _scan_source_dtype_manifest,
    check_format_applicability,
)
from .serving_profiles import resolve_target_profile
from prismaquant.probe_inventory import (
    is_packed_experts_module as _is_packed_experts_module,
    packed_expert_param_name_set as _packed_expert_param_name_set,
    packed_expert_parent_for_projection as _packed_expert_parent_for_projection,
    packed_experts_param_names as _packed_experts_param_names,
)
from .layer_config import (
    canonicalize_assignment as _canonicalize_assignment,
    canonicalize_format,
)
from .layer_config import (
    is_layer_config_meta_key as _is_layer_config_meta_key,
    layer_config_metadata as _layer_config_metadata,
)
from .schemas import validate_layer_config_payload
from .export_output_safety import (
    prepare_fresh_export_directory,
    transactional_export_directory,
    validate_fresh_export_directory,
)
from .footprint import (
    enforce_whole_artifact_budget,
    whole_artifact_budget_from_assignment_payload,
)

# ---------------------------------------------------------------------------
# NVFP4 packing. The byte layout (two 4-bit indices/byte, element-0 low nibble,
# element-1 high nibble) matches compressed-tensors'
_ACTIVATION_CACHE_FINGERPRINT: dict[str, object] | None = None


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


def _resolve_perturbed_x_export_inputs(root: str | Path) -> tuple[Path, Path]:
    """Return (layer_config, activation_cache_dir) from an iteration output."""
    root = Path(root)
    layer_config = root / "final_layer_config.json"
    summary_path = root / "summary.json"
    cache_dir: Path | None = None
    if summary_path.is_file():
        with open(summary_path) as f:
            summary = json.load(f)
        if summary.get("final_layer_config"):
            layer_config = Path(summary["final_layer_config"])
            if not layer_config.is_absolute():
                layer_config = root / layer_config
        iterations = summary.get("iterations") or []
        if iterations:
            cache_info = iterations[-1].get("cache", {})
            cache_raw = cache_info.get("cache_dir")
            if cache_raw:
                cache_dir = Path(cache_raw)
                if not cache_dir.is_absolute():
                    cache_dir = root / cache_dir
    if cache_dir is None:
        caches = sorted(root.glob("activation_cache_iter_*"))
        if caches:
            cache_dir = caches[-1]
    if not layer_config.is_file():
        raise FileNotFoundError(
            f"perturbed-X layer config not found at {layer_config}"
        )
    if cache_dir is None or not cache_dir.is_dir():
        raise FileNotFoundError(
            f"perturbed-X activation cache not found under {root}"
        )
    return layer_config, cache_dir


class _LazyFisherDiagCache:
    """HDetailIndex-backed lazy cache for per-Linear Fisher diagonal.

    Mirrors `_LazyActivationCache` but loads `h_diag` tensors of shape
    `[out, in]` from the probe's per-Linear `.pt` blobs. Returns None
    when the requested name isn't in the index — caller (GPTQ wrapper)
    falls back to unweighted Hessian. Loads are demand-driven so the
    full Fisher cache (typically a few GB) doesn't sit resident."""

    def __init__(self, index):
        self.index = index
        self.loads = 0

    def get(self, name: str):
        if name not in self.index:
            return None
        self.loads += 1
        try:
            return self.index.load(name).to(torch.float32)
        except Exception:
            return None


def _activation_index_fingerprint(index, cache_dir: Path) -> dict[str, object]:
    """Cheap cache identity for export-cache invalidation.

    The layer export cache stores quantized tensors whose values depend
    on activation-cache contents. Hash names plus file size/mtime so
    changing the activation cache or pointing at a different cache dir
    invalidates stale layer_NNN.pt files without reading tensor bytes.
    """
    from prismaquant.digests import DIRECT_ASCII_LAX

    paths = getattr(index, "_paths", {})
    rows = []
    for name, path in sorted(paths.items()):
        st = path.stat()
        rows.append([name, path.name, st.st_size, st.st_mtime_ns])
    digest = DIRECT_ASCII_LAX.sha256(rows)[:16]
    return {
        "path": str(cache_dir.resolve()),
        "n_files": len(rows),
        "hash": digest,
    }


def _production_cache_format_candidates(fmt: str) -> tuple[str, ...]:
    fmt_u = str(fmt).upper()
    if fmt_u == _native_render_codec.MXFP8_LEGACY_ALIAS:
        return ("MXFP8_E4M3", _native_render_codec.MXFP8_LEGACY_ALIAS)
    if fmt_u == "MXFP8_E4M3":
        return ("MXFP8_E4M3", _native_render_codec.MXFP8_LEGACY_ALIAS)
    return (fmt_u,)


def _production_cache_name_candidates(name: str) -> tuple[str, ...]:
    names = [name]
    if name.endswith(".weight"):
        names.append(name[:-len(".weight")])
    if name.startswith("model.language_model."):
        names.append("model." + name[len("model.language_model."):])
    return tuple(dict.fromkeys(names))


def _production_cache_lookup_key(name: str, fmt: str):
    cache = _PRODUCTION_WEIGHT_CACHE
    if cache is None:
        return None
    if hasattr(cache, "resolve_key"):
        key = cache.resolve_key(name, fmt)
        if key is not None:
            return key
    weights = getattr(cache, "weights", {}) or {}
    for cand_name in _production_cache_name_candidates(name):
        for cand_fmt in _production_cache_format_candidates(fmt):
            key = (cand_name, cand_fmt)
            if key in weights:
                return key
    return None

def _production_cache_expected_keys(
    assignment: dict[str, str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Cache keys export will actually READ, and which of them are absent.

    Passthrough formats are excluded, not just BF16: their emit paths copy
    source bytes and never call `_pack_production_cached_2d` (the
    FP8_SOURCE branch in the streaming Linear loop returns before the
    cache lookup). `build_production_cache --render-scope assignment` does
    render them — FP8_SOURCE's `quantize_dequantize` is identity — so the
    keys usually exist, but demanding a render this export cannot consume
    would fail an otherwise-valid FP8-source export over an unused entry.
    Before issue #29 the question never arose: the runtime-legality guard
    rewrote every FP8_SOURCE entry to BF16 before this check saw it.
    """
    from prismaquant.production_weight_cache import is_uncached_packed_expert_qname

    keys: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []
    for qname, fmt in assignment.items():
        cache_fmt = str(fmt).upper()
        canonical = _native_render_codec._canonical_export_format(cache_fmt)
        if canonical in PASSTHROUGH_SOURCE_REQUIREMENTS:
            continue
        key = _production_cache_lookup_key(qname, cache_fmt)
        if key is None:
            if (
                _ALLOW_PACKED_EXPERT_RTN
                and is_uncached_packed_expert_qname(qname)
            ):
                continue
            missing.append((qname, cache_fmt))
        else:
            keys.append(key)
    return keys, missing


def _production_cache_fingerprint(
    cache,
    expected_keys: Sequence[tuple[str, str]],
) -> dict[str, object]:
    """Cheap identity for direct production-cache export.

    The direct path packs already-rendered GPTQ/scale-sweep weights. Bind the
    export layer cache to the backing shard names, mtimes, lever metadata, and
    activation-scale summary so a stale export cache cannot be reused across
    production-cache changes.
    """
    from prismaquant.digests import (
        DIRECT_ASCII_LAX,
        DIRECT_ASCII_LAX_DEFAULT_STR,
    )

    weights = getattr(cache, "weights", {}) or {}
    cache_dir = getattr(cache, "cache_dir", None)
    rows = []
    for key in sorted(set(expected_keys)):
        value = weights.get(key)
        if isinstance(value, torch.Tensor):
            rows.append([key[0], key[1], "tensor",
                         list(value.shape), str(value.dtype)])
            continue
        if value is None:
            rows.append([key[0], key[1], "missing"])
            continue
        path = Path(str(value))
        if cache_dir and not path.is_absolute():
            path = Path(cache_dir) / path
        try:
            st = path.stat()
            rows.append([key[0], key[1], path.name, st.st_size, st.st_mtime_ns])
        except OSError:
            rows.append([key[0], key[1], path.name, "missing"])
    act = getattr(cache, "activation_max_abs", None) or {}
    act_digest = DIRECT_ASCII_LAX.sha256(act)[:16]
    metadata = dict(getattr(cache, "metadata", {}) or {})
    metadata_digest = DIRECT_ASCII_LAX_DEFAULT_STR.sha256(metadata)[:16]
    digest = DIRECT_ASCII_LAX.sha256(rows)[:16]
    # Hook enumeration the shipped bytes were rendered against (#147,
    # consumer 3): a shipped artifact records which enumeration its bytes
    # saw, so a cost table priced from one rendering cannot be silently
    # served from another. ``None`` for caches that predate the stamp.
    from prismaquant.production_weight_cache import activation_hook_scope_of

    return {
        "path": str(Path(cache_dir).resolve()) if cache_dir else None,
        "n_entries": len(rows),
        "hash": digest,
        "activation_max_abs_hash": act_digest,
        "metadata_hash": metadata_digest,
        "levers": dict(getattr(cache, "levers", {}) or {}),
        "activation_hook_scope": activation_hook_scope_of(cache),
    }


def _production_cache_scales(cache, *, profile=None) -> dict[str, float]:
    activation_max_abs = getattr(cache, "activation_max_abs", None) or {}
    scales = {
        name: _native_render_codec._nvfp4_input_global_scale_from_max_abs(float(max_abs), levers=_export_render_options())
        for name, max_abs in activation_max_abs.items()
        if max_abs and float(max_abs) > 0.0
    }
    return _unify_input_global_scales_across_fused_siblings(
        scales,
        profile=profile,
    )



def _source_weight_shape_for_recipe(
    src_model: str,
    recipe_key: str,
    profile=None,
) -> list[int] | None:
    idx_path = Path(src_model) / "model.safetensors.index.json"
    if not idx_path.exists():
        return None
    with open(idx_path) as f:
        weight_map = json.load(f).get("weight_map", {})
    candidates = [recipe_key + ".weight"]
    if profile is not None:
        source_name = profile.source_tensor_name(recipe_key)
        candidates.append(source_name + ".weight")
        candidates.append(profile.source_tensor_name(recipe_key + ".weight"))
    if recipe_key.startswith("model."):
        candidates.append(
            "model.language_model." + recipe_key[len("model."):] + ".weight"
        )
    for ckpt_key in dict.fromkeys(candidates):
        shard = weight_map.get(ckpt_key)
        if shard is None:
            continue
        from safetensors import safe_open
        with safe_open(str(Path(src_model) / shard), framework="pt") as sf:
            return list(sf.get_slice(ckpt_key).get_shape())
    return None


class _RuntimeCoercion(NamedTuple):
    """One Linear the runtime-legality guard rewrote to BF16.

    Positionally tuple-compatible with the legacy ``(name, shape,
    from_fmt)`` row this function used to return: `_bf16_upgrade_audit`
    and the manifest read the first three fields, so old readers keep
    working while the serving-group fields carry WHY a Linear the
    allocator chose a quantized format for is shipping unquantized.
    """
    name: str
    shape: list[int] | None
    from_fmt: str
    reason: str = ""
    detail: str = ""
    # Set only when the coercion was forced by serving-atomicity rather
    # than by this Linear's own legality verdict.
    serving_group: str | None = None
    serving_group_kind: str | None = None
    serving_group_members: tuple[str, ...] = ()
    trigger: str | None = None
    delta_bytes: int | None = None


def _bf16_coercion_delta_bytes(
    shape: Sequence[int] | None,
    from_fmt: str,
) -> int | None:
    """Bytes a Linear GAINS by shipping BF16 instead of ``from_fmt``."""
    if not shape:
        return None
    from .format_registry import get_format

    try:
        src = get_format(from_fmt)
        bf16 = get_format("BF16")
    except KeyError:
        return None
    shape_t = tuple(int(dim) for dim in shape)
    return int(
        bf16.memory_bytes_for_shape(shape_t) - src.memory_bytes_for_shape(shape_t)
    )


def _human_bytes(n_bytes: int | None) -> str:
    if n_bytes is None:
        return "unknown"
    value = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.2f} GiB"


def _group_legal_quantized_formats(
    shapes: dict[str, list[int] | None],
    members: Sequence[str],
    target_profile: str,
) -> list[tuple[str, float]]:
    """Emittable QUANTIZED formats legal for every checkable member.

    "Checkable" is the same bar the coercion itself uses: a member with a
    known 2-D source shape. Passthrough formats are excluded: BF16 is the
    fallback under discussion, and FP8_SOURCE is a *source-dependent*
    rung (legal only where every member's source is already fp8), so
    offering it as a substitute would flip refuse-vs-coerce on FP8-source
    models — a unit that must fall back to BF16 today would instead be
    refused. That rung is already in the allocator's menu, which is where
    picking it belongs; export names alternatives, it does not choose.

    A non-empty answer is the decisive fact for refuse-vs-coerce: it says
    the allocation is repairable upstream at a quantized bit rate, so
    rewriting the whole serving unit to 16 bpp is NOT what a
    shape-aware allocator would have produced.
    """
    from .format_registry import get_format

    checkable = [
        (name, tuple(int(d) for d in shapes[name]))
        for name in members
        if shapes.get(name) is not None and len(shapes[name] or ()) == 2
    ]
    if not checkable:
        return []
    legal: list[tuple[str, float]] = []
    for fmt in sorted(EXPORTABLE_FORMATS - set(PASSTHROUGH_SOURCE_REQUIREMENTS)):
        if not all(
            check_format_applicability(
                shape,
                fmt,
                qname=name,
                target_profile=target_profile,
            ).legal
            for name, shape in checkable
        ):
            continue
        try:
            bits = get_format(fmt).effective_bits_for_shape(checkable[0][1])
        except KeyError:
            bits = float("nan")
        legal.append((fmt, float(bits)))
    return sorted(legal, key=lambda row: (row[1], row[0]))


def _coerce_runtime_legal_assignment(
    src_model: str,
    assignment: dict[str, str],
    profile=None,
) -> tuple[dict[str, str], list[_RuntimeCoercion]]:
    """Adjust assignments that the target runtime cannot execute.

    Shape and format legality comes from serving-profile config. BF16 is the
    conservative runtime fallback when a format this exporter CAN emit turns
    out to be illegal on this Linear's shape or denied by the resolved
    serving profile.

    A format the exporter cannot emit at all (not in `EXPORTABLE_FORMATS`)
    is NOT coerced — it raises. Rewriting it to BF16 would ship that Linear
    at 16 bpp, blowing the byte budget the allocation was selected under and
    producing an artifact whose real bpp disagrees with its own
    `layer_config.json`. The serving profile's export lane already bounds
    the allocator's menu by this exporter's declaration, so reaching here
    means that bound regressed; masking it with a rewrite is the
    post-allocator band-aid CLAUDE.md §4.1 vetoes.

    **Serving-atomic groups (issue #28).** The coercion is group-aware. A
    packed-MoE expert unit and a fused-sibling set must carry ONE format
    (vLLM selects one scheme per FusedMoE layer / per merged-column
    Linear), and their members do NOT share a shape — an odd
    `moe_intermediate_size` can make `down_proj` indivisible while
    `gate_up_proj` is fine. Rewriting only the offending member would
    produce a quantized + BF16 mix inside one serving unit: a
    hard-serving-invariant violation that `build_quantization_config`'s
    fused-coherence gate then reports as a wrong-model-profile problem it
    is not. So when a member of a serving unit is illegal, this function
    resolves the WHOLE unit, and never leaves a unit mixed:

      * if some emittable QUANTIZED format is legal for every member, it
        **raises**. Coercing would ship the whole unit at 16 bpp — for a
        packed-expert unit that is `num_experts x` the per-Linear cost
        this branch is justified by, and the model-wide dimension that
        made one member illegal makes it every layer's unit — when a
        re-solve lands the unit on that legal format for free. Export
        must not pick the substitute itself: the format it picked is the
        one the production weight cache holds a deliberate render for
        (a substitute is a cache miss at best, an RTN render at worst),
        so naming the legal rung and refusing is the honest move.
      * if NO quantized format is legal for every member, BF16 is not a
        band-aid but the only representable answer — exactly what a
        shape-aware allocator would have had to pick — so the whole unit
        is coerced, loudly, and every member is recorded in the returned
        rows (and from there in `runtime_coercions` +
        `bf16_audit.serving_group`).

    **Passthrough source integrity (issue #29).**
    `PASSTHROUGH_SOURCE_REQUIREMENTS` makes FP8_SOURCE legal only where the
    source tensor is ALREADY fp8, which `check_format_applicability` can
    only judge with a `source_kind`. That argument used to be omitted, so
    every FP8_SOURCE Linear came back `source_dtype_mismatch` and was
    rewritten to BF16 —
    inert in the bytes (materialization copies the source fp8 verbatim and
    `_fp8_source_config_overlay` restores the config), but it filled every
    DSv4 / Hy3 / MiniMax manifest's `runtime_coercions` with rows for
    demotions that never happened, hiding any real one. The `source_kind`
    now comes from `_scan_source_dtype_manifest` — the SAME recipe-keyed
    map `build_candidates` gates the allocator's passthrough candidates
    on — so export's verdict and the gate that admitted the allocation
    cannot disagree, and a passthrough row now means the source really is
    not fp8 (i.e. an upstream `PASSTHROUGH_SOURCE_REQUIREMENTS` failure)
    and the bytes really do change. Such a verdict is therefore treated
    like any other illegality, group escalation included.

    Upstream is the real fix: the allocator's candidate mask intersects
    shape legality per member, so a promoted format is legal for every
    member by construction and this path is unreachable in normal
    operation. Treat any firing as an upstream regression worth reporting
    — `_runtime_coercion_report` is written to be impossible to miss.
    """
    out = dict(assignment)
    target_profile = _allocator_target_profile_for_audit(profile) or "research"
    shapes: dict[str, list[int] | None] = {}
    # Recipe-keyed source dtypes, read lazily: only passthrough formats
    # consult `source_kind` inside `check_format_applicability`, and the
    # scan is safetensors-header IO that a BF16-source export never needs
    # (CLAUDE.md §4.7 — no disk work on a path that cannot use it).
    source_kinds: dict[str, str] | None = None

    def _source_kind_for(qname: str) -> str | None:
        nonlocal source_kinds
        if source_kinds is None:
            source_kinds = _scan_source_dtype_manifest(src_model, profile)
        return source_kinds.get(qname)

    # qname -> (shape, from_fmt, reason, detail)
    illegal: dict[str, tuple[list[int], str, str, str]] = {}
    for qname, fmt in assignment.items():
        fmt_canonical = _native_render_codec._canonical_export_format(fmt)
        out[qname] = fmt_canonical
        if fmt_canonical == "BF16":
            continue
        if fmt_canonical not in EXPORTABLE_FORMATS:
            from prismaquant.gguf_formats import GGUF_BLOCK_BYTES
            from prismaquant import format_registry as _fr

            if _fr.RETIRED_CODEBOOK_FORMAT_RE.fullmatch(fmt_canonical):
                # A retired codebook rung (the Gridbook lane, archived
                # 2026-09-25, #1304) refuses with the registry's pointer to
                # the archive rather than as an unknown research format.
                _fr.get_format(fmt_canonical)
            if fmt_canonical in GGUF_BLOCK_BYTES:
                # Wrong container, not a research format: a GGUF assignment
                # reaching the compressed-tensors exporter means the pipeline
                # was launched without EXPORT_CONTAINER=gguf. Silently
                # coercing to BF16 would ship a ~16 bpp artifact unrelated to
                # the allocated budget.
                raise ValueError(
                    f"{qname}: format {fmt_canonical} ships via the GGUF "
                    f"container (prismaquant.export_gguf / "
                    f"EXPORT_CONTAINER=gguf), not compressed-tensors"
                )
            raise ValueError(
                f"{qname}: format {fmt_canonical} has no compressed-tensors "
                f"emit path -- it is absent from FORMAT_SCHEME, so there is "
                f"no `config_groups` scheme for vLLM to dispatch on "
                f"(emittable={sorted(EXPORTABLE_FORMATS)}). Rewriting it to "
                f"BF16 here would ship this Linear at 16 bpp and silently "
                f"blow the byte budget the allocation was selected under, "
                f"leaving the artifact's real bpp disagreeing with its own "
                f"layer_config.json. The serving profile that admitted it "
                f"(target_profile={target_profile!r}) is supposed to bound "
                f"its menu by this exporter's EXPORTABLE_FORMATS via its "
                f"export lane, so this is a regression in that bound (or an "
                f"allocation solved under a different profile than "
                f"PRISMAQUANT_TARGET_PROFILE resolves to here) -- re-solve "
                f"the allocation under the serving profile you are exporting "
                f"for rather than letting export rewrite it."
            )
        shape = _source_weight_shape_for_recipe(src_model, qname, profile)
        shapes[qname] = shape
        if shape is None or len(shape) != 2:
            continue
        verdict = check_format_applicability(
            tuple(shape),
            fmt,
            qname=qname,
            source_kind=(
                _source_kind_for(qname)
                if fmt_canonical in PASSTHROUGH_SOURCE_REQUIREMENTS
                else None
            ),
            target_profile=target_profile,
        )
        if not verdict.legal:
            illegal[qname] = (
                shape,
                fmt_canonical,
                verdict.reason or "illegal",
                verdict.detail or "",
            )

    if not illegal:
        return out, []

    coerced: list[_RuntimeCoercion] = []
    handled: set[str] = set()

    # No passthrough exemption (issue #29). It existed only because the
    # FP8_SOURCE verdict was an artifact of the missing `source_kind` —
    # escalating a bogus demotion would have coerced whole packed-expert
    # units to BF16 on every FP8-source model. Now that the verdict is
    # source-aware, a passthrough mismatch is a real illegality with a
    # real byte cost, and gets the same serving-atomic treatment as a
    # shape or policy one: refused when the unit has a legal quantized
    # rung, coerced as a whole unit when it does not.
    from .name_projection import NameProjection
    if profile is None:
        from .model_profiles import DefaultProfile
        profile = DefaultProfile()
    try:
        components = NameProjection(profile).serving_components(out, fused_metadata_fallback=True)
    except Exception as exc:
        raise ValueError(
            f"cannot verify serving-atomic coherence: the model profile could not name the serving unit: {exc}. "
            "Declare packed_experts.projection_splits rather than guessing coupling.") from exc
    component_by_name = {name: component for component in components
                         for name in component.members}
    refusals: list[str] = []
    for qname in sorted(illegal):
        if qname in handled:
            continue
        shape, from_fmt, reason, detail = illegal[qname]
        component = component_by_name[qname]
        units = component.units
        if not units:
            # Not serving-atomic: coerce this Linear alone, as before.
            handled.add(qname)
            out[qname] = "BF16"
            coerced.append(
                _RuntimeCoercion(
                    qname, shape, from_fmt, reason, detail,
                    delta_bytes=_bf16_coercion_delta_bytes(shape, from_fmt),
                )
            )
            continue
        members = list(component.members)
        co_triggers = sorted(m for m in members if m in illegal)
        handled.update(co_triggers)
        alternatives = _group_legal_quantized_formats(
            shapes, members, target_profile
        )
        # Name the component by its WIDEST unit — the FusedMoE / merged
        # column that actually constrains it. Concatenating every unit key
        # would put a 128-expert layer's whole key list on every row of
        # the manifest for no extra information: `serving_group_members`
        # already carries the exact membership.
        widest = max(units, key=lambda unit: (len(unit.members), unit.key))
        group_key = widest.key
        group_kind = "+".join(sorted({unit.kind for unit in units}))
        would_cost = sum(
            delta
            for delta in (
                _bf16_coercion_delta_bytes(
                    shapes.get(member), _native_render_codec._canonical_export_format(out[member])
                )
                for member in members
                if _native_render_codec._canonical_export_format(out[member]) != "BF16"
            )
            if delta is not None
        )
        if alternatives:
            refusals.append(
                _describe_serving_group_refusal(
                    trigger=qname,
                    shape=shape,
                    from_fmt=from_fmt,
                    reason=reason,
                    detail=detail,
                    group_key=group_key,
                    group_kind=group_kind,
                    members=members,
                    co_triggers=co_triggers,
                    alternatives=alternatives,
                    would_cost=would_cost,
                    target_profile=target_profile,
                )
            )
            continue
        # BF16 is the only representable format left for this unit: it is
        # what a shape-aware allocator must pick, so take it for the WHOLE
        # unit. Never one member — that is the mixed-scheme artifact.
        for member in members:
            member_fmt = _native_render_codec._canonical_export_format(out[member])
            if member_fmt == "BF16":
                continue
            member_shape = shapes.get(member)
            out[member] = "BF16"
            if member in illegal:
                member_reason, member_detail = illegal[member][2], illegal[member][3]
            else:
                member_reason = "serving_group_coherence"
                member_detail = (
                    f"coerced with its serving-atomic unit; {qname} is "
                    f"illegal for {from_fmt} ({reason})"
                )
            coerced.append(
                _RuntimeCoercion(
                    member,
                    member_shape,
                    member_fmt,
                    member_reason,
                    member_detail,
                    group_key,
                    group_kind,
                    tuple(members),
                    qname,
                    _bf16_coercion_delta_bytes(member_shape, member_fmt),
                )
            )

    if refusals:
        raise ValueError(
            f"serving-atomic group is not runtime-legal for its allocated "
            f"format in {len(refusals)} unit(s). vLLM selects ONE scheme per "
            f"FusedMoE layer and one per merged-column Linear, so every "
            f"member of these units must carry ONE format; export refuses to "
            f"rewrite them rather than ship a mixed (unservable) unit or a "
            f"whole unit at 16 bpp when a quantized format legal for every "
            f"member exists. "
            + " || ".join(refusals)
        )
    return out, coerced


def _describe_serving_group_refusal(
    *,
    trigger: str,
    shape: Sequence[int] | None,
    from_fmt: str,
    reason: str,
    detail: str,
    group_key: str,
    group_kind: str,
    members: Sequence[str],
    co_triggers: Sequence[str],
    alternatives: Sequence[tuple[str, float]],
    would_cost: int,
    target_profile: str,
) -> str:
    """One refusal clause: what is illegal, and what to do about it."""
    alt_text = ", ".join(f"{fmt} (~{bits:.3f} bpp)" for fmt, bits in alternatives)
    others = [name for name in co_triggers if name != trigger]
    parts = [
        f"unit {group_key!r} [{group_kind}] allocated {from_fmt} with "
        f"{len(members)} member(s): {trigger} is ILLEGAL at shape "
        f"{list(shape) if shape else None} "
        f"(reason={reason}; {detail or 'no detail'}) under "
        f"target_profile={target_profile!r}"
        + (f"; also illegal in this unit: {others}" if others else ""),
        f"but these emittable formats ARE legal for every member of the "
        f"unit: {alt_text}. Coercing the unit to BF16 instead would add "
        f"{_human_bytes(would_cost)} for nothing, and the "
        + ("source precision"
           if reason == "source_dtype_mismatch" else "dimension")
        + f" that made {trigger.rsplit('.', 1)[-1]} illegal is model-wide, "
        f"so it is that much per unit across every layer",
    ]
    parts.append(
        f"members={sorted(members)[:8]}"
        + (f" (+{len(members) - 8} more)" if len(members) > 8 else "")
    )
    parts.append(
        f"FIX: re-solve the allocation for target_profile={target_profile!r} "
        f"so this unit's promoted format is legal for every member -- "
        f"legality (shape, serving policy, and passthrough source precision) "
        f"has to be intersected ACROSS a serving unit, because that "
        f"unit is one scheme at serve time (issue #28); or, when "
        f"re-exporting an allocation that predates that, set every member of "
        f"the unit to one of the legal formats above in layer_config.json. "
        f"Export deliberately does not substitute the format itself: the "
        f"production weight cache holds a deliberate render for the format "
        f"that was ALLOCATED, so a substitution is a cache miss (or an RTN "
        f"render), and picking formats is the allocator's job"
    )
    return " -- ".join(parts)


def _runtime_coercion_report(coerced: Sequence[_RuntimeCoercion]) -> str:
    """Operator-facing report for runtime coercions, group-aware.

    A whole-serving-unit coercion means an allocation reached export with
    a format that is illegal for a member of a unit that must be uniform.
    Upstream is supposed to make that unrepresentable, so this is loud on
    purpose: it is a safety net firing, i.e. evidence of an upstream
    regression (or a pre-#28 `layer_config.json`), not routine hygiene.
    """
    if not coerced:
        return ""
    deltas = [
        row.delta_bytes for row in coerced
        if getattr(row, "delta_bytes", None) is not None
    ]
    lines = [
        f"[export-stream] runtime format coercions: {len(coerced)} Linears "
        f"-> BF16 (target runtime does not support those format/shape pairs)"
        + (
            f"; +{_human_bytes(sum(deltas))} over the allocated formats"
            if deltas else ""
        ),
    ]
    # A passthrough verdict is not a runtime-support fact but a broken
    # allocator contract: FP8_SOURCE was assigned to a Linear whose source
    # is not fp8, which `PASSTHROUGH_SOURCE_REQUIREMENTS` is supposed to
    # make unreachable (issue #29). Since #29 this can no longer be the
    # missing-`source_kind` false positive, so say so in those words.
    passthrough = [
        row for row in coerced
        if getattr(row, "reason", "") == "source_dtype_mismatch"
    ]
    if passthrough:
        lines.append(
            f"[export-stream] WARNING: {len(passthrough)} PASSTHROUGH "
            f"SOURCE MISMATCH(ES): a passthrough format was allocated to a "
            f"Linear whose source is not that precision, so the source bytes "
            f"cannot be copied and BF16 is the only representable answer. "
            f"PASSTHROUGH_SOURCE_REQUIREMENTS is supposed to make this "
            f"unreachable -- re-solve the allocation against this checkpoint "
            f"(the allocator's source manifest disagrees with the source "
            f"safetensors). "
            + str([(row.name, row.from_fmt, row.detail)
                   for row in passthrough[:6]])
        )
    groups: dict[str, list[_RuntimeCoercion]] = {}
    singles: list[_RuntimeCoercion] = []
    for row in coerced:
        key = getattr(row, "serving_group", None)
        if key:
            groups.setdefault(str(key), []).append(row)
        else:
            singles.append(row)
    if groups:
        lines.append(
            "[export-stream] " + "=" * 62
        )
        lines.append(
            f"[export-stream] WARNING: {len(groups)} SERVING-ATOMIC UNIT(S) "
            f"COERCED TO BF16 IN FULL."
        )
        lines.append(
            "[export-stream]   vLLM needs ONE format per FusedMoE layer / "
            "merged-column Linear, and no"
        )
        lines.append(
            "[export-stream]   emittable quantized format was legal for "
            "every member, so the whole unit"
        )
        lines.append(
            "[export-stream]   ships unquantized. The allocator is supposed "
            "to make this unreachable, so"
        )
        lines.append(
            "[export-stream]   treat it as an UPSTREAM REGRESSION (or a "
            "pre-#28 layer_config.json) and"
        )
        lines.append(
            "[export-stream]   report it: this artifact's real bpp EXCEEDS "
            "its own layer_config.json."
        )
        for key, rows in sorted(groups.items()):
            first = rows[0]
            delta = sum(
                row.delta_bytes for row in rows
                if row.delta_bytes is not None
            )
            lines.append(
                f"[export-stream]   unit {key} [{first.serving_group_kind}]: "
                f"{len(rows)} Linears -> BF16, +{_human_bytes(delta)}"
            )
            lines.append(
                f"[export-stream]     trigger={first.trigger} "
                f"from={first.from_fmt} reason={first.reason} "
                f"detail={first.detail}"
            )
            lines.append(
                "[export-stream]     members="
                + str(list(first.serving_group_members)[:8])
                + (
                    f" (+{len(first.serving_group_members) - 8} more)"
                    if len(first.serving_group_members) > 8
                    else ""
                )
            )
        lines.append("[export-stream] " + "=" * 62)
    if singles:
        lines.append(
            "[export-stream]   ungrouped: "
            + str([(row.name, row.shape, row.from_fmt) for row in singles[:6]])
        )
    return "\n".join(lines)


def _runtime_coercion_manifest_rows(
    coerced: Sequence[_RuntimeCoercion],
) -> list[dict[str, object]]:
    """`runtime_coercions` records for `mixed_native_manifest.json`."""
    rows: list[dict[str, object]] = []
    for row in coerced:
        entry: dict[str, object] = {
            "name": row[0],
            "shape": row[1],
            "from": row[2],
            "to": "BF16",
            "reason": getattr(row, "reason", "") or None,
            "detail": getattr(row, "detail", "") or None,
            "delta_bytes": getattr(row, "delta_bytes", None),
        }
        group = getattr(row, "serving_group", None)
        if group:
            entry["serving_group"] = {
                "key": group,
                "kind": row.serving_group_kind,
                "members": list(row.serving_group_members),
                "trigger": row.trigger,
            }
        rows.append(entry)
    return rows


# The serving profile the allocator actually solved with, read out of
# layer_config.json's reserved metadata block at load time (re-vet R11).
_ALLOCATOR_TARGET_PROFILE: str | None = None


def _allocator_target_profile_for_audit(profile) -> str | None:
    # Export must audit legality under the SAME serving profile the allocator
    # solved with. When they differ, export coerces every format the profile
    # it resolved does not serve — 2026-07-11: 226 dense FP8 Linears silently
    # -> BF16 on the Hy3 CT export, because the allocator ran vllm_packed_moe
    # while export re-resolved hy_v3's declared `gguf`.
    #
    # Precedence: PRISMAQUANT_TARGET_PROFILE (explicit operator override for
    # direct exporter invocations) > the allocator's stamp in
    # layer_config.json > the architecture spec default. The stamp travels
    # with the artifact, so the pipeline needs no env plumbing at all.
    requested = (
        os.environ.get("PRISMAQUANT_TARGET_PROFILE")
        or _ALLOCATOR_TARGET_PROFILE
        or None
    )
    if profile is None and requested is None:
        return None
    return resolve_target_profile(profile, requested)


def _bf16_passthrough_for_assignment(
    explicit_ignore: Sequence[str] | None,
    profile,
    allocator_meta: Mapping[str, object] | None,
) -> set[str]:
    """Resolve profile pins without undoing an explicit head assignment.

    Old layer configs carry no head metadata and retain the historical profile
    pin. New allocator configs stamp ``lm_head_mode``: ``fixed`` means the
    measured ``lm_head_format`` is auxiliary to body bpp, while ``dp`` means
    ``--allow-pinned lm_head`` selected it. In either case the assignment — not
    the exporter's default pin — is authoritative for that one head. An
    explicit ``--ignore`` remains the highest-precedence operator override.
    """
    if explicit_ignore is not None:
        return {str(name) for name in explicit_ignore}

    from .fixed_head import remaining_profile_pins

    meta = dict(allocator_meta or {})
    mode = str(meta.get("lm_head_mode") or "")
    fmt = str(meta.get("lm_head_format") or "BF16")
    lift_head = mode == "dp" or (
        mode == "fixed" and _native_render_codec._canonical_export_format(fmt) != "BF16"
    )
    return set(remaining_profile_pins(
        profile,
        fixed_lm_head_quantized=lift_head,
    ))


def _bf16_upgrade_audit(
    src_model: str,
    assignment: dict[str, str],
    bf16_passthrough: set[str],
    runtime_coerced: Sequence[tuple],
    profile,
) -> dict[str, object]:
    """Classify BF16 entries by immutability, runtime gates, or allocation.

    This is intentionally a manifest audit, not a policy change. It tells us
    which BF16 Linears are immutable/passthrough, which were forced by runtime
    format support, and which are real numerical/budget choices where enhanced
    MXFP8_E4M3/MXFP8_E5M2/FP8 may be worth trying next.
    """
    coerced: dict[str, tuple[list[int], str]] = {}
    # Serving-unit coercions (issue #28) are reported as such: a whole
    # FusedMoE / merged-column unit shipping unquantized is a different
    # (and louder) fact than one Linear whose own shape was illegal.
    coerced_groups: dict[str, dict[str, object]] = {}
    for row in runtime_coerced:
        if len(row) >= 3:
            name, shape, from_fmt = row[:3]
        else:
            name, shape = row[:2]
            from_fmt = "MXFP8_E4M3"
        coerced[str(name)] = (shape, str(from_fmt))
        group_key = getattr(row, "serving_group", None)
        if group_key:
            coerced_groups[str(name)] = {
                "key": str(group_key),
                "kind": row.serving_group_kind,
                "members": list(row.serving_group_members),
                "trigger": row.trigger,
                "reason": getattr(row, "reason", "") or None,
                "detail": getattr(row, "detail", "") or None,
            }
    target_profile = _allocator_target_profile_for_audit(profile)
    candidate_formats = ("MXFP8_E4M3", "MXFP8_E5M2", "FP8_E4M3", "FP8_E5M2")
    entries: list[dict[str, object]] = []
    counts: dict[str, int] = {}

    for qname, fmt in sorted(assignment.items()):
        if _native_render_codec._canonical_export_format(fmt) != "BF16":
            continue
        coerced_entry = coerced.get(qname)
        shape = (
            coerced_entry[0]
            if coerced_entry is not None
            else _source_weight_shape_for_recipe(src_model, qname, profile)
        )
        shape_tuple = tuple(shape) if shape is not None else None
        if qname in bf16_passthrough:
            reason = "passthrough_or_immutable"
        elif coerced_entry is not None:
            reason = (
                "runtime_coerced_"
                + ("serving_group_" if qname in coerced_groups else "")
                + "from_"
                + coerced_entry[1].lower().replace("-", "_")
            )
        else:
            verdicts: dict[str, dict[str, object]] = {}
            any_legal = False
            for cand_fmt in candidate_formats:
                if shape_tuple is None or len(shape_tuple) != 2:
                    verdicts[cand_fmt] = {
                        "legal": False,
                        "reason": "shape_unknown",
                        "detail": "",
                    }
                    continue
                verdict = check_format_applicability(
                    shape_tuple,
                    cand_fmt,
                    qname=qname,
                    target_profile=target_profile,
                )
                verdicts[cand_fmt] = {
                    "legal": bool(verdict.legal),
                    "reason": verdict.reason,
                    "detail": verdict.detail,
                }
                any_legal = any_legal or bool(verdict.legal)
            if not any_legal:
                reason = "no_vllm_supported_8bit_candidate"
            elif verdicts.get("MXFP8_E4M3", {}).get("legal"):
                reason = "allocator_selected_bf16_mxfp8_legal"
            else:
                reason = "allocator_selected_bf16_alternate_8bit_legal"
        counts[reason] = counts.get(reason, 0) + 1
        entry: dict[str, object] = {
            "name": qname,
            "reason": reason,
            "shape": list(shape) if shape is not None else None,
        }
        if reason.startswith("allocator_selected") or reason == "no_vllm_supported_8bit_candidate":
            verdicts = {}
            for cand_fmt in candidate_formats:
                if shape_tuple is None or len(shape_tuple) != 2:
                    verdicts[cand_fmt] = {"legal": False, "reason": "shape_unknown"}
                else:
                    verdict = check_format_applicability(
                        shape_tuple,
                        cand_fmt,
                        qname=qname,
                        target_profile=target_profile,
                    )
                    verdicts[cand_fmt] = {
                        "legal": bool(verdict.legal),
                        "reason": verdict.reason,
                        "detail": verdict.detail,
                    }
            entry["eight_bit_candidates"] = verdicts
        group_entry = coerced_groups.get(qname)
        if group_entry is not None:
            entry["serving_group"] = group_entry
        entries.append(entry)

    return {
        "counts": counts,
        "entries": entries,
        "target_profile": target_profile,
    }


def _production_cache_prefetch_assignment(
    assignment: dict[str, str],
    *,
    prefix: str | None = None,
    mode: str | None = None,
) -> int:
    """Prefetch this layer's rendered weights out of the production cache.

    ``mode`` mirrors ``ProductionWeightCache.prefetch_assignment(require=…)``:
    under ``require`` a cache that cannot supply the assignment is a hard
    failure. Every failure path here used to return ``0`` and the caller only
    logged when it prefetched something, so a TOTAL miss was invisible and the
    export silently went NVMe-bound, tensor by tensor (re-vet R24 / debt D8).
    """
    mode = str(mode or _PRODUCTION_CACHE_PREFETCH_MODE or "warn").lower()
    require = mode == "require"

    def _refuse(reason: str) -> None:
        if require:
            raise RuntimeError(
                f"production-cache prefetch (require): {reason}. The export "
                "would fall back to per-tensor NVMe reads; pass "
                "--production-cache-prefetch warn to accept that explicitly."
            )

    cache = _PRODUCTION_WEIGHT_CACHE
    if cache is None or not hasattr(cache, "prefetch"):
        _refuse("no production weight cache is installed")
        return 0
    keys: list[tuple[str, str]] = []
    quantized = 0
    for qname, fmt in assignment.items():
        if prefix is not None and not (qname == prefix or qname.startswith(prefix + ".")):
            continue
        cache_fmt = str(fmt).upper()
        if _native_render_codec._canonical_export_format(cache_fmt) == "BF16":
            continue
        quantized += 1
        key = _production_cache_lookup_key(qname, cache_fmt)
        if key is not None:
            keys.append(key)
    if not keys:
        # No quantized entries under this prefix is legitimate (an all-BF16
        # layer); quantized entries with no cache keys is a total miss.
        if quantized:
            _refuse(
                f"{quantized} quantized entries under prefix {prefix!r} "
                "resolved to zero production-cache keys"
            )
        return 0
    loaded = int(cache.prefetch(keys, max_workers=_PRODUCTION_CACHE_PREFETCH_WORKERS))
    if loaded == 0:
        _refuse(
            f"{len(keys)} cache keys under prefix {prefix!r} loaded nothing"
        )
    return loaded











def _export_render_options(cache=None) -> NativeRenderLevers:
    """Use the recorded numerical recipe for a cached native re-pack."""
    options = _RENDER_LEVERS or NativeRenderLevers.resolve()
    if cache is None:
        cache = _PRODUCTION_WEIGHT_CACHE
    if cache is not None:
        recorded = NativeRenderLevers.from_record(cache.levers)
        if options.nvfp4_export_match_render_scale:
            return recorded
    return options


def _pack_production_cached_2d(
    linear_name: str,
    fmt: str,
    *,
    nvfp4_global_real_override: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor] | None:
    """Pack a pre-rendered production weight for export.

    ProductionWeightCache stores dequantized weights after the production
    numerical passes. For export we only need to re-pack those weights into the
    native compressed-tensors layout; running GPTQ/scale-sweep again would
    measure a different artifact.
    """
    cache = _PRODUCTION_WEIGHT_CACHE
    if cache is None:
        return None
    cache_fmt = str(fmt).upper()
    fmt = _native_render_codec._canonical_export_format(cache_fmt)
    key = _production_cache_lookup_key(linear_name, cache_fmt)
    if key is None:
        return None
    w = cache.get(key[0], key[1])
    if w is None:
        return None
    target_device = device or torch.device("cpu")
    if fmt == "NVFP4":
        w_work = w.to(device=target_device, dtype=torch.float32)
        wp, ws, wg = _native_render_codec.quantize_dequantize_nvfp4(
            w_work,
            group_size=16,
            global_real_override=nvfp4_global_real_override,
            levers=_export_render_options(cache),
        )
        input_scale = _native_render_codec._resolve_nvfp4_input_global_scale(target=linear_name, calibrated_scales=_INPUT_GLOBAL_SCALES)
        return {
            "weight_packed": wp,
            "weight_scale": ws,
            "weight_global_scale": wg.reshape(1) if wg.dim() == 0 else wg,
            "input_global_scale": torch.tensor(
                [float(input_scale)], dtype=torch.float32, device=target_device,
            ),
        }
    if fmt in _native_render_codec.MXFP8_EXPLICIT_FORMATS:
        # MXFP8_E4M3/MXFP8_E5M2 activation-aware renders choose explicit
        # E8M0 group scales.
        # The production cache stores dequantized weights for KL/polish, not
        # the uint8 scale tensor. Repacking a dequantized tensor can legally
        # choose a different E8M0 scale, so when the activation cache is
        # available we recompute from source weights through `_quantize_2d`.
        cache_levers = getattr(cache, "levers", {}) or {}
        if (
            (
                bool(cache_levers.get("scale_sweep", False))
                or bool(cache_levers.get("gptq", False))
                or bool(cache_levers.get("joint_scale_opt", False))
            )
            and _CACHED_ACTIVATIONS is not None
        ):
            return None
        w_work = w.to(device=target_device, dtype=torch.float32)
        dtype, max_value = _native_render_codec._fp8_element_dtype_and_max(fmt)
        q, qs = _native_render_codec.quantize_dequantize_mxfp8(
            w_work,
            group_size=32,
            element_dtype=dtype,
            element_max=max_value,
        )
        return {"weight": q, "weight_scale": qs}
    if fmt in {"FP8_E4M3", "FP8_E5M2"}:
        cache_levers = getattr(cache, "levers", {}) or {}
        if (
            (
                bool(cache_levers.get("scale_sweep", False))
                or bool(cache_levers.get("gptq", False))
            )
            and _CACHED_ACTIVATIONS is not None
        ):
            return None
        if fmt == "FP8_E5M2":
            return None
        w_work = w.to(device=target_device, dtype=torch.float32)
        q, qs = _native_render_codec.quantize_dequantize_fp8_dynamic(w_work)
        return {"weight": q, "weight_scale": qs}
    if fmt == "MXFP4":
        cache_levers = getattr(cache, "levers", {}) or {}
        if bool(cache_levers.get("gptq", False)) and _CACHED_ACTIVATIONS is not None:
            return None
        w_work = w.to(device=target_device, dtype=torch.float32)
        q, qs = _native_render_codec.quantize_dequantize_mxfp4(w_work, group_size=32)
        return {"weight_packed": q, "weight_scale": qs}
    if fmt == "BF16":
        return {"weight": w.to(device=target_device, dtype=torch.bfloat16)}
    return None


def _read_cached_packed_expert(
    experts_param_name: str,
    fmt: str,
    *,
    device: torch.device | None = None,
    cache: "ProductionWeightCache | None" = None,
) -> torch.Tensor | None:
    """Return the GPTQ-rendered 3-D dequant ``[E, out, in]`` for a packed-MoE
    expert tensor from the production cache, or ``None`` if absent.

    ``cache`` overrides the module-level production cache — the inline-render
    export path (``PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ``) passes a transient
    per-layer cache produced by ``fill_packed_expert_cache_entries`` so no
    588 GB dequant cache has to be materialized to disk first.

    Mirrors ``_pack_production_cached_2d`` but for packed experts: the cache
    stores the per-expert GPTQ dequant (in the model dtype, usually bf16).  The
    caller splits this into per-expert slices and re-packs each by re-deriving
    NVFP4 codes + per-group scales from the cached dequant — the SAME
    re-derive-from-dequant approximation the shipped 2-D path uses
    (``_pack_production_cached_2d``).  This is NOT bit-lossless (the re-derived
    group scales differ slightly from those used during the render, ~1e-3
    weight error), but it is the established production contract and within
    NVFP4 noise; served KL is the final arbiter.
    """
    if cache is None:
        cache = _PRODUCTION_WEIGHT_CACHE
    if cache is None:
        return None
    w = cache.get(experts_param_name, str(fmt).upper())
    if w is None:
        return None
    target_device = device or torch.device("cpu")
    return w.to(device=target_device, dtype=torch.float32)


def _packed_expert_input_global_scale(
    experts_param_name: str,
    *,
    cache: "ProductionWeightCache | None" = None,
) -> float | None:
    """Policy-resolved W4A4 input_global_scale for a packed-expert tensor,
    read from the production cache's per-param activation max_abs.

    Returns ``None`` when no cache/scale is available, so the caller falls back
    to ``DEFAULT_INPUT_GLOBAL_SCALE`` (only on the no-cache research path).

    ``cache`` overrides the module-level production cache (inline-render path).
    """
    if cache is None:
        cache = _PRODUCTION_WEIGHT_CACHE
    if cache is None:
        return None
    max_abs_map = getattr(cache, "activation_max_abs", None) or {}
    mx = max_abs_map.get(experts_param_name)
    if mx is None or float(mx) <= 0:
        return None
    return _native_render_codec._nvfp4_input_global_scale_from_max_abs(float(mx), levers=_export_render_options())


def _packed_expert_stage_attestation(
    experts_param_name: str,
    *,
    cache: "ProductionWeightCache | None" = None,
    profile=None,
) -> dict[str, Any] | None:
    """Attest one packed FusedMoE module's w13/w2 stages (ROADMAP K0.2).

    This legacy container deliberately publishes no
    ``execution_contracts.nvfp4_w4a4`` record: its activation scalars are
    optional/defaultable and claim no versioned fused contract.  It still
    must not emit a routed-MoE artifact whose two stages were not both
    calibrated.  The section is built on the shared ``routed_moe_stage``
    parser and the same calibration-source vocabulary, so stage identity,
    framing and digests stay comparable.

    Returns ``None`` for anything that is not a packed routed-expert stage.
    Raises when the sibling stage of the same FusedMoE module has no calibrated
    max-abs — the exact half-calibrated state that makes fused MoE fail closed
    at serving time.
    """
    parsed = _nvfp4_activation_contract.routed_moe_stage(
        experts_param_name, profile=profile
    )
    if parsed is None:
        return None
    module, stage = parsed
    if cache is None:
        cache = _PRODUCTION_WEIGHT_CACHE
    max_abs_map = (
        getattr(cache, "activation_max_abs", None) or {}
    ) if cache is not None else {}
    max_abs_by_target: dict[str, float] = {}
    for name, value in max_abs_map.items():
        found = _nvfp4_activation_contract.routed_moe_stage(
            str(name), profile=profile
        )
        if found is None or found[0] != module:
            continue
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            continue
        max_abs_by_target[str(name)] = float(value)
    staged = {
        _nvfp4_activation_contract.routed_moe_stage(
            name, profile=profile
        )[1]: name
        for name in sorted(max_abs_by_target)
    }
    missing = [
        s for s in _nvfp4_activation_contract.NVFP4_ROUTED_MOE_STAGES
        if s not in staged
    ]
    if missing:
        raise RuntimeError(
            f"[export-native] packed FusedMoE module {module!r} has a "
            f"calibrated {stage} activation scale but no calibrated "
            f"{missing} stage. A routed-MoE artifact must attest BOTH stages "
            "(w13 from the experts-module input, w2 from the routed "
            "intermediate); re-run build_production_cache with the packed "
            "experts in scope so the missing packed_expert_max_abs entry is "
            "recomputed."
        )
    policy = _nvfp4_activation_contract.resolve_input_global_scale_policy()
    return _nvfp4_activation_contract.build_routed_moe_stage_attestation(
        {
            name: _native_render_codec._nvfp4_input_global_scale_from_max_abs(value, levers=_export_render_options())
            for name, value in max_abs_by_target.items()
        },
        policy=policy,
        calibration_sources={
            name: (
                _nvfp4_activation_contract
                .CALIBRATION_SOURCE_PACKED_EXPERT_RENDER
            )
            for name in max_abs_by_target
        },
        profile=profile,
    )


# Module-level flag bundle that controls which activation-aware
# passes run when `_quantize_2d` is invoked from main()'s streaming
# loop. Kept as module-level state (mirroring _INPUT_GLOBAL_SCALES)
# so we don't have to thread 3 boolean kwargs through every call
# site — unit tests pass the flags directly via kwargs.
_PRODUCTION_WEIGHT_CACHE = None
# Research/A-B escape hatch (Codex-recommended): when set, packed experts skip
# the production-cache GPTQ read and the RTN-by-omission hard-fail, exporting
# source RTN instead. ONLY for the served RTN-vs-GPTQ expert A/B — NEVER a
# production path (NVFP4 experts under RTN is a severe quality regression).
_ALLOW_PACKED_EXPERT_RTN = (
    os.environ.get("PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN", "0") == "1"
)
# Inline packed-expert GPTQ render at export time. When set, and NO production
# weight cache is active, packed experts are rendered ON THE FLY during the
# streaming export — one layer's stack at a time — through the SAME
# ``fill_packed_expert_cache_entries`` batched-GPTQ path the cache builder uses,
# sourcing the experts-module input snapshot from the probe's activation cache.
# This is the 295B-MoE path (Tencent Hy3): a full dequant cache is ~588 GB and
# cannot coexist with the 557 GiB source on a 1.8 TB disk, so we never write it.
# Unset (default) preserves the RTN-by-omission hard-fail exactly.
_INLINE_EXPERT_GPTQ = (
    os.environ.get("PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ", "0") == "1"
)
_PRODUCTION_CACHE_FINGERPRINT: dict[str, object] | None = None
_PRODUCTION_CACHE_PREFETCH_WORKERS = 4
# "require" | "warn" (D8). run-pipeline.sh passes require on the native
# lane, matching VALIDATED_SOURCE_PREFETCH=require.
_PRODUCTION_CACHE_PREFETCH_MODE = "warn"


def _packed_expert_render_hist_label(
    fmt: str,
    *,
    is_bf16: bool,
    source_label: str,
    cached_3d: torch.Tensor | None,
) -> str:
    if is_bf16:
        return source_label
    return f"{fmt}+cached" if cached_3d is not None else f"{fmt}+rtn"


def _packed_expert_export_provenance() -> dict[str, object]:
    cache = _PRODUCTION_WEIGHT_CACHE
    metadata = dict(getattr(cache, "metadata", {}) or {}) if cache is not None else {}
    coverage = metadata.get("packed_expert_coverage")
    return {
        "rtn_escape_enabled": bool(_ALLOW_PACKED_EXPERT_RTN),
        "inline_expert_gptq_enabled": bool(_INLINE_EXPERT_GPTQ),
        "cache_has_packed_expert_coverage": coverage is not None,
        "cache_packed_expert_coverage": coverage or {},
    }


def _inline_expert_render_levers() -> dict[str, object]:
    """The complete immutable options used by the dense inline renderer."""
    return (_RENDER_LEVERS or NativeRenderLevers.resolve()).record()



def _dump_live_cuda_tensors(where: str, min_mb: float = 200.0) -> None:
    """Diagnostic (PRISMAQUANT_EXPORT_MEM_DUMP=1): aggregate every live CUDA
    tensor above ``min_mb`` by (shape, dtype) and print one referrer summary
    per group — the direct attribution for a cuda_alloc ramp that survives
    gc.collect()+empty_cache() (a live reference is not garbage; only its
    holder's name finds it)."""
    groups: dict[tuple, list] = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and not obj.is_meta:
                nbytes = obj.numel() * obj.element_size()
                if nbytes >= min_mb * 1024 * 1024:
                    groups.setdefault(
                        (tuple(obj.shape), str(obj.dtype)), []).append(obj)
        except Exception:
            continue
    total = sum(
        t.numel() * t.element_size() for ts in groups.values() for t in ts)
    print(f"[mem-dump] {where}: cuda_alloc="
          f"{torch.cuda.memory_allocated()/1024**3:.1f}G, "
          f"{sum(len(v) for v in groups.values())} tensors >= {min_mb:.0f}MB "
          f"({total/1024**3:.1f}G python-visible)", flush=True)
    import types as _types
    own_ids = {id(groups)} | {id(ts) for ts in groups.values()}

    def _chase_iterator(it, depth: int = 0) -> str:
        """Name the frame/generator that keeps an iterator alive."""
        if depth > 4:
            return "?"
        for r in gc.get_referrers(it):
            if id(r) in own_ids or isinstance(r, type):
                continue
            if isinstance(r, _types.FrameType):
                owner = ""
                for r2 in gc.get_referrers(r):
                    if isinstance(r2, _types.GeneratorType):
                        owner = " (suspended generator)"
                        break
                return (f"frame {r.f_code.co_filename.rsplit('/', 1)[-1]}:"
                        f"{r.f_lineno} {r.f_code.co_name}{owner}")
            if isinstance(r, _types.GeneratorType):
                fr = r.gi_frame
                return (f"generator {r.gi_code.co_name}"
                        f":{fr.f_lineno if fr else 'done'}")
            if type(r).__name__ in (
                    "enumerate", "list_iterator", "tuple_iterator",
                    "zip", "map", "filter", "chain", "islice",
                    "dict_itemiterator", "dict_valueiterator"):
                return f"{type(r).__name__} <- {_chase_iterator(r, depth+1)}"
            if isinstance(r, (list, tuple, dict)):
                return f"{type(r).__name__}(n={len(r)}) <- " + _chase_iterator(
                    r, depth + 1)
        return "no-gc-referrer"

    def _describe(r, depth: int = 0) -> str:
        if isinstance(r, dict):
            holder = ""
            if depth < 2:
                for r2 in gc.get_referrers(r):
                    if id(r2) in own_ids or isinstance(
                            r2, (_types.FrameType, type)):
                        continue
                    holder = f" held-by {_describe(r2, depth + 1)}"
                    break
            return f"dict(n={len(r)}){holder}"
        if isinstance(r, (list, tuple)):
            holder = ""
            if depth < 2:
                for r2 in gc.get_referrers(r):
                    if id(r2) in own_ids or isinstance(
                            r2, (_types.FrameType, type)):
                        continue
                    holder = f" held-by {_describe(r2, depth + 1)}"
                    break
            return f"{type(r).__name__}[{len(r)}]{holder}"
        if isinstance(r, _types.FrameType):
            return (f"frame {r.f_code.co_name}:"
                    f"{[k for k, v in r.f_locals.items()][:6]}")
        return type(r).__name__
    for (shape, dtype), ts in sorted(
            groups.items(),
            key=lambda kv: -sum(t.numel() * t.element_size()
                                for t in kv[1])):
        gb = sum(t.numel() * t.element_size() for t in ts) / 1024**3
        ptrs = sorted({t.untyped_storage().data_ptr() for t in ts})
        for i, t in enumerate(ts):
            refs: list[str] = []
            for r in gc.get_referrers(t):
                if id(r) in own_ids or isinstance(r, type):
                    continue
                if isinstance(r, _types.FrameType):
                    names = [k for k, v in r.f_locals.items() if v is t]
                    refs.append(
                        f"frame {r.f_code.co_filename.rsplit('/', 1)[-1]}:"
                        f"{r.f_lineno} {r.f_code.co_name} locals{names}")
                elif isinstance(r, dict):
                    keys = [str(k) for k, v in r.items() if v is t][:2]
                    refs.append(f"dict keys{keys} {_describe(r)}")
                elif isinstance(r, (list, tuple)):
                    refs.append(
                        f"{type(r).__name__}[{len(r)}] <- "
                        f"{_chase_iterator(r)}")
                else:
                    refs.append(
                        f"{_describe(r)} <- {_chase_iterator(r)}")
                if len(refs) >= 3:
                    break
            print(f"[mem-dump]   [{i}] {list(shape)} {dtype} = "
                  f"{t.numel()*t.element_size()/1024**3:.2f}G "
                  f"storage@{hex(t.untyped_storage().data_ptr())} "
                  f"refs: {refs}", flush=True)
        print(f"[mem-dump]   group {list(shape)} {dtype}: {len(ts)} tensors "
              f"{gb:.2f}G across {len(ptrs)} storages", flush=True)


def _inline_render_packed_expert_module(
    model: nn.Module,
    experts_qname: str,
    assignment: dict[str, str],
    profile,
) -> "ProductionWeightCache | None":
    """Render ONE packed-experts module's stack into a transient in-memory
    ``ProductionWeightCache`` at export time (no disk, no whole dequant cache).

    Gated by ``PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ`` and only active when no
    production weight cache is supplied. The experts-module input snapshot X is
    sourced from the probe's activation cache (``_CACHED_ACTIVATIONS`` keyed by
    the experts-module qname); routing is recomputed offline from X + the
    resident gate weight by ``fill_packed_expert_cache_entries`` — the SAME
    batched GPTQ path (``module_acts_override``) the streaming cache builder
    uses, so the rendered dequant is identical to what the cache path would have
    produced for this (stack, format, activations).

    Returns the transient cache (its ``.weights`` hold the ``(full, fmt)`` 3-D
    dequant and ``.activation_max_abs`` the calibrated W4A4 input scale), or
    ``None`` when the gate is off, a production cache is active, or no
    activation snapshot exists for this module (the caller then decides whether
    to hard-fail or fall through to the RTN research path).
    """
    if not _INLINE_EXPERT_GPTQ:
        return None
    if _PRODUCTION_WEIGHT_CACHE is not None or _ALLOW_PACKED_EXPERT_RTN:
        return None
    if _CACHED_ACTIVATIONS is None:
        return None
    # Activation-residency landmine: _LazyActivationCache.get() returns a
    # CPU-resident fp32 tensor; fill_packed_expert_cache_entries' override path
    # reshapes + moves it to the compute device itself (derive_per_expert_
    # activations runs the router on `device`), so no manual .to() here.
    X = _CACHED_ACTIVATIONS.get(experts_qname)
    if X is None or X.numel() == 0:
        return None

    from .production_weight_cache import (
        ProductionWeightCache,
        _resolve_production_render_levers,
        fill_packed_expert_cache_entries,
    )

    # Resolve through the SAME contract the cache builder uses so the transient
    # cache records the identical nvfp4_scale_rule / damp provenance — the
    # export re-derive keys the NVFP4 codes off that recorded rule (M19/M2), so
    # a mismatch would flip the packed bytes vs the prebuilt-cache path.
    levers = _resolve_production_render_levers(_inline_expert_render_levers())
    transient = ProductionWeightCache(
        weights={},
        levers=dict(levers),
        activation_max_abs={},
        failed={},
        cache_dir=None,  # in-memory only — never spill the dequant to disk
        metadata={},
    )
    fill_packed_expert_cache_entries(
        transient,
        model,
        None,  # calib_ids unused: module_acts_override supplies X
        render_assignment=assignment,
        levers=levers,
        profile=profile,
        cache_dir=None,
        render_mode="batched",
        module_acts_override={experts_qname: X},
        progress=False,
    )
    return transient




def _explicit_regex(name: str) -> str:
    """Anchor a Linear name as a compressed-tensors regex target."""
    return f"re:^{name.replace('.', '[.]')}$"


# Matches a vLLM-internal per-expert Linear qname, e.g.
#   model.layers.10.mlp.experts.0.gate_proj
# (Qwen3.5 / MiniMax / Gemma4 layouts all normalize to this form via the
# profile's `to_vllm_internal_name`.)
_PER_EXPERT_LINEAR_RE = re.compile(
    r"^(?P<prefix>.*[.])layers[.](?P<L>\d+)[.](?P<inner>.*mlp)[.]"
    r"experts[.](?P<E>\d+)[.](?P<proj>[^.]+)$"
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

    Within a (layer, proj) bucket we always emit a `[0-9]+` expert-index
    wildcard. The exporter relies on the allocator/export coherence gates to
    keep every expert in a layer on one scheme for a given projection; sparse
    within-layer expert subsets are not represented as separate alternations.
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
    for (prefix, L, inner, proj), _experts in sorted(bucketed.items()):
        prefix_r = prefix.replace(".", "[.]")
        inner_r = inner.replace(".", "[.]")
        # Always emit the [0-9]+ wildcard for the expert position. vLLM's
        # FusedMoE.get_moe_method probes the synthetic name `experts.0.X_proj`
        # against this regex, and every expert in a layer shares the same
        # scheme, so wildcarding is semantically correct.
        expr = "[0-9]+"
        collapsed.append(
            f"re:^{prefix_r}layers[.]{L}[.]{inner_r}[.]experts[.]{expr}"
            f"[.]{proj}$"
        )

    out = (
        [_explicit_regex(n) for n in sorted(passthrough)]
        + sorted(preformed_regex)
        + sorted(collapsed)
    )
    return out


# ---------------------------------------------------------------------------
# Module / parameter discovery — the probe's own helpers
# (``sensitivity_probe``), so the export sees the same units as the probe.
# ---------------------------------------------------------------------------
def _packed_expert_projection_names(profile, param_name: str) -> tuple[str, ...]:
    if profile is None:
        try:
            from .model_profiles import DefaultProfile
            profile = DefaultProfile()
        except Exception:
            profile = None
    if profile is not None:
        try:
            projections = tuple(profile.packed_expert_projection_names(param_name))
            if projections:
                return projections
        except Exception:
            pass
    return (str(param_name),)


def _vllm_moe_scheme_projection_names(profile, param_name: str) -> tuple[str, ...]:
    """vLLM FusedMoE scheme-probe / ignore projection names for a packed
    expert param — the canonical ``gate_proj``/``up_proj``/``down_proj``
    that vLLM's ``get_moe_method`` and ignore matching dispatch on,
    regardless of the on-disk weight names. Used ONLY for config_groups
    targets + ignore regexes; weight export still uses the on-disk names.
    See ModelProfile.vllm_fused_moe_scheme_projection_names."""
    if profile is None:
        try:
            from .model_profiles import DefaultProfile
            profile = DefaultProfile()
        except Exception:
            profile = None
    if profile is not None:
        getter = getattr(profile, "vllm_fused_moe_scheme_projection_names", None)
        if callable(getter):
            try:
                projections = tuple(getter(param_name))
                if projections:
                    return projections
            except Exception:
                pass
    return _packed_expert_projection_names(profile, param_name)


def _all_packed_expert_projection_names(profile) -> tuple[str, ...]:
    projections: list[str] = []
    seen: set[str] = set()
    for param_name in sorted(_packed_expert_param_name_set(profile)):
        for projection in _packed_expert_projection_names(profile, param_name):
            if projection in seen:
                continue
            projections.append(projection)
            seen.add(projection)
    return tuple(projections)


def _split_packed_expert_tensor(
    packed_param: torch.Tensor,
    param_name: str,
    profile,
) -> list[tuple[str, torch.Tensor]]:
    projections = _packed_expert_projection_names(profile, param_name)
    if projections == (param_name,):
        return [(param_name, packed_param)]
    rows = int(packed_param.shape[1])
    n_parts = len(projections)
    if rows % n_parts != 0:
        raise ValueError(
            f"packed expert tensor {param_name!r} with rows={rows} cannot "
            f"split evenly into configured projections {projections!r}"
        )
    chunk = rows // n_parts
    return [
        (proj_name, packed_param[:, i * chunk:(i + 1) * chunk, :])
        for i, proj_name in enumerate(projections)
    ]


# ---------------------------------------------------------------------------
# Fused-sibling joint global_scale (for dense Linears)
# ---------------------------------------------------------------------------
# vLLM's compressed_tensors_w4a4_nvfp4.process_weights_after_loading warns
# (and reduces accuracy) when q/k/v or gate/up have different
# weight_global_scale. We compute the max over each fused group's natural
# global_scale and force every sibling to use it.
#
from .name_projection import fused_sibling_group_key


def _unify_input_global_scales_across_fused_siblings(
    scales: dict[str, float],
    *,
    profile=None,
) -> dict[str, float]:
    """Compatibility delegate for the shared conservative scale join."""

    return _nvfp4_activation_contract.unify_fused_sibling_input_global_scales(
        scales,
        profile=profile,
        tolerate_profile_errors=False,
        diagnostic_prefix="[export-stream]",
    )


def _compute_nvfp4_joint_global(
    model: nn.Module,
    assignment: dict[str, str],
    *,
    profile=None,
    levers: NativeRenderLevers,
) -> dict[str, torch.Tensor]:
    """Pre-pass over the model: for each fused-sibling group whose
    members are all assigned to NVFP4, compute the joint global_real
    (max across siblings). Return a dict mapping each sibling's qname
    to the shared global_real tensor.

    ``levers`` is the recipe the siblings render under.  The resident and
    streamed cache fills pass their recorded recipe and export passes
    ``_export_render_options()`` (that same record under match-render-scale),
    so the global a sibling is rendered on is the one its shipped codes and
    FP8 block scales are re-derived under."""
    # Bucket siblings by (parent_prefix, kind). Missing siblings are
    # OK — vLLM's loader handles partial fusion fine.
    groups: dict[str, list[tuple[str, nn.Linear]]] = {}
    for qname, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        live_to_recipe = getattr(profile, "live_to_recipe_name", None)
        if callable(live_to_recipe):
            try:
                recipe_qname = live_to_recipe(qname)
            except Exception:
                recipe_qname = qname
        else:
            recipe_qname = qname
        if _native_render_codec._canonical_export_format(assignment.get(recipe_qname, "BF16")) != "NVFP4":
            continue
        g = fused_sibling_group_key(recipe_qname, profile=profile)
        if g is None:
            continue
        groups.setdefault(g, []).append((recipe_qname, mod))

    out: dict[str, torch.Tensor] = {}
    for _group_key, siblings in groups.items():
        # Need every sibling to also be NVFP4 — otherwise vLLM allocates
        # the fused tensor under a different scheme and our joint scale
        # wouldn't apply consistently. The allocator's promote_fused
        # already enforces this; here we just verify and skip on partial
        # consistency (defensive — a mixed-format fused group is a bug
        # upstream of the export and would fail the load anyway).
        candidates = [
            _native_render_codec.compute_nvfp4_global_real(mod.weight.detach().float(), levers=levers)
            for _, mod in siblings
        ]
        joint = torch.stack(candidates).max()
        for qname, _ in siblings:
            out[qname] = joint
    return out


# ---------------------------------------------------------------------------
# Quantization pipeline
# ---------------------------------------------------------------------------


def _host_mem_available_bytes() -> int:
    try:
        return io_spans.mem_available_bytes()
    except (OSError, RuntimeError):
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
def _compute_layer_joint_nvfp4(layer_mod: nn.Module,
                               layer_qname: str,
                               assignment: dict[str, str],
                               profile,
                               *, levers: NativeRenderLevers) -> dict[str, torch.Tensor]:
    """Return {recipe_key -> joint global scale} for NVFP4 fused-sibling
    groups inside this decoder layer. Only keys assigned NVFP4 get an
    override entry; the rest compute per-Linear scales at quantize time.

    Semantically equivalent to a scoped `_compute_nvfp4_joint_global`
    across just this layer's modules."""
    groups: dict[str, list[tuple[str, str, nn.Linear]]] = defaultdict(list)
    for sub_name, mod in layer_mod.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        full = f"{layer_qname}.{sub_name}" if sub_name else layer_qname
        try:
            recipe_key = profile.live_to_recipe_name(full)
        except Exception:
            recipe_key = full
        group_key = fused_sibling_group_key(recipe_key, profile=profile)
        if group_key is None:
            continue
        groups[group_key].append((full, recipe_key, mod))

    out: dict[str, torch.Tensor] = {}
    for _group_key, members in groups.items():
        fqn_fmt = []
        for full, recipe_key, mod in members:
            fmt = assignment.get(recipe_key)
            fqn_fmt.append((full, recipe_key, fmt, mod))
        fmts = {_native_render_codec._canonical_export_format(f) for _, _, f, _ in fqn_fmt}
        if fmts != {"NVFP4"}:
            continue
        candidates = [
            _native_render_codec.compute_nvfp4_global_real(mod.weight.detach().float(),
                                      group_size=16, levers=levers)
            for _, _, _, mod in fqn_fmt
        ]
        joint = torch.stack(candidates).max()
        for full, recipe_key, _, _ in fqn_fmt:
            out[recipe_key] = joint
    return out


_SAFETENSORS_DTYPE_TO_TORCH = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def _build_source_dtype_map(
    model_to_shard: dict[str, str],
    model_to_ckpt: dict[str, str],
) -> dict[str, torch.dtype]:
    """Return live tensor qname -> dtype from source safetensors metadata."""
    from safetensors import safe_open

    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for model_name, shard in model_to_shard.items():
        by_shard[shard].append((model_name, model_to_ckpt[model_name]))

    out: dict[str, torch.dtype] = {}
    for shard, pairs in by_shard.items():
        with safe_open(shard, framework="pt") as f:
            keys = set(f.keys())
            for model_name, ckpt_name in pairs:
                if ckpt_name not in keys:
                    continue
                label = f.get_slice(ckpt_name).get_dtype()
                dtype = _SAFETENSORS_DTYPE_TO_TORCH.get(label)
                if dtype is not None:
                    out[model_name] = dtype
    return out


def _dtype_hist_label(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "BF16"
    if dtype == torch.float16:
        return "FP16"
    if dtype == torch.float32:
        return "FP32"
    if dtype == torch.float64:
        return "FP64"
    if dtype == torch.float8_e4m3fn:
        return "FP8_E4M3"
    if dtype == torch.float8_e5m2:
        return "FP8_E5M2"
    return str(dtype).replace("torch.", "").upper()


def _passthrough_dtype(
    qname: str,
    source_dtype: torch.dtype | None = None,
    *,
    fallback_dtype: torch.dtype | None = None,
) -> torch.dtype:
    """Pick the storage dtype for a passthrough (non-quantized) param.

    Passthrough means source-preserving. Do not silently upcast norms or
    other parameters here; if a future recipe wants FP32 norms, it should
    request that as an explicit transform and record it in the manifest.
    """
    if source_dtype is not None:
        return source_dtype
    if fallback_dtype is not None:
        return fallback_dtype
    raise ValueError(f"missing source dtype for passthrough tensor {qname}")


def _passthrough_tensor(
    qname: str,
    tensor: torch.Tensor,
    source_dtype_by_name: dict[str, torch.dtype] | None = None,
) -> tuple[torch.Tensor, str]:
    dtype = _passthrough_dtype(
        qname,
        None if source_dtype_by_name is None else source_dtype_by_name.get(qname),
        fallback_dtype=tensor.dtype,
    )
    return tensor.detach().to(dtype).cpu(), _dtype_hist_label(dtype)


# NOTE: `_init_rotary_inplace` is imported from `streaming_model` (single
# source of truth). It includes the profile-driven `init_rotaries` dispatch
# for multi-layer-type rotaries (DSv4/Gemma3/Gemma4); a stale duplicate here
# previously lacked it and would crash Gemma4 export at rotary init
# (KeyError: None on rope_parameters[None]).


def _build_fp8_source_map(
    model_path: str, *, profile=None, multimodal: bool = False,
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
    if profile is not None:
        pairs_fn = getattr(profile, "fp8_scale_pairs", None)
        if callable(pairs_fn):
            explicit = pairs_fn(model_path)
            if explicit is not None:
                return {
                    (key[:-7] if key.endswith(".weight") else key): value
                    for key, value in explicit.items()
                }

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

    def _rename_weight(k: str) -> str | None:
        weight_key = f"{k}.weight"
        if profile is not None:
            mapper = getattr(profile, "checkpoint_to_live_name", None)
            if callable(mapper):
                try:
                    live_weight = mapper(weight_key, multimodal=multimodal)
                except TypeError:
                    live_weight = mapper(weight_key)
                except Exception:
                    live_weight = None
                if live_weight is None:
                    return None
                live_weight = str(live_weight)
                return (
                    live_weight[:-7]
                    if live_weight.endswith(".weight")
                    else live_weight
                )

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
                live_base = _rename_weight(ck_base)
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


def _recipe_name_for_live_qname(qname: str, profile) -> str:
    if profile is None:
        return qname
    mapper = getattr(profile, "live_to_recipe_name", None)
    if not callable(mapper):
        return qname
    try:
        return str(mapper(qname))
    except Exception:
        return qname


def _fp8_source_passthrough_recipe_keys(
    fp8_source_map: dict[str, tuple[str, str]],
    source_dtype_by_name: dict[str, torch.dtype],
    profile,
) -> set[str]:
    """Return recipe keys that can be emitted as source-native FP8 bytes."""
    out: set[str] = set()
    for live_base in fp8_source_map:
        source_weight_key = f"{live_base}.weight"
        source_weight_dtype = source_dtype_by_name.get(source_weight_key)
        if source_weight_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            continue
        out.add(_recipe_name_for_live_qname(live_base, profile))
    return out


def _fp8_source_config_overlay(
    model_path: str,
    assignment: dict[str, str],
    bf16_passthrough: set[str],
    profile,
) -> tuple[dict[str, str], set[str], set[str]]:
    """Mirror streaming materialization's FP8_SOURCE passthrough in config.

    Native-FP8 checkpoints load source weights as dequanted BF16 for render
    work, but BF16/unassigned source-FP8 Linears are emitted by copying the
    original FP8 bytes and scale sidecars. The compressed-tensors config must
    describe those emitted bytes as FP8_SOURCE instead of putting the same
    names in ignore.
    """
    fp8_source_map = _build_fp8_source_map(model_path, profile=profile)
    if not fp8_source_map:
        return dict(assignment), set(bf16_passthrough), set()

    from .layer_streaming import _build_weight_map

    weight_shard, weight_ckpt = _build_weight_map(model_path)
    source_dtype_by_name = _build_source_dtype_map(weight_shard, weight_ckpt)
    source_recipe_keys = _fp8_source_passthrough_recipe_keys(
        fp8_source_map,
        source_dtype_by_name,
        profile,
    )
    if not source_recipe_keys:
        return dict(assignment), set(bf16_passthrough), set()

    config_assignment = dict(assignment)
    overrides: set[str] = set()
    # Packed-expert parents in the assignment (e.g.
    # `model.layers.10.mlp.experts.gate_up_proj`) own their per-expert
    # leaves: the packed emit writes those bytes in the allocated format,
    # so a per-expert source-FP8 leaf (`...mlp.experts.7.up_proj`) must
    # NOT be re-described as FP8_SOURCE — the config would contradict the
    # emitted bytes (GLM-5.3: 168 such entries double-covered the NVFP4
    # experts with the FP8 scheme).
    packed_expert_prefixes = tuple(
        {k.rsplit(".", 1)[0] + "." for k in assignment
         if ".mlp.experts." in k or k.endswith(".mlp.experts")})
    _per_expert_leaf = re.compile(r"\.experts\.[0-9]+\.")
    for recipe_key in sorted(source_recipe_keys):
        if (_per_expert_leaf.search(recipe_key)
                and recipe_key.startswith(packed_expert_prefixes)):
            continue
        recipe_fmt = config_assignment.get(recipe_key)
        fmt = (
            _native_render_codec._canonical_export_format(recipe_fmt)
            if recipe_fmt is not None
            else None
        )
        if fmt is None or fmt == "BF16" or recipe_key in bf16_passthrough:
            config_assignment[recipe_key] = "FP8_SOURCE"
            overrides.add(recipe_key)

    config_bf16_passthrough = set(bf16_passthrough) - overrides
    return config_assignment, config_bf16_passthrough, overrides


def _compressed_tensor_key(base_name: str, suffix: str) -> str:
    return f"{base_name}.{suffix}"


def materialize_tensors_streaming(
    model_path: str,
    assignment: dict[str, str],
    *,
    profile,
    bf16_passthrough: set[str],
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
    offload_folder: str | None = None,
    tensor_sink: Callable[[dict[str, torch.Tensor]], None] | None = None,
    export_cache_dir: str | None = None,
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
    from prismaquant.model_runtime import initialize_model_runtime
    initialize_model_runtime()
    from transformers import AutoConfig, AutoModelForCausalLM

    from .layer_streaming import (
        _build_concat_merger,
        _build_expert_packer,
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
    from .sensitivity_probe import stage_multimodal, stage_text_only
    # Canonical rotary init (profile-driven multi-layer-type dispatch).
    from .streaming_model import (
        _init_rotary_inplace,
        _mask_cuda_queries_during_meta_init,
        _skeleton_config_and_class,
    )

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
    # A family with no `<Arch>ForCausalLM` auto-route (glm5_next on
    # transformers 5.16) cannot build a text-only skeleton at all; the
    # profile declares that fact and the export inherits the same flip
    # `_build_streaming_context` applies for probe/cost streaming. The
    # visual tower stays on meta either way — export ships it via the
    # source-passthrough merge, never through the body walk.
    multimodal_skeleton = bool(profile.requires_multimodal_skeleton())
    if multimodal_skeleton:
        print("[export-stream] profile has no text-only skeleton route; "
              "using the multimodal construction", flush=True)
        staged = stage_multimodal(model_path)
    else:
        staged = stage_text_only(model_path)
    config = AutoConfig.from_pretrained(staged, trust_remote_code=True)
    config, model_cls = _skeleton_config_and_class(
        config, multimodal=multimodal_skeleton,
        log_prefix="[export-stream]")
    # _init_weights is globally no-op'd by prismaquant.__init__'s
    # _polyfill_transformers (wasted work + transformers-5.x compat
    # landmine on remote modeling files).
    with _mask_cuda_queries_during_meta_init("[export-stream]"):
        with init_empty_weights():
            if model_cls is AutoModelForCausalLM:
                model = AutoModelForCausalLM.from_config(
                    config, trust_remote_code=True)
            else:
                model = model_cls._from_config(config)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base_model, layers = _get_layer_list(model)
    base_prefix = _resolve_base_prefix(model, base_model)
    num_layers = len(layers)
    layers_prefix = f"{base_prefix}.layers." if base_prefix else "layers."

    weight_shard, weight_ckpt = _build_weight_map(
        model_path, multimodal=multimodal_skeleton)
    source_dtype_by_name = _build_source_dtype_map(weight_shard, weight_ckpt)
    # Persistent buffers are emitted verbatim (3e), so the read must not
    # narrow them to the parameter dtype the way `_read_layer_to_device`
    # narrows weights: `_passthrough_tensor` restores the DECLARED dtype
    # but cannot restore discarded values, and for a router bias the dtype
    # is arithmetic rather than storage -- a BF16 score plus an FP32 bias
    # sums in FP32, a BF16 bias rounds before top-k. Same map the resident
    # and streaming source loads build (`layer_streaming._materialize`,
    # `StreamingContext.buffer_dtypes`); walked once here rather than per
    # layer because the skeleton's buffer declarations never change.
    declared_buffer_dtypes = {
        name: value.dtype
        for name, value in model.named_buffers(remove_duplicate=False)
    }
    # Per-expert -> packed-3D bridge for checkpoints that ship MoE experts
    # unfused while the live module is packed (driven by the model profile;
    # None for every other model). Keeps the exporter's source read aligned
    # with the streaming probe/cost path — a raw checkpoint exports without
    # an out-of-band pre-pack.
    expert_packer = _build_expert_packer(model, weight_ckpt)
    # Sibling bridge for checkpoints that store one live parameter as several
    # source tensors (transformers' `Concatenate(dim=...)` merges, declared as
    # the profile spec's `concat_merges`). Same reason as the expert packer:
    # the exporter must load exactly what the streaming probe/cost path loads.
    concat_merger = _build_concat_merger(model, weight_ckpt)
    # Emit-side inverse of `concat_merges`: the live module holds ONE
    # merged tensor (glm5_next `self_attn.conv1d` <- q/k/v_conv1d) whose
    # key never existed in the source checkpoint. Emitting the merged
    # spelling would ship a key the serving runtime's loader does not
    # know and drop the three it expects, so the 3e passthrough skips the
    # merged live param and copies the SOURCE tensors verbatim instead.
    concat_groups = tuple(profile.concat_merge_groups())
    _src_tensor_index: dict[str, Path] | None = None

    def _read_source_tensor_verbatim(ckpt_key: str) -> torch.Tensor:
        nonlocal _src_tensor_index
        from safetensors import safe_open
        if _src_tensor_index is None:
            _src_tensor_index = {}
            src_root = Path(model_path)
            index_path = src_root / "model.safetensors.index.json"
            if index_path.exists():
                with open(index_path) as fh:
                    for key, shard in json.load(fh)["weight_map"].items():
                        _src_tensor_index[key] = src_root / shard
            else:
                for shard in sorted(src_root.glob("*.safetensors")):
                    with safe_open(str(shard), framework="pt") as sf:
                        for key in sf.keys():
                            _src_tensor_index[key] = shard
        shard_path = _src_tensor_index.get(ckpt_key)
        if shard_path is None:
            raise KeyError(
                f"[export-stream] concat-merge source tensor {ckpt_key!r} "
                f"not present in the source checkpoint — the profile's "
                f"concat_merges declaration disagrees with the shards.")
        with safe_open(str(shard_path), framework="pt") as sf:
            return sf.get_tensor(ckpt_key)
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
    fp8_source_map = _build_fp8_source_map(model_path, profile=profile)
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
    # These are resident on `device` already. Emit as source-dtype
    # passthrough UNLESS `lm_head` (or similar) is explicitly in the
    # assignment.
    t_head = time.time()

    def _emit_head_param(full_qname: str, param: nn.Parameter):
        recipe_key = profile.live_to_recipe_name(full_qname)
        # Recipe keys are module qnames (e.g. "lm_head"), not parameter
        # qnames ("lm_head.weight"). Strip the trailing `.weight` so the
        # assignment lookup hits — otherwise head params always fall
        # through to source-dtype passthrough regardless of what the
        # allocator chose for them.
        if recipe_key.endswith(".weight"):
            recipe_key = recipe_key[:-len(".weight")]
        recipe_fmt = assignment.get(recipe_key)
        fmt = recipe_fmt
        # Respect the passthrough set (e.g. `--ignore lm_head`) even if
        # the allocator assigned NVFP4/MXFP8_E4M3/MXFP8_E5M2 to this head module. See
        # the --ignore docstring for why lm_head is passthrough by
        # default despite vLLM rejecting quantized ParallelLMHead.
        if recipe_key in bf16_passthrough:
            fmt = "BF16"
        if fmt is not None:
            fmt = _native_render_codec._canonical_export_format(fmt)
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
            compressed = _pack_production_cached_2d(
                recipe_key,
                recipe_fmt if recipe_fmt is not None else fmt,
                nvfp4_global_real_override=joint,
                device=device,
            )
            if compressed is None:
                compressed = _native_render_codec._quantize_2d(
                    param.detach().float(), fmt,
                    nvfp4_global_real_override=joint,
                    linear_name=recipe_key,
                 cached_activations=(_CACHED_ACTIVATIONS.get(recipe_key) if _CACHED_ACTIVATIONS is not None else None), input_global_scale_override=_native_render_codec._resolve_nvfp4_input_global_scale(target=recipe_key, calibrated_scales=_INPUT_GLOBAL_SCALES), levers=_export_render_options())
            for suffix, t in compressed.items():
                base_name = (full_qname[:-len(".weight")]
                             if full_qname.endswith(".weight")
                             else full_qname)
                out_key = _compressed_tensor_key(base_name, suffix)
                out[out_key] = t.cpu()
            hist[("head", fmt)] += 1
        else:
            out[full_qname], label = _passthrough_tensor(
                full_qname, param, source_dtype_by_name)
            hist[("head_passthrough", label)] += 1

    for name, p in model.named_parameters():
        if p.is_meta:
            continue  # only head/embed/norm/lm_head resident here
        if name.startswith(layers_prefix):
            continue
        _emit_head_param(name, p)

    for mod_name, mod in model.named_modules():
        non_persistent = getattr(mod, "_non_persistent_buffers_set", set())
        for buf_name, buf in mod.named_buffers(recurse=False):
            if buf_name in non_persistent:
                continue
            if buf.is_meta:
                continue
            full = f"{mod_name}.{buf_name}" if mod_name else buf_name
            if full.startswith(layers_prefix):
                continue
            if full in out:
                continue
            out[full], label = _passthrough_tensor(
                full, buf, source_dtype_by_name)
            hist[("head_buffer", label)] += 1
    print(f"[export-stream] head+embed+norm+lm_head passthrough: "
          f"{time.time()-t_head:.1f}s  keys={len(out)}", flush=True)
    if tensor_sink is not None:
        tensor_sink(out)
        out = {}

    # ----- 3. Per-layer streaming quantize loop -----
    # v25: per-layer cache. When --export-cache-dir is set, each
    # layer's emitted tensor dict is torch.save'd to a per-layer file
    # AFTER quantization succeeds. On a restart the loop checks each
    # layer's cache file and SKIPS the quantization work for any layer
    # already cached — instead loads the saved dict and replays it
    # into tensor_sink. Recovers full progress from a mid-flight kill.
    cache_path = Path(export_cache_dir) if export_cache_dir else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
        # The fingerprint is computed ONLY when a cache dir was asked for, so
        # the source hash never touches an export that is not resumable.
        _admit_export_resume_cache(
            cache_path,
            _export_resume_fingerprint(
                assignment=assignment,
                model_path=model_path,
                dtype=dtype,
                declared_buffer_dtypes=declared_buffer_dtypes,
                extra_shard_paths=set(weight_shard.values()),
                digest_cache_path=cache_path / "source_identity_cache.json",
            ),
        )

    def _layer_cache_file(L: int) -> Path | None:
        return None if cache_path is None else cache_path / f"layer_{L:03d}.pt"

    t_layers = time.time()
    cache_hits = 0
    for L in range(num_layers):
        if tensor_sink is not None:
            out = {}
        layer_t0 = time.time()
        layer_qname = f"{layers_prefix}{L}".rstrip(".")
        if layer_qname.endswith("."):
            layer_qname = layer_qname[:-1]
        if _PRODUCTION_WEIGHT_CACHE is not None:
            layer_recipe_prefix = profile.live_to_recipe_name(layer_qname)
            prefetched = _production_cache_prefetch_assignment(
                assignment,
                prefix=layer_recipe_prefix,
            )
            if prefetched and (L % 4 == 0 or L == num_layers - 1):
                print(
                    f"[export-stream] layer {L:02d} production-cache "
                    f"prefetch={prefetched}",
                    flush=True,
                )

        # v25: cache hit — skip quantization, replay cached tensor dict.
        cf = _layer_cache_file(L)
        if cf is not None and cf.exists():
            cached = torch.load(str(cf), weights_only=False, map_location="cpu")
            if tensor_sink is not None:
                tensor_sink(cached)
            else:
                out.update(cached)
            cache_hits += 1
            if L % 4 == 0 or L == num_layers - 1:
                print(f"[export-stream] layer {L:02d}  CACHED "
                      f"keys={len(cached)}", flush=True)
            del cached
            continue

        # 3a. Load layer from safetensors (direct to device). When
        # `fp8_scale_inv_map` is non-empty, the loader applies the
        # 128x128 block dequant inline, so `mod.weight` receives the
        # true dequanted weight rather than raw fp8 codes cast to bf16.
        load_t0 = time.time()
        tensors = _read_layer_to_device(
            f"{layers_prefix}{L}.", weight_shard, weight_ckpt, dtype, device,
            fp8_scale_inv_map=fp8_scale_inv_map, pack_experts=expert_packer,
            merge_concat=concat_merger,
            buffer_dtypes=declared_buffer_dtypes)
        resolver = _build_install_resolver(model, layer_qname)
        _fast_install(resolver, tensors, device, model=model)
        load_s = time.time() - load_t0

        layer_mod = model.get_submodule(layer_qname)

        # 3b. Joint NVFP4 scales across fused siblings in this layer.
        # M2: computed under the render's recorded scale rule so the
        # fused joint-global pre-pass is consistent with the
        # match-render-scale re-derive (_pack_production_cached_2d).
        joint_globals = _compute_layer_joint_nvfp4(
            layer_mod, layer_qname, assignment, profile,
            levers=_export_render_options(),
        )

        # 3c. Emit Linears.
        covered: set[str] = set()
        linear_count = 0
        grouped_linears: dict[
            tuple[str, tuple[int, int]],
            list[tuple[str, str, str, nn.Linear]]  # (full, emit_full, recipe_key, mod)
        ] = defaultdict(list)
        # Batch same-shape NVFP4 Linears when act-aware passes
        # (GPTQ / scale_sweep) are on. Default ON; set
        # PRISMAQUANT_BATCHED_NVFP4_EXPORT=0 to force the slower
        # per-Linear `_quantize_2d` path for bit-exact A/Bs.
        grouped_nvfp4_batched: dict[
            tuple[int, int],
            list[tuple[str, str, str, nn.Linear]]
        ] = defaultdict(list)
        # v26: default ON.
        _batched_env_on = _export_render_options().batched_nvfp4_export
        _batched_nvfp4_enabled = (
            _batched_env_on
            and (_export_render_options().gptq or _export_render_options().scale_sweep)
            and _CACHED_ACTIVATIONS is not None
        )

        for sub_name, mod in layer_mod.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            linear_count += 1
            full = f"{layer_qname}.{sub_name}"
            emit_full = full

            recipe_key = profile.live_to_recipe_name(full)
            recipe_fmt = assignment.get(recipe_key)
            fmt = _native_render_codec._canonical_export_format(recipe_fmt) if recipe_fmt is not None else None
            source_weight_key = f"{full}.weight"
            source_weight_dtype = source_dtype_by_name.get(source_weight_key)
            source_is_fp8_scaled = (
                source_weight_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                and source_weight_key in fp8_scale_inv_map
            )
            if (source_is_fp8_scaled
                    and (fmt is None
                         or fmt == "BF16"
                         or recipe_key in bf16_passthrough)):
                fmt = "FP8_SOURCE"
            if fmt is None:
                # No assignment -> source-dtype passthrough.
                if not mod.weight.is_meta:
                    out[f"{emit_full}.weight"], label = _passthrough_tensor(
                        source_weight_key, mod.weight, source_dtype_by_name)
                    if mod.bias is not None and not mod.bias.is_meta:
                        out[f"{emit_full}.bias"], _ = _passthrough_tensor(
                            f"{full}.bias", mod.bias, source_dtype_by_name)
                    hist[("linear", label)] += 1
                    covered.add(full)
                continue

            if fmt == "BF16" or recipe_key in bf16_passthrough:
                out[f"{emit_full}.weight"], label = _passthrough_tensor(
                    source_weight_key, mod.weight, source_dtype_by_name)
                if mod.bias is not None:
                    out[f"{emit_full}.bias"], _ = _passthrough_tensor(
                        f"{full}.bias", mod.bias, source_dtype_by_name)
                hist[("linear", label)] += 1
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
                # Runtime-pinned modules (e.g. GLM MLA attention
                # projections) are dequantized by the serving runtime at
                # load, keyed on the SOURCE scale name
                # (`weight_scale_inv`); everything else serves through
                # compressed-tensors and gets the CT name.
                scale_key = ("weight_scale_inv"
                             if profile.runtime_loads_source_fp8(full)
                             else "weight_scale")
                out[f"{emit_full}.{scale_key}"] = w_scale.to(
                    torch.float32).cpu().contiguous()
                if mod.bias is not None and not mod.bias.is_meta:
                    out[f"{emit_full}.bias"], _ = _passthrough_tensor(
                        f"{full}.bias", mod.bias, source_dtype_by_name)
                hist[("linear", "FP8_SOURCE")] += 1
                covered.add(full)
                continue

            override = joint_globals.get(recipe_key) if fmt == "NVFP4" else None
            cached_compressed = _pack_production_cached_2d(
                recipe_key,
                recipe_fmt if recipe_fmt is not None else fmt,
                nvfp4_global_real_override=override,
                device=device,
            )
            if cached_compressed is not None:
                for suffix, t in cached_compressed.items():
                    out[f"{emit_full}.{suffix}"] = t.cpu()
                if mod.bias is not None:
                    out[f"{emit_full}.bias"], _ = _passthrough_tensor(
                        f"{full}.bias", mod.bias, source_dtype_by_name)
                hist[("linear", f"{fmt}_PRODUCTION_CACHE")] += 1
                covered.add(full)
                continue

            if fmt in _native_render_codec.MXFP8_EXPLICIT_FORMATS and mod.weight.dim() == 2:
                shape = (int(mod.weight.shape[0]), int(mod.weight.shape[1]))
                grouped_linears[(fmt, shape)].append((full, emit_full, recipe_key, mod))
                continue

            # v23: route NVFP4 same-shape Linears through the batched
            # GPTQ + scale_sweep path when env-gated and act-aware.
            if (_batched_nvfp4_enabled
                    and fmt == "NVFP4"
                    and mod.weight.dim() == 2):
                shape = (int(mod.weight.shape[0]), int(mod.weight.shape[1]))
                grouped_nvfp4_batched[shape].append(
                    (full, emit_full, recipe_key, mod))
                continue

            compressed = _native_render_codec._quantize_2d(
                mod.weight.detach().float(), fmt,
                nvfp4_global_real_override=override,
                linear_name=recipe_key,
             cached_activations=(_CACHED_ACTIVATIONS.get(recipe_key) if _CACHED_ACTIVATIONS is not None else None), input_global_scale_override=_native_render_codec._resolve_nvfp4_input_global_scale(target=recipe_key, calibrated_scales=_INPUT_GLOBAL_SCALES), levers=_export_render_options())
            for suffix, t in compressed.items():
                out[f"{emit_full}.{suffix}"] = t.cpu()
            if mod.bias is not None:
                out[f"{emit_full}.bias"], _ = _passthrough_tensor(
                    f"{full}.bias", mod.bias, source_dtype_by_name)
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
                compressed_batch = _native_render_codec._quantize_2d_group_same_shape(stacked, fmt)
                del stacked
                for i, (full, emit_full, _recipe_key, mod) in enumerate(chunk):
                    for suffix, tensor in compressed_batch.items():
                        piece = tensor[i]
                        if suffix == "weight_global_scale":
                            piece = piece.reshape(1)
                        out[f"{emit_full}.{suffix}"] = piece.cpu()
                    if mod.bias is not None:
                        out[f"{emit_full}.bias"], _ = _passthrough_tensor(
                            f"{full}.bias", mod.bias, source_dtype_by_name)
                    hist[("linear", fmt)] += 1
                    covered.add(full)
                del compressed_batch

        # v23: batched NVFP4 emission for same-shape groups when
        # _batched_nvfp4_enabled. Mirrors the INT/MXFP8_E4M3 grouped path
        # above but routes through the activation-aware batched path.
        if grouped_nvfp4_batched:
            export_dev = torch.device(device)
            for shape, items in grouped_nvfp4_batched.items():
                # Re-use the same E-chunk sizing as the INT/MXFP8_E4M3 path
                # so memory peaks stay bounded.
                chunk_len = _export_vector_chunk_len(
                    shape, len(items), export_dev)
                for start in range(0, len(items), chunk_len):
                    chunk = items[start:start + chunk_len]
                    compressed_per_linear = _native_render_codec._quantize_2d_nvfp4_group_batched(
                        chunk, joint_globals, export_dev,
                        expert_chunk=chunk_len,
                     activations=_CACHED_ACTIVATIONS, input_global_scales=_INPUT_GLOBAL_SCALES, levers=_export_render_options())
                    for (full, emit_full, _recipe_key, mod), compressed in zip(
                        chunk, compressed_per_linear,
                    ):
                        for suffix, t in compressed.items():
                            out[f"{emit_full}.{suffix}"] = t.cpu()
                        if mod.bias is not None:
                            out[f"{emit_full}.bias"], _ = _passthrough_tensor(
                                f"{full}.bias", mod.bias, source_dtype_by_name)
                        hist[("linear", "NVFP4")] += 1
                        covered.add(full)


        # 3d. Emit packed MoE experts, scoped to this layer.
        packed_count = 0
        # Inline packed-expert renders (PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ):
        # one transient in-memory cache per experts module in THIS layer, built
        # on first use and freed at layer end. Peak stays ~one layer's stack.
        _inline_expert_caches: dict[str, "ProductionWeightCache | None"] = {}
        for sub_name, mod in layer_mod.named_modules():
            if not _is_packed_experts_module(mod, profile):
                continue
            packed_count += 1
            for pn in _packed_experts_param_names(mod, profile):
                experts_qname = (f"{layer_qname}.{sub_name}"
                                 if sub_name else layer_qname)
                full = f"{experts_qname}.{pn}"
                recipe_key = profile.live_to_recipe_name(full)
                fmt = assignment.get(recipe_key)
                if fmt is not None:
                    fmt = _native_render_codec._canonical_export_format(fmt)
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
                packed_param_src = getattr(mod, pn).detach()
                packed_param = packed_param_src.float()
                E, M, N = packed_param.shape
                proj_split = _split_packed_expert_tensor(
                    packed_param,
                    pn,
                    profile,
                )

                is_bf16 = fmt == "BF16" or full in bf16_passthrough
                disk_qname = profile.on_disk_expert_qname(experts_qname)
                should_split = profile.split_packed_experts_for_format(fmt)

                iter_experts = [(e, e) for e in range(E)]

                if not should_split:
                    out[f"{disk_qname}.{pn}"], label = _passthrough_tensor(
                        full, packed_param_src, source_dtype_by_name)
                    covered.add(full)
                    hist[("packed_moe", label if is_bf16 else fmt)] += 1
                    del packed_param, packed_param_src
                    continue

                # Inline render (no production cache): render this experts
                # module's stack on the fly into a transient cache, memoized so
                # gate_up + down share the one fill_packed_expert_cache_entries
                # call. Falls back to _PRODUCTION_WEIGHT_CACHE reads when the
                # gate is off (active_cache stays None -> module default). Done
                # BEFORE the scale-rule + joint-global pre-pass so the re-derive
                # keys off the inline render's recorded nvfp4_scale_rule.
                if not is_bf16 and experts_qname not in _inline_expert_caches:
                    _inline_expert_caches[experts_qname] = (
                        _inline_render_packed_expert_module(
                            model, experts_qname, assignment, profile))
                active_cache = _inline_expert_caches.get(experts_qname)

                # Per-expert joint global scale when NVFP4 splits gate+up.
                # M2: computed under the render's recorded scale rule so
                # the joint global is consistent with the re-derive below.
                packed_render_options = _export_render_options(active_cache)
                per_expert_joint: list[torch.Tensor | None] = [None] * E
                if fmt == "NVFP4" and len(proj_split) > 1:
                    for orig_e, _ in iter_experts:
                        cands = [
                            _native_render_codec.compute_nvfp4_global_real(
                                sp[orig_e].float(), group_size=16, levers=packed_render_options)
                            for _, sp in proj_split
                        ]
                        per_expert_joint[orig_e] = (
                            torch.stack(cands).max())

                # Pull the GPTQ-rendered dequant from the production cache (or
                # the inline transient cache above). Packed experts go through
                # the SAME deliberate render as 2-D Linears
                # (fill_packed_expert_cache_entries); export re-packs the cached
                # dequant per expert. RTN-by-omission on packed experts is a
                # severe NVFP4 quality regression, so when a cache is active (or
                # the inline gate is set) we HARD-FAIL rather than silently RTN.
                cached_3d = None
                cached_split = None
                if not is_bf16:
                    if not _ALLOW_PACKED_EXPERT_RTN:
                        cached_3d = _read_cached_packed_expert(
                            full, fmt, device=device, cache=active_cache)
                        if cached_3d is not None and active_cache is not None:
                            # The transient inline render is read exactly once
                            # per packed param — drop the CPU fp32 entry now
                            # (18G for a GLM-class gate_up stack) instead of
                            # holding it until layer teardown.
                            _k = active_cache.resolve_key(full, fmt)
                            if _k is not None:
                                active_cache.weights.pop(_k, None)
                    if cached_3d is None:
                        if _INLINE_EXPERT_GPTQ and not _ALLOW_PACKED_EXPERT_RTN:
                            raise RuntimeError(
                                f"[export-stream] packed expert {full} @ {fmt} "
                                f"could not be rendered inline "
                                f"(PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ=1) — no "
                                f"experts-module activation snapshot for "
                                f"{experts_qname} in the activation cache. RTN "
                                f"on NVFP4 experts is banned; supply an "
                                f"--activation-cache-dir whose probe covers "
                                f"every packed-experts module, set the expert to "
                                f"BF16, or set "
                                f"PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN=1 for an "
                                f"explicit research/A-B RTN export.")
                        if (_PRODUCTION_WEIGHT_CACHE is not None
                                and not _ALLOW_PACKED_EXPERT_RTN):
                            raise RuntimeError(
                                f"[export-stream] packed expert {full} @ {fmt} "
                                f"has no production-cache render. Non-BF16 "
                                f"packed experts MUST be rendered through the "
                                f"deliberate GPTQ+JSO path "
                                f"(fill_packed_expert_cache_entries) — RTN on "
                                f"NVFP4 experts is a silent quality regression "
                                f"and is banned. Re-run build_production_cache "
                                f"with the packed experts in scope, set the "
                                f"expert to BF16, or set "
                                f"PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN=1 for an "
                                f"explicit research/A-B RTN export.")
                        print(
                            f"[export-stream] WARNING: RTN-rendering packed "
                            f"expert {full} @ {fmt} "
                            f"({'PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN research/A-B export' if _ALLOW_PACKED_EXPERT_RTN else 'no production cache active'}). "
                            f"This is NOT a production path — NVFP4 experts "
                            f"need GPTQ+JSO.",
                            flush=True)
                    else:
                        cached_split = _split_packed_expert_tensor(
                            cached_3d, pn, profile)

                # Calibrated input_global_scale (W4A4 activation clip): one
                # per packed param, calibrated from the routed activations at
                # cache-build time (or the inline render). Without it experts
                # ship the 1.0 placeholder.
                expert_input_scale = _packed_expert_input_global_scale(
                    full, cache=active_cache)
                if (cached_3d is not None and expert_input_scale is None
                        and fmt == "NVFP4"):
                    raise RuntimeError(
                        f"[export-stream] packed expert {full} @ {fmt} has a "
                        f"cached render but no calibrated input_global_scale "
                        f"(would ship the 1.0 placeholder). The cache's "
                        f"packed_expert_max_abs sidecar is missing this entry "
                        f"— re-run build_production_cache so the scale is "
                        f"recomputed, or delete the expert shard to force a "
                        f"full re-render.")
                if expert_input_scale is not None and fmt == "NVFP4":
                    # K0.2: a calibrated stage may only ship alongside its
                    # attested sibling stage.
                    _packed_expert_stage_attestation(
                        full, cache=active_cache, profile=profile)

                # M2: re-derive under the render's RECORDED NVFP4 scale
                # rule (the dense _pack_production_cached_2d wrap, lifted
                # to packed experts) — a joint_mse-rendered expert
                # re-derived under the export-entry default (static_6)
                # cannot recover its codes.
                # Index-based on purpose: `enumerate` caches its last
                # yielded result tuple internally (CPython reuse), and a
                # surviving enumerate pinned the (proj_name, 9G-view)
                # tuple — and through it an 18G fp32 stack — across
                # layers (mem-dump attribution, GLM-5.3 probe runs).
                # Never pass tensor-bearing tuples through enumerate in
                # this function's scope.
                for pi in range(len(proj_split)):
                    proj_name = proj_split[pi][0]
                    sub_packed = proj_split[pi][1]
                    cached_sub = (
                        cached_split[pi][1]
                        if cached_split is not None else None
                    )
                    for orig_e, new_e in iter_experts:
                        base = f"{disk_qname}.{new_e}.{proj_name}"
                        if is_bf16:
                            expert_2d = sub_packed[orig_e]
                            out[f"{base}.weight"], label = (
                                _passthrough_tensor(
                                    full, expert_2d,
                                    source_dtype_by_name))
                        else:
                            # Re-pack the GPTQ-rendered dequant when
                            # cached (re-derives codes from the dequant,
                            # same ~1e-3 approximation as the 2-D path —
                            # not bit-lossless); fall back to source only
                            # on the no-cache warning.
                            expert_2d = (
                                cached_sub[orig_e]
                                if cached_sub is not None
                                else sub_packed[orig_e]
                            )
                            compressed = _native_render_codec._quantize_2d(
                                expert_2d, fmt,
                                nvfp4_global_real_override=(
                                    per_expert_joint[orig_e]),
                                input_global_scale_override=(
                                    expert_input_scale),
                                levers=packed_render_options,
                            )
                            for suffix, t in compressed.items():
                                out[f"{base}.{suffix}"] = t.cpu()
                covered.add(full)
                hist[(
                    "packed_moe_per_expert",
                    _packed_expert_render_hist_label(
                        fmt,
                        is_bf16=is_bf16,
                        source_label=label if is_bf16 else "",
                        cached_3d=cached_3d,
                    ),
                )] += 1
                del packed_param, packed_param_src, proj_split
                if cached_3d is not None:
                    del cached_3d, cached_split
                # The inner-loop locals are function-scoped and survive this
                # module (and this LAYER) otherwise: `sub_packed` views the
                # 18G fp32 `packed_param` promotion, `cached_sub` views the
                # 18G fp32 `cached_3d` — together they pinned 36G through the
                # NEXT layer's render peak on GLM-5.3 (mem-dump attribution,
                # probe runs 1-2; the steady-state OOM of export attempts
                # 4-5 on the 118G unified pool).
                sub_packed = cached_sub = expert_2d = None

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
            concat_split = False
            for target_suffix, source_suffixes, _dim in concat_groups:
                if not (full == target_suffix
                        or full.endswith("." + target_suffix)):
                    continue
                stem = full[: -len(target_suffix)]
                for src_suffix in source_suffixes:
                    live_src = stem + src_suffix
                    ckpt_key = profile.export_tensor_name(
                        profile.live_to_recipe_name(live_src))
                    out[live_src] = _read_source_tensor_verbatim(ckpt_key)
                    hist[("layer_concat_source", "verbatim")] += 1
                concat_split = True
                break
            if concat_split:
                continue
            out[full], label = _passthrough_tensor(
                full, param, source_dtype_by_name)
            hist[("layer_passthrough", label)] += 1
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
                out[full], label = _passthrough_tensor(
                    full, buf, source_dtype_by_name)
                hist[("layer_buffer", label)] += 1

        # 3f. Unload.
        _unload(model, [f"{layers_prefix}{L}."])
        # Free this layer's inline expert renders (the 3-D dequant stacks) so
        # peak stays ~one layer's stack even across the whole sweep.
        del tensors, resolver, joint_globals, _inline_expert_caches
        # The per-Linear walk's loop locals survive the layer iteration:
        # `active_cache` in particular still references the layer's transient
        # expert cache (27G CPU fp32 on GLM-5.3) until the NEXT layer's packed
        # emit reassigns it — i.e. after that layer's full render peak. That
        # exact residue OOM-killed export attempt 3 at layer 4 on the 118G
        # unified pool. Drop it, then collect cycles BEFORE empty_cache so
        # freed CUDA blocks actually return to the pool, every layer.
        active_cache = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize()  # ensure outputs are CPU-resident
            torch.cuda.empty_cache()
        if os.environ.get("PRISMAQUANT_EXPORT_MEM_DUMP", "0") == "1":
            _dump_live_cuda_tensors(f"post-cleanup layer {L:02d}")
        # Every layer, unconditionally: a 306B layer can take minutes of
        # GPU render, and a silent multi-minute phase is an observability
        # defect on first-live-use exports.
        elapsed = time.time() - layer_t0
        done_n = L + 1
        sweep_rate = (time.time() - t_layers) / max(done_n, 1)
        eta_min = sweep_rate * (num_layers - done_n) / 60.0
        # Memory attribution per layer: rss (process anon), cuda_alloc
        # (live tensors), cuda_reserved (allocator pool). A layer-over-layer
        # ramp in alloc = a real reference leak; in reserved-only =
        # fragmentation; in rss-only = CPU-side. On the 118G unified pool a
        # silent ~14G/layer ramp is fatal by layer 6 — attribute, don't guess.
        try:
            rss_gb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                      / 1024 / 1024)
            with open("/proc/self/statm") as f:
                cur_rss_gb = (int(f.read().split()[1])
                              * os.sysconf("SC_PAGE_SIZE") / 1024**3)
        except Exception:
            rss_gb = cur_rss_gb = float("nan")
        if device.type == "cuda":
            mem_note = (f"  rss={cur_rss_gb:.1f}G(max {rss_gb:.1f}) "
                        f"cuda_alloc={torch.cuda.memory_allocated()/1024**3:.1f}G "
                        f"reserved={torch.cuda.memory_reserved()/1024**3:.1f}G")
        else:
            mem_note = f"  rss={cur_rss_gb:.1f}G(max {rss_gb:.1f})"
        print(f"[export-stream] layer {L:02d}  linears={linear_count} "
              f"packed={packed_count}  load={load_s:.2f}s  "
              f"total={elapsed:.2f}s  out_keys={len(out)}  "
              f"({done_n}/{num_layers}, {sweep_rate:.1f}s/layer, "
              f"ETA {eta_min:.1f} min){mem_note}", flush=True)
        # v25: save layer cache BEFORE tensor_sink consumes the dict.
        # Use a tmp + rename to keep the cache file atomic — a kill in
        # the middle of torch.save leaves a .tmp behind which we'll
        # ignore on the next run (skip and recompute the layer).
        cf = _layer_cache_file(L)
        if cf is not None and out:
            tmp = cf.with_suffix(".pt.tmp")
            torch.save(out, str(tmp))
            tmp.rename(cf)
        if tensor_sink is not None:
            tensor_sink(out)
            out = {}
        _stop_after = os.environ.get("PRISMAQUANT_EXPORT_STOP_AFTER_LAYER")
        if _stop_after is not None and L >= int(_stop_after):
            # Diagnostic runs only: exit AFTER the layer cache save + sink
            # flush so the rendered layer is resumable. Artifact INCOMPLETE.
            print(f"[export-stream] PRISMAQUANT_EXPORT_STOP_AFTER_LAYER="
                  f"{_stop_after}: stopping after layer {L:02d} "
                  f"(diagnostic run — artifact INCOMPLETE)", flush=True)
            sys.exit(0)

    print(f"[export-stream] layer sweep: {time.time()-t_layers:.1f}s "
          f"(cache_hits={cache_hits}/{num_layers})",
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
    # M2: under the render's recorded scale rule, consistent with the
    # _pack_production_cached_2d match-render-scale re-derive below.
    nvfp4_joint_global = _compute_nvfp4_joint_global(
        model,
        assignment,
        profile=profile,
        levers=_export_render_options(),
    )

    for qname, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        fmt_key = remap(qname)
        fmt = assignment.get(fmt_key)
        if fmt is not None:
            fmt = _native_render_codec._canonical_export_format(fmt)
        if fmt is None:
            continue
        if fmt == "BF16" or fmt_key in bf16_passthrough:
            out[f"{qname}.weight"], label = _passthrough_tensor(
                f"{qname}.weight", mod.weight)
            if mod.bias is not None:
                out[f"{qname}.bias"], _ = _passthrough_tensor(
                    f"{qname}.bias", mod.bias)
            covered.add(qname)
            hist[("linear", label)] += 1
            continue
        joint = nvfp4_joint_global.get(fmt_key) if fmt == "NVFP4" else None
        compressed = _pack_production_cached_2d(
            fmt_key,
            fmt,
            nvfp4_global_real_override=joint,
            device=mod.weight.device,
        )
        if compressed is None and _PRODUCTION_WEIGHT_CACHE is not None:
            raise RuntimeError(
                f"[export-inmemory] auxiliary Linear {fmt_key} @ {fmt} has "
                "no production-cache render. Non-BF16 MTP/sidecar Linears "
                "must be rendered through ProductionWeightCache before "
                "export; RTN-by-omission would ship bytes that validation "
                "did not measure."
            )
        cache_hit = compressed is not None
        if compressed is None:
            compressed = _native_render_codec._quantize_2d(
                mod.weight.detach().float(), fmt,
                nvfp4_global_real_override=joint,
                linear_name=fmt_key,
             cached_activations=(_CACHED_ACTIVATIONS.get(fmt_key) if _CACHED_ACTIVATIONS is not None else None), input_global_scale_override=_native_render_codec._resolve_nvfp4_input_global_scale(target=fmt_key, calibrated_scales=_INPUT_GLOBAL_SCALES), levers=_export_render_options())
        for suffix, tensor in compressed.items():
            out[f"{qname}.{suffix}"] = tensor.cpu()
        if mod.bias is not None:
            out[f"{qname}.bias"], _ = _passthrough_tensor(
                f"{qname}.bias", mod.bias)
        covered.add(qname)
        hist[("linear", f"{fmt}_PRODUCTION_CACHE" if cache_hit else fmt)] += 1

    # Inline packed-expert renders (PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ):
    # one transient in-memory cache per experts module, built on first use.
    _inline_expert_caches: dict[str, "ProductionWeightCache | None"] = {}
    for qname, mod in model.named_modules():
        if not _is_packed_experts_module(mod, profile):
            continue
        for pn in _packed_experts_param_names(mod, profile):
            full_name = f"{qname}.{pn}" if qname else pn
            recipe_key = remap(full_name)
            fmt = assignment.get(recipe_key)
            if fmt is not None:
                fmt = _native_render_codec._canonical_export_format(fmt)
            if fmt is None:
                continue
            packed_param_src = getattr(mod, pn).detach()
            packed_param = packed_param_src.float()
            E, M, N = packed_param.shape
            proj_split = _split_packed_expert_tensor(
                packed_param,
                pn,
                profile,
            )

            is_bf16 = fmt == "BF16" or full_name in bf16_passthrough
            disk_qname = profile.on_disk_expert_qname(qname)
            should_split = profile.split_packed_experts_for_format(fmt)

            if not should_split:
                out[f"{disk_qname}.{pn}"], label = _passthrough_tensor(
                    full_name, packed_param_src)
                covered.add(full_name)
                hist[("packed_moe", label if is_bf16 else fmt)] += 1
                continue

            # Inline render (no production cache): render this experts module's
            # stack into a transient cache, memoized. Done BEFORE the scale-rule
            # + joint-global pre-pass so the re-derive keys off the inline
            # render's recorded nvfp4_scale_rule.
            if not is_bf16 and qname not in _inline_expert_caches:
                _inline_expert_caches[qname] = (
                    _inline_render_packed_expert_module(
                        model, qname, assignment, profile))
            active_cache = _inline_expert_caches.get(qname)

            # M2: joint globals + re-derive below run under the render's
            # recorded recipe (``_export_render_options`` of the active cache).
            packed_render_options = _export_render_options(active_cache)
            per_expert_joint: list[torch.Tensor | None] = [None] * E
            if fmt == "NVFP4" and len(proj_split) > 1:
                for e in range(E):
                    candidates = [
                        _native_render_codec.compute_nvfp4_global_real(sub_packed[e].float(),
                                                  group_size=16, levers=packed_render_options)
                        for _, sub_packed in proj_split
                    ]
                    per_expert_joint[e] = torch.stack(candidates).max()

            # Read the GPTQ-rendered dequant from the production cache (same
            # contract as the streaming path); hard-fail on RTN-by-omission
            # when a cache is active.
            cached_3d = None
            cached_split = None
            if not is_bf16:
                if not _ALLOW_PACKED_EXPERT_RTN:
                    cached_3d = _read_cached_packed_expert(
                        full_name, fmt, cache=active_cache)
                if cached_3d is None:
                    if _INLINE_EXPERT_GPTQ and not _ALLOW_PACKED_EXPERT_RTN:
                        raise RuntimeError(
                            f"[export-inmemory] packed expert {full_name} @ "
                            f"{fmt} could not be rendered inline "
                            f"(PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ=1) — no "
                            f"experts-module activation snapshot for {qname} in "
                            f"the activation cache. RTN on NVFP4 experts is "
                            f"banned; supply an --activation-cache-dir whose "
                            f"probe covers this module, set the expert to BF16, "
                            f"or set PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN=1 for an "
                            f"explicit research/A-B RTN export.")
                    if (_PRODUCTION_WEIGHT_CACHE is not None
                            and not _ALLOW_PACKED_EXPERT_RTN):
                        raise RuntimeError(
                            f"[export-inmemory] packed expert {full_name} @ "
                            f"{fmt} has no production-cache render. Non-BF16 "
                            f"packed experts MUST be rendered through the "
                            f"deliberate GPTQ path "
                            f"(fill_packed_expert_cache_entries) — RTN on NVFP4 "
                            f"experts is a silent quality regression and is "
                            f"banned. Set PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN=1 "
                            f"for an explicit research/A-B RTN export.")
                    print(
                        f"[export-inmemory] WARNING: RTN-rendering packed "
                        f"expert {full_name} @ {fmt} "
                        f"({'research/A-B flag' if _ALLOW_PACKED_EXPERT_RTN else 'no production cache'}).",
                        flush=True)
                else:
                    cached_split = _split_packed_expert_tensor(
                        cached_3d, pn, profile)

            expert_input_scale = _packed_expert_input_global_scale(
                full_name, cache=active_cache)
            if (cached_3d is not None and expert_input_scale is None
                    and fmt == "NVFP4"):
                raise RuntimeError(
                    f"[export-inmemory] packed expert {full_name} @ {fmt} has "
                    f"a cached render but no calibrated input_global_scale "
                    f"(would ship the 1.0 placeholder). Re-run "
                    f"build_production_cache to recompute the scale, or delete "
                    f"the expert shard to force a full re-render.")
            if expert_input_scale is not None and fmt == "NVFP4":
                # K0.2: a calibrated stage may only ship alongside its
                # attested sibling stage.
                _packed_expert_stage_attestation(
                    full_name, cache=active_cache, profile=profile)

            # M2: re-derive under the render's RECORDED NVFP4 scale rule
            # (same wrap as the streaming packed-expert path).
            # Index-based on purpose — see the streaming emit: enumerate's
            # result-tuple reuse pins tensor-bearing tuples across layers.
            for pi in range(len(proj_split)):
                proj_name = proj_split[pi][0]
                sub_packed = proj_split[pi][1]
                cached_sub = (
                    cached_split[pi][1]
                    if cached_split is not None else None
                )
                E_p, Mp, Np = sub_packed.shape
                for e in range(E_p):
                    base = f"{disk_qname}.{e}.{proj_name}"
                    if is_bf16:
                        out[f"{base}.weight"], label = _passthrough_tensor(
                            full_name, sub_packed[e])
                    else:
                        expert_2d = (
                            cached_sub[e] if cached_sub is not None
                            else sub_packed[e]
                        )
                        compressed = _native_render_codec._quantize_2d(
                            expert_2d, fmt,
                            nvfp4_global_real_override=per_expert_joint[e],
                            input_global_scale_override=expert_input_scale,
                            levers=packed_render_options,
                        )
                        for suffix, tensor in compressed.items():
                            out[f"{base}.{suffix}"] = tensor.cpu()
            covered.add(full_name)
            hist[(
                "packed_moe_per_expert",
                _packed_expert_render_hist_label(
                    fmt,
                    is_bf16=is_bf16,
                    source_label=label if is_bf16 else "",
                    cached_3d=cached_3d,
                ),
            )] += 1
            # Same loop-local pinning class as the streaming emit: drop the
            # views so the fp32 stacks free with their owners.
            sub_packed = cached_sub = None

    for name, p in model.named_parameters():
        if any(name.startswith(c + ".") or name == c for c in covered):
            continue
        if name in out:
            continue
        out[name], label = _passthrough_tensor(name, p)
        hist[("passthrough", label)] += 1

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
            out[full], label = _passthrough_tensor(full, buf)
            hist[("passthrough_buffer", label)] += 1

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
MXFP4_SCHEME = {
    "format": "mxfp4-pack-quantized",
    "weights": {
        "num_bits": 4, "type": "float", "strategy": "group",
        "group_size": 32,
        "symmetric": True, "dynamic": False,
        "scale_dtype": "torch.uint8",
        "zp_dtype": "torch.uint8",
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
FP8_E4M3_SCHEME = {
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


def _pin_regex_to_layer(body: str, layer_idx: str | None) -> str | None:
    if layer_idx is None:
        return None
    return re.sub(
        r"layers\[\.\]\[0-9\]\+",
        f"layers[.]{layer_idx}",
        str(body),
        count=1,
    )


def _constrain_per_expert_projection_regex(
    body: str,
    proj_options: str,
) -> str:
    """Constrain a profile per-expert regex to selected projections.

    Profile specs own projection names.  Older specs spell Qwen-style
    projections as ``(gate|up|down)_proj``; newer/custom specs may provide
    complete alternatives like ``(w1_proj|w3_proj|w2)``.  This helper rewrites
    the final projection segment after ``experts.<id>.`` without hardcoding
    either naming family into the export path.
    """
    replacement = f"({proj_options})"
    for legacy in (
        "(gate|up|down)_proj",
        "(gate_proj|up_proj|down_proj)",
    ):
        if legacy in body:
            return body.replace(legacy, replacement)

    for pattern in (
        r"(?P<prefix>experts\[\.\]\[0-9\]\+\[\.\])(?P<proj>.+?)(?P<suffix>\$)$",
        r"(?P<prefix>experts\\\.\[0-9\]\+\\\.)(?P<proj>.+?)(?P<suffix>\$)$",
        r"(?P<prefix>experts\.\[0-9\]\+\.)(?P<proj>.+?)(?P<suffix>\$)$",
    ):
        constrained, count = re.subn(
            pattern,
            rf"\g<prefix>{replacement}\g<suffix>",
            body,
            count=1,
        )
        if count:
            return constrained
    return body


def _bf16_packed_expert_ignore_regex(
        recipe_key: str,
        profile,
) -> list[str]:
    """If `recipe_key` names a BF16 packed-MoE tensor, return regex
    strings for the corresponding per-expert Linear qnames at
    scheme-dispatch time.

    The packed-parameter to per-projection decomposition comes from the
    active model profile/spec, so export metadata stays aligned with model
    structure config instead of baking Qwen-specific names into this path.
    """
    import re as _re

    # Does this recipe key name a packed-expert tensor?
    if ".experts." not in recipe_key:
        return []
    pn = recipe_key.rsplit(".", 1)[-1]
    if pn not in _packed_expert_param_name_set(profile):
        return []

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
    from .qnames import DOTTED_LAYER_QNAME
    layer_idx = None
    lm = DOTTED_LAYER_QNAME.search(recipe_key)
    if lm:
        layer_idx = lm.group(1)
    # vLLM's should_ignore_layer probes the canonical gate_proj/up_proj/
    # down_proj names; emit the ignore regex with those (not the on-disk
    # w1/w3/w2), else BF16 experts are not recognized as ignored and fall
    # through to a quantized catch-all scheme.
    projections = _vllm_moe_scheme_projection_names(profile, pn)
    proj_options = "|".join(_re.escape(proj) for proj in projections)

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
            pinned = _pin_regex_to_layer(body, layer_idx)
            if pinned is None:
                return []
            pinned = _constrain_per_expert_projection_regex(pinned, proj_options)
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
    pinned = _pin_regex_to_layer(body, layer_idx)
    if pinned is None:
        return []
    pinned = _constrain_per_expert_projection_regex(pinned, proj_options)
    return [f"re:{pinned}"]


FORMAT_SCHEME = {
    "NVFP4": NVFP4_SCHEME,
    "MXFP4": MXFP4_SCHEME,
    "MXFP8": MXFP8_SCHEME,
    "MXFP8_E4M3": MXFP8_SCHEME,
    "MXFP8_E5M2": MXFP8_SCHEME,
    "FP8_E4M3": FP8_E4M3_SCHEME,
    "FP8_SOURCE": FP8_SOURCE_SCHEME,
}

# Formats this container emits *without* a `config_groups` scheme, so they
# cannot appear in `FORMAT_SCHEME` by construction. BF16 is written as a
# plain safetensors bf16 tensor and named on the checkpoint's `ignore`
# list. (FP8_SOURCE is *also* a verbatim-copy passthrough — no
# `_quantize_2d` pass runs for it — but it still describes itself to vLLM
# through `FP8_SOURCE_SCHEME`, so `FORMAT_SCHEME` already covers it.)
CONTAINER_PASSTHROUGH_FORMATS = frozenset({"BF16"})

# The authoritative set of formats THIS exporter can emit: everything it
# can describe in `config_groups` metadata plus the container
# passthroughs. Derived, never hand-listed, so a new scheme becomes
# exportable in the same commit that adds it.
#
# It is deliberately NOT derived from the `_quantize_2d` byte-packer
# branches, which do not agree with what is shippable: `FP8_E5M2` has a
# packer branch but no scheme (it raises "research-only", and bytes with
# no metadata are bytes vLLM cannot dispatch — CLAUDE.md gate #9), while
# `FP8_SOURCE` has a scheme and no packer branch at all.
#
# `serving_profile_specs/vllm_packed_moe.json` reads this constant as its
# export-lane bound, which is what keeps the allocator's menu from ever
# containing a rung export would have to rewrite (issue #22 part 2).
# `_coerce_runtime_legal_assignment` hard-fails on anything outside it.
EXPORTABLE_FORMATS = frozenset(
    {_native_render_codec._canonical_export_format(name) for name in FORMAT_SCHEME}
    | CONTAINER_PASSTHROUGH_FORMATS
)


def derive_executed_activation_formats() -> frozenset[str]:
    """The canonical formats whose scheme the runtime executes with activations.

    Derived from the producer table that owns it (``FORMAT_SCHEME``): a format
    is executed exactly when its scheme carries ``input_activations``, which
    is what vLLM's compressed-tensors dispatcher reads to pick W4A4/W8A8
    over W4A16 at RUNTIME. Canonicalized, so the ``MXFP8`` legacy alias does
    not leak in as a distinct rung. Principle 14: this is the value
    ``lane_specs/compressed_tensors.json``'s
    ``served_activation_quantization.executes`` must equal, and
    ``require_compressed_executes_derived_from_scheme`` refuses any drift --
    a scheme-table edit that adds/drops ``input_activations`` (MXFP4 is one
    field away from flipping) fails there instead of silently keeping the old
    A-side price.
    """
    return frozenset(
        _native_render_codec._canonical_export_format(fmt)
        for fmt, scheme in FORMAT_SCHEME.items()
        if "input_activations" in scheme
    )


def executed_activation_formats_in_quantization_config(qc: Mapping) -> frozenset[str]:
    """Read the executed set off an EMITTED ``quantization_config``.

    The lane spec's own ``if_this_changes`` note asks for exactly this: the
    set of ``config_groups`` carrying ``input_activations``, mapped back to
    format names by matching each group (minus its ``targets``) against the
    producer table. A group no scheme explains is refused rather than
    skipped -- an emitted group the table cannot account for is drift, not a
    clean bill. Groups without ``input_activations`` (today: MXFP4's) are
    correctly absent from the answer.
    """
    groups = qc.get("config_groups") or {}
    if not isinstance(groups, Mapping):
        raise RuntimeError(
            "quantization_config config_groups is not an object; refusing "
            "to read the executed set off it")
    out: set[str] = set()
    for name, group in groups.items():
        if not isinstance(group, Mapping) or "input_activations" not in group:
            continue
        stripped = {k: v for k, v in group.items() if k != "targets"}
        matched = {
            _native_render_codec._canonical_export_format(fmt)
            for fmt, scheme in FORMAT_SCHEME.items()
            if scheme == stripped
        }
        if not matched:
            raise RuntimeError(
                f"PRINCIPLE 14: emitted config group {name!r} carries "
                f"input_activations but matches no scheme in "
                f"export_native_compressed.FORMAT_SCHEME. The emission path "
                f"and the producer table have drifted apart; re-derive, "
                f"never edit the lane spec to silence this.")
        out |= matched
    return frozenset(out)


def require_compressed_executes_derived_from_scheme() -> frozenset[str]:
    """Principle 14: refuse when the lane spec and the scheme table disagree.

    ``lane_specs/compressed_tensors.json``'s
    ``served_activation_quantization.executes`` is a claim about what the
    serving runtime executes, and on this lane the executed contract is a
    function of the artifact we write -- so it is either equal to what the
    per-format scheme table implies, or it is refused. There is no third
    answer and in particular no "the rationale explains the difference": a
    ``rationale`` field explains, it is never the value a gate reads. The
    export preflight runs this before any GPU render.
    """
    from .lane_spec import load_lane_spec

    derived = derive_executed_activation_formats()
    spec = load_lane_spec("compressed_tensors")
    declared = spec.served_activation_quantization
    if declared is None:
        raise RuntimeError(
            "lane_specs/compressed_tensors.json declares no "
            "served_activation_quantization, so the A-side of every NVFP4/FP8 "
            "rung would price to zero; that is a currency error, not a "
            "missing annotation")
    if set(declared.executes) != set(derived):
        missing = sorted(set(derived) - set(declared.executes))
        extra = sorted(set(declared.executes) - set(derived))
        raise RuntimeError(
            "PRINCIPLE 14: lane_specs/compressed_tensors.json declares "
            f"executes={sorted(declared.executes)} but the exporter scheme "
            f"table implies {sorted(derived)} "
            f"(missing={missing or '-'}, extra={extra or '-'}).\n"
            "  The producer's claim about what the serving runtime executes "
            "must be DERIVED from the per-format scheme table's "
            "input_activations. Re-read the table; never edit the list to "
            "silence this.")
    return derived


def _fused_modules_mapping_for_profile(profile) -> dict[str, tuple[str, ...]]:
    """Return fused-module leaf mapping for target emission.

    The returned shape mirrors vLLM's ``packed_modules_mapping``:
    ``{"qkv_proj": ("q_proj", ...)}``.
    """
    if profile is None:
        return {}

    getter = getattr(profile, "fused_sibling_leaf_mapping", None)
    if callable(getter):
        # A failed lookup propagates. Falling through would return a mapping that
        # lacks the profile's own fused groups, and the export would skip the
        # fill-in of fused siblings into `ignore` without an error (#2443).
        mapping = getter()
        if mapping:
            return {
                str(fused): tuple(str(sibling) for sibling in siblings)
                for fused, siblings in mapping.items()
            }

    from .model_profiles.vllm_registry import (
        packed_modules_mapping_from_class,
        vllm_class_for_architecture,
    )
    architecture = getattr(profile, "vllm_architecture_class", None)
    vllm_cls = (vllm_class_for_architecture(architecture() or "")
                if callable(architecture) else None)
    packed_mapping = packed_modules_mapping_from_class(vllm_cls)
    if packed_mapping:
        return {
            str(fused): tuple(str(sibling) for sibling in siblings)
            for fused, siblings in packed_mapping.items()
        }

    spec_getter = getattr(profile, "structure_spec", None)
    spec = spec_getter() if callable(spec_getter) else None
    if spec is None:
        return {}

    mapping: dict[str, tuple[str, ...]] = {}
    for group in getattr(spec, "fused_groups", ()):
        target_parent, target_leaf = _suffix_parent_leaf(group.target_suffix)
        member_leafs: list[str] = []
        valid = True
        for member in group.member_suffixes:
            member_parent, member_leaf = _suffix_parent_leaf(member)
            if target_parent and member_parent and member_parent != target_parent:
                valid = False
                break
            member_leafs.append(member_leaf)
        if valid and len(member_leafs) > 1:
            mapping[target_leaf] = tuple(member_leafs)
    return mapping


def _suffix_parent_leaf(suffix: str) -> tuple[str, str]:
    if "." not in suffix:
        return "", str(suffix)
    parent, leaf = str(suffix).rsplit(".", 1)
    return parent, leaf


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

    This legacy compressed-tensors container intentionally does not publish
    ``execution_contracts.nvfp4_w4a4``.  Its optional/defaulted activation
    scalars preserve existing native artifact bytes but claim no versioned
    fused contract; no live exporter emits that record.
    """
    from .model_profiles import DefaultProfile
    profile = profile or DefaultProfile()

    by_fmt: dict[str, list[str]] = {}
    ignore: list[str] = []
    for n in bf16_passthrough:
        ignore.append(profile.to_vllm_internal_name(n))
    for n in extra_ignore:
        ignore.append(profile.to_vllm_internal_name(n))
    for name, fmt in sorted(assignment.items()):
        fmt = _native_render_codec._canonical_export_format(fmt)
        vllm_name = profile.to_vllm_internal_name(name)
        if fmt == "FP8_SOURCE" and profile.runtime_loads_source_fp8(name):
            # The pinned runtime dequantizes these modules itself at load
            # (source-style keys, module built without a quant config), so
            # they are BF16 in the serving model — not CT-schemed. A
            # config_groups entry here would double-describe them.
            ignore.append(vllm_name)
            continue
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

    # Fill in fused-sibling members that exist in the serving model
    # but weren't in the probe assignment — e.g. Gemma 4's
    # full_attention layers have no v_proj on disk, so the probe
    # never saw it, but vLLM's QKVParallelLinear still instantiates
    # a v_proj sub-module that gets k_proj's weights at load. Scheme
    # dispatch requires all fused siblings to have consistent
    # scheme. We infer missing siblings by walking the assignment for
    # fused groups that landed in `ignore` and filling in every
    # sibling from vLLM's `packed_modules_mapping` or the declarative
    # model-structure spec — including ones we never saw weights for.
    packed_mapping = _fused_modules_mapping_for_profile(profile)
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
            fmt = _native_render_codec._canonical_export_format(fmt)
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

    # Packed-3D MoE target emission. Serving runtimes such as vLLM load
    # packed expert tensors through one FusedMoE module at qname
    # `<block>.experts`. Scheme dispatch
    # (`get_moe_method`) probes targets via THREE synthetic layer
    # names built off the FusedMoE prefix:
    #   `<block>.experts.0.gate_proj`
    #   `<block>.experts.0.up_proj`
    #   `<block>.experts.0.down_proj`
    # — this is the "Linear-before-fusion" naming convention, not the
    # packed-tensor qnames (for example `experts.gate_up_proj`) we emit
    # in the safetensors. Without matching targets on that
    # per-expert form, no scheme binds to FusedMoE, `w2_input_global_scale`
    # etc. are never registered, and load_weights KeyErrors on our
    # per-expert input scale keys.
    #
    # Fix: for each packed recipe entry under `by_fmt` or `ignore`,
    # replace it with a per-expert regex pinned to that layer index so
    # vLLM's scheme dispatch gets a match on expert 0's projection
    # names. One regex per layer covers all (expert, projection)
    # combinations. The profile's packed-expert format groups ensure the
    # projections of a single FusedMoE share a scheme — we crash loud on
    # mismatch.
    packed_fused_states: dict[str, set[str]] = {}
    packed_fused_projections: dict[str, list[str]] = {}

    def _packed_expert_vllm_match(vname: str) -> tuple[str, str] | None:
        if vname.startswith("re:"):
            return None
        if "." not in vname:
            return None
        fused_qname, leaf = vname.rsplit(".", 1)
        if not fused_qname.endswith(".experts"):
            return None
        if leaf not in _packed_expert_param_name_set(profile):
            return None
        return fused_qname, leaf

    def _packed_format_group_members(fused_qname: str, leaf: str) -> tuple[str, ...]:
        group_getter = getattr(profile, "packed_expert_format_group", None)
        if callable(group_getter):
            group_key = group_getter(f"{fused_qname}.{leaf}")
            marker = "::__packed_format__:"
            if group_key and marker in group_key:
                return tuple(
                    member for member in group_key.split(marker, 1)[1].split(",")
                    if member
                )
        return (leaf,)

    def _record_packed_fused_state(fused_qname: str, leaf: str, state: str) -> None:
        packed_fused_states.setdefault(fused_qname, set()).add(state)
        seen = set(packed_fused_projections.setdefault(fused_qname, []))
        for member in _packed_format_group_members(fused_qname, leaf):
            # vLLM scheme dispatch probes canonical gate_proj/up_proj/down_proj
            # names, not the on-disk projection names (w1/w3/w2 on LFM2.5).
            for projection in _vllm_moe_scheme_projection_names(profile, member):
                if projection in seen:
                    continue
                packed_fused_projections[fused_qname].append(projection)
                seen.add(projection)

    for fmt, names in list(by_fmt.items()):
        kept = []
        for vname in names:
            packed = _packed_expert_vllm_match(vname)
            if packed is not None:
                fused_qname, leaf = packed
                _record_packed_fused_state(fused_qname, leaf, fmt)
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
        packed = _packed_expert_vllm_match(vname)
        if packed is not None:
            fused_qname, leaf = packed
            _record_packed_fused_state(fused_qname, leaf, "IGNORE")
        else:
            ignore_kept.append(vname)
    ignore = ignore_kept

    def _per_expert_regex_for(
        fused_qname: str,
        projections: list[str],
    ) -> str:
        """Regex matching any `<fused_qname>.<eid>.<proj>` where
        proj is one of the configured per-expert projections. Uses `[.]`
        for literal-dot escapes, matching the rest of this file's regex
        target style."""
        escaped = fused_qname.replace(".", "[.]")
        if not projections:
            projections = list(_all_packed_expert_projection_names(profile))
        proj_options = "|".join(re.escape(proj) for proj in projections)
        return (
            f"re:^{escaped}[.][0-9]+[.]({proj_options})$"
        )

    def _targets_name_experts(names: list[str]) -> bool:
        """True when any target names a per-expert Linear.

        `by_fmt` holds a mix of plain vLLM module names and pre-formed
        regexes, and by the time this runs every per-expert entry is a
        regex: the loop above moves plain `...experts.<eid>.<proj>` names
        out of `by_fmt` into `packed_fused_states`, then re-adds them via
        `_per_expert_regex_for`, whose literal dots are bracket-escaped
        (`layers[.]0[.]mlp[.]experts[.][0-9]+`). So a bare `".experts."`
        test matches nothing on any packed-MoE artifact. Undo the escape
        before testing, and accept either form.
        """
        for name in names:
            probe = name[len("re:"):] if name.startswith("re:") else name
            probe = probe.replace("[.]", ".")
            if ".experts." in probe or probe.endswith(".experts"):
                return True
        return False

    for fused_qname, states in packed_fused_states.items():
        if len(states) > 1:
            raise RuntimeError(
                f"[export-stream] FusedMoE at {fused_qname!r} has mixed "
                f"states across packed expert projections {states}; vLLM "
                "fuses the experts into one kernel needing a single scheme, so "
                "this is unservable. The allocator's packed-expert format group "
                "should have forced one scheme -- this is the same failure "
                "class as the fused-sibling coherence violation: usually an "
                "allocation produced under the WRONG model profile (probe "
                "lacked meta['model'] -> DefaultProfile never saw the packed-"
                "expert grouping). Re-run the allocator with --model-override "
                "<model> so detect_profile resolves the real profile."
            )
        state = next(iter(states))
        regex = _per_expert_regex_for(
            fused_qname,
            packed_fused_projections.get(fused_qname, []),
        )
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
        # vLLM fuses these siblings into ONE packed Linear with ONE scheme, so
        # any group whose present siblings carry MIXED formats yields a wrong
        # artifact -- either a crash at load (>=2 distinct quantized schemes) or
        # a silent corruption (quantized + BF16). Collect such groups and fail
        # the export rather than silently emit. Each entry is
        # (fused_vllm_name, kind, {sibling_leaf: format}).
        fused_coherence_violations: list[tuple[str, str, dict[str, str]]] = []
        parents = {p for (p, _) in leaf_state}
        for parent in sorted(parents):  # deterministic violation ordering
            for fused_name, sibs in packed_mapping.items():
                # Skip degenerate fused definitions (single-sibling).
                if len(sibs) < 2:
                    continue
                states = [leaf_state.get((parent, s)) for s in sibs]
                present = [s for s in states if s is not None]
                if not present:
                    continue  # none of this group present here
                fused_vllm_name = f"{parent}.{fused_name}"
                if fused_vllm_name in fused_emitted:
                    continue
                absent_leaves = [sibs[i] for i, s in enumerate(states)
                                 if s is None]
                if absent_leaves:
                    # Incomplete fused group: a sibling is absent from the
                    # checkpoint (e.g. Gemma4 ``attention_k_eq_v`` synthesizes
                    # v=k, so v_proj is never materialized). vLLM fuses the
                    # group into one packed Linear and requires a single scheme
                    # across q/k/v. If every PRESENT sibling is BF16-ignored,
                    # the synthesized absent sibling is BF16 too → emit the
                    # absent siblings AND the fused name into ignore so the
                    # fused module loads uniformly unquantized. (Quantized
                    # incomplete groups are pinned to BF16 upstream by the
                    # allocator's incomplete-fused-group rule, so a
                    # mixed-and-incomplete group should not reach here.)
                    #
                    # A single present state can still be a partial recipe, so
                    # leave it alone. Mixed present states are unambiguous:
                    # adding an absent sibling to a quantized+BF16 or
                    # multi-quant group cannot produce one coherent fused
                    # scheme.
                    if len(set(present)) != 1:
                        members = {s: leaf_state.get((parent, s), "ABSENT")
                                   for s in sibs}
                        quant_states = {s for s in present if s != "IGNORE"}
                        kind = ("crash@load" if len(quant_states) > 1
                                else "silent-corruption")
                        fused_coherence_violations.append(
                            (fused_vllm_name, kind, members))
                        continue
                    if set(present) == {"IGNORE"}:
                        fused_emitted.add(fused_vllm_name)
                        for leaf in absent_leaves:
                            ignore.append(f"{parent}.{leaf}")
                        ignore.append(fused_vllm_name)
                    continue
                if len(set(states)) != 1:
                    # Present siblings disagree -> a fused-coherence violation.
                    # vLLM fuses them into ONE packed Linear with ONE scheme,
                    # and BOTH mixed cases produce a wrong artifact:
                    #  - >=2 distinct QUANTIZED schemes (e.g. FP8 + NVFP4):
                    #    hard CRASH at load (merged-column scale-shape assert);
                    #  - quantized + BF16/IGNORE: LOADS but SILENTLY CORRUPTS
                    #    the merged Linear (the BF16 slice is read under the
                    #    quant sibling's scheme). Measured 4.3x worse served KL
                    #    on Qwen3.x DeltaNet in_proj_ba gate projections
                    #    (0.106 -> 0.025 at matched bpp; surgical isolation).
                    # Refuse to emit either -- a coherent allocation (correct
                    # model profile) never reaches this branch.
                    members = {s: leaf_state.get((parent, s), "ABSENT")
                               for s in sibs}
                    quant_states = {s for s in present if s != "IGNORE"}
                    kind = ("crash@load" if len(quant_states) > 1
                            else "silent-corruption")
                    fused_coherence_violations.append(
                        (fused_vllm_name, kind, members))
                    continue  # do not emit a fused target for a mixed group
                state = states[0]
                fused_emitted.add(fused_vllm_name)
                if state == "IGNORE":
                    ignore.append(fused_vllm_name)
                else:
                    by_fmt.setdefault(state, []).append(fused_vllm_name)

        if fused_coherence_violations:
            detail = "; ".join(
                f"{fused} [{kind}] <- " + ", ".join(
                    f"{leaf}={fmt}" for leaf, fmt in members.items())
                for fused, kind, members in fused_coherence_violations)
            raise RuntimeError(
                "fused-sibling coherence violation: "
                f"{len(fused_coherence_violations)} merged-column group(s) "
                "carry MIXED formats among their siblings. vLLM fuses each "
                "group into one packed Linear with one scheme, so a mix either "
                "CRASHES at load (>=2 distinct quantized schemes -> scale-shape "
                "assert) or SILENTLY CORRUPTS the merged Linear (quantized + "
                "BF16 -> the BF16 slice is read under the quant scheme; "
                f"measured 4.3x worse served KL on DeltaNet gates): {detail}. "
                "This is almost always an allocation produced under the WRONG "
                "model profile -- e.g. the probe lacked meta['model'], so the "
                "allocator fell back to DefaultProfile and never saw this "
                "architecture's fused group (in_proj_ba / in_proj_qkvz on "
                "Qwen3.x DeltaNet, etc.). Re-run the allocator with "
                "--model-override <model> (or rebuild the probe with "
                "meta['model'] set) so detect_profile resolves the real "
                "profile and promote_fused coerces each group to one format.")

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
        # is needed to route the explicit per-component MXFP8_E4M3 targets
        # to vLLM's fused parameter (in_proj_qkvz, qkv_proj, etc.).
        # `_build_target_list` collapses per-expert enumerations into
        # compact regexes so a 256-expert / 62-layer MoE emits
        # a few hundred targets instead of ~47k. The profile's
        # per-expert regexes remain as a safety-net for any
        # per-expert Linear not captured by the collapse (e.g.
        # stray experts the recipe didn't enumerate).
        # The profile per-expert regexes name on-disk projections; vLLM's
        # scheme probe uses canonical gate_proj/up_proj/down_proj. Rewrite
        # the projection group ONLY when the on-disk names differ from
        # canonical (LFM2.5's w1/w3/w2) — left verbatim when the profile is
        # already canonical (e.g. Qwen), so shipped configs don't churn.
        ondisk: set[str] = set()
        canon: set[str] = set()
        for pname in sorted(_packed_expert_param_name_set(profile)):
            ondisk.update(_packed_expert_projection_names(profile, pname))
            canon.update(_vllm_moe_scheme_projection_names(profile, pname))
        need_canon = ondisk != canon and bool(canon)
        canon_opts = "|".join(sorted(canon)) or "gate_proj|up_proj|down_proj"
        expert_regexes = []
        # Only attach the safety-net when the catch-all group actually
        # holds packed/per-expert units. When the largest group is a
        # non-expert format (GLM-5.3: FP8_SOURCE dense half vs NVFP4
        # experts), the net would claim every per-expert Linear for the
        # WRONG scheme — vLLM's get_moe_method probes `experts.0.X_proj`
        # and a match here overrides the experts' real group.
        catchall_has_experts = _targets_name_experts(by_fmt[catchall])
        if catchall_has_experts:
            for getter in (profile.per_expert_moe_regex,
                           profile.per_expert_mtp_regex):
                r = getter()
                if r is None:
                    continue
                if need_canon:
                    body = r[len("re:"):] if r.startswith("re:") else r
                    r = f"re:{_constrain_per_expert_projection_regex(body, canon_opts)}"
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


def _preflight_quantization_config(
    assignment: dict[str, str],
    bf16_passthrough: set[str],
    *,
    profile: "ModelProfile | None",
) -> None:
    """Run config-only export gates before GPU render and shard writes."""
    # Principle 14, first: the lane spec's executed-activation list must equal
    # what the scheme table implies, checked before any GPU hour is spent. A
    # scheme-table edit that adds/drops `input_activations` refuses here
    # instead of silently keeping the old A-side price (RobTand/prismaquant#163).
    require_compressed_executes_derived_from_scheme()
    try:
        build_quantization_config(
            assignment,
            bf16_passthrough,
            profile=profile,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "[export-stream] quantization-config preflight failed before "
            f"rendering or shard writes: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Recipe canonicalization + Main
# ---------------------------------------------------------------------------
# Per-expert siblings map to a fused packed parent at recipe level.
# If the parent IS quantized, the per-expert source keys are already
# covered and must NOT be added to `extra_ignore` — otherwise vLLM's
# compressed-tensors loader marks the FusedMoE layer as un-quantized
# and the NVFP4 scale params (w2_input_global_scale, ...) never get
# registered, crashing at weight-load.
_PER_EXPERT_RE = re.compile(
    r"^(?P<prefix>.+\.experts)\.\d+\.(?P<proj>[^.]+)$")


def _per_expert_parent(base: str, profile=None) -> str | None:
    """Map a per-expert source tensor base like
    `model.layers.0.mlp.experts.3.gate_proj` to its packed parent
    (for example `model.layers.0.mlp.experts.gate_up_proj`), or None
    if `base` is not a per-expert tensor."""
    m = _PER_EXPERT_RE.match(base)
    if not m:
        return None
    parent = _packed_expert_parent_for_projection(profile, m.group("proj"))
    if parent is None:
        return None
    return f"{m.group('prefix')}.{parent}"


def compute_extra_ignore(
    source_shape_iter,
    assignment: dict[str, str],
    profile=None,
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
        if profile is not None:
            recipe_name = profile.live_to_recipe_name(base)
        else:
            recipe_name = ("model." + base[len("model.language_model."):]
                           if base.startswith("model.language_model.")
                           else base)
        if recipe_name in seen_recipe:
            continue
        parent = _per_expert_parent(recipe_name, profile)
        if parent is not None and parent in seen_recipe:
            continue
        if shape is None or len(shape) != 2:
            continue
        extra_ignore.append(base)
    return extra_ignore


def _export_resume_fingerprint(
    *,
    assignment: dict,
    model_path: str,
    dtype: "torch.dtype",
    declared_buffer_dtypes: dict,
    extra_shard_paths: object = (),
    digest_cache_path: str | Path | None = None,
) -> dict:
    """Everything a `layer_NNN.pt` payload is only valid under.

    `_render_lever_provenance()` says how the export renders; this adds WHAT
    it renders. Before #340 the manifest carried the levers alone, so a cache
    dir reused against a different checkpoint replayed the first checkpoint's
    quantized bytes -- the levers matched and no field named the source. The
    `assignment_hash` beside them compared nothing: `hashlib` was in scope
    nowhere inside `materialize_tensors_streaming`, so the call raised
    NameError and the `except Exception` stamped a null on every manifest
    ever written (measured on origin/main, PB action 41e6e63eb9dd). The
    import is explicit here and there is no swallow: a recipe hash that
    cannot be computed is a crash, not a null that matches the next null.

    Three bindings, all of which a replayed payload silently bakes in:

    * ``source_identity`` -- the sha256 of every safetensors shard the run
      consumes, and of the non-shard files it reads from the checkpoint root:
      ``config.json`` (the skeleton the payloads were quantized against),
      ``model.safetensors.index.json`` (which shard each tensor comes from)
      and every root ``*.py``, which a ``trust_remote_code`` checkpoint is
      built through (all of them, not only the ones ``auto_map`` names).
      See `build_source_checkpoint_identity`. Content, not path: a relocated
      checkpoint still resumes, a same-size value edit does not.
    * ``requested_dtype`` -- the parameter dtype the source read narrows to.
    * ``declared_buffer_dtypes`` -- the skeleton's persistent-buffer dtype
      map, which is what the reader restores buffers to (#311/PR #325). The
      policy STAMP alone says the reader is correct; the map says it was
      handed the same declarations.

    Nothing here may degrade to ``None``: the manifest comparison is
    equality, and ``None == None`` admits. A source that cannot be identified
    raises instead.
    """

    from prismaquant.streamed_model_identity import build_source_checkpoint_identity

    fp_state = _render_lever_provenance()
    fp_state["assignment_hash"] = DIRECT_ASCII_SPACED_LAX.sha256(assignment)[:16]
    fp_state["source_identity"] = build_source_checkpoint_identity(
        model_path,
        extra_shard_paths=extra_shard_paths,
        digest_cache_path=digest_cache_path,
    )
    fp_state["requested_dtype"] = str(dtype)
    fp_state["declared_buffer_dtypes"] = {
        str(name): str(value)
        for name, value in sorted(dict(declared_buffer_dtypes).items())
    }
    return fp_state


# The manifest keys without which replay cannot be authorized. A pre-#340
# cache carries none of them, and a manifest that lost one is not a weaker
# match -- it is unreadable, and refused on the same branch.
_EXPORT_RESUME_REQUIRED_KEYS = (
    "source_identity",
    "requested_dtype",
    "declared_buffer_dtypes",
    "assignment_hash",
)


def _admit_export_resume_cache(cache_path: Path, fp_state: dict) -> bool:
    """Decide whether `layer_*.pt` replay is admitted, BEFORE any is read.

    Returns True when the cache may be resumed. On any refusal every
    `layer_*.pt` is removed and the manifest is rewritten, so the caller's
    per-layer `cf.exists()` check cannot find a stale payload afterwards.
    """
    manifest_path = cache_path / "manifest.json"

    def _refuse(reason: str) -> bool:
        removed = 0
        for stale in cache_path.glob("layer_*.pt"):
            stale.unlink()
            removed += 1
        with manifest_path.open("w") as handle:
            json.dump(fp_state, handle, indent=2)
        print(f"[export-stream] resume REFUSED ({reason}); "
              f"discarded {removed} cached layers", flush=True)
        return False

    if not manifest_path.exists():
        if next(cache_path.glob("layer_*.pt"), None) is not None:
            return _refuse("cached layers have no manifest binding")
        with manifest_path.open("w") as handle:
            json.dump(fp_state, handle, indent=2)
        print(f"[export-stream] wrote cache fingerprint to {manifest_path}",
              flush=True)
        return True

    try:
        with manifest_path.open() as handle:
            prev = json.load(handle)
        if not isinstance(prev, dict):
            raise ValueError("manifest is not an object")
    except Exception as exc:
        return _refuse(f"cache manifest unreadable: {exc}")

    absent = [k for k in _EXPORT_RESUME_REQUIRED_KEYS if k not in prev]
    if absent:
        return _refuse(
            f"manifest predates the source-identity contract, missing {absent}")

    if prev != fp_state:
        diffs = [
            key for key in sorted(set(prev) | set(fp_state))
            if prev.get(key) != fp_state.get(key)
        ]
        return _refuse(f"fingerprint differs in: {diffs}")

    print(f"[export-stream] cache fingerprint match — resumable from "
          f"{len(list(cache_path.glob('layer_*.pt')))} layers", flush=True)
    return True


def _render_lever_provenance() -> dict:
    """The actual resolved render recipe shared by resume and ship evidence."""
    return {
        "persistent_buffer_read_policy": "model_declared_v1",
        "native_render_levers": _export_render_options().record(),
        "PRISMAQUANT_BLOCK_OUTPUT_MATCH": "archived_2026-07-30",
        "activation_cache_fingerprint": _ACTIVATION_CACHE_FINGERPRINT,
        "production_cache_fingerprint": _PRODUCTION_CACHE_FINGERPRINT,
    }



#: The container this exporter writes.  Named from the lane vocabulary rather
#: than spelled here, so a rename cannot leave a card stamped with a lane no
#: ``lane_specs/*.json`` declares.
def _default_export_lane() -> str:
    from .model_profiles.structure import DEFAULT_EXPORT_LANE

    return DEFAULT_EXPORT_LANE


def _write_shipcard(
    out_dir: Path,
    *,
    source_model: str,
    layer_config_path: str | None,
    assignment: dict,
    config_assignment: dict,
    hist: dict,
) -> None:
    """Open the ship record (R13): build-lane facts + empty serve-lane slots.

    The build lane cannot run a quality gate — `vllm` is not importable in the
    build venv, and embedding a docker serve here would make the exporter own
    the serving stack. What it can do is state, on the artifact, exactly which
    serve-lane verdicts are still missing, so "we never ran the ship gate"
    becomes a refusal (`python -m prismaquant.shipcard_cli verify`) instead of an omission.
    """

    from . import read_traffic as _read_traffic
    from . import shipcard as _shipcard

    def _hash(payload) -> str | None:
        try:
            return DIRECT_ASCII_SPACED_LAX.sha256(payload)[:16]
        except Exception:
            return None

    build = {
        "git": _shipcard.git_provenance(),
        "source_model": source_model,
        "layer_config": layer_config_path,
        "layer_config_sha": (
            _shipcard.file_sha256(layer_config_path)
            if layer_config_path else None),
        "assignment_hash": _hash(assignment),
        "config_assignment_hash": _hash(config_assignment),
        "n_assignment_entries": len(config_assignment),
        "achieved_bpp": _shipcard.allocator_achieved_bpp(layer_config_path),
        # Disk bytes are not what decode throughput is made of: a dense
        # weight is streamed every token while a routed expert stack is
        # streamed topk/E of the time, so bpp and per-token read bytes rank
        # artifacts differently on a sparse MoE. Stamped beside the bpp,
        # measured from the shards this export just wrote.
        "read_gb_per_token": _read_traffic.read_traffic_claim(out_dir),
        "format_histogram": {f"{k[0]}/{k[1]}": v for k, v in hist.items()},
        "render_levers": _render_lever_provenance(),
        "kv_shared_fisher": _shipcard.kv_shared_fisher_echo(),
    }
    # Principle 12: the recipe's route histogram travels beside the bpp. Every
    # allocation writes serving_lane_provenance into its recipe (#1387); a
    # recipe from before that fix carries none, so the histogram is then
    # derived from the assignment being exported, under the recipe's target
    # profile, by the same summariser the allocator uses. A recipe that names
    # no target profile gets no derivation: under profile None every unit
    # would read ``no_declared_lane``, a card that looks measured and is not,
    # so the card carries no histogram and no marker. The same holds for an
    # assignment with scoped (Tessera) units, whose routes need a serving
    # context the assignment does not carry (recompute_serving_lane_provenance).
    recipe_meta: dict = {}
    if layer_config_path:
        from .layer_config import read_layer_config_metadata

        recipe_meta = read_layer_config_metadata(layer_config_path) or {}
    provenance = recipe_meta.get("serving_lane_provenance")
    if not isinstance(provenance, dict) and config_assignment:
        from .allocator_candidates import recompute_serving_lane_provenance

        provenance = recompute_serving_lane_provenance(
            config_assignment, recipe_meta.get("target_profile"))
    route_histogram = _shipcard.route_histogram_claim(provenance)
    if route_histogram is not None:
        build["route_histogram"] = route_histogram
        # A card that carries the histogram is held to it by `verify`;
        # historical cards carry no marker and keep verifying (#1387).
        build["route_histogram_owed"] = True
    # Stamp the lane the card was opened on.  This exporter writes exactly
    # one container, and until #631 it stamped none -- so `lane_gate_slots`
    # answered `()` for every native card and the lane's own declarations
    # opened nothing. Lane obligations now also come from the artifact config
    # and build block, so deleting this stamp cannot remove a declared gate.
    card = _shipcard.open_shipcard(
        out_dir, build=build, lane=_default_export_lane())
    path = out_dir / _shipcard.SHIPCARD_FILENAME
    print(f"[export-stream] shipcard opened: {path}", flush=True)
    print(f"[export-stream]   serve-lane slots still UNFILLED: "
          f"{', '.join(_shipcard.unfilled_slots(card))}", flush=True)
    print(f"[export-stream]   close them, then: python3 -m prismaquant.shipcard_cli "
          f"verify {path}", flush=True)


def _refuse_archived_block_output_match() -> None:
    """Fail loudly if an old script still asks for block-output match.

    Walled 2026-07-30 (re-vet R25, archive/block_output_match_2026-07-30/).
    Silently ignoring `PRISMAQUANT_BLOCK_OUTPUT_MATCH=1` would mean an old
    launcher exports *differently* than it did with no signal at all — the
    band-aid this house forbids. `=0` (the explicit disable) is accepted: it
    already asked for what now always happens.
    """
    raw = os.environ.get("PRISMAQUANT_BLOCK_OUTPUT_MATCH")
    if raw is None:
        return
    if str(raw).strip().lower() in {"0", "false", "no", "off", ""}:
        return
    raise SystemExit(
        "[export] ERROR: PRISMAQUANT_BLOCK_OUTPUT_MATCH="
        f"{raw} — block-output match is archived under "
        "archive/block_output_match_2026-07-30. It was UNREACHABLE on the "
        "shipping recipe (the production-cache pack fires first and "
        "`continue`s, so with PRODUCTION_CACHE=1 no dense NVFP4 Linear ever "
        "reached the branch; zero hits in two real production export logs), "
        "and had it been reached it would have re-derived NVFP4 group scales "
        "outside _export_match_render_scale_rule and discarded the render's "
        "joint_mse scales — the -6.6% KL defect M19 fixed everywhere else. "
        "Its {0.95, 1.0, 1.05} per-tensor gain re-search is subsumed by JSO. "
        "Unset the variable (or set it to 0) to export."
    )


def _replace_cli_option(argv: Sequence[str], option: str, value: str) -> list[str]:
    """Replace every spelling of one required argparse option."""
    rewritten = list(argv)
    found = False
    index = 0
    while index < len(rewritten):
        item = rewritten[index]
        if item == option:
            if index + 1 >= len(rewritten):
                break
            rewritten[index + 1] = value
            found = True
            index += 2
            continue
        if item.startswith(option + "="):
            rewritten[index] = option + "=" + value
            found = True
        index += 1
    if not found:
        raise RuntimeError(f"required CLI option {option!r} was not found")
    return rewritten


def _cleanup_successful_export_cache(
    export_cache_dir: str | None,
    *,
    keep_export_cache: bool,
) -> None:
    """Remove resumable layer state only after final artifact publication."""
    if (
        not export_cache_dir
        or keep_export_cache
        or not Path(export_cache_dir).exists()
    ):
        return
    try:
        shutil.rmtree(export_cache_dir)
        print(
            f"[export-stream] removed export cache {export_cache_dir}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[export-stream] WARN cache cleanup failed: {exc!r}",
            flush=True,
        )


def main(argv: Sequence[str] | None = None):
    """Run one native export inside an artifact-level output transaction."""
    raw_argv = list(argv) if argv is not None else None
    parse_argv = raw_argv if raw_argv is not None else os.sys.argv[1:]
    preflight = argparse.ArgumentParser(add_help=False)
    preflight.add_argument("--model")
    preflight.add_argument("--output")
    preflight.add_argument("--export-cache-dir")
    preflight.add_argument("--keep-export-cache", action="store_true")
    known, _unknown = preflight.parse_known_args(parse_argv)
    if known.model is None or known.output is None:
        # Preserve the complete parser's native --help/missing-argument errors.
        return _main_impl(parse_argv)

    # Refuse a stale/tampered or merely MATERIALIZED PrismaSnap source before
    # opening the export transaction or doing any GPU render work.  Ordinary
    # unsnapped sources remain a no-op in this preflight.
    from .prismasnap_contract import require_verified_prismasnap_if_present

    require_verified_prismasnap_if_present(known.model)

    requested_output = Path(known.output)
    with transactional_export_directory(
        known.model,
        requested_output,
        where="export_native_compressed",
    ) as staged_output:
        staged_argv = _replace_cli_option(
            parse_argv,
            "--output",
            str(staged_output),
        )
        result = _main_impl(staged_argv)

    _cleanup_successful_export_cache(
        known.export_cache_dir,
        keep_export_cache=bool(known.keep_export_cache),
    )
    print(
        "[export-stream] done. Serve with:\n"
        f"  vllm serve {requested_output.resolve()} "
        "--quantization compressed-tensors",
        flush=True,
    )
    return result


def _main_impl(argv: Sequence[str] | None = None):
    global _ACTIVATION_CACHE_FINGERPRINT
    global _RENDER_LEVERS, _CACHED_ACTIVATIONS, _INPUT_GLOBAL_SCALES
    global _PRODUCTION_WEIGHT_CACHE, _PRODUCTION_CACHE_FINGERPRINT
    global _PRODUCTION_CACHE_PREFETCH_WORKERS
    global _PRODUCTION_CACHE_PREFETCH_MODE, _ALLOCATOR_TARGET_PROFILE
    _refuse_archived_block_output_match()
    _INPUT_GLOBAL_SCALES = None
    _CACHED_ACTIVATIONS = None
    _ACTIVATION_CACHE_FINGERPRINT = None
    _PRODUCTION_WEIGHT_CACHE = None
    _PRODUCTION_CACHE_FINGERPRINT = None
    _RENDER_LEVERS = NativeRenderLevers.resolve()
    _native_render_codec._DO_NO_HARM_STATS.clear()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="HF model dir (source safetensors + config.json)")
    ap.add_argument("--layer-config", default=None,
                    help="layer_config.json from allocator.py. Optional when "
                         "--perturbed-x-dir is supplied.")
    ap.add_argument(
        "--allow-research-cost-selection",
        action="store_true",
        help="explicitly acknowledge export of a research-stamped assembled "
             "cost selection; the default ship gate refuses it",
    )
    ap.add_argument("--output", required=True,
                    help="Output directory for the compressed checkpoint")
    ap.add_argument("--shard-bytes", type=int, default=1024**3,
                    help="Approx per-shard size in bytes (default 1 GiB). "
                         "Robert's standing packaging preference for every "
                         "published artifact since 2026-08-20; it was 5 GiB "
                         "before. Smaller shards resume better on a flaky "
                         "download and let a client fetch a subset, at the "
                         "cost of more files in the index. A single tensor "
                         "larger than this still gets its own shard -- the "
                         "writer flushes it whole rather than splitting it.")
    ap.add_argument("--device", default="cuda",
                    help="CUDA device for quantization arithmetic. Layer "
                         "weights are read into this device; "
                         "_quantize_2d / _quantize_3d_packed run here; "
                         "outputs are moved to CPU before storage. CPU is "
                         "rejected for production quantization.")
    ap.add_argument("--offload-folder", default=None,
                    help="Accelerate disk-offload folder (defaults to "
                         "sibling of output).")
    ap.add_argument("--ignore", nargs="*", default=None,
                    help="Module qnames to keep at bf16 even if the "
                         "allocator assigned another format. Default: the "
                         "active model profile's pinned_names (typically "
                         "lm_head/head for current vLLM serving targets), "
                         "except when allocator metadata explicitly stamps "
                         "a fixed or DP-unpinned lm_head assignment. "
                         "Pass --ignore with no values to disable profile "
                         "pinning for a runtime that supports quantized heads.")
    ap.add_argument("--activation-cache-dir", default=None,
                    help="Probe's activation cache directory. When "
                         "supplied, per-Linear input_global_scale is "
                         "computed from cached activations "
                         "(max_abs/6.0) instead of the 1.0 default. "
                         "Typically ~1-3%% PPL improvement on NVFP4.")
    ap.add_argument("--production-weight-cache", default=None,
                    help="Pickled ProductionWeightCache containing "
                         "already-rendered production weights. When "
                         "supplied, export packs those weights directly "
                         "instead of recomputing GPTQ/scale-sweep from "
                         "raw activations. This is the faithful path for "
                         "candidates measured with production_weight_cache.")
    ap.add_argument("--production-cache-dir-override", default=None,
                    help="Override the backing shard directory stored "
                         "inside --production-weight-cache, for caches "
                         "moved between containers or host paths.")
    ap.add_argument("--production-cache-lru-gb", type=float, default=24.0,
                    help="Resident tensor budget for disk-backed production "
                         "cache loads. Layer export always prefetches the "
                         "current layer into this LRU.")
    ap.add_argument("--production-cache-prefetch-workers", type=int, default=4,
                    help="Thread count for production-cache prefetch.")
    ap.add_argument("--production-cache-prefetch",
                    choices=("require", "warn"), default="warn",
                    help="`require` fails the export when the production "
                         "cache cannot supply a layer's assignment instead of "
                         "silently falling back to per-tensor NVMe reads "
                         "(re-vet R24/D8). run-pipeline.sh passes require on "
                         "the native lane; the bare-CLI default stays warn.")
    ap.add_argument("--perturbed-x-dir", default=None,
                    help="Directory containing final_layer_config.json and "
                         "activation cache files from a prior production "
                         "calibration/polish run. When supplied, defaults "
                         "--layer-config and --activation-cache-dir from it.")
    ap.add_argument("--gptq", dest="gptq", default=None,
                    action=argparse.BooleanOptionalAction,
                    help="GPTQ one-shot OBS rounding with block-wise "
                         "error propagation (NVFP4, FP8_E4M3/FP8_E5M2, "
                         "MXFP8_E4M3/MXFP8_E5M2). Auto-on when --activation-cache-dir "
                         "is supplied. Measured -2.7%% PPL on Qwen3.6-35B.")
    ap.add_argument("--gptq-static-act-order", dest="gptq_static_act_order",
                    default=None, action=argparse.BooleanOptionalAction,
                    help="Opt-in Lift/MR-GPTQ static activation ordering. "
                         "Columns are processed by activation importance "
                         "during GPTQ but restored before export, so no "
                         "runtime permutation is introduced.")
    ap.add_argument("--gptq-joint-scale-opt", dest="gptq_joint_scale_opt",
                    default=None, action=argparse.BooleanOptionalAction,
                    help="Opt-in Lift/MR-GPTQ joint NVFP4 scale search inside "
                         "GPTQ. The candidate set includes FourOverSix and "
                         "additional codebook-aligned max-to-level scales.")
    ap.add_argument("--scale-sweep", dest="scale_sweep", default=None,
                    action=argparse.BooleanOptionalAction,
                    help="Per-group 1-D scale sweep with RTN rounding on "
                         "NVFP4 — closed-form analog of AutoRound's SGD. "
                         "Auto-on when --activation-cache-dir is supplied. "
                         "Measured best-in-bake-off when composed after "
                         "GPTQ: geomean out_mse ratio = 0.33 vs GPTQ-only "
                         "0.41 vs RTN 1.0, on Qwen3.6-35B visual+MTP "
                         "Linears.")
    ap.add_argument("--export-cache-dir", default=None,
                    help="Per-layer cache dir for resumable export. When "
                         "set, each layer's emitted tensor dict is "
                         "torch.save'd to <cache_dir>/layer_NNN.pt right "
                         "after quantization. On a restart, layers whose "
                         "cache file exists are SKIPPED — their tensors "
                         "are loaded from cache and replayed into the "
                         "shard writer without redoing the GPTQ + "
                         "scale_sweep work. Recovers full progress on a "
                         "mid-flight kill (which today restarts from "
                         "layer 0 every time). Cache is removed at end of "
                         "successful export. Disk overhead: ~2 GB per "
                         "MoE layer = ~120 GB transient on a 62-layer "
                         "MiniMax-class model, freed on completion.")
    ap.add_argument("--keep-export-cache", action="store_true",
                    default=False,
                    help="Don't remove --export-cache-dir on success. "
                         "Useful for debugging or comparing two exports "
                         "against the same cache.")
    args = ap.parse_args(argv)

    out_dir = Path(args.output)
    # Path safety is the first preflight: an in-place/aliased/stale target must
    # be rejected before parsing model tensors or allocating any exporter state.
    # Directory creation remains delayed until the recipe/config gates pass.
    validate_fresh_export_directory(
        args.model,
        out_dir,
        where="export_native_compressed",
    )

    from .model_profiles import detect_profile, DefaultProfile
    profile = detect_profile(args.model)
    print(f"[export-stream] model profile: {profile.name}", flush=True)
    if isinstance(profile, DefaultProfile):
        # The export's fused-coherence gate keys off this profile's
        # packed_modules_mapping plus fallback fused groups. DefaultProfile
        # checks qkv_proj/gate_up_proj and the known DeltaNet
        # in_proj_ba/in_proj_qkvz groups, but unknown architecture-specific
        # merged columns and packed-MoE expert constraints are still invisible
        # when --model's architecture is not registered. Make that blind spot
        # loud.
        print(
            "[export-stream] WARNING: resolved DefaultProfile for "
            f"{args.model!r} -- architecture not registered. Fused-coherence "
            "checks cover only qkv_proj/gate_up_proj; arch-specific merged "
            "columns / packed-MoE experts are NOT verified and may ship "
            "silently corrupt. Register a ModelProfile for this architecture "
            "or confirm it is a vanilla transformer.", flush=True)

    if args.perturbed_x_dir:
        px_layer_config, px_cache_dir = _resolve_perturbed_x_export_inputs(
            args.perturbed_x_dir
        )
        if args.layer_config is None:
            args.layer_config = str(px_layer_config)
        if args.activation_cache_dir is None:
            args.activation_cache_dir = str(px_cache_dir)
        print("[export-stream] perturbed-X inputs: "
              f"layer_config={args.layer_config} "
              f"activation_cache_dir={args.activation_cache_dir}",
              flush=True)
    if args.layer_config is None:
        ap.error("--layer-config is required unless --perturbed-x-dir is supplied")

    with open(args.layer_config) as _lc_for_cache:
        _layer_config_payload_for_cache = json.load(_lc_for_cache)
    validate_layer_config_payload(_layer_config_payload_for_cache, args.layer_config)
    from .research_cost_acceptance import enforce_research_export_acknowledgement
    enforce_research_export_acknowledgement(
        _layer_config_payload_for_cache,
        acknowledged=args.allow_research_cost_selection,
        where="export_native_compressed",
    )
    _assignment_for_cache = _canonicalize_assignment(_layer_config_payload_for_cache)
    _assignment_for_cache, _ = _coerce_runtime_legal_assignment(
        args.model,
        _assignment_for_cache,
        profile,
    )

    if args.production_weight_cache:
        import pickle

        with open(args.production_weight_cache, "rb") as fh:
            production_cache = pickle.load(fh)
        if args.production_cache_dir_override:
            production_cache.relocate(args.production_cache_dir_override)
        if args.production_cache_lru_gb and args.production_cache_lru_gb > 0:
            production_cache.enable_lru(
                int(float(args.production_cache_lru_gb) * 1024**3)
            )
        _PRODUCTION_WEIGHT_CACHE = production_cache
        _PRODUCTION_CACHE_PREFETCH_WORKERS = max(
            1, int(args.production_cache_prefetch_workers)
        )
        _PRODUCTION_CACHE_PREFETCH_MODE = str(
            args.production_cache_prefetch or "warn").lower()
        expected_keys, missing_keys = _production_cache_expected_keys(
            _assignment_for_cache
        )
        if missing_keys:
            mtp_missing = [k for k in missing_keys
                           if str(k[0]).startswith("mtp.")]
            mtp_hint = ""
            if mtp_missing:
                # Fail at attach time with the producer contract, not hours
                # later in _materialize_tensors_inmemory. The shared cache
                # builder can synthesize profile MTP now, but it needs the
                # probe activation cache and the concrete non-BF16 assignment.
                mtp_hint = (
                    f" {len(mtp_missing)} of these are MTP sidecar entries "
                    f"(e.g. {mtp_missing[0][0]}). Rebuild with the current "
                    "build_production_cache, the same --render-layer-config, "
                    "and --activation-cache-dir from the probe so its "
                    "profile-synthesized MTP producer can render them; or "
                    "set MTP_FORMAT=BF16."
                )
            raise RuntimeError(
                "[export-stream] production-weight-cache missing recipe "
                f"entries: {len(missing_keys)} sample={missing_keys[:8]}."
                + mtp_hint
            )
        files = production_cache.verify_files(expected_keys)
        if files["missing"]:
            raise RuntimeError(
                "[export-stream] production-weight-cache backing files "
                f"missing: {len(files['missing'])} sample={files['missing'][:8]}"
            )
        _PRODUCTION_CACHE_FINGERPRINT = _production_cache_fingerprint(
            production_cache,
            expected_keys,
        )
        _INPUT_GLOBAL_SCALES = _production_cache_scales(
            production_cache,
            profile=profile,
        )
        print(
            "[export-stream] production-weight-cache direct path: "
            f"{len(expected_keys)} entries, "
            f"lru={args.production_cache_lru_gb:.1f} GiB, "
            f"prefetch_workers={_PRODUCTION_CACHE_PREFETCH_WORKERS}",
            flush=True,
        )

    # Resolve flag defaults.
    cache_supplied = bool(args.activation_cache_dir)
    # GPTQ + scale-sweep: ON iff activation cache supplied.
    gptq_enabled = args.gptq if args.gptq is not None else cache_supplied
    # scale_sweep: ON iff activation cache supplied.
    scale_sweep_enabled = (args.scale_sweep if args.scale_sweep is not None
                           else cache_supplied)
    static_act_order_enabled = (
        args.gptq_static_act_order
        if args.gptq_static_act_order is not None
        else os.environ.get(
            "PRISMAQUANT_GPTQ_STATIC_ACT_ORDER",
            "0",
        ).strip().lower() not in {"", "0", "false", "no", "off"}
    )
    joint_scale_opt_enabled = (
        args.gptq_joint_scale_opt
        if args.gptq_joint_scale_opt is not None
        else os.environ.get(
            "PRISMAQUANT_NVFP4_JOINT_SCALE_OPT",
            "0",
        ).strip().lower() not in {"", "0", "false", "no", "off"}
    )
    static_act_order_enabled = bool(gptq_enabled and static_act_order_enabled)
    joint_scale_opt_enabled = bool(gptq_enabled and joint_scale_opt_enabled)
    act_passes_any = gptq_enabled or scale_sweep_enabled
    # The activation-aware passes need the actual activations, not just
    # the scale summary. We only load raw activations when at least one
    # pass is enabled.
    if act_passes_any and not cache_supplied:
        print("[export-stream] WARN activation-aware passes requested "
              "but no --activation-cache-dir; disabling.", flush=True)
        gptq_enabled = False
        scale_sweep_enabled = False
        static_act_order_enabled = False
        joint_scale_opt_enabled = False
        act_passes_any = False
    print(f"[export-stream] act-aware passes: "
          f"gptq={gptq_enabled} "
          f"scale_sweep={scale_sweep_enabled} "
          f"static_act_order={static_act_order_enabled} "
          f"joint_scale_opt={joint_scale_opt_enabled}", flush=True)
    _RENDER_LEVERS = NativeRenderLevers.resolve({
        "gptq": gptq_enabled,
        "scale_sweep": scale_sweep_enabled,
        "static_act_order": static_act_order_enabled,
        "joint_scale_opt": joint_scale_opt_enabled,
    })
    print(f"[export-stream] NVFP4 scale rule: {_RENDER_LEVERS.nvfp4_scale_rule}",
          flush=True)

    # Populate the module-level input-global-scale cache (used by
    # `_quantize_2d` for NVFP4 linears) from cached activations.
    # Same cache is reused to populate _CACHED_ACTIVATIONS when any
    # act-aware pass is enabled.
    if args.activation_cache_dir and _PRODUCTION_WEIGHT_CACHE is not None:
        print("[export-stream] production-weight-cache supplied; using its "
              "activation scales and pre-rendered weights for assigned "
              "Linears. Raw activation cache will not drive body export.",
              flush=True)
    elif args.activation_cache_dir:
        from .measure_quant_cost import ActivationIndex
        cache_dir = Path(args.activation_cache_dir)
        if not cache_dir.exists():
            print(f"[export-stream] WARN activation cache dir {cache_dir} "
                  f"missing; input_global_scale falls back to "
                  f"{_native_render_codec.DEFAULT_INPUT_GLOBAL_SCALE}", flush=True)
            _ACTIVATION_CACHE_FINGERPRINT = {
                "path": str(cache_dir.resolve()),
                "missing": True,
            }
        else:
            # Pull candidate names from the recipe — ActivationIndex
            # only loads for names that actually have a cached file.
            with open(args.layer_config) as _lc:
                _recipe_payload = json.load(_lc)
            validate_layer_config_payload(_recipe_payload, args.layer_config)
            _recipe_names = [n for n in _recipe_payload.keys()
                             if not _is_layer_config_meta_key(n)]
            idx = ActivationIndex(cache_dir, _recipe_names)
            _ACTIVATION_CACHE_FINGERPRINT = _activation_index_fingerprint(
                idx, cache_dir)
            scales: dict[str, float] = {}
            for name in idx.names():
                try:
                    acts = idx.load(name)
                    scales[name] = _native_render_codec.compute_nvfp4_input_global_scale(acts, levers=_export_render_options())
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
            # reciprocal (FP8_MAX·FP4_MAX/max_abs) values can drift by
            # a float-precision tick. Take the MIN over the group —
            # the smallest reciprocal == the largest max_abs == the
            # conservative (loosest-clipping) scale for every sibling.
            scales = _unify_input_global_scales_across_fused_siblings(
                scales,
                profile=profile,
            )
            _INPUT_GLOBAL_SCALES = scales
            if act_passes_any:
                _CACHED_ACTIVATIONS = _LazyActivationCache(idx)
                print(f"[export-stream] raw activations will be loaded "
                      f"lazily for GPTQ/round/scale-sweep passes "
                      f"({len(idx)}/{len(_recipe_names)} Linears indexed)",
                      flush=True)
            print(f"[export-stream] input_global_scale calibrated for "
                  f"{len(scales)}/{len(_recipe_names)} Linears from "
                  f"{cache_dir}", flush=True)

    with open(args.layer_config) as f:
        raw_recipe = json.load(f)
    validate_layer_config_payload(raw_recipe, args.layer_config)
    _allocator_meta = _layer_config_metadata(raw_recipe)
    if _allocator_meta.get("target_profile"):
        _ALLOCATOR_TARGET_PROFILE = str(_allocator_meta["target_profile"])
        print(f"[export] allocator target profile (from layer_config): "
              f"{_ALLOCATOR_TARGET_PROFILE}", flush=True)
    assignment = _canonicalize_assignment(raw_recipe)
    assignment, runtime_coerced = _coerce_runtime_legal_assignment(
        args.model,
        assignment,
        profile,
    )
    if runtime_coerced:
        print(_runtime_coercion_report(runtime_coerced), flush=True)
    whole_artifact_budget_from_assignment_payload(
        raw_recipe,
        where="export_native_compressed preflight",
        assignment=assignment,
    )
    validate_mtp_assignment_coverage(args.model, assignment, profile)
    fmts = Counter(assignment.values())
    print(f"[export-stream] recipe: {len(assignment)} entries  mix={dict(fmts)}",
          flush=True)

    bf16_passthrough = _bf16_passthrough_for_assignment(
        args.ignore,
        profile,
        _allocator_meta,
    )
    config_assignment, config_bf16_passthrough, fp8_source_overrides = (
        _fp8_source_config_overlay(
            args.model,
            assignment,
            bf16_passthrough,
            profile,
        )
    )
    if fp8_source_overrides:
        print(
            "[export-stream] config FP8_SOURCE passthrough overrides: "
            f"{len(fp8_source_overrides)} source-FP8 Linears",
            flush=True,
        )
    _preflight_quantization_config(
        config_assignment,
        config_bf16_passthrough,
        profile=profile,
    )
    print("[export-stream] quantization-config preflight passed", flush=True)
    out_dir = prepare_fresh_export_directory(
        args.model,
        out_dir,
        where="export_native_compressed",
    )

    from prismaquant.gpu_guard import require_cuda_hot_path

    dtype = torch.bfloat16
    device = require_cuda_hot_path(
        "export_native_compressed",
        args.device,
    )
    if args.offload_folder is None:
        args.offload_folder = str(out_dir / "_streaming_offload")

    def _rename_body_batch(
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Body-walk keys are LIVE module qnames. `export_tensor_name`
        # speaks the RECIPE namespace, so normalize first — a no-op for
        # every family whose live skeleton already uses recipe names
        # (`live_to_recipe_name` is idempotent prefix/segment rewriting),
        # and the only correct spelling on multimodal-forced skeletons
        # (glm5_next: `model.language_model.` + `.forget_gate.` segments
        # exist live but not in the recipe or the checkpoint).
        return {
            profile.export_tensor_name(profile.live_to_recipe_name(k)): v
            for k, v in batch.items()
        }

    writer = IncrementalSafetensorsWriter(out_dir, args.shard_bytes)
    sample_recipe_key = "model.layers.0.self_attn.q_proj.weight"
    sample_source_key = profile.export_tensor_name(sample_recipe_key)
    if sample_source_key != sample_recipe_key:
        print(
            "[export-stream] streaming source-name remap via profile: "
            f"{sample_recipe_key} -> {sample_source_key}",
            flush=True,
        )

    tensors, hist = materialize_tensors_streaming(
        args.model, assignment,
        profile=profile, bf16_passthrough=bf16_passthrough,
        dtype=dtype, device=device,
        offload_folder=args.offload_folder,
        tensor_sink=lambda batch: writer.add_tensors(_rename_body_batch(batch)),
        export_cache_dir=args.export_cache_dir,
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
            args.model, assignment, profile=profile,
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
        src_extra = _filter_source_passthrough_against_materialized(
            src_extra,
            mtp_tensors,
            profile=profile,
            seen_keys=writer.seen_keys,
        )

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

    extra_ignore = compute_extra_ignore(
        _source_shape_iter(),
        config_assignment,
        profile,
    )
    print(f"[export-stream] extra ignore (unmapped Linears): "
          f"{len(extra_ignore)}", flush=True)

    write_config_with_quantization(
        args.model, out_dir, config_assignment, config_bf16_passthrough,
        extra_ignore=extra_ignore,
        transform_config=None)
    _copy_tokenizer(args.model, out_dir)

    with open(out_dir / "mixed_native_manifest.json", "w") as f:
        json.dump({
            "source_model": args.model,
            "source_recipe": args.layer_config,
            "format_histogram": {f"{k[0]}/{k[1]}": v for k, v in hist.items()},
            "n_assignment_entries": len(config_assignment),
            "source_assignment_entries": len(assignment),
            "fp8_source_passthrough_overrides": sorted(fp8_source_overrides),
            "runtime_coercions": _runtime_coercion_manifest_rows(runtime_coerced),
            "bf16_audit": _bf16_upgrade_audit(
                args.model,
                config_assignment,
                config_bf16_passthrough,
                runtime_coerced,
                profile,
            ),
            "do_no_harm": dict(_native_render_codec._DO_NO_HARM_STATS),
            "packed_expert_export": _packed_expert_export_provenance(),
            "ignore": sorted(config_bf16_passthrough),
        }, f, indent=2)

    # R13: open the ship record. Build-lane facts are final here; the
    # serve-lane slots stay empty until the validators and the gold lane fill
    # them (docs/ARCHITECTURE.md §7.1).
    try:
        _write_shipcard(
            out_dir,
            source_model=args.model,
            layer_config_path=args.layer_config,
            assignment=assignment,
            config_assignment=config_assignment,
            hist=hist,
        )
    except Exception as e:
        print(f"[export-stream] WARN shipcard not written: {e!r}", flush=True)

    budget_attestation = enforce_whole_artifact_budget(
        out_dir,
        raw_recipe,
        where="export_native_compressed",
        assignment=assignment,
    )
    if budget_attestation is not None:
        print(
            "[export-stream] whole-artifact budget passed: "
            f"{budget_attestation['actual_bytes']}B <= "
            f"{budget_attestation['budget_bytes']}B",
            flush=True,
        )

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
        if os.path.lexists(self.out_dir):
            if self.out_dir.is_symlink() or not self.out_dir.is_dir():
                raise RuntimeError(
                    f"{self.out_dir}: native shard output must be a real "
                    "directory, not a file or symlink"
                )
        else:
            self.out_dir.mkdir(parents=True, exist_ok=False)
        stale_temps = sorted(
            path.name
            for path in self.out_dir.iterdir()
            if re.fullmatch(
                r"[.]model-[0-9]+[.]safetensors[.]tmp",
                path.name,
            ) is not None
        )
        if stale_temps:
            raise RuntimeError(
                f"{self.out_dir}: refusing preexisting native temporary "
                f"shard(s) {stale_temps[:12]}; use a fresh output directory"
            )

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
        if os.path.lexists(tmp_path):
            raise RuntimeError(
                f"{tmp_path}: native temporary shard appeared during "
                "export; refusing to overwrite a concurrent/stale writer"
            )
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

    @staticmethod
    def _is_model_artifact_name(name: str) -> bool:
        """Files this writer owns and may replace across export attempts."""
        return (
            name in {"model.safetensors", "model.safetensors.index.json"}
            or re.fullmatch(
                r"model-[0-9]{5}-of-[0-9]{5}\.safetensors", name
            ) is not None
        )

    def _remove_stale_model_artifacts(self, planned_names: set[str]) -> None:
        """Remove only obsolete model containers from an earlier export.

        Tokenizer/config/processor files are intentionally outside this
        writer's ownership. A directory at a reserved model filename is an
        error rather than a recursive-delete target.
        """
        removed: list[str] = []
        for path in self.out_dir.iterdir():
            if (
                not self._is_model_artifact_name(path.name)
                or path.name in planned_names
            ):
                continue
            if path.is_dir() and not path.is_symlink():
                raise RuntimeError(
                    f"cannot replace stale model artifact {path}: expected a "
                    "file, found a directory"
                )
            path.unlink()
            removed.append(path.name)
        if removed:
            print(
                "[export-stream] removed stale model artifact(s): "
                f"{sorted(removed)}",
                flush=True,
            )

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
            self._remove_stale_model_artifacts({final_name})
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
        self._remove_stale_model_artifacts(
            {"model.safetensors.index.json", *set(self.weight_map.values())}
        )
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
    transform_config: dict | None = None,
) -> None:
    from .model_profiles import detect_profile
    profile = detect_profile(src_model)
    src_cfg_path = Path(src_model) / "config.json"
    cfg = json.load(open(src_cfg_path))
    qc = build_quantization_config(assignment, bf16_passthrough,
                                   extra_ignore, profile=profile)
    if qc:
        if transform_config:
            qc["transform_config"] = transform_config
        cfg["quantization_config"] = qc

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def _materialize_mtp_tensors(src_model: str,
                             assignment: dict[str, str],
                             *,
                             profile,
                             bf16_passthrough: set[str],
                             hist: dict,
                             device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Quantize MTP weights per the allocator recipe.

    Transformers v5 does not instantiate MTP modules when loading
    Qwen3.5/3.6 MoE checkpoints (see `_keys_to_ignore_on_load_unexpected`),
    so the streaming decoder-layer sweep never sees any `mtp.*` entry in
    `assignment`. We ask the model profile to build a standalone MTP
    module, load the source MTP weights into it (the source prefix is
    the profile's `mtp_source_prefix()`), wrap it in a parent module
    named `mtp` (so qualified names come out as `mtp.fc`,
    `mtp.layers.0.self_attn.q_proj`, ... per `build_mtp_module`'s
    naming contract), and run the in-memory materialize helper.

    Output tensor names match the checkpoint convention (`mtp.fc.*`,
    `mtp.layers.0.<rest>`). vLLM's `qwen3_5_mtp.load_weights` remaps
    `mtp.→model.` at load time.
    """
    from prismaquant.model_runtime import initialize_model_runtime
    initialize_model_runtime()
    from transformers import AutoConfig

    # Build an MTP wrapper with source weights.
    cfg = AutoConfig.from_pretrained(src_model, trust_remote_code=True)
    text_config = getattr(cfg, "text_config", cfg)
    inner = profile.build_mtp_module(text_config)
    if inner is None:
        raise RuntimeError(
            f"profile '{profile.name}' declares has_mtp() but "
            f"build_mtp_module() returned None — MTP tensors cannot be "
            f"rendered. Either implement build_mtp_module() or take the "
            f"passthrough route (has_mtp() -> False + "
            f"source_passthrough_prefixes()).")
    wrapper = nn.Module()
    wrapper.add_module("mtp", inner)
    wrapper.to(dtype=torch.bfloat16)
    raw = profile.read_mtp_source_state_dict(src_model)
    profile.load_mtp_state_dict(inner, raw)
    # Move the whole MTP module to the export device so
    # _materialize_tensors_inmemory's per-linear quant runs on GPU when
    # EXPORT_DEVICE=cuda. Previously defaulted to CPU, costing ~10× on
    # MTP quant. The input weights (raw) are CPU, so we move after load.
    wrapper.to(device=device)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    # Filter assignment to just `mtp.*` entries. Recipe-name prefix, not
    # the source prefix: `build_mtp_module`'s contract fixes it at `mtp.`.
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


def _filter_source_passthrough_against_materialized(
    src_extra: dict[str, torch.Tensor],
    materialized: dict[str, torch.Tensor],
    *,
    profile,
    seen_keys: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Drop source passthrough tensors already represented by materialized output.

    MTP is synthesized separately from raw `mtp.*` source tensors. For BF16
    packed MTP experts, the synthesized form is the vLLM-loader aggregate
    tensor (`...experts.gate_up_proj` / `...experts.down_proj`), while the
    source checkpoint stores per-expert children
    (`...experts.0.gate_proj.weight`, etc.). Those children must not be copied
    too: vLLM loads the aggregate and then warns on the duplicate children.
    """
    materialized_bases: set[str] = set()
    for key in materialized:
        base = key
        for suffix in (".weight_packed", ".weight_scale",
                       ".weight_global_scale", ".input_global_scale",
                       ".weight"):
            if key.endswith(suffix):
                base = key[:-len(suffix)] + ".weight"
                break
        materialized_bases.add(base)
        if base.endswith(".weight"):
            parent = _per_expert_parent(base[:-len(".weight")], profile)
            if parent is not None:
                materialized_bases.add(parent)

    seen_keys = seen_keys or set()

    def _covered_by_materialized_source_form(key: str) -> bool:
        if key in materialized or key in materialized_bases or key in seen_keys:
            return True
        if key.endswith(".weight"):
            parent = _per_expert_parent(key[:-len(".weight")], profile)
            if parent is not None and parent in materialized_bases:
                return True
        return False

    return {
        key: value for key, value in src_extra.items()
        if not _covered_by_materialized_source_form(key)
    }


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
    with a uniform format (`BF16` | `NVFP4` | `MXFP8_E4M3`). For BF16 we do
    nothing — the passthrough tensor is already in the right dtype
    (typically bf16 in the source). For NVFP4 / MXFP8_E4M3 we route the
    rank-2 weight through `_quantize_2d` and replace the single
    `<name>.weight` key with the compressed-tensors tensor set
    (`<name>.weight_packed`, `<name>.weight_scale`,
    `<name>.weight_global_scale`, `<name>.input_global_scale` for NVFP4;
    `<name>.weight`, `<name>.weight_scale` for MXFP8_E4M3).

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
            fmt = _native_render_codec._canonical_export_format(fmt)
        if fmt is None or fmt == "BF16":
            out[key] = tensor
            continue
        if tensor.ndim != 2:
            # Non-2D visual weights aren't Linear modules — skip them.
            out[key] = tensor
            continue
        weight = tensor.to(device=device, dtype=torch.float32)
        try:
            compressed = _native_render_codec._quantize_2d(
                weight, fmt,
                nvfp4_global_real_override=None,
                linear_name=base,
             cached_activations=(_CACHED_ACTIVATIONS.get(base) if _CACHED_ACTIVATIONS is not None else None), input_global_scale_override=_native_render_codec._resolve_nvfp4_input_global_scale(target=base, calibrated_scales=_INPUT_GLOBAL_SCALES), levers=_export_render_options())
        except Exception as e:
            raise RuntimeError(
                f"[export-stream] visual quant failed for {base} ({fmt}); "
                "refusing to emit BF16 bytes under a quantized config"
            ) from e
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

    # PrismaSnap is an additive source-preparation pass.  Its compact,
    # self-digested provenance must survive the otherwise unchanged native
    # exporter so the final shipcard's model hash binds the treatment.  The
    # absent-source path performs no write and remains byte-for-byte identical
    # to every pre-PrismaSnap export.
    snap_path = src / "prismasnap_provenance.json"
    if os.path.lexists(snap_path):
        if snap_path.is_symlink() or not snap_path.is_file():
            raise RuntimeError(
                f"PrismaSnap provenance is not a regular file: {snap_path}"
            )
        try:
            snap = json.loads(snap_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"PrismaSnap provenance is unreadable: {snap_path}"
            ) from exc
        if (
            not isinstance(snap, dict)
            or snap.get("schema")
            not in {
                "prismaquant.prismasnap.provenance.v1",
                "prismaquant.prismasnap.provenance.v2",
            }
        ):
            raise RuntimeError(
                f"PrismaSnap provenance has an unsupported contract: {snap_path}"
            )
        from .prismasnap_validation import (
            validate_prismasnap_checkpoint,
            validate_prismasnap_provenance_payload,
        )

        # Export can run for hours.  Replay the index/shard content identity
        # immediately before the provenance is copied into the staged output;
        # a source mutation after preflight must abort the transaction.
        validate_prismasnap_checkpoint(src, require_verified=True)

        validate_prismasnap_provenance_payload(
            snap,
            require_verified=True,
            where=f"native export PrismaSnap provenance {snap_path}",
        )
        # This receipt describes the BF16 *source* tree, not the compressed
        # output shards.  Preserve it under an unambiguous name so the final
        # artifact never claims its own bytes are the materialized checkpoint.
        shutil.copy2(snap_path, out_dir / "source_prismasnap_provenance.json")


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
    src_prefix = profile.mtp_source_prefix()
    if not src_prefix or not _source_has_prefixed_weights(src_model, src_prefix):
        return
    # Recipe names are always `mtp.*` — `build_mtp_module`'s contract
    # wraps the module in a parent named `mtp` regardless of what prefix
    # the source checkpoint used.
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
