"""Exact serialized tensor-data footprint for a quantization assignment.

The fit-the-card bit-rate selector (see ``saturation_select.select_under_byte
_budget``) needs to map an allocation -> the GB the exported compressed-tensors
checkpoint will put in safetensors data spans, *before* paying for an export.
This module is that payload map, and it is exact (not the handover's hand-fit
``fixed_floor + 3.04*bpp`` linear approximation): it reproduces the exported
``model.safetensors.index.json`` ``metadata.total_size``.  That metadata is a
tensor-payload total, not a filesystem-size total: safetensors headers,
container metadata, JSON configs, tokenizer assets, and other non-weight files
are intentionally outside this pre-export budget. The exporters measure the
completed directory against the whole-artifact budget stamp instead
(:func:`enforce_whole_artifact_budget`, at the end of this module).

The accounting is the same identity the streaming exporter ships:

    artifact_bytes = floor_bytes + Σ_reencoded memory_bytes_for_shape(shape, fmt)
    floor_bytes    = source_checkpoint_total_bytes
                     − Σ_reencoded n_params · source_bytes_per_param

i.e. every quantizable Linear in the assignment is *re-encoded* from its source
precision into its chosen format (``Σ memory_bytes_for_shape`` == the allocator's
``bits_total_with_aux / 8``), and everything else — embeddings, lm_head (when
pinned), every norm/bias/rotary buffer, vision/MTP sidecars kept at source
precision — stays verbatim at its source byte size. The floor is therefore the
*residual* (source total minus the source size of the re-encoded tensors), which
is why it needs **no checkpoint-name matching**: the multimodal / vLLM name
remap (``model.language_model.layers...`` on disk vs ``model.layers...`` in the
layer_config) never has to be reconciled. Only two scalars are read from the
checkpoint — the grand total bytes and the source bytes-per-param — both via the
safetensors header (``data_offsets``), with no weight load and no torch.

``memory_bytes_for_shape`` already counts scale/zero-point overhead per format
(NVFP4's fp8 block scale per group-of-16, FP8's per-row fp32 scale, FP8_SOURCE's
128×128 block scale), so the body term is exact per shape rather than via a
nominal scalar bpp. Packed-MoE experts (3D shape) are handled by feeding the
``(num_experts, out, in)`` shape through the same primitive. NVFP4 additionally
ships an fp32 ``weight_global_scale`` and, for calibrated W4A4 targets, an fp32
``input_global_scale`` that ``memory_bytes_for_shape`` does not count.
Visual/audio/MTP targets excluded from text calibration are explicitly marked
weight-only W4A16 and ship only the weight scalar.
:func:`nvfp4_global_sidecar_bytes` adds the exact one- or two-scalar payload
(per expert × on-disk projection for packed 3-D tensors).

Name derivation is NOT done here. Every mapping between checkpoint keys,
live allocator qnames, and packed aggregates routes through
:mod:`prismaquant.name_projection` (the shared R5 layer): the leaf rule is
its ``strip_weight_leaf``, the per-expert→packed alias is its
``packed_expert_alias`` (re-exported below for the historic import path),
and :func:`source_tensor_bytes_manifest` / :func:`floor_bytes_for_model`
accept a prebuilt ``NameProjection`` so a caller holding a profile gets the
layer's fail-closed refusals and declared-drop outcomes instead of this
module re-deriving any of it.

There is exactly ONE way to run that identity: :func:`assignment_artifact_bytes`
(and :func:`floor_bytes_for_model` for the model-path convenience form, which
shares the same resolve/check helpers). Every consumer — the byte-budget ship
selector included — goes through it, so no second copy of the accounting can
drift from the one the tests pin. The ``Σ_reencoded`` term is priced from the
per-tensor :class:`SourceByteManifest`, and both ways of getting it wrong are
hard errors, never warnings: a re-encoded name the manifest cannot resolve
(source bytes left in the floor -> artifact over-count) and two re-encoded names
resolving to the SAME source span (bytes removed twice -> artifact under-count,
so an over-budget artifact "fits"). Both are caught in
:func:`resolve_reencoded_source_bytes` before any number is consumed.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import struct
from pathlib import Path
from typing import Iterable, Mapping

from . import format_registry as fr
from .allocator_solver import _shape_from_stats
from .name_projection import (
    MAPPED,
    NameProjection,
    packed_expert_alias,
    strip_weight_leaf,
)

# safetensors header dtype -> bytes per element (header carries the source
# dtype string; we only need it to derive source-bytes-per-param when the
# checkpoint is not uniformly one dtype).
_ST_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "BOOL": 1,
}

GB = 1_000_000_000.0  # decimal GB, matching index.json total_size reporting


def format_tensor_payload_breakdown(
    format_spec_or_name: fr.FormatSpec | str,
    shape: tuple[int, ...],
    *,
    qname: str,
) -> dict:
    """Return the exact additive payload for one candidate tensor.

    This is the per-unit primitive shared by allocator legality, candidate
    pricing, and whole-assignment footprint accounting. Every registered
    format uses its shape-exact producer formula. ``qname`` names the unit in
    callers' diagnostics and is part of the shared call shape.
    """
    spec = (
        format_spec_or_name
        if isinstance(format_spec_or_name, fr.FormatSpec)
        else fr.get_format(str(format_spec_or_name))
    )
    canonical = fr.canonical_format_name(spec.name)
    dims = tuple(int(dim) for dim in shape)
    payload_bytes = int(spec.memory_bytes_for_shape(dims))
    return {
        "format": canonical,
        "shape": list(dims),
        "params": int(math.prod(dims)) if dims else 1,
        "tensor_payload_bytes": payload_bytes,
        "identity_key": None,
        "sidecar_identity_key": None,
        "sidecar_payload_bytes": 0,
    }


def plain_source_dtype_tensor_payload_breakdown(
    source_dtype: str,
    shape: tuple[int, ...],
) -> dict:
    """Return exact bytes for a source tensor with no scale sidecars.

    ``_scan_source_dtype_manifest`` preserves the safetensors dtype token for
    ordinary FP16/FP32/integer sources.  The byte width comes from this
    module's existing safetensors dtype authority, not from an allocator bpp
    table.  Block-scaled FP8 and MXFP4 never use this path: their census kinds
    resolve to registered source formats that include their scale planes.
    """
    dtype = str(source_dtype).strip().upper()
    fp_alias = re.fullmatch(r"FP([0-9]+)", dtype)
    if fp_alias is not None:
        dtype = f"F{fp_alias.group(1)}"
    try:
        element_bytes = int(_ST_DTYPE_BYTES[dtype])
    except KeyError as exc:
        raise ValueError(
            f"source dtype {source_dtype!r} has no safetensors storage-width "
            "entry"
        ) from exc
    dims = tuple(int(dim) for dim in shape)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(
            f"source dtype {dtype} needs a positive tensor shape, got {dims}"
        )
    n_params = int(math.prod(dims))
    return {
        "source_dtype": dtype,
        "shape": list(dims),
        "params": n_params,
        "tensor_payload_bytes": n_params * element_bytes,
        "element_storage_bytes": element_bytes,
    }


def _read_safetensors_header(path: str) -> dict:
    """Return the JSON header of a .safetensors file (no weight load)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def source_checkpoint_bytes(model_path: str) -> tuple[int, dict[str, int]]:
    """(total_bytes, {dtype: bytes}) over all *.safetensors shards.

    Sums the safetensors ``data_offsets`` spans — the exact on-disk byte size
    of every tensor — so it is precise regardless of dtype, sharding, or
    name remapping. Reads only the headers; no tensor data is materialized.
    """
    total = 0
    by_dtype: dict[str, int] = {}
    shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            f"no *.safetensors shards under {model_path!r}; cannot size the "
            "non-quantizable floor")
    for shard in shards:
        header = _read_safetensors_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            nb = int(b) - int(a)
            total += nb
            by_dtype[meta["dtype"]] = by_dtype.get(meta["dtype"], 0) + nb
    return total, by_dtype


def dominant_source_bytes_per_param(by_dtype: Mapping[str, int]) -> int:
    """Bytes/element of the dtype holding the most bytes (the body precision).

    Production sources are uniform bf16 (-> 2) or native fp8 (-> 1). The body
    Linears dominate the byte mass, so the largest-by-bytes dtype is the source
    precision the re-encoded weights are read from. Unknown dtype -> 2 (bf16).

    NB: prefer ``source_regime`` for the re-encoded-source-bytes accounting --
    on a large-vocab FP8 model the bf16 embed+lm_head can outmass the fp8 body
    and this would mis-pick bf16. ``source_regime`` keys off the *presence* of
    fp8 (which only the body has), so it is robust to that.
    """
    if not by_dtype:
        return 2
    dt = max(by_dtype.items(), key=lambda kv: kv[1])[0]
    return _ST_DTYPE_BYTES.get(dt, 2)


def source_regime(by_dtype: Mapping[str, int]) -> str:
    """Classify the source checkpoint's *body* weight precision: 'bf16' | 'fp8'.

    The non-quantizable floor (embeddings, lm_head, norms) is always bf16/fp16,
    so the *presence* of any fp8 dtype in the checkpoint is an unambiguous tell
    that the re-encoded body weights are native fp8 (DeepSeek-V4 / MiniMax) --
    robust to the large-vocab case where bf16 embed+lm_head outmass the fp8 body
    (which fools ``dominant_source_bytes_per_param``). A re-encoded fp8 Linear
    occupies its full FP8_SOURCE layout on disk (fp8 weight + fp32 128x128
    block ``weight_scale_inv``), so its source bytes are
    ``memory_bytes_for_shape("FP8_SOURCE", shape)`` -- see
    ``reencoded_source_bytes_for_shape``. Returns 'fp8' if any F8_* dtype carries
    bytes, else 'bf16'. (A genuinely mixed bf16+fp8 *body* is not a production
    shape; it is reported via ``source_total``/floor as fp8 and should be
    cross-checked against the export.)
    """
    if any(dt.startswith("F8") and nb > 0 for dt, nb in by_dtype.items()):
        return "fp8"
    return "bf16"


# Quantization sidecar suffixes summed into their base tensor's manifest
# entry: the export removes these together with the weight when it
# re-encodes a Linear (DSv4 ``.scale`` MXFP4/E8M0 group scales, DeepSeek /
# MiniMax fp8 ``.weight_scale_inv`` 128x128 block scales, compressed-tensors
# ``.weight_scale``). A standalone sidecar with no base tensor is never
# re-encoded and stays priced in the floor.
_SIDECAR_SUFFIXES = (".scale", ".weight_scale_inv", ".weight_scale")


class SourceByteManifest(dict):
    """``{live_qname: source_bytes}`` that remembers WHICH spans it summed.

    :func:`source_tensor_bytes_manifest` deliberately stores a per-expert
    Linear's bytes twice — once under its own name, once accumulated into
    the packed-parent aggregate — so that either naming scheme resolves.
    The two entries therefore OVERLAP: charging both
    ``…experts.0.gate_proj`` and ``…experts.gate_up_proj`` subtracts the
    same on-disk bytes from the floor twice, under-counting the artifact by
    the whole expert mass (an over-budget artifact then "fits", and
    :func:`check_floor_non_negative` only notices when the over-subtraction
    exceeds the entire floor).

    ``spans[live_qname]`` is the frozenset of *checkpoint base keys* whose
    byte spans that entry's total was summed from — the underlying source
    identity, not the allocator's naming of it. Two requested names whose
    span sets intersect are double-charging the intersection, which
    :func:`resolve_reencoded_source_bytes` can then detect structurally
    instead of trusting a docstring convention.

    It is a plain ``dict`` subclass so every existing consumer (and every
    hand-built test manifest) keeps working; ``spans`` is simply absent on a
    plain dict, in which case the overlap check is skipped and the resolver
    says so. NB ``dict.copy()`` / ``{**m}`` drop the provenance — pass the
    manifest object through rather than re-wrapping it.
    """

    __slots__ = ("spans",)

    def __init__(self, *args, spans=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.spans: dict[str, frozenset[str]] = dict(spans or {})


def source_span_identity(checkpoint_key: str) -> str:
    """The SOURCE identity of a checkpoint key: the key without ``.weight``.

    The leaf rule is the name-projection layer's (``strip_weight_leaf``,
    imported above); this historic name is kept as a thin alias because
    callers outside this module (``read_traffic``) resolve spans by it.
    Unique per checkpoint tensor and shared by every live name that covers
    it (the per-expert entry and the packed aggregate both resolve to it),
    which is what makes a double charge structurally detectable in
    :class:`SourceByteManifest.spans`.
    """
    return strip_weight_leaf(checkpoint_key)


def source_tensor_span_bytes(model_path: str) -> dict[str, int]:
    """``{checkpoint key: on-disk bytes}`` — the EXACT per-tensor partition.

    Every quantization sidecar span (``.scale`` / ``.weight_scale_inv`` /
    ``.weight_scale``) is summed into its base tensor's entry — exactly the
    bytes the export removes from the checkpoint when it re-encodes that
    Linear — so a sidecar does not get an entry of its own.  A *standalone*
    sidecar (no base tensor in the checkpoint) keeps its own entry, because
    the point of this function is that nothing is lost:

        ``sum(source_tensor_span_bytes(p).values())
          == source_checkpoint_bytes(p)[0]``

    holds by construction and is asserted here.  That partition property is
    what :func:`source_tensor_bytes_manifest` renames into live qnames, and
    what :mod:`prismaquant.read_traffic` itemizes the non-allocated floor
    from — an itemization is only honest if it provably covers the whole
    checkpoint.

    Keys are the raw checkpoint names (``.weight`` intact) so a caller can
    still run them through a profile's ``checkpoint_to_live_name``; use
    :func:`source_span_identity` for the base form.
    """
    raw: dict[str, int] = {}
    shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            f"no *.safetensors shards under {model_path!r}; cannot build the "
            "per-tensor source-byte partition")
    for shard in shards:
        header = _read_safetensors_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            raw[name] = raw.get(name, 0) + (int(b) - int(a))
    out: dict[str, int] = {}
    for name, nb in raw.items():
        suffix = next(
            (s for s in _SIDECAR_SUFFIXES if name.endswith(s)), None)
        if suffix is not None:
            stem = name[: -len(suffix)]
            if stem in raw or (stem + ".weight") in raw:
                continue  # folded into its base tensor's entry below
            out[name] = nb  # standalone sidecar: its own floor tensor
            continue
        base = source_span_identity(name)
        out[name] = nb + sum(raw.get(base + s, 0) for s in _SIDECAR_SUFFIXES)
    total_in, total_out = sum(raw.values()), sum(out.values())
    if total_in != total_out:
        raise AssertionError(
            f"[footprint] source_tensor_span_bytes lost bytes on "
            f"{model_path!r}: partitioned {total_out} of {total_in} checkpoint "
            "bytes. Every span must be counted exactly once, as its own entry "
            "or folded into a base tensor's.")
    return out


def source_tensor_bytes_manifest(
    model_path: str,
    name_map=None,
    expert_parent_for_projection=None,
    *,
    projection: NameProjection | None = None,
) -> SourceByteManifest:
    """Exact on-disk source bytes per weight tensor, keyed by live qname base.

    Walks the safetensors headers and, for every weight tensor, sums its
    byte span with its quantization sidecars (``<base>.scale``,
    ``<base>.weight_scale_inv``, ``<base>.weight_scale``) — exactly the
    bytes the export removes from the checkpoint when it re-encodes that
    Linear. Keys are stored without the ``.weight`` suffix to match
    allocator qnames.

    The name mapping is the shared projection layer's, in one of two
    mutually exclusive forms:

    - **``projection=NameProjection(...)``** (preferred): every key goes
      through :meth:`NameProjection.checkpoint_to_live`. A mapped key uses
      the projected live unit; a DECLARED drop (the profile declines the
      key by contract — MTP sidecars, visual towers, fp8 scale siblings)
      keeps its RAW checkpoint spelling, because a live-graph mapper
      declining a key does NOT mean the tensor has no source bytes. A
      profile accessor that raises or returns garbage propagates as a
      structured :class:`NameProjectionError` — never a silent skip.
      Per-expert spans are aliased into their packed aggregate via
      :meth:`NameProjection.packed_parent_of_expert_param`.
    - **``name_map`` / ``expert_parent_for_projection``** (legacy form):
      the raw ``ModelProfile.checkpoint_to_live_name`` /
      ``packed_expert_parent_for_projection`` accessors; identity when
      None. Same semantics, but accessor failures surface as raw
      exceptions and an empty-string mapper result falls back to the raw
      key instead of refusing.

    Passing both forms is refused: two name authorities for one manifest
    is exactly the second-enumeration drift this module must not host.

    Both packed-MoE on-disk layouts resolve to the packed allocator names
    (``...experts.gate_up_proj`` / ``...experts.down_proj``):

    - **Packed 3-D on disk** (LFM2.5, Qwen3.6-35B): the expert param is a
      checkpoint key with NO ``.weight`` suffix. Suffix-less keys are kept
      (only sidecar keys are folded into their base), so the packed tensor
      lands in the manifest under its own name.
    - **Per-expert 2-D on disk** (``...experts.{i}.{proj}.weight``): each
      per-expert span is ALSO accumulated into the packed parent name via
      the shared alias primitive (gate+up fuse into gate_up), driven by the
      projection or by ``expert_parent_for_projection``
      (``ModelProfile.packed_expert_parent_for_projection``; legacy
      gate/up/down fallback when None). The per-expert entries are kept
      alongside the packed aggregate so per-expert-named allocations
      resolve too — a ``reencoded_names`` list must use ONE naming scheme
      per tensor (any consistent probe does), never both. That is no
      longer a convention: the returned :class:`SourceByteManifest` carries
      the checkpoint keys behind every entry in ``.spans``, and
      :func:`resolve_reencoded_source_bytes` rejects a request whose names
      resolve to overlapping source spans.

    This is the per-tensor replacement for the regime-wide
    ``reencoded_source_bytes_for_shape`` accounting, which charges EVERY
    re-encoded Linear at the FP8_SOURCE layout (1 B/param + fp32 block
    scales) as soon as any F8 dtype is present in the checkpoint. On a
    mixed-precision source that is wrong per tensor class — e.g. the
    MXFP4-packed routed experts of a DSv4-Flash checkpoint (I8 nibble
    weights + E8M0 group scales, ~0.53 B/param) were charged 1 B/param,
    "removing" 279.9 GB from a 166.9 GB checkpoint and driving the
    non-quantizable floor to −113 GB. Summing actual header byte spans can
    never exceed the checkpoint total, so a floor computed from this
    manifest is >= 0 by construction (a negative floor is always an
    accounting bug — rejected at the consumers).
    """
    if projection is not None and (
            name_map is not None or expert_parent_for_projection is not None):
        raise ValueError(
            "[footprint] source_tensor_bytes_manifest: pass EITHER a "
            "name-projection layer object (projection=NameProjection(...)) "
            "OR the raw profile accessors (name_map / "
            "expert_parent_for_projection), never both — one manifest must "
            "not carry two disagreeing name authorities")

    def _live_and_packed(name: str) -> tuple[str, str | None]:
        # THE one name mapping, owned by prismaquant.name_projection:
        # checkpoint key -> live allocator unit qname, with the profile's
        # DECLARED drops kept under their raw checkpoint spelling and each
        # per-expert span aliased into its packed aggregate.
        if projection is not None:
            projected = projection.checkpoint_to_live(name)
            live = (projected.target if projected.outcome == MAPPED
                    else strip_weight_leaf(name))
            return live, projection.packed_parent_of_expert_param(live)
        # Legacy accessor form: `or name` keeps a DECLINED key's bytes in
        # the manifest under the raw checkpoint spelling (the MTP case:
        # transformers v5 dropped the module, so the Qwen profiles decline
        # `mtp.*`, while the exporter still re-encodes them from exactly
        # these bytes and the allocator assigns them under raw names).
        # Inert for tensors nothing re-encodes: the floor is
        # `checkpoint_total - sum(resolved re-encoded spans)`, so a
        # manifest entry no `reencoded_names` member references never
        # moves it.
        live = strip_weight_leaf(
            (name_map(name) if name_map is not None else name) or name)
        return live, packed_expert_alias(live, expert_parent_for_projection)

    out = SourceByteManifest()
    provenance: dict[str, set[str]] = {}

    def _add(live: str, nb: int, span_key: str) -> None:
        out[live] = out.get(live, 0) + nb
        provenance.setdefault(live, set()).add(span_key)

    for name, total in source_tensor_span_bytes(model_path).items():
        if any(name.endswith(s) for s in _SIDECAR_SUFFIXES):
            # A standalone sidecar with no base tensor is never re-encoded and
            # stays priced in the floor, so it needs no manifest entry.
            continue
        # Packed 3-D expert params have no ".weight" suffix; the key IS the
        # base (and its sidecars still hang off `<base>.scale` etc.).
        base = source_span_identity(name)
        # `base` is the SOURCE identity: unique per checkpoint tensor, and
        # shared by the per-expert entry and the packed aggregate that both
        # cover it. That shared key is what makes the double charge
        # structurally detectable.
        live, packed = _live_and_packed(name)
        _add(live, total, base)
        if packed is not None:
            _add(packed, total, base)
    out.spans = {k: frozenset(v) for k, v in provenance.items()}
    return out


def _refuse_excluding_an_allocated_namespace(
    prefixes: tuple[str, ...],
    assigned_names: Iterable[str],
    *,
    context: str,
) -> None:
    """Refuse a partition that reaches into the allocator's own candidates.

    Both name vintages are tested, because an operator writes whichever
    spelling they have in hand -- ``mtp.`` is the checkpoint spelling and a
    recipe would say ``model.mtp.`` -- and a prefix that missed only on
    spelling would leave the double-subtraction in place while looking
    checked.
    """

    collisions: dict[str, list[str]] = {}
    for name in assigned_names or ():
        text = str(name)
        spellings = {text}
        if text.startswith("model."):
            spellings.add(text[len("model."):])
        else:
            spellings.add(f"model.{text}")
        for prefix in prefixes:
            if any(s.startswith(prefix) for s in spellings):
                collisions.setdefault(prefix, []).append(text)
    if not collisions:
        return
    detail = "; ".join(
        f"{prefix!r} matches {len(names)} allocatable unit(s) "
        f"e.g. {sorted(names)[:3]}"
        for prefix, names in sorted(collisions.items())
    )
    raise ValueError(
        f"[footprint] {context}: refusing to exclude a namespace the "
        f"allocator can assign: {detail}. Exclusion removes these bytes from "
        f"the priced source total, and re-encoding subtracts them from the "
        f"floor a second time, so every rung would be priced too cheap and "
        f"the artifact would overshoot its budget. Exclusion is only "
        f"meaningful for FLOOR tensors the allocator never reasons about."
    )


def partitioned_source_total_bytes(
    manifest: Mapping[str, int],
    source_total_bytes: int,
    excluded_prefixes: Iterable[str],
    *,
    context: str,
    assigned_names: Iterable[str] = (),
) -> dict:
    """Source total for an artifact that ships only PART of the checkpoint.

    ``assignment_artifact_bytes`` models an artifact as the source checkpoint
    with the re-encoded Linears swapped out: anything absent from the
    assignment stays counted at source precision, which is exactly right for a
    tensor that ships verbatim. It is exactly WRONG for a tensor that does not
    ship in this artifact at all, and there is no way to say so through the
    assignment — a name simply cannot be "assigned absent".

    DSv4-Flash is the live case. The draft head ships as its own directory at
    NVFP4-CB K12, so the target artifact holds no ``mtp.*`` tensor, yet the
    probe carries no MTP Linear (33,325 stats rows, zero MTP), so the recipe
    cannot mention them and ``--mtp-format`` is inert: the 10.863 GB of source
    MTP spans stay in the floor and every rung is priced ~10.9 GB heavier than
    it ships. Under a hard byte budget that does not overshoot, it UNDERSHOOTS
    — the DP buys ~10.9 GB less body than the card can hold, and nothing
    fail-closed catches it because the artifact comes in under budget.

    So the partition is declared here instead, against the same per-tensor
    manifest the floor is built from. Resolution goes through
    :func:`resolve_reencoded_source_bytes`, which charges each checkpoint span
    at most once and refuses a prefix whose matches overlap (the packed-expert
    aliases are stored twice on purpose) rather than silently subtracting the
    same bytes twice.

    A prefix matching nothing is a HARD ERROR. A typo'd prefix that quietly
    excludes zero bytes is precisely the 10.9 GB undershoot above, and it is
    invisible: every downstream number stays self-consistent and the artifact
    still "fits".

    ``assigned_names`` is the universe of names the allocator can assign. A
    prefix reaching into it is refused, because exclusion and re-encoding
    subtract from the same floor independently: ``assignment_artifact_bytes``
    computes ``source_total - reencoded``, and this function has already
    removed the excluded mass from ``source_total``, so a name in both is
    subtracted TWICE and every rung is priced too cheap. Exclusion is only
    meaningful for the floor — tensors the allocator never reasoned about —
    which is the same invariant the exporter enforces on ``--exclude-namespace``.

    Returns ``{source_total_bytes, excluded_source_bytes, excluded_prefixes,
    excluded_names, n_excluded}``; ``source_total_bytes`` is what a caller
    passes to :func:`assignment_artifact_bytes` for THIS artifact.
    """
    prefixes = tuple(dict.fromkeys(str(p) for p in excluded_prefixes))
    if not prefixes:
        return {
            "source_total_bytes": int(source_total_bytes),
            "excluded_source_bytes": 0,
            "excluded_prefixes": (),
            "excluded_names": (),
            "n_excluded": 0,
        }
    matched: list[str] = []
    for prefix in prefixes:
        hits = sorted(n for n in manifest if n.startswith(prefix))
        if not hits:
            raise ValueError(
                f"[footprint] {context}: excluded source prefix {prefix!r} "
                "matched no tensor in the source manifest. A prefix that "
                "excludes nothing silently prices this artifact as if it "
                "shipped the whole checkpoint; fix the prefix rather than "
                "dropping it."
            )
        matched.extend(hits)
    matched = sorted(dict.fromkeys(matched))
    _refuse_excluding_an_allocated_namespace(
        prefixes, assigned_names, context=context)
    resolved = resolve_reencoded_source_bytes(
        manifest, matched, context=f"{context} (excluded source spans)")
    excluded = sum(resolved.values())
    remaining = int(source_total_bytes) - int(excluded)
    if remaining <= 0:
        raise ValueError(
            f"[footprint] {context}: excluding {prefixes} removes "
            f"{excluded}B of a {int(source_total_bytes)}B checkpoint, "
            "leaving nothing to price"
        )
    return {
        "source_total_bytes": remaining,
        "excluded_source_bytes": int(excluded),
        "excluded_prefixes": prefixes,
        "excluded_names": tuple(matched),
        "n_excluded": len(matched),
    }


def resolve_reencoded_source_bytes(
    manifest: Mapping[str, int],
    reencoded_names: Iterable[str],
    *,
    context: str,
    spans: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, int]:
    """Look up each re-encoded Linear's actual source bytes in the manifest.

    A name the manifest cannot resolve is a HARD ERROR, not a warning: an
    unresolved Linear's source bytes stay in the floor while its quantized
    body bytes are still added, silently inflating the artifact estimate —
    on a packed-MoE model by the full expert mass, at which point every
    rung reads "below the floor". Raising here, before any selection
    numbers are computed, puts the offending tensor names in front of the
    operator instead of a fatal below-the-floor exit with a trailing
    warning.

    Two names resolving to the SAME underlying source span is the opposite
    error and is rejected too. The manifest stores a per-expert Linear both
    under its own name and inside its packed-parent aggregate (so either
    naming scheme resolves); summing both subtracts those bytes from the
    floor twice, under-counting the artifact by the whole expert mass so an
    over-budget allocation "fits" — and ``check_floor_non_negative`` only
    catches it once the over-subtraction exceeds the entire floor. The
    per-span provenance (``SourceByteManifest.spans``, or an explicit
    ``spans`` argument for a hand-built manifest) makes it structurally
    detectable: every legitimate request charges each checkpoint span at
    most once. A bare ``dict`` manifest carries no provenance, so the check
    is unavailable and skipped — every production manifest comes from
    :func:`source_tensor_bytes_manifest` and therefore carries it.
    """
    span_map = spans if spans is not None else getattr(manifest, "spans", None)
    out: dict[str, int] = {}
    missing: list[str] = []
    # checkpoint span key -> the first requested name that charged it.
    claimed: dict[str, str] = {}
    overlaps: list[tuple[str, str, str]] = []
    for qname in reencoded_names:
        key = qname
        nb = manifest.get(key)
        if nb is None:
            key = strip_weight_leaf(qname)
            nb = manifest.get(key)
        if nb is None:
            missing.append(qname)
            continue
        out[qname] = int(nb)
        if span_map is None:
            continue
        for span_key in span_map.get(key, ()):
            first = claimed.setdefault(span_key, qname)
            # Same name requested twice is idempotent (``out`` is keyed by
            # name, so it is summed once); only DISTINCT names sharing a
            # span are a double charge.
            if first != qname:
                overlaps.append((first, qname, span_key))
    if missing:
        shown = sorted(missing)[:10]
        raise ValueError(
            f"[footprint] {len(missing)} re-encoded Linear(s) not resolvable "
            f"in the source checkpoint manifest ({context}): "
            + ", ".join(shown)
            + (", …" if len(missing) > len(shown) else "")
            + ". Their source bytes would stay in the floor while their "
            "quantized bytes are still added (artifact over-count; on a "
            "packed-MoE model the entire expert mass is double-counted and "
            "every rung reads 'below the floor'). Fix the profile's name "
            "resolution (checkpoint_to_live_name / "
            "packed_expert_parent_for_projection) so every re-encoded "
            "tensor resolves; do not consume these numbers.")
    if overlaps:
        shown = sorted(overlaps)[:10]
        detail = "; ".join(
            f"{a!r} + {b!r} both cover {span!r}" for a, b, span in shown)
        double_charged = sum(
            int(manifest.get(b, 0)) for b in {b for _a, b, _s in overlaps})
        raise ValueError(
            f"[footprint] {len(overlaps)} re-encoded source span(s) charged "
            f"twice ({context}): {detail}"
            + (", …" if len(overlaps) > len(shown) else "")
            + f". Roughly {double_charged / GB:.3f}GB of source bytes would be "
            "subtracted from the non-quantizable floor twice, under-counting "
            "the artifact by that much — an over-budget artifact then reads as "
            "fitting the byte budget, and the negative-floor guard only fires "
            "if the over-subtraction exceeds the ENTIRE floor. The manifest "
            "stores a per-expert Linear both under its own name and inside its "
            "packed-parent aggregate so that either naming scheme resolves; a "
            "re-encoded-name list must therefore use ONE naming scheme per "
            "tensor (all per-expert, or all packed), never both.")
    return out


def reencoded_source_bytes_for_shape(shape: tuple[int, ...], regime: str) -> int:
    """On-disk source bytes of ONE re-encoded Linear, by source regime.

    bf16 source: ``n_params * 2`` (no scale sibling). fp8 source: the weight is
    native fp8 and ships with an fp32 128x128 ``weight_scale_inv`` block scale --
    exactly the FP8_SOURCE on-disk layout -- so its bytes are
    ``memory_bytes_for_shape("FP8_SOURCE", shape)`` (fp8 weight + fp32 block
    scale). This is what makes the floor exact for fp8-native sources: every
    re-encoded Linear's *full* source footprint (weight + scale_inv) is removed
    from the floor, not just the weight bytes.

    WARNING: regime-wide accounting is only correct when the body is
    uniformly bf16 or uniformly fp8. For mixed-precision sources (e.g.
    MXFP4-packed experts, I8 + E8M0 scales) use
    :func:`source_tensor_bytes_manifest`, which charges each tensor its
    actual header byte span. Consumers must reject a negative floor.
    """
    if regime == "fp8":
        return int(fr.get_format("FP8_SOURCE").memory_bytes_for_shape(shape))
    n = 1
    for d in shape:
        n *= int(d)
    return n * 2  # bf16/fp16 source weight, no scale sibling


def _tensor_class(qname: str) -> str:
    """Coarse tensor-class label for floor-accounting diagnostics."""
    if ".shared_experts." in qname:
        return "shared_experts"
    if ".experts." in qname:
        return "routed_experts"
    if ".self_attn." in qname or ".attn." in qname:
        return "attention"
    if ".mlp." in qname or ".ffn." in qname:
        return "mlp"
    return "other"


def check_floor_non_negative(
    floor_bytes: float,
    source_total_bytes: float,
    reencoded_by_name: Mapping[str, int],
    *,
    context: str,
) -> None:
    """A negative non-quantizable floor is ALWAYS an accounting bug.

    It means the source bytes 'removed' for re-encoding exceed the bytes the
    checkpoint actually holds — e.g. MXFP4-packed experts (~0.53 B/param on
    disk) charged at the FP8_SOURCE 1 B/param layout can drive the floor
    negative and let an artifact more than twice the budget 'fit' it. Raises
    with the per-tensor-class byte breakdown so the offending class is
    named, never rationalized.
    """
    if floor_bytes >= 0:
        return
    by_class: dict[str, int] = {}
    for qname, nb in reencoded_by_name.items():
        cls = _tensor_class(qname)
        by_class[cls] = by_class.get(cls, 0) + int(nb)
    detail = ", ".join(
        f"{cls}={nb / GB:.2f}GB"
        for cls, nb in sorted(by_class.items(), key=lambda kv: -kv[1]))
    raise ValueError(
        f"[footprint] negative non-quantizable floor in {context}: "
        f"floor={floor_bytes / GB:.3f}GB (source_total="
        f"{source_total_bytes / GB:.3f}GB, reencoded_source="
        f"{(source_total_bytes - floor_bytes) / GB:.3f}GB). Removed source "
        f"bytes by tensor class: {detail}. The per-class source-byte rate is "
        "wrong (mixed-precision source charged at a uniform regime?). Use "
        "source_tensor_bytes_manifest() for per-tensor accounting; do not "
        "ship a selection computed from this floor.")


# Stats identity used by allocator/validation accounting when the exporter
# deliberately emits stock NVFP4 as weight-only W4A16. This is an explicit
# producer contract, not a qname heuristic: callers set it from the resolved
# model profile's checkpoint-to-live mapping.
NVFP4_WEIGHT_ONLY_STATS_KEY = "_nvfp4_weight_only"

# Both sidecars are F32 scalars. A regular W4A4 Linear ships one of each;
# weight-only visual/audio/MTP targets ship only weight_global_scale.
_NVFP4_WEIGHT_GLOBAL_SCALE_BYTES_PER_LINEAR = 4
_NVFP4_INPUT_GLOBAL_SCALE_BYTES_PER_LINEAR = 4

# On-disk projection count for packed 3-D expert tensors, keyed by the
# assignment key's leaf name. Mirrors the exporter's
# ``ModelProfile.packed_expert_projection_names`` DefaultProfile fallback
# (a packed ``gate_up_proj`` splits into gate_proj + up_proj per-expert
# Linears on disk; every other packed param emits one Linear per expert).
# footprint deliberately carries no profile/torch dependency, so a
# profile that *declares* a differently-named multi-projection packed
# param would under-count 8·E·(P−1) bytes here — no such profile exists
# in the tree today.
_PACKED_LEAF_PROJECTIONS = {"gate_up_proj": 2}

def nvfp4_global_sidecar_bytes(
    qname: str,
    shape: tuple[int, ...],
    *,
    weight_only: bool = False,
) -> int:
    """Bytes of the fp32 NVFP4 global sidecars the export emits.

    Regular W4A4 targets ship ``weight_global_scale`` +
    ``input_global_scale`` (two fp32 scalars, 8 bytes). Explicit
    ``weight_only=True`` targets ship only ``weight_global_scale`` (4 bytes),
    matching the visual/audio/MTP W4A16 export group. A packed 3-D expert
    tensor ``(E, out, in)`` is split into E × P per-expert 2-D Linears on
    disk (P = on-disk projection count, 2 for ``gate_up_proj``).
    ``memory_bytes_for_shape`` counts weight + group-scale bytes only, so this
    is additive.
    """
    per_linear = _NVFP4_WEIGHT_GLOBAL_SCALE_BYTES_PER_LINEAR
    if not weight_only:
        per_linear += _NVFP4_INPUT_GLOBAL_SCALE_BYTES_PER_LINEAR
    if len(shape) == 3:
        leaf = qname.rsplit(".", 1)[-1]
        n_proj = _PACKED_LEAF_PROJECTIONS.get(leaf, 1)
        return per_linear * int(shape[0]) * n_proj
    return per_linear


def assignment_artifact_bytes(
    assignment: Mapping[str, str],
    stats: Mapping[str, dict],
    *,
    source_total_bytes: int,
    source_manifest: Mapping[str, int] | None,
    regime: str = "bf16",
    canonicalize: bool = True,
    context: str = "assignment_artifact_bytes",
) -> dict:
    """Exact serialized tensor-data bytes for ``assignment``.

    The historical ``artifact_bytes`` result key is retained for API and
    recipe compatibility, but its scope is explicitly tensor data spans.  It
    does *not* include safetensors headers/container metadata or non-weight
    files.

    ``assignment`` maps Linear qname -> format name (the allocator's *expanded*,
    post-promotion per-Linear assignment, so fused-sibling / packed-MoE coupling
    is already reflected). ``stats`` is the probe's per-Linear stats (carries
    ``n_params`` and the ``in/out_features`` / ``num_experts`` the byte formula
    needs). Names absent from ``stats`` are *not* a problem here: they are simply
    not subtracted from the floor, so they remain counted at source precision —
    which is correct for any tensor that ships verbatim (and explains why a
    handful of fused super-names / pins can be missing yet the total stays
    exact). They ARE a problem for a caller pricing an assignment it believes it
    allocated in full, so they are named in ``missing_stats_names`` for such a
    caller to refuse on (the byte-budget selector does).

    ``source_manifest`` (from :func:`source_tensor_bytes_manifest`) is the
    exact source-byte accounting and the one :func:`floor_bytes_for_model`
    and the allocator's byte-budget selector use: each re-encoded Linear is
    charged its ACTUAL header byte span (weight + scale siblings), so the
    two paths agree exactly. A priced Linear the manifest cannot resolve —
    or two priced names resolving to the same source span — is a hard error
    (:func:`resolve_reencoded_source_bytes`).

    It is a REQUIRED keyword with no default. The regime-wide fallback below
    is a legacy approximation that is exact only on a uniform source, and a
    caller cannot be allowed to reach it by *omission* — a forgotten kwarg
    is invisible in review, whereas an explicit ``source_manifest=None`` is
    greppable and states the intent. (The old default was ``None``, i.e.
    every caller that forgot the manifest silently got the approximation
    while the docstring restricted it to uniform sources.)

    With ``source_manifest=None``, ``regime`` ('bf16' | 'fp8', from
    source_regime) sets each re-encoded Linear's *source* byte size removed
    from the floor: bf16 -> 2 bytes/param; fp8 -> the full FP8_SOURCE layout
    (fp8 weight + fp32 128x128 weight_scale_inv), so the source scale
    sibling is removed too (else it is double-counted: left in the floor and
    re-added by the export). This is exact ONLY for a uniformly-bf16 or
    uniformly-fp8 (128x128-block-scaled) body, and is for callers that hold
    ``source_total_bytes`` without the checkpoint on disk; on any other
    source pass a manifest — a mixed source drives the floor negative and is
    rejected (``check_floor_non_negative``), never silently shipped. The
    returned ``source_accounting`` field always says which path ran.

    ``context`` labels this call in the hard-error messages
    (``resolve_reencoded_source_bytes`` / ``check_floor_non_negative``) so a
    sweeping caller can name the rung it was pricing.

    Returns a dict: compatibility alias ``artifact_bytes``, explicit
    ``artifact_payload_bytes`` / ``artifact_byte_scope``, ``floor_bytes``,
    ``body_quant_bytes``, ``reencoded_source_bytes``, ``n_reencoded``,
    ``n_missing_stats``, ``missing_stats_names``, ``regime``,
    ``source_accounting``.
    """
    from prismaquant.allocator_candidates import SOURCE_PASSTHROUGH_FORMATS

    body_quant = 0
    reenc_by_name: dict[str, int] = {}
    priced: list[str] = []
    missing_stats: list[str] = []
    passthrough_names: list[str] = []
    # A source-passthrough unit ships the checkpoint's own bytes, so its
    # contribution must be the SAME number this function subtracts from the
    # floor for it. On the manifest path those are two different computations
    # — ``resolve_reencoded_source_bytes`` reads real header spans while the
    # body loop evaluates a closed form — and they only cancel if the format's
    # arithmetic reproduces the checkpoint exactly. Resolve the spans FIRST so
    # a passthrough is charged the measured span itself, and cross-check the
    # closed form against it below rather than trusting either alone.
    passthrough_spans: dict[str, int] = {}
    if source_manifest is not None:
        passthrough_names = [
            qname for qname, fmt in assignment.items()
            if (fr.canonical_format_name(fmt) if canonicalize else fmt)
            in SOURCE_PASSTHROUGH_FORMATS
        ]
        if passthrough_names:
            passthrough_spans = resolve_reencoded_source_bytes(
                source_manifest, passthrough_names, context=context)
    for qname, fmt in assignment.items():
        entry = stats.get(qname)
        if entry is None:
            entry = stats.get(strip_weight_leaf(qname))
        if not isinstance(entry, dict):
            missing_stats.append(qname)
            continue
        shape = _shape_from_stats(entry)
        name = fr.canonical_format_name(fmt) if canonicalize else fmt
        if name in SOURCE_PASSTHROUGH_FORMATS and qname in passthrough_spans:
            span = int(passthrough_spans[qname])
            closed_form = int(fr.get_format(name).memory_bytes_for_shape(shape))
            if span != closed_form:
                raise ValueError(
                    f"[footprint] {context}: {qname} is assigned the "
                    f"passthrough format {name}, whose exporter copies the "
                    f"source slice VERBATIM, but the checkpoint's own bytes "
                    f"for it ({span}) disagree with the format's accounting "
                    f"({closed_form}). One of the two is wrong about this "
                    "checkpoint, and shipping either number would make the "
                    "artifact budget false — the floor subtracts the span "
                    "while the body would add the closed form."
                )
            body_quant += span
        else:
            body_quant += fr.get_format(name).memory_bytes_for_shape(shape)
        if name == "NVFP4":
            body_quant += nvfp4_global_sidecar_bytes(
                qname,
                shape,
                weight_only=bool(entry.get(NVFP4_WEIGHT_ONLY_STATS_KEY, False)),
            )
        if source_manifest is None:
            reenc_by_name[qname] = reencoded_source_bytes_for_shape(
                shape, regime)
        priced.append(qname)
    if source_manifest is not None:
        reenc_by_name = resolve_reencoded_source_bytes(
            source_manifest, priced, context=context)
    reenc_src = sum(reenc_by_name.values())
    floor = int(source_total_bytes) - reenc_src
    check_floor_non_negative(
        floor, int(source_total_bytes), reenc_by_name, context=context)
    artifact_payload_bytes = floor + body_quant
    return {
        # Compatibility name consumed by the allocator/selection records.
        # Scope is pinned immediately below so it cannot be confused with a
        # post-export stat(2) inventory.
        "artifact_bytes": artifact_payload_bytes,
        "artifact_payload_bytes": artifact_payload_bytes,
        "artifact_byte_scope": "safetensors_tensor_data_spans",
        "export_directory_bytes": None,
        "floor_bytes": floor,
        "body_quant_bytes": body_quant,
        "reencoded_source_bytes": reenc_src,
        "n_reencoded": len(priced),
        "n_missing_stats": len(missing_stats),
        "missing_stats_names": sorted(missing_stats),
        "regime": regime,
        "source_accounting": (
            "per_tensor_manifest" if source_manifest is not None else "regime"),
    }


def assignment_artifact_gb(
    assignment: Mapping[str, str],
    stats: Mapping[str, dict],
    *,
    source_total_bytes: int,
    source_manifest: Mapping[str, int] | None,
    regime: str = "bf16",
) -> float:
    """Convenience: tensor-data payload GB (decimal, matches index.json).

    ``source_manifest`` is required for the same reason as in
    :func:`assignment_artifact_bytes`; pass ``None`` to opt into the
    regime-wide approximation explicitly.
    """
    return assignment_artifact_bytes(
        assignment, stats,
        source_total_bytes=source_total_bytes,
        regime=regime,
        source_manifest=source_manifest,
    )["artifact_bytes"] / GB


def floor_bytes_for_model(
    model_path: str,
    reencoded_names: Iterable[str],
    stats: Mapping[str, dict],
    *,
    regime: str | None = None,
    name_map=None,
    expert_parent_for_projection=None,
    projection: NameProjection | None = None,
) -> dict:
    """Compute the non-quantizable floor (and the scalars to reuse) from a model.

    Convenience wrapper that reads the checkpoint headers once and returns
    ``{source_total_bytes, regime, source_bytes_per_param, floor_bytes,
    reencoded_source_bytes, source_manifest, source_dtype_bytes}``. The floor
    is constant across formats (only the re-encoded *format* varies, not which
    tensors are re-encoded), so callers sweeping many allocations compute this
    once and pass ``source_total_bytes`` + ``source_manifest`` to
    ``assignment_artifact_bytes`` per candidate. Each re-encoded Linear is
    charged its actual header byte span from
    :func:`source_tensor_bytes_manifest` (this function has the model path,
    so it never needs the regime-wide per-param rate); a name the manifest
    cannot resolve is a hard error (:func:`resolve_reencoded_source_bytes`) —
    an unresolved name would silently over-count the artifact.
    The name mapping goes to the shared projection layer: pass
    ``projection=NameProjection(...)`` (fail-closed refusals, declared-drop
    outcomes) OR the legacy ``name_map`` /
    ``expert_parent_for_projection`` accessor pair — never both (see
    :func:`source_tensor_bytes_manifest`). Pass the profile's accessors for
    any packed-MoE architecture; defaults handle identity naming and the
    legacy gate/up/down packing. ``regime`` defaults to
    :func:`source_regime` (robust fp8/bf16 detection) and is returned for
    reporting. ``stats`` is retained for call compatibility (shapes are no
    longer needed to price source bytes).
    """
    total, by_dtype = source_checkpoint_bytes(model_path)
    reg = regime if regime is not None else source_regime(by_dtype)
    manifest = source_tensor_bytes_manifest(
        model_path, name_map=name_map,
        expert_parent_for_projection=expert_parent_for_projection,
        projection=projection)
    reenc_by_name = resolve_reencoded_source_bytes(
        manifest, reencoded_names, context="floor_bytes_for_model")
    reenc_src = sum(reenc_by_name.values())
    check_floor_non_negative(
        int(total) - reenc_src, total, reenc_by_name,
        context="floor_bytes_for_model")
    return {
        "source_total_bytes": total,
        "regime": reg,
        "source_bytes_per_param": dominant_source_bytes_per_param(by_dtype),
        "floor_bytes": int(total) - reenc_src,
        "reencoded_source_bytes": reenc_src,
        "source_manifest": manifest,
        "source_dtype_bytes": by_dtype,
    }

# ---------------------------------------------------------------------------
# Whole-artifact byte budget: the selection-time stamp, its exporter-side
# reader, and the final recursive stat. Lane-independent: the allocator writes
# the stamp and both the compressed-tensors and GGUF exporters enforce it.
# Moved here from ``nvfp4_cb_footprint`` on 2026-09-25 (#1304), unchanged, so
# the retired codebook lane's module can be archived without taking the byte
# budget with it. The schema string is persisted and is not renamed.
# ---------------------------------------------------------------------------

WHOLE_ARTIFACT_BUDGET_SCHEMA = "prismaquant.whole_artifact_budget.v2"
WHOLE_ARTIFACT_BUDGET_FIELD = "whole_artifact_budget"


def assignment_serialization_sha256(
    assignment: Mapping[str, str],
) -> str:
    """Canonical SHA-256 binding a byte budget to one exact assignment."""
    normalized = {
        str(name): fr.canonical_format_name(str(fmt).strip().upper())
        for name, fmt in assignment.items()
    }
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def whole_artifact_budget_stamp(
    *,
    budget_bytes: int,
    selection_tensor_payload_bytes: int,
    selection_non_tensor_reserve_bytes: int,
    selection_assignment: Mapping[str, str],
    excluded_source_prefixes: Iterable[str] = (),
) -> dict:
    """Persist the conservative selection contract consumed by exporters.

    ``excluded_source_prefixes`` records the source namespaces this price was
    computed WITHOUT. It travels because the price and the artifact are two
    halves of one statement: the allocator can only spend the excluded bytes
    on the body if the exporter actually omits them, and nothing else in the
    artifact records that they were meant to be absent. Omitted when empty, so
    a run that excludes nothing writes a byte-identical stamp.
    """
    values = {
        "budget_bytes": budget_bytes,
        "selection_tensor_payload_bytes": selection_tensor_payload_bytes,
        "selection_non_tensor_reserve_bytes": selection_non_tensor_reserve_bytes,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    upper_bound = (
        selection_tensor_payload_bytes + selection_non_tensor_reserve_bytes
    )
    if upper_bound > budget_bytes:
        raise ValueError(
            "selection whole-artifact upper bound exceeds its hard budget: "
            f"{upper_bound}B > {budget_bytes}B"
        )
    excluded = tuple(
        dict.fromkeys(
            str(prefix).strip()
            for prefix in (excluded_source_prefixes or ())
            if str(prefix).strip()
        )
    )
    return {
        "schema": WHOLE_ARTIFACT_BUDGET_SCHEMA,
        "scope": "all_regular_files_recursive",
        "budget_bytes": budget_bytes,
        **({"excluded_source_prefixes": list(excluded)} if excluded else {}),
        "selection_tensor_payload_bytes": selection_tensor_payload_bytes,
        "selection_non_tensor_reserve_bytes": selection_non_tensor_reserve_bytes,
        "selection_whole_artifact_upper_bound_bytes": upper_bound,
        "selection_assignment_sha256": assignment_serialization_sha256(
            selection_assignment
        ),
        "selection_contract": (
            "tensor_payload_plus_operator_supplied_non_tensor_reserve"
        ),
        "final_contract": "stat_all_regular_files_recursive_fail_closed",
    }


def whole_artifact_budget_from_assignment_payload(
    payload: Mapping[str, object],
    *,
    where: str,
    assignment: Mapping[str, str] | None = None,
) -> Mapping[str, object] | None:
    """Read and validate an optional hard export-directory budget stamp."""
    meta = payload.get("__prismaquant__")
    raw = (
        meta.get(WHOLE_ARTIFACT_BUDGET_FIELD)
        if isinstance(meta, Mapping)
        else None
    )
    if raw is None:
        raw = payload.get(WHOLE_ARTIFACT_BUDGET_FIELD)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: whole-artifact budget stamp is not an object")
    if raw.get("schema") != WHOLE_ARTIFACT_BUDGET_SCHEMA:
        raise ValueError(
            f"{where}: unsupported whole-artifact budget schema "
            f"{raw.get('schema')!r}"
        )
    required = (
        "budget_bytes",
        "selection_tensor_payload_bytes",
        "selection_non_tensor_reserve_bytes",
        "selection_whole_artifact_upper_bound_bytes",
    )
    parsed: dict[str, int] = {}
    for name in required:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{where}: whole-artifact budget field {name!r} must be a "
                "nonnegative integer"
            )
        parsed[name] = value
    expected_upper = (
        parsed["selection_tensor_payload_bytes"]
        + parsed["selection_non_tensor_reserve_bytes"]
    )
    if parsed["selection_whole_artifact_upper_bound_bytes"] != expected_upper:
        raise ValueError(
            f"{where}: whole-artifact upper bound does not reconcile: "
            f"stamp={parsed['selection_whole_artifact_upper_bound_bytes']}B, "
            f"payload+reserve={expected_upper}B"
        )
    if expected_upper > parsed["budget_bytes"]:
        raise ValueError(
            f"{where}: selected whole-artifact upper bound {expected_upper}B "
            f"exceeds budget {parsed['budget_bytes']}B"
        )
    assignment_digest = raw.get("selection_assignment_sha256")
    if not isinstance(assignment_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", assignment_digest
    ):
        raise ValueError(
            f"{where}: whole-artifact budget stamp has no valid exact "
            "selection_assignment_sha256"
        )
    if assignment is not None:
        actual_digest = assignment_serialization_sha256(assignment)
        if actual_digest != assignment_digest:
            raise ValueError(
                f"{where}: whole-artifact budget was priced for assignment "
                f"{assignment_digest}, but the assignment being consumed "
                f"hashes to {actual_digest}"
            )
    raw_excluded = raw.get("excluded_source_prefixes")
    if raw_excluded is not None:
        if not isinstance(raw_excluded, (list, tuple)) or not all(
            isinstance(p, str) and p.strip() for p in raw_excluded
        ):
            raise ValueError(
                f"{where}: whole-artifact budget field "
                "'excluded_source_prefixes' must be a list of non-empty "
                "strings"
            )
    return dict(raw)


def budget_stamp_excluded_prefixes(
    stamp: Mapping[str, object] | None,
) -> tuple[str, ...]:
    """The source namespaces a budget stamp was priced WITHOUT.

    Absent means "none excluded", which is both the pre-existing behaviour and
    the correct reading of every stamp written before the field existed: those
    prices charged the whole checkpoint.
    """

    if not stamp:
        return ()
    return tuple(
        dict.fromkeys(
            str(p).strip()
            for p in (stamp.get("excluded_source_prefixes") or ())
            if str(p).strip()
        )
    )


def assert_exclusions_match_budget_stamp(
    stamp: Mapping[str, object] | None,
    excluded_namespaces: Iterable[str],
    *,
    where: str,
) -> None:
    """Refuse an artifact whose contents contradict the price that justified it.

    A namespace exclusion is one statement made in two places: the allocator
    declines to CHARGE for the namespace, which hands those bytes to the body,
    and the exporter declines to WRITE it. Make only the first and the artifact
    overshoots its budget by the excluded mass -- caught, but only by the final
    recursive stat, hours later. Make only the second and the artifact comes in
    UNDER budget by that mass, having bought less quality than it paid for, and
    nothing catches it at all: the selection reconciles, the export passes, the
    shipcard agrees. That asymmetry is why this is an equality check and not a
    bound.

    No stamp means no claim: namespace exclusion is a legitimate operation on
    its own, and an unbudgeted export has nothing to contradict. The check
    binds only once a price has been asserted.
    """

    if not stamp:
        return
    priced = set(budget_stamp_excluded_prefixes(stamp))
    written = {
        str(p).strip() for p in (excluded_namespaces or ()) if str(p).strip()
    }
    if priced == written:
        return
    raise ValueError(
        f"{where}: namespace exclusions disagree with the budget stamp that "
        f"priced this assignment. The price was computed WITHOUT "
        f"{sorted(priced) or '[]'}; this export omits {sorted(written) or '[]'}. "
        f"Priced-but-written ({sorted(priced - written) or '[]'}) overshoots "
        f"the budget by those bytes; written-but-priced "
        f"({sorted(written - priced) or '[]'}) silently ships under budget, "
        f"having bought less quality than the budget paid for. Re-run the "
        f"allocation and the export with the same exclusion set."
    )


def recursive_regular_file_bytes(path: str | Path) -> int:
    """Measure a completed artifact using the budget stamp's final scope."""
    root = Path(path)
    if root.is_file():
        return int(root.stat().st_size)
    if not root.is_dir():
        raise FileNotFoundError(f"export artifact does not exist: {root}")
    return sum(
        int(item.stat().st_size)
        for item in root.rglob("*")
        if item.is_file()
    )


def enforce_whole_artifact_budget(
    artifact_path: str | Path,
    assignment_payload: Mapping[str, object],
    *,
    where: str,
    assignment: Mapping[str, str] | None = None,
) -> dict | None:
    """Hard-fail a completed file/directory against its persisted budget."""
    stamp = whole_artifact_budget_from_assignment_payload(
        assignment_payload,
        where=where,
        assignment=assignment,
    )
    if stamp is None:
        return None
    actual = recursive_regular_file_bytes(artifact_path)
    budget = int(stamp["budget_bytes"])
    attestation = {
        "scope": "all_regular_files_recursive",
        "artifact_path": str(artifact_path),
        "actual_bytes": actual,
        "budget_bytes": budget,
        "headroom_bytes": budget - actual,
        "within_budget": actual <= budget,
    }
    if actual > budget:
        raise RuntimeError(
            f"{where}: exact completed artifact size is {actual}B, exceeding "
            f"the hard whole-artifact budget of {budget}B by {actual - budget}B"
        )
    return attestation
