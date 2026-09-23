"""What the streaming loader reads from a source checkpoint, enumerated once.

The streamed model reads its source checkpoint in two places:

* ``layer_streaming._materialize`` loads the always-resident head (the
  embedding, the final norm, ``lm_head``, the rotary buffers and any
  profile extras) once, when the context is built.
* ``layer_streaming._read_layer_to_device`` loads one decoder layer each
  time a layer is installed or prefetched.

Both select tensors the same way. They walk the live weight map (checkpoint
names mapped through the profile) and keep every live name under one of a
set of prefixes. That selection is :func:`select_source_tensors`, and the
head's prefix list is :func:`resident_head_prefixes`. The loader calls both
functions to decide what to read, so the read manifests that declare the
loader's reads call them too (PQ #1095). Before this module, the Stage B
executable readset listed its source reads from the parent manifest's
per-layer extents. It never listed the head, and nothing compared the
declaration with the reads. The first strictly staged quantum died on its
first head tensor.

A tensor is read as one span ``[start, end)`` of its shard, in absolute file
offsets. :func:`selection_spans` resolves a selection through each shard's
safetensors header. The strict staged reader serves one span from one
staged range that contains it whole (``residency_shard_reader``). A span that
two ranges cover together is not covered, and neither is a span no range
covers. :func:`uncovered_spans` applies that rule.

This module is standard-library only, like ``joint_layer_quanta``, so the
readset builders and the pre-submission coverage check can import it
without torch. The profile-aware composition, which maps checkpoint names
to live names, is ``layer_streaming.streaming_source_plan``.
"""
from __future__ import annotations

import io
import json
import os
import re
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence

#: A safetensors header is a u64 length and that many bytes of JSON. The
#: bound is ``layer_streaming``'s own, so every reader refuses the same files.
SAFETENSORS_HEADER_MAX_BYTES = 100_000_000

#: A span is ``(shard path, start, end)`` in absolute file offsets.
Span = tuple[str, int, int]

#: The shortest ``....layers.`` prefix followed by a layer index.
_LAYER_NAME = re.compile(r"^((?:[^.]+\.)*?layers\.)[0-9]+(?:\.|$)")


def resident_head_prefixes(base_prefix: str,
                           extra: Iterable[str] = ()) -> list[str]:
    """The live-name prefixes of the always-resident head.

    ``base_prefix`` is the base model's dotted name in the live module tree
    (``model``, ``model.language_model``, or ``""`` when the root is the
    base model). ``extra`` is the profile's
    ``head_resident_extra_prefixes``. The order is the loader's, and a
    repeated extra is kept once.
    """
    p = f"{base_prefix}." if base_prefix else ""
    prefixes = [f"{p}embed_tokens.", f"{p}norm.", "lm_head.", f"{p}rotary_emb."]
    for prefix in extra:
        if prefix not in prefixes:
            prefixes.append(prefix)
    return prefixes


def base_prefix_of_layers(layers_prefix: str) -> str:
    """The base model's dotted name, from the decoder layers' prefix.

    ``model.language_model.layers.`` gives ``model.language_model``, and a
    root whose layers sit at ``layers.`` gives ``""``. This is the rule the
    streaming context and its audit apply to ``StreamingContext.layers_prefix``.
    """
    if layers_prefix == "layers.":
        return ""
    if not layers_prefix.endswith(".layers."):
        raise ValueError(f"{layers_prefix!r} is not a decoder layers prefix")
    return layers_prefix.removesuffix(".layers.")


def roster_layers_prefix(qnames: Iterable[str]) -> str:
    """The one live decoder layers prefix a unit roster's names share.

    Unit names are live module names (``model.language_model.layers.3.mlp``).
    Names outside the decoder layers are skipped; the layer names must share
    exactly one prefix, or the roster names no single loader layout and this
    refuses.
    """
    prefixes = set()
    for name in qnames:
        match = _LAYER_NAME.match(name)
        if match is not None:
            prefixes.add(match.group(1))
    if len(prefixes) != 1:
        raise ValueError(f"the unit roster names {len(prefixes)} decoder "
                         f"layers prefixes {sorted(prefixes)!r}, not one: refusing")
    return prefixes.pop()


def live_weight_map(raw_weight_map: Mapping[str, str], model_dir: str,
                    to_live: Callable[[str], str | None],
                    ) -> tuple[dict[str, str], dict[str, str]]:
    """``({live name: shard path}, {live name: checkpoint name})``.

    ``raw_weight_map`` is the checkpoint index's ``weight_map``. ``to_live``
    is the profile's ``checkpoint_to_live_name`` at the loader's
    ``multimodal`` setting; a checkpoint name it maps to None is dropped.
    Shard paths are ``os.path.join(model_dir, shard)``, as the loader opens
    them.
    """
    model_to_shard: dict[str, str] = {}
    model_to_ckpt: dict[str, str] = {}
    for ckpt, shard in raw_weight_map.items():
        live = to_live(ckpt)
        if live is None:
            continue
        model_to_shard[live] = os.path.join(model_dir, shard)
        model_to_ckpt[live] = ckpt
    return model_to_shard, model_to_ckpt


def select_source_tensors(model_to_shard: Mapping[str, str],
                          model_to_ckpt: Mapping[str, str],
                          prefixes: Sequence[str],
                          ) -> dict[str, list[tuple[str, str]]]:
    """``{shard path: [(live name, checkpoint name), ...]}`` under ``prefixes``.

    The loader's selection. A live name is selected when it starts with any
    prefix. Shards and names keep the weight map's order, which is the order
    the loader reads them in.
    """
    prefixes = tuple(prefixes)
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for model_name, shard in model_to_shard.items():
        if model_name.startswith(prefixes):
            by_shard.setdefault(shard, []).append(
                (model_name, model_to_ckpt[model_name]))
    return by_shard


def selection_checkpoint_names(selection: Mapping[str, Sequence[tuple[str, str]]]
                               ) -> list[list[str]]:
    """``[[shard file name, checkpoint name], ...]``, sorted: a selection's identity.

    This is what a readset seals for the head, and what the loader compares
    with its own selection before it reads a byte.
    """
    return sorted([os.path.basename(shard), ckpt]
                  for shard, pairs in selection.items() for _live, ckpt in pairs)


def check_sealed_selection(selection: Mapping[str, Sequence[tuple[str, str]]],
                           sealed: Sequence[Sequence[str]], *, what: str) -> None:
    """Refuse when a loader's selection is not the one a manifest sealed.

    ``sealed`` is :func:`selection_checkpoint_names` as the read manifest's
    builder recorded it. The loader calls this before it reads the first
    tensor of ``selection``, so a readset built from another enumeration
    refuses at once, naming both sides, instead of on the first undeclared
    read (PQ #1095).
    """
    selected = selection_checkpoint_names(selection)
    declared = [list(row) for row in sealed]
    if selected != declared:
        raise RuntimeError(
            f"the {what} this loader selects differs from the one the read "
            f"manifest declares: selected {selected}, declared {declared}; "
            "refusing before the first read (PQ #1095)")


def safetensors_header_length(handle, path: str, size: int) -> int:
    """Read a safetensors file's 8-byte length prefix and check it."""
    raw = handle.read(8)
    if len(raw) != 8:
        raise ValueError(f"{path} is too short to be a safetensors file")
    (length,) = struct.unpack("<Q", raw)
    if not 0 < length <= min(SAFETENSORS_HEADER_MAX_BYTES, size - 8):
        raise ValueError(f"{path} has an invalid safetensors header length")
    return length


def read_safetensors_header(path: str, *, source_reads=None) -> tuple[dict, int, int]:
    """``(header, payload base, file size)`` of one shard.

    ``source_reads`` reads the header from somewhere other than the shard's
    own path: an object with ``prefix(path, nbytes=, where=)``, the Stage B
    preparation's staged reads (``stage_b_prep_io.StagedPreparationReads``,
    PQ #1092). None opens the file.
    """
    size = os.path.getsize(path)
    if source_reads is None:
        with open(path, "rb") as handle:
            length = safetensors_header_length(handle, path, size)
            header = json.loads(handle.read(length))
    else:
        raw = source_reads.prefix(path, nbytes=8, where="safetensors header")
        length = safetensors_header_length(io.BytesIO(raw), path, size)
        if len(raw) < 8 + length:
            raise ValueError(f"{path}: the staged header range holds {len(raw)} "
                             f"bytes, not the {8 + length} its length prefix names")
        header = json.loads(raw[8:8 + length])
    if type(header) is not dict:
        raise ValueError(f"{path}: the safetensors header is not an object")
    return header, 8 + length, size


def tensor_span(header: Mapping, base: int, size: int, path: str,
                name: str) -> Span:
    """One tensor's absolute ``(path, start, end)``, checked against its file."""
    row = header.get(name)
    if not isinstance(row, dict) or "data_offsets" not in row:
        raise ValueError(f"{path} does not hold tensor {name!r}, which the "
                         "checkpoint index places in it")
    begin, end = row["data_offsets"]
    if (type(begin) is not int or type(end) is not int
            or not 0 <= begin <= end <= size - base):
        raise ValueError(f"{path} tensor {name!r} has an invalid span")
    return (path, base + begin, base + end)


def selection_spans(selection: Mapping[str, Sequence[tuple[str, str]]], *,
                    source_reads=None,
                    extra: Mapping[str, Sequence[str]] | None = None,
                    ) -> list[Span]:
    """Every non-empty tensor span the loader reads for ``selection``, sorted.

    ``extra`` adds ``{shard path: [checkpoint name, ...]}`` read beside the
    selection: the FP8 ``weight_scale_inv`` siblings the loader reads for
    FP8-sourced weights. Zero-byte tensors carry no payload (the strict
    reader builds them locally) and are left out.
    """
    wanted: dict[str, list[str]] = {}
    for shard, pairs in selection.items():
        wanted.setdefault(os.path.normpath(shard), []).extend(
            ckpt for _live, ckpt in pairs)
    for shard, names in (extra or {}).items():
        wanted.setdefault(os.path.normpath(shard), []).extend(names)
    spans: set[Span] = set()
    for path in sorted(wanted):
        header, base, size = read_safetensors_header(path, source_reads=source_reads)
        for name in wanted[path]:
            span = tensor_span(header, base, size, path, name)
            if span[2] > span[1]:
                spans.add(span)
    return sorted(spans)


def uncovered_spans(entries: Iterable[Mapping], spans: Iterable[Span]) -> list[Span]:
    """The ``spans`` that no single one of ``entries`` covers outright.

    One entry has to cover a span whole. The staged reader serves a span from
    the one staged range that contains it, and the strict tier policy refuses
    a span that straddles two ranges or that no range covers.
    """
    by_path: dict[str, list[tuple[int, int]]] = {}
    for entry in entries:
        by_path.setdefault(os.path.normpath(entry["path"]), []).append(
            (int(entry["offset"]), int(entry["offset"]) + int(entry["bytes"])))
    return [(path, start, end) for path, start, end in spans
            if not any(low <= start and end <= high
                       for low, high in by_path.get(os.path.normpath(path), ()))]


def chain_prefetch_window(order: Sequence[int], position: int,
                          lookahead: int) -> tuple[int, ...]:
    """The layers a walk prefetches after installing ``order[position]``.

    The next ``lookahead`` layers of the walk's own install order, and never
    a layer past its end. A walk that installs 44..40 prefetches nothing
    below 40: the next process to need layer 39 reads it itself, and a read
    here would be a source read its readset does not declare (PQ #1095).
    """
    if type(position) is not int or not 0 <= position < len(order):
        raise ValueError(f"position {position!r} is outside the install order")
    return tuple(order[position + 1:position + 1 + max(0, int(lookahead))])


def chain_opening_window(order: Sequence[int], lookahead: int) -> tuple[int, ...]:
    """The layers a walk asks for before its first install.

    ``install(require_prefetched=True)`` refuses a layer that is neither
    resident nor in flight, so the first layer, and up to ``lookahead`` in
    all, are asked for first, nearest first.
    """
    return tuple(order[:max(1, int(lookahead))])
