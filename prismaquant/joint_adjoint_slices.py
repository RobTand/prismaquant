"""Stage A slices and checkpoint bands (RobTand/prismaquant#993).

A layer quantum reads a small, fixed part of Stage A: the run's identity, its
stride, its boundary storage binding, ONE checkpoint and the forward boundary
entries its own chain walks. That part is the quantum's *slice*. Every Stage B
record, readset and cost payload binds the slice digest, never a whole
receipt, so a record built from a checkpoint band (sealed hours before the
receipt) and one built from the complete receipt are the same record.

Pure data, stdlib only and torch-free, like ``joint_layer_quanta``: a CPU
checkout derives, binds and checks slices. The strided checkpoint geometry
lives here too, and ``joint_adjoint_checkpoints`` re-exports it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .cost_stage_checkpoint import (
    canonical_json,
    canonical_json_bytes,
    canonical_json_sha256,
    publish_new_bytes,
)

ADJOINT_RECEIPT_SCHEMA = "prismaquant.joint_adjoint_capture.v1"
ADJOINT_CHECKPOINT_SCHEMA = "prismaquant.joint_adjoint_checkpoint.v1"


def derive_checkpoint_boundaries(num_layers: int, stride: int) -> tuple[int, ...]:
    """The strided cotangent checkpoint boundaries, tail first (§3.4).

    Multiples of ``stride`` below the tail, plus the tail boundary itself:
    45 layers at S=8 retains {45, 40, 32, 24, 16, 8} = ceil(45/8) = 6
    checkpoints, and every layer L chains at most S-1 = 7 render-free
    backwards from the nearest checkpoint at or above L+1. This is the same
    set the producer's ``derive_stride`` seals (``joint_layer_quanta`` D3/D4
    handshake notes, PR #785): stage A must publish exactly these boundaries
    or ``check_adjoint_run_header`` refuses the run. Order here is tail-first
    (stage A serializes the tail checkpoint first); the binding compares
    sorted sets, so order carries no identity.
    """
    num_layers, stride = int(num_layers), int(stride)
    if num_layers < 1:
        raise ValueError("checkpoint stride derivation needs a positive layer count")
    if stride < 1:
        raise ValueError("checkpoint stride must be a positive integer")
    return (num_layers, *[mark for mark in range(stride, num_layers, stride)
                          if mark != num_layers][::-1])


def chain_layers_for(checkpoint_boundary: int, layer: int) -> tuple[int, ...]:
    """The render-free chain a ``layer`` quantum walks from a checkpoint.

    Descending from ``checkpoint_boundary - 1`` down to and including
    ``layer + 1`` -- the layers whose backwards produce the cotangent at
    boundary ``layer + 1`` from the checkpoint's cotangent at
    ``checkpoint_boundary``. Empty when the checkpoint *is* the incoming
    cotangent (layer ``num_layers - 1`` at the tail).
    """
    checkpoint_boundary, layer = int(checkpoint_boundary), int(layer)
    if not layer + 1 <= checkpoint_boundary:
        raise ValueError(
            f"checkpoint boundary {checkpoint_boundary} is below layer {layer} + 1")
    return tuple(range(checkpoint_boundary - 1, layer, -1))


def nearest_checkpoint_boundary(boundaries, layer: int) -> int:
    """The nearest strided checkpoint at or above ``layer + 1``."""
    layer = int(layer)
    above = [int(boundary) for boundary in boundaries if int(boundary) >= layer + 1]
    if not above:
        raise ValueError(f"no checkpoint boundary at or above layer {layer} + 1")
    return min(above)



ADJOINT_BAND_SCHEMA = "prismaquant.joint_adjoint_capture.band.v1"

#: The run-level fields of a slice: identical for every layer of one run.
STAGE_A_RUN_HEADER_FIELDS = ("run_identity", "stride", "boundary_storage")

#: The ``boundary_storage`` fields a quantum reads. ``directory`` locates the
#: original capture namespace a catalog-extension quantum attaches to
#: (``joint_cost_quantum.quantum_adjoint_space``); ``forward_recovery`` is
#: present only on a run that resumed from a capsule.
STAGE_A_BOUNDARY_STORAGE_FIELDS = ("session", "policy", "directory", "forward_recovery")
_REQUIRED_BOUNDARY_STORAGE_FIELDS = ("session", "policy", "directory")

#: Every field of a slice. Nothing else of a receipt reaches a quantum.
STAGE_A_SLICE_FIELDS = (*STAGE_A_RUN_HEADER_FIELDS, "checkpoint", "boundary_entries")

_CHECKPOINT_SEALED_FIELDS = (
    "schema", "boundary", "session", "activation_entries", "shared_state_entries")


class AdjointSliceRefused(ValueError):
    """A receipt-like, slice or band that cannot serve the requested layer."""


def checkpoint_seal_sha256(record) -> str:
    """The digest a checkpoint manifest seals over its own fields."""
    return canonical_json_sha256(
        {key: record[key] for key in _CHECKPOINT_SEALED_FIELDS},
        where="adjoint checkpoint")


def stage_a_receipt_kind(receipt_like) -> str:
    """``"complete"`` or ``"band"``; anything else refuses.

    A running, failed or foreign document is neither: only a completed
    receipt or a sealed band carries Stage A inputs a quantum may bind.
    """
    if not isinstance(receipt_like, dict):
        raise AdjointSliceRefused("a Stage A receipt must be a JSON object")
    schema, status = receipt_like.get("schema"), receipt_like.get("status")
    if schema == ADJOINT_RECEIPT_SCHEMA and status == "complete":
        return "complete"
    if schema == ADJOINT_BAND_SCHEMA and status == "band":
        return "band"
    raise AdjointSliceRefused(
        "not a completed Stage A receipt or a sealed checkpoint band: "
        f"schema={schema!r} status={status!r}")


def band_layers(boundaries, boundary: int) -> tuple[int, ...]:
    """The layers whose nearest checkpoint is ``boundary``, descending.

    Band ``b`` serves every layer from the next lower stride boundary (0
    below the lowest) up to ``b - 1``; those layers' chains and own forward
    boundaries cover the same range, so the band carries exactly those
    boundary entries.
    """
    marks = sorted({int(mark) for mark in boundaries})
    boundary = int(boundary)
    if boundary not in marks:
        raise AdjointSliceRefused(
            f"boundary {boundary} is not a stride checkpoint {marks}")
    lower = max([mark for mark in marks if mark < boundary], default=0)
    return tuple(range(boundary - 1, lower - 1, -1))


def stage_a_run_header(receipt_like) -> dict:
    """The run-level part of a slice: equal for every layer and every band."""
    stage_a_receipt_kind(receipt_like)
    storage = receipt_like.get("boundary_storage")
    if not isinstance(storage, dict) or any(
            key not in storage for key in _REQUIRED_BOUNDARY_STORAGE_FIELDS):
        raise AdjointSliceRefused(
            "the Stage A receipt carries no complete boundary storage binding")
    for key in ("run_identity", "stride"):
        if not isinstance(receipt_like.get(key), dict):
            raise AdjointSliceRefused(f"the Stage A receipt carries no {key}")
    header = {
        "run_identity": receipt_like["run_identity"],
        "stride": receipt_like["stride"],
        "boundary_storage": {key: storage[key]
                             for key in STAGE_A_BOUNDARY_STORAGE_FIELDS
                             if key in storage},
    }
    return canonical_json(header, where="Stage A run header")


def stage_a_run_header_sha256(header) -> str:
    return canonical_json_sha256(header, where="Stage A run header")


def slice_run_header(adjoint_slice) -> dict:
    """The run header a slice carries (the slice minus its layer fields)."""
    return {key: adjoint_slice[key] for key in STAGE_A_RUN_HEADER_FIELDS}


def adjoint_slice_sha256(adjoint_slice) -> str:
    """The identity every Stage B record, readset and payload binds."""
    return canonical_json_sha256(adjoint_slice, where="Stage A slice")


def verify_adjoint_slice(adjoint_slice, *, layer: int,
                         checkpoint_boundary: int | None = None) -> int:
    """Refuse a slice that is not exactly the one ``layer`` reads.

    Checks the field set, that the checkpoint is the nearest stride boundary
    above ``layer`` and seals its own manifest under the run's boundary
    session, and that the boundary entries are exactly
    ``chain_layers_for(b, layer) ∪ {layer}``. Returns ``b``.
    """
    if not isinstance(adjoint_slice, dict) or set(adjoint_slice) != set(STAGE_A_SLICE_FIELDS):
        raise AdjointSliceRefused(
            "a Stage A slice carries exactly the fields "
            f"{sorted(STAGE_A_SLICE_FIELDS)}")
    storage = adjoint_slice["boundary_storage"]
    if (not isinstance(storage, dict)
            or not set(_REQUIRED_BOUNDARY_STORAGE_FIELDS) <= set(storage)
            or not set(storage) <= set(STAGE_A_BOUNDARY_STORAGE_FIELDS)):
        raise AdjointSliceRefused("a Stage A slice has a foreign boundary storage block")
    boundaries = adjoint_slice["stride"].get("boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        raise AdjointSliceRefused("a Stage A slice seals no stride boundaries")
    layer = int(layer)
    num_layers = max(int(mark) for mark in boundaries)
    if not 0 <= layer < num_layers:
        raise AdjointSliceRefused(f"layer {layer} is outside the {num_layers}-layer run")
    boundary = nearest_checkpoint_boundary(boundaries, layer)
    if checkpoint_boundary is not None and int(checkpoint_boundary) != boundary:
        raise AdjointSliceRefused(
            f"layer {layer} reads checkpoint {boundary}, not {checkpoint_boundary}")
    checkpoint = adjoint_slice["checkpoint"]
    if (not isinstance(checkpoint, dict)
            or checkpoint.get("schema") != ADJOINT_CHECKPOINT_SCHEMA
            or checkpoint.get("boundary") != boundary):
        raise AdjointSliceRefused(
            f"the slice checkpoint is not the sealed boundary {boundary} layer {layer} reads")
    try:
        sealed = checkpoint_seal_sha256(checkpoint)
    except (KeyError, ValueError) as exc:
        raise AdjointSliceRefused("the slice checkpoint is not a sealed manifest") from exc
    if checkpoint.get("cotangent_sha256") != sealed:
        raise AdjointSliceRefused(
            f"checkpoint {boundary} does not seal its own manifest")
    session = storage["session"]
    if checkpoint["session"] != {"generation": session.get("generation"),
                                 "kind": "adjoint_checkpoint",
                                 "run_identity_sha256": session.get("run_identity_sha256")}:
        raise AdjointSliceRefused(
            f"checkpoint {boundary} was sealed by another Stage A generation")
    needed = {str(k) for k in (*chain_layers_for(boundary, layer), layer)}
    entries = adjoint_slice["boundary_entries"]
    if not isinstance(entries, dict) or set(entries) != needed or any(
            not isinstance(rows, list) or not rows for rows in entries.values()):
        raise AdjointSliceRefused(
            f"layer {layer} reads boundary entries {sorted(needed, key=int)}")
    return boundary


def stage_a_slice(receipt_like, layer: int) -> dict:
    """The only Stage A inputs a quantum for ``layer`` may read.

    Applies equally to a complete receipt and to a sealed band. Returns the
    run header plus the checkpoint record for
    ``b = nearest_checkpoint_boundary(layer)`` and ``boundary_entries[k]``
    for every ``k`` in ``chain_layers_for(b, layer) ∪ {layer}``. A band must
    be the band of ``b``: any other band refuses rather than serving a layer
    it does not own.
    """
    kind = stage_a_receipt_kind(receipt_like)
    header = stage_a_run_header(receipt_like)
    boundaries = header["stride"].get("boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        raise AdjointSliceRefused("the Stage A receipt seals no stride boundaries")
    layer = int(layer)
    boundary = nearest_checkpoint_boundary(boundaries, layer)
    if kind == "band":
        band = receipt_like.get("band")
        if not isinstance(band, dict) or band.get("boundary") != boundary:
            raise AdjointSliceRefused(
                f"a band for checkpoint {band.get('boundary') if isinstance(band, dict) else None!r} "
                f"cannot serve layer {layer}: its nearest checkpoint is {boundary}")
    checkpoints = receipt_like.get("checkpoints")
    matches = [record for record in (checkpoints if isinstance(checkpoints, list) else [])
               if isinstance(record, dict) and record.get("boundary") == boundary]
    if len(matches) != 1:
        raise AdjointSliceRefused(
            f"the Stage A receipt carries {len(matches)} checkpoint records "
            f"for boundary {boundary}, not one")
    table = receipt_like.get("boundary_entries")
    if not isinstance(table, dict):
        raise AdjointSliceRefused("the Stage A receipt carries no boundary entries")
    entries = {}
    for k in sorted({*chain_layers_for(boundary, layer), layer}):
        rows = table.get(str(k))
        if not isinstance(rows, list) or not rows:
            raise AdjointSliceRefused(
                f"the Stage A receipt carries no boundary {k} entries for layer {layer}")
        entries[str(k)] = rows
    adjoint_slice = canonical_json(
        {**header, "checkpoint": matches[0], "boundary_entries": entries},
        where="Stage A slice")
    verify_adjoint_slice(adjoint_slice, layer=layer)
    return adjoint_slice


def validate_band_receipt(band) -> int:
    """Refuse a band whose fields do not describe one sealed checkpoint band.

    Returns the band's checkpoint boundary. Every layer the band names must
    derive its slice from it.
    """
    if stage_a_receipt_kind(band) != "band":
        raise AdjointSliceRefused("not a sealed checkpoint band")
    header = stage_a_run_header(band)
    marker = band.get("band")
    if not isinstance(marker, dict) or type(marker.get("boundary")) is not int:
        raise AdjointSliceRefused("a band names no checkpoint boundary")
    boundary = marker["boundary"]
    layers = band_layers(header["stride"]["boundaries"], boundary)
    if marker.get("layers") != list(layers):
        raise AdjointSliceRefused(
            f"band {boundary} must serve layers {list(layers)}")
    if [record.get("boundary") for record in band.get("checkpoints", [])] != [boundary]:
        raise AdjointSliceRefused(f"band {boundary} must carry exactly its own checkpoint")
    if set(band.get("boundary_entries", {})) != {str(k) for k in layers}:
        raise AdjointSliceRefused(
            f"band {boundary} must carry exactly the boundary entries of layers {list(layers)}")
    for layer in layers:
        stage_a_slice(band, layer)
    return boundary


def band_set(bands) -> dict[int, dict]:
    """Index bands by checkpoint; refuse duplicates and mixed runs.

    Every band of one set shares one run header: the same run identity,
    stride and boundary storage session. A band of any other run refuses.
    """
    indexed: dict[int, dict] = {}
    header_sha = None
    for band in bands:
        boundary = validate_band_receipt(band)
        digest = stage_a_run_header_sha256(stage_a_run_header(band))
        if header_sha is None:
            header_sha = digest
        elif digest != header_sha:
            raise AdjointSliceRefused(
                f"band {boundary} carries another Stage A run header: mixed runs")
        if boundary in indexed:
            if canonical_json_sha256(indexed[boundary], where="band") != canonical_json_sha256(
                    band, where="band"):
                raise AdjointSliceRefused(f"two different bands claim checkpoint {boundary}")
        indexed[boundary] = band
    return indexed


def missing_band_boundaries(bands: dict[int, dict]) -> list[int]:
    """Stride checkpoints (tail down to the lowest) with no band, descending."""
    if not bands:
        raise AdjointSliceRefused("an empty band set covers no checkpoint")
    header = stage_a_run_header(next(iter(bands.values())))
    return sorted((int(mark) for mark in header["stride"]["boundaries"]
                   if int(mark) not in bands), reverse=True)


def write_band_receipt(path: str | os.PathLike, band: dict) -> str:
    """Publish a band once; re-publishing the same bytes is a no-op.

    Returns the file's sha256. A differing band at the same path refuses:
    a band, like the receipt, is never overwritten.
    """
    validate_band_receipt(band)
    payload = (json.dumps(band, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path = Path(path)
    if not publish_new_bytes(path, payload) and path.read_bytes() != payload:
        raise RuntimeError(
            f"a different checkpoint band already exists at {path}; a band "
            "is never overwritten")
    return hashlib.sha256(payload).hexdigest()


def load_stage_a_receipt_like(path: str | os.PathLike, sha256: str | None = None) -> dict:
    """Read a complete receipt or a sealed band, file digest checked when given."""
    raw = Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != str(sha256):
        raise RuntimeError(f"Stage A receipt digest mismatch at {path}")
    document = json.loads(raw)
    if stage_a_receipt_kind(document) == "band":
        validate_band_receipt(document)
    return document


def adjoint_slice_bytes(adjoint_slice) -> bytes:
    """The exact bytes a slice file carries: its canonical JSON.

    The file digest therefore equals :func:`adjoint_slice_sha256`, so one
    value names both the file and the slice identity.
    """
    return canonical_json_bytes(adjoint_slice, where="Stage A slice")


def write_adjoint_slice(path: str | os.PathLike, adjoint_slice, *, layer: int) -> str:
    """Publish one quantum's slice once; the same bytes again are a no-op.

    Returns the slice digest. A differing slice at the same path refuses:
    the record naming the path binds its digest, so the file never changes.
    """
    verify_adjoint_slice(adjoint_slice, layer=layer)
    payload = adjoint_slice_bytes(adjoint_slice)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not publish_new_bytes(path, payload) and path.read_bytes() != payload:
        raise RuntimeError(
            f"a different Stage A slice already exists at {path}; a slice "
            "is never overwritten")
    return hashlib.sha256(payload).hexdigest()


def load_adjoint_slice(path: str | os.PathLike, sha256: str, *, layer: int,
                       checkpoint_boundary: int | None = None) -> dict:
    """Read one quantum's slice: digest, canonical form and layer checked.

    The file must be the canonical encoding of its content, so its byte
    digest is the slice identity the record binds; then
    :func:`verify_adjoint_slice` proves it is exactly the slice ``layer``
    reads.
    """
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != str(sha256):
        raise AdjointSliceRefused(f"Stage A slice digest mismatch at {path}")
    adjoint_slice = json.loads(raw)
    if adjoint_slice_bytes(adjoint_slice) != raw:
        raise AdjointSliceRefused(f"the Stage A slice at {path} is not canonical JSON")
    verify_adjoint_slice(adjoint_slice, layer=layer,
                         checkpoint_boundary=checkpoint_boundary)
    return adjoint_slice
