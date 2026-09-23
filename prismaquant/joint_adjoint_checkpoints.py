"""Strided adjoint checkpoints and the render-free cotangent chain.

The shared machinery of the distributed joint-AURA cost campaign
(`docs/design/distributed_campaign_2026-09-19.md`): the checkpoint
serialization stage A publishes (§3.3/§3.4), the digest-checked loaders the
per-layer quanta read them back through (§6.2), and the one render-free
reverse step both stages run -- the single consumer's completed-layer leg,
reused rather than reimplemented, so the chain arithmetic a quantum replays
is by construction the arithmetic ``prismaquant.aura_cost`` runs when a layer
has no pending units (§2.2 stage A, §6.2 step 3).

This module owns no cache and no schedule. Boundary tensors move through the
existing activation artifact owner's exact-entry writer/reader pair
(``perturbed_x_cache``); every checkpoint byte is digest-checked on load.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from .cost_stage_checkpoint import (
    atomic_write_bytes,
    canonical_json,
    canonical_json_sha256,
    publish_new_bytes,
)

from .joint_adjoint_slices import (  # noqa: F401 -- re-exported: one spelling
    ADJOINT_BAND_SCHEMA,
    ADJOINT_CHECKPOINT_SCHEMA,
    ADJOINT_RECEIPT_SCHEMA,
    CHAIN_REGIME_KEY,
    DEFAULT_CHAIN_REGIME,
    STAGE_A_BOUNDARY_STORAGE_FIELDS,
    STAGE_A_RUN_HEADER_FIELDS,
    STAGE_A_SLICE_FIELDS,
    AdjointSliceRefused,
    ChainRegimeRefused,
    adjoint_slice_sha256,
    band_layers,
    band_set,
    chain_layers_for,
    chain_regime_identity,
    chain_regime_of,
    normalize_chain_regime,
    require_chain_regime,
    checkpoint_seal_sha256,
    derive_checkpoint_boundaries,
    load_stage_a_receipt_like,
    missing_band_boundaries,
    nearest_checkpoint_boundary,
    slice_run_header,
    stage_a_receipt_kind,
    stage_a_run_header,
    stage_a_run_header_sha256,
    stage_a_slice,
    validate_band_receipt,
    verify_adjoint_slice,
    write_band_receipt,
)

QUANTUM_COUNTERS_SCHEMA = "prismaquant.joint_layer_quantum.counters.v1"
QUANTUM_STATUS_SCHEMA = "prismaquant.joint_layer_quantum.status.v1"
QUANTUM_RECORD_SCHEMA = "prismaquant.joint_layer_quanta.v1"

#: The entry-point string the plan block and the adjoint read manifest carry
#: (contract §5.1/§4.1). ``prismaquant.joint_cost_stage_a`` is the executing
#: module; this name is the lane's sealed identifier.
ADJOINT_CAPTURE_ENTRY_POINT = "prismaquant.joint_adjoint_capture"

DEFAULT_STRIDE = 8


def adjoint_space(output_root: str | os.PathLike) -> Path:
    """The distributed campaign's adjoint namespace (§9.1: never the single
    run's ``run/``/``checkpoints/``/``exact-boundaries/``)."""
    return Path(output_root) / "layer-quanta" / "adjoint"


def boundary_entry_directory(space: str | os.PathLike) -> Path:
    return Path(space) / "exact-boundaries"


def checkpoint_directory(space: str | os.PathLike, boundary: int) -> Path:
    return Path(space) / "checkpoints" / f"boundary-{int(boundary):03d}"


_CHECKPOINT_NAME = re.compile(r"boundary-(\d{3,})")


def occupied_checkpoint_directories(space: str | os.PathLike) -> list[Path]:
    """Every ``checkpoints/boundary-NNN`` path that already exists in ``space``.

    ``write_adjoint_checkpoint`` creates each checkpoint directory with
    ``mkdir(exist_ok=False)`` and nothing else writes under ``checkpoints/``,
    so a name in the writer's own spelling (``checkpoint_directory``) is a
    path a capture of this campaign writes and no capture ever reuses. The
    layer count, and with it the tail boundary, is known only after the model
    is built, so the check matches the spelling rather than one stride's set:
    a stale tail checkpoint (R10 left ``boundary-045``, the 45-layer tail) is
    exactly the one a stride-only set would miss. A path renamed aside
    (``boundary-045.superseded``) no longer matches and does not block.
    """
    root = Path(space) / "checkpoints"
    if not root.is_dir():
        return []
    occupied = []
    for path in root.iterdir():
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if match and checkpoint_directory(space, int(match[1])).name == path.name:
            occupied.append(path)
    return sorted(occupied)


def adjoint_receipt_path(space: str | os.PathLike) -> Path:
    return Path(space) / "adjoint-capture.json"


# --------------------------------------------------------------------------
# Exact-entry (de)serialization: reuse the activation artifact owner's writer
# and reader so checkpoint tensors get the same per-file SHA verification the
# boundary entries themselves carry (AGENTS.md principle 3: no parallel cache).
# --------------------------------------------------------------------------


def write_checkpoint_cotangent_entry(
    checkpoint_dir: Path, *, probe_index: int, batch_index: int, tensor: torch.Tensor,
    session: dict, max_file_bytes: int | None = None,
) -> dict:
    """Publish one activation cotangent as a digest-checked exact entry.

    ``max_file_bytes`` carries the reservation-admitted per-file envelope;
    when omitted the legacy tensor-bytes-plus-header envelope applies. The
    exact writer enforces whichever bound it receives and unlinks on
    exceedance.
    """
    from .perturbed_x_cache import write_exact_activation_cache_entry

    name = f"cotangent-{int(probe_index)}-{int(batch_index)}"
    nbytes = tensor.numel() * tensor.element_size()
    if max_file_bytes is None:
        max_file_bytes = nbytes + 65536
    identity = {
        "session": dict(session),
        "slot": name,
        "kind": "adjoint_checkpoint_cotangent",
        "coordinates": {
            "probe": int(probe_index), "batch": int(batch_index),
        },
    }
    reference = write_exact_activation_cache_entry(
        checkpoint_dir / "entries", name, tensor,
        identity=identity, max_tensor_bytes=nbytes, max_file_bytes=max_file_bytes,
    )
    return exact_entry_record(reference)


def exact_entry_record(reference) -> dict:
    """The JSON-safe record of one exact entry, for receipts and manifests.

    Carries the writer's full metadata identity: the verified reader compares
    the entry's embedded metadata against the reference byte for byte, so a
    record that cannot rebuild that reference cannot be read back.
    """
    from .perturbed_x_cache import EXACT_ACTIVATION_SCHEMA

    metadata = json.loads(reference.metadata_json)
    return {
        "name": reference.name,
        "path": reference.path,
        "sha256": reference.sha256,
        "tensor_bytes": int(reference.tensor_bytes),
        "file_bytes": int(reference.file_bytes),
        "shape": [int(dim) for dim in reference.shape],
        "dtype": str(reference.dtype),
        "metadata": {
            "schema": metadata.get("schema", EXACT_ACTIVATION_SCHEMA),
            "identity": metadata["identity"],
            "shape": metadata["shape"],
            "dtype": metadata["dtype"],
            "tensor_bytes": metadata["tensor_bytes"],
        },
    }


def reference_from_record(record: dict):
    """Rebuild the immutable reference a record stands for."""
    from .perturbed_x_cache import ExactActivationReference

    metadata = record["metadata"]
    return ExactActivationReference(
        path=str(record["path"]),
        name=str(record["name"]),
        metadata_json=json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False),
        shape=tuple(int(dim) for dim in record["shape"]),
        dtype=str(record["dtype"]),
        tensor_bytes=int(record["tensor_bytes"]),
        file_bytes=int(record["file_bytes"]),
        sha256=str(record["sha256"]),
    )


def read_exact_entry_tensors(records, *, expected_session) -> dict:
    """Read whole exact entries back, digest-verified, name -> CPU tensor.

    Uses the activation owner's verified window reader (hash-then-load in one
    pass), sized to the caller's entry list. Checkpoint restoration passes one
    entry at a time so no read window owns the complete cotangent plane.
    """
    from .perturbed_x_cache import prefetch_exact_activation_cache_entries

    tensors = {}
    records = tuple(records)
    if not records:
        return tensors
    total = sum(int(record["tensor_bytes"]) for record in records)
    references = [reference_from_record(record) for record in records]
    with prefetch_exact_activation_cache_entries(
        references, max_tensor_bytes=total, expected_session=expected_session,
    ) as window:
        for reference in references:
            tensors[reference.name] = window.get(reference)
    return tensors


# --------------------------------------------------------------------------
# Checkpoint write/read
# --------------------------------------------------------------------------


def _shared_state_entry(checkpoint_dir: Path, name: str, state) -> dict:
    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    return _write_shared_state_payload(checkpoint_dir, name, payload)


def _write_shared_state_payload(checkpoint_dir: Path, name: str, payload: bytes) -> dict:
    path = checkpoint_dir / "entries" / f"{name}.pkl"
    atomic_write_bytes(path, payload)
    return {
        "name": name,
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "file_bytes": len(payload),
    }


def _write_shared_state_streaming(checkpoint_dir: Path, name: str, state,
                                  *, max_file_bytes: int) -> dict:
    """Stream one shared state to its pickle file with an admitted bound.

    Standard ``pickle.dump`` into the existing ``SerializedEntryDigest``
    sink over an atomic temp file: no aggregate payload object is ever
    materialized, so the transient peak stays at the entry's own
    serialization instead of the whole checkpoint. The digest covers
    exactly the published bytes (no rehash pass); the admitted per-file
    envelope is enforced on every sink write before the underlying handle
    can cross it, and the temp is unlinked on exceedance, so an oversized
    entry never publishes. Not a new serializer or cache: stdlib pickling
    plus the activation owner's sink and atomic-publication shape.
    """
    import os

    from .cost_stage_checkpoint import unique_temp_suffix
    from .perturbed_x_cache import SerializedEntryDigest

    if type(max_file_bytes) is not int or max_file_bytes <= 0:
        raise RuntimeError(
            "exact boundary checkpoint shared-state write needs an "
            "admitted per-file envelope")
    path = checkpoint_dir / "entries" / f"{name}.pkl"
    if path.exists() or path.with_suffix(".pkl.tmp").exists():
        raise RuntimeError("exact boundary checkpoint entry already exists")
    sink = _BoundedDigestSink(SerializedEntryDigest(), max_file_bytes,
                              label=f"shared state {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + unique_temp_suffix())
    try:
        with temporary.open("wb") as handle:
            pickle.dump(state, sink.sink(handle),
                        protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        published = temporary.stat().st_size
        if published != sink.bytes_written:
            raise RuntimeError(
                "exact boundary checkpoint entry differs from its "
                "serialized bytes")
        if published > max_file_bytes:
            raise RuntimeError(
                "exact boundary checkpoint shared-state file exceeds its "
                f"admitted envelope for {name}")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "name": name,
        "path": str(path),
        "sha256": sink.hexdigest(),
        "file_bytes": published,
    }


class _BoundedDigestSink:
    """Hash-while-writing sink that refuses before crossing an admitted ceiling.

    Wraps ``SerializedEntryDigest`` (activation owner) without changing it:
    every ``write`` is checked against the remaining admitted bytes first,
    so the underlying handle -- and therefore the staged temp file -- never
    holds more than admitted. Unbounded callers keep today's behavior by
    passing no ceiling; the exact writer's own post-write check is untouched.
    """

    def __init__(self, digest, max_bytes: int | None, label: str):
        self._digest = digest
        self._label = str(label)
        if max_bytes is not None and (
                type(max_bytes) is not int or max_bytes <= 0):
            raise RuntimeError(
                "exact boundary bounded sink needs a positive byte ceiling")
        self._ceiling = max_bytes
        self._written = 0

    def sink(self, handle):
        self._digest.sink(handle)
        return self

    def write(self, data):
        view = memoryview(data).cast("B")
        try:
            if (self._ceiling is not None
                    and self._written + view.nbytes > self._ceiling):
                raise RuntimeError(
                    "exact boundary checkpoint entry exceeds its admitted "
                    f"envelope for {self._label}")
            self._digest.write(view)
            self._written += view.nbytes
            return view.nbytes
        finally:
            view.release()

    def flush(self):
        self._digest.flush()

    def hexdigest(self):
        return self._digest.hexdigest()

    @property
    def bytes_written(self):
        return self._written


#: Per-object framing bound for the shared-state pickle size estimate below.
#: Admission-side upper bound only: the reservation commits exact serialized
#: lengths, exactly like the exact-entry writer's small PyTorch zip header
#: envelope (tensor bytes + 65536) that bounds each activation file.
_PICKLE_ESTIMATE_FRAMING_BYTES = 65536


def _shared_state_envelope_estimate(value) -> int:
    """Upper-bound the pickle size of shared checkpoint state without serializing.

    Closed grammar mirroring ``_state_tensors`` (cost_streaming): anything
    else -- meta or non-strided tensors, sets, opaque objects -- refuses
    instead of guessing. Two precision rules matter here:

    - Tensors count their actual serialized backing ownership
      (``untyped_storage().nbytes()``), once per leaf, because pickling a
      small tensor view serializes its whole backing storage, which
      ``numel`` would undercount. Measured on the pinned torch (PB
      ``4bbc5f134a53``): two 2 MiB views sharing one 4 MiB backing dump
      ~8.00 MB in one call, the same as separately -- pickle emits
      per-view backing bytes rather than memoizing shared storage, so no
      cross-leaf deduplication is sound. (Pickle memoizes only the
      identical object, which per-leaf summation already covers as an
      upper bound.) Each shared entry dumps separately, so ownership is
      per entry, never across entries.
    - Integers are sized by bit length: a short counter costs framing, but
      an arbitrary-precision giant must count its digits, never a flat 128.

    The estimate only admits the serialization attempt against the remaining
    envelope; the reservation commits the exact dumped lengths.
    """
    if isinstance(value, torch.Tensor):
        if value.is_meta or value.layout != torch.strided:
            raise TypeError(
                "exact boundary checkpoint cannot account this state tensor")
        return (value.untyped_storage().nbytes()
                + _PICKLE_ESTIMATE_FRAMING_BYTES)
    if isinstance(value, str):
        return len(value.encode("utf-8")) + 128
    if isinstance(value, bytes):
        return len(value) + 128
    if value is None or isinstance(value, (bool, float, complex)):
        return 128
    if isinstance(value, int):
        return (int(value).bit_length() + 7) // 8 + 32
    if isinstance(value, dict):
        total = 128
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise TypeError(
                    "exact boundary checkpoint cannot account shared state "
                    f"with {type(key).__name__} mapping keys")
            total += _shared_state_envelope_estimate(key)
            total += _shared_state_envelope_estimate(item)
        return total
    if isinstance(value, (list, tuple)):
        total = 128
        for item in value:
            total += _shared_state_envelope_estimate(item)
        return total
    raise TypeError(
        "exact boundary checkpoint cannot account opaque shared state "
        f"{type(value).__name__}")


def _checkpoint_coordinate_keys(mapping, *, arity: int, where: str) -> None:
    """Require exact nonnegative-int coordinates on the budgeted path.

    The legacy writer coerces coordinates with ``int()``; the budgeted path
    refuses anything but exact integers first, so a malformed key can never
    reach the envelope arithmetic.
    """
    if not isinstance(mapping, dict):
        raise RuntimeError(f"exact boundary checkpoint {where} is not a mapping")
    for key in mapping:
        if (not isinstance(key, tuple) or len(key) != arity
                or any(type(part) is not int or part < 0 for part in key)):
            raise RuntimeError(
                f"exact boundary checkpoint {where} keys must be "
                f"{arity} nonnegative integers")


def _checkpoint_manifest_envelope_bytes(*, boundary: int, session: dict,
                                        activation_plan, shared_plan) -> int:
    """Upper-bound the checkpoint manifest size before digests exist.

    Builds the exact manifest shape with 64-character placeholder digests
    and per-file envelope sizes. Every placeholder field is byte-identical
    to the final manifest except ``file_bytes`` (envelope values, whose
    decimal width can only shrink as actuals come in at or under envelope)
    and ``sha256``/``cotangent_sha256`` (fixed 64 characters either way),
    so the final manifest is never longer than this number.
    """
    from .perturbed_x_cache import EXACT_ACTIVATION_SCHEMA

    canonical_session = canonical_json(dict(session), where="adjoint checkpoint session")

    def _activation_row(plan):
        return {
            "name": plan["name"], "path": plan["path"], "sha256": "0" * 64,
            "tensor_bytes": plan["tensor_bytes"], "file_bytes": plan["file_envelope"],
            "shape": plan["shape"], "dtype": plan["dtype"],
            "metadata": {
                "schema": EXACT_ACTIVATION_SCHEMA,
                "identity": {
                    "session": canonical_session, "slot": plan["slot"],
                    "kind": "adjoint_checkpoint_cotangent",
                    "coordinates": {"probe": plan["probe_index"],
                                    "batch": plan["batch_index"]},
                },
                "shape": plan["shape"], "dtype": plan["dtype"],
                "tensor_bytes": plan["tensor_bytes"],
            },
        }

    skeleton = {
        "schema": ADJOINT_CHECKPOINT_SCHEMA,
        "boundary": int(boundary),
        "session": canonical_session,
        "activation_entries": sorted(
            (_activation_row(plan) for plan in activation_plan),
            key=lambda row: row["name"]),
        "shared_state_entries": sorted(
            ({"name": plan["name"], "path": plan["path"], "sha256": "0" * 64,
              "file_bytes": plan["file_envelope"]}
             for plan in shared_plan),
            key=lambda row: row["name"]),
        "cotangent_sha256": "0" * 64,
    }
    return len((json.dumps(skeleton, sort_keys=True, indent=2, allow_nan=False)
                + "\n").encode())


def _checkpoint_tensor_spec(value, owner):
    """Plan bytes without loading an owner's durable cotangent descriptor."""
    from .perturbed_x_cache import ExactActivationReference

    if isinstance(value, ExactActivationReference) and owner is not None:
        # Refuse stale/foreign descriptors before reserving or creating files.
        owner._entry_identity(value)
        return value.tensor_bytes, list(value.shape), value.dtype
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.is_meta):
        raise TypeError(
            "exact boundary checkpoint requires materialized strided tensors "
            "or an owner's exact-entry references")
    return value.numel() * value.element_size(), list(value.shape), str(value.dtype)


def _checkpoint_plan_windows(plans, width):
    """Metadata-only groups matching the owner's probe/batch read windows."""
    from itertools import groupby

    for _, group in groupby(
            plans, key=lambda plan: (plan["probe_index"], plan["batch_index"] // width)):
        yield list(group)


def _checkpoint_window_references(cotangents, plans):
    """The owner's entries one serializer window reads, in order."""
    from .perturbed_x_cache import ExactActivationReference

    references = [cotangents[(plan["probe_index"], plan["batch_index"])]
                  for plan in plans]
    return [value for value in references
            if isinstance(value, ExactActivationReference)]


def _checkpoint_tensor_window(cotangents, plans, owner):
    """Reuse the owner's bounded read grouping, including PB group retirement.

    A one-entry window would retire and restage a complete produced group
    for each of its entries. Match the existing prefetch batch width, keeping
    only that window plus one serializer's reservation resident.
    """
    references = _checkpoint_window_references(cotangents, plans)
    return owner.prefetch(references) if references else nullcontext()


def write_adjoint_checkpoint(
    space: str | os.PathLike, *, boundary: int, session: dict,
    cotangents, shared_adjoint, shared_pass, owner=None,
) -> dict:
    """Serialize one strided checkpoint; returns its §3.3 record.

    ``cotangents`` maps ``(probe, batch)`` -> CPU activation cotangent tensor
    at ``boundary``, or (with an owner) its existing exact-entry reference.
    Reference metadata plans admission without reading payloads; serialization
    opens one accounted, digest-verified, lease-held owner window at a time,
    bounded by the existing ``prefetch_batches`` policy.
    No full tensor plane is retained. ``shared_adjoint`` maps ``(probe, batch)`` -> the
    ``SharedStateCotangents.state_dict()`` carried beside it. ``shared_pass``
    maps ``batch`` -> the captured forward shared-pass state each chain layer
    and the layer quantum recompute from.

    With ``owner=None`` the legacy unwatched path runs exactly as before.
    With a ``StreamedBoundaryArtifacts`` owner, the whole attempt is admitted
    against ``max_artifact_bytes`` first: tensor envelopes, estimated then
    exact shared-state payloads, the manifest envelope, and one in-progress
    temp overlap, counted alongside live ordinary entries. The writer then
    commits the receipt's actual bytes/digests, returns unused envelope, and
    retains (never silently releases) on failure. See the owner's
    reserve-before-write contract.
    """
    checkpoint_dir = checkpoint_directory(space, boundary)
    if owner is None:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        activation_entries = []
        for (probe_index, batch_index) in sorted(cotangents):
            tensor = cotangents[(probe_index, batch_index)]
            activation_entries.append(write_checkpoint_cotangent_entry(
                checkpoint_dir, probe_index=probe_index, batch_index=batch_index,
                tensor=tensor, session=session))
        shared_state_entries = []
        for (probe_index, batch_index) in sorted(shared_adjoint):
            shared_state_entries.append(_shared_state_entry(
                checkpoint_dir,
                f"shared-adjoint-{int(probe_index)}-{int(batch_index)}",
                shared_adjoint[(probe_index, batch_index)]))
        for batch_index in sorted(shared_pass):
            shared_state_entries.append(_shared_state_entry(
                checkpoint_dir, f"shared-pass-{int(batch_index)}",
                shared_pass[batch_index]))
        record = {
            "schema": ADJOINT_CHECKPOINT_SCHEMA,
            "boundary": int(boundary),
            "session": canonical_json(dict(session), where="adjoint checkpoint session"),
            "activation_entries": sorted(activation_entries, key=lambda e: e["name"]),
            "shared_state_entries": sorted(shared_state_entries, key=lambda e: e["name"]),
        }
        record["cotangent_sha256"] = canonical_json_sha256(
            {key: record[key] for key in
             ("schema", "boundary", "session", "activation_entries", "shared_state_entries")},
            where="adjoint checkpoint",
        )
        atomic_write_bytes(
            checkpoint_dir / "checkpoint.json",
            (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
        )
        return record

    if type(boundary) is not int or boundary < 0:
        raise RuntimeError("exact boundary checkpoint boundary must be a nonnegative integer")
    _checkpoint_coordinate_keys(cotangents, arity=2, where="cotangents")
    _checkpoint_coordinate_keys(shared_adjoint, arity=2, where="shared_adjoint")
    if not isinstance(shared_pass, dict) or any(
            type(part) is not int or part < 0 for part in shared_pass):
        raise RuntimeError(
            "exact boundary checkpoint shared_pass keys must be nonnegative integers")
    for value in cotangents.values():
        _checkpoint_tensor_spec(value, owner)
    owner.check_transient_buffer("checkpoint shared-state serialization")
    estimates = {}
    for name, state in _iter_shared_states(shared_adjoint, shared_pass):
        try:
            estimates[name] = _shared_state_envelope_estimate(state)
        except TypeError as exc:
            raise RuntimeError(
                "exact boundary checkpoint refuses unaccountable shared state "
                f"for {name}: {exc}") from exc
    from .perturbed_x_cache import activation_cache_filename

    activation_plan = []
    for (probe_index, batch_index) in sorted(cotangents):
        value = cotangents[(probe_index, batch_index)]
        nbytes, shape, dtype = _checkpoint_tensor_spec(value, owner)
        name = f"cotangent-{probe_index}-{batch_index}"
        activation_plan.append({
            "probe_index": probe_index, "batch_index": batch_index,
            "slot": name,
            "name": name,
            "path": str(checkpoint_dir / "entries" / activation_cache_filename(name)),
            "tensor_bytes": nbytes, "file_envelope": nbytes + 65536,
            "shape": [int(dim) for dim in shape],
            "dtype": str(dtype),
        })
    shared_plan = [{"name": name,
                    "path": str(checkpoint_dir / "entries" / f"{name}.pkl"),
                    "file_envelope": estimates[name]} for name in sorted(estimates)]
    manifest_envelope = _checkpoint_manifest_envelope_bytes(
        boundary=boundary, session=session,
        activation_plan=activation_plan, shared_plan=shared_plan)
    file_envelopes = ([plan["file_envelope"] for plan in activation_plan]
                      + [plan["file_envelope"] for plan in shared_plan])
    temp_overlap = max(file_envelopes + [manifest_envelope])
    envelope = sum(file_envelopes) + manifest_envelope + temp_overlap
    file_plan = {
        "files": [{"name": plan["name"], "path": plan["path"],
                   "envelope_bytes": plan["file_envelope"]}
                  for plan in activation_plan]
        + [{"name": plan["name"], "path": plan["path"],
            "envelope_bytes": plan["file_envelope"]}
           for plan in shared_plan],
        "manifest_bytes": manifest_envelope,
        "temp_overlap_bytes": temp_overlap,
        "envelope_bytes": envelope,
    }
    reservation = owner.reserve_checkpoint_artifact(
        label=f"adjoint checkpoint boundary {boundary}",
        envelope_bytes=envelope, file_plan=file_plan,
        checkpoint_dir=checkpoint_dir)
    created = False
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        created = True
    except BaseException:
        owner.cancel_checkpoint_artifact(reservation)
        raise
    try:
        # Shared states stream, write, and release one entry at a time: no
        # aggregate payload object ever exists, so the transient peak is the
        # largest single entry's serialization, never the whole checkpoint.
        # An empty shared set simply skips the loop. Holds count live
        # auxiliary usage plus transient bytes against the one auxiliary
        # ceiling.
        shared_state_entries = []
        for plan in shared_plan:
            state = _shared_state_by_name(
                shared_adjoint, shared_pass, plan["name"])
            with owner.hold_transient_metadata(
                    estimates[plan["name"]],
                    f"checkpoint shared state {plan['name']}"):
                shared_state_entries.append(_write_shared_state_streaming(
                    checkpoint_dir, plan["name"], state,
                    max_file_bytes=plan["file_envelope"]))
        from .perturbed_x_cache import ExactActivationReference

        activation_entries = []
        windows = list(_checkpoint_plan_windows(
            activation_plan, int(owner.config["prefetch_batches"])))
        # Staging only: each open window asks for the next one's groups, so
        # their movers run while this one serializes (RobTand/prismaquant#989).
        lookahead = getattr(owner, "stage_produced_reads_ahead", None)
        for position, window_plans in enumerate(windows):
            with _checkpoint_tensor_window(cotangents, window_plans, owner) as window:
                if lookahead is not None and position + 1 < len(windows):
                    following = _checkpoint_window_references(
                        cotangents, windows[position + 1])
                    if following:
                        lookahead(following)
                for plan in window_plans:
                    value = cotangents[(plan["probe_index"], plan["batch_index"])]
                    tensor = (owner.get(window, value)
                              if isinstance(value, ExactActivationReference) else value)
                    try:
                        with owner.hold_transient_serialization(
                                plan["tensor_bytes"],
                                f"checkpoint tensor {plan['name']}"):
                            entry = write_checkpoint_cotangent_entry(
                                checkpoint_dir, probe_index=plan["probe_index"],
                                batch_index=plan["batch_index"], tensor=tensor,
                                session=session, max_file_bytes=plan["file_envelope"])
                    finally:
                        # Release before the window closes/reuses its scratch.
                        tensor = None
                    if entry["file_bytes"] > plan["file_envelope"]:
                        raise RuntimeError(
                            "exact boundary checkpoint tensor file exceeds its "
                            f"admitted envelope for {plan['name']}")
                    activation_entries.append(entry)
        record = {
            "schema": ADJOINT_CHECKPOINT_SCHEMA,
            "boundary": int(boundary),
            "session": canonical_json(dict(session), where="adjoint checkpoint session"),
            "activation_entries": sorted(activation_entries, key=lambda e: e["name"]),
            "shared_state_entries": sorted(shared_state_entries, key=lambda e: e["name"]),
        }
        record["cotangent_sha256"] = canonical_json_sha256(
            {key: record[key] for key in
             ("schema", "boundary", "session", "activation_entries", "shared_state_entries")},
            where="adjoint checkpoint",
        )
        with owner.hold_transient_metadata(
                manifest_envelope, "checkpoint manifest"):
            manifest_payload = (json.dumps(
                record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            if len(manifest_payload) > manifest_envelope:
                raise RuntimeError(
                    "exact boundary checkpoint manifest exceeds its admitted envelope")
            atomic_write_bytes(checkpoint_dir / "checkpoint.json", manifest_payload)
        owner.commit_checkpoint_artifact(reservation, record)
    except BaseException:
        # Abandon only what is still active. A commit that already recorded
        # the receipt (even one whose trailing memory hook then failed) is
        # truthful committed state, and abandoning it would mask the
        # original error; a retained attempt is already retained
        # (abandon is idempotent there, so this simply skips it).
        if owner.checkpoint_reservation_state(reservation) == "active":
            owner.abandon_checkpoint_artifact(reservation)
        raise
    return record


def _iter_shared_states(shared_adjoint, shared_pass):
    """Yield ``(entry name, state)`` in manifest order."""
    for (probe_index, batch_index) in sorted(shared_adjoint):
        yield (f"shared-adjoint-{probe_index}-{batch_index}",
               shared_adjoint[(probe_index, batch_index)])
    for batch_index in sorted(shared_pass):
        yield f"shared-pass-{batch_index}", shared_pass[batch_index]


def _shared_state_by_name(shared_adjoint, shared_pass, name: str):
    """Recover one shared state by the entry name the writer assigned."""
    for candidate, state in _iter_shared_states(shared_adjoint, shared_pass):
        if candidate == name:
            return state
    raise RuntimeError(
        f"exact boundary checkpoint has no shared state {name!r}")


def _read_shared_state_payload(path: Path, entry: dict) -> bytes:
    """One shared-state pickle payload, staged-pinned under policy.

    Under the active allowed-tier policy the bytes come from a
    lifetime-pinned window (RAM leg refused fast for want of RAM-mover
    covers, SSD re-acquired honestly): exact length plus one probe byte
    through the held descriptor, the SDK serving record registered at the
    successful actual open, the descriptor closed and the exact ref
    released before the caller deserializes. Unmapped/unfenced refuses
    before a pool bulk byte. Inactive policy reads the declared path
    exactly as before. No cache, no re-read: the owned buffer is hashed
    once by the caller.
    """
    from .staged_tier_policy import policy_is_active
    if not policy_is_active():
        return Path(path).read_bytes()
    from .residency_map import residency_resolver
    from .staged_lease import LeaseRefused, acquire_entry_window
    size = entry.get("file_bytes")
    if type(size) is not int or isinstance(size, bool) or size <= 0:
        raise RuntimeError(
            "adjoint checkpoint shared-state entry has no byte size: "
            f"{entry.get('name')}")
    sha = entry.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise RuntimeError(
            "adjoint checkpoint shared-state entry has no digest: "
            f"{entry.get('name')}")
    resolver = residency_resolver()
    if resolver is None:
        raise LeaseRefused("readset-not-staged", kind="availability")
    staged = resolver.staged_read(path, expected_sha256=sha)
    if staged is None:
        raise LeaseRefused("staged-not-serving", kind="availability")
    window, key = acquire_entry_window(resolver, path, staged)
    with window:
        try:
            fd, serving = window.open(key)
        except LeaseRefused as refusal:
            resolver.record_fallback(path, str(refusal))
            raise
        tier = window.serving_tier or "stage"
        resolver.record_serving_tier(
            path, tier, pin_id=str(serving.get("pin_id") or ""),
            range_ref=str(serving.get("range_ref") or ""))
        # Sealed bounds before allocation: the manifest size must equal
        # the staged entry's, and the held descriptor's size must match
        # both — no unbounded allocation, no read-then-check.
        if size != staged["bytes"]:
            raise LeaseRefused("shared-state-size-divergent",
                               kind="integrity")
        first = os.fstat(fd)
        if first.st_size != size:
            raise LeaseRefused("shared-state-changed-under-pin",
                               kind="integrity")
        # One owned buffer, filled in place: no parts list, no joined
        # copy, no second pass. The caller's digest hashes these bytes.
        raw = bytearray(size)
        view = memoryview(raw)
        remaining = size
        offset = 0
        while remaining > 0:
            try:
                moved = os.preadv(fd, [view[offset:offset + remaining]], offset)
            except OSError as exc:
                raise LeaseRefused(
                    f"shared-state-unreadable: {exc.strerror}",
                    kind="availability") from None
            if moved <= 0:
                break
            offset += moved
            remaining -= moved
        view.release()
        if remaining:
            raise LeaseRefused("shared-state-truncated", kind="integrity")
        if os.pread(fd, 1, size):
            raise LeaseRefused("shared-state-grew-during-read",
                               kind="integrity")
        last = os.fstat(fd)
        if (last.st_ino, last.st_size, last.st_mtime_ns) != (
                first.st_ino, first.st_size, first.st_mtime_ns):
            raise LeaseRefused("shared-state-changed-under-pin",
                               kind="integrity")
    if tier == "ram":
        resolver.record_ram_read(path, len(raw))
    else:
        resolver.record_stage_read(path, len(raw))
    return raw


def _await_checkpoint_entry(entry, *, deadline):
    """Wait within the entered checkpoint phase; the exact reader keeps its pin checks."""
    from .staged_tier_policy import policy_is_active
    if not policy_is_active():
        return
    from .residency_map import residency_resolver
    from .residency_shard_reader import await_staged_spans
    from .staged_lease import stage_cover_is_published
    resolver = residency_resolver()
    if resolver is None:
        return  # The strict reader supplies its existing missing-context refusal.
    size = entry["file_bytes"]
    return await_staged_spans(
        resolver, [(entry["path"], 0, size, size)], deadline=deadline,
        published=stage_cover_is_published)


def load_adjoint_checkpoint(
    space: str | os.PathLike, record: dict, *, cotangent_factory=None,
    shared_state_max_bytes=None,
) -> tuple[dict, dict, dict]:
    """Read one checkpoint back, verifying every digest it claims.

    Returns ``(cotangents, shared_adjoint, shared_pass)`` with CPU tensors and
    deserialized state. A caller-owned cotangent factory may provide bounded
    working storage; verified entry windows are released before the next load.
    Refuses on any digest or shape mismatch: a checkpoint
    whose bytes moved is a new identity, never a silent partial read.
    """
    checkpoint_dir = checkpoint_directory(space, int(record["boundary"]))
    manifest_path = checkpoint_dir / "checkpoint.json"
    try:
        stored = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"adjoint checkpoint manifest unreadable at {manifest_path}") from exc
    if stored.get("cotangent_sha256") != record.get("cotangent_sha256"):
        raise RuntimeError(
            "adjoint checkpoint identity differs from its receipt entry "
            f"(boundary {record.get('boundary')})")
    # The receipt's entry lists are the trust anchor: a record whose entries
    # differ from the manifest on disk -- by name, digest or size -- refuses
    # whole rather than reading whichever side happens to be present.
    for field in ("activation_entries", "shared_state_entries"):
        if stored.get(field) != record.get(field):
            raise RuntimeError(
                f"adjoint checkpoint {field} differ from its receipt entry "
                f"(boundary {record.get('boundary')})")
    if shared_state_max_bytes is not None:
        if type(shared_state_max_bytes) is not int or shared_state_max_bytes <= 0:
            raise ValueError("adjoint shared-state ceiling must be positive")
        sizes = [entry.get("file_bytes") for entry in stored["shared_state_entries"]]
        if (any(type(size) is not int or size <= 0 for size in sizes)
                or sum(sizes) > shared_state_max_bytes):
            raise RuntimeError("adjoint shared-state payloads exceed auxiliary byte ceiling")
    from .residency_shard_reader import staged_range_wait_s
    deadline = time.monotonic() + staged_range_wait_s()
    session = stored["session"]
    entries = stored["activation_entries"]
    cotangents = {} if cotangent_factory is None else cotangent_factory(entries)
    for entry in entries:
        _await_checkpoint_entry(entry, deadline=deadline)
        tensors = read_exact_entry_tensors([entry], expected_session=session)
        probe, batch = (int(part) for part in
                        entry["name"].removeprefix("cotangent-").split("-"))
        cotangents[(probe, batch)] = tensors.pop(entry["name"])
        del tensors
    shared_adjoint, shared_pass = {}, {}
    shared_tensor_bytes = 0
    for entry in stored["shared_state_entries"]:
        _await_checkpoint_entry(entry, deadline=deadline)
        path = Path(entry["path"])
        payload = _read_shared_state_payload(path, entry)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry["sha256"] or len(payload) != entry["file_bytes"]:
            raise RuntimeError(
                f"adjoint checkpoint shared-state entry changed: {entry['name']}")
        state = pickle.loads(payload)
        del payload
        parts = entry["name"].split("-")
        if entry["name"].startswith("shared-adjoint-"):
            shared_adjoint[(int(parts[2]), int(parts[3]))] = state
        else:
            shared_pass[int(parts[2])] = state
        if shared_state_max_bytes is not None:
            from .cost_streaming import _state_storage_bytes
            shared_tensor_bytes += _state_storage_bytes(state)
            if shared_tensor_bytes > shared_state_max_bytes:
                raise RuntimeError("adjoint shared-state tensors exceed auxiliary byte ceiling")
    return cotangents, shared_adjoint, shared_pass


# --------------------------------------------------------------------------
# Stage-A receipt
# --------------------------------------------------------------------------


def write_adjoint_receipt(space: str | os.PathLike, receipt: dict) -> bool:
    """Seal the receipt atomically, first writer wins (§3.3)."""
    payload = (json.dumps(
        receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    created = publish_new_bytes(adjoint_receipt_path(space), payload)
    if not created:
        raise RuntimeError(
            "adjoint-capture.json already exists; a completed adjoint capture "
            "is never overwritten -- publish a new output root or repair the "
            "existing receipt's inputs")
    return True


def load_adjoint_receipt(path: str | os.PathLike, sha256: str) -> dict:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != str(sha256):
        raise RuntimeError(
            f"adjoint receipt digest mismatch at {path}: expected {sha256}, "
            f"found {digest}")
    receipt = json.loads(raw)
    if receipt.get("schema") != ADJOINT_RECEIPT_SCHEMA:
        raise RuntimeError(
            f"adjoint receipt schema mismatch: {receipt.get('schema')!r}")
    return receipt


# --------------------------------------------------------------------------
# The render-free reverse step -- the single run's completed-layer leg
# --------------------------------------------------------------------------


def render_free_layer_roll(
    runner, *, storage, batches, layer, cotangents, n_probes,
    incoming_entries, incoming_tensor, roll, min_free_gib: float = 0.0,
    then=None, batch_size: int = 1, probe_fusion: bool = False,
) -> int:
    """Roll the cotangent through one layer with no renders and no projection.

    This is ``compute_aura_cost_streamed``'s completed-layer leg -- the
    ``replay_backward(final=True, lease=None)`` body every layer whose units
    are already journalled runs in the single consumer -- expressed once so
    stage A's chain and a quantum's chain are the same kernels in the same
    order by construction (§6.2 step 3, §9.3's bitwise gate depends on it).

    Order: probe ascending, batch ascending inside the storage policy's
    ``prefetch_batches`` windows. Per (probe, batch): RNG fence, incoming
    cotangent to device, exact input boundary to device with grad, isolated
    pass state grafted onto the shared-state cotangent owner, isolated layer
    forward, one ``torch.autograd.backward([out, *roots], ...)`` with the
    produced roots, harvest, RNG fence, input-cotangent check, ``roll``.

    The chain regime (RobTand/prismaquant#997) changes the grouping, never
    the math per sample:

    * ``batch_size`` B > 1 carries B consecutive batches through one layer
      forward and one backward, and splits the input cotangent back per
      batch for ``roll``. The GEMMs are B times taller, so rounding differs
      from B = 1: statistically equivalent, not bitwise. B must divide the
      read window, so a group never spans two windows.
    * ``probe_fusion`` runs one forward per batch group and then one backward
      per probe on the retained graph, in sample-major windows that read
      every probe's incoming entry beside the boundary
      (``prefetched_fused_boundary_windows``). At a fixed B it is bitwise
      equal to the probe-major order: the same forward, the same backward
      kernels on the same bytes. ``roll`` is called sample-major instead of
      probe-major, with the same tensors.

    Both require an empty per-sample pass state: no profile shared state for
    the layer and no shared-state cotangent in any owner. A profile with
    shared forward state (Gemma4's KV sharing) grafts and harvests per
    (probe, batch), and that cannot be merged; such a run refuses
    (``ChainRegimeRefused``) instead of rolling a wrong cotangent.

    ``incoming_entries[probe]`` is the per-batch exact-entry list (the single
    run's ``grad_outs[probe]``) read in the same window as the boundary; when
    ``None`` the incoming cotangent comes from ``incoming_tensor(probe, batch)``.
    ``roll(cpu_tensor, batch_index, probe_index)`` consumes the produced
    cotangent at boundary ``layer`` (publish it, keep it, or checkpoint it).
    ``then`` is the pass the caller reads after this roll, as
    ``(boundary_index, incoming)``: ``incoming`` is probe 0's entry list
    probe-major, and the per-probe list (the shape of ``incoming_entries``)
    when fused. Its first window is staged during the roll's last window.
    Staging only. Returns the number of per-sample cotangents rolled, which
    is the number of backwards at B = 1.
    """
    regime = normalize_chain_regime(batch_size, probe_fusion)
    # Staging only, never order: every probe pass below re-reads this
    # layer's input boundary, so a storage that stages its reads through
    # PrismaBuild keeps that plane staged across the passes, and asks for
    # the next layer's plane now so its movers run during this roll
    # (RobTand/prismaquant#887). A storage without the hooks is untouched.
    # The fused roll reads each boundary group once and keeps only what
    # its next window reads again (``retain_produced_reads``).
    retain = getattr(storage, "retain_produced_boundary", None)
    stage_ahead = getattr(storage, "stage_produced_boundary_ahead", None)
    if stage_ahead is not None and int(layer) > 0:
        stage_ahead(int(layer) - 1)
    if regime["probe_fusion"]:
        return _render_free_fused_passes(
            runner, storage=storage, batches=batches, layer=layer,
            cotangents=cotangents, n_probes=n_probes,
            incoming_entries=incoming_entries,
            incoming_tensor=incoming_tensor, roll=roll,
            min_free_gib=min_free_gib, then=then,
            batch_size=regime["batch_size"])
    with (retain(int(layer)) if retain is not None else nullcontext()):
        backwards = _render_free_probe_passes(
            runner, storage=storage, batches=batches, layer=layer,
            cotangents=cotangents, n_probes=n_probes,
            incoming_entries=incoming_entries,
            incoming_tensor=incoming_tensor, roll=roll,
            min_free_gib=min_free_gib, then=then,
            batch_size=regime["batch_size"])
    return backwards


def _chain_free_floor(min_free_gib, layer, where):
    from .aura_cost import _free_gib

    if _free_gib() < min_free_gib:
        raise RuntimeError(
            f"free UMA {_free_gib():.1f} < floor {min_free_gib:.1f}; "
            f"render-free chain layer {layer} {where}")


def _chain_rng_state(device):
    return (torch.get_rng_state(),
            torch.cuda.get_rng_state(device)
            if torch.device(device).type == "cuda" else None)


def _chain_rng_fence(saved, device):
    cpu_rng, cuda_rng = saved
    if not torch.equal(cpu_rng, torch.get_rng_state()) or (
            cuda_rng is not None
            and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device))):
        raise RuntimeError("render-free chain source consumed Torch RNG")


def _stack_to_device(tensors, *, device, dtype=None):
    """One tensor moved as the B = 1 roll always moved it; several stacked first."""
    if len(tensors) == 1:
        tensor = tensors[0]
        return tensor.to(device) if dtype is None else tensor.to(device=device, dtype=dtype)
    stacked = torch.cat(tensors, dim=0)
    return stacked.to(device) if dtype is None else stacked.to(device=device, dtype=dtype)


def _require_per_sample_state(runner, batches, layer, owners, indices, *, where):
    """Refuse a grouped roll unless every sample's pass state is empty."""
    profile = runner.profile
    for index in indices:
        state = profile.isolated_layer_pass_state(
            batches[index].shared_pass_state, runner.layers[layer])
        if state:
            raise ChainRegimeRefused(
                f"{where}: layer {layer} batch {index} carries shared pass state "
                f"{sorted(state)}; a batched or fused chain needs an empty "
                "per-sample pass state")
    for owner in owners:
        if not owner.is_empty():
            raise ChainRegimeRefused(
                f"{where}: layer {layer} carries a shared-state cotangent; a "
                "batched or fused chain needs an empty per-sample pass state")


def _chain_group_batch(runner, batches, indices, cache):
    """The layer-call metadata for one batch group, built once per roll.

    The same ``_prepare`` the capture ran, on the group's stacked token ids,
    so ids, positions, rotary embeddings and masks are exactly what a
    B-row forward builds for itself; a mask built for one row is never
    broadcast to B (GLM's DSA mask is a ``[B, S]`` padding mask).
    """
    from .cost_streaming import StreamedForwardBoundaries

    key = tuple(indices)
    group = cache.get(key)
    if group is None:
        with torch.no_grad():
            ids = torch.cat([batches[index].input_ids for index in indices], dim=0)
            ids, position_ids, hidden, embeddings, mask = runner._prepare(ids)
            del hidden
        group = StreamedForwardBoundaries(ids, position_ids, embeddings, mask, [], None)
        cache[key] = group
    return group


def _roll_rows(roll, gradient, indices, probe_index):
    """Hand each sample's input cotangent to ``roll``, batch ascending."""
    cpu = gradient.detach().to("cpu")
    if len(indices) == 1:
        roll(cpu, indices[0], probe_index)
        return
    for row, index in enumerate(indices):
        roll(cpu[row:row + 1], index, probe_index)


def _render_free_probe_passes(
    runner, *, storage, batches, layer, cotangents, n_probes,
    incoming_entries, incoming_tensor, roll, min_free_gib, then=None,
    batch_size=1,
) -> int:
    """The probe passes of :func:`render_free_layer_roll`, probe-major.

    At ``batch_size`` 1 this is the original loop, call for call.
    """
    from .cost_streaming import prefetched_boundary_batches

    batch_size = int(batch_size)
    if storage is not None and int(storage.config["prefetch_batches"]) % batch_size:
        raise ChainRegimeRefused(
            f"chain batch size {batch_size} does not divide the sealed read "
            f"window of {storage.config['prefetch_batches']} batches")
    groups = {}
    backwards = 0
    last = len(batches) - 1
    for probe_index in range(int(n_probes)):
        entries = (None if incoming_entries is None
                   else incoming_entries[probe_index])
        # Staging only: the next probe pass's first window is asked for
        # during this pass's last one (RobTand/prismaquant#989).
        following = then
        if probe_index + 1 < int(n_probes):
            following = (int(layer), None if incoming_entries is None
                         else incoming_entries[probe_index + 1])
        with prefetched_boundary_batches(
                storage, batches, int(layer), incoming=entries,
                then=following) as windows:
            pending = []
            for item in windows:
                if batch_size == 1:
                    backwards += _roll_one(
                        runner, item, layer=layer, probe_index=probe_index,
                        cotangents=cotangents, entries=entries,
                        incoming_tensor=incoming_tensor, roll=roll,
                        min_free_gib=min_free_gib)
                    # The window's tensors are dropped here, as the
                    # pre-#997 loop dropped them in its ``finally``.
                    item = None
                    continue
                pending.append(item)
                if len(pending) < batch_size and item[0] != last:
                    continue
                group, pending = pending, []
                try:
                    backwards += _roll_group(
                        runner, group, batches=batches, layer=layer,
                        probe_index=probe_index, cotangents=cotangents,
                        entries=entries, incoming_tensor=incoming_tensor,
                        roll=roll, min_free_gib=min_free_gib, cache=groups)
                finally:
                    group = None
    return backwards


def _roll_one(runner, item, *, layer, probe_index, cotangents, entries,
              incoming_tensor, roll, min_free_gib) -> int:
    """One (probe, batch) backward: the roll as it ran before #997."""
    profile = runner.profile
    device, dtype = runner.device, runner.dtype
    batch_index, batch, boundary_cpu, incoming_cpu = item
    owner = cotangents[probe_index][batch_index]
    try:
        _chain_free_floor(min_free_gib, layer, f"probe {probe_index}")
        saved = _chain_rng_state(device)
        if entries is None:
            incoming_cpu = incoming_tensor(probe_index, batch_index)
        incoming_grad = incoming_cpu.to(device)
        x_in = boundary_cpu.to(
            device=device, dtype=dtype).detach().requires_grad_(True)
        isolated = profile.isolated_layer_pass_state(
            batch.shared_pass_state, runner.layers[layer])
        isolated = owner.graft(isolated)
        out = runner.isolated_layer(batch, layer, x_in, pass_state=isolated)
        roots, root_grads = owner.produced_roots()
        torch.autograd.backward([out, *roots], [incoming_grad, *root_grads])
        owner.harvest()
        _chain_rng_fence(saved, device)
        if x_in.grad is None:
            raise RuntimeError(
                f"render-free chain layer {layer} produced no input cotangent")
        roll(x_in.grad.detach().to("cpu"), batch_index, probe_index)
        return 1
    finally:
        item = boundary_cpu = incoming_cpu = None
        out = x_in = incoming_grad = isolated = roots = root_grads = None


def _roll_group(runner, group, *, batches, layer, probe_index, cotangents,
                entries, incoming_tensor, roll, min_free_gib, cache) -> int:
    """One probe's backward for a batch group (B > 1), split back per batch."""
    if len(group) == 1:
        return _roll_one(
            runner, group[0], layer=layer, probe_index=probe_index,
            cotangents=cotangents, entries=entries,
            incoming_tensor=incoming_tensor, roll=roll, min_free_gib=min_free_gib)
    device, dtype = runner.device, runner.dtype
    indices = [item[0] for item in group]
    _require_per_sample_state(
        runner, batches, layer, [cotangents[probe_index][index] for index in indices],
        indices, where="batched render-free chain")
    try:
        _chain_free_floor(min_free_gib, layer, f"probe {probe_index}")
        saved = _chain_rng_state(device)
        incoming = [incoming_tensor(probe_index, index) if entries is None else item[3]
                    for index, item in zip(indices, group)]
        incoming_grad = _stack_to_device(incoming, device=device)
        incoming = None
        x_in = _stack_to_device([item[2] for item in group], device=device,
                                dtype=dtype).detach().requires_grad_(True)
        batch = _chain_group_batch(runner, batches, indices, cache)
        out = runner.isolated_layer(batch, layer, x_in, pass_state={})
        torch.autograd.backward([out], [incoming_grad])
        _chain_rng_fence(saved, device)
        if x_in.grad is None:
            raise RuntimeError(
                f"render-free chain layer {layer} produced no input cotangent")
        _roll_rows(roll, x_in.grad, indices, probe_index)
        return len(indices)
    finally:
        group = incoming = None
        out = x_in = incoming_grad = None


def _render_free_fused_passes(
    runner, *, storage, batches, layer, cotangents, n_probes,
    incoming_entries, incoming_tensor, roll, min_free_gib, then=None,
    batch_size=1,
) -> int:
    """The fused roll: one forward per batch group, one backward per probe.

    Sample-major (``prefetched_fused_boundary_windows``). Every probe's
    backward runs on the one retained graph, the last one frees it, and
    ``x_in.grad`` is cleared before each so every probe's input cotangent is
    its own, never a sum.
    """
    from .cost_streaming import (
        fused_window_batches,
        prefetched_fused_boundary_windows,
    )

    device, dtype = runner.device, runner.dtype
    n_probes = int(n_probes)
    batch_size = int(batch_size)
    window = fused_window_batches(
        storage, batches, int(layer), incoming_entries, batch_size=batch_size)
    groups = {}
    backwards = 0
    with prefetched_fused_boundary_windows(
            storage, batches, int(layer), incoming=incoming_entries,
            window_batches=window, then=then) as windows:
        for indices, boundary_of, incoming_of in windows:
            for start in range(0, len(indices), batch_size):
                members = list(indices[start:start + batch_size])
                owners = [cotangents[probe][index]
                          for probe in range(n_probes) for index in members]
                _require_per_sample_state(runner, batches, layer, owners, members,
                                          where="fused render-free chain")
                boundary = x_in = out = incoming_grad = None
                try:
                    _chain_free_floor(min_free_gib, layer, "fused probes")
                    saved = _chain_rng_state(device)
                    boundary = [boundary_of(index) for index in members]
                    x_in = _stack_to_device(boundary, device=device,
                                            dtype=dtype).detach().requires_grad_(True)
                    boundary = None
                    batch = (batches[members[0]] if len(members) == 1
                             else _chain_group_batch(runner, batches, members, groups))
                    out = runner.isolated_layer(batch, layer, x_in, pass_state={})
                    for probe_index in range(n_probes):
                        incoming = [incoming_tensor(probe_index, index)
                                    if incoming_entries is None
                                    else incoming_of(probe_index, index)
                                    for index in members]
                        incoming_grad = _stack_to_device(incoming, device=device)
                        incoming = None
                        x_in.grad = None
                        torch.autograd.backward(
                            [out], [incoming_grad],
                            retain_graph=probe_index + 1 < n_probes)
                        if x_in.grad is None:
                            raise RuntimeError(
                                f"render-free chain layer {layer} produced no "
                                "input cotangent")
                        _roll_rows(roll, x_in.grad, members, probe_index)
                        incoming_grad = None
                        backwards += len(members)
                    _chain_rng_fence(saved, device)
                finally:
                    boundary = incoming = None
                    out = x_in = incoming_grad = None
    return backwards


# --------------------------------------------------------------------------
# Dev-mode stamping (the interim submission lane, contract §5.2)
# --------------------------------------------------------------------------


def dev_mode_enabled(environ=None) -> bool:
    return dict(environ if environ is not None else os.environ).get(
        "PRISMAQUANT_DEV_MODE") == "1"


def require_dev_mode(where: str) -> None:
    """The distributed campaign's submission lane is dev mode (§10 defers the
    certified N-consumer grammar); anything else refuses rather than looking
    certified."""
    if not dev_mode_enabled():
        raise RuntimeError(
            f"{where} requires PRISMAQUANT_DEV_MODE=1: the certified "
            "distributed-submission grammar is deliberately deferred "
            "(distributed_campaign_2026-09-19.md §10)")


def dev_mode_stamp(environ=None) -> dict:
    """The ``dev_uncertified`` provenance stamp PR #776's lane carries."""
    stamp = {
        "dev_uncertified": True,
        "prismaquant_dev_mode": 1,
    }
    try:
        import subprocess
        root = Path(__file__).resolve().parents[1]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, timeout=10, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True,
            text=True, timeout=10, check=True).stdout.strip())
        stamp["tree"] = {"commit": commit, "dirty": dirty}
    except Exception as exc:  # noqa: BLE001 - the stamp is advisory provenance
        stamp["tree"] = {"error": f"{type(exc).__name__}: {exc}"}
    return stamp


# --------------------------------------------------------------------------
# Telemetry: GPU energy sampler (§8.1) -- power against the GB10 envelope,
# never utilization percentages.
# --------------------------------------------------------------------------


class GpuPowerSampler:
    """1 Hz ``nvidia-smi --query-gpu=power.draw`` sampling in-process.

    ``nvidia_smi.gpu_utilization`` is non-diagnostic on GB10 (AGENTS.md
    principle 13), so the counters carry joules, watts and the kernel-active
    ratio instead. A missing or failing sampler is recorded, never silent and
    never zero.
    """

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = float(interval_s)
        self.samples: list[float] = []
        self.error: str | None = None
        self._process = None
        self._thread = None
        self._stopping = False

    def start(self) -> "GpuPowerSampler":
        import subprocess
        import threading

        try:
            self._process = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits", "-l", str(int(self.interval_s))],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except (OSError, ValueError) as exc:
            self.error = f"sampler launch failed: {exc}"
            return self

        def sample():
            try:
                for line in self._process.stdout:
                    if self._stopping:
                        return
                    value = line.strip().split(",")[0].strip()
                    try:
                        self.samples.append(float(value))
                    except ValueError:
                        continue
            except (OSError, ValueError) as exc:
                if not self._stopping:
                    self.error = f"sampler read failed: {exc}"

        self._thread = threading.Thread(target=sample, daemon=True, name="gpu-power")
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stopping = True
        try:
            if self._process is not None:
                self._process.terminate()
                self._process.wait(timeout=5)
        except Exception:  # noqa: BLE001 - teardown best effort, sample list stands
            pass
        if self._thread is not None:
            self._thread.join(timeout=2)
        watts = sorted(self.samples)
        if watts:
            joules = sum(watts) * self.interval_s
            p95 = watts[max(0, int(0.95 * len(watts)) - 1)]
            block = {
                "sample_count": len(watts),
                "interval_s": self.interval_s,
                "gpu_joules": joules,
                "gpu_power_w_p50": watts[len(watts) // 2],
                "gpu_power_w_p95": p95,
                "gpu_power_w_max": watts[-1],
            }
        else:
            block = {
                "sample_count": 0,
                "interval_s": self.interval_s,
                "gpu_joules": None,
                "gpu_power_w_p50": None,
                "gpu_power_w_p95": None,
                "gpu_power_w_max": None,
            }
        if self.error:
            block["sampler_error"] = self.error
        return block


class KernelTimeProfiler:
    """CUDA kernel-time accumulation via ``torch.profiler`` (§8.1).

    ``kernel_active_s`` is the profiler's device-time sum. Enabled around the
    phases that do GPU work (chain layers, window replays); a backend that
    cannot profile records the failure instead of a zero.

    **Scope it to bounded work.** Kineto keeps every CUDA activity record in
    host memory until the session stops, and the stop then builds the whole
    trace before ``key_averages`` can sum it. Around one chain layer or one
    window replay that is small. Around a whole 512-sample Stage A capture it
    was 14 GB an hour while collecting and another 22 GB in the two minutes
    after the stop, on a box whose host and GPU share one pool (PQ #899). A
    caller that cannot bound its scope passes ``not_measured`` with the reason:
    no session is opened, and the block reports ``None`` and why.
    """

    def __init__(self, *, not_measured: str | None = None):
        self.kernel_active_s = 0.0
        self.error: str | None = not_measured
        self._profile = None
        self._measure = not_measured is None

    def __enter__(self):
        if not self._measure:
            return self
        try:
            self._profile = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA])
            self._profile.__enter__()
        except Exception as exc:  # noqa: BLE001 - profiling is telemetry
            self._profile = None
            self.error = f"torch.profiler CUDA unavailable: {exc}"
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._profile is None:
            return False
        try:
            self._profile.__exit__(exc_type, exc, traceback)
            for event in self._profile.key_averages():
                self.kernel_active_s += float(event.self_device_time_total) / 1e6
        except Exception as exc:  # noqa: BLE001
            self.error = f"kernel-time summary failed: {exc}"
        self._profile = None
        return False

    def block(self) -> dict:
        block = {"kernel_active_s": (self.kernel_active_s if not self.error
                                     else None)}
        if self.error:
            block["profiler_error"] = self.error
        return block


def wall_clock_seconds(started: float) -> float:
    return time.time() - started


__all__ = [
    "ADJOINT_CAPTURE_ENTRY_POINT", "ADJOINT_CHECKPOINT_SCHEMA",
    "ADJOINT_RECEIPT_SCHEMA", "DEFAULT_STRIDE", "GpuPowerSampler",
    "KernelTimeProfiler", "QUANTUM_COUNTERS_SCHEMA", "QUANTUM_RECORD_SCHEMA",
    "QUANTUM_STATUS_SCHEMA", "adjoint_space", "adjoint_receipt_path",
    "boundary_entry_directory", "chain_layers_for", "checkpoint_directory",
    "derive_checkpoint_boundaries", "dev_mode_enabled", "dev_mode_stamp",
    "exact_entry_record", "load_adjoint_checkpoint", "load_adjoint_receipt",
    "nearest_checkpoint_boundary", "read_exact_entry_tensors",
    "reference_from_record", "render_free_layer_roll", "require_dev_mode",
    "wall_clock_seconds", "write_adjoint_checkpoint", "write_adjoint_receipt",
    "write_checkpoint_cotangent_entry",
]
