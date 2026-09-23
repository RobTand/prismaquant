"""Stage B preparation as a PrismaBuild action: its reads and its outputs (PQ #1070).

``tools/prepare_extended_joint_quanta.py`` and the generator it runs,
``tools/regenerate_joint_quanta.py``, read the gzip parent manifest, the
production pickle, the head walk's inputs and the source checkpoint's
safetensors headers, and write the Stage B metadata generation: records,
slices, manifests and the launch recipe. Before #1070 none of those reads
was declared and every write went straight to the pool.

This module gives both halves a PrismaBuild shape:

* **Reads.** :func:`preparation_read_entries` lists what the preparation
  reads as data-manifest entries, and :func:`preparation_read_manifest`
  seals them. ``pbrun --data-manifest`` attaches the manifest, so the reads
  are declared before the action is claimed. The reads themselves stay plain
  file reads; nothing here routes them through the staged reader.
* **Writes.** :func:`build_preparation_template` is the write-only
  produced-output template (#912) over the metadata root, submitted with
  ``pbrun --produced-output-template``. With the tools' ``--produced-output``
  flag, inside the admitted action, :func:`publish_files` files one prewrite
  for the files a group creates, writes them first-writer as before, and
  commits them at their origin (``produced_output.commit_origin_batch``).
  Without the flag it writes exactly as before.

What stays as it was: every output keeps its bytes and its path, a file
already present with the same bytes is left alone, and one with other bytes
still refuses. A file this action did not create belongs to the action that
did, so only created files enter this action's batches.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

#: The template slot every Stage B preparation output is filed under.
PREPARATION_SLOT = "stage_b_metadata"
#: The template id prefix; the id's suffix digests the template body, so a
#: changed root, tier or ceiling files a template of its own (PQ #1054).
PREPARATION_TEMPLATE_PREFIX = "pq-stage-b-preparation"
#: PrismaBuild's produced-output template schema (``produced_output.
#: TEMPLATE_SCHEMA_V1``), restated for the submitter, which runs outside PB.
PRODUCED_OUTPUT_TEMPLATE_SCHEMA = "prismaquant.prismabuild.produced_output_template.v1"
#: The read phase the preparation's manifest declares.
PREPARATION_READ_PHASE = "stage-b-preparation"
#: PrismaBuild's own per-manifest bound (``prismabuild.core
#: DATA_MANIFEST_MAX_BYTES``), restated as ``experiments/glm_data_manifests``
#: restates it: this module runs where PrismaBuild is not importable.
DATA_MANIFEST_MAX_BYTES = 64 * 1024 * 1024
#: Files the preparation writes besides one set per quantum: the catalog
#: extension, the parent, the partition, the derivation input, the spec, the
#: launch recipe, the generator's derivation and its records index.
PREPARATION_CONTROL_FILES = 8
#: Files the generator writes per quantum: its record, its slice manifest,
#: its Stage A slice, its executable readset and its head slice.
PREPARATION_FILES_PER_QUANTUM = 5


class PreparationPublicationRefused(RuntimeError):
    """The preparation's produced output could not be bound, filed or committed."""


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------

def head_phase_entries(parent: Mapping) -> list[dict]:
    """The parent manifest's ``head`` phase entries, in order.

    The head is everything read before the first layer installs: the head
    walk's inputs and the control files the extended parent adds
    (``prepare_extended_joint_quanta.extend_parent``). The Stage B
    preparation runs that walk once (PQ #1010), so its reads are the head.
    """
    annotations = parent.get("annotations") or {}
    phases = annotations.get("phases") or []
    if not phases or phases[0].get("name") != "head":
        raise ValueError("the parent manifest has no leading head phase")
    end = int(phases[0]["bytes"])
    entries, offset = [], 0
    for entry in parent["entries"]:
        if offset >= end:
            break
        entries.append({"path": str(entry["path"]), "offset": int(entry["offset"]),
                        "bytes": int(entry["bytes"]), "sha256": entry.get("sha256")})
        offset += int(entry["bytes"])
    if offset != end:
        raise ValueError("the parent's head phase does not end at an entry boundary")
    return entries


def file_entry(path: str | os.PathLike) -> dict:
    """A whole-file entry for a present file."""
    path = os.path.normpath(os.path.abspath(os.fspath(path)))
    size = os.path.getsize(path)
    if size <= 0:
        raise ValueError(f"a declared read is an empty file: {path}")
    return {"path": path, "offset": 0, "bytes": int(size), "sha256": None}


def preparation_read_entries(*, head: Sequence[Mapping], files: Iterable,
                             ranges: Iterable[tuple[str, int, int]] = ()) -> list[dict]:
    """The preparation's reads as data-manifest entries, one per ``(path, offset)``.

    ``head`` is the parent's head phase, ``files`` the whole files the tools
    read besides it (their own arguments, the Stage A proofs, the production
    pickle), and ``ranges`` the partial reads (the safetensors index and
    headers, :func:`joint_layer_quanta.layer_source_header_reads`). A range
    that a whole-file entry at the same offset already covers is dropped;
    PrismaBuild refuses a repeated ``(path, offset)``.
    """
    entries: list[dict] = []
    index: dict[tuple[str, int], int] = {}

    def add(entry):
        key = (entry["path"], entry["offset"])
        if key in index:
            held = entries[index[key]]
            if entry["bytes"] > held["bytes"]:
                held["bytes"] = entry["bytes"]
            return
        index[key] = len(entries)
        entries.append(dict(entry))

    for entry in head:
        add({"path": os.path.normpath(str(entry["path"])), "offset": int(entry["offset"]),
             "bytes": int(entry["bytes"]), "sha256": entry.get("sha256")})
    for path in files:
        add(file_entry(path))
    for path, offset, size in ranges:
        add({"path": os.path.normpath(str(path)), "offset": int(offset),
             "bytes": int(size), "sha256": None})
    return entries


def preparation_read_manifest(entries: Sequence[Mapping], *, produced_by: Mapping,
                              annotations: Mapping,
                              mount_prefix: str = "/mnt/shared") -> dict:
    """The sealed-shape v2 data manifest for the preparation's reads.

    One read phase (:data:`PREPARATION_READ_PHASE`) holds every entry: the
    preparation reports no per-phase progress, and a later phase would only
    reorder bytes the action reads before it writes anything.
    """
    from .joint_layer_quanta import MANIFEST_SCHEMA_V2

    prefix = os.path.normpath(str(mount_prefix))
    rows = [dict(entry) for entry in entries]
    if not rows:
        raise ValueError("the preparation declares no read")
    for row in rows:
        if not row["path"].startswith(prefix + "/"):
            raise ValueError(f"a preparation read is outside {prefix}: {row['path']}")
    total = sum(int(row["bytes"]) for row in rows)
    return {"schema": MANIFEST_SCHEMA_V2,
            "produced_by": dict(produced_by),
            "mount_prefix": prefix,
            "entries": rows, "entry_count": len(rows), "total_bytes": total,
            "annotations": dict(annotations),
            "read_plan": {"phases": [{"name": PREPARATION_READ_PHASE,
                                      "entry_indices": list(range(len(rows))),
                                      "bytes": total, "cumulative_bytes": total}],
                          "read_bytes": total}}


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------

def preparation_payload_ceiling(quanta: int, *, indexes: int = 1) -> int:
    """The template's durable payload maximum for ``quanta`` quanta.

    Every file the preparation writes is at most PrismaBuild's own manifest
    bound: the manifests it seals must fit it to be submitted at all, and a
    record, slice or control document is smaller than the manifest that
    names it. The file count is the generator's per-quantum set plus the
    control files and one index per band (``indexes``). This caps the
    template; each batch files its exact bytes at its prewrite.
    """
    if type(quanta) is not int or quanta <= 0:
        raise ValueError("the preparation writes at least one quantum")
    if type(indexes) is not int or indexes <= 0:
        raise ValueError("the preparation writes at least one records index")
    files = quanta * PREPARATION_FILES_PER_QUANTUM + PREPARATION_CONTROL_FILES + indexes - 1
    return files * DATA_MANIFEST_MAX_BYTES


def build_preparation_template(*, metadata_root: str | os.PathLike, tier: str,
                               payload_max_bytes: int) -> dict:
    """The write-only produced-output template over ``metadata_root`` (#912).

    Write-only: the preparation never reads its outputs back through a stage,
    so every tier it names carries a zero window, and its batches commit at
    their origin. ``tier`` is the stage tier the declaration permits, a fleet
    fact the submitter names. The temp maximum equals the payload maximum:
    each file passes through a staging file of its own size before the link
    (``cost_stage_checkpoint.publish_new_bytes``).
    """
    root = os.path.normpath(os.path.abspath(os.fspath(metadata_root)))
    if type(payload_max_bytes) is not int or payload_max_bytes <= 0:
        raise ValueError("the preparation template needs a positive payload maximum")
    body = {"schema": PRODUCED_OUTPUT_TEMPLATE_SCHEMA, "version": 1,
            "output_prefix": root,
            "slots": {PREPARATION_SLOT: {"class": "payload"}},
            "durable_maxima": {"payload_max_bytes": payload_max_bytes,
                               "checkpoint_max_bytes": 0,
                               "temp_max_bytes": payload_max_bytes},
            "working_demands": {str(tier): {"minimum_gib": 0, "window_gib": 0}},
            "permitted_tiers": [str(tier)], "write_only": True}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    return {**body, "template_id": f"{PREPARATION_TEMPLATE_PREFIX}-{digest}"}


class PreparationPublication:
    """This admitted action's produced output for the preparation's files."""

    def __init__(self, publication):
        self.publication = publication
        self._po = publication._po
        if not callable(getattr(self._po, "commit_origin_batch", None)):
            raise PreparationPublicationRefused(
                "the sealed PrismaBuild produced_output API has no "
                "commit_origin_batch: this runtime cannot commit write-only "
                "outputs (#912)")
        if not self.publication.template.get("write_only"):
            raise PreparationPublicationRefused(
                "the declared produced-output template is not write-only: the "
                "preparation commits its outputs at their origin (#912)")
        self.batches: list[dict] = []
        self._kinds: set[str] = set()

    @property
    def output_prefix(self) -> str:
        return self.publication.output_prefix

    def batch_id(self, kind: str) -> str:
        if not kind or "/" in kind:
            raise ValueError("a preparation batch kind is a bare name")
        return f"stage-b-prep-{kind}-{self.publication.generation}"

    def claim_kind(self, kind: str) -> None:
        """Reserve ``kind`` for one batch in this action.

        The batch id is the kind plus the owner's generation, so a retry of
        the same action re-derives it; a second group of the same kind in one
        action would reuse it, and PrismaBuild refuses a reused batch id with
        other entries. Refused here, before any prewrite, with the kind named.
        """
        if kind in self._kinds:
            raise PreparationPublicationRefused(
                f"the preparation publishes its {kind!r} group once per action")
        self._kinds.add(kind)

    def commit(self, kind: str, created: Sequence[tuple[str, bytes]]) -> dict:
        descriptors = [self._po.validate_descriptor({
            "schema": self._po.DESCRIPTOR_SCHEMA_V2, "slot": PREPARATION_SLOT,
            "artifact_class": "payload", "path": path, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "producer_generation": self.batch_id(kind),
            "owner_action_key": self.publication.instance["owner_action_key"],
            "owner_attempt": dict(self.publication.instance["owner_attempt"]),
        }, self.publication.template, self.publication.instance)
            for path, payload in created]
        out = dict(self._po.commit_origin_batch(
            self.publication.queue, self.publication.instance,
            self.publication.template, descriptors, batch_id=self.batch_id(kind)))
        if not out.get("ok"):
            raise PreparationPublicationRefused(
                f"the preparation's {kind!r} batch did not commit: {out}")
        self.batches.append({"kind": kind, "batch_id": self.batch_id(kind),
                             "files": len(descriptors),
                             "bytes": sum(len(payload) for _, payload in created),
                             "ref": out.get("ref")})
        return out


_BOUND: dict[str, PreparationPublication] = {}


def bind_preparation_publication(metadata_root: str | os.PathLike | None, *,
                                 required: bool,
                                 env: Mapping[str, str] | None = None):
    """This action's preparation publication, or None when not asked for.

    ``required`` is the tools' ``--produced-output`` flag, which the
    submission's command line carries. Without it the preparation writes as
    before, wherever it runs: a test shard is a PrismaBuild action too, and
    its launch context names no produced-output template. With it the
    process must be an admitted action (``PRISMABUILD_ACTION_KEY``), its
    declared template must be the write-only preparation template, and the
    template's output prefix must be ``metadata_root``; anything else
    refuses before the first write. Bound once per process: the preparation
    and the generator it runs share it.
    """
    if not required:
        return None
    source = dict(os.environ) if env is None else dict(env)
    owner = source.get("PRISMABUILD_ACTION_KEY")
    if not owner:
        raise PreparationPublicationRefused(
            "--produced-output needs an admitted PrismaBuild action: "
            "PRISMABUILD_ACTION_KEY is unset")
    if metadata_root is None:
        raise PreparationPublicationRefused(
            "--produced-output writes under --metadata-root, the prefix the "
            "produced-output template declares")
    root = os.path.normpath(os.path.abspath(os.fspath(metadata_root)))
    bound = _BOUND.get(owner)
    if bound is None:
        from .stage_a_produced_output import (
            BoundaryProducedBindingError, BoundaryProducedPublication)
        try:
            publication = BoundaryProducedPublication.bind_from_admitted_owner(
                slot=PREPARATION_SLOT, env=source)
        except BoundaryProducedBindingError as exc:
            raise PreparationPublicationRefused(
                f"this preparation is an admitted PrismaBuild action ({owner[:12]}) "
                f"and cannot bind its produced output: {exc}") from exc
        bound = PreparationPublication(publication)
        _BOUND[owner] = bound
    if os.path.realpath(bound.output_prefix) != os.path.realpath(root):
        raise PreparationPublicationRefused(
            f"the produced-output template declares {bound.output_prefix}, but "
            f"this preparation writes under {root}")
    return bound


def reset_preparation_publications_for_tests() -> None:
    _BOUND.clear()


def publish_files(publication: PreparationPublication | None, kind: str,
                  files: Sequence[tuple[Path, bytes, str]]) -> list[Path]:
    """Publish ``files`` in order, first-writer, as one produced batch.

    ``files`` is ``[(path, payload, where), ...]``. A path already holding
    ``payload`` is left alone and is not this action's; a path holding other
    bytes refuses before anything is written. Without a publication each file
    is published with ``publish_new_bytes``, as before. With one, the files
    this call creates are named in one prewrite with their exact byte count
    before the first byte, written, and committed at their origin. Returns
    the paths this call created.
    """
    from .cost_stage_checkpoint import publish_new_bytes

    planned: list[tuple[Path, bytes, str]] = []
    seen: dict[str, bytes] = {}
    for path, payload, where in files:
        path = Path(path)
        payload = bytes(payload)
        key = os.path.normpath(os.path.abspath(str(path)))
        if key in seen:
            if seen[key] != payload:
                raise ValueError(f"{where}: {path} is published twice with different bytes")
            continue
        seen[key] = payload
        if path.exists() or path.is_symlink():
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise ValueError(f"{where} unreadable at {path}: {exc}") from exc
            if existing != payload:
                raise ValueError(
                    f"refusing to overwrite differing bytes at {path}: a re-seal "
                    f"is a new reviewed directory, never an edit")
            continue
        if not payload:
            raise ValueError(f"{where}: a published file cannot be empty")
        planned.append((path, payload, where))
    if not planned:
        return []
    if publication is not None:
        from .stage_a_produced_output import BoundaryProducedPrewriteRefused
        outside = [str(path) for path, _, _ in planned
                   if not publication.publication.contains(path)]
        if outside:
            raise PreparationPublicationRefused(
                f"the preparation's {kind!r} batch names paths outside the "
                f"template's prefix {publication.output_prefix}: {outside}")
        publication.claim_kind(kind)
        try:
            publication.publication.require_prewrite(
                batch_id=publication.batch_id(kind),
                payload_ceiling_bytes=sum(len(payload) for _, payload, _ in planned),
                temp_ceiling_bytes=max(len(payload) for _, payload, _ in planned),
                paths=[os.path.normpath(os.path.abspath(str(path)))
                       for path, _, _ in planned])
        except (BoundaryProducedPrewriteRefused,
                publication._po.ProducedOutputError) as exc:
            raise PreparationPublicationRefused(
                f"the preparation's {kind!r} batch: {exc}") from exc
    created: list[tuple[str, bytes]] = []
    for path, payload, where in planned:
        if publish_new_bytes(path, payload):
            created.append((os.path.normpath(os.path.abspath(str(path))), payload))
            continue
        # Another writer reached the name between the check and the link.
        if path.read_bytes() != payload:
            raise ValueError(
                f"refusing to overwrite differing bytes at {path}: a re-seal "
                f"is a new reviewed directory, never an edit")
    if publication is not None and created:
        publication.commit(kind, created)
    return [Path(path) for path, _ in created]


__all__ = [
    "DATA_MANIFEST_MAX_BYTES", "PREPARATION_READ_PHASE", "PREPARATION_SLOT",
    "PreparationPublication", "PreparationPublicationRefused",
    "bind_preparation_publication", "build_preparation_template", "file_entry",
    "head_phase_entries", "preparation_payload_ceiling",
    "preparation_read_entries", "preparation_read_manifest", "publish_files",
    "reset_preparation_publications_for_tests",
]
