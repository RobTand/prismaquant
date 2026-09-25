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
  are declared before the action is claimed. With the tools'
  ``--data-manifest-sha256`` flag, :func:`bind_staged_reads` makes every
  declared read come from PrismaBuild's stage through the residency reader,
  its bytes checked against the entry's digest, and refuses a read the stage
  does not hold instead of reading the pool (PQ #1092). Without the flag the
  reads are plain file reads, as before.
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


def _hash_range(path: str, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        handle.seek(offset)
        remaining = size
        while remaining:
            chunk = handle.read(min(remaining, 8 << 20))
            if not chunk:
                raise ValueError(f"a declared read runs past the end of {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def file_entry(path: str | os.PathLike, sha256: str | None = None) -> dict:
    """A whole-file entry for a present file, with its SHA-256.

    ``sha256`` is the file's bound digest when the caller holds one; without
    it the file is hashed. Every entry declares its digest (PQ #1092), so
    PrismaBuild's staged copy is bound to these bytes and the preparation
    checks what it reads against them.
    """
    path = os.path.normpath(os.path.abspath(os.fspath(path)))
    size = os.path.getsize(path)
    if size <= 0:
        raise ValueError(f"a declared read is an empty file: {path}")
    if sha256 is None:
        sha256 = _hash_range(path, 0, size)
    return {"path": path, "offset": 0, "bytes": int(size), "sha256": str(sha256)}


def preparation_read_entries(*, head: Sequence[Mapping], files: Iterable,
                             ranges: Iterable[tuple[str, int, int]] = ()) -> list[dict]:
    """The preparation's reads as data-manifest entries, one per ``(path, offset)``.

    ``head`` is the parent's head phase, ``files`` the whole files the tools
    read besides it (their own arguments, the Stage A proofs, the production
    pickle), and ``ranges`` the partial reads (the model config, the
    safetensors index and headers: ``header_reads`` of
    ``layer_streaming.streaming_source_plan``). A range
    that a whole-file entry at the same offset already covers is dropped;
    PrismaBuild refuses a repeated ``(path, offset)``.

    A ``files`` item is a path or a ``(path, sha256)`` pair; a bare path, and
    every range, is hashed here, so each entry the tools add declares its
    digest (PQ #1092). A head entry keeps the digest the parent gives it.
    """
    entries: list[dict] = []
    index: dict[tuple[str, int], int] = {}

    def add(entry):
        key = (entry["path"], entry["offset"])
        if key in index:
            # The longer read wins whole, digest included: a digest names
            # exactly its entry's bytes, so it is never stretched.
            if entry["bytes"] > entries[index[key]]["bytes"]:
                entries[index[key]] = dict(entry)
            return
        index[key] = len(entries)
        entries.append(dict(entry))

    for entry in head:
        add({"path": os.path.normpath(str(entry["path"])), "offset": int(entry["offset"]),
             "bytes": int(entry["bytes"]), "sha256": entry.get("sha256")})
    for item in files:
        path, sha256 = (item if isinstance(item, tuple) else (item, None))
        add(file_entry(path, sha256))
    for path, offset, size in ranges:
        path = os.path.normpath(str(path))
        key = (path, int(offset))
        if key in index and entries[index[key]]["bytes"] >= int(size):
            continue
        add({"path": path, "offset": int(offset), "bytes": int(size),
             "sha256": _hash_range(path, int(offset), int(size))})
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
# Staged reads (PQ #1092)
# --------------------------------------------------------------------------

class PreparationReadRefused(ValueError):
    """A declared read the stage could not serve; the pool is never read instead."""


class StagedPreparationReads:
    """This process's strict reads: the stage, or a refusal.

    Bound once per process by :func:`bind_staged_reads`, from the tools'
    ``--data-manifest-sha256``. Every read of a declared input goes through
    the process residency resolver, bound to that manifest digest, and a
    lifetime-pinned lease window (``staged_whole_file.read_staged_entry``).
    The bytes read must hash to the map entry's digest, which PrismaBuild
    derives from the sealed manifest's declared digest, and to the caller's
    own pinned digest when it has one. A path the stage does not hold
    refuses. It is never read from the pool.

    ``own_outputs`` are the roots this action writes (the metadata root). A
    file under one was written by this run or by an earlier run of the same
    generation. It is not a manifest input, so it is read where it is.

    Not routed yet: the Stage B head walk
    (``tessera_joint_aura.load_measured_anchor_input``, run by the
    generator's ``--head-slices``) opens its inputs itself. Those reads still
    come from the pool under this binding until PQ #1082 routes them. That
    exemption is named here and in ``docs/ARCHITECTURE.md``, not silent.
    Nor, for a source checkpoint with FP8 weights, are the config and index
    reads ``layer_streaming._build_fp8_scale_inv_map`` makes for the source
    plan (found during PQ #1139; ``docs/ARCHITECTURE.md`` names them too).
    """

    def __init__(self, resolver, *, manifest_sha256: str, own_outputs=()):
        self.resolver = resolver
        self.manifest_sha256 = manifest_sha256
        self.own_outputs = tuple(os.path.normpath(os.path.abspath(os.fspath(root)))
                                 for root in own_outputs)

    def owns(self, path) -> bool:
        path = os.path.normpath(os.path.abspath(os.fspath(path)))
        return any(path == root or path.startswith(root + os.sep)
                   for root in self.own_outputs)

    def add_own_output(self, root) -> None:
        root = os.path.normpath(os.path.abspath(os.fspath(root)))
        if root not in self.own_outputs:
            self.own_outputs = (*self.own_outputs, root)

    def _read_entry(self, path: Path, staged: dict, *, where: str,
                    sha256: str | None) -> bytes:
        from .staged_tier_policy import TierPolicyRefused
        from .staged_whole_file import read_staged_entry

        try:
            raw = read_staged_entry(self.resolver, path, staged, label=where)
        except TierPolicyRefused as exc:
            raise PreparationReadRefused(
                f"{where} at {path} was not served from the stage: {exc}") from exc
        digest = hashlib.sha256(raw).hexdigest()
        if digest != staged["sha256"] or (sha256 is not None and digest != sha256):
            raise PreparationReadRefused(
                f"{where} at {path}: the staged bytes hash to {digest}, not the "
                f"digest the manifest and the caller require")
        return raw

    def whole(self, path, *, sha256: str | None = None, where: str) -> bytes:
        """One whole declared file off the stage, digest-checked."""
        path = Path(path)
        if self.owns(path):
            return path.read_bytes()
        staged = self.resolver.staged_read(path, expected_sha256=sha256)
        if staged is None:
            raise PreparationReadRefused(
                f"{where} at {path} is not staged for manifest "
                f"{self.manifest_sha256[:12]}: refusing rather than reading the pool")
        return self._read_entry(path, staged, where=where, sha256=sha256)

    def prefix(self, path, *, nbytes: int, where: str) -> bytes:
        """The staged range entry at offset 0 that covers ``nbytes`` of ``path``.

        The safetensors header reads (``header_reads`` of
        ``layer_streaming.streaming_source_plan``) are declared as
        ``[0, 8 + length)`` ranges. This returns that whole entry, so the caller reads the
        length prefix and the header from one staged, digest-checked copy.
        """
        path = Path(path)
        staged = self.resolver.staged_range(path, 0, nbytes)
        if staged is None or staged.get("offset") != 0:
            raise PreparationReadRefused(
                f"{where} at {path} [0, {nbytes}) is not staged for manifest "
                f"{self.manifest_sha256[:12]}: refusing rather than reading the pool")
        return self._read_entry(path, staged, where=where, sha256=None)


_STAGED_READS: StagedPreparationReads | None = None


def bind_staged_reads(*, manifest_sha256: str, allowed_tiers: str,
                      own_outputs=(), env: Mapping[str, str] | None = None
                      ) -> StagedPreparationReads:
    """Make this process's declared reads strict (PQ #1092).

    ``manifest_sha256`` is the digest of the data manifest the action was
    submitted with (``stage_b_preparation_submission`` seals it into the
    command line). This activates the strict tier policy with
    ``allowed_tiers`` and binds the process residency resolver to the
    manifest. A process without ``PRISMABUILD_RESIDENCY_MAP`` was not
    launched with ``--residency stage`` and refuses. A second call with
    the same digest adds its roots to the bound reads; another digest refuses.
    """
    global _STAGED_READS
    from .residency_map import ENV_VAR, bind_residency_manifest, residency_resolver
    from .staged_tier_policy import activate_staged_tier_policy

    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64 or \
            any(c not in "0123456789abcdef" for c in manifest_sha256):
        raise PreparationReadRefused(
            "--data-manifest-sha256 is the 64-character digest of the "
            "preparation's data manifest")
    if _STAGED_READS is not None:
        if _STAGED_READS.manifest_sha256 != manifest_sha256:
            raise PreparationReadRefused(
                "this process's reads are already bound to manifest "
                f"{_STAGED_READS.manifest_sha256[:12]}, not {manifest_sha256[:12]}")
        for root in own_outputs:
            _STAGED_READS.add_own_output(root)
        return _STAGED_READS
    source = os.environ if env is None else env
    if not source.get(ENV_VAR):
        raise PreparationReadRefused(
            f"--data-manifest-sha256 reads every input off the stage, but "
            f"{ENV_VAR} is unset: the action was not launched with --residency stage")
    try:
        activate_staged_tier_policy(allowed_tiers)
    except ValueError as exc:
        raise PreparationReadRefused(f"--allowed-tiers: {exc}") from exc
    bind_residency_manifest(manifest_sha256)
    resolver = residency_resolver()
    if resolver is None:
        raise PreparationReadRefused(f"{ENV_VAR} names no residency map")
    _STAGED_READS = StagedPreparationReads(
        resolver, manifest_sha256=manifest_sha256, own_outputs=own_outputs)
    _install_bound_reader(_STAGED_READS)
    return _STAGED_READS


def staged_reads() -> StagedPreparationReads | None:
    """The bound strict reads, or None when the process reads plainly."""
    return _STAGED_READS


def _install_bound_reader(reads: StagedPreparationReads | None) -> None:
    # ``tessera_joint_allocation._read_bound`` serves the catalog pair's
    # control documents and pickles; the preparation reads them through it.
    from . import tessera_joint_allocation as allocation
    allocation.BOUND_READER = (None if reads is None else
                               lambda path, sha256, label: reads.whole(
                                   path, sha256=sha256, where=label))


def reset_staged_reads_for_tests() -> None:
    global _STAGED_READS
    _STAGED_READS = None
    _install_bound_reader(None)


def read_input(path, *, sha256: str | None = None, where: str) -> bytes:
    """A declared input's bytes: off the stage when bound, else read plainly."""
    reads = _STAGED_READS
    if reads is None:
        return Path(path).read_bytes()
    return reads.whole(path, sha256=sha256, where=where)


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
    "PreparationReadRefused", "StagedPreparationReads",
    "bind_preparation_publication", "bind_staged_reads",
    "build_preparation_template", "file_entry",
    "head_phase_entries", "preparation_payload_ceiling",
    "preparation_read_entries", "preparation_read_manifest", "publish_files",
    "read_input", "reset_preparation_publications_for_tests",
    "reset_staged_reads_for_tests", "staged_reads",
]
