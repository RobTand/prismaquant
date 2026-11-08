"""Stage B input preparation and retained output-authority boundaries.

Immutable staged inputs use the existing resolver and whole-file reader.
Read-set/template builders retain their scientific wire formats. Launch-bound outputs use the application publication owner.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .digests import DIRECT_ASCII_SPACED_LAX, bytes_sha256hex, file_range_sha256hex, text_sha256hex

#: The template slot every Stage B preparation output is filed under.
PREPARATION_SLOT = "stage_b_metadata"
#: The template id prefix; the id's suffix digests the template body, so a
#: changed root, tier or ceiling files a template of its own (PQ #1054).
PREPARATION_TEMPLATE_PREFIX = "pq-stage-b-preparation"
#: PrismaBuild's produced-output template schema (``produced_output.
#: TEMPLATE_SCHEMA_V1``), restated for the submitter, which runs outside PB.
PRODUCED_OUTPUT_TEMPLATE_SCHEMA = "prismaquant.produced_declaration.v1"
#: The read phase the preparation's manifest declares.
PREPARATION_READ_PHASE = "stage-b-preparation"
#: PrismaBuild's own per-manifest bound (``prismabuild.core
#: DATA_MANIFEST_MAX_BYTES``), restated as ``campaign_data_manifests``
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
    return file_range_sha256hex(path, offset, size)


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
    """Strict whole-file and header-prefix reads through the existing resolver.

    Every declared input is read from an immutable staged extent and checked
    against its digest. Own outputs retain their existing local read path.
    Source-generation and produced-output authority remain separate unit4 work.
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
                    sha256: str | None, max_bytes: int | None = None) -> bytes:
        from .staged_tier_policy import TierPolicyRefused
        from .staged_whole_file import read_staged_entry

        try:
            raw = read_staged_entry(self.resolver, path, staged, label=where,
                                    max_bytes=max_bytes)
        except TierPolicyRefused as exc:
            raise PreparationReadRefused(
                f"{where} at {path} was not served from the stage: {exc}") from exc
        digest = bytes_sha256hex(raw)
        if digest != staged["sha256"] or (sha256 is not None and digest != sha256):
            raise PreparationReadRefused(
                f"{where} at {path}: the staged bytes hash to {digest}, not the "
                f"digest the manifest and the caller require")
        return raw

    def whole(self, path, *, sha256: str | None = None, where: str,
              max_bytes: int | None = None) -> bytes:
        """One whole declared file off the stage, digest-checked and bounded."""
        path = Path(path)
        if self.owns(path):
            if max_bytes is None:
                return path.read_bytes()
            from .io_engine import read_file

            return read_file(path, max_bytes)[0]
        staged = self.resolver.staged_read(path, expected_sha256=sha256)
        if staged is None:
            raise PreparationReadRefused(
                f"{where} at {path} is not staged for manifest "
                f"{self.manifest_sha256[:12]}: refusing rather than reading the pool")
        return self._read_entry(path, staged, where=where, sha256=sha256,
                                max_bytes=max_bytes)

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
    manifest. A process without an explicit staged-manifest path was not
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
    if _STAGED_READS is None:
        from .staged_manifest import READ_ENV
        digest = os.environ.get(READ_ENV)
        if digest:
            return bind_staged_reads(manifest_sha256=digest, allowed_tiers='ssd')
    return _STAGED_READS


def _install_bound_reader(reads: StagedPreparationReads | None) -> None:
    # ``stage_inputs.read_bound`` serves the catalog pair's control
    # documents and pickles; the preparation reads them through it.
    from . import stage_inputs
    stage_inputs.BOUND_READER = (None if reads is None else
                                 lambda path, sha256, label, *, max_bytes=None: reads.whole(
                                     path, sha256=sha256, where=label, max_bytes=max_bytes))


def reset_staged_reads_for_tests() -> None:
    global _STAGED_READS
    _STAGED_READS = None
    _install_bound_reader(None)


def read_input(path, *, sha256: str | None = None, where: str) -> bytes:
    """A declared input's bytes: off the stage when bound, else read plainly."""
    reads = staged_reads()
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


def build_preparation_template(*, metadata_root, payload_max_bytes):
    from .stage_a_produced_output import build_boundary_template
    root = os.path.normpath(os.path.abspath(os.fspath(metadata_root)))
    declaration = build_boundary_template(output_prefix=root,
        artifact_max_bytes=payload_max_bytes, checkpoint_max_bytes=0,
        group_size=1024, max_entry_tensor_bytes=min(payload_max_bytes, (4 << 30) - 65536),
        template_id=PREPARATION_TEMPLATE_PREFIX, write_only=True)
    declaration['slot'] = PREPARATION_SLOT
    return declaration


class PreparationPublication:
    """One native launch's write-only metadata publication."""

    def __init__(self, publication):
        self.publication = publication
        if not publication.write_only:
            raise PreparationPublicationRefused('preparation requires a write-only declaration')
        publication.bind_session({'generation': publication.generation,
            'run_identity_sha256': publication.binding['sha256']})
        self.batches, self._kinds = [], set()

    @property
    def output_prefix(self):
        return self.publication.output_prefix

    def batch_id(self, kind):
        from .stage_a_produced_output import _name
        return _name(f'stage-b-prep-{_name(kind)}-{self.publication.generation}')

    def claim_kind(self, kind):
        if kind in self._kinds:
            raise PreparationPublicationRefused('preparation group may be written once per launch')
        self._kinds.add(kind)

    def commit(self, kind, created):
        from .streamed_boundary_artifacts import ProducedFileReference
        batch = self.batch_id(kind)
        descriptors = [self.publication.descriptor_for(ProducedFileReference(
            path, Path(path).name, len(payload), bytes_sha256hex(payload)),
            producer_generation=batch) for path, payload in created]
        out = self.publication.publish(batch_id=batch, descriptors=descriptors)
        self.batches.append({'kind': kind, 'batch_id': batch, 'files': len(descriptors),
            'bytes': sum(len(payload) for _, payload in created), 'ref': out['ref']})
        return out

    def finalize(self, path):
        return self.publication.finalize(path)


_BOUND = {}


def bind_preparation_publication(metadata_root, *, required, env=None):
    if not required:
        return None
    from .stage_a_produced_output import BoundaryProducedPublication
    class MetadataPublication(BoundaryProducedPublication):
        DEFAULT_SLOT = PREPARATION_SLOT
    if metadata_root is None:
        raise PreparationPublicationRefused('explicit metadata root required')
    publication = MetadataPublication.bind_from_admitted_owner()
    root = str(Path(metadata_root).absolute())
    if root != publication.output_prefix:
        raise PreparationPublicationRefused('metadata root differs from launch declaration')
    key = publication.producer['launch_sha256']
    if key not in _BOUND:
        _BOUND[key] = PreparationPublication(publication)
    elif _BOUND[key].publication.binding != publication.binding:
        raise PreparationPublicationRefused('metadata publication declaration changed')
    return _BOUND[key]


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
        except BoundaryProducedPrewriteRefused as exc:
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
