#!/usr/bin/env python3
"""Regenerate the layer-quantum records, slice manifests and adjoint manifest
from sealed inputs with an authoritative output root (PQ #838 defect 4).

Producer D3: binding (or rebinding) records is a NEW identity set, never an
edit. This tool replays the producer call the external binder made -- plan,
prepared, parent manifest, derivation, partition -- with the output root
taken from the plan (never the tool's own directory, the doubling the live
``layer-quanta/layer-quanta`` records carry), and optionally re-seals
against a stage-A receipt. Old records and history stay where they are.
Record files land in the reviewed directory the caller names; slice
manifests land at the producer-named absolute paths the records bind
(verified to resolve after writing). The adjoint manifest is the phase
worker's file and is never written here.

Inputs may be plain JSON or gzip (the parent manifest is a ``.json.gz``):
the digest always covers wire bytes, gunzip output is bounded, and trailing
bytes after a gzip member refuse.

Control-metadata generation root (PQ #884): ``--metadata-root`` places the
CONTROL files this producer seals -- slice manifests, the record
payload/index/derivation set (default ``--records-out`` becomes
``{metadata_root}/records``), and any newly bound boundary/executable
readset manifests -- into an explicit generation-specific namespace while
every reference to actual stage-A/run artifacts is retained:
``output_space`` and ``adjoint.boundary_artifacts`` keep deriving from the
data output root, which stays the plan's. Existing calls without the flag
are byte-identical (the control root defaults to the historical
``{output_root}/layer-quanta``).

Gates, all fail closed with exit 3:

* Gate 1a (``--expect-existing`` + ``--original-root``): regenerate
  receipt-less at the original root and require every on-disk record to
  match canonical JSON -- input fidelity, proved before anything moves.
  Gate 1a ALWAYS reproduces the default layout at the original root, with
  or without ``--metadata-root``; the metadata seam never weakens it.
* Gate 1b: regenerate at the new root and require only the authorized path
  fields (plus the receipt seal when binding) to move; everything else is
  strictly equal and the new identity must recompute. A regeneration
  against known-bad originals can never "reproduce" by definition, so the
  old root is explicit rather than inferred. With ``--metadata-root`` the
  authorized move is the narrower relocation instead: the data fields do
  NOT move and only the control placement (slice manifest path/digest and
  the receipt seal when binding) may change -- and it requires the data
  output root to stay the original root.
* ``--compare-existing DIR``: a read-only scientific-binding comparison of
  the produced generation against a prior on-disk generation that need not
  be exactly reproducible (a prior producer revision's sealed set, e.g. the
  pre-#852 pending generation). Requires identical quanta with identical
  campaign identity (plan/prepared/parent digests, scope, roster),
  per-layer membership/extent fields (chunks, windows, read-set extents),
  ``output_space`` and adjoint source roots, and identical slice-manifest
  entries; control placement (manifest path/digest, slice argv/phase
  tables) and receipt binding are the only drift it admits. Separate from
  Gate 1: refuse to combine the two in one run.
* Gate 2: ``--adjoint-receipt`` (the completed receipt) and/or one
  ``--adjoint-band`` per sealed checkpoint band must exist and load; the
  bound set is regenerated for the layers those proofs cover (PQ #993):
  every record binds its stage-A slice (``adjoint.slice_sha256`` and
  ``adjoint.slice_path``), every ``identity_sha256`` moves, and each slice
  file is published before the record naming it. A later run with more
  bands adds records and leaves every earlier one byte-identical; bands
  publish one index each (``records.band-NNN.json``), the completed
  receipt ``records.json``. Only with the opt-in
  ``--boundary-readsets`` / ``--executable-readsets`` flags does each
  record additionally get new bound generations (bulk readset, PQ #848;
  single executable manifest, PQ #862) whose files land under
  ``bound-readsets/`` -- the new tree's, or the metadata root's.
  With ``--executable-readsets --source-layers-prefix PREFIX`` (PQ #900),
  read the sealed plan's checkpoint index and shard headers once and
  complete every chain/own source phase against the actual tensor spans,
  and the head phase against the resident head (PQ #1095). The spans come
  from the streaming loader's own selection
  (``layer_streaming.streaming_source_plan``); PREFIX is its live decoder
  layers prefix.
  The parent, slices, chunk tiling and plan/prepared bytes stay intact;
  only the new executable manifest and its binding carry the completion.
  Without the prefix the historical readset bytes reproduce unchanged.
  ``--check-only`` runs the gates and writes nothing, mirroring the
  external binder's dry run.

This is a producer, never a scheduler: it submits no PB rows and holds no
state. Run it on a checkout inside the PB code closure (unlike the external
binder script, which lives beside the data and cannot be captured), then
submit from its outputs.

With ``--produced-output`` (PQ #1070), as an admitted PrismaBuild action
submitted with the data manifest and write-only produced-output template
that ``tools/stage_b_preparation_submission.py`` writes, it files a prewrite
for each group of files it creates under ``--metadata-root`` and commits
them at their origin (``prismaquant.stage_b_prep_io``). Without the flag it
writes exactly as before.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zlib
from pathlib import Path

if __package__:
    from prismaquant.joint_layer_quanta import (
        check_quantum_for_campaign, derive_stride,
        emit_quantum_boundary_readsets, emit_quantum_executable_readsets,
        layer_quanta, seal_manifest_bytes)
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from prismaquant.joint_layer_quanta import (
        check_quantum_for_campaign, derive_stride,
        emit_quantum_boundary_readsets, emit_quantum_executable_readsets,
        layer_quanta, seal_manifest_bytes)

EXIT_REFUSED = 3

#: Bound the gunzip the manifest read may materialize: the fleet's own
#: manifest ingestion bounds stored and decoded bytes independently
#: (``prismabuild.core`` ``DATA_MANIFEST_MAX_*``); this reader caps the
#: decoded document the same way rather than trusting the member size.
MAX_DECODED_MANIFEST_BYTES = 512 * 1024 * 1024
_GZIP_MAGIC = b"\x1f\x8b"


def _fail(message: str) -> int:
    print(f"regenerate_joint_quanta: refused: {message}", file=sys.stderr)
    return EXIT_REFUSED


def _load_json(path: Path, *, digest: str | None, where: str):
    """Read, digest-verify, and parse a JSON document, gzip-transparent.

    The digest always covers the wire bytes. Detection is by gzip magic,
    never by suffix; the gunzip output is bounded and trailing garbage
    after a gzip member refuses rather than being silently ignored."""
    from prismaquant.stage_b_prep_io import read_input
    try:
        raw = read_input(path, sha256=digest, where=where)
    except OSError as exc:
        raise ValueError(f"{where} unreadable at {path}: {exc}") from exc
    if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{where} digest mismatch at {path}")
    if raw[:2] == _GZIP_MAGIC:
        # wbits=31 decodes the gzip wrapper; the decompressor -- not a
        # filename or member count -- reports truncation (eof) and trailing
        # bytes (unused_data), and output past the bound refuses.
        decompressor = zlib.decompressobj(31)
        try:
            raw = decompressor.decompress(raw, MAX_DECODED_MANIFEST_BYTES + 1)
        except zlib.error as exc:
            raise ValueError(f"{where} is not valid gzip at {path}: "
                             f"{exc}") from exc
        if decompressor.unused_data:
            raise ValueError(f"{where} has trailing bytes after its gzip "
                             f"member at {path}: refusing")
        if not decompressor.eof or len(raw) > MAX_DECODED_MANIFEST_BYTES:
            raise ValueError(f"{where} gzip member is truncated or its "
                             f"decoded document exceeds "
                             f"{MAX_DECODED_MANIFEST_BYTES} bytes at {path}")
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"{where} is not JSON at {path}: {exc}") from exc


def _pretty(value) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


#: Record fields allowed to move between the original root and the new
#: root, with the old prefix swapped for the new one exactly. Everything
#: else must be byte-stable canonical JSON; ``identity_sha256`` and the
#: slice digest are consequences (recomputed and re-verified, never
#: compared across), and the stage-A binding (:data:`_STAGE_A_BINDING`) may
#: move only when a stage-A proof is bound in this run.
_MOVED_PATH_FIELDS = (
    ("output_space", "root"),
    ("output_space", "cost_payload"),
    ("output_space", "results"),
    ("output_space", "counters"),
    ("output_space", "checkpoint_dir"),
    ("read_set", "manifest_path"),
    ("adjoint", "boundary_artifacts"),
)


#: The record's stage-A binding: unbound records carry ``receipt_sha256:
#: None``; bound ones (PQ #993) carry their slice digest and file.
_STAGE_A_BINDING = ("receipt_sha256", "slice_sha256", "slice_path")


def _pop_stage_a_binding(adjoint: dict) -> tuple:
    return tuple(adjoint.pop(key, None) for key in _STAGE_A_BINDING)


def _moved(record: dict, field: tuple[str, str]) -> str:
    value = record.get(field[0], {})
    path = value.get(field[1]) if isinstance(value, dict) else None
    if not isinstance(path, str) or not path:
        raise ValueError(f"record {record.get('quantum_id')!r} has no "
                         f"{field[0]}.{field[1]}")
    return path


def _check_authorized_diff(old: dict, new: dict, *, old_root: str, bound: bool,
                           where: str) -> None:
    """The new record may differ from the old one only where the root move
    (and, when binding, the receipt seal) requires. Anything else refuses.

    Each moved path must be the old path with exactly the old-root prefix
    swapped for the new one. ``identity_sha256`` is not compared across --
    it is recomputed over the new record and required to match, which
    proves the seal is well-formed rather than blind. The slice digest is
    re-verified against the written slice file by the caller, for the same
    reason. ``adjoint.receipt_sha256`` may differ only when this run binds
    a receipt; otherwise it must be equal.
    """
    from prismaquant.joint_layer_quanta import canonical_sha256
    qid = new.get("quantum_id", "?")
    old_root = old_root.rstrip("/")
    # The new tree root is whatever the first moved path swapped to; every
    # other moved path must swap to that same root, so a record cannot half
    # move between trees.
    new_prefix: str | None = None
    for field in _MOVED_PATH_FIELDS:
        old_path = _moved(old, field)
        new_path = _moved(new, field)
        if not old_path.startswith(old_root + "/"):
            raise ValueError(f"Gate 1 {where}: {qid} {field[0]}.{field[1]} "
                             f"{old_path!r} is not under the original root "
                             f"{old_root!r}")
        suffix = old_path[len(old_root):]
        if new_prefix is None:
            if not suffix or not new_path.endswith(suffix):
                raise ValueError(f"Gate 1 {where}: {qid} {field[0]}.{field[1]} "
                                 f"{new_path!r} is not the moved "
                                 f"{old_path!r}")
            new_prefix = new_path[:len(new_path) - len(suffix)]
        elif new_path != new_prefix + suffix:
            raise ValueError(f"Gate 1 {where}: {qid} {field[0]}.{field[1]} "
                             f"{new_path!r} leaves the moved tree")
    old_body = {k: v for k, v in old.items() if k != "identity_sha256"}
    new_body = {k: v for k, v in new.items() if k != "identity_sha256"}
    # The moved path fields were checked by exact prefix swap above; drop
    # them here so this comparison holds everything else strictly equal.
    old_body.pop("output_space", None)
    new_body.pop("output_space", None)
    old_body.pop("read_set", None)
    read_set = dict(new_body.pop("read_set", {}))
    read_set.pop("manifest_sha256", None)
    read_set.pop("manifest_path", None)
    old_read = dict(old.get("read_set", {}))
    old_read.pop("manifest_sha256", None)
    old_read.pop("manifest_path", None)
    if old_read != read_set:
        raise ValueError(f"Gate 1 {where}: {qid} read_set differs beyond "
                         f"the moved manifest path")
    old_adjoint = dict(old_body.pop("adjoint", {}))
    new_adjoint = dict(new_body.pop("adjoint", {}))
    old_adjoint.pop("boundary_artifacts", None)
    new_adjoint.pop("boundary_artifacts", None)
    old_receipt = _pop_stage_a_binding(old_adjoint)
    new_receipt = _pop_stage_a_binding(new_adjoint)
    if old_adjoint != new_adjoint:
        raise ValueError(f"Gate 1 {where}: {qid} adjoint block differs "
                         f"beyond the stage-A binding")
    if not bound and old_receipt != new_receipt:
        raise ValueError(f"Gate 1 {where}: {qid} stage-A binding moved with "
                         f"no stage-A proof bound")
    if json.dumps(old_body, sort_keys=True) != json.dumps(new_body, sort_keys=True):
        raise ValueError(f"Gate 1 {where}: {qid} differs outside the moved "
                         f"paths")
    body = {key: value for key, value in new.items()
            if key != "identity_sha256"}
    if canonical_sha256(body, where=f"Gate 1 {where} {qid}") != new.get(
            "identity_sha256"):
        raise ValueError(f"Gate 1 {where}: {qid} identity does not recompute")


# --- Control-metadata generation seam (PQ #884) ---------------------------
#
# The data output root (plan output_root) owns execution outputs and stage-A
# artifacts; an explicit metadata root owns ONLY the control files this
# producer seals. The helpers below are the authorized-drift vocabulary
# shared by the relocation gate and the prior-generation comparison.


def _load_verified_json(path: Path, digest: str, *, where: str):
    """Load a sealed prior document, refusing bytes that do not hash to the
    digest the prior record sealed for them."""
    return _load_json(path, digest=digest, where=where)


def _strip_leading_zero_head_phase(phases: object, *, where: str) -> list:
    """Drop a leading phase row iff it is EXACTLY the known pre-#852
    zero-byte head annotation: ``{"name": "head", "bytes": 0,
    "cumulative_bytes": 0}`` (extra keys are not that shape and refuse).
    Returns the remaining ordered rows -- every nonzero phase name, byte
    count and cumulative bound must then compare exactly."""
    if not isinstance(phases, list) or not phases:
        raise ValueError(f"{where}: the slice manifest seals no phase table")
    first = phases[0]
    if (isinstance(first, dict) and first.get("name") == "head"
            and first.get("bytes") == 0
            and first.get("cumulative_bytes") == 0
            and set(first) == {"name", "bytes", "cumulative_bytes"}):
        return list(phases[1:])
    return list(phases)


def _compare_control_phases(old_phases: object, new_phases: object,
                            *, where: str) -> bool:
    """True when the nonzero phase tables are identical after removing at
    most the one known zero-byte head row. Order, names, byte counts and
    cumulative bounds compare exactly; any other phase drift refuses."""
    old_rest = _strip_leading_zero_head_phase(old_phases, where=where)
    if not isinstance(new_phases, list):
        raise ValueError(f"{where}: the new slice manifest seals no phase "
                         "table")
    return old_rest == new_phases


def _compare_control_argv(old_argv: object, new_argv: object, *,
                          old_record_path: str, new_record_path: str,
                          where: str) -> bool:
    """True when the sealed slice argv differs ONLY by the exact record-path
    relocation (``--quantum`` value). The interpreter, entry point,
    ``--output-root`` and every other binding compare exactly."""
    if not isinstance(old_argv, list) or not isinstance(new_argv, list):
        raise ValueError(f"{where}: the slice manifest seals no argv")
    if len(old_argv) != len(new_argv):
        return False
    for index, (old_item, new_item) in enumerate(zip(old_argv, new_argv)):
        if old_item == new_item:
            continue
        # The one permitted relocation: the value after the --quantum flag.
        if (index > 0 and old_argv[index - 1] == "--quantum"
                and old_item == old_record_path
                and new_item == new_record_path):
            continue
        return False
    return True


def _prior_slice_manifest(record: dict, *, where: str) -> dict:
    """Decode a prior generation's slice manifest, first refusing unless its
    raw wire bytes hash to the digest the prior record sealed (the bytes are
    verified before any semantic part of them is compared)."""
    read_set = record.get("read_set")
    if not isinstance(read_set, dict):
        raise ValueError(f"{where}: the prior record seals no read_set")
    manifest_path = read_set.get("manifest_path")
    manifest_sha256 = read_set.get("manifest_sha256")
    if type(manifest_path) is not str or not manifest_path \
            or type(manifest_sha256) is not str:
        raise ValueError(f"{where}: the prior record names no slice manifest")
    path = Path(manifest_path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{where}: prior slice manifest unreadable at "
                         f"{path}: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError(
            f"{where}: prior slice manifest bytes at {path} do not hash to "
            f"the digest the prior record sealed ({manifest_sha256}): "
            f"refusing to compare unverified bytes")
    return _load_verified_json(path, manifest_sha256, where=f"{where} slice")


def _compare_prior_slice(prior_manifest: dict, new_manifest: dict,
                         *, where: str) -> dict:
    """The scientific core of the prior-generation comparison: every
    top-level field of the new slice -- schema, producer identity, mount
    prefix, the full entries list, the entry/byte accounting, and anything
    a future producer adds -- must be EXACTLY the prior generation's, with
    ``annotations`` compared separately under the exact phase/argv
    allowances. Only the record-path argv relocation and the one known
    zero-byte head phase row may drift."""
    # Every top-level field compares exactly except ``annotations``
    # (compared below with the phase/argv allowances): an unknown
    # scientific field cannot silently vanish from the comparison.
    old_top = {key: value for key, value in prior_manifest.items()
               if key != "annotations"}
    new_top = {key: value for key, value in new_manifest.items()
               if key != "annotations"}
    if old_top != new_top:
        only_old = sorted(set(old_top) - set(new_top))
        only_new = sorted(set(new_top) - set(old_top))
        raise ValueError(
            f"{where}: slice manifest differs beyond the control "
            f"annotations (fields only in the prior: {only_old}, only in "
            f"the new: {only_new}, or a shared field moved): membership, "
            f"accounting or schema drift, refusing")
    old_annotations = dict(prior_manifest.get("annotations") or {})
    new_annotations = dict(new_manifest.get("annotations") or {})
    old_phases = old_annotations.pop("phases", None)
    new_phases = new_annotations.pop("phases", None)
    old_argv = old_annotations.pop("argv", None)
    new_argv = new_annotations.pop("argv", None)
    if old_annotations != new_annotations:
        raise ValueError(
            f"{where}: slice manifest annotations differ beyond the control "
            f"phase table and argv: refusing")
    if not _compare_control_phases(old_phases, new_phases, where=where):
        raise ValueError(
            f"{where}: slice phase tables differ beyond the single known "
            f"zero-byte head row (order, names, bytes or cumulative bounds "
            f"moved): refusing")
    old_record_path = _quantum_flag_value(old_argv)
    new_record_path = _quantum_flag_value(new_argv)
    if not _compare_control_argv(old_argv, new_argv,
                                 old_record_path=old_record_path,
                                 new_record_path=new_record_path,
                                 where=where):
        raise ValueError(
            f"{where}: slice argv differs beyond the exact record-path "
            f"relocation (entry point, output root or another binding "
            f"moved): refusing")
    return {"phases_identical_beyond_zero_head": True,
            "record_path_relocated": old_record_path != new_record_path}


def _quantum_flag_value(argv: object) -> str | None:
    """The value sealed after ``--quantum`` in a slice argv, or None."""
    if not isinstance(argv, list):
        return None
    for index, item in enumerate(argv[:-1]):
        if item == "--quantum" and type(argv[index + 1]) is str:
            return argv[index + 1]
    return None


def _check_authorized_metadata_diff(old: dict, new: dict, *,
                                    original_root: str, metadata_root: str,
                                    slice_sha256: str, bound: bool,
                                    where: str) -> None:
    """The metadata-relocation gate: over an EXACTLY reproduced source
    generation, the new generation may move ONLY the control placement --
    the slice manifest path and its digest (a consequence of the record-path
    argv inside) -- plus the receipt seal when binding. The data fields
    (``output_space``, ``adjoint.boundary_artifacts``) and every scientific
    field stay byte-equal; the new identity must recompute."""
    from prismaquant.joint_layer_quanta import canonical_sha256
    qid = new.get("quantum_id", "?")
    original_root = original_root.rstrip("/")
    if old.get("output_space") != new.get("output_space"):
        raise ValueError(f"Gate 1 {where}: {qid} output_space moved under a "
                         "metadata relocation: the data output root must be "
                         "retained")
    old_adjoint = dict(old.get("adjoint") or {})
    new_adjoint = dict(new.get("adjoint") or {})
    if old_adjoint.pop("boundary_artifacts", None) != \
            new_adjoint.pop("boundary_artifacts", None):
        raise ValueError(f"Gate 1 {where}: {qid} adjoint.boundary_artifacts "
                         "moved under a metadata relocation: the stage-A "
                         "artifact root must be retained")
    old_receipt = _pop_stage_a_binding(old_adjoint)
    new_receipt = _pop_stage_a_binding(new_adjoint)
    if old_adjoint != new_adjoint:
        raise ValueError(f"Gate 1 {where}: {qid} adjoint block differs "
                         "beyond the stage-A binding")
    if not bound and old_receipt != new_receipt:
        raise ValueError(f"Gate 1 {where}: {qid} stage-A binding moved with no "
                         "stage-A proof bound")
    old_read = dict(old.get("read_set") or {})
    new_read = dict(new.get("read_set") or {})
    old_read.pop("manifest_path", None)
    old_read.pop("manifest_sha256", None)
    new_read.pop("manifest_path", None)
    new_read.pop("manifest_sha256", None)
    if old_read != new_read:
        raise ValueError(f"Gate 1 {where}: {qid} read_set differs beyond "
                         "the relocated manifest placement")
    expected_old = (f"{original_root}/layer-quanta/manifests/"
                    f"{qid}.data-manifest.json.gz")
    expected_new = (f"{metadata_root.rstrip('/')}/manifests/"
                    f"{qid}.data-manifest.json.gz")
    if old.get("read_set", {}).get("manifest_path") != expected_old:
        raise ValueError(
            f"Gate 1 {where}: {qid} prior manifest path is not the default "
            f"producer layout at the original root ({expected_old}): "
            f"refusing")
    if new.get("read_set", {}).get("manifest_path") != expected_new:
        raise ValueError(f"Gate 1 {where}: {qid} manifest path is not the "
                         f"producer-named metadata path ({expected_new}): "
                         "refusing")
    if new.get("read_set", {}).get("manifest_sha256") != slice_sha256:
        raise ValueError(f"Gate 1 {where}: {qid} manifest digest does not "
                         "hash the newly produced slice bytes")
    old_body = {k: v for k, v in old.items()
                if k not in ("identity_sha256", "read_set", "adjoint")}
    new_body = {k: v for k, v in new.items()
                if k not in ("identity_sha256", "read_set", "adjoint")}
    if json.dumps(old_body, sort_keys=True) != json.dumps(new_body,
                                                         sort_keys=True):
        raise ValueError(f"Gate 1 {where}: {qid} differs outside the "
                         "relocated control metadata")
    body = {key: value for key, value in new.items()
            if key != "identity_sha256"}
    if canonical_sha256(body, where=f"Gate 1 {where} {qid}") != new.get(
            "identity_sha256"):
        raise ValueError(f"Gate 1 {where}: {qid} identity does not recompute")


#: NEW-record top-level blocks that are pure control placement/binding in
#: a prior-generation comparison; everything else compares strictly. The
#: prior side tolerates NONE of the binding blocks: a pre-bound prior
#: generation (boundary/executable readsets bind actual reads, not mere
#: filenames) is refused outright below rather than silently ignored.
_CONTROL_BLOCKS = ("identity_sha256", "read_set", "adjoint",
                   "boundary_readset", "executable_readset")


def _compare_existing_generation(prior_dir: Path, produced: dict, *,
                                 bound: bool, partial: bool = False) -> dict:
    """The mechanical scientific-binding comparison (``--compare-existing``).

    Compares the produced generation against a prior on-disk generation
    that need NOT be exactly reproducible. Admits exactly: the control
    placement (slice manifest path/digest and the record-path argv inside),
    the one known pre-#852 zero-byte head phase row, and the receipt seal
    (null→bound, or unchanged). Everything
    scientific -- campaign identity (plan/prepared/parent digests, scope,
    roster), per-layer membership and extent fields (chunks, windows,
    read_set extents), ``output_space``, the adjoint source roots, the
    slice entries and their accounting, the nonzero phase schedule and the
    argv bindings -- must be identical, verified against the prior record's
    own sealed digests and through the existing canonical record validator
    before comparison. A pre-bound prior generation (records carrying
    boundary/executable readset blocks -- those bind actual reads) is
    refused outright; new bindings made in THIS run through the existing
    binders from the trusted receipt remain allowed. Returns a summary
    dict; any drift outside the admitted set raises. With ``partial`` (a
    band-granular binding, PQ #993) the produced generation may cover a
    subset of the prior quanta; every produced quantum still compares.
    """
    prior_paths = sorted(prior_dir.glob("layer-*.json"))
    if not prior_paths:
        raise ValueError(f"no prior layer-*.json records under {prior_dir}")
    prior = {}
    for path in prior_paths:
        try:
            record = json.loads(path.read_bytes().decode("utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read prior record {path}: {exc}") from exc
        qid = record.get("quantum_id") if isinstance(record, dict) else None
        if type(qid) is not str:
            raise ValueError(f"prior record {path} names no quantum id")
        if qid in prior:
            raise ValueError(f"prior record {path} duplicates quantum "
                             f"{qid!r} already read: refusing")
        # The prior record is validated through the EXISTING canonical
        # record validator before anything it says is compared: a stale
        # identity (an edited body) refuses here, so a tampered prior can
        # never be certified by the comparison. The binding is the record's
        # own campaign block (plus its own receipt when bound) -- exactly
        # the recheck the consumer performs.
        campaign = record.get("campaign")
        if not isinstance(campaign, dict):
            raise ValueError(f"compare {qid}: the prior record seals no "
                             "campaign block: refusing")
        binding = dict(campaign)
        bound_slice = (record.get("adjoint") or {}).get("slice_sha256")
        if bound_slice is not None:
            binding["adjoint_slice_sha256"] = bound_slice
        try:
            check_quantum_for_campaign(record, binding)
        except ValueError as exc:
            raise ValueError(f"compare {qid}: the prior record fails its "
                             f"own identity check: {exc}") from exc
        for block in ("boundary_readset", "executable_readset"):
            if record.get(block) is not None:
                raise ValueError(
                    f"compare {qid}: the prior generation is pre-bound "
                    f"(carries {block}); a pre-bound prior generation is "
                    f"not supported by this comparison -- re-derive from "
                    f"the primary sealed inputs instead: refusing")
        prior[qid] = record
    new_by_id = {record["quantum_id"]: record for record in produced["records"]}
    if set(prior) != set(new_by_id) and not (partial and set(new_by_id) <= set(prior)):
        missing = sorted(set(prior) - set(new_by_id))
        extra = sorted(set(new_by_id) - set(prior))
        raise ValueError(f"the produced generation does not span the prior "
                         f"generation (missing {missing}, extra {extra}): "
                         "refusing")
    moved_placement = 0
    receipt_moves = 0
    zero_head_rows = 0
    record_path_moves = 0
    for qid in sorted(new_by_id):
        where = f"compare {qid}"
        old_record = prior[qid]
        new_record = new_by_id[qid]
        old_body = {k: v for k, v in old_record.items()
                    if k not in _CONTROL_BLOCKS}
        new_body = {k: v for k, v in new_record.items()
                    if k not in _CONTROL_BLOCKS}
        if json.dumps(old_body, sort_keys=True) != json.dumps(
                new_body, sort_keys=True):
            raise ValueError(
                f"{where}: scientific fields differ (campaign identity, "
                f"membership, extents, output_space or another non-control "
                f"block): refusing")
        old_read = dict(old_record.get("read_set") or {})
        new_read = dict(new_record.get("read_set") or {})
        old_placement = (old_read.pop("manifest_path", None),
                         old_read.pop("manifest_sha256", None))
        new_placement = (new_read.pop("manifest_path", None),
                         new_read.pop("manifest_sha256", None))
        if old_read != new_read:
            raise ValueError(f"{where}: read_set differs beyond the control "
                             "manifest placement: refusing")
        if old_placement != new_placement:
            moved_placement += 1
        old_adjoint = dict(old_record.get("adjoint") or {})
        new_adjoint = dict(new_record.get("adjoint") or {})
        old_receipt = _pop_stage_a_binding(old_adjoint)
        new_receipt = _pop_stage_a_binding(new_adjoint)
        if old_adjoint != new_adjoint:
            raise ValueError(f"{where}: adjoint block differs beyond the "
                             "stage-A binding: refusing")
        if old_receipt == (None, None, None) and new_receipt != old_receipt:
            if not bound:
                raise ValueError(f"{where}: stage-A binding moved with no "
                                 "stage-A proof bound: refusing")
            receipt_moves += 1
        elif old_receipt != new_receipt:
            raise ValueError(f"{where}: prior stage-A binding is neither "
                             "retained nor newly bound: refusing")
        prior_manifest = _prior_slice_manifest(old_record, where=where)
        new_manifest = produced["slice_manifests"][qid]
        verdict = _compare_prior_slice(
            prior_manifest, new_manifest, where=where)
        if verdict["record_path_relocated"]:
            record_path_moves += 1
        prior_phases = (prior_manifest.get("annotations") or {}).get("phases")
        stripped = _strip_leading_zero_head_phase(
            prior_phases, where=where)
        if len(stripped) != len(prior_phases):
            zero_head_rows += 1
    return {"quanta": len(new_by_id),
            "windows_total": sum(len(record.get("windows", []))
                                 for record in produced["records"]),
            "scientific_fields_identical": len(new_by_id),
            "manifest_placement_moved": moved_placement,
            "record_path_relocated": record_path_moves,
            "zero_byte_head_rows_dropped": zero_head_rows,
            "receipts_newly_bound": receipt_moves}


def _retained_budget_provenance(plan: dict, *, plan_sha256: str):
    """Where the plan's retained budget comes from (PQ #1022).

    A plan that binds ``stage_b_resource_policy`` must seal exactly the
    budget that policy derived from the roster: the policy is re-derived
    (``verify_policy``; cached in-process, so the head intake does not repeat
    it) and a plan carrying any other budget refuses. A plan that binds no
    policy seals an operator-declared budget; the caller then refuses it with
    :func:`_undelivered_budget_refusal` when the roster does not fit it.
    Returns the verified policy, or ``None`` for a declared budget.
    """
    retained = (plan.get("execution") or {}).get("retained_operator_windows")
    binding = plan.get("stage_b_resource_policy")
    if binding is None or retained is None:
        return None
    from prismaquant.joint_stageb_resources import verify_policy

    policy = verify_policy(binding)
    if retained.get("budget") != policy["budget"]:
        raise ValueError(
            f"plan {plan_sha256} binds Stage B resource policy "
            f"{binding.get('sha256')} but seals a different retained budget: "
            "refusing")
    return policy


def _undelivered_budget_refusal(plan: dict, *, plan_sha256: str,
                                layer: int, exc: Exception):
    """Name the fix when a declared retained budget refuses its roster.

    Only an admission refusal (the planners' ``RuntimeError``, which
    ``derive_layer_prepared_inputs`` chains) is a budget question; any other
    refusal, and any refusal under a derived policy, passes through as is.
    """
    if (plan.get("stage_b_resource_policy") is not None
            or not isinstance(exc.__cause__, RuntimeError)):
        return exc if isinstance(exc, ValueError) else ValueError(str(exc))
    return ValueError(
        f"plan {plan_sha256} seals an operator-declared retained budget "
        f"that does not admit layer {layer}'s roster ({exc}); derive the "
        "budget from the roster with "
        "prismaquant.joint_stageb_resources.derive_policy and bind it as "
        "the plan's stage_b_resource_policy")


def _load_production_cache(prepared: dict):
    """Load the prepared completion's production weight cache, once (PQ #917).

    The ``production_cache`` binding names independently bound path/SHA256;
    the bytes are digest-verified before unpickling and the result must be
    a ``ProductionWeightCache``. The single load serves every layer's
    prepared-input derivation; render payloads are never rehashed (digests
    come from the cache's sealed verified cells, sizes from current stat).
    """
    import pickle

    from prismaquant.production_weight_cache import ProductionWeightCache

    reference = prepared.get("production_cache")
    if not isinstance(reference, dict) or set(reference) != {
            "path", "sha256"}:
        raise ValueError("the prepared completion names no bound production "
                         "cache (path/SHA256): refusing")
    from prismaquant.stage_b_prep_io import read_input
    path = Path(reference["path"])
    try:
        raw = read_input(path, sha256=reference["sha256"], where="production cache")
    except OSError as exc:
        raise ValueError(f"production cache unreadable at {path}: "
                         f"{exc}") from exc
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError(f"production cache digest mismatch at {path}")
    try:
        cache = pickle.loads(raw)
    except Exception as exc:
        raise ValueError(f"production cache does not unpickle at {path}: "
                         f"{exc}") from exc
    if not isinstance(cache, ProductionWeightCache):
        raise ValueError("the prepared production cache is not a "
                         "ProductionWeightCache: refusing")
    return cache


def _build_head_slices(plan: dict, *, plan_sha256: str, prepared: dict,
                       prepared_binding: dict, production_cache, layers,
                       output_root: str, metadata_root) -> dict:
    """Run the Stage B head intake ONCE and seal one slice per layer (PQ #1010).

    The walk is the one every quantum used to repeat -- the plan's inputs,
    existing renders required, payloads unverified, the plan's historical
    encoder allowance -- with no head journal: it runs here, in the metadata
    producer, and nothing it reads is re-read by a quantum. Returns
    ``{layer: {"bytes", "binding"}}``; nothing is written.
    """
    from prismaquant.aura_cost import _aura_source_sha256
    from prismaquant.joint_layer_quanta import qname_layer
    from prismaquant.joint_stage_b_head import (
        HEAD_SLICE_SCHEMA, build_head_slices, head_slice_bytes, head_slice_path)
    from prismaquant.tessera_joint_aura import load_measured_anchor_input
    from prismaquant.tessera_reader import load_declared_reader

    started = time.monotonic()
    data = load_measured_anchor_input(
        plan["inputs"], reader=load_declared_reader(plan.get("reader")),
        synthesis_device="cpu", progress_phase=None,
        require_existing_renders=True, verify_payloads=False,
        historical_encoder_reuse=plan.get("historical_encoder_reuse"),
        file_hash_workers=plan.get("file_hash_workers", 1))
    slices = build_head_slices(
        config=plan, plan_sha256=plan_sha256, prepared=prepared_binding,
        completion=prepared, production_cache=production_cache, data=data,
        layers=layers, layer_of=qname_layer,
        implementation_sha256=_aura_source_sha256())
    sealed = {}
    for layer, head_slice in slices.items():
        raw = head_slice_bytes(head_slice)
        sealed[layer] = {"bytes": raw, "binding": {
            "path": head_slice_path(output_root, layer, metadata_root=metadata_root),
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "schema": HEAD_SLICE_SCHEMA,
            "head_files": [dict(row) for row in head_slice["head_files"]]}}
    print(f"Stage B head slices: {len(sealed)} layers from one head intake of "
          f"{len(data.formats_by_qname)} units / {len(data.cells)} cells in "
          f"{time.monotonic() - started:.1f}s")
    return sealed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--plan-sha256", required=True)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--prepared-sha256", required=True)
    ap.add_argument("--parent-manifest", type=Path, required=True)
    ap.add_argument("--parent-manifest-sha256", required=True)
    ap.add_argument("--derivation", type=Path, required=True,
                    help="derivation.json carrying chunk_target_bytes, "
                         "stride, ram_window_gib, max_resident_consumers")
    ap.add_argument("--partition", type=Path, default=None,
                    help="optional window-partition.json")
    ap.add_argument("--output-root", type=Path, default=None,
                    help="authoritative DATA run root sealed into "
                         "output_space and adjoint.boundary_artifacts "
                         "(default: the plan's output_root)")
    ap.add_argument("--data-manifest-sha256", default=None,
                    help="PQ #1092: the digest of the data manifest this "
                         "admitted action was submitted with. Every declared "
                         "input is then read off PrismaBuild's stage through "
                         "the residency reader and checked against its "
                         "digest; a read the stage does not hold refuses, "
                         "never falling back to the pool. Needs --residency "
                         "stage. Without it reads are plain, as before")
    ap.add_argument("--allowed-tiers", default=None,
                    help="with --data-manifest-sha256: the staged tiers a "
                         "read may come from (default ram,ssd)")
    ap.add_argument("--produced-output", action="store_true",
                    help="PQ #1070: commit every file this run creates under "
                         "--metadata-root as a PrismaBuild produced output of "
                         "this admitted action. The action must declare the "
                         "write-only template tools/stage_b_preparation_"
                         "submission.py writes; refuses before the first "
                         "write otherwise")
    ap.add_argument("--metadata-root", type=Path, default=None,
                    help="PQ #884: explicit immutable control-metadata "
                         "generation root. Slice manifests land at "
                         "{metadata-root}/manifests/, the record set "
                         "defaults to {metadata-root}/records/, and bound "
                         "readsets at {metadata-root}/adjoint/"
                         "bound-readsets/ -- while output_space and "
                         "adjoint.boundary_artifacts keep naming the data "
                         "output root. Default layout unchanged when absent")
    ap.add_argument("--records-out", type=Path, default=None,
                    help="reviewed directory receiving the record files "
                         "(slice manifests land at the producer-named "
                         "absolute paths the records bind, verified after "
                         "writing; defaults to {metadata-root}/records "
                         "when --metadata-root is given)")
    ap.add_argument("--expect-existing", type=Path, default=None,
                    help="Gate 1: directory of on-disk records from the "
                         "original root; requires --original-root")
    ap.add_argument("--original-root", type=Path, default=None,
                    help="the root the --expect-existing records were "
                         "produced with: Gate 1 first reproduces them "
                         "exactly (input fidelity), then validates that the "
                         "new root moves only the authorized path fields")
    ap.add_argument("--compare-existing", type=Path, default=None,
                    help="read-only scientific-binding comparison of the "
                         "produced generation against a prior on-disk "
                         "generation that need not be exactly "
                         "reproducible (e.g. a prior producer revision's "
                         "sealed set); separate from Gate 1 -- refuse to "
                         "combine with --expect-existing")
    ap.add_argument("--adjoint-receipt", type=Path, default=None,
                    help="Gate 2: the completed stage-A adjoint-capture.json "
                         "to bind")
    ap.add_argument("--adjoint-band", type=Path, action="append", default=[],
                    help="Gate 2: one sealed checkpoint band to bind "
                         "(repeatable; PQ #993). Only the layers the bands "
                         "serve get records")
    ap.add_argument("--catalog-extension", type=Path, default=None,
                    help="immutable additive-catalog proof retaining the original Stage A capture")
    ap.add_argument("--catalog-extension-sha256", default=None)
    ap.add_argument("--boundary-readsets", action="store_true",
                    help="with Gate 2: derive each record's sealed bulk "
                         "readset manifest (PQ #848) into the new tree's "
                         "bound-readsets/ directory and bind it to a new "
                         "record generation (probe count from the sealed "
                         "plan); needs --adjoint-receipt")
    ap.add_argument("--executable-readsets", action="store_true",
                    help="with Gate 2: derive each record's single "
                         "executable read manifest (PQ #862: checkpoint, "
                         "chain and own source extents, prepared renders, and "
                         "boundary/probe/replay reads) into bound-readsets/; "
                         "derives retained rosters from the bound production "
                         "cache; needs --adjoint-receipt")
    ap.add_argument("--head-slices", action="store_true",
                    help="with --executable-readsets: run the campaign head "
                         "intake once and seal one Stage B head slice per "
                         "layer, declared in each quantum's head phase "
                         "(PQ #1010)")
    ap.add_argument("--replay-mode", choices=("windowed", "spill"),
                    default="windowed",
                    help="with --executable-readsets: the replay mode the "
                         "read plan is sealed for (PQ #1011). spill stages "
                         "the own boundary run once per probe for the "
                         "one-pass spill; the quantum refuses a launch in "
                         "the other mode. Default %(default)s")
    ap.add_argument("--replay-regime", default=None,
                    help="with --replay-mode spill: the campaign spec's "
                         "PRISMAQUANT_STAGE_B_REPLAY_REGIME (unset: the "
                         "default regime). Its capture batch sets the spill "
                         "parts each layer's sealed spill bound counts")
    ap.add_argument("--spill-block-bytes", type=int, default=None,
                    help="with --replay-mode spill: the direct-I/O grid the "
                         "sealed spill bound is sized on (default: "
                         "joint_replay_spill.SPILL_SEAL_BLOCK_BYTES). The "
                         "quantum refuses a coarser live grid")
    ap.add_argument("--source-layers-prefix", default=None,
                    help="with --executable-readsets: complete the head "
                         "phase and each chain/own source phase from the "
                         "tensor spans the streaming loader reads "
                         "(layer_streaming.streaming_source_plan, PQ #1095). "
                         "PREFIX is the loader's live decoder layers prefix "
                         "(e.g. model.language_model.layers.). Reads the "
                         "sealed plan's model config, index and shard "
                         "headers; slices, chunks and campaign identity stay "
                         "unchanged")
    ap.add_argument("--check-only", action="store_true",
                    help="Gate 1 alone; write nothing")
    args = ap.parse_args(argv)
    if args.head_slices and not args.executable_readsets:
        return _fail("--head-slices needs --executable-readsets")
    if args.replay_mode != "windowed" and not args.executable_readsets:
        return _fail("--replay-mode needs --executable-readsets")
    if args.replay_mode != "spill" and (args.replay_regime is not None
                                        or args.spill_block_bytes is not None):
        return _fail("--replay-regime and --spill-block-bytes need "
                     "--replay-mode spill")
    if args.source_layers_prefix is not None and (
            not args.executable_readsets or not args.source_layers_prefix):
        return _fail("--source-layers-prefix needs --executable-readsets "
                     "and a nonempty checkpoint reader prefix")
    if args.expect_existing is not None and args.original_root is None:
        return _fail("Gate 1 needs --original-root beside --expect-existing")
    if args.expect_existing is not None and args.compare_existing is not None:
        return _fail("--expect-existing (exact reproduction) and "
                     "--compare-existing (prior-generation scientific "
                     "comparison) are separate gates: run them in separate "
                     "invocations")
    metadata_root = None
    if args.metadata_root is not None:
        from prismaquant.joint_layer_quanta import _canonical_control_root
        try:
            metadata_root = _canonical_control_root(str(args.metadata_root))
        except ValueError as exc:
            return _fail(str(exc))
    records_out = args.records_out
    if records_out is None and metadata_root is not None:
        records_out = Path(metadata_root) / "records"
    if args.allowed_tiers is not None and args.data_manifest_sha256 is None:
        return _fail("--allowed-tiers needs --data-manifest-sha256")
    if args.data_manifest_sha256 is not None:
        # PQ #1092: strict staged reads, bound before the first input read.
        # This run's own outputs (the metadata root and the records it
        # writes) are read where they are.
        from prismaquant.stage_b_prep_io import (
            PreparationReadRefused, bind_staged_reads)
        from prismaquant.staged_tier_policy import DEFAULT_ALLOWED_TIERS
        try:
            bind_staged_reads(
                manifest_sha256=args.data_manifest_sha256,
                allowed_tiers=args.allowed_tiers or DEFAULT_ALLOWED_TIERS,
                own_outputs=[root for root in (metadata_root, records_out)
                             if root is not None])
        except PreparationReadRefused as exc:
            return _fail(str(exc))
    if records_out is None and not args.check_only:
        return _fail("--records-out is required (or pass --metadata-root "
                     "to default it, or --check-only to write nothing)")
    try:
        plan = _load_json(args.plan, digest=args.plan_sha256, where="plan")
        prepared = _load_json(args.prepared, digest=args.prepared_sha256,
                              where="prepared")
        parent = _load_json(args.parent_manifest, digest=args.parent_manifest_sha256,
                            where="parent manifest")
        derivation = _load_json(args.derivation, digest=None, where="derivation")
        partition = (_load_json(args.partition, digest=None, where="partition")
                     if args.partition is not None else None)
        output_root = (str(args.output_root) if args.output_root is not None
                       else plan.get("output_root"))
        if not isinstance(output_root, str) or not output_root.startswith("/"):
            raise ValueError("no authoritative absolute output_root: pass "
                             "--output-root or seal plan.output_root")
        if args.expect_existing is not None and metadata_root is not None:
            # A metadata relocation rides ON TOP of an exactly reproduced
            # source generation: the data output root cannot move in the
            # same run (a root move plus a relocation is a different,
            # unauthorized combination here).
            if os.path.realpath(output_root) != \
                    os.path.realpath(str(args.original_root)):
                raise ValueError(
                    "--metadata-root with --expect-existing is an authorized "
                    "relocation over an exactly reproduced source "
                    "generation: the data output root must stay the "
                    "original root (pass --output-root equal to "
                    "--original-root, or drop --metadata-root)")

        if bool(args.catalog_extension) != bool(args.catalog_extension_sha256):
            raise ValueError("catalog extension path and SHA256 must be supplied together")
        if args.catalog_extension is not None and (
                (args.adjoint_receipt is None and not args.adjoint_band)
                or args.expect_existing is not None):
            raise ValueError("catalog extension requires actual Stage A proof and a new metadata generation")
        extension = (None if args.catalog_extension is None else {
            "path": str(args.catalog_extension.resolve()), "sha256": args.catalog_extension_sha256})

        def _produce(root: str, proofs=None, metadata=None):
            return layer_quanta(
                plan, prepared, parent,
                chunk_target_bytes=derivation.get("chunk_target_bytes"),
                stride=derivation.get("stride"),
                output_root=root,
                metadata_root=metadata,
                plan_path=str(args.plan), plan_sha256=args.plan_sha256,
                prepared_path=str(args.prepared),
                prepared_sha256=args.prepared_sha256,
                parent_manifest_sha256=args.parent_manifest_sha256,
                ram_window_gib=derivation.get("ram_window_gib"),
                max_resident_consumers=derivation.get("max_resident_consumers"),
                window_partition=partition,
                adjoint_receipts=proofs, catalog_extension=extension)
    except (ValueError, OSError) as exc:
        return _fail(str(exc))
    from prismaquant.joint_adjoint_slices import (
        adjoint_slice_bytes, band_set, load_stage_a_receipt_like)
    from prismaquant.stage_b_prep_io import read_input, staged_reads

    # Off the stage only when strict: without the flag the call is as before.
    proof_read = ({} if staged_reads() is None else {"read": lambda path, sha256:
                  read_input(path, sha256=sha256, where="Stage A proof")})
    proofs: list = []
    try:
        if args.adjoint_receipt is not None:
            proofs.append(load_stage_a_receipt_like(args.adjoint_receipt,
                                                    **proof_read))
            if proofs[0].get("status") != "complete":
                raise ValueError("--adjoint-receipt is not a completed receipt")
        bands = [load_stage_a_receipt_like(path, **proof_read)
                 for path in args.adjoint_band]
        if any(band.get("status") != "band" for band in bands):
            raise ValueError("--adjoint-band names a document that is not a band")
        band_index = band_set(bands) if bands else {}
        proofs.extend(band_index.values())
    except (ValueError, OSError, RuntimeError) as exc:
        return _fail(f"stage-A proof: {exc}")
    receipt = proofs or None
    if args.expect_existing is not None:
        # Step A (input fidelity): the original root must reproduce the
        # on-disk records exactly -- canonical JSON, not raw bytes, so a
        # re-serialization the producer never promised does not fail it.
        try:
            original = _produce(str(args.original_root))
        except (ValueError, OSError) as exc:
            return _fail(f"Gate 1 reproduction: {exc}")
        for record in original["records"]:
            on_disk = args.expect_existing / f"{record['quantum_id']}.json"
            try:
                stored = json.loads(on_disk.read_bytes().decode("utf-8"))
            except (OSError, ValueError) as exc:
                return _fail(f"Gate 1 cannot read {on_disk}: {exc}")
            if json.dumps(stored, sort_keys=True) != json.dumps(
                    record, sort_keys=True):
                return _fail(f"Gate 1: {record['quantum_id']} differs from "
                              f"the sealed inputs under receipt-less "
                              f"regeneration at the original root; "
                              f"re-derive by review")
        print(f"Gate 1a: {len(original['records'])}/{len(original['records'])} "
              f"records reproduce from the sealed inputs at the original root")
        # Step B (authorized move): the new root may move only the path
        # fields (and the receipt seal when binding); everything else is
        # strictly equal, and the new identity must recompute.
        try:
            moved = _produce(output_root, proofs=receipt,
                             metadata=metadata_root)
        except (ValueError, OSError) as exc:
            return _fail(f"Gate 1 move: {exc}")
        old_by_id = {}
        for record in original["records"]:
            on_disk = args.expect_existing / f"{record['quantum_id']}.json"
            try:
                old_by_id[record["quantum_id"]] = json.loads(
                    on_disk.read_bytes().decode("utf-8"))
            except (OSError, ValueError) as exc:
                return _fail(f"Gate 1 cannot read {on_disk}: {exc}")
        try:
            if metadata_root is None:
                for record in moved["records"]:
                    _check_authorized_diff(
                        old_by_id[record["quantum_id"]], record,
                        old_root=str(args.original_root),
                        bound=receipt is not None,
                        where=f"move to {output_root}")
            else:
                for record in moved["records"]:
                    qid = record["quantum_id"]
                    # The prior slice bytes are digest-verified against
                    # the prior record before the relocation is judged.
                    _prior_slice_manifest(
                        old_by_id[qid], where=f"Gate 1 relocation {qid}")
                    produced_slice_sha = hashlib.sha256(
                        seal_manifest_bytes(
                            moved["slice_manifests"][qid])).hexdigest()
                    _check_authorized_metadata_diff(
                        old_by_id[qid], record,
                        original_root=str(args.original_root),
                        metadata_root=metadata_root,
                        slice_sha256=produced_slice_sha,
                        bound=receipt is not None,
                        where=f"relocation to {metadata_root}")
        except (ValueError, KeyError) as exc:
            return _fail(str(exc))
        if metadata_root is None:
            print(f"Gate 1b: {len(moved['records'])}/{len(moved['records'])} "
                  f"records move only the authorized fields to the new root")
        else:
            print(f"Gate 1b: {len(moved['records'])}/"
                  f"{len(moved['records'])} records relocate only the "
                  f"control metadata to {metadata_root}; data roots "
                  f"retained")
        produced = moved
    else:
        try:
            produced = _produce(output_root, proofs=receipt,
                                metadata=metadata_root)
        except (ValueError, OSError) as exc:
            return _fail(str(exc))
    bound_manifests: list = []
    if args.boundary_readsets and receipt is None:
        return _fail("--boundary-readsets needs --adjoint-receipt or --adjoint-band")
    if args.executable_readsets and receipt is None:
        return _fail("--executable-readsets needs --adjoint-receipt or --adjoint-band")
    head_slices: dict = {}
    adjoint_slices = produced.get("adjoint_slices", {})
    if receipt is not None and (
            args.boundary_readsets or args.executable_readsets):
        # Post-capture readset binding (PQ #848/#862): each record gets a
        # new generation carrying its sealed manifests. Derivation is pure
        # (no writes); the probe count comes from the sealed plan, never a
        # knob. Refusal writes nothing.
        try:
            execution = plan.get("execution", {})
            n_probes = execution.get("n_probes") \
                if isinstance(execution, dict) else None
            if type(n_probes) is not int or isinstance(n_probes, bool) \
                    or n_probes < 1:
                raise ValueError(
                    "the sealed plan names no probe count: refusing")
            layers = parent.get("annotations", {}).get("layers", [])
            checkpoints = derive_stride(
                len(layers), derivation.get("stride"))["checkpoints"]
            if args.boundary_readsets:
                # Each record binds its own slice (PQ #993): a band serves
                # only its layers, so the readset derives per record.
                emitted = [row for record in produced["records"]
                           for row in emit_quantum_boundary_readsets(
                               adjoint_slices[record["quantum_id"]], [record],
                               strided_boundaries=checkpoints, n_probes=n_probes,
                               output_root=output_root, metadata_root=metadata_root)]
                produced["records"] = [row["record"] for row in emitted]
                bound_manifests = [(row["manifest_path"], row["manifest"],
                                    row["manifest_sha256"]) for row in emitted]
            if args.executable_readsets:
                calib_input = plan.get("calibration_input", {})
                calib_path = calib_input.get("path") \
                    if isinstance(calib_input, dict) else None
                calib_sha256 = calib_input.get("sha256") \
                    if isinstance(calib_input, dict) else None
                if type(calib_path) is not str or not calib_path:
                    raise ValueError(
                        "the sealed plan names no calibration input path: "
                        "refusing")
                try:
                    calib_bytes = Path(calib_path).stat().st_size
                except OSError as exc:
                    raise ValueError(
                        f"calibration input unreadable at {calib_path}: "
                        f"{exc}") from exc
                production = prepared.get("production_cache", {})
                production_sha = production.get("sha256") \
                    if isinstance(production, dict) else None
                if not production_sha:
                    raise ValueError(
                        "the prepared completion names no production pickle "
                        "digest: refusing (render prerequisite unbound)")
                roster = produced["records"][0]["campaign"].get(
                    "unit_roster_sha256")
                if not roster:
                    raise ValueError(
                        "the record campaign seals no unit roster: refusing")
                source_model_root = plan.get("model")
                if type(source_model_root) is not str or not os.path.isabs(
                        source_model_root):
                    raise ValueError(
                        "executable readsets need the sealed plan's absolute "
                        "source model directory: refusing")
                source_spans = None
                head_source = None
                if args.source_layers_prefix is not None:
                    if sorted(layers) != list(range(len(layers))):
                        raise ValueError(
                            "source completion needs parent layers starting "
                            "at zero: refusing")
                    # PQ #1095: the streaming loader's own enumeration, for
                    # the resident head and every layer alike, under the
                    # live layers prefix the sealed unit roster names.
                    from prismaquant.layer_streaming import (
                        streaming_source_plan,
                    )
                    from prismaquant.source_read_plan import (
                        roster_layers_prefix,
                    )
                    roster_prefix = roster_layers_prefix(
                        prepared.get("formats_by_qname") or {})
                    if roster_prefix != args.source_layers_prefix:
                        raise ValueError(
                            f"--source-layers-prefix "
                            f"{args.source_layers_prefix!r} is not the live "
                            f"layers prefix {roster_prefix!r} the prepared "
                            "unit roster names: refusing")
                    source_plan = streaming_source_plan(
                        source_model_root,
                        layers_prefix=args.source_layers_prefix,
                        layers=range(len(layers)),
                        source_reads=staged_reads())
                    source_spans = source_plan["layer_spans"]
                    head_source = {
                        "layers_prefix": source_plan["layers_prefix"],
                        "tensors": source_plan["head_tensors"],
                        "spans": source_plan["head_spans"]}
                    print(f"source plan: profile {source_plan['profile']} "
                          f"(multimodal={source_plan['multimodal']}), "
                          f"{len(source_plan['head_tensors'])} resident head "
                          f"tensors in {len(source_plan['head_spans'])} spans, "
                          f"{len(source_spans)} layers")
                # PQ #917 static prepared-input bridge: the production
                # pickle loads ONCE here; each layer's prepared contract is
                # derived from its verified cells through the existing
                # retained planners (no re-prepare, re-render or payload
                # rehash). Records group by layer so every emitted row
                # carries its own layer's contract.
                production_cache = _load_production_cache(prepared)
                formats_by_qname = prepared.get("formats_by_qname")
                if not isinstance(formats_by_qname, dict) or \
                        not formats_by_qname:
                    raise ValueError(
                        "the prepared completion names no unit roster: "
                        "refusing")
                from prismaquant.joint_cost_quantum import (
                    derive_layer_prepared_inputs,
                )
                by_layer: dict[int, list] = {}
                for record in produced["records"]:
                    by_layer.setdefault(record.get("layer"), []).append(
                        record)
                # PQ #1022: every layer's retained admission is settled
                # before the head intake, so a budget the roster does not
                # fit refuses in seconds, naming its fix.
                _retained_budget_provenance(plan,
                                            plan_sha256=args.plan_sha256)
                prepared_by_layer = {}
                for layer in sorted(by_layer):
                    layer_records = by_layer[layer]
                    try:
                        prepared_by_layer[layer] = derive_layer_prepared_inputs(
                            layer_records[0],
                            execution=plan.get("execution", {}),
                            formats_by_qname=formats_by_qname,
                            production_cache=production_cache,
                            prepared_sha256=args.prepared_sha256,
                            production_pkl_sha256=production_sha,
                            unit_roster_sha256=layer_records[0][
                                "campaign"].get("unit_roster_sha256"))
                    except ValueError as exc:
                        refusal = _undelivered_budget_refusal(
                            plan, plan_sha256=args.plan_sha256,
                            layer=layer, exc=exc)
                        if refusal is None:
                            raise
                        raise refusal from exc
                # The spill bound: the layer's full-roster spill
                # geometry and its reservation, which the dispatcher sets as
                # the row's spill ceiling and PrismaBuild charges.
                spill_by_layer = {}
                if args.replay_mode == "spill":
                    from prismaquant.joint_cost_quantum import (
                        derive_layer_spill_bound,
                    )
                    from prismaquant.joint_replay_spill import (
                        SPILL_SEAL_BLOCK_BYTES,
                    )
                    from prismaquant.model_profiles import detect_profile
                    # The config is a declared header read of the source
                    # plan (``streaming_source_plan``); under strict reads
                    # it comes off the stage like the checkpoint index.
                    config_path = os.path.normpath(
                        os.path.join(source_model_root, "config.json"))
                    reads = staged_reads()
                    model_config = json.loads(
                        Path(config_path).read_bytes() if reads is None
                        else reads.whole(config_path, where="source config"))
                    try:
                        profile = detect_profile(source_model_root)
                    except RuntimeError as exc:
                        raise ValueError(f"spill bound profile: {exc}") from exc
                    block = (SPILL_SEAL_BLOCK_BYTES
                             if args.spill_block_bytes is None
                             else args.spill_block_bytes)
                    for layer in sorted(by_layer):
                        bound = derive_layer_spill_bound(
                            prepared_by_layer[layer],
                            execution=plan.get("execution", {}),
                            production_cache=production_cache,
                            profile=profile, model_config=model_config,
                            replay_regime=args.replay_regime, block=block)
                        spill_by_layer[layer] = bound
                        geometry = bound["geometry"]
                        print(f"spill bound: layer {layer} payload "
                              f"{geometry['total_bytes']} bytes, "
                              f"{geometry['max_parts']} parts, reservation "
                              f"{bound['reservation_bytes']} bytes on a "
                              f"{bound['block']}-byte grid")
                if args.head_slices:
                    head_slices = _build_head_slices(
                        plan, plan_sha256=args.plan_sha256,
                        prepared=prepared,
                        prepared_binding={"path": str(args.prepared),
                                          "sha256": args.prepared_sha256},
                        production_cache=production_cache,
                        layers=sorted(by_layer), output_root=output_root,
                        metadata_root=metadata_root)
                emitted = []
                for layer in sorted(by_layer):
                    layer_records = by_layer[layer]
                    layer_prepared = prepared_by_layer[layer]
                    print(f"prepared-input bridge: layer {layer} seals "
                          f"{len(layer_prepared['windows'])} retained "
                          f"windows over "
                          f"{sum(len(window['members']) for window in layer_prepared['windows'])} "
                          f"render members")
                    for row in emit_quantum_executable_readsets(
                            adjoint_slices[layer_records[0]["quantum_id"]],
                            layer_records, parent,
                            strided_boundaries=checkpoints,
                            n_probes=n_probes,
                            calib={"path": calib_path,
                                   "bytes": calib_bytes,
                                   "sha256": calib_sha256},
                            render_prerequisite={
                                "scope": "pb732",
                                "production_pkl_sha256": production_sha,
                                "unit_roster_sha256": roster},
                            output_root=output_root,
                            metadata_root=metadata_root,
                            layer_source_spans=source_spans,
                            source_model_root=source_model_root,
                            prepared_inputs=layer_prepared,
                            head_slice=(head_slices[layer]["binding"]
                                        if head_slices else None),
                            replay_mode=args.replay_mode,
                            head_source=head_source,
                            spill_bound=spill_by_layer.get(layer)):
                        emitted.append(row)
                produced["records"] = [row["record"] for row in emitted]
                bound_manifests.extend(
                    (row["manifest_path"], row["manifest"],
                     row["manifest_sha256"]) for row in emitted)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return _fail(f"readset binding: {exc}")
    if args.compare_existing is not None:
        # Read-only scientific-binding comparison against a prior
        # generation: the produced records/slices are in memory
        # (nothing was written yet), the prior set and its sealed slice
        # bytes come from disk and are digest-verified before any
        # semantic comparison.
        try:
            summary = _compare_existing_generation(
                args.compare_existing, produced, bound=receipt is not None,
                partial=bool(args.adjoint_band) and args.adjoint_receipt is None)
        except (ValueError, OSError, KeyError) as exc:
            return _fail(f"compare-existing: {exc}")
        print(f"Compare-existing: {summary['quanta']}/{summary['quanta']} "
              f"quanta scientifically identical to the prior generation "
              f"(campaign identity, membership, extents, output_space, "
              f"adjoint roots); windows_total={summary['windows_total']}; "
              f"manifest placement moved for "
              f"{summary['manifest_placement_moved']}, record path "
              f"relocated for {summary['record_path_relocated']}, "
              f"zero-byte head rows dropped "
              f"{summary['zero_byte_head_rows_dropped']}, receipts newly "
              f"bound {summary['receipts_newly_bound']}")
    if args.check_only:
        if metadata_root is not None or args.compare_existing is not None:
            note = []
            if metadata_root is not None:
                note.append(f"metadata root {metadata_root}")
            if args.compare_existing is not None:
                note.append("prior-generation comparison")
            print(f"regenerate_joint_quanta: check-only verified "
                  f"({'; '.join(note)}); wrote nothing")
        return 0
    out = records_out
    from prismaquant.stage_b_prep_io import (
        PreparationPublicationRefused, bind_preparation_publication, publish_files)
    try:
        # With --produced-output, inside the admitted PrismaBuild action, the
        # files are produced outputs committed at their origin (PQ #1070);
        # without it they are published directly, as before.
        publication = bind_preparation_publication(
            metadata_root, required=args.produced_output)
        # Publication order is the recoverability contract: every
        # referenced manifest is published and hash-verified BEFORE any
        # record or index names it, so no discoverable record ever points
        # at absent or different bytes. A crash between manifests and
        # records reruns to completion (same bytes are idempotent);
        # differing bytes refuse instead of overwriting.
        control = [(Path(record["read_set"]["manifest_path"]),
                    seal_manifest_bytes(produced["slice_manifests"][record["quantum_id"]]),
                    "slice manifest")
                   for record in produced["records"]]
        # Bound boundary readsets land at their own producer-named absolute
        # paths -- a new immutable generation beside the records, never an
        # edit of a sealed file.
        control += [(Path(manifest_path), seal_manifest_bytes(manifest), "bound readset")
                    for manifest_path, manifest, _ in bound_manifests]
        # Each layer's Stage B head slice (PQ #1010) lands at the path its
        # record's executable readset names, before the record exists.
        control += [(Path(sealed["binding"]["path"]), sealed["bytes"], "Stage B head slice")
                    for _, sealed in sorted(head_slices.items())]
        # Each bound record's stage-A slice (PQ #993) lands at the path the
        # record names, as its canonical bytes, before the record exists.
        control += [(Path(record["adjoint"]["slice_path"]),
                     adjoint_slice_bytes(adjoint_slices[record["quantum_id"]]),
                     "stage-A slice")
                    for record in produced["records"]
                    if record["adjoint"].get("slice_path") is not None]
        publish_files(publication, "manifests", control)
        for _, sealed in sorted(head_slices.items()):
            binding = sealed["binding"]
            if hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest() \
                    != binding["sha256"]:
                return _fail(f"head slice at {binding['path']} does not hash "
                             "to the sealed digest")
        for record in produced["records"]:
            slice_path = record["adjoint"].get("slice_path")
            if slice_path is None:
                continue
            if hashlib.sha256(Path(slice_path).read_bytes()).hexdigest() != \
                    record["adjoint"]["slice_sha256"]:
                return _fail(f"stage-A slice at {slice_path} does not hash to "
                             "the sealed digest")
        # Resolution verification: every bound path must name the exact
        # file just published, whose bytes hash to the sealed digest.
        for record in produced["records"]:
            manifest_path = Path(record["read_set"]["manifest_path"])
            try:
                sealed = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            except OSError as exc:
                return _fail(f"slice manifest unreadable at {manifest_path}: "
                              f"{exc}")
            if sealed != record["read_set"]["manifest_sha256"]:
                return _fail(f"slice manifest at {manifest_path} does not "
                              f"hash to the sealed digest")
        for manifest_path, _, manifest_sha256 in bound_manifests:
            path = Path(manifest_path)
            try:
                sealed = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                return _fail(f"bound readset unreadable at {path}: {exc}")
            if sealed != manifest_sha256:
                return _fail(f"bound readset at {path} does not hash to "
                              f"the sealed digest")
        # Records and the index publish last: nothing discoverable names
        # bytes that were not just verified above. Slice manifests land at
        # the producer-named absolute paths the records bind -- never
        # beside the record files. The adjoint manifest is the phase
        # worker's file and is never written here.
        records = [(out / f"{record['quantum_id']}.json", _pretty(record),
                    "quantum record") for record in produced["records"]]
        if args.adjoint_band and args.adjoint_receipt is None:
            # Band granularity (PQ #993): one index per band, so a later run
            # with more bands adds indexes and never rewrites one.
            for boundary, band in sorted(band_index.items()):
                layers = set(band["band"]["layers"])
                records.append((out / f"records.band-{boundary:03d}.json",
                                _pretty([record for record in produced["records"]
                                         if record["layer"] in layers]),
                                "band records index"))
        else:
            records.append((out / "records.json", _pretty(produced["records"]),
                            "records index"))
        records.append((out / "derivation.json", _pretty(produced["derivation"]),
                        "derivation"))
        publish_files(publication, "records", records)
    except (ValueError, OSError, PreparationPublicationRefused) as exc:
        return _fail(str(exc))
    bound = ("unbound (pre-stage-A)" if receipt is None else
             f"bound to their stage-A slices ({len(proofs)} proof(s): "
             f"{'completed receipt' if args.adjoint_receipt is not None else ''}"
             f"{' + ' if args.adjoint_receipt is not None and args.adjoint_band else ''}"
             f"{'bands ' + ','.join(str(b) for b in sorted(band_index)) if args.adjoint_band else ''})")
    if bound_manifests:
        bound += f" with {len(bound_manifests)} bound readsets"
    placement = (f"; control metadata under {metadata_root}"
                 if metadata_root is not None else "")
    print(f"regenerate_joint_quanta: wrote {len(produced['records'])} records "
          f"{bound} under {out}{placement}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
