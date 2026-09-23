"""Retire a superseded Stage A run's checkpoints and their pinned entries (PQ #1073).

Since #1036/#1037 a Stage A checkpoint names the owner's own cotangent entries
at its boundary instead of copying them, so a checkpoint pins them: roll
retirement leaves them on the pool, and each run attempt leaves its whole set
of checkpoint planes behind. A chain resume (#1001) keeps and reads them. A
fresh restart does not; it supersedes the run, and nothing retired the pins.

This module is that retirement. It runs as a PrismaBuild action, in three
steps, and every check comes before the first unlink:

1. **Proof.** The run is superseded, cannot resume, and nothing live names it:

   * a successor run is named under a pinned digest (its ``chain-state.json``
     or its ``adjoint-capture.json``) and its run identity differs from this
     run's. The same identity would be a resume of this run, not a successor;
   * every PrismaBuild owner that wrote the run (the chain state's
     ``producer`` and each resume record's) is contained
     (``stage_a_chain_resume.require_producer_contained``). A run no owner
     wrote must not be ``running``;
   * no document under the declared binding roots names a path inside the
     run's adjoint space, or the digest of its chain state, its receipt or any
     of its checkpoints. Seed specs, band receipts and Stage B metadata all
     name a checkpoint by path and digest. The roots are declared, never
     discovered: the tool cannot know every document in the fleet, so the
     operator names where live consumers keep theirs, and at least one root
     is required.

2. **Plan.** From PrismaBuild's own records (each owner's filed instance, its
   ``commitments.json`` and its batch records), every committed batch that is
   not reclaimed is classified against what the run's checkpoints hold: the
   referenced cotangent entries and the files under ``checkpoints/``. A batch
   entirely inside that set is retired. A batch with no path in it (forward
   boundary entries, for one) is left alone and reported. A batch that mixes
   the two refuses, named: part of it is live by this tool's own account. Each
   file a retired batch names must still be the file its commit recorded
   (the batch's ``origin_identity``); a changed file refuses.

3. **Apply.** The retirement record (``stage-a-retired.json``) is sealed into
   the space first. From then on a chain resume, a seed and the band tool
   refuse the space (:func:`refuse_retired_space`). Then ``checkpoints/`` is
   removed, the referenced entries are unlinked, and ``reclaim_origin`` is
   called for each retired batch, so PrismaBuild stops charging it on its own
   evidence that every path is gone. A rerun after a crash reads the sealed
   record and finishes: a path already gone is not a refusal.

The queue root comes from each producer record, so the tool needs no
produced-output binding of its own. It reads two private PrismaBuild helpers
(``_load_batch_record``, ``_read_commitments``) because PrismaBuild publishes
no reader for a staged batch's descriptors; the batch record is the only
place the committed paths and identities live.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from .cost_stage_checkpoint import canonical_json, canonical_json_sha256, publish_new_bytes

RETIREMENT_SCHEMA = "prismaquant.stage_a.retirement.v1"
RETIREMENT_NAME = "stage-a-retired.json"
#: The largest binding document the scan reads. PrismaBuild's own data
#: manifest bound; a larger JSON file is no binding this campaign writes.
BINDING_MAX_BYTES = 64 * 1024 * 1024
_SHA256_LEN = 64


class RetirementRefused(RuntimeError):
    """The run cannot be retired; nothing was removed."""


def retirement_record_path(space) -> Path:
    return Path(space) / RETIREMENT_NAME


def refuse_retired_space(space, error=RuntimeError, *, what: str) -> None:
    """Raise ``error`` when ``space`` carries a retirement record.

    Written before the first unlink, so a space mid-retirement refuses too.
    """
    path = retirement_record_path(space)
    if path.exists():
        raise error(f"{path} retires this Stage A run: {what}")


# --------------------------------------------------------------------------
# Reading the run
# --------------------------------------------------------------------------

def _read_json(path: Path):
    return json.loads(Path(path).read_bytes())


def _session_of(document, where) -> dict:
    try:
        session = document["boundary_storage"]["session"]
        generation = str(session["generation"])
        identity = str(session["run_identity_sha256"])
    except (KeyError, TypeError) as exc:
        raise RetirementRefused(f"{where} names no boundary session: {exc}") from exc
    if len(identity) != _SHA256_LEN:
        raise RetirementRefused(f"{where} names no run identity digest")
    return {"generation": generation, "run_identity_sha256": identity}


def _load_chain_state(space: Path) -> tuple[dict, str]:
    from .stage_a_chain_resume import CHAIN_STATE_SCHEMA, _seal, chain_state_path

    path = chain_state_path(space)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise RetirementRefused(
            f"{path} is missing: without the run's chain state there is no "
            "record of the owners that wrote it, and no PrismaBuild batch to "
            "reclaim can be found") from None
    document = json.loads(raw)
    if (not isinstance(document, dict) or document.get("schema") != CHAIN_STATE_SCHEMA
            or _seal(document)["chain_state_sha256"] != document.get("chain_state_sha256")):
        raise RetirementRefused(f"{path} is not a sealed chain state")
    return document, hashlib.sha256(raw).hexdigest()


def _producers(space: Path, document) -> list[dict]:
    from .stage_a_chain_resume import ChainResumeRefused, resume_records

    session = document["boundary_storage"]["session"]
    try:
        records = resume_records(space, session)
    except ChainResumeRefused as exc:
        raise RetirementRefused(str(exc)) from exc
    producers = []
    for producer in [document.get("producer"), *(r.get("producer") for r in records)]:
        if producer is not None and producer not in producers:
            producers.append(producer)
    return producers


def _successor(path, sha256, *, space: Path, session: dict) -> dict:
    from .joint_adjoint_checkpoints import adjoint_space  # noqa: F401  (layout owner)

    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RetirementRefused(f"the successor run is not readable at {path}: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != str(sha256):
        raise RetirementRefused(f"{path} does not have the pinned digest {sha256}")
    real = Path(os.path.realpath(path))
    if real.is_relative_to(os.path.realpath(space)):
        raise RetirementRefused("the successor lies inside the run it would supersede")
    successor = _session_of(json.loads(raw), f"the successor {path}")
    if successor["run_identity_sha256"] == session["run_identity_sha256"]:
        raise RetirementRefused(
            "the successor carries this run's own identity: that is a resume of "
            "the run, which keeps its checkpoints, not a run that supersedes it")
    if successor["generation"] == session["generation"]:
        raise RetirementRefused("the successor is this run's own generation")
    return {"path": str(path), "sha256": str(sha256), **successor}


def _checkpoint_planes(space: Path, session: dict) -> tuple[list[str], dict]:
    """``(referenced entry paths, {boundary: checkpoint.json sha256})``."""
    from .joint_adjoint_band import BandRefused, read_sealed_checkpoint
    from .joint_adjoint_checkpoints import occupied_checkpoint_directories
    from .joint_adjoint_slices import checkpoint_cotangent_plane, checkpoint_is_referenced

    marker = {"generation": session["generation"], "kind": "adjoint_checkpoint",
              "run_identity_sha256": session["run_identity_sha256"]}
    paths: set[str] = set()
    digests: dict[int, str] = {}
    for directory in occupied_checkpoint_directories(space):
        manifest = directory / "checkpoint.json"
        if not manifest.is_file():
            continue  # A partial checkpoint names nothing; its directory goes.
        boundary = int(directory.name.removeprefix("boundary-"))
        try:
            record, _binding = read_sealed_checkpoint(space, boundary)
        except (BandRefused, OSError, ValueError, KeyError) as exc:
            raise RetirementRefused(f"checkpoint {boundary} is not sealed: {exc}") from exc
        if record["session"] != marker:
            raise RetirementRefused(
                f"checkpoint {boundary} was sealed by another run "
                f"({record['session'].get('generation')})")
        digests[boundary] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if checkpoint_is_referenced(record):
            try:
                plane = checkpoint_cotangent_plane(record)
            except ValueError as exc:
                raise RetirementRefused(f"checkpoint {boundary}: {exc}") from exc
            paths.update(os.path.normpath(str(row["path"])) for row in plane.values())
    return sorted(paths), digests


# --------------------------------------------------------------------------
# The binding scan
# --------------------------------------------------------------------------

def _strings(node, pointer=""):
    if isinstance(node, str):
        yield pointer, node
    elif isinstance(node, Mapping):
        for key, value in node.items():
            yield from _strings(value, f"{pointer}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _strings(value, f"{pointer}/{index}")


def _binding_documents(roots: Iterable, *, skip: Path):
    for root in roots:
        root = Path(root)
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = []
            for directory, subdirs, files in os.walk(root):
                here = Path(directory)
                if here == skip or here.is_relative_to(skip):
                    subdirs[:] = []
                    continue
                subdirs[:] = [name for name in subdirs
                              if not (here / name).is_relative_to(skip)]
                candidates += [here / name for name in files
                               if name.endswith((".json", ".json.gz"))]
        else:
            raise RetirementRefused(f"the binding root {root} does not exist")
        for path in sorted(candidates):
            if path.is_symlink() or path.stat().st_size > BINDING_MAX_BYTES:
                continue
            raw = path.read_bytes()
            if path.name.endswith(".gz"):
                try:
                    raw = gzip.decompress(raw)
                except OSError:
                    continue
            try:
                yield path, json.loads(raw)
            except ValueError:
                continue


def check_bindings(roots, *, space: Path, digests: Mapping[str, str]) -> int:
    """Refuse when a document under ``roots`` names the run; return how many were read.

    ``digests`` maps each of the run's sealed documents' sha256 to what it is.
    """
    roots = list(roots)
    if not roots:
        raise RetirementRefused(
            "no binding root was declared: the proof that nothing live names the "
            "run is only as wide as the roots it reads, and zero roots prove nothing")
    real_space = os.path.realpath(space)
    prefix = real_space.rstrip("/") + "/"
    spellings = {str(Path(space)).rstrip("/") + "/", prefix}
    read = 0
    for path, document in _binding_documents(roots, skip=Path(real_space)):
        read += 1
        for pointer, value in _strings(document):
            if value in digests:
                raise RetirementRefused(
                    f"{path} names this run's {digests[value]} by digest at "
                    f"{pointer or '/'}: a live binding holds the run")
            if value.rstrip("/") + "/" in spellings or any(
                    value.startswith(spelling) for spelling in spellings):
                raise RetirementRefused(
                    f"{path} names {value} inside this run at {pointer or '/'}: "
                    "a live binding holds the run")
    return read


# --------------------------------------------------------------------------
# PrismaBuild's records
# --------------------------------------------------------------------------

def _sdk():
    from .joint_forward_resume import _sdk as forward_sdk
    return forward_sdk()


def producer_instances(producer: Mapping) -> list[tuple]:
    """``[(queue, instance, template), ...]`` PrismaBuild filed for one owner attempt."""
    sdk = _sdk()
    po, pool = sdk["produced_output"], sdk["pool"]
    root = Path(producer["queue_root"])
    attempt = producer["owner_attempt"]
    if not isinstance(attempt, Mapping) or "nonce" not in attempt:
        raise RetirementRefused(
            f"the producer {str(producer.get('owner_action_key'))[:12]} names no "
            "attempt nonce")
    owner_dir = root / "residency" / po.OUTPUT_SCOPES_SUBDIR / str(producer["owner_action_key"])
    queue = pool.PoolQueue(root)
    found = []
    for directory in sorted(owner_dir.glob(f"*.{attempt['nonce']}")):
        try:
            instance = po.validate_instance(_read_json(directory / "instance.json"))
            template = po.validate_template(_read_json(
                root / "residency" / po.OUTPUT_TEMPLATES_SUBDIR
                / f"{instance['template_id']}.json"))
        except (OSError, ValueError, po.ProducedOutputError) as exc:
            raise RetirementRefused(f"{directory} is not a filed instance: {exc}") from exc
        if instance["template_sha256"] != po.template_sha256(template):
            raise RetirementRefused(f"{directory} is bound to another template")
        found.append((queue, instance, template))
    return found


def _batches(queue, instance, template) -> list[dict]:
    po = _sdk()["produced_output"]
    try:
        commitments = po._read_commitments(po._commitments_path(queue.root, instance))
    except po.ProducedOutputError as exc:
        raise RetirementRefused(f"unreadable commitments: {exc}") from exc
    rows = []
    for batch_id, entry in sorted(commitments["batches"].items()):
        classes = dict(entry.get("class_bytes") or {})
        if entry.get("origin_reclaimed"):
            continue
        try:
            filed, sealed = po._load_batch_record(queue.root, instance, template, entry, batch_id)
        except po.ProducedOutputError as exc:
            raise RetirementRefused(f"batch {batch_id}: {exc}") from exc
        rows.append({"batch_id": batch_id, "filed": filed,
                     "paths": sorted(os.path.normpath(str(d["path"])) for d in sealed),
                     "bytes": sum(int(classes.get(k) or 0)
                                  for k in ("payload", "checkpoint", "temp"))})
    return rows


def durable_charge(queue, instance) -> int:
    """Committed origin bytes PrismaBuild still charges this instance."""
    po = _sdk()["produced_output"]
    commitments = po._read_commitments(po._commitments_path(queue.root, instance))
    return sum(int((entry.get("class_bytes") or {}).get(name) or 0)
               for entry in commitments["batches"].values()
               if not entry.get("origin_reclaimed")
               for name in ("payload", "checkpoint", "temp"))


def _identity_holds(filed: Mapping, path: str) -> bool:
    """Is ``path`` absent, or still the file the batch's commit recorded?"""
    sdk = _sdk()
    recorded = (filed.get("origin_identity") or {}).get(path)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return True
    if not isinstance(recorded, Mapping):
        return False
    live = sdk["produced_output"]._portable_identity_of(info)
    return bool(sdk["reader_lease"].file_id_matches(recorded, live))


# --------------------------------------------------------------------------
# Plan and apply
# --------------------------------------------------------------------------

@dataclass
class Retirement:
    space: Path
    record: dict
    entry_paths: list
    batches: list = field(default_factory=list)       # [(queue, instance, template, row)]
    untouched: list = field(default_factory=list)     # [{"batch_id", "bytes", "owner"}]
    bindings_read: int = 0


def _files_under(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {os.path.normpath(str(path)) for path in directory.rglob("*")
            if path.is_file() or path.is_symlink()}


def plan_retirement(output_root, *, successor_path, successor_sha256,
                    binding_roots) -> Retirement:
    """Check everything and classify every batch. Writes nothing."""
    from .joint_adjoint_checkpoints import adjoint_receipt_path, adjoint_space
    from .stage_a_chain_resume import ChainResumeRefused, require_producer_contained

    space = adjoint_space(output_root)
    document, chain_sha = _load_chain_state(space)
    session = _session_of(document, "the chain state")
    successor = _successor(successor_path, successor_sha256, space=space, session=session)
    producers = _producers(space, document)
    for producer in producers:
        try:
            require_producer_contained(producer)
        except ChainResumeRefused as exc:
            raise RetirementRefused(str(exc)) from exc
    if not producers:
        status_path = (Path(document["boundary_storage"]["directory"])
                       / session["generation"] / "generation.json")
        try:
            status = _read_json(status_path).get("status")
        except (OSError, ValueError):
            status = None
        if status not in ("complete", "failed"):
            raise RetirementRefused(
                f"no PrismaBuild owner wrote this run and its generation status is "
                f"{status!r}: nothing proves it will not write again")

    sealed_path = retirement_record_path(space)
    if sealed_path.exists():
        # A rerun finishes the retirement its record seals.
        record = _read_json(sealed_path)
        body = {k: v for k, v in record.items() if k != "record_sha256"}
        if (record.get("schema") != RETIREMENT_SCHEMA
                or record.get("record_sha256") != canonical_json_sha256(body, where="retirement")
                or record.get("session") != session):
            raise RetirementRefused(f"{sealed_path} is not this run's sealed retirement")
        entry_paths = list(record["entry_paths"])
        checkpoint_files = list(record["checkpoint_files"])
        digests = dict(record["digests"])
    else:
        entry_paths, checkpoint_digests = _checkpoint_planes(space, session)
        checkpoint_files = sorted(_files_under(space / "checkpoints"))
        digests = {chain_sha: "chain state"}
        receipt = adjoint_receipt_path(space)
        if receipt.is_file():
            digests[hashlib.sha256(receipt.read_bytes()).hexdigest()] = "receipt"
        for boundary, digest in checkpoint_digests.items():
            digests[digest] = f"checkpoint {boundary}"
        record = None

    read = check_bindings(binding_roots, space=space, digests=digests)
    # The record's list, not the directory: on a rerun the directory is gone.
    retired_set = set(entry_paths) | set(checkpoint_files)
    plan_batches, untouched = [], []
    for producer in producers:
        for queue, instance, template in producer_instances(producer):
            for row in _batches(queue, instance, template):
                inside = [p for p in row["paths"] if p in retired_set]
                if not inside:
                    untouched.append({"batch_id": row["batch_id"], "bytes": row["bytes"],
                                      "owner": instance["owner_action_key"]})
                    continue
                if len(inside) != len(row["paths"]):
                    outside = sorted(set(row["paths"]) - set(inside))
                    raise RetirementRefused(
                        f"batch {row['batch_id']} mixes checkpoint entries with "
                        f"{len(outside)} path(s) no checkpoint names ({outside[0]}): "
                        "part of it is not this retirement's")
                changed = [p for p in row["paths"] if not _identity_holds(row["filed"], p)]
                if changed:
                    raise RetirementRefused(
                        f"batch {row['batch_id']}: {changed[0]} is not the file its "
                        "commit recorded")
                plan_batches.append((queue, instance, template, row))

    if record is None:
        body = canonical_json({
            "schema": RETIREMENT_SCHEMA,
            "session": session,
            "chain_state_sha256": chain_sha,
            "superseded_by": successor,
            "producers": producers,
            "entry_paths": entry_paths,
            "checkpoint_files": checkpoint_files,
            "digests": digests,
            "batches": sorted(row["batch_id"] for *_, row in plan_batches),
            "binding_roots": [str(Path(root)) for root in binding_roots],
        }, where="Stage A retirement")
        record = {**body, "record_sha256": canonical_json_sha256(body, where="retirement")}
    return Retirement(space=space, record=record, entry_paths=entry_paths,
                      batches=plan_batches, untouched=untouched, bindings_read=read)


def apply_retirement(plan: Retirement) -> dict:
    """Seal the record, remove the files, reclaim each batch. Returns a report."""
    path = retirement_record_path(plan.space)
    payload = (json.dumps(plan.record, sort_keys=True, indent=1, allow_nan=False) + "\n").encode()
    if not publish_new_bytes(path, payload) and path.read_bytes() != payload:
        raise RetirementRefused(f"{path} already seals another retirement")
    po = _sdk()["produced_output"]
    charged_before = {}
    for queue, instance, _template, _row in plan.batches:
        key = (str(queue.root), instance["owner_action_key"], instance["template_id"])
        charged_before.setdefault(key, durable_charge(queue, instance))

    checkpoints = plan.space / "checkpoints"
    if checkpoints.exists():
        shutil.rmtree(checkpoints)
    removed_bytes = removed = 0
    for entry in plan.entry_paths:
        try:
            size = os.lstat(entry).st_size
        except FileNotFoundError:
            continue
        os.unlink(entry)
        removed += 1
        removed_bytes += size

    outcomes = {}
    for queue, instance, template, row in plan.batches:
        out = dict(po.reclaim_origin(queue, instance, template, batch_id=row["batch_id"]))
        outcomes[row["batch_id"]] = "reclaimed" if out.get("ok") else str(out.get("refusal"))
    charged_after = {}
    for queue, instance, _template, _row in plan.batches:
        key = (str(queue.root), instance["owner_action_key"], instance["template_id"])
        charged_after.setdefault(key, durable_charge(queue, instance))
    refused = {batch: state for batch, state in outcomes.items() if state != "reclaimed"}
    report = {
        "schema": "prismaquant.stage_a.retirement_report.v1",
        "record": str(path), "record_sha256": plan.record["record_sha256"],
        "entries_removed": removed, "entry_bytes_removed": removed_bytes,
        "batches": outcomes,
        "durable_charge_before": sum(charged_before.values()),
        "durable_charge_after": sum(charged_after.values()),
        "untouched_batches": plan.untouched,
        "bindings_read": plan.bindings_read,
    }
    if refused:
        raise RetirementRefused(
            f"PrismaBuild kept {len(refused)} batch(es) charged: {refused}; report: "
            + json.dumps(report, sort_keys=True))
    return report


__all__ = [
    "RETIREMENT_NAME", "RETIREMENT_SCHEMA", "Retirement", "RetirementRefused",
    "apply_retirement", "check_bindings", "durable_charge", "plan_retirement",
    "producer_instances", "refuse_retired_space", "retirement_record_path",
]
