"""Retire exact owned files after native containment and whole binding proof.

An immutable record seals every deletion target before the first unlink.
Interrupted application rechecks all current barriers and finishes that roster;
missing targets are permitted only after the record exists. Replacements retain.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import stat
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from .cost_stage_checkpoint import canonical_json, canonical_json_sha256, publish_new_bytes
from .digests import bytes_sha256hex
from .schemas import strict_json_loads
from .staged_manifest import read_document, open_regular, open_directory, unlink_owned
from .file_identity import file_stat_signature
from .stage_a_produced_output import read_produced_selection, MAX_ENTRIES, MAX_GROUPS

RETIREMENT_SCHEMA = 'prismaquant.stage_a.retirement.v2'
RETIREMENT_NAME = 'stage-a-retired.json'
BINDING_MAX_BYTES = 64 << 20
MAX_BINDING_DOCUMENTS = 65536
MAX_BINDING_ROOTS = 64
MAX_BINDING_AGGREGATE_BYTES = 256 << 20
_SHA256_LEN = 64


class RetirementRefused(RuntimeError):
    pass


def retirement_record_path(space):
    return Path(space) / RETIREMENT_NAME


def refuse_retired_space(space, error=RuntimeError, *, what):
    path = retirement_record_path(space)
    if os.path.lexists(path):
        raise error(f'{path} retires this Stage A run: {what}')


def decode_retirement_document(raw):
    def invalid(key):
        return RetirementRefused(f'invalid retirement JSON: {key}')
    return strict_json_loads(raw, duplicate=invalid, constant=invalid)


def _read_json(path):
    return decode_retirement_document(read_document(path, BINDING_MAX_BYTES)[0])


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
        raw = read_document(path, BINDING_MAX_BYTES)[0]
    except FileNotFoundError:
        raise RetirementRefused(
            f"{path} is missing: without the run's chain state there is no "
            "record of the native owners that wrote it; no publication to "
            "reclaim can be found") from None
    document = decode_retirement_document(raw)
    if (not isinstance(document, dict) or document.get("schema") != CHAIN_STATE_SCHEMA
            or _seal(document)["chain_state_sha256"] != document.get("chain_state_sha256")):
        raise RetirementRefused(f"{path} is not a sealed chain state")
    return document, bytes_sha256hex(raw)


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
        raw = read_document(path, BINDING_MAX_BYTES)[0]
    except OSError as exc:
        raise RetirementRefused(f"the successor run is not readable at {path}: {exc}") from exc
    if bytes_sha256hex(raw) != str(sha256):
        raise RetirementRefused(f"{path} does not have the pinned digest {sha256}")
    real = Path(os.path.realpath(path))
    if real.is_relative_to(os.path.realpath(space)):
        raise RetirementRefused("the successor lies inside the run it would supersede")
    successor = _session_of(decode_retirement_document(raw), f"the successor {path}")
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
            continue  # A partial checkpoint names no referenced tensor plane.
        boundary = int(directory.name.removeprefix("boundary-"))
        try:
            record, _binding = read_sealed_checkpoint(space, boundary)
        except (BandRefused, OSError, ValueError, KeyError) as exc:
            raise RetirementRefused(f"checkpoint {boundary} is not sealed: {exc}") from exc
        if record["session"] != marker:
            raise RetirementRefused(
                f"checkpoint {boundary} was sealed by another run "
                f"({record['session'].get('generation')})")
        digests[boundary] = bytes_sha256hex(read_document(manifest, BINDING_MAX_BYTES)[0])
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
    def unreadable(error: OSError):
        raise RetirementRefused(
            f"the binding scan cannot read {error.filename}: {error}") from error

    roots = list(roots)
    if len(roots) > MAX_BINDING_ROOTS:
        raise RetirementRefused('binding root roster exceeds finite bound')
    candidates = set()
    visited = 0
    for root in roots:
        root = Path(root)
        if not root.is_absolute() or root.resolve() != root:
            raise RetirementRefused(f"the binding root {root} is a symlink")
        if root.is_file():
            candidates.add(root)
        elif root.is_dir():
            descriptor = open_directory(str(root))
            os.close(descriptor)
            for directory, subdirs, files in os.walk(root, onerror=unreadable):
                here = Path(directory)
                visited += 1 + len(subdirs) + len(files)
                if visited > MAX_BINDING_DOCUMENTS * 4 or len(here.relative_to(root).parts) > 64:
                    raise RetirementRefused('binding tree exceeds finite entry/depth bound')
                if here == skip or here.is_relative_to(skip):
                    subdirs[:] = []
                    continue
                kept = []
                for name in subdirs:
                    child = here / name
                    if child.is_relative_to(skip):
                        continue
                    if child.is_symlink():
                        raise RetirementRefused(
                            f"the binding directory {child} is a symlink")
                    kept.append(name)
                subdirs[:] = kept
                candidates.update(here / name for name in files
                                  if name.endswith((".json", ".json.gz")))
                if len(candidates) > MAX_BINDING_DOCUMENTS:
                    raise RetirementRefused('binding document roster exceeds finite bound')
        else:
            raise RetirementRefused(f"the binding root {root} does not exist")
    if len(candidates) > MAX_BINDING_DOCUMENTS:
        raise RetirementRefused('binding document roster exceeds finite bound')
    aggregate = 0
    for path in sorted(candidates):
            if path.is_symlink():
                raise RetirementRefused(f"the binding document {path} is a symlink")
            try:
                metadata = path.stat()
                if not stat.S_ISREG(metadata.st_mode):
                    raise RetirementRefused(
                        f"the binding document {path} is not a regular file")
                if metadata.st_size > BINDING_MAX_BYTES:
                    raise RetirementRefused(
                        f"the binding document {path} exceeds {BINDING_MAX_BYTES} bytes")
                raw = read_document(path, BINDING_MAX_BYTES)[0]
                if path.name.endswith('.gz'):
                    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as source:
                        raw = source.read(BINDING_MAX_BYTES + 1)
                if len(raw) > BINDING_MAX_BYTES:
                    raise RetirementRefused(
                        f"the binding document {path} exceeds {BINDING_MAX_BYTES} "
                        "decoded bytes")
                aggregate += len(raw)
                if aggregate > MAX_BINDING_AGGREGATE_BYTES:
                    raise RetirementRefused('binding documents exceed finite aggregate bound')
                document = strict_json_loads(
                    raw, duplicate=lambda key: ValueError(
                        f"duplicate JSON member {key!r}"),
                    constant=lambda name: ValueError(
                        f"non-standard JSON constant {name!r}"))
            except (OSError, EOFError, ValueError, zlib.error) as error:
                raise RetirementRefused(
                    f"the binding document {path} cannot be read: {error}") from error
            yield path, document


def _binding_names_space(value: str, spellings: set[str]) -> bool:
    """Match literal paths conservatively, and absolute aliases by real path."""
    candidates = (value, os.path.realpath(value)) if os.path.isabs(value) else (value,)
    return any(candidate.rstrip("/") + "/" in spellings or any(
        candidate.startswith(spelling) for spelling in spellings)
        for candidate in candidates)


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
        if read > MAX_BINDING_DOCUMENTS:
            raise RetirementRefused("binding roster exceeds finite bound")
        for pointer, value in _strings(document):
            if value in digests:
                raise RetirementRefused(
                    f"{path} names this run's {digests[value]} by digest at "
                    f"{pointer or '/'}: a live binding holds the run")
            if _binding_names_space(value, spellings):
                raise RetirementRefused(
                    f"{path} names {value} inside this run at {pointer or '/'}: "
                    "a live binding holds the run")
    return read


def _owned_forward_paths(document: Mapping, *, space: Path, session: Mapping) -> set[str]:
    """Sealed forward candidates owned by this run, inside its binding proof."""
    from .joint_adjoint_checkpoints import reference_from_record
    from .perturbed_x_cache import EXACT_ACTIVATION_SCHEMA

    rows = document.get("boundary_entries", {})
    if not isinstance(rows, Mapping):
        raise RetirementRefused("the chain state's boundary entries are not a mapping")
    owner_session = document["boundary_storage"]["session"]
    directory = Path(document["boundary_storage"]["directory"]) / session["generation"] / "entries"
    real_directory, real_space = Path(os.path.realpath(directory)), Path(os.path.realpath(space))
    paths = set()
    for entries in rows.values():
        if not isinstance(entries, list):
            raise RetirementRefused("the chain state's forward entries are not a list")
        for entry in entries:
            try:
                metadata = entry["metadata"]
                identity = metadata["identity"]
                _session_of({"boundary_storage": {"session": identity["session"]}},
                            "a sealed forward entry")
                if identity["session"] != owner_session:
                    continue  # Imported forward inputs are not this producer's files.
                reference = reference_from_record(entry)
                if (metadata["schema"] != EXACT_ACTIVATION_SCHEMA
                        or identity["kind"] != "boundary"
                        or metadata["shape"] != list(reference.shape)
                        or metadata["dtype"] != reference.dtype
                        or metadata["tensor_bytes"] != reference.tensor_bytes
                        or reference.file_bytes <= 0
                        or len(reference.sha256) != _SHA256_LEN):
                    raise RetirementRefused("an owned forward entry has inconsistent exact metadata")
                path = Path(os.path.normpath(reference.path))
                real = Path(os.path.realpath(path))
                if (not path.is_absolute()
                        or not path.is_relative_to(os.path.normpath(directory))
                        or not real.is_relative_to(real_directory)
                        or not real.is_relative_to(real_space)):
                    raise RetirementRefused(
                        f"owned forward entry {path} escapes this run's generation or binding proof")
                paths.add(str(path))
            except (KeyError, TypeError, ValueError) as exc:
                raise RetirementRefused(f"malformed sealed forward entry: {exc}") from exc
    return paths



@dataclass
class Retirement:
    space: Path
    record: dict
    entry_paths: list
    batches: list = field(default_factory=list)
    untouched: list = field(default_factory=list)
    bindings_read: int = 0


def _producer_key(producer):
    from .flux_jobs import job_id_number
    return (job_id_number(producer['job_id']), producer['record_dir'], producer['launch_sha256'])


def _identity_holds(row, *, missing):
    try:
        fd = open_regular(row['path'])
    except FileNotFoundError:
        if missing:
            return
        raise RetirementRefused('unsealed retirement target is absent') from None
    try:
        if list(file_stat_signature(os.fstat(fd))) != row['file_identity']:
            raise RetirementRefused(f"retirement target changed: {row['path']}")
    finally:
        os.close(fd)


def _selected(producers, selections, *, missing):
    if not isinstance(selections, (list, tuple)) or not 0 < len(selections) <= 16:
        raise RetirementRefused('explicit finite native producer selections required')
    expected, seen, groups = {_producer_key(p) for p in producers}, set(), []
    count = 0
    for row in selections:
        if not isinstance(row, dict) or set(row) != {'selection', 'checkpoint_exit'}:
            raise RetirementRefused('exact native selection/checkpoint contract required')
        verified = read_produced_selection(row['selection'], checkpoint_exit=row['checkpoint_exit'],
                                          verify_payloads=not missing)
        key = _producer_key(verified.index['producer'])
        if key not in expected or key in seen:
            raise RetirementRefused('selected producer differs from recorded run owners')
        seen.add(key)
        groups.extend(verified.groups)
        count += sum(len(g['entries']) for g in verified.groups)
        if len(groups) > MAX_GROUPS or count > MAX_ENTRIES:
            raise RetirementRefused('aggregate retirement roster exceeds bound')
    if seen != expected:
        raise RetirementRefused('selection roster does not cover every recorded producer')
    return groups


def plan_retirement(output_root, *, successor_path, successor_sha256,
                    binding_roots, producer_selections=()):
    from .joint_adjoint_checkpoints import adjoint_receipt_path, adjoint_space
    from .stage_a_chain_resume import require_producer_contained
    space = adjoint_space(output_root)
    document, chain_sha = _load_chain_state(space)
    session = _session_of(document, 'chain state')
    successor = _successor(successor_path, successor_sha256, space=space, session=session)
    producers = _producers(space, document)
    if not producers:
        raise RetirementRefused('no authenticated native producer recorded for this run')
    for producer in producers:
        require_producer_contained(producer)
    sealed_path = retirement_record_path(space)
    sealed = _read_json(sealed_path) if os.path.lexists(sealed_path) else None
    if sealed is not None:
        body = {k: v for k, v in sealed.items() if k != 'record_sha256'}
        if (sealed.get('schema') != RETIREMENT_SCHEMA or
                sealed.get('record_sha256') != canonical_json_sha256(body, where='retirement') or
                sealed.get('session') != session or sealed.get('chain_state_sha256') != chain_sha or
                sealed.get('superseded_by') != successor or sealed.get('producers') != producers or
                sealed.get('producer_selections') != list(producer_selections) or
                sealed.get('binding_roots') != [str(Path(p)) for p in binding_roots]):
            raise RetirementRefused('sealed retirement no longer matches all current barriers')
        digests = sealed['digests']
    else:
        _entries, checkpoint_digests = _checkpoint_planes(space, session)
        digests = {chain_sha: 'chain state', **{v: f'checkpoint {k}' for k, v in checkpoint_digests.items()}}
        receipt = adjoint_receipt_path(space)
        if receipt.exists():
            digests[bytes_sha256hex(read_document(receipt, BINDING_MAX_BYTES)[0])] = 'receipt'
    read = check_bindings(binding_roots, space=space, digests=digests)
    groups = _selected(producers, producer_selections, missing=sealed is not None)
    targets, batches, untouched = {}, [], []
    if sealed is None:
        entry_paths, _ = _checkpoint_planes(space, session)
        eligible = set(entry_paths) | _owned_forward_paths(document, space=space, session=session)
        for group in groups:
            if group['session'] != session:
                raise RetirementRefused('selected batch session differs from retired run')
            rows = group['entries']
            inside = [r for r in rows if r['path'] in eligible or
                      Path(r['path']).is_relative_to(space / 'checkpoints')]
            if not inside:
                untouched.append(group['batch_id'])
                continue
            if len(inside) != len(rows):
                raise RetirementRefused('batch mixes eligible and unrelated paths')
            batches.append(group['batch_id'])
            for row in rows:
                target = {k: row[k] for k in ('path', 'sha256', 'bytes', 'file_identity')}
                if row['path'] in targets:
                    raise RetirementRefused('duplicate selected retirement target')
                targets[row['path']] = target
        # Unrecorded files in checkpoints are retained, never swept.
        if not targets:
            raise RetirementRefused('no selected native files eligible for retirement')
        body = canonical_json({'schema': RETIREMENT_SCHEMA, 'session': session,
            'chain_state_sha256': chain_sha, 'superseded_by': successor, 'producers': producers,
            'producer_selections': list(producer_selections), 'digests': digests,
            'binding_roots': [str(Path(p)) for p in binding_roots],
            'targets': [targets[p] for p in sorted(targets)], 'batches': batches,
            'untouched': untouched}, where='Stage A retirement')
        sealed = {**body, 'record_sha256': canonical_json_sha256(body, where='retirement')}
    else:
        available = {r['path']: r for g in groups for r in g['entries']}
        if not isinstance(sealed['targets'], list) or not 0 < len(sealed['targets']) <= MAX_ENTRIES:
            raise RetirementRefused('finite sealed target roster required')
        for row in sealed['targets']:
            if row != {k: available.get(row['path'], {}).get(k)
                       for k in ('path', 'sha256', 'bytes', 'file_identity')}:
                raise RetirementRefused('sealed target differs from immutable selected descriptor')
    for row in sealed['targets']:
        _identity_holds(row, missing=os.path.lexists(sealed_path))
    return Retirement(space, sealed, [r['path'] for r in sealed['targets']],
                      sealed['batches'], sealed['untouched'], read)


def apply_retirement(plan):
    # Recollect stopped scopes, selected authority, successor and all current
    # consumers immediately before sealing and every apply, including restart.
    from .joint_adjoint_checkpoints import adjoint_space
    output_root = plan.space.parent.parent
    if adjoint_space(output_root) != plan.space:
        raise RetirementRefused('retirement space is not the canonical adjoint space')
    record = plan.record
    fresh = plan_retirement(output_root,
        successor_path=record['superseded_by']['path'], successor_sha256=record['superseded_by']['sha256'],
        binding_roots=record['binding_roots'], producer_selections=record['producer_selections'])
    if fresh.record != record:
        raise RetirementRefused('retirement changed between planning and apply')
    path = retirement_record_path(plan.space)
    payload = (json.dumps(record, sort_keys=True, indent=1, allow_nan=False) + '\n').encode()
    if not publish_new_bytes(path, payload) and read_document(path, BINDING_MAX_BYTES)[0] != payload:
        raise RetirementRefused('another retirement already sealed this space')
    for row in record['targets']:
        _identity_holds(row, missing=True)
    removed, nbytes = 0, 0
    for row in record['targets']:
        _identity_holds(row, missing=True)
        if not unlink_owned(row['path'], row['file_identity'], missing_ok=True):
            continue
        removed += 1
        nbytes += row['bytes']
    return {'schema': 'prismaquant.stage_a.retirement_report.v2',
        'record': str(path), 'record_sha256': record['record_sha256'],
        'entries_removed': removed, 'entry_bytes_removed': nbytes,
        'batches': record['batches'], 'untouched_batches': record['untouched'],
        'bindings_read': fresh.bindings_read}
