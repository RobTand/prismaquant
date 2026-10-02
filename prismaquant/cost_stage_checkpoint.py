"""Identity-bound, durable per-unit journals for streamed cost stages.

The journal deliberately follows the production weight-cache pair-shard
contract: the manifest binds the complete run identity, shard names are a
SHA-256 of the semantic unit qname (never a list position), and every shard
is an atomically published, checksummed envelope.  Existing but unverifiable
state is an error; it is never silently overwritten or recomputed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
import pickle
from pathlib import Path
import socket
import uuid

# The canonical JSON encoding moved to ``digests`` (PQ #1301); these names stay
# importable from here, where ~100 call sites import them.
from .digests import (  # noqa: F401 -- re-exported: one spelling
    canonical_json,
    canonical_json_bytes,
    canonical_json_sha256,
    canonical_json_sha256_normalized,
)
from .digests import bytes_sha256hex, text_sha256hex


MANIFEST_SCHEMA = "prismaquant.cost_stage_checkpoint.manifest.v1"
UNIT_SCHEMA = "prismaquant.cost_stage_checkpoint.unit.v1"
MAX_UNIT_IO_WORKERS = 16


_TEMP_SUFFIX: "tuple[int, str] | None" = None


def unique_temp_suffix() -> str:
    """The staging suffix one process appends to a file it is publishing.

    A fixed ``.tmp`` makes the staging path a function of the destination
    alone, so two processes publishing the same cell write the same inode and
    each ``os.replace`` can publish the other's half-written bytes.  The
    suffix below is unique per (host, pid), which is what a PB fan-out needs:
    rows own disjoint units, and a row that is retried or overlaps another
    stages somewhere nobody else writes.

    It adds **exactly one** dot, and that is load-bearing rather than
    cosmetic.  ``torch.save`` names the zip archive inside the file after the
    basename minus its last extension, so staging ``<name>.pt`` at
    ``<name>.pt.tmpsparky123`` keeps the archive name ``<name>.pt`` and the
    published bytes identical to a direct save, while ``<name>.pt.tmp.123``
    -- or a hostname carrying a dot -- would silently change them.
    """
    global _TEMP_SUFFIX
    # Keyed by pid, not merely memoised: a forked child inherits the parent's
    # module state, and a suffix carried across the fork would put the two
    # writers back on one staging path -- this function's own defect, one
    # level down.
    pid = os.getpid()
    if _TEMP_SUFFIX is None or _TEMP_SUFFIX[0] != pid:
        host = "".join(c for c in socket.gethostname() if c.isalnum())
        suffix = f".tmp{host}{pid}"
        if suffix.count(".") != 1:
            raise ValueError("staging suffix must add exactly one extension")
        _TEMP_SUFFIX = (pid, suffix)
    return _TEMP_SUFFIX[1]


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Publish bytes durably; a crash leaves either the old or new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + unique_temp_suffix())
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _publication_parent_nofollow(parent: Path) -> int:
    """Create/open the publication parent without following any component."""
    if not parent.is_absolute():
        raise ValueError("no-follow publication needs an absolute destination")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open("/", flags)
    try:
        for component in parent.parts[1:]:
            try:
                os.mkdir(component, dir_fd=fd)
            except FileExistsError:
                pass
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def publish_new_bytes(path: Path, payload: bytes, *, nofollow: bool = False) -> bool:
    """Publish ``payload`` at ``path`` as a NEW file; never replace one already there.

    ``os.link`` is the atomic no-clobber publication on one filesystem: the
    destination appears with the complete staged inode, or the call fails with
    ``EEXIST``.  There is no check-then-open window and no visible empty
    placeholder, which is what a content-addressed path needs -- two writers can
    reach one name, and :func:`atomic_write_bytes` publishes by replacement,
    which is the wrong shape there.  The campaign writers and
    ``export_output_safety.transactional_export_file`` make the same argument.

    With ``nofollow=True``, every parent is opened relative to its admitted
    directory descriptor with ``O_NOFOLLOW``. Staging, publication, cleanup and
    directory fsync all use that same descriptor, so a renamed/replaced parent
    cannot redirect a write. The default keeps the existing path behavior.

    Returns ``True`` when this call created the file and ``False`` when one was
    already there.  ``False`` is neither success nor failure: it means somebody
    else's bytes are at ``path``, and the caller owns the question of whether
    they are the bytes it wanted.  A caller that reads ``False`` as success is
    trusting the name -- the defect this shape exists to make visible.
    """
    directory_fd = None
    temporary = None
    owns_temporary = False
    try:
        if nofollow:
            directory_fd = _publication_parent_nofollow(path.parent)
            target = path.name
            try:
                os.stat(target, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                return False
            # Separate stages are needed even for concurrent threads in one PID.
            temporary = target + unique_temp_suffix() + uuid.uuid4().hex
            stage_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                               0o666, dir_fd=directory_fd)
            owns_temporary = True
            try:
                handle = os.fdopen(stage_fd, "wb")
            except BaseException:
                os.close(stage_fd)
                raise
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() or path.is_symlink():
                return False
            target = path
            temporary = path.with_name(path.name + unique_temp_suffix())
            owns_temporary = True
            handle = temporary.open("wb")
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            if directory_fd is None:
                os.link(temporary, target)
            else:
                os.link(temporary, target, src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd, follow_symlinks=False)
        except FileExistsError:
            return False
        if directory_fd is None:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        os.fsync(directory_fd)
        return True
    finally:
        try:
            if owns_temporary:
                try:
                    if nofollow:
                        os.unlink(temporary, dir_fd=directory_fd)
                    else:
                        temporary.unlink()
                except FileNotFoundError:
                    pass
        finally:
            if directory_fd is not None:
                os.close(directory_fd)


#: Record fields that name one row's own seals; a merged record drops them.
MIGRATION_ROW_FIELDS = frozenset({"old_identity_sha256", "new_identity_sha256", "shards",
                                  "receipt_seals", "cost_seals", "run_id"})


def merge_identity_migrations(per_source: Mapping[str, object], *,
                              error: type[Exception] = ValueError) -> "list | None":
    """The union of ``identity_migration`` records from several checkpoints, or None.

    A re-sealed checkpoint (tools/reseal_campaign_identity.py) carries the pins
    it was priced under, the pins it now carries, and the proof that licensed
    the change.  A merge or union rebuilds its manifest and payload from fixed
    keys, so without this the record would end there and the output would
    show only its new pins with nothing saying they were amended.  Records are
    deduplicated on the proof bundle and the pin pair, so sources migrated
    under one proof contribute one record (the first in source-name order);
    the fields that differ between such sources (clock, strata, tool commit)
    stay in each source's own evidence.  A source without the key contributes
    nothing: both callers refuse sources whose pins differ, so an unmigrated
    source cannot sit beside a migrated one.
    """
    merged: list = []
    seen = set()
    present = False
    for name in sorted(per_source):
        records = per_source[name]
        if records is None:
            continue
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise error(f"{name}: identity_migration is not a list of records")
        present = True
        for record in records:
            key = (record.get("proof_bundle_sha256"),
                   json.dumps(record.get("old_pins"), sort_keys=True),
                   json.dumps(record.get("new_pins"), sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            merged.append({k: v for k, v in record.items() if k not in MIGRATION_ROW_FIELDS})
    return merged if present else None


def unit_path(root: Path, qname: str) -> Path:
    return root / "units" / f"{text_sha256hex(str(qname))}.pkl"


def _mismatch(
    stage: str,
    *,
    field: str,
    stored: object,
    expected: object,
) -> None:
    from prismaquant.production_weight_cache import identity_value_for_error

    raise RuntimeError(
        f"{stage} checkpoint identity mismatch at {field}: "
        f"stored={identity_value_for_error(stored)} "
        f"current={identity_value_for_error(expected)}; refusing reuse or "
        "recompute"
    )


def write_unit(
    root: Path,
    *,
    stage: str,
    qname: str,
    identity_sha256: str,
    state: Mapping[str, object],
) -> None:
    state_bytes = pickle.dumps(dict(state), protocol=pickle.HIGHEST_PROTOCOL)
    envelope = {
        "schema": UNIT_SCHEMA,
        "stage": str(stage),
        "qname": str(qname),
        "identity_sha256": str(identity_sha256),
        "payload_sha256": bytes_sha256hex(state_bytes),
        "payload": state_bytes,
    }
    atomic_write_bytes(
        unit_path(root, qname),
        pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL),
    )


def _load_unit(
    path: Path,
    *,
    stage: str,
    qname: str,
    identity_sha256: str,
) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            envelope = pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(
            f"{stage} unit checkpoint {path} is corrupt for {qname}; "
            "refusing reuse or recompute"
        ) from exc
    if not isinstance(envelope, Mapping):
        raise RuntimeError(
            f"{stage} unit checkpoint {path} is not an envelope for {qname}; "
            "refusing reuse or recompute"
        )
    for field, expected in (
        ("schema", UNIT_SCHEMA),
        ("stage", str(stage)),
        ("qname", str(qname)),
        ("identity_sha256", str(identity_sha256)),
    ):
        if envelope.get(field) != expected:
            _mismatch(
                stage,
                field=f"unit[{qname}].{field}",
                stored=envelope.get(field),
                expected=expected,
            )
    payload = envelope.get("payload")
    if not isinstance(payload, bytes):
        raise RuntimeError(
            f"{stage} unit checkpoint {path} has no byte payload for {qname}; "
            "refusing reuse or recompute"
        )
    digest = bytes_sha256hex(payload)
    if envelope.get("payload_sha256") != digest:
        raise RuntimeError(
            f"{stage} unit checkpoint {path} payload_sha256 differs for "
            f"{qname}; refusing reuse or recompute"
        )
    try:
        state = pickle.loads(payload)
    except Exception as exc:
        raise RuntimeError(
            f"{stage} unit checkpoint {path} state is corrupt for {qname}; "
            "refusing reuse or recompute"
        ) from exc
    if not isinstance(state, Mapping):
        raise RuntimeError(
            f"{stage} unit checkpoint {path} state is not an object for "
            f"{qname}; refusing reuse or recompute"
        )
    return dict(state)


def _drive_ordered_units(roster, walk, commit, *, workers):
    """Walk units, committing strictly in the roster's one deterministic order.

    ``workers <= 1`` is the serial path: today's loop, no pool. Above it the
    per-unit walks overlap while the committer still banks and reports each
    unit only after every unit before it has committed, so the durable state
    is always a roster prefix -- the prefix a resume can re-verify -- and the
    progress sequence means exactly what it meant serially. A failing unit
    stops the walk with its prefix committed, precisely where the serial
    loop would have stopped.
    """
    if workers <= 1:
        for name in roster:
            commit(name, walk(name))
        return
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    # A bounded window: the pool is never starved, and results for units the
    # committer has not reached cannot pile up ahead of it.
    window = 2 * workers
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="joint-head-walk") as pool:
        inflight = deque()
        try:
            for name in roster:
                while len(inflight) >= window:
                    done_name, done = inflight.popleft()
                    commit(done_name, done.result())
                inflight.append((name, pool.submit(walk, name)))
            while inflight:
                done_name, done = inflight.popleft()
                commit(done_name, done.result())
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise


def stored_manifest_identity(manifest_path: str | Path) -> Mapping[str, object] | None:
    """The identity a journal manifest stores, or ``None`` when there is none to read.

    For a caller that must take some fields of its expected identity from the
    manifest itself -- receipts it can only re-derive one entry at a time --
    before ``prepare_journal`` compares every other field. ``None`` is not a
    verdict: ``prepare_journal`` refuses a missing, unreadable or malformed
    manifest by field.
    """
    try:
        manifest = json.loads(Path(manifest_path).read_text())
    except (OSError, ValueError):
        return None
    identity = manifest.get("identity") if isinstance(manifest, Mapping) else None
    return identity if isinstance(identity, Mapping) else None


def prepare_journal(
    checkpoint_dir: str | Path,
    *,
    stage: str,
    resume: bool,
    identity: Mapping[str, object],
    qnames: Sequence[str],
    manifest_path: str | Path | None = None,
    unit_workers: int = 1,
    unit_io_workers: int | None = None,
) -> tuple[Path, str, dict[str, dict[str, object]]]:
    """Create/validate a journal and return all exact completed unit states.

    File-oriented callers can retain their explicit manifest pathname while
    placing unit shards in ``checkpoint_dir``. The same manifest/unit schemas
    and refusal rules apply; directory-oriented callers keep ``manifest.json``.
    ``unit_workers`` optionally overlaps independent envelope reads within
    the assigned CPU affinity. The existing bounded ordered driver preserves
    roster order and joins reads before a corrupt journal can be set aside.
    ``unit_io_workers`` instead selects 1-16 I/O threads independently of that
    core count; every thread inherits the same assigned CPU mask. Combining
    the two worker policies refuses. Other callers remain serial by default.
    """
    if type(unit_workers) is not int or unit_workers < 1:
        raise ValueError("journal unit_workers must be a positive integer")
    try:
        assigned = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        assigned = 1
    if unit_workers > max(1, assigned):
        raise ValueError("journal unit_workers exceed the PB-assigned CPU affinity")
    workers = unit_workers
    if unit_io_workers is not None:
        if unit_workers != 1:
            raise ValueError("journal unit_workers and unit_io_workers cannot be combined")
        if type(unit_io_workers) is not int or not 0 < unit_io_workers <= MAX_UNIT_IO_WORKERS:
            raise ValueError(f"journal unit_io_workers must be an integer in 1:{MAX_UNIT_IO_WORKERS}")
        workers = unit_io_workers
    root = Path(checkpoint_dir)
    if root.exists() and not root.is_dir():
        raise RuntimeError(f"{stage} checkpoint path is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    canonical_identity = canonical_json(identity, where=f"{stage} identity")
    identity_sha256 = canonical_json_sha256(
        canonical_identity, where=f"{stage} identity"
    )
    manifest_path = Path(manifest_path) if manifest_path is not None else root / "manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise RuntimeError(
                f"{stage} checkpoint manifest already exists at "
                f"{manifest_path}; pass --resume to validate and reuse it"
            )
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            _mismatch(
                stage,
                field="manifest_json",
                stored="<invalid>",
                expected="<valid canonical JSON>",
            )
            raise AssertionError("unreachable") from exc
        if not isinstance(manifest, Mapping):
            _mismatch(stage, field="manifest", stored=manifest, expected="<object>")
        if manifest.get("schema") != MANIFEST_SCHEMA:
            _mismatch(
                stage,
                field="manifest.schema",
                stored=manifest.get("schema"),
                expected=MANIFEST_SCHEMA,
            )
        if manifest.get("stage") != str(stage):
            _mismatch(
                stage,
                field="manifest.stage",
                stored=manifest.get("stage"),
                expected=str(stage),
            )
        # Same digest and same canonical identity (a C-level compare) is the
        # accepted case, and the common one on a resume; only a difference
        # pays for the field walk that names where it is.
        if (manifest.get("identity_sha256") != identity_sha256
                or manifest.get("identity") != canonical_identity):
            from prismaquant.production_weight_cache import first_identity_difference

            difference = first_identity_difference(
                manifest.get("identity"), canonical_identity
            )
            if difference is not None:
                field, stored, expected = difference
                _mismatch(stage, field=field, stored=stored, expected=expected)
        if manifest.get("identity_sha256") != identity_sha256:
            _mismatch(
                stage,
                field="manifest.identity_sha256",
                stored=manifest.get("identity_sha256"),
                expected=identity_sha256,
            )
    else:
        existing = sorted((root / "units").glob("*.pkl"))
        if existing:
            raise RuntimeError(
                f"{stage} checkpoint units exist without a manifest; "
                f"refusing name-gated reuse or recompute. sample={existing[:8]}"
            )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "stage": str(stage),
            "identity_sha256": identity_sha256,
            "identity": canonical_identity,
            "units": [
                {
                    "qname": str(qname),
                    "file": str(unit_path(root, qname).relative_to(root)),
                }
                for qname in qnames
            ],
        }
        atomic_write_bytes(
            manifest_path,
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
        )

    expected_paths = {unit_path(root, qname): str(qname) for qname in qnames}
    unexpected = sorted(
        path for path in (root / "units").glob("*.pkl")
        if path not in expected_paths
    )
    if unexpected:
        _mismatch(
            stage,
            field="units.unexpected",
            stored=[path.name for path in unexpected[:8]],
            expected=[],
        )
    completed: dict[str, dict[str, object]] = {}

    def read_unit(path):
        if path.is_file():
            return _load_unit(path, stage=stage, qname=expected_paths[path],
                              identity_sha256=identity_sha256)
        return None

    def retain_unit(path, state):
        if state is not None:
            completed[expected_paths[path]] = state

    _drive_ordered_units(expected_paths, read_unit, retain_unit, workers=workers)
    return root, identity_sha256, completed
