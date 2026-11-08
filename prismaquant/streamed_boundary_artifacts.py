"""Exact activation boundaries and their produced-output lifecycle."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import dataclasses
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat
import time
from typing import Any

import torch

from prismaquant.dev_mode import seal_check
from .digests import bytes_sha256hex
from .joint_stageb_resources import cotangent_scratch


@dataclass
class StreamedForwardBoundaries:
    """One exact source-model forward, cut at decoder-layer boundaries."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    position_embeddings: object
    attention_mask: object
    # Explicit artifact mode stores ExactActivationReference receipts here;
    # all legacy callers continue to receive their original CPU tensors.
    activations_cpu: list[Any]
    shared_pass_state: object


@dataclass(frozen=True)
class ProducedFileReference:
    """One small file an owner wrote as a produced-output group member.

    The fields the produced-output spool reads from a writer reference
    (``path``, ``name``, ``file_bytes``, ``sha256``). It is not an exact
    entry: it has no tensor identity and no durable-progress coordinate.
    """

    path: str
    name: str
    file_bytes: int
    sha256: str


def _write_new_file(path: Path, payload: bytes) -> None:
    """Write ``payload`` to a file that must not exist yet, then fsync it."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


def _link_new_file(path: Path, payload: bytes) -> None:
    """Publish a new file through its planned ``.tmp`` name; never replace one."""
    temporary = Path(str(path) + ".tmp")
    _write_new_file(temporary, payload)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


BOUNDARY_STORAGE_SCHEMA = "prismaquant.aura.boundary_storage.v1"
LAYER_MAJOR_BOUNDARY_STORAGE_SCHEMA = "prismaquant.aura.boundary_storage.v2"


def check_boundary_storage(config):
    """Validate the explicit exact-artifact policy without touching storage.

    Returns the policy unchanged (``None`` for none). The directory is not
    resolved, so no path component is stat'ed: a caller that only admits the
    policy, before its readset is bound, uses this (PQ #1024).
    """
    if config is None:
        return None
    fields = {"schema", "directory", "max_resident_bytes", "max_auxiliary_bytes",
              "max_artifact_bytes", "prefetch_batches"}
    if not isinstance(config, dict):
        raise ValueError("exact boundary storage requires a complete versioned policy")
    if config.get("schema") == LAYER_MAJOR_BOUNDARY_STORAGE_SCHEMA:
        fields.add("capture_order")
        if config.get("capture_order") != "layer_major":
            raise ValueError("exact boundary storage v2 requires layer_major capture_order")
    elif config.get("schema") != BOUNDARY_STORAGE_SCHEMA:
        raise ValueError("exact boundary storage requires a known policy schema")
    if set(config) != fields:
        raise ValueError("exact boundary storage requires a complete closed policy")
    for key in fields - {"schema", "directory", "capture_order"}:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"exact boundary storage requires positive {key}")
    if not isinstance(config["directory"], str) or not config["directory"].strip():
        raise ValueError("exact boundary storage requires an artifact directory")
    return config


#: The boundary storage policy fields that fix how the stored entries are laid
#: out and grouped: the schema, the capture order, and the read window that
#: keys every published group (``_produced_group_key`` divides the batch index
#: by it). A reader of the generation depends on them, so a resume under other
#: values refuses in both modes (PQ #1147). The byte ceilings
#: (``max_resident_bytes``, ``max_auxiliary_bytes``, ``max_artifact_bytes``)
#: bound one run's own memory and disk and are run seals; the relaunch's guards
#: enforce its own values either way.
BOUNDARY_STORAGE_LAYOUT_FIELDS = ("schema", "capture_order", "prefetch_batches")


def boundary_storage_layout_differs(recorded, running) -> bool:
    """True when two boundary storage policies lay out stored entries differently.

    That is, when their field sets differ or any of
    ``BOUNDARY_STORAGE_LAYOUT_FIELDS`` differs. A policy that differs only in
    its byte ceilings lays the entries out the same way.
    """

    if not isinstance(recorded, dict) or not isinstance(running, dict):
        return recorded != running
    if set(recorded) != set(running):
        return True
    return any(recorded.get(name) != running.get(name)
               for name in BOUNDARY_STORAGE_LAYOUT_FIELDS)


def normalize_boundary_storage(config):
    """The checked policy with its directory resolved.

    Resolving stats every component of the directory path; callers that
    only need the admission check use :func:`check_boundary_storage`.
    """
    config = check_boundary_storage(config)
    if config is None:
        return None
    return {**config, "directory": str(Path(config["directory"]).resolve())}


BOUNDARY_PARTITION_RANGE_SCHEMA = "prismaquant.boundary_partition_range.v1"


def plan_boundary_partition_ranges(*, n_partitions, n_ranges):
    """Split P calibration partitions into N contiguous disjoint ranges.

    A joint AURA boundary capture is data-parallel over calibration
    partitions: each entry is written under its own exact ``(batch,
    boundary)`` coordinates as an independent file, and no range reads
    another's (PQ #738). This function only *names* the quanta -- it
    schedules nothing, owns no bytes and builds no dispatcher. Execution,
    placement and balancing stay in flux_jobs: one campaign row per range
    produces that range's entries, and :func:`verify_boundary_partition_coverage`
    checks the union before anything downstream reads it as one capture.
    """
    if type(n_partitions) is not int or n_partitions <= 0:
        raise ValueError("boundary partition ranges require a positive partition count")
    if type(n_ranges) is not int or not 1 <= n_ranges <= n_partitions:
        raise ValueError("boundary partition ranges require 1 <= n_ranges <= n_partitions")
    base, extra = divmod(n_partitions, n_ranges)
    ranges, start = [], 0
    for index in range(n_ranges):
        width = base + (1 if index < extra else 0)
        ranges.append({
            "schema": BOUNDARY_PARTITION_RANGE_SCHEMA,
            "range_index": index,
            "n_ranges": n_ranges,
            "n_partitions": n_partitions,
            "partition_start": start,
            "partition_end": start + width,
            "partitions": width,
        })
        start += width
    return ranges


def verify_boundary_partition_coverage(ranges, *, n_partitions):
    """Refuse a range set whose union is not exactly ``[0, n_partitions)``.

    Gaps would silently drop calibration partitions from the merged capture;
    overlaps would let two rows publish one entry. Both refuse here, before
    the merge, rather than inside it.
    """
    if type(n_partitions) is not int or n_partitions <= 0:
        raise ValueError("boundary partition coverage requires a positive partition count")
    items = list(ranges)
    if not items:
        raise ValueError("boundary partition coverage requires at least one range")
    seen = set()
    for entry in items:
        if not isinstance(entry, dict) or entry.get("schema") != BOUNDARY_PARTITION_RANGE_SCHEMA:
            raise ValueError("boundary partition range has an unknown schema")
        for key in ("range_index", "n_ranges", "n_partitions",
                    "partition_start", "partition_end", "partitions"):
            if key not in entry:
                raise ValueError(f"boundary partition range is missing {key}")
        if entry["n_partitions"] != n_partitions or entry["n_ranges"] != len(items):
            raise ValueError("boundary partition range disagrees with its coverage set")
        start, end = entry["partition_start"], entry["partition_end"]
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= n_partitions):
            raise ValueError("boundary partition range has out-of-scope coordinates")
        if end - start != entry["partitions"]:
            raise ValueError("boundary partition range width disagrees with its coordinates")
        if entry["range_index"] in seen:
            raise ValueError("boundary partition range index repeats")
        seen.add(entry["range_index"])
    covered = sorted((entry["partition_start"], entry["partition_end"]) for entry in items)
    cursor = 0
    for start, end in covered:
        if start != cursor:
            raise ValueError("boundary partition ranges leave a gap or overlap")
        cursor = end
    if cursor != n_partitions:
        raise ValueError("boundary partition ranges under-cover the partitions")
    return sorted(items, key=lambda entry: entry["range_index"])


def _checked_owner_label(label):
    """An owner label is a plain file name: ``owners/<label>.json``."""
    if type(label) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label):
        raise RuntimeError(f"an owner label is a plain file name, not {label!r}")
    return label


def _state_tensors(value):
    """Closed source-metadata grammar: opaque tensor owners must refuse."""
    from collections.abc import Mapping
    if isinstance(value, torch.Tensor):
        if value.is_meta or value.layout != torch.strided:
            raise TypeError("exact boundary storage cannot account this state tensor")
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _state_tensors(key)
            yield from _state_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _state_tensors(item)
    elif value is not None and type(value) not in (str, bool, int, float, complex):
        raise TypeError(f"exact boundary storage cannot account opaque state {type(value).__name__}")


def _state_storage_bytes(values):
    storages = {}
    for tensor in _state_tensors(values):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        storages[key] = storage.nbytes()
    return sum(storages.values())


class StreamedBoundaryArtifacts:
    """Own exact-boundary receipts and generations, never a second tensor cache.

    Tensor writing, page release and each bounded resident window are delegated
    to the existing activation artifact owner in ``perturbed_x_cache``. Working
    generations are deliberately not checkpoint inputs: an interrupted cost
    resume recaptures into a fresh generation, as the legacy producer already
    does. Only completed signed cost shards are resumable measurement state.
    """

    def __init__(self, config):
        self.config = normalize_boundary_storage(config)
        self.identity = {key: value for key, value in self.config.items() if key != "directory"}
        self.session = None
        self.directory = None
        self._readonly = False
        self._published = False
        self._references = {}
        self._slots = {}
        self._forward_inputs = {}
        # Chain resume (RobTand/prismaquant#1001): the sealed checkpoint
        # cotangents a resumed chain starts from, each mapped to the
        # checkpoint boundary it was sealed at. They are borrowed inputs
        # like ``_forward_inputs`` (and are in it), but they hold no
        # boundary slot: a checkpoint entry is not a rolling cotangent.
        self._checkpoint_inputs = {}
        #: Own entries a committed referenced checkpoint names (PQ #1036),
        #: keyed to its boundary. Retiring one drops it from the live set
        #: and moves its bytes to the checkpoint ledger; its file stays.
        self._pinned_checkpoint_entries = {}
        self._resumed = False
        #: A chain split quantum (PQ #738) is one of several owners of one
        #: generation, each over its own sample range. Its status goes to
        #: ``owners/<label>.json`` and never to ``generation.json``, which
        #: stays the run's: one owner's clean exit does not say the run is
        #: complete while another owner still rolls.
        self._owner_label = None
        self._owner_fields = {}
        self._attached_forward_inputs = frozenset()
        self._active_window = None
        self._check_memory = None
        self._progress = None
        self._n_probes = 0
        self._batches = None
        self._cotangents = None
        self._status = "unused"
        self._scratch = None
        self._cotangent_scratch = None
        self._checkpoint_reservations = {}
        self._checkpoint_committed = {}
        self._checkpoint_active = None
        self._next_checkpoint_reservation = 1
        self._transient_hold_bytes = 0
        self._produced = None
        self._produced_plan = None
        self._local_output_spool = None
        self._produced_groups = {}
        self._produced_origin_batches = []
        self._produced_release_errors = []
        self._produced_index = {}
        self._deferred_unlinks = {}
        self._produced_retained_uncommitted = []
        self._produced_retained_boundary = None
        self._produced_retained_reads = frozenset()
        self._produced_retained_read_context = None
        from .produced_stager import OwnerLock
        self._produced_lock = OwnerLock()
        self.telemetry = {"resident_tensor_bytes": 0, "peak_resident_tensor_bytes": 0,
            "peak_auxiliary_bytes": 0, "peak_shared_cotangent_reservation_bytes": 0,
            "live_artifact_bytes": 0, "peak_artifact_bytes": 0,
            "live_checkpoint_bytes": 0, "peak_checkpoint_bytes": 0,
            "peak_transient_serialization_bytes": 0,
            "checkpoint_reservations": 0, "checkpoint_refusals": 0,
            "checkpoint_envelope_unused_bytes": 0,
            "written_tensor_bytes": 0, "read_tensor_bytes": 0,
            "written_entries": 0, "retired_entries": 0, "prefetch_windows": 0,
            "hot_read_misses": 0,
            "produced_groups_prewritten": 0,
            "produced_file_groups": 0,
            "produced_groups_committed_at_origin": 0,
            "produced_groups_origin_reclaimed": 0,
            "produced_local_reads": 0, "produced_local_read_bytes": 0,
            "produced_local_windows": 0,
            "produced_deferred_unlinks": 0,
            "produced_deferred_unlinks_done": 0,
            "produced_deferred_unlink_bytes": 0}

    def __enter__(self):
        return self

    def bind(self, identity, *, n_probes, check_memory=None, published=False,
             owner_label=None):
        """Start one generation; ``published`` keeps its entries on exit.

        ``owner_label`` (a forward split prep, PQ #738) mints the generation
        for a split round's owners: ``generation.json`` is written once at
        ``running`` and never again by this owner, whose own status goes to
        ``owners/<owner_label>.json``. The quanta then rebind the generation,
        and the run stays unfinished until its chain completes.

        A working generation (the single run's) is deliberately disposable:
        closing it retires every entry, because only completed cost shards
        are resumable state. A **published** generation (the distributed
        campaign's adjoint capture, contract §3.3) hands its entries to a
        sealed receipt instead: they outlive this owner for the layer quanta
        to read back, so closing it must not unlink them. The caller owns
        deliberate retirements either way.
        """
        import uuid
        from .cost_stage_checkpoint import canonical_json_sha256
        if self.session is not None:
            raise RuntimeError("exact boundary generation is already bound")
        self._n_probes = n_probes
        self._check_memory = check_memory
        self._published = bool(published)
        self.session = {"generation": uuid.uuid4().hex,
            "run_identity_sha256": canonical_json_sha256(identity, where="exact boundary source")}
        self.directory = Path(self.config["directory"]) / self.session["generation"]
        self.directory.mkdir(parents=True, exist_ok=False)
        self._status = "running"
        self._publish_status()
        if owner_label is not None:
            if not self._published:
                raise RuntimeError("only a published generation has several owners")
            self._owner_label = _checked_owner_label(owner_label)
            # The owners that rebind this generation write into it.
            (self.directory / "entries").mkdir(exist_ok=True)
            self._publish_status()

    def inspect_published_session(self, session, *, identity, pending=False,
                                  owner_label=None, owner_fields=None):
        """Read this existing generation without attaching, rebinding or writing it."""
        from .cost_stage_checkpoint import canonical_json, canonical_json_sha256
        from .schemas import strict_json_loads

        if not isinstance(session, dict) or set(session) != {"generation", "run_identity_sha256"}:
            raise RuntimeError("a chain resume names no exact boundary session")
        session = canonical_json(dict(session), where="resumed exact boundary session")
        if canonical_json_sha256(identity, where="exact boundary source") != session["run_identity_sha256"]:
            raise RuntimeError("the relaunch's bind identity is not the one the resumed session sealed")
        directory = Path(self.config["directory"]) / str(session["generation"])
        status_path = directory / "generation.json"

        def read_status(path):
            raw = path.read_bytes()
            if not pending:
                return json.loads(raw)
            return strict_json_loads(raw,
                duplicate=lambda key: RuntimeError(f"session duplicate key {key}"),
                constant=lambda key: RuntimeError(f"session invalid constant {key}"))

        try:
            status = read_status(status_path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"the resumed generation has no readable status file at {status_path}") from exc
        if status.get("session") != session:
            raise RuntimeError(f"{status_path} names another session than the chain resume")
        policy_refusal = RuntimeError(f"{status_path} was written under another boundary storage policy")
        if boundary_storage_layout_differs(status.get("policy"), self.identity):
            raise policy_refusal
        if pending:
            if status.get("policy") != self.identity:
                raise policy_refusal
        else:
            seal_check("boundary storage policy", status.get("policy"), self.identity,
                       where=str(status_path), refusal=policy_refusal)
        if status.get("status") not in (("running",) if pending else ("running", "failed")):
            raise RuntimeError(f"the resumed generation's status is {status.get('status')!r}; "
                               "only an interrupted run (running or failed) resumes")
        entries = directory / "entries"
        if not entries.is_dir():
            raise RuntimeError(f"the resumed generation has no entries at {directory}")
        if pending and any(entries.iterdir()):
            raise RuntimeError("diagnostic pending generation already contains source entries")
        if owner_label is not None:
            label = _checked_owner_label(owner_label)
            owner_status = read_status(directory / "owners" / (label + ".json"))
            if (owner_status.get("session") != session or owner_status.get("status") != "complete"
                    or owner_status.get("policy") != self.identity
                    or owner_status.get("owner") != {"label": label, **(owner_fields or {})}):
                raise RuntimeError("diagnostic session was not issued by its actual metadata owner")
        return session, directory

    def rebind(self, session, *, identity, n_probes, check_memory=None,
               owner_label=None):
        """Adopt this run's own published generation again (PQ #1001).

        A chain resume is the same run relaunched: it keeps the original
        session byte for byte, so the checkpoints it seals after the resume
        carry the generation every checkpoint before it carries, and bands
        from both sides form one set. ``bind`` would mint a new generation;
        this reopens the one ``session`` names instead, and refuses unless
        that generation's own status file names the same session and says
        the run did not finish (``running`` after a kill, ``failed`` after
        an exception). A ``complete`` generation has a receipt, an
        ``attached`` one is read-only, and a ``retained`` one is still owned
        by owned IO that never joined.

        ``identity`` is the bind identity the relaunch recomputed; it must
        hash to the session's ``run_identity_sha256``, as it did at ``bind``.
        The status refuses in both modes, and so does a storage policy that
        lays the entries out differently (``BOUNDARY_STORAGE_LAYOUT_FIELDS``).
        A policy that differs only in its byte
        ceilings is a run seal (PQ #1147): certified mode refuses, and dev
        mode prints a ``[DEV-MODE]`` line and reopens the generation under
        this owner's own ceilings.

        The rebound owner holds no entries yet. What the resumed chain reads
        is borrowed through :meth:`authorize_resume_inputs`.

        ``owner_label`` (a chain split quantum, PQ #738) makes this owner one
        of several of the generation: its status is written to
        ``owners/<owner_label>.json`` beside ``generation.json``, which this
        owner never rewrites.
        """
        if self.session is not None:
            raise RuntimeError("exact boundary generation is already bound")
        session, directory = self.inspect_published_session(session, identity=identity)
        if owner_label is not None:
            self._owner_label = _checked_owner_label(owner_label)
        self._n_probes = n_probes
        self._check_memory = check_memory
        self._published = True
        self._resumed = True
        self.session = session
        self.directory = directory
        self._status = "running"
        self._publish_status()

    def authorize_resume_inputs(self, boundaries, checkpoint, *, boundary):
        """Borrow the sealed inputs a resumed chain reads (PQ #1001).

        ``boundaries`` are this generation's own forward boundary entries,
        recorded in the run's sealed chain state. ``checkpoint`` are the
        activation cotangents of the sealed checkpoint at ``boundary`` the
        chain resumes from. Both are read the way forward-recovery inputs
        are: through the process input map, never through this owner's
        produced output, because a relaunch is a new native producer and
        the original's groups are not its own. Neither is ever unlinked by
        this owner. A boundary entry keeps its slot, so a second writer of
        it refuses; a checkpoint entry holds none, and the first roll below
        ``boundary`` replaces it (``write(previous=...)``) without retiring
        its file.

        Forward-recovered boundaries are not passed here: the capsule
        installs them through :meth:`authorize_forward_inputs` first.
        """
        if not self._resumed or self._readonly or self._status != "running":
            raise RuntimeError("resume inputs require a rebound writable owner")
        if self._checkpoint_inputs:
            raise RuntimeError("resume inputs are authorized once")
        if type(boundary) is not int or boundary < 1:
            raise RuntimeError("a resumed chain starts at a positive checkpoint boundary")
        marker = {"generation": self.session["generation"], "kind": "adjoint_checkpoint",
                  "run_identity_sha256": self.session["run_identity_sha256"]}
        for reference in boundaries:
            identity = json.loads(reference.metadata_json)["identity"]
            if identity["kind"] != "boundary" or identity["session"] != self.session:
                raise RuntimeError("a resumed boundary entry is not this generation's")
            if (identity["slot"] in self._slots
                    or reference.name in self._references):
                raise RuntimeError("a resumed boundary entry repeats an occupied slot")
            self._forward_inputs[reference] = identity
            self._references[reference.name] = reference
            self._slots[identity["slot"]] = reference
            self.telemetry["live_artifact_bytes"] += reference.file_bytes
        self._borrow_checkpoint_plane(checkpoint, boundary=boundary, marker=marker,
                                      what="resumed")

    def authorize_seed_checkpoint(self, checkpoint, *, boundary, session):
        """Borrow another run's sealed checkpoint plane (PQ #1016).

        A seed run (``stage_a_chain_seed``) binds a fresh generation and
        continues another run's checkpoint ``boundary``. Its cotangent
        entries are read under their own ``session`` (the checkpoint's
        marker, which the caller checked against the pinned manifest), never
        unlinked, and replaced by the first roll below ``boundary`` exactly
        as a resumed checkpoint's are. The seed's forward boundaries are the
        capsule rows :meth:`authorize_forward_inputs` installs first.
        """
        if (self._resumed or self._readonly or self._status != "running"
                or self.session is None):
            raise RuntimeError("a seed checkpoint requires a freshly bound writable owner")
        if self._checkpoint_inputs:
            raise RuntimeError("seed inputs are authorized once")
        if type(boundary) is not int or boundary < 1:
            raise RuntimeError("a seeded chain starts at a positive checkpoint boundary")
        if (not isinstance(session, dict)
                or set(session) != {"generation", "kind", "run_identity_sha256"}
                or session["kind"] != "adjoint_checkpoint"
                or session["generation"] == self.session["generation"]):
            raise RuntimeError("a seed checkpoint names another run's checkpoint session")
        self._borrow_checkpoint_plane(checkpoint, boundary=boundary, marker=dict(session),
                                      what="seed")

    def _borrow_checkpoint_plane(self, checkpoint, *, boundary, marker, what):
        """Hold a sealed checkpoint's cotangents as inputs the first roll replaces.

        A copied (v1) row carries the checkpoint's own identity. A referenced
        (v2, PQ #1036) row is the checkpoint owner's cotangent entry at
        ``boundary``: kind ``cotangent``, the checkpoint session without its
        ``kind``, slot ``cotangent-{p}-{b}`` and name ``{slot}-at-{boundary}``.
        """
        owner_marker = {key: value for key, value in marker.items() if key != "kind"}
        for reference in checkpoint:
            identity = json.loads(reference.metadata_json)["identity"]
            coordinates = identity.get("coordinates") or {}
            referenced = identity["kind"] == "cotangent"
            if referenced:
                slot = (f"cotangent-{coordinates.get('probe')}-"
                        f"{coordinates.get('batch')}")
                valid = (identity["session"] == owner_marker
                         and set(coordinates) == {"batch", "boundary", "probe"}
                         and coordinates["boundary"] == boundary
                         and identity["slot"] == slot
                         and reference.name == f"{slot}-at-{boundary}")
            else:
                valid = (identity["kind"] == "adjoint_checkpoint_cotangent"
                         and identity["session"] == marker
                         and identity["slot"] == reference.name)
            if not valid:
                raise RuntimeError(
                    f"a {what} checkpoint entry is not "
                    + ("this generation's checkpoint" if what == "resumed"
                       else "the named checkpoint's"))
            if reference.name in self._references:
                raise RuntimeError(f"a {what} checkpoint entry repeats a name")
            self._forward_inputs[reference] = identity
            self._references[reference.name] = reference
            self._checkpoint_inputs[reference] = boundary
        if self._checkpoint_accounted_bytes() > self.config["max_artifact_bytes"]:
            raise RuntimeError(f"{what} inputs exceed the artifact budget")
        self.telemetry["peak_artifact_bytes"] = max(
            self.telemetry["peak_artifact_bytes"], self.telemetry["live_artifact_bytes"])

    def adopt_committed_checkpoints(self, records, directories):
        """Count the checkpoints sealed before a resume against the budget.

        The interrupted run committed them; their bytes are still on disk
        and still share ``max_artifact_bytes`` with everything this owner
        writes. Each file must be present at its sealed size.
        """
        if not self._resumed:
            raise RuntimeError("only a rebound owner adopts committed checkpoints")
        for record, directory in zip(records, directories, strict=True):
            actual = 0
            for row in record["activation_entries"] + record["shared_state_entries"]:
                size = Path(row["path"]).stat().st_size
                if size != row["file_bytes"]:
                    raise RuntimeError(
                        f"sealed checkpoint entry {row['path']} changed size")
                actual += size
            actual += (Path(directory) / "checkpoint.json").stat().st_size
            self.telemetry["live_checkpoint_bytes"] += actual
        self.telemetry["peak_checkpoint_bytes"] = max(
            self.telemetry["live_checkpoint_bytes"], self.telemetry["peak_checkpoint_bytes"])
        if self._checkpoint_accounted_bytes() > self.config["max_artifact_bytes"]:
            raise RuntimeError("sealed checkpoints exceed the artifact budget")

    def _forget_checkpoint_input(self, reference):
        """A resumed chain rolled past this checkpoint entry; its file stays."""
        del self._checkpoint_inputs[reference]
        del self._forward_inputs[reference]
        del self._references[reference.name]

    def attach(self, session, *, n_probes, forward_recovery=None):
        """Read-only bind to a published generation's exact entries.

        The distributed campaign's layer quanta read the adjoint capture's
        boundary entries through the same verified windows the producing run
        wrote them with, without owning or extending that generation: no
        entry may be written, retired or re-published through an attached
        owner, and the foreign generation's status file is never rewritten.
        ``session`` is the receipt's ``boundary_storage.session`` block.
        """
        if self.session is not None:
            raise RuntimeError("exact boundary generation is already bound")
        generation = str(session["generation"])
        directory = Path(self.config["directory"]) / generation
        if not (directory / "entries").is_dir():
            raise RuntimeError(
                f"attached exact boundary generation has no entries at {directory}")
        self._n_probes = int(n_probes)
        self.session = {"generation": generation,
            "run_identity_sha256": str(session["run_identity_sha256"])}
        self.directory = directory
        self._readonly = True
        self._status = "attached"
        if forward_recovery is not None:
            from .joint_forward_resume import SCHEMA, attached_chain
            if forward_recovery.get("schema") != SCHEMA:
                raise RuntimeError("unsupported attached forward recovery authority")
            # A chained capsule spans generations; each reference keeps its
            # own session. The chain is verified once per (path, sha256).
            identity, references = attached_chain(forward_recovery["capsule"])
            if (identity["session"] != forward_recovery["original_session"] or
                    identity["frontier"] != forward_recovery["frontier"] or
                    identity["producer"] != forward_recovery["producer"]):
                raise RuntimeError("attached forward recovery session changed")
            self._attached_forward_inputs = references

    def authorize_forward_inputs(self, references):
        """Borrow already authenticated recovery inputs without taking ownership."""
        if self._readonly or self._status != "running" or self._forward_inputs:
            raise RuntimeError("forward input authority requires a fresh writable owner")
        for reference in references:
            identity = json.loads(reference.metadata_json)["identity"]
            if identity["kind"] != "boundary" or identity["slot"] in self._slots:
                raise RuntimeError("forward recovery repeats an occupied boundary slot")
            self._forward_inputs[reference] = identity
            self._references[reference.name] = reference
            self._slots[identity["slot"]] = reference
            self.telemetry["live_artifact_bytes"] += reference.file_bytes
        if self.telemetry["live_artifact_bytes"] > self.config["max_artifact_bytes"]:
            raise RuntimeError("recovered boundaries exceed artifact budget")
        self.telemetry["peak_artifact_bytes"] = max(
            self.telemetry["peak_artifact_bytes"], self.telemetry["live_artifact_bytes"])

    def _retained_debt(self):
        return None

    def status_path(self):
        """The file this owner's status is written to.

        ``generation.json`` for the generation's one owner; a chain split
        quantum's own ``owners/<label>.json`` (PQ #738).
        """
        if self.directory is None:
            return None
        if self._owner_label is not None:
            return self.directory / "owners" / f"{self._owner_label}.json"
        return self.directory / "generation.json"

    def stamp_owner(self, **fields):
        """Add fields to a split owner's status file and write it now.

        The producer binding a later prep reads for its containment check
        is stamped here once the owner's produced output is bound.
        """
        if self._owner_label is None:
            raise RuntimeError("only a split owner stamps its own status file")
        self._owner_fields.update(fields)
        self._publish_status()

    def _publish_status(self):
        if self.directory is None or self._readonly:
            return
        from .cost_stage_checkpoint import atomic_write_bytes
        data = {"schema": self.config["schema"], "session": self.session,
                "policy": self.identity, "status": self._status,
                "working_artifacts_reusable": False, "telemetry": self.telemetry}
        if self._owner_label is not None:
            data["owner"] = {"label": self._owner_label, **self._owner_fields}
        retained = self._retained_debt()
        if retained is not None:
            data["retained"] = retained
        path = self.status_path()
        path.parent.mkdir(exist_ok=True)
        atomic_write_bytes(path,
            (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())

    def checkpoint_cotangent_sink(self, records):
        """Optional sealed local workspace for one quantum's cotangent plane."""
        scratch = cotangent_scratch(os.environ)
        if scratch is None:
            return {}
        root, ceiling = scratch
        if self._cotangent_scratch is not None:
            raise RuntimeError("boundary owner already holds cotangent scratch")
        from .perturbed_x_cache import ExactCotangentScratch
        self._cotangent_scratch = ExactCotangentScratch(
            records, directory=root, max_bytes=ceiling,
            max_tensor_bytes=self.config["max_resident_bytes"])
        return self._cotangent_scratch

    def _reserve(self, delta):
        value = self.telemetry["resident_tensor_bytes"] + delta
        if value < 0 or value > self.config["max_resident_bytes"]:
            raise RuntimeError("exact boundary tensor residency budget exceeded")
        if delta > 0 and self._check_memory is not None:
            self._check_memory("exact activation allocation")
        self.telemetry["resident_tensor_bytes"] = value
        self.telemetry["peak_resident_tensor_bytes"] = max(
            value, self.telemetry["peak_resident_tensor_bytes"])

    def reserve_resident(self, delta):
        """Charge ``delta`` exact tensor bytes to this owner's resident budget.

        Negative releases. For a reader outside the owner that holds exact
        tensors under the same ``max_resident_bytes``, such as the
        checkpoint plane stream (PQ #1142): it refuses like a window of the
        owner's own when the budget would be exceeded, and fires the bound
        memory hook on a charge.
        """
        self._reserve(delta)

    @staticmethod
    def _auxiliary_owners(batches, cotangents):
        metadata = [(batch.input_ids, batch.position_ids, batch.position_embeddings,
                     batch.attention_mask, batch.shared_pass_state) for batch in batches]
        accumulators = [cotangent.resident_tensors() for row in cotangents for cotangent in row]
        return metadata, accumulators

    def actual_auxiliary_bytes(self, batches, *, cotangents=(), extra=()):
        """Count live backing storages, distinct from the future shared reservation."""
        metadata, accumulators = self._auxiliary_owners(batches, cotangents)
        return _state_storage_bytes((metadata, accumulators, extra))

    def _auxiliary_accounted_bytes(self, batches, *, cotangents=(), extra=(),
                                   shared_extra=(), transient_bytes=0):
        """One shared auxiliary accounting for checks and transient holds.

        Retained metadata plus the larger of actual accumulator storage and
        the per-probe promised shared reservation, plus transient held
        bytes. Both ``check_auxiliary`` and transient-hold admission read
        this one calculation over watched-owner state, so a hold can never
        consume headroom already promised to shared adjoints. Point-in-time
        per-call extras belong to their own ``check_auxiliary`` admission.
        """
        metadata, actual_accumulators = self._auxiliary_owners(batches, cotangents)
        shared = [batch.shared_pass_state for batch in batches] + [shared_extra]
        reserved_shared = self._n_probes * sum(
            tensor.numel() * max(4, tensor.element_size())
            for tensor in _state_tensors(shared)
            if tensor.is_floating_point() or tensor.is_complex())
        actual_shared = _state_storage_bytes(actual_accumulators)
        return (_state_storage_bytes((metadata, extra))
                + max(actual_shared, reserved_shared) + transient_bytes,
                reserved_shared)

    def check_auxiliary(self, batches, *, cotangents=(), extra=(), shared_extra=()):
        """Bound retained metadata plus all potential per-probe shared adjoints.

        Shared state remains under the profile's original ownership/precision.
        We conservatively reserve one >=FP32 cotangent per captured occurrence
        per probe, including aliases at different shared-state keys; actual
        accumulators are checked too. No hidden tensor plane is called metadata.
        """
        total, reserved_shared = self._auxiliary_accounted_bytes(
            batches, cotangents=cotangents, extra=extra,
            shared_extra=shared_extra, transient_bytes=self._transient_hold_bytes)
        if total > self.config["max_auxiliary_bytes"]:
            raise RuntimeError("exact boundary auxiliary/shared-state residency budget exceeded")
        self.telemetry["peak_auxiliary_bytes"] = max(total, self.telemetry["peak_auxiliary_bytes"])
        self.telemetry["peak_shared_cotangent_reservation_bytes"] = max(
            reserved_shared, self.telemetry["peak_shared_cotangent_reservation_bytes"])
        if self._check_memory is not None:
            self._check_memory("exact boundary auxiliary state")

    def watch_auxiliary(self, batches, cotangents):
        """Register existing owners for deterministic end-of-call cleanup."""
        self._batches = batches
        self._cotangents = cotangents

    def watch_progress(self, reporter):
        """Report each published entry to ``joint_run_progress``, or to nothing.

        Publication is where a unit becomes durable, so progress advances only
        after verified publication, never on write intent. The
        reporter is optional so a legacy or standalone caller keeps today's
        behaviour byte for byte.
        """
        self._progress = reporter

    def _entry_identity(self, reference):
        from .perturbed_x_cache import ExactActivationReference
        if not isinstance(reference, ExactActivationReference) or (
                not self._readonly
                and self._references.get(reference.name) != reference):
            raise RuntimeError("exact boundary reference is stale or belongs to another generation")
        identity = json.loads(reference.metadata_json)["identity"]
        foreign = self._forward_inputs.get(reference) == identity or (
            self._readonly and reference in self._attached_forward_inputs)
        if (identity["session"] != self.session and not foreign) or (
                not self._readonly and reference not in self._checkpoint_inputs
                and self._slots.get(identity["slot"]) != reference):
            raise RuntimeError("exact boundary reference has a stale generation")
        return identity

    def write(self, tensor, *, batch_index, boundary_index, probe_index=None, previous=None,
              read_back=True):
        from .perturbed_x_cache import write_exact_activation_cache_entry
        if self._status != "running":
            raise RuntimeError("exact boundary generation is not running")
        if (read_back and self._produced_plan is not None
                and self._produced_plan["write_only"]):
            # Before any byte: a write-only template funds no window, so no
            # read of this group could ever be staged for this action.
            raise RuntimeError(
                "a write-only produced output is never read back by the action "
                "that writes it: write its entries with read_back=False")
        self._produced_flush_deferred_unlinks()
        kind = "boundary" if probe_index is None else "cotangent"
        coordinates = {"batch": batch_index, "boundary": boundary_index, "probe": probe_index}
        if any(type(v) is not int or v < 0 for v in (batch_index, boundary_index)):
            raise ValueError("exact boundary coordinates must be nonnegative integers")
        if probe_index is not None and (type(probe_index) is not int or not 0 <= probe_index < self._n_probes):
            raise ValueError("exact cotangent probe coordinate is outside the run")
        slot = f"boundary-{batch_index}-{boundary_index}" if probe_index is None else f"cotangent-{probe_index}-{batch_index}"
        if previous is not None and previous in self._checkpoint_inputs:
            # A resumed chain's first roll replaces a sealed checkpoint
            # cotangent (PQ #1001): same probe and batch, one boundary down.
            old = self._entry_identity(previous)
            if (kind != "cotangent" or slot in self._slots
                    or old["slot"] != slot
                    or self._checkpoint_inputs[previous] != boundary_index + 1):
                raise RuntimeError("exact cotangent rollover changed its original coordinates")
        elif previous is not None:
            old = self._entry_identity(previous)
            if (kind != "cotangent" or old["slot"] != slot
                    or old["coordinates"]["boundary"] != boundary_index + 1):
                raise RuntimeError("exact cotangent rollover changed its original coordinates")
        elif slot in self._slots:
            raise RuntimeError("exact boundary slot already exists")
        nbytes = tensor.numel() * tensor.element_size()
        # A bounded envelope for the exact writer's small PyTorch zip header.
        # Actual file length is checked before the entry can be published.
        # Admission is aggregate: live ordinary bytes plus live checkpoint
        # bytes plus every active/retained checkpoint envelope share the one
        # max_artifact_bytes ceiling in both directions, so a committed (or
        # retained) checkpoint narrows later ordinary writes exactly as live
        # ordinary entries narrow later checkpoints.
        file_limit = nbytes + 65536
        if (self._produced_plan is not None
                and nbytes > self._produced_plan["max_entry_tensor_bytes"]):
            raise RuntimeError("produced output entry exceeds its sealed group tensor ceiling")
        remaining = self.checkpoint_remaining_bytes()
        if file_limit > remaining:
            raise RuntimeError(
                "exact boundary artifact budget exceeded: "
                f"entry needs {file_limit} bytes, {remaining} remain of "
                f"{self.config['max_artifact_bytes']}")
        name = f"{slot}-at-{boundary_index}"
        identity = {"session": self.session, "slot": slot, "kind": kind, "coordinates": coordinates}
        # Reserve the whole group before the first byte of any member.
        produced_group = None
        if self._produced is not None:
            self._require_produced_copies()
            plan = self._produced_plan
            if not plan["batch_start"] <= batch_index < plan["batch_stop"]:
                # A split quantum's groups are its own range's (PQ #738):
                # this entry's group belongs to another owner.
                raise RuntimeError(
                    f"batch {batch_index} is outside this owner's produced range "
                    f"{plan['batch_start']}:{plan['batch_stop']}")
            produced_group = self._produced_group_for_write(
                self._produced_group_key(
                    kind=kind, batch_index=batch_index,
                    boundary_index=boundary_index, probe_index=probe_index,
                    group_size=self._produced_plan["group_size"]))
        write_directory = self.directory / "entries"
        if self._local_output_spool is not None:
            write_directory = self._local_output_spool.directory(produced_group["batch_id"])
        self._reserve(nbytes)
        try:
            with torch.profiler.record_function("aura.exact_activation.write"):
                # The local file is provisional. Copy publication fsyncs and
                # verifies canonical bytes; own readback verifies exact bytes.
                reference = write_exact_activation_cache_entry(write_directory, name, tensor,
                    identity=identity, max_tensor_bytes=nbytes, max_file_bytes=file_limit,
                    preallocate=self._local_output_spool is not None,
                    durable=self._local_output_spool is None)
                if self._local_output_spool is not None:
                    # ``read_back`` keeps the group's local copy after its
                    # export lands, for this box's own reads (PQ #1110).
                    reference = self._local_output_spool.record(
                        produced_group["batch_id"], reference, self.directory / "entries",
                        read_back=read_back)
        finally:
            self._reserve(-nbytes)
        self._references[name] = reference
        self._slots[slot] = reference
        if produced_group is not None:
            produced_key = self._produced_group_key(
                kind=kind, batch_index=batch_index,
                boundary_index=boundary_index, probe_index=probe_index,
                group_size=self._produced_plan["group_size"])
            with self._produced_lock.held():
                produced_group["references"].append(reference)
                if self._local_output_spool is None:
                    produced_group['file_identities'][reference.path] = reference.writer_file_identity
                produced_group["live_references"] += 1
                self._produced_index[reference] = produced_key
                complete = (len(produced_group["references"])
                            == len(produced_group["planned"]) // 2)
            if complete and self._local_output_spool is not None:
                self._local_output_spool.submit(produced_group["batch_id"],
                    lifetime=self._produced_plan["origin_lifetime"] or "retain")
            elif complete and self._produced_plan["write_only"]:
                # Written straight to its canonical paths, so it commits at
                # its origin now. Through the spool it commits once its
                # export is acknowledged (``settle_local_output``).
                self._produced_commit_origin(produced_group)
            if complete and read_back:
                # Complete direct groups publish here. Spool IO already owns
                # publication; no read follows the final unread roll.
                self._publish_complete_produced_group(produced_group)
        self.telemetry["written_entries"] += 1
        self.telemetry["written_tensor_bytes"] += nbytes
        self.telemetry["live_artifact_bytes"] += reference.file_bytes
        self.telemetry["peak_artifact_bytes"] = max(
            self.telemetry["live_artifact_bytes"], self.telemetry["peak_artifact_bytes"])
        if previous is not None and previous in self._checkpoint_inputs:
            self._forget_checkpoint_input(previous)
        elif previous is not None:
            self._retire(previous)
        if self._local_output_spool is not None:
            self._request_export_poll()
            self._commit_local_output_progress()
        elif self._progress is not None:
            self._progress.entry(layer=boundary_index, partition=batch_index, kind=kind)
        if self._check_memory is not None:
            self._check_memory("exact activation publication")
        return reference

    def _request_export_poll(self):
        self._require_produced_copies()

    def _commit_local_output_progress(self):
        """Report the entries whose exports were acknowledged. Polls nothing."""
        if self._local_output_spool is None:
            return
        for reference in self._local_output_spool.durable_entries():
            if isinstance(reference, ProducedFileReference):
                # A small file of a record group (write_produced_files):
                # durable, but not an entry the progress contract counts.
                continue
            if self._progress is not None:
                identity = json.loads(reference.metadata_json)["identity"]
                coordinates = identity["coordinates"]
                self._progress.entry(layer=coordinates["boundary"],
                                     partition=coordinates["batch"], kind=identity["kind"])

    def cancel_waits_when(self, predicate):
        """End this owner's spool waits once ``predicate()`` is true (PQ #1251).

        A writer on a thread of its own (the streamed handoff) is cancelled
        when its quantum fails; a wait on a window or an export would
        otherwise outlive the failing consumer. The wait then
        raises :class:`~prismaquant.produced_output_spool.
        ProducedOutputWaitCancelled` and the owner exits on its failure
        path. Without a spool there is no wait to end.
        """
        if self._local_output_spool is not None:
            self._local_output_spool.cancelled = predicate

    def cancel_waits(self):
        if self._local_output_spool is not None:
            self._local_output_spool.cancel_waits()

    def settle_local_output(self):
        """Join owned copies and require durable manifests before completion."""
        if self._local_output_spool is not None:
            self._local_output_spool.drain(release=True)
            self._produced_flush_deferred_unlinks(final=True)
            self._commit_local_output_progress()
        for group in list(self._produced_groups.values()):
            if not group["origin_reclaimed"] and len(group["references"]) == len(group["planned"]) // 2:
                self._produced_commit_origin(group)

    def _produced_commit_origin(self, group):
        if group.get("origin_ref") is not None:
            return group["origin_ref"]
        batch_id = group["batch_id"]
        if self._local_output_spool is not None:
            self._local_output_spool.await_group(batch_id, where="commit")
        descriptors = [self._produced.descriptor_for(ref, producer_generation=batch_id)
                       for ref in group["references"]]
        lifetime = self._produced_plan["origin_lifetime"] or "retain"
        out = self._produced.commit_origin(batch_id=batch_id, descriptors=descriptors, lifetime=lifetime)
        group["origin_ref"] = dict(out["ref"])
        group["published"] = out["ref"]
        group["manifest_digest"] = self._produced.manifest_digest_for(descriptors)
        self._produced_origin_batches.append(dict(out["ref"]))
        self.telemetry["produced_groups_committed_at_origin"] += 1
        return group["origin_ref"]

    def produced_origin_batches(self):
        """The refs of the groups committed at their origin, in commit order.

        Each binding names one immutable application batch, including its
        exact bytes, session and producer.
        """
        return [dict(ref) for ref in self._produced_origin_batches]

    def write_produced_files(self, files, *, kind, boundary_index):
        """Write small files into the generation directory as one group.

        ``files`` is an ordered list of ``(name, bytes)`` pairs. An owner
        bound to a produced-output publication writes them as one
        produced-output group (PQ #1015). It claims the group's exact
        payload before the first byte, with the final and ``.tmp`` names as
        its planned paths. Through the local spool the files land in the
        local directory and the existing IO engine copies them in list order.
        This method returns only after the export is acknowledged, so a
        caller that puts its record last can rely on the record's presence.
        Without the spool each file is written through its ``.tmp`` name
        and linked into place, in the same order.

        No read follows in this action, so a bound owner's template must be
        write-only, and the group is committed at its
        origin once its files are durable, with the owner's origin lifetime
        (PQ #1075): directly after the last file is linked, or through the
        spool after verified durable publication. A read-back template
        refuses before the first byte, because nothing could ever commit
        the group.

        An unbound owner writes each file with ``atomic_write_bytes``.
        Returns one :class:`ProducedFileReference` per file, in order.
        """

        from .cost_stage_checkpoint import atomic_write_bytes
        if self._status != "running":
            raise RuntimeError("exact boundary generation is not running")
        files = [(str(name), bytes(payload)) for name, payload in files]
        names = [name for name, _ in files]
        if not files or len(set(names)) != len(names) or any(
                not name or "/" in name or name.startswith(".")
                or name.endswith(".tmp") or name == "generation.json"
                for name in names):
            raise ValueError("produced files need distinct bare names")
        if any(not payload for _, payload in files):
            raise ValueError("a produced file cannot be empty")
        references = [
            ProducedFileReference(
                path=str(self.directory / name), name=name,
                file_bytes=len(payload),
                sha256=bytes_sha256hex(payload))
            for name, payload in files]
        total = sum(ref.file_bytes for ref in references)
        # The files share max_artifact_bytes with the entries, as the
        # declaration's payload maximum.
        remaining = self.checkpoint_remaining_bytes()
        if total > remaining:
            raise RuntimeError(
                "exact boundary artifact budget exceeded: produced files need "
                f"{total} bytes, {remaining} remain of "
                f"{self.config['max_artifact_bytes']}")
        if self._produced is None:
            for (name, payload) in files:
                atomic_write_bytes(self.directory / name, payload)
            self._count_produced_files(total)
            return references
        if not self._produced_plan["write_only"]:
            raise RuntimeError(
                "produced files are never read back by the action that writes "
                "them, so they commit at their origin, which needs a "
                "write-only produced-output declaration")
        self._require_produced_copies()
        batch_id = self._produced.batch_id_for(
            kind=kind, boundary_index=boundary_index, group_index=0)
        planned = []
        for name in names:
            planned += [str(self.directory / name),
                        str(self.directory / name) + ".tmp"]
        self._produced.require_prewrite(
            batch_id=batch_id, payload_ceiling_bytes=total,
            temp_ceiling_bytes=max(ref.file_bytes for ref in references), paths=planned)
        spool = self._local_output_spool
        try:
            if spool is None:
                for (name, payload) in files:
                    _link_new_file(self.directory / name, payload)
            else:
                local = spool.reserve(batch_id, total)
                for (name, payload), reference in zip(files, references):
                    path = Path(local) / name
                    _write_new_file(path, payload)
                    spool.record(batch_id, dataclasses.replace(
                        reference, path=str(path)), self.directory)
                spool.submit(batch_id, lifetime=self._produced_plan["origin_lifetime"])
                # The handoff record's barrier: on the export's own state,
                # never a clock (PQ #1110).
                spool.await_group(batch_id, where="produced-files")
        except BaseException:
            if spool is None or not spool.pending(batch_id):
                # Release only after every planned path is absent.
                self._produced.abort_prewrite(batch_id=batch_id)
            raise
        self._produced_commit_origin(
            {"batch_id": batch_id, "references": references})
        self.telemetry["produced_file_groups"] += 1
        self._count_produced_files(total)
        return references

    def _count_produced_files(self, nbytes):
        self.telemetry["live_artifact_bytes"] += nbytes
        self.telemetry["peak_artifact_bytes"] = max(
            self.telemetry["live_artifact_bytes"],
            self.telemetry["peak_artifact_bytes"])

    def _retire(self, reference, *, missing_ok=False):
        if self._references.get(reference.name) != reference:
            raise RuntimeError("exact boundary retirement has a stale reference")
        if reference in self._checkpoint_inputs:
            raise RuntimeError(
                "a sealed checkpoint entry is never retired by the chain that resumed from it")
        if reference in self._pinned_checkpoint_entries:
            # PQ #1036: a committed checkpoint names this entry. The roll is
            # done with it, but the file is the checkpoint's now: it leaves
            # the live set, its bytes move to the checkpoint ledger, and its
            # produced group keeps it live, so the group's origin charge is
            # never reclaimed under it.
            del self._references[reference.name]
            self.telemetry["live_artifact_bytes"] -= reference.file_bytes
            self.telemetry["live_checkpoint_bytes"] += reference.file_bytes
            self.telemetry["peak_checkpoint_bytes"] = max(
                self.telemetry["live_checkpoint_bytes"],
                self.telemetry["peak_checkpoint_bytes"])
            self.telemetry["retired_entries"] += 1
            self.telemetry["pinned_checkpoint_entries_retired"] = (
                self.telemetry.get("pinned_checkpoint_entries_retired", 0) + 1)
            # No read of it follows here, so it no longer holds its group's
            # local copy; the file itself stays, it is the checkpoint's.
            self._spool_retire_entry(reference)
            return
        if reference in self._forward_inputs:
            # Borrowed files and original ACKs remain owned by the old attempt.
            del self._references[reference.name]
            del self._forward_inputs[reference]
            self.telemetry["live_artifact_bytes"] -= reference.file_bytes
            self.telemetry["retired_entries"] += 1
            return
        if self._local_output_spool is not None and not missing_ok:
            batch_id = self._spool_retire_entry(reference)
            if batch_id is not None and (
                    self._local_output_spool.holds(batch_id)
                    or self._produced_group_read_may_follow(
                        reference, retiring=reference)):
                # Write-behind (PQ #1110): the chain does not wait for the
                # export here. The entry leaves the live set now; its
                # canonical file goes once the group's local copy is
                # released, which happens only after the copy
                # landed and which re-checks every landed destination. The
                # window bounds how many wait: a full window makes the next
                # reservation wait on the exports, never on a clock.
                # A group the window released for room while another of
                # its entries is still live keeps the file too (PQ #1236):
                # that entry's read publishes or restages the whole group,
                # and rematerialization verifies every canonical file.
                del self._references[reference.name]
                self.telemetry["retired_entries"] += 1
                self._deferred_unlinks.setdefault(batch_id, []).append(reference)
                self.telemetry["produced_deferred_unlinks"] += 1
                self.telemetry["produced_deferred_unlink_bytes"] += reference.file_bytes
                self._produced_flush_deferred_unlinks()
                return
            self._commit_local_output_progress()
        if self._produced is not None and not missing_ok:
            self._produced_await_copy_before_unlink(reference)
            _key, group = self._produced_group_for(reference)
            self._produced.unlink_origin(group['batch_id'], reference)
        else:
            Path(reference.path).unlink(missing_ok=missing_ok)
        del self._references[reference.name]
        self.telemetry["retired_entries"] += 1
        self._produced_forget_origin(reference)
        if self._local_output_spool is not None and not missing_ok:
            # The group's last live entry is gone, and with it every read
            # that could publish the group: the files held for it go too.
            self._produced_flush_deferred_unlinks()

    def _produced_group_read_may_follow(self, reference, *, retiring=None):
        """Can a read here still publish or restage ``reference``'s group?

        True while any entry of the group other than ``retiring`` is live
        (PQ #1236). A read of that entry publishes the whole group through
        the publication owner, or rematerializes it, and both verify every
        canonical file in the immutable manifest. So no retired file of the group may
        go before it. A retired entry is out of ``_references``, so no read
        of it follows here (``_entry_identity``).
        """

        _key, group = self._produced_group_for(reference)
        if group is None:
            return False
        return any(entry != retiring
                   and self._references.get(entry.name) == entry
                   for entry in group["references"])

    def _spool_retire_entry(self, reference):
        """Tell the spool no read of ``reference`` follows here.

        Returns the entry's group batch id, or None when it is in no spool
        group. The group's local copy is released once its export landed and
        every entry is retired (``ProducedOutputSpool.retire_entry``).
        """

        if self._local_output_spool is None:
            return None
        _key, group = self._produced_group_for(reference)
        if group is None or not self._local_output_spool.pending(group["batch_id"]):
            return None if group is None else group["batch_id"]
        self._local_output_spool.retire_entry(group["batch_id"], reference)
        return group["batch_id"]

    def _produced_flush_deferred_unlinks(self, *, final=False):
        """Unlink the retired canonical files no step here can need again.

        Runs on the thread that writes and retires (PQ #1110): at each
        write, each retirement and at settle. Never waits. A group still
        held keeps its deferred entries, and its local copy counts against
        the window until verified durable publication. A released group keeps
        them too while another of its entries is live (PQ #1236): the
        window can release a group for room before its reads are done, and
        the read of a live entry then publishes or restages the whole group
        from its origins. ``final`` is the action's end, when no read
        follows here, so only the local copy's release holds a file back.
        """

        if not self._deferred_unlinks:
            return
        spool = self._local_output_spool
        for batch_id in list(self._deferred_unlinks):
            if spool.holds(batch_id):
                continue
            if not final and self._produced_group_read_may_follow(
                    self._deferred_unlinks[batch_id][0]):
                continue
            for reference in list(self._deferred_unlinks[batch_id]):
                self._produced.unlink_origin(batch_id, reference)
                self.telemetry["produced_deferred_unlinks_done"] += 1
                self.telemetry["produced_deferred_unlink_bytes"] -= reference.file_bytes
                self._produced_forget_origin(reference)
                self._deferred_unlinks[batch_id].remove(reference)
            del self._deferred_unlinks[batch_id]

    def _produced_forget_origin(self, reference):
        """Bookkeeping once an entry's canonical file is gone."""

        self.telemetry["live_artifact_bytes"] -= reference.file_bytes
        if self._produced is not None:
            with self._produced_lock.held():
                key = self._produced_index.get(reference)
                group = None if key is None else self._produced_groups.get(key)
                if group is not None and group["live_references"] > 0:
                    group["live_references"] -= 1
                self._reclaim_produced_origin_if_final(reference)
                # The origin is gone, so the reference is dead: drop it from
                # the lookup index AFTER the reclaim gate has read it. The
                # group's own list is untouched -- the committed batch's
                # descriptors are immutable and are not rewritten by this
                # owner disposing of its files.
                self._produced_index.pop(reference, None)

    def _produced_await_copy_before_unlink(self, reference):
        _key, group = self._produced_group_for(reference)
        if group is not None and self._local_output_spool is not None:
            self._local_output_spool.await_group(group["batch_id"], where="before-unlink")

    def _reclaim_produced_origin_if_final(self, reference):
        _key, group = self._produced_group_for(reference)
        if group is None or group["live_references"] or group["origin_reclaimed"]:
            return
        result = self._produced.reclaim_origin(group["batch_id"])
        if not result["ok"]:
            raise RuntimeError(f"owned origin release refused: {result}")
        group["origin_reclaimed"] = True
        self.telemetry["produced_groups_origin_reclaimed"] += 1

    def retire(self, reference):
        if self._readonly:
            raise RuntimeError(
                "an attached read-only generation cannot retire entries")
        identity = self._entry_identity(reference)
        self._retire(reference)
        del self._slots[identity["slot"]]

    # ------------------------------------------------------------------
    # Checkpoint artifact budget: strided-checkpoint files counted in the
    # same max_artifact_bytes ceiling as ordinary boundary/cotangent files.
    #
    # The stage-A checkpoint writer serializes outside this owner's entry
    # directory, so write()'s pre-write check never sees those bytes. The
    # contract here is reserve-before-write: the writer computes its exact
    # file plan (tensor envelopes, serialized shared-state payloads,
    # manifest envelope, in-progress temp overlap), this owner admits the
    # whole envelope against live ordinary bytes plus live checkpoint bytes
    # plus every active/retained envelope, and only then does the writer
    # create its directory. A post-write register call could not enforce
    # the ceiling, so anything that does not fit refuses with its counts
    # before a single file lands.
    #
    # Single-owner serial writer: stage A checkpoints one boundary at a
    # time on its owning thread. At most one reservation is active; a
    # second concurrent reservation refuses rather than racing. No locks.
    # ------------------------------------------------------------------

    def check_transient_buffer(self, label):
        """Invoke the bound memory-owner hook for a transient serialization buffer.

        Durable checkpoint bytes are reserved in the artifact budget below;
        the transient pickle/manifest buffers live in RAM and are reported
        through the same hook ordinary allocations use. Unbound owners keep
        today's behavior byte for byte.
        """
        if self._check_memory is not None:
            self._check_memory(str(label))

    @contextmanager
    def hold_transient_serialization(self, estimate_bytes, label):
        """Hold estimate bytes in the resident budget around one serialization.

        The existing resident contract bounds the transient peak: the hold
        refuses (fail-closed, with counts) when live tensors plus this
        estimate would exceed max_resident_bytes, fires the bound memory hook
        on the way in, and releases on the way out. One entry at a time keeps
        the peak at the largest single payload instead of the whole
        checkpoint. Only the resident counter moves; durable bytes are a
        separate reservation below. Tensor compact copies use this hold,
        mirroring how write() reserves each exact entry; serialized metadata
        buffers use hold_transient_metadata against the auxiliary ceiling.
        """
        if type(estimate_bytes) is not int or estimate_bytes <= 0:
            raise RuntimeError(
                "exact boundary transient serialization hold needs a "
                "positive byte estimate")
        self._reserve(int(estimate_bytes))
        try:
            yield
        finally:
            self._reserve(-int(estimate_bytes))

    @contextmanager
    def hold_transient_metadata(self, estimate_bytes, label):
        """Hold transient serialized-metadata bytes in one aggregate ceiling.

        Admission reads the same accounted total ``check_auxiliary``
        enforces -- retained metadata plus the larger of actual accumulator
        storage and the per-probe promised shared reservation -- recomputed
        over the watched owners with outstanding transient holds included,
        plus this estimate. A hold that fits live-actual usage but exceeds
        the promised reservation refuses before anything serializes. Fires
        the bound memory hook on the way in and releases on the way out;
        one entry at a time keeps the peak at the largest single payload.
        """
        if type(estimate_bytes) is not int or estimate_bytes <= 0:
            raise RuntimeError(
                "exact boundary transient metadata hold needs a "
                "positive byte estimate")
        live_total, _ = self._auxiliary_accounted_bytes(
            self._batches or (), cotangents=self._cotangents or (),
            transient_bytes=self._transient_hold_bytes + int(estimate_bytes))
        if live_total > self.config["max_auxiliary_bytes"]:
            raise RuntimeError(
                "exact boundary transient serialization budget exceeded: "
                f"{label} needs {estimate_bytes} bytes against accounted "
                f"auxiliary usage, ceiling {self.config['max_auxiliary_bytes']}")
        self._transient_hold_bytes += int(estimate_bytes)
        try:
            if self._check_memory is not None:
                self._check_memory(str(label))
            self.telemetry["peak_transient_serialization_bytes"] = max(
                self._transient_hold_bytes,
                self.telemetry["peak_transient_serialization_bytes"])
            yield
        finally:
            self._transient_hold_bytes -= int(estimate_bytes)

    def checkpoint_reservation_state(self, reservation_id):
        """Report one reservation's lifecycle state, refusing unknown ids."""
        if type(reservation_id) is not int:
            raise RuntimeError("exact boundary checkpoint reservation is not an integer")
        entry = self._checkpoint_reservations.get(reservation_id)
        if entry is None:
            raise RuntimeError("exact boundary checkpoint reservation is unknown")
        return entry["state"]

    def _checkpoint_accounted_bytes(self):
        return (self.telemetry["live_artifact_bytes"]
                + self.telemetry["live_checkpoint_bytes"]
                + sum(record["envelope_bytes"]
                      for record in self._checkpoint_reservations.values()
                      if record["state"] in ("active", "retained")))

    def checkpoint_remaining_bytes(self):
        """Uncommitted headroom under max_artifact_bytes across both ledgers."""
        return self.config["max_artifact_bytes"] - self._checkpoint_accounted_bytes()

    @staticmethod
    def _validate_checkpoint_file_plan(file_plan):
        if not isinstance(file_plan, dict):
            raise RuntimeError("exact boundary checkpoint file plan is not a mapping")
        files = file_plan.get("files")
        manifest_bytes = file_plan.get("manifest_bytes")
        temp_overlap_bytes = file_plan.get("temp_overlap_bytes")
        envelope_bytes = file_plan.get("envelope_bytes")
        # A referenced checkpoint (PQ #1036) plans no cotangent file and,
        # when opened before its roll, no shared state yet: its first plan
        # is the manifest alone.
        if (not isinstance(files, list)
                or type(manifest_bytes) is not int or manifest_bytes <= 0
                or type(temp_overlap_bytes) is not int or temp_overlap_bytes < 0
                or type(envelope_bytes) is not int or envelope_bytes <= 0):
            raise RuntimeError("exact boundary checkpoint file plan is malformed")
        total = int(manifest_bytes) + int(temp_overlap_bytes)
        for entry in files:
            if (not isinstance(entry, dict) or type(entry.get("name")) is not str
                    or not entry["name"]
                    or type(entry.get("path")) is not str
                    or not entry["path"]
                    or type(entry.get("envelope_bytes")) is not int
                    or entry["envelope_bytes"] <= 0):
                raise RuntimeError("exact boundary checkpoint file plan is malformed")
            total += entry["envelope_bytes"]
        if total != envelope_bytes:
            raise RuntimeError(
                "exact boundary checkpoint envelope does not match its file plan")
        return files

    def reserve_checkpoint_artifact(self, *, label, envelope_bytes, file_plan,
                                    checkpoint_dir):
        """Admit one checkpoint attempt's whole envelope before it writes.

        Returns an integer reservation id. Refuses (counting a refusal) when
        the envelope does not fit the remaining ceiling, when another
        reservation is already active, or when the owner is not a running
        writer. Nothing is created by this call.
        """
        if self._readonly:
            raise RuntimeError(
                "an attached read-only generation cannot reserve checkpoint artifacts")
        if self._status != "running":
            raise RuntimeError("exact boundary generation is not running")
        if self._checkpoint_active is not None:
            raise RuntimeError(
                "exact boundary checkpoint reservation already active: single-owner writer")
        files = self._validate_checkpoint_file_plan(file_plan)
        if file_plan["envelope_bytes"] != envelope_bytes:
            raise RuntimeError(
                "exact boundary checkpoint envelope does not match its file plan")
        if type(envelope_bytes) is not int or envelope_bytes <= 0:
            raise RuntimeError(
                "exact boundary checkpoint envelope must be a positive byte count")
        try:
            directory = Path(checkpoint_dir)
        except TypeError as exc:
            raise RuntimeError(
                "exact boundary checkpoint directory is not a path") from exc
        if not directory.is_absolute():
            raise RuntimeError(
                "exact boundary checkpoint directory must be absolute")
        for entry in files:
            try:
                inside = Path(entry["path"]).is_relative_to(directory)
            except ValueError:
                inside = False
            if not inside:
                raise RuntimeError(
                    "exact boundary checkpoint file plan escapes its "
                    f"attempt directory: {entry['name']}")
        remaining = self.checkpoint_remaining_bytes()
        if envelope_bytes > remaining:
            self.telemetry["checkpoint_refusals"] += 1
            raise RuntimeError(
                "exact boundary checkpoint artifact budget exceeded: "
                f"{label} needs {envelope_bytes} bytes, {remaining} remain of "
                f"{self.config['max_artifact_bytes']}")
        reservation = self._next_checkpoint_reservation
        produced_batch = None
        if self._produced is not None:
            produced_batch = self._produced.batch_id_for(kind='checkpoint',
                boundary_index=0, group_index=reservation)
            paths = [str(directory / 'checkpoint.json')]
            paths += [row['path'] for row in files]
            self._produced.require_prewrite(batch_id=produced_batch,
                payload_ceiling_bytes=0,
                checkpoint_ceiling_bytes=envelope_bytes - file_plan['temp_overlap_bytes'],
                temp_ceiling_bytes=file_plan['temp_overlap_bytes'],
                paths=[p for filename in paths for p in (filename, filename + '.tmp')])
        self._next_checkpoint_reservation += 1
        self._checkpoint_reservations[reservation] = {
            "envelope_bytes": envelope_bytes, "files": files,
            "manifest_bytes": file_plan["manifest_bytes"],
            "temp_overlap_bytes": file_plan["temp_overlap_bytes"],
            "dir": str(directory), "label": str(label), "state": "active",
            "receipt_digest": None,
            "produced_batch": produced_batch,
            "directory_identity": None, "retained_files": None,
        }
        self._checkpoint_active = reservation
        self.telemetry["checkpoint_reservations"] += 1
        return reservation

    def checkpoint_directory_created(self, reservation_id):
        from .staged_manifest import open_directory
        entry = self._checkpoint_reservations[reservation_id]
        descriptor = open_directory(entry['dir'])
        try:
            info = os.fstat(descriptor)
            if entry['directory_identity'] is not None:
                raise RuntimeError('checkpoint directory ownership already recorded')
            entry['directory_identity'] = (info.st_dev, info.st_ino)
        finally:
            os.close(descriptor)

    def _checkpoint_retained_files(self, entry):
        from .staged_manifest import open_directory
        from .file_identity import file_stat_signature
        import stat
        directory = Path(entry['dir'])
        if not os.path.lexists(directory):
            return {}
        descriptor = open_directory(str(directory))
        try:
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) != entry['directory_identity']:
                raise RuntimeError('checkpoint directory ownership changed')
        finally:
            os.close(descriptor)
        allowed = {str(directory / 'checkpoint.json'), str(directory / 'checkpoint.json.tmp')}
        allowed.update(p for row in entry['files'] for p in (row['path'], row['path'] + '.tmp'))
        records = {}
        for parent, directories, files in os.walk(directory, followlinks=False):
            for name in directories:
                child = Path(parent) / name
                if child != directory / 'entries' or child.is_symlink():
                    raise RuntimeError('unknown checkpoint directory retains reservation')
            for name in files:
                child = Path(parent) / name
                info = child.lstat()
                if str(child) not in allowed or not stat.S_ISREG(info.st_mode):
                    raise RuntimeError('unknown checkpoint file retains reservation')
                records[str(child)] = list(file_stat_signature(info))
        return records

    def extend_checkpoint_artifact(self, reservation_id, *, files,
                                   temp_overlap_bytes):
        """Admit files an attempt could only plan after it started writing.

        A checkpoint written while its plane is rolled (PQ #1002) reserves
        its tensor files before the roll, but its shared states exist only
        after it. This adds their envelopes to the one active reservation,
        against the same remaining ceiling ``reserve_checkpoint_artifact``
        admits against, before any of them is written. The manifest
        envelope does not change, so the attempt must have planned the
        manifest for these files already; ``temp_overlap_bytes`` may only
        grow. A refusal (counted like a reservation refusal) changes
        nothing, and the attempt, which has written files, must abandon.
        """
        if self._readonly:
            raise RuntimeError(
                "an attached read-only generation cannot reserve checkpoint artifacts")
        if self._status != "running":
            raise RuntimeError("exact boundary generation is not running")
        if type(reservation_id) is not int:
            raise RuntimeError("exact boundary checkpoint reservation is not an integer")
        entry = self._checkpoint_reservations.get(reservation_id)
        if entry is None:
            raise RuntimeError("exact boundary checkpoint reservation is unknown")
        if entry["state"] != "active" or self._checkpoint_active != reservation_id:
            raise RuntimeError(
                "exact boundary checkpoint reservation is not active; "
                "only an open attempt extends")
        if not isinstance(files, list) or not files:
            raise RuntimeError("exact boundary checkpoint extension is malformed")
        if (type(temp_overlap_bytes) is not int
                or temp_overlap_bytes < entry["temp_overlap_bytes"]):
            raise RuntimeError(
                "exact boundary checkpoint extension may only grow its "
                "temp overlap")
        for row in files:
            if (not isinstance(row, dict) or type(row.get("name")) is not str
                    or not row["name"]
                    or type(row.get("path")) is not str or not row["path"]
                    or type(row.get("envelope_bytes")) is not int
                    or row["envelope_bytes"] <= 0):
                raise RuntimeError("exact boundary checkpoint extension is malformed")
        planned = {row["name"] for row in entry["files"]}
        names = [row["name"] for row in files]
        if len(set(names)) != len(names) or planned & set(names):
            raise RuntimeError(
                "exact boundary checkpoint extension repeats a planned file")
        directory = Path(entry["dir"])
        for row in files:
            try:
                inside = Path(row["path"]).is_relative_to(directory)
            except ValueError:
                inside = False
            if not inside:
                raise RuntimeError(
                    "exact boundary checkpoint file plan escapes its "
                    f"attempt directory: {row['name']}")
        delta = (sum(row["envelope_bytes"] for row in files)
                 + temp_overlap_bytes - entry["temp_overlap_bytes"])
        remaining = self.checkpoint_remaining_bytes()
        if delta > remaining:
            self.telemetry["checkpoint_refusals"] += 1
            raise RuntimeError(
                "exact boundary checkpoint artifact budget exceeded: "
                f"{entry['label']} needs {delta} more bytes, {remaining} "
                f"remain of {self.config['max_artifact_bytes']}")
        if entry['produced_batch'] is not None:
            self._produced.extend_prewrite(batch_id=entry['produced_batch'],
                checkpoint_ceiling_bytes=sum(row['envelope_bytes'] for row in files),
                temp_ceiling_bytes=temp_overlap_bytes,
                paths=[p for row in files for p in (row['path'], row['path'] + '.tmp')])
        entry["files"] = list(entry["files"]) + list(files)
        entry["envelope_bytes"] += delta
        entry["temp_overlap_bytes"] = temp_overlap_bytes
        return entry["envelope_bytes"]

    def await_checkpoint_references(self, references):
        """Wait until every referenced own entry is durable at its origin.

        A referenced checkpoint (PQ #1036) names the owner's canonical entry
        paths. Through a local output spool those land after verified copy
        publication; the checkpoint's manifest is
        written only after that, so a checkpoint that exists names only
        durable files. Without a spool the entries were written in place.
        Returns the seconds waited; with a spool they also accumulate in
        ``checkpoint_reference_wait_s``.
        """
        import time

        started = time.monotonic()
        batches = []
        for reference in references:
            self._entry_identity(reference)
            if self._local_output_spool is None and self._produced is None:
                continue
            _key, group = self._produced_group_for(reference)
            if group is None and self._local_output_spool is None:
                continue  # Not produced here; its existing input authority owns the bytes.
            if group is None:
                raise RuntimeError(
                    f"referenced checkpoint entry {reference.name} is in no "
                    "produced group")
            if group["batch_id"] not in batches:
                batches.append(group["batch_id"])
        if self._local_output_spool is not None:
            for batch_id in batches:
                # On the export's own state, never a clock (PQ #1110).
                self._local_output_spool.await_group(
                    batch_id, where="checkpoint-references")
            if batches:
                self._commit_local_output_progress()
        waited = time.monotonic() - started
        if self._local_output_spool is not None:
            # Only a spool has anything to wait for; without one the owner's
            # status stays free of wall-clock fields it did not have before.
            self.telemetry["checkpoint_reference_wait_s"] = (
                self.telemetry.get("checkpoint_reference_wait_s", 0.0) + waited)
        if self._produced is not None and batches:
            # Checkpoints need retained entries. Write-only consumed groups
            # cannot lend bytes to a later checkpoint.
            lifetimes = self._produced.origin_only_lifetimes()
            consumed = sorted(batch_id for batch_id in batches
                              if lifetimes.get(batch_id) == "consumed")
            if consumed:
                raise RuntimeError(
                    "a referenced checkpoint cannot name entries of an "
                    "origin-only batch committed with lifetime consumed: "
                    "write-only consumers own that lifetime; publish them with "
                    f"lifetime retain ({', '.join(consumed)})")
        return waited

    def commit_checkpoint_artifact(self, reservation_id, record, *, references=None):
        """Commit a writer receipt's ACTUAL bytes/digests against its reservation.

        Verifies every receipt-listed file exists at its receipted size plus
        the manifest, requires actuals within the reserved envelope, then
        moves the envelope into live checkpoint bytes and returns the
        commitment (envelope, actual, unused, receipt digest). The same
        reservation with an equal receipt commits idempotently; a different
        receipt, an unknown id, or a non-active reservation refuses. Failure
        verification retains the envelope (files may exist) instead of
        releasing it; reclaim explicitly.
        """
        if type(reservation_id) is not int:
            raise RuntimeError("exact boundary checkpoint reservation is not an integer")
        entry = self._checkpoint_reservations.get(reservation_id)
        if entry is None:
            raise RuntimeError("exact boundary checkpoint reservation is unknown")
        if not isinstance(record, dict):
            raise RuntimeError("exact boundary checkpoint record is not a mapping")
        digest = record.get("cotangent_sha256")
        if type(digest) is not str or not digest:
            raise RuntimeError("exact boundary checkpoint record carries no receipt digest")
        if entry["state"] == "committed":
            if entry["receipt_digest"] == digest:
                return dict(self._checkpoint_committed[reservation_id])
            raise RuntimeError(
                "exact boundary checkpoint reservation already committed a "
                "different receipt")
        if entry["state"] != "active":
            raise RuntimeError(
                "exact boundary checkpoint reservation is not active; "
                "reclaim it before any new attempt")

        def _fail(message):
            entry["state"] = "retained"
            entry['retained_files'] = self._checkpoint_retained_files(entry)
            if self._checkpoint_active == reservation_id:
                self._checkpoint_active = None
            raise RuntimeError(message)

        planned = {row["name"]: row for row in entry["files"]}
        rows = []
        for field in ("activation_entries", "shared_state_entries"):
            entries = record.get(field)
            if not isinstance(entries, list):
                _fail("exact boundary checkpoint record has no entry list "
                      f"{field!r}")
            rows.extend(entries)
        pins = {}
        if references is not None:
            # PQ #1036: the activation rows are this owner's own live
            # entries, verbatim; they are pinned, never counted as files
            # the attempt wrote.
            from .joint_adjoint_checkpoints import exact_entry_record

            activation = record["activation_entries"]
            if len(activation) != len(references):
                _fail("referenced checkpoint rows differ from its references")
            by_name = {}
            for reference in references:
                live = self._references.get(reference.name)
                if (live != reference
                        or reference in self._pinned_checkpoint_entries
                        or reference in self._checkpoint_inputs):
                    _fail("referenced checkpoint entry is not a live own entry: "
                          f"{reference.name}")
                by_name[reference.name] = reference
            for row in activation:
                reference = by_name.get(row.get("name")) if isinstance(row, dict) else None
                if reference is None or row != exact_entry_record(reference):
                    _fail("referenced checkpoint row is not its entry's record: "
                          f"{row.get('name') if isinstance(row, dict) else row}")
                try:
                    observed = Path(row["path"]).stat().st_size
                except OSError:
                    _fail(f"referenced checkpoint entry is not durable: {row['name']}")
                if observed != row["file_bytes"]:
                    _fail(f"referenced checkpoint entry size drifted: {row['name']}")
                pins[reference] = int(record["boundary"])
            rows = list(record["shared_state_entries"])
        names = [row["name"] for row in rows
                 if isinstance(row, dict) and type(row.get("name")) is str]
        if (len(names) != len(rows) or sorted(names) != sorted(planned)
                or len(set(names)) != len(names)):
            missing = sorted(set(planned) - set(names))
            extra = sorted(set(names) - set(planned))
            _fail("exact boundary checkpoint receipt does not match its "
                  f"reserved plan: missing={missing[:8]}, extra={extra[:8]}")
        for row in rows:
            expected = planned[row["name"]]
            if (type(row.get("path")) is not str
                    or row["path"] != expected["path"]):
                _fail("exact boundary checkpoint receipt path is not its "
                      f"reserved path: {row.get('name')}")
            if (type(row.get("file_bytes")) is not int
                    or row["file_bytes"] > expected["envelope_bytes"]):
                _fail("exact boundary checkpoint receipt file exceeds its "
                      f"reserved envelope: {row.get('name')}")
        from .cost_stage_checkpoint import canonical_json_sha256

        canonical_digest = canonical_json_sha256(
            {key: record[key] for key in
             ("schema", "boundary", "session", "activation_entries",
              "shared_state_entries")},
            where="adjoint checkpoint receipt",
        )
        if canonical_digest != digest:
            _fail("exact boundary checkpoint receipt digest does not match "
                  "its entry set")
        actual = 0
        for row in rows:
            try:
                observed = Path(row["path"]).stat().st_size
            except OSError:
                _fail("exact boundary checkpoint receipt file is missing: "
                      f"{row.get('name')}")
            if observed != row["file_bytes"]:
                _fail("exact boundary checkpoint receipt file size drifted: "
                      f"{row.get('name')}")
            actual += row["file_bytes"]
        manifest_path = Path(entry["dir"]) / "checkpoint.json"
        try:
            manifest_bytes = manifest_path.stat().st_size
        except OSError:
            _fail("exact boundary checkpoint manifest is missing at commit")
        actual += manifest_bytes
        if actual > entry["envelope_bytes"]:
            _fail("exact boundary checkpoint actual bytes exceed the reserved "
                  f"envelope ({actual} > {entry['envelope_bytes']})")
        unused = entry["envelope_bytes"] - actual
        if entry['produced_batch'] is not None:
            from .joint_adjoint_slices import checkpoint_manifest_entry
            manifest = checkpoint_manifest_entry(record)
            batch_id = entry['produced_batch']
            descriptors = [self._produced.descriptor_for(ProducedFileReference(
                row['path'], row['name'], row['file_bytes'], row['sha256']),
                producer_generation=batch_id, artifact_class='checkpoint')
                for row in [*rows, manifest]]
            self._produced.publish(batch_id=batch_id, descriptors=descriptors)
        entry["state"] = "committed"
        entry["receipt_digest"] = digest
        self._pinned_checkpoint_entries.update(pins)
        if self._checkpoint_active == reservation_id:
            self._checkpoint_active = None
        self.telemetry["live_checkpoint_bytes"] += actual
        self.telemetry["peak_checkpoint_bytes"] = max(
            self.telemetry["live_checkpoint_bytes"],
            self.telemetry["peak_checkpoint_bytes"])
        self.telemetry["checkpoint_envelope_unused_bytes"] += unused
        commitment = {"reservation": reservation_id,
                      "envelope_bytes": entry["envelope_bytes"],
                      "actual_bytes": actual, "unused_bytes": unused,
                      "receipt_digest": digest,
                      "checkpoint_dir": entry["dir"]}
        self._checkpoint_committed[reservation_id] = commitment
        if self._check_memory is not None:
            self._check_memory("checkpoint artifact publication")
        return dict(commitment)

    def abandon_checkpoint_artifact(self, reservation_id):
        """Retain a failed attempt's envelope: files may exist, so its bytes
        stay counted until an explicit reclaim disposes its directory."""
        if type(reservation_id) is not int:
            raise RuntimeError("exact boundary checkpoint reservation is not an integer")
        entry = self._checkpoint_reservations.get(reservation_id)
        if entry is None:
            raise RuntimeError("exact boundary checkpoint reservation is unknown")
        if entry["state"] == "committed":
            raise RuntimeError(
                "exact boundary checkpoint reservation is committed; "
                "committed bytes are final")
        entry["state"] = "retained"
        entry['retained_files'] = self._checkpoint_retained_files(entry)
        if self._checkpoint_active == reservation_id:
            self._checkpoint_active = None
        return None

    def cancel_checkpoint_artifact(self, reservation_id):
        """Release a reservation that created nothing (pre-write failure).

        Only an active reservation cancels: the attempt directory was never
        created, so no bytes can exist and the full envelope is released.
        Anything that may have written must abandon (retain) instead.
        """
        if type(reservation_id) is not int:
            raise RuntimeError("exact boundary checkpoint reservation is not an integer")
        entry = self._checkpoint_reservations.get(reservation_id)
        if entry is None:
            raise RuntimeError("exact boundary checkpoint reservation is unknown")
        if entry["state"] != "active":
            raise RuntimeError(
                "exact boundary checkpoint reservation is not active; "
                "only a pre-write attempt cancels cleanly")
        del self._checkpoint_reservations[reservation_id]
        if self._checkpoint_active == reservation_id:
            self._checkpoint_active = None
        return None

    def _dispose_retained_entry(self, reservation_id, entry):
        """Delete one retained attempt's own new directory, verified.

        Returns "deleted" after an owned deletion and "already_absent" when
        the directory is verifiably gone. Anything else -- a non-directory
        in its place, or a directory that survives removal -- retains the
        envelope and raises: bytes are only released against actual owned
        deletion or verified absence, never on uncertainty.
        """
        from .staged_manifest import unlink_owned
        directory = Path(entry["dir"])
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise RuntimeError(
                "exact boundary checkpoint reclaim refuses an unexpected "
                f"non-directory at {entry['dir']}; retaining its bytes")
        if not directory.exists():
            outcome = "already_absent"
        else:
            if (entry['retained_files'] is None or
                    self._checkpoint_retained_files(entry) != entry['retained_files']):
                raise RuntimeError('checkpoint retained roster changed; retaining its bytes')
            for path, identity in entry['retained_files'].items():
                unlink_owned(path, identity)
            entries = directory / 'entries'
            if entries.exists():
                entries.rmdir()
            directory.rmdir()
            if directory.exists():
                raise RuntimeError(
                    "exact boundary checkpoint reclaim could not dispose "
                    f"{entry['dir']}; retaining its bytes")
            outcome = "deleted"
        if entry['produced_batch'] is not None:
            released = self._produced.abort_prewrite(batch_id=entry['produced_batch'])
            if not released['ok']:
                raise RuntimeError('checkpoint prewrite retains owned files')
        del self._checkpoint_reservations[reservation_id]
        if self._checkpoint_active == reservation_id:
            self._checkpoint_active = None
        return outcome

    def reclaim_checkpoint_artifact(self, space, boundary):
        """Dispose a retained attempt's own new directory, then release it.

        Refuses when no retained reservation names that checkpoint directory:
        nothing is ever deleted blindly. Committed (durable published)
        checkpoints are never reclaimed through this path. Release follows
        verified deletion or verified absence only.
        """
        from .joint_adjoint_checkpoints import checkpoint_directory

        directory = checkpoint_directory(space, int(boundary))
        target = str(directory)
        reservation = None
        for reservation_id, entry in self._checkpoint_reservations.items():
            if entry["state"] == "retained" and entry["dir"] == target:
                reservation = reservation_id
                break
        if reservation is None:
            raise RuntimeError(
                "exact boundary checkpoint reclaim found no retained attempt at "
                f"{target}")
        outcome = self._dispose_retained_entry(
            reservation, self._checkpoint_reservations[reservation])
        return {"reservation": reservation, "checkpoint_dir": target,
                "reclaimed": True, "disposition": outcome}

    def checkpoint_commitment(self, receipt_digest):
        """Return the stored commitment for a receipt digest, if committed."""
        for commitment in self._checkpoint_committed.values():
            if commitment["receipt_digest"] == receipt_digest:
                return dict(commitment)
        return None

    def checkpoint_output_descriptor(self):
        """Application artifact accounting, independent of fleet admission."""
        reserved = sum(record["envelope_bytes"]
                       for record in self._checkpoint_reservations.values()
                       if record["state"] in ("active", "retained"))
        return {
            "schema": "prismaquant.boundary_artifact_output.v1",
            "unit": "bytes",
            "scope": "application artifact budget",
            "ceiling_bytes": self.config["max_artifact_bytes"],
            "live_ordinary_bytes": self.telemetry["live_artifact_bytes"],
            "live_checkpoint_bytes": self.telemetry["live_checkpoint_bytes"],
            "reserved_bytes": reserved,
            "note": ("Application-level artifact accounting; filesystem capacity "
                     "and fleet resource admission are independent."),
        }

    def _reclaim_retained_checkpoints(self):
        """Dispose every retained attempt directory at close and release it.

        Committed checkpoints are durable published state and survive close
        either way. Retained attempts never held a receipt, so their brand-new
        attempt directory (created only after a successful reservation) is
        safe to dispose here; anything else refuses loudly instead. A
        disposal that cannot verify absence fails close rather than leaking
        silently or deleting blindly.
        """
        for reservation_id, entry in list(self._checkpoint_reservations.items()):
            if entry["state"] != "retained":
                continue
            self._dispose_retained_entry(reservation_id, entry)

    def bind_produced_output(self, publication, *, group_size, n_batches,
                             max_entry_tensor_bytes, staging_timeout_s=900.0,
                             window_groups=None, read_order="probe_major",
                             origin_lifetime=None, batch_range=None,
                             dispose_on_failure=False):
        """Bind this session to one immutable, native-launch declaration."""
        from .produced_output_spool import ProducedOutputSpool
        if self._readonly or self.session is None or self._produced is not None:
            raise RuntimeError("produced binding requires a fresh writable bound session")
        for value in (group_size, n_batches, max_entry_tensor_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("positive produced geometry required")
        declaration = publication.declaration
        if (group_size != declaration["group_size"] or
                max_entry_tensor_bytes != declaration["max_entry_tensor_bytes"]):
            raise RuntimeError("runtime geometry differs from launch-bound declaration")
        if not publication.contains(self.directory / "entries"):
            raise RuntimeError("entry directory escapes declared output prefix")
        start, stop = (0, n_batches) if batch_range is None else batch_range
        if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= n_batches
                or start % group_size or (stop % group_size and stop != n_batches)):
            raise ValueError("produced batch range must contain whole groups")
        if publication.write_only:
            if origin_lifetime not in ("retain", "consumed") or window_groups not in (None, 0):
                raise ValueError("write-only publication requires lifetime and no read window")
            read_groups = window_groups = 0
        else:
            if origin_lifetime is not None or dispose_on_failure:
                raise ValueError("readback publication retains its origins on failure")
            if read_order not in ("probe_major", "sample_major"):
                raise ValueError("unknown produced read order")
            read_groups = 2 if read_order == "probe_major" else 1 + self._n_probes
            window_groups = declaration["window_groups"] if window_groups is None else window_groups
            if (type(window_groups) is not int or not read_groups <= window_groups <= declaration["window_groups"]):
                raise ValueError("read window exceeds declaration or cannot hold its required groups")
        publication.bind_session(self.session)
        self._produced = publication
        self._produced_plan = {"group_size": group_size, "n_batches": n_batches,
            "batch_start": start, "batch_stop": stop,
            "max_entry_tensor_bytes": max_entry_tensor_bytes,
            "staging_timeout_s": staging_timeout_s, "window_groups": window_groups,
            "read_groups": read_groups, "write_only": publication.write_only,
            "origin_lifetime": origin_lifetime, "dispose_on_failure": dispose_on_failure}
        self._local_output_spool = ProducedOutputSpool.from_publication(publication)
        if self._local_output_spool is not None and not self._published:
            raise RuntimeError("local output spool requires a published owner")
        if self._local_output_spool is not None and not publication.write_only:
            self._produced_plan["local_window_bytes"] = self._require_local_window(publication)

    def _require_local_window(self, publication):
        from .produced_output_spool import ProducedWindowRefused, two_plane_window_bytes
        plan, spool = self._produced_plan, self._local_output_spool
        need = two_plane_window_bytes(n_probes=self._n_probes,
            n_batches=plan["batch_stop"] - plan["batch_start"], group_size=plan["group_size"],
            group_ceiling=lambda entries: publication.group_ceiling_bytes(entries=entries,
                max_entry_tensor_bytes=plan["max_entry_tensor_bytes"]))
        if spool.max_bytes < need:
            raise ProducedWindowRefused(f"two cotangent planes need {need} B; declared window is {spool.max_bytes} B")
        import shutil
        if shutil.disk_usage(spool.root).free < need:
            raise ProducedWindowRefused("local disk lacks two-plane headroom")
        return need

    def _require_produced_copies(self):
        if self._local_output_spool is not None:
            with self._local_output_spool._cond:
                self._local_output_spool._check_failures()

    def _join_produced_copies(self):
        if self._local_output_spool is not None:
            self._local_output_spool.join_copies()

    @staticmethod
    def _produced_group_key(*, kind, batch_index, boundary_index, probe_index, group_size):
        return (str(kind), boundary_index, -1 if probe_index is None else probe_index,
                batch_index // group_size)

    def _produced_group_slots(self, key):
        kind, boundary, probe, group_index = key
        plan = self._produced_plan
        start = group_index * plan["group_size"]
        for batch in range(start, min(start + plan["group_size"], plan["batch_stop"])):
            slot = f"boundary-{batch}-{boundary}" if probe < 0 else f"cotangent-{probe}-{batch}"
            yield f"{slot}-at-{boundary}"

    def _produced_planned_paths(self, key):
        from .perturbed_x_cache import activation_cache_filename
        paths = []
        for name in self._produced_group_slots(key):
            final = self.directory / "entries" / activation_cache_filename(name)
            paths.extend((str(final), str(final.with_suffix(".pt.tmp"))))
        return paths

    def _produced_group_for_write(self, key):
        group = self._produced_groups.get(key)
        if group is not None:
            return group
        planned = self._produced_planned_paths(key)
        kind, boundary, probe, group_index = key
        batch = self._produced.batch_id_for(kind=kind, boundary_index=boundary,
            probe_index=None if probe < 0 else probe, group_index=group_index)
        ceiling = self._produced.group_ceiling_bytes(entries=len(planned) // 2,
            max_entry_tensor_bytes=self._produced_plan["max_entry_tensor_bytes"])
        self._produced.require_prewrite(batch_id=batch, payload_ceiling_bytes=ceiling,
            temp_ceiling_bytes=ceiling, paths=planned)
        if self._local_output_spool is not None:
            self._local_output_spool.reserve(batch, ceiling)
        group = {"batch_id": batch, "planned": planned, "references": [], "published": None,
            "context": None, "manifest_digest": None, "retired": False,
            "origin_reclaimed": False, "live_references": 0, 'file_identities': {}}
        self._produced_groups[key] = group
        self.telemetry["produced_groups_prewritten"] += 1
        return group

    def _produced_group_for(self, reference):
        key = self._produced_index.get(reference)
        return key, self._produced_groups.get(key)

    def stage_produced_boundary_ahead(self, boundary_index):
        references = [ref for group in self._produced_groups.values() for ref in group["references"]
            if json.loads(ref.metadata_json)["identity"]["kind"] == "boundary"
            and json.loads(ref.metadata_json)["identity"]["coordinates"]["boundary"] == boundary_index]
        self.stage_produced_reads_ahead(references)

    def stage_produced_reads_ahead(self, references):
        if self._local_output_spool is not None:
            self._local_output_spool.prepare_reads(
                ref for ref in references if ref in self._produced_index)

    @contextmanager
    def retain_produced_boundary(self, boundary_index):
        """Hold this plane's local aliases across its consuming windows."""
        if self._produced_retained_boundary is not None:
            raise RuntimeError("produced boundary retention does not nest")
        self._produced_retained_boundary = boundary_index
        try:
            references = [ref for key, group in self._produced_groups.items()
                if key[0] == 'boundary' and key[1] == boundary_index
                for ref in group['references']]
            if self._local_output_spool is None or not references:
                yield
            else:
                self._local_output_spool.prepare_reads(references)
                with self._local_output_spool.local_reads(references):
                    yield
        finally:
            self._produced_retained_boundary = None

    def retain_produced_reads(self, references):
        references = frozenset(ref for ref in references if ref in self._produced_index)
        if references == self._produced_retained_reads:
            return len(references)
        prior, self._produced_retained_read_context = self._produced_retained_read_context, None
        self._produced_retained_reads = frozenset()
        if prior is not None:
            prior.__exit__(None, None, None)
        if references and self._local_output_spool is not None:
            self._local_output_spool.prepare_reads(references)
            context = self._local_output_spool.local_reads(references)
            context.__enter__()
            self._produced_retained_read_context = context
            self._produced_retained_reads = references
        return len(references)

    def _publish_complete_produced_group(self, group):
        # The spool's existing IO future publishes complete groups. Direct
        # writers publish on this thread, with no scheduling or funding state.
        if self._local_output_spool is None:
            self._produced_commit_origin(group)

    def produced_group_records(self):
        return [{"batch_id": group["batch_id"], "entries": len(group["references"]),
            "manifest_digest": group["manifest_digest"], "retired": group["retired"],
            "origin_reclaimed": group["origin_reclaimed"]} for group in self._produced_groups.values()]

    def produced_output_report(self):
        if self._produced is None:
            return None
        return {"schema": "prismaquant.boundary_produced_output.v2",
            "producer": None if self._produced is None else self._produced.producer,
            "declaration": None if self._produced is None else self._produced.binding,
            "groups": self.produced_group_records(),
            "local_spool": None if self._local_output_spool is None else self._local_output_spool.report(),
            "deferred_unlinks": {batch: [ref.path for ref in refs] for batch, refs in self._deferred_unlinks.items()},
            "release_errors": list(self._produced_release_errors),
            "retained_uncommitted": list(self._produced_retained_uncommitted)}

    def _release_unpublished_prewrites(self, *, failing=False):
        if self._produced is None:
            return
        for group in self._produced_groups.values():
            if group.get("origin_ref") is None:
                if failing and self._produced_plan['dispose_on_failure']:
                    self._dispose_uncommitted_group(group)
                if (self._local_output_spool is not None and
                        self._local_output_spool.holds(group['batch_id'])):
                    self._produced_retained_uncommitted.append({
                        'batch_id': group['batch_id'], 'reason': 'owned local files remain'})
                    continue
                result = self._produced.abort_prewrite(batch_id=group["batch_id"])
                if not result["ok"]:
                    self._produced_retained_uncommitted.append({"batch_id": group["batch_id"], **result})

    def _dispose_uncommitted_group(self, group):
        """Dispose only this failed write-only writer's fenced file roster."""
        from .native_results import verify_result_artifact
        from .staged_manifest import unlink_owned
        batch, spool = group['batch_id'], self._local_output_spool
        if spool is not None:
            local = spool._groups[batch]
            if local['future'] is None:
                if any(os.path.lexists(path) for path in group['planned']):
                    raise RuntimeError('unsubmitted canonical paths changed; retaining prewrite')
                spool.discard_unsubmitted(batch)
                return
            elif not local['durable']:
                return  # The failed publication retains every source.
        if batch in self._produced._groups:
            from .stage_a_produced_output import read_bound_document
            rows = read_bound_document(self._produced._groups[batch])['entries']
        else:
            rows = [{'path': ref.path, 'bytes': ref.file_bytes, 'sha256': ref.sha256,
                'file_identity': group['file_identities'][ref.path]}
                for ref in group['references']]
        for row in rows:
            verify_result_artifact(row, max_bytes=1 << 32)
        for row in rows:
            unlink_owned(row['path'], row['file_identity'])
        group['origin_reclaimed'] = True

    def release_produced_group(self, reference):
        key, group = self._produced_group_for(reference)
        if group is not None:
            self._spool_retire_entry(reference)

    @contextmanager
    def prefetch(self, references):
        from .perturbed_x_cache import prefetch_exact_activation_cache_entries
        from contextlib import ExitStack
        if self._active_window is not None:
            raise RuntimeError("exact boundary windows may not overlap")
        references = tuple(references)
        for reference in references:
            self._entry_identity(reference)
        foreign = tuple(ref for ref in references
                        if ref in self._forward_inputs or ref in self._attached_forward_inputs)
        if foreign:
            from .joint_forward_resume import await_forward_inputs
            await_forward_inputs(foreign)
        owned = tuple(ref for ref in references if ref not in foreign)
        with ExitStack() as stack:
            local = {}
            if self._local_output_spool is not None:
                self._local_output_spool.prepare_reads(owned)
                local = stack.enter_context(self._local_output_spool.local_reads(owned))
            elif self._produced is not None:
                # This owner's durable whole files use its ordinary exact
                # reader, with the same inline digest and tensor/session gates.
                for ref in owned:
                    _key, group = self._produced_group_for(ref)
                    if group is None:
                        raise RuntimeError("owned reference is outside produced roster")
                    if len(group["references"]) != len(group["planned"]) // 2:
                        raise RuntimeError("readback cannot consume an incomplete group")
                    self._produced_commit_origin(group)
                    local[ref] = Path(ref.path)
            with torch.profiler.record_function("aura.exact_activation.prefetch"):
                if self._scratch is None:
                    from .perturbed_x_cache import EntryReadScratch
                    self._scratch = EntryReadScratch()
                context = prefetch_exact_activation_cache_entries(references,
                    expected_session=self.session, max_tensor_bytes=self.config["max_resident_bytes"],
                    residency_check=self._reserve, scratch=self._scratch, local_paths=local,
                    session_for_reference=(lambda ref: self._entry_identity(ref)["session"])
                        if self._forward_inputs or self._attached_forward_inputs else None)
                window = stack.enter_context(context)
            if local:
                self.telemetry["produced_local_windows"] += 1
                self.telemetry["produced_local_reads"] += len(local)
                self.telemetry["produced_local_read_bytes"] += sum(ref.file_bytes for ref in local)
            self._active_window = window
            self.telemetry["prefetch_windows"] += 1
            self.telemetry["read_tensor_bytes"] += sum(ref.tensor_bytes for ref in references)
            try:
                yield window
            finally:
                self._active_window = None

    def get(self, window, reference):
        self._entry_identity(reference)
        if window is not self._active_window:
            raise RuntimeError("exact boundary lookup has no active resident window")
        try:
            return window.get(reference)
        except RuntimeError:
            self.telemetry["hot_read_misses"] += 1
            raise

    def __exit__(self, exc_type, exc, traceback):
        self._status = "failed" if exc_type is not None else (
            "attached" if self._readonly else ("complete" if self.session else "unused"))
        cleanup_error = None
        try:
            # All asynchronous owners finish before source retention changes.
            self.retain_produced_reads(())
            self._join_produced_copies()
            if self._active_window is not None or self.telemetry["resident_tensor_bytes"]:
                raise RuntimeError("exact boundary generation closed with a live window")
            if self._checkpoint_active is not None:
                raise RuntimeError("exact boundary generation closed with an active checkpoint reservation")
            if exc is None:
                self.settle_local_output()
            elif self._local_output_spool is not None:
                self._local_output_spool.release_landed()
                self._produced_flush_deferred_unlinks(final=True)
            if self._published:
                self._references.clear()
                self._slots.clear()
            elif exc is None or self._produced is None:
                for reference in list(self._references.values()):
                    self._retire(reference, missing_ok=True)
                self._slots.clear()
            # Failed publications retain their sources, reservations and error.
            self._reclaim_retained_checkpoints()
            self._release_unpublished_prewrites(failing=exc is not None)
            if (self._produced is not None and self._published and
                    (exc is None or self._produced.declaration['checkpoint_exit'] is not None)):
                self._produced.finalize(Path(self._produced.producer['record_dir']) /
                    'produced-result.json', checkpoint_exit=(None if exc is None else
                        self._produced.declaration['checkpoint_exit']))
        except BaseException as error:
            self._status = "failed"
            cleanup_error = error
            self._produced_release_errors.append({"step": "close", "error": repr(error)})
            if exc is not None:
                exc.add_note(f"produced lifecycle cleanup also failed: {error!r}")
        finally:
            if self._cotangent_scratch is not None:
                self._cotangent_scratch.close()
                self._cotangent_scratch = None
            if self._scratch is not None:
                self._scratch.release()
            for batch in self._batches or ():
                batch.activations_cpu.clear()
                batch.input_ids = batch.position_ids = None
                batch.position_embeddings = batch.attention_mask = batch.shared_pass_state = None
            for row in self._cotangents or ():
                for cotangent in row:
                    cotangent.release_resident_state()
                row.clear()
            if self._batches is not None:
                self._batches.clear()
            if self._cotangents is not None:
                self._cotangents.clear()
            self._batches = self._cotangents = None
            self._check_memory = None
            try:
                self._publish_status()
            except BaseException as error:
                if exc is not None:
                    exc.add_note(f"generation status publication also failed: {error!r}")
                elif cleanup_error is None:
                    cleanup_error = error
                else:
                    cleanup_error.add_note(f"generation status publication also failed: {error!r}")
        if cleanup_error is not None and exc is None:
            raise cleanup_error
        return False

    def receipt(self):
        path = self.status_path()
        out = {"policy": self.identity, "session": self.session, "status": self._status,
                "generation_manifest": str(path) if path is not None else None,
                "working_artifacts_reusable": False, "telemetry": dict(self.telemetry)}
        retained = self._retained_debt()
        if retained is not None:
            out["retained"] = retained
        return out


def _boundary_window_references(batches, boundary_index, incoming, indices):
    """What one window of ``prefetched_boundary_batches`` reads, in order."""

    references = [batches[index].activations_cpu[boundary_index] for index in indices]
    if incoming is not None:
        references.extend(incoming[index] for index in indices)
    return references


def _fused_window_references(batches, boundary_index, incoming, indices):
    """What one window of ``prefetched_fused_boundary_windows`` reads, in order.

    The boundary entries of ``indices``, then each probe's incoming entries
    of the same indices, probe ascending. ``incoming`` is one entry list per
    probe, or ``None`` when the caller supplies incoming tensors itself.
    """

    references = [batches[index].activations_cpu[boundary_index] for index in indices]
    if incoming is not None:
        for entries in incoming:
            references.extend(entries[index] for index in indices)
    return references


def fused_window_size(*, prefetch_batches, max_resident_bytes, per_batch_bytes,
                      batch_size, write_bytes=0):
    """How many batches one fused window reads (RobTand/prismaquant#997).

    The fused roll holds, for every batch of a window, its boundary entry
    and every probe's incoming entry (``per_batch_bytes``) inside the sealed
    ``max_resident_bytes``, beside the one rolled entry being written
    (``write_bytes``; the writer reserves it while the window is open, as
    the probe-major roll's does). The window is the largest multiple of
    ``batch_size`` that fits and divides ``prefetch_batches``, so a window
    never straddles a produced group. Refuses when no window holds one
    whole batch group: that regime does not fit this run's sealed policy,
    and the policy is not changed to make it fit.
    """

    from .joint_adjoint_slices import ChainRegimeRefused

    group, batch_size = int(prefetch_batches), int(batch_size)
    per_batch_bytes = int(per_batch_bytes)
    if per_batch_bytes <= 0 or int(write_bytes) < 0:
        raise ChainRegimeRefused("a fused window needs positive entry sizes")
    fits = max(0, int(max_resident_bytes) - int(write_bytes)) // per_batch_bytes
    sizes = [size for size in range(batch_size, group + 1, batch_size)
             if group % size == 0 and size <= fits]
    if not sizes:
        raise ChainRegimeRefused(
            f"a fused window at batch size {batch_size} holds "
            f"{batch_size * per_batch_bytes} bytes; the sealed max_resident_bytes "
            f"{int(max_resident_bytes)} holds {fits} batches, and the window "
            f"must be a multiple of the batch size that divides "
            f"prefetch_batches {group}")
    return max(sizes)


def fused_window_batches(storage, batches, boundary_index, incoming, *, batch_size):
    """``fused_window_size`` for one roll, from the entries it will read."""

    if storage is None:
        return max(len(batches), 1)
    boundary = max(batch.activations_cpu[boundary_index].tensor_bytes
                   for batch in batches)
    per_batch = boundary
    if incoming is not None:
        per_batch += sum(max(entry.tensor_bytes for entry in entries)
                         for entries in incoming)
    # The rolled cotangent has the boundary's shape and dtype.
    return fused_window_size(
        prefetch_batches=storage.config["prefetch_batches"],
        max_resident_bytes=storage.config["max_resident_bytes"],
        per_batch_bytes=per_batch, batch_size=batch_size, write_bytes=boundary)


@contextmanager
def prefetched_fused_boundary_windows(storage, batches, boundary_index, incoming=None,
                                      *, window_batches, then=None, window_end=None):
    """Sample-major windows for the fused roll (RobTand/prismaquant#997).

    Each window reads ``window_batches`` boundary entries and, beside them,
    the same batches' incoming entries for every probe, so one layer forward
    serves every probe's backward. Yields ``(indices, boundary, incoming)``
    per window: ``boundary(index)`` and ``incoming(probe, index)`` return
    the window's verified CPU tensors (``incoming`` returns ``None`` when the
    caller supplies incoming tensors itself). Batch order is preserved.

    Staging as in ``prefetched_boundary_batches``: the next window's groups
    are asked for as soon as a window opens, and the groups the next window
    of the same pass reads again are kept staged across the exit. ``then``
    names the pass read after this one, ``(boundary_index, incoming)`` with
    ``incoming`` one list per probe; its first window is asked for from this
    pass's last window.

    ``window_end`` is called inside each window once the caller asks for
    the next one, before the window closes: work the caller deferred on the
    window's tensors finishes there, before any later window is asked for
    (RobTand/prismaquant#1162). It is not called when the caller stops early.
    """

    window_batches = int(window_batches)
    if window_batches < 1:
        raise ValueError("a fused window reads at least one batch")

    def iterate():
        lookahead = (None if storage is None
                     else getattr(storage, "stage_produced_reads_ahead", None))
        retain_next = (None if storage is None
                       else getattr(storage, "retain_produced_reads", None))
        count = len(batches)
        for start in range(0, count, window_batches):
            indices = range(start, min(start + window_batches, count))
            if storage is None:
                yield (indices,
                       lambda index: batches[index].activations_cpu[boundary_index],
                       lambda probe, index: (None if incoming is None
                                             else incoming[probe][index]))
                if window_end is not None:
                    window_end()
                continue
            references = _fused_window_references(
                batches, boundary_index, incoming, indices)
            again = ()
            if start + window_batches < count:
                following = _fused_window_references(
                    batches, boundary_index, incoming,
                    range(start + window_batches,
                          min(start + 2 * window_batches, count)))
                again = following
            elif then is not None:
                following = _fused_window_references(
                    batches, then[0], then[1], range(0, min(window_batches, count)))
            else:
                following = ()
            with storage.prefetch(references) as window:
                if retain_next is not None:
                    retain_next(again)
                if lookahead is not None and following:
                    lookahead(following)

                def boundary(index, window=window):
                    return storage.get(window, batches[index].activations_cpu[boundary_index])

                def incoming_of(probe, index, window=window):
                    if incoming is None:
                        return None
                    return storage.get(window, incoming[probe][index])

                yield indices, boundary, incoming_of
                if window_end is not None:
                    window_end()
    iterator = iterate()
    try:
        yield iterator
    finally:
        iterator.close()
        if storage is not None:
            retain_next = getattr(storage, "retain_produced_reads", None)
            if retain_next is not None:
                retain_next(())


@contextmanager
def prefetched_boundary_batches(storage, batches, boundary_index, incoming=None,
                                then=None, window_end=None):
    """Preserve original batch order while leasing exact tensors in windows.

    Once a window opens, the existing IO owner prepares the next window's
    own groups from their immutable canonical publication. ``then``
    names the pass read after this one, as ``(boundary_index, incoming)``:
    its first window is asked for from this pass's last window. Staging
    only: the order, the reads and the tensors are unchanged.

    ``window_end`` is called inside each window after its last batch, as in
    ``prefetched_fused_boundary_windows``.
    """
    def iterate():
        size = len(batches) if storage is None else storage.config["prefetch_batches"]
        lookahead = (None if storage is None
                     else getattr(storage, "stage_produced_reads_ahead", None))
        for start in range(0, len(batches), size):
            indices = range(start, min(start + size, len(batches)))
            if storage is None:
                for index in indices:
                    yield index, batches[index], batches[index].activations_cpu[boundary_index], (
                        None if incoming is None else incoming[index])
                if window_end is not None:
                    window_end()
            else:
                references = _boundary_window_references(
                    batches, boundary_index, incoming, indices)
                with storage.prefetch(references) as window:
                    if lookahead is not None:
                        if start + size < len(batches):
                            following = _boundary_window_references(
                                batches, boundary_index, incoming,
                                range(start + size,
                                      min(start + 2 * size, len(batches))))
                        elif then is not None:
                            following = _boundary_window_references(
                                batches, then[0], then[1],
                                range(0, min(size, len(batches))))
                        else:
                            following = ()
                        if following:
                            lookahead(following)
                    for index in indices:
                        yield index, batches[index], storage.get(window, batches[index].activations_cpu[boundary_index]), (
                            None if incoming is None else storage.get(window, incoming[index]))
                    if window_end is not None:
                        window_end()
    iterator = iterate()
    try:
        yield iterator
    finally:
        iterator.close()


#: What a selected-source consumer may use of its source runner (PQ #1316).
SELECTED_SOURCE_MEMBERS = ("source_layers", "num_layers", "layer_index_for_qname",
                           "selected_weight_specs", "snapshot_selected_weights",
                           "shutdown", "context")
