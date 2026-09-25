"""Shared layer-streaming execution adapter for cost stages.

This is intentionally a thin consumer of :class:`StreamingContext`.  It does
not own weights or maintain another cache: decoder residency, prefetch, and
unload all go through the existing streaming-model machinery.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import dataclasses
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import SimpleNamespace
from typing import Any, Iterator

import torch

from prismaquant.memory_management import reserve_allocation
from prismaquant.dev_mode import dev_mode_enabled, dev_warning, seal_check
from prismaquant.layer_streaming import (
    _call_layer,
    _compute_attention_mask,
    _compute_position_embeddings,
    _get_final_norm,
)
from .digests import DIRECT_ASCII_LAX


STREAMED_MODEL_IDENTITY_SCHEMA = "prismaquant.streamed_model.identity.v1"
STREAMED_MODEL_IDENTITY_CACHE_SCHEMA = (
    "prismaquant.streamed_model.identity_cache.v1"
)
STREAMED_MODEL_PORTABLE_CONTENT_SCHEMA = (
    "prismaquant.streamed_model.portable_content.v1"
)
_STREAMED_MODEL_CONFIG_PROVENANCE_FIELDS = frozenset({
    "_name_or_path",
    "transformers_version",
})


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
    placement and balancing stay PrismaBuild's: one campaign row per range
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

    #: Which thread drives an owner's PrismaBuild work, and so whose time its
    #: blocked spans are (PQ #1262). ``compute``: the quantum's compute
    #: thread, the GPU's cost. ``writer``: the handoff writer thread, which
    #: runs beside the passes, so its waits are not the GPU's.
    PRODUCED_DRIVERS = ("compute", "writer")

    def __init__(self, config, *, driven_by="compute"):
        if driven_by not in self.PRODUCED_DRIVERS:
            raise ValueError(f"unknown owner driver {driven_by!r}")
        self._produced_driver = driven_by
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
        # Produced-output binding (Stage A's own entries staged through PB).
        # None on every ordinary path: an unbound owner writes, reads and
        # retires exactly as it did before produced output existed.
        self._produced = None
        self._produced_plan = None
        self._local_output_spool = None
        self._produced_groups = {}
        # The refs of the groups a write-only owner committed at their origin
        # (PrismaBuild #912), in commit order: what a consumer declares.
        self._produced_origin_batches = []
        self._produced_release_errors = []
        self._produced_index = {}
        #: Retired entries whose canonical file waits for its group's local
        #: release, by batch id (PQ #1110). PrismaBuild's ``release_group``
        #: re-checks every landed destination, so the file must outlive it.
        self._deferred_unlinks = {}
        #: Groups read only on this box and kept past the owner's close, so
        #: their charge stays a prewrite (PQ #1110): what the receipt names.
        self._produced_retained_uncommitted = []
        # Each uncommitted group a failing exit disposed of (PQ #1251).
        self._produced_disposed_uncommitted = []
        self._produced_window_keys = ()
        self._produced_release_pending = {}
        self._produced_release_abandoned = {}
        self._produced_release_unclassified = {}
        # Groups whose PUBLISH never completed because PrismaBuild's
        # funding lock stayed contended for the whole staging budget.
        # Their prewrite credit is still held by this owner, so the
        # debt is reported rather than dropped.
        self._produced_publish_deferred = {}
        # Read-ahead (RobTand/prismaquant#887). ``_produced_held`` is every
        # group holding stage credit right now: published or re-staged and
        # not yet confirmed retired. ``_produced_ahead`` is the subset taken
        # opportunistically -- published at write-complete, staged ahead of
        # its read, or retained across probe passes -- and it never grows
        # past the sealed window minus the two groups the synchronous read
        # path may need, so that path always has credit and a full window is
        # backpressure rather than a refused publication.
        self._produced_held = set()
        self._produced_ahead = set()
        self._produced_ahead_refusals = []
        self._produced_retained_boundary = None
        # Probe fusion (RobTand/prismaquant#997) reads sample-major: every
        # window reads one boundary group and one incoming group per probe,
        # and the same groups again in the next window until the group
        # ends. ``_produced_retained_reads`` is what the next window of the
        # same pass reads again, kept staged across this window's exit.
        self._produced_retained_reads = frozenset()
        self._produced_read_order = "probe_major"
        # The read path's lookahead (RobTand/prismaquant#989): the groups
        # the NEXT window reads, asked for while this one computes. The set
        # is what that request still wants; a window that opens takes its
        # own keys out of it, and a newer request replaces it. The reserve
        # is the share every opportunistic admission above leaves free, so
        # the nearest read is never queued behind a group read a layer
        # later (``_produced_ahead_has_room``).
        self._produced_read_ahead_wanted = set()
        self._produced_lookahead_reserve = 0
        # The background stager (RobTand/prismaquant#895). ``None`` on every
        # unbound owner and at the default two-group window: there is then no
        # thread, and every step below runs where it is called, as before.
        # With a stager, this owner's PrismaBuild calls that can wait on a
        # lock run on its thread; the compute thread submits them and blocks
        # only for a group it must read now or a full queue. One lock guards
        # the bookkeeping both threads touch, and a thread gives it up around
        # each PrismaBuild call.
        from .produced_stager import OwnerLock
        self._produced_lock = OwnerLock()
        self._stager = None
        self._stager_stuck = False
        self._stager_failures = []
        self._stager_death_recorded = False
        self._stager_death_raised = False
        self._stager_preclaimed = set()
        #: One stager look at the oldest live export is queued (PQ #1128).
        self._export_poll_queued = False
        self._produced_release_queued = set()
        self._produced_live_keys = frozenset()
        import threading
        # Per thread: spans nest on the thread that opened them, and a frame
        # from another thread in the same stack would misattribute both.
        self._produced_blocked_local = threading.local()
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
            "produced_groups_prewritten": 0, "produced_groups_published": 0,
            "produced_file_groups": 0,
            "produced_groups_committed_at_origin": 0,
            # Seconds spent in PrismaBuild's origin commits (PQ #1225).
            "produced_commit_origin_s": 0.0,
            # Origins PrismaBuild's origin commit hashed again because their
            # timestamps moved after the spool's poll checked the export
            # receipt (PQ #1262, PB #1111). A move before that poll is
            # re-pinned in the spool's receipt, and no answer names it.
            "produced_landed_repins": 0,
            "produced_groups_materialized": 0, "produced_groups_retired": 0,
            "produced_groups_rematerialized": 0,
            "produced_group_release_failures": 0,
            "produced_group_release_retries": 0,
            "produced_group_release_deferrals": 0,
            "produced_group_funding_deferrals": 0,
            "produced_groups_origin_reclaimed": 0,
            "produced_groups_published_ahead": 0,
            "produced_groups_staged_ahead": 0,
            "produced_groups_retained": 0,
            "produced_group_ahead_refusals": 0,
            "produced_group_credit_waits": 0,
            "produced_group_stage_wait_s": 0.0,
            "produced_group_release_wait_s": 0.0,
            "produced_group_ahead_wait_s": 0.0,
            "produced_group_release_polls": 0,
            "produced_group_read_refunds": 0,
            "produced_groups_ahead_surrendered": 0,
            "produced_groups_prewritten_ahead": 0,
            "produced_group_fast_reads": 0,
            "produced_stager_tasks": 0,
            "produced_stager_busy_s": 0.0,
            "produced_stager_alive_s": 0.0,
            "produced_stager_queue_peak": 0,
            "produced_stager_urgent_tasks": 0,
            "produced_stager_urgent_delay_s": 0.0,
            "produced_stager_step_overruns": 0,
            "produced_stager_dropped": 0,
            "produced_stager_failures": 0,
            "produced_stager_requeues": 0,
            "produced_group_ahead_export_deferrals": 0,
            "produced_group_ahead_no_room": 0,
            "produced_read_ahead_requests": 0,
            "produced_groups_read_ahead": 0,
            "produced_read_ahead_deferrals": 0,
            "produced_read_ahead_missed": 0,
            # Same-box readback and write-behind (PQ #1110).
            "produced_local_reads": 0,
            "produced_local_read_bytes": 0,
            "produced_local_windows": 0,
            "produced_group_ahead_local_skips": 0,
            "produced_deferred_unlinks": 0,
            "produced_deferred_unlinks_done": 0,
            "produced_deferred_unlink_bytes": 0,
            "produced_groups_prewrite_released": 0,
            # Entry I/O off the compute thread (PQ #1128).
            "produced_export_polls": 0,
            "produced_compute_blocked_s": 0.0,
            **{f"produced_compute_blocked_{reason}_s": 0.0
               for reason in self.PRODUCED_BLOCKED_REASONS}}
        if driven_by == "writer":
            # The writer's own names (PQ #1262): the compute counters stay
            # zero, because the compute thread never waits on this owner.
            self.telemetry.update({
                "produced_writer_blocked_s": 0.0,
                **{f"produced_writer_blocked_{reason}_s": 0.0
                   for reason in self.PRODUCED_BLOCKED_REASONS}})

    #: Why the compute thread was blocked on this owner's PrismaBuild work.
    #: One telemetry counter each (``produced_compute_blocked_<reason>_s``),
    #: exclusive of each other, and ``produced_compute_blocked_s`` is their
    #: sum: the GPU's cost of staging, stated directly. Every key exists from
    #: the start, because the status file serializes the telemetry while the
    #: stager updates it and a dict may not change size under that.
    PRODUCED_BLOCKED_REASONS = (
        "prewrite", "publish_ahead", "stage_ahead", "read_fund", "stage_wait",
        "compose", "release", "copy_before_unlink", "reclaim_origin",
        "queue_full", "settle", "close")

    def __enter__(self):
        return self

    def bind(self, identity, *, n_probes, check_memory=None, published=False):
        """Start one generation; ``published`` keeps its entries on exit.

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
        by a stager that never joined.

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
        from .cost_stage_checkpoint import canonical_json, canonical_json_sha256
        if self.session is not None:
            raise RuntimeError("exact boundary generation is already bound")
        if (not isinstance(session, dict)
                or set(session) != {"generation", "run_identity_sha256"}):
            raise RuntimeError("a chain resume names no exact boundary session")
        session = canonical_json(dict(session), where="resumed exact boundary session")
        # A wall in dev mode too (PQ #1147): the digest covers the calibration
        # draw as well as the run seals. A dev chain resume compares the bind
        # identity key by key first and then rebinds the stored one.
        if canonical_json_sha256(identity, where="exact boundary source") != session[
                "run_identity_sha256"]:
            raise RuntimeError(
                "the relaunch's bind identity is not the one the resumed session sealed")
        directory = Path(self.config["directory"]) / str(session["generation"])
        status_path = directory / "generation.json"
        try:
            status = json.loads(status_path.read_bytes())
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"the resumed generation has no readable status file at {status_path}"
            ) from exc
        if status.get("session") != session:
            raise RuntimeError(
                f"{status_path} names another session than the chain resume")
        policy_refusal = RuntimeError(
            f"{status_path} was written under another boundary storage policy")
        if boundary_storage_layout_differs(status.get("policy"), self.identity):
            raise policy_refusal
        seal_check("boundary storage policy", status.get("policy"), self.identity,
                   where=str(status_path), refusal=policy_refusal)
        if status.get("status") not in ("running", "failed"):
            raise RuntimeError(
                f"the resumed generation's status is {status.get('status')!r}; "
                "only an interrupted run (running or failed) resumes")
        if not (directory / "entries").is_dir():
            raise RuntimeError(f"the resumed generation has no entries at {directory}")
        if owner_label is not None:
            if (type(owner_label) is not str
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", owner_label)):
                raise RuntimeError(
                    f"an owner label is a plain file name, not {owner_label!r}")
            self._owner_label = owner_label
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
        produced output, because a relaunch is a new PrismaBuild owner and
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
                    identity["owner"] != forward_recovery["original_owner"] or
                    identity["attempt"] != forward_recovery["original_attempt"]):
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
        """What a stuck stager still owns, for the status and the receipt.

        ``None`` on every run whose stager joined, so a clean generation
        record is unchanged.
        """

        if not self._stager_stuck:
            return None
        return {"reason": "stuck-stager-exit-retain",
                "detail": ("the stager thread did not end inside the staging "
                           "budget; teardown was skipped, so this generation "
                           "still owns its origins, checkpoint reservations "
                           "and produced-output credit"),
                "origins": len(self._references),
                "origin_bytes": self.telemetry["live_artifact_bytes"]}

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
        root = os.environ.get("PRISMAQUANT_STAGE_B_COTANGENT_ROOT")
        ceiling = os.environ.get("PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES")
        if root is None and ceiling is None:
            return {}
        if not root or not ceiling or not ceiling.isdecimal() or int(ceiling) <= 0:
            raise ValueError("cotangent scratch requires explicit root and positive max bytes")
        if self._cotangent_scratch is not None:
            raise RuntimeError("boundary owner already holds cotangent scratch")
        from .perturbed_x_cache import ExactCotangentScratch
        self._cotangent_scratch = ExactCotangentScratch(
            records, directory=root, max_bytes=int(ceiling),
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

        Publication is where a unit becomes durable, so it is the only place
        the count may move: PB #480 buys a long run time on committed work and
        a counter that ticked on intent would keep a wedged run alive. The
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
        if (self._local_output_spool is not None
                and nbytes > self._produced_plan["max_entry_tensor_bytes"]):
            raise RuntimeError("local output entry exceeds its sealed group tensor ceiling")
        remaining = self.checkpoint_remaining_bytes()
        if file_limit > remaining:
            raise RuntimeError(
                "exact boundary artifact budget exceeded: "
                f"entry needs {file_limit} bytes, {remaining} remain of "
                f"{self.config['max_artifact_bytes']}")
        name = f"{slot}-at-{boundary_index}"
        identity = {"session": self.session, "slot": slot, "kind": kind, "coordinates": coordinates}
        # BEFORE the first byte: a bound owner claims its group's durable
        # budget, or refuses with nothing written. The claim covers the
        # whole 64-entry group once, so the remaining writes in it cost no
        # further call, and it holds no stage credit -- the funding happens
        # at the commit inside the deferred publish.
        produced_group = None
        if self._produced is not None:
            self._produced_raise_stager_failure()
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
                # A spool entry is never the only copy of committed work, so
                # it is written without fsync (PQ #1225). Nothing counts it
                # until PrismaBuild acknowledges the export's own fsynced
                # copy, the export hashes every byte it copies against this
                # reference, and a same-box read (PQ #1110) checks the size
                # and the sha256 of every byte it reads. A crash that tears
                # the local file fails those checks; it never lands.
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
                produced_group["live_references"] += 1
                self._produced_index[reference] = produced_key
                complete = (len(produced_group["references"])
                            == len(produced_group["planned"]) // 2)
            if complete and self._local_output_spool is not None:
                self._local_output_spool.submit(produced_group["batch_id"])
            elif complete and self._produced_plan["write_only"]:
                # Written straight to its canonical paths, so it commits at
                # its origin now. Through the spool it commits once its
                # export is acknowledged (``settle_local_output``).
                self._produced_commit_origin(produced_group)
            if complete and read_back:
                # The group's last entry is durable, and everything its
                # publication needs is on the references already. Publishing
                # here lets PrismaBuild's mover run while the GPU works.
                # ``read_back=False`` is the writer saying no read follows
                # (the walk's last roll): staging that group would be a copy
                # nobody reads, racing the disposal of its origins.
                self._produced_publish_ahead(produced_key, produced_group)
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
        """Have the stager look at the oldest live export (PQ #1128).

        One look per write, oldest first (``ProducedOutputSpool.poll_oldest``),
        and never on the compute thread while a stager runs: before #1128
        every write polled every live export twice here. At most one look is
        queued at a time, in the stager's ordered lane, so it neither blocks
        the compute thread on a full optional lane nor piles up behind a
        slow export. A refused export raises on the stager and surfaces at
        the owner's next call, as every stager failure does. With no stager
        the look runs here, once.
        """

        from .produced_stager import ORDERED, StagerClosed

        spool = self._local_output_spool
        if spool is None or not spool.exporting():
            return
        stager = self._stager
        if stager is not None and not self._produced_on_stager():
            if self._export_poll_queued:
                return
            self._export_poll_queued = True

            def poll():
                self._export_poll_queued = False
                spool.poll_oldest(where="write-poll")

            def dropped():
                self._export_poll_queued = False

            try:
                stager.submit(poll, kind=ORDERED, label="export-poll",
                              on_drop=dropped)
                return
            except StagerClosed:
                self._export_poll_queued = False
        spool.poll_oldest(where="write-poll")

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
        otherwise last as long as PrismaBuild's evidence does. The wait then
        raises :class:`~prismaquant.produced_output_spool.
        ProducedOutputWaitCancelled` and the owner exits on its failure
        path. Without a spool there is no wait to end.
        """
        if self._local_output_spool is not None:
            self._local_output_spool.cancelled = predicate

    def settle_local_output(self):
        """Finish PB durable exports before a successful capture receipt.

        This is the barrier at the end of an action's writes (PQ #1110): it
        waits while each export is live and refuses at once when PrismaBuild
        reports one failed (:class:`~prismaquant.produced_output_spool.
        ProducedExportRefused`, recorded in the spool's report). Nothing may
        be read back through the spool after it.

        A write-only owner commits every complete group at its origin
        (PrismaBuild #912), in the order it wrote them, against the
        identities each export receipt recorded. A group already committed
        is not committed again. Each group commits as soon as its own export
        lands, while the later exports are still copying, so the commits
        overlap the wait instead of following it (PQ #1225).
        """
        if self._local_output_spool is not None:
            if self._produced_plan is not None and self._produced_plan["write_only"]:
                for group in list(self._produced_groups.values()):
                    if len(group["references"]) == len(group["planned"]) // 2:
                        self._local_output_spool.await_group(
                            group["batch_id"], where="drain")
                        self._produced_commit_origin(group)
            # The action's end: every export lands (a wait on each export's
            # own state, no clock, PQ #1110), no read follows here, so every
            # local copy goes, and then every retired entry's canonical file.
            self._local_output_spool.drain(release=True)
            self._produced_flush_deferred_unlinks(final=True)
            self._commit_local_output_progress()
            if self._produced_plan is not None and self._produced_plan["write_only"]:
                for group in list(self._produced_groups.values()):
                    if len(group["references"]) == len(group["planned"]) // 2:
                        self._produced_commit_origin(group)

    def _produced_commit_origin(self, group):
        """Commit one complete group of a write-only owner at its origin.

        The descriptors are the ones a staged publication would seal
        (``descriptor_for`` over the group's references, the batch id as
        the producer generation), so the manifest names exactly the files
        and digests the writer recorded. Through the spool, PrismaBuild
        checks them against the export it landed. Returns the batch ref.
        """
        import time

        if group.get("origin_ref") is not None:
            return group["origin_ref"]
        batch_id = group["batch_id"]
        descriptors = [self._produced.descriptor_for(
                           reference, producer_generation=batch_id)
                       for reference in group["references"]]
        lifetime = self._produced_plan["origin_lifetime"]
        started = time.monotonic()
        if self._local_output_spool is None:
            out = self._produced.commit_origin(
                batch_id=batch_id, descriptors=descriptors, lifetime=lifetime)
        else:
            out = self._local_output_spool.commit_origin(
                batch_id, descriptors, lifetime=lifetime)
        self.telemetry["produced_commit_origin_s"] += time.monotonic() - started
        # PrismaBuild names each origin its commit hashed again because its
        # timestamps moved after the spool's poll (PB #1111): a finding in
        # itself, and a read the commit paid for (PQ #1262).
        repins = len(out.get("landed_repins") or ())
        group["landed_repins"] = repins
        self.telemetry["produced_landed_repins"] += repins
        group["origin_ref"] = dict(out["ref"])
        self._produced_origin_batches.append(dict(out["ref"]))
        self.telemetry["produced_groups_committed_at_origin"] += 1
        return group["origin_ref"]

    def produced_origin_batches(self):
        """The refs of the groups committed at their origin, in commit order.

        Each is PrismaBuild's ``origin_batch_ref`` (#912): owner action,
        attempt nonce, template, batch id and manifest digest. It is what a
        consumer declares, and what pins the bytes it reads.
        """
        return [dict(ref) for ref in self._produced_origin_batches]

    def write_produced_files(self, files, *, kind, boundary_index):
        """Write small files into the generation directory as one group.

        ``files`` is an ordered list of ``(name, bytes)`` pairs. An owner
        bound to a produced-output publication writes them as one
        produced-output group (PQ #1015). It claims the group's exact
        payload before the first byte, with the final and ``.tmp`` names as
        its planned paths. Through the local spool the files land in the
        group's PB reservation and PrismaBuild exports them in list order.
        This method returns only after the export is acknowledged, so a
        caller that puts its record last can rely on the record's presence.
        Without the spool each file is written through its ``.tmp`` name
        and linked into place, in the same order.

        No read follows in this action, so a bound owner's template must be
        write-only (PrismaBuild #912), and the group is committed at its
        origin once its files are durable, with the owner's origin lifetime
        (PQ #1075): directly after the last file is linked, or through the
        spool once PrismaBuild acknowledged the export. A read-back template
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
                sha256=hashlib.sha256(payload).hexdigest())
            for name, payload in files]
        total = sum(ref.file_bytes for ref in references)
        # The files share max_artifact_bytes with the entries, as the
        # template's payload maximum does on the PrismaBuild side.
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
                "write-only produced-output template (PrismaBuild #912)")
        self._produced_raise_stager_failure()
        batch_id = self._produced.batch_id_for(
            kind=kind, boundary_index=boundary_index, group_index=0)
        planned = []
        for name in names:
            planned += [str(self.directory / name),
                        str(self.directory / name) + ".tmp"]
        self._produced.require_prewrite(
            batch_id=batch_id, payload_ceiling_bytes=total, paths=planned)
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
                spool.submit(batch_id)
                # The handoff record's barrier: on the export's own state,
                # never a clock (PQ #1110).
                spool.await_group(batch_id, where="produced-files")
        except BaseException:
            if spool is None or not spool.pending(batch_id):
                # PB proves every planned path absent before it releases
                # the claim, and retains it otherwise.
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
                # released, which PrismaBuild does only after the export
                # landed and which re-checks every landed destination. The
                # window bounds how many wait: a full window makes the next
                # reservation wait on the exports, never on a clock.
                # A group the window released for room while another of
                # its entries is still live keeps the file too (PQ #1236):
                # that entry's read publishes or restages the whole group,
                # and PrismaBuild stats every origin in it.
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
        PrismaBuild, or restages it, and both stat every origin the group's
        manifest names (``produced_output`` refuses ``descriptor-unstatable``
        or ``restage-origin-changed``). So no retired file of the group may
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
        the window until PrismaBuild's export lands. A released group keeps
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
            for reference in self._deferred_unlinks.pop(batch_id):
                Path(reference.path).unlink()
                self.telemetry["produced_deferred_unlinks_done"] += 1
                self.telemetry["produced_deferred_unlink_bytes"] -= reference.file_bytes
                self._produced_forget_origin(reference)

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
                # descriptors are PrismaBuild's and are not rewritten by this
                # owner disposing of its files.
                self._produced_index.pop(reference, None)

    def _produced_await_copy_before_unlink(self, reference):
        """Do not unlink an origin PrismaBuild's mover may still be reading.

        A group a read consumed was waited for (``context`` is set), so its
        copy is whole and its origins are the owner's to dispose of, as
        before. A group staged AHEAD of any read was only asked for: its
        mover opens the origin when it runs, not when it was published, so
        unlinking first hands the mover a missing source and the batch a
        failed materialization. Wait for the receipt, inside the staging
        budget. Not applied to the failed-run sweep (``missing_ok``), which
        must not sit on movers for a generation nothing will read.
        """

        import time

        with self._produced_lock.held():
            key = self._produced_index.get(reference)
            group = None if key is None else self._produced_groups.get(key)
            if (group is None or group["retired"]
                    or group["context"] is not None
                    or group.get("copy_awaited")):
                return
            if self._stager is not None:
                # A publication of this group may be queued or running on
                # the stager, and until it has run the owner cannot say
                # whether a mover will open these origins. Wait for it, then
                # ask the question below as before.
                with self._produced_blocked("copy_before_unlink"), (
                        self._produced_lock.yielded()):
                    idle = self._stager.wait_keys_idle(
                        (key,), labels=("publish-ahead", "stage-ahead",
                                        "read-ahead"),
                        timeout=self._produced_plan["staging_timeout_s"])
                if not idle:
                    reason = "publication did not become idle before origin retirement"
                    self._produced_release_errors.append(
                        {"batch_id": group["batch_id"],
                         "step": "retain-origin-copy-unresolved",
                         "reason": {"error": reason, "origin": reference.path}})
                    raise TimeoutError(reason)
            if (key not in self._produced_ahead or group["retired"]
                    or group["context"] is not None
                    or group.get("copy_awaited")):
                return
            started = time.monotonic()
            try:
                with self._produced_blocked("copy_before_unlink"), (
                        self._produced_lock.yielded()):
                    self._produced.await_materialized(
                        batch_id=group["batch_id"],
                        timeout_s=self._produced_plan["staging_timeout_s"])
                group["copy_awaited"] = True
            finally:
                self.telemetry["produced_group_stage_wait_s"] += (
                    time.monotonic() - started)

    def _reclaim_produced_origin_for_group(self, key, group):
        """Release a group's durable charge when both conditions hold.

        Group-keyed, because the reference-keyed gate below cannot reach
        this case: a group whose retirement was REFUSED at window exit has
        already had its entries unlinked and dropped from the lookup index,
        so when the drain later succeeds there is no live reference left to
        ask about. Without this, exactly the groups that hit the
        exceptional path would keep their charge forever -- the leak the
        gate exists to prevent, surviving in its own error branch.
        """

        if group.get("origin_reclaimed"):
            return
        if (self._local_output_spool is not None
                and group["published"] is None and group.get("origin_ref") is None):
            # Read only on this box (PQ #1110): never committed, so its
            # durable charge is still the prewrite, and it goes when the
            # group's last file does.
            self._release_produced_prewrite_if_final(key, group)
            return
        if not group["retired"]:
            return
        if group["live_references"]:
            return
        if self._produced_on_compute_with_stager():
            # The proof of absence is PrismaBuild's and takes its ownership
            # lock, so it runs on the stager. Kept at close: a durable
            # charge still has to be given back by a run that is ending.
            self._produced_submit(
                "optional", "reclaim-origin",
                lambda: self._reclaim_produced_origin_for_group(key, group),
                keys=(key,), reason="reclaim_origin", keep_on_close=True)
            return
        with self._produced_blocked("reclaim_origin"), (
                self._produced_lock.yielded()):
            out = self._produced.reclaim_origin(group["batch_id"])
        if group.get("origin_reclaimed"):
            return
        if out.get("ok"):
            group["origin_reclaimed"] = True
            self.telemetry["produced_groups_origin_reclaimed"] += 1
        else:
            self._produced_release_errors.append(
                {"batch_id": group["batch_id"], "step": "reclaim_origin",
                 "reason": {"refusal": out.get("refusal")}})
            self._produced_log(
                f"reclaim_origin refused for {group['batch_id']}: "
                f"{out.get('refusal')!r}")

    def _release_produced_prewrite_if_final(self, key, group):
        """Give back a never-committed group's prewrite once its files are gone.

        The same-box reads of PQ #1110 never publish a group this box still
        held, so a plane rolled away here is never committed: its durable
        charge is its prewrite, not a batch. Without this every such plane
        would keep its ceiling charged until the owner closes, and the
        instance's durable maxima would refuse a later prewrite. PrismaBuild
        proves every planned path absent before it lets the claim go
        (``abort_prewrite``), and a refusal is recorded, never retried.
        """

        if (group.get("origin_reclaimed") or group["live_references"]
                or len(group["references"]) < len(group["planned"]) // 2
                or self._local_output_spool.pending(group["batch_id"])):
            return
        if self._produced_on_compute_with_stager():
            self._produced_submit(
                "optional", "release-prewrite",
                lambda: self._release_produced_prewrite_if_final(key, group),
                keys=(key,), reason="reclaim_origin", keep_on_close=True)
            return
        with self._produced_blocked("reclaim_origin"), (
                self._produced_lock.yielded()):
            out = self._produced.abort_prewrite(batch_id=group["batch_id"])
        if group.get("origin_reclaimed"):
            return
        if out.get("ok"):
            group["origin_reclaimed"] = True
            self.telemetry["produced_groups_prewrite_released"] += 1
        else:
            self._produced_release_errors.append(
                {"batch_id": group["batch_id"], "step": "release_prewrite",
                 "reason": {"refusal": out.get("refusal")}})
            self._produced_log(
                f"abort_prewrite refused for {group['batch_id']}: "
                f"{out.get('refusal')!r}")

    def _reclaim_produced_origin_if_final(self, reference):
        """Free a group's DURABLE charge once its last origin file is gone.

        Two conditions, both required. This owner must have disposed of
        every entry it wrote into the group -- the cotangent roll replaces
        a plane entry by entry, so the charge is only releasable when the
        LAST one goes -- and the group's stage copy must already be
        retired, because a live materialization is material PrismaBuild is
        still holding over those origins. The first condition is the
        owner's own count of entries it has not yet unlinked, not a sweep
        of the group's paths: the sweep re-stat'd every already-deleted
        path on every retirement, and it was never the proof anyway.

        Why it matters at Stage A scale: the reverse walk rolls a
        cotangent plane across 45 boundaries. Without a group-final
        reclaim, every replaced plane keeps its durable charge for the life
        of the instance and the origin class fills with bytes that are not
        there any more. Note what this is NOT: the capture's origin peak is
        the boundaries it RETAINS, not the sum of every cotangent it ever
        wrote, so this gate is what keeps the two the same number.

        The proof of absence stays PrismaBuild's own (``reclaim_origin``
        lstats every filed origin path and retains on a present or
        unstatable one). This only decides when it is worth asking, and a
        refusal is recorded rather than retried into a loop.
        """

        key, group = self._produced_group_for(reference)
        if group is None:
            return
        self._reclaim_produced_origin_for_group(key, group)

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
        self._next_checkpoint_reservation += 1
        self._checkpoint_reservations[reservation] = {
            "envelope_bytes": envelope_bytes, "files": files,
            "manifest_bytes": file_plan["manifest_bytes"],
            "temp_overlap_bytes": file_plan["temp_overlap_bytes"],
            "dir": str(directory), "label": str(label), "state": "active",
            "receipt_digest": None,
        }
        self._checkpoint_active = reservation
        self.telemetry["checkpoint_reservations"] += 1
        return reservation

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
        entry["files"] = list(entry["files"]) + list(files)
        entry["envelope_bytes"] += delta
        entry["temp_overlap_bytes"] = temp_overlap_bytes
        return entry["envelope_bytes"]

    def await_checkpoint_references(self, references):
        """Wait until every referenced own entry is durable at its origin.

        A referenced checkpoint (PQ #1036) names the owner's canonical entry
        paths. Through a local output spool those land when PrismaBuild's
        export of their group is acknowledged; the checkpoint's manifest is
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
                continue  # Filed in no PB batch, so no PB lifetime governs it.
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
            # Fail closed (PQ #1036): the pin lives in PQ, and PrismaBuild's
            # retirement tick unlinks a ``consumed`` origin-only batch once
            # its consumers succeed. Stage A commits staged batches today;
            # a move to origin batches must file them ``retain``.
            lifetimes = self._produced.origin_only_lifetimes()
            consumed = sorted(batch_id for batch_id in batches
                              if lifetimes.get(batch_id) == "consumed")
            if consumed:
                raise RuntimeError(
                    "a referenced checkpoint cannot name entries of an "
                    "origin-only batch committed with lifetime consumed: "
                    "PrismaBuild retires those origins; commit them with "
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
        import shutil

        directory = Path(entry["dir"])
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise RuntimeError(
                "exact boundary checkpoint reclaim refuses an unexpected "
                f"non-directory at {entry['dir']}; retaining its bytes")
        if not directory.exists():
            outcome = "already_absent"
        else:
            shutil.rmtree(directory)
            if directory.exists():
                raise RuntimeError(
                    "exact boundary checkpoint reclaim could not dispose "
                    f"{entry['dir']}; retaining its bytes")
            outcome = "deleted"
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
        """Application-budget output descriptor for a future PB output lane.

        Reports this owner's exact-artifact ceiling, live ordinary bytes,
        live checkpoint bytes, and reserved bytes in bytes. This enforces
        the APPLICATION artifact budget only: it is not a PrismaBuild
        shared SSD/RAM reservation and it stages no outputs. A future PB
        output descriptor (the PB732/liveness scope) may consume these
        counts and must fund movement separately; no admission is bypassed
        from here.
        """
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
            "note": ("Application-level accounting only: not a PrismaBuild "
                     "shared SSD/RAM reservation and not staged output."),
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

    # ------------------------------------------------------------------
    # Produced-output binding: this owner's own entries, staged by PB.
    #
    # Without it a Stage A action writes its boundary entries and then
    # refuses to read them: the strict allowed-tier path resolves the
    # process residency map, which is the map of the run's SEALED INPUTS,
    # and an output this action just produced is not in it. There is no
    # own-session exemption, so the entries go through PrismaBuild's
    # produced-output lifecycle like any other staged object.
    #
    # The unit is the EXISTING 64-entry logical publication group -- the
    # same window ``prefetched_boundary_batches`` already yields -- so a
    # group is prewritten once, published once, staged by one PB mover and
    # read back through one namespaced reader context.
    #
    # Per-entry movers were considered and REJECTED, and the reason is the
    # ledger's own arithmetic rather than the size of an entry. At Stage A
    # scale one boundary ENTRY is ~16 MiB and one 64-entry GROUP is ~1 GiB.
    # PrismaBuild prices a stage reservation PER MOVER and rounds UP to a
    # whole token: ``storage_tiers.stage_tokens_for_bytes`` returns
    # ``-(-range_bytes // GIB)``, deliberately, because a token is the unit
    # PB can refuse on and rounding down would leave the last partial GiB
    # of every reservation unaccounted. So 64 movers of ~16 MiB each cost
    # 64 tokens for ~1 GiB of bytes, while the one group that already is
    # the read window costs ceil(group_bytes / GiB) -- 2 tokens once the
    # zip envelopes are counted. That is the ~32x, and it is on top of
    # replacing 64 sealed requests, admissions, claims and fragments with
    # one for a single window the reader was going to take whole anyway.
    #
    # The LAST group is smaller and is priced as what it is: the ceiling
    # rule applies to its own byte range (``group_ceiling_bytes`` over the
    # entries that group actually holds), never to an assumed full 64.
    #
    # PUBLISH IS DEFERRED TO THE FIRST READ. ``require_prewrite`` charges
    # only the durable class budget and holds no ledger tokens, while the
    # commit inside ``publish_prepaid_batch`` funds the stage window by
    # exact transfer. Publishing at write time would therefore spend the
    # whole stage credit on groups nothing has asked for yet -- the first
    # boundary's writes must all land under the durable maxima WITHOUT
    # consuming the stage credits the first read needs.
    # ------------------------------------------------------------------

    def bind_produced_output(self, publication, *, group_size, n_batches,
                             max_entry_tensor_bytes,
                             staging_timeout_s=900.0, window_groups=None,
                             read_order="probe_major", origin_lifetime=None,
                             batch_range=None, dispose_on_failure=False):
        """Stage this generation's entries through ``publication``.

        Called after :meth:`bind`, because the entry directory this owner
        already chose is what must sit inside the publication's bound
        output prefix -- an own-generation path outside it is refused here
        rather than at the first descriptor. ``group_size`` is the read
        window's batch count (``config['prefetch_batches']``, the 64-entry
        group), ``n_batches`` bounds the last, partial group, and
        ``max_entry_tensor_bytes`` is the bound per-entry tensor ceiling
        the prewrite's conservative per-group ceiling is derived from.

        ``staging_timeout_s`` is ONE budget for a group's whole staging,
        not one per step. A read opens a single absolute deadline and the
        publish, any re-materialization and the wait for PrismaBuild's own
        mover receipt all spend it: a funding transient waited out inside
        the publish shortens the receipt wait rather than extending the
        total, and nothing in that span mints a second budget. Exceeding it
        is a named failure and a withdrawal -- ``BoundaryStagingTimeout``
        when the mover never finished, ``BoundaryProducedFundingDeferred``
        when PrismaBuild's funding lock never cleared -- never a direct
        origin read and never a second publication.

        Two PrismaBuild transients are waited out inside that budget and
        nothing else. On the publish side, ``funding-race-deferred`` is the
        pool's typed answer for a contended transition lock, and the
        re-drive is the documented identical-input idempotent call. On the
        retirement side, an own-copy deferral is named by PrismaBuild's own
        egress receipt through :func:`classify_egress_outcome`. Every other
        refusal is terminal here and returns at once.

        ``window_groups`` is how many groups the sealed template's window
        funds at once. ``None`` reads it from the publication's own template
        (``window_gib`` over one group's whole-GiB ceiling) and falls back to
        two, the geometry ``build_boundary_template`` seals by default. Two
        is the synchronous loop exactly as it was: publish at first read,
        retire and wait at window exit. Every group beyond two is read-ahead
        credit (RobTand/prismaquant#887): a group is published when its last
        entry lands, a layer's input boundary groups stay staged across the
        probe passes that re-read them, the next layer's plane is staged
        ahead, and a window exit asks for a retirement without waiting for
        it. None of that may exceed the sealed window, because exceeding it
        is a refused publication and not backpressure, so two groups' worth
        stays reserved for the read path and the rest is taken only while
        it is free.

        ``read_order`` is the order the reverse roll reads in.
        ``"probe_major"`` (the default) is one boundary group and one
        incoming cotangent group per window, probe outer. ``"sample_major"``
        is the fused roll (RobTand/prismaquant#997): one boundary group and
        one incoming group PER PROBE in every window, so a window reads
        ``1 + n_probes`` groups at once, and that many, not two, are the
        read path's own share of the sealed window.

        A read-only attached generation can never take this binding: an
        attached owner does not write, so it has nothing to declare and its
        entries stay ordinary input-map reads.

        A **write-only** publication (PrismaBuild #912, the band-serial
        handoff since PQ #1075) funds no window and is never read back:
        ``window_groups`` and ``read_order`` do not apply, every entry is
        written with ``read_back=False``, and each complete group is
        committed at its origin as a batch another action declares. Without
        the local spool the group commits when its last entry lands.
        Through the spool it commits once PrismaBuild acknowledges its
        export, in :meth:`settle_local_output`. ``origin_lifetime`` is the
        lifetime of those commits (#914) and is required for a write-only
        publication and refused for any other: ``retain`` or ``consumed``.

        ``dispose_on_failure`` (write-only only; the band-serial handoff,
        PQ #1251): when the owner exits on a failure, it removes the files of
        each of its own uncommitted groups, exactly the group's planned
        paths, before it aborts the group's prewrite. PrismaBuild's
        ``abort_prewrite`` proves every planned path absent and leaves that
        disposal to the producer; without it, a group whose export landed
        keeps its prewrite and its files. A group whose export is still live
        is left alone: its files may still land.

        ``batch_range`` (a chain split quantum, PQ #738) is the ``(start,
        stop)`` of global batches this owner writes, out of ``n_batches``.
        ``start`` is a whole number of groups and ``stop`` is too, or is
        ``n_batches``, so every group this owner writes is its own. A group
        slot, a prewrite ahead of the writer and the local window stay
        inside the range: a claim on the next group would claim another
        owner's paths.
        """

        if self._produced is not None:
            raise RuntimeError("exact boundary produced output is already bound")
        if self._readonly:
            raise RuntimeError(
                "an attached read-only generation cannot declare a produced "
                "output owner: it writes nothing and its entries resolve "
                "through the ordinary input map")
        if self.session is None or self.directory is None:
            raise RuntimeError(
                "bind the exact boundary generation before its produced output")
        for name, value in (("group_size", group_size),
                            ("n_batches", n_batches),
                            ("max_entry_tensor_bytes", max_entry_tensor_bytes)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"produced output {name} must be a positive int")
        batch_start, batch_stop = (0, n_batches) if batch_range is None else batch_range
        if (type(batch_start) is not int or type(batch_stop) is not int
                or not 0 <= batch_start < batch_stop <= n_batches
                or batch_start % group_size
                or (batch_stop % group_size and batch_stop != n_batches)):
            raise ValueError(
                f"produced output batch_range {batch_range!r} is not whole groups of "
                f"{group_size} inside {n_batches} batches")
        entries = self.directory / "entries"
        if not publication.contains(entries):
            raise RuntimeError(
                f"the exact boundary entry directory {entries} is outside the "
                f"bound output prefix {publication.output_prefix}: an "
                "own-generation path must sit inside the prefix its owner "
                "declared")
        self._produced = publication
        if not isinstance(staging_timeout_s, (int, float)) or (
                staging_timeout_s <= 0):
            raise ValueError(
                "produced output staging_timeout_s must be a positive number")
        write_only = getattr(publication, "write_only", False) is True
        if write_only:
            if origin_lifetime not in ("retain", "consumed"):
                raise ValueError(
                    "a write-only produced output commits each group at its "
                    "origin: name the commit's lifetime, 'retain' or "
                    f"'consumed', not {origin_lifetime!r}")
            if window_groups not in (None, 0):
                raise ValueError(
                    "a write-only produced output funds no window: its owner "
                    "never reads its groups back")
            # Nothing is read back, so nothing is staged ahead either.
            window_groups = read_groups = 0
        else:
            if origin_lifetime is not None:
                raise ValueError(
                    "only a write-only produced output commits its groups at "
                    "their origin; a read-back one stages them for its reads")
            if dispose_on_failure:
                raise ValueError(
                    "only a write-only produced output disposes of its own "
                    "uncommitted files on failure: a read-back one's groups "
                    "are read on this box")
            if window_groups is None:
                window_groups = self._sealed_window_groups(
                    publication, group_size=int(group_size),
                    max_entry_tensor_bytes=int(max_entry_tensor_bytes))
            if type(window_groups) is not int or window_groups < 2:
                raise ValueError(
                    "produced output window_groups must be an int of at least 2: "
                    "one read window holds a boundary group and a cotangent group")
            if read_order == "probe_major":
                read_groups = 2
            elif read_order == "sample_major":
                read_groups = 1 + int(self._n_probes)
            else:
                raise ValueError(f"unknown produced read order {read_order!r}")
            if window_groups < read_groups:
                raise ValueError(
                    f"the sealed window funds {window_groups} groups; a "
                    f"{read_order} read window holds {read_groups} at once "
                    "(one boundary group and one incoming group per probe)")
        self._produced_read_order = read_order
        self._produced_plan = {"group_size": int(group_size),
                               "n_batches": int(n_batches),
                               "batch_start": int(batch_start),
                               "batch_stop": int(batch_stop),
                               "max_entry_tensor_bytes": int(max_entry_tensor_bytes),
                               "staging_timeout_s": float(staging_timeout_s),
                               "window_groups": int(window_groups),
                               "read_groups": read_groups,
                               "ahead_groups": int(window_groups) - read_groups,
                               "write_only": write_only,
                               "origin_lifetime": (str(origin_lifetime)
                                                   if write_only else None),
                               "dispose_on_failure": bool(dispose_on_failure)}

        from .produced_output_spool import ProducedOutputSpool
        self._local_output_spool = ProducedOutputSpool.from_publication(
            publication)
        if self._local_output_spool is not None and not self._published:
            raise RuntimeError("local output spool requires a published Stage A owner")
        if self._local_output_spool is not None and not write_only:
            self._produced_plan["local_window_bytes"] = (
                self._require_local_window(publication))
        self._produced_groups = {}
        self._produced_start_stager()

    def _require_local_window(self, publication):
        """Refuse, before any byte, a spool that cannot hold two planes.

        The reverse chain reads the cotangent plane it wrote one layer
        earlier from this box's own copy (PQ #1110), while it writes the
        next: one plane is live and one more is the room its writes and
        exports turn over in. The need is derived from this owner's bound
        geometry through the publication's own per-group ceiling, the unit
        PrismaBuild reserves (``group_ceiling_bytes``), and refused against
        PrismaBuild's sealed window and the spool disk's free space. Returns
        the need in bytes.
        """

        import os
        from .produced_output_spool import (ProducedWindowRefused,
                                            two_plane_window_bytes)
        plan = self._produced_plan
        # The owner's own batches: a split quantum holds its range's planes.
        need = two_plane_window_bytes(
            n_probes=int(self._n_probes),
            n_batches=plan["batch_stop"] - plan["batch_start"],
            group_size=plan["group_size"],
            group_ceiling=lambda entries: publication.group_ceiling_bytes(
                entries=entries,
                max_entry_tensor_bytes=plan["max_entry_tensor_bytes"]))
        spool = self._local_output_spool
        sealed = spool.max_bytes
        if sealed is not None and sealed < need:
            raise ProducedWindowRefused(
                f"the sealed local output window is {sealed} B; two cotangent "
                f"planes of {self._n_probes} probes x "
                f"{plan['batch_stop'] - plan['batch_start']} "
                f"batches need {need} B. Seal the producer's spool bound at "
                "the plan's two-plane window (tools/dispatch_joint_quanta.py "
                "stage_a_spool_window_bytes)")
        root = spool.root
        if root is not None:
            stat = os.statvfs(root)
            free = stat.f_bavail * stat.f_frsize
            if free < need:
                raise ProducedWindowRefused(
                    f"the local output spool disk at {root} has {free} B free; "
                    f"the two-plane window needs {need} B")
        return need

    # ------------------------------------------------------------------
    # The background stager (RobTand/prismaquant#895).
    #
    # Measured on the first production run with read-ahead: the GPU was busy
    # 18 percent of the forward pass, because every PrismaBuild call ran on
    # the thread that drives it. With a sealed window wider than two groups
    # the owner now starts one stager thread, and these rules hold:
    #
    # * The compute thread makes no PrismaBuild call that can wait on a lock.
    #   It polls a group's mover receipt and composes its reader context
    #   (both lock-free reads), and it submits everything else.
    # * A group the window must read now is an URGENT task the compute
    #   thread waits for. A window's retirement asks are ORDERED. Both run
    #   first in, first out, ahead of OPTIONAL work: publishing at
    #   write-complete, staging a plane ahead, claiming the next group's
    #   prewrite, reclaiming a durable charge.
    # * In steady state a read submits nothing: a group that is published,
    #   holds credit and has no retirement asked for goes straight to the
    #   receipt poll (``produced_group_fast_reads``).
    # * One task runs at a time, so the state machine below runs on one
    #   thread and is the code the synchronous loop runs. An urgent task
    #   waits for the task in flight: ``produced_stager_urgent_delay_s``.
    # * A failure never disappears. An optional step stays a counted refusal
    #   with its reason. An urgent step raises on the compute thread, under
    #   its own type. Any other step's failure is kept and raised by the
    #   owner's next call.
    # ------------------------------------------------------------------

    #: ``inline`` keeps every step on the calling thread at any window
    #: width: the synchronous read-ahead loop of #887, for a run that must
    #: rule the thread out. Anything else, or unset, is the default.
    PRODUCED_STAGER_ENV = "PRISMAQUANT_STAGEA_STAGER"

    #: How long one opportunistic step may run on the stager. It bounds the
    #: wait of an urgent task behind it, so it is not the staging budget.
    PRODUCED_STAGER_STEP_BUDGET_S = 120.0

    #: How often the stager asks again about retirements still in flight.
    #: Each ask is a PrismaBuild call under the ownership lock, and the
    #: synchronous loop asked once a window, so this is about a window at
    #: production size and not the two seconds a record is paced at.
    PRODUCED_STAGER_POLL_S = 5.0

    #: Tests only: wait for every submitted task where it is submitted, so
    #: the read-ahead rules can be asserted step by step on the real thread.
    _PRODUCED_STAGER_WAIT_ALL = False

    def _produced_start_stager(self):
        import os
        from .produced_stager import ProducedStager

        if (self._produced_plan["ahead_groups"] <= 0
                or os.environ.get(self.PRODUCED_STAGER_ENV, "") == "inline"):
            return
        self._stager = ProducedStager(
            name="stagea-produced-stager",
            capacity=4 * self._produced_plan["window_groups"],
            poll=self._produced_stager_poll,
            poll_s=self.PRODUCED_STAGER_POLL_S,
            on_done=self._produced_stager_done,
            on_error=self._produced_stager_error)
        self._produced_log(
            f"stager thread started: window {self._produced_plan['window_groups']} "
            f"groups, read-ahead {self._produced_plan['ahead_groups']}")

    def _produced_on_stager(self):
        import threading
        stager = self._stager
        return stager is not None and threading.get_ident() == stager.ident

    def _produced_on_compute_with_stager(self):
        return (self._stager is not None and not self._produced_on_stager()
                and not getattr(self._produced_blocked_local, "inline", 0))

    def _produced_log(self, message):
        """One line when it happens; the receipt carries the totals."""

        print(f"exact boundary owner: {message}", flush=True)

    @contextmanager
    def _produced_blocked(self, reason):
        """Count the driving thread's time inside this span under ``reason``.

        Exclusive: a span nested in another is taken out of the outer one,
        so the reasons add up to ``produced_compute_blocked_s``. The stager
        thread's time is never counted here; it is ``produced_stager_busy_s``.
        An owner the handoff writer drives (``driven_by="writer"``, PQ #1262)
        counts the same spans as ``produced_writer_blocked_<reason>_s`` and
        ``produced_writer_blocked_s``: the writer's time, not the GPU's.
        """

        import time

        if self._produced_on_stager():
            yield
            return
        stack = getattr(self._produced_blocked_local, "stack", None)
        if stack is None:
            stack = self._produced_blocked_local.stack = []
        frame = [time.monotonic(), 0.0]
        stack.append(frame)
        try:
            yield
        finally:
            stack.pop()
            elapsed = time.monotonic() - frame[0]
            own = max(elapsed - frame[1], 0.0)
            if stack:
                stack[-1][1] += elapsed
            prefix = f"produced_{self._produced_driver}_blocked"
            self.telemetry[f"{prefix}_{reason}_s"] += own
            self.telemetry[f"{prefix}_s"] += own

    def _produced_submit(self, kind, label, call, *, keys=(), reason,
                         wait=False, on_drop=None, keep_on_close=False):
        """Run ``call`` on the stager, or here when there is none.

        ``wait`` is for the step the compute thread cannot go on without;
        its exception is raised here, as it was. Without ``wait`` the result
        is ``None`` and a failure is surfaced by the next call.
        """

        from .produced_stager import StagerClosed

        if self._produced_on_compute_with_stager():
            wait = wait or self._PRODUCED_STAGER_WAIT_ALL
            try:
                with self._produced_lock.yielded():
                    with self._produced_blocked("queue_full"):
                        task = self._stager.submit(
                            lambda: self._produced_run_task(call),
                            kind=kind, label=label, keys=keys,
                            on_drop=on_drop, keep_on_close=keep_on_close,
                            waited=wait)
                    if not wait:
                        return None
                    with self._produced_blocked(reason):
                        return task.wait()
            except StagerClosed:
                # The stager is gone (closing, or it died): the step still
                # has to happen, so it happens here, as it did before.
                pass
        # Per thread: a step running here must not submit its own nested
        # steps, and that says nothing about what another thread may do.
        local = self._produced_blocked_local
        local.inline = getattr(local, "inline", 0) + 1
        try:
            with self._produced_blocked(reason), self._produced_lock.held():
                return call()
        finally:
            local.inline -= 1

    def _produced_run_task(self, call):
        with self._produced_lock.held():
            return call()

    def _produced_stager_done(self, task):
        """Stager thread: fold one finished task into the telemetry."""

        stager = self._stager
        with self._produced_lock.held():
            # Time spent running, over every run: a task that gave the lane
            # back while it waited was not busy in between (PQ #989).
            ran = task.busy_s
            if task.label == "export-poll":
                # One per write while an export is live (PQ #1128): counted
                # on its own, so the task count stays the staging steps.
                self.telemetry["produced_export_polls"] += 1
            elif task.label != "poll":
                self.telemetry["produced_stager_tasks"] += 1
            self.telemetry["produced_stager_busy_s"] += max(ran, 0.0)
            self.telemetry["produced_stager_requeues"] += task.requeues
            self.telemetry["produced_stager_queue_peak"] = max(
                self.telemetry["produced_stager_queue_peak"],
                stager.queue_peak if stager is not None else 0)
            if task.kind == "urgent":
                self.telemetry["produced_stager_urgent_tasks"] += 1
                self.telemetry["produced_stager_urgent_delay_s"] += max(
                    (task.started or 0.0) - task.submitted, 0.0)
            if (task.kind == "optional"
                    and ran > self.PRODUCED_AHEAD_PUBLISH_BUDGET_S):
                self.telemetry["produced_stager_step_overruns"] += 1
                overrun = (f"stager step {task.label} took {ran:.1f}s, over the "
                           f"{self.PRODUCED_AHEAD_PUBLISH_BUDGET_S:.0f}s a "
                           "synchronous step was given")
            else:
                overrun = None
        if overrun is not None:
            self._produced_log(overrun)

    def _produced_stager_error(self, task, exc):
        """Stager thread: a task nobody waits for failed. Keep it."""

        with self._produced_lock.held():
            self.telemetry["produced_stager_failures"] += 1
            self._stager_failures.append((task.label, exc))
        self._produced_log(
            f"stager step {task.label} failed and will be raised by the "
            f"owner's next call: {exc!r}")

    def _produced_record_stager_death(self):
        """Record a dead stager worker as owner-visible debt, once.

        A worker that dies takes its whole queue with it. The stranded
        tasks are dropped with their own callbacks on the way out, and
        the death itself is news the owner cannot act on silently: a
        stranded reclaim is a durable charge nobody gives back (PQ #959).
        Returns the death, or None. Never raises, so the close path can
        call it too.
        """

        stager = self._stager
        death = None if stager is None else stager.death()
        if death is None or self._stager_death_recorded:
            return death
        self._stager_death_recorded = True
        stranded = stager.stranded()
        self._produced_release_errors.append(
            {"batch_id": None, "step": "stager-died",
             "reason": {"error": repr(death), "stranded": list(stranded)}})
        dropped = stager.stranded_error()
        if dropped is not None:
            self._produced_release_errors.append(
                {"batch_id": None, "step": "stager-died-drop",
                 "reason": {"error": repr(dropped)}})
        self._produced_log(
            f"the stager thread died: {death!r}; it stranded "
            f"{len(stranded)} queued step(s) "
            f"({', '.join(stranded) if stranded else 'none'}), each dropped "
            "with its own callback; this owner runs its PrismaBuild calls "
            "itself from here")
        return death

    def _produced_raise_stager_failure(self):
        """Raise the oldest kept stager failure, under its own type.

        A dead worker is raised the same way, after the failures it
        already kept: the owner hears about the death at its next call
        instead of finding a silently empty lane (PQ #959).
        """

        if self._produced_on_stager():
            return
        death = self._produced_record_stager_death()
        if self._stager_failures:
            with self._produced_lock.held():
                label, exc = self._stager_failures.pop(0)
            exc.add_note(f"raised on the produced-output stager thread by its "
                         f"{label} step, and surfaced here")
            raise exc
        if death is None or self._stager_death_raised:
            return
        self._stager_death_raised = True
        raise death

    def _produced_stager_poll(self):
        with self._produced_lock.held():
            if self._produced_release_pending:
                self._drain_produced_releases()

    def drain_produced_stager(self, timeout=None):
        """Wait until the stager has nothing queued or running.

        Returns True when that held inside ``timeout``. A kept failure is
        raised here like at any other call. With no stager this is a no-op.
        """

        stager = self._stager
        idle = True if stager is None else stager.drain(timeout)
        self._produced_raise_stager_failure()
        return idle

    def _produced_stop_stager(self):
        """Stop taking work, finish what must finish, join. Never raises.

        Queued optional steps are dropped, each publication as a counted
        refusal: a mover for a group nobody will read is not worth starting
        as the owner closes. Retirement asks and charge reclaims still run.
        A successful join restores synchronous cleanup. If the worker is
        still alive, its handle and all owned resources remain retained;
        the caller must skip teardown until a later close can join it.
        """

        stager = self._stager
        if stager is None:
            return
        with self._produced_blocked("close"):
            try:
                joined = stager.close(timeout=float(
                    self._produced_plan["staging_timeout_s"]) + 60.0)
            except BaseException as exc:                # noqa: BLE001
                joined = not stager.alive()
                self._produced_release_errors.append(
                    {"batch_id": None, "step": "stager-close",
                     "reason": {"error": repr(exc)}})
        import time
        self.telemetry["produced_stager_alive_s"] = round(
            time.monotonic() - stager.started, 3)
        self.telemetry["produced_stager_queue_peak"] = max(
            self.telemetry["produced_stager_queue_peak"], stager.queue_peak)
        if not joined:
            # A PrismaBuild call is still running on that thread and this
            # one must not run the same state machine beside it.
            self._stager_stuck = True
            self._produced_release_errors.append(
                {"batch_id": None, "step": "stager-close",
                 "reason": {"error": "the stager thread did not end inside "
                            "the staging budget; teardown was skipped"}})
            self._produced_log(
                "stager thread did not end inside the staging budget; the "
                "teardown at exit is skipped; origins and credit remain owned "
                "and are reported as debt")
            # Keep the ownership handle while its callbacks can still run.
            # Do not drain its failures or dispose of any shared state here.
            return
        self._stager_stuck = False
        self._produced_record_stager_death()
        self._stager = None
        for label, exc in self._stager_failures:
            self._produced_release_errors.append(
                {"batch_id": None, "step": f"stager:{label}",
                 "reason": {"error": repr(exc)}})
        self._stager_failures = []

    def _produced_group_for_write(self, key):
        """The group an entry is written into, its prewrite already claimed.

        With a stager the claim runs there and this waits for it. The next
        group of the same plane is claimed in the background as soon as this
        one takes its first entry: the writer reaches it a group of compute
        later, and a claim is a PrismaBuild call under the ownership lock.
        """

        with self._produced_lock.held():
            group = self._produced_groups.get(key)
            if group is None:
                group = self._produced_submit(
                    "urgent", "prewrite", lambda: self._produced_prewrite(key),
                    keys=(key,), reason="prewrite", wait=True)
            if (self._stager is None or group["references"]
                    or self._PRODUCED_STAGER_WAIT_ALL):
                # Step by step there is no "ahead of the writer": the claim
                # would land before the write that the tests count it at.
                return group
            following = (key[0], key[1], key[2], key[3] + 1)
            plan = self._produced_plan
            if (following[3] * plan["group_size"] < plan["batch_stop"]
                    and following not in self._produced_groups
                    and following not in self._stager_preclaimed):
                self._stager_preclaimed.add(following)
                self._produced_submit(
                    "optional", "prewrite-ahead",
                    lambda: self._produced_prewrite_ahead(following),
                    keys=(following,), reason="prewrite")
            return group

    def _produced_prewrite_ahead(self, key):
        """Stager: claim a group the writer has not reached. Never raises.

        A refusal here costs nothing: the writer asks again at the group's
        first entry and is refused there, under the prewrite's own type.
        """

        try:
            if key not in self._produced_groups:
                # Never waits for room: the stager's lane is not held on an
                # export (PQ #989), and the writer claims it again anyway.
                self._produced_prewrite(key, wait=False)
                self.telemetry["produced_groups_prewritten_ahead"] += 1
        except Exception as exc:                        # noqa: BLE001
            self._produced_log(
                f"prewrite ahead of the writer was refused for {key!r}; the "
                f"writer will ask again: {exc!r}")

    @staticmethod
    def _sealed_window_groups(publication, *, group_size, max_entry_tensor_bytes):
        """Groups the publication's sealed window funds at once; 2 if unknown."""

        try:
            window_gib = int(publication.template["working_demands"][
                publication.tier]["window_gib"])
            per_group = -(-int(publication.group_ceiling_bytes(
                entries=group_size,
                max_entry_tensor_bytes=max_entry_tensor_bytes)) // (1 << 30))
        except (AttributeError, KeyError, TypeError, ValueError):
            return 2
        return max(window_gib // max(per_group, 1), 2)

    @staticmethod
    def _produced_group_key(*, kind, batch_index, boundary_index, probe_index,
                            group_size):
        """The logical publication group one entry belongs to.

        Keyed by the read window, not by the file: a boundary plane's 64
        batches at one boundary are one group, and so is the cotangent
        plane's 64 at one boundary for one probe. A cotangent rollover
        writes a NEW canonical name at a new boundary, so it is a new
        group -- never a second writer of a filed path.
        """

        return (str(kind), int(boundary_index),
                -1 if probe_index is None else int(probe_index),
                int(batch_index) // int(group_size))

    def _produced_group_slots(self, key):
        """Every canonical entry name the group will write, in batch order."""

        kind, boundary_index, probe, group_index = key
        plan = self._produced_plan
        start = group_index * plan["group_size"]
        stop = min(start + plan["group_size"], plan["batch_stop"])
        for batch_index in range(start, stop):
            slot = (f"boundary-{batch_index}-{boundary_index}" if probe < 0
                    else f"cotangent-{probe}-{batch_index}")
            yield f"{slot}-at-{boundary_index}"

    def _produced_planned_paths(self, key):
        """The group's planned durable-origin superset: finals and temps.

        Both spellings, because the writer creates the ``.pt.tmp`` staging
        file and renames it: the prewrite must have charged it before it
        exists, and the rename makes it absent again, which is exactly what
        the commit's planned-omitted-absent proof checks. The names are the
        existing writer's own -- nothing is copied, renamed or hashed to
        satisfy this.
        """

        from .perturbed_x_cache import activation_cache_filename
        directory = self.directory / "entries"
        paths = []
        for name in self._produced_group_slots(key):
            final = directory / activation_cache_filename(name)
            paths.append(str(final))
            paths.append(str(final.with_suffix(".pt.tmp")))
        return paths

    def _produced_prewrite(self, key, *, wait=True):
        """Claim the group's durable budget before its first byte lands.

        Through the spool the group's local ceiling is reserved too. A full
        window makes room from what no read here needs, and otherwise waits
        on a live export (``wait``) or refuses (PQ #1110).
        """

        group = self._produced_groups.get(key)
        if group is not None:
            return group
        planned = self._produced_planned_paths(key)
        kind, boundary_index, probe, group_index = key
        batch_id = self._produced.batch_id_for(
            kind=kind, boundary_index=boundary_index,
            probe_index=None if probe < 0 else probe, group_index=group_index)
        ceiling = self._produced.group_ceiling_bytes(
            entries=len(planned) // 2,
            max_entry_tensor_bytes=self._produced_plan["max_entry_tensor_bytes"])
        with self._produced_lock.yielded():
            self._produced.require_prewrite(
                batch_id=batch_id, payload_ceiling_bytes=ceiling,
                paths=planned)
        if self._local_output_spool is not None:
            with self._produced_lock.yielded():
                self._local_output_spool.reserve(batch_id, ceiling, wait=wait)
        group = {"batch_id": batch_id, "planned": planned,
                 "references": [], "published": None, "context": None,
                 "manifest_digest": None, "retired": False,
                 "origin_reclaimed": False, "live_references": 0}
        self._produced_groups[key] = group
        self.telemetry["produced_groups_prewritten"] += 1
        return group

    def _produced_group_for(self, reference):
        """Which bound group holds this reference, or None.

        Indexed. Its standing reason is identity, not speed: this is the
        canonical reference -> group mapping and not a second store --
        the groups still own their reference lists, and an entry leaves
        the index when its origin is unlinked, so a dead reference
        answers None instead of resolving to a group whose bytes are
        gone.

        It is also faster than what it replaced, measured on both sides
        through PrismaBuild on dl380g10 at the production panel's shape
        (1563 groups of 64, ~100k rotated cotangent entries, twenty
        64-entry windows), because a loop shape is not evidence. The
        previous implementation walked every bound group asking
        ``reference in group["references"]``, so a late window cost
        O(groups x entries): **0.537 s** per window and 128M ``__eq__``
        calls, in front of a window that reads 1 GiB. This implementation
        costs **19.4 us** per window on the same fixture. Beside it,
        honestly: against a bare dict written inline it measures 1.046x,
        so it carries ~4.6% overhead over an idealized control -- that
        ratio is NOT the before/after delta, which is against the walk.
        Narrow CPU metadata on a fixture; no GPU, throughput, energy or
        whole-model claim follows from it.
        """

        key = self._produced_index.get(reference)
        if key is None:
            return None, None
        return key, self._produced_groups[key]

    def _produced_publish(self, key, group, deadline=None):
        """Publish the group once, when a read first asks for it.

        ``deadline`` is the absolute instant this group's WHOLE staging
        must end by. It is passed down rather than re-derived so a funding
        transient waited out inside the publish is spent from the same
        budget the materialization wait then gets the remainder of.
        """

        from .stage_a_produced_output import BoundaryProducedFundingDeferred

        if group["published"] is not None:
            return group
        if self._local_output_spool is not None:
            # A group read through PrismaBuild is one this box no longer
            # holds, so its export has landed and this returns at once; the
            # staging budget below governs PrismaBuild's steps, not the
            # export (PQ #1110).
            with self._produced_lock.yielded():
                self._local_output_spool.await_group(
                    group["batch_id"], where="publish")
        descriptors = [self._produced.descriptor_for(
            reference, producer_generation=group["batch_id"])
            for reference in group["references"]]
        if not descriptors:
            raise RuntimeError(
                f"produced boundary group {group['batch_id']!r} has no "
                "written entry to publish")
        group["manifest_digest"] = self._produced.manifest_digest_for(descriptors)
        before = int(getattr(self._produced, "funding_deferrals", 0))
        try:
            with self._produced_lock.yielded():
                published = self._produced.publish(
                    batch_id=group["batch_id"], descriptors=descriptors,
                    deadline=deadline)
            group["published"] = published
        except BoundaryProducedFundingDeferred as exc:
            # The budget is spent and the batch is NOT published: nothing
            # was reserved and nothing transferred, so this owner still
            # holds the group's prewrite credit. Recorded in its own bucket
            # before the failure propagates, because a held window that
            # nothing names is the invisible half of a leak.
            self._produced_publish_deferred[key] = {
                "batch_id": group["batch_id"], "step": exc.step,
                "attempts": exc.attempts, "waited_s": exc.waited_s,
                "timeout_s": exc.timeout_s, "outcome": exc.outcome}
            self._produced_log(
                f"publication of {group['batch_id']} deferred on funding for "
                f"{exc.waited_s:.1f}s of {exc.timeout_s:.1f}s "
                f"({exc.attempts} attempts)")
            raise
        finally:
            self.telemetry["produced_group_funding_deferrals"] += max(
                int(getattr(self._produced, "funding_deferrals", 0)) - before,
                0)
        self.telemetry["produced_groups_published"] += 1
        return group

    #: How long an opportunistic publication may wait out PrismaBuild's
    #: funding transient. Short on purpose: it runs on the writer, and the
    #: read that needs the group publishes it under the full staging budget
    #: if this gives up.
    PRODUCED_AHEAD_PUBLISH_BUDGET_S = 10.0

    def _produced_ahead_has_room(self, *, holding=False, lookahead=False):
        """May one more group count against the read-ahead share?

        Two bounds, both required. The share itself, and the WHOLE window:
        a group whose retirement PrismaBuild refused still holds its credit
        without being read-ahead, so counting only the share would let
        read-ahead spend the two groups the read path is owed.
        ``holding`` is a group that already holds credit (a window's own
        group being kept), which adds nothing to the window.

        ``lookahead`` is the read path's own lookahead, the groups the next
        window reads (PQ #989). Every other admission is opportunistic and
        leaves the lookahead's reserve free: a group published at write
        time is read a layer later, and it must not take the slot the next
        window's read needs now.
        """

        plan = self._produced_plan
        reserve = 0 if lookahead else self._produced_lookahead_reserve
        return (len(self._produced_ahead) + reserve < plan["ahead_groups"]
                and len(self._produced_held) + (0 if holding else 1)
                + reserve <= plan["window_groups"] - plan["read_groups"])

    def _produced_ahead_deadline(self, started):
        """The instant an optional step that began at ``started`` must end.

        On the stager nobody is waiting for the step, so it gets a budget
        that rides out a loaded fleet; on the calling thread it keeps the
        short one, because there the GPU waits for it.
        """

        budget = (self.PRODUCED_STAGER_STEP_BUDGET_S
                  if self._produced_on_stager()
                  else self.PRODUCED_AHEAD_PUBLISH_BUDGET_S)
        return budget, started + min(
            budget, float(self._produced_plan["staging_timeout_s"]))

    def _produced_take_ahead(self, key, group, *, restage, deadline=None,
                             lookahead=False):
        """Publish or re-stage one group ahead of its read. Never raises.

        Opportunistic by construction: it runs only inside the read-ahead
        share of the sealed window, and a refusal leaves the group exactly
        as it was, for the read to publish under the full budget. A group
        that already holds credit is left alone. ``deadline`` is the step's
        own, fixed when a step that gave the lane back first ran; without
        it the budget starts now. ``lookahead`` is the read path's own step
        (``_produced_ahead_has_room``).
        """

        import time

        if key in self._produced_held:
            return False
        if not self._produced_ahead_has_room(lookahead=lookahead):
            # Counted, not silent: a step skipped for want of room leaves
            # the group to its read.
            self.telemetry["produced_group_ahead_no_room"] += 1
            return False
        started = time.monotonic()
        budget, own_deadline = self._produced_ahead_deadline(started)
        if deadline is None:
            deadline = own_deadline
        # Counted before the call, not after it: while the call runs without
        # the lock, the other thread's credit arithmetic must see this group.
        self._produced_held.add(key)
        self._produced_ahead.add(key)
        try:
            if restage:
                with self._produced_lock.yielded():
                    self._produced.ensure_batch_materialized(
                        batch_id=group["batch_id"], deadline=deadline)
                group["retired"] = False
                group["context"] = None
                group["copy_awaited"] = False
                self.telemetry["produced_groups_rematerialized"] += 1
            else:
                self._produced_publish(key, group, deadline=deadline)
        except Exception as exc:                        # noqa: BLE001
            # Every refusal, not a named few: this step is optional, so no
            # outcome of it may end the capture. A funding refusal reserved
            # nothing; a budget that ran out part way left at most a
            # content-addressed publication the read re-drives with
            # identical inputs under the whole staging budget. Either way
            # the read decides, and raises under its own name if it must.
            self._produced_publish_deferred.pop(key, None)
            self.telemetry["produced_group_ahead_refusals"] += 1
            self._produced_held.discard(key)
            self._produced_ahead.discard(key)
            adopted = self._produced_adopt_after_refusal(key, group)
            self._produced_ahead_refusals.append(
                {"batch_id": group["batch_id"], "restage": bool(restage),
                 "adopted": adopted, "reason": repr(exc)})
            del self._produced_ahead_refusals[:-self.PRODUCED_AHEAD_REFUSAL_LOG]
            self._produced_log(
                f"read-ahead {'re-staging' if restage else 'publication'} of "
                f"{group['batch_id']} refused after "
                f"{time.monotonic() - started:.1f}s of a {budget:.0f}s budget "
                f"(adopted={adopted}); its read will stage it: {exc!r}")
            return False
        finally:
            self.telemetry["produced_group_ahead_wait_s"] += (
                time.monotonic() - started)
        return True

    def _produced_adopt_after_refusal(self, key, group):
        """Count credit PrismaBuild took before a read-ahead step failed.

        A step that fails after its funding moved leaves a mover PrismaBuild
        will run and an owner that thinks it holds nothing. Ask, and count
        what the ledger says: over-counting costs a little read-ahead,
        under-counting costs a refused read. The group's own state is left
        for its read, which re-drives the identical, content-addressed step
        and is answered with a typed duplicate.
        """

        try:
            with self._produced_lock.yielded():
                state = self._produced.materialization_state(
                    batch_id=group["batch_id"])
        except Exception:                               # noqa: BLE001
            return False
        if not (isinstance(state, dict) and state.get("ok")
                and not state.get("stage_retired")):
            return False
        if group["retired"]:
            # A re-staging PrismaBuild did start: the copy is coming back,
            # so the group is not retired, and nothing has awaited it yet.
            group["retired"] = False
            group["context"] = None
            group["copy_awaited"] = False
        self._produced_held.add(key)
        self._produced_ahead.add(key)
        return True

    #: How many read-ahead refusals are kept for the report. The count is
    #: exact in telemetry; the reasons are a bounded tail.
    PRODUCED_AHEAD_REFUSAL_LOG = 32

    def produced_ahead_refusals(self):
        """The most recent read-ahead steps PrismaBuild did not take."""

        with self._produced_lock.held():
            return [dict(entry) for entry in self._produced_ahead_refusals]

    def produced_output_report(self):
        """What a receipt should carry about this owner's staging, or None.

        Meant to be read AFTER the owner closed, so it includes the settle:
        the counters, the stage copies PrismaBuild would not retire, the
        read-ahead steps it did not take, and the release errors recorded
        along the way (a bounded tail of each list; the counts are exact).
        """

        if self._produced is None or self._produced_plan is None:
            return None
        with self._produced_lock.held():
            return {
                "local_spool": (self._local_output_spool.report()
                                if self._local_output_spool is not None else None),
                "window_groups": self._produced_plan["window_groups"],
                "ahead_groups": self._produced_plan["ahead_groups"],
                **({"read_order": self._produced_read_order,
                    "read_groups": self._produced_plan["read_groups"]}
                   if self._produced_read_order != "probe_major" else {}),
                "telemetry": {name: value
                              for name, value in self.telemetry.items()
                              if name.startswith("produced_")},
                "release_debt": self.produced_release_debt(),
                "ahead_refusals": self.produced_ahead_refusals(),
                "release_errors": [
                    {**entry, "reason": repr(entry.get("reason"))[:400]}
                    for entry in self._produced_release_errors[
                        -self.PRODUCED_AHEAD_REFUSAL_LOG:]],
                "release_error_count": len(self._produced_release_errors),
                "retained_uncommitted": [
                    dict(entry) for entry in self._produced_retained_uncommitted[
                        -self.PRODUCED_AHEAD_REFUSAL_LOG:]],
                "retained_uncommitted_count": len(
                    self._produced_retained_uncommitted),
                **({"disposed_uncommitted": [
                        dict(entry) for entry in self._produced_disposed_uncommitted]}
                   if self._produced_plan.get("dispose_on_failure") else {}),
                "deferred_unlinks": {
                    batch_id: [reference.path for reference in references]
                    for batch_id, references in self._deferred_unlinks.items()}}

    def _produced_may_defer(self):
        """May this step give the lane back and run again later?

        Only on the stager, and never while every task is waited for where
        it was submitted (tests): the waiting thread is then the one that
        frees room or acknowledges the export, and a deferral would wait
        for itself.
        """

        return self._produced_on_stager() and not self._PRODUCED_STAGER_WAIT_ALL

    def _produced_export_landed(self, group):
        """Stager: has this group's local export landed? Looks once.

        A failed export answers True, so the publication that follows
        raises it under the refusal accounting it always had.
        """

        try:
            with self._produced_lock.yielded():
                return self._local_output_spool.landed(group["batch_id"])
        except Exception:                               # noqa: BLE001
            return True

    def _produced_held_here(self, group):
        """Does this box still hold the group's own local copy? (PQ #1110)

        Then its entries are read from here, and PrismaBuild is asked to
        stage none of them. Once the spool releases the copy (its reads
        here are done, or the window needed the room) the group is read
        through PrismaBuild as before.
        """

        spool = self._local_output_spool
        return spool is not None and spool.holds(group["batch_id"])

    def _produced_publish_ahead(self, key, group):
        import time
        from .produced_stager import Requeue

        if (self._produced_plan["ahead_groups"] <= 0
                or group["published"] is not None):
            return
        if self._produced_held_here(group):
            # Its reads on this box take the local copy (PQ #1110); staging
            # it back to the box that wrote it would be the copy #1110 ends.
            self.telemetry["produced_group_ahead_local_skips"] += 1
            return
        step = {"deadline": None}

        def publish():
            if group["published"] is not None:
                return None
            if self._produced_may_defer():
                # Do not hold the lane on the local export (PQ #989). While
                # it has not landed, look once and give the lane back, so a
                # read queued behind this step runs now. Nothing is counted
                # before the export lands, so a deferral holds no credit;
                # the step's budget runs from its first run, and a step
                # that outlives it goes on to the same refusal as before.
                now = time.monotonic()
                if step["deadline"] is None:
                    step["deadline"] = self._produced_ahead_deadline(now)[1]
                if (self._local_output_spool is not None
                        and now < step["deadline"]
                        and key not in self._produced_held
                        and self._produced_ahead_has_room()
                        and not self._produced_export_landed(group)):
                    self.telemetry["produced_group_ahead_export_deferrals"] += 1
                    return Requeue(self.PRODUCED_DEFERRAL_POLL_S)
            if self._produced_take_ahead(key, group, restage=False,
                                         deadline=step["deadline"]):
                self.telemetry["produced_groups_published_ahead"] += 1
            return None

        def dropped():
            with self._produced_lock.held():
                self.telemetry["produced_stager_dropped"] += 1
                self.telemetry["produced_group_ahead_refusals"] += 1
                self._produced_ahead_refusals.append(
                    {"batch_id": group["batch_id"], "restage": False,
                     "adopted": False, "reason": "owner-closing"})
                del self._produced_ahead_refusals[
                    :-self.PRODUCED_AHEAD_REFUSAL_LOG]

        self._produced_submit("optional", "publish-ahead", publish,
                              keys=(key,), reason="publish_ahead",
                              on_drop=dropped)

    def stage_produced_boundary_ahead(self, boundary_index):
        """Ask PrismaBuild to stage one boundary plane before it is read.

        For a plane written long before its read -- the reverse chain's
        input boundaries, captured by the forward pass and retired since.
        Asks and returns: the movers run on the tier host while this owner
        computes, and the read still waits on each group's own receipt.
        Stops at the first group that does not fit the read-ahead share.
        """

        if self._produced is None or self._produced_plan["ahead_groups"] <= 0:
            return 0
        self._produced_raise_stager_failure()
        # With a stager this only asks: the count of groups it staged is in
        # ``produced_groups_staged_ahead``, and the return is 0.
        with self._produced_lock.held():
            keys = tuple(key for key in self._produced_groups
                         if key[0] == "boundary"
                         and key[1] == int(boundary_index) and key[2] < 0)
        return self._produced_submit(
            "optional", "stage-ahead",
            lambda: self._produced_stage_boundary_ahead(boundary_index),
            keys=keys, reason="stage_ahead") or 0

    def _produced_stage_boundary_ahead(self, boundary_index):
        # A plane whose retirement was asked for and has since finished can
        # be staged again; one still in flight is left to its read.
        self._drain_produced_releases()
        staged = 0
        keys = sorted(key for key in self._produced_groups
                      if key[0] == "boundary" and key[1] == int(boundary_index)
                      and key[2] < 0)
        for key in keys:
            group = self._produced_groups[key]
            if key in self._produced_held or group["live_references"] <= 0:
                continue
            if self._produced_held_here(group):
                self.telemetry["produced_group_ahead_local_skips"] += 1
                continue
            if not self._produced_ahead_has_room():
                break
            if self._produced_take_ahead(
                    key, group, restage=group["published"] is not None):
                staged += 1
                self.telemetry["produced_groups_staged_ahead"] += 1
        return staged

    def stage_produced_reads_ahead(self, references):
        """Ask for the groups the NEXT window reads, while this one computes.

        The read path's lookahead (PQ #989). ``references`` are the entries
        the next window will read; the owner's own produced groups among
        them are published, or staged again, on the stager now, so their
        movers run under this window's compute and the next window opens on
        the fast path. Asks and returns: the read still waits on each
        group's receipt, and a group this step did not reach in time is
        staged by its read as before. Foreign inputs and groups that already
        hold credit are left alone. A newer request replaces an older one,
        and the step never takes credit for a group its window has opened.

        Only with a stager: on the calling thread this would be the read's
        own work done one window early, with the GPU waiting for it either
        way. Returns the number of the owner's groups named.
        """

        if (self._produced is None or self._produced_plan is None
                or self._produced_plan["ahead_groups"] <= 0):
            return 0
        from .perturbed_x_cache import ExactActivationReference

        self._produced_raise_stager_failure()
        if not self._produced_on_compute_with_stager():
            return 0
        with self._produced_lock.held():
            keys = []
            for reference in references:
                if (not isinstance(reference, ExactActivationReference)
                        or reference in self._forward_inputs
                        or reference in self._attached_forward_inputs):
                    continue
                key, group = self._produced_group_for(reference)
                if group is not None and key not in keys:
                    if self._produced_held_here(group):
                        # Read from this box's own copy (PQ #1110).
                        self.telemetry["produced_group_ahead_local_skips"] += 1
                        continue
                    keys.append(key)
            if not keys:
                return 0
            self._produced_read_ahead_wanted = set(keys)
            # Beyond the open window, the read path holds two windows'
            # groups at once: the next window's, asked for here, and the
            # previous window's, whose retirement was asked at its exit and
            # gives its credit back only when PrismaBuild confirms it. Leave
            # room for both, so this request never waits on that
            # confirmation for a slot a write-time publication took.
            # A fused window mostly reads the groups the open window already
            # holds; those need no new credit, so only the rest reserve it.
            fresh = (keys if self._produced_read_order != "sample_major"
                     else [key for key in keys if key not in self._produced_held])
            self._produced_lookahead_reserve = 2 * len(fresh)
            self.telemetry["produced_read_ahead_requests"] += 1
        keys = tuple(keys)
        step = {"deadline": None}

        def stage():
            return self._produced_stage_reads_ahead(keys, step)

        self._produced_submit("optional", "read-ahead", stage, keys=keys,
                              reason="stage_ahead")
        return len(keys)

    def _produced_stage_reads_ahead(self, keys, step):
        """Stager: stage what a lookahead still wants; requeue for the rest.

        A group waits, off the lane, while its local export has not landed,
        while its retirement is in flight (it is staged again once that
        retirement is done), or while the share has no room. It stops
        waiting when its window opens, when a newer lookahead replaces this
        one, or when the step's budget runs out; then its read stages it.
        Nothing is counted before a group is taken, so a deferral holds no
        credit.
        """

        import time
        from .produced_stager import Requeue

        now = time.monotonic()
        if step["deadline"] is None:
            step["deadline"] = self._produced_ahead_deadline(now)[1]
        defer = self._produced_may_defer()
        # A retirement that was asked for and has since finished frees its
        # credit, and its group can be staged again.
        self._drain_produced_releases()
        waiting = False
        for key in keys:
            if key not in self._produced_read_ahead_wanted:
                continue
            group = self._produced_groups.get(key)
            if (group is None or group["live_references"] <= 0
                    or len(group["references"]) < len(group["planned"]) // 2):
                # Nothing left to read, or a group still being written: its
                # publication belongs to its last write, never to a guess.
                self._produced_read_ahead_wanted.discard(key)
                continue
            if (key in self._produced_release_pending
                    or key in self._produced_release_queued):
                waiting = True
                continue
            if key in self._produced_held or self._produced_held_here(group):
                self._produced_read_ahead_wanted.discard(key)
                continue
            if not self._produced_ahead_has_room(lookahead=True):
                waiting = True
                continue
            if (defer and group["published"] is None
                    and self._local_output_spool is not None
                    and time.monotonic() < step["deadline"]
                    and not self._produced_export_landed(group)):
                waiting = True
                continue
            if self._produced_take_ahead(
                    key, group, restage=group["published"] is not None,
                    deadline=step["deadline"], lookahead=True):
                self.telemetry["produced_groups_read_ahead"] += 1
            self._produced_read_ahead_wanted.discard(key)
        if (waiting and defer and time.monotonic() < step["deadline"]
                and any(key in self._produced_read_ahead_wanted
                        for key in keys)):
            self.telemetry["produced_read_ahead_deferrals"] += 1
            return Requeue(self.PRODUCED_DEFERRAL_POLL_S)
        return None

    @contextmanager
    def retain_produced_boundary(self, boundary_index):
        """Keep one boundary plane's groups staged across the windows inside.

        The reverse roll reads a layer's input boundary once per probe
        pass. Without this the group is retired at each window exit and
        staged again for the next pass: three avoidable stagings and three
        avoidable retirements per group. A group is retained only while the
        read-ahead share has room; otherwise it is retired at window exit
        as before. Leaving the scope asks for every retained retirement.
        """

        if self._produced is None or self._produced_plan["ahead_groups"] <= 0:
            yield
            return
        if self._produced_retained_boundary is not None:
            raise RuntimeError("produced boundary retention does not nest")
        self._produced_raise_stager_failure()
        self._produced_retained_boundary = int(boundary_index)
        try:
            yield
        finally:
            self._produced_retained_boundary = None
            if self._active_window is None:
                with self._produced_lock.held():
                    keys = [key for key in sorted(self._produced_held)
                            if key[0] == "boundary"
                            and key[1] == int(boundary_index) and key[2] < 0]
                self._release_produced_window(keys)

    def retain_produced_reads(self, references):
        """Keep the groups ``references`` read staged across this window's exit.

        The fused roll's retention (RobTand/prismaquant#997). A sample-major
        window reads the same boundary group and incoming groups as the
        window after it, until the group ends; without this each exit would
        retire them and the next window would stage them again. Called
        while a window is open, with what the NEXT window of the same pass
        reads; a newer call replaces an older one, and ``()`` retains
        nothing. Like the boundary retention it keeps a group only while
        the read-ahead share has room for it; otherwise the group is retired
        at exit as before. Staging only: no read and no order changes.
        """

        if self._produced is None or self._produced_plan["ahead_groups"] <= 0:
            return 0
        from .perturbed_x_cache import ExactActivationReference

        with self._produced_lock.held():
            keys = set()
            for reference in references:
                if (not isinstance(reference, ExactActivationReference)
                        or reference in self._forward_inputs
                        or reference in self._attached_forward_inputs):
                    continue
                key, group = self._produced_group_for(reference)
                if group is not None:
                    keys.add(key)
            self._produced_retained_reads = frozenset(keys)
        return len(keys)

    def _produced_wait_for_credit(self, need, keep=()):
        """Get credit back until ``need`` more groups fit the window.

        Pending retirements are waited out first. If that is not enough,
        read-ahead gives back what it holds: the read path is owed its two
        groups before any opportunistic one, so a group staged ahead and
        not wanted by this window (``keep``) is retired, and its own read
        stages it again. That is the synchronous loop, which is where this
        degrades to -- never to a failure read-ahead caused.
        """

        import time

        capacity = self._produced_plan["window_groups"]
        if (self._produced_plan["ahead_groups"] <= 0
                or len(self._produced_held) + need <= capacity):
            # Without read-ahead every retirement was waited for at its
            # window exit and the drain above re-drove the rest: there is
            # nothing here to wait out that was not just asked.
            return
        self.telemetry["produced_group_credit_waits"] += 1
        started = time.monotonic()
        try:
            self._produced_reclaim_credit(
                keep, until=lambda: (
                    len(self._produced_held) + need <= capacity),
                deadline=started + float(
                    self._produced_plan["staging_timeout_s"]))
        finally:
            self.telemetry["produced_group_release_wait_s"] += (
                time.monotonic() - started)

    def _produced_reclaim_credit(self, keep=(), *, until=None,
                                 on_reclaim=None, deadline=None):
        """Take stage credit back, waiting for each group. Returns the count.

        Pending retirements first -- they were asked for already -- then
        the groups read-ahead holds, the one read LAST given up FIRST.
        ``keep`` is never touched; ``until`` stops as soon as it holds;
        ``on_reclaim`` hears each key; ``deadline`` bounds every wait in
        here to the budget of the read that asked.

        A candidate that will not retire is some OTHER group's trouble. It
        is recorded against that group and the next candidate is tried: a
        request for credit must not make an unrelated group's egress the
        failure of this read. If nothing comes back the caller raises its
        own refusal, under its own name.
        """

        import time

        reclaimed = 0
        candidates = [key for key in list(self._produced_release_pending)]
        order = (self._produced_sample_major_surrender_order
                 if self._produced_read_order == "sample_major"
                 else self._produced_surrender_order)
        candidates += [key for key in sorted(self._produced_ahead, key=order)
                       if key not in self._produced_release_pending]
        for key in candidates:
            if until is not None and until():
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            group = self._produced_groups.get(key)
            if group is None or group["retired"]:
                self._produced_release_pending.pop(key, None)
                self._produced_held.discard(key)
                self._produced_ahead.discard(key)
                continue
            if key in keep:
                continue
            ahead = key in self._produced_ahead
            try:
                gone = self._release_one_produced_group(key, group, deadline)
            except Exception as exc:                    # noqa: BLE001
                self._produced_release_errors.append(
                    {"batch_id": group["batch_id"], "step": "reclaim-credit",
                     "reason": {"error": repr(exc)}})
                continue
            if gone:
                reclaimed += 1
                if ahead:
                    self.telemetry["produced_groups_ahead_surrendered"] += 1
                if on_reclaim is not None:
                    on_reclaim(key)
        return reclaimed

    @staticmethod
    def _produced_surrender_order(key):
        """Sort key: the group whose next read is furthest away comes first.

        From the read order, not from taste. The reverse walk descends the
        boundaries, and inside a layer it runs probes outer, windows inner.
        So among cotangent groups the lowest boundary is read last (it
        feeds the NEXT layer), then the highest probe, then the last
        window. A boundary group is read once in every probe pass where a
        cotangent group is read once, so boundary planes are given up after
        every cotangent group, the lower plane (the next layer's) before
        the one the current layer is still reading.
        """

        kind, boundary_index, probe, group_index = key
        return (kind == "boundary", boundary_index, -probe, -group_index)

    @staticmethod
    def _produced_sample_major_surrender_order(key):
        """The fused roll's surrender order (RobTand/prismaquant#997).

        The fused roll reads every probe's incoming group beside the
        boundary group, group by group, so a group's next read is set by the
        layer that reads it and its group index alone. A boundary plane
        ``b`` is read by layer ``b``; a cotangent plane ``b`` is layer
        ``b - 1``'s incoming. Layers descend, so the lowest reading layer
        is read last, and inside it the highest group index.
        """

        kind, boundary_index, _probe, group_index = key
        reader = boundary_index if kind == "boundary" else boundary_index - 1
        return (reader, -group_index)

    #: How long a read waits before asking again after PrismaBuild could
    #: not take a census of this owner's funding.
    PRODUCED_CENSUS_POLL_S = 0.5

    @staticmethod
    def _produced_refill_refusal_kind(refused):
        """Name a refused window refill this owner knows how to answer.

        Only the two steps that move credit -- ``refill`` (the tier's free
        pool into this owner's window, ``produced_output.refill_window``)
        and ``fund`` (this owner's holdings onto the group's mover,
        ``PoolQueue.fund_output_batch``) -- and only the two answers both
        give. ``tier-reservation-unavailable`` is a shortfall: at refill the
        free pool cannot supply the window, at fund the owner's holdings
        cannot cover the group, which is how a full window surfaces (a
        refill with no room answers ok, acquired 0). Credit read-ahead
        gives back is credit either step can take. ``unknown-retain: ...``
        is PrismaBuild failing closed on something it could not establish
        -- with many movers in flight a row is caught mid-transition -- and
        the next ask sees it whole. Anything else, on any step, is not a
        credit question: giving credit back cannot fix it, so it returns
        ``None`` and propagates as it always did.
        """

        from collections.abc import Mapping

        refusal = getattr(refused, "refusal", None)
        if (not isinstance(refusal, Mapping)
                or refusal.get("step") not in ("refill", "fund")):
            return None
        text = str(refusal.get("refusal") or "")
        if text == "tier-reservation-unavailable":
            return "shortfall"
        if text.startswith("unknown-retain"):
            return "census"
        return None

    def _produced_fund_for_read(self, step, keep, *, deadline):
        """Run one read-path funding step, answering a refused refill.

        The owner's count of what the window holds is an estimate of
        PrismaBuild's ledger, not the ledger, and read-ahead puts many of
        this owner's movers in flight at once where the synchronous loop
        had two. A refill shortfall takes credit back from read-ahead, two
        groups at a time, and runs the step again until it funds or nothing
        is left to give. A census PrismaBuild could not complete is asked
        again inside the group's own staging ``deadline``. Every other
        refusal, and every refusal at the default window, propagates
        exactly as it did before read-ahead existed. Each refusal answered
        is counted (``produced_group_read_refunds``) and kept with its
        reason (:meth:`produced_ahead_refusals`).
        """

        import time
        from .stage_a_produced_output import BoundaryProducedPublicationFailed

        while True:
            try:
                return step()
            except BoundaryProducedPublicationFailed as refused:
                kind = (self._produced_refill_refusal_kind(refused)
                        if self._produced_plan["ahead_groups"] > 0 else None)
                if kind is None:
                    raise
                self.telemetry["produced_group_read_refunds"] += 1
                self._produced_ahead_refusals.append(
                    {"batch_id": getattr(refused, "batch_id", None),
                     "step": "read", "kind": kind, "reason": repr(refused)})
                del self._produced_ahead_refusals[
                    :-self.PRODUCED_AHEAD_REFUSAL_LOG]
                self._produced_log(
                    f"a read's funding step was refused ({kind}) for "
                    f"{getattr(refused, 'batch_id', None)}; "
                    + ("asking again" if kind == "census"
                       else "taking read-ahead credit back")
                    + f": {refused!r}")
                started = time.monotonic()
                if kind == "census":
                    if started + self.PRODUCED_CENSUS_POLL_S >= deadline:
                        raise
                    with self._produced_lock.yielded():
                        time.sleep(self.PRODUCED_CENSUS_POLL_S)
                    self.telemetry["produced_group_stage_wait_s"] += (
                        time.monotonic() - started)
                    continue
                if started >= deadline:
                    raise
                # Two groups at a time: the refusal does not say how much
                # the tier is short, and giving everything back for a
                # shortfall of one throws away the staging the next windows
                # were about to use. Bounded by what is held, and by the
                # read's own deadline: a reclaim is part of this group's
                # staging, never a budget of its own.
                taken = []
                try:
                    reclaimed = self._produced_reclaim_credit(
                        keep, until=lambda: len(taken) >= 2,
                        on_reclaim=taken.append, deadline=deadline)
                finally:
                    self.telemetry["produced_group_release_wait_s"] += (
                        time.monotonic() - started)
                if not reclaimed:
                    raise

    def settle_produced_releases(self, *, wait_for_each=True):
        """Retire every group still holding stage credit, and wait for it.

        The end of a run with read-ahead: retirements asked for without
        waiting are waited out here, and a group published ahead that no
        read ever consumed gives its credit back. ``wait_for_each=False``
        only asks -- the form a failing run uses, which owes PrismaBuild
        the request and must not sit on its answer.
        """

        import time

        if (self._produced is None or self._active_window is not None
                or self._produced_plan["ahead_groups"] <= 0):
            # Without read-ahead every retirement was waited for where it
            # was asked, and a refused one is reported as debt: asking
            # again here would change that accounting.
            return
        with self._produced_lock.held():
            # No read follows a settle: a lookahead still waiting stops now
            # instead of taking credit the settle is about to give back.
            self._produced_read_ahead_wanted.clear()
            self._produced_lookahead_reserve = 0
        if self._produced_on_compute_with_stager():
            # Everything queued runs first -- a publication still queued
            # would otherwise take credit after the settle gave it back --
            # and then the settle itself runs on the stager, the one thread
            # that drives retirements while it is alive.
            with self._produced_blocked("settle"):
                idle = self._stager.drain(
                    float(self._produced_plan["staging_timeout_s"]))
            self._produced_raise_stager_failure()
            if not idle:
                raise TimeoutError("stager did not become idle before settlement")
        return self._produced_submit(
            "urgent", "settle",
            lambda: self._produced_settle_releases(wait_for_each),
            reason="settle", wait=True)

    def _produced_settle_releases(self, wait_for_each):
        import time

        started = time.monotonic()
        try:
            # Ask for all of them before waiting for any: each retirement
            # is an action on the tier host, and asked together they run
            # side by side instead of one queue round trip after another.
            for wait in ((False, True) if wait_for_each else (False,)):
                for key in sorted(self._produced_held):
                    group = self._produced_groups.get(key)
                    if group is None or group["retired"]:
                        self._produced_held.discard(key)
                        self._produced_ahead.discard(key)
                        continue
                    if key in self._produced_release_abandoned:
                        # PrismaBuild refused it and the attempts are
                        # spent: it is reported debt, not work to repeat.
                        continue
                    self._release_one_produced_group(key, group, wait=wait)
        finally:
            self.telemetry["produced_group_release_wait_s"] += (
                time.monotonic() - started)

    def _produced_reader_context(self, references):
        """Publish and materialize every group this window reads.

        Returns a per-reference resolver: a window spanning a boundary
        plane and the incoming cotangent plane reads two batches, each
        vouched in its own material namespace. Nothing polls and nothing
        falls back -- an unstaged group raises PB's own incomplete signal.

        Without a stager this is the synchronous loop, group by group:
        prepare, then fund and wait for each group in turn. With one
        (RobTand/prismaquant#895) the funding runs there as one urgent task
        for the whole window and this thread then waits on each mover's
        receipt, which is a lock-free read. A window whose groups are all
        staged already submits nothing at all.
        """

        import time

        self._produced_raise_stager_failure()
        with self._produced_lock.held():
            wanted = {}
            for reference in references:
                key, group = self._produced_group_for(reference)
                if group is None:
                    raise RuntimeError(
                        "exact boundary reference is not in any produced group: "
                        "a bound owner reads only entries it declared")
                wanted[key] = group
            if self._produced_read_ahead_wanted:
                # This window is open: a lookahead that has not staged its
                # groups by now leaves them to this read.
                missed = self._produced_read_ahead_wanted.intersection(wanted)
                self.telemetry["produced_read_ahead_missed"] += len(missed)
                self._produced_read_ahead_wanted.difference_update(wanted)
            budget = float(self._produced_plan["staging_timeout_s"])
            if not self._produced_on_compute_with_stager():
                with self._produced_blocked("read_fund"):
                    self._produced_prepare_read(wanted)
                for key, group in wanted.items():
                    # ONE absolute instant for this group's whole staging:
                    # the publish, any re-materialization and the wait for
                    # PB's receipt all spend it and none of them resets it.
                    # A budget that starts again at each step is not a bound.
                    deadline = time.monotonic() + budget
                    with self._produced_blocked("read_fund"):
                        self._produced_fund_group_for_read(
                            key, group, wanted, deadline)
                    self._produced_await_group_for_read(key, group, deadline)
            else:
                if self._produced_read_is_staged(wanted):
                    self.telemetry["produced_group_fast_reads"] += 1
                    now = time.monotonic()
                    deadlines = {key: now + budget for key in wanted}
                else:
                    deadlines = self._produced_submit(
                        "urgent", "read",
                        lambda: self._produced_stage_for_read(wanted),
                        keys=tuple(wanted), reason="read_fund", wait=True)
                for key, group in wanted.items():
                    self._produced_await_group_for_read(
                        key, group, deadlines[key])
            contexts = {key: group["context"][0]
                        for key, group in wanted.items()}
            by_reference = {reference: contexts[key]
                            for key, group in wanted.items()
                            for reference in group["references"]}
            # The window OWNS these groups for its whole lifetime. Resolving
            # them again at exit through the reference index would lose any
            # group whose entries were disposed while the window was live --
            # and that is the ordinary production pattern, not an edge case:
            # the tail retires an activation inside the read window, and the
            # reverse roll retires the previous cotangent as it writes the
            # next. Such a group would resolve to nothing, never be offered
            # to the retirement, raise no release debt, and silently keep its
            # stage credits.
            self._produced_window_keys = tuple(wanted)
            self._produced_live_keys = frozenset(wanted)
        return lambda reference: by_reference.get(reference)

    def _produced_read_is_staged(self, wanted):
        """Does this window need nothing from PrismaBuild but its receipts?

        True when every group is published, holds credit, is not retired and
        has no retirement asked for or queued. That is the steady state of
        read-ahead, and it is why the compute thread then waits on nothing
        but the mover's receipt.
        """

        return all(
            key in self._produced_held and group["published"] is not None
            and not group["retired"]
            and key not in self._produced_release_pending
            and key not in self._produced_release_queued
            for key, group in wanted.items())

    def _produced_stage_for_read(self, wanted):
        """Stager: fund every group of one window. Returns their deadlines."""

        import time

        self._produced_prepare_read(wanted)
        deadlines = {}
        for key, group in wanted.items():
            deadlines[key] = time.monotonic() + float(
                self._produced_plan["staging_timeout_s"])
            self._produced_fund_group_for_read(
                key, group, wanted, deadlines[key])
        return deadlines

    def _produced_prepare_read(self, wanted):
        """Take back what a read needs before it funds anything."""

        import time

        # Before anything new is published: give PrismaBuild another
        # chance to take back the windows an earlier exit could not.
        self._drain_produced_releases()
        for key, group in wanted.items():
            if (self._produced_plan["ahead_groups"] > 0
                    and key in self._produced_release_pending
                    and not group["retired"]):
                # Its retirement was asked for and not waited for, so an
                # egress may be deleting this very copy. Wait it out before
                # the window reads: the group is then re-staged below, or,
                # if PrismaBuild refused the retirement, read where it is.
                waited = time.monotonic()
                self._release_one_produced_group(key, group)
                self.telemetry["produced_group_release_wait_s"] += (
                    time.monotonic() - waited)
        # Credit this window must take that it does not already hold. With
        # retirements no longer waited for at window exit, the wait happens
        # here, and only when the sealed window would otherwise be exceeded.
        self._produced_wait_for_credit(
            sum(1 for key in wanted if key not in self._produced_held),
            keep=wanted)

    def _produced_fund_group_for_read(self, key, group, wanted, deadline):
        """Publish one group, or take it back onto the tier, for its read."""

        self._produced_fund_for_read(
            lambda: self._produced_publish(key, group, deadline=deadline),
            wanted, deadline=deadline)
        self._produced_held.add(key)
        if group["retired"]:
            # The same unchanged logical batch, taken back onto the
            # tier. PB is asked first (``materialization_state``) and
            # drives the transition only if its own records say the
            # copy is gone; the batch id, manifest, descriptors,
            # namespace and durable charge are all unchanged.
            def ensure():
                with self._produced_lock.yielded():
                    return self._produced.ensure_batch_materialized(
                        batch_id=group["batch_id"], deadline=deadline)

            self._produced_fund_for_read(ensure, wanted, deadline=deadline)
            group["retired"] = False
            group["context"] = None
            self.telemetry["produced_groups_rematerialized"] += 1

    def _produced_await_group_for_read(self, key, group, deadline):
        """Wait for the mover's receipt and compose the group's context."""

        import time

        if group["context"] is not None:
            return
        # Staging is asynchronous: the publish (or the ensure) seals a
        # mover row and the FLEET runs it. Wait on PB's own receipt --
        # bounded, named, no fallback -- before composing, because
        # fragments alone compose the same whether the batch is whole or
        # half there. Both calls are lock-free reads, so they stay on the
        # thread that needs the answer.
        waited = time.monotonic()
        try:
            with self._produced_blocked("stage_wait"), (
                    self._produced_lock.yielded()):
                self._produced.await_materialized(
                    batch_id=group["batch_id"],
                    timeout_s=self._produced_plan["staging_timeout_s"],
                    deadline=deadline)
        finally:
            self.telemetry["produced_group_stage_wait_s"] += (
                time.monotonic() - waited)
        with self._produced_blocked("compose"), self._produced_lock.yielded():
            resolver, block = self._produced.reader_context(
                batch_id=group["batch_id"],
                manifest_digest=group["manifest_digest"])
        group["context"] = (resolver, block)
        self.telemetry["produced_groups_materialized"] += 1

    def produced_group_records(self):
        """What this owner published, for a receipt. Never a second ledger."""

        with self._produced_lock.held():
            return self._produced_group_records_locked()

    def _produced_group_records_locked(self):
        return [{"batch_id": group["batch_id"],
                 "entries": len(group["references"]),
                 "manifest_digest": group["manifest_digest"],
                 "staged": group["context"] is not None,
                 "retired": group["retired"],
                 "origin_reclaimed": group["origin_reclaimed"]}
                for group in self._produced_groups.values()]

    def _release_unpublished_prewrites(self, *, failing=False):
        """Give back the durable headroom of groups that produced nothing.

        A prewrite that was claimed and never committed would otherwise
        charge its ceiling against the instance's durable maxima for the
        life of the instance. Aborting is safe only when nothing durable
        remains, and PrismaBuild proves that itself -- ``abort_prewrite``
        lstats every planned path and retains on a present or unstatable
        one, so this never turns a crashed write into freed budget. A
        published (committed) group is deliberately skipped: its entries
        are durable and its charge ends at ``reclaim_origin``, not here.

        ``failing`` with ``dispose_on_failure`` bound (PQ #1251): the owner
        first removes its own uncommitted group's files, only the group's
        planned paths, which PrismaBuild's ``abort_prewrite`` leaves to the
        producer. Each such abort is recorded (``disposed_uncommitted`` in
        :meth:`produced_output_report`).
        """

        if self._produced is None:
            return
        dispose = failing and bool(self._produced_plan
                                   and self._produced_plan.get("dispose_on_failure"))
        for group in self._produced_groups.values():
            if group["published"] is not None or group.get("origin_ref") is not None:
                # Committed, staged or at its origin: the commit consumed the
                # prewrite, and the charge now ends with the batch.
                continue
            if (self._local_output_spool is not None
                    and self._local_output_spool.pending(group["batch_id"])):
                continue
            if group.get("origin_reclaimed"):
                continue  # Its prewrite went with its last file (PQ #1110).
            # A live export may still land its files: they are left alone,
            # as before (PQ #1251).
            live = bool(dispose and self._local_output_spool is not None
                        and self._local_output_spool.export_live(group["batch_id"]))
            removed = (self._dispose_uncommitted_group(group)
                       if dispose and not live else None)
            out = self._produced.abort_prewrite(batch_id=group["batch_id"])
            if dispose:
                self._produced_disposed_uncommitted.append(
                    {"batch_id": group["batch_id"], "files_removed": removed,
                     "export_live": live, "abort_ok": bool(out.get("ok")),
                     "refusal": out.get("refusal")})
            if (not out.get("ok") and self._local_output_spool is not None
                    and group["live_references"]):
                # Read only on this box and kept (a retained boundary, a
                # checkpoint's cotangent): never committed, so its charge
                # stays a prewrite whose files PrismaBuild will not remove.
                # Recorded, so the receipt says which groups ended this way.
                self._produced_retained_uncommitted.append(
                    {"batch_id": group["batch_id"],
                     "refusal": out.get("refusal"),
                     "live_references": group["live_references"]})

    def _dispose_uncommitted_group(self, group):
        """Remove one uncommitted group's own files: its planned paths only.

        The planned paths are this generation's own entry names under its
        own directory (``_produced_planned_paths``), so no other writer's
        file is among them. A path that is not a regular file, or not in
        this generation's ``entries``, is left for ``abort_prewrite`` to
        refuse on, and recorded. Returns how many files went.
        """
        own = self.directory / "entries"
        removed = 0
        for planned in group["planned"]:
            path = Path(planned)
            if path.parent != own:
                # Never raised: this runs on an exit that is already failing.
                self._produced_release_errors.append(
                    {"batch_id": group["batch_id"], "step": "dispose-uncommitted",
                     "reason": {"error": f"{path} is outside this generation's "
                                         "entries"}})
                continue
            try:
                if not stat.S_ISREG(os.lstat(path).st_mode):
                    raise OSError(f"{path} is not a regular file")
                os.unlink(path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                self._produced_release_errors.append(
                    {"batch_id": group["batch_id"], "step": "dispose-uncommitted",
                     "reason": {"error": repr(exc)}})
                continue
            removed += 1
        return removed

    def release_produced_group(self, reference):
        """Release the stage copy of the group holding ``reference``.

        Stage retirement only: the durable origin survives, its charge is
        constant, and this owner still owns disposal of its own entries.
        This is what makes the window BOUNDED -- the stage copy is a loan,
        and a later read of the same unchanged logical batch takes it again
        through PrismaBuild's repeat-materialization surface rather than
        through a second publication or a second charge.

        The cached reader context is dropped with the copy. Reusing a
        composed map after its fragments are evicted would resolve a path
        that is no longer there, which is the second failure shape of a
        bounded window (the first being never releasing at all).
        """

        self._produced_raise_stager_failure()
        with self._produced_lock.held():
            key, group = self._produced_group_for(reference)
        if group is None:
            raise RuntimeError(
                "exact boundary reference is not in any produced group")
        if self._active_window is not None:
            raise RuntimeError(
                "a produced boundary group cannot be retired while its "
                "window is live: the pin must be released first")
        return self._produced_submit(
            "urgent", "release-group",
            lambda: self._retire_produced_group(key, group),
            keys=(key,), reason="release", wait=True)

    def _retire_produced_group(self, key, group):
        """Give one group's stage copy back. The caller names the group."""

        if group["retired"]:
            return dict(group["published"] or {})
        # On the calling thread any live window refuses, as it always did.
        # The stager retires the PREVIOUS window's groups while the next one
        # is open, so there the rule is what it always meant: not a group the
        # live window reads.
        if self._active_window is not None and (
                not self._produced_on_stager()
                or key in self._produced_live_keys):
            raise RuntimeError(
                "a produced boundary group cannot be retired while its "
                "window is live: the pin must be released first")
        with self._produced_lock.yielded():
            out = self._produced.retire(group["batch_id"])
        if out.get("ok"):
            group["retired"] = True
            group["context"] = None
            self._produced_held.discard(key)
            self._produced_ahead.discard(key)
            self.telemetry["produced_groups_retired"] += 1
        return out

    #: How many times one group's stage retirement is re-driven through
    #: PrismaBuild's own egress before the failure is recorded and left
    #: standing. Bounded on purpose: a retry loop with no ceiling is a
    #: scheduler, and this lane does not own scheduling.
    PRODUCED_RELEASE_ATTEMPTS = 3

    def _release_one_produced_group(self, key, group, deadline=None,
                                    wait=True):
        """Drive one group's stage retirement. Returns True when it is gone.

        ``wait=False`` asks and does not wait: when PrismaBuild defers the
        retirement on its own in-flight work, the group stays pending and a
        later call re-drives it. Such a poll is not an attempt and not a
        failure, so it neither spends ``PRODUCED_RELEASE_ATTEMPTS`` nor
        grows the error list. Every other outcome is handled exactly as a
        waited retirement handles it.

        ``deadline`` is the absolute monotonic instant the whole wait must
        end by, threaded through so a retirement that is re-driven inside
        one cannot mint itself a fresh budget.

        A returned ``{"ok": False}`` is a RESULT, not an absence of news:
        PrismaBuild's egress refused for a reason, and that reason is
        recorded against the group so the next attempt can see it. Nothing
        here pretends the copy was retired and nothing refunds its credit
        -- the tokens stay held by the material that is still there.
        """

        record = self._produced_release_pending.setdefault(
            key, {"batch_id": group["batch_id"], "attempts": 0,
                  "first_reason": None, "last_reason": None})
        try:
            # By KEY, never by reference: a group whose entries were
            # disposed while its retirement was refused has no live
            # reference left to look itself up with, and that is exactly
            # the group a drain exists for.
            out = self._retire_produced_group(key, group)
        except Exception as exc:                        # noqa: BLE001
            record["attempts"] += 1
            return self._handle_produced_release_outcome(
                key, group, record, None, {"error": repr(exc)}, deadline)
        if not wait and not group["retired"]:
            from .stage_a_produced_output import classify_egress_outcome
            if classify_egress_outcome(out) == "own-copy-deferral":
                reason = {"refusal": out.get("refusal"),
                          "step": out.get("step"),
                          "receipt": out.get("receipt")}
                if record.get("class") != "own-copy-deferral":
                    record["class"] = "own-copy-deferral"
                    if record["first_reason"] is None:
                        record["first_reason"] = reason
                    self.telemetry["produced_group_release_deferrals"] += 1
                record["last_reason"] = reason
                return False
        record["attempts"] += 1
        if group["retired"]:
            self._produced_release_pending.pop(key, None)
            # The stage copy is gone; if this group's origins went while
            # its retirement was refused, THIS is the moment its durable
            # charge becomes releasable.
            self._reclaim_produced_origin_for_group(key, group)
            return True
        return self._handle_produced_release_outcome(
            key, group, record, out,
            {"refusal": out.get("refusal"), "step": out.get("step"),
             "receipt": out.get("receipt")}, deadline)

    def _record_produced_release_failure(self, group, record, reason):
        """Count ONE stage copy that did not come back, with its reason.

        The counter answers exactly one question -- did this owner leave a
        stage copy standing? -- so it is incremented where that is known,
        not wherever a refusal is first seen.
        """

        self.telemetry["produced_group_release_failures"] += 1
        self._produced_release_errors.append(
            {"batch_id": group["batch_id"], "attempt": record["attempts"],
             "reason": reason})

    def _handle_produced_release_outcome(self, key, group, record, out,
                                         reason, deadline):
        """Decide about ONE observed outcome. No second retirement here.

        Split out because the wait loop must be able to hand back the
        outcome it just saw instead of asking again: re-driving to "find
        out" what it already knows costs an extra egress, hides the
        outcome that changed, and -- when the new answer is another
        deferral -- restarts a budget that is supposed to be absolute.

        A deferral PrismaBuild owns is NOT recorded as a release failure
        here. It is news that the copy is still coming back, and the wait
        below is what decides: the same receipt is already not a failure
        when the poll path (``wait=False``) sees it, so counting it here
        made the counter say whether this owner happened to ask inside a
        wait, not whether a stage copy came back. A deferral that runs its
        budget out is recorded by the wait, under the reason it ended on.
        """

        record["last_reason"] = reason
        if record["first_reason"] is None:
            record["first_reason"] = reason
        self._produced_log(
            f"retirement of {group['batch_id']} not taken (attempt "
            f"{record['attempts']}): {repr(reason)[:300]}")
        if "error" not in reason:
            from .stage_a_produced_output import (
                UNCLASSIFIED_OUTCOMES, BoundaryEgressUnclassified,
                classify_egress_outcome)
            kind = classify_egress_outcome(out)
            record["class"] = kind
            if kind == "own-copy-deferral":
                # PrismaBuild deferred this on a lifecycle it owns: the
                # evicted mover's own live claimed copy. Bytes, proof and
                # full credit are kept and ordinary retry returns the
                # token, so wait it out inside the budget already bound --
                # the SAME deadline if one is already running.
                return self._await_produced_release_deferral(
                    key, group, record, out, deadline)
        self._record_produced_release_failure(group, record, reason)
        if "error" not in reason:
            if kind in UNCLASSIFIED_OUTCOMES:
                # Three receipts that are not a decision: no deferred_own
                # key at all and no positive cause, a non-empty
                # deferred_own naming a reason this lane does not
                # recognise, and a deferred_own of the wrong shape. Each is
                # surfaced, not decided -- reading any of them as "no
                # deferral" is the fail-open shape this exists for, and
                # reading an unrecognised one as a deferral would sit on a
                # wait that is not known to clear.
                self._produced_release_pending.pop(key, None)
                self._produced_release_unclassified[key] = dict(record)
                raise BoundaryEgressUnclassified(
                    group["batch_id"], out, out.get("receipt"), kind)
            # foreign-pin, promotion-handoff, egress-error: real failures
            # on other lifecycles. Preserved exactly as before -- recorded,
            # re-driven on the next drain, credits retained, never waited
            # on. None of them is this lane's to resolve.
        if record["attempts"] >= self.PRODUCED_RELEASE_ATTEMPTS:
            # Cleared EXPLICITLY, with the original reason kept and the
            # credits retained. The group stays unretired, so nothing
            # downstream may treat its window as free, and a later read of
            # it still finds its live copy rather than asking for a
            # re-materialization that would be wrong.
            self._produced_release_abandoned[key] = dict(record)
            self._produced_release_pending.pop(key, None)
        return False

    #: How long one paced re-drive waits before asking PrismaBuild again.
    #: A floor, not a schedule: without it a bounded attempt count burns a
    #: whole budget in milliseconds and reports a timeout that never
    #: waited for anything.
    PRODUCED_DEFERRAL_POLL_S = 0.5

    def _await_produced_release_deferral(self, key, group, record, outcome,
                                         deadline=None):
        """Re-drive a DEFERRED retirement, paced, inside the staging budget.

        PrismaBuild keeps the bytes, the proof and the full credit while
        its own child mover is still live, and says so by deferring rather
        than by destroying anything. Ordinary retry returns the token once
        that mover reaches terminal, so the only thing this owes is to
        keep asking at a sane cadence until the budget bound at
        :meth:`bind_produced_output` is spent -- the SAME budget the read
        already waits on for staging, not a second one.

        Two failure shapes are deliberately excluded. It does not spin: it
        sleeps :data:`PRODUCED_DEFERRAL_POLL_S` between asks, so a bounded
        attempt count cannot report a timeout it never waited for. And it
        does not continue quietly into a refill that cannot fund: running
        the budget out raises :class:`BoundaryProducedReleaseDeferred` with
        the reason, the elapsed time and the attempt count.

        Anything that stops being a deferral leaves immediately and is
        handled as what it became -- a retirement that starts deferred and
        ends refused by a foreign pin is a foreign-pin refusal, and a
        reason the vocabulary cannot name is surfaced, not absorbed.
        """

        import time
        from .stage_a_produced_output import (
            DEFERRED_OWN_FIELD, BoundaryProducedReleaseDeferred,
            classify_egress_outcome)

        budget = float(self._produced_plan["staging_timeout_s"])
        started = time.monotonic()
        # ONE absolute deadline for the whole wait. Taken from the caller
        # when a wait is already running, so a re-driven retirement that
        # reports another deferral continues the original bound instead of
        # minting a new one -- the inverse of the tight loop, and just as
        # unbounded: it never spins, and it never ends either.
        if deadline is None:
            deadline = started + budget
        receipt = outcome.get("receipt") or {}
        record["deferred_own"] = receipt.get(DEFERRED_OWN_FIELD)
        self.telemetry["produced_group_release_deferrals"] += 1
        try:
            with self._produced_lock.yielded():
                state = self._produced.materialization_state(
                    batch_id=group["batch_id"])
            record["deferred_state"] = state
        except Exception as exc:                        # noqa: BLE001
            # Evidence only. A state read that fails does not change what
            # the retirement said, and must not become the failure.
            record["deferred_state"] = {"error": repr(exc)}
        redrives = 0

        def _spent():
            """The budget is gone: record it and stop, calling nothing."""
            record["deferred_waited_s"] = time.monotonic() - started
            self._produced_release_pending.pop(key, None)
            self._produced_release_abandoned[key] = dict(record)
            # HERE is where a deferral becomes a release failure: the copy
            # did not come back inside the budget. Until this point it was
            # news, not a verdict.
            self._record_produced_release_failure(
                group, record, record["last_reason"])
            raise BoundaryProducedReleaseDeferred(
                group["batch_id"],
                waited_s=time.monotonic() - started, timeout_s=budget,
                attempts=redrives, outcome=outcome)

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _spent()
            with self._produced_lock.yielded():
                time.sleep(min(self.PRODUCED_DEFERRAL_POLL_S, remaining))
            # THE DEADLINE IS READ AGAIN HERE, immediately before the
            # retirement, because a retire is a mutation and the sleep
            # above can have landed exactly on the deadline. Checking only
            # at the top of the loop wakes at expiry and drives one more
            # egress anyway. Same rule as the funding path: a deadline
            # bounds SIDE EFFECTS, not iterations.
            if time.monotonic() >= deadline:
                _spent()
            redrives += 1
            self.telemetry["produced_group_release_retries"] += 1
            outcome = self._retire_produced_group(key, group)
            if group["retired"]:
                record["deferred_waited_s"] = time.monotonic() - started
                record["deferred_redrives"] = redrives
                self._produced_release_pending.pop(key, None)
                self._reclaim_produced_origin_for_group(key, group)
                return True
            if classify_egress_outcome(outcome) != "own-copy-deferral":
                # It changed into something else. Hand THAT outcome to the
                # one place that decides -- not another retirement. Asking
                # again would spend an extra egress, lose the outcome just
                # observed, and, if the new answer deferred, restart a
                # budget that is meant to be absolute.
                record["attempts"] += 1
                return self._handle_produced_release_outcome(
                    key, group, record, outcome,
                    {"refusal": outcome.get("refusal"),
                     "step": outcome.get("step"),
                     "receipt": outcome.get("receipt")}, deadline)

    def _drain_produced_releases(self):
        """Re-drive every still-pending retirement before the next window.

        Called BEFORE the next window publishes, because that is where a
        stalled retirement actually costs something: the tier window is
        small (the production geometry is four tokens -- current plus
        next), so one group that failed to give its two tokens back is the
        whole next advance. Draining here is the difference between
        degrading and stalling.
        """

        import time

        wait = self._produced_plan["ahead_groups"] <= 0
        now = time.monotonic()
        for key in list(self._produced_release_pending):
            group = self._produced_groups.get(key)
            if group is None or group["retired"]:
                self._produced_release_pending.pop(key, None)
                continue
            record = self._produced_release_pending[key]
            if not wait and now < record.get("next_poll", 0.0):
                # An egress takes seconds; asking again sooner only costs
                # PrismaBuild a lock and this window a round trip.
                continue
            record["next_poll"] = now + self.PRODUCED_RELEASE_POLL_S
            if wait:
                self.telemetry["produced_group_release_retries"] += 1
            else:
                # A poll of a retirement already asked for: not a re-drive.
                self.telemetry["produced_group_release_polls"] += 1
            self._release_one_produced_group(key, group, wait=wait)
        if not wait:
            self.telemetry["produced_group_release_wait_s"] += (
                time.monotonic() - now)

    #: The soonest a retirement that was asked for without waiting is asked
    #: about again.
    PRODUCED_RELEASE_POLL_S = 2.0

    def _release_produced_window(self, keys):
        """Retire every stage copy this window borrowed.

        By the KEYS the window captured when it opened, never by its
        references: a reference disposed during the window is gone from
        the lookup index, and a group found by nothing is a group whose
        credits are never asked for.
        """

        wait = self._produced_plan["ahead_groups"] <= 0
        with self._produced_lock.held():
            # Which groups stay is decided HERE, on the thread that opens the
            # next window, and never later on the stager: a group kept for
            # the next probe pass must not be found queued for retirement by
            # the read that wants it.
            retained = self._produced_retained_boundary
            again = self._produced_retained_reads
            ask = []
            for key in keys:
                group = self._produced_groups.get(key)
                if group is None or group["retired"]:
                    continue
                if (key in again) or (
                        retained is not None and key[0] == "boundary"
                        and key[1] == retained and key[2] < 0):
                    # Another probe pass reads this group. Keep it staged if
                    # the read-ahead share has room for it (or holds it).
                    if key in self._produced_ahead:
                        continue
                    if self._produced_ahead_has_room(holding=True):
                        self._produced_ahead.add(key)
                        self.telemetry["produced_groups_retained"] += 1
                        continue
                ask.append(key)
            threaded = self._produced_on_compute_with_stager()
            if threaded:
                if not ask:
                    return
                # A read that wants one of these waits for the ask below
                # instead of taking the fast path past it.
                self._produced_release_queued.update(ask)
        self._produced_submit(
            "ordered", "release",
            lambda: self._produced_ask_releases(ask, wait),
            keys=tuple(ask), reason="release")

    def _produced_ask_releases(self, ask, wait):
        """Ask for each retirement the window exit decided on."""

        import time

        started = time.monotonic()
        try:
            for key in ask:
                self._produced_release_queued.discard(key)
                group = self._produced_groups.get(key)
                if group is None or group["retired"]:
                    continue
                self._release_one_produced_group(key, group, wait=wait)
                record = self._produced_release_pending.get(key)
                if record is not None:
                    record["next_poll"] = (
                        time.monotonic() + self.PRODUCED_RELEASE_POLL_S)
        finally:
            self._produced_release_queued.difference_update(ask)
            self.telemetry["produced_group_release_wait_s"] += (
                time.monotonic() - started)

    def produced_release_debt(self):
        """Stage copies this owner asked PB to retire and PB did not.

        Reported rather than hidden: a caller that needs the window back
        can see exactly which groups still hold it and why.
        """

        with self._produced_lock.held():
            return self._produced_release_debt_locked()

    def _produced_release_debt_locked(self):
        return {"pending": {str(record["batch_id"]): record["last_reason"]
                            for record in self._produced_release_pending.values()},
                "abandoned": {str(record["batch_id"]): record["first_reason"]
                              for record in self._produced_release_abandoned.values()},
                "unclassified": {str(record["batch_id"]): record["last_reason"]
                                 for record
                                 in self._produced_release_unclassified.values()},
                "publish_deferred": {
                    str(record["batch_id"]): record
                    for record in self._produced_publish_deferred.values()}}

    @contextmanager
    def prefetch(self, references):
        from .perturbed_x_cache import prefetch_exact_activation_cache_entries
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
        # A bound owner publishes and materializes this window's groups
        # here, at the FIRST read, and hands the read their namespaced
        # contexts. An unbound owner passes None and resolves through the
        # process input map exactly as before.
        owned = tuple(ref for ref in references if ref not in self._forward_inputs)
        # An own entry this box still holds is read from its local copy and
        # PrismaBuild stages none of it (PQ #1110); the hold keeps the copy
        # until the window has read it. Only the rest is staged.
        local_reads = (nullcontext({}) if self._local_output_spool is None
                       else self._local_output_spool.local_reads(owned))
        with local_reads as local:
            staged = tuple(ref for ref in owned if ref not in local)
            resolver = (None if self._produced is None or not staged
                        else self._produced_reader_context(staged))
            if resolver is not None and len(staged) != len(references):
                own_resolver = resolver
                resolver = lambda ref: (None if ref in self._forward_inputs else own_resolver(ref))
            # Captured HERE, while every group is still resolvable.
            window_keys = self._produced_window_keys if staged else ()
            self._produced_window_keys = ()
            with torch.profiler.record_function("aura.exact_activation.prefetch"):
                if self._scratch is None:
                    from .perturbed_x_cache import EntryReadScratch
                    self._scratch = EntryReadScratch()
                context = prefetch_exact_activation_cache_entries(references,
                    expected_session=self.session, max_tensor_bytes=self.config["max_resident_bytes"],
                    residency_check=self._reserve, scratch=self._scratch,
                    resolver=resolver, local_paths=local,
                    session_for_reference=(lambda ref: self._entry_identity(ref)["session"])
                        if self._forward_inputs or self._attached_forward_inputs else None)
                window = context.__enter__()
            if local:
                self.telemetry["produced_local_windows"] += 1
                self.telemetry["produced_local_reads"] += len(local)
                self.telemetry["produced_local_read_bytes"] += sum(
                    ref.file_bytes for ref in local)
        self._active_window = window
        self.telemetry["prefetch_windows"] += 1
        self.telemetry["read_tensor_bytes"] += sum(ref.tensor_bytes for ref in references)
        primary = None
        try:
            yield window
        except BaseException as exc:
            primary = exc
            raise
        finally:
            self._active_window = None
            self._produced_live_keys = frozenset()
            # Cleanup never MASKS the failure that caused it. A cleanup
            # error raised out of a `finally` replaces the compute error as
            # the exception the caller sees, which is worse than either
            # fact alone -- so when something is already in flight, the
            # cleanup failure is attached to it and the original
            # propagates.
            for step in (lambda: context.__exit__(
                    type(primary) if primary is not None else None, primary,
                    primary.__traceback__ if primary is not None else None),
                    # The pins are gone with the window, so the stage
                    # copies it borrowed go back to free HERE -- that is
                    # the bounded-window contract, and holding them would
                    # accumulate every group the pass ever read.
                    lambda: (self._release_produced_window(window_keys)
                             if self._produced is not None else None)):
                try:
                    step()
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    primary.add_note(
                        "exact boundary window cleanup also failed: "
                        f"{cleanup!r}")

    def get(self, window, reference):
        self._entry_identity(reference)
        if window is not self._active_window:
            raise RuntimeError("exact boundary lookup has no active resident window")
        try:
            return window.get(reference)
        except RuntimeError:
            self.telemetry["hot_read_misses"] += 1
            raise

    def _settle_produced_releases_at_exit(self, primary):
        """Give read-ahead's stage credit back as the owner closes.

        Read-ahead asks for retirements without waiting and may hold groups
        no read consumed; both end here. A clean run waits for each. A
        failing run only asks. Neither may change the outcome of the run:
        the capture's bytes are already durable when this runs, and a stage
        copy PrismaBuild would not retire is reported debt
        (``produced_release_debt``), not a reason to discard hours of
        finished work or to mask the failure that is already propagating.
        """

        if self._produced is None or self._produced_plan is None:
            return
        if self._stager_stuck:
            # Recorded where the join timed out. A PrismaBuild call is still
            # running on that thread; this one does not drive the same
            # retirements beside it.
            return
        try:
            self.settle_produced_releases(wait_for_each=primary is None)
        except Exception as cleanup:                    # noqa: BLE001
            self._produced_release_errors.append(
                {"batch_id": None, "step": "settle-at-exit",
                 "reason": {"error": repr(cleanup)}})
            note = ("produced-output settle at exit did not finish: "
                    f"{cleanup!r}; debt: {self.produced_release_debt()!r}")
            if primary is not None:
                primary.add_note(note)
            else:
                print(f"exact boundary owner: {note}", flush=True)

    def __exit__(self, exc_type, exc, traceback):
        # Unfinished generations do not authorize reuse. A separately verified
        # forward-recovery capsule can lend exact inputs to a fresh generation;
        # borrowed files remain owned by their original, contained producer.
        self._status = "failed" if exc_type is not None else (
            "attached" if self._readonly else ("complete" if self.session else "unused"))
        try:
            # First, so that everything below runs with no second thread: the
            # disposal, the settle and the prewrite release are then the
            # synchronous code, unchanged.
            self._produced_stop_stager()
            if self._stager_stuck:
                note = ("stager still owns this generation; retaining origins, "
                        "checkpoint reservations and produced-output credit")
                self._produced_release_errors.append(
                    {"batch_id": None, "step": "stuck-stager-exit-retain",
                     "reason": {"error": note,
                                "origins": len(self._references),
                                "origin_bytes": self.telemetry["live_artifact_bytes"]}})
                if exc is not None:
                    exc.add_note(note)
                # A run whose stager never joined is not a complete one.
                # Its origins, checkpoint reservations and produced-output
                # credit are still owned by a thread this process cannot
                # account for, so the status says ``retained`` and the
                # receipt names the debt (PQ #960). A run that was already
                # failing keeps that verdict.
                if self._status != "failed":
                    self._status = "retained"
                return False
            if exc_type is None:
                self.settle_local_output()
            elif self._local_output_spool is not None:
                # A failing run waits for nothing, but a retired entry whose
                # export has landed still gets its canonical file removed
                # (PQ #1110); what is left is in the telemetry
                # (``produced_deferred_unlinks`` less ``_done``) and in
                # ``produced_output_report()["deferred_unlinks"]``.
                try:
                    self._local_output_spool.release_landed()
                    self._produced_flush_deferred_unlinks(final=True)
                except Exception as cleanup:            # noqa: BLE001
                    self._produced_release_errors.append(
                        {"batch_id": None, "step": "failed-exit-deferred-unlinks",
                         "reason": {"error": repr(cleanup)}})
            if not self._readonly and (
                    self._active_window is not None or self.telemetry["resident_tensor_bytes"]):
                raise RuntimeError("exact boundary generation closed with a live window")
            if self._checkpoint_active is not None:
                raise RuntimeError(
                    "exact boundary generation closed with an active checkpoint "
                    "reservation: commit, abandon, or cancel it first")
            if self._published:
                # A published generation's entries belong to its receipt; the
                # quanta read them back, so closing the owner must not unlink
                # them. Deliberate retirements already happened above.
                # Committed checkpoints are durable for the same reason and
                # survive here too; only retained (never receipted) attempts
                # are disposed below.
                self._references.clear()
                self._slots.clear()
            else:
                for reference in list(self._references.values()):
                    self._retire(reference, missing_ok=True)
                self._slots.clear()
            self._reclaim_retained_checkpoints()
            self._settle_produced_releases_at_exit(exc)
            self._release_unpublished_prewrites(failing=exc_type is not None)
        except BaseException:
            self._status = "failed"
            raise
        finally:
            if self._cotangent_scratch is not None:
                self._cotangent_scratch.close()
                self._cotangent_scratch = None
            if not self._stager_stuck:
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
            self._publish_status()

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

    A storage that stages its reads through PrismaBuild is asked, as soon as
    a window is open, for the groups the NEXT window reads, so their movers
    run under this window's compute (RobTand/prismaquant#989). ``then``
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


class StreamedCausalLM:
    """Causal-LM forward adapter over an existing ``StreamingContext``.

    ``pin_layer_for_qname`` keeps exactly one decoder layer installed while a
    caller temporarily mutates and restores a serving unit in it.  All other
    layers continue to stream through the context's cache.  This is the seam
    used by empirical expert KL; AURA additionally consumes the explicit
    boundary/isolated-layer methods for its streamed adjoint.
    """

    def __init__(
        self,
        context,
        profile,
        *,
        prefetch_lookahead: int = 2,
        require_prefetched_residency: bool = False,
    ):
        if type(require_prefetched_residency) is not bool:
            raise TypeError(
                "require_prefetched_residency must be a bool"
            )
        self.context = context
        self.model = context.model
        self.base_model = context.base_model
        self.layers = context.layers
        self.layers_prefix = str(context.layers_prefix)
        self.num_layers = int(context.num_layers)
        self.device = torch.device(context.device)
        self.dtype = context.dtype
        self.profile = profile
        self.prefetch_lookahead = max(0, int(prefetch_lookahead))
        self.require_prefetched_residency = require_prefetched_residency
        self._pinned_layer: int | None = None
        # Layers whose pre-install re-assert had to issue a fresh source read
        # during the last exact layer-major traversal (#403). Empty when every
        # speculative prefetch was still held by the runner at install time.
        self.layer_major_prefetch_retries: tuple[int, ...] = ()

    def layer_index_for_qname(self, qname: str) -> int:
        match = re.match(
            rf"^{re.escape(self.layers_prefix)}([0-9]+)(?:\.|$)",
            str(qname),
        )
        if match is None:
            raise RuntimeError(
                f"streamed cost unit {qname!r} is not under decoder prefix "
                f"{self.layers_prefix!r}"
            )
        layer = int(match.group(1))
        if not 0 <= layer < self.num_layers:
            raise RuntimeError(
                f"streamed cost unit {qname!r} resolved invalid layer {layer}"
            )
        return layer

    def snapshot_selected_weights(self, names, *, max_resident_bytes: int,
                                  resource_check=None, expected_source_keys=None,
                                  host=False):
        """Copy selected source Linears from the existing resident layer cache.

        This is preparation for a consumer that already owns its ``weights``
        mapping and needs no source forward. The finite layer sequence is
        prefetched through StreamingContext; ordinary adjacent-layer top-up
        is disabled so a sparse selection never reads unrelated layers.
        Independent copies prevent an expert view from pinning its complete
        packed parent after the source layer has been released.

        ``host=True`` makes each copy a contiguous CPU tensor. The streaming
        row head hashes weights on reader threads while the encode thread owns
        the device, so its weights must not be device tensors.
        """
        from .routed_experts import (
            profile_declared_packed_expert_projections,
            refresh_packed_expert_projections,
        )

        names = tuple(names)
        if not names or len(set(names)) != len(names):
            raise ValueError("selected source requires unique nonempty unit names")
        if type(max_resident_bytes) is not int or max_resident_bytes <= 0:
            raise ValueError("selected source requires a positive resident byte budget")
        if self._pinned_layer is not None:
            raise RuntimeError("selected source cannot start with a pinned layer")
        modules = dict(self.model.named_modules())
        projected = {member.qname: member for member in
                     profile_declared_packed_expert_projections(self.model, self.profile)}
        shapes, layers = {}, {}
        for name in sorted(names):
            layer = self.layer_index_for_qname(name)
            if name in projected:
                weight = projected[name].weight
                parameter_name = projected[name].module_qname+'.'+projected[name].param_name
            elif isinstance(modules.get(name), torch.nn.Linear):
                weight = modules[name].weight
                parameter_name = name+'.weight'
            else:
                raise RuntimeError(f"selected source unit is not a declared Linear: {name}")
            dtype = getattr(self.context, 'buffer_dtypes', {}).get(parameter_name, self.dtype)
            shapes[name] = (tuple(weight.shape), dtype,
                            weight.numel()*torch.empty((), dtype=dtype).element_size())
            layers.setdefault(layer, []).append(name)
        del weight
        required = sum(shape[2] for shape in shapes.values())
        if required > max_resident_bytes:
            raise RuntimeError("selected source weights exceed their resident byte budget")

        snapshot_only = getattr(self.context, 'source_snapshot_only', False)
        if snapshot_only:
            if expected_source_keys is None:
                raise RuntimeError('snapshot requires admitted source keys before source I/O')
            from .layer_streaming import selected_weight_source_keys
            planned_keys = tuple(expected_source_keys)
            actual_keys = selected_weight_source_keys(names, self.profile, self.context.weight_ckpt)
            if planned_keys != actual_keys:
                raise RuntimeError('snapshot source dependencies differ from the admitted plan')
            self.context.configure_selected_snapshot(names, self.profile)

        ordered = sorted(layers)
        # `max_cache_slots` is None when the layer cache is bounded by bytes and
        # carries no slot cap -- the ordinary autoscaled case, and what every
        # other reader spells out (`StreamingContext.suggest_prefetch_lookahead`).
        # This one subtracted from it, so a sparse selected-source walk on an
        # autoscaled cache died on None - 1 instead of walking. With no cap the
        # window is the caller's lookahead; with one it keeps a slot for the
        # layer `install()` still owns.
        slots = self.context.max_cache_slots
        window = max(1, self.prefetch_lookahead if slots is None
                     else min(self.prefetch_lookahead, slots - 1))
        weights, records = {}, []
        for layer in ordered[:window]:
            self.context.schedule_prefetch(layer)
        for index, layer in enumerate(ordered):
            source = self.context.install(layer, require_prefetched=True,
                                          prefetch_following=False,
                                          **({'snapshot': True} if snapshot_only else {}))
            if index + window < len(ordered):
                self.context.schedule_prefetch(ordered[index + window])
            live = {}
            try:
                live = {member.qname: member for member in refresh_packed_expert_projections(
                    [projected[name] for name in layers[layer] if name in projected], self.profile)}
                with torch.no_grad():
                    for name in layers[layer]:
                        value = live[name].weight if name in live else modules[name].weight
                        shape, dtype, nbytes = shapes[name]
                        if value.is_meta or tuple(value.shape) != shape or value.dtype != dtype:
                            raise RuntimeError(f"selected source has wrong resident tensor: {name}")
                        if resource_check is not None:
                            # Where this copy lands decides whose budget it is:
                            # the ``host`` arm materializes on the CPU, and the
                            # other arm clones the resident tensor on whatever
                            # device it already lives on. Charging a device
                            # clone to the cgroup cap is the same conflation as
                            # the capture path's, one call site over.
                            reserve_allocation(
                                resource_check,
                                f"before_selected_source_copy:{name}",
                                cpu_bytes=nbytes if host else 0,
                                device_bytes=0 if host else nbytes)
                        if host:
                            copy = torch.empty(shape, dtype=dtype, device="cpu")
                            copy.copy_(value.detach())
                            weights[name] = copy
                            del copy
                        else:
                            weights[name] = value.detach().clone(
                                memory_format=torch.contiguous_format)
                        del value
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            finally:
                live.clear()
                self.context.release_completed_layer(layer)
            records.append(dict(layer=layer, units=layers[layer], source=source))
            if resource_check is not None:
                resource_check(f"after_selected_source_release:{layer}")
        return weights, dict(schema="prismaquant.selected_source_weights.v1",
            units=sorted(weights), layers=records, resident_bytes=required,
            source_forward_count=0, packed_parent_storage_retained=False,
            **({'source_snapshot_policy': 'selected-tensors-v1',
                'source_tensor_keys': list(self.context._snapshot_source_keys),
                'nonbody_materialized': False} if snapshot_only else {}))

    @contextmanager
    def pin_layer(self, layer: int) -> Iterator[None]:
        layer = int(layer)
        if self._pinned_layer is not None:
            raise RuntimeError(
                f"streamed cost already pins layer {self._pinned_layer}; "
                f"cannot also pin layer {layer}"
            )
        self.context.install(layer)
        self._pinned_layer = layer
        try:
            yield
        finally:
            self._pinned_layer = None
            self.context.unload(layer)

    @contextmanager
    def pin_layer_for_qname(self, qname: str) -> Iterator[None]:
        with self.pin_layer(self.layer_index_for_qname(qname)):
            yield

    def _head(self):
        name = str(self.profile.lm_head_name())
        try:
            return self.model.get_submodule(name)
        except (AttributeError, KeyError):
            head = getattr(self.model, "lm_head", None)
            if head is None:
                raise RuntimeError(
                    f"streamed cost could not resolve profile lm_head {name!r}"
                )
            return head

    def _prepare(self, input_ids: torch.Tensor):
        if getattr(self.context, 'source_snapshot_only', False):
            raise RuntimeError('snapshot-only source cannot execute a forward')
        ids = input_ids.to(self.device)
        position_ids = torch.arange(
            ids.size(-1), device=self.device
        ).unsqueeze(0)
        hidden = self.base_model.embed_tokens(ids).to(self.dtype)
        position_embeddings = _compute_position_embeddings(
            self.base_model, hidden, position_ids, self.profile
        )
        attention_mask = _compute_attention_mask(
            self.base_model, hidden, position_ids
        )
        hidden = self.profile.expand_hidden_for_layers(
            hidden, self.base_model
        )
        return ids, position_ids, hidden, position_embeddings, attention_mask

    def _call(self, layer: int, hidden: torch.Tensor, *, batch, pass_state):
        return _call_layer(
            self.layers[layer],
            hidden,
            position_embeddings=batch.position_embeddings,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            **self.profile.extra_layer_kwargs(input_ids=batch.input_ids),
            pass_state=pass_state,
        )

    def _finish(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.profile.collapse_hidden_after_layers(
            hidden, self.base_model
        )
        norm = _get_final_norm(self.base_model)
        if norm is not None:
            hidden = norm(hidden)
        return self._head()(hidden)

    def capture_boundaries(
        self, input_ids: torch.Tensor, *, boundary_writer=None, resource_check=None,
    ) -> StreamedForwardBoundaries:
        """Stream a no-grad source forward and retain only boundary acts."""
        ids, position_ids, hidden, position_embeddings, attention_mask = (
            self._prepare(input_ids)
        )
        batch = StreamedForwardBoundaries(
            input_ids=ids,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            activations_cpu=[],
            shared_pass_state=None,
        )
        pass_state = self.profile.new_forward_pass_state()
        if resource_check is not None:
            resource_check(batch, pass_state)
        batch.activations_cpu.append(hidden.detach().to("cpu") if boundary_writer is None
                                     else boundary_writer(0, hidden))
        for depth in range(self.prefetch_lookahead):
            self.context.schedule_prefetch(depth)
        for layer in range(self.num_layers):
            if self._pinned_layer != layer:
                self.context.install(
                    layer,
                    require_prefetched=self.require_prefetched_residency,
                )
            self.context.schedule_prefetch(layer + self.prefetch_lookahead)
            try:
                with torch.no_grad():
                    hidden = self._call(
                        layer, hidden, batch=batch, pass_state=pass_state
                    )
                if resource_check is not None:
                    resource_check(batch, pass_state)
                batch.activations_cpu.append(hidden.detach().to("cpu") if boundary_writer is None
                                             else boundary_writer(layer + 1, hidden))
            finally:
                if self._pinned_layer != layer:
                    self.context.unload(layer)
        batch.shared_pass_state = self.profile.capture_forward_pass_state(
            pass_state
        )
        if resource_check is not None:
            resource_check(batch, pass_state)
        return batch

    def isolated_layer(
        self,
        batch: StreamedForwardBoundaries,
        layer: int,
        hidden: torch.Tensor,
        *,
        pass_state: dict | None,
    ) -> torch.Tensor:
        return self._call(layer, hidden, batch=batch, pass_state=pass_state)

    def tail_logits(
        self, batch: StreamedForwardBoundaries, hidden: torch.Tensor
    ) -> torch.Tensor:
        return self._finish(hidden)

    def schedule_reverse_prefetch(self, layer: int):
        """Prefetch the next layer in this runner's reverse traversal.

        This is the reverse-direction twin of the forward loops' explicit
        ``schedule_prefetch(layer + lookahead)`` call.  Residency remains
        owned by the existing :class:`StreamingContext` / ``LayerCache``;
        this method only supplies the traversal direction so reverse AURA
        can overlap the next source-layer read with the current layer's
        render and backward work.
        """
        target = int(layer) - self.prefetch_lookahead
        if self.prefetch_lookahead <= 0 or target < 0:
            return None
        return self.context.schedule_prefetch(target)

    def __call__(self, input_ids: torch.Tensor, **_kwargs: Any):
        """Run an end-to-end no-cache forward while streaming body layers."""
        ids, position_ids, hidden, position_embeddings, attention_mask = (
            self._prepare(input_ids)
        )
        batch = StreamedForwardBoundaries(
            input_ids=ids,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            activations_cpu=[],
            shared_pass_state=None,
        )
        pass_state = self.profile.new_forward_pass_state()
        for depth in range(self.prefetch_lookahead):
            self.context.schedule_prefetch(depth)
        for layer in range(self.num_layers):
            if self._pinned_layer != layer:
                self.context.install(
                    layer,
                    require_prefetched=self.require_prefetched_residency,
                )
            self.context.schedule_prefetch(layer + self.prefetch_lookahead)
            try:
                hidden = self._call(
                    layer, hidden, batch=batch, pass_state=pass_state
                )
            finally:
                if self._pinned_layer != layer:
                    self.context.unload(layer)
        return SimpleNamespace(logits=self.tail_logits(batch, hidden))

    def shutdown(self) -> None:
        self.context.shutdown()

    def capture_layer_major_boundaries(self, input_batches, *, storage, source_phase=None,
                                       forward_recovery=None):
        """Capture exact baseline boundaries through the existing layer visitor."""
        if storage.config.get("capture_order") != "layer_major":
            raise ValueError("layer-major boundary capture requires the explicit v2 policy")
        input_batches = tuple(input_batches)
        def visit(_layer, forward_batch):
            for input_ids in input_batches:
                forward_batch(input_ids)
        return self.visit_layer_batches(input_batches, visit, boundary_storage=storage,
                                        source_phase=source_phase, forward_recovery=forward_recovery)

    def visit_layer_batches(self, input_batches, visitor, *, output_consumer=None,
                            boundary_storage=None, source_phase=None, forward_recovery=None):
        """Visit one resident source layer over the original ordered batches.

        The original visitor retains one current hidden tensor per batch. With
        explicit v2 boundary storage, it instead leases exact input windows and
        writes each original output through the existing activation owner. All
        per-batch source kwargs/pass-state remain independent and unchanged.
        The new path requires evaluation mode and observes Torch CPU plus this
        runner's CUDA RNG state around preparation/source calls. RNG-consuming
        sources refuse; this is not equivalence for arbitrary stateful models.
        """
        if self._pinned_layer is not None:
            raise RuntimeError("layer-batch traversal cannot start with a pinned layer")
        exact = boundary_storage is not None
        start_layer = 0
        if forward_recovery is not None:
            from .joint_forward_resume import require_stateless_profile
            require_stateless_profile(self)
            if not exact or len(input_batches) != forward_recovery.n_batches:
                raise RuntimeError("forward recovery changed the calibration partition count")
            start_layer = forward_recovery.frontier
            if not 0 < start_layer <= self.num_layers:
                raise RuntimeError("forward recovery frontier is outside this model")
        if source_phase is not None and (not exact or not callable(source_phase)):
            raise ValueError('source phase admission requires exact boundaries and a callable observer')
        if exact:
            if boundary_storage.config.get("capture_order") != "layer_major":
                raise ValueError("layer visitor exact storage requires layer_major v2 policy")
            if not self.require_prefetched_residency:
                raise RuntimeError("layer-major capture requires prefetched source residency")
            if any(module.training for module in self.model.modules()):
                raise RuntimeError("layer-major capture requires evaluation mode")
        states, batches = [], []
        if exact:
            boundary_storage.watch_auxiliary(batches, [])

        def checked(function, *args, **kwargs):
            if not exact:
                return function(*args, **kwargs)
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
            result = function(*args, **kwargs)
            if (not torch.equal(cpu_rng, torch.get_rng_state()) or
                    (cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(self.device)))):
                raise RuntimeError("layer-major capture observed source Torch RNG consumption")
            return result

        def check_state():
            if exact:
                boundary_storage.check_auxiliary(batches,
                    extra=[state[2] for state in states],
                    shared_extra=[state[2] for batch, state in zip(batches, states)
                                  if batch.shared_pass_state is None])

        def report_source_phase(stage, layer):
            if source_phase is not None:
                source_phase(stage, layer, boundary_storage.actual_auxiliary_bytes(
                    batches, extra=[state[2] for state in states]))

        # The runner owns residency; this visitor keeps no residency state of
        # its own. `schedule_prefetch` is idempotent: it returns the same future
        # for a read it already holds (in flight or delivered and unclaimed),
        # owns a hot layer through a completed future until its install
        # (#1124), and submits a fresh read only when nothing is held. Each
        # layer is speculated once ahead of its turn and re-asserted once
        # immediately before its install, so a speculation the runner no
        # longer holds (the pressure floor refused the read, or a hot layer
        # left the cache before the schedule owned it) gets one bounded retry and the
        # `require_prefetched` refusal stays fail-closed for anything else (#403).
        # The speculation record is the runner's future, whose result is the
        # layer's tensors; it is released at re-assert, the same moment the
        # runner drops its own reference at install, so this visitor is never
        # a second owner of a claimed layer's source bytes.
        speculated: dict[int, object] = {}
        retried: list[int] = []
        unspeculated = object()

        def speculate(layer):
            if 0 <= layer < self.num_layers and layer not in speculated:
                speculated[layer] = self.context.schedule_prefetch(layer)

        def reassert(layer):
            previous = speculated.pop(layer, unspeculated)
            held = self.context.schedule_prefetch(layer)
            if previous is not unspeculated and held is not None and held is not previous:
                retried.append(layer)

        if exact:
            self.layer_major_prefetch_retries = ()
        try:
            with torch.no_grad():
                for batch_index, input_ids in enumerate(input_batches):
                    check_state()
                    ids, positions, hidden, embeddings, mask = checked(self._prepare, input_ids)
                    pass_state = checked(self.profile.new_forward_pass_state)
                    if exact:
                        batch = StreamedForwardBoundaries(ids, positions, embeddings, mask, [], None)
                        batches.append(batch)
                        states.append([ids, None, pass_state])
                        check_state()
                        if forward_recovery is None:
                            batch.activations_cpu.append(boundary_storage.write(hidden,
                                batch_index=batch_index, boundary_index=0))
                        else:
                            refs = forward_recovery.batch_references(batch_index)
                            if any(ref.shape != tuple(hidden.shape) or ref.dtype != str(hidden.dtype)
                                   for ref in refs):
                                raise RuntimeError("forward recovery boundary tensor geometry differs")
                            batch.activations_cpu.extend(refs)
                        del hidden, pass_state
                    else:
                        states.append([ids, hidden, pass_state])
                    del positions, embeddings, mask
                if not states:
                    raise ValueError("layer-batch traversal requires calibration batches")
                if exact:
                    if start_layer < self.num_layers:
                        report_source_phase('source_loading', start_layer)
                    elif forward_recovery is not None:
                        report_source_phase('capture_forward', self.num_layers - 1)
                    for depth in range(start_layer, min(self.num_layers,
                            start_layer + max(1, self.prefetch_lookahead))):
                        speculate(depth)
                else:
                    for depth in range(min(self.num_layers, self.prefetch_lookahead + 1)):
                        self.context.schedule_prefetch(depth)
                for layer in range(start_layer, self.num_layers):
                    if layer > start_layer:
                        report_source_phase('source_loading', layer)
                    if exact:
                        check_state()
                        reassert(layer)
                        self.context.install(layer, require_prefetched=True, prefetch_following=False)
                    else:
                        self.context.install(layer, require_prefetched=self.require_prefetched_residency)
                    try:
                        if exact:
                            speculate(layer + self.prefetch_lookahead)
                        else:
                            self.context.schedule_prefetch(layer + self.prefetch_lookahead)
                        if source_phase is not None:
                            # Loading may retain old source and packing buffers;
                            # do not overlap that admitted phase with the graph
                            # workspace. The existing source owner still owns
                            # lookahead and its storage.
                            self.context.settle_prefetched_layers(range(
                                layer + 1, min(self.num_layers, layer + self.prefetch_lookahead + 1)),
                                retry_availability=True)
                            report_source_phase('capture_forward', layer)
                        next_batch = 0
                        with prefetched_boundary_batches(boundary_storage, batches, layer) if exact else nullcontext() as resident:
                            def forward_batch(input_ids):
                                nonlocal next_batch
                                if next_batch >= len(states):
                                    raise RuntimeError("layer visitor repeated a calibration batch")
                                ids, hidden, pass_state = states[next_batch]
                                if not torch.equal(input_ids.to(device=ids.device), ids):
                                    raise RuntimeError("layer visitor changed calibration batch order or tokens")
                                if exact:
                                    index, batch, cpu_hidden, _unused = next(resident)
                                    if index != next_batch:
                                        raise RuntimeError("exact boundary window changed batch order")
                                    hidden = cpu_hidden.to(device=self.device, dtype=self.dtype)
                                else:
                                    _ids, positions, initial, embeddings, mask = self._prepare(ids)
                                    del initial
                                    batch = StreamedForwardBoundaries(_ids, positions, embeddings, mask, [], None)
                                try:
                                    output = checked(self._call, layer, hidden, batch=batch, pass_state=pass_state)
                                    if exact:
                                        check_state()
                                        batch.activations_cpu.append(boundary_storage.write(output,
                                            batch_index=next_batch, boundary_index=layer + 1))
                                    else:
                                        states[next_batch][1] = output
                                    next_batch += 1
                                finally:
                                    hidden = output = None
                                    if exact:
                                        cpu_hidden = None
                            visitor(layer, forward_batch)
                            if next_batch != len(states):
                                raise RuntimeError("layer visitor omitted calibration batches")
                    finally:
                        self.context.unload(layer)
                if exact:
                    for batch, state in zip(batches, states):
                        batch.shared_pass_state = checked(self.profile.capture_forward_pass_state, state[2])
                        check_state()  # Charge the CPU capture and still-live original state together.
                        state[2] = None
                    check_state()
                    if output_consumer is not None:
                        with prefetched_boundary_batches(boundary_storage, batches, self.num_layers) as resident:
                            for index, batch, cpu_hidden, _unused in resident:
                                hidden = cpu_hidden.to(device=self.device, dtype=self.dtype)
                                output_consumer(index, checked(self.tail_logits, batch, hidden))
                                del hidden, cpu_hidden
                    return batches
                if output_consumer is not None:
                    for index, (ids, hidden, _pass_state) in enumerate(states):
                        _ids, positions, initial, embeddings, mask = self._prepare(ids)
                        del initial
                        batch = StreamedForwardBoundaries(_ids, positions, embeddings, mask, [], None)
                        output_consumer(index, self.tail_logits(batch, hidden))
        finally:
            states.clear()
            if exact:
                self.layer_major_prefetch_retries = tuple(retried)


def build_streamed_causal_lm(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    offload_folder: str,
    profile,
    cache_headroom_gb: float | None = None,
    max_cache_slots: int | None = None,
    prefetch_workers: int | str | None = None,
    prefetch_min_available_gb: float | str | None = None,
    prefetch_lookahead: int = 2,
    require_prefetched_residency: bool = False,
    attn_implementation: str | None = None,
    source_authentication=None,
    source_derivative=None,
    source_snapshot_only=False,
    sealed_head_tensors=None,
    planned_source_window_bytes: int | None = None,
) -> StreamedCausalLM:
    """Build the repository's existing streaming context and wrap it.

    ``sealed_head_tensors`` is the resident head a read manifest declared;
    the context refuses before its first head read when its own selection
    differs (PQ #1095). None checks nothing, as before.

    ``planned_source_window_bytes`` bounds the prefetch note by the sealed
    plan's source window (PQ #1134); None leaves the note as before.
    """
    from prismaquant.streaming_model import _build_streaming_context

    context = _build_streaming_context(
        model_path,
        device=device,
        dtype=dtype,
        offload_folder=offload_folder,
        cache_headroom_gb=cache_headroom_gb,
        max_cache_slots=max_cache_slots,
        prefetch_workers=prefetch_workers,
        prefetch_min_available_gb=prefetch_min_available_gb,
        log_prefix="[cost-streaming]",
        attn_implementation=attn_implementation,
        **({'source_authentication': source_authentication} if source_authentication is not None else {}),
        **({'source_snapshot_only': True} if source_snapshot_only else {}),
        **({'sealed_head_tensors': sealed_head_tensors}
           if sealed_head_tensors is not None else {}),
        **({'planned_source_window_bytes': planned_source_window_bytes}
           if planned_source_window_bytes is not None else {}),
    )
    runner = None
    try:
        effective_lookahead = max(0, int(prefetch_lookahead))
        if context.max_cache_slots is not None:
            effective_lookahead = min(
                effective_lookahead,
                max(0, context.max_cache_slots - 1),
            )
        runner = StreamedCausalLM(
            context,
            profile,
            prefetch_lookahead=effective_lookahead,
            require_prefetched_residency=require_prefetched_residency,
        )
        from .glm_source_derivative import bind_source_derivative
        bind_source_derivative(runner.model, profile, source_derivative)
    except BaseException:
        (context if runner is None else runner).shutdown()
        raise
    return runner


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _streamed_identity_stat_fingerprint(path: Path) -> dict[str, object]:
    """Return the mutation-sensitive local cache key for one source shard."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        # Unlike mtime, ctime cannot be restored with utime after a same-size
        # rewrite.  Matching all six fields makes a previously computed
        # content SHA safe to reuse without rereading a multi-hundred-GB
        # checkpoint.
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _local_checkpoint_shards(
    source_model: str | Path,
) -> tuple[dict[str, str] | None, list[Path] | None]:
    """Resolve the exact safetensors file set consumed by a local checkpoint.

    The streaming model omits auxiliary decoder namespaces it does not execute
    (DSv4's MTP towers are one example), while the exporter copies those
    tensors byte-verbatim.  The Hugging Face index is therefore the authority
    for complete source-byte coverage, not only ``context.weight_shard``.
    """
    root = Path(source_model)
    if not root.is_dir():
        return None, None
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"streamed model identity cannot read {index_path}"
            ) from exc
        weight_map = payload.get("weight_map") if isinstance(
            payload, dict
        ) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(
                f"streamed model identity requires a non-empty weight_map in "
                f"{index_path}"
            )
        canonical_map: dict[str, str] = {}
        shard_names: set[str] = set()
        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise RuntimeError(
                    f"streamed model identity found an invalid tensor name in "
                    f"{index_path}"
                )
            if (
                not isinstance(shard_name, str)
                or not shard_name
                or Path(shard_name).name != shard_name
            ):
                raise RuntimeError(
                    f"streamed model identity found an unsafe shard name "
                    f"{shard_name!r} in {index_path}"
                )
            canonical_map[tensor_name] = shard_name
            shard_names.add(shard_name)
        shard_paths = sorted(
            ((root / name).resolve() for name in shard_names), key=str
        )
        missing = [str(path) for path in shard_paths if not path.is_file()]
        if missing:
            raise RuntimeError(
                "streamed model identity checkpoint index references missing "
                f"shards: {missing[:8]}"
            )
        return dict(sorted(canonical_map.items())), shard_paths
    single = root / "model.safetensors"
    if single.is_file():
        return None, [single.resolve()]
    return None, None


SOURCE_CHECKPOINT_IDENTITY_SCHEMA = (
    "prismaquant.source_checkpoint.identity.v1"
)
# The non-shard files the standalone export reads from the checkpoint root:
# `config.json` through `stage_text_only`/`AutoConfig` (streaming_model.py:102
# and the exporter's skeleton build) and the index through
# `_local_checkpoint_shards` and export_native_compressed.py:6092.  A file that
# is absent contributes no row, so one appearing later is a different source.
SOURCE_CHECKPOINT_METADATA_FILES = (
    "config.json",
    "model.safetensors.index.json",
)
# ... plus every `*.py` at the checkpoint root, because a `trust_remote_code`
# checkpoint executes modules from there to build the skeleton (MiniMax-M2
# ships `configuration_minimax_m2.py` and `modeling_minimax_m2.py`;
# DeepSeek-V4 uses the same pattern), discovered per call rather than named
# here.  Every one of
# them is bound, whether or not `auto_map` names it: binding a module nobody
# imports can cost a false refusal, naming only some can cost a false
# admission.
SOURCE_CHECKPOINT_DIGEST_CACHE_SCHEMA = (
    "prismaquant.source_checkpoint.digest_cache.v1"
)


def _read_source_checkpoint_digest_cache(
    cache_path: Path,
) -> dict[str, dict[str, object]]:
    """Digests keyed by the six-field stat fingerprint of the file they cover.

    A corrupt or foreign cache is not an error: it simply reuses nothing.
    Both modes then hash every shard; dev mode also prints a ``[DEV-MODE]``
    line with the byte count first (PQ #1147). The cache can only ever make the identity CHEAPER,
    never different -- the fingerprint it keys on includes ``ctime_ns``, which
    ``utime`` cannot restore after an in-place same-size rewrite.
    """
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SOURCE_CHECKPOINT_DIGEST_CACHE_SCHEMA
    ):
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return {}
    reusable: dict[str, dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        fingerprint = entry.get("fingerprint")
        digest = str(entry.get("sha256", "")).lower()
        if (
            not isinstance(fingerprint, dict)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            continue
        reusable[canonical_fingerprint_key(fingerprint)] = {
            "fingerprint": fingerprint,
            "sha256": digest,
        }
    return reusable


canonical_fingerprint_key = DIRECT_ASCII_LAX.text


#: The stat fields that prove the bytes did not change. ``device`` is
#: deliberately absent: it names the client's mount of a shared export, not
#: the file, and differs across hosts for identical shards (sparky/sparklina
#: NFS mounts of one export). Certified comparisons use the full six-field
#: fingerprint; dev-portable reuse compares on these five.
_REUSE_FINGERPRINT_FIELDS = ("path", "inode", "size", "mtime_ns", "ctime_ns")
_FINGERPRINT_FIELDS = _REUSE_FINGERPRINT_FIELDS + ("device",)


def _well_formed_fingerprint(value: object) -> bool:
    """The exact six-field stat shape with typed values, nothing else.

    Both-missing and extra fields refuse: a missing field is not a match,
    and an extra field could carry mutation signal no comparison reads.
    """
    return (
        isinstance(value, dict)
        and set(value) == set(_FINGERPRINT_FIELDS)
        and isinstance(value.get("path"), str)
        and all(type(value.get(key)) is int
                for key in ("device", "inode", "size", "mtime_ns", "ctime_ns"))
    )


def stat_fingerprint_reusable(live: object, cached: object) -> bool:
    """Whether a recorded shard digest may be reused without rereading.

    Both sides must carry the exact six-field shape first; malformed rows
    never match, in either mode. Certified mode then requires full
    equality. Dev mode additionally accepts two fingerprints that agree on
    every mutation-sensitive field and differ only in the client-local
    ``device`` number. Anything else -- a moved, resized, retouched or
    replaced file -- refuses in both modes. One predicate for every
    seed/validate/build comparison so the three cannot disagree about what
    reuse means.
    """
    if not (_well_formed_fingerprint(live) and _well_formed_fingerprint(cached)):
        return False
    if live == cached:
        return True
    if not dev_mode_enabled():
        return False
    assert isinstance(live, dict) and isinstance(cached, dict)
    return all(live[key] == cached[key] for key in _REUSE_FINGERPRINT_FIELDS)


def portable_fingerprint_key(fingerprint: dict[str, object]) -> str:
    """The digest-cache lookup key ignoring the client device number."""
    return canonical_fingerprint_key(
        {key: value for key, value in fingerprint.items()
         if key != "device"})


def source_identity_hash_threads() -> int:
    """Worker count for hashing uncached source shards.

    ``PRISMAQUANT_SOURCE_HASH_THREADS`` overrides; 1 restores the serial
    loop. The default is the CPU affinity this process was given (PrismaBuild
    assigns it per action), because each worker is one sequential read plus
    one sha256 stream and ``hashlib`` releases the GIL on the 16 MiB updates:
    the pool can use exactly the cores it was admitted to and no more. There is
    no fixed I/O cap -- a slow backing store (a spinning-disk NFS pool, say)
    is the caller's measured fact, so it sets the override rather than every
    caller paying a guessed limit.
    """
    raw = str(os.environ.get("PRISMAQUANT_SOURCE_HASH_THREADS", "")).strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, len(os.sched_getaffinity(0)))


def _hash_one_source_shard(path: Path, fingerprint: dict[str, object]) -> str:
    digest = _file_sha256(path)
    if _streamed_identity_stat_fingerprint(path) != fingerprint:
        raise RuntimeError(
            f"source checkpoint shard changed while hashing: {path}"
        )
    return digest


def _hash_source_shards(
    work: list[tuple[Path, dict[str, object]]],
) -> list[str]:
    """Hash each ``(path, fingerprint)`` pair, returning digests in input order.

    The fingerprint was taken before hashing and is re-checked after, per
    shard, exactly as the serial loop does. When several shards fail, the
    error raised is the first one in input order, whatever order the workers
    finished in, so a refusal names the same shard on every run.
    """
    threads = min(source_identity_hash_threads(), len(work))
    if threads <= 1:
        return [_hash_one_source_shard(path, fp) for path, fp in work]
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(
        max_workers=threads, thread_name_prefix="source-hash"
    )
    try:
        futures = [
            pool.submit(_hash_one_source_shard, path, fp) for path, fp in work
        ]
        # `.result()` in submission order: the first failure in `ordered`
        # order is the one raised.
        return [future.result() for future in futures]
    finally:
        # On a refusal, do not start reading shards nobody will use; workers
        # already inside a read finish it (a thread cannot be interrupted).
        pool.shutdown(wait=True, cancel_futures=True)


def build_source_checkpoint_identity(
    source_model: str | Path,
    *,
    extra_shard_paths: object = (),
    digest_cache_path: str | Path | None = None,
) -> dict[str, object]:
    """Content identity of the exact safetensors byte set a run consumes.

    This is the runner-free half of :func:`build_streamed_model_identity`, for
    consumers that hold a checkpoint path rather than a live streaming runner
    -- the standalone compressed-tensors export resume cache is the first
    (PrismaQuant #340). It binds file CONTENT, so a same-size, same-header
    value edit is a different source, and the same bytes under a different
    directory are the same source.

    The weight bytes are not the whole source.
    :data:`SOURCE_CHECKPOINT_METADATA_FILES` names the non-shard files the
    export reads from the checkpoint root, and their sha256 is folded into
    the same ``content_sha256``: ``config.json`` decides the skeleton the
    payloads were quantized against -- the dtype map, ``tie_word_embeddings``,
    any ``quantization_config``, the layer counts -- and
    ``model.safetensors.index.json`` decides which shard each tensor is read
    from.  Every ``*.py`` at the checkpoint root joins them, because a
    ``trust_remote_code`` checkpoint executes modules from there to build the
    skeleton; all of them are bound rather than only those ``auto_map`` names,
    which can refuse falsely but never admit falsely.
    Editing any of them changes what a replayed ``layer_NNN.pt`` means while
    every shard byte stays identical.  They are kilobytes, so they are hashed
    on every call rather than cached.

    ``digest_cache_path`` makes the read happen once per (file, machine): a
    shard whose complete stat fingerprint still matches reuses its recorded
    digest instead of rereading that shard. Discovery, metadata hashing,
    digest-cache JSON handling and identity construction still run. Without
    this cache, every call hashes every shard.
    """
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256

    root = Path(source_model)
    _, indexed_shards = _local_checkpoint_shards(source_model)
    shard_paths = {Path(path).resolve() for path in (indexed_shards or ())}
    for path in extra_shard_paths or ():
        shard_paths.add(Path(path).resolve())
    missing = sorted(str(path) for path in shard_paths if not path.is_file())
    if missing:
        raise RuntimeError(
            f"source checkpoint identity references missing shards: "
            f"{missing[:8]}"
        )
    if not shard_paths:
        raise RuntimeError(
            f"source checkpoint identity found no safetensors shards under "
            f"{root}; refusing to stamp an unidentified source"
        )
    ordered = sorted(shard_paths, key=str)
    # Name each shard by its position INSIDE the checkpoint. A Hugging Face
    # snapshot dir holds symlinks into `blobs/`, so the resolved path is named
    # by an LFS hash; naming shards by that would make the SAME checkpoint
    # reached through a snapshot and through a plain directory two different
    # sources. Recover the in-checkpoint spelling from the directory listing.
    name_by_resolved: dict[str, str] = {}
    if root.is_dir():
        for entry in root.rglob("*.safetensors"):
            try:
                name_by_resolved.setdefault(
                    str(entry.resolve()), str(entry.relative_to(root))
                )
            except (OSError, ValueError):
                continue

    reusable = (
        _read_source_checkpoint_digest_cache(Path(digest_cache_path))
        if digest_cache_path is not None
        and Path(digest_cache_path).is_file()
        else {}
    )

    fingerprints = [_streamed_identity_stat_fingerprint(path) for path in ordered]
    portable_index: dict[str, dict[str, object]] | None = None
    if dev_mode_enabled():
        # A digest cache written on another mount of the same export keys
        # every entry under that host's device number. Re-index by the
        # portable key without touching the file format; two entries that
        # agree on everything but bytes taint the key instead of reusing.
        # Malformed stored rows are skipped outright: without the exact
        # six-field shape a row must never match, or a cache missing
        # `device` would reuse against every host.
        portable_index = {}
        tainted: set[str] = set()
        for entry in reusable.values():
            stored = entry.get("fingerprint") if isinstance(entry, dict) else None
            if not _well_formed_fingerprint(stored):
                continue
            try:
                key = portable_fingerprint_key(stored)
            except (TypeError, ValueError):
                continue
            if key in tainted:
                continue
            prior = portable_index.get(key)
            if prior is None:
                portable_index[key] = entry
            elif prior.get("sha256") != entry.get("sha256"):
                tainted.add(key)
                portable_index.pop(key, None)
    digests: list[str | None] = []
    for fingerprint in fingerprints:
        cached = reusable.get(canonical_fingerprint_key(fingerprint))
        if (cached is None and portable_index is not None
                and _well_formed_fingerprint(fingerprint)):
            cached = portable_index.get(portable_fingerprint_key(fingerprint))
        digests.append(str(cached["sha256"]) if cached is not None else None)
    misses = [index for index, digest in enumerate(digests) if digest is None]
    if misses and dev_mode_enabled():
        # The digests key every cache, so a miss is hashed in both modes
        # (PQ #1147): dev mode only says so, loudly. Campaign rows declare a
        # covering cache and never reach this.
        total = sum(int(fingerprints[index]["size"]) for index in misses)
        where = (f"the declared digest cache {digest_cache_path} does not "
                 "cover them" if digest_cache_path is not None
                 else "no digest cache is declared")
        dev_warning(
            f"source rehash of {total} bytes across {len(misses)} shard(s): "
            f"{where}; hashing them now")
    for index, digest in zip(
        misses,
        _hash_source_shards(
            [(ordered[index], fingerprints[index]) for index in misses]
        ),
    ):
        digests[index] = digest

    entries: list[dict[str, object]] = []
    shards: list[dict[str, object]] = []
    for path, fingerprint, digest in zip(ordered, fingerprints, digests):
        assert digest is not None
        entries.append({"fingerprint": fingerprint, "sha256": digest})
        # Relocating a checkpoint does not change its bytes, so the identity
        # is never keyed on the absolute path.
        name = name_by_resolved.get(str(path))
        if name is None:
            try:
                name = str(path.relative_to(root.resolve()))
            except ValueError:
                name = path.name
        shards.append({
            "name": name,
            "size": int(fingerprint["size"]),
            "sha256": digest,
        })
    shards.sort(key=lambda row: (str(row["name"]), str(row["sha256"])))

    # The non-shard files the export reads.  Kilobytes each, so no digest
    # cache: hashing them costs less than deciding not to.
    metadata_names = list(SOURCE_CHECKPOINT_METADATA_FILES)
    if root.is_dir():
        # `trust_remote_code` checkpoints build their skeleton from modules at
        # the checkpoint root, so those are read bytes too.  All of them, not
        # only the ones `auto_map` names: over-binding refuses falsely, and
        # under-binding admits falsely.
        metadata_names += sorted(
            path.name for path in root.glob("*.py") if path.is_file()
        )
    metadata: list[dict[str, object]] = []
    for name in metadata_names:
        path = root / name
        if not path.is_file():
            continue
        fingerprint = _streamed_identity_stat_fingerprint(path)
        digest = _file_sha256(path)
        if _streamed_identity_stat_fingerprint(path) != fingerprint:
            raise RuntimeError(
                f"source checkpoint metadata changed while hashing: {path}"
            )
        metadata.append({
            "name": name,
            "size": int(fingerprint["size"]),
            "sha256": digest,
        })
    metadata.sort(key=lambda row: str(row["name"]))

    identity = {
        "schema": SOURCE_CHECKPOINT_IDENTITY_SCHEMA,
        "shards": shards,
        "metadata": metadata,
        "content_sha256": canonical_json_sha256(
            {
                "schema": SOURCE_CHECKPOINT_IDENTITY_SCHEMA,
                "shards": shards,
                "metadata": metadata,
            },
            where="source checkpoint content identity",
        ),
    }

    if digest_cache_path is not None:
        from prismaquant.cost_stage_checkpoint import atomic_write_bytes

        try:
            atomic_write_bytes(
                Path(digest_cache_path),
                json.dumps(
                    {
                        "schema": SOURCE_CHECKPOINT_DIGEST_CACHE_SCHEMA,
                        "entries": entries,
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8"),
            )
        except OSError:
            # The digest cache is an optimization. A read-only or full cache
            # directory costs a re-read next time; it never changes identity.
            pass
    return identity


def _read_streamed_model_identity_cache(
    cache_path: Path,
    *,
    source_model: str,
    raw: bytes | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8")
                            if raw is None else raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"streamed model identity cache {cache_path} is corrupt; "
            "refusing identity reuse"
        ) from exc
    if (
        not isinstance(cached, dict)
        or cached.get("schema") != STREAMED_MODEL_IDENTITY_CACHE_SCHEMA
        or cached.get("source") != str(source_model)
    ):
        raise RuntimeError(
            f"streamed model identity cache {cache_path} does not bind source "
            f"{source_model!r}"
        )
    identity = validate_streamed_model_identity(
        cached.get("identity"), where="streamed model identity cache"
    )
    return cached, identity


def build_streamed_model_identity(
    runner: StreamedCausalLM,
    source_model: str,
    *,
    identity_cache_path: str | Path | None = None,
    identity_cache_bytes: bytes | None = None,
) -> dict[str, object]:
    """Hash the complete checkpoint backing a streamed cost run.

    End-to-end KL/adjoint values depend on every body/head/norm weight, so a
    path, index, or target-unit hash is insufficient.  This hashes each unique
    source shard exactly once and folds those digests together with the live
    checkpoint-key map and resolved config.  It is an initialization integrity
    pass, not a residency mechanism; decoder execution still uses the existing
    streaming cache.

    ``identity_cache_bytes`` is a read-only cache the caller already read
    and bound by digest (the Stage B head slice, PQ #1010): it is parsed and
    reused exactly like a cache file, and nothing is ever written back.
    """
    from prismaquant.cost_stage_checkpoint import (
        canonical_json,
        canonical_json_sha256,
    )

    if identity_cache_bytes is not None and identity_cache_path is not None:
        raise ValueError("a streamed identity cache is either a path or "
                         "declared bytes, never both")

    config = getattr(runner.model, "config", None)
    config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
    if "_name_or_path" in config_dict:
        # The multi-shard staging workaround loads the model from a fresh
        # mkdtemp copy each launch, so the loaded config's _name_or_path is
        # a random per-launch path; keying the identity on it makes every
        # relaunch refuse its own journal ("identity mismatch ... refusing
        # reuse or recompute"). The shard digests below are the real
        # identity — pin the path field to the caller's canonical source.
        config_dict["_name_or_path"] = str(source_model)
    mapping = {
        str(live): str(checkpoint)
        for live, checkpoint in sorted(runner.context.weight_ckpt.items())
    }
    runner_shard_paths = {
        Path(path).resolve()
        for path in runner.context.weight_shard.values()
    }
    checkpoint_weight_map, checkpoint_shard_paths = (
        _local_checkpoint_shards(source_model)
    )
    shard_paths = sorted(
        runner_shard_paths | set(checkpoint_shard_paths or ()), key=str
    )
    if not shard_paths:
        raise RuntimeError(
            "streamed model identity found no source checkpoint shards"
        )
    fingerprints = [
        _streamed_identity_stat_fingerprint(path) for path in shard_paths
    ]
    cache_path = Path(identity_cache_path) if identity_cache_path else None
    cached: dict[str, object] | None = None
    cached_identity: dict[str, object] | None = None
    have_cache = identity_cache_bytes is not None or (
        cache_path is not None and cache_path.is_file())
    if have_cache:
        cached, cached_identity = _read_streamed_model_identity_cache(
            cache_path if cache_path is not None
            else Path("<declared identity cache>"),
            source_model=str(source_model), raw=identity_cache_bytes,
        )
        stored = cached.get("fingerprints")
        reusable = (
            isinstance(stored, list)
            and len(stored) == len(fingerprints)
            and all(stat_fingerprint_reusable(live, old)
                    for live, old in zip(stored, fingerprints, strict=True))
        )
        portable = reusable and not (
            isinstance(stored, list)
            and stored == fingerprints
        )
        if portable:
            dev_warning(
                "source identity reuses "
                f"{len(fingerprints)} recorded shard digests across a "
                "client device-number difference (dev-only portable reuse; "
                "certified mode would rehash): uncertified")
        if reusable:
            if (
                cached_identity.get("config") == canonical_json(
                    config_dict, where="streamed model config"
                )
                and cached_identity.get("weight_map") == mapping
                and cached_identity.get("checkpoint_weight_map")
                == checkpoint_weight_map
            ):
                return cached_identity

    # A schema-valid old cache may cover only the executable decoder shards.
    # Reuse each digest whose complete stat fingerprint still matches, and
    # hash only newly covered files (for DSv4 this upgrades 45 cached body
    # shards by reading the three MTP shards, rather than rereading 156 GB).
    reusable_sha: dict[str, str] = {}
    mutated_paths: list[str] = []
    portable_paths: list[str] = []
    if cached is not None and cached_identity is not None:
        cached_fingerprints = cached.get("fingerprints")
        cached_shards = cached_identity.get("shards")
        if isinstance(cached_fingerprints, list) and isinstance(
            cached_shards, list
        ):
            cached_fp_by_path = {
                str(row.get("path")): row
                for row in cached_fingerprints
                if isinstance(row, dict) and isinstance(row.get("path"), str)
            }
            cached_shard_by_path = {
                str(row.get("path")): row
                for row in cached_shards
                if isinstance(row, dict) and isinstance(row.get("path"), str)
            }
            for fingerprint in fingerprints:
                path_key = str(fingerprint["path"])
                prior_fp = cached_fp_by_path.get(path_key)
                prior_shard = cached_shard_by_path.get(path_key)
                if prior_fp is None or prior_shard is None:
                    continue
                if not stat_fingerprint_reusable(fingerprint, prior_fp):
                    mutated_paths.append(path_key)
                    continue
                if (
                    isinstance(prior_shard, dict)
                    and prior_shard.get("size") == fingerprint["size"]
                    and re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(prior_shard.get("sha256", "")).lower(),
                    )
                ):
                    reusable_sha[path_key] = str(
                        prior_shard["sha256"]
                    ).lower()
                    if prior_fp != fingerprint:
                        portable_paths.append(path_key)
    if portable_paths:
        dev_warning(
            "source identity reuses "
            f"{len(portable_paths)} recorded shard digests across a "
            "client device-number difference (dev-only portable reuse; "
            "certified mode would rehash): uncertified")
    if dev_mode_enabled():
        # The digests key every cache, so uncovered shards are hashed below in
        # both modes (PQ #1147): dev mode only says so, loudly.
        if not have_cache:
            total_live = sum(
                int(fingerprint["size"]) for fingerprint in fingerprints)
            where = (f"at the declared {cache_path}" if cache_path is not None
                     else "with none declared")
            dev_warning(
                f"source rehash of {total_live} bytes: no usable identity "
                f"cache {where}; hashing every shard now")
        else:
            uncovered = [
                (str(fingerprint["path"]), int(fingerprint["size"]))
                for fingerprint in fingerprints
                if str(fingerprint["path"]) not in reusable_sha
            ]
            if uncovered:
                total = sum(size for _, size in uncovered)
                new_paths = [path for path, _ in uncovered
                             if path not in mutated_paths]
                dev_warning(
                    f"source rehash of {total} bytes across {len(uncovered)} "
                    f"shard(s) ({len(mutated_paths)} mutated, {len(new_paths)} "
                    f"new; first: {uncovered[0][0]}): the declared cache "
                    f"{cache_path} does not cover them; hashing them now")

    shards: list[dict[str, object]] = []
    for path, fingerprint in zip(shard_paths, fingerprints, strict=True):
        path_key = str(path.resolve())
        digest = reusable_sha.get(path_key)
        if digest is None:
            digest = _file_sha256(path)
            if _streamed_identity_stat_fingerprint(path) != fingerprint:
                raise RuntimeError(
                    f"source checkpoint shard changed while hashing: {path}"
                )
        shards.append({
            "path": path_key,
            "size": int(fingerprint["size"]),
            "sha256": digest,
        })
    value_bearing = {
        "config": canonical_json(config_dict, where="streamed model config"),
        "weight_map": mapping,
        "shards": shards,
    }
    if checkpoint_weight_map is not None:
        value_bearing["checkpoint_weight_map"] = checkpoint_weight_map
    identity = {
        "schema": STREAMED_MODEL_IDENTITY_SCHEMA,
        "source": str(source_model),
        "resolved_commit": getattr(config, "_commit_hash", None),
        "content_sha256": canonical_json_sha256(
            value_bearing, where="streamed model content identity"
        ),
        **value_bearing,
    }
    if cache_path is not None:
        from prismaquant.cost_stage_checkpoint import atomic_write_bytes

        atomic_write_bytes(
            cache_path,
            json.dumps(
                {
                    "schema": STREAMED_MODEL_IDENTITY_CACHE_SCHEMA,
                    "source": str(source_model),
                    "fingerprints": fingerprints,
                    "identity": identity,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
        )
    return identity


def validate_streamed_model_identity(
    identity: object, *, where: str
) -> dict[str, object]:
    """Require a value-bearing full-checkpoint identity, never a name stamp."""
    from collections.abc import Mapping
    from prismaquant.cost_stage_checkpoint import (
        canonical_json,
        canonical_json_sha256,
    )

    if not isinstance(identity, Mapping):
        raise RuntimeError(
            f"{where} requires a full streamed model identity object"
        )
    if identity.get("schema") != STREAMED_MODEL_IDENTITY_SCHEMA:
        raise RuntimeError(
            f"{where} requires model identity schema "
            f"{STREAMED_MODEL_IDENTITY_SCHEMA!r}"
        )
    digest = str(identity.get("content_sha256", "")).lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError(
            f"{where} requires exact model content_sha256"
        )
    shards = identity.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError(f"{where} requires source shard content identities")
    for index, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            raise RuntimeError(f"{where} model shard {index} is malformed")
        shard_digest = str(shard.get("sha256", "")).lower()
        if re.fullmatch(r"[0-9a-f]{64}", shard_digest) is None:
            raise RuntimeError(
                f"{where} model shard {index} lacks content sha256"
            )
    canonical = canonical_json(identity, where=f"{where} model identity")
    value_bearing = {
        "config": canonical.get("config"),
        "weight_map": canonical.get("weight_map"),
        "shards": canonical.get("shards"),
    }
    if "checkpoint_weight_map" in canonical:
        checkpoint_weight_map = canonical.get("checkpoint_weight_map")
        if not isinstance(checkpoint_weight_map, dict) or not all(
            isinstance(name, str)
            and name
            and isinstance(shard, str)
            and shard
            for name, shard in checkpoint_weight_map.items()
        ):
            raise RuntimeError(
                f"{where} model checkpoint_weight_map is malformed"
            )
        value_bearing["checkpoint_weight_map"] = checkpoint_weight_map
    expected = canonical_json_sha256(
        value_bearing, where=f"{where} model content identity"
    )
    if digest != expected:
        raise RuntimeError(
            f"{where} model content_sha256 does not match its source shard "
            "identity"
        )
    return canonical


def canonical_streamed_model_semantic_config(
    config: object,
    *,
    where: str = "streamed model config",
) -> dict[str, object]:
    """Return config semantics without host/runtime provenance fields.

    ``PretrainedConfig.to_dict()`` records the path it was loaded from and the
    installed Transformers version.  A text-only staging directory therefore
    makes two executions of the same checkpoint byte-distinct.  Those values
    remain in the v1 host-local identity for backward compatibility, but they
    cannot participate in a cross-host content join.  Strip them recursively
    so composed configs cannot reintroduce the same provenance below the top
    level.
    """
    from prismaquant.cost_stage_checkpoint import canonical_json

    canonical = canonical_json(config, where=where)
    if not isinstance(canonical, dict):
        raise RuntimeError(f"{where} must be a JSON mapping")

    def _strip(value: object) -> object:
        if isinstance(value, dict):
            return {
                str(key): _strip(item)
                for key, item in value.items()
                if key not in _STREAMED_MODEL_CONFIG_PROVENANCE_FIELDS
            }
        if isinstance(value, list):
            return [_strip(item) for item in value]
        return value

    stripped = _strip(canonical)
    assert isinstance(stripped, dict)
    return stripped


def portable_streamed_model_content_identity(
    identity: object,
    *,
    where: str = "streamed model portable content identity",
) -> dict[str, object]:
    """Derive one path-neutral digest from a complete v1 local identity.

    The serialized v1 ``content_sha256`` intentionally remains unchanged: it
    binds absolute shard paths and the exact runtime config stored in an
    identity cache.  This additive projection is derivable from old caches and
    is the value suitable for comparing independently built caches on separate
    hosts.  Local cache validation must still happen before callers use it.
    """
    from collections.abc import Mapping
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256

    canonical = validate_streamed_model_identity(identity, where=where)
    config = canonical_streamed_model_semantic_config(
        canonical.get("config"), where=f"{where} config",
    )

    def _string_map(value: object, *, field: str) -> dict[str, str]:
        if not isinstance(value, Mapping) or not value:
            raise RuntimeError(f"{where} requires nonempty {field}")
        result: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            if (
                not isinstance(raw_key, str)
                or not raw_key
                or not isinstance(raw_value, str)
                or not raw_value
            ):
                raise RuntimeError(f"{where} {field} is malformed")
            result[raw_key] = raw_value
        return dict(sorted(result.items()))

    weight_map = _string_map(canonical.get("weight_map"), field="weight_map")
    checkpoint_weight_map = _string_map(
        canonical.get("checkpoint_weight_map"),
        field="checkpoint_weight_map",
    )
    raw_shards = canonical.get("shards")
    assert isinstance(raw_shards, list)  # validated above
    shards: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for index, raw in enumerate(raw_shards):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"{where} shard {index} is malformed")
        path = raw.get("path")
        size = raw.get("size")
        sha256 = str(raw.get("sha256", "")).lower()
        if not isinstance(path, str) or not path:
            raise RuntimeError(f"{where} shard {index} has no path")
        name = Path(path).name
        if (
            not name
            or name in seen_names
            or type(size) is not int
            or int(size) < 1
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            raise RuntimeError(
                f"{where} shard {index} has no unique basename/size/content"
            )
        seen_names.add(name)
        shards.append({"name": name, "size": int(size), "sha256": sha256})
    shards.sort(key=lambda row: str(row["name"]))

    body = {
        "schema": STREAMED_MODEL_PORTABLE_CONTENT_SCHEMA,
        "source_identity_schema": STREAMED_MODEL_IDENTITY_SCHEMA,
        "config": config,
        "weight_map": weight_map,
        "checkpoint_weight_map": checkpoint_weight_map,
        "shards": shards,
    }
    return {
        "schema": STREAMED_MODEL_PORTABLE_CONTENT_SCHEMA,
        "portable_content_sha256": canonical_json_sha256(
            body, where=where,
        ),
        "checkpoint_shards": len(shards),
        "checkpoint_tensors": len(checkpoint_weight_map),
    }


def compact_streamed_model_identity(
    identity: object,
    *,
    where: str = "streamed model identity",
) -> dict[str, object]:
    """Return the compact, value-bearing identity stored in an artifact.

    The full identity can be several MiB because it carries the complete
    tensor-to-shard map and one SHA-256 per source shard.  ``content_sha256``
    already binds those fields; the compact form retains coverage counts so a
    partial checkpoint identity cannot be mistaken for the full source.
    """

    canonical = validate_streamed_model_identity(identity, where=where)
    shards = canonical.get("shards")
    checkpoint_weight_map = canonical.get("checkpoint_weight_map")
    if not isinstance(shards, list) or not isinstance(
        checkpoint_weight_map, dict
    ):
        raise RuntimeError(
            f"{where} does not attest a complete indexed checkpoint"
        )
    return {
        "schema": canonical.get("schema"),
        "content_sha256": canonical.get("content_sha256"),
        "resolved_commit": canonical.get("resolved_commit"),
        "checkpoint_shards": len(shards),
        "checkpoint_tensors": len(checkpoint_weight_map),
    }


def live_streaming_runner_config(source_model: str | Path) -> dict[str, object]:
    """Derive the live checkpoint config exactly as the streaming runner does.

    Profile-gated staging, offline ``AutoConfig`` load, then the real meta
    skeleton: the same three calls ``_build_streaming_context`` makes before
    materializing weights (``prismaquant/streaming_model.py``).  Sharing that
    derivation -- rather than normalizing after the fact -- is what keeps
    this validator bound to the config the runner actually runs:

    - a multimodal-skeleton family (``glm5_next``) runs the umbrella config,
      so a hardcoded text-only derivation can never agree with its cache;
    - the model constructor applies config defaults (measured: nested
      ``text_config.dtype`` resolves ``"bfloat16"`` at load and ``null``
      after construction), so stopping at ``AutoConfig`` also disagrees.

    Meta tensors only: no weight payload is read, no device is touched, and
    ``attn_implementation`` is left at the context default (measured to not
    affect the derived config).  Raises on any failure; callers report it
    fail-closed.
    """
    from transformers import AutoConfig

    from prismaquant.model_profiles import detect_profile
    from prismaquant.sensitivity_probe import stage_multimodal, stage_text_only
    from prismaquant.streaming_model import build_streaming_skeleton

    source = str(source_model)
    profile = detect_profile(source)
    if profile.requires_multimodal_skeleton():
        staged = stage_multimodal(source)
        multimodal = True
    else:
        staged = stage_text_only(source)
        multimodal = False
    config = AutoConfig.from_pretrained(
        staged, trust_remote_code=True, local_files_only=True
    )
    skeleton = build_streaming_skeleton(config, multimodal=multimodal)
    config_dict = skeleton.config.to_dict()
    if not isinstance(config_dict, dict):
        raise TypeError("streaming runner config is not a mapping")
    return config_dict


def validate_cached_streamed_model_identity(
    source_model: str | Path,
    identity_cache_path: str | Path,
    *,
    require_complete_checkpoint: bool = True,
) -> dict[str, object]:
    """Validate a cached full-checkpoint identity without rereading weights.

    Exact per-shard SHA-256 values remain valid only while every mutation-
    sensitive stat fingerprint matches.  For a local indexed checkpoint the
    validator also requires coverage of the complete index shard set (not just
    the decoder shards loaded by a calibration runner) and binds the complete
    tensor-to-shard map.  This is the cheap, fail-closed handoff used before a
    large streaming export.
    """
    source = str(source_model)
    cache_path = Path(identity_cache_path)
    cached, identity = _read_streamed_model_identity_cache(
        cache_path, source_model=source
    )
    fingerprints = cached.get("fingerprints")
    if not isinstance(fingerprints, list) or not fingerprints:
        raise RuntimeError(
            f"streamed model identity cache {cache_path} has no fingerprints"
        )
    fingerprint_by_path: dict[str, dict[str, object]] = {}
    for index, row in enumerate(fingerprints):
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError(
                f"streamed model identity cache fingerprint {index} is malformed"
            )
        path = str(Path(row["path"]).resolve())
        if path in fingerprint_by_path:
            raise RuntimeError(
                f"streamed model identity cache repeats shard path {path}"
            )
        fingerprint_by_path[path] = row

    identity_shards = identity.get("shards")
    assert isinstance(identity_shards, list)  # validated above
    shard_by_path: dict[str, dict[str, object]] = {}
    for index, row in enumerate(identity_shards):
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError(
                f"streamed model identity shard {index} has no exact path"
            )
        path = str(Path(row["path"]).resolve())
        if path in shard_by_path:
            raise RuntimeError(
                f"streamed model identity repeats shard path {path}"
            )
        shard_by_path[path] = row
    if set(shard_by_path) != set(fingerprint_by_path):
        raise RuntimeError(
            "streamed model identity cache fingerprint coverage differs from "
            "its value-bearing shard identity"
        )

    checkpoint_weight_map, checkpoint_paths = _local_checkpoint_shards(source)
    if require_complete_checkpoint:
        if checkpoint_paths is None:
            raise RuntimeError(
                "complete streamed model identity validation requires a local "
                "safetensors checkpoint"
            )
        expected_paths = {str(path.resolve()) for path in checkpoint_paths}
        if set(shard_by_path) != expected_paths:
            missing = sorted(expected_paths - set(shard_by_path))
            extra = sorted(set(shard_by_path) - expected_paths)
            raise RuntimeError(
                "streamed model identity does not cover the complete source "
                f"checkpoint: missing={missing[:8]}, extra={extra[:8]}"
            )
        if checkpoint_weight_map is not None and identity.get(
            "checkpoint_weight_map"
        ) != checkpoint_weight_map:
            raise RuntimeError(
                "streamed model identity tensor-to-shard map differs from the "
                "current source checkpoint index"
            )

    # The shard/index fingerprints above do not cover config.json.  Recreate
    # the live Transformers config through the same derivation the streaming
    # runner uses (profile-gated staging plus the meta skeleton, which applies
    # constructor defaults) and compare its semantic JSON to the config carried
    # by the cached content identity.  `_name_or_path` is host-local provenance
    # and `transformers_version` belongs to the separately pinned runtime image;
    # neither is model semantics.  All other fields must agree exactly, so a
    # same-shape change such as rope scaling cannot reuse old source hashes.
    try:
        config_path = Path(source) / "config.json"
        config_before = _streamed_identity_stat_fingerprint(config_path)
        live_config = canonical_streamed_model_semantic_config(
            live_streaming_runner_config(source),
            where="live streamed model config",
        )
        cached_config = canonical_streamed_model_semantic_config(
            identity.get("config"), where="cached streamed model config",
        )
        if not isinstance(live_config, dict) or not isinstance(
            cached_config, dict
        ):
            raise TypeError("streamed model config is not a mapping")
        config_after = _streamed_identity_stat_fingerprint(config_path)
    except Exception as exc:
        raise RuntimeError(
            "streamed model identity cannot validate the live source config"
        ) from exc
    if config_before != config_after:
        raise RuntimeError(
            "streamed model identity source config changed while validating"
        )
    if live_config != cached_config:
        changed = sorted(
            key for key in set(live_config) | set(cached_config)
            if live_config.get(key) != cached_config.get(key)
        )
        raise RuntimeError(
            "streamed model identity live config differs from its cached "
            f"content identity: changed={changed[:12]}"
        )

    portable = 0
    for path_key, expected in fingerprint_by_path.items():
        path = Path(path_key)
        if not path.is_file():
            raise RuntimeError(
                f"streamed model identity source shard is missing: {path}"
            )
        observed = _streamed_identity_stat_fingerprint(path)
        if not stat_fingerprint_reusable(observed, expected):
            raise RuntimeError(
                "streamed model identity source shard stat drifted; refusing "
                f"cached content SHA for {path}"
            )
        if observed != expected:
            portable += 1
        shard = shard_by_path[path_key]
        if shard.get("size") != observed["size"]:
            raise RuntimeError(
                f"streamed model identity shard size disagrees for {path}"
            )
    if portable:
        dev_warning(
            f"streamed model identity reuses {portable} recorded shard "
            "digests across a client device-number difference (dev-only "
            "portable reuse; certified mode would rehash): uncertified"
        )
    return identity
