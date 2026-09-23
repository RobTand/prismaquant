"""Declared outputs written locally, then exported by PrismaBuild.

``StreamedBoundaryArtifacts`` is this client's one user. A bound owner whose
PrismaBuild publication seals a spool root writes through it: the boundary
and cotangent entries of its produced groups, and the small-file groups of
``write_produced_files`` (the band-serial handoff record, PQ #1015). The
writer serializes into a PB-reserved group on the producing host's local
disk; PB exports the group to its canonical destinations as an ordinary
action, and the writer's own thread never waits on the pool's disks.

Nothing else writes through it. Stage A's adjoint checkpoints are written
directly into the output root (``joint_adjoint_checkpoints``), and renders
never bind it. Without a sealed root (``PRISMABUILD_PRODUCED_SPOOL_ROOT``)
a bound owner writes its entries straight into the pool; the Stage A
dispatch therefore refuses a spec that declares none (PQ #1012).

This adapter serializes nothing and moves nothing. It remembers exact writer
references until PB acknowledges immutable canonical origins. Only that ack may
advance durable entry progress or release the PB-owned local reservation. Each
entry carries PB's artifact class. Every writer today records ``payload``;
``checkpoint`` is PB's class for a checkpoint charged to the prewrite budget
its producer declared, and no writer uses it yet.

An owner whose template is write-only (PB #912, the band-serial handoff since
PQ #1075) commits each acknowledged group at its origin through
``commit_origin``; the ack is what that commit checks the origins against.

Same-box readback and write-behind (PQ #1110)
---------------------------------------------

An acknowledged group is not released at once any more. Its local copy stays
while the writer may still read one of its entries on this box, so the
reverse chain reads the cotangent plane it wrote one layer earlier from here
and never asks PrismaBuild to stage its own bytes back to itself
(``local_path``). The copy is released when the last of three things holds:

* the export is acknowledged (PB's ``release_group`` refuses before that);
* no read of it follows here: every entry was retired (``retire_entry``),
  or every entry was written with no read to follow (``read_back=False``);
* or, earlier, when the window needs its room and nothing that is not read
  again can give it (``_make_room_locked``): the oldest acknowledged group
  that is not being read and has no retired entry goes first.

The window is PrismaBuild's sealed per-owner bound (:data:`MAX_ENV`); the
owner sizes it to two cotangent planes (``two_plane_window_bytes``). No wait
here has a clock. A barrier (``await_group``, ``drain``) waits while
PrismaBuild reports the export live and refuses the moment it reports it
failed, withdrawn or done without an acknowledgement, naming the export
action and that state; a full window waits while an export that would free
room is live and refuses at once when none is. Every such refusal is also
kept as a record (``report()["refusals"]``).

Because PB's ``release_group`` re-checks every landed destination against
the export's receipt, a retired entry's canonical file must outlive the
group's release. The owner therefore unlinks it only once ``released``
says so (``StreamedBoundaryArtifacts._retire``, the deferred unlink).
"""
from __future__ import annotations

from contextlib import contextmanager
import dataclasses
from pathlib import Path
import threading
import time

ROOT_ENV = "PRISMABUILD_PRODUCED_SPOOL_ROOT"
MAX_ENV = "PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES"
#: PB's closed set of export artifact classes (``produced_spool.submit_group``).
ARTIFACT_CLASSES = ("payload", "checkpoint")
#: How often a wait looks at PrismaBuild's export state again. A cadence, not
#: a bound: nothing ends a wait but the export's own state. The value is the
#: one this adapter always polled at (PQ #989's stager deferral uses the same
#: order), and a poll is two small local reads (``poll_group``).
POLL_S = 0.1


#: The writer's per-entry file envelope over the tensor bytes: the ``nbytes +
#: 65536`` ceiling ``StreamedBoundaryArtifacts.write`` enforces and
#: ``BoundaryProducedPublication.group_ceiling_bytes`` reserves per entry.
ENTRY_HEADER_ENVELOPE_BYTES = 65536


def two_plane_window_bytes(*, n_probes, n_batches, group_size, group_ceiling=None,
                           max_entry_tensor_bytes=None):
    """The local window a same-box reverse chain needs: two cotangent planes.

    A cotangent plane is ``n_probes`` x ``n_batches`` entries, reserved in
    groups of ``group_size`` (the last one partial). The chain reads one
    plane while it writes the next, so one plane is live and one more is the
    room its writes and exports turn over in (PQ #1110). ``group_ceiling``
    maps an entry count to the bytes PrismaBuild reserves for such a group;
    without it the writer's own per-entry bound is used,
    ``max_entry_tensor_bytes`` + :data:`ENTRY_HEADER_ENVELOPE_BYTES`, which is
    what ``group_ceiling_bytes`` computes.
    """
    for name, value in (("n_probes", n_probes), ("n_batches", n_batches),
                        ("group_size", group_size)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"two-plane window {name} must be a positive int")
    if group_ceiling is None:
        if type(max_entry_tensor_bytes) is not int or max_entry_tensor_bytes <= 0:
            raise ValueError("two-plane window needs the per-entry tensor bound")
        per_entry = max_entry_tensor_bytes + ENTRY_HEADER_ENVELOPE_BYTES
        group_ceiling = lambda entries: entries * per_entry  # noqa: E731
    full, remainder = divmod(n_batches, group_size)
    per_probe = full * int(group_ceiling(group_size)) + (
        int(group_ceiling(remainder)) if remainder else 0)
    return 2 * n_probes * per_probe


class ProducedOutputSpoolRefused(RuntimeError):
    pass


class ProducedExportRefused(ProducedOutputSpoolRefused):
    """PrismaBuild reported a group's export failed; nothing will land it.

    ``export_key`` is the export action and ``state`` PrismaBuild's refusal
    for it (``export-failed-without-ack``, ``export-withdrawn-without-ack``,
    ``export-done-without-ack``, or a receipt it could not verify).
    """

    def __init__(self, batch_id, answer, *, where):
        self.batch_id = str(batch_id)
        self.export_key = answer.get("export_key")
        self.state = str(answer.get("refusal") or "export-refused")
        self.where = str(where)
        super().__init__(
            f"PB local export failed: group {self.batch_id!r} export "
            f"{self.export_key!r} is {self.state!r} at {self.where}; "
            f"answer {answer!r}")


class ProducedWindowRefused(ProducedOutputSpoolRefused):
    """The local window is full and nothing live can free it."""


class ProducedOutputSpool:
    def __init__(self, backend, *, capacity_deferred):
        self.backend = backend
        self.capacity_deferred = capacity_deferred
        self._lock = threading.RLock()
        self._groups = {}
        self._pending = set()
        self._durable_progress = []
        self._sequence = 0
        #: Local paths by canonical entry path, while their group holds them.
        self._local = {}
        self.refusals = []
        self.telemetry = {"export_wait_s": 0.0, "window_wait_s": 0.0,
                          "groups_released_retired": 0,
                          "groups_released_write_only": 0,
                          "groups_released_for_room": 0,
                          "groups_released_at_settle": 0}

    @classmethod
    def from_publication(cls, publication):
        env = getattr(publication, "env", {})
        root = env.get(ROOT_ENV)
        if not root:
            if env.get(MAX_ENV):
                raise ProducedOutputSpoolRefused(f"{MAX_ENV} requires {ROOT_ENV}")
            return None
        from .staged_lease import sdk_submodule
        module = sdk_submodule("produced_spool")
        if getattr(module, "API_VERSION", None) != 1:
            raise ProducedOutputSpoolRefused("the sealed PB helper lacks produced_spool API v1")
        backend = module.ProducedSpool(
            publication.queue, publication.instance, publication.template,
            cas_root=publication.cas_root, root=root,
            max_bytes=int(env.get(MAX_ENV, str(32 << 30))))
        return cls(backend, capacity_deferred=module.SpoolCapacityDeferred)

    @property
    def max_bytes(self):
        """PrismaBuild's sealed per-owner window, or None when it has none."""
        value = getattr(self.backend, "max_bytes", None)
        return None if value is None else int(value)

    @property
    def root(self):
        value = getattr(self.backend, "root", None)
        return None if value is None else Path(value)

    def _refuse(self, batch_id, answer, *, where):
        """Record PB's refusal of an export, then raise it."""
        exc = ProducedExportRefused(batch_id, answer, where=where)
        self.refusals.append({"batch_id": exc.batch_id,
                              "export_key": exc.export_key,
                              "state": exc.state, "where": exc.where,
                              "unix": time.time()})
        raise exc

    def reserve(self, batch_id, ceiling_bytes, *, wait=True):
        """Reserve one group's local ceiling, making room first if needed.

        When PrismaBuild defers (the window is full, or the disk lacks the
        headroom), room is made from groups no read here needs any more, and
        then from the oldest acknowledged group not being read. If that frees
        nothing, the wait is on an export that is still live and would free
        room when it lands; with none live, nothing could, and this refuses
        at once. ``wait=False`` neither waits nor takes a live group's copy:
        the stager's claim ahead of the writer must not hold its lane on an
        export, nor push out a copy the writer may read before it gets there.
        """
        waited = None
        try:
            while True:
                with self._lock:
                    if batch_id in self._groups:
                        group = self._groups[batch_id]
                        if group["ceiling_bytes"] != ceiling_bytes or group["released"]:
                            raise ProducedOutputSpoolRefused("local reservation replay changed or was released")
                        return group["directory"]
                    try:
                        directory = self.backend.reserve_group(batch_id, ceiling_bytes=ceiling_bytes)
                    except self.capacity_deferred as deferred:
                        freed = self._make_room_locked(
                            where=f"reserve {batch_id}", evict=wait)
                        if freed:
                            continue
                        live = self._live_exports_locked()
                        if not live or not wait:
                            raise ProducedWindowRefused(
                                f"local output spool cannot hold group {batch_id!r} "
                                f"({ceiling_bytes} B): PrismaBuild deferred it "
                                f"({deferred}); {self._window_summary_locked()}; "
                                + ("no export is live, so nothing will free room"
                                   if not live else "the caller does not wait"))
                    else:
                        self._sequence += 1
                        self._groups[batch_id] = dict(
                            directory=Path(directory), references=[], submitted=False,
                            durable=False, released=False,
                            ceiling_bytes=int(ceiling_bytes), sequence=self._sequence,
                            read_back=False, retired=set(), reading=0)
                        return Path(directory)
                if waited is None:
                    waited = time.monotonic()
                time.sleep(POLL_S)
        finally:
            if waited is not None:
                self.telemetry["window_wait_s"] += time.monotonic() - waited

    def directory(self, batch_id):
        with self._lock:
            return self._groups[batch_id]["directory"]

    def record(self, batch_id, local_reference, canonical_directory, *,
               artifact_class="payload", read_back=False):
        """Remember one written entry; returns its canonical reference.

        ``read_back`` is the writer saying a read of this entry may follow
        in this action; a group with such an entry keeps its local copy
        after its export lands, until those reads are done.
        """
        if artifact_class not in ARTIFACT_CLASSES:
            raise ProducedOutputSpoolRefused(
                f"artifact class must be one of {ARTIFACT_CLASSES}, not {artifact_class!r}")
        canonical = dataclasses.replace(
            local_reference,
            path=str(Path(canonical_directory) / Path(local_reference.path).name))
        with self._lock:
            group = self._groups[batch_id]
            if group["submitted"]:
                raise ProducedOutputSpoolRefused("cannot append to a submitted output group")
            if any(ref.name == local_reference.name for ref, _, _ in group["references"]):
                raise ProducedOutputSpoolRefused("local output group repeats an entry")
            group["references"].append((local_reference, canonical, artifact_class))
            group["read_back"] = group["read_back"] or bool(read_back)
            self._local[canonical.path] = (batch_id, local_reference)
        return canonical

    def submit(self, batch_id):
        with self._lock:
            group = self._groups[batch_id]
            if group["submitted"]:
                return
            entries = []
            for local, canonical, artifact_class in group["references"]:
                entry = dict(source_path=local.path, destination_path=canonical.path,
                             bytes=local.file_bytes, sha256=local.sha256)
                # PB defaults a missing class to payload; naming only the
                # other class keeps every payload group's manifest unchanged.
                if artifact_class != "payload":
                    entry["artifact_class"] = artifact_class
                entries.append(entry)
            if not entries:
                raise ProducedOutputSpoolRefused("cannot export an empty output group")
            answer = self.backend.submit_group(batch_id, entries=entries)
            if not answer.get("ok"):
                self._refuse(batch_id, answer, where="submit")
            group["submitted"] = True
            self._pending.add(batch_id)

    def _poll_locked(self, batch_id, group, *, where="poll"):
        """Look once: True once the export is acknowledged. Never waits."""
        if not group["submitted"]:
            return False
        if not group["durable"]:
            answer = self.backend.poll_group(batch_id)
            if not answer.get("ok"):
                self._refuse(batch_id, answer, where=where)
            if answer.get("complete") is not True:
                return False
            # Only the backend's verified durable receipt establishes this.
            group["durable"] = True
            group["export_key"] = answer.get("export_key")
            self._pending.discard(batch_id)
            self._durable_progress.extend(ref for _, ref, _ in group["references"])
        if not group["released"] and self._done_locked(group):
            self._release_locked(
                batch_id, group,
                "groups_released_retired" if group["read_back"]
                else "groups_released_write_only")
        return True

    @staticmethod
    def _done_locked(group):
        """Will no read of this group follow here?"""
        if group["reading"]:
            return False
        if not group["read_back"]:
            return True
        names = {canonical.name for _, canonical, _ in group["references"]}
        return bool(names) and names <= group["retired"]

    def _release_locked(self, batch_id, group, reason):
        answer = self.backend.release_group(batch_id)
        if not answer.get("ok"):
            raise ProducedOutputSpoolRefused(f"PB retained local export debt: {answer!r}")
        group["released"] = True
        for _, canonical, _ in group["references"]:
            self._local.pop(canonical.path, None)
        group["references"] = [(None, canonical, artifact_class)
                               for _, canonical, artifact_class in group["references"]]
        self.telemetry[reason] += 1

    def _live_exports_locked(self):
        return [batch_id for batch_id in self._pending
                if not self._groups[batch_id]["durable"]]

    def _poll_all_locked(self):
        # Only bounded in-flight exports are examined at a write boundary;
        # completed historical groups do not multiply polling work.
        for batch_id in tuple(self._pending):
            self._poll_locked(batch_id, self._groups[batch_id])

    def _make_room_locked(self, *, where, evict=True):
        """Release what can go; return how many groups went.

        First every acknowledged group no read here needs (released as its
        export is polled). Then, oldest first, an acknowledged group that is
        not being read and none of whose entries was retired: its entries
        are still live, so a later read of one stages it back through
        PrismaBuild, the path every read took before PQ #1110. A group with
        a retired entry is never taken here, because its retired entries'
        canonical files are unlinked only after the release, and a group
        still being consumed is read again soon.
        """
        before = sum(g["released"] for g in self._groups.values())
        for batch_id in tuple(self._pending):
            self._poll_locked(batch_id, self._groups[batch_id], where=where)
        freed = sum(g["released"] for g in self._groups.values()) - before
        if freed or not evict:
            return freed
        candidates = sorted(
            (group["sequence"], batch_id) for batch_id, group in self._groups.items()
            if group["durable"] and not group["released"]
            and not group["reading"] and not group["retired"])
        if not candidates:
            return 0
        _sequence, batch_id = candidates[0]
        self._release_locked(batch_id, self._groups[batch_id],
                             "groups_released_for_room")
        return 1

    def _window_summary_locked(self):
        held = [g for g in self._groups.values() if not g["released"]]
        return (f"window {self.max_bytes!r} B holds {len(held)} groups "
                f"({sum(g['ceiling_bytes'] for g in held)} B): "
                f"{sum(1 for g in held if not g['submitted'])} being written, "
                f"{sum(1 for g in held if g['submitted'] and not g['durable'])} exporting, "
                f"{sum(1 for g in held if g['durable'] and g['reading'])} being read, "
                f"{sum(1 for g in held if g['durable'] and g['retired'])} partly retired, "
                f"{sum(1 for g in held if g['durable'] and not g['reading'] and not g['retired'])} releasable")

    def landed(self, batch_id):
        """Look once: True when the group's export is acknowledged.

        Never waits. A caller that must not hold its thread on an export
        (the owner's stager, PQ #989) asks this and comes back later; a
        failed export raises here exactly as it does in ``await_group``.
        """
        with self._lock:
            group = self._groups[batch_id]
            if not group["submitted"]:
                raise ProducedOutputSpoolRefused("incomplete local group has no export submission")
            return self._poll_locked(batch_id, group)

    def await_group(self, batch_id, *, where="barrier"):
        """Wait, on evidence, until the group's export is acknowledged.

        No clock: the wait lasts while PrismaBuild reports the export live
        (queued, admitted or copying) and ends the moment it reports it
        landed or refused. A refusal raises :class:`ProducedExportRefused`
        naming the export action and its state, and is recorded.
        """
        started = None
        try:
            while True:
                with self._lock:
                    group = self._groups[batch_id]
                    if not group["submitted"]:
                        raise ProducedOutputSpoolRefused(
                            "incomplete local group has no export submission")
                    if self._poll_locked(batch_id, group, where=where):
                        return
                if started is None:
                    started = time.monotonic()
                time.sleep(POLL_S)
        finally:
            if started is not None:
                self.telemetry["export_wait_s"] += time.monotonic() - started

    def holds(self, batch_id):
        """Does this spool still hold the group's local copy?"""
        with self._lock:
            group = self._groups.get(batch_id)
            return group is not None and not group["released"]

    def released(self, batch_id):
        with self._lock:
            group = self._groups.get(batch_id)
            return group is not None and group["released"]

    def local_path(self, canonical_reference):
        """The local file of an entry this spool still holds, or None."""
        with self._lock:
            held = self._local.get(canonical_reference.path)
            if held is None or self._groups[held[0]]["released"]:
                return None
            return Path(held[1].path)

    @contextmanager
    def local_reads(self, references):
        """Hold the local copies of those ``references`` this spool still has.

        Yields ``{reference: local path}`` for each entry whose group this
        spool holds, and keeps those groups from being released for room
        until the read is done. Lookup and hold happen under one lock, so a
        copy it names cannot go before it is read. Entries it does not name
        are read through PrismaBuild by the caller.
        """
        held, batches = {}, []
        with self._lock:
            for reference in references:
                entry = self._local.get(getattr(reference, "path", None))
                if entry is None:
                    continue
                batch_id, local = entry
                group = self._groups[batch_id]
                if group["released"]:
                    continue
                held[reference] = Path(local.path)
                if batch_id not in batches:
                    batches.append(batch_id)
            for batch_id in batches:
                self._groups[batch_id]["reading"] += 1
        try:
            yield held
        finally:
            with self._lock:
                for batch_id in batches:
                    self._groups[batch_id]["reading"] -= 1

    def retire_entry(self, batch_id, canonical_reference):
        """No read of this entry follows here. Returns True once released.

        The group's local copy goes when its export has landed and no read
        of any of its entries follows. Never waits.
        """
        with self._lock:
            group = self._groups[batch_id]
            group["retired"].add(canonical_reference.name)
            if group["released"]:
                return True
            if group["durable"] and self._done_locked(group):
                self._release_locked(batch_id, group, "groups_released_retired")
                return True
            return False

    def commit_origin(self, batch_id, descriptors, *, lifetime):
        """Commit one exported group of a write-only owner at its origin.

        PrismaBuild's ``ProducedSpool.commit_origin_group`` (#912): only
        after the group's export is acknowledged, against the identities
        the export receipt recorded, because a retried export can still
        replace a landed copy before that. ``descriptors`` must name exactly
        the files the export landed. ``lifetime`` is the commit's (#914),
        ``retain`` or ``consumed``. Returns PrismaBuild's answer, whose
        ``ref`` a consumer declares; a refusal raises.
        """
        commit = getattr(self.backend, "commit_origin_group", None)
        if not callable(commit):
            raise ProducedOutputSpoolRefused(
                "the sealed PB helper cannot commit an exported group at its "
                "origin (PrismaBuild #912)")
        with self._lock:
            group = self._groups[batch_id]
            if not group["durable"]:
                raise ProducedOutputSpoolRefused(
                    "a group commits at its origin only after PB acknowledged "
                    "its export")
        answer = commit(batch_id, list(descriptors), lifetime=lifetime)
        if not answer.get("ok"):
            raise ProducedOutputSpoolRefused(
                f"PB refused the origin commit: {answer!r}")
        return answer

    def durable_entries(self):
        """Drain newly acknowledged references on the compute thread only."""
        with self._lock:
            self._poll_all_locked()
            committed, self._durable_progress = self._durable_progress, []
            return committed

    def drain(self, *, release=False):
        """Wait, on evidence, for every submitted group's export to land.

        ``release`` is the action's end: no read follows anywhere in it, so
        every acknowledged local copy is released too. A reservation that
        holds no entry has nothing to export and is left to PrismaBuild; one
        whose group was never completed is a writer defect and refuses.
        """
        with self._lock:
            keys = list(self._groups)
        for batch_id in keys:
            with self._lock:
                group = self._groups[batch_id]
                if not group["submitted"]:
                    if group["references"]:
                        raise ProducedOutputSpoolRefused(
                            f"local output group {batch_id!r} was never "
                            "completed, so it has no export to wait for")
                    continue
            self.await_group(batch_id, where="drain")
        if release:
            with self._lock:
                for batch_id, group in self._groups.items():
                    if group["durable"] and not group["released"]:
                        if group["reading"]:
                            raise ProducedOutputSpoolRefused(
                                f"local copy of {batch_id!r} is being read at settle")
                        self._release_locked(batch_id, group,
                                             "groups_released_at_settle")

    def release_landed(self):
        """A failing action's end: release every copy whose export landed.

        Never waits. Each live export is asked once; a refused one is
        recorded (``refusals``) and left to PrismaBuild's recovery of a dead
        producer's spool. Returns the batch ids still held.
        """
        with self._lock:
            for batch_id in tuple(self._pending):
                try:
                    self._poll_locked(batch_id, self._groups[batch_id],
                                      where="failed-exit")
                except ProducedOutputSpoolRefused:
                    continue
            for batch_id, group in self._groups.items():
                if group["durable"] and not group["released"] and not group["reading"]:
                    self._release_locked(batch_id, group, "groups_released_at_settle")
            return [batch_id for batch_id, group in self._groups.items()
                    if not group["released"]]

    def pending(self, batch_id):
        with self._lock:
            group = self._groups.get(batch_id)
            return group is not None and not group["released"]

    def report(self):
        with self._lock:
            held = [g for g in self._groups.values() if not g["released"]]
            return dict(schema="prismaquant.produced_output_spool.v2",
                        groups=len(self._groups),
                        durable_groups=sum(g["durable"] for g in self._groups.values()),
                        pending_groups=len(held),
                        resident_landed_groups=sum(g["durable"] for g in held),
                        retained_ceiling_bytes=sum(g["ceiling_bytes"] for g in held),
                        window_bytes=self.max_bytes,
                        refusals=[dict(record) for record in self.refusals],
                        **{key: value for key, value in self.telemetry.items()})
