"""Band-serial Stage B: one quantum hands its input cotangent to the next.

Inside a checkpoint band, layer quantum ``L - 1`` needs the cotangent at
boundary ``L`` as its incoming plane. In chain mode it rebuilds that plane
from the band's Stage A checkpoint through ``render_free_layer_roll``: one
render-free backward per chain layer, probe and batch group, in the Stage A
run's chain regime (PQ #997). Quantum ``L`` has already computed the same
plane: its final replay pass writes ``grad_plane[(probe, batch)] = x_in.grad``
for every probe and batch, and harvests the shared-state cotangent owners in
place. That pass is the ``replay_backward(final=True)`` body that
``render_free_layer_roll`` also runs, per sample at ``capture_batch`` 1 and
per batch group at ``capture_batch`` B (``capture_group``, the same grouping
and kernels as ``_roll_group``), so the planes are the same arithmetic when
the producer's capture batch equals the chain regime's batch size, and only
then (:func:`handoff_chain_regime_refusal`). The producer stamps its
capture batch into the handoff; the consumer refuses any other.

This module lets quantum ``L`` publish that plane and those owner states as
a **handoff**, and lets quantum ``L - 1`` read it instead of walking the
chain (PQ #996). Nothing else changes: the record, the journal identity and
the cost payload are the chain-mode ones, byte for byte.

Transport:

* The plane entries are written by a second, **published**
  ``StreamedBoundaryArtifacts`` generation under
  ``<quantum output space>/handoff/<generation>/``. That owner is Stage A's
  writer, so the entries are digest-checked exact entries under a fresh
  session. When the action is an admitted PrismaBuild owner with a
  produced-output template, the same owner binds that publication and
  writes through PrismaBuild's produced-output lifecycle, and through the
  local spool when the sealed environment enables it. Nothing here
  moves bytes itself.
* ``owner-states.pkl`` (the harvested ``SharedStateCotangents`` states) and
  ``handoff.json`` (the sealed record) are one more group of the same
  publication, of kind ``handoff-record`` (PQ #1015). The owner writes it
  (``StreamedBoundaryArtifacts.write_produced_files``) only after every
  entry group is durable, and ``handoff.json`` is its last file, so the
  record's presence implies complete entries and owner states. Through the
  spool, durable means PrismaBuild acknowledged the group's export.
  Unbound (a local or test run), the two files are written directly.
* The handoff's template is **write-only** (PrismaBuild #912, PQ #1075):
  the producer never reads its handoff back, so the template reserves no
  stage window, and every group -- each entry group, then the record
  group -- is committed at its origin as its own batch with the
  ``consumed`` lifetime (PrismaBuild #914). Without the spool a group
  commits when its last file lands; through the spool, once PrismaBuild
  acknowledged its export. The emitter returns the batch refs in commit
  order (``origin_batches``). PrismaBuild's retirement tick deletes a batch
  once every consumer that declared it has succeeded; a batch no consumer
  declared waits while its producer attempt succeeded and is swept once
  that attempt is dead. Today's consumer stages the handoff through a
  static readset and declares nothing, so its producer's batches wait;
  declaring them is the consumer's ``--after`` edge, which waits on
  PrismaBuild #946 (PQ #1007).
* The consumer streams the plane through ``stream_exact_entry_tensors``
  (windows of the verified exact-entry reader, strict-tier staged when the
  policy is active) and the owner states through the checkpoint's staged small-file
  reader. Under the one-pass spill each probe's plane is streamed during
  that probe's final pass instead of at the head (:class:`HandoffIncoming`,
  PQ #1143). The forward shared-pass states come from the consumer's own
  Stage A slice checkpoint, exactly as chain mode reads them.

The consumer refuses (``QuantumHandoffRefused``) any handoff that is not
the one its record, slice and campaign bind. It never falls back to the
chain: the choice of mode belongs to the dispatcher, and a bad handoff is
an identity failure.

KDA kernel mode (PQ #1214) runs every Stage B layer pass of a KDA layer on
the admitted capture kernel, the producer's final pass and the consumer's
chain rolls alike, so within one mode the handoff plane is still the plane
the consumer's chain rebuild ends on. Across modes it is not. A kernel-mode
producer names its kernel in the ``producer`` block (``kda_capture_kernel``:
name and identity digest); a fallback producer names none. The consumer's
load compares the name with its own launch's (:func:`load_quantum_handoff`),
and the quantum core compares the identity digest with the kernel it
admitted. A band runs in one mode. A consumer launched with the setting
unset follows its producer's mode (PQ #1252), so a band that started on the
fallback stays on it after the default moved to the kernel.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import threading
import time

HANDOFF_SCHEMA = "prismaquant.joint_quantum_handoff.v1"
HANDOFF_OWNER_STATES_SCHEMA = "prismaquant.joint_quantum_handoff.owner_states.v1"
HANDOFF_SESSION_SCHEMA = "prismaquant.joint_quantum_handoff.session.v1"
HANDOFF_DIRECTORY = "handoff"
#: Capture groups of final rows the streamed writer's tee ring holds (PQ
#: #1251). The writer takes an entry in well under a group's pass time, so
#: two groups keep the ring nearly always free; a full ring costs a scratch
#: read, never a wait.
HANDOFF_TEE_GROUPS = 2
HANDOFF_RECORD_NAME = "handoff.json"
HANDOFF_OWNER_STATES_NAME = "owner-states.pkl"
#: The produced-output group kind of ``owner-states.pkl`` and
#: ``handoff.json`` (PQ #1015): one group per handoff, group index 0.
HANDOFF_RECORD_BATCH_KIND = "handoff-record"
#: The lifetime every handoff group is committed at its origin with
#: (PrismaBuild #914, PQ #1075): PrismaBuild retires the batch once every
#: consumer that declared it has succeeded, or sweeps it once the producer
#: attempt is dead and no consumer declared it.
HANDOFF_ORIGIN_LIFETIME = "consumed"
#: The executable read phase a band-serial quantum stages its handoff under,
#: in place of the checkpoint load and the chain phases.
HANDOFF_LOAD_PHASE = "handoff-load"
#: The ``annotations.band_serial`` block of a derived band-serial readset.
#: v2 (PQ #1143): a spill-sealed readset stages each probe's incoming plane
#: in that probe's spill phase and names the moved counts in
#: ``streamed_incoming``.
BAND_SERIAL_READSET_SCHEMA = "prismaquant.joint_quantum_handoff.readset.v2"

_SEALED_FIELDS = (
    "schema", "boundary", "producer", "source", "session", "n_probes",
    "n_batches", "activation_entries", "owner_states")
_RECORD_FIELDS = (*_SEALED_FIELDS, "handoff_sha256")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class QuantumHandoffRefused(ValueError):
    """A handoff that is not the one this quantum's record and slice bind."""


def handoff_root(output_space_root) -> Path:
    """Where a quantum's handoff generations live, inside its output space."""
    return Path(output_space_root) / HANDOFF_DIRECTORY


def handoff_seal_sha256(record: Mapping) -> str:
    from .cost_stage_checkpoint import canonical_json_sha256

    return canonical_json_sha256(
        {key: record[key] for key in _SEALED_FIELDS}, where="quantum handoff")


def handoff_record_bytes(record: Mapping) -> bytes:
    if not isinstance(record, Mapping) or set(record) != set(_RECORD_FIELDS):
        raise QuantumHandoffRefused("not a quantum handoff record")
    return (json.dumps(dict(record), sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def handoff_chain_regime_refusal(run_identity: Mapping, *,
                                 capture_batch: int) -> str | None:
    """Why this Stage A run's chain cannot take a handoff captured at
    ``capture_batch``, or None.

    The handoff plane comes from the producer's final replay pass: one
    backward per (probe, group of ``capture_batch`` stored batches). A
    chain-mode quantum rebuilds its chain in the Stage A run's own regime
    (PQ #997), one backward per (probe, group of ``batch_size`` batches).
    Probe fusion is bitwise-neutral at a fixed batch size, so the two are the
    same bytes exactly when the batch sizes are equal, fusion on or off. Any
    other capture batch changes the GEMM shapes and so the rounding: the
    handoff would not be sha256-equal to that run's chain, and the run must
    rebuild it. A malformed stamp refuses with the parser's reason.
    """
    from .joint_adjoint_slices import ChainRegimeRefused, chain_regime_of

    if type(capture_batch) is not int or capture_batch < 1:
        return f"the capture batch must be a positive integer, got {capture_batch!r}"
    if not isinstance(run_identity, Mapping):
        return "the Stage A slice carries no run identity"
    try:
        regime = chain_regime_of(dict(run_identity))
    except ChainRegimeRefused as exc:
        return str(exc)
    if regime["batch_size"] != capture_batch:
        return (f"the Stage A run's chain regime has batch size "
                f"{regime['batch_size']}, and the handoff plane is captured at "
                f"batch {capture_batch}; the chain rounds differently from the "
                "producer's final pass at any other batch size")
    return None


def _source_binding(record: Mapping, adjoint_slice: Mapping) -> dict:
    """What producer and consumer must share: campaign, run, checkpoint."""
    from .joint_adjoint_slices import slice_run_header, stage_a_run_header_sha256

    campaign = record["campaign"]
    return {
        "campaign": {key: campaign[key] for key in
                     ("plan_sha256", "prepared_sha256", "read_manifest_sha256")},
        "stage_a_run_header_sha256": stage_a_run_header_sha256(
            slice_run_header(adjoint_slice)),
        "checkpoint_boundary": int(record["adjoint"]["checkpoint_boundary"]),
        "checkpoint_cotangent_sha256": str(
            adjoint_slice["checkpoint"]["cotangent_sha256"]),
    }


def _plane_coordinates(entry: Mapping) -> tuple[int, int, int]:
    """``(probe, batch, boundary)`` from an exact entry's sealed identity."""
    try:
        identity = entry["metadata"]["identity"]
        coordinates = identity["coordinates"]
        probe, batch, boundary = (coordinates["probe"], coordinates["batch"],
                                  coordinates["boundary"])
    except (KeyError, TypeError) as exc:
        raise QuantumHandoffRefused(
            f"handoff entry {entry.get('name')!r} carries no coordinates") from exc
    if identity.get("kind") != "cotangent" or any(
            type(value) is not int or value < 0
            for value in (probe, batch, boundary)):
        raise QuantumHandoffRefused(
            f"handoff entry {entry.get('name')!r} is not a cotangent entry")
    return probe, batch, boundary


def _checkpoint_coordinates(entry: Mapping) -> tuple[int, int]:
    """``(probe, batch)`` of a checkpoint plane entry, from its sealed identity.

    Read from the writer's identity rather than parsed out of the entry
    name, so a renamed checkpoint entry cannot shift the plane.
    """
    try:
        coordinates = entry["metadata"]["identity"]["coordinates"]
        probe, batch = coordinates["probe"], coordinates["batch"]
    except (KeyError, TypeError) as exc:
        raise QuantumHandoffRefused(
            f"checkpoint entry {entry.get('name')!r} carries no coordinates") from exc
    if any(type(value) is not int or value < 0 for value in (probe, batch)):
        raise QuantumHandoffRefused(
            f"checkpoint entry {entry.get('name')!r} has malformed coordinates")
    return probe, batch


def _owner_export_report(owner) -> dict | None:
    """What the handoff owner's exports did, for the row's counters.

    The owner's ``produced_output_report``: per group the export action's
    key, bytes, and when it was submitted, seen landed and released, with
    the seconds the end-of-action drain waited on it (``local_spool.
    exports``), and the owner's origin-commit seconds (PQ #1225). None for
    an unbound owner. A report that cannot be read is recorded as that,
    never raised: it must not replace the error the emit is raising.
    """
    try:
        return owner.produced_output_report()
    except Exception as exc:  # noqa: BLE001 - telemetry must not mask
        return {"report_error": repr(exc)[:400]}


class HandoffEmitCancelled(RuntimeError):
    """The quantum failed first: the handoff writer stops and writes no record."""


class HandoffEmitter:
    """Publish one quantum's final plane and owner states for ``L - 1``.

    Built by the orchestration before the core runs, so a missing
    produced-output binding, or a capture batch the slice's chain regime
    does not roll at, refuses before any GPU work. ``capture_batch`` is the
    launch regime's (``joint_replay_regime``); it is stamped into the
    handoff's ``producer`` block.

    The core opens :meth:`stream` before its retained-window driver runs
    (PQ #1251): a writer thread binds the handoff generation, then writes
    each ``(probe, batch)`` entry as soon as a final pass has stored it,
    while the later passes still run. :meth:`emit` is the same writer for a
    plane whose every slot is final already.
    """

    def __init__(self, *, record: Mapping, adjoint_slice: Mapping,
                 boundary_storage: Mapping, capture_batch: int, publication=None):
        refusal = handoff_chain_regime_refusal(adjoint_slice.get("run_identity"),
                                               capture_batch=capture_batch)
        if refusal is not None:
            raise QuantumHandoffRefused(f"cannot emit a handoff: {refusal}")
        if int(record["layer"]) < 1:
            raise QuantumHandoffRefused(
                "layer 0 has no successor quantum to hand a cotangent to")
        if publication is not None and getattr(
                publication, "write_only", False) is not True:
            raise QuantumHandoffRefused(
                "the handoff's produced-output template is not write-only: the "
                "producer never reads its handoff back, so each group commits "
                "at its origin, which only a write-only template allows "
                "(PrismaBuild #912, PQ #1075)")
        self.record = record
        self.adjoint_slice = adjoint_slice
        self.boundary_storage = dict(boundary_storage)
        self.capture_batch = int(capture_batch)
        self.publication = publication
        self.published: dict | None = None
        #: The owner's export report once the writer ended, however it ended
        #: (PQ #1225); the row's counters keep it.
        self.export_report: dict | None = None
        self._streamed = False

    def emit(self, *, grad_plane, cotangent_owners, n_probes: int,
             n_batches: int, kda_capture_kernel) -> dict:
        """Publish a plane whose every slot is final already.

        ``kda_capture_kernel`` is ``None`` for a fallback launch, else the
        admitted kernel's ``handoff_stamp()`` (PQ #1214). It has no default,
        so no caller publishes a mode it did not state.
        """
        with self.stream(grad_plane=grad_plane, n_probes=n_probes,
                         n_batches=n_batches,
                         kda_capture_kernel=kda_capture_kernel) as stream:
            stream.mark_final([(probe, batch) for probe in range(int(n_probes))
                               for batch in range(int(n_batches))])
            return stream.finish(cotangent_owners)

    def stream(self, *, grad_plane, n_probes: int, n_batches: int,
               kda_capture_kernel) -> "HandoffStream":
        """The handoff writer for a plane whose slots become final as it runs.

        Refuses a kernel stamp that is not one before any thread starts.
        """
        from .cost_streaming import normalize_boundary_storage

        if self.published is not None or self._streamed:
            raise RuntimeError("a quantum emits its handoff once")
        if kda_capture_kernel is not None:
            refusal = _kernel_stamp_refusal(kda_capture_kernel)
            if refusal is not None:
                raise QuantumHandoffRefused(f"cannot emit a handoff: {refusal}")
        record, layer = self.record, int(self.record["layer"])
        producer = {
            "quantum_id": str(record["quantum_id"]),
            "layer": layer,
            "identity_sha256": str(record["identity_sha256"]),
            "adjoint_slice_sha256": str(record["adjoint"]["slice_sha256"]),
            "chain_layers": [int(c) for c in record["adjoint"]["chain_layers"]],
            # The batch the plane was captured at (PQ #994): the consumer
            # admits it only at its slice's chain batch size.
            "capture_batch": self.capture_batch,
        }
        if kda_capture_kernel is not None:
            # Kernel mode (PQ #1214): the plane is the kernel's arithmetic.
            producer["kda_capture_kernel"] = dict(kda_capture_kernel)
        policy = normalize_boundary_storage({
            **self.boundary_storage,
            "directory": str(handoff_root(record["output_space"]["root"]))})
        self._streamed = True
        return HandoffStream(self, grad_plane=grad_plane, n_probes=int(n_probes),
                             n_batches=int(n_batches), policy=policy,
                             producer=producer,
                             source=_source_binding(record, self.adjoint_slice),
                             layer=layer)


class HandoffStream:
    """One handoff written on a thread of its own while the passes run.

    PQ #1251. The writer thread owns the handoff's ``StreamedBoundaryArtifacts``
    for its whole life: the bind, every entry write, the settle, the record
    group and the exit. No other thread calls that owner. It writes the
    entries in the order the serial emitter wrote them, probe-major and
    batch-ascending, with the same arguments, so every entry's bytes and
    name, every produced group and its commit order, and ``handoff.json``
    are the serial emitter's. Each entry waits until a final pass has
    stored its slot (:meth:`mark_final`); the main thread keeps computing.

    What does not change: every entry group is durable before the record
    group exists (``settle_local_output`` runs before
    ``write_produced_files``, on the writer), the groups commit at their
    origin only in that settle, and ``handoff.json`` is the record group's
    last file. A failure on either thread leaves no record.

    * ``__enter__`` starts the writer and waits until it has bound the
      generation and its produced output, so a binding refusal stops the
      quantum before its passes run.
    * :meth:`mark_final` records keys whose slots a final pass has stored.
      It raises the writer's error, if the writer failed, on the main
      thread.
    * :meth:`finish` hands over the pickled owner states, waits for the
      writer, and returns what the serial emitter returned.
    * ``__exit__`` without a finish cancels the writer and joins it: the
      writer stops between entries, and its owner exits on its failure path.
    """

    def __init__(self, emitter, *, grad_plane, n_probes, n_batches, policy,
                 producer, source, layer):
        from .perturbed_x_cache import ExactCotangentScratch

        self._emitter = emitter
        self._plane = grad_plane
        self._scratch = grad_plane if isinstance(grad_plane, ExactCotangentScratch) \
            else None
        self._n_probes = n_probes
        self._n_batches = n_batches
        self._policy = policy
        self._producer = producer
        self._source = source
        self._layer = layer
        self._expected = {(probe, batch) for probe in range(n_probes)
                          for batch in range(n_batches)}
        self._cond = threading.Condition()
        self._final: set = set()
        self._states: bytes | None = None
        self._cancelled = False
        self._ready = False
        self._done = False
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._buffer = None
        self._started = None
        # The tee ring (:meth:`attach_tee`): host slots that keep a final
        # row's bytes until the writer takes them, so a scratch plane's
        # final slots are not read back from the device.
        self._tee_slots: list = []
        self._tee_free: list = []
        self._tee: dict = {}
        #: The row's ``handoff_emit`` counters (PQ #1251). Written by the
        #: writer while it runs and read by the main thread after the join.
        self.telemetry = {
            "entries": len(self._expected),
            "entries_before_finish": 0,
            "bytes_before_finish": 0,
            "entries_after_finish": 0,
            "bytes_after_finish": 0,
            "writer_busy_s": 0.0,
            "finality_wait_s": 0.0,
            "max_lag_entries": 0,
            "scratch_reads": 0,
            "tee_slots": 0,
            "tee_hits": 0,
            "tee_misses": 0,
            "finish_wait_s": 0.0,
            "writer_s": 0.0,
            # Origins PrismaBuild's origin commit hashed again because their
            # timestamps moved after the spool's poll (PQ #1262, PB #1111).
            # The spool's own re-pins are in its receipts, not here.
            "landed_repins": 0,
            "cancelled": False,
            "error": None,
        }

    # -- the main thread ---------------------------------------------------

    def __enter__(self):
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="handoff-writer",
                                        daemon=True)
        self._thread.start()
        with self._cond:
            self._cond.wait_for(lambda: self._ready or self._done)
            error = self._error
        if error is not None:
            self._thread.join()
            # ``with`` runs no ``__exit__`` when ``__enter__`` raises, so the
            # refusal is recorded, and the ring let go, here (PQ #1263).
            self._release()
            self.telemetry["error"] = repr(error)[:400]
            raise error
        return self

    def attach_tee(self, slots):
        """Give the writer a ring of host slots for final rows (PQ #1251).

        ``slots`` are flat ``uint8`` CPU tensors the caller allocated after
        its guard admitted them. :meth:`mark_final` copies each final row
        into a free slot, when one is free and the row fits, and the writer
        writes the entry from that copy instead of reading the slot back
        from the cotangent scratch. The bytes are the row's either way. A
        dict plane already holds its rows in memory and takes no ring.
        """
        if self._thread is not None or self._final:
            raise RuntimeError("the tee ring is attached before the stream starts")
        if self._scratch is None:
            raise RuntimeError("only a cotangent scratch plane takes a tee ring")
        self._tee_slots = list(slots)
        self._tee_free = list(range(len(self._tee_slots)))
        self.telemetry["tee_slots"] = len(self._tee_slots)

    def mark_final(self, keys, rows=None):
        """Record that a final pass stored each of ``keys`` (PQ #1251).

        Raises the writer's error, if it failed, so the quantum stops at its
        next store rather than at its tail. A key stored twice by a final
        pass refuses: its streamed bytes may already be on their way. A
        cotangent scratch seals each key, so a later write to it refuses
        before it lands. ``rows``, the stored tensors (valid only during
        this call), fill the tee ring when it has a free slot; the main
        thread never waits for one.
        """
        keys = [(int(probe), int(batch)) for probe, batch in keys]
        with self._cond:
            if self._error is not None:
                raise self._error
            if self._done:
                raise RuntimeError("the handoff writer already ended")
            seen = set()
            for key in keys:
                if key not in self._expected:
                    raise RuntimeError(f"handoff key {key!r} is outside the plane")
                # Twice within this call too: a second tee copy of one key
                # would hold a slot nothing frees (PQ #1263).
                if key in self._final or key in seen:
                    raise RuntimeError(
                        f"a final pass stored handoff key {key!r} twice")
                seen.add(key)
            if self._scratch is not None:
                self._scratch.seal(keys)
            if rows is not None and self._tee_slots:
                for key, row in zip(keys, rows):
                    self._tee_row(key, row)
            self._final.update(keys)
            written = (self.telemetry["entries_before_finish"]
                       + self.telemetry["entries_after_finish"])
            self.telemetry["max_lag_entries"] = max(
                self.telemetry["max_lag_entries"], len(self._final) - written)
            self._cond.notify_all()

    def finish(self, cotangent_owners) -> dict:
        """Hand over the owner states, wait for the writer, return ``published``.

        The states are pickled here, on the main thread, from the harvested
        owners the serial emitter pickled: the same bytes. A key no final
        pass stored refuses: the writer never reads a slot that is not final.
        """
        missing = sorted(self._expected - self._final)
        if missing:
            raise RuntimeError(
                f"{len(missing)} handoff key(s) were never stored by a final pass, "
                f"first {missing[0]!r}")
        states = pickle.dumps(
            {"schema": HANDOFF_OWNER_STATES_SCHEMA,
             "states": [((probe, batch),
                         cotangent_owners[probe][batch].state_dict())
                        for probe in range(self._n_probes)
                        for batch in range(self._n_batches)]},
            protocol=pickle.HIGHEST_PROTOCOL)
        started = time.monotonic()
        with self._cond:
            if self._error is not None:
                raise self._error
            self._states = states
            self._cond.notify_all()
        self._thread.join()
        self.telemetry["finish_wait_s"] = time.monotonic() - started
        if self._error is not None:
            raise self._error
        return self._emitter.published

    def __exit__(self, exc_type, exc, traceback):
        if self._thread is not None and self._thread.is_alive():
            with self._cond:
                self._cancelled = True
                self._cond.notify_all()
            self._thread.join()
        self._release()
        self.telemetry["cancelled"] = self._cancelled
        if self._error is not None and not isinstance(self._error,
                                                      HandoffEmitCancelled):
            self.telemetry["error"] = repr(self._error)[:400]
        return False

    def _release(self):
        """The ring and the read buffer go with the writer."""
        self._tee_slots, self._tee_free, self._tee = [], [], {}
        self._buffer = None

    # -- the writer thread -------------------------------------------------

    def _run(self):
        from .cost_streaming import StreamedBoundaryArtifacts

        # The writer drives this owner, so its waits are the writer's own
        # (PQ #1262), not the compute thread's.
        owner = StreamedBoundaryArtifacts(self._policy, driven_by="writer")
        try:
            self._write(owner)
        except BaseException as exc:            # noqa: BLE001 - handed over
            with self._cond:
                self._error = exc
        finally:
            # Kept on success and on failure alike (PQ #1225): the counters
            # of a row that died in its export drain name the exports it
            # waited on.
            self._emitter.export_report = _owner_export_report(owner)
            self.telemetry["landed_repins"] = int(
                owner.telemetry.get("produced_landed_repins", 0))
            self.telemetry["writer_s"] = time.monotonic() - self._started
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def _await(self, ready):
        started = time.monotonic()
        with self._cond:
            self._cond.wait_for(lambda: self._cancelled or ready())
            if self._cancelled:
                raise HandoffEmitCancelled(
                    "the quantum failed before its handoff was complete")
        return time.monotonic() - started

    def _tee_row(self, key, row):
        """Copy one final row into a free tee slot, or count a miss.

        The caller holds ``_cond``. The copy is one host memcpy of the row.
        """
        nbytes = row.numel() * row.element_size()
        if not self._tee_free or nbytes > self._tee_slots[self._tee_free[-1]].numel():
            self.telemetry["tee_misses"] += 1
            return
        index = self._tee_free.pop()
        view = self._tee_slots[index].narrow(0, 0, nbytes).view(row.dtype).view(
            tuple(row.shape))
        view.copy_(row)
        self._tee[key] = (index, view)

    def _release_tee(self, key):
        with self._cond:
            held = self._tee.pop(key, None)
            if held is not None:
                self._tee_free.append(held[0])

    def _read(self, key):
        """The final bytes of ``key``: a dict plane's tensor, the tee's copy
        of a scratch row, or a scratch read into the writer's own buffer on
        the direct-I/O grid (never the shared bounce buffer, which the
        passes use)."""
        if self._scratch is None:
            return self._plane[key]
        with self._cond:
            held = self._tee.get(key)
        if held is not None:
            self.telemetry["tee_hits"] += 1
            return held[1]
        if self._buffer is None or self._scratch.slot_layout(key)[:2] != (
                tuple(self._buffer.shape), self._buffer.dtype):
            self._buffer = self._scratch.aligned_buffer(key)
        self.telemetry["scratch_reads"] += 1
        return self._scratch.read_into(key, self._buffer, bounce=False)

    def _write(self, owner):
        from .joint_adjoint_checkpoints import exact_entry_record

        emitter, layer = self._emitter, self._layer
        references = []
        with owner:
            owner.bind({"schema": HANDOFF_SESSION_SCHEMA, "producer": self._producer,
                        "source": self._source, "boundary": layer},
                       n_probes=self._n_probes, published=True)
            if emitter.publication is not None:
                owner.bind_produced_output(
                    emitter.publication,
                    group_size=int(self._policy["prefetch_batches"]),
                    n_batches=self._n_batches,
                    max_entry_tensor_bytes=max(
                        int(entry["tensor_bytes"]) for entry in
                        emitter.adjoint_slice["checkpoint"]["activation_entries"]),
                    origin_lifetime=HANDOFF_ORIGIN_LIFETIME,
                    # A failed handoff's landed, uncommitted entries are this
                    # producer's to remove before it aborts their prewrites
                    # (PQ #1251): no record names them.
                    dispose_on_failure=True)
            # A spool wait ends when the quantum fails, so the join that
            # follows a failure is bounded by the work in hand.
            owner.cancel_waits_when(lambda: self._cancelled)
            with self._cond:
                self._ready = True
                self._cond.notify_all()
            for probe in range(self._n_probes):
                for batch in range(self._n_batches):
                    key = (probe, batch)
                    self.telemetry["finality_wait_s"] += self._await(
                        lambda key=key: key in self._final)
                    busy = time.monotonic()
                    tensor = self._read(key)
                    # No read follows in this action: the successor quantum
                    # reads these entries as its own declared inputs.
                    references.append(owner.write(
                        tensor, batch_index=batch, boundary_index=layer,
                        probe_index=probe, read_back=False))
                    tensor = None
                    self._release_tee(key)
                    self.telemetry["writer_busy_s"] += time.monotonic() - busy
                    phase = ("after" if self._states is not None else "before")
                    self.telemetry[f"entries_{phase}_finish"] += 1
                    self.telemetry[f"bytes_{phase}_finish"] += int(
                        references[-1].file_bytes)
            self._await(lambda: self._states is not None)
            states = self._states
            # Every entry is durable at its canonical path before the
            # record that names it exists.
            owner.settle_local_output()
            directory = owner.directory
            session = dict(owner.session)
            states_path = directory / HANDOFF_OWNER_STATES_NAME
            handoff = {
                "schema": HANDOFF_SCHEMA,
                "boundary": layer,
                "producer": self._producer,
                "source": self._source,
                "session": session,
                "n_probes": self._n_probes,
                "n_batches": self._n_batches,
                "activation_entries": [exact_entry_record(reference)
                                       for reference in references],
                "owner_states": {
                    "name": HANDOFF_OWNER_STATES_NAME,
                    "path": str(states_path),
                    "sha256": hashlib.sha256(states).hexdigest(),
                    "file_bytes": len(states),
                },
            }
            handoff["handoff_sha256"] = handoff_seal_sha256(handoff)
            payload = handoff_record_bytes(handoff)
            path = directory / HANDOFF_RECORD_NAME
            # One produced group after every entry group has landed; the
            # record goes last, so its presence implies complete states.
            owner.write_produced_files(
                [(HANDOFF_OWNER_STATES_NAME, states),
                 (HANDOFF_RECORD_NAME, payload)],
                kind=HANDOFF_RECORD_BATCH_KIND, boundary_index=layer)
            origin_batches = owner.produced_origin_batches()
        published = {"path": str(path),
                     "sha256": hashlib.sha256(payload).hexdigest(),
                     "handoff_sha256": handoff["handoff_sha256"],
                     "boundary": layer,
                     "generation": session["generation"]}
        if emitter.publication is not None:
            # Every entry group, then the record group: the batches a
            # consumer declares, each pinned by its manifest digest.
            published["origin_batches"] = origin_batches
        emitter.published = published


def _kernel_stamp_refusal(stamp) -> str | None:
    """Why ``stamp`` is not a KDA capture kernel's handoff stamp, or None."""
    if not isinstance(stamp, Mapping) or set(stamp) != {"name", "identity_sha256"}:
        return f"the KDA capture kernel stamp {stamp!r} is not a name and an identity digest"
    if not isinstance(stamp["name"], str) or not stamp["name"]:
        return f"the KDA capture kernel stamp names no kernel: {stamp!r}"
    if not isinstance(stamp["identity_sha256"], str) or not _HEX64.fullmatch(
            stamp["identity_sha256"]):
        return f"the KDA capture kernel stamp's identity is not a sha256: {stamp!r}"
    return None


def handoff_kernel_refusal(producer: Mapping, kda_capture_kernel) -> str | None:
    """Why a consumer launched with ``kda_capture_kernel`` cannot take this plane.

    ``kda_capture_kernel`` is the consumer launch's kernel name, ``None`` for
    the fallback (PQ #1214), or ``FOLLOW_PRODUCER`` for a launch that left the
    setting unset (PQ #1252), which takes the producer's mode. The producer's
    block names its kernel, or nothing for a fallback producer. The two must
    be the same mode and, in kernel mode, the same kernel by name; the core
    binds the identity.
    """
    from .glm_kda_capture_kernel import FALLBACK, FOLLOW_PRODUCER, KDA_KERNEL_ENV

    stamp = producer.get("kda_capture_kernel")
    if stamp is not None:
        refusal = _kernel_stamp_refusal(stamp)
        if refusal is not None:
            return refusal
    if kda_capture_kernel == FOLLOW_PRODUCER:
        return None
    if stamp is None and kda_capture_kernel is None:
        return None
    if stamp is None:
        return (f"the handoff plane was captured on the fallback, and this quantum "
                f"launches in kernel mode ({kda_capture_kernel}); a band runs in one mode. "
                f"Leave {KDA_KERNEL_ENV} unset to follow the producer, or set it to "
                f"{FALLBACK}")
    if kda_capture_kernel is None:
        return (f"the handoff plane was captured in kernel mode ({stamp['name']}), and "
                "this quantum launches on the fallback; a band runs in one mode. "
                f"Leave {KDA_KERNEL_ENV} unset to follow the producer, or set it to "
                f"{stamp['name']}")
    if stamp["name"] != kda_capture_kernel:
        return (f"the handoff plane was captured by the KDA capture kernel "
                f"{stamp['name']}, and this quantum runs {kda_capture_kernel}; a band "
                f"runs in one mode, on one kernel. Leave {KDA_KERNEL_ENV} unset to "
                f"follow the producer, or set it to {stamp['name']}")
    return None


def load_quantum_handoff(path, sha256: str, *, record: Mapping,
                         adjoint_slice: Mapping, kda_capture_kernel) -> dict:
    """Read and bind the handoff a consumer quantum was given.

    Refuses unless the file hashes to ``sha256``, carries its own seal, sits
    in the producer's output space, and names this quantum's successor
    boundary, campaign, Stage A run and checkpoint. Entry coverage, shapes
    and dtypes are checked against the consumer's own slice checkpoint.
    ``kda_capture_kernel`` is the consumer launch's KDA capture kernel name,
    ``None`` for the fallback, or ``FOLLOW_PRODUCER`` when the launch left
    the setting unset (PQ #1252); it has no default. A handoff captured in
    another mode than an explicit one, or by another kernel, refuses (PQ
    #1214).
    """
    from .joint_adjoint_slices import chain_layers_for
    from .joint_layer_quanta import quantum_id

    if not isinstance(sha256, str) or not _HEX64.fullmatch(sha256):
        raise QuantumHandoffRefused("the handoff digest is not a sha256")
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise QuantumHandoffRefused(f"handoff unreadable at {path}: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise QuantumHandoffRefused(
            f"handoff at {path} does not hash to the bound digest")
    try:
        handoff = json.loads(raw)
    except ValueError as exc:
        raise QuantumHandoffRefused(f"handoff at {path} is not JSON") from exc
    if not isinstance(handoff, dict) or set(handoff) != set(_RECORD_FIELDS) \
            or handoff.get("schema") != HANDOFF_SCHEMA:
        raise QuantumHandoffRefused("not a quantum handoff record")
    if handoff["handoff_sha256"] != handoff_seal_sha256(handoff):
        raise QuantumHandoffRefused("the handoff record does not match its seal")
    if handoff_record_bytes(handoff) != raw:
        raise QuantumHandoffRefused("the handoff record is not in canonical form")

    layer = int(record["layer"])
    successor = layer + 1
    if handoff["boundary"] != successor:
        raise QuantumHandoffRefused(
            f"the handoff carries boundary {handoff['boundary']!r}; layer "
            f"{layer} consumes boundary {successor}")
    producer = handoff["producer"]
    if not isinstance(producer, dict) or producer.get("layer") != successor \
            or producer.get("quantum_id") != quantum_id(successor):
        raise QuantumHandoffRefused(
            f"the handoff was not produced by quantum {quantum_id(successor)}")
    # The plane is the chain's only at the chain's batch size: the batch the
    # producer captured at, stamped in the handoff, must be the batch size
    # this consumer's slice rolls its chain at (PQ #994, #997).
    capture_batch = producer.get("capture_batch")
    if type(capture_batch) is not int or capture_batch < 1:
        raise QuantumHandoffRefused(
            "the handoff records no capture batch for its plane")
    refusal = handoff_chain_regime_refusal(adjoint_slice.get("run_identity"),
                                           capture_batch=capture_batch)
    if refusal is not None:
        raise QuantumHandoffRefused(refusal)
    refusal = handoff_kernel_refusal(producer, kda_capture_kernel)
    if refusal is not None:
        raise QuantumHandoffRefused(refusal)
    boundary = int(record["adjoint"]["checkpoint_boundary"])
    if not successor < boundary:
        raise QuantumHandoffRefused(
            f"layer {layer} tops band {boundary}: its incoming cotangent is "
            "the checkpoint itself, and no quantum in the band produces it")
    if list(producer.get("chain_layers", ())) != list(
            chain_layers_for(boundary, successor)) or [
            int(c) for c in record["adjoint"]["chain_layers"]] != [
            *chain_layers_for(boundary, successor), successor]:
        raise QuantumHandoffRefused(
            "the producer's chain and this quantum's chain are not one band walk")
    if handoff["source"] != _source_binding(record, adjoint_slice):
        raise QuantumHandoffRefused(
            "the handoff belongs to another campaign, Stage A run or checkpoint")
    expected_space = (Path(record["output_space"]["root"]).parent
                      / quantum_id(successor) / HANDOFF_DIRECTORY)
    directory = path.parent
    if directory.parent != expected_space or path.name != HANDOFF_RECORD_NAME:
        raise QuantumHandoffRefused(
            f"the handoff at {path} is not inside {expected_space}")
    session = handoff["session"]
    if not isinstance(session, dict) or set(session) != {
            "generation", "run_identity_sha256"} \
            or session["generation"] != directory.name:
        raise QuantumHandoffRefused("the handoff session is not its generation")
    states = handoff["owner_states"]
    if not isinstance(states, dict) or states.get("path") != str(
            directory / HANDOFF_OWNER_STATES_NAME) or not _HEX64.fullmatch(
            str(states.get("sha256"))) or type(states.get("file_bytes")) is not int \
            or states["file_bytes"] <= 0:
        raise QuantumHandoffRefused("the handoff names no owner-state file")

    n_probes, n_batches = handoff["n_probes"], handoff["n_batches"]
    expected = {}
    for entry in adjoint_slice["checkpoint"]["activation_entries"]:
        key = _checkpoint_coordinates(entry)
        if key in expected:
            raise QuantumHandoffRefused(
                f"the checkpoint plane repeats probe {key[0]} batch {key[1]}")
        expected[key] = entry
    if set(expected) != {(p, b) for p in range(n_probes) for b in range(n_batches)}:
        raise QuantumHandoffRefused(
            "the handoff's probe and batch counts differ from the checkpoint plane")
    entries = handoff["activation_entries"]
    seen = set()
    for entry in entries:
        probe, batch, at = _plane_coordinates(entry)
        if at != successor or (probe, batch) in seen or (probe, batch) not in expected:
            raise QuantumHandoffRefused(
                f"handoff entry {entry.get('name')!r} is outside the plane")
        seen.add((probe, batch))
        if entry["metadata"]["identity"].get("session") != session:
            raise QuantumHandoffRefused(
                f"handoff entry {entry['name']!r} belongs to another session")
        if Path(entry["path"]).parent != directory / "entries":
            raise QuantumHandoffRefused(
                f"handoff entry {entry['name']!r} is outside its generation")
        reference = expected[(probe, batch)]
        if (entry["shape"], entry["dtype"], entry["tensor_bytes"]) != (
                reference["shape"], reference["dtype"], reference["tensor_bytes"]):
            raise QuantumHandoffRefused(
                f"handoff entry {entry['name']!r} differs in shape or dtype "
                "from the checkpoint plane")
    if seen != set(expected):
        raise QuantumHandoffRefused("the handoff does not cover the plane")
    return handoff


def _read_row(entry: Mapping) -> dict:
    """A whole staged file as an executable manifest entry."""
    return {"path": entry["path"], "offset": 0, "bytes": int(entry["file_bytes"]),
            "sha256": entry["sha256"]}


def handoff_read_entries(handoff: Mapping, checkpoint_record: Mapping, *,
                         streamed_incoming: bool = False) -> list[dict]:
    """Every staged file a band-serial quantum reads at its head, in order.

    The plane entries, the owner-state file, and the slice checkpoint's
    forward shared-pass states: the ``handoff-load`` phase of the
    consumer's executable readset. A packed (v3) checkpoint (PQ #1037)
    holds those in its one shared-state pack, staged whole.

    ``streamed_incoming`` (PQ #1143) leaves the plane out: a spill consumer
    reads each probe's plane during that probe's final pass
    (:func:`handoff_incoming_entries`), not at its head.
    """
    rows = ([] if streamed_incoming else
            [_read_row(entry) for entry in handoff["activation_entries"]])
    rows.append(_read_row(handoff["owner_states"]))
    rows.extend(_read_row(entry) for entry in _shared_pass_entries(checkpoint_record))
    return rows


def handoff_incoming_entries(handoff: Mapping, probe: int) -> list[dict]:
    """Probe ``probe``'s plane entries in batch order: what its pass reads.

    Each batch appears once, from 0 to ``n_batches - 1``; anything else
    refuses. The entries are the handoff's own records, which
    :func:`load_quantum_handoff` has checked against the checkpoint plane.
    """
    probe = int(probe)
    if not 0 <= probe < int(handoff["n_probes"]):
        raise QuantumHandoffRefused(f"the handoff carries no probe {probe}")
    by_batch = {}
    for entry in handoff["activation_entries"]:
        entry_probe, batch, _at = _plane_coordinates(entry)
        if entry_probe != probe:
            continue
        if batch in by_batch:
            raise QuantumHandoffRefused(
                f"the handoff repeats probe {probe} batch {batch}")
        by_batch[batch] = entry
    if sorted(by_batch) != list(range(int(handoff["n_batches"]))):
        raise QuantumHandoffRefused(
            f"the handoff does not cover probe {probe}'s batches")
    return [by_batch[batch] for batch in range(int(handoff["n_batches"]))]


def _shared_pass_entries(checkpoint_record: Mapping) -> list[dict]:
    """The checkpoint files that hold its forward shared-pass states.

    One pickle per batch for v1/v2; for a packed (v3) checkpoint, its one
    pack row, which also holds the shared-adjoint states (PQ #1037).
    """
    from .joint_adjoint_slices import checkpoint_is_packed

    if checkpoint_is_packed(checkpoint_record):
        return list(checkpoint_record["shared_state_entries"])
    return [entry for entry in checkpoint_record["shared_state_entries"]
            if entry["name"].startswith("shared-pass-")]


def load_handoff_inputs(handoff: Mapping, checkpoint_record: Mapping, *,
                        n_probes: int, n_batches: int, cotangent_factory=None,
                        shared_state_max_bytes: int | None = None,
                        max_resident_bytes: int | None = None, residency_check=None,
                        stream_incoming: bool = False):
    """Read what chain mode would hold after the chain, from the handoff.

    Returns ``(grad_plane, shared_adjoint, shared_pass)`` in
    :func:`~prismaquant.joint_adjoint_checkpoints.load_adjoint_checkpoint`'s
    shape: the boundary-``L`` plane keyed ``(probe, batch)``, the owner
    states keyed the same way, and the checkpoint's forward shared-pass
    states keyed by batch. Every byte is digest-verified; under an active
    tier policy every read is staged.

    The payload ceiling charges the files read whole: for a packed (v3)
    checkpoint that is the whole pack, shared-adjoint members included
    (about 285 KB on a GLM checkpoint), of which only the shared-pass
    members are deserialized.

    The plane streams in windows under ``max_resident_bytes``, charged to
    ``residency_check``, exactly as a checkpoint load streams its own
    (:func:`~prismaquant.joint_adjoint_checkpoints.stream_exact_entry_tensors`).

    ``stream_incoming`` (PQ #1143) returns the plane empty, as
    ``cotangent_factory`` built it: each probe's final pass reads its
    incoming rows through :class:`HandoffIncoming`, and the pass's stores
    fill the plane.
    """
    from .cost_streaming import _state_storage_bytes
    from .io_spans import ReadRateReporter
    from .joint_adjoint_checkpoints import (
        _await_checkpoint_entry,
        _entry_bytes,
        _read_shared_state_payload,
        read_shared_state_pack,
        shared_state_slot,
        stream_exact_entry_tensors,
    )
    from .joint_adjoint_slices import checkpoint_is_packed
    from .residency_shard_reader import staged_range_wait_s

    if (handoff["n_probes"], handoff["n_batches"]) != (int(n_probes), int(n_batches)):
        raise QuantumHandoffRefused(
            "the handoff's probe and batch counts differ from this quantum's")
    shared_pass_entries = _shared_pass_entries(checkpoint_record)
    states = handoff["owner_states"]
    if shared_state_max_bytes is not None:
        if type(shared_state_max_bytes) is not int or shared_state_max_bytes <= 0:
            raise ValueError("handoff shared-state ceiling must be positive")
        if states["file_bytes"] + sum(
                int(entry["file_bytes"]) for entry in shared_pass_entries) \
                > shared_state_max_bytes:
            raise RuntimeError("handoff shared-state payloads exceed auxiliary byte ceiling")
    deadline = time.monotonic() + staged_range_wait_s()
    entries = sorted(handoff["activation_entries"],
                     key=lambda entry: _plane_coordinates(entry)[:2])
    scratch_rows = []
    for entry in entries:
        probe, batch, _at = _plane_coordinates(entry)
        scratch_rows.append({"name": f"cotangent-{probe}-{batch}",
                             "shape": entry["shape"], "dtype": entry["dtype"],
                             "tensor_bytes": entry["tensor_bytes"]})
    plane = {} if cotangent_factory is None else cotangent_factory(scratch_rows)
    if not stream_incoming:
        # The rate and ETA lines of the checkpoint loader; no PrismaBuild units.
        rate = ReadRateReporter(
            "handoff-load", total_entries=len(entries),
            total_bytes=sum(_entry_bytes(entry) for entry in entries))
        with closing(stream_exact_entry_tensors(
                entries, expected_session=handoff["session"],
                max_resident_bytes=max_resident_bytes, residency_check=residency_check,
                deadline=deadline)) as stream:
            for entry, tensor in stream:
                probe, batch, _at = _plane_coordinates(entry)
                plane[(probe, batch)] = tensor
                del tensor
                rate.entry(_entry_bytes(entry))
        rate.done()

    def staged(path, entry, label):
        _await_checkpoint_entry(entry, deadline=deadline)
        payload = _read_shared_state_payload(Path(path), entry)
        if (hashlib.sha256(payload).hexdigest() != entry["sha256"]
                or len(payload) != entry["file_bytes"]):
            raise RuntimeError(f"{label} changed: {entry['name']}")
        return payload

    document = pickle.loads(staged(states["path"], states, "handoff owner states"))
    if not isinstance(document, dict) or document.get("schema") != \
            HANDOFF_OWNER_STATES_SCHEMA:
        raise RuntimeError("handoff owner-state file is not one")
    shared_adjoint = {}
    for key, state in document["states"]:
        key = (int(key[0]), int(key[1]))
        if key in shared_adjoint:
            raise RuntimeError("handoff owner states repeat a coordinate")
        shared_adjoint[key] = state
    if set(shared_adjoint) != {(p, b) for p in range(int(n_probes))
                               for b in range(int(n_batches))}:
        raise RuntimeError("handoff owner states do not cover the plane")
    shared_pass = {}
    if checkpoint_is_packed(checkpoint_record):
        (pack,) = shared_pass_entries
        for name, member in read_shared_state_pack(pack, deadline=deadline):
            kind, batch = shared_state_slot(name)
            if kind == "pass":
                shared_pass[batch] = pickle.loads(member)
    else:
        for entry in shared_pass_entries:
            payload = staged(entry["path"], entry,
                             "adjoint checkpoint shared-state entry")
            shared_pass[int(entry["name"].split("-")[2])] = pickle.loads(payload)
            del payload
    if shared_state_max_bytes is not None and _state_storage_bytes(
            (list(shared_adjoint.values()), list(shared_pass.values()))) \
            > shared_state_max_bytes:
        raise RuntimeError("handoff shared-state tensors exceed auxiliary byte ceiling")
    return plane, shared_adjoint, shared_pass


class HandoffIncomingStream:
    """One probe's incoming rows, read from the handoff in capture order.

    Returned by :meth:`HandoffIncoming.open`. ``take(keys)`` returns the
    next rows as verified CPU tensors and refuses any key out of the
    capture's batch order, so each row is the entry of its own
    coordinates. ``layout(key)`` reads nothing. ``telemetry`` counts the
    rows taken, their tensor bytes and the seconds the pass waited for
    them.
    """

    def __init__(self, probe, entries, stream):
        import torch

        self.probe = int(probe)
        self._entries = list(entries)
        self._stream = stream
        self._next = 0
        self._layouts = {}
        for batch, entry in enumerate(self._entries):
            dtype = getattr(torch, str(entry["dtype"]).removeprefix("torch."), None)
            if not isinstance(dtype, torch.dtype):
                raise QuantumHandoffRefused(
                    f"handoff entry {entry['name']!r} has no torch dtype")
            self._layouts[(self.probe, batch)] = (tuple(entry["shape"]), dtype)
        self.telemetry = {"probe": self.probe, "entries": 0, "tensor_bytes": 0,
                          "wait_s": 0.0}

    def layout(self, key):
        """``(shape, dtype)`` of the row at ``key``; reads nothing."""
        try:
            return self._layouts[tuple(key)]
        except KeyError:
            raise RuntimeError(
                f"probe {self.probe}'s incoming stream has no row {key!r}") from None

    def take(self, keys):
        """The rows at ``keys``, which must be the next ones in batch order."""
        rows = []
        for key in keys:
            key = tuple(key)
            if self._next >= len(self._entries) or key != (self.probe, self._next):
                raise RuntimeError(
                    f"probe {self.probe}'s incoming stream is read in capture "
                    f"order: row {key!r} was asked for, the next is "
                    f"{(self.probe, self._next)!r}")
            started = time.monotonic()
            entry, tensor = next(self._stream)
            self.telemetry["wait_s"] += time.monotonic() - started
            if entry is not self._entries[self._next] or \
                    _plane_coordinates(entry)[:2] != key:
                raise RuntimeError(
                    f"probe {self.probe}'s incoming stream yielded "
                    f"{entry.get('name')!r} for row {key!r}")
            self._next += 1
            self.telemetry["entries"] += 1
            self.telemetry["tensor_bytes"] += int(entry["tensor_bytes"])
            rows.append(tensor)
            tensor = None
        return rows

    def materialize(self, plane):
        """Write every remaining row into ``plane`` at its own key."""
        while self._next < len(self._entries):
            key = (self.probe, self._next)
            (plane[key],) = self.take([key])

    @property
    def exhausted(self):
        return self._next == len(self._entries)


class HandoffIncoming:
    """A spill consumer's incoming plane, read probe by probe (PQ #1143).

    Under the one-pass spill each incoming row is read exactly once, by its
    probe's final pass. A spill-sealed consumer's derived readset stages
    probe ``p``'s entries in ``spill-p{p}`` (:func:`band_serial_manifest`),
    and the pass reads them through :meth:`open`, a verified
    ``stream_exact_entry_tensors`` over exactly those entries in batch
    order, instead of the cotangent scratch the head used to fill. The
    bytes are the entries' own, as the head's load wrote them.
    """

    def __init__(self, handoff: Mapping, *, n_probes: int, n_batches: int):
        if (handoff["n_probes"], handoff["n_batches"]) != (int(n_probes), int(n_batches)):
            raise QuantumHandoffRefused(
                "the handoff's probe and batch counts differ from this quantum's")
        self.session = handoff["session"]
        self._entries = {probe: handoff_incoming_entries(handoff, probe)
                         for probe in range(int(n_probes))}
        self.max_entry_bytes = max(int(entry["tensor_bytes"])
                                   for rows in self._entries.values()
                                   for entry in rows)

    def entries(self, probe: int) -> list[dict]:
        return list(self._entries[int(probe)])

    @contextmanager
    def open(self, probe: int, *, max_resident_bytes: int, residency_check):
        """Stream probe ``probe``'s rows under ``max_resident_bytes``.

        The windows are charged to ``residency_check`` and each window's
        staged wait has its own bound, as a boundary prefetch window's does.
        Nothing is read until the first :meth:`HandoffIncomingStream.take`.
        A clean exit refuses when a row was left unread; any exit closes
        the stream, which waits for a read in flight and releases its charge.
        """
        from .joint_adjoint_checkpoints import stream_exact_entry_tensors

        entries = self.entries(probe)
        generator = stream_exact_entry_tensors(
            entries, expected_session=self.session,
            max_resident_bytes=int(max_resident_bytes),
            residency_check=residency_check, deadline_per_window=True)
        stream = HandoffIncomingStream(probe, entries, generator)
        failed = True
        try:
            yield stream
            failed = False
        finally:
            generator.close()
            if not failed and not stream.exhausted:
                raise RuntimeError(
                    f"probe {int(probe)}'s pass left "
                    f"{len(entries) - stream.telemetry['entries']} incoming "
                    "rows unread")


def band_serial_manifest(sealed_manifest: Mapping, handoff: Mapping,
                         checkpoint_record: Mapping, *,
                         sealed_manifest_sha256: str) -> dict:
    """The executable read manifest of a band-serial quantum (pure).

    Derived from the quantum's sealed chain-mode executable manifest and the
    bound handoff: the ``checkpoint-load`` phase and every chain phase give
    way to one ``handoff-load`` phase, placed right after ``head``, that
    stages exactly what :func:`load_handoff_inputs` reads, in its order.
    Every other phase keeps its entries. Entries are rebuilt with the
    builder's ``(path, offset)`` deduplication, so the phase byte counts and
    the prepared-input window indices follow the new index space.

    A spill-sealed readset (``annotations.replay_mode == "spill"``) streams
    the incoming plane (PQ #1143): ``handoff-load`` keeps the owner states
    and the shared-pass entries, and each ``spill-p{p}`` phase stages probe
    ``p``'s plane entries, each right after the boundary entry of its batch,
    the order its final pass reads them and the order PrismaBuild's movers
    walk a phase. The phase sequence is unchanged.
    ``annotations.band_serial.streamed_incoming`` names each phase's moved
    count, so the dispatcher still counts a capture by its stored batches.

    The record is not re-sealed: its ``executable_readset`` still names the
    chain manifest, and the dispatcher and the quantum both re-derive this
    document from it. ``annotations.band_serial`` names what it was derived
    from. Anything that is not the chain-mode consumption order refuses.
    """
    import copy

    from .joint_layer_quanta import (
        CHECKPOINT_LOAD_PHASE,
        MANIFEST_SCHEMA_V2,
        executable_bound_phase_name,
        executable_source_phase_name,
        executable_spill_phase_name,
    )

    if not isinstance(sealed_manifest, Mapping) or \
            sealed_manifest.get("schema") != MANIFEST_SCHEMA_V2:
        raise QuantumHandoffRefused("the sealed readset is not an executable manifest")
    if not isinstance(sealed_manifest_sha256, str) or \
            not _HEX64.fullmatch(sealed_manifest_sha256):
        raise QuantumHandoffRefused("the sealed readset digest is not a sha256")
    annotations = sealed_manifest.get("annotations")
    if not isinstance(annotations, Mapping) or "band_serial" in annotations:
        raise QuantumHandoffRefused("the sealed readset is not a chain-mode manifest")
    layer = annotations.get("quantum_layer")
    chain = annotations.get("chain_layers")
    if type(layer) is not int or not isinstance(chain, list):
        raise QuantumHandoffRefused("the sealed readset names no layer or chain")
    if handoff["boundary"] != layer + 1:
        raise QuantumHandoffRefused(
            f"the handoff carries boundary {handoff['boundary']!r}; the readset "
            f"is layer {layer}'s")
    replaced = [CHECKPOINT_LOAD_PHASE]
    for boundary in chain:
        replaced += [executable_source_phase_name(boundary),
                     executable_bound_phase_name(boundary)]
    phases = sealed_manifest["read_plan"]["phases"]
    if [phase["name"] for phase in phases[:len(replaced) + 1]] != ["head", *replaced]:
        raise QuantumHandoffRefused(
            "the sealed readset does not open with head, the checkpoint load "
            "and the chain phases")
    sealed_entries = sealed_manifest["entries"]
    entries: list[dict] = []
    known: dict[tuple[str, int], int] = {}

    def take(entry: Mapping) -> int:
        key = (entry["path"], int(entry["offset"]))
        index = known.get(key)
        if index is not None:
            if (entries[index]["bytes"], entries[index]["sha256"]) != (
                    entry["bytes"], entry["sha256"]):
                raise QuantumHandoffRefused(
                    f"contradictory entries stage {key[0]} offset {key[1]}")
            return index
        entries.append({"path": entry["path"], "offset": int(entry["offset"]),
                        "bytes": int(entry["bytes"]), "sha256": entry["sha256"]})
        known[key] = len(entries) - 1
        return known[key]

    remap: dict[int, int] = {}

    def kept(index: int) -> int:
        if index not in remap:
            remap[index] = take(sealed_entries[index])
        return remap[index]

    read_phases: list[dict] = []
    cumulative = 0

    def seal(name: str, indices: list[int]) -> None:
        nonlocal cumulative
        size = sum(entries[index]["bytes"] for index in indices)
        if size <= 0:
            raise QuantumHandoffRefused(f"read phase {name} is empty")
        cumulative += size
        read_phases.append({"name": name, "entry_indices": list(indices),
                            "bytes": size, "cumulative_bytes": cumulative})

    streamed = annotations.get("replay_mode") == "spill"
    incoming: dict[str, list[dict]] = {}
    if streamed:
        n_batches = int(handoff["n_batches"])
        by_name = {phase["name"]: phase for phase in phases}
        for probe in range(int(handoff["n_probes"])):
            name = executable_spill_phase_name(probe)
            if name not in by_name:
                raise QuantumHandoffRefused(
                    f"the spill-sealed readset has no {name} phase for the "
                    f"handoff's probe {probe}")
            if len(by_name[name]["entry_indices"]) != n_batches:
                raise QuantumHandoffRefused(
                    f"{name} does not stage one boundary entry per stored batch")
            incoming[name] = [_read_row(entry)
                              for entry in handoff_incoming_entries(handoff, probe)]
    seal("head", [kept(index) for index in phases[0]["entry_indices"]])
    seal(HANDOFF_LOAD_PHASE,
         [take(row) for row in handoff_read_entries(
             handoff, checkpoint_record, streamed_incoming=streamed)])
    for phase in phases[len(replaced) + 1:]:
        indices = [kept(index) for index in phase["entry_indices"]]
        rows = incoming.get(phase["name"])
        if rows is not None:
            # Batch b's incoming row right after its boundary entry.
            indices = [index for pair in zip(indices, (take(row) for row in rows))
                       for index in pair]
        seal(phase["name"], indices)
    derived_annotations = copy.deepcopy(dict(annotations))
    prepared = derived_annotations.get("prepared_input")
    if prepared is not None:
        for window in prepared["windows"]:
            if any(index not in remap for index in window["entry_indices"]):
                raise QuantumHandoffRefused(
                    f"prepared window {window.get('window_index')!r} renders "
                    "an entry no kept phase stages")
            window["entry_indices"] = [remap[index]
                                       for index in window["entry_indices"]]
    derived_annotations["band_serial"] = {
        "schema": BAND_SERIAL_READSET_SCHEMA,
        "sealed_manifest_sha256": sealed_manifest_sha256,
        "handoff_sha256": handoff["handoff_sha256"],
        "producer": handoff["producer"]["quantum_id"],
        "generation": handoff["session"]["generation"],
        "replaced_phases": replaced,
        **({"streamed_incoming": {name: len(rows) for name, rows in incoming.items()}}
           if streamed else {}),
    }
    derived = {"schema": MANIFEST_SCHEMA_V2}
    for key in ("produced_by", "mount_prefix"):
        if key in sealed_manifest:
            derived[key] = copy.deepcopy(sealed_manifest[key])
    derived.update({
        "entries": entries,
        "entry_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "annotations": derived_annotations,
        "read_plan": {"phases": read_phases, "read_bytes": cumulative},
    })
    return derived


def band_serial_manifest_bytes(record: Mapping, handoff: Mapping,
                               checkpoint_record: Mapping, *,
                               output_root) -> bytes:
    """Read the record's sealed readset and return the derived manifest wire.

    The sealed manifest must hash to the digest the record carries; only an
    executable row (one with a sealed ``executable_readset``) runs
    band-serial.
    """
    from .joint_layer_quanta import seal_manifest_bytes

    import gzip

    block = record.get("executable_readset")
    if not isinstance(block, Mapping) or not isinstance(
            block.get("manifest_path"), str) or not _HEX64.fullmatch(
            str(block.get("manifest_sha256"))):
        raise QuantumHandoffRefused(
            "band-serial quanta are executable rows: this record seals no "
            "executable readset")
    path = Path(block["manifest_path"])
    if not path.is_absolute():
        path = Path(output_root) / path
    try:
        wire = path.read_bytes()
    except OSError as exc:
        raise QuantumHandoffRefused(
            f"the sealed readset is unreadable at {path}: {exc}") from exc
    if hashlib.sha256(wire).hexdigest() != block["manifest_sha256"]:
        raise QuantumHandoffRefused(
            f"the sealed readset at {path} does not hash to the record's digest")
    try:
        sealed = json.loads(gzip.decompress(wire))
    except (OSError, EOFError, ValueError) as exc:
        raise QuantumHandoffRefused(
            f"the sealed readset at {path} is not a manifest wire") from exc
    return seal_manifest_bytes(band_serial_manifest(
        sealed, handoff, checkpoint_record,
        sealed_manifest_sha256=block["manifest_sha256"]))


def require_band_serial_readset(record: Mapping, handoff: Mapping,
                                checkpoint_record: Mapping, *, output_root,
                                data_manifest_sha256) -> None:
    """Refuse unless the staged manifest is the one this handoff derives.

    A band-serial quantum's staged reads must be exactly the derivation from
    its sealed readset and this handoff. A manifest that stages anything else
    would also fail at its first unstaged read; this check names the cause
    before any read.
    """
    if not isinstance(data_manifest_sha256, str) or \
            not _HEX64.fullmatch(data_manifest_sha256):
        raise QuantumHandoffRefused(
            "a band-serial quantum needs --data-manifest-sha256: its staged "
            "manifest is derived, not the campaign's read manifest")
    wire = band_serial_manifest_bytes(record, handoff, checkpoint_record,
                                      output_root=output_root)
    if hashlib.sha256(wire).hexdigest() != data_manifest_sha256:
        raise QuantumHandoffRefused(
            "the staged data manifest is not the band-serial readset this "
            "handoff derives from the sealed readset")


def bind_handoff_publication(*, boundary_storage: Mapping, env=None):
    """This admitted action's produced-output publication for its handoff.

    ``None`` only when the process carries no PrismaBuild launch context
    (a local or test run): the handoff then writes its entries directly.
    An admitted action that cannot bind refuses, and so does a template
    that is not write-only (PQ #1075) or whose durable payload maximum is
    not the handoff owner's artifact ceiling -- the same rule Stage A
    applies to its own entries.
    """
    from .stage_a_produced_output import (
        BoundaryProducedBindingError, BoundaryProducedPublication)

    source = dict(os.environ) if env is None else dict(env)
    if not source.get("PRISMABUILD_ACTION_KEY"):
        return None
    try:
        publication = BoundaryProducedPublication.bind_from_admitted_owner(
            queue_root=None, tier=None, env=source, command_extra=())
    except BoundaryProducedBindingError as exc:
        raise QuantumHandoffRefused(
            "this quantum is an admitted PrismaBuild action "
            f"({source['PRISMABUILD_ACTION_KEY'][:12]}) and cannot bind the "
            f"produced output its handoff needs: {exc}") from exc
    if publication.write_only is not True:
        raise QuantumHandoffRefused(
            "the handoff's produced-output template is not write-only: the "
            "producer never reads its handoff back, so each group commits at "
            "its origin, which only a write-only template allows "
            "(PrismaBuild #912, PQ #1075)")
    declared = int(publication.durable_maxima().get("payload_max_bytes", 0))
    if declared != int(boundary_storage["max_artifact_bytes"]):
        raise QuantumHandoffRefused(
            "the handoff's produced-output template declares a durable payload "
            f"maximum of {declared} bytes, but the handoff owner runs under "
            f"{int(boundary_storage['max_artifact_bytes'])}")
    return publication


__all__ = [
    "BAND_SERIAL_READSET_SCHEMA", "HANDOFF_DIRECTORY", "HANDOFF_LOAD_PHASE",
    "HANDOFF_ORIGIN_LIFETIME", "HANDOFF_OWNER_STATES_NAME",
    "HANDOFF_RECORD_BATCH_KIND",
    "HANDOFF_RECORD_NAME", "HANDOFF_SCHEMA",
    "HandoffEmitter", "HandoffIncoming", "HandoffIncomingStream",
    "QuantumHandoffRefused", "band_serial_manifest",
    "band_serial_manifest_bytes", "bind_handoff_publication",
    "handoff_chain_regime_refusal", "handoff_incoming_entries",
    "handoff_read_entries", "handoff_root",
    "handoff_seal_sha256",
    "load_handoff_inputs", "load_quantum_handoff", "require_band_serial_readset",
]
