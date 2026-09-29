"""The streaming Tessera campaign row head (RobTand/prismaquant#640).

A selected-source campaign row used to make every selected X and H resident,
hash all of them, and write its identity-bound checkpoint before the first
encode. On a GLM-5.3 E2M1 census row that head was about two minutes of NFS
reads and single-threaded sha256 with the GPU idle. This module lets the row
start encoding once the first batch's own entries are verified and resident.

**What streams.** A pool of reader threads loads one capture entry at a time
through the same verified owner the load-all prefetch uses
(``tessera_calibration_cache._verified_capture_entry``): the per-entry
checksum, geometry, census count and maximum, and finite checks all run before
the consumer sees the entry. The reader then builds that unit's run-level
receipts (``tensor_identity`` of W, X and H) and, when the unit encodes, its
bound producer identity (``_BoundCheckpointUnitIdentity``) over a per-unit
``ActivationSource``. The encode thread receives resident tensors and a holder
whose per-anchor identity is a template copy, so it hashes nothing.

**The window.** Before batch ``b`` encodes, the consumer keeps the units of
batch ``b`` and batch ``b + 1`` and nothing else: it releases every other unit's
X, H, source, holder and encoder factors, submits reads for the two batches'
missing units, and waits only for batch ``b``. At most ``2 * batch_size``
entries are resident or completed at once, and batch ``b + 1`` is read while
batch ``b`` encodes. A unit that a later, non-adjacent batch needs again is
read again, and its receipts must equal the first read's or the row refuses.

**Projected source weights stream too (PQ #1654).** On a GLM-5.3 routed row
the head used to install the whole layer (13.8 GB) to snapshot its 864 expert
views, then re-read every one of them from its shard to compare it byte for
byte with the snapshot: about 110 s of the ~165 s the GPU sat idle. The
campaign now hands such a unit to the stream as a ``meta`` placeholder, and
the unit's reader reads the producer's source tensor
(``tessera_campaign._read_projected_unit``, ``source_unit_weight``) as the
first thing it does for the unit. That tensor is the one the exporter re-reads,
so it is what is priced: there is no second view left to compare, and the
serial pass's refusal (a snapshot that differs from the shard) has nothing to
refuse. A unit is receipted and encoded on those bytes, the reads run on every
reader in parallel, and only the first batch's reads sit in front of the first
encode.

**What is deferred, and why only that.** Six writes cite the run identity, and
the run identity binds every priced unit's W, X and H receipts, so they cannot
exist before the last entry is verified. They are written at finalize, after
the loop, from the per-entry receipts (``IDENTITY_BOUND_WRITES``). The wire
blobs and render entries the loop publishes are keyed by unit and format name
only (``_wire_path``), carry no run identity, and are not cited by anything
until the journal shards are written, so they land during the loop as they
always did. A row that fails before finalize leaves wires and render entries
and none of the six.

**What a killed row keeps (PQ #1403).** Until finalize, each flush also writes
the flushed units' rows to a stream journal beside the checkpoint
(``<checkpoint>.stream``). That journal's identity is the run identity with
every unit's W, X and H receipts deferred to the unit's own shard, which
records the receipts its reader took. A relaunch opens it before the first
batch and refuses one written under another identity. For each unit it holds,
the relaunch reads the entry through the window, requires the entry's receipts
to equal the recorded ones, and passes every row through the checkpoint
resume's gates (input identity from the entry, wire receipt from the file).
Only the remainder is encoded.

**Resuming a checkpoint (PQ #1613).** A checkpoint journal
(``<checkpoint>`` and ``<checkpoint>.parts``) that an earlier attempt left,
under either head, is resumed on the stream head too. Its manifest binds every
unit's W, X and H receipts (``RECEIPT_FIELDS``), and those can only be
re-derived one entry at a time, so the relaunch takes them from the manifest
and has ``prepare_journal`` compare every other field of the run identity by
name before the first entry is read. ``RowStream.expect_identities`` then
requires each entry's own first read to reproduce its unit's recorded
receipts before the consumer sees the entry: a journalled unit is not adopted,
and a pending one is not encoded, unless its inputs are the ones the
checkpoint was priced on. Finalize compares the whole run identity, built from
this run's own receipts, as it does for a fresh journal. The row is admitted
against the window plan it was declared with (``MEMORY_PLANS``), not the
load-all plan, which on a GLM-5.3 full routed row exceeds a GB10.

**What still needs the load-all head.** ``stream_head_dependency`` names each
case. A seed adoption needs the other campaign's run identity before the
pending work is known; adaptive rounds after the first re-price units chosen
from round one; the legacy ``hessian_capture.pt`` export serializes every H;
and a run without a verified load policy has no per-entry receipt to fold.
"""
from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROW_HEAD_STREAM = "stream"
ROW_HEAD_LOAD_ALL = "load-all"
ROW_HEADS = (ROW_HEAD_STREAM, ROW_HEAD_LOAD_ALL)
EXECUTION_SCHEMA = "prismaquant.row_head_execution.v1"
EXECUTION_FILENAME = "row-head-execution.json"
#: The selected-anchor phase plan each head is admitted against: the row's own
#: admission and the dispatcher's demand read the same key.
MEMORY_PLANS = {ROW_HEAD_STREAM: "stream_memory_bytes", ROW_HEAD_LOAD_ALL: "memory_bytes"}
#: A unit's run-level receipts, in the order the run identity lists them.
RECEIPT_FIELDS = ("weight", "scoring_rows", "hessian")

#: The writes that cite the run identity, in the order finalize makes them.
IDENTITY_BOUND_WRITES = (
    "checkpoint manifest (--checkpoint)",
    "checkpoint unit shards (<checkpoint>.parts) and their pricing progress report",
    "hessian_capture.references.json",
    "input_scales.safetensors",
    "capture-load-execution-<sha256>.json",
    "cost payload (--out)",
)


def admitted_cpus() -> int:
    """The CPUs this process may run on: PrismaBuild's assigned affinity."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def resolve_identity_threads(requested) -> int:
    """Reader and identity-builder count: the request, else the admitted CPUs."""
    if requested is None:
        return max(1, admitted_cpus())
    if type(requested) is not int or requested < 1:
        raise ValueError("campaign identity threads must be a positive int")
    return requested


def checkpoint_present(checkpoint) -> bool:
    """Whether a checkpoint journal a resume would open already exists."""
    checkpoint = Path(checkpoint)
    units = checkpoint.with_name(checkpoint.name + ".parts") / "units"
    return checkpoint.exists() or (units.is_dir() and any(units.iterdir()))


def stream_head_dependency(*, row_head, selected_source, capture_load_policy,
                           export_hessian_reference_policy, max_rounds,
                           seed_checkpoint):
    """``None`` when the stream head may run; otherwise what needs load-all."""
    if row_head == ROW_HEAD_LOAD_ALL:
        return "--row-head load-all was requested"
    if row_head != ROW_HEAD_STREAM:
        raise ValueError(f"unknown row head {row_head!r}")
    if not selected_source:
        return ("the run does not reuse a hash-bound selected capture, so its X and H "
                "are not per-entry artifacts")
    if not capture_load_policy:
        return ("no --capture-load-policy: an entry has no verified load receipt to "
                "fold into the ordered load identity")
    if not export_hessian_reference_policy:
        return ("no --export-hessian-reference-policy: the legacy hessian_capture.pt "
                "export serializes every resident H in one archive")
    if int(max_rounds) != 1:
        return ("--max-rounds is not 1: rounds after the first re-price units chosen "
                "from round one's leave-one-out, so no unit's X and H can be released")
    if seed_checkpoint:
        return ("--seed-checkpoint adopts rows under the run identity, which binds "
                "every unit's inputs before the pending work is known")
    return None


@dataclass
class _Entry:
    """One verified unit, resident on the host for as long as the window keeps it."""
    name: str
    inputs: object
    hessian: object
    source: object
    holder: object
    identities: dict
    load: dict
    count: int
    max_abs: float
    read_seconds: float
    #: The source weight the reader read for a placeholder unit, until the
    #: consumer installs it into the stream's ``weights``.
    weight: object = None
    weight_seconds: float = 0.0


class RowStream:
    """Verified capture entries on reader threads, consumed a batch at a time.

    ``bind(name, *, weight, inputs, hessian, source)`` returns ``(holder,
    identities)``: the unit's bound producer identity (or ``None`` when it
    has nothing to encode) and its run-level ``weight``/``scoring_rows``/
    ``hessian`` receipts. It runs on a reader thread and must not touch a
    device. Nothing in this class calls ``resource_check`` off the thread that
    calls ``admit``: the readers' future working set is reserved there, before
    their reads are submitted.

    A unit whose ``weights`` entry is a ``meta`` placeholder has its source
    weight read here too, on the reader thread, by ``load_unit(name)``, before
    its capture entry is read: the producer's projected expert tensor, which
    is the tensor the exporter re-reads (``tessera_campaign._read_projected_unit``,
    PQ #1654). The tensor must have the placeholder's shape and dtype and live
    on the host, or the unit is refused. It is bound in place of the
    placeholder, and the consumer installs it into ``weights`` when it collects
    the entry, so the encoder and every later reader see exactly the bytes
    that were receipted. A unit already installed is never read twice. Its
    bytes are part of the reader's reserved working set
    (``reader_reserve_bytes``).
    """

    def __init__(self, *, capture_path, expected_sha256, expected_identity, census, names,
                 policy, weights, hessian_identity, bind, threads, batch_size, device,
                 memo_capacity, resource_check=None, factor_scratch_bytes=0,
                 load_unit=None, clock=time.monotonic):
        from . import tessera_calibration_cache as store
        from .perturbed_x_cache import normalize_verified_activation_load
        self._store = store
        self._clock = clock
        self._opened = clock()
        policy = normalize_verified_activation_load(policy)
        if policy is None:
            raise ValueError("the stream head requires a verified capture load policy")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("the stream head requires a positive batch size")
        if type(memo_capacity) is not int or memo_capacity < 1:
            raise ValueError("the stream head requires a positive encoder memo capacity")
        path = Path(capture_path)
        digest = store.sha256(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError("priced calibration capture manifest changed")
        manifest = store.require_capture_contract(path, expected_sha256=expected_sha256)
        names = sorted(names)
        if (manifest.get("schema") != store.SCHEMA or manifest.get("status") != "complete" or
                manifest.get("identity") != expected_identity or
                set(manifest.get("entries", {})) != set(expected_identity["units"]) or
                not set(names) <= set(expected_identity["units"])):
            raise RuntimeError("calibration capture identity, completeness or scope mismatch")
        self._max_rows = expected_identity["max_act_rows"]
        store.preflight_verified_capture_entries(path.parent, manifest["entries"], names=names,
            policy=policy, census=census, max_rows=self._max_rows)
        self.capture = dict(path=str(path.resolve()), sha256=digest)
        self._path, self._manifest, self._census = path, manifest, census
        self._policy, self._identity = policy, expected_identity
        self._names = names
        self._weights = weights
        self._hessian_identity = dict(hessian_identity)
        self._bind = bind
        placeholders = [name for name in names if weights[name].is_meta]
        if placeholders and load_unit is None:
            raise ValueError("placeholder weights require a load_unit reader: "
                             + ", ".join(placeholders[:4]))
        self._load_unit = load_unit
        self._device = device
        self._resource_check = resource_check
        self._factor_scratch_bytes = int(factor_scratch_bytes)
        self.threads = int(threads)
        self.batch_size = batch_size
        self._memo_capacity = memo_capacity
        self._memo = OrderedDict()
        self._pool = ThreadPoolExecutor(max_workers=self.threads,
                                        thread_name_prefix="row-stream-read")
        self._live: dict[str, _Entry] = {}
        self._inflight: dict = {}
        self._first: dict[str, dict] = {}
        self._expected: dict[str, dict] | None = None
        self._batches: list[list[str]] = []
        self._closed = False
        self._lock = threading.Lock()
        self.stats = dict(entries_read=0, rereads=0, hash_only_entries=0,
                          peak_resident_units=0, read_seconds=0.0, wait_seconds=0.0,
                          first_batch_ready_seconds=None, finish_seconds=None,
                          source_weight_reads=0, source_weight_read_seconds=0.0)

    # -- readers -----------------------------------------------------------
    def entry_bytes(self, name) -> int:
        """One entry's FP32 X and H, as the loader's own storage bound states it."""
        return self._store._capture_storage_bytes(name, self._census, self._max_rows)

    def reader_reserve_bytes(self, name) -> int:
        """What one submitted read may still allocate: its entry, the serialized
        buffer and its source pages, and the loader's scratch."""
        weight = self._weights[name]
        source_bytes = weight.numel() * weight.element_size() if weight.is_meta else 0
        return (self.entry_bytes(name) + 2 * self._policy["max_buffer_bytes"]
                + self._policy["max_scratch_bytes"] + source_bytes)

    def _read(self, name):
        import torch

        from . import tessera_hessian as th
        started = self._clock()
        weight = self._weights[name]
        loaded = None
        if weight.is_meta:
            loaded = self._load_unit(name)
            if (not isinstance(loaded, torch.Tensor) or loaded.is_meta
                    or loaded.device.type != "cpu" or tuple(loaded.shape) != tuple(weight.shape)
                    or loaded.dtype != weight.dtype):
                raise RuntimeError(
                    f"{name}: the source weight read on the row stream is not the "
                    f"planned host tensor {tuple(weight.shape)} {weight.dtype}")
            weight = loaded
        load_seconds = self._clock() - started
        store = self._store
        artifact = store._capture_entry_artifact(self._path, self._manifest, name)
        payload, receipt = store._verified_capture_entry(artifact, name,
            expected_sha256=self._manifest["entries"][name].get("sha256"), census=self._census,
            max_rows=self._max_rows, policy=self._policy, execution=None,
            resource_check=None, release_file_pages=True, expected_stat=artifact.stat())
        inputs, hessian = payload["inputs"], payload["hessian"]
        count, max_abs = payload["count"], payload["max_abs"]
        payload.clear()
        source = th.activation_source({name: hessian}, self._hessian_identity)
        holder, identities = self._bind(name, weight=weight, inputs=inputs,
                                        hessian=hessian, source=source)
        return _Entry(name=name, inputs=inputs, hessian=hessian, source=source, holder=holder,
                      identities=identities, load=receipt, count=count, max_abs=max_abs,
                      read_seconds=self._clock() - started, weight=loaded,
                      weight_seconds=load_seconds)

    def _submit(self, name):
        if self._closed:
            raise RuntimeError("the row stream is closed")
        self._inflight[name] = self._pool.submit(self._read, name)

    def _collect(self, name):
        """Wait for one submitted read; record or compare its receipts."""
        if name in self._live:
            return self._live[name]
        future = self._inflight.pop(name)
        entry = future.result()
        self.stats["entries_read"] += 1
        self.stats["read_seconds"] += entry.read_seconds
        if entry.weight is not None:
            self.stats["source_weight_reads"] += 1
            self.stats["source_weight_read_seconds"] += entry.weight_seconds
        record = dict(load=entry.load, identities=entry.identities,
                      count=entry.count, max_abs=entry.max_abs)
        first = self._first.get(name)
        if first is None:
            if self._expected is not None:
                self._require_expected(entry)
            self._first[name] = record
        else:
            self.stats["rereads"] += 1
            if first != record:
                self._close_entry(entry)
                raise RuntimeError(
                    f"{name}: a second read of the capture entry disagrees with the first "
                    "read's receipts, so the unit's inputs changed during this run")
        if entry.weight is not None:
            # The consumer thread is the only writer, and it installs a unit
            # before any later read of it is submitted, so the next reader of
            # this unit sees the installed tensor and reads no source byte.
            self._weights[name] = entry.weight
            entry.weight = None
        self._live[name] = entry
        return entry

    def expect_identities(self, receipts):
        """Require each entry's first read to reproduce ``receipts[name]``.

        A resumed checkpoint's manifest records every unit's W, X and H
        receipts (``RECEIPT_FIELDS``). Armed before any read, so no entry
        reaches the consumer, and no shard is written for its unit, unless its
        receipts are the ones the checkpoint was priced on.
        """
        if self._first or self._inflight or self._live:
            raise RuntimeError("expected receipts must be armed before the first read")
        if set(receipts) != set(self._names):
            raise ValueError("expected receipts must cover exactly the stream's units")
        self._expected = {name: {field: receipts[name][field] for field in RECEIPT_FIELDS}
                          for name in self._names}

    def _require_expected(self, entry):
        """Refuse, by unit and field, a first read the checkpoint did not record."""
        from .digests import canonical_json
        expected = self._expected[entry.name]
        observed = entry.identities if isinstance(entry.identities, dict) else {}
        if set(observed) != set(RECEIPT_FIELDS):
            differing = sorted(set(observed) ^ set(RECEIPT_FIELDS))
        else:
            where = f"{entry.name} receipts"
            differing = [field for field in RECEIPT_FIELDS
                         if canonical_json(observed[field], where=where)
                         != canonical_json(expected[field], where=where)]
        if differing:
            self._close_entry(entry)
            raise RuntimeError(
                f"{entry.name}: this run's read of its capture entry gives a "
                f"{', '.join(differing)} receipt the resumed checkpoint's manifest did not "
                "record, so the checkpoint was priced on other inputs; refusing to adopt "
                "or encode the unit")

    # -- consumer ----------------------------------------------------------
    def plan(self, batches):
        """The encode order: one list of unit names per batch."""
        self._batches = [list(dict.fromkeys(batch)) for batch in batches]

    def admit(self, index):
        """Make batch ``index`` resident and start reading batch ``index + 1``."""
        current = self._batches[index]
        ahead = self._batches[index + 1] if index + 1 < len(self._batches) else []
        keep = set(current) | set(ahead)
        for name in [name for name in self._live if name not in keep]:
            self._release(name)
        wanted = [name for name in dict.fromkeys([*current, *ahead])
                  if name not in self._live and name not in self._inflight]
        if self._resource_check is not None:
            self._resource_check(f"before_row_stream_admit:{index}", reserve_bytes=sum(
                self.reader_reserve_bytes(name) for name in wanted))
        for name in wanted:
            self._submit(name)
        started = self._clock()
        for name in current:
            self._collect(name)
        self.stats["wait_seconds"] += self._clock() - started
        if self.stats["first_batch_ready_seconds"] is None:
            self.stats["first_batch_ready_seconds"] = self._clock() - self._opened
        resident = len(self._live) + sum(1 for future in self._inflight.values() if future.done())
        self.stats["peak_resident_units"] = max(self.stats["peak_resident_units"], resident)
        if self._resource_check is not None:
            self._resource_check(f"after_row_stream_admit:{index}")

    def entry(self, name) -> _Entry:
        if name not in self._live:
            raise RuntimeError(f"{name} is not resident in the row stream window")
        return self._live[name]

    def encoder_kwargs(self, name, scale_plane):
        """The unit's encoder keywords, memoized per (unit, plane) within the window."""
        from . import tessera_hessian as th
        key = (name, scale_plane)
        if key in self._memo:
            self._memo.move_to_end(key)
            return self._memo[key]
        while len(self._memo) >= self._memo_capacity:
            self._memo.popitem(last=False)
        entry = self.entry(name)
        if self._resource_check is not None:
            self._resource_check("before_selected_encoder_factors:" + name,
                                 reserve_bytes=self._factor_scratch_bytes)
        kwargs = th.encoder_kwargs(entry.source, name, int(self._weights[name].shape[1]),
                                   self._device, scale_plane=scale_plane)
        if self._resource_check is not None:
            self._resource_check("after_selected_encoder_factors:" + name)
        self._memo[key] = kwargs
        return kwargs

    def _close_entry(self, entry):
        if entry.holder is not None:
            entry.holder.close()
        entry.inputs = entry.hessian = entry.source = entry.holder = None

    def _release(self, name):
        """Drop a unit's X, H, source, holder and encoder factors."""
        entry = self._live.pop(name)
        for key in [key for key in self._memo if key[0] == name]:
            del self._memo[key]
        self._close_entry(entry)

    def finish(self):
        """Release the window, then read and receipt every unit never admitted.

        A deadline stop, a failed batch or an empty menu leaves units the loop
        never needed; the run identity binds them all the same.
        """
        started = self._clock()
        for name in list(self._live):
            self._release(name)
        for name in sorted(self._inflight):
            entry = self._collect(name)
            self._release(entry.name)
        missing = [name for name in self._names if name not in self._first]
        position = 0
        while position < len(missing):
            window = missing[position:position + self.threads]
            if self._resource_check is not None:
                self._resource_check(f"before_row_stream_receipts:{window[0]}",
                    reserve_bytes=sum(self.reader_reserve_bytes(name) for name in window))
            for name in window:
                self._submit(name)
            for name in window:
                entry = self._collect(name)
                self._release(entry.name)
                self.stats["hash_only_entries"] += 1
            position += len(window)
        self.stats["finish_seconds"] = self._clock() - started
        self.close()
        if set(self._first) < set(self._names):
            raise RuntimeError("the row stream finished without a receipt for every unit")

    def close(self):
        """Stop the readers and drop every resident entry; safe on any exit."""
        if self._closed:
            return
        self._closed = True
        for future in self._inflight.values():
            future.cancel()
        self._pool.shutdown(wait=True, cancel_futures=True)
        for future in self._inflight.values():
            if future.done() and not future.cancelled() and future.exception() is None:
                self._close_entry(future.result())
        self._inflight.clear()
        for name in list(self._live):
            self._release(name)
        self._memo.clear()

    # -- finalize ----------------------------------------------------------
    def _require_complete(self):
        if set(self._first) != set(self._names):
            raise RuntimeError("row stream receipts are incomplete")

    def load_execution(self):
        """The load execution record the serial prefetch folds, in its name order."""
        self._require_complete()
        execution = self._store._load_execution(self._policy, self._identity)
        for name in self._names:
            self._store.fold_load_receipt(execution, self._first[name]["load"])
        return execution

    def unit_identity(self, name) -> dict:
        """One unit's receipts from its first read, before the stream is complete."""
        if name not in self._first:
            raise RuntimeError(f"{name} has no receipt: the row stream never read it")
        return dict(self._first[name]["identities"])

    def unit_identities(self) -> dict:
        self._require_complete()
        return {name: dict(self._first[name]["identities"]) for name in self._names}

    def observed_counts(self) -> dict:
        self._require_complete()
        return {name: self._first[name]["count"] for name in self._names}

    def observed_max_abs(self) -> dict:
        self._require_complete()
        return {name: self._first[name]["max_abs"] for name in self._names}

    def execution_record(self, *, dependency=None):
        return dict(schema=EXECUTION_SCHEMA, row_head=ROW_HEAD_STREAM, dependency=dependency,
                    memory_plan=MEMORY_PLANS[ROW_HEAD_STREAM],
                    reader_threads=self.threads, window_units=2 * self.batch_size,
                    batches=len(self._batches), units=len(self._names),
                    identity_bound_writes=list(IDENTITY_BOUND_WRITES),
                    **{key: (round(value, 3) if isinstance(value, float) else value)
                       for key, value in self.stats.items()})
