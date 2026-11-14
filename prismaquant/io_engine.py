"""One engine moves bytes from a tier to a consumer (PQ #1294, #1291).

The repo grew one hand-rolled loader per stage, each with its own worker
count, depth constant and verification, and most of them ran load and compute
in turn. ``tests/test_io_site_freeze.py`` freezes those sites; new byte
movement goes through this module. This first version carries one caller,
Stage B's retained-window replay, and it has two layers.

**One file, read once.** :func:`load_file` is the read every ProductionWeight
Cache shard takes (moved here from ``production_weight_cache``): the tier
comes from the immutable staged manifest (``residency_map.residency_resolver``
and ``staged_lease``). The staged copy is read under its descriptor window,
the bytes are hashed once and held to the extent's digest, and the
caller's decoder turns them into a value. A staged copy that fails any fence
falls back to the declared file unless the allowed-tier policy is active, in
which case the read refuses. :func:`pin_group` pins a set of staged files under
one reader lease, which is how a window's renders were already pinned
(PQ #1210).

**An ordered stream, read ahead.** :func:`read_stream` takes an ordered stream
of :class:`ReadEntry` and a :class:`budget <ReadBudget>`, and reads entries
ahead of the consumer on this module's one thread pool. The caller never
states a depth or a worker count:

* **Depth** is the budget's live headroom. The engine admits the next entry
  only while its serialized buffer plus its decoded bytes fit the headroom the
  budget reads now, less what is already in flight, so how far it runs ahead is
  whatever the row's memory allows at that moment. Stage B's budget reads its
  capture guard (``CaptureMemoryGuard.headroom_bytes``): cgroup committed bytes
  plus the CUDA reservation, against the row's caps and host floor, less the
  reservation of the phase the consumer is in. Serialized buffers in flight are
  also held to ``budget.buffer_bytes``, the load buffer the plan sealed.
* **Owned bytes.** A stream's reads land in memfds the engine owns
  (:class:`SealedBuffer`), hashed on the pass that reads them and sealed
  against writes before a decoder sees them. A decoder maps the memfd rather
  than copying it, so the decoded value holds exactly the file's pages, and
  dropping the value returns them to the cgroup. A copy into a ``torch``
  CPU tensor would not return them on this platform (see
  :class:`SealedBuffer`), and the depth and the reclaim below both depend on
  freed bytes leaving the reading.
* **Release, not take.** A taken group's bytes stay charged to the budget
  until the consumer calls ``ReadStream.release`` (or takes the next group):
  the consumer holds the values until then, and on unified memory a stream
  that counted them free at the take would read past the row's memory.
* **Reclaim.** An entry read ahead is reclaimable until the consumer takes it.
  ``ReadStream.reclaim`` drops the farthest-ahead ones first, and a budget
  that can refuse (the capture guard) calls it before it would refuse, so a
  read-ahead never fails a check the row would otherwise pass. A dropped entry
  is read again when the budget has room or the consumer asks for it.
* **Workers** follow the measured rates. Each read's own seconds give the
  per-stream rate; the consumer's seconds per group give the time the next
  group has. The engine runs as many reads at once as it needs to land the
  next group within the consumer's shortest measured group, and its whole
  width while the consumer waits or before it has measured anything. The pool
  is sized by the CPU affinity. A stream holds one core back for its
  consumer only while the consumer's measured work (take to take) exceeds its
  measured wait in ``take`` after the first, and before it has measured a work interval
  (``ReadStream._width``, PQ #1533).
* **Order and verification.** A group is delivered only when every one of its
  entries is read, hashed, held to its digest and decoded; entries come back
  in stream order. The first failure in a taken group raises
  :class:`EntryError` at the consumer, naming the entry. A failure while the
  group was only read ahead is normally not final: the group is read again
  when the consumer asks for it, and that read decides. A reader or staging
  gate may instead raise :class:`TerminalReadError` for an irrecoverable
  failure: the failed entry is retained and is never retried, even if the
  consumer had not demanded it yet.
* **Staging.** A group's entries are read only after its ``ready`` callable
  says they are staged, and the group's files are pinned under one lease for
  the reads and released after them. A group that is not ready yet waits until
  the consumer asks for it.

**Range entries** (PQ #1348). An entry with a ``reader`` is not a file: the
stream calls ``reader()`` on a pool thread and delivers what it returns.
Stage B's spill replay reads its chunks this way, byte ranges of its own
job-local scratch file into a buffer the reader allocates. A range entry has
no path, so it is neither staged nor pinned, and no serialized buffer, so it
is charged its held bytes only. It carries no digest, and the stream checks
none: whatever holds its bytes to what was written is the reader's. The
spill verifies each operand's xxh3_64 and raises :class:`TerminalReadError`
on a mismatch, in addition to checking the read's length and grid. Depth, workers,
order, reclaim and the consumer's waits are the stream's, as for a file.

Every consumer wait is timed; ``ReadStream.counters`` is the record the caller
writes to its counters (Stage B: ``counters.json``'s ``io_engine`` block).
"""
from __future__ import annotations

from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import math
import os
from pathlib import Path
import stat
import threading
import time
from typing import Any, Callable, Hashable, Iterable, Protocol

from .io_buffers import (  # noqa: F401 -- existing engine imports share the leaf implementation
    SealedBuffer, kernel_seal_bits, _F_ADD_SEALS, _F_GET_SEALS, _SEALS,
)
from .io_buffers import copy_extent
from .digests import bytes_sha256hex
from .file_identity import file_stat_signature as _stat_signature
from .residency_map import StagedReadRefused


class TerminalReadError(RuntimeError):
    """An irrecoverable entry failure, retained without retry before demand.

    The engine preserves the diagnostic but drops the original traceback:
    a failed reader's scratch buffer must not stay held outside its budget.
    """


class EntryError(RuntimeError):
    """One stream entry failed; ``key`` names it."""

    def __init__(self, key, cause: BaseException):
        super().__init__(f"io entry {key!r}: {type(cause).__name__}: {cause}")
        self.key = key


# --------------------------------------------------------------------------
# One file, read once
# --------------------------------------------------------------------------

def _close_sealed(raw) -> None:
    if isinstance(raw, SealedBuffer):
        raw.close()


def read_file(path: Path, limit: int, *, declared_signature=None, staged=None,
              lease=None, timing: dict | None = None, sealed: bool = False):
    """Read one declared file's bytes once: ``(raw, receipt, signature)``.

    ``raw`` is ``bytes``, or with ``sealed`` a :class:`SealedBuffer` the
    caller closes.

    ``path`` is always the declared file. A staged read opens the resolver's
    immutable object, and every fence that belongs to the declared object
    still runs on the declared object: the caller's stat signature
    (``declared_signature``) and the receipt's path. The staged object gets
    its own open/read fences plus the digest the manifest published, and a
    failure of any of those raises; :func:`load_file` then refuses.

    Native staged entries use an owned descriptor window: ``lease`` from
    :func:`pin_group`, as ``(lease window, key, staged entry)``, when supplied,
    otherwise one window per read. The reader authenticates object bounds,
    descriptor identity and bytes before returning. The resolver records
    serving identity at the actual open. External cache pins or retained
    phase ownership protect the object's lifetime.

    ``timing``, when given, receives ``read_s``: the seconds spent opening and
    reading the bytes, without the hash, so a caller can measure its
    per-stream rate.
    """
    if staged is not None:
        from .staged_whole_file import read_staged_entry
        if declared_signature is not None:
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode) or _stat_signature(before) != declared_signature:
                raise StagedReadRefused('declared file changed before staged content read')
        started = time.monotonic()
        owner = staged['_resolver']
        raw = read_staged_entry(owner, path, staged, label='io-engine', sealed=sealed,
                                max_bytes=limit, lease=lease)
        try:
            if declared_signature is not None:
                after = path.lstat()
                if not stat.S_ISREG(after.st_mode) or _stat_signature(after) != declared_signature:
                    raise StagedReadRefused('declared file changed during staged content read')
        except BaseException:
            _close_sealed(raw)
            raise
        if timing is not None:
            timing['read_s'] = time.monotonic() - started
        receipt = {'path': str(path), 'bytes': len(raw), 'sha256': staged['sha256'],
                   'serving_tier': 'stage'}
        return raw, receipt, declared_signature
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError("PWC file receipt requires a regular file, not a symlink")
    if before.st_size > limit:
        raise RuntimeError("PWC file exceeds the explicit read buffer bound")
    signature = _stat_signature(before)
    if declared_signature is not None and signature != declared_signature:
        raise RuntimeError('PWC window file changed before its content read')
    started = time.monotonic()
    buffer = None
    try:
        buffer = SealedBuffer(before.st_size) if sealed else None
        with path.open("rb") as handle:
            if _stat_signature(os.fstat(handle.fileno())) != signature:
                raise RuntimeError("PWC file changed before its content read")
            if buffer is None:
                raw = handle.read(before.st_size + 1)
                exact = len(raw) == before.st_size
            else:
                exact = buffer.fill(handle.fileno())
            if (not exact
                    or _stat_signature(os.fstat(handle.fileno())) != signature
                    or _stat_signature(path.lstat()) != signature):
                raise RuntimeError("PWC file changed during its content read")
        if timing is not None:
            timing["read_s"] = time.monotonic() - started
        # The serialized buffer is private to this read and released after its
        # decoder returns. No whole-cache byte store. A sealed buffer is hashed
        # on the pass that sealed it: the bytes a decoder maps are the bytes
        # this digest names, and nothing can write them after.
        if buffer is None:
            digest = bytes_sha256hex(raw)
        else:
            digest, raw = buffer.seal(), buffer
        receipt = {"path": str(path), "bytes": len(raw), "sha256": digest}
    except BaseException:
        if buffer is not None:
            buffer.close()
        raise
    return raw, receipt, signature


def load_file(path: Path, limit: int, *, binding, decode, declared_signature=None,
              lease=None, timing: dict | None = None, sealed: bool = False):
    """Read one declared file from the tier that serves it, then decode it.

    Returns ``(value, (receipt, signature, detail))`` where ``decode(raw,
    receipt, staged)`` returned ``(value, detail)``. The decoder runs on the
    bytes :func:`read_file` just hashed; for a staged copy it raises
    ``StagedReadRefused`` on a fence of its own, which refuses like a read
    fence does. With ``sealed`` the decoder gets a :class:`SealedBuffer`,
    closed here once it returns or raises: a value that maps it keeps its
    pages, and nothing else does.

    Without ``PRISMAQUANT_STAGED_MANIFEST`` no resolver exists and the bytes
    are the pool's. A resolver exists only under the active allowed-tier
    policy, and that policy refuses every read the manifest does not serve.

    Under the active allowed-tier policy the read requires ``binding``, the
    caller's expected digest: without it a staged copy would be admitted on
    the map's word alone.
    """
    from .residency_map import residency_resolver
    from .staged_tier_policy import policy_is_active, refuse_pool_bulk_read
    resolver = residency_resolver()
    strict = policy_is_active()
    if strict and binding is None:
        # Without the digest the caller already requires, a staged
        # copy would be admitted on the map's word alone. The run
        # leg binds every render through ``require_file_load_sha256``;
        # the digest-less prepare leg runs outside campaign scope.
        raise refuse_pool_bulk_read(str(path), "missing-digest-binding")
    staged = None
    if strict and lease is not None:
        # Pinned by a group lease with the entry the pin was built from: a
        # map recomposed since then cannot unserve bytes the pin holds, and
        # asking again would refuse a perfectly pinned read.
        staged = lease[2]
    elif resolver is not None:
        staged = resolver.staged_read(path, expected_sha256=binding)
    if staged is not None:
        try:
            raw, receipt, signature = read_file(
                path, limit, declared_signature=declared_signature, staged=staged,
                lease=lease, timing=timing, sealed=sealed)
            try:
                value, detail = decode(raw, receipt, True)
            finally:
                _close_sealed(raw)
        except StagedReadRefused as refusal:
            resolver.record_fallback(path, str(refusal))
            raise refuse_pool_bulk_read(str(path), str(refusal))
        return value, (receipt, signature, detail)
    if strict:
        raise refuse_pool_bulk_read(
            str(path), "readset-not-staged" if resolver is None
            else "staged-not-serving")
    raw, receipt, signature = read_file(
        path, limit, declared_signature=declared_signature, staged=None, timing=timing,
        sealed=sealed)
    try:
        value, detail = decode(raw, receipt, False)
    finally:
        _close_sealed(raw)
    return value, (receipt, signature, detail)


class _LeaseRef:
    """The declared file one group-lease member stands for."""
    __slots__ = ("path",)

    def __init__(self, path: str):
        self.path = path


def pin_group(members, live_leases, *, counters=None):
    """Pin a set of staged files under one reader lease (PQ #1210).

    Group files in one native descriptor window through the same
    ``staged_lease.enter_group_lease`` used by the exact activation cache.
    Cache eviction pins belong to the external fluxcache owner; shared-phase
    retention belongs to the declared phase lifetime.

    ``members`` is ``[(declared path, expected sha256), ...]``. Returns
    ``{declared path: (lease window, map key, staged entry)}`` for the reads
    to open under, or ``None`` when no group applies: the policy is inactive,
    no resolver is bound, fewer than two files are staged with a bound digest,
    or the batched lease refused. Every individual read still opens through
    the immutable resolver, checks descriptor identity and hashes the bytes
    against the declared digest. A group does not supply read authority by
    itself.
    ``live_leases`` receives every lease entered; :func:`unpin` exits them.
    """
    from .staged_tier_policy import policy_is_active
    if not policy_is_active():
        return None
    from .residency_map import residency_resolver
    resolver = residency_resolver()
    if resolver is None:
        return None
    pinned, seen = [], set()
    for path, binding in members:
        if binding is None or path in seen:
            continue
        staged = resolver.staged_read(Path(path), expected_sha256=binding)
        if staged is None:
            continue
        seen.add(path)
        pinned.append((_LeaseRef(path), staged))
    if len(pinned) < 2:
        return None
    from .staged_lease import enter_group_lease
    assignments = enter_group_lease(resolver, pinned, live_leases, counters=counters)
    if assignments is None:
        return None
    staged_by_path = {ref.path: staged for ref, staged in pinned}
    return {ref.path: (lease_window, lease_key, staged_by_path[ref.path])
            for ref, (lease_window, lease_key) in assignments.items()}


def unpin(live_leases) -> None:
    """Close every pinned lease's descriptors and release its one ref.

    Every lease is exited even when one fails; the first failure is raised
    last.
    """
    failure = None
    while live_leases:
        try:
            live_leases.pop().__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001 -- raised below
            if failure is None:
                failure = exc
    if failure is not None:
        raise failure


# --------------------------------------------------------------------------
# An ordered stream, read ahead
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ReadEntry:
    """One file a stream delivers.

    ``key`` names the entry in errors and in delivery. ``path`` is the
    declared file, ``size`` its length (the serialized buffer the read holds)
    and ``limit`` the read bound. ``held_bytes`` bounds what the decoded value
    holds until the consumer takes it. ``expected_sha256`` is the digest the
    read is held to (``None`` only where the tier policy allows it).
    ``decoder(raw, receipt, staged) -> (value, detail)`` turns the verified
    bytes into the value, and ``derive(value, observed) -> dict`` (optional)
    computes what is derived from this one load on the reading thread.
    ``group`` is the delivery unit: a consumer takes a group whole.
    ``declared_stat`` is the stat the caller priced the file on; the read
    refuses a file that changed since. ``measure(value) -> int`` (optional)
    is the bytes the decoded value actually holds, at most ``held_bytes``;
    without it the charge is taken as held. ``tier_hint`` names the tier the
    caller expects (reporting only; the residency map decides).

    The decoder receives a :class:`SealedBuffer`, which the engine closes
    when the decoder returns. A decoder that maps it (``path``) holds the
    file's pages in its value; one that copies it holds its copy. The read in
    flight is charged ``size + held_bytes`` either way, which overstates a
    mapping decoder by ``size`` for the reads in flight only.

    ``reader() -> (value, observed)`` makes a range entry (module
    docstring): ``path``, ``decoder`` and ``expected_sha256`` are ``None``,
    ``size`` is the bytes it reads and ``held_bytes`` what its value holds.
    """

    key: Hashable
    path: str | None
    size: int
    limit: int
    held_bytes: int
    expected_sha256: str | None
    decoder: Callable[[bytes, dict, bool], tuple[Any, Any]] | None
    group: Hashable
    declared_stat: Any = None
    derive: Callable[[Any, tuple], dict] | None = None
    measure: Callable[[Any], int] | None = None
    tier_hint: str | None = None
    reader: Callable[[], tuple[Any, Any]] | None = None

    @property
    def raw_bytes(self) -> int:
        """The serialized buffer this read holds in flight: none for a range."""
        return 0 if self.reader is not None else self.size


@dataclass
class Delivered:
    """A verified, decoded entry, handed to the consumer in stream order."""

    entry: ReadEntry
    value: Any
    observed: tuple
    derived: dict | None


class ReadBudget(Protocol):
    """What a stream may hold, read live.

    ``headroom_bytes(held_bytes)`` is how many more bytes the stream may read
    ahead now, given the ``held_bytes`` its entries hold: read ahead and not
    taken, plus taken and not yet released. A budget that reads the process
    sees those bytes in its reading already. ``buffer_bytes`` bounds the
    serialized buffers in flight at once.
    """

    buffer_bytes: int

    def headroom_bytes(self, held_bytes: int) -> int: ...


_PENDING, _READING, _DONE, _FAILED, _DELIVERED = range(5)


class ReadStream:
    """One ordered stream's read-ahead state. Build it with :func:`read_stream`."""

    def __init__(self, engine: "IOEngine", entries, budget, ready, lease_counters=None):
        entries = tuple(entries)
        if not entries or not all(isinstance(entry, ReadEntry) for entry in entries):
            raise TypeError("an io stream needs a nonempty sequence of ReadEntry")
        if type(budget.buffer_bytes) is not int or budget.buffer_bytes <= 0:
            raise ValueError("an io stream budget needs positive serialized buffer bytes")
        if len({entry.key for entry in entries}) != len(entries):
            raise ValueError("io stream entry keys must be unique")
        groups: list = []
        members: dict = {}
        for index, entry in enumerate(entries):
            if entry.reader is not None:
                if not callable(entry.reader) or any(
                        value is not None for value in (
                            entry.path, entry.decoder, entry.expected_sha256)):
                    raise ValueError(
                        f"io range entry {entry.key!r} names a path, a decoder or a "
                        "digest; its reader reads and verifies its own bytes")
            elif entry.path is None or entry.decoder is None:
                raise ValueError(f"io entry {entry.key!r} needs a path and a decoder")
            if entry.group not in members:
                groups.append(entry.group)
                members[entry.group] = []
            elif groups[-1] != entry.group:
                raise ValueError(f"io stream group {entry.group!r} is not contiguous")
            if entry.raw_bytes > budget.buffer_bytes:
                raise ValueError(
                    f"io entry {entry.key!r}: its serialized buffer exceeds the budget's")
            members[entry.group].append(index)
        self._engine = engine
        self._lease_counters = lease_counters
        self._entries = entries
        self._budget = budget
        self._ready = ready
        self._groups = groups
        self._members = members
        self._state = [_PENDING] * len(entries)
        self._values: list = [None] * len(entries)
        self._errors: list = [None] * len(entries)
        self._reads = [0] * len(entries)
        # Group lifecycle: closed -> gating -> open -> drained (pins released),
        # or deferred when ``ready`` says the group is not staged yet.
        self._group_state = {group: "closed" for group in groups}
        self._leases: dict = {group: None for group in groups}
        self._live: dict = {group: [] for group in groups}
        self._outstanding = {group: 0 for group in groups}
        # Pins a reading thread is releasing outside the lock; a take and a
        # close wait for them, so no pin outlives the reads it protects.
        self._releasing = {group: 0 for group in groups}
        self._demanded: set = set()
        self._taken = 0
        self._gating = False
        self._active = 0
        self._inflight_raw = 0
        self._inflight_charge = 0
        self._held = 0
        self._held_actual = 0
        # Bytes of the group taken last, until the consumer releases it: its
        # values are the consumer's, and they hold memory until then.
        self._unreleased = 0
        self._actual = [0] * len(entries)
        self._paused = 0
        self._cond = threading.Condition()
        self._cancel = threading.Event()
        self._closed = False
        self._consumer_waiting = False
        self._took_at = None
        self._took_bytes = 0
        self._busy_min_s = None
        # When the last take returned; the next take's start closes the
        # consumer's work interval (``consumer_work_s``, PQ #1533).
        self._returned_at = None
        self._worked = False
        # Exposed-wait sink (PQ #1292): ``wait_sink(kind, start_unix, end_unix,
        # info)``, called once per take with the interval the consumer sat
        # blocked and the rates the wait bound derives from. None costs nothing.
        self.wait_sink = None
        # Monotonic start of the current interval with a read in flight;
        # ``read_wall_s`` sums those intervals (union, not thread-seconds).
        self._read_wall_from = None
        # The lowest index still pending: everything before it is being
        # read, read, or delivered. A reclaimed entry moves it back.
        self._cursor = 0
        self.counters = {
            "schema": "prismaquant.io_engine_stream.v1",
            "entries": len(entries), "groups": len(groups),
            "pool_width": engine.width,
            "entries_read": 0, "bytes_read": 0, "read_s": 0.0,
            "read_wall_s": 0.0,
            "rereads": 0, "evictions": 0, "evicted_bytes": 0,
            "ahead_deferrals": 0, "ahead_failures": 0,
            "peak_held_bytes": 0, "peak_workers": 0, "peak_workers_consumer_busy": 0,
            "consumer_wait_s": 0.0, "consumed_bytes": 0, "consumer_busy_s": 0.0,
            "consumer_work_s": 0.0, "consumer_steady_wait_s": 0.0,
            "groups_taken": [],
        }
        # A budget that derives a growth reserve from this stream's measured
        # consumer growth binds to it here (PQ #2602); the others are not asked.
        bind = getattr(budget, "bind_growth_source", None)
        if bind is not None:
            bind(self)

    # -- consumer side ----------------------------------------------------

    def __enter__(self) -> "ReadStream":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def group_keys(self, group) -> tuple:
        """The keys of ``group``'s entries, in stream order."""
        return tuple(self._entries[i].key for i in self._members.get(group, ()))

    def next_group(self):
        """The group the consumer takes next, or ``None`` once all are taken."""
        with self._cond:
            return self._groups[self._taken] if self._taken < len(self._groups) else None

    def unread_bytes(self, group) -> int:
        """Bytes ``group``'s unread entries will still hold; marks nothing.

        An entry not read yet will hold its ``held_bytes``; one in flight is
        charged its serialized buffer too, which a reading of the process
        may not see yet. Another stream's budget subtracts this for the group
        this stream's consumer needs next, so a stream read in parallel
        leaves it room (Stage B's spill replay leaves the next window's
        renders theirs, PQ #1348).
        """
        with self._cond:
            total = 0
            for i in self._members.get(group, ()):
                entry = self._entries[i]
                if self._state[i] in (_PENDING, _FAILED):
                    total += entry.held_bytes
                elif self._state[i] == _READING:
                    total += entry.raw_bytes + entry.held_bytes
            return total

    def window_growth_bytes(self) -> int:
        """The charged bytes the consumer's next window is expected to grow by (PQ #2602).

        Derivation (``GuardReadBudget`` subtracts this from the live reading;
        docs/ARCHITECTURE.md states the same):

        * An eviction happens only when the guard's headroom fell below the
          read-ahead the stream held, that is, when the consumer grew past
          what the read-ahead was admitted against. ``evicted_bytes`` is the
          charged size of what that growth forced out.
        * A window is a group taken, so the growth per window is the
          evicted bytes over ``max(1, groups taken)``, rounded up (a mean:
          it rises as evictions rise and falls as clean windows pass).
        * It never exceeds ``peak_held_bytes``: one window's reclaim cannot
          free more than the deepest read-ahead the stream ever held, so a
          reread-inflated total cannot reserve the stream down to nothing.
        * Zero with no eviction observed (nothing measured), and zero once
          no group is left to take or the stream is closed (no next window
          to grow in). It reads counters under the lock and marks nothing.

        Admission against ``live - growth`` leaves the guard that much room
        for the next window, so growth up to the measured mean does not
        push the guard into reclaiming read-ahead already read. The bytes of
        the group last taken stay charged through ``held_bytes`` plus
        ``_unreleased`` until :meth:`release`, not through this term.
        """
        with self._cond:
            if self._closed or self._taken >= len(self._groups):
                return 0
            windows = max(1, self._taken)
            per_window = -(-self.counters["evicted_bytes"] // windows)
            return min(per_window, self.counters["peak_held_bytes"])

    def demand(self, group) -> tuple[int, int]:
        """Mark ``group`` as the consumer's next; return what is still unread.

        A demanded group is never reclaimed and is read whatever the budget's
        headroom says: the consumer charges its remaining bytes to its own
        admission check first. Returns ``(held bytes, serialized bytes)`` of
        the group's entries not yet read, which is what that check charges.
        Reading starts at :meth:`take`.
        """
        with self._cond:
            if self._taken >= len(self._groups) or self._groups[self._taken] != group:
                raise RuntimeError(f"io stream group {group!r} is not the next in order")
            self._demanded.add(group)
            unread = [self._entries[i] for i in self._members[group]
                      if self._state[i] not in (_DONE, _DELIVERED)]
            return (sum(entry.held_bytes for entry in unread),
                    sum(entry.size for entry in unread))

    def paused(self):
        """A context in which nothing is read and no read is in flight.

        Yields the bytes the stream holds read ahead (measured where the
        entries say how), so a caller can take a reading of the process
        without the stream in it. A group gate may still be waiting for its
        staging: it allocates nothing, and the reads it would start wait for
        the pause to end.
        """
        from contextlib import contextmanager

        @contextmanager
        def pause():
            with self._cond:
                self._paused += 1
                while self._active:
                    self._cond.wait()
                held = self._held_actual
            try:
                yield held
            finally:
                with self._cond:
                    self._paused -= 1
                    self._pump()
        return pause()

    def take(self, group) -> list[Delivered]:
        """Block until ``group`` is read and verified, then hand it over.

        ``group`` must be the next group in stream order. The delivered values
        belong to the consumer from here on; the stream forgets them. The first
        failed entry, in order, raises :class:`EntryError`.
        """
        with self._cond:
            if self._closed:
                raise RuntimeError("io stream is closed")
            if self._taken >= len(self._groups) or self._groups[self._taken] != group:
                raise RuntimeError(f"io stream group {group!r} is not the next in order")
            now = time.monotonic()
            work_before = None
            if self._returned_at is not None:
                # Take to take, whether or not the consumer released early:
                # the time it spent outside the engine (PQ #1533).
                work_before = now - self._returned_at
                self.counters["consumer_work_s"] += work_before
                self._returned_at = None
                self._worked = True
            if self._took_at is not None:
                self._release_locked(now)
            self._demanded.add(group)
            if self._group_state[group] == "deferred":
                self._group_state[group] = "closed"
            started = now
            started_unix = time.time()
            first_fill = not self._worked
            self._consumer_waiting = True
            try:
                self._pump()
                indices = self._members[group]
                while (not all(self._state[i] in (_DONE, _FAILED) for i in indices)
                       or self._releasing[group]):
                    self._cond.wait()
                    if self._closed:
                        raise RuntimeError("io stream closed while its consumer waited")
            finally:
                self._consumer_waiting = False
            waited = time.monotonic() - started
            ended_unix = time.time()
            nbytes = sum(self._entries[i].size for i in indices)
            wall = self.counters["read_wall_s"]
            if self._read_wall_from is not None:
                # A read still in flight: its interval counts toward the rate
                # (bytes_read holds only finished reads, so this errs low).
                wall += time.monotonic() - self._read_wall_from
            load_rate = self.counters["bytes_read"] / wall if wall > 0 else None
            self.counters["consumer_wait_s"] += waited
            if self._worked:
                # The first take waits for the stream to fill whatever the
                # width; only waits after a work interval weigh the
                # consumer's core (``_width``, PQ #1533).
                self.counters["consumer_steady_wait_s"] += waited
            self.counters["groups_taken"].append(
                {"group": _jsonable(group), "wait_s": waited, "bytes": nbytes,
                 "entries": len(indices),
                 "reread_entries": sum(1 for i in indices if self._reads[i] > 1),
                 "at_unix": started_unix, "first_fill": first_fill,
                 "work_before_s": work_before, "load_bytes_per_s": load_rate})
            sink = self.wait_sink
            if sink is not None:
                sink("window-load", started_unix, ended_unix,
                     {"bytes": nbytes, "first_fill": first_fill,
                      "work_before_s": work_before,
                      "load_bytes_per_s": load_rate})
            for i in indices:
                if self._state[i] == _FAILED:
                    raise EntryError(self._entries[i].key, self._errors[i])
            delivered = []
            for i in indices:
                value, observed, derived = self._values[i]
                self._values[i] = None
                self._state[i] = _DELIVERED
                self._held -= self._entries[i].held_bytes
                self._held_actual -= self._actual[i]
                delivered.append(Delivered(self._entries[i], value, observed, derived))
            self._taken += 1
            self._took_at = self._returned_at = time.monotonic()
            self._took_bytes = nbytes
            self._unreleased = sum(self._entries[i].held_bytes for i in indices)
            self._pump()
            # Reads in flight as the consumer goes back to work, including
            # any started while it waited (PQ #1533).
            self.counters["peak_workers_consumer_busy"] = max(
                self.counters["peak_workers_consumer_busy"], self._active)
            return delivered

    def release(self) -> None:
        """The consumer is done with the group it took last, and has dropped it.

        The budget charges a taken group until here, so the consumer drops
        its values first; taking the next group releases the last one too.
        """
        with self._cond:
            if self._took_at is not None:
                self._release_locked(time.monotonic())
            self._pump()

    def _release_locked(self, now) -> None:
        busy = now - self._took_at
        self._took_at = None
        self._unreleased = 0
        self.counters["consumed_bytes"] += self._took_bytes
        self.counters["consumer_busy_s"] += busy
        self._busy_min_s = busy if self._busy_min_s is None else min(self._busy_min_s, busy)

    def held_bytes(self) -> int:
        """Decoded bytes read ahead and not yet taken (reclaimable)."""
        with self._cond:
            return self._held

    def held_actual_bytes(self) -> int:
        """Bytes the read-ahead values actually hold (``ReadEntry.measure``)."""
        with self._cond:
            return self._held_actual

    def reclaim(self, shortfall_bytes: int) -> int:
        """Drop read-ahead entries, farthest first, until ``shortfall_bytes`` is freed.

        Only entries of groups the consumer has not asked for are dropped;
        they are read again later. Returns the bytes freed (charged bytes).
        """
        freed = 0
        dropped = []
        with self._cond:
            if self._closed:
                return 0
            for i in range(len(self._entries) - 1, -1, -1):
                if freed >= shortfall_bytes:
                    break
                entry = self._entries[i]
                if self._state[i] != _DONE or entry.group in self._demanded:
                    continue
                dropped.append(self._values[i])
                self._values[i] = None
                self._state[i] = _PENDING
                self._cursor = min(self._cursor, i)
                self._held -= entry.held_bytes
                self._held_actual -= self._actual[i]
                freed += entry.held_bytes
                self.counters["evictions"] += 1
                self.counters["evicted_bytes"] += entry.held_bytes
        dropped = None  # the values die here, outside the lock
        return freed

    def close(self) -> None:
        """Stop reading ahead, wait for reads in flight, release every pin."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._cancel.set()
            self._cond.notify_all()
            while self._active or self._gating or any(self._releasing.values()):
                self._cond.wait()
            self._values = [None] * len(self._entries)
            self._held = self._held_actual = self._unreleased = 0
            groups = list(self._groups)
        failure = None
        for group in groups:
            try:
                unpin(self._live[group])
            except Exception as exc:  # noqa: BLE001 -- raised below
                if failure is None:
                    failure = exc
        self.counters["per_stream_bytes_per_s"] = self._stream_rate()
        busy = self.counters["consumer_busy_s"]
        self.counters["consumption_bytes_per_s"] = (
            self.counters["consumed_bytes"] / busy if busy > 0 else None)
        if failure is not None:
            raise failure

    # -- scheduling (lock held) ------------------------------------------

    def _stream_rate(self):
        seconds = self.counters["read_s"]
        return self.counters["bytes_read"] / seconds if seconds > 0 else None

    def _width(self) -> int:
        """The reads this stream may run at once: the pool, less the consumer's core when it works.

        The consumer runs on a core of the same affinity the pool is sized
        by. The engine measures where the consumer's time goes: waiting in
        :meth:`take` after its first work interval
        (``consumer_steady_wait_s``; the first take waits for the stream to
        fill at any width) or working between one take's return and the next
        take (``consumer_work_s``). A core held back for
        the consumer idles for its wait share; a core given to the reads
        costs the consumer its work share in contention. So the core is held
        back while the measured work share is the larger, and given to the
        reads while the wait share is. The decision is the measured shares,
        not the consumer's state at this instant: a read started during a
        short wait runs on into the consumer's next work interval. Before
        the consumer has finished one work interval the core is held back,
        as it always was (PQ #1533).
        """
        pool = self._engine.width
        if not self._worked:
            return max(1, pool - 1)
        if self.counters["consumer_work_s"] <= self.counters["consumer_steady_wait_s"]:
            return pool
        return max(1, pool - 1)

    def _workers(self) -> int:
        """Reads to run at once, from the measured rates (module docstring)."""
        width = self._width()
        rate = self._stream_rate()
        if (self._consumer_waiting or rate is None or self._busy_min_s is None
                or self._took_at is None or self._cursor >= len(self._entries)):
            return width
        # The next group must land within the consumer's shortest measured
        # group, less the time it has already spent on the current one.
        remaining = self._busy_min_s - (time.monotonic() - self._took_at)
        if remaining <= 0:
            return width
        group = self._entries[self._cursor].group
        need = sum(self._entries[i].size for i in self._members[group]
                   if self._state[i] == _PENDING)
        return max(1, min(width, math.ceil(need / (rate * remaining))))

    def _fits(self, entry) -> bool:
        need = entry.raw_bytes + entry.held_bytes
        held = self._held + self._unreleased
        return need <= self._budget.headroom_bytes(held) - self._inflight_charge

    def _pump(self) -> None:
        if self._closed or self._paused:
            return
        workers = self._workers()
        while self._cursor < len(self._entries) and self._state[self._cursor] != _PENDING:
            self._cursor += 1
        for index in range(self._cursor, len(self._entries)):
            if self._state[index] != _PENDING:
                continue
            entry = self._entries[index]
            group = entry.group
            demanded = group in self._demanded
            gstate = self._group_state[group]
            if gstate == "deferred" and not demanded:
                return
            if gstate in ("closed", "drained", "deferred"):
                if self._gating or (not demanded and not self._fits(entry)):
                    return
                self._gating = True
                self._group_state[group] = "gating"
                self._engine.submit(self._gate, group)
                return
            if gstate == "gating":
                return
            if self._active >= workers:
                return
            raw = entry.raw_bytes
            if raw and self._inflight_raw and self._inflight_raw + raw > self._budget.buffer_bytes:
                return
            if not demanded and not self._fits(entry):
                return
            self._state[index] = _READING
            if self._active == 0:
                self._read_wall_from = time.monotonic()
            self._active += 1
            self.counters["peak_workers"] = max(self.counters["peak_workers"], self._active)
            if not self._consumer_waiting:
                # Reads started while the consumer works, which _width holds
                # to the pool less the consumer's core when its work share
                # leads (PQ #1533).
                self.counters["peak_workers_consumer_busy"] = max(
                    self.counters["peak_workers_consumer_busy"], self._active)
            self._inflight_raw += raw
            self._inflight_charge += raw + entry.held_bytes
            self._outstanding[group] += 1
            self._engine.submit(self._read, index)

    # -- worker side ------------------------------------------------------

    def _gate(self, group) -> None:
        error, ready, leases, live = None, True, None, []
        try:
            if self._ready is not None:
                ready = bool(self._ready(group, self._cancel))
            files = [(self._entries[i].path, self._entries[i].expected_sha256)
                     for i in self._members[group] if self._entries[i].reader is None]
            if ready and not self._cancel.is_set() and files:
                # A range entry names no file: there is nothing to pin.
                leases = pin_group(files, live, counters=self._lease_counters)
        except CancelledError:
            ready = False
        except BaseException as exc:  # noqa: BLE001 -- surfaced at the consumer
            error = exc.with_traceback(None) if isinstance(exc, TerminalReadError) else exc
        with self._cond:
            self._gating = False
            self._live[group].extend(live)
            if error is not None:
                if group in self._demanded or isinstance(error, TerminalReadError):
                    if group not in self._demanded:
                        self.counters["ahead_failures"] += 1
                    for i in self._members[group]:
                        if self._state[i] == _PENDING:
                            self._state[i] = _FAILED
                            self._errors[i] = error
                    self._group_state[group] = "drained"
                else:
                    self.counters["ahead_failures"] += 1
                    self._group_state[group] = "deferred"
            elif not ready and group not in self._demanded:
                self.counters["ahead_deferrals"] += 1
                self._group_state[group] = "deferred"
            else:
                self._leases[group] = leases
                self._group_state[group] = "open"
            release = self._drained(group) if error is not None else None
            self._pump()
            self._cond.notify_all()
        if release:
            self._unpin_quietly(group)

    def _read(self, index) -> None:
        entry = self._entries[index]
        timing: dict = {}
        error = result = None
        actual = 0
        try:
            if self._cancel.is_set():
                raise CancelledError("io stream closed")
            if entry.reader is not None:
                started = time.perf_counter()
                value, observed = entry.reader()
                timing["read_s"] = time.perf_counter() - started
            else:
                leases = self._leases.get(entry.group) or {}
                decode = (entry.decoder if entry.expected_sha256 is None
                          else partial(_verified_decode, entry.expected_sha256,
                                       entry.decoder))
                value, observed = load_file(
                    Path(entry.path), entry.limit, binding=entry.expected_sha256,
                    decode=decode,
                    declared_signature=(None if entry.declared_stat is None
                                        else _stat_signature(entry.declared_stat)),
                    lease=leases.get(entry.path), timing=timing, sealed=True)
            derived = entry.derive(value, observed) if entry.derive is not None else None
            actual = entry.held_bytes if entry.measure is None else int(entry.measure(value))
            if not 0 <= actual <= entry.held_bytes:
                raise RuntimeError(
                    f"decoded value holds {actual} bytes, beyond its {entry.held_bytes}-byte charge")
            result = (value, observed, derived)
        except BaseException as exc:  # noqa: BLE001 -- surfaced at the consumer
            error = exc.with_traceback(None) if isinstance(exc, TerminalReadError) else exc
        value = observed = derived = None
        with self._cond:
            group = entry.group
            self._active -= 1
            if self._active == 0 and self._read_wall_from is not None:
                self.counters["read_wall_s"] += time.monotonic() - self._read_wall_from
                self._read_wall_from = None
            self._inflight_raw -= entry.raw_bytes
            self._inflight_charge -= entry.raw_bytes + entry.held_bytes
            self._outstanding[group] -= 1
            self._reads[index] += 1
            if self._reads[index] > 1:
                self.counters["rereads"] += 1
            if "read_s" in timing:
                self.counters["entries_read"] += 1
                self.counters["bytes_read"] += entry.size
                self.counters["read_s"] += timing["read_s"]
            if self._closed:
                result = None
            elif error is None:
                self._state[index] = _DONE
                self._values[index], result = result, None
                self._held += entry.held_bytes
                self._actual[index] = actual
                self._held_actual += actual
                self.counters["peak_held_bytes"] = max(
                    self.counters["peak_held_bytes"], self._held)
            elif group in self._demanded or isinstance(error, TerminalReadError):
                self._state[index] = _FAILED
                self._errors[index] = error
                if group not in self._demanded:
                    self.counters["ahead_failures"] += 1
            else:
                # Read ahead, not asked for yet: the consumer's own read of
                # this group decides (module docstring).
                self._state[index] = _PENDING
                self._cursor = min(self._cursor, index)
                self.counters["ahead_failures"] += 1
                self._group_state[group] = "deferred"
            release = self._drained(group)
            self._pump()
            self._cond.notify_all()
        if release:
            self._unpin_quietly(group)

    def _drained(self, group) -> bool:
        """Whether ``group``'s pin can go: no read of it is in flight or due."""
        if self._outstanding[group]:
            return False
        if self._group_state[group] not in ("open", "deferred", "drained"):
            return False
        if (self._group_state[group] == "open"
                and any(self._state[i] == _PENDING for i in self._members[group])):
            return False
        if self._group_state[group] == "open":
            self._group_state[group] = "drained"
        self._leases[group] = None
        if not self._live[group]:
            return False
        self._releasing[group] += 1
        return True

    def _unpin_quietly(self, group) -> None:
        """Release ``group``'s pins outside the lock; :meth:`_drained` said so."""
        with self._cond:
            live, self._live[group] = self._live[group], []
        failure = None
        try:
            unpin(live)
        except Exception as exc:  # noqa: BLE001 -- a stuck pin fails the take
            failure = exc
        with self._cond:
            self._releasing[group] -= 1
            if failure is not None:
                for i in self._members[group]:
                    if self._state[i] in (_PENDING, _DONE):
                        self._state[i] = _FAILED
                        self._errors[i] = failure
            self._cond.notify_all()


def _verified_decode(expected, decoder, raw, receipt, staged):
    """Hold the bytes just hashed to the entry's digest, then decode them.

    The digest is the one :func:`read_file` computed on these bytes, so the
    check costs a comparison, and bytes that are not the expected ones are
    never deserialized.
    """
    if receipt["sha256"] != expected:
        raise RuntimeError(
            f"checksum changed: read sha256 {receipt['sha256']}, expected {expected}")
    return decoder(raw, receipt, staged)


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class IOEngine:
    """The process's one IO thread pool and its read streams."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        # The whole CPU set: a stream holds its consumer's core back itself,
        # and only while the consumer works (``ReadStream._width``, PQ #1533).
        self.width = max(1, len(os.sched_getaffinity(0)))

    def submit(self, fn, *args):
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self.width, thread_name_prefix="pq-io")
            return self._pool.submit(fn, *args)

    def start_scan_process(self, target, *, args, name):
        """Start an isolated filesystem scan before threads or CUDA work.

        The caller owns the control pipe, joins the child, and closes it.
        A separate GIL keeps compilation from delaying filesystem samples.
        """
        import multiprocessing

        process = multiprocessing.get_context("fork").Process(
            target=target, args=args, name=name, daemon=True)
        try:
            process.start()
        except BaseException:
            process.close()
            raise
        return process

    def read_stream(self, entries: Iterable[ReadEntry], *, budget: ReadBudget,
                    ready: Callable[[Hashable, threading.Event], bool] | None = None,
                    lease_counters: dict | None = None) -> ReadStream:
        """Start reading ``entries`` ahead, in order, within ``budget``.

        ``ready(group, cancel)`` (optional) is called on a pool thread before
        a group's first read; it returns whether the group's files are staged
        and may wait for them, and a set ``cancel`` asks it to stop (raising
        ``CancelledError``). The stream starts reading at once; use it as a
        context manager so it is closed. ``lease_counters`` is the report a
        group pin adds to (``staged_lease.enter_group_lease``).
        """
        stream = ReadStream(self, entries, budget, ready, lease_counters)
        with stream._cond:
            stream._pump()
        return stream


ENGINE = IOEngine()


def read_stream(entries: Iterable[ReadEntry], *, budget: ReadBudget, ready=None,
                lease_counters=None) -> ReadStream:
    """:meth:`IOEngine.read_stream` on the process's one engine."""
    return ENGINE.read_stream(entries, budget=budget, ready=ready,
                              lease_counters=lease_counters)


def publish_copy(source, destination, *, length, sha256, source_identity,
                 destination_directory_identity):
    """Copy and hash on the existing IO owner, then publish without replacement."""
    import secrets
    from .staged_manifest import open_regular, open_directory
    if (not isinstance(destination_directory_identity, (tuple, list)) or
            len(destination_directory_identity) != 2 or
            any(type(value) is not int or value <= 0 for value in destination_directory_identity)):
        raise ValueError('copy requires the saved destination directory identity')
    source, destination = Path(source), Path(destination)
    fd = open_regular(str(source))
    directory = target_fd = None
    temporary = None
    try:
        before = _stat_signature(os.fstat(fd))
        if list(before) != list(source_identity) or before[2] != length:
            raise RuntimeError('copy source identity differs from writer')
        directory = open_directory(str(destination.parent), identity=destination_directory_identity)
        candidate = '.copy-' + secrets.token_hex(16)
        target_fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                            0o600, dir_fd=directory)
        temporary = candidate
        actual = copy_extent(fd, target_fd, 0, length)
        if actual != sha256 or _stat_signature(os.fstat(fd)) != before:
            raise RuntimeError('copy source bytes changed')
        check = open_regular(str(source))
        try:
            if _stat_signature(os.fstat(check)) != before:
                raise RuntimeError('copy source path changed')
        finally:
            os.close(check)
        os.fchmod(target_fd, 0o444)
        os.fsync(target_fd)
        os.link(temporary, destination.name, src_dir_fd=directory, dst_dir_fd=directory,
                follow_symlinks=False)
        # Unlink changes ctime through the link count. Publish the signature
        # only after the temporary alias is gone, at the real lifetime boundary.
        os.unlink(temporary, dir_fd=directory)
        temporary = None
        os.fsync(directory)
        check = open_regular(str(destination))
        try:
            observed = _stat_signature(os.fstat(check))
            if observed != _stat_signature(os.fstat(target_fd)):
                raise RuntimeError('copy destination changed during publication')
        finally:
            os.close(check)
        return list(observed)
    finally:
        os.close(fd)
        try:
            if directory is not None:
                try:
                    if temporary is not None:
                        try:
                            current = os.stat(temporary, dir_fd=directory, follow_symlinks=False)
                            owned = os.fstat(target_fd)
                            if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                                os.unlink(temporary, dir_fd=directory)
                        except FileNotFoundError:
                            pass
                finally:
                    os.close(directory)
        finally:
            if target_fd is not None:
                os.close(target_fd)


class FixedBudget:
    """A budget of fixed numbers: ``headroom`` bytes ahead, ``buffer_bytes`` in flight.

    For a caller with no live reading to consult (a CPU run, or a single
    window read on demand, where ``headroom`` is 0 and only demanded groups
    read). What the stream's entries hold, read ahead or taken and not yet
    released, counts against ``headroom``.
    """

    def __init__(self, *, buffer_bytes: int, headroom: int = 0):
        self.buffer_bytes = int(buffer_bytes)
        self._headroom = int(headroom)

    def headroom_bytes(self, held_bytes: int) -> int:
        return self._headroom - held_bytes
