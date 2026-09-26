"""One engine moves bytes from a tier to a consumer (PQ #1294, #1291).

The repo grew one hand-rolled loader per stage, each with its own worker
count, depth constant and verification, and most of them ran load and compute
in turn. ``tests/test_io_site_freeze.py`` freezes those sites; new byte
movement goes through this module. This first version carries one caller,
Stage B's retained-window replay, and it has two layers.

**One file, read once.** :func:`load_file` is the read every ProductionWeight
Cache shard takes (moved here from ``production_weight_cache``): the tier
comes from PrismaBuild's residency map (``residency_map.residency_resolver``
and ``staged_lease``, reused, not restated), the staged copy is read under its
reader lease, the bytes are hashed once and held to the map's digest, and the
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
  next group within the consumer's shortest measured group, and all of its
  pool while the consumer waits or before it has measured anything. The pool
  is sized by the CPU affinity less the consumer's own thread.
* **Order and verification.** A group is delivered only when every one of its
  entries is read, hashed, held to its digest and decoded; entries come back
  in stream order. The first failure in a taken group raises
  :class:`EntryError` at the consumer, naming the entry. A failure while the
  group was only read ahead is not final: the group is read again when the
  consumer asks for it, and that read decides.
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
none: whatever holds its bytes to what was written is the reader's (the
spill checks only that each read is whole and on its grid). Depth, workers,
order, reclaim and the consumer's waits are the stream's, as for a file.

Every consumer wait is timed; ``ReadStream.counters`` is the record the caller
writes to its counters (Stage B: ``counters.json``'s ``io_engine`` block).
"""
from __future__ import annotations

from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from functools import partial
import hashlib
import io
import math
import mmap
import os
from pathlib import Path
import stat
import threading
import time
from typing import Any, Callable, Hashable, Iterable, Protocol

from .perturbed_x_cache import cache_file_stat_signature as _stat_signature
from .residency_map import StagedReadRefused


class EntryError(RuntimeError):
    """One stream entry failed; ``key`` names it."""

    def __init__(self, key, cause: BaseException):
        super().__init__(f"io entry {key!r}: {type(cause).__name__}: {cause}")
        self.key = key


# --------------------------------------------------------------------------
# One file, read once
# --------------------------------------------------------------------------

_SEALS = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE


class SealedBuffer:
    """One file's bytes in a memfd this process owns, sealed once hashed.

    A stream holds what it reads ahead, so those bytes must leave the
    process's committed memory the moment the stream or its consumer drops
    them: the stream's depth is a live reading of that memory, and its
    reclaim frees bytes so a check can pass. A tensor that ``torch.load``
    allocates does not do that on this platform. Torch 2.11 on aarch64 bundles
    mimalloc in ``libc10``, which keeps freed CPU pages: freeing 3.84 GB of
    ``torch.empty`` tensors left 3.85 GB in RssAnon, where the same bytes in a
    ``bytearray`` were returned (PQ #1291, PrismaBuild probe on a GB10).
    Decoded tensors held ahead then pinned the row's committed memory: an
    eviction of 1.95 GB did not move the guard's reading, and the row was
    refused.

    So a stream's read lands in a memfd: its pages are shared memory charged
    to the reading cgroup (``committed_cgroup_bytes`` counts ``shmem``), and
    they are freed when the last descriptor and mapping go. The bytes are
    hashed on the one pass that reads them, then the memfd is sealed against
    every write, so the bytes a decoder sees are the bytes the digest names.
    A decoder maps it instead of copying it (``path`` names the memfd for
    ``torch.load(..., mmap=True)``, a private copy-on-write view): the decoded
    tensor owns the pages, and dropping the tensor returns them.
    """

    __slots__ = ("size", "_fd", "_map")

    def __init__(self, size: int):
        self.size = int(size)
        self._map = None
        self._fd = os.memfd_create("pq-io", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            os.ftruncate(self._fd, self.size)
            if self.size:
                self._map = mmap.mmap(
                    self._fd, self.size,
                    flags=mmap.MAP_SHARED | getattr(mmap, "MAP_POPULATE", 0))
        except BaseException:
            os.close(self._fd)
            raise

    def fill(self, fd: int) -> bool:
        """Read ``fd`` from offset 0 into the buffer; whether it held exactly ``size`` bytes."""
        offset = 0
        if self.size:
            view = memoryview(self._map)
            try:
                while offset < self.size:
                    got = os.preadv(fd, [view[offset:]], offset)
                    if got <= 0:
                        break
                    offset += got
            finally:
                view.release()
        return offset == self.size and os.pread(fd, 1, self.size) == b""

    def seal(self) -> str:
        """Hash the bytes read, drop the writable mapping and seal; the SHA-256 hex."""
        digest = hashlib.sha256()
        if self._map is not None:
            view = memoryview(self._map)
            try:
                digest.update(view)
            finally:
                view.release()
            self._map.close()
            self._map = None
        fcntl.fcntl(self._fd, fcntl.F_ADD_SEALS, _SEALS)
        return digest.hexdigest()

    @property
    def path(self) -> str:
        """A path that opens this memfd (``/proc/self/fd``), for decoders that map files."""
        if self._fd is None:
            raise RuntimeError("sealed io buffer is closed")
        return f"/proc/self/fd/{self._fd}"

    @contextmanager
    def readonly(self):
        """A read-only, seekable view of the bytes (an ``mmap``), without a copy."""
        if self._fd is None:
            raise RuntimeError("sealed io buffer is closed")
        if not self.size:
            yield io.BytesIO(b"")
            return
        view = mmap.mmap(self._fd, self.size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
        try:
            yield view
        finally:
            view.close()

    def __bytes__(self) -> bytes:
        with self.readonly() as view:
            return view.read()

    def __len__(self) -> int:
        return self.size

    def close(self) -> None:
        """Drop this buffer's descriptor; a decoder's mapping keeps the pages it maps."""
        if self._map is not None:
            self._map.close()
            self._map = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def _close_sealed(raw) -> None:
    if isinstance(raw, SealedBuffer):
        raw.close()


def read_file(path: Path, limit: int, *, declared_signature=None, staged=None,
              lease=None, timing: dict | None = None, sealed: bool = False):
    """Read one declared file's bytes once: ``(raw, receipt, signature)``.

    ``raw`` is ``bytes``, or with ``sealed`` a :class:`SealedBuffer` the
    caller closes.

    ``path`` is always the declared file. A staged read opens another copy,
    and every fence that belongs to the declared object still runs on the
    declared object: the caller's stat signature (``declared_signature``), the
    receipt's path, and the signature the receipt's lifetime fence re-checks
    on every later borrow. The staged copy gets its own open/read fences plus
    the digest the map published, and a failure of any of those raises
    ``StagedReadRefused``; :func:`load_file` then reads the declared path
    instead, unless the allowed-tier policy is active, in which case it
    refuses.

    Under the active policy the staged copy is read through a lifetime-pinned
    window: ``lease`` (a group pin from :func:`pin_group`, as
    ``(lease window, key, staged entry)``) when one pinned this file, else one
    per read. The RAM leg needs RAM-mover covers the composed map does not
    carry and refuses fast, the SSD copy acquires honestly with the map's
    leads, payload comes from the held descriptor, and the SDK's own serving
    record is registered at the successful actual open. Inactive policy keeps
    the legacy stage-only open order.

    ``timing``, when given, receives ``read_s``: the seconds spent opening and
    reading the bytes, without the hash, so a caller can measure its
    per-stream rate.
    """
    from .residency_map import residency_resolver
    from .staged_tier_policy import policy_is_active
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError("PWC file receipt requires a regular file, not a symlink")
    if before.st_size > limit:
        raise RuntimeError("PWC file exceeds the explicit read buffer bound")
    signature = _stat_signature(before)
    if declared_signature is not None and signature != declared_signature:
        raise RuntimeError('PWC window file changed before its content read')
    started = time.monotonic()
    source, source_before = path, before
    serving_tier = "pool"
    pinned: tuple | None = None
    if staged is not None:
        # PrismaBuild recomposes the map after every egress, so a staged
        # copy can be released between the resolver's stat and this open.
        # That is an ordinary fallback, not a failed load.
        strict = policy_is_active()
        if not strict:
            source = Path(staged["stage_path"])
            try:
                source_before = source.lstat()
            except OSError as error:
                raise StagedReadRefused(
                    f'staged copy is unreadable: {error.strerror}') from None
            if not stat.S_ISREG(source_before.st_mode):
                raise StagedReadRefused('staged copy is not a regular file')
            if source_before.st_size != before.st_size:
                raise StagedReadRefused('staged copy size differs from the declared file')
            serving_tier = "stage"
        else:
            from .staged_lease import LeaseRefused, acquire_entry_window
            recorder = residency_resolver()
            if recorder is None:
                raise StagedReadRefused("readset-not-staged")
            if lease is not None:
                # The group pin already holds this file; its owner releases
                # it once every read under it is done.
                window, key, _pinned_entry = lease
                try:
                    fd, serving = window.open(key)
                except LeaseRefused as refusal:
                    recorder.record_fallback(path, str(refusal))
                    raise
            else:
                window, key = acquire_entry_window(recorder, path, staged)
                try:
                    window.__enter__()
                except LeaseRefused as refusal:
                    recorder.record_fallback(path, str(refusal))
                    raise
                try:
                    fd, serving = window.open(key)
                except LeaseRefused as refusal:
                    recorder.record_fallback(path, str(refusal))
                    try:
                        window.__exit__(None, None, None)
                    except (LeaseRefused, RuntimeError):
                        pass
                    raise
            serving_tier = window.serving_tier or "stage"
            recorder.record_serving_tier(
                path, serving_tier,
                pin_id=str(serving.get("pin_id") or ""),
                range_ref=str(serving.get("range_ref") or ""))
            pinned = (window, fd, serving, lease is not None)
        recorder = residency_resolver()
        if recorder is not None and pinned is None:
            recorder.record_serving_tier(path, serving_tier)
    source_signature = _stat_signature(source_before)

    def changed(message):
        return (StagedReadRefused(f'staged copy {message}') if staged is not None
                else RuntimeError(f"PWC file {message}"))

    buffer = SealedBuffer(source_before.st_size) if sealed else None
    try:
        if pinned is None:
            try:
                with source.open("rb") as handle:
                    if _stat_signature(os.fstat(handle.fileno())) != source_signature:
                        raise changed("changed before its content read")
                    if buffer is None:
                        raw = handle.read(source_before.st_size + 1)
                        exact = len(raw) == source_before.st_size
                    else:
                        exact = buffer.fill(handle.fileno())
                    if (not exact
                            or _stat_signature(os.fstat(handle.fileno())) != source_signature
                            or _stat_signature(source.lstat()) != source_signature):
                        raise changed("changed during its content read")
            except OSError as error:
                if staged is None:
                    raise
                raise StagedReadRefused(
                    f'staged copy is unreadable: {error.strerror}') from None
        else:
            window, fd, serving, shared = pinned
            try:
                first = os.fstat(fd)
                if first.st_size != source_before.st_size:
                    raise changed("changed before its content read")
                if buffer is None:
                    parts = []
                    remaining = source_before.st_size + 1
                    offset = 0
                    while remaining > 0:
                        block = os.pread(fd, min(remaining, 8 << 20), offset)
                        if not block:
                            break
                        parts.append(block)
                        offset += len(block)
                        remaining -= len(block)
                    raw = b"".join(parts)
                    exact = len(raw) == source_before.st_size
                else:
                    exact = buffer.fill(fd)
                if not exact or _stat_signature(os.fstat(fd)) != _stat_signature(first):
                    raise changed("changed during its content read")
            finally:
                # The owned buffer is fully read above: the descriptor is
                # closed and the exact ref released before deserialization,
                # on success and on failure alike. A group pin outlives this
                # read: its owner releases it.
                if shared:
                    window.close_fd(fd)
                else:
                    window.__exit__(None, None, None)
        if timing is not None:
            timing["read_s"] = time.monotonic() - started
        if staged is not None and _stat_signature(path.lstat()) != signature:
            raise StagedReadRefused('declared file changed during the staged read')
        # The serialized buffer is private to this read and released after its
        # decoder returns. No whole-cache byte store. A sealed buffer is hashed
        # on the pass that sealed it: the bytes a decoder maps are the bytes
        # this digest names, and nothing can write them after.
        if buffer is None:
            digest = hashlib.sha256(raw).hexdigest()
        else:
            digest, raw = buffer.seal(), buffer
        receipt = {"path": str(path), "bytes": len(raw), "sha256": digest}
        if staged is not None:
            # Serving-tier provenance (ID-07) rides the staged receipt;
            # the pool receipt keeps its pinned shape.
            receipt["serving_tier"] = serving_tier
        if staged is not None and receipt["sha256"] != staged["sha256"]:
            # The read is already digested, so the staged bytes are held to the
            # digest the map published for them. This is the check that makes
            # the redirect safe rather than trusted, and on a prepare -- where
            # the caller has no expected digest yet -- it is the only one.
            raise StagedReadRefused('staged bytes differ from the map digest')
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
    ``StagedReadRefused`` on a fence of its own, which falls back like a read
    fence does. With ``sealed`` the decoder gets a :class:`SealedBuffer`,
    closed here once it returns or raises: a value that maps it keeps its
    pages, and nothing else does.

    The redirect lives here because this is the one place a shard's bytes are
    opened with a digest fused to the read: whatever copy is read, the same
    SHA-256 fence decides whether it is admitted, so a wrong or stale staged
    copy is refused exactly as a wrong pool copy would be. The declared path is
    the fallback and the accounting says so. Without
    ``PRISMABUILD_RESIDENCY_MAP`` no resolver exists and the bytes are the
    pool's.

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
                lease=lease if strict else None, timing=timing, sealed=sealed)
            try:
                value, detail = decode(raw, receipt, True)
            finally:
                _close_sealed(raw)
        except StagedReadRefused as refusal:
            if resolver is not None:
                resolver.record_fallback(path, str(refusal))
            if policy_is_active():
                raise refuse_pool_bulk_read(str(path), str(refusal))
        else:
            if resolver is not None:
                if receipt.get("serving_tier") == "ram":
                    resolver.record_ram_read(path, receipt["bytes"])
                else:
                    resolver.record_stage_read(path, receipt["bytes"])
            return value, (receipt, signature, detail)
    if policy_is_active():
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
    if resolver is not None:
        resolver.record_pool_read(path, receipt["bytes"])
    return value, (receipt, signature, detail)


class _LeaseRef:
    """The declared file one group-lease member stands for."""
    __slots__ = ("path",)

    def __init__(self, path: str):
        self.path = path


def pin_group(members, live_leases, *, counters=None):
    """Pin a set of staged files under one reader lease (PQ #1210).

    Under the strict tier policy each read used to acquire, open and release
    its own reader-lease window: on row 041 that was 63% of the loader
    threads' time (the SDK acquire's ownership lock and its scan of every
    retiring mover, the per-entry cover lookup, and the release), paid once
    per 16 MiB file. PrismaBuild's ``acquire`` pins a whole key set under one
    ownership-lock hold (PQ #997), so a group's entries are pinned together
    here, by the same ``_enter_group_lease`` the exact activation cache uses:
    RAM copies in one window and the rest in one SSD window.

    ``members`` is ``[(declared path, expected sha256), ...]``. Returns
    ``{declared path: (lease window, map key, staged entry)}`` for the reads
    to open under, or ``None`` when no group applies: the policy is inactive,
    no resolver is bound, fewer than two files are staged with a bound digest,
    or the batched lease refused. Then every read leases on its own exactly as
    before, so a refusal keeps its single-entry kind. Nothing is served on the
    batched proof alone: the SDK re-verifies every key under its lock at
    acquire, each file is still opened through the SDK under the pin, and its
    bytes are still hashed against the bound digest and the map's digest.
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
    from .perturbed_x_cache import _enter_group_lease
    assignments = _enter_group_lease(resolver, pinned, live_leases, counters=counters)
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
        # The lowest index still pending: everything before it is being
        # read, read, or delivered. A reclaimed entry moves it back.
        self._cursor = 0
        self.counters = {
            "schema": "prismaquant.io_engine_stream.v1",
            "entries": len(entries), "groups": len(groups),
            "pool_width": engine.width,
            "entries_read": 0, "bytes_read": 0, "read_s": 0.0,
            "rereads": 0, "evictions": 0, "evicted_bytes": 0,
            "ahead_deferrals": 0, "ahead_failures": 0,
            "peak_held_bytes": 0, "peak_workers": 0,
            "consumer_wait_s": 0.0, "consumed_bytes": 0, "consumer_busy_s": 0.0,
            "groups_taken": [],
        }

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
            if self._took_at is not None:
                self._release_locked(now)
            self._demanded.add(group)
            if self._group_state[group] == "deferred":
                self._group_state[group] = "closed"
            started = now
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
            nbytes = sum(self._entries[i].size for i in indices)
            self.counters["consumer_wait_s"] += waited
            self.counters["groups_taken"].append(
                {"group": _jsonable(group), "wait_s": waited, "bytes": nbytes,
                 "entries": len(indices),
                 "reread_entries": sum(1 for i in indices if self._reads[i] > 1)})
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
            self._took_at = time.monotonic()
            self._took_bytes = nbytes
            self._unreleased = sum(self._entries[i].held_bytes for i in indices)
            self._pump()
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

    def _workers(self) -> int:
        """Reads to run at once, from the measured rates (module docstring)."""
        width = self._engine.width
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
            self._active += 1
            self.counters["peak_workers"] = max(self.counters["peak_workers"], self._active)
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
            error = exc
        with self._cond:
            self._gating = False
            self._live[group].extend(live)
            if error is not None:
                if group in self._demanded:
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
            error = exc
        value = observed = derived = None
        with self._cond:
            group = entry.group
            self._active -= 1
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
                self._values[index] = result
                self._held += entry.held_bytes
                self._actual[index] = actual
                self._held_actual += actual
                self.counters["peak_held_bytes"] = max(
                    self.counters["peak_held_bytes"], self._held)
            elif group in self._demanded:
                self._state[index] = _FAILED
                self._errors[index] = error
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
        result = None
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
        self.width = max(1, len(os.sched_getaffinity(0)) - 1)

    def submit(self, fn, *args):
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self.width, thread_name_prefix="pq-io")
            return self._pool.submit(fn, *args)

    def read_stream(self, entries: Iterable[ReadEntry], *, budget: ReadBudget,
                    ready: Callable[[Hashable, threading.Event], bool] | None = None,
                    lease_counters: dict | None = None) -> ReadStream:
        """Start reading ``entries`` ahead, in order, within ``budget``.

        ``ready(group, cancel)`` (optional) is called on a pool thread before
        a group's first read; it returns whether the group's files are staged
        and may wait for them, and a set ``cancel`` asks it to stop (raising
        ``CancelledError``). The stream starts reading at once; use it as a
        context manager so it is closed. ``lease_counters`` is the report a
        group pin adds to (``perturbed_x_cache._enter_group_lease``).
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
