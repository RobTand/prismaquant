"""Safetensors reads through the existing staged resolver and owned buffers.

Strict readers require declared staged headers and payload extents. Source
and object offsets are independent. Bounded chunk workers join before a
window closes; inherited reader operations after fork refuse.
"""
from __future__ import annotations
from pathlib import Path

from .mount_facts import MountFacts, find_mount

from concurrent.futures import CancelledError, ThreadPoolExecutor
import errno
import json
import math
import os
import re
import stat
import threading
import time

import torch

from .file_identity import file_stat_signature
from .source_read_plan import safetensors_prefix_length
from .residency_map import RANGE_HIT, RANGE_UNCOVERED, residency_resolver
from .staged_tier_policy import (
    StagedRangeNotLanded,
    active_policy,
    policy_is_active,
    refuse_pool_bulk_read,
)
from .staged_lease import LeaseRefused, acquire_entry_window


# Retained planner setting; immutable native reads refuse immediately on gaps.
STAGED_RANGE_WAIT_ENV = "PRISMAQUANT_STAGED_RANGE_WAIT_S"
STAGED_RANGE_WAIT_S = 300.0


def await_staged_spans(resolver, wanted, *, deadline, published=None, cancel=None,
                       published_batch=None):
    """One immutable coverage check; cold/missing input never polls PB."""
    if cancel is not None and cancel.is_set():
        return RANGE_UNDECLARED
    for declared, start, end, size in wanted:
        if start == end:
            continue
        entry, verdict = resolver.staged_range_outcome(declared, start, end, declared_size=size)
        if entry is None:
            return verdict
        if published is not None and not published(resolver, declared, entry):
            return RANGE_UNDECLARED
    return RANGE_HIT

def staged_range_wait_s() -> float:
    """The configured wait bound in seconds: finite and not negative.

    ``math.isfinite`` rather than a ``>= 0`` test alone, because the one
    property this design rests on is that the wait *ends*: ``float("inf")``
    is neither negative nor NaN, and a deadline of ``started + inf`` never
    expires. An unbounded wait is not a longer bound, it is no bound.
    """
    return staged_range_wait_from_env(os.environ)


def staged_range_wait_from_env(env) -> float:
    """:func:`staged_range_wait_s` read from ``env`` rather than the process.

    The joint dispatcher reads a campaign spec's ``env`` block with the same
    rules before it submits, so the wait it compares with the row's progress
    grace is the wait the reader inside the container will use.
    """
    raw = env.get(STAGED_RANGE_WAIT_ENV)
    if raw is None or not str(raw).strip():
        return STAGED_RANGE_WAIT_S
    try:
        value = float(str(raw).strip())
    except ValueError:
        raise ValueError(
            f"{STAGED_RANGE_WAIT_ENV} must be a number of seconds, "
            f"not {raw!r}") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"{STAGED_RANGE_WAIT_ENV} must be a finite number of seconds "
            f">= 0, not {raw!r}")
    return value


try:
    # safetensors' own dtype table, so the reader reads the format's spelling
    # rather than asserting one of its own. Without it no tensor is served
    # from the stage; a hand-rolled table would drift silently.
    from safetensors.torch import _TYPES as _SAFETENSORS_DTYPES
except ImportError:  # pragma: no cover - a safetensors that moved the table
    _SAFETENSORS_DTYPES = None

# The same bound ``layer_streaming._advise_consumed_safetensors_pages`` puts on
# the same structure.
MAX_HEADER_BYTES = 100_000_000


#: The environment's override for the number of concurrent reads one staged
#: span is split into. ``1`` restores the single-stream read this reader
#: shipped with; unset takes the mount's own ``nconnect``.
STREAMS_ENV = "PRISMAQUANT_STAGED_READ_STREAMS"

#: Where the kernel publishes the mount table this reader reads its two numbers
#: from. Named so a test can point it at a table of its own.
MOUNTS_PATH = "/proc/self/mounts"

_READ_SHAPE_LOCK = threading.Lock()
_READ_SHAPE_CACHE: dict[str, tuple[int, int] | None] = {}
_CHUNK_POOL_LOCK = threading.Lock()
_CHUNK_POOL: ThreadPoolExecutor | None = None
_CHUNK_POOL_SIZE = 0

def _read_shape_for(mount: MountFacts) -> tuple[int, int] | None:
    """``(streams, chunk_bytes)`` for reads on ``mount``, or None to stay serial.

    Both numbers are the mount's, not ours (principle 2). ``nconnect`` is how
    many transports the NFS client actually holds open to the server, so it is
    the ceiling on how many of this span's reads can be in flight at once;
    ``rsize`` is the size of the read the client issues, so it is the unit a
    span is cut on and a cut anywhere else only splits one wire read in two.

    ``None`` -- a mount that is not NFS, or an NFS mount publishing neither --
    means there is no explicit to read, and a reader with no explicit reads the
    way it read before. ``STREAMS_ENV`` overrides the stream count for an A/B;
    ``1`` is the single-stream read and is what the before arm sets.
    """
    if not mount.filesystem.startswith("nfs"):
        return None
    streams = chunk = 0
    for name, value in mount.options.items():
        if name == "nconnect" and value.isdigit():
            streams = int(value)
        elif name == "rsize" and value.isdigit():
            chunk = int(value)
    override = str(os.environ.get(STREAMS_ENV, "")).strip()
    if override:
        try:
            streams = max(1, int(override))
        except ValueError:
            pass
    if streams <= 1 or chunk <= 0:
        return None
    return streams, chunk


def _read_shape(path: str) -> tuple[int, int] | None:
    """The read shape for ``path``, deriving it once per mount point.

    The mount table is read either way -- it is what says which mount the path
    is on -- and the cache saves the option parse and the environment read, not
    the table read.
    """
    mount = find_mount(path, table=MOUNTS_PATH)
    if mount is None:
        return None
    with _READ_SHAPE_LOCK:
        if mount.point in _READ_SHAPE_CACHE:
            return _READ_SHAPE_CACHE[mount.point]
    shape = _read_shape_for(mount)
    with _READ_SHAPE_LOCK:
        _READ_SHAPE_CACHE[mount.point] = shape
    return shape


def reset_read_shape_cache_for_tests() -> None:
    with _READ_SHAPE_LOCK:
        _READ_SHAPE_CACHE.clear()


def _chunk_pool(streams: int) -> ThreadPoolExecutor:
    """One pool for the whole process, sized by the mount's transport count.

    Shared on purpose. ``layer_streaming.read_prefix_tensors`` already reads a
    layer's tensors on several threads, so a per-reader pool would multiply
    (gather threads x chunk threads) into more in-flight reads than the client
    has transports to carry. The chunk tasks submit nothing themselves, so a
    gather thread waiting on this pool cannot deadlock against it.
    """
    global _CHUNK_POOL, _CHUNK_POOL_SIZE
    with _CHUNK_POOL_LOCK:
        if _CHUNK_POOL is None or _CHUNK_POOL_SIZE < streams:
            if _CHUNK_POOL is not None:
                _CHUNK_POOL.shutdown(wait=False)
            _CHUNK_POOL = ThreadPoolExecutor(
                max_workers=streams, thread_name_prefix="pq-staged-read")
            _CHUNK_POOL_SIZE = streams
        return _CHUNK_POOL


def _cuts(offset: int, count: int, chunk: int) -> list[tuple[int, int]]:
    """``[(offset, length), ...]`` covering ``[offset, offset + count)`` exactly.

    Aligned to ``chunk`` in the staged file's own offset space, so every read
    but the first and last is one whole ``rsize`` request. The pieces are
    disjoint and their lengths sum to ``count``; nothing is read twice and no
    byte is left out.
    """
    cuts = []
    position = offset
    end = offset + count
    while position < end:
        boundary = ((position // chunk) + 1) * chunk
        stop = min(boundary, end)
        cuts.append((position, stop - position))
        position = stop
    return cuts


def _read_shard_header(path: str) -> tuple[dict, int, int]:
    """The shard's header, its payload base, and the declared file's length.

    Prefix decode and bounds use the source-read-plan owner. This reader
    keeps its descriptor acquisition, regular-file fence, exact body-length
    and JSON-object checks, refusal messages, and finally-close contract;
    the best-effort consumed-page advice reader has a different policy.
    """
    handle = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
    try:
        info = os.fstat(handle)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("source shard is not a regular file")
        raw = os.pread(handle, 8, 0)
        header_bytes = safetensors_prefix_length(
            raw, info.st_size, max_bytes=MAX_HEADER_BYTES,
            short_error="source shard has no safetensors header length",
            range_error="source shard header length is out of range")
        blob = os.pread(handle, header_bytes, 8)
        if len(blob) != header_bytes:
            raise ValueError("source shard header is shorter than it declares")
        header = json.loads(blob)
        if type(header) is not dict:
            raise ValueError("source shard header is not an object")
        return header, 8 + header_bytes, info.st_size
    finally:
        os.close(handle)


def _pread_into(fd: int, view: memoryview, offset: int) -> None:
    """Fill ``view`` from ``offset``, in as many reads as the kernel needs.

    A single ``pread`` is capped near 2 GiB on Linux and an NFS read can be
    short for its own reasons, so the loop is the contract rather than an
    optimization.
    """
    count = len(view)
    done = 0
    while done < count:
        moved = os.preadv(fd, [view[done:]], offset + done)
        if moved <= 0:
            raise OSError(errno.EIO, "staged range ended before the tensor's bytes")
        done += moved


def _read_span(fd: int, count: int, offset: int,
               shape: tuple[int, int] | None) -> bytearray:
    """``count`` bytes at ``offset``, on as many streams as the mount holds.

    One buffer, cut into disjoint pieces that are read at the same time and
    written straight into their own slice of it, so the bytes are the bytes a
    single sequential read would have produced, piece by piece and offset by
    offset. The buffer is writable, which is what ``torch.frombuffer`` wants,
    and it is the tensor's own storage afterwards.

    A span shorter than one read per stream is read on one stream: splitting it
    would hand some streams nothing and cost a round trip to find out. Every
    piece is waited for before the result is looked at, including on a failure,
    so no thread is still writing into the buffer -- or reading the descriptor
    the caller is about to close -- when this returns.
    """
    buffer = bytearray(count)
    view = memoryview(buffer)
    if shape is None or count < shape[0] * shape[1]:
        _pread_into(fd, view, offset)
        return buffer
    streams, chunk = shape
    cuts = _cuts(offset, count, chunk)
    pool = _chunk_pool(streams)
    futures = [pool.submit(_pread_into, fd, view[at - offset:at - offset + size], at)
               for at, size in cuts]
    failure = None
    for future in futures:
        try:
            future.result()
        except BaseException as error:  # noqa: BLE001 - re-raised after the join
            if failure is None:
                failure = error
    if failure is not None:
        raise failure
    return buffer


class _StrictSliceProxy:
    """Header-only slice metadata; payload materializes via the staged reader.

    ``get_shape``/``get_dtype`` are served from the bounded header parse
    (metadata, allowed). Any indexing materializes the full tensor through
    the reader's staged path — the same fences, the same refusal, no pool
    fallback — then applies the index. This is what keeps the shape
    estimators working under policy without a payload exemption by naming.
    """

    def __init__(self, reader, name, dtype, shape):
        self._reader = reader
        self._name = name
        self._dtype = dtype
        self._shape = tuple(shape)

    def get_shape(self):
        return list(self._shape)

    def get_dtype(self):
        return self._dtype

    def __getitem__(self, index):
        tensor = self._reader._staged_tensor(self._name)
        if tensor is None:
            raise refuse_pool_bulk_read(
                self._reader._declared, "pool-fallback")
        return tensor[index]


class StagedShardReader:
    """A ``safe_open`` handle whose payload comes off the stage when it can.

    Drop-in at the ``_source_safe_open`` seam, including through
    ``source_authentication.safe_open(opener, path, **kwargs)``, which calls the
    opener with ``/proc/self/fd/<fd>`` rather than the declared path. The
    declared path travels in the closure ``staged_shard_opener`` builds; the
    handed path is what is opened and what the header is read from, and the two
    are the same file.

    Under the active allowed-tier policy no pool ``safe_open`` handle is
    constructed at all — no pool mmap merely for header. Keys, metadata,
    shapes and dtypes come from the existing bounded header reader
    (``_read_shard_header``); payload comes from staged ranges or refuses.

    Under active allowed-tier policy, concurrency is threads, never fork:
    payload and lifecycle operations
    (``get_tensor``, ``get_slice``, ``__enter__``, ``__exit__``) reject
    inherited handle operations in a forked child, so a child cannot
    drive reads through state the parent's release retires. ``keys``/
    ``metadata`` stay available: stateless bounded-header metadata with
    no retained descriptor. Raw descriptor escape past these operations
    is unsupported (not policed here): a forked child must not perform
    I/O on any inherited descriptor or mapping and should ``_exit``
    without it.
    """

    def __init__(self, pool_open, path, declared, resolver, kwargs):
        self._declared = declared
        self._path = os.fspath(path)
        self._resolver = resolver
        self._device = kwargs.get("device")
        self._owner_pid = os.getpid()
        # Captured at construction (opener time): the entrypoints activate
        # the process-global policy before any read, so every reader built
        # afterwards — on any thread — sees the same verdict.
        self._strict = policy_is_active()
        # The allowed set beside the verdict: tier checks read the same
        # snapshot the reader was built under, not a later global.
        self._allowed = active_policy() if self._strict else None
        if self._strict:
            self._handle = None
        else:
            # First, so a ``safe_open`` that rejects the ``device`` kwarg raises
            # where it raises today and the callers' retry without it still works.
            self._handle = pool_open(path, **kwargs)
        self._header = None
        self._base = 0
        self._declared_size = 0
        self._parsed = False
        self._shape: tuple[int, int] | None = None
        self._bound: list[tuple] = []

    def _require_owner(self, operation: str) -> None:
        if os.getpid() != self._owner_pid:
            raise RuntimeError(
                f"StagedShardReader.{operation} from a forked child is "
                "unsupported: inherited bound handles must not be driven "
                "past the holder's lifecycle. Use threads, which share the "
                "holder pid, or fork with no live reader.")

    # -- the handle interface --------------------------------------------

    def __enter__(self):
        self._require_owner("__enter__")
        if self._handle is not None:
            self._handle.__enter__()
        return self

    def __exit__(self, *args):
        self._require_owner("__exit__")
        # The window owns every bound descriptor: close each through it,
        # then release its exact ref. Release-before-close is forbidden,
        # and a forked child never releases (the window's pid guard makes
        # that explicit).
        failure = None
        while self._bound:
            row = self._bound.pop()
            window = row[6] if len(row) > 6 else None
            if window is not None:
                try:
                    window.close_fd(row[2])
                except OSError as exc:
                    if failure is None:
                        failure = exc
                try:
                    window.__exit__(None, None, None)
                except LeaseRefused as exc:
                    if failure is None:
                        failure = exc
            else:
                try:
                    os.close(row[2])
                except OSError as exc:
                    if failure is None:
                        failure = exc
        if self._handle is not None:
            try:
                result = self._handle.__exit__(*args)
            except BaseException as exc:
                if failure is None:
                    failure = exc
                result = False
            if failure is not None:
                raise failure
            return result
        if failure is not None:
            raise failure
        return False

    def keys(self):
        if not self._strict:
            return self._handle.keys()
        self._parse()
        if self._header is None:
            raise refuse_pool_bulk_read(self._declared, "header-unreadable")
        return [name for name in self._header if name != "__metadata__"]

    def metadata(self):
        if not self._strict:
            return self._handle.metadata()
        self._parse()
        if self._header is None:
            raise refuse_pool_bulk_read(self._declared, "header-unreadable")
        meta = self._header.get("__metadata__")
        return dict(meta) if type(meta) is dict else {}

    def get_slice(self, name):
        """Shape/dtype metadata from the header; payload via the staged path.

        The returned proxy serves ``get_shape``/``get_dtype`` without
        touching payload bytes. Indexing it materializes through the
        reader's staged tensor path — same fences, same refusal. No pool
        fallback either way. Inactive policy delegates to the pool handle
        as before.
        """
        if not self._strict:
            return self._handle.get_slice(name)
        self._require_owner("get_slice")
        span = self._span(name)
        if span is None:
            raise refuse_pool_bulk_read(self._declared, "span-not-bound")
        _, _, dtype, shape = span
        return _StrictSliceProxy(self, name, dtype, shape)

    def get_tensor(self, name):
        """The tensor, off the stage when a staged range covers its whole span.

        ``bytes_from_pool`` counts what this reader falls back to, not every
        shard byte the run reads: a shard the map never names is opened by
        ``safe_open`` itself and no reader sees it. The stage-side count has no
        such gap, so read the two as "what the stage served" and "what this
        reader could not get from it", not as a partition of the run.

        Under the active allowed-tier policy there is no pool fallback:
        a span no staged range covers, or a fence the staged copy fails,
        raises ``TierPolicyRefused`` before a pool payload byte is read.
        Zero-size tensors carry no payload bytes and are built locally.
        """
        if self._strict:
            self._require_owner("get_tensor")
        served = self._staged_tensor(name)
        if served is not None:
            return served
        if self._strict:
            raise refuse_pool_bulk_read(self._declared, "pool-fallback")
        tensor = self._handle.get_tensor(name)
        if self._resolver is not None:
            self._resolver.record_pool_read(self._declared, tensor.nbytes)
        return tensor

    # -- the stage -------------------------------------------------------

    def _parse(self) -> None:
        """Read the header once when payload or strict metadata first needs it.

        Strict ``keys()``, ``metadata()`` and ``get_slice()`` also parse the
        header. Inactive metadata queries delegate to the pool handle.
        """
        if self._parsed:
            return
        self._parsed = True
        if _SAFETENSORS_DTYPES is None:
            if self._resolver is not None:
                self._resolver.record_fallback(
                    self._declared,
                    "safetensors publishes no dtype table this reader can read")
            return
        try:
            if self._strict:
                if self._resolver is None:
                    raise refuse_pool_bulk_read(self._declared, 'header-not-staged')
                self._declared_size = self._resolver.source_size(self._declared)
                prefix = self._staged_bytes(0, 8)
                header_bytes = safetensors_prefix_length(
                    prefix, self._declared_size, max_bytes=MAX_HEADER_BYTES,
                    short_error='source shard has no staged header length',
                    range_error='staged shard header length is out of range')
                self._header = json.loads(self._staged_bytes(8, 8 + header_bytes))
                if type(self._header) is not dict:
                    raise ValueError('staged header is not an object')
                self._base = 8 + header_bytes
            else:
                self._header, self._base, self._declared_size = _read_shard_header(self._path)
        except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as error:
            self._header = None
            if self._resolver is not None:
                self._resolver.record_fallback(
                    self._declared, f"source shard header is unreadable: {error}")

    def _span(self, name):
        """``(start, end, dtype, shape)`` in the declared file, or None."""
        self._parse()
        if self._header is None:
            return None
        row = self._header.get(name)
        if type(row) is not dict:
            return None
        dtype = _SAFETENSORS_DTYPES.get(row.get("dtype"))
        offsets, shape = row.get("data_offsets"), row.get("shape")
        if (dtype is None or type(offsets) is not list or len(offsets) != 2
                or type(shape) is not list
                or any(type(dim) is not int or isinstance(dim, bool) or dim < 0
                       for dim in shape)):
            return None
        begin, end = offsets
        if (type(begin) is not int or type(end) is not int
                or isinstance(begin, bool) or isinstance(end, bool)
                or not 0 <= begin <= end <= self._declared_size - self._base):
            return None
        count = 1
        for dim in shape:
            count *= dim
        if count * dtype.itemsize != end - begin:
            return None
        return self._base + begin, self._base + end, dtype, tuple(shape)

    def _range_for(self, start, end):
        """The bound staged range covering ``[start, end)``, or None.

        Bound ranges are held open for the reader's life, so a layer's several
        hundred tensors cost one resolver lookup and one open per staged range
        rather than one per tensor.

        Under the active allowed-tier policy the range is pinned for the
        reader's life: one lifetime window per staged entry (never per
        tensor), payload read through the held descriptor, the SDK's own
        serving record registered at the successful actual open, and the
        exact ref released after the bound descriptors close. A RAM leg
        needs RAM-mover covers the composed map does not carry, so it
        refuses fast and the SSD copy acquires honestly with its own
        material and lifetime. A typed retiring refusal may select another
        checked overlap through its own lease; other refusals propagate,
        never becoming pool reads. Inactive policy keeps the legacy
        stage-only open order.
        """
        for row in self._bound:
            if row[0] <= start and end <= row[1]:
                return row
        strict = self._strict
        if self._resolver is None:
            if strict:
                raise refuse_pool_bulk_read(self._declared, "readset-not-staged")
            return None
        entry, outcome = self._resolver.staged_range_outcome(
            self._declared, start, end, declared_size=self._declared_size)
        # Nothing waits here. This runs on a worker of the shared, bounded
        # ``layer_streaming._LAYER_READ_POOL``, and a worker sleeping on a
        # cold future range is a worker the current layer's already-staged
        # reads queue behind. Readiness is decided one level up, before
        # these slots are occupied: ``layer_streaming._await_layer_readset``
        # (PQ #874). By the time a chunk asks, the answer is final.
        if entry is None:
            if strict:
                if outcome == RANGE_UNCOVERED:
                    # Declared and simply not landed yet: the one transient
                    # cause a demand-side retry may match on. Undeclared
                    # spans and failed covering entries keep the generic
                    # refusal below: nothing about them is on its way.
                    raise StagedRangeNotLanded(self._declared, start, end)
                raise refuse_pool_bulk_read(self._declared, "readset-not-staged")
            return None
        if not strict:
            try:
                fd = os.open(entry["stage_path"], os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            except OSError as error:
                self._resolver.record_fallback(
                    self._declared, f"staged range is unreadable: {error.strerror}")
                return None
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("staged range is not a regular file")
                if info.st_size != entry["bytes"]:
                    raise ValueError("staged range size differs from the map")
            except (OSError, ValueError) as error:
                os.close(fd)
                self._resolver.record_fallback(self._declared, f"staged range {error}")
                return None
            if not self._bound:
                # Every range of one declared shard is staged under the same root,
                # so the mount's read shape is read once per reader, not per tensor.
                self._shape = _read_shape(entry["stage_path"])
            row = (entry["offset"], entry["offset"] + entry["bytes"], fd,
                   file_stat_signature(info), entry, "stage")
            self._bound.append(row)
            return row
        from .staged_lease import LeaseRefused, acquire_entry_window
        alternatives = None
        hard_refusal = None
        while True:
            try:
                window, key = acquire_entry_window(
                    self._resolver, self._declared, entry)
                window.__enter__()
                break
            except LeaseRefused as refusal:
                # A retired phase may still have a physically valid file
                # while an older reader drains. Its closed generation cannot
                # hide another overlapping entry that admits its OWN pin.
                # Unknown/identity/integrity refusals never take this path.
                if refusal.kind != 'availability' or refusal.reason != 'retiring':
                    self._resolver.record_fallback(self._declared, str(refusal))
                    raise
                if alternatives is None:
                    candidates, hard_refusal = self._resolver.staged_range_alternatives(
                        self._declared, start, end,
                        rejected_offset=entry['offset'], declared_size=self._declared_size)
                    alternatives = iter(candidates)
                entry = next(alternatives, None)
                if entry is None:
                    if hard_refusal is not None:
                        refusal = LeaseRefused(hard_refusal, kind='integrity')
                    self._resolver.record_fallback(self._declared, str(refusal))
                    raise refusal
        try:
            fd, serving = window.open(key)
            info = os.fstat(fd)
            if info.st_size != entry.get('object_bytes', entry['bytes']):
                raise LeaseRefused("lease-open-size-changed", kind="integrity")
        except (OSError, LeaseRefused) as error:
            try:
                window.__exit__(None, None, None)
            except (LeaseRefused, RuntimeError):
                pass
            if isinstance(error, LeaseRefused):
                self._resolver.record_fallback(self._declared, str(error))
                raise
            reason = f"staged range is unreadable: {error.strerror}"
            self._resolver.record_fallback(self._declared, reason)
            raise refuse_pool_bulk_read(self._declared, reason)
        # The open fence passed on a held descriptor: the SDK's own
        # serving record is registered at this successful actual open.
        tier = window.serving_tier or "stage"
        self._resolver.record_serving_tier(
            self._declared, tier,
            pin_id=str(serving.get("pin_id") or ""),
            range_ref=str(serving.get("range_ref") or ""))
        if not self._bound:
            # Every range of one declared shard is staged under the same
            # root, so the mount's read shape is read once per reader, not
            # per tensor — from the path actually opened (the RAM copy for
            # a RAM window), never an assumed tier.
            self._shape = _read_shape(window.stage_path(key)
                                      or entry["stage_path"])
        row = (entry["offset"], entry["offset"] + entry["bytes"], fd,
               file_stat_signature(info), entry, tier, window)
        self._bound.append(row)
        return row

    def _drop(self, row) -> None:
        if row in self._bound:
            self._bound.remove(row)
        window = row[6] if len(row) > 6 else None
        if window is not None:
            # The window owns the descriptor: close through it, then
            # release its exact ref now (after the descriptor above)
            # rather than lending it to a later tensor. A release failure
            # is recorded, never silent; the caller's read error still
            # raises.
            try:
                window.close_fd(row[2])
            except OSError:
                pass
            try:
                window.__exit__(None, None, None)
            except LeaseRefused as exc:
                if self._resolver is not None:
                    self._resolver.record_fallback(self._declared, str(exc))
            return
        try:
            os.close(row[2])
        except OSError:
            pass

    def _staged_tensor(self, name):
        span = self._span(name)
        if span is None:
            if self._strict:
                raise refuse_pool_bulk_read(self._declared, "span-not-bound")
            return None
        start, end, dtype, shape = span
        if end == start:
            # No bytes to serve: an empty tensor is built locally rather
            # than read from any tier, pool included.
            if self._strict:
                tensor = torch.empty(shape, dtype=dtype)
                if self._device is not None:
                    tensor = tensor.to(self._device)
                return tensor
            return None
        row = self._range_for(start, end)
        if row is None:
            return None
        fd, signature, entry, tier = row[2], row[3], row[4], row[5]
        try:
            raw = _read_span(fd, end - start, entry.get('object_offset', 0) + start - entry["offset"], self._shape)
            if len(row) > 6:
                row[6].check_fd(fd)
            if file_stat_signature(os.fstat(fd)) != signature:
                raise ValueError("changed during its content read")
        except (OSError, ValueError) as error:
            self._drop(row)
            reason = f"staged range {getattr(error, 'strerror', None) or error}"
            if self._resolver is not None:
                self._resolver.record_fallback(self._declared, reason)
            if self._strict:
                raise refuse_pool_bulk_read(self._declared, reason)
            return None
        tensor = torch.frombuffer(raw, dtype=torch.uint8).view(dtype).reshape(shape)
        if self._device is not None:
            tensor = tensor.to(self._device)
        if self._resolver is not None:
            if tier == "ram":
                # A range read the ram tier served: ram bytes, not stage
                # bytes. ``range_hits`` stays the stage-range count, so the
                # two tiers never double-count the same bytes.
                self._resolver.record_ram_read(self._declared, end - start)
            else:
                self._resolver.record_stage_range_read(self._declared, end - start)
        return tensor

    def _staged_bytes(self, start, end):
        row = self._range_for(start, end)
        if row is None:
            raise refuse_pool_bulk_read(self._declared, 'header-not-staged')
        entry = row[4]
        if entry['bytes'] > MAX_HEADER_BYTES + 8:
            raise refuse_pool_bulk_read(self._declared, 'header-coverage-must-be-bounded')
        from .staged_whole_file import read_staged_entry
        raw = read_staged_entry(self._resolver, Path(self._declared), entry,
                                label='safetensors-header', max_bytes=MAX_HEADER_BYTES + 8)
        return raw[start - entry['offset']:end - entry['offset']]


def staged_shard_opener(declared, pool_open):
    """The opener for ``declared``: ``pool_open`` itself unless the stage holds it.

    Handing the caller back its own opener is what keeps the unmapped path
    identical -- same callable, same arguments, no wrapper and no extra
    syscall. A file the map never names takes that path too, so wrapping is
    paid for only where it can pay off.

    Under the active allowed-tier policy there is no unwrapped path: a file
    the map never names (or no map at all) still opens through a reader
    whose header/keys/metadata come from the bounded header reader,
    but whose payload reads refuse with
    ``readset-not-staged`` instead of serving pool bytes.
    """
    resolver = residency_resolver()
    if resolver is None or not resolver.stages(declared):
        if policy_is_active():
            path = os.fspath(declared)

            def strict_opener(handed, **kwargs):
                return StagedShardReader(pool_open, handed, path, resolver, kwargs)

            return strict_opener
        return pool_open
    path = os.fspath(declared)

    def opener(handed, **kwargs):
        return StagedShardReader(pool_open, handed, path, resolver, kwargs)

    return opener
