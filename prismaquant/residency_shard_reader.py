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
from .residency_map import RANGE_HIT, RANGE_UNDECLARED, residency_resolver
from .staged_tier_policy import policy_is_active, refuse_pool_bulk_read
from .staged_lease import LeaseRefused, acquire_entry_window


def await_staged_spans(resolver, wanted, *, cancel=None):
    """Check immutable coverage once; missing input never polls or waits."""
    if cancel is not None and cancel.is_set():
        return RANGE_UNDECLARED
    for declared, start, end, size in wanted:
        if start == end:
            continue
        entry, verdict = resolver.staged_range_outcome(declared, start, end, declared_size=size)
        if entry is None:
            return verdict
    return RANGE_HIT


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
        return self._reader._staged_tensor(self._name)[index]


class StagedShardReader:
    """Existing safe_open interface over immutable staged extents.

    Headers and payload both require declared coverage. The reader exists
    only under the active allowed-tier policy, so no pool handle is built.
    Descriptors remain in windows through the reader lifetime; tensors own
    their buffers. Forked operations refuse. Threads share the owning reader
    lifetime.
    """

    def __init__(self, declared, resolver, kwargs):
        self._declared = declared
        self._resolver = resolver
        self._device = kwargs.get("device")
        self._owner_pid = os.getpid()
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
            window = row[5]
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
        if failure is not None:
            raise failure
        return False

    def keys(self):
        self._parse()
        if self._header is None:
            raise refuse_pool_bulk_read(self._declared, "header-unreadable")
        return [name for name in self._header if name != "__metadata__"]

    def metadata(self):
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
        fallback either way.
        """
        self._require_owner("get_slice")
        span = self._span(name)
        if span is None:
            raise refuse_pool_bulk_read(self._declared, "span-not-bound")
        _, _, dtype, shape = span
        return _StrictSliceProxy(self, name, dtype, shape)

    def get_tensor(self, name):
        """The tensor, off the stage when a staged range covers its whole span.

        There is no pool fallback: a span no staged range covers, or a fence
        the staged copy fails, raises ``TierPolicyRefused`` before a pool
        payload byte is read. Zero-size tensors carry no payload bytes and
        are built locally.
        """
        self._require_owner("get_tensor")
        return self._staged_tensor(name)

    # -- the stage -------------------------------------------------------

    def _parse(self) -> None:
        """Read the staged header once, when payload or metadata first needs it."""
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
        """Retain a checked descriptor window per covering immutable extent."""
        for row in self._bound:
            if row[0] <= start and end <= row[1]:
                return row
        if self._resolver is None:
            raise refuse_pool_bulk_read(self._declared, "readset-not-staged")
        entry = self._resolver.staged_range(
            self._declared, start, end, declared_size=self._declared_size)
        # Nothing waits here. This runs on a worker of the shared, bounded
        # ``layer_streaming._LAYER_READ_POOL``. Readiness is decided one level
        # up, before these slots are occupied:
        # ``layer_streaming._await_layer_readset`` (PQ #874).
        if entry is None:
            raise refuse_pool_bulk_read(self._declared, "readset-not-staged")
        window, key = acquire_entry_window(self._resolver, self._declared, entry)
        try:
            window.__enter__()
        except LeaseRefused as refusal:
            self._resolver.record_fallback(self._declared, str(refusal))
            raise
        try:
            fd, serving = window.open(key)
            info = os.fstat(fd)
            if info.st_size != entry.get('object_bytes', entry['bytes']):
                raise LeaseRefused("lease-open-size-changed", kind="integrity")
            self._resolver.record_serving_tier(
                self._declared, "stage",
                pin_id=str(serving.get("pin_id") or ""),
                range_ref=str(serving.get("range_ref") or ""))
            if not self._bound:
                self._shape = _read_shape(window.stage_path(key)
                                          or entry["stage_path"])
            row = (entry["offset"], entry["offset"] + entry["bytes"], fd,
                   file_stat_signature(info), entry, window)
            self._bound.append(row)
        except BaseException as error:
            try:
                window.__exit__(None, None, None)
            except BaseException as cleanup:
                error.add_note(f"staged window cleanup also failed: {cleanup}")
            if isinstance(error, OSError):
                reason = f"staged range is unreadable: {error.strerror}"
                self._resolver.record_fallback(self._declared, reason)
                raise refuse_pool_bulk_read(self._declared, reason) from error
            self._resolver.record_fallback(self._declared, str(error))
            raise
        return row

    def _drop(self, row) -> None:
        if row in self._bound:
            self._bound.remove(row)
        window = row[5]
        # The window owns the descriptor: close through it, then release its
        # exact ref now rather than lending it to a later tensor. A release
        # failure is recorded, never silent; the caller's read error still
        # raises.
        try:
            window.close_fd(row[2])
        except OSError:
            pass
        try:
            window.__exit__(None, None, None)
        except LeaseRefused as exc:
            self._resolver.record_fallback(self._declared, str(exc))

    def _staged_tensor(self, name):
        span = self._span(name)
        if span is None:
            raise refuse_pool_bulk_read(self._declared, "span-not-bound")
        start, end, dtype, shape = span
        if end == start:
            # No bytes to serve: an empty tensor is built locally rather
            # than read from any tier, pool included.
            tensor = torch.empty(shape, dtype=dtype)
            if self._device is not None:
                tensor = tensor.to(self._device)
            return tensor
        row = self._range_for(start, end)
        fd, signature, entry, window = row[2], row[3], row[4], row[5]
        try:
            raw = _read_span(fd, end - start, entry.get('object_offset', 0) + start - entry["offset"], self._shape)
            window.check_fd(fd)
            if file_stat_signature(os.fstat(fd)) != signature:
                raise ValueError("changed during its content read")
        except (OSError, ValueError) as error:
            self._drop(row)
            reason = f"staged range {getattr(error, 'strerror', None) or error}"
            self._resolver.record_fallback(self._declared, reason)
            raise refuse_pool_bulk_read(self._declared, reason)
        tensor = torch.frombuffer(raw, dtype=torch.uint8).view(dtype).reshape(shape)
        if self._device is not None:
            tensor = tensor.to(self._device)
        self._resolver.record_stage_range_read(self._declared, end - start)
        return tensor

    def _staged_bytes(self, start, end):
        row = self._range_for(start, end)
        entry = row[4]
        if entry['bytes'] > MAX_HEADER_BYTES + 8:
            raise refuse_pool_bulk_read(self._declared, 'header-coverage-must-be-bounded')
        from .staged_whole_file import read_staged_entry
        raw = read_staged_entry(self._resolver, Path(self._declared), entry,
                                label='safetensors-header', max_bytes=MAX_HEADER_BYTES + 8)
        return raw[start - entry['offset']:end - entry['offset']]


def staged_shard_opener(declared, pool_open):
    """The opener for ``declared``: ``pool_open`` itself unless the policy is active.

    Handing the caller back its own opener is what keeps the plain path
    identical -- same callable, same arguments, no wrapper and no extra
    syscall.

    Under the active allowed-tier policy there is no unwrapped path: every
    file opens through a reader whose header/keys/metadata come from the
    staged header, and whose payload reads refuse with
    ``readset-not-staged`` instead of serving pool bytes when the manifest
    does not declare them.
    """
    resolver = residency_resolver()
    if not policy_is_active():
        return pool_open
    path = os.fspath(declared)

    def opener(handed, **kwargs):
        return StagedShardReader(path, resolver, kwargs)

    return opener
