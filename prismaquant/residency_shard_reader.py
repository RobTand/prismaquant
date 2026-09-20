"""Read a source shard's tensors off PrismaBuild's stage tier, ranges included.

``layer_streaming._source_safe_open`` is the one seam every BF16 source-shard
read passes through, and until now it opened the declared pool path and nothing
else. The stage held the bytes and no reader asked for them (PQ #732).

A shard is not staged the way the two existing consumers' files are. PrismaBuild
stages the byte *ranges* a data manifest declares, and its mover writes each
range as a file of its own whose byte 0 is the range's first byte
(``<shard>.safetensors.pbrange/<offset>-<length>``; PrismaBuild
``tools/fleet/stage_move.py``). On the live GLM-5.3-Flash run 34 of 55 staged
shard entries are ranges, so a path rewrite serves at most the other 21: byte 0
of a range file is tensor payload, not safetensors' header-length prefix, and
``safe_open`` on it parses garbage or a header whose spans overrun the file.

So the redirect is per tensor, not per file:

* The **header** always comes from the declared file. It sits at offset 0 and
  is generally outside any staged extent, and it is what turns a tensor name
  into an absolute byte span.
* Each tensor's **payload** comes from the staged range that covers its span
  outright, when the map has one, and from the declared file otherwise. A span
  straddling two staged ranges is read from the pool: never partial bytes, and
  never a concatenation this reader has not fenced.
* With no map the reader is not built at all. ``staged_shard_opener`` hands the
  caller back its own ``safe_open``, so an unmapped run's behaviour and syscalls
  are what they were.

**What binds a staged read here, and what does not.** The fences are
structural: the entry has to fit inside the declared file, the staged copy has
to be a regular file of exactly the entry's length, and its stat signature has
to be unchanged across every read this reader makes from it. They are the
pre-open fences ``ResidencyResolver.staged_read`` applies, asked of a range.
They are **not** a digest. The two whole-file consumers
(``production_weight_cache._read_file_tensor``,
``tessera_joint_aura._read_wire_bytes``) hash what they read and hold it to the
map's ``sha256``; a shard read cannot, because it reads one tensor's span out
of a multi-gigabyte range and the map's digest covers the whole range. Hashing
the range on first touch would be a second full read of every staged extent
before its first tensor could be served. So on this path the map's own check is
the only check, which is the coverage decision ``docs/ARCHITECTURE.md`` §12 D43
names and this reader makes explicit rather than quiet.

**How a span is read.** One tensor's payload is one span, and until PQ #746 it
was one sequential ``preadv`` loop on one descriptor -- one stream, whatever the
link underneath could carry. It is now cut on the mount's ``rsize`` and read on
at most the mount's ``nconnect`` threads into disjoint slices of the tensor's own
buffer, so the bytes are what the sequential loop produced and the wait is not.
Both numbers come from the mount rather than from us (principle 2); a mount that
is not NFS, or one publishing neither, is read the way it was. The threads are
one process-wide pool, so the layer gather's own reader threads
(``layer_streaming.layer_read_threads``) and this split cannot multiply into
more in-flight reads than the client has transports.

That has one consequence worth stating where it is made: under
``source_authentication`` (``tessera_calibration_cache``'s
``_CaptureSourceSafeOpen``) the payload the caller receives may come from the
stage while the ``prismaquant.selected_source_authentication.v1`` receipt's
hash is of the declared pool file. The declared file's identity fences still
run; the bytes handed over are the stage's, admitted on the map.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import errno
import json
import os
import re
import stat
import threading

import torch

from .residency_map import residency_resolver
from .staged_tier_policy import (
    active_policy,
    policy_is_active,
    refuse_pool_bulk_read,
)
from .staged_lease import LeaseRefused, acquire_entry_window

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

#: ``/proc/self/mounts`` escapes space, tab, newline and backslash in a mount
#: point as a backslash and three octal digits. Decoded in one pass, so a mount
#: point holding a literal backslash cannot be decoded twice.
_OCTAL = re.compile(r"\\([0-7]{3})")

_READ_SHAPE_LOCK = threading.Lock()
_READ_SHAPE_CACHE: dict[str, tuple[int, int] | None] = {}
_CHUNK_POOL_LOCK = threading.Lock()
_CHUNK_POOL: ThreadPoolExecutor | None = None
_CHUNK_POOL_SIZE = 0


def _mount_options(path: str) -> tuple[str, str, str] | None:
    """``(mount point, fstype, options)`` of the mount ``path`` is on.

    The longest mount point that prefixes the path wins, and on a tie the one
    later in the table does, because that is how the kernel resolves it: mounts
    stack, and the last one on a point is the one a read reaches. The tie is
    not hypothetical -- ``/stage/prewarm`` is an autofs trigger with the NFS
    mount on top of it, and reading the autofs row's options instead of the
    NFS row's is a mount with no ``nconnect``, which is a reader that quietly
    stays serial. ``/proc/self/mounts`` escapes space, tab, newline and
    backslash in the mount point as octal, so they are decoded before the
    comparison rather than compared raw.
    """
    target = os.path.abspath(path)
    best: tuple[int, str, str, str] | None = None
    try:
        with open(MOUNTS_PATH) as handle:
            rows = handle.read().splitlines()
    except OSError:
        return None
    for row in rows:
        fields = row.split(" ")
        if len(fields) < 4:
            continue
        point = _OCTAL.sub(lambda m: chr(int(m.group(1), 8)), fields[1])
        if target == point or target.startswith(point.rstrip("/") + "/"):
            if best is None or len(point) >= best[0]:
                best = (len(point), point, fields[2], fields[3])
    if best is None:
        return None
    return best[1], best[2], best[3]


def _read_shape_for(mount: tuple[str, str, str]) -> tuple[int, int] | None:
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
    _, fstype, raw = mount
    if not fstype.startswith("nfs"):
        return None
    streams = chunk = 0
    for option in raw.split(","):
        name, _, value = option.partition("=")
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
    mount = _mount_options(path)
    if mount is None:
        return None
    with _READ_SHAPE_LOCK:
        if mount[0] in _READ_SHAPE_CACHE:
            return _READ_SHAPE_CACHE[mount[0]]
    shape = _read_shape_for(mount)
    with _READ_SHAPE_LOCK:
        _READ_SHAPE_CACHE[mount[0]] = shape
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


def _signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_shard_header(path: str) -> tuple[dict, int, int]:
    """The shard's header, its payload base, and the declared file's length.

    The 8-byte little-endian length prefix and JSON header ``safetensors``
    itself reads, under the bounds
    ``layer_streaming._advise_consumed_safetensors_pages`` applies to the same
    bytes. The parse is repeated there rather than shared because that function
    is on the install hot path and this change has to stay expressible as
    minimal source hunks for the joint run's closed source transition.
    """
    handle = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
    try:
        info = os.fstat(handle)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("source shard is not a regular file")
        raw = os.pread(handle, 8, 0)
        if len(raw) != 8:
            raise ValueError("source shard has no safetensors header length")
        header_bytes = int.from_bytes(raw, "little")
        if not 0 < header_bytes <= min(MAX_HEADER_BYTES, info.st_size - 8):
            raise ValueError("source shard header length is out of range")
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
    """

    def __init__(self, pool_open, path, declared, resolver, kwargs):
        self._declared = declared
        self._path = os.fspath(path)
        self._resolver = resolver
        self._device = kwargs.get("device")
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

    # -- the handle interface --------------------------------------------

    def __enter__(self):
        if self._handle is not None:
            self._handle.__enter__()
        return self

    def __exit__(self, *args):
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
        """Read the header once, on the first payload read, never before.

        An open that only asks for ``keys()``, ``metadata()`` or a slice pays
        nothing for this reader beyond the object.
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
        material and lifetime. Any refusal — never a pool read. Inactive
        policy keeps the legacy stage-only open order.
        """
        for row in self._bound:
            if row[0] <= start and end <= row[1]:
                return row
        strict = self._strict
        if self._resolver is None:
            if strict:
                raise refuse_pool_bulk_read(self._declared, "readset-not-staged")
            return None
        entry = self._resolver.staged_range(
            self._declared, start, end, declared_size=self._declared_size)
        if entry is None:
            if strict:
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
                   _signature(info), entry, "stage")
            self._bound.append(row)
            return row
        from .staged_lease import LeaseRefused, acquire_entry_window
        window, key = acquire_entry_window(
            self._resolver, self._declared, entry)
        try:
            fd, serving = window.open(key)
            info = os.fstat(fd)
            if info.st_size != entry["bytes"]:
                raise LeaseRefused("lease-open-size-changed", kind="integrity")
        except (OSError, LeaseRefused) as error:
            try:
                window.__exit__(None, None, None)
            except LeaseRefused:
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
            # Every range of one declared shard is staged under the same root,
            # so the mount's read shape is read once per reader, not per tensor.
            self._shape = _read_shape(entry["stage_path"])
        row = (entry["offset"], entry["offset"] + entry["bytes"], fd,
               _signature(info), entry, tier, window)
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
            raw = _read_span(fd, end - start, start - entry["offset"], self._shape)
            if _signature(os.fstat(fd)) != signature:
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


def staged_shard_opener(declared, pool_open):
    """The opener for ``declared``: ``pool_open`` itself unless the stage holds it.

    Handing the caller back its own opener is what keeps the unmapped path
    identical -- same callable, same arguments, no wrapper and no extra
    syscall. A file the map never names takes that path too, so wrapping is
    paid for only where it can pay off.

    Under the active allowed-tier policy there is no unwrapped path: a file
    the map never names (or no map at all) still opens through a reader
    whose header/keys/metadata come from the pool handle — bounded header
    bytes, allowed — but whose payload reads refuse with
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
