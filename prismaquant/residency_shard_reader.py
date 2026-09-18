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

That has one consequence worth stating where it is made: under
``source_authentication`` (``tessera_calibration_cache``'s
``_CaptureSourceSafeOpen``) the payload the caller receives may come from the
stage while the ``prismaquant.selected_source_authentication.v1`` receipt's
hash is of the declared pool file. The declared file's identity fences still
run; the bytes handed over are the stage's, admitted on the map.
"""
from __future__ import annotations

import errno
import json
import os
import stat

import torch

from .residency_map import residency_resolver

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


def _pread_exact(fd: int, count: int, offset: int) -> bytearray:
    """``count`` bytes at ``offset``, in as many reads as the kernel needs.

    A single ``pread`` is capped near 2 GiB on Linux and an NFS read can be
    short for its own reasons, so the loop is the contract rather than an
    optimization. The buffer is writable, which is what ``torch.frombuffer``
    wants, and it is the tensor's own storage afterwards.
    """
    buffer = bytearray(count)
    view = memoryview(buffer)
    done = 0
    while done < count:
        moved = os.preadv(fd, [view[done:]], offset + done)
        if moved <= 0:
            raise OSError(errno.EIO, "staged range ended before the tensor's bytes")
        done += moved
    return buffer


class StagedShardReader:
    """A ``safe_open`` handle whose payload comes off the stage when it can.

    Drop-in at the ``_source_safe_open`` seam, including through
    ``source_authentication.safe_open(opener, path, **kwargs)``, which calls the
    opener with ``/proc/self/fd/<fd>`` rather than the declared path. The
    declared path travels in the closure ``staged_shard_opener`` builds; the
    handed path is what is opened and what the header is read from, and the two
    are the same file.
    """

    def __init__(self, pool_open, path, declared, resolver, kwargs):
        self._declared = declared
        self._path = os.fspath(path)
        self._resolver = resolver
        self._device = kwargs.get("device")
        # First, so a ``safe_open`` that rejects the ``device`` kwarg raises
        # where it raises today and the callers' retry without it still works.
        self._handle = pool_open(path, **kwargs)
        self._header = None
        self._base = 0
        self._declared_size = 0
        self._parsed = False
        self._bound: list[tuple] = []

    # -- the handle interface --------------------------------------------

    def __enter__(self):
        self._handle.__enter__()
        return self

    def __exit__(self, *args):
        while self._bound:
            row = self._bound.pop()
            try:
                os.close(row[2])
            except OSError:
                pass
        return self._handle.__exit__(*args)

    def keys(self):
        return self._handle.keys()

    def metadata(self):
        return self._handle.metadata()

    def get_slice(self, name):
        """The pool's slice. A slice is a partial read this reader does not serve.

        Its callers (``streaming_model._estimate_layer_cache_bytes``) read
        shape and dtype, not payload, so nothing is accounted for it either.
        """
        return self._handle.get_slice(name)

    def get_tensor(self, name):
        """The tensor, off the stage when a staged range covers its whole span.

        ``bytes_from_pool`` counts what this reader falls back to, not every
        shard byte the run reads: a shard the map never names is opened by
        ``safe_open`` itself and no reader sees it. The stage-side count has no
        such gap, so read the two as "what the stage served" and "what this
        reader could not get from it", not as a partition of the run.
        """
        served = self._staged_tensor(name)
        if served is not None:
            return served
        tensor = self._handle.get_tensor(name)
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
            self._resolver.record_fallback(
                self._declared,
                "safetensors publishes no dtype table this reader can read")
            return
        try:
            self._header, self._base, self._declared_size = _read_shard_header(self._path)
        except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as error:
            self._header = None
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
        """
        for row in self._bound:
            if row[0] <= start and end <= row[1]:
                return row
        entry = self._resolver.staged_range(
            self._declared, start, end, declared_size=self._declared_size)
        if entry is None:
            return None
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
        row = (entry["offset"], entry["offset"] + entry["bytes"], fd,
               _signature(info), entry)
        self._bound.append(row)
        return row

    def _drop(self, row) -> None:
        if row in self._bound:
            self._bound.remove(row)
        try:
            os.close(row[2])
        except OSError:
            pass

    def _staged_tensor(self, name):
        span = self._span(name)
        if span is None:
            return None
        start, end, dtype, shape = span
        if end == start:
            # No bytes to serve; the pool handle's empty tensor is the answer.
            return None
        row = self._range_for(start, end)
        if row is None:
            return None
        fd, signature, entry = row[2], row[3], row[4]
        try:
            raw = _pread_exact(fd, end - start, start - entry["offset"])
            if _signature(os.fstat(fd)) != signature:
                raise ValueError("changed during its content read")
        except (OSError, ValueError) as error:
            self._drop(row)
            self._resolver.record_fallback(
                self._declared,
                f"staged range {getattr(error, 'strerror', None) or error}")
            return None
        tensor = torch.frombuffer(raw, dtype=torch.uint8).view(dtype).reshape(shape)
        if self._device is not None:
            tensor = tensor.to(self._device)
        self._resolver.record_stage_range_read(self._declared, end - start)
        return tensor


def staged_shard_opener(declared, pool_open):
    """The opener for ``declared``: ``pool_open`` itself unless the stage holds it.

    Handing the caller back its own opener is what keeps the unmapped path
    identical -- same callable, same arguments, no wrapper and no extra
    syscall. A file the map never names takes that path too, so wrapping is
    paid for only where it can pay off.
    """
    resolver = residency_resolver()
    if resolver is None or not resolver.stages(declared):
        return pool_open
    path = os.fspath(declared)

    def opener(handed, **kwargs):
        return StagedShardReader(pool_open, handed, path, resolver, kwargs)

    return opener
