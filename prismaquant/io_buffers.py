"""Descriptor IO primitives: sealed buffers, bounded copies and no-follow directories."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import io
import mmap
import os
from pathlib import Path

# The memfd seal numbers are Linux UAPI and ABI-fixed since kernel 3.11
# (uapi/asm-generic/fcntl.h, uapi/linux/memfd.h). CPython defines the
# ``fcntl`` names only when its build headers had them, so a portable
# interpreter built against pre-glibc-2.27 headers lacks them and the module
# must not read them at import (PB be3dd1332259: the capture qualification set
# failed collection there with ``AttributeError: module 'fcntl' has no
# attribute 'F_SEAL_SEAL'``). The kernel stays the authority: the seal is
# attempted through fcntl and read back with ``F_GET_SEALS`` (tests/
# test_io_engine.py::test_seal_constants_resolve_without_cpython_build_time_names).
_F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
_SEALS = (
    getattr(fcntl, "F_SEAL_SEAL", 0x0001)
    | getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
    | getattr(fcntl, "F_SEAL_GROW", 0x0004)
    | getattr(fcntl, "F_SEAL_WRITE", 0x0008)
)
_MFD_CLOEXEC = getattr(os, "MFD_CLOEXEC", 0x0001)
_MFD_ALLOW_SEALING = getattr(os, "MFD_ALLOW_SEALING", 0x0002)

#: The runtime libc's ``memfd_create``, bound once with explicit ctypes
#: signatures (D5 review of b601b76b9b5: rebuilding the ``CDLL`` handle per
#: call re-dlopened and re-resolved on every sealed buffer). ``None`` until
#: the first portable call needs it.
_LIBC_MEMFD_CREATE = None


def _create_memfd(name: str, flags: int) -> int:
    """One anon_inode memfd, sealing-capable, this process owns.

    ``os.memfd_create`` exists only in interpreters whose build headers had
    the syscall; a portable build (pq-cpu312, PB ded8698fa4d6) lacks it and
    every stream read died on the missing attribute. The runtime libc on the
    same box still wraps the syscall (glibc 2.27+, musl 1.1.20+), so fall
    through to it. The MFD flag numbers are Linux UAPI, ABI-fixed.
    """
    global _LIBC_MEMFD_CREATE
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, flags)
    import ctypes

    if _LIBC_MEMFD_CREATE is None:
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            call = libc.memfd_create
        except AttributeError:
            raise OSError(
                f"this runtime libc exposes no memfd_create; sealed io "
                f"buffers cannot hold {name!r} here") from None
        call.argtypes = (ctypes.c_char_p, ctypes.c_int)
        call.restype = ctypes.c_int
        _LIBC_MEMFD_CREATE = call
    fd = _LIBC_MEMFD_CREATE(name.encode(), flags)
    if fd == -1:
        raise OSError(ctypes.get_errno(), f"memfd_create({name!r}) failed")
    return fd


def kernel_seal_bits(fd: int) -> int:
    """The kernel's actual seal bits on an open memfd (``F_GET_SEALS``)."""
    return fcntl.fcntl(fd, _F_GET_SEALS)


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
        self._fd = _create_memfd("pq-io", _MFD_CLOEXEC | _MFD_ALLOW_SEALING)
        try:
            os.ftruncate(self._fd, self.size)
            if self.size:
                self._map = mmap.mmap(
                    self._fd, self.size,
                    flags=mmap.MAP_SHARED | getattr(mmap, "MAP_POPULATE", 0))
        except BaseException:
            os.close(self._fd)
            raise

    def fill_bytes(self, payload: bytes) -> None:
        """Fill owned writable pages from an already-delivered exact byte payload."""
        if not isinstance(payload, bytes) or len(payload) != self.size:
            raise ValueError("sealed io payload must be bytes of the declared size")
        target = self._map
        if self._fd is None or (self.size and target is None):
            raise RuntimeError("sealed io buffer is not writable")
        if target is not None:
            target[:] = payload

    def fill(self, fd: int, *, source_offset: int = 0, require_eof: bool = True) -> bool:
        """Read ``fd`` from offset 0 into the buffer; whether it held exactly ``size`` bytes."""
        offset = 0
        if self.size:
            view = memoryview(self._map)
            try:
                while offset < self.size:
                    got = os.preadv(fd, [view[offset:]], source_offset + offset)
                    if got <= 0:
                        break
                    offset += got
            finally:
                view.release()
        return offset == self.size and (not require_eof or os.pread(fd, 1, source_offset + self.size) == b"")

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
        fcntl.fcntl(self._fd, _F_ADD_SEALS, _SEALS)
        return digest.hexdigest()

    def require_sealed(self) -> None:
        """Require actual kernel write/grow/shrink/seal protection, not a flag."""
        if self._fd is None or kernel_seal_bits(self._fd) & _SEALS != _SEALS:
            raise RuntimeError("original material lacks required kernel seals")

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
        # Before Linux 6.7, F_SEAL_WRITE rejects shared mappings of an O_RDWR
        # memfd even with PROT_READ. Private reads still use the memfd's pages;
        # without writes, they never trigger copy-on-write.
        view = mmap.mmap(self._fd, self.size, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ)
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




def copy_extent(source_fd, destination_fd, source_offset, length, *, chunk_bytes=8 << 20):
    """Offline bounded copy on the caller thread, hashing the bytes written."""
    from .digests import chunks_sha256hex
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError('copy chunk size must be a positive integer')

    def copied_chunks():
        copied = 0
        while copied < length:
            raw = os.pread(source_fd, min(chunk_bytes, length - copied), source_offset + copied)
            if not raw:
                raise RuntimeError('source extent truncated while copying')
            view = memoryview(raw)
            try:
                written = 0
                while written < len(view):
                    moved = os.write(destination_fd, view[written:])
                    if moved <= 0:
                        raise RuntimeError('object publication short write')
                    written += moved
            finally:
                view.release()
            copied += len(raw)
            yield raw

    return chunks_sha256hex(copied_chunks())



def open_nofollow_directory(filename, *, create=False, identity=None):
    """Hold a canonical directory without following any symlink component."""
    filename = os.fspath(filename)
    if not isinstance(filename, str) or not filename.startswith('/') or '\0' in filename or os.path.normpath(filename) != filename:
        raise ValueError('path must be normalized and absolute')
    directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in Path(filename).parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=directory)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        if identity is not None:
            current = os.fstat(directory)
            if (current.st_dev, current.st_ino) != tuple(identity):
                raise ValueError('writer directory identity changed')
        return directory
    except BaseException:
        os.close(directory)
        raise
