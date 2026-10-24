# Original-source witnesses, extracted without transforming bodies.
from __future__ import annotations
import hashlib, json, os, sys, subprocess
from pathlib import Path
from typing import *
import dataclasses, io
class FrozenSnapshotError(RuntimeError):
    """The local tree could not be frozen or changed after it was frozen."""

def _pread_exact(fd: int, size: int, offset: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    cursor = offset
    while remaining:
        chunk = os.pread(fd, remaining, cursor)
        if not chunk:
            raise FrozenSnapshotError(
                f"file shortened while freezing at byte {cursor}"
            )
        chunks.append(chunk)
        cursor += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

class _VerifiedBlockReader(io.BufferedIOBase):
    """Seekable reader that exposes only content matching the frozen blocks."""

    def __init__(self, entry: _FrozenEntry):
        super().__init__()
        if entry.fd is None or entry.content is not None:
            raise ValueError("verified reader requires a file-backed entry")
        self._entry = entry
        self._position = 0
        self._cached_index = -1
        self._cached_block = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._entry.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = int(position)
        return self._position

    def _block(self, index: int) -> bytes:
        if index == self._cached_index:
            return self._cached_block
        fd = self._entry.fd
        if fd is None:
            raise FrozenSnapshotError(
                f"frozen descriptor closed for {self._entry.relative_path}"
            )
        offset = index * SNAPSHOT_BLOCK_BYTES
        expected_size = min(
            SNAPSHOT_BLOCK_BYTES,
            self._entry.size - offset,
        )
        try:
            data = _pread_exact(fd, expected_size, offset)
        except (OSError, FrozenSnapshotError) as exc:
            raise FrozenSnapshotError(
                f"frozen source became unreadable: "
                f"{self._entry.relative_path}: {exc}"
            ) from exc
        observed = hashlib.sha256(data).hexdigest()
        expected = self._entry.block_sha256[index]
        if observed != expected:
            raise FrozenSnapshotError(
                "frozen source mutated after verification: "
                f"{self._entry.relative_path} block {index} "
                f"({observed[:12]} != {expected[:12]})"
            )
        self._cached_index = index
        self._cached_block = data
        return data

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("read of closed file")
        if self._position >= self._entry.size:
            return b""
        remaining_file = self._entry.size - self._position
        # Hub's streaming transports use bounded reads. Keep an accidental
        # unbounded read bounded to one block instead of materializing a
        # 112-GB object in RAM, while honoring every positive-size request.
        target = (
            min(remaining_file, SNAPSHOT_BLOCK_BYTES)
            if size is None or size < 0
            else min(remaining_file, size)
        )
        chunks: list[bytes] = []
        remaining = target
        while remaining:
            index, inside = divmod(self._position, SNAPSHOT_BLOCK_BYTES)
            block = self._block(index)
            take = min(len(block) - inside, remaining)
            if take <= 0:
                raise FrozenSnapshotError(
                    f"invalid frozen block geometry: {self._entry.relative_path}"
                )
            chunks.append(block[inside:inside + take])
            self._position += take
            remaining -= take
        return b"".join(chunks)

@dataclasses.dataclass(frozen=True)
class _ScannedFile:
    content: bytes | None
    sha256: str
    sample: bytes
    block_sha256: tuple[str, ...]
    read_calls: int

def _scan_file(fd: int, size: int) -> _ScannedFile:
    """Hash one file once, whole and in fixed blocks.

    The only retained data are inline files small enough to live in the
    snapshot tree.  A per-tensor safetensors span hash used to be computed from
    the same bytes; it fed the content receipt whose only consumer was the
    strict RTX4090 FP8-CB publication replay, retired with the Gridbook lane on
    2026-09-02 (see archive/gridbook_lane_2026-09-02/).
    """

    whole = hashlib.sha256()
    blocks: list[str] = []
    sample = b""
    inline_chunks: list[bytes] | None = [] if size <= SNAPSHOT_INLINE_BYTES else None
    read_calls = 0

    for offset in range(0, size, SNAPSHOT_BLOCK_BYTES):
        length = min(SNAPSHOT_BLOCK_BYTES, size - offset)
        data = _pread_exact(fd, length, offset)
        read_calls += 1
        if offset == 0:
            sample = data[:512]
        whole.update(data)
        blocks.append(hashlib.sha256(data).hexdigest())
        if inline_chunks is not None:
            inline_chunks.append(data)

    content = b"".join(inline_chunks) if inline_chunks is not None else None
    return _ScannedFile(
        content=content,
        sha256=whole.hexdigest(),
        sample=sample,
        block_sha256=tuple(blocks),
        read_calls=read_calls,
    )

def _replay_frozen_digests(snapshot: _FrozenSnapshot) -> None:
    """Re-read every held descriptor and compare against the frozen digests.

    The Xet transport consumes the frozen link paths rather than the
    streaming verified reader, so nothing re-checks bytes DURING upload;
    replaying the digests across the same descriptors after the commit
    detects any same-inode mutation across the upload window.  (An unlink
    or path swap cannot reach these descriptors at all.)
    """
    for entry in snapshot.entries:
        if entry.fd is None:
            continue
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.pread(entry.fd, SNAPSHOT_BLOCK_BYTES, size)
            if not block:
                break
            digest.update(block)
            size += len(block)
        if size != entry.size or digest.hexdigest() != entry.sha256:
            raise FrozenSnapshotError(
                f"held bytes diverged during upload: {entry.relative_path}"
            )


def manifest_digest(manifest_rows):
    return hashlib.sha256(json.dumps(
                manifest_rows,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")).hexdigest()

def canonical_digest(canonical_bytes):
    return hashlib.sha256(canonical_bytes).hexdigest()

def quote_path(path):
    return json.dumps(str(path.resolve()))
