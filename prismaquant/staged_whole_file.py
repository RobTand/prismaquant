"""Read one whole declared file off the stage, never the pool.

The whole-file counterpart of the range readers: a small input a PB action
declared in its data manifest (a calibration draw, a sealed control slice, a
pickled cache index) is opened through the residency resolver's
lifetime-pinned window -- RAM first where offered and allowed, SSD otherwise
-- and read once into an owned buffer under sealed size and change fences.
Extracted from ``calibration_data`` (PQ #1010) so every head input a Stage B
quantum declares is served the way the calibration draw already was.
"""
from __future__ import annotations

import os
from pathlib import Path


def read_staged_whole_file(path: Path, expected_sha256: str, *,
                           label: str) -> bytes:
    """One whole declared file, staged-pinned under the active tier policy.

    The whole-file counterpart to the checkpoint shared-state staged read:
    the bound map entry (whole file at offset 0, digest-bound to the
    caller's independently pinned ``expected_sha256``) is opened through a
    lifetime-pinned window — RAM first where offered and allowed, honest
    SSD re-acquire otherwise, never the pool — and read once into an owned
    buffer with the same sealed size/change fences. The descriptor closes
    and the exact ref releases before the caller decodes: no raw fd or
    mapping escapes, and the returned bytes outlive the lease.
    """
    from .residency_map import residency_resolver
    from .staged_tier_policy import refuse_pool_bulk_read

    where = str(path)
    resolver = residency_resolver()
    if resolver is None:
        raise refuse_pool_bulk_read(where, "readset-not-staged")
    staged = resolver.staged_read(path, expected_sha256=expected_sha256)
    if staged is None:
        raise refuse_pool_bulk_read(where, "readset-not-staged")
    return read_staged_entry(resolver, path, staged, label=label)


def read_staged_entry(resolver, path: Path, staged: dict, *, label: str) -> bytes:
    """The whole of one resolved map entry, read under a lifetime-pinned window.

    ``staged`` is the resolver's answer for ``path``: a whole-file entry
    (``staged_read``) or a range entry (``staged_range``). The entry's bytes
    are read once into an owned buffer under the same size and change
    fences as :func:`read_staged_whole_file`, never from the pool. The
    caller verifies the bytes against the digest it requires.
    """
    from .staged_lease import LeaseRefused, acquire_entry_window
    from .staged_tier_policy import refuse_pool_bulk_read

    where = str(path)
    size = staged.get("bytes")
    if type(size) is not int or isinstance(size, bool) or size <= 0:
        raise refuse_pool_bulk_read(where, "readset-not-staged")
    window, key = acquire_entry_window(resolver, path, staged)
    with window:
        try:
            fd, serving = window.open(key)
        except LeaseRefused as refusal:
            resolver.record_fallback(path, str(refusal))
            raise
        tier = window.serving_tier or "stage"
        resolver.record_serving_tier(
            path, tier, pin_id=str(serving.get("pin_id") or ""),
            range_ref=str(serving.get("range_ref") or ""))
        # Sealed bounds before allocation: the held descriptor's size must
        # match the staged entry's, or no buffer is built.
        first = os.fstat(fd)
        if first.st_size != size:
            raise LeaseRefused(f"{label}-changed-under-pin",
                               kind="integrity")
        # One owned buffer, filled in place: the caller's digest hashes
        # these same bytes and the tensor below decodes from them, so one
        # staged read serves verification and decode alike.
        raw = bytearray(size)
        view = memoryview(raw)
        try:
            remaining = size
            offset = 0
            while remaining > 0:
                try:
                    moved = os.preadv(fd, [view[offset:offset + remaining]], offset)
                except OSError as exc:
                    raise LeaseRefused(
                        f"{label}-unreadable: {exc.strerror}",
                        kind="availability") from None
                if moved <= 0:
                    break
                offset += moved
                remaining -= moved
            if remaining:
                raise LeaseRefused(f"{label}-truncated", kind="integrity")
            if os.pread(fd, 1, size):
                raise LeaseRefused(f"{label}-grew-during-read",
                                   kind="integrity")
            last = os.fstat(fd)
            if (last.st_ino, last.st_size, last.st_mtime_ns) != (
                    first.st_ino, first.st_size, first.st_mtime_ns):
                raise LeaseRefused(f"{label}-changed-under-pin",
                                   kind="integrity")
        finally:
            view.release()
    if tier == "ram":
        resolver.record_ram_read(path, len(raw))
    else:
        resolver.record_stage_read(path, len(raw))
    return bytes(raw)
