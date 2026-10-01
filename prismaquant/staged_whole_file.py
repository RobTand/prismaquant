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


def _verified_source_metadata_bytes(path: Path, *, label: str) -> tuple[bytes, str]:
    """Return staged metadata and its verified declared digest under one lease."""
    from .staged_tier_policy import refuse_pool_bulk_read
    from .digests import bytes_sha256hex
    from .residency_map import residency_resolver

    resolver = residency_resolver()
    if resolver is None:
        raise refuse_pool_bulk_read(str(path), "metadata-readset-not-staged")
    staged = resolver.staged_read(path)
    if staged is None:
        raise refuse_pool_bulk_read(str(path), "metadata-readset-not-staged")
    expected = staged.get("sha256")
    if not isinstance(expected, str) or not expected:
        raise refuse_pool_bulk_read(str(path), "metadata-digest-not-declared")
    raw = read_staged_entry(resolver, path, staged, label=label)
    if bytes_sha256hex(raw) != expected:
        raise refuse_pool_bulk_read(str(path), "metadata-digest-mismatch")
    return raw, expected


def read_source_metadata_text(path: Path, *, label: str,
                              encoding: str | None = None) -> str:
    """Read metadata at its canonical path through the active read contract.

    With no tier policy, retain the legacy text decoder. Under a policy,
    the bound whole-file entry supplies the declared digest and the existing
    lease reader supplies its bytes; metadata never falls back to the pool.
    This does not authenticate an unbound caller or change source paths.
    """
    from .staged_tier_policy import active_policy

    path = Path(path)
    if active_policy() is None:
        return path.read_text(encoding=encoding)
    raw, _ = _verified_source_metadata_bytes(path, label=label)
    # TextIOWrapper preserves Path.read_text's universal-newline behavior.
    import io

    with io.TextIOWrapper(io.BytesIO(raw), encoding=encoding) as handle:
        return handle.read()


def read_source_metadata_sha256(path: Path, *, label: str,
                                block_size: int) -> str:
    """Hash raw metadata bytes through the active source read contract.

    An inactive policy retains the caller's legacy streaming block size.
    Active reads share the whole-file lease and digest verification with
    text metadata, but never decode or normalize those bytes.
    """
    from .staged_tier_policy import active_policy

    path = Path(path)
    if active_policy() is None:
        from .digests import file_sha256hex

        return file_sha256hex(path, block_size=block_size)
    _, digest = _verified_source_metadata_bytes(path, label=label)
    return digest


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
