"""The IO engine's bounded synchronous extent copy primitive."""
from __future__ import annotations

import os

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


