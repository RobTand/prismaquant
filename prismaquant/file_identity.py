"""One stat identity for every file fence (PQ #1531, epic #1295).

A fence records a file's stat identity and later holds the file to it, so
reuse without a re-read is admitted only while the file is the same object
with the same bytes and times. Every fence in the tree reads that identity
here. A site that persists a subset of it projects the subset from this
tuple; it never builds its own from the ``stat_result``.
"""
from __future__ import annotations

import os


def file_stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    """``(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)`` of one stat."""
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)
