"""One stat identity for every file fence (PQ #1531, epic #1295).

A fence records a file's stat identity and later holds the file to it, so
reuse without a re-read is admitted only while the file is the same object
with the same bytes and times. Every fence in the tree reads that identity
here. A site that persists a subset of it projects the subset from this
tuple; it never builds its own from the ``stat_result``.
"""
from __future__ import annotations

import os
from pathlib import Path


def durable_package_path(relative: Path) -> bool:
    """Exclude only disposable interpreter bytecode from package identity."""
    return "__pycache__" not in relative.parts and relative.suffix not in {".pyc", ".pyo"}


def _production_cache_identity_paths(root: Path) -> list[Path]:
    """Every durable package file, in the production digest's original order."""
    return sorted(path for path in root.rglob("*")
                  if path.is_file() and durable_package_path(path.relative_to(root)))


def file_stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    """``(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)`` of one stat."""
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


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
