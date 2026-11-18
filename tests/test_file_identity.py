"""One stat identity for every file fence (PQ #1531, epic #1295).

The sealed catalog persists four stat keys; they are projected from the one
file identity and must stay byte-identical to what earlier catalogs recorded.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from prismaquant import file_identity as fi
from prismaquant import joint_catalog_extension as jce
from prismaquant.file_identity import file_stat_signature


def test_the_signature_is_the_full_stat_identity(tmp_path):
    path = tmp_path / "wire.bin"
    path.write_bytes(b"x" * 11)
    value = os.stat(path)
    assert file_stat_signature(value) == (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def test_the_catalog_fence_keeps_its_sealed_wire_format(tmp_path):
    path = tmp_path / "wire.bin"
    path.write_bytes(b"x" * 11)
    value = os.stat(path)
    # The dict #1519 sealed into catalogs, spelled as it was then.
    sealed = {"inode": value.st_ino, "bytes": value.st_size,
              "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}
    fence = jce.catalog_stat_fence(value)
    assert fence == sealed
    assert list(fence) == list(sealed)
    assert json.dumps(fence) == json.dumps(sealed)
    assert json.dumps(fence, sort_keys=True) == json.dumps(sealed, sort_keys=True)


def test_writer_capture_failures_aliases_and_durable_paths(tmp_path):
    writer_path = tmp_path / "writer.bin"
    alias_path = tmp_path / "alias.bin"
    vanished_path = tmp_path / "vanished.bin"
    other_path = tmp_path / "other.bin"
    directory_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        with open(writer_path, "wb") as writer, open(other_path, "wb") as other:
            writer.write(b"abcde")
            other.write(b"zz")
            os.link(writer_path, alias_path)
            os.link(writer_path, vanished_path)
            os.unlink(vanished_path)
            paths = [writer_path, alias_path, vanished_path, other_path]

            # A plain registry gets the writer, its alias and nothing else.
            plain = {}
            with fi.capture_writer_identity(writer, paths, plain, directory_fd=directory_fd):
                pass
            state = os.fstat(writer.fileno())
            assert set(plain) == {str(writer_path), str(alias_path)}
            for key, value in plain.items():
                assert isinstance(key, str)
                assert value[:3] == (state.st_dev, state.st_ino, 5)

            # The registry class uses the same context and a string key.
            registry = fi.WriterIdentities()
            with registry.capture(writer, paths, directory_fd=directory_fd):
                pass
            assert set(registry) == set(plain)
            registry.retain(tmp_path / "kept.bin", writer)
            assert registry[str(tmp_path / "kept.bin")][:3] == (state.st_dev, state.st_ino, 5)
            assert len(registry) == 3

            # A null registry skips capture, even for a closed writer.
            closed = open(tmp_path / "closed.bin", "wb")
            closed.close()
            with fi.capture_writer_identity(closed, paths, None, directory_fd=directory_fd):
                pass
            body = RuntimeError("null body failed")
            with pytest.raises(RuntimeError) as null_info:
                with fi.capture_writer_identity(closed, paths, None, directory_fd=directory_fd):
                    raise body
            assert null_info.value is body
            assert getattr(body, "__notes__", []) == []

            # A closed writer with no body error raises the capture error.
            kept = {}
            with pytest.raises(ValueError) as capture_info:
                with fi.capture_writer_identity(closed, paths, kept, directory_fd=directory_fd):
                    pass
            assert str(capture_info.value) == "flush of closed file"
            assert capture_info.value.__cause__ is None
            assert kept == {}

            # A closed writer with a body error adds one note to that error.
            body = RuntimeError("writer body failed")
            with pytest.raises(RuntimeError) as body_info:
                with fi.capture_writer_identity(closed, paths, kept, directory_fd=directory_fd):
                    raise body
            assert body_info.value is body
            assert str(body) == "writer body failed"
            assert body.__cause__ is None
            assert body.__notes__ == [
                "writer ownership capture also failed: ValueError('flush of closed file')"]
            assert kept == {}
    finally:
        os.close(directory_fd)

    for good in ("a.py", "pkg/mod.py", "pkg/sub/mod.py", "x.pyc/mod.py", "mod.PYC",
                 "pkg/mod.PYO", "mod.pyc.py"):
        assert fi.durable_package_path(Path(good)), good
    for bad in ("__pycache__/a.py", "pkg/__pycache__/a.cpython-312.pyc", "a.pyc",
                "pkg/sub/a.pyo", "pkg/a.pyc"):
        assert not fi.durable_package_path(Path(bad)), bad
