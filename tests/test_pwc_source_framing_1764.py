"""The package-source hash delegates framing, not cache/tensor behavior (#1764)."""
from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import pytest

from prismaquant import digests
from prismaquant import production_weight_cache as cache


def _legacy(entries):
    digest = hashlib.sha256()
    for name, raw in sorted(entries, key=lambda row: Path(row[0])):
        encoded = name.encode("utf-8")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def _fixture(tmp_path, payload):
    root = tmp_path / "package"
    root.mkdir()
    files = {
        "__init__.py": b"# package\r\n",
        "kernel.cu": b"source\x00bytes\n",
        "model/profile.json": b'{"format":"BF16"}\r\n',
        "nested/é.bin": payload,
        "lattice.tbl": b"\xff\x00\x01",
    }
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    for name in ("__pycache__/ignored.py", "nested/cache.pyc", "ignored.pyo"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"interpreter-only bytes")
    return root, sorted(files.items(), key=lambda row: Path(row[0]))


def _spy(monkeypatch):
    batches = []

    class RecordingProfile:
        def __init__(self):
            self.owner = digests.LengthFramedSourceSha256()
            self.rows = []
            batches.append(self.rows)

        def update(self, name, raw):
            self.rows.append((name, raw))
            self.owner.update(name, raw)

        def hexdigest(self):
            return self.owner.hexdigest()

    monkeypatch.setattr(cache, "LengthFramedSourceSha256", RecordingProfile, raising=False)
    return batches


@pytest.mark.parametrize("default_root", (False, True))
@pytest.mark.parametrize("payload", (b"", b"\x00\xff\r\nUTF-8:\xc3\xa9"))
def test_package_source_routes_owner(tmp_path, monkeypatch, payload, default_root):
    root, rows = _fixture(tmp_path, payload)
    batches = _spy(monkeypatch)
    if default_root:
        monkeypatch.setattr(cache, "__file__", str(root / "__init__.py"))
        result = cache._production_cache_source_sha256()
    else:
        result = cache._production_cache_source_sha256(root)
    assert result == _legacy(rows)
    assert batches == [rows]


def test_package_source_preserves_followed_file_symlink(tmp_path, monkeypatch):
    root, rows = _fixture(tmp_path, b"binary")
    outside = tmp_path / "external.bin"
    outside.write_bytes(b"external target\x00\xff")
    (root / "alias.dat").symlink_to(outside)
    rows = sorted([*rows, ("alias.dat", outside.read_bytes())], key=lambda row: Path(row[0]))
    batches = _spy(monkeypatch)
    assert cache._production_cache_source_sha256(root) == _legacy(rows)
    assert batches == [rows]


@pytest.mark.parametrize("missing", (False, True))
def test_package_source_missing_or_bytecode_only_root_refuses(tmp_path, missing):
    root = tmp_path / "package"
    if not missing:
        root.mkdir()
        (root / "ignored.pyc").write_bytes(b"bytecode")
    message = "cannot read package root" if missing else "found no files under"
    with pytest.raises(RuntimeError, match=message):
        cache._production_cache_source_sha256(root)


def test_package_source_preserves_read_error_and_cause(tmp_path, monkeypatch):
    root, _ = _fixture(tmp_path, b"binary")
    read = Path.read_bytes

    def failing(path):
        if path == root / "lattice.tbl":
            raise OSError("synthetic read failure")
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", failing)
    with pytest.raises(RuntimeError, match="production source identity cannot read lattice.tbl") as error:
        cache._production_cache_source_sha256(root)
    assert isinstance(error.value.__cause__, OSError)
    assert str(error.value.__cause__) == "synthetic read failure"


def test_package_source_preserves_strict_utf8_name_refusal(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    (root / "bad_\udcff.py").write_bytes(b"source")
    with pytest.raises(UnicodeEncodeError):
        cache._production_cache_source_sha256(root)
