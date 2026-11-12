"""Pin generic IO/map byte identities separately from their acquisition policies."""

import hashlib
from pathlib import Path

import pytest

from prismaquant import io_engine

BUFFERS = (b"", b"raw-\xc3\xa9\x00\r\n", bytes(range(256)))


@pytest.mark.parametrize("raw", BUFFERS)
def test_io_raw_receipt_keeps_its_one_read_and_declared_identity(tmp_path, monkeypatch, raw):
    declared = tmp_path / "declared.bin"
    declared.write_bytes(raw)
    opened = []
    original = Path.open

    def counted(path, *args, **kwargs):
        opened.append((path, args))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    timing = {}
    result, receipt, signature = io_engine.read_file(declared, len(raw), timing=timing)
    assert type(result) is bytes and result == raw
    expected = {
        "path": str(declared), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert receipt == expected
    assert len(receipt["sha256"]) == 64
    info = declared.lstat()
    assert signature == (
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
    )
    assert opened == [(declared, ("rb",))]
    assert timing.keys() == {"read_s"} and timing["read_s"] >= 0


@pytest.mark.parametrize("refusal", ("bound", "symlink"))
def test_io_pre_read_refusals_keep_their_exact_messages(tmp_path, monkeypatch, refusal):
    path = tmp_path / "declared.bin"
    path.write_bytes(b"abc")
    limit = 2
    message = "PWC file exceeds the explicit read buffer bound"
    if refusal == "symlink":
        link = tmp_path / "link.bin"
        link.symlink_to(path)
        path, limit = link, 3
        message = "PWC file receipt requires a regular file, not a symlink"

    def forbidden(*args, **kwargs):
        pytest.fail("a pre-read refusal opened payload bytes")

    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(RuntimeError) as refused:
        io_engine.read_file(path, limit)
    assert str(refused.value) == message and refused.value.__cause__ is None


def test_io_native_pool_open_failure_is_not_wrapped(tmp_path, monkeypatch):
    path = tmp_path / "declared.bin"
    path.write_bytes(b"abc")
    error = PermissionError("fixture open refused")

    def denied(*args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(PermissionError) as refused:
        io_engine.read_file(path, 3)
    assert refused.value is error and refused.value.__cause__ is None


def test_io_unsealed_buffer_routes_exactly_once_after_read(tmp_path, monkeypatch):
    path = tmp_path / "declared.bin"
    path.write_bytes(BUFFERS[1])
    seen = []
    opened = []
    original = Path.open

    def counted(p, *args, **kwargs):
        opened.append(p)
        return original(p, *args, **kwargs)

    def owner(raw):
        assert opened == [path]
        seen.append(raw)
        return hashlib.sha256(raw).hexdigest()

    monkeypatch.setattr(Path, "open", counted)
    monkeypatch.setattr(io_engine, "bytes_sha256hex", owner)
    raw, receipt, _signature = io_engine.read_file(path, len(BUFFERS[1]))
    assert seen == [BUFFERS[1]] and seen[0] is raw
    assert receipt["sha256"] == hashlib.sha256(BUFFERS[1]).hexdigest()
