"""Current #2635 consumers, after 60060a69c0 removed the private file hasher."""
from __future__ import annotations

import builtins
import hashlib
import json
import os

import pytest

from prismaquant import digests, shipcard


def test_file_wrapper_uses_one_owner_stream(tmp_path, monkeypatch):
    assert shipcard.file_sha256hex is digests.file_sha256hex
    path = tmp_path / "payload"
    raw = b"x" * digests.FILE_BLOCK_BYTES + b"\x00"
    path.write_bytes(raw)
    real_open = builtins.open
    opens, reads, calls = [], [], []

    class Reader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def read(self, size):
            chunk = self.handle.read(size)
            reads.append((size, len(chunk)))
            return chunk

    def observe_open(name, *args, **kwargs):
        handle = real_open(name, *args, **kwargs)
        if os.fspath(name) == str(path):
            opens.append((args, kwargs))
            return Reader(handle)
        return handle

    def owner(name, **kwargs):
        calls.append((name, kwargs))
        return digests.file_sha256hex(name, **kwargs)

    monkeypatch.setattr(builtins, "open", observe_open)
    monkeypatch.setattr(shipcard, "file_sha256hex", owner)
    assert shipcard.file_sha256(path) == hashlib.sha256(raw).hexdigest()
    assert calls == [(path, {})]
    assert opens == [(("rb",), {})]
    assert reads == [(digests.FILE_BLOCK_BYTES, digests.FILE_BLOCK_BYTES),
                     (digests.FILE_BLOCK_BYTES, 1), (digests.FILE_BLOCK_BYTES, 0)]


@pytest.mark.parametrize("error", [OSError("read failed"), ValueError("owner refused")])
def test_file_wrapper_keeps_error_to_none(tmp_path, monkeypatch, error):
    calls = []

    def refuse(path):
        calls.append(path)
        raise error

    monkeypatch.setattr(shipcard, "file_sha256hex", refuse)
    path = tmp_path / "payload"
    assert shipcard.file_sha256(path) is None
    assert calls == [path]


def test_model_and_manifest_consumers_use_the_file_owner(tmp_path, monkeypatch):
    config = b'{"model_type":"tiny"}'
    auxiliary = "café\x00tokenizer".encode()
    (tmp_path / "config.json").write_bytes(config)
    (tmp_path / "tokenizer.json").write_bytes(auxiliary)
    calls = []

    def owner(path):
        calls.append(path.name)
        return digests.file_sha256hex(path)

    monkeypatch.setattr(shipcard, "file_sha256hex", owner)
    payload = {
        "config_sha": hashlib.sha256(config).hexdigest(),
        "weights": {},
        "auxiliary_files": {
            name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in (("config.json", config), ("tokenizer.json", auxiliary))
        },
    }
    expected = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()).hexdigest()
    assert shipcard.compute_model_sha(tmp_path) == expected
    assert calls == ["config.json", "tokenizer.json"]
    calls.clear()
    raw = b"finished weight bytes\x00\xff"
    (tmp_path / "weights.safetensors").write_bytes(raw)
    manifest = shipcard.build_weight_content_manifest(tmp_path)
    assert manifest == {
        "schema": shipcard.WEIGHT_CONTENT_MANIFEST_SCHEMA,
        "algorithm": "sha256",
        "files": {"weights.safetensors": {
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        }},
    }
    assert calls == ["weights.safetensors"]


@pytest.mark.parametrize("drift", [False, True], ids=["one-pass", "stat-drift"])
def test_retained_descriptor_verifier_keeps_its_read_contract(tmp_path, monkeypatch, drift):
    header = json.dumps({
        "b": {"dtype": "U8", "shape": [3], "data_offsets": [3, 6]},
        "a": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]},
    }).encode()
    raw = len(header).to_bytes(8, "little") + header + b"abc\x00\xffz"
    path = tmp_path / "weights.safetensors"
    path.write_bytes(raw)
    fd = os.open(path, os.O_RDONLY)
    initial = os.fstat(fd)
    real_read, real_fstat = os.read, os.fstat
    reads, fences = [], []

    def read(held_fd, size):
        assert held_fd == fd
        offset = os.lseek(fd, 0, os.SEEK_CUR)
        chunk = real_read(fd, min(size, 5))
        reads.append((offset, size, chunk))
        return chunk

    def fstat(held_fd):
        assert held_fd == fd
        fences.append(held_fd)
        if drift:
            os.utime(path, ns=(initial.st_atime_ns, initial.st_mtime_ns + 1_000_000))
        return real_fstat(held_fd)

    monkeypatch.setattr(shipcard.os, "read", read)
    monkeypatch.setattr(shipcard.os, "fstat", fstat)
    try:
        if drift:
            with pytest.raises(ValueError, match="file stat changed"):
                shipcard._verify_open_safetensors_fd(fd, name=path.name, initial_stat=initial)
        else:
            record, byte_count, read_count = shipcard._verify_open_safetensors_fd(
                fd, name=path.name, initial_stat=initial)
            assert record == {
                "stat": shipcard._content_stat(initial),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "tensor_sha256": {"a": hashlib.sha256(b"abc").hexdigest(),
                                  "b": hashlib.sha256(b"\x00\xffz").hexdigest()},
            }
            assert (byte_count, read_count) == (len(raw), len(reads))
        assert fences == [fd]
        assert b"".join(chunk for _, _, chunk in reads) == raw
        cursor = 0
        for offset, size, chunk in reads:
            assert offset == cursor
            assert 0 < len(chunk) <= size <= 1024 * 1024
            cursor += len(chunk)
        assert os.lseek(fd, 0, os.SEEK_CUR) == len(raw)
        assert real_fstat(fd).st_ino == initial.st_ino
    finally:
        os.close(fd)
