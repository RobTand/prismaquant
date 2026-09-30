"""Share prefix validation, not acquisition or refusal policy (PQ #1800)."""
from __future__ import annotations

import io
import json
import math
import os
import stat
import struct
from types import SimpleNamespace

import pytest

from prismaquant import residency_shard_reader as shard
from prismaquant import source_read_plan as source


class _PrefixAccepted(Exception):
    """Stop at the unchanged body read without allocating a large header."""


class _LengthStream(io.BytesIO):
    def __init__(self, raw):
        super().__init__(raw)
        self.reads = []

    def read(self, size: int | None = -1) -> bytes:
        self.reads.append(size)
        return super().read(size)


def _fd_fixture(monkeypatch, raw, *, size, body=None, regular=True):
    calls = []
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)

    def open_file(path, given_flags):
        calls.append(("open", path, given_flags))
        assert given_flags == flags
        return 47

    def file_stat(fd):
        calls.append(("fstat", fd))
        return SimpleNamespace(
            st_size=size, st_mode=(stat.S_IFREG if regular else stat.S_IFIFO) | 0o600)

    def read_at(fd, length, offset):
        calls.append(("pread", fd, length, offset))
        if offset == 0:
            return raw
        if body is None:
            raise _PrefixAccepted
        return body

    def close_file(fd):
        calls.append(("close", fd))

    # Replace only this module's OS surface, never pytest's global os functions.
    monkeypatch.setattr(shard, "os", SimpleNamespace(
        O_RDONLY=os.O_RDONLY, O_NONBLOCK=os.O_NONBLOCK,
        O_CLOEXEC=getattr(os, "O_CLOEXEC", 0), open=open_file,
        fstat=file_stat, pread=read_at, close=close_file))
    return calls


@pytest.mark.parametrize("raw", [b"x" * length for length in range(8)])
def test_short_prefixes_keep_distinct_refusals_and_read_counts(raw, monkeypatch):
    stream = _LengthStream(raw)
    with pytest.raises(ValueError) as refused:
        source.safetensors_header_length(stream, "source.safetensors", 64)
    assert str(refused.value) == "source.safetensors is too short to be a safetensors file"
    assert stream.reads == [8]
    assert stream.tell() == len(raw)

    calls = _fd_fixture(monkeypatch, raw, size=64)
    with pytest.raises(ValueError) as refused:
        shard._read_shard_header("source.safetensors")
    assert str(refused.value) == "source shard has no safetensors header length"
    assert calls[1:] == [("fstat", 47), ("pread", 47, 8, 0), ("close", 47)]


@pytest.mark.parametrize("length,size,bound,accepted", [
    (0, 64, 64, False),
    (1, 9, 1, True),
    (2, 10, 2, True),
    (2, 9, 64, False),
    (100_000_000, 100_000_008, 100_000_000, True),
    (100_000_001, 100_000_009, 100_000_000, False),
    (2**64 - 1, 2**64 + 7, 100_000_000, False),
    (256, 264, 256, True),
    (256, 264, 255, False),
    (1, 8, 64, False),
    (1, 7, 64, False),
])
def test_prefix_boundaries_keep_unsigned_little_endian_and_each_bound(
        length, size, bound, accepted, monkeypatch):
    raw = struct.pack("<Q", length)
    monkeypatch.setattr(source, "SAFETENSORS_HEADER_MAX_BYTES", bound)
    monkeypatch.setattr(shard, "MAX_HEADER_BYTES", bound)
    stream = _LengthStream(raw)
    calls = _fd_fixture(monkeypatch, raw, size=size)
    if accepted:
        assert source.safetensors_header_length(stream, "source.safetensors", size) == length
        with pytest.raises(_PrefixAccepted):
            shard._read_shard_header("source.safetensors")
        assert calls[1:] == [
            ("fstat", 47), ("pread", 47, 8, 0),
            ("pread", 47, length, 8), ("close", 47)]
    else:
        with pytest.raises(ValueError) as refused:
            source.safetensors_header_length(stream, "source.safetensors", size)
        assert str(refused.value) == "source.safetensors has an invalid safetensors header length"
        with pytest.raises(ValueError) as refused:
            shard._read_shard_header("source.safetensors")
        assert str(refused.value) == "source shard header length is out of range"
        assert calls[1:] == [("fstat", 47), ("pread", 47, 8, 0), ("close", 47)]
    assert stream.reads == [8]
    assert stream.tell() == 8


def test_stream_routes_only_the_eight_bytes_and_its_own_profile(monkeypatch):
    owner = source.safetensors_prefix_length
    seen = []

    def record(raw, size, **profile):
        seen.append((raw, size, profile))
        return owner(raw, size, **profile)

    monkeypatch.setattr(source, "safetensors_prefix_length", record)
    monkeypatch.setattr(source, "SAFETENSORS_HEADER_MAX_BYTES", 4)
    raw = struct.pack("<Q", 2)
    stream = _LengthStream(raw + b"{}trailing")
    assert source.safetensors_header_length(stream, "source.safetensors", 32) == 2
    assert seen == [(raw, 32, {
        "max_bytes": 4,
        "short_error": "source.safetensors is too short to be a safetensors file",
        "range_error": "source.safetensors has an invalid safetensors header length",
    })]
    assert stream.reads == [8]
    assert stream.tell() == 8


def test_fd_routes_only_the_eight_bytes_and_its_own_profile(monkeypatch):
    owner = source.safetensors_prefix_length
    seen = []

    def record(raw, size, **profile):
        seen.append((raw, size, profile))
        return owner(raw, size, **profile)

    monkeypatch.setattr(shard, "safetensors_prefix_length", record)
    monkeypatch.setattr(shard, "MAX_HEADER_BYTES", 3)
    raw = struct.pack("<Q", 2)
    calls = _fd_fixture(monkeypatch, raw, size=32)
    with pytest.raises(_PrefixAccepted):
        shard._read_shard_header("source.safetensors")
    assert seen == [(raw, 32, {
        "max_bytes": 3,
        "short_error": "source shard has no safetensors header length",
        "range_error": "source shard header length is out of range",
    })]
    assert calls[-2:] == [("pread", 47, 2, 8), ("close", 47)]


def test_fd_regular_file_fence_precedes_any_prefix_read(monkeypatch):
    calls = _fd_fixture(monkeypatch, b"", size=64, regular=False)
    with pytest.raises(ValueError) as refused:
        shard._read_shard_header("source.safetensors")
    assert str(refused.value) == "source shard is not a regular file"
    assert calls[1:] == [("fstat", 47), ("close", 47)]


@pytest.mark.parametrize("body,declared,error,message", [
    (b"{}", 3, ValueError, "source shard header is shorter than it declares"),
    (b"[]", 2, ValueError, "source shard header is not an object"),
    (b"not-json", 8, json.JSONDecodeError, None),
    (b"\xff", 1, UnicodeDecodeError, None),
])
def test_fd_body_policy_and_finally_close_stay_local(
        body, declared, error, message, monkeypatch):
    if message is None:
        with pytest.raises(error) as old_parser:
            json.loads(body)
        message = str(old_parser.value)
    calls = _fd_fixture(monkeypatch, struct.pack("<Q", declared),
                        size=declared + 11, body=body)
    with pytest.raises(error) as refused:
        shard._read_shard_header("source.safetensors")
    assert str(refused.value) == message
    assert refused.value.__cause__ is None
    assert calls[-2:] == [("pread", 47, declared, 8), ("close", 47)]


def test_fd_ordinary_json_policy_is_not_replaced_by_strict_shipcard(monkeypatch):
    body = b'{"x":1,"x":2,"nan":NaN}'
    calls = _fd_fixture(monkeypatch, struct.pack("<Q", len(body)),
                        size=len(body) + 11, body=body)
    header, base, size = shard._read_shard_header("source.safetensors")
    assert header["x"] == 2
    assert math.isnan(header["nan"])
    assert (base, size) == (8 + len(body), 11 + len(body))
    assert calls[-2:] == [("pread", 47, len(body), 8), ("close", 47)]


def test_staged_header_keeps_one_prefix_request_and_never_opens_path(tmp_path, monkeypatch):
    path = tmp_path / "source.safetensors"
    raw = struct.pack("<Q", 2) + b"{}payload"
    path.write_bytes(raw)
    reads = []

    def prefix(given_path, **request):
        reads.append((given_path, request))
        return raw

    def refuse_open(*args, **kwargs):
        pytest.fail("staged header unexpectedly opened the declared path")

    monkeypatch.setattr(source, "open", refuse_open, raising=False)
    assert source.read_safetensors_header(
        str(path), source_reads=SimpleNamespace(prefix=prefix)) == ({}, 10, len(raw))
    assert reads == [(str(path), {"nbytes": 8, "where": "safetensors header"})]
