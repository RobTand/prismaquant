"""Pin the distinct bound-reader contracts before sharing their byte hash."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from prismaquant import joint_quanta_join as join
from prismaquant import stage_inputs as inputs

BUFFERS = [b"", b"\xc3\xa9\x00\xe9\x9b\xaa\r\n", bytes(range(256))]


@pytest.fixture(autouse=True)
def isolated_bound_reads(monkeypatch):
    monkeypatch.setattr(inputs, "_BOUND_BYTES", {})
    monkeypatch.setattr(inputs, "BOUND_READER", None)


def record(path, raw):
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.mark.parametrize("raw", BUFFERS)
def test_bound_local_bytes_are_hashed_once_and_memo_hits_return_the_same_object(
    tmp_path, monkeypatch, raw
):
    path = tmp_path / "owned.bin"
    reads = []
    comparisons = []
    monkeypatch.setattr(inputs, "bound_stat_fence", lambda _path: (1, 2, 3, len(raw)))

    def read(self):
        reads.append(self)
        return raw

    original_same = inputs.same

    def compare(actual, expected, label):
        comparisons.append((actual, expected, label))
        return original_same(actual, expected, label)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(inputs, "same", compare)
    bound = record(path, raw)
    assert inputs.read_bound(bound, "unit") is raw
    assert inputs.read_bound(bound, "unit") is raw
    assert reads == [path]
    assert comparisons == [(bound["sha256"], bound["sha256"], "unit: owned bytes")]
    assert inputs._BOUND_BYTES[(str(path), bound["sha256"])] == ((1, 2, 3, len(raw)), raw)


def test_fence_drift_reverifies_and_digest_refusal_does_not_adopt_new_bytes(
    tmp_path, monkeypatch
):
    path = tmp_path / "owned.bin"
    raw = b"old"
    bound = record(path, raw)
    current = {"fence": (1, 2, 3, 3), "raw": raw}
    reads = []
    monkeypatch.setattr(inputs, "bound_stat_fence", lambda _path: current["fence"])

    def read(self):
        reads.append(self)
        return current["raw"]

    monkeypatch.setattr(Path, "read_bytes", read)
    assert inputs.read_bound(bound, "unit") is raw
    cached = inputs._BOUND_BYTES[(str(path), bound["sha256"])]
    current.update(fence=(1, 9, 3, 3), raw=b"new")
    with pytest.raises(ValueError) as refused:
        inputs.read_bound(bound, "unit")
    assert str(refused.value) == "unit: owned bytes: identity mismatch"
    assert refused.value.__cause__ is None
    assert reads == [path, path]
    assert inputs._BOUND_BYTES[(str(path), bound["sha256"])] is cached


def test_stat_failure_does_not_stop_reading_but_never_memoizes(tmp_path, monkeypatch):
    path = tmp_path / "owned.bin"
    raw = b"unfenced"
    reads = []

    def unavailable(_path):
        raise FileNotFoundError("stat unavailable")

    def read(self):
        reads.append(self)
        return raw

    monkeypatch.setattr(inputs, "bound_stat_fence", unavailable)
    monkeypatch.setattr(Path, "read_bytes", read)
    bound = record(path, raw)
    assert inputs.read_bound(bound, "unit") is raw
    assert inputs.read_bound(bound, "unit") is raw
    assert reads == [path, path]
    assert inputs._BOUND_BYTES == {}


def test_stat_size_mismatch_never_memoizes_verified_bytes(tmp_path, monkeypatch):
    path = tmp_path / "owned.bin"
    raw = b"owned"
    reads = []
    monkeypatch.setattr(inputs, "bound_stat_fence", lambda _path: (1, 2, 3, len(raw) + 1))

    def read(self):
        reads.append(self)
        return raw

    monkeypatch.setattr(Path, "read_bytes", read)
    bound = record(path, raw)
    assert inputs.read_bound(bound, "unit") is raw
    assert inputs.read_bound(bound, "unit") is raw
    assert reads == [path, path]
    assert inputs._BOUND_BYTES == {}


def test_staged_owned_buffer_does_not_read_the_pool_path(tmp_path, monkeypatch):
    path = tmp_path / "owned.bin"
    raw = b"staged\x00bytes"
    bound = record(path, raw)
    calls = []
    monkeypatch.setattr(inputs, "bound_stat_fence", lambda _path: (1, 2, 3, len(raw)))

    def forbidden(_self):
        raise AssertionError("pool read forbidden")

    def staged(observed_path, expected, label):
        calls.append((observed_path, expected, label))
        return raw

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(inputs, "BOUND_READER", staged)
    assert inputs.read_bound(bound, "unit") is raw
    assert inputs.read_bound(bound, "unit") is raw
    assert calls == [(path, bound["sha256"], "unit")]


def test_staged_refusal_propagates_exact_exception_and_cause(tmp_path, monkeypatch):
    raw = b"owned"
    refused = RuntimeError("staged refusal")
    cause = OSError("lease unavailable")
    refused.__cause__ = cause
    monkeypatch.setattr(inputs, "bound_stat_fence", lambda _path: None)

    def staged(*_args):
        raise refused

    monkeypatch.setattr(inputs, "BOUND_READER", staged)
    with pytest.raises(RuntimeError) as caught:
        inputs.read_bound(record(tmp_path / "owned.bin", raw), "unit")
    assert caught.value is refused
    assert caught.value.__cause__ is cause
    assert inputs._BOUND_BYTES == {}


def test_local_read_refusal_is_not_wrapped(tmp_path, monkeypatch):
    refused = PermissionError("owned read denied")
    monkeypatch.setattr(inputs, "bound_stat_fence", lambda _path: None)

    def read(_self):
        raise refused

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(PermissionError) as caught:
        inputs.read_bound(record(tmp_path / "owned.bin", b"owned"), "unit")
    assert caught.value is refused
    assert caught.value.__cause__ is None
    assert inputs._BOUND_BYTES == {}


@pytest.mark.parametrize("invalid", [None, [], {}, {"path": "x"},
                                     {"path": "x", "sha256": "x", "extra": 1}])
def test_bound_shape_is_refused_before_stat_or_read(monkeypatch, invalid):
    def forbidden(_path):
        raise AssertionError("shape refusal must precede stat")

    monkeypatch.setattr(inputs, "bound_stat_fence", forbidden)
    with pytest.raises(ValueError) as refused:
        inputs.read_bound(invalid, "unit")
    assert str(refused.value) == "unit: bound path and SHA256 required"
    assert refused.value.__cause__ is None


@pytest.mark.parametrize("raw", BUFFERS)
@pytest.mark.parametrize("bound", [False, True])
def test_join_returns_the_same_single_read_buffer_and_full_digest(
    tmp_path, monkeypatch, raw, bound
):
    path = tmp_path / "owned.bin"
    expected = hashlib.sha256(raw).hexdigest()
    reads = []

    def read(self):
        reads.append(self)
        return raw

    monkeypatch.setattr(Path, "read_bytes", read)
    result, actual = join._read_checked(path, expected if bound else None, where="unit")
    assert result is raw
    assert actual == expected
    assert len(actual) == 64
    assert reads == [path]


def test_join_digest_refusal_keeps_the_exact_message(tmp_path, monkeypatch):
    path = tmp_path / "owned.bin"
    raw = b"owned"
    actual = hashlib.sha256(raw).hexdigest()
    expected = "f" * 64
    monkeypatch.setattr(Path, "read_bytes", lambda _self: raw)
    with pytest.raises(join.JoinRefused) as refused:
        join._read_checked(path, expected, where="unit")
    assert str(refused.value) == (
        f"unit: digest mismatch at {path}: expected {expected}, got {actual}")
    assert refused.value.__cause__ is None


@pytest.mark.parametrize("kind", [FileNotFoundError, PermissionError])
def test_join_read_refusal_keeps_the_message_and_native_cause(tmp_path, monkeypatch, kind):
    path = tmp_path / "owned.bin"
    cause = kind("owned read denied")

    def read(_self):
        raise cause

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(join.JoinRefused) as refused:
        join._read_checked(path, None, where="unit")
    assert str(refused.value) == f"unit: unreadable file at {path}: owned read denied"
    assert refused.value.__cause__ is cause


def test_bound_routes_only_the_acquired_bytes_before_comparison_and_memo(
    tmp_path, monkeypatch
):
    path = tmp_path / "owned.bin"
    raw = b"owned\x00\xff"
    bound = record(path, raw)
    events = []

    class Memo(dict):
        def __setitem__(self, key, value):
            events.append("memo")
            super().__setitem__(key, value)

    def fence(_path):
        events.append("fence")
        return (1, 2, 3, len(raw))

    def read(observed_path):
        assert observed_path == path
        events.append("read")
        return raw

    def owner(data):
        assert data is raw
        events.append("owner")
        return bound["sha256"]

    original_same = inputs.same

    def compare(actual, expected, label):
        events.append("compare")
        return original_same(actual, expected, label)

    monkeypatch.setattr(inputs, "bytes_sha256hex", owner)
    monkeypatch.setattr(inputs, "bound_stat_fence", fence)
    monkeypatch.setattr(inputs, "_BOUND_BYTES", Memo())
    monkeypatch.setattr(inputs, "same", compare)
    monkeypatch.setattr(Path, "read_bytes", read)
    assert inputs.read_bound(bound, "unit") is raw
    assert events == ["fence", "read", "owner", "compare", "memo"]
    events.clear()
    assert inputs.read_bound(bound, "unit") is raw
    assert events == ["fence"]


def test_join_routes_the_same_single_read_bytes_once(tmp_path, monkeypatch):
    path = tmp_path / "owned.bin"
    raw = b"owned\x00\xff"
    expected = hashlib.sha256(raw).hexdigest()
    events = []

    def read(observed_path):
        assert observed_path == path
        events.append("read")
        return raw

    def owner(data):
        assert data is raw
        events.append("owner")
        return expected

    monkeypatch.setattr(join, "bytes_sha256hex", owner)
    monkeypatch.setattr(Path, "read_bytes", read)
    result, actual = join._read_checked(path, expected, where="unit")
    assert result is raw
    assert actual == expected
    assert events == ["read", "owner"]
