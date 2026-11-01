"""Golden byte tests for the tessera_calibration_cache digest consolidation (PQ #1512).

The fixture records the OLD outputs of every consolidated site, captured on the
pre-change tree (ratchet head bddf455bdba) by a PrismaBuild run of the golden
record script.  Each test recomputes the same row through the NEW code path and
asserts equality, so receipts, seals and identity chains recorded before the
change still verify.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

from prismaquant.digests import (
    DIRECT_ASCII_LAX,
    DIRECT_ASCII_STRICT,
    bytes_sha256hex,
    file_digest_sha256hex,
    hex_chain_sha256hex,
    indent2_json_file_bytes,
    text_sha256hex,
)
from prismaquant import tessera_calibration_cache as calibration_cache
from prismaquant.tessera_calibration_cache import (
    _json,
    _load_execution,
    fold_load_receipt,
    merge_load_execution,
    sha256,
)

FIXTURE = Path(__file__).parent / "fixtures" / "digest_calibration_cache_1512.json"


def _table():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return document["inputs"], document["old_rows"]


def test_the_pretty_file_profile_reproduces_the_old_capture_form():
    inputs, rows = _table()
    for index, value in enumerate(inputs["json_file_values"]):
        assert indent2_json_file_bytes(value).hex() == rows[f"A{index}"]["hex"]


def test_the_changed_writer_writes_those_bytes(tmp_path):
    inputs, rows = _table()
    for index, value in enumerate(inputs["json_file_values"]):
        target = tmp_path / f"capture-{index}.json"
        _json(target, value)
        assert target.read_bytes().hex() == rows[f"A{index}"]["hex"]


def test_strict_compact_text_reproduces_the_old_identity_and_policy_rows():
    inputs, rows = _table()
    for index, value in enumerate(inputs["identities"]):
        assert DIRECT_ASCII_STRICT.text(value) == rows[f"B{index}"]
    for index, value in enumerate(inputs["policies"]):
        assert DIRECT_ASCII_STRICT.text(value) == rows[f"C{index}"]


def test_the_streamed_descriptor_hash_reproduces_the_old_lax_encoder():
    inputs, rows = _table()
    descriptors = [
        dict(schema="prismaquant.capture_load_execution.v1", policy=inputs["policies"][0],
             capture_identity=inputs["identities"][0]),
        dict(schema="prismaquant.capture_load_execution.v1", policy=inputs["policies"][1],
             capture_identity=inputs["identities"][1]),
        dict(schema="prismaquant.capture_load_execution.v1",
             policy=dict(inputs["policies"][0], relaxed=float("nan")), capture_identity={}),
    ]
    for index, descriptor in enumerate(descriptors):
        assert DIRECT_ASCII_LAX.sha256_streamed(descriptor) == rows[f"D{index}"]
        recorded = rows[f"D{index}_as_text"]
        if isinstance(recorded, dict) and "error" in recorded:
            with pytest.raises(ValueError):
                DIRECT_ASCII_STRICT.text(descriptor)
        else:
            assert DIRECT_ASCII_STRICT.text(descriptor) == recorded


def test_streamed_and_materialized_profile_hashes_agree_for_accepted_values():
    value = {"b": 1, "a": [2, 3.5, None, True], "s": "naïve ✓"}
    assert DIRECT_ASCII_LAX.sha256_streamed(value) == DIRECT_ASCII_LAX.sha256(value)
    assert DIRECT_ASCII_STRICT.sha256_streamed(value) == DIRECT_ASCII_STRICT.sha256(value)


def test_raw_byte_digest_sites_reproduce_the_old_hex():
    inputs, rows = _table()
    for index, blob_hex in enumerate(inputs["blobs_hex"]):
        assert bytes_sha256hex(bytes.fromhex(blob_hex)) == rows[f"E{index}"]


def test_receipt_chains_recorded_before_the_change_still_verify():
    inputs, rows = _table()
    policy = inputs["policies"][0]
    execution = _load_execution(policy, None, identity_sha256=None)
    assert execution["identity_sha256"] == rows["F_identity"]
    assert execution["ordered_load_identities_sha256"] == rows["F_seed"]
    for index, identity in enumerate(inputs["chain_hex"][1:]):
        receipt = {"identity_sha256": identity, "source_read_bytes": 1024 * (index + 1),
                   "file_bytes": 4096 * (index + 1), "archive_storage_bytes": 512 * (index + 1)}
        fold_load_receipt(execution, receipt)
        assert execution["ordered_load_identities_sha256"] == rows[f"F_fold{index}"]
    total = _load_execution(policy, None, identity_sha256=None)
    partial = _load_execution(policy, None, identity_sha256=None)
    fold_load_receipt(partial, {"identity_sha256": inputs["chain_hex"][2], "source_read_bytes": 7,
                                "file_bytes": 9, "archive_storage_bytes": 1})
    merge_load_execution(total, partial)
    assert total["ordered_load_identities_sha256"] == rows["F_merge"]


def test_the_chain_owner_is_the_old_recipe():
    left = "ab" * 32
    right = "cd" * 32
    assert hex_chain_sha256hex(left, right) == text_sha256hex(left + right)
    assert hex_chain_sha256hex("", "") == bytes_sha256hex(b"")


def _write_source(directory, name, payload):
    target = Path(directory) / name
    target.write_bytes(payload)
    return target


def test_the_ordinary_path_routes_through_the_shared_owner(tmp_path, monkeypatch):
    """The full-capture fallback must call the shared owner once (PQ #2636)."""
    assert hasattr(calibration_cache, "file_digest_sha256hex")
    target = _write_source(tmp_path, "source.bin", bytes(range(256)) * 4)
    calls = []
    real = calibration_cache.file_digest_sha256hex

    def spy(handle):
        calls.append((handle.fileno(), handle.tell()))
        return real(handle)

    monkeypatch.setattr(calibration_cache, "file_digest_sha256hex", spy)
    assert sha256(target) == hashlib.file_digest(open(target, "rb"), "sha256").hexdigest()
    assert len(calls) == 1
    assert calls[0][1] == 0


def test_the_ordinary_path_opens_once_and_hashes_the_whole_file(tmp_path, monkeypatch):
    """One open and one owner call cover the file with no extra read."""
    target = _write_source(tmp_path, "source.bin", os.urandom(70000))
    expected = hashlib.file_digest(open(target, "rb"), "sha256").hexdigest()
    real_open = Path.open
    opens = []

    def counting_open(self, *args, **kwargs):
        if Path(self) == target:
            opens.append(Path(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    assert sha256(target) == expected
    assert opens == [target]


def test_the_routed_digest_matches_the_stdlib_stream(tmp_path):
    """The lane, the owner, and the stdlib agree on varied sizes."""
    assert sha256(_write_source(tmp_path, "empty.bin", b"")) == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    assert sha256(_write_source(tmp_path, "abc.bin", b"abc")) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    for index, size in enumerate((1, 255, 4096, 65537, 200000)):
        payload = bytes((index + offset) % 256 for offset in range(size))
        target = _write_source(tmp_path, f"source_{index}.bin", payload)
        expected = hashlib.file_digest(open(target, "rb"), "sha256").hexdigest()
        assert sha256(target) == expected
        with open(target, "rb") as handle:
            assert file_digest_sha256hex(handle) == expected


class _ReadTracer:
    """A handle proxy that records every read window and seek."""

    def __init__(self, handle):
        self._handle = handle
        self.windows = []
        self.seeks = []

    def read(self, size=-1):
        start = self._handle.tell()
        data = self._handle.read(size)
        self.windows.append((start, start + len(data)))
        return data

    def readinto(self, buffer):
        start = self._handle.tell()
        count = self._handle.readinto(buffer)
        self.windows.append((start, start + count))
        return count

    def seek(self, *args, **kwargs):
        self.seeks.append(args)
        return self._handle.seek(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._handle, name)


def test_the_owner_reads_one_stream_from_the_position_to_eof(tmp_path):
    """The owner keeps a single ordered stream and hashes only the tail."""
    payload = bytes(offset % 256 for offset in range(300000))
    target = _write_source(tmp_path, "source.bin", payload)
    with open(target, "rb") as handle:
        handle.seek(100)
        tracer = _ReadTracer(handle)
        assert file_digest_sha256hex(tracer) == hashlib.sha256(payload[100:]).hexdigest()
        assert tracer.seeks == []
        assert tracer.windows
        assert tracer.windows[0][0] == 100
        for first, second in zip(tracer.windows, tracer.windows[1:]):
            assert second[0] == first[1]
        assert tracer.windows[-1][1] == len(payload)


def test_the_guarded_alias_reads_the_owned_descriptor(tmp_path):
    """A descriptor alias hashes its object, never the replaced pathname."""
    target = _write_source(tmp_path, "source.bin", b"original-content")
    expected = hashlib.sha256(b"original-content").hexdigest()
    descriptor = os.open(target, os.O_RDONLY)
    try:
        assert sha256(target, file_descriptor=descriptor) == expected
    finally:
        os.close(descriptor)


def test_the_guarded_alias_refuses_a_foreign_descriptor(tmp_path):
    """Bytes from another object never mix with the named identity."""
    first = _write_source(tmp_path, "first.bin", b"first-content")
    second = _write_source(tmp_path, "second.bin", b"second-content")
    descriptor = os.open(second, os.O_RDONLY)
    try:
        with pytest.raises(RuntimeError, match="source changed"):
            sha256(first, file_descriptor=descriptor)
    finally:
        os.close(descriptor)


def test_the_guarded_reader_refuses_a_mid_hash_mutation(tmp_path):
    """A write during the hash breaks the stat fence."""
    target = _write_source(tmp_path, "source.bin", b"stable-content")
    mutated = False

    def mutate(tag):
        nonlocal mutated
        if tag.startswith("before_capture_hash") and not mutated:
            mutated = True
            with open(target, "ab") as handle:
                handle.write(b"drift")

    with pytest.raises(RuntimeError, match="source changed"):
        sha256(target, resource_check=mutate)


def test_the_guarded_reader_refuses_page_release_on_a_device():
    """Page release needs a regular file."""
    with pytest.raises(RuntimeError, match="page release requires a regular file"):
        sha256("/dev/null", release_read_pages=True)


def test_a_bad_descriptor_refuses_before_any_read(tmp_path):
    """A non-integer or negative descriptor never opens a path."""
    target = _write_source(tmp_path, "source.bin", b"content")
    with pytest.raises(ValueError, match="open file descriptor"):
        sha256(target, file_descriptor=-1)
    with pytest.raises(ValueError, match="open file descriptor"):
        sha256(target, file_descriptor="3")

