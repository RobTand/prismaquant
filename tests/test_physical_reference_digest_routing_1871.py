"""Unchanged physical-reference bytes and shared primitive routing (PQ #1871)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import prismaquant.artifact_collection as records
import prismaquant.cost_streaming as streamed

_SCHEMA = "fixture.physical.é.v1"


def _json_bytes(value):
    # Independent old serializer, not a production digest/profile oracle.
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _record(payload, *, locators=None):
    record = {"schema": _SCHEMA, "payload": payload,
              "payload_sha256": hashlib.sha256(_json_bytes(payload)).hexdigest()}
    if locators is not None:
        record["locators"] = locators
    return record


def _portable(record):
    return _json_bytes({key: record[key]
                        for key in ("schema", "payload", "payload_sha256")}) + b"\n"


def _owner(path, *, budget=1024):
    owner = streamed.StreamedBoundaryArtifacts({
        "schema": streamed.BOUNDARY_STORAGE_SCHEMA,
        "directory": str(path), "max_resident_bytes": 1280,
        "max_auxiliary_bytes": 1 << 20, "max_artifact_bytes": budget,
        "prefetch_batches": 2,
    })
    owner.bind({"fixture": "physical-reference"}, n_probes=1)
    return owner


@pytest.mark.parametrize("payload", [
    {"empty": [], "none": None},
    {"z": 2, "a": {"right": 1, "left": 0}},
    {"text": "é\x00😀\r\n"},
    {"bool": False, "number": -0.0, "fraction": 1.125},
    {"nested": [{}, [], {"é": ["last", "first"]}]},
])
def test_record_reference_keeps_exact_portable_bytes(payload, tmp_path):
    record = _record(payload)
    raw = _portable(record)
    path = tmp_path / "record.json"
    records.write_record(path, record)
    assert path.read_bytes() == raw
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert records.reference_for_record(record) == {
        "schema": "prismaquant.artifact_collection.reference.v1",
        "subject_schema": _SCHEMA, "subject_id": record["payload_sha256"],
        "content": {"sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw)},
    }


def test_locator_and_key_order_do_not_change_physical_reference():
    first = _record({"z": 2, "a": {"right": 1, "left": 0}},
                    locators={"a" * 64: ["/mnt/fixture/first.json"]})
    second = _record({"a": {"left": 0, "right": 1}, "z": 2},
                     locators={"a" * 64: ["hf://fixture/second.json"]})
    assert _portable(first) == _portable(second)
    assert records.reference_for_record(first) == records.reference_for_record(second)


def test_reference_is_in_memory_and_retains_both_verifications(monkeypatch):
    record = _record({"text": "é"})
    seen = []
    verify = records.verify_record

    def traced(value):
        seen.append(value)
        return verify(value)

    def no_open(*args, **kwargs):
        raise AssertionError("physical reference must not acquire a file")

    monkeypatch.setattr(records, "verify_record", traced)
    monkeypatch.setattr(Path, "open", no_open)
    result = records.reference_for_record(record)
    assert len(seen) == 2
    assert seen[0] is record
    assert result["content"] == {
        "sha256": hashlib.sha256(_portable(record)).hexdigest(),
        "size_bytes": len(_portable(record)),
    }


def test_bad_semantic_id_refuses_before_physical_hash(monkeypatch):
    record = _record({"value": 1})
    expected = record["payload_sha256"]
    record["payload_sha256"] = "0" * 64

    def forbidden(raw):
        raise AssertionError("unverified record reached physical digest")

    monkeypatch.setattr(records, "bytes_sha256hex", forbidden, raising=False)
    with pytest.raises(records.ArtifactCollectionError) as refused:
        records.reference_for_record(record)
    assert str(refused.value) == (
        f"record.payload_sha256: differs (stored={'0' * 64}, computed={expected})")
    assert refused.value.__cause__ is None


def test_record_reference_uses_existing_byte_owner(monkeypatch):
    record = _record({"text": "é\x00", "empty": []})
    raw = _portable(record)
    seen = []

    def digest(value):
        seen.append(value)
        return "b" * 64

    monkeypatch.setattr(records, "bytes_sha256hex", digest)
    reference = records.reference_for_record(record)
    assert seen == [raw]
    assert reference["content"] == {"sha256": "b" * 64, "size_bytes": len(raw)}
    assert reference["subject_id"] == record["payload_sha256"]


@pytest.mark.parametrize("files", [
    [("first.bin", b"\x00\xff\x80"), ("record.json", b"\xc3\xa9\n")],
    [(17, bytearray(b"\xff\x00")), ("view", memoryview(b"A0B1C2")[::2])],
    [("single", b"\r\n\x00")],
])
def test_produced_references_keep_order_and_coerced_bytes(files, tmp_path):
    expected = [(str(name), bytes(payload)) for name, payload in files]
    with _owner(tmp_path) as owner:
        directory = owner.directory
        assert directory is not None
        references = owner.write_produced_files(
            files, kind="host-fixture", boundary_index=3)
        assert [(ref.name, ref.file_bytes, ref.sha256, ref.path) for ref in references] == [
            (name, len(raw), hashlib.sha256(raw).hexdigest(), str(directory / name))
            for name, raw in expected]
        assert [(directory / name).read_bytes() for name, _ in expected] == [
            raw for _, raw in expected]
        assert owner.telemetry["live_artifact_bytes"] == sum(len(raw) for _, raw in expected)


@pytest.mark.parametrize("files", [
    [], [("same", b"x"), ("same", b"y")], [("../file", b"x")],
    [(".hidden", b"x")], [("file.tmp", b"x")], [("generation.json", b"x")],
    [("", b"x")], [("empty", b"")],
])
def test_invalid_produced_files_refuse_before_shared_hash(files, tmp_path, monkeypatch):
    def forbidden(raw):
        raise AssertionError("invalid produced files reached digest")

    monkeypatch.setattr(streamed, "bytes_sha256hex", forbidden, raising=False)
    with _owner(tmp_path) as owner:
        directory = owner.directory
        assert directory is not None
        with pytest.raises(ValueError) as refused:
            owner.write_produced_files(files, kind="host-fixture", boundary_index=0)
        expected = ("a produced file cannot be empty" if files == [("empty", b"")]
                    else "produced files need distinct bare names")
        assert str(refused.value) == expected
        assert sorted(path.name for path in directory.iterdir()) == ["generation.json"]


def test_nonrunning_owner_refuses_before_coercion(tmp_path):
    with _owner(tmp_path) as owner:
        owner._status = "paused"
        with pytest.raises(RuntimeError, match="^exact boundary generation is not running$"):
            owner.write_produced_files([(object(), None)], kind="fixture", boundary_index=0)
        owner._status = "running"


def test_produced_budget_refuses_before_any_write(tmp_path):
    with _owner(tmp_path, budget=3) as owner:
        directory = owner.directory
        assert directory is not None
        with pytest.raises(RuntimeError) as refused:
            owner.write_produced_files([("a", b"xx"), ("b", b"yy")],
                                       kind="fixture", boundary_index=0)
        assert str(refused.value) == (
            "exact boundary artifact budget exceeded: produced files need "
            "4 bytes, 3 remain of 3")
        assert sorted(path.name for path in directory.iterdir()) == ["generation.json"]
        assert owner.telemetry["live_artifact_bytes"] == 0


def test_readback_template_refuses_before_prewrite_or_write(tmp_path):
    with _owner(tmp_path) as owner:
        directory = owner.directory
        assert directory is not None
        previous = owner._produced, owner._produced_plan
        owner._produced = object()
        owner._produced_plan = {"write_only": False}
        try:
            with pytest.raises(RuntimeError) as refused:
                owner.write_produced_files([("a", b"x")], kind="fixture", boundary_index=0)
            assert str(refused.value) == (
                "produced files are never read back by the action that writes "
                "them, so they commit at their origin, which needs a "
                "write-only produced-output template (PrismaBuild #912)")
            assert sorted(path.name for path in directory.iterdir()) == ["generation.json"]
        finally:
            owner._produced, owner._produced_plan = previous


def test_produced_reference_uses_existing_byte_owner(tmp_path, monkeypatch):
    files = [("a", bytearray(b"\xff\x00")), ("b", memoryview(b"A0B1")[::2])]
    seen = []

    def digest(value):
        seen.append(value)
        return hashlib.sha256(value).hexdigest()

    monkeypatch.setattr(streamed, "bytes_sha256hex", digest)
    with _owner(tmp_path) as owner:
        references = owner.write_produced_files(files, kind="fixture", boundary_index=0)
        assert seen == [b"\xff\x00", b"AB"]
        assert [ref.sha256 for ref in references] == [
            hashlib.sha256(raw).hexdigest() for raw in seen]
