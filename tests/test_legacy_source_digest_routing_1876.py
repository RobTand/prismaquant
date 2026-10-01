"""Pin legacy JSON source bytes and refusal ordering before owner consolidation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prismaquant import artifact_collection_legacy as legacy
from prismaquant.artifact_collection import ArtifactCollectionError

_SCHEMA = "legacy.test.source.v1"
_VALID = [
    b"{}",
    b'{"a":1,"z":2}',
    b'{ "z": 2, "a": 1 }\r\n',
    b'{"text":"\xc3\xa9\xe9\x9b\xaa","escaped":"\\u0000"}\n',
    b'{"text":"\\u00e9\\u96ea","escaped":"\\u0000"}',
    b'{"nested":{"rows":[null,true,false,-0.0,1.25]}}\n\n',
]


@pytest.mark.parametrize("raw", _VALID)
def test_source_identity_is_exact_acquired_bytes(tmp_path, monkeypatch, raw):
    path = tmp_path / "source.json"
    path.write_bytes(raw)
    reads = []
    original = Path.read_bytes

    def read_once(source):
        reads.append(source)
        return original(source)

    monkeypatch.setattr(Path, "read_bytes", read_once)
    value, reference, digest = legacy._load_json_source(path, logical_schema=_SCHEMA)
    expected = hashlib.sha256(raw).hexdigest()
    assert value == json.loads(raw.decode("utf-8"))
    assert reference == {
        "schema": "prismaquant.artifact_collection.reference.v1",
        "subject_schema": _SCHEMA,
        "subject_id": expected,
        "content": {"sha256": expected, "size_bytes": len(raw)},
    }
    assert digest == expected
    assert len(digest) == 64
    assert reads == [path]
    assert path.read_bytes() == raw


def test_json_spelling_is_not_canonicalized(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(_VALID[1])
    second.write_bytes(_VALID[2])
    a, _, a_sha = legacy._load_json_source(first, logical_schema=_SCHEMA)
    b, _, b_sha = legacy._load_json_source(second, logical_schema=_SCHEMA)
    assert a == b
    assert a_sha != b_sha


@pytest.mark.parametrize("raw", [b"[]", b'"text"', b"1", b"null", b"true"])
def test_top_level_refusal_precedes_hashing(tmp_path, monkeypatch, raw):
    path = tmp_path / "wrong.json"
    path.write_bytes(raw)

    def forbidden(_):
        pytest.fail("invalid source reached byte hashing")

    monkeypatch.setattr(legacy, "bytes_sha256hex", forbidden, raising=False)
    with pytest.raises(ArtifactCollectionError) as refused:
        legacy._load_json_source(path, logical_schema=_SCHEMA)
    assert str(refused.value) == f"{path}: expected a top-level object"
    assert refused.value.__cause__ is None


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"a":1,"a":2}', "JSON: duplicate member 'a'"),
        (b'{"a":NaN}', "JSON: non-finite value NaN"),
        (b'{"a":Infinity}', "JSON: non-finite value Infinity"),
        (b'{"a":-Infinity}', "JSON: non-finite value -Infinity"),
    ],
)
def test_strict_json_refusals_remain_unwrapped(tmp_path, monkeypatch, raw, message):
    path = tmp_path / "strict.json"
    path.write_bytes(raw)

    def forbidden(_):
        pytest.fail("strict JSON refusal reached byte hashing")

    monkeypatch.setattr(legacy, "bytes_sha256hex", forbidden, raising=False)
    with pytest.raises(ArtifactCollectionError) as refused:
        legacy._load_json_source(path, logical_schema=_SCHEMA)
    assert str(refused.value) == message
    assert refused.value.__cause__ is None


@pytest.mark.parametrize(
    ("raw", "cause_type"),
    [(b"\xff", UnicodeDecodeError), (b'{"a":', json.JSONDecodeError), (b"\x00", json.JSONDecodeError)],
)
def test_decode_errors_keep_outer_message_and_cause(tmp_path, monkeypatch, raw, cause_type):
    path = tmp_path / "broken.json"
    path.write_bytes(raw)

    def forbidden(_):
        pytest.fail("decode refusal reached byte hashing")

    monkeypatch.setattr(legacy, "bytes_sha256hex", forbidden, raising=False)
    with pytest.raises(ArtifactCollectionError) as refused:
        legacy._load_json_source(path, logical_schema=_SCHEMA)
    assert str(refused.value) == f"unreadable JSON file: {path}"
    assert isinstance(refused.value.__cause__, cause_type)


def test_io_error_keeps_cause_identity(tmp_path, monkeypatch):
    path = tmp_path / "refused.json"
    cause = PermissionError("read refused")

    def read_refused(source):
        assert source == path
        raise cause

    def forbidden(_):
        pytest.fail("IO refusal reached byte hashing")

    monkeypatch.setattr(Path, "read_bytes", read_refused)
    monkeypatch.setattr(legacy, "bytes_sha256hex", forbidden, raising=False)
    with pytest.raises(ArtifactCollectionError) as refused:
        legacy._load_json_source(path, logical_schema=_SCHEMA)
    assert str(refused.value) == f"unreadable JSON file: {path}"
    assert refused.value.__cause__ is cause


def test_parser_error_keeps_identity(tmp_path, monkeypatch):
    path = tmp_path / "parser.json"
    path.write_bytes(b"{}")
    cause = ArtifactCollectionError("sentinel strict refusal")

    def parse_refused(_):
        raise cause

    monkeypatch.setattr(legacy, "_strict_json", parse_refused)
    with pytest.raises(ArtifactCollectionError) as refused:
        legacy._load_json_source(path, logical_schema=_SCHEMA)
    assert refused.value is cause


def test_source_uses_existing_byte_owner_after_decode(tmp_path, monkeypatch):
    path = tmp_path / "route.json"
    raw = _VALID[2]
    path.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    events = []
    parse = legacy._strict_json
    reference = legacy.make_reference

    def parsed(text):
        events.append(("parse", text))
        return parse(text)

    def hashed(encoded):
        events.append(("hash", encoded))
        return expected

    def referenced(**kwargs):
        events.append(("reference", kwargs))
        return reference(**kwargs)

    monkeypatch.setattr(legacy, "_strict_json", parsed)
    monkeypatch.setattr(legacy, "bytes_sha256hex", hashed)
    monkeypatch.setattr(legacy, "make_reference", referenced)
    value, result, digest = legacy._load_json_source(path, logical_schema=_SCHEMA)
    assert value == {"z": 2, "a": 1}
    assert result["subject_id"] == expected
    assert digest == expected
    assert events == [
        ("parse", raw.decode("utf-8")),
        ("hash", raw),
        (
            "reference",
            {
                "subject_schema": _SCHEMA,
                "subject_id": expected,
                "content_sha256": expected,
                "size_bytes": len(raw),
            },
        ),
    ]
