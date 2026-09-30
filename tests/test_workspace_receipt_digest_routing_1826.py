"""Preserve the CPU receipt writer's byte and publication profile (PQ #1826)."""

from __future__ import annotations

import errno
import hashlib
from pathlib import Path

import pytest

from prismaquant import cost_stage_checkpoint, stage_b_workspace_profile


class _PrintableReceipt1826:
    def __str__(self) -> str:
        return "report é"


@pytest.mark.parametrize(
    ("value", "raw"),
    [
        ({}, b"{}\n"),
        ({"z": 2, "a": 1}, b'{\n  "a": 1,\n  "z": 2\n}\n'),
        ({"é": "雪\n\x00"}, b'{\n  "\\u00e9": "\\u96ea\\n\\u0000"\n}\n'),
        (
            {"z": [True, None, 1.25, -0.0], "a": {"b": 2, "a": 1}},
            b'{\n  "a": {\n    "a": 1,\n    "b": 2\n  },\n'
            b'  "z": [\n    true,\n    null,\n    1.25,\n    -0.0\n  ]\n}\n',
        ),
        (
            {"value": _PrintableReceipt1826()},
            b'{\n  "value": "report \\u00e9"\n}\n',
        ),
        ([1, True, None], b"[\n  1,\n  true,\n  null\n]\n"),
        (None, b"null\n"),
        ("\ud800", b'"\\ud800"\n'),
    ],
)
def test_receipt_bytes_and_full_digest_match_independent_literals(tmp_path, value, raw):
    path = tmp_path / "nested" / "receipt.json"
    digest = stage_b_workspace_profile.write_profile(str(path), value)
    assert path.read_bytes() == raw
    assert digest == hashlib.sha256(raw).hexdigest()
    assert len(digest) == 64


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_refuse_before_directory_or_publication(tmp_path, monkeypatch, value):
    published = []
    monkeypatch.setattr(cost_stage_checkpoint, "atomic_write_bytes", lambda *args: published.append(args))
    path = tmp_path / "not-created" / "receipt.json"
    with pytest.raises(ValueError) as refused:
        stage_b_workspace_profile.write_profile(path, {"value": value})
    assert str(refused.value).startswith("Out of range float values are not JSON compliant")
    assert refused.value.__cause__ is None
    assert not path.parent.exists()
    assert published == []


def test_mixed_key_sorting_refuses_before_publication(tmp_path, monkeypatch):
    published = []
    monkeypatch.setattr(cost_stage_checkpoint, "atomic_write_bytes", lambda *args: published.append(args))
    path = tmp_path / "not-created" / "receipt.json"
    with pytest.raises(TypeError) as refused:
        stage_b_workspace_profile.write_profile(path, {1: "integer", "a": "text"})
    assert str(refused.value) == "'<' not supported between instances of 'str' and 'int'"
    assert refused.value.__cause__ is None
    assert not path.parent.exists()
    assert published == []


def test_default_string_failure_keeps_its_exception_and_cause(tmp_path):
    cause = LookupError("conversion cause")
    failure = RuntimeError("receipt conversion refused")
    failure.__cause__ = cause

    class RefusingReceipt1826:
        def __str__(self) -> str:
            raise failure

    path = tmp_path / "not-created" / "receipt.json"
    with pytest.raises(RuntimeError) as refused:
        stage_b_workspace_profile.write_profile(path, {"value": RefusingReceipt1826()})
    assert refused.value is failure
    assert refused.value.__cause__ is cause
    assert not path.parent.exists()


def test_atomic_publication_failure_precedes_hashing(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "receipt.json"
    raw = b'{\n  "a": 1\n}\n'
    calls = []
    failure = OSError("receipt publication refused")

    def refuse_publication(destination, payload):
        assert path.parent.is_dir()
        calls.append((destination, payload))
        raise failure

    def refuse_hash(_raw):
        pytest.fail("publication failed before the receipt may be hashed")

    monkeypatch.setattr(cost_stage_checkpoint, "atomic_write_bytes", refuse_publication)
    monkeypatch.setattr(stage_b_workspace_profile, "bytes_sha256hex", refuse_hash, raising=False)
    with pytest.raises(OSError) as refused:
        stage_b_workspace_profile.write_profile(path, {"a": 1})
    assert refused.value is failure
    assert refused.value.__cause__ is None
    assert calls == [(path, raw)]
    assert not path.exists()


def test_parent_file_refuses_before_atomic_publication(tmp_path, monkeypatch):
    parent = tmp_path / "blocked"
    parent.write_bytes(b"existing file")
    published = []
    monkeypatch.setattr(cost_stage_checkpoint, "atomic_write_bytes", lambda *args: published.append(args))
    with pytest.raises(FileExistsError) as refused:
        stage_b_workspace_profile.write_profile(parent / "receipt.json", {})
    assert refused.value.errno == errno.EEXIST
    assert parent.read_bytes() == b"existing file"
    assert published == []


def test_only_final_published_bytes_route_once_to_the_shared_owner(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "receipt.json"
    raw = b'{\n  "a": 1,\n  "z": 2\n}\n'
    events = []
    sentinel = "ab" * 32

    def publish(destination, payload):
        assert isinstance(destination, Path)
        assert destination == path
        assert path.parent.is_dir()
        assert payload == raw
        events.append("publish")
        destination.write_bytes(payload)

    def hash_published(payload):
        assert type(payload) is bytes
        assert payload == raw
        assert path.read_bytes() == raw
        events.append("hash")
        return sentinel

    monkeypatch.setattr(cost_stage_checkpoint, "atomic_write_bytes", publish)
    monkeypatch.setattr(stage_b_workspace_profile, "bytes_sha256hex", hash_published)
    assert stage_b_workspace_profile.write_profile(path, {"z": 2, "a": 1}) == sentinel
    assert events == ["publish", "hash"]
