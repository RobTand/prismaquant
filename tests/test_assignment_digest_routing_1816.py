"""Pin assignment metadata bytes before delegating its final hash (#1816)."""

import hashlib
from collections.abc import Mapping
from typing import cast

import pytest

from prismaquant import footprint


@pytest.mark.parametrize(
    ("assignment", "encoded"),
    [
        ({}, b"{}"),
        ({"z": " fp8 ", "a": " bf16 "}, b'{"a":"BF16","z":"FP8_E4M3"}'),
        (
            {"layers.é.weight": " fp8 ", "a\n\x00": "bf16"},
            '{"a\\n\\u0000":"BF16","layers.é.weight":"FP8_E4M3"}'.encode(),
        ),
        (
            {"a": " int4_w4a16_g128 "},
            b'{"a":"INT4_W4A16_g128"}',
        ),
        ({1: "fp8", "1": "bf16"}, b'{"1":"BF16"}'),
        ({"1": "bf16", 1: "fp8"}, b'{"1":"FP8_E4M3"}'),
        ({False: None, "empty": "  "}, b'{"False":"NONE","empty":""}'),
        ({"a": "unknown-format"}, b'{"a":"UNKNOWN-FORMAT"}'),
    ],
)
def test_assignment_digest_preserves_literal_bytes(assignment, encoded):
    expected = hashlib.sha256(encoded).hexdigest()
    actual = footprint.assignment_serialization_sha256(assignment)
    assert actual == expected
    assert len(actual) == 64


def test_assignment_digest_preserves_utf8_refusal():
    with pytest.raises(UnicodeEncodeError) as refused:
        footprint.assignment_serialization_sha256({"\ud800": "fp8"})
    assert refused.value.encoding == "utf-8"
    assert refused.value.reason == "surrogates not allowed"
    assert refused.value.__cause__ is None


def test_assignment_digest_preserves_input_attribute_refusal():
    with pytest.raises(AttributeError) as refused:
        footprint.assignment_serialization_sha256(cast(Mapping[str, str], None))
    assert str(refused.value) == "'NoneType' object has no attribute 'items'"
    assert refused.value.__cause__ is None


def test_assignment_digest_preserves_registry_refusal(monkeypatch):
    calls = []

    def refuse(name):
        calls.append(name)
        raise ValueError("registry sentinel")

    monkeypatch.setattr(footprint.fr, "canonical_format_name", refuse)
    with pytest.raises(ValueError, match="^registry sentinel$") as refused:
        footprint.assignment_serialization_sha256({"z": " fp8 ", "a": "bf16"})
    assert calls == ["FP8"]
    assert refused.value.__cause__ is None


def test_assignment_digest_routes_only_final_serialized_bytes(monkeypatch):
    calls = []
    encoded = b'{"a":"BF16","z":"FP8_E4M3"}'

    def record(raw):
        calls.append(raw)
        assert isinstance(raw, bytes)
        return "b" * 64

    monkeypatch.setattr(footprint, "bytes_sha256hex", record)
    assert footprint.assignment_serialization_sha256(
        {"z": " fp8 ", "a": "bf16"},
    ) == "b" * 64
    assert calls == [encoded]
