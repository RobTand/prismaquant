"""Independent literal bytes and routing seams for PQ #1988 (Refs #1301)."""

import hashlib
from pathlib import PurePosixPath

import pytest

from prismaquant import digests, footprint


@pytest.mark.parametrize(("value", "raw"), [
    ({}, b"{}"),
    ({"z": "é😀", "a": "\n\x00\t\"\\"},
     b'{"a":"\\n\\u0000\\t\\\"\\\\","z":"\xc3\xa9\xf0\x9f\x98\x80"}'),
    ({"n": float("nan"), "p": float("inf"), "m": float("-inf")},
     b'{"m":-Infinity,"n":NaN,"p":Infinity}'),
    ({"z": -0.0, "a": [True, False, None, 1.0, 1e-7]},
     b'{"a":[true,false,null,1.0,1e-07],"z":-0.0}'),
    ({10: "x", 9: "y"}, b'{"9":"y","10":"x"}'),
    ({"tuple": (1, "é")}, b'{"tuple":[1,"\xc3\xa9"]}'),
])
def test_direct_utf8_lax_literal_bytes(value, raw):
    profile = digests.DIRECT_UTF8_LAX
    assert profile.text(value) == raw.decode("utf-8")
    assert profile.encoded(value) == raw
    assert profile.sha256(value) == hashlib.sha256(raw).hexdigest()
    assert profile.sha256_streamed(value) == hashlib.sha256(raw).hexdigest()


def test_direct_utf8_lax_options():
    profile = digests.DIRECT_UTF8_LAX
    assert profile.name == "direct-utf8-lax"
    assert profile.ensure_ascii is False
    assert profile.allow_nan is True
    assert profile.default is None
    assert profile.separators == (",", ":")
    assert profile.indent is None
    with pytest.raises(ValueError, match="Out of range float values"):
        digests.DIRECT_UTF8_STRICT.encoded(float("nan"))


@pytest.mark.parametrize(("value", "message"), [
    (PurePosixPath("/x"), "Object of type PurePosixPath is not JSON serializable"),
    (b"raw", "Object of type bytes is not JSON serializable"),
    ({(1, 2): 3}, "keys must be str, int, float, bool or None, not tuple"),
])
def test_direct_utf8_lax_native_type_refusal(value, message):
    with pytest.raises(TypeError) as refused:
        digests.DIRECT_UTF8_LAX.encoded(value)
    assert str(refused.value) == message
    assert refused.value.__cause__ is None


def test_direct_utf8_lax_cycle_refusal():
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError) as refused:
        digests.DIRECT_UTF8_LAX.encoded(cycle)
    assert str(refused.value) == "Circular reference detected"
    assert refused.value.__cause__ is None


@pytest.mark.parametrize("value", ["\ud800", {"\ud800": "x"}, {"x": "\udfff"}])
def test_direct_utf8_lax_strict_utf8_refusal(value):
    with pytest.raises(UnicodeEncodeError) as refused:
        digests.DIRECT_UTF8_LAX.encoded(value)
    assert refused.value.encoding == "utf-8"
    assert refused.value.reason == "surrogates not allowed"
    assert refused.value.__cause__ is None


def test_assignment_preserves_coercion_collision_and_call_order(monkeypatch):
    calls = []

    class Coerced:
        def __init__(self, label, text):
            self.label, self.text = label, text

        def __str__(self):
            calls.append(self.label)
            return self.text

    class Entries(dict):
        def items(self):
            calls.append("items")
            return {
                Coerced("name1", "é"): Coerced("fmt1", " fp8 "),
                Coerced("name2", "é"): Coerced("fmt2", " bf16 "),
            }.items()

    original = footprint.fr.canonical_format_name

    def canonical(value):
        calls.append(value)
        return original(value)

    def digest(raw):
        calls.append("bytes")
        assert raw == b'{"\xc3\xa9":"BF16"}'
        return "c" * 64

    monkeypatch.setattr(footprint.fr, "canonical_format_name", canonical)
    monkeypatch.setattr(footprint, "bytes_sha256hex", digest)
    assert footprint.assignment_serialization_sha256(Entries()) == "c" * 64
    assert calls == ["items", "name1", "fmt1", "FP8", "name2", "fmt2", "BF16", "bytes"]


@pytest.mark.parametrize(("value", "raw"), [
    ([float("nan"), float("inf"), float("-inf"), -0.0],
     b'{"a":[NaN,Infinity,-Infinity,-0.0]}'),
    ({10: "x", 9: "y"}, b'{"a":{"9":"y","10":"x"}}'),
    ((True, None, "é"), b'{"a":[true,null,"\xc3\xa9"]}'),
])
def test_assignment_registry_values_keep_direct_lax_bytes(monkeypatch, value, raw):
    monkeypatch.setattr(footprint.fr, "canonical_format_name", lambda fmt: value)
    assert footprint.assignment_serialization_sha256({"a": "fp8"}) == hashlib.sha256(raw).hexdigest()


def test_assignment_registry_cause_and_early_refusal(monkeypatch):
    cause = RuntimeError("registry cause")
    calls = []

    def refuse(fmt):
        calls.append(fmt)
        raise ValueError("registry sentinel") from cause

    def forbidden(raw):
        pytest.fail("refusal must precede byte hashing")

    monkeypatch.setattr(footprint.fr, "canonical_format_name", refuse)
    monkeypatch.setattr(footprint, "bytes_sha256hex", forbidden)
    with pytest.raises(ValueError, match="^registry sentinel$") as refused:
        footprint.assignment_serialization_sha256({"z": " fp8 ", "a": "bf16"})
    assert refused.value.__cause__ is cause
    assert calls == ["FP8"]


def test_assignment_routes_normalized_mapping_to_direct_utf8_lax(monkeypatch):
    calls = []

    class ProfileSeam:
        def encoded(self, value):
            calls.append(("encoded", value))
            return b"profile sentinel"

    def digest(raw):
        calls.append(("bytes", raw))
        return "d" * 64

    monkeypatch.setattr(footprint, "DIRECT_UTF8_LAX", ProfileSeam())
    monkeypatch.setattr(footprint, "bytes_sha256hex", digest)
    assert footprint.assignment_serialization_sha256({"z": " fp8 ", "a": " bf16 "}) == "d" * 64
    assert calls == [("encoded", {"z": "FP8_E4M3", "a": "BF16"}), ("bytes", b"profile sentinel")]
    assert footprint.DIRECT_UTF8_LAX is not digests.DIRECT_UTF8_STRICT
