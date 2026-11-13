"""Keep default-spaced identity JSON while consolidating its encoding owner."""

from __future__ import annotations

import hashlib
import json

import pytest

from prismaquant import digests


def test_spaced_profile_addition_keeps_legacy_positional_compact_constructor():
    profile = digests.JsonProfile("legacy-positional", True, True, str)
    assert profile.encoded({"v": "雪", "n": float("nan")}) == br'{"n":NaN,"v":"\u96ea"}'


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({}, b"{}"),
        ([], b"[]"),
        ({"z": "雪", "a": "é\n\0"}, br'{"a": "\u00e9\n\u0000", "z": "\u96ea"}'),
        (
            {"n": float("nan"), "p": float("inf"), "m": float("-inf"), "z": -0.0},
            b'{"m": -Infinity, "n": NaN, "p": Infinity, "z": -0.0}',
        ),
        ({10: "ten", 2: "two"}, b'{"2": "two", "10": "ten"}'),
        ({"tuple": (True, None, {"b": 2, "a": 1})}, b'{"tuple": [true, null, {"a": 1, "b": 2}]}'),
        ({"s": "\ud800"}, br'{"s": "\ud800"}'),
        ({"x": False, "y": True}, b'{"x": false, "y": true}'),
    ],
)
def test_named_spaced_profile_literal_bytes_and_streamed_hash(value, expected):
    profile = digests.DIRECT_ASCII_SPACED_LAX
    assert profile.text(value) == expected.decode("utf-8")
    assert profile.encoded(value) == expected
    expected_sha = hashlib.sha256(expected).hexdigest()
    assert profile.sha256(value) == expected_sha
    assert profile.sha256_streamed(value) == expected_sha


def test_named_spaced_profile_does_not_redefine_the_compact_encoders():
    spaced = digests.DIRECT_ASCII_SPACED_LAX
    assert spaced.name == "direct-ascii-spaced-lax"
    assert spaced.separators == (", ", ": ")
    assert spaced.ensure_ascii is True and spaced.allow_nan is True
    assert spaced.default is None
    for compact in (
        digests.DIRECT_UTF8_STRICT, digests.DIRECT_ASCII_STRICT,
        digests.DIRECT_ASCII_LAX, digests.DIRECT_ASCII_LAX_DEFAULT_STR,
    ):
        assert compact.separators == (",", ":")
    value = {"a": 1, "b": 2}
    assert spaced.encoded(value) == b'{"a": 1, "b": 2}'
    assert digests.DIRECT_ASCII_LAX.encoded(value) == b'{"a":1,"b":2}'
    assert spaced.sha256(value) != digests.DIRECT_ASCII_LAX.sha256(value)


@pytest.mark.parametrize("value", [object(), {("tuple",): 1}, {1: "int", "s": "str"}])
def test_named_spaced_profile_keeps_native_type_refusals_without_fallback(value):
    with pytest.raises(TypeError) as old_refusal:
        json.dumps(value, sort_keys=True)
    with pytest.raises(TypeError) as shared_refusal:
        digests.DIRECT_ASCII_SPACED_LAX.text(value)
    assert str(shared_refusal.value) == str(old_refusal.value)
    assert shared_refusal.value.__cause__ is None


def test_named_spaced_profile_keeps_circular_reference_refusal():
    value = []
    value.append(value)
    with pytest.raises(ValueError) as refused:
        digests.DIRECT_ASCII_SPACED_LAX.text(value)
    assert str(refused.value) == "Circular reference detected"
    assert refused.value.__cause__ is None
