"""Independent old-byte fixtures for ephemeral diagnostic fingerprints (#1804)."""
from __future__ import annotations

import hashlib

import pytest

from prismaquant import dev_mode


class _LongDiagnosticRepr:
    def __repr__(self) -> str:
        return "repr-é-" * 30


class _SurrogateDiagnosticRepr:
    def __repr__(self) -> str:
        return "x" * 161 + "\ud800"


def _expected_rendering(text: str, encoded: bytes) -> str:
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"{text[:160]}... ({len(text)} chars, sha256 {digest}...)"


@pytest.mark.parametrize(
    ("value", "encoded"),
    [
        pytest.param("a" * 161, b'"' + b"a" * 161 + b'"', id="ascii-json-quotes"),
        pytest.param("é" * 161, b'"' + b"\\u00e9" * 161 + b'"', id="ascii-escaped-unicode"),
        pytest.param(
            "\x00\n\r\t" * 41,
            b'"' + b"\\u0000\\n\\r\\t" * 41 + b'"',
            id="json-control-escapes",
        ),
        pytest.param(
            "\ud800" * 161,
            b'"' + b"\\ud800" * 161 + b'"',
            id="json-surrogates-remain-ascii",
        ),
        pytest.param(
            {"z": "é" * 161, "a": float("nan")},
            b'{"a": NaN, "z": "' + b"\\u00e9" * 161 + b'"}',
            id="sorted-spaced-json-lax-nan",
        ),
        pytest.param(
            _LongDiagnosticRepr(),
            b'"' + b"repr-\\u00e9-" * 30 + b'"',
            id="json-default-repr",
        ),
    ],
)
def test_long_rendering_preserves_old_json_bytes(value, encoded):
    text = value if isinstance(value, str) else repr(value)
    assert dev_mode._shown(value) == _expected_rendering(text, encoded)


@pytest.mark.parametrize("value", ["", "a" * 159, "é" * 160, 23])
def test_threshold_is_characters_and_short_values_pass_through(value):
    assert dev_mode._shown(value) == (value if isinstance(value, str) else repr(value))


def test_circular_json_keeps_the_prior_repr_fallback():
    value = []
    value.append(value)
    value.append("x" * 161)
    text = "[[...], '" + "x" * 161 + "']"
    encoded = b"[[...], '" + b"x" * 161 + b"']"
    assert repr(value) == text
    assert dev_mode._shown(value) == _expected_rendering(text, encoded)


def test_mixed_key_sort_error_keeps_strict_utf8_repr_fallback():
    value = {1: "x" * 161, "z": "é"}
    text = "{1: '" + "x" * 161 + "', 'z': 'é'}"
    encoded = b"{1: '" + b"x" * 161 + b"', 'z': '\xc3\xa9'}"
    assert repr(value) == text
    assert dev_mode._shown(value) == _expected_rendering(text, encoded)


def test_fallback_lone_surrogate_still_refuses_strict_utf8():
    value = {1: _SurrogateDiagnosticRepr(), "z": "short"}
    with pytest.raises(UnicodeEncodeError) as raised:
        dev_mode._shown(value)
    assert raised.value.encoding == "utf-8"
    assert raised.value.reason == "surrogates not allowed"


def test_warning_text_and_suspended_seal_verdict_are_unchanged(capsys):
    expected, actual = "a" * 161, "b" * 161
    assert dev_mode.seal_check("probe", expected, actual, where="fixture", environ={}) is False
    left = _expected_rendering(expected, b'"' + b"a" * 161 + b'"')
    right = _expected_rendering(actual, b'"' + b"b" * 161 + b'"')
    assert capsys.readouterr().out == (
        f"[DEV-MODE] seal probe differs (fixture): expected {left}, actual {right}; "
        "sealing is off (PQ #1147), continuing with the stored data\n"
    )


def test_diagnostic_fingerprint_routes_only_the_existing_bytes(monkeypatch):
    seen = []

    def record_digest(encoded):
        seen.append(encoded)
        return hashlib.sha256(encoded).hexdigest()

    monkeypatch.setattr(dev_mode, "bytes_sha256hex", record_digest)
    assert dev_mode._shown("a" * 160) == "a" * 160
    value = "é" * 161
    encoded = b'"' + b"\\u00e9" * 161 + b'"'
    assert dev_mode._shown(value) == _expected_rendering(value, encoded)
    assert seen == [encoded]
