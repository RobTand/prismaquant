"""Preserve the sensitivity-card identity recipe while sharing its text hash."""

from __future__ import annotations

import dataclasses
import hashlib
import re

import pytest

from prismaquant import sensitivity_card
from prismaquant.sensitivity_card import CardProvenance, RenderBasis

_SIMPLE_PAYLOAD = (
    b'{"calib_hash": "c", "model_id": "m", "n_calib_samples": 1, '
    b'"render_basis": "rtn", "seq_len": 2}'
)
_SIMPLE_SHA256 = "fc68f39f4f2f0abcd622fea68ff5786c8fcb44a0802195116a7a8f2d4721f602"


def _card_for_fingerprint(**changes: object) -> CardProvenance:
    return dataclasses.replace(
        CardProvenance("m", "c", 1, 2, "probe-original"), **changes
    )


@pytest.mark.parametrize(
    ("changes", "expected_payload", "expected_sha256"),
    [
        pytest.param({}, _SIMPLE_PAYLOAD, _SIMPLE_SHA256, id="simple-spaced-json"),
        pytest.param(
            {"model_id": '雪\n"\\\0', "calib_hash": "é"},
            br'{"calib_hash": "\u00e9", "model_id": "\u96ea\n\"\\\u0000", '
            br'"n_calib_samples": 1, "render_basis": "rtn", "seq_len": 2}',
            "1451d755bd57ec80bb145abd2b45c85c133c5ccaf5a7383dd11f28b50d95f86b",
            id="ascii-escaped-unicode-controls",
        ),
        pytest.param(
            {"model_id": "\ud800"},
            br'{"calib_hash": "c", "model_id": "\ud800", "n_calib_samples": 1, '
            br'"render_basis": "rtn", "seq_len": 2}',
            "2b6a033bfd53236b7d462b0f63fb6c6fc32ee7696cf322b8f4ea0a20e09e5267",
            id="surrogate-remains-json-escaped",
        ),
        pytest.param(
            {"n_calib_samples": 2, "seq_len": 64, "render_basis": RenderBasis.COMPENSATED},
            b'{"calib_hash": "c", "model_id": "m", "n_calib_samples": 2, '
            b'"render_basis": "compensated", "seq_len": 64}',
            "efe60e49f9c60de3df9a349ce3be2d998bd307822fa253e3504c8fe7690c3a2f",
            id="compensated-value",
        ),
        pytest.param(
            {"n_calib_samples": float("nan"), "seq_len": float("inf")},
            b'{"calib_hash": "c", "model_id": "m", "n_calib_samples": NaN, '
            b'"render_basis": "rtn", "seq_len": Infinity}',
            "b86fca1d20ee7e4f1aa337240c8afa847a88290504db9e098ad1591e15acefea",
            id="lax-nonfinite-spelling",
        ),
        pytest.param(
            {"n_calib_samples": -0.0, "seq_len": float("-inf")},
            b'{"calib_hash": "c", "model_id": "m", "n_calib_samples": -0.0, '
            b'"render_basis": "rtn", "seq_len": -Infinity}',
            "366d232a708cd17b056017a9b0256c11607860e4ea89fb2570fdd224afc569c9",
            id="negative-zero-and-infinity",
        ),
        pytest.param(
            {"model_id": "", "calib_hash": "", "n_calib_samples": 0, "seq_len": 0},
            b'{"calib_hash": "", "model_id": "", "n_calib_samples": 0, '
            b'"render_basis": "rtn", "seq_len": 0}',
            "b0f823d52e2a6c72a6780c4bc0fd632494ef4f65dbf519c93fbbc24603871ba7",
            id="empty-strings-and-zero",
        ),
        pytest.param(
            {"n_calib_samples": False, "seq_len": True},
            b'{"calib_hash": "c", "model_id": "m", "n_calib_samples": false, '
            b'"render_basis": "rtn", "seq_len": true}',
            "334ec5fcbdd108bcd4125b4d511a7cca444e05da76ae5b9fa69adbaa5b85ab4a",
            id="boolean-spelling-is-not-coerced",
        ),
    ],
)
def test_card_fingerprint_matches_literal_legacy_bytes(
    changes: dict[str, object], expected_payload: bytes, expected_sha256: str
) -> None:
    # Handwritten legacy bytes and independently captured full digests, not a
    # new owner's output used as its own oracle.
    assert hashlib.sha256(expected_payload).hexdigest() == expected_sha256
    result = _card_for_fingerprint(**changes).fingerprint()
    assert result == expected_sha256
    assert re.fullmatch(r"[0-9a-f]{64}", result)


@pytest.mark.parametrize("field", ["probe_commit", "notes"])
def test_card_fingerprint_ignores_nonidentity_fields(field: str) -> None:
    # An unserializable value makes accidental inclusion observable.
    assert _card_for_fingerprint(**{field: object()}).fingerprint() == _SIMPLE_SHA256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "other-model"),
        ("calib_hash", "other-calibration"),
        ("n_calib_samples", 3),
        ("seq_len", 32),
        ("render_basis", RenderBasis.COMPENSATED),
    ],
)
def test_card_fingerprint_keeps_all_five_identity_fields(field: str, value: object) -> None:
    assert _card_for_fingerprint(**{field: value}).fingerprint() != _SIMPLE_SHA256


def test_card_fingerprint_keeps_json_type_error_without_fallback() -> None:
    # Python 3.14 JSON adds an exception note, which pytest also matches.
    # Check the actual message rather than treating the note as message text.
    with pytest.raises(TypeError) as refused:
        _card_for_fingerprint(model_id=object()).fingerprint()
    assert str(refused.value) == "Object of type object is not JSON serializable"
    assert refused.value.__cause__ is None


def test_card_fingerprint_keeps_render_basis_attribute_error() -> None:
    with pytest.raises(AttributeError, match=r"^'str' object has no attribute 'value'$"):
        _card_for_fingerprint(render_basis="rtn").fingerprint()


def test_card_fingerprint_routes_the_exact_text_to_the_shared_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def record_text(text: str) -> str:
        observed.append(text)
        return "a" * 64

    monkeypatch.setattr(sensitivity_card, "text_sha256hex", record_text)
    assert _card_for_fingerprint().fingerprint() == "a" * 64
    assert observed == [_SIMPLE_PAYLOAD.decode("utf-8")]
