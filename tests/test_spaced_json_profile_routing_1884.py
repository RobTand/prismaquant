"""Keep default-spaced identity JSON while consolidating its encoding owner."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest

from prismaquant import digests, sensitivity_card, stage_b_prep_io
from prismaquant.sensitivity_card import CardProvenance

_SIMPLE_CARD = (
    b'{"calib_hash": "c", "model_id": "m", "n_calib_samples": 1, '
    b'"render_basis": "rtn", "seq_len": 2}'
)


def _spaced_identity_card(**changes: object) -> CardProvenance:
    return dataclasses.replace(CardProvenance("m", "c", 1, 2, "unused"), **changes)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, _SIMPLE_CARD),
        (
            {"model_id": '雪\n"\\\0', "calib_hash": "é"},
            br'{"calib_hash": "\u00e9", "model_id": "\u96ea\n\"\\\u0000", '
            br'"n_calib_samples": 1, "render_basis": "rtn", "seq_len": 2}',
        ),
        (
            {"model_id": "\ud800"},
            br'{"calib_hash": "c", "model_id": "\ud800", "n_calib_samples": 1, '
            br'"render_basis": "rtn", "seq_len": 2}',
        ),
        (
            {"n_calib_samples": False, "seq_len": True},
            b'{"calib_hash": "c", "model_id": "m", "n_calib_samples": false, '
            b'"render_basis": "rtn", "seq_len": true}',
        ),
        (
            {"n_calib_samples": float("nan"), "seq_len": float("inf")},
            b'{"calib_hash": "c", "model_id": "m", "n_calib_samples": NaN, '
            b'"render_basis": "rtn", "seq_len": Infinity}',
        ),
        (
            {"n_calib_samples": -0.0, "seq_len": float("-inf")},
            b'{"calib_hash": "c", "model_id": "m", "n_calib_samples": -0.0, '
            b'"render_basis": "rtn", "seq_len": -Infinity}',
        ),
    ],
)
def test_spaced_card_keeps_literal_encoding_and_full_hash(monkeypatch, changes, expected):
    observed = []

    def hash_text(text):
        observed.append(text.encode("utf-8"))
        return hashlib.sha256(observed[-1]).hexdigest()

    monkeypatch.setattr(sensitivity_card, "text_sha256hex", hash_text)
    assert _spaced_identity_card(**changes).fingerprint() == hashlib.sha256(expected).hexdigest()
    assert observed == [expected]


@pytest.mark.parametrize("maximum", [0, -1, True, 1.0, "13"])
def test_spaced_template_keeps_maximum_refusal_before_encoding(monkeypatch, maximum):
    def must_not_encode(value):
        pytest.fail("invalid maximum must refuse before the encoding owner")

    monkeypatch.setattr(
        stage_b_prep_io, "DIRECT_ASCII_SPACED_LAX", SimpleNamespace(text=must_not_encode),
        raising=False,
    )
    with pytest.raises(ValueError) as refused:
        stage_b_prep_io.build_preparation_template(
            metadata_root="/metadata", tier="tier", payload_max_bytes=maximum,
        )
    assert str(refused.value) == "the preparation template needs a positive payload maximum"
    assert refused.value.__cause__ is None


@pytest.mark.parametrize(
    ("root", "normalized", "tier"),
    [("/ignored/../metadata", "/metadata", "tier"), ("/méta/\ud800", "/méta/\ud800", 17)],
)
def test_spaced_template_keeps_body_ascii_and_sixteen_hex(monkeypatch, root, normalized, tier):
    body = {
        "schema": "prismaquant.prismabuild.produced_output_template.v1", "version": 1,
        "output_prefix": normalized,
        "slots": {"stage_b_metadata": {"class": "payload"}},
        "durable_maxima": {
            "payload_max_bytes": 13, "checkpoint_max_bytes": 0, "temp_max_bytes": 13,
        },
        "working_demands": {str(tier): {"minimum_gib": 0, "window_gib": 0}},
        "permitted_tiers": [str(tier)], "write_only": True,
    }
    expected = json.dumps(body, sort_keys=True).encode("utf-8")
    observed = []

    def hash_text(text):
        observed.append(text.encode("utf-8"))
        return hashlib.sha256(observed[-1]).hexdigest()

    monkeypatch.setattr(stage_b_prep_io, "text_sha256hex", hash_text)
    result = stage_b_prep_io.build_preparation_template(
        metadata_root=root, tier=tier, payload_max_bytes=13,
    )
    assert result == {
        **body, "template_id": "pq-stage-b-preparation-" + hashlib.sha256(expected).hexdigest()[:16],
    }
    assert observed == [expected]


def test_spaced_profile_addition_keeps_legacy_positional_compact_constructor():
    profile = digests.JsonProfile("legacy-positional", True, True, str)
    assert profile.encoded({"v": "雪", "n": float("nan")}) == br'{"n":NaN,"v":"\u96ea"}'


def test_spaced_card_routes_exact_identity_mapping_through_named_profile(monkeypatch):
    observed = []
    profile_text = "profile supplied card text"

    def encode(value):
        observed.append(value)
        return profile_text

    monkeypatch.setattr(
        sensitivity_card, "DIRECT_ASCII_SPACED_LAX", SimpleNamespace(text=encode),
    )
    monkeypatch.setattr(
        sensitivity_card, "text_sha256hex",
        lambda text: hashlib.sha256(text.encode()).hexdigest(),
    )
    result = _spaced_identity_card(notes=object(), probe_commit=object()).fingerprint()
    assert result == hashlib.sha256(profile_text.encode()).hexdigest()
    assert observed == [{
        "model_id": "m", "calib_hash": "c", "n_calib_samples": 1,
        "seq_len": 2, "render_basis": "rtn",
    }]


def test_spaced_template_routes_only_validated_body_through_named_profile(monkeypatch):
    observed = []
    profile_text = "profile supplied template text"

    def encode(value):
        observed.append(value)
        return profile_text

    monkeypatch.setattr(
        stage_b_prep_io, "DIRECT_ASCII_SPACED_LAX", SimpleNamespace(text=encode),
    )
    monkeypatch.setattr(
        stage_b_prep_io, "text_sha256hex",
        lambda text: hashlib.sha256(text.encode()).hexdigest(),
    )
    result = stage_b_prep_io.build_preparation_template(
        metadata_root="/ignored/../metadata", tier="tier", payload_max_bytes=13,
    )
    expected_id = "pq-stage-b-preparation-" + hashlib.sha256(profile_text.encode()).hexdigest()[:16]
    assert result["template_id"] == expected_id
    assert observed == [{key: value for key, value in result.items() if key != "template_id"}]


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
