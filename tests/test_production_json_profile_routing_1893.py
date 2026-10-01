"""Keep round-trip normalization separate from final metadata encoding."""

import hashlib

import pytest

from prismaquant import production_weight_cache as cache


@pytest.mark.parametrize(
    ("value", "encoded"),
    [
        (None, b"null"),
        ({2: "two", 10: "ten"}, b'{"10":"ten","2":"two"}'),
        (
            {"z": (True, None, -0.0), "a": "é\n\x00"},
            b'{"a":"\xc3\xa9\\n\\u0000","z":[true,null,-0.0]}',
        ),
        ({"nested": {"雪": 3.5}}, b'{"nested":{"\xe9\x9b\xaa":3.5}}'),
    ],
)
def test_final_json_profile_keeps_roundtrip_bytes(value, encoded):
    actual = cache._canonical_json_sha256(value, where="profile fixture")
    assert actual == hashlib.sha256(encoded).hexdigest()
    assert len(actual) == 64


def test_final_json_profile_routes_same_canonical_object_before_hash(monkeypatch):
    value = object()
    canonical = {"z": [2, None], "a": "é"}
    encoded = b'{"a":"\xc3\xa9","z":[2,null]}'
    calls = []

    def normalize(actual, *, where):
        assert actual is value
        calls.append(("normalize", where))
        return canonical

    class Profile:
        def encoded(self, actual):
            assert actual is canonical
            calls.append(("encode", actual))
            return encoded

    def digest(actual):
        assert actual is encoded
        calls.append(("hash", actual))
        return "forwarded-byte-owner"

    monkeypatch.setattr(cache, "_canonical_json_value", normalize)
    monkeypatch.setattr(cache, "DIRECT_UTF8_STRICT", Profile())
    monkeypatch.setattr(cache, "bytes_sha256hex", digest)
    assert cache._canonical_json_sha256(value, where="caller") == "forwarded-byte-owner"
    assert calls == [("normalize", "caller"), ("encode", canonical), ("hash", encoded)]


def test_normalizer_refusal_precedes_profile_and_hash(monkeypatch):
    refused = ValueError("normalization refused")
    calls = []

    def normalize(actual, *, where):
        calls.append((actual, where))
        raise refused

    class ForbiddenProfile:
        def encoded(self, actual):
            pytest.fail("profile must not run after normalizer refusal")

    def forbidden_hash(actual):
        pytest.fail("hash must not run after normalizer refusal")

    monkeypatch.setattr(cache, "_canonical_json_value", normalize)
    monkeypatch.setattr(cache, "DIRECT_UTF8_STRICT", ForbiddenProfile(), raising=False)
    monkeypatch.setattr(cache, "bytes_sha256hex", forbidden_hash)
    value = object()
    with pytest.raises(ValueError) as caught:
        cache._canonical_json_sha256(value, where="caller")
    assert caught.value is refused
    assert caught.value.__cause__ is None
    assert calls == [(value, "caller")]
