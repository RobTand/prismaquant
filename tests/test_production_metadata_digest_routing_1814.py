"""Pin the two distinct legacy production-cache metadata byte profiles."""

import hashlib

import pytest

from prismaquant import production_weight_cache as cache


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, b"null"),
        ({"z": 2, "a": 1}, b'{"a":1,"z":2}'),
        (
            {"unicode": "é雪😀", "control": "\x00\n"},
            b'{"control":"\\u0000\\n","unicode":"\xc3\xa9\xe9\x9b\xaa\xf0\x9f\x98\x80"}',
        ),
        # The first round trip turns numeric keys into strings. The SECOND
        # sort is lexical; a direct dump of the original dict is different.
        ({2: "two", 10: "ten"}, b'{"10":"ten","2":"two"}'),
        (
            {"tuple": (None, True, -0.0, "é"), "b": False},
            b'{"b":false,"tuple":[null,true,-0.0,"\xc3\xa9"]}',
        ),
        ({"value": [3.5, 1e-6, 1e20]}, b'{"value":[3.5,1e-06,1e+20]}'),
    ],
)
def test_canonical_metadata_keeps_roundtrip_and_strict_compact_bytes(value, expected):
    actual = cache._canonical_json_sha256(value, where="fixture")
    assert actual == hashlib.sha256(expected).hexdigest()
    assert len(actual) == 64
    assert actual == actual.lower()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_canonical_metadata_keeps_nonfinite_refusal_and_cause(value):
    with pytest.raises(ValueError) as caught:
        cache._canonical_json_sha256({"value": value}, where="fixture")
    assert str(caught.value) == "fixture is not canonical JSON data"
    assert type(caught.value.__cause__) is ValueError


@pytest.mark.parametrize("value", [object(), {1: "one", "x": "mixed"}])
def test_canonical_metadata_keeps_unsupported_refusal_and_cause(value):
    with pytest.raises(ValueError) as caught:
        cache._canonical_json_sha256(value, where="fixture")
    assert str(caught.value) == "fixture is not canonical JSON data"
    assert type(caught.value.__cause__) is TypeError


def test_canonical_metadata_keeps_cycle_refusal_and_cause():
    value = []
    value.append(value)
    with pytest.raises(ValueError) as caught:
        cache._canonical_json_sha256(value, where="fixture")
    assert str(caught.value) == "fixture is not canonical JSON data"
    assert type(caught.value.__cause__) is ValueError
    assert str(caught.value.__cause__) == "Circular reference detected"


def test_canonical_metadata_keeps_unwrapped_utf8_refusal():
    with pytest.raises(UnicodeEncodeError) as caught:
        cache._canonical_json_sha256({"value": "\ud800"}, where="fixture")
    assert caught.value.encoding == "utf-8"
    assert caught.value.object == '{"value":"\ud800"}'
    assert caught.value.reason == "surrogates not allowed"
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("qnames", "expected"),
    [
        ([], b""),
        (["z", "a"], b"a\nz"),
        (["z", "a", "a"], b"a\na\nz"),
        ([None, 3, True], b"3\nNone\nTrue"),
        (["é", "雪", "a"], b"a\n\xc3\xa9\n\xe9\x9b\xaa"),
        (["", "a", ""], b"\n\na"),
        (["a\nb"], b"a\nb"),
        (["a", "b"], b"a\nb"),
        (["a\r\nb", "\x00"], b"\x00\na\r\nb"),
    ],
)
def test_qname_metadata_keeps_coercion_sort_multiplicity_and_lf(qnames, expected):
    actual = cache._qname_set_sha256(qnames)
    assert actual == hashlib.sha256(expected).hexdigest()
    assert len(actual) == 64
    assert actual == actual.lower()


def test_qname_metadata_consumes_generator_and_coerces_once_in_input_order():
    calls = []

    class Name(str):
        def __str__(self):
            text = str.__str__(self)
            calls.append(text)
            return text

    qnames = (Name(text) for text in ("z", "a", "a"))
    assert cache._qname_set_sha256(qnames) == hashlib.sha256(b"a\na\nz").hexdigest()
    assert calls == ["z", "a", "a"]
    assert list(qnames) == []


def test_qname_metadata_keeps_native_coercion_refusal():
    class Name(str):
        def __str__(self):
            raise ValueError("name conversion refused")

    with pytest.raises(ValueError) as caught:
        cache._qname_set_sha256([Name()])
    assert str(caught.value) == "name conversion refused"
    assert caught.value.__cause__ is None


def test_qname_metadata_keeps_unwrapped_utf8_refusal():
    with pytest.raises(UnicodeEncodeError) as caught:
        cache._qname_set_sha256(["\ud800"])
    assert caught.value.encoding == "utf-8"
    assert caught.value.object == "\ud800"
    assert caught.value.reason == "surrogates not allowed"
    assert caught.value.__cause__ is None


def test_canonical_metadata_routes_only_final_bytes_after_normalization(monkeypatch):
    value = object()
    normalization_calls = []
    owner_calls = []

    def normalize(actual, *, where):
        normalization_calls.append((actual, where))
        return {"2": "two", "10": "é"}

    def owner(data):
        owner_calls.append(data)
        return "forwarded-byte-owner"

    monkeypatch.setattr(cache, "_canonical_json_value", normalize)
    monkeypatch.setattr(cache, "bytes_sha256hex", owner)
    assert cache._canonical_json_sha256(value, where="caller") == "forwarded-byte-owner"
    assert normalization_calls == [(value, "caller")]
    assert owner_calls == [b'{"10":"\xc3\xa9","2":"two"}']


def test_qname_metadata_routes_exact_final_text_once(monkeypatch):
    calls = []

    def owner(text):
        calls.append(text)
        return "forwarded-text-owner"

    monkeypatch.setattr(cache, "text_sha256hex", owner)
    assert cache._qname_set_sha256(["é", "a", "a"]) == "forwarded-text-owner"
    assert calls == ["a\na\né"]
