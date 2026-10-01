"""Pin the adapter's normalization, final JSON text and LF/error boundaries."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from prismaquant import quality_prefill_pb_adapter as adapter


@pytest.mark.parametrize(
    ("value", "encoded"),
    [
        ({}, b"{}\n"),
        ({"z": True, "a": None}, b'{"a":null,"z":true}\n'),
        ({10: "ten", 2: "two"}, b'{"10":"ten","2":"two"}\n'),
        ({"tuple": (False, 1, "x")}, b'{"tuple":[false,1,"x"]}\n'),
        ({"é": "雪\x00\n"}, b'{"\xc3\xa9":"\xe9\x9b\xaa\\u0000\\n"}\n'),
        (
            {"float": 1e-7, "negative_zero": -0.0},
            b'{"float":1e-07,"negative_zero":-0.0}\n',
        ),
        (
            ["a", True, None, {"z": 2, "a": 1}],
            b'["a",true,null,{"a":1,"z":2}]\n',
        ),
        ("", b'""\n'),
        ("\r\n", b'"\\r\\n"\n'),
    ],
)
def test_document_literal_normalized_bytes_and_single_lf(value, encoded):
    assert adapter.document_bytes(value) == encoded
    assert encoded.endswith(b"\n") and not encoded.endswith(b"\n\n")


def _cyclic_document():
    value = []
    value.append(value)
    return value


@pytest.mark.parametrize(
    "value",
    [
        float("nan"), float("inf"), float("-inf"), object(), {1, 2},
        {(1, 2): "tuple key"}, _cyclic_document(),
    ],
)
def test_document_keeps_adapter_refusal_and_normalizer_context(value):
    with pytest.raises(adapter.QualityPrefillAdapterError) as refused:
        adapter.document_bytes(value)
    assert str(refused.value) == (
        "document is not canonical JSON data: "
        "quality-prefill document is not canonical JSON data"
    )
    assert refused.value.__cause__ is None
    assert isinstance(refused.value.__context__, ValueError)
    assert str(refused.value.__context__) == (
        "quality-prefill document is not canonical JSON data"
    )
    assert isinstance(refused.value.__context__.__cause__, (TypeError, ValueError))


def test_document_keeps_utf8_refusal_outside_adapter_handler():
    with pytest.raises(UnicodeEncodeError) as refused:
        adapter.document_bytes({"bad": "\ud800"})
    assert str(refused.value) == (
        "'utf-8' codec can't encode character '\\ud800' in position 8: "
        "surrogates not allowed"
    )
    assert refused.value.__cause__ is None
    assert refused.value.__context__ is None


def test_document_normalizer_refuses_before_final_profile(monkeypatch):
    error = ValueError("normalizer refused")

    def normalize(value, *, where):
        assert where == "quality-prefill document"
        raise error

    def encode(value):
        pytest.fail("normalization refusal must precede final encoding")

    monkeypatch.setattr(adapter, "canonical_json", normalize)
    monkeypatch.setattr(
        adapter, "DIRECT_UTF8_STRICT", SimpleNamespace(text=encode), raising=False,
    )
    with pytest.raises(adapter.QualityPrefillAdapterError) as refused:
        adapter.document_bytes({"ignored": True})
    assert str(refused.value) == "document is not canonical JSON data: normalizer refused"
    assert refused.value.__cause__ is None
    assert refused.value.__context__ is error


def test_document_routes_same_normalized_object_to_existing_text_profile(monkeypatch):
    original = {2: ("value",)}
    normalized = {"2": ["value"]}
    events = []

    def normalize(value, *, where):
        assert value is original
        assert where == "quality-prefill document"
        events.append("normalize")
        return normalized

    def encode(value):
        assert value is normalized
        events.append("encode")
        return '{"2":["value"]}'

    monkeypatch.setattr(adapter, "canonical_json", normalize)
    monkeypatch.setattr(adapter, "DIRECT_UTF8_STRICT", SimpleNamespace(text=encode))
    assert adapter.document_bytes(original) == b'{"2":["value"]}\n'
    assert events == ["normalize", "encode"]
