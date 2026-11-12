"""Cache lookup precedence retained from main 05b62b474279 (PRI-226 C3-02)."""
from types import SimpleNamespace

import pytest

from prismaquant import export_native_compressed as export
from prismaquant import name_projection
from prismaquant.production_weight_cache import ProductionWeightCache


ORIGINAL = "model.language_model.layers.0.q.weight"
STRIPPED = "model.language_model.layers.0.q"
PROJECTED = "model.layers.0.q.weight"


@pytest.mark.parametrize(
    "name,fmt,stored,cache_expected,export_expected",
    [
        (ORIGINAL, "NVFP4", [(ORIGINAL, "NVFP4"), (STRIPPED, "NVFP4"),
                              (PROJECTED, "NVFP4")],
         (ORIGINAL, "NVFP4"), (ORIGINAL, "NVFP4")),
        (ORIGINAL, "NVFP4", [(STRIPPED, "NVFP4"), (PROJECTED, "NVFP4")],
         (STRIPPED, "NVFP4"), (STRIPPED, "NVFP4")),
        (ORIGINAL, "NVFP4", [(PROJECTED, "NVFP4")],
         (PROJECTED, "NVFP4"), (PROJECTED, "NVFP4")),
        (ORIGINAL, "NVFP4", [("model.layers.0.q", "NVFP4")], None, None),
        ("model.language_model.q", "NVFP4", [("model.q", "NVFP4")],
         ("model.q", "NVFP4"), ("model.q", "NVFP4")),
        ("q.weight", "NVFP4", [("q", "NVFP4")],
         ("q", "NVFP4"), ("q", "NVFP4")),
        ("q.weights", "NVFP4", [("q", "NVFP4")], None, None),
        (".weight", "NVFP4", [("", "NVFP4")], ("", "NVFP4"), ("", "NVFP4")),
        ("", "NVFP4", [("", "NVFP4")], ("", "NVFP4"), ("", "NVFP4")),
        ("q", "NVFP4", [], None, None),
        ("q", "nvfp4", [("q", "nvfp4"), ("q", "NVFP4")],
         ("q", "nvfp4"), ("q", "NVFP4")),
        ("q", " nvfp4 ", [("q", "NVFP4")], ("q", "NVFP4"), None),
        ("q", "MXFP8", [("q", "MXFP8"), ("q", "MXFP8_E4M3")],
         ("q", "MXFP8"), ("q", "MXFP8_E4M3")),
        ("q", "MXFP8_E4M3", [("q", "MXFP8")],
         ("q", "MXFP8"), ("q", "MXFP8")),
        ("q", "FP8", [("q", "FP8_E4M3")], ("q", "FP8_E4M3"), None),
        ("q", "FP8_E4M3", [("q", "FP8")], ("q", "FP8"), None),
        (ORIGINAL, "MXFP8", [(ORIGINAL, "MXFP8_E4M3"), (STRIPPED, "MXFP8")],
         (ORIGINAL, "MXFP8_E4M3"), (ORIGINAL, "MXFP8_E4M3")),
    ],
)
def test_lookup_preserves_retained_precedence(
    monkeypatch, name, fmt, stored, cache_expected, export_expected,
):
    weights = dict.fromkeys(stored, object())
    cache = ProductionWeightCache(weights=weights, levers={})
    assert cache.resolve_key(name, fmt) == cache_expected
    monkeypatch.setattr(export, "_PRODUCTION_WEIGHT_CACHE", SimpleNamespace(weights=weights))
    assert export._production_cache_lookup_key(name, fmt) == export_expected


def test_both_lookup_paths_consume_shared_name_rule(monkeypatch):
    calls = []

    def candidates(name):
        calls.append(name)
        return ("retained_alias",)

    monkeypatch.setattr(name_projection, "production_cache_name_candidates", candidates)
    key = ("retained_alias", "NVFP4")
    weights = {key: object()}
    assert ProductionWeightCache(weights=weights, levers={}).resolve_key("q", "NVFP4") == key
    monkeypatch.setattr(export, "_PRODUCTION_WEIGHT_CACHE", SimpleNamespace(weights=weights))
    assert export._production_cache_lookup_key("q", "NVFP4") == key
    assert calls == ["q", "q"]


@pytest.mark.parametrize("resolved", [("resolver_winner", "FP8"), None])
def test_export_retains_resolver_priority_and_none_fallback(monkeypatch, resolved):
    calls = []
    fallback = (STRIPPED, "NVFP4")

    def resolve(name, fmt):
        calls.append((name, fmt))
        return resolved

    monkeypatch.setattr(export, "_PRODUCTION_WEIGHT_CACHE", SimpleNamespace(
        weights={fallback: object()}, resolve_key=resolve,
    ))
    assert export._production_cache_lookup_key(ORIGINAL, "NVFP4") == (
        resolved if resolved is not None else fallback
    )
    assert calls == [(ORIGINAL, "NVFP4")]


def test_export_propagates_resolver_failure(monkeypatch):
    error = RuntimeError("cache refusal")

    def resolve(name, fmt):
        raise error

    monkeypatch.setattr(export, "_PRODUCTION_WEIGHT_CACHE", SimpleNamespace(
        weights={(STRIPPED, "NVFP4"): object()}, resolve_key=resolve,
    ))
    with pytest.raises(RuntimeError) as caught:
        export._production_cache_lookup_key(ORIGINAL, "NVFP4")
    assert caught.value is error


@pytest.mark.parametrize("cache", [None, SimpleNamespace(), SimpleNamespace(weights=None)])
def test_export_retains_empty_cache_miss(monkeypatch, cache):
    monkeypatch.setattr(export, "_PRODUCTION_WEIGHT_CACHE", cache)
    assert export._production_cache_lookup_key(ORIGINAL, "NVFP4") is None


@pytest.mark.parametrize("name,error", [(None, AttributeError), (7, AttributeError),
                                       (b"q.weight", TypeError)])
def test_lookup_retains_non_string_refusal(monkeypatch, name, error):
    with pytest.raises(error):
        ProductionWeightCache(weights={}, levers={}).resolve_key(name, "NVFP4")
    monkeypatch.setattr(export, "_PRODUCTION_WEIGHT_CACHE", SimpleNamespace(weights={}))
    with pytest.raises(error):
        export._production_cache_lookup_key(name, "NVFP4")
