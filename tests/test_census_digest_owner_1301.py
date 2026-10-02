"""The loaded checkpoint's depth-two JSON stream retains bytes and errors."""
from __future__ import annotations

import pytest

from prismaquant import digests, tessera_census_cache as cache
from tests.golden_table import GoldenTable


@pytest.fixture(scope="module")
def golden():
    return GoldenTable("census_digest_owner_1301")


CASES = [
    None, {}, [], (1, 2), {"x": (1, -0.0, None)},
    {"é": ["😀", "\u2028", -0.0, 1e-7, 2**70, True]},
    {10: "ten", 9: "nine"}, {"child": {2: "two"}},
    {"child": [{10: "ten", 9: "nine"}]},
    {"child": {"grandchild": {10: "ten", 9: "nine"}}},
    {1: "one", "1": "string"}, {"bad": {1, 2}},
    float("nan"), {"a": float("nan"), "z": {1: "bad-key"}},
    {"a": {1: "bad-key"}, "z": float("nan")},
    "\ud800", {"surrogate": "\ud800"},
]


@pytest.mark.parametrize("value", CASES)
def test_old_loaded_checkpoint_outcomes(value, golden):
    golden.call(lambda: cache.canonical_json_sha256_of_loaded(value))


def test_old_loaded_checkpoint_cycle_refusal(golden):
    value = {}
    value["self"] = value
    golden.call(lambda: cache.canonical_json_sha256_of_loaded(value))


def test_loaded_checkpoint_routes_to_exact_partition_owner(monkeypatch):
    value = {"child": [{10: "ten", 9: "nine"}], "é": "😀"}
    calls = []
    owner = getattr(digests, "checkpoint_json_sha256", None)

    def observed(graph, *, error):
        calls.append((graph, error))
        assert owner is not None
        return owner(graph, error=error)

    monkeypatch.setattr(cache, "checkpoint_json_sha256", observed)
    result = cache.canonical_json_sha256_of_loaded(value)
    assert calls == [(value, cache.CensusCacheError)]
    assert type(result) is str and len(result) == 64
