"""Pin the CLI-only union summary's inherited JSON text before sharing its encoder."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from prismaquant import union_production_cache as union


_EMPTY = ('{"coverage_identity_sha256": null, "coverage_mode": null, '
          '"entries": null, "schema": null, "shard_id": null}')


@pytest.mark.parametrize(("payload", "expected"), [
    ({}, _EMPTY),
    ({"schema": "s", "entries": 0, "shard_id": "a",
      "coverage_mode": "exact", "coverage_identity_sha256": "id"},
     '{"coverage_identity_sha256": "id", "coverage_mode": "exact", '
     '"entries": 0, "schema": "s", "shard_id": "a"}'),
    ({"campaign_identity": {"render": {
        "coverage_mode": "nested", "coverage_identity_sha256": "nested-id"}}},
     '{"coverage_identity_sha256": "nested-id", "coverage_mode": "nested", '
     '"entries": null, "schema": null, "shard_id": null}'),
    ({"coverage_mode": "top", "coverage_identity_sha256": "top-id",
      "campaign_identity": {"render": {
          "coverage_mode": "nested", "coverage_identity_sha256": "nested-id"}}},
     '{"coverage_identity_sha256": "top-id", "coverage_mode": "top", '
     '"entries": null, "schema": null, "shard_id": null}'),
    ({"coverage_mode": "", "coverage_identity_sha256": 0,
      "campaign_identity": {"render": {
          "coverage_mode": "fallback", "coverage_identity_sha256": "fallback-id"}}},
     '{"coverage_identity_sha256": "fallback-id", "coverage_mode": "fallback", '
     '"entries": null, "schema": null, "shard_id": null}'),
    ({"coverage_mode": False, "coverage_identity_sha256": None,
      "campaign_identity": {"render": {
          "coverage_mode": 0, "coverage_identity_sha256": False}}},
     '{"coverage_identity_sha256": false, "coverage_mode": 0, '
     '"entries": null, "schema": null, "shard_id": null}'),
    ({"coverage_mode": "top", "coverage_identity_sha256": "id",
      "campaign_identity": {"render": None}},
     '{"coverage_identity_sha256": "id", "coverage_mode": "top", '
     '"entries": null, "schema": null, "shard_id": null}'),
    ({"schema": "é\x00\r\n", "shard_id": '"\\', "entries": False},
     '{"coverage_identity_sha256": null, "coverage_mode": null, '
     '"entries": false, "schema": "\\u00e9\\u0000\\r\\n", '
     '"shard_id": "\\\"\\\\"}'),
    ({"schema": "\ud800"},
     '{"coverage_identity_sha256": null, "coverage_mode": null, '
     '"entries": null, "schema": "\\ud800", "shard_id": null}'),
    ({"coverage_identity_sha256": float("nan"), "coverage_mode": float("inf"),
      "entries": float("-inf"), "schema": -0.0, "shard_id": True},
     '{"coverage_identity_sha256": NaN, "coverage_mode": Infinity, '
     '"entries": -Infinity, "schema": -0.0, "shard_id": true}'),
    ({"entries": [2, 1], "coverage_mode": {"z": "é", "a": 1}},
     '{"coverage_identity_sha256": null, "coverage_mode": {"a": 1, "z": "\\u00e9"}, '
     '"entries": [2, 1], "schema": null, "shard_id": null}'),
])
def test_literal_inherited_summary_text(payload, expected):
    assert union._summary(payload) == expected


@pytest.mark.parametrize("key", ["ignored", "tensor_files", "producer"])
def test_unused_payload_fields_are_not_serialized(key):
    assert union._summary({key: object()}) == _EMPTY


@pytest.mark.parametrize("key", [
    "schema", "entries", "shard_id", "coverage_mode", "coverage_identity_sha256",
])
def test_selected_unsupported_values_keep_native_json_refusal(key):
    with pytest.raises(TypeError) as refused:
        union._summary({key: object()})
    assert str(refused.value) == "Object of type object is not JSON serializable"
    assert refused.value.__cause__ is None


@pytest.mark.parametrize(("payload", "message"), [
    ({"campaign_identity": None}, "'NoneType' object has no attribute 'get'"),
    ({"campaign_identity": []}, "'list' object has no attribute 'get'"),
    ({"campaign_identity": {"render": None}},
     "'NoneType' object has no attribute 'get'"),
    ({"coverage_mode": "top", "campaign_identity": {"render": []}},
     "'list' object has no attribute 'get'"),
])
def test_nested_lookup_refusals_remain_native(payload, message):
    with pytest.raises(AttributeError) as refused:
        union._summary(payload)
    assert str(refused.value) == message
    assert refused.value.__cause__ is None


def test_summary_routes_selected_objects_to_the_existing_profile(monkeypatch):
    nested = {"token": "é"}
    calls = []

    def text(value):
        calls.append(value)
        return "sentinel summary text"

    monkeypatch.setattr(union, "DIRECT_ASCII_SPACED_LAX", SimpleNamespace(text=text))
    result = union._summary({
        "schema": "s", "entries": 7, "coverage_identity_sha256": nested,
        "campaign_identity": {"render": {"coverage_mode": "fallback"}},
        "ignored": object(),
    })
    assert result == "sentinel summary text"
    assert calls == [{"schema": "s", "entries": 7, "coverage_mode": "fallback",
                      "coverage_identity_sha256": nested, "shard_id": None}]
    assert calls[0]["coverage_identity_sha256"] is nested
