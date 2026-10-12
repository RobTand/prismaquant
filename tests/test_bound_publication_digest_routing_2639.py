"""Bound-publication digest sites route through digests owners (pq #2639).

RED: each test patches the digests owner with a recorder and calls the
site. Before routing the site hashes inline, so the recorder sees
nothing and the test fails. After routing the recorder sees the call.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "tools"))
sys.path.insert(0, str(REPOSITORY))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUANTA = _load("route2639_quanta", "tools/dispatch_joint_quanta.py")
CAMPAIGN = _load("route2639_campaign", "tools/dispatch_tessera_campaign.py")
UNION = _load("route2639_union", "tools/union_tessera_census_tables.py")

from prismaquant import digests as owners  # noqa: E402
from prismaquant import stage_b_prep_io as preparation  # noqa: E402


def test_file_range_recipe_matches_offset_read(tmp_path):
    blob = bytes(range(256)) * 40000
    target = tmp_path / "r.bin"
    target.write_bytes(blob)
    for offset, size in ((0, len(blob)), (1, 1000), (7, 8 << 20), (len(blob) - 3, 3), (5, 0)):
        assert owners.file_range_sha256hex(target, offset, size) == hashlib.sha256(
            blob[offset:offset + size]).hexdigest()
    with pytest.raises(ValueError):
        owners.file_range_sha256hex(target, 0, len(blob) + 1)


def test_hash_range_routes_through_owner(monkeypatch, tmp_path):
    target = tmp_path / "r.bin"
    target.write_bytes(b"0123456789")
    seen = []
    monkeypatch.setattr(preparation, "file_range_sha256hex",
                        lambda *a: seen.append(a) or "routed")
    assert preparation._hash_range(str(target), 2, 4) == "routed"
    assert seen == [(str(target), 2, 4)]


def test_seed_workspace_rows_routes_through_owner(monkeypatch, tmp_path):
    census = {"model": "m"}
    census_path = tmp_path / "census.json"
    census_path.write_text(json.dumps(census))
    root = tmp_path / "ws"
    root.mkdir()
    (root / "plan.json").write_text(json.dumps({
        "schema": CAMPAIGN.PLAN_SCHEMA, "model": "m",
        "census": str(census_path), "calibration_cache": "cc",
        "rows": [{"groups": ["g"]}]}))
    seen = []
    monkeypatch.setattr(owners, "bytes_sha256hex",
                        lambda data: seen.append(bytes(data)) or "routed")
    rows, binding = CAMPAIGN._seed_workspace_rows(str(root), census=census,
                                                 calibration_cache="cc")
    assert binding["plan_sha256"] == "routed"
    assert seen == [(root / "plan.json").read_bytes()]


def test_seed_for_selection_routes_through_owner(monkeypatch, tmp_path):
    selection = {"units": ["u"]}
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    rowdir = tmp_path / "row"
    rowdir.mkdir()
    (rowdir / "cost.anchors.json").write_text(json.dumps({"identity_sha256": "s"}))
    rows = {("g",): {"units": str(selection_path), "dir": str(rowdir), "row_id": "r1"}}
    seen = []
    monkeypatch.setattr(owners, "bytes_sha256hex",
                        lambda data: seen.append(bytes(data)) or "routed")
    out = CAMPAIGN._seed_for_selection(rows, ["g"], selection)
    assert out["manifest_sha256_at_plan"] == "routed"
    assert seen == [(rowdir / "cost.anchors.json").read_bytes()]


def test_union_digest_routes_through_owner(monkeypatch):
    seen = []
    monkeypatch.setattr(owners, "text_sha256hex",
                        lambda text: seen.append(text) or "routed")
    assert UNION._digest({"b": 1}) == "routed"
    assert seen == [UNION._json_key({"b": 1})]


def test_union_indent_dump_routes_through_owner(monkeypatch):
    seen = []

    def text(value):
        seen.append(value)
        return "routed"

    monkeypatch.setattr(owners, "DIRECT_UTF8_INDENT2_STRICT", type("P", (), {"text": staticmethod(text)}))
    assert UNION._indent_dump({"b": 1}) == "routed"
    assert seen == [{"b": 1}]


def test_routed_values_match_hashlib(tmp_path):
    blob = b"abc" * 1000
    target = tmp_path / "r.bin"
    target.write_bytes(blob)
    assert owners.file_range_sha256hex(target, 7, 2500) == hashlib.sha256(blob[7:2507]).hexdigest()
    assert UNION._digest({"b": 1}) == hashlib.sha256(
        UNION._json_key({"b": 1}).encode("utf-8")).hexdigest()
    assert UNION._indent_dump({"b": [1, 2]}) == json.dumps(
        {"b": [1, 2]}, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
