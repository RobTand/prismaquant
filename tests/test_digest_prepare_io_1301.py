"""Exact old outcomes and owner routing for preparation JSON/SHA sites."""
from __future__ import annotations

import hashlib
import json
import pickle
from types import SimpleNamespace

import pytest

from prismaquant import cost_stage_checkpoint as journal, digests
from tests.golden_table import GoldenTable


@pytest.fixture(scope="module")
def golden():
    return GoldenTable("digest_prepare_io_1301")


@pytest.mark.parametrize("pins", [
    {"é": [1, -0.0, True, None]}, {10: "ten", 9: "nine"},
    {"value": float("nan")}, {"value": "\ud800"},
    {1: "int", "1": "str"}, {"bad": {1, 2}},
])
def test_migration_pin_dedup_old_outcomes(pins, golden):
    first = {"proof_bundle_sha256": "a" * 64, "old_pins": pins,
             "new_pins": {"z": 1, "a": 2}, "run_id": "a", "keep": "first"}
    second = dict(first, run_id="b", keep="second")
    golden.call(lambda: journal.merge_identity_migrations({"b": [second], "a": [first]}))


@pytest.mark.parametrize("identity", [
    {"é": [1, -0.0, True, None], "emoji": "😀"},
    {10: "ten", 9: "nine"}, {"nonfinite": float("nan")},
    {"surrogate": "\ud800"},
])
def test_journal_manifest_old_bytes_and_refusals(tmp_path, identity, golden):
    def write():
        root, seal, states = journal.prepare_journal(tmp_path / "journal",
            stage="prépare", resume=False, identity=identity, qnames=["unit-é", "unit.z"])
        return seal, (root / "manifest.json").read_bytes(), states
    golden.call(write, tmp=tmp_path)


def test_journal_manifest_routes_utf8_without_trailing_lf(tmp_path, monkeypatch):
    profile = digests.DIRECT_UTF8_INDENT2_STRICT
    calls = []
    monkeypatch.setattr(journal, "DIRECT_UTF8_INDENT2_STRICT", SimpleNamespace(encoded=
        lambda value: calls.append(value) or profile.encoded(value)))
    root, _, _ = journal.prepare_journal(tmp_path / "journal", stage="é", resume=False,
                                        identity={"label": "é"}, qnames=["unit-é"])
    assert len(calls) == 1
    raw = (root / "manifest.json").read_bytes()
    assert "é".encode() in raw and not raw.endswith(b"\n")


def test_migration_pin_keys_route_spaced_lax_profile(monkeypatch):
    calls = []
    monkeypatch.setattr(journal, "DIRECT_ASCII_SPACED_LAX", SimpleNamespace(text=
        lambda value: calls.append(value) or digests.DIRECT_ASCII_SPACED_LAX.text(value)))
    row = {"proof_bundle_sha256": "a" * 64, "old_pins": {"é": 1}, "new_pins": None}
    assert journal.merge_identity_migrations({"row": [row]}) == [row]
    assert calls == [row["old_pins"], None]


def test_prepare_journal_refuses_a_boolean_worker_count(tmp_path):
    root = tmp_path / "journal"
    with pytest.raises(ValueError) as exc:
        journal.prepare_journal(
            root, stage="fixture", resume=False, identity={"source": "fixture"},
            qnames=["unit"], unit_workers=True,
        )
    assert str(exc.value) == "journal unit_workers must be a positive integer"
    assert not root.exists()
