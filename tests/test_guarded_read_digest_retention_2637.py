"""Pin the guarded-read digest retentions of prismaquant#2637."""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CENSUS = REPO / "docs/audits/digest_site_census_pq1301_2026-10-04.json"
SOURCE_COMMIT = "30b6231df000726beda55fb6706e641a5cb2ccb0"
GENERATED = "2026-10-09"

SCOPES = {
    "prismaquant/perturbed_x_cache.py": [
        "SerializedEntryDigest.__init__",
        "_read_exact_entry",
        "_seed_from",
        "calibration_data_hash",
        "load_verified_activation_cache_entry",
    ],
    "prismaquant/tessera_census_cache.py": ["_verify_dense_blob"],
}

# The source coordinates belong to the #2540 snapshot, not the historical census.
IDS = {
    "hashlib:prismaquant/perturbed_x_cache.py::SerializedEntryDigest.__init__@207:21": (
        "write-sink",
        "tests/test_exact_activation_entry_read_amplification.py::test_write_hashes_the_serialized_bytes_without_rereading_the_entry",
    ),
    "hashlib:prismaquant/perturbed_x_cache.py::_read_exact_entry@2117:14": (
        "guarded-read",
        "tests/test_exact_activation_entry_read_amplification.py::test_prefetch_reads_each_entry_once",
    ),
    "hashlib:prismaquant/perturbed_x_cache.py::_seed_from@2405:13": (
        "scientific-seed",
        "tests/test_perturbed_x_cache.py::test_perturbed_cache_shares_row_subsample_for_fused_siblings",
    ),
    "hashlib:prismaquant/perturbed_x_cache.py::calibration_data_hash@2375:8": (
        "scientific-seed",
        "tests/test_sample_parallel_probe.py::test_prepare_artifact_binds_every_local_slice",
    ),
    "hashlib:prismaquant/perturbed_x_cache.py::load_verified_activation_cache_entry@903:17": (
        "guarded-read",
        "tests/test_verified_capture_load.py::test_large_tensor_uses_readinto_and_releases_buffer_before_return",
    ),
    "hashlib:prismaquant/tessera_census_cache.py::_verify_dense_blob@245:13": (
        "consumed-size",
        "tests/test_tessera_census_cache.py::test_same_size_changed_blob_refuses_at_export_intake_or_under_hash_blobs",
    ),
}


def _scope_node(tree: ast.Module, scope: str) -> ast.AST:
    parts = scope.split(".")
    nodes: list[ast.AST] = list(tree.body)
    found: ast.AST | None = None
    for part in parts:
        found = None
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == part:
                    found = node
                    break
        assert found is not None, f"scope is missing: {scope}"
        nodes = list(found.body)
    assert found is not None
    return found


def _uses_raw_hashlib(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "hashlib":
                return True
    return False


def _manifest_calls() -> dict[str, dict]:
    refresh = json.loads(CENSUS.read_text())["refresh_2540"]
    assert refresh["schema"] == "prismaquant.digest-census-refresh.v1"
    assert refresh["issue"] == "prismaquant#2540"
    # A pinned provenance date is not a wall-clock expiry. Content identities below
    # establish freshness without requiring historical git objects in a checkout.
    assert refresh["source_commit"] == SOURCE_COMMIT, "wrong manifest source"
    assert refresh["generated"] == GENERATED, "wrong manifest generation"
    for rel in (*SCOPES, "prismaquant/digests.py", "prismaquant/tensor_digests.py"):
        path = REPO / rel
        assert not path.is_symlink(), rel
        raw = path.read_bytes()
        identity = refresh["source_blobs"][rel]
        assert identity["mode"] == "100644", rel
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        assert blob == identity["git_blob"], f"stale manifest source: {rel}"
        assert hashlib.sha256(raw).hexdigest() == identity["sha256"], rel
        if rel in refresh["owner_contracts"]:
            assert refresh["owner_contracts"][rel]["source_blob"] == blob, rel

    calls = refresh["calls"]
    indexed = {row["id"]: row for row in calls}
    assert len(indexed) == len(calls), "duplicate manifest call"
    for call_id in IDS:
        row = indexed[call_id]
        assert row["source_blob"] == refresh["source_blobs"][row["file"]]["git_blob"]
        assert row["id"] == f"hashlib:{row['file']}::{row['scope']}@{row['line']}:{row['column']}"
    return indexed


def _assert_retention(row: dict) -> None:
    call_id = row["id"]
    kind, reference = IDS[call_id]
    assert row["disposition"] == "retain", call_id
    assert row["eligible"] is False, call_id
    assert row["cohort"] is None, call_id
    assert row["disposition_reason"], call_id
    evidence = row["retention_evidence"]
    assert evidence, call_id
    assert evidence["kind"] == kind, call_id
    assert evidence["reference"] == reference, call_id
    assert evidence["source_blob"] == row["source_blob"], call_id
    assert evidence["manifest"] == "refresh_2540", call_id
    assert evidence["source_commit"] == SOURCE_COMMIT, call_id


def test_retained_sites_keep_raw_hashlib():
    """A retained site still hashes with raw hashlib in its recorded scope."""
    _manifest_calls()  # Refuse missing/stale provenance before relying on raw sites.
    for rel, scopes in SCOPES.items():
        tree = ast.parse((REPO / rel).read_text())
        for scope in scopes:
            assert _uses_raw_hashlib(_scope_node(tree, scope)), rel + "::" + scope


def test_manifest_records_no_exact_owner():
    """The fresh manifest names no owner route for any retained site."""
    indexed = _manifest_calls()
    for call_id in IDS:
        row = indexed[call_id]
        assert row["owner"]["route"] is None, call_id
        assert row["owner"]["gap"], call_id


def test_manifest_keeps_gap_dispositions():
    """All six no-owner decisions are evidenced retentions, not eligible gaps."""
    indexed = _manifest_calls()
    for call_id in IDS:
        _assert_retention(indexed[call_id])
        rel, test_scope = IDS[call_id][1].split("::")
        _scope_node(ast.parse((REPO / rel).read_text()), test_scope)
    refresh = json.loads(CENSUS.read_text())["refresh_2540"]
    cohorts = refresh["contract_cohorts"]
    assert not set(IDS).intersection(call for cohort in cohorts for call in cohort["calls"])
    assert refresh["summary"]["eligible_cohorts"] == len(cohorts)
    dispositions = Counter(row["disposition"] for row in refresh["calls"])
    assert refresh["summary"]["calls_by_disposition"] == dict(dispositions)


@pytest.mark.parametrize("field,value", [
    ("source_commit", "0" * 40),
    ("generated", "1900-01-01"),
    ("schema", "obsolete"),
    ("issue", "prismaquant#1301"),
])
def test_manifest_refuses_wrong_provenance(tmp_path, monkeypatch, field, value):
    document = json.loads(CENSUS.read_text())
    document["refresh_2540"][field] = value
    census = tmp_path / "census.json"
    census.write_text(json.dumps(document))
    monkeypatch.setitem(globals(), "CENSUS", census)
    with pytest.raises(AssertionError):
        _manifest_calls()


def test_manifest_refuses_missing_refresh(tmp_path, monkeypatch):
    census = tmp_path / "census.json"
    monkeypatch.setitem(globals(), "CENSUS", census)
    with pytest.raises(FileNotFoundError):
        _manifest_calls()
    census.write_text("{}")
    with pytest.raises(KeyError, match="refresh_2540"):
        _manifest_calls()


@pytest.mark.parametrize("changed", [*SCOPES, "prismaquant/digests.py", "prismaquant/tensor_digests.py"])
def test_manifest_refuses_stale_caller_or_owner(tmp_path, monkeypatch, changed):
    # No .git directory or historical commit is needed, even in a transported tree.
    for rel in (*SCOPES, "prismaquant/digests.py", "prismaquant/tensor_digests.py"):
        destination = tmp_path / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO / rel).read_bytes())
    monkeypatch.setitem(globals(), "REPO", tmp_path)
    _manifest_calls()
    with (tmp_path / changed).open("ab") as stream:
        stream.write(b"\n")  # Leaves every raw-hashlib scope present.
    with pytest.raises(AssertionError, match="stale manifest source"):
        _manifest_calls()


@pytest.mark.parametrize("call_id", IDS)
@pytest.mark.parametrize("field,value", [
    ("disposition", "add-exact-owner-recipe"),
    ("eligible", True),
    ("cohort", "obsolete-singleton"),
    ("retention_evidence", None),
])
def test_retention_pin_rejects_recipe_gaps(call_id, field, value):
    row = deepcopy(_manifest_calls()[call_id])
    row[field] = value
    with pytest.raises(AssertionError):
        _assert_retention(row)


@pytest.mark.parametrize("field,value", [
    ("kind", "unqualified"),
    ("reference", ""),
    ("source_blob", "0" * 40),
    ("manifest", "historical"),
    ("source_commit", "0" * 40),
])
def test_retention_pin_rejects_unbound_evidence(field, value):
    row = deepcopy(_manifest_calls()[next(iter(IDS))])
    row["retention_evidence"][field] = value
    with pytest.raises(AssertionError):
        _assert_retention(row)
