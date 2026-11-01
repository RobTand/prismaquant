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
REVALIDATED_COMMIT = "914bf79807120fa60f433714e95cffd44b917d92"

# Keep the #2540 snapshot immutable. A separately reviewed integration snapshot
# admits main's surrounding refactors, not arbitrary changes to callers/owners.
# All six retained source segments are unchanged. The new file-range/JSON
# recipes still own none of their acquisition or BLAKE2b result contracts.
# Each identity is (Git blob, SHA-256); no historical Git objects are needed.
MANIFEST_SOURCE_BLOBS = {
    "prismaquant/perturbed_x_cache.py": (
        "55ee7e847de5c97dca65ed7a2c8727b297d5c479",
        "cebc40e00f04a9f411e5d8fb027adfef4f6011e8541b2acb19bdf36f496f5391",
    ),
    "prismaquant/tessera_census_cache.py": (
        "a7e64371421c65c16b3c587c8034b5a15336963e",
        "3914f446032630aeb20dd5517c1875f4453d7a024d7a820c7e6e483e2f958e7c",
    ),
    "prismaquant/digests.py": (
        "a2f8936b77a09eda702bd704c40ea3d6970ff1ef",
        "1cde1f0edd7b210beb5ff67b8efef6d06d05ae6008c84550b7b0db00a28be9b6",
    ),
    "prismaquant/tensor_digests.py": (
        "5c364c5d7c8fadf78ad9917668e72e1262546278",
        "98406374a7859e9a1b9e490671432372ad5dbe84a7a58c2b3c34377125af2037",
    ),
}
CURRENT_SOURCE_BLOBS = {
    **MANIFEST_SOURCE_BLOBS,
    # #2638/#3012: additive token recipe; existing owner profiles unchanged.
    "prismaquant/tensor_digests.py": (
        "0777758650869edc365cbf545280250e6fa175f2",
        "8baeaf7541a23481b9d233a7c65ee5f679f4fcc248c004e83ad0ca16ea8ee50a",
    ),
    "prismaquant/perturbed_x_cache.py": (
        "99ed093fdf903ecf0ca8e4023a8c71b252829bea",
        "4daf929763ce5eefc5c3ed31c0bfaa2489e29b3cf7827b5ecf17fd82b9bfdfbf",
    ),
    "prismaquant/tessera_census_cache.py": (
        "b8158910df6d22720611dd9b8588bf51777170c4",
        "c67e32264f35f0f1b2ab953e00f4dab8dfb0753ccd6cebf04e613cc4346c90f1",
    ),
    "prismaquant/digests.py": (
        "65c98d3bdeb16035ef4b104dc159e986e2279897",
        "732c5e8e8774491f190eb866f6005ed3014d4fe16c1f2d837f22a39215e90f18",
    ),
}

# The original producer test is archived, not runnable current coverage. Keep its
# historical evidence string below and check the active calibration-hash oracle.
CURRENT_REFERENCES = {
    "tests/test_sample_parallel_probe.py::test_prepare_artifact_binds_every_local_slice":
        "tests/test_tensor_digests_1384.py::test_calibration_data_hash_keeps_its_value",
}

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
    revalidation = refresh["verification"]["guarded_read_retention_2637"]["merge_revalidation"]
    assert revalidation["source_commit"] == REVALIDATED_COMMIT, "wrong revalidation source"
    assert revalidation["source_identity_reference"] == (
        "tests/test_guarded_read_digest_retention_2637.py::CURRENT_SOURCE_BLOBS"
    )
    assert revalidation["active_calibration_reference"] == next(iter(CURRENT_REFERENCES.values()))
    for rel, frozen_identity in MANIFEST_SOURCE_BLOBS.items():
        identity = refresh["source_blobs"][rel]
        assert identity["mode"] == "100644", rel
        assert (identity["git_blob"], identity["sha256"]) == frozen_identity, rel
        if rel in refresh["owner_contracts"]:
            assert refresh["owner_contracts"][rel]["source_blob"] == identity["git_blob"], rel

        path = REPO / rel
        assert not path.is_symlink(), rel
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        assert (blob, hashlib.sha256(raw).hexdigest()) == CURRENT_SOURCE_BLOBS[rel], (
            f"stale manifest source: {rel}"
        )

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
        reference = IDS[call_id][1]
        rel, test_scope = CURRENT_REFERENCES.get(reference, reference).split("::")
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


@pytest.mark.parametrize("rel", MANIFEST_SOURCE_BLOBS)
@pytest.mark.parametrize("field,value", [
    ("git_blob", "0" * 40),
    ("sha256", "0" * 64),
    ("mode", "100755"),
])
def test_manifest_refuses_rewritten_snapshot_identity(tmp_path, monkeypatch, rel, field, value):
    document = json.loads(CENSUS.read_text())
    document["refresh_2540"]["source_blobs"][rel][field] = value
    census = tmp_path / "census.json"
    census.write_text(json.dumps(document))
    monkeypatch.setitem(globals(), "CENSUS", census)
    with pytest.raises(AssertionError):
        _manifest_calls()


@pytest.mark.parametrize("missing", [False, True])
def test_manifest_requires_reviewed_integration_source(tmp_path, monkeypatch, missing):
    document = json.loads(CENSUS.read_text())
    retention = document["refresh_2540"]["verification"]["guarded_read_retention_2637"]
    if missing:
        del retention["merge_revalidation"]
    else:
        retention["merge_revalidation"]["source_commit"] = "0" * 40
    census = tmp_path / "census.json"
    census.write_text(json.dumps(document))
    monkeypatch.setitem(globals(), "CENSUS", census)
    with pytest.raises(KeyError if missing else AssertionError):
        _manifest_calls()


@pytest.mark.parametrize("changed", CURRENT_SOURCE_BLOBS)
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
