"""Pin the guarded-read digest retentions of prismaquant#2637."""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CENSUS = REPO / "docs/audits/digest_site_census_pq1301_2026-10-04.json"

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

IDS = [
    "hashlib:prismaquant/perturbed_x_cache.py::SerializedEntryDigest.__init__@207:21",
    "hashlib:prismaquant/perturbed_x_cache.py::_read_exact_entry@2117:14",
    "hashlib:prismaquant/perturbed_x_cache.py::_seed_from@2405:13",
    "hashlib:prismaquant/perturbed_x_cache.py::calibration_data_hash@2375:8",
    "hashlib:prismaquant/perturbed_x_cache.py::load_verified_activation_cache_entry@903:17",
    "hashlib:prismaquant/tessera_census_cache.py::_verify_dense_blob@245:13",
]


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
    document = json.loads(CENSUS.read_text())
    calls = document["refresh_2540"]["calls"]
    indexed = {row["id"]: row for row in calls}
    assert len(indexed) == len(calls), "duplicate manifest call"
    return indexed


def test_retained_sites_keep_raw_hashlib():
    """A retained site still hashes with raw hashlib in its recorded scope."""
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
    """No retained site silently gains a migrate disposition."""
    indexed = _manifest_calls()
    for call_id in IDS:
        row = indexed[call_id]
        assert row["disposition"] in {"add-exact-owner-recipe", "retain"}, call_id
        if row["disposition"] == "retain":
            evidence = row["retention_evidence"]
            assert evidence and evidence["reference"], call_id
