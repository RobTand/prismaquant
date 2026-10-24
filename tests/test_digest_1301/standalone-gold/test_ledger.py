"""Local exact-call reconciliation; does not change the default ratchet population."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from tools import duplication_inventory as inventory

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "tests/fixtures/digest_1301/standalone-gold/ledger.json"


def test_local_ledger_accounts_for_primary_and_supplemental_calls():
    ledger = json.loads(LEDGER.read_text())
    live = {}
    for relative in ledger["scope_files"]:
        path = ROOT / relative
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        tree = ast.parse(raw.decode())
        for call, scope, aliases in inventory._iter_scoped_calls(tree):
            callee = inventory._resolved_callee(call.func, aliases)
            if callee is None:
                continue
            if callee.startswith("hashlib."):
                family = "hashlib"
            elif callee in {"json.dumps", "json.JSONEncoder", "json.dump"}:
                family = "sorted-json" if callee != "json.dump" and any(
                    k.arg == "sort_keys" and isinstance(k.value, ast.Constant)
                    and k.value.value is True for k in call.keywords
                ) else "supplemental-json"
            else:
                continue
            key = f"{family}:{relative}::{scope}@{call.lineno}:{call.col_offset}"
            live[key] = (blob, callee)
    rows = ledger["calls"] + ledger["supplemental_calls"]
    assert len(rows) == len({r["id"] for r in rows})
    assert {r["id"] for r in rows} == set(live)
    for row in rows:
        assert (row["source_blob"], row["callee"]) == live[row["id"]]
        assert row["original_id"] == row["id"]
        assert row["classification"] in {"load-bearing", "ephemeral"}
        assert row["disposition"] == "retained"
        assert row["recipe"] and row["consumer_evidence"] and row["witnesses"]
    # No production substitutions are made in this retention-only slice.
    assert ledger["completed_calls"] == []


def test_ledger_witness_nodes_are_executable_and_not_parent_acceptance():
    ledger = json.loads(LEDGER.read_text())
    for row in ledger["calls"] + ledger["supplemental_calls"]:
        assert row["fixtures"] and row["tests"]
        for fixture in row["fixtures"]:
            assert (ROOT / fixture).is_file()
        for node in row["tests"]:
            filename, function = node.split("::")
            assert filename.startswith("tests/test_")
            assert "test_digest_parent_completion_1301" not in filename
            assert "[" not in function
            tree = ast.parse((ROOT / filename).read_text())
            assert any(isinstance(n, ast.FunctionDef) and n.name == function
                       for n in tree.body), node
