"""Check the eight source-bound tensor decisions for #2638."""

import ast
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import struct

import pytest

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "docs/audits/digest_site_census_pq1301_2026-10-04.json"
STAGE = "prismaquant/stage_a_chain_seed.py"
PWC = "prismaquant/production_weight_cache.py"
EXPORT = "prismaquant/tessera_export_lane.py"
HESSIAN = "prismaquant/tessera_hessian.py"
OWNER = "prismaquant/tensor_digests.py"

# Hash only the definitions that control each contract.
BINDINGS = {
    "stage-a": {STAGE: ["tensor_payload_sha256"]},
    "codebook": {PWC: ["_cb_cache_tensor_identity", "_TENSOR_DIGEST_CHUNK_BYTES"]},
    "source-weight": {PWC: ["_source_weight_value_identity"]},
    "export": {EXPORT: [
        "hessian_capture_sha256", "HESSIAN_CAPTURE_SHA256_SCHEMA",
        "PRICED_HESSIAN_IDENTITY_FIELDS", "CAPTURE_CONTEXT_FIELDS",
        "_capture_context_fields", "_tessera_capture_context", "_require_capture_context_roster",
        "_is_hessian_reference", "_tessera_capture_seal", "_crosscheck_capture_seal",
    ]},
    "token": {HESSIAN: ["token_ids_sha256"]},
    "tensor-comparison": {
        OWNER: ["tensor_identity", "tensor_hash_update", "_host", "_raw"],
        "prismaquant/digests.py": ["bytes_sha256hex"],
    },
    "host-bytes": {OWNER: ["tensor_host_bytes", "_host", "_raw"]},
    "token-int32": {OWNER: ["token_ids_int32_sha256"]},
}
DECISIONS = {
    f"hashlib:{STAGE}::tensor_payload_sha256@421:13": ("stage-a", "tensor-comparison"),
    f"hashlib:{PWC}::_cb_cache_tensor_identity@3218:13": ("codebook", "tensor-comparison"),
    f"hashlib:{PWC}::_source_weight_value_identity@2961:13": ("source-weight", "tensor-comparison"),
    f"hashlib:{EXPORT}::hessian_capture_sha256@1605:13": ("export", "host-bytes"),
    f"hashlib:{EXPORT}::hessian_capture_sha256@1611:15": ("export", "host-bytes"),
    f"sorted-json:{EXPORT}::hessian_capture_sha256@1606:18": ("export", "host-bytes"),
    f"sorted-json:{EXPORT}::hessian_capture_sha256@1612:20": ("export", "host-bytes"),
    f"hashlib:{HESSIAN}::token_ids_sha256@114:13": ("token", "token-int32"),
}


def _read_source(path):
    return (ROOT / path).read_text()


def _source_hash(binding, read_source):
    definitions = []
    for path, names in sorted(binding.items()):
        tree = ast.parse(read_source(path))
        for name in names:
            matches = [node for node in tree.body if getattr(node, "name", None) == name
                       or isinstance(node, (ast.Assign, ast.AnnAssign))
                       and any(isinstance(part, ast.Name) and part.id == name
                               and isinstance(part.ctx, ast.Store) for part in ast.walk(node))
                       or isinstance(node, ast.ImportFrom)
                       and any((alias.asname or alias.name) == name for alias in node.names)]
            assert len(matches) == 1, f"Find one source definition: {path}::{name}."
            definitions.append((path, name, ast.dump(matches[0])))
    return _identity(definitions)


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@pytest.fixture
def tensor_census():
    return json.loads(CENSUS.read_text())["refresh_2540"]


def _unique_rows(rows, key):
    index = {row[key]: row for row in rows}
    assert len(index) == len(rows), "Remove the duplicate row."
    assert DECISIONS.keys() <= index.keys(), "Add the missing required row."
    return index


def _check_tensor_audit(refresh, read_source=_read_source):
    audit = refresh["verification"].get("tensor_retention_2638")
    assert audit, "Add the current tensor revalidation."
    assert refresh["source_commit"] == "30b6231df000726beda55fb6706e641a5cb2ccb0"
    assert refresh["generated"] == "2026-10-09"
    assert _identity(refresh["source_blobs"]) == audit["snapshot_source_sha256"]
    calls = _unique_rows(refresh["calls"], "id")
    current = _unique_rows(audit["calls"], "historical_id")
    assert current.keys() == DECISIONS.keys()
    cohorts = refresh["contract_cohorts"]
    cohort_members = Counter(rid for cohort in cohorts for rid in cohort["calls"])
    check_source = lru_cache(None)(lambda name: _source_hash(BINDINGS[name], read_source))
    for rid, (caller, owner) in DECISIONS.items():
        row, decision = calls[rid], current[rid]
        assert rid == f"{row['family']}:{row['file']}::{row['scope']}@{row['line']}:{row['column']}"
        assert row["source_blob"] == refresh["source_blobs"][row["file"]]["git_blob"]
        for role, name in (("caller", caller), ("owner", owner)):
            evidence = audit["source_evidence"].get(decision.get(role))
            assert evidence and evidence.get("ast_sha256"), f"Add {role} source evidence: {rid}."
            assert decision[role] == name and evidence.get("definitions") == BINDINGS[name], (
                f"Correct the {role} source evidence: {rid}."
            )
            assert evidence["ast_sha256"] == check_source(name), f"Review stale {role} source: {rid}."
        assert decision["checks"], rid
        for ref in decision["checks"]:
            path, test = ref.split("::")
            assert any(isinstance(node, ast.FunctionDef) and node.name == test
                       for node in ast.parse(read_source(path)).body), ref
        if caller == "token":
            assert decision["disposition"] == "existing-owner-route", rid
            assert decision["owner_route"] == "prismaquant.tensor_digests.token_ids_int32_sha256"
            assert row["disposition"] == "add-exact-owner-recipe", rid
            assert row["eligible"] is True and cohort_members[rid] == 1, rid
        else:
            assert row["disposition"] == decision["disposition"] == "retain", rid
            assert row["retention_evidence"], f"Add retention evidence: {rid}."
            assert row["eligible"] is False, f"Set retention eligibility to false: {rid}."
            assert row["cohort"] is None and rid not in cohort_members, f"Remove the eligible cohort: {rid}."
            if owner == "tensor-comparison":
                assert row["owner"]["route"] is decision["owner_route"] is None, rid
                assert row["retention_evidence"]["reference"] in decision["checks"], rid
    population = refresh["calls"] + refresh["supplemental_calls"]
    assert cohort_members == Counter(row["id"] for row in population if row["eligible"])
    assert refresh["summary"]["eligible_cohorts"] == len(cohorts), "Correct the cohort summary."
    dispositions = Counter(row["disposition"] for row in refresh["calls"])
    assert refresh["summary"]["calls_by_disposition"] == dispositions, "Correct the disposition summary."


def test_current_tensor_retention_decisions(tensor_census):
    _check_tensor_audit(tensor_census)


@pytest.mark.parametrize(("fault", "message"), [
    ("missing-row", "missing required row"),
    ("duplicate-row", "duplicate row"),
    ("caller-source", "stale caller source"),
    ("owner-source", "stale owner source"),
    ("missing-evidence", "Add caller source evidence"),
    ("wrong-evidence", "Correct the caller source evidence"),
    ("no-retention", "Add retention evidence"),
    ("eligible", "eligibility to false"),
    ("cohort", "Remove the eligible cohort"),
    ("summary", "Correct the cohort summary"),
])
def test_the_tensor_audit_refuses_invalid_evidence(tensor_census, fault, message):
    audit = tensor_census["verification"].get("tensor_retention_2638")
    assert audit, "Add the current tensor revalidation."
    rid = next(iter(DECISIONS))
    rows = audit["calls"]
    decision = next(row for row in rows if row["historical_id"] == rid)
    historical = next(row for row in tensor_census["calls"] if row["id"] == rid)
    source_edits = {
        "caller-source": (PWC, "_TENSOR_DIGEST_CHUNK_BYTES = 8 << 20", "_TENSOR_DIGEST_CHUNK_BYTES = 4 << 20"),
        "owner-source": (OWNER, 'to("cpu").contiguous()', 'to("cpu")'),
    }
    if fault == "missing-row":
        rows.remove(decision)
    elif fault == "duplicate-row":
        rows.append(decision.copy())
    elif fault == "missing-evidence":
        del audit["source_evidence"][decision["caller"]]
    elif fault == "wrong-evidence":
        decision["caller"] = "codebook"
    elif fault == "no-retention":
        historical["retention_evidence"] = None
    elif fault == "eligible":
        historical["eligible"] = True
    elif fault == "cohort":
        tensor_census["contract_cohorts"].append({"calls": [rid]})
    elif fault == "summary":
        tensor_census["summary"]["eligible_cohorts"] += 1

    def read_source(path):
        source = _read_source(path)
        if fault in source_edits and path == source_edits[fault][0]:
            _, before, after = source_edits[fault]
            assert before in source
            return source.replace(before, after)
        return source

    with pytest.raises(AssertionError, match=message):
        _check_tensor_audit(tensor_census, read_source)


def test_a_new_token_recipe_does_not_expire_the_tensor_audit(tensor_census):
    def read_source(path):
        source = _read_source(path)
        if path == OWNER:
            source += "\ndef another_token_recipe(batches):\n    return token_ids_int32_sha256(batches)\n"
        return source

    _check_tensor_audit(tensor_census, read_source)


def test_source_weight_digest_recurses_before_fp32_conversion(monkeypatch):
    import torch

    from prismaquant import production_weight_cache

    cap = 4_194_304
    width = cap // 2 + 1
    weight = torch.tensor([1.5, -2.0], dtype=torch.float64).reshape(1, 2, 1).expand(1, 2, width)
    assert not weight.is_contiguous()
    conversions = []
    tensor_to = torch.Tensor.to

    def to(tensor, *args, **kwargs):
        if kwargs.get("dtype") == torch.float32:
            assert kwargs["device"] == "cpu"
            conversions.append(tensor.numel())
        return tensor_to(tensor, *args, **kwargs)

    expected = hashlib.sha256()
    for value in (1.5, -2.0):
        item = struct.pack("<f", value)
        for _ in range(width // 262_144):
            expected.update(item * 262_144)
        expected.update(item * (width % 262_144))
    monkeypatch.setattr(torch.Tensor, "to", to)
    assert production_weight_cache._source_weight_value_identity(weight) == ([1, 2, width], expected.hexdigest())
    assert conversions == [width, width]


def test_export_capture_hash_matches_the_framed_bytes():
    import torch

    from prismaquant import tessera_export_lane as export

    identity = dict.fromkeys((*export.PRICED_HESSIAN_IDENTITY_FIELDS, *export.CAPTURE_CONTEXT_FIELDS))
    identity["model"] = "fixture"
    prefix = json.dumps({"schema": "tessera.hessian_capture.v1", "identity": identity}, sort_keys=True).encode()
    unit_prefix = b'{"dtype": "torch.uint8", "shape": [1]}\0'
    a = hashlib.sha256(unit_prefix + b"\x01").hexdigest().encode("ascii")
    z = hashlib.sha256(unit_prefix + b"\x02").hexdigest().encode("ascii")
    expected = hashlib.sha256(prefix + b"\0a\0" + a + b"\0z\0" + z).hexdigest()
    units = {"z": torch.tensor([2], dtype=torch.uint8), "a": torch.tensor([1], dtype=torch.uint8)}
    assert export.hessian_capture_sha256(units, {"model": Path("fixture")}) == expected
