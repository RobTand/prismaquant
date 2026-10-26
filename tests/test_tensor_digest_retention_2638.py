"""Source-bound tensor retention audit for #3020 (continuing #2638).

Only test/fixture changes belong to research. The checked-in census is the
positive acceptance input; _example_document is exclusively a negative-control
fixture, never evidence that main's audit is complete. Old file identities and
coordinates stay historical. Current contracts are AST SHA256s, independent of
line shifts, whole-file provenance, and unrelated new token-owner functions.
Existing #1384 tensor and #1599 export-delegation goldens are preserved;
neither is relabeled as prior-attempt capture qualification.
"""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
import struct
import textwrap

import pytest

REPO = Path(__file__).resolve().parents[1]
CENSUS = REPO / "docs/audits/digest_site_census_pq1301_2026-10-04.json"
FIXTURE = json.loads((REPO / "tests/fixtures/tensor_digest_retention_2638.json").read_text())
KEY = "tensor_retention_2638"
IDS = FIXTURE["snapshot_calls"]
THREE = {rid for rid, row in IDS.items() if row["scope"] in {
    "tensor_payload_sha256", "_cb_cache_tensor_identity", "_source_weight_value_identity",
}}
TOKEN = next(rid for rid, row in IDS.items() if row["scope"] == "token_ids_sha256")
RETAINED = set(IDS) - {TOKEN}
OWNER_KEYS = [key for key in FIXTURE["contracts"] if key.startswith((
    "prismaquant/tensor_digests.py::", "prismaquant/digests.py::",
))]
CONSUMERS = {
    "token_ids_sha256": [
        "tests/test_tessera_campaign.py::test_the_token_sha_identifies_the_calibration_draw",
    ],
    "tensor_payload_sha256": [
        "tests/test_stage_a_chain_seed.py::test_the_plane_distance_of_two_seeds",
    ],
    "_cb_cache_tensor_identity": [
        "tests/test_render_identity_once_1192.py::test_the_loader_hash_equals_the_main_thread_hash",
        "tests/test_pwc_file_load_receipts.py::test_receipt_hashes_exact_single_read_and_tensor",
    ],
    "_source_weight_value_identity": [
        "tests/test_tensor_digest_retention_2638.py::test_source_identity_preserves_c_order_fp32_and_schema",
        "tests/test_production_weight_cache.py::test_prefetch_loads_disk_entries_and_respects_lru",
    ],
    "hessian_capture_sha256": [
        "tests/test_tensor_digest_retention_2638.py::test_export_keeps_payload_delegation_and_independent_framing",
        "tests/test_tessera_priced_export_inputs.py::test_the_digest_rule_is_tesseras_capture_seal",
        "tests/test_tessera_priced_export_inputs.py::test_a_drift_between_the_two_seal_rules_refuses",
        "tests/test_tessera_priced_export_inputs.py::test_the_gate_refuses_a_bind_under_a_drifted_roster",
        "tests/test_tessera_priced_export_inputs.py::test_a_payload_without_hessians_cannot_be_bound",
    ],
}


def _node(root, key):
    rel, name = key.split("::")
    path = root / rel
    assert not path.is_symlink(), key
    tree = ast.parse(path.read_text())
    matches = [n for n in tree.body if getattr(n, "name", None) == name or (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
    )]
    assert len(matches) == 1, f"missing/duplicate source contract: {key}"
    return matches[0]


def _contract_sha(root, key):
    return hashlib.sha256(ast.dump(_node(root, key), include_attributes=False).encode()).hexdigest()


def _decision(row):
    return {name: row[name] for name in ("disposition", "eligible", "cohort")}


def _refresh(document):
    return document["refresh_2540"]


def _index(refresh):
    calls = refresh["calls"]
    indexed = {row["id"]: row for row in calls}
    assert len(indexed) == len(calls), "duplicate required/audit row"
    assert set(IDS) <= indexed.keys(), "missing required tensor row"
    return indexed


def _assert_snapshot(refresh):
    assert refresh["schema"] == "prismaquant.digest-census-refresh.v1"
    assert refresh["source_commit"] == FIXTURE["snapshot_source_commit"]
    assert refresh["generated"] == FIXTURE["snapshot_generated"]
    indexed = _index(refresh)
    for rel, identity in FIXTURE["snapshot_source_blobs"].items():
        assert refresh["source_blobs"][rel] == identity, f"rewritten historical source: {rel}"
        if rel in refresh["owner_contracts"]:
            assert refresh["owner_contracts"][rel]["source_blob"] == identity["git_blob"]
    for rid, frozen in IDS.items():
        assert {field: indexed[rid][field] for field in frozen} == frozen, rid


def _assert_dispositions(refresh):
    indexed = _index(refresh)
    for rid in RETAINED:
        row = indexed[rid]
        assert _decision(row) == {"disposition": "retain", "eligible": False, "cohort": None}, rid
        assert row["disposition_reason"], rid
        evidence = row["retention_evidence"]
        assert evidence and evidence["kind"] and evidence["reference"], f"missing retention evidence: {rid}"
        assert evidence["source_blob"] == row["source_blob"], rid
        if rid in THREE:
            assert row["owner"]["route"] is None and row["owner"]["gap"], rid
            assert evidence["manifest"] == "refresh_2540"
            assert evidence["source_commit"] == FIXTURE["snapshot_source_commit"]
            assert evidence["current_revalidation"] == KEY
    # Export's pre-existing exact JSON route may remain recorded: retaining the
    # independent reference is not the same thing as claiming no route exists.
    rows = refresh["calls"] + refresh["supplemental_calls"]
    cohorts = refresh["contract_cohorts"]
    assert len({cohort["id"] for cohort in cohorts}) == len(cohorts)
    members = [rid for cohort in cohorts for rid in cohort["calls"]]
    assert len(members) == len(set(members)), "duplicate cohort member"
    assert not RETAINED.intersection(members), "retention in eligible cohort"
    assert set(members) == {row["id"] for row in rows if row["eligible"]}, "eligible cohort mismatch"
    for rid in IDS:
        row = indexed[rid]
        modules = [m for m in refresh["module_owners"] if rid in m["calls"]]
        assert len(modules) == 1 and modules[0]["file"] == row["file"], "lost module membership"
        assert row["site"] in refresh["scan_sites"] and row["site"] in refresh["ratchet_sites"]
    summary = refresh["summary"]
    assert summary["eligible_cohorts"] == len(cohorts), "inconsistent cohort summary"
    assert summary["individual_calls"] == len(refresh["calls"])
    assert summary["supplemental_calls"] == len(refresh["supplemental_calls"])
    for field in ("disposition", "family", "classification"):
        assert summary["calls_by_" + field] == dict(Counter(row[field] for row in refresh["calls"])), field


def _audit(document, root=REPO):
    refresh = _refresh(document)
    _assert_snapshot(refresh)
    _assert_dispositions(refresh)
    audit = refresh["verification"][KEY]
    assert audit["historical_manifest"] == "refresh_2540"
    assert audit["source_commit"] == FIXTURE["reviewed_source_commit"]
    assert audit["scope"] == "eight tensor calls only; not a whole-repository regeneration"
    # These identify the inspected files, not an equality assertion on today's
    # whole files. The current relevant contracts are validated separately below.
    assert audit["source_files"] == FIXTURE["reviewed_source_files"]
    contracts = audit["contracts"]
    assert set(FIXTURE["contracts"]) <= set(contracts), "missing source evidence"
    for key, frozen in FIXTURE["contracts"].items():
        actual = _contract_sha(root, key)
        assert contracts[key] == actual, f"stale relevant caller/owner source: {key}"
        # #3012 owns token migration, not this retention leaf. It must rebind
        # token current evidence if it changes that function; other contracts
        # remain invariant even when an owner gains an unrelated token recipe.
        if key != "prismaquant/tessera_hessian.py::token_ids_sha256":
            assert actual == frozen, f"changed retained/owner contract: {key}"
    current = audit["calls"]
    assert len(current) == len(IDS) and {row["id"] for row in current} == set(IDS), "missing/duplicate current row"
    indexed = _index(refresh)
    for row in current:
        rid = row["id"]
        historical = indexed[rid]
        assert row["caller_contract"] == historical["file"] + "::" + historical["scope"]
        assert set(OWNER_KEYS) <= set(row["owner_contracts"]), "missing owner evidence"
        assert row["decision"] == _decision(historical), "current decision disagrees with census"
        assert row["reason"], "missing current rationale"
        assert set(CONSUMERS[historical["scope"]]) <= set(row["consumer_checks"]), "missing consumer evidence"
        for reference in row["consumer_checks"]:
            _node(REPO, reference)  # Active executable test, not an archived test string.
    for rid in THREE:
        site = IDS[rid]["site"]
        old = [row for row in document["rows"] if row["site"] == site]
        assert len(old) == 1
        assert old[0][KEY] == {
            "refresh_call_ids": [rid], "disposition": "retain",
            "revalidation": "refresh_2540.verification." + KEY,
        }, "missing historical-row annotation"


def _example_document():
    """A locally compliant *test fixture* so negative controls cannot pass vacuously."""
    document = json.loads(CENSUS.read_text())
    refresh = _refresh(document)
    indexed = _index(refresh)
    for rid in THREE:
        row = indexed[rid]
        row.update(disposition="retain", eligible=False, cohort=None,
                   disposition_reason="Negative-control fixture: preserve bounded local contract")
        row["retention_evidence"] = {
            "kind": "bounded-tensor-contract", "reference": CONSUMERS[row["scope"]][0],
            "source_blob": row["source_blob"], "source_commit": FIXTURE["snapshot_source_commit"],
            "manifest": "refresh_2540", "current_revalidation": KEY,
        }
        old = next(old for old in document["rows"] if old["site"] == row["site"])
        old[KEY] = {"refresh_call_ids": [rid], "disposition": "retain",
                    "revalidation": "refresh_2540.verification." + KEY}
    refresh["contract_cohorts"] = [c for c in refresh["contract_cohorts"] if not THREE.intersection(c["calls"])]
    refresh["summary"]["eligible_cohorts"] = len(refresh["contract_cohorts"])
    refresh["summary"]["calls_by_disposition"] = dict(Counter(row["disposition"] for row in refresh["calls"]))
    refresh["verification"][KEY] = {
        "historical_manifest": "refresh_2540", "source_commit": FIXTURE["reviewed_source_commit"],
        "scope": "eight tensor calls only; not a whole-repository regeneration",
        "source_files": deepcopy(FIXTURE["reviewed_source_files"]),
        "contracts": {key: _contract_sha(REPO, key) for key in FIXTURE["contracts"]},
        "calls": [{"id": rid, "caller_contract": row["file"] + "::" + row["scope"],
                   "owner_contracts": list(OWNER_KEYS), "decision": _decision(row),
                   "consumer_checks": CONSUMERS[row["scope"]], "reason": "Negative-control fixture"}
                  for rid, row in indexed.items() if rid in IDS],
    }
    return document


def test_integrated_tensor_audit_is_source_bound():
    _audit(json.loads(CENSUS.read_text()))


def test_stage_a_pwc_are_evidenced_retentions_not_recipe_gaps():
    _assert_dispositions(_refresh(json.loads(CENSUS.read_text())))


@pytest.mark.parametrize("damage", [
    "missing-row", "duplicate-row", "missing-current-row", "duplicate-current-row",
    "missing-contract", "stale-contract", "missing-owner", "missing-consumer", "missing-revalidation",
    "missing-evidence", "eligible-retention", "cohort-retention", "leftover-cohort",
    "summary-disposition", "summary-cohort", "rewritten-snapshot", "rewritten-coordinate",
    "missing-module", "missing-annotation", "missing-current-reason", "stale-decision",
])
def test_audit_negative_controls_reject_incomplete_or_inconsistent_evidence(damage):
    document = _example_document()
    _audit(document)  # Every negative control first proves the pristine fixture is accepted.
    refresh = _refresh(document)
    rid = sorted(THREE)[0]
    row = _index(refresh)[rid]
    audit = refresh["verification"][KEY]
    current = next(r for r in audit["calls"] if r["id"] == rid)
    if damage == "missing-row":
        refresh["calls"].remove(row)
    elif damage == "duplicate-row":
        refresh["calls"].append(deepcopy(row))
    elif damage == "missing-current-row":
        audit["calls"].remove(current)
    elif damage == "duplicate-current-row":
        audit["calls"][-1] = deepcopy(audit["calls"][0])
    elif damage == "missing-contract":
        del audit["contracts"][OWNER_KEYS[0]]
    elif damage == "stale-contract":
        audit["contracts"][OWNER_KEYS[0]] = "0" * 64
    elif damage == "missing-owner":
        current["owner_contracts"] = []
    elif damage == "missing-consumer":
        current["consumer_checks"] = []
    elif damage == "missing-revalidation":
        del refresh["verification"][KEY]
    elif damage == "missing-evidence":
        row["retention_evidence"] = None
    elif damage == "eligible-retention":
        row["eligible"] = True
    elif damage == "cohort-retention":
        row["cohort"] = "obsolete-singleton"
    elif damage == "leftover-cohort":
        refresh["contract_cohorts"].append({"id": "obsolete-singleton", "calls": [rid]})
    elif damage == "summary-disposition":
        refresh["summary"]["calls_by_disposition"]["retain"] -= 1
    elif damage == "summary-cohort":
        refresh["summary"]["eligible_cohorts"] += 1
    elif damage == "rewritten-snapshot":
        refresh["source_blobs"][row["file"]]["sha256"] = "0" * 64
    elif damage == "rewritten-coordinate":
        row["line"] += 1
    elif damage == "missing-module":
        next(m for m in refresh["module_owners"] if rid in m["calls"])["calls"].remove(rid)
    elif damage == "missing-annotation":
        del next(old for old in document["rows"] if old["site"] == row["site"])[KEY]
    elif damage == "missing-current-reason":
        current["reason"] = ""
    elif damage == "stale-decision":
        current["decision"]["eligible"] = True
    with pytest.raises((AssertionError, KeyError)):
        _audit(document)


@pytest.mark.parametrize("key", list(FIXTURE["contracts"]))
def test_relevant_source_drift_refuses_without_historical_git(tmp_path, key):
    document = _example_document()
    for rel in FIXTURE["reviewed_source_files"]:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / rel).read_bytes())
    _audit(document, tmp_path)
    rel, _ = key.split("::")
    path = tmp_path / rel
    tree = ast.parse(path.read_text())
    target = _node(tmp_path, key)
    # Change the relevant AST, not merely its line number or unrelated text.
    node = next(n for n in tree.body if ast.dump(n) == ast.dump(target))
    if isinstance(node, ast.FunctionDef):
        node.body.append(ast.Pass())
    else:
        node.value = ast.Constant(value="changed contract")
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    with pytest.raises(AssertionError, match="stale relevant caller/owner source"):
        _audit(document, tmp_path)


def test_unrelated_token_owner_addition_is_not_retention_staleness(tmp_path):
    document = _example_document()
    for rel in FIXTURE["reviewed_source_files"]:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / rel).read_bytes())
    with (tmp_path / "prismaquant/tensor_digests.py").open("a") as stream:
        stream.write("\n\ndef unrelated_token_recipe_for_negative_control():\n    return None\n")
    _audit(document, tmp_path)


class _HashFeeds:
    """Observe feed size/type only; never retain tensor storage or memoryviews."""
    def __init__(self, real):
        self.real = real
        self.feeds = []

    def factory(self, *args, **kwargs):
        digest = self.real(*args, **kwargs)
        observer = self

        class Hash:
            def update(self, value):
                observer.feeds.append((type(value).__name__, memoryview(value).nbytes))
                digest.update(value)

            def hexdigest(self):
                return digest.hexdigest()

        return Hash()


def test_stage_a_preserves_payload_only_caller_sized_windows(monkeypatch):
    import torch
    from prismaquant import stage_a_chain_seed as stage

    assert inspect.signature(stage.tensor_payload_sha256).parameters["chunk_bytes"].default == 64 << 20
    tensor = torch.tensor([[1., -2., 3.], [4., 5., 6.]], dtype=torch.float32).T
    expected = hashlib.sha256(struct.pack("=6f", 1, 4, -2, 5, 3, 6)).hexdigest()
    spy = _HashFeeds(hashlib.sha256)
    monkeypatch.setattr(hashlib, "sha256", spy.factory)
    assert stage.tensor_payload_sha256(tensor, chunk_bytes=9) == expected
    assert spy.feeds == [("ndarray", 8)] * 3
    spy.feeds.clear()
    assert stage.tensor_payload_sha256(torch.empty(0), chunk_bytes=9) == spy.real(b"").hexdigest()
    assert spy.feeds == []


def test_codebook_identity_preserves_memoryview_windows_empty_and_receipt(monkeypatch):
    import torch
    from prismaquant import production_weight_cache as pwc

    assert pwc._TENSOR_DIGEST_CHUNK_BYTES == 8 << 20
    tensor = torch.tensor([[1., -2., 3.], [4., 5., 6.]], dtype=torch.float32).T
    expected = hashlib.sha256(struct.pack("=6f", 1, 4, -2, 5, 3, 6)).hexdigest()
    spy = _HashFeeds(hashlib.sha256)
    monkeypatch.setattr(hashlib, "sha256", spy.factory)
    monkeypatch.setattr(pwc, "_TENSOR_DIGEST_CHUNK_BYTES", 7)
    assert pwc._cb_cache_tensor_identity(tensor) == {
        "shape": [3, 2], "dtype": "torch.float32", "logical_bytes": 24, "content_sha256": expected,
    }
    assert spy.feeds == [("memoryview", 7)] * 3 + [("memoryview", 3)]
    spy.feeds.clear()
    assert pwc._cb_cache_tensor_identity(torch.empty(2, 0)) == {
        "shape": [2, 0], "dtype": "torch.float32", "logical_bytes": 0,
        "content_sha256": spy.real(b"").hexdigest(),
    }
    assert spy.feeds == []


@pytest.mark.parametrize("layout", ["transpose", "strided", "empty", "scalar"])
def test_source_identity_preserves_c_order_fp32_and_schema(layout):
    import torch
    from prismaquant import production_weight_cache as pwc

    values = {"transpose": ([1, 4, -2, 5, 3, 6], [3, 2]),
              "strided": ([1, 3, 4, 6], [2, 2]), "empty": ([], [2, 0]), "scalar": ([1.5], [])}
    base = torch.tensor([[1., -2., 3.], [4., 5., 6.]], dtype=torch.float64)
    tensor = {"transpose": base.T, "strided": base[:, ::2],
              "empty": torch.empty(2, 0), "scalar": torch.tensor(1.5)}[layout]
    numbers, shape = values[layout]
    expected = hashlib.sha256(struct.pack("<" + "f" * len(numbers), *numbers)).hexdigest()
    result = pwc._source_weight_value_identity(tensor)
    assert isinstance(result, tuple) and len(result) == 2
    assert result == (shape, expected)


def test_nearby_owner_has_a_different_schema_acquisition_and_framing():
    import torch
    from prismaquant import production_weight_cache as pwc
    from prismaquant import tensor_digests as owner

    tensor = torch.tensor([[1., -2.], [3., 4.]])
    raw = struct.pack("=4f", 1, -2, 3, 4)
    payload = hashlib.sha256(raw).hexdigest()
    assert owner.tensor_identity(tensor) == {
        "dtype": "torch.float32", "shape": [2, 2], "sha256": payload,
    }
    assert set(pwc._cb_cache_tensor_identity(tensor)) == {
        "shape", "dtype", "logical_bytes", "content_sha256",
    }
    digest = hashlib.sha256()
    owner.tensor_hash_update(digest, tensor)
    assert digest.hexdigest() == hashlib.sha256(b"(2, 2)torch.float32" + raw).hexdigest()
    assert digest.hexdigest() != payload
    # Return type is a complete bytes allocation, not an incremental feed API.
    assert type(owner.tensor_host_bytes(tensor)) is bytes


def test_source_recursive_conversion_bound_with_tiny_instrumented_ast(monkeypatch):
    """Execute the unchanged algorithm with only its local budget lowered to 4.

    This is a test-only AST transformation, not a production seam or model-sized
    allocation. The original 4,194,304-element threshold is asserted first.
    Both row-batching and recursive descent run on 24-element noncontiguous data.
    """
    import torch
    from prismaquant import production_weight_cache as pwc

    tree = ast.parse(textwrap.dedent(inspect.getsource(pwc._source_weight_value_identity)))
    function = tree.body[0]
    budget = next(n for n in function.body if isinstance(n, ast.Assign) and
                  any(isinstance(t, ast.Name) and t.id == "max_chunk_elements" for t in n.targets))
    assert ast.unparse(budget.value) == "4 * 1024 * 1024"
    budget.value = ast.Constant(4)
    namespace = {"torch": torch, "math": math, "hashlib": hashlib}
    exec(compile(ast.fix_missing_locations(tree), "<test-only-small-budget>", "exec"), namespace)
    identity = namespace[function.name]
    real_to = torch.Tensor.to
    widths = []

    def bounded_to(tensor, *args, **kwargs):
        if kwargs.get("dtype") == torch.float32:
            widths.append(tensor.numel())
            assert tensor.numel() <= 4
        return real_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", bounded_to)
    for tensor in (torch.arange(24, dtype=torch.float64).reshape(2, 3, 4).transpose(1, 2),
                   torch.arange(24, dtype=torch.float64).reshape(12, 2)):
        numbers = tensor.reshape(-1).tolist()
        expected = hashlib.sha256(struct.pack("<24f", *numbers)).hexdigest()
        assert identity(tensor) == (list(tensor.shape), expected)
    assert widths and max(widths) <= 4 and sum(widths) == 48


def test_export_keeps_payload_delegation_and_independent_framing(monkeypatch):
    import torch
    from prismaquant import tensor_digests as owner
    from prismaquant import tessera_export_lane as export

    # Independent stdlib framing, not the owner under test or a new golden
    # recorded by running a modified implementation.
    tensors = {"z": torch.tensor([[3.]]), "a": torch.tensor([[1., 2.], [4., 5.]]).T}
    provenance = {"text_sha256": "draw", "fit_tokens": 2, "fit_ids_sha256": "ids",
                  "model": "fixture", "seqlen": 2, "source": "fixture"}
    identity = {field: provenance.get(field) for field in
                (*export.PRICED_HESSIAN_IDENTITY_FIELDS, *export.CAPTURE_CONTEXT_FIELDS)}
    outer = hashlib.sha256(json.dumps({"schema": "tessera.hessian_capture.v1", "identity": identity},
                                     sort_keys=True, default=str).encode())
    for name, shape, numbers in (("a", [2, 2], [1, 4, 2, 5]), ("z", [1, 1], [3])):
        unit = hashlib.sha256(json.dumps({"dtype": "torch.float32", "shape": shape}, sort_keys=True).encode())
        unit.update(b"\0")
        unit.update(struct.pack("=" + "f" * len(numbers), *numbers))
        outer.update(b"\0" + name.encode() + b"\0")
        outer.update(unit.hexdigest().encode("ascii"))
    real = owner.tensor_host_bytes
    calls = []

    def observe(tensor):
        calls.append(tuple(tensor.shape))
        return real(tensor)

    monkeypatch.setattr(owner, "tensor_host_bytes", observe)
    assert export.hessian_capture_sha256(tensors, provenance) == outer.hexdigest()
    assert calls == [(2, 2), (1, 1)]
    # Native raw payloads follow the current CPU ABI; the numerical source
    # identity above instead explicitly specifies little-endian <f4 bytes.
