"""The PQ #2530 qualification record of GLM MTP layer 45 is complete, consistent and hash-bound.

The record is ``audit.json``, ``units.csv`` and ``cells.csv`` under the manifest's
``record_dir``. ``tools/audit_glm_mtp_layer45.py`` writes them from the original
capture, price and PrismaBuild files, and ``audit.json`` is byte for byte the CAS
payload of the audit action. These tests need no fleet mount: they check the
committed files against the manifest, the offline validator against the record
and against damaged copies of it, and the verifier's own helpers on synthetic
trees.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import re
from pathlib import Path

import pytest

from tools import audit_glm_mtp_layer45 as audit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "measurements" / "pq-2530-mtp-layer45-manifest.json"
REPORT = ROOT / "docs" / "measurements" / "pq-2530-mtp-layer45-qualification.md"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_bytes())


@pytest.fixture(scope="module")
def record(manifest):
    base = ROOT / manifest["record_dir"]

    def table(name):
        return list(csv.DictReader(io.StringIO((base / name).read_text())))

    return json.loads((base / "audit.json").read_bytes()), table("units.csv"), table("cells.csv")


# --------------------------------------------------------------------------- the committed record
def test_manifest_binds_every_record_file_and_the_audit_payload(manifest, record):
    base = ROOT / manifest["record_dir"]
    for name, digest in manifest["record_files"].items():
        assert _sha(base / name) == digest, name
    summary, _units, _cells = record
    assert _sha(base / "audit.json") == manifest["audit_action"]["payload_sha256"]
    assert summary["tables"]["units"]["sha256"] == manifest["record_files"]["units.csv"]
    assert summary["tables"]["cells"]["sha256"] == manifest["record_files"]["cells.csv"]
    assert summary["tool"]["sha256"] == manifest["tool"]["sha256"] == _sha(ROOT / manifest["tool"]["path"])


def test_the_record_is_consistent(record):
    summary, units, cells = record
    assert audit.validate_record(summary, units, cells) == []


def test_every_roster_equals_the_authenticated_census_or_its_named_subset(record):
    summary, _units, _cells = record
    names = {row["name"]: row for row in summary["rosters"]}
    census = {"capture_entries", "m3_r1024_costs", "m3_r896_costs", "m4_r1024_price", "m4_r896_price",
              "merged_price", "merged_wire_bytes", "merged_params"}
    assert census <= set(names)
    assert all(names[n]["count"] == 867 and names[n]["scope"] == "census" for n in census)
    for rate in ("r1024", "r896"):
        wires = names[f"m3_{rate}_expert_wires"]
        assert (wires["scope"], wires["count"]) == ("routed", 864)
    assert all(row["equals"] for row in names.values())


def test_the_cell_counts_reconcile_by_rung_and_class(record):
    summary, _units, cells = record
    assert len(cells) == 3468 == 867 * 4
    receipts = {kind: sum(c["receipt"] == kind for c in cells) for kind in ("expert_wires+checkpoint", "checkpoint")}
    # 3468 priced cells: 3456 routed cells carry an expert-wire receipt, and the 12 shared cells do not.
    assert receipts == {"expert_wires+checkpoint": 3456, "checkpoint": 12}
    assert 3456 == 864 * 4 and 12 == 3 * 4
    by_rung = {(m["rung"], m["class"]): m for m in summary["rung_matrix"]}
    for rung in audit.RUNG_ORDER:
        assert by_rung[(rung, "routed")]["priced_units"] == 864
        assert by_rung[(rung, "shared")]["priced_units"] == 3
        assert by_rung[(rung, "routed")]["expert_wire_receipts"] == 864
        assert by_rung[(rung, "shared")]["expert_wire_receipts"] == 0
    assert sum(int(c["offered_current"]) for c in cells) == 2604
    assert sum(int(c["offered_m6"]) for c in cells) == 1734
    # BF16 at R896 is the one priced rung the repo-pinned contract does not offer, and only for routed units.
    assert {(c["class"], c["rung"]) for c in cells if c["offered_current"] == "0"} == {
        ("routed", "TESSERA_BF16_K1_R896")}


def test_each_identity_family_is_bound_across_the_receipts(record):
    summary, _units, _cells = record
    kinds = {row["kind"]: row for row in summary["identities"]}
    for family in ("source", "calibration", "probe", "producer", "runtime"):
        assert any(kind.startswith(family + ".") for kind in kinds), family
    one_value = [kind for kind, row in kinds.items() if row["distinct"] == 1]
    split = [kind for kind, row in kinds.items() if row["distinct"] > 1]
    assert split == ["producer.prismaquant_source_sha256"] and kinds[split[0]]["split_allowed"]
    assert {"source.shard_digests_sha256", "calibration.fit_ids_sha256", "probe.identity_sha256",
            "producer.encoder_source_sha256", "runtime.container_content_sha256"} <= set(one_value)
    pair = {v["value"] for v in kinds["producer.prismaquant_source_sha256"]["values"]}
    assert len(pair) == 2 and {v["value"] for v in kinds["probe.producer_source_sha256"]["values"]} <= pair


def test_every_suspended_gate_of_the_original_runs_is_listed_and_resolved(record):
    summary, _units, _cells = record
    kinds = {}
    for line in summary["dev_ledger"]:
        kinds[line["kind"]] = kinds.get(line["kind"], 0) + 1
        assert line["resolution"]["status"] in {"explained", "verified_here", "limitation"}
    assert kinds == {"prepared_plan": 2, "prepared_implementation": 2, "checkpoint_seal": 4,
                     "projection_shape": 4, "source_rehash": 1}
    assert all(stamp["dev_uncertified"] for stamp in summary["dev_stamps"].values())
    assert summary["trees"]["r1024_prepare_vs_run"]["changed_files"] == [
        "cost_streaming.py", "tessera_calibration_cache.py", "tessera_joint_aura.py",
        "tessera_source_digest_adoption.py"]


def test_the_execution_record_names_every_action_with_a_verified_cas_chain(record):
    summary, _units, _cells = record
    roles = {a["role"] for a in summary["actions"]}
    assert {"capture-v2", "projection", "final-hidden", "m4-r1024-prepare", "m4-r1024-run",
            "m4-r896-prepare", "m4-r896-run"} <= roles
    assert {f"m3-{rate}-row-000{i}" for rate in ("r1024", "r896") for i in range(3)} <= roles
    for action in summary["actions"]:
        assert action["status"] == "executed" and action["returncode"] == 0 and action["ok"]
        assert action["receipt"]["sha256"] != action["stdout"]["sha256"]
        assert action["payload"]["sha256"] == action["receipt"]["result_sha256"] == action["blob"]["sha256"]
    assert len(summary["excluded_attempts"]) == 8
    assert {a["role"] for a in summary["excluded_attempts"]} >= {"capture-v1-superseded", "m4-r896-attempt1-withdrawn"}


def test_no_performance_claim_is_made(record, manifest):
    summary, _units, _cells = record
    assert summary["performance_claims"] == [] and manifest["performance_claims"] == []


def test_the_report_cites_the_record_it_rests_on(manifest, record):
    text = REPORT.read_text()
    summary, _units, _cells = record
    assert manifest["audit_action"]["key"] in text and manifest["audit_action"]["payload_sha256"] in text
    for digest in manifest["record_files"].values():
        assert digest in text
    for row in summary["rung_matrix"]:
        assert row["rung"] in text
    for action in summary["actions"]:
        assert action["key"][:12] in text, action["role"]


# --------------------------------------------------------------------------- the validator rejects damaged records
def _damaged(record, how):
    summary, units, cells = (copy.deepcopy(part) for part in record)
    how(summary, units, cells)
    return audit.validate_record(summary, units, cells)


@pytest.mark.parametrize("name,how", [
    ("a unit is missing", lambda s, u, c: u.pop()),
    ("a unit has no capture entry", lambda s, u, c: u[0].update(capture_sha256="")),
    ("a hessian is the wrong width", lambda s, u, c: u[0].update(hessian_shape="1x1")),
    ("a cell is missing", lambda s, u, c: c.pop()),
    ("a wire failed the producer", lambda s, u, c: c[5].update(producer_verified="0")),
    ("a wire differs from the prepared evidence", lambda s, u, c: c[7].update(prepared_wire_match="0")),
    ("a rendered shard differs", lambda s, u, c: c[9].update(prepared_render_match="0")),
    ("file bytes differ from serialized bytes", lambda s, u, c: c[11].update(wire_file_bytes="1")),
    ("a shared cell claims an expert receipt",
     lambda s, u, c: next(x for x in c if x["class"] == "shared").update(receipt="expert_wires+checkpoint")),
    ("an unoffered cell is offered", lambda s, u, c: next(
        x for x in c if x["offered_current"] == "0").update(offered_current="1")),
    ("a check failed", lambda s, u, c: s["checks"][0].update(ok=False)),
    ("a roster misses a unit", lambda s, u, c: s["rosters"][0].update(equals=False)),
    ("a receipt hash differs from its own bytes", lambda s, u, c: s["actions"][0]["receipt"].update(
        recomputed_sha256="0" * 64)),
    ("the CAS payload differs from the stdout", lambda s, u, c: s["actions"][0]["payload"].update(sha256="1" * 64)),
    ("a suspended gate has no resolution", lambda s, u, c: s["dev_ledger"][0].update(resolution=None)),
    ("a performance claim appears", lambda s, u, c: s.update(performance_claims=[{"speedup": 2.0}])),
    ("the run was a slice", lambda s, u, c: s.update(limit_units=6)),
    ("the table row count drifts", lambda s, u, c: s["tables"]["cells"].update(rows=3467)),
])
def test_the_validator_rejects_a_damaged_record(record, name, how):
    assert _damaged(record, how), name


# --------------------------------------------------------------------------- the verifier's helpers on synthetic trees
def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _pb_tree(tmp_path, *, payload=b"audit result\n", returncode=0):
    """A PrismaBuild queue and CAS holding one finished action, laid out as the fleet lays them out."""
    key = "ab" + "0" * 62
    queue, cas = tmp_path / "pb-queue", tmp_path / "cas"
    request = {"action_key": key, "task": {"argv": ["/bin/true"]},
               "params": {"checkout_snapshot": {"commit": "c" * 40, "parent": "d" * 40,
                                                "input": {"sha256": "e" * 64}}}}
    blob_sha = hashlib.sha256(payload).hexdigest()
    receipt = {"action_key": key, "action_manifest_sha256": hashlib.sha256(_canonical(request)).hexdigest(),
               "result": {"bytes": len(payload), "sha256": blob_sha},
               "schema": "prismaquant.prismabuild.cas_receipt.v3"}
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(
        {k: v for k, v in receipt.items() if k != "receipt_sha256"})).hexdigest()
    trailer = {"local_result_claim_sha256": "f" * 64, "payload_path": "x", "receipt": receipt, "status": "published"}
    stdout = payload + json.dumps(trailer).encode() + b"\n"
    stdout_sha = hashlib.sha256(stdout).hexdigest()
    attempt = queue / "attempts" / key / ("1" * 64)
    files = {
        queue / "done" / f"{key}.json": {"resources": {"cpu": 1}, "tags": []},
        attempt / "00000001.json": {"disposition": "done", "status": "executed", "claimed_host": "h",
                                    "detail": {"returncode": returncode, "elapsed_s": 1.0},
                                    "logs": {"stdout": {"path": f"attempts/{key}/{'1' * 64}/00000001.stdout.{stdout_sha}.log",
                                                        "sha256": stdout_sha, "bytes": len(stdout)}}},
        cas / "requests" / key[:2] / f"{key}.json": request,
        cas / "actions" / "v3" / key[:2] / f"{key}.json": receipt,
    }
    for path, value in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical(value))
    (attempt / f"00000001.stdout.{stdout_sha}.log").write_bytes(stdout)
    blob = cas / "blobs" / blob_sha[:2] / blob_sha
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(payload)
    return key, blob, attempt / f"00000001.stdout.{stdout_sha}.log"


def test_a_finished_action_verifies_through_its_whole_cas_chain(tmp_path):
    key, _blob, _log = _pb_tree(tmp_path)
    row = audit.verify_action(tmp_path, key, "fixture")
    assert row["ok"] and not row["problems"]
    assert row["payload"]["sha256"] == row["receipt"]["result_sha256"] == row["blob"]["sha256"]
    assert row["receipt"]["sha256"] == row["receipt"]["recomputed_sha256"] != row["stdout"]["sha256"]


def test_a_tampered_cas_blob_is_found(tmp_path):
    key, blob, _log = _pb_tree(tmp_path)
    blob.write_bytes(b"audit RESULT\n")
    assert any("CAS blob" in p for p in audit.verify_action(tmp_path, key, "fixture")["problems"])


def test_a_tampered_stdout_is_found(tmp_path):
    key, _blob, log = _pb_tree(tmp_path)
    data = bytearray(log.read_bytes())
    data[0] ^= 1
    log.write_bytes(bytes(data))
    problems = audit.verify_action(tmp_path, key, "fixture")["problems"]
    assert any("stdout log differs" in p for p in problems) and any("stdout payload differs" in p for p in problems)


def test_a_failed_action_is_not_qualifying(tmp_path):
    key, _blob, _log = _pb_tree(tmp_path, returncode=1)
    assert any("returncode 1" in p for p in audit.verify_action(tmp_path, key, "fixture")["problems"])


def test_a_receipt_that_does_not_hash_to_itself_is_found(tmp_path):
    key, _blob, _log = _pb_tree(tmp_path)
    path = tmp_path / "cas" / "actions" / "v3" / key[:2] / f"{key}.json"
    receipt = json.loads(path.read_bytes())
    receipt["producer"] = {"worker": "forged"}
    path.write_bytes(_canonical(receipt))
    assert any("own receipt_sha256" in p for p in audit.verify_action(tmp_path, key, "fixture")["problems"])


def test_a_key_prefix_names_one_request(tmp_path):
    key, _blob, _log = _pb_tree(tmp_path)
    assert audit.resolve_key(tmp_path, key[:12]) == key
    with pytest.raises(ValueError):
        audit.resolve_key(tmp_path, "ff" + key[2:12])


def test_the_package_profile_frames_files_exactly_as_the_producer_stamp_does(tmp_path):
    from prismaquant.production_weight_cache import _production_cache_source_profile

    root = tmp_path / "prismaquant"
    (root / "model_profiles" / "specs").mkdir(parents=True)
    (root / "__pycache__").mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "model_profiles" / "specs" / "m.json").write_text("{}\n")
    (root / "__pycache__" / "a.cpython-314.pyc").write_bytes(b"\0")
    assert audit.package_profile(root) == _production_cache_source_profile(root)


def test_flat_diff_names_the_leaves_two_plans_differ_at():
    plan = {"a": 1, "inputs": {"x": "p", "y": "q"}}
    other = {"a": 1, "inputs": {"x": "p", "y": "r"}, "source_identity_cache": {"path": "f", "sha256": "s"}}
    assert audit.flat_diff(plan, other) == {"/inputs/y": ["q", "r"],
                                            "/source_identity_cache": [None, {"path": "f", "sha256": "s"}]}
    assert audit.flat_diff(plan, plan) == {}


def test_a_roster_difference_is_named_and_fails(tmp_path):
    ledger = audit.Ledger()
    row = audit.roster_row(ledger, "fixture", {"a", "b", "z"}, {"a", "b", "c"})
    assert not row["equals"] and row["missing"] == ["c"] and row["extra"] == ["z"]
    assert not ledger.checks[0]["ok"]


def test_dev_mode_lines_are_classified_by_the_gate_they_suspend(tmp_path):
    key, _blob, log = _pb_tree(tmp_path, payload=b"[DEV-MODE] seal prepared plan_sha256 differs (x)\n"
                                                  b"[DEV-MODE] something nobody classified\nplain line\n")
    found = audit.dev_lines(tmp_path, key)
    assert [(f["line"], f["kind"]) for f in found] == [(1, "prepared_plan"), (2, "unclassified")]


# --------------------------------------------------------------------------- body-price admission
def test_body_cost_admission_refuses_an_mtp_payload():
    """The refusal the record quotes: an MTP payload offered as a body cost table (allocator ``--costs``)."""
    from prismaquant import cost_currency
    from test_glm_mtp_selection import _payload

    with pytest.raises(cost_currency.CostCurrencyError, match=re.escape(audit.MTP_BODY_REFUSAL)):
        cost_currency.require_run_currency(_payload())
    source = (Path(cost_currency.__file__)).read_text()
    assert audit.MTP_BODY_REFUSAL in source
