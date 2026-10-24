"""Offline acceptance of the existing #2530 owner's original-source handoff.

This is NOT a capture, pricing, source-count implementation, certification or
serving test. Do not edit #2771's protocol to satisfy it. Reuse original evidence,
including the byte-identical retained #2530 tables and final-hidden manifest
pinned below; do not rerun that work. The implementation deliverable is
documentary, not a new API. The original final-hidden manifest is retained as
final-hidden-manifest.json to make per-sequence accounting independently
checkable without the fleet, tensors, or a historical Git checkout.

Local contract: docs/measurements/pq-2530-mtp-layer45/handoff.json contains:
* w1: {path: repository-relative retained owner artifact, sha256, owner};
* source_dispositions: one row per original unit, with unit, w1_selector,
  disposition, reason, capture_sha256, source_bf16_passthrough (bool), and
  observation: {state: observed, count, max_abs};
* cells: one row per (unit, rung) in the original cells.csv, retaining its price,
  wire and receipt fields (JSON numbers instead of CSV strings), plus
  evidence_kind: original, objective: mtp_head_self_kl, render_sha256,
  render_bytes, disposition, reason, dev_uncertified, release_qualified, and
  stage_a_ref/stage_b_ref/render_ref/producer_ref. Each ref has the original
  input's sha256 and a nonempty selector locating this unit/rung inside it;
* calibration: {state, artifact_sha256, token_bytes_sha256, sequences, seqlen,
  tokens}; final_hidden: {state, manifest_sha256, calibration_sha256,
  observations: [{sequence, sha256, shape, top1, nll}, ...]};
* dev_stamps: the four original Stage B stamps; later_reprobes: a separate list.

Selectors are JSON pointers or explicit original table/pickle keys, not new
opaque receipt IDs. An owner/reviewer must check their authenticity and actual
W1 membership against the retained originals. These CPU checks establish exact
coverage, historical value preservation and evidence linkage, NOT authenticated
model-byte reads, valid certificates, numerical quality or serving qualification.
No fleet paths are opened by these tests and no historical Git objects are used.
"""

import csv
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/measurements/pq-2530-mtp-layer45"
REPORT = ROOT / "docs/measurements/pq-2530-mtp-layer45-handoff.md"
# Retained record b10937aa316fcc2b20ae2d9f077d4cd80df77b3e; not a claim
# that that revision was merged or that its qualification conclusion is current.
RETAINED = {
    "audit.json": "95d0c34279ccbdd2c11be4373f13101f51632f7f5c256e113ad5bba478f7974a",
    "units.csv": "64ce7611b5d7fbba2c39bd8b68f2a285fd15d1255eab05fb655a34a621807416",
    "cells.csv": "c8d30e8f73b2788af8b5813d7f496c35c18e0e79d080720064bbe0dca3c9c6b7",
    # Original #2530 final-hidden input, independently read and hash-checked
    # during research; also anchored by audit.json's final-hidden input role.
    "final-hidden-manifest.json": "9819766aae872b78e4af54d9d93885e275789a73eda5315564a1f3c3540438d1",
}


def _handoff():
    path = DIRECTORY / "handoff.json"
    assert path.is_file(), (
        "#2832 lacks the existing #2530 owner's W1-bound original-source "
        "handoff; inventory/protocol prose alone does not supply these joins"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _original(name):
    path = DIRECTORY / name
    assert path.is_file(), f"Retain the original #2530 evidence table: {name}"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == RETAINED[name], (
        f"{name} must preserve the original record, not a later re-probe"
    )
    if name.endswith(".json"):
        return json.loads(data)
    return list(csv.DictReader(data.decode("utf-8").splitlines()))


def _sha(value):
    assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def _text(value):
    assert isinstance(value, str) and value.strip()


def _finite(value):
    assert type(value) in (int, float) and math.isfinite(value)


def _reference(ref, role, audit):
    expected = next(row for row in audit["inputs"] if row["role"] == role)
    assert ref["sha256"] == expected["sha256"], role
    _text(ref["selector"])


def test_owner_record_lands_implementation_documentation_and_retained_w1():
    handoff = _handoff()
    for name in RETAINED:
        _original(name)
    assert REPORT.is_file(), "Land #2530 implementation/handoff documentation"
    text = REPORT.read_text(encoding="utf-8").lower()
    for concept in (
        "glm_mtp_capture", "tessera_joint_aura", "enrich_mtp_cost_wires",
        "#2530", "#2832", "#2771", "dev_uncertified",
        "output_mse", "mtp_head_self_kl", "zero", "missing", "passthrough",
        "12", "shared", "re-probe", "no capture or pricing rerun",
        "source-count", "original-model", "quality", "serving", "byte",
    ):
        assert concept in text, f"Handoff documentation must explain {concept!r}"
    # Retain the real W1 artifact; only its owner can authorize historical reuse.
    w1 = handoff["w1"]
    relative = Path(w1["path"])
    assert not relative.is_absolute() and ".." not in relative.parts
    artifact = ROOT / relative
    assert artifact.is_file(), "W1 must be retained for offline review"
    _sha(w1["sha256"])
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == w1["sha256"]
    _text(w1["owner"])


def test_exactly_867_source_dispositions_join_original_observations():
    handoff = _handoff()
    original = {row["unit"]: row for row in _original("units.csv")}
    rows = handoff["source_dispositions"]
    assert len(rows) == 867 and len({row["unit"] for row in rows}) == 867
    assert {row["unit"] for row in rows} == set(original)
    assert sum(row["class"] == "routed" for row in original.values()) == 864
    assert sum(row["class"] == "shared" for row in original.values()) == 3
    for row in rows:
        expected = original[row["unit"]]
        _text(row["w1_selector"])
        _text(row["disposition"])
        _text(row["reason"])
        assert type(row["source_bf16_passthrough"]) is bool
        assert row["capture_sha256"] == expected["capture_sha256"]
        observed = row["observation"]
        assert observed["state"] == "observed", row["unit"]
        # No .get(..., 0), truthiness test or missing-to-zero conversion.
        assert type(observed["count"]) is int and observed["count"] >= 0
        assert observed["count"] == int(expected["count"])
        _finite(observed["max_abs"])
        assert observed["max_abs"] == float(expected["max_abs"])
    assert isinstance(handoff["later_reprobes"], list)


def test_all_original_cells_have_prices_bytes_render_and_producer_joins():
    handoff = _handoff()
    audit = _original("audit.json")
    original = {(row["unit"], row["rung"]): row for row in _original("cells.csv")}
    rows = handoff["cells"]
    assert len(rows) == 3468 and len({(r["unit"], r["rung"]) for r in rows}) == 3468
    assert {(r["unit"], r["rung"]) for r in rows} == set(original)
    shared = []
    for row in rows:
        expected = original[row["unit"], row["rung"]]
        assert row["evidence_kind"] == "original"
        assert row["objective"] == "mtp_head_self_kl"
        for field in ("stage_a_output_mse", "stage_b_predicted_dloss"):
            _finite(row[field])
            assert row[field] == float(expected[field])
        assert type(row["wire_bytes"]) is int
        assert row["wire_bytes"] == int(expected["wire_bytes"]) == int(expected["wire_file_bytes"])
        assert row["wire_sha256"] == expected["wire_sha256"]
        _sha(row["render_sha256"])
        assert type(row["render_bytes"]) is int and row["render_bytes"] > 0
        assert row["receipt"] == expected["receipt"]
        part = expected["stage_b_part"]
        _reference(row["stage_a_ref"], f"m3-{part}-cost", audit)
        _reference(row["stage_b_ref"], f"m4-{part}-price", audit)
        _reference(row["render_ref"], f"m4-{part}-production-cache", audit)
        producer_role = "checkpoint" if expected["class"] == "shared" else "receipts"
        _reference(row["producer_ref"], f"m3-{part}-{producer_role}", audit)
        _text(row["disposition"])
        _text(row["reason"])
        assert row["dev_uncertified"] is True
        assert row["release_qualified"] is False, "The handoff cannot certify historical dev runs"
        if expected["class"] == "shared":
            shared.append(row)
            assert "shared" in row["reason"].lower()
            assert "selection" in row["reason"].lower()
    assert len(shared) == 12
    assert all(row["rung"] != "BF16" for row in rows), (
        "TESSERA_BF16_* is quantized, not source BF16 passthrough"
    )
    assert handoff["dev_stamps"] == audit["dev_stamps"]


def test_complete_original_calibration_and_final_hidden_accounting():
    handoff = _handoff()
    audit = _original("audit.json")
    calibration = handoff["calibration"]
    assert calibration["state"] == "observed"
    assert calibration["artifact_sha256"] == audit["calibration"]["artifact_sha256"]
    assert calibration["token_bytes_sha256"] == audit["calibration"]["token_bytes_sha256"]
    assert (calibration["sequences"], calibration["seqlen"], calibration["tokens"]) == (512, 512, 262144)
    manifest = _original("final-hidden-manifest.json")
    expected_checks = {row["sequence"]: row for row in manifest["head_check"]}
    expected_records = {
        row["metadata"]["identity"]["coordinates"]["batch"]: row
        for row in manifest["records"]
    }
    assert set(expected_checks) == set(expected_records) == set(range(512))
    hidden = handoff["final_hidden"]
    assert hidden["state"] == "observed"
    assert hidden["manifest_sha256"] == next(
        row["sha256"] for row in audit["inputs"] if row["role"] == "final-hidden"
    )
    assert hidden["calibration_sha256"] == calibration["token_bytes_sha256"]
    rows = hidden["observations"]
    assert len(rows) == 512 and {row["sequence"] for row in rows} == set(range(512))
    for row in rows:
        sequence = row["sequence"]
        _sha(row["sha256"])
        assert row["sha256"] == expected_records[sequence]["sha256"]
        assert row["shape"] == expected_records[sequence]["shape"] == [1, 512, 4096]
        for field in ("top1", "nll"):
            _finite(row[field])
            assert row[field] == expected_checks[sequence][field]
        assert 0 <= row["top1"] <= 1 and row["nll"] >= 0
