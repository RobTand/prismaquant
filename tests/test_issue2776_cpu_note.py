"""CPU-only publication checks for the standalone #2776 replacement note.

Adopt the retained report at its existing path; do not revive #2533. Fixtures
are historical records, never instructions or runtime fleet dependencies.
See fixture-data/pq2281-confirming-arm/README.md for provenance and limits.
"""

import ast
import hashlib
import json
import math
import re
import statistics
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixture-data/pq2281-confirming-arm"
REPORT = ROOT / "docs/results/pq2281-confirming-arm.md"
ORIGINAL_REGRESSION = 0.001982807772259931
ARRAY_DIGESTS = {
    "candidate-teacher2.npy": "43c0dfa831c538d8efa574ab961fe34d9f60e677dbb3a912a7d98346b5b70f69",
    "candidate-teacher04.npy": "25083251ab141744dbee33961bd7a3b19da5c851a42cfebd69de11d643b9ea4e",
    "a8s-teacher2.npy": "5e13b59a9ce27df01357c137ace36c725dd1f72dccaab65ab89e3cae94507a62",
    "a8s-teacher04.npy": "5debf6cff9bf04f0bd2dc06f81c840b087eced3f82ab6f6ce3f4616937235973",
}


def _json(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _report():
    assert REPORT.is_file(), (
        "Missing standalone #2776 result note: adopt the retained report as "
        "docs/results/pq2281-confirming-arm.md, not another GPU run or #2533 revival"
    )
    return REPORT.read_text(encoding="utf-8")


def _summary(text):
    values = [json.loads(block) for block in re.findall(
        r"^```json\s*\n(.*?)^```\s*$", text, re.M | re.S
    )]
    summaries = [v for v in values if isinstance(v, dict) and
                 v.get("schema") == "pq2281.confirming_arm.summary.v1"]
    assert len(summaries) == 1, "Retain one machine-readable numerical summary"
    return summaries[0]


def _array_window_means(name):
    """Read the frozen NPY v1 float64 array with stdlib, on either CPU arm."""
    data = (FIXTURES / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == ARRAY_DIGESTS[name]
    assert data[:8] == b"\x93NUMPY\x01\x00"
    header_size = struct.unpack("<H", data[8:10])[0]
    header = ast.literal_eval(data[10:10 + header_size].decode("ascii"))
    assert header == {"descr": "<f8", "fortran_order": False, "shape": (25, 2047)}
    payload = data[10 + header_size:]
    assert len(payload) == 25 * 2047 * 8
    values = struct.unpack("<" + "d" * (25 * 2047), payload)
    return [statistics.fmean(values[i * 2047:(i + 1) * 2047]) for i in range(25)]


def test_signed_paired_result_recomputed_from_original_arrays():
    index = _json("evidence-index.json")
    for name, record in index["raw_files"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == record["sha256"]
    arm, a8s, paired = (_json(name) for name in (
        "candidate-result.json", "a8s-result.json", "paired-result.json"
    ))
    ids = [w["window_id"] for w in arm["windows"]]
    assert ids == [w["window_id"] for w in a8s["windows"]]
    assert ids == [w["window_id"] for w in paired["windows"]]
    assert len(ids) == len(set(ids)) == 25
    for key in ("source_initialization_contract_sha256", "source_model_identity_sha256", "producer_identity"):
        assert arm[key] == a8s[key]

    reconstructed = {}
    for teacher in ("teacher2", "teacher04"):
        av = _array_window_means(f"candidate-{teacher}.npy")
        bv = _array_window_means(f"a8s-{teacher}.npy")
        assert av == pytest.approx(arm[teacher]["per_window_mean"], rel=0, abs=1e-12)
        assert bv == pytest.approx(a8s[teacher]["per_window_mean"], rel=0, abs=1e-12)
        assert arm[teacher]["positions"] == a8s[teacher]["positions"] == 51175
        for aw, bw in zip(arm["windows"], a8s["windows"], strict=True):
            assert aw[teacher]["teacher_logits_data_sha256"] == bw[teacher]["teacher_logits_data_sha256"]
        differences = [a - b for a, b in zip(av, bv, strict=True)]
        reconstructed[teacher] = {
            "n_windows": 25,
            "candidate_mean": statistics.mean(av),
            "a8s_mean": statistics.mean(bv),
            "paired_delta_mean": statistics.mean(differences),
            "paired_delta_se": statistics.stdev(differences) / math.sqrt(25),
        }
    print("CPU reconstruction (retained arrays, not a new measurement): " + json.dumps(reconstructed, sort_keys=True))
    ratio = reconstructed["teacher2"]["paired_delta_mean"] / ORIGINAL_REGRESSION
    print(f"arm-minus-A8S / original regression = {ratio:.15f} ({100 * ratio:.9f}%)")

    summary = _summary(_report())
    assert summary["candidate_action"] == paired["candidate_action"]
    assert summary["primary_teacher"] == "teacher2"
    assert summary["primary_population"] == "all25"
    assert summary["delta_definition"] == "arm-minus-A8S"
    assert summary["pre_registered_delta_threshold"] == 0.0015
    assert summary["criterion_met"] is False
    for teacher, expected in reconstructed.items():
        actual = summary["results"][teacher]["all25"]
        for field, value in expected.items():
            assert actual[field] == pytest.approx(value, rel=0, abs=1e-12)
    assert reconstructed["teacher2"]["paired_delta_mean"] < 0.0015


def test_frozen_overlay_and_execution_citations_are_preserved():
    text = _report()
    index = _json("evidence-index.json")
    for records in (index["raw_files"], index["input_records"]):
        for label, record in records.items():
            assert Path(record["path"]).name in text, f"Missing record: {label}"
            assert record["sha256"] in text, f"Missing raw digest: {label}"
    for label, digest in index["identities"].items():
        assert digest in text, f"Missing frozen identity: {label}"
    for name, digest in ARRAY_DIGESTS.items():
        assert digest in text, f"Missing recovered per-position array identity: {name}"
    receipt = index["gpu_receipt"]
    for field in ("action_key", "action_manifest_sha256", "receipt_sha256"):
        assert receipt[field] in text, f"Missing historical execution binding: {field}"
    assert receipt["producer"]["attestation_sha256"] in text
    assert receipt["result"]["sha256"] in text
    for action in index["cpu_actions"].values():
        assert action["action_key"] in text
        assert action["receipt"]["receipt_sha256"] in text
    for failure in index["preserved_failures"]["CPU"]:
        assert failure["action_key"] in text, "Do not hide failed preparation"
    for attempt in index["gpu_outcome"]["attempt_history"]:
        assert attempt["outcome"] in text, "Cite each retained terminal attempt"


def test_correction_ratio_and_chronology_are_explicit():
    text = _report()
    for literal in ("dec-1009-165401-0740", "dec-1009-211419-a139", "21:16:20",
                    "0.00083", "0.00016", "0.0015", "0.001982807772259931"):
        assert literal in text, f"Missing binding correction / comparison: {literal}"
    assert re.search(r"\b42\s*%", text), "State the approximately 42% descriptive ratio"
    assert re.search(r"\bfailed\b", text, re.I)
    assert re.search(r"\bCEO\b", text), "Distinguish CEO-attributed evidence from reconstruction"
    # Accept either retained UNIX timestamps or their UTC presentation.
    for unix, utc in (("1791206282.3232286", "13:18:02.323229"),
                      ("1791206296.931573", "13:18:16.931573"),
                      ("1791206297.1067252", "13:18:17.106725"),
                      ("1791210521.5754535", "14:28:41.575454")):
        assert unix in text or utc in text, f"Missing actual historical timestamp: {unix}"
    for topic in (r"additiv", r"remaining.*layers|layers.*remaining", r"shipping.*prices|prices.*shipping",
                  r"Window4", r"republish", r"preflight", r"review", r"retrospect"):
        assert re.search(topic, text, re.I), f"Missing explicit limits / historical audit topic: {topic}"


def test_standalone_replacement_and_changelog_without_parent_closure():
    text = _report()
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "#2776" in text and "#2533" in text and "#2281" in text
    assert re.search(r"\bstopp(?:ed)?\b", text, re.I), "Identify #2533 as stopped, not this note's live owner"
    assert re.search(r"\badopt", text, re.I), "Record adoption of the existing retained report"
    assert "4accbe7cbde18c15e0d6b26c08e87e07c32d3f8a" in text, "Credit existing publication work; do not start a second copy"
    assert "#2776" in changelog and "pq2281-confirming-arm.md" in changelog
    assert re.search(r"#2776[^\n]*failed|failed[^\n]*#2776", changelog, re.I), "Changelog must label the failed result"
    closing = r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+(?:(?:RobTand/)?prismaquant)?#2281\b"
    assert not re.search(closing, text, re.I), "Do not accidentally close unresolved parent #2281"
    for line in changelog.splitlines():
        if "#2776" in line:
            assert not re.search(closing, line, re.I)
