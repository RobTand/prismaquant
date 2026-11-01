"""CPU-only acceptance for #2533's retained-measurement publication.

The report is prose plus one fenced JSON summary; see the fixture README for
its small data contract. No fleet access, model import, subprocess, or GPU is
needed. Fixture paths are historical citations, never paths to open at runtime.
"""

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixture-data" / "pq2281-confirming-arm"
REPORT = ROOT / "docs" / "results" / "pq2281-confirming-arm.md"
SUMMARY_SCHEMA = "pq2281.confirming_arm.summary.v1"


def _report():
    assert REPORT.is_file(), (
        "#2533 still needs docs/results/pq2281-confirming-arm.md: publish the "
        "retained failed confirmation; do not rerun the GPU arm"
    )
    return REPORT.read_text(encoding="utf-8")


def _fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _summary(text):
    summaries = []
    for block in re.findall(r"^```json\s*\n(.*?)^```\s*$", text, re.M | re.S):
        value = json.loads(block)
        if isinstance(value, dict) and value.get("schema") == SUMMARY_SCHEMA:
            summaries.append(value)
    assert len(summaries) == 1, (
        f"Include exactly one fenced JSON block with schema {SUMMARY_SCHEMA}; "
        "the fixture README documents the numerical summary fields"
    )
    return summaries[0]


def _assert_citations(text, records):
    for label, record in records.items():
        assert Path(record["path"]).name in text, f"Missing artifact citation: {label}"
        assert record["sha256"] in text, f"Missing artifact digest: {label}"


def test_signed_paired_statistics_and_failed_confirmation():
    text = _report()
    summary = _summary(text)
    index = _fixture("evidence-index.json")
    for name, record in index["raw_files"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == record["sha256"]
    arm = _fixture("candidate-result.json")
    a8s = _fixture("a8s-result.json")
    paired = _fixture("paired-result.json")
    assert summary["candidate_action"] == paired["candidate_action"]
    assert summary["primary_teacher"] == "teacher2"
    assert summary["primary_population"] == "all25"
    assert summary["delta_definition"] == "arm-minus-A8S"
    assert summary["pre_registered_delta_threshold"] == 0.0015
    assert summary["criterion_met"] is False

    # Bind the pairs before computing uncertainty. No independent/unpaired SE,
    # absolute-value delta, teacher substitution, or reselected primary panel.
    arm_ids = [w["window_id"] for w in arm["windows"]]
    assert arm_ids == [w["window_id"] for w in a8s["windows"]]
    assert arm_ids == [w["window_id"] for w in paired["windows"]]
    assert len(arm_ids) == len(set(arm_ids)) == 25
    for key in ("source_initialization_contract_sha256", "source_model_identity_sha256", "producer_identity"):
        assert arm[key] == a8s[key]
    clean_ids = {w["window_id"] for w in paired["windows"] if w["clean17"]}
    for teacher in ("teacher2", "teacher04"):
        assert arm[teacher]["positions"] == a8s[teacher]["positions"] == 51175
        for aw, bw in zip(arm["windows"], a8s["windows"], strict=True):
            assert aw[teacher]["teacher_logits_data_sha256"] == bw[teacher]["teacher_logits_data_sha256"]
        # all25 is mandatory. If clean17 is also summarized, it must retain the
        # frozen subset and cannot replace all25 as the confirmation population.
        populations = summary["results"][teacher]
        assert "all25" in populations
        assert set(populations) <= {"all25", "clean17"}
        for population, actual in populations.items():
            ids = set(arm_ids) if population == "all25" else clean_ids
            av = [v for wid, v in zip(arm_ids, arm[teacher]["per_window_mean"], strict=True) if wid in ids]
            bv = [v for wid, v in zip(arm_ids, a8s[teacher]["per_window_mean"], strict=True) if wid in ids]
            differences = [a - b for a, b in zip(av, bv, strict=True)]
            expected = {
                "candidate_mean": statistics.mean(av),
                "a8s_mean": statistics.mean(bv),
                "paired_delta_mean": statistics.mean(differences),
                "paired_delta_se": statistics.stdev(differences) / math.sqrt(len(differences)),
            }
            assert actual["n_windows"] == len(differences)
            for name, value in expected.items():
                assert actual[name] == pytest.approx(value, rel=0, abs=1e-12), (
                    f"{teacher}/{population}/{name}: recompute from matched retained windows"
                )
    delta = summary["results"]["teacher2"]["all25"]["paired_delta_mean"]
    assert summary["criterion_met"] == (delta >= summary["pre_registered_delta_threshold"])


def test_report_binds_the_frozen_inputs_and_complete_selection():
    text = _report()
    index = _fixture("evidence-index.json")
    _assert_citations(text, index["raw_files"])
    _assert_citations(text, index["input_records"])
    for label, digest in index["identities"].items():
        assert digest in text, f"Missing frozen input identity: {label}"


def test_report_retains_execution_attempt_and_observer_evidence():
    text = _report()
    index = _fixture("evidence-index.json")
    receipt = index["gpu_receipt"]
    for field in ("action_key", "action_manifest_sha256", "receipt_sha256"):
        assert receipt[field] in text, f"Missing PrismaBuild {field}"
    assert receipt["producer"]["attestation_sha256"] in text
    assert receipt["result"]["sha256"] in text
    for stage, action in index["cpu_actions"].items():
        assert action["action_key"] in text, f"Missing CPU stage: {stage}"
        assert action["receipt"]["receipt_sha256"] in text
    for failure in index["preserved_failures"]["CPU"]:
        assert failure["action_key"] in text, "Do not hide failed preparation attempts"
    for attempt in index["gpu_outcome"]["attempt_history"]:
        assert attempt["outcome"] in text, "Cite the retained per-attempt outcome, not just the action summary"
    _assert_citations(text, index["profiles_and_telemetry"])
