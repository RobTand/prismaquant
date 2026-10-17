"""The served-lane gold tool: what it computes, and what it refuses.

Before this tool the only gold measurement tools were DSv4's, which construct
an in-process `LLM` under a contract pinning `tokenizer_mode="deepseek_v4"`.
Every other lane therefore measured with run-local scripts that emitted a bare
number and no identity -- no serve fingerprint, no producer commit, no
spec-decode state, no workload contract -- which is exactly the evidence a ship
record binds. These tests pin the arithmetic and, more importantly, the
refusals: a gold number that cannot say which stack produced it is the failure
mode the whole R15 fingerprint mechanism exists to prevent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools import measure_served_gold as gold


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------
def test_kl_is_zero_when_the_two_arms_agree():
    row = {"1": math.log(0.7), "2": math.log(0.2), "3": math.log(0.1)}
    assert gold._tail_logprob(row, vocab_size=3) == pytest.approx(
        min(row.values()))
    divergence = sum(
        math.exp(lp) * (lp - row[token]) for token, lp in row.items())
    assert divergence == pytest.approx(0.0, abs=1e-12)


def test_the_untabulated_tail_uses_a_clamped_max_entropy_estimate():
    """Spread residual mass uniformly, capped at the observed K-th value.

    This max-entropy estimate can be above or below an unknown token's true
    probability. It does not bound the true full-vocabulary KL.
    """
    row = {"1": math.log(0.5), "2": math.log(0.3)}      # 0.2 mass unaccounted
    tail = gold._tail_logprob(row, vocab_size=1002)
    assert tail == pytest.approx(math.log(0.2 / 1000))
    assert tail < min(row.values()), "the tail must sit below the K-th entry"

    # A row that already accounts for all its mass has no residual to spread.
    full = {"1": math.log(0.6), "2": math.log(0.4)}
    assert gold._tail_logprob(full, vocab_size=10) == pytest.approx(
        min(full.values()))


def test_coverage_is_recomputed_from_the_teacher_not_declared():
    """Top-K KL is only meaningful with its truncation reported alongside."""
    row = {"1": math.log(0.4), "2": math.log(0.3)}
    assert sum(math.exp(lp) for lp in row.values()) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def _dump(tmp_path, name, **overrides):
    payload = {
        "schema": gold.SERVED_DUMP_SCHEMA,
        "model": "qwen",
        "artifact_dir": str(tmp_path / "exported"),
        "prompt_top_k": 1024,
        "seqlen": 512,
        "n_samples": 2,
        "n_positions": 2,
        "vocab_size": 1000,
        "spec_decode_detected": False,
        "serve_fingerprint": "a" * 64,
        "performance_stack_fingerprint": "b" * 64,
        "serve_manifest": {"schema": gold.SERVE_MANIFEST_SCHEMA},
        "git_commit": "c" * 40,
        "gold_producer_identity": {"git_commit": "c" * 40},
        "calibration_contract": {"schema": gold.CALIBRATION_SCHEMA},
        "calibration_contract_sha256": "d" * 64,
        "positions": [
            {"1": math.log(0.9), "2": math.log(0.1)},
            {"1": math.log(0.6), "2": math.log(0.4)},
        ],
    }
    payload.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def _run_kl(tmp_path, teacher, student, monkeypatch):
    monkeypatch.setattr(gold, "_producer_identity",
                        lambda: {"git_commit": "e" * 40})
    out = tmp_path / "record.json"
    gold.main(["kl", "--teacher", str(teacher), "--student", str(student),
               "--out", str(out)])
    return json.loads(out.read_text())


def test_two_arms_dumped_at_different_top_k_are_refused(tmp_path, monkeypatch):
    """Their truncation floors differ, so the delta is not the quantization."""
    teacher = _dump(tmp_path, "t.json")
    student = _dump(tmp_path, "s.json", prompt_top_k=20)
    with pytest.raises(SystemExit, match="different top-K"):
        _run_kl(tmp_path, teacher, student, monkeypatch)


def test_two_arms_that_scored_different_text_are_refused(tmp_path, monkeypatch):
    """KL is only the quantization when both models saw identical inputs."""
    teacher = _dump(tmp_path, "t.json")
    student = _dump(tmp_path, "s.json", calibration_contract_sha256="f" * 64)
    with pytest.raises(SystemExit, match="different text/geometry"):
        _run_kl(tmp_path, teacher, student, monkeypatch)


def test_a_record_carries_every_field_the_ship_slot_binds(tmp_path, monkeypatch):
    """`shipcard_cli fill` lifts these by name; a missing one records
    `passed=false`, and `verify` refuses the card."""
    teacher = _dump(tmp_path, "t.json")
    student = _dump(tmp_path, "s.json")
    record = _run_kl(tmp_path, teacher, student, monkeypatch)

    assert record["score_positions"] == "all", (
        "'final' is the last-token hook screen: triage only, never promotion")
    assert record["spec_decode_detected"] is False
    assert len(record["serve_fingerprint"]) == 64
    assert len(record["git_commit"]) == 40
    assert record["n_positions"] > 0
    assert math.isfinite(record["kl_mean"])
    assert math.isfinite(record["kl_confident_mean"])
    # Identical arms: the divergence is zero and agreement is total.
    assert record["kl_mean"] == pytest.approx(0.0, abs=1e-12)
    assert record["top1_agreement_all"] == 1.0
    # The truncation is reported, never implied.
    assert record["prompt_top_k"] == 1024
    assert 0.0 < record["topk_coverage_min"] <= 1.0
    assert record["student_tail_model"] == "uniform_over_untabulated_vocabulary"
    # Only the position whose teacher top-1 exceeds 0.5 twice over is
    # "confident"; 0.6 clears it and 0.9 clears it, so both count here.
    assert record["n_confident"] == 2


def test_the_teacher_arm_is_bound_by_digest_not_by_assertion(
    tmp_path, monkeypatch,
):
    """A KL number names two serves. The second one has to be identified."""
    teacher = _dump(tmp_path, "t.json")
    student = _dump(tmp_path, "s.json", serve_fingerprint="9" * 64)
    record = _run_kl(tmp_path, teacher, student, monkeypatch)

    evidence = record["teacher_evidence"]
    assert evidence["schema"] == gold.TEACHER_ARM_SCHEMA
    assert evidence["serve_fingerprint"] == "a" * 64
    assert record["serve_fingerprint"] == "9" * 64, (
        "the record's own fingerprint is the STUDENT's serve")
    assert len(evidence["dump_sha256"]) == 64


def test_a_missing_serve_manifest_is_refused_not_warned(tmp_path):
    """A gold number with no fingerprint cannot be compared to anything."""
    artifact = tmp_path / "exported"
    artifact.mkdir()
    with pytest.raises(SystemExit, match="no .*serve_manifest.json"):
        gold._load_serve_manifest(artifact)


def test_a_manifest_without_a_fingerprint_is_refused(tmp_path):
    artifact = tmp_path / "exported"
    artifact.mkdir()
    (artifact / "serve_manifest.json").write_text(
        json.dumps({"schema": gold.SERVE_MANIFEST_SCHEMA}))
    with pytest.raises(SystemExit, match="carries no serve fingerprint"):
        gold._load_serve_manifest(artifact)


def test_a_stale_manifest_is_refused(tmp_path, monkeypatch):
    """A manifest describing a DIFFERENT serve is worse than none: it attests
    the wrong stack while looking like evidence."""
    manifest = {
        "schema": gold.SERVE_MANIFEST_SCHEMA,
        "serve_fingerprint": "a" * 64,
        "served_model_name": "some-other-model",
        "speculative_config": None,
    }
    with pytest.raises(SystemExit, match="the manifest is stale"):
        gold._check_manifest_describes_this_serve(
            manifest, base_url="http://localhost:8000",
            served_model_name="qwen")


def test_a_manifest_recording_speculative_decoding_is_refused(tmp_path):
    """With --speculative-config serving, /v1/completions logprobs are the
    DRAFT model's NLL (CLAUDE.md operational landmine)."""
    manifest = {
        "schema": gold.SERVE_MANIFEST_SCHEMA,
        "serve_fingerprint": "a" * 64,
        "served_model_name": "qwen",
        "speculative_config": {"model": "draft"},
    }
    with pytest.raises(SystemExit, match="speculative config"):
        gold._check_manifest_describes_this_serve(
            manifest, base_url="http://localhost:8000",
            served_model_name="qwen")


@pytest.mark.parametrize("observed", [None, True])
def test_spec_decode_that_is_not_a_clean_false_is_refused(monkeypatch, observed):
    """`None` means 'could not inspect', which is what the original trap
    looked like -- an unverified negative is not a negative."""
    monkeypatch.setattr(gold, "spec_decode_from_metrics", lambda url: observed)
    with pytest.raises(SystemExit, match="spec_decode_detected"):
        gold._refuse_spec_decode("http://localhost:8000")


def test_the_tool_is_a_registered_gold_producer():
    """`gold_producer_identity` refuses an unknown tool, and refuses a dirty
    tree -- which is what binds the number to reviewable source."""
    from tools.serve_fingerprint import _GOLD_PRODUCER_TOOL_FILES

    files = _GOLD_PRODUCER_TOOL_FILES[gold.MEASUREMENT_TOOL]
    assert "tools/measure_served_gold.py" in files
    root = Path(__file__).resolve().parents[1]
    for name in files:
        assert (root / name).is_file(), name
