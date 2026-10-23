"""Actual stored-dump KL intake with one post-parse path replacement."""
from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest

from tests.test_gold_comparison_population_audit import _record
from tools import measure_served_gold as gold


def _write_dump(tmp_path, tag):
    record = _record(tmp_path, tag=tag, metric="kl_confident_mean", seqlen=4)
    keys = ("model", "prompt_top_k", "seqlen", "n_samples", "n_positions",
            "vocab_size", "spec_decode_detected", "serve_fingerprint",
            "serve_manifest", "git_commit", "gold_producer_identity",
            "calibration_contract", "calibration_contract_sha256")
    payload = {key: record[key] for key in keys}
    payload.update(schema=gold.SERVED_DUMP_SCHEMA, artifact_dir=record["model"],
                   performance_stack_fingerprint=record["serve_manifest"]["performance_stack_fingerprint"],
                   positions=[{"0": math.log(0.75), "1": math.log(0.25)} for _ in range(6)])
    path = tmp_path / f"dump-{tag}.json"
    path.write_text(json.dumps(payload))
    return path, payload


def _kl(teacher, student, output):
    assert gold.main(["kl", "--teacher", str(teacher), "--student", str(student),
                      "--out", str(output)]) == 0
    return json.loads(output.read_bytes())


@pytest.mark.parametrize("replacement", ["distinct", "same_bytes", "unchanged"])
def test_teacher_receipt_names_the_bytes_actually_used_for_kl(tmp_path, monkeypatch, replacement):
    teacher, payload = _write_dump(tmp_path, "a")
    student, _ = _write_dump(tmp_path, "b")
    original = teacher.read_bytes()
    winner = copy.deepcopy(payload)
    winner["positions"] = [{"0": math.log(0.6), "1": math.log(0.4)} for _ in range(6)]
    replacement_bytes = json.dumps(winner).encode() if replacement == "distinct" else original
    real_loads = json.loads
    state = {"parsed": False}

    def loads(raw, *args, **kwargs):
        parsed = real_loads(raw, *args, **kwargs)
        if (raw == original or raw == original.decode()) and not state["parsed"]:
            state["parsed"] = True
            if replacement != "unchanged":
                staged = tmp_path / "replacement.json"
                staged.write_bytes(replacement_bytes)
                staged.replace(teacher)
        return parsed

    # Parse and hash remain real. This wrapper only schedules a real atomic
    # replacement after the actual parse. The source-identity recorder is the
    # declared external double, as in the existing stored-dump fixtures.
    monkeypatch.setattr(gold.json, "loads", loads)
    monkeypatch.setattr(gold, "_producer_identity", lambda: {
        "git_commit": "e" * 40, "measurement_tool": gold.MEASUREMENT_TOOL})
    measured = _kl(teacher, student, tmp_path / "measured.json")
    assert state["parsed"]
    assert measured["n_positions"] == 6
    assert measured["kl_mean"] == pytest.approx(0.0, abs=1e-12)
    if replacement == "distinct":
        assert teacher.read_bytes() != original
        reference = _kl(teacher, student, tmp_path / "winner-reference.json")
        assert reference["kl_mean"] > 0
    assert measured["teacher_evidence"]["dump_sha256"] == hashlib.sha256(original).hexdigest()
