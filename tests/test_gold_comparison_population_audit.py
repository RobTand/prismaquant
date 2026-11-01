"""Compare current writer-shaped synthetic records through real fingerprints."""
from __future__ import annotations

import copy
import json
import math

import pytest

from tests.test_kl_ab import BASE_MANIFEST, _result
from tools import kl_ab, measure_served_gold as gold
from tools.serve_fingerprint import fingerprint, performance_stack_fingerprint


def _contract(tmp_path, seqlen):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("one two three four five six seven eight")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir(exist_ok=True)
    # These synthetic bytes define the recorded contract, not a qualified
    # tokenizer/model/runtime. No tokenizer or network service is loaded.
    (tokenizer / "tokenizer.json").write_text('{"audit_fixture":true}')
    return gold._calibration_contract(
        corpus, tokenizer, list(range(1, 9)), seqlen=seqlen, n_tokens_requested=8)


def _record(tmp_path, *, tag, metric, seqlen):
    manifest = dict(copy.deepcopy(BASE_MANIFEST),
                    schema=gold.SERVE_MANIFEST_SCHEMA,
                    artifact_binding={"model_sha": tag * 64},
                    serve_session_id=tag * 64)
    value = (0.02 if tag == "a" else 0.01) if metric == "kl_confident_mean" else (10.0 if tag == "a" else 9.0)
    record = _result(value, manifest=manifest, metric=metric, model=f"/artifact-{tag}")
    contract = _contract(tmp_path, seqlen)
    windows = gold._windows(list(range(1, 9)), seqlen)
    scored = sum(len(window) - 1 for window in windows)
    record.update({
        "schema": gold.SERVED_KL_SCHEMA if metric == "kl_confident_mean" else gold.SERVED_PPL_SCHEMA,
        "quantization": "compressed-tensors",
        "spec_decode_detected": False,
        "n_samples": len(windows),
        "seqlen": seqlen,
        "git_commit": "e" * 40,
        "gold_producer_identity": {"measurement_tool": gold.MEASUREMENT_TOOL, "git_commit": "e" * 40},
        "calibration_contract": contract,
        "calibration_contract_sha256": gold._canonical_sha256(contract),
    })
    if metric == "kl_confident_mean":
        teacher = tmp_path / f"teacher-{seqlen}.json"
        teacher.write_text(json.dumps({"calibration_contract": contract, "n_positions": scored}))
        record.update({
            "mode": "student", "score_positions": "all", "prompt_top_k": 2,
            "vocab_size": 2, "n_positions": scored, "n_confident": scored,
            "kl_mean": value, "kl_max": value,
            "top1_agreement_all": 1.0, "top1_agreement_confident": 1.0,
            "topk_coverage_mean": 1.0, "topk_coverage_min": 1.0,
            "student_tail_model": "uniform_over_untabulated_vocabulary",
            "confident_top1_probability": gold.CONFIDENT_TOP1_PROBABILITY,
            "teacher_evidence": {"schema": gold.TEACHER_ARM_SCHEMA,
                "artifact_dir": "/teacher", "serve_fingerprint": "f" * 64,
                "spec_decode_detected": False, "git_commit": "e" * 40,
                "prompt_top_k": 2, "n_positions": scored,
                "dump_sha256": gold._sha256_file(teacher)},
        })
    else:
        record.update({"split": "wikitext-2-raw-v1/test", "n_tokens_requested": 8,
                       "n_tokens_scored": scored, "mean_nll": math.log(value),
                       "per_chunk_mean_nll": [math.log(value)] * len(windows),
                       "max_chunk_mean_nll": math.log(value)})
    assert fingerprint(record["serve_manifest"]) == record["serve_fingerprint"]
    return record


@pytest.mark.parametrize("metric", ["kl_confident_mean", "ppl"])
def test_current_records_with_different_window_contracts_refuse_a_delta(tmp_path, metric):
    a = _record(tmp_path, tag="a", metric=metric, seqlen=4)
    b = _record(tmp_path, tag="b", metric=metric, seqlen=6)
    assert a["n_samples"] == b["n_samples"] == 2
    assert a["calibration_contract_sha256"] != b["calibration_contract_sha256"]
    assert performance_stack_fingerprint(a["serve_manifest"]) == performance_stack_fingerprint(b["serve_manifest"])
    code, lines = kl_ab.compare(a, b, metric=metric)
    text = "\n".join(lines)
    assert "INVALID ATTESTATION" not in text, text
    assert code == 3, text
    assert "delta (" not in text


@pytest.mark.parametrize("metric", ["kl_confident_mean", "ppl"])
def test_same_contract_distinct_artifact_sessions_still_compare(tmp_path, metric):
    a = _record(tmp_path, tag="a", metric=metric, seqlen=4)
    b = _record(tmp_path, tag="b", metric=metric, seqlen=4)
    assert a["serve_fingerprint"] != b["serve_fingerprint"]
    assert a["calibration_contract_sha256"] == b["calibration_contract_sha256"]
    code, lines = kl_ab.compare(a, b, metric=metric)
    assert code == 0, "\n".join(lines)
    assert "delta (" in "\n".join(lines)


def test_bare_legacy_metric_pair_keeps_its_explicit_warning():
    code, lines = kl_ab.compare(_result(0.02), _result(0.01), metric="kl_confident_mean")
    assert code == 0
    assert "legacy JSONs" in "\n".join(lines)


def test_population_control_does_not_bypass_real_fingerprint_checks(tmp_path):
    a = _record(tmp_path, tag="a", metric="ppl", seqlen=4)
    b = _record(tmp_path, tag="b", metric="ppl", seqlen=4)
    b["serve_manifest"]["quantization"] = "tampered"
    code, lines = kl_ab.compare(a, b, metric="ppl")
    assert code == 3
    assert "INVALID ATTESTATION" in "\n".join(lines)


@pytest.mark.parametrize(("metric", "field", "value"), [
    ("ppl", "split", "another/test"),
    ("ppl", "n_tokens_scored", 5),
    ("kl_confident_mean", "student_tail_model", "another_tail"),
    ("kl_confident_mean", "confident_top1_probability", 0.9),
    ("kl_confident_mean", "n_confident", 5),
])
def test_known_population_and_metric_settings_cannot_silently_differ(
    tmp_path, metric, field, value,
):
    a = _record(tmp_path, tag="a", metric=metric, seqlen=4)
    b = _record(tmp_path, tag="b", metric=metric, seqlen=4)
    b[field] = value
    code, lines = kl_ab.compare(a, b, metric=metric)
    assert code == 3, "\n".join(lines)
    assert "delta (" not in "\n".join(lines)


@pytest.mark.parametrize("missing", [
    "calibration_contract", "calibration_contract_sha256", "n_tokens_scored",
])
def test_matching_incomplete_current_records_do_not_become_a_legacy_pair(tmp_path, missing):
    a = _record(tmp_path, tag="a", metric="ppl", seqlen=4)
    b = _record(tmp_path, tag="b", metric="ppl", seqlen=4)
    del a[missing]
    del b[missing]
    code, lines = kl_ab.compare(a, b, metric="ppl")
    assert code == 3, "\n".join(lines)
    assert "delta (" not in "\n".join(lines)


def test_matching_stale_or_partial_calibration_contracts_refuse(tmp_path):
    a = _record(tmp_path, tag="a", metric="ppl", seqlen=4)
    b = _record(tmp_path, tag="b", metric="ppl", seqlen=4)
    for record in (a, b):
        record["calibration_contract"]["calib_ids_sha256"] = "0" * 64
    assert kl_ab.compare(a, b, metric="ppl")[0] == 3
    for record in (a, b):
        del record["calibration_contract"]["tokenizer_files_sha256"]
        record["calibration_contract_sha256"] = gold._canonical_sha256(
            record["calibration_contract"])
    assert kl_ab.compare(a, b, metric="ppl")[0] == 3


def test_teacher_dump_identity_is_part_of_the_kl_population(tmp_path):
    a = _record(tmp_path, tag="a", metric="kl_confident_mean", seqlen=4)
    b = _record(tmp_path, tag="b", metric="kl_confident_mean", seqlen=4)
    b["teacher_evidence"]["dump_sha256"] = "0" * 64
    code, lines = kl_ab.compare(a, b, metric="kl_confident_mean")
    assert code == 3, "\n".join(lines)
    assert "delta (" not in "\n".join(lines)


def test_relocating_identical_corpus_bytes_does_not_change_the_population(tmp_path):
    a = _record(tmp_path, tag="a", metric="ppl", seqlen=4)
    b = _record(tmp_path, tag="b", metric="ppl", seqlen=4)
    b["calibration_contract"]["corpus_path"] = "/another/location/corpus.txt"
    b["calibration_contract_sha256"] = gold._canonical_sha256(b["calibration_contract"])
    code, lines = kl_ab.compare(a, b, metric="ppl")
    assert code == 0, "\n".join(lines)
    assert "comparison scope: recorded" in "\n".join(lines)


def test_cross_stack_override_cannot_waive_population_mismatch(tmp_path):
    a = _record(tmp_path, tag="a", metric="ppl", seqlen=4)
    b = _record(tmp_path, tag="b", metric="ppl", seqlen=6)
    manifest = b["serve_manifest"]
    manifest["image"] = "another-image"
    manifest["performance_stack_fingerprint"] = performance_stack_fingerprint(manifest)
    manifest["serve_fingerprint"] = fingerprint(manifest)
    b["serve_fingerprint"] = manifest["serve_fingerprint"]
    code, lines = kl_ab.compare(a, b, metric="ppl", allow_cross_fingerprint=True)
    assert code == 3, "\n".join(lines)
    assert "CROSS-STACK RANGE" not in "\n".join(lines)


def test_attested_legacy_population_warns_but_mixing_with_known_population_refuses(tmp_path):
    legacy = _result(10.0, manifest=BASE_MANIFEST, metric="ppl")
    code, lines = kl_ab.compare(legacy, legacy, metric="ppl")
    assert code == 0
    assert "population equivalence is not established" in "\n".join(lines)
    current = _record(tmp_path, tag="b", metric="ppl", seqlen=4)
    assert kl_ab.compare(legacy, current, metric="ppl")[0] == 3


@pytest.mark.parametrize(("field", "value"), [
    ("schema", "another.calibration/1"),
    ("seqlen", 0),
    ("seqlen", 1),
    ("n_windows", 0),
    ("n_windows", 3),
    ("n_tokens_selected", 1),
    ("n_tokens_requested", 1),
    ("window_overlap", 1),
])
def test_matching_invalid_current_calibration_is_not_a_comparable_population(
    tmp_path, field, value,
):
    a = _record(tmp_path, tag="a", metric="ppl", seqlen=4)
    b = _record(tmp_path, tag="b", metric="ppl", seqlen=4)
    for record in (a, b):
        record["calibration_contract"][field] = value
        record["calibration_contract_sha256"] = gold._canonical_sha256(
            record["calibration_contract"])
    code, lines = kl_ab.compare(a, b, metric="ppl")
    assert code == 3, "\n".join(lines)
    assert "delta (" not in "\n".join(lines)


@pytest.mark.parametrize(("metric", "field", "value"), [
    ("ppl", "seqlen", 6),
    ("ppl", "n_samples", 1),
    ("ppl", "n_tokens_requested", 7),
    ("ppl", "n_tokens_scored", 5),
    ("kl_confident_mean", "n_positions", 5),
])
def test_matching_payload_counts_must_agree_with_the_authenticated_contract(
    tmp_path, metric, field, value,
):
    a = _record(tmp_path, tag="a", metric=metric, seqlen=4)
    b = _record(tmp_path, tag="b", metric=metric, seqlen=4)
    for record in (a, b):
        record[field] = value
    code, lines = kl_ab.compare(a, b, metric=metric)
    assert code == 3, "\n".join(lines)
    assert "delta (" not in "\n".join(lines)
