"""Metric comparisons preserve and require their recorded evaluation settings."""
from __future__ import annotations

import json

import pytest

from prismaquant.artifact_registry import ArtifactRecord, ArtifactRegistry
from prismaquant import validation_harness


def _settings():
    return {
        "model_sha": "a" * 64,
        "model_identity_scope": "local_content",
        "eval_split": "test",
        "metric_era": "2026-06-12-test-split",
        "wikitext_split": "test",
        "n_wikitext_tokens": 32,
        "n_mmlu_questions": 1,
        "calib_seqlen": 8,
        "calib_n_samples": 2,
        "calib_split": "test",
        "calib_seed": 42,
        "wikitext_max_window_tokens": 2048,
        "device": "cpu",
        "dtype": "fp32",
    }


def _record(record_id, settings, *, improved=False):
    return ArtifactRecord(
        record_id=record_id, model_path="/source", artifact_path=None,
        layer_config_sha=("b" if improved else "c") * 64,
        layer_config_path=None, target_bpp=8.0, achieved_bpp=8.0,
        format_histogram={"FP8_E4M3": 1},
        ppl_wikitext=8.0, ppl_mmlu_acc=0.5,
        end_kl=0.9 if improved else 1.0, eval_meta=settings,
        created_at="2026-10-17T00:00:00+00:00",
    )


def _registry(tmp_path, candidate_settings, baseline_settings):
    registry = ArtifactRegistry(tmp_path / "registry.json")
    registry.add(_record("baseline", baseline_settings))
    registry.add(_record("candidate", candidate_settings, improved=True))
    return registry


def _assert_scope(result):
    assert result["comparison_scope"] == "recorded_evaluation_settings"
    assert result["content_equivalence"] == "not_established"


@pytest.mark.parametrize("field,value", [
    ("model_sha", "d" * 64),
    ("model_identity_scope", "remote_identifier"),
    ("eval_split", "train"),
    ("metric_era", "different-era"),
    ("wikitext_split", "train"),
    ("n_wikitext_tokens", 64),
    ("n_mmlu_questions", 2),
    ("calib_seqlen", 16),
    ("calib_n_samples", 3),
    ("calib_split", "validation"),
    ("calib_seed", 7),
    ("wikitext_max_window_tokens", 4096),
    ("device", "cuda"),
    ("dtype", "bf16"),
    ("n_mmlu_questions", True),
    ("calib_n_samples", 2.0),
])
def test_better_numbers_cannot_pass_incompatible_recorded_settings(tmp_path, field, value):
    candidate = {**_settings(), field: value}
    registry = _registry(tmp_path, candidate, _settings())
    result = registry.compare("candidate", "baseline")
    assert result["pass"] is False
    assert field in " ".join(result["reasons"])
    _assert_scope(result)


@pytest.mark.parametrize("missing", ["legacy", "metric_era", "wikitext_max_window_tokens"])
def test_historical_settings_are_readable_but_not_backfilled_for_comparison(tmp_path, missing):
    baseline = {} if missing == "legacy" else _settings()
    if missing != "legacy":
        baseline.pop(missing)
    registry = _registry(tmp_path, _settings(), baseline)
    before = registry.path.read_bytes()
    loaded = ArtifactRegistry(registry.path)
    assert loaded.all()[0].eval_meta == baseline
    result = loaded.compare("candidate", "baseline")
    assert result["pass"] is False
    assert result["reasons"]
    _assert_scope(result)
    assert registry.path.read_bytes() == before
    assert loaded.all()[0].eval_meta == baseline


def test_matching_explicit_settings_preserve_existing_numeric_decisions(tmp_path):
    registry = _registry(tmp_path, _settings(), _settings())
    result = registry.compare("candidate", "baseline")
    assert result["pass"] is True
    assert all(metric["passed"] for metric in result["metrics"].values())


def test_success_reports_settings_scope_without_claiming_content_equivalence(tmp_path):
    registry = _registry(tmp_path, _settings(), _settings())
    result = registry.compare("candidate", "baseline")
    assert result["pass"] is True
    _assert_scope(result)


def test_compare_cli_refuses_legacy_population_before_overall_success(tmp_path, capsys):
    registry = _registry(tmp_path, _settings(), {})
    status = validation_harness.main([
        "compare", "--registry", str(registry.path),
        "--candidate-id", "candidate", "--baseline-id", "baseline",
    ])
    result = json.loads(capsys.readouterr().out)
    assert status == 1
    assert result["pass"] is False
    _assert_scope(result)


@pytest.mark.parametrize("local_model", [True, False])
def test_cli_registration_persists_authoritative_settings_and_truthful_model_scope(
        tmp_path, monkeypatch, capsys, local_model):
    model = str(tmp_path / "model.bin") if local_model else "unresolved-org/model"
    if local_model:
        (tmp_path / "model.bin").write_bytes(b"known local model bytes")
    config = tmp_path / "layer_config.json"
    config.write_text(json.dumps({"proj": "BF16"}))
    registry_path = tmp_path / "registry.json"
    monkeypatch.setenv("PRISMAQUANT_VALIDATION_FAKE_METRICS", json.dumps({
        "ppl_wikitext": 8.0, "ppl_mmlu_acc": 0.5, "end_kl": 0.9,
    }))
    monkeypatch.setenv("PRISMAQUANT_VALIDATION_WIKITEXT_STRIDE", "37")
    status = validation_harness.main([
        "--model", model, "--layer-config", str(config),
        "--cache-dir", str(tmp_path / "cache"),
        "--device", "cpu", "--dtype", "bf16",
        "--n-wikitext-tokens", "32", "--n-mmlu-questions", "1",
        "--calib-seqlen", "8", "--calib-n-samples", "2",
        "--calib-split", "validation", "--calib-seed", "7",
        "--register", "--registry", str(registry_path), "--no-progress",
    ])
    output = json.loads(capsys.readouterr().out)
    assert status == 0
    saved = ArtifactRegistry(registry_path).all()[0].eval_meta
    assert saved["eval_split"] == output["eval_split"] == "test"
    assert saved["metric_era"] == output["metric_era"]
    assert saved["wikitext_split"] == "test"
    assert saved["calib_split"] == "validation"
    assert saved["wikitext_max_window_tokens"] == 37
    assert saved["model_identity_scope"] == (
        "local_content" if local_model else "remote_identifier")
    assert saved == output["eval_meta"]
