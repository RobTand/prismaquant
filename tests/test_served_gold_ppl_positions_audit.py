"""Actual PPL CLI population intake with declared external-service doubles."""
from __future__ import annotations

import copy
import io
import json
import math
from types import SimpleNamespace
import urllib.request

import pytest

from tests.test_kl_ab import BASE_MANIFEST, _result
from tools import measure_served_gold as gold


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text('{"vocab_size":2}')
    (artifact / "tokenizer.json").write_text('{"audit_tokenizer_double":true}')
    manifest = dict(copy.deepcopy(BASE_MANIFEST), schema=gold.SERVE_MANIFEST_SCHEMA,
                    served_model_name="tiny", speculative_config=None)
    (artifact / "serve_manifest.json").write_text(json.dumps(
        _result(0.0, manifest=manifest)["serve_manifest"]))
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("zero one zero one zero one zero one")
    state = SimpleNamespace(defect=None, requests=[], output=tmp_path / "ppl.json")

    def get(url, **kwargs):
        assert url == "http://audit.invalid/v1/models"
        return SimpleNamespace(json=lambda: {"data": [{"id": "tiny"}]})

    def urlopen(url, **kwargs):
        assert url == "http://audit.invalid/metrics"
        return io.BytesIO(b"# ordinary endpoint; no speculative decoding counters\n")

    def post(url, *, json, **kwargs):
        assert url == "http://audit.invalid/v1/completions"
        assert json["model"] == "tiny" and json["prompt_logprobs"] == 0
        assert json["prompt"] == [0, 1, 0, 1]
        state.requests.append(copy.deepcopy(json))
        rows = [None, *({str(token): {"logprob": math.log(0.75)}}
                        for token in json["prompt"][1:])]
        if state.defect == "empty_row":
            rows[2] = {}
        elif state.defect == "actual_token_missing":
            rows[2] = {"1": {"logprob": math.log(0.25)}}  # actual token is 0
        elif state.defect == "short_response":
            rows.pop()
        return SimpleNamespace(json=lambda: {"choices": [{"prompt_logprobs": rows}]})

    # Only external service, tokenizer and source-identity recorder are doubles.
    monkeypatch.setattr(gold.requests, "get", get)
    monkeypatch.setattr(gold.requests, "post", post)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(gold, "_tokenize", lambda *args: [0, 1] * 4)
    monkeypatch.setattr(gold, "_producer_identity", lambda: {
        "git_commit": "e" * 40, "measurement_tool": gold.MEASUREMENT_TOOL})
    state.argv = ["ppl", "--base-url", "http://audit.invalid", "--model", "tiny",
                  "--artifact-dir", str(artifact), "--corpus", str(corpus),
                  "--seqlen", "4", "--max-tokens", "8", "--out", str(state.output)]
    return state


@pytest.mark.parametrize("defect", ["empty_row", "actual_token_missing", "short_response"])
def test_ppl_refuses_incomplete_noninitial_position_evidence(endpoint, defect):
    endpoint.defect = defect
    refusal = None
    try:
        gold.main(endpoint.argv)
    except (SystemExit, ValueError, RuntimeError) as exc:
        refusal = str(exc)
    assert endpoint.requests, "fixture never reached actual completion intake"
    published = endpoint.output.read_text() if endpoint.output.exists() else None
    assert refusal is not None, f"accepted {defect} as full-window PPL: {published}"
    assert any(word in refusal.lower() for word in (
        "position", "logprob", "window", "coverage", "count", "incomplete",
        "token", "population", "length")), refusal
    assert not endpoint.output.exists(), published


def test_ppl_accepts_first_null_and_scores_every_other_actual_token(endpoint):
    assert gold.main(endpoint.argv) == 0
    record = json.loads(endpoint.output.read_bytes())
    assert len(endpoint.requests) == record["n_samples"] == 2
    assert record["n_tokens_scored"] == 6
    assert record["mean_nll"] == pytest.approx(-math.log(0.75))
    assert record["ppl"] == pytest.approx(1 / 0.75)
    assert record["calibration_contract"]["n_windows"] == 2
    assert record["calibration_contract_sha256"] == gold._canonical_sha256(
        record["calibration_contract"])


def test_ppl_excludes_a_trailing_singleton_from_both_window_counts(endpoint, monkeypatch):
    monkeypatch.setattr(gold, "_tokenize", lambda *args: [0, 1] * 4 + [0])
    endpoint.argv[endpoint.argv.index("--max-tokens") + 1] = "9"
    assert gold.main(endpoint.argv) == 0
    record = json.loads(endpoint.output.read_bytes())
    assert len(endpoint.requests) == record["n_samples"] == 2
    assert record["calibration_contract"]["n_tokens_selected"] == 9
    assert record["calibration_contract"]["n_windows"] == 2
    assert record["n_tokens_scored"] == 6
    assert record["ppl"] == pytest.approx(1 / 0.75)
    gold.recorded_calibration_contract(record)
