"""Real dump/KL CLI population checks with declared external-service doubles."""
from __future__ import annotations

import copy
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import pytest

from tests.test_kl_ab import BASE_MANIFEST, _result
from tools import measure_served_gold as gold


@pytest.fixture
def environment(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text('{"vocab_size":2}')
    (artifact / "tokenizer.json").write_text('{"audit_tokenizer_double":true}')
    manifest = dict(copy.deepcopy(BASE_MANIFEST), schema=gold.SERVE_MANIFEST_SCHEMA,
                    served_model_name="tiny", speculative_config=None)
    signed = _result(0.0, manifest=manifest)["serve_manifest"]
    (artifact / "serve_manifest.json").write_text(json.dumps(signed))
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("zero one zero one zero one zero one")
    env = SimpleNamespace(root=tmp_path, artifact=artifact, corpus=corpus,
                          missing_noninitial=False, response_defect=None,
                          token_ids=[0, 1] * 4, requests=[])

    def get(url, **kwargs):
        assert url == "http://audit.invalid/v1/models"
        return SimpleNamespace(json=lambda: {"data": [{"id": "tiny"}]})

    def urlopen(url, **kwargs):
        assert url == "http://audit.invalid/metrics"
        return io.BytesIO(b"# ordinary endpoint; no speculative decoding counters\n")

    def post(url, *, json, **kwargs):
        assert url == "http://audit.invalid/v1/completions"
        assert json["model"] == "tiny" and len(json["prompt"]) in (2, 4)
        env.requests.append(copy.deepcopy(json))
        row = {"0": {"logprob": math.log(0.75)},
               "1": {"logprob": math.log(0.25)}}
        positions = [None, *(copy.deepcopy(row) for _ in json["prompt"][1:])]
        if env.missing_noninitial:
            positions[2] = None
        elif env.response_defect == "short":
            positions.pop()
        elif env.response_defect == "extra":
            positions.append(copy.deepcopy(row))
        elif env.response_defect == "empty":
            positions[2] = {}
        elif env.response_defect == "invalid_logprob":
            positions[2] = {"0": {"logprob": float("nan")}}
        return SimpleNamespace(json=lambda: {"choices": [{"prompt_logprobs": positions}]})

    # External seams only: service responses, tokenization, and the source
    # identity recorder. Windowing, contracts, refusals and KL remain real.
    monkeypatch.setattr(gold.requests, "get", get)
    monkeypatch.setattr(gold.requests, "post", post)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(gold, "_tokenize", lambda corpus, tokenizer, maximum: env.token_ids[:maximum])
    monkeypatch.setattr(gold, "_producer_identity", lambda: {
        "git_commit": "e" * 40, "measurement_tool": gold.MEASUREMENT_TOOL})
    return env


def _dump(env, name):
    path = env.root / name
    argv = ["dump", "--base-url", "http://audit.invalid", "--model", "tiny",
            "--artifact-dir", str(env.artifact), "--corpus", str(env.corpus),
            "--seqlen", "4", "--max-tokens", "8", "--top-k", "2", "--out", str(path)]
    return argv, path


def _refuses_without_publication(argv, path):
    refusal = None
    try:
        gold.main(argv)
    except (SystemExit, ValueError, RuntimeError) as exc:
        refusal = str(exc)
    published = path.read_text() if path.exists() else None
    assert refusal is not None, f"accepted an incomplete population: {published}"
    assert any(word in refusal.lower() for word in (
        "position", "logprob", "window", "coverage", "count", "incomplete",
        "token", "population", "calibration", "length")), refusal
    assert not path.exists(), published


def test_dump_refuses_a_missing_noninitial_scorable_position(environment):
    env = environment
    env.missing_noninitial = True
    argv, path = _dump(env, "incomplete-dump.json")
    _refuses_without_publication(argv, path)
    assert env.requests


@pytest.mark.parametrize("truncate", ["student_only", "both"])
def test_kl_refuses_a_truncated_common_prefix_of_the_bound_population(environment, truncate):
    env = environment
    files = []
    for name in ("teacher.json", "student.json"):
        argv, path = _dump(env, name)
        assert gold.main(argv) == 0
        files.append(path)
    for path in (files[1:] if truncate == "student_only" else files):
        payload = json.loads(path.read_bytes())
        assert payload["n_positions"] == 6
        payload["positions"] = payload["positions"][:3]
        payload["n_positions"] = 3
        path.write_text(json.dumps(payload))
    before = {path: path.read_bytes() for path in files}
    output = env.root / "incomplete-kl.json"
    _refuses_without_publication([
        "kl", "--teacher", str(files[0]), "--student", str(files[1]), "--out", str(output)], output)
    assert {path: path.read_bytes() for path in files} == before


def test_dump_accepts_the_expected_first_null_and_keeps_every_other_position(environment):
    argv, path = _dump(environment, "full-dump.json")
    assert gold.main(argv) == 0
    payload = json.loads(path.read_bytes())
    assert len(environment.requests) == 2
    assert payload["n_samples"] == 2
    assert payload["n_positions"] == len(payload["positions"]) == 6
    assert payload["calibration_contract"]["n_windows"] == 2
    assert payload["calibration_contract_sha256"] == gold._canonical_sha256(payload["calibration_contract"])


def test_full_window_dumps_reach_actual_zero_kl_on_identical_distributions(environment):
    env = environment
    files = []
    for name in ("teacher-full.json", "student-full.json"):
        argv, path = _dump(env, name)
        assert gold.main(argv) == 0
        files.append(path)
    output = env.root / "full-kl.json"
    assert gold.main(["kl", "--teacher", str(files[0]), "--student", str(files[1]),
                      "--out", str(output)]) == 0
    record = json.loads(output.read_bytes())
    assert record["n_positions"] == record["n_confident"] == 6
    assert record["kl_mean"] == pytest.approx(0.0, abs=1e-12)
    assert record["kl_confident_mean"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("defect", ["short", "extra", "empty", "invalid_logprob"])
def test_dump_requires_the_exact_requested_response_position_roster(environment, defect):
    environment.response_defect = defect
    argv, path = _dump(environment, "bad-response.json")
    _refuses_without_publication(argv, path)
    assert environment.requests


@pytest.mark.parametrize("defect", ["empty_row", "actual_length", "stale_contract", "vocabulary"])
def test_kl_validates_each_stored_position_population_before_arithmetic(environment, defect):
    env = environment
    files = []
    for name in ("teacher-stored.json", "student-stored.json"):
        argv, path = _dump(env, name)
        assert gold.main(argv) == 0
        files.append(path)
    student = json.loads(files[1].read_bytes())
    if defect == "empty_row":
        student["positions"][1] = {}
    elif defect == "actual_length":
        student["positions"].pop()  # Keep the declared full count.
    elif defect == "stale_contract":
        student["calibration_contract"]["calib_ids_sha256"] = "0" * 64
    else:
        student["vocab_size"] = 3
    files[1].write_text(json.dumps(student))
    output = env.root / "bad-stored-kl.json"
    _refuses_without_publication([
        "kl", "--teacher", str(files[0]), "--student", str(files[1]), "--out", str(output)], output)


@pytest.mark.parametrize(("tokens", "windows", "positions"), [(2, 1, 1), (9, 2, 6), (10, 3, 7)])
def test_tail_windows_have_the_same_actual_and_declared_population(
    environment, tokens, windows, positions,
):
    env = environment
    env.token_ids = ([0, 1] * 5)[:tokens]
    files = []
    for name in ("teacher-tail.json", "student-tail.json"):
        argv, path = _dump(env, name)
        argv[argv.index("--max-tokens") + 1] = str(tokens)
        assert gold.main(argv) == 0
        dumped = json.loads(path.read_bytes())
        assert dumped["calibration_contract"]["n_tokens_selected"] == tokens
        assert dumped["calibration_contract"]["n_windows"] == dumped["n_samples"] == windows
        assert dumped["n_positions"] == len(dumped["positions"]) == positions
        files.append(path)
    assert len(env.requests) == 2 * windows
    output = env.root / "tail-kl.json"
    assert gold.main(["kl", "--teacher", str(files[0]), "--student", str(files[1]),
                      "--out", str(output)]) == 0
    assert json.loads(output.read_bytes())["kl_mean"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(("flag", "value"), [
    ("--seqlen", "0"), ("--seqlen", "1"),
    ("--max-tokens", "0"), ("--max-tokens", "-1"),
    ("--top-k", "0"), ("--top-k", "-1"),
])
def test_dump_refuses_unscorable_request_domains_without_publication(environment, flag, value):
    argv, path = _dump(environment, "unscorable-request.json")
    argv[argv.index(flag) + 1] = value
    _refuses_without_publication(argv, path)
    assert not environment.requests
