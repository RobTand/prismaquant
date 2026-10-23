"""Echo-logprob population checks at the real example HTTP boundary."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
from urllib.parse import urlsplit

import pytest


REFUSALS = (RuntimeError, ValueError, TypeError, KeyError, OverflowError)


@pytest.fixture
def example():
    path = Path(__file__).resolve().parents[1] / "examples/launchers/bench-quality.py"
    spec = importlib.util.spec_from_file_location("pq_bench_quality_population", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _http(monkeypatch, example, values):
    calls = []

    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    def get(url, *, timeout):
        assert url == f"{example.VLLM}/models" and timeout == 30
        return Response({"data": [{"id": example.MODEL}]})

    def post(url, *, json, timeout):
        assert url == f"{example.VLLM}/completions"
        assert json["model"] == example.MODEL
        calls.append(dict(json))
        if json["max_tokens"] == 0:
            assert json["echo"] is True and json["logprobs"] == 0
            assert timeout == 120
            return Response({"choices": [{"logprobs": {
                "token_logprobs": values,
            }}]})
        assert timeout == 300 and json["max_tokens"] == 256
        return Response({"choices": [{"text": "bounded HTTP fixture"}]})

    monkeypatch.setattr(example.requests, "get", get)
    monkeypatch.setattr(example.requests, "post", post)
    return calls


@pytest.mark.parametrize("values, expected_ppl, expected_tokens", [
    ([None, -math.log(2), -math.log(8), -math.log(4)], 4.0, 3),
    ([-math.log(2), -math.log(8)], 4.0, 2),
    ([None, 0.0, -math.log(4)], 2.0, 2),
])
def test_public_perplexity_uses_every_supplied_scored_token(
    example, monkeypatch, values, expected_ppl, expected_tokens,
):
    calls = _http(monkeypatch, example, values)
    ppl, count = example.perplexity_on_text("tiny prompt")
    assert count == expected_tokens
    assert ppl == pytest.approx(expected_ppl)
    assert len(calls) == 1


@pytest.mark.parametrize("values", [
    [None, -0.25, None, -0.75],
    [-0.25, None, -0.75],
    [None, -0.25, float("nan")],
    [None, -0.25, float("inf")],
    [None, -0.25, float("-inf")],
    [None, -0.25, True],
    [None, -0.25, 0.25],
    [None, -0.25, "missing"],
    [None, -0.25, {"logprob": -0.75}],
    [None, None],
    [None],
    [],
], ids=[
    "later-missing", "missing-with-finite-first", "nan", "positive-inf",
    "negative-inf", "boolean", "positive-logprob", "string", "mapping",
    "all-missing", "first-only", "empty",
])
def test_public_perplexity_refuses_incomplete_or_invalid_observations(
    example, monkeypatch, values,
):
    _http(monkeypatch, example, values)
    with pytest.raises(REFUSALS):
        example.perplexity_on_text("tiny prompt")


def test_real_main_prints_the_complete_population_and_runs_three_http_samples(
    example, monkeypatch, capsys,
):
    calls = _http(monkeypatch, example, [None, -math.log(2), -math.log(8), -math.log(4)])
    assert example.main() == 0
    output = capsys.readouterr().out
    assert "ppl=4.000 on 3 tokens" in output
    assert "[summary] ppl=4.000  generations=3 OK" in output
    assert len(calls) == 4


@pytest.mark.parametrize("values", [
    [None, -0.25, None, -0.75],
    [None, -0.25, float("nan")],
    [None, None],
], ids=["later-missing", "nan", "all-missing"])
def test_real_main_cannot_emit_a_metric_or_success_from_invalid_population(
    example, monkeypatch, capsys, values,
):
    calls = _http(monkeypatch, example, values)
    result = example.main()
    output = capsys.readouterr().out
    assert result == 1
    assert "FAILED" in output
    assert "perplexity response" in output
    assert "[summary]" not in output
    assert "  ppl=" not in output
    assert len(calls) == 1, "generation samples ran after invalid perplexity evidence"


def _validator_http(monkeypatch, populations, *, spec_on=False, usage=None):
    from prismaquant import validate_quantized_model as validator

    calls = []
    echoes = []

    class Response:
        status = 200

        def __init__(self, body):
            self.body = body if isinstance(body, bytes) else json.dumps(body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return self.body

    def urlopen(request, *, timeout):
        url = request if isinstance(request, str) else request.full_url
        path = urlsplit(url).path
        if path == "/health":
            return Response(b"ok")
        if path == "/metrics":
            return Response(b"vllm:spec_decode_num_drafts_total 0\n" if spec_on
                            else b"vllm:num_requests_running 0\n")
        assert request.data is not None
        body = json.loads(request.data)
        calls.append((path, body))
        if path == "/v1/chat/completions":
            return Response({"choices": [{"message": {
                "content": "<think>bounded fixture</think>42",
            }, "finish_reason": "stop"}]})
        assert path == "/v1/completions"
        if body.get("echo") is True:
            assert body["max_tokens"] == 1 and body["logprobs"] == 1
            index = len(echoes)
            values = populations[index % len(populations)]
            echoes.append(body)
            return Response({"choices": [{"logprobs": {"token_logprobs": values}}],
                             **({"usage": usage} if usage is not None else {})})
        assert body["max_tokens"] == 40
        return Response({"choices": [{"text": "Bounded generation fixture. " * 4}]})

    monkeypatch.setattr(validator.urllib.request, "urlopen", urlopen)
    return validator, calls, echoes


def _check(validator):
    return validator.check_perplexity(
        "http://fixture.invalid", "fixture-model",
        max_ppl=100.0, max_mean_nll=100.0, max_p99_nll=100.0,
    )


def test_real_ship_gate_preserves_complete_token_weighting(monkeypatch):
    validator, _calls, echoes = _validator_http(
        monkeypatch, [[None, -0.25, -0.25], [None, -0.75]],
    )
    result = _check(validator)
    assert result.passed, result.detail
    assert len(echoes) == len(validator.EVAL_PROMPTS) == 12
    assert result.metrics["n_tokens"] == 18
    assert result.metrics["mean_nll_per_tok"] == pytest.approx((0.5 + 0.75) / 3)
    assert result.metrics["perplexity"] == pytest.approx(math.exp((0.5 + 0.75) / 3))


@pytest.mark.parametrize("values", [
    [None, -0.25, None, -0.75],
    [None, -0.25, float("nan")],
    [None, -0.25, float("inf")],
    [None, -0.25, float("-inf")],
    [None, -0.25, True],
    [None, -0.25, 0.25],
    [None, -0.25, "missing"],
    [None, -0.25, {"logprob": -0.75}],
    [None, None],
], ids=["later-missing", "nan", "positive-inf", "negative-inf", "boolean",
        "positive-logprob", "string", "mapping", "all-missing"])
def test_real_ship_gate_refuses_invalid_echo_population(monkeypatch, values):
    validator, _calls, echoes = _validator_http(monkeypatch, [values])
    try:
        result = _check(validator)
    except REFUSALS:
        return
    assert echoes
    assert result.passed is False, "invalid observations produced a passing ship-gate check"


def test_real_ship_gate_spec_decode_refusal_precedes_echo_request(monkeypatch):
    validator, _calls, echoes = _validator_http(monkeypatch, [[None, -0.25]], spec_on=True)
    result = _check(validator)
    assert result.passed is False
    assert result.metrics["spec_decode_detected"] is True
    assert echoes == []


@pytest.mark.parametrize("values, expected_pass", [
    ([None, -0.25, -0.75], True),
    ([None, -0.25, None, -0.75], False),
    ([None, -0.25, float("nan")], False),
], ids=["complete", "later-missing", "nan"])
def test_real_validator_main_cannot_report_pass_from_invalid_population(
    monkeypatch, capsys, values, expected_pass,
):
    validator, _calls, echoes = _validator_http(monkeypatch, [values])
    monkeypatch.setattr(sys, "argv", [
        "validate_quantized_model", "--base-url", "http://fixture.invalid",
        "--model-name", "fixture-model", "--boundary-reps", "1",
    ])
    result = validator.main()
    output = capsys.readouterr().out
    if expected_pass:
        assert len(echoes) == 12
    else:
        assert 1 <= len(echoes) <= 12
    assert result == (0 if expected_pass else 1)
    assert ("✅ PASS" in output) is expected_pass
    if not expected_pass:
        assert "❌ FAIL" in output


def test_example_imports_from_checkout_without_pythonpath(tmp_path):
    import subprocess

    path = Path(__file__).resolve().parents[1] / "examples/launchers/bench-quality.py"
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         "import runpy, sys; runpy.run_path(sys.argv[1])", str(path)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("body", [None, [], {}, {"choices": []}, {"choices": [None]}])
def test_example_main_refuses_malformed_response_body(example, monkeypatch, capsys, body):
    _http(monkeypatch, example, [None, -0.25])

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return body

    monkeypatch.setattr(example.requests, "post", lambda *args, **kwargs: Response())
    assert example.main() == 1
    output = capsys.readouterr().out
    assert "FAILED" in output
    assert "[summary]" not in output


def test_example_main_refuses_invalid_generation_text(example, monkeypatch, capsys):
    _http(monkeypatch, example, [None, -0.25])
    real_post = example.requests.post

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"text": None}]}

    def post(url, *, json, timeout):
        if json["max_tokens"] == 0:
            return real_post(url, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(example.requests, "post", post)
    assert example.main() == 1
    output = capsys.readouterr().out
    assert "FAILED generation response" in output
    assert "[summary]" not in output


def test_validator_error_has_one_prompt_label(monkeypatch):
    validator, _calls, _echoes = _validator_http(
        monkeypatch, [[None, -0.25], [None, -0.25], [None, None]],
    )
    result = _check(validator)
    assert not result.passed
    assert result.detail.count("prompt 3") == 1
    assert "invalid token_logprobs at position 1" in result.detail
