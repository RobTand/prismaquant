"""Actual decide CLI domain controls; all inputs are explicitly assumed.

No rank, socket, external bandwidth binary or network measurement may run.
Zero link rate already raises in the historical arithmetic; it is a refusal
control, not an accepted-bad-number reproduction or a clean-diagnostic claim.
"""
from __future__ import annotations

import json
import re

import pytest

from tools import tp_decode_feasibility as tool


@pytest.fixture(autouse=True)
def no_network_work(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("decide-only metadata control reached a network workload")
    monkeypatch.setattr(tool, "_init_pg", unexpected)
    monkeypatch.setattr(tool, "_spawn_loopback", unexpected)
    monkeypatch.setattr(tool, "_iperf3_leg", unexpected)


def _entry(capsys, *options):
    try:
        code = tool.main(["--mode", "decide", *options])
    except (SystemExit, ValueError, ZeroDivisionError) as error:
        output = capsys.readouterr()
        return True, type(error).__name__, output.out, output.err
    assert type(code) is int
    output = capsys.readouterr()
    return code != 0, f"exit={code}", output.out, output.err


def _assert_refused_before_verdict(observed):
    refused, reason, stdout, stderr = observed
    assert refused, f"invalid physical input was accepted ({reason}):\n{stdout}\n{stderr}"
    assert not any(line.startswith("Qwen3.8-27B-CB") for line in stdout.splitlines()), stdout


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf"])
def test_nonphysical_assumed_latency_is_not_a_decision_number(capsys, value):
    _assert_refused_before_verdict(_entry(capsys, f"--latency-us={value}"))


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), float("-inf")],
                         ids=["negative", "nan", "inf", "negative-inf"])
def test_invalid_selected_json_latency_refuses_without_rewriting_input(tmp_path, capsys, value):
    path = tmp_path / "assumed-invalid.json"
    # Uniform coordinates isolate numeric admission from percentile selection.
    payload = {"schema": tool.SCHEMA,
               "label": "ASSUMED / artificial domain test, not measured",
               "results": [{"bytes": 10240, "p50_us": value,
                            "p90_us": value, "p99_us": value}]}
    original = (json.dumps(payload) + "\n").encode()
    path.write_bytes(original)
    _assert_refused_before_verdict(_entry(
        capsys, "--latency-json", str(path), "--latency-stat", "p90_us"))
    assert path.read_bytes() == original


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf"])
def test_accepted_nonphysical_link_bandwidth_must_refuse(capsys, value):
    _assert_refused_before_verdict(_entry(capsys, "--latency-us=100", f"--link-gbps={value}"))


def test_zero_bandwidth_remains_a_refusal_not_an_accepted_verdict(capsys):
    observed = _entry(capsys, "--latency-us=100", "--link-gbps=0")
    _assert_refused_before_verdict(observed)
    # Historical ZeroDivisionError is a real refusal, although not graceful.
    assert observed[1] in {"ZeroDivisionError", "ValueError", "SystemExit", "exit=2"}


@pytest.mark.parametrize("latency,expected", [(0, "WIN"), (100, "WIN"), (400, "LOSE")])
def test_finite_nonnegative_flat_latency_and_positive_link_keep_decisions(capsys, latency, expected):
    refused, reason, stdout, stderr = _entry(
        capsys, f"--latency-us={latency}", "--link-gbps=10")
    assert not refused, (reason, stderr)
    line = next(line for line in stdout.splitlines() if line.startswith("Qwen3.8-27B-CB"))
    assert re.search(rf"\b{expected}\b", line), stdout
    assert "latency source: ASSUMED" in stdout
    assert "UNKNOWN-NEEDS-TPOT (3)" in stdout


@pytest.mark.parametrize("latency,expected", [(0, "WIN"), (400, "LOSE")])
def test_finite_selected_json_latency_keeps_assumed_and_unknown_controls(tmp_path, capsys, latency, expected):
    path = tmp_path / "assumed-valid.json"
    payload = {"schema": tool.SCHEMA, "label": "ASSUMED / artificial positive control",
               "results": [{"bytes": 10240, "p50_us": latency,
                            "p90_us": latency, "p99_us": latency}]}
    original = json.dumps(payload).encode()
    path.write_bytes(original)
    refused, reason, stdout, stderr = _entry(
        capsys, "--latency-json", str(path), "--latency-stat", "p90_us", "--link-gbps=10")
    assert not refused, (reason, stderr)
    line = next(line for line in stdout.splitlines() if line.startswith("Qwen3.8-27B-CB"))
    assert re.search(rf"\b{expected}\b", line), stdout
    assert "latency source: ASSUMED" in stdout and "UNKNOWN-NEEDS-TPOT (3)" in stdout
    assert path.read_bytes() == original


@pytest.mark.parametrize("field", ["p50_us", "p90_us", "p99_us"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), float("-inf")],
                         ids=["negative", "nan", "inf", "negative-inf"])
def test_each_displayed_or_selected_json_coordinate_refuses_before_decision(
        tmp_path, capsys, field, value):
    path = tmp_path / "one-invalid-coordinate.json"
    result = {"bytes": 10240, "p50_us": 100.0, "p90_us": 200.0, "p99_us": 400.0}
    result[field] = value
    original = json.dumps({"label": "ASSUMED / artificial domain test",
                           "results": [result]}).encode()
    path.write_bytes(original)
    with pytest.raises(SystemExit, match=rf"latency JSON {field} .*finite and nonnegative"):
        tool.main(["--mode", "decide", "--latency-json", str(path),
                   "--latency-stat", "p90_us"])
    assert "DECIDE:" not in capsys.readouterr().out
    assert path.read_bytes() == original


@pytest.mark.parametrize("value", ["-1", "0", "nan", "inf", "-inf"])
def test_invalid_link_refuses_before_latency_artifact_resolution(monkeypatch, capsys, value):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid link input reached latency artifact resolution")
    monkeypatch.setattr(tool, "_resolve_decide_latency", unexpected)
    with pytest.raises(SystemExit, match="link-gbps must be finite and positive"):
        tool.main(["--mode", "decide", "--latency-json", "/missing.json",
                   f"--link-gbps={value}"])
    assert "DECIDE:" not in capsys.readouterr().out


def test_zero_bandwidth_has_a_named_refusal(capsys):
    with pytest.raises(SystemExit, match="link-gbps must be finite and positive"):
        tool.main(["--mode", "decide", "--latency-us=100", "--link-gbps=0"])
    assert "DECIDE:" not in capsys.readouterr().out
