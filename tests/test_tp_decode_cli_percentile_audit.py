"""Conditional CLI decisions from explicitly artificial assumed latencies."""
from __future__ import annotations

import json
import re

import pytest

from tools import tp_decode_feasibility as tool


def _run(tmp_path, capsys, *, stat, p50=100.0, p90=400.0, p99=400.0):
    path = tmp_path / "assumed-latency.json"
    payload = {
        "schema": tool.SCHEMA,
        "mode": "audit-assumption",
        "label": "ASSUMED / artificial test input, not a network measurement",
        "results": [{"bytes": 10240, "p50_us": p50, "p90_us": p90, "p99_us": p99}],
    }
    original = (json.dumps(payload, indent=2) + "\n").encode()
    path.write_bytes(original)
    argv = ["--mode", "decide", "--latency-json", str(path)]
    if stat is not None:
        argv += ["--latency-stat", stat]
    assert tool.main(argv) == 0
    output = capsys.readouterr().out
    assert "latency source: ASSUMED" in output
    assert path.read_bytes() == original
    row = next(line for line in output.splitlines() if line.startswith("Qwen3.8-27B-CB"))
    verdict = re.search(r"\b(WIN|LOSE|UNKNOWN-NEEDS-TPOT)\b", row)
    assert verdict is not None, output
    return verdict.group(1), output


@pytest.mark.parametrize("stat,expected", [
    (None, "WIN"),
    ("p50_us", "WIN"),
    ("p90_us", "LOSE"),
    ("p99_us", "LOSE"),
])
def test_requested_percentile_controls_the_conditional_cli_verdict(tmp_path, capsys, stat, expected):
    actual, output = _run(tmp_path, capsys, stat=stat)
    assert actual == expected, output


@pytest.mark.parametrize("stat", ["p50_us", "p99_us"])
def test_equal_assumed_percentiles_keep_the_same_conditional_decision(tmp_path, capsys, stat):
    actual, output = _run(tmp_path, capsys, stat=stat, p50=400.0)
    assert actual == "LOSE", output
    assert "UNKNOWN-NEEDS-TPOT (3)" in output


def test_selected_percentile_and_its_actual_value_are_stated(tmp_path, capsys):
    actual, output = _run(tmp_path, capsys, stat="p90_us", p90=300.0, p99=900.0)
    assert actual == "LOSE"
    assert "verdict percentile: p90_us" in output
    assert "verdict p90_us=300.0 us" in output
    assert "UNKNOWN-NEEDS-TPOT (3)" in output


def test_missing_selected_percentile_refuses_without_substitution(tmp_path, capsys):
    path = tmp_path / "old-p50-p99.json"
    original = json.dumps({"label": "ASSUMED / synthetic old input", "results": [
        {"bytes": 10240, "p50_us": 100.0, "p99_us": 400.0}]}).encode()
    path.write_bytes(original)
    with pytest.raises(SystemExit, match="lacks percentile.*p90_us"):
        tool.main(["--mode", "decide", "--latency-json", str(path),
                   "--latency-stat", "p90_us"])
    assert path.read_bytes() == original
    assert "DECIDE:" not in capsys.readouterr().out


@pytest.mark.parametrize("stat", ["p50_us", "p99_us"])
def test_old_input_without_unrequested_p90_remains_accepted(tmp_path, capsys, stat):
    path = tmp_path / "old-p50-p99.json"
    path.write_text(json.dumps({"label": "ASSUMED / synthetic old input", "results": [
        {"bytes": 10240, "p50_us": 100.0, "p99_us": 400.0}]}))
    assert tool.main(["--mode", "decide", "--latency-json", str(path),
                      "--latency-stat", stat]) == 0
    output = capsys.readouterr().out
    row = next(line for line in output.splitlines() if line.startswith("Qwen3.8-27B-CB"))
    assert re.search(r"\bWIN\b" if stat == "p50_us" else r"\bLOSE\b", row)
    assert f"verdict percentile: {stat}" in output


@pytest.mark.parametrize("matched", [False, True], ids=["largest-size-fallback", "exact-size"])
def test_selected_percentile_uses_the_chosen_size(tmp_path, capsys, matched):
    path = tmp_path / "assumed-sizes.json"
    rows = [{"bytes": 8192, "p50_us": 100.0, "p90_us": 400.0, "p99_us": 500.0},
            {"bytes": 16384, "p50_us": 100.0, "p90_us": 200.0, "p99_us": 500.0}]
    if matched:
        rows.append({"bytes": 10240, "p50_us": 100.0, "p90_us": 400.0, "p99_us": 500.0})
    path.write_text(json.dumps({"label": "ASSUMED / artificial size-specific inputs", "results": rows}))
    assert tool.main(["--mode", "decide", "--latency-json", str(path),
                      "--latency-stat", "p90_us"]) == 0
    output = capsys.readouterr().out
    lines = output.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("Qwen3.8-27B-CB"))
    assert re.search(r"\bLOSE\b" if matched else r"\bWIN\b", lines[index])
    coordinate = 400.0 if matched else 200.0
    assert f"verdict p90_us={coordinate:.1f} us" in lines[index + 1]
    assert ("matched size" if matched else "fallback: largest swept size (16384B)") in lines[index + 1]


def test_flat_assumption_uses_the_same_value_for_requested_percentile(capsys):
    assert tool.main(["--mode", "decide", "--latency-us", "400",
                      "--latency-stat", "p90_us"]) == 0
    output = capsys.readouterr().out
    row = next(line for line in output.splitlines() if line.startswith("Qwen3.8-27B-CB"))
    assert re.search(r"\bLOSE\b", row)
    assert "latency source: ASSUMED" in output
    assert "verdict p90_us=400.0 us" in output
    assert "UNKNOWN-NEEDS-TPOT (3)" in output
