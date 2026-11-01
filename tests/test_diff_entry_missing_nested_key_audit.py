"""Real single-entry diagnostic CLI must distinguish absence from JSON null."""
from __future__ import annotations

import json
import sys

import pytest

from experiments.stream_row_head_640 import diff_entry as cli


def _run(tmp_path, monkeypatch, a, b):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for root, value in ((left, a), (right, b)):
        (root / "artifact.json").write_text(json.dumps(value))
    monkeypatch.setattr(sys, "argv", [
        "diff_entry.py", "--a", "left=" + str(left),
        "--b", "right=" + str(right), "artifact.json"])
    return cli.main()


@pytest.mark.parametrize("missing_arm", ["left", "right"])
def test_missing_nested_key_is_reported_with_distinct_absence_marker(
    tmp_path, monkeypatch, capsys, missing_arm,
):
    present = {"nested": {"unchanged": 1, "extra": None}}
    missing = {"nested": {"unchanged": 1}}
    a, b = ((missing, present) if missing_arm == "left" else (present, missing))
    assert _run(tmp_path, monkeypatch, a, b) == 1
    output = capsys.readouterr().out
    assert "nested.extra" in output
    lines = output.splitlines()
    absent = next(line for line in lines if line.strip().startswith(missing_arm + ":"))
    other = "right" if missing_arm == "left" else "left"
    value = next(line for line in lines if line.strip().startswith(other + ":"))
    assert any(marker in absent.lower() for marker in ("missing", "absent")), absent
    assert "None" in value and absent.split(":", 1)[1] != value.split(":", 1)[1]


def test_equal_json_keeps_zero_exit(tmp_path, monkeypatch, capsys):
    value = {"nested": {"value": 2}}
    assert _run(tmp_path, monkeypatch, value, value) == 0
    assert "sha256 equal: True" in capsys.readouterr().out


def test_existing_nested_value_change_keeps_both_values(tmp_path, monkeypatch, capsys):
    assert _run(tmp_path, monkeypatch, {"nested": {"value": 2}},
                {"nested": {"value": 3}}) == 1
    output = capsys.readouterr().out
    assert "nested.value" in output and "left: 2" in output and "right: 3" in output
