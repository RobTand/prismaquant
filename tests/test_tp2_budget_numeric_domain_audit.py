"""The public residency planner must refuse invalid physical input domains."""

import json
import struct

import pytest

from tools import tp2_budget_plan as planner


@pytest.fixture
def artifact(tmp_path):
    root = tmp_path / "artifact"
    root.mkdir()
    header = {}
    for layer in range(2):
        header[f"model.layers.{layer}.self_attn.q_proj.weight"] = {
            "dtype": "BF16", "shape": [2, 2],
            "data_offsets": [layer * 8, (layer + 1) * 8]}
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(16))
    return root


def _run(artifact, capsys, *options):
    argv = ["--artifact", str(artifact), "--parallelism", "pp:2", *options]
    try:
        code = planner.main(argv)
    except SystemExit as refused:
        code = refused.code
    captured = capsys.readouterr()
    assert type(code) is int
    return code, captured.out, captured.err


def _feasibility_lines(stdout):
    return "\n".join(line for line in stdout.splitlines() if any(
        token in line for token in ("per-device", "overhead/rank", "MAX-FEASIBLE",
                                    "body scale", "MAX ARTIFACT", "OVER-BUDGET")))


@pytest.mark.parametrize("flag", ["--per-device-gb", "--overhead-gb-per-rank"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-1"])
def test_public_planner_refuses_invalid_budget_inputs(artifact, capsys, flag, value):
    other = ("--overhead-gb-per-rank=0" if flag == "--per-device-gb"
             else "--per-device-gb=1")
    code, stdout, stderr = _run(artifact, capsys, other, f"{flag}={value}")
    assert code != 0, (
        f"public planner accepted {flag}={value}\n"
        f"{_feasibility_lines(stdout)}\n{stderr}")
    reason = stderr.lower().replace("_", "-")
    assert "per-device" in reason or "budget" in reason or "overhead" in reason
    assert "TP2 CLUSTER BUDGET PLAN" not in stdout


@pytest.mark.parametrize("budget,overhead,verdict", [
    ("1", "0", "MAX-FEASIBLE-ARTIFACT  ["),
    ("0", "0", "MAX-FEASIBLE-ARTIFACT: INFEASIBLE"),
    ("1", "2", "MAX-FEASIBLE-ARTIFACT: INFEASIBLE"),
    ("1", None, "MAX-FEASIBLE-ARTIFACT: UNKNOWN-NEEDS-OVERHEAD"),
])
def test_valid_inputs_keep_actual_cli_feasibility(artifact, capsys, budget, overhead, verdict):
    options = [f"--per-device-gb={budget}"]
    if overhead is not None:
        options.append(f"--overhead-gb-per-rank={overhead}")
    code, stdout, stderr = _run(artifact, capsys, *options)
    assert code == 0, stderr
    assert verdict in stdout, _feasibility_lines(stdout)
    assert "TENSOR INVENTORY" in stdout
    if budget == "1" and overhead == "0":
        assert "MAX ARTIFACT" in stdout


@pytest.mark.parametrize("flag", ["--per-device-gb", "--overhead-gb-per-rank"])
@pytest.mark.parametrize("value", ["nan", "-1"])
def test_invalid_physical_input_refuses_before_headers_or_report_mutation(
        artifact, tmp_path, monkeypatch, capsys, flag, value):
    output = tmp_path / "existing-plan.json"
    original = b"independent previous plan\n"
    output.write_bytes(original)

    def unexpected_read(*args, **kwargs):
        pytest.fail("invalid physical input reached artifact/header traversal")

    monkeypatch.setattr(planner, "collect_artifact_tensors", unexpected_read)
    code, stdout, stderr = _run(
        artifact, capsys, f"{flag}={value}", "--json-out", str(output))
    assert code == 2 and "finite and nonnegative" in stderr
    assert "TP2 CLUSTER BUDGET PLAN" not in stdout
    assert output.read_bytes() == original
