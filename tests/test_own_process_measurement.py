"""Child-inclusive evidence for the suite cost census (PQ #2803)."""
from __future__ import annotations

import ast
import json
import pstats

import pytest

from tests.test_own_process_isolation import (
    ISOLATED, SAMPLES, _check_mapped_outcomes, _child_imports, _session,
)


@pytest.mark.parametrize("workers", [0, 2], ids=["serial", "xdist"])
def test_measurement_retains_child_phases_and_profile(tmp_path, workers):
    if workers:
        pytest.importorskip("xdist")
    proc, output, outcomes, out = _session(
        tmp_path, "sample_isolated.py", "sample_shared.py",
        extra=("-n", str(workers)) if workers else (),
        environ={"PQ_OWN_PROCESS_MEASURE": "1"})
    assert proc.returncode == 1, output
    _check_mapped_outcomes(outcomes, output)
    assert len(_child_imports(out)) == 1

    root = tmp_path / "base" / "own-process"
    metrics = [json.loads(path.read_text())
               for path in root.glob("*.measurement.json")]
    assert metrics, "the parent omitted its isolated child's cost"
    executed = [row for row in metrics if row["executed_child"]]
    assert len(executed) == 1
    assert {row["child_profile"] for row in metrics} == {
        executed[0]["child_profile"]}
    stats = pstats.Stats(executed[0]["child_profile"])
    assert any(name == "test_passes" and file.endswith("sample_isolated.py")
               for file, line, name in stats.stats)
    reports = [json.loads(line) for path in root.glob("*.jsonl")
               for line in path.read_text().splitlines()]
    phases = [row for row in reports if row["when"] != "collect"]
    assert all("duration" in row for row in phases)
    for row in metrics:
        assert set(row["child_phase_s"]) == {"setup", "call", "teardown"}
        for phase in row["child_phase_s"]:
            assert row["child_phase_s"][phase] == pytest.approx(sum(
                report["duration"] for report in phases
                if report["when"] == phase))
        assert row["child_unattributed_s"] == pytest.approx(
            row["child_elapsed_s"] - sum(row["child_phase_s"].values()))
        assert row["parent_elapsed_s"] >= row["lock_wait_s"] >= 0


def test_disabled_measurement_preserves_reports_without_artifacts(tmp_path):
    proc, output, outcomes, _out = _session(
        tmp_path, "sample_isolated.py", "sample_shared.py",
        environ={"PQ_OWN_PROCESS_MEASURE": None})
    assert proc.returncode == 1, output
    _check_mapped_outcomes(outcomes, output)
    root = tmp_path / "base" / "own-process"
    assert not list(root.glob("*.measurement.json"))
    assert not list(root.glob("*.pstats"))
    reports = [json.loads(line) for path in root.glob("*.jsonl")
               for line in path.read_text().splitlines()]
    assert all("duration" not in row for row in reports)


def test_child_only_line_has_its_test_coverage_context(tmp_path):
    coverage = pytest.importorskip("coverage")
    pytest.importorskip("pytest_cov")
    proc, output, outcomes, _out = _session(
        tmp_path, "sample_isolated.py", "sample_shared.py",
        extra=("--cov=tests.own_process_samples.sample_isolated",
               "--cov-context=test", "--cov-report="),
        environ={"PQ_OWN_PROCESS_MEASURE": "1",
                 "COVERAGE_FILE": str(tmp_path / "parent.coverage")})
    assert proc.returncode == 1, output
    _check_mapped_outcomes(outcomes, output)
    files = list((tmp_path / "base" / "own-process").glob("*.coverage"))
    assert len(files) == 1, "isolated child coverage must be retained separately"
    data = coverage.CoverageData(basename=str(files[0]))
    data.read()
    sample = SAMPLES / "sample_isolated.py"
    function = next(node for node in ast.parse(sample.read_text()).body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "test_passes")
    contexts = data.contexts_by_lineno(str(sample.resolve()))
    assert f"{ISOLATED}::test_passes|run" in contexts[function.body[0].lineno]
