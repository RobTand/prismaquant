"""Child-inclusive evidence for the suite cost census (PQ #2803)."""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import errno
import json
from pathlib import Path
import pstats
from types import SimpleNamespace

import pytest

from tests.test_own_process_isolation import (
    ISOLATED, SAMPLES, _check_mapped_outcomes, _child_imports, _session,
)


@pytest.mark.parametrize("returncode", [0, 7], ids=["passed", "unexpected-exit"])
def test_diagnostic_write_failure_does_not_relaunch_finished_child(
        tmp_path, monkeypatch, returncode):
    import conftest

    monkeypatch.setenv("PQ_OWN_PROCESS_MEASURE", "1")
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    root = tmp_path / "own-process"
    launches, failures, terminals = [], [], []
    original_write = Path.write_text
    nodeids = ("sample.py::test_passes",)

    def write(path, content, *args, **kwargs):
        if path.name.endswith(".child-measurement.json") and not failures:
            failures.append(path)
            raise OSError(errno.ENOSPC, "injected diagnostic write failure")
        return original_write(path, content, *args, **kwargs)

    def child(argv, **kwargs):
        launches.append(argv)
        rows = [{"nodeid": nodeids[0], "when": phase, "outcome": "passed",
                 "duration": 0.0} for phase in ("setup", "call", "teardown")]
        Path(kwargs["env"]["PQ_OWN_PROCESS_REPORT"]).write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
        return SimpleNamespace(returncode=returncode)

    config = SimpleNamespace(
        _tmp_path_factory=SimpleNamespace(getbasetemp=lambda: tmp_path),
        option=SimpleNamespace(), rootpath=tmp_path, inipath=None)
    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(conftest.subprocess, "run", child)
    errors = []
    for _ in range(2):
        try:
            conftest._run_own_process(config, nodeids)
        except OSError as exc:
            errors.append(exc)
        terminals.append([json.loads(path.read_text())
                          for path in root.glob("*.done.json")])

    assert len(failures) == 1 and errors[0].errno == errno.ENOSPC
    assert len(launches) == 1, "missing diagnostics relaunched a completed child"
    state = {"returncode": returncode, "error": None}
    assert terminals == [[state], [state]]
    monkeypatch.delenv("PQ_OWN_PROCESS_MEASURE")
    result = conftest._run_own_process(config, nodeids)
    assert len(launches) == 1 and result["state"] == state
    assert result["outcomes"][nodeids[0]]["outcome"] == (
        "passed" if returncode == 0 else "failed")


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
    assert all(row["child_returncode"] == 1 for row in metrics)
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
    assert not list(root.glob("*.child-measurement.json"))
    assert not list(root.glob("*.pstats"))
    reports = [json.loads(line) for path in root.glob("*.jsonl")
               for line in path.read_text().splitlines()]
    assert all("duration" not in row for row in reports)


def test_collection_skip_preserves_reason_without_phase_duration(monkeypatch):
    import conftest

    monkeypatch.setenv("PQ_OWN_PROCESS_MEASURE", "1")
    report = pytest.CollectReport(
        nodeid="sample.py", outcome="skipped",
        longrepr=("sample.py", 1, "Skipped: child collection evidence"), result=[])
    record = conftest._own_process_record(report)
    assert record["when"] == "collect" and record["outcome"] == "skipped"
    assert record["reason"] == "child collection evidence"
    assert "duration" not in record


@pytest.mark.parametrize("workers", [0, 2], ids=["serial", "xdist"])
def test_child_only_line_has_its_test_coverage_context(tmp_path, workers):
    coverage = pytest.importorskip("coverage")
    pytest.importorskip("pytest_cov")
    if workers:
        pytest.importorskip("xdist")
    parent = tmp_path / "parent.coverage"
    proc, output, outcomes, _out = _session(
        tmp_path, "sample_isolated.py", "sample_shared.py",
        extra=(("-n", str(workers)) if workers else ()) + (
            f"--cov={SAMPLES}", "--cov-context=test", "--cov-report="),
        environ={"PQ_OWN_PROCESS_MEASURE": "1",
                 "COVERAGE_FILE": str(parent)})
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
    combined = coverage.Coverage(data_file=str(tmp_path / "combined.coverage"))
    combined.combine(data_paths=[str(parent), str(files[0])], keep=True, strict=True)
    combined.save()
    assert f"{ISOLATED}::test_passes|run" in combined.get_data().contexts_by_lineno(
        str(sample.resolve()))[function.body[0].lineno]
    assert parent.is_file() and files[0].is_file()


@pytest.mark.parametrize("sources,disabled,expected", [
    (["prismaquant", "tools"], False, ["--cov=prismaquant", "--cov=tools"]),
    (None, False, ["--cov="]),
    ([], False, []),
    (["prismaquant"], True, []),
])
def test_child_coverage_arguments_preserve_selection(tmp_path, sources, disabled,
                                                    expected):
    import conftest

    option = SimpleNamespace(cov_source=sources, no_cov=disabled,
                             cov_config="custom.coveragerc", cov_branch=True,
                             cov_context="test")
    launcher, args, env = conftest._own_process_measure_args(
        SimpleNamespace(option=option), tmp_path, "child")
    assert launcher == ["-m", "tests._own_process_profile",
                        str(tmp_path / "child.pstats")]
    assert [arg for arg in args if arg.startswith("--cov=")] == expected
    if expected:
        assert args[:2] == ["-p", "pytest_cov"]
        assert "--cov-config=custom.coveragerc" in args
        assert "--cov-branch" in args and "--cov-context=test" in args
        assert "--cov-report=" in args and "--cov-fail-under=0" in args
        assert env == {"COVERAGE_FILE": str(tmp_path / "child.coverage")}
    else:
        assert args == [] and env == {}


def test_concurrent_sessions_keep_independent_artifacts(tmp_path):
    roots = [tmp_path / name for name in ("one", "two")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        sessions = []
        for root in roots:
            root.mkdir()
            sessions.append(pool.submit(
                _session, root, "sample_isolated.py", "sample_shared.py",
                environ={"PQ_OWN_PROCESS_MEASURE": "1"}))
        for session in sessions:
            proc, output, outcomes, _ = session.result()
            assert proc.returncode == 1, output
            _check_mapped_outcomes(outcomes, output)
    profiles = []
    for root in roots:
        rows = list((root / "base" / "own-process").glob("*.measurement.json"))
        assert len(rows) == 1
        record = json.loads(rows[0].read_text())
        assert record["executed_child"] is True
        profile = record["child_profile"]
        assert any(name == "test_passes" for _, _, name in pstats.Stats(profile).stats)
        profiles.append(profile)
    assert profiles[0] != profiles[1]
