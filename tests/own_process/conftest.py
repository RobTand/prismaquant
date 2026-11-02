"""Run marked tests in one child per module in a shared pytest session.

Scoped here so changes to child execution do not invalidate unrelated Bazel tests.
The root conftest keeps session-wide resets, opt-ins, time bounds and reporting.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.pytest_test_bound import test_bound_args as _test_bound_args

OWN_PROCESS_MARK = "own_process"
OWN_PROCESS_REPORT_ENV = "PQ_OWN_PROCESS_REPORT"
OWN_PROCESS_MEASURE_ENV = "PQ_OWN_PROCESS_MEASURE"


class OwnProcessFailure(Exception):
    """A test failed, or never reported, in its module's own process."""


def _own_process_root(config) -> Path:
    base = Path(config._tmp_path_factory.getbasetemp())
    if os.environ.get("PYTEST_XDIST_WORKER"):
        # Every xdist worker's basetemp sits under the run's one root.
        base = base.parent
    return base / "own-process"


def _fold_own_process_reports(records, *, log):
    """Per node id, one outcome from a child's phase reports."""
    outcomes, collection = {}, []
    for record in records:
        if record["when"] == "collect":
            collection.append(record)
            continue
        folded = outcomes.setdefault(record["nodeid"],
                                     {"outcome": "passed", "phases": []})
        folded["phases"].append(record["when"])
        if record["outcome"] == "failed":
            message = f"[{record['when']}] {record['longrepr']}"
            if folded["outcome"] == "failed":
                folded["message"] += "\n" + message
            else:
                folded.update(outcome="failed", message=message)
        elif record["outcome"] == "skipped" and folded["outcome"] != "failed":
            if record.get("wasxfail") is not None:
                folded.update(outcome="xfailed", reason=record["wasxfail"])
            else:
                folded.update(outcome="skipped", reason=record["reason"])
    for folded in outcomes.values():
        # A test counts as passed only once its call phase reported: a child
        # that died inside a test leaves a setup report and nothing else.
        if folded["outcome"] == "passed" and "call" not in folded["phases"]:
            folded.update(outcome="failed", message=(
                "the test never reported its call phase in its own process "
                f"(phases: {folded['phases']}); see {log}"))
    return outcomes, collection


def _own_process_measure_args(config, root, key):
    launcher = ["-m", "tests._own_process_profile", str(root / f"{key}.pstats")]
    argv, env = [], {}
    sources = getattr(config.option, "cov_source", [])
    if sources != [] and not getattr(config.option, "no_cov", False):
        argv = ["-p", "pytest_cov", "--cov-reset", "--cov-report=",
                "--cov-fail-under=0"]
        argv += [f"--cov={source}" for source in (sources or [""])]
        argv += ["--cov-config=" + str(config.option.cov_config)]
        if config.option.cov_branch:
            argv.append("--cov-branch")
        if config.option.cov_context:
            argv.append("--cov-context=" + config.option.cov_context)
        env["COVERAGE_FILE"] = str(root / f"{key}.coverage")
    return launcher, argv, env


def _write_own_process_measurement(root, key, nodeids, state, records,
                                   started, lock_wait, executed_child):
    phases = {phase: sum(row["duration"] for row in records
                        if row["when"] == phase)
              for phase in ("setup", "call", "teardown")}
    child = json.loads((root / f"{key}.child-measurement.json").read_text())
    record = {
        "schema": "prismaquant.own_process_measurement.v1",
        "nodeids": nodeids, "parent_pid": os.getpid(),
        "executed_child": executed_child, "lock_wait_s": lock_wait,
        "parent_elapsed_s": time.perf_counter() - started,
        "child_elapsed_s": child["elapsed_s"], "child_phase_s": phases,
        "child_unattributed_s": child["elapsed_s"] - sum(phases.values()),
        "child_profile": child["profile"], "child_coverage": child["coverage"],
        "child_returncode": state["returncode"], "child_error": state["error"],
    }
    (root / f"{key}.{os.getpid()}.measurement.json").write_text(
        json.dumps(record))


def _run_own_process(config, nodeids: tuple[str, ...]) -> dict:
    """Run ``nodeids`` in one child pytest, once per session, and read it."""
    measure = os.environ.get(OWN_PROCESS_MEASURE_ENV) == "1"
    started = time.perf_counter() if measure else None
    executed_child = False
    key = hashlib.sha256("\n".join(nodeids).encode()).hexdigest()[:16]
    root = _own_process_root(config)
    root.mkdir(parents=True, exist_ok=True)
    reports = root / f"{key}.jsonl"
    done = root / f"{key}.done.json"
    log = root / f"{key}.log"
    with open(root / f"{key}.lock", "a+") as lock:
        waiting = time.perf_counter() if measure else None
        fcntl.flock(lock, fcntl.LOCK_EX)
        lock_wait = time.perf_counter() - waiting if measure else None
        try:
            if not done.exists():
                executed_child = True
                reports.unlink(missing_ok=True)
                env = {name: value for name, value in os.environ.items()
                       if not name.startswith("PYTEST_XDIST_")
                       and name not in ("PYTEST_ADDOPTS", "PYTEST_CURRENT_TEST")}
                env[OWN_PROCESS_REPORT_ENV] = str(reports)
                launcher, measure_args = ["-m", "pytest"], []
                if measure:
                    launcher, measure_args, measure_env = _own_process_measure_args(
                        config, root, key)
                    env.update(measure_env)
                argv = [sys.executable, *launcher, "-q", "--no-header",
                        "-p", "no:cacheprovider",
                        "--rootdir", str(config.rootpath),
                        "--basetemp", str(root / f"{key}.tmp")]
                if config.inipath is not None:
                    argv += ["-c", str(config.inipath)]
                argv += _test_bound_args(config)
                argv += measure_args
                argv += list(nodeids)
                returncode, error = None, None
                child_started = time.perf_counter() if measure else None
                try:
                    with open(log, "wb") as out:
                        returncode = subprocess.run(
                            argv, cwd=str(config.rootpath), env=env,
                            stdout=out, stderr=subprocess.STDOUT).returncode
                except BaseException as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    state = {"returncode": returncode, "error": error}
                    if measure:
                        child_measurement = {
                            "elapsed_s": time.perf_counter() - child_started,
                            "profile": str(root / f"{key}.pstats"),
                            "coverage": measure_env.get("COVERAGE_FILE"),
                        }
                    done.write_text(json.dumps(state))
                    if measure:
                        (root / f"{key}.child-measurement.json").write_text(
                            json.dumps(child_measurement))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    state = json.loads(done.read_text())
    lines = reports.read_text().splitlines() if reports.exists() else []
    records = [json.loads(line) for line in lines]
    if measure:
        _write_own_process_measurement(root, key, nodeids, state, records,
                                       started, lock_wait, executed_child)
    outcomes, collection = _fold_own_process_reports(records, log=log)
    reported_failure = any(item["outcome"] == "failed" for item in outcomes.values())
    collection_failure = any(item["outcome"] == "failed" for item in collection)
    collection_skip = (not outcomes and collection
                       and all(item["outcome"] == "skipped" for item in collection))
    expected_exit = (state["returncode"] == pytest.ExitCode.OK
                     or (state["returncode"] == pytest.ExitCode.TESTS_FAILED
                         and (reported_failure or collection_failure))
                     or (state["returncode"] == pytest.ExitCode.NO_TESTS_COLLECTED
                         and collection_skip))
    if state.get("error") or not expected_exit:
        # Keep ordinary failed/skip/xfail mappings. A failure after the last
        # report still needs one proxy to carry the child's terminal failure.
        target = next((nodeid for nodeid in nodeids
                       if outcomes.get(nodeid, {}).get("outcome") == "failed"), None)
        if target is None:
            target = next((nodeid for nodeid in nodeids
                           if outcomes.get(nodeid, {}).get("outcome") == "passed"),
                          nodeids[0])
        folded = outcomes.setdefault(target, {"outcome": "passed", "phases": []})
        message = (f"the module's own pytest process exited unexpectedly "
                   f"(exit {state['returncode']}"
                   + (f", {state['error']}" if state.get("error") else "")
                   + f"); see {log}")
        if folded["outcome"] == "failed":
            message = folded["message"] + "\n" + message
        elif collection_failure:
            message += "\n" + "\n".join(item["longrepr"] for item in collection
                                           if item["outcome"] == "failed")
        folded.update(outcome="failed", message=message)
    return {"state": state, "outcomes": outcomes, "collection": collection,
            "log": log}


class _OwnProcessGroup:
    """One own-process module's collected node ids, run at most once."""

    def __init__(self, nodeids):
        self.nodeids = tuple(nodeids)
        self.result = None
        self.error = None

    def start(self, config):
        """Run the child before any test's bound is armed.

        An error is kept for the proxies to report as test failures: raised
        here, outside any test, it would end the session instead.
        """
        if self.result is None and self.error is None:
            try:
                self.result = _run_own_process(config, self.nodeids)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"

    def run(self, config):
        if self.error is not None:
            raise OwnProcessFailure(
                f"the module's own pytest process did not run: {self.error}")
        if self.result is None:
            self.result = _run_own_process(config, self.nodeids)
        return self.result


class OwnProcessItem(pytest.Item):
    """Stands in for one test of an own-process module in a shared session."""

    def __init__(self, *, group, source_reportinfo, **kwargs):
        super().__init__(**kwargs)
        self.group = group
        self.source_reportinfo = source_reportinfo

    def runtest(self):
        result = self.group.run(self.config)
        outcome = result["outcomes"].get(self.nodeid)
        if outcome is None:
            for record in result["collection"]:
                if record["outcome"] == "skipped":
                    pytest.skip(record["reason"])
            try:
                tail = result["log"].read_text(errors="replace").splitlines()[-40:]
            except OSError:
                tail = []
            state = result["state"]
            raise OwnProcessFailure(
                f"{self.nodeid} reported nothing from its own pytest process "
                f"(exit {state['returncode']}"
                + (f", {state['error']}" if state.get("error") else "")
                + f"; log {result['log']}):\n" + "\n".join(
                    [record["longrepr"] for record in result["collection"]
                     if record["outcome"] == "failed"] + tail))
        if outcome["outcome"] == "failed":
            raise OwnProcessFailure(outcome["message"])
        if outcome["outcome"] == "xfailed":
            pytest.xfail(outcome["reason"])
        if outcome["outcome"] == "skipped":
            pytest.skip(outcome["reason"])

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, OwnProcessFailure):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def reportinfo(self):
        path, line, _description = self.source_reportinfo
        return path, line, f"{self.nodeid} (own process)"


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    """Give each ``own_process`` module its own process in a shared session.

    ``trylast`` so that ``-k``, ``-m`` and every other plugin have already
    chosen the items: the child runs exactly the node ids chosen here.
    """
    if os.environ.get(OWN_PROCESS_REPORT_ENV):
        return
    if len({item.path for item in items}) < 2:
        return
    modules: dict[Path, list[int]] = {}
    for index, item in enumerate(items):
        if item.get_closest_marker(OWN_PROCESS_MARK) is not None:
            modules.setdefault(item.path, []).append(index)
    for indices in modules.values():
        group = _OwnProcessGroup(items[index].nodeid for index in indices)
        for index in indices:
            original = items[index]
            items[index] = OwnProcessItem.from_parent(
                original.parent, name=original.name, group=group,
                source_reportinfo=original.reportinfo())


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """Start an own-process module's child outside every per-test bound.

    The first proxy used to start the child inside its own call phase, so one
    test's bound covered the whole module (PQ #1027): a module could time out
    on its test count, and the proxy that failed was the first one, not the
    one that hung. ``tryfirst`` makes this the outermost wrapper, so the child
    runs before pytest-timeout's protocol timer or PrismaBuild's per-phase
    alarms are armed for the proxy, and each test in the child runs under a
    bound of its own (``test_bound_args``). The child's collection
    and exit are bounded as the parent session's own are, by the run's
    ceiling. Its wall time is not attributed to any proxy in ``--durations``
    or the junit report.
    """
    if isinstance(item, OwnProcessItem):
        item.group.start(item.config)
    return (yield)


def _own_process_record(report) -> dict:
    reason = ""
    if report.skipped:
        longrepr = report.longrepr
        reason = (str(longrepr[2]) if isinstance(longrepr, tuple)
                  and len(longrepr) == 3 else str(longrepr or ""))
        if reason.startswith("Skipped: "):
            reason = reason[len("Skipped: "):]
    record = {"nodeid": report.nodeid, "when": getattr(report, "when", "collect"),
            "outcome": report.outcome,
            "longrepr": report.longreprtext if report.failed else "",
            "reason": reason, "wasxfail": getattr(report, "wasxfail", None)}
    if (os.environ.get(OWN_PROCESS_MEASURE_ENV) == "1"
            and isinstance(report, pytest.TestReport)):
        record["duration"] = report.duration
    return record


def _append_own_process_record(report) -> None:
    path = os.environ.get(OWN_PROCESS_REPORT_ENV)
    if path:
        with open(path, "a", encoding="utf-8") as out:
            out.write(json.dumps(_own_process_record(report)) + "\n")


def pytest_runtest_logreport(report):
    """In a child session, hand each phase's report to the parent."""
    _append_own_process_record(report)


def pytest_collectreport(report):
    """In a child session, hand a failed or skipped collection to the parent."""
    if not report.passed:
        _append_own_process_record(report)


