"""``own_process`` modules run in a child pytest in a shared session (PQ #1008).

The Stage A produced-output harness and the ten modules that import it need
a process of their own (one pinned ``prismabuild`` per process). In a
multi-file session they used to skip silently: 25 tests in one #996 shard.
``tests/conftest.py`` now replaces each marked test with a proxy under the
same node id; one child pytest per module runs exactly the chosen node ids,
and each proxy reports its own test's outcome.

These tests drive real pytest sessions over two samples in
``tests/own_process_samples``: ``sample_isolated`` (marked) and
``sample_shared``, whose import plants a module that stands in for a second
``prismabuild``. The samples write their process ids, so the tests check
where each test ran, not only what it reported.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import types
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = Path(__file__).resolve().parent / "own_process_samples"
ISOLATED = "tests/own_process_samples/sample_isolated.py"


def _session(tmp_path, *files, extra=()):
    """Run one pytest session over ``files``; return what it reported."""
    out = tmp_path / "out"
    out.mkdir()
    junit = tmp_path / "junit.xml"
    env = {name: value for name, value in os.environ.items()
           if not name.startswith("PYTEST_") and name != "PQ_OWN_PROCESS_REPORT"}
    env["OWN_PROCESS_SAMPLE_OUT"] = str(out)
    argv = [sys.executable, "-m", "pytest", "-q", "--no-header",
            "-p", "no:cacheprovider", "-c", str(ROOT / "pytest.ini"),
            "--rootdir", str(ROOT), "--basetemp", str(tmp_path / "base"),
            f"--junitxml={junit}", "-k", "not deselected", *extra,
            *[str(SAMPLES / name) for name in files]]
    proc = subprocess.Popen(argv, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output, _ = proc.communicate(timeout=900)
    outcomes = {}
    for case in ET.parse(junit).getroot().iter("testcase"):
        name = case.get("name")
        failure = case.find("failure")
        skipped = case.find("skipped")
        if failure is not None:
            outcomes[name] = ("failed", (failure.get("message") or "")
                              + (failure.text or ""))
        elif skipped is not None:
            kind = ("xfailed" if skipped.get("type") == "pytest.xfail"
                    else "skipped")
            outcomes[name] = (kind, (skipped.get("message") or "")
                              + (skipped.text or ""))
        else:
            outcomes[name] = ("passed", "")
    return proc, output, outcomes, out


def _pid(out, name):
    return int((out / f"{name}.pid").read_text())


def _child_imports(out):
    path = out / "child-imports"
    return path.read_text().split() if path.exists() else []


def _check_mapped_outcomes(outcomes, output):
    assert set(outcomes) == {
        "test_passes", "test_fails", "test_skips", "test_xfails",
        "test_param[1]", "test_param[2]", "test_shared"}, output
    assert outcomes["test_passes"][0] == "passed", output
    assert outcomes["test_fails"][0] == "failed"
    assert "sample failure text 7f3a" in outcomes["test_fails"][1]
    assert outcomes["test_skips"] == ("skipped", outcomes["test_skips"][1])
    assert "sample skip reason 91c2" in outcomes["test_skips"][1]
    assert outcomes["test_xfails"][0] == "xfailed"
    assert "sample xfail reason 4be0" in outcomes["test_xfails"][1]
    assert outcomes["test_param[1]"][0] == outcomes["test_param[2]"][0] == "passed"
    assert outcomes["test_shared"][0] == "passed"


def test_a_shared_session_runs_the_marked_module_in_one_child(tmp_path):
    proc, output, outcomes, out = _session(
        tmp_path, "sample_isolated.py", "sample_shared.py")
    assert proc.returncode == 1, output
    _check_mapped_outcomes(outcomes, output)
    child = {_pid(out, "passes"), _pid(out, "param-1"), _pid(out, "param-2")}
    assert len(child) == 1 and proc.pid not in child
    assert _pid(out, "shared") == proc.pid
    assert _child_imports(out) == [str(pid) for pid in child]
    # The child ran what the parent chose: -k deselected this one.
    assert not (out / "deselected.pid").exists()


def test_a_session_of_the_module_alone_runs_in_process(tmp_path):
    proc, output, outcomes, out = _session(tmp_path, "sample_isolated.py")
    assert proc.returncode == 1, output
    assert outcomes["test_passes"][0] == "passed"
    assert outcomes["test_fails"][0] == "failed"
    assert _pid(out, "passes") == proc.pid
    assert _child_imports(out) == []


def test_xdist_workers_share_one_child_per_module(tmp_path):
    pytest.importorskip("xdist")
    proc, output, outcomes, out = _session(
        tmp_path, "sample_isolated.py", "sample_shared.py",
        extra=("-n", "2", "--dist", "worksteal"))
    assert proc.returncode == 1, output
    _check_mapped_outcomes(outcomes, output)
    assert len(_child_imports(out)) == 1, output


def test_a_test_the_child_never_reported_fails_by_name(tmp_path):
    """A child that reports nothing is a failure, never a pass or a skip."""
    import conftest

    nodeid = f"{ISOLATED}::test_passes"
    group = conftest._OwnProcessGroup([nodeid])
    group.result = {"state": {"returncode": 2, "error": None},
                    "outcomes": {}, "collection": [],
                    "log": tmp_path / "child.log"}
    proxy = types.SimpleNamespace(group=group, nodeid=nodeid, config=None)
    with pytest.raises(conftest.OwnProcessFailure, match="exit 2"):
        conftest.OwnProcessItem.runtest(proxy)


def test_a_test_that_never_reached_its_call_phase_is_not_a_pass(tmp_path):
    import conftest

    nodeid = f"{ISOLATED}::test_passes"
    outcomes, _ = conftest._fold_own_process_reports(
        [{"nodeid": nodeid, "when": "setup", "outcome": "passed",
          "longrepr": "", "reason": "", "wasxfail": None}],
        log=tmp_path / "child.log")
    assert outcomes[nodeid]["outcome"] == "failed"
    assert "never reported its call phase" in outcomes[nodeid]["message"]
