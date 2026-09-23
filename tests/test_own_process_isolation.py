"""``own_process`` modules run in a child pytest in a shared session (PQ #1008).

The Stage A produced-output harness and the ten modules that import it need
a process of their own (one pinned ``prismabuild`` per process). In a
multi-file session they used to skip silently: 25 tests in one #996 shard.
``tests/conftest.py`` now replaces each marked test with a proxy under the
same node id; one child pytest per module runs exactly the chosen node ids,
and each proxy reports its own test's outcome.

These tests drive real pytest sessions over the samples in
``tests/own_process_samples``: ``sample_isolated`` (marked),
``sample_shared``, whose import plants a module that stands in for a second
``prismabuild``, ``sample_hangs`` (marked), whose one hanging test must
fail alone under the per-test bound (PQ #1027), and ``sample_bound_loaded``
(marked), which reports the bound it runs under when only
``PRISMABUILD_TEST_TIMEOUT_S`` asks for one (PQ #1055). The samples write
their process ids, so the tests check where each test ran, not only what it
reported.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import types
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = Path(__file__).resolve().parent / "own_process_samples"
ISOLATED = "tests/own_process_samples/sample_isolated.py"


def _session(tmp_path, *files, extra=(), environ=None):
    """Run one pytest session over ``files``; return what it reported."""
    out = tmp_path / "out"
    out.mkdir()
    junit = tmp_path / "junit.xml"
    env = {name: value for name, value in os.environ.items()
           if not name.startswith("PYTEST_") and name != "PQ_OWN_PROCESS_REPORT"}
    env["OWN_PROCESS_SAMPLE_OUT"] = str(out)
    # ``None`` removes a variable: a PrismaBuild shard exports the per-test
    # bound to every test, so "unset" has to be said.
    for name, value in (environ or {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    argv = [sys.executable, "-m", "pytest", "-q", "--no-header",
            "-p", "no:cacheprovider", "-c", str(ROOT / "pytest.ini"),
            "--rootdir", str(ROOT), "--basetemp", str(tmp_path / "base"),
            f"--junitxml={junit}", "-k", "not deselected", *extra,
            *[str(SAMPLES / name) for name in files]]
    proc = subprocess.Popen(argv, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        output, _ = proc.communicate(timeout=900)
    except subprocess.TimeoutExpired:
        # A session that outlives its own bound must not leave its child
        # pytest (and a sample's hour-long sleep) behind on the box.
        os.killpg(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate()
        pytest.fail(f"the sample session ran past 900 s:\n{output[-4000:]}")
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


#: The per-test bound the hanging sample runs under. pytest-timeout bounds a
#: test's setup and call together, and the first test in the child pays the
#: conftest's first imports in its setup: at 6 s that setup and a 3.5 s test
#: ran out the bound on a loaded sparklina, so the bound leaves wide room.
BOUND_S = 30

#: Each per-test bound a session can run under: the module that must be
#: importable, the parent session's arguments and environment, and the words
#: its failure must carry.
BOUNDS = {
    # PrismaBuild's bound, which pbtest exports to every shard, named with -p.
    "prismabuild": (
        "prismabuild.pytest_test_bound",
        ("-p", "prismabuild.pytest_test_bound"),
        {"PRISMABUILD_TEST_TIMEOUT_S": str(BOUND_S)},
        (f"per-test bound of {BOUND_S}s", "PRISMABUILD_TEST_TIMEOUT_S")),
    # The same bound from the variable alone, as a pbtest shard runs: the
    # conftest loads the plugin in the parent and in the child (PQ #1055).
    "prismabuild-env": (
        "prismabuild.pytest_test_bound",
        (),
        {"PRISMABUILD_TEST_TIMEOUT_S": str(BOUND_S)},
        (f"per-test bound of {BOUND_S}s", "PRISMABUILD_TEST_TIMEOUT_S")),
    # pytest-timeout, which CI runs under ``--timeout=300``.
    "pytest-timeout": (
        "pytest_timeout",
        (f"--timeout={BOUND_S}",),
        {"PRISMABUILD_TEST_TIMEOUT_S": ""},
        (f"Timeout (>{float(BOUND_S)}s)", "pytest-timeout")),
}


@pytest.mark.parametrize("bound", sorted(BOUNDS))
def test_a_hanging_test_fails_alone_under_the_per_test_bound(tmp_path, bound):
    """The bound applies to each test in the child, not to the module (#1027).

    The hang outlasts one bound by itself, so ``test_after`` passes only if
    the module is not bounded as a whole. Each environment runs the bound it
    has: the PrismaBuild venv has no pytest-timeout, and CI does not install
    ``prismabuild``.
    """
    module, args, environ, words = BOUNDS[bound]
    pytest.importorskip(module)
    proc, output, outcomes, out = _session(
        tmp_path, "sample_hangs.py", "sample_shared.py", extra=args,
        environ=environ)
    assert proc.returncode == 1, output
    assert set(outcomes) == {"test_before", "test_hangs", "test_after",
                             "test_shared"}, output
    state, message = outcomes["test_hangs"]
    assert state == "failed", output
    for word in words:
        assert word in message, message
    for name in ("test_before", "test_after", "test_shared"):
        assert outcomes[name] == ("passed", ""), output
    # One child ran all three, and it lived through the hang.
    child = {_pid(out, "before"), _pid(out, "hangs"), _pid(out, "after")}
    assert len(child) == 1 and proc.pid not in child, output


# -- the bound from the environment alone (PQ #1055) ---------------------------

#: The variable pbtest exports to every shard, and the plugin reads.
BOUND_ENV = "PRISMABUILD_TEST_TIMEOUT_S"


def _prismabuild_installed():
    import importlib.util
    if importlib.util.find_spec("prismabuild") is None:
        pytest.skip("prismabuild is not installed: CI never sets the bound")


def test_a_hanging_test_in_process_fails_alone_under_the_bound(tmp_path):
    """A module alone in a session runs in process, bounded per test (#1055).

    Only the variable is set, as on a pbtest shard: no ``-p``.
    """
    _prismabuild_installed()
    proc, output, outcomes, out = _session(
        tmp_path, "sample_hangs.py", environ={BOUND_ENV: str(BOUND_S)})
    assert proc.returncode == 1, output
    assert set(outcomes) == {"test_before", "test_hangs", "test_after"}, output
    state, message = outcomes["test_hangs"]
    assert state == "failed", output
    for word in (f"per-test bound of {BOUND_S}s", BOUND_ENV):
        assert word in message, message
    assert outcomes["test_before"] == outcomes["test_after"] == ("passed", "")
    assert {_pid(out, "before"), _pid(out, "hangs"), _pid(out, "after")} == {
        proc.pid}


def _bound_report(out):
    return json.loads((out / "bound.json").read_text())


@pytest.mark.parametrize("shared", [False, True], ids=["alone", "shared"])
def test_the_bound_loads_from_the_environment_without_importing_prismabuild(
        tmp_path, shared):
    """The variable loads the plugin, and ``prismabuild`` stays unimported.

    The Stage A harness fails a process whose ``prismabuild`` is not its
    pinned candidate (``_pb_source``), so the plugin is loaded from its file,
    in the session and in an ``own_process`` child alike.
    """
    _prismabuild_installed()
    files = ("sample_bound_loaded.py",) + (("sample_shared.py",) if shared else ())
    proc, output, _outcomes, out = _session(
        tmp_path, *files, environ={BOUND_ENV: str(BOUND_S)})
    assert proc.returncode == 0, output
    report = _bound_report(out)
    assert report["registered"] is True, report
    assert report["bound_s"] == float(BOUND_S), report
    assert report["prismabuild_imported"] is False, report
    # Alone, the module runs in process; shared, in its own child.
    assert (report["pid"] == proc.pid) is (not shared), report


@pytest.mark.parametrize("shared", [False, True], ids=["alone", "shared"])
def test_nothing_changes_when_the_bound_is_unset(tmp_path, shared):
    files = ("sample_bound_loaded.py",) + (("sample_shared.py",) if shared else ())
    proc, output, _outcomes, out = _session(
        tmp_path, *files, environ={BOUND_ENV: None})
    assert proc.returncode == 0, output
    report = _bound_report(out)
    assert report["registered"] is False, report
    assert report["bound_s"] is None, report
    assert report["prismabuild_imported"] is False, report


def test_a_bound_that_cannot_load_is_refused(monkeypatch):
    """The variable is set but no plugin file is found: refuse, never run
    unbounded and green."""
    import importlib.util

    import conftest

    registered = []
    config = types.SimpleNamespace(pluginmanager=types.SimpleNamespace(
        has_plugin=lambda name: False,
        register=lambda plugin, name: registered.append(name)))
    monkeypatch.setenv(BOUND_ENV, "30")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(pytest.UsageError, match=BOUND_ENV):
        conftest._load_prismabuild_test_bound(config)
    assert registered == []
    monkeypatch.setenv(BOUND_ENV, "")
    conftest._load_prismabuild_test_bound(config)
    assert registered == []
