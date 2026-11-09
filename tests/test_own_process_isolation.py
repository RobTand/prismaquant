"""``own_process`` modules run in a child pytest in a shared session (PQ #1008).

Some modules need a process of their own (an import graph no other module in the
session has touched). In a multi-file session they used to skip silently: 25
tests in one #996 shard.
``tests/conftest.py`` now replaces each marked test with a proxy under the
same node id; one child pytest per module runs exactly the chosen node ids,
and each proxy reports its own test's outcome.

These tests drive real pytest sessions over the samples in
``tests/own_process_samples``: ``sample_isolated`` (marked),
``sample_shared``, whose import plants a foreign module, ``sample_hangs``
(marked), whose one hanging test must fail alone under the declared
pytest-timeout bound (PQ #1027), ``sample_collection_error`` (marked), whose
collection fails, and ``sample_child_dies`` (marked), whose child is killed
mid-module. The samples write their process ids, so the tests check where each
test ran, not only what it reported.
"""
from __future__ import annotations

import ast
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
    # ``None`` removes a variable, so "unset" can be said.
    for name, value in (environ or {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    argv = [sys.executable, "-s", "-m", "pytest", "-q", "--no-header",
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
    cases = list(ET.parse(junit).getroot().iter("testcase")) if junit.exists() else []
    for case in cases:
        name = case.get("name")
        failure = case.find("failure")
        if failure is None:
            failure = case.find("error")
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


@pytest.mark.parametrize("workers", [0, 2], ids=["serial", "xdist"])
def test_module_skip_reports_original_source_location(tmp_path, workers):
    if workers:
        pytest.importorskip("xdist")
    extra = ("-n", str(workers)) if workers else ()
    proc, output, outcomes, _out = _session(
        tmp_path, "sample_marked_skip.py", "sample_shared.py", extra=extra)
    assert proc.returncode == 0, output
    assert set(outcomes) == {"test_marked_skip", "test_shared"}, output
    assert outcomes["test_shared"] == ("passed", "")
    kind, reason = outcomes["test_marked_skip"]
    assert kind == "skipped" and "sample module skip 2113" in reason
    sample = SAMPLES / "sample_marked_skip.py"
    location = next(node.lineno for node in ast.parse(sample.read_text()).body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "test_marked_skip")
    assert f"{sample}:{location}" in reason, reason


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

#: pytest-timeout's registration in each plugin state: autoloaded (the default,
#: registered as ``timeout``) and autoload disabled with the plugin named (``-p``).
#: A child shares its parent's environment, so it is in the same state.
AUTOLOAD_ON = ((f"--timeout={BOUND_S}",), {})
AUTOLOAD_OFF = (("-p", "pytest_timeout", f"--timeout={BOUND_S}"), {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
STATES = {"autoload-on": AUTOLOAD_ON, "autoload-off": AUTOLOAD_OFF}


@pytest.mark.parametrize("state", sorted(STATES))
def test_a_hanging_test_fails_alone_under_the_declared_per_test_bound(tmp_path, state):
    """The bound applies to each test in the child, not to the module (#1027).

    The hang outlasts one bound by itself, so ``test_after`` passes only if the module is not
    bounded as a whole. pytest-timeout is the declared bound (``pyproject.toml`` test extra).
    """
    args, environ = STATES[state]
    proc, output, outcomes, out = _session(
        tmp_path, "sample_hangs.py", "sample_shared.py", extra=args, environ=environ)
    assert proc.returncode == 1, output
    assert set(outcomes) == {"test_before", "test_hangs", "test_after", "test_shared"}, output
    kind, message = outcomes["test_hangs"]
    assert kind == "failed", output
    for word in (f"Timeout (>{float(BOUND_S)}s)", "pytest-timeout"):
        assert word in message, message
    for name in ("test_before", "test_after", "test_shared"):
        assert outcomes[name] == ("passed", ""), output
    child = {_pid(out, "before"), _pid(out, "hangs"), _pid(out, "after")}
    assert len(child) == 1 and proc.pid not in child, output


@pytest.mark.parametrize("state", sorted(STATES))
def test_a_hanging_test_in_process_fails_alone_under_the_declared_bound(tmp_path, state):
    args, environ = STATES[state]
    proc, output, outcomes, out = _session(tmp_path, "sample_hangs.py", extra=args, environ=environ)
    assert proc.returncode == 1, output
    assert set(outcomes) == {"test_before", "test_hangs", "test_after"}, output
    assert outcomes["test_hangs"][0] == "failed" and f"Timeout (>{float(BOUND_S)}s)" in outcomes["test_hangs"][1], output
    assert outcomes["test_before"] == outcomes["test_after"] == ("passed", "")
    assert {_pid(out, "before"), _pid(out, "hangs"), _pid(out, "after")} == {proc.pid}


@pytest.mark.parametrize("variant", ["plugin-disabled", "autoload-off-unnamed"])
def test_a_requested_bound_without_the_plugin_is_a_failure_never_a_skip(tmp_path, variant):
    """pytest refuses an unrecognized ``--timeout``: usage error, exit 4, nothing runs, nothing skips."""
    extra, environ = ((("-p", "no:timeout", "--timeout=30"), {}) if variant == "plugin-disabled"
                      else ((f"--timeout=30",), {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}))
    proc, output, outcomes, _out = _session(tmp_path, "sample_isolated.py", "sample_shared.py", extra=extra, environ=environ)
    assert proc.returncode == 4, output
    assert "unrecognized arguments: --timeout" in output, output
    assert outcomes == {}, output


def test_a_child_that_cannot_load_the_plugin_fails_its_proxies_by_exit_status(tmp_path):
    """The parent has the plugin, its own-process child cannot: every proxy fails, none passes or skips.

    The shadow module raises ImportError only inside a child (the variable the parent sets for it) and is the
    real plugin everywhere else.
    """
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "pytest_timeout.py").write_text(
        "import os, sys\nfrom pathlib import Path\n"
        "if os.environ.get('PQ_OWN_PROCESS_REPORT'):\n"
        "    raise ImportError('sample: pytest-timeout unavailable in the child')\n"
        "here = Path(__file__).resolve().parent\n"
        "real = next(Path(p) / 'pytest_timeout.py' for p in sys.path if p and Path(p).resolve() != here\n"
        "            and (Path(p) / 'pytest_timeout.py').is_file())\n"
        "exec(compile(real.read_text(), str(real), 'exec'), globals())\n")
    args, environ = AUTOLOAD_OFF
    proc, output, outcomes, _out = _session(
        tmp_path, "sample_isolated.py", "sample_shared.py", extra=args,
        environ={**environ, "PYTHONPATH": str(shadow)})
    assert proc.returncode == 1, output
    assert outcomes["test_shared"] == ("passed", ""), output
    for name in ("test_passes", "test_fails", "test_skips", "test_xfails", "test_param[1]", "test_param[2]"):
        kind, message = outcomes[name]
        assert kind == "failed", (name, kind, output)
        assert "own pytest process" in message or "reported nothing" in message, message


def test_a_collection_error_in_an_own_process_module_fails_the_session(tmp_path):
    proc, output, outcomes, _out = _session(tmp_path, "sample_collection_error.py", "sample_shared.py")
    assert proc.returncode == 2, output
    assert "own_process_sample_module_that_does_not_exist" in output, output
    assert not any(kind == "skipped" for kind, _ in outcomes.values()), outcomes
    assert any(kind == "failed" for kind, _ in outcomes.values()), outcomes


def test_a_killed_child_fails_every_test_it_did_not_report_by_name(tmp_path):
    proc, output, outcomes, out = _session(tmp_path, "sample_child_dies.py", "sample_shared.py")
    assert proc.returncode == 1, output
    assert outcomes["test_reports_before_dying"] == ("passed", ""), output
    for name in ("test_kills_its_process", "test_never_runs"):
        kind, message = outcomes[name]
        assert kind == "failed", (name, kind, output)
        assert "never reported" in message or "reported nothing" in message or "exited unexpectedly" in message, message
    assert outcomes["test_shared"] == ("passed", "")


@pytest.mark.parametrize("registered, expected", [
    ({}, ["--timeout=30"]),
    ({"timeout": object(), "pytest_timeout": object()}, ["--timeout=30"]),
    ({"timeout": object()}, ["--timeout=30"]),
    ({"pytest_timeout": object()}, ["-p", "pytest_timeout", "--timeout=30"]),
])
def test_the_child_names_the_plugin_only_when_the_parent_did(registered, expected):
    import conftest

    manager = types.SimpleNamespace(has_plugin=lambda name: name in registered)
    config = types.SimpleNamespace(option=types.SimpleNamespace(timeout=30), pluginmanager=manager)
    assert conftest._own_process_bound_args(config) == (expected if registered else ["--timeout=30"])


def test_no_bound_requested_means_no_bound_argument():
    import conftest

    config = types.SimpleNamespace(option=types.SimpleNamespace(timeout=None),
                                   pluginmanager=types.SimpleNamespace(has_plugin=lambda name: True))
    assert conftest._own_process_bound_args(config) == []
