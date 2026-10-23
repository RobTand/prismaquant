"""Real nested pytest adapter; no fleet, GPU or workload execution claim."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("child_exit", [0, 23], ids=["normal-child", "failed-after-reports"])
def test_outer_pytest_requires_successful_own_process_child(tmp_path, child_exit):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "pytest.ini").write_text("[pytest]\nmarkers = own_process: isolated module\n")
    # Load the actual repository owner in both outer and child pytest.
    (suite / "conftest.py").write_text('pytest_plugins = ("tests.conftest",)\n')
    (suite / "test_isolated.py").write_text('''import atexit
import json
import os
from pathlib import Path
import pytest
pytestmark = pytest.mark.own_process

def test_reported():
    def after_reports():
        report = Path(os.environ["PQ_OWN_PROCESS_REPORT"])
        records = [json.loads(line) for line in report.read_text().splitlines()]
        phases = [(row["when"], row["outcome"]) for row in records
                  if row["nodeid"].endswith("::test_reported")]
        marker = {"pid": os.getpid(), "phases": phases,
                  "requested_exit": int(os.environ["OWN_PROCESS_AUDIT_EXIT"])}
        Path(os.environ["OWN_PROCESS_AUDIT_MARKER"]).write_text(json.dumps(marker))
        if marker["requested_exit"]:
            os._exit(marker["requested_exit"])
    atexit.register(after_reports)
''')
    (suite / "test_shared.py").write_text('''import os
from pathlib import Path

def test_shared():
    Path(os.environ["OWN_PROCESS_AUDIT_PARENT"]).write_text(str(os.getpid()))
''')
    marker_path = tmp_path / "child-finalization.json"
    parent_path = tmp_path / "parent.pid"
    junit = tmp_path / "outer.xml"
    base = tmp_path / "base"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("PYTEST_") and key != "PQ_OWN_PROCESS_REPORT"}
    env.update({
        "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "tests"), env.get("PYTHONPATH", "")]),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONUNBUFFERED": "1",
        "OWN_PROCESS_AUDIT_EXIT": str(child_exit),
        "OWN_PROCESS_AUDIT_MARKER": str(marker_path),
        "OWN_PROCESS_AUDIT_PARENT": str(parent_path),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    })
    argv = [sys.executable, "-m", "pytest", "-q", "--no-header",
            "-p", "no:cacheprovider", "-c", str(suite / "pytest.ini"),
            "--rootdir", str(suite), "--basetemp", str(base),
            f"--junitxml={junit}", str(suite / "test_isolated.py"),
            str(suite / "test_shared.py")]
    process = subprocess.Popen(argv, cwd=ROOT, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        output, _ = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail(f"Nested pytest exceeded its bounded control window:\n{output[-4000:]}")
    marker = json.loads(marker_path.read_bytes())
    assert marker["pid"] != process.pid
    assert marker["requested_exit"] == child_exit
    assert marker["phases"] == [["setup", "passed"], ["call", "passed"], ["teardown", "passed"]]
    states = list((base / "own-process").glob("*.done.json"))
    assert len(states) == 1, output
    state = json.loads(states[0].read_bytes())
    assert state == {"returncode": child_exit, "error": None}, output
    assert int(parent_path.read_text()) == process.pid
    cases = {case.get("name"): case for case in ET.parse(junit).getroot().iter("testcase")}
    assert set(cases) == {"test_reported", "test_shared"}, output
    assert cases["test_shared"].find("failure") is None, output
    if child_exit:
        assert process.returncode != 0, (
            "The real child exited23 after complete passed setup/call/teardown reports, "
            "but outer pytest accepted its proxy as green.\n" + output)
        assert cases["test_reported"].find("failure") is not None, output
    else:
        assert process.returncode == 0, output
        assert cases["test_reported"].find("failure") is None, output


@pytest.mark.parametrize("mode", ["signal", "mixed-outcomes", "missing-collectors", "empty-child"])
def test_child_terminal_status_preserves_reported_outcomes(tmp_path, mode):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "pytest.ini").write_text("[pytest]\nmarkers = own_process: isolated module\n")
    (suite / "conftest.py").write_text('''import os
pytest_plugins = ("tests.conftest",)

def pytest_collection_modifyitems(items):
    if os.environ.get("PQ_OWN_PROCESS_REPORT") and os.environ["OWN_PROCESS_AUDIT_MODE"] == "empty-child":
        items.clear()
''')
    (suite / "test_isolated.py").write_text('''import atexit
import json
import os
from pathlib import Path
import signal
import pytest
pytestmark = pytest.mark.own_process
mode = os.environ["OWN_PROCESS_AUDIT_MODE"]
if mode == "missing-collectors" and os.environ.get("PQ_OWN_PROCESS_REPORT"):
    pytest.skip("child collection skip evidence", allow_module_level=True)

def test_pass():
    if mode == "signal":
        def after_reports():
            rows = [json.loads(line) for line in Path(os.environ["PQ_OWN_PROCESS_REPORT"]).read_text().splitlines()]
            phases = [(row["when"], row["outcome"]) for row in rows if row["nodeid"].endswith("::test_pass")]
            Path(os.environ["OWN_PROCESS_AUDIT_MARKER"]).write_text(json.dumps(phases))
            os.kill(os.getpid(), signal.SIGTERM)
        atexit.register(after_reports)

def test_failure():
    if mode == "mixed-outcomes":
        pytest.fail("original child failure evidence")

def test_skip():
    pytest.skip("original child skip evidence")

@pytest.mark.xfail(reason="original child xfail evidence", strict=True)
def test_xfail():
    assert False
''')
    (suite / "test_shared.py").write_text("def test_shared():\n    pass\n")
    marker = tmp_path / "signal-after-reports.json"
    junit, base = tmp_path / "outer.xml", tmp_path / "base"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("PYTEST_") and key != "PQ_OWN_PROCESS_REPORT"}
    env.update({"PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "tests"), env.get("PYTHONPATH", "")]),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "OWN_PROCESS_AUDIT_MODE": mode,
                "OWN_PROCESS_AUDIT_MARKER": str(marker), "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    argv = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
            "-c", str(suite / "pytest.ini"), "--rootdir", str(suite),
            "--basetemp", str(base), f"--junitxml={junit}",
            str(suite / "test_isolated.py"), str(suite / "test_shared.py")]
    process = subprocess.Popen(argv, cwd=ROOT, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        output, _ = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail(f"Nested pytest exceeded its bounded control window:\n{output[-4000:]}")
    states = list((base / "own-process").glob("*.done.json"))
    assert len(states) == 1, output
    state = json.loads(states[0].read_bytes())
    assert state["error"] is None, output
    cases = {case.get("name"): case for case in ET.parse(junit).getroot().iter("testcase")}
    assert set(cases) == {"test_pass", "test_failure", "test_skip", "test_xfail", "test_shared"}, output
    assert cases["test_shared"].find("failure") is None, output
    if mode in ("signal", "empty-child", "missing-collectors"):
        # A child-only module skip removes the explicit node IDs selected by
        # the parent. Pytest reports missing collectors (usage4), not no-tests5.
        expected = {"signal": -signal.SIGTERM, "empty-child": 5,
                    "missing-collectors": 4}[mode]
        assert state["returncode"] == expected, output
        assert process.returncode != 0, output
        assert cases["test_pass"].find("failure") is not None, output
        if mode == "signal":
            assert json.loads(marker.read_bytes()) == [["setup", "passed"], ["call", "passed"], ["teardown", "passed"]]
            assert cases["test_failure"].find("failure") is None, output
    elif mode == "mixed-outcomes":
        assert state["returncode"] == 1 and process.returncode == 1, output
        assert cases["test_pass"].find("failure") is None, output
        assert "original child failure evidence" in ET.tostring(cases["test_failure"], encoding="unicode"), output
    if mode in ("signal", "mixed-outcomes"):
        assert "original child skip evidence" in ET.tostring(cases["test_skip"], encoding="unicode"), output
        assert cases["test_xfail"].find("skipped").get("type") == "pytest.xfail", output


@pytest.mark.parametrize("collection_skip", [True, False], ids=["recorded-skip", "truly-empty"])
def test_public_collection_skip_is_distinct_from_an_empty_selection(tmp_path, collection_skip):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "pytest.ini").write_text("[pytest]\n")
    (suite / "conftest.py").write_text('pytest_plugins = ("tests.conftest",)\n')
    sample = suite / "test_collection.py"
    sample.write_text('import pytest\npytest.skip("intentional collection skip", allow_module_level=True)\n'
                      if collection_skip else "# Intentionally contains no tests.\n")
    junit = tmp_path / "collection.xml"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("PYTEST_") and key != "PQ_OWN_PROCESS_REPORT"}
    env.update({"PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "tests"), env.get("PYTHONPATH", "")]),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header",
                             "-p", "no:cacheprovider", "-c", str(suite / "pytest.ini"),
                             "--rootdir", str(suite), f"--junitxml={junit}", str(sample)],
                            cwd=ROOT, env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == (0 if collection_skip else 5), result.stdout + result.stderr
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    if collection_skip:
        assert len(cases) == 1 and cases[0].find("skipped") is not None
        assert "intentional collection skip" in ET.tostring(cases[0], encoding="unicode")
    else:
        assert cases == []
