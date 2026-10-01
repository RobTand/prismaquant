"""``fleet_data`` tests skip unless asked for (PQ #1014).

A test that reads fleet-local campaign data or live PrismaBuild state that
no PB action declares is marked ``fleet_data``. ``tests/conftest.py`` skips
it by default with a reason that names the opt-in, and runs it when ``-m``
names the mark (the switch ``pbtest --pytest-args`` accepts) or when
``PQ_FLEET_DATA_TESTS=1``. A session of nothing but such tests exits 0, under
xdist too. These tests drive real pytest sessions over
``tests/fleet_data_samples``.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = Path(__file__).resolve().parent / "fleet_data_samples" / "sample_fleet_data.py"


def _session(tmp_path, *extra, environ=None):
    env = {name: value for name, value in os.environ.items()
           if not name.startswith("PYTEST_") and name != "PQ_FLEET_DATA_TESTS"}
    env.update(environ or {})
    argv = [sys.executable, "-m", "pytest", "-v", "--no-header",
            "-p", "no:cacheprovider", "-c", str(ROOT / "pytest.ini"),
            "--rootdir", str(ROOT), "--basetemp", str(tmp_path / "base"),
            *extra, str(SAMPLE)]
    proc = subprocess.run(argv, cwd=ROOT, env=env, text=True, timeout=600,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.returncode == 0, proc.stdout
    return proc.stdout


@pytest.mark.parametrize("extra", [(), ("-n", "2")], ids=["serial", "xdist"])
def test_fleet_data_skips_by_default(tmp_path, extra):
    if extra:
        pytest.importorskip("xdist")
    output = _session(tmp_path, "-rs", *extra)
    assert "test_reads_nothing" in output and "1 passed" in output
    assert "1 skipped" in output
    assert "run with -m fleet_data or PQ_FLEET_DATA_TESTS=1" in output


@pytest.mark.parametrize("extra", [(), ("-n", "2")], ids=["serial", "xdist"])
def test_a_session_of_only_fleet_data_exits_zero(tmp_path, extra):
    if extra:
        pytest.importorskip("xdist")
    output = _session(tmp_path, "-k", "reads_fleet_data", *extra)
    assert "1 skipped" in output


@pytest.mark.parametrize("extra, environ", [
    (("-m", "fleet_data"), None),
    ((), {"PQ_FLEET_DATA_TESTS": "1"}),
], ids=["mark", "environment"])
def test_fleet_data_runs_when_asked_for(tmp_path, extra, environ):
    output = _session(tmp_path, *extra, environ=environ)
    assert "test_reads_fleet_data PASSED" in output
    assert "skipped" not in output


#: The decorator this test looks for, spelled as its dotted parts.
#: ``pbtest`` decides fleet data per file by a textual search for the mark,
#: so writing the dotted name out here would make it leave this hermetic
#: file out of every unmarked run (PQ #1954).
FLEET_DATA_DECORATOR = ["pytest", "mark", "fleet_data"]

#: Each real fleet-data read #1014 names lives in a file of its own, so the
#: hermetic tests that used to share a file with it run in the merge queue
#: (PQ #1954).
MARKED_FILES = {
    "test_tessera_census_stats_glm53.py": {
        "test_glm53_probe_expands_onto_the_census_roster"},
    "test_glm_joint_data_manifest_at_submit_fleet_data.py": {
        "test_the_real_joint_pass_read_set_is_terabytes_in_bounded_phases"},
    "test_dispatch_shared_tag_placement_fleet_data.py": {
        "test_dispatcher_tags_are_placeable_on_the_live_gb10_fleet"},
}

#: The hermetic halves the reads were split out of: they mark nothing.
SPLIT_FROM = ("test_glm_joint_data_manifest_at_submit.py",
              "test_dispatch_shared_tag_placement.py")


def _marked_tests(name):
    import ast

    tree = ast.parse((ROOT / "tests" / name).read_text())
    return {node.name for node in tree.body
            if isinstance(node, ast.FunctionDef) and any(
                ast.unparse(decorator).split(".") == FLEET_DATA_DECORATOR
                for decorator in node.decorator_list)}


def _test_names(name):
    import ast

    tree = ast.parse((ROOT / "tests" / name).read_text())
    return {node.name for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}


def test_the_marked_real_data_tests_carry_the_mark():
    """The three reads #1014 names are marked, each alone in its file."""
    for name, expected in MARKED_FILES.items():
        assert _marked_tests(name) == expected, name
        tree_tests = _test_names(name)
        assert tree_tests == expected, (
            f"{name} holds unmarked tests {sorted(tree_tests - expected)}: "
            "pbtest leaves the whole file out of an unmarked run")
    for name in SPLIT_FROM:
        assert _marked_tests(name) == set(), name

