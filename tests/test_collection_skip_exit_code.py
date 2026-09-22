"""A shard whose only file skips at collection exits 0, not 5 (PQ #915).

``pbtest`` gives every shard its own pytest over its own files, so pytest's
``EXIT_NO_TESTS_COLLECTED`` is the verdict the fleet reads. A module that
skips at collection time -- ``pytest.importorskip`` at module scope, directly
or through an imported sibling test module -- collects nothing, so the shard
printed ``1 skipped`` and exited 5, and read as failed.

These run a real pytest in a subprocess over a copy of this tree's
``conftest.py``, because the hook under test IS a conftest hook: asserting it
by calling the function would assert the call, not the exit code pytest
actually returns.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

CONFTEST = Path(__file__).resolve().parent / "conftest.py"
REPO = CONFTEST.parent.parent


def _run(directory: Path, *args: str) -> subprocess.CompletedProcess:
    shutil.copyfile(CONFTEST, directory / "conftest.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["PYTEST_ADDOPTS"] = ""
    env.pop("PRISMABUILD_TEST_TIMEOUT_S", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", str(directory),
         *args],
        cwd=str(directory), capture_output=True, text=True, env=env,
    )


def test_collection_time_skip_alone_is_green(tmp_path) -> None:
    """One module, skipped at collection: ``1 skipped`` and exit 0."""

    (tmp_path / "test_only_skips.py").write_text(
        "import pytest\n"
        "pytest.importorskip('a_module_that_does_not_exist_pq915')\n"
        "\n"
        "def test_never_runs():\n"
        "    raise AssertionError('collection should have skipped this')\n")
    done = _run(tmp_path)
    assert "1 skipped" in done.stdout, done.stdout + done.stderr
    assert done.returncode == 0, done.stdout + done.stderr


def test_an_empty_collection_with_no_skip_still_exits_5(tmp_path) -> None:
    """Nothing collected and nothing skipped stays a defect, not a pass."""

    (tmp_path / "test_has_no_tests.py").write_text(
        "def helper():\n    return 1\n")
    done = _run(tmp_path)
    assert done.returncode == 5, done.stdout + done.stderr


def test_a_failure_beside_a_collection_skip_still_fails(tmp_path) -> None:
    """The conversion never reaches a run that collected and failed."""

    (tmp_path / "test_only_skips.py").write_text(
        "import pytest\n"
        "pytest.importorskip('a_module_that_does_not_exist_pq915')\n")
    (tmp_path / "test_fails.py").write_text(
        "def test_fails():\n    assert False\n")
    done = _run(tmp_path)
    assert done.returncode == 1, done.stdout + done.stderr
