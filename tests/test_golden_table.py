"""The golden table's opt-in check that every frozen row is used.

``tests/conftest.py`` runs the check at module teardown; these tests hold the
two decisions it delegates to ``tests/golden_table.py``: which rows are
unused, and whether this process ran the whole module.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests import golden_table
from tests.golden_table import GoldenTable, unfinished_tests_if_whole_module_ran

MODULE = Path("/repo/tests/test_m.py")
ROWS = {"test_a#0": {"returned": "1"}, "test_a#1": {"returned": "2"},
        "test_b[x]#0": {"returned": "3"}, "test_gone[y]#0": {"returned": "4"}}


@pytest.fixture
def table(tmp_path, monkeypatch):
    monkeypatch.delenv("PQ_GOLDEN_RECORD", raising=False)
    monkeypatch.setattr(golden_table, "FIXTURES", tmp_path)
    (tmp_path / "rows.json").write_text(json.dumps(ROWS), encoding="utf-8")
    return GoldenTable("rows", every_row_used=True)


def _run(monkeypatch, table, node, values):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", f"tests/test_m.py::{node} (call)")
    for value in values:
        table.value(value)


def test_a_row_whose_test_no_longer_exists_is_unused(table, monkeypatch):
    _run(monkeypatch, table, "test_a", [1, 2])
    _run(monkeypatch, table, "test_b[x]", [3])
    assert table.unused_rows(unfinished=set()) == ["test_gone[y]#0"]


def test_a_test_that_stops_early_leaves_its_later_rows_unused(table, monkeypatch):
    _run(monkeypatch, table, "test_a", [1])
    _run(monkeypatch, table, "test_b[x]", [3])
    assert table.unused_rows(unfinished=set()) == ["test_a#1", "test_gone[y]#0"]


def test_rows_of_a_test_that_did_not_pass_are_exempt(table, monkeypatch):
    _run(monkeypatch, table, "test_a", [1, 2])
    assert table.unused_rows(unfinished={"test_b[x]", "test_gone[y]"}) == []


def test_the_check_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(golden_table, "FIXTURES", tmp_path)
    (tmp_path / "rows.json").write_text(json.dumps(ROWS), encoding="utf-8")
    assert GoldenTable("rows").every_row_used is False


def _decide(outcomes, *, args=("tests",), deselected=()):
    return unfinished_tests_if_whole_module_ran(
        MODULE, args=list(args), deselected=set(deselected), outcomes=outcomes)


def test_a_whole_module_run_reports_the_tests_that_did_not_pass():
    outcomes = {"tests/test_m.py::test_a": "passed",
                "tests/test_m.py::test_b[x]": "skipped",
                "tests/test_m.py::Case::test_c": "failed"}
    assert _decide(outcomes) == {"test_b[x]", "Case::test_c"}
    assert _decide(outcomes, args=["tests/test_m.py"]) == {"test_b[x]", "Case::test_c"}


@pytest.mark.parametrize("args,deselected,outcomes", [
    # Another pytest-xdist worker ran test_b, or -x stopped before it.
    (["tests"], [], {"tests/test_m.py::test_a": "passed", "tests/test_m.py::test_b[x]": None}),
    # -k, -m or --deselect dropped some of the module's tests.
    (["tests"], [MODULE], {"tests/test_m.py::test_a": "passed"}),
    # A file::test argument narrowed collection to some of the module.
    (["tests/test_m.py::test_a"], [], {"tests/test_m.py::test_a": "passed"}),
    # Nothing of the module was selected.
    (["tests"], [], {}),
])
def test_a_partial_run_checks_nothing(args, deselected, outcomes):
    assert _decide(outcomes, args=args, deselected=deselected) is None
