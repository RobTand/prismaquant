"""Frozen outcome tables for the consolidation byte-identity tests (PQ #1330).

When a consolidation moved a helper onto its owner (#1295), the test first ran
the helper's pre-change code next to the migrated call site. Those runs
passed, so the migrated site's outcome on each input *is* the old outcome.
This module freezes those outcomes into one JSON table per test module under
``tests/fixtures/``, keyed by the pytest node id and the call's index inside
the test, so the old code need not be kept as a second implementation.

An outcome is ``{"returned": repr(value)}`` or ``{"raised": "<module>.<type>",
"text": str(exc), "cause": ["<module>.<type>", str(cause)] | None}``. ``repr``
keeps key order, container types and floats exactly (a float's ``repr``
round-trips its bits). A temporary directory in any text is written as
``<tmp>``.

Set ``PQ_GOLDEN_RECORD=<directory>`` to rewrite the tables there instead of
checking them.

A table opts into ``every_row_used=True`` to also catch rows nothing checks:
a row left behind by a deleted test or parametrization, or a row a test no
longer reaches (an early return, a deleted loop). ``tests/conftest.py``
applies the check when one process ran the whole module; rows of tests that
skipped or failed there are exempt.
"""
from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
import json
import os
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def _qualified(kind: type) -> str:
    return f"{kind.__module__}.{kind.__qualname__}"


def outcome(call, *, tmp=None) -> dict:
    """What ``call()`` did, as a JSON row two runs can be compared by."""
    def text(value: str) -> str:
        return value if tmp is None else value.replace(str(tmp), "<tmp>")

    try:
        value = call()
    except BaseException as exc:  # noqa: BLE001 - the exception is the outcome
        cause = exc.__cause__
        return {"raised": _qualified(type(exc)), "text": text(str(exc)),
                "cause": None if cause is None else [_qualified(type(cause)), text(str(cause))]}
    return {"returned": text(repr(value))}


class GoldenTable:
    """One test module's frozen outcomes, ``tests/fixtures/<name>.json``."""

    def __init__(self, name: str, *, every_row_used: bool = False):
        self.name = name
        self.every_row_used = every_row_used
        self._record_dir = os.environ.get("PQ_GOLDEN_RECORD")
        self._rows: dict[str, dict] = {}
        self._calls: dict[str, int] = {}
        if not self._record_dir:
            self._rows = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))

    def _key(self) -> str:
        node = os.environ["PYTEST_CURRENT_TEST"].split("::", 1)[1].rsplit(" (", 1)[0]
        index = self._calls[node] = self._calls.get(node, -1) + 1
        return f"{node}#{index}"

    def check(self, row: dict) -> dict:
        key = self._key()
        if self._record_dir:
            self._rows[key] = row
            path = Path(self._record_dir) / f"{self.name}.json"
            rows = ",\n".join(f"{json.dumps(k)}: {json.dumps(v, sort_keys=True)}"
                              for k, v in sorted(self._rows.items()))
            path.write_text("{\n" + rows + "\n}\n", encoding="utf-8")
        else:
            assert key in self._rows, f"{key} has no row in fixtures/{self.name}.json"
            assert row == self._rows[key], key
        return row

    def call(self, call, *, tmp=None) -> dict:
        """Check what ``call()`` returns or raises."""
        return self.check(outcome(call, tmp=tmp))

    def value(self, value, *, tmp=None) -> dict:
        """Check an already computed value by its ``repr``."""
        return self.check(outcome(lambda: value, tmp=tmp))

    def unused_rows(self, *, unfinished: Collection[str]) -> list[str]:
        """Rows no call in this process used, except rows of ``unfinished`` tests.

        A row key is ``<test>#<call index>``. A test that passed with fewer
        calls than it has rows leaves rows unused, and so does a row whose test
        no longer exists. A recording run checks nothing.
        """
        if self._record_dir:
            return []
        used = {f"{node}#{index}" for node, last in self._calls.items()
                for index in range(last + 1)}
        return sorted(key for key in self._rows
                      if key not in used and key.rsplit("#", 1)[0] not in unfinished)


def unfinished_tests_if_whole_module_ran(module: Path, *, args: Iterable[str],
                                         deselected: Collection[Path],
                                         outcomes: Mapping[str, str | None]) -> set[str] | None:
    """The module's tests that did not pass, or None if this process did not run it whole.

    ``outcomes`` maps the node id of each selected test in the module to the
    outcome it reached in this process, or None if it did not run here (it ran
    in another pytest-xdist worker, or ``-x`` stopped the run). A module
    narrowed by a ``file::test`` argument, ``-k``, ``-m`` or ``--deselect`` was
    not selected whole, so rows of its unselected tests are not orphans.
    """
    if module in deselected:
        return None
    if any("::" in arg and Path(arg.split("::", 1)[0]).name == module.name for arg in args):
        return None
    if not outcomes or any(result is None for result in outcomes.values()):
        return None
    return {nodeid.split("::", 1)[1] for nodeid, result in outcomes.items()
            if result != "passed"}
