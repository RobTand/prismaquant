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
"""
from __future__ import annotations

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

    def __init__(self, name: str):
        self.name = name
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
