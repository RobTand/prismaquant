"""Every session's output names its interpreter and gating versions (PQ #1090).

Six GLM modules skipped at collection in every PrismaBuild run for weeks,
because the test interpreter carried transformers 5.6.0 and they need 5.16.
No receipt said which transformers a run had, so nothing showed it.
"""
from __future__ import annotations

from importlib import metadata
import sys

import conftest


def test_the_terminal_summary_names_the_interpreter_and_its_versions():
    lines: list[str] = []

    class Reporter:
        def write_line(self, line):
            lines.append(line)

    conftest.pytest_terminal_summary(Reporter())

    (line,) = lines
    assert line.startswith("pq-test-environment: python=")
    assert f"transformers={metadata.version('transformers')}" in line
    assert f"torch={metadata.version('torch')}" in line
    assert line.endswith(f"({sys.executable})")
