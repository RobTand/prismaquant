"""The dispatcher reads ``pbwait``'s answer the way ``pbwait`` gives it (PQ #1197).

``Gateway.is_terminal_executed`` decides whether a band-serial producer has
run, and so whether its consumer may dispatch (``_producer_handoff``).
``pbwait`` prints its ``key status transport job host elapsed rc receipt
note`` table and returns its verdict as the exit code; it prints no JSON. A
gateway that parsed the last line as JSON read every key as not executed, so
no band-serial row could ever dispatch. These tests run the real gateway
against a stand-in ``pbwait`` that prints that table.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import dispatch_joint_quanta  # noqa: E402

KEY = "d106121a45e4e1d4dcaa88bc3beb26badc314aef70c27d560e5918247411699d"
HEADER = "key           status    transport  job  host    elapsed  rc  receipt  note"


def _fake_pbwait(tmp_path, monkeypatch, *, rows, exit_code):
    """Point the gateway at a ``pbwait`` that prints ``rows`` and exits."""
    script = tmp_path / "pbwait.py"
    table = "\n".join([HEADER, *rows])
    script.write_text(textwrap.dedent(f"""\
        import sys
        assert sys.argv[1:3] == ["--wait-s", "0"], sys.argv
        print({table!r})
        sys.exit({exit_code})
        """))
    monkeypatch.setattr(dispatch_joint_quanta, "PBWAIT", script)
    return dispatch_joint_quanta.Gateway()


def test_an_executed_producer_reads_as_executed(tmp_path, monkeypatch):
    # The row pbwait printed for layer 043 (PB d106121a45e4), 2026-09-24.
    gateway = _fake_pbwait(tmp_path, monkeypatch, exit_code=0, rows=[
        "d106121a45e4  executed  pool       -    sparky  2504.8s  0   -        -"])
    assert gateway.is_terminal_executed(KEY) is True


def test_a_memoized_producer_reads_as_executed(tmp_path, monkeypatch):
    """``pbwait.verdict`` counts ``cache_hit`` as done: the same result."""
    gateway = _fake_pbwait(tmp_path, monkeypatch, exit_code=0, rows=[
        "d106121a45e4  cache_hit  cas        -    -       -        -   cas/a    -"])
    assert gateway.is_terminal_executed(KEY) is True


@pytest.mark.parametrize("row, exit_code", [
    ("d106121a45e4  waiting   pool       -    sparky  12.0s    -   -        running", 3),
    ("d106121a45e4  failed    pool       -    sparky  12.0s    1   -        -", 1),
])
def test_an_unfinished_or_failed_producer_does_not(tmp_path, monkeypatch, row, exit_code):
    gateway = _fake_pbwait(tmp_path, monkeypatch, rows=[row], exit_code=exit_code)
    assert gateway.is_terminal_executed(KEY) is False


def test_the_verdict_must_be_about_this_key(tmp_path, monkeypatch):
    """A zero exit about another action says nothing about this one."""
    gateway = _fake_pbwait(tmp_path, monkeypatch, exit_code=0, rows=[
        "9bf8f58d174f  executed  pool       -    lina    12.0s    0   -        -"])
    assert gateway.is_terminal_executed(KEY) is False


def test_a_row_that_says_executed_under_a_failing_verdict_does_not(tmp_path, monkeypatch):
    """The exit code is the fleet's own reading; the row alone is not enough."""
    gateway = _fake_pbwait(tmp_path, monkeypatch, exit_code=1, rows=[
        "d106121a45e4  executed  pool       -    sparky  12.0s    1 (action 137)  -  killed"])
    assert gateway.is_terminal_executed(KEY) is False
