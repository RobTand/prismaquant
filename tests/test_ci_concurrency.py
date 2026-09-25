"""CI's concurrency cancels superseded PR runs only (#962).

Every push to main must get a complete hosted run: cancelling main's run when
the next merge landed left no hosted run certifying any tree on a busy day. A
shared group without ``cancel-in-progress`` is not enough either, because a
concurrency group keeps only one pending run and cancels the rest, so a main
run's group must be its own.

The two expressions are evaluated here for each trigger with a small
translation of GitHub's ``&&``/``||``/``==`` (which, like Python's
``and``/``or``, return an operand), so the test pins behaviour, not spelling.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _concurrency() -> dict[str, str]:
    block = re.search(r"(?ms)^concurrency:\n((?:  .*\n)+)", WORKFLOW.read_text(encoding="utf-8"))
    assert block is not None, "ci.yml has no top-level concurrency block"
    return dict(re.findall(r"(?m)^  ([a-z-]+): (.*)$", block.group(1)))


def _evaluate(value: str, github: dict[str, str]):
    """Render a workflow value: literal text with ``${{ ... }}`` expressions."""

    def expression(match: re.Match) -> str:
        text = match.group(1).replace("&&", " and ").replace("||", " or ")
        text = re.sub(r"\bgithub\.([a-z_]+)", lambda m: repr(github[m.group(1)]), text)
        return str(eval(text, {"__builtins__": {}}))  # noqa: S307 - fixed test input

    rendered = re.sub(r"\$\{\{(.*?)\}\}", expression, value)
    return {"True": True, "False": False, "true": True, "false": False}.get(rendered, rendered)


def _run(event: str, ref: str, run_id: str) -> dict:
    github = {"workflow": "CI", "event_name": event, "ref": ref, "run_id": run_id,
              "sha": "0" * 40}
    concurrency = _concurrency()
    return {"group": _evaluate(concurrency["group"], github),
            "cancel": _evaluate(concurrency["cancel-in-progress"], github)}


def test_a_new_pr_push_cancels_the_superseded_run_of_that_pr():
    first = _run("pull_request", "refs/pull/7/merge", "1")
    second = _run("pull_request", "refs/pull/7/merge", "2")
    other = _run("pull_request", "refs/pull/8/merge", "3")
    assert second["cancel"] is True
    assert first["group"] == second["group"]
    assert other["group"] != first["group"]


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_a_main_run_is_never_cancelled_or_replaced_by_the_next_one(event):
    first = _run(event, "refs/heads/main", "1")
    second = _run(event, "refs/heads/main", "2")
    assert first["cancel"] is False and second["cancel"] is False
    # Distinct groups: a shared group would still drop all but one pending run.
    assert first["group"] != second["group"]
