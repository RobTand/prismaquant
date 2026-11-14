"""CI's concurrency cancels superseded PR runs only (#962), and main runs one at a time.

A running main run is never cancelled, so a busy day still certifies trees
(#962). Main shares one group: GitHub then runs one main run at a time and
keeps only the newest pending one, so the newest commit always gets a run and
overlapping full-population runs no longer overload the executor (mode B).

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
def test_main_runs_queue_in_one_group_and_a_running_one_is_never_cancelled(event):
    first = _run(event, "refs/heads/main", "1")
    second = _run(event, "refs/heads/main", "2")
    assert first["cancel"] is False and second["cancel"] is False
    assert first["group"] == second["group"]
