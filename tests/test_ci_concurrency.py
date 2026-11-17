"""CI's concurrency cancels superseded PR runs only (#962), and the record runs one at a time.

A running main run is never cancelled, so a busy day still certifies trees
(#962). Scheduled and manual records share one group: GitHub runs one at a time
and keeps only the newest pending one, so overlapping full-population runs never
overload the executor (mode B). Each push has its own group: every merge gets its
cheap hosted lanes (#3763).

The two expressions are evaluated here for each trigger with a small
translation of GitHub's ``&&``/``||``/``==`` (which, like Python's
``and``/``or``, return an operand), so the test pins behaviour, not spelling.
"""
from __future__ import annotations

import re
from pathlib import Path

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


def _run(event: str, ref: str, run_id: str, sha: str = "0" * 40) -> dict:
    github = {"workflow": "CI", "event_name": event, "ref": ref, "run_id": run_id, "sha": sha}
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


def test_records_queue_in_one_group_and_a_running_one_is_never_cancelled():
    first = _run("workflow_dispatch", "refs/heads/main", "1", "a" * 40)
    second = _run("workflow_dispatch", "refs/heads/main", "2", "b" * 40)
    assert first["cancel"] is False and second["cancel"] is False
    assert first["group"] == second["group"] != _run("push", "refs/heads/main", "3", "b" * 40)["group"]


def test_each_push_to_main_has_its_own_group_and_is_never_cancelled():
    first, second = (_run("push", "refs/heads/main", run, sha) for run, sha in (("1", "a" * 40), ("2", "b" * 40)))
    assert first["group"] != second["group"] and first["cancel"] is False is second["cancel"]
