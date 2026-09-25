"""A re-declared Stage B plan rides its own path, never the sealed one (PQ #1191).

In dev mode (PQ #1147) the dispatcher read a row's plan from the record's
``campaign.plan_path`` and dispatched under the digest of the bytes it read.
The only way to hand a row a re-declared plan was to write it over the sealed
plan path, and every later band prepare, which reads that path against the
sealed digest, then refused ("extended plan digest mismatch").

``--execution-plan PATH`` names the re-declared plan instead. Every
execution-time plan read of the dispatch takes that file, the row carries
``--plan PATH`` with that file's digest, and the sealed plan file is never
written. Dev mode stamps a differing digest; certified mode refuses it, as it
refuses a re-declared plan at the sealed path.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from test_dispatch_redeclared_plan_coverage_1147 import (  # noqa: E402,F401
    _campaign, _main, dispatch,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _redeclare(plan_path: Path, target: Path, *, resource_bound: bool) -> Path:
    """An execution plan with another resource policy, at its own path."""
    plan = json.loads(plan_path.read_text())
    if resource_bound:
        plan["stage_b_resource_policy"] = {"path": "/resource-gpu68",
                                           "sha256": "1" * 64}
    else:
        plan["source_prefetch"]["prefetch_workers"] = 2
    target.write_text(json.dumps(plan, sort_keys=True))
    return target


def _rows(out: str) -> list[dict]:
    return json.loads(out[out.index("{\n"):])


@pytest.mark.parametrize("resource_bound", [True, False],
                         ids=["resource-plan", "plain-plan"])
def test_dev_mode_row_runs_the_execution_plan_and_keeps_the_sealed_file(
        tmp_path, dispatch, monkeypatch, capsys, resource_bound):
    plan_path, _prepared, argv = _campaign(tmp_path, resource_bound=resource_bound)
    sealed_bytes = plan_path.read_bytes()
    execution = _redeclare(plan_path, tmp_path / "execution-plan.json",
                           resource_bound=resource_bound)
    assert _sha(execution) != _sha(plan_path)
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    code, out, err = _main(dispatch, [*argv, "--execution-plan", str(execution)],
                           capsys)
    assert code == 0, err
    assert "[DEV-MODE] seal execution plan differs" in out
    document = _rows(out)
    assert document["execution_plan"] == {
        "path": str(execution), "sha256": _sha(execution),
        "sealed_plan_path": str(plan_path), "sealed_plan_sha256": _sha(plan_path)}
    rows = document["rows"]
    assert [row["quantum_id"] for row in rows] == ["layer-002"]
    payload = rows[0]["argv"]
    assert payload[payload.index("--plan") + 1] == str(execution)
    assert payload[payload.index("--plan-sha256") + 1] == _sha(execution)
    # The sealed plan is never rewritten, and the record still names it.
    assert plan_path.read_bytes() == sealed_bytes
    record = json.loads(next((tmp_path / "records").glob("layer-*.json")).read_text())
    assert record["campaign"]["plan_path"] == str(plan_path)


@pytest.mark.parametrize("resource_bound", [True, False],
                         ids=["resource-plan", "plain-plan"])
def test_certified_mode_refuses_an_execution_plan_that_differs(
        tmp_path, dispatch, monkeypatch, capsys, resource_bound):
    plan_path, _prepared, argv = _campaign(tmp_path, resource_bound=resource_bound)
    execution = _redeclare(plan_path, tmp_path / "execution-plan.json",
                           resource_bound=resource_bound)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    code, out, err = _main(dispatch, [*argv, "--execution-plan", str(execution)],
                           capsys)
    assert code == dispatch.EXIT_PRECONDITION_REFUSED
    assert "[DEV-MODE]" not in out
    assert "execution plan" in err and "differs from the records' sealed plan" in err


def test_certified_mode_runs_a_byte_identical_execution_plan(
        tmp_path, dispatch, monkeypatch, capsys):
    plan_path, _prepared, argv = _campaign(tmp_path, resource_bound=True)
    execution = tmp_path / "execution-plan.json"
    execution.write_bytes(plan_path.read_bytes())
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    code, out, err = _main(dispatch, [*argv, "--execution-plan", str(execution)],
                           capsys)
    assert code == 0, err
    payload = _rows(out)["rows"][0]["argv"]
    assert payload[payload.index("--plan") + 1] == str(execution)
    assert payload[payload.index("--plan-sha256") + 1] == _sha(plan_path)


def test_an_unreadable_execution_plan_refuses(tmp_path, dispatch, capsys):
    _plan_path, _prepared, argv = _campaign(tmp_path, resource_bound=True)
    code, _out, err = _main(
        dispatch, [*argv, "--execution-plan", str(tmp_path / "missing.json")], capsys)
    assert code == dispatch.EXIT_PRECONDITION_REFUSED
    assert "--execution-plan" in err


def test_without_the_flag_the_dry_run_names_no_execution_plan(
        tmp_path, dispatch, capsys):
    _plan_path, _prepared, argv = _campaign(tmp_path, resource_bound=True)
    code, out, err = _main(dispatch, argv, capsys)
    assert code == 0, err
    assert _rows(out)["execution_plan"] is None
