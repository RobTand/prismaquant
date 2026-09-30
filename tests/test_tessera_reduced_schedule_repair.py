"""Replayed repair history describes scalar work, never a verified wire."""
from copy import deepcopy
from typing import Any, NotRequired, TypedDict

import pytest

from prismaquant.tessera_formats import TesseraFormatError


class _RepairSnapshot(TypedDict):
    schema: str
    records: dict[str, dict[str, Any]]
    unit_bytes: dict[str, dict[str, int]]
    byte_budget: int
    winners: dict[str, int]
    measured_cells: list[list[str | int]]
    draws: int
    seed: int
    budget_sensitivity: NotRequired[list[float]]


def _initial() -> _RepairSnapshot:
    records: dict[str, dict[str, Any]] = {}
    cells: list[list[str | int]] = []
    for name in ("a", "b"):
        reference = {str(e): {"gate": float(2 ** e)} for e in range(5)}
        records[name] = dict(stack=name, projections=["gate"], experts=list(range(5)),
            reference_q256=960, reference_mse=reference, sampled_experts=[0, 1, 2],
            sampled_mse={str(q): {str(e): {"gate": reference[str(e)]["gate"] * factor}
                                 for e in (0, 1, 2)}
                         for q, factor in ((832, 2), (1088, .5))},
            weights=None, currency="uniform_output_mse")
        cells += [[name, e, 960] for e in range(5)]
        cells += [[name, e, q] for e in (0, 1, 2) for q in (832, 1088)]
    return _RepairSnapshot(schema="prismaquant.stack_reduced_schedule_input.v1", records=records,
        unit_bytes={n: {"832": 10, "960": 20, "1088": 30} for n in records},
        byte_budget=40, winners={"a": 832, "b": 832}, measured_cells=cells,
        draws=50, seed=495)


def _complete(snapshot: _RepairSnapshot, pending: list[dict[str, Any]]) -> _RepairSnapshot:
    new = deepcopy(snapshot)
    for c in pending:
        n, e, q = c["stack"], c["expert"], c["rate_q256"]
        factor = {832: 2, 1088: .5}[q]
        new["records"][n]["sampled_mse"][str(q)][str(e)] = {
            "gate": new["records"][n]["reference_mse"][str(e)]["gate"] * factor}
        new["measured_cells"].append([n, e, q])
    return new


def _replay(history, **kwargs) -> dict[str, Any]:
    from prismaquant.tessera_reduced_schedule_repair import plan_reduced_schedule_repair
    return plan_reduced_schedule_repair(history, **kwargs)


def _second() -> tuple[_RepairSnapshot, dict[str, Any], _RepairSnapshot]:
    first = _initial()
    root = _replay([first], max_rounds=2)
    second = _complete(first, root["current_plan"]["pending_cells"])
    second["winners"]["a"] = 1088
    return first, root, second


def test_real_plan_composition_then_changed_winner_requests_only_new_cells():
    first, root, second = _second()
    out = _replay([first, second], max_rounds=2,
                  expected_previous_sha256=root["binding_sha256"])
    assert out["rounds_used"] == 1
    assert out["steps"][-1]["winners_changed"] is True
    assert out["steps"][-1]["added_measured_cell_count"] == 4
    assert out["current_plan"]["pending_cells"] == [
        {"stack": "a", "expert": e, "rate_q256": 1088} for e in (3, 4)]
    assert out["state"] == "awaiting_measurements"
    assert out["wire_ready"] is False
    assert out["production_solver_run"] is False


def test_completed_history_is_idempotent_and_only_scalar_coverage():
    first, root, second = _second()
    previous = _replay([first, second], max_rounds=2,
                       expected_previous_sha256=root["binding_sha256"])
    third = _complete(second, previous["current_plan"]["pending_cells"])
    args = dict(max_rounds=2, expected_previous_sha256=previous["binding_sha256"])
    out = _replay([first, second, third], **args)
    assert out == _replay([first, second, third], **args)
    assert out["rounds_used"] == 2
    assert out["state"] == "scalar_coverage_only"
    assert out["current_plan"]["pending_cells"] == []
    assert out["steps"][-1]["winners_changed"] is False
    assert not out["wire_ready"] and not out["production_solver_run"]


@pytest.mark.parametrize("problem", ["wrong_parent", "missing_parent", "changed_cap",
    "incomplete_requests", "lost_cell", "no_scalar_row", "rewritten_row",
    "currency", "budget", "gate_seed", "cap_exhaustion"])
def test_progress_refuses_unbound_incomplete_or_changed_experiments(problem):
    first, root, second = _second()
    cap, parent = 2, root["binding_sha256"]
    if problem == "wrong_parent":
        parent = "f" * 64
    elif problem == "missing_parent":
        parent = None
    elif problem == "changed_cap":
        cap = 3
    elif problem == "incomplete_requests":
        second = deepcopy(first)
    elif problem == "lost_cell":
        second["measured_cells"].remove(["a", 0, 1088])
    elif problem == "no_scalar_row":
        del second["records"]["a"]["sampled_mse"]["832"]["3"]
    elif problem == "rewritten_row":
        second["records"]["a"]["sampled_mse"]["832"]["0"]["gate"] *= 1.1
    elif problem == "currency":
        for s in second["records"].values():
            s["currency"] = "different_currency"
    elif problem == "budget":
        second["byte_budget"] += 1
    elif problem == "gate_seed":
        second["seed"] += 1
    elif problem == "cap_exhaustion":
        cap = 1
        parent = _replay([first], max_rounds=1)["binding_sha256"]
    with pytest.raises(TesseraFormatError):
        _replay([first, second], max_rounds=cap, expected_previous_sha256=parent)


@pytest.mark.parametrize("cap", [True, 0, -1])
def test_round_cap_is_an_explicit_positive_integer(cap):
    with pytest.raises(TesseraFormatError):
        _replay([_initial()], max_rounds=cap)


@pytest.mark.parametrize("retyping", ["scalar_row", "fixed_config"])
def test_numeric_retyping_cannot_change_canonical_history_identity(retyping):
    first = _initial()
    first["budget_sensitivity"] = [.5, .75, 1.0, 1.25]
    root = _replay([first], max_rounds=2)
    second = _complete(first, root["current_plan"]["pending_cells"])
    if retyping == "scalar_row":
        second["records"]["a"]["reference_mse"]["1"]["gate"] = 2
    else:
        second["budget_sensitivity"] = [.5, .75, 1, 1.25]
    with pytest.raises(TesseraFormatError):
        _replay([first, second], max_rounds=2,
                expected_previous_sha256=root["binding_sha256"])
