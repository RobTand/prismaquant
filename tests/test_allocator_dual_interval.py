import json
import math

import pytest

from prismaquant.allocator_solver import (
    Candidate,
    selected_rung_dual_intervals,
    solve_allocation,
)


def _menu(middle_loss: float = 4.0):
    stats = {"unit": {"n_params": 8}}
    candidates = {
        "unit": [
            Candidate("K_LOW", 4.0, 4, 10.0),
            Candidate("K_SELECTED", 8.0, 8, middle_loss),
            Candidate("K_HIGH", 12.0, 12, 2.0),
        ]
    }
    return stats, candidates


def test_selected_rung_dual_interval_reads_break_even_slopes():
    stats, candidates = _menu()
    solved = solve_allocation(
        stats, candidates, target_bits=8.0, bit_precision=1.0
    )
    assert solved is not None
    assignment, _ = solved
    assert assignment == {"unit": "K_SELECTED"}

    intervals = selected_rung_dual_intervals(
        stats, candidates, assignment, bit_precision=1.0
    )

    # Each DP bin is one byte here.  K_SELECTED ties K_HIGH at lambda=0.5
    # and K_LOW at lambda=1.5, so it is supported throughout that interval.
    assert intervals["unit"].lambda_lo == pytest.approx(0.5)
    assert intervals["unit"].lambda_hi == pytest.approx(1.5)
    assert not intervals["unit"].is_empty


def test_integer_dp_choice_can_have_empty_dual_interval_without_changing_it():
    stats = {"unit": {"n_params": 8}}
    candidates = {
        "unit": [
            Candidate("K_LOW", 4.0, 4, 5.0),
            Candidate("K_SELECTED", 8.0, 8, 4.0),
            Candidate("K_HIGH", 12.0, 12, 0.0),
        ]
    }
    solved_before = solve_allocation(
        stats, candidates, target_bits=8.0, bit_precision=1.0
    )
    assert solved_before is not None
    assignment_before, _ = solved_before
    allocation_bytes_before = json.dumps(
        assignment_before, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    interval = selected_rung_dual_intervals(
        stats, candidates, assignment_before, bit_precision=1.0
    )["unit"]

    assert interval.lambda_lo == pytest.approx(1.0)
    assert interval.lambda_hi == pytest.approx(0.25)
    assert interval.is_empty
    solved_after = solve_allocation(
        stats, candidates, target_bits=8.0, bit_precision=1.0
    )
    assert solved_after is not None
    allocation_bytes_after = json.dumps(
        solved_after[0], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert allocation_bytes_after == allocation_bytes_before


def test_dual_interval_rejects_ambiguous_or_missing_selected_rungs():
    stats, candidates = _menu()
    with pytest.raises(KeyError, match="missing candidate unit"):
        selected_rung_dual_intervals(stats, candidates, {}, bit_precision=1.0)

    candidates["unit"].append(
        Candidate("K_SELECTED", 9.0, 9, 3.0)
    )
    with pytest.raises(ValueError, match="resolves to 2 candidates"):
        selected_rung_dual_intervals(
            stats,
            candidates,
            {"unit": "K_SELECTED"},
            bit_precision=1.0,
        )


def test_dual_interval_handles_equal_charge_dominance_and_unbounded_high():
    stats = {"unit": {"n_params": 8}}
    unbounded = {
        "unit": [
            Candidate("BEST_LOW", 4.0, 4, 0.0),
            Candidate("WORSE_HIGH", 8.0, 8, 1.0),
        ]
    }
    interval = selected_rung_dual_intervals(
        stats, unbounded, {"unit": "BEST_LOW"}, bit_precision=1.0
    )["unit"]
    assert interval.lambda_lo == 0.0
    assert math.isinf(interval.lambda_hi) and interval.lambda_hi > 0.0
    assert not interval.is_empty

    same_charge = {
        "unit": [
            Candidate("DOMINATED", 4.0, 4, 2.0),
            Candidate("BETTER", 4.0, 4, 1.0),
        ]
    }
    interval = selected_rung_dual_intervals(
        stats, same_charge, {"unit": "DOMINATED"}, bit_precision=1.0
    )["unit"]
    assert interval.is_empty


def test_dual_interval_refuses_zero_bit_precision():
    stats, candidates = _menu()
    assignment = {"unit": "K_SELECTED"}
    with pytest.raises(ValueError) as exc_info:
        selected_rung_dual_intervals(
            stats, candidates, assignment, bit_precision=0.0
        )
    assert str(exc_info.value) == "bit_precision must be positive"
