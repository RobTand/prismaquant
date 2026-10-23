"""Issue #2686: reserve the real price-table workload for the fill proof.

Run the existing boundary assertions with the real allocator, but reject a
full-table input before entering its per-pick rescan.  A structural workload
bound is deterministic across machines; a wall-clock deadline would reproduce
the shard flake rather than guard against its cause.
"""
from __future__ import annotations

import pytest

import test_tessera_per_unit_rates as rates


@pytest.mark.parametrize("test_name, minimum_calls", [
    ("test_allocator_refuses_rows_without_exact_price_or_wire_bytes", 8),
    ("test_allocator_is_deterministic_across_row_insertion_order", 2),
    ("test_allocator_honors_an_explicit_reserve_off_the_hard_cap", 4),
    ("test_allocator_stops_at_the_infeasible_floor_without_inventing_a_pick", 3),
    ("test_allocator_refuses_non_integer_budget_or_reserve", 5),
    ("test_allocator_record_does_not_claim_whole_artifact_compliance", 1),
])
def test_non_fill_allocator_checks_use_small_priced_worlds(
        monkeypatch, test_name, minimum_calls):
    real_select = rates.tep.select_priced_unit_upgrades
    calls = []

    def bounded_select(costs, assignment, **kwargs):
        # Sixteen is a test-fixture ceiling, not a production allocator limit.
        # It leaves room for independent priced, cross-grid, BF16 and damaged
        # units without repeating the 11,232-unit / 718-pick derivation.
        sizes = (len(costs), len(assignment))
        assert max(sizes) <= 16, (
            f"{test_name} submitted {sizes[0]} cost units and "
            f"{sizes[1]} assignment units; use a small priced fixture here "
            "and keep the real derivation in the dedicated fill test only"
        )
        calls.append(sizes)
        return real_select(costs, assignment, **kwargs)

    monkeypatch.setattr(rates.tep, "select_priced_unit_upgrades", bounded_select)
    if test_name == "test_allocator_refuses_rows_without_exact_price_or_wire_bytes":
        # This test is already small on the current main.  Keep executing its
        # positive pick plus cross-grid, stack-only and BF16 refusal assertions
        # alongside the missing-field cases, rather than mocking any results.
        rates.test_allocator_keeps_stack_only_units_grouped_and_refuses_cross_grid_rows()
        assert calls, "the grouped-unit check must exercise the allocator"
        calls.clear()

    getattr(rates, test_name)()
    assert len(calls) >= minimum_calls, "retain the existing boundary-case calls"
