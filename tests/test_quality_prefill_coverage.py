"""Section 5.2: when four memberships are one set, and what happens when they are not."""

from __future__ import annotations

import pytest

from prismaquant.quality_prefill_coverage import (
    LABEL_RESTRICTED_PILOT,
    LABEL_WHOLE_MODEL,
    CoverageError,
    RoutedStack,
    frontier_label,
    require_census_equality,
    routed_stack_gaps,
)

UNITS = ("layers.0.attn.qkv", "layers.0.mlp.down", "layers.1.mlp.gate_up")
COMPLETE = frozenset({"wire", "scale", "source_pwc", "joint_row"})


def memberships(**overrides):
    base = {name: list(UNITS) for name in (
        "census_units",
        "quality_table_units",
        "runtime_binding_units",
        "assignment_units",
    )}
    base.update(overrides)
    return base


def routed(members=("e0", "e1", "e2"), covered=None, diagnostic=()):
    covered = set(members) if covered is None else set(covered)
    return RoutedStack(
        unit_id="layers.2.moe",
        required_members=tuple(members),
        member_evidence={m: COMPLETE for m in covered},
        diagnostic_members=tuple(diagnostic),
    )


def test_four_equal_memberships_return_the_common_set():
    assert require_census_equality(memberships()) == frozenset(UNITS)


def test_a_unit_missing_from_the_runtime_binding_is_named_by_id():
    partial = memberships(runtime_binding_units=list(UNITS[:2]))
    with pytest.raises(CoverageError) as excinfo:
        require_census_equality(partial)
    message = str(excinfo.value)
    assert "runtime_binding_units" in message
    assert "layers.1.mlp.gate_up" in message
    assert "absent from" in message


def test_a_unit_the_census_does_not_contain_is_reported_the_other_way():
    surplus = memberships(assignment_units=list(UNITS) + ["layers.9.ghost"])
    with pytest.raises(CoverageError) as excinfo:
        require_census_equality(surplus)
    message = str(excinfo.value)
    assert "layers.9.ghost" in message
    assert "not mutable census units" in message


def test_both_directions_are_reported_in_one_message():
    both = memberships(quality_table_units=[UNITS[0], "layers.9.ghost"])
    with pytest.raises(CoverageError) as excinfo:
        require_census_equality(both)
    message = str(excinfo.value)
    assert "absent from quality_table_units" in message
    assert "not mutable census units" in message


def test_a_missing_membership_name_refuses():
    incomplete = memberships()
    del incomplete["assignment_units"]
    with pytest.raises(CoverageError, match="missing assignment_units"):
        require_census_equality(incomplete)


def test_an_unknown_membership_name_refuses():
    extra = memberships()
    extra["vibes_units"] = list(UNITS)
    with pytest.raises(CoverageError, match="unknown vibes_units"):
        require_census_equality(extra)


def test_a_repeated_unit_id_refuses():
    doubled = memberships(census_units=list(UNITS) + [UNITS[0]])
    with pytest.raises(CoverageError, match="repeats unit id"):
        require_census_equality(doubled)


def test_an_empty_census_refuses():
    with pytest.raises(CoverageError, match="census is empty"):
        require_census_equality(memberships(**{name: [] for name in (
            "census_units",
            "quality_table_units",
            "runtime_binding_units",
            "assignment_units",
        )}))


def test_a_complete_routed_stack_reports_no_gap():
    assert routed_stack_gaps([routed()]) == {}


def test_a_member_missing_one_evidence_kind_is_a_gap():
    stack = RoutedStack(
        unit_id="layers.2.moe",
        required_members=("e0", "e1"),
        member_evidence={"e0": COMPLETE, "e1": frozenset({"wire", "scale", "joint_row"})},
    )
    assert routed_stack_gaps([stack]) == {"layers.2.moe": {"e1": ["source_pwc"]}}


def test_a_high_low_diagnostic_draw_does_not_satisfy_the_gate():
    """Two experts out of many is section 5.1's diagnostic, not coverage."""
    stack = routed(
        members=tuple(f"e{i}" for i in range(8)),
        covered=("e0", "e7"),
        diagnostic=("e0", "e7"),
    )
    gaps = routed_stack_gaps([stack])["layers.2.moe"]
    assert set(gaps) == {f"e{i}" for i in range(1, 7)}


def test_an_unknown_evidence_kind_refuses():
    stack = RoutedStack(
        unit_id="layers.2.moe",
        required_members=("e0",),
        member_evidence={"e0": frozenset({"wire", "vibes"})},
    )
    with pytest.raises(CoverageError, match="unknown evidence kind"):
        routed_stack_gaps([stack])


def test_evidence_for_a_member_outside_the_roster_refuses():
    with pytest.raises(CoverageError, match="does not require"):
        RoutedStack(
            unit_id="layers.2.moe",
            required_members=("e0",),
            member_evidence={"e0": COMPLETE, "e9": COMPLETE},
        )


def test_a_diagnostic_member_outside_the_roster_refuses():
    with pytest.raises(CoverageError, match="outside the roster"):
        RoutedStack(
            unit_id="layers.2.moe",
            required_members=("e0",),
            member_evidence={"e0": COMPLETE},
            diagnostic_members=("e9",),
        )


def test_complete_coverage_earns_the_whole_model_label():
    verdict = frontier_label(memberships(), routed_stacks=[routed()])
    assert verdict.label == LABEL_WHOLE_MODEL
    assert verdict.covered_units == tuple(sorted(UNITS))


def test_an_unequal_membership_downgrades_to_a_restricted_pilot():
    verdict = frontier_label(memberships(runtime_binding_units=list(UNITS[:2])))
    assert verdict.label == LABEL_RESTRICTED_PILOT
    assert any("absent from runtime_binding_units" in r for r in verdict.reasons)


def test_an_incomplete_routed_stack_downgrades_even_with_equal_memberships():
    stack = routed(members=("e0", "e1"), covered=("e0",))
    verdict = frontier_label(memberships(), routed_stacks=[stack])
    assert verdict.label == LABEL_RESTRICTED_PILOT
    assert "layers.2.moe" in verdict.routed_gaps


def test_a_downgrade_is_a_verdict_not_an_exception():
    """An honest restricted pilot is a valid outcome, not an error."""
    verdict = frontier_label(memberships(assignment_units=[UNITS[0]]))
    assert verdict.label == LABEL_RESTRICTED_PILOT
    assert verdict.reasons


def test_the_verdict_is_reproducible():
    first = frontier_label(memberships(), routed_stacks=[routed()])
    second = frontier_label(memberships(), routed_stacks=[routed()])
    assert first.identity_sha256() == second.identity_sha256()


def test_membership_order_does_not_change_the_verdict():
    shuffled = memberships(census_units=list(reversed(UNITS)))
    assert (
        frontier_label(shuffled, routed_stacks=[routed()]).identity_sha256()
        == frontier_label(memberships(), routed_stacks=[routed()]).identity_sha256()
    )
