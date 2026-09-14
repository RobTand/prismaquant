"""``coverage_first_v1``: what the initial screen keeps, defers and refuses."""

from __future__ import annotations

import pathlib

import pytest

from prismaquant.quality_prefill_contract import validate_screen_decision
from prismaquant.quality_prefill_screen import (
    REPRESENTATIVE_TIE_RULE,
    SCREEN_RULE,
    ScreenCandidate,
    ScreenCaps,
    ScreenError,
    apply_coverage_first_v1,
    expand_licensed_composites,
    refuse_dominance_screen,
)

SOURCE = "a" * 64


def candidate(
    cid,
    *,
    stratum="rate.e4m3.0256-0512",
    route="a4w4",
    mandatory=False,
    interior=False,
    composites=1,
    tasks=1,
    uncertainty=0.25,
    runtime_known=False,
):
    return ScreenCandidate(
        candidate_id=cid,
        rate_stratum_id=stratum,
        route_class=route,
        mandatory=mandatory,
        interior_draw=interior,
        composite_options=composites,
        tasks_per_phase=tasks,
        uncertainty=uncertainty,
        runtime_price_known=runtime_known,
    )


def roster():
    return [
        candidate("c.boundary.0256", mandatory=True),
        candidate("c.boundary.0512", mandatory=True),
        candidate("c.interior.0300", interior=True),
        candidate("c.spare.0310"),
        candidate("c.spare.0320"),
        candidate("c.spare.0330"),
        candidate("c.a16.0340", route="a16w4"),
        candidate("c.a16.0350", route="a16w4"),
    ]


def apply(entries=None, **overrides):
    kwargs = {
        "caps": ScreenCaps(
            max_retained_candidates=16,
            max_composite_options=32,
            max_tasks_per_phase=64,
        ),
        "source_sha256": SOURCE,
        "audit_count": 2,
        "decision_id": "screen.pilot.01",
    }
    kwargs.update(overrides)
    return apply_coverage_first_v1(roster() if entries is None else entries, **kwargs)


def test_the_module_imports_no_rng():
    source = pathlib.Path("prismaquant/quality_prefill_screen.py").read_text()
    assert "import random" not in source
    assert "from random" not in source
    assert "secrets" not in source


def test_coverage_first_v1_never_prunes_anything():
    """The defining negative property of the initial policy."""
    result = apply()
    dispositions = {
        entry["candidate_id"]: entry["disposition"]
        for entry in result.decision["dispositions"]
    }
    assert set(dispositions.values()) <= {"retained", "deferred"}
    assert "pruned" not in set(dispositions.values())


def test_every_mandatory_candidate_is_retained():
    result = apply()
    assert "c.boundary.0256" in result.retained
    assert "c.boundary.0512" in result.retained


def test_the_frozen_interior_draw_is_retained():
    assert "c.interior.0300" in apply().retained


def test_each_route_class_keeps_a_representative():
    result = apply()
    kept_classes = {
        candidate("x", route=r).route_class
        for r in ("a4w4", "a16w4")
    }
    retained = set(result.retained)
    assert retained & {"c.a16.0340", "c.a16.0350"}, "the a16w4 class lost its route"
    assert kept_classes == {"a4w4", "a16w4"}


def test_every_candidate_gets_exactly_one_disposition():
    result = apply()
    ids = [entry["candidate_id"] for entry in result.decision["dispositions"]]
    assert sorted(ids) == sorted(c.candidate_id for c in roster())
    assert len(ids) == len(set(ids))


def test_the_decision_validates_against_the_milestone_1_schema():
    validate_screen_decision(dict(apply().decision))


def test_the_rule_is_named_and_versioned_in_the_decision():
    assert apply().decision["rule"] == dict(SCREEN_RULE)
    assert SCREEN_RULE["name"] == "coverage_first_v1"


def test_the_decision_is_reproducible_from_the_same_inputs():
    assert apply().identity_sha256() == apply().identity_sha256()


def test_input_order_does_not_change_the_ledger():
    forward = apply()
    backward = apply(list(reversed(roster())))
    assert forward.identity_sha256() == backward.identity_sha256()
    assert forward.audit == backward.audit


def test_a_missing_uncertainty_refuses_rather_than_writing_zero():
    entries = roster()
    entries[3] = candidate("c.spare.0310", uncertainty=None)
    with pytest.raises(ScreenError, match="Unknown uncertainty is not zero"):
        apply(entries)


def test_a_cap_the_mandatory_set_exceeds_refuses_with_both_counts():
    with pytest.raises(ScreenError) as excinfo:
        apply(
            caps=ScreenCaps(
                max_retained_candidates=2,
                max_composite_options=32,
                max_tasks_per_phase=64,
            )
        )
    message = str(excinfo.value)
    assert "retained_candidates requires 4 but the manifest allows 2" in message
    assert "never reduces a quota" in message


def test_a_cap_overflow_never_returns_a_truncated_prefix():
    caps = ScreenCaps(
        max_retained_candidates=3, max_composite_options=32, max_tasks_per_phase=64
    )
    with pytest.raises(ScreenError):
        apply(caps=caps)


def test_composite_options_are_capped_after_expansion_not_before():
    entries = [
        candidate("c.boundary.0256", mandatory=True, composites=9),
        candidate("c.boundary.0512", mandatory=True, composites=9),
        candidate("c.spare.0310"),
    ]
    assert expand_licensed_composites(entries) == 19
    with pytest.raises(ScreenError, match="composite_options requires 18"):
        apply(
            entries,
            caps=ScreenCaps(
                max_retained_candidates=16,
                max_composite_options=8,
                max_tasks_per_phase=64,
            ),
        )


def test_a_tasks_per_phase_overflow_refuses():
    entries = [
        candidate("c.boundary.0256", mandatory=True, tasks=50),
        candidate("c.spare.0310"),
    ]
    with pytest.raises(ScreenError, match="tasks_per_phase requires 50"):
        apply(
            entries,
            caps=ScreenCaps(
                max_retained_candidates=16,
                max_composite_options=32,
                max_tasks_per_phase=10,
            ),
        )


def test_a_dominance_screen_refuses_while_a_runtime_price_is_unknown():
    with pytest.raises(ScreenError, match="An unmeasured route is not a slow route"):
        refuse_dominance_screen(roster())


def test_a_dominance_screen_is_allowed_once_every_runtime_price_exists():
    priced = [
        candidate(c.candidate_id, route=c.route_class, runtime_known=True)
        for c in roster()
    ]
    refuse_dominance_screen(priced)


def test_the_audit_roster_is_drawn_from_deferred_candidates_only():
    result = apply()
    assert len(result.audit) == 2
    assert set(result.audit) <= set(result.deferred)


def test_an_audit_larger_than_the_deferred_pool_refuses():
    with pytest.raises(ScreenError, match="cannot draw more than"):
        apply(audit_count=99)


def test_the_audit_draw_is_stable_across_seeds_only_by_declaration():
    zero = apply(selection_seed=0).audit
    one = apply(selection_seed=1).audit
    assert zero == apply(selection_seed=0).audit
    assert one == apply(selection_seed=1).audit


def test_a_duplicate_candidate_id_refuses():
    entries = roster() + [candidate("c.spare.0310")]
    with pytest.raises(ScreenError, match="duplicate candidate id"):
        apply(entries)


def test_an_empty_roster_refuses():
    with pytest.raises(ScreenError, match="at least one candidate"):
        apply([])


def test_a_zero_cap_refuses_at_construction():
    with pytest.raises(ScreenError, match="must be a positive int"):
        ScreenCaps(
            max_retained_candidates=0, max_composite_options=1, max_tasks_per_phase=1
        )


def test_retained_candidates_are_pending_measurement_never_measured():
    statuses = {
        entry["candidate_id"]: entry["measurement_status"]
        for entry in apply().decision["dispositions"]
    }
    assert statuses["c.boundary.0256"] == "retained_pending_measurement"
    assert "measured" not in set(statuses.values())


def test_a_deferred_candidate_reads_as_not_yet_acquired():
    result = apply()
    statuses = {
        entry["candidate_id"]: entry["measurement_status"]
        for entry in result.decision["dispositions"]
    }
    for candidate_id in result.deferred:
        assert statuses[candidate_id] == "not_yet_acquired"


def test_every_deferred_candidate_records_a_reason():
    for entry in apply().decision["dispositions"]:
        assert entry["reason"]
        if entry["disposition"] == "deferred":
            assert "not a deletion" in entry["reason"]


def test_the_route_class_representative_is_the_lowest_candidate_id():
    """Deterministic by a stated rule, not by borrowing another purpose's draw."""
    entries = [
        candidate("c.boundary.0256", mandatory=True),
        candidate("c.a16.0350", route="a16w4"),
        candidate("c.a16.0340", route="a16w4"),
        candidate("c.a16.0360", route="a16w4"),
    ]
    result = apply(entries)
    assert "c.a16.0340" in result.retained
    assert {"c.a16.0350", "c.a16.0360"} <= set(result.deferred)


def test_the_representative_reason_publishes_the_tie_rule():
    reasons = {
        entry["candidate_id"]: entry["reason"]
        for entry in apply().decision["dispositions"]
    }
    representative = next(
        cid for cid in ("c.a16.0340", "c.a16.0350") if "representative" in reasons[cid]
    )
    assert REPRESENTATIVE_TIE_RULE in reasons[representative]


def test_a_retained_candidate_holds_no_discarded_candidate_audit_rank():
    """Section 5.1 keeps purpose domains apart: retention is not an audit draw."""
    result = apply()
    assert set(result.ranks) == set(result.deferred)
    for candidate_id in result.retained:
        assert candidate_id not in result.ranks
