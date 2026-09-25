"""PQ #1134: the prefetch note is bounded by the sealed plan's source window.

The Stage B layer-44 gate (``6f4f058751e6``) measured 107.0 GB of idle
head-time cache budget, 13.8 GB per layer, and printed advice to widen a
sealed ``max_cache_slots=2`` to 7. Its plan reserves a 33.3 GB source window
inside a 94 GiB cap; seven slots do not fit that rule. Stage A (#737) has no
replay budgeted after the head phase and keeps the measured-budget note.
"""
from __future__ import annotations

import pytest

from prismaquant.joint_retained_window_plan import (
    EXECUTION_SCHEMA, RetainedWindowBudget, planned_source_window_bytes)
from prismaquant.streaming_model import _prefetch_widening_note

GiB = 1024 ** 3
#: The layer-44 banner: ``layer cache budget=107.0 GB``, ``est_layer=13.8 GB``,
#: ``cache_slots=7, max_cache_slots=2, memory_slots=7``.
CACHE_BYTES = int(107.0 * GiB)
LAYER_BYTES = int(13.8 * GiB)
#: ``execution.retained_operator_windows.source_reserve_bytes`` of the
#: sealed plan ``a4/overlay/plan.json`` (``c9b879b4``).
SOURCE_RESERVE = 33_285_996_544
#: The retained budget of the Stage B resource policy (r13): a 94 GiB cap.
BUDGET = {
    "schema": "prismaquant.joint_retained_window_budget.v1",
    "auxiliary_reserve_bytes": 2147483648, "boundary_reserve_bytes": 2281701376,
    "candidate_delta_bytes": 201326592, "load_buffer_bytes": 402662764,
    "max_windows_per_layer": 25, "metadata_reserve_bytes": 21474836480,
    "physical_limit_bytes": 94 * GiB, "read_page_reserve_bytes": 4096,
    "retained_render_cap_bytes": 6023929799, "runtime_reserve_bytes": 4294967296,
    "safety_margin_bytes": 2147483648, "statistics_cap_bytes": 5939134464,
    "workspace_reserve_bytes": 17179869184,
}
RETAINED = {"schema": EXECUTION_SCHEMA, "budget": BUDGET,
            "source_reserve_bytes": SOURCE_RESERVE,
            "source_loading_reserve_bytes": 1 * GiB}


def _layer44_slots():
    cache_slots = CACHE_BYTES // LAYER_BYTES
    assert cache_slots == 7
    return cache_slots


def test_layer44_plan_window_gives_no_widening_advice():
    window = planned_source_window_bytes(RETAINED)
    assert window == SOURCE_RESERVE
    assert window // LAYER_BYTES == 2
    assert _prefetch_widening_note(
        max_cache_slots=2, cache_slots=_layer44_slots(), memory_slots=7,
        estimated_layer_bytes=LAYER_BYTES,
        planned_source_window_bytes=window) is None


def test_stage_a_without_a_planned_window_keeps_the_737_note():
    assert planned_source_window_bytes(None) is None
    note = _prefetch_widening_note(
        max_cache_slots=2, cache_slots=_layer44_slots(), memory_slots=7,
        estimated_layer_bytes=LAYER_BYTES, planned_source_window_bytes=None)
    assert note is not None
    assert "sealed max_cache_slots=2" in note
    assert "the measured budget" in note
    assert "admitted_slots=7" in note


def test_a_wider_plan_window_recommends_only_what_the_plan_admits():
    note = _prefetch_widening_note(
        max_cache_slots=2, cache_slots=_layer44_slots(), memory_slots=7,
        estimated_layer_bytes=LAYER_BYTES,
        planned_source_window_bytes=4 * LAYER_BYTES)
    assert note is not None
    assert "admitted_slots=4" in note
    assert "the sealed plan's source window" in note


def test_the_cap_bounds_a_reserve_its_other_owners_leave_no_room_for():
    budget = RetainedWindowBudget.from_dict(BUDGET)
    others = budget.fixed_bytes(SOURCE_RESERVE) - SOURCE_RESERVE
    cap_share = BUDGET["physical_limit_bytes"] - BUDGET["safety_margin_bytes"] - others
    wide = dict(RETAINED, source_reserve_bytes=cap_share + 7 * LAYER_BYTES)
    assert planned_source_window_bytes(wide) == cap_share


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_refuses_a_malformed_plan_window(bad):
    with pytest.raises(ValueError, match="planned_source_window_bytes"):
        _prefetch_widening_note(
            max_cache_slots=2, cache_slots=7, memory_slots=7,
            estimated_layer_bytes=LAYER_BYTES, planned_source_window_bytes=bad)


def test_stage_b_builds_thread_the_plan_window_and_stage_a_plans_do_not():
    from prismaquant.tessera_joint_aura import _planned_source_window

    stage_b = {"execution": {"retained_operator_windows": RETAINED}}
    assert _planned_source_window(stage_b) == {
        "planned_source_window_bytes": SOURCE_RESERVE}
    assert _planned_source_window({"execution": {}}) == {}
