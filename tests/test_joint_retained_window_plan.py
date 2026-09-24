"""Combined COST owner accounting, without allocating a model or matrices."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from prismaquant.joint_retained_window_plan import (
    RetainedTarget, RetainedWindowBudget, plan_retained_targets,
    targets_from_statistics_plan,
)


def budget(**changes):
    base = RetainedWindowBudget(2000, 20, 100, 100, 100, 20, 20, 100, 100, 100, 1000, 1000, 4)
    return replace(base, **changes)


def target(name, **changes):
    return replace(RetainedTarget(name, 300, 400, 100, 80, 4), **changes)


def test_individual_stats_and_render_caps_do_not_admit_combined_overflow():
    b = budget()
    assert 2 * 300 <= b.statistics_cap_bytes
    assert 2 * 400 <= b.retained_render_cap_bytes
    result = plan_retained_targets([target('b'), target('a')], budget=b,
        source_bytes=100, footprint_scope='pwc_selected_archive_storage')
    assert [w.names for w in result.windows] == [('a',), ('b',)]
    assert all(w.peak_planned_bytes <= 1980 for w in result.windows)
    assert result.as_dict()['archive_admission'] is True


def test_source_and_metadata_are_load_bearing_not_free_capacity():
    with pytest.raises(RuntimeError, match='indivisible'):
        plan_retained_targets([target('a')], budget=budget(metadata_reserve_bytes=1000),
            source_bytes=100, footprint_scope='pwc_selected_archive_storage')
    with pytest.raises(RuntimeError, match='indivisible'):
        plan_retained_targets([target('a')], budget=budget(), source_bytes=700,
            footprint_scope='pwc_selected_archive_storage')


def test_early_floor_check_refuses_unmeasured_extra_owners():
    b = budget()
    b.require_observed_baseline(observed_bytes=320, source_bytes=100, label='before_capture', actual_auxiliary_bytes=20)
    with pytest.raises(RuntimeError, match='committed cgroup-plus-CUDA baseline'):
        b.require_observed_baseline(observed_bytes=321, source_bytes=100, label='before_capture', actual_auxiliary_bytes=20)


@pytest.mark.parametrize('changes', [dict(largest_serialized_bytes=101), dict(candidate_delta_bytes=101),
                                    dict(statistics_bytes=1001), dict(render_bytes=1001)])
def test_indivisible_loader_delta_and_resident_caps_refuse(changes):
    with pytest.raises(RuntimeError, match='indivisible'):
        plan_retained_targets([target('a', **changes)], budget=budget(), source_bytes=0,
            footprint_scope='pwc_selected_archive_storage')


def test_no_unbounded_replay_fallback_when_fit_shrinks():
    with pytest.raises(RuntimeError, match='replay-work cap'):
        plan_retained_targets([target(str(i)) for i in range(5)], budget=budget(),
            source_bytes=100, footprint_scope='pwc_selected_archive_storage')


def test_logical_bodies_are_only_forecasts_not_archive_admission():
    plan = plan_retained_targets([target('a')], budget=budget(), source_bytes=100,
        footprint_scope='logical_bf16_geometry_forecast')
    assert plan.as_dict()['archive_admission'] is False


def test_auxiliary_fork_counted_once_and_read_pages_have_separate_charge():
    b = budget()
    assert b.fixed_bytes(100) == 740
    assert b.fixed_bytes(100) - replace(b, read_page_reserve_bytes=1).fixed_bytes(100) == 99
    encoded = b.as_dict()
    assert RetainedWindowBudget.from_dict(encoded) == b
    with pytest.raises(ValueError, match='complete versioned'):
        RetainedWindowBudget.from_dict({**encoded, 'max_replay_cotangent_bytes': 256})


def test_selected_archive_cost_join_rejects_aliases_and_unrelated_footprints():
    plan = SimpleNamespace(targets=[SimpleNamespace(name='a', statistics_bytes=256, shape=(4, 16))])
    rows = targets_from_statistics_plan(plan, {'a': (('a', 'fmt'),)},
        {('a', 'fmt'): {'incoming_storage_bytes': 128, 'serialized_bytes': 2048}})
    assert rows == (RetainedTarget('a', 256, 2048, 2048, 256, 1),)
    with pytest.raises(ValueError, match='unique'):
        targets_from_statistics_plan(plan, {'a': (('a', 'fmt'), ('a', 'fmt'))}, {})
    with pytest.raises(ValueError, match='empty PWC baseline'):
        targets_from_statistics_plan(plan, {'a': (('a', 'fmt'),)},
            {('a', 'fmt'): {'incoming_storage_bytes': 0, 'serialized_bytes': 0}})


def test_bool_and_nonfinite_budget_coordinates_refuse():
    with pytest.raises(ValueError, match='exact'):
        budget(runtime_reserve_bytes=True)
    with pytest.raises(ValueError, match='exact'):
        budget(metadata_reserve_bytes=float('nan'))
