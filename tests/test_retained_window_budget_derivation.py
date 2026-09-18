"""The retained COST budget's demand-driven caps, derived from a real roster.

Issue #743. The GLM-5.3-Flash joint COST run refused
``model.language_model.layers.44.mlp.experts.0.down_proj`` after 137.4 minutes
of boundary capture, against a sealed ``candidate_delta_bytes`` of 4 MiB that
no target in the roster could satisfy. These cases carry that model's measured
geometry, so a budget that cannot admit it fails here in milliseconds.
"""
import pytest

from prismaquant.joint_retained_window_plan import (
    DECLARED_BUDGET_FIELDS, DERIVATION_SCHEMA, HOST_RESIDENT_BUDGET_FIELDS,
    RetainedTarget, RetainedWindowBudget,
    derive_retained_window_budget, plan_retained_targets,
)
from test_joint_replay_aggregate_guard import CAMPAIGN_BUDGET

#: ``execution.retained_operator_windows.source_reserve_bytes`` of the same plan.
SOURCE_RESERVE_BYTES = 33285996544
#: ``execution.operator_windows.prefetch_workers`` of the same plan.
PREFETCH_WORKERS = 4
#: ``physical_limit_bytes`` less the plan's ``max_gpu_bytes`` (80 GiB): the
#: container's own cgroup cap, which the aggregate capture guard still holds
#: on its own and which therefore bounds the host-resident retained renders.
HOST_CAP_BYTES = 111669149696 - 85899345920
SCOPE = 'pwc_serialized_upper_bound'

#: Measured over the campaign roster of plan ``0b2cc0066bb612e3…``: 36,423
#: targets, joined from ``extension-r1024-02/workspace/plan.json`` members to
#: ``census.json`` ``unit_shapes`` and to the candidate ``.pt`` file sizes the
#: run's own data manifest records. ``statistics_bytes`` is
#: ``numel * 4 * (1 + act-quantizing groups)``; the shipped menus put every
#: ``TESSERA_E4M3_K1_R*`` rung in one activation group beside one weight-only
#: ``TESSERA_BF16_K1_R*`` group, so the multiplier is two.
#:   name, statistics_bytes, render_bytes, largest_serialized_bytes,
#:   candidate_delta_bytes, candidate_count
DENSE_MLP = ('model.language_model.layers.0.mlp.down_proj',
             12288 * 4096 * 4 * 2, 905_991_184, 100_665_691, 4 * 12288 * 4096, 5)
ROUTED_EXPERT = ('model.language_model.layers.44.mlp.experts.0.down_proj',
                 4096 * 2048 * 4 * 2, 151_017_852, 16_779_766, 4 * 4096 * 2048, 9)


def _target(row, name=None):
    return RetainedTarget(name or row[0], *row[1:])


def _moe_layer(layer, count=867):
    """One sparse layer's roster: 288 routed experts x 3 projections, plus shared."""
    return [_target(ROUTED_EXPERT, f'model.language_model.layers.{layer}.unit.{index}')
            for index in range(count)]


def _roster():
    layers = {layer: [_target(DENSE_MLP, f'model.language_model.layers.{layer}.mlp.{role}')
                      for role in ('down_proj', 'gate_proj', 'up_proj')]
              for layer in range(3)}
    layers.update({layer: _moe_layer(layer) for layer in range(3, 45)})
    return layers


def _declared():
    return {name: CAMPAIGN_BUDGET[name] for name in DECLARED_BUDGET_FIELDS}


def test_the_sealed_budget_refuses_every_target_class_of_this_model():
    budget = RetainedWindowBudget.from_dict(CAMPAIGN_BUDGET)
    for row in (ROUTED_EXPERT, DENSE_MLP):
        with pytest.raises(RuntimeError, match='indivisible target does not fit'):
            plan_retained_targets([_target(row)], budget=budget,
                                  source_bytes=SOURCE_RESERVE_BYTES, footprint_scope=SCOPE)
    # The sealed cap is not merely tight: the smallest matrix in the roster is
    # eight times it, and the largest forty-eight times.
    assert budget.candidate_delta_bytes * 8 == _target(ROUTED_EXPERT).candidate_delta_bytes
    assert budget.candidate_delta_bytes * 48 == _target(DENSE_MLP).candidate_delta_bytes


def test_repairing_only_the_delta_reserve_moves_the_refusal_to_the_window_cap():
    """The second latent refusal, before another capture pays for finding it."""
    from dataclasses import replace
    budget = replace(RetainedWindowBudget.from_dict(CAMPAIGN_BUDGET),
                     candidate_delta_bytes=_target(DENSE_MLP).candidate_delta_bytes)
    with pytest.raises(RuntimeError, match='exceeding explicit replay-work cap'):
        plan_retained_targets(_moe_layer(44), budget=budget,
                              source_bytes=SOURCE_RESERVE_BYTES, footprint_scope=SCOPE)


def test_the_derivation_admits_every_target_class_of_this_model():
    budget, record = derive_retained_window_budget(
        _roster(), declared=_declared(), source_bytes=SOURCE_RESERVE_BYTES,
        prefetch_workers=PREFETCH_WORKERS, host_cap_bytes=HOST_CAP_BYTES, footprint_scope=SCOPE)
    assert record['schema'] == DERIVATION_SCHEMA
    # Exactly the demand: one fp32 delta over the largest matrix in the roster.
    assert budget.candidate_delta_bytes == 4 * 12288 * 4096 == 201_326_592
    # Ties over equal demand break on the name, deterministically.
    assert record['demand']['candidate_delta_bytes']['maximizing_target'] == (
        'model.language_model.layers.2.mlp.up_proj')
    assert budget.load_buffer_bytes == PREFETCH_WORKERS * 100_665_691
    assert (record['demand']['load_buffer_bytes']['largest_serialized_bytes']
            == 100_665_691)
    # Every declared owner is the operator's, unchanged.
    for name in DECLARED_BUDGET_FIELDS:
        assert getattr(budget, name) == CAMPAIGN_BUDGET[name]
    # And the roster it was derived from now packs, layer by layer.
    for layer, targets in _roster().items():
        plan = plan_retained_targets(targets, budget=budget,
                                     source_bytes=SOURCE_RESERVE_BYTES, footprint_scope=SCOPE)
        assert len(plan.windows) <= budget.max_windows_per_layer
        assert all(window.peak_planned_bytes <=
                   budget.physical_limit_bytes - budget.safety_margin_bytes
                   for window in plan.windows)
        assert record['windows_by_layer'][str(layer)] == len(plan.windows)


def test_the_derived_window_caps_are_the_packing_s_own_maxima():
    budget, record = derive_retained_window_budget(
        _roster(), declared=_declared(), source_bytes=SOURCE_RESERVE_BYTES,
        prefetch_workers=PREFETCH_WORKERS, host_cap_bytes=HOST_CAP_BYTES, footprint_scope=SCOPE)
    windows = [window for targets in _roster().values()
               for window in plan_retained_targets(
                   targets, budget=budget, source_bytes=SOURCE_RESERVE_BYTES,
                   footprint_scope=SCOPE).windows]
    assert budget.statistics_cap_bytes == max(window.statistics_bytes for window in windows)
    assert budget.retained_render_cap_bytes == max(window.render_bytes for window in windows)
    assert budget.retained_render_cap_bytes <= budget.available_window_bytes(SOURCE_RESERVE_BYTES)
    # Renders are host-resident, so the container's own cgroup cap bounds them
    # whatever the aggregate window allows.
    assert budget.retained_render_cap_bytes <= HOST_CAP_BYTES - sum(
        getattr(budget, name) for name in HOST_RESIDENT_BUDGET_FIELDS)
    assert record['available_window_bytes'] == budget.available_window_bytes(SOURCE_RESERVE_BYTES)


def test_a_roster_that_cannot_fit_the_physical_budget_is_refused_not_rounded():
    huge = RetainedTarget('model.language_model.layers.0.enormous',
                          1 << 46, 1 << 20, 1 << 20, 4096, 1)
    with pytest.raises(RuntimeError, match='indivisible target does not fit'):
        derive_retained_window_budget({0: [huge]}, declared=_declared(),
                                      source_bytes=SOURCE_RESERVE_BYTES,
                                      prefetch_workers=PREFETCH_WORKERS, host_cap_bytes=HOST_CAP_BYTES, footprint_scope=SCOPE)


def test_the_derivation_refuses_a_partial_owner_declaration():
    declared = _declared()
    declared.pop('workspace_reserve_bytes')
    with pytest.raises(ValueError, match='exactly the declared owners'):
        derive_retained_window_budget(_roster(), declared=declared,
                                      source_bytes=SOURCE_RESERVE_BYTES,
                                      prefetch_workers=PREFETCH_WORKERS, host_cap_bytes=HOST_CAP_BYTES, footprint_scope=SCOPE)


def test_the_host_cap_not_the_aggregate_window_bounds_retained_renders():
    """Renders load with ``map_location="cpu"``, so the cgroup cap binds them.

    The sealed plan's own 2 GiB render cap is exactly this arithmetic --
    24 GiB container cap less the 2 GiB safety margin less the 20 GiB metadata
    reserve -- which is why it was so much smaller than the aggregate window
    the same budget declares. What the sealed plan then got wrong was
    ``max_windows_per_layer``: a cap that small needs far more than two windows
    to carry a sparse layer's renders.
    """
    budget, record = derive_retained_window_budget(
        _roster(), declared=_declared(), source_bytes=SOURCE_RESERVE_BYTES,
        prefetch_workers=PREFETCH_WORKERS, host_cap_bytes=HOST_CAP_BYTES,
        footprint_scope=SCOPE)
    demand = record['demand']['retained_render_cap_bytes']
    assert demand['host_render_bound_bytes'] < demand['aggregate_window_bytes']
    assert budget.retained_render_cap_bytes <= demand['host_render_bound_bytes']
    # A larger container host cap buys back windows, and so buys back replay.
    roomier, roomier_record = derive_retained_window_budget(
        _roster(), declared=_declared(), source_bytes=SOURCE_RESERVE_BYTES,
        prefetch_workers=PREFETCH_WORKERS, host_cap_bytes=64 * 1024 ** 3,
        footprint_scope=SCOPE)
    assert roomier.max_windows_per_layer < budget.max_windows_per_layer
    assert (roomier_record['retained_window_replay_multiplier']
            < record['retained_window_replay_multiplier'])
