"""Combined physical admission for one statistics plane and retained PWC renders.

This module owns scalar plans only. Source, boundary/auxiliary state, PWC and
operator statistics retain their existing owners. A geometry forecast using
logical tensor sizes is expressly not an archive-storage admission receipt.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping


SCHEMA = 'prismaquant.joint_retained_window_budget.v1'
EXECUTION_SCHEMA = 'prismaquant.joint_retained_execution.v1'
DERIVATION_SCHEMA = 'prismaquant.joint_retained_window_budget_derivation.v1'

#: The owners an operator declares: a physical bound and the reserves that are
#: properties of the box, the runtime and the capture, not of the roster.
DECLARED_BUDGET_FIELDS = ('physical_limit_bytes', 'safety_margin_bytes',
                          'metadata_reserve_bytes', 'runtime_reserve_bytes',
                          'workspace_reserve_bytes', 'boundary_reserve_bytes',
                          'auxiliary_reserve_bytes', 'read_page_reserve_bytes')
#: The caps that are a function of the roster the budget must admit. Every one
#: of these is a maximum over declared bytes, so none of them is a judgement
#: call and none of them belongs in a hand-written plan.
DERIVED_BUDGET_FIELDS = ('load_buffer_bytes', 'candidate_delta_bytes',
                         'statistics_cap_bytes', 'retained_render_cap_bytes',
                         'max_windows_per_layer')


def _integer(value, name, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f'{name} must be an exact {"positive" if positive else "nonnegative"} integer')
    return value


@dataclass(frozen=True)
class RetainedWindowBudget:
    physical_limit_bytes: int
    safety_margin_bytes: int
    metadata_reserve_bytes: int
    runtime_reserve_bytes: int
    workspace_reserve_bytes: int
    boundary_reserve_bytes: int
    auxiliary_reserve_bytes: int
    load_buffer_bytes: int
    read_page_reserve_bytes: int
    candidate_delta_bytes: int
    statistics_cap_bytes: int
    retained_render_cap_bytes: int
    max_windows_per_layer: int

    def __post_init__(self):
        for name, value in asdict(self).items():
            _integer(value, name, positive=name not in ('boundary_reserve_bytes', 'auxiliary_reserve_bytes'))
        if self.safety_margin_bytes >= self.physical_limit_bytes:
            raise ValueError('physical cap cannot hold its safety margin')

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get('schema') != SCHEMA or set(value) != {'schema', *cls.__dataclass_fields__}:
            raise ValueError('complete versioned retained-window budget required')
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})

    def as_dict(self):
        return {'schema': SCHEMA, **asdict(self)}

    def source_baseline_limit(self, source_bytes):
        _integer(source_bytes, 'source_bytes')
        return self.metadata_reserve_bytes + self.runtime_reserve_bytes + source_bytes + self.auxiliary_reserve_bytes

    def fixed_bytes(self, source_bytes):
        # Replay forks belong to ExactBoundaryStorage's auxiliary owner. Do
        # not add max_replay_cotangent_bytes again here.
        return (self.source_baseline_limit(source_bytes) + self.workspace_reserve_bytes
                + self.boundary_reserve_bytes + self.load_buffer_bytes
                + self.read_page_reserve_bytes + self.candidate_delta_bytes)

    def available_window_bytes(self, source_bytes):
        available = self.physical_limit_bytes - self.safety_margin_bytes - self.fixed_bytes(source_bytes)
        if available <= 0:
            raise RuntimeError('retained COST fixed owners exhaust the physical budget before statistics/PWC')
        return available

    def require_physical_guard(self, guard):
        """Refuse unless the live guard bounds at least what this plan states.

        ``physical_limit_bytes`` is the plan's whole physical bound -- on the
        joint replay path the cgroup cap plus the device envelope -- so it is
        compared with the guard's ``physical_cap_bytes``, which is that same
        sum for an aggregate guard and the cgroup cap for a plain one. A plan
        wider than the guard, or a plan holding back less margin than the
        guard refuses at, cannot run under that guard.
        """
        cap = int(guard.physical_cap_bytes)
        margin = int(guard.margin_bytes)
        if self.physical_limit_bytes > cap or self.safety_margin_bytes < margin:
            raise RuntimeError(
                f'retained COST plan exceeds the actual PB physical guard: plan '
                f'{self.physical_limit_bytes} bytes less {self.safety_margin_bytes} '
                f'margin against a {cap}-byte guard less {margin} margin')

    def require_observed_baseline(self, *, observed_bytes, source_bytes, label, actual_auxiliary_bytes=0):
        _integer(observed_bytes, 'observed_bytes')
        _integer(actual_auxiliary_bytes, 'actual_auxiliary_bytes')
        if actual_auxiliary_bytes > self.auxiliary_reserve_bytes:
            raise RuntimeError('actual auxiliary owner exceeds the declared auxiliary cap')
        # Unallocated auxiliary capacity is not spare metadata allowance.
        limit = (self.metadata_reserve_bytes + self.runtime_reserve_bytes
                 + source_bytes + actual_auxiliary_bytes)
        if observed_bytes > limit:
            raise RuntimeError(f'{label}: actual cgroup-plus-CUDA baseline {observed_bytes} exceeds declared '
                               f'metadata/runtime/source/auxiliary owners {limit}; reseal an admitted plan')


@dataclass(frozen=True)
class RetainedTarget:
    name: str
    statistics_bytes: int
    render_bytes: int
    largest_serialized_bytes: int
    candidate_delta_bytes: int
    candidate_count: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('target name must be nonempty')
        for key in ('statistics_bytes', 'render_bytes', 'largest_serialized_bytes', 'candidate_delta_bytes', 'candidate_count'):
            _integer(getattr(self, key), key, positive=True)


@dataclass(frozen=True)
class RetainedTargetWindow:
    names: tuple[str, ...]
    statistics_bytes: int
    render_bytes: int
    candidate_count: int
    peak_planned_bytes: int


@dataclass(frozen=True)
class RetainedTargetPlan:
    budget: RetainedWindowBudget
    source_bytes: int
    footprint_scope: str
    windows: tuple[RetainedTargetWindow, ...]

    def as_dict(self):
        return {'schema': 'prismaquant.joint_retained_target_plan.v1',
                'budget': self.budget.as_dict(), 'source_bytes': self.source_bytes,
                'fixed_bytes': self.budget.fixed_bytes(self.source_bytes),
                'footprint_scope': self.footprint_scope,
                'archive_admission': self.footprint_scope in ('pwc_selected_archive_storage', 'pwc_serialized_upper_bound'),
                'windows': [asdict(window) for window in self.windows]}


def plan_retained_targets(targets, *, budget, source_bytes, footprint_scope):
    """Greedily pack complete targets, enforcing every owner cap together."""
    if footprint_scope not in ('pwc_selected_archive_storage', 'pwc_serialized_upper_bound',
                               'logical_bf16_geometry_forecast'):
        raise ValueError('explicit owned archive or non-admitting geometry scope required')
    targets = tuple(sorted(targets, key=lambda target: target.name))
    if not targets or len({target.name for target in targets}) != len(targets):
        raise ValueError('nonempty unique target roster required')
    available = budget.available_window_bytes(source_bytes)
    fixed = budget.fixed_bytes(source_bytes)
    windows, names = [], []
    stats = renders = count = 0

    def fits(s, r):
        return s <= budget.statistics_cap_bytes and r <= budget.retained_render_cap_bytes and s + r <= available

    def flush():
        nonlocal names, stats, renders, count
        windows.append(RetainedTargetWindow(tuple(names), stats, renders, count, fixed + stats + renders))
        names, stats, renders, count = [], 0, 0, 0

    for target in targets:
        if (target.largest_serialized_bytes > budget.load_buffer_bytes
                or target.candidate_delta_bytes > budget.candidate_delta_bytes
                or not fits(target.statistics_bytes, target.render_bytes)):
            raise RuntimeError(f'{target.name}: indivisible target does not fit combined COST/loader/delta admission')
        if names and not fits(stats + target.statistics_bytes, renders + target.render_bytes):
            flush()
        names.append(target.name)
        stats += target.statistics_bytes
        renders += target.render_bytes
        count += target.candidate_count
    flush()
    if len(windows) > budget.max_windows_per_layer:
        raise RuntimeError(f'retained COST needs {len(windows)} windows, exceeding explicit replay-work cap '
                           f'{budget.max_windows_per_layer}; refusing an unbounded slow fallback')
    return RetainedTargetPlan(budget, source_bytes, footprint_scope, tuple(windows))


def targets_from_statistics_plan(statistics_plan, keys_by_name: Mapping, key_costs: Mapping):
    """Join matrices to serialized upper bounds, stable in a sealed read plan.

    PWC proves that an uncompressed archive's backing storage is no larger
    than its file. Charging the file size deliberately retains its small
    header slack, so runtime window membership matches the sealed file-size
    plan instead of silently repartitioning after ZIP inspection.
    """
    result, seen = [], set()
    for target in statistics_plan.targets:
        keys = tuple(keys_by_name[target.name])
        if not keys or len(set(keys)) != len(keys) or set(keys) & seen:
            raise ValueError('retained target candidate keys must be unique and owned by one target')
        seen.update(keys)
        costs = [key_costs[key] for key in keys]
        if any(set(cost) != {'incoming_storage_bytes', 'serialized_bytes'} for cost in costs):
            raise ValueError('PWC selected-key footprint grammar differs')
        if any(type(cost[field]) is not int or cost[field] <= 0 for cost in costs for field in cost):
            raise ValueError('retained COST requires an empty PWC baseline and file-backed candidate entries')
        if any(cost['incoming_storage_bytes'] > cost['serialized_bytes'] for cost in costs):
            raise ValueError('PWC archive storage exceeds its sealed serialized upper bound')
        result.append(RetainedTarget(target.name, target.statistics_bytes,
            sum(cost['serialized_bytes'] for cost in costs),
            max(cost['serialized_bytes'] for cost in costs),
            4 * target.shape[0] * target.shape[1], len(keys)))
    if seen != set(key_costs):
        raise ValueError('PWC footprint roster differs from retained joint targets')
    return tuple(result)


def normalize_retained_execution(value, *, operator_windows, boundary_storage):
    """An explicit COST-only lifetime policy; PREPARE's cap is unchanged."""
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) !=
            {'schema', 'budget', 'source_reserve_bytes', 'source_loading_reserve_bytes'}
            or value.get('schema') != EXECUTION_SCHEMA):
        raise ValueError('complete versioned retained COST execution policy required')
    budget = RetainedWindowBudget.from_dict(value['budget'])
    source = _integer(value['source_reserve_bytes'], 'source_reserve_bytes', positive=True)
    load = _integer(value['source_loading_reserve_bytes'], 'source_loading_reserve_bytes', positive=True)
    if operator_windows is None or not isinstance(boundary_storage, dict):
        raise ValueError('retained COST needs existing operator and exact boundary owners')
    if boundary_storage.get('capture_order') != 'layer_major':
        raise ValueError('retained COST requires layer-major source capture')
    for field, bound in [('max_resident_bytes', budget.boundary_reserve_bytes),
                         ('max_auxiliary_bytes', budget.auxiliary_reserve_bytes)]:
        if boundary_storage[field] > bound:
            raise RuntimeError(f'retained COST undercharges boundary owner {field}')
    for field, bound in [('max_statistics_bytes', budget.statistics_cap_bytes),
                         ('max_candidate_bytes', budget.candidate_delta_bytes),
                         ('max_load_buffer_bytes', budget.load_buffer_bytes)]:
        if operator_windows[field] < bound:
            raise RuntimeError(f'retained COST budget exceeds its operator owner {field}')
    budget.available_window_bytes(source)
    # Loading and forward/adjoint work are separate phases: settled source
    # prefetch prevents this allowance overlapping the graph workspace.
    source_peak = (budget.source_baseline_limit(source) + budget.boundary_reserve_bytes + load)
    if source_peak > budget.physical_limit_bytes - budget.safety_margin_bytes:
        raise RuntimeError('source-loading phase cannot fit the retained COST physical budget')
    return {'schema': EXECUTION_SCHEMA, 'budget': budget.as_dict(),
            'source_reserve_bytes': source, 'source_loading_reserve_bytes': load}


def _roster_maximum(targets, field):
    """The largest declared value of ``field``, and the target that sets it."""
    winner = max(targets, key=lambda target: (getattr(target, field), target.name))
    return getattr(winner, field), winner.name


def derive_retained_window_budget(targets_by_layer, *, declared, source_bytes,
                                  prefetch_workers,
                                  footprint_scope='pwc_serialized_upper_bound'):
    """Derive every demand-driven cap from the roster the budget must admit.

    An operator declares the physical bound and the reserves that belong to
    the box, the runtime and the capture (``DECLARED_BUDGET_FIELDS``). The
    five caps in ``DERIVED_BUDGET_FIELDS`` are not judgement calls: each is a
    maximum over bytes the roster already states, so each is computed here and
    the maximizing target is recorded beside it.

    * ``candidate_delta_bytes`` is one FP32 delta over one whole matrix.
      ``JointOperatorStatisticsLease.project`` charges the storages of a single
      quantum and the replay hands it one ``(name, format)`` pair at a time, so
      the demand is the largest single matrix in the roster, not a sum.
    * ``load_buffer_bytes`` bounds the serialized bytes one load quantum reads
      *at once*, and ``ProductionWeightCache.plan_resident_windows`` closes a
      quantum on it while ``prefetch_workers`` sets the loader concurrency. Its
      floor is the single largest candidate file; at exactly that floor the
      declared concurrency cannot be reached, because one file fills the
      buffer. The derived value is therefore the declared concurrency times
      that floor -- the smallest buffer at which the policy's own
      ``prefetch_workers`` is honest.
    * ``statistics_cap_bytes`` and ``retained_render_cap_bytes`` bound one
      window. Below ``available_window_bytes`` neither has an independent
      physical meaning -- the packer already refuses ``statistics + renders >
      available`` -- so the packing is solved against the physical bound alone
      and each cap is then set to the largest window the packing actually
      produces. Tightening a cap to a maximum the packing already satisfies
      cannot change a packing decision, and the fixed point is asserted below
      rather than assumed.
    * ``max_windows_per_layer`` is the worst layer's window count under that
      packing, i.e. the fewest retained windows the physical budget admits. It
      remains a refusal -- runtime geometry needing more windows than the
      sealed packing still stops -- but it is no longer free headroom, and the
      replay multiplier it implies is recorded so the cost is visible.

    Returns ``(budget, derivation_record)``. The record is data for a plan's
    top level; it is deliberately not a field of the budget, whose ``from_dict``
    admits exactly its own keys.
    """
    if (not isinstance(declared, Mapping)
            or set(declared) != set(DECLARED_BUDGET_FIELDS)):
        raise ValueError('retained budget derivation requires exactly the declared owners')
    for name in DECLARED_BUDGET_FIELDS:
        _integer(declared[name], name,
                 positive=name not in ('boundary_reserve_bytes', 'auxiliary_reserve_bytes'))
    _integer(source_bytes, 'source_bytes')
    _integer(prefetch_workers, 'prefetch_workers', positive=True)
    if not isinstance(targets_by_layer, Mapping) or not targets_by_layer:
        raise ValueError('retained budget derivation requires a nonempty per-layer roster')
    roster = []
    for layer, targets in sorted(targets_by_layer.items()):
        targets = tuple(targets)
        if not targets or any(not isinstance(target, RetainedTarget) for target in targets):
            raise ValueError(f'layer {layer} has no declared retained targets')
        roster.extend(targets)
    if len({target.name for target in roster}) != len(roster):
        raise ValueError('retained budget derivation requires one owner per target name')

    candidate_delta_bytes, candidate_delta_target = _roster_maximum(roster, 'candidate_delta_bytes')
    largest_serialized_bytes, load_buffer_target = _roster_maximum(roster, 'largest_serialized_bytes')
    load_buffer_bytes = prefetch_workers * largest_serialized_bytes
    # Neither window cap enters ``fixed_bytes``, so the window space is settled
    # once the two per-quantum owners above are.
    probe = RetainedWindowBudget(
        **declared, load_buffer_bytes=load_buffer_bytes,
        candidate_delta_bytes=candidate_delta_bytes, statistics_cap_bytes=1,
        retained_render_cap_bytes=1, max_windows_per_layer=1)
    available = probe.available_window_bytes(source_bytes)
    open_budget = replace(probe, statistics_cap_bytes=available,
                          retained_render_cap_bytes=available,
                          max_windows_per_layer=max(len(tuple(targets))
                                                    for targets in targets_by_layer.values()))
    plans = {layer: plan_retained_targets(targets, budget=open_budget,
                                          source_bytes=source_bytes,
                                          footprint_scope=footprint_scope)
             for layer, targets in sorted(targets_by_layer.items())}
    windows = [window for plan in plans.values() for window in plan.windows]
    budget = replace(open_budget,
                     statistics_cap_bytes=max(window.statistics_bytes for window in windows),
                     retained_render_cap_bytes=max(window.render_bytes for window in windows),
                     max_windows_per_layer=max(len(plan.windows) for plan in plans.values()))
    settled = {layer: plan_retained_targets(targets, budget=budget, source_bytes=source_bytes,
                                            footprint_scope=footprint_scope)
               for layer, targets in sorted(targets_by_layer.items())}
    if any(settled[layer].windows != plan.windows for layer, plan in plans.items()):
        raise RuntimeError('retained budget derivation did not reach a fixed point')

    record = {
        'schema': DERIVATION_SCHEMA,
        'footprint_scope': footprint_scope,
        'source_bytes': source_bytes,
        'declared': {name: declared[name] for name in DECLARED_BUDGET_FIELDS},
        'prefetch_workers': prefetch_workers,
        'roster': {'targets': len(roster), 'layers': len(targets_by_layer),
                   'targets_by_layer': {str(layer): len(tuple(targets))
                                        for layer, targets in sorted(targets_by_layer.items())}},
        'demand': {
            'candidate_delta_bytes': {'bytes': candidate_delta_bytes,
                                      'maximizing_target': candidate_delta_target,
                                      'basis': 'one fp32 delta over one whole matrix'},
            'load_buffer_bytes': {'bytes': load_buffer_bytes,
                                  'maximizing_target': load_buffer_target,
                                  'largest_serialized_bytes': largest_serialized_bytes,
                                  'prefetch_workers': prefetch_workers,
                                  'basis': 'declared loader concurrency times the largest '
                                           'single serialized candidate file'},
            'statistics_cap_bytes': {'bytes': budget.statistics_cap_bytes,
                                     'basis': 'largest packed window statistics'},
            'retained_render_cap_bytes': {'bytes': budget.retained_render_cap_bytes,
                                          'basis': 'largest packed window render files'},
            'max_windows_per_layer': {'windows': budget.max_windows_per_layer,
                                      'basis': 'worst layer under the physical window bound'},
        },
        'fixed_bytes': budget.fixed_bytes(source_bytes),
        'available_window_bytes': budget.available_window_bytes(source_bytes),
        'windows_by_layer': {str(layer): len(plan.windows) for layer, plan in sorted(settled.items())},
        'peak_planned_bytes': max(window.peak_planned_bytes
                                  for plan in settled.values() for window in plan.windows),
        'retained_window_replay_multiplier': (sum(len(plan.windows) for plan in settled.values())
                                              / len(settled)),
        'budget': budget.as_dict(),
    }
    return budget, record
