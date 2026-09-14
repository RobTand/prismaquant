"""Combined physical admission for one statistics plane and retained PWC renders.

This module owns scalar plans only. Source, boundary/auxiliary state, PWC and
operator statistics retain their existing owners. A geometry forecast using
logical tensor sizes is expressly not an archive-storage admission receipt.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


SCHEMA = 'prismaquant.joint_retained_window_budget.v1'


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

    def require_observed_baseline(self, *, observed_bytes, source_bytes, label):
        _integer(observed_bytes, 'observed_bytes')
        limit = self.source_baseline_limit(source_bytes)
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
