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
#: The declared owners a policy may take from a measurement instead
#: (PQ #1151). A measured owner carries its bytes and the receipt they came
#: from; the receipt is a reference for readers, and nothing compares it at
#: run time.
MEASURED_BUDGET_FIELDS = ('workspace_reserve_bytes',)
_RECEIPT_FIELDS = ('action_key', 'path', 'sha256')
#: The caps that are a function of the roster the budget must admit. Every one
#: of these is a maximum over declared bytes, so none of them is a judgement
#: call and none of them belongs in a hand-written plan.
DERIVED_BUDGET_FIELDS = ('load_buffer_bytes', 'candidate_delta_bytes',
                         'statistics_cap_bytes', 'retained_render_cap_bytes',
                         'max_windows_per_layer')
#: The owners that land on the HOST side of a unified-memory box, which the
#: kernel bounds with the container's own cgroup cap whatever the aggregate
#: says (``CaptureMemoryGuard._check`` holds the cgroup's committed bytes
#: against ``cap - margin`` in aggregate mode too). Retained renders belong here:
#: ``ProductionWeightCache._load_file_tensor`` reads every candidate with
#: ``map_location="cpu"`` and the retained window holds those CPU tensors for
#: the whole window, while the fp32 delta and the statistics matrices are
#: built on the device.
HOST_RESIDENT_BUDGET_FIELDS = ('safety_margin_bytes', 'metadata_reserve_bytes',
                               'load_buffer_bytes', 'read_page_reserve_bytes')


#: The capture-guard reading ``require_observed_baseline`` compares with the
#: declared owners: committed cgroup memory plus the whole CUDA reservation
#: (``memory_management.committed_cgroup_bytes``). One key, named once, so the
#: joint-cost quantum and the AURA retained path cannot read two definitions
#: (PQ #1157).
OBSERVED_BASELINE_KEY = 'committed_cgroup_plus_cuda_reserved_bytes'


def _integer(value, name, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f'{name} must be an exact {"positive" if positive else "nonnegative"} integer')
    return value


def capture_workspace_bytes(workspace_reserve_bytes, capture_batch):
    """The device workspace one Stage B pass holds: one reserve per stored batch.

    The one quantity both sides price (PQ #1151): the guard charges it before
    every window backward and capture pass (``joint_cost_quantum``), and the
    derivation plans the capture pass with it
    (:meth:`RetainedWindowBudget.capture_peak_bytes`). A pass without the
    spill observer carries one stored batch; a capture pass carries the
    replay regime's ``capture_batch``.
    """
    _integer(workspace_reserve_bytes, 'workspace_reserve_bytes', positive=True)
    _integer(capture_batch, 'capture_batch', positive=True)
    return workspace_reserve_bytes * capture_batch


def _measured_owner(name, value):
    """A measured owner: exact positive bytes and the receipt they came from."""
    if (not isinstance(value, Mapping) or set(value) != {'bytes', 'receipt', 'basis'}
            or not isinstance(value['basis'], str) or not value['basis']):
        raise ValueError(f'measured {name} needs exactly bytes, receipt and basis')
    _integer(value['bytes'], name, positive=True)
    receipt = value['receipt']
    if not isinstance(receipt, Mapping) or set(receipt) != set(_RECEIPT_FIELDS):
        raise ValueError(f'measured {name} receipt needs exactly {list(_RECEIPT_FIELDS)}')
    for field in ('action_key', 'sha256'):
        digest = receipt[field]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in '0123456789abcdef' for char in digest)):
            raise ValueError(f'measured {name} receipt {field} must be 64 lowercase hex')
    if not isinstance(receipt['path'], str) or not receipt['path'].startswith('/'):
        raise ValueError(f'measured {name} receipt path must be absolute')
    return value['bytes']


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

    def capture_workspace_bytes(self, capture_batch):
        return capture_workspace_bytes(self.workspace_reserve_bytes, capture_batch)

    def capture_peak_bytes(self, source_bytes, *, capture_batch, render_bytes):
        """The planned peak of one capture pass (PQ #1151).

        A capture runs inside a retained window with that window's renders
        resident and no statistics lease open, so its peak is the fixed
        owners with the one workspace reserve they hold replaced by the
        pass's :func:`capture_workspace_bytes`, plus the window's renders.
        """
        _integer(render_bytes, 'render_bytes')
        return (self.fixed_bytes(source_bytes) - self.workspace_reserve_bytes
                + self.capture_workspace_bytes(capture_batch) + render_bytes)

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
        """Refuse a process floor its declared owners do not hold.

        ``observed_bytes`` is the capture guard's ``OBSERVED_BASELINE_KEY``
        reading: committed cgroup memory plus the whole CUDA reservation. Clean
        file pages left by a source or checkpoint read are not in it, because
        the kernel drops them before it refuses an allocation (PQ #1157).
        """
        _integer(observed_bytes, 'observed_bytes')
        _integer(actual_auxiliary_bytes, 'actual_auxiliary_bytes')
        if actual_auxiliary_bytes > self.auxiliary_reserve_bytes:
            raise RuntimeError('actual auxiliary owner exceeds the declared auxiliary cap')
        # Unallocated auxiliary capacity is not spare metadata allowance.
        limit = (self.metadata_reserve_bytes + self.runtime_reserve_bytes
                 + source_bytes + actual_auxiliary_bytes)
        if observed_bytes > limit:
            raise RuntimeError(f'{label}: committed cgroup-plus-CUDA baseline {observed_bytes} exceeds declared '
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
                                  prefetch_workers, host_cap_bytes,
                                  footprint_scope='pwc_serialized_upper_bound',
                                  measured=None, capture_batch=None):
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
      window. The statistics matrices are device-side, so the aggregate
      ``available_window_bytes`` is their only bound and the packer's
      ``statistics + renders > available`` refusal already holds it. Retained
      renders are host-side, so they are bounded twice: by that same aggregate
      window and by ``host_cap_bytes`` less the host-resident owners
      (``HOST_RESIDENT_BUDGET_FIELDS``), because the kernel holds the
      container's cgroup cap whatever the aggregate says. The packing is
      solved against both, and each cap is then set to the largest window the
      packing actually produces. Tightening a cap to a maximum the packing
      already satisfies cannot change a packing decision, and the fixed point
      is asserted below rather than assumed.
    * ``max_windows_per_layer`` is the worst layer's window count under that
      packing, i.e. the fewest retained windows the physical budget admits. It
      remains a refusal -- runtime geometry needing more windows than the
      sealed packing still stops -- but it is no longer free headroom, and the
      replay multiplier it implies is recorded so the cost is visible.

    ``measured`` (PQ #1151) takes the owners in ``MEASURED_BUDGET_FIELDS``
    from a measurement instead of ``declared``: each is ``{bytes, receipt,
    basis}``, and the receipt names the action key, path and sha256 of the
    measurement it came from. ``capture_batch`` plans the Stage B capture pass
    too: :meth:`RetainedWindowBudget.capture_peak_bytes` beside the largest
    window's renders, since a resumed quantum captures in whichever window
    is its first active one. A capture that does not fit refuses here, before
    any run pays for finding it. With neither, the derivation and its record
    are exactly the ones before #1151.

    Returns ``(budget, derivation_record)``. The record is data for a plan's
    top level; it is deliberately not a field of the budget, whose ``from_dict``
    admits exactly its own keys.
    """
    expected = set(DECLARED_BUDGET_FIELDS) - (set() if measured is None
                                              else set(MEASURED_BUDGET_FIELDS))
    if not isinstance(declared, Mapping) or set(declared) != expected:
        raise ValueError('retained budget derivation requires exactly the declared owners')
    for name in sorted(expected):
        _integer(declared[name], name,
                 positive=name not in ('boundary_reserve_bytes', 'auxiliary_reserve_bytes'))
    owners = dict(declared)
    if measured is not None:
        if not isinstance(measured, Mapping) or set(measured) != set(MEASURED_BUDGET_FIELDS):
            raise ValueError('retained budget derivation requires exactly the measured owners')
        owners.update({name: _measured_owner(name, measured[name])
                       for name in MEASURED_BUDGET_FIELDS})
    if capture_batch is not None:
        _integer(capture_batch, 'capture_batch', positive=True)
    _integer(source_bytes, 'source_bytes')
    _integer(prefetch_workers, 'prefetch_workers', positive=True)
    _integer(host_cap_bytes, 'host_cap_bytes', positive=True)
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
        **owners, load_buffer_bytes=load_buffer_bytes,
        candidate_delta_bytes=candidate_delta_bytes, statistics_cap_bytes=1,
        retained_render_cap_bytes=1, max_windows_per_layer=1)
    available = probe.available_window_bytes(source_bytes)
    host_render_bound = host_cap_bytes - sum(getattr(probe, name)
                                             for name in HOST_RESIDENT_BUDGET_FIELDS)
    if host_render_bound <= 0:
        raise RuntimeError('retained COST host owners exhaust the container cap before renders')
    open_budget = replace(probe, statistics_cap_bytes=available,
                          retained_render_cap_bytes=min(available, host_render_bound),
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
    window_peak = max(window.peak_planned_bytes
                      for plan in settled.values() for window in plan.windows)
    capture = None
    if capture_batch is not None:
        capture_peak = budget.capture_peak_bytes(
            source_bytes, capture_batch=capture_batch,
            render_bytes=budget.retained_render_cap_bytes)
        bound = budget.physical_limit_bytes - budget.safety_margin_bytes
        if capture_peak > bound:
            raise RuntimeError(
                f'retained COST capture pass at capture_batch {capture_batch} plans '
                f'{capture_peak} bytes ({budget.capture_workspace_bytes(capture_batch)} '
                f'workspace beside {budget.retained_render_cap_bytes} of renders) against '
                f'{bound} bytes of physical budget less margin')
        capture = {'capture_batch': capture_batch,
                   'workspace_bytes': budget.capture_workspace_bytes(capture_batch),
                   'render_bytes': budget.retained_render_cap_bytes,
                   'peak_planned_bytes': capture_peak,
                   'basis': 'fixed owners with workspace_reserve_bytes times capture_batch '
                            'in place of one reserve, beside the largest window renders; '
                            'no statistics lease is open during a capture'}

    record = {
        'schema': DERIVATION_SCHEMA,
        'footprint_scope': footprint_scope,
        'source_bytes': source_bytes,
        'declared': {name: declared[name] for name in DECLARED_BUDGET_FIELDS
                     if name in declared},
        'prefetch_workers': prefetch_workers,
        'host_cap_bytes': host_cap_bytes,
        'host_render_bound_bytes': host_render_bound,
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
        'host_cap_bytes': host_cap_bytes,
        'host_render_bound_bytes': host_render_bound,
                                  'basis': 'declared loader concurrency times the largest '
                                           'single serialized candidate file'},
            'statistics_cap_bytes': {'bytes': budget.statistics_cap_bytes,
                                     'basis': 'largest packed window statistics'},
            'retained_render_cap_bytes': {'bytes': budget.retained_render_cap_bytes,
                                          'host_cap_bytes': host_cap_bytes,
                                          'host_render_bound_bytes': host_render_bound,
                                          'aggregate_window_bytes': available,
                                          'basis': 'largest packed window render files, under the '
                                                   'smaller of the aggregate window and the '
                                                   "container's host-side headroom"},
            'max_windows_per_layer': {'windows': budget.max_windows_per_layer,
                                      'basis': 'worst layer under the physical window bound'},
        },
        'fixed_bytes': budget.fixed_bytes(source_bytes),
        'available_window_bytes': budget.available_window_bytes(source_bytes),
        'windows_by_layer': {str(layer): len(plan.windows) for layer, plan in sorted(settled.items())},
        'peak_planned_bytes': (window_peak if capture is None
                               else max(window_peak, capture['peak_planned_bytes'])),
        'retained_window_replay_multiplier': (sum(len(plan.windows) for plan in settled.values())
                                              / len(settled)),
        'budget': budget.as_dict(),
    }
    if measured is not None:
        record['measured'] = {name: {'bytes': measured[name]['bytes'],
                                     'receipt': dict(measured[name]['receipt']),
                                     'basis': measured[name]['basis']}
                              for name in MEASURED_BUDGET_FIELDS}
    if capture is not None:
        record['capture'] = capture
    return budget, record
