"""Explicit bounded target replay over existing source and PWC owners."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import os

import torch

from .joint_aura import JointOperatorStatisticsLease, arithmetic_identity
from .joint_retained_window_plan import (
    RetainedWindowBudget, plan_retained_targets, targets_from_statistics_plan,
)
from .joint_statistics_plan import plan_joint_statistics_target_windows
from .joint_served_activation import joint_activation_maxima

SCHEMA = 'prismaquant.joint_operator_windows.v1'
_FIELDS = {'schema', 'max_statistics_bytes', 'max_candidate_bytes',
           'max_render_resident_bytes', 'max_load_buffer_bytes',
           'workspace_reserve_bytes', 'max_replay_cotangent_bytes', 'prefetch_workers'}


def normalize_operator_windows(config):
    if config is None:
        return None
    if not isinstance(config, dict) or set(config) != _FIELDS or config.get('schema') != SCHEMA:
        raise ValueError('joint operator windows require a complete v1 policy')
    if any(type(config[key]) is not int or config[key] <= 0 for key in _FIELDS - {'schema'}):
        raise ValueError('joint operator window budgets must be positive integers')
    return dict(config)


def statistics_arithmetic_identity(dtype, backend, replay_regime=None):
    """The statistics arithmetic, with any non-default Stage B replay regime.

    The default regime (``None``) returns the same identity as before regimes
    existed; see :mod:`prismaquant.joint_replay_regime`.
    """
    from .joint_replay_regime import stamp_replay_regime
    result = arithmetic_identity(dtype, backend)
    result.update(weight_projection='summed_output_operator_fp32_gemm',
        operator_accumulation='sum_fp32_matrices_in_backward_invocation_order',
        contraction_order='sum_operators_then_project_each_signed_component')
    return stamp_replay_regime(result, replay_regime)


def operator_window_guard(device, *, device_bytes=None):
    """The capture guard the joint replay charges its phases to.

    With ``device_bytes`` -- the plan's ``max_gpu_bytes``, the envelope
    ``enforce_device_envelope`` already holds on the allocator -- the guard is
    an AGGREGATE one: every phase reservation below is the plan's conservative
    sum (a resident window, a statistics lease, a boundary reserve: none of them
    says which side of a unified-memory box it lands on), and the retained COST
    budget states its ``physical_limit_bytes`` as ``cpu cap + device envelope``.
    A cgroup-only cap refused that sum at 24 GiB against 104 GiB on the
    GLM-5.3-Flash run stage (2026-09-18). Without ``device_bytes`` the guard is
    the original conservative one against the cgroup cap, unchanged.
    """
    if torch.device(device).type != 'cuda':
        return None
    from .autoscale import require_bounded_capture_environment
    from .memory_management import CaptureMemoryGuard
    require_bounded_capture_environment(os.environ)
    if device_bytes is None:
        guard = CaptureMemoryGuard(device)
    else:
        guard = CaptureMemoryGuard(device, device_bytes=device_bytes,
                                   aggregate_envelope=True)
    guard.check('before_joint_operator_identity')
    return guard


def check_operator_allocation(guard, label, *, reserve_bytes, reserve_device_bytes=0):
    """Release retired blocks before charging a phase's future allocations.

    Live source, statistics and PWC owners remain intact. The existing physical
    guard still charges their entire CUDA reservation and cgroup footprint.

    ``reserve_device_bytes`` is the part of the future allocation the CUDA
    allocator will hold. A guard with a declared device envelope takes it on
    the device side, where it is also held against ``device_bytes``; a guard
    without one has no device budget, so the two sides are charged as one sum
    exactly as before (PQ #1157).
    """
    from .aura_cost import _release_streamed_anchor_allocator_cache
    _release_streamed_anchor_allocator_cache(guard.device)
    if not reserve_device_bytes:
        return guard.check(label, reserve_bytes=reserve_bytes)
    if getattr(guard, 'device_bytes', None) is None:
        return guard.check(label, reserve_bytes=reserve_bytes + reserve_device_bytes)
    return guard.check(label, reserve_bytes=reserve_bytes,
                       reserve_device_bytes=reserve_device_bytes)


@contextmanager
def resident_candidates(cache, keys, policy, *, guard=None):
    """Use PWC's finite windows; borrowed renders must not escape the yield."""
    # TWO BUDGETS, PLANNED AS TWO. The serialized buffer cap can be smaller than
    # the resident cap, and the planner now closes a quantum on each of them, so
    # planning against their minimum no longer buys safety -- it only charged the
    # buffer cap against residency and made a quantum narrower than the bytes
    # this policy admits (#693). The guard below is reserved on the same two
    # terms, so what a quantum may hold is what was priced for it.
    cap = policy['max_render_resident_bytes']
    requested = {}
    for key in keys:
        requested.setdefault(cache.resolve_key(*key), []).append(key)
    windows = cache.plan_resident_windows(keys, max_resident_bytes=cap,
                                         max_load_buffer_bytes=policy['max_load_buffer_bytes'],
                                         max_workers=policy['prefetch_workers'])
    # This context supplies an iterator; each PWC context owns exactly one
    # quantum until the consumer advances or closes the iterator.
    def iterate():
        for keys in windows:
            if guard is not None:
                check_operator_allocation(guard, 'before_joint_candidate_load', reserve_bytes=(
                    cap + policy['max_load_buffer_bytes'] + policy['max_candidate_bytes']
                    + policy['workspace_reserve_bytes']))
            with cache.resident_window(keys, max_resident_bytes=cap,
                    max_load_buffer_bytes=policy['max_load_buffer_bytes'],
                    max_workers=policy['prefetch_workers'], release_file_pages=True) as receipt:
                yield tuple(pair for key in keys for pair in requested[key]), receipt
    iterator = iterate()
    try:
        yield iterator
    finally:
        iterator.close()


def observe_and_project_windows(modules, specs, cache, policy, *, backward,
                                record_operator, collect_col_energy, backend, guard=None, source_fingerprints=None):
    """Run all batches before contracting a target, committing backward once.

    ``backward`` receives final=True only for the last window; the caller owns
    exact incoming boundaries and forks shared cotangents for earlier windows.
    No source install or checkpoint read occurs here. Modules stay installed
    throughout every window of this probe. A fresh lease owns each matrix set.
    """
    plan = plan_joint_statistics_target_windows(modules, specs,
        max_statistics_bytes=policy['max_statistics_bytes'],
        activation_max_abs=joint_activation_maxima(cache), projection_backend=backend)
    largest = max(4 * target.shape[0] * target.shape[1] for target in plan.targets)
    if largest > min(policy['max_candidate_bytes'], policy['workspace_reserve_bytes']):
        raise RuntimeError('joint single target exceeds candidate or matrix workspace budget')
    if source_fingerprints is None:
        source_fingerprints = {name: JointOperatorStatisticsLease._source_fingerprint(module.weight)
                               for name, module in modules.items()}
    if set(source_fingerprints) != set(modules):
        raise RuntimeError('joint source seal coverage differs')
    def require_sources():
        if any(JointOperatorStatisticsLease._source_fingerprint(modules[name].weight) != fingerprint
               for name, fingerprint in source_fingerprints.items()):
            raise RuntimeError('joint source changed between target windows')
    results, diagnostics, receipts = {}, {}, []
    for index, names in enumerate(plan.windows):
        require_sources()
        selected = {name: modules[name] for name in names}
        if guard is not None:
            check_operator_allocation(guard, 'before_joint_statistics_window', reserve_bytes=(
                plan.window_statistics_bytes[index] + policy['workspace_reserve_bytes']
                + policy['max_replay_cotangent_bytes']))
        with JointOperatorStatisticsLease(selected, {name: specs[name] for name in names},
                max_statistics_bytes=policy['max_statistics_bytes'],
                max_candidate_bytes=policy['max_candidate_bytes'],
                activation_max_abs=joint_activation_maxima(cache), projection_backend=backend) as lease:
            lease.begin_probe()
            backward(final=index == len(plan.windows)-1, lease=lease)
            require_sources()
            lease.finish_observations()
            diagnostics.update(lease.operator_diagnostics(collect_col_energy=collect_col_energy))
            keys = [(name, fmt) for name in names for fmt in specs[name]]
            with resident_candidates(cache, keys, policy, guard=guard) as windows:
                for quantum, receipt in windows:
                    for name, fmt in quantum:
                        rendered = source = delta = None
                        try:
                            source = modules[name].weight.detach()
                            rendered = cache.get_resident(name, fmt)
                            record_operator(name, fmt, source, rendered)
                            # One exact FP32 dW quantum, independent of menu size.
                            delta = rendered.to(device=source.device, dtype=torch.float32, copy=True)
                            delta.sub_(source)
                            lease.project({(name, fmt): delta})
                        finally:
                            rendered = source = delta = None
                    receipts.append(dict(receipt))
            results.update(lease.finish_projections())
    return results, diagnostics, dict(plan=plan.as_dict(), candidate_windows=receipts)


@dataclass(frozen=True)
class PreflightRetainedWindow:
    """One admitted window, in the shape ``sealed_windows`` already compares.

    A sealed PrismaBuild read schedule and this preflight answer the same
    question from the same declared bytes, so they reach the per-layer replay
    through one channel instead of two.
    """
    original_full_target_names: tuple[str, ...]
    statistics_bytes: int
    render_file_upper_bound_bytes: int
    candidate_count: int


def retained_window_keys(window_names, specs, cache, *, completed_names=()):
    """``[(window index, keys)]`` for every window with a pending target.

    The keys each retained window opens on, in the order
    :func:`observe_and_project_retained_windows` opens them: the window's
    pending targets in window order, each with its candidates in ``specs``
    order. A caller builds its read stream from these (PQ #1291), and the
    replay refuses a stream whose groups differ.
    """
    completed = set(completed_names)
    result = []
    for index, names in enumerate(window_names):
        pending = [name for name in names if name not in completed]
        if not pending:
            continue
        requested = [(name, fmt) for name in pending for fmt in specs[name]]
        keys = tuple(cache.resolve_key(name, fmt) for name, fmt in requested)
        if any(key is None for key in keys):
            missing = [pair for pair, key in zip(requested, keys) if key is None]
            raise RuntimeError(f'retained joint PWC candidate entry missing: {missing}')
        result.append((index, keys))
    return result


class GuardReadBudget:
    """A read stream's budget, read off the replay's capture guard (PQ #1291).

    ``headroom_bytes`` is ``CaptureMemoryGuard.headroom_bytes``: the host
    bytes the row can still take beside the phase it is in. That reading is
    of the process, so the renders the stream already holds are in it and
    ``held_bytes`` is not subtracted again. The serialized buffers in flight
    are held to the sealed load buffer.

    Two terms for a stream read beside another (PQ #1348, Stage B's spill
    replay beside the render stream). ``yield_to`` is the stream whose next
    group comes first: what reading that group still charges
    (``ReadStream.unread_bytes``) is taken off the reading, so the second
    stream never reads into the room the first one's next group needs.
    ``floor_bytes`` is what the phase already reserved for this stream at its
    guard check (the spill's own read buffers): the stream may always hold
    that much, read ahead or taken, whatever the reading says.
    """

    def __init__(self, guard, *, buffer_bytes, yield_to=None, floor_bytes=0):
        if type(floor_bytes) is not int or floor_bytes < 0:
            raise ValueError('a read budget floor must be nonnegative bytes')
        self.guard = guard
        self.buffer_bytes = int(buffer_bytes)
        self.yield_to = yield_to
        self.floor_bytes = floor_bytes

    def headroom_bytes(self, held_bytes: int) -> int:
        live = self.guard.headroom_bytes()
        if self.yield_to is not None:
            group = self.yield_to.next_group()
            if group is not None:
                live -= self.yield_to.unread_bytes(group)
        return max(live, self.floor_bytes - held_bytes)


#: The device bytes a render cache may hold when no capture guard reads the
#: row (a CPU run, or a test that drives the quantum unguarded): none. Every
#: CUDA quantum runs under a guard, whose device headroom admits instead.
UNGUARDED_RENDER_CACHE_BYTES = 0


class RetainedRenderDeviceCache:
    """One retained window's renders, kept on the device across its probes (PQ #1348).

    Each probe of a window builds every candidate's delta,
    ``render - source`` in FP32 on the source's device, from the same
    resident render. The first probe copies the render to the device as it
    is (its own dtype) and keeps that copy while ``headroom()`` admits its
    bytes; a later probe widens the kept copy instead of copying the host
    render again. The delta is the same bytes either way: the device copy is
    exact, and so is the widening of a narrower float to FP32. A render not
    admitted, or dropped by :meth:`reclaim`, takes the uncached path, the
    one every probe took before.

    ``headroom()`` is the device bytes that may still be allocated
    (``CaptureMemoryGuard.device_headroom_bytes``). :meth:`reclaim` is a
    device-side guard reclaimer. ``counters`` are cumulative over the pass;
    ``bytes_held`` is what the window holds now, and
    ``last_window_peak_bytes`` the most the last cleared window held.
    """

    def __init__(self, headroom):
        if not callable(headroom):
            raise TypeError('a render device cache needs a headroom callable')
        self._headroom = headroom
        self._held: dict = {}
        self.bytes_held = 0
        self._window_peak = 0
        self.last_window_peak_bytes = 0
        self.counters = {'hits': 0, 'misses': 0, 'admitted': 0, 'refused': 0,
                         'reclaims': 0, 'reclaimed_bytes': 0, 'peak_bytes_held': 0}

    def delta(self, key, rendered, source, *, keep):
        """``rendered - source`` in FP32 on ``source.device``.

        ``rendered`` is the resident render, or a callable returning it,
        called only on a miss. ``keep`` says a later probe of this window
        will ask for ``key`` again, so a miss may keep its device copy.
        """
        held = self._held.get(key)
        if held is not None:
            self.counters['hits'] += 1
            delta = held.to(dtype=torch.float32, copy=True)
        else:
            self.counters['misses'] += 1
            if callable(rendered):
                rendered = rendered()
            nbytes = rendered.numel() * rendered.element_size()
            if keep and nbytes <= self._headroom():
                held = rendered.to(device=source.device, copy=True)
                self._held[key] = held
                self.bytes_held += nbytes
                self.counters['admitted'] += 1
                self._window_peak = max(self._window_peak, self.bytes_held)
                self.counters['peak_bytes_held'] = max(
                    self.counters['peak_bytes_held'], self.bytes_held)
                delta = held.to(dtype=torch.float32, copy=True)
            else:
                self.counters['refused'] += int(keep)
                delta = rendered.to(device=source.device, dtype=torch.float32, copy=True)
        held = rendered = None
        delta.sub_(source)
        return delta

    def reclaim(self, shortfall_bytes):
        """Drop kept renders until ``shortfall_bytes`` is freed; a guard reclaimer.

        The dropped copies are CUDA allocations the caching allocator would
        keep reserved, so their blocks go back to the device before the
        guard reads it again.
        """
        freed, cuda = 0, False
        for key in list(self._held):
            if freed >= shortfall_bytes:
                break
            tensor = self._held.pop(key)
            nbytes = tensor.numel() * tensor.element_size()
            cuda = cuda or tensor.is_cuda
            tensor = None
            self.bytes_held -= nbytes
            freed += nbytes
        if freed:
            self.counters['reclaims'] += 1
            self.counters['reclaimed_bytes'] += freed
            if cuda:
                torch.cuda.empty_cache()
        return freed

    def clear(self):
        """The window is done: drop every kept render."""
        self._held.clear()
        self.bytes_held = 0
        self.last_window_peak_bytes, self._window_peak = self._window_peak, 0


def retained_admission_targets(statistics_plan, specs, cache):
    """Join a statistics plan to the PWC's declared candidate file sizes.

    Reads no tensor. ``resolve_key`` is an index lookup and ``estimate_nbytes``
    is one ``stat`` per candidate file, so every byte this returns is declared
    before any capture, probe or projection runs.
    """
    keys_by_name, requested_by_name = {}, {}
    for target in statistics_plan.targets:
        requested = tuple((target.name, fmt) for fmt in specs[target.name])
        keys = tuple(cache.resolve_key(name, fmt) for name, fmt in requested)
        if any(key is None for key in keys):
            missing = [pair for pair, key in zip(requested, keys) if key is None]
            raise RuntimeError(f'retained joint PWC candidate entry missing: {missing}')
        keys_by_name[target.name] = keys
        requested_by_name[target.name] = requested
    selected_keys = tuple(key for target in statistics_plan.targets
                          for key in keys_by_name[target.name])
    # File length is the sealed conservative storage bound. Archive validation
    # belongs to the existing PWC window immediately before its first read,
    # not an all-candidate header walk ahead of the PB read frontier.
    key_costs = {}
    for key in selected_keys:
        size = cache.estimate_nbytes([key])
        key_costs[key] = {'incoming_storage_bytes': size, 'serialized_bytes': size}
    targets = targets_from_statistics_plan(statistics_plan, keys_by_name, key_costs)
    return keys_by_name, requested_by_name, targets


def preflight_joint_operator_admission(names_by_layer, modules, formats_by_name, cache, *,
                                       policy, retained_budget=None, source_bytes=None):
    """Refuse an inadmissible operator-window plan before any capture work.

    Every input is declared now: the target roster, each matrix's geometry, the
    PWC's candidate file sizes and the sealed budget. Nothing here reads a
    captured activation, a cotangent or a probe, which is exactly why the
    refusals it raises do not belong after a boundary capture (#743).

    The decoder's weights are still the streamed meta skeleton at this point,
    so the statistics plan is built on meta twins of the real modules against
    the torch reference backend. ``_joint_projection_requirements`` groups on
    the resolved ``FormatSpec`` and the calibrated activation maximum and sizes
    statistics from ``numel``; it consults the backend only to refuse a device
    it was not prewarmed for. The roster it returns here is therefore the one
    the fused backend returns on the installed tensors, and the per-layer call
    re-derives it and compares through ``sealed_windows``.

    Returns the admitted windows per layer when a retained budget is in force,
    and ``None`` otherwise.
    """
    from . import format_registry as fr

    if policy is None:
        raise ValueError('joint operator admission requires an operator-window policy')
    policy = normalize_operator_windows(policy)
    if retained_budget is not None:
        if isinstance(retained_budget, dict):
            retained_budget = RetainedWindowBudget.from_dict(retained_budget)
        if not isinstance(retained_budget, RetainedWindowBudget):
            raise TypeError('retained joint admission requires a versioned retained budget')
        if type(source_bytes) is not int or source_bytes < 0:
            raise ValueError('retained joint admission requires a declared source byte cap')
    windows_by_layer = {}
    for layer, names in sorted(names_by_layer.items()):
        names = tuple(names)
        if not names:
            continue
        twins, specs = {}, {}
        for name in names:
            rows, columns = tuple(modules[name].weight.shape)
            twins[name] = torch.nn.Linear(columns, rows, bias=False,
                                          device='meta', dtype=torch.bfloat16)
            specs[name] = {fmt: fr.get_format(fmt) for fmt in formats_by_name[name]}
        # The same geometry bound ``observe_and_project_windows`` applies per
        # layer, applied to every layer before the first of them is captured.
        largest = max(4 * rows * columns for rows, columns in
                      (tuple(module.weight.shape) for module in twins.values()))
        if largest > min(policy['max_candidate_bytes'], policy['workspace_reserve_bytes']):
            raise RuntimeError('joint single target exceeds candidate or matrix workspace budget')
        # Both replay paths need every candidate to have a PWC entry, and both
        # used to find out per layer: the retained one through
        # ``retained_admission_targets`` and the windowed one when
        # ``resident_candidates`` planned its first quantum.
        missing = [(name, fmt) for name in names for fmt in formats_by_name[name]
                   if cache.resolve_key(name, fmt) is None]
        if missing:
            raise RuntimeError('joint operator-window PWC candidate entry missing: '
                               f'{len(missing)} of {sum(len(formats_by_name[n]) for n in names)}, '
                               f'first {missing[:8]}')
        if retained_budget is None:
            continue
        statistics_plan = plan_joint_statistics_target_windows(
            twins, specs, max_statistics_bytes=retained_budget.statistics_cap_bytes,
            activation_max_abs=joint_activation_maxima(cache), projection_backend=None)
        _, _, targets = retained_admission_targets(statistics_plan, specs, cache)
        plan = plan_retained_targets(targets, budget=retained_budget,
                                     source_bytes=source_bytes,
                                     footprint_scope='pwc_serialized_upper_bound')
        windows_by_layer[layer] = tuple(
            PreflightRetainedWindow(window.names, window.statistics_bytes,
                                    window.render_bytes, window.candidate_count)
            for window in plan.windows)
    return None if retained_budget is None else windows_by_layer


def observe_and_project_retained_windows(
        modules, specs, cache, policy, *, retained_budget, n_probes,
        source_bytes, backward, record_operator, consume_probe,
        collect_col_energy, backend, guard=None, source_fingerprints=None,
        completed_names=(), sealed_windows=None, before_window=None, after_window=None,
        spill=None, render_identities=False, render_stream=None, render_cache=None):
    """Replay all probes inside each admitted target's retained PWC lifetime.

    The selected-key-only PWC preflight and scalar target planner run before
    any cache load. Every target window owns one PWC context; each probe owns a
    new statistics lease. ``consume_probe`` receives scalar signed components
    and compact diagnostics only after that lease has released its matrices.
    Callbacks must not retain borrowed source/render tensor references.

    ``spill`` selects the one-pass replay (PQ #994): an object whose
    ``capture(probe_index)`` runs the probe's single forward/backward and
    whose ``replay(window_index=, probe_index=, lease=)`` feeds a lease from
    it. Captures run where the sealed spill order puts them, after window
    zero's ``before_window`` and before any later window's
    (``joint_layer_quanta.quantum_executable_phase_names``), and before a
    probe's lease exists, so no statistics hook fires during them; every
    active window is then fed from the spill. When window zero is active,
    each probe's capture runs inside its retained lifetime, just before that
    probe's replay of it. When a resume has already committed window zero,
    every probe is captured right after window zero's ``before_window``,
    with no retained window open: a capture reads source weights and
    boundaries, never renders. It would otherwise run after the first active
    window k's ``before_window``, under a progress phase that prices no
    capture, which the row cannot leave (PQ #1172).
    ``backward`` is not called for an active window in this mode.

    ``render_identities`` asks the PWC loader threads to hash each render as
    they load it (PQ #1192), for a ``record_operator`` that reads the hash
    through ``cache.resident_render_identity`` instead of hashing the render
    itself on every probe.

    ``render_stream`` is an ``io_engine.ReadStream`` over
    :func:`retained_window_keys`'s windows, one group per window index, built
    with ``cache.retained_read_entries`` (PQ #1291). Each retained window then
    takes its renders from the stream, which reads the next window while this
    one computes, and its guard is charged only the renders still unread. The
    stream's groups must equal the windows opened here, or the replay refuses
    before any window opens. Without it each window reads its own renders when
    it opens.

    ``record_operator`` runs twice per window and candidate (PQ #1348): on
    the window's first probe, which records it, and after its last probe,
    which compares the resident render with the record once more before
    ``after_window`` commits the window. A probe between them reads the
    render the first probe recorded, from the same resident PWC tensor.
    ``render_cache`` (a :class:`RetainedRenderDeviceCache`) keeps the
    window's renders on the device across its probes; without it every
    probe copies each render from the host.

    ``source_bytes`` is the caller's declared, separately checked source-owner
    cap. Passing a varying per-layer observation here would change the sealed
    target-window roster. This utility does not check the pre-capture physical
    baseline or manage source, boundary and auxiliary owners.
    """
    policy = normalize_operator_windows(policy)
    if policy is None:
        raise ValueError('retained joint replay requires an operator-window policy')
    if isinstance(retained_budget, dict):
        retained_budget = RetainedWindowBudget.from_dict(retained_budget)
    if not isinstance(retained_budget, RetainedWindowBudget):
        raise TypeError('retained joint replay requires a versioned retained budget')
    if type(n_probes) is not int or n_probes <= 0:
        raise ValueError('retained joint replay requires a positive probe count')
    if type(source_bytes) is not int or source_bytes < 0:
        raise ValueError('retained joint replay requires a declared source byte cap')
    if any(not callable(callback) for callback in (backward, record_operator, consume_probe)):
        raise TypeError('retained joint replay callbacks must be callable')
    if spill is not None and not all(callable(getattr(spill, attr, None))
                                     for attr in ('capture', 'replay')):
        raise TypeError('retained joint spill replay needs capture and replay callables')
    if type(collect_col_energy) is not bool:
        raise ValueError('retained joint replay column-energy flag must be boolean')
    if type(render_identities) is not bool:
        raise ValueError('retained joint replay render-identity flag must be boolean')
    for policy_key, budget_value in (
            ('max_statistics_bytes', retained_budget.statistics_cap_bytes),
            ('max_candidate_bytes', retained_budget.candidate_delta_bytes),
            ('max_load_buffer_bytes', retained_budget.load_buffer_bytes)):
        if policy[policy_key] < budget_value:
            raise RuntimeError(f'retained joint {policy_key} is narrower than its sealed budget')

    statistics_plan = plan_joint_statistics_target_windows(
        modules, specs, max_statistics_bytes=retained_budget.statistics_cap_bytes,
        activation_max_abs=joint_activation_maxima(cache), projection_backend=backend)
    if source_fingerprints is None:
        source_fingerprints = {
            name: JointOperatorStatisticsLease._source_fingerprint(module.weight)
            for name, module in modules.items()}
    if set(source_fingerprints) != set(modules):
        raise RuntimeError('joint source seal coverage differs')

    def require_sources():
        if any(JointOperatorStatisticsLease._source_fingerprint(modules[name].weight)
               != fingerprint for name, fingerprint in source_fingerprints.items()):
            raise RuntimeError('joint source changed between retained target windows')

    # Existing PWC tensors, even unrelated ones, would make selected-file
    # prices understate this otherwise empty baseline. The owner scans live
    # tensor keys only after its one-time in-memory index build; no file walk.
    if cache._window_resident_storages():
        raise RuntimeError('retained joint replay requires an empty PWC resident baseline')

    keys_by_name, requested_by_name, targets = retained_admission_targets(
        statistics_plan, specs, cache)
    retained_plan = plan_retained_targets(
        targets, budget=retained_budget, source_bytes=source_bytes,
        footprint_scope='pwc_serialized_upper_bound')
    plan_receipt = retained_plan.as_dict()
    require_sources()

    completed_names = set(completed_names)
    if not completed_names <= set(modules):
        raise RuntimeError('retained completed target roster differs')
    if sealed_windows is not None:
        if len(sealed_windows) != len(retained_plan.windows):
            raise RuntimeError('sealed retained window count differs from runtime geometry/files')
        for sealed, actual in zip(sealed_windows, retained_plan.windows):
            if (tuple(sealed.original_full_target_names) != actual.names or
                    sealed.statistics_bytes != actual.statistics_bytes or
                    sealed.render_file_upper_bound_bytes != actual.render_bytes or
                    sealed.candidate_count != actual.candidate_count):
                raise RuntimeError('sealed retained window membership or footprint differs')
    active_indices = [i for i, window in enumerate(retained_plan.windows)
                      if set(window.names) - completed_names]
    if render_stream is not None:
        for window_index, keys in retained_window_keys(
                [window.names for window in retained_plan.windows], specs, cache,
                completed_names=completed_names):
            if render_stream.group_keys(window_index) != tuple(
                    key for key in cache._window_keys(keys)
                    if not isinstance(cache.weights[key], torch.Tensor)):
                raise RuntimeError(
                    f'retained joint read stream differs from window {window_index}')
    first_active = active_indices[0] if active_indices else None
    last_active = active_indices[-1] if active_indices else None
    # Window zero's slot holds the captures (PQ #1172). A resume that has
    # committed window zero captures there too, outside any retained window.
    capture_ahead = spill is not None and first_active is not None and first_active > 0
    candidate_receipts = []
    for window_index, window in enumerate(retained_plan.windows):
        if before_window is not None:
            before_window(window_index, window.names)
        require_sources()
        if capture_ahead and window_index == 0:
            for probe_index in range(n_probes):
                spill.capture(probe_index)
                require_sources()
        names = tuple(name for name in window.names if name not in completed_names)
        if not names:
            continue
        selected = {name: modules[name] for name in names}
        selected_specs = {name: specs[name] for name in names}
        keys = tuple(key for name in names for key in keys_by_name[name])
        requested = tuple(pair for name in names for pair in requested_by_name[name])

        def before_load_quantum(state):
            if guard is None:
                return
            # Each reservation on the side it lands (PQ #1291): the incoming
            # renders, their serialized buffers, the read pages and the
            # boundaries on the host; the statistics, the workspace and the
            # candidate delta on the device. A guard without a device envelope
            # charges the sum, as before.
            check_operator_allocation(
                guard, 'before_joint_retained_candidate_load',
                reserve_bytes=(state['remaining_incoming_storage_bytes']
                               + retained_budget.boundary_reserve_bytes
                               + retained_budget.load_buffer_bytes
                               + retained_budget.read_page_reserve_bytes),
                reserve_device_bytes=(window.statistics_bytes
                                      + retained_budget.workspace_reserve_bytes
                                      + retained_budget.candidate_delta_bytes))

        with cache.retained_window(
                keys, max_resident_bytes=retained_budget.retained_render_cap_bytes,
                max_workers=policy['prefetch_workers'],
                max_load_buffer_bytes=retained_budget.load_buffer_bytes,
                release_file_pages=True,
                before_load_quantum=before_load_quantum if guard is not None else None,
                render_identities=render_identities,
                stream=render_stream, stream_group=window_index,
                ) as candidate_receipt:
            try:
                for probe_index in range(n_probes):
                    require_sources()
                    if spill is not None and window_index == first_active == 0:
                        spill.capture(probe_index)
                        require_sources()
                    if guard is not None:
                        check_operator_allocation(
                            guard, 'before_joint_retained_statistics_probe',
                            reserve_bytes=retained_budget.boundary_reserve_bytes,
                            reserve_device_bytes=(window.statistics_bytes
                                                  + retained_budget.workspace_reserve_bytes
                                                  + retained_budget.candidate_delta_bytes))
                    with JointOperatorStatisticsLease(
                            selected, selected_specs,
                            max_statistics_bytes=retained_budget.statistics_cap_bytes,
                            max_candidate_bytes=retained_budget.candidate_delta_bytes,
                            activation_max_abs=joint_activation_maxima(cache),
                            projection_backend=backend) as lease:
                        lease.begin_probe()
                        if spill is None:
                            backward(probe_index=probe_index,
                                     final=window_index == last_active,
                                     lease=lease)
                        else:
                            spill.replay(window_index=window_index,
                                         probe_index=probe_index, lease=lease)
                        require_sources()
                        lease.finish_observations()
                        diagnostics = lease.operator_diagnostics(
                            collect_col_energy=collect_col_energy)
                        keep = probe_index + 1 < n_probes
                        for name, fmt in requested:
                            rendered = source = delta = None
                            try:
                                source = modules[name].weight.detach()
                                if probe_index == 0:
                                    rendered = cache.get_resident(name, fmt)
                                    record_operator(name, fmt, source, rendered)
                                require_sources()
                                if render_cache is None:
                                    if rendered is None:
                                        rendered = cache.get_resident(name, fmt)
                                    delta = rendered.to(
                                        device=source.device, dtype=torch.float32, copy=True)
                                    delta.sub_(source)
                                else:
                                    delta = render_cache.delta(
                                        (name, fmt),
                                        (rendered if rendered is not None else
                                         partial(cache.get_resident, name, fmt)),
                                        source, keep=keep)
                                lease.project({(name, fmt): delta})
                            finally:
                                rendered = source = delta = None
                        terms = lease.finish_projections()
                    probe_receipt = {
                        'plan': {'schema': 'prismaquant.joint_retained_target_plan.v1',
                                 'window_count': len(retained_plan.windows),
                                 'footprint_scope': retained_plan.footprint_scope},
                        'window_index': window_index,
                        'window_names': names,
                        'candidate_window': dict(candidate_receipt),
                    }
                    consume_probe(probe_index, terms, diagnostics, probe_receipt)
                    require_sources()
            finally:
                # Every path out of the window drops its kept renders.
                if render_cache is not None:
                    render_cache.clear()
            # The window's close: every candidate's resident render is
            # compared with its record once more before the window commits.
            for name, fmt in requested:
                record_operator(name, fmt, modules[name].weight.detach(),
                                cache.get_resident(name, fmt))
            require_sources()
            candidate_receipts.append(dict(candidate_receipt))
        if after_window is not None:
            after_window(window_index, names)
    if not active_indices:
        for probe_index in range(n_probes):
            backward(probe_index=probe_index, final=True, lease=None)
    return {'plan': plan_receipt, 'candidate_windows': candidate_receipts}
