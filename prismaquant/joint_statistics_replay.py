"""Explicit bounded target replay over existing source and PWC owners."""
from __future__ import annotations

from contextlib import contextmanager
import os

import torch

from .joint_aura import JointOperatorStatisticsLease, arithmetic_identity
from .joint_statistics_plan import plan_joint_statistics_target_windows

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


def statistics_arithmetic_identity(dtype, backend):
    result = arithmetic_identity(dtype, backend)
    result.update(weight_projection='summed_output_operator_fp32_gemm',
        operator_accumulation='sum_fp32_matrices_in_backward_invocation_order',
        contraction_order='sum_operators_then_project_each_signed_component')
    return result


def operator_window_guard(device):
    if torch.device(device).type != 'cuda':
        return None
    from .autoscale import require_bounded_capture_environment
    from .memory_management import CaptureMemoryGuard
    require_bounded_capture_environment(os.environ)
    guard = CaptureMemoryGuard(device)
    guard.check('before_joint_operator_identity')
    return guard


@contextmanager
def resident_candidates(cache, keys, policy, *, guard=None):
    """Use PWC's finite windows; borrowed renders must not escape the yield."""
    # The serialized buffer cap can be smaller than the resident cap. Planning
    # conservatively against their minimum avoids admitting an oversized read.
    cap = min(policy['max_render_resident_bytes'], policy['max_load_buffer_bytes'])
    requested = {}
    for key in keys:
        requested.setdefault(cache.resolve_key(*key), []).append(key)
    windows = cache.plan_resident_windows(keys, max_resident_bytes=cap,
                                         max_workers=policy['prefetch_workers'])
    # This context supplies an iterator; each PWC context owns exactly one
    # quantum until the consumer advances or closes the iterator.
    def iterate():
        for keys in windows:
            if guard is not None:
                guard.check('before_joint_candidate_load', reserve_bytes=(
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
        activation_max_abs=cache.activation_max_abs, projection_backend=backend)
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
            guard.check('before_joint_statistics_window', reserve_bytes=(
                plan.window_statistics_bytes[index] + policy['workspace_reserve_bytes']
                + policy['max_replay_cotangent_bytes']))
        with JointOperatorStatisticsLease(selected, {name: specs[name] for name in names},
                max_statistics_bytes=policy['max_statistics_bytes'],
                max_candidate_bytes=policy['max_candidate_bytes'],
                activation_max_abs=cache.activation_max_abs, projection_backend=backend) as lease:
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
