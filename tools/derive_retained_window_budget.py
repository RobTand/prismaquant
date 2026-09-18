#!/usr/bin/env python3
"""Compose a joint plan's retained COST budget from the roster it must admit.

Issue #743. The demand-driven caps of
``execution.retained_operator_windows.budget`` are maxima over bytes the
campaign already states: the geometry of each target matrix and the size of
each prepared candidate file. Authoring them by hand produced a
``candidate_delta_bytes`` of 4 MiB against a smallest demand of 32 MiB, and the
run spent 137.4 minutes of boundary capture before finding out.

This tool reads the same three declared sources the run reads -- the plan, the
prepared completion and its ProductionWeightCache, and the campaign census --
derives the five caps with ``derive_retained_window_budget``, and writes a new
plan carrying the derived budget plus a top-level
``retained_window_budget_derivation`` record. It changes nothing else, opens no
render archive and needs no GPU.

The operator keeps every owner that belongs to the box rather than the roster
(``DECLARED_BUDGET_FIELDS``); those are copied from the input plan unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time

_LAYER = re.compile(r'^.*\.layers\.(\d+)(?:\.|$)')


def _sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _bound(path, digest, label):
    actual = _sha256(path)
    if digest is not None and actual != digest:
        raise SystemExit(f'{label}: bytes are not the bound {digest}')
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--prepared-sha256')
    parser.add_argument('--census', required=True, type=Path,
                        help='campaign census.json, for unit_shapes and max_abs')
    parser.add_argument('--out', required=True, type=Path,
                        help='the new plan JSON; never the input path')
    parser.add_argument('--host-cap-bytes', type=int,
                        help="the container's own memory cgroup cap, which bounds the "
                             'host-resident retained renders. Defaults to the plan\'s '
                             'physical_limit_bytes less max_gpu_bytes, the same split the '
                             'aggregate capture guard holds.')
    parser.add_argument('--render-size-cache', type=Path,
                        help='read/write the per-key candidate file sizes here, so a '
                             'rerun does not restat every prepared render')
    args = parser.parse_args(argv)

    import torch  # noqa: F401  (the format registry and meta modules need it)
    from prismaquant import format_registry as fr
    from prismaquant.joint_retained_window_plan import (
        DECLARED_BUDGET_FIELDS, RetainedWindowBudget, derive_retained_window_budget,
        normalize_retained_execution, targets_from_statistics_plan,
    )
    from prismaquant.joint_statistics_plan import plan_joint_statistics_target_windows

    if args.out.resolve() == args.plan.resolve():
        raise SystemExit('refusing to overwrite the input plan')
    plan_sha256 = _bound(args.plan, args.plan_sha256, 'plan')
    prepared_sha256 = _bound(args.prepared, args.prepared_sha256, 'prepared')
    plan = json.loads(args.plan.read_text())
    if plan.get('schema') != 'prismaquant.tessera_joint_aura.plan.v1':
        raise SystemExit('not a prismaquant.tessera_joint_aura.plan.v1 plan')
    execution = plan['execution']
    retained = execution.get('retained_operator_windows')
    if retained is None:
        raise SystemExit('plan declares no execution.retained_operator_windows to derive')
    sealed = RetainedWindowBudget.from_dict(retained['budget'])
    source_bytes = retained['source_reserve_bytes']

    prepared = json.loads(args.prepared.read_text())
    if prepared.get('status') != 'complete':
        raise SystemExit('prepared completion is not complete')
    census = json.loads(args.census.read_text())
    shapes, maxima = census['unit_shapes'], census['max_abs']

    # The measured roster is exactly what the run measures: every prepared
    # candidate menu with its zero-cost passthrough removed (aura_cost
    # ``_ZERO_COST_FORMATS``), grouped by the decoder layer that owns it.
    from prismaquant.aura_cost import _ZERO_COST_FORMATS
    formats_by_name, names_by_layer = {}, {}
    for name, menu in prepared['formats_by_qname'].items():
        measured = tuple(fmt for fmt in menu if fmt not in _ZERO_COST_FORMATS)
        if not measured:
            raise SystemExit(f'{name}: prepared menu has no measured candidate')
        match = _LAYER.match(name)
        if match is None:
            raise SystemExit(f'{name}: no decoder layer owns this target')
        formats_by_name[name] = measured
        names_by_layer.setdefault(int(match.group(1)), []).append(name)

    cache_path = Path(prepared['production_cache']['path'])
    _bound(cache_path, prepared['production_cache']['sha256'], 'production cache')
    cache = pickle.loads(cache_path.read_bytes())

    sizes, started = {}, time.time()
    if args.render_size_cache is not None and args.render_size_cache.is_file():
        sizes = {tuple(json.loads(key)): value for key, value
                 in json.loads(args.render_size_cache.read_text()).items()}
    stats_taken = 0
    for name, measured in formats_by_name.items():
        for fmt in measured:
            key = cache.resolve_key(name, fmt)
            if key is None:
                raise SystemExit(f'prepared PWC has no candidate entry for {name}@{fmt}')
            if key not in sizes:
                sizes[key] = cache.estimate_nbytes([key])
                stats_taken += 1
    measured_seconds = time.time() - started
    if args.render_size_cache is not None:
        args.render_size_cache.write_text(json.dumps(
            {json.dumps(list(key)): value for key, value in sizes.items()}))

    targets_by_layer = {}
    for layer, names in sorted(names_by_layer.items()):
        modules, specs, keys_by_name, costs = {}, {}, {}, {}
        for name in sorted(names):
            rows, columns = shapes[name]
            modules[name] = torch.nn.Linear(columns, rows, bias=False,
                                            device='meta', dtype=torch.bfloat16)
            specs[name] = {fmt: fr.get_format(fmt) for fmt in formats_by_name[name]}
            keys = tuple(cache.resolve_key(name, fmt) for fmt in formats_by_name[name])
            keys_by_name[name] = keys
            for key in keys:
                costs[key] = {'incoming_storage_bytes': sizes[key],
                              'serialized_bytes': sizes[key]}
        # ``statistics_bytes`` is a property of the matrix and its activation
        # groups, not of this cap, so the physical bound serves as a provisional
        # ceiling; the derived cap is checked against the same targets below.
        statistics = plan_joint_statistics_target_windows(
            modules, specs, max_statistics_bytes=sealed.physical_limit_bytes,
            activation_max_abs={name: maxima[name] for name in names})
        targets_by_layer[layer] = targets_from_statistics_plan(statistics, keys_by_name, costs)

    host_cap_bytes = args.host_cap_bytes
    if host_cap_bytes is None:
        host_cap_bytes = sealed.physical_limit_bytes - plan['max_gpu_bytes']
        if host_cap_bytes <= 0:
            raise SystemExit('plan states no host side; pass --host-cap-bytes')
    declared = {name: getattr(sealed, name) for name in DECLARED_BUDGET_FIELDS}
    budget, record = derive_retained_window_budget(
        targets_by_layer, declared=declared, source_bytes=source_bytes,
        prefetch_workers=execution['operator_windows']['prefetch_workers'],
        host_cap_bytes=host_cap_bytes,
        footprint_scope='pwc_serialized_upper_bound')
    record['derived_from'] = {
        'plan': {'path': str(args.plan.resolve()), 'sha256': plan_sha256},
        'prepared': {'path': str(args.prepared.resolve()), 'sha256': prepared_sha256},
        'census': {'path': str(args.census.resolve()), 'sha256': _sha256(args.census)},
        'production_cache_sha256': prepared['production_cache']['sha256'],
        'candidate_files_measured': len(sizes),
        'candidate_files_stated_now': stats_taken,
        'measure_seconds': round(measured_seconds, 3),
        'tool': 'tools/derive_retained_window_budget.py',
    }
    record['superseded_budget'] = sealed.as_dict()

    plan['execution']['retained_operator_windows'] = {
        **retained, 'budget': budget.as_dict()}
    plan['retained_window_budget_derivation'] = record
    # A plan whose own loader would refuse it is not a plan.
    normalize_retained_execution(plan['execution']['retained_operator_windows'],
                                 operator_windows=execution['operator_windows'],
                                 boundary_storage=execution['boundary_storage'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=1, sort_keys=True) + '\n')
    print(json.dumps({
        'out': str(args.out.resolve()), 'sha256': _sha256(args.out),
        'superseded_plan_sha256': plan_sha256,
        'changed': {name: [sealed.as_dict()[name], budget.as_dict()[name]]
                    for name in budget.as_dict()
                    if sealed.as_dict()[name] != budget.as_dict()[name]},
        'available_window_bytes': record['available_window_bytes'],
        'host_cap_bytes': host_cap_bytes,
        'host_render_bound_bytes': record['host_render_bound_bytes'],
        'windows_by_layer_max': max(record['windows_by_layer'].values()),
        'retained_window_replay_multiplier': record['retained_window_replay_multiplier'],
        'candidate_files_measured': len(sizes),
        'measure_seconds': round(measured_seconds, 3),
    }, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
