"""Two planned-size operator leases on actual GLM census shapes.

Frozen independent Linear shape stand-ins and a live geometry remainder keep
the ledger's source footprint resident. They are not original GLM weights or
packed routing. Eight-row operands and controlled dW measure operator storage,
sampled FP64 contractions and lifetime only; no PWC, full-model or quality gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import threading
import time
import traceback

import torch
from torch import nn
from torch.multiprocessing.reductions import StorageWeakRef

from experiments.joint_allocator_turnover_profile import GIB, plan_geometry, profile_phase
from experiments.joint_operator_windows_profile import physical_snapshot, sha, tensor_identity, write_json
from experiments.workspace_netdata import NetdataWriter, sample_netdata
from prismaquant import format_registry as fr
from prismaquant.joint_aura import JointOperatorStatisticsLease
from prismaquant.joint_statistics_plan import plan_joint_statistics_target_windows
from prismaquant.perturbed_x_cache import _activation_qdq


FORMATS = ('FP8_E4M3', 'NVFP4')
ROWS = 8
SAMPLES = 64
ORACLE_RTOL, ORACLE_ATOL = 3e-5, 2e-9


def derive_plan(census, ledger):
    geometry = plan_geometry(ledger)
    shapes = {name: tuple(shape) for name, shape in census['unit_shapes'].items()
              if name.startswith('model.language_model.layers.3.')}
    if (len(shapes) != 867 or Counter(shapes.values()) != {(2048, 4096): 578, (4096, 2048): 289}):
        raise ValueError('planned native fixture requires exact layer-three census geometry')
    modules = {name: nn.Linear(shape[1], shape[0], bias=False, device='meta', dtype=torch.bfloat16)
               for name, shape in shapes.items()}
    specs = {name: {fmt: fr.get_format(fmt) for fmt in FORMATS} for name in shapes}
    plan = plan_joint_statistics_target_windows(modules, specs,
        max_statistics_bytes=geometry['retired_statistics_bytes'])
    if list(map(len, plan.windows)) != [341, 341, 185] or list(plan.window_statistics_bytes[:2]) != [34326183936] * 2:
        raise RuntimeError('actual registry groups do not match the bounded two-GA plan')
    for target in plan.targets:
        if len(target.groups) != 2 or any(json.loads(group.activation_identity_json)['input_global_scale'] is not None
                                          for group in target.groups):
            raise RuntimeError('shape fixture requires two original dynamic QDQ groups')
    shape_source_bytes = sum(math.prod(shape) * 2 for shape in shapes.values())
    geometry.update(shape_source_bytes=shape_source_bytes,
        source_remainder_bytes=geometry['source_bytes'] - shape_source_bytes,
        candidate_storage_bytes=32 * 1024**2,
        max_candidate_bytes=ledger['terms_bytes']['candidate_delta_cap'],
        replay_allowance_bytes=ledger['terms_bytes']['replay_fork_cap'])
    if geometry['source_remainder_bytes'] <= 0 or geometry['max_candidate_bytes'] != 256 * 1024**2:
        raise ValueError('source remainder or candidate allowance differs from the bounded ledger')
    if geometry['replay_allowance_bytes'] != 256 * 1024**2:
        raise ValueError('planned fixture requires the explicit replay allowance')
    return shapes, specs, plan, geometry


def sample_coordinates(shape):
    rows = [0, shape[0] - 1, *[(i * 127 + 13) % shape[0] for i in range(SAMPLES - 2)]]
    cols = [0, shape[1] - 1, *[(i * 251 + 7) % shape[1] for i in range(SAMPLES - 2)]]
    return rows, cols


def fp64_samples(x_columns, gradient_rows):
    return (gradient_rows.to(device='cpu', dtype=torch.float64)
            * x_columns.to(device='cpu', dtype=torch.float64)).sum(dim=0).tolist()


def check_samples(expected, observed):
    if len(expected) != SAMPLES or len(observed) != SAMPLES:
        raise RuntimeError('operator sample coverage differs')
    for reference, actual in zip(expected, observed):
        if not (math.isfinite(reference) and math.isfinite(actual)
                and math.isclose(reference, actual, rel_tol=ORACLE_RTOL, abs_tol=ORACLE_ATOL)):
            raise RuntimeError(f'planned operator FP64 sample differs: {reference} versus {actual}')
    if not any(value != 0 for value in observed):
        raise RuntimeError('planned operator sample is entirely zero')
    return dict(max_absolute=max(abs(a - b) for a, b in zip(expected, observed)),
                max_relative_above_atol=max((abs(a-b)/abs(a) for a, b in zip(expected, observed)
                                            if abs(a) > ORACLE_ATOL), default=0.))


def source_identity(modules, remainder):
    result = {}
    for name, module in modules.items():
        weight = module.weight
        rows, cols = sample_coordinates(weight.shape)
        values = weight[torch.tensor(rows, device=weight.device), torch.tensor(cols, device=weight.device)]
        result[name] = dict(data_ptr=weight.data_ptr(), version=weight._version,
            storage_bytes=weight.untyped_storage().nbytes(), shape=list(weight.shape),
            stride=list(weight.stride()), dtype=str(weight.dtype), samples=tensor_identity(values))
    positions = [i * (remainder.numel() - 1) // (SAMPLES - 1) for i in range(SAMPLES)]
    result['geometry_remainder'] = dict(data_ptr=remainder.data_ptr(), version=remainder._version,
        storage_bytes=remainder.untyped_storage().nbytes(),
        samples=tensor_identity(remainder[torch.tensor(positions, device=remainder.device)]))
    return result


def prepare_operands(selected, specs, seed):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    operands, expected = {}, {}
    for name, module in selected.items():
        weight = module.weight
        x = torch.randn((ROWS, weight.shape[1]), generator=generator, device='cuda', dtype=torch.bfloat16)
        x.mul_(0.125).requires_grad_()
        gradient = torch.randn((ROWS, weight.shape[0]), generator=generator, device='cuda', dtype=torch.bfloat16)
        gradient.mul_(0.0625)
        baseline = module(x)
        baseline.backward(gradient)
        identity = tensor_identity(x.grad)
        x.grad = None
        del baseline
        rows, cols = sample_coordinates(weight.shape)
        row_index, col_index = torch.tensor(rows, device='cuda'), torch.tensor(cols, device='cuda')
        g_columns = gradient[:, row_index].float()
        values = {None: fp64_samples(x[:, col_index].float(), g_columns)}
        with torch.no_grad():
            for group, fmt in enumerate(FORMATS):
                q = _activation_qdq(x.detach(), specs[name][fmt], None, name)
                dx = q.float() - x.detach().float()
                values[group] = fp64_samples(dx[:, col_index], g_columns)
        expected[name] = values
        operands[name] = (x, gradient, identity)
    return operands, expected


def sample_operators(lease, expected):
    records, owners = [], []
    keys = {(name, group) for name in expected for group in (None, 0, 1)}
    if set(lease._operators) != keys:
        raise RuntimeError('planned statistics matrix roster differs')
    for (name, group), matrix in lease._operators.items():
        if matrix.dtype != torch.float32 or matrix.untyped_storage().nbytes() != matrix.numel() * 4:
            raise RuntimeError('planned statistics does not own a complete FP32 matrix backing')
        owners.append(StorageWeakRef(matrix.untyped_storage()))
        rows, cols = sample_coordinates(matrix.shape)
        values = matrix[torch.tensor(rows, device='cuda'), torch.tensor(cols, device='cuda')].cpu().tolist()
        reference = expected[name][group]
        records.append(dict(name=name, group=group, storage_bytes=matrix.untyped_storage().nbytes(),
            expected_fp64=reference, observed_fp32=values, error=check_samples(reference, values)))
    if len({owner.cdata for owner in owners}) != len(owners):
        raise RuntimeError('planned statistics matrices alias a backing storage')
    return records, owners


def project_controlled_candidates(lease, selected, delta_flat):
    for name, module in selected.items():
        delta = delta_flat.view(module.weight.shape)
        for fmt in FORMATS:
            lease.project({(name, fmt): delta})
    return lease.finish_projections()


def run_window(index, names, modules, specs, geometry, guard, delta_flat):
    from prismaquant.joint_statistics_replay import check_operator_allocation
    selected = {name: modules[name] for name in names}
    record = dict(index=index, names=list(names), started_unix=time.time(),
                  before=physical_snapshot(torch.device('cuda')))
    reserve = 34326183936 + geometry['future_workspace_bytes'] + geometry['replay_allowance_bytes']
    record['admission'] = check_operator_allocation(guard, 'before_planned_statistics_window', reserve_bytes=reserve)
    record['after_admission'] = physical_snapshot(torch.device('cuda'))
    operands, expected = prepare_operands(selected, specs, 8100 + index)
    with JointOperatorStatisticsLease(selected, {name: specs[name] for name in names},
            max_statistics_bytes=geometry['retired_statistics_bytes'],
            max_candidate_bytes=geometry['max_candidate_bytes']) as lease:
        lease.begin_probe()
        outputs = [selected[name](operands[name][0]) for name in names]
        record['backward_admission'] = check_operator_allocation(guard, 'before_planned_statistics_backward',
            reserve_bytes=lease.statistics_capacity_bytes + geometry['future_workspace_bytes'])
        torch.autograd.backward(outputs, [operands[name][1] for name in names])
        torch.cuda.synchronize()
        if lease.resident_statistics_bytes != 34326183936:
            raise RuntimeError('actual retained matrix bytes differ from planned full window')
        record['all_statistics_live'] = physical_snapshot(torch.device('cuda'))
        record['statistics_bytes'] = lease.resident_statistics_bytes
        for name in names:
            if tensor_identity(operands[name][0].grad) != operands[name][2]:
                raise RuntimeError('statistics observation changed the outgoing cotangent bytes')
        record['exact_cotangent_count'] = len(names)
        record['matrix_samples'], owners = sample_operators(lease, expected)
        lease.finish_observations()
        diagnostics = lease.operator_diagnostics(collect_col_energy=True)
        record['diagnostics'] = {name: dict(g_trace=row['g_trace'], col_energy=tensor_identity(row['col_energy']))
                                 for name, row in diagnostics.items()}
        del diagnostics, outputs, operands, expected
        record['projection_admission'] = check_operator_allocation(guard, 'before_controlled_candidate_projection',
            reserve_bytes=geometry['future_workspace_bytes'])
        results = project_controlled_candidates(lease, selected, delta_flat)
        if any(not owner.expired() for owner in owners):
            raise RuntimeError('finished planned statistics retained a matrix backing owner')
        record['expired_statistics_backings'] = len(owners)
        record['projected'] = [dict(name=name, format=fmt, **value) for (name, fmt), value in results.items()]
        record['nonzero_projected_terms'] = {term: sum(value[term] != 0 for value in results.values())
                                           for term in ('weight', 'activation', 'mixed')}
        if len(results) != 682 or not all(record['nonzero_projected_terms'].values()):
            raise RuntimeError('planned candidate projection coverage is incomplete or degenerate')
        record['lease_telemetry'] = dict(lease.telemetry)
    if any(module.weight.requires_grad or module.weight.grad is not None for module in modules.values()):
        raise RuntimeError('shape source acquired a gradient plane')
    record.update(after=physical_snapshot(torch.device('cuda')), finished_unix=time.time(), status='complete')
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('census', 'ledger'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    for name in ('census', 'ledger'):
        if sha(getattr(args, name)) != getattr(args, name + '_sha256'):
            parser.error(name + ' differs from its frozen digest')
    shapes, specs, plan, geometry = derive_plan(json.loads(args.census.read_text()), json.loads(args.ledger.read_text()))
    args.out.mkdir(parents=True, exist_ok=False)
    record = dict(schema='prismaquant.joint_statistics_scale_profile.v1', status='running', scope=__doc__,
        geometry=geometry, plan=plan.as_dict(), rows=ROWS, formats=list(FORMATS),
        oracle_tolerance=dict(relative=ORACLE_RTOL, absolute=ORACLE_ATOL,
            rationale='FP32 sums of eight products; independent FP64 sampled contraction of the exact BF16/FP32 operands; no full matrix oracle'),
        source_inputs={name: dict(path=str(getattr(args, name)), sha256=getattr(args, name + '_sha256'))
                       for name in ('census', 'ledger')})
    write_json(args.out / 'plan.json', record)
    if args.plan_only:
        print(json.dumps(dict(status='CPU_PLAN_ONLY', targets=len(shapes), windows=list(plan.window_statistics_bytes))))
        return 0
    if not torch.cuda.is_available():
        parser.error('planned statistics execution requires admitted native CUDA')
    from prismaquant.autoscale import require_bounded_capture_environment
    from prismaquant.memory_management import CaptureMemoryGuard
    require_bounded_capture_environment(os.environ)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    guard = CaptureMemoryGuard('cuda')
    if guard.cap_bytes != geometry['physical_admission_bytes']:
        raise RuntimeError('actual cgroup differs from declared104GiB physical admission')
    record.update(started_unix=time.time(), torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0), affinity=sorted(os.sched_getaffinity(0)), windows=[], telemetry_errors=[])
    stop = threading.Event()
    def monitor():
        try:
            with (args.out / 'netdata.jsonl').open('x') as stream:
                writer = NetdataWriter(stream)
                while not stop.is_set():
                    for host in ('sparky', 'sparklina'):
                        writer.write(sample_netdata(host))
                    stop.wait(1.)
        except BaseException as error:
            record['telemetry_errors'].append(repr(error))
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    modules, remainder, delta_flat, owned_sources = {}, None, None, []
    try:
        torch.cuda.empty_cache()
        record['source_admission'] = guard.check('before_shape_source_allocation',
            reserve_bytes=geometry['source_bytes'] + geometry['candidate_storage_bytes'] + geometry['future_workspace_bytes'])
        torch.manual_seed(20260908)
        modules = {name: nn.Linear(shape[1], shape[0], bias=False, device='cuda', dtype=torch.bfloat16).eval()
                   for name, shape in sorted(shapes.items())}
        for module in modules.values():
            module.weight.requires_grad_(False)
        remainder = torch.empty(geometry['source_remainder_bytes'], dtype=torch.uint8, device='cuda').fill_(17)
        delta_flat = torch.empty(8388608, dtype=torch.float32, device='cuda').fill_(1 / 512)
        owned_sources = [StorageWeakRef(module.weight.untyped_storage()) for module in modules.values()]
        owned_sources.extend((StorageWeakRef(remainder.untyped_storage()), StorageWeakRef(delta_flat.untyped_storage())))
        record['source_identity'] = source_identity(modules, remainder)
        actual_source = sum(module.weight.untyped_storage().nbytes() for module in modules.values()) + remainder.untyped_storage().nbytes()
        if actual_source != geometry['source_bytes']:
            raise RuntimeError('actual complete source storage differs from ledger geometry')
        record['source_ready'] = physical_snapshot(torch.device('cuda'))
        for index, names in enumerate(plan.windows[:2]):
            value, trace = profile_phase(args.out, f'window_{index}',
                lambda: run_window(index, names, modules, specs, geometry, guard, delta_flat))
            value['profile'] = trace
            record['windows'].append(value)
            if source_identity(modules, remainder) != record['source_identity']:
                raise RuntimeError('planned source storage or sampled values changed across a window')
            if value['all_statistics_live']['cuda_reserved_bytes'] > geometry['gpu_subset_admission_bytes']:
                raise RuntimeError('actual statistics reservation exceeded92GiB GPU subset')
            write_json(args.out / 'progress.json', record)
        for name in ('census', 'ledger'):
            if sha(getattr(args, name)) != getattr(args, name + '_sha256'):
                raise RuntimeError('frozen source geometry input changed during native gate')
        record['status'] = 'complete'
    except BaseException:
        record.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        modules.clear()
        module = remainder = delta_flat = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        record['after_source_release'] = physical_snapshot(torch.device('cuda'))
        record['final_owned_backings'] = len(owned_sources)
        record['final_owned_backings_expired'] = sum(owner.expired() for owner in owned_sources)
        if any(not owner.expired() for owner in owned_sources):
            record['status'] = 'failed'
            record['cleanup_error'] = 'a source/remainder/candidate backing owner survived cleanup'
        stop.set()
        thread.join(timeout=15)
        if thread.is_alive():
            record['telemetry_errors'].append('Netdata monitor did not stop')
        if record['telemetry_errors']:
            record['status'] = 'failed'
        record['finished_unix'] = time.time()
        write_json(args.out / 'result.json', record)
        print(json.dumps(dict(status=record['status'], path=str(args.out / 'result.json'),
                              sha256=sha(args.out / 'result.json'))), flush=True)
    if record['status'] != 'complete':
        raise RuntimeError('planned statistics gate did not complete and release its CUDA owners')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
