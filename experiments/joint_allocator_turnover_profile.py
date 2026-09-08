"""Planned-size CUDA allocator turnover proxy, without model or fit claims.

The source proxy takes the exact settled-source byte count from the hashed
planning ledger. Old statistics are real allocated/touched CUDA bytes. The
next statistics and workspace are future reservations, not live allocations.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time
import traceback

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from experiments.joint_operator_windows_profile import physical_snapshot, sha, write_json
from experiments.workspace_netdata import NetdataWriter, sample_netdata


GIB = 1024**3


def plan_geometry(ledger):
    if ledger.get('schema') != 'prismaquant.glm_joint_resource_ledger.v2':
        raise ValueError('allocator proxy requires the versioned planning ledger')
    terms = ledger['terms_bytes']
    result = dict(source_bytes=terms['source_settled'],
        retired_statistics_bytes=terms['statistics_cap'],
        future_statistics_bytes=terms['statistics_cap'],
        future_workspace_bytes=terms['workspace_allowance'],
        physical_admission_bytes=ledger['physical_admission_bytes'],
        gpu_subset_admission_bytes=ledger['gpu_subset_admission_bytes'])
    if any(type(value) is not int or value <= 0 for value in result.values()):
        raise ValueError('allocator proxy byte counts must be positive integers')
    if (result['physical_admission_bytes'] != 104 * GIB
            or result['gpu_subset_admission_bytes'] != 92 * GIB
            or result['retired_statistics_bytes'] != 32 * GIB
            or result['future_workspace_bytes'] != 16 * GIB):
        raise ValueError('allocator proxy requires the declared 104/92/32/16 GiB bounds')
    result['future_reserve_bytes'] = result['future_statistics_bytes'] + result['future_workspace_bytes']
    corrected = result['source_bytes'] + result['future_reserve_bytes']
    old = corrected + result['retired_statistics_bytes']
    if corrected > result['gpu_subset_admission_bytes'] or old <= result['physical_admission_bytes'] - 2 * GIB:
        raise ValueError('ledger geometry does not distinguish retired double charge within the admitted GPU subset')
    return result


def live_identity(tensor):
    # Only 64 scalar samples cross to CPU. Never allocate a full-sized bool
    # comparison or clone of the live source proxy.
    positions = [i * (tensor.numel() - 1) // 63 for i in range(64)]
    indices = torch.tensor(positions, dtype=torch.int64, device=tensor.device)
    samples = tensor.index_select(0, indices).to('cpu').tolist()
    return dict(data_ptr=tensor.data_ptr(), storage_bytes=tensor.untyped_storage().nbytes(),
                shape=list(tensor.shape), dtype=str(tensor.dtype), positions=positions, samples=samples)


def profile_phase(out, name, body):
    path = out / f'{name}.trace.json'
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                 torch.profiler.ProfilerActivity.CUDA],
                                      record_shapes=False, profile_memory=False)
    try:
        with profiler:
            with torch.profiler.record_function(name):
                value = body()
    finally:
        profiler.export_chrome_trace(str(path))
    return value, dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                       record_shapes=False, profile_memory=False)


def run_proxy(plan, out, record):
    from prismaquant.autoscale import require_bounded_capture_environment
    from prismaquant.joint_statistics_replay import check_operator_allocation
    from prismaquant.memory_management import CaptureMemoryGuard
    require_bounded_capture_environment(os.environ)
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    admission = CaptureMemoryGuard(device)
    if admission.cap_bytes != plan['physical_admission_bytes']:
        raise RuntimeError('actual finite cgroup differs from declared physical admission')
    record['admission'] = admission.check('before_live_and_retired_proxy_allocation',
        reserve_bytes=plan['source_bytes'] + plan['retired_statistics_bytes'])
    record['before_allocation'] = physical_snapshot(device)
    live = retired = None
    try:
        def before():
            nonlocal live, retired
            live = torch.empty(plan['source_bytes'], dtype=torch.uint8, device=device)
            live.fill_(17)
            retired = torch.empty(plan['retired_statistics_bytes'], dtype=torch.uint8, device=device)
            retired.fill_(29)
            torch.cuda.synchronize(device)
            owner = StorageWeakRef(retired.untyped_storage())
            record['both_allocations_live'] = physical_snapshot(device)
            record['live_identity_before'] = live_identity(live)
            if record['both_allocations_live']['cuda_reserved_bytes'] > plan['gpu_subset_admission_bytes']:
                raise RuntimeError('actual proxy CUDA reservation exceeded declared GPU subset')
            retired = None
            if not owner.expired():
                raise RuntimeError('statistics proxy retained a real backing owner')
            record['retired_owner_expired'] = True
            record['before_old_guard'] = physical_snapshot(device)
            if (record['before_old_guard']['cuda_reserved_bytes']
                    - record['before_old_guard']['cuda_allocated_bytes'] < plan['retired_statistics_bytes']):
                raise RuntimeError('actual allocator did not retain the required retired reservation')
            old = CaptureMemoryGuard(device)
            try:
                old.check('old_guard_with_inactive_statistics', reserve_bytes=plan['future_reserve_bytes'])
            except RuntimeError as error:
                if not str(error).startswith('capture physical memory refusal:'):
                    raise
                record['old_guard_refusal'] = dict(error=str(error), snapshot=old.snapshot())
            else:
                raise RuntimeError('old guard did not refuse the native inactive-plus-future double charge')
            return old
        old, record['before_profile'] = profile_phase(out, 'before_allocator_retirement', before)
        write_json(out / 'before-checked.json', record)

        def after():
            fresh = CaptureMemoryGuard(device)
            record['before_corrected_guard'] = physical_snapshot(device)
            record['corrected_guard'] = check_operator_allocation(fresh,
                'corrected_guard_after_retiring_statistics', reserve_bytes=plan['future_reserve_bytes'])
            record['after_corrected_guard'] = physical_snapshot(device)
            record['live_identity_after'] = live_identity(live)
            if record['live_identity_before'] != record['live_identity_after']:
                raise RuntimeError('allocator retirement changed the live source proxy')
            if set(record['live_identity_after']['samples']) != {17}:
                raise RuntimeError('live source proxy sampled contents are wrong')
            released = (record['before_corrected_guard']['cuda_reserved_bytes']
                        - record['after_corrected_guard']['cuda_reserved_bytes'])
            if released < plan['retired_statistics_bytes']:
                raise RuntimeError('corrected helper did not release the planned retired reservation')
            record['released_reserved_bytes'] = released
            if (record['before_corrected_guard']['cuda_allocated_bytes']
                    != record['after_corrected_guard']['cuda_allocated_bytes']):
                raise RuntimeError('corrected helper changed actual live allocated bytes')
            # Refusal is sticky on the original guard; corrected admission
            # must use the fresh guard above, not clear its failure state.
            try:
                old.check('old_guard_remains_failed', reserve_bytes=plan['future_reserve_bytes'])
            except RuntimeError as error:
                if str(error) != record['old_guard_refusal']['error']:
                    raise
                record['old_guard_failure_sticky'] = True
            else:
                raise RuntimeError('old guard unexpectedly recovered after refusal')
        _, record['after_profile'] = profile_phase(out, 'after_allocator_retirement', after)
    finally:
        live = retired = None
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        record['after_all_owner_release'] = physical_snapshot(device)
    if record['after_all_owner_release']['cuda_allocated_bytes'] != record['before_allocation']['cuda_allocated_bytes']:
        raise RuntimeError('proxy action retained a CUDA tensor after cleanup')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--ledger-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if sha(args.ledger) != args.ledger_sha256:
        parser.error('planning ledger differs from its frozen digest')
    ledger = json.loads(args.ledger.read_text())
    plan = plan_geometry(ledger)
    if not torch.cuda.is_available():
        parser.error('planned-size allocator proxy requires admitted native CUDA')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    record = dict(schema='prismaquant.joint_allocator_turnover_proxy.v1', status='running',
        scope=__doc__, started_unix=time.time(), plan=plan,
        ledger=dict(path=str(args.ledger), sha256=args.ledger_sha256, status=ledger.get('status')),
        torch=str(torch.__version__), cuda=torch.version.cuda, gpu_name=torch.cuda.get_device_name(0),
        affinity=sorted(os.sched_getaffinity(0)), telemetry_errors=[])
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
    try:
        run_proxy(plan, args.out, record)
        if sha(args.ledger) != args.ledger_sha256:
            raise RuntimeError('planning ledger changed during the proxy gate')
        record['status'] = 'complete'
    except BaseException:
        record.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
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
        raise RuntimeError('native allocator proxy evidence is incomplete')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
