"""Paired synthetic lease qualification at GLM expert projection dimensions.

This measures source observation plus synthetic resident-delta construction and
contraction. It does not measure a ProductionWeightCache fill, calibration,
source streaming, full-model residency, KL, bpp or a serving path.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import threading
import time
import traceback

import torch

from experiments.workspace_netdata import NetdataWriter, sample_netdata
from prismaquant.format_registry import FormatSpec
from prismaquant.joint_aura import JointOperatorStatisticsLease, SignedJointProjectionLease


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def memory():
    torch.cuda.synchronize()
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved())


def invoke(kind, module, specs, draws):
    shape = tuple(module.weight.shape)
    plane_bytes = module.weight.numel() * 4
    kwargs = dict(activation_max_abs={})
    if kind == 'legacy':
        deltas = {('u', fmt): torch.full(shape, (index + 1) / 1024, device='cuda')
                  for index, fmt in enumerate(specs)}
        lease = SignedJointProjectionLease({'u': module}, {'u': specs}, deltas, **kwargs)
    else:
        lease = JointOperatorStatisticsLease({'u': module}, {'u': specs},
            max_statistics_bytes=2 * plane_bytes, max_candidate_bytes=plane_bytes, **kwargs)
    with lease:
        lease.begin_probe()
        for x, gradient in draws:
            output = module(x)
            output.backward(gradient)
            module.zero_grad(set_to_none=True)
            del output
        if kind == 'legacy':
            result = lease.finish_probe()
        else:
            lease.finish_observations()
            for index, fmt in enumerate(specs):
                delta = torch.full(shape, (index + 1) / 1024, device='cuda')
                lease.project({('u', fmt): delta})
                del delta
            result = lease.finish_projections()
    return {f'{name}@{fmt}': values for (name, fmt), values in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=10.)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 60:
        parser.error('--seconds must be between 1 and 60')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    record = dict(schema='prismaquant.joint_operator_statistics_profile.v1', status='running',
        started_unix=time.time(), torch=str(torch.__version__), cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(), affinity=sorted(os.sched_getaffinity(0)),
        candidates=8, draws=4, tokens_per_draw=32, seconds_per_arm=args.seconds,
        source_dtype='torch.bfloat16', delta_dtype='torch.float32',
        matmul_precision=torch.get_float32_matmul_precision(), allow_tf32=False,
        seed=1913, delta_rule='full((rows,cols),(candidate_index+1)/1024,float32)',
        qdq_rule='round(X*8)/8 in source dtype', shapes=[], telemetry_errors=[])
    stopped = threading.Event()

    def monitor():
        try:
            with (args.out/'netdata.jsonl').open('x') as stream:
                writer = NetdataWriter(stream)
                while not stopped.is_set():
                    for host in ('sparky', 'sparklina'):
                        writer.write(sample_netdata(host))
                    stopped.wait(1.)
        except Exception as exc:
            record['telemetry_errors'].append(repr(exc))

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        qdq = lambda x: torch.round(x * 8) / 8
        specs = {f'candidate{i}': FormatSpec(name=f'candidate{i}', weight_bits=8,
            group_size=0, scale_bits=0, scale_dtype_name='fp32', weight_element_dtype='int8',
            act_bits=8, activation_quantize_dequantize=qdq) for i in range(8)}
        for rows, columns in ((2048, 4096), (4096, 2048)):
            generator = torch.Generator().manual_seed(record['seed'])
            weight = (torch.randn(rows, columns, generator=generator) / columns**.5).bfloat16()
            draws = [(torch.randn(32, columns, generator=generator).bfloat16().cuda(),
                      torch.randn(32, rows, generator=generator).bfloat16().cuda()) for _ in range(4)]
            module = torch.nn.Linear(columns, rows, bias=False, device='cuda', dtype=torch.bfloat16)
            with torch.no_grad():
                module.weight.copy_(weight)
            del weight
            shape_row = dict(shape=[rows, columns], arms=[])
            record['shapes'].append(shape_row)
            baseline = None
            for index, kind in enumerate(('legacy', 'statistics', 'statistics', 'legacy')):
                warm = invoke(kind, module, specs, draws)
                if baseline is None:
                    baseline = warm
                gc.collect()
                torch.cuda.empty_cache()
                before = memory()
                torch.cuda.reset_peak_memory_stats()
                start = time.time()
                times = []
                wall_start = time.perf_counter()
                while time.perf_counter() - wall_start < args.seconds:
                    tick = time.perf_counter()
                    result = invoke(kind, module, specs, draws)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - tick) * 1000)
                elapsed = time.perf_counter() - wall_start
                end = time.time()
                after = memory()
                assert after['allocated'] == before['allocated'], 'lease retained CUDA allocations'
                errors = {key: {term: result[key][term] - baseline[key][term] for term in result[key]}
                          for key in result}
                # Native tests separately bind both GLM shapes to an independent
                # FP64 residual. This row records the old/new arithmetic delta.
                for key in result:
                    for term, value in result[key].items():
                        assert abs(value - baseline[key][term]) <= 1e-4 + abs(baseline[key][term]) * 1e-4
                with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True, profile_memory=True) as profile:
                    with torch.profiler.record_function(f'joint_{kind}_whole_lease'):
                        invoke(kind, module, specs, draws)
                    torch.cuda.synchronize()
                trace = args.out/f'{rows}x{columns}-{index}-{kind}.trace.json'
                profile.export_chrome_trace(str(trace))
                events = [dict(key=item.key, calls=item.count,
                    self_cpu_us=item.self_cpu_time_total,
                    self_device_us=item.self_device_time_total)
                    for item in profile.key_averages()]
                del profile
                gc.collect()
                shape_row['arms'].append(dict(kind=kind, index=index, start_unix=start, end_unix=end,
                    invocations=len(times), elapsed_seconds=elapsed,
                    median_ms=statistics.median(times), samples_ms=times,
                    before=before, after=after, after_profile=memory(),
                    signed_component_delta=errors, trace=dict(path=str(trace), sha256=digest(trace)),
                    events=sorted(events, key=lambda item: item['self_device_us'], reverse=True)[:30]))
                (args.out/'progress.json').write_text(json.dumps(record, indent=2)+'\n')
            del module, draws
            gc.collect()
            torch.cuda.empty_cache()
        record['status'] = 'complete'
    except BaseException:
        record.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        stopped.set()
        monitor_thread.join(timeout=15)
        if monitor_thread.is_alive():
            record['telemetry_errors'].append('Netdata sampler did not stop')
        if record['telemetry_errors']:
            record['status'] = 'failed'
        record['finished_unix'] = time.time()
        result_path = args.out/'result.json'
        result_path.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(dict(path=str(result_path), sha256=digest(result_path), status=record['status'])))
    if record['status'] != 'complete':
        raise RuntimeError('required joint operator qualification evidence is incomplete')


if __name__ == '__main__':
    main()
