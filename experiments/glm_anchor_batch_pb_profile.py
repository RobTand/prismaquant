"""Bounded PB-profiled batch comparison on the real selected campaign inputs.

No full cost table is produced. Existing campaign residency, Hessian preparation,
producer encoding, decoding and pricing run unchanged; only call grouping varies.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time


class BenchmarkComplete(BaseException):
    pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container-spec')
    parser.add_argument('--out', required=True)
    parser.add_argument('campaign', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.campaign[1:] if args.campaign[:1] == ['--'] else args.campaign
    if args.container_spec:
        # PB owns the output path. Explicitly expose that directory inside the
        # existing container adapter; daemon-created Docker children do not
        # inherit the launcher's environment or filesystem.
        from tools.tessera_campaign_container import main as container_main
        spec = json.loads(args.container_spec)
        profile = Path(os.environ['PRISMABUILD_PROFILE_TORCH_OUT'])
        spec['container']['mounts'].append(dict(source=str(profile.parent), target='/pb-profile'))
        spec['env']['PRISMABUILD_PROFILE_TORCH_OUT'] = '/pb-profile/' + profile.name
        return container_main(['--spec', json.dumps(spec), '--', 'python3', '-u', '-m',
            'experiments.glm_anchor_batch_pb_profile', '--out', args.out, '--', *command])

    import torch
    from prismaquant import tessera_campaign as campaign
    from experiments.glm_full_capture_profile import CaptureObserver
    from prismaquant.tessera_campaign import _wire_path
    if command[command.index('--anchor-batch-size') + 1] != '16':
        raise ValueError('comparison requires the selected planner to reserve batch 16')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='prismaquant.glm_anchor_batch_pb_profile.v1',
        status='running', scope='first compatible 16 experts at the first campaign rung',
        full_campaign_complete=False, arms=[], command=command,
        torch=torch.__version__, cuda=torch.version.cuda, started_unix=time.time())
    original = campaign._measure_anchor_batch
    original_scalar = campaign._measure_anchor

    def save():
        tmp = out/'result.json.tmp'
        tmp.write_text(json.dumps(result, indent=2) + '\n')
        tmp.replace(out/'result.json')

    def refuse_scalar(**kw):
        raise RuntimeError('the expected first compatible expert batch was not reached')

    def compare(**kw):
        names = list(kw['qnames'])
        if len(names) != 16 or not all('.experts.' in name for name in names):
            raise RuntimeError('expected 16 compatible routed experts')
        result.update(qnames=names, format_name=kw['format_name'],
            shapes=[list(w.shape) for w in kw['weights']])
        reference = None

        def arm(width, label, prof=None):
            nonlocal reference
            record = dict(label=label, batch_size=width, units=16, started_unix=time.time())
            result['arms'].append(record)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            outputs = []
            timer = None
            stop = threading.Event()
            # Sample the SAME warmed encode offset in each arm. The complete
            # arm is timed and sampled by the existing Python-stack observer.
            # Four 0.5-second CUDA windows bound profiler memory independently
            # of the roughly minute-long encode calls.
            if prof is not None:
                def window():
                    if not stop.wait(5):
                        record['trace_started_unix'] = time.time()
                        prof.toggle_collection_dynamic(True, [torch.profiler.ProfilerActivity.CUDA])
                        stop.wait(0.5)
                        prof.toggle_collection_dynamic(False, [torch.profiler.ProfilerActivity.CUDA])
                        record['trace_finished_unix'] = time.time()
                timer = threading.Thread(target=window, daemon=True)
                timer.start()
            start = time.perf_counter()
            try:
                for offset in range(0, 16, width):
                    subset = dict(kw)
                    for field in ('qnames', 'weights', 'activations'):
                        subset[field] = kw[field][offset:offset + width]
                    outputs.extend(original(**subset))
                torch.cuda.synchronize()
            finally:
                record.update(seconds=time.perf_counter()-start, finished_unix=time.time(),
                    allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                    reserved_peak_bytes=torch.cuda.max_memory_reserved())
                stop.set()
                if timer is not None:
                    timer.join()
            signatures = []
            for anchor in outputs:
                wire = _wire_path(kw['wire_dir'], anchor.qname, anchor.format_name)
                with wire.open('rb') as handle:
                    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
                signatures.append(dict(qname=anchor.qname, wire_sha256=digest,
                    wire_bytes=wire.stat().st_size, dloss=anchor.dloss,
                    hessian_applied=anchor.hessian_applied))
            if reference is None:
                reference = signatures
            if signatures != reference:
                record['signatures'] = signatures
                raise RuntimeError('batch width changed actual wire bytes or pricing scores')
            record.update(signatures=signatures, exact_parity=True,
                units_per_second=16 / record['seconds'])
            print(json.dumps(record), flush=True)
            save()

        # Warm both shapes. Compare ABBA with identical instrumentation on all
        # four measured arms; exports occur after all arm timers stop.
        arm(8, 'warm-b8')
        arm(16, 'warm-b16')
        # Re-enabling one profiler across long CUDA-graph replay intervals
        # corrupts later kernel durations in the real PB trace (2026-09-09).
        # Fresh contexts isolate CUPTI's correlation state. Retain every raw
        # trace; PB also files the first B16 trace as its primary profile.
        primary = Path(os.environ['PRISMABUILD_PROFILE_TORCH_OUT'])
        event_count = 0
        for index, width in enumerate((8, 16, 16, 8)):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False, profile_memory=False, with_stack=False) as prof:
                prof.toggle_collection_dynamic(False, [torch.profiler.ProfilerActivity.CUDA])
                arm(width, f'measured-{index}-b{width}', prof)
            trace_path = out/f'measured-{index}-b{width}.trace.json.gz'
            prof.export_chrome_trace(str(trace_path))
            with trace_path.open('rb') as handle:
                trace_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
            events = sum(1 for event in prof.events()
                if event.device_type == torch.autograd.DeviceType.CUDA)
            if events == 0:
                raise RuntimeError('observed arm has no CUDA activity')
            event_count += events
            result['arms'][-1]['trace'] = dict(path=str(trace_path),
                sha256=trace_sha,bytes=trace_path.stat().st_size,cuda_events=events)
            if index == 1:
                import shutil
                shutil.copyfile(trace_path, primary)
            save()
        result.update(profile_path=str(primary), profile_bytes=primary.stat().st_size,
            profile_scope='four independent half-second CUDA contexts at +5 seconds; full-arm wall and Python stacks',
            cuda_events=event_count)
        raise BenchmarkComplete()

    campaign._measure_anchor_batch = compare
    campaign._measure_anchor = refuse_scalar
    save()
    try:
        with CaptureObserver(out/'observer', profile_layers=()):
            try:
                campaign.main(command)
            except BenchmarkComplete:
                pass
            else:
                raise RuntimeError('campaign ended without the bounded comparison')
        result['status'] = 'complete'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        campaign._measure_anchor_batch = original
        campaign._measure_anchor = original_scalar
        result['finished_unix'] = time.time()
        save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
