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
    parser.add_argument('--comparison', choices=('batch-width', 'trellis-best-form',
                        'trellis-call-profile', 'trellis-best-form-complete', 'best-batch-complete'),
                        default='batch-width')
    parser.add_argument('--batch-pair', choices=('8,32','8,16'), default='8,32')
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
            'experiments.glm_anchor_batch_pb_profile', '--comparison', args.comparison,
            '--batch-pair', args.batch_pair, '--out', args.out, '--', *command])

    import torch
    from prismaquant import tessera_campaign as campaign
    from experiments.glm_full_capture_profile import CaptureObserver
    from prismaquant.tessera_campaign import _wire_path
    batch_compare = args.comparison == 'best-batch-complete'
    complete = args.comparison in ('trellis-best-form-complete', 'best-batch-complete')
    widths = tuple(int(v) for v in args.batch_pair.split(','))
    unit_count = max(widths) if batch_compare else 16
    if command[command.index('--anchor-batch-size') + 1] != str(unit_count):
        raise ValueError(f'comparison requires the selected planner to reserve batch {unit_count}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='prismaquant.glm_anchor_batch_pb_profile.v1',
        status='running', scope=f'first compatible {unit_count} experts at the first campaign rung',
        comparison=args.comparison,batch_pair=widths if batch_compare else None,
        full_campaign_complete=False, arms=[], command=command,
        torch=torch.__version__, cuda=torch.version.cuda, started_unix=time.time())
    from tessera.cached_unit import encoder_source_sha256
    from tessera import window_viterbi
    result['producer_source_sha256'] = encoder_source_sha256()
    result['trellis_module_path'] = window_viterbi.__file__
    result['best_tile'] = os.environ.get('TESSERA_WINDOW_BEST_TILE')
    if args.comparison != 'batch-width' and not hasattr(window_viterbi, '_BEST_FORM_ENV'):
        raise RuntimeError('the selected producer does not implement the candidate')
    original_best_form = os.environ.get('TESSERA_WINDOW_BEST_FORM')
    plan_builds = [0]
    original_plan = window_viterbi._WindowPlan
    def counted_plan(**kwargs):
        plan_builds[0] += 1
        return original_plan(**kwargs)
    if batch_compare:
        window_viterbi._WindowPlan = counted_plan
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
        if len(names) != unit_count or not all('.experts.' in name for name in names):
            raise RuntimeError(f'expected {unit_count} compatible routed experts')
        result.update(qnames=names, format_name=kw['format_name'],
            shapes=[list(w.shape) for w in kw['weights']])
        if args.comparison == 'trellis-call-profile':
            # Profile complete, warmed calls without dynamically toggling CUPTI.
            # Inputs come from the first real campaign encode's residual, table
            # and weights. This is attribution only, not another anchor A/B.
            import shutil
            original_viterbi = window_viterbi.viterbi_window_fused

            def profile_call(targets, vectors, window_bits, rate, weights=None, chunk=512):
                result['call'] = dict(targets_shape=list(targets.shape),
                    vectors_shape=list(vectors.shape), window_bits=window_bits,
                    rate=rate, chunk=chunk, has_weights=weights is not None)
                reference = None
                for best, label in ((False, 'front'), (True, 'best')):
                    os.environ['TESSERA_WINDOW_BEST_FORM'] = str(int(best))
                    for _ in range(3):
                        original_viterbi(targets, vectors, window_bits, rate, weights, chunk)
                    torch.cuda.synchronize()
                    started = time.time()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
                            profile_memory=False, with_stack=False) as prof:
                        states, sse = original_viterbi(targets, vectors, window_bits, rate, weights, chunk)
                        torch.cuda.synchronize()
                    finished = time.time()
                    if reference is None:
                        reference = (states.clone(), sse)
                    if not torch.equal(states, reference[0]) or sse != reference[1]:
                        raise RuntimeError('paired actual-residual profile changed states or SSE')
                    trace = out/(label+'.trace.json.gz')
                    prof.export_chrome_trace(str(trace))
                    record = dict(label=label, best_form=best, started_unix=started,
                        finished_unix=finished, exact_parity=True, sse=sse,
                        trace=dict(path=str(trace), bytes=trace.stat().st_size,
                            sha256=hashlib.sha256(trace.read_bytes()).hexdigest()))
                    result['arms'].append(record)
                    print(json.dumps(record), flush=True)
                    if best:
                        shutil.copyfile(trace, os.environ['PRISMABUILD_PROFILE_TORCH_OUT'])
                    save()
                result['profile_scope'] = 'two complete warmed Viterbi calls on identical actual GLM residual; no dynamic toggling'
                raise BenchmarkComplete()

            subset = dict(kw)
            for field in ('qnames', 'weights', 'activations'):
                subset[field] = kw[field][:8]
            window_viterbi.viterbi_window_fused = profile_call
            try:
                original(**subset)
            finally:
                window_viterbi.viterbi_window_fused = original_viterbi
            raise RuntimeError('real encode did not reach the Viterbi profile hook')
        reference = None

        def arm(width, label, prof=None, best_form=None):
            nonlocal reference
            if best_form is not None:
                os.environ['TESSERA_WINDOW_BEST_FORM'] = '1' if best_form else '0'
            record = dict(label=label, batch_size=width, best_form=best_form,
                          units=unit_count, started_unix=time.time())
            result['arms'].append(record)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            outputs = []
            # Timed wrappers distinguish durability waits from GPU work. They
            # are symmetric across all arms and count overlapping operations
            # separately: torch.save includes its own serialization and I/O.
            io_calls = []
            original_fsync, original_save, original_write = os.fsync, torch.save, Path.write_bytes
            def timed(kind, function):
                def call(*positional, **named):
                    begun = time.perf_counter()
                    try:
                        return function(*positional, **named)
                    finally:
                        io_calls.append(dict(kind=kind, seconds=time.perf_counter()-begun))
                return call
            if complete:
                os.fsync = timed('fsync', original_fsync)
                torch.save = timed('torch_save', original_save)
                Path.write_bytes = timed('path_write_bytes', original_write)
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
            built_before = plan_builds[0]
            start = time.perf_counter()
            try:
                for offset in range(0, unit_count, width):
                    subset = dict(kw)
                    for field in ('qnames', 'weights', 'activations'):
                        subset[field] = kw[field][offset:offset + width]
                    outputs.extend(original(**subset))
                torch.cuda.synchronize()
            finally:
                os.fsync, torch.save, Path.write_bytes = original_fsync, original_save, original_write
                record.update(seconds=time.perf_counter()-start, finished_unix=time.time(),
                    allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                    reserved_peak_bytes=torch.cuda.max_memory_reserved())
                stop.set()
                if timer is not None:
                    timer.join()
                if complete:
                    record['io_calls'] = io_calls
            record['plans_built'] = plan_builds[0] - built_before
            if batch_compare and label.startswith('measured-') and record['plans_built']:
                raise RuntimeError('timed arm rebuilt an execution plan')
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
                units_per_second=unit_count / record['seconds'])
            print(json.dumps(record), flush=True)
            save()

        # Warm both controls, then ABBA with identical observation. The
        # best-form comparison holds actual encode batch size fixed at eight.
        variants = ([(8, 'front', False), (8, 'best', True)]
                    if args.comparison in ('trellis-best-form', 'trellis-best-form-complete')
                    else [(8, 'b8', None), (16, 'b16', None)])
        if batch_compare:
            variants = [(width, f'b{width}', True) for width in widths]
        if complete:
            import shutil
            original_viterbi = window_viterbi.viterbi_window_fused
            reference = None
            call_reference = None
            result['call_profiles'] = []
            for width, label, best in variants:
                captured = False
                def capture_first(*positional, **named):
                    nonlocal captured, call_reference
                    if captured:
                        return original_viterbi(*positional, **named)
                    captured = True
                    targets, vectors, window_bits, rate = positional[:4]
                    weights = named.get('weights', positional[4] if len(positional)>4 else None)
                    chunk = named.get('chunk', positional[5] if len(positional)>5 else 512)
                    inputs = dict(targets=targets.detach().cpu().contiguous(),
                        vectors=vectors.detach().cpu().contiguous(),
                        weights=None if weights is None else weights.detach().cpu().contiguous())
                    identities = {k: None if v is None else dict(shape=list(v.shape),dtype=str(v.dtype),
                        sha256=hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest())
                        for k,v in inputs.items()}
                    call = dict(inputs=identities,window_bits=window_bits,rate=rate,chunk=chunk)
                    if call_reference is None or batch_compare:
                        call_reference = call
                        evidence = out/(label+'-actual-first-viterbi-inputs.pt' if batch_compare else 'actual-first-viterbi-inputs.pt')
                        torch.save(dict(**inputs,window_bits=window_bits,rate=rate,chunk=chunk),evidence)
                        input_record = dict(path=str(evidence),bytes=evidence.stat().st_size,
                            sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
                        if batch_compare:
                            result.setdefault('batch_call_inputs', {})[label] = input_record
                        else:
                            result['call_inputs'] = input_record
                    if not batch_compare and call_reference != call:
                        raise RuntimeError('complete-call profiles received different actual inputs')
                    for _ in range(3):
                        original_viterbi(*positional, **named)
                    torch.cuda.synchronize()
                    started = time.time()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
                            profile_memory=False, with_stack=False) as profile:
                        answer = original_viterbi(*positional, **named)
                        torch.cuda.synchronize()
                    finished = time.time()
                    trace = out/(label+'.complete-call.trace.json.gz')
                    profile.export_chrome_trace(str(trace))
                    result['call_profiles'].append(dict(label=label,best_form=best,call=call,
                        started_unix=started,finished_unix=finished,sse=answer[1],
                        states_sha256=hashlib.sha256(answer[0].cpu().contiguous().numpy().tobytes()).hexdigest(),
                        trace=dict(path=str(trace),bytes=trace.stat().st_size,
                            sha256=hashlib.sha256(trace.read_bytes()).hexdigest())))
                    if best:
                        shutil.copyfile(trace,os.environ['PRISMABUILD_PROFILE_TORCH_OUT'])
                    save()
                    return answer
                window_viterbi.viterbi_window_fused = capture_first
                try:
                    arm(width,'warm-'+label,best_form=best)
                    if not captured:
                        raise RuntimeError('warm encode missed complete-call profile hook')
                finally:
                    window_viterbi.viterbi_window_fused = original_viterbi
            first,second = result['call_profiles']
            if not batch_compare and (first['sse'] != second['sse'] or first['states_sha256'] != second['states_sha256']):
                raise RuntimeError('complete-call profiles changed actual states or SSE')
            for index,(width,label,best) in enumerate((variants[0],variants[1],variants[1],variants[0])):
                arm(width,f'measured-{index}-{label}',best_form=best)
            result['profile_scope'] = 'complete warmed actual calls before timed ABBA; no dynamic toggling; symmetric timed I/O wrappers'
            raise BenchmarkComplete()
        for width, label, best_form in variants:
            arm(width, 'warm-'+label, best_form=best_form)
        # Re-enabling one profiler across long CUDA-graph replay intervals
        # corrupts later kernel durations in the real PB trace (2026-09-09).
        # Fresh contexts ALSO failed physical duration checks on torch 2.13.
        # Retain this reproducible negative screen; use trellis-call-profile
        # for a complete-call capture without dynamic collection toggling.
        primary = Path(os.environ['PRISMABUILD_PROFILE_TORCH_OUT'])
        event_count = 0
        for index, (width, label, best_form) in enumerate(
                (variants[0], variants[1], variants[1], variants[0])):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False, profile_memory=False, with_stack=False) as prof:
                prof.toggle_collection_dynamic(False, [torch.profiler.ProfilerActivity.CUDA])
                arm(width, f'measured-{index}-{label}', prof, best_form=best_form)
            trace_path = out/f'measured-{index}-{label}.trace.json.gz'
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
        window_viterbi._WindowPlan = original_plan
        campaign._measure_anchor_batch = original
        campaign._measure_anchor = original_scalar
        if original_best_form is None:
            os.environ.pop('TESSERA_WINDOW_BEST_FORM', None)
        else:
            os.environ['TESSERA_WINDOW_BEST_FORM'] = original_best_form
        result['finished_unix'] = time.time()
        save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
