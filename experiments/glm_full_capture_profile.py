"""Observe the original campaign traversal; no extra forwards or scheduling.

This is experiment instrumentation, not an alternate capture implementation.
Python stacks cover the main thread at 1 Hz; selected original forward or
scalar-anchor windows use torch.profiler. Both Sparks' host series use the existing Netdata contract
at 5-second intervals, with explicit disk caps suitable for a 24-hour action.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid

import torch

from experiments.workspace_netdata import NetdataWriter, sample_netdata


class CaptureObserver:
    def __init__(self, out, *, profile_layers=(0, 3, 4, 23, 44)):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.profile_layers = set(profile_layers)
        self.stopped = threading.Event()
        self.main_thread = threading.get_ident()
        self.index = 0
        self.result = dict(schema='prismaquant.glm_full_capture_profile.v1',
            status='running', started_unix=time.time(), collections=[], errors=[],
            torch=torch.__version__, cuda=torch.version.cuda,
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            kernel=os.uname().release,
            netdata=dict(hosts=['sparky', 'sparklina'], interval_seconds=5,
                         byte_cap=3*1024**3),
            python_sampler=dict(scope='main_thread_only', interval_seconds=1,
                                byte_cap=512*1024**2),
            profile_layers=sorted(self.profile_layers),
            forward_windows_zero_based=[[0, 1], [31, 32]])
        for name in ('srcversion', 'parameters/delegation_watermark'):
            path = Path('/sys/module/nfsv4')/name
            self.result['nfsv4_'+name.replace('/', '_')] = (
                path.read_text().strip() if path.exists() else None)
        self.threads = []

    def monitor(self, kind):
        try:
            cap = self.result[kind]['byte_cap']
            with (self.out/(kind+'.jsonl')).open('x') as handle:
                writer = NetdataWriter(handle, max_bytes=cap)
                while not self.stopped.is_set():
                    if kind == 'netdata':
                        for host in self.result[kind]['hosts']:
                            writer.write(sample_netdata(host))
                    else:
                        frame = sys._current_frames().get(self.main_thread)
                        frames = traceback.extract_stack(frame)
                        del frame
                        writer.write(dict(time=time.time(),
                            frames=[dict(file=x.filename, line=x.lineno, function=x.name)
                                    for x in frames],
                            process_io=Path('/proc/self/io').read_text()))
                    self.result[kind]['bytes_written'] = writer.bytes_written
                    self.result[kind]['samples'] = self.result[kind].get('samples', 0)+1
                    self.stopped.wait(self.result[kind]['interval_seconds'])
        except BaseException as error:
            self.result['errors'].append(dict(instrument=kind, error=repr(error)))

    def wrap_collector(self, original):
        def collect(*args, **kwargs):
            index = self.index
            self.index += 1
            record = dict(collection_index=index, started_unix=time.time(), batches=0,
                          traces=[], status='running')
            self.result['collections'].append(record)
            forward = kwargs['forward_batch']
            profiler = None

            def observed_forward(batch):
                value = forward(batch)
                record['batches'] += 1
                if profiler is not None:
                    profiler.step()
                return value

            def schedule(step):
                if step in (1, 32):
                    return torch.profiler.ProfilerAction.RECORD_AND_SAVE
                if step in (0, 31):
                    return torch.profiler.ProfilerAction.RECORD
                return torch.profiler.ProfilerAction.NONE

            def ready(prof):
                stem = f'collection-{index:02d}-window-{len(record["traces"]):02d}'
                path = self.out/(stem+'.trace.json')
                prof.export_chrome_trace(str(path))
                (self.out/(stem+'.profile.txt')).write_text(
                    prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=60))
                with path.open('rb') as handle:
                    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
                record['traces'].append(dict(path=path.name, bytes=path.stat().st_size,
                                             sha256=digest, after_batches=record['batches']))

            context = (torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                schedule=schedule, record_shapes=True, profile_memory=True,
                on_trace_ready=ready) if index in self.profile_layers else nullcontext())
            kwargs['forward_batch'] = observed_forward
            try:
                with context as profiler:
                    value = original(*args, **kwargs)
                record['status'] = 'complete'
                return value
            except BaseException as error:
                record.update(status='failed', error=repr(error))
                raise
            finally:
                record['finished_unix'] = time.time()
                (self.out/'progress.json').write_text(json.dumps(self.result, indent=2)+'\n')
        return collect

    def __enter__(self):
        for kind in ('netdata', 'python_sampler'):
            thread = threading.Thread(target=self.monitor, args=(kind,), daemon=True)
            thread.start()
            self.threads.append(thread)
        return self

    def __exit__(self, error_type, error, tb):
        self.stopped.set()
        for thread in self.threads:
            thread.join(timeout=12)
            if thread.is_alive():
                self.result['errors'].append(dict(instrument='shutdown', error='monitor did not stop'))
        self.validate_result()
        self.result.update(finished_unix=time.time(),
            status='failed' if error or self.result['errors'] else 'complete',
            campaign_error=None if error is None else repr(error))
        final = json.dumps(self.result, indent=2)+'\n'
        (self.out/'result.json').write_text(final)
        (self.out/'progress.json').write_text(final)
        if error is None and self.result['errors']:
            raise RuntimeError('campaign completed but required profiler evidence is incomplete')

    def validate_result(self):
        pass


class AnchorObserver(CaptureObserver):
    """Observe finite original scalar calls; retain no weight or activation tensors.

    Observation failures are reported after the campaign can journal successful
    anchors. They must not turn an encoded anchor into an apparent encoder
    failure, or make a retry encode that successful anchor again.
    """
    def __init__(self, out, *, profile_calls, trace_max_bytes, command):
        if (not profile_calls or 0 not in profile_calls or len(profile_calls) > 4
                or any(type(i) is not int or i < 0 for i in profile_calls)
                or len(set(profile_calls)) != len(profile_calls)):
            raise ValueError('anchor profile calls require zero and at most four distinct nonnegative indices')
        if type(trace_max_bytes) is not int or not 0 < trace_max_bytes <= 2*1024**3:
            raise ValueError('anchor trace byte cap must be positive and at most 2 GiB')
        parent = Path(out)
        parent.mkdir(parents=True, exist_ok=True)
        super().__init__(parent/('attempt-'+uuid.uuid4().hex), profile_layers=())
        self.profile_calls = set(profile_calls)
        self.trace_max_bytes = trace_max_bytes
        self.result.update(schema='prismaquant.glm_selected_anchor_profile.v1',
            mode='selected_anchors', campaign_argv=list(command), anchor_calls=0,
            anchors=[], profile_calls_zero_based=sorted(profile_calls),
            trace_max_bytes=trace_max_bytes, native_anchor_profiled=False,
            trace_cap_scope='Exported bytes per window; not a live profiler-memory bound.')
        self.result.pop('forward_windows_zero_based')
        self.result.pop('profile_layers')

    def observation_error(self, error):
        self.result['errors'].append(dict(instrument='anchor_profiler', error=repr(error)))

    def wrap_anchor(self, original):
        def anchor(*args, **kwargs):
            index = self.result['anchor_calls']
            self.result['anchor_calls'] += 1
            if index not in self.profile_calls:
                return original(*args, **kwargs)
            record = dict(call_index=index, qname=kwargs.get('qname'),
                format_name=kwargs.get('format_name'), started_unix=time.time(),
                status='running', cuda_events=0)
            self.result['anchors'].append(record)
            called = False
            original_error = None
            value = None
            try:
                try:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
                            profile_memory=False, with_stack=False) as profiler:
                        called = True
                        try:
                            value = original(*args, **kwargs)
                        except BaseException as error:
                            original_error = error
                            raise
                    path = self.out/f'anchor-{index:06d}.trace.json'
                    profiler.export_chrome_trace(str(path))
                    size = path.stat().st_size
                    if not 0 < size <= self.trace_max_bytes:
                        record['rejected_trace_bytes'] = size
                        path.unlink()
                        raise RuntimeError('anchor trace exceeded its exported byte cap or was empty')
                    record['cuda_events'] = sum(
                        event.device_type == torch.autograd.DeviceType.CUDA
                        for event in profiler.events())
                    if record['cuda_events'] == 0:
                        raise RuntimeError('anchor trace contains no CUDA events')
                    with path.open('rb') as stream:
                        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                    record.update(status='complete', trace=dict(path=path.name, bytes=size, sha256=digest))
                    (self.out/f'anchor-{index:06d}.profile.txt').write_text(
                        profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=60))
                except Exception as error:
                    if original_error is not None:
                        raise original_error.with_traceback(original_error.__traceback__)
                    self.observation_error(error)
                    record.update(status='observation_failed', error=repr(error))
                    if not called:
                        called = True
                        value = original(*args, **kwargs)
                return value
            except BaseException as error:
                record.update(status='anchor_failed', error=repr(error))
                raise
            finally:
                record['finished_unix'] = time.time()
                # Progress is bounded by the explicit windows, not every anchor.
                try:
                    (self.out/'progress.json').write_text(json.dumps(self.result, indent=2)+'\n')
                except OSError as error:
                    self.observation_error(error)
        return anchor

    def validate_result(self):
        profiled = any(r['status'] == 'complete' and r['cuda_events'] > 0
                       for r in self.result['anchors'])
        self.result.update(native_anchor_profiled=profiled,
            profile_status=('observed' if profiled else
                            'no_new_anchors' if self.result['anchor_calls'] == 0 else 'missing'),
            unobserved_calls=sorted(self.profile_calls -
                                   {r['call_index'] for r in self.result['anchors']}))
        if self.result['anchor_calls'] and not profiled:
            self.observation_error(RuntimeError('no native anchor window was recorded'))
        if self.result['anchor_calls']:
            for kind in ('netdata', 'python_sampler'):
                if not self.result[kind].get('samples'):
                    self.observation_error(RuntimeError(f'no {kind} sample was recorded'))


def selected_anchor_command(command):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--streaming', action='store_true')
    for flag in ('--units', '--calibration-cache', '--calibration-cache-sha256',
                 '--census-out', '--capture-calibration-out'):
        parser.add_argument(flag)
    parser.add_argument('--anchor-batch-size', type=int, default=1)
    args, _ = parser.parse_known_args(command)
    if not (args.streaming and args.units and args.calibration_cache
            and args.calibration_cache_sha256 and not args.census_out
            and not args.capture_calibration_out and args.anchor_batch_size == 1):
        raise ValueError('anchor observer requires selected canonical reuse and anchor batch size one')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-out', type=Path, required=True)
    parser.add_argument('--selected-anchors', action='store_true')
    parser.add_argument('--anchor-profile-calls',
                        help='Explicit comma-separated scalar-call indices; includes zero, at most four.')
    parser.add_argument('--anchor-trace-max-bytes', type=int,
                        help='Required exported-byte cap per selected-anchor trace; at most 2 GiB.')
    parser.add_argument('campaign_argv', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.campaign_argv
    if command[:1] == ['--']:
        command = command[1:]
    if args.selected_anchors:
        try:
            selected_anchor_command(command)
            calls = [int(i) for i in args.anchor_profile_calls.split(',')]
            if args.anchor_trace_max_bytes is None:
                raise ValueError('selected anchors require an explicit trace byte cap')
        except (AttributeError, ValueError) as error:
            parser.error(str(error))
    elif args.anchor_profile_calls is not None or args.anchor_trace_max_bytes is not None:
        parser.error('anchor profiling options require --selected-anchors')
    elif '--capture-calibration-out' not in command or '--streaming' not in command:
        parser.error('observer requires the streamed canonical capture action')
    if not torch.cuda.is_available():
        raise RuntimeError('full capture profiler requires CUDA')
    from prismaquant import tessera_campaign as campaign
    method = '_measure_anchor' if args.selected_anchors else '_collect_activations'
    original = getattr(campaign, method)
    observer = (AnchorObserver(args.evidence_out, profile_calls=calls,
                              trace_max_bytes=args.anchor_trace_max_bytes, command=command)
                if args.selected_anchors else CaptureObserver(args.evidence_out))
    with observer:
        setattr(campaign, method, observer.wrap_anchor(original) if args.selected_anchors
                else observer.wrap_collector(original))
        try:
            return campaign.main(command)
        finally:
            setattr(campaign, method, original)


if __name__ == '__main__':
    raise SystemExit(main())
