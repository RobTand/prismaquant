"""Observe the original campaign traversal; no extra forwards or scheduling.

This is experiment instrumentation, not an alternate capture implementation.
Python stacks cover the main thread at 1 Hz; selected original forward windows
use torch.profiler. Both Sparks' host series use the existing Netdata contract
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
        self.result.update(finished_unix=time.time(),
            status='failed' if error or self.result['errors'] else 'complete',
            campaign_error=None if error is None else repr(error))
        (self.out/'result.json').write_text(json.dumps(self.result, indent=2)+'\n')
        if error is None and self.result['errors']:
            raise RuntimeError('capture completed but required profiler evidence is incomplete')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-out', type=Path, required=True)
    parser.add_argument('campaign_argv', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.campaign_argv
    if command[:1] == ['--']:
        command = command[1:]
    if '--capture-calibration-out' not in command or '--streaming' not in command:
        parser.error('observer requires the streamed canonical capture action')
    if not torch.cuda.is_available():
        raise RuntimeError('full capture profiler requires CUDA')
    from prismaquant import tessera_campaign as campaign
    original = campaign._collect_activations
    with CaptureObserver(args.evidence_out) as observer:
        campaign._collect_activations = observer.wrap_collector(original)
        try:
            return campaign.main(command)
        finally:
            campaign._collect_activations = original


if __name__ == '__main__':
    raise SystemExit(main())
