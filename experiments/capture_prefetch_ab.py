"""Cold/warm A/B of the verified capture prefetch at one or more reader counts.

One campaign row's capture file set is loaded exactly the way
``tessera_campaign._prefetch_selected_capture`` loads it: the row's own
identity, census, verified load policy, memory guard, ``release_file_pages``
and device. The only knob is ``PRISMAQUANT_CAPTURE_READ_THREADS``.

The arms run inside one process so the interpreter, the guard baseline and
the CUDA context are the same for both: COLD is the first read of that row's
files by this experiment, WARM is the immediate re-read. Neither arm can drop
the server's ARC, so COLD means "not read by us before", and the server-side
Netdata samples beside it are what says whether it behaved like a cold read.

No campaign output is written or read; the capture and the row unit list are
opened read-only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.request

SCHEMA = 'prismaquant.capture_prefetch_ab.v1'
SERVER_CHARTS = ('zfs.*', 'zfspool.*', 'nfsd.*', 'disk.sd*', 'disk.nvme*',
                 'system.cpu', 'system.io', 'system.ram', 'mem.available')
MAX_RESPONSE_BYTES = 8*1024**2


def proc_io():
    return {key.strip(): int(value) for key, value in
            (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


def proc_status():
    fields = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmHWM', 'VmRSS', 'VmPeak')):
            key, value = line.split(':')
            fields[key] = int(value.split()[0])*1024
    return fields


def nfs_server_address(mount='/mnt/shared'):
    """The address this box actually reads the capture from."""
    for line in Path('/proc/mounts').read_text().splitlines():
        parts = line.split()
        if len(parts) > 3 and parts[1] == mount and parts[2].startswith('nfs'):
            found = re.search(r'\baddr=([0-9a-fA-F.:]+)', parts[3])
            if found:
                return found.group(1)
    raise RuntimeError(f'no NFS mount found at {mount}')


def sample_host(host, charts):
    query = f"format=json&filter={'%20'.join(charts)}"
    with urllib.request.urlopen(
            f'http://{host}:19999/api/v1/allmetrics?{query}', timeout=6) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError(f'{host}: Netdata response exceeded its byte cap')
    metrics = json.loads(raw)
    return {name: {'units': chart.get('units'), 'last_updated': chart.get('last_updated'),
                   'dimensions': {key: value['value']
                                  for key, value in chart.get('dimensions', {}).items()}}
            for name, chart in metrics.items()}


class Sampler(threading.Thread):
    """Bounded background host evidence for both ends of the transfer."""

    def __init__(self, server, period=2.0, cap=4000):
        super().__init__(daemon=True, name='netdata-sampler')
        self.server = server
        self.period = period
        self.cap = cap
        self.samples = []
        self.errors = []
        self.mark = []
        self._stop = threading.Event()

    def note(self, label):
        self.mark.append(dict(label=label, time=time.time(), monotonic=time.monotonic()))

    def run(self):
        from experiments.workspace_netdata import sample_netdata
        while not self._stop.wait(self.period):
            if len(self.samples) >= self.cap:
                self.errors.append('sample cap reached')
                return
            record = dict(time=time.time(), monotonic=time.monotonic())
            for key, call in (('spark', lambda: sample_netdata('127.0.0.1')),
                              ('server', lambda: sample_host(self.server, SERVER_CHARTS))):
                try:
                    record[key] = call()
                except Exception as failure:  # noqa: BLE001 - evidence, not control flow
                    self.errors.append(f'{key}: {type(failure).__name__}: {failure}')
            self.samples.append(record)

    def stop(self):
        self._stop.set()
        self.join(timeout=15)


def row_inputs(base, row_id):
    workspace = Path(base)/'first-proof-anchor-preparation-05/workspace'
    plan = json.loads((workspace/'plan.json').read_text())
    manifest = json.loads((workspace/'manifest.json').read_text())
    row = next(entry for entry in plan['rows'] if entry['row_id'] == row_id)
    action = next(entry for entry in manifest
                  if f'/rows/{row_id}/' in ' '.join(entry['argv']))
    argv = action['argv']
    policy = json.loads(argv[argv.index('--capture-load-policy')+1])
    units = json.loads(Path(row['units']).read_text())
    names = sorted({member for group in units['groups'] for member in group['members']})
    return dict(row=row, capture=plan['calibration_cache'],
                census_path=str(workspace/'census.json'), policy=policy, names=names,
                max_act_rows=int(argv[argv.index('--max-act-rows')+1]),
                memory_gb=plan['row_memory_gb'][row_id], argv_env=action['env'])


def run_arm(*, arm, readers, capture, identity, census, names, policy, device, guard,
            sampler):
    """One prefetch, instrumented only by the guard callback the campaign passes."""
    from prismaquant import tessera_calibration_cache as cc
    import torch
    events = []
    lock = threading.Lock()
    start_monotonic = time.monotonic()

    def resource_check(label, *, reserve_bytes=0):
        moment = time.monotonic()
        result = guard.check(label, reserve_bytes=reserve_bytes)
        with lock:
            events.append((round(moment - start_monotonic, 6), label, reserve_bytes,
                           threading.get_ident()))
        return result

    os.environ['PRISMAQUANT_CAPTURE_READ_THREADS'] = str(readers)
    execution = {}
    sampler.note(f'{arm}:{readers}:begin')
    io_before, started = proc_io(), time.monotonic()
    values, receipt = cc.prefetch_capture(capture['path'], expected_identity=identity,
        census=census, names=names, device=device, expected_sha256=capture['sha256'],
        resource_check=resource_check, release_file_pages=True,
        verified_load_policy=policy, load_execution=execution)
    seconds = time.monotonic() - started
    io_after = proc_io()
    sampler.note(f'{arm}:{readers}:end')
    acts, hessians, counts, maxima = values
    resident = sum(t.numel()*t.element_size() for t in (*acts.values(), *hessians.values()))
    summary = dict(arm=arm, readers=readers, seconds=seconds, units=len(names),
        source_read_bytes=execution['source_read_bytes'], resident_bytes=resident,
        read_bytes_per_second=execution['source_read_bytes']/seconds,
        ordered_load_identities_sha256=execution['ordered_load_identities_sha256'],
        loaded_entries=execution['loaded_entries'],
        peak_buffer_bytes=execution['peak_buffer_bytes'],
        peak_archive_storage_bytes=execution['peak_archive_storage_bytes'],
        identity_sha256=execution['identity_sha256'], capture_receipt=receipt,
        proc_io_delta={key: io_after[key]-value for key, value in io_before.items()},
        proc_status=proc_status(), phases=phase_summary(events),
        guard=dict(peak_bytes=guard.peak_bytes, peak_checkpoint=guard.peak_checkpoint,
                   baseline=None if guard.baseline is None else dict(guard.baseline),
                   min_available_bytes=guard.min_available_bytes),
        metadata_sha256=hashlib.sha256(json.dumps(
            [[name, counts[name], maxima[name]] for name in names],
            sort_keys=True, allow_nan=False).encode()).hexdigest())
    values = acts = hessians = counts = maxima = None
    del values, acts, hessians, counts, maxima
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return summary, events


def phase_summary(events):
    """Wall time per unit split into its file work and its consumer work.

    Labels carry the file name for the reader's own calls and the unit name
    for the consumer's, so the split needs no extra instrumentation.
    """
    read, decode, consume = {}, {}, {}
    for moment, label, _reserve, _thread in events:
        stem, _, target = label.partition(':')
        if not target:
            continue
        table = (read if stem in ('before_verified_capture_buffer',
                                  'before_verified_capture_read') else
                 decode if stem in ('before_verified_capture_decode',
                                    'after_verified_capture_buffer_release') else
                 consume if stem in ('before_capture_prefetch', 'after_capture_prefetch')
                 else None)
        if table is None:
            continue
        first, last = table.get(target, (moment, moment))
        table[target] = (min(first, moment), max(last, moment))
    def total(table):
        return round(sum(last-first for first, last in table.values()), 3)
    span = (max(moment for moment, *_ in events) - min(moment for moment, *_ in events)
            if events else 0.0)
    return dict(events=len(events), wall_span_seconds=round(span, 3),
                file_thread_seconds=total(read)+total(decode),
                read_thread_seconds=total(read), decode_thread_seconds=total(decode),
                consumer_seconds=total(consume))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='/mnt/shared/tessera-measurements/glm-canonical-census-20260908')
    parser.add_argument('--row', required=True)
    parser.add_argument('--readers', type=int, required=True)
    parser.add_argument('--arms', default='cold,warm')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--server', default=None)
    parser.add_argument('--limit', type=int, default=0,
        help='smoke only: load the first N units of the row instead of all of them')
    args = parser.parse_args()

    import torch
    from prismaquant import tessera_calibration_cache as cc
    from prismaquant.memory_management import CaptureMemoryGuard

    torch.set_num_threads(1)
    inputs = row_inputs(args.base, args.row)
    capture = inputs['capture']
    manifest = cc.require_capture_contract(capture['path'], expected_sha256=capture['sha256'])
    identity = manifest['identity']
    census = json.loads(Path(inputs['census_path']).read_text())
    if identity['max_act_rows'] != inputs['max_act_rows']:
        raise RuntimeError('row argv and capture identity disagree on max_act_rows')
    names = inputs['names'][:args.limit] if args.limit else inputs['names']
    files = [Path(capture['path']).parent/manifest['entries'][name]['path'] for name in names]
    sizes = [path.stat().st_size for path in files]

    args.out.mkdir(parents=True, exist_ok=True)
    server = args.server or nfs_server_address()
    sampler = Sampler(server)
    sampler.start()
    guard = None if not str(args.device).startswith('cuda') else CaptureMemoryGuard(args.device)
    if guard is None:
        raise RuntimeError('this experiment measures the guarded campaign path')
    results, failure = [], None
    try:
        for arm in args.arms.split(','):
            summary, events = run_arm(arm=arm, readers=args.readers, capture=capture,
                identity=identity, census=census, names=names, policy=inputs['policy'],
                device=args.device, guard=guard, sampler=sampler)
            results.append(summary)
            (args.out/f'events-{arm}-{args.readers}.jsonl').write_text(
                ''.join(json.dumps(event)+'\n' for event in events))
            print(f"[ab] {arm} readers={args.readers} {summary['seconds']:.1f}s "
                  f"{summary['read_bytes_per_second']/1e6:.1f} MB/s "
                  f"chain={summary['ordered_load_identities_sha256'][:16]}", flush=True)
    except BaseException as error:  # noqa: BLE001 - recorded before re-raising
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        sampler.stop()
        record = dict(schema=SCHEMA, row=args.row, readers=args.readers,
            device=args.device, arms=args.arms, failure=failure, results=results,
            units=len(names), source_bytes=sum(sizes),
            file_bytes=dict(min=min(sizes), max=max(sizes), mean=sum(sizes)//len(sizes)),
            capture=capture, census=inputs['census_path'], policy=inputs['policy'],
            row_memory_gb=inputs['memory_gb'], server=server, limit=args.limit,
            host=os.uname().nodename, torch=torch.__version__,
            environment={key: os.environ.get(key) for key in
                         ('PRISMAQUANT_CAPTURE_READ_THREADS', 'OMP_NUM_THREADS',
                          'PRISMAQUANT_LAYER_READ_THREADS', 'PYTORCH_ALLOC_CONF')},
            netdata=dict(errors=sampler.errors, marks=sampler.mark,
                         samples=len(sampler.samples)))
        (args.out/f'summary-{args.row}-{args.readers}.json').write_text(
            json.dumps(record, indent=2, sort_keys=True)+'\n')
        (args.out/f'netdata-{args.row}-{args.readers}.jsonl').write_text(
            ''.join(json.dumps(sample, separators=(',', ':'))+'\n'
                    for sample in sampler.samples))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
