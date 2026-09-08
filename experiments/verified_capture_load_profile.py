"""Isolated ABBA of sealed canonical entry copies; no source forward or fit claim."""
from __future__ import annotations
import argparse
import builtins
import cProfile
from contextlib import ExitStack
import copy
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import shutil
import threading
import time
from unittest.mock import patch
import weakref

import torch
from experiments.workspace_netdata import NetdataWriter, sample_netdata
from prismaquant import tessera_calibration_cache as cc, perturbed_x_cache as px
from prismaquant.cost_stage_checkpoint import _load_unit, unit_path, prepare_journal, write_unit
from prismaquant.memory_management import CaptureMemoryGuard

NAMES = ('model.language_model.layers.0.mlp.down_proj',
         'model.language_model.layers.3.mlp.experts.0.gate_proj')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def proc_io():
    return {key: int(value) for key, value in
            (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


def tensor_sha(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(tensor.numpy()).cast('B')).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--census', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.init()
    guard = CaptureMemoryGuard('cuda')
    source_journal = json.loads((args.source_root/'journal/manifest.json').read_text())
    census = json.loads(args.census.read_text())
    if cc.sha256(args.census) != source_journal['identity']['census_sha256']:
        raise RuntimeError('census differs from source capture identity')
    records, source_records = {}, {}
    root = args.out/'capture'
    (root/'inputs').mkdir(parents=True)
    for name in NAMES:
        record = _load_unit(unit_path(args.source_root/'journal', name), stage=cc.STAGE,
            qname=name, identity_sha256=source_journal['identity_sha256'])
        expected = str(Path('inputs')/px.activation_cache_filename(name))
        if record['path'] != expected:
            raise RuntimeError('source journal path is noncanonical')
        source = args.source_root/expected
        before = source.stat()
        shutil.copyfile(source, root/expected)
        if (px.cache_file_stat_signature(before) != px.cache_file_stat_signature(source.stat()) or
                cc.sha256(root/expected) != record['sha256']):
            raise RuntimeError('independent copy differs from source journal seal')
        if (root/expected).stat().st_ino == before.st_ino:
            raise RuntimeError('diagnostic copy aliases frozen artifact inode')
        records[name] = record
        source_records[name] = dict(record, source_file_bytes=before.st_size,
            source_signature=px.cache_file_stat_signature(before),
            copy_signature=px.cache_file_stat_signature((root/expected).stat()))
    # A diagnostic subset envelope over unchanged full-size canonical unit files.
    identity = copy.deepcopy(source_journal['identity'])
    identity['units'] = {name: identity['units'][name] for name in NAMES}
    for key in ('unit_shapes', 'counts', 'max_abs'):
        census[key] = {name: census[key][name] for name in NAMES}
    census_path = args.out/'subset-census.json'
    write(census_path, census)
    journal, digest, completed = prepare_journal(root/'journal', stage=cc.STAGE,
        resume=True, identity=identity, qnames=sorted(NAMES))
    for name, record in records.items():
        write_unit(journal, stage=cc.STAGE, qname=name, identity_sha256=digest, state=record)
    manifest = root/'capture_manifest.json'
    cc._json(manifest, dict(schema=cc.SCHEMA, status='complete', identity=identity, entries=records))
    original_manifest = manifest.read_bytes()
    policy = dict(schema=px.VERIFIED_ACTIVATION_LOAD_SCHEMA,
        max_buffer_bytes=max(record['source_file_bytes'] for record in source_records.values()),
        max_scratch_bytes=4*1024**2)
    s = sum(cc._capture_storage_bytes(name, census, identity['max_act_rows']) for name in NAMES)
    # Persistent expected CPU S, largest serialized F, loaded CPU S, possible
    # GPU S, 4 MiB scratch plus 2 GiB explicit runtime/profile/metadata allowance.
    plan = dict(expected_cpu=s, serialized=policy['max_buffer_bytes'],
        loaded_cpu=s, loaded_gpu=s, scratch=policy['max_scratch_bytes'], runtime=2*1024**3)
    if sum(plan.values()) > guard.cap_bytes:
        raise RuntimeError('unchanged PB cap refuses diagnostic phase plan')
    write(args.out/'inputs.json', dict(scope='two full-size unit copies; diagnostic subset envelope',
        source_identity_sha256=source_journal['identity_sha256'], source_entries=source_records,
        policy=policy, phase_bound=plan, phase_bound_bytes=sum(plan.values()),
        torch=torch.__version__, cuda=torch.version.cuda))
    expected = {name: torch.load(root/record['path'], weights_only=True) for name, record in records.items()}
    acts = {name: value['inputs'] for name, value in expected.items()}
    hessians = {name: value['hessian'] for name, value in expected.items()}
    expected_hashes = {name: {key: tensor_sha(value[key]) for key in ('inputs','hessian')}
                       for name, value in expected.items()}
    stop, errors, state = threading.Event(), [], {'arm': None, 'operation': None}
    def observe():
        with (args.out/'netdata.jsonl').open('w') as stream:
            writer = NetdataWriter(stream)
            while not stop.is_set():
                for host in ('sparky', 'sparklina'):
                    try:
                        writer.write(dict(sample_netdata(host), arm=state['arm'], operation=state['operation']))
                    except Exception as exc:
                        errors.append(str(exc))
                stop.wait(.5)
    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    results = []
    raw_refs, buffer_io = [], []
    original_reader = px._VerifiedBufferReader
    class WatchedReader(original_reader):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            raw_refs.append(weakref.ref(self))
            self.observed = dict(copy_read_bytes=0, max_copy_read_bytes=0,
                                 readinto_bytes=0, max_readinto_bytes=0,
                                 copy_cap_bytes=self._max_copy_bytes)
            buffer_io.append(self.observed)
        def read(self, *args):
            value = super().read(*args)
            self.observed['copy_read_bytes'] += len(value)
            self.observed['max_copy_read_bytes'] = max(self.observed['max_copy_read_bytes'], len(value))
            return value
        def readinto(self, target):
            value = super().readinto(target)
            self.observed['readinto_bytes'] += value
            self.observed['max_readinto_bytes'] = max(self.observed['max_readinto_bytes'], value)
            return value
    def check(label, **kwargs):
        if label.startswith(('after_verified_capture_buffer_release', 'before_capture_prefetch')):
            if any(ref() is not None and ref()._view is not None for ref in raw_refs):
                raise RuntimeError('serialized buffer overlaps return/transfer boundary')
        guard.check(label, **kwargs)
    try:
        for arm, mode in enumerate(('legacy', 'verified', 'verified', 'legacy')):
            state['arm'] = arm
            for operation in ('replay', 'seal', 'prefetch'):
                state['operation'] = operation
                gc.collect()
                torch.cuda.empty_cache()
                for record in records.values():
                    path = root/record['path']
                    px.release_activation_cache_file_pages(path, expected_stat=path.stat())
                buffer_start = len(buffer_io)
                reads = {'bytes': 0, 'opens': 0}
                class Counted:
                    def __init__(self, handle): self.handle = handle
                    def __getattr__(self, name): return getattr(self.handle, name)
                    def read(self, *a):
                        value = self.handle.read(*a); reads['bytes'] += len(value); return value
                    def readinto(self, target):
                        value = self.handle.readinto(target); reads['bytes'] += value or 0; return value
                    def __enter__(self): return self
                    def __exit__(self, *a): return self.handle.__exit__(*a)
                selected = {root/record['path'] for record in records.values()}
                def counting(original):
                    def opened(path, *a, **k):
                        value = original(path, *a, **k)
                        actual = Path(os.readlink(f'/proc/self/fd/{path}')) if isinstance(path,int) else Path(path)
                        if actual in selected:
                            reads['opens'] += 1
                            return Counted(value)
                        return value
                    return opened
                extra = dict(verified_load_policy=policy) if mode == 'verified' else {}
                execution = {}
                cpu_profile = cProfile.Profile()
                before = proc_io()
                started = time.time()
                with ExitStack() as stack:
                    stack.enter_context(patch.object(builtins, 'open', counting(builtins.open)))
                    stack.enter_context(patch.object(io, 'open', counting(io.open)))
                    stack.enter_context(patch.object(px, '_VerifiedBufferReader', WatchedReader))
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
                        cpu_profile.enable()
                        if operation == 'replay':
                            owner = cc.CaptureWriter(root, census_path=census_path, identity=identity,
                                release_file_pages=True, resource_check=check, **extra)
                            owner.write(acts=acts, hessians=hessians, counts=census['counts'], maxima=census['max_abs'])
                            execution = owner.load_execution
                        elif operation == 'seal':
                            cc.publish_capture(root, census_path=census_path, identity=identity,
                                existing_entries=records, release_file_pages=True, resource_check=check,
                                **extra, **(dict(load_execution=execution) if extra else {}))
                        else:
                            values, _ = cc.prefetch_capture(manifest, expected_identity=identity,
                                census=census, names=NAMES, device='cuda', release_file_pages=True,
                                resource_check=check, **extra,
                                **(dict(load_execution=execution) if extra else {}))
                        torch.cuda.synchronize()
                        cpu_profile.disable()
                elapsed = time.time()-started
                after = proc_io()
                stem = args.out/f'arm-{arm}-{mode}-{operation}'
                cpu_profile.dump_stats(str(stem)+'.cprofile')
                with Path(str(stem)+'.cprofile.txt').open('w') as stream:
                    pstats.Stats(cpu_profile, stream=stream).sort_stats('cumulative').print_stats(40)
                profiler.export_chrome_trace(str(stem)+'.trace.json')
                if operation == 'prefetch':
                    actual = {name: {'inputs': tensor_sha(values[0][name]),
                        'hessian': tensor_sha(values[1][name])} for name in NAMES}
                    if actual != expected_hashes:
                        raise RuntimeError('native prefetch tensor hash mismatch')
                    del values
                if manifest.read_bytes() != original_manifest:
                    raise RuntimeError('loader changed canonical manifest bytes')
                if mode == 'verified' and reads['bytes'] != sum(
                        record['source_file_bytes'] for record in source_records.values()):
                    raise RuntimeError('verified operation reread source bytes')
                row = dict(arm=arm, mode=mode, operation=operation, started=started,
                    finished=started+elapsed, elapsed_s=elapsed, source_reads=reads,
                    proc_io_delta={key: after[key]-before[key] for key in before},
                    execution=execution, buffer_io=buffer_io[buffer_start:],
                    memory_guard=guard.snapshot(),
                    tensor_hashes=expected_hashes, live_serialized_buffers=sum(
                        ref() is not None and ref()._view is not None for ref in raw_refs))
                write(Path(str(stem)+'.json'), row)
                results.append(row)
                print(json.dumps(dict(arm=arm, mode=mode, operation=operation, elapsed_s=elapsed, reads=reads)), flush=True)
        for name, record in records.items():
            if cc.sha256(root/record['path']) != record['sha256']:
                raise RuntimeError('measurement changed copied artifact bytes')
    finally:
        stop.set(); observer.join(timeout=15)
    if errors or observer.is_alive():
        raise RuntimeError(f'Netdata collection failed: {errors[:3]}')
    write(args.out/'summary.json', dict(status='passed', results=results,
        limits='Independent warm/cold-advised file copies; no model forward, KL, serving, full-capture fit or production speed claim.'))


if __name__ == '__main__':
    main()
