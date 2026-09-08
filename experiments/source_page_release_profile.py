"""Bounded native paired qualification of consumed safetensors page release.

Run only inside an admitted PB action. Reads the same real checkpoint subset
with original per-key advice and coalesced advice, retaining exact GPU bytes.
"""
import argparse
import ctypes
import json
import mmap
import os
from pathlib import Path
import threading
import time

import torch
from prismaquant import layer_streaming as ls
from prismaquant.memory_management import CaptureMemoryGuard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    model = Path(args.model)
    weights = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    available = sorted(k for k in weights if 'layers.20.mlp.experts.' in k)
    shard = model/weights[available[0]]
    with shard.open('rb') as handle:
        length = int.from_bytes(handle.read(8), 'little')
        header = json.loads(handle.read(length))
    keys = sorted((k for k in available if weights[k] == shard.name),
                  key=lambda k: header[k]['data_offsets'][0])[:64]
    assert len(keys) == 64
    assert all(header[k]['dtype'] == 'BF16' for k in keys)
    assert all(header[a]['data_offsets'][1] == header[b]['data_offsets'][0]
               for a,b in zip(keys, keys[1:]))
    base = length+8
    begin = base+header[keys[0]]['data_offsets'][0]
    end = base+header[keys[-1]]['data_offsets'][1]
    page = os.sysconf('SC_PAGE_SIZE')
    first, last = (begin+page-1)//page*page, end//page*page
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.mincore.restype = ctypes.c_int
    def residency():
        with shard.open('rb') as handle:
            region = mmap.mmap(handle.fileno(), last-first, access=mmap.ACCESS_COPY, offset=first)
            address = ctypes.addressof(ctypes.c_char.from_buffer(region))
            vector = (ctypes.c_ubyte*((last-first)//page))()
            result = libc.mincore(address, last-first, vector)
            if result:
                raise OSError(ctypes.get_errno(), 'mincore')
            resident = sum(bool(value & 1) for value in vector)*page
            region.close()
        return resident
    guard = CaptureMemoryGuard('cuda')
    def memory():
        stats = dict(line.split() for line in (guard.scope/'memory.stat').read_text().splitlines())
        return dict(time_unix=time.time(), current_bytes=int((guard.scope/'memory.current').read_text()),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(),
                    **{k:int(stats[k]) for k in ('anon','file','file_thp','file_dirty','file_writeback')})
    mapping = dict.fromkeys(keys, str(shard))
    names = {k:k for k in keys}
    original = ls._advise_consumed_safetensors_pages
    source_stat = shard.stat()
    def read():
        return ls._read_layer_to_device('', mapping, names, torch.bfloat16, torch.device('cuda:0'))
    reference = read()
    torch.cuda.synchronize()
    assert sum(t.numel()*t.element_size() for t in reference.values()) == 1024**3
    arms = []
    for index, mode in enumerate(('per_key','coalesced','coalesced','per_key')):
        # All advised bytes were consumed by this action's completed warm/read.
        # This changes no header, unread neighbor, outer edge or global cache.
        original(str(shard), keys, source_stat)
        events, samples, errors = [], [], []
        stop = threading.Event()
        def sample():
            while not stop.wait(.02):
                try:
                    samples.append(memory())
                except Exception as exc:
                    errors.append(str(exc))
        def advice(path, selected, expected_stat=None):
            before = memory()
            started = time.perf_counter()
            if mode == 'per_key':
                for key in sorted(set(selected)):
                    original(path, [key], expected_stat)
            else:
                original(path, selected, expected_stat)
            events.append(dict(keys=len(selected), elapsed_s=time.perf_counter()-started,
                               before=before, after=memory()))
        ls._advise_consumed_safetensors_pages = advice
        initial = dict(memory=memory(), resident_payload_bytes=residency())
        worker = threading.Thread(target=sample, daemon=True)
        worker.start()
        started = time.time()
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], profile_memory=True,
                    record_shapes=True) as prof:
                with torch.profiler.record_function('source_read_and_page_release'):
                    values = read()
                    torch.cuda.synchronize()
            ended = time.time()
        finally:
            stop.set()
            worker.join()
            ls._advise_consumed_safetensors_pages = original
        final = dict(memory=memory(), resident_payload_bytes=residency())
        assert not errors
        equal = all(torch.equal(values[key], reference[key]) for key in keys)
        assert equal
        del values
        prof.export_chrome_trace(str(root/f'{index}-{mode}.trace.json'))
        arms.append(dict(mode=mode, start_unix=started, end_unix=ended, elapsed_s=ended-started,
                         initial=initial, final=final, exact_gpu_bytes=equal,
                         page_advice=events, memory_samples=samples))
        (root/'measurement.json').write_text(json.dumps(dict(arms=arms), indent=2)+'\n')
    original(str(shard), keys, source_stat)
    current = shard.stat()
    assert all(getattr(current,k) == getattr(source_stat,k) for k in
               ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))
    report = dict(arms=arms, source=str(shard), keys=keys, tensor_bytes=end-begin,
        payload_begin=begin, payload_end=end, advised_begin=first, advised_end=last,
        source_stat={k:getattr(source_stat,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')},
        torch=torch.__version__, cuda=torch.version.cuda,
        contract='Same 64 contiguous real BF16 expert tensors; ABBA original per-key vs coalesced consumed-page advice; exact GPU equality. No full-capture fit claim.')
    (root/'measurement.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('tensor_bytes','contract')}), flush=True)

if __name__ == '__main__':
    main()
