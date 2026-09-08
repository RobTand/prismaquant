"""Bounded diagnostic of the capture materialization/write/release boundary."""
import cProfile
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from experiments.glm_full_capture_profile import CaptureObserver
from prismaquant.perturbed_x_cache import write_activation_cache_entry, release_activation_cache_file_pages


def main():
    out = Path(sys.argv[1])
    with CaptureObserver(out, profile_layers=()) as observer:
        records = []
        (out/'torch-config.txt').write_text(torch.__config__.show())
        (out/'process-maps.txt').write_text(Path('/proc/self/maps').read_text())
        (out/'memory-api.json').write_text(json.dumps([n for n in dir(torch._C) if any(k in n.lower() for k in ('cache','alloc','malloc','host'))]))
        def snapshot(label, refs=()):
            stats = dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
            result = dict(label=label, time=time.time(), cgroup_bytes=int(Path('/sys/fs/cgroup/memory.current').read_text()),
                          memory_stat={k:int(v) for k,v in stats.items()},
                          smaps=Path('/proc/self/smaps_rollup').read_text(),
                          cuda_allocated=torch.cuda.memory_allocated(), cuda_reserved=torch.cuda.memory_reserved(),
                          live_storage_count=sum(not ref.expired() for ref in refs),
                          gc_count=gc.get_count(), host_allocator=torch.cuda.memory.host_memory_stats())
            records.append(result)
            print(json.dumps(result), flush=True)
            (out/'memory.json').write_text(json.dumps(records, indent=2)+'\n')
        def materialize():
            groups = {f'{i}.{kind}': torch.ones((cols,cols),device='cuda')
                      for i in range(32) for kind,cols in [('gate',4096),('down',2048)]}
            snapshot('device_groups')
            acts, hess = {}, {}
            for name in list(groups):
                value = groups.pop(name).cpu()
                hess[name] = value
                acts[name] = torch.ones((512,value.shape[0]))
                if name.endswith('gate'):
                    hess[name+'.up'] = value.clone()
                    acts[name+'.up'] = acts[name].clone()
                del value
                torch.cuda.empty_cache()
            return acts,hess
        snapshot('initial')
        profiler = cProfile.Profile()
        with profiler:
            acts,hess = materialize()
            refs = [StorageWeakRef(v.untyped_storage()) for d in (acts,hess) for v in d.values()]
            snapshot('materialized',refs)
            observer.result['pinned_outputs'] = {name: value.is_pinned() for name,value in hess.items()}
            with tempfile.TemporaryDirectory(prefix='capture-lifetime-') as tmp:
                for name in sorted(acts):
                    path = write_activation_cache_entry(tmp,name,acts[name],hessian=hess[name],durable=True)
                    with path.open('rb') as handle:
                        hashlib.file_digest(handle,'sha256')
                    release_activation_cache_file_pages(path, expected_stat=path.stat())
                snapshot('written',refs)
                del acts,hess
                torch.cuda.empty_cache()
                snapshot('outputs_deleted',refs)
                gc.collect()
                torch.cuda.empty_cache()
                snapshot('cycles_collected',refs)
                ctypes.CDLL(None).malloc_trim(0)
                snapshot('heap_trimmed',refs)
                torch._C._host_emptyCache()
                snapshot('torch_host_cache_released',refs)
                torch._C._accelerator_emptyHostCache()
                snapshot('torch_accelerator_host_cache_released',refs)
        profiler.dump_stats(out/'lifetime.pstats')
        observer.result['memory_stages'] = records
        observer.result['serialization_sha256'] = hashlib.sha256(Path(torch.serialization.__file__).read_bytes()).hexdigest()


if __name__ == '__main__':
    main()
