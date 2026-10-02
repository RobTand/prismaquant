"""Small PB-only kernel-sealed source delivery and CUDA-lifetime control."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import weakref

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def memfds():
    found = set()
    for name in os.listdir('/proc/self/fd'):
        try:
            if '/memfd:pq-io' in os.readlink('/proc/self/fd/'+name):
                found.add(name)
        except FileNotFoundError:
            pass
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--sha256')
    p.add_argument('--bytes', type=int)
    p.add_argument('--data-manifest-sha256')
    p.add_argument('--write-fixture', action='store_true')
    args = p.parse_args()
    if args.write_fixture:
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        if args.fixture.exists():
            raise RuntimeError('fixture already exists')
        save_file({'weight':torch.arange(1024*1024, dtype=torch.float32).reshape(1024,1024)}, str(args.fixture))
        print(json.dumps(dict(path=str(args.fixture), bytes=args.fixture.stat().st_size,
            sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest())), flush=True)
        return
    from prismaquant import io_engine
    from prismaquant.residency_map import bind_residency_manifest, residency_resolver
    from prismaquant.staged_tier_policy import activate_staged_tier_policy
    from prismaquant.staged_whole_file import read_staged_sealed_file
    activate_staged_tier_policy('ram,ssd')
    bind_residency_manifest(args.data_manifest_sha256)
    torch.set_num_threads(1); torch.cuda.init()
    before = memfds()
    lifecycle = []
    for stop_after_copy in (False, True):
        buffer = read_staged_sealed_file(args.fixture,args.sha256,args.bytes,label='gpu-lifetime-fixture')
        source = alias = pinned = result = event = stream = None
        try:
            buffer.require_sealed()
            with safe_open(buffer.path,framework='pt',device='cpu') as handle:
                source = handle.get_tensor('weight')
                alias = source[5:]
            reference = weakref.ref(source)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                pinned = alias.pin_memory()
                result = torch.empty_like(alias, device='cuda')
                result.copy_(pinned,non_blocking=True)
                event = torch.cuda.Event(); event.record(stream)
            assert reference() is source
            buffer.require_sealed()
            if stop_after_copy:
                raise KeyboardInterrupt('bounded cancellation after async enqueue')
            event.synchronize()
            assert torch.equal(result.cpu(),alias)
            lifecycle.append('equal')
        except KeyboardInterrupt:
            lifecycle.append('cancelled_after_enqueue')
        finally:
            if stream is not None:
                stream.synchronize()
            source = alias = pinned = result = event = stream = None
            gc.collect()
            buffer.close()
        gc.collect()
        assert memfds() == before
    failures = []
    for case in ('digest','cancel'):
        original = io_engine.SealedBuffer.fill
        if case == 'cancel':
            def cancelled(raw,fd):
                raise KeyboardInterrupt('bounded cancellation fixture')
            io_engine.SealedBuffer.fill = cancelled
        try:
            try:
                raw = read_staged_sealed_file(args.fixture,
                    args.sha256 if case == 'cancel' else '0'*64,args.bytes,label='failure-fixture')
            except (KeyboardInterrupt,Exception) as error:
                failures.append(dict(case=case,error=type(error).__name__))
            else:
                raw.close()
                raise AssertionError('failed delivery returned decoder-visible material')
        finally:
            io_engine.SealedBuffer.fill = original
        gc.collect()
        assert memfds() == before
    assert len(failures) == 2
    print(json.dumps(dict(schema='prismaquant.pq1934_sealed_cuda_lifetime.v1', passed=True,
        gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
        affinity=sorted(os.sched_getaffinity(0)),source_bytes=args.bytes,
        kernel_sealed=True,async_copy_equal=True,native_alias_held_until_completion=True,
        memfd_cleanup=True,lifecycle=lifecycle,failures=failures,residency=residency_resolver().report())),flush=True)


if __name__=='__main__':main()
