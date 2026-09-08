"""The existing Torch archive can release completed records without new bytes."""
from pathlib import Path

import pytest
import torch


def test_export_releases_stable_tensor_prefixes_with_exact_archive_bytes(tmp_path, monkeypatch):
    from prismaquant import perturbed_x_cache, tessera_campaign as campaign
    hessians = {f'unit.{index}': torch.arange(128*128, dtype=torch.float32).reshape(128, 128)+index
                for index in range(4)}
    hessians['shared.view'] = hessians['unit.0'].t()
    calls = []
    original = perturbed_x_cache.release_activation_cache_file_pages
    def observe(path, *, expected_stat):
        calls.append((Path(path).name, expected_stat.st_size))
        return original(path, expected_stat=expected_stat)
    monkeypatch.setattr(perturbed_x_cache, 'release_activation_cache_file_pages', observe)
    outcomes = []
    for bounded in (False, True):
        directory = tmp_path/str(bounded)
        directory.mkdir()
        path, _scales, digest = campaign.write_export_inputs(directory,
            hessians=hessians, hessian_rows=dict.fromkeys(hessians, 512),
            hessian_identity={'fit_ids_sha256': 'unchanged-draw'},
            static_scales={}, static_scale_policy='fixture', release_file_pages=bounded)
        outcomes.append((path.read_bytes(), digest))
    assert outcomes[0] == outcomes[1]
    prefixes = [size for name, size in calls if name == 'hessian_capture.pt.tmp']
    assert len(prefixes) >= 4, 'the writer retained the whole file until publication'
    assert prefixes == sorted(prefixes)
    assert prefixes[0] < len(outcomes[1][0])//2
    assert calls[-1] == ('hessian_capture.pt', len(outcomes[1][0]))


def test_native_hessian_writer_profile(tmp_path, monkeypatch):
    """Opt-in, attributed writer qualification, never a full-GLM fit claim."""
    import hashlib
    import json
    import os
    import threading
    import time
    from prismaquant import perturbed_x_cache, tessera_campaign as campaign
    from prismaquant.memory_management import CaptureMemoryGuard
    destination = os.environ.get('PRISMAQUANT_HESSIAN_WRITER_PROFILE')
    if not destination:
        pytest.skip('explicit native writer measurement only')
    assert torch.cuda.is_available()
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    hessians = {f'unit.{index}': torch.full((4096, 4096), index/16,
                device='cuda', dtype=torch.float32) for index in range(16)}
    torch.cuda.synchronize()
    guard = CaptureMemoryGuard('cuda')
    events, samples = [], []
    original = perturbed_x_cache.release_activation_cache_file_pages
    def advise(path, *, expected_stat):
        events.append(dict(name=Path(path).name, bytes=expected_stat.st_size,
                           before=guard.check('before_prefix_advice')))
        result = original(path, expected_stat=expected_stat)
        events[-1]['after'] = guard.check('after_prefix_advice')
        return result
    monkeypatch.setattr(perturbed_x_cache, 'release_activation_cache_file_pages', advise)
    stop = threading.Event()
    errors = []
    def sample():
        while not stop.wait(.02):
            try:
                stat = dict(line.split() for line in (guard.scope/'memory.stat').read_text().splitlines())
                samples.append(dict(time_unix=time.time(),
                    current_bytes=int((guard.scope/'memory.current').read_text()),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(),
                    **{name: int(stat[name]) for name in ('anon', 'file', 'file_dirty', 'file_writeback')}))
            except Exception as error:
                errors.append(str(error))
    def process_io():
        return dict((key.rstrip(':'), int(value)) for key, value in
                    (line.split() for line in Path('/proc/self/io').read_text().splitlines()))
    before = guard.check('before_writer')
    io_before = process_io()
    started = time.time()
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], profile_memory=True,
                record_shapes=True) as prof:
            path, _, content_digest = campaign.write_export_inputs(root,
                hessians=hessians, hessian_rows=dict.fromkeys(hessians, 512),
                hessian_identity={'fit_ids_sha256': 'writer-qualification-fixed'},
                static_scales={}, static_scale_policy='fixture', release_file_pages=True,
                resource_check=guard.check)
        torch.cuda.synchronize()
    finally:
        ended = time.time()
        stop.set()
        sampler.join()
    after = guard.check('after_writer')
    assert not errors
    io_after = process_io()
    prof.export_chrome_trace(str(root/'writer.trace.json'))
    file_sha = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8*1024**2):
            file_sha.update(block)
    perturbed_x_cache.release_activation_cache_file_pages(path, expected_stat=path.stat())
    report = dict(start_unix=started, end_unix=ended, elapsed_s=ended-started,
        workload=dict(tensors=16, shape=[4096,4096], dtype='float32', device='cuda',
                      tensor_bytes=16*4096*4096*4),
        before=before, after=after, page_advice=events, memory_samples=samples,
        io_delta={key: io_after[key]-io_before[key] for key in io_before},
        file_bytes=path.stat().st_size, file_sha256=file_sha.hexdigest(),
        content_digest=content_digest, torch=torch.__version__, cuda=torch.version.cuda,
        claim='writer ownership qualification; no full GLM fit or throughput claim')
    (root/'measurement.json').write_text(json.dumps(report, indent=2)+'\n')
