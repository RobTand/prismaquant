"""Both real group consumers update the same lease-owner accounting."""
import hashlib
import os
from pathlib import Path

import pytest
import torch

from staged_native_fixtures import bind, native_staged_state, publish_inputs
from prismaquant.staged_tier_policy import staged_tier_policy_test_context
from test_streamed_metadata_staged_reads import _deny_pool_opens


pytestmark = pytest.mark.usefixtures('native_staged_state')


def test_engine_and_activation_group_reads_share_exact_process_counters(tmp_path, monkeypatch):
    from prismaquant import io_engine, staged_lease
    from prismaquant.perturbed_x_cache import (
        write_exact_activation_cache_entry, prefetch_exact_activation_cache_entries)

    monkeypatch.setenv("PRISMAQUANT_LAYER_READ_THREADS", "1")
    directory = tmp_path / 'entries'
    directory.mkdir()
    tensors = [torch.arange(32, dtype=torch.float32).reshape(8, 4) + 100 * i
               for i in range(4)]
    refs = [write_exact_activation_cache_entry(directory, f'entry-{i}', tensor,
        identity={'session': 'strict-tier-session', 'slot': f'entry-{i}',
                  'kind': 'boundary', 'coordinates': {'batch': 0, 'boundary': 0, 'probe': None}},
        max_tensor_bytes=128, max_file_bytes=128 + 65536)
        for i, tensor in enumerate(tensors)]
    wire = {ref.path: Path(ref.path).read_bytes() for ref in refs}
    resolver = bind(monkeypatch, publish_inputs(tmp_path, wire))
    # Exact references retain canonical metadata fences but never read its bytes.
    _deny_pool_opens(monkeypatch, directory,
                     names=[Path(ref.path).name for ref in refs])
    before = staged_lease.exact_lease_counters()
    live = []
    descriptors = []
    real_open = staged_lease.LeaseWindow.open

    def observed(window, key):
        fd, serving = real_open(window, key)
        descriptors.append(fd)
        assert resolver._windows == 1
        return fd, serving

    monkeypatch.setattr(staged_lease.LeaseWindow, 'open', observed)
    policy = staged_tier_policy_test_context('ssd')
    policy.__enter__()
    try:
        assignments = io_engine.pin_group([(ref.path, ref.sha256) for ref in refs], live)
        assert len(live) == 1 and len(assignments) == len(refs)
        for ref in refs:
            raw, receipt, _signature = io_engine.read_file(
                Path(ref.path), ref.file_bytes,
                staged=assignments[ref.path][2], lease=assignments[ref.path])
            assert hashlib.sha256(raw).hexdigest() == ref.sha256 == receipt["sha256"]
            assert raw == wire[ref.path]
            with pytest.raises(OSError):
                os.fstat(descriptors[-1])
        # Per-entry descriptor closure must leave the borrowed group pinned.
        assert resolver._windows == 1 and not live[0]._fds
        with pytest.raises(RuntimeError, match='live reader windows'):
            resolver.close()
    finally:
        io_engine.unpin(live)
        policy.__exit__(None, None, None)
    assert resolver._windows == 0
    engine = staged_lease.exact_lease_counters()
    assert {key: engine[key] - before[key] for key in engine} == {
        "windows_batched": 1, "entries_batched": 4, "entries_single": 0,
        "batch_fallbacks": 0}
    with staged_tier_policy_test_context('ssd'):
        with prefetch_exact_activation_cache_entries(refs, max_tensor_bytes=4 * 128,
                expected_session='strict-tier-session', release_file_pages=False) as window:
            actual = {ref: window._tensors[ref].clone() for ref in refs}
    for ref, expected in zip(refs, tensors):
        assert torch.equal(actual[ref], expected)
    activation = staged_lease.exact_lease_counters()
    assert {key: activation[key] - engine[key] for key in activation} == {
        "windows_batched": 1, "entries_batched": 4, "entries_single": 0,
        "batch_fallbacks": 0}
    assert len(descriptors) == 8
    assert resolver.report()["bytes_from_pool"] == 0
    assert resolver._windows == 0
