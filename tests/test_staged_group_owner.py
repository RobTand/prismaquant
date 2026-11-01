"""Both real group consumers update the same lease-owner accounting."""
import hashlib
from pathlib import Path

import torch

from test_strict_reader_tier_enforcement import (
    _count_sdk_calls, _exact_entries, _forget_state, _leased_fixture, _pb,
    _pins_live, _read_exact, _stage_root, _stage_whole,
)


def test_engine_and_activation_group_reads_share_exact_process_counters(tmp_path, monkeypatch):
    from prismaquant import io_engine, staged_lease

    monkeypatch.setenv("PRISMAQUANT_LAYER_READ_THREADS", "1")
    refs, tensors = _exact_entries(tmp_path, 4)
    root = _stage_root(tmp_path)
    resolver, consumer, _mover = _leased_fixture(
        tmp_path, monkeypatch,
        {str(i): (Path(ref.path), _stage_whole(root, Path(ref.path)), None)
         for i, ref in enumerate(refs)})
    calls = _count_sdk_calls(monkeypatch, _pb()[0])
    before = staged_lease.exact_lease_counters()
    live = []
    try:
        assignments = io_engine.pin_group([(ref.path, ref.sha256) for ref in refs], live)
        assert len(live) == 1 and len(assignments) == len(refs)
        for ref in refs:
            raw, receipt, _signature = io_engine.read_file(
                Path(ref.path), ref.file_bytes,
                staged=assignments[ref.path][2], lease=assignments[ref.path])
            assert hashlib.sha256(raw).hexdigest() == ref.sha256 == receipt["sha256"]
            assert raw == Path(ref.path).read_bytes()
        # Per-entry descriptor closure must leave the borrowed group pinned.
        assert len(_pins_live(tmp_path, consumer)) == 1
    finally:
        io_engine.unpin(live)
    assert _pins_live(tmp_path, consumer) == []
    engine = staged_lease.exact_lease_counters()
    assert {key: engine[key] - before[key] for key in engine} == {
        "windows_batched": 1, "entries_batched": 4, "entries_single": 0,
        "batch_fallbacks": 0}
    actual = _read_exact(refs)
    for ref, expected in zip(refs, tensors):
        assert torch.equal(actual[ref], expected)
    activation = staged_lease.exact_lease_counters()
    assert {key: activation[key] - engine[key] for key in activation} == {
        "windows_batched": 1, "entries_batched": 4, "entries_single": 0,
        "batch_fallbacks": 0}
    assert calls == {"acquire_for": 2, "release": 2, "covers_for_keys": 2}
    assert resolver.report()["bytes_from_pool"] == 0
    assert _pins_live(tmp_path, consumer) == []
