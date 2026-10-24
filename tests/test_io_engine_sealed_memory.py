"""Real PWC read-ahead owns sealed mappings, not allocator-retained copies.

PQ #1291 needs disposal of the backing objects, not just zero byte counters.
Observe this process's uniquely labelled memfds: reclaim unmaps only the
farthest entries, take transfers ownership, and consumer drop unmaps them.
No observer retains tensors or storages. This proves mapping lifetime, not an
exact physical-memory delta in a cgroup shared with unrelated processes.
Fixture-based committed-memory semantics live in test_committed_memory_1157.py.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import itertools
import os
from pathlib import Path
import threading

import pytest
import torch

from prismaquant import io_engine
from prismaquant.production_weight_cache import ProductionWeightCache

RENDER_SHAPE = (128, 128)
PER_GROUP = 2


@pytest.fixture(autouse=True)
def _no_residency_map(monkeypatch):
    monkeypatch.delenv("PRISMABUILD_RESIDENCY_MAP", raising=False)


def test_freed_read_ahead_leaves_the_committed_reading(tmp_path):
    paths = {}
    for index in range(2 * PER_GROUP):
        key = (f"model.layers.0.unit{index:02d}", "NVFP4")
        paths[key] = tmp_path / f"unit{index:02d}.pt"
        torch.save(torch.full(RENDER_SHAPE, float(index), dtype=torch.bfloat16), paths[key])
    cache = ProductionWeightCache(
        weights={key: str(path) for key, path in paths.items()}, levers={})
    keys = list(paths)
    each = paths[keys[0]].stat().st_size
    assert all(path.stat().st_size == each for path in paths.values())
    cache.enable_lru(len(keys) * each)
    entries = [entry for group in range(2)
               for entry in cache.retained_read_entries(
                   keys[group * PER_GROUP:(group + 1) * PER_GROUP], group=group)]

    # Labels are never recycled during this test. No inode, fd number or
    # virtual address is treated as backing-object identity.
    prefix = "pq-test-" + hashlib.sha256(os.fsencode(tmp_path)).hexdigest()[:24] + "-"
    labels = itertools.count()
    decoded = {}
    label_lock = threading.Lock()

    def owned_mappings():
        return [line.split(maxsplit=5)[5] for line in
                Path("/proc/self/maps").read_text().splitlines()
                if f"/memfd:{prefix}" in line]

    def owned_descriptors():
        targets = []
        for fd in Path("/proc/self/fd").iterdir():
            try:
                target = os.readlink(fd)
            except FileNotFoundError:  # the directory iterator's own fd
                continue
            if target.startswith(f"/memfd:{prefix}"):
                targets.append(target)
        return targets

    create = io_engine._create_memfd

    def labelled_memfd(name, flags):
        with label_lock:
            label = prefix + str(next(labels))
        return create(label, flags)

    def observe_decoder(index, decode):
        def observed(raw, receipt, staged):
            assert isinstance(raw, io_engine.SealedBuffer)
            with open(raw.path, "rb") as handle:
                assert io_engine.kernel_seal_bits(handle.fileno()) & 0xF == 0xF
            label = os.readlink(raw.path)
            value = decode(raw, receipt, staged)  # the real PWC/Torch mmap path
            with label_lock:
                decoded[index] = label  # strings only; never retain the value
            return value
        return observed

    entries = [replace(entry, decoder=observe_decoder(i, entry.decoder))
               for i, entry in enumerate(entries)]
    budget = io_engine.FixedBudget(buffer_bytes=PER_GROUP * each, headroom=1 << 40)
    assert owned_mappings() == []
    assert owned_descriptors() == []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io_engine, "_create_memfd", labelled_memfd)
        with io_engine.read_stream(entries, budget=budget) as stream:
            with stream._cond:
                assert stream._cond.wait_for(
                    lambda: not stream._active and not stream._gating, timeout=60
                ), "reader never published completion"
            with stream.paused():
                assert stream.held_bytes() == len(entries) * each
                assert stream.held_actual_bytes() == len(entries) * each
                assert set(decoded) == set(range(len(entries)))
                assert len(set(decoded.values())) == len(entries)
                assert sorted(owned_mappings()) == sorted(decoded.values())
                # The raw buffer descriptors have closed; only Torch mappings
                # own the sealed files now, independently of byte accounting.
                assert owned_descriptors() == []
                dropped = PER_GROUP // 2
                assert stream.reclaim(dropped * each) == dropped * each
                remaining = [decoded[i] for i in range(len(entries) - dropped)]
                assert sorted(owned_mappings()) == sorted(remaining)
                assert stream.counters["evictions"] == dropped
                assert stream.held_bytes() == (len(entries) - dropped) * each
                assert stream.held_actual_bytes() == (len(entries) - dropped) * each

                delivered = stream.take(0)
                assert len(delivered) == PER_GROUP
                assert sorted(owned_mappings()) == sorted(remaining)
                assert all(torch.equal(item.value, torch.full(
                    RENDER_SHAPE, float(index), dtype=torch.bfloat16))
                    for index, item in enumerate(delivered))
                assert stream.held_bytes() == (PER_GROUP - dropped) * each
                assert stream.held_actual_bytes() == (PER_GROUP - dropped) * each
                delivered = None
                assert sorted(owned_mappings()) == sorted(
                    decoded[i] for i in range(PER_GROUP, len(entries) - dropped))
                stream.release()
                # Keep read-ahead paused so reclaimed entries cannot be read
                # again before closure proves disposal of the remaining group.
                stream.close()
                assert stream.held_bytes() == stream.held_actual_bytes() == 0
        assert owned_mappings() == []
        assert owned_descriptors() == []
