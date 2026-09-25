"""Read-ahead bytes leave the row's committed memory when they are freed (PQ #1291).

The IO engine's depth is a live reading of the row's committed memory, and
its reclaim frees read-ahead bytes so that a check can pass. Both work only if
freeing a value actually lowers that reading. On the GB10 torch build
(2.11+cu130, aarch64) a ``torch`` CPU tensor does not: ``libc10`` bundles
mimalloc as the CPU allocator, and mimalloc keeps the pages it frees. The
PrismaBuild probe of 2026-09-25 allocated 3.84 GB of ``torch.empty`` tensors
and freed them all; 3.85 GB stayed in RssAnon, where the same bytes in a
``bytearray`` went back. Stage B's first read-ahead row then evicted 1.95 GB
of renders to pass a check, the guard's reading did not move, and the row was
refused. So the engine never decodes a stream entry into a ``torch.empty``
style allocation: it reads each file into a sealed memfd, and the PWC decoder
maps the memfd (``torch.load(..., mmap=True)``), so freeing the tensor unmaps
the only reference and the kernel frees the pages.

This test runs the real path: real Torch archives the size of a Stage B
render, the PWC's own stream entries, and the committed-memory reading the
capture guard admits against (``memory_management.committed_cgroup_bytes``),
read from this process's own cgroup. It checks both ways bytes are freed: a
reclaim of entries read ahead, and a consumer that takes a group, drops it and
releases it. Each drop must be the freed files' pages, to within the page
rounding each memfd carries. It must run where it can read its cgroup; a
missing cgroup fails it rather than skipping it, because a skipped probe
certifies nothing.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import prismaquant.production_weight_cache as pwc
from prismaquant import io_engine
from prismaquant import memory_management as mm

PAGE = os.sysconf("SC_PAGE_SIZE")
# How exact a cgroup reading is: the kernel charges ``memory.current`` in
# per-CPU batches of MEMCG_CHARGE_BATCH (64) pages and flushes ``memory.stat``
# once the per-CPU deltas pass that batch on every CPU, so a reading is exact
# to within this many bytes and no closer.
READING_GRAIN = 64 * PAGE * os.cpu_count()
# A Stage B render file: 16 MiB of storage, as the GLM layer's renders are.
RENDER_SHAPE = (2048, 4096)
PER_GROUP = 8


@pytest.fixture(autouse=True)
def _no_residency_map(monkeypatch):
    monkeypatch.delenv("PRISMABUILD_RESIDENCY_MAP", raising=False)


def _own_cgroup() -> Path:
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    unified = [line.split("::", 1)[1] for line in lines if line.startswith("0::")]
    assert unified, f"no cgroup v2 membership in /proc/self/cgroup: {lines}"
    scope = Path("/sys/fs/cgroup") / unified[0].lstrip("/")
    assert (scope / "memory.stat").is_file(), f"cannot read {scope}/memory.stat"
    return scope


def _committed(scope: Path) -> dict:
    """The guard's reading: stat first, then ``memory.current``."""
    stat = mm.read_memory_stat(scope / "memory.stat")
    current = int((scope / "memory.current").read_text())
    return {"committed": mm.committed_cgroup_bytes(current, stat),
            "anon_shmem": stat["anon"] + stat["shmem"]}


def _pages(nbytes: int) -> int:
    return -(-nbytes // PAGE) * PAGE


def _renders(tmp_path, count):
    paths = {}
    for index in range(count):
        key = (f"model.layers.0.unit{index}", "NVFP4")
        paths[key] = tmp_path / f"unit{index}.pt"
        torch.save(torch.full(RENDER_SHAPE, float(index), dtype=torch.bfloat16), paths[key])
        # Written back before anything is measured: a dirty page is
        # committed, and writeback during a reading would read as a drop.
        with open(paths[key], "rb") as handle:
            os.fsync(handle.fileno())
    cache = pwc.ProductionWeightCache(
        weights={key: str(path) for key, path in paths.items()}, levers={})
    each = paths[next(iter(paths))].stat().st_size
    assert all(path.stat().st_size == each for path in paths.values())
    cache.enable_lru(count * each)
    return cache, list(paths), each


def _quiet(stream, timeout=60.0):
    with stream._cond:
        while stream._active or stream._gating:
            assert stream._cond.wait(timeout), "the stream never went quiet"
        return stream._held_actual


def test_freed_read_ahead_leaves_the_committed_reading(tmp_path):
    scope = _own_cgroup()
    procs = (scope / "cgroup.procs").read_text().split()
    cache, keys, each = _renders(tmp_path, 2 * PER_GROUP)
    entries = [entry for group in range(2)
               for entry in cache.retained_read_entries(
                   keys[group * PER_GROUP:(group + 1) * PER_GROUP], group=group)]
    budget = io_engine.FixedBudget(buffer_bytes=PER_GROUP * each, headroom=1 << 40)

    def check(label, before, after, dropped):
        # Each memfd holds its file's pages; nothing else of the entry may
        # stay behind, and nothing more than those pages may go, to within
        # the grain of the reading.
        low = dropped * each - READING_GRAIN
        high = dropped * _pages(each) + READING_GRAIN
        for measure in ("committed", "anon_shmem"):
            drop = before[measure] - after[measure]
            assert low <= drop <= high, (
                f"{label}: {measure} dropped {drop} bytes; freeing {dropped} "
                f"entries of {each} bytes must drop between {low} and {high} "
                f"(cgroup {scope}, {len(procs)} processes; before {before}, after {after})")

    with io_engine.read_stream(entries, budget=budget) as stream:
        held = _quiet(stream)
        # Each reading is taken with the stream paused, so no read of its own
        # lands between the two readings.
        with stream.paused():
            # A reclaim of the farthest entries read ahead.
            before = _committed(scope)
            freed = stream.reclaim(PER_GROUP // 2 * each)
            after = _committed(scope)
        assert freed == PER_GROUP // 2 * each
        check("reclaim", before, after, PER_GROUP // 2)
        # A consumer takes a group, drops it, and releases it.
        delivered = stream.take(0)
        assert [float(item.value[0, 0]) for item in delivered] == \
            [float(index) for index in range(PER_GROUP)]
        with stream.paused():
            before = _committed(scope)
            delivered = None
            after = _committed(scope)
        check("take, drop", before, after, PER_GROUP)
        stream.release()
    # Every render read ahead held its whole file's pages.
    assert held == 2 * PER_GROUP * each
