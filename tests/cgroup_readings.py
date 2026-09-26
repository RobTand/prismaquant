"""The capture guard's readings of this process, for tests that free bytes (PQ #1291, #1383).

A test that frees bytes and asks whether a reading moved reads the same
things the guard reads (``CaptureMemoryGuard._observe``): the cgroup's
committed bytes, the CUDA caching allocator's reservation and the host's
MemAvailable. It must run where it can read its own cgroup; a missing cgroup
fails it rather than skipping it, because a skipped probe certifies nothing.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

import prismaquant.production_weight_cache as pwc
from prismaquant import memory_management as mm

PAGE = os.sysconf("SC_PAGE_SIZE")
# How exact a cgroup reading is: the kernel charges ``memory.current`` in
# per-CPU batches of MEMCG_CHARGE_BATCH (64) pages and flushes ``memory.stat``
# once the per-CPU deltas pass that batch on every CPU, so a reading is exact
# to within this many bytes and no closer.
READING_GRAIN = 64 * PAGE * os.cpu_count()


def own_cgroup() -> Path:
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    unified = [line.split("::", 1)[1] for line in lines if line.startswith("0::")]
    assert unified, f"no cgroup v2 membership in /proc/self/cgroup: {lines}"
    scope = Path("/sys/fs/cgroup") / unified[0].lstrip("/")
    assert (scope / "memory.stat").is_file(), f"cannot read {scope}/memory.stat"
    return scope


def committed(scope: Path) -> dict:
    """The guard's cgroup reading: stat first, then ``memory.current``."""
    stat = mm.read_memory_stat(scope / "memory.stat")
    current = int((scope / "memory.current").read_text())
    return {"committed": mm.committed_cgroup_bytes(current, stat),
            "anon_shmem": stat["anon"] + stat["shmem"],
            "shmem": stat["shmem"], "anon": stat["anon"]}


def pages(nbytes: int) -> int:
    return -(-nbytes // PAGE) * PAGE


def renders(tmp_path, count, shape):
    """``count`` BF16 renders of ``shape`` on disk, and a cache that reads them."""
    paths = {}
    for index in range(count):
        key = (f"model.layers.0.unit{index:02d}", "NVFP4")
        paths[key] = tmp_path / f"unit{index:02d}.pt"
        torch.save(torch.full(shape, float(index), dtype=torch.bfloat16), paths[key])
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


def quiet(stream, timeout=60.0):
    """Wait until ``stream`` has no read in flight; the bytes it holds."""
    with stream._cond:
        while stream._active or stream._gating:
            assert stream._cond.wait(timeout), "the stream never went quiet"
        return stream._held_actual
