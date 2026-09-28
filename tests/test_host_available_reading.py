"""The guard's host reading counts free pages on the per-CPU lists (PQ #1431).

A freed page can park on the kernel's per-CPU free list, where ``MemAvailable``
does not count it until the kernel trims the list. One host reading serves the
guard and the GB10 reclaim probe: ``MemAvailable`` plus those pages. A free
that lands entirely on the lists must therefore count as help to the guard.
"""
from __future__ import annotations

import os

import pytest
import torch

from prismaquant import io_spans, memory_management as mm

GiB = 1024 ** 3
PAGE = os.sysconf("SC_PAGE_SIZE")


def _meminfo(path, available, total=121 * GiB):
    path.write_text(f"MemTotal:       {total // 1024} kB\n"
                    f"MemFree:        1 kB\n"
                    f"MemAvailable:   {available // 1024} kB\n")


def _zoneinfo(path, per_cpu_bytes):
    """Two zones, two CPUs each; the counts sum to ``per_cpu_bytes``."""
    pages = per_cpu_bytes // PAGE
    share, extra = divmod(pages, 4)
    counts = [share + extra, share, share, share]
    body = []
    for zone, (first, second) in (("DMA", counts[:2]), ("Normal", counts[2:])):
        body.append(f"Node 0, zone   {zone}\n  pages free     123\n"
                    f"        min      10\n  pagesets\n")
        for cpu, count in enumerate((first, second)):
            body.append(f"    cpu: {cpu}\n              count: {count}\n"
                        f"              high:  378\n              batch: 63\n")
        body.append("  vm stats threshold: 32\n")
    path.write_text("".join(body))


@pytest.fixture
def proc(tmp_path, monkeypatch):
    meminfo, zoneinfo = tmp_path / "meminfo", tmp_path / "zoneinfo"
    monkeypatch.setattr(io_spans, "PROC_MEMINFO", meminfo, raising=False)
    monkeypatch.setattr(io_spans, "PROC_ZONEINFO", zoneinfo, raising=False)
    return meminfo, zoneinfo


def test_a_free_that_lands_on_the_per_cpu_lists_helps_the_guard(
        tmp_path, monkeypatch, proc):
    meminfo, zoneinfo = proc
    scope = tmp_path / "cgroup" / "scope"
    scope.mkdir(parents=True)
    (scope / "memory.max").write_text(str(28 * GiB))
    (scope / "memory.current").write_text(str(13 * GiB))
    (scope / "memory.stat").write_text(
        "anon 0\nfile 0\nshmem 0\nfile_dirty 0\nfile_writeback 0\n")
    membership = tmp_path / "self.cgroup"
    membership.write_text("0::/scope\n")
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 20 * GiB)
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=68 * GiB,
                                  aggregate_envelope=True,
                                  cgroup_root=tmp_path / "cgroup",
                                  membership=membership)
    # MemAvailable is one GiB under what the phase needs; the free pages are
    # not on MemAvailable's books until the kernel trims the lists.
    _meminfo(meminfo, guard.host_floor_bytes + GiB - GiB)
    _zoneinfo(zoneinfo, 0)
    asked = []

    def reclaim(shortfall):
        asked.append(shortfall)
        _zoneinfo(zoneinfo, 2 * GiB)  # MemAvailable is unchanged
        return 2 * GiB

    guard.add_reclaimer(reclaim, lowers={"available"})
    record = guard.check("phase", reserve_bytes=GiB)
    assert asked == [GiB]
    assert record["host_available_bytes"] == guard.host_floor_bytes + 2 * GiB


def test_the_host_reading_is_memavailable_plus_the_per_cpu_pages(tmp_path, proc):
    meminfo, zoneinfo = proc
    _meminfo(meminfo, 10 * GiB)
    _zoneinfo(zoneinfo, 3 * GiB)
    assert io_spans.per_cpu_free_bytes() == 3 * GiB
    assert io_spans.host_available_bytes() == 13 * GiB
    assert mm._host_memory_info() == (13 * GiB, 121 * GiB)


def test_a_host_without_zoneinfo_reads_memavailable_alone(tmp_path, proc):
    meminfo, zoneinfo = proc
    _meminfo(meminfo, 10 * GiB)
    assert not zoneinfo.exists()
    assert io_spans.per_cpu_free_bytes() == 0
    assert io_spans.host_available_bytes() == 10 * GiB


def test_a_host_without_meminfo_is_unreadable(tmp_path, proc):
    meminfo, zoneinfo = proc
    _zoneinfo(zoneinfo, GiB)
    with pytest.raises(OSError):
        io_spans.host_available_bytes()
    assert mm._host_memory_info() is None
