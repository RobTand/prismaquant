"""What each Stage B reclaimer's frees lower, read on the GB10 (PQ #1383).

Before the capture guard refuses a check, it asks its reclaimers to free
bytes, and then it reads the process again. Each reclaimer is registered with
the readings its frees lower (``CaptureMemoryGuard.add_reclaimer(...,
lowers=...)``): the cgroup's committed bytes, the CUDA caching allocator's
reservation and the host's MemAvailable. The guard asks a reclaimer only for
a refused term that reads one of those. A declaration that names a reading
the frees do not move makes the guard ask a step that cannot help, and one
that leaves out a reading they do move makes it skip a step that could.
prof-2 of #1348 was refused because the render device cache's frees counted
against a cgroup shortfall, and on the GB10 a CUDA allocation is not charged
to the memcg.

This test frees about a gigabyte through each reclaimer that the Stage B
replay registers (``joint_statistics_replay.register_replay_reclaimers``).
It reads all three before and after each free, and holds each declaration to
what moved:

- the render stream (``io_engine.ReadStream.reclaim``): renders read ahead
  into sealed memfds (#1315);
- the render device cache (``RetainedRenderDeviceCache.reclaim``): CUDA
  allocations;
- the spill replay (``StageBReplaySpill.reclaim_replay``): chunks read ahead
  into pinned host buffers, with the caching host allocator's idle blocks
  handed back. On the GB10 these are shmem charged to the cgroup.

The readings after a free are taken at once, as the guard takes them. A
reading moved when it moved by more than half of what was freed, the
midpoint between "all of it" and "none of it". The cgroup reading is also
held to the grain the kernel keeps it to: the render stream's and the spill
replay's drops must be the freed pages, and the device cache's must be none.
MemAvailable is the whole host's reading, and other processes move it too, so
it is held only to the midpoint.

Each line also records the free pages on the kernel's per-CPU lists, which
MemAvailable does not count, and both host readings again two seconds later.
Neither is held to anything: they say where freed pages went that
MemAvailable did not show.

It needs CUDA and its own cgroup, and fails without either rather than
skipping: a skipped probe certifies nothing. Each reading is printed as one
``reclaim-reading`` JSON line, so a run with ``-s`` is the receipt.
"""
from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prismaquant import io_engine
from prismaquant import joint_replay_spill as spill_mod
from prismaquant import joint_statistics_replay as replay
from prismaquant import memory_management as mm

from cgroup_readings import PAGE, READING_GRAIN, committed, own_cgroup, pages, quiet, renders

#: A 32 MiB BF16 render; 32 of them are the gigabyte each reclaimer holds.
SHAPE = (4096, 4096)
COUNT = 32
PER_GROUP = 8
#: Freed by each reclaim: three quarters of what is held, so a reclaimer
#: that frees farthest first leaves some of its values behind.
FREED_FRACTION = (3, 4)
#: How long after a free the host readings are taken again: two of the
#: kernel's vmstat intervals, in which it trims the per-CPU lists.
SETTLE_SECONDS = 2.0


@pytest.fixture(autouse=True)
def _no_residency_map(monkeypatch):
    monkeypatch.delenv("PRISMABUILD_RESIDENCY_MAP", raising=False)


@pytest.fixture
def device():
    assert torch.cuda.is_available(), "this probe measures the GB10's CUDA allocations"
    return torch.device("cuda")


def _host():
    """MemAvailable, and the free pages on the kernel's per-CPU lists."""
    host = mm._host_memory_info()
    assert host is not None, "MemAvailable is unreadable"
    lines = Path("/proc/zoneinfo").read_text().splitlines()
    per_cpu = sum(int(line.split()[1]) for line in lines
                  if line.strip().startswith("count:"))
    return {"available": host[0], "per_cpu_free": per_cpu * PAGE}


def _reading(scope, device):
    """The guard's three readings, in its order, with the rest of the cgroup's."""
    torch.cuda.synchronize(device)
    reading = committed(scope)
    reading["reserved"] = int(torch.cuda.memory_reserved(device))
    reading.update(_host())
    return reading


def _settled():
    time.sleep(SETTLE_SECONDS)
    return _host()


def _moved(name, before, after, settled, freed):
    """Which readings the free lowered, each by more than half of ``freed``."""
    drops = {"committed": before["committed"] - after["committed"],
             "reserved": before["reserved"] - after["reserved"],
             "available": after["available"] - before["available"]}
    lowered = frozenset(reading for reading, drop in drops.items() if 2 * drop > freed)
    print("reclaim-reading " + json.dumps(dict(
        reclaimer=name, freed_bytes=freed, drops=drops, lowered=sorted(lowered),
        per_cpu_free_rise=after["per_cpu_free"] - before["per_cpu_free"],
        settled=dict(available_rise=settled["available"] - before["available"],
                     per_cpu_free_rise=settled["per_cpu_free"] - before["per_cpu_free"],
                     seconds=SETTLE_SECONDS),
        before=before, after=after, reading_grain_bytes=READING_GRAIN), sort_keys=True))
    return drops, lowered


def _freed(total):
    return total * FREED_FRACTION[0] // FREED_FRACTION[1]


def test_a_render_stream_reclaim_lowers_what_it_declares(tmp_path, device):
    scope = own_cgroup()
    cache, keys, each = renders(tmp_path, COUNT, SHAPE)
    entries = [entry for group in range(COUNT // PER_GROUP)
               for entry in cache.retained_read_entries(
                   keys[group * PER_GROUP:(group + 1) * PER_GROUP], group=group)]
    budget = io_engine.FixedBudget(buffer_bytes=PER_GROUP * each, headroom=1 << 40)
    with io_engine.read_stream(entries, budget=budget) as stream:
        assert quiet(stream) == COUNT * each
        with stream.paused():
            before = _reading(scope, device)
            freed = stream.reclaim(_freed(COUNT * each))
            after = _reading(scope, device)
            settled = _settled()
    dropped = freed // each
    assert freed == dropped * each > 0
    drops, lowered = _moved("render_stream", before, after, settled, freed)
    # The freed memfds' pages, to within the cgroup reading's grain.
    assert freed - READING_GRAIN <= drops["committed"] <= dropped * pages(each) + READING_GRAIN
    assert drops["reserved"] == 0
    assert lowered == replay.RENDER_STREAM_RECLAIM_LOWERS


def test_a_render_device_cache_reclaim_lowers_what_it_declares(device):
    scope = own_cgroup()
    source = torch.zeros(SHAPE, dtype=torch.float32, device=device)
    rendered = torch.ones(SHAPE, dtype=torch.bfloat16)
    each = rendered.numel() * rendered.element_size()
    cache = replay.RetainedRenderDeviceCache(lambda: 1 << 40)
    for index in range(COUNT):
        cache.delta(("unit", index), rendered, source, keep=True)
    assert cache.bytes_held == COUNT * each
    # The deltas' blocks go back to the device, so the reservation left is
    # the kept renders and the source.
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    before = _reading(scope, device)
    freed = cache.reclaim(_freed(COUNT * each))
    after = _reading(scope, device)
    settled = _settled()
    assert freed == _freed(COUNT * each)
    drops, lowered = _moved("render_cache", before, after, settled, freed)
    # None of it was charged to the memcg.
    assert abs(drops["committed"]) <= READING_GRAIN
    assert lowered == replay.RENDER_CACHE_RECLAIM_LOWERS


def test_a_spill_replay_reclaim_lowers_what_it_declares(device):
    scope = own_cgroup()
    block = spill_mod.SPILL_SEAL_BLOCK_BYTES
    read_bytes = spill_mod.READ_BYTES - block
    held = spill_mod._host_buffer_bytes(read_bytes, block, True)
    count = (COUNT * 32 << 20) // held

    def read_chunk(index):
        # The spill's own reader: a new pinned buffer, filled.
        buffer = spill_mod._aligned_buffer(read_bytes, block, True)
        buffer.fill_(index % 251)
        return buffer, (read_bytes,)

    entries = [io_engine.ReadEntry(
        key=("chunk", index), path=None, size=read_bytes, limit=read_bytes,
        held_bytes=held, expected_sha256=None, decoder=None, group=("chunk", index),
        reader=partial(read_chunk, index)) for index in range(count)]
    budget = io_engine.FixedBudget(buffer_bytes=held, headroom=1 << 40)
    with io_engine.read_stream(entries, budget=budget) as stream:
        quiet(stream)
        assert stream.held_bytes() == count * held
        # The replay's reclaimer, on a spill that holds only this stream.
        spill = SimpleNamespace(_replay_stream=stream, _cuda=True,
                                telemetry={"replay_reclaims": 0, "replay_reclaimed_bytes": 0})
        with stream.paused():
            before = _reading(scope, device)
            freed = spill_mod.StageBReplaySpill.reclaim_replay(spill, _freed(count * held))
            after = _reading(scope, device)
            settled = _settled()
    assert freed >= _freed(count * held)
    drops, lowered = _moved("spill_replay", before, after, settled, freed)
    # The pinned buffers' pages, to within the cgroup reading's grain.
    assert freed - READING_GRAIN <= drops["committed"] <= freed + READING_GRAIN
    assert drops["reserved"] == 0
    assert lowered == replay.SPILL_REPLAY_RECLAIM_LOWERS
