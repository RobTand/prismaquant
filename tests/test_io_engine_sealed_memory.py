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
rounding each memfd carries, and it must come out of ``shmem``, where the
memfd pages live, and not out of ``anon``: the committed reading counts
``shmem`` as it counts ``anon``, so the depth and the reclaim see these bytes.
A take alone frees nothing, which is why the stream charges a taken group to
its budget until the consumer releases it. It must run where it can read its
cgroup; a missing cgroup fails it rather than skipping it, because a skipped
probe certifies nothing.
"""
from __future__ import annotations

import pytest

from prismaquant import io_engine

from cgroup_readings import READING_GRAIN, own_cgroup as _own_cgroup
from cgroup_readings import committed as _committed, pages as _pages
from cgroup_readings import quiet as _quiet, renders

# A Stage B render file: 16 MiB of storage, as the GLM layer's renders are.
RENDER_SHAPE = (2048, 4096)
PER_GROUP = 8


@pytest.fixture(autouse=True)
def _no_residency_map(monkeypatch):
    monkeypatch.delenv("PRISMABUILD_RESIDENCY_MAP", raising=False)


def test_freed_read_ahead_leaves_the_committed_reading(tmp_path):
    scope = _own_cgroup()
    procs = (scope / "cgroup.procs").read_text().split()
    cache, keys, each = renders(tmp_path, 2 * PER_GROUP, RENDER_SHAPE)
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
        assert dropped == 0 or low > 0, "the reading's grain hides the freed pages"
        for measure in ("committed", "anon_shmem", "shmem"):
            drop = before[measure] - after[measure]
            assert low <= drop <= high, (
                f"{label}: {measure} dropped {drop} bytes; freeing {dropped} "
                f"entries of {each} bytes must drop between {low} and {high} "
                f"(cgroup {scope}, {len(procs)} processes; before {before}, after {after})")
        # The pages were shared memory, not anonymous memory the allocator
        # could keep: anon does not move beyond the reading's grain.
        moved = before["anon"] - after["anon"]
        assert abs(moved) <= READING_GRAIN, (
            f"{label}: anon moved {moved} bytes; the freed pages must be shmem "
            f"(before {before}, after {after})")

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
        _quiet(stream)
        # A consumer takes a group: the take alone frees nothing, since the
        # consumer holds the values it was handed.
        with stream.paused():
            before = _committed(scope)
        delivered = stream.take(0)
        with stream.paused():
            after = _committed(scope)
        check("take", before, after, 0)
        # The consumer drops the group, and releases it.
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
