"""Check the life of sealed read-ahead mappings, not total cgroup memory."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from prismaquant import io_engine

from cgroup_readings import quiet as _quiet, renders


@pytest.fixture(autouse=True)
def _no_residency_map(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_STAGED_MANIFEST", raising=False)


def test_freed_read_ahead_leaves_the_committed_reading(tmp_path):
    cache, keys, each = renders(tmp_path, 4, (16, 16))
    identities = {}
    maps_path = None

    def observe(entry):
        def decode(raw, receipt, staged):
            nonlocal maps_path
            result = entry.decoder(raw, receipt, staged)
            descriptor = Path(raw.path)
            stat = descriptor.stat()
            maps_path = descriptor.parents[1] / "maps"
            device = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}"
            identities[entry.key] = (device, str(stat.st_ino))
            return result

        return replace(entry, decoder=decode)

    def live_keys():
        assert maps_path is not None
        maps = maps_path.read_text().splitlines()
        mapped = {tuple(line.split(maxsplit=5)[3:5]) for line in maps}
        return {key for key, identity in identities.items() if identity in mapped}

    entries = [observe(entry) for group, members in enumerate((keys[:2], keys[2:]))
               for entry in cache.retained_read_entries(members, group=group)]
    # Limit each worker to one read in flight and five archive charges.
    budget = io_engine.FixedBudget(buffer_bytes=each, headroom=5 * each)
    with io_engine.read_stream(entries, budget=budget) as stream:
        _quiet(stream)
        # Prevent new mappings and inode reuse in this stream until close.
        with stream.paused():
            assert live_keys() == set(keys)
            assert len(set(identities.values())) == len(keys)
            assert stream.reclaim(each) == each
            assert live_keys() == set(keys[:3])

            delivered = stream.take(0)
            assert [item.entry.key for item in delivered] == keys[:2]
            assert all(item.value.eq(index).all() for index, item in enumerate(delivered))
            assert live_keys() == set(keys[:3])
            del delivered
            assert live_keys() == {keys[2]}
            stream.close()
            assert not live_keys()


def test_sealed_read_ahead_ignores_unrelated_cgroup_allocations(tmp_path, monkeypatch):
    (tmp_path / "cgroup.procs").write_text("1\n")
    readings = iter((0, 1 << 30))

    def committed(_scope):
        charge = next(readings)
        return {"committed": charge, "anon_shmem": charge, "shmem": charge, "anon": 0}

    # Replace the old reading and its isolation check on the base.
    namespace = test_freed_read_ahead_leaves_the_committed_reading.__globals__
    monkeypatch.setitem(namespace, "_own_cgroup", lambda: tmp_path)
    monkeypatch.setitem(namespace, "READING_GRAIN", 1)
    monkeypatch.setitem(namespace, "_committed", committed)
    test_freed_read_ahead_leaves_the_committed_reading(tmp_path)
