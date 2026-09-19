"""PQ #737: derive the sealed ``source_prefetch`` depth from the measured
budget instead of re-sealing the pinned ``max_cache_slots: 2 /
prefetch_workers: 1``. The numbers stay explicit in the sealed plan;
only their provenance changes from constant to measurement."""
from __future__ import annotations

import pytest

GiB = 1024 ** 3


def test_derives_seven_slot_window_from_live_run_budgets():
    """The live run's own banner: 97.4 GB budget, 13.8 GB layer."""
    from prismaquant.tessera_joint_aura import recommend_source_prefetch

    got = recommend_source_prefetch(cache_bytes=int(97.4 * GiB),
        layer_bytes=int(13.8 * GiB), cpu_count=8,
        cache_headroom_gb=2.0, prefetch_min_available_gb=2.0)
    assert got["max_cache_slots"] == 7
    assert got["prefetch_workers"] == 4
    assert got["prefetch_lookahead"] == 4
    assert got["require_prefetched_residency"] is True


def test_floors_a_small_budget():
    from prismaquant.tessera_joint_aura import recommend_source_prefetch

    got = recommend_source_prefetch(cache_bytes=int(8 * GiB),
        layer_bytes=int(13.8 * GiB), cpu_count=8,
        cache_headroom_gb=2.0, prefetch_min_available_gb=2.0)
    assert got["max_cache_slots"] == 2
    assert got["prefetch_workers"] == 2
    assert got["prefetch_lookahead"] == 1


def test_bounds_workers_by_cpus():
    from prismaquant.tessera_joint_aura import recommend_source_prefetch

    got = recommend_source_prefetch(cache_bytes=int(97.4 * GiB),
        layer_bytes=int(13.8 * GiB), cpu_count=1,
        cache_headroom_gb=2.0, prefetch_min_available_gb=2.0)
    assert got["prefetch_workers"] == 1
    assert got["prefetch_lookahead"] == 1


@pytest.mark.parametrize("kwargs", [
    dict(cache_bytes=0, layer_bytes=8, cpu_count=8),
    dict(cache_bytes=8, layer_bytes=0, cpu_count=8),
    dict(cache_bytes=8, layer_bytes=8, cpu_count=0),
])
def test_refuses_nonpositive_budgets(kwargs):
    from prismaquant.tessera_joint_aura import recommend_source_prefetch

    with pytest.raises(ValueError, match="positive"):
        recommend_source_prefetch(cache_headroom_gb=2.0,
            prefetch_min_available_gb=2.0, **kwargs)
