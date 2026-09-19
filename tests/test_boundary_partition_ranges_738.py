"""PQ #738: the boundary capture's partition axis as named, verifiable
quanta. This module names ranges and checks their union; it schedules
nothing -- execution, placement and balancing stay PrismaBuild's, via its
fanout/campaign interfaces, once a campaign row per range exists."""
from __future__ import annotations

import pytest


def test_512_partitions_cover_without_gaps_or_overlaps():
    from prismaquant.cost_streaming import (
        verify_boundary_partition_coverage, plan_boundary_partition_ranges)

    for n_ranges in (1, 2, 3, 7, 512):
        ranges = plan_boundary_partition_ranges(n_partitions=512, n_ranges=n_ranges)
        assert len(ranges) == n_ranges
        assert sum(entry["partitions"] for entry in ranges) == 512
        assert verify_boundary_partition_coverage(ranges, n_partitions=512) == sorted(
            ranges, key=lambda entry: entry["range_index"])


def test_widths_differ_by_at_most_one():
    from prismaquant.cost_streaming import plan_boundary_partition_ranges

    widths = [entry["partitions"]
              for entry in plan_boundary_partition_ranges(n_partitions=512, n_ranges=7)]
    assert max(widths) - min(widths) <= 1


def test_coverage_refuses_gaps_and_overlaps():
    from prismaquant.cost_streaming import (
        verify_boundary_partition_coverage, plan_boundary_partition_ranges)

    full = plan_boundary_partition_ranges(n_partitions=8, n_ranges=3)
    gapped = [{**full[0]}, {**full[1], "partition_start": 4, "partitions": 2},
              {**full[2]}]
    with pytest.raises(ValueError, match="gap or overlap"):
        verify_boundary_partition_coverage(gapped, n_partitions=8)
    overlapped = [{**full[0], "partition_end": 4, "partitions": 4}] + full[1:]
    with pytest.raises(ValueError, match="gap or overlap"):
        verify_boundary_partition_coverage(overlapped, n_partitions=8)


def test_coverage_refuses_drift_and_bad_counts():
    from prismaquant.cost_streaming import (
        verify_boundary_partition_coverage, plan_boundary_partition_ranges)

    full = plan_boundary_partition_ranges(n_partitions=8, n_ranges=3)
    doubled = full + [full[0]]
    with pytest.raises(ValueError, match="repeats|disagrees"):
        verify_boundary_partition_coverage(doubled, n_partitions=8)
    drifted = [{**full[0], "n_partitions": 9}] + full[1:]
    with pytest.raises(ValueError, match="disagrees"):
        verify_boundary_partition_coverage(drifted, n_partitions=8)
    with pytest.raises(ValueError, match="at least one range"):
        verify_boundary_partition_coverage([], n_partitions=8)
    with pytest.raises(ValueError, match="1 <="):
        plan_boundary_partition_ranges(n_partitions=4, n_ranges=5)
