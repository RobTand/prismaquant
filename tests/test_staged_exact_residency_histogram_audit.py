"""Report only the resolver's bounded serving rows, not workload admission or throughput.

The resolver, record API, report, and bench projection are real. There are no
mocks. Explicitly recording ram/stage observations does not perform or attest a
payload read, stage lease, fleet action, or complete workload census.
"""
from pathlib import Path

import pytest

from prismaquant.residency_map import ResidencyResolver
from tools import staged_exact_read_bench as bench


@pytest.mark.parametrize("populated", [True, False], ids=["recorded_ram_and_stage", "empty_control"])
def test_staged_exact_report_preserves_recorded_serving_tier_names(tmp_path: Path, populated: bool):
    resolver = ResidencyResolver(tmp_path / "unused-residency-map.json")
    expected = {}
    if populated:
        for name, tier in [("ram-a", "ram"), ("stage-b", "stage"), ("ram-c", "ram")]:
            resolver.record_serving_tier(tmp_path / name, tier, detail="owned audit observation")
        expected = {"ram": 2, "stage": 1}

    owner_report = resolver.report()
    # Establish the actual producer schema before testing the report adapter.
    assert owner_report["serving_tier_count"] == (3 if populated else 0)
    assert [row["serving_tier"] for row in owner_report["serving_tiers"]] == (
        ["ram", "stage", "ram"] if populated else []
    )
    result = bench._residency(resolver)
    assert result["serving_tier_rows"] == expected, (
        "staged exact report must preserve the existing serving_tier field "
        f"for the bounded recorded rows: {result!r}"
    )
    assert sum(result["serving_tier_rows"].values()) == len(owner_report["serving_tiers"])


def test_histogram_counts_retained_rows_after_the_record_cap(tmp_path):
    from prismaquant.residency_map import MAX_RECORDED_FALLBACKS

    resolver = ResidencyResolver(tmp_path / "unused-residency-map.json")
    for index in range(MAX_RECORDED_FALLBACKS + 3):
        tier = "ram" if index < MAX_RECORDED_FALLBACKS else "stage"
        resolver.record_serving_tier(tmp_path / str(index), tier)
    owner_report = resolver.report()
    assert owner_report["serving_tier_count"] == MAX_RECORDED_FALLBACKS + 3
    assert len(owner_report["serving_tiers"]) == MAX_RECORDED_FALLBACKS
    assert bench._residency(resolver)["serving_tier_rows"] == {"ram": MAX_RECORDED_FALLBACKS}
