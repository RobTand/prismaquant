"""Report only the resolver's recorded serving rows, not workload admission or throughput.

The resolver (over a real materialized manifest), record API, report and bench projection are
real. There are no mocks. Recording an observation does not perform or attest a payload read,
a stage lease, a fleet action or a complete workload census.
"""
from pathlib import Path

import pytest

from prismaquant.residency_map import ResidencyResolver
from staged_native_fixtures import publish_inputs
from tools import staged_exact_read_bench as bench


@pytest.mark.parametrize("populated", [True, False], ids=["recorded_rows", "empty_control"])
def test_staged_exact_report_preserves_recorded_serving_tier_names(tmp_path: Path, populated: bool):
    source = tmp_path / "source.bin"
    source.write_bytes(b"known-bytes")
    binding = publish_inputs(tmp_path, [source])
    resolver = ResidencyResolver(binding["path"], manifest_sha256=binding["sha256"],
                                 read_contract_sha256=binding["read_contract_sha256"])
    expected = {}
    if populated:
        for name, tier in [("a", "stage"), ("b", "stage"), ("c", "other")]:
            resolver.record_serving_tier(tmp_path / name, tier, detail="owned audit observation")
        expected = {"stage": 2, "other": 1}
    owner_report = resolver.report()
    assert [row["tier"] for row in owner_report["serving_tiers"]] == (
        ["stage", "stage", "other"] if populated else [])
    result = bench._residency(resolver)
    assert result["serving_tier_rows"] == expected, result
    assert sum(result["serving_tier_rows"].values()) == len(owner_report["serving_tiers"])
    resolver.close()
