"""Slice manifests must carry usable phase ranges (PQ #849).

A v1 slice whose phase table opens with a zero-byte ``head`` voids the whole table (a
cumulative outside the entries' own prefix sums), so no phase range exists and the quantum
row stages nothing. The builder seals chunk phases only; this file proves it with an
independent range oracle, not by shape expectation alone.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

import readset_oracles

pytestmark = pytest.mark.own_process

def _parent():
    entries = [{"path": f"/mnt/shared/model/shard-{i}.safetensors",
                "offset": 0, "bytes": 1024, "sha256": None}
               for i in range(8)]
    return {"schema": "prismaquant.prismabuild.data_manifest.v1",
            "mount_prefix": "/mnt/shared",
            "entries": entries, "entry_count": 8, "total_bytes": 8192,
            "annotations": {
                "campaign_scope": {"test": "slice-pb-staging"},
                "phases": [{"name": "head", "bytes": 2048,
                            "cumulative_bytes": 2048},
                           {"name": "layer-0", "bytes": 6144,
                            "cumulative_bytes": 8192}]}}


def test_slice_phase_ranges_cover_the_fixed_slice():
    from prismaquant.joint_layer_quanta import slice_layer_manifest
    manifest = slice_layer_manifest(
        _parent(), 0, chunk_target_bytes=2048, quantum="layer-000",
        output_root="/mnt/shared/run", plan_path="plan.json",
        plan_sha256="0" * 64, prepared_sha256="1" * 64,
        parent_manifest_sha256="2" * 64)
    assert all(p["bytes"] > 0 for p in manifest["annotations"]["phases"])
    ranges = readset_oracles.phase_ranges(manifest)
    assert ranges, "no phase range for the fixed slice"
    assert [r["name"] for r in ranges] == [
        p["name"] for p in manifest["annotations"]["phases"]]
    assert ranges[-1]["end_bytes"] == manifest["total_bytes"]
    assert ranges[0]["start_bytes"] == 0
