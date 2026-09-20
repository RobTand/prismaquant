"""Slice manifests must stage under published PB (PQ #849).

A v1 slice whose phase table opens with a zero-byte ``head`` voids the
whole table under PB's phase rule (#594: a cumulative outside the
entries' own prefix sums refuses the manifest), so
``manifest_phase_ranges`` returns ``[]`` and the quantum row stages
nothing. The builder now seals chunk phases only; this file proves it
against the real published generation -- not by shape expectation alone.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

FLEET_ROOT = Path("/mnt/shared/prismabuild-fleet")


def _published_pb_src():
    """The immutable PB generation pinned by the fleet receipt, or skip."""
    receipt_path = FLEET_ROOT / "repo" / "RUNTIME_VERSION.json"
    try:
        receipt = json.loads(receipt_path.read_text())
        generation = str(receipt["generation"])
        pinned = dict(receipt["files"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        pytest.skip(f"no PB runtime receipt: {exc}")
    gen_src = FLEET_ROOT / "runtime-generations" / generation / "src"
    core = gen_src / "prismabuild" / "core.py"
    try:
        actual = hashlib.sha256(core.read_bytes()).hexdigest()
    except OSError as exc:
        pytest.skip(f"unreadable PB generation {generation}: {exc}")
    if actual != pinned.get("src/prismabuild/core.py"):
        pytest.skip(f"PB generation {generation} is not the published bytes")
    return gen_src


def _pb():
    src = _published_pb_src()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import prismabuild.core as core
    import prismabuild.storage_tiers as tiers
    for module in (core, tiers):
        if not Path(module.__file__).resolve().is_relative_to(src.resolve()):
            pytest.skip("a different prismabuild is already imported")
    return core, tiers


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


def test_slice_stages_under_published_pb():
    from prismaquant.joint_layer_quanta import slice_layer_manifest
    core, tiers = _pb()
    manifest = slice_layer_manifest(
        _parent(), 0, chunk_target_bytes=2048, quantum="layer-000",
        output_root="/mnt/shared/run", plan_path="plan.json",
        plan_sha256="0" * 64, prepared_sha256="1" * 64,
        parent_manifest_sha256="2" * 64)
    assert all(p["bytes"] > 0 for p in manifest["annotations"]["phases"])
    normalized = core.validate_data_manifest(manifest)
    ranges = tiers.manifest_phase_ranges(normalized)
    assert ranges, "published PB stages nothing for the fixed slice"
    assert [r["name"] for r in ranges] == [
        p["name"] for p in manifest["annotations"]["phases"]]
    assert ranges[-1]["end_bytes"] == manifest["total_bytes"]
    assert ranges[0]["start_bytes"] == 0


def test_slice_table_holds_no_zero_byte_head():
    from prismaquant.joint_layer_quanta import slice_layer_manifest
    manifest = slice_layer_manifest(
        _parent(), 0, chunk_target_bytes=2048, quantum="layer-000",
        output_root="/mnt/shared/run", plan_path="plan.json",
        plan_sha256="0" * 64, prepared_sha256="1" * 64,
        parent_manifest_sha256="2" * 64)
    names = [p["name"] for p in manifest["annotations"]["phases"]]
    assert "head" not in names
    assert sum(p["bytes"] for p in manifest["annotations"]["phases"]) == \
        manifest["total_bytes"]
