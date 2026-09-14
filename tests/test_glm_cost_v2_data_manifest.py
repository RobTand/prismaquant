"""COST's V2 manifest describes forward/reverse reads and stable resume windows."""
import copy
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from experiments import glm_data_manifests as producer
from prismaquant import format_registry as fr
from prismaquant.joint_retained_window_plan import RetainedWindowBudget
from prismaquant.joint_statistics_plan import plan_joint_statistics_target_windows
import test_glm_joint_data_manifest_at_submit as existing
from test_glm_joint_data_manifest_at_submit import scratch, shared_mount


@pytest.fixture()
def cost_fixture(scratch, shared_mount, monkeypatch):
    # Three source layers make the reverse lookahead reread observable. The
    # established fixture builds the actual campaign files and cache roster.
    units = {layer: [existing.PREFIX + f"{layer}.mlp.{suffix}"
                     for suffix in ("gate_proj", "up_proj")]
             for layer in range(3)}
    monkeypatch.setattr(existing, "UNITS", units)
    fixture = existing._workspace(scratch)
    (fixture["model"] / "config.json").write_text(json.dumps({"num_hidden_layers": 3}))
    census_path = fixture["workspace"] / "census.json"
    census = json.loads(census_path.read_text())
    census["unit_shapes"] = {name: [1024, 1024] for name in fixture["names"]}
    census["max_abs"] = {name: 1.0 for name in fixture["names"]}
    census_path.write_text(json.dumps(census))

    production_cache = scratch / "production.pkl"
    production_cache.write_bytes(b"p" * 5000)
    prepared = scratch / "prepared.json"
    prepared.write_text(json.dumps({
        "schema": producer.JOINT_PREPARED_SCHEMA, "status": "complete",
        "production_cache": {"path": str(production_cache), "sha256": None},
        "formats_by_qname": {
            name: [*existing.MEASURED, "BF16"] for name in fixture["names"]},
    }))
    first = fixture["names"][0]
    module = torch.nn.Linear(1024, 1024, bias=False,
                             device="meta", dtype=torch.bfloat16)
    formats = {fmt: fr.get_format(fmt) for fmt in existing.MEASURED}
    stats = plan_joint_statistics_target_windows(
        {first: module}, {first: formats}, max_statistics_bytes=1 << 30,
        activation_max_abs={first: 1.0})
    single_target = stats.targets[0].statistics_bytes
    maximum_render = max(
        path.stat().st_size
        for path in (fixture["workspace"] / "rows").glob("*/cache/*.pt"))
    budget = RetainedWindowBudget(
        physical_limit_bytes=104 << 30, safety_margin_bytes=2 << 30,
        metadata_reserve_bytes=20 << 30, runtime_reserve_bytes=4 << 30,
        workspace_reserve_bytes=16 << 30, boundary_reserve_bytes=2 << 30,
        auxiliary_reserve_bytes=2 << 30, load_buffer_bytes=maximum_render,
        read_page_reserve_bytes=4096, candidate_delta_bytes=4 << 20,
        statistics_cap_bytes=single_target,
        retained_render_cap_bytes=4 * maximum_render,
        max_windows_per_layer=2)
    return fixture, prepared, budget


def _build(fixture, prepared, budget, *, completed=None):
    return producer.build_joint_cost_v2_manifest(
        str(fixture["plan"]), prepared=str(prepared),
        produced_by=existing.PRODUCED_BY, retained_budget=budget,
        source_bytes=31 << 30, n_probes=4,
        validated_completed_units=completed)


def _phases(manifest):
    return {phase["name"]: phase for phase in manifest["read_plan"]["phases"]}


def test_cost_v2_read_plan_repeats_source_without_repeating_entries(
        cost_fixture, monkeypatch):
    fixture, prepared, budget = cost_fixture
    monkeypatch.setattr(producer, "_unit_state",
                        lambda *args: pytest.fail("COST inspected every unit payload"))
    monkeypatch.setattr(producer.Campaign, "capture_files_for",
                        lambda *args: pytest.fail("COST read PREPARE activation captures"))
    manifest = _build(fixture, prepared, budget)
    assert manifest["schema"] == producer.SCHEMA_V2
    assert manifest["annotations"]["source_owner_cap_bytes"] == 31 << 30
    assert manifest["annotations"]["probes_per_window"] == 4
    assert manifest["entry_count"] == len(manifest["entries"])
    assert len({(item["path"], item["offset"]) for item in manifest["entries"]}) == len(manifest["entries"])
    assert not any(item["path"].endswith(".tessera") for item in manifest["entries"])
    assert not any("/inputs/" in item["path"] for item in manifest["entries"])
    assert manifest["read_plan"]["read_bytes"] > manifest["total_bytes"]
    phases = _phases(manifest)
    assert list(phases)[:6] == [
        "cost_setup", "cost_head", "cost_capture_000", "cost_capture_001",
        "cost_capture_002", "cost_tail"]
    assert not phases["cost_reverse_002_source"]["entry_indices"]
    assert set(phases["cost_reverse_001_source"]["entry_indices"]) & set(
        phases["cost_capture_000"]["entry_indices"])
    assert not phases["cost_reverse_000_source"]["entry_indices"]
    assert all(manifest["entries"][index]["path"].endswith(".safetensors")
               for layer in range(3)
               for index in phases[f"cost_capture_{layer:03d}"]["entry_indices"])
    windows = manifest["annotations"]["windows"]
    assert len(windows) == 6  # two immutable whole-target windows per layer
    assert [window["layer"] for window in windows] == [2, 2, 1, 1, 0, 0]
    assert all(window["original_full_target_names"] == window["active_pending_names"]
               for window in windows)
    assert all(len(window["original_full_target_names"]) == 1 for window in windows)
    render_refs = [index for phase in phases.values() if "_window_" in phase["name"]
                   for index in phase["entry_indices"]]
    assert len(render_refs) == len(fixture["names"]) * len(existing.MEASURED)
    assert len(render_refs) == len(set(render_refs))
    assert manifest == _build(fixture, prepared, budget)

    # The published PB V2 validator is the final schema authority.
    import sys
    sys.path.insert(0, "/mnt/shared/prismabuild-fleet/repo/src")
    from prismabuild.core import validate_data_manifest
    assert validate_data_manifest(manifest)["read_plan"] == manifest["read_plan"]


def test_cost_v2_resume_filters_reads_without_repartitioning(cost_fixture):
    fixture, prepared, budget = cost_fixture
    full = _build(fixture, prepared, budget)
    names = fixture["names"]
    completed_names = sorted([names[0], *names[2:4]])
    binding = {
        "schema": producer.COST_COMPLETED_BINDING_SCHEMA,
        "plan_sha256": hashlib.sha256(fixture["plan"].read_bytes()).hexdigest(),
        "prepared_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
        "units": completed_names,
    }
    resumed = _build(fixture, prepared, budget, completed=binding)
    assert resumed["annotations"]["window_partition_sha256"] == (
        full["annotations"]["window_partition_sha256"])
    assert [phase["name"] for phase in resumed["read_plan"]["phases"]] == [
        phase["name"] for phase in full["read_plan"]["phases"]]
    windows = {tuple(window["original_full_target_names"]): window
               for window in resumed["annotations"]["windows"]}
    for name in completed_names:
        assert windows[(name,)]["active_pending_names"] == []
        assert _phases(resumed)[windows[(name,)]["phase"]]["entry_indices"] == []
    for name in set(names) - set(completed_names):
        assert windows[(name,)]["active_pending_names"] == [name]
    declared = {entry["path"] for entry in resumed["entries"]}
    for name in completed_names:
        assert not any(producer._cache_weight_filename(name, fmt) in path
                       for path in declared for fmt in existing.MEASURED)
    assert resumed["read_plan"]["read_bytes"] < full["read_plan"]["read_bytes"]
    for layer in range(3):
        phase = f"cost_capture_{layer:03d}"
        assert _phases(resumed)[phase]["entry_indices"] == _phases(full)[phase]["entry_indices"]
    assert producer.check_manifest_v2(resumed) is resumed

    all_complete = {**binding, "units": sorted(names)}
    finished = _build(fixture, prepared, budget, completed=all_complete)
    assert all(not window["active_pending_names"]
               for window in finished["annotations"]["windows"])
    assert all(not any("/inputs/" in finished["entries"][index]["path"]
                       for index in _phases(finished)[f"cost_capture_{layer:03d}"]["entry_indices"])
               for layer in range(3))
    assert finished["annotations"]["window_partition_sha256"] == (
        full["annotations"]["window_partition_sha256"])


def test_cost_v2_refuses_unvalidated_resume_and_broken_read_references(cost_fixture):
    fixture, prepared, budget = cost_fixture
    with pytest.raises(SystemExit, match="validated plan/prepared binding"):
        _build(fixture, prepared, budget, completed={"units": fixture["names"]})
    manifest = _build(fixture, prepared, budget)
    broken = copy.deepcopy(manifest)
    broken["read_plan"]["phases"][0]["cumulative_bytes"] += 1
    with pytest.raises(SystemExit, match="phase byte accounting"):
        producer.check_manifest_v2(broken)
    broken = copy.deepcopy(manifest)
    broken["read_plan"]["phases"][0]["entry_indices"].append(
        broken["read_plan"]["phases"][0]["entry_indices"][0])
    with pytest.raises(SystemExit, match="repeated phase entry"):
        producer.check_manifest_v2(broken)


def test_cost_v2_refuses_missing_prepared_render_or_indivisible_target(cost_fixture):
    fixture, prepared, budget = cost_fixture
    name = fixture["names"][0]
    render = (fixture["workspace"] / "rows" / "row-0000" / "cache"
              / producer._cache_weight_filename(name, existing.MEASURED[0]))
    render.unlink()
    with pytest.raises(SystemExit, match="prepared COST render is missing"):
        _build(fixture, prepared, budget)
    render.write_bytes(b"r" * 2048)
    with pytest.raises(RuntimeError, match="indivisible target"):
        _build(fixture, prepared,
               replace(budget, retained_render_cap_bytes=1))
    changed = json.loads(fixture["plan"].read_text())
    changed["source_prefetch"]["prefetch_lookahead"] = 2
    fixture["plan"].write_text(json.dumps(changed))
    with pytest.raises(SystemExit, match="source lookahead one"):
        _build(fixture, prepared, budget)
