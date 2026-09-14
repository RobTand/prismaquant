"""The sealed PB V2 read plan is COST's authoritative traversal and roster."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json

import pytest

from prismaquant.joint_cost_read_schedule import load_joint_cost_read_schedule
from prismaquant.joint_retained_window_plan import RetainedWindowBudget


PLAN_SHA = "a" * 64
PREPARED_SHA = "b" * 64
NAMES = {0: ("model.layers.0.a",), 1: ("model.layers.1.b",)}
PHASES = (
    "cost_setup", "cost_head", "cost_capture_000", "cost_capture_001",
    "cost_tail", "cost_reverse_001_source", "cost_reverse_001_window_000",
    "cost_reverse_000_source", "cost_reverse_000_window_000",
)


def _budget():
    return RetainedWindowBudget(
        physical_limit_bytes=1000, safety_margin_bytes=100,
        metadata_reserve_bytes=50, runtime_reserve_bytes=50,
        workspace_reserve_bytes=50, boundary_reserve_bytes=0,
        auxiliary_reserve_bytes=0, load_buffer_bytes=50,
        read_page_reserve_bytes=50, candidate_delta_bytes=50,
        statistics_cap_bytes=100, retained_render_cap_bytes=100,
        max_windows_per_layer=2)


def _binding_sha(completed):
    if not completed:
        return None
    binding = {"schema": "prismaquant.joint_cost.validated_completed_units.v1",
               "plan_sha256": PLAN_SHA, "prepared_sha256": PREPARED_SHA,
               "units": sorted(completed)}
    return hashlib.sha256(json.dumps(binding, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _manifest(completed=()):
    windows = []
    for layer in (1, 0):
        name = NAMES[layer][0]
        windows.append({"phase": f"cost_reverse_{layer:03d}_window_000",
                        "layer": layer, "window_index": 0,
                        "original_full_target_names": [name],
                        "active_pending_names": [] if name in completed else [name],
                        "statistics_bytes": 25,
                        "render_file_upper_bound_bytes": 30,
                        "candidate_count": 1})
    partition = [[w["layer"], w["window_index"], w["original_full_target_names"]]
                 for w in windows]
    digest = hashlib.sha256(json.dumps(partition, separators=(",", ":"),
                                       ensure_ascii=False).encode()).hexdigest()
    annotations = {
        "entry_point": "prismaquant.tessera_joint_aura:run",
        "mode": "retained_cost_v2", "plan": "/mnt/shared/plan.json",
        "plan_sha256": PLAN_SHA, "prepared": "/mnt/shared/prepared.json",
        "prepared_sha256": PREPARED_SHA, "source_owner_cap_bytes": 100,
        "retained_budget": _budget().as_dict(), "probes_per_window": 4,
        "source_prefetch_lookahead_layers": 1,
        "tail_retained_source_layers": [0, 1],
        "window_partition_sha256": digest, "windows": windows,
        "validated_completed_units_sha256": _binding_sha(completed),
        "validated_completed_units": len(completed),
        "sha256_present": False, "sha256_absent_reason": "read set is a residency hint",
        "argv": None,
    }
    phases, cumulative = [], 0
    for name in PHASES:
        refs = [] if name == "cost_reverse_000_window_000" and NAMES[0][0] in completed else [0]
        cumulative += len(refs) * 10
        phases.append({"name": name, "entry_indices": refs,
                       "bytes": len(refs) * 10, "cumulative_bytes": cumulative})
    return {"schema": "prismaquant.prismabuild.data_manifest.v2",
            "produced_by": {}, "mount_prefix": "/mnt/shared", "annotations": annotations,
            "entries": [{"path": "/mnt/shared/source.bin", "offset": 0,
                         "bytes": 10, "sha256": None}],
            "entry_count": 1, "total_bytes": 10,
            "read_plan": {"phases": phases, "read_bytes": cumulative}}


def _load(tmp_path, manifest=None, *, completed=(), gzip_wire=False, **overrides):
    manifest = _manifest(completed) if manifest is None else manifest
    raw = json.dumps(manifest, separators=(",", ":")).encode()
    if gzip_wire:
        raw = gzip.compress(raw, mtime=0)
    path = tmp_path / "manifest.cas"
    path.write_bytes(raw)
    kwargs = dict(manifest_path=path, manifest_sha256=hashlib.sha256(raw).hexdigest(),
                  manifest_bytes=len(raw), plan_path="/mnt/shared/plan.json",
                  plan_sha256=PLAN_SHA, prepared_path="/mnt/shared/prepared.json",
                  prepared_sha256=PREPARED_SHA, retained_budget=_budget(),
                  source_owner_cap_bytes=100, n_probes=4,
                  target_names_by_layer=NAMES, validated_completed_units=completed)
    kwargs.update(overrides)
    return load_joint_cost_read_schedule(**kwargs)


def test_sealed_schedule_exposes_original_windows_and_ordered_progress(tmp_path):
    calls = []
    completed = (NAMES[0][0],)
    schedule = _load(tmp_path, completed=completed, gzip_wire=True,
                     progress_callback=lambda phase, units: calls.append((phase, units)))
    assert schedule.current_phase is None
    assert schedule.identity["manifest_bytes"] > 0
    assert len(schedule.identity["window_partition_sha256"]) == 64
    assert schedule.windows_for_layer(1)[0].active_pending_names == NAMES[1]
    assert schedule.windows_for_layer(0)[0].original_full_target_names == NAMES[0]
    assert schedule.windows_for_layer(0)[0].active_pending_names == ()
    for phase in PHASES:
        schedule.enter_phase(phase, 1)
    schedule.enter_phase(PHASES[-1], 2)
    assert schedule.current_phase == PHASES[-1]
    assert calls == [(phase, 1) for phase in PHASES] + [(PHASES[-1], 2)]
    with pytest.raises(ValueError, match="phase traversal"):
        schedule.enter_phase(PHASES[0], 3)


def test_early_binding_allows_only_setup_head_until_actual_resume_is_known(tmp_path):
    calls = []
    schedule = _load(tmp_path, target_names_by_layer=None,
                     validated_completed_units=None,
                     progress_callback=lambda phase, units: calls.append((phase, units)))
    schedule.enter_phase("cost_setup", 0)
    schedule.enter_phase("cost_head", 0)
    with pytest.raises(ValueError, match="must be bound"):
        schedule.enter_phase("cost_capture_000", 0)
    with pytest.raises(ValueError, match="not bound"):
        schedule.windows_for_layer(0)
    schedule.bind_runtime(NAMES, ())
    with pytest.raises(ValueError, match="already bound"):
        schedule.bind_runtime(NAMES, ())
    schedule.enter_phase("cost_capture_000", 0)
    assert calls == [("cost_setup", 0), ("cost_head", 0), ("cost_capture_000", 0)]


def test_late_binding_checks_actual_checkpoint_set(tmp_path):
    schedule = _load(tmp_path, target_names_by_layer=None,
                     validated_completed_units=None)
    schedule.enter_phase("cost_setup", 0)
    schedule.enter_phase("cost_head", 0)
    with pytest.raises(ValueError, match="checkpoint binding"):
        schedule.bind_runtime(NAMES, (NAMES[0][0],))
    schedule.bind_runtime(NAMES, ())


def test_descriptor_and_json_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="SHA-256 differs"):
        _load(tmp_path, manifest_sha256="f" * 64)
    with pytest.raises(ValueError, match="byte count differs"):
        _load(tmp_path, manifest_bytes=1)
    manifest = _manifest()
    raw = json.dumps(manifest).encode() + b" "
    path = tmp_path / "manifest.cas"
    path.write_bytes(raw)
    path.unlink()
    path.symlink_to(tmp_path / "target")
    with pytest.raises(OSError):
        _load(tmp_path)
    path.unlink()
    duplicate = b'{"schema":"x","schema":"y"}'
    path.write_bytes(duplicate)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_joint_cost_read_schedule(
            manifest_path=path, manifest_sha256=hashlib.sha256(duplicate).hexdigest(),
            manifest_bytes=len(duplicate), plan_path="/mnt/shared/plan.json",
            plan_sha256=PLAN_SHA, prepared_path="/mnt/shared/prepared.json",
            prepared_sha256=PREPARED_SHA, retained_budget=_budget(),
            source_owner_cap_bytes=100, n_probes=4,
            target_names_by_layer=NAMES, validated_completed_units=())


def test_gzip_requires_one_complete_member(tmp_path):
    raw = json.dumps(_manifest(), separators=(",", ":")).encode()
    compressed = gzip.compress(raw, mtime=0) + gzip.compress(b"{}", mtime=0)
    path = tmp_path / "manifest.cas"
    path.write_bytes(compressed)
    with pytest.raises(ValueError, match="exactly one"):
        load_joint_cost_read_schedule(
            manifest_path=path, manifest_sha256=hashlib.sha256(compressed).hexdigest(),
            manifest_bytes=len(compressed), plan_path="/mnt/shared/plan.json",
            plan_sha256=PLAN_SHA, prepared_path="/mnt/shared/prepared.json",
            prepared_sha256=PREPARED_SHA, retained_budget=_budget(),
            source_owner_cap_bytes=100, n_probes=4,
            target_names_by_layer=NAMES, validated_completed_units=())


@pytest.mark.parametrize("mutate,reason", [
    (lambda m: m["annotations"].update(extra=True), "annotations"),
    (lambda m: m["annotations"].update(plan_sha256="c" * 64), "joint plan binding"),
    (lambda m: m["annotations"].update(validated_completed_units=1), "checkpoint binding"),
    (lambda m: m["annotations"]["windows"][0].update(active_pending_names=[]), "pending names"),
    (lambda m: m["annotations"]["windows"][0].update(statistics_bytes=1000), "retained budget"),
    (lambda m: m["annotations"]["windows"][0].update(window_index=1), "nonsequential index"),
    (lambda m: m["annotations"]["windows"][0].update(original_full_target_names=[NAMES[0][0]]), "full names"),
    (lambda m: m["annotations"].update(window_partition_sha256="0" * 64), "partition SHA-256"),
    (lambda m: m["read_plan"]["phases"][0].update(bytes=11), "byte accounting"),
    (lambda m: m["read_plan"]["phases"][0].update(entry_indices=[0, 0]), "repeated entry"),
    (lambda m: m["read_plan"]["phases"][1].update(name="cost_unknown"), "setup/head phase prefix"),
])
def test_mutated_manifest_refused(tmp_path, mutate, reason):
    manifest = copy.deepcopy(_manifest())
    mutate(manifest)
    with pytest.raises(ValueError, match=reason):
        _load(tmp_path, manifest)
