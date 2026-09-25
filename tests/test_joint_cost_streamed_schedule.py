"""Exercise the sealed read schedule inside real streamed retained COST."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

import prismaquant.aura_cost as aura
from prismaquant import io_engine
from prismaquant import format_registry as fr
from prismaquant.joint_cost_read_schedule import load_joint_cost_read_schedule
from prismaquant.joint_retained_window_plan import (
    RetainedWindowBudget, plan_retained_targets, targets_from_statistics_plan,
)
from prismaquant.joint_statistics_plan import plan_joint_statistics_target_windows
from test_joint_retained_streamed import _case


PLAN_SHA = "a" * 64
PREPARED_SHA = "b" * 64


def _binding_sha(completed):
    if not completed:
        return None
    value = {"schema": "prismaquant.joint_cost.validated_completed_units.v1",
             "plan_sha256": PLAN_SHA, "prepared_sha256": PREPARED_SHA,
             "units": sorted(completed)}
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _sealed_case_schedule(root: Path, runner, formats, cache, execution, completed, events):
    """Price the tiny fixture with production planners; point reads at its real files."""
    budget = RetainedWindowBudget.from_dict(execution["budget"])
    fmts = tuple(fr.canonical_format_name(fmt) for fmt in formats if fmt != "BF16")
    modules = {name: module for name, module in runner.model.named_modules()
               if name.endswith(".proj")}
    by_layer = {}
    for name in sorted(modules):
        by_layer.setdefault(runner.layer_index_for_qname(name), []).append(name)
    windows_by_layer = {}
    for layer, names in by_layer.items():
        specs = {name: {fmt: fr.get_format(fmt) for fmt in fmts} for name in names}
        stats = plan_joint_statistics_target_windows(
            {name: modules[name] for name in names}, specs,
            max_statistics_bytes=budget.statistics_cap_bytes,
            activation_max_abs=cache.activation_max_abs)
        keys = {name: tuple(cache.resolve_key(name, fmt) for fmt in fmts) for name in names}
        costs = {key: {"incoming_storage_bytes": cache.estimate_nbytes([key]),
                       "serialized_bytes": cache.estimate_nbytes([key])}
                 for values in keys.values() for key in values}
        targets = targets_from_statistics_plan(stats, keys, costs)
        windows_by_layer[layer] = plan_retained_targets(
            targets, budget=budget, source_bytes=execution["source_reserve_bytes"],
            footprint_scope="pwc_serialized_upper_bound").windows

    bound = root / "bound"
    bound.mkdir(parents=True, exist_ok=True)
    plan = bound / "plan.json"
    prepared = bound / "prepared.json"
    plan.write_text("{}")
    prepared.write_text("{}")
    entries, indices = [], {}
    for path in (plan, prepared):
        indices[str(path)] = len(entries)
        entries.append({"path": str(path), "offset": 0, "bytes": path.stat().st_size,
                        "sha256": None})
    for key in sorted(cache.weights):
        path = str(cache.weights[key])
        indices[path] = len(entries)
        entries.append({"path": path, "offset": 0, "bytes": Path(path).stat().st_size,
                        "sha256": None})

    phases = []
    cumulative = 0
    def add_phase(name, paths=()):
        nonlocal cumulative
        refs = [indices[str(path)] for path in paths]
        size = sum(entries[i]["bytes"] for i in refs)
        cumulative += size
        phases.append({"name": name, "entry_indices": refs, "bytes": size,
                       "cumulative_bytes": cumulative})

    add_phase("cost_setup", (plan,))
    add_phase("cost_head", (prepared,))
    for layer in range(runner.num_layers):
        add_phase(f"cost_capture_{layer:03d}")
    add_phase("cost_tail")
    windows = []
    for layer in reversed(range(runner.num_layers)):
        add_phase(f"cost_reverse_{layer:03d}_source")
        for index, window in enumerate(windows_by_layer.get(layer, ())):
            active = [name for name in window.names if name not in completed]
            phase = f"cost_reverse_{layer:03d}_window_{index:03d}"
            add_phase(phase, [cache.weights[cache.resolve_key(name, fmt)]
                              for name in active for fmt in fmts])
            windows.append({"phase": phase, "layer": layer, "window_index": index,
                            "original_full_target_names": list(window.names),
                            "active_pending_names": active,
                            "statistics_bytes": window.statistics_bytes,
                            "render_file_upper_bound_bytes": window.render_bytes,
                            "candidate_count": window.candidate_count})
    # A completed resume still declares the unique archive roster from the
    # original partition; put otherwise-unused files in setup as metadata hints.
    used = {i for phase in phases for i in phase["entry_indices"]}
    if len(used) != len(entries):
        missing = [i for i in range(len(entries)) if i not in used]
        phases[0]["entry_indices"].extend(missing)
        extra = sum(entries[i]["bytes"] for i in missing)
        for phase in phases:
            phase["cumulative_bytes"] += extra
        phases[0]["bytes"] += extra
        cumulative += extra
    partition = [[w["layer"], w["window_index"], w["original_full_target_names"]]
                 for w in windows]
    partition_sha = hashlib.sha256(json.dumps(partition, separators=(",", ":"),
                                              ensure_ascii=False).encode()).hexdigest()
    annotations = {
        "entry_point": "prismaquant.tessera_joint_aura:run", "mode": "retained_cost_v2",
        "plan": str(plan), "plan_sha256": PLAN_SHA,
        "prepared": str(prepared), "prepared_sha256": PREPARED_SHA,
        "source_owner_cap_bytes": execution["source_reserve_bytes"],
        "retained_budget": budget.as_dict(), "probes_per_window": 4,
        "source_prefetch_lookahead_layers": 1,
        "tail_retained_source_layers": list(range(max(0, runner.num_layers - 2), runner.num_layers)),
        "window_partition_sha256": partition_sha, "windows": windows,
        "validated_completed_units_sha256": _binding_sha(completed),
        "validated_completed_units": len(completed), "sha256_present": False,
        "sha256_absent_reason": "fixture file roster", "argv": None,
    }
    manifest = {"schema": "prismaquant.prismabuild.data_manifest.v2",
                "produced_by": {}, "mount_prefix": str(root), "annotations": annotations,
                "entries": entries, "entry_count": len(entries),
                "total_bytes": sum(row["bytes"] for row in entries),
                "read_plan": {"phases": phases, "read_bytes": cumulative}}
    wire = gzip.compress(json.dumps(manifest, separators=(",", ":")).encode(), mtime=0)
    cas = bound / "cost-v2.cas"
    cas.write_bytes(wire)
    schedule = load_joint_cost_read_schedule(
        manifest_path=cas, manifest_sha256=hashlib.sha256(wire).hexdigest(),
        manifest_bytes=len(wire), plan_path=plan, plan_sha256=PLAN_SHA,
        prepared_path=prepared, prepared_sha256=PREPARED_SHA,
        retained_budget=budget, source_owner_cap_bytes=execution["source_reserve_bytes"],
        n_probes=4, progress_callback=lambda phase, units: events.append(("phase", phase, units)))
    schedule.enter_phase("cost_setup", 0)
    schedule.enter_phase("cost_head", 0)
    return schedule


def _run_case(root, monkeypatch, *, checkpoint, resume=False, completed=()):
    original = aura.compute_aura_cost_streamed
    events = []
    holder = {}
    checkpoint = Path(checkpoint)
    def wrapped(runner, ids, formats, **kwargs):
        cache = kwargs["production_cache"]
        schedule = _sealed_case_schedule(root, runner, formats, cache,
                                         kwargs["retained_operator_windows"],
                                         completed, events)
        holder["schedule"] = schedule
        install = runner.context.install
        def source_install(layer, **options):
            events.append(("source_install", layer, schedule.current_phase))
            return install(layer, **options)
        runner.context.install = source_install
        return original(runner, ids, formats, **kwargs, cost_read_schedule=schedule)
    loaded = io_engine.load_file
    def read(path, limit, **kwargs):
        # Every render read goes through the IO engine (PQ #1294); the spy
        # forwards the digest binding and the decoder unchanged.
        events.append(("render_read", str(path), holder["schedule"].current_phase))
        return loaded(path, limit, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(aura, "compute_aura_cost_streamed", wrapped)
        patch.setattr(io_engine, "load_file", read)
        save = aura._write_aura_unit_checkpoint
        def write(*args, **kwargs):
            result = save(*args, **kwargs)
            files = list((checkpoint / "units").glob("*.pkl"))
            events.append(("checkpoint", kwargs["qname"], len(files)))
            return result
        patch.setattr(aura, "_write_aura_unit_checkpoint", write)
        result, reads, context = _case(root / "same", monkeypatch, True,
                                       checkpoint=checkpoint, resume=resume)
    return result, reads, context, events, holder["schedule"]


def test_real_streamed_cost_reports_before_reads_and_after_durable_windows(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    result, reads, context, events, schedule = _run_case(
        tmp_path, monkeypatch, checkpoint=checkpoint)
    assert reads and not context.active
    assert result["provenance"]["cost_read_schedule"] == schedule.identity
    phases = [event for event in events if event[0] == "phase"]
    # Every declared phase is entered once, in roster order; a durable window
    # re-reports its own phase with the new cumulative count (PB progress-v1).
    names = [event[1] for event in phases]
    assert [name for i, name in enumerate(names)
            if i == 0 or name != names[i - 1]] == list(schedule.phases)
    counts = [event[2] for event in phases]
    assert counts == sorted(counts) and counts[-1] == len(result["stats"])
    assert all(phase.startswith("cost_reverse_") and "_window_" in phase
               for kind, _, phase in events if kind == "render_read")
    for index, event in enumerate(events):
        if event[0] == "source_install":
            layer, phase = event[1:]
            assert phase in (f"cost_capture_{layer:03d}",
                             f"cost_reverse_{layer:03d}_source")
        if event[0] == "phase" and event[2] > 0:
            assert any(prior[0] == "checkpoint" and prior[2] >= event[2]
                       for prior in events[:index])


def test_partial_and_complete_resume_keep_original_windows(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    first, all_reads, _, _, first_schedule = _run_case(
        tmp_path, monkeypatch, checkpoint=checkpoint)
    pending = sorted(first["stats"])[0]
    aura._aura_unit_checkpoint_path(checkpoint, pending).unlink()
    completed = set(first["stats"]) - {pending}
    resumed, reads, context, events, schedule = _run_case(
        tmp_path, monkeypatch, checkpoint=checkpoint, resume=True, completed=completed)
    assert resumed["costs"] == first["costs"]
    assert len(reads) * 2 == len(all_reads) and context.install_calls > 0
    assert schedule.identity["window_partition_sha256"] == first_schedule.identity["window_partition_sha256"]
    assert sum(len(w.active_pending_names) for w in schedule.windows) == 1
    assert all(w.original_full_target_names == x.original_full_target_names
               for w, x in zip(schedule.windows, first_schedule.windows))
    resumed_counts = [event[2] for event in events if event[0] == "phase"]
    assert resumed_counts == sorted(resumed_counts)
    assert resumed_counts[-1] == len(first["stats"])

    complete, no_reads, context, complete_events, complete_schedule = _run_case(
        tmp_path, monkeypatch, checkpoint=checkpoint, resume=True,
        completed=set(first["stats"]))
    assert complete["costs"] == first["costs"]
    assert not no_reads and context.install_calls == 0
    assert complete_schedule.current_phase == "cost_head"
    assert [event[1] for event in complete_events if event[0] == "phase"] == [
        "cost_setup", "cost_head"]
