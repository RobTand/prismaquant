"""Producer tests: executable combined quantum read plan (PQ #862).

Self-contained tiny fixtures: a four-layer parent, real `layer_quanta`
records, real writer-built receipts. Proves the ONE executable v2
manifest covers checkpoint, chain/own source extents, boundary/probe/
replay reads in true consumption order with byte-exact accounting;
binding through the existing owners; dispatcher selection with derived
progress declarations; runtime read-phase reporting through the existing
semantic reporter. Full GPU execution belongs to a later lane, not here.
"""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from prismaquant.joint_adjoint_checkpoints import exact_entry_record  # noqa: E402
from prismaquant.joint_adjoint_checkpoints import (  # noqa: E402
    write_adjoint_checkpoint, write_adjoint_receipt)
from prismaquant.joint_layer_quanta import (  # noqa: E402
    ADJOINT_CAPTURE_SCHEMA,
    CHECKPOINT_LOAD_PHASE,
    MANIFEST_SCHEMA_V2,
    bind_adjoint_receipt,
    bind_quantum_executable,
    build_quantum_executable_manifest,
    emit_quantum_executable_readsets,
    executable_bound_phase_name,
    executable_own_source_phase_name,
    executable_replay_phase_name,
    executable_source_phase_name,
    layer_quanta,
    quantum_executable_phase_names,
    seal_manifest_bytes,
)
from prismaquant.perturbed_x_cache import (  # noqa: E402
    write_exact_activation_cache_entry)

N_PROBES = 2
N_BATCHES = 3
PREFETCH_BATCHES = 2
STRIDED = [2, 4]
STRIDED = [2, 4]
SESSION = {"generation": "fixture-gen-862",
           "run_identity_sha256": "cd" * 32}
CALIB = {"path": "/fixture/calib/tokens.pt", "bytes": 512,
         "sha256": "9" * 64}
RENDER_PREREQ = {"scope": "pb732", "production_pkl_sha256": "8" * 64,
                 "unit_roster_sha256": "7" * 64}


def _layers():
    return [0, 1, 2, 3]


def _tiny_parent():
    entries = [{"path": "/fixture/model/shard-h.pt", "offset": 0,
                "bytes": 100, "sha256": None}]
    phases = [{"name": "head", "bytes": 100, "cumulative_bytes": 100}]
    total = 100
    for layer in _layers():
        entries.append({"path": f"/fixture/model/shard-l{layer}.pt",
                        "offset": 0, "bytes": 200, "sha256": None})
        total += 200
        phases.append({"name": f"layer-{layer}", "bytes": 200,
                       "cumulative_bytes": total})
    return {"schema": "prismaquant.prismabuild.data_manifest.v1",
            "mount_prefix": "/mnt/shared",
            "entries": entries, "entry_count": len(entries),
            "total_bytes": total,
            "annotations": {
                "campaign_scope": {"fixture": "exec-862"},
                "layers": _layers(),
                "phases": phases}}


def _tiny_records(tmp_path):
    """Actual layer_quanta records (with identity/chunks/read_set)."""
    from prismaquant.joint_layer_quanta import layer_quanta as _produce
    parent = _tiny_parent()
    prepared = {"formats_by_qname": {
        f"model.layers.{layer}.mlp.gate_proj": {} for layer in _layers()}}
    out = _produce(
        {"model": "/fixture/model", "output_root": str(tmp_path / "run"),
         "distributed_campaign": {}},
        prepared, parent, chunk_target_bytes=200, stride=2,
        output_root=str(tmp_path / "run"),
        plan_path=str(tmp_path / "plan.json"),
        plan_sha256="0" * 64, prepared_path=str(tmp_path / "prep.json"),
        prepared_sha256="1" * 64, parent_manifest_sha256="2" * 64,
        window_partition={"windows_by_layer": {str(n): 2 for n in _layers()}},
        ram_window_gib=160, max_resident_consumers=2)
    return out["records"], parent


def _write_boundary(space, boundary, batch):
    ref = write_exact_activation_cache_entry(
        space / "entries",
        f"boundary-{batch}-{boundary}-at-{boundary}",
        torch.zeros(2, 4),
        identity={"session": dict(SESSION),
                  "slot": f"boundary-{batch}-{boundary}", "kind": "boundary",
                  "coordinates": {"batch": batch, "boundary": boundary,
                                  "probe": None}},
        max_tensor_bytes=1 << 20, max_file_bytes=1 << 20)
    return exact_entry_record(ref)


def _tiny_receipt(tmp_path, campaign):
    space = tmp_path / "adjoint"
    (space / "entries").mkdir(parents=True, exist_ok=True)
    boundary_entries = {}
    for boundary in (1, 2, 3):
        boundary_entries[str(boundary)] = [
            _write_boundary(space, boundary, batch)
            for batch in range(N_BATCHES)]
    checkpoints = []
    for boundary in STRIDED:
        checkpoints.append(write_adjoint_checkpoint(
            space, boundary=boundary,
            session={"generation": SESSION["generation"],
                     "kind": "adjoint_checkpoint",
                     "run_identity_sha256": SESSION["run_identity_sha256"]},
            cotangents={(p, b): torch.zeros(2, 4) for p in range(N_PROBES)
                        for b in range(N_BATCHES)},
            shared_adjoint={(p, b): {"scale": 1.0} for p in range(N_PROBES)
                            for b in range(N_BATCHES)},
            shared_pass={b: {"mask": [0, 1]} for b in range(N_BATCHES)}))
    return {
        "schema": ADJOINT_CAPTURE_SCHEMA,
        "run_identity": {"plan_sha256": campaign["plan_sha256"],
                         "prepared_sha256": campaign["prepared_sha256"],
                         "campaign_scope": campaign["campaign_scope"]},
        "boundary_storage": {
            "session": dict(SESSION),
            "policy": {"prefetch_batches": PREFETCH_BATCHES}},
        "boundary_entries": boundary_entries,
        "checkpoints": checkpoints,
        "status": "complete",
    }


def _layer2(tmp_path):
    """Layer 2: checkpoint 4, chain [3], two replay windows."""
    records, parent = _tiny_records(tmp_path)
    record = next(r for r in records if r["layer"] == 2)
    assert record["adjoint"]["checkpoint_boundary"] == 4
    assert record["adjoint"]["chain_layers"] == [3]
    receipt = _tiny_receipt(tmp_path, record["campaign"])
    return record, receipt, parent


def test_phase_names_frozen_consumption_order():
    names = quantum_executable_phase_names(
        [3], 2, n_probes=2, replay_windows=2)
    assert names == (
        "head", "checkpoint-load",
        "chain-003-source", "chain-003-bound",
        "own-002-source",
        "replay-00-p0", "replay-00-p1",
        "replay-01-p0", "replay-01-p1")
    assert executable_source_phase_name(3) == "chain-003-source"
    assert executable_bound_phase_name(3) == "chain-003-bound"
    assert executable_own_source_phase_name(2) == "own-002-source"
    assert executable_replay_phase_name(None, 1) == "replay-00-p1"
    assert CHECKPOINT_LOAD_PHASE == "checkpoint-load"


def test_build_covers_whole_consumption_corpus(tmp_path):
    record, receipt, parent = _layer2(tmp_path)
    manifest = build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite=dict(RENDER_PREREQ))
    assert manifest["schema"] == MANIFEST_SCHEMA_V2
    assert "phases" not in manifest["annotations"]
    names = [p["name"] for p in manifest["read_plan"]["phases"]]
    assert names == list(quantum_executable_phase_names(
        [3], 2, n_probes=N_PROBES, replay_windows=2))
    staged = {(e["path"], e["offset"]) for e in manifest["entries"]}
    # Chain + own source extents from the parent table.
    assert ("/fixture/model/shard-l3.pt", 0) in staged
    assert ("/fixture/model/shard-l2.pt", 0) in staged
    # Calibration intake and the checkpoint plane.
    assert (CALIB["path"], 0) in staged
    assert any("cotangent-0-0" in e["path"]
               for e in manifest["entries"])
    # Own boundary corpus repeats across both replay windows.
    replay = [p for p in manifest["read_plan"]["phases"]
              if p["name"].startswith("replay-")]
    assert len(replay) == 2 * N_PROBES
    first = {manifest["entries"][i]["path"] for i in replay[0]["entry_indices"]}
    assert first == {manifest["entries"][i]["path"]
                     for i in replay[-1]["entry_indices"]}
    assert manifest["read_plan"]["read_bytes"] == sum(
        p["bytes"] for p in manifest["read_plan"]["phases"])
    assert manifest["total_bytes"] == sum(
        e["bytes"] for e in manifest["entries"])
    assert manifest["annotations"]["render_prerequisite"]["scope"] == "pb732"


def test_build_refuses_contradictory_triples(tmp_path):
    record, receipt, parent = _layer2(tmp_path)
    parent = json.loads(json.dumps(parent))
    parent["entries"][3]["bytes"] = 201
    with pytest.raises(ValueError, match="disagree"):
        build_quantum_executable_manifest(
            record, receipt, parent, strided_boundaries=STRIDED,
            n_probes=N_PROBES, calib=dict(CALIB),
            render_prerequisite=dict(RENDER_PREREQ))


def test_build_refuses_incomplete_status(tmp_path):
    record, receipt, parent = _layer2(tmp_path)
    receipt["status"] = "running"
    with pytest.raises(ValueError, match="completed capture"):
        build_quantum_executable_manifest(
            record, receipt, parent, strided_boundaries=STRIDED,
            n_probes=N_PROBES, calib=dict(CALIB),
            render_prerequisite=dict(RENDER_PREREQ))


def test_seal_deterministic_and_pb_validated(tmp_path):
    core, tiers, _plans = _pb()
    record, receipt, parent = _layer2(tmp_path)
    kwargs = dict(strided_boundaries=STRIDED, n_probes=N_PROBES,
                  calib=dict(CALIB),
                  render_prerequisite=dict(RENDER_PREREQ))
    first = seal_manifest_bytes(
        build_quantum_executable_manifest(record, receipt, parent, **kwargs))
    second = seal_manifest_bytes(
        build_quantum_executable_manifest(record, receipt, parent, **kwargs))
    assert first == second
    manifest = build_quantum_executable_manifest(
        record, receipt, parent, **kwargs)
    for entry in manifest["entries"]:
        entry["path"] = "/mnt/shared/fixture" + entry["path"][len("/fixture"):]
    normalized = core.validate_data_manifest(manifest)
    ranges = tiers.manifest_phase_ranges(normalized)
    assert ranges, "published PB stages nothing for the executable manifest"
    assert [r["name"] for r in ranges] == [
        p["name"] for p in manifest["read_plan"]["phases"]]
    assert ranges[-1]["end_bytes"] == manifest["read_plan"]["read_bytes"]


def _pb():
    fleet = Path("/mnt/shared/prismabuild-fleet")
    try:
        receipt = json.loads(
            (fleet / "repo" / "RUNTIME_VERSION.json").read_text())
        generation = str(receipt["generation"])
        pinned = dict(receipt["files"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        pytest.skip(f"no PB runtime receipt: {exc}")
    src = fleet / "runtime-generations" / generation / "src"
    try:
        actual = hashlib.sha256(
            (src / "prismabuild" / "core.py").read_bytes()).hexdigest()
    except OSError as exc:
        pytest.skip(f"unreadable PB generation: {exc}")
    if actual != pinned.get("src/prismabuild/core.py"):
        pytest.skip("PB generation is not the published bytes")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import prismabuild.core as core
    import prismabuild.storage_tiers as tiers
    import prismabuild.residency_plan as plans
    for module in (core, tiers, plans):
        if not Path(module.__file__).resolve().is_relative_to(src.resolve()):
            pytest.skip("a different prismabuild is already imported")
    return core, tiers, plans


def _bound_inputs(tmp_path, root="/mnt/shared/run"):
    """Record bound to its receipt with resealed identity, plus manifest."""
    from prismaquant.joint_layer_quanta import canonical_sha256
    record, receipt, parent = _layer2(tmp_path)
    campaign = record["campaign"]
    digest = bind_adjoint_receipt(
        receipt, plan_sha256=campaign["plan_sha256"],
        prepared_sha256=campaign["prepared_sha256"],
        scope=campaign["campaign_scope"], checkpoints=STRIDED)
    record = dict(
        record, adjoint=dict(record["adjoint"], receipt_sha256=digest))
    body = {k: v for k, v in record.items() if k != "identity_sha256"}
    record["identity_sha256"] = canonical_sha256(body, where="fixture")
    manifest = build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite=dict(RENDER_PREREQ))
    wire_sha = hashlib.sha256(seal_manifest_bytes(manifest)).hexdigest()
    kwargs = dict(
        manifest=manifest,
        manifest_path=f"{root}/layer-quanta/adjoint/bound-readsets/"
                      "layer-002.executable.json.gz",
        manifest_sha256=wire_sha, output_root=root,
        strided_boundaries=STRIDED, n_probes=N_PROBES,
        calib=dict(CALIB), render_prerequisite=dict(RENDER_PREREQ))
    return record, receipt, parent, kwargs


def test_binder_recomputes_identity_and_anchors_rebuild(tmp_path):
    from prismaquant.joint_layer_quanta import (
        bind_quantum_executable, check_quantum_for_campaign,
        canonical_sha256)
    record, receipt, parent, kwargs = _bound_inputs(tmp_path)
    new = bind_quantum_executable(
        record, receipt, parent, **kwargs)
    assert new["executable_readset"]["manifest_sha256"] == kwargs[
        "manifest_sha256"]
    assert "executable_readset" not in record
    campaign = dict(record["campaign"],
                    adjoint_receipt_sha256=new["executable_readset"][
                        "receipt_sha256"])
    check_quantum_for_campaign(new, campaign)
    body = {k: v for k, v in new.items() if k != "identity_sha256"}
    assert canonical_sha256(body, where="quantum record") == new[
        "identity_sha256"]


def test_binder_refuses_tampered_input_identity(tmp_path):
    from prismaquant.joint_layer_quanta import bind_quantum_executable
    record, receipt, parent, kwargs = _bound_inputs(tmp_path)
    record["windows"].append({"window_index": 99})
    with pytest.raises(ValueError, match="identity does not recompute"):
        bind_quantum_executable(record, receipt, parent, **kwargs)


def test_binder_refuses_forged_triples_consistent_rehash(tmp_path):
    import copy
    from prismaquant.joint_layer_quanta import bind_quantum_executable
    record, receipt, parent, kwargs = _bound_inputs(tmp_path)
    forged = copy.deepcopy(kwargs["manifest"])
    forged["entries"][1]["bytes"] += 8
    forged["entries"][1]["sha256"] = "c" * 64
    total = 0
    for phase in forged["read_plan"]["phases"]:
        size = sum(forged["entries"][i]["bytes"]
                   for i in phase["entry_indices"])
        phase["bytes"] = size
        total += size
        phase["cumulative_bytes"] = total
    forged["total_bytes"] = sum(e["bytes"] for e in forged["entries"])
    forged["read_plan"]["read_bytes"] = total
    forged_sha = hashlib.sha256(seal_manifest_bytes(forged)).hexdigest()
    with pytest.raises(ValueError, match="do not originate"):
        bind_quantum_executable(
            record, receipt, parent, manifest=forged,
            manifest_path=kwargs["manifest_path"],
            manifest_sha256=forged_sha, output_root="/mnt/shared/run",
            strided_boundaries=STRIDED, n_probes=N_PROBES,
            calib=dict(CALIB), render_prerequisite=dict(RENDER_PREREQ))


def _dispatcher_record(tmp_path, manifest, wire_sha):
    adjoint_path = tmp_path / "adjoint.json"
    adjoint_path.write_text("{}")
    record_path = tmp_path / "record.json"
    record = {"quantum_id": "layer-002", "layer": 2,
              "campaign": {"plan_path": "plan.json", "plan_sha256": "0" * 64,
                           "prepared_path": "prep.json",
                           "prepared_sha256": "1" * 64},
              "read_set": {"manifest_path": str(tmp_path / "slice.gz"),
                           "manifest_sha256": hashlib.sha256(
                               b"slice").hexdigest()},
              "chunks": [{"name": "layer-002-c000"}],
              "adjoint": {}}
    (tmp_path / "slice.gz").write_bytes(b"slice")
    manifest = copy.deepcopy(manifest)
    manifest["annotations"]["render_prerequisite"]["binding"] = {
        "scope": "pb732", "material": "e" * 64}
    manifest_path = tmp_path / "exec.json.gz"
    manifest_path.write_bytes(seal_manifest_bytes(manifest))
    record["executable_readset"] = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()).hexdigest(),
        "phases": [p["name"] for p in manifest["read_plan"]["phases"]]}
    record_path.write_text("{}")
    return record, record_path, adjoint_path


def test_dispatcher_selects_executable_and_declares_phases(tmp_path,
                                                           monkeypatch):
    import dispatch_joint_quanta as dispatch
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"container": {"image": "sha256:" + "0" * 64},
                                "env": {}}))
    monkeypatch.setattr(dispatch, "SPEC_PATH", spec)
    record, receipt, parent = _layer2(tmp_path)
    manifest = build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite=dict(RENDER_PREREQ))
    wire_sha = hashlib.sha256(seal_manifest_bytes(manifest)).hexdigest()
    bound, record_path, adjoint_path = _dispatcher_record(
        tmp_path, manifest, wire_sha)
    argv = dispatch.quantum_argv(
        bound, record_path=record_path, output_root=tmp_path,
        adjoint_path=adjoint_path)
    assert argv[argv.index("--data-manifest") + 1].endswith("exec.json.gz")
    assert argv[argv.index("--data-manifest-sha256") + 1] == hashlib.sha256(
        Path(argv[argv.index("--data-manifest") + 1]).read_bytes()
    ).hexdigest()
    declared = [argv[i + 1] for i, word in enumerate(argv[:-1])
                if word == "--progress-phase"]
    assert declared[0].startswith("head=")
    assert declared[1].startswith("checkpoint-load=")
    assert [name.split("=")[0] for name in declared] == (
        ["head"] + [p["name"]
                    for p in manifest["read_plan"]["phases"]
                    if p["name"] != "head"])
    # Strict-owned lanes are byte-identical in both selections.
    legacy = dict(bound)
    del legacy["executable_readset"]
    argv_legacy = dispatch.quantum_argv(
        legacy, record_path=record_path, output_root=tmp_path,
        adjoint_path=adjoint_path)

    def _lane(argv, *flags):
        return [argv[i + 1] for i, word in enumerate(argv[:-1])
                if word in flags]
    for flags in (("--residency",), ("--residency-ram",), ("--tag",),
                  ("--demand",), ("--env",)):
        assert _lane(argv, *flags) == _lane(argv_legacy, *flags)
    assert argv_legacy[argv_legacy.index("--data-manifest") + 1].endswith(
        "slice.gz")


def test_runtime_reports_read_phases_without_pricing_units():
    from prismaquant.joint_cost_quantum import ChunkFrontier, QuantumProgress
    frontier = ChunkFrontier(
        chunks=[{"name": "layer-002-c000", "start_bytes": 0,
                 "end_bytes": 200}],
        windows=[{"render_file_upper_bound_bytes": 200}])
    progress = QuantumProgress(frontier=frontier, base_units=0,
                               log=lambda message: None)
    progress._phases = ["head", "checkpoint-load",
                        "chain-003-source", "chain-003-bound",
                        "own-002-source", "replay-00-p0"]
    progress.enter_head(3)
    units = progress.units()
    assert progress._committed == (units, "head")
    progress.enter_read_phase("checkpoint-load")
    assert progress._phase == "checkpoint-load"
    assert progress.units() == units
    assert progress._committed == (units, "checkpoint-load")
    progress.enter_read_phase("no-such-phase")
    assert progress._phase == "checkpoint-load"
    assert progress.units() == units


def test_replay_phase_zero_pending_convention():
    from prismaquant.joint_cost_quantum import (
        executable_replay_phase_name as _replay)
    assert _replay(None, 1) == "replay-00-p1"
    assert _replay(2, 0) == "replay-02-p0"


def _acceptance_setup(tmp_path, monkeypatch):
    """Real tiny CPU campaign: runner, capture receipt, producer records.

    Returns a dict with everything the acceptance runs need. Every file
    below is real bytes on disk; every record is real producer output.
    """
    import re as _re

    import test_joint_cost_quantum_runtime as rt
    import test_layer_major_boundary_capture as lm
    import prismaquant.aura_cost as aura
    from prismaquant.joint_layer_quanta import roster_digest

    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    torch.manual_seed(85)
    model, context, runner, cache = lm.fixture()
    context.settle_prefetched_layers = lambda layers: None
    context.source_residency_snapshot = lambda layers, include_head=False: {
        "owners": [], "unique_storage_bytes": sum(
            p.numel() * p.element_size() for p in model.parameters())}
    cache, _proofs = rt._prepared_cache(
        model, context, runner, cache, tmp_path / "shared")

    layer_files = {}
    for (name, _fmt), path in cache.weights.items():
        match = _re.match(r"^model\.layers\.(\d+)\.", name)
        assert match is not None, name
        layer_files.setdefault(int(match.group(1)), []).append(path)
    assert sorted(layer_files) == [0, 1]
    calib_path = tmp_path / "calib.pt"
    torch.save(lm.draw(), calib_path)
    calib_sha = hashlib.sha256(calib_path.read_bytes()).hexdigest()

    entries = [{"path": str(calib_path), "offset": 0,
                "bytes": calib_path.stat().st_size, "sha256": calib_sha}]
    phases = [{"name": "head", "bytes": entries[0]["bytes"],
               "cumulative_bytes": entries[0]["bytes"]}]
    total = entries[0]["bytes"]
    for layer in (0, 1):
        start = total
        for path in sorted(layer_files[layer]):
            size = Path(path).stat().st_size
            entries.append({
                "path": path, "offset": 0, "bytes": size,
                "sha256": hashlib.sha256(
                    Path(path).read_bytes()).hexdigest()})
            total += size
        phases.append({"name": f"layer-{layer}", "bytes": total - start,
                       "cumulative_bytes": total})
    scope = {"fixture": "exec-acceptance"}
    parent = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "mount_prefix": "/mnt/shared",
        "entries": entries, "entry_count": len(entries),
        "total_bytes": total,
        "annotations": {"campaign_scope": scope, "layers": [0, 1],
                        "phases": phases},
    }
    plan = {"output_root": str(tmp_path / "campaign"),
            "model": str(tmp_path / "shared" / "assets"),
            "distributed_campaign": {},
            "execution": {"n_probes": 4},
            "calibration_input": {"path": str(calib_path),
                                  "sha256": calib_sha}}
    qnames = [f"model.layers.{layer}.proj" for layer in (0, 1)]
    prepared = {"formats_by_qname": {name: ["BF16"] for name in qnames}}

    def _sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(json.dumps(prepared, sort_keys=True))
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent, sort_keys=True))
    plan_sha, prepared_sha, parent_sha = (
        _sha(plan_path), _sha(prepared_path), _sha(parent_path))
    import test_joint_cost_quantum_runtime as rt
    preflight = rt._preflight_windows(runner, cache, None)
    partition = {"windows_by_layer": {
        str(layer): len(preflight[layer]) for layer in (0, 1)}}
    from prismaquant.joint_layer_quanta import (
        layer_quanta as _produce, roster_digest)
    torch.manual_seed(85)
    _model_b, _context_b, runner_b, _cache_b = lm.fixture()
    _context_b.settle_prefetch_layers = lambda layers: None
    receipt = run_adjoint_capture_core_alias(
        runner_b, lm.draw(), tmp_path, plan_sha, prepared_sha, parent_sha,
        roster_digest(sorted(prepared["formats_by_qname"])))
    assert receipt.get("status") == "complete"
    assert sorted(receipt["boundary_entries"]) == ["0", "1"]
    bound = _produce(
        plan, prepared, parent, chunk_target_bytes=1 << 20, stride=2,
        output_root=str(tmp_path / "campaign"),
        plan_path=str(plan_path), plan_sha256=plan_sha,
        prepared_path=str(prepared_path), prepared_sha256=prepared_sha,
        parent_manifest_sha256=parent_sha,
        window_partition=partition,
        ram_window_gib=160, max_resident_consumers=2,
        adjoint_receipt=receipt)
    assert all(r["adjoint"]["receipt_sha256"] is not None
               for r in bound["records"])
    production = {
        "weights": {f"{name}@{fmt}": {
            "path": path,
            "sha256": hashlib.sha256(
                Path(path).read_bytes()).hexdigest()}
            for (name, fmt), path in cache.weights.items()}}
    production_path = tmp_path / "production.json"
    production_path.write_text(json.dumps(production, sort_keys=True))
    render_prerequisite = {
        "scope": "pb732",
        "production_pkl_sha256": hashlib.sha256(
            production_path.read_bytes()).hexdigest(),
        "unit_roster_sha256": bound["records"][0]["campaign"][
            "unit_roster_sha256"],
        "binding": None,
    }
    return {"runner": runner, "cache": cache,
            "records": {r["quantum_id"]: r for r in bound["records"]},
            "receipt": receipt, "parent": parent, "plan": plan,
            "layer_files": layer_files, "calib_path": calib_path,
            "render_prerequisite": render_prerequisite,
            "output_root": tmp_path / "campaign"}


def run_adjoint_capture_core_alias(runner, calib, tmp_path, plan_sha,
                                   prepared_sha, parent_sha, roster_sha):
    import test_joint_cost_quantum_runtime as rt
    import test_streamed_cost_checkpoints as tsc
    import prismaquant.aura_cost as aura
    from prismaquant.joint_cost_stage_a import run_adjoint_capture_core
    return run_adjoint_capture_core(
        runner, calib, execution=rt._execution(tmp_path / "capture"),
        output_root=str(tmp_path / "campaign"), stride=2,
        source_model_identity=tsc._model_identity("joint-source"),
        unit_roster_sha256=roster_sha, plan_sha256=plan_sha,
        prepared_sha256=prepared_sha, read_manifest_sha256=parent_sha,
        implementation_sha256=aura._aura_source_sha256(),
        campaign_scope={"fixture": "exec-acceptance"})


def _check_event_order(events, manifest, *, layer, chain):
    """Fail-closed order checker: every bulk read occurs under its
    already-reported owning phase. A hook moved after its read flips an
    event pair and fails here -- that is the mutation sensitivity.

    The candidate retained_window must already hold when a replay phase
    is entered: the first replay report requires a preceding window-open,
    and every replay boundary read requires both.
    """
    names = [p["name"] for p in manifest["read_plan"]["phases"]]
    by_path = {e["path"]: e for e in manifest.get("entries", [])}
    current = None
    seen_window_open = False
    for event in events:
        kind = event[0]
        if kind == "report":
            assert event[1] in names, event[1]
            if event[1].startswith("replay-"):
                assert seen_window_open, (
                    f"replay phase {event[1]!r} entered before any "
                    "retained_window opened")
            current = event[1]
        elif kind == "checkpoint-open":
            assert current == "checkpoint-load", events
        elif kind == "source-open":
            _, opened, _status = event
            if opened in chain:
                assert current == f"chain-{opened:03d}-source", events
            else:
                assert opened == layer, events
                assert current == f"own-{layer:03d}-source", events
        elif kind == "boundary-open":
            _, opened, _count = event
            if opened in chain:
                assert current == f"chain-{opened:03d}-bound", events
            else:
                assert opened == layer, events
                assert seen_window_open, events
                assert current is not None and current.startswith(
                    "replay-"), events
        elif kind == "setup-open":
            pass
        elif kind == "window-open":
            seen_window_open = True
        elif kind == "checkpoint-plane-open":
            assert current == "checkpoint-load", events
        elif kind == "boundary-path-open":
            _, path, file_bytes, sha256 = event
            assert path in by_path, path
            entry = by_path[path]
            assert entry["offset"] == 0, (path, entry)
            assert entry["bytes"] == file_bytes, (path, entry, file_bytes)
            assert entry["sha256"] == sha256, (path, entry)
        else:
            raise AssertionError(f"unknown event {event!r}")


def _drive_quantum(tmp_path, monkeypatch, setup, *, layer, resume):
    """One real run_layer_quantum_core with instrumented seams."""
    import contextlib

    import test_joint_cost_quantum_runtime as rt
    import prismaquant.joint_cost_quantum as qc
    import prismaquant.perturbed_x_cache as pxc
    from prismaquant import prismabuild_progress as pbprog
    from prismaquant.joint_cost_quantum import (
        ChunkFrontier, QuantumCounters, QuantumProgress,
        quantum_layer_roster, quantum_retained_state,
        resolve_quantum_windows, run_layer_quantum_core)

    runner = setup["runner"]
    cache = setup["cache"]
    record = setup["records"][f"layer-{layer:03d}"]
    receipt = setup["receipt"]
    events = []

    def _report(phase, units, **kwargs):
        events.append(("report", phase, units))
        return True

    monkeypatch.setattr(pbprog, "report", _report)
    orig_install = runner.context.install

    def install_logged(layer, *, require_prefetched=False,
                       prefetch_following=True):
        events.append(("source-open", int(layer), "installed"))
        return orig_install(
            layer, require_prefetched=require_prefetched,
            prefetch_following=prefetch_following)

    runner.context.install = install_logged
    orig_load = qc.load_adjoint_checkpoint

    def load_logged(space, checkpoint_record):
        events.append(("checkpoint-open",))
        return orig_load(space, checkpoint_record)

    monkeypatch.setattr(qc, "load_adjoint_checkpoint", load_logged)
    orig_prefetch = pxc.prefetch_exact_activation_cache_entries

    def prefetch_logged(references, **kwargs):
        bounds = set()
        plane = True
        for ref in references:
            parts = ref.name.split("-")
            if parts[0] == "boundary":
                plane = False
                bounds.add(int(parts[2]))
            events.append(("boundary-path-open", ref.path,
                           int(ref.file_bytes), str(ref.sha256)))
        if plane:
            events.append(("checkpoint-plane-open", len(references)))
        else:
            assert len(bounds) == 1
            events.append(("boundary-open", next(iter(bounds)),
                           len(references)))
        return orig_prefetch(references, **kwargs)

    monkeypatch.setattr(
        pxc, "prefetch_exact_activation_cache_entries", prefetch_logged)
    orig_rw = type(cache).retained_window

    @contextlib.contextmanager
    def rw_logged(self, *args, **kwargs):
        events.append(("window-open",))
        with orig_rw(self, *args, **kwargs) as receipt_obj:
            yield receipt_obj

    monkeypatch.setattr(type(cache), "retained_window", rw_logged)

    execution = rt._execution(tmp_path / f"qexec-{layer}-{int(resume)}")
    retained = quantum_retained_state(execution)
    roster = quantum_layer_roster(
        runner, {f"model.layers.{i}.proj": list(rt.FORMATS)
                 for i in range(runner.num_layers)}, layer)
    resolved = resolve_quantum_windows(
        record, layer=layer, names=roster.names, linears=roster.linears,
        render_formats=roster.render_formats, production_cache=cache,
        operator_windows=retained.operator_windows,
        retained_budget=retained.retained_budget,
        source_bytes=retained.source_bytes)
    frontier = ChunkFrontier(chunks=record["chunks"], windows=resolved)
    counters = QuantumCounters(
        quantum_id=record["quantum_id"],
        identity_sha256=record["identity_sha256"],
        chunks=record["chunks"], frontier=frontier)
    calib = {"path": str(setup["calib_path"]),
             "bytes": setup["calib_path"].stat().st_size,
             "sha256": hashlib.sha256(
                 setup["calib_path"].read_bytes()).hexdigest()}
    manifest = build_quantum_executable_manifest(
        record, receipt, setup["parent"], strided_boundaries=[2],
        n_probes=4, calib=dict(calib),
        render_prerequisite=dict(setup["render_prerequisite"]))
    # Bind the executable block exactly as the post-capture regen does,
    # and run the bound generation: the hooks follow the block.
    from prismaquant.joint_layer_quanta import bind_quantum_executable
    output_root = str(setup["output_root"])
    manifest_path = (
        f"{output_root}/layer-quanta/adjoint/bound-readsets/"
        f"{record['quantum_id']}.executable.json.gz")
    wire_sha256 = hashlib.sha256(
        seal_manifest_bytes(manifest)).hexdigest()
    record = bind_quantum_executable(
        record, receipt, setup["parent"], manifest=manifest,
        manifest_path=manifest_path, manifest_sha256=wire_sha256,
        output_root=output_root, strided_boundaries=[2], n_probes=4,
        calib=dict(calib),
        render_prerequisite=dict(setup["render_prerequisite"]))
    monkeypatch.setenv(
        "PRISMABUILD_ACTION_PROGRESS_PHASES",
        json.dumps([p["name"] for p in manifest["read_plan"]["phases"]]))
    progress = QuantumProgress(frontier=frontier, base_units=0)
    calib_ids = torch.load(setup["calib_path"])
    events.append(("setup-open", str(setup["calib_path"])))
    payload = run_layer_quantum_core(
        runner, cache, calib_ids,
        {f"model.layers.{i}.proj": list(rt.FORMATS)
         for i in range(runner.num_layers)},
        record=record, receipt=receipt, execution=execution,
        output_root=setup["output_root"], projection_backend=None,
        resume=resume, resolved_windows=resolved,
        counters=counters, progress=progress)
    assert payload["costs"], "quantum produced no cost rows"
    return events, manifest


def test_acceptance_real_quantum_reports_before_reads(tmp_path, monkeypatch):
    """Real tiny-CPU quantum sequencing: phase before first payload read.

    Reuses the existing tiny runner fixture (``test_layer_major_boundary_
    capture.fixture``), the prepared cache helper and ``run_layer_quantum_
    core`` from ``test_joint_cost_quantum_runtime`` -- no pretend execution.
    Layer 0 carries a nonempty chain ([1]); every run uses 4 probes; the
    replay corpus repeats per (window, probe); one resume run exercises the
    zero-pending window-zero convention. The actual ``QuantumProgress``
    reporter is instrumented (not hand-appended phase names); the actual PB
    ``residency_plan.accepted/remaining`` reader stages the manifest-derived
    phases. Every observed bulk path/bytes/digest must match the manifest,
    and moving a hook after its read fails the order checker.
    """
    setup = _acceptance_setup(tmp_path, monkeypatch)
    record0 = setup["records"]["layer-000"]
    assert list(record0["adjoint"]["chain_layers"]) == [1]
    assert len(record0["windows"]) >= 1
    assert setup["receipt"].get("status") == "complete"
    events, manifest = _drive_quantum(
        tmp_path, monkeypatch, setup, layer=0, resume=False)
    record = setup["records"]["layer-000"]
    _assert_acceptance_run(events, manifest, record, tmp_path,
                           expect_replay_windows="all",
                           layer_files=setup["layer_files"])
    # Four probes => four replay phases per window at minimum.
    replayed = [n for n in _reported(events) if n.startswith("replay-")]
    assert len(replayed) >= 4, replayed
    assert {n.split("-")[2] for n in replayed} == {"p0", "p1", "p2", "p3"}
    # Resume: all windows complete, zero-pending replay under window zero.
    events2, _ = _drive_quantum(
        tmp_path, monkeypatch, setup, layer=0, resume=True)
    _assert_acceptance_run(events2, manifest, record, tmp_path,
                           expect_replay_windows={0},
                           layer_files=setup["layer_files"])
    # No-chain path still works.
    events3, manifest1 = _drive_quantum(
        tmp_path, monkeypatch, setup, layer=1, resume=False)
    _assert_acceptance_run(
        events3, manifest1, setup["records"]["layer-001"], tmp_path,
        expect_replay_windows="all",
        layer_files=setup["layer_files"])
    assert not [n for n in _reported(events3) if n.startswith("chain-")]



def test_event_order_checker_rejects_post_read_reports():
    assert _reported([]) == []
    with pytest.raises(AssertionError):
        _check_event_order(
            [("report", "head", 0),
             ("boundary-open", 1, 5),
             ("report", "chain-001-bound", 0)],
            {"read_plan": {"phases": [
                {"name": "head", "entry_indices": []},
                {"name": "chain-001-bound", "entry_indices": []}]}},
            layer=0, chain=[1])


def _reported(events):
    return [event[1] for event in events
            if event[0] == "report" and len(event) >= 2]


def _assert_acceptance_run(events, manifest, record, tmp_path,
                           expect_replay_windows, layer_files):
    _, _, plans = _pb()
    names = [p["name"] for p in manifest["read_plan"]["phases"]]
    reported = _reported(events)
    assert reported and reported[0] == "head"
    # Same-phase repeats are unit advancements, not order violations.
    deduped = [reported[0]] + [
        phase for phase, prev in zip(reported[1:], reported) if phase != prev]
    assert deduped == [n for n in names if n in deduped]
    _check_event_order(
        events, manifest, layer=record["layer"],
        chain=list(record["adjoint"]["chain_layers"]))
    staged = {e["path"]: e for e in manifest["entries"]}
    for kind, *rest in events:
        if kind == "boundary-path-open":
            path, file_bytes, sha256 = rest
            assert path in staged, path
            assert staged[path]["bytes"] == file_bytes, (path, staged[path])
            assert staged[path]["sha256"] == sha256, (path, staged[path])
        elif kind == "setup-open":
            (path,) = rest
            assert path in staged, path
            assert Path(path).stat().st_size == staged[path]["bytes"]
            assert hashlib.sha256(
                Path(path).read_bytes()).hexdigest() == staged[path]["sha256"]
    installed = {e[1] for e in events if e[0] == "source-open"}
    for layer in installed:
        for path in layer_files[layer]:
            assert path in staged, path
            assert Path(path).stat().st_size == staged[path]["bytes"], path
            assert hashlib.sha256(
                Path(path).read_bytes()).hexdigest() == staged[path][
                    "sha256"], path
    units = [e[2] for e in events if e[0] == "report"]
    assert all(b >= a for a, b in zip(units, units[1:]))
    plan_view = {"phases": [{"name": name} for name in names]}
    for name in deduped:
        assert plans.accepted(plan_view, name)
    remaining = [p["name"] for p in plans.remaining(
        plan_view, deduped[-1])]
    assert remaining == names[names.index(deduped[-1]) + 1:]
    replayed = {int(n.split("-")[1])
                for n in reported if n.startswith("replay-")}
    if expect_replay_windows == "all":
        assert replayed == set(range(len(record["windows"])))
    else:
        assert replayed <= set(expect_replay_windows)
    assert manifest["schema"] == "prismaquant.prismabuild.data_manifest.v2"


def _exec_campaign(tmp_path):
    """Tiny campaign files with a calibration input for the regen CLI."""
    import regenerate_joint_quanta as _regen  # noqa: F401
    root = tmp_path / "campaign"
    calib_path = tmp_path / "calib.pt"
    calib_path.write_bytes(b"\x00" * 512)
    plan = {"output_root": str(root), "model": "/fixture/model",
            "distributed_campaign": {},
            "execution": {"n_probes": N_PROBES},
            "calibration_input": {
                "path": str(calib_path), "sha256": "a" * 64}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    prepared = {"formats_by_qname": {
        "model.layers.0.mlp.gate_proj": {},
        "model.layers.1.mlp.gate_proj": {},
        "model.layers.2.mlp.gate_proj": {},
        "model.layers.3.mlp.gate_proj": {}},
        "production_cache": {"path": str(tmp_path / "production.pkl"),
                             "sha256": "b" * 64}}
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(json.dumps(prepared, sort_keys=True))
    entries = [{"path": "/fixture/model/shard-h.pt", "offset": 0,
                "bytes": 100, "sha256": None}]
    phases = [{"name": "head", "bytes": 100, "cumulative_bytes": 100}]
    total = 100
    for layer in (0, 1, 2, 3):
        entries.append({"path": f"/fixture/model/shard-l{layer}.pt",
                        "offset": 0, "bytes": 200, "sha256": None})
        total += 200
        phases.append({"name": f"layer-{layer}", "bytes": 200,
                       "cumulative_bytes": total})
    parent = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "produced_by": {"plan": str(plan_path)},
        "mount_prefix": "/mnt/shared",
        "entries": entries, "entry_count": len(entries),
        "total_bytes": total,
        "annotations": {
            "campaign_scope": {"fixture": "exec-cli"},
            "argv": ["python3", "-m", "prismaquant.joint_adjoint_capture",
                     "--prepared", str(prepared_path),
                     "--prepared-sha256", hashlib.sha256(
                         prepared_path.read_bytes()).hexdigest()],
            "layers": [0, 1, 2, 3],
            "phases": phases}}
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent, sort_keys=True))

    def _sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    derivation = tmp_path / "derivation.json"
    derivation.write_text(json.dumps(
        {"chunk_target_bytes": 200, "stride": 2, "ram_window_gib": 160,
         "max_resident_consumers": 2}, sort_keys=True))
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps(
        {"windows_by_layer": {str(n): 2 for n in (0, 1, 2, 3)}},
        sort_keys=True))
    return {"root": root, "plan_path": plan_path, "plan_sha": _sha(plan_path),
            "prepared_path": prepared_path, "prepared_sha": _sha(
                prepared_path),
            "parent_path": parent_path, "parent_sha": _sha(parent_path),
            "derivation": derivation, "partition": partition}


def _exec_argv(tmp_path, campaign):
    return ["--plan", str(campaign["plan_path"]),
            "--plan-sha256", campaign["plan_sha"],
            "--prepared", str(campaign["prepared_path"]),
            "--prepared-sha256", campaign["prepared_sha"],
            "--parent-manifest", str(campaign["parent_path"]),
            "--parent-manifest-sha256", campaign["parent_sha"],
            "--derivation", str(campaign["derivation"]),
            "--partition", str(campaign["partition"])]


def _exec_receipt(tmp_path, campaign):
    from prismaquant.joint_layer_quanta import bind_adjoint_receipt
    space = tmp_path / "adjoint-space"
    space.mkdir(exist_ok=True)
    (space / "entries").mkdir(parents=True, exist_ok=True)
    scope = {"fixture": "exec-cli"}
    boundary_entries = {}
    for boundary in (0, 1, 2, 3):
        rows = []
        for batch in range(N_BATCHES):
            ref = write_exact_activation_cache_entry(
                space / "entries",
                f"boundary-{batch}-{boundary}-at-{boundary}",
                torch.zeros(2, 4),
                identity={"session": dict(SESSION),
                          "slot": f"boundary-{batch}-{boundary}",
                          "kind": "boundary",
                          "coordinates": {"batch": batch,
                                          "boundary": boundary,
                                          "probe": None}},
                max_tensor_bytes=1 << 20, max_file_bytes=1 << 20)
            rows.append(exact_entry_record(ref))
        boundary_entries[str(boundary)] = rows
    checkpoints = []
    for boundary in (2, 4):
        checkpoints.append(write_adjoint_checkpoint(
            space, boundary=boundary,
            session={"generation": SESSION["generation"],
                     "kind": "adjoint_checkpoint",
                     "run_identity_sha256": SESSION["run_identity_sha256"]},
            cotangents={(p, b): torch.zeros(2, 4) for p in range(N_PROBES)
                        for b in range(N_BATCHES)},
            shared_adjoint={(p, b): {"scale": 1.0} for p in range(N_PROBES)
                            for b in range(N_BATCHES)},
            shared_pass={b: {"mask": [0, 1]} for b in range(N_BATCHES)}))
    receipt = {
        "schema": ADJOINT_CAPTURE_SCHEMA,
        "run_identity": {"plan_sha256": campaign["plan_sha"],
                         "prepared_sha256": campaign["prepared_sha"],
                         "campaign_scope": scope},
        "boundary_storage": {
            "session": dict(SESSION),
            "policy": {"prefetch_batches": PREFETCH_BATCHES}},
        "boundary_entries": boundary_entries,
        "checkpoints": checkpoints,
        "status": "complete",
    }
    # The campaign scope lives in the parent annotations; the receipt
    # must answer for it, so seal the parent scope into the campaign.
    receipt["run_identity"]["campaign_scope"] = scope
    return receipt, space


def test_cli_executable_readsets_end_to_end(tmp_path):
    import regenerate_joint_quanta as regen
    from prismaquant import joint_cost_quantum as quantum
    from prismaquant.joint_layer_quanta import check_quantum_for_campaign
    campaign = _exec_campaign(tmp_path)
    receipt, space = _exec_receipt(tmp_path, campaign)
    receipt_path = space / "adjoint-capture.json"
    write_adjoint_receipt(space, receipt)
    # Seal the parent scope the receipt answers for.
    parent = json.loads(campaign["parent_path"].read_text())
    parent["annotations"]["campaign_scope"] = {"fixture": "exec-cli"}
    campaign["parent_path"].write_text(json.dumps(parent, sort_keys=True))
    campaign["parent_sha"] = hashlib.sha256(
        campaign["parent_path"].read_bytes()).hexdigest()
    out = tmp_path / "reviewed"
    assert regen.main(
        _exec_argv(tmp_path, campaign)
        + ["--output-root", str(campaign["root"]),
           "--records-out", str(out),
           "--adjoint-receipt", str(receipt_path),
           "--executable-readsets"]) == 0
    canonical = bind_adjoint_receipt(
        receipt, plan_sha256=campaign["plan_sha"],
        prepared_sha256=campaign["prepared_sha"],
        scope={"fixture": "exec-cli"}, checkpoints=[2, 4])
    records = [json.loads(path.read_text())
               for path in sorted(out.glob("layer-*.json"))]
    assert len(records) == 4
    assert (out / "records.json").is_file()
    for record in records:
        bound = record["executable_readset"]
        manifest_path = Path(bound["manifest_path"])
        assert manifest_path.is_file()
        assert manifest_path.parent.name == "bound-readsets"
        assert manifest_path.name.endswith(".executable.json.gz")
        assert hashlib.sha256(
            manifest_path.read_bytes()).hexdigest() == bound[
            "manifest_sha256"]
        assert bound["receipt_sha256"] == canonical
        campaign_binding = {
            "plan_sha256": campaign["plan_sha"],
            "prepared_sha256": campaign["prepared_sha"],
            "read_manifest_sha256": campaign["parent_sha"],
            "unit_roster_sha256": record["campaign"][
                "unit_roster_sha256"],
            "campaign_scope": {"fixture": "exec-cli"},
            "adjoint_receipt_sha256": canonical,
        }
        check_quantum_for_campaign(record, campaign_binding)
        quantum_path = out / f"{record['quantum_id']}.json"
        quantum.verify_quantum_identity(
            quantum_path=quantum_path,
            quantum_sha256=hashlib.sha256(
                quantum_path.read_bytes()).hexdigest(),
            plan_path=campaign["plan_path"],
            plan_sha256=campaign["plan_sha"],
            prepared_path=campaign["prepared_path"],
            prepared_sha256=campaign["prepared_sha"],
            adjoint_path=receipt_path,
            adjoint_sha256=hashlib.sha256(
                receipt_path.read_bytes()).hexdigest(),
            output_root=Path(str(campaign["root"])))


def test_cli_executable_refusal_writes_nothing(tmp_path):
    import regenerate_joint_quanta as regen
    campaign = _exec_campaign(tmp_path)
    receipt, space = _exec_receipt(tmp_path, campaign)
    del receipt["boundary_entries"]["3"]
    bad = space / "bad-receipt.json"
    bad.write_text(json.dumps(receipt, sort_keys=True))
    parent = json.loads(campaign["parent_path"].read_text())
    parent["annotations"]["campaign_scope"] = {"fixture": "exec-cli"}
    campaign["parent_path"].write_text(json.dumps(parent, sort_keys=True))
    campaign["parent_sha"] = hashlib.sha256(
        campaign["parent_path"].read_bytes()).hexdigest()
    out = tmp_path / "reviewed"
    assert regen.main(
        _exec_argv(tmp_path, campaign)
        + ["--output-root", str(campaign["root"]),
           "--records-out", str(out),
           "--adjoint-receipt", str(bad),
           "--executable-readsets"]) == 3
    assert not out.exists()
    assert not (Path(str(campaign["root"])) / "layer-quanta" / "adjoint" /
                "bound-readsets").exists()


def _gate_manifest(tmp_path, manifest, binding):
    import copy
    manifest = copy.deepcopy(manifest)
    manifest["annotations"]["render_prerequisite"]["binding"] = binding
    path = tmp_path / "exec-gate.json.gz"
    path.write_bytes(seal_manifest_bytes(manifest))
    return path


def test_dispatcher_accepts_bound_render_binding(tmp_path, monkeypatch):
    import dispatch_joint_quanta as dispatch
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"container": {"image": "sha256:" + "0" * 64},
                                "env": {}}))
    monkeypatch.setattr(dispatch, "SPEC_PATH", spec)
    record, receipt, parent = _layer2(tmp_path)
    manifest = build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite=dict(RENDER_PREREQ))
    bound_path = _gate_manifest(
        tmp_path, manifest, {"scope": "pb732", "material": "f" * 64})
    wire_sha = hashlib.sha256(bound_path.read_bytes()).hexdigest()
    adjoint_path = tmp_path / "adjoint.json"
    adjoint_path.write_text("{}")
    record_path = tmp_path / "record.json"
    record_path.write_text("{}")
    record = {"quantum_id": "layer-003", "layer": 3,
              "campaign": {"plan_path": "plan.json", "plan_sha256": "0" * 64,
                           "prepared_path": "prep.json",
                           "prepared_sha256": "1" * 64},
              "read_set": {"manifest_path": str(tmp_path / "slice.gz"),
                           "manifest_sha256": hashlib.sha256(
                               b"slice").hexdigest()},
              "chunks": [{"name": "layer-003-c000"}],
              "adjoint": {},
              "executable_readset": {
                  "manifest_path": str(bound_path),
                  "manifest_sha256": wire_sha,
                  "phases": [p["name"]
                             for p in manifest["read_plan"]["phases"]]}}
    (tmp_path / "slice.gz").write_bytes(b"slice")
    argv = dispatch.quantum_argv(
        record, record_path=record_path, output_root=tmp_path,
        adjoint_path=adjoint_path)
    assert argv[argv.index("--data-manifest-sha256") + 1] == wire_sha


def test_dispatcher_refuses_unbound_render_prerequisite(tmp_path,
                                                        monkeypatch):
    import dispatch_joint_quanta as dispatch
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"container": {"image": "sha256:" + "0" * 64},
                                "env": {}}))
    monkeypatch.setattr(dispatch, "SPEC_PATH", spec)
    record, receipt, parent = _layer2(tmp_path)
    manifest = build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite=dict(RENDER_PREREQ))
    assert manifest["annotations"]["render_prerequisite"]["binding"] is None
    bound_path = tmp_path / "exec-gate.json.gz"
    bound_path.write_bytes(seal_manifest_bytes(manifest))
    wire_sha = hashlib.sha256(bound_path.read_bytes()).hexdigest()
    adjoint_path = tmp_path / "adjoint.json"
    adjoint_path.write_text("{}")
    record_path = tmp_path / "record.json"
    record_path.write_text("{}")
    record = {"quantum_id": "layer-003", "layer": 3,
              "campaign": {"plan_path": "plan.json", "plan_sha256": "0" * 64,
                           "prepared_path": "prep.json",
                           "prepared_sha256": "1" * 64},
              "read_set": {"manifest_path": str(tmp_path / "slice.gz"),
                           "manifest_sha256": hashlib.sha256(
                               b"slice").hexdigest()},
              "chunks": [{"name": "layer-003-c000"}],
              "adjoint": {},
              "executable_readset": {
                  "manifest_path": str(bound_path),
                  "manifest_sha256": wire_sha,
                  "phases": [p["name"]
                             for p in manifest["read_plan"]["phases"]]}}
    (tmp_path / "slice.gz").write_bytes(b"slice")
    with pytest.raises(dispatch.DispatchRefused, match="not runnable"):
        dispatch.quantum_argv(
            record, record_path=record_path, output_root=tmp_path,
            adjoint_path=adjoint_path)
