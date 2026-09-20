"""Producer tests: executable combined quantum read plan (PQ #862).

Self-contained tiny fixtures: a four-layer parent, real `layer_quanta`
records, real writer-built receipts. Proves the ONE executable v2
manifest covers checkpoint, chain/own source extents, boundary/probe/
replay reads in true consumption order with byte-exact accounting;
binding through the existing owners; dispatcher selection with derived
progress declarations; runtime read-phase reporting through the existing
semantic reporter. Full GPU execution belongs to a later lane, not here.
"""

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
    assert any("cotangent-0-0-at-4" in e["path"]
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
    core, tiers = _pb()
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
    for module in (core, tiers):
        if not Path(module.__file__).resolve().is_relative_to(src.resolve()):
            pytest.skip("a different prismabuild is already imported")
    return core, tiers


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
    manifest_path = tmp_path / "exec.json.gz"
    manifest_path.write_bytes(seal_manifest_bytes(manifest))
    record["executable_readset"] = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": wire_sha,
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
    assert argv[argv.index("--data-manifest-sha256") + 1] == wire_sha
    declared = [argv[i + 1] for i, word in enumerate(argv[:-1])
                if word == "--progress-phase"]
    assert declared[0].startswith("head=")
    assert declared[1].startswith("checkpoint-load=")
    assert [name.split("=")[0] for name in declared] == (
        ["head"] + [p["name"]
                    for p in manifest["read_plan"]["phases"]])
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
    progress.enter_read_phase("checkpoint-load")
    assert progress._phase == "checkpoint-load"
    assert progress.units() == units
    assert progress.commits == 2
    progress.enter_read_phase("no-such-phase")
    assert progress._phase == "checkpoint-load"
    assert progress.commits == 2


def test_replay_phase_zero_pending_convention():
    from prismaquant.joint_cost_quantum import (
        executable_replay_phase_name as _replay)
    assert _replay(None, 1) == "replay-00-p1"
    assert _replay(2, 0) == "replay-02-p0"


def test_acceptance_sequence_opens_only_admitted_entries(tmp_path):
    """Scripted replay of the reader call sequence on fixture bytes.

    Every payload open lands on a manifest entry with a matching digest,
    in manifest phase order; PB parser checks plus an
    accepted/remaining walk over the reported phases prove the order the
    tier loop would stage ahead of.
    """
    from prismaquant.joint_adjoint_checkpoints import (
        load_adjoint_checkpoint, reference_from_record)
    from prismaquant.perturbed_x_cache import (
        prefetch_exact_activation_cache_entries)
    core, tiers = _pb()
    import prismabuild.residency_plan as plans
    record, receipt, parent = _layer2(tmp_path)
    manifest = build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite=dict(RENDER_PREREQ))
    by_path = {e["path"]: e for e in manifest["entries"]}
    opened = []
    reported = []

    def _open(path, sha256):
        assert path in by_path, path
        assert by_path[path]["sha256"] == sha256
        opened.append(path)

    checkpoint = next(
        entry for entry in receipt["checkpoints"]
        if entry["boundary"] == record["adjoint"]["checkpoint_boundary"])
    space = tmp_path / "adjoint"
    cotangents, _, _ = load_adjoint_checkpoint(space, checkpoint)
    assert cotangents, "checkpoint plane loads no tensors"
    for entry in (checkpoint["activation_entries"]
                  + checkpoint["shared_state_entries"]):
        _open(entry["path"], entry["sha256"])
    reported.append("checkpoint-load")
    # Chain layer 3, probe 0: the real owner-level prefetch window over
    # references rebuilt from the receipt records (verified reads).
    refs = [reference_from_record(entry)
            for entry in receipt["boundary_entries"]["3"]]
    with prefetch_exact_activation_cache_entries(
            refs, expected_session=receipt["boundary_storage"]["session"],
            max_tensor_bytes=1 << 30) as window:
        for ref in refs:
            tensor = window.get(ref)
            assert tensor.numel() > 0
            _open(ref.path, ref.sha256)
    reported.append("chain-003-bound")
    # Replay window 0, probe 0 over the own boundary: same verified seam.
    own = [reference_from_record(entry)
           for entry in receipt["boundary_entries"]["2"]]
    with prefetch_exact_activation_cache_entries(
            own, expected_session=receipt["boundary_storage"]["session"],
            max_tensor_bytes=1 << 30) as window:
        for ref in own:
            window.get(ref)
            _open(ref.path, ref.sha256)
    reported.append("replay-00-p0")
    # Every open landed on an admitted entry; reports follow manifest order.
    assert opened
    phase_of = {}
    for phase in manifest["read_plan"]["phases"]:
        for index in phase["entry_indices"]:
            phase_of.setdefault(manifest["entries"][index]["path"], []).append(
                phase["name"])
    assert all("checkpoint-load" in phase_of[path]
               for path in opened[:len(checkpoint["activation_entries"])
                                   + len(checkpoint["shared_state_entries"])])
    normalized = core.validate_data_manifest(manifest)
    ranges = tiers.manifest_phase_ranges(normalized)
    assert [r["name"] for r in ranges] == [
        p["name"] for p in manifest["read_plan"]["phases"]]
    names = [p["name"] for p in manifest["read_plan"]["phases"]]
    plan_view = {"phases": [{"name": name} for name in names]}
    for position, name in enumerate(names):
        assert plans.accepted(plan_view, name)
        remaining = [p["name"] for p in plans.remaining(plan_view, name)]
        assert remaining == names[position:]


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
    for boundary in (2, 3):
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
