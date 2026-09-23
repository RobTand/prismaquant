"""Producer tests: quantum boundary/checkpoint bulk readset binding.

PQ #848, root reviews R2/R3 (post-capture bound metadata generation --
runtime staging/phase consumption stays a concrete follow-up). Fixtures
are built by the REAL producer at tiny CPU scale
(`write_adjoint_checkpoint` + the exact activation owner +
`exact_entry_record`, real `layer_quanta` records, the real regen CLI) --
no handbuilt receipt shapes, no payloads. Proves the manifest freezes the
reader's exact repeated schedule, seals byte-exact accounting, binds new
record generations with recomputed identity through the existing binder,
and refuses malformed input. Full-chain staging tests belong to the
integration worker, not this file.
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

import regenerate_joint_quanta as regen  # noqa: E402
from prismaquant.joint_adjoint_checkpoints import exact_entry_record  # noqa: E402
from prismaquant.joint_adjoint_checkpoints import (  # noqa: E402
    write_adjoint_checkpoint, write_adjoint_receipt)
from prismaquant.joint_layer_quanta import (
    ADJOINT_CAPTURE_SCHEMA,
    LAYER_QUANTUM_SCHEMA,
    MANIFEST_SCHEMA_V1,
    MANIFEST_SCHEMA_V2,
    adjoint_binding_fields,
    bind_adjoint_slice,
    bind_quantum_boundary_readset,
    build_quantum_boundary_readset,
    emit_quantum_boundary_readsets,
    quantum_boundary_read_phase_names,
    seal_manifest_bytes,
)
from prismaquant.perturbed_x_cache import write_exact_activation_cache_entry

RECOVERY_RECORDS = (
    "/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913"
    "/allocation/joint-panel/stage-a-recovery-20260920/records")
STRIDED = [8]
N_PROBES = 2
N_BATCHES = 3
PREFETCH_BATCHES = 2
REPLAY_WINDOWS = [{"window_index": 0}, {"window_index": 1}]
SESSION = {"generation": "fixture-gen-848",
           "run_identity_sha256": "ab" * 32}

# Real record metadata lives on the fleet's shared mount; skip -- never
# fail -- where it is absent (the convention test_joint_layer_quanta.py
# established).
needs_recovery = pytest.mark.skipif(
    not os.path.isdir(RECOVERY_RECORDS),
    reason="the recovery records are absent (GitHub CI has no /mnt/shared)")


def _write_boundary(space, boundary, batch):
    tensor = torch.zeros(2, 4)
    ref = write_exact_activation_cache_entry(
        Path(space) / "entries",
        f"boundary-{batch}-{boundary}-at-{boundary}", tensor,
        identity={"session": dict(SESSION),
                  "slot": f"boundary-{batch}-{boundary}", "kind": "boundary",
                  "coordinates": {"batch": batch, "boundary": boundary,
                                  "probe": None}},
        max_tensor_bytes=1 << 20, max_file_bytes=1 << 20)
    return exact_entry_record(ref)


def _fixture(tmp_path, layer=3, chain=(7, 6, 5, 4), checkpoint=8,
             boundaries=(3, 4, 5, 6, 7)):
    """A completed capture at tiny scale, built only by real writers."""
    space = tmp_path / "adjoint"
    (space / "entries").mkdir(parents=True)
    boundary_entries = {}
    for boundary in boundaries:
        boundary_entries[str(boundary)] = [
            _write_boundary(space, boundary, batch)
            for batch in range(N_BATCHES)]
    cotangents = {(probe, batch): torch.zeros(2, 4) + probe
                  for probe in range(N_PROBES)
                  for batch in range(N_BATCHES)}
    shared_adjoint = {(probe, batch): {"scale": 1.0}
                      for probe in range(N_PROBES)
                      for batch in range(N_BATCHES)}
    shared_pass = {batch: {"mask": [0, 1]} for batch in range(N_BATCHES)}
    checkpoint_record = write_adjoint_checkpoint(
        space, boundary=checkpoint,
        session={"generation": SESSION["generation"],
                 "kind": "adjoint_checkpoint",
                 "run_identity_sha256": SESSION["run_identity_sha256"]},
        cotangents=cotangents, shared_adjoint=shared_adjoint,
        shared_pass=shared_pass)
    campaign = {"plan_path": "/mnt/shared/run.plan.json",
                "plan_sha256": "0" * 64, "prepared_path": "/mnt/shared/prepared.json",
                "prepared_sha256": "1" * 64, "read_manifest_sha256": "2" * 64,
                "unit_roster_sha256": "3" * 64,
                "campaign_scope": {"fixture": "readset-r2"}}
    receipt = {
        "schema": ADJOINT_CAPTURE_SCHEMA,
        "run_identity": {"plan_sha256": campaign["plan_sha256"],
                         "prepared_sha256": campaign["prepared_sha256"],
                         "campaign_scope": campaign["campaign_scope"]},
        "stride": _stride(STRIDED),
        "boundary_storage": {
            "session": dict(SESSION),
            "policy": {"prefetch_batches": PREFETCH_BATCHES},
            "directory": str(space / "entries")},
        "boundary_entries": boundary_entries,
        "checkpoints": [checkpoint_record],
        "status": "complete",
    }
    record = {"schema": LAYER_QUANTUM_SCHEMA,
              "quantum_id": f"layer-{layer:03d}", "layer": layer,
              "adjoint": {"checkpoint_boundary": checkpoint,
                          "chain_layers": list(chain),
                          "receipt_sha256": None},
              "campaign": campaign, "windows": list(REPLAY_WINDOWS)}
    return record, receipt


def _stride(boundaries):
    """The stride block Stage A seals: the checkpoint marks tail first."""
    marks = sorted(boundaries, reverse=True)
    value = marks[-1] if len(marks) > 1 or marks[0] > 1 else 1
    return {"value": value, "source": None, "boundaries": marks,
            "max_chain_layers": value - 1}


def _build(record, receipt):
    return build_quantum_boundary_readset(
        record, receipt, strided_boundaries=STRIDED, n_probes=N_PROBES)


def _bind_receipt(record, receipt):
    """Bind the record's stage-A slice the way the regen path does (PQ
    #993), then reseal."""
    adjoint_slice, digest = bind_adjoint_slice(
        receipt, record["layer"], plan_sha256=record["campaign"]["plan_sha256"],
        prepared_sha256=record["campaign"]["prepared_sha256"],
        scope=record["campaign"]["campaign_scope"],
        checkpoints=receipt["stride"]["boundaries"])
    adjoint = {key: value for key, value in record["adjoint"].items()
               if key not in ("receipt_sha256", "slice_sha256", "slice_path")}
    adjoint.update(adjoint_binding_fields(
        adjoint_slice, slice_path="/mnt/shared/run/layer-quanta/adjoint-slices/"
                                  f"{record['quantum_id']}.json"))
    record = dict(record, adjoint=adjoint)
    return _reseal_like_regen(record), digest


def test_phase_names_freeze_repeated_reader_schedule():
    names = quantum_boundary_read_phase_names(
        [7, 6, 5, 4], 3, batch_windows=2, n_probes=2, replay_windows=2)
    assert names[0] == "checkpoint"
    chain = [n for n in names[1:] if n.startswith("chain-")]
    replay = [n for n in names[1:] if n.startswith("replay-")]
    # Four chain layers x two probes x two windows; two replay windows x
    # two probes x two windows: every repeat the reader performs is named.
    assert len(chain) == 4 * 2 * 2
    assert len(replay) == 2 * 2 * 2
    assert chain[0] == "chain-007-p0-w00"
    assert chain[1] == "chain-007-p0-w01"
    assert chain[2] == "chain-007-p1-w00"
    assert replay[0] == "replay-00-p0-w00"
    assert replay[-1] == "replay-01-p1-w01"


def test_tail_quantum_chains_nothing_but_replays():
    names = quantum_boundary_read_phase_names(
        [], 5, batch_windows=1, n_probes=2, replay_windows=1)
    assert names == ("checkpoint", "replay-00-p0-w00", "replay-00-p1-w00")


def test_declared_repeat_arithmetic(tmp_path):
    """The frozen phase list repeats exactly what the reader schedule
    declares -- checked as metadata arithmetic here, not runtime cursor
    enforcement. Counts trace to their owners: per-probe chain passes to
    ``render_free_layer_roll``'s probe loop
    (joint_adjoint_checkpoints.py:391), per-(window, probe) replay passes
    to ``replay_backward`` via ``observe_and_project_retained_windows``
    (joint_statistics_replay.py:410-452), batch windows to the sealed
    prefetch_batches."""
    record, receipt = _fixture(tmp_path)
    manifest = _build(record, receipt)
    phases = {p["name"]: p["entry_indices"]
              for p in manifest["read_plan"]["phases"]}
    # Checkpoint plane exactly once; chain boundaries once per probe;
    # own boundary once per (replay window, probe).
    runs = {}
    for boundary in (3, 4, 5, 6, 7):
        runs[boundary] = [
            i for i, e in enumerate(manifest["entries"])
            if f"-{boundary}-at-{boundary}.pt" in e["path"]]
    for boundary in (4, 5, 6, 7):
        hits = sum(1 for p in manifest["read_plan"]["phases"]
                   if p["name"].startswith("chain-")
                   for i in p["entry_indices"] if i in runs[boundary])
        assert hits == len(runs[boundary]) * N_PROBES
    own = sum(1 for p in manifest["read_plan"]["phases"]
              if p["name"].startswith("replay-")
              for i in p["entry_indices"] if i in runs[3])
    assert own == len(runs[3]) * len(REPLAY_WINDOWS) * N_PROBES
    first = phases["checkpoint"]
    # checkpoint.json, then the cotangent plane, then the shared states.
    assert len(first) == len(set(first)) == 1 + 2 * N_BATCHES + (
        N_PROBES * N_BATCHES + N_BATCHES)
    manifest_path = tmp_path / "adjoint" / "checkpoints" / "boundary-008" / "checkpoint.json"
    payload = manifest_path.read_bytes()
    assert manifest["entries"][first[0]] == {
        "path": str(manifest_path), "offset": 0, "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest()}
    assert manifest["read_plan"]["read_bytes"] == sum(
        p["bytes"] for p in manifest["read_plan"]["phases"])


def test_partial_last_batch_window(tmp_path):
    record, receipt = _fixture(tmp_path)
    manifest = _build(record, receipt)
    phases = {p["name"]: p for p in manifest["read_plan"]["phases"]}
    # Three batches in windows of two: the last window holds batch 2 alone.
    assert len(phases["chain-007-p0-w01"]["entry_indices"]) == 1
    assert len(phases["chain-007-p0-w00"]["entry_indices"]) == 2
    assert len(phases["replay-01-p1-w01"]["entry_indices"]) == 1


def test_resume_reads_a_subset(tmp_path):
    record, receipt = _fixture(tmp_path)
    manifest = _build(record, receipt)
    names = [p["name"] for p in manifest["read_plan"]["phases"]]
    # A resume that completed replay window 00 reports nothing under it;
    # every remaining phase is still a declared, nonempty staged read.
    resumed = [n for n in names if not n.startswith("replay-00-")]
    assert resumed[0] == "checkpoint"
    assert len(resumed) == len(names) - N_PROBES * 2
    by_name = {p["name"]: p for p in manifest["read_plan"]["phases"]}
    assert all(by_name[n]["bytes"] > 0 for n in resumed)


def test_pb_validation_admits_repeated_schedule(tmp_path):
    core, tiers = _pb()
    record, receipt = _fixture(tmp_path)
    manifest = _build(record, receipt)
    # PB admits only paths under the sealed mount prefix; production
    # entries always live there, so the test re-roots the fixture paths
    # (test-only scaffolding) and reseals before validating.
    root = str(tmp_path)
    for entry in manifest["entries"]:
        assert entry["path"].startswith(root)
        entry["path"] = "/mnt/shared/fixture" + entry["path"][len(root):]
    normalized = core.validate_data_manifest(manifest)
    ranges = tiers.manifest_phase_ranges(normalized)
    assert ranges, "published PB stages nothing for the bound readset"
    assert [r["name"] for r in ranges] == [
        p["name"] for p in manifest["read_plan"]["phases"]]
    assert ranges[-1]["end_bytes"] == manifest["read_plan"]["read_bytes"]


def _pb():
    from pathlib import Path as _Path
    fleet = _Path("/mnt/shared/prismabuild-fleet")
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
        if not _Path(module.__file__).resolve().is_relative_to(src.resolve()):
            pytest.skip("a different prismabuild is already imported")
    return core, tiers


def test_seal_deterministic_wire(tmp_path):
    record, receipt = _fixture(tmp_path)
    first = seal_manifest_bytes(_build(record, receipt))
    second = seal_manifest_bytes(_build(record, receipt))
    assert first == second
    # Wire identity (what a reader admits) is not the canonical slice
    # identity (what the binder seals): both present, never conflated.
    assert hashlib.sha256(first).hexdigest() != bind_adjoint_slice(
        receipt, record["layer"], plan_sha256=record["campaign"]["plan_sha256"],
        prepared_sha256=record["campaign"]["prepared_sha256"],
        scope=record["campaign"]["campaign_scope"], checkpoints=STRIDED)[1]


def _reseal_like_regen(record):
    """Recompute identity the way layer_quanta does when it binds a
    receipt -- fixtures that move the adjoint seal must reseal."""
    from prismaquant.joint_layer_quanta import canonical_sha256
    record = dict(record)
    body = {k: v for k, v in record.items() if k != "identity_sha256"}
    record["identity_sha256"] = canonical_sha256(
        body, where="fixture record")
    return record


@needs_recovery
def test_emit_binds_new_record_generations(tmp_path):
    """Actual layer_quanta records through the emission path: new
    generations bound, inputs byte-identical, digests reproducing."""
    from prismaquant.joint_layer_quanta import (
        check_quantum_for_campaign, canonical_sha256)
    base3 = json.load(open(f"{RECOVERY_RECORDS}/layer-003.json"))
    base4 = json.load(open(f"{RECOVERY_RECORDS}/layer-004.json"))
    assert base3["campaign"] == base4["campaign"]
    _, receipt = _real_record_receipt(tmp_path)
    assert receipt["run_identity"]["plan_sha256"] == base3["campaign"][
        "plan_sha256"]
    # The regen path reseals identity when it binds each record's slice;
    # the binder reverifies that seal before deriving anything further.
    rec3, digest3 = _bind_receipt(base3, receipt)
    rec4, digest4 = _bind_receipt(base4, receipt)
    assert digest3 != digest4, "each layer binds its own slice"
    before = [json.dumps(r, sort_keys=True) for r in (rec3, rec4)]
    root = str(tmp_path / "run")
    emitted = emit_quantum_boundary_readsets(
        receipt, [rec3, rec4],
        strided_boundaries=[8, 16, 24, 32, 40, 45], n_probes=N_PROBES,
        output_root=root)
    assert [json.dumps(r, sort_keys=True) for r in (rec3, rec4)] == before
    assert len(emitted) == 2
    paths = set()
    for row, source, digest in zip(emitted, (rec3, rec4), (digest3, digest4)):
        assert row["manifest_path"] == (
            f"{root}/layer-quanta/adjoint/bound-readsets/"
            f"{source['quantum_id']}.boundary-readset.json.gz")
        assert row["manifest_path"] not in paths
        paths.add(row["manifest_path"])
        assert row["manifest_sha256"] == hashlib.sha256(
            seal_manifest_bytes(row["manifest"])).hexdigest()
        bound = row["record"]["boundary_readset"]
        assert bound["manifest_path"] == row["manifest_path"]
        assert bound["manifest_sha256"] == row["manifest_sha256"]
        assert bound["slice_sha256"] == digest
        body = {k: v for k, v in row["record"].items()
                if k != "identity_sha256"}
        assert canonical_sha256(body, where="quantum record") == row[
            "record"]["identity_sha256"]
        campaign = dict(source["campaign"],
                        adjoint_slice_sha256=digest)
        check_quantum_for_campaign(row["record"], campaign)
    assert "boundary_readset" not in rec3
    assert "boundary_readset" not in rec4


def test_refuses_chain_outside_stride_owner(tmp_path):
    record, receipt = _fixture(tmp_path)
    record["adjoint"]["chain_layers"] = [7, 6, 5, 3]
    with pytest.raises(ValueError, match="sealed stride chain"):
        _build(record, receipt)


def test_refuses_missing_checkpoint_boundary(tmp_path):
    record, receipt = _fixture(tmp_path)
    receipt["checkpoints"] = []
    with pytest.raises(ValueError, match="0 checkpoint records for boundary 8"):
        _build(record, receipt)


def test_refuses_foreign_stride(tmp_path):
    record, receipt = _fixture(tmp_path)
    receipt["stride"] = _stride([8, 4])
    # Under the foreign stride the record's layer reads checkpoint 4, which
    # this run never sealed: the slice itself refuses before the header check.
    with pytest.raises(ValueError, match="0 checkpoint records for boundary 4"):
        _build(record, receipt)


def test_refuses_missing_needed_boundary(tmp_path):
    record, receipt = _fixture(tmp_path)
    del receipt["boundary_entries"]["5"]
    with pytest.raises(ValueError, match="no boundary 5 entries"):
        _build(record, receipt)


def test_refuses_malformed_exact_record(tmp_path):
    record, receipt = _fixture(tmp_path)
    receipt["boundary_entries"]["4"][0]["sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="no entry digest"):
        _build(record, receipt)


def test_refuses_duplicate_staged_path(tmp_path):
    record, receipt = _fixture(tmp_path)
    receipt["boundary_entries"]["4"][0]["path"] = \
        receipt["boundary_entries"]["6"][0]["path"]
    with pytest.raises(ValueError, match="duplicate staged path"):
        _build(record, receipt)


def test_refuses_mixed_batch_counts(tmp_path):
    record, receipt = _fixture(tmp_path)
    receipt["boundary_entries"]["4"].pop()
    with pytest.raises(ValueError, match="different batch counts"):
        _build(record, receipt)


def test_refuses_foreign_campaign(tmp_path):
    record, receipt = _fixture(tmp_path)
    receipt["run_identity"]["plan_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="another plan_sha256"):
        _build(record, receipt)


def test_refuses_missing_prefetch_window(tmp_path):
    record, receipt = _fixture(tmp_path)
    receipt["boundary_storage"]["policy"] = {}
    with pytest.raises(ValueError, match="no prefetch batch window"):
        _build(record, receipt)


def test_refuses_missing_replay_windows(tmp_path):
    record, receipt = _fixture(tmp_path)
    record["windows"] = []
    with pytest.raises(ValueError, match="no replay windows"):
        _build(record, receipt)


@needs_recovery
def test_real_campaign_metadata_phase_shape():
    record = json.load(open(f"{RECOVERY_RECORDS}/layer-003.json"))
    names = quantum_boundary_read_phase_names(
        record["adjoint"]["chain_layers"], record["layer"],
        batch_windows=512 // 64, n_probes=4,
        replay_windows=len(record["windows"]))
    # Four chain layers x four probes x eight windows, thirteen replay
    # windows x four probes x eight windows, plus the checkpoint plane.
    assert len(names) == 1 + 4 * 4 * 8 + 13 * 4 * 8
    assert names[0] == "checkpoint"
    assert names[1] == "chain-007-p0-w00"
    assert names[-1] == "replay-12-p3-w07"


def _real_record_receipt(tmp_path):
    """Actual layer-003 record with a campaign-bound fixture receipt."""
    from prismaquant.joint_layer_quanta import check_quantum_for_campaign
    record = json.load(open(f"{RECOVERY_RECORDS}/layer-003.json"))
    campaign = record["campaign"]
    space = tmp_path / "adjoint"
    (space / "entries").mkdir(parents=True)
    boundary_entries = {}
    for boundary in (3, 4, 5, 6, 7):
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
    cotangents = {(p, b): torch.zeros(2, 4) for p in range(N_PROBES)
                  for b in range(N_BATCHES)}
    full = write_adjoint_checkpoint(
        space, boundary=8,
        session={"generation": SESSION["generation"],
                 "kind": "adjoint_checkpoint",
                 "run_identity_sha256": SESSION["run_identity_sha256"]},
        cotangents=cotangents,
        shared_adjoint={(p, b): {"scale": 1.0} for p in range(N_PROBES)
                        for b in range(N_BATCHES)},
        shared_pass={b: {"mask": [0, 1]} for b in range(N_BATCHES)})
    checkpoints = [full] + [{"boundary": b} for b in (16, 24, 32, 40, 45)]
    receipt = {
        "schema": ADJOINT_CAPTURE_SCHEMA,
        "run_identity": {"plan_sha256": campaign["plan_sha256"],
                         "prepared_sha256": campaign["prepared_sha256"],
                         "campaign_scope": campaign["campaign_scope"]},
        "stride": _stride([8, 16, 24, 32, 40, 45]),
        "boundary_storage": {
            "session": dict(SESSION),
            "policy": {"prefetch_batches": PREFETCH_BATCHES},
            "directory": str(space / "entries")},
        "boundary_entries": boundary_entries,
        "checkpoints": checkpoints,
        "status": "complete",
    }
    return record, receipt


@needs_recovery
def test_bound_record_passes_real_campaign_validator(tmp_path):
    """RED for R3: the bound generation must recompute identity through
    the existing canonical owner, or both real validators refuse it."""
    from prismaquant.joint_layer_quanta import (
        check_quantum_for_campaign, canonical_sha256)
    record, receipt = _real_record_receipt(tmp_path)
    manifest = build_quantum_boundary_readset(
        record, receipt, strided_boundaries=[8, 16, 24, 32, 40, 45],
        n_probes=4)
    wire_sha256 = hashlib.sha256(seal_manifest_bytes(manifest)).hexdigest()
    record, digest = _bind_receipt(record, receipt)
    assert digest == manifest["annotations"]["slice_sha256"]
    run_root = str(Path(
        record["output_space"]["root"]).resolve().parents[1])
    new = bind_quantum_boundary_readset(
        record, receipt, manifest=manifest,
        manifest_path=f"{run_root}/layer-quanta/adjoint/bound-readsets/"
                      "layer-003.boundary-readset.json.gz",
        manifest_sha256=wire_sha256, output_root=run_root,
        strided_boundaries=[8, 16, 24, 32, 40, 45], n_probes=4)
    campaign = dict(record["campaign"],
                    unit_roster_sha256=record["campaign"].get(
                        "unit_roster_sha256"),
                    adjoint_slice_sha256=manifest["annotations"][
                        "slice_sha256"])
    check_quantum_for_campaign(new, campaign)
    body = {k: v for k, v in new.items() if k != "identity_sha256"}
    assert canonical_sha256(body, where="quantum record") == new[
        "identity_sha256"]


@needs_recovery
def test_binder_rejects_foreign_receipt_for_same_layer(tmp_path):
    record, receipt = _real_record_receipt(tmp_path)
    manifest = build_quantum_boundary_readset(
        record, receipt, strided_boundaries=[8, 16, 24, 32, 40, 45],
        n_probes=4)
    wire_sha256 = hashlib.sha256(seal_manifest_bytes(manifest)).hexdigest()
    record, _ = _bind_receipt(record, receipt)
    bound = dict(record["adjoint"])
    bound["slice_sha256"] = "f" * 64
    record = _reseal_like_regen(dict(record, adjoint=bound))
    run_root = str(Path(
        record["output_space"]["root"]).resolve().parents[1])
    with pytest.raises(ValueError, match="another stage-A slice"):
        bind_quantum_boundary_readset(
            record, receipt, manifest=manifest,
            manifest_path=f"{run_root}/layer-quanta/adjoint/bound-readsets/"
                          "layer-003.boundary-readset.json.gz",
            manifest_sha256=wire_sha256, output_root=run_root,
            strided_boundaries=[8, 16, 24, 32, 40, 45], n_probes=4)


def _tiny_campaign(tmp_path):
    """Two-layer plan + prepared + parent files the real producer binds."""
    root = tmp_path / "campaign"
    plan = {"output_root": str(root), "model": "/fixture/model",
            "distributed_campaign": {},
            "execution": {"n_probes": 2}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    prepared = {"formats_by_qname": {
        "model.layers.0.mlp.gate_proj": {},
        "model.layers.1.mlp.gate_proj": {}}}
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(json.dumps(prepared, sort_keys=True))
    prepared_sha = hashlib.sha256(prepared_path.read_bytes()).hexdigest()
    entries = [
        {"path": "/fixture/model/shard-h0.pt", "offset": 0, "bytes": 100,
         "sha256": None},
        {"path": "/fixture/model/shard-h1.pt", "offset": 0, "bytes": 100,
         "sha256": None},
        {"path": "/fixture/model/shard-l0.pt", "offset": 0, "bytes": 200,
         "sha256": None},
        {"path": "/fixture/model/shard-l1.pt", "offset": 0, "bytes": 300,
         "sha256": None},
    ]
    parent = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "produced_by": {"plan": str(plan_path)},
        "mount_prefix": "/mnt/shared",
        "entries": entries, "entry_count": 4, "total_bytes": 700,
        "annotations": {
            "campaign_scope": {"campaign": "readset-r3-fixture"},
            "argv": ["python3", "-m", "prismaquant.joint_adjoint_capture",
                     "--prepared", str(prepared_path),
                     "--prepared-sha256", prepared_sha],
            "layers": [0, 1],
            "phases": [
                {"name": "head", "bytes": 200, "cumulative_bytes": 200},
                {"name": "layer-0", "bytes": 200, "cumulative_bytes": 400},
                {"name": "layer-1", "bytes": 300, "cumulative_bytes": 700},
            ],
        },
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent, sort_keys=True))

    def _sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return {"root": root, "plan_path": plan_path, "plan_sha": _sha(plan_path),
            "prepared_path": prepared_path, "prepared_sha": _sha(prepared_path),
            "parent_path": parent_path, "parent_sha": _sha(parent_path)}


def _tiny_receipt(campaign, space):
    """A completed capture at tiny scale, sealed by the real writer."""
    from prismaquant.perturbed_x_cache import (
        write_exact_activation_cache_entry)
    session = {"generation": "fixture-gen-r3",
               "run_identity_sha256": "cd" * 32}
    entries_dir = space / "entries"
    entries_dir.mkdir(parents=True, exist_ok=True)
    boundary_entries = {}
    for boundary in (0, 1):
        rows = []
        for batch in range(3):
            ref = write_exact_activation_cache_entry(
                entries_dir,
                f"boundary-{batch}-{boundary}-at-{boundary}",
                torch.zeros(2, 4),
                identity={"session": dict(session),
                          "slot": f"boundary-{batch}-{boundary}",
                          "kind": "boundary",
                          "coordinates": {"batch": batch,
                                          "boundary": boundary,
                                          "probe": None}},
                max_tensor_bytes=1 << 20, max_file_bytes=1 << 20)
            rows.append(exact_entry_record(ref))
        boundary_entries[str(boundary)] = rows
    checkpoints = []
    for boundary in (1, 2):
        checkpoints.append(write_adjoint_checkpoint(
            space, boundary=boundary, session=dict(
                session, kind="adjoint_checkpoint"),
            cotangents={(p, b): torch.zeros(2, 4) for p in range(2)
                        for b in range(3)},
            shared_adjoint={(p, b): {"scale": 1.0} for p in range(2)
                            for b in range(3)},
            shared_pass={b: {"mask": [0, 1]} for b in range(3)}))
    receipt = {
        "schema": ADJOINT_CAPTURE_SCHEMA,
        "run_identity": {"plan_sha256": campaign["plan_sha"],
                         "prepared_sha256": campaign["prepared_sha"],
                         "campaign_scope": {"campaign": "readset-r3-fixture"}},
        "stride": _stride([1, 2]),
        "boundary_storage": {
            "session": dict(session),
            "policy": {"prefetch_batches": 2},
            "directory": str(entries_dir)},
        "boundary_entries": boundary_entries,
        "checkpoints": checkpoints,
        "status": "complete",
    }
    write_adjoint_receipt(space, receipt)
    return receipt


def _regen_argv(tmp_path, campaign):
    derivation = tmp_path / "derivation.json"
    derivation.write_text(json.dumps(
        {"chunk_target_bytes": 400, "stride": 1, "ram_window_gib": 160,
         "max_resident_consumers": 2}, sort_keys=True))
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps(
        {"windows_by_layer": {"0": 2, "1": 1}}, sort_keys=True))
    return ["--plan", str(campaign["plan_path"]),
            "--plan-sha256", campaign["plan_sha"],
            "--prepared", str(campaign["prepared_path"]),
            "--prepared-sha256", campaign["prepared_sha"],
            "--parent-manifest", str(campaign["parent_path"]),
            "--parent-manifest-sha256", campaign["parent_sha"],
            "--derivation", str(derivation),
            "--partition", str(partition)]


@pytest.mark.parametrize("changed_field", ["windows", "plan_path"])
def test_binder_refuses_stale_record_even_with_matching_manifest(tmp_path, changed_field):
    """An internally consistent new manifest cannot authorize an edited record."""
    record, receipt = _fixture(tmp_path)
    record, _ = _bind_receipt(record, receipt)
    if changed_field == "windows":
        record["windows"] = record["windows"] + [{"window_index": 2}]
    else:
        record["campaign"] = dict(record["campaign"], plan_path="/mnt/shared/other-plan.json")
    manifest = _build(record, receipt)
    with pytest.raises(ValueError, match="identity does not recompute"):
        bind_quantum_boundary_readset(
            record, receipt, manifest=manifest,
            manifest_path="/mnt/shared/run/layer-quanta/adjoint/"
                          "bound-readsets/layer-003.boundary-readset.json.gz",
            manifest_sha256=hashlib.sha256(seal_manifest_bytes(manifest)).hexdigest(),
            output_root="/mnt/shared/run", strided_boundaries=STRIDED,
            n_probes=N_PROBES)


def test_binder_refuses_consistent_manifest_for_another_probe_count(tmp_path):
    """The caller's sealed count must dominate a self-consistent manifest."""
    record, receipt = _fixture(tmp_path)
    record, _ = _bind_receipt(record, receipt)
    manifest = build_quantum_boundary_readset(
        record, receipt, strided_boundaries=STRIDED, n_probes=N_PROBES + 1)
    with pytest.raises(ValueError, match="another probe count"):
        bind_quantum_boundary_readset(
            record, receipt, manifest=manifest,
            manifest_path="/mnt/shared/run/layer-quanta/adjoint/"
                          "bound-readsets/layer-003.boundary-readset.json.gz",
            manifest_sha256=hashlib.sha256(seal_manifest_bytes(manifest)).hexdigest(),
            output_root="/mnt/shared/run", strided_boundaries=STRIDED,
            n_probes=N_PROBES)


def test_cli_binds_readsets_through_real_validators(tmp_path):
    """The complete bounded unit: real CLI output passes the real
    record/wire/receipt validation, including the consumer's own
    verify_quantum_identity on files."""
    from prismaquant import joint_cost_quantum as quantum
    from prismaquant.joint_layer_quanta import check_quantum_for_campaign
    campaign = _tiny_campaign(tmp_path)
    space = tmp_path / "adjoint-space"
    space.mkdir()
    receipt = _tiny_receipt(campaign, space)
    receipt_path = space / "adjoint-capture.json"
    out = tmp_path / "reviewed"
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--output-root", str(campaign["root"]),
           "--records-out", str(out),
           "--adjoint-receipt", str(receipt_path),
           "--boundary-readsets"]) == 0
    records = [json.loads(path.read_text())
               for path in sorted(out.glob("layer-*.json"))]
    assert len(records) == 2
    assert (out / "records.json").is_file()
    for record in records:
        # Each record binds its own layer's slice (PQ #993), whose file is
        # the slice's canonical bytes.
        canonical = bind_adjoint_slice(
            receipt, record["layer"], plan_sha256=campaign["plan_sha"],
            prepared_sha256=campaign["prepared_sha"],
            scope={"campaign": "readset-r3-fixture"}, checkpoints=[1, 2])[1]
        assert record["adjoint"]["slice_sha256"] == canonical
        assert "receipt_sha256" not in record["adjoint"]
        slice_path = Path(record["adjoint"]["slice_path"])
        assert hashlib.sha256(slice_path.read_bytes()).hexdigest() == canonical
        bound = record["boundary_readset"]
        manifest_path = Path(bound["manifest_path"])
        assert manifest_path.is_file()
        assert hashlib.sha256(
            manifest_path.read_bytes()).hexdigest() == bound[
            "manifest_sha256"]
        assert bound["slice_sha256"] == canonical
        assert manifest_path.parent.name == "bound-readsets"
        campaign_binding = {
            "plan_sha256": campaign["plan_sha"],
            "prepared_sha256": campaign["prepared_sha"],
            "read_manifest_sha256": campaign["parent_sha"],
            "unit_roster_sha256": record["campaign"][
                "unit_roster_sha256"],
            "campaign_scope": {"campaign": "readset-r3-fixture"},
            "adjoint_slice_sha256": canonical,
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
            adjoint_path=slice_path, adjoint_sha256=canonical,
            output_root=Path(str(campaign["root"])))


def test_cli_refusal_writes_no_partial_generation(tmp_path):
    campaign = _tiny_campaign(tmp_path)
    space = tmp_path / "adjoint-space"
    space.mkdir()
    receipt = _tiny_receipt(campaign, space)
    del receipt["boundary_entries"]["1"]
    bad = space / "bad-receipt.json"
    bad.write_text(json.dumps(receipt, sort_keys=True))
    out = tmp_path / "reviewed"
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--output-root", str(campaign["root"]),
           "--records-out", str(out),
           "--adjoint-receipt", str(bad),
           "--boundary-readsets"]) == 3
    assert not out.exists()
    assert not (Path(str(campaign["root"])) / "layer-quanta" / "adjoint" /
                "bound-readsets").exists()


def test_cli_readsets_need_a_receipt(tmp_path):
    campaign = _tiny_campaign(tmp_path)
    out = tmp_path / "reviewed"
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--output-root", str(campaign["root"]),
           "--records-out", str(out),
           "--boundary-readsets"]) == 3
    assert not out.exists()


@pytest.mark.parametrize("status", ["running", "failed"])
def test_builder_refuses_incomplete_status(tmp_path, status):
    record, receipt = _fixture(tmp_path)
    receipt["status"] = status
    with pytest.raises(ValueError, match="completed capture"):
        _build(record, receipt)


def test_builder_refuses_missing_status(tmp_path):
    record, receipt = _fixture(tmp_path)
    del receipt["status"]
    with pytest.raises(ValueError, match="completed capture"):
        _build(record, receipt)


def test_binder_refuses_mutated_entry_consistent_rehash(tmp_path):
    """A different path/bytes/digest set with matching annotations,
    recomputed counts and a fresh wire hash must still refuse: the triples
    must originate from the bound receipt, not merely agree with it."""
    import copy
    record, receipt = _fixture(tmp_path)
    manifest = _build(record, receipt)
    forged = copy.deepcopy(manifest)
    forged["entries"][0]["bytes"] += 8
    forged["entries"][0]["sha256"] = "c" * 64
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
    record, _ = _bind_receipt(record, receipt)
    with pytest.raises(ValueError, match="originate from the bound slice"):
        bind_quantum_boundary_readset(
            record, receipt, manifest=forged,
            manifest_path="/mnt/shared/run/layer-quanta/adjoint/"
                          "bound-readsets/layer-003.boundary-readset.json.gz",
            manifest_sha256=forged_sha, output_root="/mnt/shared/run",
            strided_boundaries=STRIDED, n_probes=N_PROBES)


def test_binder_refuses_foreign_schema(tmp_path):
    record, receipt = _fixture(tmp_path)
    manifest = _build(record, receipt)
    manifest["schema"] = MANIFEST_SCHEMA_V1
    wire_sha = hashlib.sha256(seal_manifest_bytes(manifest)).hexdigest()
    record, _ = _bind_receipt(record, receipt)
    with pytest.raises(ValueError, match="foreign schema"):
        bind_quantum_boundary_readset(
            record, receipt, manifest=manifest,
            manifest_path="/mnt/shared/run/layer-quanta/adjoint/"
                          "bound-readsets/layer-003.boundary-readset.json.gz",
            manifest_sha256=wire_sha, output_root="/mnt/shared/run",
            strided_boundaries=STRIDED, n_probes=N_PROBES)


@pytest.mark.parametrize("status", ["running", "failed"])
def test_cli_refuses_incomplete_status(tmp_path, status):
    campaign = _tiny_campaign(tmp_path)
    space = tmp_path / "adjoint-space"
    space.mkdir()
    receipt = _tiny_receipt(campaign, space)
    receipt["status"] = status
    partial = space / "partial-receipt.json"
    partial.write_text(json.dumps(receipt, sort_keys=True))
    out = tmp_path / "reviewed"
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--output-root", str(campaign["root"]),
           "--records-out", str(out),
           "--adjoint-receipt", str(partial),
           "--boundary-readsets"]) == 3
    assert not out.exists()
    assert not (Path(str(campaign["root"])) / "layer-quanta" / "adjoint" /
                "bound-readsets").exists()


def test_cli_rerun_is_same_byte_idempotent(tmp_path):
    campaign = _tiny_campaign(tmp_path)
    space = tmp_path / "adjoint-space"
    space.mkdir()
    receipt = _tiny_receipt(campaign, space)
    receipt_path = space / "adjoint-capture.json"
    out = tmp_path / "reviewed"
    argv = (_regen_argv(tmp_path, campaign)
            + ["--output-root", str(campaign["root"]),
               "--records-out", str(out),
               "--adjoint-receipt", str(receipt_path),
               "--boundary-readsets"])
    assert regen.main(argv) == 0
    first = {p.relative_to(out): p.read_bytes() for p in sorted(
        list(out.rglob("layer-*.json")) + list(out.glob("*.json")))}
    first_manifests = {}
    for path in sorted((Path(str(campaign["root"])) / "layer-quanta" /
                        "adjoint" / "bound-readsets").glob("*.gz")):
        first_manifests[path.name] = path.read_bytes()
    assert regen.main(argv) == 0
    second = {p.relative_to(out): p.read_bytes() for p in sorted(
        list(out.rglob("layer-*.json")) + list(out.glob("*.json")))}
    assert second == first
    for path in sorted((Path(str(campaign["root"])) / "layer-quanta" /
                        "adjoint" / "bound-readsets").glob("*.gz")):
        assert path.read_bytes() == first_manifests[path.name]
