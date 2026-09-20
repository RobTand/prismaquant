"""Producer tests: quantum boundary/checkpoint bulk readset binding.

PQ #848, root review R2. Fixtures are built by the REAL producer at tiny
CPU scale (`write_adjoint_checkpoint` + the exact activation owner +
`exact_entry_record`) -- no handbuilt receipt shapes, no payloads. Proves
the manifest freezes the reader's exact repeated schedule (checkpoint
once, chain boundaries once per probe, own boundary once per
replay-window per probe), seals byte-exact accounting, binds new record
generations through the existing binder, and refuses malformed input.
Full-chain staging tests belong to the integration worker, not this file.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch

from prismaquant.joint_adjoint_checkpoints import exact_entry_record
from prismaquant.joint_adjoint_checkpoints import write_adjoint_checkpoint
from prismaquant.joint_layer_quanta import (
    ADJOINT_CAPTURE_SCHEMA,
    MANIFEST_SCHEMA_V2,
    bind_adjoint_receipt,
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
                "campaign_scope": {"fixture": "readset-r2"}}
    receipt = {
        "schema": ADJOINT_CAPTURE_SCHEMA,
        "run_identity": {"plan_sha256": campaign["plan_sha256"],
                         "prepared_sha256": campaign["prepared_sha256"],
                         "campaign_scope": campaign["campaign_scope"]},
        "boundary_storage": {
            "session": dict(SESSION),
            "policy": {"prefetch_batches": PREFETCH_BATCHES}},
        "boundary_entries": boundary_entries,
        "checkpoints": [checkpoint_record],
    }
    record = {"quantum_id": f"layer-{layer:03d}", "layer": layer,
              "adjoint": {"checkpoint_boundary": checkpoint,
                          "chain_layers": list(chain),
                          "receipt_sha256": None},
              "campaign": campaign, "windows": list(REPLAY_WINDOWS)}
    return record, receipt


def _build(record, receipt):
    return build_quantum_boundary_readset(
        record, receipt, strided_boundaries=STRIDED, n_probes=N_PROBES)


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


def test_repeat_counts_match_reader_iteration(tmp_path):
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
    assert len(first) == len(set(first)) == 2 * N_BATCHES + (
        N_PROBES * N_BATCHES + N_BATCHES)
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
    # Wire identity (what a reader admits) is not the canonical receipt
    # identity (what the binder seals): both present, never conflated.
    assert hashlib.sha256(first).hexdigest() != bind_adjoint_receipt(
        receipt, plan_sha256=record["campaign"]["plan_sha256"],
        prepared_sha256=record["campaign"]["prepared_sha256"],
        scope=record["campaign"]["campaign_scope"], checkpoints=STRIDED)


def test_emit_binds_new_record_generations(tmp_path):
    record, receipt = _fixture(tmp_path)
    before = json.dumps(record, sort_keys=True)
    emitted = emit_quantum_boundary_readsets(
        receipt, [record], strided_boundaries=STRIDED, n_probes=N_PROBES,
        output_root="/mnt/shared/run")
    assert json.dumps(record, sort_keys=True) == before
    assert len(emitted) == 1
    row = emitted[0]
    assert row["manifest_path"] == (
        "/mnt/shared/run/layer-quanta/adjoint/bound-readsets/"
        "layer-003.boundary-readset.json.gz")
    assert row["manifest_sha256"] == hashlib.sha256(
        seal_manifest_bytes(row["manifest"])).hexdigest()
    bound = row["record"]["boundary_readset"]
    assert bound["manifest_path"] == row["manifest_path"]
    assert bound["manifest_sha256"] == row["manifest_sha256"]
    assert bound["receipt_sha256"] == row["manifest"]["annotations"][
        "receipt_sha256"]
    assert "boundary_readset" not in record


def test_refuses_chain_outside_stride_owner(tmp_path):
    record, receipt = _fixture(tmp_path)
    record["adjoint"]["chain_layers"] = [7, 6, 5, 3]
    with pytest.raises(ValueError, match="sealed stride chain"):
        _build(record, receipt)


def test_refuses_missing_checkpoint_boundary(tmp_path):
    record, receipt = _fixture(tmp_path)
    receipt["checkpoints"] = []
    with pytest.raises(ValueError, match="checkpoints differ"):
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
