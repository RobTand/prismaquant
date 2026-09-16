"""A resumed windowed prepare seals the order it actually reads (#607).

A fresh prepare reads in the layer/part order its data manifest declares, so
the phases it seals are true. A resumed prepare reads two blocks: the
qualification journal's already-held units, re-authenticated by
``_qualification_replay`` in journal order, and only then the layer walk over
the units the journal does not hold. Those orders differ, so before #607 the
resumed path sealed no phases at all: PB charged the whole manifest against
the ARC budget and the row stayed cold.

These tests hold the replacement contract, and they are torch-free on purpose
so they run on the fleet's CPU boxes:

* a resumed submission seals the replay roster, its exact read order and the
  journal identity those units were committed under;
* the sealed phase table is the *resumed* order -- the head, then one phase
  per replayed unit, then the layer/part phases over the units still to
  qualify -- and a replayed unit's files no longer appear in a layer phase;
* nothing leaves the read set: the resumed manifest still names every byte
  the fresh one names, only in the order the resumed run reads it;
* a journal bound to another checkpoint, or one naming a unit outside the
  campaign roster, is refused at submission rather than sealed;
* at runtime the sealed roster and identity are re-checked, and a journal,
  checkpoint or unit set that moved after submission is refused rather than
  reported as a consumed prefix.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from experiments import glm_data_manifests  # noqa: E402
import test_glm_joint_data_manifest_at_submit as campaign_fixture  # noqa: E402


def _load_by_path(name: str, relative: str):
    """Load a torch-free module without importing ``prismaquant``."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = _load_by_path("joint_replay_frontier", "prismaquant/joint_replay_frontier.py")

_canonical_sha256 = campaign_fixture.canonical_sha256
_workspace_with_sealed_checkpoint = campaign_fixture.workspace_with_sealed_checkpoint
_plant_journal = campaign_fixture.plant_journal

U1, U2 = campaign_fixture.UNITS[0][0], campaign_fixture.UNITS[0][1]
V1, V2 = campaign_fixture.UNITS[1][0], campaign_fixture.UNITS[1][1]
PRODUCED_BY = campaign_fixture.PRODUCED_BY
FRESH_ARGV = ["python3", "-u", "-m", "prismaquant.tessera_joint_aura", "prepare"]
RESUME_ARGV = [*FRESH_ARGV, "--resume"]


@pytest.fixture()
def scratch(request):
    root = ROOT / ".replay-frontier-scratch" / request.node.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def shared_mount(scratch, monkeypatch):
    monkeypatch.setattr(glm_data_manifests, "SHARED_MOUNT", str(scratch))
    return scratch


def _phase_paths(manifest, index):
    """The entry paths of phase ``index``, in manifest order."""
    phases = manifest["annotations"]["phases"]
    start = 0 if index == 0 else phases[index - 1]["cumulative_bytes"]
    end = phases[index]["cumulative_bytes"]
    seen, out = 0, []
    for entry in manifest["entries"]:
        if start <= seen < end:
            out.append(entry["path"])
        seen += entry["bytes"]
    return out


def _phase_index(manifest, name):
    return next(i for i, phase in enumerate(manifest["annotations"]["phases"])
                if phase["name"] == name)


def _prepare(plan, argv):
    return glm_data_manifests.build_joint_pass_manifest(
        str(plan), command="prepare", produced_by=PRODUCED_BY, argv=argv)


# --- the sealed contract ---------------------------------------------------


def test_a_resumed_prepare_seals_the_replay_roster_and_its_read_order(
    scratch, shared_mount,
):
    fixture, checkpoint_sha256 = _workspace_with_sealed_checkpoint(scratch)
    journal_sha256 = _plant_journal(fixture, {U1}, checkpoint_sha256=checkpoint_sha256)

    manifest = _prepare(fixture["plan"], RESUME_ARGV)
    annotations = manifest["annotations"]

    assert annotations["replay_roster"] == [U1]
    assert annotations["replay_journal_identity_sha256"] == journal_sha256
    assert annotations["replay_roster_sha256"] == replay.roster_sha256([U1])
    assert annotations["replay_read_order_sha256"] == replay.read_order_sha256(
        replay.replay_read_items([U1], {U1: campaign_fixture.MEASURED}))

    names = [phase["name"] for phase in annotations["phases"]]
    assert names[0] == "head"
    assert names[1] == "replay-0000", names
    assert annotations["replay_phase_start_units"] == {U1: "replay-0000"}

    # One phase per unit, and every unit of the campaign still has one: the
    # replayed unit in the replay block, the rest in their layer.
    assert set(annotations["phase_start_units"]) == set(fixture["names"])
    assert annotations["phase_start_units"][U1] == "replay-0000"
    assert annotations["phase_start_units"][U2].startswith("layer-0-part-")
    assert annotations["phase_start_units"][V1].startswith("layer-1-part-")


def test_the_replay_phase_precedes_the_walk_and_carries_only_its_own_bytes(
    scratch, shared_mount,
):
    fixture, checkpoint_sha256 = _workspace_with_sealed_checkpoint(scratch)
    _plant_journal(fixture, {U1}, checkpoint_sha256=checkpoint_sha256)
    manifest = glm_data_manifests.build_joint_pass_manifest(
        str(fixture["plan"]), command="prepare", produced_by=PRODUCED_BY,
        argv=RESUME_ARGV)

    captures = Path(fixture["captures"]) / "inputs"
    index = fixture["names"].index(U1)
    capture = str(captures / f"{index}.pt")

    replay_paths = _phase_paths(manifest, _phase_index(manifest, "replay-0000"))
    assert capture in replay_paths
    # Its wires and renders are read in the same block, wire before render.
    layer_paths = [path for index in range(len(manifest["annotations"]["phases"]))
                   for path in _phase_paths(manifest, index)
                   if "layer-" in manifest["annotations"]["phases"][index]["name"]]
    assert capture not in layer_paths

    # Reordering, not shrinking: the resumed read set is the fresh one.
    fresh = glm_data_manifests.build_joint_pass_manifest(
        str(fixture["plan"]), command="prepare", produced_by=PRODUCED_BY,
        argv=FRESH_ARGV)
    assert manifest["total_bytes"] == fresh["total_bytes"]
    assert ([entry["path"] for entry in manifest["entries"]]
            != [entry["path"] for entry in fresh["entries"]])


def test_a_resumed_read_set_omits_the_replayed_units_from_the_walk(
    scratch, shared_mount,
):
    fixture, checkpoint_sha256 = _workspace_with_sealed_checkpoint(scratch)
    _plant_journal(fixture, {U1, V1, V2}, checkpoint_sha256=checkpoint_sha256)
    manifest = glm_data_manifests.build_joint_pass_manifest(
        str(fixture["plan"]), command="prepare", produced_by=PRODUCED_BY,
        argv=RESUME_ARGV)
    annotations = manifest["annotations"]
    assert annotations["replay_roster"] == [U1, V1, V2]
    names = [phase["name"] for phase in annotations["phases"]]
    # Layer 1 is complete, so it has no walk phase left at all.
    assert not any(name.startswith("layer-1") for name in names), names
    assert any(name.startswith("layer-0-part-") for name in names), names


def test_a_journal_bound_to_another_checkpoint_is_refused(scratch, shared_mount):
    fixture, _ = _workspace_with_sealed_checkpoint(scratch)
    _plant_journal(fixture, {U1}, checkpoint_sha256="d" * 64)
    with pytest.raises(SystemExit, match="campaign checkpoint"):
        _prepare(fixture["plan"], RESUME_ARGV)


def test_a_journal_naming_a_foreign_unit_is_refused(scratch, shared_mount):
    fixture, checkpoint_sha256 = _workspace_with_sealed_checkpoint(scratch)
    _plant_journal(fixture, {U1}, checkpoint_sha256=checkpoint_sha256,
                   roster=[*fixture["names"], "model.layers.9.mlp.gate_proj"])
    with pytest.raises(SystemExit, match="roster"):
        _prepare(fixture["plan"], RESUME_ARGV)


def test_a_resume_with_no_journal_seals_an_empty_replay_block(scratch, shared_mount):
    fixture, _ = _workspace_with_sealed_checkpoint(scratch)
    manifest = _prepare(fixture["plan"], RESUME_ARGV)
    annotations = manifest["annotations"]
    assert annotations["replay_roster"] == []
    assert annotations["replay_phase_start_units"] == {}
    # Nothing to replay, so the walk is the fresh order.
    fresh = glm_data_manifests.build_joint_pass_manifest(
        str(fixture["plan"]), command="prepare", produced_by=PRODUCED_BY,
        argv=FRESH_ARGV)
    assert ([phase["name"] for phase in annotations["phases"]]
            == [phase["name"] for phase in fresh["annotations"]["phases"]])


def test_a_long_replay_block_is_split_into_bounded_parts(
    scratch, shared_mount, monkeypatch,
):
    """One PB phase per unit would be a phase table as long as the journal.

    The submission declares one progress phase per manifest phase and refuses
    more than 2048 of them, while the census journal holds 36,423 units, so the
    replay block is windowed by the walk's own byte budget.
    """
    fixture, checkpoint_sha256 = _workspace_with_sealed_checkpoint(scratch)
    _plant_journal(fixture, set(fixture["names"]), checkpoint_sha256=checkpoint_sha256)
    monkeypatch.setattr(glm_data_manifests._phase_module, "MAX_PHASE_BYTES", 1)
    manifest = _prepare(fixture["plan"], RESUME_ARGV)
    annotations = manifest["annotations"]

    starts = annotations["replay_phase_start_units"]
    assert set(starts) == set(fixture["names"])
    # Every unit still starts a part, and the parts are consecutive: with a
    # one-byte budget each unit starts its own.
    assert [starts[name] for name in sorted(starts)] == [
        replay.replay_phase_name(index) for index in range(len(starts))]
    names = [phase["name"] for phase in annotations["phases"]]
    assert [name for name in names if name.startswith("replay-")] == [
        replay.replay_phase_name(index) for index in range(len(starts))]
    assert set(annotations["phase_start_units"]) == set(fixture["names"])


def test_a_fully_completed_resume_seals_no_unit_into_the_walk(
    scratch, shared_mount,
):
    """Nothing is left to qualify: the walk declares only its own bytes."""
    fixture, checkpoint_sha256 = _workspace_with_sealed_checkpoint(scratch)
    _plant_journal(fixture, set(fixture["names"]), checkpoint_sha256=checkpoint_sha256)
    manifest = _prepare(fixture["plan"], RESUME_ARGV)
    annotations = manifest["annotations"]
    assert annotations["replay_roster"] == sorted(fixture["names"])
    starts = annotations["phase_start_units"]
    assert set(starts) == set(fixture["names"])
    assert all(replay.is_replay_phase(phase) for phase in starts.values())
    # Layer 1 reads no bytes of its own once its units are replayed, so it has
    # no walk phase; layer 0 keeps the one that declares its source extents.
    names = [phase["name"] for phase in annotations["phases"]]
    assert not any(name.startswith("layer-1") for name in names), names
    fresh = glm_data_manifests.build_joint_pass_manifest(
        str(fixture["plan"]), command="prepare", produced_by=PRODUCED_BY,
        argv=FRESH_ARGV)
    assert manifest["total_bytes"] == fresh["total_bytes"]


# --- the runtime-side refusals ---------------------------------------------


def _sealed(**overrides):
    cells = {U1: list(campaign_fixture.MEASURED), U2: list(campaign_fixture.MEASURED)}
    starts = {name: replay.replay_phase_name(0) for name in sorted(cells)}
    sealed = replay.seal_frontier(sorted(cells), cells,
                                 phase_start_units=starts,
                                 journal_identity_sha256="e" * 64)
    sealed.update(overrides)
    return sealed, cells


def test_a_replay_phase_table_is_a_contiguous_partition_of_the_roster():
    cells = {name: list(campaign_fixture.MEASURED) for name in (U1, V1)}
    good = {U1: replay.replay_phase_name(0), V1: replay.replay_phase_name(1)}
    replay.seal_frontier(sorted(cells), cells, phase_start_units=good,
                         journal_identity_sha256="e" * 64)
    for broken in ({U1: replay.replay_phase_name(1), V1: replay.replay_phase_name(0)},
                   {U1: replay.replay_phase_name(0), V1: replay.replay_phase_name(2)},
                   {U1: "layer-0-part-0", V1: replay.replay_phase_name(1)},
                   {U1: replay.replay_phase_name(0)}):
        with pytest.raises(ValueError):
            replay.seal_frontier(sorted(cells), cells, phase_start_units=broken,
                                 journal_identity_sha256="e" * 64)


def test_a_sealed_manifest_carries_its_part_contiguity_into_the_runtime():
    sealed, cells = _sealed()
    sealed[replay.PHASE_START_UNITS_KEY] = {U1: replay.replay_phase_name(0),
                                            U2: replay.replay_phase_name(2)}
    with pytest.raises(ValueError, match="skips a part"):
        replay.sealed_from_annotations(sealed)
    # A part cannot start after a later one has already begun: the runtime
    # announces a part when it reaches its first unit, so such a table would
    # release a prefix the action has not read.
    three = {name: list(campaign_fixture.MEASURED) for name in (U1, V1, U2)}
    sealed = replay.seal_frontier(
        sorted(three), three,
        phase_start_units={name: replay.replay_phase_name(0) for name in three},
        journal_identity_sha256="e" * 64)
    # In roster order (U1, U2, V1) the parts must never step backwards.
    sealed[replay.PHASE_START_UNITS_KEY] = {U1: replay.replay_phase_name(0),
                                            U2: replay.replay_phase_name(1),
                                            V1: replay.replay_phase_name(0)}
    with pytest.raises(ValueError, match="read order"):
        replay.sealed_from_annotations(sealed)


def test_the_sealed_frontier_round_trips_through_a_manifest():
    sealed, _ = _sealed()
    assert replay.sealed_from_annotations(sealed) == sealed


def test_a_partly_sealed_frontier_is_refused():
    sealed, _ = _sealed()
    for key in (replay.ROSTER_SHA256_KEY, replay.READ_ORDER_SHA256_KEY,
                replay.JOURNAL_IDENTITY_KEY, replay.PHASE_START_UNITS_KEY):
        partial = {k: v for k, v in sealed.items() if k != key}
        with pytest.raises(ValueError, match="sealed replay frontier"):
            replay.sealed_from_annotations(partial)


def test_a_roster_that_does_not_hash_to_its_digest_is_refused():
    sealed, _ = _sealed()
    sealed[replay.ROSTER_SHA256_KEY] = "0" * 64
    with pytest.raises(ValueError, match="does not hash"):
        replay.sealed_from_annotations(sealed)


def test_a_replayed_unit_must_start_in_a_replay_phase():
    sealed, _ = _sealed()
    sealed[replay.PHASE_START_UNITS_KEY] = {name: "layer-0-part-0"
                                           for name in sealed[replay.ROSTER_KEY]}
    with pytest.raises(ValueError, match="replay phase"):
        replay.sealed_from_annotations(sealed)


def test_the_matching_roster_is_accepted_and_every_move_is_refused():
    sealed, cells = _sealed()
    replay.require_replay_matches(sealed, completed=sorted(cells),
                                  cells_by_unit=cells,
                                  journal_identity_sha256="e" * 64)

    with pytest.raises(RuntimeError, match="journal identity changed"):
        replay.require_replay_matches(sealed, completed=sorted(cells),
                                      cells_by_unit=cells,
                                      journal_identity_sha256="f" * 64)

    with pytest.raises(RuntimeError, match="no longer holds the sealed replay roster"):
        replay.require_replay_matches(sealed, completed=[U1], cells_by_unit=cells,
                                      journal_identity_sha256="e" * 64)

    with pytest.raises(RuntimeError, match="no longer holds the sealed replay roster"):
        replay.require_replay_matches(sealed, completed=[*sorted(cells), V1],
                                      cells_by_unit={**cells, V1: list(campaign_fixture.MEASURED)},
                                      journal_identity_sha256="e" * 64)


def test_a_reordered_read_set_is_refused():
    sealed, cells = _sealed()
    reordered = {name: list(reversed(cells[name])) for name in cells}
    with pytest.raises(RuntimeError, match="read order"):
        replay.require_replay_matches(sealed, completed=sorted(cells),
                                      cells_by_unit=reordered,
                                      journal_identity_sha256="e" * 64)


def test_the_replay_read_order_is_a_complete_unit_at_a_time():
    cells = {U1: list(campaign_fixture.MEASURED)}
    items = replay.replay_read_items(sorted(cells), cells)
    assert items == [
        (replay.CAPTURE, U1, None),
        (replay.WIRE, U1, campaign_fixture.MEASURED[0]),
        (replay.RENDER, U1, campaign_fixture.MEASURED[0]),
        (replay.WIRE, U1, campaign_fixture.MEASURED[1]),
        (replay.RENDER, U1, campaign_fixture.MEASURED[1]),
    ]
    assert replay.replay_phase_name(7) == "replay-0007"
