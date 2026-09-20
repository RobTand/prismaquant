"""Stage A read plan: forward lifetimes in true consumption order (PQ #837).

RED-first: ``build_adjoint_manifest`` seals v1 ``head, chain-000..044``
ascending with no forward lifetimes, while the capture consumes head →
forward 0..44 → tail work → reverse 44..0 and reports only head +
chain_044..000. Only long-landed names are imported at module scope, so RED
collects and fails on the v1 shape/order assertions; new-helper imports live
inside the tests that need them. There is deliberately NO standalone tail
phase: published ``manifest_phase_ranges`` drops cumulative==previous
phases, and reporting a name the sealed plan does not carry resets
``remaining`` to start / ``accepted`` to false -- the tail checkpoint work
commits durable units under forward-last instead. The timeline tests drive
the true callback order through the real monotonic reporter and check
byte-level eviction accounting plus exact durable-unit counting.
"""
from __future__ import annotations

import re
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant import joint_run_progress as jrp
from prismaquant.joint_layer_quanta import build_adjoint_manifest

N_LAYERS = 3
PUBLISHED_PB_SRC = Path("/mnt/shared/prismabuild-fleet/repo/src")


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class _Recorder:
    def __init__(self):
        self.lines = []
        self.records = []

    def log(self, message):
        self.lines.append(message)

    def commit(self, phase, units):
        self.records.append((phase, units))
        return True


def _expected_order():
    """The frozen consumption order (coordination note stage-a-phase-fix):
    head, forward ascending, reverse descending. No tail phase."""
    return (["head"]
            + [f"forward-{layer:03d}" for layer in range(N_LAYERS)]
            + [f"chain-{layer:03d}" for layer in reversed(range(N_LAYERS))])


def _parent():
    """A small v1 parent manifest: head + three layer phases of model-dir
    source extents, the shape ``phase_ranges`` cuts."""
    entries = [
        {"path": "/mnt/shared/calib/calib_ids.pt", "offset": 0,
         "bytes": 100, "sha256": None},
        {"path": "/mnt/shared/model/head.pt", "offset": 0,
         "bytes": 200, "sha256": None},
        {"path": "/mnt/shared/model/layer0.pt", "offset": 0,
         "bytes": 1000, "sha256": None},
        {"path": "/mnt/shared/model/layer0.pt", "offset": 1000,
         "bytes": 2000, "sha256": None},
        {"path": "/mnt/shared/model/layer1.pt", "offset": 0,
         "bytes": 4000, "sha256": None},
        {"path": "/mnt/shared/model/layer2.pt", "offset": 0,
         "bytes": 500, "sha256": None},
        {"path": "/mnt/shared/model/layer2.pt", "offset": 500,
         "bytes": 600, "sha256": None},
    ]
    phases = []
    cumulative = 0
    for name, size in (("head", 300), ("layer-0", 3000), ("layer-1", 4000),
                       ("layer-2", 1100)):
        cumulative += size
        phases.append({"name": name, "bytes": size,
                       "cumulative_bytes": cumulative})
    return {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "produced_by": {"tool": "fixture"},
        "mount_prefix": "/mnt/shared",
        "entries": entries,
        "entry_count": len(entries),
        "total_bytes": cumulative,
        "annotations": {
            "campaign_scope": {"campaign": "phase-fixture"},
            "phases": phases,
        },
    }


def _manifest():
    parent = _parent()
    return build_adjoint_manifest(
        {"model": "/mnt/shared/model"}, parent,
        plan_path="/fixture/plan.json", plan_sha256="a" * 64,
        prepared_path="/fixture/prepared.json", prepared_sha256="b" * 64,
        parent_manifest_sha256="c" * 64, output_root="/fixture/out")


def _reporter(order):
    recorder = _Recorder()
    progress = jrp.JointRunProgress(
        layers=N_LAYERS, partitions=1, log=recorder.log, clock=_Clock(),
        commit=recorder.commit, interval_s=60.0, phases=tuple(order),
        resolver=None)
    return recorder, progress


def _published_pb():
    """The published PrismaBuild contract owners, or a skip naming why not.

    The fleet's own tier loops and pbrun read these bytes; a test that
    re-implemented the phase rules could stay green while the sealed plan
    still dropped a phase. A half-visible mount is a skip, not a pass."""
    if not (PUBLISHED_PB_SRC / "prismabuild" / "core.py").is_file():
        pytest.skip(f"published PrismaBuild not visible at {PUBLISHED_PB_SRC}")
    if str(PUBLISHED_PB_SRC) not in sys.path:
        sys.path.insert(0, str(PUBLISHED_PB_SRC))
    import prismabuild.core as core
    import prismabuild.storage_tiers as tiers
    import prismabuild.residency_plan as plans
    for module in (core, tiers, plans):
        if not Path(module.__file__).resolve().is_relative_to(
                PUBLISHED_PB_SRC.resolve()):
            pytest.skip(f"a different prismabuild is already imported: {module}")
    return core, tiers, plans


# -- the v2 table (fails on base: v1 shape, ascending order) --------------

def test_single_ordered_helper_owns_all_phase_names():
    """One spelling of the consumption order, shared by builder, capture
    and (through the manifest) the dispatch lane."""
    from prismaquant.joint_layer_quanta import (
        ADJOINT_TAIL_PHASE,
        adjoint_chain_phase_name,
        adjoint_forward_phase_name,
        adjoint_read_plan_phase_names,
    )
    assert list(adjoint_read_plan_phase_names(N_LAYERS)) == _expected_order()
    assert ADJOINT_TAIL_PHASE == "tail"
    assert [adjoint_forward_phase_name(layer) for layer in range(N_LAYERS)] == [
        "forward-000", "forward-001", "forward-002"]
    assert [adjoint_chain_phase_name(layer) for layer in range(N_LAYERS)] == [
        "chain-000", "chain-001", "chain-002"]


def test_adjoint_manifest_is_v2_in_true_consumption_order():
    """Schema v2, no v1 phase table, phases head → forward → reverse, over
    the unchanged entries list. No standalone tail phase."""
    manifest = _manifest()
    assert manifest["schema"] == "prismaquant.prismabuild.data_manifest.v2"
    assert "phases" not in manifest["annotations"]
    assert manifest["annotations"]["entry_point"] == \
        "prismaquant.joint_adjoint_capture"
    names = [phase["name"] for phase in manifest["read_plan"]["phases"]]
    assert names == _expected_order()
    assert "tail" not in names
    # The entries list is the old assembly, untouched: head entries plus
    # the per-layer source-extent groups, no duplication for the repeats.
    assert manifest["entry_count"] == 7
    assert manifest["total_bytes"] == 300 + 3000 + 4000 + 1100
    assert [entry["path"] for entry in manifest["entries"]][:2] == [
        "/mnt/shared/calib/calib_ids.pt", "/mnt/shared/model/head.pt"]


def test_read_plan_bytes_agree_and_cover_every_entry():
    """Independent recomputation: per-phase bytes, running cumulative,
    timeline total, every unique entry referenced at least once, no
    within-phase repeat, names unique."""
    manifest = _manifest()
    entries = manifest["entries"]
    seen_global: set[int] = set()
    seen_names: set[str] = set()
    cumulative = 0
    per_phase = {"head": 300, "forward-000": 3000, "forward-001": 4000,
                 "forward-002": 1100, "chain-002": 1100,
                 "chain-001": 4000, "chain-000": 3000}
    for phase in manifest["read_plan"]["phases"]:
        assert phase["name"] not in seen_names
        seen_names.add(phase["name"])
        indices = phase["entry_indices"]
        assert len(set(indices)) == len(indices)
        size = sum(entries[index]["bytes"] for index in indices)
        assert size == per_phase[phase["name"]] == phase["bytes"]
        cumulative += size
        assert phase["cumulative_bytes"] == cumulative
        seen_global.update(indices)
    assert manifest["read_plan"]["read_bytes"] == cumulative == (
        300 + 2 * (3000 + 4000 + 1100))
    assert seen_global == set(range(len(entries)))


def test_forward_and_chain_share_extents_without_duplicating_entries():
    """The repeated read is two timeline references to one entry list, not
    two copies of the bytes. Non-layer weights (embed, lm_head) load once
    at build and are never unloaded per layer -- install/unload are scoped
    to the `{layers_prefix}{L}.` tensor prefix -- so they ride the head
    entries and need no phase of their own."""
    manifest = _manifest()
    by_name = {phase["name"]: phase["entry_indices"]
               for phase in manifest["read_plan"]["phases"]}
    for layer in range(N_LAYERS):
        assert by_name[f"forward-{layer:03d}"] == by_name[f"chain-{layer:03d}"]


def test_generated_plan_is_interoperable_with_published_pb():
    """The exact gap this change exists to close, against the published
    bytes: the generated table validates, every read phase survives into
    the sealed residency ranges (a dropped phase is a phase the window
    never stages), and driving the phases in order keeps `accepted` true
    with `remaining` shrinking from the reported phase -- never a reset."""
    core, tiers, plans = _published_pb()
    manifest = _manifest()
    normalized = core.validate_data_manifest(manifest)
    ranges = tiers.manifest_phase_ranges(normalized)
    plan_names = [entry["name"] for entry in ranges]
    read_names = [phase["name"] for phase in normalized["read_plan"]["phases"]]
    assert plan_names == read_names
    plan = {"phases": [{"name": name} for name in plan_names]}
    for position, name in enumerate(read_names):
        assert plans.accepted(plan, name), f"{name} is not a plan phase"
        remaining = [phase["name"] for phase in
                     plans.remaining(plan, name)]
        assert remaining == read_names[position:]
    # The failure mode a standalone tail phase would buy: a reported name
    # the plan does not carry resets tracking to the beginning.
    assert plans.accepted(plan, "tail") is False
    assert [phase["name"] for phase in plans.remaining(plan, "tail")] == \
        read_names


# -- the capture timeline, through the real reporter ---------------------

def test_capture_timeline_advances_monotonically_with_exact_units():
    """The full true sequence, storage writes interleaved the way the
    capture lands boundary entries between phase advances, tail-checkpoint
    counting under forward-last: every phase accepted in order, and every
    commit carries exactly the entries landed so far -- phases never
    fabricate units, entries never move the phase."""
    from prismaquant.joint_cost_stage_a import stage_a_forward_observer
    from prismaquant.joint_layer_quanta import adjoint_chain_phase_name
    order = _expected_order()
    recorder, progress = _reporter(order)
    observe = stage_a_forward_observer(progress)
    landed = 0

    def _write_entries(count):
        nonlocal landed
        for _ in range(count):
            progress.entry(layer=0, partition=0)
            landed += 1

    progress.enter("head")
    progress.flush(force=True)
    assert recorder.records[-1] == ("head", 0)
    for layer in range(N_LAYERS):
        observe("source_loading", layer, 0)
        observe("capture_forward", layer, 0)
        assert recorder.records[-1] == (f"forward-{layer:03d}", landed)
        _write_entries(layer + 1)
    # The tail checkpoint lands while the read plan stays on forward-last:
    # units advance, the phase does not move.
    tail_units = landed
    progress.entry(layer=N_LAYERS, partition=0, kind="tail_checkpoint")
    progress.flush(force=True)
    assert progress.units == tail_units + 1
    assert progress.phase == "forward-002"
    assert recorder.records[-1] == ("forward-002", tail_units + 1)
    landed = tail_units + 1
    for layer in reversed(range(N_LAYERS)):
        progress.enter(adjoint_chain_phase_name(layer))
        progress.flush(force=True)
        assert recorder.records[-1] == (
            adjoint_chain_phase_name(layer), landed)
        _write_entries(1)
    committed = [phase for phase, _units in recorder.records]
    # The tail checkpoint commits a second record under forward-last
    # (units advanced, phase unchanged); every read-plan phase is committed
    # at least once, in order, and no commit ever steps back.
    indices = [order.index(phase) for phase in committed]
    assert all(later >= earlier
               for earlier, later in zip(indices, indices[1:]))
    assert sorted(set(indices)) == list(range(len(order)))
    # The only names the reporter may drop here are storage's default
    # `layer-N` entry namings (the joint-run convention, undeclared in a
    # Stage A declaration); no head/forward/chain name may drop.
    dropped = {match.group(1) for line in recorder.lines
               for match in [re.search(r"phase '(.*)' is not one", line)]
               if match is not None}
    assert dropped and all(
        re.fullmatch(r"layer-\d+", name) for name in dropped)


def test_reverse_entries_stay_ahead_until_their_turn():
    """The eviction accounting the fix exists for: when chain-044 is
    reported, the forward timeline is released but every other reverse
    reference is still ahead; reporting a phase releases only the bytes
    before it, never its own."""
    manifest = _manifest()
    phases = manifest["read_plan"]["phases"]
    before = {}
    running = 0
    for phase in phases:
        before[phase["name"]] = running
        running += phase["bytes"]
    assert before["chain-002"] == 300 + 3000 + 4000 + 1100
    assert before["chain-002"] < manifest["read_plan"]["read_bytes"]
    assert before["chain-001"] == before["chain-002"] + 1100
    # chain-000's entries are referenced at the very end of the timeline.
    last = phases[-1]
    assert last["name"] == "chain-000"
    assert last["cumulative_bytes"] == manifest["read_plan"]["read_bytes"]


def test_boundary_writes_keep_counting_without_moving_the_phase():
    """Storage `entry()` calls still increment durable units, but their
    default `layer-N` naming (the joint-run convention) is undeclared in a
    Stage A declaration: it must neither advance to a future phase nor erase
    the current forward phase."""
    recorder, progress = _reporter(_expected_order())
    from prismaquant.joint_cost_stage_a import stage_a_forward_observer
    observe = stage_a_forward_observer(progress)
    progress.enter("head")
    observe("source_loading", 0, 0)
    observe("capture_forward", 0, 0)
    observe("source_loading", 1, 0)
    assert progress.phase == "forward-001"
    units_before = progress.units
    progress.entry(layer=2, partition=0)
    assert progress.units == units_before + 1
    assert progress.phase == "forward-001"
