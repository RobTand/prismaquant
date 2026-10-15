"""Quantum phase values retain the sealed wire and runtime work currency."""
from __future__ import annotations

import pytest

from prismaquant import joint_quantum_row as row


@pytest.mark.parametrize("name,kind,coordinates", [
    ("chain-003-source", row.ChainSourcePhase, {"layer": 3}),
    ("chain-003-bound", row.ChainBoundaryPhase, {"layer": 3}),
    ("own-002-source", row.OwnSourcePhase, {"layer": 2}),
    ("render-12", row.RenderPhase, {"window": 12}),
    ("replay-12-p3", row.ReplayPhase, {"window": 12, "probe": 3}),
    ("spill-p3", row.SpillPhase, {"probe": 3}),
])
def test_wire_phases_parse_into_their_execution_coordinates(name, kind, coordinates):
    phase = row.parse_quantum_phase(name)
    assert isinstance(phase, kind)
    assert vars(phase) == coordinates
    assert phase.name == name


def test_windowed_and_spill_plans_keep_literal_consumption_order():
    common = {"n_probes": 2, "replay_windows": 2, "render_phases": True}
    assert row.quantum_executable_phase_names([3], 2, **common) == (
        "head", "checkpoint-load", "chain-003-source", "chain-003-bound",
        "own-002-source", "render-00", "replay-00-p0", "replay-00-p1",
        "render-01", "replay-01-p0", "replay-01-p1")
    assert row.quantum_executable_phase_names([3], 2, replay_mode="spill", **common) == (
        "head", "checkpoint-load", "chain-003-source", "chain-003-bound",
        "own-002-source", "render-00", "spill-p0", "spill-p1", "render-01")


@pytest.mark.parametrize("name", ["head", "checkpoint-load", "handoff-load", "unknown", "render-1"])
def test_noncompute_phases_keep_no_work_and_optional_tail(name):
    phase = row.parse_quantum_phase(name)
    assert isinstance(phase, row.NamedPhase)
    assert phase.name == name
    common = {"replay_mode": "spill", "entries": 9, "n_probes": 3, "capture_batch": 2}
    assert row.compute_phase_work(name, **common) is None
    assert row.compute_phase_work(name, runs_tail=True, **common) == [{
        "kind": "tail", "units": 1,
        "work": "the quantum's tail after its last window, which declares no phase and commits no unit"}]


def test_spill_work_counts_capture_groups_then_replay_then_tail():
    terms = row.compute_phase_work("spill-p2", replay_mode="spill",
        entries=9, n_probes=3, capture_batch=2, runs_tail=True)
    assert [(term["kind"], term["units"]) for term in terms] == [
        ("spill-capture", 5), ("spill-replay", 1), ("tail", 1)]
    assert terms[0]["work"] == "probe 2's spill capture: one group per capture_batch stored batches"
    assert terms[1]["work"] == "the first window's spill replay for probe 2"


def test_streamed_incoming_entries_remain_read_bytes_and_do_not_invent_work():
    annotations = {"band_serial": {"streamed_incoming": {"spill-p0": 2}},
                   "checkpoint_incoming": {"streamed_incoming": {"spill-p0": 1}}}
    count = row.phase_work_entries("spill-p0", {"entries": 7}, annotations)
    assert count == 4
    terms = row.compute_phase_work("spill-p0", replay_mode="spill", entries=count,
                                  n_probes=2, capture_batch=2)
    assert [(term["kind"], term["units"]) for term in terms] == [
        ("spill-capture", 2), ("spill-replay", 1)]
    assert row.phase_work_entries("spill-p0", {"entries": 1}, annotations) is None
