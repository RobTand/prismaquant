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


def _run_inputs():
    return row.QuantumRunInputs(
        row.QuantumFileBinding("/records/q.json", "q" * 64),
        row.QuantumFileBinding("/plan.json", "p" * 64),
        row.QuantumFileBinding("/prepared.json", "r" * 64),
        row.QuantumFileBinding("/slice.json", "s" * 64), "m" * 64, "/output")


def test_run_input_pairs_keep_literal_consumer_argv_and_optional_band_order():
    band = row.QuantumBandArguments(row.QuantumFileBinding("/handoff.json", "h" * 64),
                                    True, '[{"phase":"handoff-load"}]')
    assert _run_inputs().argv(allowed_tiers="ram,ssd", band=band) == [
        "python3", "-m", "prismaquant.joint_cost_quantum",
        "--quantum", "/records/q.json", "--quantum-sha256", "q" * 64,
        "--plan", "/plan.json", "--plan-sha256", "p" * 64,
        "--prepared", "/prepared.json", "--prepared-sha256", "r" * 64,
        "--adjoint-slice", "/slice.json", "--adjoint-slice-sha256", "s" * 64,
        "--data-manifest-sha256", "m" * 64, "--allowed-tiers", "ram,ssd",
        "--resume", "--output-root", "/output", "--adjoint-handoff", "/handoff.json",
        "--adjoint-handoff-sha256", "h" * 64, "--emit-adjoint-handoff",
        "--progress-grace-derivation", '[{"phase":"handoff-load"}]']
    assert _run_inputs().argv(allowed_tiers="ram,ssd")[-3:] == ["--resume", "--output-root", "/output"]


def test_manifest_annotation_remains_an_unbound_locator():
    assert row.QuantumManifestInvocation("/records/q.json", "/output").argv() == [
        "python3", "-m", "prismaquant.joint_cost_quantum", "--quantum", "/records/q.json",
        "--output-root", "/output"]


@pytest.mark.parametrize("reader,source,operator,expected", [(1, 3, 2, 4), (8, 3, 2, 8), (1, 0, 5, 6)])
def test_row_demand_counts_the_existing_native_and_prefetch_workers(reader, source, operator, expected):
    demand = row.QuantumRowDemand.from_policy(
        {"limits": {"physical_bytes": int(7.5 * 1024**3), "gpu_bytes": 3 * 1024**3}},
        environment={"PRISMAQUANT_LAYER_READ_THREADS": str(reader)},
        source_prefetch={"prefetch_workers": source}, operator_prefetch={"prefetch_workers": operator})
    assert (demand.memory_gib, demand.gpu_gib, demand.cpus) == ("7.5", "3", expected)
    default = row.QuantumRowDemand.from_policy(None,
        environment={"PRISMAQUANT_LAYER_READ_THREADS": "unused-invalid"})
    assert (default.memory_gib, default.gpu_gib, default.cpus) == ("104", "80", 10)


def test_pb_envelope_keeps_progress_options_environment_duplicates_and_separator():
    envelope = row.QuantumPBEnvelope("/manifest.gz", [("head", 45), ("handoff-load", 91)],
        row.QuantumRowDemand("7.5", "3", 4), ["gb10", "native-image"], -5,
        ["python3", "launcher.py", "--spec", "{}", "--", "payload"],
        [("SCRATCH", "a"), ("SCRATCH", "b")], "/template.json", "repo@sha256:image")
    assert envelope.argv(python="/python", pbrun="/pbrun.py", container_image_flag="--container-image",
                         dev_mode_assignment="PRISMAQUANT_DEV_MODE=1") == [
        "/python", "/pbrun.py", "--tag", "gb10", "--tag", "native-image",
        "--data-manifest", "/manifest.gz", "--residency", "stage", "--residency-ram", "auto",
        "--progress-phase", "head=45", "--progress-phase", "handoff-load=91",
        "--priority", "-5", "--demand", "gpu=1,mem_gb=7.5", "--gpu-memory-gb", "3", "--cpus", "4",
        "--produced-output-template", "/template.json", "--container-image", "repo@sha256:image",
        "--env", "SCRATCH=a", "--env", "SCRATCH=b", "--env", "PRISMAQUANT_DEV_MODE=1",
        "--detach", "--", "python3", "launcher.py", "--spec", "{}", "--", "payload"]
