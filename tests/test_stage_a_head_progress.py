"""Stage A's boundary-0 capture reports under its declared head phase.

The adjoint read plan declares ``head, forward-*, chain-*`` -- never the
generic ``layer-<L>`` name the boundary-storage callback derives -- so a
Stage A reporter that starts with no phase committed nothing for the whole
input-boundary-0 loop: every ``entry(layer=0)`` tried the undeclared
``layer-0`` and ``_report`` returned while ``phase is None``. The live f995
capture showed exactly that (the undeclared-phase warning, then the first
record only at ``forward-000``/36935 once the forward observer fired).
The wiring fix enters the declared head phase before boundary capture --
moving no units: the head walk's own committed total is already the
reporter's ``base_units``, and only published entry files advance the
count (Astra startup audit, 2026-09-20).

RED-first: the helper imports live inside the tests that need them, and the
wiring assertion reads the real core, so this module collects unmodified
and fails before the source change.
"""
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from prismaquant.joint_layer_quanta import (  # noqa: E402
    adjoint_read_plan_phase_names)
from prismaquant.joint_run_progress import JointRunProgress  # noqa: E402


class Clock:
    """A clock the test moves, so the cadence is tested and not the wall."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Recorder:
    def __init__(self):
        self.lines = []
        self.records = []

    def log(self, message):
        self.lines.append(message)

    def commit(self, phase, units):
        self.records.append((phase, units))
        return True


def stage_a_reporter(base_units, *, layers=3, partitions=2):
    """The reporter Stage A constructs, under the adjoint phase list."""
    recorder, clock = Recorder(), Clock()
    progress = JointRunProgress(
        layers=layers, partitions=partitions, base_units=base_units,
        log=recorder.log, clock=clock, commit=recorder.commit,
        interval_s=60.0, phases=adjoint_read_plan_phase_names(layers))
    return recorder, clock, progress


def stage_a_storage(tmp_path):
    from prismaquant.cost_streaming import (
        BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts)

    storage = StreamedBoundaryArtifacts({
        "schema": BOUNDARY_STORAGE_SCHEMA,
        "directory": str(tmp_path / "exact"),
        "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
        "max_artifact_bytes": 1 << 24, "prefetch_batches": 1})
    storage.bind({"source_model": "fixture"}, n_probes=1)
    return storage


def _head_wired(tmp_path, base_units=7, writes=2):
    """Stage A's own wiring order: head phase first, then the watcher."""
    from prismaquant.joint_cost_stage_a import stage_a_head_progress_start

    recorder, clock, progress = stage_a_reporter(base_units)
    storage = stage_a_storage(tmp_path)
    stage_a_head_progress_start(progress)
    storage.watch_progress(progress)
    return recorder, clock, progress, storage


def test_boundary_zero_commits_under_head_before_any_forward_observer(tmp_path):
    """The input-boundary loop reports head units without the observer.

    The head walk's committed total is the base; each published boundary-0
    entry file adds one unit on top; the record lands on the cadence with
    no forward observer in sight. Pre-fix the same sequence committed
    nothing at all -- the live f995 shape.
    """
    torch = pytest.importorskip("torch")
    recorder, clock, progress, storage = _head_wired(tmp_path, base_units=7)
    for batch in range(2):
        storage.write(torch.zeros(4), batch_index=batch, boundary_index=0)
    assert progress.units == 9
    assert progress.phase == "head"
    assert recorder.records == []          # inside the report interval
    clock.advance(61)
    progress.flush(force=True)
    assert recorder.records == [("head", 9)]


def test_forward_zero_comes_only_from_the_source_observer(tmp_path):
    """The head phase holds until the runner's first forward callback."""
    torch = pytest.importorskip("torch")
    from prismaquant.joint_cost_stage_a import stage_a_forward_observer

    recorder, clock, progress, storage = _head_wired(tmp_path, base_units=7)
    storage.write(torch.zeros(4), batch_index=0, boundary_index=0)
    clock.advance(61)
    progress.flush(force=True)
    assert recorder.records == [("head", 8)]
    observe = stage_a_forward_observer(progress)
    observe("source_loading", 0, 0)
    assert recorder.records == [("head", 8), ("forward-000", 8)]
    observe("something_else", 1, 0)
    assert progress.phase == "forward-000"


def test_head_base_and_intents_alone_move_no_units(tmp_path):
    """Entering head fabricates nothing; only a published file moves the count.

    Landed names only -- this pins the generic contract on both sides of
    the wiring fix: the base is durable history, entering a phase is not a
    unit, and a cadence with nothing new commits nothing.
    """
    recorder, clock, progress = stage_a_reporter(base_units=7)
    from prismaquant.joint_run_progress import HEAD_PHASE

    progress.enter(HEAD_PHASE)
    assert progress.units == 7
    clock.advance(120)
    progress.flush(force=True)
    assert recorder.records == [("head", 7)]
    progress.flush(force=True)
    assert recorder.records == [("head", 7)]      # replay commits nothing


def test_the_core_wires_the_head_phase_before_boundary_capture():
    """The real call site: head enters before the storage watcher is set.

    Reads the shipped core, so the wiring itself -- not a test's replica --
    is what this pins.
    """
    source = (REPO / "prismaquant" / "joint_cost_stage_a.py").read_text(
        encoding="utf-8")
    core = source[source.index("def run_adjoint_capture_core"):]
    start = core.index("stage_a_head_progress_start(progress)")
    watcher = core.index("storage.watch_progress(progress)")
    assert start < watcher
