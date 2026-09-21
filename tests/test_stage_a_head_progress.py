"""Stage A's boundary-0 capture reports under its declared head phase.

The adjoint read plan declares ``head, forward-*, chain-*`` -- never the
generic ``layer-<L>`` name the boundary-storage callback derives -- so a
Stage A reporter that starts with no phase commits nothing for the whole
input-boundary-0 loop: every ``entry(layer=0)`` tries the undeclared
``layer-0`` and ``_report`` returns while ``phase is None``. These tests
drive the REAL ``run_adjoint_capture_core`` wiring (the runtime fixtures
from ``test_layer_major_boundary_capture``, CPU-only): a capture that
publishes two boundary-0 entries across the report interval and then
raises before any forward work must leave a ``head`` record behind, with
the head walk's committed total as the base and only published entries
counted. The forward source observer stays the only transition to
``forward-000``.
"""
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from prismaquant.joint_cost_stage_a import (  # noqa: E402
    run_adjoint_capture_core, stage_a_forward_observer)
from prismaquant.joint_layer_quanta import (  # noqa: E402
    adjoint_read_plan_phase_names)
from prismaquant.joint_run_progress import JointRunProgress  # noqa: E402
from test_joint_cost_quantum_runtime import _execution  # noqa: E402
from test_layer_major_boundary_capture import draw, fixture  # noqa: E402
from test_streamed_cost_checkpoints import _model_identity  # noqa: E402


class Clock:
    """A clock the test moves, so the cadence is tested and not the wall."""

    def __init__(self):
        self.now = 1000.0

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


class StoppedBeforeForward(Exception):
    """The sentinel: the fake capture ends before any forward work."""


def test_the_core_reports_boundary_zero_under_head_before_forward_work(
        tmp_path, monkeypatch):
    """The real core wiring: durable boundary-0 output reports as head.

    The capture writes two boundary-0 entries across one report interval
    and then raises, so nothing after the input-boundary loop runs: no
    forward observer fires, and the only record that may exist is the head
    phase's, carrying the head walk's committed base plus exactly the two
    published entries.
    """
    _model, context, runner, _cache = fixture()
    context.settle_prefetched_layers = lambda layers: None
    recorder, clock = Recorder(), Clock()
    progress = JointRunProgress(
        layers=runner.num_layers, partitions=1, base_units=7,
        log=recorder.log, clock=clock, commit=recorder.commit,
        interval_s=60.0, phases=adjoint_read_plan_phase_names(runner.num_layers))
    wired = {}

    def capture(partitions, *, storage, source_phase):
        wired["observer"] = source_phase
        for batch in range(2):
            storage.write(torch.zeros(4), batch_index=batch,
                          boundary_index=0)
            clock.advance(61)
        raise StoppedBeforeForward("input boundary loop complete")

    monkeypatch.setattr(runner, "capture_layer_major_boundaries", capture)
    with pytest.raises(StoppedBeforeForward):
        run_adjoint_capture_core(
            runner, draw(), execution=_execution(tmp_path),
            output_root=tmp_path / "campaign", stride=2,
            source_model_identity=_model_identity("joint-source"),
            unit_roster_sha256="a" * 64, plan_sha256="d" * 64,
            prepared_sha256="e" * 64, read_manifest_sha256="f" * 64,
            implementation_sha256="0" * 64, progress=progress)
    assert wired["observer"] is not None
    assert recorder.records == [("head", 9)]


def test_the_head_base_and_phases_alone_move_no_units():
    """Entering head fabricates nothing; only a published file moves the count.

    Landed names only: the base is durable history, entering a phase is not
    a unit, and a cadence with nothing new commits nothing twice.
    """
    recorder, clock = Recorder(), Clock()
    progress = JointRunProgress(
        layers=2, partitions=1, base_units=7, log=recorder.log,
        clock=clock, commit=recorder.commit, interval_s=60.0,
        phases=adjoint_read_plan_phase_names(2))
    progress.enter("head")
    assert progress.units == 7
    clock.advance(120)
    progress.flush(force=True)
    assert recorder.records == [("head", 7)]
    progress.flush(force=True)
    assert recorder.records == [("head", 7)]      # replay commits nothing


def test_forward_zero_comes_only_from_the_source_observer():
    """After head, the observer -- and nothing else -- names forward-000."""
    recorder, clock = Recorder(), Clock()
    progress = JointRunProgress(
        layers=2, partitions=1, base_units=7, log=recorder.log,
        clock=clock, commit=recorder.commit, interval_s=60.0,
        phases=adjoint_read_plan_phase_names(2))
    progress.enter("head")
    observe = stage_a_forward_observer(progress)
    observe("source_loading", 0, 0)
    assert recorder.records == [("forward-000", 7)]
    observe("something_else", 1, 0)
    assert progress.phase == "forward-000"
