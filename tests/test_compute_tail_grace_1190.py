"""The quantum's tail runs under the last declared phase's grace (PQ #1190).

After a quantum's last window commits, the work that remains declares no
phase: the band-serial handoff write, the payload assembly and the final
check of every row (the ``payload`` span), then the runner shutdown and the
residency report (the ``teardown`` span; PQ #1187). It commits no unit, so it
runs on the last declared phase's no-progress clock, which restarts at that
window's commit. Before this, that phase's derived grace priced only its own
pass: nothing checked that it also covers the tail. On v7 (layer 44, before
#1184) the tail took 796 s under ``render-14``'s 2,168 s grace.

The dispatcher now adds a ``tail`` term to the last declared phase: the
measured tail from a ``tail`` compute ceiling when the row is in its scope,
the blanket otherwise, like every other compute term.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tests", ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from test_compute_phase_grace_1165 import (  # noqa: E402
    FIXTURE_CEILINGS, _ceiling, _dispatch, _graces, _plan, _read_term, _stamps)

#: A measured tail far above any grace the fixture row derives, so a grace
#: that leaves the tail out cannot cover it by accident.
LONG_TAIL_S = 50_000.0


def _last_phase(manifest):
    return [phase["name"] for phase in manifest["read_plan"]["phases"]][-1]


def _tail_terms(stamps):
    return {stamp["phase"]: term for stamp in stamps
            for term in stamp.get("compute", ()) if term["kind"] == "tail"}


# -- red: the defect, at the dispatcher ------------------------------------


@pytest.mark.parametrize("replay_mode", ["spill", None], ids=["spill", "windowed"])
def test_a_measured_tail_longer_than_the_last_grace_raises_that_grace(
        tmp_path, monkeypatch, capsys, replay_mode):
    tail = _ceiling("tail", LONG_TAIL_S)
    dispatch, manifest, argv = _dispatch(
        tmp_path, monkeypatch, capsys, replay_mode=replay_mode,
        ceilings=(*FIXTURE_CEILINGS, tail))
    last = _last_phase(manifest)
    graces, stamps = _graces(argv), _stamps(argv)
    assert graces[last] > LONG_TAIL_S, (
        f"{last}: the tail after the last window runs on this phase's clock "
        f"for {LONG_TAIL_S} s, but its grace is {graces[last]} s")
    by_phase = {stamp.get("phase"): stamp for stamp in stamps}
    stamp = by_phase[last]
    term = _tail_terms(stamps)[last]
    assert (term["mode"], term["units"], term["unit_s"], term["grace_s"]) == (
        "derived", 1, LONG_TAIL_S, math.ceil(LONG_TAIL_S))
    # The phase's own pass is still priced; the tail is added to it.
    assert stamp["grace_s"] == graces[last] == stamp["read"]["grace_s"] + sum(
        item["grace_s"] for item in stamp["compute"])
    assert [item["kind"] for item in stamp["compute"]][-1] == "tail"
    assert len(stamp["compute"]) >= 2, stamp["compute"]
    basis = stamps[-1]
    assert basis["schema"] == dispatch.COMPUTE_GRACE_BASIS_SCHEMA
    assert basis["ceilings"]["tail"]["unit_s"] == LONG_TAIL_S


def test_only_the_last_declared_phase_carries_the_tail(
        tmp_path, monkeypatch, capsys):
    _dispatch_module, manifest, argv = _dispatch(
        tmp_path, monkeypatch, capsys, replay_mode="spill",
        ceilings=(*FIXTURE_CEILINGS, _ceiling("tail", 7.0)))
    assert list(_tail_terms(_stamps(argv))) == [_last_phase(manifest)]


def test_an_unmeasured_tail_takes_the_blanket_and_says_what_will_measure_it(
        tmp_path, monkeypatch, capsys):
    dispatch, manifest, argv = _dispatch(
        tmp_path, monkeypatch, capsys, replay_mode="spill",
        ceilings=FIXTURE_CEILINGS)
    last = _last_phase(manifest)
    term = _tail_terms(_stamps(argv))[last]
    assert (term["mode"], term["grace_s"]) == (
        "blanket", dispatch.HEAD_PROGRESS_GRACE_S)
    assert "payload" in term["reason"] and "teardown" in term["reason"]
    plan = _plan(manifest)
    # Whatever the phase's own pass costs, the tail adds its blanket.
    assert _graces(argv)[last] >= (_read_term(dispatch, plan[last]["bytes"])
                                   + dispatch.HEAD_PROGRESS_GRACE_S)


def test_a_tail_measured_outside_the_row_takes_the_blanket(
        tmp_path, monkeypatch, capsys):
    """A producer's tail writes the band-serial handoff; a measurement taken
    on rows that write none does not cover it, and a scope can say so."""
    tail = _ceiling("tail", 7.0, equal={"emits_handoff": True})
    _dispatch_module, manifest, argv = _dispatch(
        tmp_path, monkeypatch, capsys, replay_mode="spill",
        ceilings=(*FIXTURE_CEILINGS, tail))
    term = _tail_terms(_stamps(argv))[_last_phase(manifest)]
    assert term["mode"] == "blanket"
    assert "emits_handoff is False, measured at True" in term["reason"]


# -- the derivation, pinned ------------------------------------------------


def test_the_tail_is_the_last_term_of_the_phase_it_runs_under():
    import dispatch_joint_quanta as dispatch

    work = dispatch.compute_phase_work
    common = {"entries": 512, "n_probes": 4, "capture_batch": 4}
    replay, tail = work("render-14", replay_mode="spill", runs_tail=True,
                        **common)
    assert (replay["kind"], tail["kind"], tail["units"]) == (
        "spill-replay", "tail", 1)
    assert [term["kind"] for term in work(
        "spill-p3", replay_mode="spill", runs_tail=True, **common)] == [
        "spill-capture", "spill-replay", "tail"]
    # A last phase with no pass of its own still carries the tail.
    (only,) = work("render-00", replay_mode="spill", runs_tail=True, **common)
    assert only["kind"] == "tail"
    assert work("render-00", replay_mode="spill", **common) is None
    assert "tail" in dispatch.COMPUTE_KINDS
    assert "tail" in dispatch.COMPUTE_PHASE_BOUND


def test_a_tail_ceiling_document_is_accepted_by_file(tmp_path, monkeypatch,
                                                      capsys):
    import json

    path = tmp_path / "tail-ceiling.json"
    path.write_text(json.dumps(_ceiling("tail", 11.5)))
    _dispatch_module, manifest, argv = _dispatch(
        tmp_path, monkeypatch, capsys, replay_mode="spill",
        ceilings=FIXTURE_CEILINGS, extra=("--compute-ceiling", str(path)))
    stamps = _stamps(argv)
    term = _tail_terms(stamps)[_last_phase(manifest)]
    assert (term["mode"], term["grace_s"]) == ("derived", 12)
    assert stamps[-1]["ceilings"]["tail"]["document"]["path"] == str(path)


# -- the runtime: the tail runs where the dispatcher prices it --------------


def test_the_quantum_runs_its_tail_under_the_last_declared_phase(
        tmp_path, monkeypatch):
    """The real quantum, spill mode: the ``payload`` span opens with the
    last declared phase in effect, and nothing commits after it opens."""
    import test_quantum_executable_readset as phases
    import test_stageb_one_pass_spill as one_pass
    import prismaquant.io_spans as io_spans
    import prismaquant.joint_cost_quantum as qc
    import prismaquant.joint_replay_spill as spill_mod

    monkeypatch.setattr(phases, "_expert_fixture", one_pass._bf16_expert_fixture)
    for name in spill_mod.SPILL_ENV:
        monkeypatch.delenv(name, raising=False)
    setup = phases._acceptance_setup_expert(tmp_path, monkeypatch)
    spill_root = one_pass._spill_root(tmp_path)
    monkeypatch.setenv(spill_mod.SPILL_ENV[0], str(spill_root))
    monkeypatch.setenv(spill_mod.SPILL_ENV[1], str(1 << 30))

    log = []
    original_enter = qc.QuantumProgress._enter
    original_commit = qc.QuantumProgress.commit
    original_open = io_spans.IoSpanLog.open

    def logging_enter(self, name):
        original_enter(self, name)
        log.append(("phase", self._phase))

    def logging_commit(self):
        advanced = original_commit(self)
        if advanced:
            log.append(("report", self._phase))
        return advanced

    def logging_open(self, name, **attrs):
        log.append(("open", name))
        return original_open(self, name, **attrs)

    monkeypatch.setattr(qc.QuantumProgress, "_enter", logging_enter)
    monkeypatch.setattr(qc.QuantumProgress, "commit", logging_commit)
    monkeypatch.setattr(io_spans.IoSpanLog, "open", logging_open)
    _events, manifest, _payload, _resolved = phases._drive_quantum(
        tmp_path, monkeypatch, setup, layer=0, resume=False,
        replay_mode="spill")

    opened = log.index(("open", "payload"))
    in_effect = next(event[1] for event in reversed(log[:opened])
                     if event[0] == "phase")
    assert in_effect == _last_phase(manifest)
    assert not [event for event in log[opened:] if event[0] == "report"]
