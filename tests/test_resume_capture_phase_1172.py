"""A resumed spill row captures under its spill phases (PQ #1172).

A spill-sealed plan orders ``render-00``, then one ``spill-pP`` phase per
probe, then ``render-01`` and later (``quantum_executable_phase_names``). The
dispatcher prices each ``spill-pP`` with one probe's capture and each later
``render-NN`` with that window's spill replay, and no capture term (#1165).

A resume whose first active window is k >= 1 captures every probe again,
because the spill scratch is an ``O_TMPFILE``. If those captures run after
``render-k`` is entered, they run under a phase whose grace has no capture
term. Progress phases only move forward, so the row cannot report
``spill-pP`` again, and PrismaBuild ends it as ``no_progress`` on every
retry.

Both tests drive the real quantum in spill mode, with one chain layer and
the prepared-render phases a production row seals. Window 0's units are
committed by a first run; the second run resumes at window 1.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tests", ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))


def _resumed_spill_run(tmp_path, monkeypatch):
    """Commit window 0, then resume at window 1 with a unit log.

    Returns the resume's manifest, the resolved windows and the log. The log
    holds ``("report", phase, units)`` for every progress report, and
    ``("unit", phase, kind)`` for every unit of work, where ``phase`` is the
    phase reported last before the unit ran: a chain-roll row, a spill
    capture group, or a spill replay of one (window, probe).
    """
    import test_quantum_executable_readset as phases
    import test_stageb_one_pass_spill as one_pass
    import prismaquant.joint_adjoint_checkpoints as checkpoints
    import prismaquant.joint_cost_quantum as qc
    import prismaquant.joint_replay_spill as spill_mod
    from prismaquant.aura_cost import _aura_unit_checkpoint_path

    monkeypatch.setattr(phases, "_expert_fixture", one_pass._bf16_expert_fixture)
    for name in spill_mod.SPILL_ENV:
        monkeypatch.delenv(name, raising=False)
    setup = phases._acceptance_setup_expert(tmp_path, monkeypatch)
    record0 = setup["records"]["layer-000"]
    assert list(record0["adjoint"]["chain_layers"]) == [1]
    spill_root = one_pass._spill_root(tmp_path)
    monkeypatch.setenv(spill_mod.SPILL_ENV[0], str(spill_root))
    monkeypatch.setenv(spill_mod.SPILL_ENV[1], str(1 << 30))

    # The first run commits every unit; deleting windows 1+ leaves window
    # 0 committed, the state a row ended after window 0 resumes from.
    _events, manifest, _payload, resolved = phases._drive_quantum(
        tmp_path, monkeypatch, setup, layer=0, resume=False,
        replay_mode="spill", prepared_render_phases=True)
    assert len(resolved) >= 2, resolved
    names = [phase["name"] for phase in manifest["read_plan"]["phases"]]
    assert "render-00" in names and "render-01" in names, names
    checkpoint_dir = Path(record0["output_space"]["checkpoint_dir"])
    for name in (n for window in resolved[1:] for n in window["names"]):
        _aura_unit_checkpoint_path(checkpoint_dir, name).unlink()

    log = []

    def current():
        return next((event[1] for event in reversed(log)
                     if event[0] == "report"), None)

    original_commit = qc.QuantumProgress.commit

    def logging_commit(self):
        advanced = original_commit(self)
        if advanced:
            log.append(("report", self._phase, self.units()))
        return advanced

    monkeypatch.setattr(qc.QuantumProgress, "commit", logging_commit)
    original_roll = checkpoints.render_free_layer_roll

    def counting_roll(*args, roll, **kwargs):
        def counted(tensor, batch, probe):
            log.append(("unit", current(), "chain-roll"))
            return roll(tensor, batch, probe)
        return original_roll(*args, roll=counted, **kwargs)

    monkeypatch.setattr(checkpoints, "render_free_layer_roll", counting_roll)
    original_end = spill_mod._SpillObserver.end_batch

    def counting_end(self):
        log.append(("unit", current(), "spill-capture"))
        return original_end(self)

    monkeypatch.setattr(spill_mod._SpillObserver, "end_batch", counting_end)
    original_replay = spill_mod.StageBReplaySpill.replay

    def counting_replay(self, window_index, probe_index, lease):
        log.append(("unit", current(), "spill-replay"))
        return original_replay(self, window_index, probe_index, lease)

    monkeypatch.setattr(spill_mod.StageBReplaySpill, "replay", counting_replay)
    _events, manifest_r, payload, _resolved = phases._drive_quantum(
        tmp_path, monkeypatch, setup, layer=0, resume=True,
        replay_mode="spill", prepared_render_phases=True)
    assert payload["costs"]
    assert not os.listdir(spill_root)
    shutil.rmtree(spill_root, ignore_errors=True)
    return manifest_r, resolved, log


def test_a_resumed_spill_row_captures_under_its_spill_phases(
        tmp_path, monkeypatch):
    """Resumed at window 1, each probe's capture runs under ``spill-pP``,
    never under ``render-01``, whose grace prices no capture."""
    manifest, _resolved, log = _resumed_spill_run(tmp_path, monkeypatch)
    n_probes = manifest["annotations"]["n_probes"]
    plan = {phase["name"]: phase for phase in manifest["read_plan"]["phases"]}
    captures = [event[1] for event in log
                if event[0] == "unit" and event[2] == "spill-capture"]
    per_probe = len(plan["spill-p0"]["entry_indices"])
    assert len(captures) == n_probes * per_probe, captures
    expected = [f"spill-p{probe}" for probe in range(n_probes)
                for _group in range(per_probe)]
    assert captures == expected, (
        f"a resume at window 1 captured under {sorted(set(captures))}, not "
        f"under the sealed spill phases {sorted(set(expected))}: render-01's "
        "grace prices no capture, so PrismaBuild ends the row as no_progress")


def test_a_resumed_quantum_runs_only_the_units_its_phases_price(
        tmp_path, monkeypatch):
    """The #1165 runtime pin, extended to a resume at window 1.

    Every unit runs under a phase whose dispatcher pricing names its kind.
    Chain-roll rows and capture groups equal what each phase prices. A
    replay under ``spill-pP`` replays window 0, which the resume has
    committed, so none runs there; window 1's replays equal what
    ``render-01`` prices. Nothing commits between a pass's first and last
    unit.
    """
    import dispatch_joint_quanta as dispatch

    manifest, resolved, log = _resumed_spill_run(tmp_path, monkeypatch)
    plan = {phase["name"]: phase for phase in manifest["read_plan"]["phases"]}
    n_probes = manifest["annotations"]["n_probes"]
    priced = {}
    for name, phase in plan.items():
        for term in dispatch.compute_phase_work(
                name, replay_mode="spill",
                entries=len(phase["entry_indices"]), n_probes=n_probes,
                capture_batch=1) or ():
            priced[(name, term["kind"])] = term["units"]
    ran = {}
    for event in log:
        if event[0] == "unit":
            ran[(event[1], event[2])] = ran.get((event[1], event[2]), 0) + 1
    unpriced = sorted(set(ran) - set(priced))
    assert not unpriced, (
        f"work ran under a phase that does not price it: "
        f"{[(key, ran[key]) for key in unpriced]}")
    first_active = 1
    expected = {}
    for (name, kind), units in priced.items():
        if kind == "spill-replay" and name.startswith("spill-p"):
            continue  # window 0 is committed: nothing to replay
        if kind == "spill-replay" and int(name.split("-")[1]) < first_active:
            continue
        expected[(name, kind)] = units
    assert ran == expected, (ran, expected)
    assert len(resolved) - first_active == sum(
        1 for name, kind in ran if kind == "spill-replay")
    for key in ran:
        indices = [i for i, event in enumerate(log)
                   if event[0] == "unit" and (event[1], event[2]) == key]
        inside = [log[i] for i in range(indices[0], indices[-1])
                  if log[i][0] == "report"]
        assert inside == [], (key, inside)
