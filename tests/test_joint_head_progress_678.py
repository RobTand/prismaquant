"""A joint prepare's head phase reports the work it walks (#678).

The anchor intake resolves every cell's render origin and authenticates every
unit's journal shard. It used to report only the shards it had to *write*, so a
resume -- where every render is already durable -- walked the whole roster in
silence and PrismaBuild's watchdog reaped an action that was working.

The fixture is the campaign intake fixture of ``test_tessera_joint_aura``: two
units, one measured rung each, both renders already on disk. That is the resume
shape in miniature.
"""
from __future__ import annotations

import pytest

from tests.test_tessera_joint_aura import fixture


def _reports(monkeypatch, bridge):
    seen = []
    monkeypatch.setattr(bridge, "_pb_commit",
                        lambda units, phase, unit=None: seen.append((units, phase, unit)))
    return seen


def test_a_walk_that_synthesizes_nothing_still_reports_its_units(tmp_path, monkeypatch):
    """The defect: present renders, real work, no committed units."""
    from prismaquant import tessera_joint_aura as bridge

    config, names, _fmt, _payload, _states = fixture(tmp_path)
    seen = _reports(monkeypatch, bridge)
    data = bridge.load_measured_anchor_input(config, verify_payloads=False)

    assert data.synthesized_now == 0, "the fixture's renders are already durable"
    # One cumulative report per resolved unit, in roster order, after that
    # unit's own reads -- never a count the walk has not reached.
    assert [count for count, _, _ in seen] == [1, 2]
    assert [unit for _, _, unit in seen] == sorted(names)
    assert data.progress_committed == 2


def test_the_caller_names_the_phase_and_can_decline_to_report(tmp_path, monkeypatch):
    """A name outside the submission's declared set grants no continuation, so
    the loader reports under the phase its caller declares and nothing else."""
    from prismaquant import tessera_joint_aura as bridge
    from prismaquant.joint_prewarm_phases import HEAD_PHASE

    config, _names, _fmt, _payload, _states = fixture(tmp_path)
    seen = _reports(monkeypatch, bridge)

    head = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                             progress_phase=HEAD_PHASE)
    assert {phase for _, phase, _ in seen} == {"head"}
    assert head.progress_committed == 2

    seen.clear()
    silent = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                               progress_phase=None)
    assert seen == [] and silent.progress_committed == 0

    with pytest.raises(ValueError, match="progress_phase"):
        bridge.load_measured_anchor_input(config, verify_payloads=False, progress_phase="")

# ``execute`` is the seam that hands the loader the phase its submission
# declared. That it passes ``head`` for a prepare and nothing for a COST run is
# asserted where both commands are already driven:
# ``tests/test_tessera_joint_aura.py::test_explicit_source_prefetch_reaches_streamed_builder``.
