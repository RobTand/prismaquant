"""Measured-unit completion exercises real rows, journal bytes and resume gates."""
from __future__ import annotations

import copy

import pytest

from prismaquant import aura_checkpoint_journal as journal
from prismaquant.cost_stage_checkpoint import unit_path
from prismaquant.joint_price_step import (
    AuraUnitMeasurements, JointPriceStep, JointResumeContract, ProbeResumePolicy,
)
from test_joint_aura_assignment_diagnostics import UNIT_A, _row

RECORDED = [0.09054819587618113, -0.022860320983454585]
FORMAT = "FP8_E4M3"


def _step(*, policy=ProbeResumePolicy.EXACT, probe_changes=None):
    row = _row(UNIT_A, RECORDED)
    probe = copy.deepcopy(row["probe_identity"])
    if probe_changes:
        probe.update(probe_changes)
    measurements = AuraUnitMeasurements(n_probes=2, render_formats={UNIT_A: [FORMAT]})
    step = JointPriceStep(measurements, probe_identity=probe,
        formats_by_qname={UNIT_A: [FORMAT]}, zero_cost_formats=(),
        rendered_identities={UNIT_A: {FORMAT: row["joint_operator_identity"]["rendered_weight"]}},
        resume_contract=JointResumeContract(policy))
    step.operators[UNIT_A, FORMAT] = copy.deepcopy(row["joint_operator_identity"])
    step.components[UNIT_A, FORMAT] = []
    measurements.s2[UNIT_A, FORMAT] = measurements.s4[UNIT_A, FORMAT] = 0.0
    measurements.x2_probe[UNIT_A, FORMAT] = []
    measurements.dw_src[UNIT_A, FORMAT] = "rendered"
    measurements.g_trace[UNIT_A] = 0.0
    return step, row


def _completed():
    step, row = _step()
    for index, components in enumerate(row["signed_components_per_probe"]):
        step.consume_probe(index, {(UNIT_A, FORMAT): components}, {UNIT_A: {"g_trace": index + 1.0}})
    return step, step.complete_unit(UNIT_A)


def test_real_price_accumulation_preserves_multiply_samples_and_completed_state():
    step, state = _completed()
    expected = [value * value for value in RECORDED]
    assert step.measurements.x2_probe[UNIT_A, FORMAT] == expected
    assert state["rows"][FORMAT]["x2_probe"] == expected
    assert state["rows"][FORMAT]["s2"] == sum(expected)
    assert state["rows"][FORMAT]["s4"] == sum(value * value for value in expected)
    assert state["g_trace"] == 3.0
    row = state["joint_aura_rows"][FORMAT]
    assert row["signed_per_probe"] == RECORDED
    assert row["x2_per_probe"] == expected
    assert row["predicted_dloss"] == 0.5 * sum(expected) / 2


@pytest.mark.parametrize("policy,dev,refuses", [
    (ProbeResumePolicy.EXACT, "1", True),
    (ProbeResumePolicy.EXACT, "0", True),
    (ProbeResumePolicy.RUN_SEALS, "1", False),
    (ProbeResumePolicy.RUN_SEALS, "0", True),
])
def test_resume_retains_exact_and_quantum_run_seal_policies(tmp_path, monkeypatch, policy, dev, refuses):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev)
    _old_step, state = _completed()
    first = journal.AuraCheckpointJournal.open(tmp_path, resume=False,
        identity={"fixture": "row-policy"}, names=[UNIT_A])
    first.publish_state(UNIT_A, state)
    original = unit_path(tmp_path, UNIT_A).read_bytes()
    resumed = journal.AuraCheckpointJournal.open(tmp_path, resume=True,
        identity={"fixture": "row-policy"}, names=[UNIT_A])
    price, _row_value = _step(policy=policy, probe_changes={"producer_source_sha256": "e" * 64})
    if refuses:
        with pytest.raises(RuntimeError, match="probe/operator alignment mismatch"):
            resumed.resume_units(price)
        assert resumed.completed == set()
    else:
        assert resumed.resume_units(price) == {UNIT_A}
        assert price.measurements.x2_probe[UNIT_A, FORMAT] == [value * value for value in RECORDED]
    assert unit_path(tmp_path, UNIT_A).read_bytes() == original


@pytest.mark.parametrize("dev", ["1", "0"])
def test_quantum_resume_refuses_calibration_wall_before_completion(tmp_path, monkeypatch, dev):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev)
    _step_value, state = _completed()
    first = journal.AuraCheckpointJournal.open(tmp_path, resume=False,
        identity={"fixture": "wall"}, names=[UNIT_A])
    first.publish_state(UNIT_A, state)
    resumed = journal.AuraCheckpointJournal.open(tmp_path, resume=True,
        identity={"fixture": "wall"}, names=[UNIT_A])
    price, _row_value = _step(policy=ProbeResumePolicy.RUN_SEALS,
                            probe_changes={"calibration_sha256": "f" * 64})
    with pytest.raises(RuntimeError, match="probe/operator alignment mismatch"):
        resumed.resume_units(price)
    assert resumed.completed == set()


@pytest.mark.parametrize("require_activation", [False, True])
def test_full_resume_preserves_optional_cache_activation_binding(tmp_path, require_activation):
    _old_step, state = _completed()
    first = journal.AuraCheckpointJournal.open(tmp_path, resume=False,
        identity={"fixture": "optional-cache"}, names=[UNIT_A])
    first.publish_state(UNIT_A, state)
    resumed = journal.AuraCheckpointJournal.open(tmp_path, resume=True,
        identity={"fixture": "optional-cache"}, names=[UNIT_A])
    price, row = _step()
    expected = copy.deepcopy(row["joint_operator_identity"]["activation"])
    expected["fixture_change"] = "another cache contract"
    price.resume_contract = JointResumeContract(ProbeResumePolicy.EXACT,
        activation_identities={UNIT_A: {FORMAT: expected}} if require_activation else None)
    if require_activation:
        with pytest.raises(RuntimeError, match="activation identity mismatch"):
            resumed.resume_units(price)
        assert resumed.completed == set()
    else:
        assert resumed.resume_units(price) == {UNIT_A}


def test_group_publication_does_not_acknowledge_a_partial_write(tmp_path, monkeypatch):
    step, state = _completed()
    first = journal.AuraCheckpointJournal.open(tmp_path, resume=False,
        identity={"fixture": "group"}, names=[UNIT_A, "second"])
    atomic = journal.atomic_write_bytes
    def refuse_second(path, data):
        if path == unit_path(tmp_path, "second"):
            raise OSError("second unit failed")
        atomic(path, data)
    monkeypatch.setattr(journal, "atomic_write_bytes", refuse_second)
    class Prices:
        def captured_state(self, name):
            return state
    with pytest.raises(OSError, match="second unit failed"):
        first.publish_group(Prices(), [UNIT_A, "second"])
    assert unit_path(tmp_path, UNIT_A).is_file()
    assert not unit_path(tmp_path, "second").exists()
    assert first.completed == set()
    resumed = journal.AuraCheckpointJournal.open(tmp_path, resume=True,
        identity={"fixture": "group"}, names=[UNIT_A, "second"])
    assert resumed.resume_units(step) == {UNIT_A}
