"""New measurement context strengthens the existing optional control comparison."""
import copy

import pytest

from experiments import measure_glm_tr3_vllm as served
from prismaquant import shipcard
from test_glm_tr3_gold_bridge import _completed_result


def _arms():
    candidate = served._tr3_gold_record(
        _completed_result(), model_sha="d" * 64, model="candidate",
        spec_decode_detected=False)
    control = copy.deepcopy(candidate)
    control["model_sha"] = "e" * 64
    control["metrics"]["kl_mean"] = 0.25
    return candidate, control


def _problems(candidate, control):
    return shipcard._replay_control_arms(
        "uniform_control", {"metrics": {"gold_metric_key": "kl_mean"}, "control_arm": control},
        {"candidate": 0.125, "control": 0.25},
        card={"model_sha": "d" * 64, "slots": {"gold.kl": candidate}})


@pytest.mark.parametrize("field", ["measurement_fidelity", "calibration_contract_sha256",
                                  "teacher_evidence"])
@pytest.mark.parametrize("damage", ["missing", "different"])
def test_new_candidate_contract_requires_the_same_control_context(field, damage):
    candidate, control = _arms()
    if damage == "missing":
        control["metrics"].pop(field)
    else:
        control["metrics"][field] = "different"
    assert any(field in problem for problem in _problems(candidate, control))


def test_equal_actual_producer_contexts_pass_the_existing_comparison():
    assert _problems(*_arms()) == []


@pytest.mark.parametrize("damage", ["numeric_boolean", "nonfinite"])
def test_exact_context_comparison_is_typed_and_finite(damage):
    candidate, control = _arms()
    control["metrics"]["measurement_fidelity"]["tail_bucket"] = (
        0 if damage == "numeric_boolean" else float("nan"))
    assert any("measurement_fidelity" in problem for problem in _problems(candidate, control))


def test_absence_does_not_match_a_present_null_candidate_field():
    candidate, control = _arms()
    candidate["metrics"]["teacher_evidence"] = None
    control["metrics"].pop("teacher_evidence")
    assert any("teacher_evidence" in problem for problem in _problems(candidate, control))


def test_legacy_candidate_without_new_fields_keeps_its_original_comparison():
    candidate, control = _arms()
    for field in ("measurement_fidelity", "calibration_contract_sha256", "teacher_evidence"):
        candidate["metrics"].pop(field)
        control["metrics"].pop(field)
    assert _problems(candidate, control) == []
