"""Quality receipts separate measurements from configured decisions."""
import copy
import pytest
from prismaquant.quality_stage import evaluate_criteria, verify_result, g3_candidate_binding


def test_missing_criteria_is_not_a_pass():
    assert evaluate_criteria({"mean_kl": 0.0}, None)["status"] == "not_evaluated"


@pytest.mark.parametrize("criteria", [[{"metric": "mean_kl", "op": "le", "threshold": 0.1}],
    [{"metric": "accuracy", "op": "ge", "threshold": 0.7}]])
def test_decision_uses_measured_values(criteria):
    metric = criteria[0]["metric"]
    assert evaluate_criteria({metric: criteria[0]["threshold"]}, criteria)["status"] == "passed"
    value = 0.2 if metric == "mean_kl" else 0.6
    assert evaluate_criteria({metric: value}, criteria)["status"] == "failed"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0"])
def test_criteria_reject_invalid_numbers(bad):
    with pytest.raises(ValueError):
        evaluate_criteria({"x": bad}, [{"metric": "x", "op": "le", "threshold": 1}])


def test_missing_metric_refuses():
    with pytest.raises(ValueError, match="missing"):
        evaluate_criteria({}, [{"metric": "x", "op": "le", "threshold": 1}])


def test_verifier_replays_gate_and_config_identity():
    config = {"schema": "prismaquant.g3_v2/1", "candidate": {"backend": "retained_logits",
        "receipt": {"path": "candidate.json", "sha256": "c"*64}},
        "criteria": [{"metric": "mean_kl", "op": "le", "threshold": 0.1}]}
    result = {"schema": "prismaquant.quality_stage/1", "stage": "g3_v2",
        "configuration": {"schema": config["schema"], "sha256": "a"*64, "path": "config.json"},
        "measurement": {"status": "succeeded", "metric_kind": "offline_decoded_kl",
                        "metrics": {"mean_kl": 0.2}, "error": None},
        "gate": evaluate_criteria({"mean_kl": 0.2}, config["criteria"]),
        "identity": {"candidate_inputs": g3_candidate_binding(config)},
        "population": {}, "artifacts": [], "limitations": []}
    assert verify_result(result, config, config_sha256="a"*64)["status"] == "failed"
    changed = copy.deepcopy(result)
    changed["gate"]["status"] = "passed"
    with pytest.raises(ValueError, match="gate"):
        verify_result(changed, config, config_sha256="a"*64)
    with pytest.raises(ValueError, match="configuration"):
        verify_result(result, config, config_sha256="b"*64)
    changed = copy.deepcopy(result)
    changed["identity"]["candidate_inputs"]["receipt"]["sha256"] = "d"*64
    with pytest.raises(ValueError, match="candidate"):
        verify_result(changed, config, config_sha256="a"*64)
