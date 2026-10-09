"""Quality receipts separate measurements from configured decisions."""
import copy
import pytest
from prismaquant.quality_stage import evaluate_criteria, verify_result


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


def test_verifier_replays_gate_and_config_identity(tmp_path, monkeypatch):
    from test_g3_quality_replay import make_g3_replay_fixture
    config, result, _, _ = make_g3_replay_fixture(tmp_path)
    config["criteria"] = [{"metric": "mean_kl", "op": "le", "threshold": 0.0}]
    result["gate"] = evaluate_criteria(result["measurement"]["metrics"], config["criteria"])
    assert verify_result(result, config, config_sha256="a" * 64)["status"] == "failed"
    changed = copy.deepcopy(result)
    changed["gate"]["status"] = "passed"
    with pytest.raises(ValueError, match="gate"):
        verify_result(changed, config, config_sha256="a" * 64)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(ValueError, match="configuration"):
        verify_result(result, config, config_sha256="b" * 64)
    changed = copy.deepcopy(result)
    changed["identity"]["candidate_inputs"]["receipt"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="candidate"):
        verify_result(changed, config, config_sha256="a" * 64)

@pytest.mark.parametrize("kind", ["configuration", "candidate"])
def test_default_provenance_drift_stamps_and_retains_owned_g3(owned_g3_case, monkeypatch, capsys, kind):
    config, result, _, _ = owned_g3_case
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    stored = copy.deepcopy(result["measurement"])
    if kind == "candidate":
        config["candidate"]["receipt"]["sha256"] = "d" * 64
    gate = verify_result(result, config, config_sha256="b" * 64 if kind == "configuration" else "a" * 64)
    assert gate["status"] == "passed"
    assert result["measurement"] == stored
    assert result["dev_uncertified"] is True
    assert "[DEV-MODE]" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["configuration", "candidate"])
def test_certified_fixture_provenance_drift_refuses(owned_g3_case, monkeypatch, kind):
    config, result, _, _ = owned_g3_case
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    if kind == "candidate":
        config["candidate"]["receipt"]["sha256"] = "d" * 64
    with pytest.raises(ValueError):
        verify_result(result, config, config_sha256="b" * 64 if kind == "configuration" else "a" * 64)


@pytest.fixture
def owned_g3_case(tmp_path):
    from test_g3_quality_replay import make_g3_replay_fixture
    return make_g3_replay_fixture(tmp_path)

