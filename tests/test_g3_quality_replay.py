"""G3 replay verifies owned numeric data in both execution modes."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from prismaquant.g3_v2 import measure_g3
from prismaquant.quality_stage import artifact, evaluate_criteria, verify_result
from test_g3_v2 import make_retained_configuration


def make_g3_replay_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    config = make_retained_configuration(root)
    facts = measure_g3(config, root / "fixture.json")
    teacher = json.loads(Path(config["teacher"]["path"]).read_text())
    candidate = json.loads(Path(config["candidate"]["receipt"]["path"]).read_text())
    agreements = np.stack([np.load(left["path"]).argmax(-1) == np.load(right["path"]).argmax(-1)
                          for left, right in zip(teacher["arrays"], candidate["arrays"], strict=True)])
    agreement_path = root / "fixture.agreement.npy"
    np.save(agreement_path, agreements, allow_pickle=False)
    facts["artifacts"] = [facts["artifacts"][0], artifact(agreement_path)]
    facts["identity"]["panel"] = config["panel"]
    config["criteria"] = [{"metric": "mean_kl", "op": "le", "threshold": 1.0}]
    result = {"schema": "prismaquant.quality_stage/1", "stage": "g3_v2",
        "configuration": {"schema": config["schema"], "sha256": "a" * 64, "path": str(root / "config.json")},
        "measurement": {"status": "succeeded", "metric_kind": "offline_decoded_kl",
                        "metrics": facts.pop("metrics"), "error": None}, **facts}
    result["gate"] = evaluate_criteria(result["measurement"]["metrics"], config["criteria"])
    return config, json.loads(json.dumps(result)), Path(result["artifacts"][0]["path"]), agreement_path


@pytest.fixture
def g3_replay_case(tmp_path):
    return make_g3_replay_fixture(tmp_path)


@pytest.mark.parametrize("dev", [False, True])
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_g3_owned_kl_array_refuses_damage_in_both_modes(g3_replay_case, monkeypatch, dev, damage):
    config, result, kl, _ = g3_replay_case
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    kl.unlink() if damage == "missing" else kl.write_bytes(b"corrupt KL array")
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_result(result, config, config_sha256="a" * 64)


@pytest.mark.parametrize("dev", [False, True])
@pytest.mark.parametrize("metric", ["mean_kl", "p99_kl", "max_kl", "top1_agreement"])
def test_g3_edited_metric_and_consistent_gate_still_refuse(g3_replay_case, monkeypatch, dev, metric):
    config, result, _, _ = g3_replay_case
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    config["criteria"] = [{"metric": metric, "op": "le", "threshold": 2.0}]
    result["measurement"]["metrics"][metric] = 0.75
    result["gate"] = evaluate_criteria(result["measurement"]["metrics"], config["criteria"])
    with pytest.raises(ValueError, match="metric"):
        verify_result(result, config, config_sha256="a" * 64)


@pytest.mark.parametrize("dev", [False, True])
@pytest.mark.parametrize("field", ["positions", "windows", "window_ids", "per_window"])
def test_g3_population_comes_from_owned_arrays(g3_replay_case, monkeypatch, dev, field):
    config, result, _, _ = g3_replay_case
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    if field in ("positions", "windows"):
        result["population"][field] += 1
    elif field == "window_ids":
        result["population"][field].reverse()
    else:
        result["population"][field][0]["mean_kl"] = 0.75
    with pytest.raises(ValueError, match="population|window"):
        verify_result(result, config, config_sha256="a" * 64)


@pytest.mark.parametrize("dev", [False, True])
def test_g3_owned_agreement_array_is_not_an_unchecked_gate_metric(g3_replay_case, monkeypatch, dev):
    config, result, _, agreement = g3_replay_case
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    agreement.write_bytes(b"corrupt agreement array")
    with pytest.raises(ValueError):
        verify_result(result, config, config_sha256="a" * 64)


