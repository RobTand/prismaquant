"""Task replay preserves dev provenance and verifies owned result bytes."""
import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.cost_streaming import build_source_checkpoint_identity
from prismaquant.quality_stage import artifact, evaluate_criteria, verify_result


def make_task_replay_fixture(root):
    model, tokenizer = root / "model", root / "tokenizer"
    model.mkdir(parents=True)
    tokenizer.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 4}))
    save_file({"weight": torch.arange(8, dtype=torch.float32).reshape(2, 4)}, str(model / "model.safetensors"))
    (tokenizer / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"fixture": 0}}}))
    config = {"schema": "prismaquant.task_suite/1", "backend": {"name": "hf", "pretrained": str(model),
        "tokenizer": str(tokenizer), "device": "cpu", "dtype": "float32", "batch_size": 1,
        "max_length": 128, "trust_remote_code": False}, "tasks": ["facts"],
        "sampling": {"limit": 2, "num_fewshot": 0, "random_seed": 0, "numpy_seed": 0,
                     "torch_seed": 0, "fewshot_seed": 0},
        "criteria": [{"metric": "facts/acc,none", "op": "ge", "threshold": 0.5}]}
    raw = {"results": {"facts": {"name": "facts", "alias": "Facts", "sample_len": 2, "acc,none": 1.0}},
        "config": {"limit": 2, "random_seed": 0, "numpy_seed": 0, "torch_seed": 0, "fewshot_seed": 0},
        "n-samples": {"facts": {"original": 2, "effective": 2}}, "n-shot": {"facts": 0},
        "versions": {"facts": "fixture"}, "configs": {"facts": {"task": "facts", "num_fewshot": 0}}}
    raw_path = root / "raw.lm_eval.json"
    raw_path.write_text(json.dumps(raw))
    metrics = {"facts/acc,none": 1.0}
    result = {"schema": "prismaquant.quality_stage/1", "stage": "task_suite",
        "configuration": {"schema": config["schema"], "sha256": "a" * 64, "path": str(root / "config.json")},
        "measurement": {"status": "succeeded", "metric_kind": "task_metrics", "metrics": metrics, "error": None},
        "gate": evaluate_criteria(metrics, config["criteria"]),
        "identity": {"model": str(model), "tokenizer": str(tokenizer),
            "model_artifact": build_source_checkpoint_identity(model),
            "tokenizer_files": [artifact(tokenizer / "tokenizer.json")], "tokenizer_commit": None,
            "task_versions": raw["versions"], "task_configs": raw["configs"]},
        "population": {"device": "cpu", "tasks": ["facts"], "samples": raw["n-samples"],
                       "sampling": config["sampling"], "skips": []},
        "artifacts": [artifact(raw_path)], "limitations": []}
    return config, result, raw_path


def replace_task_fixture(config, kind):
    if kind == "config":
        Path(config["backend"]["pretrained"], "config.json").write_text(
            json.dumps({"model_type": "llama", "hidden_size": 8}))
    elif kind == "weight":
        path = Path(config["backend"]["pretrained"], "model.safetensors")
        save_file({"weight": torch.ones((2, 4), dtype=torch.float32)}, str(path))
    else:
        Path(config["backend"]["tokenizer"], "tokenizer.json").write_text(
            json.dumps({"model": {"vocab": {"replacement": 0}}}))


@pytest.fixture
def replay_case(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    return make_task_replay_fixture(tmp_path)


@pytest.mark.parametrize("kind", ["config", "weight", "tokenizer"])
def test_certified_fixture_replay_detects_same_path_replacement(replay_case, kind):
    config, result, _ = replay_case
    replace_task_fixture(config, kind)
    with pytest.raises(ValueError, match="provenance"):
        verify_result(result, config, config_sha256="a" * 64)


@pytest.mark.parametrize("kind", ["config", "weight", "tokenizer"])
def test_default_dev_replay_retains_metrics_without_checkpoint_rehash(replay_case, kind, monkeypatch, capsys):
    config, result, _ = replay_case
    replace_task_fixture(config, kind)
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE")
    import prismaquant.cost_streaming as source_owner
    def refuse_rehash(*args, **kwargs):
        raise AssertionError("dev replay must not rehash checkpoint data")
    monkeypatch.setattr(source_owner, "build_source_checkpoint_identity", refuse_rehash)
    stored = copy.deepcopy(result["measurement"])
    assert verify_result(result, config, config_sha256="a" * 64)["status"] == "passed"
    assert result["measurement"] == stored
    assert result["dev_uncertified"] is True
    assert "[DEV-MODE]" in capsys.readouterr().out


@pytest.mark.parametrize("dev", [False, True])
@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_owned_raw_result_bytes_refuse_in_both_modes(replay_case, monkeypatch, dev, damage):
    config, result, path = replay_case
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"corrupt owned result")
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_result(result, config, config_sha256="a" * 64)


@pytest.mark.parametrize("dev", [False, True])
@pytest.mark.parametrize("field", ["metrics", "sampling"])
def test_task_replay_checks_raw_math_even_with_a_new_own_digest(replay_case, monkeypatch, dev, field):
    config, result, path = replay_case
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    raw = json.loads(path.read_text())
    if field == "metrics":
        raw["results"]["facts"]["acc,none"] = 0.0
    else:
        raw["config"]["torch_seed"] = 1
    path.write_text(json.dumps(raw))
    result["artifacts"] = [artifact(path)]
    with pytest.raises(ValueError, match="metrics|sampling"):
        verify_result(result, config, config_sha256="a" * 64)


@pytest.mark.parametrize("dev", [False, True])
def test_task_replay_rejects_boolean_seed(replay_case, monkeypatch, dev):
    config, result, path = replay_case
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    raw = json.loads(path.read_text())
    raw["config"]["torch_seed"] = False
    path.write_text(json.dumps(raw))
    result["artifacts"] = [artifact(path)]
    with pytest.raises(ValueError, match="sampling"):
        verify_result(result, config, config_sha256="a" * 64)


@pytest.mark.parametrize("dev", [False, True])
def test_task_replay_rejects_boolean_effective_count(replay_case, monkeypatch, dev):
    config, result, path = replay_case
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    raw = json.loads(path.read_text())
    raw["n-samples"]["facts"]["effective"] = True
    path.write_text(json.dumps(raw))
    result["artifacts"] = [artifact(path)]
    result["population"]["samples"] = raw["n-samples"]
    with pytest.raises(ValueError, match="task|population"):
        verify_result(result, config, config_sha256="a" * 64)


def test_certified_fixture_replay_ignores_unrelated_card_json(replay_case):
    config, result, _ = replay_case
    Path(config["backend"]["tokenizer"], "shipcard.json").write_text(json.dumps({"status": "updated"}))
    assert verify_result(result, config, config_sha256="a" * 64)["status"] == "passed"
