"""Wire the served backend into the task-suite config path."""
import pytest
from test_served_task_backend import (
    make_expected, make_witness, served_config, write_json)
from prismaquant import task_suite
from prismaquant.served_task_backend import is_served_config

SAMPLING = {"limit": 1, "num_fewshot": 0, "random_seed": 0,
            "numpy_seed": 0, "torch_seed": 0, "fewshot_seed": 0}
TASKS = ["arc_easy"]


def _config(tmp_path):
    witness = make_witness()
    witness_path = write_json(tmp_path / "witness.json", witness)
    expected_path = write_json(tmp_path / "expected.json",
                               make_expected(witness))
    base = served_config(witness_path, expected_path, witness)
    return {"backend": base["backend"], "sampling": dict(SAMPLING),
            "tasks": list(TASKS)}


def test_served_config_validates(tmp_path):
    config = _config(tmp_path)
    assert is_served_config(config)
    assert task_suite.validate_config(config) is config


def test_served_config_without_binding_refuses():
    config = {"backend": {"name": "served"}, "sampling": dict(SAMPLING),
              "tasks": list(TASKS)}
    with pytest.raises(ValueError, match="serving_runtime"):
        task_suite.validate_config(config)


def test_served_preflight_binds_without_hf(tmp_path):
    config = _config(tmp_path)
    record = task_suite.preflight_tasks(config)
    assert record["binding"]["verdict"] == "pass"
    assert record["binding"]["served_alias"] == "fixture"


def test_served_preflight_refuses_bad_witness(tmp_path):
    config = _config(tmp_path)
    config["backend"]["serving_runtime"]["served_alias"] = "other-alias"
    with pytest.raises(ValueError, match="differ|refused"):
        task_suite.preflight_tasks(config)


def test_served_measure_waits_for_parent_scope(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="2430"):
        task_suite.measure_tasks(config, tmp_path / "out.json")


def test_hf_backend_still_validates():
    config = {"backend": {"name": "hf", "pretrained": "/model",
                          "tokenizer": "/model", "device": "cpu",
                          "dtype": "float32", "batch_size": 1,
                          "max_length": 128, "trust_remote_code": False},
              "tasks": list(TASKS), "sampling": dict(SAMPLING)}
    assert task_suite.validate_config(config)["backend"]["name"] == "hf"


def _verifier_setup(tmp_path):
    from test_served_task_public_verifier import STUB
    cli = tmp_path / "stub_cli.py"
    cli.write_text(STUB)
    served = tmp_path / "served"
    served.mkdir()
    return cli, served


def test_served_preflight_runs_public_verifier(tmp_path):
    config = _config(tmp_path)
    cli, served = _verifier_setup(tmp_path)
    runtime = config["backend"]["serving_runtime"]
    runtime["verifier"] = str(cli)
    runtime["served_dir"] = str(served)
    record = task_suite.preflight_tasks(config)
    assert record["binding"]["public_verifier"]["verdict"] == "valid"


def test_served_preflight_names_missing_verifier_dir(tmp_path):
    config = _config(tmp_path)
    runtime = config["backend"]["serving_runtime"]
    runtime["verifier"] = str(tmp_path / "stub_cli.py")
    with pytest.raises(ValueError, match="served_dir"):
        task_suite.preflight_tasks(config)


def test_served_preflight_marks_absent_verifier(tmp_path):
    record = task_suite.preflight_tasks(_config(tmp_path))
    assert record["binding"]["public_verifier"] == "not_configured"
