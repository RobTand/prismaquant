"""A task run must declare its backend, sample policy and tokenizer."""
import pytest
from prismaquant.task_suite import validate_config, task_metrics


def config():
    return {"schema": "prismaquant.task_suite/1", "backend": {"name": "hf", "pretrained": "/model",
        "tokenizer": "/model", "device": "cpu", "dtype": "float32", "batch_size": 1,
        "max_length": 128, "trust_remote_code": False}, "tasks": ["arc_easy"],
        "sampling": {"limit": 1, "num_fewshot": 0, "random_seed": 0, "numpy_seed": 0,
                     "torch_seed": 0, "fewshot_seed": 0}, "criteria": []}


def test_task_inputs_are_explicit():
    assert validate_config(config())["backend"]["name"] == "hf"


@pytest.mark.parametrize("field", ["tokenizer", "device", "dtype", "pretrained", "batch_size"])
def test_backend_refuses_missing_fields(field):
    c = config()
    del c["backend"][field]
    with pytest.raises(ValueError):
        validate_config(c)


@pytest.mark.parametrize("field", ["limit", "num_fewshot", "torch_seed", "fewshot_seed"])
def test_sample_policy_refuses_missing_fields(field):
    c = config()
    del c["sampling"][field]
    with pytest.raises(ValueError):
        validate_config(c)


def test_task_metrics_ignore_errors_not_metrics():
    assert task_metrics({"results": {"arc_easy": {"acc,none": 0.5, "acc_stderr,none": 0.1,
                                                  "alias": "ARC"}}}) == {"arc_easy/acc,none": 0.5}


def test_empty_task_population_refuses():
    with pytest.raises(ValueError):
        task_metrics({"results": {}})
