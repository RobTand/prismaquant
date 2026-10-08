"""Actual CPU configs must not enter HF dense fallback for unreadable bytes."""
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlavaConfig, PreTrainedTokenizerFast

from prismaquant.task_suite import preflight_tasks, measure_tasks


def make_reader_configuration(root, *, scope, quant_method="tessera"):
    model = root / "artifact"
    model.mkdir(parents=True)
    text = LlamaConfig(vocab_size=4, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    if quant_method is not None:
        if scope == "root":
            model_config = text
            model_config.quantization_config = {"quant_method": quant_method}
        else:
            text.quantization_config = {"quant_method": quant_method}
            model_config = LlavaConfig(text_config=text)
    else:
        model_config = text
    model_config.save_pretrained(model)
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "<pad>": 1, "a": 2, "b": 3}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>").save_pretrained(model)
    documents = root / "documents.jsonl"
    documents.write_text(json.dumps({"prompt": "a", "choices": [" a", " b"], "answer": 0}) + "\n")
    return {"schema": "prismaquant.task_suite/1", "backend": {"name": "hf", "pretrained": str(model),
        "tokenizer": str(model), "device": "cpu", "dtype": "float32", "batch_size": 1,
        "max_length": 32, "trust_remote_code": False}, "tasks": [{"task": "reader_fixture", "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"train": str(documents)}}, "test_split": "train",
        "output_type": "multiple_choice", "doc_to_text": "prompt", "doc_to_choice": "choices", "doc_to_target": "answer",
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}]}],
        "sampling": {"limit": 1, "num_fewshot": 0, "random_seed": 0, "numpy_seed": 0,
                     "torch_seed": 0, "fewshot_seed": 0}, "criteria": []}


@pytest.mark.parametrize("dev", [False, True])
@pytest.mark.parametrize("scope", ["root", "text"])
def test_declared_tessera_config_refuses_before_hf_dense_fallback(tmp_path, monkeypatch, dev, scope):
    config = make_reader_configuration(tmp_path, scope=scope)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    with pytest.raises(ValueError, match="quant_method.*tessera"):
        preflight_tasks(config)
    with pytest.raises(ValueError, match="quant_method.*tessera"):
        measure_tasks(config, tmp_path / "result.json")


@pytest.mark.parametrize("dev", [False, True])
def test_historical_hf_result_cannot_replay_declared_tessera(tmp_path, monkeypatch, dev):
    from test_task_quality_replay import make_task_replay_fixture
    from prismaquant.quality_stage import verify_result
    config, result, _ = make_task_replay_fixture(tmp_path)
    path = Path(config["backend"]["pretrained"], "config.json")
    model_config = json.loads(path.read_text())
    model_config["quantization_config"] = {"quant_method": "tessera"}
    path.write_text(json.dumps(model_config))
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    with pytest.raises(ValueError, match="quant_method.*tessera"):
        verify_result(result, config, config_sha256="a" * 64)
