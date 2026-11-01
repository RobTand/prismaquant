from __future__ import annotations

import math
import pickle
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BertConfig

from prismaquant import format_registry, validation_harness
from prismaquant.model_runtime import genuine_weight_initialization
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.streamed_model_identity import build_source_model_binding


UNIT = "bert.encoder.layer.0.intermediate.dense"
FORMAT = "INT8_W8A16"


class _Tokenizer:
    pad_token_id = 0
    eos_token = None
    model_max_length = 128

    def __call__(self, text, *, add_special_tokens=False, return_tensors="pt"):
        del add_special_tokens, return_tensors
        return SimpleNamespace(input_ids=torch.tensor(
            [[ord(char) % 8 for char in text] or [0]], dtype=torch.long))


def _dataset(_name, _config, *, split, cache_dir):
    del split, cache_dir
    return [{"text": "abcdefgh" * 16}]


def _local_inputs(monkeypatch):
    tokenizer_paths = []

    def load_tokenizer(path, *_args, **_kwargs):
        tokenizer_paths.append(str(path))
        return _Tokenizer()

    datasets = ModuleType("datasets")
    datasets.load_dataset = _dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    for name in ("PRISMAQUANT_VALIDATION_CUDA_GRAPHS", "PRISMAQUANT_KL_CUDA_GRAPHS",
                 "PRISMAQUANT_VALIDATION_SKIP_END_KL",
                 "PRISMAQUANT_SHARED_WEIGHT_FORMAT_CACHE"):
        monkeypatch.setenv(name, "0")
    monkeypatch.delenv("PRISMAQUANT_VALIDATION_FAKE_METRICS", raising=False)
    monkeypatch.delenv("PRISMAQUANT_VALIDATION_WIKITEXT_STRIDE", raising=False)
    return tokenizer_paths


@pytest.mark.parametrize(
    "cache_case", [
        "changed_weights", "missing_binding", "effective_dtype", "matching", "remote_snapshot",
    ]
)
def test_validation_binds_configured_cache_before_attachment(
    tmp_path, monkeypatch, cache_case,
):
    tokenizer_paths = _local_inputs(monkeypatch)
    torch.manual_seed(121)
    with genuine_weight_initialization():
        source_model = AutoModelForCausalLM.from_config(BertConfig(
            vocab_size=8, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2,
            max_position_embeddings=128, is_decoder=True,
        )).eval()
    model_dir = tmp_path / "source"
    source_model.save_pretrained(model_dir, safe_serialization=True)
    cache_dtype = torch.bfloat16 if cache_case == "effective_dtype" else torch.float32
    binding = build_source_model_binding(model_dir, dtype=cache_dtype)
    original = source_model.get_submodule(UNIT).weight.detach().clone()
    rendered = format_registry.get_format(FORMAT).quantize_dequantize(original)
    metadata = {} if cache_case == "missing_binding" else {"source_binding": binding}
    cache = ProductionWeightCache(
        weights={(UNIT, FORMAT): rendered.to(cache_dtype)}, levers={}, metadata=metadata,
    )
    cache_path = tmp_path / "production.pkl"
    cache_path.write_bytes(pickle.dumps(cache))
    if cache_case == "changed_weights":
        with torch.no_grad():
            source_model.get_submodule(UNIT).weight.add_(0.5)
        source_model.save_pretrained(model_dir, safe_serialization=True)

    model_reference = str(model_dir)
    snapshot_calls, model_paths = [], []
    if cache_case == "remote_snapshot":
        model_reference = "audit/remote-source"

        def snapshot_download(*, repo_id, local_files_only):
            snapshot_calls.append((repo_id, local_files_only))
            return str(model_dir)

        real_loader = AutoModelForCausalLM.from_pretrained

        def load_model(path, *args, **kwargs):
            model_paths.append(str(path))
            return real_loader(path, *args, **kwargs)

        monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
        monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_model)

    events = []
    for name in ("relocate", "enable_lru", "get"):
        real_method = getattr(ProductionWeightCache, name)

        def observed(self, *args, _name=name, _method=real_method, **kwargs):
            events.append(_name)
            return _method(self, *args, **kwargs)

        monkeypatch.setattr(ProductionWeightCache, name, observed)
    monkeypatch.setenv("PRISMAQUANT_VALIDATION_PROD_CACHE", str(cache_path))
    monkeypatch.setenv("PRISMAQUANT_VALIDATION_PROD_CACHE_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("PRISMAQUANT_VALIDATION_PROD_CACHE_LRU_GB", "1")

    def validate():
        return validation_harness.validate_artifact(
            model_reference, {UNIT: FORMAT}, cache_dir=tmp_path / "metrics",
            device="cpu", dtype="bf16", n_wikitext_tokens=32, n_mmlu_questions=0,
            calib_seqlen=8, calib_n_samples=2, progress=False,
        )

    if cache_case not in {"matching", "remote_snapshot"}:
        with pytest.raises(ValueError, match="source binding|source.*differs"):
            validate()
        assert events == []
        assert tokenizer_paths == []
        return

    result = validate()
    assert math.isfinite(result["ppl_wikitext"])
    assert math.isfinite(result["end_kl"])
    assert events[:2] == ["relocate", "enable_lru"]
    assert "get" in events
    assert tokenizer_paths == [str(model_dir)]
    if cache_case == "remote_snapshot":
        assert snapshot_calls == [(model_reference, False)]
        assert model_paths == [str(model_dir)]
