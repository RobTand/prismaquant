"""Every validation metric must measure the configured production render."""
from __future__ import annotations

import json
import math
import pickle
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from prismaquant import format_registry, model_runtime, validation_harness
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.streamed_model_identity import build_source_model_binding


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_position_embeddings=128)
        self.embed = nn.Embedding(8, 128)
        self.proj = nn.Linear(128, 8, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.eye(8, 128))
            self.proj.weight.copy_(torch.linspace(-2, 2, 1024).reshape(8, 128))

    def forward(self, input_ids, labels=None):
        logits = self.proj(self.embed(input_ids))
        loss = logits.sum() * 0
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, 8), labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


class _Tokenizer:
    pad_token_id = 0
    eos_token = None
    model_max_length = 128

    def __call__(self, text, *, add_special_tokens=False, return_tensors="pt"):
        del add_special_tokens, return_tensors
        return SimpleNamespace(input_ids=torch.tensor(
            [[ord(char) % 8 for char in text] or [0]], dtype=torch.long))


def _dataset(name, config, *, split, cache_dir):
    del config, split, cache_dir
    if "mmlu" in name:
        return [{"question": "Which letter comes first?", "choices": ["A", "B"],
                 "answer": 0, "subject": "letters"}]
    return [{"text": "abcdefgh" * 16}]


def _install_local_inputs(monkeypatch, model, tokenizer):
    datasets = ModuleType("datasets")
    datasets.load_dataset = _dataset
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: model)
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(model_runtime, "initialize_model_runtime", lambda: None)
    for name in ("PRISMAQUANT_VALIDATION_CUDA_GRAPHS", "PRISMAQUANT_KL_CUDA_GRAPHS",
                 "PRISMAQUANT_VALIDATION_SKIP_END_KL",
                 "PRISMAQUANT_SHARED_WEIGHT_FORMAT_CACHE"):
        monkeypatch.setenv(name, "0")
    monkeypatch.delenv("PRISMAQUANT_VALIDATION_FAKE_METRICS", raising=False)
    monkeypatch.delenv("PRISMAQUANT_VALIDATION_WIKITEXT_STRIDE", raising=False)
    for name in ("PRISMAQUANT_VALIDATION_PROD_CACHE",
                 "PRISMAQUANT_VALIDATION_PROD_CACHE_DIR",
                 "PRISMAQUANT_VALIDATION_PROD_CACHE_LRU_GB"):
        monkeypatch.delenv(name, raising=False)


def _expected_last_token_kl(model, ids, student_weight):
    with torch.no_grad():
        hidden = model.embed(ids)[:, -1:, :]
        teacher = F.log_softmax(F.linear(hidden, model.proj.weight).float(), dim=-1)
        student = F.log_softmax(F.linear(hidden, student_weight).float(), dim=-1)
        return float((teacher.exp() * (teacher - student)).sum(dim=-1).mean())


def test_real_validation_uses_one_production_render_for_ppl_mmlu_and_end_kl(
        tmp_path, monkeypatch):
    model = _Model().eval()
    tokenizer = _Tokenizer()
    _install_local_inputs(monkeypatch, model, tokenizer)
    original = model.proj.weight.detach().clone()
    cached_weight = torch.zeros_like(original)
    fmt = "INT8_W8A16"
    model_dir = tmp_path / "source"
    model_dir.mkdir()
    save_file(model.state_dict(), str(model_dir / "model.safetensors"))
    (model_dir / "config.json").write_text(json.dumps(vars(model.config)))
    cache = ProductionWeightCache(
        weights={("proj", fmt): cached_weight}, levers={},
        metadata={
            "source_binding": build_source_model_binding(model_dir, dtype=torch.float32),
        },
    )
    cache_path = tmp_path / "production.pkl"
    with cache_path.open("wb") as stream:
        pickle.dump(cache, stream)
    monkeypatch.setenv("PRISMAQUANT_VALIDATION_PROD_CACHE", str(cache_path))
    monkeypatch.setenv("PRISMAQUANT_VALIDATION_PROD_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PRISMAQUANT_VALIDATION_PROD_CACHE_LRU_GB", "1")

    configured = []
    reads = []
    real_enable_lru, real_get = ProductionWeightCache.enable_lru, ProductionWeightCache.get

    def enable_lru(self, max_bytes):
        configured.append(id(self))
        return real_enable_lru(self, max_bytes)

    def get(self, *args, **kwargs):
        reads.append(id(self))
        return real_get(self, *args, **kwargs)

    monkeypatch.setattr(ProductionWeightCache, "enable_lru", enable_lru)
    monkeypatch.setattr(ProductionWeightCache, "get", get)
    ids = validation_harness._fixed_calib_ids(
        tokenizer, _dataset, cache_dir=tmp_path, n_samples=2, seqlen=8,
        split="test", seed=42)
    expected = _expected_last_token_kl(model, ids, cached_weight)
    rtn_weight = format_registry.get_format(fmt).quantize_dequantize(original)
    rtn = _expected_last_token_kl(model, ids, rtn_weight)
    assert abs(expected - rtn) > 0.1, "fixture must distinguish cached and RTN weights"

    result = validation_harness.validate_artifact(
        str(model_dir), {"proj": fmt}, cache_dir=tmp_path, device="cpu",
        dtype="fp32", n_wikitext_tokens=32, n_mmlu_questions=1,
        calib_seqlen=8, calib_n_samples=2, calib_split="test", calib_seed=42,
        progress=False,
    )

    assert result["ppl_wikitext"] == pytest.approx(8.0, rel=1e-6)
    assert result["ppl_mmlu_acc"] == 1.0
    assert math.isfinite(result["end_kl"])
    assert result["end_kl"] == pytest.approx(expected, abs=1e-6)
    assert abs(result["end_kl"] - rtn) > 0.1
    assert len(configured) == 1
    assert reads and set(reads) == set(configured)
    assert torch.equal(model.proj.weight, original)
    assert not model.proj._forward_pre_hooks
    assert not model.proj._forward_hooks


def test_real_validation_without_production_cache_preserves_rtn_and_teacher(
        tmp_path, monkeypatch):
    model = _Model().eval()
    tokenizer = _Tokenizer()
    _install_local_inputs(monkeypatch, model, tokenizer)
    original = model.proj.weight.detach().clone()
    fmt = "INT8_W8A16"
    ids = validation_harness._fixed_calib_ids(
        tokenizer, _dataset, cache_dir=tmp_path, n_samples=2, seqlen=8,
        split="test", seed=42)
    rtn_weight = format_registry.get_format(fmt).quantize_dequantize(original)
    expected = _expected_last_token_kl(model, ids, rtn_weight)
    result = validation_harness.validate_artifact(
        "local-toy-model", {"proj": fmt}, cache_dir=tmp_path, device="cpu",
        dtype="fp32", n_wikitext_tokens=32, n_mmlu_questions=1,
        calib_seqlen=8, calib_n_samples=2, progress=False,
    )
    assert result["end_kl"] == pytest.approx(expected, abs=1e-6)
    assert torch.equal(model.proj.weight, original)
    assert not model.proj._forward_pre_hooks
    assert not model.proj._forward_hooks
