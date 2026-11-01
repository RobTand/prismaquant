"""Guarded dense resume must refuse a changed checkpoint before direct packing."""
from __future__ import annotations

import json
import pickle

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from prismaquant import build_production_cache as builder
from prismaquant import export_native_compressed as exporter
from prismaquant import production_weight_cache as pwc
from prismaquant.model_profiles import DefaultProfile
from test_production_weight_cache import _TinyChain


@pytest.mark.parametrize("changed_source", [False, True], ids=["unchanged", "changed_weights"])
def test_guarded_dense_resume_binds_source_before_direct_export(
    tmp_path, monkeypatch, changed_source,
):
    source = _TinyChain().eval()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "test", "hidden_size": 32}))
    checkpoint = model_dir / "model.safetensors"
    save_file(source.state_dict(), str(checkpoint))
    assignment = tmp_path / "layer_config.json"
    assignment.write_text(json.dumps({"l1": "FP8_E4M3"}))
    calibration = torch.tensor([[0, 1]], dtype=torch.long)

    def load_model(path, **kwargs):
        model = _TinyChain().eval()
        model.load_state_dict(load_file(str(checkpoint)))
        return model.to(dtype=kwargs["torch_dtype"])

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(builder, "detect_profile_with_warning", lambda *args, **kwargs: DefaultProfile())
    monkeypatch.setattr(builder, "require_cuda_hot_path", lambda *args: torch.device("cpu"))
    monkeypatch.setattr(builder, "_load_cache_calibration", lambda *args: calibration)
    monkeypatch.setattr(exporter, "_CACHED_ACTIVATIONS", None)

    def build(directory):
        directory.mkdir(exist_ok=True)
        output = directory / "cache.pkl"
        assert builder.main([
            "--model", str(model_dir), "--output", str(output),
            "--cache-dir", str(directory / "weights"),
            "--render-scope", "assignment", "--render-layer-config", str(assignment),
            "--formats", "FP8_E4M3", "--enable", "none", "--dtype", "fp32",
            "--max-act-rows", "8",
        ]) == 0
        return pickle.loads(output.read_bytes())

    def pack(cache):
        monkeypatch.setattr(exporter, "_PRODUCTION_WEIGHT_CACHE", cache)
        result = exporter._pack_production_cached_2d(
            "l1", "FP8_E4M3", device=torch.device("cpu"),
        )
        assert result is not None
        return {name: tensor.clone() for name, tensor in result.items()}

    original_dir = tmp_path / "original"
    first = build(original_dir)
    first_packed = pack(first)
    identity = original_dir / "weights" / pwc.RENDER_IDENTITY_SIDECAR_FILENAME
    assert identity.is_file()
    assert pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY not in json.loads(identity.read_text())

    if not changed_source:
        resumed_packed = pack(build(original_dir))
        assert resumed_packed.keys() == first_packed.keys()
        assert all(torch.equal(resumed_packed[name], tensor) for name, tensor in first_packed.items())
        return

    with torch.no_grad():
        source.l1.weight.mul_(2)
    save_file(source.state_dict(), str(checkpoint))
    fresh_packed = pack(build(tmp_path / "changed_fresh"))
    assert any(not torch.equal(fresh_packed[name], tensor) for name, tensor in first_packed.items())

    with pytest.raises((RuntimeError, ValueError), match=r"source|weight|identity"):
        resumed_packed = pack(build(original_dir))
        assert resumed_packed.keys() == first_packed.keys()
        assert all(torch.equal(resumed_packed[name], tensor) for name, tensor in first_packed.items())
