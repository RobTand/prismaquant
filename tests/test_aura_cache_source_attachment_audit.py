"""AURA must bind an independent rendered cache before pricing its deltas."""
from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from prismaquant import aura_cost, build_production_cache as builder
from prismaquant import calibration_data, cost_streaming, gpu_guard, model_profiles
from prismaquant.model_profiles import DefaultProfile
from test_production_weight_cache import _TinyChain


class _ReachedPricing(Exception):
    pass


@pytest.mark.parametrize("streaming", [False, True], ids=["resident", "streamed"])
@pytest.mark.parametrize("case", [
    "unchanged", "relocated", "changed_weights", "changed_config",
    "dtype_mismatch", "missing_binding",
])
def test_aura_main_binds_independent_cache_before_pricing(
    tmp_path, monkeypatch, streaming, case,
):
    source = _TinyChain().eval()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = model_dir / "config.json"
    config.write_text(json.dumps({"model_type": "test", "hidden_size": 32}))
    checkpoint = model_dir / "model.safetensors"
    save_file(source.state_dict(), str(checkpoint))
    assignment = tmp_path / "assignment.json"
    assignment.write_text(json.dumps({"l1": "FP8_E4M3"}))
    calibration = torch.tensor([[0, 1]], dtype=torch.long)

    def load_model(path, **kwargs):
        model = _TinyChain().eval()
        model.load_state_dict(load_file(str(Path(path) / "model.safetensors")))
        dtype = kwargs.get("torch_dtype", kwargs.get("dtype"))
        assert isinstance(dtype, torch.dtype)
        return model.to(dtype=dtype)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(builder, "detect_profile_with_warning", lambda *args, **kwargs: DefaultProfile())
    monkeypatch.setattr(builder, "require_cuda_hot_path", lambda *args: torch.device("cpu"))
    monkeypatch.setattr(builder, "_load_cache_calibration", lambda *args: calibration)
    monkeypatch.setattr(model_profiles, "detect_profile", lambda *args, **kwargs: DefaultProfile())
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *args: torch.device("cpu"))
    monkeypatch.setattr(calibration_data, "load_wikitext_calibration_windowed", lambda *args, **kwargs: calibration)
    monkeypatch.setenv("PRISMAQUANT_TARGET_PROFILE", "research")
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)

    def build(directory):
        directory.mkdir()
        output = directory / "cache.pkl"
        assert builder.main([
            "--model", str(model_dir), "--output", str(output),
            "--cache-dir", str(directory / "weights"),
            "--render-scope", "assignment", "--render-layer-config", str(assignment),
            "--formats", "FP8_E4M3", "--enable", "none",
            "--dtype", "fp16" if case == "dtype_mismatch" else "bf16",
            "--max-act-rows", "8",
        ]) == 0
        return output

    produced = build(tmp_path / "produced")
    cache = pickle.loads(produced.read_bytes())
    original_render = cache.get("l1", "FP8_E4M3").clone()
    attach_source = model_dir
    if case == "relocated":
        attach_source = tmp_path / "same_bytes_elsewhere"
        shutil.copytree(model_dir, attach_source)
    elif case == "changed_weights":
        with torch.no_grad():
            source.l1.weight.mul_(2)
        save_file(source.state_dict(), str(checkpoint))
        fresh = pickle.loads(build(tmp_path / "fresh_changed").read_bytes())
        assert not torch.equal(fresh.get("l1", "FP8_E4M3"), original_render)
    elif case == "changed_config":
        config.write_text(json.dumps({"model_type": "test", "hidden_size": 32, "source_revision": 2}))
    elif case == "missing_binding":
        cache.metadata.pop("source_binding", None)
        produced.write_bytes(pickle.dumps(cache))

    pricing_calls = []

    def price(model_or_runner, calibration_ids, formats, **kwargs):
        pricing_calls.append(streaming)
        model = model_or_runner.model if streaming else model_or_runner
        assert kwargs["require_production_cache"] is True
        assert formats == ["FP8_E4M3"]
        delta, origin = aura_cost._delta_w(
            "l1", "FP8_E4M3", model.l1.weight,
            kwargs["production_cache"], strict=True,
        )
        assert origin == "rendered"
        assert torch.equal(
            delta,
            original_render.float() - model.l1.weight.float(),
        )
        raise _ReachedPricing(f"cached dW squared norm={float(delta.square().sum())}")

    def build_runner(path, *, dtype, **kwargs):
        return SimpleNamespace(
            model=load_model(path, dtype=dtype),
            shutdown=lambda: None,
        )

    monkeypatch.setattr(cost_streaming, "build_streamed_causal_lm", build_runner)
    monkeypatch.setattr(aura_cost, "compute_aura_cost", price)
    monkeypatch.setattr(aura_cost, "compute_aura_cost_streamed", price)
    argv = [
        "--model", str(attach_source), "--output", str(tmp_path / "cost.pkl"),
        "--production-cache", str(produced), "--require-production-cache",
        "--formats", "FP8_E4M3", "--dtype", "bfloat16", "--device", "cpu",
        "--n-calib-samples", "1", "--calib-seqlen", "2",
    ]
    if streaming:
        argv.append("--streaming")
    if case in {"unchanged", "relocated"}:
        with pytest.raises(_ReachedPricing):
            aura_cost.main(argv)
        assert pricing_calls == [streaming]
    else:
        with pytest.raises((RuntimeError, ValueError), match=r"source|dtype|identity|binding"):
            aura_cost.main(argv)
        assert pricing_calls == []
