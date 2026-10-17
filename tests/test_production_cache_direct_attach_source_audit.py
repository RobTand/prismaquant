"""Native CLI attachment must bind an independent producer pickle to its source."""
from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from prismaquant import build_production_cache as builder
from prismaquant import export_native_compressed as exporter
from prismaquant import model_profiles
from prismaquant.model_profiles import DefaultProfile
from test_production_weight_cache import _TinyChain


class _ReachedNativePacker(Exception):
    pass


@pytest.mark.parametrize(
    "case", ["unchanged", "relocated", "changed_weights", "changed_config", "dtype_mismatch"],
)
def test_native_main_checks_source_before_independent_cache_attachment(
    tmp_path, monkeypatch, case,
):
    source = _TinyChain().eval()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = model_dir / "config.json"
    config.write_text(json.dumps({"model_type": "test", "hidden_size": 32}))
    checkpoint = model_dir / "model.safetensors"
    save_file(source.state_dict(), str(checkpoint))
    assignment = tmp_path / "layer_config.json"
    assignment.write_text(json.dumps({"l1": "FP8_E4M3"}))
    calibration = torch.tensor([[0, 1]], dtype=torch.long)

    def load_model(path, **kwargs):
        model = _TinyChain().eval()
        model.load_state_dict(load_file(str(Path(path) / "model.safetensors")))
        return model.to(dtype=kwargs["torch_dtype"])

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(builder, "detect_profile_with_warning", lambda *args, **kwargs: DefaultProfile())
    monkeypatch.setattr(builder, "require_cuda_hot_path", lambda *args: torch.device("cpu"))
    monkeypatch.setattr(builder, "_load_cache_calibration", lambda *args: calibration)
    monkeypatch.setattr(model_profiles, "detect_profile", lambda *args, **kwargs: DefaultProfile())
    monkeypatch.setattr(exporter, "_CACHED_ACTIVATIONS", None)
    monkeypatch.setattr(exporter, "_PRODUCTION_WEIGHT_CACHE", None)
    monkeypatch.setattr(exporter, "_ALLOCATOR_TARGET_PROFILE", "research")
    monkeypatch.setenv("PRISMAQUANT_TARGET_PROFILE", "research")

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
    monkeypatch.setattr(exporter, "_PRODUCTION_WEIGHT_CACHE", cache)
    original = exporter._pack_production_cached_2d(
        "l1", "FP8_E4M3", device=torch.device("cpu"),
    )
    assert original is not None
    original = {name: value.clone() for name, value in original.items()}
    attach_source = model_dir
    if case == "relocated":
        attach_source = tmp_path / "same_bytes_elsewhere"
        shutil.copytree(model_dir, attach_source)
    elif case == "changed_weights":
        with torch.no_grad():
            source.l1.weight.mul_(2)
        save_file(source.state_dict(), str(checkpoint))
        fresh = pickle.loads(build(tmp_path / "fresh_changed").read_bytes())
        monkeypatch.setattr(exporter, "_PRODUCTION_WEIGHT_CACHE", fresh)
        fresh_packed = exporter._pack_production_cached_2d(
            "l1", "FP8_E4M3", device=torch.device("cpu"),
        )
        assert any(not torch.equal(fresh_packed[name], value) for name, value in original.items())
    elif case == "changed_config":
        config.write_text(json.dumps({"model_type": "test", "hidden_size": 32, "source_revision": 2}))

    packed = {}

    def observe_attached_cache(actual_cache, *, profile=None):
        assert actual_cache is exporter._PRODUCTION_WEIGHT_CACHE
        result = exporter._pack_production_cached_2d(
            "l1", "FP8_E4M3", device=torch.device("cpu"),
        )
        assert result is not None
        packed.update({name: value.clone() for name, value in result.items()})
        raise _ReachedNativePacker

    monkeypatch.setattr(exporter, "_production_cache_scales", observe_attached_cache)

    def attach():
        try:
            exporter.main([
                "--model", str(attach_source), "--layer-config", str(assignment),
                "--production-weight-cache", str(produced),
                "--production-cache-lru-gb", "0",
                "--output", str(tmp_path / "exported"),
                "--no-gptq", "--no-scale-sweep",
            ])
        except _ReachedNativePacker:
            return
        pytest.fail("native attachment did not reach the observed packer")

    if case in {"unchanged", "relocated"}:
        attach()
    else:
        with pytest.raises((RuntimeError, ValueError), match=r"source|dtype|identity|binding"):
            attach()
    if packed:
        assert packed.keys() == original.keys()
        assert all(torch.equal(packed[name], value) for name, value in original.items())
