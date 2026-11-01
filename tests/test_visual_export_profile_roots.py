"""Profile-owned visual assignments must agree with emitted native bytes."""
import json
import re

import pytest
import torch
from safetensors.torch import load_file, save_file

from prismaquant import export_native_compressed as exporter
from prismaquant import gpu_guard
from prismaquant.model_profiles.gemma4 import Gemma4Profile


VISION = "model.vision_tower.vision_model.encoder.layers.0.attn.qkv"
PROJECTOR = "model.embed_vision.projection"


def _weight():
    values = torch.arange(32 * 32).reshape(32, 32) % 31 - 15
    return values.to(torch.bfloat16) / 64


def _matches_name(name, targets):
    # vLLM checks exact names or re.match after the compressed-tensors re: prefix.
    return any(
        re.match(target[3:], name) is not None
        if target.startswith("re:") else target == name
        for target in targets
    )


@pytest.mark.parametrize("fmt", ["FP8_E4M3", "NVFP4"])
def test_native_main_emits_gemma_visual_recipe_and_matching_config(
    tmp_path, monkeypatch, fmt
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "model_type": "gemma4",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "vision_config": {"num_hidden_layers": 1},
    }))
    bf16 = "model.vision_tower.kept"
    unassigned = "model.embed_vision.unassigned"
    audio = "model.audio_tower.projection"
    tensors = {
        name + ".weight": _weight()
        for name in (VISION, PROJECTOR, bf16, unassigned, audio)
    }
    tensors[VISION + ".bias"] = torch.arange(32, dtype=torch.bfloat16)
    save_file(tensors, str(source / "model.safetensors"))
    recipe = tmp_path / "layer_config.json"
    recipe.write_text(json.dumps({VISION: fmt, PROJECTOR: fmt, bf16: "BF16"}))
    output = tmp_path / "output"

    # Isolate the unrelated text-body GPU sweep. Source passthrough intake,
    # recipe admission, visual codec, shard writer and config emission are real.
    monkeypatch.setattr(exporter, "materialize_tensors_streaming",
                        lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path",
                        lambda *args, **kwargs: torch.device("cpu"))
    monkeypatch.setattr(exporter, "_ALLOCATOR_TARGET_PROFILE", "research")
    exporter.main([
        "--model", str(source), "--layer-config", str(recipe),
        "--output", str(output), "--no-gptq", "--no-scale-sweep",
    ])

    emitted = load_file(str(output / "model.safetensors"))
    config = json.loads((output / "config.json").read_text())
    quantization = config["quantization_config"]
    groups = quantization["config_groups"]
    profile = Gemma4Profile()
    for name in (VISION, PROJECTOR):
        assert name + ".weight_scale" in emitted
        if fmt == "FP8_E4M3":
            assert emitted[name + ".weight"].dtype == torch.float8_e4m3fn
            assert emitted[name + ".weight_scale"].shape == (32, 1)
        else:
            assert name + ".weight" not in emitted
            assert emitted[name + ".weight_packed"].dtype == torch.uint8
            assert emitted[name + ".weight_packed"].shape == (32, 16)
            assert name + ".weight_global_scale" in emitted
            assert name + ".input_global_scale" in emitted
        target = profile.to_vllm_internal_name(name)
        matched = [group for group in groups.values()
                   if _matches_name(target, group["targets"])]
        assert len(matched) == 1
        assert matched[0]["weights"]["num_bits"] == (8 if fmt == "FP8_E4M3" else 4)
        assert not _matches_name(target, quantization["ignore"])
    for name in (bf16, unassigned, audio):
        assert torch.equal(emitted[name + ".weight"], tensors[name + ".weight"])
        assert name + ".weight_scale" not in emitted
        target = profile.to_vllm_internal_name(name)
        assert not any(_matches_name(target, group["targets"])
                       for group in groups.values())
    assert torch.equal(emitted[VISION + ".bias"], tensors[VISION + ".bias"])


@pytest.mark.parametrize("name", [
    "model.vision_tower.projection", "vision_tower.projection",
    "model.embed_vision.projection", "embed_vision.projection",
])
def test_declared_visual_roots_preserve_source_recipe_alias(name):
    emitted = exporter._apply_visual_recipe_quant(
        {name + ".weight": _weight()}, {name: "FP8_E4M3"},
        profile=Gemma4Profile(), device=torch.device("cpu"),
    )
    assert emitted[name + ".weight"].dtype == torch.float8_e4m3fn
    assert name + ".weight_scale" in emitted


@pytest.mark.parametrize("name", [
    "model.audio_tower.projection", "model.embed_audio.projection",
    "model.layers.0.mlp.up_proj", "model.vision_tower_extra.projection",
    "model.visual.projection",
])
def test_declared_visual_scope_does_not_admit_other_namespaces(name):
    weight = _weight()
    emitted = exporter._apply_visual_recipe_quant(
        {name + ".weight": weight}, {name: "FP8_E4M3"},
        profile=Gemma4Profile(), device=torch.device("cpu"),
    )
    assert list(emitted) == [name + ".weight"]
    assert emitted[name + ".weight"] is weight


@pytest.mark.parametrize("name", ["model.visual.projection", "visual.projection"])
def test_legacy_visual_namespace_without_profile_still_quantizes(name):
    emitted = exporter._apply_visual_recipe_quant(
        {name + ".weight": _weight()}, {name: "FP8_E4M3"},
        device=torch.device("cpu"),
    )
    assert emitted[name + ".weight"].dtype == torch.float8_e4m3fn
    assert name + ".weight_scale" in emitted
