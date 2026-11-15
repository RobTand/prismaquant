"""Exercise buffer preservation through the actual streaming exporter."""
import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from prismaquant import export_native_compressed as export
from prismaquant import layer_streaming, sensitivity_probe, streaming_model


def _export_fixture(tmp_path, monkeypatch, layers=1, width=2):
    """Replace architecture construction only; source reads and emission are real."""
    expected = torch.tensor([0.5, 0.5 + 2.0 ** -12], dtype=torch.float32)
    source = tmp_path / "source"
    source.mkdir()
    save_file({
        key: tensor for i in range(layers) for key, tensor in {
            f"model.layers.{i}.expert_bias": expected.clone(),
            f"model.layers.{i}.proj.weight": torch.eye(width, dtype=torch.bfloat16),
        }.items()
    }, str(source / "model.safetensors"))

    class Skeleton:
        @staticmethod
        def _from_config(config):
            model = nn.Module()
            model.model = nn.Module()
            model.model.layers = nn.ModuleList([nn.Module() for _ in range(layers)])
            for layer in model.model.layers:
                layer.proj = nn.Linear(width, width, bias=False, device="meta")
                layer.register_buffer("expert_bias", torch.empty(2, device="meta", dtype=torch.float32))
            return model

    profile = SimpleNamespace(
        requires_multimodal_skeleton=lambda: False,
        concat_merge_groups=lambda: (),
        live_to_recipe_name=lambda name: name,
        packed_expert_param_names=lambda: (),
    )
    from transformers import AutoConfig
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(sensitivity_probe, "stage_text_only", lambda path: path)
    monkeypatch.setattr(streaming_model, "_skeleton_config_and_class",
                        lambda config, **kw: (config, Skeleton))
    monkeypatch.setattr(streaming_model, "_init_rotary_inplace", lambda *a: None)
    monkeypatch.setattr(export, "_PRODUCTION_WEIGHT_CACHE", None)

    def run(cache=None, layer_range=None, assignment=None):
        emitted = {}
        def sink(tensors):
            emitted.update(tensors)
        _, run.hist = export.materialize_tensors_streaming(
            str(source), assignment or {}, profile=profile, bf16_passthrough=set(),
            device=torch.device("cpu"), tensor_sink=sink,
            export_cache_dir=None if cache is None else str(cache), layer_range=layer_range)
        path = tmp_path / "emitted.safetensors"
        save_file(emitted, str(path))
        return load_file(str(path))
    return run, expected


def test_streaming_export_serializes_original_fp32_buffer_bytes(tmp_path, monkeypatch):
    run, expected = _export_fixture(tmp_path, monkeypatch)
    emitted = run()
    actual = emitted["model.layers.0.expert_bias"]
    assert actual.dtype == torch.float32
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert emitted["model.layers.0.proj.weight"].dtype == torch.bfloat16


def test_streaming_export_invalidates_resume_from_old_buffer_policy(tmp_path, monkeypatch):
    run, expected = _export_fixture(tmp_path, monkeypatch)
    cache = tmp_path / "resume"
    real_read = layer_streaming._read_layer_to_device
    def old_read(*args, **kwargs):
        kwargs.pop("buffer_dtypes", None)
        return real_read(*args, **kwargs)
    # Produce the prior implementation's layer payload through the real exporter.
    with monkeypatch.context() as old:
        old.setattr(layer_streaming, "_read_layer_to_device", old_read)
        narrowed = run(cache)["model.layers.0.expert_bias"]
    assert not torch.equal(narrowed, expected)
    manifest = cache / "manifest.json"
    legacy = json.loads(manifest.read_text())
    legacy.pop("persistent_buffer_read_policy", None)
    manifest.write_text(json.dumps(legacy))

    # Same source/assignment/levers, but the corrected reader must execute.
    emitted = run(cache)["model.layers.0.expert_bias"]
    assert torch.equal(emitted.view(torch.uint8), expected.view(torch.uint8))
    assert json.loads(manifest.read_text())["persistent_buffer_read_policy"]


def test_range_replay_preserves_tensors_and_counts(tmp_path, monkeypatch):
    run, _ = _export_fixture(tmp_path, monkeypatch, layers=2, width=32)
    stats = Counter()
    monkeypatch.setattr(export._native_render_codec, "_DO_NO_HARM_STATS", stats)
    reads = []
    real_read = layer_streaming._read_layer_to_device
    def read(prefix, *args, **kwargs):
        reads.append(prefix)
        stats["NVFP4_reverts"] += 1
        return real_read(prefix, *args, **kwargs)
    monkeypatch.setattr(layer_streaming, "_read_layer_to_device", read)
    baseline = run()
    baseline_bytes = (tmp_path / "emitted.safetensors").read_bytes()
    baseline_hist = run.hist
    cache = tmp_path / "ranges"
    reads.clear()
    for layer in (1, 0):
        stats.clear()
        assert len(run(cache, (layer, layer + 1))) == 2
        assert reads[-1] == f"model.layers.{layer}."
        record = torch.load(cache / f"layer_{layer:03d}.pt", weights_only=False)
        assert record["do_no_harm"] == {"NVFP4_reverts": 1}
    assert len(reads) == 2
    stats.clear()
    replay = run(cache)
    assert len(reads) == 2
    assert replay.keys() == baseline.keys()
    assert (tmp_path / "emitted.safetensors").read_bytes() == baseline_bytes
    assert run.hist == baseline_hist == {("layer_buffer", "FP32"): 2, ("linear", "BF16"): 2}
    assert stats == {"NVFP4_reverts": 2}


@pytest.mark.parametrize("bounds", [(-1, 1), (1, 1), (0, 3)])
def test_export_refuses_invalid_layer_range(tmp_path, monkeypatch, bounds):
    run, _ = _export_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="invalid layer range"):
        run(tmp_path / "ranges", bounds)


def test_recache_render_units_feed_export(tmp_path, monkeypatch):
    from prismaquant.measure_quant_cost import ActivationIndex
    from prismaquant.model_profiles import DefaultProfile
    from prismaquant.streaming_production_cache import run_streaming_render
    from prismaquant import streaming_production_cache as streaming

    run, _ = _export_fixture(tmp_path, monkeypatch, layers=2, width=32)
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module() for _ in range(2)])
    model.root = nn.Linear(32, 32, bias=False)
    for layer in model.model.layers:
        layer.proj = nn.Linear(32, 32, bias=False)
    names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    act = tmp_path / "act"
    act.mkdir()
    for name in names:
        torch.save({"name": name, "inputs": torch.eye(32)}, act / (name.replace(".", "__") + ".pt"))
    monkeypatch.setattr(streaming, "render_production_weight", lambda weight, *a, **kw: weight.detach().clone())
    installed = []
    def render(bounds=None):
        return run_streaming_render(
            model, layers_prefix="model.layers.", num_layers=2,
            render_assignment=dict.fromkeys(names, "FP8_E4M3"),
            act_index=ActivationIndex(act, []), formats=["FP8_E4M3"], levers={"gptq": False},
            cache_dir_path=None, profile=DefaultProfile(), skip_tokens=[], device=torch.device("cpu"),
            layer_range=bounds, install=installed.append, unload=lambda layer: None, progress=False)
    single = render()
    installed.clear()
    units = [render((layer, layer + 1)) for layer in (1, 0)]
    assert installed == [1, 0]
    assert set(units[0].weights).isdisjoint(units[1].weights)
    assert ("root", "FP8_E4M3") in units[1].weights
    assert ("root", "FP8_E4M3") not in units[0].weights
    for cache in units:
        assert cache.metadata["requested_entries"] == len(cache)
        for key in cache.weights:
            assert torch.equal(cache.get(*key), single.get(*key))
    monkeypatch.setattr(export, "_PRODUCTION_WEIGHT_CACHE", single)
    assignment = {name: "FP8_E4M3" for name in names if name != "root"}
    baseline = run(assignment=assignment)
    baseline_bytes = (tmp_path / "emitted.safetensors").read_bytes()
    baseline_hist = run.hist
    union = units[1]
    union.weights.update(units[0].weights)
    union.activation_max_abs.update(units[0].activation_max_abs)
    monkeypatch.setattr(export, "_PRODUCTION_WEIGHT_CACHE", union)
    replay = run(assignment=assignment)
    assert (tmp_path / "emitted.safetensors").read_bytes() == baseline_bytes
    assert run.hist == baseline_hist
    assert all(torch.equal(replay[name], tensor) for name, tensor in baseline.items())
