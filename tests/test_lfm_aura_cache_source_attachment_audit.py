"""The LFM cost adapter must bind independent renders before pricing dW."""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from experiments import lfm_packed_joint_screen as screen
from prismaquant import aura_cost
from prismaquant import cost_streaming, model_profiles
from prismaquant.model_profiles import DefaultProfile
from prismaquant.production_weight_cache import fill_production_weight_cache
from prismaquant.streamed_model_identity import build_source_model_binding
from prismaquant.tessera_hessian import calibration_identity


FORMAT = "TESSERA_E4M3_K1_R1024"
PREFIX = "model.layers.2.feed_forward.experts"
UNITS = tuple(f"{PREFIX}.{expert}.{role}" for expert in range(32) for role in ("w1", "w3", "w2"))


class _TinyExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("w1", "w3", "w2"):
            linear = torch.nn.Linear(32, 32, bias=False)
            with torch.no_grad():
                linear.weight.copy_(torch.eye(32) / 8)
            setattr(self, name, linear)


class _TinyLFMSource(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(4, 32)
        with torch.no_grad():
            self.embed.weight.copy_(torch.arange(128).reshape(4, 32) / 128)
        layer = torch.nn.Module()
        layer.feed_forward = torch.nn.Module()
        layer.feed_forward.experts = torch.nn.ModuleList(_TinyExpert() for _ in range(32))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity(), layer])

    def forward(self, input_ids, use_cache=False):
        x = self.embed(input_ids)
        return torch.stack([
            expert.w2(expert.w1(x) * expert.w3(x))
            for expert in self.model.layers[2].feed_forward.experts
        ]).mean(0)


def _load_model(path, **kwargs):
    model = _TinyLFMSource().eval()
    model.load_state_dict(load_file(str(Path(path) / "model.safetensors")))
    return model.to(dtype=kwargs.get("torch_dtype", kwargs.get("dtype")))


def _produce(model_dir, output_dir, units=UNITS):
    output_dir.mkdir()
    output = output_dir / "production.pkl"
    text = "zero one two three"
    vocabulary = {word: index for index, word in enumerate(text.split())}
    calib_ids = torch.tensor([[vocabulary[word] for word in text.split()]])
    hessian_identity = calibration_identity(
        text, [calib_ids], fit_tokens=calib_ids.numel(),
    )
    model = _load_model(model_dir, dtype=torch.bfloat16)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PRISMAQUANT_TARGET_PROFILE", "research")
        cache = fill_production_weight_cache(
            model, calib_ids, UNITS, formats=[FORMAT],
            render_assignment=dict.fromkeys(units, FORMAT),
            levers={"none": True, "tessera_hessian_identity": hessian_identity},
            max_act_rows=4, progress=False,
            cache_dir=output_dir / "weights", recache_profile=DefaultProfile(),
            source_binding=build_source_model_binding(model_dir, dtype=torch.bfloat16),
        )
    assert not cache.failed, cache.failed
    assert set(cache.weights) == {(unit, FORMAT) for unit in units}
    assert cache.levers["tessera_hessian_identity"] == hessian_identity
    output.write_bytes(pickle.dumps(cache))
    return output


@pytest.fixture(scope="module")
def rendered_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("lfm-source-attachment")
    model_dir = root / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "test", "hidden_size": 32, "num_hidden_layers": 3, "num_experts": 32,
    }))
    save_file(_TinyLFMSource().state_dict(), str(model_dir / "model.safetensors"))
    produced = _produce(model_dir, root / "rendered")
    cache = pickle.loads(produced.read_bytes())
    assert set(cache.weights) == {(unit, FORMAT) for unit in UNITS}
    return model_dir, produced


class _ReachedPricing(Exception):
    pass


@pytest.mark.parametrize("case", [
    "unchanged", "relocated", "changed_weights", "changed_config", "missing_binding",
    "malformed_binding", "hf_id",
])
def test_lfm_cost_main_binds_independent_cache_before_pricing(
    tmp_path, monkeypatch, rendered_source, case,
):
    original_model, original_pickle = rendered_source
    model_dir = tmp_path / "source"
    shutil.copytree(original_model, model_dir)
    produced = tmp_path / "production.pkl"
    shutil.copyfile(original_pickle, produced)
    cache = pickle.loads(produced.read_bytes())
    original_render = cache.get(UNITS[0], FORMAT).clone()
    attach_source = model_dir
    snapshot_calls = []
    if case == "relocated":
        relocated = tmp_path / "same_source_elsewhere"
        shutil.copytree(model_dir, relocated)
        model_dir = relocated
        attach_source = relocated
    elif case == "changed_weights":
        values = load_file(str(model_dir / "model.safetensors"))
        values[UNITS[0] + ".weight"].mul_(2)
        save_file(values, str(model_dir / "model.safetensors"))
        fresh = pickle.loads(_produce(model_dir, tmp_path / "fresh", units=UNITS[:1]).read_bytes())
        assert not torch.equal(fresh.get(UNITS[0], FORMAT), original_render)
    elif case == "changed_config":
        path = model_dir / "config.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), "source_revision": 2}))
    elif case == "missing_binding":
        cache.metadata.pop("source_binding", None)
        produced.write_bytes(pickle.dumps(cache))
    elif case == "malformed_binding":
        cache.metadata["source_binding"]["effective_dtype"] = "torch.float64"
        produced.write_bytes(pickle.dumps(cache))
    elif case == "hf_id":
        import huggingface_hub

        def snapshot_download(*, repo_id, local_files_only=False):
            assert repo_id == "audit/lfm-source"
            assert local_files_only is False
            assert not snapshot_calls, "LFM resolved its remote source twice"
            snapshot_calls.append(repo_id)
            return str(model_dir)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
        attach_source = "audit/lfm-source"

    tokens = tmp_path / "tokens.safetensors"
    save_file({"calibration_ids": torch.zeros(1, 512, dtype=torch.int64)}, str(tokens))
    pricing_calls = []
    runner_sources = []

    def build_runner(path, *, dtype, **kwargs):
        assert Path(path) == model_dir
        runner_sources.append(str(path))
        model = _load_model(path, dtype=dtype)
        config = json.loads((Path(path) / "config.json").read_text())
        model.config = SimpleNamespace(
            _attn_implementation="eager", _experts_implementation="eager", to_dict=lambda: dict(config),
        )

        def unexpected_model_execution(*args, **kwargs):
            raise AssertionError("source-attachment fixture must stop before model execution")

        return SimpleNamespace(
            model=model, dtype=dtype, device=torch.device("cpu"), num_layers=3,
            capture_boundaries=unexpected_model_execution,
            tail_logits=unexpected_model_execution,
            context=SimpleNamespace(
                weight_ckpt={name: name for name in model.state_dict()},
                weight_shard={name: str(Path(path) / "model.safetensors") for name in model.state_dict()},
                schedule_prefetch=lambda layer: None, shutdown=lambda: None,
            ),
        )

    def price(runner, ids, formats, **kwargs):
        pricing_calls.append(case)
        assert formats == [FORMAT, "BF16"]
        assert set(kwargs["production_cache"].weights) == {(unit, FORMAT) for unit in UNITS}
        delta, origin = aura_cost._delta_w(
            UNITS[0], FORMAT, runner.model.get_parameter(UNITS[0] + ".weight"),
            kwargs["production_cache"], strict=True,
        )
        assert origin == "rendered"
        raise _ReachedPricing(f"cached dW squared norm={float(delta.square().sum())}")

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_profiles, "detect_profile", lambda *args, **kwargs: DefaultProfile())
    monkeypatch.setattr(cost_streaming, "build_streamed_causal_lm", build_runner)
    monkeypatch.setattr(aura_cost, "compute_aura_cost_streamed", price)
    monkeypatch.setattr("sys.argv", [
        "lfm_packed_joint_screen.py", "--mode", "cost", "--model", str(attach_source),
        "--tokens", str(tokens), "--tokens-sha256", hashlib.sha256(tokens.read_bytes()).hexdigest(),
        "--cache", str(produced), "--cache-sha256", hashlib.sha256(produced.read_bytes()).hexdigest(),
        "--profile-tool", "none", "--out", str(tmp_path / "out"),
    ])
    if case in {"unchanged", "relocated", "hf_id"}:
        with pytest.raises(_ReachedPricing):
            screen.main()
        assert pricing_calls == [case]
        assert runner_sources == [str(model_dir)]
    else:
        with pytest.raises((RuntimeError, ValueError), match=r"source|identity|binding"):
            screen.main()
        assert pricing_calls == []
        assert runner_sources == []
    if case == "hf_id":
        assert snapshot_calls == ["audit/lfm-source"]
