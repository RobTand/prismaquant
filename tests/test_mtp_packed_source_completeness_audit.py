"""The actual exact-wrapper gate must refuse partially loaded packed experts."""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile
from prismaquant.mtp_production_cache import _load_exact_mtp_wrapper
from test_mtp_module_arch import _minimal_moe_text_config


def _write_mtp_source(tmp_path, scope):
    config = _minimal_moe_text_config()
    config.architectures = ["Qwen3_5MoeForCausalLM"]
    profile = Qwen3_5Profile()
    template = profile.build_mtp_module(config)
    state = template.state_dict()
    gate_name = "layers.0.mlp.experts.gate_up_proj"
    down_name = "layers.0.mlp.experts.down_proj"
    experts, two_inter, hidden = state[gate_name].shape
    inter = two_inter // 2
    raw = {
        "mtp." + name: torch.full_like(value, 0.125)
        for name, value in state.items() if name not in {gate_name, down_name}
    }
    for expert in range(experts):
        for projection, shape, value in (
            ("gate_proj", (inter, hidden), (expert + 1) / 8),
            ("up_proj", (inter, hidden), (expert + 1) / 4),
            ("down_proj", (hidden, inter), (expert + 1) / 16),
        ):
            if expert == experts - 1 and (
                scope == "missing_expert"
                or scope == "missing_up" and projection == "up_proj"
                or scope == "missing_down" and projection == "down_proj"
            ):
                continue
            raw[f"mtp.layers.0.mlp.experts.{expert}.{projection}.weight"] = torch.full(shape, value)
    if scope == "missing_fc":
        del raw["mtp.fc.weight"]
    elif scope == "empty":
        raw.clear()
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(config.to_json_string())
    save_file(raw, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in raw},
    }))
    return root, config, profile, raw, experts, inter


@pytest.mark.parametrize("scope", ["complete", "missing_expert", "missing_up", "missing_down"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_exact_mtp_wrapper_requires_every_packed_expert_source_slice(tmp_path, monkeypatch, scope, dtype):
    from transformers import AutoConfig

    root, config, profile, raw, experts, inter = _write_mtp_source(tmp_path, scope)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)

    if scope != "complete":
        with pytest.raises(RuntimeError, match="MTP source/module coverage failure"):
            _load_exact_mtp_wrapper(str(root), profile, device=torch.device("cpu"), dtype=dtype)
        return
    wrapper, linears, source_count = _load_exact_mtp_wrapper(
        str(root), profile, device=torch.device("cpu"), dtype=dtype,
    )
    assert source_count == len(raw)
    assert "mtp.fc" in linears
    parameters = dict(wrapper.mtp.named_parameters())
    gate_name = "layers.0.mlp.experts.gate_up_proj"
    down_name = "layers.0.mlp.experts.down_proj"
    for expert in range(experts):
        assert torch.all(parameters[gate_name][expert, :inter] == (expert + 1) / 8)
        assert torch.all(parameters[gate_name][expert, inter:] == (expert + 1) / 4)
        assert torch.all(parameters[down_name][expert] == (expert + 1) / 16)


@pytest.mark.parametrize("scope", ["complete", "missing_expert", "missing_up", "missing_down", "empty"])
def test_probe_mtp_source_admission_preserves_empty_skip(tmp_path, scope):
    from prismaquant.incremental_probe import _load_mtp_source_for_probe

    root, config, profile, raw, _, _ = _write_mtp_source(tmp_path, scope)
    module = profile.build_mtp_module(config)
    if scope not in {"complete", "empty"}:
        with pytest.raises(RuntimeError, match="incremental MTP probe: MTP source/module coverage failure"):
            _load_mtp_source_for_probe(profile, module, str(root))
    else:
        loaded = _load_mtp_source_for_probe(profile, module, str(root))
        assert loaded.keys() == {key.removeprefix("mtp.") for key in raw}
        if scope == "complete":
            assert torch.all(module.fc.weight == 0.125)
        else:
            assert loaded == {}


@pytest.mark.parametrize("scope", ["complete", "missing_up", "missing_down", "missing_fc", "empty", "no_targets"])
def test_mtp_cost_admits_source_before_measurement_and_preserves_no_target_skip(tmp_path, monkeypatch, scope):
    import pickle
    from types import SimpleNamespace
    from prismaquant import incremental_measure_quant_cost as cost

    root, config, _, _, _, _ = _write_mtp_source(tmp_path, scope)
    observed = []

    def measure(module, **kwargs):
        assert torch.all(module.mtp.fc.weight == 0.125)
        observed.append(kwargs["target_names"])
        return {}

    monkeypatch.setattr(cost, "_run_cost_measurement", measure)
    output = tmp_path / "cost.pkl"

    def run():
        cost._run_mtp_cost_shard(
            SimpleNamespace(model=SimpleNamespace(config=config)),
            model_path=str(root if scope != "no_targets" else tmp_path / "absent"),
            linear_include=r"^mtp\.",
            probe_stats={} if scope == "no_targets" else {"mtp.fc": {}},
            act_cache=None, specs=[], device="cpu", dtype=torch.bfloat16,
            mode="unbatched", chunk_size=1, h_detail=None,
            output_path=str(output), model_name=str(root), probe_path="probe.pkl",
        )

    if scope not in {"complete", "no_targets"}:
        with pytest.raises(RuntimeError, match="incremental MTP cost: MTP source/module coverage failure"):
            run()
        assert observed == []
        assert not output.exists()
    else:
        run()
        assert observed == ([{"mtp.fc"}] if scope == "complete" else [])
        with output.open("rb") as file:
            assert pickle.load(file)["costs"] == {}
