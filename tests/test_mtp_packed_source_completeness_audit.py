"""The actual exact-wrapper gate must refuse partially loaded packed experts."""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile
from prismaquant.mtp_production_cache import _load_exact_mtp_wrapper
from test_mtp_module_arch import _minimal_moe_text_config


@pytest.mark.parametrize("scope", ["complete", "missing_expert", "missing_up", "missing_down"])
def test_exact_mtp_wrapper_requires_every_packed_expert_source_slice(tmp_path, monkeypatch, scope):
    from transformers import AutoConfig

    config = _minimal_moe_text_config()
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
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(config.to_json_string())
    save_file(raw, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in raw},
    }))
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)

    if scope != "complete":
        with pytest.raises(RuntimeError, match="MTP source/module coverage failure"):
            _load_exact_mtp_wrapper(str(root), profile, device=torch.device("cpu"), dtype=torch.bfloat16)
        return
    wrapper, linears, source_count = _load_exact_mtp_wrapper(
        str(root), profile, device=torch.device("cpu"), dtype=torch.bfloat16,
    )
    assert source_count == len(raw)
    assert "mtp.fc" in linears
    parameters = dict(wrapper.mtp.named_parameters())
    for expert in range(experts):
        assert torch.all(parameters[gate_name][expert, :inter] == (expert + 1) / 8)
        assert torch.all(parameters[gate_name][expert, inter:] == (expert + 1) / 4)
        assert torch.all(parameters[down_name][expert] == (expert + 1) / 16)
