"""Profile validation must check the source weights named in its result."""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile
from prismaquant.model_profiles.validate import _check_mtp
from test_mtp_module_arch import _minimal_moe_text_config


@pytest.mark.parametrize("scope", [
    "complete", "missing_fc", "missing_packed_parent", "empty", "unmatched",
    "missing_up_slice", "missing_file", "malformed",
])
def test_profile_mtp_validation_checks_actual_source_coverage(tmp_path, monkeypatch, scope):
    from transformers import AutoConfig

    config = _minimal_moe_text_config()
    profile = Qwen3_5Profile()
    module = profile.build_mtp_module(config)
    raw = {"mtp." + name: torch.full_like(tensor, 0.125)
           for name, tensor in module.state_dict().items()}
    if scope == "missing_fc":
        del raw["mtp.fc.weight"]
    elif scope == "missing_packed_parent":
        del raw["mtp.layers.0.mlp.experts.down_proj"]
    elif scope == "empty":
        raw.clear()
    elif scope == "unmatched":
        raw["mtp.not_a_module.weight"] = torch.zeros(1)
    elif scope == "missing_up_slice":
        gate_up = raw.pop("mtp.layers.0.mlp.experts.gate_up_proj")
        inter = gate_up.shape[1] // 2
        for expert in range(gate_up.shape[0]):
            raw[f"mtp.layers.0.mlp.experts.{expert}.gate_proj.weight"] = gate_up[expert, :inter].clone()
            if expert != gate_up.shape[0] - 1:
                raw[f"mtp.layers.0.mlp.experts.{expert}.up_proj.weight"] = gate_up[expert, inter:].clone()
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(config.to_json_string())
    save_file(raw, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in raw},
    }))
    if scope == "missing_file":
        (root / "model.safetensors").unlink()
    elif scope == "malformed":
        (root / "model.safetensors").write_bytes(b"not a safetensors container")
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)

    result = _check_mtp(profile, config.to_dict(), str(root))
    if scope == "complete":
        assert result.ok, str(result)
    else:
        assert not result.ok, (
            f"MTP source validator accepted {scope}: {result}; "
            "constructing a module does not establish source coverage"
        )
        assert "MTP source load failed" in result.detail


def test_profile_validation_preserves_no_mtp_skip(tmp_path):
    from prismaquant.model_profiles import DefaultProfile

    result = _check_mtp(DefaultProfile(), {}, str(tmp_path / "absent"))
    assert result.ok, str(result)
    assert "skipped" in result.detail
