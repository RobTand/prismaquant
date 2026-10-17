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
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(config.to_json_string())
    save_file(raw, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in raw},
    }))
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)

    result = _check_mtp(profile, config.to_dict(), str(root))
    if scope == "complete":
        assert result.ok, str(result)
    else:
        assert not result.ok, (
            f"MTP source validator accepted {scope}: {result}; "
            "constructing a module does not establish source coverage"
        )
