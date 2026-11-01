"""Native MTP materialization must not emit constructor bytes absent from source."""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import export_native_compressed as exporter
from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile
from test_mtp_module_arch import _minimal_moe_text_config


INITIAL_VALUE = -7.0
GATE = "layers.0.mlp.experts.gate_up_proj"
DOWN = "layers.0.mlp.experts.down_proj"


class _InitializedQwenMtpProfile(Qwen3_5Profile):
    def build_mtp_module(self, text_config):
        module = super().build_mtp_module(text_config)
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.fill_(INITIAL_VALUE)
        return module


@pytest.mark.parametrize("scope", [
    "complete", "missing_expert", "missing_up", "missing_down", "missing_fc",
])
def test_native_mtp_materialization_requires_complete_source(tmp_path, monkeypatch, scope):
    from transformers import AutoConfig

    config = _minimal_moe_text_config()
    profile = _InitializedQwenMtpProfile()
    template = profile.build_mtp_module(config)
    state = template.state_dict()
    experts, two_inter, hidden = state[GATE].shape
    inter = two_inter // 2
    expected = {name: torch.full_like(value, 0.125) for name, value in state.items()}
    raw = {
        "mtp." + name: value.clone()
        for name, value in expected.items() if name not in {GATE, DOWN}
    }
    for expert in range(experts):
        for projection, shape, value in (
            ("gate_proj", (inter, hidden), (expert + 1) / 8),
            ("up_proj", (inter, hidden), (expert + 1) / 4),
            ("down_proj", (hidden, inter), (expert + 1) / 16),
        ):
            if projection == "gate_proj":
                expected[GATE][expert, :inter].fill_(value)
            elif projection == "up_proj":
                expected[GATE][expert, inter:].fill_(value)
            else:
                expected[DOWN][expert].fill_(value)
            if expert == experts - 1 and (
                scope == "missing_expert"
                or scope == "missing_up" and projection == "up_proj"
                or scope == "missing_down" and projection == "down_proj"
            ):
                continue
            raw[f"mtp.layers.0.mlp.experts.{expert}.{projection}.weight"] = torch.full(shape, value)
    if scope == "missing_fc":
        del raw["mtp.fc.weight"]
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(config.to_json_string())
    save_file(raw, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in raw},
    }))
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(exporter, "_PRODUCTION_WEIGHT_CACHE", None)

    def materialize():
        return exporter._materialize_mtp_tensors(
            str(root), {"mtp.fc": "BF16"}, profile=profile,
            bf16_passthrough=set(), hist={}, device="cpu",
        )

    if scope == "complete":
        output = materialize()
        for name, value in expected.items():
            actual = output["mtp." + name]
            assert actual.dtype == torch.bfloat16
            assert torch.equal(actual, value.to(torch.bfloat16)), name
        return

    with pytest.raises(RuntimeError, match="MTP source/module coverage failure"):
        output = materialize()
        if scope == "missing_fc":
            name = "mtp.fc.weight"
            witness = output[name]
        elif scope == "missing_down":
            name = "mtp." + DOWN
            witness = output[name][experts - 1]
        else:
            name = "mtp." + GATE
            witness = output[name][experts - 1, inter:]
        assert witness.numel() > 0
        assert torch.all(witness == INITIAL_VALUE)
        pytest.fail(
            f"native MTP materialization returned {name} shape={tuple(output[name].shape)} "
            f"with {witness.numel()} BF16 constructor values {INITIAL_VALUE} absent from "
            f"checkpoint source ({scope}); complete source coverage must refuse before output"
        )
