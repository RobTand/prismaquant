"""Compare real resident HF and packed-restore values with serialized source."""
from __future__ import annotations

import json
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file

from prismaquant.build_rtn_cache import stage_multimodal
from prismaquant.layer_streaming import fill_packed_experts_from_source
from prismaquant.model_profiles import detect_profile
from prismaquant.model_runtime import initialize_model_runtime
from test_mtp_module_arch import _minimal_moe_text_config


@pytest.mark.parametrize("source_kind", ["bf16", "block_fp8"])
def test_actual_resident_packed_values_match_serialized_decoded_source(tmp_path, source_kind):
    initialize_model_runtime()
    from transformers import AutoModelForCausalLM

    config = _minimal_moe_text_config()
    config.architectures = ["Qwen3_5MoeForCausalLM"]
    template = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    raw = {}
    source_parts = {}
    for name, tensor in template.state_dict().items():
        if not (name.endswith(".experts.gate_up_proj") or name.endswith(".experts.down_proj")):
            raw[name] = torch.full_like(tensor, 0.125)
            continue
        prefix, projection = name.rsplit(".", 1)
        source_parts[name] = []
        for expert in range(tensor.shape[0]):
            if projection == "gate_up_proj":
                inter = tensor.shape[1] // 2
                projections = (
                    ("gate_proj", (inter, tensor.shape[2]), slice(0, inter), 0.5, expert + 2),
                    ("up_proj", (inter, tensor.shape[2]), slice(inter, 2 * inter), 1.5, expert + 6),
                )
            else:
                projections = (("down_proj", tuple(tensor.shape[1:]), slice(None), 0.25, expert + 10),)
            for leaf, shape, rows, code, scale in projections:
                key = f"{prefix}.{expert}.{leaf}.weight"
                source_parts[name].append((expert, rows, key))
                if source_kind == "block_fp8":
                    raw[key] = torch.full(shape, code).to(torch.float8_e4m3fn)
                    raw[key + "_scale_inv"] = torch.full((1, 1), float(scale))
                else:
                    raw[key] = torch.full(shape, code * scale, dtype=torch.bfloat16)
    assert source_parts, "real HF fixture did not construct packed expert Parameters"
    root = tmp_path / "model"
    root.mkdir()
    payload = config.to_dict()
    if source_kind == "block_fp8":
        payload["quantization_config"] = {
            "quant_method": "fp8", "weight_block_size": [128, 128],
        }
    (root / "config.json").write_text(json.dumps(payload))
    save_file(raw, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in raw},
    }))
    serialized = load_file(root / "model.safetensors")
    expected = {}
    for name, parts in source_parts.items():
        expected[name] = torch.empty_like(template.state_dict()[name])
        for expert, rows, key in parts:
            decoded = serialized[key].float()
            if source_kind == "block_fp8":
                scale = serialized[key + "_scale_inv"]
                assert tuple(scale.shape) == (1, 1)
                assert max(decoded.shape) <= 128
                decoded = decoded * scale[0, 0]
            expected[name][expert, rows] = decoded.to(torch.bfloat16)
    del template, serialized, raw
    profile = detect_profile(str(root))
    staged, cleanup = stage_multimodal(str(root))
    try:
        loaded, info = AutoModelForCausalLM.from_pretrained(
            staged, torch_dtype=torch.bfloat16, trust_remote_code=True,
            local_files_only=True, output_loading_info=True,
        )
        parameters = dict(loaded.named_parameters())
        post_hf = {name: parameters[name].detach().clone() for name in expected}
        filled = fill_packed_experts_from_source(loaded, str(root), profile)
        mismatches = []
        for name, wanted in expected.items():
            actual = parameters[name].detach()
            if not torch.equal(actual, wanted):
                mismatches.append(
                    f"{name}: post_hf_first={post_hf[name].flatten()[0].item()} "
                    f"post_helper_first={actual.flatten()[0].item()} "
                    f"decoded_source_first={wanted.flatten()[0].item()} "
                    f"post_hf_max_error={(post_hf[name].float() - wanted.float()).abs().max().item()} "
                    f"post_helper_max_error={(actual.float() - wanted.float()).abs().max().item()} "
                    f"source_keys={[key for _, _, key in source_parts[name]]}"
                )
        assert not mismatches, (
            f"{source_kind} actual resident values differ from serialized decoded source; "
            f"filled={filled}, missing_keys={info['missing_keys']}:\n" + "\n".join(mismatches)
        )
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup)
