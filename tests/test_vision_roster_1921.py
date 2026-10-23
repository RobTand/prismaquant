"""Audit the quantizable vision/merger roster with shape and name provenance (#2570).

Covers GLM-5-next, Qwen3.5 MoE, Qwen3.5 dense and Gemma4: each roster lists
every quantizable vision and merger Linear with its exact source shape, its
name source and its admitting visual root, while embedding, conv, norm,
router and fixed parts stay BF16 by name. Census role tags (`vis_*`,
`merger_*`) never admit. Qwen3 text-only has no visual roots. Body, audio
and MTP names never enter the vision roster.
"""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.allocator import VISUAL_NON_LINEAR_NAME_RE, _is_visual_linear
from prismaquant.model_profiles.registry import detect_profile
from prismaquant.model_profiles.structure import ModelStructureSpec, load_structure_spec


def _write_case(tmp_path, model_type, architecture, tensors, *, indexed=False):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": model_type, "architectures": [architecture]}))
    save_file(tensors, str(tmp_path / "model.safetensors"))
    if indexed:
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors" for name in tensors}}))
    return detect_profile(str(tmp_path))


def _w(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


GLM_CONFIG = ("glm5_next", "Glm5NextForConditionalGeneration")
GLM_QUANT = {
    "model.visual.blocks.0.attn.qkv": (96, 32),
    "model.visual.blocks.0.attn.proj": (32, 32),
    "model.visual.blocks.0.mlp.gate_proj": (128, 32),
    "model.visual.blocks.0.mlp.up_proj": (128, 32),
    "model.visual.blocks.0.mlp.down_proj": (32, 128),
    "model.visual.merger.down_proj": (32, 80),
    "model.visual.merger.gate_proj": (80, 32),
    "model.visual.merger.up_proj": (80, 32),
    "model.visual.merger.proj": (32, 32),
}
GLM_OUT = {
    # norm by name (rank-2 proves the name contract, not the rank filter)
    "model.visual.blocks.0.attn.q_norm.weight": (16, 16),
    "model.visual.blocks.0.norm1.weight": (32,),
    "model.visual.blocks.0.attn.qkv.bias": (96,),
    "model.visual.merger.router.weight": (8, 8),
    "model.visual.merger.post_projection_norm.weight": (32,),
    "model.visual.downsample.weight": (8, 8),
    "model.visual.patch_embed.proj.weight": (8, 3, 2, 2, 2),
    "model.visual.lm_head.weight": (8, 8),
    "model.language_model.layers.0.mlp.gate_proj.weight": (16, 16),
    "model.language_model.embed_tokens.weight": (16, 16),
    "lm_head.weight": (16, 16),
    "vis_attn_qkv.weight": (8, 8),
    "merger_fc1.weight": (8, 8),
}

QWEN_QUANT = {
    "model.visual.blocks.0.attn.qkv": (108, 36),
    "model.visual.blocks.0.attn.proj": (36, 36),
    "model.visual.blocks.0.mlp.linear_fc1": (144, 36),
    "model.visual.blocks.0.mlp.linear_fc2": (36, 144),
    "model.visual.merger.linear_fc1": (72, 72),
    "model.visual.merger.linear_fc2": (40, 72),
}
QWEN_OUT = {
    # pos_embed is rank-2 yet an embedding table, not a Linear
    "model.visual.pos_embed.weight": (72, 36),
    "model.visual.blocks.0.norm1.weight": (36,),
    "model.visual.blocks.0.attn.qkv.bias": (108,),
    "model.visual.merger.norm.weight": (36,),
    "model.visual.merger.router.weight": (8, 8),
    "model.visual.patch_embed.proj.weight": (36, 3, 2, 4, 4),
    "model.visual.lm_head.weight": (8, 8),
    "model.language_model.layers.0.mlp.gate_proj.weight": (16, 16),
    "mtp.fc.weight": (16, 16),
    "vis_attn_qkv.weight": (8, 8),
    "merger_fc1.weight": (8, 8),
}

GEMMA_QUANT = {
    "model.vision_tower.vision_model.encoder.layers.0.attn.qkv": (48, 16),
    "model.vision_tower.vision_model.encoder.layers.0.attn.proj": (16, 16),
    "model.vision_tower.vision_model.encoder.layers.0.mlp.fc1": (64, 16),
    "model.vision_tower.vision_model.encoder.layers.0.mlp.fc2": (16, 64),
    "model.vision_tower.vision_model.side_projection": (24, 16),
    "model.embed_vision.projection": (24, 16),
}
GEMMA_OUT = {
    "model.vision_tower.vision_model.pos_embed.weight": (36, 16),
    "model.vision_tower.vision_model.patch_conv.weight": (16, 3, 2, 2),
    "model.vision_tower.vision_model.encoder.layers.0.norm.weight": (16,),
    "model.audio_tower.projection.weight": (24, 16),
    "model.embed_audio.projection.weight": (24, 16),
    "model.language_model.layers.0.mlp.gate_proj.weight": (16, 16),
    "vis_attn_qkv.weight": (8, 8),
    "merger_fc1.weight": (8, 8),
}


def _tensors(quant, out):
    tensors = {f"{name}.weight": _w(*shape) for name, shape in quant.items()}
    for key, shape in out.items():
        tensors[key] = _w(*shape)
    return tensors


def _check_roster(roster, quant, root, name_source):
    assert set(roster) == set(quant)
    for name, (out_features, in_features) in quant.items():
        entry = roster[name]
        assert entry["shape"] == [out_features, in_features]
        assert (entry["out_features"], entry["in_features"]) == (out_features, in_features)
        assert entry["n_params"] == out_features * in_features
        assert entry["name_source"] == name_source
        assert entry["visual_root"] == root
        assert entry["source_dtype"] == "bf16"


def test_glm_roster_lists_shapes_and_sources(tmp_path):
    profile = _write_case(tmp_path, *GLM_CONFIG, _tensors(GLM_QUANT, GLM_OUT))
    assert profile.visual_root_prefixes() == ("model.visual",)
    _check_roster(profile.visual_linear_roster(str(tmp_path)),
                   GLM_QUANT, "model.visual", "source_header")


@pytest.mark.parametrize("config", [
    ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
    ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
], ids=["moe", "dense"])
def test_qwen_moe_and_dense_share_visual_roots(tmp_path, config):
    profile = _write_case(tmp_path, *config, _tensors(QWEN_QUANT, QWEN_OUT))
    assert profile.visual_root_prefixes() == ("model.visual",)
    _check_roster(profile.visual_linear_roster(str(tmp_path)),
                   QWEN_QUANT, "model.visual", "source_header")


def test_gemma_roster_covers_both_roots(tmp_path):
    profile = _write_case(tmp_path, "gemma4", "Gemma4ForConditionalGeneration",
                           _tensors(GEMMA_QUANT, GEMMA_OUT))
    assert profile.visual_root_prefixes() == ("model.vision_tower", "model.embed_vision")
    roster = profile.visual_linear_roster(str(tmp_path))
    _check_roster({k: v for k, v in roster.items()
                   if v["visual_root"] == "model.vision_tower"},
                  {k: v for k, v in GEMMA_QUANT.items() if not k.startswith("model.embed_vision")},
                  "model.vision_tower", "source_header")
    _check_roster({k: v for k, v in roster.items()
                   if v["visual_root"] == "model.embed_vision"},
                  {"model.embed_vision.projection": GEMMA_QUANT["model.embed_vision.projection"]},
                  "model.embed_vision", "source_header")


def test_indexed_checkpoint_reports_index_name_source(tmp_path):
    profile = _write_case(tmp_path, *GLM_CONFIG, _tensors(GLM_QUANT, GLM_OUT), indexed=True)
    _check_roster(profile.visual_linear_roster(str(tmp_path)),
                   GLM_QUANT, "model.visual", "checkpoint_index")


def test_text_only_qwen3_has_empty_roster(tmp_path):
    profile = _write_case(tmp_path, "qwen3", "Qwen3ForCausalLM",
                           {"model.layers.0.mlp.gate_proj.weight": _w(16, 16)})
    assert profile.visual_root_prefixes() == ()
    assert profile.visual_linear_roster(str(tmp_path)) == {}


@pytest.mark.parametrize("name,admitted", [
    ("model.visual.blocks.0.attn.qkv", True),
    ("model.visual.merger.linear_fc1", True),
    ("model.vision_tower.vision_model.encoder.layers.0.attn.qkv", True),
    ("model.embed_vision.projection", True),
    ("model.visual.pos_embed", False),
    ("model.visual.patch_embed.proj", False),
    ("model.visual.blocks.0.norm1", False),
    ("model.visual.blocks.0.attn.q_norm", False),
    ("model.visual.merger.router", False),
    ("model.visual.downsample", False),
    ("model.visual.lm_head", False),
    ("model.vision_tower.vision_model.patch_conv", False),
])
def test_exclusion_contract_by_name(name, admitted):
    assert (VISUAL_NON_LINEAR_NAME_RE.search(name) is None) is admitted


@pytest.mark.parametrize("model_type,architecture", [
    GLM_CONFIG,
    ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
    ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
    ("gemma4", "Gemma4ForConditionalGeneration"),
], ids=["glm", "qwen-moe", "qwen-dense", "gemma"])
def test_role_tags_never_admit_under_declared_roots(tmp_path, model_type, architecture):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": model_type, "architectures": [architecture]}))
    profile = detect_profile(str(tmp_path))
    assert profile.visual_root_prefixes()
    for tag in ("vis_attn_qkv", "merger_fc1", "vis", "merger"):
        assert not _is_visual_linear(tag, profile)


@pytest.mark.parametrize("root", ["vis_attn", "merger_fc", "vis", "merger"])
def test_structure_spec_rejects_role_tag_roots(root):
    with pytest.raises(ValueError, match="role tags"):
        ModelStructureSpec.from_dict({
            "id": "role_tag_probe",
            "match": {},
            "shard_regexes": {
                "visual_config_key": "vision_config",
                "visual_layer_prefix": f"{root}.blocks",
                "visual_root_prefixes": [root],
            },
        })


@pytest.mark.parametrize("spec_id", ["glm5_next", "qwen3_5", "qwen3_5_dense", "gemma4", "qwen3"])
def test_shipped_specs_keep_declared_roots(spec_id):
    spec = load_structure_spec(spec_id)
    assert spec is not None
    if spec_id == "qwen3":
        assert spec.visual_root_prefixes == ()
    else:
        assert spec.visual_root_prefixes
