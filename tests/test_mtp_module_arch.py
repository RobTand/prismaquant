"""Tests for the Qwen3.5/3.6 MTP module's dense-vs-MoE decoder selection,
and for the profile route that replaced the top-level `mtp_module.py`.

The MTP module mirrors vLLM's Qwen3_5MultiTokenPredictor but is built from
HF primitives so PrismaQuant's probe / cost / export can attach Fisher
hooks and autograd. Dense Qwen3.5/3.6 and MoE Qwen3.5/3.6 share the same
outer shape (fc + 1 decoder layer + norms) but use different decoder
classes under the hood — the MoE decoder touches `num_experts_per_tok`
eagerly in __init__, which dense configs don't define.

These tests pin:
  - Dense text_config → Qwen3_5DecoderLayer (from transformers.models.qwen3_5).
  - MoE text_config   → Qwen3_5MoeDecoderLayer (from transformers.models.qwen3_5_moe).
  - MtpModule construction never crashes on either config shape.
  - The R12 move (2026-07-30) was VERBATIM: constructing through
    `profile.build_mtp_module()` yields exactly the parameter-name set the
    deleted `prismaquant/mtp_module.MtpModule` produced, for both the dense
    and the MoE profile. Those names are the allocator's recipe names once
    wrapped in a parent named `mtp`, so any drift silently mis-keys probe
    stats, cost entries and exported tensors.
  - `build_mtp_module`'s naming contract holds under that wrapper.
"""
from __future__ import annotations

import pytest

# Parameter names produced by `prismaquant/mtp_module.py::MtpModule` on the
# two synthetic configs below, captured from the pre-move tree at HEAD
# 97af8fa. Frozen deliberately: the point of the gate is that the body
# moved verbatim, so these must not be regenerated from the new code.
GOLDEN_DENSE_PARAM_NAMES = {
    "fc.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.mlp.down_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.k_norm.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
}

GOLDEN_MOE_PARAM_NAMES = {
    "fc.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.mlp.experts.down_proj",
    "layers.0.mlp.experts.gate_up_proj",
    "layers.0.mlp.gate.weight",
    "layers.0.mlp.shared_expert.down_proj.weight",
    "layers.0.mlp.shared_expert.gate_proj.weight",
    "layers.0.mlp.shared_expert.up_proj.weight",
    "layers.0.mlp.shared_expert_gate.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.k_norm.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
}


def _minimal_dense_text_config():
    """Dense Qwen3.5/3.6 text_config — no num_experts, no num_experts_per_tok."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    return Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        attn_output_gate=True,
        tie_word_embeddings=False,
        layer_types=["full_attention", "full_attention"],
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        mtp_num_hidden_layers=1,
        partial_rotary_factor=0.25,
    )


def _minimal_moe_text_config():
    """MoE Qwen3.5/3.6 text_config — has num_experts and num_experts_per_tok."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    return Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        attn_output_gate=True,
        tie_word_embeddings=False,
        layer_types=["full_attention", "full_attention"],
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        mtp_num_hidden_layers=1,
        partial_rotary_factor=0.25,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
    )


def test_mtp_module_uses_dense_decoder_for_dense_config():
    """Dense text_config should produce a Qwen3_5DecoderLayer (NOT the Moe
    variant) as layers[0]. Regression test for the crash where the
    hardcoded Moe import failed with 'Qwen3_5TextConfig has no attribute
    num_experts_per_tok' when probing Qwen3.6-27B dense."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    from prismaquant.model_profiles.qwen3_5 import MtpModule

    cfg = _minimal_dense_text_config()
    mtp = MtpModule(cfg)

    assert len(mtp.layers) == 1
    assert isinstance(mtp.layers[0], Qwen3_5DecoderLayer), (
        f"expected Qwen3_5DecoderLayer (dense), got "
        f"{type(mtp.layers[0]).__name__}"
    )


def test_mtp_module_uses_moe_decoder_for_moe_config():
    """MoE text_config should route to Qwen3_5MoeDecoderLayer — the 35B-A3B
    path that was already working must not regress from the dense-aware
    refactor."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
    )

    from prismaquant.model_profiles.qwen3_5 import MtpModule

    cfg = _minimal_moe_text_config()
    mtp = MtpModule(cfg)

    assert len(mtp.layers) == 1
    assert isinstance(mtp.layers[0], Qwen3_5MoeDecoderLayer), (
        f"expected Qwen3_5MoeDecoderLayer, got "
        f"{type(mtp.layers[0]).__name__}"
    )


def test_mtp_module_shape_is_the_same_for_dense_and_moe():
    """Outer shape (fc, layers, norm, pre_fc_norm_embedding, pre_fc_norm_hidden)
    is arch-independent — only the inner DecoderLayer differs."""
    from prismaquant.model_profiles.qwen3_5 import MtpModule

    dense = MtpModule(_minimal_dense_text_config())
    moe = MtpModule(_minimal_moe_text_config())

    for name in ("fc", "layers", "norm", "pre_fc_norm_embedding",
                 "pre_fc_norm_hidden"):
        assert hasattr(dense, name), f"dense MtpModule missing {name}"
        assert hasattr(moe, name), f"moe MtpModule missing {name}"


# --------------------------------------------------------- the profile route
#
# R12 (2026-07-30): probe / cost / export used to import
# `prismaquant.mtp_module` directly; they now go through
# `profile.build_mtp_module()`. The tests below are the verbatim-move gate.


def test_moe_profile_build_mtp_module_matches_golden_param_names():
    """`Qwen3_5Profile.build_mtp_module` must reproduce the deleted
    `mtp_module.MtpModule` layout exactly — these names ARE the recipe
    names the allocator assigned to shipped 27B/35B artifacts."""
    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    mtp = Qwen3_5Profile().build_mtp_module(_minimal_moe_text_config())
    assert mtp is not None
    names = {n for n, _ in mtp.named_parameters()}
    assert names == GOLDEN_MOE_PARAM_NAMES, (
        f"missing={sorted(GOLDEN_MOE_PARAM_NAMES - names)} "
        f"unexpected={sorted(names - GOLDEN_MOE_PARAM_NAMES)}"
    )


def test_dense_profile_build_mtp_module_matches_golden_param_names():
    """The dense profile inherits the same `MtpModule`, which picks the
    dense decoder from the config. This is what production already did
    (it imported `MtpModule` unconditionally); the dense profile's own
    near-copy was dead code and was removed with the move."""
    from prismaquant.model_profiles.qwen3_5_dense import Qwen3_5DenseProfile

    mtp = Qwen3_5DenseProfile().build_mtp_module(_minimal_dense_text_config())
    assert mtp is not None
    names = {n for n, _ in mtp.named_parameters()}
    assert names == GOLDEN_DENSE_PARAM_NAMES, (
        f"missing={sorted(GOLDEN_DENSE_PARAM_NAMES - names)} "
        f"unexpected={sorted(names - GOLDEN_DENSE_PARAM_NAMES)}"
    )


def test_wrapped_names_are_the_recipe_names():
    """`build_mtp_module`'s stated contract: wrapped in a parent named
    `mtp`, qualified names equal the allocator's recipe names. Probe,
    cost and export all key straight into `assignment` / probe stats by
    these, so a layout change silently measures and exports nothing."""
    import torch.nn as nn

    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    wrapper = nn.Module()
    wrapper.add_module(
        "mtp", Qwen3_5Profile().build_mtp_module(_minimal_moe_text_config()))
    names = {n for n, _ in wrapper.named_parameters()}
    assert "mtp.fc.weight" in names
    assert "mtp.layers.0.self_attn.q_proj.weight" in names
    assert names == {f"mtp.{n}" for n in GOLDEN_MOE_PARAM_NAMES}


def test_mtp_source_prefix_default_and_spec_override():
    """The new fourth accessor. Qwen3.5/3.6 keep the `mtp.` default;
    architectures whose MTP is body-indexed declare has_mtp() False and
    ship the block through passthrough instead."""
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile
    from prismaquant.model_profiles.hy_v3 import HyV3Profile
    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    assert Qwen3_5Profile().mtp_source_prefix() == "mtp."

    # hy_v3 / DSv4: MTP is not probed; the block ships verbatim.
    assert HyV3Profile().has_mtp() is False
    assert "model.layers.80." in HyV3Profile().source_passthrough_prefixes()
    assert DeepseekV4Profile().has_mtp() is False
    assert "mtp." in DeepseekV4Profile().source_passthrough_prefixes()


def test_load_mtp_state_dict_folds_packed_experts():
    """The `_load_into_mtp` logic moved onto the base profile. Per-expert
    checkpoint keys must land in the packed 3D Parameters (gate low half,
    up high half, down per-expert row) and NOT be reported missing."""
    import torch

    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    profile = Qwen3_5Profile()
    mtp = profile.build_mtp_module(_minimal_moe_text_config())
    packed_gate_up = dict(mtp.named_parameters())[
        "layers.0.mlp.experts.gate_up_proj"]
    n_experts, two_i, hidden = packed_gate_up.shape
    inter = two_i // 2

    raw = {
        "layers.0.mlp.experts.1.gate_proj.weight":
            torch.full((inter, hidden), 0.25),
        "layers.0.mlp.experts.1.up_proj.weight":
            torch.full((inter, hidden), 0.75),
        "layers.0.mlp.experts.1.down_proj.weight":
            torch.full((hidden, inter), 0.5),
    }
    missing, extra = profile.load_mtp_state_dict(mtp, raw)
    assert missing == [], missing
    assert "layers.0.mlp.experts.gate_up_proj" in extra
    assert "layers.0.mlp.experts.down_proj" in extra

    params = dict(mtp.named_parameters())
    gate_up = params["layers.0.mlp.experts.gate_up_proj"]
    down = params["layers.0.mlp.experts.down_proj"]
    assert torch.allclose(gate_up[1, :inter], torch.full((inter, hidden), 0.25))
    assert torch.allclose(gate_up[1, inter:], torch.full((inter, hidden), 0.75))
    assert torch.allclose(down[1], torch.full((hidden, inter), 0.5))
    # Untouched expert 0 must not have been written.
    assert not torch.allclose(gate_up[0, :inter],
                              torch.full((inter, hidden), 0.25))


def test_load_mtp_state_dict_reports_unmatched_keys():
    from prismaquant.model_profiles.base import load_complete_mtp_state_dict
    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    import torch

    profile = Qwen3_5Profile()
    mtp = profile.build_mtp_module(_minimal_dense_text_config())
    missing, _ = profile.load_mtp_state_dict(
        mtp, {"not.a.real.key": torch.zeros(1)})
    assert missing == ["not.a.real.key"]
    with pytest.raises(RuntimeError, match="unmatched_source=.*not.a.real.key"):
        load_complete_mtp_state_dict(
            profile, mtp, {"not.a.real.key": torch.zeros(1)}, where="unmatched control",
        )


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float32"])
def test_load_mtp_state_dict_accepts_complete_exact_packed_zero_weights(dtype_name):
    import torch
    from prismaquant.model_profiles.base import load_complete_mtp_state_dict
    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    profile = Qwen3_5Profile()
    module = profile.build_mtp_module(_minimal_moe_text_config()).to(getattr(torch, dtype_name))
    raw = {name: torch.zeros_like(tensor) for name, tensor in module.state_dict().items()}
    assert profile.load_mtp_state_dict(module, raw) == ([], [])
    load_complete_mtp_state_dict(profile, module, raw, where="exact packed control")
    assert all(torch.count_nonzero(tensor).item() == 0 for tensor in module.state_dict().values())


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
def test_load_mtp_state_dict_refuses_broadcastable_incomplete_projection(projection):
    import torch
    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    profile = Qwen3_5Profile()
    module = profile.build_mtp_module(_minimal_moe_text_config())
    gate_up = dict(module.named_parameters())["layers.0.mlp.experts.gate_up_proj"]
    inter, hidden = gate_up.shape[1] // 2, gate_up.shape[2]
    width = inter if projection == "down_proj" else hidden
    raw = {f"layers.0.mlp.experts.0.{projection}.weight": torch.full((1, width), 0.25)}
    with pytest.raises(RuntimeError, match="MTP source/module coverage failure.*shape=.*expected="):
        profile.load_mtp_state_dict(module, raw)


def test_load_mtp_state_dict_refuses_expert_outside_packed_shape():
    import torch
    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    profile = Qwen3_5Profile()
    module = profile.build_mtp_module(_minimal_moe_text_config())
    gate_up = dict(module.named_parameters())["layers.0.mlp.experts.gate_up_proj"]
    experts, two_inter, hidden = gate_up.shape
    raw = {f"layers.0.mlp.experts.{experts}.gate_proj.weight": torch.zeros(two_inter // 2, hidden)}
    with pytest.raises(RuntimeError, match="MTP source/module coverage failure.*cannot address packed shape"):
        profile.load_mtp_state_dict(module, raw)


def test_read_mtp_source_state_dict_strips_the_prefix(tmp_path):
    """Generic reader: keyed on `mtp_source_prefix()`, opens only the
    shards that hold MTP keys, and strips the prefix so the keys match
    `build_mtp_module()`'s layout."""
    import json

    import torch
    from safetensors.torch import save_file

    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    save_file({"mtp.fc.weight": torch.zeros(2, 4)},
              str(tmp_path / "a.safetensors"))
    save_file({"model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2)},
              str(tmp_path / "b.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "mtp.fc.weight": "a.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "b.safetensors",
        }}))

    raw = Qwen3_5Profile().read_mtp_source_state_dict(str(tmp_path))
    assert set(raw) == {"fc.weight"}


def test_read_mtp_source_state_dict_empty_when_finetune_stripped_mtp(tmp_path):
    """Finetunes that inherit num_nextn_predict_layers but drop the
    weights must yield {} rather than raise — the probe writes an empty
    shard pickle off that signal (PR #1)."""
    import json

    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model.layers.0.self_attn.q_proj.weight": "b.safetensors"}}))
    assert Qwen3_5Profile().read_mtp_source_state_dict(str(tmp_path)) == {}
