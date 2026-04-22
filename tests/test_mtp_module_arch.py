"""Tests for MtpModule's dense-vs-MoE decoder selection.

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
"""
from __future__ import annotations

import pytest


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

    from prismaquant.mtp_module import MtpModule

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

    from prismaquant.mtp_module import MtpModule

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
    from prismaquant.mtp_module import MtpModule

    dense = MtpModule(_minimal_dense_text_config())
    moe = MtpModule(_minimal_moe_text_config())

    for name in ("fc", "layers", "norm", "pre_fc_norm_embedding",
                 "pre_fc_norm_hidden"):
        assert hasattr(dense, name), f"dense MtpModule missing {name}"
        assert hasattr(moe, name), f"moe MtpModule missing {name}"
