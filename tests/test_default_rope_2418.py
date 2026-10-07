"""Default RoPE accepts upstream parameters and preserves legacy arithmetic."""
from types import SimpleNamespace
import pytest
import torch
import prismaquant
from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


def test_actual_llama_config_uses_the_upstream_representation():
    config = LlamaConfig(hidden_size=32, num_attention_heads=4,
                         rope_parameters={"rope_type": "default", "rope_theta": 32123.0})
    assert not hasattr(config, "rope_theta")
    actual, scaling = ROPE_INIT_FUNCTIONS["default"](config, device="cpu")
    expected, expected_scaling = LlamaRotaryEmbedding.compute_default_rope_parameters(config, device="cpu")
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert scaling == expected_scaling == 1.0


def test_legacy_rope_keeps_the_original_operation_order():
    config = SimpleNamespace(rope_theta=500000.0, partial_rotary_factor=0.5,
                             head_dim=128, hidden_size=4096, num_attention_heads=32)
    dim = int(config.head_dim*config.partial_rotary_factor)
    expected = 1.0/(config.rope_theta**(torch.arange(0, dim, 2, dtype=torch.int64)
                                     .to(dtype=torch.float32, device="cpu")/dim))
    actual, scaling = ROPE_INIT_FUNCTIONS["default"](config, device="cpu")
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert scaling == 1.0


def test_missing_theta_is_not_a_fallback_value():
    config = LlamaConfig(hidden_size=32, num_attention_heads=4,
                         rope_parameters={"rope_type": "default", "rope_theta": 32123.0})
    del config.rope_parameters["rope_theta"]
    with pytest.raises((KeyError, ValueError)):
        ROPE_INIT_FUNCTIONS["default"](config, device="cpu")
