"""Integration: streaming forward through REAL Qwen3.5 hybrid layers.

The fakes in test_linear_attention_mask_routing.py pin the contract; this
file pins it against transformers' actual Qwen3_5 modules (tiny random-weight
hybrid, CPU) — the family whose transformers hybrid refactors (`.block_type` rename in
5.13, removal of the `apply_mask_to_padding_states` shape guard in 5.15)
motivated the fix. The forward-path tests run on every transformers version that ships
qwen3_5 (the locked 5.8 environment included — same contract, pre-refactor
attribute names); the two assertions specific to the 5.15 refactor
(`.block_type`/child `layer_type`, guardless ``apply_mask_to_padding_states``)
are version-gated.
"""
import pytest
import torch

pytest.importorskip(
    "transformers.models.qwen3_5",
    reason="qwen3_5 modeling not available in this transformers version",
)

import transformers
from packaging.version import parse as _parse_version

_TV = _parse_version(transformers.__version__)
_HAS_BLOCK_TYPE = _TV >= _parse_version("5.13.0")   # .block_type + child attrs
_GUARDLESS_MASK = _TV >= _parse_version("5.15.0")   # padding-mask shape guard removed

from transformers.models.qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)

from prismaquant.layer_streaming import (
    _call_layer,
    _compute_attention_mask,
    _compute_position_embeddings,
    _layer_attention_type,
)
from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile


@pytest.fixture(autouse=True)
def _force_pure_torch_kernels(monkeypatch):
    """Keep the CPU integration CPU-safe in CUDA-toolkit environments.

    transformers selects `causal_conv1d`/`fla` extensions by package
    availability, not tensor device — in a cu130 stack the installed CUDA
    extension gets picked for CPU tensors. Transformers <=5.14 binds raw
    optional-extension globals/classes onto each DeltaNet instance at
    construction; 5.15 wraps module-level torch fallbacks and dispatches
    through them.

    Select the package-provided PyTorch implementation for both layouts
    before `_tiny_hybrid()` constructs its layers."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as m

    # <=5.14: replace the raw optional-extension globals that __init__ copies
    # onto Qwen3_5GatedDeltaNet. A None full-convolution function selects the
    # module's nn.Conv1d branch; the other globals have explicit torch twins.
    if hasattr(m, "torch_causal_conv1d_update"):
        monkeypatch.setattr(m, "causal_conv1d_fn", None)
        monkeypatch.setattr(m, "causal_conv1d_update",
                            m.torch_causal_conv1d_update)
        monkeypatch.setattr(m, "chunk_gated_delta_rule",
                            m.torch_chunk_gated_delta_rule)
        monkeypatch.setattr(m, "fused_recurrent_gated_delta_rule",
                            m.torch_recurrent_gated_delta_rule)
        monkeypatch.setattr(m, "FusedRMSNormGated", None)
        return

    # >=5.15: the module-level fallbacks are kernel-dispatch wrappers whose
    # undecorated PyTorch implementation remains reachable via __wrapped__.
    for name in ("causal_conv1d_fn", "causal_conv1d_update",
                 "torch_chunk_gated_delta_rule",
                 "torch_recurrent_gated_delta_rule"):
        fn = getattr(m, name, None)
        if fn is None:
            continue
        inner = fn
        while hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__
        if inner is not fn:
            monkeypatch.setattr(m, name, inner)


def _tiny_hybrid():
    cfg = Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        layer_types=["linear_attention", "full_attention"],
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        vocab_size=128, max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    layers = [Qwen3_5DecoderLayer(cfg, i) for i in range(2)]
    rotary = Qwen3_5TextRotaryEmbedding(cfg)

    class _Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = cfg
            self.rotary_emb = rotary

    return cfg, _Base(), layers, rotary


def _stream(base, layers, rotary, hidden, attention_mask):
    position_ids = torch.arange(hidden.size(1)).unsqueeze(0).expand(
        hidden.size(0), -1)
    masks = _compute_attention_mask(base, hidden, position_ids,
                                    attention_mask=attention_mask)
    pe = _compute_position_embeddings(base, hidden, position_ids, Qwen3_5Profile())
    expected = rotary(hidden, position_ids.unsqueeze(0).expand(3, -1, -1))
    for actual, reference in zip(pe, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    out = hidden
    for layer in layers:
        out = _call_layer(layer, out, position_embeddings=pe,
                          attention_mask=masks, position_ids=position_ids)
    return masks, out


def test_real_layers_resolve():
    # Passes pre- and post-refactor: `.layer_type` up to 5.12, `.block_type`
    # (plus the recurrent child's own attributes) from 5.13 on.
    _, _, layers, _ = _tiny_hybrid()
    assert _layer_attention_type(layers[0]) == "linear_attention"
    assert _layer_attention_type(layers[1]) == "full_attention"


@pytest.mark.skipif(not _HAS_BLOCK_TYPE,
                    reason="`.block_type` rename landed in transformers 5.13")
def test_post_refactor_child_carries_fallback_attributes():
    _, _, layers, _ = _tiny_hybrid()
    assert layers[0].block_type == "linear_attention"
    assert layers[0].linear_attn.layer_type == "linear_attention"
    assert layers[0].linear_attn.layer_idx == 0


def test_streaming_forward_left_padded_batch():
    _, base, layers, rotary = _tiny_hybrid()
    hidden = torch.randn(2, 6, 64)
    pad = torch.tensor([[0, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1]])

    masks, out = _stream(base, layers, rotary, hidden, pad)

    lin = masks["linear_attention"]
    assert lin is not None and lin.ndim == 2 and torch.equal(lin, pad)
    assert out.shape == hidden.shape
    assert bool(torch.isfinite(out).all())


def test_streaming_forward_unpadded_batch():
    _, base, layers, rotary = _tiny_hybrid()
    hidden = torch.randn(1, 6, 64)

    # all-ones → linear_attention mask is None; forward must still work
    masks, out = _stream(base, layers, rotary, hidden, torch.ones(1, 6))

    assert masks["linear_attention"] is None
    assert out.shape == hidden.shape
    assert bool(torch.isfinite(out).all())


def test_explicit_rotary_axes_are_preserved():
    _, base, _, rotary = _tiny_hybrid()
    hidden = torch.randn(2, 6, 64)
    positions = torch.arange(36).reshape(3, 2, 6)
    expected = rotary(hidden, positions)
    actual = _compute_position_embeddings(base, hidden, positions, Qwen3_5Profile())
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(6,), (2, 2, 6), (4, 2, 6)])
def test_invalid_rotary_axes_refuse(shape):
    with pytest.raises(ValueError, match="Qwen rotary positions"):
        Qwen3_5Profile().rotary_position_ids(torch.zeros(shape, dtype=torch.long))


@pytest.mark.skipif(
    not _GUARDLESS_MASK,
    reason="pre-5.15 apply_mask_to_padding_states still shape-guards "
           "non-2D masks, so the dense mask is silently ignored there")
def test_dense_mask_crashes_real_linear_layer():
    # Documents the pre-fix failure mode this branch removes: a dense
    # [1, 1, T, T] additive mask fed to a real GatedDeltaNet layer
    # broadcasts against hidden_states and raises a trailing-dim mismatch.
    _, base, layers, rotary = _tiny_hybrid()
    hidden = torch.randn(1, 6, 64)
    position_ids = torch.arange(6).unsqueeze(0)
    pe = _compute_position_embeddings(base, hidden, position_ids, Qwen3_5Profile())
    dense = torch.zeros(1, 1, 6, 6)
    with pytest.raises(RuntimeError, match="must match the size"):
        layers[0](hidden_states=hidden, attention_mask=dense,
                  position_ids=position_ids, past_key_values=None,
                  use_cache=False, position_embeddings=pe)
