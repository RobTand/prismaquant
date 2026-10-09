"""Meta-skeleton rotary init must cover NESTED rotary instances.

The faithful DSv4 forward gives each compressor and indexer its own
``rotary_emb`` (compress-theta RoPE with per-call positions). A skeleton
built under ``torch.device("meta")`` leaves those buffers on meta, and
the first CSA forward dies with "Cannot copy out of meta tensor"
(probe attempt 4, 2026-08-09) unless ``_init_rotary_inplace`` →
``DeepseekV4Profile.init_rotaries(base_model=...)`` walks the skeleton
and materializes every instance.
"""
import pytest

torch = pytest.importorskip("torch")

# Own the DSv4 namespace in a shared session: an earlier test that imports
# transformers' native deepseek_v4 makes register_deepseek_v4() refuse (#2279).
# The refusal itself stays visible in its own controls.
pytestmark = pytest.mark.own_process


def _tiny_cfg():
    from prismaquant.vendored import register_deepseek_v4

    register_deepseek_v4()
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config,
    )

    return DeepseekV4Config(
        num_hidden_layers=4,
        compress_ratios=[0, 0, 4, 128],
        hidden_size=64,
        num_attention_heads=4,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=2,
        index_n_heads=4,
        index_head_dim=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        intermediate_size=64,
        vocab_size=128,
        num_hash_layers=1,
    )


def test_meta_skeleton_materializes_all_rotary_buffers():
    import transformers

    from prismaquant.streaming_model import _init_rotary_inplace

    cfg = _tiny_cfg()
    with torch.device("meta"):
        wrapper = transformers.AutoModelForCausalLM.from_config(cfg)
    # The streaming stack operates on the BASE model (base_prefix
    # "model"); _get_rotary looks for `.rotary_emb` there.
    model = wrapper.model

    # Sanity: the skeleton must actually contain nested rotary instances
    # (a compressor/indexer regression would silently weaken this test).
    nested = [
        n for n, m in model.named_modules()
        if type(m).__name__ == "DeepseekV4RotaryEmbedding"
        and ("compressor" in n or "indexer" in n)
    ]
    assert nested, "expected nested compressor/indexer rotary instances"

    _init_rotary_inplace(model, torch.device("cpu"), torch.bfloat16)

    def _stale():
        return [
            f"{n}.{bn}"
            for n, m in model.named_modules()
            if type(m).__name__ == "DeepseekV4RotaryEmbedding"
            for bn, b in m.named_buffers(recurse=False)
            if b.is_meta and bn.endswith("inv_freq")
        ]

    assert not _stale(), f"rotary buffers left on meta: {_stale()}"

    # Eviction symmetry: _unload must NOT meta-ize derived non-persistent
    # buffers — install can never restore them (they are not in the
    # checkpoint), so an evicted-then-reinstalled layer would crash at
    # its first rotary use (probe attempt 5, phase-3 reverse sweep).
    from prismaquant.layer_streaming import _unload

    _unload(model, ["layers.2."])
    assert not _stale(), (
        f"_unload meta-ized derived rotary buffers: {_stale()}"
    )
    # ...while checkpoint-backed parameters under the prefix DO go meta.
    assert model.layers[2].self_attn.wq_a.weight.is_meta
