"""Unit tests for the DSv4 ckpt→live rename.

Forward-pair to test_deepseek_v4_profile.py: the profile owns
`source_tensor_name` (live → ckpt); this file pins the inverse
direction (`DeepseekV4Profile.checkpoint_to_live_name(ckpt_key)`)
used by `_build_weight_map` to populate the streaming loader's
{model_key: ckpt_key} dict.
"""
from __future__ import annotations

import pytest

from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile

_profile = DeepseekV4Profile()
_rename = _profile.checkpoint_to_live_name


def test_top_level_rename():
    assert _rename("embed.weight") == "model.embed_tokens.weight"
    assert _rename("head.weight") == "lm_head.weight"
    assert _rename("norm.weight") == "model.norm.weight"
    assert _rename("hc_head_fn") == "model.hc_head.hc_fn"
    assert _rename("hc_head_base") == "model.hc_head.hc_base"
    assert _rename("hc_head_scale") == "model.hc_head.hc_scale"


def test_top_level_drops():
    """MTP, scale-only siblings, and weight_scale_inv all drop. (hc_head_*
    are now mapped, not dropped — they live under model.hc_head and are
    needed for the multi-stream → single-stream collapse at end-of-body.)"""
    assert _rename("mtp.0.norm.weight") is None
    assert _rename("mtp.0.hc_attn_base") is None
    assert _rename("head.scale") is None
    assert _rename("embed.scale") is None
    assert _rename("layers.0.attn.wkv.weight_scale_inv") is None


def test_attn_rename():
    cases = [
        ("layers.0.attn.wkv.weight",
         "model.layers.0.self_attn.wkv.weight"),
        ("layers.5.attn.wq_a.weight",
         "model.layers.5.self_attn.wq_a.weight"),
        ("layers.42.attn.wq_b.weight",
         "model.layers.42.self_attn.wq_b.weight"),
        ("layers.0.attn.q_norm.weight",
         "model.layers.0.self_attn.q_norm.weight"),
        # PR #45643's `DeepseekV4Attention` exposes the per-head bias
        # buffer as `self.sinks`; checkpoint stores it as `attn.attn_sink`.
        ("layers.0.attn.attn_sink",
         "model.layers.0.self_attn.sinks"),
        # PRISMAQUANT probe mode drops compressor + indexer keys
        # entirely (see comment in `_rename_dsv4_text_only`). PR
        # #45643's compressor wkv shape doesn't match the checkpoint
        # (K and V are concatenated in the source), and the modeling
        # patch makes attention skip the compressor branch always.
        # Test that the drop is consistent for every variant we know
        # about.
    ]
    for ck, live in cases:
        assert _rename(ck) == live, f"{ck} ↦ {_rename(ck)}, expected {live}"


def test_ffn_rename():
    """`ffn` infix → `mlp`, including for routing gate and shared experts."""
    cases = [
        ("layers.0.ffn.gate.weight", "model.layers.0.mlp.gate.weight"),
        ("layers.0.ffn.gate.bias",   "model.layers.0.mlp.gate.bias"),
        # Shared expert leafs renamed: w1→gate_proj, w2→down_proj, w3→up_proj.
        ("layers.0.ffn.shared_experts.w1.weight",
         "model.layers.0.mlp.shared_experts.gate_proj.weight"),
        ("layers.0.ffn.shared_experts.w2.weight",
         "model.layers.0.mlp.shared_experts.down_proj.weight"),
        ("layers.0.ffn.shared_experts.w3.weight",
         "model.layers.0.mlp.shared_experts.up_proj.weight"),
        # Note: .scale siblings of FP8 weights are dropped from the body
        # map; they're consumed by `fp8_scale_pairs` in the dequant pass.
    ]
    for ck, live in cases:
        assert _rename(ck) == live, f"{ck} ↦ {_rename(ck)}, expected {live}"


def test_compressor_and_indexer_keep_faithful_mapping():
    """The faithful forward (87ca027 + census fix) KEEPS compressor and
    indexer weights. Live targets, census-verified against the vendored
    meta-init model (72,317 checkpoint keys, 0 missing):

    - `attn.compressor.*` → `self_attn.compressor.*` with the ape →
      position_bias and norm.weight → kv_norm.weight renames.
    - The indexer lives ON the CSA compressor
      (`DeepseekV4CSACompressor.indexer`), not on the attention module,
      and its checkpoint `indexer.compressor.*` tensors live FLAT on the
      Indexer (it pools inline; no inner compressor submodule).

    NAME resolution is necessary but NOT sufficient — see
    `test_indexer_pooling_carries_the_coff_overlap_widening` below, which
    pins the SHAPES.  Every mapping asserted here already resolved while
    three of these tensors were unloadable.
    """
    cases = [
        ("layers.5.attn.compressor.wkv.weight",
         "model.layers.5.self_attn.compressor.wkv.weight"),
        ("layers.5.attn.compressor.wgate.weight",
         "model.layers.5.self_attn.compressor.wgate.weight"),
        ("layers.5.attn.compressor.ape",
         "model.layers.5.self_attn.compressor.position_bias"),
        ("layers.5.attn.compressor.norm.weight",
         "model.layers.5.self_attn.compressor.kv_norm.weight"),
        ("layers.2.attn.indexer.wq_b.weight",
         "model.layers.2.self_attn.compressor.indexer.wq_b.weight"),
        ("layers.2.attn.indexer.weights_proj.weight",
         "model.layers.2.self_attn.compressor.indexer.weights_proj.weight"),
        ("layers.2.attn.indexer.compressor.wkv.weight",
         "model.layers.2.self_attn.compressor.indexer.wkv.weight"),
        ("layers.2.attn.indexer.compressor.wgate.weight",
         "model.layers.2.self_attn.compressor.indexer.wgate.weight"),
        ("layers.2.attn.indexer.compressor.ape",
         "model.layers.2.self_attn.compressor.indexer.position_bias"),
        ("layers.2.attn.indexer.compressor.norm.weight",
         "model.layers.2.self_attn.compressor.indexer.kv_norm.weight"),
    ]
    for ck, live in cases:
        assert _rename(ck) == live, f"{ck} ↦ {_rename(ck)}, expected {live}"


@pytest.mark.own_process
@pytest.mark.usefixtures("pinned_pb_source")
def test_indexer_pooling_carries_the_coff_overlap_widening():
    """The Lightning Indexer pools at ``coff * index_head_dim``, not ``index_head_dim``.

    model.py:404 builds the Indexer's pooling from the *same* ``Compressor``
    class as the outer compressor, so it inherits ``coff = 1 + (ratio == 4)``
    (model.py:296-304).  The DSv4-Flash checkpoint agrees:

        layers.N.attn.indexer.compressor.wkv.weight   [256, 4096]
        layers.N.attn.indexer.compressor.wgate.weight [256, 4096]
        layers.N.attn.indexer.compressor.ape          [4, 256]

    i.e. ``compress_rate(4)`` rows x ``coff(2) * index_head_dim(128)``.

    This is a SHAPE test on purpose.  The rename assertions above all passed
    while the vendored Indexer was sized at a bare ``index_head_dim``, so those
    three tensors resolved to a live parameter of the WRONG shape and could not
    be loaded at all — a name-only census cannot see it.  ``kv_norm`` stays at
    ``index_head_dim`` because pooling reduces the doubled width back down
    before the norm (checkpoint ``indexer.compressor.norm.weight`` is [128]).
    """
    torch = pytest.importorskip("torch")
    from prismaquant.vendored import register_deepseek_v4

    register_deepseek_v4()
    import transformers.models.deepseek_v4.modeling_deepseek_v4 as vmod
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    cfg = DeepseekV4Config(num_hidden_layers=4, compress_ratios=[0, 0, 4, 128])
    orig_reset = torch.nn.Linear.reset_parameters
    torch.nn.Linear.reset_parameters = lambda self: None  # allocate, don't initialise
    try:
        indexer = vmod.DeepseekV4Indexer(cfg)
        compressor = vmod.DeepseekV4CSACompressor(cfg, cfg.compress_rate_csa)
    finally:
        torch.nn.Linear.reset_parameters = orig_reset

    coff = 2  # compress_rate_csa == 4 -> overlapping windows
    assert indexer.coff == coff and indexer.overlap
    assert tuple(indexer.wkv.weight.shape) == (coff * cfg.index_head_dim, cfg.hidden_size)
    assert tuple(indexer.wgate.weight.shape) == (coff * cfg.index_head_dim, cfg.hidden_size)
    assert tuple(indexer.position_bias.shape) == (cfg.compress_rate_csa, coff * cfg.index_head_dim)
    assert tuple(indexer.kv_norm.weight.shape) == (cfg.index_head_dim,)
    assert tuple(indexer.wq_b.weight.shape) == (cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank)
    assert tuple(indexer.weights_proj.weight.shape) == (cfg.index_n_heads, cfg.hidden_size)
    # The outer compressor carries the same widening at the attention head_dim.
    assert tuple(compressor.wkv.weight.shape) == (coff * cfg.head_dim, cfg.hidden_size)
    assert tuple(compressor.position_bias.shape) == (cfg.compress_rate_csa, coff * cfg.head_dim)


@pytest.mark.own_process
@pytest.mark.usefixtures("pinned_pb_source")
def test_csa_compressor_returns_indices_not_a_gather():
    """CSA must hand its top-k out as indices, never gather with them.

    The indexer's output carries the reference's ``-1`` sentinel (model.py:436)
    and a ``+offset`` rebasing indices onto the concatenated
    ``[window_kv ; pool]`` sequence (model.py:515-520).  Using it as a
    ``torch.gather`` index raised on CPU and silently read out of bounds on
    CUDA.  Pin the contract: ``(pool, topk)`` for CSA, ``(pool, None)`` for HCA.
    """
    torch = pytest.importorskip("torch")
    from prismaquant.vendored import register_deepseek_v4

    register_deepseek_v4()
    import transformers.models.deepseek_v4.modeling_deepseek_v4 as vmod
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    # Small but structurally faithful: ratio 4 -> overlap, and seq_len 8 closes
    # two windows so the pool is non-empty and sentinels are actually produced.
    cfg = DeepseekV4Config(
        hidden_size=64, head_dim=16, qk_rope_head_dim=8, num_attention_heads=2,
        index_head_dim=8, index_n_heads=2, q_lora_rank=16, index_topk=4,
        sliding_window=8, num_hidden_layers=4, compress_ratios=[0, 0, 4, 128],
    )
    torch.manual_seed(0)
    comp = vmod.DeepseekV4CSACompressor(cfg, cfg.compress_rate_csa)
    for p in comp.parameters():
        p.data.normal_(0, 0.02)

    batch, seq_len = 1, 8
    hidden = torch.randn(batch, seq_len, cfg.hidden_size) * 0.02
    q_residual = torch.randn(batch, seq_len, cfg.q_lora_rank) * 0.02
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cache_layer = vmod.DeepseekV4CSALayer(cfg.sliding_window, cfg.compress_rate_csa)

    # Must NOT raise: the -1 sentinels used to reach torch.gather.
    out = comp(hidden, q_residual, position_ids, cache_layer, offset=seq_len, start_pos=0)
    assert isinstance(out, tuple) and len(out) == 2
    pooled, topk = out
    assert pooled.shape == (batch, seq_len // cfg.compress_rate_csa, cfg.head_dim)
    # Sentinels survive as -1 (they are mask instructions, not indices), and no
    # index may address the pool block before its `offset` rebasing.
    assert (topk == -1).any(), "expected -1 sentinels for non-causal pooled slots"
    real = topk[topk >= 0]
    assert real.numel() and int(real.min()) >= seq_len
    assert int(real.max()) < seq_len + pooled.shape[1]

    # HCA has no indexer, so it reports no index list and the caller derives one.
    hca = vmod.DeepseekV4HCACompressor(cfg, cfg.compress_rate_hca)
    for p in hca.parameters():
        p.data.normal_(0, 0.02)
    hca_cache = vmod.DeepseekV4HCALayer(cfg.sliding_window, cfg.compress_rate_hca)
    hca_pooled, hca_topk = hca(hidden, q_residual, position_ids, hca_cache, offset=seq_len, start_pos=0)
    assert hca_topk is None and hca_pooled.shape[-1] == cfg.head_dim


def test_routed_experts_per_expert_rename():
    """Routed experts map to per-expert ModuleList live names (set up
    by `enable_per_expert_experts()`). w1→gate_proj, w2→down_proj,
    w3→up_proj per expert. Suffixes (.weight, .scale) preserved."""
    cases = [
        ("layers.0.ffn.experts.0.w1.weight",
         "model.layers.0.mlp.experts.0.gate_proj.weight"),
        ("layers.0.ffn.experts.0.w2.weight",
         "model.layers.0.mlp.experts.0.down_proj.weight"),
        ("layers.0.ffn.experts.0.w3.weight",
         "model.layers.0.mlp.experts.0.up_proj.weight"),
        ("layers.0.ffn.experts.255.w3.weight",
         "model.layers.0.mlp.experts.255.up_proj.weight"),
        ("layers.42.ffn.experts.128.w2.weight",
         "model.layers.42.mlp.experts.128.down_proj.weight"),
        # Note: FP8 block-scale siblings of routed experts drop from
        # the body map; they're consumed by `fp8_scale_pairs`.
    ]
    for ck, live in cases:
        assert _rename(ck) == live, f"{ck} ↦ {_rename(ck)}, expected {live}"


def test_hyper_connection_rename():
    """`hc_attn_X` → `attn_hc.X`, `hc_ffn_X` → `ffn_hc.X` (inverse of
    profile's compaction)."""
    cases = [
        ("layers.0.hc_attn_base", "model.layers.0.attn_hc.base"),
        ("layers.0.hc_attn_fn",   "model.layers.0.attn_hc.fn"),
        ("layers.0.hc_attn_scale", "model.layers.0.attn_hc.scale"),
        ("layers.5.hc_ffn_base",  "model.layers.5.ffn_hc.base"),
        ("layers.5.hc_ffn_scale", "model.layers.5.ffn_hc.scale"),
    ]
    for ck, live in cases:
        assert _rename(ck) == live


def test_attn_norm_ffn_norm_to_layernorms():
    """The DSv4 checkpoint stores `attn_norm` (pre-attention RMSNorm)
    and `ffn_norm` (pre-MLP RMSNorm) at the layer scope. The
    transformers `DecoderLayer` exposes them as `input_layernorm`
    and `post_attention_layernorm`."""
    cases = [
        ("layers.0.attn_norm.weight",
         "model.layers.0.input_layernorm.weight"),
        ("layers.5.attn_norm.weight",
         "model.layers.5.input_layernorm.weight"),
        ("layers.0.ffn_norm.weight",
         "model.layers.0.post_attention_layernorm.weight"),
        ("layers.42.ffn_norm.weight",
         "model.layers.42.post_attention_layernorm.weight"),
    ]
    for ck, live in cases:
        assert _rename(ck) == live, f"{ck} ↦ {_rename(ck)}, expected {live}"


def test_layernorm_passthrough_already_standard():
    """If a checkpoint already uses the standard names, pass through with
    just the `model.` prefix add (defensive)."""
    assert (_rename("layers.0.input_layernorm.weight")
            == "model.layers.0.input_layernorm.weight")
    assert (_rename("layers.0.post_attention_layernorm.weight")
            == "model.layers.0.post_attention_layernorm.weight")


def test_unrecognized_drops():
    """Unrecognized top-level keys drop rather than risk shadowing real
    body tensors. The probe will report which keys went missing if the
    DSv4 checkpoint introduces new top-level structure."""
    # No real DSv4 keys collide here, but verify the conservative drop.
    assert _rename("some_unknown_key") is None
    assert _rename("debug_only_thing") is None
