"""GLM's MTP layer, priced as the served drafter computes it (PQ #1271).

`prismaquant.glm_mtp.Glm5NextMtpLayer` is built from Transformers' GLM modules
and wired the way vLLM's `Glm5NextMTP` drafter runs layer 45. Whether that
wiring is right is not checkable by inspection: a swapped `[enorm | hnorm]`
concat, a missed position-0 zeroing, or a residual taken after the wrong norm
all give plausible logits. So the layer is checked against an independent
float64 reimplementation of the served forward, written from the weights
alone, on a tiny random config on CPU.

The seed is checked against the quantity it claims to estimate: the
second-order KL of the draft distribution, averaged per MTP row. Its identity
is checked against the join guard that must keep MTP rows out of the body's
table.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
glm5 = pytest.importorskip(
    "transformers.models.glm5_next",
    reason="glm5_next requires transformers >= 5.16",
)

from prismaquant import genuine_weight_initialization  # noqa: E402
from prismaquant import glm_mtp  # noqa: E402
from prismaquant.cost_currency import probe_identity_walls_differ  # noqa: E402
from prismaquant.kl_fisher import fisher_quadratic_form  # noqa: E402

VOCAB = 97
BACKBONE = 2  # the MTP layer is index 2, as GLM-5.3-Flash's is 45
SWIGLU_LIMIT = 0.3  # small enough that both clamps bind on random weights

#: GLM-5.3-Flash-BF16's layer-45 tensors other than the 864 per-expert ones,
#: relative to `model.language_model.layers.45.` (safetensors index,
#: /mnt/shared/models/GLM-5.3-Flash-BF16, read 2026-09-25).
CHECKPOINT_LAYER_45_NON_EXPERT = {
    "eh_proj.weight", "enorm.weight", "hnorm.weight", "input_layernorm.weight",
    "mlp.gate.e_score_correction_bias", "mlp.gate.weight",
    "mlp.shared_experts.down_proj.weight", "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight", "post_attention_layernorm.weight",
    "self_attn.indexer.index_kpool_compress_ape",
    "self_attn.indexer.index_kpool_compress_gate", "self_attn.indexer.k_norm.bias",
    "self_attn.indexer.k_norm.weight", "self_attn.indexer.weights_proj.weight",
    "self_attn.indexer.wk.weight", "self_attn.indexer.wq_b.weight",
    "self_attn.kv_a_layernorm.weight", "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight",
    "self_attn.q_a_layernorm.weight", "self_attn.q_a_proj.weight",
    "self_attn.q_b_proj.weight", "shared_head.norm.weight",
}


def _text_config(**overrides):
    fields = dict(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        moe_intermediate_size=32, num_hidden_layers=BACKBONE,
        num_attention_heads=4, num_key_value_heads=4, head_dim=0,
        q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=0,
        v_head_dim=16, mla_use_nope=True, n_routed_experts=6, n_shared_experts=1,
        num_experts_per_tok=2, n_group=1, topk_group=1, first_k_dense_replace=1,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"], indexer_types=["full", "full"],
        index_head_dim=16, index_n_heads=2, index_topk=16, index_kpool=2,
        index_kpool_compress=True, index_kpool_always_select_tail=True,
        norm_topk_prob=True,
        linear_attn_config={"num_heads": 4, "head_dim": 16,
                            "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
        mhc=True, hc_mult=4, hc_sinkhorn_iters=4, hc_eps=1e-6,
        num_nextn_predict_layers=1, pad_token_id=0, eos_token_id=1,
        tie_word_embeddings=False, rms_norm_eps=1e-5, swiglu_limit=SWIGLU_LIMIT,
        routed_scaling_factor=2.5, dtype=torch.float32,
    )
    fields.update(overrides)
    config = glm5.Glm5NextTextConfig(**fields)
    config._attn_implementation = "eager"
    return config


def _randomize(module, seed):
    """Every tensor random: `torch.empty` experts and zero-init indexer
    parameters would otherwise make a wrong wiring agree with a right one."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float64)
            if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                tensor.copy_(1.0 + 0.1 * noise)
            else:
                scale = 1.0 / max(tensor.shape[-1], 1) ** 0.5
                tensor.copy_(noise * scale)


@pytest.fixture(scope="module")
def tiny():
    config = _text_config()
    with genuine_weight_initialization():
        layer = glm_mtp.Glm5NextMtpLayer(config)
    _randomize(layer, 20260925)
    layer = layer.to(torch.float32).eval()
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator().manual_seed(7)
    embed = torch.randn(VOCAB, config.hidden_size, generator=generator)
    head = torch.randn(VOCAB, config.hidden_size, generator=generator) / 8.0
    return config, layer, embed, head


def _rms(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def _reference_logits(layer, embed, head, ids, final_hidden):
    """vLLM's GLM drafter, float64, from the weights (mtp.py, model.py)."""
    config = layer.config
    assert config.n_group == 1, "the reference router implements one expert group"
    w = {name: t.detach().double() for name, t in layer.state_dict().items()}
    eps = config.rms_norm_eps
    e = embed.double()[ids[:, 1:]]
    e[:, 0] = 0.0                                  # fused_eh_norm: position 0
    h = final_hidden.double()[:, :-1]
    x = torch.cat([_rms(e, w["enorm.weight"], eps), _rms(h, w["hnorm.weight"], eps)], -1)
    x = x @ w["eh_proj.weight"].T
    batch, rows, _ = x.shape

    # MLA, no RoPE, dense causal (the indexer keeps every key in this regime).
    heads = config.num_attention_heads
    nope, vdim = config.qk_nope_head_dim, config.v_head_dim
    a_in = _rms(x, w["input_layernorm.weight"], eps)
    q = _rms(a_in @ w["self_attn.q_a_proj.weight"].T, w["self_attn.q_a_layernorm.weight"], eps)
    q = (q @ w["self_attn.q_b_proj.weight"].T).view(batch, rows, heads, nope)
    latent = (a_in @ w["self_attn.kv_a_proj_with_mqa.weight"].T)[..., :config.kv_lora_rank]
    kv = _rms(latent, w["self_attn.kv_a_layernorm.weight"], eps)
    kv = (kv @ w["self_attn.kv_b_proj.weight"].T).view(batch, rows, heads, nope + vdim)
    k, v = kv[..., :nope], kv[..., nope:]
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * nope ** -0.5
    causal = torch.ones(rows, rows, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    attn = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v).reshape(batch, rows, -1)
    residual = attn @ w["self_attn.o_proj.weight"].T + x

    # MoE: sigmoid router with correction bias, renormalized top-k, scaled.
    m_in = _rms(residual, w["post_attention_layernorm.weight"], eps)
    flat = m_in.reshape(-1, m_in.shape[-1])
    scores = torch.sigmoid(flat @ w["mlp.gate.weight"].T)
    choice = scores + w["mlp.gate.e_score_correction_bias"]
    top = choice.topk(config.num_experts_per_tok, dim=-1).indices
    weights = scores.gather(1, top)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * config.routed_scaling_factor

    def swiglu(gate, up):
        gate = gate.clamp(max=config.swiglu_limit)
        up = up.clamp(min=-config.swiglu_limit, max=config.swiglu_limit)
        return torch.nn.functional.silu(gate) * up

    gate_up, down = w["mlp.experts.gate_up_proj"], w["mlp.experts.down_proj"]
    inter = config.moe_intermediate_size
    routed = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for slot in range(top.shape[1]):
            expert = int(top[token, slot])
            projected = gate_up[expert] @ flat[token]
            hidden = swiglu(projected[:inter], projected[inter:])
            routed[token] += weights[token, slot] * (down[expert] @ hidden)
    shared = swiglu(flat @ w["mlp.shared_experts.gate_proj.weight"].T,
                    flat @ w["mlp.shared_experts.up_proj.weight"].T)
    shared = shared @ w["mlp.shared_experts.down_proj.weight"].T
    mlp_out = (routed + shared).view_as(m_in)

    final = _rms(mlp_out + residual, w["shared_head.norm.weight"], eps)
    return final @ head.double().T


def _logits(tiny, ids, final_hidden):
    config, layer, embed, head = tiny
    return glm_mtp.mtp_logits(
        layer, lambda t: embed[t], lambda x: x @ head.T, ids, final_hidden)


@pytest.mark.parametrize("shape", [(1, 12), (3, 7)])
def test_mtp_layer_matches_the_served_drafter_reference(tiny, shape):
    config = tiny[0]
    generator = torch.Generator().manual_seed(shape[1])
    ids = torch.randint(0, VOCAB, shape, generator=generator)
    final_hidden = torch.randn(*shape, config.hidden_size, generator=generator)

    with torch.inference_mode():
        got = _logits(tiny, ids, final_hidden)
    want = _reference_logits(tiny[1], tiny[2], tiny[3], ids, final_hidden)

    assert got.shape == (shape[0], shape[1] - 1, VOCAB)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got.double(), want, rtol=2e-4, atol=2e-4)


def test_row_zero_ignores_its_embedding(tiny):
    """Row 0 is the only reader of token 1, and the drafter zeroes it there."""
    config = tiny[0]
    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(2, VOCAB, (2, 9), generator=generator)
    final_hidden = torch.randn(2, 9, config.hidden_size, generator=generator)
    changed = ids.clone()
    changed[:, 1] = (changed[:, 1] + 17) % VOCAB
    changed_later = ids.clone()
    changed_later[:, 2] = (changed_later[:, 2] + 17) % VOCAB

    with torch.inference_mode():
        base = _logits(tiny, ids, final_hidden)
        same = _logits(tiny, changed, final_hidden)
        moved = _logits(tiny, changed_later, final_hidden)

    torch.testing.assert_close(same, base, rtol=0, atol=0)
    assert not torch.allclose(moved, base)


def test_rows_pair_each_hidden_state_with_the_next_token():
    ids = torch.tensor([[10, 11, 12, 13]])
    hidden = torch.arange(4.0).view(1, 4, 1).expand(1, 4, 3)
    next_ids, previous, positions = glm_mtp.mtp_rows(ids, hidden)
    assert next_ids.tolist() == [[11, 12, 13]]
    assert previous[0, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert positions.tolist() == [[0, 1, 2]]


def test_parameter_names_are_the_checkpoints(tiny):
    """Everything but the packed routed experts is named as in layer 45."""
    names = set(tiny[1].state_dict())
    packed = {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
    assert packed <= names
    assert names - packed == CHECKPOINT_LAYER_45_NON_EXPERT


def test_config_reaches_the_mtp_index_without_touching_the_backbone():
    config = _text_config()
    before = (list(config.layer_types), list(config.indexer_types),
              list(config.mlp_layer_types))
    extended = glm_mtp.mtp_text_config(config)
    assert glm_mtp.mtp_layer_index(config) == BACKBONE
    assert extended.layer_types[BACKBONE] == "deepseek_sparse_attention"
    assert extended.indexer_types[BACKBONE] == "full"
    assert extended.mlp_layer_types[BACKBONE] == "sparse"
    assert (config.layer_types, config.indexer_types, config.mlp_layer_types) == before


def test_one_mtp_layer_or_refuse():
    with pytest.raises(ValueError, match="exactly one MTP layer"):
        glm_mtp.mtp_layer_index(_text_config(num_nextn_predict_layers=2))


def test_a_sequence_the_indexer_would_sparsify_refuses(tiny):
    config = tiny[0]
    # topk 16, kpool 2: 8 pools are selectable, so 16 rows fit and 17 do not.
    glm_mtp.require_dense_indexer_regime(tiny[1].config, 16)
    with pytest.raises(ValueError, match="attention would be sparse"):
        glm_mtp.require_dense_indexer_regime(tiny[1].config, 17)
    ids = torch.zeros(1, 18, dtype=torch.long)
    with pytest.raises(ValueError, match="attention would be sparse"):
        _logits(tiny, ids, torch.zeros(1, 18, config.hidden_size))


def test_glm_calibration_is_in_the_dense_regime():
    """GLM-5.3-Flash: topk 2048, kpool 4, 512-token calibration, 511 rows."""
    config = _text_config(index_topk=2048, index_kpool=4)
    glm_mtp.require_dense_indexer_regime(glm_mtp.mtp_text_config(config), 511)


def test_without_tail_selection_the_regime_is_unchecked():
    config = _text_config(index_kpool_always_select_tail=False)
    with pytest.raises(ValueError, match="always_select_tail"):
        glm_mtp.require_dense_indexer_regime(glm_mtp.mtp_text_config(config), 4)


def test_seed_estimates_the_draft_self_kl_per_row():
    """0.5 * mean_k <grad_k, dz>^2 converges on the per-row second-order KL."""
    generator = torch.Generator().manual_seed(5)
    batch, rows = 2, 5
    logits = torch.randn(batch, rows, 11, generator=generator) * 2.0
    delta = torch.randn(batch, rows, 11, generator=generator) * 3e-3

    samples = []
    for seed in range(10000):
        z = logits.clone().requires_grad_(True)
        glm_mtp.mtp_probe_scalar(z, seed=seed, global_row_offset=0, n_sequences=batch).backward()
        samples.append(float((z.grad * delta).sum()) ** 2)
    estimate = 0.5 * sum(samples) / len(samples)

    second_order = float(fisher_quadratic_form(logits, delta, token_scope="all"))
    p = logits.log_softmax(-1)
    q = (logits + delta).log_softmax(-1)
    kl = float((p.exp() * (p - q)).sum(-1).mean())

    assert estimate == pytest.approx(second_order, rel=0.05)
    assert second_order == pytest.approx(kl, rel=0.02)


def test_seed_normalizes_by_every_row_of_every_sequence():
    assert glm_mtp.mtp_global_token_count(512, 512) == 512 * 511
    logits = torch.randn(1, 4, 7).requires_grad_(True)
    glm_mtp.mtp_probe_scalar(logits, seed=1, global_row_offset=3, n_sequences=8).backward()
    wide = logits.grad.clone()
    logits.grad = None
    glm_mtp.mtp_probe_scalar(logits, seed=1, global_row_offset=3, n_sequences=2).backward()
    torch.testing.assert_close(logits.grad, wide * 2.0)


def test_mtp_rows_cannot_join_body_rows():
    body = {
        "schema": "prismaquant.joint_aura.probes.v2", "n_probes": 4, "seed_base": 7000,
        "token_scope": "all", "temperature": 1.0, "distribution": "rademacher",
        "normalization": "global_kl_fisher", "calibration_sha256": "a" * 64,
        "producer_source_sha256": "b" * 64, "arithmetic": {"dtype": "bf16"},
    }
    objective = glm_mtp.mtp_objective_identity(mtp_layer=45, sequence_length=512,
                                               n_sequences=512)
    mtp = glm_mtp.with_mtp_objective(body, objective)

    assert objective["global_token_count"] == 261632
    assert "objective" not in body
    assert probe_identity_walls_differ(body, mtp)
    assert not probe_identity_walls_differ(
        mtp, glm_mtp.with_mtp_objective(dict(body, producer_source_sha256="c" * 64), objective))
    with pytest.raises(ValueError, match="already names an objective"):
        glm_mtp.with_mtp_objective(mtp, objective)
