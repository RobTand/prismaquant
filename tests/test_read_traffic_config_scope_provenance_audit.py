from __future__ import annotations

import pytest


@pytest.mark.parametrize("config,source", [
    (
        {"language_model_config": {"num_experts_per_tok": 2, "n_routed_experts": 8}},
        "config.language_model_config.num_experts_per_tok / "
        "config.language_model_config.n_routed_experts",
    ),
    (
        {"text_config": {"n_routed_experts": 8}, "language_model_config": {
            "num_experts_per_tok": 2, "n_routed_experts": 32}},
        "config.language_model_config.num_experts_per_tok / config.text_config.n_routed_experts",
    ),
    (
        {"text_config": {"num_experts_per_tok": 2, "n_routed_experts": 8},
         "language_model_config": {"num_experts_per_tok": 7, "n_routed_experts": 32}},
        "config.text_config.num_experts_per_tok / config.text_config.n_routed_experts",
    ),
    (
        {"num_experts_per_tok": 2, "num_experts": 8,
         "text_config": {"num_experts_per_tok": 3, "n_routed_experts": 16},
         "language_model_config": {"num_experts_per_tok": 7, "n_routed_experts": 32}},
        "config.num_experts_per_tok / config.num_experts",
    ),
], ids=["language_only", "mixed_nested_fields", "text_precedence", "top_level_precedence"])
def test_routing_factor_names_its_actual_declared_scopes(config, source):
    from prismaquant.read_traffic import resolve_routing_factor

    routing = resolve_routing_factor(config, observed_expert_counts={"expert_stack": 8})

    assert routing.num_experts_per_tok == 2
    assert routing.n_routed_experts == 8
    assert routing.read_probability == 0.25
    assert routing.source == source


@pytest.mark.parametrize("config,tied", [
    ({"language_model_config": {"vocab_size": 17, "tie_word_embeddings": True}}, True),
    ({"text_config": {"vocab_size": 17, "tie_word_embeddings": False},
      "language_model_config": {"vocab_size": 29, "tie_word_embeddings": True}}, False),
    ({"vocab_size": 17, "tie_word_embeddings": True,
      "text_config": {"vocab_size": 29, "tie_word_embeddings": False},
      "language_model_config": {"vocab_size": 31, "tie_word_embeddings": False}}, True),
], ids=["language_only", "text_precedence", "top_level_precedence"])
def test_other_scope_consumers_keep_declared_values_and_precedence(config, tied):
    from prismaquant.model_profiles.registry import profile_from_config
    from prismaquant.read_traffic import resolve_embedding_disposition, resolve_vocab_size

    profile = profile_from_config({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})
    embedding = resolve_embedding_disposition(
        ["model.embed_tokens.weight", "lm_head.weight"], profile=profile, config=config,
    )

    assert resolve_vocab_size(config) == 17
    assert embedding.tie_declared is tied
    assert embedding.lm_head_present is True
    assert embedding.streamed is False
