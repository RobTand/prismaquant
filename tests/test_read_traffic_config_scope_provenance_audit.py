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
