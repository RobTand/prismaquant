import pytest
import torch

from prismaquant.glm_routing_replay import select_original_routes, router_normalization_epsilon


class Packed:
    def forward(self, hidden_states, top_k_index, top_k_weights):
        raise AssertionError("capture must not execute experts")


class Router:
    def forward(self, weights):
        denominator = weights.sum(dim=-1, keepdim=True) + 1e-20
        return weights / denominator


class WrongRouter:
    def forward(self, weights):
        denominator = weights.sum(dim=-1, keepdim=True) + 1e-6
        return weights / denominator


def test_preserves_original_route_order_weights_and_integer_width():
    x = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    ids = torch.tensor([[2, 0], [1, 2], [0, 1], [2, 1]], dtype=torch.int64)
    weights = torch.tensor([[1.7, .8], [.2, 2.3], [1., 1.5], [1.8, .7]], dtype=torch.float32)
    result = select_original_routes(Packed(), (x, ids), {"top_k_weights": weights}, sequence_length=4)
    assert torch.equal(result["inputs"], x)
    assert torch.equal(result["top_k_index"], ids) and result["top_k_index"].dtype == torch.int64
    assert torch.equal(result["top_k_weights"], weights) and result["top_k_weights"].dtype == torch.float32
    assert result["coordinates"].tolist() == [[0, 0], [0, 1], [0, 2], [0, 3]]
    with pytest.raises(ValueError, match="exactly the original first sequence"):
        select_original_routes(Packed(), (x, ids, weights), {}, sequence_length=3)


def test_source_router_epsilon_is_inspected_and_never_assumed_lfm():
    assert router_normalization_epsilon(Router()) == 1e-20
    with pytest.raises(ValueError, match="denominator differs"):
        router_normalization_epsilon(WrongRouter())
