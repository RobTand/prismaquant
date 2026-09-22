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


def test_bounded_prefix_refuses_certified_fallback_before_cache_or_payload(monkeypatch):
    from prismaquant.glm_routing_replay import require_cached_prefix_source_identity
    from prismaquant import cost_streaming
    monkeypatch.delenv('PRISMAQUANT_DEV_MODE',raising=False)
    monkeypatch.setattr(cost_streaming,'validate_cached_streamed_model_identity',lambda *a,**k:pytest.fail('cache validation preceded explicit mode gate'))
    with pytest.raises(RuntimeError,match='automatic source rehash is forbidden'):
        require_cached_prefix_source_identity('source','cache',{})


def test_bounded_prefix_keeps_owner_complete_checkpoint_and_exact_identity(monkeypatch):
    from prismaquant.glm_routing_replay import require_cached_prefix_source_identity
    from prismaquant import cost_streaming
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE','1')
    def validate(source,cache,*,require_complete_checkpoint):
        assert require_complete_checkpoint is True
        return {'identity':'actual'}
    monkeypatch.setattr(cost_streaming,'validate_cached_streamed_model_identity',validate)
    assert require_cached_prefix_source_identity('source','cache',{'identity':'actual'})=={'identity':'actual'}
    with pytest.raises(RuntimeError,match='differs from original preparation'):
        require_cached_prefix_source_identity('source','cache',{'identity':'foreign'})
