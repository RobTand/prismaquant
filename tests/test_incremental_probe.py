import pickle
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant.incremental_probe import (
    build_layer_shard_regexes,
    merge_probe_pickles,
    _set_minimax_fast_moe,
)
from prismaquant.observers import ExpertSaliencyTracker


class TestIncrementalProbe(unittest.TestCase):
    def test_build_layer_shard_regexes_groups_layers(self):
        regexes = build_layer_shard_regexes(5, 2)
        self.assertEqual(regexes, [
            r"model\.layers\.(?:0|1)\.",
            r"model\.layers\.(?:2|3)\.",
            r"model\.layers\.4\.",
        ])

    def test_merge_probe_pickles_sums_router_counts(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p1 = td / "a.pkl"
            p2 = td / "b.pkl"
            out = td / "merged.pkl"
            with open(p1, "wb") as f:
                pickle.dump({
                    "stats": {"layer.0": {"h_trace": 1.0}},
                    "router_counts": {"r": {"0": 1.5}},
                    "router_totals": {"r": 3},
                    "expert_info": {"layer.0": ("r", "0")},
                    "meta": {"model": "toy"},
                }, f)
            with open(p2, "wb") as f:
                pickle.dump({
                    "stats": {"layer.1": {"h_trace": 2.0}},
                    "router_counts": {"r": {"0": 0.5, "1": 2.0}},
                    "router_totals": {"r": 5},
                    "expert_info": {"layer.1": ("r", "1")},
                    "meta": {"model": "toy"},
                }, f)

            merge_probe_pickles([p1, p2], out)
            with open(out, "rb") as f:
                merged = pickle.load(f)
            self.assertEqual(set(merged["stats"]), {"layer.0", "layer.1"})
            self.assertEqual(merged["router_counts"]["r"]["0"], 2.0)
            self.assertEqual(merged["router_counts"]["r"]["1"], 2.0)
            self.assertEqual(merged["router_totals"]["r"], 8)
            self.assertEqual(merged["meta"]["n_shards"], 2)

    def test_minimax_fast_moe_matches_modulelist_forward_and_backward(self):
        class ToyMLP(nn.Module):
            def __init__(self, hidden=5, ffn=7):
                super().__init__()
                self.w1 = nn.Linear(hidden, ffn, bias=False)
                self.w2 = nn.Linear(ffn, hidden, bias=False)
                self.w3 = nn.Linear(hidden, ffn, bias=False)
                self.act_fn = F.silu

            def forward(self, hidden_states):
                return self.w2(self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states))

        class MiniMaxM2Experts(nn.ModuleList):
            def __init__(self, n_experts=6, top_k=3):
                super().__init__([ToyMLP() for _ in range(n_experts)])
                self.num_experts = n_experts
                self.top_k = top_k

            def forward(self, hidden_states, top_k_index, top_k_weights):
                final_hidden_states = torch.zeros_like(hidden_states)
                expert_mask = torch.nn.functional.one_hot(
                    top_k_index, num_classes=self.num_experts
                ).permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                for expert_idx in expert_hit:
                    idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                    current_state = hidden_states[None, top_x].reshape(
                        -1, hidden_states.shape[-1])
                    expert_id = int(expert_idx.item())
                    current_hidden_states = (
                        self[expert_id](current_state)
                        * top_k_weights[top_x, idx, None]
                    )
                    final_hidden_states.index_add_(0, top_x, current_hidden_states)
                return final_hidden_states

        torch.manual_seed(123)
        ref = MiniMaxM2Experts()
        fast = MiniMaxM2Experts()
        fast.load_state_dict(ref.state_dict())
        self.assertEqual(_set_minimax_fast_moe(fast, True, chunk_size=2), 1)

        hidden_ref = torch.randn(11, 5, requires_grad=True)
        hidden_fast = hidden_ref.detach().clone().requires_grad_(True)
        top_k_index = torch.tensor([
            [0, 1, 3], [1, 4, 5], [2, 3, 0], [5, 4, 1],
            [3, 2, 1], [0, 5, 4], [2, 4, 3], [1, 0, 5],
            [4, 2, 0], [3, 5, 1], [2, 0, 4],
        ])
        weights = torch.rand(11, 3)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        upstream = torch.randn(11, 5)

        out_ref = ref(hidden_ref, top_k_index, weights)
        out_fast = fast(hidden_fast, top_k_index, weights)
        self.assertTrue(torch.allclose(out_fast, out_ref, atol=1e-6, rtol=1e-6))

        out_ref.backward(upstream)
        out_fast.backward(upstream)
        self.assertTrue(torch.allclose(
            hidden_fast.grad, hidden_ref.grad, atol=1e-6, rtol=1e-6))
        for (_, p_ref), (_, p_fast) in zip(ref.named_parameters(), fast.named_parameters()):
            self.assertTrue(torch.allclose(
                p_fast.grad, p_ref.grad, atol=1e-6, rtol=1e-6))

    def test_minimax_fast_moe_accumulates_reap_dropout_saliency(self):
        class ToyMLP(nn.Module):
            def __init__(self, hidden=5, ffn=7):
                super().__init__()
                self.w1 = nn.Linear(hidden, ffn, bias=False)
                self.w2 = nn.Linear(ffn, hidden, bias=False)
                self.w3 = nn.Linear(hidden, ffn, bias=False)
                self.act_fn = F.silu

            def forward(self, hidden_states):
                return self.w2(self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states))

        class MiniMaxM2Experts(nn.ModuleList):
            def __init__(self, n_experts=6, top_k=3):
                super().__init__([ToyMLP() for _ in range(n_experts)])
                self.num_experts = n_experts
                self.top_k = top_k

            def forward(self, hidden_states, top_k_index, top_k_weights):
                final_hidden_states = torch.zeros_like(hidden_states)
                expert_mask = torch.nn.functional.one_hot(
                    top_k_index, num_classes=self.num_experts
                ).permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                for expert_idx in expert_hit:
                    idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                    expert_id = int(expert_idx.item())
                    current_state = hidden_states[None, top_x].reshape(
                        -1, hidden_states.shape[-1])
                    current_hidden_states = (
                        self[expert_id](current_state)
                        * top_k_weights[top_x, idx, None]
                    )
                    final_hidden_states.index_add_(0, top_x, current_hidden_states)
                return final_hidden_states

        torch.manual_seed(321)
        ref = MiniMaxM2Experts()
        fast = MiniMaxM2Experts()
        fast.load_state_dict(ref.state_dict())
        parent = nn.Module()
        parent.gate = nn.Linear(5, fast.num_experts, bias=False)
        parent.experts = fast

        hidden = torch.randn(9, 5)
        top_k_index = torch.tensor([
            [0, 1, 3], [1, 4, 5], [2, 3, 0],
            [5, 4, 1], [3, 2, 1], [0, 5, 4],
            [2, 4, 3], [1, 0, 5], [4, 2, 0],
        ])
        weights = torch.rand(9, 3)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        expected = {e: 0.0 for e in range(ref.num_experts)}
        with torch.no_grad():
            for token in range(hidden.shape[0]):
                for rank in range(top_k_index.shape[1]):
                    eid = int(top_k_index[token, rank])
                    y = ref[eid](hidden[token:token + 1])
                    expected[eid] += (
                        float(weights[token, rank])
                        * float(y.to(torch.float64).norm(dim=-1).pow(2).item())
                    )
        expected = {e: v / hidden.shape[0] for e, v in expected.items()}

        tracker = ExpertSaliencyTracker(
            parent,
            routers_and_experts=[("gate", "experts", list(range(fast.num_experts)))],
            top_k=fast.top_k,
        )
        try:
            self.assertEqual(_set_minimax_fast_moe(parent.experts, True, chunk_size=2), 1)
            # This test calls the experts module directly, so no router
            # hook fires. The real probe calls the parent MoE block and
            # increments this in the router hook before experts.forward.
            tracker._ensure_accumulators("gate", hidden.device)
            tracker.total_tokens_by_router["gate"] += hidden.shape[0]
            with torch.no_grad():
                _ = parent.experts(hidden, top_k_index, weights)
            got = tracker.saliency(reduction="reap_dropout")["gate"]
            for eid in range(fast.num_experts):
                self.assertLess(abs(got[eid] - expected[eid]), 1e-7)
        finally:
            tracker.remove_hooks()


if __name__ == "__main__":
    unittest.main()
