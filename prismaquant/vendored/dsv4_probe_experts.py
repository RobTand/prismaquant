"""Per-expert ModuleList replacement for `DeepseekV4Experts` (probe time).

PR #45643's `DeepseekV4Experts` packs 256 routed experts into two 3D
Parameters per layer (`gate_up_proj` (E, 2I, H), `down_proj` (E, H, I))
and uses `F.linear` with slice indexing. That's correct math but
prismaquant's per-Linear Fisher hooks fire on `nn.Linear.forward` /
`backward` and can't observe a packed Parameter slice without
substantial core changes.

This module provides a structurally equivalent variant where the 256
experts live as `nn.ModuleList[DeepseekV4ExpertMLP_PerExpert]`, each
with three vanilla `nn.Linear` projections (`gate_proj`, `up_proj`,
`down_proj`). Forward is the same per-expert iteration with identical
math. Memory is identical (same parameter count, no aliasing).

Only used at probe time; vLLM serving uses the packed format from
PR #40860 directly. `register_deepseek_v4()` selects this constructor
unconditionally — having per-expert structure during probe is what
makes Fisher and expert-saliency scoring work.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepseekV4ExpertMLP_PerExpert(nn.Module):
    """One routed expert. Mixtral-style MLP with SwiGLU + clamp.

    Mirrors the `_apply_gate` math from PR #45643's `DeepseekV4Experts`:
    gate is clamped to `swiglu_limit` (one-sided max), up is symmetric-
    clamped, then `act(gate) * up` and project down.
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 swiglu_limit: float, act_fn: nn.Module):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x).clamp(max=self.swiglu_limit)
        up = self.up_proj(x).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class DeepseekV4Experts_PerExpert(nn.ModuleList):
    """Drop-in replacement for `DeepseekV4Experts`. Same forward
    signature `(hidden_states, top_k_index, top_k_weights)`, same
    output. The container IS a `nn.ModuleList` of 256
    `DeepseekV4ExpertMLP_PerExpert` modules — each expert is
    accessible directly as `experts[E]` AND as `experts.E` (which is
    what `set_module_tensor_to_device` uses to traverse parameter
    qnames like `mlp.experts.42.gate_proj.weight`).

    The token routing logic (one_hot mask + per-expert dispatch) is
    copied verbatim from PR #45643's `DeepseekV4Experts.forward` so
    numerical behavior is identical.
    """

    def __init__(self, config):
        from transformers.activations import ACT2FN
        # ACT2FN is a ClassInstantier; calling [name] returns an
        # instance. SiLU has no per-expert state so all experts can
        # share one instance.
        act_fn = ACT2FN[config.hidden_act]
        modules = [
            DeepseekV4ExpertMLP_PerExpert(
                config.hidden_size,
                config.moe_intermediate_size,
                config.swiglu_limit,
                act_fn,
            )
            for _ in range(config.n_routed_experts)
        ]
        super().__init__(modules)
        self.num_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self[expert_idx](hidden_states[token_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


def enable_per_expert_experts() -> None:
    """Select the per-expert constructor for vendored sparse blocks.

    Idempotent, and invoked by `register_deepseek_v4()` before any block
    is constructed. The packed class keeps its identity for HF's packed
    initialization dispatch; the adapter's Linear leaves initialize normally.
    """
    import transformers.models.deepseek_v4.modeling_deepseek_v4 as _m
    if getattr(_m, "_prismaquant_per_expert_swapped", False):
        return
    _m.DeepseekV4SparseMoeBlock.experts_type = DeepseekV4Experts_PerExpert
    _m._prismaquant_per_expert_swapped = True
