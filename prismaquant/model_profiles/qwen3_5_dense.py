"""Qwen3.5 / Qwen3.6 dense profile.

Covers:
  - Qwen3_5ForConditionalGeneration (multimodal, dense MLP, MTP retained)
  - Qwen3_5ForCausalLM (text-only, dense MLP, MTP retained)

Dense variants keep the same hybrid DeltaNet + full-attention layer-mix
and the same MTP head as the MoE sibling, but the per-layer MLP is a
plain gate/up/down Linear stack instead of an experts bank. This
profile inherits body/visual/MTP naming + vLLM remap logic from
Qwen3_5Profile and flips off the MoE-specific hooks so the allocator
and export pipeline treat every MLP as a regular dense Linear.
"""
from __future__ import annotations

import copy

import torch.nn as nn

from .qwen3_5 import Qwen3_5Profile


class Qwen3_5DenseProfile(Qwen3_5Profile):

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        # Catch the dense arch before the MoE catch-all in Qwen3_5Profile.
        # Dense: Qwen3_5ForConditionalGeneration / Qwen3_5ForCausalLM
        # MoE:   Qwen3_5MoeForConditionalGeneration / Qwen3_5MoeForCausalLM
        for arch in architectures:
            if "Moe" in arch:
                return False
            if arch.startswith("Qwen3_5For") or arch.startswith("Qwen3_6For"):
                return True
            if arch.startswith("Qwen3.5For") or arch.startswith("Qwen3.6For"):
                return True
        return False

    @property
    def name(self) -> str:
        return "qwen3_5_dense"

    def vllm_architecture_class(self) -> str | None:
        # Dense vLLM class. Falls back to the MoE class's packed_modules
        # mapping via _QWEN3_5_FALLBACK_PACKED_MODULES if vLLM doesn't
        # ship the dense class yet on the host.
        return "Qwen3_5ForConditionalGeneration"

    # ------------------------------------------------------------
    # MoE — disabled
    # ------------------------------------------------------------
    def packed_expert_param_names(self) -> frozenset[str]:
        return frozenset()

    def per_expert_moe_regex(self) -> str | None:
        return None

    def per_expert_mtp_regex(self) -> str | None:
        return None

    # ------------------------------------------------------------
    # MTP — dense decoder layer (Qwen3_5 / Qwen3_6 share the class)
    # ------------------------------------------------------------
    def build_mtp_module(self, text_config) -> nn.Module:
        """Mirror vLLM's `Qwen3_5MultiTokenPredictor.forward` but using
        the dense `Qwen3_5DecoderLayer` / `Qwen3_5RMSNorm`."""
        import torch
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5DecoderLayer, Qwen3_5RMSNorm,
            )
        except ImportError:
            # Fallback for transformers builds where the dense module
            # isn't split out yet — reuse the MoE decoder class; with
            # num_experts=0 it degenerates to dense MLP.
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeDecoderLayer as Qwen3_5DecoderLayer,
                Qwen3_5MoeRMSNorm as Qwen3_5RMSNorm,
            )

        mtp_cfg = copy.deepcopy(text_config)
        mtp_cfg.layer_types = ["full_attention"]
        mtp_cfg.num_hidden_layers = 1
        hidden = mtp_cfg.hidden_size
        eps = mtp_cfg.rms_norm_eps

        class _DenseMtpModule(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.fc = nn.Linear(hidden * 2, hidden, bias=False)
                self.layers = nn.ModuleList([Qwen3_5DecoderLayer(cfg, layer_idx=0)])
                self.norm = Qwen3_5RMSNorm(hidden, eps=eps)
                self.pre_fc_norm_hidden = Qwen3_5RMSNorm(hidden, eps=eps)
                self.pre_fc_norm_embedding = Qwen3_5RMSNorm(hidden, eps=eps)

            def forward(self, inputs_embeds, body_hidden_states,
                        position_embeddings, causal_mask, position_ids):
                e = self.pre_fc_norm_embedding(inputs_embeds)
                h = self.pre_fc_norm_hidden(body_hidden_states)
                h = torch.cat([e, h], dim=-1)
                h = self.fc(h)
                h = self.layers[0](
                    hidden_states=h,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                )
                if isinstance(h, tuple):
                    h = h[0]
                h = self.norm(h)
                return h

        return _DenseMtpModule(mtp_cfg)
