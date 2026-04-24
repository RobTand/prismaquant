"""MiniMax M2 / M2.7 profile.

Covers:
  - MiniMaxM2ForCausalLM (230B / 256-expert MoE)

Naming conventions PrismaQuant must bridge:

  | where                        | body                                                   |
  |------------------------------|--------------------------------------------------------|
  | HF source safetensors        | model.layers.X.block_sparse_moe.experts.Y.w{1,2,3}    |
  | vLLM runtime module path     | model.layers.X.block_sparse_moe.experts.Y.*           |
  | vLLM scheme-dispatch target  | model.layers.X.mlp.experts.Y.{gate,down,up}_proj      |

The scheme-dispatch rename is not cosmetic — `MiniMaxM2DecoderLayer.__init__`
hard-codes `prefix=f"{prefix}.mlp"` when constructing the MoE submodule
(even though the attribute name is `block_sparse_moe`). And vLLM's MoE
weight loader translates `w1/w2/w3` → `gate_proj/down_proj/up_proj` via
`ckpt_{gate,down,up}_proj_name` kwargs at load time — but the scheme
lookup runs first and doesn't see those kwargs, so the config_groups
must already use the `mlp.*_proj` form.
"""
from __future__ import annotations

import re

from .base import ModelProfile


_MINIMAX_PACKED_MODULES = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
}

# HF checkpoint expert-leaf → vLLM scheme-dispatch leaf
_EXPERT_LEAF_RENAME = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
}

# Matches anywhere-in-name expert leafs and the `block_sparse_moe` infix.
# Applied as a one-shot rewrite inside to_vllm_internal_name().
_BSM_INFIX = "block_sparse_moe"
_EXPERT_LEAF_RE = re.compile(
    r"(?P<prefix>\." + _BSM_INFIX + r"\.experts\.\d+\.)(?P<leaf>w[123])(?P<suffix>$|[.])"
)


class MiniMaxM2Profile(ModelProfile):

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        if model_type in {"minimax_m2", "minimax-m2", "minimax_m2.7"}:
            return True
        for arch in architectures:
            if arch.startswith("MiniMaxM2") or arch.startswith("MiniMax-M2"):
                return True
        return False

    @property
    def name(self) -> str:
        return "minimax_m2"

    def vllm_architecture_class(self) -> str | None:
        return "MiniMaxM2ForCausalLM"

    # ------------------------------------------------------------
    # Fused-sibling promotion (q/k/v/o, gate/up if any)
    # ------------------------------------------------------------
    def fused_sibling_group(self, linear_qname: str) -> str | None:
        if self._fused_matcher is None:
            self._ensure_vllm_class()
            from .vllm_registry import (
                fused_sibling_matcher_from_packed_mapping,
                packed_modules_mapping_from_class,
            )
            pm = (
                packed_modules_mapping_from_class(self._vllm_cls)
                or _MINIMAX_PACKED_MODULES
            )
            self._fused_matcher = fused_sibling_matcher_from_packed_mapping(pm)
        return self._fused_matcher(linear_qname)

    # ------------------------------------------------------------
    # MoE — per-expert scheme target regex
    # ------------------------------------------------------------
    def packed_expert_param_names(self) -> frozenset[str]:
        # vLLM MoE packs experts as (gate_up_proj, down_proj) after the
        # w1+w3 fusion; exported per-Linear names are the UNFUSED leafs.
        return frozenset({"gate_proj", "up_proj", "down_proj"})

    def per_expert_moe_regex(self) -> str | None:
        # Matches every per-expert projection in the vLLM-internal
        # dispatch form. 62 layers × 256 experts × 3 projs collapse to
        # this one regex in config_groups.
        return (r"re:^model[.]layers[.][0-9]+"
                r"[.]mlp[.]experts[.][0-9]+[.](gate|up|down)_proj$")

    # ------------------------------------------------------------
    # MTP — not present in MiniMax M2.7 default recipe
    # ------------------------------------------------------------
    def has_mtp(self) -> bool:
        return False

    def per_expert_mtp_regex(self) -> str | None:
        return None

    # ------------------------------------------------------------
    # Name remap: HF → vLLM scheme-dispatch
    # ------------------------------------------------------------
    def to_vllm_internal_name(self, checkpoint_name: str) -> str:
        """Translate HF `block_sparse_moe.experts.Y.w{1,2,3}` → vLLM
        `mlp.experts.Y.{gate,down,up}_proj`. Also the non-expert body
        goes through the base prefix mapper (identity for MiniMax since
        there's no hf_to_vllm_mapper on MiniMaxM2ForCausalLM)."""
        # First apply the expert leaf rename if it matches.
        def _rewrite_expert(match):
            new_leaf = _EXPERT_LEAF_RENAME[match.group("leaf")]
            return (match.group("prefix").replace(_BSM_INFIX, "mlp")
                    + new_leaf
                    + match.group("suffix"))
        name = _EXPERT_LEAF_RE.sub(_rewrite_expert, checkpoint_name)
        # Any remaining `block_sparse_moe` → `mlp` (covers non-expert
        # submodules like routing gates that vLLM also places under
        # `mlp.*` at dispatch time — defensive; MiniMax currently has
        # none but we don't want to quietly miss future additions).
        if _BSM_INFIX in name:
            name = name.replace(_BSM_INFIX, "mlp")
        # Fall through to the base prefix mapper for any non-MoE
        # renames (lm_head etc.). For MiniMaxM2 there's no vLLM-side
        # prefix remap so this is effectively identity.
        return super().to_vllm_internal_name(name)

    def source_tensor_name(self, model_qname: str) -> str:
        # HF tree is flat: `model.layers.X.*`. No rewrite needed.
        return model_qname

    def live_to_recipe_name(self, live_qname: str) -> str:
        # No multimodal umbrella. Recipe/live names align.
        return live_qname

    # Streaming probe/cost bypass transformers' FP8 module rewrite for
    # MiniMax under transformers 5.x: HF replaces the ModuleList experts
    # with FP8Experts, then attempts to set `experts.0.w1`, which is not
    # valid on that container. PrismaQuant instead reads source FP8
    # bytes directly and applies each 128×128 `weight_scale_inv` block in
    # `_read_layer_to_device`, so the live Linear still receives true
    # dequanted bf16 values for Fisher/cost math.
    #
    # On the EXPORT side, FP8-sourced Linears we're keeping at 8 bits
    # bypass the dequanted BF16 view entirely — the FP8_SOURCE branch
    # in export_native_compressed opens the source safetensors directly
    # and copies `.weight` + `.weight_scale_inv` verbatim. See the
    # passthrough-integrity filter in allocator.py for the gating.
