"""Qwen3.5 / Qwen3.6 MoE profile.

Covers:
  - Qwen3_5MoeForConditionalGeneration (multimodal, MoE)
  - Qwen3_5MoeForCausalLM (text-only MoE)
  - Qwen3_5MoeTextModel  (headless)

The two naming conventions PrismaQuant must juggle:

  | where                    | body                                         |
  |--------------------------|----------------------------------------------|
  | HF multimodal source     | model.language_model.layers.X.*              |
  | vLLM scheme-dispatch     | language_model.model.layers.X.*              |
  | HF text-only / lm_head   | lm_head                                      |
  | vLLM scheme-dispatch     | language_model.lm_head                       |
  | MTP source               | mtp.layers.0.*   (mtp.fc, mtp.norm, ...)     |
  | vLLM MTP scheme-dispatch | mtp.layers.0.*   (IDENTITY — mtp. → model.   |
  |                          |                    remap only at weight-load)|

Visual encoder blocks pass through as BF16 (no real calibration yet).
"""
from __future__ import annotations

import copy
import os
import re

import torch.nn as nn

from .base import ModelProfile


_QWEN3_5_FALLBACK_PACKED_MODULES = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}


class Qwen3_5Profile(ModelProfile):

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        if model_type in {"qwen3_5_moe", "qwen3_5_moe_text", "qwen3_5"}:
            return True
        for arch in architectures:
            if arch.startswith("Qwen3_5") or arch.startswith("Qwen3.5") \
                    or arch.startswith("Qwen3_6") or arch.startswith("Qwen3.6"):
                return True
        return False

    @property
    def name(self) -> str:
        return "qwen3_5"

    def vllm_architecture_class(self) -> str | None:
        """vLLM class to read `packed_modules_mapping` +
        `hf_to_vllm_mapper` from. The base class auto-derives
        `fused_sibling_group()` and the body-part of
        `to_vllm_internal_name()` from these two attributes. We only
        override `to_vllm_internal_name()` below to handle the MTP
        prefix specially."""
        return "Qwen3_5MoeForConditionalGeneration"

    def fused_sibling_group(self, linear_qname: str) -> str | None:
        """Return the canonical fused-module key for Qwen3.5/3.6 Linears.

        Allocation often runs on a CPU-only host where vLLM is not
        importable. In that environment the base implementation silently
        loses the packed-module mapping and fused-sibling promotion is
        skipped, which can produce mixed-format `linear_attn` groups that
        vLLM cannot load. Keep a local fallback map so allocation remains
        correct without vLLM in-process.
        """
        if self._fused_matcher is None:
            self._ensure_vllm_class()
            from .vllm_registry import (
                fused_sibling_matcher_from_packed_mapping,
                packed_modules_mapping_from_class,
            )
            pm = (
                packed_modules_mapping_from_class(self._vllm_cls)
                or _QWEN3_5_FALLBACK_PACKED_MODULES
            )
            self._fused_matcher = fused_sibling_matcher_from_packed_mapping(pm)
        return self._fused_matcher(linear_qname)

    # ------------------------------------------------------------
    # MoE
    # ------------------------------------------------------------
    def packed_expert_param_names(self) -> frozenset[str]:
        return frozenset({"gate_up_proj", "down_proj"})

    def per_expert_moe_regex(self) -> str | None:
        # vLLM constructs per-expert Linears under
        # language_model.model.layers.X.mlp.experts.Y.{gate|up|down}_proj
        return (r"re:^language_model[.]model[.]layers[.][0-9]+"
                r"[.]mlp[.]experts[.][0-9]+[.](gate|up|down)_proj$")

    def split_packed_experts_for_format(self, fmt: str) -> bool:
        # Qwen3.5's vLLM load_weights has a latching is_fused_expert
        # flag: the first tensor name containing `experts.gate_up_proj`
        # or `experts.down_proj` (the UNSPLIT packed form) flips the
        # flag to True permanently for the rest of weight iteration,
        # dropping `expert_params_mapping` down to a 2-entry fused
        # form that only matches packed names. Any per-expert split
        # tensors encountered afterwards (which is the form our
        # quantized NVFP4/MXFP8 MoE layers use) fail the substring
        # test and silently skip loading → broken model.
        #
        # Mixed recipes (NVFP4 MoE layers + BF16 MoE layers in the
        # same artifact) can hit this: the BF16 layers would default
        # to packed under the base profile, then trip the flag mid-
        # iteration. Force split for EVERY format on Qwen3.5 so the
        # artifact has a uniform per-expert form and the latching bug
        # never triggers.
        return True

    # ------------------------------------------------------------
    # MTP
    # ------------------------------------------------------------
    def has_mtp(self) -> bool:
        return True

    def per_expert_mtp_regex(self) -> str | None:
        # MTP prefix stays `mtp.` at scheme dispatch (Qwen3_5MTP passes
        # prefix="mtp" to its inner predictor). The `mtp.→model.` rewrite
        # only happens in the weight loader.
        return (r"re:^mtp[.]layers[.][0-9]+"
                r"[.]mlp[.]experts[.][0-9]+[.](gate|up|down)_proj$")

    def mtp_layer_count(self, cfg: dict) -> int:
        # Use base implementation first.
        n = super().mtp_layer_count(cfg)
        if n > 0:
            return n
        # Qwen3.6 uses `mtp_num_hidden_layers` on text_config. Covered
        # above. If still zero, scan safetensors as a last resort.
        return 0  # caller can scan safetensors separately if desired

    def build_mtp_module(self, text_config) -> nn.Module:
        """Mirror vLLM's `Qwen3_5MultiTokenPredictor.forward`:

            e = pre_fc_norm_embedding(inputs_embeds)
            h = pre_fc_norm_hidden(body_hidden_states)
            h = fc(cat([e, h]))
            h = layers[0](h, pos_embeddings, causal_mask)
            h = norm(h)
            return h

        Implemented on HF primitives so Fisher hooks and autograd work
        normally."""
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeDecoderLayer, Qwen3_5MoeRMSNorm,
        )
        mtp_cfg = _single_layer_full_attention_config(text_config)
        hidden = mtp_cfg.hidden_size
        eps = mtp_cfg.rms_norm_eps

        class _MtpModule(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.fc = nn.Linear(hidden * 2, hidden, bias=False)
                self.layers = nn.ModuleList([Qwen3_5MoeDecoderLayer(cfg, layer_idx=0)])
                self.norm = Qwen3_5MoeRMSNorm(hidden, eps=eps)
                self.pre_fc_norm_hidden = Qwen3_5MoeRMSNorm(hidden, eps=eps)
                self.pre_fc_norm_embedding = Qwen3_5MoeRMSNorm(hidden, eps=eps)

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

        import torch
        return _MtpModule(mtp_cfg)

    def mtp_objective_example(self) -> str:
        return ("CE(lm_head(MTP(embed_{t+1}, body_hidden_t)), ids_{t+2}) — "
                "the aux-loss Qwen3.5/3.6 MTP was trained under.")

    # ------------------------------------------------------------
    # Naming remap for compressed-tensors scheme dispatch
    # ------------------------------------------------------------
    def live_to_recipe_name(self, live_qname: str) -> str:
        """Qwen 3.5/3.6 loads as `Qwen3_5MoeForConditionalGeneration`
        which exposes body Linears as `model.language_model.layers.X.*`.
        The allocator's recipe keys come from the text-only probe so
        they're `model.layers.X.*`. Strip the `language_model.` infix."""
        if live_qname.startswith("model.language_model."):
            return "model." + live_qname[len("model.language_model."):]
        return live_qname

    def to_vllm_internal_name(self, name: str) -> str:
        """Apply vLLM's HF→vLLM name remap (`hf_to_vllm_mapper`) for
        body/visual/lm_head, with two arch-specific overrides:

        1. MTP: preserve the `mtp.` prefix. vLLM's `mtp.→model.`
           rewrite is a weight-loader transform that runs AFTER
           scheme dispatch, so it does NOT affect target matching
           — we need the literal `mtp.*` prefix in config_groups.

        2. Text-only recipe form: PrismaQuant's allocator emits body
           entries as `model.layers.X.*` (from the staged text-only
           probe), but vLLM's mapper only has a `model.language_model.`
           -> `language_model.model.` prefix. We rewrite the
           text-only form to the multimodal source form first so the
           base-class mapper handles it uniformly.
        """
        # MTP — preserve prefix. See docstring.
        if name.startswith("mtp."):
            return name
        # Bare `lm_head` (no trailing dot) — vLLM's prefix mapper has
        # `lm_head. -> language_model.lm_head.` which only matches when
        # a `.suffix` follows. At scheme-dispatch the module qname is
        # just `lm_head`, so we map explicitly.
        if name == "lm_head":
            return "language_model.lm_head"
        # Lift text-only recipe form to multimodal source form so the
        # vLLM-derived prefix mapper (from `hf_to_vllm_mapper`)
        # catches it. The vLLM mapper's prefixes are
        # `{model.visual. -> visual., lm_head. -> language_model.lm_head.,
        #   model.language_model. -> language_model.model.}`
        if (name.startswith("model.layers.")
                or name.startswith("model.embed_tokens")
                or name.startswith("model.norm")
                or name == "model"):
            name = "model.language_model." + name[len("model."):]
        mapped = super().to_vllm_internal_name(name)
        # Fallback when vLLM isn't importable locally: preserve the
        # expected scheme-dispatch names rather than returning the HF
        # checkpoint names unchanged.
        if mapped == name:
            if name.startswith("model.language_model."):
                return "language_model.model." + name[len("model.language_model."):]
            if name.startswith("model.visual."):
                return name[len("model."):]
        return mapped

    # ------------------------------------------------------------
    # Source passthrough + staging
    # ------------------------------------------------------------
    def source_passthrough_prefixes(self) -> tuple[str, ...]:
        # Visual encoder is passthrough (real calibration deferred); MTP
        # weights without a layer_config entry go through passthrough too.
        return ("model.visual.", "mtp.")

    def stage_text_only_strip_keys(self) -> tuple[str, ...]:
        return (
            "vision_config", "audio_config", "speech_config",
            "image_token_id", "video_token_id",
            "vision_start_token_id", "vision_end_token_id",
        )

    def stage_text_only_promote_inner_model_type(self) -> bool:
        # Qwen3.5-122B-A10B ships `Qwen3_5MoeForConditionalGeneration`
        # with an empty top-level `hidden_size` — all body dims live in
        # `text_config`. Qwen3.6-35B-A3B duplicates keys at both levels
        # so promotion is idempotent there. Flip True unconditionally
        # so either variant loads as the text-only inner class.
        return True

    # ------------------------------------------------------------
    # Shard prefixes
    # ------------------------------------------------------------
    def visual_layer_prefix(self) -> str:
        return "model.visual.blocks"

    def visual_config_key(self) -> str:
        return "vision_config"


def _single_layer_full_attention_config(text_config):
    """Shallow-copy the text config and pin it to one full-attention
    decoder layer — that's what vLLM's MTP uses."""
    cfg = copy.deepcopy(text_config)
    cfg.layer_types = ["full_attention"]
    cfg.num_hidden_layers = 1
    return cfg
