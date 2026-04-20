"""Architecture profile — PrismQuant's adapter layer between a model
family's checkpoint conventions and the format-agnostic core pipeline.

Each profile captures three kinds of knowledge:

  1. **Naming**: how checkpoint parameter names map to vLLM's internal
     Linear qnames at compressed-tensors scheme dispatch, and the regex
     patterns vLLM uses for per-expert MoE loading.

  2. **Structure**: which Linear groups are fused siblings (q/k/v,
     gate/up, etc.), what 3D Parameters represent packed MoE experts,
     whether the architecture has MTP heads.

  3. **MTP construction**: how to stand up an HF-module replica of the
     architecture's MTP forward (for Fisher probing), and how to load
     `mtp.*` safetensors into it.

Profiles are picked per-run by `registry.detect_profile(model_path)`
from HF config + architectures. Unknown architectures fall back to
`DefaultProfile` which runs the generic path (no fused-sibling
promotion, no MTP support, plain `model.layers.*` naming).
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import torch.nn as nn


class ModelProfile(ABC):
    """Base class for all PrismQuant architecture profiles.

    Where possible, default implementations auto-derive their return
    values from the vLLM model class registered for this architecture
    (`vllm_architecture_class()`). That way, adding a new architecture
    typically only requires `matches()`, `vllm_architecture_class()`,
    and an optional `build_mtp_module()` — the rest comes from vLLM's
    `packed_modules_mapping` and `hf_to_vllm_mapper` class attributes.
    """

    def __init__(self) -> None:
        # Lazy-compiled derivations from the vLLM class. Computed on
        # first access so profile construction stays cheap.
        self._vllm_cls = None
        self._vllm_cls_loaded = False
        self._fused_matcher = None
        self._name_remapper = None

    # ------------------------------------------------------------
    # Identity + match
    # ------------------------------------------------------------
    @classmethod
    @abstractmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        """Return True if this profile claims responsibility for the
        given HF `model_type` / `architectures`."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Profile identifier (e.g. 'qwen3_5', 'default')."""

    def vllm_architecture_class(self) -> str | None:
        """Return the HF `architectures[0]` string whose vLLM class
        PrismQuant should read `packed_modules_mapping` and
        `hf_to_vllm_mapper` from. Profiles that don't have a vLLM
        counterpart (dev-only architectures) can return None and
        override the dependent methods manually."""
        return None

    def _ensure_vllm_class(self):
        if self._vllm_cls_loaded:
            return
        self._vllm_cls_loaded = True
        arch = self.vllm_architecture_class()
        if arch is None:
            return
        from .vllm_registry import vllm_class_for_architecture
        self._vllm_cls = vllm_class_for_architecture(arch)

    # ------------------------------------------------------------
    # Fused-sibling promotion (allocator.py)
    # ------------------------------------------------------------
    def fused_sibling_group(self, linear_qname: str) -> str | None:
        """Return a canonical 'group key' if this Linear belongs to a
        fused-sibling group (q/k/v/o, gate/up, etc.), otherwise None.

        Default implementation derives sibling groups from the vLLM
        class's `packed_modules_mapping` attribute. Profiles can
        override to add arch-specific groups vLLM doesn't know about,
        or to bypass the vLLM lookup entirely.

        Example (Qwen3.5 via vLLM's `Qwen3_5MoeForConditionalGeneration`):
          model.layers.3.self_attn.q_proj -> 'model.layers.3.self_attn.qkv_proj'
          model.layers.3.self_attn.k_proj -> 'model.layers.3.self_attn.qkv_proj'
          model.layers.3.mlp.gate_proj    -> 'model.layers.3.mlp.gate_up_proj'
        """
        if self._fused_matcher is None:
            self._ensure_vllm_class()
            from .vllm_registry import (
                fused_sibling_matcher_from_packed_mapping,
                packed_modules_mapping_from_class,
            )
            pm = packed_modules_mapping_from_class(self._vllm_cls)
            if not pm:
                self._fused_matcher = lambda _qname: None
            else:
                self._fused_matcher = fused_sibling_matcher_from_packed_mapping(pm)
        return self._fused_matcher(linear_qname)

    # ------------------------------------------------------------
    # MoE packing
    # ------------------------------------------------------------
    def packed_expert_param_names(self) -> frozenset[str]:
        """Parameter attribute names (on a `*Experts` module) that hold
        3D packed MoE weight tensors. Union across all known architectures
        is a safe default; specific profiles can narrow."""
        return frozenset({
            "gate_up_proj", "down_proj",   # Qwen3.5 / 3.6
            "w1", "w2", "w3",              # Mixtral
            "gate_proj", "up_proj",        # some HF layouts
        })

    def per_expert_moe_regex(self) -> str | None:
        """Regex matching vLLM's per-expert Linear qnames at scheme
        dispatch time. Added to the config_groups catch-all so every
        per-expert per-projection tensor picks up the catch-all format
        without ~30k explicit targets."""
        return None

    # ------------------------------------------------------------
    # MTP
    # ------------------------------------------------------------
    def has_mtp(self) -> bool:
        """True if this architecture has Multi-Token-Prediction heads
        in its checkpoint (`mtp.*` tensors) that PrismQuant can probe
        and quantize."""
        return False

    def build_mtp_module(self, text_config) -> nn.Module | None:
        """Construct an HF-module replica of the MTP forward (mirrors
        what vLLM's MTP class does at inference time). Return None if
        `has_mtp()` is False.

        The returned module must be wrappable — after `load_state_dict`
        with the stripped-prefix MTP weights it should forward a hidden
        state + next-token embed into the MTP block exactly as vLLM does."""
        return None

    def load_mtp_state_dict(self, mtp_module: nn.Module,
                            raw: dict) -> tuple[list[str], list[str]]:
        """Load raw `mtp.*` tensors (with `mtp.` stripped) into
        `mtp_module`. Return `(unmatched_keys, module_params_without_weight)`.

        Default implementation uses `mtp_module.load_state_dict(raw, strict=False)`."""
        mapped: dict = {}
        for k, v in raw.items():
            mapped[k] = v
        sd = mtp_module.state_dict()
        mapped_filtered = {k: v for k, v in mapped.items() if k in sd}
        missing = [k for k in mapped if k not in sd]
        extra = [k for k in sd if k not in mapped_filtered]
        mtp_module.load_state_dict(mapped_filtered, strict=False)
        return missing, extra

    def mtp_objective_example(self) -> str:
        """One-line description of the MTP training objective for the
        probe's metadata. Generic fallback is fine for most architectures."""
        return "MTP auxiliary loss (predict token t+k given hidden_t)"

    def per_expert_mtp_regex(self) -> str | None:
        """Regex matching MTP per-expert Linear qnames at scheme dispatch.
        Returns None if no MoE MTP in this architecture."""
        return None

    # ------------------------------------------------------------
    # Naming remap for compressed-tensors
    # ------------------------------------------------------------
    def to_vllm_internal_name(self, checkpoint_name: str) -> str:
        """Remap a checkpoint parameter name (as stored in safetensors)
        to the vLLM-internal module qname that `find_matched_target`
        compares against at scheme dispatch.

        Default implementation uses the vLLM class's `hf_to_vllm_mapper`
        (specifically its `orig_to_new_prefix` dict). Matches vLLM's
        own weight-loader remap, so the allocator's config_groups
        targets and the runtime scheme-dispatch names stay in sync
        without PrismQuant duplicating the mapping.

        Profiles override when: (a) there's no vLLM class for this
        arch, (b) the vLLM mapper is regex/substring-based (we only
        consume the prefix form), or (c) there are arch-specific
        quirks like MTP that need special handling beyond the simple
        prefix rewrite."""
        if self._name_remapper is None:
            self._ensure_vllm_class()
            from .vllm_registry import (
                hf_to_vllm_prefix_map_from_class,
                name_remapper_from_prefix_map,
            )
            prefix = hf_to_vllm_prefix_map_from_class(self._vllm_cls)
            self._name_remapper = name_remapper_from_prefix_map(prefix)
        return self._name_remapper(checkpoint_name)

    def source_tensor_name(self, model_qname: str) -> str:
        """Rewrite an in-memory HF module qname (from `named_parameters`)
        to the name that should land on disk in the exported
        safetensors. For multimodal HF checkpoints loaded via
        AutoModelForCausalLM, the module tree is flat (`model.layers.X.*`)
        but the source safetensors use the multimodal convention
        (`model.language_model.layers.X.*`) that vLLM expects.

        Default: identity. Multimodal architectures override."""
        return model_qname

    def live_to_recipe_name(self, live_qname: str) -> str:
        """Map a live HF-module qname (from `named_modules()` on the
        loaded export-time model) to the allocator-recipe qname (from
        the probe's text-only staged model).

        Multimodal architectures where AutoModelForCausalLM returns
        the `ForConditionalGeneration` sibling class get live names
        like `model.language_model.layers.X.*`, but the probe ran
        on a text-only staging that produced recipe keys like
        `model.layers.X.*`. This method strips the language_model
        infix so the allocator's assignment dict lookups succeed.

        Default: identity. Multimodal architectures override."""
        return live_qname

    def on_disk_expert_qname(self, live_hf_qname: str) -> str:
        """Reserved for future profile-specific expert-tensor name
        rewrites. Default: identity. Currently unused by the export
        path (vLLM's architecture-specific weight-loaders handle
        `.moe.` insertion themselves via substring remaps in their
        own `load_weights` code), but kept as an extension point for
        architectures where vLLM's own remap is absent."""
        return live_hf_qname

    def split_packed_experts_for_format(self, fmt: str) -> bool:
        """Whether to split packed MoE experts into per-expert
        per-projection 2D tensors on disk for the given format.

        vLLM's MoE weight loaders vary:

          - Qwen 3.5/3.6 + compressed-tensors NVFP4: expects per-expert
            per-projection 2D tensors with compressed suffixes
            (`experts.0.gate_proj.weight_packed` etc.). We must split.

          - Gemma 4 + BF16: expects 3D packed checkpoint tensors
            (`experts.gate_up_proj`, `experts.down_proj`) and its own
            `_weight_iterator` explodes them into per-expert shards
            for FusedMoE. We must NOT split — a pre-split checkpoint
            lands under a name (`...experts.0.gate_proj`) that vLLM's
            remap turns into `...moe.experts.0.gate_proj`, which then
            misses the 3D-only explode path and fails to route onto
            the fused `w13_weight` / `w2_weight` params.

        Default: split for every non-BF16 format (NVFP4, MXFP8, etc.)
        and keep packed for BF16. Profiles can override when their
        vLLM loader has different expectations — for instance, Qwen
        3.5/3.6 would be free to split even at BF16, though there's
        no known quality or compatibility reason to.

        When False, the exporter emits a single 3D tensor named by
        the packed param's live HF qname (e.g.
        `model.language_model.layers.0.experts.gate_up_proj`). vLLM's
        own remap inserts `.moe.` and explodes.

        When True, the exporter splits along the row dim (gate/up
        halves for `gate_up_proj`) and emits per-expert 2D tensors
        named `<parent>.{expert_id}.{proj_name}.weight[.suffix]`."""
        return fmt != "BF16"

    # ------------------------------------------------------------
    # Source passthrough + text-only staging
    # ------------------------------------------------------------
    def source_passthrough_prefixes(self) -> tuple[str, ...]:
        """Prefixes of checkpoint keys that should be copied from the
        source checkpoint as-is (typically visual encoder + MTP when
        not being quantized)."""
        return ()

    def stage_text_only_strip_keys(self) -> tuple[str, ...]:
        """HF config keys to drop when creating a text-only staged
        config for probe/cost model loading (e.g. `vision_config` on
        multimodal models so `AutoModelForCausalLM` can load)."""
        return ("vision_config", "audio_config", "speech_config")

    def stage_text_only_promote_inner_model_type(self) -> bool:
        """When lifting `text_config` keys to top-level during
        text-only staging, should `text_config.model_type` (e.g.
        `gemma4_text`) shadow the outer `model_type` (e.g. `gemma4`)?

        This depends on which HF config class the family's
        `<Arch>ForCausalLM` expects:

        - Gemma 4: `Gemma4ForCausalLM.config: Gemma4TextConfig` — the
          text-specific config class. We must promote `gemma4_text`
          so `AutoConfig` loads `Gemma4TextConfig` and the flat text
          schema's `hidden_size` / `num_hidden_layers` etc. all line
          up with the text checkpoint tensors.

        - Qwen 3.5 MoE: `Qwen3_5MoeForCausalLM.config: Qwen3_5MoeConfig`
          — the multimodal-umbrella config class (with nested
          `text_config`). We must KEEP the outer `qwen3_5_moe` so
          `AutoConfig` loads `Qwen3_5MoeConfig` and the nested
          text_config gets wired in normally.

        Default False (Qwen-like). Families that take a standalone
        text config class override to True."""
        return False

    # ------------------------------------------------------------
    # Extended shard regexes (incremental_probe)
    # ------------------------------------------------------------
    def extended_shard_regexes(self, model_path: str,
                               layers_per_shard: int,
                               *, include_body: bool = True,
                               include_mtp: bool = True,
                               include_visual: bool = True,
                               include_lm_head: bool = True) -> list[str]:
        """Return the list of Linear-name regexes covering every shard
        of the probe — body, MTP, visual, lm_head.

        Reads the SOURCE config (not a staged copy) so vision/MTP
        metadata that text-only staging might strip remains visible."""
        src_cfg_path = Path(model_path) / "config.json"
        with open(src_cfg_path) as f:
            cfg = json.load(f)
        text_cfg = cfg.get("text_config", cfg)
        vis_cfg = cfg.get("vision_config", {})

        regexes: list[str] = []
        if include_body:
            n_body = int(text_cfg.get("num_hidden_layers",
                                       cfg.get("num_hidden_layers", 0)))
            regexes.extend(
                _build_layer_shard_regexes(n_body, layers_per_shard,
                                           layer_prefix=self.body_layer_prefix()))
        if include_mtp and self.has_mtp():
            n_mtp = int(self.mtp_layer_count(cfg) or 0)
            if n_mtp > 0:
                regexes.extend(
                    _build_layer_shard_regexes(n_mtp, layers_per_shard,
                                               layer_prefix=self.mtp_layer_prefix()))
        if include_visual and self.visual_config_key():
            n_vis = int(vis_cfg.get("depth")
                        or vis_cfg.get("num_hidden_layers") or 0)
            if n_vis > 0:
                regexes.extend(
                    _build_layer_shard_regexes(n_vis,
                                               max(layers_per_shard, 4),
                                               layer_prefix=self.visual_layer_prefix()))
        if include_lm_head:
            regexes.append(rf"^{re.escape(self.lm_head_name())}$")
        return regexes

    def body_layer_prefix(self) -> str:
        """Prefix used for body-layer names in the checkpoint (before
        the numeric index)."""
        return "model.layers"

    def mtp_layer_prefix(self) -> str:
        """Prefix used for MTP-layer names in the checkpoint."""
        return "mtp.layers"

    def visual_layer_prefix(self) -> str | None:
        """Prefix used for visual-encoder block names, or None if this
        model has no visual encoder."""
        return None

    def visual_config_key(self) -> str | None:
        """Top-level HF config key under which the vision_config dict
        lives, or None if this model has no visual encoder."""
        return None

    def lm_head_name(self) -> str:
        """Qualified name of the lm_head Linear in the checkpoint."""
        return "lm_head"

    def mtp_layer_count(self, cfg: dict) -> int:
        """Count of MTP layers from the HF config. Fall back to
        scanning the safetensors index via `_count_mtp_layers_from_safetensors`
        in subclasses when the config doesn't report it."""
        text = cfg.get("text_config", cfg)
        return int(
            text.get("num_nextn_predict_layers")
            or cfg.get("num_nextn_predict_layers")
            or text.get("num_mtp_layers")
            or cfg.get("num_mtp_layers")
            or text.get("mtp_num_hidden_layers")
            or cfg.get("mtp_num_hidden_layers")
            or 0
        )


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def _build_layer_shard_regexes(num_layers: int,
                               layers_per_shard: int,
                               *, layer_prefix: str) -> list[str]:
    out: list[str] = []
    for start in range(0, num_layers, layers_per_shard):
        end = min(start + layers_per_shard, num_layers)
        if end - start == 1:
            body = rf"{re.escape(layer_prefix)}\.{start}\."
        else:
            idxs = "|".join(str(i) for i in range(start, end))
            body = rf"{re.escape(layer_prefix)}\.(?:{idxs})\."
        out.append(body)
    return out
