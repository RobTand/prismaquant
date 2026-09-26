"""Tencent Hy3 (HYV3ForCausalLM, model_type=hy_v3) profile.

Hy3-preview: 80 decoder layers + 1 MTP layer, hidden 4096, GQA 64/8 with
q/k norms, layer 0 dense SwiGLU (intermediate 13312), layers 1..79 sparse
MoE (192 routed experts top-8, moe_intermediate 1536, + 1 shared expert).
Source checkpoints (BF16 and FP8) ship experts PER-EXPERT
(``…mlp.experts.{i}.{gate,up,down}_proj.weight``); the live transformers
module packs them into 3-D params (``HYV3Experts.gate_up_proj/down_proj``)
— the streaming loader bridges via this profile's packed-experts spec,
exactly like Qwen3.5/Ornith.

Checkpoint→live renames mirror transformers' own conversion table
(conversion_mapping.py, hy_v3 entry):

  | checkpoint (safetensors)        | live transformers module          |
  |---------------------------------|-----------------------------------|
  | mlp.router.gate.weight          | mlp.gate.weight                   |
  | mlp.expert_bias                 | mlp.e_score_correction_bias       |
  | mlp.shared_mlp.*                | mlp.shared_experts.*              |

The MTP sidecar is ``model.layers.80`` — it shares the body prefix, but
``config.num_hidden_layers == 80`` means the meta skeleton simply never
instantiates it; its checkpoint keys are dropped here so probe/cost/
allocator never see it. Export handling is per-lane: GGUF exports pass
``--exclude '^model\\.layers\\.80\\.'`` (no draft model in the GGUF
artifact); compressed-tensors exports ship it VERBATIM BF16 via
``passthrough_prefixes`` so vLLM's HYV3MTP spec decode can load it.
``has_mtp`` stays False because the ``mtp.*``-keyed sidecar machinery
(``mtp_source_prefix`` / ``build_mtp_module``) does not apply to this
body-indexed layout.

vLLM class: HYV3ForCausalLM exists upstream but not in the local serving
stacks yet; like DeepSeek-V4 this profile returns None and runs entirely
on spec fallback (fused groups + naming rules from specs/hy_v3.json).
"""
from __future__ import annotations

import re

from .base import ModelProfile

_MTP_LAYER_RE = re.compile(r"^model\.layers\.80\.")
_ROUTER_RE = re.compile(r"^(model\.layers\.\d+\.mlp)\.router\.gate\.weight$")
_EXPERT_BIAS_RE = re.compile(r"^(model\.layers\.\d+\.mlp)\.expert_bias$")
_SHARED_RE = re.compile(r"^(model\.layers\.\d+\.mlp)\.shared_mlp\.")


class HyV3Profile(ModelProfile):

    # Detection priority (lower = consulted first): disjoint.
    priority = 180
    """Tencent Hy3 / hy_v3 family."""

    @classmethod
    def matches(cls, model_type: str | None,
                architectures: list[str] | None) -> bool:
        return cls.claims_by_name(
            model_type, architectures,
            frozenset({"hy_v3"}), ("HYV3",))

    @property
    def name(self) -> str:
        return "hy_v3"

    def vllm_architecture_class(self):
        # Not present in the local serving stacks; spec fallback covers
        # fused groups and naming (see module docstring).
        return None

    def has_mtp(self) -> bool:
        # model.layers.80 is the MTP layer, body-indexed rather than
        # mtp.*-keyed, so the mtp_source_prefix/build_mtp_module
        # machinery doesn't apply. The
        # skeleton never instantiates it and checkpoint_to_live_name
        # drops its keys; export lanes handle it explicitly (GGUF:
        # --exclude; compressed-tensors: BF16 passthrough via
        # passthrough_prefixes for vLLM spec decode).
        return False

    def to_vllm_internal_name(self, checkpoint_name: str) -> str:
        # Scheme dispatch compares config_groups targets/ignore against
        # the PREFIX strings vLLM passes at Linear construction. vLLM's
        # HYV3MoEFused builds the shared MLP with prefix=f"{prefix}" (the
        # parent mlp prefix, NOT .shared_mlp/.shared_experts), so its
        # Linears dispatch as `...mlp.gate_up_proj` / `...mlp.down_proj`
        # even though their params live under `...mlp.shared_mlp.*`.
        # Map both the live (shared_experts) and checkpoint (shared_mlp)
        # namings onto those dispatch prefixes. Upstream-version-specific:
        # revisit if vLLM ever threads the .shared_mlp prefix through.
        # (2026-07-11: shared_experts targets matched nothing -> shared
        # MLP built unquantized -> load KeyError on its weight_scale.)
        name = checkpoint_name.replace(".mlp.shared_experts.", ".mlp.")
        name = name.replace(".mlp.shared_mlp.", ".mlp.")
        return name.replace(".mlp.router.gate", ".mlp.gate")

    def checkpoint_to_live_name(self, ckpt_key: str, *,
                                multimodal: bool = False) -> str | None:
        if _MTP_LAYER_RE.match(ckpt_key):
            return None
        m = _ROUTER_RE.match(ckpt_key)
        if m:
            return f"{m.group(1)}.gate.weight"
        m = _EXPERT_BIAS_RE.match(ckpt_key)
        if m:
            return f"{m.group(1)}.e_score_correction_bias"
        m = _SHARED_RE.match(ckpt_key)
        if m:
            return _SHARED_RE.sub(r"\1.shared_experts.", ckpt_key)
        return super().checkpoint_to_live_name(
            ckpt_key, multimodal=multimodal)
