"""poolside Laguna (LagunaForCausalLM, model_type=laguna) profile.

Laguna-S-2.1: 48 decoder layers, hidden 3072, GQA 48/8 head_dim 128,
sparse MoE every layer — 256 routed experts top-10 (moe_intermediate 1024,
sigmoid routing with norm_topk) + 1 shared expert (intermediate 1024).
~117B params, 235 GB bf16, text-only, untied embeddings (vocab 100352).

On-disk layout is the Qwen3.5-MoE/Ornith per-expert convention
(``…mlp.experts.{i}.{gate,up,down}_proj.weight``) — the streaming loader's
per-expert→packed bridge applies unchanged. Router ships as
``mlp.gate.weight`` (already the live/vLLM name; no rename). The shared
expert ships AND dispatches as ``mlp.shared_expert.*`` — unlike Hy3, vLLM's
Laguna class threads the ``.shared_expert`` prefix through to its Linears
(laguna.py: ``prefix=f"{prefix}.shared_expert"``), so no dispatch aliasing
is needed.

vLLM serves the class natively (registry: LagunaForCausalLM → laguna.py),
and the spec-decode drafter has its own first-class class
(DFlashLagunaForCausalLM / laguna_dflash.py) — the DFlash drafter is this
family's MTP-analog; its rung goes through the canon throughput selector
at export time. ``has_mtp`` is False here because the drafter is a separate
checkpoint (poolside/Laguna-S-2.1-DFlash), not an in-body sidecar.

All quantizable in-dims are 256-superblock-aligned (attn in 3072/6144,
expert gate_up in 3072, expert down in 1024) — CB-clean without ignores.
"""
from __future__ import annotations

from .base import ModelProfile


class LagunaProfile(ModelProfile):

    # Detection priority (lower = consulted first): disjoint.
    priority = 190
    """poolside Laguna family (Laguna-S/XS 2.x)."""

    @classmethod
    def matches(cls, model_type: str | None,
                architectures: list[str] | None) -> bool:
        return cls.claims_by_name(
            model_type, architectures,
            frozenset({"laguna"}), ("Laguna",))

    @property
    def name(self) -> str:
        return "laguna"

    def vllm_architecture_class(self) -> str | None:
        return "LagunaForCausalLM"

    def has_mtp(self) -> bool:
        # The DFlash drafter is a SEPARATE checkpoint (…-DFlash), not an
        # in-body/mtp.* sidecar; the mtp_source_prefix/build_mtp_module
        # machinery does not apply.
        return False

    def checkpoint_to_live_name(self, ckpt_key: str, *,
                                multimodal: bool = False) -> str | None:
        # vLLM-trained Laguna checkpoints store the aux-loss-free routing
        # bias on the experts module (`mlp.experts.e_score_correction_bias`);
        # poolside's HF class remaps it onto the router via
        # `_checkpoint_conversion_mapping`. Mirror that mapping — the live
        # LagunaTopKRouter declares the parameter, LagunaExperts does not.
        suffix = ".mlp.experts.e_score_correction_bias"
        if ckpt_key.endswith(suffix):
            return ckpt_key[: -len(suffix)] + ".mlp.gate.e_score_correction_bias"
        return super().checkpoint_to_live_name(
            ckpt_key, multimodal=multimodal)

    def register_vendored_modeling(self) -> None:
        # Laguna ships PER-LAYER-TYPE rope_parameters ({"full_attention":
        # {..., rope_type: yarn, original_max_position_embeddings: 8192},
        # "sliding_attention": {...}}) — the venv transformers' rope
        # validator indexes the TOP level flat and KeyErrors on the yarn
        # fields that live one level down. Patch the yarn validator to skip
        # dicts whose values are themselves per-layer-type sub-dicts; the
        # sub-dicts carry complete yarn params, so nothing real is skipped.
        # (DSv4-precedent config monkey-patch; drop when the venv
        # transformers understands the per-layer-type format.)
        try:
            import transformers.modeling_rope_utils as mru
        except Exception:
            return
        if getattr(mru, "_pq_laguna_rope_patch", False):
            return
        # The validators are METHODS dispatched via
        # getattr(self, f"_validate_{rope_type}_rope_parameters") — patch the
        # yarn method on whichever class carries it.
        cls = None
        for name in dir(mru):
            obj = getattr(mru, name)
            if isinstance(obj, type) and hasattr(
                    obj, "_validate_yarn_rope_parameters"):
                cls = obj
                break
        if cls is None:
            return
        orig = cls._validate_yarn_rope_parameters

        def _tolerant_yarn_validate(self, rope_parameters, *a, **kw):
            if ("original_max_position_embeddings" not in rope_parameters
                    and any(isinstance(v, dict)
                            for v in rope_parameters.values())):
                return  # per-layer-type shape: sub-dicts validated by use
            return orig(self, rope_parameters, *a, **kw)

        cls._validate_yarn_rope_parameters = _tolerant_yarn_validate
        mru._pq_laguna_rope_patch = True
