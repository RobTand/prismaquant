"""Gemma 4 profile (Google's multimodal family — text + vision + audio).

Covers:
  - Gemma4ForConditionalGeneration (multimodal MoE + dense, all sizes)
  - Gemma4ForCausalLM (text-only)

Almost entirely vLLM-metadata-derived — Gemma 4 has a clean
`packed_modules_mapping` (`qkv_proj`, `gate_up_proj`) and a standard
`hf_to_vllm_mapper` that matches Qwen3.5/3.6's body-prefix convention.
No MTP heads (not in vLLM's speculative registry at this vLLM version),
so PrismaQuant doesn't need a custom MTP forward builder.

Source passthrough prefixes cover the three modality towers (vision,
audio, and their embedding projectors) — these pass through as BF16
until we wire real multimodal calibration, matching the Qwen3.6 visual
encoder policy.

Minimal size: ~30 lines. Everything else inherits from base.
"""
from __future__ import annotations

from .base import ModelProfile


class Gemma4Profile(ModelProfile):

    # Detection priority (lower = consulted first): disjoint from the Qwen family.
    priority = 140

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        return cls.claims_by_name(
            model_type, architectures,
            frozenset({"gemma4", "gemma4_text"}), ("Gemma4",))

    @property
    def name(self) -> str:
        return "gemma4"

    def vllm_architecture_class(self) -> str:
        # `Gemma4ForConditionalGeneration` exposes the full multimodal
        # prefix map (vision_tower, audio_tower, embed_vision,
        # embed_audio, language_model). Auto-derived
        # `fused_sibling_group` and `to_vllm_internal_name` inherit
        # from base — no overrides needed.
        return "Gemma4ForConditionalGeneration"

    # `on_disk_expert_qname` intentionally NOT overridden: vLLM's
    # `Gemma4TextModel.load_weights` already runs a substring remap
    # `.experts.{id}.{proj}` → `.moe.experts.{id}.{proj}` (see
    # `vllm.model_executor.models.gemma4.py:1554`). Emitting the HF
    # naming (no `.moe.`) lets vLLM's own remap path land the per-expert
    # tensors correctly on `FusedMoE.w13_weight` / `w2_weight`.
    # Overriding to inject `.moe.` ourselves produces a double `.moe.`
    # after vLLM's remap runs — verified experimentally.

    def init_rotaries(self, rotary, cfg, device, dtype,
                      base_model=None) -> bool:
        """Gemma 4's text rotary is multi-layer-type: it registers one
        ``<layer_type>_inv_freq`` buffer per entry in ``config.layer_types``,
        with *mixed* rope types (e.g. ``sliding_attention``=default,
        ``full_attention``=proportional). The generic single-rope fallback in
        ``_init_rotary_inplace`` calls ``compute_default_rope_parameters(cfg,
        device)`` with no ``layer_type`` → ``KeyError: None`` on
        ``config.rope_parameters[layer_type]`` (issue #6).

        Re-run the rotary's own ``__init__`` on the real device: it rebuilds
        every ``<layer_type>_inv_freq`` / ``<layer_type>_attention_scaling``
        with the correct per-type rope init function (proportional / linear /
        default, plus any per-type kwargs). A hand-rolled
        ``compute_default_rope_parameters`` loop would silently apply the
        *default* formula to the proportional layer and produce wrong
        frequencies."""
        if getattr(rotary, "layer_types", None) is None:
            return False
        if getattr(cfg, "rope_parameters", None) is None:
            return False
        try:
            type(rotary).__init__(rotary, cfg, device=device)
        except Exception:
            return False
        return True

    # ------------------------------------------------------------
    # Cross-layer KV sharing.  Gemma4's last `num_kv_shared_layers`
    # attention layers have no k/v_proj — they reuse the K/V computed by
    # the last non-shared layer of their `layer_type`, passed via a
    # `shared_kv_states` dict the model forward threads through every layer.
    # ------------------------------------------------------------
    def new_forward_pass_state(self) -> dict:
        """A FRESH `shared_kv_states` dict per forward pass.

        `Gemma4TextModel.forward` creates `shared_kv_states = {}` once per
        pass and threads the same object through every decoder layer
        (`transformers/models/gemma4/modeling_gemma4.py:1669`); storing
        layers write `shared_kv_states[layer_idx] = (k, v)` and KV-sharing
        layers read `shared_kv_states[kv_shared_layer_index]`. A new dict per
        call is the contract — reusing one across passes would feed batch
        N-1's K/V (and its sequence length) to batch N."""
        return {"shared_kv_states": {}}

    def capture_forward_pass_state(self, pass_state: dict):
        """Snapshot Gemma4's integer-indexed shared K/V states to CPU."""
        skv = (pass_state or {}).get("shared_kv_states") or {}
        out = {}
        for layer_idx, kv in skv.items():
            try:
                key = int(layer_idx)
                out[key] = tuple(t.detach().to("cpu") for t in kv)
            except Exception as exc:
                raise RuntimeError(
                    f"Gemma4 shared_kv_states[{layer_idx!r}] could not be "
                    "captured as a CPU tensor tuple"
                ) from exc
        return out

    def isolated_layer_pass_state(self, captured, layer) -> dict:
        """For an isolated (phase-3) layer forward: a shared layer needs its
        source layer's captured K/V (the attention moves them to the right device
        itself); a non-shared layer just needs a writable dict to store into.
        Always returns a `shared_kv_states` dict so the layer never sees
        `None`.

        A KV-sharing layer with no captured source K/V cannot be forwarded at
        all — its attention does an unconditional
        `shared_kv_states[kv_shared_layer_index]` lookup. Fail loud here
        instead of letting that surface as a bare `KeyError` from inside
        attention: the only way to get here is a phase-1 capture that never
        ran or a precompute cache written before the capture was persisted."""
        attn = getattr(layer, "self_attn", None)
        if not getattr(attn, "is_kv_shared_layer", False):
            return {"shared_kv_states": {}}
        source_idx = getattr(attn, "kv_shared_layer_index", None)
        kv = None
        if source_idx is not None and captured:
            kv = captured.get(int(source_idx))
        if kv is None:
            raise RuntimeError(
                "Gemma4 KV-sharing layer "
                f"{getattr(attn, 'layer_idx', '?')} needs shared_kv_states"
                f"[{source_idx!r}] from the phase-1 capture, but the captured "
                f"state has {sorted(captured) if captured else 'no'} entries. "
                "Delete the probe precompute cache (WORK_DIR/work/"
                "precomputed.pt) so phase-1 re-runs and re-captures it."
            )
        return {"shared_kv_states": {int(source_idx): kv}}

    def export_tensor_name(self, model_qname: str) -> str:
        """Keep body/expert export keys in recipe form.

        Gemma 4's vLLM weight iterator performs its own body and
        `.experts` -> `.moe.experts` remaps. Source lookup still uses the
        declarative `recipe_to_source` rules, but export must not pre-apply
        those remaps or vLLM sees doubled `.moe.` prefixes.
        """
        if (
            model_qname.startswith("model.layers.")
            or model_qname.startswith("model.embed_tokens")
            or model_qname.startswith("model.norm")
        ):
            return model_qname
        return super().export_tensor_name(model_qname)
