"""Default ModelProfile — used when no specific profile matches the model.

Minimal coverage:
  - Fused-sibling promotion via vLLM's `packed_modules_mapping` when
    the model's declared HF architecture resolves to a vLLM class
    (most standard LLaMA / Qwen / Mistral / Gemma variants do).
    Falls back to "no promotion" silently when vLLM doesn't know the
    class.
  - No MTP support.
  - No visual encoder.
  - Identity name remap (`to_vllm_internal_name` returns input unchanged).
  - Generic packed-expert attribute names (`gate_up_proj`, `down_proj`,
    `w1`/`w2`/`w3`, `gate_proj`/`up_proj`) so Mixtral-style MoEs still
    work out of the box.

If you're adding a new architecture, start from this as a baseline and
override only what differs.
"""
from __future__ import annotations

from .base import ModelProfile


class DefaultProfile(ModelProfile):

    def __init__(self, architectures: list[str] | None = None) -> None:
        super().__init__()
        # Stash the HF `architectures` list so `vllm_architecture_class`
        # can surface it for the base class's packed_modules_mapping
        # / hf_to_vllm_mapper lookups. Without this, fused-sibling
        # promotion is a no-op on every unknown arch — which produces
        # independent formats for q_proj / k_proj / v_proj and vLLM
        # refuses to load the artifact ("Found a different quantization
        # schemes for ['q_proj', 'k_proj', 'v_proj'] in
        # self_attn.qkv_proj").
        self._architectures = list(architectures or [])

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        # DefaultProfile is the terminal fallback — never claims anything
        # affirmatively. `registry.detect_profile` picks it when every
        # other profile returns False.
        return False

    @property
    def name(self) -> str:
        return "default"

    def vllm_architecture_class(self) -> str | None:
        # Hand the first declared HF architecture through to the base
        # class's vLLM lookup. If vLLM doesn't know the class, the
        # base's `_ensure_vllm_class` gracefully falls back to None and
        # every derived behavior (fused_sibling_group,
        # to_vllm_internal_name) degrades to its pre-registry identity
        # default — same as before this override.
        return self._architectures[0] if self._architectures else None
