"""Original Qwen3 producer profile: dense and routed-MoE text models.

The producer id is deliberately ``qwen3`` for both variants.  That spelling
is the one Gridbook's runtime contract serves; ``qwen3_moe`` is an HF/vLLM
model-type/module spelling, not a producer-contract id.

The routed model selected for the serving smoke is
``Qwen/Qwen3-30B-A3B`` / ``Qwen3MoeForCausalLM``.  Its checkpoint stores
per-expert 2-D ``gate_proj`` / ``up_proj`` / ``down_proj`` tensors while the
Transformers model packs them into 3-D ``gate_up_proj`` / ``down_proj``
Parameters.  The structure spec owns that reversible bridge.  Dense Qwen3
models continue to resolve here; the MoE declarations are inert when no
packed expert Parameters are present.
"""
from __future__ import annotations

from .base import ModelProfile


class Qwen3Profile(ModelProfile):
    """Producer-side structure adapter for original Qwen3 text models."""

    priority = 120

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        if model_type in {"qwen3", "qwen3_moe"}:
            return True
        return any(
            arch in {"Qwen3ForCausalLM", "Qwen3MoeForCausalLM"}
            for arch in architectures
        )

    @property
    def name(self) -> str:
        return "qwen3"

    def vllm_architecture_class(self) -> str | None:
        # The selected routed-MoE smoke must be checked against this exact
        # installed-vLLM registry key.  The declarative spec supplies the
        # dense gate/up fusion that the MoE class adds only conditionally.
        return "Qwen3MoeForCausalLM"
    def pact_cohort_values(self, config: dict | None) -> dict:
        """PACT cohort from explicit plus declared config, no GLM fallback."""
        values = {}
        for key in (
            "vocab_size",
            "scored_positions_per_sequence",
            "raw_tokens_per_sequence",
            "sample_range",
            "local_prefix_rows",
            "input_contract",
            "global_original_tokens",
        ):
            entry = self.pact_measurement_entry(config, key)
            if entry is not None:
                values[key] = entry
        prefix = self.pact_measurement_alias(
            config, "prefix_ids", "serving_prefix_ids"
        )
        missing = [
            key
            for key in (
                "vocab_size",
                "scored_positions_per_sequence",
                "raw_tokens_per_sequence",
            )
            if values.get(key) is None
        ]
        for key in (
            "sample_range",
            "local_prefix_rows",
            "input_contract",
            "global_original_tokens",
        ):
            if values.get(key) is None:
                missing.append(key)
        if prefix is None:
            missing.append("prefix_ids")
        if missing:
            raise ValueError(
                "qwen3: PACT cohort needs explicit cohort fields "
                + ",".join(sorted(missing))
            )
        return {
            "sample_range": list(values["sample_range"]),
            "raw_tokens_per_sequence": int(values["raw_tokens_per_sequence"]),
            "prefix_ids": list(prefix),
            "local_prefix_rows": values["local_prefix_rows"],
            "input_contract": values["input_contract"],
            "global_original_tokens": int(values["global_original_tokens"]),
            "scored_positions_per_sequence": int(
                values["scored_positions_per_sequence"]
            ),
            "vocab_size": int(values["vocab_size"]),
        }


    def fused_sibling_group(self, linear_qname: str) -> str | None:
        """Consult stable spec groups before config-dependent vLLM metadata."""
        spec = self.structure_spec()
        if spec is not None:
            group = spec.fused_group_for(linear_qname)
            if group is not None:
                return group
        return super().fused_sibling_group(linear_qname)

    def fused_sibling_leaf_mapping(self) -> dict[str, tuple[str, ...]]:
        """Merge the family spec with the installed vLLM class metadata."""
        mapping = dict(super().fused_sibling_leaf_mapping())
        spec = self.structure_spec()
        if spec is None:
            return mapping
        for group in spec.fused_groups:
            target = str(group.target_suffix).rsplit(".", 1)[-1]
            members = tuple(
                str(member).rsplit(".", 1)[-1]
                for member in group.member_suffixes
            )
            existing = mapping.get(target)
            if existing is not None and existing != members:
                raise ValueError(
                    f"Qwen3 fused group {target!r} disagrees between vLLM "
                    f"{existing!r} and the structure spec {members!r}"
                )
            mapping[target] = members
        return mapping

    def register_vendored_modeling(self) -> None:
        # This registration targets only model_type=qwen3.  Calling it while
        # handling qwen3_moe is harmless and leaves the installed MoE class
        # untouched.
        from ..vendored import register_qwen3

        register_qwen3()
