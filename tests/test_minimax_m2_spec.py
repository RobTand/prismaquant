"""R22 / R27 — `minimax_m2` becomes declarative.

`minimax_m2` was the standing counter-example to "the spec is the contract":
eight hand-coded overrides and no `specs/*.json`. All of them are expressible
in `prismaquant.model_structure.v1` today, so this file is the equivalence gate
that has to be green before the Python bodies come out (deletion is a later
cycle, per the R8 mitigation — land the declarative form alongside, prove it,
then delete one architecture at a time).

The comparison is between two profiles that share no code path:

  - `python_only` — `MiniMaxM2Profile` with its spec forced off and its vLLM
    class forced absent, i.e. exactly the behaviour that shipped before this
    spec existed (its `fused_sibling_group` override falls back to the
    module-level `_MINIMAX_PACKED_MODULES` on a box without vLLM, which is
    every build box);
  - `spec_only` — a bare `SpecMatchProfile` over `specs/minimax_m2.json`, with
    no MiniMax Python at all.

Three deliberate, named divergences are asserted *as* divergences rather than
smoothed over — see `test_spec_fixes_the_expert_format_group_split` and
`test_declared_fields_that_had_no_python_equivalent`.
"""
from __future__ import annotations

import pytest

from prismaquant.model_profiles.minimax_m2 import MiniMaxM2Profile
from prismaquant.model_profiles.spec_profile import SpecMatchProfile
from prismaquant.model_profiles.structure import load_structure_spec

VLLM_NAMES = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.3.block_sparse_moe.experts.7.w1.weight",
    "model.layers.3.block_sparse_moe.experts.7.w2.weight",
    "model.layers.3.block_sparse_moe.experts.7.w3.weight",
    "model.layers.61.block_sparse_moe.experts.255.w3",
    "model.layers.3.block_sparse_moe.gate.weight",
    "model.embed_tokens.weight",
    "lm_head.weight",
]

FUSED_NAMES = VLLM_NAMES[:4] + ["model.layers.0.mlp.gate_proj"]


@pytest.fixture(scope="module")
def spec():
    loaded = load_structure_spec("minimax_m2")
    assert loaded is not None, "specs/minimax_m2.json is missing"
    return loaded


@pytest.fixture(scope="module")
def python_only():
    """MiniMaxM2Profile as it behaved before the spec existed."""
    profile = MiniMaxM2Profile()
    profile._structure_spec = None
    profile._structure_spec_loaded = True
    # No vLLM on the build venv; pin it so the comparison is deterministic on
    # a box that does have it (the override's own fallback is the transcribed
    # behaviour).
    profile._vllm_cls = None
    profile._vllm_cls_loaded = True
    return profile


@pytest.fixture(scope="module")
def spec_only(spec):
    return SpecMatchProfile(spec)


# ------------------------------------------------------------ the 8 overrides


def test_match_and_identity(spec):
    assert spec.id == "minimax_m2"
    for model_type, archs in [
        ("minimax_m2", ["MiniMaxM2ForCausalLM"]),
        ("minimax_m2.7", ["MiniMaxM2ForCausalLM"]),
        ("minimax-m2", ["MiniMax-M2ForCausalLM"]),
    ]:
        assert spec.match.claims(model_type, archs)
        assert MiniMaxM2Profile.matches(model_type, archs)
    assert not spec.match.claims("qwen3", ["Qwen3ForCausalLM"])


def test_fused_sibling_group_equivalence(python_only, spec_only):
    """`minimax_m2.py:69` -> `fused_groups`. qkv only; no gate/up fusion."""
    for name in FUSED_NAMES:
        assert python_only.fused_sibling_group(name) == \
            spec_only.fused_sibling_group(name), name
    assert spec_only.fused_sibling_group(
        "model.layers.0.self_attn.q_proj") == "model.layers.0.self_attn.qkv_proj"


def test_packed_expert_param_names_equivalence(python_only, spec_only):
    """`:86` -> `packed_experts.param_names` (the UNFUSED vLLM leafs)."""
    assert python_only.packed_expert_param_names() == \
        spec_only.packed_expert_param_names()
    assert spec_only.packed_expert_param_names() == frozenset(
        {"gate_proj", "up_proj", "down_proj"})


def test_per_expert_moe_regex_equivalence(python_only, spec_only):
    """`:91` -> `moe.per_expert_regex`."""
    assert python_only.per_expert_moe_regex() == spec_only.per_expert_moe_regex()
    assert spec_only.per_expert_moe_regex().startswith("re:^model[.]layers")


def test_mtp_answers_equivalence(python_only, spec_only):
    """`:101` / `:104` -> the spec defaults (M2.7 ships no MTP)."""
    assert python_only.has_mtp() is False
    assert spec_only.has_mtp() is False
    assert python_only.per_expert_mtp_regex() is None
    assert spec_only.per_expert_mtp_regex() is None


def test_to_vllm_internal_name_equivalence(python_only, spec_only):
    """`:110` -> `naming.recipe_to_vllm` regex rules.

    This is the rename vLLM's scheme lookup depends on: the config_groups
    targets must already be `mlp.experts.N.{gate,up,down}_proj`, because the
    scheme lookup runs before the weight loader's `ckpt_*_proj_name` remap.
    """
    for name in VLLM_NAMES:
        assert python_only.to_vllm_internal_name(name) == \
            spec_only.to_vllm_internal_name(name), name
    assert spec_only.to_vllm_internal_name(
        "model.layers.3.block_sparse_moe.experts.7.w1.weight"
    ) == "model.layers.3.mlp.experts.7.gate_proj.weight"
    # the defensive tail: a non-expert submodule under block_sparse_moe
    assert spec_only.to_vllm_internal_name(
        "model.layers.3.block_sparse_moe.gate.weight"
    ) == "model.layers.3.mlp.gate.weight"


def test_name_identities_equivalence(python_only, spec_only):
    """`:133` / `:137` -> the spec defaults (flat HF tree, no umbrella)."""
    for name in VLLM_NAMES:
        assert python_only.source_tensor_name(name) == name
        assert spec_only.source_tensor_name(name) == name
        assert python_only.live_to_recipe_name(name) == name
        assert spec_only.live_to_recipe_name(name) == name


def test_pinned_and_passthrough_equivalence(python_only, spec_only):
    assert python_only.pinned_names() == spec_only.pinned_names()
    assert python_only.source_passthrough_prefixes() == \
        spec_only.source_passthrough_prefixes()
    assert python_only.stage_text_only_strip_keys() == \
        spec_only.stage_text_only_strip_keys()
    assert python_only.stage_text_only_promote_inner_model_type() == \
        spec_only.stage_text_only_promote_inner_model_type()
    for fmt in ("BF16", "NVFP4", "FP8_DYNAMIC"):
        assert python_only.split_packed_experts_for_format(fmt) == \
            spec_only.split_packed_experts_for_format(fmt)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert python_only.packed_expert_projection_names(proj) == \
            spec_only.packed_expert_projection_names(proj)


# --------------------------------------------- named, deliberate divergences


def test_spec_fixes_the_expert_format_group_split():
    """A latent unservable-artifact bug the spec closes.

    Without a spec, `packed_expert_format_group` falls through to
    `_fallback_packed_expert_format_groups()`, whose FIRST group is
    `("gate_up_proj", "down_proj")`. MiniMax's `down_proj` matches that group
    while its `gate_proj`/`up_proj` match the next one — so the three
    projections of one expert bank got **two different coupling keys** and the
    solver would not have held them to one format. vLLM's
    CompressedTensorsMoEMethod selects one scheme per FusedMoE layer (§6.4),
    so that is an unservable checkpoint.

    The declared `format_groups` gives all three one key. Asserted here rather
    than silently "fixed" because it is a behaviour change on the live
    `MiniMaxM2Profile`, on an architecture unshipped since M2.7.
    """
    python_only = MiniMaxM2Profile()
    python_only._structure_spec = None
    python_only._structure_spec_loaded = True
    live = MiniMaxM2Profile()

    qnames = [f"model.layers.0.mlp.experts.0.{p}"
              for p in ("gate_proj", "up_proj", "down_proj")]
    before = {python_only.packed_expert_format_group(q) for q in qnames}
    after = {live.packed_expert_format_group(q) for q in qnames}
    assert len(before) == 2, before      # the bug: two keys for one bank
    assert len(after) == 1, after        # the fix: one key
    assert next(iter(after)).endswith("gate_proj,up_proj,down_proj")


def test_projection_role_buckets_match_the_legacy_answers(python_only):
    """The gate_up/down ROLE buckets, declared rather than inferred.

    `packed_expert_parent_for_projection` used to answer from the legacy
    fallback table; `packed_experts.projection_splits` now declares the same
    mapping explicitly (as `lfm2_moe.json` does for w1/w3/w2), so the answers
    are identical and `packed_expert_role_group` keeps working for the
    per-role expert split.
    """
    live = MiniMaxM2Profile()
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert live.packed_expert_parent_for_projection(proj) == \
            python_only.packed_expert_parent_for_projection(proj)
    assert live.packed_expert_parent_for_projection("gate_proj") == "gate_up_proj"
    assert live.packed_expert_parent_for_projection("down_proj") == "down_proj"
    for leaf, role in [("gate_proj", "gate_up_proj"), ("up_proj", "gate_up_proj"),
                       ("down_proj", "down_proj")]:
        assert live.packed_expert_role_group(
            f"model.layers.0.mlp.experts.3.{leaf}") == role


def test_declared_fields_that_had_no_python_equivalent(spec_only):
    """R27's two moves: both were hardcoded arch tests in the core stack.

    `packed_expert_module_class_names` existed as an accessor with no spec
    declaring it; `unpacked_expert_projection_names` was the field no spec
    had ever declared (`base.py:470-495`). Declaring them here is what lets
    consumers ask the profile instead of comparing a class name.
    """
    assert spec_only.packed_expert_module_class_names() == frozenset(
        {"MiniMaxM2Experts"})
    assert spec_only.unpacked_expert_projection_names() == ("w1", "w2", "w3")
    assert spec_only.bypass_hf_fp8_module_rewrite() is True


def test_serving_profile_is_now_declared():
    """Before: `serving_profile_id() -> None` -> `research`, which carries no
    format allow-list, so any menu format passed the serving gate."""
    assert MiniMaxM2Profile().serving_profile_id() == "vllm_packed_moe"


def test_deepseek_v4_declares_the_conservative_serving_profile():
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile

    assert DeepseekV4Profile().serving_profile_id() == "vllm_packed_moe"


def test_minimax_lane_declaration_is_native_only():
    """No gridbook CB loader for MiniMax; declaring otherwise would be the
    over-declaration R6 exists to prevent."""
    assert MiniMaxM2Profile().supported_export_lanes() == ("compressed-tensors",)


# ------------------------------------------- R27: the two core-stack hardcodes


def _write_config(tmp_path, cfg):
    import json

    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return str(tmp_path)


@pytest.mark.parametrize(
    "cfg,expected",
    [
        # MiniMax native FP8 with block scales — the case the bypass exists for
        ({"model_type": "minimax_m2",
          "architectures": ["MiniMaxM2ForCausalLM"],
          "quantization_config": {"quant_method": "fp8",
                                  "weight_block_size": [128, 128]}}, True),
        ({"model_type": "minimax-m2",
          "architectures": ["MiniMax-M2ForCausalLM"],
          "quantization_config": {"quant_method": "fp8",
                                  "weight_block_size": [128, 128]}}, True),
        # right arch, but a bf16 checkpoint: nothing to bypass
        ({"model_type": "minimax_m2",
          "architectures": ["MiniMaxM2ForCausalLM"]}, False),
        # right arch, fp8 but per-tensor scales (no block size)
        ({"model_type": "minimax_m2",
          "architectures": ["MiniMaxM2ForCausalLM"],
          "quantization_config": {"quant_method": "fp8"}}, False),
        # native-FP8 checkpoint of an arch that does NOT declare the bypass
        ({"model_type": "qwen3_5_moe",
          "architectures": ["Qwen3_5MoeForCausalLM"],
          "quantization_config": {"quant_method": "fp8",
                                  "weight_block_size": [128, 128]}}, False),
        # unknown arch -> DefaultProfile -> no declaration
        ({"model_type": "llama", "architectures": ["LlamaForCausalLM"],
          "quantization_config": {"quant_method": "fp8",
                                  "weight_block_size": [128, 128]}}, False),
    ],
)
def test_fp8_rewrite_bypass_is_profile_driven(tmp_path, cfg, expected):
    """`streaming_model.py:98-104` was half config-derived and half a
    `model_type.startswith("minimax_m2")` literal. The architecture half is a
    static property, so it moved into the spec; the checkpoint half stays a
    config read because it varies per checkpoint."""
    from prismaquant.streaming_model import _bypass_hf_fp8_module_rewrite

    assert _bypass_hf_fp8_module_rewrite(_write_config(tmp_path, cfg)) is expected


def test_fp8_rewrite_bypass_is_false_without_a_config(tmp_path):
    from prismaquant.streaming_model import _bypass_hf_fp8_module_rewrite

    assert _bypass_hf_fp8_module_rewrite(str(tmp_path)) is False
