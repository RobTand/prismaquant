"""glm5_next attention fused groups: the construction runtime's stacked params (PQ #1837).

Image 487ecf187, vllm/models/glm5next/nvidia/model.py:752-768 loads q/k/v/b/f_a/g_a into
``in_proj_qkvbfg_a``, q_a/kv_a_proj_with_mqa into ``fused_qkv_a_proj`` and the indexer's
wk/weights_proj into ``wk_weights_proj``. The profile must name one group key for every member
of each, under both the checkpoint and the converted live spelling, and none for the Linears the
runtime loads alone. The pins do not move: these declarations only matter to a scope that
unpins attention.
"""
import pytest

from prismaquant.model_profiles.glm5_next import Glm5NextProfile

KDA_FUSED = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "forget_gate.f_a_proj", "g_a_proj")
KDA_ALONE = ("f_b_proj", "forget_gate.f_b_proj", "g_b_proj", "o_proj")
MLA_FUSED = ("q_a_proj", "kv_a_proj_with_mqa")
MLA_ALONE = ("q_b_proj", "kv_b_proj", "o_proj", "indexer.wq_b")
INDEXER_FUSED = ("indexer.wk", "indexer.weights_proj")


@pytest.fixture(scope="module")
def profile():
    result = Glm5NextProfile()
    kinds = ["linear_attention"] * 46
    kinds[3] = kinds[45] = "deepseek_sparse_attention"
    result._declare_config_document({"text_config": {"layer_types": kinds}})
    return result


@pytest.mark.parametrize("prefix", ["model.language_model.layers.0.", "model.layers.0."])
def test_kda_in_proj_members_share_one_group_in_both_spellings(profile, prefix):
    keys = {profile.fused_sibling_group(prefix + "self_attn." + leaf) for leaf in KDA_FUSED}
    assert keys == {prefix + "self_attn.in_proj_qkvbfg_a"}
    for leaf in KDA_ALONE:
        assert profile.fused_sibling_group(prefix + "self_attn." + leaf) is None, leaf


@pytest.mark.parametrize("layer", [3, 45])          # a body MLA layer and the MTP layer
def test_mla_and_indexer_pairs_share_their_runtime_modules(profile, layer):
    prefix = f"model.language_model.layers.{layer}.self_attn."
    assert {profile.fused_sibling_group(prefix + leaf) for leaf in MLA_FUSED} == {prefix + "fused_qkv_a_proj"}
    assert ({profile.fused_sibling_group(prefix + leaf) for leaf in INDEXER_FUSED}
            == {prefix + "indexer.wk_weights_proj"})
    for leaf in MLA_ALONE:
        assert profile.fused_sibling_group(prefix + leaf) is None, leaf


def test_mlp_groups_are_unchanged(profile):
    p = "model.language_model.layers.5.mlp.shared_experts."
    assert {profile.fused_sibling_group(p + r) for r in ("gate_proj", "up_proj")} == {p + "gate_up_proj"}
    assert profile.fused_sibling_group(p + "down_proj") is None
    d = "model.language_model.layers.0.mlp."
    assert {profile.fused_sibling_group(d + r) for r in ("gate_proj", "up_proj")} == {d + "gate_up_proj"}


@pytest.mark.parametrize("prefix", ["model.language_model.layers.0.", "model.layers.0."])
@pytest.mark.parametrize("leaf", KDA_FUSED)
def test_kda_member_without_declared_config_refuses(prefix, leaf):
    """A hand-built profile holds no config.json, so a KDA member refuses (PQ #2456)."""
    name = prefix + "self_attn." + leaf
    with pytest.raises(ValueError, match="declared config"):
        Glm5NextProfile().fused_sibling_group(name)


@pytest.mark.parametrize("prefix", ["model.language_model.layers.5.", "model.layers.5."])
def test_shared_expert_gate_needs_no_declared_config(prefix):
    """A shared-expert gate still has an owner without a config (PQ #2456)."""
    name = prefix + "mlp.shared_experts.gate_proj"
    assert Glm5NextProfile().fused_sibling_group(name) == (
        prefix + "mlp.shared_experts.gate_up_proj")


@pytest.mark.parametrize("prefix", ["model.language_model.layers.3.", "model.layers.3."])
def test_config_independent_attention_names_keep_their_owners(prefix):
    profile = Glm5NextProfile()
    attention = prefix + "self_attn."
    for leaf in MLA_FUSED:
        assert profile.fused_sibling_group(attention + leaf) == attention + "fused_qkv_a_proj"
    for leaf in INDEXER_FUSED:
        assert profile.fused_sibling_group(attention + leaf) == attention + "indexer.wk_weights_proj"
    for leaf in KDA_ALONE + MLA_ALONE:
        assert profile.fused_sibling_group(attention + leaf) is None, leaf


def test_attention_stays_pinned_and_no_group_is_partly_pinned(profile):
    """Declaring the groups moves no pin, and a group is pinned whole or not at all."""
    prefix = "model.language_model.layers.3.self_attn."
    for leaf in KDA_FUSED + KDA_ALONE + MLA_FUSED + MLA_ALONE + INDEXER_FUSED:
        assert profile.is_pinned_name(prefix + leaf + ".weight"), leaf
    for target, members in profile.fused_sibling_leaf_mapping().items():
        prefix = "model.language_model.layers.3."
        suffix = "self_attn.indexer." if target == "wk_weights_proj" else "self_attn."
        if target == "gate_up_proj":
            suffix = "mlp."
        pinned = {profile.is_pinned_name(prefix + suffix + m) for m in members}
        assert len(pinned) == 1, (target, pinned)
