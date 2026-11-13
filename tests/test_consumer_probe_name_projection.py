"""The shared name-projection layer the probe's name derivation used.

Pins the NameProjection rules (2026-08-22, ``walker/consumer-probe``):

1. :meth:`NameProjection.block_id` is the shared
   ``decision_units.block_id_from_qname`` rule, not a positional ``[:3]``
   slice of the qname. On every qname shape reachable in this repo today the
   two agree exactly.
2. The layer refuses to run without a profile: the constructor rejects
   ``None`` rather than degrading every mapping to identity.
3. Names are logical whole-tensor facts: nothing accepts a TP degree.

No GPU, no checkpoint weights, no network.
"""
from __future__ import annotations

import pytest

from prismaquant.decision_units import block_id_from_qname
from prismaquant.model_profiles.qwen3 import Qwen3Profile
from prismaquant.name_projection import NameProjection


# --------------------------------------------------------------- block id


def test_packed_block_filter_matches_the_legacy_slice_on_reachable_shapes():
    """Value preservation: for every packed-experts qname this repo can
    produce today, NameProjection.block_id returns exactly what the old
    private `".".join(qname.split(".")[:3])` returned, so no stats row
    can gain or lose inclusion."""
    proj = NameProjection(Qwen3Profile())
    # NOTE: every live tree the probe can be handed today starts
    # with a two-component layer prefix (`model.layers`, `mtp.layers`),
    # so the `layers.<int>` pair sits at component depth 1 and the two
    # derivations coincide exactly. Deeper wrappers diverge (pinned as a
    # divergence below); a relative `layers.N...` root is a CHECKPOINT
    # spelling that never reaches this filter.
    for qname in (
        "model.layers.7.mlp.experts",          # body MoE block
        "model.layers.0.mlp.experts",          # first block
        "mtp.layers.0.ffn.experts",            # MTP shard wrapper
    ):
        assert proj.block_id(qname) == ".".join(qname.split(".")[:3])


def test_block_id_is_the_shared_rule_not_a_private_slice():
    """The projection delegates to the one shared block-id
    rule, and diverges from the legacy slice exactly where the slice was
    only correct by accident (a layer prefix deeper than two components).
    The divergence is the layer's explicit rule winning over a heuristic;
    it is documented in the migration report, not smuggled in."""
    acc_proj = NameProjection(Qwen3Profile())
    assert acc_proj.block_id("model.layers.2.self_attn.q_proj") == (
        block_id_from_qname("model.layers.2.self_attn.q_proj"))
    deep = "model.language_model.layers.7.mlp.experts"
    assert acc_proj.block_id(deep) == "model.language_model.layers.7"
    assert ".".join(deep.split(".")[:3]) != acc_proj.block_id(deep)


# ------------------------------------------------- construction refusal


def test_projection_seam_refuses_a_none_profile():
    """Requirement: a refusal propagates instead of degrading. The probe
    builds its projection from a resolved profile; if it ever handed the
    layer a None profile, construction itself must fail loudly."""
    with pytest.raises(ValueError):
        NameProjection(None)


def test_projection_seam_carries_no_tp_degree():
    """TP invariant 1: names are logical whole-tensor facts; nothing in
    the seam accepts a rank/shard/degree."""
    assert not hasattr(NameProjection(Qwen3Profile()), "tp_degree")
    with pytest.raises(TypeError):
        NameProjection(Qwen3Profile(), tp_degree=8)  # type: ignore[call-arg]
