"""Consumer checks: profile scope drives price, wire, sample and refusal paths."""

import pytest

from prismaquant import measured_runtime_prices as mrp
from prismaquant import shape_runtime_prices as srp
from prismaquant.pact_frontier_profile import (
    PACT_CONSUMER_PREREQUISITES,
    glm_paths_identical,
    pact_bands,
    pact_cohort_from_profile,
    require_pact_consumer_migration,
    tp_splits_for_role,
)


def _glm_profile():
    from prismaquant.model_profiles import profile_from_config

    return profile_from_config(
        {
            "model_type": "glm5_next",
            "architectures": ["Glm5NextForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 45,
                "hidden_size": 4096,
                "vocab_size": 154880,
            },
        }
    )


def _qwen3_profile():
    from prismaquant.model_profiles import profile_from_config

    return profile_from_config(
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3MoeForCausalLM"],
            "num_hidden_layers": 48,
            "hidden_size": 2048,
            "vocab_size": 151936,
            "prefix_ids": [151935, 151934],
            "scored_positions_per_sequence": 1023,
            "raw_tokens_per_sequence": 1024,
            "sample_range": [0, 64],
            "local_prefix_rows": "included",
            "input_contract": "raw",
            "global_original_tokens": 65536,
        }
    )


def test_consumer_prices_dense_operators_at_profile_tp_world():
    for profile in (_glm_profile(), _qwen3_profile()):
        world = tp_splits_for_role("down_proj", profile)
        assert world == 2
        fused = srp.served_operator(
            {"L.mlp.gate_proj": (12288, 4096), "L.mlp.up_proj": (12288, 4096)},
            structure="dense",
            tensor_parallel=world,
        )
        assert fused == "12288x4096"
        down = srp.served_operator(
            {"L.mlp.down_proj": (4096, 12288)},
            structure="dense",
            tensor_parallel=world,
        )
        assert down == "4096x6144"
        local = mrp.rank_local_member_shapes(
            {"m.down_proj": (4096, 2048)}, tensor_parallel=world
        )
        assert local == {"m.down_proj": (4096, 1024)}


def test_consumer_prices_routed_operator_at_profile_tp_world():
    profile = _glm_profile()
    world = tp_splits_for_role("down_proj", profile)
    experts = {
        f"m.experts.{e}.{role}": shape
        for e in range(2)
        for role, shape in (
            ("gate_proj", (8, 16)),
            ("up_proj", (8, 16)),
            ("down_proj", (16, 8)),
        )
    }
    assert srp.served_operator(
        experts, structure="routed_moe", tensor_parallel=world
    ) == "E2:w13=8x16:w2=16x4"
    ragged = dict(experts)
    ragged["m.experts.1.down_proj"] = (16, 4)
    with pytest.raises(srp.ShapeRuntimeError):
        srp.served_operator(ragged, structure="routed_moe", tensor_parallel=world)
    with pytest.raises(srp.ShapeRuntimeError, match="no tensor-parallel cut rule"):
        srp.served_operator(
            {"m.down_proj": (16, 8), "m.plain": (16, 8)},
            structure="dense",
            tensor_parallel=world,
        )


def _rank_row(world, wire, resident, prefill, priced=None):
    ranks = tuple(
        mrp.RankResources(
            rank=rank,
            resident_bytes=resident,
            peak_scratch_bytes=7,
            activation_bytes=9,
            workspace_resident_bytes=0,
            workspace_sha256="a" * 64,
            bound_sha256="b" * 64,
        )
        for rank in range(world)
    )
    medians = {"prefill": tuple(prefill for _ in range(world)), "decode": (None,) * world}
    return mrp.RuntimeRankResources(
        prefill_ms=prefill if priced is None else priced,
        decode_ms=None,
        world_size=world,
        rank_medians_ms=medians,
        wire_bytes=wire,
        wire_sha256="c" * 64,
        ranks=ranks,
    )


def test_consumer_wire_bytes_count_once_per_owner():
    world = tp_splits_for_role("down_proj", _glm_profile())
    totals = mrp.compose_rank_totals(
        [_rank_row(world, 1000, 100, 10.0), _rank_row(world, 500, 50, 10.0)]
    )
    assert totals.world_size == world
    assert totals.wire_bytes == 1500
    assert totals.resident_bytes == (150, 150)
    scalar = mrp.RuntimeResources(
        prefill_ms=1.0,
        decode_ms=None,
        serialized_bytes=10,
        resident_bytes=10,
        peak_scratch_bytes=1,
        activation_bytes=1,
    )
    with pytest.raises(mrp.RuntimePriceError, match="mixes a scalar row"):
        mrp.compose_rank_totals([_rank_row(world, 1000, 100, 10.0), scalar])


def test_consumer_sample_identity_draws_each_row_once():
    first = mrp.bootstrap_sum([[1.0, 1.0, 1.0]], draws=64, seed=7)
    second = mrp.bootstrap_sum([[1.0, 1.0, 1.0]], draws=64, seed=7)
    assert first == second
    assert first["samples_per_row"] == [3]
    assert first["seed"] == 7
    shared = mrp.bootstrap_sum(
        [[1.0, 1.0, 1.0]], draws=64, seed=7, multiplicities=[3]
    )
    assert shared["multiplicities"] == [3]
    assert shared["p50"] == pytest.approx(3 * first["p50"])
    offset = mrp.bootstrap_sum([[1.0, 1.0, 1.0]], draws=64, seed=7, offset_ms=5.0)
    assert offset["p50"] == pytest.approx(first["p50"] + 5.0)
    with pytest.raises(mrp.RuntimePriceError, match="slowest rank"):
        _rank_row(2, 100, 100, 9.0, priced=8.0)


def test_consumer_refuses_bad_scope_geometry_and_inputs():
    from prismaquant.model_profiles import profile_from_config

    bare = profile_from_config(
        {"model_type": "lfm2_moe", "architectures": ["Lfm2MoeForCausalLM"]}
    )
    assert not bare.pact_scope_declared()
    with pytest.raises(ValueError, match="declares no PACT scope"):
        pact_cohort_from_profile(bare, {"num_hidden_layers": 24})
    with pytest.raises(ValueError, match="declares no PACT scope"):
        pact_bands(profile=bare, config={"num_hidden_layers": 24})
    with pytest.raises(ValueError, match="declares no PACT"):
        tp_splits_for_role("down_proj", bare)
    with pytest.raises(mrp.RuntimePriceError, match="not divisible"):
        mrp.rank_local_member_shapes({"m.down_proj": (4096, 2047)}, tensor_parallel=2)
    with pytest.raises(mrp.RuntimePriceError, match="2-D"):
        mrp.rank_local_member_shapes({"m.down_proj": (4096,)}, tensor_parallel=2)
    with pytest.raises(srp.ShapeRuntimeError, match="structure must be"):
        srp.served_operator(
            {"m.down_proj": (16, 8)}, structure="fused", tensor_parallel=1
        )
    with pytest.raises(mrp.RuntimePriceError, match="at least 1"):
        mrp.bootstrap_sum([[1.0]], draws=0, seed=1)


def test_consumer_uses_explicit_measurement_inputs():
    glm = _glm_profile()
    assert glm_paths_identical(pact_cohort_from_profile(glm, {"num_hidden_layers": 45}))
    qwen = _qwen3_profile()
    cohort = pact_cohort_from_profile(qwen)
    assert not glm_paths_identical(cohort)
    assert cohort["vocab_size"] == 151936
    assert "154822" not in str(cohort["prefix_ids"])
    got = pact_cohort_from_profile(qwen, {"vocab_size": 100001, "prefix_ids": [1, 2]})
    assert got["vocab_size"] == 100001
    assert got["prefix_ids"] == [1, 2]
    assert got["sample_range"] == [0, 64]
    assert pact_bands(profile=qwen, config={"num_hidden_layers": 12}) == (
        (0, 6),
        (6, 12),
    )


def test_consumer_migration_gates_on_head_identities_closure():
    with pytest.raises(ValueError, match="canonical adapter head"):
        require_pact_consumer_migration()
    with pytest.raises(ValueError, match="prismaquant#2483"):
        require_pact_consumer_migration()
    with pytest.raises(ValueError, match="input identities"):
        require_pact_consumer_migration(canonical_head="a" * 40)
    record = require_pact_consumer_migration(
        canonical_head="a" * 40,
        input_identities={"frontier": "b" * 64},
        caller_closure=["prismaquant.shape_runtime_prices"],
    )
    assert record["canonical_head"] == "a" * 40
    assert record["input_identities"] == {"frontier": "b" * 64}
    assert record["caller_closure"] == ("prismaquant.shape_runtime_prices",)
    assert record["prerequisites"] == list(PACT_CONSUMER_PREREQUISITES)
