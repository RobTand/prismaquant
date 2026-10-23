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
    from dataclasses import replace

    from conftest import down_convert_lane_table
    from prismaquant import lane_eligibility as lane
    from prismaquant.allocator_solver import Candidate
    from test_shape_runtime_prices import (
        COMMIT, SHA, SCOPE, DENSE_E4M3, E4M3, _doc, _payload, _row,
    )

    # Own a legacy-grammar fixture, not a current-runtime qualification. All
    # admission and launch checks still run; no GPU measurement is claimed.
    payload = down_convert_lane_table(_payload(), "tessera.lane-eligibility.v8")
    eligibility = lane._parse_table(
        payload["lane_eligibility"], payload["formats"], "", COMMIT, SHA,
        native_extensions=payload["native_extensions"],
    )
    samples = (0.01, 0.02, 0.04, 0.08, 0.16)
    doc = _doc([
        _row("dense", shape, E4M3, 1024, 8, DENSE_E4M3, samples)
        for shape in ("4x8", "4x16")
    ], regimes=(8,), tensor_parallel=1)
    table = srp.admit_shape_table(
        srp.parse_shape_table(doc), scope=replace(SCOPE, tensor_parallel=1),
        eligibility=eligibility,
    )
    fmt = f"{E4M3}_R1024"
    assignment = {unit: fmt for unit in ("first", "shared", "independent")}
    pricing = srp.build_shape_runtime_resources(
        table,
        {unit: [Candidate(fmt=fmt, bits_per_param=4.0, memory_bytes=128,
                          predicted_dloss=0.0)] for unit in assignment},
        option_members={(unit, fmt): {unit: fmt} for unit in assignment},
        member_shapes={"first": (4, 8), "shared": (4, 8), "independent": (4, 16)},
        member_structure={unit: "dense" for unit in assignment}, regime_m=8,
        published_formats={row["family"]: row for row in payload["formats"]},
    )
    assert not pricing.gaps
    first, shared, independent = (pricing.prefill[(unit, fmt)] for unit in assignment)
    assert first.source_id == shared.source_id != independent.source_id
    # Equal sample values do not make distinct measurements the same source.
    assert first.samples_ms == shared.samples_ms == independent.samples_ms == samples
    result = pricing.operator_sum_bootstrap(assignment, draws=4096, seed=17)
    assert result == pricing.operator_sum_bootstrap(assignment, draws=4096, seed=17)
    assert result["distinct_measurements"] == 2
    assert result["multiplicities"] == [1, 2]  # sorted source IDs: 4x16, 4x8
    assert result["samples_per_row"] == [5, 5]
    expected = mrp.bootstrap_sum([samples, samples], draws=4096, seed=17,
                                 multiplicities=[1, 2])
    resampled_independently = mrp.bootstrap_sum([samples] * 3, draws=4096, seed=17)
    for quantile in ("p2.5", "p50", "p97.5"):
        assert result[quantile] == pytest.approx(expected[quantile])
    # An independent draw per caller spuriously narrows the uncertainty band.
    shared_width = result["p97.5"] - result["p2.5"]
    independent_width = resampled_independently["p97.5"] - resampled_independently["p2.5"]
    assert shared_width > independent_width
    offset = pricing.operator_sum_bootstrap(assignment, draws=4096, seed=17, offset_ms=5.0)
    assert offset["p50"] == pytest.approx(result["p50"] + 5.0)
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


@pytest.fixture
def migration_evidence():
    return {
        "canonical_head": "a" * 40,
        "input_identities": {
            "harness": "b" * 64, "frontier": "c" * 64, "price_adapter": "d" * 64,
        },
        "caller_closure": {
            "harness": ["campaign.harness.main"],
            "frontier": ["campaign.frontier.main", "campaign.frontier.replay"],
            "price_adapter": ["campaign.price_adapter.price"],
        },
    }


@pytest.mark.parametrize("head_length", [40, 64])
def test_consumer_migration_gates_on_head_identities_closure(migration_evidence, head_length):
    migration_evidence["canonical_head"] = "a" * head_length
    with pytest.raises(ValueError, match="canonical adapter head"):
        require_pact_consumer_migration()
    with pytest.raises(ValueError, match="prismaquant#2483"):
        require_pact_consumer_migration()
    with pytest.raises(ValueError, match="input identities"):
        require_pact_consumer_migration(canonical_head="a" * 40)
    record = require_pact_consumer_migration(**migration_evidence)
    assert record["canonical_head"] == migration_evidence["canonical_head"]
    assert record["input_identities"] == migration_evidence["input_identities"]
    assert record["caller_closure"] == {
        role: tuple(callers) for role, callers in migration_evidence["caller_closure"].items()
    }
    assert record["prerequisites"] == list(PACT_CONSUMER_PREREQUISITES)
    migration_evidence["input_identities"]["frontier"] = None
    migration_evidence["caller_closure"]["frontier"].clear()
    assert record["input_identities"]["frontier"] == "c" * 64
    assert record["caller_closure"]["frontier"] == (
        "campaign.frontier.main", "campaign.frontier.replay",
    )


@pytest.mark.parametrize("head", [None, 1, True, "not-a-git-head", "a" * 7,
                                       "a" * 39, "g" * 40, "0" * 40, "a" * 40 + "\n"])
def test_consumer_migration_refuses_malformed_head(migration_evidence, head):
    migration_evidence["canonical_head"] = head
    with pytest.raises(ValueError, match="canonical adapter head") as refusal:
        require_pact_consumer_migration(**migration_evidence)
    assert all(reason in str(refusal.value) for reason in PACT_CONSUMER_PREREQUISITES)


@pytest.mark.parametrize("identities", [None, [], [("frontier", "b" * 64)],
                                             {}, {"frontier": "b" * 64}])
def test_consumer_migration_refuses_incomplete_identities(migration_evidence, identities):
    migration_evidence["input_identities"] = identities
    with pytest.raises(ValueError, match="input identities"):
        require_pact_consumer_migration(**migration_evidence)


@pytest.mark.parametrize("role", ["harness", "frontier", "price_adapter"])
@pytest.mark.parametrize("digest", [None, True, 1, "", "b" * 63, "g" * 64, "0" * 64])
def test_consumer_migration_refuses_unusable_identity(migration_evidence, role, digest):
    migration_evidence["input_identities"][role] = digest
    with pytest.raises(ValueError, match="input identities"):
        require_pact_consumer_migration(**migration_evidence)


@pytest.mark.parametrize("closure", [None, "campaign.frontier", [None],
                                          ["campaign.frontier.main"], {},
                                          {"frontier": ["campaign.frontier.main"]}])
def test_consumer_migration_refuses_incomplete_closure(migration_evidence, closure):
    migration_evidence["caller_closure"] = closure
    with pytest.raises(ValueError, match="caller closure"):
        require_pact_consumer_migration(**migration_evidence)


@pytest.mark.parametrize("role", ["harness", "frontier", "price_adapter"])
@pytest.mark.parametrize("callers", [None, [], "campaign.frontier.main", [None], [1],
    [""], [" "], ["campaign.frontier.main\n"], ["/workspace/frontier.py"],
    ["campaign.frontier.main", "campaign.frontier.main"]])
def test_consumer_migration_refuses_unusable_callers(migration_evidence, role, callers):
    migration_evidence["caller_closure"][role] = callers
    with pytest.raises(ValueError, match="caller closure"):
        require_pact_consumer_migration(**migration_evidence)


@pytest.mark.parametrize("field", ["input_identities", "caller_closure"])
def test_consumer_migration_refuses_unknown_roles(migration_evidence, field):
    migration_evidence[field]["typo"] = migration_evidence[field]["frontier"]
    with pytest.raises(ValueError):
        require_pact_consumer_migration(**migration_evidence)


def test_consumer_migration_reports_all_invalid_prerequisites():
    with pytest.raises(ValueError) as refusal:
        require_pact_consumer_migration(
            canonical_head="not-a-git-head", input_identities={"frontier": None},
            caller_closure=[None],
        )
    for reason in ("canonical adapter head", "input identities", "caller closure",
                   *PACT_CONSUMER_PREREQUISITES):
        assert reason in str(refusal.value)
