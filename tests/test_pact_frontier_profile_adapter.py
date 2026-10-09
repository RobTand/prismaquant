"""Adapter tests: shared contract drives GLM and non-GLM resolution."""

import pytest

from prismaquant.pact_frontier_profile import (
    GLM_LEGACY,
    frontier_manifest_overlay,
    glm_paths_identical,
    num_layers_from_profile,
    pact_bands,
    pact_cohort_from_profile,
    pact_hidden_layout,
    tp_splits_for_role,
)


def _glm_profile(**extra):
    from prismaquant.model_profiles import profile_from_config

    declared = {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 45,
            "num_nextn_predict_layers": 1,
            "hidden_size": 4096,
            "vocab_size": 154880,
        },
    }
    declared["text_config"].update(extra)
    profile = profile_from_config(declared)
    assert profile.name == "glm5_next"
    return profile


def _glm_config(**extra):
    config = {"num_hidden_layers": 45, "hidden_size": 4096}
    config.update(extra)
    return config


def test_glm_shared_contract_matches_legacy_oracle():
    profile = _glm_profile()
    cohort = pact_cohort_from_profile(profile, _glm_config())
    assert cohort["sample_range"] == [384, 448]
    assert cohort["raw_tokens_per_sequence"] == 512
    assert cohort["prefix_ids"] == [154822, 154824]
    assert cohort["local_prefix_rows"] == "excluded"
    assert cohort["input_contract"] == "prefixed_514"
    assert cohort["scored_positions_per_sequence"] == 511
    assert cohort["vocab_size"] == 154880
    assert cohort["global_original_tokens"] == 32768
    assert glm_paths_identical(cohort)
    bands = pact_bands(profile=profile, config=_glm_config())
    assert bands == (
        (0, 3), (3, 9), (9, 15), (15, 21),
        (21, 27), (27, 33), (33, 39), (39, 45),
    )
    assert num_layers_from_profile(profile, _glm_config()) == 45
    assert tp_splits_for_role("down_proj", profile) == 2
    assert tp_splits_for_role("gate_proj", profile) == 1
    assert tp_splits_for_role("up_proj", profile) == 1
    layout = pact_hidden_layout(profile, _glm_config())
    assert layout == {"hidden_streams": 4, "hidden_size": 4096}


def test_glm_legacy_oracle_values_stay_frozen():
    assert GLM_LEGACY["num_layers"] == 45
    assert GLM_LEGACY["prefix_ids"] == (154822, 154824)
    assert GLM_LEGACY["local_prefix_rows"] == "excluded"
    assert GLM_LEGACY["input_contract"] == "prefixed_514"
    assert GLM_LEGACY["vocab_size"] == 154880
    assert GLM_LEGACY["hidden_streams"] == 4
    assert GLM_LEGACY["hidden_size"] == 4096


def test_glm_guard_rejects_missing_semantic_fields():
    profile = _glm_profile()
    cohort = pact_cohort_from_profile(profile, _glm_config())
    assert glm_paths_identical(cohort)
    for field in ("local_prefix_rows", "input_contract"):
        stripped = {key: value for key, value in cohort.items() if key != field}
        assert not glm_paths_identical(stripped)


def test_glm_guard_rejects_wrong_semantic_fields():
    profile = _glm_profile()
    cohort = pact_cohort_from_profile(profile, _glm_config())
    assert glm_paths_identical(cohort)
    wrong_rows = dict(cohort, local_prefix_rows="included")
    assert not glm_paths_identical(wrong_rows)
    wrong_contract = dict(cohort, input_contract="raw")
    assert not glm_paths_identical(wrong_contract)


def test_glm_config_overrides_flow_through_contract():
    profile = _glm_profile()
    config = _glm_config(vocab_size=100000)
    cohort = pact_cohort_from_profile(profile, config)
    assert cohort["vocab_size"] == 100000
    assert cohort["prefix_ids"] == [154822, 154824]


def test_glm_declared_config_supplies_layer_count():
    profile = _glm_profile()
    assert num_layers_from_profile(profile, None) == 45
    assert pact_hidden_layout(profile, {"hidden_size": 4096}) == {
        "hidden_streams": 4, "hidden_size": 4096,
    }


def test_glm_manifest_overlay_keeps_paths():
    profile = _glm_profile()
    manifest = {
        "d41_index": "/mnt/shared/fleet-ceo/rung-allowability/index.json",
        "price_tables": [],
        "anchors": "a",
        "heldout_preregistration": "b",
        "stack_scope_source": "c",
        "thresholds": {},
        "predictor": {},
    }
    overlay = frontier_manifest_overlay(manifest, profile, _glm_config())
    for key in manifest:
        assert overlay[key] == manifest[key]
    assert overlay["model_profile"] == "glm5_next"
    assert overlay["cohort"]["prefix_ids"] == [154822, 154824]
    assert overlay["cohort"]["local_prefix_rows"] == "excluded"
    assert overlay["cohort"]["input_contract"] == "prefixed_514"
    assert overlay["bands"][0] == [0, 3]


def test_glm_equivalence_old_adapter_vs_shared_contract():
    """Identical GLM inputs give equal bands, cohort, roles, dims, TP cuts."""
    import prismaquant.pact_frontier_profile as adapter

    profile = _glm_profile()
    config = _glm_config()
    cohort = pact_cohort_from_profile(profile, config)
    legacy = dict(
        adapter.pact_cohort_from_profile(profile, config),
        sample_range=list(GLM_LEGACY["sample_range"]),
        raw_tokens_per_sequence=GLM_LEGACY["raw_tokens_per_sequence"],
        prefix_ids=list(GLM_LEGACY["prefix_ids"]),
        local_prefix_rows=GLM_LEGACY["local_prefix_rows"],
        input_contract=GLM_LEGACY["input_contract"],
        global_original_tokens=GLM_LEGACY["global_original_tokens"],
        scored_positions_per_sequence=GLM_LEGACY["scored_positions"],
        vocab_size=GLM_LEGACY["vocab_size"],
    )
    assert cohort == legacy
    assert glm_paths_identical(cohort)
    assert glm_paths_identical(legacy)
    bands = pact_bands(profile=profile, config=config)
    assert bands[0] == GLM_LEGACY["dense_layers"]
    assert bands[-1][1] == GLM_LEGACY["num_layers"]
    roles = {
        role: tp_splits_for_role(role, profile)
        for role in ("down_proj", "gate_proj", "up_proj")
    }
    assert roles == {
        "down_proj": GLM_LEGACY["tp_splits_down_proj"],
        "gate_proj": GLM_LEGACY["tp_splits_other"],
        "up_proj": GLM_LEGACY["tp_splits_other"],
    }
    layout = pact_hidden_layout(profile, config)
    assert layout == {
        "hidden_streams": GLM_LEGACY["hidden_streams"],
        "hidden_size": GLM_LEGACY["hidden_size"],
    }


def _qwen3_declared_profile():
    from prismaquant.model_profiles import profile_from_config

    declared = {
        "model_type": "qwen3",
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "vocab_size": 151936,
        "prefix_ids": [151935, 151934],
        "scored_positions_per_sequence": 1023,
        "raw_tokens_per_sequence": 1024,
    }
    profile = profile_from_config(declared)
    assert profile.name == "qwen3"
    return profile


def test_qwen3_resolves_declared_scope_without_glm_constants():
    profile = _qwen3_declared_profile()
    assert profile.pact_scope_declared()
    cohort = pact_cohort_from_profile(profile)
    assert cohort["vocab_size"] == 151936
    assert cohort["prefix_ids"] == [151935, 151934]
    assert cohort["scored_positions_per_sequence"] == 1023
    assert cohort["raw_tokens_per_sequence"] == 1024
    assert cohort["local_prefix_rows"] == "included"
    assert cohort["input_contract"] == "raw"
    assert not glm_paths_identical(cohort)
    bands = pact_bands(profile=profile, config={"num_hidden_layers": 48})
    assert bands[0] == (0, 6)
    assert bands[-1][1] == 48
    assert tp_splits_for_role("down_proj", profile) == 2
    assert tp_splits_for_role("gate_proj", profile) == 1
    layout = pact_hidden_layout(profile, {"hidden_size": 2048})
    assert layout == {"hidden_streams": 1, "hidden_size": 2048}


def test_qwen3_explicit_config_beats_declared_config():
    profile = _qwen3_declared_profile()
    explicit = {
        "vocab_size": 100001,
        "prefix_ids": [1, 2],
        "scored_positions_per_sequence": 777,
        "raw_tokens_per_sequence": 778,
    }
    cohort = pact_cohort_from_profile(profile, explicit)
    assert cohort["vocab_size"] == 100001
    assert cohort["prefix_ids"] == [1, 2]
    assert cohort["scored_positions_per_sequence"] == 777
    assert cohort["raw_tokens_per_sequence"] == 778


def test_qwen3_declared_nested_text_config_and_prefix_alias():
    from prismaquant.model_profiles import profile_from_config

    declared = {
        "model_type": "qwen3",
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "serving_prefix_ids": [11, 12],
        "text_config": {
            "vocab_size": 151937,
            "scored_positions_per_sequence": 779,
            "raw_tokens_per_sequence": 780,
        },
    }
    profile = profile_from_config(declared)
    assert profile.name == "qwen3"
    cohort = pact_cohort_from_profile(profile)
    assert cohort["prefix_ids"] == [11, 12]
    assert cohort["vocab_size"] == 151937
    assert cohort["scored_positions_per_sequence"] == 779
    assert cohort["raw_tokens_per_sequence"] == 780
    explicit = {"prefix_ids": [1, 2], "serving_prefix_ids": [11, 12]}
    assert pact_cohort_from_profile(profile, explicit)["prefix_ids"] == [1, 2]


def test_no_profile_refuses_without_glm_fallback():
    with pytest.raises(ValueError, match="declared model profile"):
        pact_cohort_from_profile(None, {"num_hidden_layers": 48})
    with pytest.raises(ValueError, match="declared model profile"):
        pact_bands(config={"num_hidden_layers": 48})
    with pytest.raises(ValueError, match="declared model profile"):
        tp_splits_for_role("down_proj", None)
    with pytest.raises(ValueError, match="declared model profile"):
        num_layers_from_profile(None, {"num_hidden_layers": 48})


def test_undeclared_scope_refuses():
    from prismaquant.model_profiles import profile_from_config

    profile = profile_from_config(
        {"model_type": "lfm2_moe", "architectures": ["Lfm2MoeForCausalLM"]}
    )
    assert not profile.pact_scope_declared()
    with pytest.raises(ValueError, match="declares no PACT scope"):
        pact_cohort_from_profile(profile, {"num_hidden_layers": 24})
    with pytest.raises(ValueError, match="declares no PACT scope"):
        pact_bands(profile=profile, config={"num_hidden_layers": 24})
    with pytest.raises(ValueError, match="declares no PACT"):
        tp_splits_for_role("down_proj", profile)


def test_inconsistent_declarations_refuse():
    from prismaquant.model_profiles.structure import PactScopeSpec

    with pytest.raises(ValueError, match="dense_layer_end"):
        PactScopeSpec.from_dict({"dense_layer_end": -1})
    with pytest.raises(ValueError, match="band_width"):
        PactScopeSpec.from_dict({"band_width": 0})
    with pytest.raises(ValueError, match="tp_splits"):
        PactScopeSpec.from_dict({"tp_splits": {"down_proj": 0}})
    with pytest.raises(ValueError, match="unsupported pact keys"):
        PactScopeSpec.from_dict({"unknown_field": 1})
    profile = _glm_profile()
    with pytest.raises(ValueError, match="positive integer"):
        pact_bands(profile=profile, config={"num_hidden_layers": 0})
    with pytest.raises(ValueError, match="dense_layer_end"):
        pact_bands(profile=profile, config={"num_hidden_layers": 2})
    from prismaquant.model_profiles import profile_from_config
    bare = profile_from_config(
        {"model_type": "qwen3", "architectures": ["Qwen3MoeForCausalLM"]}
    )
    with pytest.raises(ValueError, match="explicit cohort fields"):
        pact_cohort_from_profile(bare)


def test_dense_and_routed_boundaries():
    glm = _glm_profile()
    bands = pact_bands(profile=glm, config={"num_hidden_layers": 45})
    assert bands[0] == (0, 3)
    assert all(stop - start == 6 for start, stop in bands[1:-1])
    qwen = _qwen3_declared_profile()
    routed_only = pact_bands(profile=qwen, config={"num_hidden_layers": 12})
    assert routed_only == ((0, 6), (6, 12))
    single = pact_bands(profile=qwen, config={"num_hidden_layers": 4})
    assert single == ((0, 4),)


def test_role_dependent_tp_dimensions():
    glm = _glm_profile()
    assert tp_splits_for_role("down_proj", glm) == 2
    assert tp_splits_for_role("gate_proj", glm) == 1
    assert tp_splits_for_role("up_proj", glm) == 1
    assert tp_splits_for_role("q_proj", glm) == 1
    qwen = _qwen3_declared_profile()
    assert tp_splits_for_role("down_proj", qwen) == 2
    assert tp_splits_for_role("w2", qwen) == 1
