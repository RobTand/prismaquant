"""Adapter tests: GLM paths stay bit-identical; profile values flow through."""

from prismaquant.pact_frontier_profile import (
    GLM_LEGACY,
    frontier_manifest_overlay,
    glm_paths_identical,
    pact_bands,
    pact_cohort_from_profile,
    tp_splits_for_role,
)


def test_glm_cohort_matches_legacy():
    cohort = pact_cohort_from_profile()
    assert cohort["sample_range"] == [384, 448]
    assert cohort["raw_tokens_per_sequence"] == 512
    assert cohort["prefix_ids"] == [154822, 154824]
    assert cohort["local_prefix_rows"] == "excluded"
    assert cohort["input_contract"] == "prefixed_514"
    assert cohort["scored_positions_per_sequence"] == 511
    assert cohort["vocab_size"] == 154880
    assert cohort["global_original_tokens"] == 32768
    assert glm_paths_identical(cohort)


def test_glm_guard_rejects_missing_semantic_fields():
    cohort = pact_cohort_from_profile()
    assert glm_paths_identical(cohort)
    for field in ("local_prefix_rows", "input_contract"):
        stripped = {key: value for key, value in cohort.items() if key != field}
        assert not glm_paths_identical(stripped)


def test_glm_guard_rejects_wrong_semantic_fields():
    cohort = pact_cohort_from_profile()
    assert glm_paths_identical(cohort)
    wrong_rows = dict(cohort, local_prefix_rows=2)
    assert not glm_paths_identical(wrong_rows)
    wrong_contract = dict(cohort, input_contract="raw_512")
    assert not glm_paths_identical(wrong_contract)


def test_glm_bands_match_legacy_shape():
    bands = pact_bands()
    assert bands[0] == (0, 3)
    assert bands[-1][1] == GLM_LEGACY["num_layers"]
    assert bands == ((0, 3), (3, 9), (9, 15), (15, 21), (21, 27), (27, 33), (33, 39), (39, 45))


def test_tp_rule_keeps_down_split():
    assert tp_splits_for_role("down_proj") == 2
    assert tp_splits_for_role("gate_proj") == 1
    assert tp_splits_for_role("up_proj") == 1


def test_config_overrides_flow_through():
    config = {"num_hidden_layers": 32, "vocab_size": 100000}
    cohort = pact_cohort_from_profile(config=config)
    assert cohort["vocab_size"] == 100000
    # Prefix ids stay GLM without an explicit or declared value.
    assert cohort["prefix_ids"] == [154822, 154824]
    bands = pact_bands(config=config)
    assert bands[-1][1] == 32


def test_manifest_overlay_keeps_paths():
    manifest = {
        "d41_index": "/mnt/shared/fleet-ceo/rung-allowability/index.json",
        "price_tables": [],
        "anchors": "a",
        "heldout_preregistration": "b",
        "stack_scope_source": "c",
        "thresholds": {},
        "predictor": {},
    }
    overlay = frontier_manifest_overlay(manifest, config={"num_hidden_layers": 45})
    for key in manifest:
        assert overlay[key] == manifest[key]
    assert overlay["cohort"]["prefix_ids"] == [154822, 154824]
    assert overlay["cohort"]["local_prefix_rows"] == "excluded"
    assert overlay["cohort"]["input_contract"] == "prefixed_514"
    assert overlay["bands"][0] == [0, 3]


def _qwen3_declared_profile():
    from prismaquant.model_profiles import profile_from_config
    declared = {
        "model_type": "qwen3",
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_hidden_layers": 48,
        "vocab_size": 151936,
        "prefix_ids": [151935, 151934],
        "scored_positions_per_sequence": 1023,
        "raw_tokens_per_sequence": 1024,
    }
    profile = profile_from_config(declared)
    assert profile.name == "qwen3"
    assert profile._declared_config == declared
    return profile


def test_profile_declared_cohort_flows_without_explicit_config():
    profile = _qwen3_declared_profile()
    cohort = pact_cohort_from_profile(profile)
    assert cohort["vocab_size"] == 151936
    assert cohort["prefix_ids"] == [151935, 151934]
    assert cohort["scored_positions_per_sequence"] == 1023
    assert cohort["raw_tokens_per_sequence"] == 1024


def test_profile_declared_overlay_flows_without_explicit_config():
    profile = _qwen3_declared_profile()
    manifest = {
        "d41_index": "/mnt/shared/fleet-ceo/rung-allowability/index.json",
        "price_tables": [],
        "anchors": "a",
        "heldout_preregistration": "b",
        "stack_scope_source": "c",
        "thresholds": {},
        "predictor": {},
    }
    overlay = frontier_manifest_overlay(manifest, profile)
    for key in manifest:
        assert overlay[key] == manifest[key]
    assert overlay["cohort"]["vocab_size"] == 151936
    assert overlay["cohort"]["prefix_ids"] == [151935, 151934]
    assert overlay["cohort"]["scored_positions_per_sequence"] == 1023
    assert overlay["cohort"]["raw_tokens_per_sequence"] == 1024


def test_explicit_config_beats_declared_config():
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


def test_declared_nested_text_config_and_prefix_alias():
    from prismaquant.model_profiles import profile_from_config
    declared = {
        "model_type": "qwen3",
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_hidden_layers": 48,
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
    nested_explicit = {"text_config": {"vocab_size": 99999}}
    assert pact_cohort_from_profile(None, nested_explicit)["vocab_size"] == 99999
