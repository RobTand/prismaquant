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


def _old_glm_cohort(config, declared):
    """Legacy adapter math at 8d441ebf7: explicit, declared, then frozen."""
    def _entry(cfg, name):
        if not isinstance(cfg, dict):
            return None
        if cfg.get(name) is not None:
            return cfg[name]
        text = cfg.get("text_config")
        if isinstance(text, dict) and text.get(name) is not None:
            return text[name]
        return None
    prefix = _entry(config, "prefix_ids")
    if prefix is None:
        prefix = _entry(config, "serving_prefix_ids")
    if prefix is None:
        prefix = _entry(declared, "prefix_ids")
    if prefix is None:
        prefix = _entry(declared, "serving_prefix_ids")
    if prefix is None:
        prefix = GLM_LEGACY["prefix_ids"]
    vocab = _entry(config, "vocab_size")
    if vocab is None:
        vocab = _entry(declared, "vocab_size")
    if vocab is None:
        vocab = GLM_LEGACY["vocab_size"]
    scored = _entry(config, "scored_positions_per_sequence")
    if scored is None:
        scored = _entry(declared, "scored_positions_per_sequence")
    if scored is None:
        scored = GLM_LEGACY["scored_positions"]
    raw = _entry(config, "raw_tokens_per_sequence")
    if raw is None:
        raw = _entry(declared, "raw_tokens_per_sequence")
    if raw is None:
        raw = GLM_LEGACY["raw_tokens_per_sequence"]
    return {
        "sample_range": list(GLM_LEGACY["sample_range"]),
        "raw_tokens_per_sequence": int(raw),
        "prefix_ids": list(prefix),
        "local_prefix_rows": GLM_LEGACY["local_prefix_rows"],
        "input_contract": GLM_LEGACY["input_contract"],
        "global_original_tokens": GLM_LEGACY["global_original_tokens"],
        "scored_positions_per_sequence": int(scored),
        "vocab_size": int(vocab),
    }


def _old_glm_bands(count):
    """Legacy adapter bands at 8d441ebf7: dense head plus width-6 tail."""
    bands = [GLM_LEGACY["dense_layers"]]
    start = GLM_LEGACY["dense_layers"][1]
    while start < count:
        bands.append((start, min(start + GLM_LEGACY["band_width"], count)))
        start += GLM_LEGACY["band_width"]
    return tuple(bands)


def test_glm_equivalence_old_adapter_vs_shared_contract():
    """Identical GLM inputs give equal bands, cohort, roles, dims, TP cuts."""
    profile = _glm_profile()
    config = _glm_config()
    cohort = pact_cohort_from_profile(profile, config)
    legacy = _old_glm_cohort(config, profile._declared_config)
    assert cohort == legacy
    assert glm_paths_identical(cohort)
    assert glm_paths_identical(legacy)
    # Frozen prefix-row and input contract survive unchanged.
    assert cohort["local_prefix_rows"] == GLM_LEGACY["local_prefix_rows"]
    assert cohort["input_contract"] == GLM_LEGACY["input_contract"]
    assert cohort["sample_range"] == list(GLM_LEGACY["sample_range"])
    assert cohort["raw_tokens_per_sequence"] == GLM_LEGACY["raw_tokens_per_sequence"]
    assert cohort["global_original_tokens"] == GLM_LEGACY["global_original_tokens"]
    assert cohort["prefix_ids"] == list(GLM_LEGACY["prefix_ids"])
    bands = pact_bands(profile=profile, config=config)
    assert bands == _old_glm_bands(GLM_LEGACY["num_layers"])
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
    assert num_layers_from_profile(profile, config) == GLM_LEGACY["num_layers"]


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
        "sample_range": [0, 64],
        "local_prefix_rows": "included",
        "input_contract": "raw",
        "global_original_tokens": 65536,
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
    assert cohort["sample_range"] == [0, 64]
    assert cohort["local_prefix_rows"] == "included"
    assert cohort["input_contract"] == "raw"
    assert cohort["global_original_tokens"] == 65536
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
        "sample_range": [10, 20],
        "local_prefix_rows": "excluded",
        "input_contract": "prefixed_514",
        "global_original_tokens": 111,
    }
    cohort = pact_cohort_from_profile(profile, explicit)
    assert cohort["vocab_size"] == 100001
    assert cohort["prefix_ids"] == [1, 2]
    assert cohort["scored_positions_per_sequence"] == 777
    assert cohort["raw_tokens_per_sequence"] == 778
    assert cohort["sample_range"] == [10, 20]
    assert cohort["local_prefix_rows"] == "excluded"
    assert cohort["input_contract"] == "prefixed_514"
    assert cohort["global_original_tokens"] == 111


def test_qwen3_declared_nested_text_config_and_prefix_alias():
    from prismaquant.model_profiles import profile_from_config

    declared = {
        "model_type": "qwen3",
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "serving_prefix_ids": [11, 12],
        "sample_range": [5, 9],
        "local_prefix_rows": "included",
        "input_contract": "raw",
        "global_original_tokens": 777,
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
    assert cohort["sample_range"] == [5, 9]
    assert cohort["global_original_tokens"] == 777
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

def test_explicit_prefix_ids_beat_declared_prefix_ids():
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
        "sample_range": [0, 64],
        "local_prefix_rows": "included",
        "input_contract": "raw",
        "global_original_tokens": 65536,
    }
    profile = profile_from_config(declared)
    explicit = {"serving_prefix_ids": [11, 12]}
    assert pact_cohort_from_profile(profile, explicit)["prefix_ids"] == [11, 12]
    alias_only = dict(declared)
    del alias_only["prefix_ids"]
    alias_only["serving_prefix_ids"] = [11, 12]
    alias_profile = profile_from_config(alias_only)
    assert pact_cohort_from_profile(alias_profile, explicit)["prefix_ids"] == [11, 12]
    direct = {"prefix_ids": [1, 2]}
    assert pact_cohort_from_profile(alias_profile, direct)["prefix_ids"] == [1, 2]
    assert pact_cohort_from_profile(profile, direct)["prefix_ids"] == [1, 2]


def test_explicit_text_config_beats_top_level_declared_dimensions():
    from prismaquant.model_profiles import profile_from_config

    declared = {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "num_hidden_layers": 45,
        "hidden_size": 4096,
        "text_config": {
            "num_hidden_layers": 44,
            "hidden_size": 4095,
            "vocab_size": 154880,
        },
    }
    profile = profile_from_config(declared)
    assert profile.name == "glm5_next"
    explicit = {"text_config": {"num_hidden_layers": 43, "hidden_size": 4094}}
    assert num_layers_from_profile(profile, explicit) == 43
    assert pact_hidden_layout(profile, explicit) == {
        "hidden_streams": 4, "hidden_size": 4094,
    }


def test_declared_dimensions_supply_missing_explicit_dimensions():
    from prismaquant.model_profiles import profile_from_config

    declared = {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 45,
            "hidden_size": 4096,
            "vocab_size": 154880,
        },
    }
    profile = profile_from_config(declared)
    assert num_layers_from_profile(profile, None) == 45
    assert pact_hidden_layout(profile, None) == {
        "hidden_streams": 4, "hidden_size": 4096,
    }
    assert pact_cohort_from_profile(profile, None)["vocab_size"] == 154880


def test_inconsistent_explicit_dimensions_refuse():
    profile = _glm_profile()
    with pytest.raises(ValueError, match="positive integer"):
        num_layers_from_profile(profile, {"num_hidden_layers": 0})
    with pytest.raises(ValueError, match="positive integer"):
        pact_hidden_layout(profile, {"hidden_size": -1})
    with pytest.raises(ValueError, match="positive integer"):
        profile.pact_vocab_size({"vocab_size": "big"})


def test_qwen3_missing_measurement_fields_refuse():
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
    with pytest.raises(ValueError, match="explicit cohort fields"):
        pact_cohort_from_profile(profile)


def test_glm_legacy_guard_uses_frozen_values_without_profile():
    import prismaquant.pact_frontier_profile as adapter

    frozen = adapter._legacy_cohort()
    assert frozen["sample_range"] == list(GLM_LEGACY["sample_range"])
    assert frozen["prefix_ids"] == list(GLM_LEGACY["prefix_ids"])
    assert frozen["local_prefix_rows"] == GLM_LEGACY["local_prefix_rows"]
    assert frozen["input_contract"] == GLM_LEGACY["input_contract"]
    assert glm_paths_identical(frozen)
    assert not glm_paths_identical(dict(frozen, input_contract="raw"))
