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
    assert cohort["scored_positions_per_sequence"] == 511
    assert cohort["vocab_size"] == 154880
    assert cohort["global_original_tokens"] == 32768
    assert glm_paths_identical(cohort)


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
    # Prefix ids stay GLM until the profile owner exposes them.
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
    assert overlay["bands"][0] == [0, 3]
