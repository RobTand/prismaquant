"""Real DSv4 CPU contracts survive the unsupported-config import predecessor."""

import pytest
from test_layer_streaming_mxfp4_isolation import native_dsv4_session


@pytest.mark.parametrize("targets", [
    pytest.param((
        "../test_deepseek_v4_profile.py::"
        "test_rope_axis_mapping_matches_the_vendored_definition",
    ), id="rope-axis-1990-1992"),
    pytest.param((
        "../test_model_walk.py::test_dsv4_real_cpu_walk_discovers_and_decides_wo_a",
        "../test_model_walk.py::test_dsv4_walk_fails_without_the_profile_rules",
    ), id="model-walk-1957-1958"),
    pytest.param((
        "../test_grouped_linear_fisher.py::test_real_dsv4_wo_a_gets_a_priced_probe_row",
    ), id="grouped-fisher-1959"),
    pytest.param((
        "../test_dsv4_layer_streaming_rename.py::"
        "test_indexer_pooling_carries_the_coff_overlap_widening",
        "../test_dsv4_layer_streaming_rename.py::"
        "test_csa_compressor_returns_indices_not_a_gather",
    ), id="compressor-layout-1984-1985"),
])
def test_real_dsv4_contracts_after_native_import(tmp_path, targets):
    proc, output, outcomes, _out = native_dsv4_session(tmp_path, *targets)
    expected = {target.split("::")[-1] for target in targets} | {
        "test_vl_wrapper_config_reproduces_issue_12",
        "test_native_dsv4_import_still_refuses_the_vendored_override",
    }
    assert set(outcomes) == expected, output
    assert proc.returncode == 0, output
    assert all(outcome == "passed" for outcome, _ in outcomes.values()), output
