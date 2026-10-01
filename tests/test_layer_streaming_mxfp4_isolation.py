"""MXFP4 CPU fixtures must not inherit a native DSv4 import (PQ #1977-1981)."""
from test_own_process_isolation import _session

_WRAPPER_REFUSAL = (
    "../test_streaming_text_only_wrapper_config.py::"
    "test_vl_wrapper_config_reproduces_issue_12")
_NATIVE_GUARD = "sample_dsv4_native_guard.py"


def _assert_mixed_session_passes(tmp_path, *targets):
    """Use the real lazy-mapping import trigger, not a fake module or profile."""
    proc, output, outcomes, _out = _session(
        tmp_path, _WRAPPER_REFUSAL,
        *([f"../test_layer_streaming_mxfp4.py::{name}" for name in targets]
          if targets else ["../test_layer_streaming_mxfp4.py"]),
        _NATIVE_GUARD)
    controls = {
        "test_vl_wrapper_config_reproduces_issue_12",
        "test_native_dsv4_import_still_refuses_the_vendored_override",
    }
    if targets:
        assert set(outcomes) == controls | set(targets), output
    else:
        # All 20 original cases, including both resident-byte dtype variants.
        assert controls <= set(outcomes) and len(outcomes) == 22, output
    assert proc.returncode == 0, output
    assert all(outcome == "passed" for outcome, _ in outcomes.values()), output


def test_non_expert_fp8_grid_after_native_dsv4_import(tmp_path):
    _assert_mixed_session_passes(
        tmp_path, "test_non_expert_tensor_still_flows_through_check_fp8_scale_grid")


def test_full_mxfp4_file_after_native_dsv4_import(tmp_path):
    _assert_mixed_session_passes(tmp_path)


def test_shared_expert_scope_after_native_dsv4_import(tmp_path):
    _assert_mixed_session_passes(
        tmp_path, "test_shared_expert_is_not_covered_by_the_declaration")


def test_undeclared_checkpoint_after_native_dsv4_import(tmp_path):
    _assert_mixed_session_passes(
        tmp_path, "test_undeclared_checkpoint_has_no_mxfp4_names")


def test_undeclared_int8_group16_after_native_dsv4_import(tmp_path):
    _assert_mixed_session_passes(
        tmp_path, "test_undeclared_int8_group16_is_not_decoded_as_nibbles")
