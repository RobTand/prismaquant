"""Exercise the fake-trace import order from PQ #2279 with the real controls."""
from pathlib import Path

import pytest

from test_layer_streaming_mxfp4_isolation import _NATIVE_GUARD, _WRAPPER_REFUSAL
from test_own_process_isolation import _session


@pytest.mark.parametrize("native_first", [True, False], ids=["native-first", "native-last"])
def test_dsv4_fake_trace_import_orders(tmp_path, native_first):
    """The fake-trace ratchet survives the native DSv4 import predecessor."""
    target = "../test_model_walk.py::test_dsv4_fake_trace_block_is_still_real"
    # The source config retains the full checkpoint dimensions.
    # Source: deepseek-ai/DeepSeek-V4-Flash at
    # 60d8d70770c6776ff598c94bb586a859a38244f1, config.json.
    source = Path(__file__).parent / "fixtures" / "dsv4_source"
    env = {"DSV4_FLASH_SOURCE": str(source.resolve())}
    files = ((_WRAPPER_REFUSAL, target) if native_first
             else (target, _WRAPPER_REFUSAL))
    proc, output, outcomes, _ = _session(
        tmp_path, *files, _NATIVE_GUARD, environ=env)
    print(output)
    expected = {target.split("::")[-1],
                "test_vl_wrapper_config_reproduces_issue_12",
                "test_native_dsv4_import_still_refuses_the_vendored_override"}
    assert set(outcomes) == expected, output
    assert proc.returncode == 0, output
    assert all(state == "passed" for state, _ in outcomes.values()), output
