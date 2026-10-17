from __future__ import annotations

import pytest

from prismaquant.joint_aura_rows import joint_identity_sha256
from prismaquant.native_operator_panel import consume_native_receipt
from test_native_operator_panel import joined, receipt_fixture, write


@pytest.mark.parametrize("older_phases", [(), ("prefill",), ("decode",), ("prefill", "decode")],
                         ids=["all_native", "prefill_older", "decode_older", "both_older"])
def test_schedule_summary_retains_any_observed_older_phase(joined, tmp_path, older_phases):
    panel, receipt, _trace = receipt_fixture(joined)
    expected_arches = {}
    for phase in ("prefill", "decode"):
        route = receipt["phases"][phase]["route"]
        route["platform"] = "sm_121"
        route["kernel_schedule"] = (
            "cutlass_80_tensorop_s16816gemm_bf16_256x128" if phase in older_phases else
            "cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1"
        )
        panel["phases"][phase]["expected_route"] = dict(route)
        expected_arches[phase] = "older_arch" if phase in older_phases else "native"
    receipt["panel"] = panel
    receipt["panel_sha256"] = joint_identity_sha256(panel)
    path = tmp_path / "receipt.json"
    digest = write(path, receipt)

    observation = consume_native_receipt(path, expected_sha256=digest, expected_panel=panel)

    assert observation["status"] == "operator_evidence"
    assert observation["runtime_table_admissible"] is False
    assert {phase: row["kernel_arch"] for phase, row in observation["phases"].items()} == expected_arches
    assert ("native_kernel_schedule" in observation["unknown"]) is bool(older_phases)
