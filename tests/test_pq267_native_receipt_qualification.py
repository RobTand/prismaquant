"""The real-receipt qualification driver must fail closed, not summarize.

The consumer's own gates live in ``prismaquant.native_operator_panel`` and have
their own boundary suite. What is asserted here is the CLI's own share: it
reaches those gates with the caller's bytes, writes no report when one refuses,
and reports operator evidence as operator evidence rather than as a runtime row.
"""
import hashlib
import json

import pytest

from experiments import pq267_native_receipt_qualification as driver


def _paths(tmp_path, receipt, panel):
    receipt_path, panel_path = tmp_path / "receipt.json", tmp_path / "panel.json"
    receipt_path.write_text(json.dumps(receipt))
    panel_path.write_text(json.dumps(panel))
    return receipt_path, panel_path, tmp_path / "report.json"


def _argv(receipt_path, panel_path, out, trace=None):
    argv = ["--receipt", str(receipt_path), "--panel", str(panel_path), "--out", str(out)]
    return argv if trace is None else argv + ["--memory-trace", str(trace)]


@pytest.mark.parametrize("receipt", [
    {"schema": "tessera.native_dense_operator_receipt.v1", "status": "numerical_refused"},
    {"schema": "tessera.native_dense_operator_receipt.v1", "status": "resources_observed"},
    {"schema": "tessera.native_moe_operator_receipt.v1", "status": "timing_admissible"},
])
def test_unadmitted_receipt_never_becomes_operator_evidence(tmp_path, receipt):
    receipt_path, panel_path, out = _paths(tmp_path, receipt, {})
    with pytest.raises(ValueError, match="no admitted numerical/timing observation"):
        driver.main(_argv(receipt_path, panel_path, out))
    assert not out.exists()


def test_panel_the_producer_did_not_freeze_is_refused(tmp_path):
    receipt = {"schema": "tessera.native_dense_operator_receipt.v1", "status": "timing_admissible",
               "panel": {"unit": "model.layers.0.mlp.down_proj"}}
    receipt_path, panel_path, out = _paths(tmp_path, receipt, {"unit": "model.layers.1.mlp.down_proj"})
    with pytest.raises(ValueError, match="receipt panel"):
        driver.main(_argv(receipt_path, panel_path, out))
    assert not out.exists()


def test_report_binds_the_exact_bytes_and_keeps_the_row_inadmissible(tmp_path, monkeypatch, capsys):
    observation = {"schema": "prismaquant.native_dense_observation.v1", "status": "operator_evidence",
                   "unit": "model.layers.0.feed_forward.w1", "format": "TESSERA_BF16_K1_R1792",
                   "phases": {"prefill": {"median_ms": 0.39, "peak_scratch_bytes": 36700160,
                                          "input_bytes": 2097152, "output_bytes": 7340032,
                                          "measurement": {"samples_ms": [0.39]}},
                              "decode": {"median_ms": 0.178, "peak_scratch_bytes": 71680,
                                         "input_bytes": 4096, "output_bytes": 14336,
                                         "measurement": {"samples_ms": [0.178]}}},
                   "resident_bytes": 29388800, "serialized_unit_bytes": 12892880,
                   "full_model_resources": None, "runtime_table_admissible": False,
                   "unknown": ["fixed_and_full_model_resources"]}
    seen = {}

    def _consume(path, *, expected_sha256, expected_panel, memory_trace_path=None):
        seen.update(path=str(path), sha=expected_sha256, trace=str(memory_trace_path))
        return observation

    monkeypatch.setattr("prismaquant.native_operator_panel.consume_native_receipt", _consume)
    panel = {"phases": {"prefill": {"m": 512}, "decode": {"m": 1}}, "runtime": {"schema": "r"},
             "source_sha256": "a" * 64, "calibration_sha256": "b" * 64,
             "probe_identity_sha256": "c" * 64}
    receipt_path, panel_path, out = _paths(tmp_path, {"any": "bytes"}, panel)
    trace = tmp_path / "trace.json"
    trace.write_text("{}")
    assert driver.main(_argv(receipt_path, panel_path, out, trace)) == 0
    report = json.loads(out.read_text())
    assert report["schema"] == "prismaquant.native_receipt_qualification.v1"
    # The consumer is handed the digest of the bytes on disk, not a declaration.
    assert seen["sha"] == report["receipt_sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert seen["trace"] == str(trace) and report["memory_trace_path"] == str(trace)
    # Operator evidence exits zero and still refuses to be a runtime row.
    assert report["runtime_table_admissible"] is False
    assert report["unknown"] == ["fixed_and_full_model_resources"]
    printed = json.loads(capsys.readouterr().out)
    assert printed["runtime_table_admissible"] is False
    # The row obligations are read off the observation and panel, not asserted.
    owed = report["runtime_row_obligations"]
    assert owed["satisfied"]["both_phases_measured"] is True
    assert owed["satisfied"]["peak_scratch_bytes"] == {"prefill": 36700160, "decode": 71680}
    assert owed["satisfied"]["context_must_declare"]["prompt_tokens"] == 512
    assert owed["satisfied"]["joint_cost_row"] is None
    assert any("joint_aura" in reason for reason in owed["owed"])
    assert any("full_engine_resource_report" in reason for reason in owed["owed"])
