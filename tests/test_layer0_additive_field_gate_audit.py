"""The standalone identity CLI must require its declared additive field.

Tiny synthetic cost envelopes exercise the real public main and verdict file.
They establish no encoder parity, GPU, artifact-quality or serving result.
"""
from __future__ import annotations

import pickle
import sys

import pytest

from tools import cost_layer0_identity_gate as gate


QNAME = "model.layers.0.self_attn.q_proj"
FORMAT = "TESSERA_E2M1_K2_R128"


def _invoke(tmp_path, monkeypatch, shipped_payload, new_payload):
    shipped = tmp_path / "shipped.pkl"
    candidate = tmp_path / "candidate.pkl"
    verdict = tmp_path / "verdict.txt"
    shipped.write_bytes(pickle.dumps(shipped_payload))
    candidate.write_bytes(pickle.dumps(new_payload))
    monkeypatch.setattr(sys, "argv", [
        str(gate.__file__), "--shipped", str(shipped), "--new", str(candidate),
        "--min-rows", "1", "--verdict", str(verdict),
    ])
    return gate.main(), verdict.read_text()


def _run(tmp_path, monkeypatch, *, activation_rows):
    base_row = {"weight_mse": 0.125, "predicted_dloss": 0.25}
    new_row = dict(base_row)
    if activation_rows is not None:
        new_row["n_activation_rows"] = activation_rows
    return _invoke(tmp_path, monkeypatch, {"costs": {QNAME: {FORMAT: base_row}}},
                   {"costs": {QNAME: {FORMAT: new_row}}})


def test_standalone_gate_refuses_missing_required_activation_row_field(tmp_path, monkeypatch):
    code, verdict = _run(tmp_path, monkeypatch, activation_rows=None)

    assert code != 0, "standalone gate accepted an omitted required additive field"
    assert verdict != "PASS"


def test_standalone_gate_accepts_matching_base_and_required_64(tmp_path, monkeypatch):
    code, verdict = _run(tmp_path, monkeypatch, activation_rows=64)

    assert (code, verdict) == (0, "PASS")


def test_standalone_gate_keeps_existing_wrong_additive_value_refusal(tmp_path, monkeypatch):
    code, verdict = _run(tmp_path, monkeypatch, activation_rows=63)

    assert code != 0
    assert verdict == "UNEXPECTED_FIELDS"


def test_standalone_gate_does_not_pass_matching_scalar_format_entries(tmp_path, monkeypatch):
    shipped = tmp_path / "shipped.pkl"
    candidate = tmp_path / "candidate.pkl"
    verdict = tmp_path / "verdict.txt"
    malformed = {"costs": {QNAME: {FORMAT: 0.25}}}
    shipped.write_bytes(pickle.dumps(malformed))
    candidate.write_bytes(pickle.dumps(malformed))
    monkeypatch.setattr(sys, "argv", [
        str(gate.__file__), "--shipped", str(shipped), "--new", str(candidate),
        "--min-rows", "1", "--verdict", str(verdict),
    ])
    try:
        code = gate.main()
    except Exception as refused:
        outcome = {"kind": "fail_closed_exception", "type": type(refused).__name__}
    else:
        outcome = {"kind": "returned", "code": code}
        assert code is not None and code != 0, outcome

    assert not verdict.exists() or verdict.read_text() != "PASS", outcome


@pytest.mark.parametrize("arm", ["shipped", "new"])
@pytest.mark.parametrize("malformed", [
    [],
    {"costs": []},
    {"costs": {QNAME: 0.25}},
    {"costs": {QNAME: {}}},
    {"costs": {QNAME: {FORMAT: None}}},
    {"costs": {QNAME: {FORMAT: {}}}},
    {"costs": {QNAME: {FORMAT: 0.25}}},
])
def test_malformed_intake_writes_existing_unusable_shard_verdict(
    tmp_path, monkeypatch, arm, malformed,
):
    valid = {"costs": {QNAME: {FORMAT: {"weight_mse": 0.125, "n_activation_rows": 64}}}}
    shipped, new = (malformed, valid) if arm == "shipped" else (valid, malformed)
    code, verdict = _invoke(tmp_path, monkeypatch, shipped, new)
    assert code == 1
    assert verdict == ("NO_SHIPPED_REFERENCE" if arm == "shipped" else "NO_OUTPUT_NEW")


@pytest.mark.parametrize("rows,expected", [(64, "PASS"), (63, "UNEXPECTED_FIELDS")])
def test_required_value_is_checked_even_when_already_in_shipped_arm(
    tmp_path, monkeypatch, rows, expected,
):
    payload = {"costs": {QNAME: {FORMAT: {"weight_mse": 0.125, "n_activation_rows": rows}}}}
    code, verdict = _invoke(tmp_path, monkeypatch, payload, payload)
    assert (code, verdict) == (0 if expected == "PASS" else 1, expected)


def test_required_field_must_exist_on_every_format_entry(tmp_path, monkeypatch):
    base = {"weight_mse": 0.125}
    shipped = {"costs": {QNAME: {FORMAT: base, "BF16": base}}}
    new = {"costs": {QNAME: {FORMAT: dict(base, n_activation_rows=64), "BF16": base}}}
    assert _invoke(tmp_path, monkeypatch, shipped, new) == (1, "UNEXPECTED_FIELDS")


def test_unknown_additions_still_refuse(tmp_path, monkeypatch):
    base = {"weight_mse": 0.125}
    shipped = {"costs": {QNAME: {FORMAT: base}}}
    new = {"costs": {QNAME: {FORMAT: dict(base, n_activation_rows=64, unrecognized=1)}}}
    assert _invoke(tmp_path, monkeypatch, shipped, new) == (1, "UNEXPECTED_FIELDS")


def test_existing_invariant_difference_still_refuses(tmp_path, monkeypatch):
    shipped = {"costs": {QNAME: {FORMAT: {"weight_mse": 0.125}}}}
    new = {"costs": {QNAME: {FORMAT: {"weight_mse": 0.5, "n_activation_rows": 64}}}}
    assert _invoke(tmp_path, monkeypatch, shipped, new) == (1, "MISMATCH")
