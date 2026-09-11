"""The hidden complete curve must never steer acquisition or stopping."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.sparse_rate_adaptive import (
    CAPS, CURRENCY, CURVE_SCHEMA, MEASUREMENT_PLAN_SCHEMA, error_metrics, evaluate_curve, fixed_schedule,
    validate_curve,
)


@pytest.fixture(autouse=True)
def isolated_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def bind_receipts(source):
    for receipt in source["receipts"]:
        record = {key: receipt[key] for key in ("rate", "value", "kind")}
        record.update(curve_id=source["curve_id"], measurement_plan_sha256=source["measurement_plan"]["sha256"])
        path = Path(f"rate-{receipt['rate']}.json").resolve()
        path.write_text(json.dumps(record))
        receipt.update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def curve(values=None):
    rates = list(range(832, 1089))
    values = values or [100.0 - .25 * (rate - 832) for rate in rates]
    result = {"schema": CURVE_SCHEMA, "currency": CURRENCY, "curve_id": "test-curve",
        "measurement_kind": "measured", "qname": "test.linear", "family": "TESSERA_E4M3_K1",
        "activation_contract": "fp8_per_token_dynamic",
        "source_identity": {"model": "test", "weight_identity": "test", "producer": "test"},
        "calibration_identity": {"capture": "test"}, "recipe_identity": {"recipe": "test"},
        "rates": rates, "legal_rates": rates.copy(), "values": values,
        "receipts": [{"rate": rate, "value": value, "kind": "measured", "sha256": "0" * 64}
                     for rate, value in zip(rates, values)]}
    plan = {key: result[key] for key in ("curve_id", "qname", "family", "activation_contract",
        "source_identity", "calibration_identity", "recipe_identity", "legal_rates")}
    plan["schema"] = MEASUREMENT_PLAN_SCHEMA
    path = Path("measurement-plan.json").resolve()
    path.write_text(json.dumps(plan))
    result["measurement_plan"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    bind_receipts(result)
    return result


def run(source, **kwargs):
    return evaluate_curve(source, mode="value", tolerance=.005, checks_per_interval=1, **kwargs)


def test_unrevealed_truth_changes_audit_but_never_the_trajectory():
    source = curve()
    first = run(source)
    altered = deepcopy(source)
    index = altered["rates"].index(900)
    altered["values"][index] *= .8
    altered["receipts"][index]["value"] = altered["values"][index]
    bind_receipts(altered)
    second = run(altered)
    assert first["probes"] == second["probes"]
    assert first["actual_measurements"] == second["actual_measurements"] == 3
    assert first["snapshots"][-1]["never_revealed_metrics"]["screen_pass"]
    assert not second["snapshots"][-1]["never_revealed_metrics"]["screen_pass"]


def test_audit_excludes_every_revealed_point_and_charges_endpoints():
    result = run(curve())
    for snapshot in result["snapshots"]:
        count = snapshot["measurement_count"]
        assert snapshot["never_revealed_metrics"]["count"] == 257 - count
        assert len(snapshot["fixed_measured_rates"]) == count
    assert result["probes"][0]["rate"] == 960
    assert result["snapshots"][-1]["measured_rates"] == [832, 960, 1088]


def test_rejected_nonmonotone_reveal_is_charged_and_never_clamped():
    source = curve()
    index = source["rates"].index(960)
    source["values"][index] = 150.0
    source["receipts"][index]["value"] = 150.0
    bind_receipts(source)
    result = run(source)
    assert result["status"] == "invalid_measurement"
    assert result["actual_measurements"] == 3
    assert result["probes"][-1]["measured_value"] == 150.0
    assert "interpolation refused" in result["required_fallback"]


def test_fixed_baseline_order_is_geometry_only_and_complete():
    order = fixed_schedule(tuple(range(9)))
    assert order[:5] == (0, 8, 4, 2, 6)
    assert sorted(order) == list(range(9))


@pytest.mark.parametrize("change", ["partial", "interpolated", "receipt", "currency"])
def test_refuse_unbound_or_incomplete_measurement_inputs(change):
    source = curve()
    if change == "partial":
        source["rates"].pop()
    elif change == "interpolated":
        source["measurement_kind"] = "interpolated"
    elif change == "receipt":
        source["receipts"][1]["value"] += 1
    else:
        source["currency"] = "joint_aura"
    with pytest.raises(ValueError):
        validate_curve(source)


def test_empty_audit_does_not_claim_a_pass_or_measurement_coverage():
    assert error_metrics([], [], .01) == {"count": 0, "empty_audit": True, "screen_pass": None}


def test_cannot_drop_a_hard_rate_from_the_sealed_measurement_roster():
    source = curve()
    for key in ("rates", "values", "legal_rates", "receipts"):
        source[key].pop(10)
    with pytest.raises(ValueError, match="pre-measurement plan"):
        validate_curve(source)


def test_tampered_receipt_bytes_are_refused():
    source = curve()
    Path(source["receipts"][0]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="receipt reference"):
        validate_curve(source)


def test_caps_are_increasing_prefixes_of_one_trace():
    source = curve([10000.0 / (1 + i / 10) for i in range(257)])
    result = evaluate_curve(source, mode="value", tolerance=.001, checks_per_interval=2)
    previous = set()
    for snapshot in result["snapshots"]:
        current = set(snapshot["measured_rates"])
        assert previous <= current
        previous = current
        assert snapshot["measurement_count"] in CAPS or snapshot is result["snapshots"][-1]
