"""Price-domain intake, with synthetic eligibility and no served claim."""
from __future__ import annotations


import pytest

from prismaquant import tessera_expert_projection as projection
from test_tessera_per_unit_rates import BASE_FMT, UP_FMT


@pytest.mark.parametrize("rung", [BASE_FMT, UP_FMT])
@pytest.mark.parametrize("bad_loss", [float("nan"), float("inf"), float("-inf")])
def test_public_selector_refuses_nonfinite_exact_loss(rung, bad_loss):
    costs = {"unit": {
        BASE_FMT: {"predicted_dloss": 1.0, "wire_bytes": 10},
        UP_FMT: {"predicted_dloss": 0.5, "wire_bytes": 20},
    }}
    costs["unit"][rung]["predicted_dloss"] = bad_loss
    with pytest.raises(projection.ExpertProjectionError, match="unit"):
        projection.select_priced_unit_upgrades(
            costs, {"unit": BASE_FMT}, byte_budget=20)


@pytest.mark.parametrize("rung", [BASE_FMT, UP_FMT])
def test_public_selector_refuses_negative_exact_wire_bytes(rung):
    costs = {"unit": {
        BASE_FMT: {"predicted_dloss": 1.0, "wire_bytes": 10},
        UP_FMT: {"predicted_dloss": 0.5, "wire_bytes": 20},
    }}
    costs["unit"][rung]["wire_bytes"] = -1
    with pytest.raises(projection.ExpertProjectionError, match="unit"):
        projection.select_priced_unit_upgrades(
            costs, {"unit": BASE_FMT}, byte_budget=30)


@pytest.mark.parametrize("field,value", [
    ("predicted_dloss", True),
    ("wire_bytes", True),
    ("predicted_dloss", 10 ** 400),
])
def test_public_selector_keeps_typed_refusal_for_unusable_numeric_values(field, value):
    costs = {"unit": {
        BASE_FMT: {"predicted_dloss": 1.0, "wire_bytes": 10},
        UP_FMT: {"predicted_dloss": 0.5, "wire_bytes": 20},
    }}
    costs["unit"][UP_FMT][field] = value
    with pytest.raises(projection.ExpertProjectionError, match="unit"):
        projection.select_priced_unit_upgrades(
            costs, {"unit": BASE_FMT}, byte_budget=10)


@pytest.mark.parametrize("base_loss,up_loss,base_wire", [
    (-1.0, -2.0, 10),
    (1.0, 0.5, 0),
])
def test_public_selector_retains_signed_loss_and_zero_byte_domains(
        base_loss, up_loss, base_wire):
    costs = {"unit": {
        BASE_FMT: {"predicted_dloss": base_loss, "wire_bytes": base_wire},
        UP_FMT: {"predicted_dloss": up_loss, "wire_bytes": base_wire + 10},
    }}
    picks, record = projection.select_priced_unit_upgrades(
        costs, {"unit": BASE_FMT}, byte_budget=10)
    assert picks == {"unit": UP_FMT}
    assert record["spent_wire_delta_bytes"] == 10


def test_actual_allocator_cost_schema_already_refuses_nonfinite_loss():
    from prismaquant.schemas import SchemaValidationError, validate_cost_payload

    payload = {"costs": {"unit": {BASE_FMT: {
        "output_mse": 1.0, "predicted_dloss": float("nan"), "wire_bytes": 1}}}}
    with pytest.raises(SchemaValidationError, match="finite"):
        validate_cost_payload(payload)
