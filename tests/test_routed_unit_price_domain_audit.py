"""Price-domain intake, with synthetic eligibility and no served claim."""
from __future__ import annotations

import json
import sys

import pytest

from prismaquant import tessera_expert_projection as projection
from test_tessera_per_unit_rates import (
    BASE_FMT,
    UP_FMT,
    _alloc_argv,
    _alloc_cost_payload,
    _alloc_units,
    _synthetic_two_rung_contract,
    _write_cli_source,
)


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


def _cli_case(tmp_path, monkeypatch, *, damaged_rung=None):
    from prismaquant import tessera_runtime_contract as contract

    _synthetic_two_rung_contract(monkeypatch)
    monkeypatch.setattr(contract, "packaged_routed_unit_capability",
                        lambda: ("d" * 64, dict(contract.ROUTED_UNIT_ASSIGNMENT_BLOCK)))
    payload = _alloc_cost_payload(tmp_path, formats=(BASE_FMT, UP_FMT))
    for name in _alloc_units():
        payload["costs"][name][BASE_FMT].update(
            predicted_dloss=1.0, wire_bytes=4096, output_mse=4e-4)
        payload["costs"][name][UP_FMT].update(
            predicted_dloss=0.5, wire_bytes=4608, output_mse=8e-4)
    damaged_unit = _alloc_units()[0]
    if damaged_rung is not None:
        payload["costs"][damaged_unit][damaged_rung]["wire_bytes"] = -1
    argv = _alloc_argv(tmp_path, payload)
    _write_cli_source(tmp_path)
    index = argv.index("--target-bits")
    del argv[index:index + 2]
    argv += ["--target-disk-gb", "16", "--artifact-overhead-reserve-bytes", "1048576",
             "--routed-unit-rates", "--formats", BASE_FMT + "," + UP_FMT]
    monkeypatch.setattr(sys, "argv", argv)
    return damaged_unit


@pytest.mark.parametrize("rung", [BASE_FMT, UP_FMT])
def test_actual_allocator_refuses_negative_wire_price_before_emission(
        tmp_path, monkeypatch, rung):
    from prismaquant import allocator

    unit = _cli_case(tmp_path, monkeypatch, damaged_rung=rung)
    with pytest.raises(SystemExit, match=unit):
        allocator.main()
    assert not (tmp_path / "layer.json").exists()


def test_actual_allocator_reaches_legal_rates_with_real_body_solver(tmp_path, monkeypatch):
    from prismaquant import allocator
    from prismaquant.layer_config import load_assignment

    _cli_case(tmp_path, monkeypatch)
    allocator.main()
    placed = load_assignment(tmp_path / "layer.json")
    assert all(placed[name] == UP_FMT for name in _alloc_units())
    record = json.loads((tmp_path / "layer.json").read_text())["__prismaquant__"][
        "tessera_routed_unit_rates"]
    assert record["upgrades"] == len(_alloc_units())
    assert record["spent_wire_delta_bytes"] == 512 * len(_alloc_units())


def test_actual_allocator_cost_schema_already_refuses_nonfinite_loss():
    from prismaquant.schemas import SchemaValidationError, validate_cost_payload

    payload = {"costs": {"unit": {BASE_FMT: {
        "output_mse": 1.0, "predicted_dloss": float("nan"), "wire_bytes": 1}}}}
    with pytest.raises(SchemaValidationError, match="finite"):
        validate_cost_payload(payload)
