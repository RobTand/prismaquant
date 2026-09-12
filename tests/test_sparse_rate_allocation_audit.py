"""Numerical contracts for the scalar sparse-rate allocation replay."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.sparse_rate_allocation_audit import AllocationAuditError, audit_model


def _data():
    # One development expert stack and one final-holdout stack.  Each scalar has
    # 2,048 quantizable parameters, so its exact wire bpp is rate / 256.
    qnames = []
    layers = []
    for layer in (1, 3):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            qnames.append(f"model.layers.{layer}.mlp.experts.0.{projection}")
            layers.append(layer)
    rates = np.asarray([832, 960, 1088], dtype=np.int64)
    values = np.asarray([[4.0, 2.0, 1.0], [3.0, 1.5, .75], [8.0, 4.0, 2.0],
                         [9.0, 5.0, 2.5], [8.0, 4.0, 2.0], [16.0, 8.0, 4.0]])
    # bytes = bpp * 2048 / 8 = rate exactly for each scalar.
    return {"qnames": np.asarray(qnames), "families": np.asarray(["TESSERA_E4M3_K1"] * 6),
            "activation_contracts": np.asarray(["fp8"] * 6), "roles": np.asarray(["expert"] * 6),
            "structures": np.asarray(["gate_proj", "up_proj", "down_proj"] * 2),
            "layers": np.asarray(layers, dtype=np.int64),
            "rows": np.asarray([32] * 6), "cols": np.asarray([64] * 6),
            "counts": np.asarray([7, 7, 11, 7, 7, 11]), "rates": rates, "values": values,
            "wire_bytes": np.asarray([[832, 960, 1088]] * 6, dtype=np.int64)}


def _at_bpp(result, mode, bpp):
    return next(row for row in result["modes"][mode] if row["requested_bpp"] == bpp)


def _development_predictions(data):
    prediction = np.full_like(data["values"], np.nan)
    prediction[:3] = data["values"][:3]
    return prediction


def test_perfect_predictor_has_zero_regret_and_exact_measured_bytes():
    data = _data()
    result = audit_model(data, _development_predictions(data), stage="development")

    assert result["evaluated_groups"] == 1
    assert result["quantizable_params"] == 6144
    assert result["coverage"]["grouping"] == "whole_routed_layer__all_experts_gate_up_down"
    for mode in result["modes"].values():
        for row in mode:
            assert row["regret"] == 0.0
            assert row["oracle_bytes"] <= row["bytes_budget"]
            assert row["predicted_choice_bytes"] <= row["bytes_budget"]
    # The final group is excluded in development even though its truth exists
    # in the snapshot; the audit never puts it in the allocation population.
    assert all(group.get("group", [1])[0] != 3 for group in result["omissions"])


def test_prediction_reversal_has_positive_regret_at_fixed_exact_budget():
    data = _data()
    reversed_prediction = _development_predictions(data)
    # At 3.75 bpp the dev group can afford R960 but not R1088.  Make R832 look
    # best to the predictor; truth still ranks R960 best among feasible rates.
    reversed_prediction[:3] = np.asarray([[.1, 10.0, 20.0]] * 3)
    result = audit_model(data, reversed_prediction, stage="development")

    for mode in ("uniform", "routed_token_count_weighted"):
        row = _at_bpp(result, mode, 3.75)
        assert row["bytes_budget"] == 2880
        assert row["oracle_bytes"] == 2880
        assert row["predicted_choice_bytes"] == 2496
        assert row["regret"] > 0


def test_final_stage_cannot_include_development_predictions():
    data = _data()
    final_only = np.full_like(data["values"], np.nan)
    final_only[3:] = data["values"][3:]
    result = audit_model(data, final_only, stage="final")
    assert result["evaluated_groups"] == 1
    assert result["quantizable_params"] == 6144

    with pytest.raises(AllocationAuditError, match="outside its declared stage"):
        audit_model(data, data["values"], stage="final")
