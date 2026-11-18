"""Per-row selection pricing.

The synthetic case pins P5a's applied marker so a regression that multiplies
an already-corrected aggregate by the family factor fails loudly.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from prismaquant.activation_fair_pricing import APPLIED_MARKER_KEY
from prismaquant.per_row_pricing import price_row, reprice_assignment


_P5A = {
    "enabled": True,
    "reason": "calibrated",
    "families": {"nv": {"penalty": 10.0}},
}


def test_p5a_gain_is_applied_once_even_to_an_already_priced_row():
    """The applied marker prevents the historical 10x -> 100x inflation."""
    assignment = {"model.layers.0.experts.0.w2": "NVFP4"}
    stats = {
        "model.layers.0.experts.0.w2": {
            "h_trace": 1.0,
            "n_params": 256,
            "in_features": 16,
            "out_features": 16,
        },
    }
    raw_costs = {
        "model.layers.0.experts.0.w2": {
            "NVFP4": {
                "predicted_dloss": 2.0,
                "output_mse_measured": False,
            },
        },
    }
    raw = reprice_assignment(
        assignment, stats, raw_costs, activation_pricing=_P5A)
    assert raw["model.layers.0.experts.0.w2"] == pytest.approx(20.0)

    already_priced = {
        "model.layers.0.experts.0.w2": {
            "NVFP4": {
                "predicted_dloss": 20.0,
                "output_mse_measured": False,
                APPLIED_MARKER_KEY: True,
            },
        },
    }
    replayed = reprice_assignment(
        assignment, stats, already_priced, activation_pricing=_P5A)
    assert replayed["model.layers.0.experts.0.w2"] == pytest.approx(20.0)
    assert replayed["model.layers.0.experts.0.w2"] != pytest.approx(200.0)


def test_scalar_price_refuses_missing_and_failed_cost_rows_without_changes():
    for costs in ({}, {"NVFP4": {"error": "measurement failed"}}):
        expected_costs = deepcopy(costs)
        stats = {"n_params": 256, "h_trace": 1.0}
        with pytest.raises(KeyError) as exc:
            price_row("model.layers.0.weight", "NVFP4", stats, costs,
                      activation_pricing=None)
        assert exc.value.args == ("model.layers.0.weight: missing cost for NVFP4",)
        assert costs == expected_costs
        assert stats == {"n_params": 256, "h_trace": 1.0}


def test_recorded_pricing_fallbacks_keep_the_weight_only_row_price():
    from prismaquant.activation_fair_pricing import REASON_KILL_SWITCH

    cases = (("NVFP4A16", _P5A),
             ("NVFP4", {**_P5A, "reason": REASON_KILL_SWITCH}),
             ("NVFP4", {**_P5A, "families": {}}))
    stats = {"n_params": 256, "h_trace": 1.0}
    for format_name, pricing in cases:
        costs = {format_name: {"predicted_dloss": 2.0, "output_mse_measured": False}}
        saved = deepcopy((stats, costs, pricing))
        assert price_row("model.proj", format_name, stats, costs,
                         activation_pricing=pricing) == 2.0
        assert (stats, costs, pricing) == saved


def test_missing_mxfp4_source_cost_keeps_its_exact_zero_price_and_inputs():
    stats = {"n_params": 256, "h_trace": 3.0, "in_features": 16, "out_features": 16}
    gains = {"MXFP4_SOURCE": 11.0}
    for costs in ({}, {"NVFP4": {"predicted_dloss": 17.0, "output_mse_measured": False}}):
        pricing = deepcopy(_P5A)
        before = deepcopy((stats, costs, gains, pricing))
        assert price_row("model.proj", "MXFP4_SOURCE", stats, costs,
                         activation_pricing=pricing, calibrated_gains=gains) == 0.0
        assert (stats, costs, gains, pricing) == before
