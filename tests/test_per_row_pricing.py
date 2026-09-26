"""Per-row selection pricing and reference-artifact reconciliation.

The synthetic case pins P5a's applied marker so a regression that multiplies
an already-corrected aggregate by the family factor fails loudly.  The
read-only integration gate then reconciles the two named DSv4 selections that
authorized the tier-2 counterfactual work.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from prismaquant.activation_fair_pricing import APPLIED_MARKER_KEY
from prismaquant.per_row_pricing import reconcile_selection, reprice_assignment


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


_REFERENCE_ROOT = Path(
    "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2"
)
_REFERENCE_SELECTIONS = [
    _REFERENCE_ROOT / "artifacts-mxfp4-sm121",
]
_RETIRED_ACTIVATION_BLIND_SELECTION = (
    _REFERENCE_ROOT / "artifacts-mxfp4/oldmenu-grid/b-92"
)


@pytest.mark.integration
@pytest.mark.parametrize("selection_dir", _REFERENCE_SELECTIONS)
def test_reference_selections_reconcile_within_one_part_per_thousand(
    selection_dir,
):
    """Hard gate: tier-2 analysis cannot proceed on a mis-scaled pricer."""
    if not selection_dir.exists():
        pytest.skip(f"read-only reference artifact is not mounted: {selection_dir}")
    reconstructed, recorded, ratio = reconcile_selection(selection_dir)
    assert reconstructed == pytest.approx(recorded, rel=1.0e-3)
    assert ratio == pytest.approx(1.0, abs=1.0e-3)


@pytest.mark.integration
def test_block_fp8_reference_reconciles_under_w8a16_source_identity():
    """The raw-source W8A16 route restores exact zero-cost reconciliation."""
    selection_dir = _RETIRED_ACTIVATION_BLIND_SELECTION
    if not selection_dir.exists():
        pytest.skip(f"read-only reference artifact is not mounted: {selection_dir}")
    reconstructed, recorded, ratio = reconcile_selection(selection_dir)
    assert reconstructed == pytest.approx(recorded, rel=1.0e-3)
    assert ratio == pytest.approx(1.0, abs=1.0e-3)
