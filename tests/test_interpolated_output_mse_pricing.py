"""Band-interpolated rungs are priced from their OWN output_mse, not the
per-family activation constant.

The defect this file pins (found 2026-08-07 validating the NVFP4 menu ahead of
the DSv4-Flash allocation grid, ``dq-runs/nvfp4-cbl/MENU_PRICING_BUG.md``):

``activation_fair_pricing``'s design argues a per-family multiplicative
constant is safe because it "cannot reorder rungs *inside* a family, so the
holdout-gated ladder shape survives untouched". That is true only while every
rung of the family takes the SAME pricing branch. It did not. A CB ladder's
interpolated rungs are stamped ``output_mse_measured=False`` with
``cost_source: band_interpolated``, so ``_has_measured_output_mse`` rejected
them and they fell to ``weight_mse x family_penalty`` while their measured
neighbours were priced from ``output_mse`` directly — two bases inside one
ladder.

The constant is a family-wide geometric mean, and ``nvfp4_cb``'s true
``output_mse/weight_mse`` ratio is not family-wide at all: 187-320 on
gate/up_proj against 9.4-22 on down_proj, a ~20x spread. Applying 112.5 to
both over-priced the down_proj interpolated rungs ~12x (a HIGHER rung cost
MORE than the one below it on 85% of experts, so it could never be selected)
and under-priced the gate/up_proj ones ~1.6x. The menu was wrong in both
directions at once, with the direction set by the projection — the real
mechanism behind "the 4-bit band has only 4 usable rungs, not 7".

What this file pins:

1. Rung monotonicity inside the family, through the REAL pricing function, on
   two projections with ratios ~10 and ~250 — and the same property FAILING
   under the old rule, so the test demonstrably catches the bug.
2. The calibration sample is untouched: same rows, same count, same digest,
   same fitted penalty whether or not interpolated rows are present. Their
   ``output_mse`` is derived from this family's own measured anchors, so
   admitting it would be circular.
3. The new branch has its own label, ``BRANCH_INTERPOLATED_OUTPUT``, so a
   shipped artifact can still say which of its selected prices were
   predictions.
4. Nothing else moves: the ``output_mse == 0.0`` placeholders keep the
   weight-only branch, the provenance and noise-band guards keep full
   strength, and the row's own claims about itself are unchanged.
"""
from __future__ import annotations

import pytest

from prismaquant import format_registry as fr
from prismaquant.activation_fair_pricing import (
    BRANCH_CALIBRATED,
    BRANCH_INTERPOLATED_OUTPUT,
    BRANCH_MEASURED,
)
from prismaquant.allocator_candidates import (
    _super_item_ucb_hedge,
    build_candidates,
    calibrate_activation_fair_pricing,
    collect_activation_calibration_rows,
    cost_entry_activation_pricing_branch,
    cost_entry_is_band_interpolated,
    cost_entry_predicted_dloss,
    cost_entry_source,
    cost_entry_uses_measured_output_mse,
    drop_interpolated_candidates_dominated_by_measured,
)
from prismaquant.allocator_solver import Candidate

# One family, four rungs: the two ends measured, the two middle ones
# band-interpolated. Until 2026-09-25 these were four consecutive rungs of the
# retired NVFP4 codebook ladder (K12..K15; archived, #1304). A stale table can
# still carry the ``band_interpolated`` stamp, and the Tessera campaign's
# fitted rows take the same branch, so the rule is pinned here on the four
# activation-quantizing rungs of the live ``mx`` family. The pricing never
# reads a rung's bytes, so their order is only a label order.
_RUNGS = ["MXFP4", "MXFP8_E5M2", "MXFP8_UE8M0_G32", "MXFP8_E4M3"]
_MEASURED_RUNGS = ("MXFP4", "MXFP8_E4M3")
_INTERPOLATED_RUNGS = ("MXFP8_E5M2", "MXFP8_UE8M0_G32")
_FAMILY = "mx"

# h_trace = 2 makes predicted_dloss = 0.5*h_trace*mse exactly the mse, so every
# expected number below is readable as an MSE.
_H_TRACE = 2.0

# The two projections of the bug report, at their measured extremes. The whole
# defect lives in this spread: one family constant cannot serve both.
_RATIO_DOWN_PROJ = 10.0
_RATIO_GATE_PROJ = 250.0

# Geometric mean over the rows that ENTER calibration (two measured rungs on
# each projection) = sqrt(10 * 250) = 50: 5x too big for down_proj, 5x too
# small for gate_proj.
_EXPECTED_PENALTY = 50.0

_DOWN_PROJ = "model.layers.0.mlp.experts.0.down_proj"
_GATE_PROJ = "model.layers.0.mlp.experts.0.gate_proj"

_W12 = 1.0e-4
# Adjacent-rung weight_mse step. Small next to the 5x branch error, which is
# exactly the point: a mis-scaled rung jumps clean over its neighbour.
_STEP = 1.2


def _weight_mse(rung: str) -> float:
    return _W12 / (_STEP ** _RUNGS.index(rung))


def _stats_entry() -> dict:
    # in_features % 32 == 0 and both dims >= 128 keep every MX rung legal,
    # so nothing drops out of build_candidates for shape reasons.
    return {
        "h_trace": _H_TRACE,
        "n_params": 1024 * 512,
        "in_features": 1024,
        "out_features": 512,
    }


def _ladder_rows(ratio: float) -> dict:
    """One tensor's four-rung ladder at a given output/weight ratio.

    Both metrics are geometric in K, so the interpolated rungs carry exactly
    what a log-space ladder fit through the measured K12/K15 anchors produces
    — which is what ``_ladder_metric_fit`` writes, and what the banked shards
    contain (``weight_mse``, ``output_mse``, ``rel_output_mse``,
    ``output_mse_measured: false``, ``cost_source: band_interpolated``; no
    ``predicted_dloss``).
    """
    rows: dict[str, dict] = {}
    for rung in _RUNGS:
        weight_mse = _weight_mse(rung)
        row = {"weight_mse": weight_mse, "output_mse": ratio * weight_mse}
        if rung in _INTERPOLATED_RUNGS:
            row["rel_output_mse"] = 0.1 * ratio * weight_mse
            row["output_mse_measured"] = False
            row["cost_source"] = "band_interpolated"
        else:
            row["n_activation_rows"] = 64
        rows[rung] = row
    return rows


def _tables() -> tuple[dict, dict]:
    stats = {_DOWN_PROJ: _stats_entry(), _GATE_PROJ: _stats_entry()}
    costs = {
        _DOWN_PROJ: _ladder_rows(_RATIO_DOWN_PROJ),
        _GATE_PROJ: _ladder_rows(_RATIO_GATE_PROJ),
    }
    return stats, costs


def _specs():
    return [fr.get_format(name) for name in _RUNGS]


def _priced_ladder(stats: dict, costs: dict, qname: str, pricing) -> list[float]:
    """Every rung of one tensor, through the allocator's own pricing."""
    return [
        cost_entry_predicted_dloss(
            stats[qname],
            costs[qname][rung],
            format_name=rung,
            activation_pricing=pricing,
        )
        for rung in _RUNGS
    ]


def _pre_fix_costs(costs: dict) -> dict:
    """The same table as the OLD rule saw it.

    Not a reimplementation of the old pricing — the old branch never read
    ``output_mse`` on an ``output_mse_measured=False`` row, so deleting the key
    makes the CURRENT function take the weight-only branch and produce
    arithmetically the pre-fix number. Everything else (the calibration, the
    family penalty, the weight_mse ladder) is identical, so the only thing the
    comparison varies is the branch.
    """
    return {
        qname: {
            rung: {k: v for k, v in row.items()
                   if not (k == "output_mse" and rung in _INTERPOLATED_RUNGS)}
            for rung, row in rows.items()
        }
        for qname, rows in costs.items()
    }


def _violations(prices: list[float]) -> list[int]:
    """Indices where a HIGHER rung costs more than the rung below it."""
    return [i for i in range(1, len(prices)) if prices[i] > prices[i - 1]]


# ---------------------------------------------------------------------------
# 1. Rung monotonicity inside the family
# ---------------------------------------------------------------------------

def test_interpolated_rungs_are_priced_between_their_measured_neighbours():
    """The acceptance property, on BOTH projections, through the real code.

    K13 and K14 are interpolated; K12 and K15 are measured. A CB ladder spends
    bits to buy error, so price must fall monotonically in K — and each
    interpolated rung must sit strictly between the rungs on either side of
    it, on a projection whose ratio is 25x below the family constant and on
    one 5x above it.
    """
    stats, costs = _tables()
    pricing = calibrate_activation_fair_pricing(stats, costs, _specs())
    assert pricing.enabled
    assert pricing.families[_FAMILY].penalty == pytest.approx(
        _EXPECTED_PENALTY)

    for qname, ratio in ((_DOWN_PROJ, _RATIO_DOWN_PROJ),
                         (_GATE_PROJ, _RATIO_GATE_PROJ)):
        prices = _priced_ladder(stats, costs, qname, pricing)
        assert _violations(prices) == [], (
            f"{qname}: rung order broken at {_violations(prices)} "
            f"(prices={prices})")
        # Strictly between its neighbours, both interpolated rungs.
        assert prices[0] > prices[1] > prices[2] > prices[3]
        # Every rung, interpolated or not, is now 0.5*h_trace*output_mse: one
        # basis for the whole ladder, which is what makes the order meaningful.
        for rung, price in zip(_RUNGS, prices):
            assert price == pytest.approx(ratio * _weight_mse(rung))
        # The bug report's own yardstick: an interpolated rung against the
        # geometric mean of its neighbours. 12x/0.43x before, ~1.00x now.
        for i in (1, 2):
            neighbour_gm = (prices[i - 1] * prices[i + 1]) ** 0.5
            assert prices[i] / neighbour_gm == pytest.approx(1.0, rel=0.015)


def test_the_old_rule_breaks_rung_order_in_both_directions():
    """The same assertions FAIL pre-fix, so the test above catches the bug.

    Down_proj's true ratio is 10 and the family constant is 50, so its
    interpolated rungs are priced 5x high and K13 leaps over K12. Gate_proj's
    ratio is 250, so its interpolated rungs are priced 5x LOW and the measured
    K15 leaps over the under-priced K14. One constant, two failure directions,
    exactly as observed on the banked menu.
    """
    stats, costs = _tables()
    pre_fix = _pre_fix_costs(costs)
    pricing = calibrate_activation_fair_pricing(stats, costs, _specs())
    # The calibration is not what changed: the interpolated rows never entered
    # it, so stripping their output_mse leaves the same fit.
    assert calibrate_activation_fair_pricing(
        stats, pre_fix, _specs()).families[_FAMILY].penalty == pytest.approx(
            _EXPECTED_PENALTY)

    down = _priced_ladder(stats, pre_fix, _DOWN_PROJ, pricing)
    gate = _priced_ladder(stats, pre_fix, _GATE_PROJ, pricing)

    # Both projections are non-monotone, at opposite seams of the ladder.
    assert _violations(down) == [1], f"down_proj prices={down}"
    assert _violations(gate) == [3], f"gate_proj prices={gate}"
    # And the misprice is the family constant standing in for the ratio.
    assert down[1] == pytest.approx(
        _EXPECTED_PENALTY * _weight_mse("MXFP8_E5M2"))
    assert down[1] / (_RATIO_DOWN_PROJ * _weight_mse("MXFP8_E5M2")) == (
        pytest.approx(_EXPECTED_PENALTY / _RATIO_DOWN_PROJ))   # 5x over
    assert gate[1] / (_RATIO_GATE_PROJ * _weight_mse("MXFP8_E5M2")) == (
        pytest.approx(_EXPECTED_PENALTY / _RATIO_GATE_PROJ))   # 0.2x under
    # The measured rungs are untouched by the fix, on both tables.
    fixed_down = _priced_ladder(stats, costs, _DOWN_PROJ, pricing)
    assert down[0] == pytest.approx(fixed_down[0])
    assert down[3] == pytest.approx(fixed_down[3])


def test_build_candidates_prices_and_labels_the_interpolated_rung():
    """End to end through candidate construction, not just the scalar path."""
    stats, costs = _tables()
    pricing = calibrate_activation_fair_pricing(stats, costs, _specs())
    cands = build_candidates(
        stats, costs, _specs(), target_profile="research",
        activation_pricing=pricing)

    for qname, ratio in ((_DOWN_PROJ, _RATIO_DOWN_PROJ),
                         (_GATE_PROJ, _RATIO_GATE_PROJ)):
        priced = {c.fmt: c for c in cands[qname]}
        assert set(priced) == set(_RUNGS)
        ladder = [priced[rung].predicted_dloss for rung in _RUNGS]
        assert _violations(ladder) == [], f"{qname}: {ladder}"
        for rung in _INTERPOLATED_RUNGS:
            assert priced[rung].predicted_dloss == pytest.approx(
                ratio * _weight_mse(rung))
            # Its own label: priced in output space, but still a PREDICTION.
            assert priced[rung].activation_pricing == BRANCH_INTERPOLATED_OUTPUT
        for rung in _MEASURED_RUNGS:
            assert priced[rung].activation_pricing == BRANCH_MEASURED


# ---------------------------------------------------------------------------
# 2. The calibration sample is untouched
# ---------------------------------------------------------------------------

def test_interpolated_rows_never_enter_the_calibration_sample():
    """Same rows, same count, same digest, same fit — present or absent.

    Their ``output_mse`` is a fit THROUGH this family's measured anchors, so
    letting it into the sample would fit the transfer constant partly on its
    own output. Good enough to price the row it belongs to; not good enough to
    define the constant other rows are priced by.
    """
    stats, costs = _tables()
    measured_only = {
        qname: {rung: row for rung, row in rows.items()
                if rung in _MEASURED_RUNGS}
        for qname, rows in costs.items()
    }

    rows_all, measured_all, weight_only_all = (
        collect_activation_calibration_rows(stats, costs, _specs()))
    rows_ref, measured_ref, weight_only_ref = (
        collect_activation_calibration_rows(stats, measured_only, _specs()))

    assert rows_all == rows_ref
    assert measured_all == measured_ref == {_FAMILY: 4}
    assert len(rows_all) == 4
    # Not one of the four came from an interpolated rung.
    assert {row.fmt for row in rows_all} == set(_MEASURED_RUNGS)

    fit_all = calibrate_activation_fair_pricing(
        stats, costs, _specs()).families[_FAMILY]
    fit_ref = calibrate_activation_fair_pricing(
        stats, measured_only, _specs()).families[_FAMILY]
    assert fit_all.n_rows == fit_ref.n_rows == 4
    assert fit_all.rows_digest == fit_ref.rows_digest
    assert fit_all.penalty == fit_ref.penalty == pytest.approx(
        _EXPECTED_PENALTY)
    assert fit_all.sample == fit_ref.sample

    # The one census that DOES see them, recorded rather than silently
    # divergent: the weight-only population feeds calibrate()'s fail-closed
    # refusal, where over-counting can only make a run refuse more eagerly.
    assert weight_only_all == {_FAMILY: 4}
    assert weight_only_ref == {_FAMILY: 0}


# ---------------------------------------------------------------------------
# 3 & 4. Provenance, guards, and everything that must NOT move
# ---------------------------------------------------------------------------

def test_the_row_still_says_exactly_what_it_said_about_itself():
    """Only the number's USE changed, never its claims."""
    stats, costs = _tables()
    entry = costs[_DOWN_PROJ]["MXFP8_E5M2"]
    assert entry["cost_source"] == "band_interpolated"
    assert entry["output_mse_measured"] is False
    # The provenance guard (constraint: not weakened) still classifies it.
    assert cost_entry_is_band_interpolated(entry)
    # The cost-FIELD source keeps reporting the explicit provenance string...
    assert cost_entry_source(stats[_DOWN_PROJ], entry, "MXFP8_E5M2") == (
        "band_interpolated")
    # ...and "is a real measurement behind this row" is still answered no.
    assert not cost_entry_uses_measured_output_mse(
        stats[_DOWN_PROJ], entry, "MXFP8_E5M2")
    assert cost_entry_activation_pricing_branch(
        stats[_DOWN_PROJ], entry, "MXFP8_E5M2",
        calibrate_activation_fair_pricing(stats, costs, _specs()),
    ) == BRANCH_INTERPOLATED_OUTPUT


def test_the_noise_band_guard_still_drops_a_dominated_interpolated_rung():
    """``drop_interpolated_candidates_dominated_by_measured`` is unchanged,
    and becomes meaningful again now that both sides are on one basis."""
    _stats, costs = _tables()
    cands = {
        _DOWN_PROJ: [
            Candidate(fmt="MXFP4", bits_per_param=4.5,
                      memory_bytes=1000, predicted_dloss=1.0e-3),
            Candidate(fmt="MXFP8_E5M2", bits_per_param=4.6,
                      memory_bytes=1000, predicted_dloss=1.001e-3),
        ]
    }
    kept, dropped = drop_interpolated_candidates_dominated_by_measured(
        cands, costs, band=0.05)
    assert dropped == 1
    assert [c.fmt for c in kept[_DOWN_PROJ]] == ["MXFP4"]


@pytest.mark.parametrize("packed", [False, True])
def test_a_zero_output_mse_placeholder_still_takes_the_weight_only_branch(
        packed):
    """``output_mse == 0.0`` is not a usable price, whatever stamped it.

    Two real producers of that row: the packed-expert ladder path, which fits
    in WEIGHT space only and accumulates ``output_mse=0.0`` next to
    ``cost_source=band_interpolated``/``mixed``; and the dense path's
    ``float(fills["output_mse"] or 0.0)`` when the output fit could not be
    made. Neither carries output-space information, and 0.0 is the DP's global
    optimum — so both must keep falling through to weight-only pricing.
    """
    stats, costs = _tables()
    pricing = calibrate_activation_fair_pricing(stats, costs, _specs())
    stats_entry = dict(_stats_entry())
    if packed:
        stats_entry["num_experts"] = 256
    entry = {
        "weight_mse": _weight_mse("MXFP8_E5M2"),
        "output_mse": 0.0,
        "rel_output_mse": 0.0,
        "output_mse_measured": False,
        "cost_source": "band_interpolated",
    }
    priced = cost_entry_predicted_dloss(
        stats_entry, entry, format_name="MXFP8_E5M2",
        activation_pricing=pricing)
    assert priced == pytest.approx(
        _EXPECTED_PENALTY * _weight_mse("MXFP8_E5M2"))
    assert cost_entry_activation_pricing_branch(
        stats_entry, entry, "MXFP8_E5M2", pricing) == BRANCH_CALIBRATED


def test_a_mixed_row_with_no_output_number_stays_weight_only():
    """``cost_source: mixed`` is the packed path's accepted+rejected slice
    row; it too carries ``output_mse=0.0``."""
    stats, costs = _tables()
    pricing = calibrate_activation_fair_pricing(stats, costs, _specs())
    entry = {
        "weight_mse": 1.0e-4,
        "output_mse": 0.0,
        "output_mse_measured": False,
        "cost_source": "mixed",
    }
    assert cost_entry_activation_pricing_branch(
        _stats_entry(), entry, "MXFP8_E5M2", pricing) == BRANCH_CALIBRATED
    assert cost_entry_predicted_dloss(
        _stats_entry(), entry, format_name="MXFP8_E5M2",
        activation_pricing=pricing,
    ) == pytest.approx(_EXPECTED_PENALTY * 1.0e-4)


def test_a_genuinely_measured_row_is_priced_exactly_as_before():
    stats, costs = _tables()
    pricing = calibrate_activation_fair_pricing(stats, costs, _specs())
    entry = costs[_GATE_PROJ]["MXFP4"]
    assert cost_entry_predicted_dloss(
        stats[_GATE_PROJ], entry, format_name="MXFP4",
        activation_pricing=pricing,
    ) == pytest.approx(_RATIO_GATE_PROJ * _W12)
    assert cost_entry_activation_pricing_branch(
        stats[_GATE_PROJ], entry, "MXFP4", pricing) == BRANCH_MEASURED


def test_the_ucb_hedge_conversion_follows_the_pricing_branch():
    """``_super_item_ucb_hedge`` mirrors ``cost_entry_predicted_dloss``.

    A member priced from its own interpolated ``output_mse`` never had a
    ``z*stderr`` hedge added to its dloss, so the super-item conversion must
    not subtract one — that would over-subtract the linear hedge it exists to
    undo. The control below is the same row with nothing to price it in output
    space, which still hedges.
    """
    stats_entry = _stats_entry()
    interpolated = {
        "weight_mse": 1.0e-4,
        "output_mse": 1.0e-3,
        "predicted_dloss": 1.0e-4,
        "predicted_dloss_stderr": 2.0e-5,
        "output_mse_measured": False,
        "cost_source": "band_interpolated",
    }
    hedge, stderr_agg = _super_item_ucb_hedge(
        [(stats_entry, interpolated, 1.0e-3, 1.0)], ucb_z=2.0)
    assert hedge == 0.0 and stderr_agg == 0.0

    weight_only = {k: v for k, v in interpolated.items() if k != "output_mse"}
    hedge, stderr_agg = _super_item_ucb_hedge(
        [(stats_entry, weight_only, 1.0e-4, 1.0)], ucb_z=2.0)
    assert hedge == pytest.approx(2.0 * 2.0e-5)
    assert stderr_agg == pytest.approx(2.0e-5)


# ---------------------------------------------------------------------------
# 5. The census interpolated-rung guard reads the campaign's measured LOO band
# ---------------------------------------------------------------------------
#
# prismaquant #615. The GLM-5.3 Flash E4M3 allocation picked
# layers.{0,2}.mlp.{gate,up}_proj at interpolated TESSERA_E4M3_K1_R1026, a
# 0.34% predicted gain over measured R1024 against a measured
# leave-one-anchor-out error of 0.168 log2, and the census manifest then
# refused the export: the rung has no measured wire. The guard the docs said
# covered these rows had no call site.
#
# Rows below are the real layer-0 gate/up cells of that table
# (extension-r1024-02/workspace/merged/cost.pkl, sha256 cd215410...):
# (output_mse, memory_bytes, measured). The DP prices every one of them as
# 0.5 * h_trace * output_mse, and h_trace cancels in the log ratio the guard
# compares, so output_mse stands in for predicted_dloss.

_CENSUS_GATE = "model.language_model.layers.0.mlp.gate_proj"
_CENSUS_UP = "model.language_model.layers.0.mlp.up_proj"
_CENSUS_ROWS = {
    _CENSUS_GATE: {
        "TESSERA_E4M3_K1_R832": (6.568958269781433e-06, 20488192, True),
        "TESSERA_E4M3_K1_R960": (4.176926571138513e-06, 23633920, True),
        "TESSERA_E4M3_K1_R1000": (3.3746392472655855e-06, 24616960, False),
        "TESSERA_E4M3_K1_R1024": (2.9692697959641614e-06, 25206784, True),
        "TESSERA_E4M3_K1_R1026": (2.9592090667764175e-06, 25255936, False),
        "TESSERA_E4M3_K1_R1088": (2.6636753318598494e-06, 26779648, True),
        "TESSERA_BF16_K1_R1024": (1.7195825421367772e-06, 25223168, True),
        "TESSERA_BF16_K1_R1026": (1.708709278092314e-06, 25272320, False),
    },
    _CENSUS_UP: {
        "TESSERA_E4M3_K1_R832": (7.395995756572423e-06, 20488192, True),
        "TESSERA_E4M3_K1_R960": (4.592295226757415e-06, 23633920, True),
        "TESSERA_E4M3_K1_R1000": (3.6464070725970356e-06, 24616960, False),
        "TESSERA_E4M3_K1_R1024": (3.1751654508601255e-06, 25206784, True),
        "TESSERA_E4M3_K1_R1026": (3.163383422190777e-06, 25255936, False),
        "TESSERA_E4M3_K1_R1088": (2.819041886444514e-06, 26779648, True),
        "TESSERA_BF16_K1_R1024": (2.0141892491665203e-06, 25223168, True),
        "TESSERA_BF16_K1_R1026": (2.0011594897739525e-06, 25272320, False),
    },
}
#: leave_one_anchor_out[unit][family]["max_abs_log2_error"] from the same table.
_CENSUS_LOO_BANDS = {
    _CENSUS_GATE: {"TESSERA_E4M3_K1": 0.16782182781580038,
                   "TESSERA_BF16_K1": 0.23847425908406586},
    _CENSUS_UP: {"TESSERA_E4M3_K1": 0.1803783836655495,
                 "TESSERA_BF16_K1": 0.23379736289017633},
}
_CENSUS_GROUP = {("fused", "model.language_model.layers.0.mlp"):
                 (_CENSUS_GATE, _CENSUS_UP)}


def _census_world(units=(_CENSUS_GATE,)):
    candidates, costs, loo = {}, {}, {}
    for unit in units:
        rows = _CENSUS_ROWS[unit]
        costs[unit] = {
            fmt: {
                "output_mse": mse,
                "output_mse_measured": measured,
                "cost_source": ("tessera_campaign_measured" if measured
                                else "tessera_campaign_interpolated"),
                "tessera_family": fmt.rsplit("_R", 1)[0],
            }
            for fmt, (mse, _bytes, measured) in rows.items()
        }
        candidates[unit] = [
            Candidate(fmt=fmt, bits_per_param=8.0 * size / (12288 * 4096),
                      memory_bytes=size, predicted_dloss=mse)
            for fmt, (mse, size, _measured) in rows.items()
        ]
        loo[unit] = {
            family: {"interior_anchors": 2, "max_abs_log2_error": band,
                     "median_abs_log2_error": band}
            for family, band in _CENSUS_LOO_BANDS[unit].items()
        }
    return candidates, costs, loo


def _menu(candidates, unit):
    return {c.fmt for c in candidates[unit]}


def test_census_loo_band_drops_r1026_within_holdout_error():
    import math

    from prismaquant.allocator_candidates import (
        drop_census_interpolated_within_loo,
    )

    candidates, costs, loo = _census_world()
    rows = _CENSUS_ROWS[_CENSUS_GATE]
    r1024 = rows["TESSERA_E4M3_K1_R1024"][0]
    r1026 = rows["TESSERA_E4M3_K1_R1026"][0]
    r1000 = rows["TESSERA_E4M3_K1_R1000"][0]
    r960 = rows["TESSERA_E4M3_K1_R960"][0]
    band = _CENSUS_LOO_BANDS[_CENSUS_GATE]["TESSERA_E4M3_K1"]
    assert abs(math.log2(r1024 / r1026)) < 0.005 < band
    assert abs(math.log2(r1000 / r1024)) == pytest.approx(0.185, abs=1e-3)

    kept = drop_census_interpolated_within_loo(candidates, costs, loo)
    menu = _menu(kept, _CENSUS_GATE)
    # Measured R1024 is smaller in bytes and 0.005 log2 away, inside 0.168.
    assert "TESSERA_E4M3_K1_R1026" not in menu
    # Same shape in the other family: measured BF16 R1024 vs its R1026.
    assert "TESSERA_BF16_K1_R1026" not in menu
    # R1000 is 0.185 log2 from R1024, outside the band (and R1024 is larger
    # in bytes); the measured rungs no larger in bytes are R832 and R960, and
    # R960 is 0.308 log2 away. A genuine trade stays on the menu.
    assert "TESSERA_E4M3_K1_R1000" in menu
    assert {fmt for fmt, (_m, _b, measured) in rows.items() if measured} <= menu

    # The band is what decides R1000: widen it past R960's distance and it
    # goes too.
    assert abs(math.log2(r960 / r1000)) == pytest.approx(0.3077, abs=1e-3)
    wide = {unit: {family: dict(record, max_abs_log2_error=0.31)
                   for family, record in by_family.items()}
            for unit, by_family in loo.items()}
    assert "TESSERA_E4M3_K1_R1000" not in _menu(
        drop_census_interpolated_within_loo(candidates, costs, wide),
        _CENSUS_GATE)


def test_census_loo_band_is_conjunctive_across_fused_members():
    from prismaquant.allocator_candidates import (
        drop_census_interpolated_within_loo,
    )

    candidates, costs, loo = _census_world((_CENSUS_GATE, _CENSUS_UP))
    both = drop_census_interpolated_within_loo(
        candidates, costs, loo, fused_groups=_CENSUS_GROUP)
    for unit in (_CENSUS_GATE, _CENSUS_UP):
        assert "TESSERA_E4M3_K1_R1026" not in _menu(both, unit)

    # up_proj's R1026 sits 0.0054 log2 from its R1024. With a band below that,
    # only gate_proj is dominated at R1026.
    narrow = {unit: {family: dict(record) for family, record in by.items()}
              for unit, by in loo.items()}
    narrow[_CENSUS_UP]["TESSERA_E4M3_K1"]["max_abs_log2_error"] = 0.001
    alone = drop_census_interpolated_within_loo(candidates, costs, narrow)
    assert "TESSERA_E4M3_K1_R1026" not in _menu(alone, _CENSUS_GATE)
    assert "TESSERA_E4M3_K1_R1026" in _menu(alone, _CENSUS_UP)

    # Aggregation offers the group only names every member carries, so a
    # per-member drop would remove R1026 from the group although up_proj is
    # not dominated at it. The group keeps R1026 on both members.
    report = {}
    grouped = drop_census_interpolated_within_loo(
        candidates, costs, narrow, fused_groups=_CENSUS_GROUP, report=report)
    assert "TESSERA_E4M3_K1_R1026" in _menu(grouped, _CENSUS_GATE)
    assert "TESSERA_E4M3_K1_R1026" in _menu(grouped, _CENSUS_UP)
    # BF16 R1026 is dominated on both members, so it still drops.
    for unit in (_CENSUS_GATE, _CENSUS_UP):
        assert "TESSERA_BF16_K1_R1026" not in _menu(grouped, unit)
    assert report["census_loo_band"]["kept_by_group_conjunction"] == 1


@pytest.mark.parametrize("record", [
    "absent_family", {"error": "fewer than three anchors"},
    {"interior_anchors": 0, "note": "no interior anchor"},
    {"interior_anchors": 2, "max_abs_log2_error": float("nan")},
    "absent_table",
])
def test_census_loo_band_refuses_interpolated_cell_without_loo_record(record):
    from prismaquant.allocator_candidates import (
        CensusLooBandError, drop_census_interpolated_within_loo,
    )

    candidates, costs, loo = _census_world()
    if record == "absent_table":
        with pytest.raises(CensusLooBandError,
                           match="carries no leave_one_anchor_out table"):
            drop_census_interpolated_within_loo(candidates, costs, None)
        # No interpolated row anywhere: an absent table is a no-op.
        measured = {unit: {fmt: row for fmt, row in rows.items()
                           if row["output_mse_measured"]}
                    for unit, rows in costs.items()}
        assert drop_census_interpolated_within_loo(
            candidates, measured, None) is candidates
        return
    if record == "absent_family":
        del loo[_CENSUS_GATE]["TESSERA_E4M3_K1"]
    else:
        loo[_CENSUS_GATE]["TESSERA_E4M3_K1"] = record
    with pytest.raises(CensusLooBandError,
                       match="no finite leave_one_anchor_out max_abs_log2_error"):
        drop_census_interpolated_within_loo(candidates, costs, loo)


def test_census_loo_band_prints_its_own_report_line(capsys, monkeypatch):
    from prismaquant import allocator_candidates as ac

    candidates, costs, loo = _census_world()
    report = {}
    ac.drop_census_interpolated_within_loo(candidates, costs, loo,
                                           report=report)
    lines = [line for line in capsys.readouterr().out.splitlines()
             if line.startswith("[alloc] census LOO band guard:")]
    assert len(lines) == 1, lines
    assert ("dropped 2 of 3 interpolated Tessera candidate(s) on 1 unit(s)"
            in lines[0])
    assert "measured leave_one_anchor_out max_abs_log2_error" in lines[0]
    assert "stricter than --loo-gate" in lines[0]
    assert report["census_loo_band_dropped"] == 2
    assert report["census_loo_band"]["per_unit"] == {_CENSUS_GATE: 2}

    # build_candidates runs it, with the table, the groups and the same report
    # dict the menu reduction writes into.
    seen = {}

    def fake_guard(out, costs_arg, loo_arg, **kwargs):
        seen.update(loo=loo_arg, **kwargs)
        return out

    monkeypatch.setattr(ac, "drop_census_interpolated_within_loo", fake_guard)
    monkeypatch.setattr(ac, "reduce_continuous_menu",
                        lambda out, stats, **kwargs: out)
    names = ("layer.self_attn.o_proj", "layer.mlp.down_proj")
    stats = {name: {"n_params": 1600, "h_trace": 1.0, "in_features": 40,
                    "out_features": 40} for name in names}
    stock = {name: {"NVFP4": {"weight_mse": 0.01, "predicted_dloss": 0.005},
                    "BF16": {"weight_mse": 0.0, "predicted_dloss": 0.0}}
             for name in names}
    menu_report = {}
    ac.build_candidates(stats, stock, [fr.REGISTRY["NVFP4"], fr.REGISTRY["BF16"]],
                        tessera_menu_report=menu_report, census_loo=loo,
                        census_loo_groups=_CENSUS_GROUP)
    assert seen["loo"] is loo
    assert seen["fused_groups"] is _CENSUS_GROUP
    assert seen["report"] is menu_report
