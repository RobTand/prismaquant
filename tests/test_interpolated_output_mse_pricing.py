"""Band-interpolated rungs: provenance, calibration and guards.

Until #3448 an interpolated rung (``output_mse_measured=False`` with an
interpolated ``cost_source``) was priced from its OWN ``output_mse`` (times
``0.5*h_trace``), because one per-family activation constant cannot serve
projections whose ``output_mse/weight_mse`` ratios span ~20x (2026-08-07,
``dq-runs/nvfp4-cbl/MENU_PRICING_BUG.md``). The probe measures no Fisher since
#3448, so no row is priced from ``output_mse``: a row without
``predicted_dloss`` has no price, and one with it takes the weight-only branch.

What this file still pins:

1. The calibration sample is untouched by interpolated rows: same rows,
   same count, same digest, same fitted penalty whether or not they are
   present. Their ``output_mse`` is derived from this family's own measured
   anchors, so admitting it would be circular.
2. The row's own claims about itself are unchanged, and the provenance and
   noise-band guards keep full strength.
3. The ``output_mse == 0.0`` placeholder takes the calibrated weight-only
   branch, and the super-item UCB conversion removes every member's hedge.
4. The census interpolated-rung guard reads the campaign's measured LOO band.
"""
from __future__ import annotations

import pytest

from prismaquant import format_registry as fr
from prismaquant.activation_fair_pricing import BRANCH_CALIBRATED
from prismaquant.allocator_candidates import (
    _super_item_ucb_hedge,
    calibrate_activation_fair_pricing,
    collect_activation_calibration_rows,
    drop_interpolated_candidates_dominated_by_measured,
)
from prismaquant.priced_cell import (
    cost_entry_activation_pricing_branch,
    TESSERA_INTERPOLATED_COST_SOURCE,
    cost_entry_is_band_interpolated,
    cost_entry_predicted_dloss,
    cost_entry_source,
    cost_entry_uses_measured_output_mse,
)
from prismaquant.allocator_solver import Candidate

# One family, four rungs: the two ends measured, the two middle ones
# interpolated. Until 2026-09-25 these were four consecutive rungs of the
# retired NVFP4 codebook ladder (K12..K15; archived, #1304), stamped
# ``band_interpolated``, which now refuses (``test_a_retired_ladder_row_refuses``).
# The Tessera campaign's fitted rows take the same branch under their own
# stamp, so the rule is pinned here with that stamp on the four
# activation-quantizing rungs of the live ``mx`` family. The pricing never
# reads a rung's bytes, so their order is only a label order.
_RUNGS = ["MXFP4", "MXFP8_E5M2", "MXFP8_UE8M0_G32", "MXFP8_E4M3"]
_MEASURED_RUNGS = ("MXFP4", "MXFP8_E4M3")
_INTERPOLATED_RUNGS = ("MXFP8_E5M2", "MXFP8_UE8M0_G32")
_FAMILY = "mx"

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
    ``output_mse_measured: false``, and an interpolated ``cost_source``, now
    the Tessera campaign's; no ``predicted_dloss``).
    """
    rows: dict[str, dict] = {}
    for rung in _RUNGS:
        weight_mse = _weight_mse(rung)
        row = {"weight_mse": weight_mse, "output_mse": ratio * weight_mse}
        if rung in _INTERPOLATED_RUNGS:
            row["rel_output_mse"] = 0.1 * ratio * weight_mse
            row["output_mse_measured"] = False
            row["cost_source"] = TESSERA_INTERPOLATED_COST_SOURCE
            row["tessera_family"] = _FAMILY
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
    assert entry["cost_source"] == TESSERA_INTERPOLATED_COST_SOURCE
    assert entry["output_mse_measured"] is False
    # The provenance guard (constraint: not weakened) still classifies it.
    assert cost_entry_is_band_interpolated(entry)
    # The cost-FIELD source keeps reporting the explicit provenance string...
    assert cost_entry_source(stats[_DOWN_PROJ], entry, "MXFP8_E5M2") == (
        TESSERA_INTERPOLATED_COST_SOURCE)
    # ...and "is a real measurement behind this row" is still answered no.
    assert not cost_entry_uses_measured_output_mse(
        stats[_DOWN_PROJ], entry, "MXFP8_E5M2")
    # No output-space price since #3448: the row takes the calibrated
    # weight-only branch.
    assert cost_entry_activation_pricing_branch(
        stats[_DOWN_PROJ], entry, "MXFP8_E5M2",
        calibrate_activation_fair_pricing(stats, costs, _specs()),
    ) == BRANCH_CALIBRATED


def test_the_noise_band_guard_still_drops_a_dominated_interpolated_rung():
    """``drop_interpolated_candidates_dominated_by_measured`` is unchanged."""
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

    The retired ladder wrote such rows (a weight-space-only packed fit, and a
    dense fit that could not be made); an interpolated row of any stamp at 0.0
    carries no output-space information, and 0.0 is the DP's global optimum —
    so it takes the calibrated weight-only branch on its ``predicted_dloss``.
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
        "cost_source": TESSERA_INTERPOLATED_COST_SOURCE,
        "predicted_dloss": _weight_mse("MXFP8_E5M2"),
    }
    priced = cost_entry_predicted_dloss(
        stats_entry, entry, format_name="MXFP8_E5M2",
        activation_pricing=pricing)
    assert priced == pytest.approx(
        _EXPECTED_PENALTY * _weight_mse("MXFP8_E5M2"))
    assert cost_entry_activation_pricing_branch(
        stats_entry, entry, "MXFP8_E5M2", pricing) == BRANCH_CALIBRATED


@pytest.mark.parametrize("source", ["band_interpolated", "mixed"])
@pytest.mark.parametrize("output_mse", [0.0, 1.0e-3])
def test_a_retired_ladder_row_refuses(source, output_mse):
    """The retired RD ladder's stamps (codebook lane, archived 2026-09-25,
    #1304) have no producer left. A stale row carrying one refuses, naming
    the archive, instead of being priced on either branch."""
    from prismaquant.format_registry import RetiredFormatError

    stats, costs = _tables()
    pricing = calibrate_activation_fair_pricing(stats, costs, _specs())
    entry = {
        "weight_mse": 1.0e-4,
        "output_mse": output_mse,
        "output_mse_measured": False,
        "cost_source": source,
    }
    with pytest.raises(RetiredFormatError, match="gridbook_lane"):
        cost_entry_activation_pricing_branch(
            _stats_entry(), entry, "MXFP8_E5M2", pricing)
    with pytest.raises(RetiredFormatError, match="gridbook_lane"):
        cost_entry_predicted_dloss(
            _stats_entry(), entry, format_name="MXFP8_E5M2",
            activation_pricing=pricing)
    with pytest.raises(RetiredFormatError, match="gridbook_lane"):
        cost_entry_is_band_interpolated(entry)


def test_the_ucb_hedge_conversion_removes_every_member_hedge():
    """``_super_item_ucb_hedge`` mirrors ``cost_entry_predicted_dloss``.

    Every member is priced on its ``predicted_dloss`` with the ``z*stderr``
    hedge added (no member is priced from its ``output_mse`` since #3448), so
    the super-item conversion removes the hedge from an interpolated member
    and from the same row without an ``output_mse`` alike.
    """
    stats_entry = _stats_entry()
    interpolated = {
        "weight_mse": 1.0e-4,
        "output_mse": 1.0e-3,
        "predicted_dloss": 1.0e-4,
        "predicted_dloss_stderr": 2.0e-5,
        "output_mse_measured": False,
        "cost_source": TESSERA_INTERPOLATED_COST_SOURCE,
    }
    weight_only = {k: v for k, v in interpolated.items() if k != "output_mse"}
    for row in (interpolated, weight_only):
        hedge, stderr_agg = _super_item_ucb_hedge(
            [(stats_entry, row, 1.0e-4, 1.0)], ucb_z=2.0)
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
# (output_mse, memory_bytes, measured). The guard compares log ratios of the
# candidates' prices, so output_mse stands in for predicted_dloss.

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
    stats = {name: {"n_params": 1600, "in_features": 40,
                    "out_features": 40} for name in names}
    stock = {name: {"NVFP4": {"weight_mse": 0.01, "predicted_dloss": 0.005},
                    "BF16": {"weight_mse": 0.0, "predicted_dloss": 0.0}}
             for name in names}
    menu_report = {}
    ac.build_candidates(stats, stock, [fr.REGISTRY["NVFP4"], fr.REGISTRY["BF16"]],
                        tessera_menu_report=menu_report, census_loo=loo,
                        census_loo_groups=_CENSUS_GROUP).candidates
    assert seen["loo"] is loo
    assert seen["fused_groups"] is _CENSUS_GROUP
    assert seen["report"] is menu_report
