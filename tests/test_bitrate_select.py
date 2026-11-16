"""Unit tests for the AURA bit-rate selectors (saturation + byte-budget).

Pure-function tests over synthetic curves — no model, no torch math (though
importing the package pulls torch via __init__ polyfills, so run under the
cu130 venv). Mirrors tests/test_allocator_kneedle.py.
"""
from __future__ import annotations

import pytest

from prismaquant.saturation_select import (
    find_saturation_bpp,
    select_under_byte_budget,
)


# --------------------------------------------------------------------------
# find_saturation_bpp
# --------------------------------------------------------------------------
def _measure_fn(kl_by, se_by):
    return lambda b: (kl_by[b], se_by[b])


def test_saturation_strictly_decreasing_picks_asymptote():
    # KL halves every bit, noise tiny: no interior point is within the band of
    # the asymptote, so B* is the densest (max-bpp) point. The trace covers the
    # full grid so non-monotone noisy points cannot be hidden by bisection.
    grid = [4.0, 5.0, 6.0, 7.0, 8.0]
    kl = {4.0: 0.16, 5.0: 0.08, 6.0: 0.04, 7.0: 0.02, 8.0: 0.01}
    se = {b: 1e-4 for b in grid}
    res = find_saturation_bpp(grid, _measure_fn(kl, se), z=2.0)
    assert res["bpp"] == pytest.approx(8.0)
    assert res["asymptote_bpp"] == pytest.approx(8.0)
    assert res["kl_asymptote"] == pytest.approx(0.01)
    assert res["n_measurements"] == len(grid)


def test_saturation_flat_tail_within_noise_picks_early_bstar():
    # 6.0/7.0/8.0 are statistically indistinguishable (band wider than their KL
    # gaps): B* should drop to 6.0, not the asymptote.
    grid = [4.0, 5.0, 6.0, 7.0, 8.0]
    kl = {4.0: 0.10, 5.0: 0.05, 6.0: 0.030, 7.0: 0.029, 8.0: 0.028}
    se = {b: 3e-3 for b in grid}
    res = find_saturation_bpp(grid, _measure_fn(kl, se), z=2.0)
    assert res["bpp"] == pytest.approx(6.0)
    assert res["kl_at_bstar"] == pytest.approx(0.030)
    # the 6.0 probe must be flagged within-noise in the trace
    within = {t["bpp"]: t["within_noise"] for t in res["trace"]}
    assert within.get(6.0) is True
    assert within.get(5.0) is False


def test_saturation_zero_stderr_degenerates_to_asymptote():
    # With no noise floor the band is 0, so only the asymptote is "within noise"
    # of itself -> B* collapses to the densest point. This is the degenerate the
    # select_validated_frontier wiring warns about (needs calib-repeats>=4).
    grid = [4.0, 5.0, 6.0, 7.0, 8.0]
    kl = {4.0: 0.10, 5.0: 0.05, 6.0: 0.030, 7.0: 0.029, 8.0: 0.028}
    se = {b: 0.0 for b in grid}
    res = find_saturation_bpp(grid, _measure_fn(kl, se), z=2.0)
    assert res["bpp"] == pytest.approx(8.0)


def test_saturation_z_widens_band_lowers_bstar():
    grid = [4.0, 5.0, 6.0, 7.0, 8.0]
    kl = {4.0: 0.10, 5.0: 0.06, 6.0: 0.040, 7.0: 0.034, 8.0: 0.030}
    se = {b: 2e-3 for b in grid}
    tight = find_saturation_bpp(grid, _measure_fn(kl, se), z=1.0)
    loose = find_saturation_bpp(grid, _measure_fn(kl, se), z=6.0)
    # a wider significance band admits lower-bpp points as "saturated"
    assert loose["bpp"] <= tight["bpp"]


def test_saturation_nonmonotone_not_within_mid_does_not_hide_lower_bstar():
    # 5.0 is within the asymptote band, 6.0 is a noisy not-within outlier.
    # The old monotone bisection probed 6.0 first and discarded the lower half,
    # returning 7.0. A full grid scan returns the actual leftmost within point.
    grid = [4.0, 5.0, 6.0, 7.0, 8.0]
    kl = {4.0: 0.10, 5.0: 0.031, 6.0: 0.050, 7.0: 0.030, 8.0: 0.028}
    se = {b: 2e-3 for b in grid}
    res = find_saturation_bpp(grid, _measure_fn(kl, se), z=2.0)

    assert res["bpp"] == pytest.approx(5.0)
    assert res["n_measurements"] == len(grid)
    within = {t["bpp"]: t["within_noise"] for t in res["trace"]}
    assert within[5.0] is True
    assert within[6.0] is False


# --------------------------------------------------------------------------
# select_under_byte_budget
# --------------------------------------------------------------------------
def _cands():
    # monotone: more bytes = more bits = lower dloss
    return [
        {"bpp": 4.5, "dloss": 0.060, "disk_bytes": 20.0e9, "label": "a45"},
        {"bpp": 5.0, "dloss": 0.040, "disk_bytes": 22.0e9, "label": "a50"},
        {"bpp": 5.5, "dloss": 0.030, "disk_bytes": 24.0e9, "label": "a55"},
        {"bpp": 6.0, "dloss": 0.024, "disk_bytes": 26.0e9, "label": "a60"},
    ]


def test_byte_budget_picks_largest_that_fits():
    sel = select_under_byte_budget(_cands(), 25.0e9)
    assert sel["feasible"] and not sel["below_floor"] and not sel["has_slack"]
    assert sel["chosen"]["label"] == "a55"          # 24GB fits, 26GB does not
    assert sel["rejected_next"]["label"] == "a60"   # next rung up
    assert sel["headroom_bytes"] == pytest.approx(1.0e9)


def test_byte_budget_below_floor():
    sel = select_under_byte_budget(_cands(), 10.0e9)
    assert not sel["feasible"] and sel["below_floor"]
    assert sel["chosen"] is None
    assert sel["rejected_next"]["label"] == "a45"   # the cheapest, still too big


def test_byte_budget_has_slack_above_ceiling():
    sel = select_under_byte_budget(_cands(), 30.0e9)
    assert sel["feasible"] and sel["has_slack"]
    assert sel["chosen"]["label"] == "a60"          # densest rung
    assert sel["rejected_next"] is None
    assert sel["headroom_bytes"] == pytest.approx(4.0e9)


def test_byte_budget_exact_fit_inclusive():
    sel = select_under_byte_budget(_cands(), 24.0e9)
    assert sel["chosen"]["label"] == "a55"          # <= budget is inclusive
    assert sel["headroom_bytes"] == pytest.approx(0.0)


def test_byte_budget_tie_breaks_to_lower_loss():
    cands = [
        {"bpp": 5.0, "dloss": 0.050, "disk_bytes": 24.0e9, "label": "hi_loss"},
        {"bpp": 5.0, "dloss": 0.030, "disk_bytes": 24.0e9, "label": "lo_loss"},
    ]
    sel = select_under_byte_budget(cands, 25.0e9)
    assert sel["chosen"]["label"] == "lo_loss"


def test_byte_budget_empty():
    sel = select_under_byte_budget([], 25.0e9)
    assert not sel["feasible"] and sel["below_floor"] and sel["chosen"] is None


def test_the_unused_bootstrap_helper_stays_deleted():
    import prismaquant.saturation_select as saturation_select
    assert not hasattr(saturation_select, "bootstrap_mean_ci")
