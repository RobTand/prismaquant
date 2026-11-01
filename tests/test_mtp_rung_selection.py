"""Unit tests for prismaquant.mtp_rung_selection (CPU, no torch).

Validates the canon selector (docs/design/mtp_rung_selection.md) against hand-derived
synthetic constants: the degenerate branch picks the highest-fidelity rung under
the memory gate; the memory gate excludes rungs; the interior argmax matches an
independent brute force; the acceptance fit degrades correctly with 0/1/2+
points; larger k pushes the argmax to a higher-fidelity rung; provenance is
complete and JSON-serialisable.
"""
from __future__ import annotations

import json
import math
import sys

import pytest

from prismaquant.mtp_rung_selection import (
    AcceptancePoint,
    RungPoint,
    ServeConstants,
    _continuous_bstar_lambertw,
    fit_acceptance,
    select_rung,
)

_BIG = 1 << 40  # a memory budget that admits every rung in these tests


def _ideal_menu(bits, bytes_by_bits=None):
    """Menu with the idealised E(b)=2^{-2b} law (sqrt(E)=2^{-b}), so a(b) is
    exactly linear in sqrt(E) and a 2-point fit is exact."""
    out = []
    for b in bits:
        rb = bytes_by_bits[b] if bytes_by_bits else 1_000_000
        out.append(RungPoint(name=f"b{b}", bits=float(b),
                             resident_bytes=int(rb), E=2.0 ** (-2.0 * b)))
    return out


# --------------------------------------------------------------------------- #
# fit_acceptance: 0 / 1 / 2+ points
# --------------------------------------------------------------------------- #
def test_fit_zero_points_is_no_data():
    a_inf, beta, mode = fit_acceptance([], {})
    assert (a_inf, beta, mode) == (None, None, "no_data")


def test_fit_one_point_is_single_point_no_slope():
    # a_inf = the one measured acceptance; no slope (beta None) per the doc.
    pts = [AcceptancePoint(0.83, rung_name="b4")]
    a_inf, beta, mode = fit_acceptance(pts, {"b4": 2.0 ** -8})
    assert mode == "single_point"
    assert beta is None
    assert a_inf == pytest.approx(0.83)


def test_fit_two_points_least_squares_recovers_line():
    # points (sqrt E, a) = (0.2, 0.6) and (0.1, 0.9) -> beta=3, a_inf=1.2.
    pts = [AcceptancePoint(0.6, rung_name="lo"),
           AcceptancePoint(0.9, rung_name="hi")]
    a_inf, beta, mode = fit_acceptance(pts, {"lo": 0.04, "hi": 0.01})
    assert mode == "least_squares"
    assert beta == pytest.approx(3.0)
    assert a_inf == pytest.approx(1.2)


def test_fit_same_E_multiple_points_has_no_slope():
    # >=2 points but identical E -> no fidelity spread -> single_point (mean).
    pts = [AcceptancePoint(0.80, rung_name="a"),
           AcceptancePoint(0.84, rung_name="b")]
    a_inf, beta, mode = fit_acceptance(pts, {"a": 0.01, "b": 0.01})
    assert mode == "single_point"
    assert beta is None
    assert a_inf == pytest.approx(0.82)


def test_fit_missing_E_is_hard_error():
    with pytest.raises(KeyError):
        fit_acceptance([AcceptancePoint(0.8, rung_name="ghost")], {"real": 0.01})


@pytest.mark.parametrize("include_name", [False, True])
def test_selector_resolves_bits_acceptance_like_named_acceptance(include_name):
    menu = _ideal_menu([2, 3, 4, 5])
    constants = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.1)
    named = [AcceptancePoint(0.6, rung_name="b2"),
             AcceptancePoint(0.9, rung_name="b5")]
    numeric = [
        AcceptancePoint(0.6, bits=2.0, rung_name="b2" if include_name else None),
        AcceptancePoint(0.9, bits=5.0, rung_name="b5" if include_name else None),
    ]
    expected = select_rung(menu, constants, named, _BIG)
    actual = select_rung(menu, constants, numeric, _BIG)

    assert actual.rung == expected.rung
    assert actual.regime == expected.regime
    assert actual.per_rung_T == expected.per_rung_T
    for field in ("a_inf", "beta", "fit_mode"):
        assert actual.provenance["fit"][field] == expected.provenance["fit"][field]
    assert [point["E"] for point in actual.provenance["fit"]["points"]] == [
        menu[0].E, menu[-1].E,
    ]
    assert [point["rung"] for point in actual.provenance["fit"]["points"]] == [
        "b2", "b5",
    ]
    assert [point["bits"] for point in actual.provenance["fit"]["points"]] == [
        2.0, 5.0,
    ]


@pytest.mark.parametrize("include_name", [False, True])
def test_selector_refuses_ambiguous_bits_acceptance(include_name):
    menu = [RungPoint("first", 4.0, 1_000_000, 0.04),
            RungPoint("second", 4.0, 1_000_000, 0.01)]
    constants = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.1)
    point = AcceptancePoint(
        0.8, bits=4.0, rung_name="second" if include_name else None)

    with pytest.raises(ValueError, match="ambiguous acceptance bits"):
        select_rung(menu, constants, [point], _BIG)

    named = select_rung(
        menu, constants, [AcceptancePoint(0.8, rung_name="second")], _BIG)
    assert named.provenance["fit"]["points"][0]["E"] == 0.01
    assert named.rung.name == "second"


def test_selector_refuses_bits_absent_from_menu():
    constants = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.1)
    with pytest.raises(KeyError, match="no rung has bits"):
        select_rung(_ideal_menu([2, 3]), constants,
                    [AcceptancePoint(0.8, bits=4.0)], _BIG)


def test_selector_uses_numeric_calibration_from_memory_excluded_rung():
    menu = _ideal_menu([2, 5], bytes_by_bits={2: 100, 5: 200})
    constants = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.1)
    named = select_rung(
        menu, constants, [AcceptancePoint(0.9, rung_name="b5")], 100)
    numeric = select_rung(
        menu, constants, [AcceptancePoint(0.9, bits=5.0)], 100)

    assert numeric.rung == named.rung == menu[0]
    assert numeric.per_rung_T == named.per_rung_T
    assert numeric.provenance["fit"]["points"][0]["E"] == menu[1].E


# --------------------------------------------------------------------------- #
# Interior argmax matches an independent brute force
# --------------------------------------------------------------------------- #
def test_interior_argmax_matches_brute_force():
    menu = _ideal_menu([2, 3, 4, 5])
    const = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.1)  # c large
    # calibrate at the two spanning rungs using their exact a(b) values
    a_inf_true, beta_true = 0.98, 1.5
    accepts = [
        AcceptancePoint(a_inf_true - beta_true * 2.0 ** -2, rung_name="b2"),
        AcceptancePoint(a_inf_true - beta_true * 2.0 ** -5, rung_name="b5"),
    ]
    res = select_rung(menu, const, accepts, _BIG, k=1)

    # independent brute force over the same fitted curve
    a_inf, beta, _ = fit_acceptance(accepts, {r.name: r.E for r in menu})
    assert (a_inf, beta) == pytest.approx((a_inf_true, beta_true))

    def T(r):
        a = a_inf - beta * math.sqrt(r.E)
        d = const.d0_ms + const.c_ms_per_bit * r.bits
        return (1.0 + a) / (const.t_ms + d)

    best = max(menu, key=T)
    assert res.regime == "interior"
    assert res.rung.name == best.name == "b3"


def test_continuous_bstar_present_and_interior():
    menu = _ideal_menu([2, 3, 4, 5])
    const = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.1)
    accepts = [AcceptancePoint(0.98 - 1.5 * 2.0 ** -2, rung_name="b2"),
               AcceptancePoint(0.98 - 1.5 * 2.0 ** -5, rung_name="b5")]
    res = select_rung(menu, const, accepts, _BIG, k=1)
    bstar = res.provenance["continuous_bstar"]
    assert bstar is not None and 2.0 < bstar < 4.0  # ~2.92, near the discrete b3


def test_lambertw_agrees_with_fixed_point_when_scipy_present():
    pytest.importorskip("scipy")
    menu = _ideal_menu([2, 3, 4, 5])
    const = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.1)
    accepts = [AcceptancePoint(0.98 - 1.5 * 2.0 ** -2, rung_name="b2"),
               AcceptancePoint(0.98 - 1.5 * 2.0 ** -5, rung_name="b5")]
    prov = select_rung(menu, const, accepts, _BIG, k=1).provenance
    fp, lw = prov["continuous_bstar"], prov["continuous_bstar_lambertw"]
    assert lw is not None
    assert abs(fp - lw) < 0.5


# --------------------------------------------------------------------------- #
# Lambert-W must survive the d0-dominated overflow regime ((t+d0)/c >~ 1023)
# --------------------------------------------------------------------------- #
# Audit 2026-08-21: the closed form computed M via math.exp(g_over_c) with
# g_over_c = ln2*(t+d0)/c + 1, which raises OverflowError once (t+d0)/c
# passes ~1022.6 -- and returned a bare None indistinguishable in provenance
# from "scipy absent" or "no real W_-1", even though a real W_-1 exists
# there. Hy3's eager-drafter constants (t=76, d0=50, c=0.1 -> ratio 1260)
# sit deep inside the dead zone; only the fixed point kept answering.


def _lw_residual(b_star, a_inf, beta, const):
    """|stationarity residual| of 2^-b*beta*(ln2*(t+d0+c*b)+c) == c*(1+a_inf)."""
    lhs = 2.0 ** (-b_star) * beta * (
        math.log(2) * (const.t_ms + const.d0_ms + const.c_ms_per_bit * b_star)
        + const.c_ms_per_bit)
    return abs(lhs - const.c_ms_per_bit * (1.0 + a_inf))


def test_lambertw_survives_d0_dominated_overflow_regime():
    pytest.importorskip("scipy")
    const = ServeConstants(t_ms=76.0, d0_ms=50.0, c_ms_per_bit=0.1)
    # (t+d0)/c = 1260 puts ln M at -874: below even denormal float64 range,
    # so the answer comes from the exact log-space continuation of W_-1.
    value, status = _continuous_bstar_lambertw(0.98, 1.5, const)
    assert status == "log_space_continuation"
    assert value is not None
    # The stationarity equation is solved to float precision there.
    assert _lw_residual(value, 0.98, 1.5, const) < 1e-9 * 0.1 * 1.98
    # ...and agrees with the fixed point iterated to convergence.
    b = 3.0
    for _ in range(60):
        bracket = math.log(2) * (76.0 + 50.0 + 0.1 * b) + 0.1
        b = math.log(1.5 * bracket / (0.1 * 1.98)) / math.log(2)
    assert abs(value - b) < 1e-6


def test_lambertw_uses_scipy_while_the_argument_is_representable():
    pytest.importorskip("scipy")
    const = ServeConstants(t_ms=90.0, d0_ms=10.0, c_ms_per_bit=0.1)
    value, status = _continuous_bstar_lambertw(0.98, 1.5, const)
    assert status == "scipy_lambertw"
    assert value is not None
    assert _lw_residual(value, 0.98, 1.5, const) < 1e-9 * 0.1 * 1.98


def test_lambertw_overflow_threshold_now_continuous_across_the_boundary():
    pytest.importorskip("scipy")
    # Status hands over where ln((1+a_inf)/beta) - ln2*(t+d0)/c - 1 crosses
    # the -700 representability floor (between ratio 1000 and 1022 here);
    # the VALUE must stay correct and smooth on both sides.
    values = []
    statuses = []
    for ratio in (1000.0, 1022.0, 1023.0, 1260.0, 5000.0):
        span = ratio * 0.1
        const = ServeConstants(t_ms=span * 0.9, d0_ms=span * 0.1,
                               c_ms_per_bit=0.1)
        value, status = _continuous_bstar_lambertw(0.98, 1.5, const)
        assert value is not None, ratio
        assert _lw_residual(value, 0.98, 1.5, const) < 1e-8 * 0.1 * 1.98
        values.append(value)
        statuses.append(status)
    assert statuses[0] == "scipy_lambertw"
    assert all(s == "log_space_continuation" for s in statuses[1:])
    # The optimum drifts smoothly upward with (t+d0)/c; no cliff at ~1023.
    assert all(b2 > b1 for b1, b2 in zip(values, values[1:]))


def test_lambertw_reports_no_real_solution_and_invalid_constants():
    pytest.importorskip("scipy")
    # beta -> 0 makes M = (1+a_inf)/(beta*e^g) huge: outside [-1/e, 0).
    const = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=1.0)
    assert _continuous_bstar_lambertw(0.5, 0.001, const) == (
        None, "no_real_solution")
    assert _continuous_bstar_lambertw(None, 1.5, const) == (
        None, "invalid_fit_constants")
    assert _continuous_bstar_lambertw(0.5, None, const) == (
        None, "invalid_fit_constants")


def test_lambertw_scipy_absent_is_a_distinct_status(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy.special", None)
    const = ServeConstants(t_ms=76.0, d0_ms=50.0, c_ms_per_bit=0.1)
    assert _continuous_bstar_lambertw(0.98, 1.5, const) == (
        None, "scipy_absent")


def test_provenance_records_which_continuous_solver_answered():
    pytest.importorskip("scipy")
    menu = _ideal_menu([2, 3, 4, 5])
    const = ServeConstants(t_ms=76.0, d0_ms=50.0, c_ms_per_bit=0.1)
    accepts = [AcceptancePoint(0.98 - 1.5 * 2.0 ** -2, rung_name="b2"),
               AcceptancePoint(0.98 - 1.5 * 2.0 ** -5, rung_name="b5")]
    p = select_rung(menu, const, accepts, _BIG, k=1).provenance
    # In the formerly-dead overflow regime both estimators answer, and
    # provenance records which solver produced each continuous estimate.
    assert p["continuous_bstar"] is not None
    assert p["continuous_method"] == "fixed_point"
    assert p["continuous_bstar_lambertw"] is not None
    assert p["continuous_bstar_lambertw_status"] == "log_space_continuation"
    assert json.loads(json.dumps(p))["continuous_bstar_lambertw_status"] == (
        "log_space_continuation")


# --------------------------------------------------------------------------- #
# k>1 pushes the argmax up
# --------------------------------------------------------------------------- #
def test_larger_k_pushes_argmax_to_higher_fidelity():
    # Two rungs; a(b3)=0.6, a(b4)=0.9 via E=(0.04, 0.01). With t=1,d0=0,c=0.5,
    # k=1 prefers the cheaper b3; k=2 amplifies acceptance and flips to b4.
    menu = [RungPoint("b3", 3.0, 1_000_000, 0.04),
            RungPoint("b4", 4.0, 1_000_000, 0.01)]
    const = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.5)
    accepts = [AcceptancePoint(0.6, rung_name="b3"),
               AcceptancePoint(0.9, rung_name="b4")]

    r1 = select_rung(menu, const, accepts, _BIG, k=1)
    r2 = select_rung(menu, const, accepts, _BIG, k=2)
    assert r1.regime == "interior" and r2.regime == "interior"
    assert r1.rung.name == "b3"
    assert r2.rung.name == "b4"
    assert r2.rung.bits > r1.rung.bits


# --------------------------------------------------------------------------- #
# Degenerate regime (cost-flat) picks highest fidelity under the gate
# --------------------------------------------------------------------------- #
def _hy3_menu():
    # NVFP4_CB K14..K20 (k/8+0.5 bpw) then FP8_CB K28..K44 (k/8 bpw); resident
    # bytes grow with bits so the gate can bind on the top rungs.
    bits = [2.25, 2.75, 3.5, 4.5, 5.5]
    return [RungPoint(name=f"r{b}", bits=b, resident_bytes=int(b * 1e8),
                      E=2.0 ** (-2.0 * b)) for b in bits]


def test_degenerate_regime_picks_highest_fidelity():
    menu = _hy3_menu()
    const = ServeConstants(t_ms=76.0, d0_ms=50.0, c_ms_per_bit=0.1)  # eager d0
    accepts = [AcceptancePoint(0.78, rung_name="r2.75"),
               AcceptancePoint(0.92, rung_name="r5.5")]
    res = select_rung(menu, const, accepts, _BIG, k=1)
    assert res.regime == "degenerate"
    assert res.provenance["degenerate_reason"] == "cost_flat"
    assert res.provenance["degenerate_test"]["ratio"] < 0.01
    # highest fidelity == lowest E == highest bits == the 5.5 rung
    assert res.rung.name == "r5.5"


def test_memory_gate_excludes_rungs_and_reshapes_choice():
    menu = _hy3_menu()
    const = ServeConstants(t_ms=76.0, d0_ms=50.0, c_ms_per_bit=0.1)
    accepts = [AcceptancePoint(0.78, rung_name="r2.75"),
               AcceptancePoint(0.92, rung_name="r5.5")]
    # budget below the 4.5 and 5.5 rungs' bytes -> only <=3.5 pass
    budget = int(3.5 * 1e8)
    res = select_rung(menu, const, accepts, budget, k=1)
    excluded = {e["name"] for e in res.provenance["memory"]["excluded"]}
    assert excluded == {"r4.5", "r5.5"}
    assert res.provenance["memory"]["passing"] == ["r2.25", "r2.75", "r3.5"]
    # degenerate -> highest fidelity AMONG PASSING == r3.5
    assert res.regime == "degenerate"
    assert res.rung.name == "r3.5"


def test_no_rung_fits_budget_raises():
    menu = _hy3_menu()
    const = ServeConstants(t_ms=76.0, d0_ms=50.0, c_ms_per_bit=0.1)
    accepts = [AcceptancePoint(0.9, rung_name="r5.5")]
    with pytest.raises(ValueError, match="no rung fits"):
        select_rung(menu, const, accepts, 1, k=1)


# --------------------------------------------------------------------------- #
# Single-point / no-data fall back to the degenerate branch (not cost_flat)
# --------------------------------------------------------------------------- #
def test_single_point_falls_back_to_degenerate():
    # c is large (cost NOT flat), so the ONLY reason to be degenerate is the
    # missing acceptance slope from a single calibration point.
    menu = [RungPoint("b3", 3.0, 1_000_000, 0.04),
            RungPoint("b4", 4.0, 1_000_000, 0.01)]
    const = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.5)
    res = select_rung(menu, const, [AcceptancePoint(0.8, rung_name="b4")],
                      _BIG, k=1)
    assert res.regime == "degenerate"
    assert res.provenance["degenerate_reason"] == "insufficient_acceptance_data"
    assert res.provenance["degenerate_test"]["cost_flat"] is False
    assert res.provenance["fit"]["fit_mode"] == "single_point"
    assert res.rung.name == "b4"  # highest fidelity (lowest E)


def test_no_acceptance_points_still_selects_highest_fidelity():
    menu = [RungPoint("b3", 3.0, 1_000_000, 0.04),
            RungPoint("b4", 4.0, 1_000_000, 0.01)]
    const = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.5)
    res = select_rung(menu, const, [], _BIG, k=1)
    assert res.regime == "degenerate"
    assert res.provenance["fit"]["fit_mode"] == "no_data"
    assert res.rung.name == "b4"
    assert all(v is None for v in res.per_rung_T.values())  # no curve to score


# --------------------------------------------------------------------------- #
# Provenance completeness / serialisability
# --------------------------------------------------------------------------- #
def test_provenance_fields_present_and_json_serialisable():
    menu = _ideal_menu([2, 3, 4, 5])
    const = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.1)
    accepts = [AcceptancePoint(0.6, rung_name="b2"),
               AcceptancePoint(0.9, rung_name="b5")]
    res = select_rung(menu, const, accepts, _BIG, k=1, h_source="uniform")
    p = res.provenance
    for key in ("schema", "selected_rung", "regime", "degenerate_reason", "k",
                "h_source", "constants", "fit", "memory", "menu",
                "degenerate_test", "per_rung_T", "continuous_bstar",
                "continuous_method"):
        assert key in p, f"missing provenance key {key}"
    assert p["h_source"] == "uniform"
    assert p["fit"]["fit_mode"] == "least_squares"
    assert set(p["per_rung_T"]) == {"b2", "b3", "b4", "b5"}
    # documented sub-fields present
    for k in ("a_inf", "beta", "fit_mode", "n_points", "beta_negative", "points"):
        assert k in p["fit"]
    for k in ("cost_span_ms", "cycle_ms", "ratio", "cost_flat",
              "insufficient_slope"):
        assert k in p["degenerate_test"]
    assert "a_clamped" in p and "continuous_bstar_lambertw" in p
    assert p["fit"]["n_points"] == 2
    # must round-trip through JSON (doc §3.7)
    assert json.loads(json.dumps(p))["selected_rung"] == res.rung.name


def test_input_validation_fail_fast():
    with pytest.raises(ValueError):
        RungPoint("bad", -1.0, 10, 0.01)
    with pytest.raises(ValueError):
        ServeConstants(t_ms=0.0, d0_ms=0.0, c_ms_per_bit=0.1)
    with pytest.raises(ValueError):
        AcceptancePoint(1.5, rung_name="x")  # acceptance outside [0,1]
    with pytest.raises(ValueError):
        AcceptancePoint(0.5)  # neither rung_name nor bits
