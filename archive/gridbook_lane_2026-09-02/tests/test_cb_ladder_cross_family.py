"""Cross-family CB-ladder calibration check — ultraplan P5a item 2.

Gridbook's ``docs/audits/ultraplan_perf_2026-08-01.md`` §6 ("Three cost-model
asymmetries", #2): *per-family fitted ladders, never cross-calibrated*.
``_cb_ladder_split`` fits ``NVFP4_CB_K`` and ``FP8_CB_K`` as separate curves
with their own anchors, holdout and law, and the DP then compares the mixed
estimators as if identical. The audit's gate for P5a is that "per-family
predicted-vs-measured Δloss residuals on held-out layers must sit in
family-symmetric bands before any cross-family verdict is published".

What this file pins:

1. Symmetric bands publish; asymmetric bands refuse to publish and name the
   offending pair and its numbers.
2. The tolerance is DERIVED, never chosen: the sampling noise of the
   difference between the two family means, floored at the resolution each
   family's own holdout gate already derived from its between-window noise
   (``_cb_ladder_holdout_tol``). No taste constant appears.
3. A failure is a verdict, not an abort — the run stays solvable and the flag
   is what travels into the artifact.
4. The expert ladder records what the check needs: the family each curve
   belongs to and the SIGNED holdout residual (magnitude alone cannot tell a
   symmetric miss from a biased one).
5. The allocator can read the verdict back out of a cost payload wherever the
   cost lane stamped it.
"""
from __future__ import annotations

import pytest

from prismaquant.cb_ladder_cross_family import (
    MIN_HOLDOUTS_PER_FAMILY,
    PROVENANCE_KEY,
    VERDICT_FAIL,
    VERDICT_INSUFFICIENT,
    VERDICT_PASS,
    VERDICT_SINGLE_FAMILY,
    cross_family_holdout_verdict,
    cross_family_verdict_from_cost_payload,
    ladder_records_from_unit_kls,
    verdict_from_unit_kls,
)


def _record(family, signed, *, tol=0.0, unit="u", accepted=True):
    return {
        "unit": unit,
        "family": family,
        "accepted": accepted,
        "holdout": f"{family}39",
        "holdout_rel_err": abs(signed),
        "holdout_signed_rel_resid": signed,
        "holdout_tol": tol,
    }


# ---------------------------------------------------------------------------
# 1. Pass / fail
# ---------------------------------------------------------------------------

def test_symmetric_family_bands_publish_a_cross_family_verdict():
    """Both ladders miss by about the same amount in the same direction, and
    the gap between their means is inside their own sampling noise: the
    estimators are on one scale, so an NVFP4-CB vs FP8-CB claim drawn from
    them compares FORMATS."""
    records = (
        [_record("NVFP4_CB_K", v, unit=f"a{i}")
         for i, v in enumerate((0.008, 0.010, 0.012, 0.014))]
        + [_record("FP8_CB_K", v, unit=f"b{i}")
           for i, v in enumerate((0.009, 0.011, 0.013, 0.015))]
    )
    verdict = cross_family_holdout_verdict(records)
    assert verdict["verdict"] == VERDICT_PASS
    assert verdict["cross_family_comparison_publishable"]
    assert verdict["asymmetric_pairs"] == []
    assert set(verdict["families"]) == {"NVFP4_CB_K", "FP8_CB_K"}
    assert verdict["families"]["NVFP4_CB_K"]["n_holdouts"] == 4
    [pair] = verdict["pairs"]
    assert pair["delta"] < pair["tolerance"]
    assert "publish" in verdict["detail"]


def test_a_zero_width_band_certifies_nothing_either_way():
    """Identical residuals on every unit and no gate-derived floor is a
    degenerate estimator with no resolution — the same input
    ``_cb_ladder_holdout_tol`` refuses rather than trusts. A zero-width band
    would call any nonzero gap asymmetric on no evidence at all."""
    records = (
        [_record("NVFP4_CB_K", 0.010, unit=f"a{i}") for i in range(4)]
        + [_record("FP8_CB_K", 0.011, unit=f"b{i}") for i in range(4)]
    )
    verdict = cross_family_holdout_verdict(records)
    assert verdict["verdict"] == VERDICT_INSUFFICIENT
    assert not verdict["cross_family_comparison_publishable"]
    [pair] = verdict["pairs"]
    assert pair["degenerate_resolution"] and pair["symmetric"] is None
    assert "no resolution" in verdict["detail"]


def test_asymmetric_family_bands_refuse_to_publish_and_say_which():
    """One family's interpolated rungs are biased +20% while the other's sit
    at zero. The allocation is still solvable — the CROSS-FAMILY verdict is
    what becomes unpublishable."""
    records = (
        [_record("NVFP4_CB_K", 0.20 + 0.001 * i, unit=f"a{i}")
         for i in range(4)]
        + [_record("FP8_CB_K", 0.001 * i, unit=f"b{i}") for i in range(4)]
    )
    verdict = cross_family_holdout_verdict(records)
    assert verdict["verdict"] == VERDICT_FAIL
    assert not verdict["cross_family_comparison_publishable"]
    assert verdict["asymmetric_pairs"] == [["FP8_CB_K", "NVFP4_CB_K"]]
    [pair] = verdict["pairs"]
    assert pair["delta"] == pytest.approx(0.20, abs=1e-6)
    assert pair["delta"] > pair["tolerance"]
    assert "ASYMMETRIC" in verdict["detail"]
    assert "compares" in verdict["detail"] and "estimators" in verdict["detail"]


def test_opposite_sign_misses_of_equal_magnitude_are_caught():
    """The reason the SIGNED residual is recorded: |residual| alone reads
    these two families as identically well fitted, while their interpolated
    rungs are biased in opposite directions — which is exactly a systematic
    cross-family tilt in the DP."""
    records = (
        [_record("NVFP4_CB_K", +v, tol=0.05, unit=f"a{i}")
         for i, v in enumerate((0.14, 0.15, 0.16))]
        + [_record("FP8_CB_K", -v, tol=0.05, unit=f"b{i}")
           for i, v in enumerate((0.14, 0.15, 0.16))]
    )
    magnitudes = sorted({rec["holdout_rel_err"] for rec in records})
    assert magnitudes == [0.14, 0.15, 0.16]   # identical by magnitude
    verdict = cross_family_holdout_verdict(records)
    assert verdict["verdict"] == VERDICT_FAIL
    assert verdict["pairs"][0]["delta"] == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# 2. The tolerance is derived, not chosen
# ---------------------------------------------------------------------------

def test_tolerance_is_the_sampling_noise_of_the_difference():
    """With no per-family derived floor, the band is exactly
    sqrt(se_a**2 + se_b**2) over the two mean residuals."""
    a = [0.00, 0.02, 0.04, 0.06]
    b = [0.10, 0.12, 0.14, 0.16]
    records = (
        [_record("NVFP4_CB_K", v, unit=f"a{i}") for i, v in enumerate(a)]
        + [_record("FP8_CB_K", v, unit=f"b{i}") for i, v in enumerate(b)]
    )
    verdict = cross_family_holdout_verdict(records)
    [pair] = verdict["pairs"]
    band_a = verdict["families"]["NVFP4_CB_K"]
    band_b = verdict["families"]["FP8_CB_K"]
    expect = (band_a["stderr_signed_rel_resid"] ** 2
              + band_b["stderr_signed_rel_resid"] ** 2) ** 0.5
    assert pair["tolerance_sampling_term"] == pytest.approx(expect)
    assert pair["tolerance_derived_floor"] == 0.0
    assert pair["tolerance"] == pytest.approx(expect)
    # The two means differ by 0.10, far past that noise.
    assert pair["delta"] == pytest.approx(0.10)
    assert verdict["verdict"] == VERDICT_FAIL


def test_tolerance_floors_at_the_resolution_each_gate_already_declared():
    """A gap smaller than the tolerance a family's OWN holdout gate derived
    from its between-window noise is not a measurable asymmetry — the same
    datum ``_cb_ladder_holdout_tol`` builds, reused one level up."""
    a = [0.00, 0.02, 0.04, 0.06]
    b = [0.10, 0.12, 0.14, 0.16]
    records = (
        [_record("NVFP4_CB_K", v, tol=0.5, unit=f"a{i}")
         for i, v in enumerate(a)]
        + [_record("FP8_CB_K", v, tol=0.5, unit=f"b{i}")
           for i, v in enumerate(b)]
    )
    verdict = cross_family_holdout_verdict(records)
    [pair] = verdict["pairs"]
    assert pair["tolerance_derived_floor"] == pytest.approx(0.5)
    assert pair["tolerance"] == pytest.approx(0.5)
    # Same 0.10 gap as the test above, now inside the declared resolution.
    assert pair["delta"] == pytest.approx(0.10)
    assert verdict["verdict"] == VERDICT_PASS
    assert "max(sqrt(se_a^2 + se_b^2)" in verdict["tolerance_rule"]


# ---------------------------------------------------------------------------
# 3. Not-enough-evidence is not a pass
# ---------------------------------------------------------------------------

def test_one_family_alone_is_not_a_cross_family_verdict():
    records = [_record("FP8_CB_K", 0.01, unit=f"b{i}") for i in range(4)]
    verdict = cross_family_holdout_verdict(records)
    assert verdict["verdict"] == VERDICT_SINGLE_FAMILY
    assert not verdict["cross_family_comparison_publishable"]


def test_too_few_holdouts_is_insufficient_data_not_a_pass():
    """One held-out unit has a zero-width band, which would declare every gap
    asymmetric; two is the smallest sample with a spread, matching the guard
    ``_cb_ladder_holdout_tol`` puts on its own window sample."""
    assert MIN_HOLDOUTS_PER_FAMILY == 2
    records = [
        _record("NVFP4_CB_K", 0.01, unit="a0"),
        _record("FP8_CB_K", 0.20, unit="b0"),
    ]
    verdict = cross_family_holdout_verdict(records)
    assert verdict["verdict"] == VERDICT_INSUFFICIENT
    assert not verdict["cross_family_comparison_publishable"]
    assert verdict["families_below_min_holdouts"] == {
        "NVFP4_CB_K": 1, "FP8_CB_K": 1}


def test_no_ladder_records_at_all_is_insufficient_data():
    verdict = cross_family_holdout_verdict([])
    assert verdict["verdict"] == VERDICT_INSUFFICIENT
    assert verdict["families"] == {}


# ---------------------------------------------------------------------------
# 4. What the expert ladder records
# ---------------------------------------------------------------------------

def test_records_are_flattened_from_the_per_unit_ladder_metadata():
    unit_kls = {
        "model.layers.1.mlp.experts": {
            "FP8_CB_K36": 1e-3,
            "_ladder": [
                {"accepted": True, "family": "FP8_CB_K",
                 "holdout": "FP8_CB_K40", "holdout_rel_err": 0.02,
                 "holdout_signed_rel_resid": -0.02, "holdout_tol": 0.05},
                {"accepted": False, "family": "NVFP4_CB_K",
                 "holdout": "NVFP4_CB_K18", "holdout_rel_err": 0.3,
                 "holdout_signed_rel_resid": 0.3, "holdout_tol": 0.05},
            ],
        },
        "model.layers.0.mlp.experts": {"_sampling": {"num_experts": 8}},
    }
    records = ladder_records_from_unit_kls(unit_kls)
    assert len(records) == 2
    assert {r["family"] for r in records} == {"FP8_CB_K", "NVFP4_CB_K"}
    # Rejected ladders count too: their residual is still evidence about that
    # family's law, and dropping them would bias the band toward the fits
    # that happened to pass.
    assert sorted(r["accepted"] for r in records) == [False, True]
    assert verdict_from_unit_kls(unit_kls)["verdict"] == VERDICT_INSUFFICIENT


def test_the_expert_ladder_records_family_and_signed_residual():
    """Both new fields come from the SHARED law, so the signed residual and
    the gate's magnitude cannot disagree about which law produced them."""
    from prismaquant.expert_empirical_cost import (
        _cb_ladder_gate,
        _cb_ladder_signed_residual,
        _ladder_family_prefix,
    )

    kmap = {f"FP8_CB_K{k}": k for k in (28, 38, 39, 48)}
    assert _ladder_family_prefix(kmap) == "FP8_CB_K"
    anchors = ["FP8_CB_K28", "FP8_CB_K38", "FP8_CB_K48"]
    holdout = "FP8_CB_K39"
    # An exactly-on-law ladder: zero residual, either sign.
    from prismaquant.expert_empirical_cost import _cb_ladder_rate_factor
    exact = {
        f: 0.25 + 2.0 * _cb_ladder_rate_factor(f, kmap[f]) for f in kmap
    }
    assert _cb_ladder_signed_residual(
        kmap, anchors, exact, holdout) == pytest.approx(0.0, abs=1e-9)
    # Over-predicted holdout -> POSITIVE signed residual, and its magnitude
    # is what the gate reports.
    biased = dict(exact)
    biased[holdout] = exact[holdout] / 1.25
    signed = _cb_ladder_signed_residual(kmap, anchors, biased, holdout)
    assert signed > 0.0
    _law, rel, _tol = _cb_ladder_gate(kmap, anchors, biased, holdout, 0.10)
    assert rel == pytest.approx(abs(signed), rel=1e-9)


def test_signed_residual_is_none_when_the_law_or_holdout_is_unusable():
    from prismaquant.expert_empirical_cost import _cb_ladder_signed_residual

    kmap = {f"FP8_CB_K{k}": k for k in (28, 38, 39, 48)}
    anchors = ["FP8_CB_K28", "FP8_CB_K38", "FP8_CB_K48"]
    values = {f: 1.0 for f in kmap}
    values["FP8_CB_K39"] = 0.0
    assert _cb_ladder_signed_residual(
        kmap, anchors, values, "FP8_CB_K39") is None
    assert _cb_ladder_signed_residual(
        kmap, anchors, {}, "FP8_CB_K39") is None


# ---------------------------------------------------------------------------
# 5. Reading it back
# ---------------------------------------------------------------------------

def test_verdict_is_read_back_from_a_cost_payload_wherever_it_was_stamped():
    verdict = {"schema": "x", "verdict": VERDICT_FAIL,
               "cross_family_comparison_publishable": False}
    nested = {"provenance": {"expert_empirical_cost": {
        PROVENANCE_KEY: verdict}}}
    root = {"provenance": {PROVENANCE_KEY: verdict}}
    assert cross_family_verdict_from_cost_payload(nested) == verdict
    assert cross_family_verdict_from_cost_payload(root) == verdict
    # Payloads that predate the stamp read back as None, not as a pass.
    assert cross_family_verdict_from_cost_payload({"provenance": {}}) is None
    assert cross_family_verdict_from_cost_payload({}) is None
    assert cross_family_verdict_from_cost_payload(None) is None
