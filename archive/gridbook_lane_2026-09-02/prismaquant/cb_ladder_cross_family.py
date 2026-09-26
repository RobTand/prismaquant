"""Cross-family CB-ladder calibration check (ultraplan P5a, item 2).

Gridbook's ultraplan performance audit
(``docs/audits/ultraplan_perf_2026-08-01.md`` §6) names the second cost-model
asymmetry: **per-family fitted ladders, never cross-calibrated.**
``expert_empirical_cost._cb_ladder_split`` fits ``NVFP4_CB_K`` and
``FP8_CB_K`` as separate curves — own anchors, own holdout, own law — and the
DP then compares the two families' rungs as if their estimators were
interchangeable. The audit's gate for P5a is explicit:

    per-family predicted-vs-measured Δloss residuals on held-out layers must
    sit in **family-symmetric bands** before any cross-family verdict is
    published.

This module owns that check. It is deliberately torch-free so the allocator
can read a verdict out of a cost payload without importing the measurement
stack, and it computes nothing the cost run did not already measure: every
number here comes from the per-unit ladder metadata
``measure_expert_unit_costs`` records (anchors, holdout, the holdout's
relative residual, and the tolerance the gate derived from that unit's
between-window noise).

**The tolerance is derived, not chosen.** ``_cb_ladder_holdout_tol`` builds a
per-unit tolerance as the *standard error of the paired holdout residual*
relative to the holdout value — the between-draw noise that survives pairing.
This check reuses exactly that construction one level up: each family's band
is the standard error of ITS mean holdout residual across held-out units, and
two families are symmetric when the gap between their mean residuals fits
inside the combined band

    tol = max( sqrt(se_a**2 + se_b**2), max(median_tol_a, median_tol_b) )

The first term is the sampling noise of the difference; the second floors it
at the resolution each family's OWN gate already declared it could not see
past, because a difference smaller than that is not a measurable asymmetry.
No taste constant appears anywhere (house rule 2).

**A failure does not abort the run.** An asymmetric band means the
cross-family *verdict* is unpublishable, not that the allocation is invalid —
so the check stamps ``publishable: false`` with the numbers, the cost stage
logs it, and the allocator surfaces it in its diagnostics and selection
provenance where a consumer of the artifact can see it.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

SCHEMA = "prismaquant.cb_ladder.cross_family_verdict.v1"

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_INSUFFICIENT = "insufficient_data"
VERDICT_SINGLE_FAMILY = "single_family_no_cross_verdict"

# Two held-out units is the smallest sample with a spread; one gives a point
# estimate whose band is 0 and would declare every gap asymmetric. Matches the
# `n < 2` guard `_cb_ladder_holdout_tol` uses on its own window sample.
MIN_HOLDOUTS_PER_FAMILY = 2

PROVENANCE_KEY = "cb_ladder_cross_family_verdict"


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def ladder_records_from_unit_kls(
    unit_kls: Mapping[str, Mapping],
) -> list[dict]:
    """Flatten ``{unit: {"_ladder": [meta, ...]}}`` into per-family records.

    One record per (held-out unit, family) — a unit is the "held-out layer"
    of the audit's gate: its ladder anchors are measured on that unit and its
    holdout rung predicted from them, so the residual is out-of-sample for
    that unit's law.
    """
    records: list[dict] = []
    for unit in sorted(unit_kls):
        meta = unit_kls[unit]
        if not isinstance(meta, Mapping):
            continue
        ladders = meta.get("_ladder")
        if not isinstance(ladders, (list, tuple)):
            continue
        for entry in ladders:
            if not isinstance(entry, Mapping):
                continue
            family = entry.get("family")
            if not family:
                continue
            records.append({
                "unit": unit,
                "family": str(family),
                "accepted": bool(entry.get("accepted", False)),
                "holdout": entry.get("holdout"),
                "holdout_rel_err": entry.get("holdout_rel_err"),
                "holdout_signed_rel_resid": entry.get(
                    "holdout_signed_rel_resid"),
                "holdout_tol": entry.get("holdout_tol"),
            })
    return records


def _family_band(records: Sequence[Mapping]) -> dict | None:
    """Mean signed holdout residual + its standard error, for one family."""
    signed = [
        float(rec["holdout_signed_rel_resid"])
        for rec in records
        if rec.get("holdout_signed_rel_resid") is not None
        and math.isfinite(float(rec["holdout_signed_rel_resid"]))
    ]
    if len(signed) < MIN_HOLDOUTS_PER_FAMILY:
        return None
    n = len(signed)
    mean = sum(signed) / n
    var = sum((value - mean) ** 2 for value in signed) / (n - 1)
    stdev = math.sqrt(var)
    stderr = stdev / math.sqrt(n)
    abs_rel = [
        float(rec["holdout_rel_err"])
        for rec in records
        if rec.get("holdout_rel_err") is not None
        and math.isfinite(float(rec["holdout_rel_err"]))
    ]
    tols = [
        float(rec["holdout_tol"])
        for rec in records
        if rec.get("holdout_tol") is not None
        and math.isfinite(float(rec["holdout_tol"]))
    ]
    return {
        "n_holdouts": n,
        "mean_signed_rel_resid": mean,
        "stdev_signed_rel_resid": stdev,
        "stderr_signed_rel_resid": stderr,
        "band": [mean - stderr, mean + stderr],
        "mean_abs_rel_err": (sum(abs_rel) / len(abs_rel)) if abs_rel else None,
        "max_abs_rel_err": max(abs_rel) if abs_rel else None,
        "median_derived_tol": _median(tols) if tols else None,
        "n_accepted": sum(1 for rec in records if rec.get("accepted")),
        "n_rejected": sum(1 for rec in records if not rec.get("accepted")),
    }


def cross_family_holdout_verdict(records: Iterable[Mapping]) -> dict:
    """Compute the family-symmetry verdict from per-family holdout records.

    Deterministic in the record order it is given (families and pairs are
    sorted), so the same cost run always stamps the same verdict.
    """
    by_family: dict[str, list[Mapping]] = {}
    for rec in records:
        family = str(rec.get("family", "") or "")
        if family:
            by_family.setdefault(family, []).append(rec)

    bands: dict[str, dict] = {}
    skipped: dict[str, int] = {}
    for family in sorted(by_family):
        band = _family_band(by_family[family])
        if band is None:
            skipped[family] = len(by_family[family])
        else:
            bands[family] = band

    pairs: list[dict] = []
    families = sorted(bands)
    for i, fam_a in enumerate(families):
        for fam_b in families[i + 1:]:
            band_a, band_b = bands[fam_a], bands[fam_b]
            delta = abs(
                band_a["mean_signed_rel_resid"]
                - band_b["mean_signed_rel_resid"]
            )
            sampling = math.sqrt(
                band_a["stderr_signed_rel_resid"] ** 2
                + band_b["stderr_signed_rel_resid"] ** 2
            )
            derived = [
                band["median_derived_tol"]
                for band in (band_a, band_b)
                if band["median_derived_tol"] is not None
            ]
            floor = max(derived) if derived else 0.0
            tolerance = max(sampling, floor)
            # Zero tolerance means the datum has NO resolution: both families'
            # holdout residuals were identical across every unit and neither
            # gate derived a floor. `_cb_ladder_holdout_tol` refuses the same
            # degenerate input rather than trusting it, and so must this — a
            # zero-width band would call any nonzero gap asymmetric on no
            # evidence at all.
            degenerate = tolerance <= 0.0
            pairs.append({
                "families": [fam_a, fam_b],
                "mean_signed_rel_resid": [
                    band_a["mean_signed_rel_resid"],
                    band_b["mean_signed_rel_resid"],
                ],
                "delta": delta,
                "tolerance": tolerance,
                "tolerance_sampling_term": sampling,
                "tolerance_derived_floor": floor,
                "degenerate_resolution": degenerate,
                "symmetric": None if degenerate else bool(delta <= tolerance),
            })

    resolved = [pair for pair in pairs if not pair["degenerate_resolution"]]
    if len(families) < 2:
        verdict = (
            VERDICT_SINGLE_FAMILY if families else VERDICT_INSUFFICIENT
        )
    elif not resolved:
        verdict = VERDICT_INSUFFICIENT
    elif all(pair["symmetric"] for pair in resolved):
        verdict = VERDICT_PASS
    else:
        verdict = VERDICT_FAIL

    asymmetric = [
        pair for pair in resolved if not pair["symmetric"]
    ]
    return {
        "schema": SCHEMA,
        "verdict": verdict,
        # The one flag a consumer needs: may a cross-family (NVFP4-CB vs
        # FP8-CB) claim be drawn from this run's ladder fits at all?
        "cross_family_comparison_publishable": verdict == VERDICT_PASS,
        "min_holdouts_per_family": MIN_HOLDOUTS_PER_FAMILY,
        "tolerance_rule": (
            "max(sqrt(se_a^2 + se_b^2), max(median_derived_holdout_tol)) — "
            "the sampling noise of the difference, floored at the resolution "
            "each family's own holdout gate derived from its between-window "
            "noise (expert_empirical_cost._cb_ladder_holdout_tol)"
        ),
        "families": bands,
        "families_below_min_holdouts": skipped,
        "pairs": pairs,
        "asymmetric_pairs": [pair["families"] for pair in asymmetric],
        "detail": _verdict_detail(verdict, pairs, bands, skipped),
    }


def _verdict_detail(verdict: str, pairs: Sequence[Mapping],
                    bands: Mapping[str, Mapping],
                    skipped: Mapping[str, int]) -> str:
    if verdict == VERDICT_PASS:
        return (
            "per-family holdout residual bands are symmetric within the "
            "derived tolerance; a cross-family verdict may be published"
        )
    if verdict == VERDICT_SINGLE_FAMILY:
        return (
            "only one CB family produced holdout residuals "
            f"({sorted(bands)}), so there is no cross-family comparison to "
            "certify"
        )
    if verdict == VERDICT_INSUFFICIENT:
        if pairs:
            return (
                "every family pair has a zero-width residual band (identical "
                "holdout residuals across units and no gate-derived floor), "
                "so the ladder fits have no resolution to certify a "
                "cross-family comparison with"
            )
        return (
            "no CB family reached "
            f"{MIN_HOLDOUTS_PER_FAMILY} held-out units "
            f"(seen: {dict(sorted(skipped.items()))}); the ladder fits carry "
            "no cross-family evidence either way"
        )
    worst = max(
        (pair for pair in pairs if not pair["symmetric"]),
        key=lambda pair: pair["delta"] - pair["tolerance"],
    )
    return (
        "per-family holdout residual bands are ASYMMETRIC: "
        f"{worst['families'][0]} mean {worst['mean_signed_rel_resid'][0]:+.2%} "
        f"vs {worst['families'][1]} mean "
        f"{worst['mean_signed_rel_resid'][1]:+.2%} "
        f"(|Δ|={worst['delta']:.2%} > tol {worst['tolerance']:.2%}). One "
        "family's interpolated rungs are biased relative to the other's, so "
        "any NVFP4-CB vs FP8-CB verdict drawn from this cost run compares "
        "estimators, not formats. The allocation is still solvable — this "
        "flag says the CROSS-FAMILY claim is not publishable "
        "(gridbook docs/audits/ultraplan_perf_2026-08-01.md §6 P5a)."
    )


def verdict_from_unit_kls(unit_kls: Mapping[str, Mapping]) -> dict:
    """Convenience: records + verdict straight from ``unit_kls``."""
    return cross_family_holdout_verdict(
        ladder_records_from_unit_kls(unit_kls))


def cross_family_verdict_from_cost_payload(payload: Mapping) -> dict | None:
    """Find a stamped verdict in a cost pickle, wherever the lane put it.

    The expert lane writes it under ``provenance.expert_empirical_cost``; a
    merged hybrid payload keeps that nesting, and a standalone run may carry
    it at the provenance root. Returns None when the run predates the stamp
    or never fitted a ladder.
    """
    if not isinstance(payload, Mapping):
        return None
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    for container in (
        provenance,
        provenance.get("expert_empirical_cost"),
    ):
        if isinstance(container, Mapping):
            verdict = container.get(PROVENANCE_KEY)
            if isinstance(verdict, Mapping):
                return dict(verdict)
    return None
