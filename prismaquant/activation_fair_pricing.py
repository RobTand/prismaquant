"""Per-family activation-fair pricing for weight-only cost rows (P5a).

Gridbook's ultraplan performance audit
(``docs/audits/ultraplan_perf_2026-08-01.md`` §6, "Three cost-model
asymmetries") names two defects in this producer's cost precedence:

1. **W4A4 vs W8A8 activation cost is priced only on the measured
   ``output_mse`` branch.** Packed experts (every production script sets
   ``PRISMAQUANT_EXPERT_COST_SAMPLE=16``, and ``measure_quant_cost`` writes
   ``output_mse_measured=False`` for every row of such a run) and
   ladder-interpolated rungs (``CB_LADDER_INTERP=1``; both the dense
   ``_ladder_metric_fit`` fills and the expert unit-KL rows are weight-only)
   fall through to the ``predicted_dloss``/``weight_mse`` branches, where the
   activation-contract difference is *structurally invisible*. NVFP4-CB gets
   credit for its cheaper index stream with none of its A-side cost, on most
   rows of a production run.
2. **The DP compares mixed estimators as if identical.** Anchors are
   activation-aware, interpolated rungs are weight-only, and nothing puts the
   two on one scale.

This module owns the fix's *math and policy*; ``allocator_candidates`` owns
the *extraction* (it is the module that knows the precedence chain) and calls
in. The dependency is deliberately one-directional — ``allocator_candidates``
imports this module, never the reverse — so there is no import cycle and the
calibration stays unit-testable without a cost pickle.

Functional form: a **per-family multiplicative factor, fit as the geometric
mean of the per-row estimator ratio**

    penalty(family) = exp( mean_i ln( d_measured_i / d_weight_only_i ) )

over every (Linear, format) row of that family where BOTH estimators exist.
``d_measured`` is the activation-inclusive price the ``output_mse`` branch
produces (``measure_quant_cost`` applies
``activation_quantize_dequantize(X)`` before measuring it, so it contains the
A side); ``d_weight_only`` is what the ``predicted_dloss``/``weight_mse``
branch would have priced the same row at. Four reasons for that form, in the
order they bind:

* **Transferability.** The correction must move from *measured dense rows* to
  *unmeasured packed-expert rows and interpolated rungs* — different shapes,
  different ``h_trace``, Δloss values orders of magnitude apart. A
  dimensionless ratio is the only form that transfers without a shared
  normalizer; an additive nats offset calibrated on attention rows is
  meaningless on an expert row.
* **It matches the estimator's own algebra.** Both branches are
  ``½·h_trace·MSE`` over the SAME ``h_trace`` (``allocator_solver.predicted_dloss``),
  so the ratio is exactly an MSE-space inflation factor — which is how the A
  side enters the Fisher expansion in the first place.
* **Heavy tails ⇒ geometric mean.** Δloss spans decades across layers; an
  arithmetic mean of ratios is set by a handful of high-``h_trace`` rows,
  while the correction is applied to all of them. The geometric mean is the
  MLE of a multiplicative factor under log-normal residuals, and it is the
  same log-space least squares the retired codebook lane's ladder law used
  (archived 2026-09-25, #1304).
* **It cannot corrupt what is already validated** — *provided every rung of
  the family takes the same pricing branch* (see the correction below). A
  per-family constant applied uniformly cannot reorder rungs *inside* a
  family, so the holdout-gated ladder shape survives untouched; it re-levels
  families against each other, which is exactly the audit's complaint. And
  because it is multiplicative it cannot turn a 0.0 price into a nonzero one,
  so ``allocator_candidates.cost_entry_prices_unmeasured_activation_at_zero``
  — the existing candidate-removal gate — keeps precisely its current
  strength.

**Correction (2026-08-07), and the reason ``BRANCH_INTERPOLATED_OUTPUT``
exists.** That last bullet's premise did not hold as shipped. A family whose
rungs are priced on a MIX of branches — some from a measured ``output_mse``,
some weight-only — has the constant applied to only some of its rungs, and a
constant applied non-uniformly *does* reorder rungs inside the family. On
DSv4-Flash's ``nvfp4_cb`` menu the band-interpolated rungs (K13/K14/K17, stamped
``output_mse_measured=False``) fell to the weight-only branch while their
measured neighbours (K12/K15/K16/K18) did not, so the ladder was priced on two
scales at once. The applied constant is the family geometric mean — 2^6.81 =
112.5 on the bank it was diagnosed against — over a family whose true
``output_mse/weight_mse`` ratio runs 187–320 on gate/up_proj but only 9.4–22 on
down_proj: a ~20x spread inside one family, so no constant can serve it. That
over-priced down_proj rungs ~12x (the higher rung cost MORE than the lower one
on 85% of experts, so it could never be selected) and under-priced gate/up_proj
rungs ~1.6x. The menu was wrong in both directions at once, direction depending
on the projection.

The fix is not a better constant: a band-interpolated row is priced from its
OWN banked ``output_mse``, which is the tensor's holdout-gated ladder
interpolation *in output space* and therefore strictly better information than
any family aggregate. That restores one basis per family. Measured on the 32
banked DSv4-Flash layers, K13/K14/K17 against their neighbours: rung-order
violations fall from 84.8%/0%/84.6% (down_proj) and 2.5%/0%/3.5% (gate/up_proj)
to 0.014%/0.014%/0.043% on down_proj and exactly zero elsewhere, and the
price-vs-neighbour-geometric-mean ratio moves from 2.39x/2.34x/5.75x and
0.67x/0.65x/0.44x to within 1.5% of unity on every projection. The 5 residual
inversions in 63,459 comparisons are adjacent-rung ties of 0.02–0.53%, inside
the interpolator's own validated holdout error, which is the regime
``allocator_candidates.drop_interpolated_candidates_dominated_by_measured``
already owns. It does not make the row a measurement: the row keeps
``cost_source: band_interpolated`` and ``output_mse_measured: false``, it is
still barred from the calibration sample below (its ``output_mse`` is derived,
so admitting it would be circular), and it is stamped
``BRANCH_INTERPOLATED_OUTPUT`` rather than ``BRANCH_MEASURED``. (That ladder's
``band_interpolated``/``mixed`` stamps are retired with the codebook lane and
now refused by ``schemas.refuse_retired_ladder_cost_source`` (#1345); the live
producer of interpolated rows is the Tessera campaign, whose rows carry
``cost_source: tessera_campaign_interpolated``.) The general
lesson is recorded here rather than in the NVFP4 code: ANY family that mixes
the measured and weight-only branches and has a heterogeneous ratio has this
defect. ``fp8_cb`` escaped it only because its interpolated rows happen to be
stamped ``output_mse_measured=True``, so its whole ladder took one branch.

What the factor actually measures, stated honestly: it is an
**estimator-transfer calibration**, not an isolated A-side term. The measured
branch differs from the weight-only branch by (a) the activation contract and
(b) output-space vs weight-space error propagation. Both differences are
family-structural — a W4A4 family and a W8A8 family differ in (a) by
construction — and both are exactly what makes the DP's mixed-estimator
comparison unsound. Naming it an activation penalty is the audit's framing;
naming it an estimator transfer is the mechanism.

Known, recorded bias: the A-side error is rung-INDEPENDENT while the W-side
shrinks with k, so the true ratio grows along a CB ladder. A per-family
constant therefore under-corrects the top rungs and over-corrects the bottom
ones *within* a family. That bias does not touch the cross-family verdict
this item exists to fix, and it is not hidden: ``FamilyCalibration`` records
the per-format mean log2-ratio and the spread across formats
(``rung_dependence_log2_range``), so a family whose ratio is strongly
rung-dependent is visible in the artifact. Making the correction rung-wise
would need a measured sample on the very rows that are unmeasured — the gap
being patched.

Kill switch: ``PRISMAQUANT_ACTIVATION_FAIR_PRICING=0`` restores pre-P5a
pricing bit-for-bit (and suppresses the fail-closed refusal below).
Documented in ``docs/design/runtime_flags.md`` §1.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

SCHEMA = "prismaquant.activation_fair_pricing.v1"

ENV_FLAG = "PRISMAQUANT_ACTIVATION_FAIR_PRICING"

# A single row gives a point estimate with NO residual band, so the fit cannot
# be audited — and an auditable fit is the whole point (the correction moves
# every weight-only-priced row of a family). Two is the smallest sample with a
# spread; the recorded stderr is what an operator judges the fit by.
MIN_CALIBRATION_ROWS = 2

# Bounded provenance sample. The full row list is attested by digest instead,
# so the artifact stays small without becoming unverifiable.
_PROVENANCE_SAMPLE_ROWS = 12

# Branch labels stamped on every candidate. "Which estimator priced this row"
# must be recoverable from the artifact, not inferred from the code version
# that produced it (the same rule selection.json's ``ratchet_objective``
# follows).
BRANCH_BIT_EXACT = "bit_exact"
# A byte-verbatim SOURCE passthrough (allocator_candidates
# .SOURCE_PASSTHROUGH_FORMATS): the exporter copies the checkpoint's own bytes,
# so there is no re-encode to price and no encoder ran. Distinct from
# ``bit_exact``, which is a MEASURED zero produced by actually re-encoding and
# finding the result identical — both are free, and the artifact should be able
# to say which one it got.
BRANCH_SOURCE_PASSTHROUGH = "source_passthrough"
BRANCH_MEASURED = "measured_output_mse"
# Priced from the row's OWN ``output_mse``, which for this row is the tensor's
# holdout-gated ladder interpolation rather than a measurement
# (``cost_source: tessera_campaign_interpolated``; ``output_mse_measured:
# false``). It shares the output-space branch with ``BRANCH_MEASURED`` because
# the number is already in output space and already carries the activation
# contract — the anchors it interpolates between were measured with
# ``activation_quantize_dequantize(X)`` applied — so the per-family penalty
# must NOT be layered on top of it. It gets a label of its own because it is a
# PREDICTION: ``cost_entry_is_band_interpolated`` stays the provenance guard,
# and a shipped artifact must still be able to say which of its selected prices
# were predicted. Distinct from ``BRANCH_CALIBRATED``, which is a WEIGHT-space
# number converted to the output scale by a family constant; this is the row's
# own output-space number, which is strictly better information.
BRANCH_INTERPOLATED_OUTPUT = "interpolated_output_mse"
BRANCH_ACTIVATION_IDENTITY = "activation_identity"
BRANCH_CALIBRATED = "weight_only_activation_calibrated"
BRANCH_UNCALIBRATED = "weight_only_uncalibrated"
BRANCH_KILL_SWITCH = "weight_only_kill_switch"

# Reason codes for the run-level verdict.
REASON_CALIBRATED = "calibrated"
REASON_KILL_SWITCH = "env_kill_switch"
REASON_NO_MEASURED_ROWS = "no_measured_activation_rows"
REASON_NOTHING_TO_CORRECT = "no_weight_only_activation_rows"

# Marker written onto AGGREGATED (fused-sibling / packed-serving-group) cost
# entries whose ``predicted_dloss`` already has the family penalty folded in.
# Without it a super entry — which carries a plain ``predicted_dloss`` and no
# ``cost_source`` — is indistinguishable from a raw weight-only row and would
# be penalized twice by anything that re-prices it.
APPLIED_MARKER_KEY = "activation_pricing_applied"


def env_enabled() -> bool:
    """``PRISMAQUANT_ACTIVATION_FAIR_PRICING``: on unless explicitly disabled.

    Off-by-default is the wrong default here — the audit's finding is that
    *production runs are currently mispriced*, so a correction nobody enables
    fixes nothing. The switch exists to reproduce a pre-P5a artifact exactly,
    which is a real need (bisecting an allocation change), not to hedge the
    decision.
    """
    value = os.environ.get(ENV_FLAG)
    if value is None:
        return True
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


@dataclass(frozen=True)
class CalibrationRow:
    """One (Linear, format) pair that carries BOTH estimators.

    ``measured_dloss`` is the activation-inclusive price (the ``output_mse``
    branch); ``weight_only_dloss`` is what the same row would be priced at by
    the ``predicted_dloss``/``weight_mse`` branch. Both are computed at
    ``gain=1.0``: a calibrated gain multiplies the two identically, so the
    ratio is gain-invariant and the calibration does not have to be redone
    when ``--calibration`` changes.
    """

    qname: str
    fmt: str
    family: str
    measured_dloss: float
    weight_only_dloss: float

    @property
    def log2_ratio(self) -> float:
        return math.log2(self.measured_dloss / self.weight_only_dloss)


@dataclass(frozen=True)
class FamilyCalibration:
    """The fit for one format family, with its residuals."""

    family: str
    n_rows: int
    penalty: float
    log2_penalty: float
    log2_stdev: float
    log2_stderr: float
    log2_residual_min: float
    log2_residual_max: float
    formats: tuple[str, ...]
    per_format_log2_penalty: tuple[tuple[str, float, int], ...]
    rung_dependence_log2_range: float
    sample: tuple[tuple[str, str], ...]
    rows_digest: str

    def as_dict(self) -> dict:
        return {
            "family": self.family,
            "n_rows": int(self.n_rows),
            "penalty": float(self.penalty),
            "log2_penalty": float(self.log2_penalty),
            "log2_stdev": float(self.log2_stdev),
            "log2_stderr": float(self.log2_stderr),
            "log2_residual_min": float(self.log2_residual_min),
            "log2_residual_max": float(self.log2_residual_max),
            "formats": list(self.formats),
            "per_format_log2_penalty": [
                {"format": fmt, "log2_penalty": float(value), "n_rows": int(n)}
                for fmt, value, n in self.per_format_log2_penalty
            ],
            "rung_dependence_log2_range": float(
                self.rung_dependence_log2_range),
            "calibration_sample": [
                {"qname": qname, "format": fmt} for qname, fmt in self.sample
            ],
            "calibration_sample_truncated_to": _PROVENANCE_SAMPLE_ROWS,
            "calibration_rows_sha256": self.rows_digest,
        }


@dataclass(frozen=True)
class ActivationFairPricing:
    """Run-level verdict: which families were calibrated, and from what.

    Immutable and deterministic — the same (stats, costs, menu) always yields
    the same factors, in the same order, with the same digest.
    """

    enabled: bool
    reason: str
    families: Mapping[str, FamilyCalibration]
    measured_rows_by_family: Mapping[str, int]
    weight_only_rows_by_family: Mapping[str, int]
    uncalibrated_families: tuple[str, ...]

    def penalty_for(self, format_name: str | None,
                    act_quant_changes_input: bool) -> tuple[float, str]:
        """Return ``(multiplier, branch_label)`` for a weight-only-priced row.

        Only ever consulted on the weight-only branches; the measured and
        bit-exact branches are decided before this is reached.
        """
        if not act_quant_changes_input:
            return 1.0, BRANCH_ACTIVATION_IDENTITY
        if self.reason == REASON_KILL_SWITCH:
            # Operator asked for pre-P5a pricing; say so rather than blame a
            # missing sample the run never looked for.
            return 1.0, BRANCH_KILL_SWITCH
        family = _family_of(format_name)
        fit = self.families.get(family) if family else None
        if fit is None:
            return 1.0, BRANCH_UNCALIBRATED
        return float(fit.penalty), BRANCH_CALIBRATED

    def as_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "env_flag": ENV_FLAG,
            "enabled": bool(self.enabled),
            "reason": self.reason,
            "min_calibration_rows": MIN_CALIBRATION_ROWS,
            "functional_form": (
                "per_family_multiplicative__geometric_mean_of_"
                "measured_over_weight_only_dloss"
            ),
            "applied_to": (
                "weight_only_priced_rows_of_activation_quantizing_formats__"
                "predicted_dloss_and_weight_mse_branches"
            ),
            "families": {
                family: fit.as_dict()
                for family, fit in sorted(self.families.items())
            },
            "uncalibrated_families": list(self.uncalibrated_families),
            "measured_rows_by_family": {
                family: int(count)
                for family, count in sorted(self.measured_rows_by_family.items())
            },
            "weight_only_rows_by_family": {
                family: int(count)
                for family, count
                in sorted(self.weight_only_rows_by_family.items())
            },
        }


DISABLED = ActivationFairPricing(
    enabled=False,
    reason=REASON_KILL_SWITCH,
    families={},
    measured_rows_by_family={},
    weight_only_rows_by_family={},
    uncalibrated_families=(),
)


def _family_of(format_name: str | None) -> str | None:
    if not format_name:
        return None
    from . import format_registry as fr

    try:
        return str(fr.get_format(str(format_name)).family)
    except KeyError:
        return None


def _rows_digest(rows: Sequence[CalibrationRow]) -> str:
    """SHA-256 over the FULL calibration row list.

    The artifact carries a bounded sample for readability; the digest makes
    the untruncated sample reproducible, so "which rows was this fit from"
    stays an answerable question on a 295B-class run.
    """
    payload = json.dumps(
        [
            [row.qname, row.fmt, repr(row.measured_dloss),
             repr(row.weight_only_dloss)]
            for row in rows
        ],
        separators=(",", ":"),
        sort_keys=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fit_family(family: str,
                rows: Sequence[CalibrationRow]) -> FamilyCalibration:
    """Geometric-mean fit plus its residual band, for one family."""
    ordered = sorted(rows, key=lambda row: (row.qname, row.fmt))
    log2_ratios = [row.log2_ratio for row in ordered]
    n = len(log2_ratios)
    mean = sum(log2_ratios) / n
    residuals = [value - mean for value in log2_ratios]
    if n >= 2:
        var = sum(r * r for r in residuals) / (n - 1)
        stdev = math.sqrt(var)
    else:
        stdev = 0.0
    per_format: dict[str, list[float]] = {}
    for row in ordered:
        per_format.setdefault(row.fmt, []).append(row.log2_ratio)
    per_format_means = tuple(
        (fmt, sum(values) / len(values), len(values))
        for fmt, values in sorted(per_format.items())
    )
    fmt_means = [value for _fmt, value, _n in per_format_means]
    return FamilyCalibration(
        family=family,
        n_rows=n,
        penalty=2.0 ** mean,
        log2_penalty=mean,
        log2_stdev=stdev,
        log2_stderr=stdev / math.sqrt(n) if n > 0 else 0.0,
        log2_residual_min=min(residuals),
        log2_residual_max=max(residuals),
        formats=tuple(sorted(per_format)),
        per_format_log2_penalty=per_format_means,
        rung_dependence_log2_range=(
            max(fmt_means) - min(fmt_means) if fmt_means else 0.0),
        sample=tuple(
            (row.qname, row.fmt)
            for row in ordered[:_PROVENANCE_SAMPLE_ROWS]
        ),
        rows_digest=_rows_digest(ordered),
    )


def calibrate(
    rows: Iterable[CalibrationRow],
    *,
    measured_rows_by_family: Mapping[str, int],
    weight_only_rows_by_family: Mapping[str, int],
    enabled: bool | None = None,
) -> ActivationFairPricing:
    """Fit every family that has a usable sample; refuse a mixed estimator.

    ``measured_rows_by_family`` / ``weight_only_rows_by_family`` count the
    ACTIVATION-QUANTIZING rows the run actually priced on each branch (the
    caller extracts them alongside ``rows``); they are what the fail-closed
    policy below is decided on, and they are stamped whether or not a fit
    happened.

    Fail-closed policy (audit P5a: "if a family has activation-changing
    formats but NO measured rows to calibrate from, do not silently pass
    weight-only prices through as before"):

    * **Refuse** (``AssertionError`` naming the env var) when the DP would be
      handed a MIXED scale — at least one family calibrated while another
      activation-quantizing family still has weight-only-priced rows and no
      fit of its own. That is the audit's asymmetry made worse, not better:
      correcting a W4A4 family while leaving a W8A8 one uncorrected tilts the
      comparison this work exists to make fair.
    * **Pass through, loudly**, when NO activation-quantizing family has a
      measured sample. Nothing can be corrected, no asymmetry is introduced,
      and the pre-existing gates — including the
      ``ACTIVATION_COST_UNMEASURED_REASON`` candidate removal, which this
      change deliberately leaves at full strength — still apply. The verdict
      is recorded (``reason=no_measured_activation_rows``) and the allocator
      prints it, so it is never silent; refusing here instead would make
      currently-legal research runs illegal for no gain in fairness.
    """
    if enabled is None:
        enabled = env_enabled()
    if not enabled:
        return ActivationFairPricing(
            enabled=False,
            reason=REASON_KILL_SWITCH,
            families={},
            measured_rows_by_family=dict(measured_rows_by_family),
            weight_only_rows_by_family=dict(weight_only_rows_by_family),
            uncalibrated_families=(),
        )

    by_family: dict[str, list[CalibrationRow]] = {}
    for row in rows:
        by_family.setdefault(row.family, []).append(row)
    families = {
        family: _fit_family(family, family_rows)
        for family, family_rows in sorted(by_family.items())
        if len(family_rows) >= MIN_CALIBRATION_ROWS
    }
    uncalibrated = tuple(sorted(
        family
        for family, count in weight_only_rows_by_family.items()
        if count > 0 and family not in families
    ))

    if families and uncalibrated:
        raise AssertionError(
            "activation-fair pricing would hand the DP a MIXED cost scale: "
            f"{sorted(families)} calibrated from measured output_mse rows, "
            f"while {list(uncalibrated)} still price "
            + ", ".join(
                f"{weight_only_rows_by_family[family]} {family}"
                for family in uncalibrated
            )
            + " activation-quantizing row(s) weight-only with no calibration "
            "sample of their own.\n"
            "    calibrated:   "
            + "; ".join(
                f"{family} penalty={fit.penalty:.4g} "
                f"(n={fit.n_rows}, log2 sd={fit.log2_stdev:.3f})"
                for family, fit in sorted(families.items())
            )
            + "\n    uncalibrated: "
            + "; ".join(
                f"{family} measured_rows="
                f"{measured_rows_by_family.get(family, 0)} "
                f"weight_only_rows={weight_only_rows_by_family[family]}"
                for family in uncalibrated
            )
            + "\nCorrecting one activation contract and not the other tilts "
            "exactly the W4A4-vs-W8A8 comparison this calibration exists "
            "to make fair (gridbook docs/audits/ultraplan_perf_2026-08-01.md "
            "§6, asymmetries 1 and 2), so the run refuses rather than ship a "
            "silently biased allocation.\n"
            "Close the measurement gap for the uncalibrated family — unset "
            "PRISMAQUANT_EXPERT_COST_SAMPLE (and make the expert activation "
            "cache available) so measure_quant_cost records output_mse with "
            "activation_quantize_dequantize applied, or measure at least "
            f"{MIN_CALIBRATION_ROWS} rows of that family — or set "
            f"{ENV_FLAG}=0 to restore pre-P5a weight-only pricing for the "
            "whole run (both families uncorrected, which is at least "
            "symmetric)."
        )

    if not families:
        reason = (
            REASON_NOTHING_TO_CORRECT
            if not any(weight_only_rows_by_family.values())
            else REASON_NO_MEASURED_ROWS
        )
        return ActivationFairPricing(
            enabled=False,
            reason=reason,
            families={},
            measured_rows_by_family=dict(measured_rows_by_family),
            weight_only_rows_by_family=dict(weight_only_rows_by_family),
            uncalibrated_families=uncalibrated,
        )

    return ActivationFairPricing(
        enabled=True,
        reason=REASON_CALIBRATED,
        families=families,
        measured_rows_by_family=dict(measured_rows_by_family),
        weight_only_rows_by_family=dict(weight_only_rows_by_family),
        uncalibrated_families=uncalibrated,
    )
