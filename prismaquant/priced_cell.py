"""Cost-row evidence, objective and activation-price interpretation."""
from __future__ import annotations

from dataclasses import dataclass
import math
import operator
import os
from typing import Literal
from . import format_registry as fr
from .source_format_contracts import SOURCE_PASSTHROUGH_FORMATS
from .allocator_solver import predicted_dloss
from .schemas import refuse_retired_ladder_cost_source
from .activation_fair_pricing import (APPLIED_MARKER_KEY, ActivationFairPricing,
    BRANCH_ACTIVATION_IDENTITY, BRANCH_BIT_EXACT, BRANCH_CALIBRATED,
    BRANCH_INTERPOLATED_OUTPUT, BRANCH_MEASURED, BRANCH_SOURCE_PASSTHROUGH, BRANCH_UNCALIBRATED)


@dataclass(frozen=True)
class CostPricingPolicy:
    ucb_z: float
    fisher_output_mse: bool

    @classmethod
    def from_environment(cls):
        return cls(_cost_ucb_z(), _fisher_output_mse_allocator_enabled())


@dataclass(frozen=True)
class PricedCell:
    predicted_dloss: float
    source: str
    activation_branch: str
    unmeasured_activation_at_zero: bool

    @property
    def candidate_dloss(self) -> float:
        return max(self.predicted_dloss, 0.0)


@dataclass(frozen=True)
class _FixedPrice:
    value: float
    activation_branch: str


@dataclass(frozen=True)
class _JointPrice:
    activation_branch: str = "joint_aura"


@dataclass(frozen=True)
class _OutputPrice:
    activation_branch: str


@dataclass(frozen=True)
class _WeightPrice:
    kind: Literal["aggregate", "anchored", "transferred"]
    activation_branch: str
    penalty: float = 1.0


_PriceBasis = _FixedPrice | _JointPrice | _OutputPrice | _WeightPrice


def _price_basis(stats_entry: dict, cost_entry: dict, format_name: str | None,
                 activation_pricing: ActivationFairPricing | None) -> _PriceBasis:
    from .rung_allowability import complete_scientific_price
    complete = complete_scientific_price(cost_entry, format_name=format_name)
    if complete is not None:
        branch = ("canonical_qualified_chord" if cost_entry.get("canonical_quality") is not None
                  else "qualified_served_kl")
        return _FixedPrice(complete, branch)
    if cost_entry_is_joint_aura(cost_entry):
        return _JointPrice()
    if cost_entry_is_source_passthrough(cost_entry, format_name):
        return _FixedPrice(0.0, BRANCH_SOURCE_PASSTHROUGH)
    if cost_entry_is_bit_exact(cost_entry, format_name):
        return _FixedPrice(0.0, BRANCH_BIT_EXACT)
    if _has_measured_output_mse(stats_entry, cost_entry):
        return _OutputPrice(BRANCH_MEASURED)
    if _has_interpolated_output_mse(cost_entry):
        return _OutputPrice(BRANCH_INTERPOLATED_OUTPUT)
    if cost_entry.get(APPLIED_MARKER_KEY) is True:
        return _WeightPrice("aggregate", BRANCH_CALIBRATED)
    if cost_entry_is_anchored_aura_supersurrogate(cost_entry):
        return _WeightPrice("anchored", ANCHORED_AURA_BRANCH)
    act_changes = _format_act_quant_changes_input(format_name)
    if activation_pricing is None:
        branch = BRANCH_ACTIVATION_IDENTITY if not act_changes else BRANCH_UNCALIBRATED
        return _WeightPrice("transferred", branch)
    penalty, branch = activation_pricing.penalty_for(format_name, act_changes)
    return _WeightPrice("transferred", branch, penalty)


def _basis_dloss(basis: _PriceBasis, stats_entry: dict, cost_entry: dict, *,
                 gain: float, format_name: str | None, policy: CostPricingPolicy) -> float:
    if isinstance(basis, _FixedPrice):
        return basis.value
    if isinstance(basis, _JointPrice):
        if float(gain) != 1.0 or cost_entry.get(APPLIED_MARKER_KEY) is True:
            raise ValueError("joint AURA refuses calibrated gain or a second activation transfer")
        if (format_name is not None and
                cost_entry["joint_operator_identity"]["format"] != format_name):
            raise ValueError("joint AURA row cannot price a different format")
        return cost_entry_weight_only_dloss(stats_entry, cost_entry, gain=1.0, policy=policy)
    if isinstance(basis, _OutputPrice):
        return cost_entry_measured_activation_dloss(stats_entry, cost_entry, gain=gain, policy=policy)
    base = cost_entry_weight_only_dloss(stats_entry, cost_entry, gain=gain, policy=policy)
    if basis.kind == "aggregate":
        return base
    if basis.kind == "anchored":
        return base + cost_entry_act_dloss(cost_entry)
    return (base + cost_entry_act_dloss(cost_entry)) * basis.penalty


def _unknown_activation_zero(basis: _PriceBasis, stats_entry: dict, cost_entry: dict,
                             priced_dloss: float, format_name: str | None) -> bool:
    if isinstance(basis, (_FixedPrice, _JointPrice)):
        return False
    if cost_entry_is_anchored_aura_supersurrogate(cost_entry):
        return False
    if isinstance(basis, _OutputPrice) and basis.activation_branch == BRANCH_MEASURED:
        return False
    if not _format_act_quant_changes_input(format_name):
        return False
    try:
        if float(priced_dloss) != 0.0:
            return False
    except (TypeError, ValueError):
        return False
    try:
        h_trace = float(stats_entry.get("h_trace", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return h_trace > 0.0


def price_cell(stats_entry: dict, cost_entry: dict, *, format_name: str | None,
               policy: CostPricingPolicy, gain: float = 1.0,
               activation_pricing: ActivationFairPricing | None = None) -> PricedCell:
    basis = _price_basis(stats_entry, cost_entry, format_name, activation_pricing)
    value = _basis_dloss(basis, stats_entry, cost_entry,
        gain=gain, format_name=format_name, policy=policy)
    return PricedCell(value,
        cost_entry_source(stats_entry, cost_entry, format_name, policy=policy),
        basis.activation_branch,
        _unknown_activation_zero(basis, stats_entry, cost_entry, max(value, 0.0), format_name))


SOURCE_PASSTHROUGH_COST_SOURCE = "source_passthrough"

ANCHORED_AURA_COST_CURRENCY = "aura_predicted_dloss"

ANCHORED_AURA_COST_SOURCE = "production_arm_render"

ANCHORED_AURA_BRANCH = "anchored_aura_extrapolation"

def _stats_indicates_packed_expert(stats_entry: dict) -> bool:
    """True for probe entries representing a 3D packed-expert tensor."""
    return bool(
        stats_entry.get("_packed_experts_module")
        or stats_entry.get("_packed_param")
        or int(stats_entry.get("num_experts", 0) or 0) > 0
    )

def _has_measured_output_mse(stats_entry: dict, cost_entry: dict) -> bool:
    """Whether ``output_mse`` is a real joint-output MEASUREMENT.

    Packed experts historically stored ``output_mse=0.0`` as a placeholder
    because the routed expert forward was not reconstructed offline. That
    placeholder must not outrank the scalar predicted_dloss / weight_mse path.

    This answers the PROVENANCE question — "was this number observed?" — and
    is deliberately NOT the question pricing asks. See
    ``_prices_from_output_mse``.
    """
    if "output_mse" not in cost_entry:
        return False
    if cost_entry.get("output_mse_measured") is False:
        return False
    if (_stats_indicates_packed_expert(stats_entry)
            and float(cost_entry.get("output_mse", 0.0)) == 0.0
            and ("predicted_dloss" in cost_entry or "weight_mse" in cost_entry)):
        return False
    return True

def _has_interpolated_output_mse(cost_entry: dict) -> bool:
    """Whether this row carries a USABLE interpolated ``output_mse``.

    An interpolated row (``cost_entry_is_band_interpolated``) is stamped
    ``output_mse_measured=False`` and its ``output_mse`` is a fit over the
    unit's own measured anchors rather than an observation. It is still an
    output-space number, which is what makes it usable as a price.

    Strict positivity is load-bearing, not a defensive nicety: a row at
    exactly 0.0 carries no output-space information, and a 0.0 price would be
    the DP's global optimum — precisely what
    ``cost_entry_prices_unmeasured_activation_at_zero`` exists to catch. A
    non-finite value is rejected for the same reason: it is not a price.
    """
    if not cost_entry_is_band_interpolated(cost_entry):
        return False
    try:
        value = float(cost_entry.get("output_mse", 0.0))
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0

def _prices_from_output_mse(stats_entry: dict, cost_entry: dict) -> bool:
    """Whether PRICING should read this row's ``output_mse``.

    Deliberately a different question from ``_has_measured_output_mse``, and
    the two must not be collapsed back together:

      * *"Is this ``output_mse`` an observation?"* decides whether the row may
        enter the P5a calibration SAMPLE
        (``collect_activation_calibration_rows``). A band-interpolated row's
        output_mse is derived from the family's own measured anchors, so
        admitting it would fit the transfer constant partly on its own output
        — circular. It stays out.
      * *"Is this ``output_mse`` the best price available for this row?"*
        decides which branch prices it. Here the answer for a band-interpolated
        row is yes: its value is the tensor's OWN holdout-gated interpolation
        in output space, while the alternative is its weight_mse scaled by a
        family-wide geometric-mean constant. The row's own output-space number
        beats a family aggregate — and, decisively, mixing the two bases
        WITHIN one family reorders that family's rungs, which is the failure
        ``activation_fair_pricing``'s design explicitly ruled out and which
        shipped anyway (see that module's 2026-08-07 correction: NVFP4-CB
        K13/K14/K17 priced 12x high on down_proj and 1.6x low on gate/up_proj,
        because the family constant 112.5 spans a per-projection ratio range
        of 9.4-320).

    Nothing about the row's own claims changes: ``cost_source`` and
    ``output_mse_measured`` keep their values and their meanings, the
    candidate is stamped ``BRANCH_INTERPOLATED_OUTPUT`` rather than
    ``BRANCH_MEASURED``, and ``cost_entry_is_band_interpolated`` /
    ``drop_interpolated_candidates_dominated_by_measured`` keep full strength
    as the provenance and noise-band guards. Only the number's USE changes.

    Not gated on ``act_quant_changes_input``: the measured branch is not
    gated either, and gating this one would re-create exactly the same
    measured/weight-only basis mix inside an identity-activation family (at
    penalty 1.0, but weight-space vs output-space all the same). No such row
    exists: the RD-ladder sites that stamped ``band_interpolated`` served the
    retired codebook lane (archived 2026-09-25, #1304), whose formats all
    quantized activations — so this is a statement about the rule, not a live
    code path.
    """
    return (
        _has_measured_output_mse(stats_entry, cost_entry)
        or _has_interpolated_output_mse(cost_entry)
    )

def cost_entry_is_bit_exact(
    cost_entry: dict,
    format_name: str | None = None,
) -> bool:
    """Whether this entry proves a LOSSLESS re-encode end to end: measured
    ``weight_mse`` of exactly 0.0 AND a format whose activation path is the
    identity.

    ``weight_mse`` is a mean of squared per-element deltas: it is exactly
    zero only when the format stores the source weights verbatim (W' == W)
    — e.g. MXFP8 over an FP8 128-block source, or MXFP4/MXFP8 over
    an MXFP4-packed QAT source. But W' == W only silences the WEIGHT side.
    For W·A· formats (``FormatSpec.act_quant_changes_input`` — NVFP4,
    FP8 dynamic, the MX family, GGUF Q8_1 compute) the cost pipeline
    applies ``activation_quantize_dequantize(X)`` before measuring
    ``output_mse`` (measure_quant_cost), so a weight-lossless entry's
    output_mse is REAL A-side error, not noise — on an MXFP4-packed
    source, an MXFP4 re-encode priced from weight_mse alone would cost
    dloss 0.0, the unbeatable global minimum at any budget, while its
    served activations are still 4-bit. The short-circuit therefore
    requires the format's activation quantization to be the identity — a
    dtype-level fact (``FormatSpec.act_quant_changes_input``, i.e. ``act_bits``
    absent or >= 16: BF16, FP8_SOURCE, NVFP4A16, MXFP8A16, INT-W·A16), not a
    heuristic. Formats we cannot identify
    (``format_name`` None or unregistered) never short-circuit.

    For qualifying passthrough-activation formats, measured zero is a
    valid, indeed optimal, cost (see ``_log_error_values`` in
    allocator.py): the entry short-circuits to predicted dloss 0.0 ahead
    of any noisy output_mse measurement.

    Entries that declare an explicit ``cost_source`` (e.g. the
    production-render score pipeline) carry their own authoritative
    pricing and default ``weight_mse`` to 0.0 as a placeholder, not a
    measurement — they are never treated as bit-exact, matching the
    precedence ``cost_entry_source`` already gives the explicit source.
    """
    explicit = cost_entry.get("cost_source")
    if isinstance(explicit, str) and explicit:
        return False
    if format_name is None:
        return False
    try:
        spec = fr.get_format(str(format_name))
    except KeyError:
        return False
    if spec.act_quant_changes_input:
        return False
    weight_mse = cost_entry.get("weight_mse")
    try:
        return weight_mse is not None and float(weight_mse) == 0.0
    except (TypeError, ValueError):
        return False

def cost_entry_is_source_passthrough(
    cost_entry: dict,
    format_name: str | None = None,
) -> bool:
    """Whether this entry ships the SOURCE bytes and is therefore exact.

    The dual of ``cost_entry_is_bit_exact``. That predicate proves exactness
    from a MEASUREMENT (``weight_mse == 0.0`` recorded by a real cost run);
    this one proves it from a CONTRACT — the exporter copies the source slice
    verbatim, so the re-encode error is not small, it does not exist.

    Three independent conditions, all required, so the claim cannot be forged
    by writing a string into a cost table:

      * the entry declares ``cost_source="source_passthrough"``;
      * the format is a declared member of ``SOURCE_PASSTHROUGH_FORMATS``;
      * the format's activation path is the identity
        (``FormatSpec.act_quant_changes_input`` is False) — the same
        dtype-level gate ``cost_entry_is_bit_exact`` applies, because
        shipping the weights verbatim only silences the W side.

    ``cost_entry_is_bit_exact`` deliberately refuses ANY entry carrying an
    explicit ``cost_source``, since those normally mean "an upstream pipeline
    priced this row and defaulted weight_mse to a placeholder 0.0". A
    source-passthrough entry is the one case where the explicit provenance is
    itself the proof, which is why it gets a predicate of its own rather than
    a hole punched in that rule.
    """
    if cost_entry.get("cost_source") != SOURCE_PASSTHROUGH_COST_SOURCE:
        return False
    if format_name is None:
        return False
    canonical = fr.canonical_format_name(str(format_name))
    if canonical not in SOURCE_PASSTHROUGH_FORMATS:
        return False
    try:
        spec = fr.get_format(canonical)
    except KeyError:
        return False
    return not spec.act_quant_changes_input

TESSERA_INTERPOLATED_COST_SOURCE = "tessera_campaign_interpolated"

def cost_entry_is_band_interpolated(cost_entry: dict) -> bool:
    """Whether this row's cost was FITTED from anchors, not measured.

    Stamped by the Tessera anchor campaign. Such a row is not a guess — the
    fit had to clear the campaign's leave-one-anchor-out check — but it IS a
    prediction, and a shipped artifact must be able to say which of its
    selected prices were predictions. A row the retired RD ladder stamped
    (``band_interpolated``/``mixed``, codebook lane archived 2026-09-25,
    #1304) refuses here instead of being priced.
    """
    refuse_retired_ladder_cost_source(cost_entry)
    return cost_entry.get("cost_source") == TESSERA_INTERPOLATED_COST_SOURCE

def cost_entry_is_exact_by_construction(
    cost_entry: dict,
    format_name: str | None = None,
) -> bool:
    """Whether this row's Δloss is exactly 0.0 with no measurement involved.

    The union of the two ways a row can be free: a measured lossless
    re-encode (``cost_entry_is_bit_exact``) and a byte-verbatim source
    passthrough (``cost_entry_is_source_passthrough``). Pricing, the measured
    branch test and the P5a branch label all key off THIS predicate so the two
    cannot drift apart; ``cost_entry_source`` still reports which of the two
    it was.
    """
    return (
        cost_entry_is_bit_exact(cost_entry, format_name)
        or cost_entry_is_source_passthrough(cost_entry, format_name)
    )

def synthesized_source_passthrough_cost_entry(format_name: str) -> dict:
    """The cost row for a passthrough candidate the cost table cannot hold.

    Not a placeholder and not an estimate: ``predicted_dloss`` is 0.0 because
    the exporter ships the source slice unchanged, and the explicit
    ``cost_source`` records that provenance so no downstream reader mistakes
    the zero for an unmeasured activation cost (the failure mode
    ``cost_entry_prices_unmeasured_activation_at_zero`` exists to catch).
    ``output_mse_measured=False`` states plainly that no output measurement
    was taken — because none was needed.
    """
    return {
        "cost_source": SOURCE_PASSTHROUGH_COST_SOURCE,
        "predicted_dloss": 0.0,
        "weight_mse": 0.0,
        "output_mse": 0.0,
        "output_mse_measured": False,
        "source_passthrough_format": fr.canonical_format_name(
            str(format_name)
        ),
    }

def cost_entry_uses_measured_output_mse(
    stats_entry: dict,
    cost_entry: dict,
    format_name: str | None = None,
) -> bool:
    """Whether ``cost_entry_predicted_dloss`` will read a MEASURED
    ``output_mse``.

    Not the same as "will read ``output_mse``": the band-interpolated branch
    (``_prices_from_output_mse``) reads it too, from a row whose value is a
    ladder fit rather than an observation. This predicate keeps the narrower,
    measurement-only meaning its name states — it is what callers asking "does
    a real output-side observation back this row" want.
    """
    if cost_entry_is_exact_by_construction(cost_entry, format_name):
        return False
    return _has_measured_output_mse(stats_entry, cost_entry)

def cost_entry_source(
    stats_entry: dict,
    cost_entry: dict,
    format_name: str | None = None,
    *, policy: CostPricingPolicy | None = None,
) -> str:
    """Return the named cost source the allocator will use for one row."""
    policy = CostPricingPolicy.from_environment() if policy is None else policy
    explicit = cost_entry.get("cost_source")
    if isinstance(explicit, str) and explicit:
        return explicit
    if cost_entry_is_bit_exact(cost_entry, format_name):
        return "bit_exact"
    if _has_measured_output_mse(stats_entry, cost_entry):
        if (
            policy.fisher_output_mse
            and "fisher_output_mse" in cost_entry
        ):
            return "fisher_output_mse"
        return "output_mse"
    if "predicted_dloss" in cost_entry:
        return "predicted_dloss"
    return "weight_mse"

def _fisher_output_mse_allocator_enabled() -> bool:
    value = os.environ.get("PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR")
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}

def cost_entry_measured_activation_dloss(
    stats_entry: dict,
    cost_entry: dict,
    *,
    gain: float = 1.0,
    policy: CostPricingPolicy | None = None,
) -> float:
    """The ACTIVATION-INCLUSIVE price of one row, or 0.0 when unmeasured.

    Split out of ``cost_entry_predicted_dloss`` so the P5a calibration reads
    exactly the number the measured branch would have priced — one
    implementation of "what does the output_mse branch say", not a second
    copy that can drift from the precedence chain it calibrates against.
    ``measure_quant_cost`` applies ``activation_quantize_dequantize(X)``
    before measuring ``output_mse``, which is why this branch — and only this
    branch — carries the A side.

    Both output-space provenances go through here — a measured row and a
    band-interpolated one (whose fit interpolates between activation-inclusive
    measured anchors, so it carries the A side too) — so gain, Fisher-variant
    selection and the ½·h_trace algebra stay in one place. WHICH rows qualify
    is ``_prices_from_output_mse``'s decision, not this function's; the
    calibration still admits only measured ones.
    """
    policy = CostPricingPolicy.from_environment() if policy is None else policy
    if (
        policy.fisher_output_mse
        and "fisher_output_mse" in cost_entry
    ):
        return predicted_dloss(
            stats_entry["h_trace"],
            float(cost_entry["fisher_output_mse"]),
            gain=gain,
        )
    return predicted_dloss(
        stats_entry["h_trace"],
        float(cost_entry.get("output_mse", 0.0)),
        gain=gain,
    )

def cost_entry_weight_only_dloss(
    stats_entry: dict,
    cost_entry: dict,
    *,
    gain: float = 1.0,
    policy: CostPricingPolicy | None = None,
) -> float:
    """The WEIGHT-ONLY price of one row (``predicted_dloss``/``weight_mse``).

    Uncorrected by any activation calibration — this is the number the
    calibration divides into, and the number the correction multiplies.
    """
    policy = CostPricingPolicy.from_environment() if policy is None else policy
    if "predicted_dloss" in cost_entry:
        base = float(cost_entry["predicted_dloss"])
        # Uncertainty-aware allocation (opt-in): charge z·stderr on top of the
        # point estimate. The knapsack optimizes over noisy estimates, so it
        # systematically harvests lucky draws (winner's curse — the observed
        # ±0.017-KL between-seed allocation lottery). UCB takes an aggressive
        # format choice only when it is CONFIDENTLY cheap: a non-regressive
        # bias whose penalty is derived from the measurement's own sampling
        # noise, not a tuned constant. z=0 (default) is bit-identical to
        # prior behavior.
        z = policy.ucb_z
        if z > 0.0:
            base += z * float(cost_entry.get("predicted_dloss_stderr", 0.0))
        return base * float(gain)
    return predicted_dloss(
        stats_entry["h_trace"],
        float(cost_entry.get("weight_mse", 0.0)),
        gain=gain,
    )

def cost_entry_is_joint_aura(cost_entry: dict) -> bool:
    """Validate any joint claim before interpreting generic cost fields."""
    from .joint_aura import validate_joint_aura_entry
    from .rung_allowability import CANONICAL_CHORD_SOURCE, CANONICAL_SUM_SOURCE, complete_scientific_price
    if (cost_entry.get("cost_source") in (CANONICAL_CHORD_SOURCE, CANONICAL_SUM_SOURCE)
            or "canonical_quality" in cost_entry or "canonical_members" in cost_entry):
        complete_scientific_price(cost_entry)
        return False
    return validate_joint_aura_entry(cost_entry)

def cost_entry_is_anchored_aura_supersurrogate(cost_entry: dict) -> bool:
    """Whether one row was priced by the anchored-AURA campaign.

    Three independent stamps, ALL required, in the same forgery-refusing style
    as ``cost_entry_is_source_passthrough`` — a hand-written table cannot claim
    this provenance by writing one string:

      * ``cost_currency`` is the AURA projection currency, so the number is
        ``0.5*mean_k<gW,dW>^2`` and not an MSE wearing the same field name;
      * ``cost_source`` says a PRODUCTION-ARM render produced the anchor.
        RTN-vs-rendered ``dW`` is result-changing (+36% at fp8), and the
        byte-verbatim terminals carry ``SOURCE_PASSTHROUGH_COST_SOURCE``
        instead, so they are excluded here and handled by the exact branch;
      * ``fisher_application_count == 1`` — the h^2 guard. ``predicted_dloss``
        already contains the KL-Fisher; extrapolation multiplies by a ratio of
        ``g``, never by a sensitivity a second time.

    **The standing limitation this admission accepts.** Anchored rows price
    activation-quantizing rungs (``act_quant_changes_input`` is True for NVFP4
    and FP8, as it was for every rung of the retired codebook lane, archived
    2026-09-25, #1304), while AURA's ``dW`` is weights-only. So these
    prices are activation-quantization-blind, and the P5a penalty cannot fix it
    here: an anchored table carries no measured ``output_mse`` rows at all, so
    every family is uncalibrated and ``penalty_for`` already returns exactly
    1.0. Skipping the penalty is a provenance statement, not a number change.

    The exposure remains explicitly bounded by its measurement scope:

      * a constant activation operator across K does not make the joint
        residual constant: its cross terms depend on each rendered weight and
        can reorder rungs inside a family. The opt-in joint AURA path prices
        those terms; these legacy weight-only rows retain that limitation;
      * AURA's validated wins (-38%/-39.5% @4B, -17.9% @27B on served KL) were
        measured against ``h_trace x output_mse``, and THAT baseline carried
        the A side (``measure_quant_cost`` applies
        ``activation_quantize_dequantize(X)``). The activation-blind projection
        beat an activation-inclusive cost, on menus already mixing W4A4 NVFP4,
        W8A8 FP8 and BF16 — the same family-choice margin.

    This is the route-flip pattern (``expert_empirical_cost`` module contract):
    a named limitation carried forward, with the served A/B as the arbiter.
    """
    if not isinstance(cost_entry, dict):
        return False
    if cost_entry.get("cost_currency") != ANCHORED_AURA_COST_CURRENCY:
        return False
    if cost_entry.get("cost_source") != ANCHORED_AURA_COST_SOURCE:
        return False
    applications = cost_entry.get("fisher_application_count")
    if isinstance(applications, bool):
        # ``True == 1``; a boolean is not a count and must not forge one.
        return False
    try:
        # ``__index__`` rather than ``int()``: it admits the numpy integers a
        # pickled cost table really carries, and refuses the string "1" and
        # the float 1.0, neither of which any writer of this stamp emits.
        applications = operator.index(applications)
    except TypeError:
        return False
    return applications == 1

def cost_entry_activation_pricing_branch(
    stats_entry: dict,
    cost_entry: dict,
    format_name: str | None = None,
    activation_pricing: ActivationFairPricing | None = None,
) -> str:
    """Name the estimator that priced one row's ACTIVATION contract.

    Orthogonal to ``cost_entry_source`` (which names the cost *field*): this
    answers the audit's question — did this row's price ever see the A side,
    and if not, was it corrected? Stamped on every ``Candidate`` so the
    question is answerable from the artifact rather than from the code
    version that produced it.
    """
    return _price_basis(stats_entry, cost_entry, format_name, activation_pricing).activation_branch

def _format_act_quant_changes_input(format_name: str | None) -> bool:
    if format_name is None:
        return False
    try:
        return bool(fr.get_format(str(format_name)).act_quant_changes_input)
    except KeyError:
        return False

def _activation_penalty(
    format_name: str | None,
    activation_pricing: ActivationFairPricing | None,
) -> float:
    if activation_pricing is None:
        return 1.0
    return activation_pricing.penalty_for(
        format_name, _format_act_quant_changes_input(format_name))[0]

def cost_entry_predicted_dloss(
    stats_entry: dict,
    cost_entry: dict,
    *,
    gain: float = 1.0,
    format_name: str | None = None,
    activation_pricing: ActivationFairPricing | None = None,
    policy: CostPricingPolicy | None = None,
) -> float:
    """Return the allocator's authoritative Δloss for one cost entry.

    ``activation_pricing`` (P5a) applies the per-family activation calibration
    to the WEIGHT-ONLY branches only — the measured branch is already
    activation-inclusive and the bit-exact branch is, by construction, an
    identity activation path. ``None`` (the default) is bit-for-bit the
    pre-P5a precedence, which is what ``kl_measurement`` and every direct
    caller outside candidate construction still want.

    The correction is multiplicative, so it scales the UCB hedge with the
    point estimate (both are in the same weight-only units and the transfer
    to the measured scale applies to both), and it cannot lift an exactly-0.0
    price off zero — ``cost_entry_prices_unmeasured_activation_at_zero``
    keeps its full strength.
    """
    policy = CostPricingPolicy.from_environment() if policy is None else policy
    basis = _price_basis(stats_entry, cost_entry, format_name, activation_pricing)
    return _basis_dloss(basis, stats_entry, cost_entry,
        gain=gain, format_name=format_name, policy=policy)

ACT_DLOSS_KEY = "act_dloss"

def cost_entry_act_dloss(cost_entry: dict) -> float:
    """AQUA-AURA: the row's ACTIVATION-side Δloss, or 0.0 when absent.

    WHY THIS IS ADDED, NOT MULTIPLIED, AND ONLY HERE
    ------------------------------------------------
    Choosing NVFP4 commits the Linear's ACTIVATIONS to 4 bits as well as its
    weights; choosing FP8 commits them to 8; BF16 leaves them alone. A
    weight-only surrogate cannot see any of that -- NVFP4 and NVFP4A16 render
    weights BIT-IDENTICALLY (T1) -- so the DP was pricing a W4A4 format as if it
    were weight-only and systematically over-buying 4-bit.

    The term is a diagonal activation Δloss approximation derived from the
    layer's own activation statistics and is added to the weight estimate.
    A shared Δloss label does not establish equivalence to AURA's KL-adjoint
    currency or capture weight/activation cross terms (PrismaQuant #237).
    It is deliberately not a multiplicative penalty:
    ``ActivationFairPricing`` (P5a) is the multiplicative, per-FAMILY, fitted
    transfer of a weight-space number onto the measured output scale, and this
    is a per-UNIT, mechanistic, already-output-space quantity. The current
    weight-only branches add this term before applying the family penalty.

    It is added ONLY on the weight-only branches. The ``_prices_from_output_mse``
    branch is already activation-inclusive by construction (the row's own
    output-space measurement saw the activation path), and adding here would
    double-count it. ``cost_entry_is_exact_by_construction`` returns 0.0 for a
    row whose activation path is an identity, so there is nothing to add there
    either; a lossless re-encode whose SERVED activations are 4-bit is excluded
    from the menu by ``cost_entry_prices_unmeasured_activation_at_zero`` rather
    than priced at all.

    Defaults to 0.0 rather than raising because every cost artifact written
    before AQUA lacks the key, and those runs must remain bit-for-bit
    reproducible. An unmeasured A-side on an activation-quantizing format is a
    HOLE, not a zero, and is reported as such by the writer -- it is not this
    function's job to guess.
    """
    try:
        return float(cost_entry.get(ACT_DLOSS_KEY, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0

def cost_entry_prices_unmeasured_activation_at_zero(
    stats_entry: dict,
    cost_entry: dict,
    priced_dloss: float,
    format_name: str | None = None,
) -> bool:
    """Whether this row prices a W-and-A format's UNKNOWN cost at the global
    optimum (Δloss exactly 0.0).

    ``cost_entry_is_bit_exact`` closed this for the ``output_mse`` branch: a
    weight-lossless entry on an activation-quantizing format keeps its measured
    A-side output_mse instead of short-circuiting to 0.0. But that branch is
    only taken when output_mse is a real measurement. Packed-expert rows whose
    routed forward could not be reconstructed (``can_measure_output`` false),
    and EVERY row in a run with ``PRISMAQUANT_EXPERT_COST_SAMPLE`` set, are
    written with ``output_mse_measured=False`` (measure_quant_cost), so pricing
    falls through to ``predicted_dloss``/``weight_mse`` — both of which are
    exactly 0.0 for a weight-lossless re-encode (the source is already in that
    format: MXFP4 over an MXFP4-packed source, NVFP4 over an NVFP4 source).
    Nothing in that row ever looked at the activation path, yet the DP reads a
    cost of 0.0: the unbeatable global minimum at any budget, for an assignment
    whose served activations are 4-bit.

    This is not a mis-estimate to be corrected — a positive weight-side
    surrogate is the accepted L1 design, biased but tradeable — it is a cost the
    optimizer CANNOT trade off: zero is the argmin, so the format is selected at
    every target, unconditionally. The unknown must therefore be excluded from
    the menu (``build_candidates``, counted and logged like any other
    inapplicable format) rather than priced.

    The predicate is exact, not thresholded:

      * the format's activation path is provably non-identity
        (``FormatSpec.act_quant_changes_input`` — a dtype-level fact);
      * no measured output-side evidence exists for this row
        (``_has_measured_output_mse``), so the A-side error is unknown
        whatever produced the number;
      * the resulting price is exactly 0.0 — the DP's global optimum;
      * the row's measured sensitivity is POSITIVE. ``h_trace == 0`` prices
        every format at 0.0 including the passthrough ones, which is a measured
        statement that no perturbation of this Linear's output moves the loss —
        W-side or A-side, since the same Fisher expansion multiplies both. A
        zero-token expert at thin calibration is exactly that row, and it must
        stay free to take the cheapest format instead of being forced onto
        BF16.
    """
    basis = _price_basis(stats_entry, cost_entry, format_name, None)
    return _unknown_activation_zero(basis, stats_entry, cost_entry, priced_dloss, format_name)

def _cost_ucb_z() -> float:
    """PRISMAQUANT_COST_UCB_Z: stderr multiples added to predicted_dloss."""
    try:
        return max(0.0, float(os.environ.get("PRISMAQUANT_COST_UCB_Z", "0")))
    except Exception:
        return 0.0

def resolve_cost_row(cost_rows: dict, fmt_name: str) -> tuple[dict | None, str]:
    """Resolve one Linear's cost row for ``fmt_name``, alias-aware.

    Returns ``(entry, entry_fmt)`` where ``entry_fmt`` is the alias actually
    present in the cost table (what ``calibrated_gains`` may be keyed by), or
    ``(None, fmt_name)`` when the format was never measured.
    """
    for candidate_name in fr.aliases_for(fmt_name):
        if candidate_name in cost_rows:
            return cost_rows[candidate_name], candidate_name
    return None, fmt_name
