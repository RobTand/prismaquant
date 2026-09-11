"""Continuous trellis rate surface: dense rungs from sparse measured anchors.

RESEARCH ONLY.  Nothing here is imported by ``run-pipeline.sh``, the default
format menu, or any export path.

The problem this closes
-----------------------
``tessera_formats`` addresses 761 integer E2M1 rungs and 1785 E4M3 rungs, and
``tessera_allocator`` can price and rank any of them exactly.  What was
missing is the step between: a rung has no cost until something *encodes* it,
and encoding every rung on every Linear is not affordable.  So the addressable
surface was continuous while the *measured* surface was five points.

``adaptive_trellis_rate_surface`` is the complement of this module, not a
substitute: it proposes **which rung to measure next** by bisecting hull
brackets around a target marginal price.  It never predicts an unmeasured
rung.  This module predicts, so that the allocator sees a dense menu; the two
compose as measure -> interpolate -> allocate -> propose the next anchor where
lambda* actually landed.

What it does NOT assume
-----------------------
No parametric rate law.  A global log-linear fit is wrong on this family and
the ladder shows exactly where: E4M3 wSNR runs 11.04/16.79/22.12/27.35/29.57
dB at rates 2-6, i.e. ~5.4 dB per bit until it saturates against the scalar
E4M3 ceiling, where the last step delivers 2.22.  A straight line through
those anchors overpredicts the top of the range badly.  So interpolation is
**monotone piecewise-linear in (q256, log2 dloss) between bracketing anchors
only**, and extrapolation beyond the measured envelope is refused rather than
guessed.  Interpolation error is then a measurable quantity, not a modelling
assumption: :func:`leave_one_anchor_out` reports it, and
:func:`allocation_regret` reports the only thing that actually matters.

Gate on the decision, not the residual
--------------------------------------
The one-anchor campaign established this the expensive way: per-superblock
D(R2)->D(R3) transfer has a 0.32 log2 residual (~25% relative D) and yet
costs 0.04-0.23% allocation regret, because noise only misranks blocks that
were near-indifferent anyway.  A residual-style bar would have refused a
near-omniscient decision.  :func:`leave_one_anchor_out` is therefore a
diagnostic and :func:`allocation_regret` is the gate.

Currency
--------
``predicted_dloss`` means whatever objective measured the anchors, and a DP
that mixes objectives is meaningless.  A surface therefore carries a declared
``currency`` string, and :func:`rate_surface_solver_menu` refuses to build a
menu from surfaces that disagree about it.  Note for anyone wiring the
trellis ladder in directly: the ladder's weighted SSE uses a per-input-channel
activation second moment, which is an output-MSE proxy, **not** the AURA
KL-adjoint objective the production DP prices in.  Those are different
currencies and must not be mixed.

Provenance
----------
Every emitted candidate is stamped ``interpolated`` unless its rung exactly
matches a measured anchor, in which case it is stamped ``measured`` and
carries the anchor's own value.  An interpolated rung is a *proposal*: the
render encodes it for real and the held-out KL gate judges it, which is the
house rule (surrogates generate, real KL selects).  Nothing here may be
reported as a measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence, TYPE_CHECKING

from tessera.errors import GrammarError

from .tessera_allocator import (
    TesseraAllocatorCandidate,
    build_tessera_allocator_candidate,
    tessera_solver_candidate_menu,
)
from .tessera_formats import (
    SUPERBLOCK_WEIGHTS,
    TesseraFamily,
    TesseraFormatError,
    get_tessera_family,
    validate_body_rate_q256,
)

if TYPE_CHECKING:
    from tessera.export import WireRecipe

__all__ = [
    "STACK_TRANSFER_REFERENCE_Q256",
    "StackRateSample",
    "StackTransferLaw",
    "TesseraRateSurface",
    "allocation_regret",
    "densify_rate_surface",
    "fit_rate_surface",
    "fit_stack_transfer_law",
    "leave_one_anchor_out",
    "predict_stack_rates",
    "rate_surface_solver_menu",
    "uniform_column_schedule",
]


PROVENANCE_MEASURED = "measured"
PROVENANCE_INTERPOLATED = "interpolated"


def uniform_column_schedule(
    columns: int,
    body_rate_q256: int,
    *,
    family: str | TesseraFamily,
    recipe: "WireRecipe | None" = None,
) -> tuple[int, ...]:
    """Return the flattest legal per-input-column schedule hitting a rate.

    Tessera's own wire (``prismaquant.tessera.v1``) carries one rate code per
    input column, shared across every output row -- ``Manifest`` refuses a
    schedule whose length is not ``geometry.columns``, and Bresenham placement
    plus ``superblock_quota_ok`` is what bounds the variation -- so the
    tensor-wide total lands within one physical body bit of the declared q256.
    (This paragraph cited ``gridbook.trellis.wire.v1`` until 2026-09-02.  That
    wire is the retired predecessor's, priced by ``trellis_footprint``, and it
    is not the authority for anything Tessera writes; the property is the same
    but it is Tessera's manifest that asserts it.)  That is
    what makes a continuous rate surface expressible at all: the achievable
    totals are the integers, so the rate resolution is ``256/columns`` q256
    per step -- 0.25 q256 on a 1024-column Linear.

    This is ``TesseraFamily.column_schedule`` -- ``bresenham_rate_schedule``
    -- asked directly, so the schedule offered is the schedule the encoder
    accepts, by construction rather than by a second implementation of its
    rule.  Flattest-legal is what the grammar returns: the two rates
    bracketing the root, spread deterministically.  There is deliberately no
    per-block floor on coded positions here: the grammar has none (a schedule
    is legal iff every rate is in ``1..cap`` and the quota closes), and the
    ``MIN_TRELLIS_STEPS = 8`` floor this helper used to carry was the retired
    Gridbook wire's ``STATE_MEMORY_BITS``, refused by no Tessera check --
    which is why it punched menu holes at exactly the high-rate rungs the
    allocator most wants (#161).  A rung this width cannot realise is refused
    by the grammar and re-raised in this module's error type, so every caller
    that guards with ``except TesseraFormatError`` keeps working.
    """

    spec = get_tessera_family(family)
    rate = validate_body_rate_q256(spec, body_rate_q256, recipe=recipe)
    if type(columns) is not int or columns <= 0:
        raise TesseraFormatError("columns must be a positive integer")
    if columns % SUPERBLOCK_WEIGHTS:
        raise TesseraFormatError(
            f"columns must be a multiple of {SUPERBLOCK_WEIGHTS}; a short "
            f"final block is legal on the wire but its rate accounting is "
            f"the caller's to declare, not this helper's to guess"
        )
    try:
        return spec.column_schedule(rate, columns, recipe=recipe)
    except GrammarError as exc:
        raise TesseraFormatError(
            f"{spec.name}: rung {rate} is not realisable over {columns} "
            f"columns -- {exc}"
        ) from exc


@dataclass(frozen=True, slots=True)
class TesseraRateSurface:
    """One unit's monotone rate surface, interpolated between measured anchors."""

    unit_name: str
    family: str
    layout: str
    currency: str
    anchor_q256: tuple[int, ...]
    anchor_dloss: tuple[float, ...]
    anchor_stderr: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.anchor_q256) < 2:
            raise TesseraFormatError(
                "a rate surface needs at least two measured anchors; one "
                "anchor is a point, not a surface"
            )
        if not (
            len(self.anchor_q256)
            == len(self.anchor_dloss)
            == len(self.anchor_stderr)
        ):
            raise TesseraFormatError("anchor arrays must agree in length")
        for left, right in zip(self.anchor_q256, self.anchor_q256[1:]):
            if left >= right:
                raise TesseraFormatError(
                    "anchor rates must be strictly increasing"
                )
        for value in self.anchor_dloss:
            if not math.isfinite(value) or value <= 0.0:
                raise TesseraFormatError(
                    "anchor dloss must be positive and finite; a zero or "
                    "negative loss cannot be interpolated in log space"
                )
        for left, right in zip(self.anchor_dloss, self.anchor_dloss[1:]):
            if right >= left:
                raise TesseraFormatError(
                    "anchor dloss must strictly decrease as rate rises; a "
                    "non-monotone anchor set is a measurement problem, and "
                    "interpolating through it would launder it into a cost"
                )

    @property
    def q256_range(self) -> tuple[int, int]:
        return (self.anchor_q256[0], self.anchor_q256[-1])

    def predict(self, body_rate_q256: int) -> float:
        """Interpolated dloss at one rung.  Refuses to extrapolate."""

        if type(body_rate_q256) is not int:
            raise TesseraFormatError("body_rate_q256 must be an integer")
        low, high = self.q256_range
        if not low <= body_rate_q256 <= high:
            raise TesseraFormatError(
                f"rung {body_rate_q256} is outside the measured envelope "
                f"[{low}, {high}]; extrapolating a trellis rate surface is "
                f"refused -- measure another anchor instead"
            )
        for index, rate in enumerate(self.anchor_q256):
            if rate == body_rate_q256:
                return self.anchor_dloss[index]
        upper = next(
            index
            for index, rate in enumerate(self.anchor_q256)
            if rate > body_rate_q256
        )
        lower = upper - 1
        span = self.anchor_q256[upper] - self.anchor_q256[lower]
        weight = (body_rate_q256 - self.anchor_q256[lower]) / span
        log_low = math.log2(self.anchor_dloss[lower])
        log_high = math.log2(self.anchor_dloss[upper])
        return float(2.0 ** (log_low + weight * (log_high - log_low)))

    def predict_stderr(self, body_rate_q256: int) -> float:
        """Anchor stderr at an anchor; the wider bracketing one between them.

        Deliberately conservative: an interpolated rung inherits the larger
        of the two anchors it sits between, so uncertainty never shrinks by
        the act of interpolating.
        """

        low, high = self.q256_range
        if not low <= body_rate_q256 <= high:
            raise TesseraFormatError(
                f"rung {body_rate_q256} is outside [{low}, {high}]"
            )
        for index, rate in enumerate(self.anchor_q256):
            if rate == body_rate_q256:
                return self.anchor_stderr[index]
        upper = next(
            index
            for index, rate in enumerate(self.anchor_q256)
            if rate > body_rate_q256
        )
        return max(self.anchor_stderr[upper - 1], self.anchor_stderr[upper])

    def provenance(self, body_rate_q256: int) -> str:
        return (
            PROVENANCE_MEASURED
            if body_rate_q256 in self.anchor_q256
            else PROVENANCE_INTERPOLATED
        )


def fit_rate_surface(
    records: Sequence[TesseraAllocatorCandidate],
    *,
    currency: str,
) -> TesseraRateSurface:
    """Build one unit's surface from its measured anchor candidates."""

    if not isinstance(currency, str) or not currency:
        raise TesseraFormatError(
            "currency must be a nonempty string naming the objective the "
            "anchors were measured in"
        )
    materialized = list(records)
    if not materialized:
        raise TesseraFormatError("a rate surface needs anchor records")
    units = {record.unit_name for record in materialized}
    families = {record.family for record in materialized}
    layouts = {record.layout for record in materialized}
    if len(units) != 1:
        raise TesseraFormatError("anchors must describe exactly one unit")
    if len(families) != 1:
        raise TesseraFormatError("anchors must share one trellis family")
    if len(layouts) != 1:
        raise TesseraFormatError(
            "anchors must share one layout; fixed-quota and tight-offset "
            "curves are different rate axes and cannot be interpolated across"
        )
    ordered = sorted(materialized, key=lambda record: record.body_rate_q256)
    rates = tuple(record.body_rate_q256 for record in ordered)
    if len(set(rates)) != len(rates):
        raise TesseraFormatError("anchors must not repeat a rate")
    return TesseraRateSurface(
        unit_name=next(iter(units)),
        family=next(iter(families)),
        layout=next(iter(layouts)),
        currency=currency,
        anchor_q256=rates,
        anchor_dloss=tuple(
            float(record.predicted_dloss_mean) for record in ordered
        ),
        anchor_stderr=tuple(
            float(record.predicted_dloss_stderr) for record in ordered
        ),
    )


def densify_rate_surface(
    surface: TesseraRateSurface,
    shape: Sequence[int],
    *,
    q256_values: Sequence[int],
    alphabets: Mapping[int, Sequence[int]],
    schedule_for: "callable | None" = None,
    target_profile: str | None = "research",
    qname: str | None = None,
    packed_expert: bool | None = None,
    sidecar_header_bytes: int = 0,
) -> tuple[TesseraAllocatorCandidate, ...]:
    """Price a dense set of rungs from one interpolated surface.

    Each rung gets a real schedule and therefore an exact byte footprint --
    the rate is interpolated, the *bytes* never are.  ``schedule_for`` may
    override the flattest-legal default for callers holding per-column
    importance; it receives ``(columns, body_rate_q256)`` and must return a
    schedule the wire accepts.
    """

    dims = tuple(shape)
    if len(dims) != 2:
        raise TesseraFormatError("shape must be two dimensions")
    columns = dims[1]
    built: list[TesseraAllocatorCandidate] = []
    for rate in sorted(set(int(value) for value in q256_values)):
        schedule = (
            schedule_for(columns, rate)
            if schedule_for is not None
            else uniform_column_schedule(
                columns, rate, family=surface.family,
            )
        )
        # Each rung uses only the rates its own schedule contains, and
        # `validate_alphabets` requires the mapping to match EXACTLY -- a
        # superset is refused.  So the caller supplies every alphabet it
        # holds and the rung selects the ones it actually spends.  Every
        # scheduled rate is a coded trellis rate needing its table: "bypass"
        # was the retired Gridbook wire's word and Tessera has no uncoded
        # rate, so the set is the schedule's rates, unfiltered (#161).
        used = set(schedule)
        missing = used - set(alphabets)
        if missing:
            raise TesseraFormatError(
                f"rung {rate} needs alphabets for rates {sorted(missing)}; "
                f"got {sorted(alphabets)}"
            )
        built.append(
            build_tessera_allocator_candidate(
                surface.unit_name,
                dims,
                family=surface.family,
                body_rate_q256=rate,
                layout=surface.layout,
                schedule=schedule,
                alphabets={key: alphabets[key] for key in sorted(used)},
                predicted_dloss=surface.predict(rate),
                predicted_dloss_stderr=surface.predict_stderr(rate),
                target_profile=target_profile,
                qname=qname,
                packed_expert=packed_expert,
                sidecar_header_bytes=sidecar_header_bytes,
                variant_label=surface.provenance(rate),
            )
        )
    return tuple(built)


def rate_surface_solver_menu(
    densified: Mapping[str, Sequence[TesseraAllocatorCandidate]],
    *,
    currencies: Mapping[str, str],
    fail_on_denied: bool = False,
) -> dict[str, list]:
    """Hand a dense multi-unit surface to the unchanged global allocation DP.

    Refuses a menu whose units disagree about currency: a DP that ranks an
    AURA-priced unit against an output-MSE-priced one is not solving any
    stated objective.
    """

    declared = set(currencies.values())
    if len(declared) != 1:
        raise TesseraFormatError(
            f"rate surface menu mixes objectives {sorted(declared)}; one DP "
            f"prices in one currency"
        )
    missing = set(densified) - set(currencies)
    if missing:
        raise TesseraFormatError(
            f"units {sorted(missing)} have no declared currency"
        )
    flat: list[TesseraAllocatorCandidate] = []
    for unit_name in sorted(densified):
        flat.extend(densified[unit_name])
    return tessera_solver_candidate_menu(flat, fail_on_denied=fail_on_denied)


def leave_one_anchor_out(
    surface: TesseraRateSurface,
) -> dict[str, object]:
    """Diagnostic: how well do the anchors predict each other?

    Drops one interior anchor at a time, rebuilds the surface from the rest,
    and reports the log2 error at the dropped rung.  Endpoints cannot be
    dropped -- removing one would require extrapolation, which is refused.

    This is a DIAGNOSTIC, not a gate.  Use :func:`allocation_regret`.
    """

    if len(surface.anchor_q256) < 3:
        return {
            "interior_anchors": 0,
            "note": "fewer than three anchors: nothing droppable",
        }
    errors: list[dict[str, float]] = []
    for index in range(1, len(surface.anchor_q256) - 1):
        kept = TesseraRateSurface(
            unit_name=surface.unit_name,
            family=surface.family,
            layout=surface.layout,
            currency=surface.currency,
            anchor_q256=(
                surface.anchor_q256[:index] + surface.anchor_q256[index + 1:]
            ),
            anchor_dloss=(
                surface.anchor_dloss[:index] + surface.anchor_dloss[index + 1:]
            ),
            anchor_stderr=(
                surface.anchor_stderr[:index]
                + surface.anchor_stderr[index + 1:]
            ),
        )
        rate = surface.anchor_q256[index]
        truth = surface.anchor_dloss[index]
        predicted = kept.predict(rate)
        errors.append(
            {
                "q256": float(rate),
                "true_dloss": truth,
                "predicted_dloss": predicted,
                "log2_error": math.log2(predicted / truth),
                "rel_error_pct": (predicted / truth - 1.0) * 100.0,
            }
        )
    magnitudes = [abs(entry["log2_error"]) for entry in errors]
    return {
        "interior_anchors": len(errors),
        "max_abs_log2_error": max(magnitudes),
        "median_abs_log2_error": sorted(magnitudes)[len(magnitudes) // 2],
        "per_anchor": errors,
    }


def allocation_regret(
    surfaces: Mapping[str, TesseraRateSurface],
    truth: Mapping[str, Mapping[int, float]],
    *,
    unit_bytes: Mapping[str, Mapping[int, int]],
    byte_budget: int,
) -> dict[str, object]:
    """The gate: what does interpolating COST the decision, not the estimate?

    Allocates a byte budget across units twice -- once on the interpolated
    surface, once on measured truth -- and reports the true objective of each.
    Both allocations are scored with TRUTH, so the number is the real cost of
    deciding on interpolated values.

    Greedy marginal allocation, which is the same rule the lambda path uses.
    It is exact only for separable CONVEX per-unit curves, and real trellis
    interiors are not reliably convex -- 142 of 196 tensors have native
    fractional rates sagging above their own chord.  The regret number stays
    valid because BOTH arms run the identical allocator, so the comparison
    isolates the interpolation; it is not a claim of optimal allocation.
    """

    if type(byte_budget) is not int or byte_budget <= 0:
        raise TesseraFormatError("byte_budget must be a positive integer")

    def allocate(score: Mapping[str, Mapping[int, float]]) -> dict[str, int]:
        chosen: dict[str, int] = {}
        for unit_name in sorted(score):
            rates = sorted(score[unit_name])
            chosen[unit_name] = rates[0]
        spent = sum(
            unit_bytes[unit_name][rate] for unit_name, rate in chosen.items()
        )
        if spent > byte_budget:
            raise TesseraFormatError(
                f"the cheapest assignment already costs {spent} bytes, over "
                f"the {byte_budget}-byte budget"
            )
        while True:
            best_unit = None
            best_rate = None
            best_gain = 0.0
            for unit_name, rate in chosen.items():
                rates = sorted(score[unit_name])
                position = rates.index(rate)
                for candidate in rates[position + 1:]:
                    delta_bytes = (
                        unit_bytes[unit_name][candidate]
                        - unit_bytes[unit_name][rate]
                    )
                    if delta_bytes <= 0 or spent + delta_bytes > byte_budget:
                        continue
                    delta_loss = (
                        score[unit_name][rate] - score[unit_name][candidate]
                    )
                    gain = delta_loss / delta_bytes
                    if gain > best_gain:
                        best_gain = gain
                        best_unit = unit_name
                        best_rate = candidate
            if best_unit is None:
                return chosen
            spent += (
                unit_bytes[best_unit][best_rate]
                - unit_bytes[best_unit][chosen[best_unit]]
            )
            chosen[best_unit] = best_rate

    predicted_score = {
        unit_name: {
            rate: surfaces[unit_name].predict(rate)
            for rate in truth[unit_name]
        }
        for unit_name in surfaces
    }
    on_interpolated = allocate(predicted_score)
    on_truth = allocate(truth)
    loss_interpolated = sum(
        truth[unit_name][rate] for unit_name, rate in on_interpolated.items()
    )
    loss_truth = sum(
        truth[unit_name][rate] for unit_name, rate in on_truth.items()
    )
    return {
        "byte_budget": byte_budget,
        "true_loss_deciding_on_interpolated": loss_interpolated,
        "true_loss_deciding_on_truth": loss_truth,
        "regret_pct": (loss_interpolated / loss_truth - 1.0) * 100.0,
        "assignment_agreement": sum(
            on_interpolated[unit] == on_truth[unit] for unit in on_truth
        )
        / len(on_truth),
        "assignment_interpolated": on_interpolated,
        "assignment_truth": on_truth,
    }


# ---------------------------------------------------------------------------
# The stack transfer law
# ---------------------------------------------------------------------------
#
# Everything above this line prices ONE unit from ITS OWN anchors: a dense
# Linear measured at three rungs, interpolated between them.  A packed routed
# MoE stack is a different shape of problem and needs a different instrument.
#
# Why a second instrument.  A routed stack is ONE rate decision covering every
# expert in the layer (``allocator_candidates.aggregate_packed_serving_groups``
# makes the packed group a single multi-choice DP item, and the pinned Tessera
# contract's two ``routed_moe`` cells publish no per-expert rate licence), yet
# a census pays 3 x E x R encodes to resolve it.  The offline regret study
# ``docs/results/glm_tessera_probe_reduction_regret_2026-09-10.md`` measured
# what happens if a stack is instead measured as a CENSUS at one reference
# rung plus a small expert SAMPLE at the others, with the missing rungs
# predicted per expert by
#
#     log2 mse_{e,p}(r) = a_{p,r} + b_{p,r} * log2 mse_{e,p}(reference)
#
# where ``b`` is pooled over the OTHER stacks (their per-stack spread is sd
# 0.02-0.03 over 28 stacks, so pooling costs nothing) and ``a`` is fitted per
# (stack, projection) from that stack's own sampled experts.  At a 3% sample
# this reproduced the omniscient allocation to 0.004% mean / 0.000% p90
# regret at the campaign's byte target and cut routed anchor-encode time by
# 66% before the selective encode of the winners (~45% net).
#
# ``TesseraRateSurface`` is NOT this and must not be asked to be: it
# interpolates one unit's own measured anchors and refuses to extrapolate.
# Dense units keep three anchors and keep that surface (report section 4).
#
# What this is not.  These are MODEL PREDICTIONS -- not measurements, and not
# sampling estimates.  The distinction has teeth here: a Horvitz-Thompson
# ``dloss_stderr`` describes the variance of a DESIGN over repeated draws and
# is meaningful only for a row whose value came from a draw.  A transfer-law
# row's value came from a regression, so it carries a model error under its
# own name and never an ``estimator`` or a ``dloss_stderr``
# (RobTand/prismaquant#495 part 3).

#: The rung a stack is censused at, and therefore the regressor every
#: prediction is made from.  Named rather than spelled at the call sites,
#: because the law is defined relative to whichever rung was censused.
STACK_TRANSFER_REFERENCE_Q256 = 960


def _check_positive_mse(stack, row, projections, where) -> None:
    for projection in projections:
        value = row.get(projection)
        if value is None:
            raise TesseraFormatError(
                f"{stack}: {where} is missing projection {projection!r}")
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise TesseraFormatError(
                f"{stack}: {where} projection {projection!r} is {value!r}; the "
                "law is fitted in log2 space and a non-positive mse has no log")


@dataclass(frozen=True, slots=True)
class StackRateSample:
    """One packed stack's evidence: a census at one rung, a sample at others.

    ``reference_mse`` is expert -> projection -> ``output_mse`` at
    ``reference_q256`` over EVERY expert in the stack (the census tier).
    ``sampled_mse`` is q256 -> expert -> projection -> ``output_mse`` over the
    drawn experts only (the transfer-law tier).

    ``weights`` is the weight applied to EACH projection's mse, i.e. the
    production ``h_trace_per_expert[e] / roles`` that
    ``tessera_campaign._stack_member_weight`` applies, so that
    ``sum_e weights[e] * sum_p mse_{e,p}`` reproduces exactly the total
    ``_horvitz_thompson_stack`` estimates on the same evidence.  ``None`` means
    uniform 1.0, which is the study's ``uniform`` weighting and the only
    honest convention when no Fisher probe exists.
    """

    stack: str
    projections: tuple[str, ...]
    experts: tuple[int, ...]
    reference_q256: int
    reference_mse: "Mapping[int, Mapping[str, float]]"
    sampled_experts: tuple[int, ...]
    sampled_mse: "Mapping[int, Mapping[int, Mapping[str, float]]]"
    weights: "Mapping[int, float] | None" = None
    currency: str = ""

    def __post_init__(self) -> None:
        if not self.projections:
            raise TesseraFormatError(f"{self.stack}: a stack sample needs projections")
        if len(set(self.projections)) != len(self.projections):
            raise TesseraFormatError(f"{self.stack}: duplicate projection name")
        if not self.experts:
            raise TesseraFormatError(f"{self.stack}: a stack sample needs experts")
        if len(set(self.experts)) != len(self.experts):
            raise TesseraFormatError(f"{self.stack}: duplicate expert id")
        if not self.sampled_experts:
            raise TesseraFormatError(
                f"{self.stack}: no sampled experts; the per-stack intercept is "
                "fitted on the sample and cannot be fitted on nothing")
        if not set(self.sampled_experts) <= set(self.experts):
            raise TesseraFormatError(
                f"{self.stack}: sampled experts are not a subset of the frame")
        for expert in self.experts:
            row = self.reference_mse.get(expert)
            if not isinstance(row, Mapping):
                raise TesseraFormatError(
                    f"{self.stack}: expert {expert} has no reference row; the "
                    f"reference rung {self.reference_q256} must be a census, "
                    "because every expert's prediction is made from its own "
                    "measured reference value")
            _check_positive_mse(self.stack, row, self.projections,
                                f"reference {self.reference_q256}")
        if not self.sampled_mse:
            raise TesseraFormatError(f"{self.stack}: no sampled rungs to fit")
        for rate, by_expert in self.sampled_mse.items():
            if int(rate) == int(self.reference_q256):
                raise TesseraFormatError(
                    f"{self.stack}: rung {rate} is the reference; a rung cannot "
                    "be both the regressor and the prediction")
            for expert in self.sampled_experts:
                row = by_expert.get(expert)
                if not isinstance(row, Mapping):
                    raise TesseraFormatError(
                        f"{self.stack}: sampled expert {expert} has no row at "
                        f"rung {rate}; a rung measured on only some of the "
                        "drawn experts cannot fit this stack's intercept")
                _check_positive_mse(self.stack, row, self.projections,
                                    f"rung {rate}")

    @property
    def target_q256(self) -> tuple[int, ...]:
        return tuple(sorted(int(rate) for rate in self.sampled_mse))

    def weight(self, expert: int) -> float:
        return 1.0 if self.weights is None else float(self.weights[expert])

    def reference_total(self) -> float:
        """The weighted total at the censused rung -- the one measured total.

        Only the reference rung covers every expert, so this is the only total
        this record can state without a model.
        """
        return math.fsum(
            self.weight(e) * math.fsum(
                float(self.reference_mse[e][p]) for p in self.projections)
            for e in self.experts)


@dataclass(frozen=True, slots=True)
class StackTransferLaw:
    """Pooled per-projection slopes, and what their residuals were.

    ``slope[rate][projection]`` is the OLS slope of ``log2 mse(rate)`` on
    ``log2 mse(reference)`` over the pooled experts of every stack other than
    ``held_out`` -- ONE pooled centring, exactly as the study fitted it, not a
    fixed-effects fit.  ``residual_sd_log2`` is that fit's residual standard
    deviation in log2 units: the per-expert pointwise MODEL error, and not a
    sampling error of any design.
    """

    reference_q256: int
    target_q256: tuple[int, ...]
    projections: tuple[str, ...]
    slope: "Mapping[int, Mapping[str, float]]"
    residual_sd_log2: "Mapping[int, Mapping[str, float]]"
    pooled_rows: "Mapping[int, int]"
    pooled_stacks: tuple[str, ...]
    held_out: str | None
    currency: str = ""

    def as_dict(self) -> dict:
        """The JSON-portable form a cost row carries."""
        return {
            "reference_q256": int(self.reference_q256),
            "target_q256": [int(r) for r in self.target_q256],
            "projections": list(self.projections),
            "slope": {str(r): {p: float(v) for p, v in sorted(row.items())}
                      for r, row in sorted(self.slope.items())},
            "residual_sd_log2": {
                str(r): {p: float(v) for p, v in sorted(row.items())}
                for r, row in sorted(self.residual_sd_log2.items())},
            "pooled_rows": {str(r): int(n)
                            for r, n in sorted(self.pooled_rows.items())},
            "pooled_stacks": list(self.pooled_stacks),
            "held_out": self.held_out,
            "currency": self.currency,
        }


def _stack_rate_samples(stacks_measured) -> "dict[str, StackRateSample]":
    if isinstance(stacks_measured, Mapping):
        items = list(stacks_measured.values())
    else:
        items = list(stacks_measured)
    samples: dict[str, StackRateSample] = {}
    for sample in items:
        if not isinstance(sample, StackRateSample):
            raise TesseraFormatError(
                "fit_stack_transfer_law takes StackRateSample records; "
                f"got {type(sample).__name__}")
        if sample.stack in samples:
            raise TesseraFormatError(f"{sample.stack}: duplicate stack record")
        samples[sample.stack] = sample
    if not samples:
        raise TesseraFormatError("no stack records to fit a transfer law on")
    return samples


def fit_stack_transfer_law(
    stacks_measured: "Mapping[str, StackRateSample] | Sequence[StackRateSample]",
    hold_out: str | None = None,
) -> StackTransferLaw:
    """Pool per-projection slopes over every stack except ``hold_out``.

    The fit is the study's, reproduced rather than reinvented: for each target
    rung and projection, concatenate the sampled experts of all the OTHER
    stacks, centre ``log2 mse(reference)`` and ``log2 mse(rate)`` on one pooled
    mean each, and take ``sum(xc*yc) / sum(xc*xc)``.  The pooled centring is
    what leaves the intercept a per-stack quantity: the slope carries the
    cross-expert shape, the intercept carries each stack's own level.

    ``hold_out`` names the stack this law will be used to predict, which is
    what keeps a prediction out of its own fit.  The study fitted every
    reported number leave-one-stack-out and found the per-layer slope spread
    (sd 0.02-0.03 over 28 stacks) small enough that the hold-out costs nothing.

    Known sensitivity, stated rather than hidden.  ONE pooled centring is the
    study's estimator and is reproduced here because the regret it measured is
    this estimator's; but it is unbiased for ``b`` only when the pooled stacks'
    per-stack intercepts are uncorrelated with their per-stack MEAN reference
    level.  A population whose deeper layers sit both higher on the reference
    axis and higher in intercept would tilt the pooled slope toward the
    BETWEEN-stack relation, which is a different quantity from the
    within-stack one the prediction uses.  A within-stack (fixed-effects)
    centring would remove that, and would also change the estimator that was
    validated, so it is not done here.  What the study offers against the risk
    is evidence rather than an argument: the 28 per-layer slopes agree to sd
    0.02-0.03, which a strong between-stack tilt would not survive.  Any
    residual tilt inflates ``residual_sd_log2``, so the reported model error
    errs conservatively.
    """
    samples = _stack_rate_samples(stacks_measured)
    pooled = [s for name, s in samples.items() if name != hold_out]
    if not pooled:
        raise TesseraFormatError(
            "a pooled slope needs at least one stack other than the held-out "
            "one; a law fitted on the stack it predicts is not a hold-out")
    references = {int(s.reference_q256) for s in pooled}
    if len(references) != 1:
        raise TesseraFormatError(
            f"the pooled stacks disagree about the reference rung "
            f"({sorted(references)}); a slope regressed on two different "
            "regressors is not one slope")
    reference_q256 = references.pop()
    projections = pooled[0].projections
    for sample in pooled:
        if sample.projections != projections:
            raise TesseraFormatError(
                f"{sample.stack}: projections {sample.projections} differ from "
                f"{projections}; a per-projection slope needs one projection set")
    currencies = {s.currency for s in pooled}
    if len(currencies) != 1:
        raise TesseraFormatError(
            f"the pooled stacks disagree about the currency "
            f"({sorted(currencies)}); a law fitted across objectives prices "
            "nothing")
    targets = sorted(set.intersection(*(set(s.target_q256) for s in pooled)))
    if not targets:
        raise TesseraFormatError(
            "the pooled stacks share no target rung; there is nothing to fit")

    slope: dict[int, dict[str, float]] = {}
    residual: dict[int, dict[str, float]] = {}
    rows: dict[int, int] = {}
    for rate in targets:
        slope[rate], residual[rate] = {}, {}
        for projection in projections:
            xs: list[float] = []
            ys: list[float] = []
            for sample in pooled:
                for expert in sample.sampled_experts:
                    xs.append(math.log2(float(
                        sample.reference_mse[expert][projection])))
                    ys.append(math.log2(float(
                        sample.sampled_mse[rate][expert][projection])))
            count = len(xs)
            if count < 3:
                raise TesseraFormatError(
                    f"rung {rate} projection {projection!r}: {count} pooled "
                    "row(s); a slope and a residual spread need at least three")
            x_mean = math.fsum(xs) / count
            y_mean = math.fsum(ys) / count
            sxx = math.fsum((x - x_mean) ** 2 for x in xs)
            if sxx <= 0.0:
                raise TesseraFormatError(
                    f"rung {rate} projection {projection!r}: the reference "
                    "values have no spread, so no slope is identified")
            sxy = math.fsum((x - x_mean) * (y - y_mean)
                            for x, y in zip(xs, ys))
            b = sxy / sxx
            slope[rate][projection] = float(b)
            sse = math.fsum(((y - y_mean) - b * (x - x_mean)) ** 2
                            for x, y in zip(xs, ys))
            # n - 2: this fit estimated a slope and a pooled intercept.
            residual[rate][projection] = float(math.sqrt(sse / (count - 2)))
            rows[rate] = count
    return StackTransferLaw(
        reference_q256=reference_q256,
        target_q256=tuple(targets),
        projections=tuple(projections),
        slope={r: dict(v) for r, v in slope.items()},
        residual_sd_log2={r: dict(v) for r, v in residual.items()},
        pooled_rows=dict(rows),
        pooled_stacks=tuple(sorted(s.stack for s in pooled)),
        held_out=hold_out,
        currency=currencies.pop(),
    )


def predict_stack_rates(
    stack: StackRateSample,
    law: StackTransferLaw,
) -> dict:
    """The stack's total at each target rung, and what the model error is.

    Per target rung and projection the intercept is
    ``mean_{e in sample} (log2 mse_e(rate) - b * log2 mse_e(reference))``;
    every expert in the frame is then predicted from its OWN measured
    reference value, and the weighted per-expert predictions are summed.

    The error field, and why it is what it is
    -----------------------------------------
    Two errors ride on this prediction and they behave differently under the
    sum over E experts:

    * the per-expert residual, sd ``residual_sd_log2``.  It is idiosyncratic,
      so over a few hundred experts it largely averages away in the total.
    * the intercept error, sd ``residual_sd_log2 / sqrt(n)`` on ``n`` sampled
      experts.  It is COMMON to every expert in the stack -- one number shifts
      the whole predicted curve -- so it does not average away at all, and it
      is what the error on the stack TOTAL actually is.

    ``model_error`` therefore reports the common-mode term as
    ``stack_total_sd_log2`` (a root-sum-square over projections, each weighted
    by its share of the predicted total, because the projections' intercepts
    are fitted independently) and reports the pointwise per-expert term beside
    it under its own name, so neither can be read as the other.  Both are
    MODEL errors, in log2 units.  Neither is a sampling standard error and
    neither may be written to ``dloss_stderr``.

    No smearing correction.  ``sum_e 2^lhat`` underestimates ``E[sum_e mse]``
    by roughly ``exp((sigma ln2)^2 / 2)`` under lognormal residuals, and that
    factor is deliberately NOT applied: the regret the study measured is the
    regret of this uncorrected estimator, and a correction fitted on the same
    ``n`` points would move the number that was validated.  Its size is
    reported as ``smearing_factor_not_applied`` so a reader can see it.
    """
    if int(stack.reference_q256) != int(law.reference_q256):
        raise TesseraFormatError(
            f"{stack.stack}: censused at {stack.reference_q256}, the law "
            f"regresses on {law.reference_q256}")
    if stack.projections != law.projections:
        raise TesseraFormatError(
            f"{stack.stack}: projections {stack.projections} are not the law's "
            f"{law.projections}")
    if stack.currency != law.currency:
        raise TesseraFormatError(
            f"{stack.stack}: currency {stack.currency!r} is not the law's "
            f"{law.currency!r}")
    if law.held_out is not None and law.held_out != stack.stack:
        raise TesseraFormatError(
            f"{stack.stack}: this law held out {law.held_out!r}; predicting one "
            "stack from a law fitted with a different hold-out mixes designs")
    if law.held_out is None and stack.stack in law.pooled_stacks:
        raise TesseraFormatError(
            f"{stack.stack}: the law pooled this stack's own experts; fit it "
            "with hold_out=<this stack> before predicting it")
    targets = [r for r in stack.target_q256 if r in law.target_q256]
    if not targets:
        raise TesseraFormatError(
            f"{stack.stack}: the law fits {list(law.target_q256)} and this "
            f"stack samples {list(stack.target_q256)}; no rung is predictable")

    sampled = list(stack.sampled_experts)
    n = len(sampled)
    predicted: dict[int, float] = {}
    per_expert: dict[int, dict[int, float]] = {}
    intercepts: dict[int, dict[str, float]] = {}
    errors: dict[int, dict] = {}
    for rate in targets:
        intercept: dict[str, float] = {}
        share: dict[str, float] = {}
        predictions: dict[int, dict[str, float]] = {e: {} for e in stack.experts}
        for projection in stack.projections:
            b = float(law.slope[rate][projection])
            value = math.fsum(
                math.log2(float(stack.sampled_mse[rate][e][projection]))
                - b * math.log2(float(stack.reference_mse[e][projection]))
                for e in sampled) / n
            intercept[projection] = float(value)
            for expert in stack.experts:
                predictions[expert][projection] = 2.0 ** (
                    value + b * math.log2(
                        float(stack.reference_mse[expert][projection])))
            share[projection] = math.fsum(
                stack.weight(e) * predictions[e][projection]
                for e in stack.experts)
        total = math.fsum(share[p] for p in stack.projections)
        if not math.isfinite(total) or total <= 0.0:
            raise TesseraFormatError(
                f"{stack.stack}: predicted total {total!r} at rung {rate} is "
                "not a positive finite cost")
        predicted[rate] = float(total)
        per_expert[rate] = {
            int(e): float(math.fsum(predictions[e][p] for p in stack.projections))
            for e in stack.experts}
        intercepts[rate] = intercept
        pointwise = {p: float(law.residual_sd_log2[rate][p])
                     for p in stack.projections}
        common = math.sqrt(math.fsum(
            ((share[p] / total) * (pointwise[p] / math.sqrt(n))) ** 2
            for p in stack.projections))
        worst = max(pointwise.values())
        errors[rate] = {
            # The stack TOTAL's own error: the common-mode intercept term.
            "stack_total_sd_log2": float(common),
            # The per-expert pointwise model error, named so it cannot be read
            # as the line above.
            "per_expert_residual_sd_log2": dict(sorted(pointwise.items())),
            "intercept_sample_size": int(n),
            "smearing_factor_not_applied": float(
                math.exp((worst * math.log(2.0)) ** 2 / 2.0)),
            "kind": "transfer_law_model_error_log2",
            "is_sampling_error": False,
        }
    return {
        "stack": stack.stack,
        "reference_q256": int(stack.reference_q256),
        "predicted": predicted,
        "predicted_per_expert": per_expert,
        "intercept": intercepts,
        "slope": {rate: dict(law.slope[rate]) for rate in targets},
        "sample_experts": [int(e) for e in sampled],
        "intercept_sample_size": int(n),
        "model_error": errors,
        "currency": stack.currency,
    }
