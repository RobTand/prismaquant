"""Campaign table values, payloads, identity and merge policy.

Numerical campaign execution supplies measured anchors. This owner writes their
metadata and combines only tables whose published inputs and scope agree.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .digests import (
    DIRECT_ASCII_STRICT, DIRECT_ASCII_SPACED_LAX_DEFAULT_STR,
    DIRECT_UTF8_INDENT2_STRICT, text_sha256hex, file_sha256hex,
)
from .cost_stage_checkpoint import merge_identity_migrations
from .schemas import strict_json_loads
from .tessera_campaign_selection import (
    UNITS_SCHEMA_V3, EXPERT_PARTITION_SCHEMA,
    selection_priced_units, validate_unit_selection,
)

STAGE = "Tessera campaign"
COST_NAME = "cost.pkl"
MANIFEST_NAME = "cost.anchors.json"
PARTS_NAME = MANIFEST_NAME + ".parts"
REFERENCES_NAME = "hessian_capture.references.json"
SCALES_NAME = "input_scales.safetensors"

SCHEMA = "prismaquant.tessera_campaign_cost.v1"


#: The currency every row is measured or interpolated in.  Named, because the
#: allocator's cost-source precedence reads a field, not a convention.
CURRENCY = "output_mse_under_route_activation_contract"


@dataclass(frozen=True)
class CampaignAnchor:
    """One measured (unit, family, rung) point."""

    qname: str
    family: str
    format_name: str
    body_rate_q256: int
    dloss: float
    dloss_stderr: float
    memory_bytes: int
    bits_per_param: float
    activation_contract: str
    activation_quantized: bool
    wire_bytes: int
    #: Encoding wall time; joined batches apportion it equally among units.
    seconds: float
    #: Was a Hessian actually applied to these bytes? Admission comes from the
    #: producer's ActivationSource settings for this rung's scale plane, via
    #: rung_accepts_hessian, not a campaign-owned plane roster. Stamped per
    #: measured row so weights-only and H-aware results cannot be conflated.
    hessian_applied: bool = False
    #: The static NVFP4 ``input_global_scale`` this anchor's A side was scored
    #: under, when the rung's route executes the static UE4M3 contract; None
    #: on every other route.  Carried per row so the price's activation
    #: identity survives into the cost table beside the Hessian identity.
    input_global_scale: "float | None" = None
    encoding_batch_size: int = 1


FAMILY_RESTRICTION_SCHEMA = "prismaquant.tessera_campaign_family_restriction.v1"


def parse_family_restriction(value):
    """Canonical opt-in pricing scope; it grants no reader or serving support."""
    if value is None:
        return None
    from .tessera_formats import get_tessera_family

    if isinstance(value, str):
        value = strict_json_loads(value, duplicate=lambda key: ValueError(
            f"family restriction repeats field {key!r}"))
    if not isinstance(value, Mapping) or set(value) != {"schema", "dense", "routed_moe"}:
        raise ValueError("family restriction requires exactly schema, dense and routed_moe")
    if value["schema"] != FAMILY_RESTRICTION_SCHEMA:
        raise ValueError("family restriction has an unsupported schema")
    result = {"schema": FAMILY_RESTRICTION_SCHEMA}
    for structure in ("dense", "routed_moe"):
        names = value[structure]
        if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
            raise ValueError(f"family restriction {structure} requires a nonempty family list")
        if len(set(names)) != len(names):
            raise ValueError(f"family restriction {structure} repeats a family")
        for name in names:
            try:
                family = get_tessera_family(name)
            except (ValueError, KeyError) as exc:
                raise ValueError(f"family restriction has unknown family {name!r}") from exc
            if family.name != name:
                raise ValueError(f"family restriction requires canonical family name {name!r}")
        result[structure] = sorted(names)
    return result


class StackSampleError(ValueError):
    """A routed-stack sampling record cannot be turned into an honest row."""


@dataclass(frozen=True)
class StackExpertSample:
    """Which experts of one packed stack were measured, and with what weight.

    Built by the campaign driver from the PROBE row plus its own draw --
    ``stack_sample_from_probe`` is the constructor that reads the profile so no
    caller has to spell a projection split.  Everything here that describes the
    stack (``num_experts``, ``stack_h_trace``, ``h_trace_per_expert``,
    ``packed_experts_module``) is copied from the probe row, so the row this
    record prices and the row the allocator multiplies are the same row.
    """

    #: The packed parameter's qname -- the key the stack row is written under,
    #: and a key the probe already has (e.g.
    #: ``model.layers.18.feed_forward.experts.gate_up_proj``).
    packed_qname: str
    #: ``_packed_experts_module`` from the probe row, copied onto the cost row
    #: so ``tessera_serving_scope.unit_structure_from_stats`` resolves
    #: ``routed_moe`` on its EXISTING packed branch rather than being taught a
    #: new topology (principle 14: the scope may not invent structure).
    packed_experts_module: str
    #: ``gate_up_proj`` / ``down_proj``: the packed parameter, used only to ask
    #: the profile for its projection names.
    packed_param: str
    num_experts: int
    #: The probe row's ``h_trace`` -- the exact multiplier the allocator applies.
    stack_h_trace: float
    #: The probe row's ``h_trace_per_expert``; must sum to ``stack_h_trace``.
    h_trace_per_expert: tuple[float, ...]
    #: Expert ids actually encoded, ascending.
    sampled_experts: tuple[int, ...]
    #: pi_e for every sampled expert. 1.0 marks a certainty stratum (or a
    #: census), which contributes zero sampling variance.
    inclusion_prob: "Mapping[int, float]"
    #: expert id -> the anchor qnames measured for it, one per projection.
    members: "Mapping[int, tuple[str, ...]]"
    #: The draw's seed, and its design name, carried for reproduction.
    seed: int
    design: str = "pps_wor"
    #: The second tier of a two-tier schedule: the experts that were ALSO
    #: encoded at the rungs the stack was not censused at, from which the
    #: transfer law fits this stack's intercept
    #: (``docs/results/glm_tessera_probe_reduction_regret_2026-09-10.md``
    #: section 5).  Empty means one tier, which is every schedule before it:
    #: every rung is measured on every expert in ``sampled_experts`` or it is
    #: not priced at all.
    transfer_law_experts: tuple[int, ...] = ()

    @property
    def is_census(self) -> bool:
        return len(self.sampled_experts) == self.num_experts and all(
            float(self.inclusion_prob[e]) == 1.0 for e in self.sampled_experts)


def stack_sample_from_probe(
    packed_qname: str,
    probe_row: "Mapping[str, object]",
    profile,
    *,
    sampled_experts,
    inclusion_prob,
    seed: int,
    design: str = "pps_wor",
    transfer_law_experts=(),
) -> StackExpertSample:
    """Build a sampling record from the PROBE row and the model profile.

    The caller supplies only its draw.  Everything structural -- the expert
    count, the Fisher weights, the packed module, and the per-expert member
    qnames -- is read here from the probe row and the profile's packed-expert
    accessors, because a driver-supplied weight that drifts from the probe
    would break the currency invisibly: the allocator would still multiply by
    the probe's ``h_trace``.
    """
    packed_param = str(probe_row.get("_packed_param") or "")
    module = probe_row.get("_packed_experts_module")
    num_experts = int(probe_row.get("num_experts", 0) or 0)
    stack_h = float(probe_row.get("h_trace", 0.0) or 0.0)
    per_expert = probe_row.get("h_trace_per_expert")
    if not isinstance(module, str) or not module:
        raise StackSampleError(
            f"{packed_qname}: probe row has no _packed_experts_module; this is "
            "not a packed routed stack and must not be priced as one")
    if num_experts <= 0:
        raise StackSampleError(f"{packed_qname}: probe row has no num_experts")
    if not isinstance(per_expert, Sequence) or len(per_expert) != num_experts:
        raise StackSampleError(
            f"{packed_qname}: stack pricing requires "
            f"h_trace_per_expert[{num_experts}] from the probe")
    if not packed_param:
        raise StackSampleError(f"{packed_qname}: probe row has no _packed_param")
    if not packed_qname.endswith(packed_param):
        raise StackSampleError(
            f"{packed_qname}: does not end in its own _packed_param "
            f"{packed_param!r}; refusing to guess the stem")
    # The projection split comes from the profile, so a family whose leaves are
    # ``w1/w3/w2`` and one whose leaves are ``gate_proj/up_proj/down_proj`` are
    # both handled without this module knowing either spelling.
    roles = tuple(profile.packed_expert_projection_names(packed_param))
    if not roles:
        raise StackSampleError(
            f"{packed_qname}: profile declares no projections for "
            f"{packed_param!r}")
    stem = packed_qname[: -len(packed_param)].rstrip(".")
    ordered = tuple(sorted(int(e) for e in sampled_experts))
    if len(set(ordered)) != len(ordered):
        raise StackSampleError(f"{packed_qname}: duplicate sampled expert id")
    members = {e: tuple(f"{stem}.{e}.{role}" for role in roles) for e in ordered}
    return StackExpertSample(
        packed_qname=str(packed_qname),
        packed_experts_module=module,
        packed_param=packed_param,
        num_experts=num_experts,
        stack_h_trace=stack_h,
        h_trace_per_expert=tuple(float(v) for v in per_expert),
        sampled_experts=ordered,
        inclusion_prob={int(e): float(p) for e, p in dict(inclusion_prob).items()},
        members=members,
        seed=int(seed),
        design=str(design),
        transfer_law_experts=tuple(sorted(int(e) for e in transfer_law_experts)),
    )


#: ``float32`` machine epsilon.  The probe accumulates the packed row's
#: ``h_trace`` and its ``h_trace_per_expert`` vector from the same raw sums and
#: divides both by the same global token count
#: (``sensitivity_probe.finalize_fisher_stats``), so the two agree exactly in
#: exact arithmetic.  They are stored at ``float32``, which is the only reason
#: they differ at all, so the admissible discrepancy is a property of that
#: dtype and of how many terms are summed -- not a tolerance anyone chose.
#: Measured on ``probe-02``'s two layer-18 packed rows: 2.0e-7 and 6.6e-8
#: relative, against the bound below of 3.8e-6 at E=32.
_FLOAT32_EPS = 1.1920928955078125e-07


def validate_stack_sample(sample: StackExpertSample) -> None:
    """Refuse a record whose weights or probabilities cannot carry an estimate."""
    q = sample.packed_qname
    if len(sample.h_trace_per_expert) != sample.num_experts:
        raise StackSampleError(
            f"{q}: h_trace_per_expert has {len(sample.h_trace_per_expert)} "
            f"entries for {sample.num_experts} experts")
    if any(not math.isfinite(h) or h < 0.0 for h in sample.h_trace_per_expert):
        raise StackSampleError(
            f"{q}: per-expert Fisher weights must be finite and nonnegative")
    total = math.fsum(sample.h_trace_per_expert)
    if not math.isfinite(sample.stack_h_trace) or sample.stack_h_trace <= 0.0:
        raise StackSampleError(f"{q}: probe h_trace must be finite and positive")
    # E terms summed at float32 storage precision: the worst-case accumulated
    # relative error is E * eps, so that IS the bound, computed per stack.
    tolerance = sample.num_experts * _FLOAT32_EPS * sample.stack_h_trace
    if abs(total - sample.stack_h_trace) > tolerance:
        # The whole currency rests on this identity; a row written against a
        # broken one would be multiplied by a number its weights do not sum to.
        raise StackSampleError(
            f"{q}: sum(h_trace_per_expert)={total!r} does not equal the probe "
            f"row's h_trace={sample.stack_h_trace!r}; the stack row's currency "
            "assumes the per-expert weights sum to the multiplier the "
            f"allocator applies (tolerance {tolerance!r} = "
            f"{sample.num_experts} * float32 eps)")
    if not sample.sampled_experts:
        raise StackSampleError(f"{q}: no sampled experts")
    if len(set(sample.sampled_experts)) != len(sample.sampled_experts):
        raise StackSampleError(f"{q}: duplicate sampled expert id")
    for e in sample.sampled_experts:
        if not 0 <= e < sample.num_experts:
            raise StackSampleError(f"{q}: expert id {e} out of range")
        if e not in sample.inclusion_prob:
            raise StackSampleError(f"{q}: expert {e} has no inclusion probability")
    # Older callers carry only sampled probabilities. When the full frame is
    # supplied, validate the unsampled entries too: zero-probability positive
    # contributions and omitted certainty units both bias the stack total.
    for e, probability in sample.inclusion_prob.items():
        if not 0 <= e < sample.num_experts:
            raise StackSampleError(f"{q}: expert id {e} out of range")
        pi = float(probability)
        if pi == 0.0:
            # A zero-probability unit contributes an exactly-zero term ONLY if
            # its own weight is zero (a never-routed expert). Otherwise it is a
            # unit the design can never draw, and HT is biased by exactly its
            # contribution -- silently.
            if sample.h_trace_per_expert[e] != 0.0:
                raise StackSampleError(
                    f"{q}: expert {e} has inclusion probability 0 but Fisher "
                    f"weight {sample.h_trace_per_expert[e]!r}; a unit that "
                    "cannot be drawn biases the estimate")
            continue
        if not 0.0 < pi <= 1.0:
            raise StackSampleError(
                f"{q}: expert {e} inclusion probability {pi!r} outside (0, 1]")
        if pi == 1.0 and e not in sample.sampled_experts:
            raise StackSampleError(f"{q}: certainty expert {e} is absent from the sample")
    if sample.transfer_law_experts:
        law = set(sample.transfer_law_experts)
        if len(law) != len(sample.transfer_law_experts):
            raise StackSampleError(f"{q}: duplicate transfer-law expert id")
        if not law <= set(sample.sampled_experts):
            raise StackSampleError(
                f"{q}: the transfer-law tier names experts the sample does not "
                "measure; a stack's intercept is fitted on experts that were "
                "encoded at BOTH the reference rung and the predicted one")
        if law == set(sample.sampled_experts):
            raise StackSampleError(
                f"{q}: the transfer-law tier is the whole sample, so nothing "
                "would be predicted; drop it rather than declare a law over a "
                "census")
        if len(law) < 2:
            raise StackSampleError(
                f"{q}: {len(law)} transfer-law expert(s); an intercept fitted "
                "on one expert has no residual spread to report")


def _stack_member_weight(sample: StackExpertSample, expert: int, roles: int) -> float:
    """The Fisher weight of ONE measured member of one expert.

    ``h_trace_per_expert[e] / R`` -- the same equal split
    ``expand_packed_expert_rows`` applies, so a census through this path and an
    expansion through that one price the stack identically.
    """
    return float(sample.h_trace_per_expert[expert]) / float(roles)


def _horvitz_thompson_stack(
    sample: StackExpertSample,
    member_dloss: "Mapping[int, float]",
) -> "tuple[float, float, int]":
    """Estimate ``sum_e h_e * mse_e`` and its standard error from the sample.

    ``member_dloss`` maps a sampled expert id to that expert's h-weighted
    contribution ``y_e = sum_roles (h_e/R) * mse_role`` already summed over the
    packed parameter's projections.

    Returns ``(T_hat, stderr, m)`` where ``m`` is the size of the random
    stratum.

    The point estimate is Horvitz-Thompson, ``sum_{e in S} y_e / pi_e``, which
    is unbiased for the stack total under any design with known positive
    inclusion probabilities.

    The VARIANCE is Hartley-Rao's, over the non-certainty stratum only:

        v = m/(m-1) * sum_{e in S_R} (1 - (m-1)/m * pi_e)
                                     * (y_e/pi_e - T_R/m)^2

    A unit with ``pi_e == 1`` is in every possible sample, so it contributes
    exactly zero sampling variance and is excluded from the sum. This is a
    plug-in Hartley-Rao approximation for randomized-order systematic PPS,
    ``pi_i = min(1, c*h_i)``, after removing the take-all stratum. It is not
    an exact design-unbiased variance estimator. Exact Sen-Yates-Grundy would
    require joint inclusion probabilities averaged over the randomized order;
    zero probabilities conditional on one fixed order do not establish zeros
    under the randomized design.  The Hansen-Hurwitz with-replacement form
    is NOT used: it ignores both the finite-population correction and the
    certainty stratum, and overstates the standard error by 25-48% at E=32
    (simulated on the LFM2.5 layer-18 Fisher vector). The approximation was
    conservative in those simulations (+1% to +15%); that is not a guarantee
    for other populations. With equal weights, its expected variance is
    (N-m+1)/(N-m) times the exact SRS variance, reaching twice the exact value
    for m=N-1. The current allocator does not consume this uncertainty field.
    """
    certainty = [e for e in sample.sampled_experts
                 if float(sample.inclusion_prob[e]) == 1.0]
    random = [e for e in sample.sampled_experts
              if 0.0 < float(sample.inclusion_prob[e]) < 1.0]
    total = math.fsum(member_dloss[e] for e in certainty)
    t_random = math.fsum(member_dloss[e] / float(sample.inclusion_prob[e])
                         for e in random)
    total += t_random
    m = len(random)
    if m == 0:
        # Every measured unit was certain: a census, or a fully take-all
        # stratum. The estimate IS the total and the error is a true zero.
        return total, 0.0, 0
    if m < 2:
        # 0.0 already means "no sampling error" on this row's siblings, and a
        # single random draw is not that. The draw plan is supposed to refuse
        # m == 1 before anything is encoded; refusing again here keeps a plan
        # that did not from being laundered into a published zero.
        raise StackSampleError(
            f"{sample.packed_qname}: {m} non-certainty draw(s); a sampling "
            "variance needs at least two, and writing 0.0 would claim the "
            "zero that a census means")
    mean_r = t_random / m
    variance = (m / (m - 1)) * math.fsum(
        (1.0 - ((m - 1) / m) * float(sample.inclusion_prob[e]))
        * (member_dloss[e] / float(sample.inclusion_prob[e]) - mean_r) ** 2
        for e in random)
    return total, math.sqrt(max(variance, 0.0)), m


def _stack_menu(sample: StackExpertSample, menus: "Mapping[str, list]") -> list:
    """The menu the stack row is interpolated over.

    Preferred source is a menu the driver keyed at the packed qname.  Failing
    that the stack inherits its members' menu -- but only if every measured
    member offers the SAME rungs, because a stack is one decision and a menu
    that differs between the experts inside it is not a menu for that decision.
    """
    packed = menus.get(sample.packed_qname)
    if packed:
        return list(packed)
    seen: dict[tuple, list] = {}
    for expert in sample.sampled_experts:
        for member in sample.members[expert]:
            rungs = list(menus.get(member, []))
            key = tuple(sorted(
                (r.family, r.format_name, int(r.body_rate_q256)) for r in rungs))
            seen.setdefault(key, rungs)
    if not seen:
        return []
    if len(seen) > 1:
        raise StackSampleError(
            f"{sample.packed_qname}: the measured experts do not share one "
            f"menu ({len(seen)} distinct rung sets); a packed stack is a "
            "single serving decision and cannot be priced over a menu that "
            "differs between the experts inside it")
    return next(iter(seen.values()))


#: The anchor fields that must agree across every measured member of one stack
#: rung, mapped to the row field they are written to.  Taking member zero's
#: value the way the dense interpolation path takes ``ordered[0]``'s would let
#: one expert scored under a different activation contract, or without the
#: Hessian, disappear into a stack average.
_STACK_UNIFORM_FIELDS = (
    "family", "body_rate_q256", "activation_contract",
    "activation_quantized", "hessian_applied",
)


def _stack_cost_rows(
    sample: StackExpertSample,
    anchors: "Mapping[str, Mapping[str, list]]",
    menus: "Mapping[str, list]",
    hessian_identity: dict,
    refused: list,
    wire_backed: "frozenset[str] | set[str]" = frozenset(),
    transfer_law: "Mapping[str, object] | None" = None,
) -> "tuple[dict[str, dict], object]":
    """Build every measured + interpolated row for one packed stack.

    ``transfer_law`` maps a family to the ``StackTransferLaw`` fitted on the
    OTHER stacks of that family.  It is consulted only for a rung that covers
    exactly ``sample.transfer_law_experts`` -- the second tier of a two-tier
    schedule -- and its rows are model predictions, written with
    ``PROVENANCE_INTERPOLATED`` and a ``transfer_law`` block and never with
    the Horvitz-Thompson fields, which describe a draw this value did not come
    from.  ``None`` (the default) leaves every schedule that measured one tier
    behaving exactly as before.
    """
    from .tessera_rate_surface import (
        PROVENANCE_INTERPOLATED, PROVENANCE_MEASURED, TesseraRateSurface,
        predict_stack_rates,
    )

    validate_stack_sample(sample)
    q = sample.packed_qname
    roles = len(sample.members[sample.sampled_experts[0]])
    if roles <= 0:
        raise StackSampleError(f"{q}: sampled experts have no measured members")

    # (format_name) -> {expert -> [anchor, ...]}, one anchor per member.
    by_format: dict[str, dict[int, list]] = {}
    for expert in sample.sampled_experts:
        if len(sample.members[expert]) != roles:
            raise StackSampleError(
                f"{q}: expert {expert} contributes "
                f"{len(sample.members[expert])} members, expert "
                f"{sample.sampled_experts[0]} contributes {roles}")
        for member in sample.members[expert]:
            member_anchors = anchors.get(member)
            if not member_anchors:
                raise StackSampleError(
                    f"{q}: sampled member {member} has no measured anchors")
            for family_anchors in member_anchors.values():
                for anchor in family_anchors:
                    by_format.setdefault(
                        anchor.format_name, {}).setdefault(expert, []).append(anchor)

    rows: dict[str, dict] = {}
    measured_by_family: dict[str, list[tuple[int, float, float]]] = {}
    #: family -> [(format_name, rung), ...] measured on the law draw only.
    law_tier: dict[str, list[tuple[str, int]]] = {}
    law_experts = set(sample.transfer_law_experts)
    for format_name in sorted(by_format):
        per_expert = by_format[format_name]
        missing = [e for e in sample.sampled_experts
                   if len(per_expert.get(e, ())) != roles]
        if missing:
            covered = {e for e in sample.sampled_experts
                       if len(per_expert.get(e, ())) == roles}
            if law_experts and covered == law_experts:
                # The second tier of a two-tier schedule: this rung was encoded
                # on the transfer-law draw only, by design.  It is priced by
                # the law below, not by HT -- the drawn experts here fit an
                # intercept, they do not stand in for the stack.
                contributing = [a for e in sorted(covered) for a in per_expert[e]]
                families = {a.family for a in contributing}
                if len(families) != 1:
                    raise StackSampleError(
                        f"{q}/{format_name}: transfer-law tier spans families "
                        f"{sorted(families)}")
                rates = {int(a.body_rate_q256) for a in contributing}
                if len(rates) != 1:
                    raise StackSampleError(
                        f"{q}/{format_name}: transfer-law tier spans rungs "
                        f"{sorted(rates)}")
                law_tier.setdefault(next(iter(families)), []).append(
                    (format_name, rates.pop()))
                continue
            # A rung measured on only some of the drawn experts is not a rung
            # this sample can price: HT needs every drawn unit's y_e.
            refused.append({
                "qname": q, "format_name": format_name,
                "reason": "stack_rung_incomplete_over_sample",
                "detail": (f"{len(missing)} of {len(sample.sampled_experts)} "
                           "sampled experts lack a full set of member anchors"),
                "missing_experts": missing,
            })
            continue
        contributing = [a for e in sample.sampled_experts for a in per_expert[e]]
        uniform = {}
        for field in _STACK_UNIFORM_FIELDS:
            values = {getattr(a, field) for a in contributing}
            if len(values) != 1:
                raise StackSampleError(
                    f"{q}/{format_name}: measured members disagree on {field} "
                    f"({sorted(map(repr, values))}); a stack row may not "
                    "average measurements taken under different contracts")
            uniform[field] = next(iter(values))
        member_dloss = {
            e: math.fsum(_stack_member_weight(sample, e, roles) * float(a.dloss)
                         for a in per_expert[e])
            for e in sample.sampled_experts
        }
        total, stderr, n_random = _horvitz_thompson_stack(sample, member_dloss)
        h_stack = float(sample.stack_h_trace)
        rows[format_name] = {
            # The h-weighted MEAN expert MSE -- see the section header. The
            # allocator multiplies this by the PROBE row's h_trace, which the
            # per-expert weights sum to, so 0.5*h_stack*output_mse reproduces
            # the stack's summed per-expert predicted dloss.
            "output_mse": total / h_stack,
            "output_mse_measured": True,
            # A rung encoded on EVERY expert of the stack with certainty is a
            # census, not a sample: the value is the total, the HT variance is
            # an exact zero, and nothing about it is an estimate.  It therefore
            # carries the same ``cost_source`` a dense measured row does
            # (#495 part 3).  That removes one refusal in
            # ``tessera_joint_aura.load_measured_anchor_input`` but does not
            # make a stack row consumable there: that reader matches the cost
            # roster against the census's SOURCE units and a stack's key is the
            # packed qname, which is debt D35(iii).  A rung covering only the
            # draw keeps the sampled spelling.
            "cost_source": ("tessera_campaign_measured" if sample.is_census
                            else "tessera_campaign_measured_stack_sample"),
            "currency": CURRENCY,
            "tessera_provenance": PROVENANCE_MEASURED,
            "tessera_family": uniform["family"],
            "tessera_body_rate_q256": uniform["body_rate_q256"],
            "activation_quantized": uniform["activation_quantized"],
            "activation_contract": uniform["activation_contract"],
            # The Horvitz-Thompson standard error, in the SAME currency as
            # ``output_mse``.  REPORTED, NOT CONSUMED: the allocator's UCB
            # hedge (``allocator_candidates._super_item_ucb_hedge``) skips
            # rows priced from ``output_mse``, so no DP behaviour depends on
            # this field today.  It is written because a sampled price whose
            # sampling error is nowhere on the row is a sampled price nothing
            # can audit.
            "dloss_stderr": stderr / h_stack,
            "dloss_stderr_currency": CURRENCY,
            "dloss_stderr_consumed_by_allocator": False,
            # Copied from the PROBE row so the explicit Tessera serving scope
            # resolves ``routed_moe`` on ``unit_structure_from_stats``'s
            # existing packed branch. The scope is not taught a new topology;
            # it is handed the one the probe already recorded.
            "_packed_experts_module": sample.packed_experts_module,
            "num_experts": sample.num_experts,
            "encode_seconds": math.fsum(float(a.seconds) for a in contributing),
            "hessian_identity": {
                **hessian_identity, "applied": bool(uniform["hessian_applied"]),
            },
            "sampled_experts": {
                "design": sample.design,
                "seed": sample.seed,
                "n_experts": sample.num_experts,
                "n_sampled": len(sample.sampled_experts),
                "n_random_stratum": n_random,
                "experts": list(sample.sampled_experts),
                "inclusion_prob": {int(e): float(sample.inclusion_prob[e])
                                   for e in sample.sampled_experts},
                "packed_param": sample.packed_param,
                "projections_per_expert": roles,
                "estimator": "horvitz_thompson",
                "variance_estimator": (
                    "census" if n_random == 0 else "hartley_rao"),
                "h_trace_stack": h_stack,
                "h_trace_per_sampled_expert": {
                    int(e): float(sample.h_trace_per_expert[e])
                    for e in sample.sampled_experts},
                # The per-member evidence the stack row was estimated from,
                # kept on the row because the members are NOT cost keys: a
                # stack price whose members are unreadable is unauditable.
                "members": {
                    int(e): [
                        {"qname": a.qname, "dloss": float(a.dloss),
                         "wire_bytes": int(a.wire_bytes),
                         "memory_bytes": int(a.memory_bytes),
                         "input_global_scale": (
                             None if a.input_global_scale is None
                             else float(a.input_global_scale))}
                        for a in per_expert[e]]
                    for e in sample.sampled_experts},
            },
        }
        # No ``wire_bytes`` and no ``input_global_scale`` scalar on a stack
        # row, deliberately.  Both are per-expert facts: a sample has neither a
        # stack wire nor one A-side scale, and the per-member values are on the
        # row above.  ``tessera_menu.priced_static_scales`` will therefore find
        # no scale for a selected W4A4 stack and the export lane will refuse it
        # by name -- which is the correct refusal until the driver calibrates
        # a scale for every expert, sampled or not (reported, not papered over).
        measured_by_family.setdefault(uniform["family"], []).append(
            (int(uniform["body_rate_q256"]), total / h_stack, stderr / h_stack))

    # Interpolation, on the STACK's own anchors: the same monotone surface the
    # dense path uses, fitted to HT estimates rather than to one unit's
    # measurements, and refusing to extrapolate for the same reason.
    surfaces: dict[str, object] = {}
    measured_names = set(rows)
    for family, points in measured_by_family.items():
        if family in law_tier:
            # A two-tier family has ONE censused rung, so there is no bracket
            # to interpolate between and the transfer law below is what prices
            # its other rungs.  Skipping here rather than letting the surface
            # refuse keeps a designed schedule out of the refusal record.
            continue
        ordered = sorted(points)
        try:
            surface = TesseraRateSurface(
                unit_name=q, family=family, layout="tight", currency=CURRENCY,
                anchor_q256=tuple(p[0] for p in ordered),
                anchor_dloss=tuple(p[1] for p in ordered),
                anchor_stderr=tuple(p[2] for p in ordered),
            )
        except Exception as exc:
            refused.append({
                "qname": q, "family": family,
                "reason": "non_interpolable_anchors",
                "detail": str(exc),
                "anchor_q256": [p[0] for p in ordered],
                "anchor_dloss": [p[1] for p in ordered],
            })
            continue
        surfaces[family] = surface
        if sample.is_census and all(
                member in wire_backed for members in sample.members.values()
                for member in members):
            # A full census handed to the cached-wire exporter can select
            # only rungs with actual member bytes. Keep its surface for
            # diagnostics, as the dense path does, but never price an
            # interpolated rung as though the corresponding wires existed.
            continue
        low, high = surface.q256_range
        template = next(r for r in rows.values()
                        if r["tessera_family"] == family)
        for rung in _stack_menu(sample, menus):
            if rung.family != family or rung.format_name in measured_names:
                continue
            if not low <= rung.body_rate_q256 <= high:
                continue
            rows[rung.format_name] = {
                "output_mse": surface.predict(rung.body_rate_q256),
                "output_mse_measured": False,
                "cost_source": "tessera_campaign_interpolated",
                "currency": CURRENCY,
                "tessera_provenance": PROVENANCE_INTERPOLATED,
                "tessera_family": family,
                "tessera_body_rate_q256": rung.body_rate_q256,
                "activation_contract": rung.admission.activation_contract,
                "_packed_experts_module": sample.packed_experts_module,
                "num_experts": sample.num_experts,
                "hessian_identity": {
                    **hessian_identity,
                    "applied": bool(template["hessian_identity"]["applied"]),
                },
                # An interpolated stack row inherits the sample its bracketing
                # anchors were estimated from; it is a prediction ABOUT that
                # sample, so the sample travels with it -- but only the fields
                # that describe the DRAW.  Built from an allowlist rather than
                # by subtracting ``members``, because subtraction is how
                # ``estimator: horvitz_thompson`` used to reach a row whose
                # value no estimator produced (#495 part 3).
                "sampled_experts": {
                    **_stack_draw_description(template["sampled_experts"]),
                    "interpolated_from": sorted(measured_names),
                },
            }

    # The transfer-law tier: rungs encoded on the law draw only, predicted for
    # every expert from its own censused reference value.  Model predictions,
    # so they carry a ``transfer_law`` block and none of the sampling fields.
    for family, entries in sorted(law_tier.items()):
        pair = None if transfer_law is None else transfer_law.get(family)
        if pair is None:
            for format_name, rung in entries:
                refused.append({
                    "qname": q, "family": family, "format_name": format_name,
                    "reason": "stack_transfer_law_absent",
                    "detail": (f"rung {rung} was encoded on the transfer-law "
                               "draw only and no law was fitted for this "
                               "family; the rung is left unpriced rather than "
                               "estimated from a partial draw"),
                })
            continue
        record, law, reference_format = pair
        try:
            prediction = predict_stack_rates(record, law)
        except Exception as exc:
            for format_name, rung in entries:
                refused.append({
                    "qname": q, "family": family, "format_name": format_name,
                    "reason": "stack_transfer_law_refused",
                    "detail": str(exc),
                })
            continue
        h_stack = float(sample.stack_h_trace)
        template = rows[reference_format]
        law_block = law.as_dict()
        for format_name, rung in entries:
            if format_name in rows:
                raise StackSampleError(
                    f"{q}/{format_name}: a transfer-law rung may not overwrite "
                    "a measured row")
            if rung not in prediction["predicted"]:
                refused.append({
                    "qname": q, "family": family, "format_name": format_name,
                    "reason": "stack_transfer_law_rung_unfitted",
                    "detail": (f"rung {rung} is not in the fitted law "
                               f"{list(law.target_q256)}"),
                })
                continue
            rows[format_name] = {
                "output_mse": prediction["predicted"][rung] / h_stack,
                "output_mse_measured": False,
                "cost_source": "tessera_campaign_interpolated",
                "currency": CURRENCY,
                "tessera_provenance": PROVENANCE_INTERPOLATED,
                "tessera_family": family,
                "tessera_body_rate_q256": rung,
                "activation_contract": template["activation_contract"],
                "activation_quantized": template["activation_quantized"],
                "_packed_experts_module": sample.packed_experts_module,
                "num_experts": sample.num_experts,
                "hessian_identity": {
                    **hessian_identity,
                    "applied": bool(template["hessian_identity"]["applied"]),
                },
                # Everything the prediction rests on, on the row: the pooled
                # slopes and which stacks they were pooled over, this stack's
                # own intercepts, the experts that fitted them, how many there
                # were, and the residual spread of the pooled fit.  A reader
                # can refit it; that is the point of writing it down.
                "transfer_law": {
                    **law_block,
                    "predicted_q256": rung,
                    "reference_format_name": reference_format,
                    "intercept": {p: float(v) for p, v in sorted(
                        prediction["intercept"][rung].items())},
                    "sample_experts": list(prediction["sample_experts"]),
                    "n": int(prediction["intercept_sample_size"]),
                    "model_error": prediction["model_error"][rung],
                },
                # The draw that fitted the intercept, described as a draw and
                # nothing more.  No ``estimator`` and no ``dloss_stderr``: this
                # value came from a regression, and a sampling error would be a
                # claim about a design that did not produce it.
                "sampled_experts": {
                    **_stack_draw_description(template["sampled_experts"]),
                    "transfer_law_experts": sorted(law_experts),
                },
            }
    return rows, surfaces


#: The fields of a stack row's ``sampled_experts`` block that describe the
#: DRAW rather than an estimate made from it.  ``estimator``,
#: ``variance_estimator`` and ``n_random_stratum`` are deliberately absent:
#: they name a Horvitz-Thompson computation, and a row whose value no
#: estimator produced may not claim one (#495 part 3).
_STACK_DRAW_FIELDS = (
    "design", "seed", "n_experts", "n_sampled", "experts", "inclusion_prob",
    "packed_param", "projections_per_expert", "h_trace_stack",
    "h_trace_per_sampled_expert",
)


def _stack_draw_description(block: "Mapping[str, object]") -> dict:
    """The draw's own fields, by allowlist -- never by subtraction."""
    return {field: block[field] for field in _STACK_DRAW_FIELDS if field in block}


def _stack_rate_evidence(sample: StackExpertSample, anchors, refused: list):
    """One two-tier stack's evidence, per family, as a ``StackRateSample``.

    A family qualifies only when exactly one of its rungs was encoded on every
    expert of the frame (the census tier, which every prediction regresses on)
    and at least one other was encoded on exactly the transfer-law draw.  A
    family that fails either test is recorded and skipped: the rungs it does
    hold are still priced by the measured path.

    Returns ``{family: (record, reference_format_name)}``.
    """
    from .tessera_rate_surface import StackRateSample

    if not sample.transfer_law_experts:
        return {}
    q = sample.packed_qname
    roles = len(sample.members[sample.sampled_experts[0]])
    projections = tuple(name.rsplit(".", 1)[-1]
                        for name in sample.members[sample.sampled_experts[0]])
    law_experts = tuple(sorted(sample.transfer_law_experts))
    # (family, rung) -> {expert: {projection: mse}} plus the format name.
    seen: dict[tuple[str, int], dict] = {}
    names: dict[tuple[str, int], str] = {}
    for expert in sample.sampled_experts:
        members = sample.members[expert]
        if len(members) != roles:
            return {}
        for index, member in enumerate(members):
            for family_anchors in (anchors.get(member) or {}).values():
                for anchor in family_anchors:
                    key = (anchor.family, int(anchor.body_rate_q256))
                    names.setdefault(key, anchor.format_name)
                    seen.setdefault(key, {}).setdefault(
                        expert, {})[projections[index]] = float(anchor.dloss)
    frame = set(sample.sampled_experts)
    law_set = set(law_experts)
    records: dict[str, tuple] = {}
    for family in sorted({key[0] for key in seen}):
        census = [rung for (fam, rung), rows in sorted(seen.items())
                  if fam == family
                  and {e for e, row in rows.items() if len(row) == roles} == frame]
        law_rungs = [rung for (fam, rung), rows in sorted(seen.items())
                     if fam == family
                     and {e for e, row in rows.items() if len(row) == roles} == law_set]
        if not law_rungs:
            continue
        if len(census) != 1:
            refused.append({
                "qname": q, "family": family,
                "reason": "stack_transfer_law_reference_missing",
                "detail": (f"{len(census)} rung(s) of family {family} cover the "
                           "whole frame; exactly one is needed, because every "
                           "expert is predicted from its own measured "
                           "reference value"),
                "census_rungs": census,
            })
            continue
        reference = census[0]
        try:
            record = StackRateSample(
                stack=q, projections=projections,
                experts=tuple(sample.sampled_experts),
                reference_q256=reference,
                reference_mse=seen[(family, reference)],
                sampled_experts=law_experts,
                sampled_mse={rung: {e: seen[(family, rung)][e] for e in law_experts}
                             for rung in law_rungs},
                weights={e: _stack_member_weight(sample, e, roles)
                         for e in sample.sampled_experts},
                currency=CURRENCY)
        except Exception as exc:
            refused.append({
                "qname": q, "family": family,
                "reason": "stack_transfer_law_refused",
                "detail": str(exc),
            })
            continue
        records[family] = (record, names[(family, reference)])
    return records


def _fit_stack_transfer_laws(samples, anchors, refused: list) -> dict:
    """Fit one pooled law per family, leaving out the stack it will predict.

    The slope pools every OTHER stack of the same family, which is exactly the
    hold-out the study fitted every reported number under; a stack whose family
    has no other two-tier stack gets no law, and its unmeasured rungs are left
    unpriced rather than predicted from their own evidence.

    Returns ``{packed_qname: {family: (record, law, reference_format_name)}}``.
    """
    from .tessera_rate_surface import fit_stack_transfer_law

    evidence = {}
    for packed_qname, sample in sorted(samples.items()):
        found = _stack_rate_evidence(sample, anchors, refused)
        if found:
            evidence[packed_qname] = found
    by_family: dict[str, dict[str, object]] = {}
    for packed_qname, families in evidence.items():
        for family, (record, _) in families.items():
            by_family.setdefault(family, {})[packed_qname] = record
    laws: dict[str, dict[str, tuple]] = {}
    for packed_qname, families in evidence.items():
        for family, (record, reference_format) in families.items():
            population = by_family[family]
            if len(population) < 2:
                refused.append({
                    "qname": packed_qname, "family": family,
                    "reason": "stack_transfer_law_population_too_small",
                    "detail": (f"family {family} has {len(population)} two-tier "
                               "stack(s); a pooled slope needs at least one "
                               "other stack to hold this one out against"),
                })
                continue
            try:
                law = fit_stack_transfer_law(population, hold_out=packed_qname)
            except Exception as exc:
                refused.append({
                    "qname": packed_qname, "family": family,
                    "reason": "stack_transfer_law_fit_refused",
                    "detail": str(exc),
                })
                continue
            laws.setdefault(packed_qname, {})[family] = (
                record, law, reference_format)
    return laws


def canonical_refusals(refusals: Sequence[dict]) -> list[dict]:
    """Order diagnostic records identically for a monolith and a shard union.

    A surface refusal names a family; an incomplete sampled rung names a
    format. Both are legitimate records. The full canonical record breaks
    ties without depending on dictionary or shard insertion order.
    """
    return sorted(refusals, key=lambda entry: (
        entry["qname"], entry.get("family", ""), entry.get("format_name", ""),
        DIRECT_ASCII_STRICT.text(entry)))


def campaign_cost_payload(
    anchors: Mapping[str, Mapping[str, list[CampaignAnchor]]],
    menus: Mapping[str, list],
    *,
    loo: Mapping[str, Mapping[str, dict]],
    provenance: dict,
    wire_backed: "frozenset[str] | set[str]" = frozenset(),
    stack_samples: "Mapping[str, StackExpertSample] | None" = None,
) -> dict:
    """Turn measured anchors plus a legal menu into a cost payload.

    Measured rungs keep their measurement and are marked so; every other legal
    rung inside the measured envelope is interpolated and marked so; a rung
    outside the envelope is **omitted**, because ``TesseraRateSurface.predict``
    refuses to extrapolate and a menu row the surface will not price is a row
    nothing measured.

    ``wire_backed`` names the units whose priced wire IS the exported wire --
    the producer-projected packed experts, which the export lane hands to the
    exporter as cached bytes rather than re-encoding.  For those, an
    interpolated row would price a rung that has no bytes to ship, so they get
    measured rows only: the allocator can select for them exactly the rungs a
    wire exists for (priced == written; PrismaQuant #183).

    ``stack_samples`` maps a packed expert parameter's qname to the
    ``StackExpertSample`` describing which of its experts were measured.  Each
    one collapses its members' per-expert anchors into ONE stack-level row per
    (family, rung) at the packed qname -- the key the probe already has, so the
    row carries the packed topology the serving scope needs and the ``h_trace``
    the allocator will multiply.  The members are then NOT top-level cost keys
    (their measurements ride along on the stack row's ``sampled_experts``
    block), which is what stops the same experts being priced twice: once as a
    stack and once as themselves.
    """
    from .tessera_rate_surface import (
        PROVENANCE_INTERPOLATED, PROVENANCE_MEASURED, TesseraRateSurface,
    )

    # One identity for every row this payload writes, so a cost table that is
    # half H-aware and half weights-only is detectable downstream instead of
    # being an invisible merge of two encoders -- the exact shape of the
    # encoder-drift bug this project has already paid for once.
    _prov = dict(provenance.get("provenance", {}))
    _h = dict(_prov.get("hessian", {})) if isinstance(
        _prov.get("hessian"), dict) else {}
    hessian_identity = {
        "supplied": bool(_h.get("supplied", False)),
        # The legacy spelling, kept because ``assert_uniform_hessian_identity``
        # compares tables written before Tessera's triple existed.
        "text_sha": _h.get("text_sha"),
        "token_count": _h.get("token_count"),
        # Tessera's own required triple, so a row can be checked against the
        # ActivationSource that would encode it.
        "text_sha256": _h.get("text_sha256"),
        "fit_ids_sha256": _h.get("fit_ids_sha256"),
        "fit_tokens": _h.get("fit_tokens"),
        # The content digest of the capture written for the export leg, so
        # the allocation binds to the payload and not only to the draw's
        # triple (RobTand/prismaquant#204); None on a weights-only campaign.
        "capture_sha256": _h.get("capture_sha256"),
        **({'reference_binding':dict(_h['reference_binding'])} if _h.get('reference_binding') is not None else {}),
        "kwarg": tuple(_h.get("kwargs", ())) or _h.get("kwarg"),
    }

    costs: dict[str, dict[str, dict]] = {}
    formats: set[str] = set()
    surfaces: dict[str, dict[str, TesseraRateSurface]] = {}
    refused: list[dict] = []
    # Every unit that is a MEMBER of a sampled stack, and the stack that owns
    # it. A member never becomes its own cost key: the DP would otherwise see
    # the same experts twice -- once inside the packed stack row and once as
    # standalone units -- and the union-find promotion cannot merge what it
    # cannot see as one group.
    samples: dict[str, StackExpertSample] = dict(stack_samples or {})
    member_owner: dict[str, str] = {}
    for packed_qname, sample in samples.items():
        if sample.packed_qname != packed_qname:
            raise StackSampleError(
                f"{packed_qname}: sampling record names "
                f"{sample.packed_qname!r}")
        if packed_qname in anchors:
            raise StackSampleError(
                f"{packed_qname}: the packed stack itself carries measured "
                "anchors; a stack that was measured whole is not a sample and "
                "must not also be estimated from one")
        for expert, members in sample.members.items():
            for member in members:
                owner = member_owner.setdefault(member, packed_qname)
                if owner != packed_qname:
                    raise StackSampleError(
                        f"{member}: claimed by both {owner} and "
                        f"{packed_qname}; a measured expert belongs to exactly "
                        "one stack")
    for qname, by_family in anchors.items():
        if qname in member_owner:
            continue
        rows: dict[str, dict] = {}
        measured_names: set[str] = set()
        for family, family_anchors in by_family.items():
            ordered = sorted(family_anchors, key=lambda a: a.body_rate_q256)
            for anchor in ordered:
                rows[anchor.format_name] = {
                    # ``output_mse`` ONLY, deliberately no ``predicted_dloss``.
                    # In this codebase ``output_mse`` is a raw MSE and
                    # ``predicted_dloss`` is already the 1/2*h_trace*mse
                    # product; writing the MSE into both fields would have the
                    # weight-only branch price a Tessera rung ~h_trace/2 times
                    # low against every other format on the menu. Leaving the
                    # field out puts these rows on the same branch, in the same
                    # currency, as production_render_cost's.
                    "output_mse": anchor.dloss,
                    "output_mse_measured": True,
                    "cost_source": "tessera_campaign_measured",
                    "currency": CURRENCY,
                    "tessera_provenance": PROVENANCE_MEASURED,
                    "tessera_family": family,
                    "tessera_body_rate_q256": anchor.body_rate_q256,
                    "activation_quantized": anchor.activation_quantized,
                    "activation_contract": anchor.activation_contract,
                    "wire_bytes": anchor.wire_bytes,
                    "encode_seconds": anchor.seconds,
                    "encode_seconds_accounting": ("unit" if anchor.encoding_batch_size == 1
                        else "batch_wall_time_divided_by_batch_size"),
                    "encoding_batch_size": anchor.encoding_batch_size,
                    # The static A-side scale this row was scored under, when
                    # its route executes the static NVFP4 contract; the value
                    # identity the export's scale file must reproduce.
                    **({"input_global_scale": float(anchor.input_global_scale)}
                       if anchor.input_global_scale is not None else {}),
                    "hessian_identity": {
                        **hessian_identity, "applied": bool(anchor.hessian_applied),
                    },
                }
                measured_names.add(anchor.format_name)
                formats.add(anchor.format_name)
            try:
                surface = TesseraRateSurface(
                    unit_name=qname,
                    family=family,
                    layout="tight",
                    currency=CURRENCY,
                    anchor_q256=tuple(a.body_rate_q256 for a in ordered),
                    anchor_dloss=tuple(a.dloss for a in ordered),
                    anchor_stderr=tuple(a.dloss_stderr for a in ordered),
                )
            except Exception as exc:
                # A non-monotone anchor set is a measurement fact, and the
                # surface refuses to launder it into a cost.  Record it and
                # keep the measured rows; the family is then priced only where
                # it was measured, which is the honest reduction.
                refused.append({
                    "qname": qname, "family": family,
                    "reason": "non_interpolable_anchors",
                    "detail": str(exc),
                    "anchor_q256": [a.body_rate_q256 for a in ordered],
                    "anchor_dloss": [a.dloss for a in ordered],
                })
                continue
            surfaces.setdefault(qname, {})[family] = surface
            if qname in wire_backed:
                # Measured rows only: see the docstring.  The surface is still
                # built so leave-one-out reporting covers these units.
                continue
            low, high = surface.q256_range
            for rung in menus.get(qname, []):
                if rung.family != family or rung.format_name in measured_names:
                    continue
                if not low <= rung.body_rate_q256 <= high:
                    continue
                rows[rung.format_name] = {
                    # Same currency as the measured rows above, and for the
                    # same reason: no ``predicted_dloss`` field.
                    "output_mse": surface.predict(rung.body_rate_q256),
                    "output_mse_measured": False,
                    "cost_source": "tessera_campaign_interpolated",
                    "currency": CURRENCY,
                    "tessera_provenance": PROVENANCE_INTERPOLATED,
                    "tessera_family": family,
                    "tessera_body_rate_q256": rung.body_rate_q256,
                    "activation_contract": rung.admission.activation_contract,
                    # An interpolated row inherits the applicability of the
                    # anchors it was interpolated between; they share a family,
                    # so they share a scale plane and therefore an answer --
                    # and, on a static-contract route, a unit-level A scale.
                    **({"input_global_scale": float(ordered[0].input_global_scale)}
                       if ordered[0].input_global_scale is not None else {}),
                    "hessian_identity": {
                        **hessian_identity,
                        "applied": bool(ordered[0].hessian_applied),
                    },
                }
                formats.add(rung.format_name)
        if rows:
            costs[qname] = rows
    laws = _fit_stack_transfer_laws(samples, anchors, refused)
    for packed_qname, sample in sorted(samples.items()):
        stack_rows, stack_surfaces = _stack_cost_rows(
            sample, anchors, menus, hessian_identity, refused,
            wire_backed=wire_backed, transfer_law=laws.get(packed_qname))
        if not stack_rows:
            continue
        costs[packed_qname] = stack_rows
        formats.update(stack_rows)
        if stack_surfaces:
            surfaces.setdefault(packed_qname, {}).update(stack_surfaces)
    payload = dict(provenance)
    payload.update({
        "schema": SCHEMA,
        "costs": costs,
        "formats": sorted(formats),
        "currency": CURRENCY,
        "leave_one_anchor_out": {
            q: {f: dict(v) for f, v in by_f.items()} for q, by_f in loo.items()
        },
        "non_interpolable": canonical_refusals(refused),
    })
    # The attested objective this table prices (re-vet R2; read for reuse by
    # `cost_currency.require_run_currency`, never from the environment). Every
    # row here is an `output_mse` under the route's activation contract --
    # the render-score objective -- so the stamp is unconditional: a caller
    # claim of any other mode would launder cross-currency rows into that
    # run's knapsack (RobTand/prismaquant#127).
    from .cost_currency import RENDER_SCORE_COST_MODE

    nested = payload.get("provenance")
    if not isinstance(nested, dict):
        nested = {}
        payload["provenance"] = nested
    nested["cost_mode"] = RENDER_SCORE_COST_MODE
    return payload


#: The calibration census ``--calibration-census`` reads and ``--census-out``
#: writes: the per-unit calibration row counts of the whole priced scope.
CENSUS_SCHEMA = "prismaquant.tessera_campaign_census.v1"


def _keystream(seed: int, label: str):
    """A stable, dependency-free stream of uniforms in [0, 1).

    SHA-256 in counter mode rather than :mod:`random`, because the draw is
    stamped into a checkpoint identity: a sample that changed when CPython
    changed its sampler would silently re-key every journal it appears in.
    """
    counter = 0
    while True:
        digest = bytes.fromhex(text_sha256hex(f"{int(seed)}:{label}:{counter}"))
        for offset in range(0, 32, 8):
            yield int.from_bytes(digest[offset:offset + 8], "big") / float(1 << 64)
        counter += 1


def _permute(names: "Sequence[str]", stream) -> list[str]:
    """Fisher-Yates over ``names`` from ``stream``."""
    order = list(names)
    for index in range(len(order) - 1, 0, -1):
        pick = int(next(stream) * (index + 1))
        pick = min(pick, index)
        order[index], order[pick] = order[pick], order[index]
    return order


def draw_stack_sample(weights: "Mapping[str, float]", n: int, *,
                      seed: int, stack: str) -> dict:
    """A fixed-size PPS draw over one stack's experts, with exact ``pi``.

    Randomized systematic (Madow) probability-proportional-to-size sampling
    without replacement, with a take-all stratum.  Three parts, each of which
    is load-bearing:

    **Proportional to what.** The stack total the allocator needs is
    ``sum_e h_e * mse_e``.  ``h_e`` is free (the probe already has it) and
    ``mse_e`` costs an encode, so drawing proportional to ``h_e`` leaves only
    the spread of ``mse_e`` in the estimator's variance.  On LFM2.5 layer 18
    the per-expert ``h_trace`` spans 9.6e4 to 7.7e6 (CV ~1.0) while the
    cross-expert spread of ``mse`` at a fixed rung is CV 0.33-0.55, so this is
    where the variance is. A uniform draw with the correct Horvitz-Thompson
    weights remains unbiased, but carries the product's spread. The planner
    requires Fisher weights and provides no uniform fallback.

    **The take-all stratum.** With ``h`` this dispersed, ``n * h_i / sum(h)``
    exceeds one for the largest experts.  Clipping that to one would leave
    ``sum(pi) < n`` and bias the estimate low.  Instead those experts are
    taken with certainty, removed from the frame, and the rest re-solved --
    iterated to a fixed point, which is ``pi_i = min(1, c*h_i)`` for the
    unique ``c`` with ``sum_i min(1, c*h_i) = n``.  The post-conditions
    asserted below (``sum(pi) == n``, every ``pi`` in ``[0, 1]``) are what
    make the Horvitz-Thompson estimator downstream exactly unbiased rather
    than approximately so.

    **The randomized order.** The frame is permuted from the seeded stream
    before the systematic pass.  This is not cosmetic: the practical variance
    estimator for this design (Hartley-Rao, which #290 applies to the prices)
    is derived for randomized-order systematic sampling and is not justified
    under expert-index or ``h``-sorted order.

    Determinism, and what the seed is deliberately NOT: the stream is keyed on
    ``(seed, stack)`` only, never on ``h``.  The probe is not bit-reproducible
    across runs, and systematic sampling is continuous in ``pi``, so a
    re-probed ``h`` perturbs this draw slightly; hashing ``h`` into the seed
    would instead redraw the whole stack on every re-probe.

    An expert with ``h_e == 0`` gets ``pi_e = 0`` and is never encoded, which
    is exact rather than a rounding: under the campaign's Fisher convention a
    token never routed to an expert contributes zero gradient, so its term in
    the total is zero, not merely small.

    Returns the whole draw, including everything a variance estimate needs:
    ``units``, ``inclusion_probability`` (over the full frame), ``certainty``,
    ``permutation``, ``start``, ``size``, ``size_sha256``, ``frame_size``,
    ``random_draws`` and ``method``.
    """
    names = sorted(weights)
    if not names:
        raise RuntimeError(f"stack {stack}: no unit to sample")
    sizes = {name: float(weights[name]) for name in names}
    if any(not math.isfinite(value) for value in sizes.values()):
        raise RuntimeError(f"stack {stack}: size weights must be finite")
    if any(value < 0.0 for value in sizes.values()):
        raise RuntimeError(f"stack {stack}: a size weight is negative")
    digest = text_sha256hex("|".join(f"{name}={sizes[name]!r}" for name in names))
    want = int(n)
    if want < 1:
        raise RuntimeError(f"stack {stack}: --stack-sample must be at least 1")

    frame = [name for name in names if sizes[name] > 0.0]
    zero = [name for name in names if sizes[name] <= 0.0]
    common = {"size": sizes, "size_sha256": digest, "seed": int(seed),
              "stack": str(stack), "frame_size": len(frame),
              "zero_size": zero}
    if want >= len(frame):
        return {**common, "units": list(frame), "method": "census",
                "inclusion_probability": {name: (1.0 if sizes[name] > 0.0 else 0.0)
                                          for name in names},
                "certainty": list(frame), "permutation": list(frame),
                "start": None, "random_draws": 0}

    certainty: list[str] = []
    rest = list(frame)
    remaining = want
    while rest and remaining > 0:
        total = sum(sizes[name] for name in rest)
        over = [name for name in rest if remaining * sizes[name] / total >= 1.0]
        if not over:
            break
        certainty.extend(over)
        rest = [name for name in rest if name not in set(over)]
        remaining -= len(over)
    certainty.sort()

    # A single random draw has no variance estimate. Stamping 0.0 for it would
    # read downstream as "measured exactly", so it refuses here instead --
    # before any GPU second is spent on a sample nothing can put an error bar
    # on. Either take one more expert, or take the stack whole.
    if remaining == 1:
        raise RuntimeError(
            f"stack {stack}: --stack-sample {want} leaves exactly one "
            f"randomly drawn expert after {len(certainty)} certainty unit(s); "
            "a one-draw sample admits no variance estimate. Use "
            f"--stack-sample {want + 1} or price the stack whole.")

    pi = {name: 0.0 for name in names}
    pi.update({name: 1.0 for name in certainty})
    drawn = list(certainty)
    stream = _keystream(seed, f"pps:{stack}")
    permutation = _permute(rest, stream) if rest else []
    start = None
    if remaining > 0 and rest:
        total = sum(sizes[name] for name in rest)
        pi.update({name: remaining * sizes[name] / total for name in rest})
        start = next(stream)
        cumulative = 0.0
        ladder = []
        for name in permutation:
            cumulative += pi[name]
            ladder.append((cumulative, name))
        # Renormalise the last edge so float drift cannot drop the final draw.
        ladder[-1] = (float(remaining), ladder[-1][1])
        step = 0
        for edge, name in ladder:
            while step < remaining and start + step < edge:
                drawn.append(name)
                step += 1
    total_pi = sum(pi.values())
    if abs(total_pi - want) > 1e-9:
        raise RuntimeError(
            f"stack {stack}: inclusion probabilities sum to {total_pi!r}, not "
            f"{want}; the estimator built on them would be biased")
    if len(set(drawn)) != want:
        raise RuntimeError(
            f"stack {stack}: the systematic pass drew {len(set(drawn))} "
            f"distinct units, not {want}")
    return {**common, "units": sorted(drawn),
            "method": "randomized_systematic_pps_with_take_all_v1",
            "inclusion_probability": {name: float(pi[name]) for name in names},
            "certainty": certainty, "permutation": permutation,
            "start": start, "random_draws": int(remaining)}


#: The size sources a stack draw may be proportional to.  ``probe`` is the
#: per-expert Fisher vector, which is what the estimator's variance argument
#: rests on; ``counts`` is the census's per-expert routed-row count, which is
#: the ONLY per-expert size available when no probe exists -- a routed-token
#: proxy for ``h_trace`` and never a substitute for it (RobTand/prismaquant#495
#: part 1, and the report's section 3.6 bullet on sizes).
STACK_SAMPLE_SIZE_SOURCES = ("probe", "counts")


#: The design a ``counts``-sized draw declares, so a reader of ``units.json``
#: can tell which vector the inclusion probabilities are proportional to
#: without re-deriving it.
STACK_SAMPLE_COUNTS_SUFFIX = "_counts"


def _stack_draw_sizes(record: Mapping, sample: StackExpertSample) -> dict:
    """The sizes a stack's draw was made proportional to.

    A record that declares its own ``sizes`` replays on those; one that does
    not is a probe-sized draw and replays on the Fisher vector, exactly as
    every record written before #495 does.
    """
    sizes = record.get("sizes")
    if sizes is None:
        return {str(e): h for e, h in enumerate(sample.h_trace_per_expert)}
    if str(sizes.get("source")) not in STACK_SAMPLE_SIZE_SOURCES:
        raise StackSampleError(
            f"{sample.packed_qname}: size source {sizes.get('source')!r} is "
            f"not one of {list(STACK_SAMPLE_SIZE_SOURCES)}")
    values = sizes.get("values")
    if not isinstance(values, Mapping) or len(values) != sample.num_experts:
        raise StackSampleError(
            f"{sample.packed_qname}: a declared size vector must carry one "
            f"value per expert ({sample.num_experts})")
    return {str(e): float(values[str(e)]) for e in range(sample.num_experts)}


def _stack_design(record: Mapping, draw: Mapping) -> str:
    """The design name a draw carries, including which sizes it used."""
    sizes = record.get("sizes")
    method = str(draw["method"])
    if sizes is None or str(sizes.get("source")) == "probe":
        return method
    return method + STACK_SAMPLE_COUNTS_SUFFIX


def selection_stack_samples(selection: Mapping, profile) -> dict[str, StackExpertSample]:
    """Rehydrate packed probe/draw records and bind them to the whole group.

    A selection may narrow measured experts; it may not invent projections,
    lose an expert from the frame, or give two roles different expert draws.
    Legacy unsampled selections retain their existing per-unit behavior.
    """
    result = {}
    for entry in selection["groups"]:
        records = entry.get("stack_samples")
        if records is None:
            if entry.get("sampled") and str(entry["key"]).startswith("s:"):
                raise StackSampleError(
                    f"{entry['key']}: sampled stack requires original packed probe/draw records")
            continue
        if not isinstance(records, Mapping) or not records:
            raise StackSampleError(f"{entry['key']}: stack_samples must name packed parameters")
        frame_members, sampled_members, frame_pi, audit_members = set(), set(), {}, set()
        for name, record in sorted(records.items()):
            if name in result:
                raise StackSampleError(f"{name}: duplicate packed sampling record")
            sample = stack_sample_from_probe(
                name, record["probe_row"], profile,
                sampled_experts=record["sampled_experts"],
                inclusion_prob=record["inclusion_prob"], seed=record["seed"],
                design=record["design"],
                transfer_law_experts=record.get("transfer_law_experts", ()))
            validate_stack_sample(sample)
            replay = draw_stack_sample(
                _stack_draw_sizes(record, sample),
                len(sample.sampled_experts), seed=sample.seed, stack=name)
            if (record.get("draw") != replay
                    or sample.design != _stack_design(record, replay)
                    or list(sample.sampled_experts) != sorted(int(e) for e in replay["units"])
                    or dict(sample.inclusion_prob) != {
                        int(e): p for e, p in replay["inclusion_probability"].items()}):
                raise StackSampleError(f"{name}: packed draw receipt does not replay from probe and seed")
            declared = record.get("sizes")
            if declared is not None and str(declared.get("sha256")) != replay["size_sha256"]:
                # The digest is what binds the DECLARED size vector to the one
                # the draw was actually made proportional to; without it a
                # record could name ``counts`` and carry a probe-sized draw.
                raise StackSampleError(
                    f"{name}: declared size digest does not match the draw's")
            if set(sample.inclusion_prob) != set(range(sample.num_experts)):
                raise StackSampleError(f"{name}: selection needs full-frame inclusion probabilities")
            if entry["key"] != "s:" + sample.packed_experts_module:
                raise StackSampleError(f"{name}: packed module disagrees with anchor group")
            frame = stack_sample_from_probe(
                name, record["probe_row"], profile,
                sampled_experts=range(sample.num_experts),
                inclusion_prob={e: 1.0 for e in range(sample.num_experts)},
                seed=sample.seed, design="census")
            for e, members in frame.members.items():
                for member in members:
                    if member in frame_members:
                        raise StackSampleError(f"{member}: two packed parameters claim one projection")
                    frame_members.add(member)
                    frame_pi[member] = sample.inclusion_prob[e]
                    if e in record.get("audit_experts", []):
                        audit_members.add(member)
            sampled_members.update(m for members in sample.members.values() for m in members)
            result[name] = sample
        if frame_members != set(entry["members"]):
            raise StackSampleError(f"{entry['key']}: packed probe projections disagree with census members")
        if sampled_members != set(entry.get("sampled", entry["members"])):
            raise StackSampleError(f"{entry['key']}: packed draw disagrees with sampled members")
        if audit_members != set(entry.get("audit", [])):
            raise StackSampleError(f"{entry['key']}: packed audit disagrees with audit members")
        if frame_pi != entry.get("inclusion_probability"):
            raise StackSampleError(f"{entry['key']}: packed draw disagrees with full-frame probabilities")
    return result


def select_anchor_groups(selection: Mapping, resolved: Mapping[str, list[str]],
                         *, where: str) -> list[str]:
    """The selected group keys, refusing any disagreement with this run's scope.

    The selection is checked against the grouping **this run** resolved, member
    for member.  A plan written against another model, stride or profile names
    a group whose membership differs here, and a shard that silently measured a
    different set would leave the merged table short of rows nothing reported.
    """
    keys: list[str] = []
    for entry in selection["groups"]:
        key = entry["key"]
        members = sorted(entry["members"])
        if key not in resolved:
            raise RuntimeError(
                f"{where}: selection names anchor group {key!r}, which this "
                f"run's scope does not contain")
        if sorted(resolved[key]) != members:
            raise RuntimeError(
                f"{where}: anchor group {key!r} has members "
                f"{sorted(resolved[key])} here and {members} in the selection")
        if key in keys:
            raise RuntimeError(f"{where}: anchor group {key!r} selected twice")
        keys.append(key)
    return keys


@dataclass(frozen=True)
class ExpertPopulation:
    """The packed expert population the bridge covers, and what it omits.

    ``members`` are the profile-declared per-expert projections inside the
    layer scope (``PackedExpertProjection``: the live 2-D view, its packed
    parent and its stack); ``declared`` is ``{stack: {qname: (rows, cols)}}``
    for the producer request; ``omitted_outside_layer_stride`` names the
    packed parameters the stride left out, so the payload can say which
    population was priced and which was not.
    """

    members: tuple
    declared: dict
    packed_in_scope: dict
    omitted_outside_layer_stride: dict

    @property
    def qnames(self) -> list[str]:
        return [member.qname for member in self.members]


def campaign_population_block(*, dense_targets, expert_targets, dense_all, pinned,
                      population: ExpertPopulation, layer_stride: int,
                      costs, menus, stack_samples=None, profile=None) -> dict:
    """Distinguish selected targets from units with actual emitted prices."""
    from .tessera_expert_projection import POPULATION_SCHEMA

    dense_omitted = sorted(set(dense_all) - set(dense_targets))
    priced = {name for name, rows in costs.items() if rows}
    stack_decisions = {}
    represented = set()
    for packed_qname, sample in sorted((stack_samples or {}).items()):
        if profile is None:
            raise StackSampleError("stack population requires the model profile")
        if (packed_qname != sample.packed_qname or
                packed_qname not in population.packed_in_scope or
                int(population.packed_in_scope[packed_qname][0]) != sample.num_experts):
            raise StackSampleError(f"{packed_qname}: sample disagrees with packed population")
        roles = tuple(profile.packed_expert_projection_names(sample.packed_param))
        stem = packed_qname[: -len(sample.packed_param)].rstrip(".")
        members = {f"{stem}.{e}.{role}" for e in range(sample.num_experts) for role in roles}
        declared = set(population.declared.get(sample.packed_experts_module, {}))
        measured = {member for group in sample.members.values() for member in group}
        expected_measured = {f"{stem}.{e}.{role}" for e in sample.sampled_experts for role in roles}
        if (not members or not members <= declared or measured != expected_measured
                or not measured <= members):
            raise StackSampleError(f"{packed_qname}: sample members disagree with declared population")
        if represented & members:
            raise StackSampleError(f"{packed_qname}: source members belong to multiple stack decisions")
        if members & priced:
            raise StackSampleError(f"{packed_qname}: both packed decision and source members are priced")
        represented.update(members)
        stack_decisions[packed_qname] = {
            "stack": sample.packed_experts_module,
            "members": sorted(members), "sampled_members": sorted(measured),
        }
    # Source experts represented by an HT row are neither separate prices nor
    # unpriced BF16 units. Keep their full frame in the explicit decision map.
    expert_targets = (set(expert_targets) - represented) | set(stack_decisions)
    dense_priced = sorted(set(dense_targets) & priced)
    expert_priced = sorted(set(expert_targets) & priced)
    unpriced = {
        kind: {name: ("no_admitted_menu" if not (menus.get(name) or
                          any(menus.get(member) for member in
                              stack_decisions.get(name, {}).get("sampled_members", ())))
                      else "no_successful_anchor")
               for name in sorted(set(targets) - priced)}
        for kind, targets in (("dense", dense_targets),
                              ("routed_experts", expert_targets))
    }
    represented_priced = priced | {
        member for name, decision in stack_decisions.items() if name in priced
        for member in decision["members"]}
    complete_stacks = sorted(stack for stack, units in population.declared.items()
                             if set(units) <= represented_priced)
    packed = {name: list(shape) for name, shape
              in sorted(population.packed_in_scope.items())}
    packed_omitted = {name: list(shape) for name, shape
                      in sorted(population.omitted_outside_layer_stride.items())}
    result = {
        "schema": POPULATION_SCHEMA,
        "layer_stride": int(layer_stride),
        "enumerated": {"dense": sorted(dense_targets),
                       "routed_experts": sorted(expert_targets),
                       "packed_parameters": packed,
                       "stacks": sorted(population.declared)},
        "unpriced": unpriced,
        "priced": {
            "dense": dense_priced,
            "routed_experts": expert_priced,
            "packed_parameters": {name: shape for name, shape in packed.items()
                                  if name.rsplit(".", 1)[0] in complete_stacks},
            "stacks": complete_stacks,
        },
        "omitted": {
            "dense_outside_layer_stride": dense_omitted,
            "packed_outside_layer_stride": packed_omitted,
            "pinned": sorted(pinned),
        },
        "counts": {
            "dense_priced": len(dense_priced),
            "routed_experts_priced": len(expert_priced),
            "dense_unpriced": len(unpriced["dense"]),
            "routed_experts_unpriced": len(unpriced["routed_experts"]),
            "dense_omitted": len(dense_omitted),
            "packed_omitted": len(packed_omitted),
            "pinned": len(pinned),
        },
    }
    if stack_decisions:
        result["stack_decisions"] = stack_decisions
    return result


#: What ``plan`` writes beside the manifest, so ``merge`` reads the row layout
#: from the plan rather than from the directory listing it happens to find.
PLAN_SCHEMA = "prismaquant.tessera_campaign_plan.v1"


COVERAGE_SCHEMA = "prismaquant.tessera_campaign_coverage.v1"


#: Provenance fields every row must agree on before a merge is possible.  Each
#: one describes the run, not the selection, so a disagreement means two
#: campaigns are being merged into one table.
SHARED_PROVENANCE = (
    "menu_mode", "tp_degree", "model", "nsamples", "seqlen", "max_act_rows",
    "layer_stride", "anchors_round_one", "max_rounds", "anchor_budget",
    "loo_gate", "max_artifact_bpp", "cost_mode", "rate_band", "calibration_cache",
)


#: Hessian identity fields every row must already agree on.  ``capture_sha256``
#: is deliberately absent: it is the digest of the capture a row wrote for its
#: own units, and reconciling it is what the merge is for.
SHARED_HESSIAN = (
    "supplied", "text_sha", "token_count", "text_sha256", "fit_ids_sha256",
    "fit_tokens", "kwarg", "reference_binding",
)


#: Checkpoint-identity keys the merge RECONCILES instead of requiring equal.
#: Every one of them is per-selection: it describes the units this row was
#: given, not the campaign. Everything outside this set must already agree,
#: because a difference there means the rows priced two different campaigns.
RECONCILED_IDENTITY_KEYS = frozenset({
    "units", "serving_scope", "expert_projection", "stack_sampling_identity",
    "family_restriction",
})


class MergeRefused(RuntimeError):
    """The rows do not describe one campaign."""


#: The scope a joint pass evaluates, and the two shapes it can take.  A
#: campaign-scoped pass evaluates every unit and every window the campaign's
#: census defines; a diagnostic pass evaluates an explicitly frozen window
#: subset of that same roster.  The scope is *derived* from the census and
#: campaign plan the joint plan binds by sha256 -- never from a tally the plan
#: states about itself -- so a roster or window set narrowed without moving the
#: declared count cannot read as complete.  Two rosters of equal length with
#: different members are different scopes; so are two window sets of equal size
#: with different membership.
CAMPAIGN_SCOPE_SCHEMA = "prismaquant.tessera_joint_campaign_scope.v1"


COMPLETE_CAMPAIGN_SCOPE = "complete_campaign"


DIAGNOSTIC_SCOPE = "diagnostic_window_subset"


CAMPAIGN_SCOPE_KINDS = (COMPLETE_CAMPAIGN_SCOPE, DIAGNOSTIC_SCOPE)


PANEL_SCHEMA = "prismaquant.tessera_joint_eval_panel.v1"


PANEL_STATUS = "diagnostic_pilot"


class ScopeRefused(RuntimeError):
    """A joint row's evaluated scope is not the scope it is submitted for."""


def _bound_json(plan: dict, key: str, *, label: str) -> dict:
    """The JSON artifact a joint plan binds by path and sha256, re-checked."""
    declared = (plan.get("inputs") or {}).get(key)
    if (not isinstance(declared, dict) or not isinstance(declared.get("path"), str)
            or not isinstance(declared.get("sha256"), str)):
        raise ScopeRefused(f"{label}: joint plan inputs.{key} is not a bound artifact")
    path = Path(declared["path"])
    actual = file_sha256hex(path)
    if actual != declared["sha256"]:
        raise ScopeRefused(
            f"{label}: plan inputs.{key} is {path}, which hashes to {actual}, "
            f"not the bound {declared['sha256']}")
    return json.loads(path.read_text())


def _bound_digest(container: dict, key: str, *, label: str, where: str) -> str:
    """One artifact a plan binds by path and sha256, re-checked, as its digest.

    The same check ``_bound_json`` makes, for the artifacts a scope *names* but
    never parses: the calibration tokens, the canonical capture manifest and the
    merged campaign checkpoint are large or non-JSON, and a digest is all the
    identity needs.  Omitting the check would let a plan bind a digest it does
    not hold.
    """
    declared = (container or {}).get(key)
    if (not isinstance(declared, dict) or not isinstance(declared.get("path"), str)
            or not isinstance(declared.get("sha256"), str) or not declared["sha256"]):
        raise ScopeRefused(f"{label}: {where} is not a bound artifact")
    path = Path(declared["path"])
    actual = file_sha256hex(path)
    if actual != declared["sha256"]:
        raise ScopeRefused(
            f"{label}: {where} is {path}, which hashes to {actual}, not the "
            f"bound {declared['sha256']}")
    return actual


def _bound_pickle(container: dict, key: str, *, label: str, where: str) -> dict:
    """The pickled artifact a plan binds by path and sha256, re-read and checked.

    ``_bound_json``'s check and then the object itself, for the campaign cost
    table: the plan's own record of which ``(unit, format)`` cells the campaign
    priced.  The digest is what makes that record the plan's rather than the
    caller's -- an edited table no longer hashes to what the plan declares --
    and reading it is one sequential pass over a few hundred MiB, not the
    multi-GiB candidate roster a per-cell re-derivation from the checkpoint
    would need.
    """
    import pickle

    declared = (container or {}).get(key)
    if (not isinstance(declared, dict) or not isinstance(declared.get("path"), str)
            or not isinstance(declared.get("sha256"), str) or not declared["sha256"]):
        raise ScopeRefused(f"{label}: {where} is not a bound artifact")
    path = Path(declared["path"])
    actual = file_sha256hex(path)
    if actual != declared["sha256"]:
        raise ScopeRefused(
            f"{label}: {where} is {path}, which hashes to {actual}, not the "
            f"bound {declared['sha256']}; the campaign cost table the plan names "
            "is not the one on disk, so the cells it priced are not the plan's")
    with path.open("rb") as handle:
        blob = pickle.load(handle)
    if not isinstance(blob, dict):
        raise ScopeRefused(f"{label}: {where} ({path}) is not a cost payload")
    return blob


def _census_draw(census: dict, *, label: str) -> dict:
    """The frozen draw a census was taken on, or a refusal naming the field."""
    draw: dict = {}
    for field in ("model", "text_sha256", "fit_ids_sha256"):
        value = census.get(field)
        if not isinstance(value, str) or not value:
            raise ScopeRefused(
                f"{label}: the census declares no {field}, so the calibration "
                "draw it was taken on is unidentified; a roster and a window "
                "count are not a draw")
        draw[field] = value
    for field in ("seed", "layer_stride"):
        value = census.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ScopeRefused(
                f"{label}: the census declares no integer {field}, so the "
                "calibration draw it was taken on is unidentified")
        draw[field] = value
    return draw


def joint_campaign_scope(plan: dict, *, label: str = "joint row") -> dict:
    """The exact roster and window identity one joint plan evaluates.

    Read from the plan's own bound census and campaign plan.  A plan whose
    census roster is not the campaign plan's roster, whose ``calib_seqlen`` is
    not the census ``seqlen``, or whose declared unit/group tallies do not
    equal the roster it actually bound refuses here -- the counts are checked
    against the identity, not accepted in place of it.
    """
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256

    inputs = plan.get("inputs")
    if not isinstance(inputs, dict):
        raise ScopeRefused(f"{label}: the joint plan declares no inputs block")
    census = _bound_json(plan, "census", label=label)
    campaign = _bound_json(plan, "campaign_plan", label=label)
    if census.get("schema") != CENSUS_SCHEMA:
        raise ScopeRefused(
            f"{label}: census schema is {census.get('schema')!r}, expected {CENSUS_SCHEMA}")
    if Path(campaign["census"]).resolve() != Path(inputs["census"]["path"]).resolve():
        raise ScopeRefused(
            f"{label}: the campaign plan and the joint plan name different censuses")
    roster = census.get("unit_shapes")
    groups = census.get("anchor_groups")
    if (not isinstance(roster, dict) or not roster
            or not isinstance(groups, dict) or not groups):
        raise ScopeRefused(f"{label}: the census carries no exact roster")
    owners: set = set()
    for row in campaign.get("rows") or []:
        owners.update(row.get("members") or [])
    if owners != set(roster):
        raise ScopeRefused(
            f"{label}: the census roster is not the campaign plan's roster "
            f"({len(owners)} campaign members, {len(roster)} census units)")
    # The census is the window authority: a plan that narrows its own
    # ``n_calib_samples`` is a narrower scope, not the campaign's.
    campaign_windows = census.get("nsamples")
    seqlen = census.get("seqlen")
    if (isinstance(campaign_windows, bool) or not isinstance(campaign_windows, int)
            or campaign_windows <= 0 or not isinstance(seqlen, int) or seqlen <= 0):
        raise ScopeRefused(
            f"{label}: the census declares no window count or sequence length")
    execution = plan.get("execution") or {}
    if execution.get("calib_seqlen") != seqlen:
        raise ScopeRefused(
            f"{label}: plan calib_seqlen {execution.get('calib_seqlen')!r} is not "
            f"the census seqlen {seqlen!r}")
    evaluated = execution.get("n_calib_samples")
    panel = plan.get("joint_eval")
    if panel is None:
        kind, selection = COMPLETE_CAMPAIGN_SCOPE, None
        windows = evaluated
    else:
        kind = DIAGNOSTIC_SCOPE
        if (not isinstance(panel, dict) or panel.get("schema") != PANEL_SCHEMA
                or panel.get("status") != PANEL_STATUS):
            raise ScopeRefused(
                f"{label}: a narrow scope needs an explicit {PANEL_SCHEMA} "
                f"panel with status {PANEL_STATUS!r}")
        selected = panel.get("selection")
        windows = selected.get("size") if isinstance(selected, dict) else None
        if (isinstance(windows, bool) or not isinstance(windows, int)
                or not 0 < windows < campaign_windows):
            raise ScopeRefused(
                f"{label}: a diagnostic scope selects a strict, sized subset of the "
                f"{campaign_windows}-window campaign, got {windows!r}")
        if panel.get("shape") != [windows, seqlen]:
            raise ScopeRefused(
                f"{label}: the diagnostic panel shape {panel.get('shape')!r} is not "
                f"[{windows}, {seqlen}]")
        selection = canonical_json_sha256(panel, where=f"{label} joint evaluation panel")
    if isinstance(windows, bool) or not isinstance(windows, int) or windows <= 0:
        raise ScopeRefused(f"{label}: the plan evaluates no windows")
    if kind == COMPLETE_CAMPAIGN_SCOPE and windows != campaign_windows:
        raise ScopeRefused(
            f"{label}: a campaign-scoped plan evaluates all {campaign_windows} "
            f"windows, got {windows}; freeze the subset as an explicit diagnostic panel")
    declared = (inputs.get("required_source_units"),
                inputs.get("required_campaign_groups"))
    if declared != (len(roster), len(groups)):
        raise ScopeRefused(
            f"{label}: the plan declares {declared[0]!r}/{declared[1]!r} units/groups "
            f"but bound {len(roster)}/{len(groups)}; a count is not the roster")
    # The draw and the capture are the other half of "the same calibration
    # contract": the roster says which Linears were priced, this says over which
    # windows, which corpus revision and which tokenizer ids, and which captured
    # activations were read. Read from the plan's own bound artifacts, so a plan
    # cannot restate one it does not hold.
    calibration_sha256 = canonical_json_sha256({
        "census_draw": _census_draw(census, label=label),
        "window_count": campaign_windows,
        "calib_seqlen": seqlen,
        "calibration_input_sha256": _bound_digest(
            plan, "calibration_input", label=label,
            where="joint plan calibration_input"),
        "canonical_capture_sha256": _bound_digest(
            plan, "canonical_capture", label=label,
            where="joint plan canonical_capture"),
    }, where=f"{label} calibration identity")
    # The candidate roster is `identity.units[unit].menu` inside the campaign's
    # merged checkpoint, and the joint loader refuses any priced rung outside
    # it. Binding that artifact binds the exact (unit, format) roster the pass
    # is admitted against: a reduced or substituted menu is different bytes, so
    # it is a different campaign. The bind costs ONE sequential read of the
    # artifact per submission (``_bound_digest`` hashes it rather than trusting
    # the plan's own number); what it avoids is parsing the identity into the
    # graph, which is the memory the joint loader's streaming seal exists to
    # remove.
    campaign_checkpoint_sha256 = _bound_digest(
        inputs, "merged_checkpoint", label=label,
        where="joint plan inputs.merged_checkpoint")
    return {
        "schema": CAMPAIGN_SCOPE_SCHEMA,
        "kind": kind,
        "source_roster_sha256": canonical_json_sha256(
            sorted(roster), where=f"{label} source roster"),
        "source_unit_count": len(roster),
        "campaign_group_roster_sha256": canonical_json_sha256(
            groups, where=f"{label} campaign group roster"),
        "campaign_group_count": len(groups),
        "window_count": windows,
        "campaign_window_count": campaign_windows,
        "calib_seqlen": seqlen,
        "selection_sha256": selection,
        "calibration_sha256": calibration_sha256,
        "campaign_checkpoint_sha256": campaign_checkpoint_sha256,
    }


CAMPAIGN_IDENTITY_SCHEMA = "prismaquant.tessera_joint_campaign_identity.v1"


#: The fields of the frozen campaign identity a submission has to reproduce.
#: ``window_count`` is deliberately absent: the campaign identity names the
#: campaign's own window total, and whether a plan evaluates all of it is the
#: ``kind`` the caller requires, not a property of the campaign.
#: ``selection_sha256`` is absent for the same reason -- a diagnostic subset is
#: a different ``kind`` of the same campaign, and its panel identity travels in
#: the scope the submission is stamped with. ``calibration_sha256`` is NOT
#: absent: the pilot and the full continuation read one draw and one capture, so
#: both have to reproduce it, and it deliberately excludes the panel so that
#: they can.
CAMPAIGN_IDENTITY_FIELDS = (
    "source_unit_count", "source_roster_sha256",
    "campaign_group_count", "campaign_group_roster_sha256",
    "campaign_window_count", "calib_seqlen",
    "calibration_sha256", "campaign_checkpoint_sha256")


def campaign_identity(scope: dict) -> dict:
    """The frozen identity of the campaign a scope belongs to.

    Derived from a plan's bound census so that the pilot and the full
    continuation can be held to one roster and one window total without either
    of them restating it.  An operator seals this once and passes it to every
    joint submission; the generic path never hardcodes the numbers.
    """
    record = {"schema": CAMPAIGN_IDENTITY_SCHEMA,
              **{field: scope[field] for field in CAMPAIGN_IDENTITY_FIELDS}}
    return record


def verify_joint_campaign_scope(plan: dict, *, require_scope: str,
                                campaign: dict, label: str = "joint row") -> dict:
    """Refuse a joint submission whose scope is not the scope it claims.

    ``require_scope`` is what the caller is submitting *for*, and a diagnostic
    subset is not the campaign's score: asking for ``complete_campaign`` and
    handing it a plan that evaluates 16 of 512 windows refuses here, before the
    read set is built, rather than surfacing as a shortfall in the merged cost
    after the GPU window has closed.

    ``campaign`` is the frozen campaign identity the submission belongs to.  It
    is required: a plan is self-consistent with whatever census it binds, so
    without an identity that is fixed *outside* the plan a coherently narrowed
    roster would still read as the whole campaign.  Every identity field is
    compared exactly -- the counts are a convenience of the record, not the
    acceptance.
    """
    if require_scope not in CAMPAIGN_SCOPE_KINDS:
        raise ScopeRefused(
            f"{label}: unknown required scope {require_scope!r}; expected one of "
            f"{', '.join(CAMPAIGN_SCOPE_KINDS)}")
    if not isinstance(campaign, dict) or campaign.get("schema") != CAMPAIGN_IDENTITY_SCHEMA:
        raise ScopeRefused(
            f"{label}: a joint submission must bind a {CAMPAIGN_IDENTITY_SCHEMA} "
            "campaign identity; a plan's own census cannot show its own scope is "
            "the campaign's")
    # A field the identity does not carry cannot be compared, and "absent"
    # compares unequal to every value -- which is how an identity sealed before
    # the calibration and candidate-roster binding would refuse the *right*
    # plan with a message about the wrong thing. Name the re-seal instead.
    absent = [field for field in CAMPAIGN_IDENTITY_FIELDS if field not in campaign]
    if absent:
        raise ScopeRefused(
            f"{label}: the frozen campaign identity carries no "
            f"{', '.join(absent)}; it predates binding the calibration draw, the "
            "capture and the campaign's candidate roster, and must be re-sealed "
            "from a campaign-scoped plan before it can stand for the campaign")
    scope = joint_campaign_scope(plan, label=label)
    for field in CAMPAIGN_IDENTITY_FIELDS:
        declared = campaign.get(field)
        if declared != scope[field]:
            raise ScopeRefused(
                f"{label}: the plan's {field} is {scope[field]!r}, not the frozen "
                f"campaign identity's {declared!r}; a roster of equal length with "
                "different members is a different campaign")
    if scope["kind"] != require_scope:
        raise ScopeRefused(
            f"{label}: this submission requires a {require_scope} scope, but the "
            f"plan evaluates {scope['kind']} ({scope['window_count']} of "
            f"{scope['campaign_window_count']} windows over "
            f"{scope['source_unit_count']} units, selection "
            f"{scope['selection_sha256']}). A diagnostic subset is its own "
            "evidence; submit the campaign-scoped plan for the campaign's score.")
    return scope


def _joint_row_binds_cell(row: dict, name: str, fmt: str, *, label: str) -> bool:
    """Whether this row is a joint A-side **for this cell**.

    The predicate itself is the shared one
    (:func:`prismaquant.joint_aura_rows.joint_row_binds_cell`) so that the
    submission-time gate and the stage that computes coverage cannot drift; the
    only thing added here is this tool's own refusal type and label.
    """
    from prismaquant.joint_aura_rows import joint_row_binds_cell

    try:
        return joint_row_binds_cell(row, name, fmt, where=label)
    except ValueError as error:
        raise ScopeRefused(str(error)) from error


def bound_joint_cells(costs: dict, *, label: str) -> set:
    """The ``(unit, format)`` cells whose own joint A-side is bound to them.

    One walk, one answer, for both readers of "this artifact already carries an
    activation term at this cell": the coverage roster below, and ``submit-aqua``
    deciding whether a stage would add anything. A row that does not bind its own
    key is a refusal here rather than a cell either reader may count, so the two
    cannot disagree about which cells are covered.
    """
    bound: set = set()
    for name, entry in costs.items():
        if not isinstance(entry, dict):
            continue
        for fmt, row in entry.items():
            if not isinstance(row, dict):
                continue
            if _joint_row_binds_cell(row, name, fmt, label=label):
                bound.add((name, fmt))
    return bound


def aqua_requested_cells(plan: dict, payload: dict, formats: "list[str]",
                         *, label: str = "aqua row",
                         accept_joint_cells_outside_plan: bool = False,
                         ) -> "tuple[dict, frozenset]":
    """The exact ``(unit, format)`` roster AQUA is asked to price, from the plan.

    Returns ``(record, cells)``. ``cells`` is the requested roster itself; the
    record is the same thing as counts and a digest, so a submission can stamp
    it without carrying hundreds of thousands of tuples in its summary.

    The roster is the **plan's**, read from the cost table the plan binds at
    ``inputs.merged_cost`` and re-checked against the sha256 it declares there.
    That table is the campaign's own priced surface: the cells the allocation is
    allowed to select from. ``--cost-in`` -- the weight-only or joint-merged
    table the stage merges an A-side into -- has to reproduce it unit for unit.

    Deriving the requested set from the artifact under test is the hole this
    closes. The stage states its acceptance against the cells it is given, so a
    table narrowed to the cells that already carry a price leaves the gate
    holding a denominator that the narrowing moved: deleting one unit's cell
    while another unit still carries that format leaves the unit roster exact
    and the carried-format union exact, and if the remainder is joint-priced the
    run reads as complete coverage while a planned cell has no A-side at all.
    Binding to the plan's own table makes that case a refusal, and makes a cost
    table edited after the plan was sealed a refusal of its own.

    ``formats`` is the campaign's menu and has to be exactly the union of the
    plan's cells. Naming a subset would move the denominator the same way: the
    cells it left out are exactly the ones whose A-side would then go unchecked.

    Extra cells are read the same way round. A cell the plan never priced that
    is *not* already joint-priced would be priced by this stage, which makes the
    gate stricter rather than narrower -- but it is still a row the campaign's
    own table does not have, so it refuses.

    A cell the plan never priced that already carries its own joint A-side is a
    joint pass's addition. Being joint-priced is not by itself evidence that the
    row belongs to *this* plan: the row is internally valid and its own cell is
    the one it names, but nothing in it was compared with the plan's draw, its
    capture or its candidate menu, so a pass over a wider roster would place
    prices this campaign never priced beside the plan's own table and the
    artifact would read as the plan's surface. It refuses, naming the re-seal
    instead, unless the caller passes
    ``accept_joint_cells_outside_plan`` -- which is how an operator reusing an
    artifact sealed against an older roster keeps that reuse explicit and
    recorded as unverified rather than silent. A row whose operator coordinate
    is not the key it was found under is a refusal either way.
    """
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256

    census = _bound_json(plan, "census", label=label)
    roster = census.get("unit_shapes")
    if not isinstance(roster, dict) or not roster:
        raise ScopeRefused(f"{label}: the plan's bound census carries no roster")
    planned = _bound_pickle(
        plan.get("inputs") or {}, "merged_cost", label=label,
        where="joint plan inputs.merged_cost")
    planned_costs = planned.get("costs")
    if not isinstance(planned_costs, dict) or not planned_costs:
        raise ScopeRefused(
            f"{label}: the cost table the plan binds carries no 'costs' table, "
            "so it does not say which cells the campaign priced")
    costs = payload.get("costs")
    if not isinstance(costs, dict) or not costs:
        raise ScopeRefused(f"{label}: the cost artifact carries no 'costs' table")
    # One walk, before anything is counted: every joint row has to be its own
    # cell. A row that is not raises here, so no later reader sees a set with a
    # row whose coordinate this function never checked.
    bound_joints = bound_joint_cells(costs, label=f"{label} cost artifact")
    # The unit roster is checked on BOTH tables against the plan's bound census.
    # A unit the plan priced and the artifact lacks would leave the campaign
    # short an A-side while reading as complete; a unit the artifact carries
    # that the plan never priced is not the campaign's cell at all.
    for what, table in (("the cost table the plan binds", planned_costs),
                        ("the cost artifact", costs)):
        missing = sorted(set(roster) - set(table))
        extra = sorted(set(table) - set(roster))
        if missing or extra:
            raise ScopeRefused(
                f"{label}: {what} is not the plan's roster -- "
                f"{len(missing)} unit(s) the plan priced are absent (e.g. "
                f"{missing[:3]}) and {len(extra)} unit(s) it never priced are "
                f"carried (e.g. {extra[:3]}).")
    wanted = [str(item) for item in formats]
    if len(set(wanted)) != len(wanted) or not wanted:
        raise ScopeRefused(f"{label}: --formats must name each format once")

    cells: set = set()
    planned_formats: set = set()
    units_missing: list = []
    missing_total = 0
    units_extra: list = []
    extra_total = 0
    joint_extra = 0
    for name in sorted(roster):
        planned_entry = planned_costs[name]
        if not isinstance(planned_entry, dict):
            raise ScopeRefused(
                f"{label}: the cost table the plan binds has no per-format row "
                f"for {name}")
        entry = costs[name]
        if not isinstance(entry, dict):
            raise ScopeRefused(f"{label}: {name} has no per-format row")
        expected = {fmt for fmt, row in planned_entry.items()
                    if isinstance(row, dict)}
        carried = {fmt for fmt, row in entry.items() if isinstance(row, dict)}
        planned_formats |= expected
        # The requested roster is the plan's cells, so it is the same digest for
        # the same plan whether or not a joint pass has since added rows of its
        # own.
        cells.update((name, fmt) for fmt in expected)
        gone = sorted(expected - carried)
        if gone:
            missing_total += len(gone)
            units_missing.append(
                # An empty entry is named as such rather than by one of the
                # cells it lost: it is the shape a truncating write or a
                # dropped merge produces, and the unit is present exactly as
                # the roster demands.
                f"{name}@<entry empty>" if not carried else f"{name}@{gone[0]}")
        outside = carried - expected
        invented = []
        outside_joints = []
        for fmt in sorted(outside):
            if (name, fmt) in bound_joints:
                outside_joints.append(fmt)
                joint_extra += 1
            else:
                invented.append(fmt)
        if outside_joints and not accept_joint_cells_outside_plan:
            joints = ", ".join(f"{name}@{fmt}" for fmt in outside_joints[:5])
            raise ScopeRefused(
                f"{label}: the cost artifact carries {len(outside_joints)} "
                f"joint-priced cell(s) the plan's own cost table never priced "
                f"(e.g. {joints}). A joint row is bound to the coordinate it "
                "names and to nothing else -- it is not compared with this "
                "plan's draw, capture or candidate menu -- so a pass over a "
                "wider roster would place this campaign's table beside prices "
                "it never priced. Seal a requested roster that names them, or "
                "pass --accept-joint-cells-outside-plan to reuse the artifact "
                "as unverified.")
        if invented:
            extra_total += len(invented)
            units_extra.append(f"{name}@{invented[0]}")
    if missing_total or extra_total:
        # Bounded, like every other refusal here: a campaign has tens of
        # thousands of units, and naming each one floods the log the operator
        # has to read. The totals are the contract; the sample is what makes
        # them checkable.
        missing_units, extra_units = len(units_missing), len(units_extra)
        units_missing, units_extra = units_missing[:5], units_extra[:5]
        raise ScopeRefused(
            f"{label}: the cost artifact is not the plan's priced surface -- it "
            f"does not carry {missing_total} planned (unit, format) cell(s) "
            f"across {missing_units} unit(s) (e.g. "
            f"{', '.join(units_missing) or 'none'}) and carries {extra_total} "
            f"cell(s) the plan's cost table never priced across "
            f"{extra_units} unit(s) (e.g. "
            f"{', '.join(units_extra) or 'none'}). The stage states its "
            "acceptance against the requested cells, so an artifact that drops "
            "one unit's cell while another unit still carries that format would "
            "narrow the denominator instead of filling the hole.")
    unnamed = sorted(planned_formats - set(wanted))
    absent = sorted(set(wanted) - planned_formats)
    if unnamed or absent:
        raise ScopeRefused(
            f"{label}: --formats {sorted(wanted)} is not the menu the plan "
            f"prices -- the plan's cost table holds {unnamed} that are unnamed "
            f"and none of {absent}. The requested set is the denominator the "
            "coverage gate reads, so it is bound to the plan's cells rather "
            "than narrowed to the ones that already have a price.")
    record = {"requested_cells": len(cells),
              "requested_units": len(roster),
              "formats": sorted(wanted),
              # Joint rows a joint pass added beyond the plan's own table. They
              # are not part of the requested roster, and they are accepted only
              # when the caller asked for that reuse explicitly: the flag is
              # recorded here so an artifact holding cells this campaign never
              # priced carries that fact into every receipt that stamps this
              # record rather than reading as the plan's own surface.
              "joint_cells_outside_plan": joint_extra,
              "joint_cells_outside_plan_accepted_unverified": bool(
                  joint_extra and accept_joint_cells_outside_plan),
              "roster_sha256": canonical_json_sha256(
                  {"units": sorted(roster), "cells": sorted(cells)},
                  where=f"{label} requested roster")}
    return record, frozenset(cells)


def require_equal(field: str, values: dict) -> object:
    distinct = {DIRECT_ASCII_SPACED_LAX_DEFAULT_STR.text(value) for value in values.values()}
    if len(distinct) != 1:
        detail = ", ".join(f"{row}={DIRECT_ASCII_SPACED_LAX_DEFAULT_STR.text(value)[:160]}"
                           for row, value in sorted(values.items()))
        raise MergeRefused(f"rows disagree on {field}: {detail}")
    return next(iter(values.values()))


def hessian_identities(payload: dict) -> list[dict]:
    return [row["hessian_identity"]
            for rows in payload["costs"].values() for row in rows.values()
            if "hessian_identity" in row]


def check_acquisition_identity(actual, expected, *, where):
    """Authenticate control comparability before stamping only raw request provenance."""
    from prismaquant.tessera_acquisition_inputs import joint_campaign_acquisition_comparable_identity
    from prismaquant.dev_mode import seal_check

    try:
        actual_control_identity = joint_campaign_acquisition_comparable_identity(actual)
        expected_control_identity = joint_campaign_acquisition_comparable_identity(expected)
    except ValueError as exc:
        raise MergeRefused(f"{where}: acquisition global control identity malformed: {exc}") from exc
    if actual_control_identity != expected_control_identity or set(actual) != set(expected):
        raise MergeRefused(f"{where}: acquisition global authenticated control/cost/run/probe/regime identity differs")
    seal_check("acquisition request byte identity", expected["request_sha256"],
        actual["request_sha256"], where=where,
        refusal=lambda: MergeRefused(f"{where}: acquisition request byte identity differs"))


def merge_acquisition_rows(records, *, acquisition, scope_groups):
    """Strict disjoint complete requested-work union; no exemption for settings.

    Original intake authenticates raw joint rows. This compares their global
    identity and the rows' actual source proofs/schedules, never creates a
    Fisher/native price or interprets an unsupported empty family as priced.
    """
    present = [record.get("origin") is not None for record in records.values()]
    if not any(present):
        if any(record.get("schedule") is not None or any(
                proof is not None for proof in (record.get("sources") or {}).values())
               for record in records.values()):
            raise MergeRefused("acquisition schedule/source proof lacks its global origin")
        if acquisition is not None:
            raise MergeRefused("acquisition plan has rows without acquisition identity")
        return None
    if not all(present) or acquisition is None or scope_groups is None:
        raise MergeRefused("acquisition merge requires the original authenticated request and actual scope")
    from prismaquant.tessera_formats import parse_tessera_format_name
    try:
        keys, _ = campaign_acquisition_scope(acquisition, scope_groups)
    except (ValueError, KeyError, TypeError) as exc:
        raise MergeRefused(f"acquisition original scope refuses: {exc}") from exc
    active = {key for key in keys if any(qs for name in scope_groups[key]
              for qs in acquisition["requests"][name].values())}
    expected_units = {name for key in active for name in scope_groups[key]}
    schedule, sources, deferred = {}, {}, {}
    common = {**acquisition["identity"],
        "purpose": "actual_scalar_render_journal_and_wire_preparation", "currency": CURRENCY}
    for row_id, record in sorted(records.items()):
        members = set(record["units"])
        if members.intersection(schedule):
            raise MergeRefused(f"{row_id}: acquisition units overlap another row")
        try:
            projected, row_keys, _ = campaign_acquisition_row_scope(
                acquisition, scope_groups, selected=members)
        except (ValueError, KeyError, TypeError) as exc:
            raise MergeRefused(f"{row_id}: acquisition partial/unknown row scope: {exc}") from exc
        if not set(row_keys) <= active:
            raise MergeRefused(f"{row_id}: acquisition row includes an all-deferred atomic group")
        expected_schedule = requested_acquisition_schedule(
            {key: scope_groups[key] for key in row_keys}, projected["requests"])
        if record.get("schedule") != expected_schedule:
            raise MergeRefused(f"{row_id}: acquisition requested schedule differs")
        if record.get("sources") != projected["source_weights"]:
            raise MergeRefused(f"{row_id}: acquisition source weight proof differs")
        origin = record["origin"]
        if (not isinstance(origin, dict) or set(origin) != set(common) | {
                "expanded_actual_work_count", "deferred_domain"}):
            raise MergeRefused(f"{row_id}: acquisition global request/cost/run/probe/regime identity differs")
        check_acquisition_identity({key: origin[key] for key in common}, common, where=row_id)
        count = sum(len(qs) for families in expected_schedule.values() for qs in families.values())
        domain = origin["deferred_domain"]
        if type(origin["expanded_actual_work_count"]) is not int or origin["expanded_actual_work_count"] != count:
            raise MergeRefused(f"{row_id}: acquisition expanded work count differs")
        if not isinstance(domain, dict) or set(domain) != members:
            raise MergeRefused(f"{row_id}: acquisition deferred domain scope differs")
        for name, families in expected_schedule.items():
            unit = record["units"][name]
            if not isinstance(unit, dict) or not isinstance(unit.get("menu"), list):
                raise MergeRefused(f"{row_id}: acquisition requires bound checkpoint menu for {name}")
            source = unit.get("weight")
            expected_source = projected["source_weights"][name]
            if (unit.get("acquisition_source_weight") != expected_source
                    or not isinstance(source, dict) or source.get("shape") != expected_source["shape"]
                    or source.get("dtype") != expected_source["dtype"]):
                raise MergeRefused(f"{row_id}: acquisition checkpoint source proof differs for {name}")
            menu_families = set()
            try:
                for fmt in unit["menu"]:
                    parsed = parse_tessera_format_name(fmt)
                    if parsed is None:
                        raise ValueError("not a canonical Tessera menu format")
                    menu_families.add(parsed[0].name)
            except (ValueError, TypeError) as exc:
                raise MergeRefused(f"{row_id}: acquisition checkpoint menu malformed for {name}") from exc
            expected_deferred = campaign_acquisition_deferred_families(families, menu_families)
            if domain[name] != expected_deferred:
                raise MergeRefused(f"{row_id}: acquisition exact deferred families differ for {name}")
            expected_formats = {f"{family}_R{q}" for family, qs in families.items() for q in qs}
            if not expected_formats <= set(unit["menu"]):
                raise MergeRefused(f"{row_id}: acquisition requested menu missing for {name}")
            if "cells" in record and (set(record["cells"].get(name, [])) != expected_formats
                    or len(record["cells"].get(name, [])) != len(expected_formats)):
                raise MergeRefused(f"{row_id}: acquisition missing/extra measured cells for {name}")
        if "cells" in record and set(record["cells"]) != members:
            raise MergeRefused(f"{row_id}: acquisition measured cell scope differs")
        schedule.update(expected_schedule)
        sources.update(projected["source_weights"])
        deferred.update(domain)
    if set(schedule) != expected_units:
        raise MergeRefused("acquisition active atomic group coverage incomplete")
    return {"origin": {**common,
                "expanded_actual_work_count": sum(len(qs) for families in schedule.values() for qs in families.values()),
                "deferred_domain": dict(sorted(deferred.items()))},
            "schedule": dict(sorted(schedule.items())), "sources": dict(sorted(sources.items()))}


def payload_acquisition_records(payloads, *, unit_identities):
    """Join payload rows to the existing authenticated checkpoint menu owner."""
    if not isinstance(unit_identities, dict):
        raise MergeRefused("acquisition payload merge requires bound checkpoint unit identities")
    selected = {row: selection_priced_units(payload["provenance"]["unit_selection"])[0]
                for row, payload in payloads.items()}
    if set(unit_identities) != {name for names in selected.values() for name in names}:
        raise MergeRefused("acquisition checkpoint/payload unit scope differs")
    records = {}
    for row, payload in payloads.items():
        prov = payload["provenance"]
        if prov.get("menu_mode") != "research" or prov.get("max_rounds") != 1 or prov.get("rate_band") is not None:
            raise MergeRefused(f"{row}: acquisition payload regime differs")
        for name, formats in payload["costs"].items():
            for fmt, price in formats.items():
                value = price.get("output_mse")
                if (price.get("currency") != CURRENCY or price.get("output_mse_measured") is not True
                        or price.get("cost_source") != "tessera_campaign_measured"
                        or type(value) not in (int, float) or not math.isfinite(value) or value < 0
                        or fmt != f"{price.get('tessera_family')}_R{price.get('tessera_body_rate_q256')}"):
                    raise MergeRefused(f"{row}: acquisition scalar render measurement missing/foreign for {name}@{fmt}")
        records[row] = {"origin": prov.get("acquisition"),
            "schedule": prov.get("acquisition_schedule"), "sources": prov.get("acquisition_source_weights"),
            "units": {name: unit_identities[name] for name in selected[row]},
            "cells": {name: list(formats) for name, formats in payload["costs"].items()}}
    return records


def _merge_acquisition_settings(identities, states_by_row, *, acquisition, scope_groups):
    if acquisition is None and not any(
            "acquisition_origin" in (identity.get("settings") or ())
            or "acquisition_schedule" in (identity.get("settings") or ())
            or any("acquisition_source_weight" in unit
                   for unit in (identity.get("units") or {}).values())
            for identity in identities.values()):
        return None
    records = {}
    for row, identity in identities.items():
        settings = identity.get("settings") or {}
        if (settings.get("menu_mode") != "research" or settings.get("max_rounds") != 1
                or settings.get("rate_band") is not None or settings.get("exhaustive_rate_grid")):
            raise MergeRefused(f"{row}: acquisition checkpoint regime differs")
        for name, state in states_by_row[row].items():
            for anchor in state["anchors"]:
                require_campaign_acquisition_anchor(CampaignAnchor(**anchor), settings["acquisition_schedule"])
                record = state["wire_records"].get(anchor["format_name"], {})
                if record.get("identity", {}).get("source") != identity["units"][name]["weight"]:
                    raise MergeRefused(f"{row}: acquisition wire source differs from checkpoint unit {name}")
            if state.get("unservable") or set(state["wire_records"]) != {
                    anchor["format_name"] for anchor in state["anchors"]}:
                raise MergeRefused(f"{row}: acquisition has unservable/extra wire work for {name}")
        records[row] = {"origin": settings.get("acquisition_origin"),
            "schedule": settings.get("acquisition_schedule"), "units": identity["units"],
            "sources": {name: unit.get("acquisition_source_weight") for name, unit in identity["units"].items()},
            "cells": {name: [anchor["format_name"] for anchor in state["anchors"]]
                      for name, state in states_by_row[row].items()}}
    merged = merge_acquisition_rows(records, acquisition=acquisition, scope_groups=scope_groups)
    if merged is None:
        return None
    common = require_equal("checkpoint settings outside exact acquisition scope", {
        row: {key: value for key, value in identity["settings"].items()
              if key not in {"acquisition_origin", "acquisition_schedule"}}
        for row, identity in identities.items()})
    return {**common, "acquisition_origin": merged["origin"], "acquisition_schedule": merged["schedule"]}


def expected_expert_partition_coverage(census: dict, plan_coverage, *, rate_band, max_rounds) -> dict:
    """Derive complete expert coverage from the plan, never surviving outputs."""
    expected = {} if plan_coverage is None else plan_coverage.get("expected_partitions", {})
    if not isinstance(expected, dict):
        raise MergeRefused("expert partition plan is not a row-selection mapping")
    if not expected:
        return {}
    from prismaquant.tessera_expert_projection import carried_units
    _source, records, stack_of = carried_units(census["expert_projection"])
    covered, indices, counts = {}, {}, {}
    try:
        for row_id, selection in expected.items():
            validate_unit_selection(selection, path=row_id)
            if selection["schema"] != UNITS_SCHEMA_V3:
                raise RuntimeError("partition plan carries a legacy selection")
            select_anchor_groups(selection, census["anchor_groups"], where=row_id)
            subsets = expert_partition_members(selection, unit_records=records,
                stack_of=stack_of, rate_band=rate_band, max_rounds=max_rounds)
            entry = selection["groups"][0]
            key, part = entry["key"], entry["partition"]
            members = set(subsets[key])
            if covered.setdefault(key, set()) & members:
                raise RuntimeError(f"expert partition overlap for {key}")
            covered[key].update(members)
            if part["index"] in indices.setdefault(key, set()):
                raise RuntimeError(f"expert partition repeats an index for {key}")
            indices[key].add(part["index"])
            counts.setdefault(key, set()).add(part["count"])
        for key, members in covered.items():
            if members != set(census["anchor_groups"][key]) or len(counts[key]) != 1:
                raise RuntimeError(f"expert partition incomplete roster for {key}")
            if indices[key] != set(range(next(iter(counts[key])))):
                raise RuntimeError(f"expert partition incomplete index coverage for {key}")
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise MergeRefused(f"invalid expert partition plan: {exc}") from exc
    return expected


def merge_payloads(row_payloads: dict, *, census: dict, capture_sha256: str,
                   plan_coverage: dict | None = None, acquisition=None,
                   acquisition_unit_identities=None) -> dict:
    """One cost payload from N rows, refusing anything they do not share.

    The merged table is the monolith's on every field the monolith's rows would
    carry: the union of the per-unit prices, one Hessian identity, and a
    coverage block rebuilt over the **scope** rather than over any one row's
    selection.

    ``plan_coverage`` is what :func:`declared_coverage` read from the plan.
    Without it the rows must price every anchor group of the scope; with it
    they must price exactly the groups the plan declares, and the merged table
    says which scope groups it leaves unpriced and why under
    ``provenance.coverage``, with ``unit_selection.selected`` True.

    An acquisition also requires ``acquisition_unit_identities`` from the
    authenticated checkpoint merge. Its existing full producer menus bind
    exact deferred-domain claims; payload provenance alone cannot supply them.
    """
    # The keys a merged payload must land under are the ones the campaign and
    # the allocation share.  Spelling them here as literals is how a merge
    # writes a block nothing reads: POPULATION_KEY is "population", not
    # "tessera_population", and the allocation reads only the former.
    from prismaquant.tessera_expert_projection import (
        EXPERT_WIRES_KEY, POPULATION_KEY, PROJECTION_KEY)

    for row_id, payload in row_payloads.items():
        if payload.get("schema") != SCHEMA:
            raise MergeRefused(f"{row_id}: not a {SCHEMA} payload")

    provenances = {row: payload["provenance"] for row, payload in row_payloads.items()}
    acquisition_union = None
    if acquisition is not None or any(any(key in prov for key in (
            "acquisition", "acquisition_schedule", "acquisition_source_weights")) for prov in provenances.values()):
        acquisition_union = merge_acquisition_rows(payload_acquisition_records(
            row_payloads, unit_identities=acquisition_unit_identities),
            acquisition=acquisition, scope_groups=census.get("anchor_groups"))
    for row_id, provenance in provenances.items():
        if provenance.get("research_exact_member_scope") is not None:
            raise MergeRefused(
                f"{row_id}: research exact-member scalar cannot be merged as a full group or stack estimate")
    reference = next(iter(provenances.values()))
    expected_partitions = expected_expert_partition_coverage(census, plan_coverage,
        rate_band=reference.get("rate_band"), max_rounds=reference.get("max_rounds", 0))
    actual_partitions = set()
    for row_id, prov in provenances.items():
        units = prov.get("unit_selection")
        try:
            validate_unit_selection(units, path=row_id)
            select_anchor_groups(units, census.get("anchor_groups", reference["campaign_scope"]["anchor_groups"]), where=row_id)
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            raise MergeRefused(f"{row_id}: invalid bound selection: {exc}") from exc
        if units["schema"] == UNITS_SCHEMA_V3:
            declared = expected_partitions.get(row_id)
            if (declared is None or prov.get("seed_checkpoint") is not None
                    or units.get("selected", True) is not True
                    or {key: value for key, value in units.items() if key != "selected"}
                    != {key: value for key, value in declared.items() if key != "selected"}):
                raise MergeRefused(f"{row_id}: expert partition differs from its declared plan")
            if set(row_payloads[row_id]["costs"]) != selection_priced_units(units)[0]:
                raise MergeRefused(f"{row_id}: expert partition prices differ from its members")
            from prismaquant.tessera_formats import parse_tessera_format_name
            rate = units["groups"][0]["partition"]["rate_q256"]
            priced_formats = set()
            for name, prices in row_payloads[row_id]["costs"].items():
                if not isinstance(prices, dict) or not prices:
                    raise MergeRefused(f"{row_id}: expert partition has no prices for {name}")
                for fmt in prices:
                    try:
                        parsed = parse_tessera_format_name(fmt)
                    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                        raise MergeRefused(f"{row_id}: invalid expert partition format {fmt!r}") from exc
                    if parsed is None or parsed[1] != rate:
                        raise MergeRefused(f"{row_id}: expert partition price {fmt!r} is not its pinned rung")
                    priced_formats.add(fmt)
            if set(row_payloads[row_id]["formats"]) != priced_formats:
                raise MergeRefused(f"{row_id}: expert partition format roster differs from its prices")
            actual_partitions.add(row_id)
    if actual_partitions != set(expected_partitions):
        raise MergeRefused("expert partition row coverage incomplete")
    family_policies, restricted_structures = {}, {}
    for row_id, prov in provenances.items():
        restriction = prov.get("family_restriction")
        if restriction is None:
            family_policies[row_id] = None
            continue
        if not isinstance(restriction, dict) or set(restriction) != {"policy", "structure_by_unit"}:
            raise MergeRefused(f"{row_id}: invalid family restriction provenance")
        try:
            policy = parse_family_restriction(restriction["policy"])
        except (ValueError, TypeError) as exc:
            raise MergeRefused(f"{row_id}: invalid family restriction policy: {exc}") from exc
        structures = restriction["structure_by_unit"]
        members = selection_priced_units(prov["unit_selection"])[0]
        if (policy is None or not isinstance(structures, dict) or set(structures) != members
                or any(s not in ("dense", "routed_moe") for s in structures.values())):
            raise MergeRefused(f"{row_id}: family restriction must cover exact selected unit structures")
        for name, structure in structures.items():
            if name in restricted_structures:
                raise MergeRefused(f"{row_id}: family restriction repeats selected unit {name}")
            if name in prov["campaign_scope"]["expert_targets"] and structure != "routed_moe":
                raise MergeRefused(f"{row_id}: family restriction contradicts projected expert {name}")
            restricted_structures[name] = structure
        family_policies[row_id] = policy
    family_policy = require_equal("provenance.family_restriction.policy", family_policies)
    for field in SHARED_PROVENANCE:
        require_equal(
            f"provenance.{field}",
            {row: prov.get(field) for row, prov in provenances.items()},
        )
    require_equal(
        "provenance.hessian.calibration_identity",
        {row: prov["hessian"]["calibration_identity"]
         for row, prov in provenances.items()},
    )
    require_equal(
        "provenance.activation_static_scales.policy",
        {row: prov["activation_static_scales"]["policy"]
         for row, prov in provenances.items()},
    )
    require_equal(
        "currency",
        {row: payload["currency"] for row, payload in row_payloads.items()},
    )
    scope = require_equal(
        "provenance.campaign_scope",
        {row: {key: value for key, value in prov["campaign_scope"].items()}
         for row, prov in provenances.items()})

    # One Hessian identity across every priced row, on every field but the
    # capture digest the merge is about to replace.
    for row_id, payload in row_payloads.items():
        for field in SHARED_HESSIAN:
            require_equal(
                f"hessian_identity.{field}",
                {f"{row_id}:{index}": identity.get(field)
                 for index, identity in enumerate(hessian_identities(payload))}
                or {row_id: None})
    for field in SHARED_HESSIAN:
        require_equal(
            f"hessian_identity.{field} across rows",
            {row: (hessian_identities(payload)[0].get(field)
                   if hessian_identities(payload) else None)
             for row, payload in row_payloads.items()})

    # Coverage: every expected group priced exactly once.  The expectation is
    # the scope, or the groups the plan declares when it declares fewer.
    selected: dict[str, str] = {}
    selection_entries = {}
    for row_id, prov in provenances.items():
        for entry in prov["unit_selection"]["groups"]:
            key = entry["key"]
            if key in selected and not (row_id in expected_partitions and selected[key] in expected_partitions):
                raise MergeRefused(
                    f"anchor group {key!r} is priced by both {selected[key]} and {row_id}")
            if key not in scope["anchor_groups"] or sorted(entry["members"]) != sorted(scope["anchor_groups"][key]):
                raise MergeRefused(f"{row_id}: selection {key!r} differs from campaign scope")
            selected[key] = row_id
            selection_entries[key] = entry
    if plan_coverage is None:
        expected = set(scope["anchor_groups"])
        expected_by = "scope"
    else:
        expected = set(plan_coverage["expected_groups"])
        expected_by = "plan"
        outside = sorted(expected - set(scope["anchor_groups"]))
        if outside:
            raise MergeRefused(
                f"the plan declares {len(outside)} anchor group(s) outside the campaign "
                "scope: " + ", ".join(outside[:8]))
    missing = sorted(expected - set(selected))
    if missing:
        raise MergeRefused(
            f"the rows do not cover {len(missing)} anchor group(s) of the {expected_by}: "
            + ", ".join(missing[:8]))
    undeclared = sorted(set(selected) - expected)
    if undeclared:
        raise MergeRefused(
            f"the rows price {len(undeclared)} anchor group(s) the plan does not declare: "
            + ", ".join(undeclared[:8]))
    unpriced = sorted(set(scope["anchor_groups"]) - expected)

    sampled = any("stack_samples" in entry or entry.get("sampled")
                  for entry in selection_entries.values())
    merged_selection = {
        "schema": "prismaquant.tessera_campaign_units.v2" if sampled else "prismaquant.tessera_campaign_units.v1",
        # False is the whole-scope claim a monolith writes; a table the plan
        # left partly unpriced is a selection and says so.
        "selected": bool(unpriced),
        "groups": [selection_entries[key] if sampled else
                   {"key": key, "members": list(scope["anchor_groups"][key])}
                   for key in sorted(selection_entries)],
    }
    stack_samples, profile = {}, None
    if sampled:
        from prismaquant.model_profiles import detect_profile
        profile = detect_profile(next(iter(provenances.values()))["model"])
        stack_samples = selection_stack_samples(merged_selection, profile)
    costs: dict[str, dict] = {}
    loo: dict[str, dict] = {}
    surfaces: dict[str, dict] = {}
    anchor_counts: dict[str, dict] = {}
    menu_sizes: dict[str, int] = {}
    anchor_groups: dict[str, list] = {}
    non_interpolable: list[dict] = []
    expert_wires: dict[str, dict] = {}
    # Evidence, not prices: rows a shard adopted from another campaign whose
    # rungs its menu does not admit.  The union is taken here for the same
    # reason the prices are -- the reference row's block describes one slice.
    unservable: dict[str, dict] = {}
    formats: set[str] = set()
    stopped_early = False
    wall_seconds = 0.0
    rounds_run = 0
    seeds: list[dict] = []
    projection_block = None
    serving_by_unit: dict[str, dict] = {}
    serving_target = None
    for row_id in sorted(row_payloads):
        payload = row_payloads[row_id]
        prov = payload["provenance"]
        for qname, rows in payload["costs"].items():
            if qname in costs:
                raise MergeRefused(f"unit {qname} is priced by more than one row")
            costs[qname] = {
                fmt: {**row, "hessian_identity": {**row["hessian_identity"],
                                                  "capture_sha256": capture_sha256}}
                if "hessian_identity" in row else row
                for fmt, row in rows.items()
            }
        formats.update(payload["formats"])
        loo.update(payload["leave_one_anchor_out"])
        non_interpolable.extend(payload["non_interpolable"])
        surfaces.update(prov["surfaces"])
        if row_id in expected_partitions:
            expected_groups = {entry["key"]: entry["partition"]["members"]
                               for entry in expected_partitions[row_id]["groups"]}
            if prov["anchor_groups"] != expected_groups:
                raise MergeRefused(f"{row_id}: expert partition anchor groups differ from its declared keys/members")
            for key, names in prov["anchor_groups"].items():
                anchor_groups[key] = sorted(set(anchor_groups.get(key, ())) | set(names))
        else:
            anchor_groups.update(prov["anchor_groups"])
        anchor_counts.update(payload["anchor_counts"])
        menu_sizes.update(payload["menu_sizes"])
        expert_wires.update(payload.get(EXPERT_WIRES_KEY, {}))
        for qname, rungs in (prov.get("unservable") or {}).items():
            held = unservable.setdefault(qname, {})
            for fmt, record in rungs.items():
                if fmt in held and held[fmt] != record:
                    raise MergeRefused(
                        f"{row_id}: it carries different unservable evidence for "
                        f"{qname} {fmt} than an earlier row")
                held[fmt] = record
        stopped_early = stopped_early or bool(prov["stopped_early"])
        wall_seconds += float(prov["wall_seconds"])
        rounds_run = max(rounds_run, int(prov["rounds_run"]))
        if prov.get("seed_checkpoint"):
            seeds.append({"row": row_id, **prov["seed_checkpoint"]})
        projection = prov.get(PROJECTION_KEY)
        if projection is not None:
            # Every row carries the SCOPE's projection block, because the
            # allocation rebinds the producer's answer over every stack the
            # block names. Two different blocks would be two producer answers.
            if projection_block is None:
                projection_block = projection
            elif projection_block != projection:
                raise MergeRefused(
                    f"{row_id}: its producer expert projection differs from the "
                    "other rows'; they did not read one census projection")
        serving = prov.get("tessera_serving_scope")
        if serving:
            serving_target = serving["target"] if serving_target is None else serving_target
            if serving["target"] != serving_target:
                raise MergeRefused("rows disagree on the serving target")
            serving_by_unit.update(serving["by_unit"])

    reference = provenances[sorted(provenances)[0]]
    provenance = {key: value for key, value in reference.items()}
    provenance.pop("identity_migration", None)
    if acquisition_union is not None:
        provenance.update(acquisition=acquisition_union["origin"],
            acquisition_schedule=acquisition_union["schedule"],
            acquisition_source_weights=acquisition_union["sources"])
    carried_migration = merge_identity_migrations(
        {row: prov.get("identity_migration") for row, prov in provenances.items()},
        error=MergeRefused)
    if carried_migration is not None:
        provenance["identity_migration"] = carried_migration
    if family_policy is not None:
        provenance["family_restriction"] = {"policy": family_policy,
            "structure_by_unit": dict(sorted(restricted_structures.items()))}
    provenance.update({
        "surfaces": dict(sorted(surfaces.items())),
        "anchor_groups": dict(sorted(anchor_groups.items())),
        "unservable": {name: {fmt: rungs[fmt] for fmt in sorted(rungs)}
                       for name, rungs in sorted(unservable.items())},
        "stopped_early": stopped_early,
        "wall_seconds": wall_seconds,
        "rounds_run": rounds_run,
        "unit_selection": merged_selection,
        "activation_static_scales": dict(reference["activation_static_scales"]),
        "hessian": {**reference["hessian"], "capture_sha256": capture_sha256},
        "campaign_fanout": {
            "schema": PLAN_SCHEMA,
            "rows": {row_id: sorted(
                entry["key"] for entry in provenances[row_id]["unit_selection"]["groups"])
                for row_id in sorted(provenances)},
            "seed_checkpoints": seeds,
        },
    })
    if plan_coverage is not None:
        provenance["coverage"] = {
            "schema": COVERAGE_SCHEMA,
            "scope_groups": len(scope["anchor_groups"]),
            "priced_groups": len(expected),
            "unpriced_groups": unpriced,
            "excluded_rows": list(plan_coverage["excluded_rows"]),
            "reason": plan_coverage["reason"],
            **({"deferred_acquisition_groups": plan_coverage["deferred_acquisition_groups"],
                "requested_groups": plan_coverage["requested_groups"]}
               if "deferred_acquisition_groups" in plan_coverage else {}),
        }
    if any("no_admitted_rung" in prov for prov in provenances.values()):
        provenance["no_admitted_rung"] = sorted({name for prov in provenances.values()
                                               for name in prov.get("no_admitted_rung", [])})
    if any("unit_selection_sample" in prov for prov in provenances.values()):
        audit, probabilities = set(), {}
        for row_id, prov in provenances.items():
            sample = prov.get("unit_selection_sample", {})
            audit.update(sample.get("audit_units", []))
            for name, probability in sample.get("inclusion_probability", {}).items():
                if name in probabilities and probabilities[name] != probability:
                    raise MergeRefused(f"{row_id}: different inclusion probability for {name}")
                probabilities[name] = probability
        provenance["unit_selection_sample"] = {
            "audit_units": sorted(audit), "inclusion_probability": dict(sorted(probabilities.items()))}
    if serving_target is not None:
        provenance["tessera_serving_scope"] = {
            "target": serving_target, "by_unit": dict(sorted(serving_by_unit.items()))}

    payload = {
        **{key: value for key, value in row_payloads[sorted(row_payloads)[0]].items()
           if key not in {"costs", "formats", "leave_one_anchor_out",
                          "non_interpolable", "menu_sizes", "anchor_counts",
                          "provenance", EXPERT_WIRES_KEY}},
        "schema": SCHEMA,
        "costs": dict(sorted(costs.items())),
        "formats": sorted(formats),
        "leave_one_anchor_out": dict(sorted(loo.items())),
        "non_interpolable": canonical_refusals(non_interpolable),
        "menu_sizes": dict(sorted(menu_sizes.items())),
        "anchor_counts": dict(sorted(anchor_counts.items())),
        "provenance": provenance,
    }
    if expert_wires:
        payload[EXPERT_WIRES_KEY] = dict(sorted(expert_wires.items()))
    population = ExpertPopulation(
        members=(),
        declared={stack: {name: tuple(shape) for name, shape in units.items()}
                  for stack, units in scope["declared_stacks"].items()},
        packed_in_scope={name: tuple(shape) for name, shape
                         in scope["packed_in_scope"].items()},
        omitted_outside_layer_stride={
            name: tuple(shape) for name, shape
            in scope["packed_outside_layer_stride"].items()},
    )
    # Overwrites the reference row's block, which describes that row's slice.
    payload["provenance"][POPULATION_KEY] = campaign_population_block(
        dense_targets=scope["dense_targets"], expert_targets=scope["expert_targets"],
        dense_all=scope["dense_all"], pinned=scope["pinned"],
        population=population, layer_stride=int(reference["layer_stride"]),
        costs=payload["costs"], menus=menu_sizes,
        stack_samples=stack_samples, profile=profile)
    return payload


def merge_checkpoint(row_dirs: dict, out_manifest: Path, *, acquisition=None, scope_groups=None) -> dict:
    """One journal from the rows', under the identity their union describes.

    The rows' identities differ only where the selection does: the ``units``
    map, the serving scope's ``by_unit`` and the producer's projected stacks.
    Everything else must already be equal, and the union of the three is what a
    whole-scope run of this code computes -- so the merged journal is one a
    later whole-scope invocation can resume, and refuses by field if it is not.
    """
    from prismaquant.cost_stage_checkpoint import (
        atomic_write_bytes, canonical_json, canonical_json_sha256, unit_path,
        MANIFEST_SCHEMA, prepare_journal, write_unit,
    )

    identities = {}
    migrations = {}
    states: dict[str, dict] = {}
    states_by_row = {}
    stage = STAGE
    for row_id in sorted(row_dirs):
        manifest_path = Path(row_dirs[row_id]) / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        identities[row_id] = manifest["identity"]
        migrations[row_id] = manifest.get("identity_migration")
        parts = manifest_path.with_name(PARTS_NAME)
        listed: list[str] = []
        for entry in manifest["units"]:
            qname = entry["qname"]
            listed.append(qname)
            shard = parts / entry["file"]
            if not shard.is_file():
                raise MergeRefused(
                    f"{row_id}: its journal names {qname} and the shard "
                    f"{shard} is not there; the row's anchors would be "
                    "dropped from the merged journal")
            if shard != unit_path(parts, qname):
                raise MergeRefused(f"{row_id}: unit {qname} names a noncanonical shard")
        expected = set(manifest["identity"]["units"])
        if len(listed) != len(set(listed)) or set(listed) != expected:
            raise MergeRefused(
                f"{row_id}: manifest units differ from its checkpoint identity units")
        # Reuse the journal's reader: it validates the manifest and every
        # envelope before returning state. Rehashing unchecked payload bytes
        # here would turn corrupt or foreign shards into a trusted journal.
        try:
            _, _, completed = prepare_journal(
                parts, stage=stage, resume=True, identity=manifest["identity"],
                qnames=sorted(expected), manifest_path=manifest_path)
        except RuntimeError as exc:
            raise MergeRefused(f"{row_id}: {exc}") from exc
        states_by_row[row_id] = completed
        for qname, state in completed.items():
            if qname in states:
                raise MergeRefused(f"unit {qname} has a journal shard in two rows")
            states[qname] = state

    acquisition_settings = _merge_acquisition_settings(identities, states_by_row,
        acquisition=acquisition, scope_groups=scope_groups)
    if acquisition_settings is not None:
        identities = {row: {**identity, "settings": acquisition_settings}
                      for row, identity in identities.items()}
    merged_identity = None
    for row_id, identity in sorted(identities.items()):
        if merged_identity is None:
            merged_identity = {key: value for key, value in identity.items()}
            merged_identity["units"] = dict(identity["units"])
            continue
        for key in sorted(set(merged_identity) | set(identity)):
            if key in RECONCILED_IDENTITY_KEYS:
                continue
            if (key not in merged_identity or key not in identity
                    or merged_identity[key] != identity[key]):
                raise MergeRefused(
                    f"{row_id}: checkpoint identity differs at {key!r}")
        for name, unit in identity["units"].items():
            if name in merged_identity["units"] and merged_identity["units"][name] != unit:
                raise MergeRefused(f"{row_id}: two rows bind different inputs for {name}")
            merged_identity["units"][name] = unit
        if "stack_sampling_identity" in merged_identity or "stack_sampling_identity" in identity:
            combined = dict(merged_identity.get("stack_sampling_identity", {}))
            for name, sample in identity.get("stack_sampling_identity", {}).items():
                if name in combined and combined[name] != sample:
                    raise MergeRefused(f"{row_id}: different stack_sampling_identity for {name}")
                combined[name] = sample
            merged_identity["stack_sampling_identity"] = dict(sorted(combined.items()))
        merged_identity["serving_scope"] = _merge_scope(
            merged_identity.get("serving_scope"), identity.get("serving_scope"), row_id)
        merged_identity["expert_projection"] = _merge_projection(
            merged_identity.get("expert_projection"), identity.get("expert_projection"),
            row_id)
        merged_identity["family_restriction"] = _merge_family_restriction(
            merged_identity.get("family_restriction"),
            identity.get("family_restriction"), row_id)
        if merged_identity["family_restriction"] is None:
            del merged_identity["family_restriction"]
    merged_identity["units"] = dict(sorted(merged_identity["units"].items()))

    canonical = canonical_json(merged_identity, where="merged campaign identity")
    identity_sha256 = canonical_json_sha256(canonical, where="merged campaign identity")
    parts = out_manifest.with_name(out_manifest.name + ".parts")
    for qname, state in sorted(states.items()):
        write_unit(parts, stage=stage, qname=qname,
                   identity_sha256=identity_sha256, state=state)
    manifest = {
        "schema": MANIFEST_SCHEMA, "stage": stage,
        "identity_sha256": identity_sha256, "identity": canonical,
        "units": [{"qname": qname,
                   "file": str(unit_path(parts, qname).relative_to(parts))}
                  for qname in sorted(merged_identity["units"])],
    }
    carried = merge_identity_migrations(migrations, error=MergeRefused)
    if carried is not None:
        manifest["identity_migration"] = carried
    atomic_write_bytes(out_manifest, DIRECT_UTF8_INDENT2_STRICT.encoded(manifest))
    return manifest


def _merge_family_restriction(left, right, row_id):
    """One policy, and the union of the rows' per-unit structure maps.

    ``structure_by_unit`` is keyed by the row's OWN selected units, so a dense
    row and a routed row of one campaign never carry the same map. Comparing
    the whole restriction for equality therefore refuses every census that
    fans dense and routed units onto different rows, which is every GLM census
    (RobTand/prismaquant#487). ``merge_payloads`` already reconciles the same
    field this way; this is the journal side of it.
    """
    if left is None or right is None:
        if left != right:
            raise MergeRefused(
                f"{row_id}: one row restricts families and another does not")
        return left
    if left["policy"] != right["policy"]:
        raise MergeRefused(f"{row_id}: rows disagree on the family restriction policy")
    for name in left["structure_by_unit"].keys() & right["structure_by_unit"].keys():
        if left["structure_by_unit"][name] != right["structure_by_unit"][name]:
            raise MergeRefused(
                f"{row_id}: different restricted structure for {name}")
    return {"policy": left["policy"],
            "structure_by_unit": dict(sorted(
                {**left["structure_by_unit"], **right["structure_by_unit"]}.items()))}


def _merge_scope(left, right, row_id):
    if left is None or right is None:
        if left != right:
            raise MergeRefused(f"{row_id}: one row has a serving scope and another does not")
        return left
    if left["target"] != right["target"]:
        raise MergeRefused(f"{row_id}: rows disagree on the serving target")
    for name in left["by_unit"].keys() & right["by_unit"].keys():
        if left["by_unit"][name] != right["by_unit"][name]:
            raise MergeRefused(f"{row_id}: different serving context for {name}")
    return {"target": left["target"],
            "by_unit": dict(sorted({**left["by_unit"], **right["by_unit"]}.items()))}


def _merge_projection(left, right, row_id):
    if left is None:
        return right
    if right is None:
        return left
    if left["source"] != right["source"]:
        raise MergeRefused(f"{row_id}: rows projected different source checkpoints")
    for name in left["stacks"].keys() & right["stacks"].keys():
        if left["stacks"][name] != right["stacks"][name]:
            raise MergeRefused(f"{row_id}: different producer projection for stack {name}")
    stacks = {**left["stacks"], **right["stacks"]}
    return {"source": left["source"], "stacks": dict(sorted(stacks.items()))}


def require_campaign_receipts(workspace: Path, expected: int) -> tuple[str, ...]:
    """Refuse to merge a row the fleet did not report as executed.

    A ``cost.pkl`` on disk says a process wrote a file; the fleet's row table
    says which action it was, where it ran and what it exited with.  Both, or
    neither.
    """
    path = workspace / "receipts.json"
    if not path.is_file():
        raise MergeRefused(
            f"no fleet receipts at {path}; submit the manifest before merging")
    receipts = json.loads(path.read_text())
    rows = receipts.get("rows") or []
    if len(rows) != expected:
        raise MergeRefused(
            f"{path} reports {len(rows)} rows and the plan has {expected}")
    # ``pbwait.verdict`` is the fleet's own reading of the table: 0 when every
    # row's work is done, and a memoized ``cache_hit`` counts as done there.
    # It is the gate, because a re-submitted row that was already priced
    # reports no launcher status of its own and renders ``rc`` as ``-``.
    if receipts.get("returncode") not in {0, "0"}:
        raise MergeRefused(
            f"{path} records pbcampaign exit {receipts.get('returncode')!r}; "
            "not every row is done")
    failed = [row for row in rows
              if row.get("rc") not in {"0", 0, "-", ""}]
    if failed:
        raise MergeRefused(
            "the fleet reports a non-zero exit for "
            + ", ".join(f"{row.get('key')} (rc={row.get('rc')})" for row in failed))
    where = sorted({f"{row.get('host') or '?'} ({row.get('status')})"
                    for row in rows})
    return tuple(where)


def parse_rate_band(text) -> "tuple[int, int] | None":
    """CLI ``"lo,hi"`` or a strict integer provenance pair, in q256 units."""
    if text is None or str(text).strip() == "":
        return None
    if isinstance(text, (list, tuple)):
        if len(text) != 2 or any(type(value) is not int for value in text):
            raise RuntimeError(f"--rate-band {text!r}: want two integer q256 values")
        parts = text
    else:
        parts = str(text).split(",")
    if len(parts) != 2:
        raise RuntimeError(f"--rate-band {text!r}: want 'lo,hi' in q256 units")
    try:
        lo, hi = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise RuntimeError(f"--rate-band {text!r}: not two integers") from exc
    if lo <= 0 or hi <= 0 or lo > hi:
        raise RuntimeError(f"--rate-band {text!r}: want 0 < lo <= hi")
    return lo, hi


def audit_subsample(drawn: "Sequence[str]", *, rate: int = 10, seed: int = 0,
                    stack: str = "") -> list[str]:
    """A simple random subsample of the draw: one unit in ``rate``, at least 1.

    Drawn from a stream of its own, so changing the audit fraction cannot
    disturb the sample the prices are built from -- the audit buys evidence
    about interpolation, and must not be able to move the estimate.
    """
    units = sorted(drawn)
    if not units:
        return []
    count = max(1, len(units) // int(rate))
    stream = _keystream(seed, f"audit:{stack}")
    return sorted(_permute(units, stream)[:count])


def campaign_acquisition_scope(acquisition, scope_groups, *, selected=None):
    """Require every real atomic member and exactly the requested scope."""
    requests, sources = acquisition["requests"], acquisition["source_weights"]
    flat = [name for members in scope_groups.values() for name in members]
    known = set(flat)
    if len(flat) != len(known):
        raise ValueError("acquisition actual atomic scope overlaps or repeats members")
    if not requests or set(requests) - known:
        raise ValueError("acquisition names empty or unknown unit scope")
    keys = [key for key, members in sorted(scope_groups.items()) if set(members) & set(requests)]
    expanded = {name for key in keys for name in scope_groups[key]}
    if set(requests) != expanded or set(sources) != expanded:
        raise ValueError("acquisition requires request and source identity for every atomic member")
    if selected is not None and set(selected) != expanded:
        raise ValueError("acquisition selected scope differs from requested atomic-expanded scope")
    return keys, expanded


def campaign_acquisition_row_scope(acquisition, scope_groups, *, selected=None):
    """Project only an explicit complete actual --units selection after full validation."""
    keys, expanded = campaign_acquisition_scope(acquisition, scope_groups)
    if selected is None:
        return acquisition, keys, expanded
    if isinstance(selected, (str, bytes)):
        raise ValueError("acquisition row selection requires explicit unit names, not a string")
    listed = list(selected)
    selected = set(listed)
    if len(listed) != len(selected):
        raise ValueError("acquisition row selection repeats a unit")
    row_keys = [key for key in keys if selected.intersection(scope_groups[key])]
    row_members = {name for key in row_keys for name in scope_groups[key]}
    if not selected or row_members != selected or not selected <= expanded:
        raise ValueError("acquisition row selection must contain complete requested actual atomic groups")
    from .tessera_full_domain_acquisition import project_joint_campaign_acquisition
    projected = project_joint_campaign_acquisition(acquisition, units=sorted(selected))
    campaign_acquisition_scope(projected, scope_groups, selected=selected)
    return projected, row_keys, row_members


def requested_acquisition_schedule(groups, requests):
    """Exact atomic-expanded requested work only, not an assertion of legal prices."""
    result = {}
    for _key, members in sorted(groups.items()):
        families = sorted({family for name in members for family in requests[name]})
        union = {family: sorted({q for name in members for q in requests[name].get(family, [])})
                 for family in families}
        for name in members:
            result[name] = {family: list(qs) for family, qs in union.items()}
    return dict(sorted(result.items()))


def campaign_acquisition_deferred_families(schedule, menu_families):
    """Exact deferred producer-menu/request families, never a price claim."""
    return sorted((set(menu_families) | set(schedule))
                  - {family for family, qs in schedule.items() if qs})


def require_campaign_acquisition_anchor(anchor, schedule):
    if (anchor.qname not in schedule or anchor.body_rate_q256 not in
            schedule[anchor.qname].get(anchor.family, [])):
        raise ValueError("acquisition refuses extra checkpoint/seed anchor work")


def expert_partition_members(selection: Mapping, *, unit_records: Mapping,
                             stack_of: Mapping, rate_band, max_rounds,
                             seeded: bool = False) -> dict[str, list[str]]:
    """Validate deterministic, expert-complete chunks against producer records.

    This is not sampling: every projection of each selected expert is priced,
    and the complete original group remains the identity/menu frame.
    """
    entries = selection.get("groups", [])
    if selection.get("schema") != UNITS_SCHEMA_V3:
        if any("partition" in entry for entry in entries):
            raise RuntimeError("expert partition requires the v3 selection schema")
        return {}
    validate_unit_selection(selection, path="expert partition")
    band = parse_rate_band(rate_band) if rate_band else None
    if band is None or band[0] != band[1] or type(max_rounds) is not int or max_rounds != 1 or seeded:
        raise RuntimeError("expert partition requires pinned rate-band, max-rounds=1 and no seeds")
    if len(entries) != 1:
        raise RuntimeError("expert partition selection must name one routed group")
    result = {}
    for entry in entries:
        key = entry["key"]
        full = entry["members"]
        part = entry.get("partition")
        fields = {"schema", "experts_per_row", "index", "count", "rate_q256", "members"}
        if (not key.startswith("s:") or len(set(full)) != len(full)
                or any(field in entry for field in ("sampled", "audit", "inclusion_probability", "stack_samples"))
                or not isinstance(part, Mapping) or set(part) != fields
                or part.get("schema") != EXPERT_PARTITION_SCHEMA):
            raise RuntimeError("expert partition requires an unsampled routed group and a closed descriptor")
        if (any(type(part[name]) is not int for name in ("experts_per_row", "index", "count", "rate_q256"))
                or part["experts_per_row"] <= 0 or part["index"] < 0
                or part["rate_q256"] != band[0]):
            raise RuntimeError("expert partition has invalid size/index/rate")
        by_expert = {}
        for name in full:
            record = unit_records.get(name)
            if (not isinstance(record, Mapping) or type(record.get("expert")) is not int
                    or record["expert"] < 0 or "s:" + str(stack_of.get(name)) != key):
                raise RuntimeError(f"expert partition {key}: {name} is not a producer-bound expert")
            by_expert.setdefault(record["expert"], []).append(name)
        experts = sorted(by_expert)
        # Each chunk carries every role/shape used by the full group's menu
        # cache. Heterogeneous expert geometry needs a separately charged
        # metadata contract rather than enlarging this row's cache silently.
        signatures = {tuple(sorted((unit_records[name].get("projection"),
                                    unit_records[name].get("rows"), unit_records[name].get("cols"))
                                   for name in names)) for names in by_expert.values()}
        if len(signatures) != 1:
            raise RuntimeError("expert partition requires uniform per-expert projection geometry")
        size = part["experts_per_row"]
        count = (len(experts) + size - 1) // size
        start = part["index"] * size
        expected = sorted(name for expert in experts[start:start + size] for name in by_expert[expert])
        if (part["count"] != count or part["index"] >= count or not expected
                or part["members"] != expected):
            raise RuntimeError("expert partition does not match its complete deterministic expert chunk")
        result[key] = expected
    return result


def campaign_acquisition_origin(acquisition, schedule, menus):
    return {**acquisition["identity"],
        "purpose": "actual_scalar_render_journal_and_wire_preparation",
        "currency": CURRENCY,
        "expanded_actual_work_count": sum(len(qs) for families in schedule.values() for qs in families.values()),
        "deferred_domain": {name: campaign_acquisition_deferred_families(
            schedule[name], (r.family for r in menus[name])) for name in sorted(schedule)}}
