"""Price Tessera's continuous rate axis by anchor campaign.

Four families address thousands of rungs at every shape, and one encode is
seconds.
Rendering the menu is not a cost model, it is a week of a shared GPU.  So this
stage renders a small **measured anchor** set per (unit, family) and fills the
rest of the axis from ``tessera_rate_surface``, which interpolates monotonically
in ``(q256, log2 dloss)`` between bracketing anchors and **refuses to
extrapolate**.  How many anchors that is, is not a constant: round 1 measures
the two endpoints and the middle of each family's legal range, and every
following round adds one anchor wherever leave-one-anchor-out says the
interpolation is worst, until the LOO gate closes or the round budget runs out.

Priced as served
----------------
Every measured rung is scored by the production scorer,
``production_weight_cache._local_forward_render_score`` -- the same function the
render-cost stage uses -- with the **route's own activation quantiser** applied
to the calibration rows.  A Tessera rung on the E2M1 grid over a block plane
decodes to an NVFP4 tensor and executes W4A4, so its cost is measured against an
A4 input; a rung on E4M3 over a CHANNEL plane executes W8A8 and is measured
against a per-token FP8 input.  The same *weight* rate therefore costs
differently on the two routes, which is the whole reason the allocator may not
rank them on weight error alone.

The W4A4 leg is the **served static contract**, not the registry's dynamic
RTN.  ``tessera.serving.nvfp4_route`` reads the artifact's
``trellis_input_global_scale`` and calls vLLM's compiled ``scaled_fp4_quant``
-- a static-global-scale operation whose per-16 block scales are stored as
UE4M3 -- so an E2M1 rung here is scored through
``format_registry.nvfp4_activation_qdq_served`` at the unit's own calibrated
``input_global_scale`` (fused-sibling unified, the same joint the exporter's
scale file carries).  NVFP4's registry callback is a dynamic FP32-scale RTN
that never snaps through UE4M3 and rounds midpoints differently; pricing with
it prices an activation tensor the runtime does not execute
(RobTand/prismaquant#194).  A missing scale refuses rather than falling back
to the dynamic quantiser.

Rendering identity, and the wire
--------------------------------
The render is not ``render_tessera_weight``'s reconstruction; it is
``read_unit_artifact(encode_linear(...).blob)`` (or the equivalent batch
entry) -- **the bytes, decoded**.  So the
cache entry holds the wire beside the dequantised render. Checkpoint resume
verifies those bytes against their producer input receipt. That proves the
cached wire is the priced wire. The packed producer-plan/cached-wire bridge
carries those receipts through allocation and export; actual export/serve
qualification remains the measurement tracked by PrismaQuant #183.

What this stage does NOT do
---------------------------
It measures ``output_mse`` under the route's activation contract, which is the
render-cost currency (``COST_MODE=production-render-score``'s
``--score-field output_mse``). It is not an AURA price. The separate streamed
``--joint-activation`` path projects the complete signed weight/activation
residual through KL cotangents before squaring. This campaign does not perform
that projection; its payload declares its own currency so downstream code
cannot mistake output MSE for measured joint AURA.
"""
from __future__ import annotations

from .tessera_campaign_requests import (
    CaptureJoinRequest, CapturePrepRequest, capture_chain_bookend_request,
    CensusRequest, CaptureRequest, CaptureQuantum, census_request, streamed_calibration_request,
    CampaignScopeRequest, campaign_scope_request,
    AnchorPlacement, anchor_placement, LoadedCensusRequest, LoadedPricingRequest,
    ResidentCampaignWeights, SelectedCampaignWeights, loaded_campaign_request,
)

from prismaquant import tessera_campaign_wire as campaign_wire

from prismaquant import tessera_calibration_forward as calibration_forward
from prismaquant.tessera_campaign_table import (
    CENSUS_SCHEMA,
    CURRENCY,
    CampaignAnchor,
    ExpertPopulation,
    SCHEMA,
    StackSampleError,
    campaign_population_block,
    campaign_cost_payload,
    parse_family_restriction,
    select_anchor_groups,
    selection_stack_samples,
    parse_rate_band,
    campaign_acquisition_row_scope,
    requested_acquisition_schedule,
    campaign_acquisition_origin,
    require_campaign_acquisition_anchor,
    expert_partition_members,
)


from contextlib import contextmanager

import re
import functools
import json
import math
import os
import pickle
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from .digests import (
    DIRECT_ASCII_INDENT2_LAX,
    DIRECT_ASCII_SPACED_LAX,
    DIRECT_ASCII_STRICT,
    bytes_sha256hex,
    indent2_json_file_bytes,
    text_sha256hex,
)

if TYPE_CHECKING:
    from .lane_eligibility import ServingContext

from . import tessera_hessian as th
from .nvfp4_activation_contract import (
    ActivationScaleContractError as _OwnedActivationScaleContractError,
    ServedQuantizerUnboundError,
)
from .tessera_expert_projection import EXPERT_WIRES_KEY, POPULATION_KEY, PROJECTION_KEY
from .tessera_publication import PublicationJob
from .schemas import strict_json_loads

__all__ = [
    'EXIT_EMPTY_MENU',
    'PINNED_ROSTER_SCHEMA',
    'CampaignRoster',
    'anchor_group_key',
    'anchor_schedule',
    'calibration_census',
    'campaign_roster',
    'campaign_checkpoint_identity',
    'census_max_abs',
    'census_token_counts',
    'contract_source_label',
    'load_calibration_census',
    'main',
    'next_anchor_rate',
    'pinned_roster_block',
    'report_empty_menus',
    'require_census_draw',
    'resolve_anchor_groups',
    'round_one_rates',
    'write_export_inputs',
]


#: The stream head's journal before finalize (PQ #1403), beside the checkpoint
#: and never inside ``<checkpoint>.parts``: a present checkpoint means a run
#: identity was already bound, and the stream head resumes that journal instead,
#: taking each unit's receipts from its manifest (PQ #1613).
STREAM_JOURNAL_SUFFIX = ".stream"
STREAM_JOURNAL_STAGE = "Tessera campaign stream head"
#: What the stream journal's identity holds in place of each unit's W, X and H
#: receipts. The unit's own shard carries them, and adoption compares them
#: with the relaunch's read of the same entry.
STREAM_JOURNAL_RECEIPT = "bound per unit in its stream journal shard"
#: What a resumed checkpoint's expected identity holds for a receipt its
#: manifest does not record. It equals no receipt, so ``prepare_journal``
#: refuses that manifest by the field's name (PQ #1613).
CHECKPOINT_RECEIPT_ABSENT = "absent from the checkpoint manifest"

def anchor_schedule(lo: int, hi: int, count: int) -> list[int]:
    """``count`` rungs spanning ``[lo, hi]``, endpoints included.

    Evenly spaced in rate because the surface interpolates in rate; the
    *adaptive* rounds are what put anchors where the curve actually bends, and
    they are driven by measured leave-one-out error rather than by a prior
    about where a rate-distortion curve is steep.
    """
    if hi <= lo:
        return [lo]
    count = max(2, int(count))
    if count == 2:
        return [lo, hi]
    step = (hi - lo) / (count - 1)
    out = sorted({int(round(lo + i * step)) for i in range(count)})
    out[0], out[-1] = lo, hi
    return sorted(set(out))


def _unit_family_names(name, restriction, structure_by_unit):
    """The declared structural family scope, absent when unrestricted."""
    return None if restriction is None else tuple(restriction[structure_by_unit[name]])


def require_seed_family_scope(name, state, *, family_restriction, structure_by_unit,
                              rate_band=None, profile=None):
    """Refuse incompatible active seed anchors before their wires are linked."""
    from .tessera_formats import parse_tessera_format_name
    from . import format_registry as fr
    policy = parse_family_restriction(family_restriction)
    structure = (structure_by_unit or {}).get(name)
    if policy is not None and structure not in ("dense", "routed_moe"):
        raise RuntimeError(f"{name}: family restriction requires authoritative structure")
    families = _unit_family_names(name, policy, structure_by_unit)
    unquantized = profile is not None and profile.linear_requires_unquantized_activations(name)
    if families is None and not unquantized:
        return
    if not isinstance(state, Mapping) or not isinstance(state.get("anchors"), list):
        raise RuntimeError(f"{name}: family restriction received an invalid seed state")
    for row in state["anchors"]:
        try:
            family, rate = parse_tessera_format_name(row["format_name"])
        except (TypeError, ValueError, KeyError) as exc:
            raise RuntimeError(f"{name}: family restriction received an invalid seed format") from exc
        if (row.get("qname") != name or row.get("family") != family.name
                or type(row.get("body_rate_q256")) is not int or row["body_rate_q256"] != rate):
            raise RuntimeError(f"{name}: family restriction seed format/identity disagree")
        if families is not None and family.name not in families:
            raise RuntimeError(f"{name}: seed {row['format_name']} violates unit family restriction")
        if unquantized and fr.get_format(row["format_name"]).act_quant_changes_input:
            raise RuntimeError(f"{name}: seed {row['format_name']} violates unit activation precision")
        if rate_band is not None and not rate_band[0] <= rate <= rate_band[1]:
            raise RuntimeError(f"{name}: seed {row['format_name']} is outside restricted rate band {rate_band}")


def round_one_rates(allowed: "Sequence[int]", *, band, anchors: int,
                    snap, exhaustive_band: bool = False) -> list[int]:
    """Where round one puts this family's anchors on this group's grid.

    Without a band the schedule spans the family's whole realisable range,
    which is what every artifact before 2026-09-06 was priced under.  With
    one, the anchors go at the ends of the **band**, and the reason is
    measured rather than economised: the surface is not one line over the
    whole 1--8 bit range.  On campaign-01 a two-anchor fit that spans the full
    range and drops the interior anchor of a 1 / 4.5 / 8-bit triple misses the
    interior by median 0.32 and p90 0.76 log2, with the same sign at both
    ends -- curvature, not noise -- while the same construction inside a
    one-bit bracket lands at median 0.08.  So the fix is not more anchors over
    a range nothing will be allocated in; it is the same two anchors placed
    around the rates the artifact will actually use.

    A family whose realisable rungs do not reach the band gets no anchor here
    and is left to say so, rather than being priced at a rate outside it.  An
    explicit research acquisition may instead request every legal rung in a
    declared band.  That mode remains bounded by the band's realisable grid;
    it does not widen a family or change serving admission.
    """
    if not allowed:
        return []
    if band is None:
        if exhaustive_band:
            raise RuntimeError("an exhaustive rate grid requires --rate-band")
        return sorted({r for r in (snap(rate, allowed) for rate in
                                   anchor_schedule(allowed[0], allowed[-1], anchors))
                       if r is not None})
    lo, hi = band
    inside = [rate for rate in allowed if lo <= int(rate) <= hi]
    if not inside:
        return []
    if exhaustive_band:
        return sorted(set(inside))
    return sorted({snap(inside[0], allowed), snap(inside[-1], allowed)} - {None})


def require_exhaustive_rate_research_mode(enabled: bool, mode: str) -> None:
    """Keep complete-grid acquisition in its declared research menu."""
    if enabled and mode != "research":
        raise RuntimeError("--exhaustive-rate-grid requires --menu-mode research")


def audit_extra_rate(allowed: "Sequence[int]", placed: "Sequence[int]",
                     *, snap) -> "int | None":
    """The one interior rung an audit unit measures on top of ``placed``.

    Two anchors define a line and can never disagree with it, so a bracket
    priced at its ends carries no evidence about its own interpolation.  The
    audit subsample buys that evidence for the price of one extra encode per
    ten sampled units: a third anchor in the middle of the bracket, which the
    existing leave-one-anchor-out report then scores against the line the
    other two draw.  Returns None when the bracket has no interior rung to
    put it on.
    """
    if len(placed) < 2:
        return None
    lo, hi = int(min(placed)), int(max(placed))
    interior = [int(r) for r in allowed if lo < int(r) < hi]
    if not interior:
        return None
    return snap((lo + hi) // 2, interior)


def next_anchor_rate(
    rates: Sequence[int], loo: Mapping[str, object],
) -> "int | None":
    """The rung the next round should measure, or None if there is no room.

    The interval to split is the one whose *interior* anchor the rest of the
    surface predicts worst -- leave-one-anchor-out's own per-anchor error, so
    the campaign spends its next encode where the interpolation is measurably
    failing rather than where a heuristic guesses it might.  With fewer than
    three anchors LOO has no interior to report, and the widest gap is the only
    information available; that fallback is stated rather than silent.
    """
    ordered = sorted(int(r) for r in rates)
    if len(ordered) < 2:
        return None

    def midpoint(left: int, right: int) -> "int | None":
        if right - left < 2:
            return None
        mid = (left + right) // 2
        return mid if mid not in ordered else None

    per_anchor = loo.get("per_anchor") if isinstance(loo, Mapping) else None
    if isinstance(per_anchor, Sequence) and per_anchor:
        worst = max(per_anchor, key=lambda e: abs(float(e["log2_error"])))
        rate = int(worst["q256"])
        index = ordered.index(rate)
        left = midpoint(ordered[index - 1], rate)
        right = midpoint(rate, ordered[index + 1])
        for candidate in (left, right):
            if candidate is not None:
                return candidate
    gaps = sorted(
        ((right - left, left, right)
         for left, right in zip(ordered, ordered[1:])),
        reverse=True,
    )
    for _width, left, right in gaps:
        candidate = midpoint(left, right)
        if candidate is not None:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

class ActivationScaleContractError(_OwnedActivationScaleContractError):
    """A static-activation-contract rung has no calibrated input scale.

    Its own class for the same reason ``HessianContractError`` has one: the
    anchor loop absorbs per-anchor failures with ``except Exception:
    continue``, and a scale-contract refusal is about every W4A4 row this run
    would write, not about one anchor.  Falling back to the registry's dynamic
    FP32-scale quantiser instead would price an activation regime the serve
    does not execute -- exactly the defect this refusal exists to make loud
    (RobTand/prismaquant#194).

    The same refusal, raised by the assignment-KL hooks and the production
    cache scorer, is the owner's
    ``nvfp4_activation_contract.ActivationScaleContractError``; this is that
    class under the campaign's historical name (#205), so a caller catching
    either sees one contract error.
    """


def _encode_and_render(weight, format_name: str, *, activation_kwargs=None,
                       hessian_required: bool = True, recipe=None):
    """``(render, blob)`` for one rung: the bytes, and what they decode to.

    A thin adapter over ``tessera_render.encode_tessera_unit``, which is the
    shared scalar/batch adapter that calls Tessera's byte path.  Everything
    that made this function worth having -- the render is
    ``read_unit_artifact(blob)``, i.e. the bytes decoded rather than a second
    reconstruction (principle 8), and ``verify=False`` because the tensor
    ``verify`` would compare against is one nothing downstream sees -- now
    lives there, so a caller cannot reach the encoder around it and skip the
    Hessian contract.
    """
    from .tessera_render import encode_tessera_unit

    return encode_tessera_unit(
        weight, format_name,
        activation_kwargs=activation_kwargs,
        hessian_required=bool(hessian_required), verify=False,
        recipe=recipe,
    )


def _activation_kwargs_memo(source, weights, device, *, max_entries=None,
                            resource_check=None, factor_scratch_bytes=0):
    """The campaign's existing plane-keyed memo, with an explicit owner bound."""
    if max_entries is not None and (type(max_entries) is not int or max_entries < 1):
        raise ValueError('encoder memo capacity must be positive or unbounded')

    @functools.lru_cache(maxsize=max_entries)
    def for_unit(name, scale_plane):
        if resource_check is not None:
            resource_check('before_selected_encoder_factors:'+name,
                           reserve_bytes=factor_scratch_bytes)
        kwargs = th.encoder_kwargs(source, name, int(weights[name].shape[1]),
                                   device, scale_plane=scale_plane)
        if resource_check is not None:
            resource_check('after_selected_encoder_factors:'+name)
        return kwargs

    return for_unit


def _measure_anchor(
    *, qname: str, weight, activations, format_name: str, cache, wire_dir: Path,
    activation_kwargs_for=None, hessian_required: bool = True,
    static_input_scale: "float | None" = None, publisher=None,
    structure: "str | None" = None, profile=None,
):
    """Render one rung, price it as served, and store the wire beside it.

    ``structure`` is the unit's serving structure (``dense``/``routed_moe``)
    when the run declares one; the wire is then the one that structure is
    served on (``tessera_served_wire_recipe``, #1502), not the research table.

    ``static_input_scale`` is the unit's calibrated NVFP4
    ``input_global_scale`` (fused-sibling unified), required whenever the
    rung's route executes the static UE4M3 activation contract and ignored on
    every other route.  The refusal for a missing one runs BEFORE the encode,
    because the encode is the expensive half and the refusal is about the
    whole run.

    ``activation_kwargs_for`` is a callable ``(qname, scale_plane) -> encoder
    kwargs`` -- the block-LDL of that unit's regularised ``XᵀX`` and the
    plane's refit metric -- which the caller memoises per ``(unit, plane)``.
    Per unit alone would be wrong and silently so: the refit objective is keyed
    by scale plane, so a memo that ignored the plane would price a unit's
    second family under the first family's objective.  Per rung would be
    wasteful: no rate reaches that call, and a twelve-anchor surface would
    otherwise factorise the same Hessian twelve times.  The plane is constant
    across every rung of a family and differs between families, so the bound is
    one factorisation per unit per family-plane while retained. Selected
    source campaigns bound the memo to the compatible anchor batch width.

    A **missing key is a hard failure**, never a silently H-free encode: this
    codebase has already been bitten once by a render whose activation lookup
    missed and quietly fell back to RTN, raising nothing, and an H-free encode
    of a rung whose shipping bytes are H-aware is exactly that bug with
    different bytes.
    """
    prepared = _prepare_anchor(
        qname=qname, format_name=format_name,
        activation_kwargs_for=activation_kwargs_for,
        hessian_required=hessian_required, static_input_scale=static_input_scale,
        structure=structure, profile=profile)
    started = time.time()
    render, blob = _encode_and_render(
        weight, format_name, recipe=prepared["wire"],
        activation_kwargs=prepared["activation_kwargs"],
        hessian_required=prepared["hessian_required"])
    return _finish_anchor(qname=qname, weight=weight, activations=activations,
        format_name=format_name, cache=cache, wire_dir=wire_dir,
        prepared=prepared, render=render, blob=blob, elapsed=time.time() - started,
        publisher=publisher, profile=profile)


#: The context the campaign's pricing seam names when it binds the served
#: activation quantiser (RobTand/prismaquant#1481).
SERVED_QUANTIZER_CONTEXT = "Stage A Tessera campaign activation pricing"

#: The identity record this process bound through the owner, or ``None`` when
#: the campaign bound nothing itself. Stamped into the cost table's provenance.
_SERVED_QUANTIZER_RECORD = None


def _bind_served_quantizer(qname, format_name):
    """Bind the served activation arithmetic before this rung is scored.

    A rung whose route executes the static NVFP4 activation contract is priced
    by vLLM's registered ``scaled_fp4_quant``, and the contract refuses to price
    it unbound (RobTand/prismaquant#567). Until #1481 nothing on the campaign
    path bound it, so every A4 rung raised ``ServedQuantizerUnboundError``.

    The rule has one owner, ``joint_cost_quantum.bind_joint_served_quantizer``,
    which Stage B already calls; this seam calls it too. It requires the
    registered operator for a served contract, never falls back to the Torch
    model, and binds nothing for A8/A16 rungs, so those rosters never import the
    serving extension.

    A process that already holds a binding keeps it. Either an earlier rung of
    this run bound it, or a CPU screen declared the model on purpose
    (``tests/priced_model_screen.py``). The owner would refuse a second,
    different binding anyway, so re-asking could only turn a declared choice
    into a refusal.
    """
    global _SERVED_QUANTIZER_RECORD
    from .nvfp4_activation_contract import active_served_quantizer_identity
    from .joint_cost_quantum import bind_joint_served_quantizer

    if active_served_quantizer_identity() is not None:
        return
    record = bind_joint_served_quantizer(
        {qname: (format_name,)}, context=SERVED_QUANTIZER_CONTEXT)
    if record is not None:
        _SERVED_QUANTIZER_RECORD = dict(record)
        print(f"[campaign] served quantizer bound for {format_name}: backend "
              f"{record.get('backend')} op {record.get('op')} image "
              f"{record.get('image_content_sha256')} dequant "
              f"{record.get('dequant_kernel')}", flush=True)


def _is_glm_direct_consumer_profile(profile) -> bool:
    """Use the declared architecture before the undeclared profile identity."""
    if profile is None:
        return False
    architectures = profile.declared_architectures()
    if architectures:
        return architectures == ("Glm5NextForConditionalGeneration",)
    return profile.name == "glm5_next"


def _direct_consumer_activation_contract(qname, *, profile):
    source = qname.removesuffix(".weight")
    if not source.endswith((".self_attn.kv_b_proj", ".self_attn.indexer.weights_proj")):
        return None
    if profile is None:
        raise ActivationScaleContractError(
            f"{qname}: a direct-consumer leaf needs the run profile to bind "
            "its activation contract; profile=None drops the GLM a16/a32 "
            "contract silently, so it refuses instead.")
    if not _is_glm_direct_consumer_profile(profile):
        return None
    from tessera.serving.dense_ownership import fused_module, role_name
    from tessera.serving.projection_routes import direct_consumer_activation_contract
    tensor = source + ".weight"
    fused = fused_module(tensor, "Glm5NextForConditionalGeneration")
    return direct_consumer_activation_contract(source if fused is None else fused[0], role_name(tensor))


def _prepare_anchor(*, qname, format_name, activation_kwargs_for,
                    hessian_required, static_input_scale, structure=None, profile=None):
    """Admit each unit's Hessian and served activation contract before encode."""
    direct_contract = _direct_consumer_activation_contract(qname, profile=profile)
    if direct_contract is None:
        _bind_served_quantizer(qname, format_name)
    from . import format_registry as fr
    from .tessera_formats import (
        parse_tessera_format_name, tessera_served_wire_recipe, tessera_serving_route,
    )
    from .tessera_render import HessianContractError

    from .tessera_render import rung_accepts_hessian

    spec = fr.get_format(format_name)
    family, rung = parse_tessera_format_name(format_name)
    # ONE resolve for this anchor. The plane the kwargs are built on, the plane
    # the predicate reads and the plane the encode writes are this object --
    # not three lookups that agree only while nothing clears the recipe memo.
    # It is the wire this unit's STRUCTURE is served on (#1502): a routed
    # E2M1x2 stack below the cap ships span-2 TCQ, not the research WINDOW
    # recipe, and a structure the pinned contract attests no wire for refuses
    # here rather than pricing bytes the serve cannot read.
    wire = tessera_served_wire_recipe(family, rung, structure=structure)
    # The A side, as served.  A route with a STATIC activation contract
    # executes vLLM's static-global-scale ``scaled_fp4_quant`` (UE4M3 block
    # scales) against the artifact's ``trellis_input_global_scale``; every
    # other route keeps the serving format's own dynamic quantiser, taken by
    # reference from the spec.
    #
    # WHICH of the two is the SPEC's answer, never a compare of the route's
    # source-format name against ``"NVFP4"`` (#205's rule, #221's fix): the
    # spec resolved above already carries the contract, because
    # ``synthesize_tessera_spec`` derived it from the registry row the route
    # names.  A Tessera rung routed through the same kernel has the same
    # contract and a different name, and the day a second registry row gets a
    # contract, a name compare here would price it under a quantiser the
    # runtime never runs while the cache scorer and the KL hooks -- which
    # already read the row -- refuse the same unit.
    route = tessera_serving_route(family, wire, rung)
    contract = spec.static_activation_contract
    if contract is not None:
        if static_input_scale is None or not float(static_input_scale) > 0:
            raise ActivationScaleContractError(
                f"{qname} {format_name}: this rung's route executes the "
                f"static activation contract {contract.execution} "
                f"({route.contract}) and no calibrated input_global_scale was "
                "supplied. Scoring it with the registry's dynamic FP32-scale "
                "quantiser would price an activation regime the serve does "
                "not execute; a lookup that misses must refuse, not fall back."
            )
        input_scale = float(static_input_scale)
        # The contract's own oracle, not a second spelling of it.
        activation_qdq = (
            lambda x: contract.quantize_dequantize(x, input_scale))
    else:
        input_scale = None
        activation_qdq = spec.activation_quantize_dequantize
    if direct_contract is not None:
        activation_qdq = fr.get_format("BF16").activation_quantize_dequantize
    activation_kwargs = None
    # Whether an H can be applied is a property of the RUNG'S WIRE, not of the
    # run, and it is DERIVED from what the pinned ActivationSource emits for
    # that wire's plane rather than restated here (``rung_accepts_hessian``).
    # It reads True on every plane PrismaQuant resolves at this pin.
    hessian_required = bool(hessian_required) and rung_accepts_hessian(
        format_name, wire)
    if hessian_required:
        if activation_kwargs_for is None:
            raise HessianContractError(
                f"{qname}: hessian_required=True but no activation-kwargs "
                "source was passed to _measure_anchor")
        activation_kwargs = activation_kwargs_for(qname, wire.scale_plane)
        if not activation_kwargs:
            raise HessianContractError(
                f"{qname}: no Hessian for this Linear. A lookup that misses "
                "must not fall through to a weights-only encode.")
    return dict(spec=spec, family=family, rung=rung, wire=wire, structure=structure,
                activation_qdq=activation_qdq, input_scale=input_scale,
                activation_contract=direct_contract or str(spec.act_dtype_name or "a16"),
                activation_kwargs=activation_kwargs,
                hessian_required=hessian_required)


def _direct_consumer_memory_bytes(qname, family, shape, *, profile):
    """Charge each direct cache once through its source member."""
    source = qname.removesuffix(".weight")
    if not source.endswith((".self_attn.kv_b_proj", ".self_attn.indexer.weights_proj")):
        return 0
    if not _is_glm_direct_consumer_profile(profile):
        return 0
    from tessera.serving.dense_ownership import fused_module, role_name
    from tessera.serving.projection_routes import direct_consumer_resident_bytes
    from tessera.serving.scheme import ROUTES
    tensor = source + ".weight"
    fused = fused_module(tensor, "Glm5NextForConditionalGeneration")
    owner = source if fused is None else fused[0]
    grid = family.base + (f"x{family.arity}" if family.arity > 1 else "")
    route_family = next(label for label, route in ROUTES.items() if grid in route["grids"])
    rows, columns = map(int, shape)
    extra = direct_consumer_resident_bytes(owner, route_family, rows, columns,
                                           [(role_name(tensor), rows)])
    return extra["resident_bytes_resident_mode"]


def _direct_consumer_render(blob, qname, family, device):
    """Use the runtime owner's direct decoder for the priced source unit."""
    from tessera.serving.dense_ownership import fused_module, role_name
    from tessera.serving.projection_routes import direct_consumer_weight
    from tessera.serving.scheme import ROUTES
    tensor = qname.removesuffix(".weight") + ".weight"
    fused = fused_module(tensor, "Glm5NextForConditionalGeneration")
    owner = tensor.removesuffix(".weight") if fused is None else fused[0]
    grid = family.base + (f"x{family.arity}" if family.arity > 1 else "")
    route_family = next(label for label, route in ROUTES.items() if grid in route["grids"])
    return direct_consumer_weight(blob, owner, role_name(tensor), route_family, device=str(device))


def _finish_anchor(*, qname, weight, activations, format_name, cache, wire_dir,
                   prepared, render, blob, elapsed, encoding_batch_size=1,
                   publisher=None, profile=None):
    """Score decoded bytes and publish the existing cache/wire entries.

    ``publisher``, when given, is a
    :class:`~prismaquant.tessera_publication.BoundedPublisher` that performs
    the two writes on its own thread instead of here.  The bytes handed to it
    are the same bytes and the writes are the same calls; what moves is only
    which thread runs them, and the caller then owes the completion an
    ordering rule -- the anchor is not journalled until its files exist.  With
    no publisher this function is what it has always been, line for line.
    """
    import torch
    from .production_weight_cache import (
        _canonical_rendered_weight_tensor, _local_forward_render_score,
        _store_rendered_weight_entry)

    spec, family, rung = (prepared[key] for key in ("spec", "family", "rung"))
    activation_qdq, input_scale = (prepared[key] for key in (
        "activation_qdq", "input_scale"))
    direct_contract = _direct_consumer_activation_contract(qname, profile=profile)
    preserve_fp32 = direct_contract == "a32"
    cache_dtype = torch.float32 if preserve_fp32 else torch.bfloat16
    if direct_contract is not None:
        render = _direct_consumer_render(blob, qname, family, weight.device)
    publication_bytes = render.numel() * (4 if preserve_fp32 else 2) + len(blob)
    score, metric, quantized, _clipped = _local_forward_render_score(
        reference_weight=weight,
        rendered_weight=render,
        activations=activations,
        activation_quantize=activation_qdq,
        # No pre-clip on top: the static route's clamp lives inside the served
        # oracle (``nvfp4_activation_qdq_served`` clamps at the stored UE4M3
        # scale), and the dynamic routes have no calibrated clip at all.
        activation_max_abs=None,
    )
    if metric != "output_mse":
        raise RuntimeError(f"unexpected render-score metric {metric!r}")
    hessian_applied = bool(prepared["activation_kwargs"])

    # Room first, then the bytes.  The copy below is a new host allocation the
    # writer will own, so the budget has to admit it BEFORE it exists: charging
    # it after the fact would leave this thread holding one artifact more than
    # the bound allows, every time the writer is behind.  The size is known
    # without a copy: the declared cache dtype sets each element's bytes.
    if publisher is not None:
        publisher.reserve(publication_bytes)
    try:
        # The device-to-host copy stays on the thread that owns the device
        # work, whichever way the bytes are written.
        # ``_store_rendered_weight_entry`` canonicalises again below and that
        # second call is an identity on a CPU tensor already in the target
        # dtype and contiguous, so the synchronous path stores the same object
        # it always stored.  Staging it here is what gives the writer thread
        # bytes nobody else owns.
        staged = _canonical_rendered_weight_tensor(
            render, weight_dtype=cache_dtype, preserve_fp32=preserve_fp32)
    except BaseException:
        if publisher is not None:
            publisher.release(publication_bytes)
        raise
    # The wire, beside the render.  A ``.tessera`` shard per (qname, rung),
    # named the way the cache names its weight shards, so the export leg can
    # find the exact bytes this row was priced on instead of re-encoding.
    wire_path = campaign_wire.wire_path(wire_dir, qname, format_name)

    def _publish():
        # Check on the writer thread as well as at batch admission, so queued
        # publications see every earlier durable coordinate before any write.
        campaign_wire.require_injective_anchor_filenames(
            cache, wire_dir, ((qname, format_name),))
        _store_rendered_weight_entry(
            weights=cache.weights,
            qname=qname,
            fmt=format_name,
            tensor=staged,
            cache_dir_path=Path(cache.cache_dir) if cache.cache_dir else None,
            weight_dtype=cache_dtype, preserve_fp32=preserve_fp32,
        )
        tmp = wire_path.with_suffix(".tessera.tmp")
        tmp.write_bytes(blob)
        os.replace(tmp, wire_path)
        campaign_wire.wire_coordinates(cache, wire_dir).add((qname, format_name))
        if getattr(cache, 'metadata', {}).get('release_completed_anchor_file_pages'):
            # The existing PWC entry is already disk-backed. Completed anchor
            # files must not accumulate an unbounded page-cache owner across
            # rungs. The release helper checks the entry's identity, fsyncs
            # and advises it from its own descriptor. No verification consumes
            # a second hash of these completed files here.
            from .perturbed_x_cache import release_activation_cache_file_pages
            rendered_path = Path(cache.cache_dir)/cache.weights[(qname, format_name)]
            for path in (rendered_path, wire_path):
                release_activation_cache_file_pages(path, expected_stat=path.stat())

    if publisher is None:
        _publish()
    else:
        # The reservation above was taken for exactly these bytes; the
        # The selected direct head cache keeps FP32; other caches keep BF16.
        publisher.submit(PublicationJob(
            key=(FILES_JOB, qname, format_name),
            charged_bytes=publication_bytes,
            publish=_publish,
        ))

    bits = spec.bits_for_shape(tuple(weight.shape))
    return CampaignAnchor(
        qname=qname,
        family=family.name,
        format_name=format_name,
        body_rate_q256=int(rung),
        dloss=float(score),
        # One calibration draw: the stderr a single measurement supports is
        # zero, and inventing one would put a fabricated uncertainty into the
        # allocator's UCB hedge.  Reported as zero and named as such.
        dloss_stderr=0.0,
        memory_bytes=(int(spec.memory_bytes_for_shape(tuple(weight.shape)))
                      + _direct_consumer_memory_bytes(qname, family, weight.shape, profile=profile)),
        bits_per_param=float(bits) / max(1, int(weight.numel())),
        activation_contract=prepared.get("activation_contract", str(spec.act_dtype_name or "a16")),
        activation_quantized=bool(quantized),
        wire_bytes=len(blob),
        seconds=elapsed,
        hessian_applied=hessian_applied,
        input_global_scale=input_scale,
        encoding_batch_size=encoding_batch_size,
    )


def _measure_anchor_batch(*, qnames, weights, activations, format_name,
                          cache, wire_dir, activation_kwargs_for=None,
                          hessian_required=True, static_input_scales=None,
                          publisher=None, structures=None, profile=None):
    """One producer batch, with the scalar path's per-unit gates and storage.

    ``structures`` maps a unit to its serving structure, as ``structure`` does
    for :func:`_measure_anchor`; a unit it does not name declares none.
    """
    from .tessera_render import encode_tessera_units

    if not qnames or len(set(qnames)) != len(qnames):
        raise ValueError("anchor batch needs distinct nonempty unit names")
    if len(weights) != len(qnames) or len(activations) != len(qnames):
        raise ValueError("anchor batch needs one weight and activation input per unit")
    prepared = [_prepare_anchor(
        qname=name, format_name=format_name,
        activation_kwargs_for=activation_kwargs_for,
        hessian_required=hessian_required,
        static_input_scale=(static_input_scales or {}).get(name),
        structure=(structures or {}).get(name), profile=profile) for name in qnames]
    # The producer call takes ONE recipe and ONE Hessian requirement for the
    # whole batch, from its first entry.  Both derive from (family, rung,
    # structure) only, so a batch built by ``_anchor_batches`` agrees by
    # construction; a batch
    # that does not would encode some unit under another unit's recipe, and is
    # refused rather than priced.
    for name, entry in zip(qnames[1:], prepared[1:]):
        if (entry["wire"] != prepared[0]["wire"]
                or entry["hessian_required"] != prepared[0]["hessian_required"]):
            raise ValueError(
                f"anchor batch {format_name}: {name} prepares a different wire "
                f"recipe or Hessian requirement than {qnames[0]}")
    campaign_wire.require_injective_anchor_filenames(
        cache, wire_dir, ((name, format_name) for name in qnames))
    started = time.time()
    encoded = encode_tessera_units(
        weights, format_name, recipe=prepared[0]["wire"],
        activation_kwargs=[entry["activation_kwargs"] for entry in prepared],
        hessian_required=prepared[0]["hessian_required"])
    # Batch wall time is apportioned, not attributed as a measured per-unit
    # duration. Summing anchor.seconds still reports the actual encoding time.
    elapsed = (time.time() - started) / len(qnames)
    return [_finish_anchor(qname=name, weight=weight, activations=acts,
        format_name=format_name, cache=cache, wire_dir=wire_dir, prepared=entry,
        render=render, blob=blob, elapsed=elapsed,
        encoding_batch_size=len(qnames), publisher=publisher, profile=profile)
        for name, weight, acts, entry, (render, blob) in zip(
            qnames, weights, activations, prepared, encoded)]


def _anchor_batches(pending, *, weights, batch_size, structures=None):
    """Bound compatible encodes inside this action; never assign hosts.

    Membership is by ``(family, rung, structure, shape, dtype, device)`` for
    every unit, dense or projected expert.  A unit's wire recipe and Hessian
    requirement are functions of ``(family, rung, structure)`` alone
    (``_prepare_anchor``; structure since #1502, because a routed E2M1x2 stack
    below the cap is served on another wire than a dense unit), and the
    joined producer call is column-independent, so a dense unit joins
    same-shape units of other groups exactly as an expert joins its stack.
    Dense units used to be keyed by ``(unit, family, rung)``, which kept every
    dense encode at batch one: a long serial chain of narrow Viterbi calls at
    about 25 W of the 140 W envelope (RobTand/prismaquant#1479).  The ORDER of
    the batches is unit-major: the batches of one compatible key differ only
    by rung, and the chunk at one position holds the same members at every
    rung when the round pended every member at every rate (round one does,
    per member in rate order), so consecutive batches encode the same units
    at successive rungs.  The encoder memo is sized to the batch width
    (``encoder_memo_capacity``), so that order is what lets a unit's block-LDL
    factorization be reused across its rungs instead of refactorized once per
    (unit, rung) with 864 batches of other units in between
    (RobTand/prismaquant#389).  Rung-major emission -- every batch of rung A,
    then every batch of rung B -- is what the insertion-ordered grouping
    produced before, and with the memo at the batch width it hit nothing.
    """
    if batch_size < 1:
        raise ValueError("anchor batch size must be positive")
    if batch_size == 1:
        return [[item] for item in pending]
    groups = {}
    for item in pending:
        name, family, rung = item
        weight = weights[name]
        # A ``meta`` placeholder is a host weight the row stream has not read
        # yet (PQ #1654); it batches exactly as the host copy it stands for.
        base = (family, (structures or {}).get(name), tuple(weight.shape),
                weight.dtype, _host_device() if weight.is_meta else weight.device)
        key = (family, rung, *base[1:])
        groups.setdefault(key, (base, []))[1].append(item)
    # Sort key: the base (the key minus its rung) in first-appearance order,
    # then the chunk position, then the rung in first-appearance order -- so
    # chunk 0 of every rung of one base precedes chunk 1 of any of them.
    base_rank, key_rank, chunks = {}, {}, []
    for key, (base, group) in groups.items():
        base_rank.setdefault(base, len(base_rank))
        key_rank[key] = len(key_rank)
        for position, start in enumerate(range(0, len(group), batch_size)):
            chunks.append(((base_rank[base], position, key_rank[key]),
                           group[start:start + batch_size]))
    chunks.sort(key=lambda chunk: chunk[0])
    return [batch for _rank, batch in chunks]


def _stale_served_wire(anchor, record, *, structure) -> "str | None":
    """Why a resumed row's receipt names a wire ``structure`` is not served on.

    ``None`` when the receipt's stamped recipe is the served recipe for the
    row's rung, or when the receipt carries no readable recipe (the full
    identity check that follows refuses those).  Only a receipt whose recipe
    is readable and differs is stale; every other disagreement stays a
    refusal of the whole resume.
    """
    if structure is None:
        return None
    from .tessera_formats import parse_tessera_format_name, tessera_served_wire_recipe

    stamped = ((record or {}).get("identity") or {}).get("recipe")
    parsed = parse_tessera_format_name(anchor.format_name)
    if not isinstance(stamped, dict) or parsed is None:
        return None
    family, rung = parsed
    served = tessera_served_wire_recipe(family, rung, structure=structure,
                                        refuse_unattested=False).to_config()
    observed = {key: value for key, value in stamped.items() if key not in ("grid", "q256")}
    if observed == served:
        return None
    return (f"{anchor.format_name}: receipt stamped {observed}, but a {structure} unit is "
            f"served on {served} at this rung (#1502); re-priced, not adopted")


def _load_campaign_acquisition(args, parser):
    """Authenticate opt-in work before any campaign preparation has side effects."""
    binding = (args.acquisition_request, args.acquisition_request_sha256)
    if all(value is None for value in binding):
        return None
    if not all(binding):
        parser.error("--acquisition-request and --acquisition-request-sha256 go together")
    if args.menu_mode != "research" or args.max_rounds != 1:
        parser.error("acquisition requires explicit --menu-mode research and --max-rounds 1")
    if (args.rate_band is not None or args.exhaustive_rate_grid or args.census_out
            or args.capture_calibration_out or args.capture_chain
            or args.research_exact_member is not None):
        parser.error("acquisition refuses rate-band, exhaustive, capture-only and partial-member modes")
    # Selection metadata is small and must refuse partial/audit work before
    # calibration intake, profile discovery, source pools or output creation.
    if args.units:
        selection = load_unit_selection(args.units)
        if any(any(field in entry for field in
                   ("partition", "sampled", "audit", "inclusion_probability", "stack_samples"))
               for entry in selection["groups"]):
            parser.error("acquisition refuses sampled, audit and expert partition selections")
    from .tessera_full_domain_acquisition import load_joint_campaign_acquisition
    return load_joint_campaign_acquisition(binding={"path": binding[0], "sha256": binding[1]})


def _campaign_round_one_schedule(groups, group_rates, *, placement: AnchorPlacement, audit_units, snap,
                                 requests=None, rates_by_unit=None):
    """One seam for early priming and actual round one; validate the whole plan.

    An acquisition takes the exact union of a group's requested q256s. Empty
    report families buy no work, and neither endpoints nor an audit add work.
    """
    names = {name for members in groups.values() for name in members}
    schedule = {name: {} for name in sorted(names)}
    if requests is not None:
        if set(requests) != names:
            raise ValueError("acquisition schedule requires every atomic member, no extra units")
        if (audit_units or placement.rate_band is not None
                or placement.exhaustive_rate_grid or placement.max_rounds != 1):
            raise ValueError("acquisition schedule refuses audit, band, exhaustive or adaptive work")
    requested_schedule = None if requests is None else requested_acquisition_schedule(groups, requests)
    for key, members in sorted(groups.items()):
        if requests is None:
            families = group_rates[key]
        else:
            families = sorted({family for name in members for family in requests[name]})
        for family in families:
            allowed = group_rates[key].get(family, [])
            if requests is None:
                want = round_one_rates(allowed, band=parse_rate_band(placement.rate_band),
                    anchors=placement.anchors, snap=snap, exhaustive_band=placement.exhaustive_rate_grid)
                extra = None if not audit_units else audit_extra_rate(allowed, want, snap=snap)
            else:
                proposed = [q for name in members for q in requests[name].get(family, [])]
                # An explicitly deferred report family requests no encode. Its
                # lack of an admitted grid is evidence to retain, not a reason
                # to invent work or refuse another family's actual work.
                if proposed and (rates_by_unit is None or any(
                        family not in rates_by_unit[name] for name in members)):
                    raise ValueError(f"acquisition family {family} has no actual atomic-member grid")
                if any(type(q) is not int or q not in allowed for q in proposed):
                    raise ValueError(f"acquisition {key}:{family} requests illegal or route-refused q256")
                want, extra = requested_schedule[members[0]][family], None
            for name in members:
                rates = set(want)
                if extra is not None and name in audit_units:
                    rates.add(extra)
                schedule[name][family] = sorted(rates)
    return schedule




def _require_campaign_acquisition_source(name, weight, expected, *, receipt=None):
    """Compare actual source bytes, never reinterpret a producer's other digest.

    The current producer hashes dtype/shape + NUL + bytes, unlike joint AURA's
    raw-byte digest. That receipt cannot project its content digest: use the
    existing joint helper once per unit only when a raw-byte receipt is absent.
    """
    if (expected["shape"] != list(weight.shape) or expected["dtype"] != str(weight.dtype)
            or expected["logical_bytes"] != weight.numel() * weight.element_size()):
        raise ValueError(f"acquisition source weight identity differs for {name}")
    fields = {"shape", "dtype", "logical_bytes", "content_sha256"}
    if isinstance(receipt, Mapping) and fields <= set(receipt):
        actual = {field: receipt[field] for field in fields}
    else:
        from .production_weight_cache import _cb_cache_tensor_identity
        actual = _cb_cache_tensor_identity(weight)
    if actual != expected:
        raise ValueError(f"acquisition source weight identity differs for {name}")


def _prime_first_anchor_batch(row_stream, *, placement: AnchorPlacement, seed_checkpoint,
                              batch_size, max_artifact_bpp, checkpoint, targets, menus,
                              weights, profile, expert_members, encode_structure,
                              projected_units, audit_units, route_cache,
                              partitioned=False, acquisition_requests=None, finalize_checkpoint=False):
    """Overlap a new head's exact first batch; never read ahead of resume gates.

    Use the existing group/anchor/batch mechanisms and row-local refusal memo.
    Any existing journal, even empty, keeps the before-read identity path. An
    inaccessible path fails closed instead of being treated as absent.
    """
    # A partition's legal grid depends on unpriced full-group siblings. Keep
    # its existing admit-time reads until this early-batch planner carries that
    # full-group contract; a subset-derived prime could disagree with admit.
    if partitioned or seed_checkpoint or finalize_checkpoint:
        return
    for path in (checkpoint, checkpoint.with_name(checkpoint.name + ".parts"),
                 checkpoint.with_name(checkpoint.name + STREAM_JOURNAL_SUFFIX)):
        try:
            path.stat()
        except FileNotFoundError:
            continue
        else:
            return

    def snap(rate, allowed):
        if not allowed:
            return None
        return min(allowed, key=lambda r: (abs(int(r) - int(rate)), int(r)))

    cap = float(max_artifact_bpp)
    rates_by_unit = {}
    for name in targets:
        per_family = {}
        for rung in menus[name]:
            if cap > 0 and rung.bpp > cap:
                continue
            per_family.setdefault(rung.family, set()).add(rung.body_rate_q256)
        rates_by_unit[name] = per_family
    groups = resolve_anchor_groups(targets, profile=profile, expert_members=expert_members)
    group_rates, _refused = anchor_group_rate_grids(groups, rates_by_unit,
        encode_structure=encode_structure, projected_units=projected_units,
        route_cache=route_cache)
    schedule = _campaign_round_one_schedule(groups, group_rates, placement=placement,
        audit_units=audit_units, snap=snap, requests=acquisition_requests,
        rates_by_unit=rates_by_unit)
    pending = [(name, family, rate) for key, members in sorted(groups.items())
               for family in sorted({f for name in members for f in schedule[name]})
               for name in members for rate in schedule[name].get(family, [])]
    batches = _anchor_batches(pending, weights=weights, batch_size=batch_size,
                              structures=encode_structure)
    if batches:
        row_stream.prime([item[0] for item in batches[0]])


def _served_route_refusals(family, rungs, members, *, encode_structure,
                           projected_units, route_cache=None) -> "dict[str, dict]":
    """The rungs of ``family`` some member cannot be served on, with why.

    ``{str(rung): {"reason": ..., "members": [...]}}`` for every refused rung.
    Two facts refuse a rung, neither a judgement about the format
    (principle 9):

    * the pinned contract attests no served wire for the member's structure
      there (:func:`~prismaquant.tessera_formats.tessera_served_route_refusal`);
    * the member is declared routed but carries no producer projection, and
      the routed served wire differs from the unstructured one: Tessera's
      export intake adopts an unprojected unit on the dense receipt, which
      cannot stamp the routed wire (tessera#662), so the bytes could be
      neither the served ones nor adopted.
    """
    from .tessera_formats import (
        tessera_served_route_refusal, tessera_served_wire_recipe, tessera_wire_recipe)

    # The pinned contract answer is shared by every member of a structure.
    # A caller may share this memo across one row's groups, never across runs.
    # Projection membership is still checked separately for each member.
    memo = {} if route_cache is None else route_cache
    refused: dict[str, dict] = {}
    for rung in sorted(int(r) for r in rungs):
        reasons: dict[str, list[str]] = {}
        for member in members:
            structure = (encode_structure or {}).get(member)
            key = (family, rung, structure)
            if key not in memo:
                memo[key] = tessera_served_route_refusal(family, rung, structure=structure)
            reason = memo[key]
            if (reason is None and structure == "routed_moe"
                    and member not in (projected_units or {})):
                unprojected_key = (*key, "unprojected")
                if unprojected_key not in memo:
                    differs = (tessera_served_wire_recipe(
                        family, rung, structure=structure, refuse_unattested=False)
                        != tessera_wire_recipe(family, rung))
                    memo[unprojected_key] = (
                        f"{family}_R{rung}: a routed unit with no producer projection is "
                        "adopted by Tessera's export intake on the dense receipt, which "
                        "cannot stamp the routed served wire" if differs else None)
                reason = memo[unprojected_key]
            if reason is not None:
                reasons.setdefault(reason, []).append(member)
        if reasons:
            (reason, names), *rest = sorted(reasons.items())
            refused[str(rung)] = {"reason": reason, "members": sorted(names),
                                  **({"other_reasons": {r: sorted(n) for r, n in rest}}
                                     if rest else {})}
    return refused


# ---------------------------------------------------------------------------
# The routed-expert stack, and the sample that prices it
# ---------------------------------------------------------------------------
#
# A routed MoE stack is ONE decision: vLLM loads a packed ``[E, M, N]`` expert
# tensor under a single quantization scheme, and PrismaQuant's union-find
# serving-unit promotion already enforces that (``NameProjection.serving_units``
# keys ``...experts.gate_up_proj`` and
# ``...experts.down_proj`` into the same ``__packed_format__`` group).  The
# campaign measures separate expert Linears, optionally encoding compatible
# units together. A full census still requires every expert's wire and score.
#
# So the campaign may measure a SAMPLE of experts and this module estimates the
# stack from it.  Two things have to be right for that to be honest:
#
# 1. The estimator must be unbiased for the stack's total, not for "the average
#    expert".  With a probability-proportional-to-size draw the Horvitz-Thompson
#    estimator is the one that is: ``T_hat = sum_{e in S} y_e / pi_e``.
#
# 2. The number written into the row must be the quantity the ALLOCATOR will
#    multiply correctly.  ``allocator_solver.predicted_dloss`` prices a row as
#    ``0.5 * h_trace * output_mse``, and it reads ``h_trace`` from the PROBE, so
#    the stack row is multiplied by the packed row's own ``h_trace``.  The
#    quantity that makes that product reproduce the stack's summed per-expert
#    dloss is therefore
#
#        output_mse_stack = ( sum_e h_e * mse_e ) / h_stack
#
#    i.e. the h-weighted MEAN expert MSE, not the sum.  Two independent facts
#    in this repo fix that convention rather than taste:
#
#      * the probe's own identity ``sum_e h_trace_per_expert[e] == h_trace``,
#        which holds to float32 storage precision on all 44 packed rows of
#        ``pq275-2026-09-06/probe-02/probe.pkl`` (2.0e-7 relative on layer 18),
#        so the weights sum to the multiplier the allocator will apply; and
#      * ``allocator_candidates``' own super-item aggregation, which inverts the
#        same product the same way when it has to express a group's summed
#        predicted dloss as one member-shaped MSE:
#        ``effective_mse = base_pred / (0.5 * sum_h)``.
#
# A per-expert member's Fisher weight is ``h_trace_per_expert[e] / R``, where R
# is the number of projections the packed parameter splits into -- the SAME
# split the retired codebook lane's ``expand_packed_expert_rows`` used
# (archived 2026-09-25 with ``tier2_per_expert_counterfactual``, #1304), and
# R comes from the profile's ``packed_expert_projection_names``, never from a
# hardcoded ``gate_up_proj -> 2``.


#: ``float32`` machine epsilon.  The probe accumulates the packed row's
#: ``h_trace`` and its ``h_trace_per_expert`` vector from the same raw sums and
#: divides both by the same global token count
#: (``sensitivity_probe.finalize_fisher_stats``), so the two agree exactly in
#: exact arithmetic.  They are stored at ``float32``, which is the only reason
#: they differ at all, so the admissible discrepancy is a property of that
#: dtype and of how many terms are summed -- not a tolerance anyone chose.
#: Measured on ``probe-02``'s two layer-18 packed rows: 2.0e-7 and 6.6e-8
#: relative, against the bound below of 3.8e-6 at E=32.


#: The anchor fields that must agree across every measured member of one stack
#: rung, mapped to the row field they are written to.  Taking member zero's
#: value the way the dense interpolation path takes ``ordered[0]``'s would let
#: one expert scored under a different activation contract, or without the
#: Hessian, disappear into a stack average.


#: The fields of a stack row's ``sampled_experts`` block that describe the
#: DRAW rather than an estimate made from it.  ``estimator``,
#: ``variance_estimator`` and ``n_random_stratum`` are deliberately absent:
#: they name a Horvitz-Thompson computation, and a row whose value no
#: estimator produced may not claim one (#495 part 3).


# ---------------------------------------------------------------------------
# The dense payload
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Checkpoint identity and resume
# ---------------------------------------------------------------------------













def _manifest_unit_receipts(identity, names):
    """Each unit's W, X and H receipts as a checkpoint manifest recorded them.

    ``identity`` is the manifest's stored identity, or ``None`` when it has
    none. A receipt it does not record reads as ``CHECKPOINT_RECEIPT_ABSENT``,
    which equals no receipt, so the journal refuses that manifest by the
    field's name rather than this function guessing at it.
    """
    from .tessera_row_stream import RECEIPT_FIELDS
    units = identity.get("units") if isinstance(identity, Mapping) else None
    units = units if isinstance(units, Mapping) else {}
    receipts = {}
    for name in sorted(names):
        recorded = units.get(name)
        recorded = recorded if isinstance(recorded, Mapping) else {}
        receipts[name] = {field: recorded.get(field, CHECKPOINT_RECEIPT_ABSENT)
                          for field in RECEIPT_FIELDS}
    return receipts




def campaign_checkpoint_identity(*, weights, acts, hessians, menus, settings: Mapping[str, object],
                                  calibration_identity, serving_scope,
                                  static_scales, static_scale_policy,
                                  expert_projection=None, stack_sampling_identity=None,
                                  structure_by_unit=None, bound_units=None,
                                  unit_receipts=None):
    """Bind the priced population, including score inputs when H is off.

    ``settings`` is the original raw CLI wire mapping, after acquisition
    normalization. The operation copies it before applying the existing wire
    filters; request/dataclass field names are never serialized here.

    ``unit_receipts`` (``{unit: {weight, scoring_rows, hessian}}``) are the
    streaming row head's per-entry receipts (``tessera_row_stream``): the same
    producer ``tensor_identity`` records, taken on reader threads from each
    entry while it was resident, so this identity can be built after every X
    and H has been released. They replace ``acts``/``hessians``/``bound_units``
    and must cover exactly the priced units.

    The static A-side contract is a scoring input like the score rows: the
    policy is env-resolved (``PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE``) and
    the per-unit ``input_global_scale`` is a function of every calibration
    row, not of the bounded rows or the Hessian bound beside it.  Binding
    both here is what makes a checkpoint priced under another calibration or
    policy refuse at the journal (``checkpoint identity mismatch at
    units.<unit>.input_global_scale``) before any of its rows is read; the
    per-row half of the same rule lives in :func:`_checkpoint_anchor_identity`.
    """
    from .production_weight_cache import _production_cache_source_sha256

    api = campaign_wire.require_checkpoint_identity_api()
    settings = dict(settings)
    acquisition_source_weights = settings.get("acquisition_source_weights")
    # Locations and a duplicate digest are never scheduling identity. Opted-in
    # origin + exact atomic-expanded schedule below stay bound by value; unset
    # flags leave the historical settings bytes unchanged.
    settings.pop("acquisition_request", None)
    settings.pop("acquisition_request_sha256", None)
    settings.pop("acquisition_source_weights", None)
    restriction = parse_family_restriction(settings.get("family_restriction"))
    if settings.get("source_scope") is None:
        # Unset, the body's identity is byte-identical to before the flag.
        settings.pop("source_scope", None)
    # PQ #1843: unset, the body's identity is byte-identical to before the flags.
    if not settings.get("allow_pinned"):
        settings.pop("allow_pinned", None)
    if not settings.get("pinned_roster_only"):
        settings.pop("pinned_roster_only", None)
    if restriction is None:
        settings.pop("family_restriction", None)
    else:
        settings["family_restriction"] = restriction
        if (not isinstance(structure_by_unit, Mapping) or set(structure_by_unit) != set(menus)
                or any(s not in ("dense", "routed_moe") for s in structure_by_unit.values())):
            raise ValueError("family restriction identity requires every priced unit's structure")
    # Locations, batch width, publication staging and a wall-clock
    # interruption limit are not encoding/scoring inputs.  All other explicit
    # campaign settings remain bound by default.
    #
    # ``publication_overlap_bytes`` chooses which thread performs two writes
    # whose arguments it does not touch. ``campaign_identity_bytes`` reserves
    # metadata for retaining the same producer receipt. Neither changes the
    # receipt, so binding either would make scheduling wear an identity's clothes.
    #
    # ``streaming_cache_slots``, ``streaming_prefetch_workers`` and
    # ``streaming_cache_headroom_gb`` size the layer cache, its prefetch pool and
    # the free-memory floor it keeps (`cost_streaming.build_streamed_causal_lm`); every
    # captured activation, Hessian and wire is the same object at any of their
    # values, so they are scheduling too.  ``streaming`` itself and
    # ``streaming_capture_policy`` stay bound: they choose what is captured.
    #
    # ``units``, ``calibration_census`` and ``census_out`` are locations too,
    # and each one's load-bearing content is already bound by value somewhere
    # in this identity: the selection by the ``units`` map below (which holds
    # exactly the selected units) and ``stack_sampling_identity`` (the probe,
    # inclusion probabilities and audit draw), the census by ``calibration.fit_tokens`` /
    # ``fit_tokens_min``, and ``census_out`` by nothing, because a census run
    # writes no checkpoint. Binding the paths instead would make two shards
    # given the same selection under different filenames two identities, and
    # would make every sharded run's identity differ under two spellings of one
    # selection.  A sharded identity is still narrower than a whole-scope one --
    # the ``units`` map holds exactly the selection -- so a shard does not
    # resume a whole-scope journal; it adopts that journal's rows through
    # ``--seed-checkpoint``, one verified row at a time.
    for name in ("out", "cache_dir", "checkpoint", "deadline_seconds",
                 "units", "calibration_census", "census_out",
                 "capture_calibration_out", "calibration_cache", "calibration_cache_sha256",
                 # The optional intake is bound by its calibration receipt, not
                 # path spelling or duplicated settings. Absent intake keeps
                 # the historical settings identity unchanged.
                 "calibration_input", "calibration_input_sha256", "calibration_corpus",
                 "calibration_input_receipt",
                 "seed_checkpoint", "seed_wire_dir", "anchor_batch_size",
                 "publication_overlap_bytes", "campaign_identity_bytes",
                 "campaign_identity_threads", "source_snapshot_policy",
                 "streaming_cache_slots", "streaming_prefetch_workers",
                 "streaming_cache_headroom_gb", "row_head",
                 # PQ #1532: the source-identity proof is a *read shortcut* (it
                 # seeds the streamed source identity the run re-derives), not a
                 # setting of the run; binding its path or digest would fork the
                 # checkpoint identity between a proved and an unproved launch.
                 "source_identity_cache", "source_identity_cache_sha256"):
        settings.pop(name, None)
    if unit_receipts is not None and (bound_units is not None
                                      or set(unit_receipts) != set(weights)):
        raise ValueError("unit receipts must cover exactly the priced units, without a hold")
    return {
        **({"family_restriction": {"policy": restriction,
             "structure_by_unit": dict(sorted(structure_by_unit.items()))}}
           if restriction is not None else {}),
        **({"stack_sampling_identity": stack_sampling_identity}
           if stack_sampling_identity else {}),
        "campaign_schema": SCHEMA,
        "currency": CURRENCY,
        "settings": settings,
        "calibration": calibration_identity,
        "serving_scope": serving_scope,
        "encoder_recipe": th.encoder_recipe(),
        "prismaquant_source_sha256": _production_cache_source_sha256(),
        "encoder_source_sha256": api.encoder_source_sha256(),
        "input_global_scale_policy": str(static_scale_policy),
        # The producer's projection the packed units were priced under: its
        # source checkpoint identity and every sealed unit record.  A resume
        # from a campaign that projected another checkpoint, or none, refuses
        # here before a packed row is read (PrismaQuant #183).
        "expert_projection": (
            None if expert_projection is None else {
                "source": dict(expert_projection["producer"]["source"]),
                "stacks": {
                    stack: {name: dict(unit) for name, unit in sorted(units.items())}
                    for stack, units in sorted(expert_projection["stacks"].items())
                },
            }),
        "units": {
            name: {
                **({"acquisition_source_weight": dict(acquisition_source_weights[name])}
                   if acquisition_source_weights is not None else {}),
                # A campaign hold creates this exact producer template once
                # before journal admission.  The journal retains the same
                # source/H records the former direct calls made, without a
                # second DtoH copy later for every published receipt.
                "weight": (unit_receipts[name]["weight"] if unit_receipts is not None
                           else bound_units[name].campaign_inputs()["source"]
                           if bound_units is not None else api.tensor_identity(weight)),
                "scoring_rows": (unit_receipts[name]["scoring_rows"] if unit_receipts is not None
                                 else None if acts.get(name) is None
                                 else api.tensor_identity(acts[name])),
                "hessian": (unit_receipts[name]["hessian"] if unit_receipts is not None
                            else None if hessians.get(name) is None else
                            (bound_units[name].campaign_inputs()["hessian"]
                             if (bound_units is not None and
                                 bound_units[name].campaign_inputs()["hessian"] is not None)
                             else api.tensor_identity(hessians[name]))),
                "input_global_scale": (
                    None if static_scales.get(name) is None
                    else float(static_scales[name])),
                "menu": sorted(rung.format_name for rung in menus[name]),
            }
            for name, weight in sorted(weights.items())
        },
    }


def _bound_tensor_signature(tensor):
    """Track one versioned tensor during a caller-owned immutable lifetime."""
    try:
        version = tensor._version
    except RuntimeError as exc:
        raise ValueError("bound checkpoint identity requires a version-tracked tensor") from exc
    return (id(tensor), tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
            tuple(tensor.shape), tuple(tensor.stride()), tensor.device, tensor.dtype, version)


class _BoundCheckpointUnitIdentity:
    """Actual producer inputs shared only across one unit's closed format roster.

    This owns identity records, never weights or a second tensor cache. Strong
    references and storage/version guards delimit the caller's immutable source
    and H lifetime; close before unloading, replacing or mutating either tensor.
    The ordinary producer API supplies the source/H/settings identity once. Its
    recipe remains Tessera's per-format wire_recipe, not a copied receipt field.
    """

    def __init__(self, anchors, *, source_weight, calibration_source,
                 projected_unit, static_scales, retain_source_receipt=True):
        import copy
        import torch
        from types import SimpleNamespace
        from .production_weight_cache import _cb_cache_tensor_identity

        anchors = tuple(anchors)
        names = {anchor.qname for anchor in anchors}
        formats = {anchor.format_name for anchor in anchors}
        if len(names) != 1 or not formats or len(formats) != len(anchors):
            raise ValueError("bound checkpoint identity requires one unit's unique closed anchor roster")
        self._closed = False
        self._name = next(iter(names))
        self._formats = frozenset(formats)
        self._weight = source_weight
        self._weight_signature = _bound_tensor_signature(source_weight)
        if not bool(torch.isfinite(source_weight).all()):
            raise ValueError("bound checkpoint identity source is nonfinite")
        # A representative H-bearing anchor supplies the common calibration
        # record; H-free members remove that record, as the ordinary API does.
        representative = max(anchors, key=lambda anchor: bool(anchor.hessian_applied))
        self._calibration = calibration_source if representative.hessian_applied else None
        self._hessian = None if self._calibration is None else self._calibration.hessians[self._name]
        self._hessian_signature = None if self._hessian is None else _bound_tensor_signature(self._hessian)
        if self._hessian is not None and not bool(torch.isfinite(self._hessian).all()):
            raise ValueError("bound checkpoint identity Hessian is nonfinite")
        self._projection = projected_unit
        self._projection_record = copy.deepcopy(projected_unit)
        self._template = _checkpoint_anchor_identity(representative,
            weights={self._name: source_weight},
            menus={self._name: [SimpleNamespace(format_name=fmt) for fmt in formats]},
            calibration_source=calibration_source, static_scales=static_scales,
            projected_units={} if projected_unit is None else {self._name: projected_unit},
            check_scoring_metadata=False)
        self._settings = self._calibration_settings()
        # H-free rosters retain no H receipt. The run-level identity uses its
        # ordinary direct H receipt for those units, keeping every retained H
        # record behind this holder's source/version guard.
        self._campaign_hessian = (None if self._template["calibration"] is None else
                                  copy.deepcopy(self._template["calibration"]["hessian"]))
        # Joint AURA needs this separate cache receipt. Campaign pricing only
        # keeps producer identities, so it must not add a second full CPU hash.
        self._source_receipt = (_cb_cache_tensor_identity(source_weight)
                                if retain_source_receipt else None)
        self._guard()

    def _calibration_settings(self):
        """Snapshot producer settings without resealing the resident Hessian."""
        if self._calibration is None:
            return None
        import copy
        api = campaign_wire.require_checkpoint_identity_api()
        source = self._calibration
        # This is ActivationSource.config_block() minus its capture_sha256()
        # call. The producer receipt has already sealed H in `_template`; a
        # lifetime guard must compare the same scalar/provenance inputs without
        # synchronizing and hashing H a second time.
        return {
            "ldlq_sigma": source.ldlq_sigma,
            "ldlq_block": (source.ldlq_block if isinstance(source.ldlq_block, int)
                           else copy.deepcopy(dict(source.ldlq_block))),
            "refit_objective": (source.refit_objective if isinstance(source.refit_objective, str)
                                else copy.deepcopy(dict(source.refit_objective))),
            "refit_objective_trailing": (
                None if source.refit_objective_trailing is None
                else source.refit_objective_trailing
                if isinstance(source.refit_objective_trailing, str)
                else copy.deepcopy(dict(source.refit_objective_trailing))),
            "refit_reach_floor": bool(source.refit_reach_floor),
            "refit_gauss_seidel": (bool(source.refit_gauss_seidel)
                                   if isinstance(source.refit_gauss_seidel, bool)
                                   else copy.deepcopy(dict(source.refit_gauss_seidel))),
            "hessian": {key: source.provenance[key] for key in api.HESSIAN_IDENTITY},
        }

    def _guard(self):
        if self._closed:
            raise ValueError("bound checkpoint identity is closed")
        if _bound_tensor_signature(self._weight) != self._weight_signature:
            raise ValueError("bound checkpoint source tensor changed")
        if self._calibration is not None:
            if (self._calibration.hessians.get(self._name) is not self._hessian or
                    _bound_tensor_signature(self._hessian) != self._hessian_signature):
                raise ValueError("bound checkpoint Hessian tensor changed")
            if self._calibration_settings() != self._settings:
                raise ValueError("bound checkpoint calibration settings changed")
        if self._projection != self._projection_record:
            raise ValueError("bound checkpoint producer projection changed")

    def derive(self, *, source_weight, qname, format_name, grid, rung,
               activation, projected_unit, structure=None):
        import copy
        self._guard()
        if (source_weight is not self._weight or qname != self._name or
                format_name not in self._formats or projected_unit != self._projection_record or
                (activation is not None and activation is not self._calibration)):
            raise ValueError("inputs differ from bound checkpoint unit")
        result = copy.deepcopy(self._template)
        result["calibration"] = None if activation is None else result["calibration"]
        if activation is not None and result["calibration"] is None:
            raise ValueError("bound checkpoint unit has no H-bearing identity")
        # served_recipe is the owner encoding_input_identity/unit_input_identity
        # stamp (tessera#662): the research wire_recipe wherever no structure
        # moves it, the served wire of a routed stack where one does.  Its full
        # settings, including body/plane/reach, are resolved anew; the template
        # holds nothing else that depends on structure.
        from tessera.export import served_recipe
        recipe = (served_recipe(grid, rung) if structure is None
                  else served_recipe(grid, rung, structure))
        result["recipe"] = {"grid": grid.name, "q256": rung, **recipe.to_config()}
        return result

    def source_receipt(self, source_weight):
        import copy
        self._guard()
        if source_weight is not self._weight or self._source_receipt is None:
            raise ValueError("bound checkpoint unit has no source receipt")
        return copy.deepcopy(self._source_receipt)

    def replace_calibration_source(self, calibration_source):
        """Adopt the post-export resident owner without changing the receipt.

        The reference source authenticates the same resident H objects.  The
        equality and identity checks make that handoff explicit: a new source,
        H, or numerical setting cannot silently inherit the old receipt.
        """
        if self._calibration is None:
            # This unit has an H-free closed roster. Its campaign H receipt
            # was sealed at construction, while derive() never consults an
            # activation owner for this roster; the later shared reference
            # owner therefore needs no per-unit rebinding.
            return
        if calibration_source is None or calibration_source.hessians.get(self._name) is not self._hessian:
            raise ValueError("replacement calibration Hessian differs from bound checkpoint unit")
        old = self._calibration
        self._calibration = calibration_source
        try:
            if self._calibration_settings() != self._settings:
                raise ValueError("replacement calibration settings differ from bound checkpoint unit")
            self._guard()
        except BaseException:
            self._calibration = old
            raise

    def campaign_inputs(self):
        """The unchanged producer source/H fields for the run-level receipt."""
        import copy
        self._guard()
        calibration = self._template["calibration"]
        return {"source": copy.deepcopy(self._template["source"]),
                "hessian": copy.deepcopy(self._campaign_hessian)}

    def hessian_identity(self, hessian):
        """The sealed producer H receipt, for exactly the bound tensor object.

        ``None`` for an H-free roster, which retains no H receipt.  Anything
        but the guarded tensor is refused: the receipt is a digest of that
        object's bytes and of nothing else, so it may stand in for
        ``tensor_identity`` only where the caller holds the same object.
        """
        import copy
        self._guard()
        if self._campaign_hessian is None:
            return None
        if hessian is not self._hessian:
            raise ValueError("bound checkpoint unit holds no H receipt for that tensor")
        return copy.deepcopy(self._campaign_hessian)

    def observed_metadata_bytes(self):
        """Report CPython reachable metadata after construction; not admission."""
        import sys
        # The source maps, tensors, and menu strings pre-date this hold.  This
        # diagnostic counts only its newly retained graph and is intentionally
        # not an allocator promise: `getsizeof` is interpreter-specific.
        borrowed = {id(self._weight), id(self._hessian), id(self._calibration),
                    id(self._name), *(id(value) for value in self._formats)}
        seen = set()
        def size(value):
            marker = id(value)
            if marker in seen or marker in borrowed:
                return 0
            seen.add(marker)
            total = sys.getsizeof(value)
            if isinstance(value, dict):
                total += sum(size(k) + size(v) for k, v in value.items())
            elif isinstance(value, (tuple, list, frozenset, set)):
                total += sum(size(item) for item in value)
            return total
        return size(self) + size(self.__dict__)

    def close(self):
        self._closed = True
        self._weight = self._hessian = self._calibration = None
        self._template = self._source_receipt = self._campaign_hessian = self._projection = None

    def __enter__(self):
        self._guard()
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        try:
            if exc_type is None:
                self._guard()
        finally:
            self.close()


def bind_checkpoint_unit_identity(anchors, *, source_weight, calibration_source,
                                  projected_unit, static_scales,
                                  retain_source_receipt=True):
    """Bind actual inputs once; accepts no caller-supplied hash or receipt."""
    return _BoundCheckpointUnitIdentity(anchors, source_weight=source_weight,
        calibration_source=calibration_source, projected_unit=projected_unit,
        static_scales=static_scales, retain_source_receipt=retain_source_receipt)


def _campaign_identity_anchor_roster(name, menu, *, calibration_source, static_scales):
    """Construct the closed producer-format roster once from existing menu refs."""
    from types import SimpleNamespace
    from .tessera_formats import parse_tessera_format_name, tessera_wire_recipe
    from .tessera_render import rung_accepts_hessian
    anchors = []
    for entry in menu:
        family, rung = parse_tessera_format_name(entry.format_name)
        if family is None:
            raise RuntimeError(f"campaign menu is not Tessera: {entry.format_name!r}")
        wire = tessera_wire_recipe(family, rung)
        anchors.append(SimpleNamespace(
            qname=name, format_name=entry.format_name, family=family.name,
            body_rate_q256=rung,
            hessian_applied=(calibration_source is not None and
                             rung_accepts_hessian(entry.format_name, wire)),
            input_global_scale=(static_scales.get(name)
                                if _format_executes_static_activation_contract(entry.format_name)
                                else None)))
    return anchors


# An opt-in campaign retains producer receipt dictionaries for the whole closed
# roster. These terms bound CPython object headers/slots and the producer JSON
# receipt topology independently of a particular digest value. They are
# deliberately stated as admission terms, not as a measurement: the live
# `observed_metadata_bytes` diagnostic below tests the bound on each run.
IDENTITY_HOLD_UNIT_OBJECT_BYTES = 32 * 1024
IDENTITY_HOLD_SERIALIZED_BYTE_MULTIPLIER = 8
IDENTITY_HOLD_PLAN_MAPPING_ENTRY_BYTES = 256
IDENTITY_HOLD_PLAN_UNIT_FIXED_BYTES = 4096
IDENTITY_HOLD_PLAN_SERIALIZED_BYTE_MULTIPLIER = 4
# The roster a holder is built from is transient: one ``SimpleNamespace`` per
# closed-menu format with six attributes (`_campaign_identity_anchor_roster`),
# plus the holder constructor's own one-attribute namespace per format, its
# working tuple/sets, and the rung integers those carry.  Two rosters are live
# at the peak, because the next unit's is built before the previous binding is
# released.  The per-format figure is the interpreter's own object cost with
# the six-slot instance dict counted at its CPython 3.12 size, rounded up.
IDENTITY_ROSTER_TRANSIENT_FORMAT_BYTES = 1024
IDENTITY_ROSTER_TRANSIENT_LIVE_ROSTERS = 2


def _identity_threads(threads) -> int:
    """The builder count the identity hold and the resumed-wire verify use."""
    if type(threads) is not int or threads < 1:
        raise ValueError("campaign identity threads must be a positive int")
    return threads


def _identity_threads_for_this_process(requested) -> int:
    """Never more builders than the CPUs this row was admitted with."""
    import os
    requested = _identity_threads(requested)
    try:
        admitted = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        admitted = os.cpu_count() or 1
    return max(1, min(requested, admitted))


class _EncoderSourceSealAhead:
    """Overlap the producer's exact cold source seal, using its own cache.

    This is a row-owned future on the existing affinity-bound I/O executor,
    not another digest or memo. Join before a producer identity can first read
    the seal; the unchanged identity/bind API then sees the producer's cache.
    """

    def __init__(self):
        from .io_engine import ENGINE
        self._future = ENGINE.submit(self._take)

    @staticmethod
    def _take():
        return campaign_wire.require_checkpoint_identity_api().encoder_source_sha256()

    def wait(self) -> str:
        return self._future.result()

    def finish(self) -> None:
        # Drain on every exit, but do not mask a prior preparation failure.
        # Successful pricing MUST wait explicitly and propagate seal failure.
        try:
            self._future.result()
        except BaseException:
            pass


def _start_encoder_source_seal_ahead(args, source_scope):
    if args.census_out or args.capture_calibration_out:
        return None
    ahead = _EncoderSourceSealAhead()
    source_scope.callback(ahead.finish)
    return ahead


class _SealAhead:
    """Take the calibration owner's capture seal on a helper thread.

    ``ActivationSource.capture_sha256`` digests every resident H the first
    time it is read (``_seal``), and that first read used to be the first
    ``config_block`` inside the identity hold: ~40 GiB of sha256 at the head
    of an 864-unit expert row, serial, with the GPU idle behind it.  Started
    the moment the owner exists, the same digest runs under the producer
    projection and the identity plan instead.  ``wait`` re-raises the helper's
    failure and MUST precede every other first read of the owner, because
    ``_seal`` has no lock and two first readers would each digest the
    population.  The seal is the producer's own, taken by the producer's own
    method; nothing here reads a tensor or changes what is sealed.
    """

    def __init__(self, source):
        import threading
        import time
        self._source = source
        self._error = None
        self._started = time.monotonic()
        self.seconds = None
        self._thread = threading.Thread(target=self._run, name="campaign-seal-ahead",
                                        daemon=True)
        self._thread.start()

    def _run(self):
        import time
        try:
            self._source.capture_sha256()
        except BaseException as error:  # re-raised on the campaign thread
            self._error = error
        finally:
            self.seconds = time.monotonic() - self._started

    def wait(self) -> float:
        """Join; the seconds the campaign thread spent waiting."""
        import time
        started = time.monotonic()
        self._thread.join()
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError("the calibration owner's capture seal failed ahead of "
                               "the identity hold") from error
        return time.monotonic() - started

    def finish(self) -> None:
        self._thread.join()


def _frozenset_table_bytes(entries: int) -> int:
    """The interpreter's own size of a frozenset holding ``entries`` items.

    The holder retains one frozenset of format-name references; the strings
    are the menu's and pre-date the hold, so what it adds is the set's slot
    table, which CPython sizes by a fill rule this asks the interpreter for
    rather than restates.  ``observed_metadata_bytes`` counts the same table.
    """
    import sys
    return sys.getsizeof(frozenset(range(int(entries))))


def _campaign_identity_metadata_plan(*, weights, menus, calibration_source,
                                     projected_units, static_scales, threads=1):
    """Conservatively bound holder metadata without producer or tensor mutation.

    One unit retains a holder/result mapping, signatures, the producer receipt
    dictionaries, and a frozenset of references to every closed menu format.
    The fixed object term covers the first three; the frozenset term is the
    interpreter's own table size for that many references.  Receipt strings
    and dict keys are bounded from known unit/shape/settings/projection
    serialization lengths, multiplied by the documented CPython object/slot
    envelope.  The format names themselves are the menu's objects, borrowed by
    reference, and the sealed template carries one recipe, not the roster, so
    no per-format serialization is retained.

    The second value is the planning-and-construction transient: the bound
    map and its largest JSON string, plus the closed rosters live while
    holders are built (`IDENTITY_ROSTER_TRANSIENT_*`): two for one builder
    (the next unit's roster is built before the previous binding is released)
    and one more for every further builder ``threads`` adds, each holding
    the roster of the unit it is binding.
    """
    import json
    live_rosters = IDENTITY_ROSTER_TRANSIENT_LIVE_ROSTERS + (_identity_threads(threads) - 1)
    # The producer config owner may itself verify H commitments. Planning must
    # not invoke it before the one real receipt; reserve its bounded JSON
    # settings envelope in the fixed holder term instead.
    settings_bytes = 4096 if calibration_source is not None else 0
    planned, largest_serialization, widest_roster = {}, 0, 0
    for name in sorted(weights):
        shape_bytes = len(json.dumps(list(weights[name].shape), separators=(",", ":")).encode())
        projection_bytes = len(DIRECT_ASCII_STRICT.encoded((projected_units or {}).get(name)))
        receipt_bytes = len(name.encode()) + shape_bytes + settings_bytes + projection_bytes
        planned[name] = (IDENTITY_HOLD_UNIT_OBJECT_BYTES +
                         _frozenset_table_bytes(len(menus[name])) +
                         IDENTITY_HOLD_SERIALIZED_BYTE_MULTIPLIER * receipt_bytes)
        largest_serialization = max(largest_serialization, receipt_bytes)
        widest_roster = max(widest_roster, len(menus[name]))
    # `planned` remains live until holders are built. Each loop iteration also
    # materializes one shape/projection JSON string; its worst live string plus
    # the map's entry table is a separate, pre-admitted planning transient, and
    # so is the closed roster each holder is constructed from.
    scratch = (IDENTITY_HOLD_PLAN_UNIT_FIXED_BYTES +
               IDENTITY_HOLD_PLAN_MAPPING_ENTRY_BYTES * len(planned) +
               IDENTITY_HOLD_PLAN_SERIALIZED_BYTE_MULTIPLIER * largest_serialization +
               live_rosters * (
                   IDENTITY_ROSTER_TRANSIENT_FORMAT_BYTES * widest_roster +
                   _frozenset_table_bytes(widest_roster)))
    return planned, scratch


def _campaign_bound_identities(*, weights, menus, calibration_source,
                               projected_units, static_scales, metadata_bounds=None,
                               threads=1):
    """One producer receipt template per priced unit, held through publication.

    Binding a unit hashes its resident weight and Hessian through the
    producer's ``encoding_input_identity``; on an 864-unit expert row that is
    ~55 GiB of sha256 at the head, so with ``threads > 1`` the holders are
    built on that many workers.  hashlib and the tensor staging release the
    GIL, each producer receipt is a function of its own unit's tensors and
    the already-sealed owner, and the mapping is filled in ``sorted(weights)``
    order whatever the completion order, so every template and the run-level
    identity derived from them are byte for byte the serial ones
    (``tests/test_tessera_bound_identity.py``).  The owner MUST be sealed
    before the workers start: ``ActivationSource._seal`` has no lock, and two
    first ``config_block`` readers would each digest the population.  With
    ``threads == 1`` this is the historical serial loop on the calling thread.
    """
    threads = _identity_threads(threads)
    names = sorted(weights)

    def build(name):
        anchors = _campaign_identity_anchor_roster(
            name, menus[name], calibration_source=calibration_source,
            static_scales=static_scales)
        return bind_checkpoint_unit_identity(
            anchors, source_weight=weights[name], calibration_source=calibration_source,
            projected_unit=(projected_units or {}).get(name), static_scales=static_scales,
            retain_source_receipt=False)

    def admit(name, unit):
        if metadata_bounds is not None and unit.observed_metadata_bytes() > metadata_bounds[name]:
            raise RuntimeError("campaign identity metadata exceeded its preallocation bound")

    result = {}
    if threads == 1 or len(names) <= 1:
        try:
            for name in names:
                result[name] = build(name)
                admit(name, result[name])
            return result
        except BaseException:
            for unit in result.values():
                unit.close()
            raise

    from concurrent.futures import ThreadPoolExecutor
    futures = []
    try:
        with ThreadPoolExecutor(max_workers=min(threads, len(names)),
                                thread_name_prefix="campaign-identity") as pool:
            try:
                futures = [pool.submit(build, name) for name in names]
                for name, future in zip(names, futures):
                    result[name] = future.result()
                    admit(name, result[name])
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        if list(result) != names:
            raise RuntimeError("campaign identity hold was not filled in sorted unit order")
        return result
    except BaseException:
        # Every holder any worker finished, whether or not it reached the
        # result mapping, is closed on the way out.
        for future in futures:
            if future.done() and not future.cancelled() and future.exception() is None:
                future.result().close()
        raise


def _bound_hessian_identities(bound_units, hessians):
    """The hold's sealed per-unit H receipts, for the reference descriptor.

    Each is the ``tensor_identity`` the producer stamped on the unit's
    template, handed back only for the very tensor object it was taken from;
    H-free rosters contribute nothing and are digested by the descriptor.
    """
    identities = {}
    for name, unit in bound_units.items():
        tensor = hessians.get(name)
        if tensor is None:
            continue
        identity = unit.hessian_identity(tensor)
        if identity is not None:
            identities[name] = identity
    return identities


def _stream_unit_identity(name, *, weight, inputs, hessian, source, menu, projected_unit,
                          static_scales, metadata_bound=None):
    """One unit's bound producer identity and run-level receipts, for the stream head.

    Runs on a row-stream reader thread over host tensors only
    (``tessera_row_stream.RowStream``). With a nonempty menu the unit's
    closed-roster holder is built exactly as the load-all hold builds it, over
    this unit's own ``ActivationSource``, and the run-level weight and H
    receipts are the ones its template sealed; a unit whose menu admits
    nothing has no holder and is receipted directly. Returns ``(holder,
    {weight, scoring_rows, hessian})``.
    """
    api = campaign_wire.require_checkpoint_identity_api()
    holder = None
    try:
        weight_receipt = hessian_receipt = None
        if menu:
            holder = bind_checkpoint_unit_identity(
                _campaign_identity_anchor_roster(name, menu, calibration_source=source,
                                                 static_scales=static_scales),
                source_weight=weight, calibration_source=source,
                projected_unit=projected_unit, static_scales=static_scales,
                retain_source_receipt=False)
            if metadata_bound is not None and holder.observed_metadata_bytes() > metadata_bound:
                raise RuntimeError("campaign identity metadata exceeded its preallocation bound")
            receipts = holder.campaign_inputs()
            weight_receipt, hessian_receipt = receipts["source"], receipts["hessian"]
        return holder, dict(
            weight=api.tensor_identity(weight) if weight_receipt is None else weight_receipt,
            scoring_rows=api.tensor_identity(inputs),
            hessian=api.tensor_identity(hessian) if hessian_receipt is None else hessian_receipt)
    except BaseException:
        if holder is not None:
            holder.close()
        raise


def _verify_wire_records_on_threads(pending, wire_dir, *, threads):
    """Run ``tessera_campaign_wire.checkpoint_wire_record`` for pending resumed rows.

    ``pending`` is ``[(anchor, identity, existing), ...]`` in the adopt loop's
    own order and the records come back in that order; the first failure in
    that order is what raises, so a refusal names the same row it always
    did.  Each verification reads one published wire blob and re-digests it
    through the producer's receipt grammar, which is the whole of a resumed
    row's head, so the blobs are read and digested on ``threads`` workers.
    """
    threads = _identity_threads(threads)
    if threads == 1 or len(pending) <= 1:
        return [campaign_wire.checkpoint_wire_record(anchor, wire_dir, identity, existing=existing)
                for anchor, identity, existing in pending]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(threads, len(pending)),
                            thread_name_prefix="campaign-wire-verify") as pool:
        futures = [pool.submit(campaign_wire.checkpoint_wire_record, anchor, wire_dir, identity,
                               existing=existing)
                   for anchor, identity, existing in pending]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def _checkpoint_anchor_identity(anchor, *, weights, menus, calibration_source,
                                static_scales, projected_units=None, bound_unit=None,
                                structure=None, check_scoring_metadata: bool = True, profile=None):
    """The resumed row's inputs, as this run's producer would stamp them.

    A unit in ``projected_units`` (``{qname: producer unit record}``) is a
    packed expert the producer projects; its receipt is sealed with the
    producer's ``unit_input_identity`` -- the same encoding inputs plus the
    projection record -- because that is the identity the exporter's
    ``--cached-expert-units`` intake recomputes from the source bytes.  A
    dense unit keeps ``encoding_input_identity``.

    ONE gate per resumed row, in the order the producer resolves the inputs:
    the unit is priced, the format is a Tessera rung on the current menu, the
    Hessian applicability is what ``rung_accepts_hessian`` says for that
    rung's wire, and the A-side contract on the row is what
    :func:`_measure_anchor` stamps for that rung under this run's static
    scales (:func:`_require_resumable_anchor`).  Only then is the producer's
    ``encoding_input_identity`` asked for, so the wire receipt is verified
    against a row already known to be a price of this run.

    ``structure`` is the unit's serving structure when the run declares one
    (#1502). It moves the stamped recipe to the wire that structure is served
    on, exactly as :func:`_prepare_anchor` moves the encode, and is passed to
    the producer's identity functions only when set, so a run that declares
    none stamps what it always stamped.
    """
    from .tessera_formats import parse_tessera_format_name, tessera_served_wire_recipe
    from .tessera_render import rung_accepts_hessian

    if anchor.qname not in weights:
        raise RuntimeError(f"checkpoint anchor names an unknown unit: {anchor.qname!r}")
    parsed = parse_tessera_format_name(anchor.format_name)
    if parsed is None:
        raise RuntimeError(f"checkpoint anchor format is not Tessera: {anchor.format_name!r}")
    family, rung = parsed
    if (anchor.family, anchor.body_rate_q256) != (family.name, rung):
        raise RuntimeError("checkpoint anchor family/rung disagrees with its format")
    if anchor.format_name not in {entry.format_name for entry in menus[anchor.qname]}:
        raise RuntimeError(f"checkpoint anchor is outside the current menu: {anchor.format_name}")
    # A receipt is checked against the wire it was stamped on; whether that
    # wire serves is the planner's refusal, not this reader's.
    wire = tessera_served_wire_recipe(family, rung, structure=structure,
                                      refuse_unattested=False)
    activation = (calibration_source if calibration_source is not None
                  and rung_accepts_hessian(anchor.format_name, wire) else None)
    if bool(anchor.hessian_applied) != (activation is not None):
        raise RuntimeError("checkpoint anchor Hessian applicability disagrees with the producer")
    _require_resumable_anchor(anchor, static_scales,
                             check_scoring_metadata=check_scoring_metadata, profile=profile)
    api = campaign_wire.require_checkpoint_identity_api()
    projected = (projected_units or {}).get(anchor.qname)
    if bound_unit is not None:
        if not isinstance(bound_unit, _BoundCheckpointUnitIdentity):
            raise ValueError("expected an actual bound checkpoint unit identity")
        return bound_unit.derive(source_weight=weights[anchor.qname], qname=anchor.qname,
            format_name=anchor.format_name, grid=family.payload_grid(), rung=int(rung),
            activation=activation, projected_unit=projected, structure=structure)
    structured = {} if structure is None else {"structure": structure}
    if projected is not None:
        return api.unit_input_identity(
            weights[anchor.qname], dict(projected), family.payload_grid(), int(rung),
            activation=activation, **structured,
        )
    return api.encoding_input_identity(
        weights[anchor.qname], anchor.qname, family.payload_grid(), int(rung),
        activation=activation, **structured,
    )




# Job namespaces on the one publication queue. They exist so a completion
# says which link of the chain finished: the files, the receipt read back off
# them, or the journal write that cites the receipt.
FILES_JOB = "files"
RECEIPT_JOB = "receipt"
CHECKPOINT_JOB = "checkpoint"


class _AnchorPublicationLedger:
    """Decides when a measured anchor is allowed to become a journal row.

    The campaign's rule is a chain: the render and wire files are written,
    then the wire receipt is made by reading the published file back, then the
    anchor row and that receipt go into the checkpoint together.  Each link
    must not precede the one before it.  Doing all three on the encode thread
    made that ordering free; doing any of them elsewhere makes it something
    this object has to hold.

    It holds it with one queue.  The publisher runs one writer thread in
    submission order, so a receipt job submitted after a file job runs after
    it, and a checkpoint job submitted after both runs after both.  Nothing
    here reorders; it only decides what to put on the queue and when to
    believe a completion.

    ``make_record(anchor, identity)`` is the caller's receipt call, and
    ``journal_anchor(anchor, record)`` is the caller's three lines that put a
    row and its receipt into the pending checkpoint.  With no publisher both
    run inline exactly where they always ran, which is what makes the default
    path the historical path rather than a re-implementation of it.
    """

    def __init__(self, *, publisher, make_record, journal_anchor):
        self._publisher = publisher
        self._make_record = make_record
        self._journal = journal_anchor
        self._staged: dict = {}
        self._records: dict = {}
        self._checkpoint_seq = 0

    @property
    def staged(self) -> dict:
        return self._staged

    @property
    def active(self) -> bool:
        return self._publisher is not None

    def open(self, publisher) -> None:
        """Start deferring, from a pass-through that has nothing outstanding.

        The ledger is constructed without a publisher so the rows a seed
        adopted are journalled inline, on the historical path, before any
        thread exists.  Attaching one afterwards is only legal while nothing
        is staged, because a staged anchor belongs to a queue and there was
        no queue to put it on.
        """
        if self._staged or self._records:
            raise RuntimeError(
                "the publication ledger cannot start deferring with "
                f"{len(self._staged)} anchor(s) already staged")
        self._publisher = publisher

    def record(self, anchor, identity=None, *, derive=None) -> None:
        """Journal now, or when this anchor's own bytes have been written.

        ``identity`` is the anchor's producer input identity, already
        computed.  ``derive`` is instead a zero-argument callable that
        computes it, and the two are exclusive.  A caller passes ``derive``
        only when the derivation touches no device: the writer is a CPU/IO
        thread by contract (the producer's graph capture forbids surprise
        device work from another thread while a capture may be running), so
        an identity that hashes a resident weight or Hessian is computed on
        the encode thread and handed over as a value.  With a publisher the
        callable runs on the writer inside this anchor's receipt job, ahead
        of the read-back it seals; without one it runs inline, here, exactly
        where the value would have been computed.
        """
        if (identity is None) == (derive is None):
            raise ValueError("record takes exactly one of identity or derive")
        if self._publisher is None:
            if derive is not None:
                identity = derive()
            self._journal(anchor, self._make_record(anchor, identity))
            return
        key = (anchor.qname, anchor.format_name)
        if key in self._staged:
            # Two anchors for one (unit, rung) share a publication key, and
            # one of them would be journalled under the other's receipt. The
            # anchor schedule does not produce this; if it ever does it is a
            # scheduling defect, not something to average over.
            raise RuntimeError(
                f"{anchor.qname} {anchor.format_name} is already staged")
        self._staged[key] = anchor

        def make():
            # Runs on the writer, behind this unit's own file job, so the
            # read-back inside the receipt call reads a file that exists and
            # the encode thread never waits for it.  A deferred identity is
            # derived first, on this thread, and a refusal there fails the
            # publication the same way a refused read-back does.
            sealed = identity if derive is None else derive()
            self._records[key] = self._make_record(anchor, sealed)

        self._publisher.submit(PublicationJob(
            key=(RECEIPT_JOB, *key), charged_bytes=0, publish=make))

    def submit_checkpoint(self, write) -> None:
        """Order one journal write behind every receipt it depends on."""
        if self._publisher is None:
            write()
            return
        key = (CHECKPOINT_JOB, self._checkpoint_seq)
        self._checkpoint_seq += 1
        self._publisher.submit(PublicationJob(
            key=key, charged_bytes=0, publish=write))

    def apply_completed(self) -> int:
        """Journal whatever the writer has finished since the last call."""
        if self._publisher is None:
            return 0
        return self._apply(self._publisher.completed())

    def close(self) -> int:
        """Journal every receipt that really completed, then stop staging.

        Called on the way out of the pricing loop, including on the way out of
        an exception, and always after the publisher's own thread has stopped.
        What is left then is a list of completions the writer finished before
        it stopped: those are files that exist and receipts that were read
        back off them, and a batch that succeeded before a later one failed
        must not lose its journal row because the loop is unwinding.  That
        loss would be a regression against the synchronous path, which
        checkpointed each batch before starting the next.

        Everything still staged is dropped.  No receipt, no row: those units
        are re-priced by the next run over the same paths.  Nothing is
        refused here, because this runs while an exception is already on its
        way out and the original failure is the one worth reporting.  After
        this the ledger writes inline again, so a caller may flush.
        """
        if self._publisher is None:
            return 0
        completed = self._publisher.completed()
        self._publisher = None
        applied = 0
        for tag, *rest in completed:
            key = tuple(rest)
            if tag != RECEIPT_JOB or key not in self._records \
                    or key not in self._staged:
                continue
            self._journal(self._staged.pop(key), self._records.pop(key))
            applied += 1
        self._staged.clear()
        self._records.clear()
        return applied

    def drain(self) -> int:
        """Wait for every staged anchor, journal it, and refuse a remainder."""
        if self._publisher is None:
            return 0
        applied = self._apply(self._publisher.drain())
        if self._staged:
            raise RuntimeError(
                "publication drained with anchors still staged: "
                + ", ".join(f"{q} {f}" for q, f in sorted(self._staged)))
        return applied

    def _apply(self, keys) -> int:
        applied = 0
        for tag, *rest in keys:
            if tag in (FILES_JOB, CHECKPOINT_JOB):
                # Both exist to be ordered, not to be reported: the files a
                # receipt reads and the journal write a receipt precedes.
                continue
            if tag != RECEIPT_JOB:
                raise RuntimeError(f"publication reported an unknown job {tag!r}")
            key = tuple(rest)
            if key not in self._records or key not in self._staged:
                raise RuntimeError(
                    f"publication reported a receipt nothing staged: {key!r}")
            self._journal(self._staged.pop(key), self._records.pop(key))
            applied += 1
        return applied


def _adopt_seed_checkpoint(manifest_path, wire_dir_arg, *, targets, wire_dir,
                           adopt, admits, identity_sha256, expected_identity,
                           validate_state=None) -> dict:
    """Offer another campaign's stored anchors to this run's row gates.

    A whole-scope campaign already priced rows this run would price again.  Its
    journal cannot be resumed as a journal -- ``prepare_journal`` binds the run
    identity, and a sharded run's identity is narrower by construction, while
    ``prismaquant_source_sha256`` moves with any change to this package -- so
    the rows are offered one at a time to the same gates a resume uses.  Those
    gates are content checks and strictly stronger than the journal's: the
    producer's ``encoding_input_identity`` is recomputed from THIS run's
    weight, menu, Hessian applicability and static scale, and Tessera's
    ``verify_cached_unit`` re-reads the blob and re-validates the wire against
    it.  A row that does not describe this run's bytes is refused by name.

    Stored ``dloss`` is inherited only after the seed's calibration, currency,
    static-scale policy and actual per-unit scoring tensors match this run.
    Producer input identity binds encoded bytes; it does not bind the X rows
    used to measure decoded-weight error. The manifest and unit envelope are
    authenticated through the existing checkpoint digest and loader.

    Returns the record stamped into provenance: which manifest, which identity
    it was written under, and which units were adopted.
    """
    from .cost_stage_checkpoint import unit_path, _load_unit, canonical_json_sha256

    manifest = Path(manifest_path)
    parts = manifest.with_name(manifest.name + ".parts")
    if not parts.is_dir():
        raise RuntimeError(
            f"--seed-checkpoint {manifest}: no unit shards at {parts}")
    seed_wire = (Path(wire_dir_arg) if wire_dir_arg
                 else manifest.parent / "cache" / "wire")
    try:
        seed_manifest = json.loads(manifest.read_text())
        seed_identity = seed_manifest.get("identity_sha256")
        seed_inputs = seed_manifest.get("identity")
        if (not isinstance(seed_inputs, dict) or
                canonical_json_sha256(seed_inputs, where='seed checkpoint identity') != seed_identity):
            raise ValueError('seed checkpoint identity digest differs')
    except Exception as exc:
        raise RuntimeError(
            f"--seed-checkpoint {manifest}: unreadable manifest: {exc}") from exc
    for field in ('currency', 'calibration', 'input_global_scale_policy'):
        if (field not in seed_inputs or field not in expected_identity or
                seed_inputs[field] != expected_identity[field]):
            raise RuntimeError(f'seed checkpoint scoring identity mismatch at {field}')
    adopted: list[str] = []
    for name in targets:
        path = unit_path(parts, name)
        if not path.is_file():
            continue
        seed_unit = seed_inputs.get('units', {}).get(name, {})
        current_unit = expected_identity.get('units', {}).get(name, {})
        for field in ('scoring_rows', 'input_global_scale'):
            if (field not in seed_unit or field not in current_unit or
                    seed_unit[field] != current_unit[field]):
                raise RuntimeError(f'seed checkpoint scoring identity mismatch at units.{name}.{field}')
        state = _load_unit(path, stage='Tessera campaign', qname=name,
                           identity_sha256=seed_identity)
        if validate_state is not None:
            validate_state(name, state)
        # Only the rows this run's menu admits get their bytes linked in.  An
        # unservable row's blob must not appear in this run's wire directory:
        # that directory is what the export intake reads, and a wire nothing
        # priced has no business being offered to it.
        for fmt, record in state.get("wire_records", {}).items():
            if not admits(name, fmt):
                continue
            campaign_wire.link_seed_wire(seed_wire, wire_dir, record.get("file"),
                            qname=name, format_name=fmt)
        adopt(name, state, where=f"seed checkpoint {manifest}")
        adopted.append(name)
    print(f"[campaign] adopted verified anchors for {len(adopted)} units from "
          f"{manifest}", flush=True)
    return {
        "manifest": str(manifest),
        "wire_dir": str(seed_wire),
        "seed_identity_sha256": seed_identity,
        "run_identity_sha256": identity_sha256,
        "units": sorted(adopted),
    }








def _static_input_scales(max_abs: "Mapping[str, float]", *, profile=None):
    """Per-unit static NVFP4 ``input_global_scale``, fused-sibling unified.

    ``(scales, policy)``.  Everything here is the owned NVFP4 activation
    contract, reused rather than restated: the resolved default policy names
    the formula (``resolve_input_global_scale_policy``), the scalar is the
    F32-rounded value an exported tensor would carry
    (``input_global_scale_from_max_abs``), and fused siblings share one
    conservative calibration maximum (``unify_fused_sibling_max_abs``) --
    vLLM concatenates q/k/v and gate/up and applies ONE activation scale, and
    Tessera's exporter joins its members' scales for the same module, so a
    per-member scale would price an A side no fused module executes.

    A unit whose calibration never saw a row keeps no scale; a W4A4 anchor on
    it then refuses in :func:`_measure_anchor` rather than pricing dynamically.
    """
    from .nvfp4_activation_contract import (
        input_global_scale_from_max_abs, resolve_input_global_scale_policy,
        unify_fused_sibling_max_abs,
    )

    policy = resolve_input_global_scale_policy()
    positive = {name: float(value) for name, value in max_abs.items()
                if float(value) > 0.0}
    unified = unify_fused_sibling_max_abs(
        positive, profile=profile, tolerate_profile_errors=True)
    return {
        name: input_global_scale_from_max_abs(value, policy=policy)
        for name, value in unified.items()
    }, policy


def _calibration_tokens(model_path: str, n: int, seqlen: int, seed: int):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    # Use the canonical repository: newer Hub URI validation rejects the
    # legacy namespace-free alias while resolving the dataset metadata.
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(data["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    # The corpus text travels beside the ids: Tessera's identity triple wants
    # both, and it is right to want both -- two tokenizers over one corpus are
    # two calibrations that a text sha alone would call the same, and one
    # tokenizer over two corpora is two calibrations an id sha would separate
    # only by luck.
    generator = torch.Generator().manual_seed(int(seed))
    out = []
    for _ in range(int(n)):
        start = int(torch.randint(
            0, max(1, ids.shape[0] - seqlen - 1), (1,), generator=generator,
        ).item())
        out.append(ids[start:start + seqlen].unsqueeze(0))
    return out, text


def _campaign_calibration_parameters(args) -> dict[str, object]:
    """Draw metadata, separate from the CLI seed when a saved draw is supplied."""
    parameters: dict[str, object] = {"source": "wikitext-2-raw-v1/train", "split_role": "calibration",
                  "model": str(args.model), "seed": int(args.seed),
                  "nsamples": int(args.nsamples), "seqlen": int(args.seqlen)}
    receipt = getattr(args, "calibration_input_receipt", None)
    if receipt is not None:
        provenance = receipt["provenance"]
        parameters.update(source=provenance["source"], seed=provenance["seed"],
                          calibration_input=receipt)
    return parameters


#: The exit status a run uses when the resolved menu admits nothing.  It is
#: not 1: a menu that admits nothing is a configuration answer, not a crash,
#: and a caller that fans out over hundreds of rows needs to tell the two
#: apart without parsing a traceback.
EXIT_EMPTY_MENU = 2


def contract_source_label() -> str:
    """Which Tessera contract the menu was resolved against, in one phrase.

    A refusal has to say this or it is unactionable: ``attested`` resolves to
    nothing at all unless the dev pin names a commit, and the difference
    between "the pinned reader admits no rung of this shape" and "no contract
    was consulted" is the whole diagnosis.
    """
    import os

    from .tessera_runtime_contract import TESSERA_DEV_PIN_ENV

    pin = str(os.environ.get(TESSERA_DEV_PIN_ENV, "")).strip()
    if pin:
        return f"{TESSERA_DEV_PIN_ENV}={pin}"
    return "the packaged Tessera runtime_contract.json (no dev pin set)"


def report_empty_menus(menus: "Mapping[str, Sequence]", *, mode: str) -> list[str]:
    """The units with no admitted rung, printed with the mode and the contract.

    Printed rather than raised, because one unit whose shape admits nothing is
    a fact about that unit and the rest of the run is still a real answer --
    it is stamped ``no_admitted_rung`` so a merge cannot mistake a refused
    unit for one nobody selected.  Only the *total* emptiness is fatal, and
    that decision is the caller's (:data:`EXIT_EMPTY_MENU`).
    """
    empty = sorted(name for name, menu in menus.items() if not menu)
    if not empty:
        return []
    print(f"[campaign] {len(empty)} of {len(menus)} units admit no rung under "
          f"mode={mode} against {contract_source_label()}:", flush=True)
    for name in empty:
        print(f"[campaign]   menu_sizes[{name}] = 0", flush=True)
    return empty


def expand_menus_for_targets(weights, targets, *, mode, tp_degree,
                             parallel_kind,
                             context_by_unit: "Mapping[str, ServingContext] | None" = None,
                             family_restriction=None, structure_by_unit=None,
                             profile=None,
                             ) -> dict[str, list]:
    """One Tessera menu per distinct shape and explicit serving context.

    ``expand_tessera_menu`` takes nothing but the shape and the run
    configuration and serving context, so units with equal values of those
    inputs get identical lists. Dense and routed expert units may share one
    shape; their structural class comes from owned topology, never shape or
    name. Without a restriction, a missing context remains unbound. An explicit
    family restriction requires exact structure coverage and adds the allowed
    family tuple to the cache key. A profile's generic per-unit identity-input
    requirement is interpreted by the lane's existing menu through the format
    registry and included in that key. An empty intersection stays empty. Units
    repeat shapes ~1500:1 on a production MoE, so expanding per Linear repeats
    the same answer thousands of times; keying by shape and context expands once per
    distinct answer instead.  Exact rather than approximate: same arguments,
    same list.  The lists are shared, not copied -- downstream only iterates
    them -- which is also what makes ``menu_cache_shapes``' retention the
    thing that bounds the work.
    """
    from .tessera_menu import expand_tessera_menu
    from .tessera_formats import get_tessera_family

    restriction = parse_family_restriction(family_restriction)
    if restriction is not None:
        if (not isinstance(structure_by_unit, Mapping) or set(structure_by_unit) != set(targets)
                or any(s not in ("dense", "routed_moe") for s in structure_by_unit.values())):
            raise ValueError("family restriction requires exact authoritative structure coverage")

    by_shape_and_context: dict[tuple, list] = {}
    menus: dict[str, list] = {}
    for name in targets:
        shape = tuple(weights[name].shape)
        context = None if context_by_unit is None else context_by_unit.get(name)
        if restriction is not None:
            structure = structure_by_unit[name]
            if context is not None and context.structure != structure:
                raise ValueError(f"{name}: family restriction structure conflicts with serving context")
        families = _unit_family_names(name, restriction, structure_by_unit)
        unquantized = profile is not None and profile.linear_requires_unquantized_activations(name)
        key = (shape, None if context is None else context.key(), families, unquantized)
        if key not in by_shape_and_context:
            by_shape_and_context[key] = expand_tessera_menu(
                shape, mode=mode, tp_degree=tp_degree,
                parallel_kind=parallel_kind,
                **({"serving_context": context} if context is not None else {}),
                **({"families": tuple(get_tessera_family(n) for n in families)}
                   if families is not None else {}),
                **({"require_unquantized_activations": True} if unquantized else {}),
            )
        menus[name] = by_shape_and_context[key]
    return menus


def anchor_group_rate_grids(groups: Mapping, rates_by_unit: Mapping, *,
                           encode_structure, projected_units,
                           route_cache: dict[tuple, str | None] | None = None) -> tuple[dict, dict]:
    """The existing full-group legal intersection, shared with partitions."""
    if route_cache is None:
        route_cache = {}
    group_rates, route_refused = {}, {}
    for key, members in groups.items():
        per_family = {}
        families = set.intersection(*[
            set(rates_by_unit[m].keys()) for m in members]) if members else set()
        for family in sorted(families):
            shared = set.intersection(*[rates_by_unit[m][family] for m in members])
            refused = _served_route_refusals(
                family, shared, members, encode_structure=encode_structure,
                projected_units=projected_units, route_cache=route_cache)
            if refused:
                route_refused.setdefault(key, {})[family] = refused
                shared = shared - {int(rung) for rung in refused}
            if shared:
                per_family[family] = sorted(shared)
        group_rates[key] = per_family
    return group_rates, route_refused


#: The selection file ``--units`` reads and ``tools/dispatch_tessera_campaign.py``
#: writes.  A selection names **fused anchor groups**, never bare units: anchor
#: placement is a group property (one shared rung grid, the group's worst
#: member drives the split), so a group is the smallest scope whose measured
#: values do not depend on what else the run priced.
from prismaquant.tessera_campaign_selection import (
    UNITS_SCHEMA, UNITS_SCHEMA_V2, UNITS_SCHEMA_V3, EXPERT_PARTITION_SCHEMA,
    selection_priced_units, validate_unit_selection, load_unit_selection,
)

#: The same selection, plus the planner's expert sample.  A ``v1`` file prices
#: every member of every group it names; a ``v2`` file may additionally give a
#: group a ``sampled`` subset (with the ``inclusion_probability`` it was drawn
#: under and the ``audit`` units inside it).  The group's ``members`` list
#: still names the WHOLE group in both, because that list is what
#: :func:`select_anchor_groups` checks against this run's own grouping: the
#: sample says which members are priced, never which members exist.  A run
#: given a v1 file behaves exactly as it did before v2.

#: The calibration census ``--calibration-census`` reads and ``--census-out``
#: writes: the per-unit calibration row counts of the whole priced scope.


def anchor_group_key(name: str, *, profile, expert_members: Mapping) -> str:
    """The fused anchor group ``name`` belongs to.

    A projected expert unit anchors with its whole stack: the producer plans
    ONE rung per stack, so every member must measure the same rungs for the
    allocator's stack-uniform choice to have a priced (and wire-backed) row on
    every member.  A dense unit anchors with its fused siblings, for the reason
    the round loop states: the grid is shared, so a rung added for one sibling
    is measured for all of them anyway.
    """
    member = expert_members.get(name)
    if member is not None:
        return f"s:{member.module_qname}"
    try:
        key = profile.fused_sibling_group(name)
    except Exception:
        key = None
    return f"g:{key}" if key else f"u:{name}"


def resolve_anchor_groups(targets: Sequence[str], *, profile,
                          expert_members: Mapping) -> dict[str, list[str]]:
    """``{group key: sorted members}`` over ``targets``."""
    groups: dict[str, list[str]] = {}
    for name in targets:
        groups.setdefault(
            anchor_group_key(name, profile=profile, expert_members=expert_members),
            []).append(name)
    for key in groups:
        groups[key].sort()
    return groups


#: The size sources a stack draw may be proportional to.  ``probe`` is the
#: per-expert Fisher vector, which is what the estimator's variance argument
#: rests on; ``counts`` is the census's per-expert routed-row count, which is
#: the ONLY per-expert size available when no probe exists -- a routed-token
#: proxy for ``h_trace`` and never a substitute for it (RobTand/prismaquant#495
#: part 1, and the report's section 3.6 bullet on sizes).

#: The design a ``counts``-sized draw declares, so a reader of ``units.json``
#: can tell which vector the inclusion probabilities are proportional to
#: without re-deriving it.


def research_exact_member_scope(selection: Mapping, resolved: Mapping[str, list[str]],
                                member: str) -> dict:
    """Price one named member of a validated whole group, without a stack estimate.

    This is an endpoint measurement only. The full group still comes from the
    census and ``select_anchor_groups`` checks every member before this narrow
    operation; no inclusion probability or representative-stack claim exists.
    """
    if not isinstance(member, str) or not member:
        raise RuntimeError("--research-exact-member requires one nonempty unit name")
    groups = selection["groups"]
    if len(groups) != 1 or any(
            key in groups[0] for key in ("sampled", "audit", "inclusion_probability", "stack_samples")):
        raise RuntimeError("--research-exact-member requires one unsampled whole group")
    group = groups[0]
    if member not in resolved[group["key"]]:
        raise RuntimeError(f"--research-exact-member {member}: outside selected group {group['key']}")
    return {"member": member, "full_group_key": group["key"],
            "full_group_size": len(resolved[group["key"]]),
            "purpose": "research_scalar_endpoint", "allocator_payload": False,
            "stack_estimate": False}


def selected_capture_unit_names(selection, *, args, resolved) -> list[str]:
    """The exact unit scope one full-group ``--units`` fresh capture may take.

    A streamed capture today takes whole anchor groups, so a selection that
    prices a part of its groups refuses. A sampled or audited group prices a
    draw under inclusion probabilities, an expert partition prices a
    deterministic subset of each stack's experts, and an exact member prices
    one unit: each measures exactly what it names, and none is the whole
    group a capture would otherwise have to record. Both the prep (which has
    no model, so it resolves against the census's ``anchor_groups``) and the
    model rows (which resolve against the scope they loaded) call this one
    authority; :func:`select_anchor_groups` holds the selection to the
    resolved grouping member for member before a unit name is derived.
    ``args`` supplies the campaign's model and layer stride, which the
    selection file itself must declare.
    """
    where = f"--units {getattr(args, 'units', None)}"
    if (selection.get("model", args.model) != args.model
            or selection.get("layer_stride", args.layer_stride) != args.layer_stride):
        raise StackSampleError("--units model/layer_stride disagrees with this campaign")
    if getattr(args, "research_exact_member", None) is not None:
        raise RuntimeError(
            f"{where}: --research-exact-member prices one member, not the "
            "whole group a fresh capture records")
    for entry in selection["groups"]:
        if entry.get("sampled") or entry.get("audit") or entry.get("inclusion_probability"):
            raise RuntimeError(
                f"{where}: group {entry['key']!r} prices a sample of its "
                "members under inclusion probabilities; a fresh capture "
                "records whole groups")
        if entry.get("partition") is not None or entry.get("stack_samples"):
            raise RuntimeError(
                f"{where}: group {entry['key']!r} prices an expert partition "
                "of its stack; a fresh capture records whole groups")
    selected_groups = select_anchor_groups(selection, resolved, where=where)
    priced, _audit, _pi = selection_priced_units(selection)
    unit_names = sorted({name for key in selected_groups for name in resolved[key]})
    if priced != set(unit_names):
        raise RuntimeError(
            f"{where}: the selection prices {len(priced)} of the "
            f"{len(unit_names)} units of its groups; a fresh capture takes a "
            "whole group or none of it")
    if not unit_names:
        raise RuntimeError(f"{where}: the selection names no unit")
    return unit_names


def calibration_census(counts: Mapping[str, int], max_abs: Mapping[str, float], *,
                       args, groups: Mapping, dense_targets: Sequence[str],
                       expert_targets: Sequence[str], shapes: Mapping,
                       identity: Mapping, expert_projection=None, model_load_contract=None,
                       attention_implementation=None, capture_runtime=None,
                       pinned_roster=None) -> dict:
    """The whole priced scope, as one run that loaded the model saw it.

    Everything here is scope-wide and selection-independent, and it is here
    precisely so that no shard has to re-derive it and no two shards can
    disagree about it: the per-unit calibration row counts and activation
    maxima, the anchor grouping, the unit shapes a planner sizes rows with, the
    draw's own identity, and the producer's expert projection of every declared
    stack. ``pinned_roster`` (:func:`pinned_roster_block`) records a lifted-pin
    roster; a census without one carries no such key.
    """
    return {
        "schema": CENSUS_SCHEMA,
        **({"model_load_contract": model_load_contract,
            "attention_implementation": attention_implementation,
            "capture_runtime": capture_runtime} if model_load_contract is not None else {}),
        "model": str(args.model),
        "nsamples": int(args.nsamples),
        "seqlen": int(args.seqlen),
        "seed": int(identity.get("seed", args.seed)),
        **({"calibration_input": identity["calibration_input"]}
           if "calibration_input" in identity else {}),
        "layer_stride": int(args.layer_stride),
        # The draw itself, in Tessera's own vocabulary: a census taken on a
        # different corpus or tokenizer is refused before any GPU is spent.
        "text_sha256": str(identity["text_sha256"]),
        "fit_ids_sha256": str(identity["fit_ids_sha256"]),
        "counts": {str(name): int(value) for name, value in sorted(counts.items())},
        "max_abs": {str(name): float(value) for name, value in sorted(max_abs.items())},
        "unit_shapes": {str(name): [int(dim) for dim in shape]
                        for name, shape in sorted(shapes.items())},
        "anchor_groups": {str(key): sorted(members)
                          for key, members in sorted(groups.items())},
        "dense_targets": sorted(dense_targets),
        "expert_targets": sorted(expert_targets),
        "expert_projection": expert_projection,
        **({"pinned_roster": pinned_roster} if pinned_roster is not None else {}),
    }


def load_calibration_census(path, *, args) -> dict:
    """Read a census and refuse one taken on a different draw or scope."""
    census = json.loads(Path(path).read_text())
    if not isinstance(census, dict) or census.get("schema") != CENSUS_SCHEMA:
        raise RuntimeError(
            f"--calibration-census {path}: not a {CENSUS_SCHEMA} census")
    for field, value in (("model", str(args.model)),
                         ("nsamples", int(args.nsamples)),
                         ("seqlen", int(args.seqlen)),
                         ("seed", _campaign_calibration_parameters(args)["seed"]),
                         ("layer_stride", int(args.layer_stride))):
        if census.get(field) != value:
            raise RuntimeError(
                f"--calibration-census {path}: {field} is {census.get(field)!r} "
                f"in the census and {value!r} in this run; the census must be "
                "the same draw over the same scope")
    _require_census_pinned_roster(census, args, path)
    counts = census.get("counts")
    maxima = census.get("max_abs")
    if not isinstance(counts, dict) or not counts:
        raise RuntimeError(f"--calibration-census {path}: names no unit count")
    if not isinstance(maxima, dict) or set(maxima) != set(counts):
        raise RuntimeError(
            f"--calibration-census {path}: its activation maxima do not cover "
            "exactly the units it counted")
    census["counts"] = {str(name): int(value) for name, value in counts.items()}
    census["max_abs"] = {str(name): float(value) for name, value in maxima.items()}
    return census


def require_census_draw(census: Mapping, identity: Mapping, *, where: str) -> None:
    """Refuse a census taken on another draw than this run's.

    ``load_calibration_census`` compares the arguments that *name* a draw;
    this compares the draw itself, which is the thing the identity is of. The
    two are different checks: the same ``--seed 0 --nsamples 32`` over a
    different corpus revision is a different calibration with identical flags.
    """
    if census.get("calibration_input") != identity.get("calibration_input"):
        raise RuntimeError(f"{where}: the census's calibration input receipt differs")
    for field in ("text_sha256", "fit_ids_sha256"):
        if str(census.get(field)) != str(identity[field]):
            raise RuntimeError(
                f"{where}: the census's {field} is {census.get(field)!r} and "
                f"this run's is {identity[field]!r}; they are different draws")


def census_token_counts(census: "Mapping | None", observed: Mapping[str, int]):
    """``(max, min)`` calibration rows over the priced scope, verified.

    Without a census this is the run's own observation, which is what a
    whole-scope campaign measures.  With one it is the **scope's** maximum and
    minimum, and every unit this run actually hooked is checked against the
    census row for it first: the census is a measurement another invocation of
    this same stage published, and a shard that disagrees with it about its own
    units is not measuring the draw the census describes.  That check is what
    makes a sharded campaign's ``fit_tokens`` the whole scope's -- and therefore
    equal to the monolith's -- without any shard asserting a count it did not
    see (principle 14).

    A **derived** census (``mtp_extension``: the GLM MTP layer's census,
    covering units the body never ran over the body's draw) takes the pair
    from the hash-bound base census it names instead, because the capture
    identity it carries is the base draw's. The run's own rows are still
    checked against the derived census's counts.
    """
    if census is None:
        if not observed:
            return 0, 0
        return max(observed.values()), min(observed.values())
    counts = census["counts"]
    base = _derived_census_base(census)
    missing = sorted(set(observed) - set(counts))
    if missing:
        raise RuntimeError(
            "calibration census does not cover units this run priced: "
            + ", ".join(missing))
    disagree = sorted(name for name, value in observed.items()
                      if int(counts[name]) != int(value))
    if disagree:
        raise RuntimeError(
            "calibration census disagrees with this run's observed rows for "
            + ", ".join(f"{name} (census {counts[name]}, observed {observed[name]})"
                        for name in disagree))
    if base is not None:
        return census_token_counts(base, {})
    return max(counts.values()), min(counts.values())


@functools.lru_cache(maxsize=4)
def _bound_base_census(path: str, sha256: str):
    from .glm_mtp_capture import read_bound_json

    return read_bound_json(path, sha256)[0]


def _derived_census_base(census: Mapping):
    """The base census a derived census names, or None for a body census.

    The base is bound by its bytes and must describe the same draw
    (:func:`require_census_draw`'s fields); an extension of another schema is
    refused rather than read as a body census.
    """
    extension = census.get("mtp_extension")
    if extension is None:
        return None
    from .glm_mtp_capture import CENSUS_EXTENSION_SCHEMA

    if extension.get("schema") != CENSUS_EXTENSION_SCHEMA:
        raise RuntimeError(
            f"calibration census carries an unknown extension {extension.get('schema')!r}")
    ref = extension["base_census"]
    base = _bound_base_census(str(ref["path"]), str(ref["sha256"]))
    for field in ("text_sha256", "fit_ids_sha256"):
        if str(base.get(field)) != str(census.get(field)):
            raise RuntimeError(
                f"derived census names a base census of another draw: its {field} is "
                f"{base.get(field)!r} and the derived census's is {census.get(field)!r}")
    return base


def census_max_abs(census: Mapping, observed: Mapping[str, float]) -> dict[str, float]:
    """The scope's calibration maxima, with this run's own units verified.

    Same rule as :func:`census_token_counts`, for the other unconditional
    output of the same hook: the value used is the scope's, and a run whose own
    observation disagrees with the census about a unit it hooked is refusing,
    not adopting.
    """
    maxima = census["max_abs"]
    missing = sorted(set(observed) - set(maxima))
    if missing:
        raise RuntimeError(
            "calibration census has no activation maximum for units this run "
            "priced: " + ", ".join(missing))
    disagree = sorted(name for name, value in observed.items()
                      if float(maxima[name]) != float(value))
    if disagree:
        raise RuntimeError(
            "calibration census disagrees with this run's observed activation "
            "maximum for " + ", ".join(
                f"{name} (census {maxima[name]!r}, observed {observed[name]!r})"
                for name in disagree))
    return {str(name): float(value) for name, value in maxima.items()}


def _campaign_layer_scope(names, layer_stride: int) -> list[str]:
    """The same explicit layer scope for supported and unsupported units."""
    if layer_stride <= 1:
        return list(names)
    from .qnames import DOTTED_LAYER_QNAME

    selected = []
    for name in names:
        match = DOTTED_LAYER_QNAME.search(name)
        if match is None or int(match.group(1)) % layer_stride == 0:
            selected.append(name)
    return selected


#: The census block that records a lifted-pin roster (PQ #1843). Absent from a
#: census taken without ``--allow-pinned``, so a body census is byte-identical.
PINNED_ROSTER_SCHEMA = "prismaquant.tessera_campaign.pinned_roster.v1"


@dataclass(frozen=True)
class CampaignRoster:
    """The Linears a campaign prices, the pins it keeps and the pins it lifts.

    ``dense`` is the priced non-expert roster before the layer stride;
    ``pinned`` the Linears the profile keeps out of it (pins and
    ``probe_linear_exclude_extra`` matches); ``lifted`` the kept-out Linears
    ``--allow-pinned`` put back into ``dense``.
    """

    dense: tuple
    pinned: tuple
    lifted: tuple


def campaign_roster(linear_names, profile, *, allow_pinned=None,
                    pinned_roster_only: bool = False) -> CampaignRoster:
    """The campaign's non-expert roster from its Linear names (PQ #1843).

    Without ``allow_pinned`` this is the roster the campaign always built:
    every Linear except the head, the embeddings, the profile's pins and its
    ``probe_linear_exclude_extra`` matches. ``allow_pinned`` lifts a Linear
    the profile keeps out by either rule when its name contains one of the
    tokens, with the allocator's grammar and ``token in qname`` semantics
    (``fixed_head.parse_allow_pinned`` / ``allow_pinned_lifts_name``), so a
    census and the allocation that consumes it read one spelling. Both rules
    must yield to an explicit token: GLM keeps attention out by pin and by
    exclude-extra alike, because its construction runtime builds attention
    with ``quant_config=None`` (principle 9). Pricing such a unit reports a
    serving gap; export still refuses its unbacked route.
    ``pinned_roster_only`` keeps only the lifted Linears: the roster of a
    scoped campaign over units the body campaign does not price (GLM
    attention). A token that lifts nothing refuses, so a misspelled token
    cannot shrink the roster silently.

    An explicit head token uses the existing fixed-head alias policy to lift
    that one unit (PQ #1949). This declares a research census/capture roster,
    not a head serving route or a joint-cost measurement. Default head
    exclusion, profile pins and independent export admission remain unchanged.
    """
    from .fixed_head import (
        allow_pinned_lifts_lm_head, is_lm_head_name, parse_allow_pinned,
    )

    tokens = parse_allow_pinned(allow_pinned)
    if pinned_roster_only and not tokens:
        raise ValueError("--pinned-roster-only requires --allow-pinned")
    extra = profile.probe_linear_exclude_extra()
    head_tokens = tuple(token for token in tokens
                        if allow_pinned_lifts_lm_head(profile, (token,)))
    dense, pinned, lifted, used = [], [], [], set()
    for name in linear_names:
        if head_tokens and is_lm_head_name(name, profile):
            used.update(head_tokens)
            lifted.append(name)
            dense.append(name)
            continue
        if name.endswith("lm_head") or "embed" in name:
            continue
        if profile.is_pinned_name(name) or (extra and re.search(extra, name)):
            hits = [token for token in tokens if token in name]
            if not hits:
                pinned.append(name)
                continue
            used.update(hits)
            lifted.append(name)
            dense.append(name)
            continue
        if not pinned_roster_only:
            dense.append(name)
    unused = [token for token in tokens if token not in used]
    if unused:
        raise ValueError(f"--allow-pinned tokens lift no profile-pinned Linear: {unused}")
    return CampaignRoster(dense=tuple(dense), pinned=tuple(pinned), lifted=tuple(lifted))


def pinned_roster_block(args, roster: CampaignRoster):
    """The census's record of a lifted-pin roster, or None without a lift."""
    from .fixed_head import parse_allow_pinned

    tokens = parse_allow_pinned(getattr(args, "allow_pinned", None))
    if not tokens:
        return None
    return {"schema": PINNED_ROSTER_SCHEMA, "allow_pinned": list(tokens),
            "pinned_roster_only": bool(getattr(args, "pinned_roster_only", False)),
            "lifted": sorted(roster.lifted)}


def _require_census_pinned_roster(census: Mapping, args, path) -> None:
    """Refuse a census whose lifted-pin roster is not this run's."""
    from .fixed_head import parse_allow_pinned

    block = census.get("pinned_roster")
    tokens = list(parse_allow_pinned(getattr(args, "allow_pinned", None)))
    only = bool(getattr(args, "pinned_roster_only", False))
    if block is None and not tokens:
        return
    if block is not None and block.get("schema") != PINNED_ROSTER_SCHEMA:
        raise RuntimeError(
            f"--calibration-census {path}: unknown pinned roster {block.get('schema')!r}")
    census_tokens = [] if block is None else list(block.get("allow_pinned") or [])
    census_only = False if block is None else bool(block.get("pinned_roster_only"))
    if census_tokens != tokens or census_only != only:
        raise RuntimeError(
            f"--calibration-census {path}: its roster lifts {census_tokens!r} "
            f"(pinned roster only: {census_only}) and this run lifts {tokens!r} "
            f"(pinned roster only: {only}); the census must be the same scope")


def _require_campaign_population(model, profile, layer_stride: int) -> ExpertPopulation:
    """Admit only the packed units the producer bridge can carry; refuse the rest.

    The profile's routed-target discovery owns membership. An arbitrary 3-D
    parameter (for example a convolution) is not an expert by shape alone.
    The bridge covers a packed parameter when the profile declares its split
    into per-expert 2-D projections whose input kind the existing routed
    derivation captures (``gate_up`` / ``down``) and when the producer's
    projection tool is declared and present -- checked here, before an hour
    of calibration, so a campaign that cannot ask the producer refuses by
    name instead of pricing a dense-only table (PrismaQuant #183).
    """
    from .routed_experts import (
        profile_declared_packed_expert_projections,
        profile_declared_routed_expert_targets,
    )
    from .tessera_expert_projection import (
        ExpertProjectionError, declared_stacks_from_members, producer_plan_tool,
    )

    parameters = dict(model.named_parameters())
    packed = {
        name: tuple(parameters[name].shape)
        for name in profile_declared_routed_expert_targets(model, profile)
        if name in parameters and parameters[name].ndim == 3
    }
    in_scope = _campaign_layer_scope(packed, layer_stride)
    omitted = {name: packed[name] for name in packed if name not in set(in_scope)}
    if not in_scope:
        return ExpertPopulation(members=(), declared={}, packed_in_scope={},
                                omitted_outside_layer_stride=omitted)
    members = [
        member for member in profile_declared_packed_expert_projections(model, profile)
        if member.packed_qname in set(in_scope)
    ]
    covered = {member.packed_qname for member in members}
    # The same input-kind roster ``capture_activation_statistics`` derives from; a
    # packed parameter the derivation cannot feed is one the bridge does not
    # cover, and it is named here rather than discovered mid-capture.
    supported_kinds = {"gate_up_proj", "down_proj"}
    uncovered = sorted(
        name for name in in_scope
        if name not in covered or name.rsplit(".", 1)[-1] not in supported_kinds)
    if uncovered:
        raise RuntimeError(
            "Tessera campaign cannot price the packed expert population: "
            + ", ".join(f"{name} {packed[name]}" for name in uncovered)
            + ". The producer bridge covers only profile-declared gate_up/down "
            "splits into per-expert 2-D projections; refusing an incomplete "
            "dense-only cost payload (PrismaQuant #183).")
    try:
        producer_plan_tool()
    except ExpertProjectionError as exc:
        raise RuntimeError(
            "Tessera campaign cannot ask the producer for its expert projection, "
            f"so the packed population {sorted(in_scope)} cannot be priced: {exc}. "
            "Refusing an incomplete dense-only cost payload (PrismaQuant #183).") from exc
    return ExpertPopulation(
        members=tuple(members), declared=declared_stacks_from_members(members),
        packed_in_scope={name: packed[name] for name in in_scope},
        omitted_outside_layer_stride=omitted)


def _project_expert_population(population: ExpertPopulation, *, weights, menus,
                               model_path, cache_dir: Path, measured=None,
                               projection=None, resource_check=None,
                               release_source_pages=False, source_authentication=None,
                               check_units=True) -> tuple[dict, dict]:
    """Ask the producer to project every in-scope stack; bind it; check the bytes.

    Each request covers the whole campaign (the producer hashes the checkpoint
    to identify its source); family retries never request one stack at a time.
    The answer is bound exactly to the
    profile-declared units -- no name outside the profile's declaration, no
    2-D slice PrismaQuant chose -- and each unit's source tensor is read from
    the shard the producer hashed and compared byte-for-byte with the live
    view this run prices.  Returns ``(carried_block, {qname: unit})``; any
    disagreement refuses by name (PrismaQuant #183).

    ``projection`` supplies a producer answer already obtained for this scope
    (the census's), so a sharded campaign asks the producer once instead of
    once per row and every shard carries the identical block.  ``measured``
    narrows only what is byte-checked and priced here; the carried block always
    covers every declared stack, because that is what the allocation rebinds.

    ``check_units=False`` returns the same records without reading a source
    byte, for the stream head, which has no snapshot view of these units: its
    readers read each unit's source tensor (``_read_projected_unit``) and
    price exactly that (PQ #1654). The records are the ones the serial check
    returns when it passes.
    """
    from .tessera_expert_projection import (
        ExpertProjectionError, bind_expert_projection, carried_projection,
        producer_plan_tool, request_expert_projection, source_unit_weight,
        stack_plan_request,
    )
    from .tessera_formats import parse_tessera_format_name
    import torch

    # The producer's plan asks for a nominal rung per stack; the unit records
    # it returns do not depend on it -- the producer checks only that the
    # family has an expert route on its build.  The menu is ordered by rate,
    # so its first rung's family is arbitrary with respect to the route, and
    # asking on it refuses a stack whose OTHER families are routable.  Ask the
    # producer family by family in menu order and keep the first it accepts;
    # every refusal is carried, because "this build has no expert route for
    # this family" is a measured fact the payload should record rather than
    # lose in a traceback (PrismaQuant #280).
    if projection is not None:
        # A census already asked the producer for exactly this scope. Bind its
        # answer here anyway: binding is the check, and it is the check this
        # run needs -- the carried block is only usable if it covers every
        # declared stack with the declared geometry.
        carried = dict(projection)
        if carried.get("schema") is None or "producer" not in carried:
            raise RuntimeError(
                "the census carries no producer expert projection to reuse "
                "(PrismaQuant #183).")
        try:
            bound = bind_expert_projection(carried["producer"],
                                           declared=population.declared)
        except ExpertProjectionError as exc:
            raise RuntimeError(
                "the census's producer expert projection does not bind to this "
                f"run's declared population: {exc} (PrismaQuant #183).") from exc
        if not check_units:
            return carried, _measured_projected_units(bound, measured)
        return carried, _checked_projected_units(
            bound, weights=weights, model_path=model_path,
            source=carried["producer"]["source"],
            measured=measured, resource_check=resource_check,
            release_source_pages=release_source_pages,
            **({'source_authentication': source_authentication} if source_authentication is not None else {}))

    ladders: dict[str, list[tuple[str, int]]] = {}
    for stack, units in sorted(population.declared.items()):
        first = sorted(units)[0]
        menu = list(menus.get(first) or [])
        if not menu:
            raise RuntimeError(
                f"Tessera campaign has no menu for projected expert unit {first} "
                f"(stack {stack}); refusing to price a stack the allocator could "
                "not choose a rung for (PrismaQuant #183).")
        ladder: list[tuple[str, int]] = []
        seen: set[str] = set()
        for entry in menu:
            parsed = parse_tessera_format_name(entry.format_name)
            if parsed is None:
                continue
            family, rung = parsed
            grid = family.payload_grid().name
            if grid in seen:
                continue
            seen.add(grid)
            ladder.append((grid, int(rung)))
        if not ladder:
            raise RuntimeError(
                f"Tessera campaign menu for projected expert unit {first} names no "
                "Tessera rung the producer could plan (PrismaQuant #183).")
        ladders[stack] = ladder
    out_path = Path(cache_dir) / "expert_projection.json"
    try:
        tool = producer_plan_tool()
    except ExpertProjectionError as exc:
        raise RuntimeError(
            "Tessera campaign cannot ask the producer for its expert projection; "
            f"refusing to price it: {exc} (PrismaQuant #183).") from exc
    # Keep every previously attempted mixed nominal plan. Also ask at each
    # common family by NAME: different stack menus can place E4M3 at different
    # indices, so moving all ladders together can miss the one routed family
    # even when every stack offers it (#295). At most two plans per distinct
    # family count, never the Cartesian product of stack assignments and never
    # one checkpoint-hashing producer call per stack.
    plans = [{stack: ladder[min(index, len(ladder) - 1)]
              for stack, ladder in ladders.items()}
             for index in range(max(map(len, ladders.values())))]
    by_family = {stack: dict(ladder) for stack, ladder in ladders.items()}
    common_families = set.intersection(*(set(rows) for rows in by_family.values()))
    for grid in next(iter(by_family.values())):
        if grid in common_families:
            asked = {stack: (grid, rows[grid]) for stack, rows in by_family.items()}
            if asked not in plans:
                plans.append(asked)
    attempts: list[dict] = []
    stacks: dict[str, tuple[str, int]] = {}
    answer = bound = None
    for asked in plans:
        stacks = asked
        try:
            answer = request_expert_projection(model_path, stacks, out_path=out_path)
            bound = bind_expert_projection(answer, declared=population.declared)
        except ExpertProjectionError as exc:
            attempts.append({"request": stack_plan_request(stacks), "refused": str(exc)})
            answer = bound = None
            continue
        attempts.append({"request": stack_plan_request(stacks), "refused": None})
        break
    if bound is None or answer is None:
        raise RuntimeError(
            "Tessera campaign cannot bind the producer's expert projection to the "
            "profile-declared population on the attempted nominal family plans; refusing to "
            "price it: "
            + " || ".join(a["refused"] for a in attempts)
            + " (PrismaQuant #183).")
    carried = carried_projection(answer, bound, request=stack_plan_request(stacks),
                                 tool=str(tool))
    carried["plan_attempts"] = attempts
    if not check_units:
        return carried, _measured_projected_units(bound, measured)
    return carried, _checked_projected_units(
        bound, weights=weights, model_path=model_path,
        source=answer["source"], measured=measured,
        resource_check=resource_check, release_source_pages=release_source_pages,
        **({'source_authentication': source_authentication} if source_authentication is not None else {}))


def _measured_projected_units(bound, measured=None) -> dict[str, dict]:
    """The bound unit records this run prices, in the order the check reads them.

    One definition of *which* units are priced from the producer's source
    tensors, for both heads: the serial byte check (``_checked_projected_units``)
    and the stream head's readers (``RowStream(load_unit=...)``, PQ #1654).
    """
    return {name: unit for _stack, units in sorted(bound.items())
            for name, unit in sorted(units.items())
            if measured is None or name in measured}


#: What a refused projection byte check says, for one unit or for many.
PROJECTED_BYTES_REFUSAL = (
    "Tessera campaign's live expert view disagrees byte-for-byte with the "
    "producer's source tensor for {units}; the exporter would encode bytes this "
    "table did not price. Refusing (PrismaQuant #183).")


def _check_projected_unit(name, unit, *, live, model_path, source,
                          release_source_pages=False, source_authentication=None):
    """Read one unit's source tensor and compare it byte-for-byte with ``live``.

    Returns ``None`` when the two are equal, else the mismatch description the
    refusal names. The bytes read are the shard the producer hashed, through
    the owner the snapshot read it with. With ``release_source_pages`` the
    unit's own consumed span is advised away once its bytes are read. Nothing
    here reserves memory: the serial caller reserves around it.
    """
    check = _start_projected_unit_check(
        name, unit, live=live, model_path=model_path, source=source,
        release_source_pages=release_source_pages,
        source_authentication=source_authentication)
    return _settle_projected_unit_checks([check])[0]


class _UnitCheck:
    """One unit's comparison: decided on the host, or a device flag to read later."""

    __slots__ = ("description", "differs", "flag", "pinned")

    def __init__(self, description, *, differs=None, flag=None, pinned=None):
        self.description = description
        self.differs = differs
        self.flag = flag
        self.pinned = pinned


def _device_comparable(dtype) -> bool:
    """Whether ``!=`` on the device is the element comparison ``torch.equal`` makes.

    Floating and integer dtypes compare elementwise in both, NaN unequal and
    signed zeros equal. Any other dtype keeps the host comparison.
    """
    import torch
    return dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64,
                     torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                     torch.bool)


def _device_differs(live, staged):
    """A 0-d device flag: does any element of ``live`` differ from ``staged``."""
    import torch
    return torch.ne(live, staged).any()


def _host_differs(live, weight) -> bool:
    import torch
    return not torch.equal(live.cpu(), weight)


def _start_projected_unit_check(name, unit, *, live, model_path, source,
                                release_source_pages=False, source_authentication=None):
    """Read one unit's source tensor and start its comparison with ``live``.

    A unit whose dtype or shape differs is decided at once. A device-resident
    ``live`` of a comparable dtype is compared on its device: the source bytes
    are staged through pinned host memory, copied without blocking, and the
    verdict stays a device flag until ``_settle_projected_unit_checks`` reads
    every flag with one sync (PQ #1931's pattern; PQ #1935). Anything else is
    compared on the host, as before. The unit's pages are advised away once
    its bytes are staged: the copy no longer reads them.
    """
    import torch

    live = live.detach()
    if live.device.type == "cuda" and _device_comparable(live.dtype):
        check = _prepare_device_projected_check(name, unit,
            live_shape=tuple(live.shape), live_dtype=live.dtype,
            model_path=model_path, source=source,
            release_source_pages=release_source_pages,
            source_authentication=source_authentication)
        if check.pinned is not None:
            _launch_prepared_projected_check(check, live)
            check.pinned = None
        return check
    weight, release = _read_projected_unit(
        name, unit, model_path=model_path, source=source,
        release_source_pages=release_source_pages,
        source_authentication=source_authentication)
    live = live.detach()
    description = (f"{name} (live {tuple(live.shape)} {live.dtype} vs source "
                   f"{unit['source_tensor']} {tuple(weight.shape)} {weight.dtype})")
    if live.dtype != weight.dtype or live.shape != weight.shape:
        return _UnitCheck(description, differs=True)
    if live.device.type != "cuda" or not _device_comparable(live.dtype):
        differs = _host_differs(live, weight)
        if not differs:
            del weight
            release()
        return _UnitCheck(description, differs=differs)


def _prepare_device_projected_check(name, unit, *, live_shape, live_dtype,
                                    model_path, source, release_source_pages,
                                    source_authentication, cancelled=lambda:False):
    """CPU preparation only; the caller owns authenticated whole-file residency."""
    import torch
    from concurrent.futures import CancelledError
    weight = pinned = release = None
    try:
        if cancelled():
            raise CancelledError('projected preparation cancelled before source read')
        weight, release = _read_projected_unit(name, unit, model_path=model_path,
            source=source, release_source_pages=release_source_pages,
            source_authentication=source_authentication)
        description = (f"{name} (live {tuple(live_shape)} {live_dtype} vs source "
            f"{unit['source_tensor']} {tuple(weight.shape)} {weight.dtype})")
        if live_dtype != weight.dtype or tuple(live_shape) != tuple(weight.shape):
            return _UnitCheck(description, differs=True)
        if cancelled():
            raise CancelledError('projected preparation cancelled before private copy')
        pinned = torch.empty_like(weight, pin_memory=True)
        pinned.copy_(weight)
        weight = None
        if cancelled():
            raise CancelledError('projected preparation cancelled after private copy')
        return _UnitCheck(description, pinned=pinned)
    finally:
        # Failed futures must not retain native source aliases in their traceback.
        weight = pinned = None
        if release is not None:
            release()


def _launch_prepared_projected_check(check, live):
    """The ordered coordinator alone enqueues CUDA work."""
    staged = check.pinned.to(live.device, non_blocking=True)
    check.flag = _device_differs(live.detach(), staged)
    return staged


def _projected_cuda_reservation(weights):
    """Bound native CUDA reserved growth without assuming cache or stream reuse.

    The guard reads segments, not tensor payloads. Charge a fresh segment for
    EVERY allocation in the pass; a freed mask/staging block can remain cached.
    Native sizing is from PyTorch c10/core/AllocatorConfig.h and
    cuda/CUDACachingAllocator.cpp (round_size/get_allocation_size): 512-byte
    blocks, 2-MiB small segments, 20-MiB medium segments, 2-MiB large rounding.
    Nondefault allocators/settings need their own reviewed residency bound.
    """
    import torch

    eligible = [live for live in weights
                if live.device.type == 'cuda' and _device_comparable(live.dtype)]
    if not eligible:
        return 0
    refusal = ('parallel projected preparation requires the default native CUDA '
               'allocator residency model')
    if torch.cuda.get_allocator_backend() != 'native':
        raise RuntimeError(refusal)
    settings = torch.cuda.memory._snapshot().get('allocator_settings', {})
    # parseArgs("") resets the displayed string, not all effective settings.
    # Qualified 2.11 snapshot serialization casts SIZE_MAX to signed int64 (-1)
    # and emits the 16 1-MiB..64-GiB rounding intervals as power-of-two keys.
    intervals = (64 * 1024**3 // 1024**2).bit_length() - 1
    defaults = {'PYTORCH_CUDA_ALLOC_CONF': '', 'expandable_segments': False,
                'max_split_size': -1, 'garbage_collection_threshold': 0.0,
                'roundup_power2_divisions': {str(1 << i): 0 for i in range(intervals)}}
    if not isinstance(settings, dict):
        raise RuntimeError(refusal)
    for field, default in defaults.items():
        value = settings.get(field)
        if type(value) is not type(default) or value != default:
            raise RuntimeError(f'{refusal}: unpriced effective {field}')
    # Dict equality alone would accept False in place of the native integer 0.
    if any(type(value) is not int for value in settings['roundup_power2_divisions'].values()):
        raise RuntimeError(f'{refusal}: unpriced effective roundup_power2_divisions')
    # The snapshot omits these sticky sizes. Read them from the executing
    # libC10 through its public getters; never infer them from configuration text.
    from .kernels import cuda_allocator_state
    try:
        sizing = cuda_allocator_state.sizing()
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(f'{refusal}: effective sizing accessor unavailable') from error
    if (type(sizing) is not tuple or len(sizing) != 2
            or any(type(value) is not int for value in sizing)
            or sizing != (20 * 1024**2, 20 * 1024**2)):
        raise RuntimeError(f'{refusal}: unpriced effective large-segment/nonsplit sizing')

    def segment(size):
        if not size:
            return 0
        size = ((size + 511) // 512) * 512
        if size <= 1024**2:
            return 2 * 1024**2
        if size < 10 * 1024**2:
            return 20 * 1024**2
        return ((size + 2 * 1024**2 - 1) // (2 * 1024**2)) * (2 * 1024**2)

    total = 0
    counts = {}
    for live in eligible:
        elements = live.numel()
        # A contiguous bool mask's full reduction has one output. Reduce.cuh
        # bounds its global partials by the input count and uses one int
        # semaphore. Split 32-bit iterators may allocate scratch repeatedly;
        # charge each possible slice separately, plus its scalar accumulator.
        slice_count = max(1, (elements + (2**31 - 2)) // (2**31 - 1))
        slices = 1 << (slice_count - 1).bit_length()
        total += segment(elements * live.element_size()) + segment(elements)
        total += slices * (segment(min(elements, 2**31 - 1)) + segment(4))
        total += 2 * segment(1)  # verdict plus possible reduction accumulator
        counts[live.device] = counts.get(live.device, 0) + 1
    # One stacked bool allocation per device, beside all original verdicts.
    return total + sum(segment(count) for count in counts.values())


@contextmanager
def _parallel_projected_checks(units, *, weights, model_path, source,
                               resource_check, source_authentication,
                               release_source_pages, preparation_max_bytes, cancel):
    """Four finite credits on the existing read pool; no resident weight cache."""
    from collections import deque
    from concurrent.futures import CancelledError, TimeoutError, wait
    import threading
    import torch
    from . import layer_streaming
    from .memory_management import reserve_allocation

    if (type(preparation_max_bytes) is not int or preparation_max_bytes <= 0
            or not callable(resource_check) or source_authentication is None):
        raise RuntimeError('parallel projected preparation requires finite bytes, guard and source owner')
    if layer_streaming.layer_read_threads() != 2:
        raise RuntimeError('parallel projected preparation requires the shared two-thread read pool')
    rows = list(units.items())
    sizes = [weights[n].numel()*weights[n].element_size() for n,_u in rows]
    if any(size > preparation_max_bytes for size in sizes):
        raise RuntimeError('one projected unit exceeds the private preparation byte cap')
    pin_bound = min(preparation_max_bytes, sum(sorted(sizes, reverse=True)[:4]))
    gpu_bound = _projected_cuda_reservation([weights[n] for n, _u in rows])
    reserve_allocation(resource_check, 'before_parallel_projected_preparation',
                       cpu_bytes=pin_bound, device_bytes=gpu_bound)
    pool = layer_streaming._layer_read_pool(2, allow_resize=False)
    pending, inflight, futures, checks = deque(), deque(), [], []
    stop = threading.Event()
    devices = set()
    index = held_bytes = 0
    succeeded = False
    def cancelled():
        return stop.is_set() or (cancel is not None and cancel.is_set())
    def require_running():
        if cancelled():
            raise CancelledError('projected preparation cancelled')
    def admit():
        nonlocal index, held_bytes
        while index < len(rows) and len(pending)+len(inflight) < 4:
            name, unit = rows[index]
            live, size = weights[name], sizes[index]
            eligible = live.device.type == 'cuda' and _device_comparable(live.dtype)
            # Serial fallbacks never coexist with pending private staging.
            if not eligible and (pending or inflight):
                break
            if held_bytes + size > preparation_max_bytes:
                break
            resource_check(f'before_source_projection_check:{name}')
            if eligible:
                future = pool.submit(_prepare_device_projected_check, name, unit,
                    live_shape=tuple(live.shape), live_dtype=live.dtype,
                    model_path=model_path, source=source,
                    release_source_pages=release_source_pages,
                    source_authentication=source_authentication, cancelled=cancelled)
                futures.append(future)
            else:
                future = None
            pending.append((name, unit, future, size))
            held_bytes += size
            index += 1
            if not eligible:
                break
    def reap():
        nonlocal held_bytes
        while inflight and inflight[0][2].query():
            check, staged, event, size = inflight.popleft()
            check.pinned = None
            held_bytes -= size
    try:
        while index < len(rows) or pending:
            require_running()
            reap()
            admit()
            if not pending:
                # Event polling grants no progress and never reads a verdict.
                stop.wait(.001)
                continue
            name, unit, future, size = pending[0]
            if future is None:
                check = _start_projected_unit_check(name, unit, live=weights[name],
                    model_path=model_path, source=source,
                    release_source_pages=release_source_pages,
                    source_authentication=source_authentication)
            else:
                while True:
                    require_running()
                    # A launch completed during this wait frees its credit
                    # here, not at the next loop top: the readers must not
                    # idle while the ordered head stages.
                    reap()
                    admit()
                    try:
                        check = future.result(timeout=.05)
                        break
                    except TimeoutError:
                        continue
                require_running()
            if check.pinned is not None:
                live = weights[name]
                # Register the device BEFORE a partially failing H2D/compare.
                devices.add(live.device)
                staged = _launch_prepared_projected_check(check, live)
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(live.device))
                inflight.append((check, staged, event, size))
                staged = None
            else:
                held_bytes -= size
            pending.popleft()
            checks.append(check)
            resource_check(f'after_source_projection_check:{name}')
        yield checks
        succeeded = True
    finally:
        stop.set()
        for future in futures:
            future.cancel()
        # CPU source aliases may outlive a failed/abandoned foreground future.
        wait(futures)
        if not succeeded:
            for device in devices:
                torch.cuda.synchronize(device)
        else:
            for check, staged, event, size in inflight:
                if not event.query():
                    event.synchronize()
        # Only a proven completion fence permits dropping these owners. If a
        # fence fails, the traceback retains the lists and their bounded data.
        for future in futures:
            if not future.cancelled() and future.exception() is None:
                future.result().pinned = None
        for check, staged, event, size in inflight:
            check.pinned = None
        pending.clear(); inflight.clear(); futures.clear()


def _settle_projected_unit_checks(checks) -> list:
    """Each check's mismatch description, or ``None``: one sync per device."""
    import torch

    by_device: dict = {}
    for index, check in enumerate(checks):
        if check.differs is None:
            by_device.setdefault(check.flag.device, []).append(index)
    for indices in by_device.values():
        flags = torch.stack([checks[i].flag for i in indices]).tolist()
        for i, differs in zip(indices, flags):
            checks[i].differs = bool(differs)
            checks[i].flag = None
    return [check.description if check.differs else None for check in checks]


def _read_projected_unit(name, unit, *, model_path, source, release_source_pages=False,
                         source_authentication=None):
    """The producer's source tensor for one projected unit, and its page release.

    Returns ``(weight, release)``: ``weight`` is ``source_unit_weight``'s own
    tensor -- the bytes the exporter re-reads -- and ``release()`` advises the
    unit's consumed span away when ``release_source_pages`` is set (else it
    does nothing). The serial byte check compares this tensor with a snapshot
    view; the stream head prices it directly on a reader thread, so there is
    no second view to compare (PQ #1654).
    """
    from .tessera_expert_projection import ExpertProjectionError, source_unit_weight

    try:
        if release_source_pages:
            # The declared shard, stat and advice alike, as the layer gather
            # does (layer_streaming ``source_stats``): under a residency map
            # the read goes through the same staged opener, the declared file
            # supplies the header and any span no staged range covers, and
            # advising a payload span the stage served drops nothing (PQ #1529).
            path = Path(model_path)/source['tensors'][unit['source_tensor']]
            source_stat = (path.stat() if source_authentication is None
                           else source_authentication.file_stat(path))
        weight = source_unit_weight(model_path, source, unit,
            **({'source_authentication': source_authentication} if source_authentication is not None else {}))
    except ExpertProjectionError as exc:
        raise RuntimeError(
            f"Tessera campaign cannot read the producer's source tensor for "
            f"{name}: {exc} (PrismaQuant #183).") from exc

    def release():
        if release_source_pages:
            from .layer_streaming import _advise_consumed_safetensors_pages
            _advise_consumed_safetensors_pages(
                str(path) if source_authentication is None
                else source_authentication.descriptor_path(path),
                [unit['source_tensor']], source_stat)
    return weight, release


def _host_device():
    import torch
    return torch.device("cpu")


#: What the stream head records beside the snapshot's own preparation record
#: for the projected units it left to the row stream's readers (PQ #1654).
STREAMED_SOURCE_POLICY = "row-stream-producer-source-v1"


def _streamed_source_weights(runner, *, profile, dense_targets, expert_targets, resources,
                             snapshot_policy, resource_check=None):
    """The stream head's weights: dense snapshots, projected-unit placeholders.

    A projected expert unit gets a ``meta`` tensor with the shape and dtype a
    snapshot copy would carry (``selected_weight_specs``); its reader reads
    the producer's source tensor in its place (``_read_projected_unit``), so
    no layer is installed for it here. Dense targets, if any, are snapshotted
    as before, restricted to their own source keys, which must be admitted
    keys. Returns ``(weights, preparation)`` in the snapshot's own record
    shape, with the streamed units named (PQ #1654).
    """
    import torch

    specs = runner.selected_weight_specs([*dense_targets, *expert_targets])
    required = sum(spec[2] for spec in specs.values())
    if required > resources['selected_source_weight_bytes']:
        raise RuntimeError("selected source weights exceed their resident byte budget")
    snapshot_only = snapshot_policy == 'selected-tensors-v1'
    weights, preparation = {}, None
    if dense_targets:
        dense_keys = None
        if snapshot_only:
            from .layer_streaming import selected_weight_source_keys
            dense_keys = selected_weight_source_keys(
                dense_targets, profile, runner.context.weight_ckpt)
            if not set(dense_keys) <= set(resources['source_tensor_keys']):
                raise RuntimeError('snapshot source dependencies differ from the admitted plan')
        weights, preparation = runner.snapshot_selected_weights(
            dense_targets, max_resident_bytes=resources['selected_source_weight_bytes'],
            **({'expected_source_keys': dense_keys} if snapshot_only else {}),
            resource_check=resource_check, host=True)
    for name in expert_targets:
        shape, dtype, _nbytes = specs[name]
        weights[name] = torch.empty(shape, dtype=dtype, device='meta')
    record = dict(schema="prismaquant.selected_source_weights.v1",
        units=sorted(weights), layers=[] if preparation is None else preparation['layers'],
        resident_bytes=required, source_forward_count=0, packed_parent_storage_retained=False,
        **({'source_snapshot_policy': 'selected-tensors-v1',
            'source_tensor_keys': [] if preparation is None else preparation['source_tensor_keys'],
            'nonbody_materialized': False} if snapshot_only else {}),
        streamed_source_policy=STREAMED_SOURCE_POLICY,
        streamed_source_units=sorted(expert_targets))
    return weights, record


def _require_streamed_projection(streamed, projected_units, source, admitted_keys):
    """Every streamed unit has a producer record whose tensor the plan admitted.

    A placeholder whose unit the producer did not project could never be read,
    and a source tensor outside the admitted keys would be a read the row's
    memory and residency plan never priced (PQ #1654).
    """
    missing = sorted(set(streamed) - set(projected_units))
    if missing:
        raise RuntimeError("streamed source units have no producer projection: "
                           + ", ".join(missing[:4]))
    tensors = {projected_units[name]['source_tensor'] for name in streamed}
    unrostered = sorted(tensors - set(source['tensors']))
    if unrostered:
        raise RuntimeError("streamed source tensors are not in the producer's roster: "
                           + ", ".join(unrostered[:4]))
    if admitted_keys is not None:
        unadmitted = sorted(tensors - set(admitted_keys))
        if unadmitted:
            raise RuntimeError("streamed source tensors are not admitted source keys: "
                               + ", ".join(unadmitted[:4]))


def _checked_projected_units(bound, *, weights, model_path, source,
                             measured=None, resource_check=None,
                             release_source_pages=False, source_authentication=None,
                             parallel_preparation=False, preparation_max_bytes=None,
                             cancel=None) -> dict[str, dict]:
    """The producer's unit records for the units this run prices, bytes checked.

    Each unit's source tensor is read from the shard the producer hashed and
    compared byte-for-byte with the live view this run prices, so the exporter
    cannot encode bytes this table did not price (PrismaQuant #183).  Only the
    ``measured`` units are read: a shard cannot check a tensor it never loaded,
    and claiming it had would be the assertion the check exists to replace.

    This is the serial pass the load-all head and the census run. The stream
    head snapshots no projected unit, so it has nothing to compare: its readers
    price the source tensor itself (``_read_projected_unit``, PQ #1654).
    Device-resident views are compared on their device and their verdicts read
    with one sync for the whole pass (PQ #1935).
    """
    units = _measured_projected_units(bound, measured)
    if parallel_preparation:
        with _parallel_projected_checks(units, weights=weights,
                model_path=model_path, source=source, resource_check=resource_check,
                source_authentication=source_authentication,
                release_source_pages=release_source_pages,
                preparation_max_bytes=preparation_max_bytes, cancel=cancel) as checks:
            outcomes = _settle_projected_unit_checks(checks)
    else:
        checks = []
        for name, unit in units.items():
            if resource_check is not None:
                resource_check(f'before_source_projection_check:{name}')
            checks.append(_start_projected_unit_check(
                name, unit, live=weights[name], model_path=model_path, source=source,
                release_source_pages=release_source_pages,
                source_authentication=source_authentication))
            if resource_check is not None:
                resource_check(f'after_source_projection_check:{name}')
        outcomes = _settle_projected_unit_checks(checks)
    projected: dict[str, dict] = {}
    mismatched: list[str] = []
    for (name, unit), mismatch in zip(units.items(), outcomes):
        if mismatch is None:
            projected[name] = unit
        else:
            mismatched.append(mismatch)
    if mismatched:
        raise RuntimeError(PROJECTED_BYTES_REFUSAL.format(units=", ".join(mismatched)))
    return projected


def _format_executes_static_activation_contract(format_name: str) -> bool:
    """Does this rung's route execute a STATIC activation contract?

    The one derivation (#205, #221): the route names the registry row whose
    contract it executes (``activation_source_format``) and the ROW owns the
    answer (``FormatSpec.static_activation_contract``) -- never a compare of
    that row's NAME against ``"NVFP4"``.  It is
    ``tessera_formats.route_static_activation_contract`` off the same route
    ``synthesize_tessera_spec`` stamps onto the rung's spec, so the rung this
    refuses to resume is exactly the rung ``_measure_anchor`` prices.

    The row is read live, every time: it is a dictionary lookup, and the
    registry is what a test (or a lane) replaces when a second row gains a
    contract.  What is memoised is the pure part -- canonical name to serving
    route -- sized by the format key space: the closed-roster bind asks it
    once per format per unit, 1,793 routes for each of an 864-unit row's
    holders, measured at ~1.1 s per unit on the CPU worker when each ask
    synthesized a whole spec, a GPU-idle startup phase of a quarter hour per
    row.
    """
    from . import format_registry as fr
    from .tessera_lane import is_tessera_format_name

    canonical = fr.canonical_format_name(format_name)
    row = fr.REGISTRY.get(canonical)
    if row is not None:
        return row.static_activation_contract is not None
    if not is_tessera_format_name(canonical):
        fr.get_format(canonical)  # raises the registry's KeyError
    from .tessera_formats import route_static_activation_contract

    return route_static_activation_contract(_tessera_route_memo()(canonical)) is not None


def _tessera_route(canonical: str):
    from .tessera_formats import (
        parse_tessera_format_name, tessera_serving_route, tessera_wire_recipe,
    )

    family, rung = parse_tessera_format_name(canonical)
    return tessera_serving_route(family, tessera_wire_recipe(family, rung), rung)


@functools.lru_cache(maxsize=1)
def _tessera_route_memo():
    # Built on first use so this module keeps importing without the Tessera
    # package; the memo itself is sized by the format key space on its first
    # call, the way every wire-recipe memo is.
    from .tessera_formats import lazily_sized_cache, recipe_cache_bound
    return lazily_sized_cache(recipe_cache_bound)(_tessera_route)


def _require_resumable_anchor(anchor: CampaignAnchor, static_scales, *,
                              check_scoring_metadata: bool = True, profile=None) -> None:
    """Refuse a resumed anchor priced under a different activation contract.

    This is the per-row half of the resume identity rule, shared by pre-link
    seed admission and :func:`_checkpoint_anchor_identity`. The run-level
    half, :func:`campaign_checkpoint_identity`, binds this run's static scales
    and policy into the journal. Resume merges checkpoint rows into
    this run's table, and the table's rows must be one currency.  A W4A4
    anchor with no ``input_global_scale`` was measured under the pre-#194
    dynamic FP32-scale quantiser; one with a *different* scale was measured
    on a different calibration; a dynamic-route row carrying a scale was
    stamped by no producer this campaign has.  None is a price of this run's
    served A side, and merging one silently is the exact mixed-table failure
    the Hessian identity guard exists to catch on its own axis.
    """
    from . import format_registry as fr

    spec = fr.get_format(anchor.format_name)
    if spec.static_activation_contract is None:
        if anchor.input_global_scale is not None:
            raise ActivationScaleContractError(
                f"checkpoint anchor {anchor.qname} {anchor.format_name} "
                f"carries input_global_scale={anchor.input_global_scale!r} "
                "but its route keeps the serving format's dynamic activation "
                "quantiser: no producer of this campaign stamps a static scale "
                "on that route, so the row is not one of this run's prices.")
    elif anchor.input_global_scale is None:
        raise ActivationScaleContractError(
            f"checkpoint anchor {anchor.qname} {anchor.format_name} carries "
            "no input_global_scale: it was priced under the pre-served-"
            "contract dynamic activation quantiser and cannot be merged into "
            "a served-contract table. Delete the checkpoint (or pass a fresh "
            "--checkpoint) to re-measure these anchors."
        )
    elif (static_scales.get(anchor.qname) is None or
          float(anchor.input_global_scale) != float(static_scales[anchor.qname])):
        expected = static_scales.get(anchor.qname)
        raise ActivationScaleContractError(
            f"checkpoint anchor {anchor.qname} {anchor.format_name} was "
            f"priced at input_global_scale={anchor.input_global_scale!r} but "
            f"this run's calibration yields {expected!r}. A resumed anchor "
            "must have been scored under this run's own static scales, or "
            "the table mixes two activation calibrations under one identity."
        )
    # The closed-roster identity template is built before scoring from input
    # descriptors. It has no measured contract/observation to validate. Only
    # that caller opts out; actual published and reused rows use the default.
    if check_scoring_metadata is False:
        return
    # The same declaration _finish_anchor stamps on a fresh measured row.
    # Producer wire/input integrity does not authenticate scoring metadata.
    # Direct consumers use the same contract as the measured row.
    direct_contract = _direct_consumer_activation_contract(anchor.qname, profile=profile)
    expected_contract = direct_contract or str(spec.act_dtype_name or "a16")
    if anchor.activation_contract != expected_contract:
        raise ActivationScaleContractError(
            f"checkpoint anchor {anchor.qname} {anchor.format_name} activation contract "
            f"{anchor.activation_contract!r} differs from the expected {expected_contract!r}")
    # A direct consumer never quantizes its input rows.
    allows_quantized = direct_contract is None and spec.act_quant_changes_input
    if (type(anchor.activation_quantized) is not bool or
            (anchor.activation_quantized and not allows_quantized)):
        raise ActivationScaleContractError(
            f"checkpoint anchor {anchor.qname} {anchor.format_name} activation observation "
            f"{anchor.activation_quantized!r} is incompatible with {expected_contract!r}")


def _save_hessian_capture_with_page_release(payload, path, *, resource_check=None):
    """Use Torch's path writer unchanged, advising each stable tensor prefix.

    A file-like torch.save target changes archive record names. Keep the same
    filename writer and serializer as torch.save, and interpose only after its
    synchronous storage writes. No storage, pickle or archive byte is rewritten.
    """
    import torch
    import torch.serialization as serialization
    from .perturbed_x_cache import release_activation_cache_file_pages

    class RecordBoundary:
        def __init__(self, writer):
            self.writer = writer
            self.stable_records = 0

        def __getattr__(self, name):
            return getattr(self.writer, name)

        def write_record(self, name, *args, **kwargs):
            result = self.writer.write_record(name, *args, **kwargs)
            if name.startswith('data/'):
                # The serializer is paused: the visible file extent cannot
                # change while the existing helper checks identity and fsyncs.
                self.stable_records += 1
                expected = path.stat()
                release_activation_cache_file_pages(path, expected_stat=expected)
                if resource_check is not None:
                    resource_check('after_hessian_tensor_record:'+name)
            return result

    def holds_tensor(value):
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, dict):
            return any(holds_tensor(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(holds_tensor(item) for item in value)
        return False

    with serialization._open_zipfile_writer(os.fspath(path)) as writer:
        boundary = RecordBoundary(writer)
        serialization._save(payload, boundary, serialization.pickle,
                            serialization.DEFAULT_PROTOCOL, False)
    if holds_tensor(payload) and boundary.stable_records == 0:
        # Fail closed: a torch archive layout that files tensor bytes under
        # another prefix would otherwise turn this writer into a plain
        # torch.save with the whole sidecar left resident and no signal.
        raise RuntimeError(
            'hessian capture writer saw no stable tensor record (data/*) for a '
            'payload holding tensors; the torch archive layout changed and the '
            'page release would be silently skipped (RobTand/prismaquant#396)')


def memory_admission_detail(*, plan_bytes, cap_bytes, baseline_bytes=None):
    """Every term of a cgroup admission predicate, as numbers.

    The predicate is ``plan > cap - baseline``: a phase plan states deltas, the
    cap is absolute, and the baseline is the floor the row measured for itself.
    A refusal that prints only its verdict leaves the cause to be recovered by
    unpickling a completed row's ``cost.pkl``, which is what three rows of the
    GLM extension cost on 2026-09-12 (RobTand/prismaquant#522).

    ``slack_bytes`` is what the plan had left; negative is the shortfall.
    ``baseline_bytes`` is absent, not zero, where no reading has been taken:
    zero would read as a floor that was measured and found empty.
    """
    detail = dict(plan_bytes=int(plan_bytes), cap_bytes=int(cap_bytes))
    if baseline_bytes is not None:
        detail['baseline_bytes'] = int(baseline_bytes)
    detail['slack_bytes'] = (int(cap_bytes) - int(baseline_bytes or 0)
                             - int(plan_bytes))
    return detail


def write_export_inputs(cache_dir: Path, *, hessians, hessian_rows,
                        hessian_identity, static_scales, static_scale_policy,
                        release_file_pages=False, resource_check=None,
                        hessian_reference=None, hessian_identities=None):
    """Write the exporter's ``--hessian`` and ``--input-scales`` inputs.

    ``(hessian_capture_path | None, input_scales_path | None,
    capture_sha256 | None)``.  The allocation an allocator builds on this
    campaign's table is priced under exactly these Hessians and these static
    activation scales, so the export leg must be handed them back or the
    artifact built is not the artifact priced (RobTand/prismaquant#193).
    Legacy files use the shapes Tessera's exporter consumes:

    * ``hessian_capture.pt`` -- ``{"H": {unit: XᵀX}, "counts", "provenance"}``,
      what ``ActivationSource.from_capture`` loads; the H tensors are the
      campaign's own un-normalised accumulators, the identity is the same
      triple stamped on every cost row, and ``hessian_role: "fit"`` marks it
      as bytes-shaping (Tessera refuses a held-out capture there).  The
      returned ``capture_sha256`` is the content digest of exactly this
      payload (``tessera_export_lane.hessian_capture_sha256``, Tessera's own
      seal rule); every cost row carries it and the export gate binds the
      allocation to the payload by it (RobTand/prismaquant#204).  A JSON
      sidecar ``<capture>.provenance.json`` repeats the identity and carries
      the same digest, so it can only ever describe the payload beside it:
      both files are staged and renamed, the old sidecar is removed before
      the new payload lands, and the sidecar lands last -- at no point does a
      sidecar sit beside a payload it does not seal.
    * ``input_scales.safetensors`` -- one ``<unit>.input_global_scale`` F32
      scalar per unit, the exporter's stock-NVFP4 spelling, valued exactly as
      the W4A4 costs were scored.

    Opt-in ``hessian_reference`` writes only a canonical reference JSON with
    the same H content seal and explicit load bounds. The producer verifies
    actual H bytes on consumption; metadata intake does not verify untouched H.
    ``hessian_identities`` (``{unit: tensor_identity}``) are per-unit H
    receipts the campaign's identity hold already sealed from these same
    resident tensors; the descriptor takes them for the units it covers and
    digests only the rest, so the commitments are not a second full pass.

    ``hessians=None`` is the deliberate weights-only campaign: no capture is
    written, matching the ``supplied=false`` stamp the rows carry.
    """
    import torch

    from .tessera_export_lane import (
        HESSIAN_CAPTURE_SHA256_SCHEMA, hessian_capture_sha256,
    )

    hessian_capture_path = None
    capture_sha256 = None
    if hessian_reference is not None:
        from . import tessera_calibration_cache as capture_store
        if hessians is None or set(hessian_reference) != {'canonical_capture', 'census_path', 'load_policy'}:
            raise RuntimeError('Hessian reference requires resident H and its canonical capture/census/load policy')
        descriptor = capture_store.canonical_hessian_reference_descriptor(
            hessians=hessians, counts=hessian_rows,
            provenance={**dict(hessian_identity), 'hessian_role':'fit'},
            **({} if hessian_identities is None else dict(identities=hessian_identities)),
            **hessian_reference)
        hessian_capture_path = cache_dir/'hessian_capture.references.json'
        capture_sha256 = capture_store.write_hessian_reference(hessian_capture_path, descriptor)
        if resource_check is not None:
            resource_check('after_selected_export_reference_write')
        print(f'[campaign] wrote {hessian_capture_path} '
              f'({len(descriptor["hessians"])} H commitments; no copied H payloads)', flush=True)
    elif hessians is not None:
        hessian_capture_path = cache_dir / "hessian_capture.pt"
        capture_provenance = {**dict(hessian_identity), "hessian_role": "fit"}
        saved_hessians = {name: h for name, h in hessians.items()
                          if h is not None}
        capture_sha256 = hessian_capture_sha256(saved_hessians,
                                                capture_provenance)
        sidecar = hessian_capture_path.with_name(
            hessian_capture_path.name + ".provenance.json")
        tmp_capture = hessian_capture_path.with_suffix(".pt.tmp")
        tmp_sidecar = sidecar.with_suffix(".json.tmp")
        payload = {"H": saved_hessians, "counts": dict(hessian_rows),
                   "provenance": capture_provenance}
        try:
            if release_file_pages:
                _save_hessian_capture_with_page_release(payload, tmp_capture,
                                                        resource_check=resource_check)
            else:
                torch.save(payload, tmp_capture)
        except BaseException:
            # The previously published capture and sidecar stay untouched;
            # do not leave a half-written .pt.tmp beside them.
            tmp_capture.unlink(missing_ok=True)
            raise
        tmp_sidecar.write_text(DIRECT_ASCII_INDENT2_LAX.text({
            **capture_provenance,
            "capture_sha256": capture_sha256,
            "capture_sha256_schema": HESSIAN_CAPTURE_SHA256_SCHEMA,
        }) + "\n")
        if sidecar.exists():
            sidecar.unlink()
        os.replace(tmp_capture, hessian_capture_path)
        os.replace(tmp_sidecar, sidecar)
        if release_file_pages:
            # ``capture_sha256`` was sealed from the in-memory bytes above;
            # the release helper checks identity, fsyncs and advises from its
            # own descriptor. Reading the archive back here would be a full
            # pass over the Hessian sidecar per row with no consumer.
            from .perturbed_x_cache import release_activation_cache_file_pages
            release_activation_cache_file_pages(
                hessian_capture_path, expected_stat=hessian_capture_path.stat())
        if resource_check is not None:
            resource_check('after_selected_export_input_write')
        print(f"[campaign] wrote {hessian_capture_path} "
              f"({len(saved_hessians)} Hessians, capture_sha256 "
              f"{capture_sha256[:12]})", flush=True)
    input_scales_path = None
    if static_scales:
        from safetensors.torch import save_file

        from .nvfp4_activation_contract import (
            INPUT_GLOBAL_SCALE_POLICY_METADATA_KEY, input_global_scale_tensor,
        )

        input_scales_path = cache_dir / "input_scales.safetensors"
        save_file(
            {f"{name}.input_global_scale": input_global_scale_tensor(value)
             for name, value in static_scales.items()},
            str(input_scales_path),
            metadata={INPUT_GLOBAL_SCALE_POLICY_METADATA_KEY:
                      str(static_scale_policy)},
        )
        print(f"[campaign] wrote {input_scales_path} "
              f"({len(static_scales)} static input scales)", flush=True)
    return hessian_capture_path, input_scales_path, capture_sha256


def _streamed_capture_identity(args, census, tokens, corpus_text, *, attention_implementation,
                               guard=None, release_read_pages=None, source_authentication=None,
                               unit_names=None):
    """The canonical capture identity of this draw over the census's source.

    Through a streamed capture's recording owner (PQ #1896) this is the
    traversal identity: nothing is hashed here, and the capture binds the
    digests of the bytes it read at its seal. A full-scope capture passes no
    ``unit_names`` and takes the unchanged full identity; a selected capture
    passes its explicit full-group unit scope, and the identity carries
    ``unit_scope="selected"`` with exactly those units' shape map.
    """
    from . import tessera_calibration_cache as store
    if (census.get('model_load_contract') or {}).get('schema') != 'prismaquant.streaming_initialization.v1':
        raise RuntimeError('streamed capture requires a census from the qualified streaming source route')
    hi, lo = census_token_counts(census, {})
    calibration = th.calibration_identity(corpus_text, tokens, fit_tokens=hi,
        **_campaign_calibration_parameters(args), fit_tokens_min=lo)
    require_census_draw(census, calibration, where="streamed calibration capture")
    # The recorded witness describes the census. The new traversal's own
    # witness is compared at completion; no from-config model is stamped
    # as an already completed checkpoint load before it runs.
    return store.capture_identity(args.calibration_census, calibration=calibration,
        max_act_rows=args.max_act_rows, model_load_contract=census['model_load_contract'],
        attention_implementation=attention_implementation,
        **({} if unit_names is None else {'unit_names': unit_names}),
        resource_check=None if guard is None else guard.check,
        release_read_pages=guard is not None if release_read_pages is None else release_read_pages,
        source_authentication=source_authentication)


def _require_automatic_capture_recording(args):
    """The campaign's automatic recording decision, before original-source reads.

    Explicit descriptor-owner operations and selected capture reuse retain
    their existing contract. This seam owns automatic monolith and chain
    admission; source digests and stat/lifetime fences are not immutability.
    """
    if args.streaming and args.capture_calibration_out:
        from . import tessera_calibration_cache as store
        store.require_automatic_capture_source_recording()


def run_capture_chain_join(request: CaptureJoinRequest) -> int:
    """Publish a completed chain after the existing automatic-source gate."""
    from . import tessera_calibration_cache as store
    store.require_automatic_capture_source_recording()
    from . import capture_layer_chain as chain
    record = chain.join(request.capture_calibration_out,
                        census_path=request.calibration_census)
    print(f"[campaign] joined streamed calibration capture: {record['manifest']}", flush=True)
    return 0


def run_capture_chain_prep(request: CapturePrepRequest, saved_calibration) -> int:
    """Seal a chain's draw, source identity, scope and layer-boundary plan."""
    from . import tessera_calibration_cache as store
    store.require_automatic_capture_source_recording()
    from . import capture_layer_chain as chain
    census = load_calibration_census(request.calibration_census, args=request)
    # The prep has no model: the selected scope is derived from the census's
    # own anchor groups, through the same helper the model rows confirm with.
    unit_names = None
    if request.units:
        unit_names = selected_capture_unit_names(
            load_unit_selection(request.units), args=request,
            resolved=census["anchor_groups"])
        print(f"[campaign] --units full-group capture takes {len(unit_names)} "
              "units of the census scope", flush=True)
    tokens, corpus_text = (saved_calibration if saved_calibration is not None else
                           _calibration_tokens(request.model, request.nsamples, request.seqlen, request.seed))
    # The prep reads no payload (PQ #1896): it seals the traversal identity,
    # the quanta record the digests of what they read, and the join binds them.
    with store.record_capture_source(request.calibration_census, model=request.model) as source:
        record = chain.prepare(request.capture_calibration_out, census_path=request.calibration_census,
            ranges=request.capture_chain_ranges, n_batches=len(tokens),
            boundary_storage=request.capture_chain_boundary_storage,
            identity=lambda: _streamed_capture_identity(request, census, tokens, corpus_text,
                attention_implementation=request.attention_implementation,
                source_authentication=source, unit_names=unit_names))
    print(f"[campaign] prepped streamed calibration capture chain: {record['path']}", flush=True)
    return 0


def _write_capture_load_execution(args, writer, *, guard, resources):
    """A verified-load capture row's replay and seal receipts, beside its cache."""
    if writer is None or writer.load_execution is None:
        return
    from .cost_stage_checkpoint import atomic_write_bytes
    execution_record = dict(schema='prismaquant.capture_load_run.v1',
        replay=writer.load_execution, seal=writer.seal_load_execution,
        resources=resources, memory_guard=None if guard is None else guard.snapshot())
    atomic_write_bytes(Path(args.cache_dir)/'capture-load-execution.json',
        (DIRECT_ASCII_INDENT2_LAX.text(execution_record)+'\n').encode())


def run_census(request: CensusRequest, *, counts, maxima, groups,
               dense_targets, expert_targets, shapes, identity, expert_projection,
               model_load_contract, attention_implementation, capture_runtime,
               pinned_roster, expected_units) -> int:
    """Publish the counted scope only when every expected unit was observed."""
    payload = calibration_census(
        counts, maxima, args=request, groups=groups,
        dense_targets=dense_targets,
        expert_targets=expert_targets,
        shapes=shapes,
        identity=identity, expert_projection=expert_projection,
        model_load_contract=model_load_contract,
        attention_implementation=attention_implementation,capture_runtime=capture_runtime,
        pinned_roster=pinned_roster)
    if set(payload["counts"]) != set(expected_units):
        raise RuntimeError(
            "the census did not observe every unit in scope: missing "
            + ", ".join(sorted(set(expected_units) - set(payload["counts"]))))
    out = Path(request.census_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(DIRECT_ASCII_INDENT2_LAX.text(payload) + "\n")
    print(f"[campaign] wrote {out}: {len(payload['counts'])} units, rows "
          f"{min(payload['counts'].values())}..{max(payload['counts'].values())}, "
          f"{len(payload['anchor_groups'])} anchor groups", flush=True)
    return 0


def run_streamed_calibration(request: CensusRequest | CaptureRequest, runner, profile, *, mode, population,
                              dense_targets, expert_targets, scope_groups,
                              tokens, corpus_text, census, context_by_unit,
                              attention_implementation, capture_runtime,
                              structure_by_unit=None, pinned_roster=None,
                              capture_guard=None):
    """Wire canonical collection to the existing resident layer traversal.

    A capture reads its source once (PQ #1896): the runner reads through the
    recording owner that hashes each file at its first payload read, and the
    seal binds those digests. ``capture_guard`` is the bounded capture's
    memory guard, created before the runner so the owner's hashes check it.
    """
    import torch
    from . import tessera_calibration_cache as store
    from .routed_experts import refresh_packed_expert_projections
    from .tessera_menu import PARALLEL_NONE

    capture = request if isinstance(request, CaptureRequest) else None
    targets = [*dense_targets, *expert_targets]
    modules = dict(runner.model.named_modules())
    # Meta views are used only for shape/menu/projection planning. Live views
    # are refreshed from the shared loader at each layer's consumption.
    # A selected capture narrows ``targets`` but not ``population.members``:
    # only the selected units' meta views are bound, so shapes, menus, the
    # resource plan and the producer check stay at the selected scope, while
    # the full population still binds the census's producer projection.
    selected_expert_targets = set(expert_targets)
    selected_members = [member for member in population.members
                        if member.qname in selected_expert_targets]
    weights = {name: modules[name].weight for name in dense_targets}
    weights.update({member.qname: member.weight for member in selected_members})
    shapes = {name: list(weight.shape) for name, weight in weights.items()}
    # The capture's unit scope is the whole selection's, before a quantum
    # narrows ``targets`` to its own layer range: the prep sealed exactly
    # this identity, and every quantum must re-derive the same one.
    capture_unit_names = sorted(targets) if (capture is not None and capture.units) else None
    names_by_layer = {}
    for name in targets:
        names_by_layer.setdefault(runner.layer_index_for_qname(name), []).append(name)
    menus = expand_menus_for_targets(weights, targets, mode=mode, tp_degree=request.tp_degree,
        parallel_kind=PARALLEL_NONE, context_by_unit=context_by_unit,
        family_restriction=request.family_restriction,
        structure_by_unit=structure_by_unit, profile=profile)
    report_empty_menus(menus, mode=mode)
    projection = None
    if population.declared:
        projection, _ = _project_expert_population(population, weights=weights,
            menus=menus, model_path=request.model, cache_dir=Path(request.cache_dir),
            measured=set(), projection=None if census is None else census.get('expert_projection'))
    del weights

    capture_policy = 'legacy' if capture is None else capture.streaming_capture_policy
    shared_capture = capture_policy in ('shared-inputs-release-v1', 'shared-inputs-bounded-v1')
    bounded_capture = capture_policy == 'shared-inputs-bounded-v1'
    capture_load_policy = None if capture is None else capture.capture_load_policy
    guard = capture_guard
    if guard is None and bounded_capture and runner.device.type == 'cuda':
        from .memory_management import CaptureMemoryGuard
        guard = CaptureMemoryGuard(runner.device)
    if guard is not None:
        guard.check('before_capture_identity')
    source = runner.context.source_authentication
    # A capture chain quantum (PQ #1885) runs layers [start, stop) of this
    # same traversal; everything else about the capture is unchanged.
    quantum = None
    if capture is not None and isinstance(capture.phase, CaptureQuantum):
        from .capture_layer_chain import ChainQuantum
        quantum = ChainQuantum(capture.capture_calibration_out, capture.phase.layer_range,
                               num_layers=runner.num_layers,
                               source_authentication=runner.context.source_authentication)
    layer_stop = runner.num_layers if quantum is None else quantum.stop
    if quantum is not None:
        targets = [name for name in targets
                   if quantum.start <= runner.layer_index_for_qname(name) < quantum.stop]
    writer = None
    if census is not None:
        # The capture reads its source once (PQ #1896): the runner reads
        # through a recording owner, which hashes each file at its first
        # payload read through the descriptor its tensors are read through.
        # The identity binds no digests yet; the seal (or the chain's join)
        # binds what was read.
        if not getattr(source, 'is_recording', False):
            raise RuntimeError('a streamed capture reads its source through its recording owner')
        identity = _streamed_capture_identity(request, census, tokens, corpus_text,
            attention_implementation=attention_implementation, source_authentication=source,
            unit_names=capture_unit_names)
        if quantum is not None:
            identity = quantum.require_identity(identity, n_batches=len(tokens))
        writer = store.CaptureWriter(capture.capture_calibration_out,
            census_path=capture.calibration_census, identity=identity,
            release_file_pages=bounded_capture,
            resource_check=None if guard is None else guard.check,
            verified_load_policy=capture_load_policy)

    if shared_capture and writer is None:
        raise RuntimeError('shared-inputs-release-v1 requires streamed calibration capture')
    counts, maxima, telemetry = {}, {}, []
    if guard is not None:
        from .autoscale import streamed_calibration_resources
        resources = streamed_calibration_resources(request.model, unit_shapes=shapes,
            counts=census['counts'], nsamples=request.nsamples, seqlen=request.seqlen,
            max_act_rows=capture.max_act_rows, cache_slots=capture.streaming_cache_slots,
            prefetch_workers=capture.streaming_prefetch_workers,
            headroom_gb=capture.streaming_cache_headroom_gb, capture_policy=capture_policy,
            capture_load_policy=capture_load_policy,
            source_recording=getattr(source, 'is_recording', False))
        if resources['memory_bytes'] > guard.cap_bytes:
            # The numbers, not just the verdict: a row that dies here is
            # otherwise diagnosable only by unpickling a completed row's
            # cost.pkl (RobTand/prismaquant#522). This predicate is the
            # capture path's, so it carries no baseline term -- the guard has
            # taken no reading yet.
            raise RuntimeError(
                'capture cgroup budget is smaller than its checked phase plan: '
                + DIRECT_ASCII_SPACED_LAX.text(memory_admission_detail(
                    plan_bytes=resources['memory_bytes'],
                    cap_bytes=guard.cap_bytes)))
        if 'source_retained_page_bytes' in resources:
            # Clean page cache, outside the plan (PQ #1896). Past the slack the
            # kernel reclaims retained pages and their ranges are read again,
            # so this line says whether the source can be read once.
            print(json.dumps({'capture_source_retained_pages': {
                'retained_bytes': resources['source_retained_page_bytes'],
                'slack_bytes': guard.cap_bytes - resources['memory_bytes']}}), flush=True)
        additional = max(sum(value for key, value in phase.items() if key not in
            ('nonbody_source_bytes', 'declared_headroom_bytes'))
            for phase in resources['phases'].values())
        guard.check('before_capture_source_traversal', reserve_bytes=additional)
    runner.context.begin_source_initialization_audit()

    def visit(layer, forward_batch):
        names = names_by_layer.get(layer, [])
        if bounded_capture:
            # No loader allocation can race with growing capture tensors.
            runner.context.settle_prefetched_layers(range(
                layer+1, min(layer_stop, layer+1+runner.prefetch_lookahead)))
        members = [m for m in population.members if m.qname in names]
        live = refresh_packed_expert_projections(members, profile)
        if live:
            _checked_projected_units(projection['stacks'],
                weights={m.qname: m.weight for m in live}, model_path=request.model,
                source=projection['producer']['source'], measured={m.qname for m in live},
                resource_check=None if guard is None else guard.check,
                release_source_pages=guard is not None,
                # Every source read of a capture goes through its recording owner.
                **({} if source is None else {'source_authentication': source}))
        # The source identity check is complete. The collector creates its own
        # live views; this caller must not pin old packed storage through return.
        del live
        completed_source_released = False
        settled_prefetch = None
        def release_completed_source():
            nonlocal completed_source_released, settled_prefetch
            if bounded_capture:
                settled_prefetch = runner.context.settle_prefetched_layers(range(
                    layer+1, min(layer_stop, layer+1+runner.prefetch_lookahead)))
            runner.context.release_completed_layer(layer)
            completed_source_released = True

        expected_groups = None
        if bounded_capture:
            from .routed_experts import declared_shared_capture_groups
            expected_groups = declared_shared_capture_groups({name: shapes[name] for name in names}, profile)
            if guard is not None:
                capture_bytes = sum(4*(shapes[name][1]**2+capture.max_act_rows*shapes[name][1])
                                    for name in expected_groups)
                guard.check('before_capture_tensor_growth', reserve_bytes=capture_bytes)
        def checked_forward(batch):
            guard.check('before_capture_forward')
            value = forward_batch(batch)
            guard.check('after_capture_forward')
            return value

        before = time.perf_counter()
        acts, hessians, rows, amax = calibration_forward.capture_activation_statistics(runner.model, names, tokens,
            capture.max_act_rows if writer is not None else 0, runner.device,
            want_hessian=writer is not None, profile=profile,
            forward_batch=forward_batch if guard is None else checked_forward,
            shared_packed_inputs=shared_capture,
            on_forwards_complete=release_completed_source if shared_capture else None,
            expected_shared_input_groups=expected_groups,
            resource_check=None if guard is None else guard.check)
        collected = time.perf_counter()
        counts.update(rows)
        maxima.update(amax)
        if writer is not None:
            writer.write(acts=acts, hessians=hessians, counts=rows, maxima=amax)
        if quantum is not None:
            # A no-op unless the row declared the progress contract; the
            # layer's units are journalled, so it is durable work.
            from .prismabuild_progress import report
            report("capture", layer - quantum.start + 1, unit="layers")
        flushed = time.perf_counter()
        record = dict(layer=layer, units=len(names), capture_policy=capture_policy,
            completed_source_released=completed_source_released, collect_seconds=collected-before,
            flush_seconds=flushed-collected,
            capture_tensor_bytes=sum(t.numel()*t.element_size()
                for t in [*acts.values(), *hessians.values()] if t is not None))
        if bounded_capture:
            record['settled_prefetch'] = settled_prefetch
            record['physical_memory_guard'] = None if guard is None else guard.snapshot()
        telemetry.append(record)
        del acts, hessians
        if runner.device.type == 'cuda':
            torch.cuda.empty_cache()
        if guard is not None:
            record['released_capture_memory'] = guard.check('after_capture_output_release')
        print(json.dumps({'streamed_calibration_layer': record}), flush=True)

    try:
        if quantum is None:
            runner.visit_layer_batches(tokens, visit)
        else:
            with quantum.owner():
                runner.visit_layer_batches(tokens, visit, start=quantum.frontier(),
                    stop_layer=quantum.stop, boundary_consumer=quantum.boundary_consumer())
                if set(counts) != set(targets) or any(value <= 0 for value in counts.values()):
                    raise RuntimeError('capture chain quantum did not observe every unit of its layers')
                # The census checks hold per unit, so the quanta's union is the monolith's.
                census_token_counts(census, counts)
                census_max_abs(census, maxima)
                fragment = quantum.complete(
                    witness=runner.context.source_selected_initialization_witness(
                        range(quantum.start, quantum.stop)),
                    verified=writer.verify_entries(targets))
    except BaseException:
        if guard is not None:
            from .cost_stage_checkpoint import atomic_write_bytes
            failure = dict(completed_layers=telemetry, physical_memory_guard=guard.snapshot())
            atomic_write_bytes(Path(request.cache_dir)/'capture-memory-refusal.json',
                (DIRECT_ASCII_INDENT2_LAX.text(failure)+'\n').encode())
        raise
    if quantum is not None:
        _write_capture_load_execution(request, writer, guard=guard,
                                      resources=resources if guard is not None else None)
        Path(request.cache_dir, 'streamed-calibration-telemetry.json').write_text(
            DIRECT_ASCII_INDENT2_LAX.text(telemetry)+'\n')
        print(f"[campaign] capture chain layers {quantum.start}:{quantum.stop} complete: "
              f"{fragment}", flush=True)
        return 0
    contract = runner.context.source_initialization_contract()
    if set(counts) != set(targets) or any(value <= 0 for value in counts.values()):
        raise RuntimeError('streamed calibration did not observe every in-scope unit')
    hi, lo = census_token_counts(census, counts)
    calibration = th.calibration_identity(corpus_text, tokens, fit_tokens=hi,
        **_campaign_calibration_parameters(request), fit_tokens_min=lo)
    if writer is not None:
        census_max_abs(census, maxima)
        # The files the forward never read (MTP sidecars, tokenizer assets)
        # are hashed now; every other digest was recorded by its read.
        source_receipt = source.authenticate_complete_source()
        receipt = writer.finish(model_load_contract=contract,
                                source_files=source.recorded_source_files())
        _write_capture_load_execution(request, writer, guard=guard,
                                      resources=resources if guard is not None else None)
        from .cost_stage_checkpoint import atomic_write_bytes
        atomic_write_bytes(Path(request.cache_dir)/'capture-source-authentication.json',
                           indent2_json_file_bytes(source_receipt))
        print(f"[campaign] complete streamed calibration capture: {receipt}", flush=True)
    else:
        payload = calibration_census(counts, maxima, args=request, groups=scope_groups,
            dense_targets=dense_targets, expert_targets=expert_targets, shapes=shapes,
            identity=calibration, expert_projection=projection, model_load_contract=contract,
            attention_implementation=attention_implementation, capture_runtime=capture_runtime,
            pinned_roster=pinned_roster)
        from .cost_stage_checkpoint import atomic_write_bytes
        atomic_write_bytes(Path(request.census_out),
            indent2_json_file_bytes(payload))
        print(f"[campaign] wrote streamed calibration census {request.census_out}", flush=True)
    telemetry_path = Path(request.cache_dir)/'streamed-calibration-telemetry.json'
    telemetry_path.write_text(DIRECT_ASCII_INDENT2_LAX.text(telemetry)+'\n')
    return 0


def _prefetch_selected_capture(args, *, expected_identity, census, names, device,
                               resources, guard=None):
    """Use the existing resident prefetch and publish its separate load receipt."""
    from . import tessera_calibration_cache as store
    from .perturbed_x_cache import normalize_verified_activation_load
    policy = normalize_verified_activation_load(args.capture_load_policy)
    execution = {} if policy is not None else None
    values, capture = store.prefetch_capture(args.calibration_cache,
        expected_identity=expected_identity, census=census, names=names, device=device,
        expected_sha256=args.calibration_cache_sha256,
        resource_check=None if guard is None else guard.check, release_file_pages=True,
        **(dict(verified_load_policy=policy, load_execution=execution) if policy is not None else {}))
    if execution is None:
        return values, capture, None
    return values, capture, _publish_capture_load_execution(
        args.cache_dir, capture=capture, execution=execution, resources=resources, guard=guard)


def _publish_capture_load_execution(cache_dir, *, capture, execution, resources, guard=None):
    """Write one row's capture load execution record; ``{path, sha256}``.

    The one writer for both row heads: the load-all prefetch's folded record,
    and the streaming head's per-entry receipts folded in the same name order
    at finalize (``tessera_row_stream.RowStream.load_execution``).
    """
    import hashlib
    from .cost_stage_checkpoint import atomic_write_bytes
    record = dict(schema='prismaquant.capture_load_run.v1', capture=capture,
        prefetch=execution, resources=resources,
        memory_guard=None if guard is None else guard.snapshot())
    raw = indent2_json_file_bytes(record)
    digest = bytes_sha256hex(raw)
    # A later refused/interrupted resume must not replace execution evidence
    # already referenced by a surviving priced output.
    output = Path(cache_dir)/f'capture-load-execution-{digest}.json'
    atomic_write_bytes(output, raw)
    return dict(path=str(output.resolve()), sha256=digest)


def main(argv: "Sequence[str] | None" = None) -> int:
    from contextlib import ExitStack
    from .tessera_row_stream import RowWaitTelemetry
    with RowWaitTelemetry() as waits, ExitStack() as source_scope:
        result = _main(argv, source_scope=source_scope, waits=waits)
        waits.returncode = result
        return result


@dataclass(frozen=True, slots=True)
class ResolvedCampaignScope:
    """The loaded model's full scope and the exact subset this row prices."""
    population: ExpertPopulation
    roster: CampaignRoster
    pinned_roster: dict | None
    dense_targets: list[str]
    expert_targets: list[str]
    expert_members: dict
    census_dense_targets: list[str]
    census_expert_targets: list[str]
    scope_groups: dict[str, list[str]]
    census: dict | None
    census_only: bool
    selection: dict | None
    stack_samples: dict
    exact_member_scope: dict | None
    partition_menu_targets: list[str] | None
    partition_unit_records: dict
    selected_groups: list[str]
    audit_units: set[str]
    inclusion_probability: dict[str, float]
    context_by_unit: dict | None
    structure_by_unit: dict | None
    encode_structure: dict | None
    acquisition: dict | None


def resolve_campaign_scope(request: CampaignScopeRequest, model, profile, *,
                           mode, device, serving_target, selected_capture_census,
                           capture_unit_names, acquisition) -> ResolvedCampaignScope:
    """Resolve and refuse roster, draw, selection and topology in one order."""
    from .tessera_serving_scope import context_by_unit_from_stats, unit_structure_from_stats

    # The packed expert population the producer bridge covers, or a refusal by
    # name before calibration.  Its members are priced as the producer's
    # projected units below (PrismaQuant #183).
    # A pinned-roster-only campaign prices the lifted Linears and no expert
    # (PQ #1843); its census records that the population was left out.
    population = (ExpertPopulation(members=(), declared={}, packed_in_scope={},
                                   omitted_outside_layer_stride={})
                  if request.pinned_roster_only else
                  _require_campaign_population(model, profile, request.layer_stride))
    # Names only: no module is retained after source teardown.
    roster = campaign_roster(
        profile.campaign_dense_unit_names(model, allow_pinned=request.allow_pinned),
        profile, allow_pinned=request.allow_pinned, pinned_roster_only=request.pinned_roster_only)
    all_dense = list(roster.dense)
    pinned = list(roster.pinned)
    pinned_roster = pinned_roster_block(request, roster)
    if pinned_roster is not None:
        print(f"[campaign] --allow-pinned {pinned_roster['allow_pinned']} lifts "
              f"{len(roster.lifted)} profile-pinned Linears"
              + (" (pinned roster only)" if request.pinned_roster_only else ""), flush=True)
    dense_targets = _campaign_layer_scope(all_dense, request.layer_stride)
    expert_targets = population.qnames
    expert_members = {member.qname: member for member in population.members}
    targets = [*dense_targets, *expert_targets]
    # The whole priced scope, before any selection narrows the encoding work.
    # It is what the census counts, what the population block enumerates, and
    # what a merge checks a set of shards covers.
    census_dense_targets = list(dense_targets)
    census_expert_targets = list(expert_targets)
    census_targets = list(targets)
    scope_groups = resolve_anchor_groups(
        census_targets, profile=profile, expert_members=expert_members)
    print(f"[campaign] {len(dense_targets)} target Linears + {len(expert_targets)} "
          f"projected expert units in {len(population.declared)} stacks, "
          f"{len(scope_groups)} anchor groups, mode={mode}, "
          f"device={device}", flush=True)

    # The census quantum: the whole prologue over the whole scope -- one
    # calibration forward that counts rows and calibrates the static A-side
    # maxima, and one producer projection request -- written out and stopped
    # before a single anchor is encoded.  Every later shard reads it, so no
    # shard re-derives a scope-wide answer and none of them can disagree
    # about one.
    census_only = bool(request.census_out)
    if census_only and request.units:
        raise RuntimeError(
            "--census-out takes the whole scope; it cannot be narrowed by --units")
    if census_only and request.calibration_census:
        raise RuntimeError("--census-out writes a census; it does not read one")

    selection = None
    stack_samples = {}
    exact_member_scope = None
    partition_members = {}
    partition_menu_targets = None
    partition_unit_records = {}
    full_expert_members = expert_members
    census = (selected_capture_census if selected_capture_census is not None else
              (None if not request.calibration_census
               else load_calibration_census(request.calibration_census, args=request)))
    selected_groups: list[str] = sorted(scope_groups)
    audit_units: set = set()
    inclusion_probability: dict = {}
    if request.units:
        selection = load_unit_selection(request.units)
        if (selection.get("model", request.model) != request.model
                or selection.get("layer_stride", request.layer_stride) != request.layer_stride):
            raise StackSampleError("--units model/layer_stride disagrees with this campaign")
        if selection.get("schema") == UNITS_SCHEMA_V3:
            if census is None or request.research_exact_member is not None:
                raise RuntimeError("expert partition requires a census and excludes research exact-member")
            from .tessera_expert_projection import carried_units
            _source, partition_unit_records, partition_stacks = carried_units(census["expert_projection"])
            partition_members = expert_partition_members(
                selection, unit_records=partition_unit_records, stack_of=partition_stacks,
                rate_band=request.rate_band, max_rounds=request.max_rounds,
                seeded=bool(request.seed_checkpoint or request.seed_wire_dir))
            partition_menu_targets = sorted(name for entry in selection["groups"] for name in entry["members"])
            # Existing unit receipts bind the local subset; the scope-wide
            # producer projection binds the full group geometry/source frame.
        stack_samples = selection_stack_samples(selection, profile)
        selected_groups = select_anchor_groups(
            selection, scope_groups, where=f"--units {request.units}")
        priced, audit_units, inclusion_probability = selection_priced_units(
            selection)
        if request.research_exact_member is not None:
            if (mode != "research" or not request.calibration_census or
                    not request.calibration_cache or parse_rate_band(request.rate_band) is None):
                raise RuntimeError("--research-exact-member requires research menu, "
                                   "calibration census/cache and --rate-band")
            exact_member_scope = research_exact_member_scope(
                selection, scope_groups, request.research_exact_member)
            priced = {exact_member_scope["member"]}
            audit_units = set()
            inclusion_probability = {}
        if request.capture_calibration_out:
            # The census-derived scope the pre-model check derived; the run's
            # own grouping must resolve the selection to exactly that set.
            confirmed = selected_capture_unit_names(selection, args=request,
                                                    resolved=scope_groups)
            if confirmed != capture_unit_names:
                raise RuntimeError(
                    f"--units {request.units}: this run's grouping resolves the "
                    f"selection to {confirmed} but its census to "
                    f"{capture_unit_names}; the scope is not exact")
        # The group's membership was already checked whole; what the sample
        # narrows is only which of those members this run encodes. Keeping the
        # two separate is what lets a sampled run stay identity-honest: the
        # checkpoint's ``units`` map holds exactly the priced units, so a
        # different draw is a different identity and cannot silently resume
        # over this one.
        keep = {name for key in selected_groups for name in scope_groups[key]
                if name in priced}
        dense_targets = [name for name in dense_targets if name in keep]
        expert_targets = [name for name in expert_targets if name in keep]
        expert_members = {name: member for name, member in expert_members.items()
                          if name in keep}
        targets = [*dense_targets, *expert_targets]
        sampled_groups = sum(1 for entry in selection["groups"]
                             if entry.get("sampled"))
        print(f"[campaign] --units selects {len(selected_groups)} of "
              f"{len(scope_groups)} anchor groups: {len(dense_targets)} dense + "
              f"{len(expert_targets)} projected expert units"
              + (f"; {sampled_groups} group(s) sampled, "
                 f"{len(audit_units)} audit unit(s)" if sampled_groups else ""),
              flush=True)
        if not targets:
            raise RuntimeError(
                f"--units {request.units}: the selection prices no unit")
    elif request.research_exact_member is not None:
        raise RuntimeError("--research-exact-member requires --units")

    if acquisition is not None:
        acquisition, selected_groups, keep = campaign_acquisition_row_scope(
            acquisition, scope_groups, selected=targets if request.units else None)
        dense_targets = [name for name in dense_targets if name in keep]
        expert_targets = [name for name in expert_targets if name in keep]
        expert_members = {name: member for name, member in expert_members.items() if name in keep}
        targets = [*dense_targets, *expert_targets]

    context_by_unit = None
    structure_by_unit = None
    if serving_target is not None or request.family_restriction is not None:
        from .sensitivity_probe import discover_moe_structure
        routed = discover_moe_structure(model, profile=profile)
        topology = {
            name: dict(zip(("router_path", "expert_id"), routed.get(name, (None, None))))
            for name in dense_targets
        }
        topology_experts = ({name: full_expert_members[name] for name in partition_menu_targets}
                            if partition_menu_targets is not None else expert_members)
        topology_targets = partition_menu_targets or targets
        for name, member in topology_experts.items():
            if name not in topology_targets:
                continue
            # The packed facts the probe would record for this unit: its
            # packed module and expert count, never a shape guess.
            topology[name] = {
                "_packed_experts_module": member.module_qname,
                "num_experts": int(getattr(member.module, member.param_name).shape[0]),
            }
        context_by_unit = context_by_unit_from_stats(serving_target, topology, profile)
        if request.family_restriction is not None:
            if len(set(topology_targets)) != len(topology_targets) or set(topology) != set(topology_targets):
                raise ValueError("family restriction requires unambiguous topology for every target")
            structure_by_unit = {name: unit_structure_from_stats(name, topology[name], profile)
                                 for name in topology_targets}
    # The serving structure each unit's wire is resolved on (#1502): the
    # restriction's authoritative map when there is one, else the serving
    # context's. A run that declares neither prices the research recipe it
    # always priced, and stamps that it did (``encode_structure: null``).
    encode_structure = (dict(structure_by_unit) if structure_by_unit is not None
                        else None if context_by_unit is None
                        else {name: context.structure for name, context in context_by_unit.items()})

    return ResolvedCampaignScope(
        population=population,
        roster=roster,
        pinned_roster=pinned_roster,
        dense_targets=dense_targets,
        expert_targets=expert_targets,
        expert_members=expert_members,
        census_dense_targets=census_dense_targets,
        census_expert_targets=census_expert_targets,
        scope_groups=scope_groups,
        census=census,
        census_only=census_only,
        selection=selection,
        stack_samples=stack_samples,
        exact_member_scope=exact_member_scope,
        partition_menu_targets=partition_menu_targets,
        partition_unit_records=partition_unit_records,
        selected_groups=selected_groups,
        audit_units=audit_units,
        inclusion_probability=inclusion_probability,
        context_by_unit=context_by_unit,
        structure_by_unit=structure_by_unit,
        encode_structure=encode_structure,
        acquisition=acquisition,
    )


@dataclass(frozen=True, slots=True)
class PartitionCampaignMenu:
    full_menus: dict
    encode_structure: dict[str, str]


@dataclass(frozen=True, slots=True)
class ResolvedCampaignMenu:
    """Resolved menus and the exact projection/structure they describe."""
    menus: dict
    context_by_unit: dict | None
    structure_by_unit: dict | None
    encode_structure: dict | None
    no_admitted_rung: list[str]
    expert_projection: dict | None
    projected_units: dict
    partition: PartitionCampaignMenu | None


def resolve_campaign_menu(weights, scope: ResolvedCampaignScope, profile, *,
                          mode, tp_degree, family_restriction, model_path, cache_dir,
                          selected_source, streaming_head, source_authentication,
                          selected_guard) -> ResolvedCampaignMenu:
    """Resolve full-group grids, local menus and producer-bound structure."""
    import torch
    from .tessera_menu import PARALLEL_NONE

    partition_menu_targets = scope.partition_menu_targets
    census = scope.census
    context_by_unit = scope.context_by_unit
    structure_by_unit = scope.structure_by_unit
    encode_structure = scope.encode_structure
    population = scope.population
    expert_targets = scope.expert_targets
    targets = [*scope.dense_targets, *expert_targets]
    if partition_menu_targets is not None:
        # Geometry-only menu inputs: no unpriced W/X/H is loaded or cached.
        menu_weights = {name: torch.empty(census["unit_shapes"][name], device="meta")
                        for name in partition_menu_targets}
        full_partition_menus = expand_menus_for_targets(
            menu_weights, partition_menu_targets, mode=mode, tp_degree=tp_degree,
            parallel_kind=PARALLEL_NONE, context_by_unit=context_by_unit,
            family_restriction=family_restriction, structure_by_unit=structure_by_unit,
            profile=profile)
        menus = {name: full_partition_menus[name] for name in targets}
        partition_encode_structure = {name: "routed_moe" for name in partition_menu_targets}
        # Consumers and identity receipts describe only this row's priced units.
        context_by_unit = (None if context_by_unit is None else
                           {name: context_by_unit[name] for name in targets})
        structure_by_unit = (None if structure_by_unit is None else
                             {name: structure_by_unit[name] for name in targets})
        encode_structure = {name: "routed_moe" for name in targets}
    else:
        menus = expand_menus_for_targets(
            weights, targets, mode=mode, tp_degree=tp_degree,
            parallel_kind=PARALLEL_NONE,
            context_by_unit=context_by_unit,
            family_restriction=family_restriction,
            structure_by_unit=structure_by_unit,
            profile=profile,
        )
    # PrismaQuant #291 (filed here first as #288). A narrowing menu mode --
    # ``attested`` without a dev pin, ``readable`` against a contract that
    # publishes no reader for these shapes -- used to resolve to nothing and
    # let the run finish successfully with ``costs: {}``. A zero-row cost
    # table is not a cheap answer, it is a missing one, and downstream it is
    # indistinguishable from a scope that legitimately holds no unit. So the
    # run refuses, and says which mode and which contract produced the
    # emptiness. This is principle 1's line: the platform reports the gap
    # instead of shipping a table nobody can tell is empty on purpose.
    no_admitted_rung = report_empty_menus(menus, mode=mode)

    # The producer's projection of every in-scope stack, asked for ONCE (it
    # hashes the whole checkpoint), bound exactly to the profile-declared
    # units, and its source bytes checked against the live views this run
    # prices.  What the producer will read at export is what is priced here.
    expert_projection = None
    projected_units: dict[str, dict] = {}
    # The projection is a SCOPE-wide answer, always: the block a shard carries
    # covers every declared stack, because the allocation rebinds the
    # producer's answer against every stack the block names
    # (``carried_units``), and a block trimmed to one shard's stacks would be
    # refused there. What narrows is only the byte-check and the priced units.
    # A shard therefore reads the census's projection rather than asking the
    # producer again -- the request hashes the whole checkpoint, so asking per
    # row would put the campaign's most expensive serial step on every row and
    # let two rows answer it differently.
    if population.declared:
        expert_projection, projected_units = _project_expert_population(
            population, weights=weights, menus=menus,
            model_path=model_path, cache_dir=cache_dir,
            measured=set(expert_targets),
            projection=(None if census is None else census.get("expert_projection")),
            **(dict(resource_check=None if selected_guard is None else selected_guard.check,
                    release_source_pages=True, source_authentication=source_authentication)
               if selected_source else {}),
            # The stream head reads each unit's source tensor on its reader
            # threads and prices it directly, so there is no snapshot view to
            # check against it (PQ #1654).
            check_units=not streaming_head)
        print(f"[campaign] producer projected {len(expert_projection['stacks'])} stacks; "
              f"{len(projected_units)} expert units priced here"
              + ("; their source tensors are read on the row stream's readers"
                 if streaming_head and projected_units else ""), flush=True)
    # A projected unit is a routed stack member whatever the run declares:
    # Tessera's export intake reads a projection as ``routed_moe`` and expects
    # the routed served wire in its receipt (tessera#662), so pricing one on
    # the unstructured spelling would write a receipt the exporter refuses.
    if projected_units:
        declared_dense = sorted(name for name in projected_units
                                if (encode_structure or {}).get(name, "routed_moe") != "routed_moe")
        if declared_dense:
            raise ValueError(f"projected expert units declared non-routed: {declared_dense[:4]}")
        encode_structure = {**(encode_structure or {}),
                            **{name: "routed_moe" for name in projected_units}}

    return ResolvedCampaignMenu(
        menus=menus,
        context_by_unit=context_by_unit,
        structure_by_unit=structure_by_unit,
        encode_structure=encode_structure,
        no_admitted_rung=no_admitted_rung,
        expert_projection=expert_projection,
        projected_units=projected_units,
        partition=(PartitionCampaignMenu(full_partition_menus, partition_encode_structure)
                   if partition_menu_targets is not None else None))


def _main(argv, *, source_scope, waits) -> int:
    from prismaquant.model_runtime import initialize_model_runtime
    initialize_model_runtime()
    import torch

    from . import format_registry as fr
    from .production_weight_cache import ProductionWeightCache
    from .tessera_menu import menu_mode
    from .tessera_rate_surface import leave_one_anchor_out
    from .tessera_publication import PublicationError
    from .tessera_render import (
        HessianContractError, tessera_encoder_hessian_status,
    )
    from .tessera_serving_scope import (
        serving_target_from_args,
        scope_provenance,
    )

    from .tessera_campaign_cli import campaign_argument_parser
    ap = campaign_argument_parser(description=__doc__)
    args = ap.parse_args(argv)
    acquisition = _load_campaign_acquisition(args, ap)
    calibration_options = (args.calibration_input, args.calibration_input_sha256,
                           args.calibration_corpus)
    if any(value is not None for value in calibration_options) and not all(calibration_options):
        ap.error("--calibration-input, --calibration-input-sha256 and --calibration-corpus go together")
    args.calibration_input_receipt = None
    saved_calibration = None
    if args.calibration_input is not None:
        from .calibration_data import load_calibration_corpus, load_calibration_input
        ids, receipt = load_calibration_input(args.calibration_input,
            expected_sha256=args.calibration_input_sha256,
            n_samples=args.nsamples, seqlen=args.seqlen)
        provenance = receipt["provenance"]
        if type(provenance.get("seed")) is not int:
            raise ValueError("exact calibration provenance requires an integer seed")
        if not isinstance(provenance.get("source"), str) or not provenance["source"].strip():
            raise ValueError("exact calibration provenance requires a nonempty source")
        if provenance.get("model") != str(args.model):
            raise ValueError("exact calibration provenance model differs from this campaign")
        corpus_text = load_calibration_corpus(args.calibration_corpus,
            expected_sha256=provenance.get("text_sha256"))
        saved_calibration = (list(ids.split(1, dim=0)), corpus_text)
        args.calibration_input_receipt = receipt
    if args.exhaustive_rate_grid and parse_rate_band(args.rate_band) is None:
        ap.error("--exhaustive-rate-grid requires --rate-band")
    from .perturbed_x_cache import normalize_verified_activation_load
    try:
        args.capture_load_policy = normalize_verified_activation_load(args.capture_load_policy)
    except ValueError as exc:
        ap.error(str(exc))
    if args.streaming_capture_policy != "legacy" and not (args.streaming and args.capture_calibration_out):
        ap.error("--streaming-capture-policy requires --streaming and --capture-calibration-out")
    selected_source = bool(args.streaming and args.units and args.calibration_cache
                           and args.calibration_cache_sha256
                           and not (args.census_out or args.capture_calibration_out))
    if bool(args.source_identity_cache) != bool(args.source_identity_cache_sha256):
        ap.error('--source-identity-cache and --source-identity-cache-sha256 go together')
    if args.source_identity_cache and not selected_source:
        ap.error('--source-identity-cache requires selected streaming capture reuse')
    if args.campaign_identity_bytes and not selected_source:
        ap.error('--campaign-identity-bytes requires selected streaming capture reuse')
    if args.source_snapshot_policy != 'whole-layer-v1' and not selected_source:
        ap.error('--source-snapshot-policy requires selected streaming capture reuse')
    if args.source_scope is not None and args.source_snapshot_policy != 'selected-tensors-v1':
        ap.error('--source-scope requires --source-snapshot-policy selected-tensors-v1')
    if args.pinned_roster_only and not args.allow_pinned:
        ap.error('--pinned-roster-only requires --allow-pinned')
    if args.capture_load_policy is not None and not (selected_source or (
            args.streaming and args.capture_calibration_out and
            args.streaming_capture_policy == 'shared-inputs-bounded-v1')):
        ap.error('--capture-load-policy requires streamed shared-inputs-bounded-v1 capture or hash-bound selected streaming reuse')
    if args.export_hessian_reference_policy is not None:
        if not selected_source:
            ap.error('--export-hessian-reference-policy requires selected reuse of a hash-bound complete capture')
        try:
            from tessera.hessian_capture import normalize_reference_load_policy
            args.export_hessian_reference_policy = normalize_reference_load_policy(args.export_hessian_reference_policy)
        except (ImportError, ValueError) as error:
            ap.error(f'canonical H references need a compatible producer and valid load policy: {error}')
    if args.streaming and not selected_source and (
            not (args.census_out or args.capture_calibration_out) or
            (args.units and not (args.capture_calibration_out or args.calibration_cache))):
        # Streaming prices a whole scope, or an explicit scope that is
        # itself a whole priced object: a --units fresh capture (validated
        # full-group before the model loads) or a --units reuse of a
        # hash-bound complete capture. A census-out run is never narrowed,
        # and unsupported streaming pricing without one of those refuses.
        ap.error("--streaming requires full-scope census/capture, a --units "
                 "full-group fresh capture, or --units with a hash-bound "
                 "complete calibration cache")
    if args.streaming and (args.streaming_cache_slots < 2 or args.streaming_prefetch_workers < 1):
        ap.error("streaming calibration requires at least two cache slots and one prefetch worker")
    if (args.census_out or args.capture_calibration_out or args.calibration_cache) and not args.attention_implementation:
        ap.error("canonical census/capture requires explicit --attention-implementation")
    if args.calibration_cache_sha256 and not args.calibration_cache:
        ap.error("--calibration-cache-sha256 requires --calibration-cache")
    if args.capture_calibration_out or args.calibration_cache:
        if not args.calibration_census or args.census_out or args.hessian != "require":
            ap.error("capture/reuse requires --calibration-census and --hessian require")
        if args.capture_calibration_out and args.calibration_cache:
            ap.error("--capture-calibration-out cannot also reuse; a fresh capture and a reuse read are different rows")
        if args.max_act_rows < 1:
            ap.error("capture/reuse requires positive --max-act-rows")
    if args.capture_chain is not None:
        if not (args.streaming and args.capture_calibration_out):
            ap.error("--capture-chain requires --streaming and --capture-calibration-out")
        if bool(args.capture_layer_range) != (args.capture_chain == "quantum"):
            ap.error("--capture-layer-range is a capture chain quantum's, and every quantum names one")
        if (args.capture_chain_ranges is None or args.capture_chain_boundary_storage is None) != (
                args.capture_chain != "prep"):
            ap.error("--capture-chain-ranges and --capture-chain-boundary-storage are the "
                     "capture chain prep's, and the prep names both")
        from .capture_layer_chain import (CaptureChainRefused, parse_layer_range,
                                          parse_layer_ranges, require_layer_tiling)
        from .streamed_boundary_artifacts import check_boundary_storage
        try:
            if args.capture_layer_range is not None:
                args.capture_layer_range = parse_layer_range(args.capture_layer_range)
            if args.capture_chain_ranges is not None:
                args.capture_chain_ranges = require_layer_tiling(
                    parse_layer_ranges(args.capture_chain_ranges))
                check_boundary_storage(args.capture_chain_boundary_storage)
        except (CaptureChainRefused, ValueError) as exc:
            ap.error(str(exc))
    elif (args.capture_layer_range or args.capture_chain_ranges
          or args.capture_chain_boundary_storage is not None):
        ap.error("--capture-layer-range, --capture-chain-ranges and "
                 "--capture-chain-boundary-storage require --capture-chain")
    if args.anchor_batch_size < 1:
        ap.error("--anchor-batch-size must be positive")
    if args.publication_overlap_bytes < 0:
        ap.error("--publication-overlap-bytes cannot be negative")
    if args.campaign_identity_bytes < 0:
        ap.error("--campaign-identity-bytes cannot be negative")
    if args.campaign_identity_threads is not None and args.campaign_identity_threads < 1:
        ap.error("--campaign-identity-threads must be positive")
    _require_automatic_capture_recording(args)
    if args.capture_chain in ("prep", "join"):
        # Neither runs a forward, so neither loads the model.
        request = capture_chain_bookend_request(args)
        if isinstance(request, CaptureJoinRequest):
            return run_capture_chain_join(request)
        return run_capture_chain_prep(request, saved_calibration)
    if args.anchor_batch_size > 1:
        from .tessera_render import require_tessera_batch_encoder
        require_tessera_batch_encoder()
    serving_target = serving_target_from_args(args)

    mode = menu_mode(args.menu_mode)
    require_exhaustive_rate_research_mode(args.exhaustive_rate_grid, mode)
    hessian_status = tessera_encoder_hessian_status()
    if args.hessian == "require" and not hessian_status["accepted"]:
        # Refuse before the model load, not after an hour of encodes.
        raise HessianContractError(
            "--hessian require: " + str(hessian_status["reason"]) + ". The "
            "encoder's shipping default consumes a per-unit Hessian (LDLQ + "
            "full-H row-scale refit), so a campaign that prices without one "
            "prices bytes that are not the bytes that ship. Re-run with "
            "--hessian off to price weights-only deliberately -- every row "
            "and the payload are stamped hessian.supplied=false -- or pin "
            "prismaquant.tessera_render.TESSERA_HESSIAN_KWARG once the "
            "H-aware encoder branch is merged."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    waits.start(args.out, device)
    if (args.streaming_capture_policy == "shared-inputs-bounded-v1" or selected_source) and device == "cuda":
        from .autoscale import require_bounded_capture_environment
        require_bounded_capture_environment(os.environ)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wire_dir = cache_dir / "wire"
    wire_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        Path(args.out).with_suffix(".anchors.json")
    )

    from .tessera_row_stream import (EXECUTION_FILENAME, EXECUTION_SCHEMA, MEMORY_PLANS,
                                     RECEIPT_FIELDS, ROW_HEAD_LOAD_ALL, ROW_HEAD_STREAM,
                                     checkpoint_present, resolve_identity_threads,
                                     stream_head_dependency)
    # One reader/builder count for the plan and both heads: argv's, else the
    # CPUs PrismaBuild admitted this row with (RobTand/prismaquant#640).
    identity_threads_requested = resolve_identity_threads(args.campaign_identity_threads)
    streaming_head = False
    if selected_source:
        row_head_dependency = stream_head_dependency(
            row_head=args.row_head, selected_source=selected_source,
            capture_load_policy=args.capture_load_policy,
            export_hessian_reference_policy=args.export_hessian_reference_policy,
            max_rounds=args.max_rounds, seed_checkpoint=args.seed_checkpoint)
        streaming_head = row_head_dependency is None
        if streaming_head:
            print(f"[campaign] row head: stream "
                  f"({_identity_threads_for_this_process(identity_threads_requested)} "
                  f"reader threads, window {2 * args.anchor_batch_size} units)", flush=True)
        else:
            # Load-all is a performance defect unless something needs the whole
            # set first, so the row names what did.
            print(f"[campaign] row head: load-all ({row_head_dependency})", flush=True)
            from .cost_stage_checkpoint import atomic_write_bytes
            atomic_write_bytes(cache_dir / EXECUTION_FILENAME, (json.dumps(dict(
                schema=EXECUTION_SCHEMA, row_head=ROW_HEAD_LOAD_ALL,
                dependency=row_head_dependency, memory_plan=MEMORY_PLANS[ROW_HEAD_LOAD_ALL]),
                indent=2, sort_keys=True) + "\n").encode())

    source_authentication = None
    selected_guard = None
    waits.row_head = ROW_HEAD_STREAM if streaming_head else ROW_HEAD_LOAD_ALL
    waits.before_gpu_work()
    if selected_source:
        from . import tessera_calibration_cache as calibration_store
        if device == 'cuda':
            from .memory_management import CaptureMemoryGuard
            selected_guard = CaptureMemoryGuard(device)
        source_authentication = source_scope.enter_context(
            calibration_store.authenticate_selected_capture_source(
                args.calibration_census, args.calibration_cache,
                expected_sha256=args.calibration_cache_sha256, model=args.model,
                max_act_rows=args.max_act_rows, attention_implementation=args.attention_implementation,
                calibration_parameters={key: _campaign_calibration_parameters(args)[key]
                                        for key in ('nsamples', 'seqlen', 'seed')},
                resource_check=None if selected_guard is None else selected_guard.check,
                release_read_pages=True))
        if args.source_identity_cache:
            # Before the runner or any payload read: every shard the snapshot
            # reads then carries the campaign's proof instead of a fresh hash
            # under this GPU reservation (PQ #1497).
            source_authentication.adopt_identity_proof_or_hash(
                args.source_identity_cache, args.source_identity_cache_sha256)

    # A selected fresh capture is held to the census's own anchor groups
    # before the model loads: the selection file's model/layer_stride and the
    # exact group memberships are checked against the census here, so a plan
    # written against another scope refuses before any source is opened. The
    # prep row (no model) makes the same call through the same helper, and the
    # model rows confirm it against the grouping they loaded.
    selected_capture_census = None
    capture_unit_names = None
    if args.capture_calibration_out and args.units:
        if acquisition is not None:
            ap.error("--acquisition-request narrows the run's priced scope; a "
                     "--units fresh capture takes exactly the selection's whole groups")
        selected_capture_census = load_calibration_census(args.calibration_census, args=args)
        capture_unit_names = selected_capture_unit_names(
            load_unit_selection(args.units), args=args,
            resolved=selected_capture_census["anchor_groups"])
        print(f"[campaign] --units full-group capture takes {len(capture_unit_names)} "
              "units of the census scope", flush=True)

    capture_guard = None
    if args.streaming and args.capture_calibration_out:
        # A streamed capture reads its source once (PQ #1896): the runner
        # reads through a recording owner that hashes each file at its first
        # payload read, through the held descriptor the tensors are then read
        # through, and keeps its pages for them. A bounded capture's guard
        # exists before the runner so those hashes check it, and the owner
        # drops each file's pages after its last consumer.
        bounded = args.streaming_capture_policy == 'shared-inputs-bounded-v1'
        if bounded and device == 'cuda':
            from .memory_management import CaptureMemoryGuard
            capture_guard = CaptureMemoryGuard(device)
        page_policy = dict(resource_check=None if capture_guard is None else capture_guard.check,
                           release_read_pages=bounded)
        if args.capture_chain == "quantum":
            # A chain quantum's owner is bound to the prep's seal and stat
            # record; nothing it does not read is hashed (PQ #1885).
            from .capture_layer_chain import authenticate_quantum_source
            source_authentication = source_scope.enter_context(authenticate_quantum_source(
                args.capture_calibration_out, census_path=args.calibration_census,
                model=args.model, **page_policy))
        else:
            from . import tessera_calibration_cache as calibration_store
            source_authentication = source_scope.enter_context(
                calibration_store.record_capture_source(
                    args.calibration_census, model=args.model, **page_policy))

    # The source proof is adopted before this point. The producer's immutable
    # package seal can run under skeleton/tokenizer/calibration preparation
    # instead of serially at the head of run_identity (PQ #1742).
    encoder_seal_ahead = _start_encoder_source_seal_ahead(args, source_scope)

    from .model_profiles import detect_profile
    profile = detect_profile(args.model)
    runner = None
    if source_authentication is not None:
        # Drain any failed preparation before the descriptor owner closes.
        source_scope.callback(lambda: runner.shutdown() if runner is not None else None)
    if args.streaming:
        from .cost_streaming import build_streamed_causal_lm
        runner = build_streamed_causal_lm(args.model, device=torch.device(device), dtype=torch.bfloat16,
            profile=profile, offload_folder=str(cache_dir/'source-offload'),
            max_cache_slots=args.streaming_cache_slots,
            prefetch_workers=args.streaming_prefetch_workers,
            cache_headroom_gb=args.streaming_cache_headroom_gb,
            prefetch_min_available_gb=args.streaming_cache_headroom_gb,
            prefetch_lookahead=args.streaming_cache_slots-1, require_prefetched_residency=True,
            attn_implementation=args.attention_implementation,
            **({'source_snapshot_only': True}
               if args.source_snapshot_policy == 'selected-tensors-v1' else {}),
            **({'source_scope': args.source_scope} if args.source_scope is not None else {}),
            **({'source_authentication': source_authentication} if source_authentication is not None else {}))
        model = runner.model
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map=device,
            **({"attn_implementation": args.attention_implementation}
               if args.attention_implementation is not None else {}),
        )
    model.eval()
    if saved_calibration is not None:
        # The decoder's vocabulary: a multimodal wrapper config (GLM-5.3's Glm5NextConfig) keeps
        # vocab_size on its text sub-config, and get_text_config() returns a text-only config
        # itself (#1913).
        config = model.config
        text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
        vocab_size = getattr(text_config, "vocab_size", None)
        if type(vocab_size) is not int or vocab_size < 1:
            raise ValueError("exact calibration input requires a positive model vocab_size")
        if any(bool((batch >= vocab_size).any()) for batch in saved_calibration[0]):
            raise ValueError("exact calibration token IDs exceed the model vocabulary")
    model_load_contract = None
    attention_implementation = None
    capture_runtime = None
    if args.census_out or args.capture_calibration_out or args.calibration_cache:
        import importlib.metadata
        from prismaquant import pretrained_initialization_contract
        if runner is None:
            model_load_contract = pretrained_initialization_contract(model)
        attention_implementation = model.config._attn_implementation
        if attention_implementation != args.attention_implementation:
            raise RuntimeError("loaded model attention backend differs from explicit request")
        capture_runtime = dict(torch=torch.__version__,cuda=torch.version.cuda,
            transformers=importlib.metadata.version("transformers"))

    scope = resolve_campaign_scope(campaign_scope_request(args), model, profile,
        mode=mode, device=device, serving_target=serving_target,
        selected_capture_census=selected_capture_census,
        capture_unit_names=capture_unit_names, acquisition=acquisition)
    population = scope.population
    roster = scope.roster
    pinned_roster = scope.pinned_roster
    dense_targets = scope.dense_targets
    expert_targets = scope.expert_targets
    expert_members = scope.expert_members
    census_dense_targets = scope.census_dense_targets
    census_expert_targets = scope.census_expert_targets
    scope_groups = scope.scope_groups
    census = scope.census
    census_only = scope.census_only
    selection = scope.selection
    stack_samples = scope.stack_samples
    exact_member_scope = scope.exact_member_scope
    partition_menu_targets = scope.partition_menu_targets
    partition_unit_records = scope.partition_unit_records
    selected_groups = scope.selected_groups
    audit_units = scope.audit_units
    inclusion_probability = scope.inclusion_probability
    context_by_unit = scope.context_by_unit
    structure_by_unit = scope.structure_by_unit
    encode_structure = scope.encode_structure
    acquisition = scope.acquisition
    all_dense = list(roster.dense)
    pinned = list(roster.pinned)
    targets = [*dense_targets, *expert_targets]
    census_targets = [*census_dense_targets, *census_expert_targets]

    if saved_calibration is None:
        tokens, corpus_text = _calibration_tokens(
            args.model, args.nsamples, args.seqlen, args.seed)
    else:
        tokens, corpus_text = saved_calibration
    want_h = args.hessian == "require" and not census_only
    calibration_cache = None
    capture_identity = None
    selected_source_preparation = None
    selected_weights = None
    if census is not None:
        # Validate the whole scope before a selected unit's artifact is read.
        if (set(census["counts"]) != set(census_targets) or
                census["anchor_groups"] != scope_groups):
            raise RuntimeError("calibration census scope differs from the loaded model")
        scope_shapes = {name: list(dict(model.named_modules())[name].weight.shape)
                        for name in census_dense_targets}
        scope_shapes.update({member.qname: list(member.weight.shape)
                             for member in population.members})
        if census["unit_shapes"] != scope_shapes:
            raise RuntimeError("calibration census geometry differs from the loaded model")
    completed_capture_reuse = bool(args.capture_calibration_out and args.capture_chain is None
        and (Path(args.capture_calibration_out) / "capture_manifest.json").is_file())
    if completed_capture_reuse and runner is not None:
        # Reuse expects the recorded full witness; no new traversal is claimed.
        model_load_contract = census["model_load_contract"]
        runner.shutdown()
        runner = None
    if runner is not None and not selected_source and not completed_capture_reuse:
        try:
            return run_streamed_calibration(streamed_calibration_request(args), runner, profile, mode=mode, population=population,
                dense_targets=dense_targets, expert_targets=expert_targets,
                scope_groups=scope_groups, tokens=tokens, corpus_text=corpus_text, census=census,
                context_by_unit=context_by_unit, attention_implementation=attention_implementation,
                capture_runtime=capture_runtime, structure_by_unit=structure_by_unit,
                pinned_roster=pinned_roster, capture_guard=capture_guard)
        finally:
            runner.shutdown()
    if selected_source:
        from .autoscale import selected_anchor_resources
        witness_schema = ('prismaquant.streaming_initialization.v1' if args.source_scope is None
                          else profile.source_scope(args.source_scope, args.model).load_contract_schema)
        if (census.get('model_load_contract') or {}).get('schema') != witness_schema:
            raise RuntimeError('selected source requires the qualified streaming census witness'
                               + ('' if args.source_scope is None else
                                  f' of source scope {args.source_scope!r} ({witness_schema})'))
        # This describes the historical complete capture. This sparse source
        # preparation makes no claim to repeat the full initialization audit.
        model_load_contract = census['model_load_contract']
        selected_resources = selected_anchor_resources(args.model,
            unit_shapes={name: census['unit_shapes'][name] for name in targets},
            counts=census['counts'], max_act_rows=args.max_act_rows,
            cache_slots=args.streaming_cache_slots,
            prefetch_workers=args.streaming_prefetch_workers,
            headroom_gb=args.streaming_cache_headroom_gb,
            anchor_batch_size=args.anchor_batch_size,
            publication_overlap_bytes=args.publication_overlap_bytes,
            campaign_identity_bytes=args.campaign_identity_bytes,
            campaign_identity_threads=identity_threads_requested,
            source_snapshot_policy=args.source_snapshot_policy,
            **({'source_scope': args.source_scope} if args.source_scope is not None else {}),
            **(dict(capture_load_policy=args.capture_load_policy)
               if args.capture_load_policy is not None else {}))
        if device == 'cuda':
            # Read first, then admit. The plan states DELTAS over whatever this
            # process already holds -- interpreter, torch, the CUDA runtime,
            # every page touched so far -- while the cap is an absolute cgroup
            # limit, so admitting a plan against the raw cap compared two
            # different quantities and left the difference to declared headroom
            # (RobTand/prismaquant#390). The guard's first reading is that
            # floor, measured in this row's own process. The guard's own
            # refusal arithmetic needs no change: its readings are already
            # absolute, so current + reserved + reserve_bytes against
            # cap - margin is one unit throughout, and subtracting the baseline
            # there would count the floor twice.
            selected_guard.check('before_selected_capture_identity')
            # The stream head holds a window, not the population, and is
            # admitted against the plan for the head it will actually run --
            # a resumed checkpoint included (PQ #1613).
            plan_key = MEMORY_PLANS[ROW_HEAD_STREAM if streaming_head else ROW_HEAD_LOAD_ALL]
            if (selected_resources[plan_key] >
                    selected_guard.cap_bytes - selected_guard.baseline_bytes()):
                # Every term of the predicate, in the message and on disk.
                # Three rows died here on 2026-09-12 with 8-13 MB of slack and
                # the message named none of it, so the cause had to be
                # recovered by unpickling completed rows' cost.pkl
                # (RobTand/prismaquant#522).
                detail = memory_admission_detail(
                    plan_bytes=selected_resources[plan_key],
                    cap_bytes=selected_guard.cap_bytes,
                    baseline_bytes=selected_guard.baseline_bytes())
                from .cost_stage_checkpoint import atomic_write_bytes
                atomic_write_bytes(
                    cache_dir/'selected-anchor-memory-refusal.json',
                    (json.dumps(dict(detail, plan=plan_key,
                                     memory_guard=selected_guard.snapshot()),
                                indent=2, sort_keys=True)+'\n').encode())
                raise RuntimeError(
                    'selected anchor cgroup budget is smaller than its checked '
                    'phase plan: ' + json.dumps(detail, sort_keys=True))
    if args.capture_calibration_out or args.calibration_cache:
        from . import tessera_calibration_cache as calibration_store
        hi, lo = census_token_counts(census, {})
        bound_calibration = th.calibration_identity(
            corpus_text, tokens, fit_tokens=hi,
            **_campaign_calibration_parameters(args), fit_tokens_min=lo)
        require_census_draw(census, bound_calibration, where="calibration capture")

        def _capture_identity(unit_names=None, *, source_owner=None):
            return calibration_store.capture_identity(
                args.calibration_census, calibration=bound_calibration,
                max_act_rows=args.max_act_rows, model_load_contract=model_load_contract,
                attention_implementation=attention_implementation,
                **({} if unit_names is None else {'unit_names': unit_names}),
                **({'source_authentication': source_owner}
                   if source_owner is not None else {}),
                **(dict(resource_check=None if selected_guard is None else selected_guard.check,
                        release_read_pages=True) if selected_source else {}))
        # A fresh selected capture carries the census/model-validated scope
        # this run requested. A reuse row takes the stored capture's own
        # scope instead: only when its manifest declares a selected scope
        # does the expected identity carry one; a full capture keeps the full
        # identity, selected run or not. A completed OUTPUT hashes its source
        # the plain way: this row's recording owner read no payload, and the
        # stored identity binds the digests it must equal. A --calibration-cache
        # reader keeps the owner it authenticated with, so a selected reader
        # still hashes only the objects it consumes.
        fresh_unit_names = capture_unit_names \
            if args.capture_calibration_out and args.units else None
        capture_identity = None
        capture_manifest = None
        completed_capture = (Path(args.capture_calibration_out) / "capture_manifest.json"
                             if args.capture_calibration_out else None)
        completed_output = completed_capture is not None and completed_capture.exists()
        if completed_output:
            capture_manifest = calibration_store.require_capture_contract(completed_capture)
        elif args.calibration_cache:
            capture_manifest = calibration_store.require_capture_contract(
                args.calibration_cache, expected_sha256=args.calibration_cache_sha256)
        expected_unit_names = fresh_unit_names
        if capture_manifest is not None:
            stored_units = (sorted(capture_manifest['identity']['units'])
                            if capture_manifest['identity'].get('unit_scope') == 'selected'
                            else None)
            # A completed capture is reused only for the scope it actually
            # published: a run requesting units the stored capture does not
            # cover -- a different selection, or the full scope over a
            # selected capture -- refuses here, before any forward.
            if completed_capture is not None and fresh_unit_names != stored_units:
                raise RuntimeError(
                    f"scope mismatch: the completed capture at "
                    f"{completed_capture} published "
                    f"{'the whole census scope' if stored_units is None else sorted(stored_units)} "
                    f"and this run requests "
                    f"{'the whole census scope' if fresh_unit_names is None else sorted(fresh_unit_names)}")
            if args.calibration_cache:
                missing = sorted(set(targets) - set(capture_manifest['entries']))
                if missing:
                    raise RuntimeError(
                        f"calibration cache does not cover requested units: {missing}")
            expected_unit_names = stored_units
        capture_identity = _capture_identity(
            expected_unit_names,
            source_owner=None if completed_output else source_authentication)
    if selected_source:
        manifest = (capture_manifest if capture_manifest is not None else
                    calibration_store.require_capture_contract(args.calibration_cache,
                        expected_sha256=args.calibration_cache_sha256))
        if manifest['identity'] != capture_identity:
            raise RuntimeError('selected source capture identity differs from the canonical census')
        try:
            if streaming_head and expert_targets:
                selected_weights, selected_source_preparation = _streamed_source_weights(
                    runner, profile=profile, dense_targets=dense_targets,
                    expert_targets=expert_targets,
                    resources=selected_resources, snapshot_policy=args.source_snapshot_policy,
                    resource_check=None if selected_guard is None else selected_guard.check)
            else:
                selected_weights, selected_source_preparation = runner.snapshot_selected_weights(
                    targets, max_resident_bytes=selected_resources['selected_source_weight_bytes'],
                    **({'expected_source_keys': selected_resources['source_tensor_keys']}
                       if args.source_snapshot_policy == 'selected-tensors-v1' else {}),
                    resource_check=None if selected_guard is None else selected_guard.check,
                    **({'host': True} if streaming_head else {}))
        finally:
            runner.shutdown()
        selected_source_preparation.update(resources=selected_resources,
            # The plan beside it states deltas, so the receipt has to carry
            # the floor those deltas are over, measured in this row's own
            # process rather than assumed from another host's run
            # (RobTand/prismaquant#390). None on a CPU row, where there is no
            # guard and therefore no measurement to report.
            baseline=(None if selected_guard is None
                      else dict(selected_guard.baseline)),
            initialization_witness_origin='complete-canonical-capture',
            full_source_initialization_repeated=False)
        # Release fixed non-body state and the source context before selected
        # H/X become resident. Packed members now reference only meta tensors.
        del model, runner
        runner = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if selected_guard is not None and not streaming_head:
            selected_guard.check('before_selected_capture_prefetch', reserve_bytes=
                selected_resources['phases']['resident_anchors']['selected_hessian_bytes']+
                selected_resources['phases']['resident_anchors']['selected_prefix_bytes'])
    if args.capture_calibration_out:
        completed_capture = Path(args.capture_calibration_out) / "capture_manifest.json"
        if completed_capture.exists():
            _values, record = calibration_store.prefetch_capture(
                completed_capture, expected_identity=capture_identity,
                census=census, names=sorted(capture_identity['units']), device="cpu")
            print(f"[campaign] complete calibration capture reused: {record}", flush=True)
            return 0
    if args.calibration_cache:
        if streaming_head:
            # Nothing is read here. The row stream below reads each entry
            # when its batch needs it, and finalize checks every observed
            # count and maximum against the census. The receipt key is
            # placed now so the payload keeps the load-all head's key order.
            values = ({}, {}, {}, {})
            selected_source_preparation['capture_load_execution'] = None
        elif selected_source:
            values, calibration_cache, load_receipt = _prefetch_selected_capture(args,
                expected_identity=capture_identity, census=census, names=targets,
                device=device, resources=selected_resources, guard=selected_guard)
            if load_receipt is not None:
                selected_source_preparation['capture_load_execution'] = load_receipt
        else:
            values, calibration_cache = calibration_store.prefetch_capture(
                args.calibration_cache, expected_identity=capture_identity,
                census=census, names=targets, device=device,
                expected_sha256=args.calibration_cache_sha256)
        acts, hessians, hessian_rows, act_max_abs = values
        if selected_guard is not None and not streaming_head:
            selected_guard.check('after_selected_capture_prefetch')
    else:
        acts, hessians, hessian_rows, act_max_abs = calibration_forward.capture_activation_statistics(
            model, targets, tokens, 0 if census_only else args.max_act_rows, device,
            want_hessian=want_h, profile=profile)
    hessian_token_count, hessian_token_min = census_token_counts(census, hessian_rows)
    hessian_identity = th.calibration_identity(
        corpus_text, tokens, fit_tokens=int(hessian_token_count),
        **_campaign_calibration_parameters(args), fit_tokens_min=int(hessian_token_min))
    if census is not None:
        require_census_draw(census, hessian_identity,
                            where=f"--calibration-census {args.calibration_census}")
    if args.capture_calibration_out:
        calibration_cache = calibration_store.publish_capture(
            args.capture_calibration_out, census_path=args.calibration_census,
            identity=capture_identity, acts=acts, hessians=hessians,
            counts=hessian_rows, maxima=act_max_abs)
        print(f"[campaign] complete calibration capture: {calibration_cache}", flush=True)
        return 0
    print(f"[campaign] activations ready (hessian={args.hessian}, rows/Linear "
          f"{hessian_token_min}..{hessian_token_count})", flush=True)

    # The static A-side calibration, from the same forward passes: one
    # input_global_scale per unit under the resolved contract policy, fused
    # siblings sharing one value.  Priced by every W4A4 anchor below and
    # written beside the payload for the export leg, so the scale that priced
    # the table is the scale the artifact serves (priced == served).
    #
    # Under a census the maxima are the SCOPE's, not this selection's, and for
    # a reason the anchor grouping does not cover: fused-sibling unification
    # (``unify_fused_sibling_max_abs``) has its own fallbacks and can group
    # units the profile's ``fused_sibling_group`` does not, so a selection that
    # is whole by the anchor partition can still be partial by the scale
    # partition -- and a partial fused group calibrates a different
    # ``input_global_scale`` than the module vLLM executes. Taking the scope's
    # maxima removes that dependence entirely, and every unit this run hooked
    # is checked against them first.
    static_scales, static_scale_policy = _static_input_scales(
        act_max_abs if census is None else census_max_abs(census, act_max_abs),
        profile=profile)

    if selected_weights is not None:
        weights = selected_weights
    else:
        weights = {name: dict(model.named_modules())[name].weight.detach()
                   for name in dense_targets}
        for name, member in expert_members.items():
            weights[name] = member.weight.detach()
        del model
    torch.cuda.empty_cache()

    checkpoint_settings = vars(args).copy()
    request = loaded_campaign_request(args, mode=mode, selected_source=selected_source,
        streaming_head=streaming_head, wire_dir=wire_dir, checkpoint=checkpoint,
        identity_threads=identity_threads_requested)
    return run_loaded_campaign(request, weights=weights,
        statistics=calibration_forward.CalibrationStatistics(acts, hessians, hessian_rows, act_max_abs),
        hessian_identity=hessian_identity, hessian_token_count=hessian_token_count,
        hessian_token_min=hessian_token_min, scope=scope, profile=profile,
        static_scales=static_scales, static_scale_policy=static_scale_policy,
        device=device, cache_dir=cache_dir, calibration_cache=calibration_cache,
        capture_identity=capture_identity, hessian_status=hessian_status,
        model_load_contract=model_load_contract, attention_implementation=attention_implementation,
        capture_runtime=capture_runtime, serving_target=serving_target,
        source_scope=source_scope, waits=waits, encoder_seal_ahead=encoder_seal_ahead,
        source_authentication=source_authentication,
        selected_source_preparation=selected_source_preparation,
        selected_resources=selected_resources if selected_source else None,
        selected_guard=selected_guard, checkpoint_settings=checkpoint_settings)


def run_loaded_campaign(request: LoadedCensusRequest | LoadedPricingRequest, *,
                        weights, statistics: calibration_forward.CalibrationStatistics,
                        hessian_identity, hessian_token_count, hessian_token_min,
                        scope: ResolvedCampaignScope, profile, static_scales,
                        static_scale_policy, device, cache_dir,
                        calibration_cache, capture_identity, hessian_status,
                        model_load_contract, attention_implementation, capture_runtime,
                        serving_target, source_scope, waits, encoder_seal_ahead,
                        source_authentication, selected_source_preparation,
                        selected_resources, selected_guard, checkpoint_settings) -> int:
    """Execute one loaded scope with its existing source and publication custody.

    Census and pricing share the established source/memo/cache/menu construction.
    The caller retains source-scope lifetime and calibration/source authority;
    this operation owns the synchronous row stream, journals and publisher.
    Raw checkpoint settings are only a wire input, never a policy fallback.
    """
    import torch
    from .production_weight_cache import ProductionWeightCache
    from .tessera_publication import PublicationError
    from .tessera_rate_surface import leave_one_anchor_out
    from .tessera_render import HessianContractError
    from .tessera_serving_scope import scope_provenance
    from .tessera_row_stream import RECEIPT_FIELDS, checkpoint_present

    census_only = isinstance(request, LoadedCensusRequest)
    if census_only:
        mode = request.menu_mode
        tp_degree = request.census.tp_degree
        family_restriction = request.census.family_restriction
        model_path = request.census.model
        identity_bytes = 0  # Actual CLI admits no census identity budget.
        source = ResidentCampaignWeights()
        want_h = False
    else:
        mode = request.encoding.mode
        tp_degree = request.encoding.tp_degree
        family_restriction = request.encoding.family_restriction
        model_path = request.draw.model
        identity_bytes = request.encoding.identity_bytes
        source = request.source
        want_h = request.encoding.hessian == 'require'
    selected_source = isinstance(source, SelectedCampaignWeights)
    streaming_head = selected_source and source.head == 'stream'
    if selected_source:
        from . import tessera_calibration_cache as calibration_store

    population = scope.population
    roster = scope.roster
    pinned_roster = scope.pinned_roster
    dense_targets = scope.dense_targets
    expert_targets = scope.expert_targets
    expert_members = scope.expert_members
    census_dense_targets = scope.census_dense_targets
    census_expert_targets = scope.census_expert_targets
    scope_groups = scope.scope_groups
    census = scope.census
    selection = scope.selection
    stack_samples = scope.stack_samples
    exact_member_scope = scope.exact_member_scope
    partition_menu_targets = scope.partition_menu_targets
    partition_unit_records = scope.partition_unit_records
    selected_groups = scope.selected_groups
    audit_units = scope.audit_units
    inclusion_probability = scope.inclusion_probability
    context_by_unit = scope.context_by_unit
    structure_by_unit = scope.structure_by_unit
    encode_structure = scope.encode_structure
    acquisition = scope.acquisition
    all_dense = list(roster.dense)
    pinned = list(roster.pinned)
    targets = [*dense_targets, *expert_targets]
    census_targets = [*census_dense_targets, *census_expert_targets]
    acts, hessians, hessian_rows, act_max_abs = statistics

    # ONE ActivationSource for the whole campaign, and ONE set of encoder
    # keywords per unit PER SCALE PLANE. The block-LDL is a function of the
    # unit's Hessian alone -- not of its rate -- so a twelve-anchor surface
    # would otherwise factorise the same [in, in] matrix twelve times. The
    # refit metric is a function of the Hessian AND the plane, because Tessera
    # keys the refit objective by plane (the exact quadratic on a CHANNEL row
    # scale, a diagonal power on the LUT plane's coupled blocks) and the two
    # measured answers disagree. The plane is a property of the family, not of
    # the rung, so the memo is keyed by (unit, plane). Selected-source rows
    # retain at most one compatible batch's factors and deterministically
    # recompute evicted entries; resident-source rows keep the historical
    # memo. The source is built from the
    # same functions the production render calls (``tessera_hessian``), so the
    # campaign's price and the cache's render are one rendering of one draw
    # (principle 8).
    calibration_source = None
    seal_ahead = None
    # The stream head builds one source per entry on its reader threads.
    if want_h and not streaming_head:
        calibration_source = th.activation_source(hessians, hessian_identity)
        if identity_bytes > 0:
            # The producer's capture seal, taken now on a helper thread so it
            # runs under the projection and the identity plan rather than as
            # the first receipt of the hold.  Joined before the hold and, on
            # any exit, before the resident tensors go.
            seal_ahead = _SealAhead(calibration_source)
            source_scope.callback(seal_ahead.finish)

    def activation_kwargs_for(source):
        return _activation_kwargs_memo(source, weights, device,
            # The capacity the plan CHARGED, not the batch width it was derived
            # from, so the memo policy has one owner (RobTand/prismaquant#389 can
            # move it without the charge and the construction drifting apart).
            max_entries=(selected_resources['encoder_memo_capacity']
                         if selected_source else None),
            resource_check=None if selected_guard is None else selected_guard.check,
            factor_scratch_bytes=(selected_resources['phases']['resident_anchors']['factorization_scratch_bytes']
                                  if selected_source else 0))

    # Before the first producer binding/metadata plan or run identity. Keep
    # those APIs and every identity field unchanged; only their cold seal's
    # scheduling moves. No unused work is submitted for census/capture heads.
    if encoder_seal_ahead is not None:
        encoder_seal_ahead.wait()
    _activation_kwargs_for = activation_kwargs_for(calibration_source)

    cache = ProductionWeightCache(
        weights={}, levers={"tessera_campaign": True},
        cache_dir=str(cache_dir),
        metadata={"schema": SCHEMA, "menu_mode": mode,
                  **({'release_completed_anchor_file_pages': True} if selected_source else {})},
    )
    menu = resolve_campaign_menu(weights, scope, profile, mode=mode,
        tp_degree=tp_degree, family_restriction=family_restriction,
        model_path=model_path, cache_dir=cache_dir,
        selected_source=selected_source, streaming_head=streaming_head,
        source_authentication=source_authentication, selected_guard=selected_guard)
    menus = menu.menus
    context_by_unit = menu.context_by_unit
    structure_by_unit = menu.structure_by_unit
    encode_structure = menu.encode_structure
    no_admitted_rung = menu.no_admitted_rung
    expert_projection = menu.expert_projection
    projected_units = menu.projected_units
    if menu.partition is not None:
        full_partition_menus = menu.partition.full_menus
        partition_encode_structure = menu.partition.encode_structure

    def close_source_authentication():
        """Record the source owner's receipt, then release its descriptors.

        The stream head's readers still read projected source tensors through
        this owner, so it closes after ``RowStream.finish`` (PQ #1654); every
        other head closes it here, once the projection check is done.
        """
        if source_authentication is not None:
            selected_source_preparation['source_authentication'] = source_authentication.receipt()
            source_authentication.close()

    route_cache: dict[tuple, str | None] = {}
    acquisition_schedule = None
    if isinstance(request, LoadedPricingRequest) and acquisition is not None:
        acquisition_groups = resolve_anchor_groups(targets, profile=profile, expert_members=expert_members)
        acquisition_rates = {name: {} for name in targets}
        for name in targets:
            for rung in menus[name]:
                if request.recipe.max_artifact_bpp > 0 and rung.bpp > request.recipe.max_artifact_bpp:
                    continue
                acquisition_rates[name].setdefault(rung.family, set()).add(rung.body_rate_q256)
        acquisition_grids, _ = anchor_group_rate_grids(acquisition_groups, acquisition_rates,
            encode_structure=encode_structure, projected_units=projected_units, route_cache=route_cache)
        acquisition_schedule = _campaign_round_one_schedule(acquisition_groups, acquisition_grids,
            placement=request.recipe.placement, audit_units=audit_units, snap=None, requests=acquisition["requests"],
            rates_by_unit=acquisition_rates)
        checkpoint_settings["acquisition_schedule"] = acquisition_schedule
        acquisition_origin = campaign_acquisition_origin(acquisition, acquisition_schedule, menus)
        checkpoint_settings["acquisition_origin"] = acquisition_origin
        checkpoint_settings["acquisition_source_weights"] = acquisition["source_weights"]

    if not streaming_head:
        close_source_authentication()
    elif source_authentication is not None:
        # The receipt is filled at finalize; its key is placed here so the
        # preparation record keeps the load-all head's field order, and the
        # two heads' cost payloads stay byte-identical.
        selected_source_preparation['source_authentication'] = None

    if isinstance(request, LoadedCensusRequest):
        return run_census(request.census, counts=hessian_rows, maxima=act_max_abs,
            groups=scope_groups, dense_targets=census_dense_targets,
            expert_targets=census_expert_targets,
            shapes={name: tuple(weight.shape) for name, weight in weights.items()},
            identity=hessian_identity, expert_projection=expert_projection,
            model_load_contract=model_load_contract,
            attention_implementation=attention_implementation, capture_runtime=capture_runtime,
            pinned_roster=pinned_roster, expected_units=census_targets)

    recipe = request.recipe
    placement = recipe.placement
    publication = request.publication
    outputs = request.outputs
    draw = request.draw
    checkpoint = outputs.checkpoint
    wire_dir = outputs.wire_dir
    seed_checkpoint = None if request.seed is None else request.seed.checkpoint
    seed_wire_dir = None if request.seed is None else request.seed.wire_dir
    identity_threads_requested = publication.requested_identity_threads

    checkpoint_settings = MappingProxyType(checkpoint_settings)

    from .cost_stage_checkpoint import prepare_journal, write_unit
    from .prismabuild_progress import report as report_progress

    # The source/H template is deliberately built BEFORE the resume gate: it
    # is the producer computation that supplies that gate's exact records.
    # The hold owns metadata only; caller-owned tensors remain in the existing
    # resident maps.  Register its close before replacement sources are added
    # to the ExitStack, so publication drains before either owner is released.
    identity_metadata_bounds, identity_planning_scratch_bytes = ({}, 0)
    # An empty menu has no closed producer roster. Preserve the existing
    # empty-menu refusal path instead of constructing a synthetic holder.
    # The stream head builds each unit's holder on its reader thread, inside
    # the window, so it never holds the population's.
    reuse_campaign_identity = bool(not streaming_head and identity_bytes > 0
                                   and all(menus.values()))
    # The plan charges the requested builder count; the row never runs more
    # builders, readers or wire verifiers than the CPUs it was admitted with.
    identity_threads = _identity_threads_for_this_process(identity_threads_requested)
    identity_hold_seconds = seal_wait_seconds = None
    if reuse_campaign_identity:
        # No real receipt nor tensor value is read here. This source-free
        # model is charged as its own transient alongside the retained hold.
        identity_metadata_bounds, identity_planning_scratch_bytes = \
            _campaign_identity_metadata_plan(
                weights=weights, menus=menus, calibration_source=calibration_source,
                projected_units=projected_units, static_scales=static_scales,
                threads=identity_threads_requested)
        identity_metadata_bytes = sum(identity_metadata_bounds.values())
        if selected_source:
            reserved_identity_bytes = int(identity_bytes)
            if identity_metadata_bytes + identity_planning_scratch_bytes > reserved_identity_bytes:
                raise RuntimeError('campaign identity closed-roster plan exceeds --campaign-identity-bytes')
            if selected_guard is not None:
                selected_guard.check('before_selected_campaign_identity_bind',
                    reserve_bytes=reserved_identity_bytes + int(
                        selected_resources['phases']['resident_anchors']
                        ['campaign_identity_hold_scratch_bytes']))
        if seal_ahead is not None:
            # The one first reader of the owner. Every builder below reads a
            # sealed owner, so none of them digests the population.
            seal_wait_seconds = seal_ahead.wait()
    import time as _time
    identity_hold_started = _time.monotonic()
    bound_checkpoint_units = ({
        } if not reuse_campaign_identity else _campaign_bound_identities(
            weights=weights, menus=menus, calibration_source=calibration_source,
            projected_units=projected_units, static_scales=static_scales,
            metadata_bounds=identity_metadata_bounds, threads=identity_threads))
    identity_hold_seconds = _time.monotonic() - identity_hold_started
    if bound_checkpoint_units:
        source_scope.callback(lambda: [unit.close() for unit in bound_checkpoint_units.values()])
    identity_metadata_observed_bytes = sum(unit.observed_metadata_bytes()
                                           for unit in bound_checkpoint_units.values())
    if bound_checkpoint_units:
        print(f"[campaign] campaign identity hold: {len(bound_checkpoint_units)} units, "
              f"observed metadata {identity_metadata_observed_bytes} B, planned bound "
              f"{sum(identity_metadata_bounds.values())} B + scratch "
              f"{identity_planning_scratch_bytes} B, reserved {identity_bytes} B; "
              f"{identity_threads} threads, {identity_hold_seconds:.1f} s"
              + ("" if seal_ahead is None else
                 f"; capture seal ahead {seal_ahead.seconds:.1f} s, waited {seal_wait_seconds:.1f} s"),
              flush=True)

    if acquisition is not None and not streaming_head:
        for name in sorted(targets):
            receipt = (bound_checkpoint_units[name].campaign_inputs()["source"]
                       if bound_checkpoint_units else None)
            _require_campaign_acquisition_source(name, weights[name],
                acquisition["source_weights"][name], receipt=receipt)

    row_stream = None
    if streaming_head:
        from .tessera_row_stream import RowStream
        stream_threads = identity_threads
        # Holder metadata for the units the window can hold at once: its two
        # batches plus one in flight on every reader.
        window_bounds, window_scratch_bytes = _campaign_identity_metadata_plan(
            weights=weights, menus=menus, calibration_source=(True if want_h else None),
            projected_units=projected_units, static_scales=static_scales,
            threads=stream_threads)
        window_metadata_bytes = window_scratch_bytes + sum(sorted(
            window_bounds.values(), reverse=True)[:2 * publication.batch_size + stream_threads])
        if 0 < identity_bytes < window_metadata_bytes:
            raise RuntimeError('row stream identity window plan exceeds --campaign-identity-bytes')
        if selected_guard is not None:
            selected_guard.check('before_row_stream_identity_window',
                                 reserve_bytes=window_metadata_bytes)

        streamed_units = [name for name in targets if weights[name].is_meta]
        if streamed_units:
            _require_streamed_projection(
                streamed_units, projected_units, expert_projection['producer']['source'],
                selected_resources.get('source_tensor_keys')
                if source.snapshot_policy == 'selected-tensors-v1' else None)

        def load_projected_source(name):
            """One projected unit's source tensor, on a reader thread (PQ #1654).

            The tensor the exporter re-reads, through the owner the row
            authenticated its shards with; its pages are advised away once the
            copy is taken.
            """
            weight, release = _read_projected_unit(
                name, projected_units[name], model_path=model_path,
                source=expert_projection['producer']['source'], release_source_pages=True,
                source_authentication=source_authentication)
            release()
            return weight

        acquisition_source_checked = set()

        def bind_stream_unit(name, *, weight, inputs, hessian, source):
            holder, receipts = _stream_unit_identity(name, weight=weight,
                inputs=inputs, hessian=hessian, source=source if want_h else None,
                menu=menus[name], projected_unit=projected_units.get(name),
                static_scales=static_scales, metadata_bound=window_bounds[name])
            try:
                if acquisition is not None and name not in acquisition_source_checked:
                    _require_campaign_acquisition_source(name, weight,
                        acquisition["source_weights"][name], receipt=receipts["weight"])
                    acquisition_source_checked.add(name)
            except BaseException:
                if holder is not None:
                    holder.close()
                raise
            # A reread is already bound to the first producer receipt by
            # RowStream; do not add another full raw-byte hash for that unit.
            return holder, receipts

        row_stream = RowStream(
            capture_path=source.capture, expected_sha256=source.capture_sha256,
            expected_identity=capture_identity, census=census, names=targets,
            policy=source.load_policy, weights=weights, hessian_identity=hessian_identity,
            bind=bind_stream_unit,
            load_unit=load_projected_source if streamed_units else None,
            threads=stream_threads, batch_size=publication.batch_size, device=device,
            memo_capacity=selected_resources['encoder_memo_capacity'],
            resource_check=None if selected_guard is None else selected_guard.check,
            wait_sink=waits.ledger.sink,
            factor_scratch_bytes=(selected_resources['phases']['resident_anchors']
                                  ['factorization_scratch_bytes']))
        source_scope.callback(row_stream.close)
        calibration_cache = row_stream.capture
        _prime_first_anchor_batch(row_stream, placement=placement, seed_checkpoint=seed_checkpoint,
            batch_size=publication.batch_size, max_artifact_bpp=recipe.max_artifact_bpp,
            checkpoint=checkpoint,
            targets=targets, menus=menus, weights=weights, profile=profile,
            expert_members=expert_members, encode_structure=encode_structure,
            projected_units=projected_units, audit_units=audit_units, route_cache=route_cache,
            partitioned=partition_menu_targets is not None,
            acquisition_requests=None if acquisition is None else acquisition["requests"])

    def run_identity(**receipts):
        # The resume identity, run level: everything a price is a function of,
        # including the static A-side contract (scales + policy) the W4A4 rows
        # are scored under.  A checkpoint from another calibration or policy is
        # refused here, by field, before a row of it is read.
        return campaign_checkpoint_identity(
            weights=weights, acts=acts, hessians=hessians, menus=menus, settings=checkpoint_settings,
            calibration_identity=hessian_identity,
            serving_scope=(scope_provenance(serving_target, context_by_unit)
                           if serving_target is not None else None),
            static_scales=static_scales, static_scale_policy=static_scale_policy,
            expert_projection=expert_projection,
            stack_sampling_identity={name: record
                for entry in (selection or {}).get("groups", [])
                for name, record in entry.get("stack_samples", {}).items()},
            structure_by_unit=structure_by_unit, **receipts)

    def open_journal(identity):
        return prepare_journal(
            checkpoint.with_name(checkpoint.name + ".parts"), manifest_path=checkpoint,
            stage="Tessera campaign", resume=True, identity=identity,
            qnames=targets, seal_fields=campaign_wire.CAMPAIGN_CHECKPOINT_SEAL_FIELDS,
        )

    # Under the stream head the identity, and so the journal, exist only at
    # finalize (``finalize_row_stream``): the identity binds every unit's
    # receipts, and the last of them is taken when the last entry is read.
    checkpoint_identity = journal = identity_sha256 = None
    resumed = {}
    # Until then the stream head keeps its own journal beside the checkpoint
    # (PQ #1403). Its identity is the run identity with every unit's W, X and H
    # receipts left to the unit's shard, which carries the receipts its reader
    # took; so a killed row's anchors can be adopted by the relaunch, one unit
    # at a time, once that unit's entry is read again and its receipts match.
    # It opens like the checkpoint journal: a stream journal under any other
    # identity is refused by field, never silently discarded.
    stream_journal = stream_identity_sha256 = None
    stream_resumed = {}
    dirty_stream_units = set()
    # A checkpoint an earlier attempt left, under either head, resumes on the
    # stream head (PQ #1613). Its manifest records every unit's W, X and H
    # receipts, which this run can only re-derive one entry at a time, so they
    # are taken from the manifest and every other field of the run identity is
    # compared by name here, before the first entry is read. The stream then
    # requires each entry's first read to reproduce its unit's receipts before
    # the consumer sees it, and finalize compares the whole identity again,
    # built from this run's own receipts. The journal is open from here on, so
    # a unit encoded now is journalled under the checkpoint's identity as soon
    # as it flushes: its own receipts were already checked when it was read.
    window_resumed = {}
    if not streaming_head:
        checkpoint_identity = run_identity(
            **({"bound_units": bound_checkpoint_units} if bound_checkpoint_units else {}))
        journal, identity_sha256, resumed = open_journal(checkpoint_identity)
    elif checkpoint_present(checkpoint):
        from .cost_stage_checkpoint import stored_manifest_identity
        recorded_receipts = _manifest_unit_receipts(stored_manifest_identity(checkpoint), weights)
        checkpoint_identity = run_identity(unit_receipts=recorded_receipts)
        journal, identity_sha256, window_resumed = open_journal(checkpoint_identity)
        row_stream.expect_identities({name: recorded_receipts[name] for name in targets})
    else:
        stream_journal, stream_identity_sha256, stream_resumed = prepare_journal(
            checkpoint.with_name(checkpoint.name + STREAM_JOURNAL_SUFFIX),
            stage=STREAM_JOURNAL_STAGE, resume=True, qnames=targets,
            seal_fields=campaign_wire.CAMPAIGN_CHECKPOINT_SEAL_FIELDS,
            identity=run_identity(unit_receipts={
                name: dict.fromkeys(RECEIPT_FIELDS, STREAM_JOURNAL_RECEIPT)
                for name in weights}))
    measured: dict[str, dict[str, list[CampaignAnchor]]] = {}
    # Rows adopted from another campaign whose rungs THIS run's menu does not
    # admit.  They are measurements of the same rate/distortion law and cost
    # the same GPU seconds to make, so they are carried rather than discarded;
    # they are held apart from ``measured`` because a price the pinned reader
    # cannot decode is not a price.  Nothing in ``costs`` is ever built from
    # this, and no fresh encode ever lands here: the menu decides what is
    # encoded, so a row can only become unservable by being adopted.
    unservable: dict[str, dict[str, dict]] = {}
    wire_records = {name: {} for name in targets}
    dirty_checkpoint_units = set()
    restricted_rate_band = (parse_rate_band(placement.rate_band)
                            if family_restriction is not None else None)

    def validate_seed_scope(name, state):
        require_seed_family_scope(name, state, family_restriction=family_restriction,
                                  structure_by_unit=structure_by_unit,
                                  rate_band=restricted_rate_band, profile=profile)
        on_menu = {entry.format_name for entry in menus[name]}
        for row in state["anchors"]:
            if row["format_name"] in on_menu:
                _require_resumable_anchor(CampaignAnchor(**row), static_scales, profile=profile)
        if acquisition_schedule is not None:
            for row in state["anchors"]:
                require_campaign_acquisition_anchor(CampaignAnchor(**row), acquisition_schedule)
            if state.get("unservable"):
                raise ValueError("acquisition refuses unservable checkpoint/seed anchor work")
        # Resume does not reconstruct cache.weights; seed wires are linked
        # before adopt_state reads them. Reserve every admitted coordinate
        # here, before either path touches those bytes, including stale rows
        # whose seed files are linked but whose prices are not adopted.
        campaign_wire.register_wire_coordinates(
            cache, wire_dir,
            ((row["qname"], row["format_name"]) for row in state["anchors"]
             if row["format_name"] in on_menu))

    def adopt_state(name: str, state, *, where: str, deferred=None, entry=None) -> None:
        """Verify one unit's stored anchors against this run and take them.

        The one path for both a resume of this run's own checkpoint and an
        adoption from another campaign's: what makes a stored row usable is
        that its inputs and its bytes are this run's, and that is checked here
        rather than inferred from which file the row came out of.

        A stored row whose rung is outside this run's menu is neither priced
        nor refused under the legacy unrestricted behavior: it is recorded as
        ``unservable`` evidence. An explicit family restriction instead refuses
        active seed rows outside its family list or requested rate band. The two are
        different failures.  A row that disagrees with this run's weights,
        Hessian applicability or static scale is a row about some other
        campaign and is refused by name; a row this menu does not admit is a
        correct measurement of a rung the pinned reader cannot decode, and the
        honest thing to do with it is keep it out of the prices and in the
        record.

        On the ``checkpoint`` path that branch is unreachable, and
        deliberately so: ``menu_mode`` is bound in the journal identity (it is
        one of the ``settings``, and only locations are popped), so a resume
        under a different menu is refused by ``prepare_journal`` before a row
        is read. Evidence arrives only through ``--seed-checkpoint``, which is
        the path whose whole purpose is to cross that boundary.

        With ``deferred`` (a list), the row's input gates still run here, in
        order, but its wire receipt is appended as ``(anchor, identity,
        existing)`` instead of being verified inline; the caller verifies the
        whole list on worker threads and fills ``wire_records`` in this same
        order.  Without it, the receipt is verified where it always was.

        ``entry`` is the stream head's resident entry for this unit
        (``RowStream.entry``). Each row's input identity is then derived from
        that entry's own source and holder, exactly as the encode loop derives
        a fresh anchor's, rather than from a run-wide source the stream head
        does not have.
        """
        if not isinstance(state, dict) \
                or set(state) - {"unservable"} != {"anchors", "wire_records"} \
                or not isinstance(state["anchors"], list) \
                or not isinstance(state["wire_records"], dict):
            raise RuntimeError(f"{where} state has an invalid anchor/record envelope for {name}")
        if name not in menus:
            raise RuntimeError(f"{where} state names a unit this run does not price: {name}")
        validate_seed_scope(name, state)
        on_menu = {entry.format_name for entry in menus[name]}
        formats = set()
        for row in state["anchors"]:
            anchor = CampaignAnchor(**row)
            if anchor.qname != name or anchor.format_name in formats:
                raise RuntimeError(f"{where} anchor has a wrong or duplicate unit/rung: {name}")
            formats.add(anchor.format_name)
            if anchor.format_name not in state["wire_records"]:
                raise RuntimeError(f"{where} anchor has no priced-wire receipt: {name}")
            if anchor.format_name not in on_menu:
                unservable.setdefault(name, {})[anchor.format_name] = {
                    "anchor": dict(row),
                    "wire_record": dict(state["wire_records"][anchor.format_name]),
                    "adopted_from": where,
                }
                continue
            stale = _stale_served_wire(anchor, state["wire_records"][anchor.format_name],
                                       structure=(encode_structure or {}).get(name))
            if stale is not None:
                # On the menu, but stamped on a wire this unit's structure is
                # not served on (#1502): a routed E2M1x2 row priced on the
                # research WINDOW wire before the served recipe reached the
                # encoder.  Kept as evidence, never adopted as a price; the
                # round re-measures the rung on the served wire.
                unservable.setdefault(name, {})[anchor.format_name] = {
                    "anchor": dict(row),
                    "wire_record": dict(state["wire_records"][anchor.format_name]),
                    "adopted_from": where,
                    "reason": stale,
                }
                continue
            # Row level, the same rule: the row's inputs (Hessian
            # applicability, static scale) must be what this run's producer
            # stamps for its rung, then its wire receipt must verify.
            if entry is not None:
                source, bound = (entry.source if want_h else None), {"bound_unit": entry.holder}
            else:
                source = calibration_source
                bound = ({"bound_unit": bound_checkpoint_units[name]}
                         if bound_checkpoint_units else {})
            identity = _checkpoint_anchor_identity(
                anchor, weights=weights, menus=menus,
                calibration_source=source, static_scales=static_scales,
                projected_units=projected_units, profile=profile,
                structure=(encode_structure or {}).get(name), **bound)
            existing = state["wire_records"][anchor.format_name]
            if deferred is not None:
                deferred.append((name, anchor, identity, existing))
            else:
                wire_records[name][anchor.format_name] = campaign_wire.checkpoint_wire_record(
                    anchor, wire_dir, identity, existing=existing)
            measured.setdefault(name, {}).setdefault(anchor.family, []).append(anchor)
        if formats != set(state["wire_records"]):
            raise RuntimeError(f"{where} has wire receipts outside its measured anchors: {name}")
        # Evidence the source had already set aside stays set aside, unless
        # this run's menu admits the rung -- in which case the source and this
        # run disagree about what is servable, and a merge of the two would
        # hold the same row on both sides of the line.
        for fmt, record in (state.get("unservable") or {}).items():
            if fmt in on_menu:
                raise RuntimeError(
                    f"{where} carries {fmt} for {name} as unservable and this "
                    "run's menu admits that rung")
            unservable.setdefault(name, {}).setdefault(fmt, record)

    deferred_wire = [] if identity_threads > 1 else None
    resume_started = _time.monotonic()
    for name, state in resumed.items():
        adopt_state(name, state, where="checkpoint", deferred=deferred_wire)
    if deferred_wire:
        records = _verify_wire_records_on_threads(
            [(anchor, identity, existing) for _name, anchor, identity, existing in deferred_wire],
            wire_dir, threads=identity_threads)
        for (name, anchor, _identity, _existing), record in zip(deferred_wire, records):
            wire_records[name][anchor.format_name] = record
    if resumed:
        print(f"[campaign] resumed {sum(len(v) for f in measured.values() for v in f.values())} "
              f"verified anchors from {checkpoint} in {_time.monotonic() - resume_started:.1f} s "
              f"({identity_threads} threads)", flush=True)

    if stream_resumed:
        # A killed stream-head row's anchors (PQ #1403). Each unit's entry is
        # read and verified through the window like any batch, its receipts
        # must equal the ones its shard recorded, and then every row passes
        # the same gates as a checkpoint resume: its input identity against
        # this entry, its wire receipt against the file on disk. The adopted
        # units stay dirty, so the finalized journal carries them.
        adopt_started = _time.monotonic()
        names = sorted(stream_resumed)
        chunks = [names[start:start + publication.batch_size]
                  for start in range(0, len(names), publication.batch_size)]
        row_stream.plan(chunks)
        for index, chunk in enumerate(chunks):
            row_stream.admit(index)
            for name in chunk:
                state = dict(stream_resumed[name])
                recorded = state.pop("stream_receipts", None)
                entry = row_stream.entry(name)
                if entry.holder is None or recorded != entry.identities:
                    raise RuntimeError(
                        f"stream checkpoint unit {name}: its recorded W/X/H receipts are not "
                        "this run's read of the same entry; refusing to adopt its anchors")
                adopt_state(name, state, where="stream checkpoint", entry=entry)
                dirty_checkpoint_units.add(name)
        print(f"[campaign] adopted {sum(len(state['anchors']) for state in stream_resumed.values())} verified anchors "
              f"for {len(names)} units from {checkpoint.name}{STREAM_JOURNAL_SUFFIX} in "
              f"{_time.monotonic() - adopt_started:.1f} s", flush=True)

    if window_resumed:
        # A resumed checkpoint on the stream head (PQ #1613): the load-all
        # resume's gates, one window at a time. Each unit's entry is read and
        # verified like any batch, and ``expect_identities`` has required its
        # receipts to be the ones the manifest recorded before this sees it.
        # Then every row passes ``adopt_state``: its input identity against
        # this entry, its wire receipt against the file on disk. Wires are
        # verified inline, one blob at a time, as the stream journal's
        # adoption does: a verification holds its whole blob, and no stream
        # phase charges a pool of them beside the window's readers. The units'
        # shards already cite the checkpoint's identity, so, as on the
        # load-all resume, they are not rewritten.
        adopt_started = _time.monotonic()
        names = sorted(window_resumed)
        chunks = [names[start:start + publication.batch_size]
                  for start in range(0, len(names), publication.batch_size)]
        row_stream.plan(chunks)
        for index, chunk in enumerate(chunks):
            row_stream.admit(index)
            for name in chunk:
                entry = row_stream.entry(name)
                if entry.holder is None:
                    raise RuntimeError(
                        f"checkpoint unit {name}: this run's read of its entry binds no "
                        "producer identity; refusing to adopt its anchors")
                adopt_state(name, window_resumed[name], where="checkpoint", entry=entry)
        print(f"[campaign] resumed {sum(len(state['anchors']) for state in window_resumed.values())} "
              f"verified anchors for {len(names)} units from {checkpoint.name} through the "
              f"window in {_time.monotonic() - adopt_started:.1f} s", flush=True)

    seed_provenance = None
    if seed_checkpoint:
        seed_provenance = _adopt_seed_checkpoint(
            seed_checkpoint, seed_wire_dir,
            targets=[name for name in targets if name not in resumed],
            wire_dir=wire_dir, adopt=adopt_state,
            admits=lambda name, fmt: any(
                entry.format_name == fmt for entry in menus.get(name, ())),
            identity_sha256=identity_sha256, expected_identity=checkpoint_identity,
            validate_state=validate_seed_scope)
        for name in seed_provenance["units"]:
            dirty_checkpoint_units.add(name)

    # The export leg's inputs, written AFTER the resume identity has accepted
    # this run's inputs and BEFORE the anchor loop.  Before the loop, so even
    # a deadline-stopped campaign leaves them (RobTand/prismaquant#193); after
    # the gate, because they are the export half of whatever table survives a
    # refusal (RobTand/prismaquant#211).  A refused resume leaves the
    # checkpoint and the previous cost file alone, so it must leave these
    # alone too -- overwriting them with the refused draw's Hessians and
    # scales strands the surviving table, which can then only be re-priced
    # from scratch.  Nothing above consumes the returned paths or digest, and
    # an accepted resume re-writes byte-identical files: the identity binds
    # ``hessians``, ``static_scales`` and ``static_scale_policy``, which are
    # exactly this call's inputs.
    # The capture's ``counts`` describe the DRAW over the priced scope, so a
    # shard writes the census's counts rather than its selection's: the merged
    # capture is then the whole scope's H under the whole scope's counts --
    # exactly the object a whole-scope run writes -- and the merge can prove it
    # by recomputing the digest.
    def publish_export_inputs(export_hessians, hessian_identities):
        return write_export_inputs(
            cache_dir,
            hessians=export_hessians if want_h else None,
            hessian_rows=(hessian_rows if census is None else census["counts"]),
            hessian_identity=hessian_identity,
            static_scales=static_scales,
            static_scale_policy=static_scale_policy,
            **(dict(hessian_reference=dict(canonical_capture=calibration_cache,
                    census_path=source.census,load_policy=outputs.hessian_reference_policy),
                    **({} if hessian_identities is None
                       else dict(hessian_identities=hessian_identities)))
               if outputs.hessian_reference_policy is not None else {}),
            **(dict(release_file_pages=True,
                    resource_check=None if selected_guard is None else selected_guard.check)
               if selected_source else {}),
        )

    hessian_capture_path = input_scales_path = capture_sha256 = None
    if not streaming_head:
        if selected_guard is not None:
            phase = selected_resources['phases']['export_inputs']
            selected_guard.check('before_selected_export_input_write', reserve_bytes=
                phase['export_input_page_window_bytes']+phase['serialization_scratch_bytes'])
        if seal_ahead is not None:
            # Already joined before the hold; here for the empty-menu path, so no
            # helper is digesting resident H while the encode loop runs.
            seal_ahead.wait()
        hessian_capture_path, input_scales_path, capture_sha256 = publish_export_inputs(
            hessians,
            # The H commitments are the receipts the hold sealed from these
            # same resident tensors; only H-free rosters are digested here.
            None if not bound_checkpoint_units or not want_h
            else _bound_hessian_identities(bound_checkpoint_units, hessians))

    def finalize_row_stream():
        """Write the stream head's six identity-bound outputs' inputs, in order.

        Every entry the loop never admitted is read, byte-checked against its
        projected source tensor, and receipted first; the source owner's
        receipt is then taken and its descriptors closed; and
        every observed count and maximum is checked against the census, so a
        corrupt or changed entry refuses before any of the writes exists. Then
        the capture load execution record, the journal manifest, and the
        export inputs; the caller flushes the unit shards and writes the cost
        payload after this returns.
        """
        nonlocal checkpoint_identity, journal, identity_sha256
        nonlocal hessian_capture_path, input_scales_path, capture_sha256
        started = _time.monotonic()
        row_stream.finish()
        # Every projected unit's source bytes have now been checked on a
        # reader, so the owner's receipt covers the reads the serial check made.
        close_source_authentication()
        census_token_counts(census, row_stream.observed_counts())
        census_max_abs(census, row_stream.observed_max_abs())
        receipts = row_stream.unit_identities()
        selected_source_preparation['capture_load_execution'] = _publish_capture_load_execution(
            cache_dir, capture=row_stream.capture, execution=row_stream.load_execution(),
            resources=selected_resources, guard=selected_guard)
        checkpoint_identity = run_identity(unit_receipts=receipts)
        if journal is None:
            journal, identity_sha256, adopted = open_journal(checkpoint_identity)
            if adopted:
                raise RuntimeError("a checkpoint appeared under the stream head while it "
                                   "encoded; refusing to adopt it at finalize")
        else:
            # A resumed checkpoint (PQ #1613): its journal has been open since
            # before the first read, under receipts its manifest recorded. The
            # whole run identity is compared again here, now built from this
            # run's own receipts, exactly as a fresh journal's is; its shards
            # are this run's and the checkpoint's, so they are not refused.
            _root, resumed_identity_sha256, _shards = open_journal(checkpoint_identity)
            if resumed_identity_sha256 != identity_sha256:
                raise RuntimeError("the resumed checkpoint's identity changed at finalize; "
                                   "refusing to write under either")
        if selected_guard is not None:
            phase = selected_resources['stream_phases']['stream_finalize']
            selected_guard.check('before_selected_export_input_write', reserve_bytes=
                phase['export_input_page_window_bytes'])
        # Geometry-only stand-ins: every H commitment is the receipt its
        # reader sealed while the entry was resident, so no H is read again.
        stand_ins = {name: torch.empty([census['unit_shapes'][name][1]] * 2,
                                       dtype=torch.float32, device='meta')
                     for name in sorted(targets)}
        hessian_capture_path, input_scales_path, capture_sha256 = publish_export_inputs(
            stand_ins, {name: receipts[name]['hessian'] for name in sorted(targets)})
        from .cost_stage_checkpoint import atomic_write_bytes
        from .tessera_row_stream import EXECUTION_FILENAME
        record = dict(row_stream.execution_record(),
                      finalize_seconds=round(_time.monotonic() - started, 3))
        atomic_write_bytes(cache_dir / EXECUTION_FILENAME,
                           DIRECT_ASCII_INDENT2_LAX.encoded(record) + b"\n")
        print(f"[campaign] row head: stream finalized {record['units']} units "
              f"(first batch ready {record['first_batch_ready_seconds']} s, "
              f"reads {record['read_seconds']} s, waited {record['wait_seconds']} s, "
              f"{record['hash_only_entries']} receipt-only, {record['rereads']} re-read, "
              f"{record['source_weight_reads']} source weights read in "
              f"{record['source_weight_read_seconds']} s, "
              f"peak {record['peak_resident_units']} units; finalize "
              f"{record['finalize_seconds']} s)", flush=True)

    if outputs.hessian_reference_policy is not None and not streaming_head:
        # Resume was checked before any export input changed. Now reuse the
        # commitments just computed from these same resident tensors, so the
        # first anchor does not hash the full population again. Both encoder
        # kwargs and checkpoint receipts must see the producer's resident
        # mapping; its per-unit content checks remain at consumption.
        calibration_source = th.activation_source(hessians, hessian_identity,
            reference_path=hessian_capture_path, source_scope=source_scope)
        for unit in bound_checkpoint_units.values():
            unit.replace_calibration_source(calibration_source)
        _activation_kwargs_for = activation_kwargs_for(calibration_source)

    # PrismaQuant #291 (filed here first as #288). A narrowing menu mode --
    # ``attested`` without a dev pin, ``readable`` against a contract that
    # publishes no reader for these shapes -- used to resolve to nothing and
    # let the run finish successfully with ``costs: {}``. A zero-row cost
    # table is not a cheap answer, it is a missing one, and downstream it is
    # indistinguishable from a scope that legitimately holds no unit. So the
    # run refuses, and says which mode and which contract produced it.
    #
    # Two things about where this sits. It is AFTER the journal gates because
    # an empty menu is the weakest diagnosis this run can offer: if the
    # checkpoint it was pointed at also describes different weights, a
    # different calibration or a different menu, THAT is what the operator has
    # to fix, and refusing the other way round would hide it behind "no rung
    # admitted". And it is after ``write_export_inputs`` because the Hessian
    # capture and the static A-side scales are facts about the calibration,
    # not about the menu: they were measured correctly, the next run under a
    # menu that admits something will want them, and throwing them away would
    # charge a second forward for a flag change. What is refused is the empty
    # cost table, and only that -- nothing has been encoded here either way.
    if menus and len(no_admitted_rung) == len(menus):
        print(f"[campaign] mode={mode} admits no rung for any of "
              f"{len(menus)} units against {contract_source_label()}; "
              "refusing to write an empty cost table", file=sys.stderr,
              flush=True)
        if row_stream is not None:
            # The load-all head leaves the journal manifest and the export
            # inputs on this path, so the stream head does too.
            finalize_row_stream()
        return EXIT_EMPTY_MENU

    # An anchor waits in the ledger, rather than in ``measured``, for exactly
    # as long as its files do not exist.  That is what keeps the checkpoint
    # invariant true without changing it: every anchor row journalled here has
    # a wire receipt beside it, and every one of those receipts was read back
    # off a file that had already landed.
    # The writer itself is started INSIDE the try below, not here.  A thread
    # that exists before the region whose ``finally`` closes it is a thread
    # nothing joins if the setup between the two raises, and the seeded run's
    # adopted rows are flushed in that gap.  So the ledger is built as a
    # pass-through, the adopted flush runs inline exactly as it always did,
    # and the publisher is attached as the first act of the guarded region.
    publisher = None

    def journal_anchor(anchor, record) -> None:
        """Put one anchor row and its wire receipt into the pending state."""
        name = anchor.qname
        wire_records[name][anchor.format_name] = record
        measured.setdefault(name, {}).setdefault(
            anchor.family, []).append(anchor)
        dirty_checkpoint_units.add(name)
        if journal is None and stream_journal is not None:
            dirty_stream_units.add(name)

    ledger = _AnchorPublicationLedger(
        publisher=None,
        make_record=lambda anchor, identity: campaign_wire.checkpoint_wire_record(
            anchor, wire_dir, identity),
        journal_anchor=journal_anchor)

    def committed_anchors() -> int:
        """Every anchor the run holds, each with a receipt read off its landed file.

        This is the count to report, not the anchors one flush touched. A
        resumed row continues from the shards its journal already has, and a
        counter that restarted at zero would look like a regression to the
        watchdog, which refuses it.
        """
        return sum(len(anchors) for by_format in measured.values()
                   for anchors in by_format.values())

    def unit_state(name) -> dict:
        """One unit's journal state: its anchor rows and their wire receipts.

        Snapshotted on the thread that owns the rows; only the write is
        ordered behind the receipts it cites. ``vars`` returns the anchor's
        live ``__dict__``, so it is copied rather than handed over.
        """
        state = {"anchors": [dict(vars(anchor))
                             for anchors in measured.get(name, {}).values()
                             for anchor in anchors],
                 "wire_records": dict(wire_records[name])}
        if unservable.get(name):
            state["unservable"] = dict(unservable[name])
        return state

    def flush_checkpoint() -> None:
        if journal is None:
            # The stream head before finalize: the checkpoint shards cite a
            # run identity that does not exist yet, so the dirty units stay
            # dirty until then. Their rows go to the stream journal now, each
            # with the receipts its entry's reader took, so a relaunch can
            # adopt them (PQ #1403). Progress is reported after that write and
            # only for rows it holds: before it, PB was told of anchors no
            # restart could reuse. Staying silent until finalize instead held
            # a 1728-anchor row in ``startup`` for its whole encode, and PB
            # killed it with 1608 anchors on disk (PQ #1362, PB #480).
            states = [(name, {**unit_state(name),
                              "stream_receipts": row_stream.unit_identity(name)})
                      for name in sorted(dirty_stream_units)]
            dirty_stream_units.clear()
            committed = committed_anchors()

            def write_stream():
                for name, state in states:
                    write_unit(stream_journal, stage=STREAM_JOURNAL_STAGE, qname=name,
                               identity_sha256=stream_identity_sha256, state=state)
                report_progress("pricing", committed)

            ledger.submit_checkpoint(write_stream)
            return
        states = [(name, unit_state(name)) for name in sorted(dirty_checkpoint_units)]
        dirty_checkpoint_units.clear()
        if not states:
            return

        committed = committed_anchors()

        def write():
            for name, state in states:
                write_unit(journal, stage="Tessera campaign", qname=name,
                           identity_sha256=identity_sha256, state=state)
            # After the shards are on disk, never before.  This is what tells
            # PrismaBuild the row is working rather than merely running, and
            # it must not be able to say so on behalf of work that has not
            # landed (PB #480).
            report_progress("pricing", committed)

        ledger.submit_checkpoint(write)

    # Adopted rows are journalled BEFORE the anchor loop, because the loop can
    # end without reaching its own flush: a group whose gate is already closed
    # has nothing pending in round one and breaks out.  A seeded run is exactly
    # that case, so without this the units a seed adopted would price into
    # ``cost.pkl`` and leave no journal shard for ``merge`` to carry.
    if dirty_checkpoint_units:
        flush_checkpoint()

    started = time.time()
    deadline = float(publication.deadline_seconds)
    stopped_early = False
    # Every (unit, format) whose last attempt failed to price, with its error.
    # The loop keeps pricing past a failure, so the round's successes are
    # journalled for the retry and a later round may still price the rung;
    # whatever is left here fails the row before it writes a table (#1481).
    anchor_failures = {}

    def out_of_time() -> bool:
        return deadline > 0 and (time.time() - started) > deadline

    # The measured ENVELOPE per (unit, family): the rung range anchors span.
    # Capped by --max-artifact-bpp, which is a wall-clock budget and says so:
    # the menu keeps every legal rung, and the ones above the envelope are
    # left unpriced rather than extrapolated into.
    cap = float(recipe.max_artifact_bpp)
    rates_by_unit: dict[str, dict[str, set[int]]] = {}
    rate_menu_targets = partition_menu_targets or targets
    rate_menus = full_partition_menus if partition_menu_targets is not None else menus
    for name in rate_menu_targets:
        per_family: dict[str, set[int]] = {}
        for rung in rate_menus[name]:
            if cap > 0 and rung.bpp > cap:
                continue
            per_family.setdefault(rung.family, set()).add(rung.body_rate_q256)
        rates_by_unit[name] = per_family

    # Anchors are placed per FUSED GROUP, not per (unit, family).
    #
    # The cap is a WIRE bpp and the wire->body map is shape-dependent (the
    # CHANNEL plane amortises over rows), so solving each member's top anchor
    # independently put fused siblings on different body grids even when they
    # share ``in_features`` and therefore share the realisable set: on
    # Qwen3-0.6B ``q_proj`` topped out at R1388 where ``k/v_proj`` topped out
    # at R1372, and every bisected anchor below inherited the offset. A
    # measured-only table then has almost nothing in the intersection of a
    # group's menus -- the qkv E4M3 intersection was exactly one rung -- which
    # made the group's measured baseline weak for a reason that has nothing to
    # do with Tessera. One grid per group, from the intersection of its
    # members' realisable sets, and every member measures the same rungs.
    def _group_key(name: str) -> str:
        return anchor_group_key(name, profile=profile, expert_members=expert_members)

    anchor_groups = resolve_anchor_groups(
        targets, profile=profile, expert_members=expert_members)

    # The rungs every member of the group can realise, per family. A family
    # missing from one member is not a group family at all: a shared grid over
    # a rung one sibling cannot build is not shared.
    group_rates: dict[str, dict[str, list[int]]] = {}
    # Rungs a member's serving structure has no attested wire for (#1502),
    # read off the pinned contract: kept in the menu, not measured, and
    # recorded here with the contract's reason rather than hidden.
    route_refused: dict[str, dict[str, dict[str, dict]]] = {}
    rate_groups = ({entry["key"]: entry["members"] for entry in selection["groups"]}
                   if partition_menu_targets is not None else anchor_groups)
    group_rates, route_refused = anchor_group_rate_grids(
        rate_groups, rates_by_unit,
        encode_structure=(partition_encode_structure if partition_menu_targets is not None else encode_structure),
        projected_units=(partition_unit_records if partition_menu_targets is not None else projected_units),
        route_cache=route_cache)
    if partition_menu_targets is not None:
        fixed_rate = parse_rate_band(placement.rate_band)[0]
        if any(not any(fixed_rate in rates for rates in families.values())
               for families in group_rates.values()):
            raise RuntimeError("expert partition pinned rate has no legal full-group family/grid")
    if route_refused:
        print("[campaign] route-refused rungs (kept in the menu, not measured): "
              + ", ".join(f"{key}:{family}@{sorted(int(r) for r in rungs)}"
                          for key, by_family in sorted(route_refused.items())
                          for family, rungs in sorted(by_family.items())), flush=True)

    print("[campaign] anchor groups: "
          + ", ".join(
              f"{key}({len(members)})" for key, members
              in sorted(anchor_groups.items())), flush=True)

    def _snap(rate: int, allowed: Sequence[int]) -> "int | None":
        """The realisable rung nearest ``rate``, or None if there is none.

        ``anchor_schedule`` and ``next_anchor_rate`` both work in rate space
        and can name a q256 no member can build; snapping keeps the group's
        grid shared, which is the whole point of placing anchors per group.
        """
        if not allowed:
            return None
        return min(allowed, key=lambda r: (abs(int(r) - int(rate)), int(r)))

    round_one_schedule = _campaign_round_one_schedule(anchor_groups, group_rates,
        placement=placement, audit_units=audit_units, snap=_snap,
        requests=None if acquisition is None else acquisition["requests"], rates_by_unit=rates_by_unit)

    # Round 1, breadth-first over units, so a deadline yields every unit priced
    # at the same depth rather than a prefix priced deeply and a tail not at all.
    budget = int(recipe.anchor_budget)
    # Why the group's own worst member drives the split: the grid is shared,
    # so a rung added for one sibling is measured for all of them anyway. The
    # gate closes for a GROUP surface only when it closes for every member.
    surface_stop: dict[tuple[str, str], str] = {}
    rate_band = parse_rate_band(placement.rate_band)
    if rate_band is not None:
        print(f"[campaign] rate band q256 {rate_band[0]}..{rate_band[1]}: two "
              "anchors per window family at the band ends"
              + (f", a third for {len(audit_units)} audit unit(s)"
                 if audit_units else ""), flush=True)
    # The publisher owns a thread; every exit from the pricing loop,
    # including one that raises, has to go past its close. Close drains
    # what is queued, so files already computed still land, and it cannot
    # raise, so it cannot mask the exception that brought us here.
    try:
        if int(publication.overlap_bytes) > 0:
            from .tessera_publication import BoundedPublisher
            publisher = BoundedPublisher(
                budget_bytes=int(publication.overlap_bytes))
            ledger.open(publisher)
            print(f"[campaign] publication overlap: staging up to "
                  f"{int(publication.overlap_bytes)} bytes on one writer "
                  "thread", flush=True)

        round_index = 0
        while True:
            round_index += 1
            if int(placement.max_rounds) > 0 and round_index > int(placement.max_rounds):
                print(f"[campaign] --max-rounds {placement.max_rounds} reached",
                      flush=True)
                break
            pending: list[tuple[str, str, int]] = []
            for key, members in sorted(anchor_groups.items()):
                for family, allowed in group_rates[key].items():
                    # The grid is what EVERY member actually measured: one
                    # member's failed encode must not let the group think it has
                    # an anchor there.
                    member_rates = {
                        m: {a.body_rate_q256 for a in measured.get(m, {}).get(family, [])}
                        for m in members}
                    grid = sorted(set.intersection(*member_rates.values())) if members else []
                    if round_index == 1:
                        for m in members:
                            rates = set(round_one_schedule[m].get(family, []))
                            have = member_rates[m]
                            pending.extend((m, family, rate)
                                           for rate in sorted(rates - have))
                        continue
                    if len(grid) >= budget:
                        surface_stop.setdefault((key, family), "anchor_budget")
                        continue
                    worst = 0.0
                    worst_loo: dict = {}
                    ready = True
                    interpolable = 0
                    for m in members:
                        anchors = measured.get(m, {}).get(family, [])
                        if len(anchors) < 3:
                            ready = False
                            break
                        member_loo = _loo_for(anchors, leave_one_anchor_out)
                        if _loo_refused(member_loo):
                            # A refused surface has no leave-one-out error, and a
                            # missing error is not a zero one.  Reading it as 0.0
                            # would say "this surface interpolates perfectly" about
                            # the one surface that does not interpolate at all --
                            # and would close the gate on it.  It contributes
                            # nothing to ``worst`` instead, so the group keeps
                            # spending anchors for whichever members can still use
                            # them, and the refusal is reported as itself.
                            continue
                        interpolable += 1
                        value = float(
                            member_loo.get("max_abs_log2_error", 0.0) or 0.0)
                        if value > worst:
                            worst, worst_loo = value, member_loo
                    if not ready:
                        # LOO cannot judge two endpoints. Bootstrap an interior
                        # anchor through next_anchor_rate's widest-gap fallback;
                        # missing LOO is neither a closed gate nor a refusal.
                        worst_loo = {}
                    elif not interpolable:
                        surface_stop.setdefault((key, family), "non_interpolable")
                        continue
                    elif worst <= recipe.loo_gate:
                        surface_stop.setdefault((key, family), "gate_closed")
                        continue
                    nxt = next_anchor_rate(grid, worst_loo)
                    nxt = _snap(nxt, allowed) if nxt is not None else None
                    if nxt is None or nxt in grid:
                        surface_stop.setdefault((key, family), "no_room")
                        continue
                    # A failed sibling does not invalidate an already measured
                    # member/rung or authorize overwriting its original cost.
                    pending.extend((m, family, nxt) for m in members
                                   if nxt not in member_rates[m])
            if not pending:
                if round_index == 1 and measured:
                    # A resumed/seeded endpoint set still needs its adaptive gate
                    # checked; an empty bootstrap is not a completed surface.
                    continue
                print(f"[campaign] round {round_index}: nothing pending", flush=True)
                break
            print(f"[campaign] round {round_index}: {len(pending)} anchors",
                  flush=True)
            batches = _anchor_batches(
                [item for item in pending
                 if row_stream is not None or acts.get(item[0]) is not None],
                weights=weights, batch_size=publication.batch_size,
                structures=encode_structure)
            if row_stream is not None:
                row_stream.plan([[item[0] for item in batch] for batch in batches])
            completed = 0
            # One entry per encode step, in order, each the growth across that
            # step's own bracket. The list is stamped on the selected receipt now
            # and filled as the loop runs, so a reader gets the steady-state cost
            # separately from the first batch's one-time runtime charge.
            anchor_batch_growth = []
            if selected_source and selected_source_preparation is not None:
                selected_source_preparation['anchor_batch_growth_bytes'] = anchor_batch_growth
            for batch_index, batch in enumerate(batches):
                # Rows whose files landed while the last batch encoded. Applied at
                # the top of the batch rather than the bottom, so the receipt read
                # back off each published file runs one batch behind the write
                # instead of immediately after it.
                ledger.apply_completed()
                if out_of_time():
                    stopped_early = True
                    print("[campaign] deadline reached; stopping", flush=True)
                    # A deadline is a termination, so the staged bytes are written
                    # and journalled before the loop is left; they were paid for.
                    ledger.drain()
                    break
                if row_stream is not None:
                    # This batch's entries verified and resident, the next
                    # batch's reads started, everything else released. A
                    # refused entry raises here, outside the per-batch retry.
                    row_stream.admit(batch_index)
                names = [item[0] for item in batch]
                scoring_rows = (acts if row_stream is None else
                                {name: row_stream.entry(name).inputs for name in names})
                family, rung = batch[0][1:]
                fmt = f"{family}_R{rung}"
                batch_floor = None
                if selected_guard is not None:
                    # The lower bracket of the encode step. Without it the step's
                    # growth can only be read against whichever checkpoint
                    # happened to precede it, which is a different phase's charge,
                    # and the pair is taken PER OCCURRENCE because the first batch
                    # carries the runtime's one-time first-use cost while later
                    # batches are the steady state the plan actually charges for
                    # (RobTand/prismaquant#390).
                    selected_guard.check('before_selected_anchor_batch')
                    batch_floor = selected_guard.last[
                        'conservative_cgroup_plus_cuda_reserved_bytes']
                try:
                    common = dict(format_name=fmt, cache=cache, wire_dir=wire_dir,
                        activation_kwargs_for=(
                            ((_activation_kwargs_for if row_stream is None
                              else row_stream.encoder_kwargs) if want_h else None)),
                        hessian_required=want_h, publisher=publisher, profile=profile)
                    if len(batch) == 1:
                        name = names[0]
                        anchors = [_measure_anchor(
                            qname=name, weight=weights[name].to(device),
                            activations=scoring_rows[name].to(device),
                            static_input_scale=static_scales.get(name),
                            structure=(encode_structure or {}).get(name), **common)]
                    else:
                        anchors = _measure_anchor_batch(
                            qnames=names,
                            weights=[weights[name].to(device) for name in names],
                            activations=[scoring_rows[name].to(device) for name in names],
                            static_input_scales=static_scales,
                            structures=encode_structure, **common)
                except (HessianContractError, ActivationScaleContractError,
                        PublicationError, ServedQuantizerUnboundError):
                    # A staged artifact that did not reach its disk is not one
                    # batch's bad luck: the writer has stopped and everything
                    # behind it was dropped unwritten. Printing and continuing
                    # here would advance the loop past files that do not exist.
                    # A served quantiser that cannot bind refuses every A4 rung
                    # of the row, not one anchor (#1481), so it stops the row too.
                    raise
                except Exception as exc:
                    if selected_guard is not None and selected_guard.failure is not None:
                        raise  # A physical memory refusal must stop the action.
                    # A batch that raises part way through has already handed
                    # the writer files for its early units. Nothing records
                    # those units, so no receipt job follows and nothing is
                    # staged; their file completions are ignored, the bytes
                    # land, and the next round re-prices them over the same
                    # paths. That is what the synchronous path already does
                    # when a partial batch writes files and journals none of
                    # them, so the failure semantics do not change here.
                    #
                    # The failure is kept, not forgotten: the row fails before
                    # it writes a table (RobTand/prismaquant#1481).
                    print(f"[campaign] {names} {fmt}: FAILED {type(exc).__name__}: "
                          f"{exc}", flush=True)
                    for name in names:
                        anchor_failures[(name, fmt)] = f"{type(exc).__name__}: {exc}"
                    continue
                if selected_guard is not None:
                    # With publication staging on, this upper bracket includes the
                    # staged CPU bytes of any batch the writer has not finished.
                    # That is real resident memory the plan is charging for, so it
                    # belongs in the growth figure rather than being filtered out
                    # of it; the publisher's own budget and peak are stamped
                    # separately on the receipt so a reader can attribute it.
                    selected_guard.check('after_selected_anchor_batch')
                    anchor_batch_growth.append(selected_guard.last[
                        'conservative_cgroup_plus_cuda_reserved_bytes'] - batch_floor)
                for anchor in anchors:
                    anchor_failures.pop((anchor.qname, anchor.format_name), None)
                    if row_stream is not None:
                        # The reader's holder, over this entry's own source:
                        # a template copy on the encode thread, taken while
                        # the window still holds the entry and its holder.
                        entry = row_stream.entry(anchor.qname)
                        ledger.record(anchor, _checkpoint_anchor_identity(
                            anchor, weights=weights, menus=menus,
                            calibration_source=entry.source if want_h else None,
                            static_scales=static_scales, projected_units=projected_units,
                            bound_unit=entry.holder, profile=profile,
                            structure=(encode_structure or {}).get(anchor.qname)))
                        continue
                    identity_of = functools.partial(
                        _checkpoint_anchor_identity, anchor, weights=weights,
                        menus=menus, calibration_source=calibration_source,
                        static_scales=static_scales, projected_units=projected_units,
                        profile=profile,
                        structure=(encode_structure or {}).get(anchor.qname))
                    if bound_checkpoint_units:
                        # Derived from the unit's sealed template: a deepcopy
                        # and Tessera's wire_recipe, no tensor read.  The
                        # writer does it behind this anchor's own files, so
                        # the next batch's encode is not waiting on it.
                        ledger.record(anchor, derive=functools.partial(
                            identity_of, bound_unit=bound_checkpoint_units[anchor.qname]))
                    else:
                        # The producer's ``encoding_input_identity`` hashes
                        # the resident weight and Hessian, which are device
                        # tensors here: that stays on the encode thread.
                        ledger.record(anchor, identity_of())
                completed += len(anchors)
                # Commit every joined quantum before advancing. The scalar mode
                # keeps its existing ten-anchor flush cadence.
                if publication.batch_size > 1 or (completed - 1) % 10 == 0:
                    flush_checkpoint()
                    print(f"[campaign] r{round_index} {completed}/{len(pending)} "
                          f"batch={len(anchors)} {fmt} "
                          f"encode_seconds={sum(a.seconds for a in anchors):.3f}", flush=True)
            # The round is a consumer barrier: the next round reads ``measured``
            # to decide what is still pending, so nothing may still be in flight.
            ledger.drain()
            flush_checkpoint()
            if stopped_early:
                break
            if round_index > 1 and completed == 0:
                raise RuntimeError(
                    f"campaign adaptive round {round_index} made no progress: "
                    "all pending anchors failed; successful anchors are journaled. "
                    "Refusing to repeat unchanged work; retry after resolving the failure.")

        if row_stream is not None:
            # The identity-bound writes, now that every entry has a receipt;
            # then the shards the loop could not yet journal.
            finalize_row_stream()
            flush_checkpoint()

        if anchor_failures:
            # A rung that failed to price fails the row (RobTand/prismaquant#1481).
            # Writing the table anyway published it without those rungs, and
            # when every rung failed, an EMPTY table that exited 0 and was
            # counted as done. The successes are journalled above, so PB's
            # retry encodes only what is missing; no cost.pkl is written here.
            failed = sorted(anchor_failures.items())
            shown = "; ".join(f"{name} {fmt}: {error}" for (name, fmt), error in failed[:8])
            more = len(failed) - 8
            raise RuntimeError(
                f"campaign row failed to price {len(failed)} anchor(s); "
                f"successful anchors are journaled and no cost table "
                f"was written: {shown}" + (f"; and {more} more" if more > 0 else ""))

        # Finalization is quiet on purpose -- the last drain, the leave-one-out
        # checks and the cost payload commit nothing the journal counts -- so
        # the row says it has reached that phase and is bounded by the
        # finalization allowance its submission declared rather than by the
        # pricing loop's, which is much shorter.
        #
        # The drain below still calls ``flush_checkpoint``, which reports
        # "pricing" again after this.  Neither outcome shortens the allowance:
        # an unchanged count is refused as replayed, and a higher one is
        # accepted while the watchdog keeps finalize's grace, because the
        # allowance follows the furthest phase entered and never an earlier
        # name.  What such a record does change is the phase the observation
        # displays, so a late drain can read "pricing" while finalize governs.
        report_progress("finalize", committed_anchors())
        publication_stats = None
        if publisher is not None:
            # The last barrier, and where a writer failure nothing else looked at
            # is raised. Drain the receipts, journal whatever they made dirty,
            # then wait for that journal write too: the checkpoint is the last
            # thing published and it cites everything before it.
            ledger.drain()
            flush_checkpoint()
            ledger.drain()
            publication_stats = publisher.stats()

    finally:
        if publisher is not None:
            # The thread first, so nothing is still writing, and then the rows
            # for whatever it finished. On the way out of an exception this is
            # the batch that succeeded before the one that failed; on the
            # normal path everything is already journalled and both calls are
            # no-ops. Any second failure here is reported and dropped: it must
            # not replace the exception that brought us here.
            publisher.close()
            try:
                # Unconditional, and that is the point: ``close`` journals
                # only what the writer finished after the last
                # ``apply_completed``, but rows journalled BY that call are
                # in ``dirty_checkpoint_units`` and unwritten until a flush,
                # and the batch cadence may not have reached one. Flushing
                # only when ``close`` itself journalled something loses them.
                # ``flush_checkpoint`` returns on an empty set, so the normal
                # path still costs nothing.
                ledger.close()
                flush_checkpoint()
            except Exception as cleanup_error:  # noqa: BLE001
                print("[campaign] could not journal the completed anchors "
                      f"while unwinding: {type(cleanup_error).__name__}: "
                      f"{cleanup_error}", file=sys.stderr, flush=True)

    loo: dict[str, dict[str, dict]] = {}
    for name, by_family in measured.items():
        for family, anchors in by_family.items():
            if len(anchors) >= 3:
                loo.setdefault(name, {})[family] = _loo_for(
                    anchors, leave_one_anchor_out)
    loo_pre = loo

    provenance = {
        "provenance": {
            "menu_mode": mode,
            **({"acquisition": acquisition_origin,
                "acquisition_schedule": acquisition_schedule,
                "acquisition_source_weights": acquisition["source_weights"]} if acquisition is not None else {}),
            # How the artifacts were written, and what that cost. Absent means
            # the default: every render and wire published on the encode
            # thread before the next batch started. Present means one bounded
            # writer thread ran them alongside the following encode, and the
            # row says the budget, the peak charge and how long the encode
            # thread spent blocked on it -- which is the number that says
            # whether the bound was the limit or the disk was.
            "publication_overlap": publication_stats,
            # An interpreter-specific observation of the opt-in holder's
            # retained metadata. This is evidence for the future admission
            # contract, never an admitted bound.
            "campaign_identity_hold": (
                None if not bound_checkpoint_units else {
                    "units": len(bound_checkpoint_units),
                    "metadata_observed_bytes": identity_metadata_observed_bytes,
                    "metadata_bound_bytes": sum(identity_metadata_bounds.values()),
                    "planning_scratch_bytes": identity_planning_scratch_bytes,
                    "admission": "selected-resource-phase",
                }),
            # Units the mode admitted no rung for. Empty on a healthy run;
            # never absent, so a reader never has to guess whether the run
            # was asked the question.
            "no_admitted_rung": list(no_admitted_rung),
            # Where round one put the anchors, and (when the planner sampled)
            # which experts stand for their stack and under what inclusion
            # probability. This travels with the prices because an estimate
            # built from them is only unbiased if the reader knows the pi it
            # was drawn under; the packed draw records build the stack rows.
            "rate_band": (None if rate_band is None
                          else [int(rate_band[0]), int(rate_band[1])]),
            "unit_selection_sample": {
                "audit_units": sorted(audit_units),
                "inclusion_probability": {
                    name: float(inclusion_probability[name])
                    for name in sorted(inclusion_probability)},
            },
            **({"research_exact_member_scope": exact_member_scope}
               if exact_member_scope is not None else {}),
            "tp_degree": int(tp_degree),
            "model": str(model_path),
            "nsamples": int(draw.nsamples),
            "seqlen": int(draw.seqlen),
            "max_act_rows": int(draw.max_act_rows),
            "layer_stride": int(draw.layer_stride),
            **({PROJECTION_KEY: expert_projection}
               if expert_projection is not None else {}),
            "anchors_round_one": int(placement.anchors),
            "max_rounds": int(placement.max_rounds),
            "anchor_budget": int(recipe.anchor_budget),
            "rounds_run": int(round_index),
            "anchor_placement": "fused_group",
            "anchor_groups": {
                key: list(members)
                for key, members in sorted(anchor_groups.items())
            },
            # What this invocation MEASURED, out of what the scope resolves to.
            # A merge reads this to prove a set of shards covers the scope
            # exactly once; a whole-scope run selects every group and says so.
            # Rows this run did not encode itself, and where they came from.
            # None when every row was measured here.
            "seed_checkpoint": seed_provenance,
            "unit_selection": ({**selection, "selected": True} if selection else {
                "schema": UNITS_SCHEMA,
                "selected": True if request.selection_requested else False,
                "groups": [
                    {"key": key, "members": list(scope_groups[key])}
                    for key in sorted(selected_groups)
                ],
            }),
            # The scope every shard shares: the enumeration the population
            # block is built from, the full grouping, and the census the
            # Hessian identity's token counts came from.
            "campaign_scope": {
                "dense_targets": sorted(census_dense_targets),
                "expert_targets": sorted(census_expert_targets),
                "dense_all": sorted(all_dense),
                "pinned": sorted(pinned),
                "declared_stacks": {
                    stack: {name: list(shape) for name, shape in sorted(units.items())}
                    for stack, units in sorted(population.declared.items())},
                "packed_in_scope": {
                    name: list(shape) for name, shape
                    in sorted(population.packed_in_scope.items())},
                "packed_outside_layer_stride": {
                    name: list(shape) for name, shape
                    in sorted(population.omitted_outside_layer_stride.items())},
                "anchor_groups": {
                    key: list(members) for key, members in sorted(scope_groups.items())},
                "calibration_census": (
                    None if census is None else
                    {"counts": dict(census["counts"]),
                     "token_count": int(hessian_token_count),
                     "token_count_min": int(hessian_token_min)}),
            },
            # Per surface: what it cost and whether its gate closed. The
            # adaptive loop's whole purpose is to spend encodes where the
            # interpolation is measurably failing, so "how many anchors and
            # how many seconds did that take, and did it work" is the readout
            # that says whether the budget was the binding constraint.
            "surfaces": {
                name: {
                    family: {
                        "anchors": len(anchors),
                        "encode_seconds": round(
                            sum(float(a.seconds) for a in anchors), 3),
                        "rungs": sorted(a.body_rate_q256 for a in anchors),
                        # ``None``, never 0.0, when there is no leave-one-out
                        # error to report: a refused surface did not fit its
                        # own anchors perfectly, it failed to become a surface,
                        # and a zero here would read as the former.
                        "loo_max_abs_log2_error": _surface_loo(
                            loo_pre.get(name, {}).get(family),
                            float(recipe.loo_gate))[0],
                        "gate_closed": _surface_loo(
                            loo_pre.get(name, {}).get(family),
                            float(recipe.loo_gate))[1],
                        "non_interpolable": _loo_refused(
                            loo_pre.get(name, {}).get(family) or {}),
                        "stopped_by": (
                            "non_interpolable"
                            if _loo_refused(
                                loo_pre.get(name, {}).get(family) or {})
                            else surface_stop.get(
                                (_group_key(name), family), "round_limit")),
                    }
                    for family, anchors in sorted(by_family.items())
                }
                for name, by_family in sorted(measured.items())
            },
            # Adopted rows this run's menu does not admit: the rate/distortion
            # law keeps them, the allocator never sees them.  Always present,
            # so "is there evidence outside the priced menu" is a lookup and
            # not a distinction between an absent key and an empty one.
            "unservable": {
                name: {fmt: rows[fmt] for fmt in sorted(rows)}
                for name, rows in sorted(unservable.items())
            },
            # Menu rungs a group's serving structure has no attested wire
            # for on this pin (#1502): priced in the menu, never measured,
            # with the contract's reason.  Always present, like "unservable".
            "route_refused": {key: {family: dict(sorted(rungs.items()))
                                    for family, rungs in sorted(by_family.items())}
                              for key, by_family in sorted(route_refused.items())},
            "encode_structure": (None if encode_structure is None else {
                structure: sum(1 for value in encode_structure.values() if value == structure)
                for structure in sorted(set(encode_structure.values()))}),
            "loo_gate": float(recipe.loo_gate),
            "max_artifact_bpp": float(recipe.max_artifact_bpp),
            **({"family_restriction": {"policy": family_restriction,
                 "structure_by_unit": dict(sorted(structure_by_unit.items()))}}
               if family_restriction is not None else {}),
            "stopped_early": bool(stopped_early),
            "wall_seconds": time.time() - started,
            "cache_dir": str(cache_dir),
            **({"tessera_serving_scope": scope_provenance(serving_target, context_by_unit)}
               if serving_target is not None else {}),
            "wire_dir": str(wire_dir),
            "tessera_commit": os.environ.get("TESSERA_COMMIT", ""),
            # The static A-side identity every W4A4 row was priced under: the
            # policy names the formula, the values are the F32-rounded scalars
            # an exported input_global_scale tensor carries, fused siblings
            # unified. The served contract reads exactly one such scalar per
            # module (trellis_input_global_scale), so this block is what makes
            # "the priced A side is the served A side" checkable downstream.
            "calibration_cache": calibration_cache,
            **({"selected_source_preparation": dict(selected_source_preparation,
                  memory_guard=None if selected_guard is None else selected_guard.snapshot())}
               if selected_source else {}),
            "activation_static_scales": {
                "policy": str(static_scale_policy),
                "source": "campaign_calibration_amax_fused_unified",
                "path": (None if input_scales_path is None
                         else str(input_scales_path)),
                "units": {name: float(value)
                          for name, value in sorted(static_scales.items())},
            },
            "hessian": {
                "supplied": bool(want_h),
                "mode": str(request.encoding.hessian),
                "reason": str(hessian_status["reason"]),
                "consumed_by": (
                    "prismaquant.tessera_render.encode_tessera_unit"
                    + (" -> tessera.export.ActivationSource.for_unit -> "
                       "tessera.export.encode_linear_planes("
                       + ", ".join(f"{k}=" for k in hessian_status["kwargs"])
                       + ")" if want_h else "")),
                "kwargs": list(hessian_status["kwargs"]),
                "recipe": dict(hessian_status["recipe"]),
                # The exporter-shaped capture of the exact Hessians above,
                # for the export leg's --hessian input; None on --hessian off.
                "capture_path": (None if hessian_capture_path is None
                                 else str(hessian_capture_path)),
                **({'reference_binding':calibration_store.hessian_reference_binding(
                    calibration_cache['sha256'],capture_identity['census_sha256'])}
                   if outputs.hessian_reference_policy is not None else {}),
                # The content digest of that payload (Tessera's own seal
                # rule), stamped on every row so the allocation binds to
                # the capture BY CONTENT, not by the draw's triple alone
                # (RobTand/prismaquant#204); None on --hessian off.
                "capture_sha256": capture_sha256,
                "token_count": int(hessian_token_count),
                "token_count_min": int(hessian_token_min),
                # The calibration identity verbatim, under its own key, so a
                # merge can hand exactly this dict back to
                # ``write_export_inputs`` instead of reconstructing it by
                # subtracting the keys around it.
                "calibration_identity": dict(hessian_identity),
                # The identity triple Tessera requires, plus its context. The
                # legacy ``text_sha`` spelling is kept because older cost
                # tables carry it and ``assert_uniform_hessian_identity``
                # compares tables across runs.
                **dict(hessian_identity),
                "text_sha": hessian_identity["fit_ids_sha256"],
            },
        },
    }
    payload_menus = menus
    if acquisition_schedule is not None:
        # This acquisition prepares exactly requested measured wires; a dense
        # surface must not turn their interior into additional scalar prices.
        payload_menus = {name: [rung for rung in menus[name] if rung.body_rate_q256 in
            acquisition_schedule[name].get(rung.family, [])] for name in targets}
    payload = campaign_cost_payload(
        measured, payload_menus, loo=loo, provenance=provenance,
        wire_backed=frozenset(projected_units), stack_samples=stack_samples)
    # Empty menus, failed anchors and interrupted work do not establish a
    # price. Publish coverage only after the cost rows have been constructed.
    payload["provenance"][POPULATION_KEY] = campaign_population_block(
        dense_targets=dense_targets, expert_targets=expert_targets,
        dense_all=all_dense, pinned=pinned, population=population,
        layer_stride=int(draw.layer_stride), costs=payload["costs"], menus=menus,
        stack_samples=stack_samples, profile=profile)
    if projected_units:
        # The producer's receipts for every priced expert wire, keyed by unit
        # then rung; the allocator carries the selected rung's receipt into
        # the allocation and the export lane hands the bytes to the exporter
        # unchanged (PrismaQuant #183).
        payload[EXPERT_WIRES_KEY] = {
            name: {fmt: dict(record) for fmt, record in sorted(wire_records[name].items())}
            for name in sorted(projected_units) if wire_records.get(name)
        }
    if _SERVED_QUANTIZER_RECORD is not None:
        # Which build of the served activation quantiser priced this table's
        # served-contract rungs (RobTand/prismaquant#1481). Absent when the run
        # priced none, so an A8/A16 table's bytes are unchanged.
        payload["provenance"]["served_quantizer"] = dict(_SERVED_QUANTIZER_RECORD)
    payload["menu_sizes"] = {n: len(m) for n, m in menus.items()}
    payload["anchor_counts"] = {
        n: {f: len(a) for f, a in by_f.items()} for n, by_f in measured.items()
    }
    total = sum(len(rows) for rows in payload["costs"].values())
    if total == 0 and any(menus.values()):
        # A row whose menu admitted rungs but priced none, with no rung raising
        # (a --max-artifact-bpp cap below every rung, or a deadline before the
        # first anchor), would publish a table of 0 priced rungs with rc 0, and
        # PrismaBuild would count it as done (RobTand/prismaquant#1481). An
        # empty menu refuses earlier with EXIT_EMPTY_MENU; this is the other
        # half. Nothing is written.
        raise RuntimeError(
            f"campaign row priced no rung out of a non-empty menu "
            f"({sum(len(m) for m in menus.values())} admitted rungs over "
            f"{sum(1 for m in menus.values() if m)} units); no cost table was written")
    Path(outputs.out).parent.mkdir(parents=True, exist_ok=True)
    # Canonical bytes: a resumed or adopted row is an equal but distinct object
    # from the one a fresh encode shares, and plain ``pickle.dump`` would make
    # the file's digest depend on which path built it (PQ #1403).
    from .digests import canonical_pickle_bytes
    Path(outputs.out).write_bytes(canonical_pickle_bytes(payload))
    print(f"[campaign] wrote {outputs.out}: {len(payload['costs'])} units, "
          f"{total} priced rungs, {len(payload['formats'])} distinct formats",
          flush=True)
    return 0


def _loo_refused(member_loo) -> bool:
    """Did the surface refuse to exist at all?

    :func:`_loo_for` reports a refusal as an ``error`` key.  Everywhere else
    reads ``max_abs_log2_error`` with a default, so the distinction between
    "fits perfectly" and "is not a surface" has to be asked for explicitly.
    """

    return bool(member_loo) and "error" in member_loo


def _surface_loo(member_loo, gate: float):
    """``(loo_error, gate_closed)`` for one surface, refusal-aware."""

    if not member_loo or _loo_refused(member_loo):
        return None, False
    value = float(member_loo.get("max_abs_log2_error", 0.0) or 0.0)
    return value, bool(value <= gate)


def _loo_for(anchors, leave_one_anchor_out) -> dict:
    from .tessera_rate_surface import TesseraRateSurface

    ordered = sorted(anchors, key=lambda a: a.body_rate_q256)
    try:
        surface = TesseraRateSurface(
            unit_name=ordered[0].qname,
            family=ordered[0].family,
            layout="tight",
            currency=CURRENCY,
            anchor_q256=tuple(a.body_rate_q256 for a in ordered),
            anchor_dloss=tuple(a.dloss for a in ordered),
            anchor_stderr=tuple(a.dloss_stderr for a in ordered),
        )
    except Exception as exc:
        return {"error": str(exc)}
    return dict(leave_one_anchor_out(surface))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
