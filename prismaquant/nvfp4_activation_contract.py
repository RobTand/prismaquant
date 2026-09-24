"""Single-owner NVFP4 W4A4 activation execution contract.

The compressed-tensors tensor ABI already names the calibrated scalar
``<target>.input_global_scale``.  PrismaQuant reuses that exact tensor for
FP4-CB instead of inventing a second spelling.  This module owns everything
that is not expressible by the compressed-tensors scheme itself:

* the versioned execution-contract and scale-policy identities;
* calibrated max-abs -> input-global-scale conversion;
* fused-sibling scale unification;
* the canonical mapping digest stamped into ``quant_config.json``; and
* the serve-faithful activation QDQ oracle used by producer tests/costs.

Old CB artifacts can lack both the contract record and scalar tensors; legacy
native artifacts may carry an unversioned/defaultable scalar.  Both remain
readable by their baseline paths, but neither is eligible for Gridbook fused
W4A4 dispatch.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import struct
from typing import Any

import torch


NVFP4_ACTIVATION_CONTRACT_KEY = "nvfp4_w4a4"
NVFP4_ACTIVATION_CONTRACT_SCHEMA = (
    "prismaquant.nvfp4_w4a4_activation.v1"
)
NVFP4_ACTIVATION_EXECUTION = "e2m1_group16_ue4m3_static"
NVFP4_INPUT_GLOBAL_SCALE_SUFFIX = "input_global_scale"

# Public numerical/compatibility constants.  Exporters and loaders must import
# these rather than grow a second activation-scale convention.  In particular,
# the uncalibrated value is a legacy compressed-tensors compatibility fallback;
# a versioned Gridbook activation contract never uses it implicitly.
UNCALIBRATED_INPUT_GLOBAL_SCALE = 1.0
FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
FP4_GROUP_SIZE = 16

#: The two arithmetic owners a priced activation row can have.  The name IS the
#: identity a receipt carries: a row stamped ``registered_scaled_fp4_quant`` was
#: priced by the vLLM operator a serve executes, one stamped
#: ``prismaquant_model`` by this module's Torch re-implementation of the same
#: rounding rule.  They are not interchangeable, and a served rung may only be
#: priced by the first (RobTand/prismaquant#567).
SERVED_QUANTIZER_BACKEND_MODEL = "prismaquant_model"
SERVED_QUANTIZER_BACKEND_REGISTERED_OP = "registered_scaled_fp4_quant"
#: The operator's registered name, spelled once.  Its home is the ``_C``
#: namespace vLLM's extension registers into; nothing here imports Tessera.
SERVED_QUANTIZER_OP = "scaled_fp4_quant"
#: The implementation of the registered-operator leg's dequantisation: the
#: operator decides the codes, and these two Triton kernels
#: (``prismaquant/kernels/nvfp4_served_dequant.py``) take each group's maximum
#: and dequantise the codes under the contract's scale rule.  Named in
#: ``ServedQuantizerIdentity.dequant_kernel`` so a row states which
#: implementation priced it; a process that cannot load the kernels cannot bind
#: the registered-operator arithmetic (RobTand/prismaquant#1211).
SERVED_QUANTIZER_DEQUANT_KERNEL = "prismaquant.triton_nvfp4_served_dequant.v1"
SERVED_QUANTIZER_IDENTITY_SCHEMA = (
    "prismaquant.served_quantizer_identity.v1"
)

LEGACY_INPUT_GLOBAL_SCALE_POLICY = (
    "legacy_6_over_calibration_amax.v1"
)
FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY = (
    "full_e4m3_range_448x6_over_calibration_amax.v1"
)
MSE_GRID_INPUT_GLOBAL_SCALE_POLICY = "mse_grid_calibrated.v1"
#: The key the campaign's ``input_scales.safetensors`` carries its policy under
#: in the container's ``__metadata__``, and the key the export gate reads it
#: back from.  Spelled once so the writer and the reader cannot drift: a scale
#: file whose label and whose values were produced by two different policies is
#: exactly what RobTand/prismaquant#624 asks a gate to refuse.
INPUT_GLOBAL_SCALE_POLICY_METADATA_KEY = "input_global_scale_policy"
NVFP4_INPUT_GLOBAL_SCALE_POLICIES = frozenset({
    LEGACY_INPUT_GLOBAL_SCALE_POLICY,
    FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
    MSE_GRID_INPUT_GLOBAL_SCALE_POLICY,
})

# ---------------------------------------------------------------------------
# Executed activation-scale grouping (RobTand/prismaquant#624)
# ---------------------------------------------------------------------------
# What a static ``input_global_scale`` MAPS onto is a second question beside
# the policy that produced each value.  PrismaQuant prices one scale per unit,
# so on the Tessera routed NVFP4 wire it prices one per EXPERT projection.
# The routed stage it would execute on takes one activation scale per
# ``(module, stage)`` -- the minimum ``input_global_scale`` of the group, w13
# and w2 separately.  Scope of that statement: vLLM FLASHINFER_CUTLASS behind
# Tessera #507's ``nvfp4_moe_route``, read from source and from the probe
# receipt ``nvfp4_moe_oracle_probe_spark_a5424378.json``.  It is prose, and no
# gate reads it; the design note
# ``docs/design/routed_executed_scale_grouping_2026-09-14.md`` carries the
# citations.
#
# ``per_unit.v1`` is the only grouping this producer prices or exports.  An
# allocation that carries per-expert static scales says so EXPLICITLY, and the
# export gate records it as not qualified.  The executed (collapsed) grouping
# has no identity here yet on purpose: it needs rescored rows and an attested
# runtime table, and naming it before either exists would invite a stamp with
# nothing behind it (RobTand/prismaquant#624).
ACTIVATION_SCALE_GROUPING_PER_UNIT = "per_unit.v1"
#: The grouping a routed stage actually EXECUTES: one activation scale per
#: ``(module, stage)``.  Named here only now, with a producer and a consumer,
#: because #624 deliberately refused to name it while "it needs rescored rows
#: and an attested runtime table, and naming it before either exists would
#: invite a stamp with nothing behind it".  The producer is
#: :func:`routed_executed_max_abs`; the consumer is the joint AURA activation
#: leg, which must price the scale the kernel executes rather than the per-unit
#: scale the campaign priced.
ACTIVATION_SCALE_GROUPING_EXECUTED = "executed_group.v1"
ROUTED_EXECUTED_SCALE_GROUPING_SCHEMA = (
    "prismaquant.routed_executed_scale_grouping.v1"
)


def is_routed_expert_projection_name(name: str) -> bool:
    """Whether ``name`` has the per-expert routed spelling ``<parent>.experts.<e>.<leaf>``.

    A SHAPE predicate only, deliberately separate from
    :func:`routed_expert_scale_group`: a name with this shape whose group cannot
    be resolved is refused by the callers, never read as dense.
    """

    parts = str(name).split(".")
    return len(parts) >= 4 and parts[-3] == "experts" and parts[-2].isdigit()


def routed_expert_scale_group(
    name: str,
    *,
    profile=None,
) -> tuple[str, str, str] | None:
    """``(group_key, module_prefix, stage)`` for one PER-EXPERT projection.

    Membership is not restated: the name is respelled as the PACKED target
    :func:`routed_moe_stage` already owns and that owner is asked, so the
    profile hook, the leaf table and the stage map live in one place.

    ``None`` for any name without the per-expert spelling (dense units, and the
    native packed spelling ``<parent>.experts.<packed-parameter>``, which
    already carries one scale for the whole stack), AND for a per-expert name
    whose leaf, role or stage is unknown.  The second case is not "not routed":
    callers test :func:`is_routed_expert_projection_name` first and refuse it
    (#624).
    """

    if not is_routed_expert_projection_name(name):
        return None
    parts = str(name).split(".")
    packed_leaf = _PACKED_LEAF_ROLES.get(parts[-1])
    if packed_leaf is None:
        return None
    parsed = routed_moe_stage(
        f"{'.'.join(parts[:-3])}.experts.{packed_leaf}", profile=profile)
    if parsed is None:
        return None
    module, stage = parsed
    return f"{module}::{stage}", module, stage


def grouping_member(declaration, qname):
    """The declared executed group one unit belongs to, or ``None``."""
    if not isinstance(declaration, Mapping):
        return None
    for key, entry in (declaration.get("groups") or {}).items():
        if qname in entry.get("members", ()):
            return str(key), entry
    return None


def executed_static_max_abs(*, spec, qname, unit_max_abs, grouping):
    """``(effective_max_abs, provenance)`` for one unit's static A-scale.

    The ONE resolution both the activation receipt and the arithmetic consult,
    so a price and the quantizer it was measured through cannot disagree.

    A spec with no static activation contract -- FP8/MX dynamic W8A8, or an
    A16 identity row that reached the hook -- is returned unchanged with no
    provenance: there is no static scale, and the reduction has nothing to say
    about it.  This is a FORMAT decision, not a name-shape decision, which is
    what keeps A8 clipping and A16 identity byte-identical.

    A static-contract unit inside a declared group takes the group maximum --
    the value the routed stage executes -- and returns provenance naming the
    group, so a receipt can say which scale it used and which one the historical
    wire was rendered under.  A static-contract unit NOT in any declared group
    keeps its own per-unit maximum: that is the dense NVFP4 case, where the
    per-unit scale IS the executed scale because there is no group to reduce.
    """

    contract = getattr(spec, "static_activation_contract", None)
    if contract is None:
        return unit_max_abs, None
    found = grouping_member(grouping, qname)
    if found is None:
        # A routed per-expert unit whose contract is the SERVED static one has
        # no per-expert scale the artifact can execute.  Reading the unit
        # maximum here is exactly the defect: it silently reverts a rostered
        # run to per-expert pricing.  Dense per-unit scales stay valid (there is
        # no group to reduce), and a generic stock NVFP4 screen contract keeps
        # its historical path, so only the measured-as-served routed case
        # refuses.
        if (getattr(contract, "measured_as_served", False)
                and is_routed_expert_projection_name(qname)):
            raise ValueError(
                f"{qname}: the served static activation contract executes one "
                "scale per (module, stage), so this unit requires a "
                "roster-checked executed grouping declaration; pricing its own "
                "per-unit maximum would use a scale the stage never executes")
        return unit_max_abs, None
    key, entry = found
    if not (grouping or {}).get("roster_complete"):
        raise ValueError(
            f"{qname}: the executed grouping for group {key!r} is not marked "
            "roster-complete, so its maximum is not authoritative")
    # The consumer recomputes G from this maximum with
    # ``require_input_global_scale(maximum)``, which resolves its policy from
    # the ambient environment.  A declaration naming one policy while the
    # active policy is another would let the producer stamp the right G and the
    # consumer price a different one.  Fail closed here, at the ONE shared
    # resolution both sides consult, before any residual exists.
    declared = (grouping or {}).get("input_global_scale_policy")
    if declared is None:
        raise ValueError(
            f"{qname}: the executed grouping for group {key!r} declares no "
            "input_global_scale policy; a missing policy cannot be checked "
            "against the active one")
    if declared not in NVFP4_INPUT_GLOBAL_SCALE_POLICIES:
        raise ValueError(
            f"{qname}: the executed grouping declares unknown "
            f"input_global_scale policy {declared!r}")
    active = resolve_input_global_scale_policy()
    if declared != active:
        raise ValueError(
            f"{qname}: the executed activation-scale grouping declares policy "
            f"{declared!r} but the active calculation policy is {active!r}; "
            "refusing to price a scale the receipt does not describe")
    expected = contract.input_global_scale_from_max_abs(
        float(entry["max_abs"]), policy=active)
    if "input_global_scale" not in entry:
        raise ValueError(
            f"{qname}: the executed grouping for group {key!r} carries no "
            "declared group scale to check against the active policy")
    if float(entry["input_global_scale"]) != float(expected):
        raise ValueError(
            f"{qname}: the declared group scale {entry['input_global_scale']!r} "
            f"is not the active policy's value {expected!r} for group maximum "
            f"{entry['max_abs']!r}")
    return float(entry["max_abs"]), {
        "schema": ROUTED_EXECUTED_SCALE_GROUPING_SCHEMA,
        "grouping": ACTIVATION_SCALE_GROUPING_EXECUTED,
        "group": key,
        "module": entry.get("module"),
        "stage": entry.get("stage"),
        "group_max_abs": float(entry["max_abs"]),
        "unit_max_abs": None if unit_max_abs is None else float(unit_max_abs),
        "group_members": list(entry.get("members", ())),
        "input_global_scale_policy": (
            (grouping or {}).get("input_global_scale_policy")),
        "source": "routed_executed_scale_grouping"
                  + (".roster_checked" if (grouping or {}).get("roster_complete")
                     else ".roster_unchecked"),
    }


def executed_static_max_abs_map(*, specs_by_qname, shared_max_abs, grouping):
    """``{qname: effective_max_abs}`` for every static unit in ``specs_by_qname``.

    Built once and handed to both the receipt and the arithmetic.  ``shared``
    per-unit maxima stay the caller's own map: this returns a derived view and
    mutates nothing.
    """

    out = {}
    for qname, choices in specs_by_qname.items():
        # A unit carries one spec per candidate format; the static contract is
        # a property of the format, so the unit is static if ANY of its
        # candidates is.  A unit whose candidates are all dynamic (A8) or
        # identity (A16) stays out of the map and hears nothing from this.
        pool = list(choices.values()) if isinstance(choices, Mapping) else [choices]
        spec = next((item for item in pool
                     if getattr(item, "static_activation_contract", None) is not None),
                    None)
        if spec is None:
            continue
        unit_max = shared_max_abs.get(qname)
        effective, _provenance = executed_static_max_abs(
            spec=spec, qname=qname, unit_max_abs=unit_max, grouping=grouping)
        if effective is not None:
            out[qname] = effective
    return out


def routed_static_scale_grouping(unit_names, *, profile=None) -> dict | None:
    """The grouping declaration for a set of static-scale units, or ``None``.

    ``None`` when no unit has the per-expert routed spelling: dense and native
    packed scales are one per executed tensor already, and they carry no
    declaration.  Otherwise ``per_unit.v1`` -- the only grouping the campaign
    prices -- stated explicitly so the export gate reads an answer, not an
    absence.  A per-expert name that does not resolve to a ``(module, stage)``
    group raises: it cannot be declared anything.

    The one producer of the declaration; the allocator
    (``tessera_menu.priced_static_scales``) and selected-wire completion
    (``tessera_materialization``) both call it (#624).
    """

    routed = False
    for name in sorted(unit_names):
        if not is_routed_expert_projection_name(name):
            continue
        if routed_expert_scale_group(name, profile=profile) is None:
            raise ValueError(
                f"{name}: per-expert routed spelling, but no routed-MoE "
                "(module, stage) group resolves for it, so its executed "
                "activation-scale grouping cannot be declared "
                "(RobTand/prismaquant#624)")
        routed = True
    if not routed:
        return None
    return {"schema": ROUTED_EXECUTED_SCALE_GROUPING_SCHEMA,
            "grouping": ACTIVATION_SCALE_GROUPING_PER_UNIT}


def routed_executed_max_abs(
    max_abs: "Mapping[str, float]",
    *,
    profile=None,
    expected_members=None,
    policy=None,
) -> "tuple[dict[str, float], dict | None]":
    """Collapse per-expert routed maxima onto the groups the kernel EXECUTES.

    ``(effective, declaration)``.  The routed stage takes one activation scale
    per ``(module, stage)`` -- ``min_e G_e``, equivalently the largest ``amax``
    of the group, w13 and w2 separately
    (``tessera/serving/nvfp4_moe_route.py`` ``gs13 = 1 / input_small["w13"].max()``
    and the same reduction in vLLM ``amax_for_moe_activation_quant``).  This
    function is that reduction on the calibration side, so a joint price stops
    reading each expert's own scale where the artifact cannot execute it.

    Membership is not restated: each per-expert name is respelled as the PACKED
    target :func:`routed_moe_stage` already owns, through
    :func:`routed_expert_scale_group`, so gate_proj and up_proj of one expert
    land in ONE ``w13`` group and down_proj lands in its ``w2`` group.  The
    profile hook, the leaf table and the stage map stay in one place.

    Dense names, and native packed names (``<parent>.experts.<packed-parameter>``
    -- already one scale for the whole stack), pass through byte-identically.
    ``declaration`` is ``None`` when nothing was routed, so a dense-only set
    keeps emitting no declaration at all rather than an empty one.

    Fails closed, in the order the checks bind:

    * a value that is not a finite positive number: a static-scale maximum of
      zero or NaN is not a maximum, and pricing it would divide by it;
    * a per-expert routed name whose ``(module, stage)`` group does not
      resolve: it cannot be grouped, and reading it as dense would price a
      scale the stage never executes;
    * a group whose members disagree about ``(module, stage)`` -- impossible
      through this path, and refused rather than smoothed if it ever occurs.
    """

    import math

    if not isinstance(max_abs, Mapping):
        raise ValueError("executed activation-scale grouping needs a mapping")
    values: dict[str, float] = {}
    for name, raw in max_abs.items():
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{name}: executed activation-scale grouping needs a finite "
                f"positive maximum, got {raw!r}")
        values[str(name)] = value
    # The policy is bound HERE, explicitly, and used both for the arithmetic
    # and for the stamp.  Resolving it per call site instead would let a
    # declaration name one policy while the scale was computed under another.
    canonical_policy = (resolve_input_global_scale_policy(policy)
                        if policy is not None else None)
    groups: dict[str, dict] = {}
    for name in sorted(values):
        parsed = routed_expert_scale_group(name, profile=profile)
        if parsed is None:
            if is_routed_expert_projection_name(name):
                raise ValueError(
                    f"{name}: per-expert routed spelling, but no routed-MoE "
                    "(module, stage) group resolves for it, so the scale the "
                    "kernel executes cannot be derived (RobTand/prismaquant#624)")
            continue
        group_key, module, stage = parsed
        entry = groups.setdefault(group_key, {"module": module, "stage": stage,
                                              "members": []})
        if (entry["module"], entry["stage"]) != (module, stage):
            raise ValueError(
                f"{name}: group {group_key!r} already names "
                f"{entry['module']!r}/{entry['stage']!r}, not {module!r}/{stage!r}")
        entry["members"].append(name)
    effective = dict(values)
    if groups and canonical_policy is None:
        raise ValueError(
            "routed executed activation-scale grouping requires an explicit "
            "input_global_scale policy; resolving it from the ambient "
            "environment would let the stamp and the arithmetic disagree")
    for group_key, entry in sorted(groups.items()):
        members = sorted(entry["members"])
        group_max = max(values[name] for name in members)
        entry["members"] = members
        entry["max_abs"] = float(group_max)
        entry["input_global_scale"] = (
            NVFP4_SERVED_ACTIVATION_CONTRACT.input_global_scale_from_max_abs(
                group_max, policy=canonical_policy))
        entry["spread_ratio"] = float(group_max / min(values[name] for name in members))
        for name in members:
            effective[name] = float(group_max)
    if expected_members is not None:
        # The authoritative full roster.  Grouping only the maxima that happen
        # to be present would let one missing expert silently redefine a group:
        # the survivor's value becomes the group maximum, the spread shrinks,
        # and no gate notices.  Compare group membership against the roster
        # rather than trusting the observed subset.
        expected: dict[str, set] = {}
        for name in sorted(str(item) for item in expected_members):
            parsed = routed_expert_scale_group(name, profile=profile)
            if parsed is None:
                if is_routed_expert_projection_name(name):
                    raise ValueError(
                        f"{name}: roster member has the per-expert routed "
                        "spelling but no (module, stage) group resolves for it")
                continue
            expected.setdefault(parsed[0], set()).add(name)
        for group_key, members in sorted(expected.items()):
            observed = set(groups.get(group_key, {}).get("members", ()))
            missing = sorted(members - observed)
            unexpected = sorted(observed - members)
            if missing or unexpected:
                raise ValueError(
                    f"{group_key}: executed activation-scale group does not "
                    f"match its roster (missing {missing}, unexpected "
                    f"{unexpected}); grouping the observed maxima alone would "
                    "redefine the group")
        for group_key in sorted(set(groups) - set(expected)):
            raise ValueError(
                f"{group_key}: executed activation-scale group has no roster "
                "members, so its group maximum is not authoritative")
    if not groups:
        return effective, None
    return effective, {
        "schema": ROUTED_EXECUTED_SCALE_GROUPING_SCHEMA,
        "grouping": ACTIVATION_SCALE_GROUPING_EXECUTED,
        "groups": {key: dict(entry) for key, entry in sorted(groups.items())},
        "member_count": sum(len(entry["members"]) for entry in groups.values()),
        # Explicit, not inferred: the policy the scales were computed under, and
        # the runtime whose reduction this is.  Neither is read from the live
        # environment, so a replay cannot silently pick up a different one.
        "input_global_scale_policy": canonical_policy,
        "roster_complete": bool(expected_members is not None),
        "source": ("tessera/serving/nvfp4_moe_route.py gs13 = 1 / "
                   "input_small['w13'].max(), and vLLM "
                   "amax_for_moe_activation_quant: one scale per (module, stage)"),
    }


def routed_executed_max_abs_for_census(
    max_abs: "Mapping[str, float]",
    *,
    census,
    profile=None,
    policy=None,
) -> "tuple[dict[str, float], dict | None]":
    """The census-side entry to :func:`routed_executed_max_abs`, roster included.

    A census names every unit it measured in ``unit_shapes``, and that set -- not
    the maxima that happen to be present -- is what says whether an expert is
    missing or merely zero. Grouping the observed maxima alone would let one
    absent expert silently redefine its group: the survivor's value becomes the
    group maximum, the spread shrinks, and no gate notices. So a static-scale
    set that DOES contain a per-expert routed projection but carries no
    authoritative roster refuses here instead of reducing over survivors, while
    a dense-only census needs no roster and keeps emitting no declaration.

    The requirement lives here, once, rather than at each caller: the joint
    prepare pass and any plan-time derivation of the same declaration must agree
    about when a grouping is admissible, and two copies of a fail-closed rule
    are two chances to disagree.
    """
    roster = census.get("unit_shapes") if isinstance(census, Mapping) else None
    effective, declaration = routed_executed_max_abs(
        max_abs, profile=profile,
        expected_members=(sorted(roster) if roster else None), policy=policy)
    if declaration is not None and not roster:
        raise ValueError(
            "the executed activation-scale grouping needs the census's "
            "authoritative unit roster (unit_shapes); the maxima present cannot "
            "show that an expert is missing rather than unmeasured")
    return effective, declaration


# ---------------------------------------------------------------------------
# Routed-MoE stage attestation (ROADMAP K0.2)
# ---------------------------------------------------------------------------
# A packed FusedMoE module runs TWO activation-quantized stages against two
# different tensors: ``w13`` consumes the experts-module input, ``w2`` consumes
# the routed intermediate.  The scales have always been stage-specific by
# construction (distinct physical targets, and
# :func:`unify_fused_sibling_input_global_scales` never joins across them), but
# nothing in the exported record said so — so a consumer could not distinguish
# a fully calibrated routed-MoE artifact from one that merely happened to carry
# two scalars.  The stage section makes the two stages, their calibration
# inputs, and their values explicitly attested and independently verifiable.
#
# ``NVFP4_ACTIVATION_CONTRACT_SCHEMA`` stays the DIGEST FRAMING constant and the
# dense-only record schema: bumping it would move every existing
# ``target_values_sha256`` and silently invalidate shipped artifacts.  The v2
# literal is the RECORD schema and appears only when the stage section does, so
# a reader that predates stage attestation fails closed on a routed-MoE
# artifact instead of accepting a fused-readiness claim it cannot verify.  A
# dense-only artifact keeps emitting v1, byte-for-byte.
NVFP4_ACTIVATION_CONTRACT_SCHEMA_V2 = "prismaquant.nvfp4_w4a4_activation.v2"
NVFP4_ROUTED_MOE_STAGE_SCHEMA = (
    "prismaquant.nvfp4_w4a4_activation_stages.v1"
)
NVFP4_ROUTED_MOE_STAGE_KEY = "routed_moe_stages"
NVFP4_STAGE_W13 = "w13"
NVFP4_STAGE_W2 = "w2"
NVFP4_ROUTED_MOE_STAGES = (NVFP4_STAGE_W13, NVFP4_STAGE_W2)
_PACKED_ROLE_STAGES = {
    "gate_up_proj": NVFP4_STAGE_W13,
    "down_proj": NVFP4_STAGE_W2,
}
# Profile-free fallback leaf -> packed-parameter role, kept identical to
# ``ModelProfile._fallback_packed_expert_role_parents`` so a name the profile
# can bucket is a name this module can stage without one.
_PACKED_LEAF_ROLES = {
    "gate_up_proj": "gate_up_proj",
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
    "w1": "gate_up_proj",
    "w3": "gate_up_proj",
    "down_proj": "down_proj",
    "w2": "down_proj",
}

# Calibration-source vocabulary: exactly the resolution mechanisms
# :func:`calibrated_input_global_scales` implements, named so an artifact
# reader can tell the experts-module input apart from the routed intermediate.
CALIBRATION_SOURCE_TARGET_CACHE = "target_activation_cache"
CALIBRATION_SOURCE_PARENT_MODULE_CACHE = "parent_module_activation_cache"
CALIBRATION_SOURCE_SUPPLEMENTAL_MODULE_INPUT = (
    "supplemental_module_input_sample"
)
CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY = (
    "supplemental_routed_intermediate_replay"
)
CALIBRATION_SOURCE_SUPPLEMENTAL_MAX_ABS = "supplemental_max_abs"
# The legacy native container measures both packed-expert stages during its
# GPTQ render (module input for gate/up, routed intermediate for down) and
# stores them in the production cache's packed-expert max-abs sidecar.
CALIBRATION_SOURCE_PACKED_EXPERT_RENDER = "packed_expert_render_max_abs"
NVFP4_CALIBRATION_SOURCES = frozenset({
    CALIBRATION_SOURCE_TARGET_CACHE,
    CALIBRATION_SOURCE_PARENT_MODULE_CACHE,
    CALIBRATION_SOURCE_SUPPLEMENTAL_MODULE_INPUT,
    CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY,
    CALIBRATION_SOURCE_SUPPLEMENTAL_MAX_ABS,
    CALIBRATION_SOURCE_PACKED_EXPERT_RENDER,
})
# ``w2`` must never be calibrated from the experts-module input — that is the
# exact defect this attestation exists to make impossible — and ``w13`` must
# never be calibrated from a routed-intermediate replay.
NVFP4_STAGE_CALIBRATION_SOURCES = {
    NVFP4_STAGE_W13: frozenset({
        CALIBRATION_SOURCE_TARGET_CACHE,
        CALIBRATION_SOURCE_PARENT_MODULE_CACHE,
        CALIBRATION_SOURCE_SUPPLEMENTAL_MODULE_INPUT,
        CALIBRATION_SOURCE_SUPPLEMENTAL_MAX_ABS,
        CALIBRATION_SOURCE_PACKED_EXPERT_RENDER,
    }),
    NVFP4_STAGE_W2: frozenset({
        CALIBRATION_SOURCE_TARGET_CACHE,
        CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY,
        CALIBRATION_SOURCE_SUPPLEMENTAL_MAX_ABS,
        CALIBRATION_SOURCE_PACKED_EXPERT_RENDER,
    }),
}

_E2M1_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

#: The seven ties of the positive E2M1 grid, in normalised units
#: (``t = |x| * G / stored_scale``).  Every one of them is a point where the
#: rounding RULE decides -- not the arithmetic -- so a quantiser attestation
#: that omits one attests nothing about the thing that is actually in dispute.
#: :func:`prismaquant.tessera_runtime_contract.require_activation_quantizer_attested`
#: refuses a published table that does not cover all seven.
E2M1_MIDPOINTS = tuple(
    (_E2M1_POSITIVE[i] + _E2M1_POSITIVE[i + 1]) / 2.0
    for i in range(len(_E2M1_POSITIVE) - 1)
)


# Compatibility fallback for profiles that cannot expose serving fusion
# metadata.  This catalog lives here because activation calibration, legacy
# native export, and the Gridbook execution contract must never infer different
# sibling units.  New architectures should still declare their groups in the
# model profile/structure spec.
_FUSED_DENSE_PATTERNS = (
    (
        re.compile(
            r"^(?P<pre>.+)\.self_attn\.(?P<sib>q_proj|k_proj|v_proj)$"
        ),
        ("q_proj", "k_proj", "v_proj"),
    ),
    (
        re.compile(r"^(?P<pre>.+)\.mlp\.(?P<sib>gate_proj|up_proj)$"),
        ("gate_proj", "up_proj"),
    ),
    (
        re.compile(
            r"^(?P<pre>.+)\.mlp\.shared_expert\."
            r"(?P<sib>gate_proj|up_proj)$"
        ),
        ("gate_proj", "up_proj"),
    ),
    (
        re.compile(
            r"^(?P<pre>.+)\.linear_attn\."
            r"(?P<sib>in_proj_qkv|in_proj_z)$"
        ),
        ("in_proj_qkv", "in_proj_z"),
    ),
    (
        re.compile(
            r"^(?P<pre>.+)\.linear_attn\."
            r"(?P<sib>in_proj_a|in_proj_b)$"
        ),
        ("in_proj_a", "in_proj_b"),
    ),
)


def resolve_input_global_scale_policy(
    value: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve one explicit, stampable input-global-scale policy.

    ``PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE`` remains a compatibility
    input, but it is resolved once at export startup and the resulting policy
    identity is serialized.  No runtime consumer needs to consult the env.
    """

    aliases = {
        "legacy": LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        "legacy_6_over_calibration_amax": LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        LEGACY_INPUT_GLOBAL_SCALE_POLICY: LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        "full": FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
        "full_e4m3": FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
        "full_e4m3_range": FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
        "full_e4m3_range_448x6_over_calibration_amax": (
            FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY
        ),
        FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY: (
            FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY
        ),
        "mse": MSE_GRID_INPUT_GLOBAL_SCALE_POLICY,
        "mse_grid": MSE_GRID_INPUT_GLOBAL_SCALE_POLICY,
        "mse_grid_calibrated": MSE_GRID_INPUT_GLOBAL_SCALE_POLICY,
        MSE_GRID_INPUT_GLOBAL_SCALE_POLICY: MSE_GRID_INPUT_GLOBAL_SCALE_POLICY,
    }
    if value is None:
        env = os.environ if environ is None else environ
        value = (
            FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY
            if str(env.get(
                "PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE", "0"
            )).strip() == "1"
            else LEGACY_INPUT_GLOBAL_SCALE_POLICY
        )
    canonical = aliases.get(str(value).strip().lower())
    if canonical is None:
        raise ValueError(
            f"unknown NVFP4 input-global-scale policy {value!r}; expected "
            f"one of {sorted(NVFP4_INPUT_GLOBAL_SCALE_POLICIES)}"
        )
    return canonical


def input_global_scale_from_max_abs(
    max_abs: float,
    *,
    policy: str,
    nonpositive_fallback: float | None = None,
) -> float:
    """Return a finite positive F32-representable calibrated scalar.

    ``nonpositive_fallback`` exists only for legacy compressed-tensors export,
    whose historical all-zero behavior serialized ``1.0``.  It is opt-in so a
    versioned activation contract continues to reject missing/degenerate
    calibration.  Non-finite positive values and NaNs always fail closed.
    """

    canonical = resolve_input_global_scale_policy(policy)
    if canonical == MSE_GRID_INPUT_GLOBAL_SCALE_POLICY:
        raise ValueError(
            "mse_grid_calibrated.v1 requires activation samples; call "
            "select_mse_grid_input_global_scale"
        )
    value = float(max_abs)
    if value <= 0.0 and nonpositive_fallback is not None:
        return float(input_global_scale_tensor(nonpositive_fallback).item())
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"NVFP4 activation calibration max_abs must be finite and > 0, "
            f"got {max_abs!r}"
        )
    numerator = FP4_E2M1_MAX
    if canonical == FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY:
        numerator *= FP8_E4M3_MAX
    # The artifact tensor is F32.  Digest and export the rounded value, not a
    # Python-f64 value that the loader can never observe.
    result = struct.unpack("<f", struct.pack("<f", numerator / value))[0]
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(
            f"NVFP4 input_global_scale is not finite/positive after F32 "
            f"rounding: max_abs={value}, policy={canonical}, value={result}"
        )
    return result


def input_global_scale_tensor(value: float) -> torch.Tensor:
    """Canonical compressed-tensors scalar representation: F32 shape ``[1]``."""

    rounded = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    if not math.isfinite(rounded) or rounded <= 0.0:
        raise ValueError(
            f"input_global_scale must be finite and > 0, got {value!r}"
        )
    return torch.tensor([rounded], dtype=torch.float32)


def resolve_input_global_scale_value(
    override: float | None = None,
    *,
    target: str | None = None,
    calibrated_scales: Mapping[str, float] | None = None,
    allow_uncalibrated_fallback: bool = False,
) -> float:
    """Resolve explicit override -> calibrated mapping -> legacy fallback.

    The fallback is deliberately disabled by default.  Calling with
    ``allow_uncalibrated_fallback=True`` preserves old native-export bytes but
    also means the result cannot attest the versioned fused-W4A4 contract.
    """

    value = override
    if value is None and target is not None and calibrated_scales:
        value = calibrated_scales.get(str(target))
    if value is None:
        if not allow_uncalibrated_fallback:
            raise ValueError(
                "NVFP4 activation contract has no calibrated "
                f"{NVFP4_INPUT_GLOBAL_SCALE_SUFFIX}"
                + (f" for {target!r}" if target is not None else "")
            )
        value = UNCALIBRATED_INPUT_GLOBAL_SCALE
    # Preserve legacy override/map semantics.  Artifact construction performs
    # the existing F32 cast; strict contract paths use input_global_scale_tensor
    # when they build and digest their physical mapping.
    return float(value)


def fused_dense_group(name: str) -> tuple[str, tuple[str, ...]] | None:
    """Return the legacy fallback group prefix and sibling leaf names."""

    for pattern, members in _FUSED_DENSE_PATTERNS:
        match = pattern.match(str(name))
        if match:
            return match.group("pre"), members
    return None


def fused_sibling_group_key(
    name: str,
    *,
    profile=None,
    tolerate_profile_errors: bool = False,
) -> str | None:
    """Resolve one canonical fused-sibling key.

    Versioned contract callers use the strict default: a profile that cannot
    attest its execution unit is an export error.  The legacy native exporter
    passes ``tolerate_profile_errors=True`` to preserve its historical
    profile-metadata fallback behavior without owning a second algorithm.
    """

    target = str(name)
    group_fn = getattr(profile, "fused_sibling_group", None)
    if callable(group_fn):
        try:
            group = group_fn(target)
        except Exception:
            if not tolerate_profile_errors:
                raise
            group = None
        if group:
            return str(group)

    mapping_fn = getattr(profile, "fused_sibling_leaf_mapping", None)
    if callable(mapping_fn) and "." in target:
        try:
            mapping = mapping_fn()
        except Exception:
            if not tolerate_profile_errors:
                raise
            mapping = None
        if mapping:
            prefix, leaf = target.rsplit(".", 1)
            for fused, members in mapping.items():
                if leaf in {str(member) for member in members}:
                    return f"{prefix}.{fused}"

    fallback = fused_dense_group(target)
    if fallback is None:
        return None
    prefix, members = fallback
    return f"{prefix}::__fused__:{','.join(members)}"


def group_fused_sibling_targets(
    targets: Iterable[str],
    *,
    profile=None,
    tolerate_profile_errors: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Bucket targets by their runtime fused execution unit.

    Unfused targets remain singleton groups so calibration can use this one
    primitive without maintaining a parallel grouping loop.
    """

    groups: dict[str, list[str]] = {}
    for raw_target in targets:
        target = str(raw_target)
        group = fused_sibling_group_key(
            target,
            profile=profile,
            tolerate_profile_errors=tolerate_profile_errors,
        )
        groups.setdefault(str(group or target), []).append(target)
    return {key: tuple(members) for key, members in groups.items()}


def unify_fused_sibling_max_abs(
    max_abs_by_target: Mapping[str, float],
    *,
    profile=None,
    tolerate_profile_errors: bool = False,
) -> dict[str, float]:
    """Use one conservative calibration maximum for every fused sibling.

    vLLM concatenates q/k/v and gate/up and applies one activation scale.  The
    largest max-abs (equivalently the smallest reciprocal scale) is shared.
    """

    result = {str(k): float(v) for k, v in max_abs_by_target.items()}
    groups = group_fused_sibling_targets(
        max_abs_by_target,
        profile=profile,
        tolerate_profile_errors=tolerate_profile_errors,
    )
    for members in groups.values():
        shared = max(float(result[name]) for name in members)
        for name in members:
            result[name] = shared
    return result


def unify_fused_sibling_input_global_scales(
    scales: Mapping[str, float],
    *,
    profile=None,
    tolerate_profile_errors: bool = False,
    diagnostic_prefix: str | None = None,
) -> dict[str, float]:
    """Conservatively join reciprocal scales across fused siblings.

    ``input_global_scale`` is proportional to ``1 / calibration_amax`` under
    every static policy, so the safe fused join is the minimum scale (the
    largest observed activation range).  Only siblings present in ``scales``
    participate; singleton/unfused targets pass through byte-for-byte.
    """

    groups = group_fused_sibling_targets(
        scales,
        profile=profile,
        tolerate_profile_errors=tolerate_profile_errors,
    )
    result = {str(name): float(value) for name, value in scales.items()}
    unified = 0
    max_drift = 0.0
    for members in groups.values():
        if len(members) < 2:
            continue
        values = [float(scales[member]) for member in members]
        shared = min(values)
        max_drift = max(
            max_drift,
            max(abs(shared - value) for value in values),
        )
        for member in members:
            result[member] = shared
        unified += 1
    if diagnostic_prefix and unified:
        print(
            f"{diagnostic_prefix} unified input_global_scale across "
            f"{unified} fused-sibling groups "
            f"(max pre-unify drift: {max_drift:.3e})",
            flush=True,
        )
    return result


def _activation_cache_candidates(targets: Iterable[str]) -> set[str]:
    candidates = {str(target) for target in targets}
    for target in tuple(candidates):
        for suffix in (".gate_up_proj", ".gate_proj", ".up_proj", ".down_proj"):
            if target.endswith(suffix):
                candidates.add(target[: -len(suffix)])
    return candidates


def load_activation_cache_samples(
    cache_dir: str | Path,
    targets: Iterable[str],
) -> dict[str, torch.Tensor]:
    """Load only relevant input samples from the existing probe cache."""

    from prismaquant.measure_quant_cost import ActivationIndex

    root = Path(cache_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"NVFP4 activation cache directory does not exist: {root}"
        )
    candidates = _activation_cache_candidates(targets)
    index = ActivationIndex(root, sorted(candidates))
    values: dict[str, torch.Tensor] = {}
    for name in sorted(candidates):
        if name not in index:
            continue
        tensor = index.load(name)
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            raise ValueError(
                f"NVFP4 activation cache entry {name!r} has no input tensor"
            )
        tensor = tensor.detach().to("cpu").float().contiguous()
        value = float(tensor.abs().max().item())
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"NVFP4 activation cache entry {name!r} has invalid max_abs "
                f"{value!r}"
            )
        values[name] = tensor
    return values


def load_activation_cache_max_abs(
    cache_dir: str | Path,
    targets: Iterable[str],
) -> dict[str, float]:
    """Compatibility view of :func:`load_activation_cache_samples`."""

    return {
        name: float(tensor.abs().max().item())
        for name, tensor in load_activation_cache_samples(
            cache_dir,
            targets,
        ).items()
    }


def select_mse_grid_input_global_scale(
    activation_samples: Iterable[torch.Tensor],
    *,
    device: str | torch.device | None = None,
) -> float:
    """Pick a deterministic static G on the serve-QDQ MSE objective.

    The grid spans the legacy ``6/amax`` and full-E4M3 ``448*6/amax``
    endpoints at quarter-octave resolution.  Both endpoints are always
    included exactly.  A fused sibling group is optimized jointly by passing
    all of its cached samples here.  Equal scores select the smaller G (less
    aggressive clipping).
    """

    samples = [
        tensor.detach().reshape(-1, tensor.shape[-1]).float()
        for tensor in activation_samples
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0
    ]
    if not samples:
        raise ValueError("MSE-grid NVFP4 calibration has no activation samples")
    if any(tensor.shape[-1] % FP4_GROUP_SIZE for tensor in samples):
        bad = [tuple(tensor.shape) for tensor in samples
               if tensor.shape[-1] % FP4_GROUP_SIZE]
        raise ValueError(
            f"MSE-grid NVFP4 calibration needs width divisible by 16: {bad}"
        )
    max_abs = max(float(tensor.abs().max().item()) for tensor in samples)
    if not math.isfinite(max_abs) or max_abs <= 0.0:
        raise ValueError(
            f"MSE-grid NVFP4 calibration has invalid max_abs {max_abs!r}"
        )
    legacy = FP4_E2M1_MAX / max_abs
    full = FP8_E4M3_MAX * legacy
    factors = [2.0 ** (step / 4.0) for step in range(36)]
    factors.append(FP8_E4M3_MAX)
    candidates = sorted({
        struct.unpack("<f", struct.pack("<f", legacy * factor))[0]
        for factor in factors
        if legacy * factor <= full
    } | {struct.unpack("<f", struct.pack("<f", full))[0]})

    target_device = torch.device(device or "cpu")
    device_samples = [tensor.to(target_device) for tensor in samples]
    best_scale = candidates[0]
    best_error = math.inf
    for candidate in candidates:
        squared_error = 0.0
        count = 0
        for sample in device_samples:
            qdq = nvfp4_activation_qdq_served(sample, candidate).float()
            squared_error += float(
                (qdq - sample).square().sum(dtype=torch.float64).item()
            )
            count += int(sample.numel())
        error = squared_error / max(count, 1)
        if error < best_error:
            best_error = error
            best_scale = candidate
    return float(best_scale)


def calibrated_input_global_scales(
    targets: Iterable[str],
    *,
    activation_cache_dir: str | Path,
    policy: str,
    profile=None,
    supplemental_max_abs: Mapping[str, float] | None = None,
    supplemental_activations: Mapping[str, Any] | None = None,
    calibration_device: str | torch.device | None = None,
) -> dict[str, float]:
    """Resolve complete target coverage and return fused-coherent scalars."""

    scales, _sources = calibrated_input_global_scales_with_sources(
        targets,
        activation_cache_dir=activation_cache_dir,
        policy=policy,
        profile=profile,
        supplemental_max_abs=supplemental_max_abs,
        supplemental_activations=supplemental_activations,
        calibration_device=calibration_device,
    )
    return scales


def calibrated_input_global_scales_with_sources(
    targets: Iterable[str],
    *,
    activation_cache_dir: str | Path,
    policy: str,
    profile=None,
    supplemental_max_abs: Mapping[str, float] | None = None,
    supplemental_activations: Mapping[str, Any] | None = None,
    calibration_device: str | torch.device | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Resolve coverage and return ``(scales, calibration source per target)``.

    Packed gate/up targets consume the experts-module input and therefore may
    use that parent cache entry.  Packed down targets require their routed
    intermediate max-abs in ``supplemental_max_abs``; exporters synthesize it
    with the same checkpoint replay used by the imatrix harvester.

    The second return value names which of those mechanisms actually produced
    each scalar.  That distinction is not diagnostic decoration: it is what the
    routed-MoE stage attestation publishes so a consumer can verify that ``w2``
    was calibrated on the routed intermediate rather than on the module input.
    """

    requested = tuple(sorted({str(target) for target in targets}))
    supplemental_samples: dict[str, torch.Tensor] = {}
    supplemental_sample_sources: dict[str, str] = {}
    for name, raw_sample in (supplemental_activations or {}).items():
        sample = raw_sample
        source = CALIBRATION_SOURCE_SUPPLEMENTAL_MODULE_INPUT
        if not isinstance(sample, torch.Tensor):
            validate = getattr(sample, "validate", None)
            if not callable(validate):
                raise TypeError(
                    f"supplemental activation {name!r} is neither a tensor "
                    "nor a validated routed sample"
                )
            validate()
            sample = getattr(sample, "values", None)
            source = CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY
        if not isinstance(sample, torch.Tensor) or sample.numel() == 0:
            raise ValueError(
                f"supplemental activation {name!r} has no value-bearing rows"
            )
        supplemental_samples[str(name)] = (
            sample.detach().to("cpu").float().contiguous()
        )
        supplemental_sample_sources[str(name)] = source
    supplemental = {
        str(name): float(value)
        for name, value in (supplemental_max_abs or {}).items()
    }
    canonical_policy = resolve_input_global_scale_policy(policy)
    groups = group_fused_sibling_targets(requested, profile=profile)
    result: dict[str, float] = {}
    sources: dict[str, str] = {}
    for members in groups.values():
        # Cache rows can be very large (tens of GB on 27B+ models).  Load and
        # fit one execution/fusion unit at a time; no policy needs samples from
        # unrelated modules.  This also makes the q/k/v and gate/up union an
        # explicit boundary rather than an accidental whole-model reduction.
        cached = load_activation_cache_samples(
            activation_cache_dir,
            members,
        )
        resolved_samples: dict[str, torch.Tensor] = {}
        resolved_max_abs: dict[str, float] = {}
        for target in members:
            if target in supplemental_samples:
                sample = supplemental_samples[target]
                source = supplemental_sample_sources[target]
            elif target in cached:
                sample = cached[target]
                source = CALIBRATION_SOURCE_TARGET_CACHE
            else:
                sample = None
                source = None
            value = supplemental.get(target)
            if value is not None and sample is None:
                source = CALIBRATION_SOURCE_SUPPLEMENTAL_MAX_ABS
            if sample is None and value is None and target.endswith((
                ".gate_up_proj", ".gate_proj", ".up_proj"
            )):
                parent = target.rsplit(".", 1)[0]
                sample = cached.get(parent)
                if sample is not None:
                    source = CALIBRATION_SOURCE_PARENT_MODULE_CACHE
            if sample is not None:
                value = float(sample.abs().max().item())
                resolved_samples[target] = sample
            if source is not None:
                sources[target] = source
            if value is None:
                raise ValueError(
                    f"NVFP4 activation contract has no calibrated input for "
                    f"{target!r}; production export refuses an incomplete "
                    "scale mapping"
                )
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"NVFP4 activation contract has invalid max_abs for "
                    f"{target!r}: {value!r}"
                )
            resolved_max_abs[target] = float(value)
        if canonical_policy == MSE_GRID_INPUT_GLOBAL_SCALE_POLICY:
            missing_samples = [
                target for target in members if target not in resolved_samples
            ]
            if missing_samples:
                raise ValueError(
                    "MSE-grid NVFP4 activation calibration needs value-bearing "
                    f"samples for every fused target, missing {missing_samples}"
                )
            shared_scale = select_mse_grid_input_global_scale(
                (resolved_samples[target] for target in members),
                device=calibration_device,
            )
        else:
            shared_max_abs = max(resolved_max_abs[target] for target in members)
            shared_scale = input_global_scale_from_max_abs(
                shared_max_abs,
                policy=canonical_policy,
            )
        for target in members:
            result[target] = shared_scale
    return result, sources


def routed_moe_stage(name: str, *, profile=None) -> tuple[str, str] | None:
    """Return ``(module_prefix, stage)`` for one packed FusedMoE stage target.

    ``None`` means the name is not a packed routed-expert stage: a dense
    ``mlp.down_proj`` and the per-expert split form
    ``<parent>.experts.7.gate_proj`` are Linears, not stages, and must not
    contribute a stage claim.  The profile owns leaf naming (LFM2.5 spells the
    stages ``w1``/``w3``/``w2``); the leaf table is only the profile-free
    fallback the rest of this module already uses.
    """

    target = str(name)
    parts = target.split(".")
    # Packed form only: ``<parent>.experts.<packed-parameter>``.
    if len(parts) < 2 or parts[-2] != "experts":
        return None
    role = None
    role_fn = getattr(profile, "packed_expert_role_group", None)
    if callable(role_fn):
        role = role_fn(target)
    if role is None:
        role = _PACKED_LEAF_ROLES.get(parts[-1])
    stage = _PACKED_ROLE_STAGES.get(str(role)) if role is not None else None
    if stage is None:
        return None
    return ".".join(parts[:-1]), stage


def stage_values_sha256(
    *,
    stage: str,
    target: str,
    policy: str,
    calibration_source: str,
    value: float,
) -> str:
    """Digest one routed-MoE stage's complete attested identity and value.

    Framing mirrors :func:`target_values_sha256` (length-prefixed UTF-8 fields
    then the serialized F32) but is rooted at the stage schema, so a stage
    digest can never collide with a whole-model digest.  Every attested field
    participates: a stage whose policy or calibration source changed is a
    different attestation even at an identical scalar.
    """

    canonical_policy = resolve_input_global_scale_policy(policy)
    if stage not in NVFP4_ROUTED_MOE_STAGES:
        raise ValueError(
            f"unknown routed-MoE stage {stage!r}; expected one of "
            f"{list(NVFP4_ROUTED_MOE_STAGES)}"
        )
    if calibration_source not in NVFP4_CALIBRATION_SOURCES:
        raise ValueError(
            f"unknown NVFP4 calibration source {calibration_source!r}; "
            f"expected one of {sorted(NVFP4_CALIBRATION_SOURCES)}"
        )
    digest = hashlib.sha256()
    for field in (
        NVFP4_ROUTED_MOE_STAGE_SCHEMA,
        canonical_policy,
        str(stage),
        str(target),
        str(calibration_source),
    ):
        encoded = field.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    digest.update(
        struct.pack("<f", float(input_global_scale_tensor(value).item()))
    )
    return digest.hexdigest()


def routed_moe_stages_sha256(
    modules: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> str:
    """Digest the whole stage section from its per-stage digests."""

    digest = hashlib.sha256()
    encoded = NVFP4_ROUTED_MOE_STAGE_SCHEMA.encode("utf-8")
    digest.update(struct.pack("<I", len(encoded)))
    digest.update(encoded)
    for module in sorted(modules):
        encoded = str(module).encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
        entries = modules[module]
        for stage in NVFP4_ROUTED_MOE_STAGES:
            if stage not in entries:
                raise ValueError(
                    f"packed FusedMoE module {module!r} has no {stage} stage; "
                    "a routed-MoE contract must attest both stages"
                )
            encoded = str(stage).encode("utf-8")
            digest.update(struct.pack("<I", len(encoded)))
            digest.update(encoded)
            digest.update(
                bytes.fromhex(str(entries[stage]["stage_values_sha256"]))
            )
    return digest.hexdigest()


def routed_moe_attested_module_names(
    record: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """The routed-MoE modules an execution-contract record actually attests.

    ONE definition of "what is in K0.2's scope", read off the record rather
    than re-derived from the assignment, so the producer's reconciliation and
    a consumer's audit cannot disagree about it.  ``()`` means the record
    attests no routed-MoE module at all — either it is dense-only or it is
    absent.

    THE SCOPE PROBLEM THIS EXISTS FOR.  :func:`build_routed_moe_stage_attestation`
    can only attest a module that CONTRIBUTED a calibrated fp4 stage scalar.  A
    routed-expert layer that ships on a source-passthrough rung
    (``MXFP4_SOURCE``: the checkpoint's own bytes, served by the model's native
    kernel) has no CB activation contract to attest and is therefore simply
    ABSENT here — which, read alone, is indistinguishable from a partial or
    failed export that dropped a module it should have calibrated.  The
    completeness guard below cannot close that: it only fires for a module that
    reached the record with ONE of its two stages.

    The gap is closed OUTSIDE this record, deliberately, by the top-level
    ``source_passthrough`` declaration
    (``cb_export_config.build_source_passthrough_declaration``): it positively
    names every unit handed to the model's own loader, and
    ``export_nvfp4_cb_streaming.assert_routes_reconcile`` proves that scope is
    disjoint from this one before an artifact ships.  The K0.2 record's own
    field set stays byte-identical, because it is a pinned cross-repository
    contract whose consumer rejects unknown keys outright
    (archive/gridbook_lane_2026-09-02/tests/test_gridbook_attestation_interop.py,
    archived 2026-09-02 with the lane) — widening it "additively" is
    exactly the failure that test exists to catch.
    """

    if not isinstance(record, Mapping):
        return ()
    section = record.get(NVFP4_ROUTED_MOE_STAGE_KEY)
    if not isinstance(section, Mapping):
        return ()
    names = section.get("module_names")
    if not isinstance(names, (list, tuple)):
        raise ValueError(
            "routed-MoE stage section has no module_names list; refusing to "
            "guess the attested scope"
        )
    return tuple(str(name) for name in names)


def build_routed_moe_stage_attestation(
    scales: Mapping[str, float],
    *,
    policy: str,
    calibration_sources: Mapping[str, str],
    profile=None,
    target_name: Callable[[str], str] | None = None,
) -> dict[str, Any] | None:
    """Return the per-module ``w13``/``w2`` section, or ``None`` when dense.

    This is the single builder every exporter uses, so the same logical
    scales, policy, and calibration sources produce a byte-identical section
    whichever emit path ran.  It fails closed on a packed FusedMoE module that
    reaches export with only one attested stage: that is precisely the state
    the existing partial LFM artifact is in, and no artifact may claim
    fused-MoE readiness from it.

    What it CANNOT see is a routed-expert layer that contributed no stage at
    all — see :func:`routed_moe_attested_module_names` for why that is not this
    record's job to declare, and where it is declared instead.
    """

    mapper = target_name or (lambda name: name)
    canonical_policy = resolve_input_global_scale_policy(policy)
    modules: dict[str, dict[str, dict[str, Any]]] = {}
    for logical_name in sorted(scales):
        logical = str(logical_name)
        if routed_moe_stage(logical, profile=profile) is None:
            continue
        physical = str(mapper(logical))
        parsed = routed_moe_stage(physical, profile=profile)
        if parsed is None:
            raise ValueError(
                f"routed-MoE stage target {logical!r} maps to physical prefix "
                f"{physical!r}, which does not spell a packed FusedMoE stage; "
                "the stage attestation must name the exact serialized prefix"
            )
        module, stage = parsed
        source = calibration_sources.get(logical)
        if source is None:
            raise ValueError(
                f"routed-MoE stage target {logical!r} has no attested "
                "calibration source; production export refuses an unattested "
                "fused-MoE stage"
            )
        allowed = NVFP4_STAGE_CALIBRATION_SOURCES[stage]
        if source not in allowed:
            raise ValueError(
                f"routed-MoE stage {stage} target {logical!r} was calibrated "
                f"from {source!r}, which is not a legal input for that stage "
                f"(expected one of {sorted(allowed)})"
            )
        entries = modules.setdefault(module, {})
        if stage in entries:
            raise ValueError(
                f"packed FusedMoE module {module!r} has two {stage} stage "
                f"targets: {entries[stage]['target']!r} and {physical!r}"
            )
        entries[stage] = {
            "stage": stage,
            "target": physical,
            "input_global_scale_policy": canonical_policy,
            "calibration_source": source,
            "stage_values_sha256": stage_values_sha256(
                stage=stage,
                target=physical,
                policy=canonical_policy,
                calibration_source=source,
                value=float(scales[logical_name]),
            ),
        }
    if not modules:
        return None
    incomplete = {
        module: [
            stage for stage in NVFP4_ROUTED_MOE_STAGES if stage not in entries
        ]
        for module, entries in sorted(modules.items())
        if len(entries) != len(NVFP4_ROUTED_MOE_STAGES)
    }
    if incomplete:
        raise ValueError(
            "routed-MoE activation contract requires both w13 and w2 stages "
            f"for every packed FusedMoE module; missing {incomplete}"
        )
    return {
        "schema": NVFP4_ROUTED_MOE_STAGE_SCHEMA,
        "stages": list(NVFP4_ROUTED_MOE_STAGES),
        "module_count": len(modules),
        "module_names": sorted(modules),
        "modules": {
            module: {
                stage: modules[module][stage]
                for stage in NVFP4_ROUTED_MOE_STAGES
            }
            for module in sorted(modules)
        },
        "stages_sha256": routed_moe_stages_sha256(modules),
    }


def target_values_sha256(
    scales: Mapping[str, float],
    *,
    policy: str,
) -> str:
    """Digest exact physical target names and their serialized F32 values.

    The framing constant is deliberately pinned at
    ``NVFP4_ACTIVATION_CONTRACT_SCHEMA`` (the v1 literal) even when the record
    declares the v2 stage-attested schema, so the whole-model digest a shipped
    artifact carries never moves under a record-schema bump.
    """

    canonical_policy = resolve_input_global_scale_policy(policy)
    digest = hashlib.sha256()
    for field in (NVFP4_ACTIVATION_CONTRACT_SCHEMA, canonical_policy):
        encoded = field.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    for target in sorted(scales):
        encoded = str(target).encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<f", float(scales[target])))
    return digest.hexdigest()


def build_execution_contract(
    scales: Mapping[str, float],
    *,
    policy: str,
    target_name: Callable[[str], str] | None = None,
    calibration_sources: Mapping[str, str] | None = None,
    profile=None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Return the top-level record plus scales keyed by physical target name.

    ``calibration_sources`` is what :func:`calibrated_input_global_scales_with_sources`
    returns.  Supplying it is mandatory whenever any target is a packed
    FusedMoE stage: without it the record cannot attest which tensor calibrated
    each stage, and a routed-MoE artifact must never claim fused readiness it
    cannot back.  Dense-only exports may omit it and keep emitting the v1
    record byte-for-byte.
    """

    mapper = target_name or (lambda name: name)
    physical: dict[str, float] = {}
    for logical_name, raw_value in scales.items():
        name = str(mapper(str(logical_name)))
        if not name:
            raise ValueError(
                f"NVFP4 logical target {logical_name!r} maps to an empty "
                "physical prefix"
            )
        value = float(input_global_scale_tensor(raw_value).item())
        if name in physical:
            raise ValueError(
                f"multiple NVFP4 logical targets map to physical prefix "
                f"{name!r}; the activation namespace must be one-to-one"
            )
        physical[name] = value
    if not physical:
        raise ValueError("NVFP4 execution contract requires at least one target")
    canonical_policy = resolve_input_global_scale_policy(policy)
    record = {
        "schema": NVFP4_ACTIVATION_CONTRACT_SCHEMA,
        "contract": NVFP4_ACTIVATION_EXECUTION,
        "group_size": FP4_GROUP_SIZE,
        "tensor_suffix": NVFP4_INPUT_GLOBAL_SCALE_SUFFIX,
        "value_dtype": "float32",
        "input_global_scale_policy": canonical_policy,
        "target_count": len(physical),
        # Config-group targets are not a sufficient namespace oracle: stock
        # compressed-tensors groups use regex targets, while nested models may
        # use a logical module name that differs from the checkpoint prefix.
        # Publish the exact physical prefixes hashed below and used for the
        # serialized ``<target>.input_global_scale`` tensors.
        "target_names": sorted(physical),
        "target_values_sha256": target_values_sha256(
            physical,
            policy=canonical_policy,
        ),
    }
    if calibration_sources is None:
        unattested = sorted(
            str(name) for name in scales
            if routed_moe_stage(str(name), profile=profile) is not None
        )
        if unattested:
            raise ValueError(
                "packed FusedMoE stage targets require calibration-source "
                "attestation in the NVFP4 execution contract; "
                f"{unattested} were offered without one"
            )
    else:
        stage_section = build_routed_moe_stage_attestation(
            scales,
            policy=canonical_policy,
            calibration_sources=calibration_sources,
            profile=profile,
            target_name=mapper,
        )
        if stage_section is not None:
            # Deliberate record-schema bump: the whole-model fields above stay
            # bit-identical, but a reader that cannot verify stage attestation
            # must fail closed on a routed-MoE artifact rather than accept it.
            record["schema"] = NVFP4_ACTIVATION_CONTRACT_SCHEMA_V2
            record[NVFP4_ROUTED_MOE_STAGE_KEY] = stage_section
    return record, physical


def nvfp4_activation_qdq_served(
    x: torch.Tensor,
    input_global_scale: float,
) -> torch.Tensor:
    """Serve-faithful static-G NVFP4 activation QDQ oracle.

    vLLM stores each 16-value block scale as UE4M3 and converts activation
    values to E2M1 with round-to-nearest, ties-to-even over the encoded positive
    index.  There is no minimum-scale clamp: with G=1, exactly
    ``6 * 2**-10`` ties the E4M3 scale to byte zero; a value just above it
    becomes byte one and the block is nonzero.

    Installed kernels use ``rcp.approx.ftz.f32`` in ``outputScale``.  Therefore
    arbitrary random Torch results are a numerical oracle, not a packed-byte
    equivalence claim; midpoint and underflow boundary cases are authoritative.
    """

    if x.shape[-1] % FP4_GROUP_SIZE != 0:
        raise ValueError(
            "nvfp4_activation_qdq_served needs last dim divisible by 16, "
            f"got {tuple(x.shape)}"
        )
    g = float(input_global_scale)
    if not math.isfinite(g) or g <= 0.0:
        raise ValueError(
            f"input_global_scale must be finite and > 0, got {g!r}"
        )
    original_shape = x.shape
    original_dtype = x.dtype
    grouped = x.reshape(-1, x.shape[-1] // FP4_GROUP_SIZE,
                        FP4_GROUP_SIZE).float()
    stored_scale = nvfp4_group_stored_scale(grouped, g).float()
    used_scale = stored_scale / g

    nonzero_scale = stored_scale != 0
    safe_scale = torch.where(
        nonzero_scale,
        used_scale,
        torch.ones_like(used_scale),
    )
    normalized = nvfp4_e2m1_normalize(grouped, safe_scale)
    index = nvfp4_e2m1_magnitude_index(normalized)
    rounded = _e2m1_positive_table(
        normalized.device)[index].copysign(normalized)
    output = rounded * used_scale
    output = torch.where(nonzero_scale, output, torch.zeros_like(output))
    return output.reshape(original_shape).to(original_dtype)


def nvfp4_group_stored_scale(grouped: torch.Tensor, g) -> torch.Tensor:
    """The UE4M3 scale byte the runtime stores for each group, as ``e4m3``.

    The first half of the quantiser, split out so the attestation preflight
    drives the arithmetic the priced path runs rather than a second copy of
    it.  Returned in ``float8_e4m3fn`` rather than ``float32`` so a caller can
    read the stored BYTE -- which is what the runtime publishes -- with
    ``.view(torch.uint8)``; the oracle takes ``.float()`` of it as before.

    ``g`` may be a float or a broadcastable tensor: the preflight drives one
    group at a time at a published global scale, the oracle a whole tensor at
    the unit's.
    """
    return nvfp4_stored_scale_from_amax(grouped.abs().amax(dim=-1, keepdim=True), g)


def nvfp4_stored_scale_from_amax(amax: torch.Tensor, g) -> torch.Tensor:
    """THE scale rule: ``e4m3(min(amax / 6 * G, 448))`` for a group maximum.

    :func:`nvfp4_group_stored_scale` is this rule applied to a grouped
    tensor's ``abs().amax(-1, keepdim=True)``; the served operator leg applies
    it to the same maximum taken by a fused kernel
    (:func:`prismaquant.kernels.nvfp4_served_dequant.group_abs_max`).  One
    function, so the priced scale rule and the attested one cannot become two
    objects (RobTand/prismaquant#1211).
    """
    return (amax / FP4_E2M1_MAX * g).clamp(max=FP8_E4M3_MAX).to(
        torch.float8_e4m3fn)


def _e2m1_positive_table(device) -> torch.Tensor:
    """The eight positive E2M1 magnitudes, device-resident.

    The registry already owns device-resident format constants; its sorted
    E2M1 table ends with these eight positive encodings, including ``+0``.
    Reuse that view instead of copying a Python tuple to CUDA on every call.
    """
    from .format_registry import _CODEBOOKS, _codebook_on_device

    return _codebook_on_device(
        _CODEBOOKS["fp4_e2m1"], device=device, dtype=torch.float32,
    )[-len(_E2M1_POSITIVE):]


def nvfp4_e2m1_normalize(
    values: torch.Tensor,
    used_scale: torch.Tensor,
) -> torch.Tensor:
    """``values / used_scale``, clamped to the grid, in the served order.

    Split out of :func:`nvfp4_activation_qdq_served` so the attestation
    preflight drives the ARITHMETIC this oracle actually runs rather than a
    float64 idealisation of it.  The division order and the single-precision
    intermediate are part of what is under test: a preflight that recomputed
    the normalisation its own way would attest its own rounding, not PQ's.
    """
    return (values / used_scale).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)


def nvfp4_e2m1_magnitude_index(normalized: torch.Tensor) -> torch.Tensor:
    """The positive-E2M1 index 0..7 a normalised value rounds to.

    THE tie-break, in one place.  Positive E2M1 encodings are indices 0..7; on
    an exact midpoint round-to-nearest-even selects the candidate whose encoded
    index has an even least-significant bit, rather than always selecting the
    lower magnitude -- which is what makes this rule, and not merely this
    arithmetic, the thing a runtime attestation has to confirm.  Both
    :func:`nvfp4_activation_qdq_served` and
    :func:`prismaquant.tessera_runtime_contract.require_activation_quantizer_attested`
    call it, so an attested rule and a priced rule cannot be two objects
    (principle 8).
    """
    positive = _e2m1_positive_table(normalized.device)
    magnitude = normalized.abs().contiguous()
    upper_index = torch.bucketize(magnitude, positive).clamp_max(
        positive.numel() - 1
    )
    lower_index = (upper_index - 1).clamp_min(0)
    lower = positive[lower_index]
    upper = positive[upper_index]
    lower_distance = (magnitude - lower).abs()
    upper_distance = (upper - magnitude).abs()
    tie = upper_distance == lower_distance
    choose_upper = (upper_distance < lower_distance) | (
        tie & ((upper_index & 1) == 0)
    )
    return torch.where(choose_upper, upper_index, lower_index)


#: How a published code decomposes: bit 3 is the sign (1 = negative) and bits
#: 0..2 index :data:`_E2M1_POSITIVE`.  That is the order the runtime's own
#: ``E2M1_VALUES`` table is indexed in -- Tessera's
#: ``represented_native_input`` reads ``levels[codes]`` with exactly these
#: nibbles -- and what pins it in a published attestation is that table's
#: ``grid`` field, which
#: :func:`prismaquant.tessera_runtime_contract.require_activation_quantizer_attested`
#: compares rather than assumes.  A table under another grid is refused, not
#: read with this decomposition.
E2M1_CODE_LAYOUT = "sign_bit_3_magnitude_index_bits_0_2"


def nvfp4_e2m1_code(normalized: torch.Tensor) -> torch.Tensor:
    """The 4-bit code under :data:`E2M1_CODE_LAYOUT`.

    ``torch.signbit`` rather than a comparison against zero, so a negative
    value that rounds to the zero magnitude emits the negative zero code the
    grid actually has.  The two zero codes dequantise to the same number, so
    this distinction is inert for the GEMM and visible only here.
    """
    index = nvfp4_e2m1_magnitude_index(normalized)
    return index | (torch.signbit(normalized).to(index.dtype) << 3)


class ActivationScaleContractError(RuntimeError):
    """A consumer had to price the served static-scale contract and had no
    calibrated maximum to derive its ``input_global_scale`` from.

    Raised by name (the qname) so the refusal says which unit, and raised
    BEFORE anything is measured: an activation-KL hook or a cache score that
    silently fell back to the dynamic FP32-scale RTN would price an activation
    tensor the runtime never executes (RobTand/prismaquant#194, #205).  The
    campaign's own refusal (``tessera_campaign.ActivationScaleContractError``)
    is this class under its historical name.
    """


class ActivationScalePolicyMismatchError(ActivationScaleContractError):
    """A stored activation-aware cost is being reused at a different G.

    The resolved input-global-scale policy is part of what a score MEANS: the
    same calibration maximum prices ``G = 6/amax`` under
    ``legacy_6_over_calibration_amax.v1`` and ``G = 448*6/amax`` under
    ``full_e4m3_range_448x6_over_calibration_amax.v1``, and the served oracle
    at those two G values quantizes the same activation differently (a block
    far below the maximum underflows to zero at the smaller G).  So a cached
    score priced under one policy is not a cost under the other, even though
    the weights, the calibration hash and the maximum are all unchanged
    (RobTand/prismaquant#227).

    Raised by name (the qname), naming both G values, wherever a persisted
    cost meets a live policy: the production cache's resume, and the
    assignment-KL hooks that measure against those costs.  A subclass of
    :class:`ActivationScaleContractError` because it is the same contract
    refusing -- a consumer already catching that one keeps catching this.
    """


def require_matching_input_global_scale(
    priced: float | None,
    applied: float,
    *,
    qname: str | None,
    consumer: str,
    priced_policy: str | None = None,
    applied_policy: str | None = None,
) -> float:
    """ONE rule for "was this cost priced at the G I am about to apply?".

    ``priced`` is the G recorded beside a persisted activation-aware cost
    (``None`` when nothing was recorded -- there is then nothing to disagree
    with, and the caller proceeds).  ``applied`` is the G the caller resolved
    for the same unit now.  Equality is exact because both sides are the same
    F32-rounded ``FP4_MAX[*FP8_MAX]/max_abs`` value of the same maximum; a
    difference therefore means the policy (or the maximum the policy was
    applied to) changed under a retained cost, which is
    :class:`ActivationScalePolicyMismatchError` and never a rounding artefact.

    Both the production cache's score resume and the assignment-KL hook call
    this, so the two cannot answer the question differently (principle 8).
    """
    value = float(applied)
    if priced is None:
        return value
    recorded = float(priced)
    if recorded == value:
        return value
    policies = ""
    if priced_policy is not None or applied_policy is not None:
        policies = (
            f" (priced under {priced_policy!r}, applying "
            f"{applied_policy or resolve_input_global_scale_policy()!r})"
        )
    raise ActivationScalePolicyMismatchError(
        f"{consumer}: {qname!r} carries an activation-aware cost priced at "
        f"input_global_scale={recorded!r} but this run applies "
        f"{value!r}{policies}.  The resolved input-global-scale policy is "
        "part of what that cost means; refusing to reuse it under another "
        "one.  Re-render the affected scores under a fresh --cache-dir, or "
        "restore the policy the cache was priced under "
        "(PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE)."
    )


def _nvfp4_dequantize_registered_codes(
    codes: torch.Tensor,
    stored_scale: torch.Tensor,
    input_global_scale: float,
) -> torch.Tensor:
    """``rounded * used_scale`` for codes the operator produced.

    Split out of the operator leg so the ARITHMETIC is testable without a GPU and
    without a registered operator, and so there is exactly one place that can get
    it wrong.  ``used_scale`` is ``stored_scale / G`` -- the contract's own rule
    (:meth:`StaticActivationContract.quantize_dequantize`'s oracle computes the
    same quotient at ``nvfp4_activation_qdq_served``).  Multiplying by the stored
    scale alone would be ``G`` times too large; a test pins the quotient.

    ``stored_scale`` arrives from :func:`nvfp4_group_stored_scale`, which takes
    each group's amax with ``keepdim=True`` and therefore returns
    ``(..., groups, 1)``: one scale per group, carried with a trailing unit
    axis.  That is the real shape from the real caller (the operator leg), so it
    is folded onto the group axis rather than refused; any other geometry is
    still a refusal.
    """
    # The nibble view the operator's packed output yields is uint8, and
    # ``index_select`` wants a long index: converting here is what keeps the
    # leg usable on a real [M, K] activation instead of only on a fixture that
    # already happened to hand it int64.
    codes = codes.to(torch.long)
    if codes.ndim < 2 or codes.shape[-1] % FP4_GROUP_SIZE:
        raise ValueError(
            "served quantiser codes must end in whole "
            f"{FP4_GROUP_SIZE}-element groups, got {tuple(codes.shape)}"
        )
    expected = (*codes.shape[:-1], codes.shape[-1] // FP4_GROUP_SIZE)
    if tuple(stored_scale.shape) == (*expected, 1):
        stored_scale = stored_scale.reshape(expected)
    if tuple(stored_scale.shape) != expected:
        raise ValueError(
            "served quantiser scale plane does not cover these codes: codes "
            f"{tuple(codes.shape)} need a stored scale of {expected}, got "
            f"{tuple(stored_scale.shape)}"
        )
    stored_scale = stored_scale.to(torch.float32)
    used_scale = stored_scale / float(input_global_scale)
    positive = _e2m1_positive_table(codes.device)
    magnitude = positive.index_select(0, (codes & 0x7).reshape(-1)).reshape(codes.shape)
    signed = torch.where((codes & 0x8).bool(), -magnitude, magnitude)
    output = signed * used_scale.repeat_interleave(FP4_GROUP_SIZE, dim=-1)
    # One group scale covers FP4_GROUP_SIZE elements, so BOTH the value and the
    # zero-scale mask are repeated out to the element axis: a [M, K/16] mask
    # cannot broadcast against a [M, K] output.
    nonzero = (stored_scale != 0).repeat_interleave(FP4_GROUP_SIZE, dim=-1)
    return torch.where(nonzero, output, torch.zeros_like(output))


def _nvfp4_registered_stored_plane(
    rows: torch.Tensor,
    input_global_scale: float,
) -> torch.Tensor:
    """The kernel's own block-scale derivation for the operator leg.

    The operator is handed bf16 rows, but the block scale it stores is
    ``e4m3(amax / 6 * G)`` computed in FP32: the amax is taken over the values
    the operator saw, and the scale arithmetic keeps full precision.  Deriving
    it in bf16 instead rounds the scale itself -- ``keepdim`` amax of a bf16
    tensor stays bf16, and a Python float operand does not promote it -- which
    moves the stored UE4M3 byte on some blocks and moves the dequantised value
    with it (measured on the retained 84 groups: 2 of 128 probe rows in one
    group, 47 -> 46).  One place owns the derivation so the priced scale rule
    and the kernel's are the same object.
    """
    groups = rows.reshape(-1, rows.shape[-1] // FP4_GROUP_SIZE, FP4_GROUP_SIZE)
    return nvfp4_group_stored_scale(groups.float(), input_global_scale).float()


#: The operator's ``input_global_scale`` operand, one device scalar per
#: ``(device, G)``.  Built with the same ``torch.tensor([g], dtype=float32)``
#: expression the leg used per call, so the value is the same float32; only the
#: first call at each G pays the pageable copy and its stream wait.  A process
#: meets one G per priced unit, so the table is bounded by the unit roster.
_SERVED_GLOBAL_SCALES: dict[tuple[torch.device, float], torch.Tensor] = {}
_SERVED_DEQUANT_KERNELS = None


def _served_global_scale(g: float, device: torch.device) -> torch.Tensor:
    """The cached ``float32[1]`` device scalar the operator reads G from."""
    key = (device, g)
    scale = _SERVED_GLOBAL_SCALES.get(key)
    if scale is None:
        scale = torch.tensor([g], dtype=torch.float32, device=device)
        _SERVED_GLOBAL_SCALES[key] = scale
    return scale


def _served_dequant_kernels():
    """The declared dequantisation kernels, or a refusal by name.

    Loaded once.  A module whose ``KERNEL_ID`` is not the one this contract
    declares is refused too: the identity a row carries must name the code that
    priced it.
    """
    global _SERVED_DEQUANT_KERNELS

    module = _SERVED_DEQUANT_KERNELS
    if module is None:
        try:
            from .kernels import nvfp4_served_dequant as module
        except Exception as exc:
            raise ServedQuantizerUnboundError(
                f"the registered-operator leg dequantises with "
                f"{SERVED_QUANTIZER_DEQUANT_KERNEL!r}, and this process cannot "
                f"load it ({type(exc).__name__}: {exc}); refusing rather than "
                "pricing with another implementation (RobTand/prismaquant#1211)"
            ) from exc
        if getattr(module, "KERNEL_ID", None) != SERVED_QUANTIZER_DEQUANT_KERNEL:
            raise ServedQuantizerUnboundError(
                f"prismaquant.kernels.nvfp4_served_dequant declares "
                f"{getattr(module, 'KERNEL_ID', None)!r}, not the "
                f"{SERVED_QUANTIZER_DEQUANT_KERNEL!r} this contract names"
            )
        _SERVED_DEQUANT_KERNELS = module
    return module


def _nvfp4_activation_qdq_registered_op(
    x: torch.Tensor,
    input_global_scale: float,
) -> torch.Tensor:
    """The served operator's own decisions, dequantised the contract's way.

    The codes come from the operator, so the rounding decision is the runtime's
    rather than this tree's.  The block scale is deliberately NOT read back out
    of the operator's returned plane: that plane's 128x4 permutation is the
    hardware's layout and this tree may not own a second copy of it.  The byte is
    derived by :func:`nvfp4_stored_scale_from_amax` -- the rule inside
    :func:`nvfp4_group_stored_scale`, the function the attestation gate compares
    against the published table and the one the retained 84-group differential
    measured identical on 10,752/10,752 blocks -- and the dequantisation is the
    contract's own ``rounded * used_scale`` with ``used_scale = stored_scale / G``.

    HOW IT RUNS (RobTand/prismaquant#1211).  The group maximum and the
    dequantisation are the two Triton kernels in
    :mod:`prismaquant.kernels.nvfp4_served_dequant`, declared as
    :data:`SERVED_QUANTIZER_DEQUANT_KERNEL`; the scale rule and ``stored / G``
    stay the Torch ops they were.  The output is bit-identical to the Torch
    composition this replaced, which is kept as
    :func:`_nvfp4_activation_qdq_registered_op_unfused` so the equality is
    tested rather than asserted.  The operator's ``input_global_scale`` is a
    device scalar cached per ``(device, G)`` (:func:`_served_global_scale`),
    because building it per call was a pageable host-to-device copy that
    blocked the host until the stream drained.  A process that cannot load the
    kernels refuses (:class:`ServedQuantizerUnboundError`); it never falls back
    to the Torch composition.

    The returned plane's shape is not asserted here; a real BF16 capture has to
    establish that layout (row padding in particular) before this leg is priced
    against a serve, so the priced contract is the operator's ELEMENT decisions
    plus this module's scale rule, not a guess about the plane.

    AN EMPTY ACTIVATION IS ALLOCATED, NEVER LAUNCHED.  ``scaled_fp4_quant``
    derives its launch grid from the token count, so a zero-row activation asks
    for an empty grid: measured on the pinned image (2026-09-17, Tessera's
    binding of the SAME registered operator) the call returns,
    ``torch.cuda.synchronize()`` reports nothing, and the next CHECKED launch
    anywhere in the process raises ``CUDA error: invalid argument`` -- the
    error is sticky in the context, so an empty activation batch poisons an
    unrelated later kernel.  Nothing here is the operator's own defect and
    nothing here can fix it, so this leg does not reach it: an empty activation
    has no code to quantise, and its dequantised answer is the empty tensor
    itself, in the input's own shape, dtype and device.  The guards above run
    FIRST, so an empty tensor with a bad last dim, a bad ``G`` or a non-CUDA
    device is still refused by name.
    """
    if x.shape[-1] % FP4_GROUP_SIZE:
        raise ValueError(
            "the served NVFP4 quantiser needs a last dim divisible by "
            f"{FP4_GROUP_SIZE}, got {tuple(x.shape)}"
        )
    g = float(input_global_scale)
    if not math.isfinite(g) or g <= 0.0:
        raise ValueError(f"input_global_scale must be finite and > 0, got {g!r}")
    if x.device.type != "cuda":
        raise ServedQuantizerUnboundError(
            f"torch.ops._C.{SERVED_QUANTIZER_OP} is a CUDA operator; this tensor "
            f"is on {x.device.type}"
        )
    original_shape, original_dtype = x.shape, x.dtype
    if x.numel() == 0:
        # ``new_empty`` preserves this tensor's dtype and device by torch's own
        # contract, which is the answer the non-empty path returns too
        # (``... .to(original_dtype)``).
        return x.new_empty(original_shape)
    kernels = _served_dequant_kernels()
    rows = x.reshape(-1, x.shape[-1]).contiguous().to(torch.bfloat16)
    packed, _scale_plane = torch.ops._C.scaled_fp4_quant(
        rows, _served_global_scale(g, rows.device), True)
    stored = nvfp4_stored_scale_from_amax(kernels.group_abs_max(rows), g).float()
    # The same Torch op, on the same plane, as ``_nvfp4_dequantize_registered_codes``.
    used = stored / g
    output = kernels.dequantize_codes(
        packed.view(torch.uint8).reshape(rows.shape[0], rows.shape[1] // 2),
        stored, used, _e2m1_positive_table(rows.device), original_dtype)
    return output.reshape(original_shape)


def _nvfp4_activation_qdq_registered_op_unfused(
    x: torch.Tensor,
    input_global_scale: float,
) -> torch.Tensor:
    """The pre-#1211 Torch composition of the registered-operator leg.

    Kept verbatim as the oracle the fused leg is tested bit-identical against
    (``tests/test_nvfp4_served_dequant_kernel.py``) and as the A arm of
    ``tools/nvfp4_served_qdq_bench.py``.  Never dispatched: no identity names
    it, so no priced row can be produced by it.
    """
    if x.shape[-1] % FP4_GROUP_SIZE:
        raise ValueError(
            "the served NVFP4 quantiser needs a last dim divisible by "
            f"{FP4_GROUP_SIZE}, got {tuple(x.shape)}"
        )
    g = float(input_global_scale)
    if not math.isfinite(g) or g <= 0.0:
        raise ValueError(f"input_global_scale must be finite and > 0, got {g!r}")
    if x.device.type != "cuda":
        raise ServedQuantizerUnboundError(
            f"torch.ops._C.{SERVED_QUANTIZER_OP} is a CUDA operator; this tensor "
            f"is on {x.device.type}"
        )
    original_shape, original_dtype = x.shape, x.dtype
    if x.numel() == 0:
        # ``new_empty`` preserves this tensor's dtype and device by torch's own
        # contract, which is the answer the non-empty path returns too
        # (``... .to(original_dtype)``).
        return x.new_empty(original_shape)
    rows = x.reshape(-1, x.shape[-1]).contiguous().to(torch.bfloat16)
    packed, _scale_plane = torch.ops._C.scaled_fp4_quant(
        rows,
        torch.tensor([g], dtype=torch.float32, device=rows.device),
        True,
    )
    stored = _nvfp4_registered_stored_plane(rows, g)
    bytes_ = packed.view(torch.uint8).reshape(rows.shape[0], rows.shape[1] // 2)
    codes = torch.stack((bytes_ & 0xF, bytes_ >> 4), dim=-1).reshape(rows.shape)
    output = _nvfp4_dequantize_registered_codes(codes, stored, g)
    return output.reshape(original_shape).to(original_dtype)


@dataclass(frozen=True, slots=True)
class ServedQuantizerIdentity:
    """Which arithmetic priced a served activation row, and what it ran in.

    Resolved ONCE per process/config (``resolve_served_quantizer_identity``) and
    then carried as a frozen value, so a joint row's identity and the arithmetic
    that produced it cannot be two different objects (principle 8), and no hot
    path ever re-probes the environment or re-imports an extension to decide how
    to quantise a tensor.

    ``backend`` is the arithmetic; every other field is the provenance a reader
    needs to tell one build of it from another: ``torch``/``torch_git`` are the
    axes the joint-projection qualification also pins, ``vllm`` names the
    extension's build, ``platform`` is the contract table's key, and
    ``image_content_sha256`` is the launcher-stamped executing-image identity the
    joint pass already refuses on (``joint_projection_backend.executing_image``).

    ``dequant_kernel`` names the implementation of the registered-operator leg's
    dequantisation (:data:`SERVED_QUANTIZER_DEQUANT_KERNEL`), so a row states
    which code priced it.  It is recorded, not a reuse axis: the kernels are
    tested bit-identical to the Torch composition they replaced, so a retained
    cost priced by either is the same number (RobTand/prismaquant#1211).
    """

    backend: str
    op: str | None = None
    platform: str | None = None
    torch: str | None = None
    torch_git: str | None = None
    vllm: str | None = None
    image_content_sha256: str | None = None
    dequant_kernel: str | None = None
    schema: str = SERVED_QUANTIZER_IDENTITY_SCHEMA

    def as_record(self) -> dict:
        """The serialisable form a receipt and a row identity both carry."""
        return {
            "schema": self.schema,
            "backend": self.backend,
            "op": self.op,
            "platform": self.platform,
            "torch": self.torch,
            "torch_git": self.torch_git,
            "vllm": self.vllm,
            "image_content_sha256": self.image_content_sha256,
            "dequant_kernel": self.dequant_kernel,
        }


#: The identity this process resolved, and the one a run bound before pricing.
#: One slot each, on purpose: resolution is a process/config question, not a
#: per-tensor one, and a second answer would be a second identity.
_RESOLVED_SERVED_QUANTIZER: ServedQuantizerIdentity | None = None
_ACTIVE_SERVED_QUANTIZER: ServedQuantizerIdentity | None = None


class ServedQuantizerUnboundError(RuntimeError):
    """A rung whose measurement contract IS the served quantiser reached the
    priced path with no registered-operator binding behind it.

    Raised instead of silently pricing with this module's Torch model.  The two
    are not equivalent objects: the model divides by the used scale where the
    kernel takes its reciprocal through ``rcp.approx.ftz.f32``, and the retained
    84-group differential priced 24 of 172,032 probed elements one E2M1 code
    away from the kernel.  A caller that means to price under the model -- a
    screen, a CPU test -- binds it explicitly; a rung that serves cannot fall
    back into it by omission.
    """


def _served_quantizer_platform() -> str | None:
    """``sm_<major><minor>``, the spelling ``native_platform`` builds."""
    try:
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(None)
    except Exception:  # pragma: no cover - reported as absent, never guessed
        return None
    return f"sm_{major}{minor}"


def _register_served_quantizer_op() -> bool:
    """Whether ``torch.ops._C.scaled_fp4_quant`` is callable after one import.

    The import is the same one the pin's ``native_ops._load_native_ops`` makes
    (``vllm._custom_ops`` registers the ``_C`` namespace) and it happens here and
    nowhere else: no hot path imports, probes or hashes anything.
    """
    if callable(getattr(torch.ops._C, SERVED_QUANTIZER_OP, None)):
        return True
    import vllm._custom_ops  # noqa: F401  -- registers torch.ops._C

    return callable(getattr(torch.ops._C, SERVED_QUANTIZER_OP, None))


def resolve_served_quantizer_identity(
    *,
    require: bool,
    context: str = "served activation quantizer",
) -> ServedQuantizerIdentity:
    """Resolve the served activation quantiser once for this process.

    ``require=True`` is the priced seam: an operator this process cannot register
    is then a REFUSAL, never a fallback to the Torch model.  ``require=False``
    answers what is available (a screen, a CPU preflight, a plan reader) and is
    recorded as such.

    Cached in one slot: the answer is a property of the process, so asking twice
    must not import twice, and a caller that wants a different answer is
    declaring a different process.
    """
    global _RESOLVED_SERVED_QUANTIZER

    resolved = _RESOLVED_SERVED_QUANTIZER
    if resolved is None:
        registered = False
        try:
            registered = _register_served_quantizer_op()
        except Exception:
            # A build with no extension, a CPU-only container, a missing
            # libcuda: all the same answer to "what will price this tensor".
            registered = False
        try:
            import vllm as _vllm

            vllm_version = str(getattr(_vllm, "__version__", "")) or None
        except Exception:
            vllm_version = None
        image = None
        try:
            from .joint_projection_backend import executing_image

            image = executing_image()
        except Exception:  # pragma: no cover - an unstamped launcher
            image = None
        dequant_kernel = None
        if registered:
            try:
                _served_dequant_kernels()
                dequant_kernel = SERVED_QUANTIZER_DEQUANT_KERNEL
            except ServedQuantizerUnboundError:
                # Recorded as absent; ``require`` below and the binding
                # validator both refuse a registered identity without it.
                dequant_kernel = None
        resolved = ServedQuantizerIdentity(
            backend=(SERVED_QUANTIZER_BACKEND_REGISTERED_OP if registered
                     else SERVED_QUANTIZER_BACKEND_MODEL),
            op=SERVED_QUANTIZER_OP if registered else None,
            platform=_served_quantizer_platform(),
            torch=str(torch.__version__),
            torch_git=getattr(torch.version, "git_version", None),
            vllm=vllm_version,
            image_content_sha256=image,
            dequant_kernel=dequant_kernel,
        )
        _RESOLVED_SERVED_QUANTIZER = resolved

    if require and resolved.backend != SERVED_QUANTIZER_BACKEND_REGISTERED_OP:
        raise ServedQuantizerUnboundError(
            f"{context}: this row is served by torch.ops._C.{SERVED_QUANTIZER_OP} "
            "with a static global scale, and that operator is not registered in "
            "this process, so the served arithmetic cannot be run here. "
            "PrismaQuant's own Torch re-implementation of the same rule is NOT a "
            "substitute: it divides by the used scale where the kernel takes "
            "rcp.approx.ftz.f32, and the retained 84-group differential priced 24 "
            "of 172,032 probed elements one E2M1 code away from the kernel "
            "(RobTand/prismaquant#567). Price in an image that registers the "
            "operator, or bind the model explicitly if this is a screen rather "
            "than a price."
        )
    if require and resolved.dequant_kernel != SERVED_QUANTIZER_DEQUANT_KERNEL:
        raise ServedQuantizerUnboundError(
            f"{context}: torch.ops._C.{SERVED_QUANTIZER_OP} is registered, but "
            f"the leg's declared dequantisation kernel "
            f"{SERVED_QUANTIZER_DEQUANT_KERNEL!r} cannot be loaded in this "
            "process, so the registered-operator arithmetic cannot be run here "
            "(RobTand/prismaquant#1211)."
        )
    return resolved


def bind_served_quantizer_identity(
    *,
    require: bool = True,
    identity: ServedQuantizerIdentity | None = None,
    context: str = "served activation quantizer",
) -> ServedQuantizerIdentity:
    """Bind the arithmetic this process prices with, before score or cache work.

    ``identity=None`` resolves (and, with ``require=True``, requires the
    registered operator).  An explicit ``identity`` is how a screen or a CPU test
    declares the model deliberately -- it is a binding, not a fallback: nothing
    here chooses for the caller, and nothing re-binds per tensor.
    """
    global _ACTIVE_SERVED_QUANTIZER

    bound = (identity if identity is not None
             else resolve_served_quantizer_identity(require=False, context=context))
    _validate_served_quantizer_identity(bound, context=context)
    # ``require`` is about the ARITHMETIC, not about how the identity arrived: a
    # caller that demands the served operator cannot satisfy that demand by
    # asserting the model.
    if require and bound.backend != SERVED_QUANTIZER_BACKEND_REGISTERED_OP:
        raise ServedQuantizerUnboundError(
            f"{context}: the priced path requires the registered served "
            f"quantiser torch.ops._C.{SERVED_QUANTIZER_OP}, and this process "
            f"bound {bound.backend!r} instead"
        )
    # A process that has already priced under one arithmetic may not quietly
    # price under another: rendered scores, cache rows and joint rows carry the
    # identity they were produced under, so a second binding is a refusal rather
    # than a silent re-stamp.  Tests and screens re-bind after an explicit
    # ``_reset_served_quantizer_identity_for_tests``.
    existing = _ACTIVE_SERVED_QUANTIZER
    if existing is not None and existing != bound:
        raise ServedQuantizerUnboundError(
            f"{context}: this process already priced rows under "
            f"{existing.backend!r} ({existing.op or 'no operator'}, image "
            f"{existing.image_content_sha256 or 'unstamped'}) and cannot be "
            f"re-bound to {bound.backend!r} afterwards; rows produced under the "
            "first binding would be read as rows of the second"
        )
    _ACTIVE_SERVED_QUANTIZER = bound
    return bound


def _validate_served_quantizer_identity(
    identity: ServedQuantizerIdentity,
    *,
    context: str,
) -> None:
    """Refuse an identity that cannot stand behind the provenance it claims.

    A served row's identity is what a reader uses to tell one build of the
    operator from another, so a registered-operator binding that does not name
    the operator, the platform it ran on or the build it ran in is not a weaker
    identity -- it is an unusable one, and it is refused here rather than
    published (RobTand/prismaquant#567).
    """
    if identity.schema != SERVED_QUANTIZER_IDENTITY_SCHEMA:
        raise ServedQuantizerUnboundError(
            f"{context}: unknown served-quantizer identity schema "
            f"{identity.schema!r}")
    if identity.backend == SERVED_QUANTIZER_BACKEND_MODEL:
        return
    if identity.backend != SERVED_QUANTIZER_BACKEND_REGISTERED_OP:
        raise ServedQuantizerUnboundError(
            f"{context}: unknown served-quantizer backend {identity.backend!r}; "
            f"the known ones are {SERVED_QUANTIZER_BACKEND_REGISTERED_OP!r} and "
            f"{SERVED_QUANTIZER_BACKEND_MODEL!r}"
        )
    missing = [name for name, value in (
        ("op", identity.op), ("platform", identity.platform),
        ("torch", identity.torch), ("vllm", identity.vllm),
        ("image_content_sha256", identity.image_content_sha256),
        ("dequant_kernel", identity.dequant_kernel),
    ) if not value]
    if identity.op != SERVED_QUANTIZER_OP:
        raise ServedQuantizerUnboundError(
            f"{context}: a served binding must name "
            f"{SERVED_QUANTIZER_OP!r}, got {identity.op!r}")
    if identity.dequant_kernel is not None and (
            identity.dequant_kernel != SERVED_QUANTIZER_DEQUANT_KERNEL):
        raise ServedQuantizerUnboundError(
            f"{context}: a served binding names dequantisation kernel "
            f"{identity.dequant_kernel!r}; this build implements "
            f"{SERVED_QUANTIZER_DEQUANT_KERNEL!r} and nothing else")
    if missing:
        raise ServedQuantizerUnboundError(
            f"{context}: a served binding is missing {', '.join(missing)}; an "
            "identity that cannot name the build, the platform and the image it "
            "ran in does not attest which arithmetic priced the row"
        )


def active_served_quantizer_identity() -> ServedQuantizerIdentity | None:
    """The bound identity, or ``None`` when nothing has bound one yet."""
    return _ACTIVE_SERVED_QUANTIZER


def effective_served_quantizer_identity(contract=None) -> ServedQuantizerIdentity | None:
    """The ONE answer to "which arithmetic is this row priced with".

    A contract may carry its own binding (an explicit, frozen choice) and the
    process may carry another (the run's binding).  The explicit one wins -- and
    it has to win in exactly one place, because three callers must agree:

    * :meth:`StaticActivationContract.quantize_dequantize`, which runs the
      arithmetic;
    * the render-score writer, which stamps which arithmetic priced the row; and
    * :func:`require_matching_served_quantizer`, which decides on reuse.

    When they asked separately they could disagree -- a contract explicitly
    bound to one arithmetic pricing while the record stamped the other (or
    nothing), which is a row that lies about how it was produced.
    """
    bound = getattr(contract, "served_quantizer", None)
    return bound if bound is not None else _ACTIVE_SERVED_QUANTIZER


#: The identity axes that decide whether a retained activation-aware cost may be
#: reused.  They are the arithmetic (``backend``, ``op``) and the build that
#: arithmetic ran in (``platform``, ``torch``, ``torch_git``, ``vllm``, the
#: launcher-stamped ``image_content_sha256``).  A different build of the same
#: operator is a different quantiser to price against: the producer build and
#: the serving build are already known to differ (§ the closure note), and the
#: published table could not tell them apart.
SERVED_QUANTIZER_REUSE_AXES = (
    "backend", "op", "platform", "torch", "torch_git", "vllm",
    "image_content_sha256",
)


def served_quantizer_reuse_differences(recorded, current) -> list[str]:
    """The fields on which two served-quantiser identities price different numbers.

    ``schema`` plus :data:`SERVED_QUANTIZER_REUSE_AXES`, in that order. Either
    side is a record (a mapping, as ``as_record`` writes it) or a
    :class:`ServedQuantizerIdentity`. Values are compared as written, never
    through ``str``: ``None`` and the string ``'None'`` are different claims.
    A field outside these, such as ``dequant_kernel`` (#1211), is recorded
    and never compared: both implementations it names give bit-identical
    output, so a cost priced by either is the same number.
    """
    def value(side, field):
        return side.get(field) if isinstance(side, Mapping) else getattr(side, field)

    return [field for field in ("schema", *SERVED_QUANTIZER_REUSE_AXES)
            if value(recorded, field) != value(current, field)]


def require_matching_served_quantizer(
    recorded: "Mapping[str, object] | None",
    *,
    qname: str,
    consumer: str,
    contract=None,
) -> None:
    """Refuse to reuse an activation-aware cost under another arithmetic.

    The render-score key is ``qname|FMT`` and the retained record carries the
    static G it was priced at; neither changes when the *arithmetic* changes.
    A model-priced row and a registered-operator row are different objects --
    the retained 84-group differential priced 24 of 172,032 probed elements one
    E2M1 code apart -- so a cache filled under one is refused for the other
    rather than silently contributing its cost (RobTand/prismaquant#567).

    ``recorded`` is the identity the row was priced under, or ``None`` when the
    record carries none (a row written before the stamp existed, or one priced
    with nothing bound).  Callers pass it only for rows that carry a static G:
    a weight-only or dynamically scored row has no activation quantiser in its
    cost, and stays reusable, which is what keeps the exemption mathematically
    honest rather than convenient.
    """
    current = effective_served_quantizer_identity(contract)
    if current is None:
        raise ServedQuantizerUnboundError(
            f"{consumer}: {qname!r} carries an activation-aware cost, but this "
            "process has bound no activation quantizer, so there is nothing the "
            "retained cost can be checked against"
        )
    if not isinstance(recorded, Mapping):
        if current.backend == SERVED_QUANTIZER_BACKEND_REGISTERED_OP:
            raise ServedQuantizerUnboundError(
                f"{consumer}: {qname!r} carries an activation-aware cost with no "
                "served-quantizer identity recorded, so it cannot be reused as "
                "registered-operator pricing: a record that predates the stamp "
                "was priced by this tree's Torch model, and the two disagree by "
                "one E2M1 code on the midpoint cases the 84-group differential "
                "measured. Re-render the affected scores under a fresh "
                "--cache-dir, or price under the model explicitly "
                "(RobTand/prismaquant#567)."
            )
        return
    # The stamp's own vocabulary is checked before its values: a record written
    # by another schema is a different claim, not a weaker one.
    if recorded.get("schema") != SERVED_QUANTIZER_IDENTITY_SCHEMA:
        raise ServedQuantizerUnboundError(
            f"{consumer}: {qname!r} carries a served-quantizer identity of "
            f"schema {recorded.get('schema')!r}; this reader transcribes "
            f"{SERVED_QUANTIZER_IDENTITY_SCHEMA!r} and will not compare fields "
            "it may be misreading"
        )
    # Compared as written, never through ``str``: ``None`` and the STRING
    # ``'None'`` are different claims about the operator.
    differing = served_quantizer_reuse_differences(recorded, current)
    if differing:
        raise ServedQuantizerUnboundError(
            f"{consumer}: {qname!r} was priced under "
            f"{recorded.get('backend')!r} (op {recorded.get('op')!r}, build "
            f"{recorded.get('vllm') or 'unnamed'}, image "
            f"{str(recorded.get('image_content_sha256') or 'unstamped')[:12]}) "
            f"and this run prices under {current.backend!r} (op {current.op!r}, "
            f"build {current.vllm or 'unnamed'}, image "
            f"{str(current.image_content_sha256 or 'unstamped')[:12]}). The "
            f"differing axes are {', '.join(differing)}; a cost priced by one "
            "build of the quantiser is not a cost of another. Re-render the "
            "affected scores under a fresh --cache-dir"
        )


def _reset_served_quantizer_identity_for_tests() -> None:
    """Drop both slots so a test can observe resolution from a clean process."""
    global _RESOLVED_SERVED_QUANTIZER, _ACTIVE_SERVED_QUANTIZER
    _RESOLVED_SERVED_QUANTIZER = None
    _ACTIVE_SERVED_QUANTIZER = None


@dataclass(frozen=True)
class StaticActivationContract:
    """What a spec's activations execute as under a STATIC per-unit scale.

    The served W4A4 kernel (vLLM ``scaled_fp4_quant``) takes a calibrated
    ``input_global_scale`` G per unit, snaps each ``group_size`` block's scale
    to UE4M3 relative to G, and rounds to the E2M1 grid -- the oracle
    :func:`nvfp4_activation_qdq_served`.  That is a different quantiser from
    the dynamic FP32-scale RTN a registry row's ``activation_quantize_dequantize``
    runs without a G, and the two disagree hardest exactly where it matters
    (a block far below the calibration amax underflows to zero when served).

    A ``FormatSpec`` that is served this way carries one of these as
    ``static_activation_contract`` so the question "which activation quantizer
    does this spec serve, and what is its G for this unit" is answered by the
    spec -- never by comparing the format's NAME to ``"NVFP4"``, which a
    Tessera rung routed through the same kernel does not have.

    ``measured_as_served`` is the measurement policy the spec asks for:

    * ``False`` -- stock ``NVFP4``: the dynamic RTN stays the long-standing
      screen baseline and the served emulation is the documented env opt-in
      (``PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES``, runtime_flags.md);
    * ``True`` -- a Tessera W4A4 rung: the plugin has no dynamic path, so the
      served oracle IS the measurement, in the hooks and the cache scorer as
      it already is in the campaign (#196), and a unit without a calibrated
      maximum refuses (:class:`ActivationScaleContractError`) rather than
      being priced under a quantiser the runtime never runs.
    """

    execution: str = NVFP4_ACTIVATION_EXECUTION
    group_size: int = FP4_GROUP_SIZE
    measured_as_served: bool = False
    #: The arithmetic a ``measured_as_served`` row was priced by, frozen before
    #: any score or cache work (``bind_served_quantizer_identity``).  ``None``
    #: means "this process has not said", which is a refusal for a served rung,
    #: not an invitation to pick one by capability.
    served_quantizer: ServedQuantizerIdentity | None = None

    def input_global_scale_from_max_abs(
        self,
        max_abs: float,
        *,
        policy: str | None = None,
    ) -> float:
        """The owner's G rule at the resolved policy -- the same number the
        campaign stamps (``_static_input_scales``) and the export ships.

        ``policy`` is deliberately optional and deliberately NOT a field of
        this frozen dataclass (#227).  The scale policy is a live process
        setting wherever nothing has priced anything yet -- a screen, a fresh
        fill, an export -- so the default stays "resolve it now".  A caller
        that has already resolved ONE policy for a whole operation (the
        production cache's render levers, the campaign's ``_static_input
        _scales``) passes it here instead, so the G it stamps and the G it
        scores with cannot drift within that operation.
        """
        return input_global_scale_from_max_abs(
            max_abs,
            policy=resolve_input_global_scale_policy(policy),
        )

    def require_input_global_scale(
        self,
        max_abs: float | None,
        *,
        qname: str | None,
        consumer: str,
        policy: str | None = None,
    ) -> float:
        """G for ``qname``, or the refusal by name when it has no maximum."""
        if max_abs is None or not math.isfinite(float(max_abs)) or float(max_abs) <= 0.0:
            raise ActivationScaleContractError(
                f"{consumer}: {qname!r} is served under the static "
                f"{self.execution} activation contract but has no calibrated "
                f"activation maximum (got {max_abs!r}); refusing to price it "
                "under a dynamic quantiser the runtime never executes.  The "
                "production cache's activation_max_abs (fused-sibling unified) "
                "is the scale identity this needs."
            )
        return self.input_global_scale_from_max_abs(
            float(max_abs), policy=policy)

    def quantize_dequantize(
        self,
        x: torch.Tensor,
        input_global_scale: float,
    ) -> torch.Tensor:
        """The served arithmetic at G.  No pre-clip: the clamp lives in G.

        Which arithmetic that is comes from the binding, never from import
        order or GPU visibility: a registered-operator binding runs the operator
        the serve runs, an explicit model binding runs this module's oracle, and
        a served rung with no binding refuses rather than pricing a model of the
        server.
        """
        identity = effective_served_quantizer_identity(self)
        if identity is None:
            if self.measured_as_served:
                raise ServedQuantizerUnboundError(
                    "a contract whose measurement IS the served quantiser reached the "
                    f"priced path with no binding: bind torch.ops._C.{SERVED_QUANTIZER_OP} "
                    f"({SERVED_QUANTIZER_BACKEND_REGISTERED_OP}), or bind "
                    f"{SERVED_QUANTIZER_BACKEND_MODEL!r} "
                    "explicitly to price under the model on purpose "
                    "(RobTand/prismaquant#567)."
                )
            return nvfp4_activation_qdq_served(x, input_global_scale)
        if identity.backend == SERVED_QUANTIZER_BACKEND_REGISTERED_OP:
            if identity.dequant_kernel != SERVED_QUANTIZER_DEQUANT_KERNEL:
                raise ServedQuantizerUnboundError(
                    "a registered-operator binding reached the priced path naming "
                    f"dequantisation kernel {identity.dequant_kernel!r}; this build "
                    f"prices only with {SERVED_QUANTIZER_DEQUANT_KERNEL!r} "
                    "(RobTand/prismaquant#1211)"
                )
            return _nvfp4_activation_qdq_registered_op(x, input_global_scale)
        if identity.backend == SERVED_QUANTIZER_BACKEND_MODEL:
            return nvfp4_activation_qdq_served(x, input_global_scale)
        # An unrecognised backend is never a reason to price with the model.
        raise ServedQuantizerUnboundError(
            "unknown served-quantizer backend "
            f"{identity.backend!r}; refusing to price with any arithmetic this "
            "process did not bind"
        )


# The one served static-scale contract there is today, as stock ``NVFP4``
# carries it (screen default).  A spec routed through the same kernel whose
# measurement must be the served contract derives from this with
# ``dataclasses.replace(..., measured_as_served=True)``.
NVFP4_SERVED_ACTIVATION_CONTRACT = StaticActivationContract()


__all__ = [
    "ACTIVATION_SCALE_GROUPING_PER_UNIT",
    "ACTIVATION_SCALE_GROUPING_EXECUTED",
    "ActivationScaleContractError",
    "ActivationScalePolicyMismatchError",
    "NVFP4_SERVED_ACTIVATION_CONTRACT",
    "StaticActivationContract",
    "ServedQuantizerIdentity",
    "ServedQuantizerUnboundError",
    "SERVED_QUANTIZER_BACKEND_MODEL",
    "SERVED_QUANTIZER_BACKEND_REGISTERED_OP",
    "SERVED_QUANTIZER_DEQUANT_KERNEL",
    "SERVED_QUANTIZER_IDENTITY_SCHEMA",
    "SERVED_QUANTIZER_OP",
    "active_served_quantizer_identity",
    "bind_served_quantizer_identity",
    "resolve_served_quantizer_identity",
    "CALIBRATION_SOURCE_PACKED_EXPERT_RENDER",
    "CALIBRATION_SOURCE_PARENT_MODULE_CACHE",
    "CALIBRATION_SOURCE_SUPPLEMENTAL_MAX_ABS",
    "CALIBRATION_SOURCE_SUPPLEMENTAL_MODULE_INPUT",
    "CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY",
    "CALIBRATION_SOURCE_TARGET_CACHE",
    "FP4_E2M1_MAX",
    "FP4_GROUP_SIZE",
    "FP8_E4M3_MAX",
    "FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY",
    "LEGACY_INPUT_GLOBAL_SCALE_POLICY",
    "MSE_GRID_INPUT_GLOBAL_SCALE_POLICY",
    "NVFP4_ACTIVATION_CONTRACT_KEY",
    "NVFP4_ACTIVATION_CONTRACT_SCHEMA",
    "NVFP4_ACTIVATION_CONTRACT_SCHEMA_V2",
    "NVFP4_ACTIVATION_EXECUTION",
    "NVFP4_CALIBRATION_SOURCES",
    "NVFP4_INPUT_GLOBAL_SCALE_POLICIES",
    "NVFP4_INPUT_GLOBAL_SCALE_SUFFIX",
    "NVFP4_ROUTED_MOE_STAGES",
    "NVFP4_ROUTED_MOE_STAGE_KEY",
    "NVFP4_ROUTED_MOE_STAGE_SCHEMA",
    "NVFP4_STAGE_CALIBRATION_SOURCES",
    "NVFP4_STAGE_W2",
    "NVFP4_STAGE_W13",
    "ROUTED_EXECUTED_SCALE_GROUPING_SCHEMA",
    "UNCALIBRATED_INPUT_GLOBAL_SCALE",
    "build_execution_contract",
    "build_routed_moe_stage_attestation",
    "calibrated_input_global_scales",
    "calibrated_input_global_scales_with_sources",
    "fused_dense_group",
    "fused_sibling_group_key",
    "group_fused_sibling_targets",
    "input_global_scale_from_max_abs",
    "input_global_scale_tensor",
    "load_activation_cache_max_abs",
    "load_activation_cache_samples",
    "nvfp4_activation_qdq_served",
    "require_matching_input_global_scale",
    "resolve_input_global_scale_policy",
    "INPUT_GLOBAL_SCALE_POLICY_METADATA_KEY",
    "resolve_input_global_scale_value",
    "is_routed_expert_projection_name",
    "routed_expert_scale_group",
    "routed_executed_max_abs",
    "routed_executed_max_abs_for_census",
    "grouping_member",
    "executed_static_max_abs",
    "executed_static_max_abs_map",
    "routed_static_scale_grouping",
    "routed_moe_attested_module_names",
    "routed_moe_stage",
    "routed_moe_stages_sha256",
    "select_mse_grid_input_global_scale",
    "stage_values_sha256",
    "target_values_sha256",
    "unify_fused_sibling_max_abs",
    "unify_fused_sibling_input_global_scales",
]
