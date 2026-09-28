"""The Tessera lane's plugin: the hooks core reaches this lane through.

``lane_specs/tessera.json`` names this module as the lane's ``plugin`` and
declares the ``tessera`` format family (``TESSERA_`` names) as data. Core
modules -- the format registry, the render path, the cache-miss fallbacks,
the serving profiles -- never import a ``tessera_*`` module; they look a hook
up here through ``lane_spec`` (decoupling step 6, PQ #1550).

Two rules keep the seam honest:

* **The import is free.** This module imports no ``tessera_*`` module and not
  the ``tessera`` package at module scope, so resolving a hook costs a stock
  run nothing. Every hook imports what it needs when it is called.
* **Hooks bind at call time.** Each hook reads the lane function through its
  module when it runs, exactly as the in-function imports it replaces did, so
  a test that substitutes ``tessera_menu.route_admission`` still reaches every
  caller.
"""
from __future__ import annotations

from typing import Any

#: The family this lane declares in ``lane_specs/tessera.json``.
FAMILY_ID = "tessera"


def is_tessera_format_name(name: object) -> bool:
    """True for a name the Tessera family claims, without importing Tessera.

    The family's name grammar is data in the lane spec; this reads it there.
    For lane modules that need the question by name. Core asks
    ``format_registry.format_family_of`` instead.
    """
    from .lane_spec import format_family_for_name

    family = format_family_for_name(name)
    return family is not None and family.id == FAMILY_ID


# -- format family hooks (format_registry) ---------------------------------

def synthesize_format(canonical: str):
    """A ``FormatSpec`` for one Tessera rung, or ``None`` for a non-rung."""
    from .tessera_render import synthesize_tessera_spec

    return synthesize_tessera_spec(canonical)


def format_admitted_in_contexts(canonical: str, contexts) -> bool:
    """Does at least one of these serving contexts admit ``canonical``?

    Shared-menu intake only: each candidate still passes its own unit's scope.
    """
    from .tessera_menu import menu_mode, route_admission

    return any(route_admission(canonical, serving_context=context).admits(menu_mode())
               for context in contexts)


def render_production(weight, fmt: str, *, qname, activations, levers):
    """The production render: the H-aware encode that ships, decoded."""
    from .tessera_render import render_tessera_production

    return render_tessera_production(
        weight,
        fmt,
        qname=qname,
        activations=activations,
        levers=levers,
    )


# -- the family's name grammar (format_registry, the allocator and its DP) --

def promotion_class(fmt: object) -> str:
    """The family a rung's serving unit must share: the name without its rate."""
    from .tessera_formats import format_promotion_class

    return format_promotion_class(fmt)


def is_group_option(fmt: object) -> bool:
    """True for a whole-group option name (one family, a rung per member)."""
    from .tessera_formats import is_tessera_group_option

    return is_tessera_group_option(fmt)


def group_option_name(promotion_class: str, index: int) -> str:
    """The name of the ``index``-th whole-group option of one family."""
    from .tessera_formats import tessera_group_option_name

    return tessera_group_option_name(promotion_class, index)


def fused_shared_signature(fmt: object, shared_fields):
    """What ``fmt`` commits one fused module to, over the licence's shared set."""
    from .tessera_formats import fused_shared_signature as signature

    return signature(fmt, shared_fields)


def parse_format_name(fmt: object):
    """``(family, rung)``, ``None`` for a non-rung, or the family's own error."""
    from .tessera_formats import parse_tessera_format_name

    return parse_tessera_format_name(fmt)


# -- candidate admission (allocator_candidates) ------------------------------

def route_admission(name: str, **scope):
    """The pinned runtime's verdict on one rung, under an optional serving scope."""
    from .tessera_menu import route_admission as admission

    return admission(name, **scope)


def menu_mode(value: "str | None" = None) -> str:
    """The menu mode in force; ``value`` overrides the environment."""
    from .tessera_menu import menu_mode as mode

    return mode(value)


def tensor_parallel_applicability(fmt: str, *, qname, target_profile,
                                  in_features: int, out_features: int,
                                  packed_expert: bool):
    """Is this rung legal on the shard each rank will hold?

    A Tessera unit's trellis runs across ``arity x 32``, and its rate schedule
    is realisable only over column counts the Bresenham root divides, so a
    rung can be legal on a tensor and illegal on an Nth of it. The granularity
    is read through ``tessera_menu.tessera_shard_granularity``, which asks
    Tessera's own derivation. A name this family does not parse is legal here:
    the gate adds a refusal, it never invents one.
    """
    from .allocator_candidates import FormatApplicability, TP_SHARD_REASON
    from .serving_profiles import load_serving_profile
    from .tessera_menu import (
        MENU_ATTESTED, PARALLEL_NONE, TesseraMenuError, menu_mode,
        tessera_tp_legal,
    )
    from .tessera_formats import parse_tessera_format_name

    if not str(fmt).startswith("TESSERA_"):
        return FormatApplicability(True)
    try:
        profile = load_serving_profile(target_profile)
    except FileNotFoundError:
        # ``None`` is the research spelling and loads above.  A *string* that
        # names no profile is a typo, and a typo is not a research run: the
        # same ``profile_mismatch`` ``check_serving_format`` answers one seam
        # over (#120).  Loading ``research`` here priced Tessera rungs under
        # the research world size for a profile the export would then refuse.
        return FormatApplicability(
            False,
            reason="profile_mismatch",
            detail=f"unknown target profile {target_profile!r}",
            provenance={"target_profile": target_profile},
        )
    world = int(profile.tensor_parallel.world_size)
    kind = (
        PARALLEL_NONE if packed_expert
        else profile.tensor_parallel.kind_for(qname)
    )
    provenance = {"tp_degree": world, "tp_parallel_kind": kind}
    try:
        parsed = parse_tessera_format_name(fmt)
    except Exception as exc:
        return FormatApplicability(
            False, "unknown_format", str(exc), provenance)
    if parsed is None:
        return FormatApplicability(True)
    family, rung = parsed
    try:
        legal, reason = tessera_tp_legal(
            family, rung, (out_features, in_features),
            tp_degree=world, parallel_kind=kind, unit=qname,
            require_attested_world=(menu_mode() == MENU_ATTESTED),
        )
    except TesseraMenuError as exc:
        return FormatApplicability(False, TP_SHARD_REASON, str(exc), provenance)
    if legal:
        return FormatApplicability(True, None, "", provenance)
    return FormatApplicability(False, TP_SHARD_REASON, reason, provenance)


# -- the fused-module licence (allocator_candidates, allocator_solver) -------

def fused_module_licence():
    """What the pinned runtime lets one fused module's roles disagree about.

    ``None`` is the absence of a licence (no contract pinned), never a
    permissive default. Read through ``tessera_menu``'s one contract read.
    """
    from .tessera_menu import fused_module_licence as licence

    return licence()


def fused_module_fields():
    """``(rung_fields, shape_fields, rate_field)``: the licence's vocabulary.

    The fields a rung decides, the fields fixed before the allocator chooses,
    and the one field the group knapsack varies. A licence field outside all
    three is one the allocator cannot evaluate.
    """
    from .tessera_formats import (
        FUSED_MODULE_RATE_FIELD, FUSED_MODULE_RUNG_FIELDS,
        FUSED_MODULE_SHAPE_FIELDS,
    )

    return FUSED_MODULE_RUNG_FIELDS, FUSED_MODULE_SHAPE_FIELDS, FUSED_MODULE_RATE_FIELD


# -- cost tables (prepriced_cost, unit_topology_restamp) ---------------------

def hessian_identity(costs, *, references=None) -> dict:
    """The one Hessian identity a cost table's rungs share, or a refusal."""
    from .tessera_menu import assert_uniform_hessian_identity

    return assert_uniform_hessian_identity(costs, references=references)


def restamp_unit_topology(payload, profile, *, input_sha256=None):
    """Stamp each row without producer topology from the profile grammar."""
    from .tessera_serving_scope import restamp_unit_topology as restamp

    return restamp(payload, profile, input_sha256=input_sha256)


# -- serving profile hooks (serving_profiles) --------------------------------

def resolved_serving_lane(fmt: str, *, runtime_version: str,
                          serving_context: Any = None):
    """The route the Tessera admission seam resolves for one rung."""
    from .tessera_menu import tessera_resolved_serving_lane

    return tessera_resolved_serving_lane(
        fmt, runtime_version=runtime_version,
        **({"serving_context": serving_context}
           if serving_context is not None else {}))


def require_canonical_subfamily(value: object, *, owner: str) -> str:
    """Refuse a value that is not a canonical Tessera family name."""
    from .tessera_formats import get_tessera_family, TesseraFormatError

    try:
        family = get_tessera_family(value)
    except TesseraFormatError as exc:
        raise ValueError(f"{owner}: invalid Tessera family {value!r}") from exc
    if not isinstance(value, str) or family.name != value:
        raise ValueError(f"{owner}: expected a canonical Tessera family, got {value!r}")
    return value


def format_subfamily(canonical: str) -> str | None:
    """The Tessera family a rung name belongs to, or ``None`` if it parses to none."""
    from .tessera_formats import parse_tessera_format_name, TesseraFormatError

    try:
        parsed = parse_tessera_format_name(canonical)
    except TesseraFormatError:
        return None
    return None if parsed is None else parsed[0].name


# -- the pinned serving runtime (serving_profiles) ---------------------------

class ServingRuntimePinError(ValueError):
    """The lane's serving-runtime pin exists and cannot be read."""


def serving_runtime_pin_path():
    from .tessera_serving_runtime_pin import tessera_serving_runtime_pin_path

    return tessera_serving_runtime_pin_path()


def load_serving_runtime_pin():
    """The pin, or :class:`ServingRuntimePinError` when it is malformed."""
    from .tessera_serving_runtime_pin import (
        TesseraServingRuntimePinError, load_tessera_serving_runtime_pin,
    )

    try:
        return load_tessera_serving_runtime_pin()
    except TesseraServingRuntimePinError as exc:
        raise ServingRuntimePinError(str(exc)) from exc


def serving_runtime_contract_path():
    """The ``runtime_contract.json`` the importable serving runtime packages."""
    from .tessera_runtime_contract import contract_path

    return contract_path()
