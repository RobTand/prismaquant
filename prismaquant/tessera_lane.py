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


def fused_signature(fmt: object, shared_fields):
    """What ``fmt`` commits one fused module to, over the licence's shared set."""
    from .tessera_formats import fused_shared_signature as signature

    return signature(fmt, shared_fields)


def parse_format_name(fmt: object):
    """``(family, rung)``, ``None`` for a non-rung, or the family's own error."""
    from .tessera_formats import parse_tessera_format_name

    return parse_tessera_format_name(fmt)


# -- candidate admission (allocator_candidates) ------------------------------

def rung_admission(name: str, *, allowability=None, require_allowability=False, **scope):
    """The pinned runtime and measured geometry share one admission seam."""
    from .tessera_menu import route_admission as admission
    from .tessera_formats import parse_tessera_format_name

    table = None
    if allowability is not None:
        parsed = parse_tessera_format_name(name)
        table = None if parsed is None else allowability.get(parsed[0].name)
    return admission(name, allowability=table, require_allowability=require_allowability, **scope)


def menu_mode_in_force(value: "str | None" = None) -> str:
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

def fused_licence():
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


def restamp_topology(payload, profile, *, input_sha256=None):
    """Stamp each row without producer topology from the profile grammar."""
    from .tessera_serving_scope import restamp_unit_topology as restamp

    return restamp(payload, profile, input_sha256=input_sha256)


# -- the allocation protocol (allocator) -------------------------------------
#
# ``allocator.main`` reads these off the one lane plugin that provides
# ``allocation_menu`` (``allocator._allocation_lane``). Each hook is the lane
# half of a step the allocator used to take by importing a ``tessera_*``
# module; the code is moved here unchanged, so a run prints, refuses and
# stamps exactly what it did.

def allocation_arguments(parser) -> None:
    """The lane's allocator flags: the serving scope and the selection request."""
    from .tessera_serving_scope import add_serving_scope_arguments

    add_serving_scope_arguments(parser)
    parser.add_argument("--tessera-materialization-plan", default=None,
                        help="Write a non-exportable selected-wire request here instead of layer-config; "
                             "finalize through prismaquant.tessera_materialization after selected wires exist")
    parser.add_argument("--tessera-rung-allowability-root", default=None,
                        help="D41 publication root containing the current index.json")
    parser.add_argument("--tessera-rung-kernel-builds", default=None,
                        help="Independent observed format-to-kernel_build JSON; required with D41 root")


def allocation_serving_target(args, *, target_platform):
    """The explicit serving target the flags name, or ``None`` for none."""
    from .tessera_serving_scope import serving_target_from_args

    return serving_target_from_args(args, target_platform=target_platform)


def allocation_contexts(serving_target, stats, profile):
    """Each unit's serving context under ``serving_target``, or ``None``."""
    from .tessera_serving_scope import context_by_unit_from_stats

    return context_by_unit_from_stats(serving_target, stats, profile)


def allocation_unit_context(serving_target, unit, profile):
    """One unit's serving context, its structure read from the profile grammar."""
    from .tessera_serving_scope import unit_structure_from_profile

    return serving_target.context(unit_structure_from_profile(unit, profile))


def allocation_scope_meta(serving_target, context_by_unit) -> dict:
    """The serving-scope block an allocation stamps when a target was named."""
    from .tessera_serving_scope import scope_provenance

    return {"tessera_serving_scope": scope_provenance(serving_target, context_by_unit)}


def allocation_rung_allowability(args):
    """Load all explicitly supplied D41 builds once, through producer admission."""
    from .rung_allowability import load_rung_allowability, read_allowability_json
    from .lane_eligibility import load_published_formats
    from .tessera_runtime_contract import contract_path

    root = args.tessera_rung_allowability_root
    builds_path = args.tessera_rung_kernel_builds
    if root is None and builds_path is None:
        return None
    if root is None or builds_path is None:
        raise ValueError("D41 allowability root and independent kernel builds are required together")
    formats = load_published_formats(contract_path=contract_path())
    return {family: load_rung_allowability(root, format_entry=formats[family],
                                         expected_kernel_build=build)
            for family, build in read_allowability_json(builds_path).items()}


def allocation_hessian_identity(costs, cost_data) -> dict:
    """One Hessian identity per cost table, or refuse.

    The Tessera encoder's shipping default consumes a per-unit XtX, so rows
    priced with and without one describe different bytes at the same format
    name, and the DP would trade them against each other. Raises on a mix;
    reports what it found otherwise, so "no claim" stays distinguishable from
    "a matching claim" (principle 14). A joined table whose overlay rows carry
    another reference file's seal over the same per-unit H is one identity
    (RobTand/prismaquant#1270); the files are found through the table's own
    hash-bound inputs, and only when a second seal appears.
    """
    from .joint_catalog_extension import hessian_references
    from .tessera_menu import assert_uniform_hessian_identity

    tessera_hessian_identity = assert_uniform_hessian_identity(
        costs, references=lambda: hessian_references(cost_data))
    if tessera_hessian_identity.get("stamped_rows") or \
            tessera_hessian_identity.get("unstamped_rows"):
        print(f"[alloc] tessera hessian identity: "
              f"supplied={tessera_hessian_identity['supplied']} "
              f"tokens={tessera_hessian_identity['token_count']} "
              f"sha={str(tessera_hessian_identity['text_sha'])[:12]} "
              f"({tessera_hessian_identity['stamped_rows']} stamped, "
              f"{tessera_hessian_identity['unstamped_rows']} unstamped rows)")
        if tessera_hessian_identity.get("captures"):
            print("[alloc] tessera hessian captures (per-unit content equal): "
                  + ", ".join(f"{digest[:12]}={n}" for digest, n in
                              tessera_hessian_identity["captures"].items())
                  + f"; export binds {str(tessera_hessian_identity['capture_sha256'])[:12]}")
    return tessera_hessian_identity


def allocation_runtime_identity() -> dict:
    """Which runtime contract answered this run's route queries, if any.

    The block names the exact Tessera commit and consumed contract digest;
    its presence does not by itself identify a development override or prove
    that every candidate route was attested for the requested context. Read
    through ``tessera_menu``'s ONE read, not through ``load_tessera_contract``
    directly: the menu's every attestation goes through that function, and a
    provenance block that read the pin a second time could name a table the
    menu never consulted. "Which table answered" is one fact per run, so it
    is read once.
    """
    from .tessera_menu import tessera_runtime_contract

    contract = tessera_runtime_contract()
    tessera_dev_pin = {} if contract is None else contract.identity()
    if tessera_dev_pin:
        from .tessera_runtime_contract import describe_dev_pin
        print(f"[alloc] tessera dev pin: {describe_dev_pin(tessera_dev_pin)}")
    return tessera_dev_pin


def allocation_shape_price_scope(serving_target, *, tensor_parallel):
    """What a PACT shape-time table must equal, and what admits its launches.

    Returns ``(ShapeTableScope, eligibility table, published formats)``. The
    scope is the TRACKED serving pin, after the live gate has checked the
    installed contract's bytes against it (``require_pinned_tessera_runtime``),
    plus the serving target the flags name and the world size the caller
    declares; the eligibility table and the format rows are the installed
    contract's own (``tessera_render._pinned_serving_table``). The table's
    ``tessera_commit`` is the commit the serve installs, so it is compared with
    the pin's ``serving_commit``.
    """
    from . import tessera_render, tessera_serving_runtime_pin as pins
    from .shape_runtime_prices import ShapeTableScope

    if serving_target is None:
        raise LookupError("a shape-time table is admitted only under an explicit serving "
                          "target: pass --tessera-platform, --tessera-runtime-image, "
                          "--tessera-execution-mode and --tessera-residency")
    pin = pins.load_tessera_serving_runtime_pin()
    pins.require_pinned_tessera_runtime(pin)
    eligibility, published = tessera_render._pinned_serving_table()
    scope = ShapeTableScope(
        contract_sha256=pin.contract_sha256, tessera_commit=pin.serving_commit,
        runtime_image_digest=serving_target.runtime_image,
        tensor_parallel=int(tensor_parallel), platform=serving_target.platform,
        residency=serving_target.residency, execution_mode=serving_target.execution_mode)
    return scope, eligibility, published


class AllocationMenu:
    """The allocator's menu after this lane expanded its token.

    ``formats`` is the menu the registry resolves next; ``widths`` the width
    provenance the allocation stamps (empty when no rung of this lane is on
    the menu); ``refusal_cause`` the sentence a fatal menu refusal carries.
    """

    __slots__ = ("formats", "widths", "refusal_cause")

    def __init__(self, formats, widths, refusal_cause):
        self.formats = formats
        self.widths = widths
        self.refusal_cause = refusal_cause


def allocation_menu(fmt_names, priced_formats, *, context_by_unit) -> AllocationMenu:
    """Expand the ``TESSERA`` menu token to the rungs this run priced.

    ``TESSERA`` is a menu TOKEN, not a format. It cannot expand to a fixed
    list the way ``NVFP4`` names one rung: a Tessera family addresses a
    continuous rate axis, the realisable set depends on the unit's column
    count, and one 0.6B Linear carries thousands of legal rungs across the
    four families. So the token expands to exactly the rungs THIS RUN PRICED
    -- the cost table's own Tessera columns -- which is both the widest menu
    the DP could honestly consider and a set that needs no second copy of the
    campaign's legality decisions. A rung the campaign did not price would be
    dropped by ``build_candidates`` anyway (no cost row); naming it here would
    only have ``require_producer_formats`` refuse the whole run. The expansion
    is intersected with what the pinned runtime attests, so a research-priced
    table read back on the default path allocates over the backed axis
    instead of refusing wholesale. The narrowing is printed, not inferred: an
    allocation over 2 rungs and one over 3060 must not look the same in a log
    (P9, P12).
    """
    from .tessera_menu import (
        expand_menu_tokens_report,
        menu_mode,
        menu_width_report,
        partition_attested,
        tessera_refusal_cause,
        unattested_diagnosis,
    )

    tessera_context_by_unit = context_by_unit
    priced_tessera = [
        n for n in priced_formats
        if isinstance(n, str) and n.startswith("TESSERA_")
    ]
    fmt_names, unattested = expand_menu_tokens_report(
        fmt_names, priced_formats,
        **({"context_by_unit": tessera_context_by_unit}
           if tessera_context_by_unit is not None else {}))
    tessera_menu_widths: dict = {}
    tessera_diagnosis: dict | None = None
    if priced_tessera:
        kept = [n for n in fmt_names if n.startswith("TESSERA_")]
        # The count is the admission predicate's, not the caller's: an
        # explicitly named rung stays on the menu so the eligibility gate can
        # refuse it out loud, and it is not "attested" until then (#278).
        admitted, explicit_unattested = partition_attested(
            kept, **({"context_by_unit": tessera_context_by_unit}
                     if tessera_context_by_unit is not None else {}))
        # A refusal that cannot name its own cause costs an investigation
        # (#572: "0 of 16" was read three ways at once). Both causes the
        # contract can tell apart are structured -- whether it needs a serving
        # scope, and which rungs it attests instead -- so the report reads
        # them rather than leaving the reader to guess. It admits nothing:
        # `admitted` above is already the predicate's answer.
        tessera_diagnosis = (unattested_diagnosis(
            list(unattested) + list(explicit_unattested), priced=priced_tessera,
            context_by_unit=tessera_context_by_unit)
            if (unattested or explicit_unattested) else None)
        widths, line = menu_width_report(
            priced_tessera, admitted, unattested, explicit_unattested, menu_mode(),
            diagnosis=tessera_diagnosis)
        if kept:
            tessera_menu_widths = widths
        print(line, flush=True)
        # The measured status of the ranking this DP is about to do. Printed
        # here rather than at the end because it governs how the whole run's
        # output is to be read, and stamped into provenance because a
        # terminal line is not a property of the artifact (P12).
        if kept:
            print(
                "[alloc] WARNING: Tessera rungs are on this menu and the DP "
                "ranks them on a surrogate MEASURED to mis-rank them at "
                "matched bytes -- served KL 2.00x worse than a byte-matched "
                "uniform arm at 4.0 bpp (2.33x at 3.0, 2.88x at 5.0), and "
                "1.93x on the priced units alone. See "
                "tessera_menu.surrogate_selection_caveat() and "
                "docs/measurements/tessera-allocated-served-2026-09-02.md. "
                "This assignment is a CANDIDATE, not a selection: promote it "
                "only through SELECTION_MODE=validated-surrogate with a "
                "byte-matched uniform arm served beside it.",
                flush=True,
            )
        if unattested and not kept:
            raise SystemExit(
                "[alloc] ERROR: the cost table prices "
                f"{len(priced_tessera)} Tessera rungs and the pinned runtime "
                "attests none of them, so the TESSERA menu token expands to "
                "nothing. Either widen the runtime's attested rungs "
                "(attested_rungs_q256 in the packaged runtime_contract.json) "
                "or price a table under the attested menu; "
                "PRISMAQUANT_TESSERA_MENU=readable allocates over the rungs "
                "the pinned decoder accepts and "
                "PRISMAQUANT_TESSERA_MENU=research over the whole realisable "
                "axis -- both for research runs that do not export."
                # "attests none of them" is only true if the contract was asked
                # under a scope it can answer; say which case this is.
                + tessera_refusal_cause(tessera_diagnosis)
            )
    return AllocationMenu(fmt_names, tessera_menu_widths,
                          tessera_refusal_cause(tessera_diagnosis))


def _menu_block(menu_report, menu_report_agg, menu_widths) -> dict:
    from .tessera_menu import surrogate_selection_caveat

    return {
        "per_linear": dict(menu_report),
        "aggregated": dict(menu_report_agg),
        **menu_widths,
        **({"selection_caveat": surrogate_selection_caveat()}
           if menu_widths else {}),
    }


def allocation_layer_config_meta(*, menu_report, menu_report_agg, menu_widths,
                                 group_menu_report, hessian_identity, dev_pin,
                                 assignment, cost_data) -> dict:
    """The lane's blocks in ``layer_config.json``'s metadata, in stamp order.

    Continuous-menu provenance, written on EVERY run rather than only on the
    byte-budget path: how wide the Tessera menu was before the DP saw it,
    which of the two exact reductions shrank it (per-Linear and, for
    aggregated super items, again after aggregation). Absent keys mean no
    Tessera rung was on the menu, so a stock run's metadata is unchanged.
    """
    from .tessera_menu import priced_static_scales, project_hessian_identity

    return {
        **({"tessera_menu": _menu_block(menu_report, menu_report_agg, menu_widths)}
           if (menu_report or menu_report_agg or menu_widths) else {}),
        **({"tessera_group_knapsack": dict(group_menu_report)}
           if group_menu_report else {}),
        # Per selection: a content-equal second seal names only the selected
        # units priced under it (``unit_capture_sha256``), never the whole
        # table's row map (RobTand/prismaquant#1270).
        **({"tessera_hessian": project_hessian_identity(
                hessian_identity, assignment)}
           if (hessian_identity.get("stamped_rows")
               or hessian_identity.get("unstamped_rows")) else {}),
        # The static A-side scale VALUE each selected Tessera unit was priced
        # under, read from its own cost row (RobTand/prismaquant#204). The
        # export gate compares the exporter's --input-scales file against
        # this, value for value; until it existed the gate could only check
        # that a key was present. Absent when no Tessera unit is selected, so
        # a stock run's metadata is unchanged. Read from the unfiltered table
        # (`cost_data["costs"]`): every selected unit's row is there whatever
        # the lm_head / visual filters removed from the DP's view.
        **({"tessera_activation_static_scales": priced_static_scales(
                {name: fmt for name, fmt in assignment.items()
                 if str(fmt).startswith("TESSERA_")},
                cost_data["costs"],
                # The FORMULA those values came out of, read from the table
                # that priced them and never from this process's environment:
                # a legacy value under a full-E4M3 label is a scale nothing
                # served (RobTand/prismaquant#624).
                policy=(cost_data.get("provenance", {})
                        .get("activation_static_scales", {})
                        .get("policy")),
                served_activation_policy=cost_data.get("provenance", {}).get("served_activation_policy"))}
           if any(str(fmt).startswith("TESSERA_")
                  for fmt in assignment.values()) else {}),
        **({"tessera_dev_pin": dict(dev_pin)} if dev_pin else {}),
    }


def allocation_selection_meta(*, menu_report, menu_report_agg, menu_widths,
                              group_menu_report, dev_pin) -> dict:
    """The lane's blocks in ``selection.json``, in stamp order.

    How big the per-unit menu was BEFORE the DP saw it, and which of the two
    reductions shrank it (see ``reduce_continuous_menu``). Empty on a run with
    no Tessera rung on the menu. Without this a coarse-looking set of selected
    rates cannot be attributed: a campaign that priced few rungs and a bin
    width that swallowed many look identical in the output.
    """
    return {
        "tessera_group_knapsack": dict(group_menu_report),
        **({"tessera_dev_pin": dict(dev_pin)} if dev_pin else {}),
        "tessera_menu": _menu_block(menu_report, menu_report_agg, menu_widths),
    }


def allocation_selection_request_path(args):
    """Where ``--tessera-materialization-plan`` asks for a selection request."""
    return getattr(args, "tessera_materialization_plan", None)


def write_allocation_selection_request(path, *, layer_config, assignment,
                                       cost_path, cost_payload, output_path) -> None:
    """Write the non-exportable selected-wire request instead of a layer config."""
    from .tessera_materialization import write_selection_request

    write_selection_request(path,
        layer_config=layer_config, assignment=assignment,
        cost_path=cost_path, cost_payload=cost_payload, output_path=output_path)
    print(f"[alloc] non-exportable selected-wire request → {path}")


def allocation_expert_projection(cost_data, assignment) -> dict:
    """The expert-population block an allocation carries (PrismaQuant #183).

    The campaign's population statement, the producer's projection the units
    were priced under, and the receipt of each projected unit's selected rung.
    A stock cost table adds no keys. A refusal ends the run.
    """
    from .tessera_expert_projection import (
        ExpertProjectionError, allocation_expert_projection_block,
    )

    try:
        return allocation_expert_projection_block(cost_data, assignment)
    except ExpertProjectionError as exc:
        raise SystemExit(f"[alloc] ERROR: expert projection: {exc}") from exc


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


# -- the ship record (shipcard, shipcard_cli) --------------------------------

def shipcard_slot_verifiers() -> dict:
    """``{slot: replay}`` for the evidence slots this lane's gates declare.

    ``shipcard.verify`` runs each replay as ``(slot, record, *, card,
    model_dir)``, and the replay owns the slot's whole verdict.
    """
    from . import tessera_shipcard as receipts

    return {
        receipts.ROUTE_CENSUS_SLOT: receipts.verify_route_census_record,
        receipts.ROUTE_TRACE_SLOT: receipts.verify_route_trace_record,
    }


def shipcard_cli_commands(subparsers) -> None:
    """Add ``fill-route-census`` and ``fill-route-trace`` to the shipcard CLI."""
    from .tessera_shipcard import register_cli

    register_cli(subparsers)
