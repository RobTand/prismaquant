"""Bounded acquisition over the complete legal Tessera rate domain.

A missing price remains unknown. This adapter seeds unobserved domain/recipe
boundaries and then calls the existing multiplier-aware RD-hull refiner. It
never assigns an extrapolated price, changes a measured row, or makes complete
grid measurement a prerequisite for a proposal. Exact selected-rate scoring
and downstream validation remain separate from this acquisition schedule.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from .quality_prefill_population import RateDomain

SCHEMA = 'prismaquant.tessera_full_domain_acquisition.v1'


def propose_full_domain_acquisition(
    domain: RateDomain,
    measured_q256: Sequence[int],
    *,
    max_new_points: int,
    boundary_policy: str = "seed",
    refine: Callable[[int], Sequence[int]] | None = None,
) -> dict:
    """Publish a bounded next-step roster while retaining every legal rate.

    Endpoints establish interpolation support; recipe boundary witnesses preserve changes in the wire table. These are acquisition priorities, not numerical prices or a
    claim that endpoint interpolation passed an accuracy screen. ``refine``
    supplies decision-focused interior requests after those dependencies. It
    may return fewer points than its cap; no work is manufactured to fill it.
    The full remaining domain is always retained, including off-hull points.
    ``boundary_policy="defer"`` postpones unknown boundary measurements and
    allows the existing refiner to work from measured support. It asserts no
    dominance or price bound; the caller supplies this acquisition policy.
    """
    if not isinstance(domain, RateDomain):
        raise ValueError('domain must be the existing RateDomain contract')
    if type(max_new_points) is not int or max_new_points < 0:
        raise ValueError('max_new_points must be a nonnegative integer')
    if boundary_policy not in ('seed', 'defer'):
        raise ValueError('boundary_policy must be seed or defer')
    measured = tuple(measured_q256)
    if any(type(q) is not int for q in measured) or len(set(measured)) != len(measured):
        raise ValueError('measured rates must be distinct integers')
    legal = set(domain.rates)
    if not set(measured) <= legal:
        raise ValueError('measured rate outside legal domain')
    observed = set(measured)
    endpoints = (domain.rates[0], domain.rates[-1])
    boundary_queue = list(dict.fromkeys((*endpoints, *domain.transition_rates)))
    missing_boundaries = [q for q in boundary_queue if q not in observed]
    pending_boundaries = missing_boundaries if boundary_policy == 'seed' else []
    chosen = pending_boundaries[:max_new_points]
    reasons = {str(q): ('missing_domain_endpoint' if q in endpoints else 'missing_recipe_boundary')
               for q in chosen}
    if not pending_boundaries and refine is not None and max_new_points:
        refined = tuple(refine(max_new_points))
        if len(refined) > max_new_points or len(set(refined)) != len(refined):
            raise ValueError('decision refiner exceeded cap or repeated work')
        if any(type(q) is not int or q not in legal for q in refined):
            raise ValueError('decision refiner proposed a rate outside legal domain')
        if set(refined) & observed:
            raise ValueError('decision refiner repeated measured work')
        chosen.extend(refined)
        reasons.update({str(q): 'decision_focused_interior' for q in refined})
    remaining = len(legal - observed)
    if chosen:
        dependency = 'exact_measurements_for_proposed_rates'
    elif not remaining:
        dependency = None
    elif max_new_points == 0:
        dependency = 'measurement_budget'
    elif boundary_policy == 'defer' and missing_boundaries and refine is not None:
        dependency = 'decision_bound_for_deferred_boundary'
    elif refine is None:
        dependency = 'decision_refiner'
    else:
        dependency = 'selection_validation_or_new_refinement_policy'
    return {
        'schema': SCHEMA,
        'family': domain.family,
        'legal_q256': list(domain.rates),
        'legal_rate_count': len(domain.rates),
        'all_legal_rates_retained': True,
        'measured_q256': sorted(observed),
        'measured_envelope_q256': [min(observed), max(observed)] if observed else None,
        'missing_legal_rate_count': remaining,
        'required_boundary_q256': boundary_queue,
        'missing_boundary_q256': missing_boundaries,
        'boundary_policy': boundary_policy,
        'deferred_boundary_q256': missing_boundaries if boundary_policy == 'defer' else [],
        'proposed_q256': chosen,
        'proposal_reasons': reasons,
        'max_new_points': max_new_points,
        'next_dependency': dependency,
        'full_domain_measured': not remaining,
        'adaptive_converged': None,
        'full_grid_measurement_required_for_proposal': False,
        'prices': None,
        'interpolation_error_bound': None,
        'allocator_payload': False,
        'research_only': True,
        'production_qualified': False,
    }


def adaptive_acquisition_from_records(
    family: str,
    records: Sequence,
    *,
    max_new_points: int,
    alpha_loss_per_byte: float | None = None,
    boundary_policy: str = "seed",
) -> dict:
    """Join the grammar-derived domain to the existing adaptive allocator.

    One exact-shape quality member per call. The caller owns atomic serving
    groups and must acquire required sibling measurements as one PB task.
    No routed sample becomes a full expert stack by passing this adapter.
    """
    from .tessera_allocator import adaptive_trellis_rate_surface
    from .tessera_formats import get_tessera_family


    records = tuple(records)
    if not records or len({r.unit_name for r in records}) != 1:
        raise ValueError('exactly one nonempty quality unit is required')
    if len({r.shape for r in records}) != 1:
        raise ValueError('quality unit records mix shapes')
    spec = get_tessera_family(family)
    if any(r.family != spec.family for r in records):
        raise ValueError('quality unit records mix families')
    domain, refused, transitions = _shape_acquisition_domain(family, records[0].shape)
    if not {r.body_rate_q256 for r in records} <= set(domain.rates):
        raise ValueError("measured rate outside the producer-legal shape domain")

    def refine(limit):
        proposal = adaptive_trellis_rate_surface(
            family, records, alpha_loss_per_byte=alpha_loss_per_byte,
            max_new_points=limit,
        )
        observed = {r.body_rate_q256 for r in records}
        chosen = []
        for bracket in proposal.ranked_brackets:
            if not bracket["selected_for_refinement"]:
                continue
            lo, hi = bracket["q256_interval"]
            allowed = [q for q in domain.rates if lo < q < hi
                       and q not in observed and q not in chosen]
            if allowed:
                midpoint = bracket["proposed_q256"]
                chosen.append(min(allowed, key=lambda q: (abs(q - midpoint), q)))
        return tuple(chosen)

    result = propose_full_domain_acquisition(
        domain, tuple(r.body_rate_q256 for r in records),
        max_new_points=max_new_points, refine=refine, boundary_policy=boundary_policy,
    )
    result.update({
        'unit_name': records[0].unit_name,
        'shape': list(records[0].shape),
        'alpha_loss_per_byte': alpha_loss_per_byte,
        'measured_record_identities': sorted(r.identity_sha256 for r in records),
        'decision_refiner': 'tessera_allocator.adaptive_trellis_rate_surface',
        'resolver_transition_q256': list(transitions),
        'producer_refused_q256': refused,
    })
    return result


def require_measured_recipe_binding(unit, shape, row, record, *, encoder_source_sha256):
    """Refuse attaching an old scalar price to a newly resolved producer recipe.

    This verifies recorded metadata, not source tensors or wire bodies. Its
    digest binds exactly which producer receipt supplied the measured price.
    The caller additionally matches the cost row against its anchor journal.
    """
    from .cost_stage_checkpoint import canonical_json_sha256
    from .tessera_formats import get_tessera_family, tessera_wire_recipe

    identity = record.get('identity', {})
    family = get_tessera_family(row['tessera_family'])
    rate = row['tessera_body_rate_q256']
    expected = {'grid': family.base, 'q256': rate,
                **tessera_wire_recipe(family, rate).to_config()}
    if identity.get('schema') not in {'tessera.encoding_inputs.v1', 'tessera.cached_unit_inputs.v1'}:
        raise ValueError('missing measured producer input identity')
    if identity.get('unit') != unit or identity.get('source', {}).get('shape') != list(shape):
        raise ValueError('measured producer source unit/shape differs')
    if identity.get('encoder_source_sha256') != encoder_source_sha256:
        raise ValueError('measured encoder source differs from active producer')
    if identity.get('recipe') != expected:
        raise ValueError('measured wire recipe differs from active producer recipe')
    if record.get('blob_bytes') != row.get('wire_bytes'):
        raise ValueError('measured wire byte count differs from cost row')
    digest = record.get('blob_sha256')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('measured wire record lacks a valid blob digest')
    return canonical_json_sha256(record, where='measured acquisition wire record')


def _shape_acquisition_domain(family, shape):
    from bisect import bisect_left
    from .tessera_legal_domain import legal_rates, resolver_transitions, table_width_transitions

    rates, refusals = legal_rates(family, (shape,))
    if not rates:
        raise ValueError("acquisition has no producer-legal rates at the source shape")
    witnesses = set()
    for start, _width in table_width_transitions(family):
        index = bisect_left(rates, start)
        if 0 < index < len(rates):
            witnesses.update((rates[index - 1], rates[index]))
    return (RateDomain(family, rates, tuple(sorted(witnesses))),
            {str(rate): list(reasons) for rate, reasons in sorted(refusals.items())},
            resolver_transitions(family, rates, (shape,)))


def joint_acquisition_from_cost_data(cost_data, unit_shapes, families, *,
                                     max_new_points, alpha_loss_per_byte=None,
                                     boundary_policy="seed"):
    """Research requests from one attested joint run, retaining its raw evidence.

    Candidate views only rank acquisition brackets. Writer-derived bytes are
    estimates, not measured wires or new prices. Signed W/A/mixed samples and
    run/probe/operator identities survive unchanged; no scalar MSE conversion.
    """
    from .cost_currency import require_run_currency, CostCurrencyError
    from .cost_stage_checkpoint import canonical_json_sha256
    from .joint_aura import JOINT_CURRENCY, identity_sha256
    from .tessera_allocator import build_tessera_allocator_candidate
    from .tessera_formats import parse_tessera_format_name, get_tessera_family

    currency = require_run_currency(cost_data)
    if currency.get("cost_currency") != JOINT_CURRENCY or not currency.get("joint_aura_rows"):
        raise CostCurrencyError("joint acquisition requires attested homogeneous joint AURA rows")
    if not unit_shapes or not families or len(set(families)) != len(families):
        raise ValueError("joint acquisition requires distinct families and a nonempty unit shape roster")
    for family in families:
        if get_tessera_family(family).name != family:
            raise ValueError("joint acquisition requires canonical family names")
    provenance = cost_data["provenance"]
    run = provenance.get("joint_aura_identity")
    if (not isinstance(run, dict) or run.get("schema") != "prismaquant.joint_aura.run.v2"
            or identity_sha256(run) != provenance.get("joint_aura_identity_sha256")):
        raise ValueError("joint acquisition requires the bound v2 joint run identity")
    if run.get("probe_identity") != provenance.get("probe_identity"):
        raise ValueError("joint acquisition run/provenance probe identity differs")
    for field in ("cached_rendered_weights", "activation_contracts"):
        if not isinstance(run.get(field), dict):
            raise ValueError(f"joint acquisition requires bound run {field}")
    probe_digest = identity_sha256(run["probe_identity"])
    reports = []
    for unit, shape in sorted(unit_shapes.items()):
        shape = tuple(shape)
        if len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError("joint acquisition requires positive integer Linear shapes")
        if unit not in cost_data["costs"]:
            raise ValueError(f"joint acquisition has no source-bound cost unit: {unit}")
        records_by_family = {family: [] for family in families}
        raw_by_family = {family: {} for family in families}
        source_identity = None
        if not isinstance(cost_data["costs"][unit], dict):
            raise ValueError(f"joint acquisition cost unit is not a row mapping: {unit}")
        for fmt, row in sorted(cost_data["costs"][unit].items()):
            if not isinstance(row, dict):
                raise ValueError(f"joint acquisition requires raw row mappings: {unit}/{fmt}")
            if "error" in row:
                continue
            operator = row["joint_operator_identity"]
            if row["probe_identity_sha256"] != probe_digest:
                raise ValueError("joint acquisition rows do not share the exact run probe identity")
            source = operator["source_weight"]
            if source["shape"] != list(shape):
                raise ValueError(f"joint acquisition source shape differs for {unit}")
            if source_identity is not None and source_identity != source:
                raise ValueError(f"joint acquisition mixes source weights for {unit}")
            source_identity = source
            if not fmt.startswith("TESSERA_"):
                continue
            family, rate = parse_tessera_format_name(fmt)
            if (row.get("tessera_family", family.name) != family.name
                    or row.get("tessera_body_rate_q256", rate) != rate):
                raise ValueError("joint acquisition format/family/rate metadata differs")
            if (run.get("cached_rendered_weights", {}).get(unit, {}).get(fmt)
                    != operator["rendered_weight"]):
                raise ValueError("joint acquisition rendered weight differs from the bound run")
            if (run.get("activation_contracts", {}).get(unit, {}).get(fmt)
                    != operator["activation"]):
                raise ValueError("joint acquisition activation differs from the bound run")
            if family.name not in records_by_family:
                continue
            records_by_family[family.name].append(build_tessera_allocator_candidate(
                unit, shape, family=family, body_rate_q256=rate, layout="tight",
                schedule=None, alphabets=None, predicted_dloss=row["predicted_dloss"],
                predicted_dloss_stderr=row["predicted_dloss_stderr"], target_profile="research"))
            raw_by_family[family.name][fmt] = row
        if source_identity is None:
            raise ValueError(f"joint acquisition has no measured source identity for {unit}")
        for family in families:
            records = records_by_family[family]
            if records:
                report = adaptive_acquisition_from_records(
                    family, records, max_new_points=max_new_points,
                    alpha_loss_per_byte=alpha_loss_per_byte, boundary_policy=boundary_policy)
            else:
                domain, refused, transitions = _shape_acquisition_domain(family, shape)
                report = propose_full_domain_acquisition(domain, (),
                    max_new_points=max_new_points, boundary_policy=boundary_policy)
                report.update(unit_name=unit, shape=list(shape),
                    producer_refused_q256=refused, resolver_transition_q256=list(transitions))
            report.update(currency=JOINT_CURRENCY, joint_measurement_records=raw_by_family[family],
                joint_source_weight=source_identity, probe_identity_sha256=probe_digest,
                joint_aura_identity_sha256=provenance["joint_aura_identity_sha256"],
                uncertainty_scope="probe_sampling_conditional_on_fixed_calibration",
                acquisition_byte_basis="current_public_writer_estimate_not_measured_wire",
                interpolation_qualified=False, selected_assignment_confirmed=False)
            reports.append(report)
    return {"reports": reports, "cost_currency": currency,
            "joint_provenance_sha256": canonical_json_sha256(provenance, where="joint acquisition provenance"),
            "measurement_verification": "validated attested joint rows and bound raw run/operator/probe metadata; no tensor or wire payload reread"}


def load_joint_campaign_acquisition(binding: dict, *, units: Sequence[str] | None = None) -> dict:
    """Authenticate research requests before the existing renderer consumes them.

    The root SHA binds the small request document, whose cost SHA binds the
    original pickle. Ordinary currency and raw-v2 validation are rerun against
    that pickle, not the document's claims. This checks recorded identities
    only, without rereading source tensors or rendered wire bodies. Explicit
    units are projected only after validating the whole original request.
    Atomic member coverage and active scope belong to the scheduler.
    """
    import pickle

    from .cost_stage_checkpoint import canonical_json_sha256
    from .tessera_acquisition_inputs import (
        read_joint_campaign_acquisition_document, joint_campaign_acquisition_control_sha256)
    from .joint_aura import identity_sha256
    from .stage_inputs import read_bound, require
    from .tessera_formats import get_tessera_family
    from .tessera_legal_domain import live_pins as current_domain_pins, tessera_source_state
    from .dev_mode import seal_check

    def recorded_same(actual, expected, label):
        seal_check(f"joint acquisition {label}", actual, expected, where=binding["path"],
            same=(canonical_json_sha256(actual, where=label) ==
                  canonical_json_sha256(expected, where=label)),
            refusal=lambda: ValueError(
                f"joint acquisition {label} differs from actual cost/domain evidence"))

    def same(actual, expected, label):
        # Digests distinguish bools from integers and preserve raw signed samples.
        require(canonical_json_sha256(actual, where=label) ==
                canonical_json_sha256(expected, where=label),
                f"joint acquisition {label} differs from actual cost/domain evidence")

    def request_only(record):
        for field, expected in (("prices", None), ("interpolation_error_bound", None),
                                ("allocator_payload", False), ("production_qualified", False),
                                ("interpolation_qualified", False),
                                ("selected_assignment_confirmed", False)):
            if field in record:
                require(record[field] is expected, f"joint acquisition refuses claimed {field}")

    document, _ = read_joint_campaign_acquisition_document(binding, reader=read_bound)
    request_only(document)
    require(document.get("allocator_payload") is False and
            document.get("production_qualified") is False and
            document.get("atomic_serving_group_expansion_required") is True,
            "joint acquisition requires request-only atomic expansion")
    require(document.get("journal_bindings") == {} and
            document.get("active_encoder_source_sha256") is None,
            "joint acquisition refuses scalar anchor/encoder bindings")
    recorded_same(document.get("domain_pins"), current_domain_pins().as_dict(), "domain_pins")
    state = tessera_source_state()
    claimed_state = document.get("producer_source_state")
    require(isinstance(claimed_state, dict), "joint acquisition requires producer source state")
    same(claimed_state.get("schema"), state["schema"], "producer_source_state.schema")
    for field in ("export_sha256", "grammar_sha256"):
        recorded_same(claimed_state.get(field), state[field], f"producer_source_state.{field}")

    reports = document.get("reports")
    require(isinstance(reports, list) and bool(reports), "joint acquisition requires reports")
    shapes, families, pairs = {}, set(), set()
    for report in reports:
        require(isinstance(report, dict), "joint acquisition report must be a mapping")
        unit, family, shape = report.get("unit_name"), report.get("family"), report.get("shape")
        require(isinstance(unit, str) and bool(unit), "joint acquisition requires a unit name")
        require(isinstance(family, str) and get_tessera_family(family).name == family,
                "joint acquisition requires canonical families")
        require(isinstance(shape, list) and len(shape) == 2 and
                all(type(d) is int and d > 0 for d in shape),
                "joint acquisition requires positive integer Linear shapes")
        require((unit, family) not in pairs, "joint acquisition duplicate unit/family report")
        if unit in shapes:
            same(shape, shapes[unit], "shared source shape")
        shapes[unit] = shape
        families.add(family)
        pairs.add((unit, family))

    cost_binding = {"path": document.get("cost_path"), "sha256": document.get("cost_sha256")}
    require(isinstance(cost_binding["path"], str) and bool(cost_binding["path"]),
            "joint acquisition requires bound cost_path")
    cost_data = pickle.loads(read_bound(cost_binding, "joint acquisition cost"))
    actual = joint_acquisition_from_cost_data(cost_data, shapes, sorted(families), max_new_points=0)
    for field in ("cost_currency", "joint_provenance_sha256", "measurement_verification"):
        same(document.get(field), actual[field], field)
    expected_reports = {(r["unit_name"], r["family"]): r for r in actual["reports"]}
    provenance = cost_data["provenance"]
    probe_digest = identity_sha256(provenance["joint_aura_identity"]["probe_identity"])
    same(provenance.get("probe_identity_sha256"), probe_digest, "run probe digest")
    # Currency permits a development seal override; this intake does not.
    for rows in cost_data["costs"].values():
        require(isinstance(rows, dict), "joint acquisition cost unit must be a row mapping")
        for row in rows.values():
            require(isinstance(row, dict), "joint acquisition requires raw row mappings")
            if "error" not in row:
                same(row.get("probe_identity_sha256"), probe_digest, "row/run probe digest")

    proposal_fields = {"proposed_q256", "proposal_reasons", "max_new_points", "next_dependency",
                       "boundary_policy", "deferred_boundary_q256", "alpha_loss_per_byte"}
    requests, source_weights, total = {}, {}, 0
    for report in reports:
        unit, family = report["unit_name"], report["family"]
        expected = expected_reports[unit, family]
        request_only(report)
        for field, value in expected.items():
            if field not in proposal_fields:
                require(field in report, f"joint acquisition report lacks {field}")
                same(report[field], value, f"{unit}/{family}.{field}")
        policy = report.get("boundary_policy")
        require(policy in ("seed", "defer"), "joint acquisition boundary policy must be seed or defer")
        same(report.get("deferred_boundary_q256"),
             expected["missing_boundary_q256"] if policy == "defer" else [],
             f"{unit}/{family}.deferred_boundary_q256")
        proposed, cap = report.get("proposed_q256"), report.get("max_new_points")
        require(type(cap) is int and cap >= 0, "joint acquisition cap must be a nonnegative integer")
        require(isinstance(proposed, list) and all(type(q) is int for q in proposed),
                "joint acquisition proposed rates must be integers")
        require(len(set(proposed)) == len(proposed), "joint acquisition duplicate proposed rate")
        require(len(proposed) <= cap, "joint acquisition proposed rates exceed cap")
        require(set(proposed) <= set(expected["legal_q256"]),
                "joint acquisition proposed rate outside producer-legal domain")
        require(not set(proposed).intersection(expected["measured_q256"]),
                "joint acquisition proposed rate is already measured")
        reasons = report.get("proposal_reasons")
        require(isinstance(reasons, dict) and set(reasons) == {str(q) for q in proposed},
                "joint acquisition proposal reasons differ from requested rates")
        requests.setdefault(unit, {})[family] = list(proposed)
        source_weights[unit] = dict(expected["joint_source_weight"])
        total += len(proposed)
    require(total > 0, "joint acquisition has no requested measurement work")
    same(document.get("total_requested_quality_measurements"), total, "total requested measurements")
    acquisition = {"requests": requests, "source_weights": source_weights,
                   "identity": {"request_sha256": binding["sha256"],
                                "request_control_sha256": joint_campaign_acquisition_control_sha256(document),
                                "cost_sha256": cost_binding["sha256"],
                                "joint_aura_identity_sha256": provenance["joint_aura_identity_sha256"],
                                "probe_identity_sha256": probe_digest}}
    return acquisition if units is None else project_joint_campaign_acquisition(acquisition, units=units)



def project_joint_campaign_acquisition(acquisition: dict, *, units: Sequence[str]) -> dict:
    """Project a fully validated intake onto explicit existing campaign units.

    Callers must first obtain 'acquisition' from the authenticated loader.
    This is whole-unit selection, not a new request or rate controller. It
    deliberately knows no atomic groups: the runtime still requires every
    actual member, including members whose requested families are deferred.
    The coordinator must omit rows with no selected measurement work.
    """
    from copy import deepcopy
    from .stage_inputs import require

    require(isinstance(units, Sequence) and not isinstance(units, (str, bytes)),
            "joint acquisition selected units must be a sequence of unit names")
    require(bool(units) and all(isinstance(unit, str) and bool(unit) for unit in units),
            "joint acquisition selected units must be nonempty unit names")
    require(len(set(units)) == len(units), "joint acquisition duplicate selected unit")
    selected = sorted(units)
    require(set(selected) <= set(acquisition["requests"]),
            "joint acquisition selected unit is outside authenticated request scope")
    require(set(selected) <= set(acquisition["source_weights"]),
            "joint acquisition selected unit lacks authenticated source identity")
    requests = {unit: {family: list(rates) for family, rates in
                       sorted(acquisition["requests"][unit].items())} for unit in selected}
    require(any(rates for families in requests.values() for rates in families.values()),
            "joint acquisition selected units have no requested measurement work")
    return {"requests": requests,
            "source_weights": {unit: deepcopy(acquisition["source_weights"][unit]) for unit in selected},
            "identity": dict(acquisition["identity"])}
