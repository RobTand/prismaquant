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
    refine: Callable[[int], Sequence[int]] | None = None,
) -> dict:
    """Publish a bounded next-step roster while retaining every legal rate.

    Endpoints establish interpolation support; recipe boundary witnesses preserve changes in the wire table. These are acquisition priorities, not numerical prices or a
    claim that endpoint interpolation passed an accuracy screen. ``refine``
    supplies decision-focused interior requests after those dependencies. It
    may return fewer points than its cap; no work is manufactured to fill it.
    The full remaining domain is always retained, including off-hull points.
    """
    if not isinstance(domain, RateDomain):
        raise ValueError('domain must be the existing RateDomain contract')
    if type(max_new_points) is not int or max_new_points < 0:
        raise ValueError('max_new_points must be a nonnegative integer')
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
    chosen = missing_boundaries[:max_new_points]
    reasons = {str(q): ('missing_domain_endpoint' if q in endpoints else 'missing_recipe_boundary')
               for q in chosen}
    if not missing_boundaries and refine is not None and max_new_points:
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
        'proposed_q256': chosen,
        'proposal_reasons': reasons,
        'max_new_points': max_new_points,
        'next_dependency': dependency,
        'full_domain_measured': not remaining,
        'acquisition_complete': not remaining,
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
) -> dict:
    """Join the grammar-derived domain to the existing adaptive allocator.

    One exact-shape quality member per call. The caller owns atomic serving
    groups and must acquire required sibling measurements as one PB task.
    No routed sample becomes a full expert stack by passing this adapter.
    """
    from .tessera_allocator import adaptive_trellis_rate_surface
    from .tessera_formats import get_tessera_family
    from .tessera_legal_domain import legal_rate_domain, table_width_transitions

    records = tuple(records)
    if not records or len({r.unit_name for r in records}) != 1:
        raise ValueError('exactly one nonempty quality unit is required')
    if len({r.shape for r in records}) != 1:
        raise ValueError('quality unit records mix shapes')
    spec = get_tessera_family(family)
    if any(r.family != spec.family for r in records):
        raise ValueError('quality unit records mix families')
    complete = legal_rate_domain(family, (records[0].shape,))
    # Record both sides of table-width changes. This proposes exact new
    # measurements; it does not fit a smooth curve across schedule or recipe
    # changes. All schedule transitions remain available to the refiner.
    boundary_rates = set()
    legal = set(complete.rates)
    for start, _width in table_width_transitions(family):
        if start == complete.rates[0]:
            continue
        boundary_rates.update(q for q in (start - 1, start) if q in legal)
    domain = RateDomain(complete.family, complete.rates, tuple(sorted(boundary_rates)))

    def refine(limit):
        proposal = adaptive_trellis_rate_surface(
            family, records, alpha_loss_per_byte=alpha_loss_per_byte,
            max_new_points=limit,
        )
        return proposal.surface.proposed_q256

    result = propose_full_domain_acquisition(
        domain, tuple(r.body_rate_q256 for r in records),
        max_new_points=max_new_points, refine=refine,
    )
    result.update({
        'unit_name': records[0].unit_name,
        'shape': list(records[0].shape),
        'alpha_loss_per_byte': alpha_loss_per_byte,
        'measured_record_identities': sorted(r.identity_sha256 for r in records),
        'decision_refiner': 'tessera_allocator.adaptive_trellis_rate_surface',
        'resolver_transition_q256': list(complete.transition_rates),
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
