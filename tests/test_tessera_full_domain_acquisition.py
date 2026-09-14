"""Sparse measurements cannot make the unobserved legal tails unreachable."""
from __future__ import annotations

import pytest

from prismaquant.quality_prefill_population import RateDomain
from prismaquant.tessera_full_domain_acquisition import propose_full_domain_acquisition


def test_singleton_requests_outer_endpoints_without_assigning_them_prices():
    domain = RateDomain('TESSERA_BF16_K1', tuple(range(256, 4097)), (3584, 3585, 3840, 3841))
    result = propose_full_domain_acquisition(domain, (1024,), max_new_points=2)
    assert result['proposed_q256'] == [256, 4096]
    assert result['legal_rate_count'] == 3841
    assert result['measured_envelope_q256'] == [1024, 1024]
    assert result['missing_legal_rate_count'] == 3840
    assert result['prices'] is None
    assert result['full_domain_measured'] is False
    assert result['all_legal_rates_retained'] is True
    assert result['proposal_reasons'] == {'256': 'missing_domain_endpoint', '4096': 'missing_domain_endpoint'}


def test_repeated_rounds_are_bounded_and_never_repeat_measured_work():
    domain = RateDomain('TESSERA_BF16_K1', tuple(range(256, 4097)), (3584, 3585, 3840, 3841))
    result = propose_full_domain_acquisition(domain, (256, 1024, 4096), max_new_points=2)
    assert result['proposed_q256'] == [3584, 3585]
    result2 = propose_full_domain_acquisition(domain, (256, 1024, 3584, 3585, 4096), max_new_points=2)
    assert result2['proposed_q256'] == [3840, 3841]


def test_decision_picker_runs_after_endpoint_coverage_and_cannot_escape_domain():
    domain = RateDomain('TESSERA_E4M3_K1', (256, 512, 768, 1024), ())
    calls = []
    def picker(limit):
        calls.append(limit)
        return (512,)
    result = propose_full_domain_acquisition(domain, (256, 1024), max_new_points=1, refine=picker)
    assert calls == [1]
    assert result['proposed_q256'] == [512]
    with pytest.raises(ValueError, match='outside'):
        propose_full_domain_acquisition(domain, (256, 1024), max_new_points=1, refine=lambda n: (600,))


def test_recipe_boundaries_do_not_require_every_interior_grid_measurement():
    domain = RateDomain('TESSERA_E4M3_K1', tuple(range(256, 2049)), ())
    result = propose_full_domain_acquisition(domain, (256, 832, 960, 1088, 2048), max_new_points=1,
                                           refine=lambda n: (1024,))
    assert result['proposed_q256'] == [1024]
    assert result['legal_rate_count'] == 1793
    assert result['missing_legal_rate_count'] == 1788
    assert result['adaptive_converged'] is None
    assert 'acquisition_complete' not in result


@pytest.mark.parametrize('measured', [(True,), (256, 256), (128,)])
def test_invalid_measured_domain_is_refused(measured):
    with pytest.raises(ValueError):
        propose_full_domain_acquisition(RateDomain('f', (256, 512), ()), measured, max_new_points=1)


def test_missing_refiner_is_explicit_and_zero_cap_does_not_claim_done():
    domain = RateDomain('f', (256, 512, 768), ())
    result = propose_full_domain_acquisition(domain, (256, 768), max_new_points=1)
    assert result['next_dependency'] == 'decision_refiner'
    result = propose_full_domain_acquisition(domain, (), max_new_points=0)
    assert result['proposed_q256'] == []
    assert result['adaptive_converged'] is None
    assert 'acquisition_complete' not in result
    assert result['next_dependency'] == 'measurement_budget'


def test_full_domain_bridge_calls_the_actual_adaptive_refiner():
    from prismaquant.tessera_allocator import build_tessera_allocator_candidate
    from prismaquant.tessera_full_domain_acquisition import adaptive_acquisition_from_records
    records = tuple(build_tessera_allocator_candidate(
        "u", (256, 256), family="TESSERA_E4M3_K1", body_rate_q256=q,
        layout="tight", schedule=None, alphabets=None,
        predicted_dloss=value, target_profile="research",
    ) for q, value in ((256, 8.0), (2048, 1.0)))
    result = adaptive_acquisition_from_records("TESSERA_E4M3_K1", records, max_new_points=1)
    assert result["legal_rate_count"] == 1793
    assert len(result["proposed_q256"]) == 1
    proposed = result["proposed_q256"][0]
    assert 256 < proposed < 2048
    assert result["proposal_reasons"][str(proposed)] == "decision_focused_interior"


@pytest.mark.parametrize('schema', ['tessera.encoding_inputs.v1', 'tessera.cached_unit_inputs.v1'])
def test_historical_price_cannot_be_rebound_to_another_producer_recipe(schema):
    from copy import deepcopy
    from prismaquant.tessera_formats import get_tessera_family, tessera_wire_recipe
    from prismaquant.tessera_full_domain_acquisition import require_measured_recipe_binding
    family = get_tessera_family('TESSERA_E4M3_K1')
    row = {'tessera_family': family.name, 'tessera_body_rate_q256': 832, 'wire_bytes': 1234}
    record = {'blob_bytes': 1234, 'blob_sha256': 'a' * 64, 'identity': {
        'schema': schema, 'unit': 'u', 'source': {'shape': [256, 256]},
        'encoder_source_sha256': 'b' * 64,
        'recipe': {'grid': family.base, 'q256': 832, **tessera_wire_recipe(family, 832).to_config()},
    }}
    assert len(require_measured_recipe_binding('u', (256, 256), row, record,
                                               encoder_source_sha256='b' * 64)) == 64
    for field, value in [('encoder_source_sha256', 'c' * 64), ('recipe', {}), ('unit', 'v')]:
        changed = deepcopy(record)
        changed['identity'][field] = value
        with pytest.raises(ValueError):
            require_measured_recipe_binding('u', (256, 256), row, changed,
                                             encoder_source_sha256='b' * 64)
    with pytest.raises(ValueError, match='missing'):
        require_measured_recipe_binding('u', (256, 256), row, {}, encoder_source_sha256='b' * 64)


def test_explicit_boundary_deferral_keeps_full_domain_and_refines_measured_support():
    domain = RateDomain('f', (256, 512, 768, 1024), ())
    result = propose_full_domain_acquisition(domain, (512, 1024), max_new_points=1,
        boundary_policy='defer', refine=lambda cap: (768,))
    assert result['proposed_q256'] == [768]
    assert result['deferred_boundary_q256'] == [256]
    assert result['legal_q256'] == [256, 512, 768, 1024]
    assert result['prices'] is None
    assert result['adaptive_converged'] is None


def test_deferring_expensive_bookends_cannot_claim_the_unknown_domain_converged():
    domain = RateDomain('f', (256, 512, 768), ())
    result = propose_full_domain_acquisition(domain, (512,), max_new_points=2,
        boundary_policy='defer', refine=lambda cap: ())
    assert result['proposed_q256'] == []
    assert result['deferred_boundary_q256'] == [256, 768]
    assert result['next_dependency'] == 'decision_bound_for_deferred_boundary'
    assert result['full_domain_measured'] is False
    with pytest.raises(ValueError, match='boundary_policy'):
        propose_full_domain_acquisition(domain, (512,), max_new_points=1, boundary_policy='guess')
