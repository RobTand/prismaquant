"""Assignment representations and payload scopes stay explicit."""
import pytest

from prismaquant.assignment_space import AggregationPolicy, AssignmentSpace
from prismaquant.allocator_candidates import _FUSED_SIBLING_MARKER
from prismaquant.model_profiles import DefaultProfile


def space(*, stats=None, fixed_stats=None, accounting_stats=None,
          fixed_formats=None, packed=True, fused=True):
    return AssignmentSpace(
        stats=stats or {}, fixed_stats=fixed_stats or {},
        accounting_stats=accounting_stats or {}, fixed_formats=fixed_formats or {},
        aggregation=AggregationPolicy(packed=packed, fused=fused),
        profile=DefaultProfile(),
        format_rank={'NVFP4': 0, 'FP8_E4M3': 1, 'BF16': 2},
        legal_formats={},
    )


def test_per_name_lookup_skips_non_dicts_but_scope_overwrites_them():
    fixed = {'n_params': 8, 'weight_shape': (2, 4)}
    original = {'n_params': 12, 'weight_shape': (3, 4)}
    owner = space(stats={'q': None}, fixed_stats={'q': fixed, 'k': False},
                  accounting_stats={'q': original, 'k': original})
    assert owner.stats_for('q') is fixed
    assert owner.stats_for('k') is original
    assert owner.stats_for('missing') is None
    scope = owner.scope_stats()
    assert scope == {'q': None, 'k': False}
    scope['q'] = original
    assert owner.scope_stats()['q'] is None


def test_expansion_keeps_both_sides_of_real_serving_promotion():
    prefix = 'model.layers.0.self_attn'
    q, k, v = [prefix + '.' + leaf + '_proj' for leaf in ('q', 'k', 'v')]
    assignment = {q: 'NVFP4', k: 'FP8_E4M3', v: 'NVFP4'}
    owner = space(fixed_formats={'lm_head': 'BF16'})
    result = owner.expand(assignment)
    assert result.before_promotion == {**assignment, 'lm_head': 'BF16'}
    assert result.assignment == {q: 'FP8_E4M3', k: 'FP8_E4M3', v: 'FP8_E4M3',
                                 'lm_head': 'BF16'}
    result.before_promotion[q] = 'BF16'
    assert result.assignment[q] == 'FP8_E4M3'
    assert assignment[q] == 'NVFP4'


def test_in_place_promotion_cannot_blind_the_emitted_assignment_guard(monkeypatch):
    from prismaquant import assignment_space
    from prismaquant.allocator import validate_final_serving_promotion_noop

    def mutate_promotion(assignment, *args, **kwargs):
        assignment['q'] = 'BF16'
        return assignment

    monkeypatch.setattr(assignment_space, 'promote_serving_units', mutate_promotion)
    result = space().expand({'q': 'NVFP4'})
    with pytest.raises(SystemExit, match='final serving-unit promotion changed'):
        validate_final_serving_promotion_noop(result.before_promotion, result.assignment)


def test_unit_members_and_fixed_join_are_distinct_operations():
    names = ['model.layers.0.self_attn.' + leaf + '_proj' for leaf in ('q', 'k', 'v')]
    unit = 'model.layers.0.self_attn' + _FUSED_SIBLING_MARKER + 'qkv'
    owner = space(stats={unit: {'_fused_siblings': names}},
                  fixed_formats={'lm_head': 'BF16'})
    members = dict.fromkeys(names, 'FP8_E4M3')
    assert owner.unit_members(unit, 'FP8_E4M3') == members
    assert owner.expand({unit: 'FP8_E4M3'}, include_auxiliary=False).assignment == members
    assert owner.expand({unit: 'FP8_E4M3'}).assignment == {**members, 'lm_head': 'BF16'}


def test_exact_payload_uses_the_requested_parameter_scope():
    owner = space(stats={'body': {'n_params': 12, 'weight_shape': (3, 4)}},
                  fixed_stats={'lm_head': {'n_params': 8, 'weight_shape': (2, 4)}})
    body = owner.payload({'body': 'BF16'}, require_all_stats=True)
    complete = owner.payload({'body': 'BF16', 'lm_head': 'BF16'}, require_all_stats=True)
    assert body == {'bits_total': 192.0, 'quantizable_params': 12,
                    'bits_per_param': 16.0, 'missing_stats_names': []}
    assert complete['bits_total'] == 320.0
    assert complete['quantizable_params'] == 20
    assert owner.bits_total({'body': 'BF16'}) == body['bits_total']


def test_missing_payload_stats_refuse_or_remain_explicit():
    owner = space()
    with pytest.raises(AssertionError, match='no shape/stats'):
        owner.payload({'unknown': 'BF16'}, require_all_stats=True)
    assert owner.payload({'unknown': 'BF16'}, require_all_stats=False) == {
        'bits_total': 0.0, 'quantizable_params': 0,
        'bits_per_param': 0.0, 'missing_stats_names': ['unknown'],
    }
