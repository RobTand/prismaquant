"""A pilot zero is observable only inside a complete atomic serving stack."""
import copy
from types import SimpleNamespace

import pytest
import torch

from prismaquant import tessera_expert_projection as tep
from prismaquant.joint_aura import identity_sha256, make_joint_aura_entry
from prismaquant.production_weight_cache import _cb_cache_tensor_identity
from prismaquant.tessera_joint_eval_panel import make_panel
from prismaquant.tessera_sampled_stack_proposal import propose_bound_payload
from test_tessera_expert_projection import STACK, _declared, _projection
from test_tessera_joint_allocation import fixture

PRIMARY = ['TESSERA_E4M3_K1', 'TESSERA_BF16_K1']


class AtomicProfile:
    def packed_expert_format_group(self, name):
        return STACK if name.startswith(STACK + '.') else None

    def fused_sibling_group(self, name):
        return None


def pilot_fixture():
    from test_joint_aura_assignment_diagnostics import _rebuild
    names = list(_declared(experts=(0, 1), n=256, k=256)[STACK])
    joint, data, prepared, metadata, kwargs = fixture(names)
    projection = _projection(experts=(0, 1), n=256, k=256)
    carried = tep.carried_projection(projection, tep.bind_expert_projection(
        projection, declared=_declared(experts=(0, 1), n=256, k=256)),
        request=tep.stack_plan_request({STACK: ('E4M3', 1024)}), tool='/fixture/producer')
    joint['provenance'][tep.PROJECTION_KEY] = carried
    payload_names = set(names)
    assert payload_names == set(tep.carried_units(carried)[1])
    # One selected window of the original two, with its OWN calibration hash.
    ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int64)
    panel = make_panel(ids, artifact_sha256='a'*64, seed=1, size=1)
    joint['provenance']['joint_eval'] = panel
    joint['provenance']['tessera_joint_anchors']['joint_eval'] = panel
    prepared['calibration_input']['artifact_sha256'] = 'a'*64
    original = torch.arange(256 * 256, dtype=torch.bfloat16).reshape(256, 256)
    source = _cb_cache_tensor_identity(original)
    observed_names = set(names[:3])
    for name in names:
        count = {'tokens': 4 if name in observed_names else 0,
                 'calls': 1 if name in observed_names else 0,
                 'count_scope': 'summed_over_probes', 'n_probes': 4,
                 'per_probe': [{'tokens': 1 if name in observed_names else 0,
                                'calls': 1 if name in observed_names else 0} for _ in range(4)]}
        status = 'observed' if name in observed_names else 'unknown_unobserved'
        # Diagnostic counters sum the probes; four observed calls, four tokens.
        count['calls'] = 4 if name in observed_names else 0
        joint['stats'][name]['joint_eval_status'] = status
        joint['stats'][name]['joint_eval_observations'] = count
        joint['stats'][name].update(n_params=256*256, in_features=256, out_features=256)
        for fmt, row in list(joint['costs'][name].items()):
            def change_probe(probe):
                probe.update(calibration_sha256=panel['eval_ids_sha256'], calibration_shape=[1, 4])
            def change_operator(op):
                op['source_weight'] = copy.deepcopy(source)
                op['rendered_weight'] = copy.deepcopy(source)
            value = _rebuild(row, probe_change=change_probe, operator_change=change_operator)
            if status == 'unknown_unobserved':
                operator = value['joint_operator_identity']
                value = make_joint_aura_entry(operator_identity=operator,
                    probe_identity=value['probe_identity'],
                    signed_components=[{'weight': 0., 'activation': 0.,
                                        'mixed': 0., 'total': 0.} for _ in range(4)])
            value.update(joint_eval_status=status, joint_eval_observations=copy.deepcopy(count))
            joint['costs'][name][fmt] = value
    joint['provenance']['tessera_joint_allocation'] = {
        'status': 'research_sampled_joint_panel', 'export_authority': False,
        'plan_sha256': kwargs['plan_sha256'], 'prepared': kwargs['prepared_binding']}
    return joint, data, prepared, metadata, kwargs, names


def test_atomic_panel_price_and_unknown_are_scoped(monkeypatch):
    # The full domain test checks the producer range separately; keep the
    # allocation fixture small while exercising real rows and real candidate
    # construction/aggregation/solver/serialization.
    from prismaquant import tessera_sampled_stack_proposal as mod
    from prismaquant import tessera_menu
    from prismaquant import allocator_candidates
    joint, *_tail, names = pilot_fixture()
    monkeypatch.setattr(mod, '_domain', lambda *args, **kwargs: {'fixture': 'producer-tested-separately'})
    admitted_modes = []
    def admits(mode):
        admitted_modes.append(mode)
        return mode == 'research'
    monkeypatch.setattr(tessera_menu, 'route_admission',
                        lambda *args, **kwargs: SimpleNamespace(requires_serving_context=True,
                                                                 admits=admits))
    monkeypatch.setattr(allocator_candidates, 'serving_lane_route',
                        lambda *args, **kwargs: None)
    result = propose_bound_payload(joint, profile=AtomicProfile(),
        mutable_budget_bytes=600000, immutable_bytes=101, reserve_bytes=11,
        full_legal_families=PRIMARY)
    assert result['status'] == 'research_proposal' and result['export_authority'] is False
    assert 'research' in admitted_modes
    assert set(result['expanded_assignment']) == set(names)
    assert len(set(result['expanded_assignment'].values())) == 1
    assert len(result['solver_assignment']) == 1
    assert result['sampled_panel_summary']['uncertainty_scope'] == 'probe_sampling_conditional_on_fixed_calibration'
    assert result['observations'][names[3]]['full_population_price'] is None
    assert result['observations'][names[3]]['observed_panel_contribution'] == 0.0
    assert result['exact_bytes']['total'] <= result['exact_bytes']['total_budget']
    assert result['independent_selected_assignment_validation']['required_for_production_promotion']


def test_stack_members_reach_aggregation_with_whole_menus(monkeypatch):
    # Per-member reduction before aggregation can empty a stack's name
    # intersection; test_allocator_packed_group_menu_reduction has the case.
    from prismaquant import tessera_sampled_stack_proposal as mod
    joint, *_tail, names = pilot_fixture()
    monkeypatch.setattr(mod, '_domain', lambda *args, **kwargs: {})
    seen = {}

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        seen.update(kwargs)
        raise Captured
    monkeypatch.setattr(mod, 'build_candidates', capture)
    with pytest.raises(Captured):
        propose_bound_payload(joint, profile=AtomicProfile(), mutable_budget_bytes=600000,
            immutable_bytes=101, reserve_bytes=11, full_legal_families=PRIMARY)
    assert seen['defer_menu_reduction'] == frozenset(names)


def test_unknown_independent_and_incomplete_stack_refuse(monkeypatch):
    from prismaquant import tessera_sampled_stack_proposal as mod
    joint, *_tail, names = pilot_fixture()
    monkeypatch.setattr(mod, '_domain', lambda *args, **kwargs: {})
    joint['provenance'][tep.PROJECTION_KEY]['stacks'][STACK].pop(names[1])
    with pytest.raises(tep.ExpertProjectionError, match='producer|projected|atomic|projects'):
        propose_bound_payload(joint, profile=AtomicProfile(), mutable_budget_bytes=10000,
            immutable_bytes=0, reserve_bytes=0, full_legal_families=PRIMARY)


def test_inconsistent_same_expert_panel_counts_and_nonzero_unknown_refuse(monkeypatch):
    from prismaquant import tessera_sampled_stack_proposal as mod
    from prismaquant.cost_currency import CostCurrencyError
    joint, *_tail, names = pilot_fixture()
    monkeypatch.setattr(mod, '_domain', lambda *args, **kwargs: {})
    changed = joint['stats'][names[1]]['joint_eval_observations']
    changed['tokens'] += 1
    changed['per_probe'][0]['tokens'] += 1
    for row in joint['costs'][names[1]].values():
        row['joint_eval_observations'] = copy.deepcopy(changed)
    with pytest.raises(ValueError, match='inconsistent per-probe route counts'):
        propose_bound_payload(joint, profile=AtomicProfile(), mutable_budget_bytes=1000000,
            immutable_bytes=0, reserve_bytes=0, full_legal_families=PRIMARY)
    joint, *_tail, names = pilot_fixture()
    unknown = joint['costs'][names[3]]['TESSERA_E4M3_K1_R1024']
    unknown['signed_per_probe'][0] = 1.0
    unknown['x2_per_probe'][0] = 1.0
    with pytest.raises(CostCurrencyError, match='unobserved panel row has nonzero'):
        propose_bound_payload(joint, profile=AtomicProfile(), mutable_budget_bytes=1000000,
            immutable_bytes=0, reserve_bytes=0, full_legal_families=PRIMARY)


def test_byte_budget_refuses_even_when_solver_has_a_solution(monkeypatch):
    from prismaquant import tessera_sampled_stack_proposal as mod
    joint, *_tail, names = pilot_fixture()
    monkeypatch.setattr(mod, '_domain', lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match='exact-byte-feasible'):
        propose_bound_payload(joint, profile=AtomicProfile(), mutable_budget_bytes=1,
            immutable_bytes=101, reserve_bytes=11, full_legal_families=PRIMARY)


def test_scoped_binding_retains_pilot_and_generic_path_refuses():
    from prismaquant.cost_currency import CostCurrencyError, require_run_currency
    from prismaquant.tessera_joint_allocation import bind_allocation_payload
    joint, data, prepared, metadata, kwargs, names = pilot_fixture()
    with pytest.raises(CostCurrencyError, match='separate sampled-proposal'):
        require_run_currency(joint)
    # This fixture's wire/source identities predate its BF16 conversion above;
    # verify the narrow currency and panel-calibration seam independently.
    from prismaquant.cost_currency import require_sampled_joint_run_currency
    assert require_sampled_joint_run_currency(joint)['joint_aura_rows'] == 2 * len(names)
    with pytest.raises(ValueError, match='separate sampled-proposal'):
        bind_allocation_payload(joint, data, prepared, metadata, **kwargs)


def test_full_legal_inventory_contains_unmeasured_producer_rates():
    from prismaquant.tessera_formats import family_q256_bounds, get_tessera_family
    from prismaquant.tessera_sampled_stack_proposal import _domain
    assert family_q256_bounds(get_tessera_family('TESSERA_E4M3_K1')) == (256, 2048)
    assert family_q256_bounds(get_tessera_family('TESSERA_BF16_K1')) == (256, 4096)
    joint, *_tail = pilot_fixture()
    with pytest.raises(ValueError, match='retain primary'):
        _domain(['TESSERA_E4M3_K1'], joint['costs'], joint['stats'])
    inventory = _domain(PRIMARY,
                        joint['costs'], joint['stats'])
    assert inventory['TESSERA_E4M3_K1']['producer_grammar_q256'] == [256, 2048]
    assert inventory['TESSERA_BF16_K1']['producer_grammar_q256'] == [256, 4096]
    assert inventory['TESSERA_E4M3_K1']['measured_rungs_by_shape']['256x256'] == [1024]
    assert inventory['TESSERA_E4M3_K1']['producer_legal_by_shape']['256x256']['count'] == 1793
    assert inventory['TESSERA_BF16_K1']['producer_legal_by_shape']['256x256']['count'] == 3841
    for width in (2048, 4096, 12288):
        shape_stats = {'shape': {'out_features': 256, 'in_features': width}}
        shape_costs = {'shape': {'BF16': {}}}
        all_rates = _domain(PRIMARY,
                            shape_costs, shape_stats)
        assert all_rates['TESSERA_E4M3_K1']['producer_legal_by_shape'][f'256x{width}']['count'] == 1793
        assert all_rates['TESSERA_BF16_K1']['producer_legal_by_shape'][f'256x{width}']['count'] == 3841


def test_research_selected_cache_requires_proposal_and_exact_assignment():
    from prismaquant.tessera_sampled_stack_proposal import (
        SCHEMA, require_research_proposal_assignment, selected_assignment_sha256)
    from prismaquant.tessera_export_lane import selected_cached_units_manifest, TesseraExportLaneError
    from tessera.cached_unit import CACHE_SCHEMA
    joint, *_tail, names = pilot_fixture()
    assignment = {name: 'BF16' for name in names}
    proposal = {'schema': SCHEMA, 'status': 'research_proposal',
        'production_export_authority': False, 'validation_export_eligible': None,
        'research_validation_permitted': True, 'pilot': joint['provenance']['joint_eval'],
        'expanded_assignment': assignment,
        'selected_assignment_sha256': selected_assignment_sha256(assignment),
        'original_joint_plan_sha256': joint['provenance']['tessera_joint_allocation']['plan_sha256'],
        'original_prepared': joint['provenance']['tessera_joint_allocation']['prepared']}
    assert require_research_proposal_assignment(proposal, assignment)['production_export_authority'] is False
    changed = dict(assignment)
    changed[names[0]] = 'TESSERA_E4M3_K1_R1024'
    with pytest.raises(ValueError, match='differs'):
        require_research_proposal_assignment(proposal, changed)
    with pytest.raises(TesseraExportLaneError, match='completed joint allocation handoff'):
        selected_cached_units_manifest(assignment, {}, joint, None, schema=CACHE_SCHEMA)


def test_missing_separate_validation_target_refuses_before_large_input_reads():
    from prismaquant.tessera_sampled_stack_proposal import propose_from_bound_inputs
    with pytest.raises(ValueError, match='before reading large pilot'):
        propose_from_bound_inputs(joint_binding={'path': '/missing/pilot', 'sha256': 'a'*64},
            plan_binding={'path': '/missing/plan', 'sha256': 'b'*64},
            output_path='/missing/proposal', mutable_budget_bytes=1,
            immutable_bytes=1, reserve_bytes=1, full_legal_families=PRIMARY)


def test_official_preflight_refuses_research_marker_even_if_rung_is_native(tmp_path):
    import json
    from prismaquant import tessera_export_lane as tel
    path = tmp_path / 'layer_config.json'
    path.write_text(json.dumps({'model.layers.0.mlp.down_proj': 'TESSERA_E4M3_K1_R1024',
        '__prismaquant__': {'sampled_joint_proposal': {
            'schema': 'prismaquant.tessera_sampled_validation_export_binding.v1',
            'proposal_sha256': 'a'*64}}}))
    with pytest.raises(tel.TesseraExportLaneError, match='ordinary native export preflight remains closed'):
        tel.preflight(tmp_path, assignment_path=path)
