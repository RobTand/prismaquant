"""CPU contract checks for the experimental four-unit screen preparation."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_path = Path(__file__).parents[1] / 'experiments/glm_native_wire_screen_plan.py'
_spec = importlib.util.spec_from_file_location('glm_screen_plan', _path)
plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan)


def roster():
    names = [f'unit{i}' for i in range(4)]
    shapes = {name: [8, 4] for name in names}
    families = [dict(family='TESSERA_BF16_K1', round_one_rungs_q256=[256, 1147, 2038]),
        dict(family='TESSERA_E4M3_K1', round_one_rungs_q256=[256, 1149, 2042]),
        dict(family='TESSERA_E2M1_K2', round_one_rungs_q256=[896])]
    cells = [dict(qname=name, shape=shapes[name], source_group='complete',
        format=f"{family['family']}_R{rate}", memory_bytes=16)
        for name in names for family in families for rate in family['round_one_rungs_q256']]
    return (dict(schema='prismaquant.glm_native_candidate_screen.proposal.v3',
                 selected_logical_units=names, cells=cells),
            dict(unit_shapes=shapes),
            dict(groups={'complete': dict(members=list(names), families=families)}))


def test_full_group_grid_is_authoritative_over_old_per_shape_rate():
    proposal, census, groups = roster()
    assert len(plan.validate_cells(proposal, census, groups)) == 4
    proposal['cells'][1]['format'] = 'TESSERA_BF16_K1_R1148'
    with pytest.raises(ValueError, match='complete-group initial grid'):
        plan.validate_cells(proposal, census, groups)


@pytest.mark.parametrize('change,match', [
    ('duplicate', 'duplicate'), ('shape', 'original census'),
    ('membership', 'complete source group'), ('missing', '28 distinct')])
def test_untrusted_screen_roster_refuses_drift(change, match):
    proposal, census, groups = roster()
    if change == 'duplicate':
        proposal['cells'][1] = copy.deepcopy(proposal['cells'][0])
    elif change == 'shape':
        proposal['cells'][0]['shape'] = [4, 8]
    elif change == 'membership':
        groups['groups']['complete']['members'].pop()
    else:
        proposal['cells'].pop()
    with pytest.raises(ValueError, match=match):
        plan.validate_cells(proposal, census, groups)


def test_input_content_seal_refuses_replacement(tmp_path):
    source = tmp_path/'input.json'
    source.write_text('{}')
    digest = plan.hashlib.sha256(source.read_bytes()).hexdigest()
    assert plan.sealed_json(source, digest) == {}
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='sealed input changed'):
        plan.sealed_json(source, digest)


def test_verified_loader_owners_do_not_enter_gpu_subset():
    proposal, census, _groups = roster()
    common = dict(selected_source_weight_bytes=256, declared_headroom_bytes=24*plan.GIB)
    base = dict(schema='prismaquant.selected_anchor_resources.v2',
        selected_source_weight_bytes=256,
        phases=dict(source_preparation=dict(common, nonbody_source_bytes=64,
            source_window_bytes=128, loader_transient_bytes=128),
            export_inputs=dict(common),
            resident_anchors=dict(common, selected_hessian_bytes=256,
                selected_prefix_bytes=128, encoder_memo_bytes=64,
                factorization_scratch_bytes=256, compatible_batch_weight_bytes=128)))
    result = plan.resource_plan(base, census['unit_shapes'],
        {name: 2 for name in census['unit_shapes']}, proposal['cells'])
    cpu = result['physical_phases']['capture_prefetch']
    gpu = result['gpu_phases']['capture_prefetch']
    assert cpu['serialized_private_buffer_bytes'] == plan.GIB
    assert cpu['source_page_cache_bytes'] == 4*plan.GIB
    assert cpu['decoded_cpu_entry_bytes'] == 96
    assert cpu['selected_device_capture_bytes'] == 384
    assert not any('buffer' in key or 'page' in key or 'cpu' in key for key in gpu)
    assert result['physical_bytes'] > result['gpu_bytes']
    assert result['guard_envelope_bytes'] > result['physical_bytes']
    assert result['requested_mem_gib']*plan.GIB >= result['guard_envelope_bytes']
    assert result['status'] == 'DERIVED_UNMEASURED'
    assert base['phases']['resident_anchors'] == dict(common,
        selected_hessian_bytes=256, selected_prefix_bytes=128, encoder_memo_bytes=64,
        factorization_scratch_bytes=256, compatible_batch_weight_bytes=128)


def test_native_entry_point_has_no_unauthenticated_source_fallback():
    from experiments.glm_native_wire_screen import require_source_api
    cc = SimpleNamespace()
    tc = SimpleNamespace(_checked_projected_units=lambda: None)
    with pytest.raises(RuntimeError, match='authentication is not integrated'):
        require_source_api(cc, tc, lambda: None)
    cc.authenticate_selected_capture_source = lambda: None
    with pytest.raises(RuntimeError, match='consumer propagation'):
        require_source_api(cc, tc, lambda: None)
    def authenticated(*, source_authentication=None):
        pass
    tc._checked_projected_units = authenticated
    cc.prefetch_capture = lambda: None
    with pytest.raises(RuntimeError, match='verified capture loader'):
        require_source_api(cc, tc, authenticated)
    def verified(*, verified_load_policy=None):
        pass
    cc.prefetch_capture = verified
    require_source_api(cc, tc, authenticated)
