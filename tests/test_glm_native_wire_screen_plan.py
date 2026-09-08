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


def test_two_early_netdata_samples_then_failure_cannot_pass():
    from experiments.glm_native_wire_screen import telemetry_coverage
    samples = _observations([0, 1])
    phases = [dict(phase='encode', started_monotonic=0.5, finished_monotonic=3599)]
    healthy = _observations(range(0, 3601, 10))
    assert telemetry_coverage(healthy, phases, [], thread_complete=True)['passed']
    errors = [dict(host='sparky', unix=2, error='endpoint stopped')]
    failed_reads = telemetry_coverage(healthy, phases, errors, thread_complete=True)
    assert failed_reads['failures'] == ['one or more required telemetry reads/writes failed']
    assert not failed_reads['passed']
    assert not telemetry_coverage(samples, phases, [], thread_complete=True)['passed']
    gap = telemetry_coverage(_observations([0, 1, 3600]), phases, [], thread_complete=True)
    assert any('coverage gap' in reason for reason in gap['failures'])
    assert not gap['passed']


def _observations(times):
    return {host: [dict(time=value, monotonic=value, oldest_chart_age_seconds=10,
                       max_future_chart_seconds=0) for value in times]
            for host in ('sparky', 'sparklina')}


def test_telemetry_requires_fresh_bracketed_continuous_coverage_on_both_hosts():
    from experiments.glm_native_wire_screen import telemetry_coverage
    phases = [dict(phase='encode', started_monotonic=1, finished_monotonic=39)]
    samples = _observations([0, 10, 20, 30, 40])
    assert telemetry_coverage(samples, phases, [], thread_complete=True)['passed']
    for broken in (_observations([0, 40]), _observations([10, 20, 30, 40]),
                   _observations([0, 10, 20, 30])):
        assert not telemetry_coverage(broken, phases, [], thread_complete=True)['passed']
    samples['sparklina'][2]['oldest_chart_age_seconds'] = 21
    assert not telemetry_coverage(samples, phases, [], thread_complete=True)['passed']


@pytest.mark.parametrize('changed', ['source_weight', 'hessian', 'inputs'])
def test_unversioned_input_write_is_caught_by_final_byte_gate(changed):
    import torch
    from experiments.glm_native_wire_screen_evidence import require_original_input_bytes
    from prismaquant.production_weight_cache import _cb_cache_tensor_identity as identity
    from prismaquant.tessera_campaign import _bound_tensor_signature
    tensors = dict(source_weight=torch.ones(2, 2, dtype=torch.bfloat16),
                   hessian=torch.eye(2), inputs=torch.ones(2, 2))
    expected = {name: identity(value) for name, value in tensors.items()}
    signatures = {name: _bound_tensor_signature(value) for name, value in tensors.items()}
    assert require_original_input_bytes(tensors, expected, identity=identity) == expected
    tensors[changed].view(torch.uint8).numpy()[0, 0] ^= 1
    assert signatures == {name: _bound_tensor_signature(value) for name, value in tensors.items()}
    with pytest.raises(RuntimeError, match=changed):
        require_original_input_bytes(tensors, expected, identity=identity)


def test_telemetry_chart_freshness_checks_the_actual_required_charts():
    from experiments.glm_native_wire_screen_evidence import telemetry_sample
    from experiments.workspace_netdata import REQUIRED_CHARTS, REQUIRED_GPU_SUFFIXES
    names = set(REQUIRED_CHARTS) | {'nvidia_smi.gpu0_power_draw'}
    names.update('nvidia_smi.gpu0_'+suffix for suffix in REQUIRED_GPU_SUFFIXES)
    sample = dict(host='sparky', time=100,
                  metrics={name: dict(last_updated=90) for name in names})
    assert telemetry_sample(sample, monotonic=50)['oldest_chart_age_seconds'] == 10
    sample['metrics']['nvidia_smi.gpu0_power_draw']['last_updated'] = 79
    with pytest.raises(ValueError, match='stale'):
        telemetry_sample(sample, monotonic=51)
    sample['metrics']['nvidia_smi.gpu0_power_draw']['last_updated'] = 103
    with pytest.raises(ValueError, match='clock-skewed'):
        telemetry_sample(sample, monotonic=51)


def _environment():
    activation = dict(PRISMAQUANT_PROD_ACT_SCALES='0',
        PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES='0', PRISMAQUANT_TESSERA_DEV_PIN='')
    return activation, dict(activation, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', PRISMAQUANT_LAYER_READ_THREADS='4',
        PRISMAQUANT_RELEASE_SOURCE_PAGES='1', MIMALLOC_PURGE_DELAY='0',
        PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE='0')


def test_cpu_preflight_checks_actual_integrated_api_and_frozen_environment():
    import torch
    from experiments.glm_native_wire_screen import integrated_cpu_preflight
    activation, environment = _environment()
    result = integrated_cpu_preflight(environment, activation, environ=environment)
    assert result['passed'] and not result['gpu_initialized']
    assert not torch.cuda.is_initialized()
    for name in environment:
        missing = dict(environment); missing.pop(name)
        with pytest.raises(ValueError, match='omits a required'):
            integrated_cpu_preflight(missing, activation, environ=environment)
        changed = dict(environment, **{name: 'different'})
        with pytest.raises(ValueError, match='environment differs'):
            integrated_cpu_preflight(environment, activation, environ=changed)


def test_actual_integrated_authenticator_refuses_partial_capture_before_source_read(tmp_path):
    import json
    from prismaquant import tessera_calibration_cache as cc
    capture = tmp_path/'capture.json'
    capture.write_text(json.dumps(dict(schema=cc.SCHEMA, status='partial', identity={}, entries={})))
    with pytest.raises((ValueError, RuntimeError), match='complete|identity|canonical'):
        cc.authenticate_selected_capture_source(tmp_path/'absent-census.json', capture,
            expected_sha256=cc.sha256(capture), model='unopened-model', max_act_rows=512,
            attention_implementation='eager')
