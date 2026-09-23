"""Resource changes retain the source/probe contract and use the existing planner."""
import copy
import hashlib
import json
import pickle
from pathlib import Path

import pytest

from prismaquant.joint_served_activation import FORMAT
from prismaquant.joint_stageb_resources import derive_policy, verify_policy, require_plan_resources


def bound(path, value, binary=False):
    raw = pickle.dumps(value) if binary else json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def resource_fixture(tmp_path, monkeypatch):
    from prismaquant import model_profiles
    from prismaquant.production_weight_cache import ProductionWeightCache
    from prismaquant.joint_retained_window_plan import RetainedWindowBudget
    from prismaquant.joint_served_activation import derive_policy as activation_policy
    monkeypatch.setattr(model_profiles, 'detect_profile', lambda _: None)
    names = [f'model.language_model.layers.{layer}.mlp.experts.{e}.{role}_proj'
             for layer in (8, 9) for e in range(2) for role in ('gate', 'up', 'down')]
    fmt = 'TESSERA_E4M3_K1_R1024'
    budget = RetainedWindowBudget(64 << 20, 1 << 20, 1 << 20, 1 << 20,
        1 << 20, 1 << 20, 1 << 20, 4096, 1 << 20, 16, 1 << 20, 1 << 20, 1)
    plan = {'execution': {'seed_base': 7, 'n_probes': 2, 'probe_microbatch': 1,
        'source_derivative': {'fixture': 'original'},
        'operator_windows': {'prefetch_workers': 1},
        'retained_operator_windows': {'budget': budget.as_dict(), 'source_reserve_bytes': 1 << 20}},
        'max_gpu_bytes': 48 << 20, 'inputs': {}}
    old_plan = bound(tmp_path/'plan.json', plan)
    weights = {}; cells = {}; additions = []
    for name in names:
        path = tmp_path/(name+'.old.pt');path.write_bytes(b'o'*2048)
        weights[name, fmt] = str(path)
        cells[name, fmt] = {'source_weight': {'shape': [16, 16]}}
        new = tmp_path/(name+'.new.pt');new.write_bytes(b'n'*2048);s=new.stat()
        additions.append({'qname': name, 'format': FORMAT, 'render': str(new),
            'render_stat': {'inode': s.st_ino, 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}})
    census = bound(tmp_path/'census.json', {'model': '/synthetic', 'unit_shapes': {n: [16, 16] for n in names}})
    cache = ProductionWeightCache(weights=weights, levers={}, activation_max_abs={n: float(i+1) for i,n in enumerate(names)},
        metadata={'inputs': {'census': census}, 'verified_cells': cells})
    prepared = {'status': 'complete', 'plan_sha256': old_plan['sha256'],
        'calibration_input': {'shape': [512, 512]}, 'source_model_identity': {'fixture': True},
        'production_cache': bound(tmp_path/'cache.pkl', cache, True),
        'formats_by_qname': {n: [fmt, 'BF16'] for n in names}}
    old_prepared = bound(tmp_path/'prepared.json', prepared)
    catalog = bound(tmp_path/'catalog.json', {'old_prepared': old_prepared, 'cells': additions})
    activation = bound(tmp_path/'activation.json', activation_policy(old_prepared))
    inputs = {'original_plan': old_plan, 'original_prepared': old_prepared,
              'candidate_overlay': catalog, 'served_activation_policy': activation}
    limits = dict(host_bytes=16 << 20, physical_bytes=64 << 20, gpu_bytes=48 << 20)
    policy = derive_policy(inputs, **limits)
    policy_binding = bound(tmp_path/'resources.json', policy)
    extended = copy.deepcopy(plan)
    extended['execution']['retained_operator_windows']['budget'] = policy['budget']
    extended['inputs']['candidate_overlay'] = catalog
    extended['served_activation_policy'] = activation
    extended['stage_b_resource_policy'] = policy_binding
    return inputs, plan, extended, policy_binding, policy


def test_resource_derivation_fixes_indivisible_delta_and_preserves_science(resource_fixture):
    inputs, original, extended, binding, policy = resource_fixture
    assert original['execution']['retained_operator_windows']['budget']['candidate_delta_bytes'] == 16
    assert policy['budget']['candidate_delta_bytes'] == 16 * 16 * 4
    assert verify_policy(binding) == policy
    assert require_plan_resources(original, extended, inputs['original_plan'], inputs['original_prepared']) == policy


@pytest.mark.parametrize('field', ['seed_base', 'n_probes', 'probe_microbatch', 'source_derivative'])
def test_resource_exception_cannot_change_scientific_execution(resource_fixture, field):
    inputs, original, extended, binding, policy = resource_fixture
    extended['execution'][field] = 'changed'
    with pytest.raises(ValueError, match='non-resource execution'):
        require_plan_resources(original, extended, inputs['original_plan'], inputs['original_prepared'])


def test_forged_budget_and_changed_candidate_sizes_refuse(resource_fixture, tmp_path):
    inputs, original, extended, binding, policy = resource_fixture
    forged = copy.deepcopy(policy);forged['budget']['max_windows_per_layer'] += 100
    with pytest.raises(ValueError, match='independent resource derivation'):
        verify_policy(bound(tmp_path/'forged.json', forged))
    Path(policy['candidate_files'][0]['path']).write_bytes(b'changed size')
    with pytest.raises(ValueError, match='independent resource derivation|candidate file changed'):
        verify_policy(binding, verify_files=True)


def test_resource_gpu_envelope_is_applied_before_any_pricing(resource_fixture, monkeypatch):
    from prismaquant import memory_management
    from prismaquant.joint_stageb_resources import enforce_device_policy
    inputs, original, extended, binding, policy = resource_fixture
    calls = []
    monkeypatch.setattr(memory_management, 'enforce_device_envelope',
        lambda device, limit, **kwargs: calls.append((device, limit)) or {'limit': limit})
    record = enforce_device_policy(extended)
    assert calls == [('cuda', policy['limits']['gpu_bytes'])]
    assert record['policy'] == binding


def test_extended_catalog_reads_original_capture_namespace_without_rewriting_it():
    from prismaquant.joint_cost_quantum import quantum_adjoint_space
    # The quantum's stage-A slice (PQ #993): one checkpoint, the one it reads.
    adjoint_slice = {'boundary_storage': {'directory': '/original/layer-quanta/adjoint/exact-boundaries'},
        'checkpoint': {'boundary': 8, 'activation_entries': [
            {'path': '/original/layer-quanta/adjoint/checkpoints/boundary-008/entries/cotangent.pt'}],
            'shared_state_entries': []}}
    original = copy.deepcopy(adjoint_slice)
    assert quantum_adjoint_space({'catalog_extension': {'path': '/proof'}}, adjoint_slice, '/new') == Path('/original/layer-quanta/adjoint')
    assert adjoint_slice == original
    adjoint_slice['checkpoint']['activation_entries'][0]['path'] = '/new/foreign.pt'
    with pytest.raises(RuntimeError, match='escaped'):
        quantum_adjoint_space({'catalog_extension': {'path': '/proof'}}, adjoint_slice, '/new')


def test_worker_resource_verification_uses_sealed_geometry_without_render_stats(resource_fixture, monkeypatch):
    from prismaquant import joint_stageb_resources as resources
    _, _, _, binding, policy = resource_fixture
    resources._VERIFIED.clear()
    original = resources._bound_stat_fence
    def metadata_only(path):
        assert path.suffix != '.pt', 'worker repeated whole-catalog render stats'
        return original(path)
    monkeypatch.setattr(resources, '_bound_stat_fence', metadata_only)
    assert resources.verify_policy(binding) == policy
