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


NAMES = tuple(f'model.language_model.layers.{layer}.mlp.experts.{e}.{role}_proj'
              for layer in (8, 9) for e in range(2) for role in ('gate', 'up', 'down'))


def _inputs(tmp_path, monkeypatch, *, added=None, activation_formats=None):
    """The four bindings a resource derivation reads, and the original plan.

    With ``added`` unset, a v1 catalog adds ``FORMAT`` to every unit. With
    ``added`` (``{format: qnames}``), a v2 catalog adds those cells, and the
    served activation policy prices ``activation_formats``.
    """
    from prismaquant import model_profiles
    from prismaquant.production_weight_cache import ProductionWeightCache
    from prismaquant.joint_retained_window_plan import RetainedWindowBudget
    from prismaquant.joint_served_activation import derive_policy as activation_policy
    monkeypatch.setattr(model_profiles, 'detect_profile', lambda _: None)
    names = list(NAMES)
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    for added_fmt, qnames in ({FORMAT: names} if added is None else added).items():
        for name in qnames:
            new = tmp_path/(name+'.'+added_fmt+'.new.pt');new.write_bytes(b'n'*2048);s=new.stat()
            additions.append({'qname': name, 'format': added_fmt, 'render': str(new),
                'render_stat': {'inode': s.st_ino, 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}})
    census = bound(tmp_path/'census.json', {'model': '/synthetic', 'unit_shapes': {n: [16, 16] for n in names}})
    cache = ProductionWeightCache(weights=weights, levers={}, activation_max_abs={n: float(i+1) for i,n in enumerate(names)},
        metadata={'inputs': {'census': census}, 'verified_cells': cells})
    prepared = {'status': 'complete', 'plan_sha256': old_plan['sha256'],
        'calibration_input': {'shape': [512, 512]}, 'source_model_identity': {'fixture': True},
        'production_cache': bound(tmp_path/'cache.pkl', cache, True),
        'formats_by_qname': {n: [fmt, 'BF16'] for n in names}}
    old_prepared = bound(tmp_path/'prepared.json', prepared)
    if added is None:
        catalog = bound(tmp_path/'catalog.json', {'old_prepared': old_prepared, 'cells': additions})
        activation = bound(tmp_path/'activation.json', activation_policy(old_prepared))
    else:
        from prismaquant.joint_catalog_extension import CATALOG_SCHEMA_V2
        source = {'cost': {'path': str(tmp_path/'cost.pkl'), 'sha256': '0'*64},
                  'anchor_journal': None, 'reseal_proof': None}
        catalog = bound(tmp_path/'catalog.json', {
            'schema': CATALOG_SCHEMA_V2, 'old_prepared': old_prepared, 'cells': additions,
            'formats': sorted(added), 'sources': [source], 'cell_sources': [0]*len(additions),
            'carried_from': []})
        activation = bound(tmp_path/'activation.json',
                           activation_policy(old_prepared, formats=activation_formats))
    inputs = {'original_plan': old_plan, 'original_prepared': old_prepared,
              'candidate_overlay': catalog, 'served_activation_policy': activation}
    return inputs, plan, names


LIMITS = dict(host_bytes=16 << 20, physical_bytes=64 << 20, gpu_bytes=48 << 20)


@pytest.fixture
def resource_fixture(tmp_path, monkeypatch):
    inputs, plan, _names = _inputs(tmp_path, monkeypatch)
    catalog, activation = inputs['candidate_overlay'], inputs['served_activation_policy']
    limits = LIMITS
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



def test_resource_geometry_progress_goes_to_stderr(resource_fixture, capsys):
    """PQ #1087: ``verify_policy`` runs inside ``dispatch_joint_quanta
    --dry-run``, whose stdout must parse as one JSON document. The geometry
    progress lines are logs: stderr, never stdout."""
    _inputs, _original, _extended, binding, policy = resource_fixture
    capsys.readouterr()
    assert verify_policy(binding) == policy
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "resource geometry: reused sealed observations for" in captured.err

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


CAPTURE_RECEIPT = {'action_key': 'c' * 64, 'path': '/receipts/profile.json', 'sha256': 'd' * 64}


def _capture(batch, workspace):
    return {'capture_batch': batch, 'workspace_reserve_bytes': {
        'bytes': workspace, 'receipt': dict(CAPTURE_RECEIPT), 'basis': 'test'}}


def test_a_measured_capture_rides_the_policy_and_its_rederivation(resource_fixture, tmp_path):
    """PQ #1151: the workspace is measured, named by its receipt, and re-derived."""
    inputs, _original, _extended, _binding, legacy = resource_fixture
    assert 'capture' not in legacy and 'capture' not in legacy['derivation']
    assert 'workspace_reserve_bytes' in legacy['derivation']['declared']
    limits = dict(host_bytes=16 << 20, physical_bytes=64 << 20, gpu_bytes=48 << 20)
    policy = derive_policy(inputs, **limits, capture=_capture(4, 2 << 20))
    assert policy['capture'] == _capture(4, 2 << 20)
    assert policy['budget']['workspace_reserve_bytes'] == 2 << 20
    derivation = policy['derivation']
    assert 'workspace_reserve_bytes' not in derivation['declared']
    assert derivation['measured']['workspace_reserve_bytes']['receipt'] == CAPTURE_RECEIPT
    assert derivation['capture']['workspace_bytes'] == 4 * (2 << 20)
    assert derivation['peak_planned_bytes'] >= derivation['capture']['peak_planned_bytes']
    assert verify_policy(bound(tmp_path / 'measured.json', policy)) == policy
    # A policy whose capture block was edited no longer re-derives.
    forged = copy.deepcopy(policy)
    forged['capture']['workspace_reserve_bytes']['bytes'] = 1 << 20
    with pytest.raises(ValueError, match='independent resource derivation'):
        verify_policy(bound(tmp_path / 'forged-capture.json', forged))


def test_a_capture_that_does_not_fit_refuses_the_policy(resource_fixture):
    inputs, *_ = resource_fixture
    limits = dict(host_bytes=16 << 20, physical_bytes=64 << 20, gpu_bytes=48 << 20)
    with pytest.raises(RuntimeError, match='capture pass at capture_batch 4 plans'):
        derive_policy(inputs, **limits, capture=_capture(4, 16 << 20))


def test_a_host_held_cotangent_plane_is_priced_into_the_render_bound(resource_fixture, tmp_path):
    """PQ #1141: the plane the checkpoint load holds on the host is a host owner."""
    from prismaquant.joint_retained_window_plan import HOST_RESIDENT_BUDGET_FIELDS
    inputs, _original, _extended, _binding, legacy = resource_fixture
    assert 'cotangent' not in legacy and 'cotangent' not in legacy['derivation']
    limits = dict(host_bytes=16 << 20, physical_bytes=64 << 20, gpu_bytes=48 << 20)
    plane = 6 << 20
    policy = derive_policy(inputs, **limits, cotangent={'host_plane_bytes': plane})
    assert policy['cotangent'] == {'host_plane_bytes': plane}
    budget = policy['budget']
    resident = sum(budget[name] for name in HOST_RESIDENT_BUDGET_FIELDS)
    assert budget['retained_render_cap_bytes'] + resident + plane <= limits['host_bytes']
    assert verify_policy(bound(tmp_path / 'plane.json', policy)) == policy
    forged = copy.deepcopy(policy)
    forged['cotangent']['host_plane_bytes'] = 1
    with pytest.raises(ValueError, match='independent resource derivation'):
        verify_policy(bound(tmp_path / 'forged-plane.json', forged))


def test_a_cotangent_plane_that_exhausts_the_host_refuses_the_policy(resource_fixture):
    inputs, *_ = resource_fixture
    limits = dict(host_bytes=16 << 20, physical_bytes=64 << 20, gpu_bytes=48 << 20)
    with pytest.raises(RuntimeError, match='cotangent plane'):
        derive_policy(inputs, **limits, cotangent={'host_plane_bytes': 16 << 20})


def test_the_receipt_is_read_once_for_its_bytes_and_digest(tmp_path):
    import hashlib
    from prismaquant.joint_stageb_resources import workspace_from_receipt
    from prismaquant.stage_b_workspace_profile import SCHEMA, write_profile

    path = tmp_path / 'profile.json'
    profile = {'schema': SCHEMA, 'ladder_complete': True,
               'measured': {'workspace_per_batch_bytes': 3 << 20, 'basis': 'device peak'}}
    digest = write_profile(path, profile)
    owner = workspace_from_receipt(path, action_key='e' * 64)
    assert owner == {'bytes': 3 << 20, 'basis': 'device peak',
                     'receipt': {'action_key': 'e' * 64, 'path': str(path), 'sha256': digest}}
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    incomplete = tmp_path / 'incomplete.json'
    write_profile(incomplete, {**profile, 'ladder_complete': False})
    with pytest.raises(ValueError, match='incomplete ladder'):
        workspace_from_receipt(incomplete, action_key='e' * 64)


#: The fixture's Stage A chain regime, and a chain block pricing layer 9's
#: shape (the fixture's two layers share one) at it (PQ #1163).
CHAIN_REGIME = {'batch_size': 4, 'probe_fusion': True}


def _chain(workspace, *, source='measured'):
    owner = {'layers': [9], 'bytes': workspace, 'device_resident_bytes': 1 << 20,
             'host_committed_bytes': 1 << 20, 'regime': dict(CHAIN_REGIME),
             'source': source, 'basis': 'test'}
    if source == 'measured':
        owner['receipt'] = dict(CAPTURE_RECEIPT)
    return {'chain_regime': dict(CHAIN_REGIME), 'shapes': {'routed': owner}}


def test_a_priced_chain_rides_the_policy_and_its_rederivation(resource_fixture, tmp_path):
    """PQ #1163: the chain phase is planned per shape and re-derived."""
    inputs, _original, _extended, _binding, legacy = resource_fixture
    assert 'chain' not in legacy and 'chain' not in legacy['derivation']
    assert not any(key.startswith('chain_') for key in legacy['budget'])
    limits = dict(host_bytes=16 << 20, physical_bytes=64 << 20, gpu_bytes=48 << 20)
    policy = derive_policy(inputs, **limits, chain=_chain(2 << 20))
    assert policy['chain'] == _chain(2 << 20)
    assert policy['budget']['chain_workspace_reserve_bytes'] == 2 << 20
    assert policy['budget']['chain_layers'] == [9]
    chain = policy['derivation']['chain']
    # The policy's device ceiling is the envelope the chain is checked against.
    assert chain['device_limit_bytes'] == policy['budget']['chain_device_limit_bytes'] == 48 << 20
    assert chain['device_margin_bytes'] == (48 << 20) - (3 << 20)
    assert chain['shapes']['routed']['receipt'] == CAPTURE_RECEIPT
    assert policy['derivation']['peak_planned_bytes'] >= chain['peak_planned_bytes']
    assert verify_policy(bound(tmp_path / 'chain.json', policy)) == policy
    forged = copy.deepcopy(policy)
    forged['chain']['shapes']['routed']['bytes'] = 1 << 20
    with pytest.raises(ValueError, match='independent resource derivation'):
        verify_policy(bound(tmp_path / 'forged-chain.json', forged))
    # A declared owner states itself and carries no receipt.
    declared = derive_policy(inputs, **limits, chain=_chain(2 << 20, source='declared'))
    assert declared['derivation']['chain']['shapes']['routed']['source'] == 'declared'
    assert 'receipt' not in declared['derivation']['chain']['shapes']['routed']


def test_a_chain_that_does_not_fit_refuses_the_policy(resource_fixture):
    inputs, *_ = resource_fixture
    limits = dict(host_bytes=16 << 20, physical_bytes=64 << 20, gpu_bytes=48 << 20)
    with pytest.raises(RuntimeError, match="chain layer shape 'routed' at batch 4 .* device"):
        derive_policy(inputs, **limits, chain=_chain((47 << 20) + 1))


def _roll(layer, *, failure=None, admission=True):
    return {'layer': layer, 'failure': failure,
            'allocated_delta_bytes': 4 << 20, 'reserved_delta_bytes': 5 << 20,
            'admission': ({'cuda_reserved_bytes': 3 << 20, 'cgroup_committed_bytes': 2 << 20}
                          if admission else None),
            'host': {'peak_current_bytes': 12 << 20,
                     'peak_stat_all': {'anon': 1 << 20, 'file': 8 << 20, 'shmem': 0,
                                       'file_dirty': 0, 'file_writeback': 0}}}


def test_the_chain_receipt_is_read_once_for_a_complete_roll(tmp_path):
    from prismaquant.joint_stageb_resources import chain_owner_from_receipt
    from prismaquant.stage_b_workspace_profile import CHAIN_SCHEMA, write_profile

    path = tmp_path / 'chain.json'
    profile = {'schema': CHAIN_SCHEMA, 'identity': {'chain_regime': dict(CHAIN_REGIME)},
               'rolls': [_roll(44), _roll(43, failure='MemoryError: roll failed'),
                         _roll(41, admission=False)]}
    digest = write_profile(path, profile)
    owner = chain_owner_from_receipt(path, action_key='e' * 64, layer=44, layers=[44, 4])
    basis = owner.pop('basis')
    # The host side is the larger of the admission's committed bytes (2 MiB)
    # and the committed bytes at the roll's cgroup peak (12 - 8 MiB of clean file).
    assert owner == {'layers': [4, 44], 'bytes': 5 << 20, 'device_resident_bytes': 3 << 20,
                     'host_committed_bytes': 4 << 20, 'regime': CHAIN_REGIME,
                     'source': 'measured',
                     'receipt': {'action_key': 'e' * 64, 'path': str(path), 'sha256': digest}}
    assert basis.startswith("layer 44's chain roll: workspace = max(allocated delta")
    with pytest.raises(ValueError, match='incomplete roll of layer 43'):
        chain_owner_from_receipt(path, action_key='e' * 64, layer=43, layers=[43])
    with pytest.raises(ValueError, match='no chain roll of layer 42'):
        chain_owner_from_receipt(path, action_key='e' * 64, layer=42, layers=[42])
    with pytest.raises(ValueError, match='no admission reading'):
        chain_owner_from_receipt(path, action_key='e' * 64, layer=41, layers=[41])


# -- a v2 catalog over several formats (PQ #1437) ------------------------------

R768 = 'TESSERA_E2M1_K2_R768'
E4M3 = 'TESSERA_E4M3_K1_R1152'


def test_a_v2_catalog_derives_with_every_added_format_before_bf16(tmp_path, monkeypatch):
    """Every member gains the policy's two A4 formats; half also gain an A8
    rung, which the policy does not price. The derivation observes every
    added render and re-derives from its own record."""
    names = list(NAMES)
    added = {R768: names, FORMAT: names, E4M3: names[::2]}
    inputs, _plan, _ = _inputs(tmp_path, monkeypatch, added=added, activation_formats=[R768, FORMAT])
    policy = derive_policy(inputs, **LIMITS)
    members = {tuple(row['member']) for row in policy['candidate_files']}
    assert len(members) == len(names) + sum(len(qs) for qs in added.values())
    assert {(n, f) for f, qs in added.items() for n in qs} <= members
    assert verify_policy(bound(tmp_path / 'resources.json', policy)) == policy


@pytest.mark.parametrize('case, match', [
    ('member_missing_a_policy_format', 'not the served policy'),
    ('a4_format_outside_the_policy', 'not the served policy'),
    ('format_already_offered', 'added candidate roster differs'),
])
def test_a_v2_catalog_refuses_a_routed_a4_rung_the_policy_does_not_price(tmp_path, monkeypatch, case, match):
    names = list(NAMES)
    added = {R768: names, FORMAT: names}
    if case == 'member_missing_a_policy_format':
        added[R768] = names[1:]
    elif case == 'a4_format_outside_the_policy':
        added['TESSERA_E2M1_K2_R640'] = names
    else:
        added['TESSERA_E4M3_K1_R1024'] = names[:1]
    inputs, _plan, _ = _inputs(tmp_path, monkeypatch, added=added, activation_formats=[R768, FORMAT])
    with pytest.raises(ValueError, match=match):
        derive_policy(inputs, **LIMITS)
