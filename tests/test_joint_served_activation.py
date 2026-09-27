"""A4 grouping changes priced arithmetic, never original qualification."""
import copy
import hashlib
import json
import pickle
from types import SimpleNamespace

import pytest

from prismaquant import format_registry
from prismaquant.joint_aura import activation_identity
from prismaquant.joint_served_activation import (
    FORMAT, FORMAT_MAXIMA_KEY, derive_policy, verify_policy, activate_policy,
    joint_activation_maxima, operator_policy_record, policy_group, require_priced_activation,
)


def write(path, value, binary=False):
    raw = pickle.dumps(value) if binary else json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def policy_fixture(tmp_path, monkeypatch):
    from prismaquant.production_weight_cache import ProductionWeightCache
    from prismaquant import model_profiles
    monkeypatch.setattr(model_profiles, 'detect_profile', lambda _: None)
    names = [f'model.language_model.layers.8.mlp.experts.{expert}.{role}_proj'
             for expert in range(2) for role in ('gate', 'up', 'down')]
    maxima = {name: float(index + 1) for index, name in enumerate(names)}
    census = write(tmp_path/'census.json', {'model': '/synthetic', 'unit_shapes': {n: [16, 16] for n in names}})
    cache = ProductionWeightCache(weights={}, levers={}, activation_max_abs=maxima,
                                  metadata={'inputs': {'census': census}})
    prepared = write(tmp_path/'prepared.json', {'status': 'complete',
        'calibration_input': {'shape': [512, 512], 'calibration_sha256': '1'*64},
        'production_cache': write(tmp_path/'cache.pkl', cache, True),
        'source_model_identity': {'synthetic': True},
        'formats_by_qname': {n: ['TESSERA_E4M3_K1_R1024', 'BF16'] for n in names}})
    policy = derive_policy(prepared)
    bound = write(tmp_path/'policy.json', policy)
    return bound, policy, cache, names


def test_rederives_complete_groups_and_refuses_forged_maximum(policy_fixture, tmp_path):
    bound, policy, cache, names = policy_fixture
    assert verify_policy(bound) == policy
    assert len(policy['executed_grouping']['groups']) == 2
    assert policy['effective_max_abs'][names[0]] == 5.0
    assert policy['effective_max_abs'][names[2]] == 6.0
    bad = copy.deepcopy(policy)
    bad['effective_max_abs'][names[0]] = 2.0
    with pytest.raises(ValueError, match='group maxima'):
        verify_policy(write(tmp_path/'forged.json', bad))


def test_format_scoped_view_preserves_all_old_metadata_and_a8_identity(policy_fixture, monkeypatch):
    bound, policy, cache, names = policy_fixture
    monkeypatch.setenv('PRISMAQUANT_PROD_ACT_SCALES', '0')
    before = copy.deepcopy(cache.activation_max_abs)
    cache.weights = {(n, FORMAT): '/synthetic' for n in names}
    old_metadata = copy.deepcopy(cache.metadata)
    old8 = activation_identity(format_registry.get_format('TESSERA_E4M3_K1_R1024'), before, names[0])
    activate_policy(cache, bound)
    view = joint_activation_maxima(cache)
    assert cache.activation_max_abs == before and cache.metadata == old_metadata
    assert activation_identity(format_registry.get_format('TESSERA_E4M3_K1_R1024'), view, names[0]) == old8
    a4 = activation_identity(format_registry.get_format(FORMAT), view, names[0])
    expected_scale = policy_group(policy, names[0], FORMAT)[1]['input_global_scale']
    assert a4['activation_max_abs'] == 5.0 and a4['input_global_scale'] == expected_scale


def test_price_receipt_and_allocator_use_actual_group_scale(policy_fixture, monkeypatch):
    from prismaquant.tessera_menu import priced_static_scales
    bound, policy, cache, names = policy_fixture
    monkeypatch.setenv('PRISMAQUANT_PROD_ACT_SCALES', '0')
    cache.weights = {(n, FORMAT): '/synthetic' for n in names}
    activate_policy(cache, bound)
    name = names[0]
    qualified = activation_identity(format_registry.get_format(FORMAT), cache.activation_max_abs, name)
    priced = activation_identity(format_registry.get_format(FORMAT), joint_activation_maxima(cache), name)
    operator = {'activation': priced, 'served_activation_policy': operator_policy_record(bound, policy, name, FORMAT, qualified)}
    expected_scale = policy_group(policy, name, FORMAT)[1]['input_global_scale']
    assert require_priced_activation(bound, policy, name, FORMAT, qualified, operator) == expected_scale
    assert qualified['input_global_scale'] == 6.0
    rows = {name: {FORMAT: {'joint_operator_identity': operator, 'input_global_scale': expected_scale}}}
    selected = priced_static_scales({name: FORMAT}, rows,
        policy=policy['executed_grouping']['input_global_scale_policy'], served_activation_policy=bound)
    assert selected['units'][name] == expected_scale
    assert selected['served_activation_policy'] == bound
    assert selected['activation_scale_grouping'] == policy['executed_grouping']
    operator['activation'] = qualified
    with pytest.raises(ValueError, match='priced activation differs'):
        require_priced_activation(bound, policy, name, FORMAT, qualified, operator)


def test_missing_members_cannot_activate_group(policy_fixture):
    bound, policy, cache, names = policy_fixture
    cache.weights = {(n, FORMAT): '/synthetic' for n in names[:-1]}
    with pytest.raises(ValueError, match='candidate missing'):
        activate_policy(cache, bound)


def test_policy_refuses_dependency_mutation_during_verification(policy_fixture, monkeypatch):
    from pathlib import Path
    from prismaquant import joint_served_activation as owner
    bound, policy, cache, names = policy_fixture
    original = owner.derive_policy
    def racing_read(prepared):
        result = original(prepared)
        path = Path(policy['census']['path'])
        path.write_bytes(path.read_bytes() + b'\n')
        return result
    monkeypatch.setattr(owner, 'derive_policy', racing_read)
    with pytest.raises(ValueError, match='changed during verification'):
        owner.verify_policy(bound)


def test_handoff_keeps_qualified_activation_but_exports_priced_group_value(policy_fixture, tmp_path, monkeypatch):
    from test_tessera_joint_allocation import fixture
    from test_joint_aura_assignment_diagnostics import _rebuild
    from prismaquant import joint_catalog_extension
    from prismaquant.tessera_joint_allocation import bind_allocation_payload
    bound, policy, cache, names = policy_fixture
    joint, data, prepared, metadata, kwargs = fixture(names)
    oldfmt = 'TESSERA_E4M3_K1_R1024'
    cache.weights = {(n, FORMAT): '/synthetic' for n in names}
    activate_policy(cache, bound)
    for name in names:
        qualified = activation_identity(format_registry.get_format(FORMAT), cache.activation_max_abs, name)
        priced = activation_identity(format_registry.get_format(FORMAT), joint_activation_maxima(cache), name)
        def change_probe(probe):
            probe['arithmetic']['served_activation_policy'] = bound
        def change_operator(operator):
            operator.update(format=FORMAT, activation=priced,
                served_activation_policy=operator_policy_record(bound, policy, name, FORMAT, qualified))
        joint['costs'][name][FORMAT] = _rebuild(joint['costs'][name].pop(oldfmt),
            probe_change=change_probe, operator_change=change_operator)
        joint['costs'][name]['BF16'] = _rebuild(joint['costs'][name]['BF16'], probe_change=change_probe)
        receipt = metadata['verified_cells'].pop((name, oldfmt))
        receipt['activation'] = qualified
        metadata['verified_cells'][name, FORMAT] = receipt
        data.cells[name, FORMAT] = data.cells.pop((name, oldfmt))
        anchor = data.payload['costs'][name].pop(oldfmt)
        anchor['input_global_scale'] = qualified['input_global_scale']
        data.payload['costs'][name][FORMAT] = anchor
        data.formats_by_qname[name] = (FORMAT, 'BF16')
        prepared['formats_by_qname'][name] = [FORMAT, 'BF16']
    joint['formats'] = [FORMAT, 'BF16']
    prepared['served_activation_policy'] = bound
    joint['provenance']['served_activation_policy'] = bound
    # This test isolates the handoff from the independently tested full
    # capture/extension authenticator, and verifies that gate is still called.
    checked = []
    monkeypatch.setattr(joint_catalog_extension, 'require_extension', lambda *a, **k: checked.append(k))
    # A v2 extension embeds the original Stage A run header (PQ #993); the
    # gate receives exactly that header.
    header = {'synthetic': True}
    extension = {'schema': joint_catalog_extension.SCHEMA,
                 'inputs': {'original_prepared': policy['original_prepared'],
                            'extended_prepared': kwargs['prepared_binding'],
                            'extended_plan': {'sha256': kwargs['plan_sha256']}},
                 'adjoint_run_header': header}
    joint['provenance']['catalog_extension'] = write(tmp_path/'extension.json', extension)
    original = copy.deepcopy(metadata['verified_cells'])
    result = bind_allocation_payload(joint, data, prepared, metadata, **kwargs)
    assert checked and metadata['verified_cells'] == original
    assert checked[0]['run_header'] == header
    for name in names:
        expected = policy_group(policy, name, FORMAT)[1]['input_global_scale']
        assert result['costs'][name][FORMAT]['input_global_scale'] == expected
        assert result['costs'][name][FORMAT]['joint_operator_identity'] == joint['costs'][name][FORMAT]['joint_operator_identity']
        assert result['costs'][name][FORMAT]['predicted_dloss'] == joint['costs'][name][FORMAT]['predicted_dloss']


def test_alternate_reader_is_hash_bound_and_used_for_every_dependency(policy_fixture, monkeypatch):
    from pathlib import Path
    from prismaquant import joint_served_activation as owner
    bound, policy, cache, names = policy_fixture
    monkeypatch.setattr(owner, '_VERIFIED', {})
    reads = []
    def pinned_read(binding):
        reads.append(binding['path'])
        return Path(binding['path']).read_bytes()
    assert owner.verify_policy(bound, read_bound=pinned_read) == policy
    # The policy and all three of its calibrated dependencies came through the reader.
    assert set(reads) == {bound['path'], policy['original_prepared']['path'],
                          policy['original_cache']['path'], policy['census']['path']}

    monkeypatch.setattr(owner, '_VERIFIED', {})
    def drifting_read(binding):
        raw = Path(binding['path']).read_bytes()
        return raw + b' ' if binding['path'] == policy['census']['path'] else raw
    with pytest.raises(ValueError, match='alternate pinned reader changed bound content'):
        owner.verify_policy(bound, read_bound=drifting_read)


# -- several added routed A4 formats (PQ #1437) --------------------------------

R768 = 'TESSERA_E2M1_K2_R768'


def _prepared_binding(policy):
    return policy['original_prepared']


def test_an_unset_format_list_derives_the_v1_policy(policy_fixture):
    from prismaquant.joint_served_activation import SCHEMA, policy_formats
    bound, policy, cache, names = policy_fixture
    assert policy['schema'] == SCHEMA and policy['format'] == FORMAT and 'formats' not in policy
    assert policy_formats(policy) == (FORMAT,)
    assert derive_policy(_prepared_binding(policy)) == policy


def test_a_v2_policy_prices_every_listed_a4_format_at_the_group_scale(policy_fixture, tmp_path, monkeypatch):
    from prismaquant.joint_served_activation import SCHEMA_V2, policy_formats
    _bound, v1, cache, names = policy_fixture
    monkeypatch.setenv('PRISMAQUANT_PROD_ACT_SCALES', '0')
    policy = derive_policy(_prepared_binding(v1), formats=[R768, FORMAT])
    assert policy['schema'] == SCHEMA_V2 and policy['formats'] == [R768, FORMAT] and 'format' not in policy
    assert policy_formats(policy) == (R768, FORMAT)
    # The group maxima are the unit's, so they are the v1 policy's.
    assert policy['effective_max_abs'] == v1['effective_max_abs']
    bound = write(tmp_path/'policy-v2.json', policy)
    assert verify_policy(bound) == policy
    for fmt in (R768, FORMAT):
        assert policy_group(policy, names[0], fmt) == policy_group(v1, names[0], FORMAT)
    assert policy_group(policy, names[0], 'TESSERA_E4M3_K1_R1024') is None
    cache.weights = {(n, fmt): '/synthetic' for n in names for fmt in (R768, FORMAT)}
    activate_policy(cache, bound)
    view = joint_activation_maxima(cache)
    expected = policy_group(policy, names[0], R768)[1]['input_global_scale']
    for fmt in (R768, FORMAT):
        a4 = activation_identity(format_registry.get_format(fmt), view, names[0])
        assert a4['activation_max_abs'] == 5.0 and a4['input_global_scale'] == expected


def test_a_v2_policy_needs_every_format_for_every_member(policy_fixture, tmp_path):
    _bound, v1, cache, names = policy_fixture
    policy = derive_policy(_prepared_binding(v1), formats=[R768, FORMAT])
    bound = write(tmp_path/'policy-v2.json', policy)
    cache.weights = {(n, fmt): '/synthetic' for n in names for fmt in (R768, FORMAT)}
    del cache.weights[names[1], R768]
    with pytest.raises(ValueError, match=f'candidate missing for .*: {R768}'):
        activate_policy(cache, bound)


@pytest.mark.parametrize('formats, match', [
    ([FORMAT, R768], 'sorted unique'),
    ([R768, R768], 'sorted unique'),
    (['TESSERA_E4M3_K1_R1152'], 'not A4'),
    (['TESSERA_E4M3_K1_R1024'], 'not A4'),
])
def test_a_v2_policy_refuses_an_unsorted_or_non_a4_format_list(policy_fixture, formats, match):
    _bound, v1, _cache, _names = policy_fixture
    with pytest.raises(ValueError, match=match):
        derive_policy(_prepared_binding(v1), formats=formats)


def test_a_v2_policy_refuses_a_format_a_member_already_offers(policy_fixture, tmp_path):
    from pathlib import Path
    _bound, v1, _cache, _names = policy_fixture
    prepared = json.loads(Path(v1['original_prepared']['path']).read_bytes())
    prepared['formats_by_qname'] = {n: ['TESSERA_E4M3_K1_R1024', R768, 'BF16']
                                    for n in prepared['formats_by_qname']}
    offered = write(tmp_path/'prepared-offers-r768.json', prepared)
    with pytest.raises(ValueError, match='only price newly added'):
        derive_policy(offered, formats=[R768, FORMAT])
    assert derive_policy(offered, formats=[FORMAT])['formats'] == [FORMAT]
