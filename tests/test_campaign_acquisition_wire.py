"""Actual acquired-row settings bind the real verified source and exact work."""
from copy import deepcopy
from types import MappingProxyType
import pickle

import pytest

from prismaquant import tessera_full_domain_acquisition as intake
from prismaquant.tensor_digests import tensor_sha256
from test_tessera_row_stream import UNITS, stream_fixture


FAMILY = 'TESSERA_E4M3_K1'


@pytest.mark.parametrize('source_changed', [False, True])
def test_actual_acquired_main_binds_raw_settings_and_source_before_pricing(
        monkeypatch, tmp_path, source_changed):
    from safetensors.torch import load_file, save_file
    from prismaquant import cost_streaming

    campaign, argv, _state = stream_fixture(monkeypatch, tmp_path)
    values = {}
    for shard in sorted((tmp_path/'source').glob('*.safetensors')):
        values.update(load_file(shard))
    sources = {}
    for name in UNITS:
        weight = values[name+'.weight']
        sources[name] = {'shape': list(weight.shape), 'dtype': str(weight.dtype),
            'logical_bytes': weight.numel()*weight.element_size(),
            'content_sha256': tensor_sha256(weight)}
    requests = {name: {FAMILY: [1024]} for name in UNITS}
    identity = {field: str(index)*64 for index, field in enumerate((
        'request_sha256', 'cost_sha256', 'joint_aura_identity_sha256',
        'probe_identity_sha256', 'request_control_sha256'), 1)}
    packet = {'requests': requests, 'source_weights': deepcopy(sources), 'identity': identity}
    binding = {'path': str(tmp_path/'authenticated-request.json'), 'sha256': 'a'*64}
    events, observed = [], []

    def authenticated(binding):
        assert binding == {'path': str(tmp_path/'authenticated-request.json'), 'sha256': 'a'*64}
        events.append('authenticated-intake')
        return deepcopy(packet)

    actual_identity = campaign.campaign_checkpoint_identity
    def record_identity(**kwargs):
        raw = kwargs['settings']
        assert isinstance(raw, MappingProxyType)
        with pytest.raises(TypeError):
            raw['injected'] = True
        before = deepcopy(dict(raw))
        result = actual_identity(**kwargs)
        assert dict(raw) == before
        observed.append((before, result))
        events.append('checkpoint-identity')
        return result

    actual_measure = campaign._measure_anchor
    def measured(**kwargs):
        events.append(('priced', kwargs['qname']))
        return actual_measure(**kwargs)

    monkeypatch.setattr(intake, 'load_joint_campaign_acquisition', authenticated)
    monkeypatch.setattr(campaign, 'campaign_checkpoint_identity', record_identity)
    monkeypatch.setattr(campaign, '_measure_anchor', measured)
    command = [*argv, '--row-head', 'load-all', '--acquisition-request', binding['path'],
               '--acquisition-request-sha256', binding['sha256']]
    if source_changed:
        actual_build = cost_streaming.build_streamed_causal_lm
        def change_held_source(*args, **kwargs):
            # The valid capture/intake attest the original bytes. Replace the
            # physical source only after actual source authentication enters.
            assert kwargs.get('source_authentication') is not None
            path = tmp_path/'source/layer0.safetensors'
            changed = load_file(path)
            key = UNITS[0]+'.weight'
            changed[key] = changed[key].clone()
            changed[key][0, 0] += 1
            replacement = tmp_path/'changed_layer0.safetensors'
            save_file(changed, replacement)
            replacement.replace(path)
            events.append('physical-source-mutated')
            return actual_build(*args, **kwargs)
        monkeypatch.setattr(cost_streaming, 'build_streamed_causal_lm', change_held_source)
        with pytest.raises(RuntimeError, match='changed|content differs|identity differs|another object'):
            campaign.main(command)
        assert events == ['authenticated-intake', 'physical-source-mutated']
        assert not observed
        assert not (tmp_path/'cost.pkl').exists()
        assert not list((tmp_path/'cache/wire').glob('*.tessera'))
        return

    assert campaign.main(command) == 0
    assert events[0] == 'authenticated-intake'
    assert {event[1] for event in events if isinstance(event, tuple)} == set(UNITS)
    assert observed
    expected_schedule = {name: {FAMILY: [1024]} for name in UNITS}
    expected_origin = {**identity,
        'purpose': 'actual_scalar_render_journal_and_wire_preparation',
        'currency': 'output_mse_under_route_activation_contract',
        'expanded_actual_work_count': 3, 'deferred_domain': dict.fromkeys(UNITS, [])}
    for raw, record in observed:
        assert raw['acquisition_request'] == binding['path']
        assert raw['acquisition_schedule'] == expected_schedule
        assert raw['acquisition_origin'] == expected_origin
        assert raw['acquisition_source_weights'] == sources
        assert record['settings']['acquisition_schedule'] == expected_schedule
        assert record['settings']['acquisition_origin'] == expected_origin
        assert 'acquisition_request' not in record['settings']
        assert 'acquisition_source_weights' not in record['settings']
        assert {name: record['units'][name]['acquisition_source_weight'] for name in UNITS} == sources
    with (tmp_path/'cost.pkl').open('rb') as handle:
        payload = pickle.load(handle)
    assert payload['provenance']['acquisition_schedule'] == expected_schedule
    assert payload['provenance']['acquisition'] == expected_origin
