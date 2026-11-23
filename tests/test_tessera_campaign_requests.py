"""Campaign phases bind only their actual operation inputs."""
import json

import pytest

from prismaquant import tessera_campaign as campaign
from prismaquant.tessera_campaign_cli import parse_campaign_arguments
from prismaquant.tessera_campaign_requests import (
    CaptureMonolith, CaptureQuantum, CaptureRequest, CensusRequest,
    census_request, streamed_calibration_request,
)


BASE = ['--model', 'checkpoint', '--out', 'cost.pkl', '--cache-dir', 'cache']


def _parsed(extra):
    args = parse_campaign_arguments(BASE + extra)
    args.calibration_input_receipt = None
    return args


def test_census_binds_draw_scope_and_destination_without_capture_options():
    args = _parsed(['--census-out', 'census.json', '--seed=0'])
    request = streamed_calibration_request(args)
    assert request == CensusRequest('checkpoint', 8, 512, 0, 1, None,
                                    'cache', 'census.json', 1, None)
    assert not hasattr(request, 'capture_calibration_out')


@pytest.mark.parametrize('phase', [None, 'quantum'])
def test_capture_binds_its_actual_phase_and_draw_receipt(phase):
    args = _parsed(['--calibration-census', 'census.json',
                    '--capture-calibration-out', 'capture', '--max-act-rows=32'])
    receipt = {'provenance': {'source': 'owned-input', 'seed': 72}}
    args.calibration_input_receipt = receipt
    args.capture_chain = phase
    args.capture_layer_range = (1, 3) if phase else None
    request = streamed_calibration_request(args)
    assert isinstance(request, CaptureRequest)
    assert request.calibration_input_receipt == receipt
    assert request.max_act_rows == 32
    assert request.calibration_census == 'census.json'
    assert request.capture_calibration_out == 'capture'
    assert request.phase == (CaptureMonolith() if phase is None else CaptureQuantum((1, 3)))
    assert not hasattr(request, 'census_out')


@pytest.mark.parametrize('mode', ['prep', 'join'])
def test_model_running_request_refuses_bookend_modes(mode):
    args = _parsed(['--capture-calibration-out', 'capture', '--capture-chain', mode])
    with pytest.raises(ValueError, match='monolith or quantum'):
        streamed_calibration_request(args)


def test_census_request_requires_its_destination():
    with pytest.raises(ValueError, match='census request requires its output'):
        census_request(_parsed([]))


def test_reference_law_request_uses_configured_rungs():
    from dataclasses import replace
    from types import SimpleNamespace

    from prismaquant.tessera_campaign_requests import loaded_campaign_request

    family = 'TESSERA_E4M3_K1'
    args = _parsed(['--reference-law-anchors', '--units', 'units.json'])
    request = loaded_campaign_request(args, mode='research', selected_source=False,
        streaming_head=False, wire_dir=None, checkpoint=None, identity_threads=1)
    sample = SimpleNamespace(is_census=True, reference_q256=960,
        transfer_law_experts=(0,), members={0: ('a',), 1: ('b',)})
    placement = replace(request.recipe.placement, reference_law_samples={'stack': sample})
    groups = {'s:stack': ['a', 'b']}
    legal = {'s:stack': {family: [768, 832, 896, 960, 1024, 1088, 1152]}}
    assert request.reference_law_anchors
    actual = campaign._campaign_round_one_schedule(groups, legal,
        placement=placement, audit_units=set(), snap=None)
    assert actual == {'a': {family: [832, 960, 1088]}, 'b': {family: [960]}}

    args = _parsed(['--reference-law-anchors', '--units', 'units.json',
        '--reference-q256', '1024', '--lower-target-q256', '896', '--upper-target-q256', '960'])
    request = loaded_campaign_request(args, mode='research', selected_source=False,
        streaming_head=False, wire_dir=None, checkpoint=None, identity_threads=1)
    sample.reference_q256 = 1024
    placement = replace(request.recipe.placement, reference_law_samples={'stack': sample})
    assert campaign._campaign_round_one_schedule(groups, legal,
        placement=placement, audit_units=set(), snap=None) == {
            'a': {family: [896, 960, 1024]}, 'b': {family: [1024]}}


def _publish(tmp_path, expected, maximum=1.5):
    request = CensusRequest('checkpoint', 2, 4, 0, 1, None, 'cache',
                            str(tmp_path / 'results/census.json'), 1, None)
    result = campaign.run_census(request, counts={'unit': 8}, maxima={'unit': maximum},
        groups={'group': ['unit']}, dense_targets=['unit'], expert_targets=[],
        shapes={'unit': (3, 4)}, identity={'text_sha256': 'a'*64, 'fit_ids_sha256': 'b'*64},
        expert_projection=None, model_load_contract=None,
        attention_implementation=None, capture_runtime=None, pinned_roster=None,
        expected_units=expected)
    return result, tmp_path / 'results/census.json'


@pytest.mark.parametrize('maximum', [1.5, float('nan'), float('inf')])
def test_resident_census_publishes_exact_existing_record_bytes(tmp_path, maximum):
    result, path = _publish(tmp_path, ['unit'], maximum)
    assert result == 0
    expected = {'schema': campaign.CENSUS_SCHEMA, 'model': 'checkpoint', 'nsamples': 2,
        'seqlen': 4, 'seed': 0, 'layer_stride': 1, 'text_sha256': 'a'*64,
        'fit_ids_sha256': 'b'*64, 'counts': {'unit': 8}, 'max_abs': {'unit': maximum},
        'unit_shapes': {'unit': [3, 4]}, 'anchor_groups': {'group': ['unit']},
        'dense_targets': ['unit'], 'expert_targets': [], 'expert_projection': None}
    assert path.read_bytes() == (json.dumps(expected, indent=2, sort_keys=True) + '\n').encode()


def test_resident_census_refuses_unobserved_units_before_creating_output(tmp_path):
    with pytest.raises(RuntimeError, match='did not observe every unit.*missing unit-missing'):
        _publish(tmp_path, ['unit', 'unit-missing'])
    assert not (tmp_path / 'results').exists()
