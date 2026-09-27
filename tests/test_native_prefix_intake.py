"""Fresh prefix acquisition must retain its narrower actual initialization scope."""
import copy

import pytest

from prismaquant.native_moe_panel import (
    _calibration_and_capture,
    _validate_prefix_capture,
    identity_sha256,
)
from test_native_moe_glm_geometry import glm_routing, glm_shape


def fixture(layer=3):
    heads=['lm_head.weight','model.language_model.embed_tokens.weight','model.language_model.norm.weight']
    names=heads+[f'model.language_model.layers.{i}.norm.weight' for i in range(layer + 1)]
    state={name:{'shape':[4],'dtype':'torch.bfloat16','kind':'checkpoint'} for name in names}
    checkpoint={name:'shard.safetensors' for name in names};live={name:name for name in names}
    contract={'schema':'prismaquant.streaming_prefix_initialization.v1','scope':'streamed_text_source_prefix',
        'status':'completed','transformers_version':'actual','model_class':'transformers.models.glm5_next.Glm5NextForConditionalGeneration',
        'dtype':'torch.bfloat16','layers_prefix':'model.language_model.layers.','total_model_layers':45,'observed_layers':list(range(layer + 1)),
        'head_state_names':sorted(heads),'state':state,'persistent_tensors':len(names),'derived_buffers':0,'state_sha256':identity_sha256(state),
        'source_map_sha256':identity_sha256({n:{'tensor':n,'file':'shard.safetensors'} for n in names})}
    source={'content_sha256':'a'*64,'weight_map':live,'checkpoint_weight_map':checkpoint}
    if layer == 44:
        contract = {key: value for key, value in contract.items()
                    if key not in {'total_model_layers', 'observed_layers', 'head_state_names', 'state'}}
        contract.update(schema='prismaquant.streaming_initialization.v1',
                        scope='streamed_text_source_forward', num_layers=45)
    capture={'unit':f'model.language_model.layers.{layer}.mlp.experts','shape':glm_shape(),'model_load_contract':contract,
        'producer_source':{'tensors':checkpoint},'replay':{'schema':'prismaquant.glm_routing_boundary_replay.v1',
            'layer':layer,'sample':0,'stop':'before_original_packed_experts_forward'},
        'dev_uncertified':True,'dev_mode':{'PRISMAQUANT_DEV_MODE':'1'},
        'source_cache_reuse':{'binding':{'path':'/cache','sha256':'b'*64},'complete_checkpoint':True,
            'validator':'validate_cached_streamed_model_identity','content_sha256':'a'*64}}
    return capture,source


def test_explicit_source_prefix_preserves_complete_source_binding():
    capture,source=fixture()
    assert _validate_prefix_capture(capture,source)['observed_layers']==[0,1,2,3]


@pytest.mark.parametrize('change',['source','map','head','layer','sample','certified','state'])
def test_prefix_claim_cannot_change_scope_or_source(change):
    capture,source=fixture()
    if change=='source':source['content_sha256']='c'*64
    elif change=='map':source['weight_map']['extra']='lm_head.weight'
    elif change=='head':capture['model_load_contract']['head_state_names'].remove('lm_head.weight')
    elif change=='layer':capture['unit']='model.language_model.layers.8.mlp.experts'
    elif change=='sample':capture['replay']['sample']=1
    elif change=='certified':capture['dev_uncertified']=False
    else:capture['model_load_contract']['state']['lm_head.weight']['dtype']='torch.float32'
    with pytest.raises(ValueError):_validate_prefix_capture(capture,source)


def calibration_capture(layer):
    capture, source = fixture(layer)
    calibration = {'schema': 'prismaquant.calibration_input.v1',
                   'calibration_sha256': '1' * 64, 'shape': [512, 512], 'dtype': 'torch.int64'}
    capture.update(schema='prismaquant.routed_boundary_capture.v1', routing=glm_routing(),
        calibration_sha256=calibration['calibration_sha256'],
        calibration_shape=calibration['shape'], calibration_dtype=calibration['dtype'],
        runtime_config={'model_type': 'glm5_next'}, capture_source_sha256='c' * 64,
        attention_implementation='eager',
        capture_runtime={'torch': 'actual', 'cuda': 'actual', 'transformers': 'actual'})
    capture['producer_source'].update(files={'shard.safetensors': 'd' * 64},
        config_sha256='e' * 64, auxiliary_sha256={'config.json': 'e' * 64})
    return calibration, capture, source


@pytest.mark.parametrize('layer', range(3, 45))
def test_every_routed_layer_passes_actual_calibration_gate(layer):
    calibration, capture, source = calibration_capture(layer)
    _validate_prefix_capture(capture, source)
    _calibration_and_capture(calibration, capture, unit=capture['unit'],
                             shape=capture['shape'], routing=capture['routing'])


@pytest.mark.parametrize('wrong', ['short', 'long', 'gap', 'replay', 'namespace',
                                 'full_early', 'wrong_total', 'boolean_sample'])
def test_wrong_prefix_never_reaches_calibration_consumer(wrong):
    calibration, capture, _ = calibration_capture(17)
    if wrong in {'short', 'long'}:
        other, _ = fixture(16 if wrong == 'short' else 18)
        capture['model_load_contract'] = other['model_load_contract']
    elif wrong == 'gap':
        capture['model_load_contract']['observed_layers'].remove(8)
    elif wrong == 'replay':
        capture['replay']['layer'] = 3
    elif wrong == 'namespace':
        capture['unit'] = 'model.layers.17.mlp.experts'
    elif wrong == 'full_early':
        full, _ = fixture(44)
        capture['model_load_contract'] = full['model_load_contract']
    elif wrong == 'wrong_total':
        capture['model_load_contract']['total_model_layers'] = 46
    else:
        capture['replay']['sample'] = False
    with pytest.raises(ValueError):
        _calibration_and_capture(calibration, capture, unit=capture['unit'],
                                 shape=capture['shape'], routing=capture['routing'])


@pytest.mark.parametrize('layer', [3, 8, 43, 44])
def test_streamed_boundary_round_trips_the_native_intake(layer):
    import torch

    from prismaquant.native_moe_panel import routed_boundary_inputs
    from prismaquant.production_weight_cache import _cb_cache_tensor_identity as tensor_identity

    calibration, capture, source = calibration_capture(layer)
    bias = torch.zeros(288, dtype=torch.float32)
    tensors = {
        'inputs': torch.ones(512, 4096, dtype=torch.bfloat16),
        'top_k_index': torch.arange(8).expand(512, 8).contiguous(),
        'top_k_weights': torch.full((512, 8), 2.5 / 8, dtype=torch.float32),
        'coordinates': torch.stack((torch.zeros(512, dtype=torch.int64), torch.arange(512)), dim=1),
        'expert_bias': bias,
    }
    capture.update(schema='prismaquant.native_moe_raw_boundary.v1',
                   profile_role_order=['w1', 'w3', 'w2'], scope='synthetic CPU first-sequence fixture',
                   tensors={name: tensor_identity(value) for name, value in tensors.items()})
    capture['routing']['topk_ids_dtype'] = 'torch.int64'
    capture['routing']['source_protocol']['correction_bias'] = {
        key: tensor_identity(bias)[key] for key in ('content_sha256', 'dtype')}
    manifest = {'schema': 'prismaquant.tessera_calibration_cache.v2', 'status': 'complete',
                'identity': {'attention_implementation': 'eager',
                             'capture_runtime': capture['capture_runtime'],
                             'source_files': capture['producer_source']['files']}}
    routed, phases, _ = routed_boundary_inputs(
        {'source': 'routed_boundary_capture', 'boundary_metadata': capture, **tensors},
        calibration_receipt=calibration, capture_manifest=manifest,
        device='cpu', source_model_identity=source)
    assert routed['unit'] == capture['unit']
    assert routed['source_acquisition']['dev_uncertified'] is True
    assert phases['prefill']['input'].shape == (512, 4096)
    assert phases['decode']['input'].shape == (1, 4096)


def test_terminal_layer_does_not_weaken_proper_prefix_contract():
    from prismaquant.streaming_model import validate_streaming_prefix_initialization_contract

    capture, _ = fixture(43)
    contract = copy.deepcopy(capture['model_load_contract'])
    contract['observed_layers'].append(44)
    with pytest.raises(ValueError, match='proper prefix'):
        validate_streaming_prefix_initialization_contract(contract)
    terminal, source = fixture(44)
    terminal['model_load_contract']['source_map_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='source map'):
        _validate_prefix_capture(terminal, source)
