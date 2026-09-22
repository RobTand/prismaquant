"""Fresh prefix acquisition must retain its narrower actual initialization scope."""
import copy
import pytest
from prismaquant.native_moe_panel import _validate_prefix_capture,identity_sha256
from test_native_moe_glm_geometry import glm_shape


def fixture():
    heads=['lm_head.weight','model.language_model.embed_tokens.weight','model.language_model.norm.weight']
    names=heads+[f'model.language_model.layers.{i}.norm.weight' for i in range(4)]
    state={name:{'shape':[4],'dtype':'torch.bfloat16','kind':'checkpoint'} for name in names}
    checkpoint={name:'shard.safetensors' for name in names};live={name:name for name in names}
    contract={'schema':'prismaquant.streaming_prefix_initialization.v1','scope':'streamed_text_source_prefix',
        'status':'completed','transformers_version':'actual','model_class':'transformers.models.glm5_next.Glm5NextForConditionalGeneration',
        'dtype':'torch.bfloat16','layers_prefix':'model.language_model.layers.','total_model_layers':45,'observed_layers':[0,1,2,3],
        'head_state_names':sorted(heads),'state':state,'persistent_tensors':7,'derived_buffers':0,'state_sha256':identity_sha256(state),
        'source_map_sha256':identity_sha256({n:{'tensor':n,'file':'shard.safetensors'} for n in names})}
    source={'content_sha256':'a'*64,'weight_map':live,'checkpoint_weight_map':checkpoint}
    capture={'unit':'model.language_model.layers.3.mlp.experts','shape':glm_shape(),'model_load_contract':contract,
        'producer_source':{'tensors':checkpoint},'replay':{'schema':'prismaquant.glm_routing_boundary_replay.v1',
            'layer':3,'sample':0,'stop':'before_original_packed_experts_forward'},
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
