"""Prefix audit under the exact source image without replacing authenticated HF code."""
import os
import pytest
import torch
from test_glm_campaign_streaming import glm_checkpoint
from prismaquant.cost_streaming import build_streamed_causal_lm
from prismaquant.model_profiles.glm5_next import Glm5NextProfile

def _observed_fixture_derivative(tmp_path):
    from prismaquant.glm_source_derivative import CORRECTED_IMAGE_CONTENT_SHA256
    if os.environ.get('PRISMAQUANT_CONTAINER_CONTENT_SHA256') == CORRECTED_IMAGE_CONTENT_SHA256:
        from test_glm_source_derivative import fixture_policy
        return fixture_policy(tmp_path)
    return None


def test_streamed_partial_source_returns_only_a_prefix_contract(glm_checkpoint, tmp_path):
    import copy
    from prismaquant.streaming_model import (
        validate_streaming_prefix_initialization_contract,
        validate_streaming_initialization_contract,
        _initialization_digest,
    )
    reference, source = glm_checkpoint
    runner = build_streamed_causal_lm(str(source), device=torch.device('cpu'),
        dtype=torch.float32, offload_folder=str(tmp_path/'prefix-offload'),
        profile=Glm5NextProfile(), max_cache_slots=2, prefetch_workers=1,
        prefetch_min_available_gb=0, cache_headroom_gb=0,
        prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation='eager', source_derivative=_observed_fixture_derivative(tmp_path))
    try:
        runner.context.begin_source_initialization_audit()
        with pytest.raises(RuntimeError,match='contiguous'):
            runner.context.source_prefix_initialization_contract(0)
        runner.context.schedule_prefetch(0)
        runner.context.install(0,require_prefetched=True,prefetch_following=False)
        contract=runner.context.source_prefix_initialization_contract(0)
        assert contract['observed_layers']==[0] and contract['total_model_layers']==2
        assert {'model.language_model.embed_tokens.weight','model.language_model.norm.weight','lm_head.weight'} <= set(contract['head_state_names'])
        assert validate_streaming_prefix_initialization_contract(contract)==contract
        with pytest.raises(ValueError):validate_streaming_initialization_contract(contract)
        with pytest.raises(RuntimeError,match='every source layer'):
            runner.context.source_initialization_contract()
        for changed in ({'observed_layers':[1]}, {'observed_layers':[0,1]},
                        {'head_state_names':[]}, {'state_sha256':'0'*64}):
            bad=copy.deepcopy(contract);bad.update(changed)
            with pytest.raises(ValueError):validate_streaming_prefix_initialization_contract(bad)
        bad=copy.deepcopy(contract)
        bad['state']['model.language_model.layers.1.foreign.weight']={'shape':[1],'dtype':'torch.float32','kind':'checkpoint'}
        bad['state_sha256']=_initialization_digest(bad['state']);bad['persistent_tensors']+=1
        with pytest.raises(ValueError,match='outside'):
            validate_streaming_prefix_initialization_contract(bad)
    finally:
        runner.shutdown()


def test_complete_owner_still_requires_and_accepts_every_actual_layer(glm_checkpoint,tmp_path):
    from prismaquant.streaming_model import validate_streaming_initialization_contract
    reference,source=glm_checkpoint
    runner=build_streamed_causal_lm(str(source),device=torch.device('cpu'),dtype=torch.float32,
        offload_folder=str(tmp_path/'full-offload'),profile=Glm5NextProfile(),
        max_cache_slots=2,prefetch_workers=1,prefetch_min_available_gb=0,
        cache_headroom_gb=0,prefetch_lookahead=1,require_prefetched_residency=True,
        attn_implementation='eager',source_derivative=_observed_fixture_derivative(tmp_path))
    try:
        runner.context.begin_source_initialization_audit()
        for layer in range(runner.num_layers):
            runner.context.schedule_prefetch(layer)
            runner.context.install(layer,require_prefetched=True,prefetch_following=False)
            runner.context.unload(layer)
        contract=runner.context.source_initialization_contract()
        assert contract['num_layers']==2
        assert validate_streaming_initialization_contract(contract)==contract
        with pytest.raises(RuntimeError,match='contiguous'):
            runner.context.source_prefix_initialization_contract(0)
    finally:runner.shutdown()
