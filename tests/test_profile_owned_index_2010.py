"""Owned checkpoint namespace evidence cannot reopen the logical pool index."""
import json
from pathlib import Path

import pytest

from prismaquant.model_profiles import profile_from_config


@pytest.mark.parametrize('owned_names,expected', [
    (['model.layers.0.mlp.gate.weight'], 'model.layers.0.mlp.gate.weight'),
    (['model.language_model.layers.0.mlp.gate.weight'],
     'model.language_model.layers.0.mlp.gate.weight'),
])
def test_owned_qwen_index_wins_over_conflicting_pool_namespace(tmp_path, monkeypatch, owned_names, expected):
    profile = profile_from_config({'model_type': 'qwen3_5_moe',
                                  'architectures': ['Qwen3_5MoeForCausalLM']})
    profile._declare_model_path(tmp_path)
    path = tmp_path / 'model.safetensors.index.json'
    path.write_text(json.dumps({'weight_map': {'untrusted_pool': 'wrong.safetensors'}}))
    # Attach the explicit private metadata intake. The causal predecessor has
    # the same profile source lookup but ignores this admitted evidence.
    profile._declared_checkpoint_index = {'weight_map': {name: 'owned.safetensors' for name in owned_names}}
    real = Path.read_text
    def opened(path, *args, **kwargs):
        assert path != tmp_path / 'model.safetensors.index.json', 'owned profile reopened mutable pool index'
        return real(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', opened)
    assert profile.source_tensor_name('model.layers.0.mlp.gate.weight') == expected


@pytest.mark.parametrize('weight_map', [{}, {'unrelated': 'owned.safetensors'},
    {'model.layers.0.w': 'a.safetensors', 'model.language_model.layers.0.w': 'b.safetensors'}])
def test_ambiguous_owned_index_refuses_without_path_discovery(tmp_path, monkeypatch, weight_map):
    profile = profile_from_config({'model_type': 'qwen3_5_moe',
                                  'architectures': ['Qwen3_5MoeForCausalLM']})
    profile._declare_model_path(tmp_path)
    profile._declare_checkpoint_index({'weight_map': weight_map})
    with pytest.raises(RuntimeError, match='no nonempty|neither|both'):
        profile.source_tensor_name('model.layers.0.w')


def test_new_owned_index_invalidates_previous_namespace_evidence():
    profile = profile_from_config({'model_type': 'qwen3_5_moe',
                                  'architectures': ['Qwen3_5MoeForCausalLM']})
    profile._declare_checkpoint_index({'weight_map': {'model.layers.0.w': 'a.safetensors'}})
    assert profile.source_tensor_name('model.layers.0.w') == 'model.layers.0.w'
    profile._declare_checkpoint_index({'weight_map': {'model.language_model.layers.0.w': 'b.safetensors'}})
    assert profile.source_tensor_name('model.layers.0.w') == 'model.language_model.layers.0.w'
