"""The existing text-only rules can operate on authenticated in-memory config."""
from __future__ import annotations

import copy
import json

from prismaquant.sensitivity_probe import text_only_stage_config


class _Profile:
    def stage_text_only_strip_keys(self):
        return ('vision_config',)

    def stage_text_only_promote_inner_model_type(self):
        return True


def test_normalization_keeps_original_metadata_and_returns_independent_derived_values():
    original = dict(model_type='wrapper', architectures=['WrapperForConditionalGeneration'],
                    vision_config={'width': 16}, num_local_experts=2,
                    hidden_size=8, text_config={'model_type': 'text', 'hidden_size': 4,
                                              'rope_parameters': {'rope_type': 'default'}})
    before = copy.deepcopy(original)
    derived = text_only_stage_config(original, profile=_Profile())
    assert original == before
    assert derived == dict(model_type='text', architectures=['WrapperForCausalLM'],
                           num_local_experts=2, num_experts=2, hidden_size=4,
                           rope_parameters={'rope_type': 'default'})
    derived['rope_parameters']['rope_type'] = 'changed'
    derived['architectures'].append('changed')
    assert original == before


def test_plain_config_needs_no_staged_derivative():
    original = dict(model_type='text', architectures=['PlainForCausalLM'], num_experts=2)
    before = json.dumps(original)
    assert text_only_stage_config(original, profile=_Profile()) is None
    assert json.dumps(original) == before


def test_unclaimed_profile_uses_the_existing_fallback_rules():
    original = dict(model_type='unclaimed', architectures=['OtherForConditionalGeneration'],
                    speech_config={'width': 16}, image_token_id=7,
                    text_config={'model_type': 'inner', 'hidden_size': 4})
    assert text_only_stage_config(original, profile=None) == dict(
        model_type='unclaimed', architectures=['OtherForCausalLM'], hidden_size=4)
    assert 'speech_config' in original and 'text_config' in original
