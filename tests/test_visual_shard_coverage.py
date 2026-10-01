"""Full declared vision scopes must not lose merger/non-block cost inputs."""

import ast
import inspect
import json
import re

import pytest

from prismaquant import incremental_measure_quant_cost, incremental_probe
from prismaquant.incremental_measure_quant_cost import _scheduled_probe_targets
from prismaquant.incremental_probe import build_extended_shard_regexes, build_shard_schedule
from prismaquant.model_profiles.registry import detect_profile
from prismaquant.model_profiles.structure import ModelStructureSpec


CASES = [
    ('glm5_next', 'Glm5NextForConditionalGeneration', 'model.visual.blocks',
     ('model.visual.merger.mlp.0', 'model.visual.deepstack_merger_list.0.mlp.0')),
    ('qwen3_5_moe', 'Qwen3_5MoeForConditionalGeneration', 'model.visual.blocks',
     ('model.visual.merger.mlp.0', 'model.visual.deepstack_merger_list.0.mlp.0')),
    ('qwen3_5', 'Qwen3_5ForConditionalGeneration', 'model.visual.blocks',
     ('model.visual.merger.mlp.0', 'model.visual.deepstack_merger_list.0.mlp.0')),
    ('qwen4_exp', 'Qwen4ExpForConditionalGeneration', 'model.visual.blocks',
     ('model.visual.merger.mlp.0', 'model.visual.deepstack_merger_list.0.mlp.0')),
    ('gemma4', 'Gemma4ForConditionalGeneration',
     'model.vision_tower.vision_model.encoder.layers',
     ('model.vision_tower.vision_model.side_projection', 'model.embed_vision.projection')),
]


def _config(tmp_path, model_type, architecture):
    (tmp_path / 'config.json').write_text(json.dumps({
        'model_type': model_type,
        'architectures': [architecture],
        'num_hidden_layers': 2,
        'vision_config': {'depth': 6, 'num_hidden_layers': 6},
    }))
    return str(tmp_path)


def _shards(builder, model_path, *, include_visual=True):
    kwargs = dict(include_body=False, include_mtp=False,
                  include_visual=include_visual, include_lm_head=False)
    if builder == 'profile':
        return detect_profile(model_path).extended_shard_regexes(model_path, 2, **kwargs)
    if builder == 'schedule':
        schedule = build_shard_schedule(
            model_path=model_path, num_body_layers=0, body_layers_per_shard=2,
            body_layer_range=(0, 0), include_mtp=False, include_visual=include_visual,
            include_lm_head=False, unified_body_sweep=False,
        )
        assert all(entry.kind == 'visual' for entry in schedule.entries)
        return [entry.linear_include for entry in schedule.entries]
    return build_extended_shard_regexes(model_path, 2, **kwargs)


@pytest.mark.parametrize('builder', ['profile', 'incremental', 'schedule'])
@pytest.mark.parametrize('model_type,architecture,block_prefix,extras', CASES)
def test_each_declared_vision_and_merger_linear_has_exactly_one_cost_shard(
    tmp_path, builder, model_type, architecture, block_prefix, extras,
):
    model_path = _config(tmp_path, model_type, architecture)
    regexes = _shards(builder, model_path)
    names = [f'{block_prefix}.{i}.attn.qkv' for i in range(6)] + list(extras)
    assert {
        name: sum(bool(re.search(regex, name)) for regex in regexes)
        for name in names
    } == {name: 1 for name in names}
    # The production cost intake must actually reach these probe rows.
    assert _scheduled_probe_targets(dict.fromkeys(names, {}), regexes) == set(names)
    # A scope is a dotted namespace, not a loose prefix or an audio/body guess.
    for unrelated in ('model.visual_extra.projection', 'model.audio_tower.projection',
                      'model.embed_audio.projection', 'model.layers.0.self_attn.q_proj'):
        assert not any(re.search(regex, unrelated) for regex in regexes)


@pytest.mark.parametrize('builder', ['profile', 'incremental', 'schedule'])
def test_explicit_text_only_cost_control_does_not_schedule_vision(tmp_path, builder):
    model_path = _config(tmp_path, 'glm5_next', 'Glm5NextForConditionalGeneration')
    assert _shards(builder, model_path, include_visual=False) == []


@pytest.mark.parametrize('builder', ['profile', 'incremental', 'schedule'])
def test_zero_visual_blocks_can_still_schedule_declared_merger(tmp_path, builder):
    model_path = _config(tmp_path, 'glm5_next', 'Glm5NextForConditionalGeneration')
    config_path = tmp_path / 'config.json'
    cfg = json.loads(config_path.read_text())
    cfg['vision_config'] = {'depth': 0}
    config_path.write_text(json.dumps(cfg))
    regexes = _shards(builder, model_path)
    assert sum(bool(re.search(regex, 'model.visual.merger.mlp.0'))
               for regex in regexes) == 1


@pytest.mark.parametrize('builder', ['profile', 'incremental', 'schedule'])
def test_absent_vision_config_does_not_invent_visual_shards(tmp_path, builder):
    model_path = _config(tmp_path, 'glm5_next', 'Glm5NextForConditionalGeneration')
    config_path = tmp_path / 'config.json'
    cfg = json.loads(config_path.read_text())
    del cfg['vision_config']
    config_path.write_text(json.dumps(cfg))
    assert _shards(builder, model_path) == []


@pytest.mark.parametrize('roots', ['model.visual', [None], [''], ['model.visual.'],
                                 ['model.*'], ['model.audio_tower']])
def test_invalid_visual_root_contract_refuses(roots):
    with pytest.raises(ValueError, match='visual_root_prefixes'):
        ModelStructureSpec.from_dict({
            'id': 'invalid-visual-roots',
            'shard_regexes': {
                'visual_layer_prefix': 'model.visual.blocks',
                'visual_root_prefixes': roots,
            },
        })


def test_undeclared_roots_preserve_legacy_block_only_scope(tmp_path, monkeypatch):
    model_path = _config(tmp_path, 'glm5_next', 'Glm5NextForConditionalGeneration')
    profile = detect_profile(model_path)
    monkeypatch.setattr(profile, 'visual_root_prefixes', lambda: ())
    cfg = json.loads((tmp_path / 'config.json').read_text())
    regexes = profile.visual_shard_regexes(cfg, 2)
    assert len(regexes) == 2
    assert not any(re.search(regex, 'model.visual.merger.mlp.0') for regex in regexes)


def test_missing_visual_block_prefix_does_not_invent_scope(tmp_path, monkeypatch):
    model_path = _config(tmp_path, 'glm5_next', 'Glm5NextForConditionalGeneration')
    profile = detect_profile(model_path)
    monkeypatch.setattr(profile, 'visual_layer_prefix', lambda: None)
    cfg = json.loads((tmp_path / 'config.json').read_text())
    assert profile.visual_shard_regexes(cfg, 2) == []


@pytest.mark.parametrize('module', [incremental_probe, incremental_measure_quant_cost])
def test_production_dispatch_uses_typed_schedule_not_regex_guess(module):
    """Structural policy guard, not a GPU execution/qualification test."""
    tree = ast.parse(inspect.getsource(module.main))
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == '_classify_shard'
        for node in ast.walk(tree)
    )
    assert any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'kind'
                for target in node.targets)
        and isinstance(node.value, ast.Attribute) and node.value.attr == 'kind'
        and isinstance(node.value.value, ast.Subscript)
        and isinstance(node.value.value.value, ast.Name)
        and node.value.value.value.id == 'schedule'
        for node in ast.walk(tree)
    )
