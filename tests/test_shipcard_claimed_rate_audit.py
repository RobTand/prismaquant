from __future__ import annotations

import json

import pytest

from prismaquant.shipcard import allocator_achieved_bpp, build_shipcard, open_shipcard, verify


@pytest.mark.parametrize('value', [-1.0, float('nan'), float('inf'), True])
def test_invalid_bpp_cannot_pass_build_replay(value):
    card = {'build': {'achieved_bpp': {
        'value': value,
        'source': 'layer_config.json:achieved_bits',
        'cross_check': {'verdict': 'not_applicable'},
    }}}
    problems = verify(card, required=[])
    assert any('bpp' in problem.lower() for problem in problems), problems


@pytest.mark.parametrize('value', [4.75, None])
def test_valid_or_absent_bpp_remains_accepted(value):
    assert verify({'build': {'achieved_bpp': {'value': value}}}, required=[]) == []


@pytest.mark.parametrize('value', [-1.0, 'NaN', 'Infinity', True])
def test_invalid_bpp_is_refused_at_recipe_intake(tmp_path, value):
    recipe = tmp_path / 'layer_config.json'
    recipe.write_text(json.dumps({
        'model.layers.0.mlp.up_proj': {'bits': 4},
        '__prismaquant__': {'achieved_bits': value},
    }))
    with pytest.raises(ValueError):
        allocator_achieved_bpp(recipe)


@pytest.mark.parametrize('value', [4.75, None])
def test_recipe_intake_keeps_valid_or_absent_bpp(tmp_path, value):
    recipe = tmp_path / 'layer_config.json'
    recipe.write_text(json.dumps({
        'model.layers.0.mlp.up_proj': {'bits': 4},
        '__prismaquant__': {'achieved_bits': value},
    }))
    result = allocator_achieved_bpp(recipe)
    assert result['value'] == value
    assert result['cross_check']['verdict'] == 'not_applicable'


INVALID_RATES = [0, -1.0, float('nan'), float('inf'), float('-inf'), True, False,
                 '4.75', 'NaN', 'Infinity', [], {}]


@pytest.mark.parametrize('value', INVALID_RATES)
@pytest.mark.parametrize('source', ['allocator', 'selected', 'knee'])
def test_all_rate_claim_sources_refuse_invalid_numbers(tmp_path, source, value):
    recipe = tmp_path / 'layer_config.json'
    metadata = {}
    if source == 'allocator':
        metadata['achieved_bits'] = value
    elif source == 'selected':
        metadata.update(selected_by='validated-surrogate', selected_achieved_bits=value,
                        achieved_bits=4.75)
    else:
        (tmp_path / 'pareto.knees.json').write_text(json.dumps({
            'primary': 'log_error', 'log_error': {'achieved_bits': value},
        }))
    recipe.write_text(json.dumps({
        'model.layers.0.mlp.up_proj': {'bits': 4}, '__prismaquant__': metadata,
    }))
    with pytest.raises(ValueError, match='bpp'):
        allocator_achieved_bpp(recipe)


@pytest.mark.parametrize('value', INVALID_RATES)
@pytest.mark.parametrize('constructor', [build_shipcard, open_shipcard])
def test_build_intake_refuses_invalid_rate_before_writing(tmp_path, value, constructor):
    root = tmp_path / 'artifact'
    root.mkdir()
    (root / 'config.json').write_text(json.dumps({'model_type': 'fixture'}))
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(ValueError, match='bpp'):
        constructor(root, build={'achieved_bpp': {'value': value}})
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


@pytest.mark.parametrize('value', INVALID_RATES)
def test_saved_build_refuses_invalid_rate_on_replay(tmp_path, value):
    from prismaquant.shipcard import load_shipcard

    root = tmp_path / 'artifact'
    root.mkdir()
    card = open_shipcard(root)
    card['build']['achieved_bpp'] = {'value': value, 'cross_check': {'verdict': 'not_applicable'}}
    path = tmp_path / 'replayed.json'
    path.write_text(json.dumps(card))
    assert any('bpp' in problem.lower() for problem in verify(load_shipcard(path), required=[]))


@pytest.mark.parametrize('value', [4, 4.75, None])
@pytest.mark.parametrize('source', ['allocator', 'selected', 'knee'])
def test_rate_sources_preserve_valid_or_absent_claims(tmp_path, source, value):
    recipe = tmp_path / 'layer_config.json'
    metadata = {}
    if source == 'allocator':
        metadata['achieved_bits'] = value
    elif source == 'selected':
        metadata.update(selected_by='validated-surrogate', selected_achieved_bits=value,
                        achieved_bits=-1.0)
    else:
        (tmp_path / 'pareto.knees.json').write_text(json.dumps({
            'log_error': {'achieved_bits': value},
        }))
    recipe.write_text(json.dumps({'__prismaquant__': metadata}))
    claim = allocator_achieved_bpp(recipe)
    assert claim['value'] == value
    assert claim['cross_check']['verdict'] == 'not_applicable'
