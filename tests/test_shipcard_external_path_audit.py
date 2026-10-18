from __future__ import annotations

import json
import struct

import pytest

from prismaquant.shipcard import fill_slot, load_shipcard, open_shipcard


def _model(tmp_path):
    root = tmp_path / 'model'
    root.mkdir()
    (root / 'config.json').write_text(json.dumps({'model_type': 'qwen3'}))
    header = json.dumps({
        'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]},
    }).encode()
    header += b' ' * (-len(header) % 8)
    (root / 'model.safetensors').write_bytes(
        struct.pack('<Q', len(header)) + header + struct.pack('<f', 1.0)
    )
    return root


@pytest.mark.parametrize('external', [False, True], ids=['canonical-control', 'external-red'])
def test_fill_uses_the_model_that_open_shipcard_attested(tmp_path, external):
    root = _model(tmp_path)
    path = tmp_path / 'records' / 'shipcard.json' if external else root / 'shipcard.json'
    card = open_shipcard(root, shipcard_path=path)
    record = {
        'slot': 'native_export.eager',
        'model_sha': card['model_sha'],
        'passed': False,
        'detail': 'Persistence fixture. No serve ran.',
    }
    fill_slot(path, 'native_export.eager', record)
    assert load_shipcard(path)['slots']['native_export.eager'] == record


def test_external_card_checks_its_artifact_when_parent_is_another_artifact(tmp_path):
    root = _model(tmp_path)
    records = tmp_path / 'records'
    records.mkdir()
    other = _model(records)
    path = other / 'shipcard.json'
    card = open_shipcard(root, shipcard_path=path)
    record = {'model_sha': card['model_sha'], 'passed': False}
    fill_slot(path, 'native_export.eager', record)
    assert load_shipcard(path)['slots']['native_export.eager'] == record


def test_historical_external_card_uses_its_bound_artifact(tmp_path):
    from prismaquant.shipcard import write_shipcard

    root = _model(tmp_path)
    path = tmp_path / 'records' / 'shipcard.json'
    card = open_shipcard(root, shipcard_path=path)
    card.pop('external_artifact', None)
    write_shipcard(path, card)
    record = {'model_sha': card['model_sha'], 'passed': False}
    fill_slot(path, 'native_export.eager', record)
    assert load_shipcard(path)['slots']['native_export.eager'] == record


@pytest.mark.parametrize('change', ['weights', 'config', 'model_dir'])
def test_external_card_refuses_changed_or_wrong_artifact(tmp_path, change):
    from prismaquant.shipcard import write_shipcard

    root = _model(tmp_path)
    path = tmp_path / 'records' / 'shipcard.json'
    card = open_shipcard(root, shipcard_path=path)
    if change == 'weights':
        weights = root / 'model.safetensors'
        raw = weights.read_bytes()
        weights.write_bytes(raw[:-4] + struct.pack('<f', 2.0))
    elif change == 'config':
        (root / 'config.json').write_text(json.dumps({'model_type': 'changed'}))
    else:
        other_parent = tmp_path / 'other'
        other_parent.mkdir()
        other = _model(other_parent)
        (other / 'config.json').write_text(json.dumps({'model_type': 'other'}))
        open_shipcard(other)
        card['model_dir'] = str(other)
        write_shipcard(path, card)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        fill_slot(path, 'native_export.eager', {'model_sha': card['model_sha'], 'passed': False})
    assert path.read_bytes() == before
    assert load_shipcard(path)['slots']['native_export.eager'] is None


def test_copied_canonical_card_checks_the_copy(tmp_path):
    import shutil

    root = _model(tmp_path)
    card = open_shipcard(root)
    copied = tmp_path / 'copied'
    shutil.copytree(root, copied)
    weights = root / 'model.safetensors'
    raw = weights.read_bytes()
    weights.write_bytes(raw[:-4] + struct.pack('<f', 2.0))
    record = {'model_sha': card['model_sha'], 'passed': False}
    fill_slot(copied / 'shipcard.json', 'native_export.eager', record)
    assert load_shipcard(copied / 'shipcard.json')['slots']['native_export.eager'] == record
