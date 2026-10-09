"""Selected fresh captures retain full-draw H and publish only their explicit scope."""
import copy

import pytest
import torch

from prismaquant import tessera_calibration_cache as cc
from test_tessera_calibration_cache import capture, canonical_fields  # noqa: F401


def selected_identity(path, names):
    fields = canonical_fields()
    return cc.capture_identity(
        path, calibration={'fit_ids_sha256': 'draw'}, max_act_rows=2,
        model_load_contract=fields['model_load_contract'],
        attention_implementation='eager', unit_names=names)


def test_fresh_selected_capture_roundtrips_full_h_and_counts_without_other_units(capture, tmp_path):
    _root, path, census, _identity, acts, hessians, _record = capture
    identity = selected_identity(path, ['a'])
    root = tmp_path / 'selected-fresh'
    writer = cc.CaptureWriter(root, census_path=path, identity=identity)
    writer.write(acts={'a': acts['a']}, hessians={'a': hessians['a']},
                 counts={'a': census['counts']['a']}, maxima={'a': census['max_abs']['a']})
    record = writer.finish(model_load_contract=identity['model_load_contract'])
    manifest = cc.require_capture_contract(record['path'], expected_sha256=record['sha256'])
    assert set(manifest['entries']) == {'a'}
    assert not (root / 'inputs/b.pt').exists()
    values, _receipt = cc.prefetch_capture(record['path'], expected_identity=identity,
        census=census, names=['a'], device='cpu', expected_sha256=record['sha256'])
    assert torch.equal(values[0]['a'], acts['a'])
    assert torch.equal(values[1]['a'], hessians['a'])
    assert values[2] == {'a': 5}  # Count describes full H, not the two retained X rows.
    assert values[3] == {'a': 4.0}


@pytest.mark.parametrize('names', [[], ['unknown']])
def test_selected_capture_refuses_empty_or_unknown_scope(capture, names):
    _root, path, _census, _identity, _acts, _hessians, _record = capture
    with pytest.raises(ValueError, match='capture.*(unit|scope)'):
        selected_identity(path, names)


def test_selected_capture_still_refuses_a_missing_requested_unit(capture, tmp_path):
    _root, path, census, _identity, acts, hessians, _record = capture
    identity = selected_identity(path, ['a', 'b'])
    writer = cc.CaptureWriter(tmp_path / 'incomplete-selected', census_path=path, identity=identity)
    writer.write(acts={'a': acts['a']}, hessians={'a': hessians['a']},
                 counts={'a': census['counts']['a']}, maxima={'a': census['max_abs']['a']})
    with pytest.raises(RuntimeError):
        writer.finish(model_load_contract=identity['model_load_contract'])


def test_reducing_implicit_full_capture_scope_is_not_a_selected_capture(capture, tmp_path):
    _root, path, _census, identity, _acts, _hessians, _record = capture
    partial = copy.deepcopy(identity)
    partial['units'].pop('b')
    with pytest.raises(RuntimeError):
        cc.CaptureWriter(tmp_path / 'unannounced-partial', census_path=path, identity=partial)
