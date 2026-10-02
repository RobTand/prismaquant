"""Original-generation bootstrap must have one strict JSON interpretation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from prismaquant import tessera_calibration_cache as cc
from prismaquant.residency_map import bind_residency_manifest
from test_capture_original_material import _bound, _owner, _sha, material  # noqa: F401
from test_stage_b_prep_staged_reads_1092 import _stage_manifest
from test_strict_reader_tier_enforcement import MANIFEST, _forget_state  # noqa: F401

pytestmark = pytest.mark.own_process


@pytest.mark.parametrize('raw,message', [
    (b'{"weight_map":{},"weight_map":{"w":"one.safetensors","v":"two.safetensors"}}',
     'original bootstrap duplicate key weight_map'),
    (b'{"weight_map":{"w":"two.safetensors","w":"one.safetensors","v":"two.safetensors"}}',
     'original bootstrap duplicate key w'),
    (b'{"metadata":{"total_size":NaN},"weight_map":{"w":"one.safetensors","v":"two.safetensors"}}',
     'original bootstrap invalid constant NaN'),
])
def test_native_authenticated_index_refuses_ambiguous_or_nonfinite_json(
        material, monkeypatch, raw, message):  # noqa: F811
    m = material
    name = 'model.safetensors.index.json'
    Path(m['paths'][name]).write_bytes(raw)
    for row in m['publisher']['siblings']:
        if row['rfilename'] == name:
            row['size'] = len(raw)
            row['blobId'] = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    for row in m['readset']['entries']:
        if row['path'] == m['paths'][name]:
            row.update(bytes=len(raw), sha256=_sha(raw))
    m['readset']['total_bytes'] = sum(row['bytes'] for row in m['readset']['entries'])
    m['producer']['auxiliary_sha256'][name] = _sha(raw)
    m['options']['publisher_input'] = _bound(m['tmp'] / 'index-publisher.json', m['publisher'])
    m['options']['readset_input'] = _bound(m['tmp'] / 'index-readset.json', m['readset'])
    root = m['tmp'] / 'index-stage'
    root.mkdir()
    _stage_manifest(root, monkeypatch, m['readset'])
    bind_residency_manifest(MANIFEST)
    with pytest.raises(RuntimeError, match=message):
        with _owner(m):
            pass


def test_original_json_reader_reuses_the_strict_bootstrap_decoder(material, monkeypatch):  # noqa: F811
    with _owner(material) as owner:
        real = cc._original_bootstrap_json
        called = []

        def decode(path):
            called.append(Path(path))
            return real(path)

        monkeypatch.setattr(cc, '_original_bootstrap_json', decode)
        with owner.material_window([material['root'] / 'config.json']):
            assert owner.read_json(material['root'] / 'config.json') == {'model_type': 'fixture'}
            assert called == [Path(owner.descriptor_path(material['root'] / 'config.json'))]
