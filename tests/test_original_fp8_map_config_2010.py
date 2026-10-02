"""Direct original FP8 map declarations come from the owned config bytes."""
import json
from pathlib import Path

import pytest

from prismaquant import layer_streaming, model_profiles
from prismaquant.model_profiles.default import DefaultProfile
from test_capture_original_material import material, _owner  # noqa: F401
from test_original_streaming_bootstrap_2010 import _rebind
from test_strict_reader_tier_enforcement import _forget_state  # noqa: F401

pytestmark = pytest.mark.own_process


@pytest.mark.parametrize('selection', ['passed-profile', 'inferred-profile'])
def test_nonempty_original_map_uses_owned_block_when_caller_omits_config(material, monkeypatch, selection):
    m = material
    original = {'model_type': 'fixture', 'quantization_config': {'weight_block_size': [2, 4]}}
    Path(m['paths']['config.json']).write_text(json.dumps(original))
    _rebind(m, monkeypatch)
    (m['root'] / 'config.json').write_text(json.dumps({
        'model_type': 'untrusted_pool', 'quantization_config': {'weight_block_size': [128, 128]}}))

    class ScaleProfile(DefaultProfile):
        def fp8_scale_pairs(self, path, *, raw_weight_map=None):
            assert raw_weight_map == m['producer']['tensors']
            return {'proj.weight': (str(m['root'] / 'two.safetensors'), 'v')}

    selected = ScaleProfile()
    options = {'profile': selected} if selection == 'passed-profile' else {}
    if not options:
        def detect(path, *, config=None):
            assert config == original
            return selected
        monkeypatch.setattr(model_profiles, 'detect_profile', detect)
    with _owner(m) as owner:
        mapping = layer_streaming._build_fp8_scale_inv_map(str(m['root']),
            source_authentication=owner, **options)
        assert mapping.block == (2, 4), 'mutable logical config selected the dequant block'
        assert mapping['proj.weight'] == (str(m['root'] / 'two.safetensors'), 'v')
        assert mapping.mxfp4_names == frozenset()
        assert owner.material_live_bytes == 0
