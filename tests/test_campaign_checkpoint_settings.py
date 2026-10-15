"""Raw checkpoint settings retain the historical producer wire identity."""
from copy import deepcopy
import json
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

from prismaquant import tessera_campaign as campaign
from prismaquant.cost_stage_checkpoint import canonical_json_bytes


ORACLE = Path(__file__).parent / 'fixtures/tessera_checkpoint_settings_identity.json'


@pytest.mark.parametrize('record_index', [0, 1])
def test_raw_mapping_preserves_historical_identity_and_acquisition_proof(record_index):
    record = json.loads(ORACLE.read_text())['records'][record_index]
    settings = deepcopy(record['raw_settings'])
    original = deepcopy(settings)
    actual = campaign.campaign_checkpoint_identity(
        weights={'unit': torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32)},
        acts={'unit': None}, hessians={'unit': None}, menus={'unit': []},
        settings=MappingProxyType(settings),
        calibration_identity={'text_sha256': 'a'*64, 'fit_ids_sha256': 'b'*64},
        serving_scope=None, static_scales={}, static_scale_policy='fixture')
    assert settings == original
    # This source closure records the running implementation. Moving its
    # input API intentionally changes the seal, never the historical oracle.
    from prismaquant.production_weight_cache import _production_cache_source_sha256
    assert actual['prismaquant_source_sha256'] == _production_cache_source_sha256()
    expected = deepcopy(record['identity'])
    expected['prismaquant_source_sha256'] = actual['prismaquant_source_sha256']
    assert actual == expected
    assert canonical_json_bytes(actual['settings'], where='raw checkpoint settings') == (
        '{"custom":{"flag":false,"n":0,"value":null},"false_setting":false,'
        '"model":"fixture-é","seed":0,"unknown_none":null}').encode('utf-8')
